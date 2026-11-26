# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Control collecting runs without waiting on their execution or admission fences."""

import ctypes
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait as wait_futures

from simpler.task_interface import MAILBOX_RUN_ENTRY_STRIDE, MAILBOX_RUN_ENTRY_TABLE_OFFSET
from simpler.worker import (
    _ACTIVATE,
    _NATIVE_PREPARED,
    _OFF_ENTRY_ACCEPTED,
    _OFF_ENTRY_GENERATION,
    _OFF_ENTRY_PREPARATION_DISPOSITION,
    _OFF_ENTRY_STATE,
    _TASK_ACCEPTED,
    _TASK_LAUNCHED,
    _TASK_PROTOCOL_VERSION,
    _mailbox_load_i32,
    _mailbox_load_u64,
    _read_run_entry_identity,
)

from .notify import check_identity


def _dispatch_identity(prefix, run_id, pids):
    paths = list(prefix.rglob("dispatch_identity.json"))
    if not paths:
        return None
    assert len(paths) == 1, f"run {run_id} has ambiguous dispatch sidecars: {paths}"
    identity = json.loads(paths[0].read_text())
    assert identity["run_id"] == run_id and identity["host_pid"] in pids, (
        f"sidecar does not belong to run {run_id} on children {pids}: {identity}"
    )
    assert identity["owner_instance_id"] and identity["endpoint_dispatch_id"] > 0, identity
    assert identity["pipeline_slot"] >= 0 and identity["pipeline_generation"] > 0, identity
    return identity


def matching_receipt(spans, predecessor, successor):
    """Match both native dispatches on their owning child's log, retaining their epochs."""
    assert predecessor["host_pid"] == successor["host_pid"], "joined runs changed chip child"
    assert predecessor["owner_instance_id"] == successor["owner_instance_id"], "joined runs changed owner"
    assert predecessor["run_id"] != successor["run_id"], "a run cannot join itself"
    wanted = (
        predecessor["endpoint_dispatch_id"],
        predecessor["pipeline_slot"],
        successor["endpoint_dispatch_id"],
        successor["pipeline_slot"],
    )
    matches = []
    for span in spans:
        if span.name != "chip.run.joined_launch" or span.pid != predecessor["host_pid"]:
            continue
        attrs = dict(pair.split("=", 1) for pair in span.attrs.split() if "=" in pair)
        try:
            actual = tuple(int(attrs[key]) for key in ("p_disp", "p_slot", "s_disp", "s_slot"))
        except (KeyError, ValueError):
            continue
        if actual != wanted:
            continue
        assert attrs.get("observed") == "1" and attrs.get("unfired") == "1" and attrs.get("rc") == "0", (
            f"dispatch pair {wanted} did not establish early enqueue: {attrs}"
        )
        epochs = (int(attrs["p_epoch"]), int(attrs["s_epoch"]))
        assert min(epochs) > 0 and epochs[0] != epochs[1], f"invalid joined epochs: {attrs}"
        matches.append({"predecessor": predecessor, "successor": successor, "epochs": epochs})
    assert len(matches) <= 1, f"dispatch pair {wanted} has ambiguous receipts: {matches}"
    return matches[0] if matches else None


def _notified_pair(trace, prefixes, handles, notification):
    identities = [_dispatch_identity(prefix, handle._run_id, trace._paths) for prefix, handle in zip(prefixes, handles)]
    assert all(identity is not None for identity in identities), "notification did not publish its dispatch sidecar"
    check_identity(notification, identities[-1])
    assert notification["activated"], f"native activation failed: {notification}"
    return notification, identities


def wait_receipt(trace, prefixes, handles, timeout):
    # Activation may defer for retained DFX capacity. Only the real launch
    # transition, with its receipt log flushed, qualifies this pair.
    launch, identities = _notified_pair(trace, prefixes, handles, trace.await_launch(handles[-1]._run_id, timeout))
    assert launch["launched"], f"successor was not launched: {launch}"
    receipt = matching_receipt(trace.spans(), *identities)
    assert receipt is not None, f"no exact early-enqueue receipt for submitted runs: {identities}"
    return receipt


def _pair_entries(worker, identities):
    rows = []
    for wanted in identities:
        rank = worker._chip_pids.index(wanted["host_pid"])  # noqa: SLF001 -- sidecar's actual chip child
        mapping = worker._chip_shms[rank].buf  # noqa: SLF001 -- read-only run-state evidence
        base = ctypes.addressof(ctypes.c_char.from_buffer(mapping))
        found = None
        count = (len(mapping) - MAILBOX_RUN_ENTRY_TABLE_OFFSET) // MAILBOX_RUN_ENTRY_STRIDE
        for index in range(count):
            offset = MAILBOX_RUN_ENTRY_TABLE_OFFSET + index * MAILBOX_RUN_ENTRY_STRIDE
            address = base + offset
            generation = _mailbox_load_u64(address + _OFF_ENTRY_GENERATION)
            if not generation % 2:
                continue
            with mapping[offset : offset + MAILBOX_RUN_ENTRY_STRIDE] as entry:
                identity = _read_run_entry_identity(entry)
            expected = (
                _TASK_PROTOCOL_VERSION,
                *(
                    wanted[key]
                    for key in (
                        "run_id",
                        "pipeline_slot",
                        "pipeline_generation",
                        "endpoint_dispatch_id",
                        "task_slot",
                        "group_index",
                        "group_size",
                    )
                ),
            )
            row = {
                "identity": identity,
                "state": _mailbox_load_i32(address + _OFF_ENTRY_STATE),
                "disposition": _mailbox_load_i32(address + _OFF_ENTRY_PREPARATION_DISPOSITION),
                "accepted": _mailbox_load_i32(address + _OFF_ENTRY_ACCEPTED),
                "entry_generation": generation,
            }
            if identity == expected and generation == _mailbox_load_u64(address + _OFF_ENTRY_GENERATION):
                assert found is None, f"ambiguous live entries for {wanted}"
                found = row
        rows.append(found)
    return rows


def wait_serial_fallback(worker, gate, trace, prefixes, handles, timeout):
    """Actual native activation returned with this successor prepared and unaccepted."""
    activation, identities = _notified_pair(
        trace, prefixes, handles, trace.await_activation(handles[-1]._run_id, timeout)
    )
    assert not activation["launched"], f"incompatible successor launched: {activation}"
    assert activation["disposition"] == _NATIVE_PREPARED, activation
    predecessor, successor = identities
    assert (
        predecessor["host_pid"] == successor["host_pid"]
        and predecessor["owner_instance_id"] == successor["owner_instance_id"]
    )
    gate.verify(predecessor)
    rows = _pair_entries(worker, identities)
    assert all(row is not None for row in rows), f"activated pair has no exact live entries: {identities}"
    ahead, behind = rows
    assert ahead["state"] == _TASK_LAUNCHED and ahead["accepted"] == _TASK_ACCEPTED, rows
    assert behind["state"] == _ACTIVATE and behind["disposition"] == _NATIVE_PREPARED and behind["accepted"] == 0, rows
    assert matching_receipt(trace.spans(), *identities) is None, "incompatible pair joined"
    return {"predecessor": predecessor, "successor": successor, "mailbox": rows}


class UnsettledSubmission(TimeoutError):
    """An operation still owns its callback storage; only owner-process exit may reclaim it."""


def gated_refill(gates, submit, prefixes, trace, handles, *, timeout=10.0, verify_pair=wait_receipt):
    """Only an exact pair's verified admission decision permits normal predecessor release.

    The controller owns no submission/admission lock. A blocked submit runs on
    another thread; cleanup releases every gate before a bounded wait for it.
    Each gate has one following submission. Later submissions run without a
    gate so result-entry pressure can observe the stream suffix to completion.
    Accepted handles are retained even if waiting for their submission times out.
    The hardware scene's isolated owner is the final backstop for a submitter
    that remains stuck. Its callback storage must stay live until owner exit.
    """
    assert 0 < len(gates) < len(prefixes), "each held predecessor needs a following submission"
    released = set()
    receipts = []

    def accept(index):
        handle = submit(index)
        handles.append(handle)

    submitter = ThreadPoolExecutor(max_workers=1)
    pending = None
    try:
        for index in range(len(prefixes)):
            if 0 < index <= len(gates):
                gates[index - 1].wait_entered(timeout)
            pending = submitter.submit(accept, index)
            pending.result(timeout=timeout)
            if 0 < index <= len(gates):
                receipts.append(verify_pair(trace, prefixes[index - 1 : index + 1], handles[-2:], timeout))
                gates[index - 1].verify(receipts[-1]["predecessor"])
                gates[index - 1].release()
                released.add(index - 1)
    finally:
        primary = sys.exc_info()[1]
        errors = []
        for index, gate in enumerate(gates):
            if index not in released:
                try:
                    gate.release()
                except BaseException as error:  # noqa: BLE001 -- every remaining gate must be attempted
                    errors.append(f"gate {index}: {error}")
        try:
            outstanding = wait_futures([pending], timeout=timeout).not_done if pending is not None else set()
        except BaseException as error:
            if pending is not None and not pending.done():
                raise UnsettledSubmission("submission retirement was interrupted; retain callback storage") from error
            raise
        finally:
            # __exit__ would do an unbounded join. A stuck native call cannot
            # be cancelled by Future.cancel; the caller retains its storage
            # and the scenario supervisor reclaims the whole owner process.
            submitter.shutdown(wait=False, cancel_futures=True)
        if outstanding:
            raise UnsettledSubmission(
                f"submission remains active after gate release; cleanup errors: {errors}"
            ) from primary
        if errors:
            raise RuntimeError(f"gate cleanup failed: {errors}; original failure: {primary!r}") from primary
    return receipts
