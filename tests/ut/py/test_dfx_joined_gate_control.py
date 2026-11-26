# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exact joined receipts and release-before-join cleanup, without a device or sleeps."""

import ast
import contextlib
import json
import socket
import struct
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.st.a2a3.tensormap_and_ringbuffer.dfx_joined_launch import control


@pytest.mark.parametrize("backend_allows_join", [False, True])
def test_mixed_scene_establishes_fallback_while_predecessor_is_held(tmp_path, backend_allows_join):
    import torch  # noqa: PLC0415

    scene = (
        Path(__file__).resolve().parents[2] / "st/a2a3/tensormap_and_ringbuffer/test_dfx_joined_launch_attribution.py"
    )
    tree = ast.parse(scene.read_text())
    method = next(
        member
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "TestDfxJoinedLaunchAttribution"
        for member in node.body
        if isinstance(member, ast.FunctionDef) and member.name == "_submit_pair"
    )
    gate = SimpleNamespace(entered=False, released=False, verify=lambda identity: identity == _identity(0))
    observations, handles = [], []

    def entered(_timeout=10):
        assert handles and not handles[0].ended
        gate.entered = True

    gate.wait_entered = entered
    gate.release = lambda: setattr(gate, "released", True)
    gate.close = lambda: None

    def submit(_worker, _buffers, _config):
        handle = SimpleNamespace(_run_id=len(handles) + 1, ended=False)
        handle.wait = lambda: setattr(handle, "ended", True)
        handles.append(handle)
        return handle

    def verify(_worker, _gate, _trace, _prefixes, pair, _timeout):
        assert gate.entered and not gate.released and not pair[0].ended
        observations.append("activated-successor-kept-prepared")
        if backend_allows_join:
            raise AssertionError("incompatible successor launched")
        return {"predecessor": _identity(0), "successor": _identity(1)}

    namespace = {
        "torch": torch,
        "contextlib": contextlib,
        "Path": Path,
        "SIZE": 16,
        "DTYPE": torch.float32,
        "_PAIR_INPUT_VALUES": (5.0, 11.0),
        "gated_refill": control.gated_refill,
        "UnsettledSubmission": control.UnsettledSubmission,
        "wait_serial_fallback": verify,
        "_ChildTrace": lambda _worker: None,
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(scene), "exec"), namespace)  # noqa: S102
    case = SimpleNamespace(
        _notify=SimpleNamespace(gate=lambda _prefix: gate),
        _diagnostic_config=lambda prefix, **_kwargs: prefix,
        _submit=submit,
        _drain=lambda accepted: [handle.wait() for handle in accepted],
    )
    worker = SimpleNamespace(
        _config={"device_ids": [0]},
        create_buffer=lambda size: SimpleNamespace(shm=SimpleNamespace(buf=bytearray(size))),
    )
    if backend_allows_join:
        with pytest.raises(AssertionError, match="incompatible successor launched"):
            namespace["_submit_pair"](case, worker, tmp_path, False, disagree=True)
        assert gate.released and all(handle.ended for handle in handles)
        return
    result = namespace["_submit_pair"](case, worker, tmp_path, False, disagree=True)
    try:
        assert observations == ["activated-successor-kept-prepared"], "mixed control never established live fallback"
        assert gate.released
    finally:
        case._drain(result[0])
        result[2].close()
        for tensors, _prefix in result[1]:
            tensors.clear()


def _identity(index):
    return {
        "run_id": 11 + index,
        "host_pid": 4242,
        "owner_instance_id": "test-owner",
        "endpoint_dispatch_id": 101 + index,
        "pipeline_slot": index % 2,
        "pipeline_generation": index // 2 + 1,
        "task_slot": index,
        "group_index": 0,
        "group_size": 1,
    }


def _activation(index, *, launched=True):
    identity = _identity(index)
    keys = (
        "run_id",
        "pipeline_slot",
        "pipeline_generation",
        "endpoint_dispatch_id",
        "task_slot",
        "group_index",
        "group_size",
    )
    return {
        "pid": identity["host_pid"],
        "identity": [control._TASK_PROTOCOL_VERSION, *(identity[key] for key in keys)],
        "activated": 1,
        "launched": int(launched),
        "disposition": control._NATIVE_PREPARED,
    }


def _receipt(index, **overrides):
    predecessor, successor = _identity(index), _identity(index + 1)
    attrs = {
        "p_disp": predecessor["endpoint_dispatch_id"],
        "p_slot": predecessor["pipeline_slot"],
        "s_disp": successor["endpoint_dispatch_id"],
        "s_slot": successor["pipeline_slot"],
        "p_epoch": 501 + index,
        "s_epoch": 502 + index,
        "observed": 1,
        "unfired": 1,
        "rc": 0,
    }
    pid = overrides.pop("pid", 4242)
    attrs.update(overrides)
    return SimpleNamespace(
        name="chip.run.joined_launch", pid=pid, attrs=" ".join(f"{key}={value}" for key, value in attrs.items())
    )


@pytest.mark.parametrize(
    "wrong",
    [
        {"pid": 4243},
        {"p_disp": 99},
        {"s_disp": 99},
        {"p_slot": 1},
        {"s_slot": 0},
        {"p_disp": 103, "s_disp": 104},
    ],
)
def test_receipt_from_another_pair_cannot_release_predecessor(wrong):
    assert control.matching_receipt([_receipt(0, **wrong)], _identity(0), _identity(1)) is None
    result = control.matching_receipt([_receipt(0, **wrong), _receipt(0)], _identity(0), _identity(1))
    assert result is not None
    assert result["epochs"] == (501, 502)


@pytest.mark.parametrize("wrong", [{"observed": 0}, {"unfired": 0}, {"rc": 123}])
def test_negative_exact_receipt_is_a_failure(wrong):
    with pytest.raises(AssertionError, match="did not establish early enqueue"):
        control.matching_receipt([_receipt(0, **wrong)], _identity(0), _identity(1))


def test_two_exact_receipts_are_ambiguous():
    with pytest.raises(AssertionError, match="ambiguous receipts"):
        control.matching_receipt([_receipt(0), _receipt(0)], _identity(0), _identity(1))


@pytest.mark.parametrize("field,value", [("host_pid", 99), ("owner_instance_id", "another-owner")])
def test_pair_must_share_child_and_owner(field, value):
    successor = {**_identity(1), field: value}
    with pytest.raises(AssertionError, match="changed"):
        control.matching_receipt([_receipt(0)], _identity(0), successor)


def _model(tmp_path, count=3):
    prefixes = [tmp_path / f"run{index}" for index in range(count)]
    accepted, events, spans = [], [], []

    class Gate:
        def __init__(self, index):
            self.index = index
            self.released = threading.Event()

        def wait_entered(self, _timeout):
            assert self.index < len(accepted)
            if self.index:
                assert gates[self.index - 1].released.is_set()
            assert not self.released.is_set()
            events.append(("entered", self.index))

        def verify(self, identity):
            assert identity == _identity(self.index)

        def release(self):
            events.append(("released", self.index))
            self.released.set()

    gates = [Gate(index) for index in range(count - 1)]

    def submit(index):
        if index:
            assert not gates[index - 1].released.is_set()
        prefix = prefixes[index]
        prefix.mkdir()
        (prefix / "dispatch_identity.json").write_text(json.dumps(_identity(index)))
        handle = SimpleNamespace(_run_id=11 + index)
        accepted.append(handle)
        events.append(("accepted", index))
        if index:
            spans.append(_receipt(index - 1))
        return handle

    trace = SimpleNamespace(
        _paths={4242: None},
        spans=lambda: list(spans),
        await_activation=lambda run_id, _timeout: _activation(run_id - 11),
        await_launch=lambda run_id, _timeout: _activation(run_id - 11),
    )
    return SimpleNamespace(
        gates=gates, submit=submit, prefixes=prefixes, trace=trace, accepted=accepted, events=events, spans=spans
    )


def test_delayed_submission_keeps_predecessor_held_and_reuses_slots(tmp_path):
    model = _model(tmp_path)
    second_pending, resume_second = threading.Event(), threading.Event()
    handles = []

    def submit(index):
        if index == 1:
            second_pending.set()
            assert resume_second.wait(5), "test controller did not resume the submission"
        return model.submit(index)

    with ThreadPoolExecutor(max_workers=1) as controller:
        future = controller.submit(control.gated_refill, model.gates, submit, model.prefixes, model.trace, handles)
        try:
            assert second_pending.wait(5), "the submission did not reach its explicit pause"
            assert not model.gates[0].released.is_set()
            resume_second.set()
            receipts = future.result(timeout=5)
        finally:
            resume_second.set()
    assert [receipt["epochs"] for receipt in receipts] == [(501, 502), (502, 503)]
    assert receipts[1]["successor"]["pipeline_generation"] == 2
    assert len(handles) == 3 and all(gate.released.is_set() for gate in model.gates)
    for index in range(2):
        assert model.events.index(("accepted", index + 1)) < model.events.index(("released", index))


def test_blocked_submit_is_released_before_executor_is_joined(tmp_path, monkeypatch):
    model = _model(tmp_path, count=2)
    entered = threading.Event()
    handles = []

    def submit(index):
        if index:
            entered.set()
            assert model.gates[0].released.wait(5), "cleanup waited on the blocked submission before releasing"
            # Admission resumes during cleanup; this accepted handle still belongs to the caller.
            model.gates[0].released.clear()
            result = model.submit(index)
            model.gates[0].released.set()
            return result
        return model.submit(index)

    class Executor(ThreadPoolExecutor):
        def submit(self, function, index):
            future = super().submit(function, index)
            if index:

                def result(*, timeout):
                    assert entered.wait(timeout)
                    raise TimeoutError("injected submission deadline")

                future.result = result
                return future
            return future

    monkeypatch.setattr(control, "ThreadPoolExecutor", Executor)
    with pytest.raises(TimeoutError, match="injected submission deadline"):
        control.gated_refill(model.gates, submit, model.prefixes, model.trace, handles)
    assert len(handles) == 2
    assert all(gate.released.is_set() for gate in model.gates)
    assert model.events.index(("released", 0)) < model.events.index(("accepted", 1))


def test_receipt_failure_releases_every_gate(tmp_path):
    model = _model(tmp_path)

    def submit(index):
        result = model.submit(index)
        if index:
            model.spans[-1] = _receipt(index - 1, unfired=0)
        return result

    with pytest.raises(AssertionError, match="did not establish early enqueue"):
        control.gated_refill(model.gates, submit, model.prefixes, model.trace, [])
    assert all(gate.released.is_set() for gate in model.gates)


@pytest.mark.parametrize("gate_count", [0, 2])
def test_every_gate_needs_a_successor_before_starting_submission(gate_count):
    with pytest.raises(AssertionError, match="each held predecessor needs a following submission"):
        control.gated_refill([None] * gate_count, None, [None, None], None, [])


def test_unrelated_receipt_never_permits_release(tmp_path):
    model = _model(tmp_path, count=2)

    def submit(index):
        result = model.submit(index)
        if index:
            model.spans[-1] = _receipt(0, p_disp=999)
        return result

    with pytest.raises(AssertionError, match="no exact early-enqueue receipt"):
        control.gated_refill(model.gates, submit, model.prefixes, model.trace, [], timeout=1.0)
    assert model.events[-1:] == [("released", 0)]


def _mixed_mailbox(tmp_path, *, join_on_activation=False):
    from simpler import worker as mailbox  # noqa: PLC0415

    storage = memoryview(bytearray(control.MAILBOX_RUN_ENTRY_TABLE_OFFSET + 4 * control.MAILBOX_RUN_ENTRY_STRIDE))
    prefixes, handles, events = [], [], []
    for index in range(2):
        identity = _identity(index)
        prefix = tmp_path / f"run{index}"
        prefix.mkdir()
        (prefix / "dispatch_identity.json").write_text(json.dumps(identity))
        prefixes.append(prefix)
        handles.append(SimpleNamespace(_run_id=identity["run_id"]))
        start = control.MAILBOX_RUN_ENTRY_TABLE_OFFSET + index * control.MAILBOX_RUN_ENTRY_STRIDE
        values = {
            "GENERATION": 2 * index + 1,
            "PROTOCOL": mailbox._TASK_PROTOCOL_VERSION,
            "RUN_ID": identity["run_id"],
            "SLOT_ID": identity["pipeline_slot"],
            "LEASE_GENERATION": identity["pipeline_generation"],
            "DISPATCH_ID": identity["endpoint_dispatch_id"],
            "TASK_SLOT": identity["task_slot"],
            "GROUP_INDEX": identity["group_index"],
            "GROUP_SIZE": identity["group_size"],
        }
        for name, value in values.items():
            struct.pack_into("Q", storage, start + getattr(mailbox, f"_OFF_ENTRY_{name}"), value)
        struct.pack_into(
            "i", storage, start + mailbox._OFF_ENTRY_STATE, mailbox._TASK_LAUNCHED if index == 0 else mailbox._ACTIVATE
        )
        struct.pack_into("i", storage, start + mailbox._OFF_ENTRY_ACCEPTED, 1 if index == 0 else 0)
        struct.pack_into("i", storage, start + mailbox._OFF_ENTRY_PREPARATION_DISPOSITION, mailbox._NATIVE_PREPARED)

    def write_successor(name, value, fmt="i"):
        start = control.MAILBOX_RUN_ENTRY_TABLE_OFFSET + control.MAILBOX_RUN_ENTRY_STRIDE
        struct.pack_into(fmt, storage, start + getattr(mailbox, f"_OFF_ENTRY_{name}"), value)

    def activated(run_id, _timeout):
        assert run_id == _identity(1)["run_id"]
        events.append("activation-processed")
        if join_on_activation:
            write_successor("ACCEPTED", 1)
            write_successor("STATE", mailbox._TASK_LAUNCHED)
        return _activation(1, launched=join_on_activation)

    def verify(identity):
        assert "activation-processed" in events and identity == _identity(0)
        events.append("predecessor-still-held")

    worker = SimpleNamespace(_chip_pids=[4242], _chip_shms=[SimpleNamespace(buf=storage)])
    gate = SimpleNamespace(verify=verify)
    trace = SimpleNamespace(_paths={4242: None}, spans=lambda: [], await_activation=activated)
    return SimpleNamespace(
        worker=worker,
        gate=gate,
        trace=trace,
        prefixes=prefixes,
        handles=handles,
        events=events,
        write_successor=write_successor,
        storage=storage,
    )


def test_mixed_fallback_requires_completed_activation_checkpoint(tmp_path):
    model = _mixed_mailbox(tmp_path)
    witness = control.wait_serial_fallback(model.worker, model.gate, model.trace, model.prefixes, model.handles, 1)
    assert model.events == ["activation-processed", "predecessor-still-held"]
    assert witness["mailbox"][1]["state"] == control._ACTIVATE
    assert witness["mailbox"][1]["accepted"] == 0


def test_mixed_control_catches_allowed_join_even_before_log_arrives(tmp_path):
    model = _mixed_mailbox(tmp_path, join_on_activation=True)
    with pytest.raises(AssertionError, match="incompatible successor launched"):
        control.wait_serial_fallback(model.worker, model.gate, model.trace, model.prefixes, model.handles, 1)
    assert "activation-processed" in model.events
    assert "predecessor-still-held" not in model.events


@pytest.mark.parametrize(
    "field,value,fmt",
    [
        ("STATE", 8, "i"),  # FRAME_STAGED: activation is not authorized.
        ("PREPARATION_DISPOSITION", 1, "i"),  # VALIDATED_ONLY: backend preparation has not happened.
        ("DISPATCH_ID", 999, "Q"),
        ("LEASE_GENERATION", 99, "Q"),
        ("GENERATION", 2, "Q"),  # Reclaimed entry.
    ],
)
def test_missing_or_stale_fallback_state_is_not_negative_join_proof(tmp_path, field, value, fmt):
    model = _mixed_mailbox(tmp_path)
    model.write_successor(field, value, fmt)
    with pytest.raises(AssertionError):
        control.wait_serial_fallback(model.worker, model.gate, model.trace, model.prefixes, model.handles, 1)
    assert "activation-processed" in model.events


@pytest.mark.parametrize("field", ["pipeline_generation", "run_id", "endpoint_dispatch_id", "task_slot"])
def test_launch_from_another_dispatch_cannot_qualify_an_exact_receipt(tmp_path, field):
    model = _model(tmp_path, count=2)
    for index in range(2):
        model.submit(index)
    activation = _activation(1)
    keys = ("run_id", "pipeline_slot", "pipeline_generation", "endpoint_dispatch_id", "task_slot")
    activation["identity"][keys.index(field) + 1] += 1
    model.trace.await_launch = lambda _run_id, _timeout: activation
    with pytest.raises(AssertionError, match="different dispatch"):
        control.wait_receipt(model.trace, model.prefixes, model.accepted, 1)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux native-owner SOCK_SEQPACKET transport")
def test_reused_slot_waits_for_launch_after_retained_capacity_release(tmp_path):
    """The socket protocol can report activation before any joined launch exists."""
    from tests.st.a2a3.tensormap_and_ringbuffer.dfx_joined_launch.notify import (  # noqa: PLC0415
        NativeNotifications,
        NotificationInbox,
    )

    model = _model(tmp_path)
    handles = [model.submit(index) for index in range(2)]
    model.gates[0].release()  # Run 1 returns execution credit before its retained writer finishes.
    handles.append(model.submit(2))
    model.spans.clear()
    receiver, sender = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    capacity_released = threading.Event()
    notifications = NativeNotifications.__new__(NativeNotifications)
    notifications.inbox = NotificationInbox(receiver)
    notifications.pids = {4242}
    notifications.activations = {}
    notifications.launches = {}
    model.trace.await_activation = notifications.await_activation
    model.trace.await_launch = notifications.await_launch
    successor = {**_activation(2, launched=False), "type": "activation", "run_id": handles[-1]._run_id}

    def producer():
        sender.send(json.dumps(successor).encode())
        assert capacity_released.wait(5), "the test did not release retained capacity"
        model.spans.append(_receipt(1))
        sender.send(json.dumps({**successor, "type": "launch", "launched": 1}).encode())

    with receiver, sender, ThreadPoolExecutor(max_workers=2) as executor:
        publishing = executor.submit(producer)
        try:
            # The real socket carries activation first, without a joined receipt.
            activation = notifications.await_activation(handles[-1]._run_id, 5)
            assert not activation["launched"] and not model.spans
            waiting = executor.submit(control.wait_receipt, model.trace, model.prefixes[1:], handles[1:], 5)
            capacity_released.set()
            witness = waiting.result(timeout=5)
            publishing.result(timeout=5)
        finally:
            capacity_released.set()
            sender.shutdown(socket.SHUT_WR)
    assert witness["epochs"] == (502, 503)
    assert witness["successor"]["pipeline_generation"] == 2
