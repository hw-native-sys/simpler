#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Reject stale mailbox capacity and join native epochs to exact sampled leases."""

import contextlib
import ctypes
import struct
import threading
from types import SimpleNamespace

import pytest
import torch
from simpler import worker as worker_mod

from tests.st import automatic_admission_support as support
from tests.st.a2a3.tensormap_and_ringbuffer.early_enqueue import test_early_enqueue as stream_evidence


@pytest.mark.parametrize("state,generation", [(support._TASK_DONE, 3), (support._IDLE, 4)])
def test_completion_capacity_survives_parent_entry_release(state, generation):
    identity = (1, 7, 0, 9, 11, 0, 0, 1)
    row = (identity, state, support._TASK_ACCEPTED, generation, 2)
    assert support.completed_capacity(row, 7)
    assert not support.completed_capacity(row, 8)


@pytest.mark.parametrize(
    "state,accepted,generation,capacity",
    [
        (support._IDLE, support._TASK_ACCEPTED, 5, 2),
        (support._IDLE, 0, 6, 2),
        (support._IDLE, support._TASK_ACCEPTED, 0, 2),
        (support._TASK_LAUNCHED, support._TASK_ACCEPTED, 5, 2),
        (support._TASK_DONE, support._TASK_ACCEPTED, 5, 1),
    ],
)
def test_completion_capacity_rejects_unproven_or_reused_entries(state, accepted, generation, capacity):
    identity = (1, 7, 0, 9, 11, 0, 0, 1)
    assert not support.completed_capacity((identity, state, accepted, generation, capacity), 7)


def test_native_trace_preserves_retry_epochs_and_distinct_slot_generations(monkeypatch, tmp_path):
    monkeypatch.setattr(support.evidence, "_child_log_directory", lambda: str(tmp_path))
    trace = support.NativeRunTrace(SimpleNamespace(_chip_pids=[]))
    for dispatch, generation, epoch in [(11, 9, 101), (11, 9, 102), (12, 10, 103)]:
        span = SimpleNamespace(
            inv=epoch,
            attrs=f"dispatch_id={dispatch} slot_id=0 generation={generation} run_epoch={epoch}",
        )
        trace._note_identity(23, span)
    assert trace.epochs == {(11, 0, 9): {(23, 101), (23, 102)}, (12, 0, 10): {(23, 103)}}
    assert trace.run_of_invocation == {(23, 101): (11, 0), (23, 102): (11, 0), (23, 103): (12, 0)}
    assert (11, 0, 10) not in trace.epochs
    with pytest.raises(AssertionError):
        trace._note_identity(23, SimpleNamespace(inv=104, attrs="dispatch_id=13 slot_id=0 generation=11 run_epoch=0"))


def _controlled_admission_fault(monkeypatch, tmp_path, fault="propagated"):
    """Actual scene with synthetic owned transport/logs, never a device verdict."""
    case = object.__new__(support.AutomaticAdmissionScene)
    case._st_runtime = "host_build_graph" if fault == "hbg_prepare" else "tensormap_and_ringbuffer"
    pid, device = 4242, 2
    log = tmp_path / f"host.{pid}.log"
    log.touch()
    handles, rows, waits = [], [], []
    original_until = support.until

    def append(text):
        with log.open("a") as output:
            output.write(text + "\n")

    def span(index, name, attrs):
        return f"[STRACE] v=1 pid={pid} tid={pid} inv={100 + index} hid=abc depth=1 name={name} ts=100 dur=1 {attrs}"

    first_observer_good = fault.startswith("first_observer_good") or fault == "missing_independent_anchor"
    observer_run = 31 if first_observer_good else 33 if fault == "third_observer" else 32
    observer_slot = observer_run - 31
    error = (
        "finalize_native_run failed with code 507018 "
        f"(run_id={observer_run} slot={observer_slot} generation=5 "
        f"dispatch_id={observer_run} run_epoch={observer_run})"
    )

    class Buffer:
        def __init__(self, size):
            self.shm = SimpleNamespace(buf=memoryview(bytearray(size)))

    class Handle:
        def __init__(self, index, buffers):
            self.index, self._run_id = index, index + 31
            self.output = torch.frombuffer(buffers[2].shm.buf, dtype=torch.float32)

        def wait(self, timeout):
            assert timeout == support._TIMEOUT
            waits.append(self.index)
            if self.index == 1:
                if fault not in ("missing_native", "first_observer_good_missing_native"):
                    native_row = (
                        "Run device failure: "
                        f"device={3 if fault in ('foreign_native', 'first_observer_good_foreign_native') else device} "
                        f"runner=0xabc rc=507018 run_epoch={observer_run} generation=5 "
                        f"dispatch_id={observer_run} pipeline_slot={observer_slot} "
                        "cid=1 chip_hash=0x1 chip_dev=0x2 runtime_args=(nil)"
                    )
                    append(native_row)
                    if fault == "conflicting_native":
                        append(native_row.replace("rc=507018", "rc=507046"))
                if fault == "missing_bad_error":
                    return
                if first_observer_good or fault == "third_observer":
                    raise RuntimeError("chip run lane is poisoned: " + error)
                raise RuntimeError(error)
            if first_observer_good and fault != "missing_independent_anchor":
                raise RuntimeError(error)
            if fault in ("good_success", "wrong_good_output", "hbg_prepare"):
                self.output.fill_(case.expected(19.0))
                if fault == "wrong_good_output":
                    self.output[-1] += 1
                return
            if fault == "copied_failure":
                self.output[-1] = 1
            if fault == "arbitrary_error":
                raise RuntimeError("unrelated teardown failure")
            if fault in ("independent_observations", "missing_independent_anchor"):
                run, slot = (32, 1) if fault == "missing_independent_anchor" else (31, 0)
                append(
                    f"Run device failure: device={device} runner=0xabc rc=507018 run_epoch={run} generation=5 "
                    f"dispatch_id={run} pipeline_slot={slot} cid=1 chip_hash=0x1 chip_dev=0x2 runtime_args=(nil)"
                )
                raise RuntimeError(
                    "finalize_native_run failed with code 507018 "
                    f"(run_id={run} slot={slot} generation=5 dispatch_id={run} run_epoch={run})"
                )
            raise RuntimeError("chip run lane is poisoned: " + error)

    def submit(_worker, buffers, **kwargs):
        if fault == "hbg_prepare" and kwargs.get("name") == "reported_fatal":
            raise RuntimeError("HBG callback failed before launch")
        index = len(handles)
        handle = Handle(index, buffers)
        handles.append(handle)
        run = index + 31
        rows.append(((1, run, index, 5, run, run), support._TASK_LAUNCHED, support._TASK_ACCEPTED, 1, 2))
        if fault != "missing_root" or index == 0:
            append(
                span(index, "chip.run", f"run_id={run} slot_id={index} generation=5 dispatch_id={run} run_epoch={run}")
            )
        if index == 0 and fault == "first_observer_good_root_conflict":
            append(span(index, "chip.run", "run_id=31 slot_id=0 generation=6 dispatch_id=31 run_epoch=31"))
        if index == 1 and fault != "missing_join":
            append(
                span(index, "chip.run.joined_launch", "p_disp=31 p_slot=0 s_disp=32 s_slot=1 observed=1 unfired=1 rc=0")
            )
        if index == 1 and fault == "third_observer":
            append(span(2, "chip.run", "run_id=33 slot_id=2 generation=5 dispatch_id=33 run_epoch=33"))
        return handle

    monkeypatch.setattr(support, "entries", lambda *_: contextlib.nullcontext([]))
    monkeypatch.setattr(support, "snapshot", lambda _: list(rows))
    monkeypatch.setattr(support.evidence, "_coherent_snapshot", lambda _: [(r[0], r[1], r[2]) for r in rows])
    monkeypatch.setattr(support.evidence, "_child_log_directory", lambda: str(tmp_path))
    monkeypatch.setattr(stream_evidence, "_child_log_directory", lambda: str(tmp_path))
    monkeypatch.setattr(stream_evidence, "_EVIDENCE_BUDGET_S", 0.0)
    monkeypatch.setattr(support, "until", lambda predicate, label: original_until(predicate, label, timeout=0.1))
    monkeypatch.setattr(case, "submit", submit)
    worker = SimpleNamespace(_chip_pids=[pid], _config={"device_ids": [device]}, create_buffer=Buffer)
    return SimpleNamespace(invoke=lambda: case.error_and_close(worker), handles=handles, waits=waits)


@pytest.mark.parametrize("fault", ["propagated", "good_success", "hbg_prepare"])
def test_automatic_admission_accepts_only_the_owned_propagated_pair(monkeypatch, tmp_path, fault):
    controlled = _controlled_admission_fault(monkeypatch, tmp_path, fault)
    controlled.invoke()
    assert controlled.waits == ([0, 0] if fault == "hbg_prepare" else [1, 0, 0, 1])


@pytest.mark.parametrize("fault", ["first_observer_good", "independent_observations"])
def test_automatic_admission_accepts_each_owned_stream_observer(monkeypatch, tmp_path, fault):
    controlled = _controlled_admission_fault(monkeypatch, tmp_path, fault)
    controlled.invoke()
    assert controlled.waits == [1, 0, 0, 1]


@pytest.mark.parametrize(
    "fault,reason",
    [
        ("arbitrary_error", "unsupported or unowned stream error"),
        ("foreign_native", "unsupported or unowned stream error"),
        ("missing_native", "unsupported or unowned stream error"),
        ("missing_root", "missing or conflicting owned invocation"),
        ("conflicting_native", "unsupported or unowned stream error"),
        ("missing_independent_anchor", "no handle independently reported"),
        ("copied_failure", "failed observation copied output"),
        ("wrong_good_output", "successful call.*incorrect output"),
        ("missing_join", "timed out observing this failing successor joined"),
        ("first_observer_good_missing_native", "unsupported or unowned stream error"),
        ("first_observer_good_foreign_native", "unsupported or unowned stream error"),
        ("third_observer", "no handle independently reported"),
        ("first_observer_good_root_conflict", "missing or conflicting owned invocation"),
    ],
)
def test_automatic_admission_retains_owned_error_output_and_join_guards(monkeypatch, tmp_path, fault, reason):
    controlled = _controlled_admission_fault(monkeypatch, tmp_path, fault)
    with pytest.raises(AssertionError, match=reason):
        controlled.invoke()
    assert controlled.waits == [1, 0, 0, 1]


def test_automatic_admission_still_requires_the_intentional_fault_to_raise(monkeypatch, tmp_path):
    controlled = _controlled_admission_fault(monkeypatch, tmp_path, "missing_bad_error")
    with pytest.raises(pytest.fail.Exception, match="DID NOT RAISE"):
        controlled.invoke()
    assert controlled.waits == [1, 0, 1]


@pytest.mark.parametrize("diagnostic_failure", [False, True])
def test_refill_failure_keeps_original_error_and_owned_child_window(monkeypatch, tmp_path, capsys, diagnostic_failure):
    """Actual refill failure path; transport progress is controlled, not an SDK verdict."""
    case = object.__new__(support.AutomaticAdmissionScene)
    case._st_runtime = "tensormap_and_ringbuffer"
    pid = 4242
    log = tmp_path / f"host.{pid}.log"
    log.write_text("before this case\n")
    (tmp_path / "host.9999.log").write_text("foreign child\n")
    monkeypatch.setattr(support.evidence, "_child_log_directory", lambda: str(tmp_path))
    trace = support.NativeRunTrace(SimpleNamespace(_chip_pids=[pid]))
    log.write_text(log.read_text() + "owned launch/capacity before failure\n")
    trace.take()  # Reading records must not consume the diagnostic window's origin.
    rows, handles, waits = [], [], []
    original = RuntimeError("controlled original refill failure")

    class Handle:
        def __init__(self, index):
            self._run_id = index + 1

        def wait(self, timeout):
            assert timeout == support._TIMEOUT
            waits.append(self._run_id)
            if len(waits) == 1:
                raise original

    observer = SimpleNamespace(
        trace=trace,
        accepted=set(),
        completed_identities={((1, run), 1) for run in range(1, support._ROUNDS + 1)},
        failed_identities=set(),
        windows=set(),
        by_slot={},
        case_runs=SimpleNamespace(sorted_keys=lambda: []),
    )

    def submit(_worker, _buffers):
        handle = Handle(len(handles))
        handles.append(handle)
        rows.append(((1, handle._run_id), support._TASK_LAUNCHED, support._TASK_ACCEPTED, 1, 2))
        return handle

    monkeypatch.setattr(support, "entries", lambda *_: contextlib.nullcontext([]))
    monkeypatch.setattr(support, "Observe", lambda *_: contextlib.nullcontext(observer))
    monkeypatch.setattr(support, "snapshot", lambda _: list(rows))
    monkeypatch.setattr(case, "group", lambda *_: ([], [torch.zeros(1) for _ in range(3)]))
    monkeypatch.setattr(case, "submit", submit)
    if diagnostic_failure:

        class Unwritable:
            def write(self, _text):
                raise OSError("controlled diagnostic write failure")

        monkeypatch.setattr(support, "sys", SimpleNamespace(stderr=Unwritable()), raising=False)
    with pytest.raises(RuntimeError) as caught:
        case.refill(SimpleNamespace())
    assert caught.value is original
    assert len(handles) == support._ROUNDS
    assert waits == [1, *range(1, support._ROUNDS + 1)]
    if not diagnostic_failure:
        stderr = capsys.readouterr().err
        assert "owned launch/capacity before failure" in stderr
        assert f"--- child {pid} ---" in stderr
        assert "before this case" not in stderr
        assert "foreign child" not in stderr


@pytest.mark.parametrize(
    "completion", ["complete", "released", "withheld", "stale", "stale_entry", "foreign", "odd_idle", "failed"]
)
def test_refill_waits_for_its_actual_mailbox_window(monkeypatch, tmp_path, completion):
    """Real sampler and mailbox words; no SDK error or fake completion query."""
    case = object.__new__(support.AutomaticAdmissionScene)
    case._st_runtime = "tensormap_and_ringbuffer"
    count = support.mailbox_run_entry_count(support._ROUNDS)
    storage = bytearray(support.MAILBOX_RUN_ENTRY_TABLE_OFFSET + count * support.MAILBOX_RUN_ENTRY_STRIDE)
    mapping = memoryview(storage)
    address = ctypes.addressof(ctypes.c_char.from_buffer(mapping))
    worker = SimpleNamespace(
        _chip_pids=[],
        _chip_run_resource_set_counts=[support._ROUNDS],
        _chip_shms=[SimpleNamespace(buf=mapping)],
    )
    monkeypatch.setattr(support.evidence, "_child_log_directory", lambda: str(tmp_path))
    handles, tensors, waits = [], [], []
    first_poll = threading.Event()
    stop = threading.Event()
    producer_errors = []
    phase = {"barrier_failed": False}
    original_until = support.until
    original_error = RuntimeError("controlled original failed call")
    label = "this refill window physically completed before stream observation"

    def entry(index):
        return address + support.MAILBOX_RUN_ENTRY_TABLE_OFFSET + index * support.MAILBOX_RUN_ENTRY_STRIDE

    def publish(index, state, *, run_id=None, lease_generation=1, entry_generation=1, accepted=support._TASK_ACCEPTED):
        offset = support.MAILBOX_RUN_ENTRY_TABLE_OFFSET + index * support.MAILBOX_RUN_ENTRY_STRIDE
        run = index + 1 if run_id is None else run_id
        values = (1, run, index, lease_generation, run, index, 0, 1)
        for field, value in zip(
            (
                "PROTOCOL",
                "RUN_ID",
                "SLOT_ID",
                "LEASE_GENERATION",
                "DISPATCH_ID",
                "TASK_SLOT",
                "GROUP_INDEX",
                "GROUP_SIZE",
            ),
            values,
        ):
            struct.pack_into("=Q", storage, offset + getattr(worker_mod, "_OFF_ENTRY_" + field), value)
        worker_mod._mailbox_store_u64(entry(index) + worker_mod._OFF_ENTRY_GENERATION, entry_generation)
        worker_mod._mailbox_store_i32(entry(index) + worker_mod._OFF_ENTRY_ACCEPTED, accepted)
        worker_mod._mailbox_store_i32(entry(index) + worker_mod._OFF_ENTRY_ADMISSIBLE_SETS, 2)
        worker_mod._mailbox_store_i32(entry(index) + worker_mod._OFF_ENTRY_STATE, state)

    class Handle:
        def __init__(self, index):
            self._run_id = index + 1

        def wait(self, timeout):
            assert timeout == support._TIMEOUT
            waits.append((self._run_id, phase["barrier_failed"]))
            if not phase["barrier_failed"]:
                expected = (
                    worker_mod._TASK_FAILED
                    if completion == "failed"
                    else support._IDLE
                    if completion == "released"
                    else worker_mod._TASK_ENDED
                )
                assert worker_mod._mailbox_load_i32(entry(support._ROUNDS - 1)) == expected, (
                    "strong observation started while its queued suffix was still pending"
                )
            if completion == "failed" and self._run_id == support._ROUNDS:
                raise original_error

    def group(_worker, value):
        result = [torch.full((support._SIZE,), value), torch.full((support._SIZE,), 0.5), torch.zeros(support._SIZE)]
        tensors.append(result)
        return [], result

    def submit(_worker, _buffers):
        index = len(handles)
        handle = Handle(index)
        handles.append(handle)
        publish(index, support._TASK_LAUNCHED)
        return handle

    def until(predicate, description, **kwargs):
        if description != label:
            return original_until(predicate, description, **kwargs)
        # The independent producer cannot complete the held suffix before a real
        # barrier predicate has found it pending after all submissions/mutations.
        assert not predicate()
        first_poll.set()
        try:
            return original_until(predicate, description, timeout=0.5)
        except AssertionError:
            phase["barrier_failed"] = True
            raise

    def produce():
        try:
            while not first_poll.wait(0.01):
                if stop.is_set():
                    return
            assert len(handles) == support._ROUNDS
            assert all(torch.all(row[0] == -999) and torch.all(row[1] == -999) for row in tensors)
            for index in range(support._ROUNDS - 1):
                tensors[index][2].fill_(case.expected(index))
                publish(index, worker_mod._TASK_ENDED)
            last = support._ROUNDS - 1
            if completion in ("complete", "released"):
                tensors[last][2].fill_(case.expected(last))
                if completion == "released":
                    publish(last, support._IDLE, entry_generation=2)
                else:
                    publish(last, worker_mod._TASK_ENDED)
            elif completion == "stale":
                publish(last, worker_mod._TASK_ENDED, lease_generation=0)
            elif completion == "stale_entry":
                publish(last, worker_mod._TASK_ENDED, entry_generation=3)
            elif completion == "foreign":
                publish(last + 1, worker_mod._TASK_ENDED, run_id=999)
            elif completion == "odd_idle":
                publish(last, support._IDLE)
            elif completion == "failed":
                publish(last, worker_mod._TASK_FAILED, accepted=0)
        except BaseException as error:  # noqa: BLE001 -- returned after checked shutdown
            producer_errors.append(error)

    monkeypatch.setattr(case, "group", group)
    monkeypatch.setattr(case, "submit", submit)
    monkeypatch.setattr(support, "until", until)
    # Join/numeric tests already exercise assert_refilled; this control isolates
    # the real method's pre-observation ordering without fabricating native spans.
    monkeypatch.setattr(support.Observe, "assert_refilled", lambda _: None)
    producer = threading.Thread(target=produce, daemon=True)
    producer.start()
    try:
        if completion in ("complete", "released"):
            case.refill(worker)
            assert waits == [(run, False) for run in range(1, support._ROUNDS + 1)] * 2
        elif completion == "failed":
            with pytest.raises(RuntimeError) as caught:
                case.refill(worker)
            assert caught.value is original_error
            assert waits == [(run, False) for run in range(1, support._ROUNDS + 1)] * 2
        else:
            with pytest.raises(AssertionError, match="timed out observing " + label):
                case.refill(worker)
            assert waits == [(run, True) for run in range(1, support._ROUNDS + 1)]
        assert first_poll.is_set()
    finally:
        stop.set()
        producer.join(2)
        assert not producer.is_alive()
        mapping.release()
    assert not producer_errors
