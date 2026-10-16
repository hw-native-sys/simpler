#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Default-admission scenes shared by the four onboard program backends.

The finite workload widens the observation window; only coherent mailbox identities,
native admission waiters and recorded joined-launch boundaries establish ordering.
"""

import contextlib
import copy
import ctypes
import importlib
import threading
import time
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path

import pytest
import torch
from simpler.task_interface import (
    MAILBOX_RUN_ENTRY_STRIDE,
    MAILBOX_RUN_ENTRY_TABLE_OFFSET,
    DataType,
    TaskArgs,
    TensorArgType,
    TensorTransfer,
    mailbox_run_entry_count,
)
from simpler.worker import (
    _IDLE,
    _OFF_ENTRY_ADMISSIBLE_SETS,
    _OFF_ENTRY_GENERATION,
    _TASK_ACCEPTED,
    _TASK_DONE,
    _TASK_LAUNCHED,
    _mailbox_load_i32,
    _mailbox_load_u64,
)

from simpler_setup import SceneTestCase
from simpler_setup.scene_test import build_output_prefix
from tests.st.a2a3.host_build_graph.early_enqueue import test_early_enqueue as evidence
from tests.st.a2a3.host_build_graph.worker_async_fifo.test_worker_async_fifo import (
    _SUB_ENTERED,
    _SUB_RELEASE,
    _wait_for_release,
)

_SIZE = 128 * 128
_TIMEOUT = 60.0
_SPIN = 100_000_000
_ROUNDS = 24  # Workload length, never an admission or resource-set request.


def callable_spec(arch, runtime):
    module = importlib.import_module(f"tests.st.{arch}.{runtime}.early_enqueue.test_early_enqueue")
    spec = copy.deepcopy(module._CALLABLES)
    for entry in spec["callables"]:
        for item in [entry["orchestration"], *entry["incores"]]:
            item["source"] = str((Path(module.__file__).parent / item["source"]).resolve())
    if runtime == "host_build_graph":
        spec["callables"].append(
            {
                "name": "reported_fatal",
                "orchestration": {
                    "source": str(Path(__file__).with_name("automatic_reported_fatal.cpp")),
                    "function_name": "aicpu_orchestration_entry",
                    "signature": [],
                },
                "incores": [],
            }
        )
    spec["callables"].append({"name": "hold_open", "callable": _wait_for_release})
    return spec


def until(predicate, label, timeout=_TIMEOUT):
    deadline = time.monotonic() + timeout
    cadence = threading.Event()
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        cadence.wait(0.001)
    raise AssertionError(f"timed out observing {label}")


@contextlib.contextmanager
def entries(worker, endpoint=0):
    grant = int(worker._chip_run_resource_set_counts[endpoint])
    mapping = worker._chip_shms[endpoint].buf
    assert mapping is not None
    address = ctypes.addressof(ctypes.c_char.from_buffer(mapping))
    views = []
    try:
        for index in range(mailbox_run_entry_count(grant)):
            offset = MAILBOX_RUN_ENTRY_TABLE_OFFSET + index * MAILBOX_RUN_ENTRY_STRIDE
            views.append((address + offset, mapping[offset : offset + MAILBOX_RUN_ENTRY_STRIDE]))
        yield views
    finally:
        for _, view in views:
            view.release()


def snapshot(views):
    before = [_mailbox_load_u64(a + _OFF_ENTRY_GENERATION) for a, _ in views]
    states = evidence._coherent_snapshot(views)
    capacities = [_mailbox_load_i32(a + _OFF_ENTRY_ADMISSIBLE_SETS) for a, _ in views]
    stable_states = states == evidence._coherent_snapshot(views)
    stable_capacities = capacities == [_mailbox_load_i32(a + _OFF_ENTRY_ADMISSIBLE_SETS) for a, _ in views]
    after = [_mailbox_load_u64(a + _OFF_ENTRY_GENERATION) for a, _ in views]
    if states is None or before != after or not stable_states or not stable_capacities:
        return []
    return [
        (identity, state, accepted, generation, capacity)
        for (identity, state, accepted), generation, capacity in zip(states, after, capacities)
    ]


def completed_capacity(row, run_id):
    identity, state, accepted, generation, capacity = row
    # The parent consumes DONE and releases the entry before the whole run
    # retires. Its even generation preserves this dispatch's terminal report;
    # an odd IDLE entry is newly allocated and may still carry an old capacity.
    terminal = state == _TASK_DONE or (state == _IDLE and generation > 0 and generation % 2 == 0)
    return identity[1] == run_id and accepted == _TASK_ACCEPTED and terminal and capacity > 1


class NativeRunTrace(evidence._RunTrace):
    """Join mailbox lease identities to actual native epochs, including retries."""

    def __init__(self, worker):
        super().__init__(worker)
        self.epochs = {}

    def _note_identity(self, pid, span):
        super()._note_identity(pid, span)
        attributes = evidence._attributes(span)
        key = tuple(int(attributes[name]) for name in ("dispatch_id", "slot_id", "generation"))
        epoch = int(attributes["run_epoch"])
        assert epoch > 0 and key[2] > 0, attributes
        self.epochs.setdefault(key, set()).add((pid, epoch))


class Observe:
    def __init__(self, worker, views):
        self.views = views
        self.case_runs = evidence._CaseRuns(views)
        self.trace = NativeRunTrace(worker)
        self.stop = threading.Event()
        self.ready = threading.Event()
        self.error = None
        self.windows = set()
        self.by_slot = {}
        self.accepted = set()
        self.mapping_hold = views[0][1][0:1]
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        try:
            while not self.stop.is_set():
                rows = snapshot(self.views)
                self.case_runs.note([(r[0], r[1], r[2]) for r in rows])
                own_rows = [r for r in rows if evidence._run_key(r[0]) in self.case_runs]
                active_ids = [r[0][4] for r in own_rows if r[1] == _TASK_LAUNCHED]
                done_ids = [r[0][4] for r in own_rows if r[1] == _TASK_DONE]
                if active_ids and done_ids:
                    assert max(done_ids) < min(active_ids), "endpoint completion overtook FIFO order"
                launched = []
                for identity, state, accepted, generation, _ in rows:
                    key = evidence._run_key(identity)
                    if key is None or key not in self.case_runs:
                        continue
                    if state == _TASK_LAUNCHED and accepted == _TASK_ACCEPTED:
                        assert generation % 2 == 1 and identity[3] > 0
                        full = (*key, identity[3])
                        launched.append(full)
                        self.accepted.add(full)
                        history = self.by_slot.setdefault(key[1], [])
                        if not history or history[-1] != full:
                            history.append(full)
                if launched:
                    self.windows.add(tuple(sorted(launched)))
                self.ready.set()
                self.stop.wait(0.001)
        except BaseException as error:  # noqa: BLE001 -- relayed after the reader stops
            self.error = error
            self.ready.set()

    def __enter__(self):
        self.thread.start()
        if not self.ready.wait(_TIMEOUT):
            self.__exit__()
            raise AssertionError("mailbox reader did not start")
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(_TIMEOUT)
        assert not self.thread.is_alive(), "mailbox reader did not stop; its mapping export is retained"
        self.mapping_hold.release()
        if self.error:
            raise self.error

    def assert_refilled(self):
        wide = [window for window in self.windows if len(window) >= 3]
        assert len(wide) >= 2, f"no sustained third-set use: {self.windows}"
        assert len({max(row[0] for row in window) for window in wide}) >= 2
        assert any(len(history) >= 2 for history in self.by_slot.values()), self.by_slot
        for history in self.by_slot.values():
            assert [row[0] for row in history] == sorted({row[0] for row in history}), history
            assert len({row[2] for row in history}) == len(history), history
        records = []

        def native_identities():
            records.extend(self.trace.take())
            return self.accepted <= self.trace.epochs.keys()

        until(native_identities, "native epochs for every sampled accepted lease generation")

        def joined():
            records.extend(self.trace.take())
            # A failed prepare can emit its identity before the successful retry
            # span arrives. Include newly read attempts, never freeze that set.
            accepted = {(pid, row[0], row[1], epoch) for row in self.accepted for pid, epoch in self.trace.epochs[row]}
            assert len({(pid, epoch) for pid, _, _, epoch in accepted}) == len(accepted), self.trace.epochs
            return [
                r
                for r in records
                if r.get("observed")
                and r.get("unfired")
                and (r["pid"], *r["successor"], r["successor_epoch"]) in accepted
                and (r["pid"], *r["predecessor"], r["predecessor_epoch"]) in accepted
            ]

        accepted_records = until(joined, "an accepted successor enqueued before its named predecessor completed")

        def ordered():
            records.extend(self.trace.take())
            order, overlap, same_tick, missing = evidence._whole_operator_order(
                self.trace, accepted_records, self.case_runs
            )
            assert not overlap, f"whole operators overlapped: {overlap}"
            return (order or same_tick) and not missing

        assert until(ordered, "device whole-operator order for the observed joined identities")


class AutomaticAdmissionScene(SceneTestCase):
    """One fresh Worker per scene, with the cold observation preceding every warm one."""

    def _run_and_validate_l3(self, worker, compiled_callables, sub_handles, case, **kwargs):
        del kwargs
        type(self)._st_chip_handles = compiled_callables
        type(self)._st_sub_handles = sub_handles
        self.test_run(case["platforms"][0], worker)

    def host(self, worker, value):
        buffer = worker.create_buffer(_SIZE * torch.float32.itemsize)
        tensor = torch.frombuffer(buffer.shm.buf, dtype=torch.float32, count=_SIZE)
        tensor.fill_(value)
        return buffer, tensor

    def group(self, worker, value=2.0):
        pairs = [self.host(worker, v) for v in (value, 0.5, 0.0)]
        return [p[0] for p in pairs], [p[1] for p in pairs]

    def expected(self, value):
        # The HBG graph adds 512 device operands and one host scalar; the TMR
        # chain adds 64 device operands. Every scene supplies the operand 0.5.
        return value + (256.5 if self._st_runtime == "host_build_graph" else 32.0)

    def submit(self, worker, buffers, *, endpoint=0, spin=_SPIN, hold=False, entered=None, name="vector"):
        config = self._build_config({}, output_prefix=self.prefix)
        handle = type(self)._st_chip_handles[name]

        def graph(orch, _args, _config):
            if entered:
                entered.set()
            args = TaskArgs()
            for buffer, direction in zip(
                buffers, [TensorArgType.INPUT, TensorArgType.INPUT, TensorArgType.OUTPUT_EXISTING]
            ):
                args.add_tensor(buffer.tensor((_SIZE,), DataType.FLOAT32), direction)
            if self._st_runtime == "host_build_graph" and name != "reported_fatal":
                args.add_tensor(
                    buffers[1].tensor((_SIZE,), DataType.FLOAT32), TensorArgType.INPUT, transfer=TensorTransfer.NONE
                )
            if name != "reported_fatal" or self._st_runtime == "tensormap_and_ringbuffer":
                args.add_scalar(spin)
            orch.submit_next_level(handle, args, config, worker=endpoint)
            if hold:
                orch.submit_sub(type(self)._st_sub_handles["hold_open"])

        return worker.submit(graph)

    def assert_mode(self, worker, platform):
        assert platform in ("a2a3", "a5")
        assert worker._automatic_admission_requested
        assert not worker._pipeline_depth_requested_explicitly
        assert not {"launch_depth", "pending_run_depth", "workspace_budget_bytes"}.intersection(worker._config)

    def cold(self, worker):
        _SUB_ENTERED.clear()
        _SUB_RELEASE.clear()
        buffers, tensors = self.group(worker)
        second_buffers, second_tensors = self.group(worker, 7.0)
        handles = []
        result = {}
        entered = threading.Event()
        thread = None
        try:
            first = self.submit(worker, buffers, spin=0, hold=True)
            handles.append(first)
            assert _SUB_ENTERED.wait(_TIMEOUT)
            with entries(worker) as views:
                rows = until(
                    lambda: [r for r in snapshot(views) if completed_capacity(r, first._run_id)],
                    "chip completion capacity",
                )
                assert not first.done, "chip completion incorrectly retired the whole logical run"
                assert {r[0][2] for r in rows} == {0}

                def submit_second():
                    try:
                        result["handle"] = self.submit(worker, second_buffers, spin=0, entered=entered)
                    except BaseException as error:  # noqa: BLE001 -- rethrown after join
                        result["error"] = error

                thread = threading.Thread(target=submit_second, daemon=True)
                thread.start()
                until(lambda: worker._orch._o._begin_run_waiter_count_for_test() == 1, "native cold admission waiter")
                assert not entered.is_set(), "Python graph entered before whole-run retirement"
                _SUB_RELEASE.set()
                first.wait(_TIMEOUT)
                thread.join(_TIMEOUT)
                assert not thread.is_alive()
                if "error" in result:
                    raise result["error"]
                handles.append(result["handle"])
                assert entered.is_set()
                handles[-1].wait(_TIMEOUT)
            torch.testing.assert_close(tensors[2], torch.full_like(tensors[2], self.expected(2.0)))
            torch.testing.assert_close(second_tensors[2], torch.full_like(second_tensors[2], self.expected(7.0)))
        finally:
            _SUB_RELEASE.set()
            if thread:
                thread.join(_TIMEOUT)
                assert not thread.is_alive()
            for handle in {*handles, *([result["handle"]] if "handle" in result else [])}:
                handle.wait(_TIMEOUT)

    def sequential(self, worker):
        samples = []
        with entries(worker) as views:
            for value in range(4):
                buffers, tensors = self.group(worker, float(value))
                handle = self.submit(worker, buffers, spin=0)
                run_id = handle._run_id
                handle.wait(_TIMEOUT)
                rows = [r for r in snapshot(views) if r[0][1] == run_id]
                assert rows and {r[0][2] for r in rows} == {0}, rows
                samples.append(max(r[4] for r in rows))
                torch.testing.assert_close(tensors[2], torch.full_like(tensors[2], self.expected(value)))
        assert len(set(samples)) == 1, f"sequential calls grew capacity by call count: {samples}"

    def refill(self, worker):
        handles, outputs, keepalive = [], [], []
        with entries(worker) as views, Observe(worker, views) as observer:
            try:
                for index in range(_ROUNDS):
                    buffers, tensors = self.group(worker, float(index))
                    keepalive.append((buffers, tensors))
                    handles.append(self.submit(worker, buffers))
                    outputs.append((tensors[2], self.expected(index)))
                    run_id = handles[-1]._run_id
                    until(
                        lambda: any(
                            r[0][1] == run_id and r[1] in (_TASK_LAUNCHED, _TASK_DONE) and r[2] == _TASK_ACCEPTED
                            for r in snapshot(views)
                        ),
                        "owned HOST inputs after native acceptance",
                    )
                    # submit returns after Python graph construction. Accepted LAUNCHED
                    # proves native materialization and input copy have completed.
                    tensors[0].fill_(-999)
                    tensors[1].fill_(-999)
                for handle in handles:
                    handle.wait(_TIMEOUT)
            finally:
                for handle in handles:
                    handle.wait(_TIMEOUT)
        for output, expected in outputs:
            torch.testing.assert_close(output, torch.full_like(output, expected))
        observer.assert_refilled()

    def device_chain(self, worker):
        host_input, _ = self.host(worker, 2.0)
        control, _ = self.host(worker, 0.5)
        readback, tensor = self.host(worker, 0.0)
        device = [worker.alloc_child_tensor(0, (_SIZE,), DataType.FLOAT32) for _ in range(_ROUNDS + 2)]
        handles = []
        try:
            worker.copy_to(device[0], host_input)
            operand = control
            if self._st_runtime == "tensormap_and_ringbuffer":
                worker.copy_to(device[-1], control)
                operand = device[-1]
            with entries(worker) as views, Observe(worker, views) as observer:
                try:
                    for source, target in zip(device[:_ROUNDS], device[1 : _ROUNDS + 1]):
                        handles.append(self.submit(worker, [source, operand, target]))
                    last_run = handles[-1]._run_id
                    until(
                        lambda: any(
                            r[0][1] == last_run and r[1] == _TASK_LAUNCHED and r[2] == _TASK_ACCEPTED
                            for r in snapshot(views)
                        ),
                        "last DEVICE consumer",
                    )
                    with pytest.raises(RuntimeError, match="still referenced by an in-flight"):
                        worker.free(device[_ROUNDS - 1])
                    for handle in handles:
                        handle.wait(_TIMEOUT)
                finally:
                    for handle in handles:
                        handle.wait(_TIMEOUT)
            worker.copy_from(readback, device[_ROUNDS])
            torch.testing.assert_close(tensor, torch.full_like(tensor, 2.0 + _ROUNDS * self.expected(0)))
            # A TMR run that used only DEVICE args proved no private staging bytes.
            # Reuse its slots with HOST demand and validate the new input copies.
            buffers, tensors = self.group(worker, 17.0)
            self.submit(worker, buffers, spin=0).wait(_TIMEOUT)
            torch.testing.assert_close(tensors[2], torch.full_like(tensors[2], self.expected(17.0)))
            observer.assert_refilled()
        finally:
            for handle in handles:
                handle.wait(_TIMEOUT)
            for buffer in device:
                worker.free(buffer)

    def error_and_close(self, worker):
        buffers, tensors = self.group(worker, 19.0)
        good, failure = None, None
        trace = evidence._RunTrace(worker)
        with entries(worker) as views:
            case_runs = evidence._CaseRuns(views)
            try:
                good = self.submit(worker, buffers)
                good_id = good._run_id
                until(
                    lambda: any(
                        r[0][1] == good_id and r[1] == _TASK_LAUNCHED and r[2] == _TASK_ACCEPTED
                        for r in snapshot(views)
                    ),
                    "good predecessor launched",
                )
                bad_buffers, _ = self.group(worker)
                if self._st_runtime == "host_build_graph":
                    # The HBG callback reports its failure before device launch.
                    with pytest.raises(RuntimeError):
                        failure = self.submit(worker, [], name="reported_fatal", spin=0)
                        failure.wait(_TIMEOUT)
                else:
                    failure = self.submit(worker, bad_buffers, name="reported_fatal", spin=0)
                    failure_id = failure._run_id
                    with pytest.raises(RuntimeError):
                        failure.wait(_TIMEOUT)
                good.wait(_TIMEOUT)
                torch.testing.assert_close(tensors[2], torch.full_like(tensors[2], self.expected(19.0)))
                rows = snapshot(views)
                case_runs.note([(r[0], r[1], r[2]) for r in rows])
                if self._st_runtime == "tensormap_and_ringbuffer":
                    assert failure is not None
                    identities = {r[0][1]: evidence._run_key(r[0]) for r in rows}
                    pair = (identities[good_id], identities[failure_id])
                    records = []

                    def joined_failure():
                        records.extend(trace.take())
                        return pair in evidence._established_pairs(records, case_runs)

                    assert until(joined_failure, "this failing successor joined behind its successful predecessor")
            finally:
                if good:
                    good.wait(_TIMEOUT)
                if failure:
                    with contextlib.suppress(RuntimeError):
                        failure.wait(_TIMEOUT)
        # Fatal reporting can poison the endpoint; outstanding-close has its own fresh scene.

    def test_run(self, st_platform, st_worker):
        self.assert_mode(st_worker, st_platform)
        self.prefix = str(build_output_prefix("automatic_admission"))
        try:
            self.cold(st_worker)
            self.sequential(st_worker)
            self.refill(st_worker)
            self.error_and_close(st_worker)
        finally:
            _SUB_RELEASE.set()
            st_worker.close()


class AutomaticDeviceDemandScene(AutomaticAdmissionScene):
    def test_run(self, st_platform, st_worker):
        self.assert_mode(st_worker, st_platform)
        self.prefix = str(build_output_prefix("automatic_device_demand"))
        try:
            self.device_chain(st_worker)
            self.refill(st_worker)
        finally:
            st_worker.close()


class AutomaticOutstandingCloseScene(AutomaticAdmissionScene):
    def test_run(self, st_platform, st_worker):
        self.assert_mode(st_worker, st_platform)
        self.prefix = str(build_output_prefix("automatic_outstanding_close"))
        handles = []
        with contextlib.ExitStack() as readers:
            outputs = []
            try:
                for index in range(8):
                    buffers, tensors = self.group(st_worker, float(index))
                    # This separate mapping survives the Worker's close/unlink.
                    # Drop every tensor over the owner's mapping before teardown.
                    assert buffers[2].shm is not None
                    output = SharedMemory(name=buffers[2].shm.name, create=False)
                    readers.callback(output.close)
                    outputs.append(output)
                    del tensors
                    handles.append(self.submit(st_worker, buffers))
                with entries(st_worker) as views:
                    until(
                        lambda: any(r[1] == _TASK_LAUNCHED and r[2] == _TASK_ACCEPTED for r in snapshot(views)),
                        "outstanding accepted device work before close",
                    )
                assert st_worker._accepted_run_handles
                st_worker.close()
                for handle in handles:
                    handle.wait(_TIMEOUT)
                for index, output in enumerate(outputs):
                    # Clone from the independent mapping; no tensor view outlives it.
                    actual = torch.frombuffer(output.buf, dtype=torch.float32, count=_SIZE).clone()
                    torch.testing.assert_close(actual, torch.full_like(actual, self.expected(index)))
            finally:
                st_worker.close()


class AutomaticTwoEndpointScene(AutomaticAdmissionScene):
    def test_run(self, st_platform, st_worker):
        self.assert_mode(st_worker, st_platform)
        assert len(st_worker._chip_shms) == 2
        self.prefix = str(build_output_prefix("automatic_two_endpoint_fold"))
        groups = [self.group(st_worker, value) for value in (2.0, 7.0)]
        config = self._build_config({}, output_prefix=self.prefix)
        first = None
        thread = None
        result = {}
        entered = threading.Event()

        def graph(orch, _args, _config):
            for endpoint, (buffers, _) in enumerate(groups):
                args = TaskArgs()
                for buffer, direction in zip(
                    buffers, [TensorArgType.INPUT, TensorArgType.INPUT, TensorArgType.OUTPUT_EXISTING]
                ):
                    args.add_tensor(buffer.tensor((_SIZE,), DataType.FLOAT32), direction)
                if self._st_runtime == "host_build_graph":
                    args.add_tensor(
                        buffers[1].tensor((_SIZE,), DataType.FLOAT32), TensorArgType.INPUT, transfer=TensorTransfer.NONE
                    )
                args.add_scalar(0 if endpoint == 0 else _SPIN * 2)
                orch.submit_next_level(type(self)._st_chip_handles["vector"], args, config, worker=endpoint)

        def next_graph(_orch, _args, _config):
            entered.set()

        def submit_second():
            try:
                result["handle"] = st_worker.submit(next_graph)
            except BaseException as error:  # noqa: BLE001 -- relayed after join
                result["error"] = error

        try:
            with entries(st_worker, 0) as fast, entries(st_worker, 1) as slow:
                first = st_worker.submit(graph)
                run_id = first._run_id
                thread = threading.Thread(target=submit_second, daemon=True)
                thread.start()

                def asymmetric_completion():
                    fast_before = snapshot(fast)
                    slow_rows = snapshot(slow)
                    if fast_before != snapshot(fast):
                        return False
                    return (
                        any(completed_capacity(r, run_id) for r in fast_before)
                        and any(
                            r[0][1] == run_id and r[1] == _TASK_LAUNCHED and r[2] == _TASK_ACCEPTED for r in slow_rows
                        )
                        and st_worker._orch._o._begin_run_waiter_count_for_test() == 1
                    )

                until(asymmetric_completion, "fast endpoint complete while slow endpoint holds the same logical run")
                assert not first.done
                assert not entered.is_set(), "fast endpoint widened admission before the whole-run fold"
                first.wait(_TIMEOUT)
                thread.join(_TIMEOUT)
                assert not thread.is_alive()
                if "error" in result:
                    raise result["error"]
                assert entered.is_set()
                result["handle"].wait(_TIMEOUT)
                capacities = [max(r[4] for r in snapshot(v) if r[0][1] == run_id) for v in (fast, slow)]
                assert min(capacities) > 1
                for index, (_, tensors) in enumerate(groups):
                    torch.testing.assert_close(
                        tensors[2], torch.full_like(tensors[2], self.expected((2.0, 7.0)[index]))
                    )
        finally:
            st_worker.close()
            if thread:
                thread.join(_TIMEOUT)
                assert not thread.is_alive()
