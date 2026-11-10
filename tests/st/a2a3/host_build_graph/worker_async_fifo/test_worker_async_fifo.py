#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Onboard validation for bounded whole-run FIFO admission.

The second run builds and backend-prepares in the other pipeline slot while the
first run acknowledges its device gate, but cannot launch until that gate and
the first run's SubTask fence release. A third submission blocks before its
graph callback while both slots are admitted.
"""

import atexit
import copy
import ctypes
import tempfile
import threading
import time
import uuid
from contextlib import suppress
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
)
from simpler.task_interface import ArgDirection as D
from simpler.worker import (
    _FRAME_STAGED,
    _NATIVE_PREPARED,
    _OFF_ENTRY_ACCEPTED,
    _OFF_ENTRY_GENERATION,
    _OFF_ENTRY_PREPARATION_DISPOSITION,
    _OFF_ENTRY_RUN_ID,
    _OFF_ENTRY_STATE,
    _TASK_LAUNCHED,
    _VALIDATED_ONLY,
    _mailbox_load_i32,
    _mailbox_load_u64,
)

from simpler_setup import SceneTestCase, scene_test
from tests.st.a2a3.host_build_graph.worker_async_fifo.device_gate import DeviceGate, device_gate_environment

_PIPELINED_VECTOR_ORCH = "kernels/orchestration/pipelined_vector_orch.cpp"
_TMR_KERNELS = "../../tensormap_and_ringbuffer/worker_async_fifo/kernels"


def _rebase_callable(callable_spec: dict, kernels_dir: str) -> dict:
    """Copy a CALLABLE with every ``kernels/...`` source rebased onto ``kernels_dir``."""
    out = copy.deepcopy(callable_spec)
    for entry in out["callables"]:
        if "orchestration" not in entry:
            continue
        entry["orchestration"]["source"] = entry["orchestration"]["source"].replace("kernels", kernels_dir, 1)
        for incore in entry["incores"]:
            incore["source"] = incore["source"].replace("kernels", kernels_dir, 1)
    return out


_CHAIN_LENGTH = 512


@pytest.fixture(scope="class", autouse=True)
def _device_gate_failure_budgets():
    with device_gate_environment():
        yield


_SIZE = 128 * 128

_DIR_TAGS = {
    D.IN: TensorArgType.INPUT,
    D.OUT: TensorArgType.OUTPUT_EXISTING,
    D.INOUT: TensorArgType.INOUT,
}


def _chip_args(handles, orch_signature, *scalars):
    """A ``TaskArgs`` naming each ``Buffer`` as a whole-buffer float32 view, tagged by the signature."""
    args = TaskArgs()
    for handle, direction in zip(handles, orch_signature):
        args.add_tensor(handle.tensor((_SIZE,), DataType.FLOAT32), _DIR_TAGS[direction])
    if len(orch_signature) == 4:
        args.add_tensor(
            handles[1].tensor((_SIZE,), DataType.FLOAT32), TensorArgType.INPUT, transfer=TensorTransfer.NONE
        )
    for value in scalars:
        args.add_scalar(value)
    return args


class _FileSignal:
    """Pickle-safe parent/child signal for the standalone scene runner."""

    def __init__(self, label: str):
        token = uuid.uuid4().hex
        self._path = Path(tempfile.gettempdir()) / f"simpler-worker-async-fifo-{label}-{token}"

    def clear(self) -> None:
        with suppress(FileNotFoundError):
            self._path.unlink()

    def set(self) -> None:
        self._path.touch(exist_ok=True)

    def wait(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._path.exists():
                return True
            time.sleep(0.001)
        return self._path.exists()


_SUB_ENTERED = _FileSignal("entered")
_SUB_RELEASE = _FileSignal("release")


def _clear_signals() -> None:
    _SUB_ENTERED.clear()
    _SUB_RELEASE.clear()


atexit.register(_clear_signals)


def _subtask_fence(entered, release):
    # Serialize the signal paths with the callable, including in a fresh child.
    def wait_for_release(_args):
        entered.set()
        if not release.wait(30.0):
            raise RuntimeError("whole-run FIFO test timed out waiting for the release fence")

    return wait_for_release


_wait_for_release = _subtask_fence(_SUB_ENTERED, _SUB_RELEASE)


def _wait_for_backend_prepared_successor(
    worker, predecessor, successor, timeout=10.0, worker_indices=(0,), *, disposition=_NATIVE_PREPARED
):
    entries = []
    for rank in worker_indices:
        mapping = worker._chip_shms[rank].buf  # noqa: SLF001 -- identified mailbox evidence
        assert mapping is not None
        base = ctypes.addressof(ctypes.c_char.from_buffer(mapping))
        entries.append([base + MAILBOX_RUN_ENTRY_TABLE_OFFSET + i * MAILBOX_RUN_ENTRY_STRIDE for i in range(2)])

    def read(address):
        generation = _mailbox_load_u64(address + _OFF_ENTRY_GENERATION)
        row = (
            _mailbox_load_u64(address + _OFF_ENTRY_RUN_ID),
            _mailbox_load_i32(address + _OFF_ENTRY_STATE),
            _mailbox_load_i32(address + _OFF_ENTRY_PREPARATION_DISPOSITION),
            _mailbox_load_i32(address + _OFF_ENTRY_ACCEPTED),
        )
        return row if generation % 2 == 1 and generation == _mailbox_load_u64(address + _OFF_ENTRY_GENERATION) else None

    snapshots = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snapshots = [[read(address) for address in rank_entries] for rank_entries in entries]
        if snapshots != [[read(address) for address in rank_entries] for rank_entries in entries]:
            continue
        if all(
            any(row is not None and row[0] == predecessor and row[1] == _TASK_LAUNCHED for row in rows)
            and (successor, _FRAME_STAGED, disposition, 0) in rows
            for rows in snapshots
        ):
            return
        time.sleep(0.001)
    raise AssertionError(
        f"run {successor} never published disposition {disposition} behind active run {predecessor}; "
        f"last rank snapshots (run, state, disposition, accepted): {snapshots}"
    )


@scene_test(level=3, runtime="host_build_graph")
class TestWorkerAsyncWholeRunFifo(SceneTestCase):
    """Backend preparation overlaps the predecessor, with bounded run admission."""

    CALLABLE = {
        "callables": [
            {
                "name": "vector",
                "orchestration": {
                    "source": _PIPELINED_VECTOR_ORCH,
                    "function_name": "aicpu_orchestration_entry",
                    "signature": [D.IN, D.IN, D.OUT, D.IN],
                },
                "incores": [
                    {
                        "func_id": 0,
                        "source": "kernels/aiv/gated_add.cpp",
                        "core_type": "aiv",
                        "signature": [D.IN, D.IN, D.OUT],
                    },
                    {
                        "func_id": 1,
                        "source": "kernels/aiv/kernel_add_scalar.cpp",
                        "core_type": "aiv",
                        "signature": [D.IN, D.OUT],
                    },
                ],
            },
            {"name": "wait_for_release", "callable": _wait_for_release},
        ],
    }

    CASES = [
        {
            "name": "whole_run_fifo",
            "platforms": ["a2a3"],
            "config": {"device_count": 1, "num_sub_workers": 1},
            "params": {},
        },
    ]

    @staticmethod
    def _tensor_from_host_buffer(worker, value):
        buffer = worker.create_buffer(_SIZE * torch.float32.itemsize)
        tensor = torch.frombuffer(buffer.shm.buf, dtype=torch.float32, count=_SIZE)
        tensor.fill_(value)
        return buffer, tensor

    def _run_and_validate_l3(  # noqa: PLR0913 -- mirror the scene-test runner hook
        self,
        worker,
        compiled_callables,
        sub_handles,
        case,
        rounds=1,
        skip_golden=False,
        enable_chip_swimlane=0,
        enable_dump_args=False,
        enable_pmu=0,
        enable_dep_gen=False,
        enable_scope_stats=False,
        output_prefix="",
    ):
        """Run the custom concurrency checks from the standalone scene runner.

        Unlike a regular scene case, this test has no single orchestration
        callback or generated argument set.  Pytest collects the test methods
        below directly. The standalone runner reaches them through this hook.
        """
        del (
            rounds,
            skip_golden,
            enable_chip_swimlane,
            enable_dump_args,
            enable_pmu,
            enable_dep_gen,
            enable_scope_stats,
            output_prefix,
        )
        # The pytest fixture publishes these maps on the class.  Standalone
        # execution passes the same registered handles into this hook instead.
        type(self)._st_chip_handles = compiled_callables
        type(self)._st_sub_handles = sub_handles
        platform = str(worker._config["platform"])  # noqa: SLF001 -- scene-test white-box validation
        assert platform in case["platforms"]
        self.test_a_run_whose_cleanup_touches_the_device_degrades_to_depth_one(platform, worker)
        for shared_control in (False, True):
            self.test_run(platform, worker, shared_control)

    def test_a_run_whose_cleanup_touches_the_device_degrades_to_depth_one(self, st_platform, st_worker):
        """A CommDomain release is mailbox control the whole-run FIFO cannot order.

        The FIFO orders tasks. A domain's teardown happens after the native
        fence and reaches a child through the mailbox, so N+1 allocating a
        domain while N is still releasing one can leave two collectives each
        holding a different chip's mailbox. A run that takes such a resource
        therefore degrades this worker to depth one: the successor's graph
        callback does not start at all until that teardown has run.

        The sibling test above pins the other half — a run that only dispatches
        tasks keeps the full depth, and its successor's callback does run while
        it is still fenced.
        """
        if st_platform != "a2a3":
            pytest.skip("whole-run FIFO leases require an a2a3 onboard worker")

        _SUB_ENTERED.clear()
        _SUB_RELEASE.clear()
        entered_callback = threading.Event()
        cleanup_entered = threading.Event()
        cleanup_release = threading.Event()
        handoff_waiting = threading.Event()
        result = {}
        submitter = None
        finisher = None
        first = None
        original_release = st_worker._release_domain_now  # noqa: SLF001 -- gate the actual device cleanup
        original_wait = None

        def gated_release(handle):
            cleanup_entered.set()
            assert cleanup_release.wait(30.0), "cleanup gate was never released"
            original_release(handle)

        st_worker._release_domain_now = gated_release  # noqa: SLF001
        try:
            sub_handle = type(self)._st_sub_handles["wait_for_release"]

            def first_graph(orch, _args, _cfg):
                # Control before any submit: a task travels the ready queue and
                # this travels the mailbox, so the two are not ordered.
                with orch.allocate_domain(name="degrade", workers=[0], window_size=4096, buffers=[]):
                    pass
                orch.submit_sub(sub_handle)

            first = st_worker.submit(first_graph)
            assert _SUB_ENTERED.wait(30.0), "the first run never reached its SUB fence"

            def finish_first():
                try:
                    first.wait(30.0)
                except BaseException as error:  # noqa: BLE001 -- re-raised on the test thread
                    result["first_error"] = error

            finisher = threading.Thread(target=finish_first, daemon=True)
            finisher.start()
            _SUB_RELEASE.set()
            assert cleanup_entered.wait(10.0), "the predecessor never reached device cleanup"
            original_wait = first._cv.wait  # noqa: SLF001 -- observe the real condition wait

            def observed_wait(timeout=None):
                handoff_waiting.set()
                return original_wait(timeout)

            first._cv.wait = observed_wait  # noqa: SLF001

            def second_graph(_orch, _args, _cfg):
                with first._cv:  # noqa: SLF001 -- check the cleanup boundary at callback entry
                    assert first._cleanup_published, "successor callback entered before cleanup publication"  # noqa: SLF001
                entered_callback.set()

            def submit_second():
                try:
                    result["handle"] = st_worker.submit(second_graph)
                except BaseException as error:  # noqa: BLE001 -- re-raised on the test thread
                    result["error"] = error

            submitter = threading.Thread(target=submit_second, daemon=True)
            submitter.start()

            assert handoff_waiting.wait(10.0), "the successor never entered the cleanup handoff wait"
            # Taking the same lock proves Condition.wait atomically registered
            # its waiter and released the lock, rather than merely announcing a call.
            with first._cv:  # noqa: SLF001
                assert not first._cleanup_published  # noqa: SLF001
                assert not entered_callback.is_set(), "successor entered while predecessor cleanup was held"
            cleanup_release.set()
            finisher.join(30.0)
            assert not finisher.is_alive()
            if "first_error" in result:
                raise result["first_error"]
            submitter.join(30.0)
            assert not submitter.is_alive()
            if "error" in result:
                raise result["error"]
            assert entered_callback.is_set(), "the successor stayed blocked after its predecessor's cleanup ran"
            result["handle"].wait(30.0)
        finally:
            cleanup_release.set()
            _SUB_RELEASE.set()
            if finisher is not None:
                finisher.join(30.0)
            if submitter is not None:
                submitter.join(30.0)
            st_worker._release_domain_now = original_release  # noqa: SLF001
            if original_wait is not None:
                first._cv.wait = original_wait  # noqa: SLF001
            for handle in (first, result.get("handle")):
                if handle is not None:
                    with suppress(Exception):
                        handle.wait(30.0)

    @pytest.mark.parametrize("shared_control", [False, True], ids=["independent-input", "shared-read-only-input"])
    def test_run(self, st_platform, st_worker, shared_control):
        if st_platform != "a2a3":
            pytest.skip("whole-run FIFO leases require an a2a3 onboard worker")

        _SUB_ENTERED.clear()
        _SUB_RELEASE.clear()
        third_callback = threading.Event()
        third_result = {}
        buffers = []
        tensors = []
        submitter = None
        first = None
        second = None
        gate = None
        try:
            gate = DeviceGate(st_worker)
            for value in (2.0, 3.0, 0.0, 5.0, 7.0, 0.0):
                buffer, tensor = self._tensor_from_host_buffer(st_worker, value)
                buffers.append(buffer)
                tensors.append(tensor)
            first_a, first_b, first_out, second_a, second_b, second_out = tensors
            first_bufs, second_bufs = buffers[:3], buffers[3:]
            if shared_control:
                second_bufs[1] = first_bufs[1]
                second_b = first_b
            vector_handle = type(self)._st_chip_handles["vector"]
            vector_signature = type(self)._st_chip_handles["vector_sig"]
            sub_handle = type(self)._st_sub_handles["wait_for_release"]

            def submit_vector(orch, arg_buffers, *, gate_address=0, hold_open=False):
                chip_args = _chip_args(arg_buffers, vector_signature, gate_address)
                orch.submit_next_level(vector_handle, chip_args, self._build_config(self.CASES[0]["config"]), worker=0)
                if hold_open:
                    orch.submit_sub(sub_handle)

            first = st_worker.submit(
                lambda orch, _args, _cfg: submit_vector(orch, first_bufs, gate_address=gate.address, hold_open=True)
            )
            # HBG reads the scalar from b during native host construction;
            # TMR's device orchestration uses its constant step increment.
            first_scalar = first_b[0] if self._st_runtime == "host_build_graph" else 1
            first_expected = first_a + first_b + _CHAIN_LENGTH * first_scalar
            # Run-level acceptance also includes the intentionally blocked SUB
            # task, whose compatibility endpoint acknowledges only on return.
            # Observe the chip frame directly so that fence cannot postpone the
            # successor submission until its 30-second timeout.
            gate.wait_entered()
            assert _SUB_ENTERED.wait(10.0), "the first run's SubTask did not start"
            second_graph_done = threading.Event()

            def second_graph(orch, _args, _cfg):
                submit_vector(orch, second_bufs)
                second_graph_done.set()

            second = st_worker.submit(second_graph)
            assert second_graph_done.is_set(), "the second run did not build ahead"
            _wait_for_backend_prepared_successor(st_worker, first._run_id, second._run_id)

            def third_graph(_orch, _args, _cfg):
                third_callback.set()

            def submit_third():
                try:
                    third_result["handle"] = st_worker.submit(third_graph)
                except BaseException as error:  # noqa: BLE001 -- re-raised on the test thread
                    third_result["error"] = error

            submitter = threading.Thread(target=submit_third, daemon=True)
            submitter.start()
            native = st_worker._orch._o  # noqa: SLF001 -- real admission waiter
            deadline = time.monotonic() + 10.0
            while native._begin_run_waiter_count_for_test() != 1:
                assert not third_callback.is_set(), "the third callback bypassed full admission capacity"
                assert time.monotonic() < deadline, "third submission never registered its native admission wait"
                time.sleep(0.001)
            assert not third_callback.is_set(), "third callback entered while admission was parked"

            gate.release()
            _SUB_RELEASE.set()
            first.wait(10.0)
            submitter.join(10.0)
            assert not submitter.is_alive()
            if "error" in third_result:
                raise third_result["error"]
            assert third_callback.is_set(), "the third submission did not enter after the first run freed its slot"
            second.wait(10.0)
            second_scalar = second_b[0] if self._st_runtime == "host_build_graph" else 1
            second_expected = second_a + second_b + _CHAIN_LENGTH * second_scalar
            assert torch.allclose(first_out, first_expected), "the first run did not execute correctly on the NPU"
            assert torch.allclose(second_out, second_expected), "the prepared run did not execute correctly on the NPU"
            third_result["handle"].wait(10.0)
        finally:
            if gate is not None:
                with suppress(Exception):
                    gate.release()
            _SUB_RELEASE.set()
            if submitter is not None:
                submitter.join(10.0)
            handles = [first, second, third_result.get("handle")]
            for handle in handles:
                if handle is not None:
                    with suppress(Exception):
                        handle.wait(10.0)
            tensors.clear()
            tensor = None
            first_a = first_b = first_out = second_a = second_b = second_out = None
            if all(handle is None or handle.done for handle in handles):
                for buffer in buffers:
                    buffer.close()
                if gate is not None:
                    gate.close()


@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestWorkerAsyncWholeRunFifoTmr(TestWorkerAsyncWholeRunFifo):
    """TMR uses the common FIFO with shared-arena compatibility fallback."""

    # Device orchestration names TMR's Tensor; the AIV gate is shared.
    CALLABLE = _rebase_callable(TestWorkerAsyncWholeRunFifo.CALLABLE, _TMR_KERNELS)
    CALLABLE["callables"][0]["orchestration"]["signature"] = [D.IN, D.IN, D.OUT]
    CALLABLE["callables"][0]["incores"][0]["source"] = "kernels/aiv/gated_add.cpp"

    def _run_and_validate_l3(self, worker, compiled_callables, sub_handles, case, **kwargs):
        super()._run_and_validate_l3(worker, compiled_callables, sub_handles, case, **kwargs)
        self.test_incompatible_runtime_env_falls_back_to_depth_one(worker._config["platform"], worker)

    def test_incompatible_runtime_env_falls_back_to_depth_one(self, st_platform, st_worker):
        if st_platform != "a2a3":
            pytest.skip("TMR prepared-state validation requires an a2a3 onboard worker")

        _SUB_ENTERED.clear()
        _SUB_RELEASE.clear()
        buffers = []
        tensors = []
        first = None
        second = None
        gate = None
        try:
            gate = DeviceGate(st_worker)
            for value in (2.0, 3.0, 0.0, 5.0, 7.0, 0.0):
                buffer, tensor = self._tensor_from_host_buffer(st_worker, value)
                buffers.append(buffer)
                tensors.append(tensor)
            first_a, first_b, first_out, second_a, second_b, second_out = tensors
            first_bufs, second_bufs = buffers[:3], buffers[3:]
            vector_handle = type(self)._st_chip_handles["vector"]
            vector_signature = type(self)._st_chip_handles["vector_sig"]
            sub_handle = type(self)._st_sub_handles["wait_for_release"]

            def submit_vector(orch, arg_buffers, config, *, gate_address=0, hold_open=False):
                chip_args = _chip_args(arg_buffers, vector_signature, gate_address)
                orch.submit_next_level(vector_handle, chip_args, config, worker=0)
                if hold_open:
                    orch.submit_sub(sub_handle)

            default_config = self._build_config(self.CASES[0]["config"])
            incompatible_config = self._build_config(
                {
                    **self.CASES[0]["config"],
                    "runtime_env": {"ring_heap": [128 * 1024 * 1024] * 4},
                }
            )
            first = st_worker.submit(
                lambda orch, _args, _cfg: submit_vector(
                    orch, first_bufs, default_config, gate_address=gate.address, hold_open=True
                )
            )
            gate.wait_entered()
            assert _SUB_ENTERED.wait(10.0)

            second = st_worker.submit(lambda orch, _args, _cfg: submit_vector(orch, second_bufs, incompatible_config))
            _wait_for_backend_prepared_successor(st_worker, first._run_id, second._run_id, disposition=_VALIDATED_ONLY)

            gate.release()
            _SUB_RELEASE.set()
            first.wait(30.0)
            second.wait(30.0)
            assert torch.allclose(first_out, first_a + first_b + _CHAIN_LENGTH)
            assert torch.allclose(second_out, second_a + second_b + _CHAIN_LENGTH)
        finally:
            if gate is not None:
                with suppress(Exception):
                    gate.release()
            _SUB_RELEASE.set()
            for handle in (first, second):
                if handle is not None:
                    with suppress(Exception):
                        handle.wait(30.0)
            tensors.clear()
            tensor = None
            first_a = first_b = first_out = second_a = second_b = second_out = None
            if all(handle is None or handle.done for handle in (first, second)):
                for buffer in buffers:
                    buffer.close()
                if gate is not None:
                    gate.close()


@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestWorkerAsyncWholeRunFifoTmrMultiRank(SceneTestCase):
    """Every rank of a prepared run stages before the active run retires."""

    CALLABLE = {
        "callables": [
            {
                "name": "vector",
                "orchestration": {
                    "source": f"{_TMR_KERNELS}/orchestration/single_vector_add_orch.cpp",
                    "function_name": "aicpu_orchestration_entry",
                    "signature": [D.IN, D.IN, D.OUT],
                },
                "incores": [
                    {
                        "func_id": 0,
                        "source": "kernels/aiv/gated_add.cpp",
                        "core_type": "aiv",
                        "signature": [D.IN, D.IN, D.OUT],
                    },
                ],
            },
            {"name": "wait_for_release", "callable": _wait_for_release},
        ],
    }
    CASES = [
        {
            "name": "whole_run_fifo_multi_rank",
            "platforms": ["a2a3"],
            "config": {"device_count": 2, "num_sub_workers": 1},
            "params": {},
        },
    ]

    _tensor_from_host_buffer = staticmethod(TestWorkerAsyncWholeRunFifo._tensor_from_host_buffer)

    def _run_and_validate_l3(  # noqa: PLR0913 -- mirror the scene-test runner hook
        self,
        worker,
        compiled_callables,
        sub_handles,
        case,
        rounds=1,
        skip_golden=False,
        enable_chip_swimlane=0,
        enable_dump_args=False,
        enable_pmu=0,
        enable_dep_gen=False,
        enable_scope_stats=False,
        output_prefix="",
    ):
        del (
            rounds,
            skip_golden,
            enable_chip_swimlane,
            enable_dump_args,
            enable_pmu,
            enable_dep_gen,
            enable_scope_stats,
            output_prefix,
        )
        type(self)._st_chip_handles = compiled_callables
        type(self)._st_sub_handles = sub_handles
        platform = str(worker._config["platform"])  # noqa: SLF001 -- scene-test white-box validation
        assert platform in case["platforms"]
        self._validate_all_ranks_prepare_successor_while_predecessor_active(platform, worker)

    def _validate_all_ranks_prepare_successor_while_predecessor_active(self, st_platform, st_worker):
        if st_platform != "a2a3":
            pytest.skip("multi-rank prepared-state validation requires an a2a3 onboard worker")

        _SUB_ENTERED.clear()
        _SUB_RELEASE.clear()
        worker_indices = (0, 1)
        buffers = []
        tensors = []
        first = None
        second = None
        gates = []
        try:
            for rank in worker_indices:
                gates.append(DeviceGate(st_worker, rank))
            for rank in worker_indices:
                for value in (2.0 + rank, 3.0 + rank, 0.0, 5.0 + rank, 7.0 + rank, 0.0):
                    buffer, tensor = self._tensor_from_host_buffer(st_worker, value)
                    buffers.append(buffer)
                    tensors.append(tensor)

            vector_handle = type(self)._st_chip_handles["vector"]
            vector_signature = type(self)._st_chip_handles["vector_sig"]
            sub_handle = type(self)._st_sub_handles["wait_for_release"]
            config = self._build_config(self.CASES[0]["config"])

            def submit_rank_vectors(orch, run_offset, *, hold_open=False):
                for rank in worker_indices:
                    tensor_offset = rank * 6 + run_offset
                    chip_args = _chip_args(
                        buffers[tensor_offset : tensor_offset + 3],
                        vector_signature,
                        gates[rank].address if hold_open else 0,
                    )
                    orch.submit_next_level(vector_handle, chip_args, config, worker=rank)
                if hold_open:
                    orch.submit_sub(sub_handle)

            first = st_worker.submit(lambda orch, _args, _cfg: submit_rank_vectors(orch, 0, hold_open=True))
            for gate in gates:
                gate.wait_entered()
            assert _SUB_ENTERED.wait(10.0), "the first run's SubTask did not start"

            second = st_worker.submit(lambda orch, _args, _cfg: submit_rank_vectors(orch, 3))
            _wait_for_backend_prepared_successor(
                st_worker, first._run_id, second._run_id, worker_indices=worker_indices
            )

            for gate in gates:
                gate.release()
            _SUB_RELEASE.set()
            first.wait(30.0)
            second.wait(30.0)
            for rank in worker_indices:
                offset = rank * 6
                assert torch.allclose(tensors[offset + 2], tensors[offset] + tensors[offset + 1])
                assert torch.allclose(tensors[offset + 5], tensors[offset + 3] + tensors[offset + 4])
        finally:
            for gate in gates:
                with suppress(Exception):
                    gate.release()
            _SUB_RELEASE.set()
            for handle in (first, second):
                if handle is not None:
                    with suppress(Exception):
                        handle.wait(30.0)
            tensors.clear()
            tensor = None
            if all(handle is None or handle.done for handle in (first, second)):
                for buffer in buffers:
                    buffer.close()
                for gate in gates:
                    gate.close()


if __name__ == "__main__":
    with device_gate_environment():
        SceneTestCase.run_module(__name__)
