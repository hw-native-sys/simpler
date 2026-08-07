# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Unit tests for Worker (Python L3 wrapper over _Worker).

Tests use SubWorker (fork/shm) as the only worker type — no NPU device required.
Each test verifies a distinct aspect of the L3 scheduling pipeline.
"""

import _thread
import ctypes
import dis
import gc
import inspect
import multiprocessing.shared_memory as shared_memory_mod
import struct
import sys
import threading
import time
import weakref
from multiprocessing.shared_memory import SharedMemory
from types import SimpleNamespace
from typing import Any, Optional, cast
from unittest.mock import patch

import pytest
import simpler.orchestrator as orch_mod
import simpler.worker as worker_mod
from _task_interface import MAX_REGISTERED_CALLABLE_IDS  # pyright: ignore[reportMissingImports]
from simpler.callable_identity import (
    CallableHandle,
    build_chip_callable_descriptor,
    build_python_serialized_descriptor,
    compute_callable_hashid,
    hashid_to_digest,
)
from simpler.task_interface import (
    MAILBOX_ERROR_MSG_SIZE,
    MAILBOX_OFF_ERROR_MSG,
    MAILBOX_SIZE,
    ChipCallable,
    DataType,
    TaskArgs,
    TensorArgType,
    WorkerType,
    _Worker,
)
from simpler.worker import (
    _CONTROL_REQUEST,
    _CTRL_PY_REGISTER,
    _CTRL_PY_UNREGISTER,
    _CTRL_UNREGISTER,
    _IDLE,
    _OFF_STATE,
    RunHandle,
    Worker,
    _buffer_field_addr,
    _mailbox_addr,
    _mailbox_load_i32,
    _mailbox_store_i32,
    _pack_py_callable_payload,
)

from ._harness import chip_callable, fake_chip_l3, requires_sim_binaries

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_shared_counter():
    """Allocate a 4-byte shared counter accessible from forked subprocesses."""
    shm = SharedMemory(create=True, size=4)
    buf = shm.buf
    assert buf is not None
    struct.pack_into("i", buf, 0, 0)
    return shm, buf


def _read_counter(buf) -> int:
    return struct.unpack_from("i", buf, 0)[0]


def _increment_counter(buf) -> None:
    v = struct.unpack_from("i", buf, 0)[0]
    struct.pack_into("i", buf, 0, v + 1)


def _add_counter(buf, delta: int) -> None:
    v = struct.unpack_from("i", buf, 0)[0]
    struct.pack_into("i", buf, 0, v + delta)


def _set_flag(buf, offset: int, value: int) -> None:
    struct.pack_into("i", buf, offset, value)


def _get_flag(buf, offset: int) -> int:
    return struct.unpack_from("i", buf, offset)[0]


def _roundtrip_py_callable_payload(target):
    from simpler.worker import _load_py_callable_from_shm, _pack_py_callable_payload  # noqa: PLC0415

    payload = _pack_py_callable_payload(target)
    shm = SharedMemory(create=True, size=len(payload))
    try:
        assert shm.buf is not None
        shm.buf[: len(payload)] = payload
        return _load_py_callable_from_shm(shm.name)
    finally:
        shm.close()
        shm.unlink()


def _slot_for(worker: Worker, handle: CallableHandle) -> int:
    return worker._identity_registry[handle.digest].slot_id


class _FakeControlResult:
    def __init__(self, worker_type: str, worker_id: int = 0, ok: bool = True, error_message: str = ""):
        self.worker_type = worker_type
        self.worker_id = worker_id
        self.ok = ok
        self.error_message = error_message


def _chip_payload_shm(callable_obj: ChipCallable) -> SharedMemory:
    payload = ctypes.string_at(int(callable_obj.buffer_ptr()), int(callable_obj.buffer_size()))
    shm = SharedMemory(create=True, size=len(payload))
    assert shm.buf is not None
    shm.buf[: len(payload)] = payload
    return shm


def test_chip_process_loop_inits_runs_and_finalizes(monkeypatch):
    events: list[tuple] = []
    published_depths: list[int] = []
    published_frame_counts: list[int] = []

    class FakeChipWorker:
        pipeline_depth = 2

        def init(self, device_id, bins, *, log_level, prewarm_config=None, enable_sdma=False):
            events.append(("init", device_id, bins, log_level, prewarm_config, enable_sdma))

        def finalize(self) -> None:
            events.append(("finalize",))

    def fake_run_chip_main_loop(cw, *_args, chip_platform, chip_runtime, prepared=None, task_frame_count=1):
        published_depths.append(worker_mod._PIPELINE_LEASE_FMT.unpack_from(_args[0], worker_mod._OFF_PIPELINE_LEASE)[0])
        published_frame_counts.append(task_frame_count)
        events.append(("main_loop", cw, chip_platform, chip_runtime))

    monkeypatch.setattr(worker_mod, "ChipWorker", FakeChipWorker)
    monkeypatch.setattr(worker_mod, "_run_chip_main_loop", fake_run_chip_main_loop)

    shm = SharedMemory(create=True, size=MAILBOX_SIZE)
    try:
        assert shm.buf is not None
        worker_mod._chip_process_loop(
            shm.buf,
            "bins",
            7,
            {},
            {},
            {},
            platform="a2a3",
            runtime="tensormap_and_ringbuffer",
        )
    finally:
        shm.close()
        shm.unlink()

    assert events[0] == ("init", 7, "bins", 25, None, False)
    assert events[1][0] == "main_loop"
    assert events[1][2:] == ("a2a3", "tensormap_and_ringbuffer")
    assert events[2] == ("finalize",)
    assert published_depths == [2]
    assert published_frame_counts == [2]


@pytest.mark.parametrize(
    ("platform", "runtime", "depth", "expected"),
    [
        ("a2a3", "host_build_graph", 2, 2),
        ("a2a3", "host_build_graph", 1, 1),
        ("a2a3", "tensormap_and_ringbuffer", 2, 2),
        ("a5", "host_build_graph", 2, 1),
        ("a5", "tensormap_and_ringbuffer", 2, 1),
        ("a5sim", "tensormap_and_ringbuffer", 2, 1),
        ("a2a3sim", "host_build_graph", 2, 1),
    ],
)
def test_local_task_frame_count_uses_direct_a2a3_pipeline_depth(platform, runtime, depth, expected):
    assert worker_mod._local_task_frame_count(platform, runtime, depth) == expected


def test_start_hierarchical_passes_each_chip_its_negotiated_frame_count(monkeypatch):
    class FakeParentWorker:
        def __init__(self) -> None:
            self.configured_depths: list[int] = []
            self.next_level_calls: list[tuple[int, int, int]] = []
            self.initialized = False

        def configure_pipeline_depth(self, depth: int) -> None:
            self.configured_depths.append(int(depth))

        def add_next_level_worker(self, mailbox_addr: int, pid: int, task_frame_count: int) -> None:
            self.next_level_calls.append((int(mailbox_addr), int(pid), int(task_frame_count)))

        def init(self) -> None:
            self.initialized = True

        def get_orchestrator(self):
            return object()

    worker = Worker(
        level=3,
        device_ids=[0, 1],
        num_sub_workers=0,
        platform="a2a3",
        runtime="host_build_graph",
    )
    worker._chip_shms = [SharedMemory(create=True, size=MAILBOX_SIZE) for _ in range(2)]
    fake_parent = FakeParentWorker()
    worker._worker = cast(Any, fake_parent)
    worker._startup_deadline = time.monotonic() + 5.0
    fork_pids = iter((12001, 12002))

    def fake_await_children_ready(shms, _pids, kind: str, _deadline: float) -> None:
        if kind != "chip":
            return
        for shm, depth in zip(shms, (2, 1)):
            assert shm.buf is not None
            worker_mod._PIPELINE_LEASE_FMT.pack_into(shm.buf, worker_mod._OFF_PIPELINE_LEASE, depth, 0, 0)

    monkeypatch.setattr(worker_mod.os, "fork", lambda: next(fork_pids))
    monkeypatch.setattr(worker, "_await_children_ready", fake_await_children_ready)
    monkeypatch.setattr(worker_mod, "Orchestrator", lambda native, owner: (native, owner))
    try:
        worker._start_hierarchical()
    finally:
        for shm in worker._chip_shms:
            shm.close()
            shm.unlink()

    assert fake_parent.configured_depths == [1]
    assert [call[1:] for call in fake_parent.next_level_calls] == [(12001, 2), (12002, 1)]
    assert fake_parent.initialized


class _FakeNativeRunImpl:
    def __init__(self, *, supports_concurrent_native_prepare: bool = False) -> None:
        self.supports_concurrent_native_prepare = supports_concurrent_native_prepare
        self.events: list[tuple] = []
        self.completed = [threading.Event(), threading.Event()]
        self.prepared = [threading.Event(), threading.Event()]
        self.launched = [threading.Event(), threading.Event()]
        self.finalized = [threading.Event(), threading.Event()]
        self.launch_errors: dict[tuple[int, int], BaseException] = {}
        self.prepare_errors: dict[tuple[int, int], BaseException] = {}
        self.poll_errors: dict[tuple[int, int], BaseException] = {}
        self.finalize_errors: dict[tuple[int, int], BaseException] = {}
        self.prepare_identities: list[tuple[int, int, int, int]] = []
        self.register_calls: list[tuple[int, int]] = []
        self.register_called = threading.Event()
        self._finalized_runs: set[tuple[int, int]] = set()
        self._polled_slots: set[int] = set()

    def register_callable_from_blob(self, cid: int, blob_addr: int) -> None:
        self.register_calls.append((int(cid), int(blob_addr)))
        self.register_called.set()

    def _prepare_native_run_from_blob(
        self,
        _cid,
        _blob_addr,
        _capacity,
        _cfg,
        slot_id,
        generation,
        _run_id=0,
        _dispatch_id=0,
        accepted_addr=0,
        accepted_value=0,
    ):
        slot = int(slot_id)
        prepare_error = self.prepare_errors.get((slot, int(generation)))
        if prepare_error is not None:
            raise prepare_error
        self.prepare_identities.append((slot, int(generation), int(_run_id), int(_dispatch_id)))
        token = SimpleNamespace(
            slot_id=slot,
            generation=int(generation),
            run_epoch=slot + 1,
            accepted_addr=int(accepted_addr),
            accepted_value=int(accepted_value),
        )
        self.events.append(("prepare", slot))
        self.prepared[slot].set()
        return token

    def _launch_native_run(self, token) -> None:
        slot = int(token.slot_id)
        accepted_addr = int(token.accepted_addr)
        accepted_value = int(token.accepted_value)
        state_addr = int(accepted_addr) - worker_mod._OFF_ACCEPTED
        self.events.append(
            (
                "launch_enter",
                slot,
                _mailbox_load_i32(int(accepted_addr)),
                _mailbox_load_i32(state_addr),
            )
        )
        launch_error = self.launch_errors.get((slot, int(token.generation)))
        if launch_error is not None:
            raise launch_error
        _mailbox_store_i32(int(accepted_addr), int(accepted_value))
        self.events.append(("launch", slot))
        self.launched[slot].set()

    def _poll_native_run(self, token) -> bool:
        slot = int(token.slot_id)
        error = self.poll_errors.get((slot, int(token.generation)))
        if error is not None:
            raise error
        if slot not in self._polled_slots:
            self._polled_slots.add(slot)
            self.events.append(("poll", slot))
        return self.completed[slot].is_set()

    def _finalize_native_run(self, token) -> None:
        slot = int(token.slot_id)
        run_key = (slot, int(token.generation))
        assert run_key not in self._finalized_runs
        self._finalized_runs.add(run_key)
        self.events.append(("finalize", slot))
        self.finalized[slot].set()
        error = self.finalize_errors.get(run_key)
        if error is not None:
            raise error


class _FakeTwoFrameChipWorker:
    """Collaborator double handed straight to ``_run_chip_main_loop`` in-process.

    Distinct from ``_harness.FakeChipWorker``, which stands in for the class the
    forked chip child instantiates: this one is never bound as
    ``worker.ChipWorker`` and never init'd or finalized, so it implements only
    the two-frame native-run surface the loop drives.
    """

    pipeline_depth = 2

    def __init__(self, *, supports_concurrent_native_prepare: bool = False) -> None:
        self._impl = _FakeNativeRunImpl(supports_concurrent_native_prepare=supports_concurrent_native_prepare)
        self.malloc_called = threading.Event()
        self.unregister_calls: list[int] = []
        self.unregister_called = threading.Event()
        self.unregister_error: Optional[BaseException] = None

    def malloc(self, size: int) -> int:
        self._impl.events.append(("malloc", int(size)))
        self.malloc_called.set()
        return 0xCAFE

    def _unregister_slot(self, cid: int) -> None:
        self.unregister_calls.append(int(cid))
        self.unregister_called.set()
        if self.unregister_error is not None:
            raise self.unregister_error


class _TwoFrameLoopHarness:
    def __init__(
        self,
        *,
        supports_concurrent_native_prepare: bool = False,
        chip_runtime: str = "",
    ) -> None:
        self.shm = SharedMemory(create=True, size=MAILBOX_SIZE)
        self.buf = cast(memoryview, self.shm.buf)
        assert self.buf is not None
        self.mailbox_addr = _mailbox_addr(self.shm)
        self.digest = bytes([0x42]) * worker_mod.CALLABLE_HASH_DIGEST_BYTES
        self.cw = _FakeTwoFrameChipWorker(supports_concurrent_native_prepare=supports_concurrent_native_prepare)
        self.registry = {7: object()}
        self.identity_table = {self.digest: 7}
        self.identity_refs = {self.digest: 1}
        self.prepared = {7}
        self.thread = threading.Thread(
            target=worker_mod._run_chip_main_loop,
            args=(
                self.cw,
                self.buf,
                self.mailbox_addr,
                self.mailbox_addr + _OFF_STATE,
                0,
                self.registry,
                self.identity_table,
                self.identity_refs,
            ),
            kwargs={
                "chip_platform": "a2a3",
                "chip_runtime": chip_runtime,
                "prepared": self.prepared,
                "task_frame_count": 2,
            },
        )

    def _frame_offset(self, index: int) -> int:
        return (1 + index) * worker_mod.MAILBOX_FRAME_SIZE

    def state_addr(self, index: int) -> int:
        return self.mailbox_addr + self._frame_offset(index) + _OFF_STATE

    def accepted_addr(self, index: int) -> int:
        return self.mailbox_addr + self._frame_offset(index) + worker_mod._OFF_ACCEPTED

    def publish(
        self,
        index: int,
        dispatch_id: int,
        *,
        state: int = worker_mod._TASK_READY,
        generation: int = 11,
        diagnostics: bool = False,
    ) -> None:
        offset = self._frame_offset(index)
        frame = self.buf[offset : offset + worker_mod.MAILBOX_FRAME_SIZE]
        try:
            frame[worker_mod._OFF_TASK_CALLABLE_HASH : worker_mod._OFF_TASK_ARGS_BLOB] = self.digest
            struct.pack_into("=ii", frame, worker_mod._OFF_TASK_ARGS_BLOB, 0, 0)
            cfg_values = [0] * (6 + 3 * worker_mod.RUNTIME_ENV_RING_COUNT)
            cfg_values[3] = int(diagnostics)
            output_prefix = b"/tmp/simpler-test" if diagnostics else b""
            worker_mod._CFG_FMT.pack_into(frame, worker_mod._OFF_CONFIG, *cfg_values, output_prefix)
            worker_mod._PIPELINE_LEASE_FMT.pack_into(frame, worker_mod._OFF_PIPELINE_LEASE, index, 0, generation)
            struct.pack_into("=Q", frame, worker_mod._OFF_FRAME_PROTOCOL, worker_mod._TASK_PROTOCOL_VERSION)
            struct.pack_into("=Q", frame, worker_mod._OFF_FRAME_RUN_ID, 5)
            struct.pack_into("=Q", frame, worker_mod._OFF_FRAME_SLOT_ID, index)
            struct.pack_into("=Q", frame, worker_mod._OFF_FRAME_GENERATION, generation)
            struct.pack_into("=Q", frame, worker_mod._OFF_FRAME_DISPATCH_ID, dispatch_id)
        finally:
            frame.release()
        _mailbox_store_i32(self.accepted_addr(index), 0)
        _mailbox_store_i32(self.state_addr(index), state)

    def wait_state(self, index: int, expected: int) -> None:
        deadline = time.monotonic() + 5.0
        while _mailbox_load_i32(self.state_addr(index)) != expected:
            assert time.monotonic() < deadline
            time.sleep(0.001)

    def wait_control_state(self, expected: int) -> None:
        deadline = time.monotonic() + 5.0
        while _mailbox_load_i32(self.mailbox_addr + _OFF_STATE) != expected:
            assert time.monotonic() < deadline
            time.sleep(0.001)

    def assert_control_stays_pending(self, duration: float = 0.05) -> None:
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            assert _mailbox_load_i32(self.mailbox_addr + _OFF_STATE) == worker_mod._CONTROL_REQUEST
            time.sleep(0.001)

    def publish_unregister(self, digest: Optional[bytes] = None) -> None:
        digest = self.digest if digest is None else digest
        struct.pack_into("Q", self.buf, worker_mod._OFF_CALLABLE, worker_mod._CTRL_UNREGISTER)
        self.buf[worker_mod._OFF_CONTROL_CALLABLE_HASH : worker_mod._OFF_CONTROL_CALLABLE_HASH + len(digest)] = digest
        _mailbox_store_i32(self.mailbox_addr + _OFF_STATE, worker_mod._CONTROL_REQUEST)

    def publish_register(self, callable_obj: ChipCallable, payload_shm: SharedMemory, digest: bytes) -> None:
        struct.pack_into("Q", self.buf, worker_mod._OFF_CALLABLE, worker_mod._CTRL_REGISTER)
        struct.pack_into("Q", self.buf, worker_mod._CTRL_OFF_ARG0, int(callable_obj.buffer_size()))
        self.buf[worker_mod._OFF_CONTROL_CALLABLE_HASH : worker_mod._OFF_CONTROL_CALLABLE_HASH + len(digest)] = digest
        encoded_name = payload_shm.name.encode("utf-8")
        assert len(encoded_name) + 1 <= worker_mod._CTRL_SHM_NAME_BYTES
        self.buf[worker_mod._OFF_ARGS : worker_mod._OFF_ARGS + worker_mod._CTRL_SHM_NAME_BYTES] = b"\x00" * (
            worker_mod._CTRL_SHM_NAME_BYTES
        )
        self.buf[worker_mod._OFF_ARGS : worker_mod._OFF_ARGS + len(encoded_name)] = encoded_name
        _mailbox_store_i32(self.mailbox_addr + _OFF_STATE, worker_mod._CONTROL_REQUEST)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        if self.thread.is_alive():
            _mailbox_store_i32(self.mailbox_addr + _OFF_STATE, worker_mod._SHUTDOWN)
            self.thread.join(5.0)
        assert not self.thread.is_alive()

    def close(self) -> None:
        self.stop()
        self.shm.close()
        self.shm.unlink()


def test_two_frame_stages_b_without_native_prepare_until_a_finalizes():
    harness = _TwoFrameLoopHarness(chip_runtime="tensormap_and_ringbuffer")
    try:
        harness.publish(0, 1)
        harness.start()
        assert harness.cw._impl.launched[0].wait(5.0)

        harness.publish(1, 2)
        harness.wait_state(1, worker_mod._FRAME_STAGED)
        assert not harness.cw._impl.prepared[1].is_set()
        assert _mailbox_load_i32(harness.accepted_addr(1)) == 0

        harness.cw._impl.completed[0].set()
        assert harness.cw._impl.finalized[0].wait(5.0)
        assert harness.cw._impl.launched[1].wait(5.0)
        assert _mailbox_load_i32(harness.accepted_addr(1)) == worker_mod._TASK_ACCEPTED
        harness.cw._impl.completed[1].set()
        harness.wait_state(1, worker_mod._TASK_DONE)

        lifecycle = [event[:2] for event in harness.cw._impl.events if event[0] in {"prepare", "launch", "finalize"}]
        assert lifecycle == [
            ("prepare", 0),
            ("launch", 0),
            ("finalize", 0),
            ("prepare", 1),
            ("launch", 1),
            ("finalize", 1),
        ]
        launch_entries = [event for event in harness.cw._impl.events if event[0] == "launch_enter"]
        assert launch_entries == [
            ("launch_enter", 0, 0, worker_mod._FRAME_STAGED),
            ("launch_enter", 1, 0, worker_mod._FRAME_STAGED),
        ]
    finally:
        harness.close()


def test_two_frame_hbg_prepares_b_while_a_runs_but_accepts_only_after_launch():
    harness = _TwoFrameLoopHarness(
        supports_concurrent_native_prepare=True,
        chip_runtime="host_build_graph",
    )
    try:
        harness.publish(0, 1)
        harness.start()
        assert harness.cw._impl.launched[0].wait(5.0)

        harness.publish(1, 2, state=worker_mod._PREPARE_READY)
        harness.wait_state(1, worker_mod._FRAME_STAGED)
        assert harness.cw._impl.prepared[1].is_set()
        assert not harness.cw._impl.finalized[0].is_set()
        assert not harness.cw._impl.launched[1].is_set()
        assert _mailbox_load_i32(harness.accepted_addr(1)) == 0

        _mailbox_store_i32(harness.state_addr(1), worker_mod._ACTIVATE)
        assert not harness.cw._impl.launched[1].wait(0.05)
        assert _mailbox_load_i32(harness.accepted_addr(1)) == 0

        harness.cw._impl.completed[0].set()
        assert harness.cw._impl.finalized[0].wait(5.0)
        assert harness.cw._impl.launched[1].wait(5.0)
        assert _mailbox_load_i32(harness.accepted_addr(1)) == worker_mod._TASK_ACCEPTED
        harness.cw._impl.completed[1].set()
        harness.wait_state(1, worker_mod._TASK_DONE)

        lifecycle = [event[:2] for event in harness.cw._impl.events if event[0] in {"prepare", "launch", "finalize"}]
        assert lifecycle == [
            ("prepare", 0),
            ("launch", 0),
            ("prepare", 1),
            ("finalize", 0),
            ("launch", 1),
            ("finalize", 1),
        ]
    finally:
        harness.close()


def test_two_frame_hbg_publishes_failure_instead_of_staged_when_prepare_fails():
    harness = _TwoFrameLoopHarness(
        supports_concurrent_native_prepare=True,
        chip_runtime="host_build_graph",
    )
    try:
        harness.cw._impl.prepare_errors[(0, 11)] = RuntimeError("prepare failed")
        harness.publish(0, 1)
        harness.start()
        harness.wait_state(0, worker_mod._TASK_FAILED)
        assert not harness.cw._impl.prepared[0].is_set()
        assert not harness.cw._impl.launched[0].is_set()
        assert _mailbox_load_i32(harness.accepted_addr(0)) == 0
    finally:
        harness.close()


def test_two_frame_hbg_waits_for_first_token_to_launch_before_preparing_second():
    harness = _TwoFrameLoopHarness(
        supports_concurrent_native_prepare=True,
        chip_runtime="host_build_graph",
    )
    try:
        harness.publish(0, 1)
        harness.publish(1, 2, state=worker_mod._PREPARE_READY)
        harness.start()

        assert harness.cw._impl.launched[0].wait(5.0)
        harness.wait_state(1, worker_mod._FRAME_STAGED)
        assert harness.cw._impl.prepared[1].is_set()
        assert not harness.cw._impl.finalized[0].is_set()
        assert [event[:2] for event in harness.cw._impl.events if event[0] in {"prepare", "launch"}] == [
            ("prepare", 0),
            ("launch", 0),
            ("prepare", 1),
        ]

        _mailbox_store_i32(harness.state_addr(1), worker_mod._ACTIVATE)
        harness.cw._impl.completed[0].set()
        assert harness.cw._impl.launched[1].wait(5.0)
        harness.cw._impl.completed[1].set()
        harness.wait_state(1, worker_mod._TASK_DONE)
    finally:
        harness.close()


def test_two_frame_hbg_prepares_and_launches_reverse_ready_frames_by_dispatch_id():
    harness = _TwoFrameLoopHarness(
        supports_concurrent_native_prepare=True,
        chip_runtime="host_build_graph",
    )
    try:
        harness.publish(0, 2)
        harness.publish(1, 1)
        harness.start()

        deadline = time.monotonic() + 5.0
        while not any(event.is_set() for event in harness.cw._impl.launched):
            assert time.monotonic() < deadline
            time.sleep(0.001)
        assert harness.cw._impl.launched[1].is_set()
        assert not harness.cw._impl.launched[0].is_set()
        assert [event[:2] for event in harness.cw._impl.events if event[0] in {"prepare", "launch"}] == [
            ("prepare", 1),
            ("launch", 1),
            ("prepare", 0),
        ]
        assert harness.cw._impl.prepare_identities == [(1, 11, 5, 1), (0, 11, 5, 2)]

        harness.cw._impl.completed[1].set()
        assert harness.cw._impl.launched[0].wait(5.0)
        harness.cw._impl.completed[0].set()
        harness.wait_state(0, worker_mod._TASK_DONE)
    finally:
        harness.close()


def test_two_frame_hbg_does_not_prepare_high_dispatch_successor_before_active_frame():
    harness = _TwoFrameLoopHarness(
        supports_concurrent_native_prepare=True,
        chip_runtime="host_build_graph",
    )
    try:
        harness.publish(0, 2, state=worker_mod._PREPARE_READY)
        harness.publish(1, 1)
        harness.start()

        assert harness.cw._impl.launched[1].wait(5.0)
        harness.wait_state(0, worker_mod._FRAME_STAGED)
        assert harness.cw._impl.prepared[0].is_set()
        assert not harness.cw._impl.launched[0].is_set()
        assert [event[:2] for event in harness.cw._impl.events if event[0] in {"prepare", "launch"}] == [
            ("prepare", 1),
            ("launch", 1),
            ("prepare", 0),
        ]

        _mailbox_store_i32(harness.state_addr(0), worker_mod._ACTIVATE)
        harness.cw._impl.completed[1].set()
        assert harness.cw._impl.launched[0].wait(5.0)
        harness.cw._impl.completed[0].set()
        harness.wait_state(0, worker_mod._TASK_DONE)
    finally:
        harness.close()


@pytest.mark.parametrize("diagnostic_frame", ["active", "successor"])
def test_two_frame_hbg_defers_diagnostic_native_prepare_until_predecessor_finalizes(diagnostic_frame):
    harness = _TwoFrameLoopHarness(
        supports_concurrent_native_prepare=True,
        chip_runtime="host_build_graph",
    )
    try:
        harness.publish(0, 1, diagnostics=diagnostic_frame == "active")
        harness.start()
        assert harness.cw._impl.launched[0].wait(5.0)

        harness.publish(
            1,
            2,
            state=worker_mod._PREPARE_READY,
            diagnostics=diagnostic_frame == "successor",
        )
        harness.wait_state(1, worker_mod._FRAME_STAGED)
        assert not harness.cw._impl.prepared[1].is_set()

        _mailbox_store_i32(harness.state_addr(1), worker_mod._ACTIVATE)
        harness.cw._impl.completed[0].set()
        assert harness.cw._impl.finalized[0].wait(5.0)
        assert harness.cw._impl.launched[1].wait(5.0)
        harness.cw._impl.completed[1].set()
        harness.wait_state(1, worker_mod._TASK_DONE)

        lifecycle = [event[:2] for event in harness.cw._impl.events if event[0] in {"prepare", "launch", "finalize"}]
        assert lifecycle == [
            ("prepare", 0),
            ("launch", 0),
            ("finalize", 0),
            ("prepare", 1),
            ("launch", 1),
            ("finalize", 1),
        ]
    finally:
        harness.close()


def test_two_frame_launches_by_dispatch_id_when_frames_are_ready_in_reverse_order():
    harness = _TwoFrameLoopHarness()
    try:
        harness.publish(0, 2)
        harness.publish(1, 1)
        harness.start()
        assert harness.cw._impl.launched[1].wait(5.0)
        assert not harness.cw._impl.prepared[0].is_set()
        harness.cw._impl.completed[1].set()
        assert harness.cw._impl.launched[0].wait(5.0)
        harness.cw._impl.completed[0].set()
        harness.wait_state(0, worker_mod._TASK_DONE)
        launches = [event for event in harness.cw._impl.events if event[0] == "launch"]
        assert launches == [("launch", 1), ("launch", 0)]
    finally:
        harness.close()


def test_two_frame_prepare_ready_waits_for_sticky_activation():
    harness = _TwoFrameLoopHarness()
    try:
        harness.publish(0, 1, state=worker_mod._PREPARE_READY)
        harness.start()
        harness.wait_state(0, worker_mod._FRAME_STAGED)
        assert not harness.cw._impl.prepared[0].is_set()

        _mailbox_store_i32(harness.state_addr(0), worker_mod._ACTIVATE)
        assert harness.cw._impl.launched[0].wait(5.0)
        harness.cw._impl.completed[0].set()
        harness.wait_state(0, worker_mod._TASK_DONE)
    finally:
        harness.close()


def test_two_frame_hbg_lone_prepare_ready_stages_before_native_prepare():
    harness = _TwoFrameLoopHarness(
        supports_concurrent_native_prepare=True,
        chip_runtime="host_build_graph",
    )
    try:
        harness.publish(0, 1, state=worker_mod._PREPARE_READY)
        harness.start()
        harness.wait_state(0, worker_mod._FRAME_STAGED)
        assert not harness.cw._impl.prepared[0].is_set()

        _mailbox_store_i32(harness.state_addr(0), worker_mod._ACTIVATE)
        assert harness.cw._impl.launched[0].wait(5.0)
        assert harness.cw._impl.prepared[0].is_set()
        harness.cw._impl.completed[0].set()
        harness.wait_state(0, worker_mod._TASK_DONE)
    finally:
        harness.close()


def test_two_frame_hbg_prepares_already_staged_successor_after_active_claim():
    harness = _TwoFrameLoopHarness(
        supports_concurrent_native_prepare=True,
        chip_runtime="host_build_graph",
    )
    try:
        harness.publish(1, 2, state=worker_mod._PREPARE_READY)
        harness.start()
        harness.wait_state(1, worker_mod._FRAME_STAGED)
        assert not harness.cw._impl.prepared[1].is_set()

        harness.publish(0, 1)
        assert harness.cw._impl.launched[0].wait(5.0)
        assert harness.cw._impl.prepared[1].wait(5.0)
        assert [event[:2] for event in harness.cw._impl.events if event[0] in {"prepare", "launch"}] == [
            ("prepare", 0),
            ("launch", 0),
            ("prepare", 1),
        ]

        _mailbox_store_i32(harness.state_addr(1), worker_mod._ACTIVATE)
        harness.cw._impl.completed[0].set()
        assert harness.cw._impl.launched[1].wait(5.0)
        harness.cw._impl.completed[1].set()
        harness.wait_state(1, worker_mod._TASK_DONE)
    finally:
        harness.close()


def test_two_frame_processes_control_while_native_run_is_active():
    harness = _TwoFrameLoopHarness()
    try:
        harness.publish(0, 1)
        harness.start()
        assert harness.cw._impl.launched[0].wait(5.0)

        struct.pack_into("Q", harness.buf, worker_mod._OFF_CALLABLE, worker_mod._CTRL_MALLOC)
        struct.pack_into("Q", harness.buf, worker_mod._CTRL_OFF_ARG0, 4096)
        _mailbox_store_i32(harness.mailbox_addr + _OFF_STATE, worker_mod._CONTROL_REQUEST)
        assert harness.cw.malloc_called.wait(5.0)
        assert _mailbox_load_i32(harness.mailbox_addr + _OFF_STATE) == worker_mod._CONTROL_DONE
        assert struct.unpack_from("Q", harness.buf, worker_mod._CTRL_OFF_RESULT)[0] == 0xCAFE
        assert not harness.cw._impl.completed[0].is_set()

        harness.cw._impl.completed[0].set()
        harness.wait_state(0, worker_mod._TASK_DONE)
    finally:
        harness.close()


def test_two_frame_defers_unregister_until_active_native_run_finalizes():
    harness = _TwoFrameLoopHarness()
    try:
        harness.publish(0, 1)
        harness.start()
        assert harness.cw._impl.launched[0].wait(5.0)

        harness.publish_unregister()
        harness.assert_control_stays_pending()
        assert harness.cw.unregister_calls == []
        assert harness.identity_table == {harness.digest: 7}

        harness.cw._impl.completed[0].set()
        assert harness.cw._impl.finalized[0].wait(5.0)
        harness.wait_control_state(worker_mod._CONTROL_DONE)
        assert harness.cw.unregister_calls == [7]
        assert harness.identity_table == {}
        assert harness.registry == {}
        assert harness.prepared == set()
    finally:
        harness.close()


def test_two_frame_defers_register_until_active_native_run_finalizes():
    harness = _TwoFrameLoopHarness()
    callable_obj = _unique_chip_callable(23)
    digest = _chip_digest(callable_obj, platform="a2a3")
    payload_shm = _chip_payload_shm(callable_obj)
    try:
        harness.publish(0, 1)
        harness.start()
        assert harness.cw._impl.launched[0].wait(5.0)

        harness.publish_register(callable_obj, payload_shm, digest)
        harness.assert_control_stays_pending()
        assert harness.cw._impl.register_calls == []
        assert digest not in harness.identity_table

        harness.cw._impl.completed[0].set()
        assert harness.cw._impl.finalized[0].wait(5.0)
        harness.wait_control_state(worker_mod._CONTROL_DONE)
        assert len(harness.cw._impl.register_calls) == 1
        cid = harness.identity_table[digest]
        assert harness.cw._impl.register_calls[0][0] == cid
        assert cid in harness.registry
        assert cid in harness.prepared
    finally:
        harness.close()
        payload_shm.close()
        payload_shm.unlink()


def test_two_frame_defers_register_until_backend_prepared_frame_finalizes():
    harness = _TwoFrameLoopHarness(
        supports_concurrent_native_prepare=True,
        chip_runtime="host_build_graph",
    )
    callable_obj = _unique_chip_callable(29)
    digest = _chip_digest(callable_obj, platform="a2a3", runtime="host_build_graph")
    payload_shm = _chip_payload_shm(callable_obj)
    try:
        harness.publish(0, 1)
        harness.start()
        assert harness.cw._impl.launched[0].wait(5.0)
        harness.publish(1, 2, state=worker_mod._PREPARE_READY)
        harness.wait_state(1, worker_mod._FRAME_STAGED)
        assert harness.cw._impl.prepared[1].is_set()

        harness.publish_register(callable_obj, payload_shm, digest)
        harness.assert_control_stays_pending()
        assert harness.cw._impl.register_calls == []

        _mailbox_store_i32(harness.state_addr(1), worker_mod._ACTIVATE)
        harness.cw._impl.completed[0].set()
        assert harness.cw._impl.launched[1].wait(5.0)
        harness.assert_control_stays_pending()
        harness.cw._impl.completed[1].set()
        harness.wait_state(1, worker_mod._TASK_DONE)
        harness.wait_control_state(worker_mod._CONTROL_DONE)
        assert len(harness.cw._impl.register_calls) == 1
    finally:
        harness.close()
        payload_shm.close()
        payload_shm.unlink()


def test_two_frame_defers_final_unregister_while_matching_frame_is_staged():
    harness = _TwoFrameLoopHarness()
    try:
        harness.publish(0, 1, state=worker_mod._PREPARE_READY)
        harness.start()
        harness.wait_state(0, worker_mod._FRAME_STAGED)

        harness.publish_unregister()
        harness.assert_control_stays_pending()
        assert harness.cw.unregister_calls == []

        _mailbox_store_i32(harness.state_addr(0), worker_mod._ACTIVATE)
        assert harness.cw._impl.launched[0].wait(5.0)
        harness.assert_control_stays_pending()
        assert harness.cw.unregister_calls == []

        harness.cw._impl.completed[0].set()
        harness.wait_state(0, worker_mod._TASK_DONE)
        harness.wait_control_state(worker_mod._CONTROL_DONE)
        assert harness.cw.unregister_calls == [7]
        assert harness.identity_table == {}
        assert harness.registry == {}
        assert harness.prepared == set()
    finally:
        harness.close()


def test_two_frame_native_unregister_failure_preserves_local_identity_state():
    harness = _TwoFrameLoopHarness()
    original_callable = harness.registry[7]
    harness.cw.unregister_error = RuntimeError("native unregister failed")
    try:
        harness.publish_unregister()
        harness.start()
        harness.wait_control_state(worker_mod._CONTROL_DONE)

        assert harness.cw.unregister_calls == [7]
        assert harness.identity_table == {harness.digest: 7}
        assert harness.identity_refs == {harness.digest: 1}
        assert harness.registry == {7: original_callable}
        assert harness.prepared == {7}
        assert struct.unpack_from("i", harness.buf, worker_mod._OFF_ERROR)[0] == 1
        error = bytes(
            harness.buf[
                worker_mod.MAILBOX_OFF_ERROR_MSG : worker_mod.MAILBOX_OFF_ERROR_MSG + worker_mod.MAILBOX_ERROR_MSG_SIZE
            ]
        ).split(b"\x00", 1)[0]
        assert b"native unregister failed" in error
    finally:
        harness.close()


def test_two_frame_shutdown_finalizes_active_and_fails_staged_without_launch():
    harness = _TwoFrameLoopHarness()
    try:
        harness.publish(0, 1)
        harness.start()
        assert harness.cw._impl.launched[0].wait(5.0)
        harness.publish(1, 2)
        harness.wait_state(1, worker_mod._FRAME_STAGED)

        _mailbox_store_i32(harness.mailbox_addr + _OFF_STATE, worker_mod._SHUTDOWN)
        harness.thread.join(5.0)
        assert not harness.thread.is_alive()
        assert harness.cw._impl.finalized[0].is_set()
        assert not harness.cw._impl.prepared[1].is_set()
        assert _mailbox_load_i32(harness.state_addr(1)) == worker_mod._TASK_FAILED
    finally:
        harness.close()


def test_two_frame_shutdown_finalizes_backend_prepared_successor_once():
    harness = _TwoFrameLoopHarness(
        supports_concurrent_native_prepare=True,
        chip_runtime="host_build_graph",
    )
    try:
        harness.publish(0, 1)
        harness.start()
        assert harness.cw._impl.launched[0].wait(5.0)
        harness.publish(1, 2, state=worker_mod._PREPARE_READY)
        harness.wait_state(1, worker_mod._FRAME_STAGED)
        assert harness.cw._impl.prepared[1].is_set()

        _mailbox_store_i32(harness.mailbox_addr + _OFF_STATE, worker_mod._SHUTDOWN)
        harness.thread.join(5.0)
        assert not harness.thread.is_alive()
        assert _mailbox_load_i32(harness.state_addr(1)) == worker_mod._TASK_FAILED
        assert not harness.cw._impl.launched[1].is_set()
        assert sum(event == ("finalize", 0) for event in harness.cw._impl.events) == 1
        assert sum(event == ("finalize", 1) for event in harness.cw._impl.events) == 1
    finally:
        harness.close()


def test_two_frame_stale_activation_finalizes_backend_prepared_successor_once():
    harness = _TwoFrameLoopHarness(
        supports_concurrent_native_prepare=True,
        chip_runtime="host_build_graph",
    )
    try:
        harness.publish(0, 1)
        harness.start()
        assert harness.cw._impl.launched[0].wait(5.0)
        harness.publish(1, 2, state=worker_mod._PREPARE_READY)
        harness.wait_state(1, worker_mod._FRAME_STAGED)
        assert harness.cw._impl.prepared[1].is_set()

        frame_offset = harness._frame_offset(1)
        struct.pack_into("=Q", harness.buf, frame_offset + worker_mod._OFF_FRAME_DISPATCH_ID, 99)
        _mailbox_store_i32(harness.state_addr(1), worker_mod._ACTIVATE)
        harness.wait_state(1, worker_mod._TASK_FAILED)
        assert not harness.cw._impl.launched[1].is_set()
        assert sum(event == ("finalize", 1) for event in harness.cw._impl.events) == 1

        harness.cw._impl.completed[0].set()
        harness.wait_state(0, worker_mod._TASK_DONE)
        assert sum(event == ("finalize", 1) for event in harness.cw._impl.events) == 1
    finally:
        harness.close()


def test_two_frame_reuses_completed_frame_for_next_staged_successor():
    harness = _TwoFrameLoopHarness()
    try:
        harness.publish(0, 1)
        harness.start()
        assert harness.cw._impl.launched[0].wait(5.0)

        harness.publish(1, 2, state=worker_mod._PREPARE_READY)
        harness.wait_state(1, worker_mod._FRAME_STAGED)
        assert _mailbox_load_i32(harness.accepted_addr(1)) == 0

        harness.cw._impl.completed[0].set()
        harness.wait_state(0, worker_mod._TASK_DONE)
        _mailbox_store_i32(harness.state_addr(1), worker_mod._ACTIVATE)
        harness.wait_state(1, worker_mod._TASK_LAUNCHED)
        assert _mailbox_load_i32(harness.accepted_addr(1)) == worker_mod._TASK_ACCEPTED

        harness.cw._impl.completed[0].clear()
        harness.publish(0, 3, state=worker_mod._PREPARE_READY, generation=12)
        harness.wait_state(0, worker_mod._FRAME_STAGED)
        assert _mailbox_load_i32(harness.accepted_addr(0)) == 0
        lifecycle_before_c_activation = [
            event[:2] for event in harness.cw._impl.events if event[0] in {"prepare", "launch", "finalize"}
        ]
        assert lifecycle_before_c_activation == [
            ("prepare", 0),
            ("launch", 0),
            ("finalize", 0),
            ("prepare", 1),
            ("launch", 1),
        ]

        harness.cw._impl.completed[1].set()
        harness.wait_state(1, worker_mod._TASK_DONE)
        _mailbox_store_i32(harness.state_addr(0), worker_mod._ACTIVATE)
        harness.wait_state(0, worker_mod._TASK_LAUNCHED)
        assert _mailbox_load_i32(harness.accepted_addr(0)) == worker_mod._TASK_ACCEPTED
        harness.cw._impl.completed[0].set()
        harness.wait_state(0, worker_mod._TASK_DONE)

        lifecycle = [event[:2] for event in harness.cw._impl.events if event[0] in {"prepare", "launch", "finalize"}]
        assert lifecycle == [
            ("prepare", 0),
            ("launch", 0),
            ("finalize", 0),
            ("prepare", 1),
            ("launch", 1),
            ("finalize", 1),
            ("prepare", 0),
            ("launch", 0),
            ("finalize", 0),
        ]
    finally:
        harness.close()


def test_two_frame_launch_failure_does_not_accept_and_finalizes_once():
    harness = _TwoFrameLoopHarness()
    try:
        harness.cw._impl.launch_errors[(0, 11)] = RuntimeError("launch failed")
        harness.publish(0, 1)
        harness.start()
        harness.wait_state(0, worker_mod._TASK_FAILED)

        assert _mailbox_load_i32(harness.accepted_addr(0)) == 0
        lifecycle = [event[:2] for event in harness.cw._impl.events if event[0] in {"prepare", "launch", "finalize"}]
        assert lifecycle == [("prepare", 0), ("finalize", 0)]
        assert harness.cw._impl.finalized[0].is_set()
        assert sum(event == ("finalize", 0) for event in harness.cw._impl.events) == 1
    finally:
        harness.close()


def test_two_frame_launch_cleanup_failure_terminalizes_staged_successor():
    harness = _TwoFrameLoopHarness()
    try:
        harness.cw._impl.launch_errors[(0, 11)] = RuntimeError("launch failed")
        harness.cw._impl.finalize_errors[(0, 11)] = RuntimeError("launch cleanup failed")
        harness.publish(0, 1)
        harness.publish(1, 2)
        harness.start()

        harness.wait_state(0, worker_mod._TASK_FAILED)
        harness.wait_state(1, worker_mod._TASK_FAILED)
        harness.thread.join(5.0)
        assert not harness.thread.is_alive()
        assert not harness.cw._impl.prepared[1].is_set()
        assert not harness.cw._impl.launched[1].is_set()
        assert _mailbox_load_i32(harness.accepted_addr(1)) == 0
    finally:
        harness.close()


@pytest.mark.parametrize("failure_point", ["poll", "finalize"])
def test_two_frame_native_progress_failure_terminalizes_staged_successor(failure_point):
    harness = _TwoFrameLoopHarness()
    try:
        harness.publish(0, 1)
        harness.start()
        assert harness.cw._impl.launched[0].wait(5.0)
        harness.publish(1, 2)
        harness.wait_state(1, worker_mod._FRAME_STAGED)

        if failure_point == "poll":
            harness.cw._impl.poll_errors[(0, 11)] = RuntimeError("poll failed")
        else:
            harness.cw._impl.finalize_errors[(0, 11)] = RuntimeError("finalize failed")
            harness.cw._impl.completed[0].set()

        harness.wait_state(0, worker_mod._TASK_FAILED)
        harness.wait_state(1, worker_mod._TASK_FAILED)
        harness.thread.join(5.0)
        assert not harness.thread.is_alive()
        assert not harness.cw._impl.prepared[1].is_set()
        assert not harness.cw._impl.launched[1].is_set()
        assert _mailbox_load_i32(harness.accepted_addr(1)) == 0
    finally:
        harness.close()


def _chip_digest(callable_obj: ChipCallable, *, platform: str = "", runtime: str = "") -> bytes:
    descriptor = build_chip_callable_descriptor(target=callable_obj, platform=platform, runtime=runtime)
    return hashid_to_digest(compute_callable_hashid(descriptor))


def _py_payload_digest(payload: bytes) -> bytes:
    return hashid_to_digest(compute_callable_hashid(build_python_serialized_descriptor(payload)))


def _unique_py_callable(index: int):
    def fn(args, _index=index):
        return _index

    return fn


def _unique_chip_callable(index: int):
    return ChipCallable.build(signature=[], func_name=f"x{index}", binary=bytes([index & 0xFF]), children=[])


# ---------------------------------------------------------------------------
# Test: lifecycle (init / close without submitting any tasks)
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_init_close_no_workers(self):
        hw = Worker(level=3, num_sub_workers=0)
        hw.init()
        hw.close()

    def test_init_close_with_sub_workers(self):
        hw = Worker(level=3, num_sub_workers=2)
        hw.init()
        hw.close()

    def test_context_manager(self):
        with Worker(level=3, num_sub_workers=1) as hw:
            hw.register(lambda args: None)
        # close() called by __exit__, no exception

    def test_l2_rejects_python_callable(self):
        hw = Worker(level=2, device_id=0, platform="a2a3sim", runtime="tensormap_and_ringbuffer")
        with pytest.raises(TypeError, match="level 2 only supports ChipCallable"):
            hw.register(lambda args: None)

    def test_close_releases_registered_callables(self):
        # close() must drop every Worker-held reference to registered callables.
        hw = Worker(level=3, num_sub_workers=1)
        hw.init()
        handle = hw.register(lambda args: None)
        assert _slot_for(hw, handle) in hw._callable_registry
        assert hw._identity_registry and hw._live_handles
        hw.close()
        assert hw._callable_registry == {}
        assert hw._identity_registry == {}
        assert hw._live_handles == {}

    def test_register_python_fn_before_init_lands_in_snapshot(self):
        # init() is eager, so there is no init-before-start window. A Python
        # callable registered before init() is frozen into the startup snapshot
        # and COW-inherited by the sub child during init — no run() needed.
        hw = Worker(level=3, num_sub_workers=1)
        handle = hw.register(lambda args: None)
        hw.init()
        try:
            assert _slot_for(hw, handle) in hw._callable_registry
        finally:
            hw.close()

    def test_pre_init_register_does_not_broadcast(self):
        # A registration issued before init() takes the pre-start snapshot path
        # and must not attempt a control broadcast — there is no started
        # hierarchy to broadcast to yet.
        hw = Worker(level=3, num_sub_workers=1)

        def _trap(*_a, **_k):
            raise AssertionError("pre-init Python register must not broadcast")

        hw._broadcast_py_control = _trap
        handle = hw.register(lambda args: None)
        hw.init()
        try:
            assert _slot_for(hw, handle) in hw._callable_registry
        finally:
            hw.close()

    def test_prepare_python_fn_after_start_no_python_children_raises(self):
        hw = Worker(level=3, num_sub_workers=0)
        hw.init()
        try:
            hw.run(lambda orch, args, cfg: None)
            with pytest.raises(ValueError, match=r"\(needs a SUB or next-level child\)"):
                hw.register(lambda args: None)
        finally:
            hw.close()

    def test_prepare_waits_for_first_startup_then_uses_post_start_path(self):
        hw = Worker(level=3, num_sub_workers=1)
        hw.init()
        try:
            with hw._hierarchical_start_cv:
                hw._lifecycle = worker_mod._Lifecycle.INITIALIZING

            observed = {}

            def fake_post_start_register(target):
                observed["target"] = target
                observed["state"] = hw._lifecycle
                observed["hierarchical_started"] = hw._hierarchical_started
                return 7

            hw._post_start_register_python = fake_post_start_register
            result: list[object] = []
            errors: list[BaseException] = []
            wait_entered = threading.Event()
            original_wait = hw._hierarchical_start_cv.wait

            def wait_with_signal(timeout=None):
                wait_entered.set()
                return original_wait(timeout)

            hw._hierarchical_start_cv.wait = wait_with_signal

            def do_register():
                try:
                    result.append(hw.register(lambda args: None))
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)

            t = threading.Thread(target=do_register)
            t.start()
            assert wait_entered.wait(timeout=2.0)
            with hw._hierarchical_start_cv:
                hw._lifecycle = worker_mod._Lifecycle.READY
                hw._hierarchical_start_cv.notify_all()
            t.join(timeout=2.0)

            assert not t.is_alive()
            assert errors == []
            assert result == [7]
            assert observed["state"] is worker_mod._Lifecycle.READY
            assert observed["hierarchical_started"] is True
        finally:
            if "original_wait" in locals():
                hw._hierarchical_start_cv.wait = original_wait
            hw.close()

    def test_unregister_waits_for_first_startup_then_uses_post_start_path(self):
        hw = Worker(level=3, num_sub_workers=1)
        handle = hw.register(lambda args: None)
        hw.init()
        try:
            with hw._hierarchical_start_cv:
                hw._lifecycle = worker_mod._Lifecycle.INITIALIZING

            observed = {}

            def fake_broadcast_py_control(worker_types, sub_cmd, *, digest=None, payload=None, strict=False):
                observed["worker_types"] = worker_types
                observed["sub_cmd"] = sub_cmd
                observed["digest"] = digest
                observed["state"] = hw._lifecycle
                observed["hierarchical_started"] = hw._hierarchical_started
                return []

            hw._broadcast_py_control = fake_broadcast_py_control
            errors: list[BaseException] = []
            wait_entered = threading.Event()
            original_wait = hw._hierarchical_start_cv.wait

            def wait_with_signal(timeout=None):
                wait_entered.set()
                return original_wait(timeout)

            hw._hierarchical_start_cv.wait = wait_with_signal

            def do_unregister():
                try:
                    hw.unregister(handle)
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)

            t = threading.Thread(target=do_unregister)
            t.start()
            assert wait_entered.wait(timeout=2.0)
            assert handle.digest in hw._identity_registry

            with hw._hierarchical_start_cv:
                hw._lifecycle = worker_mod._Lifecycle.READY
                hw._hierarchical_start_cv.notify_all()
            t.join(timeout=2.0)

            assert not t.is_alive()
            assert errors == []
            assert observed["sub_cmd"] == _CTRL_PY_UNREGISTER
            assert observed["digest"] == handle.digest
            assert observed["state"] is worker_mod._Lifecycle.READY
            assert observed["hierarchical_started"] is True
            assert handle.digest not in hw._identity_registry
        finally:
            if "original_wait" in locals():
                hw._hierarchical_start_cv.wait = original_wait
            hw.close()

    def test_register_during_initializing_waits_then_takes_post_start_path(self):
        # A register that races an in-progress startup epoch (INITIALIZING) must
        # block on the lifecycle condition rather than mutate the registry, then
        # resolve via the post-start broadcast path once the epoch commits READY.
        hw = Worker(level=3, num_sub_workers=1)
        hw.init()
        try:
            with hw._hierarchical_start_cv:
                hw._lifecycle = worker_mod._Lifecycle.INITIALIZING

            errors: list[BaseException] = []
            result: list[object] = []
            wait_entered = threading.Event()
            original_wait = hw._hierarchical_start_cv.wait

            def wait_with_signal(timeout=None):
                wait_entered.set()
                return original_wait(timeout)

            hw._hierarchical_start_cv.wait = wait_with_signal

            def do_register():
                try:
                    result.append(hw.register(lambda args: None))
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)

            t = threading.Thread(target=do_register)
            t.start()
            assert wait_entered.wait(timeout=2.0)
            # Still INITIALIZING: the register must be parked, not installed.
            assert len(hw._identity_registry) == 0

            with hw._hierarchical_start_cv:
                hw._lifecycle = worker_mod._Lifecycle.READY
                hw._hierarchical_start_cv.notify_all()
            t.join(timeout=2.0)

            assert not t.is_alive()
            assert errors == []
            assert len(result) == 1
        finally:
            if "original_wait" in locals():
                hw._hierarchical_start_cv.wait = original_wait
            hw.close()

    @requires_sim_binaries
    def test_prepare_chip_callable_after_init_succeeds(self, monkeypatch):
        # A post-init ChipCallable travels the whole dynamic-prepare path —
        # registry lock, cid allocation, broadcast, chip-child CTRL_REGISTER,
        # native register — and returns a handle bound to a live slot. The
        # chipless counterpart is the opposite claim and lives with the rest of
        # the eligibility rule, in
        # test_startup_readiness.py::TestEligibleTargetPrecheck.
        with fake_chip_l3(monkeypatch) as hw:
            handle = hw.register(chip_callable())
            assert isinstance(handle, CallableHandle)
            assert _slot_for(hw, handle) >= 0

    @requires_sim_binaries
    def test_prepare_chip_callable_surfaces_chip_child_register_failure(self, monkeypatch):
        # The broadcast reaches the chip child rather than terminating in the
        # parent: a native register that raises on the child comes back to the
        # caller as REGISTER_PARTIAL_FAILURE, and the handle is rolled back.
        with fake_chip_l3(monkeypatch, register_error="injected chip register failure") as hw:
            with pytest.raises(RuntimeError, match="REGISTER_PARTIAL_FAILURE"):
                hw.register(chip_callable())
            assert hw._callable_registry == {}

    def test_prepare_chip_callable_at_cid_overflow_raises(self):
        # cid budget is enforced under the new dynamic-prepare path.
        # Pre-fill registry with lambdas pre-init, init, then attempt one
        # post-init ChipCallable prepare. A sub child gives the pre-registered
        # python callables an eligible dispatch target.
        #
        # This worker is chipless, so the post-init ChipCallable fails the
        # eligibility re-check before reaching the cid budget — eligibility
        # fires first by design (more actionable than "out of slots").
        hw = Worker(level=3, num_sub_workers=1)
        try:
            for i in range(MAX_REGISTERED_CALLABLE_IDS):
                hw.register(_unique_py_callable(i))
            hw.init()
            with pytest.raises(ValueError, match=r"\(needs a chip device \(device_ids\)\)"):
                hw.register(chip_callable())
        finally:
            hw.close()

    def test_unregister_rejects_raw_slot_id(self):
        # Public unregister is handle-based. Raw slot ids are internal and
        # should not be accepted as a compatibility alias.
        hw = Worker(level=3, num_sub_workers=0)
        hw.init()
        try:
            with pytest.raises(TypeError, match="CallableHandle returned by Worker.register"):
                hw.unregister(999)
        finally:
            hw.close()

    @requires_sim_binaries
    def test_unregister_chip_callable_after_init_succeeds(self, monkeypatch):
        # Register and unregister both broadcast to a live chip child, which
        # drops the digest from its identity table and releases the native
        # slot. Also verifies slot reuse — unregistering frees the slot and the
        # next register reuses the same slot via `_allocate_cid`
        # (smallest-unused-integer).
        with fake_chip_l3(monkeypatch) as hw:
            callable_obj = chip_callable()
            handle_a = hw.register(callable_obj)
            slot_a = _slot_for(hw, handle_a)
            assert slot_a in hw._callable_registry
            hw.unregister(handle_a)
            assert slot_a not in hw._callable_registry
            handle_b = hw.register(callable_obj)
            assert _slot_for(hw, handle_b) == slot_a, "smallest-unused-cid policy should reuse the freed slot"

    def test_prepare_chip_callable_broadcast_runs_without_registry_lock(self):
        hw = Worker(level=3, num_sub_workers=0, device_ids=[0])
        hw._lifecycle = worker_mod._Lifecycle.READY
        callable_obj = ChipCallable.build(signature=[], func_name="x", binary=b"\x00", children=[])
        observed = {}

        def fake_post_init_register(target, digest, *, is_new):
            observed["target"] = target
            observed["digest"] = digest
            observed["is_new"] = is_new
            observed["locked"] = hw._registry_lock.locked()

        hw._post_init_register = fake_post_init_register

        handle = hw.register(callable_obj)

        slot = _slot_for(hw, handle)
        assert observed == {"target": callable_obj, "digest": handle.digest, "is_new": True, "locked": False}
        assert hw._callable_registry[slot] is callable_obj

    def test_register_child_chip_broadcast_runs_without_registry_lock(self):
        from simpler.worker import _build_callable_registration  # noqa: PLC0415

        hw = Worker(level=3, num_sub_workers=0, device_ids=[0])
        hw._lifecycle = worker_mod._Lifecycle.READY
        callable_obj = ChipCallable.build(signature=[], func_name="x", binary=b"\x00", children=[])
        digest = _build_callable_registration(hw, callable_obj).digest
        observed = {}

        def fake_post_init_register(target, digest, *, is_new):
            observed["target"] = target
            observed["digest"] = digest
            observed["is_new"] = is_new
            observed["locked"] = hw._registry_lock.locked()

        hw._post_init_register = fake_post_init_register

        result = hw._register_child_chip(callable_obj, digest=digest)

        assert result is None
        assert observed == {
            "target": callable_obj,
            "digest": digest,
            "is_new": True,
            "locked": False,
        }
        slot = hw._identity_registry[digest].slot_id
        assert hw._callable_registry[slot] is callable_obj

    def test_register_child_chip_rejects_tombstone_active_identity(self):
        hw = Worker(level=3, num_sub_workers=0)
        callable_obj = _unique_chip_callable(3)
        digest = _chip_digest(callable_obj)
        hw._register_child_chip(callable_obj, digest=digest)
        state = hw._identity_registry[digest]
        hw._pending_unregister_cids.add(state.slot_id)

        with pytest.raises(RuntimeError, match="REGISTER_TOMBSTONE_ACTIVE"):
            hw._register_child_chip(callable_obj, digest=digest)

        assert state.ref_count == 1
        assert hw._identity_registry[digest] is state

    def test_startup_identity_snapshot_filters_by_target_namespace(self):
        from simpler.worker import _make_local_identity_tables  # noqa: PLC0415

        hw = Worker(level=3, num_sub_workers=0)
        py_target = _unique_py_callable(1)
        chip_target = _unique_chip_callable(2)
        py_handle = hw.register(py_target)
        py_duplicate = hw.register(py_target)
        chip_handle = hw.register(chip_target)
        chip_duplicate = hw.register(chip_target)
        snapshot = [
            (digest, state.target, state.ref_count, state.kind, state.target_namespace)
            for digest, state in hw._identity_registry.items()
        ]

        py_registry, py_identity_table, py_refs = _make_local_identity_tables(
            snapshot,
            callable_kind="PYTHON_SERIALIZED",
            target_namespace="LOCAL_PYTHON",
        )
        chip_registry, chip_identity_table, chip_refs = _make_local_identity_tables(
            snapshot,
            callable_kind="CHIP_CALLABLE",
            target_namespace="LOCAL_CHIP",
        )

        assert set(py_identity_table) == {py_handle.digest}
        assert py_duplicate.digest == py_handle.digest
        assert py_refs == {py_handle.digest: 2}
        assert len(py_registry) == 1
        assert next(iter(py_registry.values())) is py_target
        assert chip_handle.digest not in py_identity_table

        assert set(chip_identity_table) == {chip_handle.digest}
        assert chip_duplicate.digest == chip_handle.digest
        assert chip_refs == {chip_handle.digest: 2}
        assert len(chip_registry) == 1
        assert next(iter(chip_registry.values())) is chip_target
        assert py_handle.digest not in chip_identity_table

    def test_python_control_broadcast_passes_default_timeout(self):
        from simpler.worker import _CTRL_PY_UNREGISTER, _PY_CONTROL_TIMEOUT_S  # noqa: PLC0415

        class FakeControlWorker:
            def __init__(self):
                self.calls = []

            def broadcast_control_all(self, worker_type, sub_cmd, payload=None, digest=None, timeout_s=None):
                self.calls.append((worker_type, sub_cmd, payload, digest, timeout_s))
                return []

        fake = FakeControlWorker()
        hw = Worker(level=3, num_sub_workers=1)
        hw._worker = fake
        digest = bytes([3]) * 32

        errors = hw._broadcast_py_control([WorkerType.SUB], _CTRL_PY_UNREGISTER, digest=digest, strict=False)

        assert errors == []
        assert fake.calls == [(WorkerType.SUB, _CTRL_PY_UNREGISTER, None, digest, _PY_CONTROL_TIMEOUT_S)]

    def test_cloudpickle_payload_roundtrip_supported_callable_shapes(self):
        class AddValue:
            def __init__(self, value):
                self.value = value

            def __call__(self, arg):
                return arg + self.value

        scale = 3

        def nested(arg):
            return arg * scale

        cases = [
            (lambda arg: arg + 1, 4, 5),
            (nested, 4, 12),
            (AddValue(7), 4, 11),
        ]
        for target, arg, expected in cases:
            loaded = _roundtrip_py_callable_payload(target)
            assert callable(loaded)
            assert loaded(arg) == expected

    def test_python_unregister_child_failure_warns_pops_and_allows_reuse(self, capsys):
        from simpler.worker import _CTRL_PY_REGISTER, _CTRL_PY_UNREGISTER  # noqa: PLC0415

        hw = Worker(level=3, num_sub_workers=1)
        handle = hw.register(lambda args: None)
        hw._lifecycle = worker_mod._Lifecycle.READY
        calls = []

        class FakeWorker:
            def broadcast_control_all(self, worker_type, sub_cmd, payload=None, digest=None, timeout_s=None):
                calls.append((worker_type, sub_cmd, digest, payload is not None, timeout_s))
                if sub_cmd == _CTRL_PY_UNREGISTER:
                    return [_FakeControlResult("SUB", 0, False, "injected unregister failure")]
                if sub_cmd == _CTRL_PY_REGISTER:
                    return [_FakeControlResult("SUB", 0, True)]
                raise AssertionError(f"unexpected sub_cmd={sub_cmd}")

        hw._worker = FakeWorker()

        slot = _slot_for(hw, handle)
        hw.unregister(handle)

        captured = capsys.readouterr()
        assert "Python children reported errors" in captured.err
        assert "injected unregister failure" in captured.err
        assert slot not in hw._callable_registry
        assert slot not in hw._pending_unregister_cids

        reused = hw.register(lambda args: None)
        assert _slot_for(hw, reused) == slot
        assert calls[0][:4] == (WorkerType.SUB, _CTRL_PY_UNREGISTER, handle.digest, False)
        assert calls[1][:4] == (WorkerType.SUB, _CTRL_PY_REGISTER, reused.digest, True)

    def test_pending_unregister_cid_is_not_reused_until_broadcast_returns(self):
        from simpler.worker import _CTRL_PY_REGISTER, _CTRL_PY_UNREGISTER  # noqa: PLC0415

        hw = Worker(level=3, num_sub_workers=1)
        handle = hw.register(lambda args: None)
        hw._lifecycle = worker_mod._Lifecycle.READY

        broadcast_started = threading.Event()
        release_broadcast = threading.Event()
        errors: list[BaseException] = []

        class FakeWorker:
            def broadcast_control_all(self, worker_type, sub_cmd, payload=None, digest=None, timeout_s=None):
                if sub_cmd == _CTRL_PY_UNREGISTER:
                    broadcast_started.set()
                    assert release_broadcast.wait(timeout=2.0)
                elif sub_cmd != _CTRL_PY_REGISTER:
                    raise AssertionError(f"unexpected sub_cmd={sub_cmd}")
                return [_FakeControlResult("SUB", 0, True)]

        hw._worker = FakeWorker()
        slot = _slot_for(hw, handle)

        def do_unregister():
            try:
                hw.unregister(handle)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        t = threading.Thread(target=do_unregister)
        t.start()
        assert broadcast_started.wait(timeout=2.0)

        handle_during_unregister = hw.register(lambda args: None)
        assert _slot_for(hw, handle_during_unregister) != slot
        assert slot in hw._pending_unregister_cids

        release_broadcast.set()
        t.join(timeout=2.0)
        assert not t.is_alive()
        assert errors == []

        handle_after_unregister = hw.register(lambda args: None)
        assert _slot_for(hw, handle_after_unregister) == slot

    def test_same_hashid_register_is_rejected_during_final_unregister(self):
        from simpler.worker import _CTRL_PY_REGISTER, _CTRL_PY_UNREGISTER  # noqa: PLC0415

        def target(args):
            return None

        hw = Worker(level=3, num_sub_workers=1)
        handle = hw.register(target)
        hw._lifecycle = worker_mod._Lifecycle.READY

        unregister_started = threading.Event()
        release_unregister = threading.Event()
        errors: list[BaseException] = []

        class FakeWorker:
            def broadcast_control_all(self, worker_type, sub_cmd, payload=None, digest=None, timeout_s=None):
                if sub_cmd == _CTRL_PY_UNREGISTER:
                    unregister_started.set()
                    assert release_unregister.wait(timeout=2.0)
                elif sub_cmd != _CTRL_PY_REGISTER:
                    raise AssertionError(f"unexpected sub_cmd={sub_cmd}")
                return [_FakeControlResult("SUB", 0, True)]

        hw._worker = FakeWorker()
        slot = _slot_for(hw, handle)

        def do_unregister():
            try:
                hw.unregister(handle)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        t = threading.Thread(target=do_unregister)
        t.start()
        assert unregister_started.wait(timeout=2.0)

        with pytest.raises(RuntimeError, match="REGISTER_TOMBSTONE_ACTIVE"):
            hw.register(target)

        release_unregister.set()
        t.join(timeout=2.0)
        assert not t.is_alive()
        assert errors == []

        handle_after_unregister = hw.register(target)
        assert _slot_for(hw, handle_after_unregister) == slot

    def test_same_hashid_register_is_rejected_during_nonfinal_unregister(self):
        from simpler.worker import _CTRL_PY_REGISTER, _CTRL_PY_UNREGISTER  # noqa: PLC0415

        def target(args):
            return None

        hw = Worker(level=3, num_sub_workers=1)
        first = hw.register(target)
        second = hw.register(target)
        hw._lifecycle = worker_mod._Lifecycle.READY

        unregister_started = threading.Event()
        release_unregister = threading.Event()
        errors: list[BaseException] = []

        class FakeWorker:
            def broadcast_control_all(self, worker_type, sub_cmd, payload=None, digest=None, timeout_s=None):
                if sub_cmd == _CTRL_PY_UNREGISTER:
                    unregister_started.set()
                    assert release_unregister.wait(timeout=2.0)
                elif sub_cmd != _CTRL_PY_REGISTER:
                    raise AssertionError(f"unexpected sub_cmd={sub_cmd}")
                return [_FakeControlResult("SUB", 0, True)]

        hw._worker = FakeWorker()
        slot = _slot_for(hw, first)

        def do_unregister():
            try:
                hw.unregister(first)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        t = threading.Thread(target=do_unregister)
        t.start()
        assert unregister_started.wait(timeout=2.0)

        with pytest.raises(RuntimeError, match="REGISTER_TOMBSTONE_ACTIVE"):
            hw.register(target)
        assert hw._identity_registry[second.digest].ref_count == 1

        release_unregister.set()
        t.join(timeout=2.0)
        assert not t.is_alive()
        assert errors == []
        assert slot not in hw._pending_unregister_cids

    def test_child_digest_unregister_tombstone_error_does_not_decrement_refcount(self):
        hw = Worker(level=3, num_sub_workers=0)
        handle = hw.register(lambda args: None)
        slot = _slot_for(hw, handle)
        state = hw._identity_registry[handle.digest]
        initial_ref_count = state.ref_count
        hw._pending_unregister_cids.add(slot)

        with pytest.raises(KeyError, match="UNREGISTER_TOMBSTONE_ACTIVE"):
            hw._unregister_child_digest(digest=handle.digest)

        assert state.ref_count == initial_ref_count
        assert hw._identity_registry[handle.digest] is state
        assert hw._callable_registry[slot] is state.target

    def test_register_python_sub_callable_after_start_succeeds(self):
        counter_shm, counter_buf = _make_shared_counter()
        try:
            hw = Worker(level=3, num_sub_workers=1)
            bootstrap_handle = hw.register(lambda args: None)
            hw.init()

            def bootstrap(orch, args, cfg):
                orch.submit_sub(bootstrap_handle)

            hw.run(bootstrap)
            counter_name = counter_shm.name

            def dynamic_sub(args):
                shm = SharedMemory(name=counter_name)
                try:
                    _increment_counter(shm.buf)
                finally:
                    shm.close()

            dynamic_handle = hw.register(dynamic_sub)

            def run_dynamic(orch, args, cfg):
                orch.submit_sub(dynamic_handle)

            hw.run(run_dynamic)
            hw.close()

            assert _read_counter(counter_buf) == 1
        finally:
            counter_shm.close()
            counter_shm.unlink()

    def test_post_start_python_register_waits_for_active_sub_mailbox(self):
        import time  # noqa: PLC0415

        control_shm = SharedMemory(create=True, size=8)
        counter_shm, counter_buf = _make_shared_counter()
        hw = Worker(level=3, num_sub_workers=1)
        run_errors: list[BaseException] = []
        register_errors: list[BaseException] = []
        dynamic_handles: list[CallableHandle] = []
        run_thread = None
        register_thread = None
        try:
            assert control_shm.buf is not None
            _set_flag(control_shm.buf, 0, 0)  # started
            _set_flag(control_shm.buf, 4, 0)  # release
            control_name = control_shm.name
            counter_name = counter_shm.name

            def blocking_sub(args):
                import time as child_time  # noqa: PLC0415

                shm = SharedMemory(name=control_name)
                try:
                    _set_flag(shm.buf, 0, 1)
                    while _get_flag(shm.buf, 4) == 0:
                        child_time.sleep(0.001)
                finally:
                    shm.close()

            blocking_handle = hw.register(blocking_sub)
            hw.init()

            def run_blocking():
                try:
                    hw.run(lambda orch, args, cfg: orch.submit_sub(blocking_handle))
                except BaseException as exc:  # noqa: BLE001
                    run_errors.append(exc)

            run_thread = threading.Thread(target=run_blocking)
            run_thread.start()

            deadline = time.monotonic() + 2.0
            while _get_flag(control_shm.buf, 0) == 0 and time.monotonic() < deadline:
                time.sleep(0.001)
            assert _get_flag(control_shm.buf, 0) == 1

            def dynamic_sub(args):
                shm = SharedMemory(name=counter_name)
                try:
                    _increment_counter(shm.buf)
                finally:
                    shm.close()

            def do_register():
                try:
                    dynamic_handles.append(hw.register(dynamic_sub))
                except BaseException as exc:  # noqa: BLE001
                    register_errors.append(exc)

            register_thread = threading.Thread(target=do_register)
            register_thread.start()
            register_thread.join(timeout=0.05)
            assert register_thread.is_alive()

            _set_flag(control_shm.buf, 4, 1)
            run_thread.join(timeout=2.0)
            register_thread.join(timeout=2.0)

            assert not run_thread.is_alive()
            assert not register_thread.is_alive()
            assert run_errors == []
            assert register_errors == []
            assert len(dynamic_handles) == 1

            hw.run(lambda orch, args, cfg: orch.submit_sub(dynamic_handles[0]))
            assert _read_counter(counter_buf) == 1
        finally:
            if control_shm.buf is not None:
                _set_flag(control_shm.buf, 4, 1)
            if run_thread is not None:
                run_thread.join(timeout=2.0)
            if register_thread is not None:
                register_thread.join(timeout=2.0)
            hw.close()
            control_shm.close()
            control_shm.unlink()
            counter_shm.close()
            counter_shm.unlink()

    def test_post_start_unregister_pre_start_python_callable_removes_child_entry(self):
        counter_shm, counter_buf = _make_shared_counter()
        try:
            hw = Worker(level=3, num_sub_workers=1)
            handle = hw.register(lambda args: _increment_counter(counter_buf))
            hw.init()

            hw.run(lambda orch, args, cfg: orch.submit_sub(handle))
            assert _read_counter(counter_buf) == 1

            slot = _slot_for(hw, handle)
            hw.unregister(handle)
            assert slot not in hw._callable_registry
            with pytest.raises(KeyError, match="not live"):
                hw.run(lambda orch, args, cfg: orch.submit_sub(handle))

            counter_name = counter_shm.name

            def replacement(args):
                shm = SharedMemory(name=counter_name)
                try:
                    _add_counter(shm.buf, 10)
                finally:
                    shm.close()

            reused = hw.register(replacement)
            assert _slot_for(hw, reused) == slot
            hw.run(lambda orch, args, cfg: orch.submit_sub(reused))
            hw.close()

            assert _read_counter(counter_buf) == 11
        finally:
            counter_shm.close()
            counter_shm.unlink()

    def test_post_start_unregister_post_start_python_callable_removes_child_entry(self):
        counter_shm, counter_buf = _make_shared_counter()
        try:
            hw = Worker(level=3, num_sub_workers=1)
            bootstrap_handle = hw.register(lambda args: None)
            hw.init()
            hw.run(lambda orch, args, cfg: orch.submit_sub(bootstrap_handle))

            counter_name = counter_shm.name

            def dynamic(args):
                shm = SharedMemory(name=counter_name)
                try:
                    _increment_counter(shm.buf)
                finally:
                    shm.close()

            handle = hw.register(dynamic)
            hw.run(lambda orch, args, cfg: orch.submit_sub(handle))
            assert _read_counter(counter_buf) == 1

            slot = _slot_for(hw, handle)
            hw.unregister(handle)
            assert slot not in hw._callable_registry
            with pytest.raises(KeyError, match="not live"):
                hw.run(lambda orch, args, cfg: orch.submit_sub(handle))

            reused = hw.register(dynamic)
            assert _slot_for(hw, reused) == slot
            hw.run(lambda orch, args, cfg: orch.submit_sub(reused))
            hw.close()

            assert _read_counter(counter_buf) == 2
        finally:
            counter_shm.close()
            counter_shm.unlink()

    def test_post_start_dynamic_python_callable_execute_failure_propagates(self):
        hw = Worker(level=3, num_sub_workers=1)
        bootstrap_handle = hw.register(lambda args: None)
        hw.init()
        try:
            hw.run(lambda orch, args, cfg: orch.submit_sub(bootstrap_handle))

            def boom(args):
                raise RuntimeError("dynamic callable boom")

            handle = hw.register(boom)
            with pytest.raises(RuntimeError, match="dynamic callable boom"):
                hw.run(lambda orch, args, cfg: orch.submit_sub(handle))
        finally:
            hw.close()

    def test_broadcast_control_all_accepts_memoryview_payload(self):
        counter_shm, counter_buf = _make_shared_counter()
        try:
            hw = Worker(level=3, num_sub_workers=1)
            bootstrap_handle = hw.register(lambda args: None)
            hw.init()

            def bootstrap(orch, args, cfg):
                orch.submit_sub(bootstrap_handle)

            hw.run(bootstrap)
            counter_name = counter_shm.name

            def dynamic_sub(args):
                shm = SharedMemory(name=counter_name)
                try:
                    _increment_counter(shm.buf)
                finally:
                    shm.close()

            worker_impl = hw._worker
            assert worker_impl is not None
            payload = _pack_py_callable_payload(dynamic_sub)
            digest = _py_payload_digest(payload)
            results = worker_impl.broadcast_control_all(
                WorkerType.SUB,
                _CTRL_PY_REGISTER,
                memoryview(payload),
                digest,
            )
            assert len(results) == 1
            assert results[0].ok
            unregister_results = worker_impl.broadcast_control_all(WorkerType.SUB, _CTRL_PY_UNREGISTER, None, digest)
            assert len(unregister_results) == 1
            assert unregister_results[0].ok
            hw.close()
        finally:
            counter_shm.close()
            counter_shm.unlink()

    def test_broadcast_control_all_reports_malformed_payload(self):
        hw = Worker(level=3, num_sub_workers=1)
        bootstrap_handle = hw.register(lambda args: None)
        hw.init()
        try:
            hw.run(lambda orch, args, cfg: orch.submit_sub(bootstrap_handle))
            worker_impl = hw._worker
            assert worker_impl is not None
            results = worker_impl.broadcast_control_all(WorkerType.SUB, _CTRL_PY_REGISTER, b"bad", bytes([6]) * 32)
            assert len(results) == 1
            assert not results[0].ok
            assert "payload" in results[0].error_message
        finally:
            hw.close()

    def test_broadcast_control_all_empty_payload_raises_before_fanout(self):
        hw = Worker(level=3, num_sub_workers=1)
        bootstrap_handle = hw.register(lambda args: None)
        hw.init()
        try:
            hw.run(lambda orch, args, cfg: orch.submit_sub(bootstrap_handle))
            worker_impl = hw._worker
            assert worker_impl is not None
            with pytest.raises(RuntimeError, match="payload pointer and size"):
                worker_impl.broadcast_control_all(WorkerType.SUB, _CTRL_PY_REGISTER, b"", bytes([7]) * 32)
        finally:
            hw.close()

    def test_broadcast_control_all_timeout_reports_failed_child(self):
        shm = SharedMemory(create=True, size=MAILBOX_SIZE)
        dw = _Worker(3)
        try:
            assert shm.buf is not None
            _mailbox_store_i32(_buffer_field_addr(shm.buf, _OFF_STATE), _IDLE)
            dw.add_sub_worker(_mailbox_addr(shm))
            dw.init()
            results = dw.broadcast_control_all(
                WorkerType.SUB,
                _CTRL_PY_UNREGISTER,
                None,
                bytes([8]) * 32,
                timeout_s=0.001,
            )
            assert len(results) == 1
            assert not results[0].ok
            assert "timed out" in results[0].error_message
        finally:
            dw.close()
            shm.close()
            shm.unlink()

    def test_broadcast_control_all_selected_pool_routing(self):
        def make_mailbox():
            shm = SharedMemory(create=True, size=MAILBOX_SIZE)
            assert shm.buf is not None
            _mailbox_store_i32(_buffer_field_addr(shm.buf, _OFF_STATE), _IDLE)
            return shm

        for selected_type, selected_kind in (
            (WorkerType.SUB, "SUB"),
            (WorkerType.NEXT_LEVEL, "NEXT_LEVEL"),
        ):
            sub_shm = make_mailbox()
            next_shm = make_mailbox()
            dw = _Worker(3)
            try:
                dw.add_sub_worker(_mailbox_addr(sub_shm))
                dw.add_next_level_worker(_mailbox_addr(next_shm))
                dw.init()
                results = dw.broadcast_control_all(
                    selected_type,
                    _CTRL_PY_UNREGISTER,
                    None,
                    bytes([9]) * 32,
                    timeout_s=0.001,
                )
                assert len(results) == 1
                assert results[0].worker_type == selected_kind
                sub_state = _mailbox_load_i32(_buffer_field_addr(sub_shm.buf, _OFF_STATE))
                next_state = _mailbox_load_i32(_buffer_field_addr(next_shm.buf, _OFF_STATE))
                if selected_type == WorkerType.SUB:
                    assert sub_state == _CONTROL_REQUEST
                    assert next_state == _IDLE
                else:
                    assert sub_state == _IDLE
                    assert next_state == _CONTROL_REQUEST
            finally:
                dw.close()
                sub_shm.close()
                sub_shm.unlink()
                next_shm.close()
                next_shm.unlink()

    def test_broadcast_control_all_result_shape_for_register_and_unregister(self):
        hw = Worker(level=3, num_sub_workers=1)
        bootstrap_handle = hw.register(lambda args: None)
        hw.init()
        try:
            hw.run(lambda orch, args, cfg: orch.submit_sub(bootstrap_handle))
            worker_impl = hw._worker
            assert worker_impl is not None
            register_results = worker_impl.broadcast_control_all(
                WorkerType.SUB,
                _CTRL_PY_REGISTER,
                b"bad",
                bytes([10]) * 32,
            )
            unregister_results = worker_impl.broadcast_control_all(
                WorkerType.SUB,
                _CTRL_PY_UNREGISTER,
                None,
                bootstrap_handle.digest,
            )

            for result in (register_results[0], unregister_results[0]):
                assert isinstance(result.worker_type, str)
                assert isinstance(result.worker_id, int)
                assert isinstance(result.ok, bool)
                assert isinstance(result.error_message, str)
            assert not register_results[0].ok
            assert unregister_results[0].ok
        finally:
            hw.close()

    def test_nonserializable_dynamic_python_callable_does_not_consume_cid(self):
        lock = threading.Lock()
        hw = Worker(level=3, num_sub_workers=1)
        bootstrap_handle = hw.register(lambda args: None)
        hw.init()
        try:
            hw.run(lambda orch, args, cfg: orch.submit_sub(bootstrap_handle))
            before = dict(hw._callable_registry)

            def captures_lock(args):
                lock.acquire(False)

            with pytest.raises(TypeError, match="lock"):
                hw.register(captures_lock)
            assert hw._callable_registry == before
        finally:
            hw.close()

    def test_duplicate_chip_prepare_broadcasts_ref_increment_without_new_slot(self):
        calls = []

        class FakeWorker:
            def broadcast_register_all(self, blob_ptr, blob_size, digest):
                calls.append(("binary_register", blob_size, digest))
                return [_FakeControlResult("NEXT_LEVEL", 0, True)]

        hw = Worker(level=3, num_sub_workers=1, device_ids=[0])
        hw._lifecycle = worker_mod._Lifecycle.READY
        hw._worker = FakeWorker()
        callable_obj = ChipCallable.build(signature=[], func_name="x", binary=b"\x00", children=[])

        first = hw.register(callable_obj)
        second = hw.register(callable_obj)

        slot = _slot_for(hw, first)
        assert slot == 0
        assert _slot_for(hw, second) == slot
        assert hw._identity_registry[first.digest].ref_count == 2
        assert calls == [
            ("binary_register", int(callable_obj.buffer_size()), first.digest),
            ("binary_register", int(callable_obj.buffer_size()), second.digest),
        ]

    def test_duplicate_chip_prepare_partial_failure_preserves_existing_handle(self):
        calls = []

        class FakeWorker:
            def __init__(self):
                self.register_count = 0

            def broadcast_register_all(self, blob_ptr, blob_size, digest):
                self.register_count += 1
                calls.append(("binary_register", self.register_count, digest))
                if self.register_count == 1:
                    return [_FakeControlResult("NEXT_LEVEL", 0, True), _FakeControlResult("NEXT_LEVEL", 1, True)]
                return [_FakeControlResult("NEXT_LEVEL", 0, True), _FakeControlResult("NEXT_LEVEL", 1, False, "boom")]

            def control_digest_only(self, worker_type, worker_id, sub_cmd, digest, timeout_s=None):
                calls.append(("cleanup_one", worker_type, worker_id, sub_cmd, digest))
                return _FakeControlResult("NEXT_LEVEL", worker_id, True)

        hw = Worker(level=3, num_sub_workers=1, device_ids=[0])
        hw._lifecycle = worker_mod._Lifecycle.READY
        hw._worker = FakeWorker()
        callable_obj = ChipCallable.build(signature=[], func_name="x", binary=b"\x00", children=[])

        first = hw.register(callable_obj)
        with pytest.raises(RuntimeError, match="REGISTER_PARTIAL_FAILURE"):
            hw.register(callable_obj)

        state = hw._resolve_handle(first)
        assert state.ref_count == 1
        assert hw._callable_registry[state.slot_id] is callable_obj
        assert first.digest not in hw._uncertain_hashids
        assert calls == [
            ("binary_register", 1, first.digest),
            ("binary_register", 2, first.digest),
            ("cleanup_one", WorkerType.NEXT_LEVEL, 0, _CTRL_UNREGISTER, first.digest),
        ]

    def test_chip_prepare_failure_rolls_back_handle_and_marks_uncertain_when_cleanup_fails(self):
        calls = []

        class FakeWorker:
            def broadcast_register_all(self, blob_ptr, blob_size, digest):
                calls.append(("binary_register", digest))
                raise RuntimeError("register failed")

            def broadcast_unregister_all(self, digest):
                calls.append(("cleanup", digest))
                return ["cleanup failed"]

        hw = Worker(level=3, num_sub_workers=1, device_ids=[0])
        hw._lifecycle = worker_mod._Lifecycle.READY
        hw._worker = FakeWorker()
        callable_obj = ChipCallable.build(signature=[], func_name="x", binary=b"\x00", children=[])

        with pytest.raises(RuntimeError, match="register failed"):
            hw.register(callable_obj)

        digest = next(iter(hw._uncertain_hashids))
        assert calls == [("binary_register", digest), ("cleanup", digest)]
        assert hw._callable_registry == {}
        with pytest.raises(RuntimeError, match="REGISTER_CLEANUP_UNCERTAIN"):
            hw.register(callable_obj)

    def test_unregister_middle_cid_reuses_hole(self):
        # `_allocate_cid` must fill the smallest hole, not append at
        # len(registry). The bug it guards against: fill slots 0/1/2,
        # unregister slot 1, next register would silently overwrite the
        # existing cid=2 under a `len(registry)` policy.
        hw = Worker(level=3, num_sub_workers=1)
        hw.init()
        try:
            cb0 = _unique_py_callable(0)
            cb1 = _unique_py_callable(1)
            cb2 = _unique_py_callable(2)
            cb3 = _unique_py_callable(3)
            handle0 = hw.register(cb0)
            handle1 = hw.register(cb1)
            handle2 = hw.register(cb2)
            slot0 = _slot_for(hw, handle0)
            slot1 = _slot_for(hw, handle1)
            slot2 = _slot_for(hw, handle2)
            assert (slot0, slot1, slot2) == (0, 1, 2)
            hw.unregister(handle1)
            reused_handle = hw.register(cb3)
            assert _slot_for(hw, reused_handle) == 1, "hole at cid=1 should be reused before appending"
            # cid=2 entry must still be the original callable, not silently overwritten.
            assert hw._callable_registry[slot2] is cb2
            # Next register fills cid=3 since 0..2 are all occupied.
            next_handle = hw.register(_unique_py_callable(4))
            assert _slot_for(hw, next_handle) == 3
        finally:
            hw.close()

    def test_prepare_overflow_raises(self):
        # The AICPU side reserves a fixed-size orch_so_table_[MAX_REGISTERED_CALLABLE_IDS];
        # Worker.register must surface the bound at register-time, not later when
        # DeviceRunner::register_callable rejects the private slot.
        hw = Worker(level=3, num_sub_workers=0)
        try:
            for i in range(MAX_REGISTERED_CALLABLE_IDS):
                hw.register(_unique_py_callable(i))
            with pytest.raises(RuntimeError, match="MAX_REGISTERED_CALLABLE_IDS"):
                hw.register(_unique_py_callable(MAX_REGISTERED_CALLABLE_IDS))
        finally:
            # init() was never called; close() is still safe (idempotent
            # against an uninitialised Worker).
            hw.close()


# ---------------------------------------------------------------------------
# Test: single independent SUB task executes and completes
# ---------------------------------------------------------------------------


class TestSingleSubTask:
    def test_sub_task_executes(self):
        counter_shm, counter_buf = _make_shared_counter()

        try:
            hw = Worker(level=3, num_sub_workers=1)
            handle = hw.register(lambda args: _increment_counter(counter_buf))
            hw.init()

            def orch(o, args, cfg):
                o.submit_sub(handle)

            hw.run(orch)
            hw.close()

            assert _read_counter(counter_buf) == 1
        finally:
            counter_shm.close()
            counter_shm.unlink()

    def test_sub_task_runs_multiple_times(self):
        counter_shm, counter_buf = _make_shared_counter()

        try:
            hw = Worker(level=3, num_sub_workers=1)
            handle = hw.register(lambda args: _increment_counter(counter_buf))
            hw.init()

            def orch(o, args, cfg):
                for _ in range(3):
                    o.submit_sub(handle)

            hw.run(orch)
            hw.close()

            assert _read_counter(counter_buf) == 3
        finally:
            counter_shm.close()
            counter_shm.unlink()


class TestRunHandle:
    @staticmethod
    def _submission_failure_worker(failures: int):
        events: list[str] = []

        class NativeWorker:
            def close(self):
                events.append("close")

        class NativeOrchestrator:
            def __init__(self):
                self.failures_left = failures

            def _begin_run(self):
                events.append("begin")
                return 1

            def _scope_begin(self):
                events.append("scope_begin")

            def _scope_end(self):
                events.append("scope_end")

            def _fail_run_submission(self, run_id, _error):
                assert run_id == 1
                events.append("fail")
                if self.failures_left:
                    self.failures_left -= 1
                    raise RuntimeError("injected cancellation failure")

            def _wait_run(self, run_id):
                assert run_id == 1
                events.append("wait")
                raise RuntimeError("native graph failure")

            def _release_run(self, run_id):
                assert run_id == 1
                events.append("release")

        worker = Worker(level=3, num_sub_workers=0)
        worker._worker = cast(Any, NativeWorker())
        worker._orch = cast(Any, NativeOrchestrator())
        return worker, events

    def test_graph_failure_retries_native_cancellation_before_waiting(self):
        worker, events = self._submission_failure_worker(failures=1)
        graph_error = ValueError("bad graph")

        def bad_graph(*_args):
            raise graph_error

        with pytest.raises(ValueError) as excinfo:
            worker._submit_l3_locked(bad_graph, None, cast(Any, object()))

        assert excinfo.value is graph_error
        assert events == ["begin", "scope_begin", "scope_end", "fail", "fail", "wait", "release"]
        assert not worker._accepted_run_handles
        assert worker._ordered_cleanup_error is None

    def test_graph_failure_still_emits_graph_build_span(self, monkeypatch):
        worker, _events = self._submission_failure_worker(failures=0)
        emitted = []
        timestamps = iter((100, 275))
        worker._host_trace_enabled = True
        monkeypatch.setattr(worker_mod, "HOST_STRACE_ENABLED", True)
        monkeypatch.setattr(worker_mod.time, "monotonic_ns", lambda: next(timestamps))
        monkeypatch.setattr(worker_mod, "_emit_host_span", lambda *args: emitted.append(args))

        def bad_graph(*_args):
            raise ValueError("bad graph")

        with pytest.raises(ValueError, match="bad graph"):
            worker._submit_l3_locked(bad_graph, None, cast(Any, object()))

        assert emitted == [("l3.graph_build", 1, 0, 0, 100, 175, "run_id=1 role=facade")]

    def test_unsettled_graph_cancellation_abandons_the_handle_before_close(self):
        worker, events = self._submission_failure_worker(failures=2)
        graph_error = ValueError("bad graph")

        def bad_graph(*_args):
            raise graph_error

        with pytest.raises(ValueError) as excinfo:
            worker._submit_l3_locked(bad_graph, None, cast(Any, object()))

        assert excinfo.value is graph_error
        assert events == ["begin", "scope_begin", "scope_end", "fail", "fail"]
        assert not worker._accepted_run_handles
        assert len(worker._abandoned_run_handles) == 1
        assert worker._abandoned_run_handles[0]._keepalive is not None
        assert worker._ordered_cleanup_error is not None
        with pytest.raises(RuntimeError, match="no further work is admitted"):
            worker._require_no_ordered_cleanup_failure("submit")

        worker.close()
        assert events[-1] == "close"
        assert not worker._abandoned_run_handles

    def test_unsettled_graph_cancellation_publishes_through_a_cv_enter_interrupt(self):
        publication_interrupt = KeyboardInterrupt("cancellation publication")
        cancellation_error = RuntimeError("injected cancellation failure")
        graph_error = ValueError("bad graph")

        class ArmableCV:
            def __init__(self, cv):
                self.cv = cv
                self.armed = False
                self.interrupts = 0

            def __enter__(self):
                if self.armed:
                    self.armed = False
                    self.interrupts += 1
                    raise publication_interrupt
                return self.cv.__enter__()

            def __exit__(self, *exc_info):
                return self.cv.__exit__(*exc_info)

            def notify_all(self):
                self.cv.notify_all()

        class NativeWorker:
            def close(self):
                return None

        class NativeOrchestrator:
            def __init__(self, lifecycle_cv):
                self.lifecycle_cv = lifecycle_cv
                self.cancellations = 0

            def _begin_run(self):
                return 1

            def _scope_begin(self):
                return None

            def _scope_end(self):
                return None

            def _fail_run_submission(self, run_id, _error):
                assert run_id == 1
                self.cancellations += 1
                if self.cancellations == worker_mod._RUN_CANCELLATION_ATTEMPTS:
                    self.lifecycle_cv.armed = True
                raise cancellation_error

        worker = Worker(level=3, num_sub_workers=0)
        lifecycle_cv = ArmableCV(worker._hierarchical_start_cv)
        worker._hierarchical_start_cv = cast(Any, lifecycle_cv)
        worker._worker = cast(Any, NativeWorker())
        worker._orch = cast(Any, NativeOrchestrator(lifecycle_cv))

        def bad_graph(*_args):
            raise graph_error

        with pytest.raises(ValueError) as caught:
            worker._submit_l3_locked(bad_graph, None, cast(Any, object()))

        assert caught.value is graph_error
        assert lifecycle_cv.interrupts == 1
        assert not worker._accepted_run_handles
        assert len(worker._abandoned_run_handles) == 1
        abandoned = worker._abandoned_run_handles[0]
        assert abandoned._terminal
        assert isinstance(abandoned._error, RuntimeError)
        assert abandoned._error is worker._ordered_cleanup_error
        assert abandoned._error.__cause__ is cancellation_error
        assert abandoned._keepalive is not None

        worker.close()
        assert abandoned._keepalive is None

    def test_submit_returns_before_completion_and_timeout_is_retryable(self):
        state_shm = SharedMemory(create=True, size=8)
        state_buf = state_shm.buf
        assert state_buf is not None
        _set_flag(state_buf, 0, 0)
        _set_flag(state_buf, 4, 0)
        try:
            hw = Worker(level=3, num_sub_workers=1)

            def delayed(_args):
                while _get_flag(state_buf, 0) == 0:
                    time.sleep(0.001)
                _set_flag(state_buf, 4, 1)

            target = hw.register(delayed)
            hw.init()
            callback_done = False

            def orch(o, _args, _cfg):
                nonlocal callback_done
                o.submit_sub(target)
                callback_done = True

            handle = hw.submit(orch)
            assert isinstance(handle, RunHandle)
            assert callback_done
            assert not handle.done
            with pytest.raises(TimeoutError, match="timed out"):
                handle.wait(0.01)
            assert not handle.done

            _set_flag(state_buf, 0, 1)
            handle.wait(5.0)
            assert handle.done
            assert _get_flag(state_buf, 4) == 1
            handle.wait()
            assert handle.result() is None
            hw.close()
        finally:
            state_shm.close()
            state_shm.unlink()

    def test_depth_two_prepares_next_run_and_blocks_third_callback(self):
        state_shm = SharedMemory(create=True, size=16)
        state_buf = state_shm.buf
        assert state_buf is not None
        for offset in range(0, 16, 4):
            _set_flag(state_buf, offset, 0)

        third_callback = threading.Event()
        third_result: dict[str, RunHandle] = {}
        submitter: Optional[threading.Thread] = None
        hw = Worker(level=3, num_sub_workers=2)
        try:

            def first_task(_args):
                _set_flag(state_buf, 0, 1)
                while _get_flag(state_buf, 4) == 0:
                    time.sleep(0.001)

            def second_task(_args):
                _set_flag(state_buf, 8, 1)
                while _get_flag(state_buf, 12) == 0:
                    time.sleep(0.001)

            first_target = hw.register(first_task)
            second_target = hw.register(second_task)
            hw.init()

            first = hw.submit(lambda o, _args, _cfg: o.submit_sub(first_target))
            deadline = time.monotonic() + 3.0
            while _get_flag(state_buf, 0) == 0 and time.monotonic() < deadline:
                time.sleep(0.001)
            assert _get_flag(state_buf, 0) == 1

            second_callback_done = False

            def second_graph(o, _args, _cfg):
                nonlocal second_callback_done
                o.submit_sub(second_target)
                second_callback_done = True

            second = hw.submit(second_graph)
            assert second_callback_done
            time.sleep(0.05)
            assert _get_flag(state_buf, 8) == 0, "prepared run dispatched before the active run became terminal"

            def third_graph(_o, _args, _cfg):
                third_callback.set()

            submitter = threading.Thread(target=lambda: third_result.setdefault("handle", hw.submit(third_graph)))
            submitter.start()
            assert not third_callback.wait(0.05), "third callback ran before depth-two admission freed a slot"

            _set_flag(state_buf, 4, 1)
            assert third_callback.wait(3.0)
            deadline = time.monotonic() + 3.0
            while _get_flag(state_buf, 8) == 0 and time.monotonic() < deadline:
                time.sleep(0.001)
            assert _get_flag(state_buf, 8) == 1

            _set_flag(state_buf, 12, 1)
            submitter.join(5.0)
            assert not submitter.is_alive()
            first.wait(5.0)
            second.wait(5.0)
            third_result["handle"].wait(5.0)
        finally:
            _set_flag(state_buf, 4, 1)
            _set_flag(state_buf, 12, 1)
            if submitter is not None:
                submitter.join(5.0)
            hw.close()
            state_shm.close()
            state_shm.unlink()

    def test_graph_error_is_synchronous(self):
        hw = Worker(level=3, num_sub_workers=0)
        hw.init()
        try:

            def bad_graph(_orch, _args, _cfg):
                raise ValueError("bad graph")

            with pytest.raises(ValueError, match="bad graph"):
                hw.submit(bad_graph)
            assert not hw._accepted_run_handles
        finally:
            hw.close()

    def test_failed_run_does_not_poison_next_submit(self):
        counter_shm, counter_buf = _make_shared_counter()
        try:
            hw = Worker(level=3, num_sub_workers=1)

            def fail(_args):
                raise RuntimeError("first run failed")

            failed_target = hw.register(fail)
            good_target = hw.register(lambda _args: _increment_counter(counter_buf))
            hw.init()
            failed = hw.submit(lambda o, _args, _cfg: o.submit_sub(failed_target))
            good = hw.submit(lambda o, _args, _cfg: o.submit_sub(good_target))

            with pytest.raises(RuntimeError, match="first run failed"):
                failed.wait()
            good.wait()
            assert _read_counter(counter_buf) == 1
            hw.close()
        finally:
            counter_shm.close()
            counter_shm.unlink()

    def test_close_drains_accepted_handle_and_rejects_later_submit(self):
        state_shm = SharedMemory(create=True, size=4)
        state_buf = state_shm.buf
        assert state_buf is not None
        _set_flag(state_buf, 0, 0)
        try:
            hw = Worker(level=3, num_sub_workers=1)

            def delayed(_args):
                while _get_flag(state_buf, 0) == 0:
                    time.sleep(0.001)

            target = hw.register(delayed)
            hw.init()
            handle = hw.submit(lambda o, _args, _cfg: o.submit_sub(target))
            releaser = threading.Thread(target=lambda: (time.sleep(0.1), _set_flag(state_buf, 0, 1)))
            releaser.start()
            try:
                hw.close()
            finally:
                _set_flag(state_buf, 0, 1)
                releaser.join(5.0)
            assert handle.done
            with pytest.raises(RuntimeError, match="requires an initialized"):
                hw.submit(lambda *_args: None)
        finally:
            state_shm.close()
            state_shm.unlink()

    def test_close_uses_one_deadline_for_operations_and_run_fences_then_retries(self, monkeypatch):
        class Clock:
            now = 0.0

        clock = Clock()
        monkeypatch.setattr(worker_mod.time, "monotonic", lambda: clock.now)
        monkeypatch.setattr(worker_mod, "_ROLLBACK_GRACEFUL_TIMEOUT_S", 10.0)

        wait_budgets: list[float] = []
        released_runs: list[int] = []
        native_closes: list[str] = []

        class NativeWorker:
            def close(self):
                native_closes.append("close")

        class NativeOrchestrator:
            complete = False

            def _wait_run_for(self, run_id, timeout):
                assert run_id == 1
                wait_budgets.append(timeout)
                if not self.complete:
                    clock.now += timeout
                    return False
                return True

            def _release_run(self, run_id):
                released_runs.append(run_id)

        worker = Worker(level=3, num_sub_workers=0)
        native_worker = NativeWorker()
        native_orch = NativeOrchestrator()
        worker._worker = cast(Any, native_worker)
        worker._orch = cast(Any, native_orch)
        handle = RunHandle(worker, 1, ())
        worker._accepted_run_handles.add(handle)
        worker._active_ops = 1

        real_cv = worker._hierarchical_start_cv

        class AdvancingCondition:
            def __enter__(self):
                return real_cv.__enter__()

            def __exit__(self, *exc_info):
                return real_cv.__exit__(*exc_info)

            def wait(self, timeout=None):
                assert timeout == pytest.approx(10.0)
                clock.now += 7.0
                worker._active_ops = 0
                return True

            def notify_all(self):
                real_cv.notify_all()

        worker._hierarchical_start_cv = cast(Any, AdvancingCondition())

        with pytest.raises(TimeoutError, match="run fence.*cleanup budget"):
            worker.close()

        assert wait_budgets == [pytest.approx(3.0)]
        assert handle in worker._accepted_run_handles
        assert not handle._terminal
        assert worker._worker is native_worker
        assert not worker._teardown_attempted
        assert worker._close_completion is not None and worker._close_completion.incomplete
        assert native_closes == []

        native_orch.complete = True
        worker.close()

        assert wait_budgets == [pytest.approx(3.0), pytest.approx(10.0)]
        assert released_runs == [1]
        assert handle._terminal
        assert not worker._accepted_run_handles
        assert worker._teardown_attempted
        assert worker._close_completion is not None and not worker._close_completion.incomplete
        assert native_closes == ["close"]

    def test_submit_close_race_accepts_and_drains_admitted_run(self):
        callback_entered = threading.Event()
        callback_release = threading.Event()
        result: dict[str, object] = {}
        hw = Worker(level=3, num_sub_workers=0)
        hw.init()

        def orch(_o, _args, _cfg):
            callback_entered.set()
            assert callback_release.wait(5.0)

        submitter = threading.Thread(target=lambda: result.setdefault("handle", hw.submit(orch)))
        submitter.start()
        assert callback_entered.wait(3.0)
        releaser = threading.Thread(target=lambda: (time.sleep(0.1), callback_release.set()))
        releaser.start()
        try:
            hw.close()
        finally:
            callback_release.set()
            submitter.join(5.0)
            releaser.join(5.0)
        handle = result["handle"]
        assert isinstance(handle, RunHandle)
        assert handle.done

    def test_orchestration_callable_is_kept_alive_until_completion(self):
        class Keepalive:
            pass

        state_shm = SharedMemory(create=True, size=4)
        state_buf = state_shm.buf
        assert state_buf is not None
        _set_flag(state_buf, 0, 0)
        token = Keepalive()
        token_ref = weakref.ref(token)
        try:
            hw = Worker(level=3, num_sub_workers=1)

            def delayed(_args):
                while _get_flag(state_buf, 0) == 0:
                    time.sleep(0.001)

            target = hw.register(delayed)
            hw.init()

            def orch(o, _args, _cfg, held=token):
                assert held is not None
                o.submit_sub(target)

            handle = hw.submit(orch)
            del orch, token
            gc.collect()
            assert token_ref() is not None
            _set_flag(state_buf, 0, 1)
            handle.wait(5.0)
            gc.collect()
            assert token_ref() is None
            hw.close()
        finally:
            _set_flag(state_buf, 0, 1)
            state_shm.close()
            state_shm.unlink()

    def test_run_delegates_to_submit_and_wait(self, monkeypatch):
        events: list[tuple] = []

        class FakeHandle:
            def wait(self):
                events.append(("wait",))

        def fake_submit(self, callable, args=None, config=None):
            events.append(("submit", callable, args, config))
            return FakeHandle()

        monkeypatch.setattr(Worker, "submit", fake_submit)
        worker = Worker(level=3, num_sub_workers=0)

        def callback(*_args):
            return None

        worker.run(callback, args="args", config="config")
        assert events == [("submit", callback, "args", "config"), ("wait",)]

    def test_serialization_drain_does_not_swallow_async_interrupt(self, monkeypatch):
        handle = RunHandle._completed(Worker(level=3, num_sub_workers=0))

        def interrupted_wait(_self, _timeout=None):
            raise KeyboardInterrupt

        monkeypatch.setattr(RunHandle, "wait", interrupted_wait)
        with pytest.raises(KeyboardInterrupt):
            handle._wait_for_serialization()

    def test_acceptance_wait_does_not_block_completion_waiter(self):
        acceptance_entered = threading.Event()
        acceptance_release = threading.Event()
        completion_entered = threading.Event()
        completion_release = threading.Event()

        class FakeWorker:
            def _wait_run_handle_accepted(self, run_id):
                assert run_id == 1
                acceptance_entered.set()
                assert acceptance_release.wait(5.0)

            def _wait_run_handle(self, run_id, timeout):
                assert run_id == 1
                completion_entered.set()
                assert completion_release.wait(5.0)
                return True

            def _finalize_run_handle(self, handle, run_id, error):
                return error

        handle = RunHandle(cast(Worker, FakeWorker.__new__(FakeWorker)), 1, ())
        completion_thread = threading.Thread(target=handle.wait)
        acceptance_thread = threading.Thread(target=handle._wait_for_acceptance)
        completion_thread.start()
        assert completion_entered.wait(3.0)
        acceptance_thread.start()
        try:
            assert acceptance_entered.wait(3.0)
        finally:
            acceptance_release.set()
            completion_release.set()
            acceptance_thread.join(5.0)
            completion_thread.join(5.0)
        assert not acceptance_thread.is_alive()
        assert not completion_thread.is_alive()

    def test_done_query_cannot_race_native_run_release(self):
        done_entered = threading.Event()
        done_release = threading.Event()
        wait_entered = threading.Event()

        class FakeWorker:
            def _run_handle_done(self, run_id):
                assert run_id == 1
                done_entered.set()
                assert done_release.wait(5.0)
                return True

            def _wait_run_handle(self, run_id, timeout):
                assert run_id == 1
                wait_entered.set()
                return True

            def _finalize_run_handle(self, handle, run_id, error):
                assert run_id == 1
                return error

        handle = RunHandle(cast(Worker, FakeWorker.__new__(FakeWorker)), 1, ())
        observed: list[bool] = []
        done_thread = threading.Thread(target=lambda: observed.append(handle.done))
        wait_thread = threading.Thread(target=handle.wait)
        done_thread.start()
        assert done_entered.wait(3.0)
        wait_thread.start()
        try:
            assert not wait_entered.wait(0.1)
        finally:
            done_release.set()
            done_thread.join(5.0)
            wait_thread.join(5.0)
        assert observed == [True]
        assert wait_entered.is_set()

    def test_interrupted_finalize_still_publishes_terminal_state(self):
        interrupt = KeyboardInterrupt()

        class FakeWorker:
            def _wait_run_handle(self, run_id, timeout):
                return True

            def _finalize_run_handle(self, handle, run_id, error):
                raise interrupt

        handle = RunHandle(cast(Worker, FakeWorker.__new__(FakeWorker)), 1, ())
        with pytest.raises(KeyboardInterrupt) as first:
            handle.wait()
        assert first.value is interrupt
        assert handle.done
        # The handle is terminal, so this must resolve from the cached result
        # instead of re-electing a waiter that blocks on the native fence.
        with pytest.raises(KeyboardInterrupt) as second:
            handle.wait(5.0)
        assert second.value is interrupt

    def test_waiter_is_not_stranded_when_finalize_is_interrupted(self):
        finalize_entered = threading.Event()
        finalize_release = threading.Event()

        class FakeWorker:
            def _wait_run_handle(self, run_id, timeout):
                return True

            def _finalize_run_handle(self, handle, run_id, error):
                finalize_entered.set()
                assert finalize_release.wait(5.0)
                raise KeyboardInterrupt

        handle = RunHandle(cast(Worker, FakeWorker.__new__(FakeWorker)), 1, ())
        elected: list[BaseException] = []
        parked: list[BaseException] = []

        def run(sink, timeout):
            try:
                handle.wait(timeout)
            except BaseException as exc:  # noqa: BLE001
                sink.append(exc)

        elected_thread = threading.Thread(target=run, args=(elected, None))
        elected_thread.start()
        assert finalize_entered.wait(3.0)
        parked_thread = threading.Thread(target=run, args=(parked, 5.0))
        parked_thread.start()
        finalize_release.set()
        elected_thread.join(5.0)
        parked_thread.join(5.0)
        assert not elected_thread.is_alive()
        assert not parked_thread.is_alive()
        assert [type(exc) for exc in elected] == [KeyboardInterrupt]
        assert [type(exc) for exc in parked] == [KeyboardInterrupt]
        assert handle.done

    def test_finalize_retires_handle_when_its_cv_acquire_is_interrupted(self):
        interrupt = KeyboardInterrupt()

        class OnceInterruptingCV:
            def __init__(self, cv):
                self._cv = cv
                self._armed = True

            def __enter__(self):
                if self._armed:
                    self._armed = False
                    raise interrupt
                return self._cv.__enter__()

            def __exit__(self, *exc_info):
                return self._cv.__exit__(*exc_info)

            def notify_all(self):
                self._cv.notify_all()

        hw = Worker(level=3, num_sub_workers=0)
        handle = RunHandle(hw, 1, ())
        hw._accepted_run_handles.add(handle)
        hw._orch = cast(object, type("FakeOrch", (), {"_release_run": lambda self, run_id: None})())
        hw._hierarchical_start_cv = cast(threading.Condition, OnceInterruptingCV(hw._hierarchical_start_cv))

        assert hw._finalize_run_handle(handle, 1, None) is interrupt
        assert not hw._accepted_run_handles

    def test_finalize_drains_after_a_post_step_interrupt_and_close_completes(self):
        interrupt = KeyboardInterrupt("post-step")
        released_refs: list[str] = []
        released_runs: list[int] = []

        class SlotRef:
            def _release_slot_ref(self):
                released_refs.append("released")

        class NativeOrchestrator:
            def _wait_run(self, run_id):
                assert run_id == 1

            def _release_run(self, run_id):
                released_runs.append(run_id)

        worker = Worker(level=3, num_sub_workers=0)
        worker._orch = cast(Any, NativeOrchestrator())
        resources = worker_mod._RunResources()
        resources.remote_slot_refs.append(cast(Any, SlotRef()))
        handle = RunHandle(worker, 1, (object(),), resources)
        worker._accepted_run_handles.add(handle)
        original_finalize = worker._finalize_run_handle
        interrupted = False

        def interrupt_after_remote_refs(step: str) -> None:
            nonlocal interrupted
            if step == "remote_slot_refs" and not interrupted:
                interrupted = True
                raise interrupt

        worker._finalize_run_handle = cast(
            Any,
            lambda finalized, run_id, error: original_finalize(
                finalized,
                run_id,
                error,
                _after_step=interrupt_after_remote_refs,
            ),
        )

        with pytest.raises(KeyboardInterrupt) as caught:
            handle.wait()

        assert caught.value is interrupt
        assert released_refs == ["released"]
        assert released_runs == [1]
        assert handle._cleanup_published
        assert handle not in worker._accepted_run_handles
        assert handle._keepalive is None
        worker.close()

    def test_wait_interrupted_after_election_is_re_electable(self):
        interrupt = KeyboardInterrupt("after election")
        nested_interrupt = SystemExit("while clearing election")
        native_waits: list[int] = []
        released_runs: list[int] = []

        class InterruptingHandle(RunHandle):
            interrupt_clear = False

            def __setattr__(self, name, value):
                if name == "_wait_in_progress" and value is False and self.interrupt_clear:
                    self.interrupt_clear = False
                    raise nested_interrupt
                return super().__setattr__(name, value)

        class NativeOrchestrator:
            def _run_done(self, run_id):
                assert run_id == 1
                return False

            def _wait_run(self, run_id):
                native_waits.append(run_id)

            def _release_run(self, run_id):
                released_runs.append(run_id)

        worker = Worker(level=3, num_sub_workers=0)
        worker._orch = cast(Any, NativeOrchestrator())
        handle = InterruptingHandle(worker, 1, ())
        worker._accepted_run_handles.add(handle)
        interrupted = False

        def interrupt_after_election(phase: str) -> None:
            nonlocal interrupted
            if phase == "after_election" and not interrupted:
                interrupted = True
                raise interrupt

        handle._wait_boundary_hook = interrupt_after_election
        handle.interrupt_clear = True

        with pytest.raises(KeyboardInterrupt) as caught:
            handle.wait()

        assert caught.value is interrupt
        assert not handle._wait_in_progress
        assert not handle.done
        assert native_waits == []
        assert released_runs == []
        assert handle in worker._accepted_run_handles

        handle.wait()

        assert native_waits == [1]
        assert released_runs == [1]
        assert handle.done
        assert handle not in worker._accepted_run_handles
        worker.close()

    def test_wait_interrupted_after_finalize_publishes_terminal_once(self):
        interrupt = KeyboardInterrupt("after finalize")
        nested_interrupt = SystemExit("while publishing terminal")
        native_waits: list[int] = []
        released_runs: list[int] = []

        class InterruptingHandle(RunHandle):
            interrupt_terminal = False

            def __setattr__(self, name, value):
                if name == "_terminal" and value is True and self.interrupt_terminal:
                    self.interrupt_terminal = False
                    raise nested_interrupt
                return super().__setattr__(name, value)

        class NativeOrchestrator:
            def _wait_run(self, run_id):
                native_waits.append(run_id)

            def _release_run(self, run_id):
                released_runs.append(run_id)

        worker = Worker(level=3, num_sub_workers=0)
        worker._orch = cast(Any, NativeOrchestrator())
        handle = InterruptingHandle(worker, 1, ())
        worker._accepted_run_handles.add(handle)
        interrupted = False

        def interrupt_after_finalize(phase: str) -> None:
            nonlocal interrupted
            if phase == "after_finalize" and not interrupted:
                interrupted = True
                raise interrupt

        handle._wait_boundary_hook = interrupt_after_finalize
        handle.interrupt_terminal = True

        with pytest.raises(KeyboardInterrupt) as caught:
            handle.wait()

        assert caught.value is interrupt
        assert native_waits == [1]
        assert released_runs == [1]
        assert handle.done
        assert not handle._wait_in_progress
        assert handle not in worker._accepted_run_handles
        with pytest.raises(KeyboardInterrupt) as repeated:
            handle.wait()
        assert repeated.value is interrupt
        assert native_waits == [1]
        assert released_runs == [1]
        worker.close()

    def test_finalization_recovery_survives_an_interrupt_after_accepted_retirement(self):
        finalization_interrupt = KeyboardInterrupt("finalization escaped")
        recovery_interrupt = SystemExit("after accepted retirement")
        recoveries: list[str] = []

        class NativeWorker:
            def close(self):
                return None

        class NativeOrchestrator:
            def _wait_run(self, run_id):
                assert run_id == 1

        worker = Worker(level=3, num_sub_workers=0)
        worker._worker = cast(Any, NativeWorker())
        worker._orch = cast(Any, NativeOrchestrator())
        keepalive = object()
        handle = RunHandle(worker, 1, (keepalive,))
        worker._accepted_run_handles.add(handle)
        worker._finalize_run_handle = cast(Any, lambda *_args: (_ for _ in ()).throw(finalization_interrupt))
        recover = worker._recover_interrupted_run_finalization

        def interrupt_after_recovery(recovering, error):
            recoveries.append("recover")
            recovered = recover(recovering, error)
            if len(recoveries) == 1:
                raise recovery_interrupt
            return recovered

        worker._recover_interrupted_run_finalization = cast(Any, interrupt_after_recovery)

        with pytest.raises(KeyboardInterrupt) as caught:
            handle.wait()

        assert caught.value is finalization_interrupt
        assert recoveries == ["recover", "recover"]
        assert handle._terminal
        assert not handle._wait_in_progress
        assert handle._error is finalization_interrupt
        assert handle not in worker._accepted_run_handles
        assert worker._abandoned_run_handles == [handle]
        assert handle._keepalive == (keepalive,)
        with pytest.raises(KeyboardInterrupt) as repeated:
            handle.wait()
        assert repeated.value is finalization_interrupt

        worker.close()
        assert handle._keepalive is None

    def test_wait_timeout_includes_acceptance_owner_handoff(self):
        native_waits: list[tuple[str, Any]] = []
        released_runs: list[int] = []

        class NativeOrchestrator:
            def _wait_run_for(self, run_id, timeout):
                assert run_id == 1
                native_waits.append(("timed", timeout))
                return True

            def _wait_run(self, run_id):
                assert run_id == 1
                native_waits.append(("untimed", None))

            def _release_run(self, run_id):
                released_runs.append(run_id)

        worker = Worker(level=3, num_sub_workers=0)
        worker._orch = cast(Any, NativeOrchestrator())
        handle = RunHandle(worker, 1, ())
        handle._accept_wait_in_progress = True
        worker._accepted_run_handles.add(handle)
        finished = threading.Event()
        errors: list[BaseException] = []

        def wait_with_expired_deadline() -> None:
            try:
                handle.wait(0)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
            finally:
                finished.set()

        waiter = threading.Thread(target=wait_with_expired_deadline)
        waiter.start()
        try:
            assert finished.wait(0.5), "RunHandle.wait(0) ignored its deadline during acceptance handoff"
            assert len(errors) == 1
            assert isinstance(errors[0], TimeoutError)
            assert handle._accept_wait_in_progress
            assert not handle._wait_in_progress
            assert handle in worker._accepted_run_handles
            assert released_runs == []
        finally:
            handle._clear_acceptance_owner()
            waiter.join(5.0)

        handle.wait()
        assert native_waits[0][0] == "timed"
        assert native_waits[1] == ("untimed", None)
        assert released_runs == [1]
        worker.close()

    def test_finalizer_escape_after_retirement_preserves_cleanup_error(self):
        cleanup_error = RuntimeError("cleanup failed")
        boundary_error = KeyboardInterrupt("return boundary")
        native_waits: list[int] = []
        released_runs: list[int] = []

        class SlotRef:
            def _release_slot_ref(self):
                raise cleanup_error

        class NativeOrchestrator:
            def _wait_run(self, run_id):
                native_waits.append(run_id)

            def _release_run(self, run_id):
                released_runs.append(run_id)

        worker = Worker(level=3, num_sub_workers=0)
        worker._orch = cast(Any, NativeOrchestrator())
        resources = worker_mod._RunResources()
        resources.remote_slot_refs.append(cast(Any, SlotRef()))
        handle = RunHandle(worker, 1, (), resources)
        worker._accepted_run_handles.add(handle)
        original_finalize = worker._finalize_run_handle

        def interrupt_return(finalized, run_id, native_error):
            assert original_finalize(finalized, run_id, native_error) is cleanup_error
            raise boundary_error

        worker._finalize_run_handle = cast(Any, interrupt_return)

        with pytest.raises(RuntimeError) as caught:
            handle.wait()

        assert caught.value is cleanup_error
        assert handle._error is cleanup_error
        assert worker._ordered_cleanup_error is cleanup_error
        assert native_waits == [1]
        assert released_runs == [1]
        assert handle not in worker._accepted_run_handles
        with pytest.raises(RuntimeError) as repeated:
            handle.wait()
        assert repeated.value is cleanup_error

    def test_acceptance_wait_interrupted_after_election_is_re_electable(self):
        interrupt = KeyboardInterrupt("after acceptance election")
        nested_interrupt = SystemExit("while clearing acceptance election")
        native_waits: list[int] = []

        class InterruptingHandle(RunHandle):
            interrupt_clear = False

            def __setattr__(self, name, value):
                if name == "_accept_wait_in_progress" and value is False and self.interrupt_clear:
                    self.interrupt_clear = False
                    raise nested_interrupt
                return super().__setattr__(name, value)

        class FakeWorker:
            def _wait_run_handle_accepted(self, run_id):
                native_waits.append(run_id)

        handle = InterruptingHandle(cast(Worker, FakeWorker()), 1, ())
        interrupted = False

        def interrupt_after_election(phase: str) -> None:
            nonlocal interrupted
            if phase == "after_acceptance_election" and not interrupted:
                interrupted = True
                raise interrupt

        handle._wait_boundary_hook = interrupt_after_election
        handle.interrupt_clear = True

        with pytest.raises(KeyboardInterrupt) as caught:
            handle._wait_for_acceptance()

        assert caught.value is interrupt
        assert not handle._accept_wait_in_progress
        assert not handle._launch_accepted
        assert native_waits == []

        handle._wait_for_acceptance()

        assert native_waits == [1]
        assert handle._launch_accepted
        assert not handle._accept_wait_in_progress

    def test_acceptance_wait_interrupted_after_native_wait_stays_published(self):
        interrupt = KeyboardInterrupt("after acceptance wait")
        nested_interrupt = SystemExit("while publishing acceptance")
        native_waits: list[int] = []

        class InterruptingHandle(RunHandle):
            interrupt_publish = False

            def __setattr__(self, name, value):
                if name == "_accept_wait_in_progress" and value is False and self.interrupt_publish:
                    self.interrupt_publish = False
                    raise nested_interrupt
                return super().__setattr__(name, value)

        class FakeWorker:
            def _wait_run_handle_accepted(self, run_id):
                native_waits.append(run_id)

        handle = InterruptingHandle(cast(Worker, FakeWorker()), 1, ())
        interrupted = False

        def interrupt_after_wait(phase: str) -> None:
            nonlocal interrupted
            if phase == "after_acceptance_wait" and not interrupted:
                interrupted = True
                raise interrupt

        handle._wait_boundary_hook = interrupt_after_wait
        handle.interrupt_publish = True

        with pytest.raises(KeyboardInterrupt) as caught:
            handle._wait_for_acceptance()

        assert caught.value is interrupt
        assert native_waits == [1]
        assert handle._launch_accepted
        assert not handle._accept_wait_in_progress

        handle._wait_for_acceptance()
        assert native_waits == [1]

    def test_interrupted_cleanup_is_abandoned_until_tree_teardown(self, monkeypatch):
        interrupt = KeyboardInterrupt("ambiguous step boundary")
        released_refs: list[str] = []
        released_runs: list[int] = []
        teardown_calls: list[str] = []

        class SlotRef:
            def __init__(self, name, *, interrupts=False):
                self.name = name
                self.interrupts = interrupts

            def _release_slot_ref(self):
                released_refs.append(self.name)
                if self.interrupts:
                    raise interrupt

        class NativeOrchestrator:
            def _wait_run(self, run_id):
                assert run_id == 1

            def _release_run(self, run_id):
                released_runs.append(run_id)

        worker = Worker(level=3, num_sub_workers=0)
        worker._worker = cast(Any, object())
        worker._orch = cast(Any, NativeOrchestrator())
        resources = worker_mod._RunResources()
        first_ref = SlotRef("first", interrupts=True)
        second_ref = SlotRef("second")
        resources.remote_slot_refs.extend([cast(Any, first_ref), cast(Any, second_ref)])
        keepalive = object()
        handle = RunHandle(worker, 1, (keepalive, first_ref, second_ref), resources)
        worker._accepted_run_handles.add(handle)

        def teardown_tree():
            teardown_calls.append("teardown")
            worker._worker = None
            worker._orch = None

        monkeypatch.setattr(worker, "_teardown_ready_tree", teardown_tree)

        with pytest.raises(KeyboardInterrupt) as caught:
            handle.wait()

        assert caught.value is interrupt
        assert released_refs == ["first"]
        assert released_runs == []
        assert handle not in worker._accepted_run_handles
        assert handle._cleanup_published
        assert handle._keepalive == (keepalive, first_ref, second_ref)
        assert worker._abandoned_run_handles == [handle]
        assert isinstance(worker._ordered_cleanup_error, RuntimeError)
        assert worker._ordered_cleanup_error.__cause__ is interrupt

        worker.close()

        assert teardown_calls == ["teardown"]
        assert handle._keepalive is None
        assert not worker._abandoned_run_handles

    def test_post_teardown_keepalive_drain_survives_an_interrupt(self, monkeypatch):
        interrupt = KeyboardInterrupt("keepalive release")
        nested_interrupt = SystemExit("keepalive traversal back-edge")
        teardown_calls: list[str] = []

        class InterruptingHandleList(list):
            interrupt_bool = False

            def __bool__(self):
                if self.interrupt_bool:
                    self.interrupt_bool = False
                    raise nested_interrupt
                return len(self) != 0

        retained_handles = InterruptingHandleList()

        class InterruptingRetainedHandle:
            def __init__(self):
                self.value = object()
                self.interrupted = False

            @property
            def _keepalive(self):
                return self.value

            @_keepalive.setter
            def _keepalive(self, value):
                if not self.interrupted:
                    self.interrupted = True
                    retained_handles.interrupt_bool = True
                    raise interrupt
                self.value = value

        worker = Worker(level=3, num_sub_workers=0)
        worker._worker = cast(Any, object())
        retained = InterruptingRetainedHandle()
        retained_handles.append(retained)
        worker._abandoned_run_handles = cast(Any, retained_handles)

        def teardown_tree():
            teardown_calls.append("teardown")
            worker._worker = None

        monkeypatch.setattr(worker, "_teardown_ready_tree", teardown_tree)

        with pytest.raises(KeyboardInterrupt) as caught:
            worker.close()

        assert caught.value is interrupt
        assert teardown_calls == ["teardown"]
        assert retained._keepalive is None
        assert not worker._abandoned_run_handles

    def test_post_teardown_keepalive_drain_precedes_the_residual_probe(self, monkeypatch):
        residual_probe_interrupt = KeyboardInterrupt("residual probe")
        teardown_calls: list[str] = []
        probe_calls: list[str] = []
        worker = Worker(level=3, num_sub_workers=0)
        worker._worker = cast(Any, object())
        keepalive = object()
        retained = RunHandle(worker, 1, (keepalive,))
        retained._finalization_abandoned = True
        worker._abandoned_run_handles.append(retained)

        def teardown_tree():
            teardown_calls.append("teardown")
            worker._worker = None

        def has_live_resources():
            probe_calls.append("probe")
            if len(probe_calls) == 1:
                return True
            raise residual_probe_interrupt

        monkeypatch.setattr(worker, "_teardown_ready_tree", teardown_tree)
        monkeypatch.setattr(worker, "_has_live_resources", has_live_resources)

        with pytest.raises(KeyboardInterrupt) as caught:
            worker.close()

        assert caught.value is residual_probe_interrupt
        assert teardown_calls == ["teardown"]
        assert probe_calls == ["probe", "probe"]
        assert retained._keepalive is None
        assert not worker._abandoned_run_handles

        replayed: list[BaseException] = []

        def retry_close() -> None:
            try:
                worker.close()
            except BaseException as exc:  # noqa: BLE001
                replayed.append(exc)

        retry = threading.Thread(target=retry_close)
        retry.start()
        retry.join(1.0)
        assert not retry.is_alive()
        assert replayed == [residual_probe_interrupt]
        assert teardown_calls == ["teardown"]

    def test_cancellation_abandonment_survives_nested_publication_interrupts(self):
        lifecycle_interrupt = KeyboardInterrupt("lifecycle publication")
        terminal_interrupt = SystemExit("terminal publication")
        cancellation_error = RuntimeError("cancellation did not settle")

        class OnceInterruptingCV:
            def __init__(self, cv):
                self.cv = cv
                self.interrupted = False

            def __enter__(self):
                if not self.interrupted:
                    self.interrupted = True
                    raise lifecycle_interrupt
                return self.cv.__enter__()

            def __exit__(self, *exc_info):
                return self.cv.__exit__(*exc_info)

            def notify_all(self):
                self.cv.notify_all()

        class InterruptingHandle(RunHandle):
            interrupt_terminal = False

            def __setattr__(self, name, value):
                if name == "_terminal" and value is True and self.interrupt_terminal:
                    self.interrupt_terminal = False
                    raise terminal_interrupt
                return super().__setattr__(name, value)

        worker = Worker(level=3, num_sub_workers=0)
        handle = InterruptingHandle(worker, 1, (object(),))
        handle.interrupt_terminal = True
        worker._accepted_run_handles.add(handle)
        worker._hierarchical_start_cv = cast(Any, OnceInterruptingCV(worker._hierarchical_start_cv))

        worker._abandon_unsettled_run(handle, str(cancellation_error), cancellation_error)

        assert handle._terminal
        assert not handle._wait_in_progress
        assert not handle._accept_wait_in_progress
        assert isinstance(handle._error, RuntimeError)
        assert str(handle._error) == str(cancellation_error)
        assert handle._error.__cause__ is cancellation_error
        assert worker._ordered_cleanup_error is handle._error
        assert handle._keepalive is not None
        assert handle not in worker._accepted_run_handles
        assert worker._abandoned_run_handles == [handle]

    def test_run_finalization_releases_only_its_resources(self, monkeypatch):
        class SlotRef:
            def __init__(self):
                self.releases = 0

            def _release_slot_ref(self):
                self.releases += 1

        class Region:
            def __init__(self, region_id):
                self.region_id = region_id
                self._worker_id = 0
                self.mapping_closed = False
                self.expired = False

            def _close_worker_host_mapping(self):
                self.mapping_closed = True

            def _expire(self):
                self.expired = True

        class NativeWorker:
            def __init__(self):
                self.released_regions = []

            def control_worker_chip_region_release(self, worker_id, region_id):
                self.released_regions.append((worker_id, region_id))

        class NativeOrchestrator:
            def __init__(self):
                self.released_runs = []

            def _release_run(self, run_id):
                self.released_runs.append(run_id)

        def domain(name, allocation_id):
            return worker_mod.CommDomainHandle(
                name=name,
                workers=(),
                contexts={},
                allocation_id=allocation_id,
                _release_fn=lambda _handle: None,
            )

        worker = Worker(level=3, num_sub_workers=0)
        native_worker = NativeWorker()
        native_orch = NativeOrchestrator()
        worker._worker = cast(Any, native_worker)
        worker._orch = cast(Any, native_orch)

        first = worker_mod._RunResources()
        second = worker_mod._RunResources()
        first_ref, second_ref = SlotRef(), SlotRef()
        first_region, second_region = Region(11), Region(22)
        first_live, second_live = domain("first-live", 1), domain("second-live", 2)
        first_pending, second_pending = domain("first-pending", 3), domain("second-pending", 4)
        first.remote_slot_refs.append(cast(Any, first_ref))
        second.remote_slot_refs.append(cast(Any, second_ref))
        first.worker_chip_regions.append(first_region)
        second.worker_chip_regions.append(second_region)
        first.worker_chip_orch_comm_host_buffers[0x1000] = 64
        second.worker_chip_orch_comm_host_buffers[0x2000] = 128
        first.live_domains[first_live.name] = first_live
        second.live_domains[second_live.name] = second_live
        first.pending_release_domains.append(first_pending)
        second.pending_release_domains.append(second_pending)
        worker._live_worker_chip_regions.extend([first_region, second_region])
        worker._live_domains.update({first_live.name: first_live, second_live.name: second_live})

        released_domains = []

        def release_domain_now(handle):
            released_domains.append(handle)
            if worker._live_domains.get(handle.name) is handle:
                worker._live_domains.pop(handle.name)

        monkeypatch.setattr(worker, "_release_domain_now", release_domain_now)
        first_handle = RunHandle(worker, 1, (), first)
        second_handle = RunHandle(worker, 2, (), second)
        worker._accepted_run_handles.update({first_handle, second_handle})

        assert worker._finalize_run_handle(first_handle, 1, None) is None

        assert first_ref.releases == 1
        assert second_ref.releases == 0
        assert native_worker.released_regions == [(0, 11)]
        assert first_region.mapping_closed and first_region.expired
        assert not second_region.mapping_closed and not second_region.expired
        assert worker._live_worker_chip_regions == [second_region]
        assert first.worker_chip_orch_comm_host_buffers == {}
        assert second.worker_chip_orch_comm_host_buffers == {0x2000: 128}
        assert released_domains == [first_pending, first_live]
        assert first_pending.freed and first_live.freed
        assert not second_pending.freed and not second_live.freed
        assert worker._live_domains == {second_live.name: second_live}
        assert native_orch.released_runs == [1]
        assert worker._accepted_run_handles == {second_handle}

    def test_region_mapping_failure_still_releases_child_and_poisons_successor(self):
        mapping_error = KeyboardInterrupt("mapping close")
        release_error = SystemExit("child release")

        class Region:
            region_id = 11
            _worker_id = 0

            def __init__(self):
                self.expired = False

            def _close_worker_host_mapping(self):
                raise mapping_error

            def _expire(self):
                self.expired = True

        class NativeWorker:
            def __init__(self):
                self.released_regions = []

            def control_worker_chip_region_release(self, worker_id, region_id):
                self.released_regions.append((worker_id, region_id))
                raise release_error

        class NativeOrchestrator:
            def _release_run(self, run_id):
                raise AssertionError(f"ambiguous run {run_id} must stay owned until tree teardown")

        worker = Worker(level=3, num_sub_workers=0)
        native_worker = NativeWorker()
        worker._worker = cast(Any, native_worker)
        worker._orch = cast(Any, NativeOrchestrator())
        resources = worker_mod._RunResources()
        resources.requires_ordered_cleanup = True
        region = Region()
        resources.worker_chip_regions.append(region)
        worker._live_worker_chip_regions.append(region)
        handle = RunHandle(worker, 1, (), resources)
        worker._accepted_run_handles.add(handle)

        assert worker._finalize_run_handle(handle, 1, None) is mapping_error

        assert native_worker.released_regions == [(0, 11)]
        assert region.expired
        assert resources.worker_chip_regions == []
        assert worker._live_worker_chip_regions == []
        assert worker._ordered_cleanup_error is not None
        with pytest.raises(RuntimeError, match="no further work is admitted"):
            worker._require_no_ordered_cleanup_failure("submit")

    def test_region_tracking_interrupt_still_retires_every_region(self):
        interrupt = KeyboardInterrupt("region tracking removal")

        class InterruptingList(list):
            def __init__(self, values):
                super().__init__(values)
                self.interrupted = False

            def __setitem__(self, key, value):
                super().__setitem__(key, value)
                if isinstance(key, slice) and not self.interrupted:
                    self.interrupted = True
                    raise interrupt

        class Region:
            def __init__(self, region_id):
                self.region_id = region_id
                self._worker_id = 0
                self.mapping_closes = 0
                self.expires = 0

            def _close_worker_host_mapping(self):
                self.mapping_closes += 1

            def _expire(self):
                self.expires += 1

        class NativeWorker:
            def __init__(self):
                self.released_regions = []

            def control_worker_chip_region_release(self, worker_id, region_id):
                self.released_regions.append((worker_id, region_id))

        first, second = Region(11), Region(22)
        resources = worker_mod._RunResources()
        resources.worker_chip_regions = cast(Any, InterruptingList([first, second]))
        worker = Worker(level=3, num_sub_workers=0)
        native_worker = NativeWorker()
        worker._worker = cast(Any, native_worker)
        worker._live_worker_chip_regions.extend([first, second])

        with pytest.raises(KeyboardInterrupt) as caught:
            worker._cleanup_worker_chip_regions(resources)

        assert caught.value is interrupt
        assert native_worker.released_regions == [(0, 11), (0, 22)]
        assert (first.mapping_closes, first.expires) == (1, 1)
        assert (second.mapping_closes, second.expires) == (1, 1)
        assert resources.worker_chip_regions == []
        assert worker._live_worker_chip_regions == []

    def test_domain_released_after_its_run_retired_is_freed_inline(self):
        """A late release has no fence left to defer behind, so it frees now.

        Both deferred paths are closed to it: the run's queue is never drained
        again, and _release_domain_handle has already dropped the handle from
        _live_domains, so close()'s live sweep cannot reach it either.
        """
        worker = Worker(level=3, num_sub_workers=0)
        worker._worker = cast(Any, object())
        worker._orch = cast(Any, type("FakeOrch", (), {"_release_run": lambda self, run_id: None})())

        freed: list[str] = []

        def release_now(handle):
            freed.append(handle.name)
            if worker._live_domains.get(handle.name) is handle:
                worker._live_domains.pop(handle.name)

        worker._release_domain_now = cast(Any, release_now)

        resources = worker_mod._RunResources()
        late = worker_mod.CommDomainHandle(
            name="late",
            workers=(),
            contexts={},
            allocation_id=1,
            _release_fn=lambda released, owner=resources: worker._release_domain_handle(released, owner),
        )
        resources.live_domains[late.name] = late
        worker._live_domains[late.name] = late

        handle = RunHandle(worker, 1, (), resources)
        worker._accepted_run_handles.add(handle)
        assert worker._finalize_run_handle(handle, 1, None) is None
        assert freed == ["late"], "a domain still live at its run's fence is swept there"

        second = worker_mod.CommDomainHandle(
            name="later",
            workers=(),
            contexts={},
            allocation_id=2,
            _release_fn=lambda released, owner=resources: worker._release_domain_handle(released, owner),
        )
        resources.live_domains[second.name] = second
        worker._live_domains[second.name] = second

        second.release()

        assert freed == ["late", "later"]
        assert second.freed
        assert resources.pending_release_domains == []

    def test_interrupted_domain_queue_publication_keeps_live_owners(self):
        interrupt = KeyboardInterrupt("pending-domain publication")

        class InterruptingList(list):
            def append(self, value):
                raise interrupt

        worker = Worker(level=3, num_sub_workers=0)
        worker._worker = cast(Any, object())
        resources = worker_mod._RunResources()
        resources.pending_release_domains = cast(Any, InterruptingList())
        handle = worker_mod.CommDomainHandle(
            name="owned",
            workers=(),
            contexts={},
            allocation_id=1,
            _release_fn=lambda released, owner=resources: worker._release_domain_handle(released, owner),
        )
        resources.live_domains[handle.name] = handle
        worker._live_domains[handle.name] = handle

        with pytest.raises(KeyboardInterrupt) as caught:
            handle.release()

        assert caught.value is interrupt
        assert handle.released
        assert resources.live_domains == {handle.name: handle}
        assert worker._live_domains == {handle.name: handle}
        assert resources.pending_release_domains == []

    def test_interrupted_post_fence_domain_free_keeps_live_owners(self):
        interrupt = KeyboardInterrupt("post-fence domain free")
        worker = Worker(level=3, num_sub_workers=0)
        worker._worker = cast(Any, object())
        resources = worker_mod._RunResources(retired=True)
        handle = worker_mod.CommDomainHandle(
            name="owned",
            workers=(),
            contexts={},
            allocation_id=1,
            _release_fn=lambda released, owner=resources: worker._release_domain_handle(released, owner),
        )
        resources.live_domains[handle.name] = handle
        worker._live_domains[handle.name] = handle
        worker._free_domain_after_fence = cast(Any, lambda _handle: (_ for _ in ()).throw(interrupt))

        with pytest.raises(KeyboardInterrupt) as caught:
            handle.release()

        assert caught.value is interrupt
        assert handle.released
        assert resources.live_domains == {handle.name: handle}
        assert worker._live_domains == {handle.name: handle}
        assert resources.pending_release_domains == []

    def test_pending_domain_drain_does_not_clear_unclaimed_owners(self):
        class NoClearList(list):
            def clear(self):
                raise AssertionError("pending owners must stay published until backend success")

        worker = Worker(level=3, num_sub_workers=0)
        worker._worker = cast(Any, object())
        resources = worker_mod._RunResources()
        handle = worker_mod.CommDomainHandle(
            name="pending",
            workers=(),
            contexts={},
            allocation_id=1,
            _release_fn=lambda released: None,
        )
        handle._released = True
        resources.pending_release_domains = cast(Any, NoClearList([handle]))
        freed = []

        def free_domain(pending):
            freed.append(pending)
            pending._freed = True

        worker._free_domain_after_fence = cast(Any, free_domain)

        worker._execute_pending_domain_releases(resources)

        assert freed == [handle]
        assert resources.pending_release_domains == []

    def test_failed_pending_domain_drain_retains_its_claim(self):
        interrupt = KeyboardInterrupt("pending domain free")
        worker = Worker(level=3, num_sub_workers=0)
        worker._worker = cast(Any, object())
        resources = worker_mod._RunResources()
        handle = worker_mod.CommDomainHandle(
            name="pending",
            workers=(),
            contexts={},
            allocation_id=1,
            _release_fn=lambda released: None,
        )
        handle._released = True
        resources.pending_release_domains.append(handle)
        worker._free_domain_after_fence = cast(Any, lambda _handle: (_ for _ in ()).throw(interrupt))

        with pytest.raises(KeyboardInterrupt) as caught:
            worker._execute_pending_domain_releases(resources)

        assert caught.value is interrupt
        assert resources.pending_release_domains == [handle]

    def test_domain_free_outcome_precedes_caller_interrupt(self, monkeypatch):
        interrupt = KeyboardInterrupt("after isolated domain free")
        worker = Worker(level=3, num_sub_workers=0)
        worker._worker = cast(Any, object())
        handle = worker_mod.CommDomainHandle(
            name="domain",
            workers=(),
            contexts={},
            allocation_id=7,
            _release_fn=lambda released: None,
        )
        backend_calls = []
        worker._release_domain_claimed = cast(Any, lambda claimed: backend_calls.append(claimed.allocation_id))

        def interrupt_after_target(items, target, **kwargs):
            for item in items:
                target(item)
            raise interrupt

        monkeypatch.setattr(worker_mod, "_start_and_join_threads", interrupt_after_target)

        with pytest.raises(KeyboardInterrupt) as caught:
            worker._release_domain_now(handle)

        assert caught.value is interrupt
        assert backend_calls == [7]
        assert worker._domain_free_results == {7: None}

        worker._release_domain_now(handle)
        assert backend_calls == [7]

    def test_domain_free_interrupted_before_isolated_admission_is_retryable(self, monkeypatch):
        interrupt = KeyboardInterrupt("before isolated domain free admission")
        worker = Worker(level=3, num_sub_workers=0)
        worker._worker = cast(Any, object())
        handle = worker_mod.CommDomainHandle(
            name="domain",
            workers=(),
            contexts={},
            allocation_id=7,
            _release_fn=lambda released: None,
        )
        backend_calls: list[int] = []
        worker._release_domain_claimed = cast(Any, lambda claimed: backend_calls.append(claimed.allocation_id))
        real_isolated_call = worker_mod._run_isolated_call
        monkeypatch.setattr(
            worker_mod,
            "_run_isolated_call",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(interrupt),
        )

        with pytest.raises(KeyboardInterrupt) as caught:
            worker._release_domain_now(handle)

        assert caught.value is interrupt
        assert backend_calls == []
        assert 7 not in worker._domain_free_results

        monkeypatch.setattr(worker_mod, "_run_isolated_call", real_isolated_call)
        worker._release_domain_now(handle)

        assert backend_calls == [7]
        assert worker._domain_free_results == {7: None}

    def test_domain_free_publication_failure_never_replays_backend(self):
        publication_error = MemoryError("domain outcome store")

        class InterruptingResults(dict):
            def __init__(self):
                super().__init__()
                self.interrupted = False

            def __setitem__(self, key, value):
                if value is None and not self.interrupted:
                    self.interrupted = True
                    raise publication_error
                super().__setitem__(key, value)

        worker = Worker(level=3, num_sub_workers=0)
        worker._worker = cast(Any, object())
        worker._domain_free_results = cast(Any, InterruptingResults())
        handle = worker_mod.CommDomainHandle(
            name="domain",
            workers=(),
            contexts={},
            allocation_id=7,
            _release_fn=lambda released: None,
        )
        backend_calls = []
        worker._release_domain_claimed = cast(Any, lambda claimed: backend_calls.append(claimed.allocation_id))

        with pytest.raises(MemoryError) as caught:
            worker._release_domain_now(handle)

        assert caught.value is publication_error
        assert backend_calls == [7]
        assert isinstance(worker._domain_free_results[7], RuntimeError)

        with pytest.raises(RuntimeError, match="refusing to replay"):
            worker._release_domain_now(handle)
        assert backend_calls == [7]

    def test_domain_released_while_its_run_retires_is_still_freed(self):
        """A release in flight across retirement must not be stranded.

        The window the fence has to close: the deferred queue has already been
        drained for the last time, so a handle appended after that is reachable
        from nothing — the queue is never read again, and the release itself
        popped the handle from ``_live_domains``, blinding ``close()``'s sweep.

        Forced deterministically by parking the fence in its live-domain sweep,
        which holds no lock, while the releasing thread runs to completion.
        """
        worker = Worker(level=3, num_sub_workers=0)
        worker._worker = cast(Any, object())
        worker._orch = cast(Any, type("FakeOrch", (), {"_release_run": lambda self, run_id: None})())

        freed: list[str] = []

        def release_now(handle):
            freed.append(handle.name)
            if worker._live_domains.get(handle.name) is handle:
                worker._live_domains.pop(handle.name)

        worker._release_domain_now = cast(Any, release_now)

        resources = worker_mod._RunResources()

        def domain(name, allocation_id):
            return worker_mod.CommDomainHandle(
                name=name,
                workers=(),
                contexts={},
                allocation_id=allocation_id,
                _release_fn=lambda released, owner=resources: worker._release_domain_handle(released, owner),
            )

        raced = domain("raced", 1)
        # A second domain keeps the sweep branch reachable so the fence parks there.
        swept = domain("swept", 2)
        for handle in (raced, swept):
            resources.live_domains[handle.name] = handle
            worker._live_domains[handle.name] = handle

        in_sweep = threading.Event()
        release_done = threading.Event()
        real_sweep = worker._release_all_live_domains

        def parked_sweep(res=None):
            in_sweep.set()
            assert release_done.wait(5.0), "releasing thread did not finish"
            real_sweep(res)

        worker._release_all_live_domains = cast(Any, parked_sweep)

        def do_release():
            assert in_sweep.wait(5.0), "fence never reached its sweep"
            raced.release()
            release_done.set()

        releaser = threading.Thread(target=do_release)
        releaser.start()
        try:
            handle = RunHandle(worker, 1, (), resources)
            worker._accepted_run_handles.add(handle)
            assert worker._finalize_run_handle(handle, 1, None) is None
        finally:
            releaser.join(5.0)

        assert not releaser.is_alive()
        assert raced.freed, "a release racing retirement was stranded on a drained queue"
        assert swept.freed
        assert resources.pending_release_domains == []
        assert freed.count("raced") == 1, f"raced domain freed more than once: {freed}"

    @staticmethod
    def _gated_domain_worker(worker, target_id, outcome=None):
        """Park `worker`'s backend release for `target_id`; optionally raise."""
        entered: list[int] = []
        in_backend = threading.Event()
        let_go = threading.Event()

        def gated(handle):
            entered.append(handle.allocation_id)
            if handle.allocation_id == target_id:
                in_backend.set()
                let_go.wait(10.0)
                if outcome is not None:
                    raise outcome

        worker._release_domain_claimed = cast(Any, gated)
        return entered, in_backend, let_go

    def _retired_domain(self, worker, resources, name, allocation_id):
        handle = worker_mod.CommDomainHandle(
            name=name,
            workers=(),
            contexts={},
            allocation_id=allocation_id,
            _release_fn=lambda released, owner=resources: worker._release_domain_handle(released, owner),
        )
        worker._live_domains[name] = handle
        return handle

    def test_sweep_and_post_fence_release_free_a_domain_once(self):
        """Two paths reaching one handle must not both drive the backend free.

        The dangerous order is release-then-sweep: ``release()`` wins the
        ``_released`` flag, so the sweep — still holding the handle in the
        snapshot it took earlier — skips setting that flag and goes straight to
        the backend call the release is already making. (Sweep-then-release is
        already safe: the sweep sets ``_released`` before freeing, which makes
        the later ``release()`` a no-op.)
        """
        worker = Worker(level=3, num_sub_workers=0)
        worker._worker = cast(Any, object())
        entered, in_backend, let_go = self._gated_domain_worker(worker, target_id=7)

        resources = worker_mod._RunResources()
        resources.retired = True  # the owning run's fence has already passed
        contested = self._retired_domain(worker, resources, "contested", 7)

        releaser = threading.Thread(target=contested.release)
        releaser.start()
        try:
            assert in_backend.wait(5.0), "release never reached the backend"
            # The sweep's snapshot still holds `contested`; it must not free it
            # a second time.
            sweeper = threading.Thread(target=worker._release_all_live_domains)
            sweeper.start()
            let_go.set()
            sweeper.join(5.0)
            assert not sweeper.is_alive()
        finally:
            let_go.set()
            releaser.join(5.0)

        assert entered.count(7) == 1, f"allocation 7 was released {entered.count(7)} times"
        assert contested.freed

    def test_second_domain_release_waits_for_the_first(self):
        """A second caller blocks until the owner's backend call returns.

        Returning early would let the caller mark the handle freed, drop it
        from ``_live_domains`` and — on the ``close()`` path — tear down the
        mailboxes the in-flight release is still using, which is exactly what
        ``close()`` orders its domain sweep before ``_worker.close()`` to avoid.
        """
        worker = Worker(level=3, num_sub_workers=0)
        worker._worker = cast(Any, object())
        entered, in_backend, let_go = self._gated_domain_worker(worker, target_id=7)

        resources = worker_mod._RunResources()
        contested = self._retired_domain(worker, resources, "contested", 7)

        owner = threading.Thread(target=worker._release_domain_now, args=(contested,))
        second_done = threading.Event()

        def second_caller():
            worker._release_domain_now(contested)
            second_done.set()

        second = threading.Thread(target=second_caller)
        owner.start()
        try:
            assert in_backend.wait(5.0), "owner never reached the backend"
            assert not contested.freed, "freed must stay false while the backend call is in flight"
            second.start()
            assert not second_done.wait(0.5), "the second caller returned while the owner was still releasing"
            assert not contested.freed
        finally:
            let_go.set()
            owner.join(5.0)
            second.join(5.0)

        assert second_done.is_set()
        assert entered.count(7) == 1, f"allocation 7 reached the backend {entered.count(7)} times"

    def test_failed_domain_release_is_replayed_to_a_second_caller(self):
        """The owner's failure reaches every later caller, so no path reports
        success for an allocation whose backend release did not happen.

        ``_release_all_live_domains`` keeps an un-freed handle in
        ``_live_domains`` precisely so ``close()`` reports it as a residual; a
        second caller that returned success would erase that.
        """
        worker = Worker(level=3, num_sub_workers=0)
        worker._worker = cast(Any, object())
        boom = RuntimeError("backend release failed")
        _entered, in_backend, let_go = self._gated_domain_worker(worker, target_id=7, outcome=boom)
        let_go.set()

        resources = worker_mod._RunResources()
        contested = self._retired_domain(worker, resources, "contested", 7)

        with pytest.raises(RuntimeError, match="backend release failed"):
            worker._release_domain_now(contested)
        assert in_backend.is_set()
        assert not contested.freed

        # The sweep is the second caller: it must see the failure, keep the
        # handle, leave `freed` false, and report the failure to its own caller
        # rather than returning as if the domain were reclaimed.
        with pytest.raises(RuntimeError, match="backend release failed"):
            worker._release_all_live_domains()
        assert not contested.freed, "a failed release must not be reported as freed"
        assert "contested" in worker._live_domains, "a failed release must stay a detectable residual"

    def test_allocate_domain_outside_graph_construction_is_rejected(self):
        worker = Worker(level=3, num_sub_workers=0)
        worker._worker = cast(Any, object())
        assert worker._building_run_resources is None

        with pytest.raises(RuntimeError, match="graph is being built"):
            worker._allocate_domain(name="d", workers=(0,), window_size=4096, buffers=[])


# ---------------------------------------------------------------------------
# Test: conditional serial degradation
#
# The whole-run FIFO orders tasks. It cannot order a run's *cleanup*, which
# happens after the native fence and reaches a child through mailbox control
# rather than a TaskSlot. A run that acquires device-touching cleanup therefore
# degrades this worker to depth one for exactly that run; runs that only
# dispatch tasks keep the full pipeline depth.
# ---------------------------------------------------------------------------


class TestOrderedCleanupDegradation:
    @staticmethod
    def _worker():
        worker = Worker(level=3, num_sub_workers=0)
        worker._worker = cast(Any, object())
        return worker

    def _accepted(self, worker, *, bears_cleanup: bool):
        resources = worker_mod._RunResources()
        resources.requires_ordered_cleanup = bears_cleanup
        handle = RunHandle(worker, 1, (), resources)
        worker._accepted_run_handles.add(handle)
        return handle, resources

    def test_only_device_touching_resources_make_a_run_cleanup_bearing(self):
        """The sticky flag is set where teardown reaches a child, and nowhere else.

        Marking every control-using run would drag a run that merely mallocs
        down to depth one, which is the overlap the pipeline exists to buy.
        """
        worker = self._worker()

        # A CommDomain: its release drives CTRL_RELEASE_DOMAIN on every member.
        resources = worker_mod._RunResources()
        assert not resources.requires_ordered_cleanup
        handle = worker_mod.CommDomainHandle(
            name="d", workers=(), contexts={}, allocation_id=1, _release_fn=lambda released: None
        )
        resources.live_domains[handle.name] = handle
        resources.requires_ordered_cleanup = True  # set by _allocate_domain at this point
        assert resources.requires_ordered_cleanup

        # A remote slot reference: releasing it is an RPC to the owning worker.
        # `create_worker_chip_queue` is not a fourth set-point — it builds its region
        # through `create_worker_chip_region`, and inherits the flag from there.
        remote = worker_mod._RunResources()
        worker._building_run_resources = remote
        ref = cast(Any, object())
        worker._adopt_remote_slot_refs([ref])
        assert remote.requires_ordered_cleanup
        assert remote.remote_slot_refs == [ref]

        # Adopting nothing is not an acquisition.
        empty = worker_mod._RunResources()
        worker._building_run_resources = empty
        worker._adopt_remote_slot_refs([])
        assert not empty.requires_ordered_cleanup

    def test_a_task_only_run_does_not_block_the_next_submission(self):
        worker = self._worker()
        self._accepted(worker, bears_cleanup=False)
        assert worker._cleanup_bearing_predecessor() is None

    def test_a_cleanup_bearing_run_blocks_until_its_cleanup_is_published(self):
        worker = self._worker()
        handle, _ = self._accepted(worker, bears_cleanup=True)
        assert worker._cleanup_bearing_predecessor() is handle

        handle._cleanup_published = True
        assert worker._cleanup_bearing_predecessor() is None

    def test_a_fired_native_fence_does_not_release_the_successor(self):
        """`done` is the device draining; it is not the cleanup boundary.

        A successor keyed on the native answer would be admitted while the
        CommDomain / L3-L2 / remote-slot teardown is still outstanding.
        """
        worker = self._worker()
        handle, _ = self._accepted(worker, bears_cleanup=True)
        with handle._cv:
            handle._terminal = True
        assert handle.done
        assert not handle._cleanup_published

        assert worker._cleanup_bearing_predecessor() is handle

    def test_a_failed_task_does_not_poison_the_worker(self):
        """A kernel that failed says nothing about whether cleanup succeeded.

        Merging the two would shut a worker permanently for an ordinary task
        error, and there is no reopening it.
        """
        worker = self._worker()
        handle, _ = self._accepted(worker, bears_cleanup=True)
        worker._orch = cast(Any, _StubOrch())

        native = RuntimeError("kernel failed")
        assert worker._finalize_run_handle(handle, 1, native) is native
        assert worker._ordered_cleanup_error is None
        assert handle._cleanup_published
        assert handle not in worker._accepted_run_handles
        worker._require_no_ordered_cleanup_failure("submit")  # admits

    def test_a_failed_cleanup_poisons_the_worker_and_refuses_admission(self):
        worker = self._worker()
        handle, resources = self._accepted(worker, bears_cleanup=True)
        worker._orch = cast(Any, _StubOrch())
        boom = RuntimeError("domain release failed")

        def failing_release(res):
            raise boom

        worker._release_all_live_domains = cast(Any, failing_release)
        resources.live_domains["d"] = cast(Any, object())

        assert worker._finalize_run_handle(handle, 1, None) is boom
        assert worker._ordered_cleanup_error is boom
        with pytest.raises(RuntimeError, match="no further work is admitted"):
            worker._require_no_ordered_cleanup_failure("submit")

    def test_the_poison_is_published_before_the_handle_is_dropped(self):
        """Ordering, not just outcome.

        A submitter that saw neither — handle already gone, poison not yet set
        — would find no cleanup-bearing predecessor and no reason to refuse,
        and would be admitted on top of unreclaimed device state.
        """
        worker = self._worker()
        handle, resources = self._accepted(worker, bears_cleanup=True)
        worker._orch = cast(Any, _StubOrch())
        observed: list[tuple[bool, bool]] = []

        def watching_discard(item):
            observed.append((worker._ordered_cleanup_error is not None, item._cleanup_published))

        worker._accepted_run_handles = cast(Any, _WatchedSet(worker._accepted_run_handles, watching_discard))
        resources.live_domains["d"] = cast(Any, object())
        worker._release_all_live_domains = cast(Any, _raiser(RuntimeError("cleanup failed")))

        worker._finalize_run_handle(handle, 1, None)
        assert observed == [(True, True)], "the handle was dropped before its poison was visible"

    def test_a_submission_waiting_on_cleanup_cannot_bypass_its_poison(self):
        """The lease-time check cannot have seen a poison this call just waited on.

        Two submissions can both pass native admission before either reaches
        the serializer, so the refusal is re-tested under `_submit_mu`, after
        the handoff wait.
        """
        worker = self._worker()
        handle, _ = self._accepted(worker, bears_cleanup=True)
        boom = RuntimeError("cleanup failed")

        def handoff():
            # Whoever ran the cleanup published the poison before dropping the
            # handle; this waiter only has to let the check see it.
            worker._ordered_cleanup_error = boom
            handle._cleanup_published = True
            worker._accepted_run_handles.discard(handle)

        handle._wait_for_handoff = cast(Any, handoff)
        worker._submit_l3_locked = cast(Any, _raiser(AssertionError("admitted on top of a failed cleanup")))

        with pytest.raises(RuntimeError, match="no further work is admitted") as excinfo:
            worker._submit_locked(lambda *a: None, None, None)
        assert excinfo.value.__cause__ is boom


class TestDirectControlOrdering:
    """`malloc` / `free` / `copy_*` / `committed_device_memory` / domain and
    region creation / every `remote_*` buffer call reach a child directly
    rather than through a TaskSlot, so the ready-queue FIFO does not order
    them. Two boundaries close that: control issued inside a run waits for that
    run to hold the FIFO head, and control issued outside every run reserves
    the worker for the whole call.
    """

    @staticmethod
    def _orch(worker):
        from unittest.mock import MagicMock  # noqa: PLC0415

        from simpler.orchestrator import Orchestrator  # noqa: PLC0415

        native = MagicMock()
        native.malloc.return_value = 0x1000
        return Orchestrator(native, worker), native

    def test_control_inside_a_run_waits_for_that_run_to_hold_the_fifo_head(self):
        worker = Worker(level=3, num_sub_workers=0)
        worker._worker = cast(Any, object())
        orch, native = self._orch(worker)

        with orch_mod._callback_run(7, worker):
            orch.malloc(0, 64)
        native.await_run_admission.assert_called_once_with(7)

    def test_control_after_a_task_submission_in_the_same_run_is_refused(self):
        """A task travels the ready queue and control travels the mailbox.

        Their order is undefined, and two such pairs on different chips can
        each hold the mailbox the other is waiting for.
        """
        worker = Worker(level=3, num_sub_workers=0)
        worker._worker = cast(Any, object())
        orch, native = self._orch(worker)

        with orch_mod._callback_run(7, worker):
            orch.malloc(0, 64)  # before any submit: fine
            orch_mod._admit_task_submission(worker)
            with pytest.raises(RuntimeError, match="cannot follow a task submission"):
                orch.malloc(0, 64)
        assert native.malloc.call_count == 1

    def test_caught_submit_validation_failure_does_not_forbid_later_control(self):
        worker = Worker(level=3, num_sub_workers=0)
        worker._worker = cast(Any, object())
        orch, native = self._orch(worker)

        with orch_mod._callback_run(7, worker):
            with pytest.raises(TypeError, match="expects a CallableHandle"):
                orch.submit_sub(object())
            orch.malloc(0, 64)

        native.submit_sub.assert_not_called()
        native.malloc.assert_called_once()

    def test_native_submit_attempt_forbids_later_control_even_when_it_raises(self, monkeypatch):
        worker = Worker(level=3, num_sub_workers=0)
        worker._worker = cast(Any, object())
        orch, native = self._orch(worker)
        monkeypatch.setattr(
            orch_mod,
            "_require_handle",
            lambda *_args, **_kwargs: (b"digest", "python", "LOCAL_PYTHON", ()),
        )
        native.submit_sub.side_effect = RuntimeError("native submit failed")

        with orch_mod._callback_run(7, worker):
            with pytest.raises(RuntimeError, match="native submit failed"):
                orch.submit_sub(object())
            with pytest.raises(RuntimeError, match="cannot follow a task submission"):
                orch.malloc(0, 64)

        native.submit_sub.assert_called_once()
        native.malloc.assert_not_called()

    def test_each_run_starts_with_no_submissions_of_its_own(self):
        worker = Worker(level=3, num_sub_workers=0)
        worker._worker = cast(Any, object())
        orch, _native = self._orch(worker)

        with orch_mod._callback_run(7, worker):
            orch_mod._admit_task_submission(worker)
        with orch_mod._callback_run(8, worker):
            orch.malloc(0, 64)

    def test_a_nested_run_restores_its_callers_marker(self):
        """An L4 callback drives its children's runs on its own thread.

        The inner run's context is not the outer one's: it must neither inherit
        the outer submission nor erase it on the way out.
        """
        worker = Worker(level=3, num_sub_workers=0)
        worker._worker = cast(Any, object())
        orch, _native = self._orch(worker)

        with orch_mod._callback_run(7, worker):
            orch_mod._admit_task_submission(worker)
            with orch_mod._callback_run(8, worker):
                orch.malloc(0, 64)  # the inner run has submitted nothing
            with pytest.raises(RuntimeError, match="cannot follow a task submission"):
                orch.malloc(0, 64)
        assert orch_mod._callback_frames() == []

    def test_the_reservation_spans_the_call_not_just_the_check(self):
        """A sampled check leaves the command itself outside the decision.

        Between "no run is in flight" and the mailbox write, a submit can be
        admitted and dispatch a task that races it — which is the whole thing
        the check was for. The serializer submission holds is held here too.
        """
        worker = Worker(level=3, num_sub_workers=0)
        worker._worker = cast(Any, object())
        orch, _native = self._orch(worker)

        in_control = threading.Event()
        release_control = threading.Event()
        submit_entered = threading.Event()

        def _blocking_native_malloc(*_args):
            in_control.set()
            assert release_control.wait(5.0), "test never released the control call"
            return 0x1000

        _native.malloc = _blocking_native_malloc
        worker._submit_l3_locked = cast(Any, lambda *a: submit_entered.set())

        control = threading.Thread(target=lambda: orch.malloc(0, 64), daemon=True)
        control.start()
        assert in_control.wait(5.0), "the control call never reached the child"

        submitter = threading.Thread(target=lambda: worker._submit_locked(lambda *a: None, None, None), daemon=True)
        submitter.start()
        assert not submit_entered.wait(0.5), "a run was admitted while a control call was still in flight"

        release_control.set()
        control.join(5.0)
        assert submit_entered.wait(5.0), "the submission stayed blocked after the control call finished"
        submitter.join(5.0)

    def test_the_reservation_is_reentrant_within_one_thread(self):
        """One control call can be built out of others — a queue out of a region.

        The serializer it takes is not re-entrant, so the inner call has to
        join the outer reservation rather than deadlock on it.
        """
        worker = Worker(level=3, num_sub_workers=0)
        worker._worker = cast(Any, object())

        with worker._control_reservation("outer"), worker._control_reservation("inner"):
            pass
        # And the reservation is released, not leaked, on the way out.
        with worker._control_reservation("again"):
            pass

    def test_a_run_on_one_worker_does_not_admit_control_on_another(self):
        """A run id names nothing on its own — run 1 exists on every Worker.

        Inside Worker A's run, a call on Worker B is B's business: it must take
        B's own reservation, not wait on whatever B happens to call run 1, and
        certainly not skip B's admission because A has a run open.
        """
        worker_a = Worker(level=3, num_sub_workers=0)
        worker_a._worker = cast(Any, object())
        worker_b = Worker(level=3, num_sub_workers=0)
        worker_b._worker = cast(Any, object())
        orch_b, native_b = self._orch(worker_b)

        # B has its own run 1 in flight — colliding id, unrelated run.
        b_handle = RunHandle(worker_b, 1, (), worker_mod._RunResources())
        worker_b._accepted_run_handles.add(b_handle)

        with orch_mod._callback_run(1, worker_a):
            with pytest.raises(RuntimeError, match="still in flight"):
                orch_b.malloc(0, 64)
        native_b.await_run_admission.assert_not_called()

    def test_a_reservation_on_one_worker_does_not_cover_another(self):
        worker_a = Worker(level=3, num_sub_workers=0)
        worker_a._worker = cast(Any, object())
        worker_b = Worker(level=3, num_sub_workers=0)
        worker_b._worker = cast(Any, object())
        orch_b, _native_b = self._orch(worker_b)

        b_handle = RunHandle(worker_b, 1, (), worker_mod._RunResources())
        worker_b._accepted_run_handles.add(b_handle)

        with worker_a._control_reservation("Worker.malloc"):
            with pytest.raises(RuntimeError, match="still in flight"):
                orch_b.malloc(0, 64)

    def test_a_nested_worker_callback_keeps_its_callers_ordering(self):
        """An L4 callback drives its child's run on its own thread.

        The inner frame belongs to the child, so control on the parent still
        finds the parent's frame rather than falling through to a reservation
        the parent's own open callback would deadlock on.
        """
        parent = Worker(level=4, num_sub_workers=0)
        parent._worker = cast(Any, object())
        child = Worker(level=3, num_sub_workers=0)
        child._worker = cast(Any, object())
        orch_parent, native_parent = self._orch(parent)
        orch_child, native_child = self._orch(child)

        with orch_mod._callback_run(5, parent):
            with orch_mod._callback_run(9, child):
                orch_child.malloc(0, 64)
                orch_parent.malloc(0, 64)
        native_child.await_run_admission.assert_called_once_with(9)
        native_parent.await_run_admission.assert_called_once_with(5)

    def test_run_owned_control_rechecks_the_sticky_poison(self):
        """A callback can catch a rollback failure and carry on.

        Holding the FIFO head says nothing about whether this worker still has
        reclaimable device state, so the refusal is re-read on every call.
        """
        worker = Worker(level=3, num_sub_workers=0)
        worker._worker = cast(Any, object())
        orch, native = self._orch(worker)

        with orch_mod._callback_run(7, worker):
            orch.malloc(0, 64)
            worker._ordered_cleanup_error = RuntimeError("region rollback leaked")
            with pytest.raises(RuntimeError, match="no further work is admitted"):
                orch.malloc(0, 64)
        assert native.malloc.call_count == 1

    def test_task_submission_rechecks_the_sticky_poison(self):
        """Same reason control does: the callback may have caught the failure.

        A run whose own graph construction leaked device state must not keep
        putting work behind it.
        """
        worker = Worker(level=3, num_sub_workers=0)
        worker._worker = cast(Any, object())
        worker._ordered_cleanup_error = RuntimeError("region rollback leaked")

        with orch_mod._callback_run(7, worker):
            with pytest.raises(RuntimeError, match="no further work is admitted"):
                orch_mod._admit_task_submission(worker)

    def test_owner_less_control_is_refused_after_a_cleanup_failure(self):
        worker = Worker(level=3, num_sub_workers=0)
        worker._worker = cast(Any, object())
        orch, _native = self._orch(worker)

        worker._ordered_cleanup_error = RuntimeError("domain release failed")
        with pytest.raises(RuntimeError, match="no further work is admitted"):
            orch.malloc(0, 64)

    def test_a_mailbox_query_takes_the_same_ordering_as_a_command(self):
        """`committed_device_memory` is a read, but it travels the same mailbox.

        Answered from behind a run that is still allocating, the number
        describes neither the state before nor the state after.
        """
        worker = Worker(level=3, num_sub_workers=0)
        worker._worker = cast(Any, object())
        orch, native = self._orch(worker)
        native.committed_device_memory.return_value = 4096

        with orch_mod._callback_run(11, worker):
            assert orch.committed_device_memory(0) == 4096
        native.await_run_admission.assert_called_once_with(11)

        handle = RunHandle(worker, 1, (), worker_mod._RunResources())
        worker._accepted_run_handles.add(handle)
        with pytest.raises(RuntimeError, match="still in flight"):
            orch.committed_device_memory(0)

    def test_owner_less_control_is_refused_while_a_run_is_in_flight(self):
        worker = Worker(level=3, num_sub_workers=0)
        worker._worker = cast(Any, object())
        orch, native = self._orch(worker)

        orch.malloc(0, 64)  # quiescent: admitted
        handle = RunHandle(worker, 1, (), worker_mod._RunResources())
        worker._accepted_run_handles.add(handle)

        with pytest.raises(RuntimeError, match="still in flight"):
            orch.malloc(0, 64)
        native.await_run_admission.assert_not_called()

        # A run whose cleanup has been published is no longer in flight, even
        # though nothing removed the handle here.
        handle._cleanup_published = True
        orch.malloc(0, 64)
        assert native.malloc.call_count == 2


class TestRemoteControlOrdering:
    """`remote_*` buffer commands and remote task dispatch both end up
    contending for the endpoint's command mutex, so which arrives first is the
    scheduler's choice. They take the same admission local control does.
    """

    @staticmethod
    def _worker():
        from unittest.mock import MagicMock  # noqa: PLC0415

        worker = Worker(level=4, num_sub_workers=0)
        native = MagicMock()
        native.remote_malloc.return_value = (0, 1, 1, 2, 4, 0, 0, 0)
        worker._worker = native
        worker._lifecycle = worker_mod._Lifecycle.READY
        worker._require_remote_worker_started = cast(Any, lambda _wid: None)
        native_orch = MagicMock()
        worker._orch = cast(Any, SimpleNamespace(_o=native_orch))
        return worker, native, native_orch

    def test_a_remote_command_inside_a_run_waits_for_the_fifo_head(self):
        worker, _native, native_orch = self._worker()
        with orch_mod._callback_run(3, worker):
            worker.remote_malloc(worker=0, nbytes=4)
        native_orch.await_run_admission.assert_called_once_with(3)

    def test_a_remote_command_after_a_task_submission_is_refused(self):
        worker, native, _native_orch = self._worker()
        with orch_mod._callback_run(3, worker):
            orch_mod._admit_task_submission(worker)
            with pytest.raises(RuntimeError, match="cannot follow a task submission"):
                worker.remote_malloc(worker=0, nbytes=4)

    def test_an_owner_less_remote_command_is_refused_while_a_run_is_in_flight(self):
        worker, native, _native_orch = self._worker()
        worker.remote_malloc(worker=0, nbytes=4)  # quiescent: admitted

        handle = RunHandle(worker, 1, (), worker_mod._RunResources())
        worker._accepted_run_handles.add(handle)
        with pytest.raises(RuntimeError, match="still in flight"):
            worker.remote_malloc(worker=0, nbytes=4)

    def test_a_deferred_free_sends_nothing_and_is_not_ordered(self):
        """A free behind a live slot ref only records the debt.

        Nothing reaches the owner, so there is no command for admission to
        order — and refusing it would break the ordinary shape of freeing an
        input right after the task that reads it.
        """
        worker, native, _native_orch = self._worker()
        buffer = worker.remote_malloc(worker=0, nbytes=4)
        buffer._acquire_slot_ref()

        with orch_mod._callback_run(3, worker):
            orch_mod._admit_task_submission(worker)
            worker.remote_free(buffer)

        assert buffer.released
        assert worker._pending_remote_buffer_frees == [buffer]
        native.remote_free.assert_not_called()

    def test_interrupted_import_release_attempts_the_rest_without_losing_debt(self, monkeypatch):
        worker = Worker(level=4, num_sub_workers=0)
        interrupt = KeyboardInterrupt("release import")

        def imported(buffer_id, import_id):
            owner = worker_mod.RemoteBufferHandle._from_remote_allocation(
                worker_id=0,
                buffer_id=buffer_id,
                generation=1,
                address_space=worker_mod.RemoteAddressSpace.REMOTE_DEVICE,
                nbytes=4,
            )
            owner._acquire_import_ref()
            handle = worker_mod.RemoteBufferHandle._from_imported_mapping(
                worker_id=1,
                owner_worker_id=0,
                buffer_id=buffer_id,
                generation=1,
                import_id=import_id,
                address_space=worker_mod.RemoteAddressSpace.REMOTE_WINDOW,
                nbytes=4,
                offset=0,
                owner_handle_ref=owner,
            )
            return handle, owner

        interrupted, interrupted_owner = imported(1, 11)
        released, released_owner = imported(2, 22)
        blocked, blocked_owner = imported(3, 33)
        blocked._acquire_slot_ref()
        worker._pending_remote_import_releases.extend([interrupted, released, blocked])
        calls = []

        def release_import(handle):
            calls.append(handle.import_id)
            if handle is interrupted:
                raise interrupt

        monkeypatch.setattr(worker, "_send_remote_release_import", release_import)

        with pytest.raises(KeyboardInterrupt) as caught:
            worker._flush_pending_remote_frees()

        assert caught.value is interrupt
        assert calls == [11, 22]
        assert worker._pending_remote_import_releases == [interrupted, blocked]
        assert interrupted._owner_handle_ref is interrupted_owner
        assert interrupted_owner._live_import_refs == 1
        assert released._owner_handle_ref is None
        assert released_owner._live_import_refs == 0
        assert blocked._owner_handle_ref is blocked_owner
        assert blocked_owner._live_import_refs == 1

    def test_post_rpc_import_retirement_interrupt_never_replays_the_debt(self, monkeypatch):
        worker = Worker(level=4, num_sub_workers=0)
        interrupt = KeyboardInterrupt("owner reference retirement")

        class Owner:
            def __init__(self, *, interrupts=False):
                self.interrupts = interrupts
                self.release_calls = 0

            def _release_import_ref(self):
                self.release_calls += 1
                if self.interrupts:
                    raise interrupt

        def imported(buffer_id, import_id, owner):
            return worker_mod.RemoteBufferHandle._from_imported_mapping(
                worker_id=1,
                owner_worker_id=0,
                buffer_id=buffer_id,
                generation=1,
                import_id=import_id,
                address_space=worker_mod.RemoteAddressSpace.REMOTE_WINDOW,
                nbytes=4,
                offset=0,
                owner_handle_ref=cast(Any, owner),
            )

        interrupted_owner = Owner(interrupts=True)
        released_owner = Owner()
        interrupted = imported(1, 11, interrupted_owner)
        released = imported(2, 22, released_owner)
        worker._pending_remote_import_releases.extend([interrupted, released])
        rpc_calls = []
        monkeypatch.setattr(worker, "_send_remote_release_import", lambda handle: rpc_calls.append(handle.import_id))

        with pytest.raises(KeyboardInterrupt) as caught:
            worker._flush_pending_remote_frees()

        assert caught.value is interrupt
        assert rpc_calls == [11, 22]
        assert interrupted_owner.release_calls == 1
        assert released_owner.release_calls == 1
        assert worker._pending_remote_import_releases == [interrupted]

        with pytest.raises(KeyboardInterrupt) as repeated:
            worker._flush_pending_remote_frees()

        assert repeated.value is interrupt
        assert rpc_calls == [11, 22]
        assert interrupted_owner.release_calls == 1
        assert released_owner.release_calls == 1
        assert worker._pending_remote_import_releases == [interrupted]

    def test_interrupted_owner_free_attempts_the_rest_without_losing_debt(self, monkeypatch):
        worker = Worker(level=4, num_sub_workers=0)
        interrupt = SystemExit("remote free")

        def owner(buffer_id):
            return worker_mod.RemoteBufferHandle._from_remote_allocation(
                worker_id=0,
                buffer_id=buffer_id,
                generation=1,
                address_space=worker_mod.RemoteAddressSpace.REMOTE_DEVICE,
                nbytes=4,
            )

        interrupted = owner(1)
        freed = owner(2)
        blocked = owner(3)
        blocked._acquire_import_ref()
        worker._pending_remote_buffer_frees.extend([interrupted, freed, blocked])
        calls = []

        def remote_free(handle):
            calls.append(handle._buffer_id)
            if handle is interrupted:
                raise interrupt

        monkeypatch.setattr(worker, "_send_remote_free", remote_free)

        with pytest.raises(SystemExit) as caught:
            worker._flush_pending_remote_frees()

        assert caught.value is interrupt
        assert calls == [1, 2]
        assert worker._pending_remote_buffer_frees == [interrupted, blocked]


class TestUnreclaimedDeviceStateIsNeverSilent:
    """Every path that leaves a resource on a chip either hands it to the run's
    cleanup or refuses further work. Reporting a plain error over it lets the
    next run start on top of state nothing can name.
    """

    @staticmethod
    def _worker():
        worker = Worker(level=3, num_sub_workers=0)
        worker._worker = cast(Any, object())
        return worker

    def test_a_partial_domain_allocation_keeps_its_original_release_ranks(self, monkeypatch):
        """Two chips of three committed a window and no handle exists.

        The release has to reach exactly the two that allocated: driving it at
        the third would fail on a debt it does not hold, and poison the worker
        for a partial failure that was handled correctly.
        """
        worker = self._worker()
        worker._config = {"device_ids": [0, 1, 2]}
        resources = worker_mod._RunResources()
        worker._building_run_resources = resources
        monkeypatch.setattr(worker, "_ensure_comm_base", lambda: None)

        def partial_failure(*, reply_shms, **_kwargs):
            assert reply_shms is not None
            for chip_idx in (0, 2):
                reply_buf = reply_shms[chip_idx].buf
                assert reply_buf is not None
                struct.pack_into("<Q", reply_buf, worker_mod._OFF_DOMAIN_REPLY_COMMITTED, 1)
            raise RuntimeError("rank 1 failed")

        monkeypatch.setattr(worker, "_dispatch_control_domain", partial_failure)
        with pytest.raises(RuntimeError, match="rank 1 failed"):
            worker._allocate_domain(name="d", workers=(0, 1, 2), window_size=64, buffers=[])

        handle = resources.live_domains["d"]
        assert handle.workers == (0, 2), "release would reach a chip that never allocated"
        assert worker._live_domains["d"] is handle
        assert resources.requires_ordered_cleanup, "a leaked window did not degrade the successor"

        release_headers = {}

        def capture_release(*, workers, request_shms, **_kwargs):
            for chip_idx in workers:
                request_buf = request_shms[chip_idx].buf
                assert request_buf is not None
                release_headers[chip_idx] = worker_mod._DOMAIN_REQ_HEADER.unpack_from(request_buf, 0)[:3]

        monkeypatch.setattr(worker, "_dispatch_control_domain", capture_release)
        worker._release_domain_claimed(handle)

        assert release_headers == {
            0: (handle.allocation_id, 3, 0),
            2: (handle.allocation_id, 3, 2),
        }

    def test_a_chip_that_committed_before_its_reply_failed_is_still_reclaimed(self):
        """The window exists before the reply is written.

        A chip whose RPC failed after `comm_alloc_domain_windows` returned is
        holding an allocation, so "which RPCs failed" is the wrong question —
        each chip publishes its own commit and the parent reads that.
        """
        reply = SharedMemory(create=True, size=worker_mod._DOMAIN_REPLY_HEADER.size)
        try:
            assert not worker_mod._domain_reply_committed(reply), "a zero-filled reply must not read as committed"
            struct.pack_into("<Q", cast(Any, reply.buf), worker_mod._OFF_DOMAIN_REPLY_COMMITTED, 1)
            assert worker_mod._domain_reply_committed(reply)
        finally:
            reply.close()
            reply.unlink()

        assert not worker_mod._domain_reply_committed(None), "a chip that never got a reply slot owes nothing"

    def test_the_child_publishes_its_commit_before_anything_that_can_fail(self):
        """A carving overflow is raised after the window is already allocated.

        The chip must have said so first, or the parent excludes it from
        cleanup and the window is leaked for the worker's lifetime.
        """
        request = SharedMemory(create=True, size=worker_mod._DOMAIN_REQ_HEADER.size + 8 + 4)
        reply = SharedMemory(create=True, size=worker_mod._DOMAIN_REPLY_HEADER.size + 8)
        try:
            req_buf = cast(Any, request.buf)
            worker_mod._DOMAIN_REQ_HEADER.pack_into(
                req_buf, 0, 7, 1, 0, 64, 1
            )  # allocation_id, rank_count, domain_rank, window_size, buffer_count
            # One buffer larger than the window: the carve raises after the
            # collective has already committed.
            struct.pack_into("<Q", req_buf, worker_mod._DOMAIN_REQ_HEADER.size, 4096)
            struct.pack_into("<I", req_buf, worker_mod._DOMAIN_REQ_HEADER.size + 8, 0)

            cw = cast(Any, SimpleNamespace(_impl=SimpleNamespace()))

            def committed(*args):
                ctypes.c_uint64.from_address(args[5]).value = 1
                return 0xC7, 0xB000

            cw._impl.comm_alloc_domain_windows = committed
            mailbox = memoryview(bytearray(MAILBOX_SIZE))
            for offset, shm_name in (
                (worker_mod._OFF_ARGS, request.name),
                (worker_mod._OFF_ARGS + worker_mod._CTRL_SHM_NAME_BYTES, reply.name),
            ):
                encoded = shm_name.encode("utf-8")
                mailbox[offset : offset + len(encoded)] = encoded

            with (
                patch.object(worker_mod, "_comm_base_handle", lambda _cw: 1),
                pytest.raises(ValueError, match="overflows window_size"),
            ):
                worker_mod._handle_ctrl_alloc_domain(cw, mailbox)

            assert worker_mod._domain_reply_committed(reply), (
                "the window was allocated and the chip did not say so before failing"
            )
        finally:
            for shm in (request, reply):
                shm.close()
                shm.unlink()

    def test_native_commit_publication_survives_result_conversion_failure(self):
        request = SharedMemory(create=True, size=worker_mod._DOMAIN_REQ_HEADER.size + 4)
        reply = SharedMemory(create=True, size=worker_mod._DOMAIN_REPLY_HEADER.size)
        try:
            req_buf = cast(Any, request.buf)
            worker_mod._DOMAIN_REQ_HEADER.pack_into(req_buf, 0, 7, 1, 0, 64, 0)
            struct.pack_into("<I", req_buf, worker_mod._DOMAIN_REQ_HEADER.size, 0)

            def committed_then_failed(*args):
                commit_address = args[5]
                ctypes.c_uint64.from_address(commit_address).value = 1
                raise MemoryError("tuple conversion failed")

            cw = cast(Any, SimpleNamespace(_impl=SimpleNamespace()))
            cw._impl.comm_alloc_domain_windows = committed_then_failed
            mailbox = memoryview(bytearray(MAILBOX_SIZE))
            for offset, shm_name in (
                (worker_mod._OFF_ARGS, request.name),
                (worker_mod._OFF_ARGS + worker_mod._CTRL_SHM_NAME_BYTES, reply.name),
            ):
                encoded = shm_name.encode("utf-8")
                mailbox[offset : offset + len(encoded)] = encoded

            with (
                patch.object(worker_mod, "_comm_base_handle", lambda _cw: 1),
                pytest.raises(MemoryError, match="tuple conversion failed"),
            ):
                worker_mod._handle_ctrl_alloc_domain(cw, mailbox)

            assert worker_mod._domain_reply_committed(reply)
        finally:
            for shm in (request, reply):
                shm.close()
                shm.unlink()

    def test_a_fully_failed_domain_allocation_owes_nothing(self, monkeypatch):
        worker = self._worker()
        worker._config = {"device_ids": [0, 1, 2]}
        resources = worker_mod._RunResources()
        worker._building_run_resources = resources
        monkeypatch.setattr(worker, "_ensure_comm_base", lambda: None)
        monkeypatch.setattr(
            worker,
            "_dispatch_control_domain",
            lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("no rank committed")),
        )

        with pytest.raises(RuntimeError, match="no rank committed"):
            worker._allocate_domain(name="d", workers=(0, 1, 2), window_size=64, buffers=[])

        assert "d" not in resources.live_domains
        assert "d" not in worker._live_domains
        assert not resources.requires_ordered_cleanup

    def test_domain_fanout_launches_later_ranks_after_a_start_boundary_interrupt(self, monkeypatch):
        worker = self._worker()
        calls: list[int] = []
        worker._worker = cast(
            Any,
            SimpleNamespace(control_release_domain=lambda chip_idx, _request_name: calls.append(chip_idx)),
        )
        requests = {chip_idx: SharedMemory(create=True, size=worker_mod._DOMAIN_REQ_HEADER.size) for chip_idx in (0, 1)}
        real_thread = threading.Thread

        class StartBoundaryInterrupt(real_thread):
            armed = True

            def start(self):
                super().start()
                if StartBoundaryInterrupt.armed:
                    StartBoundaryInterrupt.armed = False
                    raise KeyboardInterrupt

        monkeypatch.setattr(worker_mod.threading, "Thread", StartBoundaryInterrupt)
        try:
            with pytest.raises(KeyboardInterrupt):
                worker._dispatch_control_domain(
                    workers=(0, 1),
                    request_shms=requests,
                    reply_shms=None,
                    op="release",
                    allocation_id=7,
                )
            assert sorted(calls) == [0, 1]
        finally:
            for request in requests.values():
                request.close()
                request.unlink()

    def test_fanout_defers_an_interrupt_after_a_confirmed_launch_until_every_target_finishes(self):
        rank_zero_entered = threading.Event()
        rank_one_completed = threading.Event()
        allow_rank_zero = threading.Event()
        runner_completed = threading.Event()
        calls: list[int] = []
        first_interrupt = KeyboardInterrupt("after confirmed launch")
        raised: list[BaseException] = []
        hook_calls: list[int] = []

        def target(rank: int) -> None:
            if rank == 0:
                rank_zero_entered.set()
                assert allow_rank_zero.wait(5.0)
            calls.append(rank)
            if rank == 1:
                rank_one_completed.set()

        def interrupt_after_first_start(rank: int) -> None:
            hook_calls.append(rank)
            if rank == 0:
                raise first_interrupt

        def run_fanout() -> None:
            try:
                worker_mod._start_and_join_threads(
                    (0, 1),
                    target,
                    name_prefix="test_post_launch_",
                    _after_start=interrupt_after_first_start,
                )
            except BaseException as exc:  # noqa: BLE001
                raised.append(exc)
            finally:
                runner_completed.set()

        runner = threading.Thread(target=run_fanout)
        try:
            runner.start()
            assert rank_zero_entered.wait(5.0)
            assert rank_one_completed.wait(5.0), "the later rank was not launched after the boundary interrupt"
            assert not runner_completed.wait(0.1), "fanout returned while rank zero still owned caller state"
            allow_rank_zero.set()
            runner.join(5.0)

            assert not runner.is_alive()
            assert sorted(calls) == [0, 1]
            assert hook_calls == [0, 1]
            assert raised == [first_interrupt]
        finally:
            allow_rank_zero.set()
            runner.join(5.0)

    def test_fanout_defers_a_phase_advance_interrupt_until_every_target_finishes(self):
        rank_zero_entered = threading.Event()
        rank_one_completed = threading.Event()
        allow_rank_zero = threading.Event()
        runner_completed = threading.Event()
        first_interrupt = KeyboardInterrupt("phase advance")
        phase_advances: list[int] = []
        raised: list[BaseException] = []

        def target(rank: int) -> None:
            if rank == 0:
                rank_zero_entered.set()
                assert allow_rank_zero.wait(5.0)
            if rank == 1:
                rank_one_completed.set()

        def interrupt_after_launch_phase(phase: int) -> None:
            phase_advances.append(phase)
            if phase == 0:
                raise first_interrupt

        def run_fanout() -> None:
            try:
                worker_mod._start_and_join_threads(
                    (0, 1),
                    target,
                    name_prefix="test_phase_advance_",
                    _after_phase=interrupt_after_launch_phase,
                )
            except BaseException as exc:  # noqa: BLE001
                raised.append(exc)
            finally:
                runner_completed.set()

        runner = threading.Thread(target=run_fanout)
        try:
            runner.start()
            assert rank_zero_entered.wait(5.0)
            assert rank_one_completed.wait(5.0)
            assert not runner_completed.wait(0.1), "fanout returned before the launched target released caller state"
            allow_rank_zero.set()
            runner.join(5.0)

            assert not runner.is_alive()
            assert phase_advances == [0, 1, 2, 3]
            assert len(raised) == 1 and raised[0] is first_interrupt
        finally:
            allow_rank_zero.set()
            runner.join(5.0)

    def test_fanout_drain_uses_constant_stack_through_many_interruptions(self):
        first_interrupt = KeyboardInterrupt("first phase interruption")
        interruptions_remaining = 1_250

        def interrupted_phase() -> None:
            nonlocal interruptions_remaining
            if interruptions_remaining <= 0:
                return
            interruptions_remaining -= 1
            if interruptions_remaining == 1_249:
                raise first_interrupt
            raise KeyboardInterrupt(f"phase interruption {interruptions_remaining}")

        cursor = worker_mod._ThreadFanoutDrainCursor(phases=(interrupted_phase,), after_phase=None)
        fanout = worker_mod._ThreadFanout((), lambda _item: None, "test_constant_stack_", None, None)

        fanout._drain(cursor)

        assert cursor.exhausted
        assert fanout._first_error is first_interrupt

    def test_abandoned_keepalive_drain_uses_constant_stack_through_many_interruptions(self):
        first_interrupt = KeyboardInterrupt("first keepalive interruption")

        class InterruptingHandleList(list):
            def __init__(self, *items):
                super().__init__(items)
                self.interruptions_remaining = 1_250

            def __bool__(self):
                if self.interruptions_remaining <= 0:
                    return len(self) != 0
                self.interruptions_remaining -= 1
                if self.interruptions_remaining == 1_249:
                    raise first_interrupt
                raise KeyboardInterrupt(f"keepalive interruption {self.interruptions_remaining}")

        retained = SimpleNamespace(_keepalive=object())
        handles = InterruptingHandleList(retained)
        cursor = worker_mod._AbandonedRunKeepaliveCursor(cast(Any, handles))

        cursor.drain()

        assert cursor.first_error is first_interrupt
        assert retained._keepalive is None
        assert not handles

    def test_domain_fanout_cancels_and_retries_an_ambiguously_launched_thread(self, monkeypatch):
        worker = self._worker()
        rank_zero_entered = threading.Event()
        rank_one_completed = threading.Event()
        allow_rank_zero = threading.Event()
        calls: list[int] = []

        def release(chip_idx, _request_name):
            if chip_idx == 0:
                rank_zero_entered.set()
                assert allow_rank_zero.wait(5.0)
            calls.append(chip_idx)
            if chip_idx == 1:
                rank_one_completed.set()

        worker._worker = cast(Any, SimpleNamespace(control_release_domain=release))
        requests = {chip_idx: SharedMemory(create=True, size=worker_mod._DOMAIN_REQ_HEADER.size) for chip_idx in (0, 1)}
        real_thread = threading.Thread
        backing_threads: list[threading.Thread] = []

        class AmbiguousStartInterrupt(real_thread):
            armed = True

            def start(self):
                if AmbiguousStartInterrupt.armed:
                    AmbiguousStartInterrupt.armed = False
                    backing = real_thread(target=self.run)
                    backing_threads.append(backing)
                    backing.start()
                    raise KeyboardInterrupt("ambiguous start")
                return super().start()

        monkeypatch.setattr(worker_mod.threading, "Thread", AmbiguousStartInterrupt)
        raised: list[BaseException] = []
        runner_completed = threading.Event()

        def run_dispatch():
            try:
                worker._dispatch_control_domain(
                    workers=(0, 1),
                    request_shms=requests,
                    reply_shms=None,
                    op="release",
                    allocation_id=7,
                )
            except BaseException as exc:  # noqa: BLE001
                raised.append(exc)
            finally:
                runner_completed.set()

        runner = real_thread(target=run_dispatch)
        try:
            runner.start()
            assert rank_one_completed.wait(5.0), "a later collective rank was not launched"
            assert rank_zero_entered.wait(5.0), "the interrupted rank was not retried"
            assert not runner_completed.wait(0.1), "fanout returned while the retried rank still owned the request shm"
            allow_rank_zero.set()
            runner.join(5.0)

            assert not runner.is_alive()
            assert sorted(calls) == [0, 1]
            assert len(raised) == 1 and isinstance(raised[0], KeyboardInterrupt)
        finally:
            allow_rank_zero.set()
            runner.join(5.0)
            for backing in backing_threads:
                backing.join(5.0)
            for request in requests.values():
                request.close()
                request.unlink()

    def test_domain_fanout_joins_through_repeated_interruptions(self, monkeypatch):
        worker = self._worker()
        entered = threading.Event()
        allow_finish = threading.Event()
        completed: list[int] = []

        def release(chip_idx, _request_name):
            entered.set()
            assert allow_finish.wait(5.0)
            completed.append(chip_idx)

        worker._worker = cast(Any, SimpleNamespace(control_release_domain=release))
        request = SharedMemory(create=True, size=worker_mod._DOMAIN_REQ_HEADER.size)
        real_thread = threading.Thread
        instances = []

        class RepeatedJoinInterrupt(real_thread):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.join_attempts = 0
                instances.append(self)

            def join(self, *args, **kwargs):
                self.join_attempts += 1
                if self.join_attempts <= 3:
                    raise KeyboardInterrupt
                return super().join(*args, **kwargs)

        monkeypatch.setattr(worker_mod.threading, "Thread", RepeatedJoinInterrupt)
        raised: list[BaseException] = []

        def run_dispatch():
            try:
                worker._dispatch_control_domain(
                    workers=(0,),
                    request_shms={0: request},
                    reply_shms=None,
                    op="release",
                    allocation_id=7,
                )
            except BaseException as exc:  # noqa: BLE001
                raised.append(exc)

        runner = real_thread(target=run_dispatch)
        try:
            runner.start()
            assert entered.wait(5.0)
            assert runner.is_alive(), "fanout returned while its target still owned the request shm"
            allow_finish.set()
            runner.join(5.0)

            assert not runner.is_alive()
            assert completed == [0]
            assert instances[0].join_attempts == 4
            assert len(raised) == 1 and isinstance(raised[0], KeyboardInterrupt)
        finally:
            allow_finish.set()
            runner.join(5.0)
            for instance in instances:
                if instance.ident is not None:
                    real_thread.join(instance, 5.0)
            request.close()
            request.unlink()

    def test_comm_init_launches_later_ranks_after_a_start_boundary_interrupt(self, monkeypatch):
        worker = self._worker()
        worker._config = {"device_ids": [0, 1]}
        monkeypatch.setattr(worker, "_comm_plan_rootinfo_path", lambda: "/tmp/comm-rootinfo")
        calls: list[int] = []
        worker._worker = cast(
            Any,
            SimpleNamespace(control_comm_init=lambda chip_idx, _request_name: calls.append(chip_idx)),
        )
        real_thread = threading.Thread

        class StartBoundaryInterrupt(real_thread):
            armed = True

            def start(self):
                super().start()
                if StartBoundaryInterrupt.armed and not self.name.startswith("shm-"):
                    StartBoundaryInterrupt.armed = False
                    raise KeyboardInterrupt

        monkeypatch.setattr(worker_mod.threading, "Thread", StartBoundaryInterrupt)

        with pytest.raises(KeyboardInterrupt):
            worker._ensure_comm_base()

        assert sorted(calls) == [0, 1]
        assert not worker._comm_base_ready

    def test_comm_init_preserves_the_first_interrupt_through_shm_cleanup(self, monkeypatch):
        worker = self._worker()
        worker._config = {"device_ids": [0]}
        monkeypatch.setattr(worker, "_comm_plan_rootinfo_path", lambda: "/tmp/comm-rootinfo")
        worker._worker = cast(Any, SimpleNamespace(control_comm_init=lambda _chip_idx, _request_name: None))

        created: list[SharedMemory] = []
        created_fds: list[int] = []
        real_init = SharedMemory.__init__

        def tracked_shared_memory(shm, *args, **kwargs):
            real_init(shm, *args, **kwargs)
            created.append(shm)
            created_fds.append(shm._fd)

        monkeypatch.setattr(SharedMemory, "__init__", tracked_shared_memory)
        first_interrupt = KeyboardInterrupt("start boundary")
        real_thread = threading.Thread

        class StartBoundaryInterrupt(real_thread):
            armed = True

            def start(self):
                super().start()
                if StartBoundaryInterrupt.armed:
                    StartBoundaryInterrupt.armed = False
                    raise first_interrupt

        monkeypatch.setattr(worker_mod.threading, "Thread", StartBoundaryInterrupt)
        real_close = worker_mod.os.close
        close_interrupted = False

        def interrupt_first_close(fd):
            nonlocal close_interrupted
            if created_fds and fd == created_fds[0] and not close_interrupted:
                close_interrupted = True
                raise SystemExit("shm close")
            return real_close(fd)

        monkeypatch.setattr(worker_mod.os, "close", interrupt_first_close)
        try:
            with pytest.raises(KeyboardInterrupt) as caught:
                worker._ensure_comm_base()
            assert caught.value is first_interrupt
            assert close_interrupted
        finally:
            monkeypatch.setattr(worker_mod.os, "close", real_close)
            for shm in created:
                try:
                    shm.close()
                    shm.unlink()
                except FileNotFoundError:
                    pass

    def test_shm_cleanup_drains_later_owners_after_a_traversal_interrupt(self, monkeypatch):
        owner = worker_mod._SharedMemoryOwner(2)
        shms = [owner.create(1), owner.create(1)]
        names = [shm.name for shm in shms]
        first_interrupt = KeyboardInterrupt("between shm owners")
        real_unlink = SharedMemory.unlink
        interrupted = False

        def interrupt_after_first_unlink(shm):
            nonlocal interrupted
            real_unlink(shm)
            if shm is shms[0] and not interrupted:
                interrupted = True
                raise first_interrupt

        monkeypatch.setattr(SharedMemory, "unlink", interrupt_after_first_unlink)

        try:
            cleanup_error = worker_mod._close_unlink_shms(owner)

            assert cleanup_error is first_interrupt
            for name in names:
                with pytest.raises(FileNotFoundError):
                    SharedMemory(name=name)
        finally:
            for shm in shms:
                try:
                    shm.close()
                    shm.unlink()
                except FileNotFoundError:
                    pass

    def test_shm_cleanup_recovers_an_interrupt_during_cursor_setup(self, monkeypatch):
        owner = worker_mod._SharedMemoryOwner(1)
        shm = owner.create(1)
        name = shm.name
        interrupt = KeyboardInterrupt("before shm cleanup fanout")
        real_cursor = worker_mod._SharedMemoryCleanupCursor
        attempts = 0

        def interrupt_first_cursor(*args, **kwargs):
            nonlocal attempts
            cursor = real_cursor(*args, **kwargs)
            attempts += 1
            if attempts == 1:
                raise interrupt
            return cursor

        monkeypatch.setattr(worker_mod, "_SharedMemoryCleanupCursor", interrupt_first_cursor)
        try:
            cleanup_error = worker_mod._close_unlink_shms(owner)

            assert cleanup_error is interrupt
            assert owner._cleanup_step == worker_mod._SHM_CLEANUP_PHASES
            with pytest.raises(FileNotFoundError):
                SharedMemory(name=name)
        finally:
            try:
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                pass

    @pytest.mark.parametrize("operation", ("comm_init", "domain_alloc", "domain_release"))
    def test_shm_owner_retries_an_interrupt_at_cleanup_call_entry(self, monkeypatch, operation):
        worker = self._worker()
        worker._config = {"device_ids": [0]}
        resources = worker_mod._RunResources()
        worker._building_run_resources = resources
        monkeypatch.setattr(worker, "_comm_plan_rootinfo_path", lambda: "/tmp/comm-rootinfo")
        real_ensure_comm_base = worker._ensure_comm_base

        if operation == "comm_init":
            worker._worker = cast(Any, SimpleNamespace(control_comm_init=lambda *_args: None))
        elif operation == "domain_alloc":
            monkeypatch.setattr(worker, "_ensure_comm_base", lambda: None)

            def commit_domain(*, reply_shms, **_kwargs):
                assert reply_shms is not None
                reply = reply_shms[0]
                assert reply.buf is not None
                worker_mod._DOMAIN_REPLY_HEADER.pack_into(reply.buf, 0, 1, 0, 0, 0)

            monkeypatch.setattr(worker, "_dispatch_control_domain", commit_domain)
        else:
            monkeypatch.setattr(worker, "_dispatch_control_domain", lambda **_kwargs: None)

        real_init = SharedMemory.__init__
        created: list[SharedMemory] = []
        names: list[str] = []

        def track_created(shm, *args, **kwargs):
            real_init(shm, *args, **kwargs)
            if kwargs.get("create"):
                created.append(shm)
                names.append(shm.name)

        monkeypatch.setattr(SharedMemory, "__init__", track_created)
        real_cleanup = worker_mod._close_unlink_shms
        boundary_interrupt = KeyboardInterrupt(f"{operation} cleanup call entry")
        cleanup_calls = 0

        def interrupt_first_cleanup(*args, **kwargs):
            nonlocal cleanup_calls
            cleanup_calls += 1
            if cleanup_calls == 1:
                raise boundary_interrupt
            return real_cleanup(*args, **kwargs)

        monkeypatch.setattr(worker_mod, "_close_unlink_shms", interrupt_first_cleanup)
        handle = worker_mod.CommDomainHandle(
            name="d",
            workers=(0,),
            contexts={},
            allocation_id=7,
            _release_fn=lambda _released: None,
            _domain_ranks={0: 0},
        )
        try:
            with pytest.raises(KeyboardInterrupt) as caught:
                if operation == "comm_init":
                    real_ensure_comm_base()
                elif operation == "domain_alloc":
                    worker._allocate_domain(name="d", workers=(0,), window_size=64, buffers=[])
                else:
                    worker._release_domain_claimed(handle)

            assert caught.value is boundary_interrupt
            assert cleanup_calls == 2
            assert names
            monkeypatch.setattr(SharedMemory, "__init__", real_init)
            for name in names:
                with pytest.raises(FileNotFoundError):
                    SharedMemory(name=name)
        finally:
            monkeypatch.setattr(SharedMemory, "__init__", real_init)
            for shm in created:
                try:
                    shm.close()
                    shm.unlink()
                except FileNotFoundError:
                    pass

    @pytest.mark.parametrize("operation", ("comm_init", "domain_alloc", "domain_release"))
    def test_shm_lifecycle_defers_main_thread_interrupt_until_cleanup(self, monkeypatch, operation):
        worker = self._worker()
        worker._config = {"device_ids": [0]}
        resources = worker_mod._RunResources()
        worker._building_run_resources = resources
        monkeypatch.setattr(worker, "_comm_plan_rootinfo_path", lambda: "/tmp/comm-rootinfo")
        real_ensure_comm_base = worker._ensure_comm_base
        interrupt_sent = threading.Event()

        def interrupt_main() -> None:
            if interrupt_sent.is_set():
                return
            interrupt_sent.set()
            _thread.interrupt_main()
            # Keep the lifecycle active long enough for the main thread to
            # observe the signal while the owned name is still live.
            time.sleep(0.01)

        if operation == "comm_init":
            worker._worker = cast(Any, SimpleNamespace(control_comm_init=lambda *_args: interrupt_main()))
        elif operation == "domain_alloc":
            monkeypatch.setattr(worker, "_ensure_comm_base", lambda: None)

            def commit_domain(*, reply_shms, **_kwargs):
                assert reply_shms is not None
                interrupt_main()
                reply = reply_shms[0]
                assert reply.buf is not None
                worker_mod._DOMAIN_REPLY_HEADER.pack_into(reply.buf, 0, 1, 0, 0, 0)

            monkeypatch.setattr(worker, "_dispatch_control_domain", commit_domain)
        else:
            monkeypatch.setattr(worker, "_dispatch_control_domain", lambda **_kwargs: interrupt_main())

        real_init = SharedMemory.__init__
        created: list[SharedMemory] = []
        names: list[str] = []

        def track_created(shm, *args, **kwargs):
            real_init(shm, *args, **kwargs)
            if kwargs.get("create"):
                created.append(shm)
                names.append(shm.name)

        monkeypatch.setattr(SharedMemory, "__init__", track_created)
        handle = worker_mod.CommDomainHandle(
            name="d",
            workers=(0,),
            contexts={},
            allocation_id=7,
            _release_fn=lambda _released: None,
            _domain_ranks={0: 0},
        )
        try:
            with pytest.raises(KeyboardInterrupt):
                if operation == "comm_init":
                    real_ensure_comm_base()
                elif operation == "domain_alloc":
                    worker._allocate_domain(name="d", workers=(0,), window_size=64, buffers=[])
                else:
                    worker._release_domain_claimed(handle)

            assert interrupt_sent.is_set()
            assert names
            monkeypatch.setattr(SharedMemory, "__init__", real_init)
            for name in names:
                with pytest.raises(FileNotFoundError):
                    SharedMemory(name=name)
        finally:
            monkeypatch.setattr(SharedMemory, "__init__", real_init)
            for shm in created:
                try:
                    shm.close()
                    shm.unlink()
                except FileNotFoundError:
                    pass

    def test_shm_cleanup_drains_when_error_recording_is_interrupted(self, monkeypatch):
        owner = worker_mod._SharedMemoryOwner(2)
        shms = [owner.create(1), owner.create(1)]
        names = [shm.name for shm in shms]
        first_interrupt = KeyboardInterrupt("unlink return")
        recording_interrupt = SystemExit("recording cleanup error")
        real_unlink = SharedMemory.unlink
        real_remember = worker_mod._remember_cleanup_error
        unlink_interrupted = False
        recording_interrupted = False

        def interrupt_after_unlink(shm):
            nonlocal unlink_interrupted
            real_unlink(shm)
            if not unlink_interrupted:
                unlink_interrupted = True
                raise first_interrupt

        def interrupt_error_recording(first_error, cleanup_error):
            nonlocal recording_interrupted
            if not recording_interrupted:
                recording_interrupted = True
                raise recording_interrupt
            return real_remember(first_error, cleanup_error)

        monkeypatch.setattr(SharedMemory, "unlink", interrupt_after_unlink)
        monkeypatch.setattr(worker_mod, "_remember_cleanup_error", interrupt_error_recording)
        try:
            cleanup_error = worker_mod._close_unlink_shms(owner)

            assert cleanup_error is first_interrupt
            assert recording_interrupted
            for name in names:
                with pytest.raises(FileNotFoundError):
                    SharedMemory(name=name)
        finally:
            monkeypatch.setattr(SharedMemory, "unlink", real_unlink)
            monkeypatch.setattr(worker_mod, "_remember_cleanup_error", real_remember)
            for shm in shms:
                try:
                    shm.close()
                    shm.unlink()
                except FileNotFoundError:
                    pass

    def test_shm_cleanup_replays_an_ordinary_unlink_error_after_draining(self, monkeypatch):
        owner = worker_mod._SharedMemoryOwner(2)
        shms = [owner.create(1), owner.create(1)]
        names = [shm.name for shm in shms]
        first_error = OSError("resource tracker write failed")
        real_unlink = SharedMemory.unlink
        injected = False

        def unlink_then_fail(shm):
            nonlocal injected
            real_unlink(shm)
            if not injected:
                injected = True
                raise first_error

        monkeypatch.setattr(SharedMemory, "unlink", unlink_then_fail)
        try:
            cleanup_error = worker_mod._close_unlink_shms(owner)

            assert cleanup_error is first_error
            for name in names:
                with pytest.raises(FileNotFoundError):
                    SharedMemory(name=name)
        finally:
            monkeypatch.setattr(SharedMemory, "unlink", real_unlink)
            for shm in shms:
                try:
                    shm.close()
                    shm.unlink()
                except FileNotFoundError:
                    pass

    def test_shm_cleanup_drains_through_repeated_error_and_step_boundary_interrupts(self, monkeypatch):
        owner = worker_mod._SharedMemoryOwner(2)
        shms = [owner.create(1), owner.create(1)]
        names = [shm.name for shm in shms]
        first_error = OSError("unlink completion")
        real_unlink = SharedMemory.unlink
        unlink_interrupted = False
        error_boundary_interrupts = 0
        step_boundary_interrupts = 0

        def unlink_then_fail(shm):
            nonlocal unlink_interrupted
            real_unlink(shm)
            if not unlink_interrupted:
                unlink_interrupted = True
                raise first_error

        def interrupt_error_boundary():
            nonlocal error_boundary_interrupts
            if error_boundary_interrupts < 2:
                error_boundary_interrupts += 1
                raise KeyboardInterrupt(f"error boundary {error_boundary_interrupts}")

        def interrupt_step_boundary():
            nonlocal step_boundary_interrupts
            if unlink_interrupted and step_boundary_interrupts < 2:
                step_boundary_interrupts += 1
                raise SystemExit(f"step boundary {step_boundary_interrupts}")

        monkeypatch.setattr(SharedMemory, "unlink", unlink_then_fail)
        try:
            cleanup_error = worker_mod._close_unlink_shms(
                owner,
                _after_error=interrupt_error_boundary,
                _after_step=interrupt_step_boundary,
            )

            assert cleanup_error is first_error
            assert error_boundary_interrupts == 2
            assert step_boundary_interrupts == 2
            for name in names:
                with pytest.raises(FileNotFoundError):
                    SharedMemory(name=name)
        finally:
            monkeypatch.setattr(SharedMemory, "unlink", real_unlink)
            for shm in shms:
                try:
                    shm.close()
                    shm.unlink()
                except FileNotFoundError:
                    pass

    def test_shm_cleanup_protects_the_recursive_handoff_from_a_second_interrupt(self, monkeypatch):
        owner = worker_mod._SharedMemoryOwner(2)
        shms = [owner.create(1), owner.create(1)]
        names = [shm.name for shm in shms]
        first_interrupt = KeyboardInterrupt("step boundary")
        second_interrupt = SystemExit("recursive handoff")
        step_interrupted = False
        handoff_interrupted = False
        first_seen = threading.Event()
        real_thread = threading.Thread

        class HandoffInterrupt(real_thread):
            def join(self, *args, **kwargs):
                nonlocal handoff_interrupted
                if not handoff_interrupted:
                    assert first_seen.wait(5.0)
                    handoff_interrupted = True
                    raise second_interrupt
                return super().join(*args, **kwargs)

        def interrupt_step_boundary():
            nonlocal step_interrupted
            if not step_interrupted:
                step_interrupted = True
                first_seen.set()
                raise first_interrupt

        monkeypatch.setattr(worker_mod.threading, "Thread", HandoffInterrupt)
        try:
            cleanup_error = worker_mod._close_unlink_shms(owner, _after_step=interrupt_step_boundary)

            assert cleanup_error is first_interrupt
            assert handoff_interrupted
            for name in names:
                with pytest.raises(FileNotFoundError):
                    SharedMemory(name=name)
        finally:
            for shm in shms:
                try:
                    shm.close()
                    shm.unlink()
                except FileNotFoundError:
                    pass

    @pytest.mark.parametrize(
        ("operation", "interrupt_create_call"),
        (("comm_init", 1), ("domain_alloc", 1), ("domain_alloc", 2), ("domain_release", 1)),
    )
    def test_created_shm_is_owned_before_the_caller_can_register_it(
        self, monkeypatch, operation, interrupt_create_call
    ):
        worker = self._worker()
        worker._config = {"device_ids": [0]}
        resources = worker_mod._RunResources()
        worker._building_run_resources = resources
        monkeypatch.setattr(worker, "_comm_plan_rootinfo_path", lambda: "/tmp/comm-rootinfo")
        if operation == "domain_alloc":
            monkeypatch.setattr(worker, "_ensure_comm_base", lambda: None)

        real_init = SharedMemory.__init__
        create_calls = 0
        created: list[SharedMemory] = []
        interrupted_name = None
        first_interrupt = KeyboardInterrupt(f"{operation} create return")

        def interrupt_after_create(shm, *args, **kwargs):
            nonlocal create_calls, interrupted_name
            real_init(shm, *args, **kwargs)
            if kwargs.get("create"):
                create_calls += 1
                created.append(shm)
                if create_calls == interrupt_create_call:
                    interrupted_name = shm.name
                    raise first_interrupt

        monkeypatch.setattr(SharedMemory, "__init__", interrupt_after_create)
        handle = worker_mod.CommDomainHandle(
            name="d",
            workers=(0,),
            contexts={},
            allocation_id=7,
            _release_fn=lambda _released: None,
        )
        try:
            with pytest.raises(KeyboardInterrupt) as caught:
                if operation == "comm_init":
                    worker._ensure_comm_base()
                elif operation == "domain_alloc":
                    worker._allocate_domain(name="d", workers=(0,), window_size=64, buffers=[])
                else:
                    worker._release_domain_claimed(handle)

            assert caught.value is first_interrupt
            assert interrupted_name is not None
            with pytest.raises(FileNotFoundError):
                worker_mod.SharedMemory(name=interrupted_name)
        finally:
            monkeypatch.setattr(SharedMemory, "__init__", real_init)
            for shm in created:
                try:
                    shm.close()
                    shm.unlink()
                except FileNotFoundError:
                    pass

    def test_shm_owner_unlinks_a_name_interrupted_before_ftruncate(self, monkeypatch):
        owner = worker_mod._SharedMemoryOwner(1)
        first_interrupt = KeyboardInterrupt("before ftruncate")
        real_ftruncate = worker_mod.os.ftruncate

        monkeypatch.setattr(
            worker_mod.os,
            "ftruncate",
            lambda _fd, _size: (_ for _ in ()).throw(first_interrupt),
        )
        with pytest.raises(KeyboardInterrupt) as caught:
            owner.create(1)
        assert caught.value is first_interrupt

        shm = owner._slots[0].shm
        assert shm is not None
        name = shm.name
        monkeypatch.setattr(worker_mod.os, "ftruncate", real_ftruncate)

        assert worker_mod._close_unlink_shms(owner) is None
        with pytest.raises(FileNotFoundError):
            SharedMemory(name=name)

    def test_shm_owner_unlinks_a_name_created_before_handle_publication(self, monkeypatch):
        owner = worker_mod._SharedMemoryOwner(1)
        first_interrupt = KeyboardInterrupt("after shm_open")
        real_init = SharedMemory.__init__
        created = None

        def create_then_interrupt(shm, *args, **kwargs):
            nonlocal created
            real_init(shm, *args, **kwargs)
            created = shm
            raise first_interrupt

        monkeypatch.setattr(SharedMemory, "__init__", create_then_interrupt)
        try:
            with pytest.raises(KeyboardInterrupt) as caught:
                owner.create(1)
            assert caught.value is first_interrupt
            assert created is not None
            name = created.name
            monkeypatch.setattr(SharedMemory, "__init__", real_init)

            assert worker_mod._close_unlink_shms(owner) is None
            with pytest.raises(FileNotFoundError):
                SharedMemory(name=name)
        finally:
            monkeypatch.setattr(SharedMemory, "__init__", real_init)
            if created is not None:
                created.close()
                try:
                    created.unlink()
                except FileNotFoundError:
                    pass

    @pytest.mark.skipif(worker_mod.os.name != "posix", reason="requires POSIX shm_open")
    def test_shm_owner_isolates_the_shm_open_to_fd_publication_gap(self):
        owner = worker_mod._SharedMemoryOwner(1)
        instructions = list(dis.get_instructions(SharedMemory.__init__))
        fd_stores = [
            index
            for index, instruction in enumerate(instructions)
            if instruction.opname == "STORE_ATTR" and instruction.argval == "_fd"
        ]
        assert fd_stores
        gap_offset = instructions[fd_stores[-1] - 1].offset
        interrupted = False
        first_interrupt = KeyboardInterrupt("after shm_open before fd publication")

        def interrupt_fd_publication(frame, event, _arg):
            nonlocal interrupted
            if frame.f_code is SharedMemory.__init__.__code__:
                frame.f_trace_opcodes = True
                if event == "opcode" and frame.f_lasti == gap_offset:
                    interrupted = True
                    sys.settrace(None)
                    raise first_interrupt
            return interrupt_fd_publication

        create_error = None
        leaked = False
        name = None
        try:
            sys.settrace(interrupt_fd_publication)
            try:
                owner.create(1)
            except BaseException as exc:  # noqa: BLE001
                create_error = exc
            finally:
                sys.settrace(None)

            shm = owner._slots[0].shm
            assert shm is not None
            name = shm.name
            cleanup_error = worker_mod._close_unlink_shms(owner)
            try:
                getattr(shared_memory_mod, "_posixshmem").shm_unlink(name)
            except FileNotFoundError:
                pass
            else:
                leaked = True

            assert create_error is None
            assert not interrupted
            assert cleanup_error is None
            assert not leaked
        finally:
            sys.settrace(None)
            if name is not None and not leaked:
                try:
                    getattr(shared_memory_mod, "_posixshmem").shm_unlink(name)
                except FileNotFoundError:
                    pass

    def test_shm_owner_never_unlinks_a_colliding_foreign_name(self, monkeypatch):
        owner = worker_mod._SharedMemoryOwner(1)
        real_init = SharedMemory.__init__
        foreign = None
        collide = True

        def collide_once(shm, *args, **kwargs):
            nonlocal collide, foreign
            if collide:
                collide = False
                foreign = SharedMemory.__new__(SharedMemory)
                real_init(foreign, *args, **kwargs)
                raise FileExistsError(kwargs["name"])
            real_init(shm, *args, **kwargs)

        monkeypatch.setattr(SharedMemory, "__init__", collide_once)
        attached = None
        try:
            owned = owner.create(1)
            assert foreign is not None
            foreign_name = foreign.name
            assert owned.name != foreign_name

            assert worker_mod._close_unlink_shms(owner) is None
            attached = SharedMemory(name=foreign_name)
        finally:
            monkeypatch.setattr(SharedMemory, "__init__", real_init)
            if attached is not None:
                attached.close()
            if foreign is not None:
                foreign.close()
                try:
                    foreign.unlink()
                except FileNotFoundError:
                    pass

    def test_shm_owner_collision_stays_unowned_when_the_handler_is_interrupted(self, monkeypatch):
        owner = worker_mod._SharedMemoryOwner(1)
        real_init = SharedMemory.__init__
        foreign = None
        first_interrupt = KeyboardInterrupt("collision handler")

        def collide(shm, *args, **kwargs):
            nonlocal foreign
            foreign = SharedMemory.__new__(SharedMemory)
            real_init(foreign, *args, **kwargs)
            raise FileExistsError(kwargs["name"])

        source, first_line = inspect.getsourcelines(worker_mod._SharedMemoryOwner._create_in_helper)
        disarm_line = next(first_line + offset for offset, line in enumerate(source) if "slot.shm = None" in line)

        def interrupt_disarm(frame, event, _arg):
            if (
                frame.f_code is worker_mod._SharedMemoryOwner._create_in_helper.__code__
                and event == "line"
                and frame.f_lineno == disarm_line
            ):
                raise first_interrupt
            return interrupt_disarm

        monkeypatch.setattr(SharedMemory, "__init__", collide)
        attached = None
        try:
            threading.settrace(interrupt_disarm)
            with pytest.raises(KeyboardInterrupt) as caught:
                owner.create(1)
            assert caught.value is first_interrupt
            assert foreign is not None
            foreign_name = foreign.name

            monkeypatch.setattr(SharedMemory, "__init__", real_init)
            worker_mod._close_unlink_shms(owner)
            attached = SharedMemory(name=foreign_name)
        finally:
            threading.settrace(cast(Any, None))
            monkeypatch.setattr(SharedMemory, "__init__", real_init)
            if attached is not None:
                attached.close()
            if foreign is not None:
                foreign.close()
                try:
                    foreign.unlink()
                except FileNotFoundError:
                    pass

    def test_shm_cleanup_closes_fd_after_an_interrupt_returning_from_mmap_close(self):
        owner = worker_mod._SharedMemoryOwner(1)
        shm = owner.create(1)
        fd = shm._fd
        underlying_mmap = shm._mmap
        first_interrupt = KeyboardInterrupt("mmap close return")

        class InterruptingMmap:
            interrupted = False

            @property
            def closed(self):
                return underlying_mmap.closed

            def close(self):
                if underlying_mmap.closed:
                    raise AssertionError("closed mmap was closed twice")
                underlying_mmap.close()
                if not self.interrupted:
                    self.interrupted = True
                    raise first_interrupt

        shm._mmap = InterruptingMmap()
        try:
            cleanup_error = worker_mod._close_unlink_shms(owner)

            assert cleanup_error is first_interrupt
            with pytest.raises(OSError):
                worker_mod.os.fstat(fd)
        finally:
            if shm._buf is not None:
                shm._buf.release()
                shm._buf = None
            if not underlying_mmap.closed:
                underlying_mmap.close()
            try:
                worker_mod.os.close(fd)
            except OSError:
                pass
            try:
                shm.unlink()
            except FileNotFoundError:
                pass

    def test_shm_cleanup_does_not_retry_close_on_a_reused_fd(self, monkeypatch):
        owner = worker_mod._SharedMemoryOwner(1)
        shm = owner.create(1)
        fd = shm._fd
        real_close = worker_mod.os.close
        sentinel_fd = worker_mod.os.open("/dev/null", worker_mod.os.O_RDONLY)
        first_interrupt = KeyboardInterrupt("fd close return")
        injected = False
        replacement_installed = False

        def close_then_reuse(close_fd):
            nonlocal injected, replacement_installed
            if close_fd == fd and not injected:
                injected = True
                real_close(close_fd)
                worker_mod.os.dup2(sentinel_fd, close_fd)
                replacement_installed = True
                raise first_interrupt
            real_close(close_fd)

        monkeypatch.setattr(worker_mod.os, "close", close_then_reuse)
        try:
            cleanup_error = worker_mod._close_unlink_shms(owner)

            assert cleanup_error is first_interrupt
            assert replacement_installed
            worker_mod.os.fstat(fd)
        finally:
            monkeypatch.setattr(worker_mod.os, "close", real_close)
            if replacement_installed:
                try:
                    real_close(fd)
                except OSError:
                    pass
            real_close(sentinel_fd)
            try:
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                pass

    def test_shm_cleanup_does_not_retry_close_on_a_same_inode_reused_fd(self, monkeypatch):
        owner = worker_mod._SharedMemoryOwner(1)
        shm = owner.create(1)
        fd = shm._fd
        real_close = worker_mod.os.close
        duplicate_fd = worker_mod.os.dup(fd)
        first_interrupt = KeyboardInterrupt("fd close return")
        injected = False
        replacement_installed = False

        def close_then_reuse(close_fd):
            nonlocal injected, replacement_installed
            if close_fd == fd and not injected:
                injected = True
                real_close(close_fd)
                worker_mod.os.dup2(duplicate_fd, close_fd)
                replacement_installed = True
                raise first_interrupt
            real_close(close_fd)

        monkeypatch.setattr(worker_mod.os, "close", close_then_reuse)
        try:
            cleanup_error = worker_mod._close_unlink_shms(owner)

            assert cleanup_error is first_interrupt
            assert replacement_installed
            worker_mod.os.fstat(fd)
        finally:
            monkeypatch.setattr(worker_mod.os, "close", real_close)
            if replacement_installed:
                try:
                    real_close(fd)
                except OSError:
                    pass
            real_close(duplicate_fd)
            try:
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                pass

    def test_domain_release_preserves_the_dispatch_interrupt_through_shm_cleanup(self, monkeypatch):
        worker = self._worker()
        first_interrupt = KeyboardInterrupt("dispatch join")
        monkeypatch.setattr(
            worker,
            "_dispatch_control_domain",
            lambda **_kwargs: (_ for _ in ()).throw(first_interrupt),
        )
        handle = worker_mod.CommDomainHandle(
            name="d",
            workers=(0,),
            contexts={},
            allocation_id=7,
            _release_fn=lambda _released: None,
        )

        created: list[SharedMemory] = []
        created_fds: list[int] = []
        real_init = SharedMemory.__init__

        def tracked_shared_memory(shm, *args, **kwargs):
            real_init(shm, *args, **kwargs)
            created.append(shm)
            created_fds.append(shm._fd)

        monkeypatch.setattr(SharedMemory, "__init__", tracked_shared_memory)
        real_close = worker_mod.os.close
        close_interrupted = False

        def interrupt_first_close(fd):
            nonlocal close_interrupted
            if created_fds and fd == created_fds[0] and not close_interrupted:
                close_interrupted = True
                raise SystemExit("shm close")
            return real_close(fd)

        monkeypatch.setattr(worker_mod.os, "close", interrupt_first_close)
        try:
            with pytest.raises(KeyboardInterrupt) as caught:
                worker._release_domain_claimed(handle)
            assert caught.value is first_interrupt
            assert close_interrupted
        finally:
            monkeypatch.setattr(worker_mod.os, "close", real_close)
            for shm in created:
                try:
                    shm.close()
                    shm.unlink()
                except FileNotFoundError:
                    pass

    def test_committed_domain_is_owned_before_reply_shm_teardown(self, monkeypatch):
        worker = self._worker()
        worker._config = {"device_ids": [0]}
        resources = worker_mod._RunResources()
        worker._building_run_resources = resources
        monkeypatch.setattr(worker, "_ensure_comm_base", lambda: None)
        staged_shms: list[SharedMemory] = []

        def successful_dispatch(*, request_shms, reply_shms, **_kwargs):
            assert reply_shms is not None
            staged_shms.extend(request_shms.values())
            staged_shms.extend(reply_shms.values())
            reply_buf = reply_shms[0].buf
            assert reply_buf is not None
            worker_mod._DOMAIN_REPLY_HEADER.pack_into(reply_buf, 0, 1, 0xC7, 0xB000, 0)

        monkeypatch.setattr(worker, "_dispatch_control_domain", successful_dispatch)
        real_close = worker_mod.os.close
        interrupted = False
        target_fd = None

        def remember_target_fd(*, request_shms, reply_shms, **kwargs):
            nonlocal target_fd
            successful_dispatch(request_shms=request_shms, reply_shms=reply_shms, **kwargs)
            target_fd = staged_shms[0]._fd

        def interrupt_first_close(fd):
            nonlocal interrupted
            if target_fd is not None and fd == target_fd and not interrupted:
                interrupted = True
                raise KeyboardInterrupt
            return real_close(fd)

        monkeypatch.setattr(worker, "_dispatch_control_domain", remember_target_fd)
        monkeypatch.setattr(worker_mod.os, "close", interrupt_first_close)
        try:
            with pytest.raises(KeyboardInterrupt):
                worker._allocate_domain(name="d", workers=(0,), window_size=64, buffers=[])

            handle = resources.live_domains["d"]
            assert worker._live_domains["d"] is handle
            assert handle.workers == (0,)
            assert resources.requires_ordered_cleanup
            assert interrupted
        finally:
            monkeypatch.setattr(worker_mod.os, "close", real_close)
            for shm in staged_shms:
                try:
                    shm.close()
                    shm.unlink()
                except FileNotFoundError:
                    pass

    def test_an_interrupt_through_a_region_create_refuses_further_work(self):
        """The create releases the GIL, so an interrupt can land mid-flight.

        The child may still be finishing a region whose id would be written
        into a reply this frame is about to unlink — something on the chip that
        nothing here can name. An ordinary create failure is not that case: the
        child releases its own region before reporting one.
        """
        worker = self._worker()
        worker._config = {"platform": "a2a3sim"}
        worker._validate_worker_chip_id = cast(Any, lambda _wid: None)

        def _interrupted(*_args):
            raise KeyboardInterrupt

        worker._worker = cast(Any, SimpleNamespace(control_worker_chip_region_create=_interrupted))
        with pytest.raises(KeyboardInterrupt):
            worker._create_worker_chip_region(0, 4096, 64)
        assert worker._ordered_cleanup_error is not None
        with pytest.raises(RuntimeError, match="no further work is admitted"):
            worker._require_no_ordered_cleanup_failure("submit")

    def test_an_ordinary_region_create_failure_does_not_refuse_further_work(self):
        worker = self._worker()
        worker._config = {"platform": "a2a3sim"}
        worker._validate_worker_chip_id = cast(Any, lambda _wid: None)

        def _failed(*_args):
            raise RuntimeError("chip refused the region")

        worker._worker = cast(Any, SimpleNamespace(control_worker_chip_region_create=_failed))
        with pytest.raises(RuntimeError, match="chip refused the region"):
            worker._create_worker_chip_region(0, 4096, 64)
        assert worker._ordered_cleanup_error is None, "an ordinary failure must not shut the worker"

    def test_a_region_rollback_that_cannot_release_refuses_further_work(self):
        """The id was never tracked, so no later cleanup can reclaim it.

        There is no handle for a fence to fail on, which is why the refusal is
        recorded here rather than left to one.
        """
        worker = self._worker()
        assert worker._ordered_cleanup_error is None

        # The shape _create_worker_chip_region's rollback reaches: the region exists
        # on the chip and the release for it failed.
        leaked = RuntimeError("create_worker_chip_region: rollback could not release region 4")
        with worker._hierarchical_start_cv:
            worker._ordered_cleanup_error = leaked

        with pytest.raises(RuntimeError, match="no further work is admitted"):
            worker._require_no_ordered_cleanup_failure("submit")
        with pytest.raises(RuntimeError, match="no further work is admitted"):
            with worker._control_reservation("Worker.malloc"):
                pass


class _StubOrch:
    def _release_run(self, run_id: int) -> None:
        pass


class _WatchedSet(set):
    """A set that reports each discard to `on_discard` before performing it."""

    def __init__(self, source, on_discard):
        super().__init__(source)
        self._on_discard = on_discard

    def discard(self, item) -> None:
        self._on_discard(item)
        super().discard(item)


def _raiser(exc):
    def _raise(*args, **kwargs):
        raise exc

    return _raise


# ---------------------------------------------------------------------------
# Test: multiple SUB workers execute in parallel
# ---------------------------------------------------------------------------


class TestParallelSubWorkers:
    # test_parallel_wall_time was dropped: wall-clock timing assertions on
    # shared CI runners (macOS in particular) are too flaky — scheduling
    # jitter routinely pushes observed elapsed past a 0.9-factor-of-serial
    # threshold. Parallel SubWorker execution is still covered via
    # test_many_tasks_two_workers_all_complete (all tasks run) and the
    # scheduler's dispatch tests in tests/ut/cpp.
    pass


# ---------------------------------------------------------------------------
# Test: submit_* returns None at the Python facade; task slots stay internal.
# ---------------------------------------------------------------------------


class TestSubmitReturnValue:
    def test_submit_returns_none(self):
        counter_shm, counter_buf = _make_shared_counter()
        try:
            hw = Worker(level=3, num_sub_workers=1)
            handle = hw.register(lambda args: _increment_counter(counter_buf))
            hw.init()

            captured = []

            def orch(o, args, cfg):
                result = o.submit_sub(handle)
                captured.append(result)

            hw.run(orch)
            hw.close()

            assert captured == [None]
            assert _read_counter(counter_buf) == 1
        finally:
            counter_shm.close()
            counter_shm.unlink()


# ---------------------------------------------------------------------------
# Test: scope management (owned by Worker.run; user doesn't see scope_begin/end)
# ---------------------------------------------------------------------------


class TestScope:
    def test_scope_managed_by_run(self):
        counter_shm, counter_buf = _make_shared_counter()

        try:
            hw = Worker(level=3, num_sub_workers=1)
            handle = hw.register(lambda args: _increment_counter(counter_buf))
            hw.init()

            def orch(o, args, cfg):
                o.submit_sub(handle)

            hw.run(orch)
            hw.close()

            assert _read_counter(counter_buf) == 1
        finally:
            counter_shm.close()
            counter_shm.unlink()

    def test_user_nested_scope_runs_to_completion(self):
        """User opens a nested scope with ``with orch.scope():``; all tasks run."""
        counter_shm, counter_buf = _make_shared_counter()
        try:
            # Use one sub worker so the increments serialize — _increment_counter
            # is a non-atomic RMW and races across parallel SubWorker processes.
            hw = Worker(level=3, num_sub_workers=1)
            handle = hw.register(lambda args: _increment_counter(counter_buf))
            hw.init()

            def orch(o, args, cfg):
                with o.scope():
                    o.submit_sub(handle)
                    o.submit_sub(handle)
                o.submit_sub(handle)  # back on outer-scope ring

            hw.run(orch)
            hw.close()

            assert _read_counter(counter_buf) == 3
        finally:
            counter_shm.close()
            counter_shm.unlink()

    def test_user_nested_scope_binding_is_exposed(self):
        """The scope context manager and raw scope_begin / scope_end are bound."""
        from simpler.orchestrator import Orchestrator  # noqa: PLC0415

        assert hasattr(Orchestrator, "scope_begin")
        assert hasattr(Orchestrator, "scope_end")

        hw = Worker(level=3, num_sub_workers=1)
        hw.register(lambda args: None)
        hw.init()

        def orch(o, args, cfg):
            # Raw calls — match L2's pto2_scope_begin / pto2_scope_end.
            o.scope_begin()
            o.scope_end()
            # Context-manager form.
            with o.scope():
                pass
            # Mixed with submits.
            with o.scope():
                inner = o.alloc((32,), DataType.FLOAT32)
                assert inner.data != 0

        hw.run(orch)
        hw.close()

    def test_user_nested_scope_three_deep(self):
        """Three levels of nested scopes drain cleanly (no leaked refs)."""
        counter_shm, counter_buf = _make_shared_counter()
        try:
            hw = Worker(level=3, num_sub_workers=1)
            handle = hw.register(lambda args: _increment_counter(counter_buf))
            hw.init()

            def orch(o, args, cfg):
                o.submit_sub(handle)  # outer scope (ring 0)
                with o.scope():
                    o.submit_sub(handle)  # ring 1
                    with o.scope():
                        o.submit_sub(handle)  # ring 2
                        with o.scope():
                            o.submit_sub(handle)  # ring 3
                            with o.scope():
                                o.submit_sub(handle)  # clamps to ring 3

            hw.run(orch)
            hw.close()
            assert _read_counter(counter_buf) == 5
        finally:
            counter_shm.close()
            counter_shm.unlink()


# ---------------------------------------------------------------------------
# Test: orch.alloc — runtime-managed intermediate buffer lifecycle
# ---------------------------------------------------------------------------


class TestOrchAlloc:
    def test_alloc_returns_valid_tensor(self):
        """alloc returns a ChipTensor whose data ptr is non-zero and writeable."""
        captured = []

        hw = Worker(level=3, num_sub_workers=1)
        handle = hw.register(lambda args: None)  # sub callable doesn't actually read
        hw.init()

        def orch(o, args, cfg):
            inter = o.alloc((64,), DataType.FLOAT32)
            captured.append((inter.data, inter.ndims, inter.shapes[0]))

            # Tag as OUTPUT in some submit so the synthetic alloc slot has a
            # downstream consumer (otherwise scope_end consumes alone — still fine).
            sub_args = TaskArgs()
            sub_args.add_tensor(inter, TensorArgType.INPUT)
            o.submit_sub(handle, sub_args)

        hw.run(orch)
        hw.close()

        assert len(captured) == 1
        data_ptr, ndims, shape0 = captured[0]
        assert data_ptr != 0
        assert ndims == 1
        assert shape0 == 64

    def test_alloc_dep_wires_via_tensormap(self):
        """INOUT producer -> alloc'd ptr -> INPUT consumer wires the dep."""
        marker_shm, marker_buf = _make_shared_counter()

        try:
            hw = Worker(level=3, num_sub_workers=2)
            producer_handle = hw.register(lambda args: _increment_counter(marker_buf))
            consumer_handle = hw.register(lambda args: _increment_counter(marker_buf))
            hw.init()

            def orch(o, args, cfg):
                inter = o.alloc((128,), DataType.FLOAT32)

                # Producer writes into the alloc'd slab and must depend on
                # the alloc-slot (the creator) so the slab is not reclaimed
                # while the producer is still writing. That lifetime link
                # goes through INOUT — matching L2, only INPUT and INOUT
                # do TensorMap.lookup. Plain OUTPUT / OUTPUT_EXISTING are
                # pure inserts and would leave no dep on the alloc slot.
                p_args = TaskArgs()
                p_args.add_tensor(inter, TensorArgType.INOUT)
                o.submit_sub(producer_handle, p_args)

                # Consumer tags inter as INPUT — tensormap.lookup finds the
                # producer slot, dep wired automatically.
                c_args = TaskArgs()
                c_args.add_tensor(inter, TensorArgType.INPUT)
                o.submit_sub(consumer_handle, c_args)

            hw.run(orch)
            hw.close()

            # Both ran (we don't assert order strictly — relies on dep enforcement
            # which we'd need a write-then-read assert to verify; counter==2 at
            # least confirms both fired and no deadlock).
            assert _read_counter(marker_buf) == 2
        finally:
            marker_shm.close()
            marker_shm.unlink()

    def test_alloc_unused_freed_at_scope_end(self):
        """alloc that's never tagged still consumes cleanly via scope ref."""
        hw = Worker(level=3, num_sub_workers=0)
        hw.init()

        def orch(o, args, cfg):
            o.alloc((16,), DataType.UINT8)
            o.alloc((32,), DataType.FLOAT32)
            # No submits using these — synthetic slots' fanout_total = 1 (scope only)
            # scope_end's release_ref alone hits the threshold (sim self + scope = 2 = total + 1).

        hw.run(orch)
        hw.close()
        # If munmap leaks or the slot doesn't reach CONSUMED, drain hangs above.

    def test_alloc_across_runs_does_not_leak(self):
        """Repeated runs each alloc + use; slots must be released between runs."""
        marker_shm, marker_buf = _make_shared_counter()

        try:
            hw = Worker(level=3, num_sub_workers=1)
            handle = hw.register(lambda args: _increment_counter(marker_buf))
            hw.init()

            def orch(o, args, cfg):
                inter = o.alloc((64,), DataType.FLOAT32)
                args = TaskArgs()
                args.add_tensor(inter, TensorArgType.INPUT)
                o.submit_sub(handle, args)

            for _ in range(8):
                hw.run(orch)

            hw.close()
            assert _read_counter(marker_buf) == 8
        finally:
            marker_shm.close()
            marker_shm.unlink()


# ---------------------------------------------------------------------------
# Test: sub callable receives args blob correctly
# ---------------------------------------------------------------------------


class TestSubCallableArgs:
    def test_sub_callable_receives_tensor_metadata(self):
        """Sub callable receives TaskArgs with correct tensor count and shape."""
        from simpler.task_interface import ChipTensor  # noqa: PLC0415

        result_shm, result_buf = _make_shared_counter()
        try:
            hw = Worker(level=3, num_sub_workers=1)

            def check_args(args):
                # Verify args decoded correctly: 1 tensor, shape (4,), FLOAT32
                if args.tensor_count() == 1 and args.scalar_count() == 0:
                    t = args.tensor(0)
                    if t.ndims == 1 and t.shapes[0] == 4:
                        _increment_counter(result_buf)

            handle = hw.register(check_args)
            hw.init()

            # Use a synthetic non-zero pointer — sub callable only checks metadata,
            # doesn't dereference the pointer.
            ct = ChipTensor.make(0xCAFE0000, (4,), DataType.FLOAT32)

            def orch(o, args, cfg):
                sub_args = TaskArgs()
                sub_args.add_tensor(ct, TensorArgType.INPUT)
                o.submit_sub(handle, sub_args)

            hw.run(orch)
            hw.close()

            assert _read_counter(result_buf) == 1, "Sub callable did not receive correct args"
        finally:
            result_shm.close()
            result_shm.unlink()

    def test_sub_callable_receives_scalar(self):
        """Sub callable receives TaskArgs with a scalar value."""
        result_shm, result_buf = _make_shared_counter()
        try:
            hw = Worker(level=3, num_sub_workers=1)

            def check_scalar(args):
                if args.scalar_count() == 1 and args.scalar(0) == 42:
                    _increment_counter(result_buf)

            handle = hw.register(check_scalar)
            hw.init()

            def orch(o, args, cfg):
                sub_args = TaskArgs()
                sub_args.add_scalar(42)
                o.submit_sub(handle, sub_args)

            hw.run(orch)
            hw.close()

            assert _read_counter(result_buf) == 1, "Sub callable did not receive correct scalar"
        finally:
            result_shm.close()
            result_shm.unlink()

    def test_sub_callable_empty_args(self):
        """Sub callable receives empty TaskArgs when no args submitted."""
        result_shm, result_buf = _make_shared_counter()
        try:
            hw = Worker(level=3, num_sub_workers=1)

            def check_empty(args):
                if args.tensor_count() == 0 and args.scalar_count() == 0:
                    _increment_counter(result_buf)

            handle = hw.register(check_empty)
            hw.init()

            def orch(o, args, cfg):
                o.submit_sub(handle)

            hw.run(orch)
            hw.close()

            assert _read_counter(result_buf) == 1, "Sub callable did not receive empty args"
        finally:
            result_shm.close()
            result_shm.unlink()


# ---------------------------------------------------------------------------
# Test: _CTRL_REGISTER digest-owned child slots
# ---------------------------------------------------------------------------


class TestChipMainLoopDigestRegister:
    """Direct white-box tests on _run_chip_main_loop dynamic registration."""

    @staticmethod
    def _build_mailbox():
        from simpler.task_interface import MAILBOX_SIZE  # noqa: PLC0415
        from simpler.worker import _IDLE, _OFF_STATE, _buffer_field_addr, _mailbox_store_i32  # noqa: PLC0415

        shm = SharedMemory(create=True, size=MAILBOX_SIZE)
        buf = shm.buf
        assert buf is not None
        # Loop reads the state field via a raw address (atomic_int32 in C++),
        # so we hand it the absolute address and let it cast back inside.
        state_addr = _buffer_field_addr(buf, _OFF_STATE)
        _mailbox_store_i32(state_addr, _IDLE)
        # `mailbox_addr` is only consumed by the TASK_READY branch, which we
        # never reach in these tests; passing 0 keeps the harness lean.
        return shm, buf, state_addr

    @staticmethod
    def _send_ctrl_register(
        buf,
        state_addr,
        shm_name: str,
        *,
        payload_size: int,
        digest: bytes = b"\x07" * 32,
    ):
        """Stage a CTRL_REGISTER request and flip the state to CONTROL_REQUEST."""
        from simpler.worker import (  # noqa: PLC0415
            _CONTROL_REQUEST,
            _CTRL_OFF_ARG0,
            _CTRL_REGISTER,
            _CTRL_SHM_NAME_BYTES,
            _OFF_ARGS,
            _OFF_CALLABLE,
            _OFF_CONTROL_CALLABLE_HASH,
            _mailbox_store_i32,
        )

        struct.pack_into("Q", buf, _OFF_CALLABLE, _CTRL_REGISTER)
        struct.pack_into("Q", buf, _CTRL_OFF_ARG0, int(payload_size))
        assert len(digest) == 32
        buf[_OFF_CONTROL_CALLABLE_HASH : _OFF_CONTROL_CALLABLE_HASH + len(digest)] = digest
        encoded = shm_name.encode("utf-8")
        assert len(encoded) + 1 <= _CTRL_SHM_NAME_BYTES
        buf[_OFF_ARGS : _OFF_ARGS + len(encoded)] = encoded
        buf[_OFF_ARGS + len(encoded) : _OFF_ARGS + _CTRL_SHM_NAME_BYTES] = b"\x00" * (
            _CTRL_SHM_NAME_BYTES - len(encoded)
        )
        _mailbox_store_i32(state_addr, _CONTROL_REQUEST)

    @staticmethod
    def _send_ctrl_unregister(buf, state_addr, digest: bytes = b"\x07" * 32):
        from simpler.worker import (  # noqa: PLC0415
            _CONTROL_REQUEST,
            _CTRL_UNREGISTER,
            _OFF_CALLABLE,
            _OFF_CONTROL_CALLABLE_HASH,
            _mailbox_store_i32,
        )

        struct.pack_into("Q", buf, _OFF_CALLABLE, _CTRL_UNREGISTER)
        assert len(digest) == 32
        buf[_OFF_CONTROL_CALLABLE_HASH : _OFF_CONTROL_CALLABLE_HASH + len(digest)] = digest
        _mailbox_store_i32(state_addr, _CONTROL_REQUEST)

    @staticmethod
    def _wait_for_done_and_reset(buf, state_addr, timeout: float = 5.0):
        """Block until the loop publishes _CONTROL_DONE, then read the error
        code and reset the mailbox to _IDLE so the next round can start."""
        import time  # noqa: PLC0415

        from simpler.worker import (  # noqa: PLC0415
            _CONTROL_DONE,
            _IDLE,
            _OFF_ERROR,
            _mailbox_load_i32,
            _mailbox_store_i32,
        )

        deadline = time.monotonic() + timeout
        while _mailbox_load_i32(state_addr) != _CONTROL_DONE:
            if time.monotonic() > deadline:
                raise TimeoutError("loop did not publish CONTROL_DONE")
            time.sleep(0.001)
        err_code = struct.unpack_from("i", buf, _OFF_ERROR)[0]
        _mailbox_store_i32(state_addr, _IDLE)
        return err_code

    @staticmethod
    def _read_error_message(buf) -> str:
        raw = bytes(buf[MAILBOX_OFF_ERROR_MSG : MAILBOX_OFF_ERROR_MSG + MAILBOX_ERROR_MSG_SIZE])
        return raw.split(b"\x00", 1)[0].decode("utf-8", "replace")

    @staticmethod
    def _shutdown(state_addr):
        from simpler.worker import _SHUTDOWN, _mailbox_store_i32  # noqa: PLC0415

        _mailbox_store_i32(state_addr, _SHUTDOWN)

    @staticmethod
    def _spawn_loop(cw, buf, state_addr, registry=None, identity_table=None, identity_refs=None):
        from simpler.worker import _run_chip_main_loop  # noqa: PLC0415

        if registry is None:
            registry = {}
        if identity_table is None:
            identity_table = {}
        if identity_refs is None:
            identity_refs = {}
        t = threading.Thread(
            target=_run_chip_main_loop,
            args=(cw, buf, 0, state_addr, 0, registry, identity_table, identity_refs),
            kwargs={"chip_platform": ""},
            daemon=True,
        )
        t.start()
        return t

    def test_register_uses_payload_size_and_allocates_local_slot(self):
        from unittest.mock import MagicMock  # noqa: PLC0415

        cw = MagicMock()
        cw._impl = MagicMock()
        cw._unregister_slot = MagicMock()
        cw._impl.register_callable_from_blob = MagicMock()

        callable_obj = _unique_chip_callable(7)
        digest = _chip_digest(callable_obj)
        payload_shm = _chip_payload_shm(callable_obj)
        shm, buf, state_addr = self._build_mailbox()
        try:
            t = self._spawn_loop(cw, buf, state_addr)
            try:
                self._send_ctrl_register(
                    buf,
                    state_addr,
                    shm_name=payload_shm.name,
                    digest=digest,
                    payload_size=int(callable_obj.buffer_size()),
                )
                err = self._wait_for_done_and_reset(buf, state_addr)
                assert err == 0
                assert cw._unregister_slot.call_count == 0
                cw._impl.register_callable_from_blob.assert_called_once()
                assert cw._impl.register_callable_from_blob.call_args.args[0] == 0
            finally:
                self._shutdown(state_addr)
                t.join(timeout=2.0)
        finally:
            shm.close()
            shm.unlink()
            payload_shm.close()
            payload_shm.unlink()

    def test_register_reads_only_declared_payload_size(self):
        from unittest.mock import MagicMock  # noqa: PLC0415

        cw = MagicMock()
        cw._impl = MagicMock()
        cw._unregister_slot = MagicMock()
        cw._impl.register_callable_from_blob = MagicMock()

        callable_obj = _unique_chip_callable(7)
        payload = ctypes.string_at(int(callable_obj.buffer_ptr()), int(callable_obj.buffer_size()))
        digest = _chip_digest(callable_obj)
        payload_shm = SharedMemory(create=True, size=len(payload) + 4096)
        payload_buf = payload_shm.buf
        assert payload_buf is not None
        try:
            payload_buf[: len(payload)] = payload
            payload_buf[len(payload) : len(payload) + 4096] = b"\xff" * 4096
        finally:
            payload_buf.release()
        shm, buf, state_addr = self._build_mailbox()
        try:
            t = self._spawn_loop(cw, buf, state_addr)
            try:
                self._send_ctrl_register(
                    buf,
                    state_addr,
                    shm_name=payload_shm.name,
                    digest=digest,
                    payload_size=len(payload),
                )
                assert self._wait_for_done_and_reset(buf, state_addr) == 0
                cw._impl.register_callable_from_blob.assert_called_once()
            finally:
                self._shutdown(state_addr)
                t.join(timeout=2.0)
        finally:
            shm.close()
            shm.unlink()
            payload_shm.close()
            payload_shm.unlink()

    def test_register_rejects_digest_descriptor_mismatch(self):
        from unittest.mock import MagicMock  # noqa: PLC0415

        cw = MagicMock()
        cw._impl = MagicMock()
        cw._unregister_slot = MagicMock()
        cw._impl.register_callable_from_blob = MagicMock()

        callable_obj = _unique_chip_callable(7)
        wrong_digest = _chip_digest(_unique_chip_callable(8))
        payload_shm = _chip_payload_shm(callable_obj)
        shm, buf, state_addr = self._build_mailbox()
        registry = {}
        identity_table = {}
        identity_refs = {}
        try:
            t = self._spawn_loop(cw, buf, state_addr, registry, identity_table, identity_refs)
            try:
                self._send_ctrl_register(
                    buf,
                    state_addr,
                    shm_name=payload_shm.name,
                    payload_size=int(callable_obj.buffer_size()),
                    digest=wrong_digest,
                )
                err = self._wait_for_done_and_reset(buf, state_addr)
                assert err == 1
                assert "HASHID_DESCRIPTOR_MISMATCH" in self._read_error_message(buf)
                cw._impl.register_callable_from_blob.assert_not_called()
                cw._unregister_slot.assert_not_called()
                assert registry == {}
                assert identity_table == {}
                assert identity_refs == {}
            finally:
                self._shutdown(state_addr)
                t.join(timeout=2.0)
        finally:
            shm.close()
            shm.unlink()
            payload_shm.close()
            payload_shm.unlink()

    def test_duplicate_register_increments_ref_without_reprepare(self):
        from unittest.mock import MagicMock  # noqa: PLC0415

        cw = MagicMock()
        cw._impl = MagicMock()
        cw._unregister_slot = MagicMock()
        cw._impl.register_callable_from_blob = MagicMock()

        callable_obj = _unique_chip_callable(7)
        digest = _chip_digest(callable_obj)
        payload_shm = _chip_payload_shm(callable_obj)
        payload_size = int(callable_obj.buffer_size())
        shm, buf, state_addr = self._build_mailbox()
        try:
            t = self._spawn_loop(cw, buf, state_addr)
            try:
                self._send_ctrl_register(
                    buf,
                    state_addr,
                    shm_name=payload_shm.name,
                    payload_size=payload_size,
                    digest=digest,
                )
                assert self._wait_for_done_and_reset(buf, state_addr) == 0
                self._send_ctrl_register(
                    buf,
                    state_addr,
                    shm_name=payload_shm.name,
                    payload_size=payload_size,
                    digest=digest,
                )
                assert self._wait_for_done_and_reset(buf, state_addr) == 0
                assert cw._unregister_slot.call_count == 0
                assert cw._impl.register_callable_from_blob.call_count == 1
            finally:
                self._shutdown(state_addr)
                t.join(timeout=2.0)
        finally:
            shm.close()
            shm.unlink()
            payload_shm.close()
            payload_shm.unlink()

    def test_unregister_removes_only_after_last_digest_ref(self):
        from unittest.mock import MagicMock  # noqa: PLC0415

        cw = MagicMock()
        cw._impl = MagicMock()
        cw._unregister_slot = MagicMock()
        cw._impl.register_callable_from_blob = MagicMock()

        callable_obj = _unique_chip_callable(7)
        digest = _chip_digest(callable_obj)
        payload_shm = _chip_payload_shm(callable_obj)
        payload_size = int(callable_obj.buffer_size())
        shm, buf, state_addr = self._build_mailbox()
        try:
            t = self._spawn_loop(cw, buf, state_addr)
            try:
                self._send_ctrl_register(
                    buf,
                    state_addr,
                    shm_name=payload_shm.name,
                    payload_size=payload_size,
                    digest=digest,
                )
                assert self._wait_for_done_and_reset(buf, state_addr) == 0
                self._send_ctrl_register(
                    buf,
                    state_addr,
                    shm_name=payload_shm.name,
                    payload_size=payload_size,
                    digest=digest,
                )
                assert self._wait_for_done_and_reset(buf, state_addr) == 0

                self._send_ctrl_unregister(buf, state_addr, digest=digest)
                assert self._wait_for_done_and_reset(buf, state_addr) == 0
                assert cw._unregister_slot.call_count == 0

                self._send_ctrl_unregister(buf, state_addr, digest=digest)
                assert self._wait_for_done_and_reset(buf, state_addr) == 0
                cw._unregister_slot.assert_called_once_with(0)
            finally:
                self._shutdown(state_addr)
                t.join(timeout=2.0)
        finally:
            shm.close()
            shm.unlink()
            payload_shm.close()
            payload_shm.unlink()
