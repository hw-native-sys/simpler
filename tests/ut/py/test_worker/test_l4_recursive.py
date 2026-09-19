# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Unit tests for L4 → L3 → L2 recursive Worker composition.

L4 Worker dispatches to L3 Worker children (PROCESS mode). Each L3 Worker
has its own SubWorkers. Verifies the full DAG completes and sub callables
at L3 level see correct data.

No NPU device required: the dataflow tests give their L3 children SubWorkers
only, and the chip-callable cascade gives them a fake chip (``_harness``).
"""

from __future__ import annotations

import ast
import os
import struct
import threading
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path

import pytest
from _task_interface import DataType
from simpler.buffer import mint_owner_instance_id, wrap_device_malloc
from simpler.callable_identity import CallableHandle
from simpler.task_interface import CallConfig, TaskArgs, TaskHandle
from simpler.worker import RunHandle, Worker, _CloseAttempt, _Lifecycle, attach_exception_note

from tests.st.worker.comm_region.recursive_single_owner._helpers import close_owned_workers

from ._harness import (
    SIM_PLATFORM,
    SIM_RUNTIME,
    TEST_WALL_BUDGET_S,
    chip_callable,
    hard_timeout,
    install_fake_chip,
    requires_sim_binaries,
)

_OID = mint_owner_instance_id()
_HOSTBUF = bytearray(64)


def _dev_handle(ptr: int, *, wid: int = 0):
    return wrap_device_malloc(ptr, 64, _OID, buffer_id=ptr, owner_worker_id=wid)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_shared_counter(slots: int = 1):
    """Allocate `slots` 4-byte shared counters accessible from forked subprocesses.

    `_increment_counter` is a non-atomic read-modify-write, so a slot tolerates
    at most one writing process: two processes that read before either writes
    both store the same value and one increment is lost. A test whose
    increments come from more than one process must give each process its own
    slot.
    """
    shm = SharedMemory(create=True, size=4 * slots)
    buf = shm.buf
    assert buf is not None
    for slot in range(slots):
        struct.pack_into("i", buf, 4 * slot, 0)
    return shm, buf


def _read_counter(buf, slot: int = 0) -> int:
    return struct.unpack_from("i", buf, 4 * slot)[0]


def _increment_counter(buf, slot: int = 0) -> None:
    v = struct.unpack_from("i", buf, 4 * slot)[0]
    struct.pack_into("i", buf, 4 * slot, v + 1)


def _slot_for(worker: Worker, handle: CallableHandle) -> int:
    return worker._identity_registry[handle.digest].slot_id


# ---------------------------------------------------------------------------
# Test: L4 lifecycle (init / close without submitting any tasks)
# ---------------------------------------------------------------------------


class TestL4Lifecycle:
    def test_init_close_no_children(self):
        """L4 with zero next-level workers and zero sub workers."""
        w4 = Worker(level=4, num_sub_workers=0)
        w4.init()
        w4.close()

    def test_init_close_with_l3_child(self):
        """L4 with one L3 child (no device, sub-only) — init and close cleanly."""
        l3 = Worker(level=3, num_sub_workers=1)
        l3.register(lambda args: None)

        w4 = Worker(level=4, num_sub_workers=0)
        w4.register(lambda orch, args, config: None)
        w4.add_worker(l3)
        w4.init()
        w4.close()

    def test_context_manager(self):
        """L4 via context manager cleans up correctly."""
        l3 = Worker(level=3, num_sub_workers=1)
        l3.register(lambda args: None)

        with Worker(level=4, num_sub_workers=0) as w4:
            w4.register(lambda orch, args, config: None)
            w4.add_worker(l3)
            w4.init()


class TestL4Validation:
    def test_add_worker_requires_level4(self):
        """add_worker on level 3 raises."""
        w3 = Worker(level=3, num_sub_workers=0)
        child = Worker(level=3, num_sub_workers=0)
        with pytest.raises(RuntimeError, match="level >= 4"):
            w3.add_worker(child)

    def test_add_worker_after_init_raises(self):
        w4 = Worker(level=4, num_sub_workers=0)
        w4.init()
        child = Worker(level=3, num_sub_workers=0)
        with pytest.raises(RuntimeError, match="before init"):
            w4.add_worker(child)
        w4.close()

    def test_add_initialized_child_raises(self):
        child = Worker(level=3, num_sub_workers=0)
        child.init()
        w4 = Worker(level=4, num_sub_workers=0)
        with pytest.raises(RuntimeError, match="must be NEW"):
            w4.add_worker(child)
        child.close()
        w4.close()

    def test_l4_device_ids_rejected(self):
        w4 = Worker(level=4, device_ids=[0], num_sub_workers=0)
        with pytest.raises(RuntimeError, match="device_ids are only supported on L3"):
            w4.init()

    def test_add_worker_with_device_ids_rejected(self):
        w4 = Worker(level=4, device_ids=[0], num_sub_workers=0)
        child = Worker(level=3, num_sub_workers=0)
        with pytest.raises(RuntimeError, match="cannot be combined with device_ids"):
            w4.add_worker(child)

    def test_device_mem_on_l4_rejected(self):
        # L4 has no chip mailboxes — device memory ops must be rejected rather than silently dispatch
        # CTRL_MALLOC/FREE/COPY to a next_level (L3 worker) child whose `_child_worker_loop` doesn't
        # recognise them and would return a garbage pointer from an uninitialised mailbox result slot.
        # `malloc` is L2-only (TypeError); the child-device ops guard the worker id (IndexError).
        l3_child = Worker(level=3, num_sub_workers=0)
        w4 = Worker(level=4, num_sub_workers=0)
        w4.add_worker(l3_child)
        w4.init()
        try:
            with pytest.raises(TypeError, match="L2-only"):
                w4.malloc(1024)
            with pytest.raises(IndexError, match="out of range"):
                w4.alloc_child_tensor(0, (256,), DataType.FLOAT32)
            with pytest.raises(IndexError, match="out of range"):
                w4.free(_dev_handle(0xDEADBEEF, wid=0))
            with pytest.raises(IndexError, match="out of range"):
                w4.copy_to(_dev_handle(0xDEAD, wid=0), _HOSTBUF)
            with pytest.raises(IndexError, match="out of range"):
                w4.copy_from(_HOSTBUF, _dev_handle(0xBEEF, wid=0))
        finally:
            w4.close()


@requires_sim_binaries
class TestL4DynamicRegister:
    """Cascade of CTRL_REGISTER / CTRL_UNREGISTER through an L4 → L3 worker tree.

    The L3 child carries a fake chip, so the cascade runs the whole path from
    L4 parent → next_level mailbox → ``_child_worker_loop`` CONTROL handler →
    ``inner_worker._register_child_chip`` → the L3's own chip mailbox → the
    chip child's native register. The fake crosses both forks, so no NPU is
    required.
    """

    @staticmethod
    def _l4_over_chip_backed_l3(monkeypatch, *, register_error: str | None = None) -> Worker:
        install_fake_chip(monkeypatch, register_error=register_error)
        l3 = Worker(
            level=3,
            device_ids=[0],
            platform=SIM_PLATFORM,
            runtime=SIM_RUNTIME,
            num_sub_workers=1,
            startup_timeout_s=10.0,
        )
        l3.register(lambda args: None)  # at least one registered callable so L3 init is happy

        w4 = Worker(level=4, num_sub_workers=0, startup_timeout_s=10.0)
        w4.register(lambda orch, args, config: None)
        w4.add_worker(l3)
        return w4

    def test_l4_register_chip_callable_after_init_succeeds(self, monkeypatch):
        w4 = self._l4_over_chip_backed_l3(monkeypatch)
        try:
            with hard_timeout(TEST_WALL_BUDGET_S):
                w4.init()
                handle = w4.register(chip_callable())
                assert isinstance(handle, CallableHandle)
                assert _slot_for(w4, handle) >= 0
        finally:
            w4.close()

    def test_l4_register_then_unregister_recycles_cid(self, monkeypatch):
        w4 = self._l4_over_chip_backed_l3(monkeypatch)
        try:
            with hard_timeout(TEST_WALL_BUDGET_S):
                w4.init()
                callable_obj = chip_callable()
                handle_a = w4.register(callable_obj)
                slot_a = _slot_for(w4, handle_a)
                w4.unregister(handle_a)
                assert slot_a not in w4._callable_registry
                # Slot is freed; the next register reuses it.
                handle_b = w4.register(callable_obj)
                assert _slot_for(w4, handle_b) == slot_a
        finally:
            w4.close()

    def test_l4_register_surfaces_grandchild_chip_register_failure(self, monkeypatch):
        # The cascade terminates at the chip child two forks down, not at the
        # L3 child: a native register that raises there travels back up as the
        # L4's REGISTER_PARTIAL_FAILURE.
        w4 = self._l4_over_chip_backed_l3(monkeypatch, register_error="injected chip register failure")
        try:
            with hard_timeout(TEST_WALL_BUDGET_S):
                w4.init()
                with pytest.raises(RuntimeError, match="REGISTER_PARTIAL_FAILURE"):
                    w4.register(chip_callable())
        finally:
            w4.close()

    def test_l4_register_python_orch_after_start_succeeds(self):
        counter_shm, counter_buf = _make_shared_counter()
        try:
            l3 = Worker(level=3, num_sub_workers=1)
            l3_sub_handle = l3.register(lambda args: _increment_counter(counter_buf))

            w4 = Worker(level=4, num_sub_workers=0)
            bootstrap_handle = w4.register(lambda orch, args, config: None)
            l3_worker_id = w4.add_worker(l3)
            w4.init()

            def bootstrap(orch, args, config):
                orch.submit_next_level(bootstrap_handle, TaskArgs(), CallConfig(), worker=l3_worker_id)

            w4.run(bootstrap)

            def dynamic_l3_orch(orch, args, config):
                orch.submit_sub(l3_sub_handle)

            dynamic_handle = w4.register(dynamic_l3_orch)

            def l4_orch(orch, args, config):
                orch.submit_next_level(dynamic_handle, TaskArgs(), CallConfig(), worker=l3_worker_id)

            w4.run(l4_orch)
            w4.close()

            assert _read_counter(counter_buf) == 1
        finally:
            counter_shm.close()
            counter_shm.unlink()


# ---------------------------------------------------------------------------
# Test: L4 → L3 PROCESS mode — single dispatch
# ---------------------------------------------------------------------------


class TestL4ToL3SingleDispatch:
    def test_l4_dispatches_to_l3_sub(self):
        """L4 orch submits one task to L3 child. L3 orch runs a sub callable.

        Verifies that the sub callable's shared counter is incremented,
        proving the full L4 → L3 → sub dispatch chain works.
        """
        counter_shm, counter_buf = _make_shared_counter()

        try:
            # L3 child: one sub worker, one sub callable that increments counter
            l3 = Worker(level=3, num_sub_workers=1)
            l3_sub_handle = l3.register(lambda args: _increment_counter(counter_buf))

            def l3_orch(orch, args, config):
                orch.submit_sub(l3_sub_handle)

            # L4 parent: one next-level child, register L3 orch fn
            w4 = Worker(level=4, num_sub_workers=0)
            l3_handle = w4.register(l3_orch)
            l3_worker_id = w4.add_worker(l3)
            w4.init()

            def l4_orch(orch, args, config):
                orch.submit_next_level(l3_handle, TaskArgs(), CallConfig(), worker=l3_worker_id)

            w4.run(l4_orch)
            w4.close()

            assert _read_counter(counter_buf) == 1
        finally:
            counter_shm.close()
            counter_shm.unlink()


# ---------------------------------------------------------------------------
# Test: L4 → L3 — multiple dispatches
# ---------------------------------------------------------------------------


class TestL4ToL3MultipleDispatches:
    @pytest.mark.parametrize("dep_method", ["add_dep", "add_dep_wait"])
    def test_l4_explicit_task_dependency_uses_returned_handle(self, dep_method):
        counter_shm, counter_buf = _make_shared_counter()

        try:
            l3 = Worker(level=3, num_sub_workers=1)
            l3_sub_handle = l3.register(lambda args: _increment_counter(counter_buf))

            def l3_orch(orch, args, config):
                orch.submit_sub(l3_sub_handle)

            w4 = Worker(level=4, num_sub_workers=0)
            l3_handle = w4.register(l3_orch)
            l3_worker_id = w4.add_worker(l3)
            w4.init()

            def l4_orch(orch, args, config):
                producer = orch.submit_next_level(l3_handle, TaskArgs(), CallConfig(), worker=l3_worker_id)
                assert isinstance(producer, TaskHandle)
                consumer_args = TaskArgs()
                getattr(consumer_args, dep_method)(producer)
                consumer = orch.submit_next_level(l3_handle, consumer_args, CallConfig(), worker=l3_worker_id)
                assert isinstance(consumer, TaskHandle)

            w4.run(l4_orch)
            w4.close()

            assert _read_counter(counter_buf) == 2
        finally:
            counter_shm.close()
            counter_shm.unlink()

    def test_l4_dispatches_three_times(self):
        """L4 orch submits 3 tasks to L3 child, each running a sub callable."""
        counter_shm, counter_buf = _make_shared_counter()

        try:
            l3 = Worker(level=3, num_sub_workers=1)
            l3_sub_handle = l3.register(lambda args: _increment_counter(counter_buf))

            def l3_orch(orch, args, config):
                orch.submit_sub(l3_sub_handle)

            w4 = Worker(level=4, num_sub_workers=0)
            l3_handle = w4.register(l3_orch)
            l3_worker_id = w4.add_worker(l3)
            w4.init()

            def l4_orch(orch, args, config):
                for _ in range(3):
                    orch.submit_next_level(l3_handle, TaskArgs(), CallConfig(), worker=l3_worker_id)

            w4.run(l4_orch)
            w4.close()

            assert _read_counter(counter_buf) == 3
        finally:
            counter_shm.close()
            counter_shm.unlink()


# ---------------------------------------------------------------------------
# Test: L4 → L3 — per-child capture namespace
# ---------------------------------------------------------------------------


class TestL4ChildCaptureNamespace:
    def test_attached_l3_children_write_under_distinct_namespaces(self, tmp_path):
        """Two attached L3 children never share one artifact directory.

        Every diagnostic below `output_prefix` uses a fixed filename, and each
        L3 numbers its own chips from zero, so siblings handed one prefix write
        over each other. The probe stands in for such an artifact: it is
        written at the prefix the child was actually handed, so both survive
        only if the two children were handed different ones.
        """

        def l3_orch(orch, args, config):
            os.makedirs(config.output_prefix, exist_ok=True)
            with open(os.path.join(config.output_prefix, "probe.txt"), "w") as probe:
                probe.write(str(os.getpid()))

        first = Worker(level=3, num_sub_workers=1)
        first.register(lambda args: None)
        second = Worker(level=3, num_sub_workers=1)
        second.register(lambda args: None)

        w4 = Worker(level=4, num_sub_workers=0)
        l3_handle = w4.register(l3_orch)
        first_id = w4.add_worker(first)
        second_id = w4.add_worker(second)
        w4.init()

        child_config = CallConfig()
        child_config.output_prefix = str(tmp_path)

        def l4_orch(orch, args, config):
            for worker_id in (first_id, second_id):
                orch.submit_next_level(l3_handle, TaskArgs(), child_config, worker=worker_id)

        w4.run(l4_orch)
        w4.close()

        probes = {path.parent.name: path.read_text() for path in tmp_path.rglob("probe.txt")}
        assert set(probes) == {f"node{first_id}", f"node{second_id}"}
        assert len(set(probes.values())) == 2


# ---------------------------------------------------------------------------
# Test: L4 with own sub workers + L3 child
# ---------------------------------------------------------------------------


class TestL4WithOwnSubs:
    def test_l4_sub_and_l3_dispatch(self):
        """L4 has its own sub workers AND dispatches to an L3 child.

        L4's sub callable and L3's sub callable each increment a counter, and
        both paths must execute. They are the only two increments in this file
        that run in *different* processes — L4's own SubWorker and the L3
        child's SubWorker — dispatched from independent WorkerThreads with no
        ordering between them, so each gets its own slot. Sharing one slot
        would make the pair a cross-process non-atomic read-modify-write.
        """
        counter_shm, counter_buf = _make_shared_counter(slots=2)
        L3_SLOT, L4_SLOT = 0, 1

        try:
            # L3 child: sub worker increments counter
            l3 = Worker(level=3, num_sub_workers=1)
            l3_sub_handle = l3.register(lambda args: _increment_counter(counter_buf, L3_SLOT))

            def l3_orch(orch, args, config):
                orch.submit_sub(l3_sub_handle)

            # L4: own sub worker + L3 child
            w4 = Worker(level=4, num_sub_workers=1)
            l3_handle = w4.register(l3_orch)
            l4_verify_handle = w4.register(lambda args: _increment_counter(counter_buf, L4_SLOT))
            l3_worker_id = w4.add_worker(l3)
            w4.init()

            def l4_orch(orch, args, config):
                orch.submit_next_level(l3_handle, TaskArgs(), CallConfig(), worker=l3_worker_id)
                orch.submit_sub(l4_verify_handle)

            w4.run(l4_orch)
            w4.close()

            assert _read_counter(counter_buf, L3_SLOT) == 1, "L3 sub callable did not run"
            assert _read_counter(counter_buf, L4_SLOT) == 1, "L4 sub callable did not run"
        finally:
            counter_shm.close()
            counter_shm.unlink()


# ---------------------------------------------------------------------------
# Test: L4 → L3 — multiple runs
# ---------------------------------------------------------------------------


class TestL4MultipleRuns:
    def test_l4_multiple_runs_no_leak(self):
        """Multiple w4.run() calls on the same Worker — slots don't leak."""
        counter_shm, counter_buf = _make_shared_counter()

        try:
            l3 = Worker(level=3, num_sub_workers=1)
            l3_sub_handle = l3.register(lambda args: _increment_counter(counter_buf))

            def l3_orch(orch, args, config):
                orch.submit_sub(l3_sub_handle)

            w4 = Worker(level=4, num_sub_workers=0)
            l3_handle = w4.register(l3_orch)
            l3_worker_id = w4.add_worker(l3)
            w4.init()

            def l4_orch(orch, args, config):
                orch.submit_next_level(l3_handle, TaskArgs(), CallConfig(), worker=l3_worker_id)

            for _ in range(5):
                w4.run(l4_orch)

            w4.close()

            assert _read_counter(counter_buf) == 5
        finally:
            counter_shm.close()
            counter_shm.unlink()


# ---------------------------------------------------------------------------
# Test: L4 → L3 — L3 uses multiple sub workers
# ---------------------------------------------------------------------------


class TestL4L3WithMultipleSubs:
    def test_l3_child_runs_multiple_subs(self):
        """L3 child submits 2 sub tasks per dispatch (serialized through 1 worker).

        Uses 1 sub worker because _increment_counter is a non-atomic RMW
        that races across parallel SubWorker processes.
        """
        counter_shm, counter_buf = _make_shared_counter()

        try:
            l3 = Worker(level=3, num_sub_workers=1)
            l3_sub_handle = l3.register(lambda args: _increment_counter(counter_buf))

            def l3_orch(orch, args, config):
                orch.submit_sub(l3_sub_handle)
                orch.submit_sub(l3_sub_handle)

            w4 = Worker(level=4, num_sub_workers=0)
            l3_handle = w4.register(l3_orch)
            l3_worker_id = w4.add_worker(l3)
            w4.init()

            def l4_orch(orch, args, config):
                orch.submit_next_level(l3_handle, TaskArgs(), CallConfig(), worker=l3_worker_id)

            w4.run(l4_orch)
            w4.close()

            assert _read_counter(counter_buf) == 2
        finally:
            counter_shm.close()
            counter_shm.unlink()


# ---------------------------------------------------------------------------
# Test: L3 orch receives its own Orchestrator (not L4's)
# ---------------------------------------------------------------------------


class TestL3OwnOrchestrator:
    def test_l3_gets_own_orchestrator(self):
        """The L3 orch fn receives an Orchestrator from the L3 inner worker,
        not the L4 parent. Prove by checking orch.alloc works at L3 level."""
        counter_shm, counter_buf = _make_shared_counter()

        try:
            l3 = Worker(level=3, num_sub_workers=1)
            l3_sub_handle = l3.register(lambda args: _increment_counter(counter_buf))

            def l3_orch(orch, args, config):
                # orch is L3's own Orchestrator — alloc + submit_sub should work
                orch.submit_sub(l3_sub_handle)

            w4 = Worker(level=4, num_sub_workers=0)
            l3_handle = w4.register(l3_orch)
            l3_worker_id = w4.add_worker(l3)
            w4.init()

            def l4_orch(orch, args, config):
                orch.submit_next_level(l3_handle, TaskArgs(), CallConfig(), worker=l3_worker_id)

            w4.run(l4_orch)
            w4.close()

            assert _read_counter(counter_buf) == 1
        finally:
            counter_shm.close()
            counter_shm.unlink()


# ---------------------------------------------------------------------------
# Test: generalised _Worker(level) — no hardcoded 3
# ---------------------------------------------------------------------------


class TestGeneralised_Worker:
    def test_worker_level_param(self):
        """_Worker accepts level != 3 without error."""
        from simpler.task_interface import _Worker  # noqa: PLC0415

        for level in (3, 4, 5):
            dw = _Worker(level)
            dw.close()


# ---------------------------------------------------------------------------
# Delegated-region routing, fatal latch, and unified materializer entry


_SESSION = b"\x11\x22\x33\x44\x55\x66\x77\x88"
_REPO_ROOT = Path(__file__).resolve().parents[4]
_WORKER_PY = _REPO_ROOT / "python" / "simpler" / "worker.py"


class _RecordingNative:
    def __init__(self, replies):
        self.calls: list[tuple[int, int, bytes]] = []
        self._replies = replies

    def control_payload(self, _worker_type, worker_id, sub_cmd, payload, _timeout):
        payload_bytes = bytes(payload)
        self.calls.append((int(worker_id), int(sub_cmd), payload_bytes))
        return self._replies(payload_bytes)


def _allocate_request(*, initiator_path: bytes, provider_path: bytes, transaction_id: int = 1):
    from simpler.comm_provider_control import DelegatedAllocateRequest  # noqa: PLC0415

    return DelegatedAllocateRequest(
        session_instance_id=_SESSION,
        transaction_id=transaction_id,
        initiator_path=initiator_path,
        provider_path=provider_path,
        payload_logical_bytes=64,
        counter_logical_bytes=8,
    )


def _backend_error_reply(payload: bytes) -> bytes:
    from simpler.comm_provider import RegionControlErrorKind  # noqa: PLC0415
    from simpler.comm_provider_control import (  # noqa: PLC0415
        DelegatedAllocateReply,
        DelegatedAllocateReplyTag,
        encode_reply,
        parse_request,
        publish_reply,
    )

    envelope = parse_request(payload)
    committed = encode_reply(
        DelegatedAllocateReply(
            tag=DelegatedAllocateReplyTag.ERROR,
            session_instance_id=envelope.session_instance_id,
            transaction_id=envelope.transaction_id,
            error_kind=RegionControlErrorKind.INVALID_FIELD_VALUE,
        )
    )
    staged = bytearray(len(payload))
    publish_reply(memoryview(staged), committed)
    return bytes(staged)


class TestDelegatedRouting:
    def test_l3_to_l2_forwards_unchanged_active_bytes(self):
        from simpler.comm_provider_control import (  # noqa: PLC0415
            ALLOCATE_REPLY_BYTES,
            encode_request,
            parse_request,
        )
        from simpler.worker import _CTRL_DELEGATED_REGION, _forward_delegated_region  # noqa: PLC0415

        native = _RecordingNative(_backend_error_reply)
        worker = Worker(level=3, device_ids=[8, 9], num_sub_workers=0)
        worker._worker = native
        request = encode_request(
            _allocate_request(initiator_path=b"L3", provider_path=b"L3/L2[1]"),
            staged_capacity=ALLOCATE_REPLY_BYTES,
        )
        envelope = parse_request(request)
        _forward_delegated_region(worker, "L3", memoryview(request))
        assert len(native.calls) == 1
        child_id, sub_cmd, hop = native.calls[0]
        assert child_id == 1
        assert sub_cmd == _CTRL_DELEGATED_REGION
        assert hop[: envelope.request_bytes] == envelope.frame
        assert len(hop) == max(envelope.request_bytes, ALLOCATE_REPLY_BYTES)
        assert not any(hop[envelope.request_bytes :])

    def test_l4_to_l3_to_l2_keeps_request_bytes_and_creates_no_table(self):
        from simpler.comm_provider_control import (  # noqa: PLC0415
            ALLOCATE_REPLY_BYTES,
            ProviderTransactionTable,
            encode_request,
            parse_request,
        )
        from simpler.worker import _CTRL_DELEGATED_REGION, _forward_delegated_region  # noqa: PLC0415

        created = []
        original_init = ProviderTransactionTable.__init__

        def _spy_init(self, *args, **kwargs):
            created.append(True)
            return original_init(self, *args, **kwargs)

        ProviderTransactionTable.__init__ = _spy_init
        try:
            l3_native = _RecordingNative(_backend_error_reply)
            l3 = Worker(level=3, device_ids=[4], num_sub_workers=0)
            l3._worker = l3_native
            l3._delegated_control_path = "L4/L3[0]"

            def _l4_replies(payload: bytes) -> bytes:
                staged = bytearray(payload)
                _forward_delegated_region(l3, "L4/L3[0]", memoryview(staged))
                return bytes(staged)

            l4_native = _RecordingNative(_l4_replies)
            l4 = Worker(level=4, num_sub_workers=0)
            l4._worker = l4_native
            l4._next_level_worker_ids = [0]
            l4._next_level_workers = [l3]
            l4._delegated_control_path = "L4"
            request = encode_request(
                _allocate_request(initiator_path=b"L4", provider_path=b"L4/L3[0]/L2[0]"),
                staged_capacity=ALLOCATE_REPLY_BYTES,
            )
            envelope = parse_request(request)
            _forward_delegated_region(l4, "L4", memoryview(request))
            assert [call[0] for call in l4_native.calls] == [0]
            assert [call[0] for call in l3_native.calls] == [0]
            assert l4_native.calls[0][1] == _CTRL_DELEGATED_REGION
            assert l3_native.calls[0][1] == _CTRL_DELEGATED_REGION
            assert l4_native.calls[0][2][: envelope.request_bytes] == envelope.frame
            assert l3_native.calls[0][2][: envelope.request_bytes] == envelope.frame
            assert created == []
        finally:
            ProviderTransactionTable.__init__ = original_init

    def test_routing_rejects_before_control_payload(self):
        from simpler.comm_provider import RegionControlError  # noqa: PLC0415
        from simpler.comm_provider_control import ALLOCATE_REPLY_BYTES, encode_request  # noqa: PLC0415
        from simpler.worker import _forward_delegated_region  # noqa: PLC0415

        native = _RecordingNative(_backend_error_reply)
        missing = Worker(level=3, device_ids=[8], num_sub_workers=0)
        missing._worker = native
        staged = encode_request(
            _allocate_request(initiator_path=b"L3", provider_path=b"L3/L2[1]"),
            staged_capacity=ALLOCATE_REPLY_BYTES,
        )
        with pytest.raises(RegionControlError, match="next L2 child does not exist"):
            _forward_delegated_region(missing, "L3", memoryview(staged))
        remote = Worker(level=4, num_sub_workers=0)
        remote._worker = native
        remote._remote_worker_ids = [0]
        remote._next_level_worker_ids = [0]
        with pytest.raises(RegionControlError, match="remote child is not supported"):
            _forward_delegated_region(
                remote,
                "L4",
                memoryview(
                    encode_request(
                        _allocate_request(initiator_path=b"L4", provider_path=b"L4/L3[0]/L2[0]"),
                        staged_capacity=ALLOCATE_REPLY_BYTES,
                    )
                ),
            )
        assert native.calls == []


class TestFatalBoundary:
    def test_fatal_latch_is_first_wins_and_does_not_close_inside_lease(self):
        worker = Worker(level=3, num_sub_workers=0)
        first = RuntimeError("first-fatal")
        worker._latch_delegated_session_fatal(first)
        worker._latch_delegated_session_fatal(RuntimeError("second-fatal"))
        assert worker._delegated_session_fatal is first
        assert worker._region_instance_registry._delegated_admission_closed is True
        closes: list[str] = []
        worker.close = lambda: closes.append("close")
        worker._lease_depth[threading.get_ident()] = 1
        worker._teardown_delegated_fatal_if_safe()
        assert closes == []
        worker._lease_depth.clear()
        worker._run_finalization_depth[threading.get_ident()] = 1
        worker._teardown_delegated_fatal_if_safe()
        assert closes == []
        worker._run_finalization_depth.clear()
        worker._close_completion = object()
        worker._teardown_delegated_fatal_if_safe()
        assert closes == []
        worker._close_completion = None
        worker._teardown_delegated_fatal_if_safe()
        assert closes == ["close"]

    def test_callback_fatal_rejects_later_submit_and_control(self):
        worker = Worker(level=3, device_ids=[0], num_sub_workers=0)
        worker._lifecycle = _Lifecycle.READY
        worker._worker = object()
        worker._config = {**worker._config, "platform": "a2a3sim", "device_ids": [0]}
        first = RuntimeError("callback-fatal")
        worker._latch_delegated_session_fatal(first)
        with pytest.raises(RuntimeError, match="no further work is admitted") as control_exc:
            worker._create_worker_chip_region(0, 64, 8)
        assert control_exc.value.__cause__ is first
        with pytest.raises(RuntimeError, match="no further work is admitted") as submit_exc:
            worker.submit(lambda *_args, **_kwargs: None)
        assert submit_exc.value.__cause__ is first
        assert worker._delegated_session_fatal is first

    def test_outer_submit_closes_after_lease_depth_returns_to_zero(self):
        worker = Worker(level=3, num_sub_workers=0)
        worker._lifecycle = _Lifecycle.READY
        closes: list[int] = []

        def _close():
            closes.append(worker._lease_depth.get(threading.get_ident(), 0))

        worker.close = _close

        def _submit_locked(*_args, **_kwargs):
            worker._latch_delegated_session_fatal(RuntimeError("callback-fatal"))
            return RunHandle._completed(worker)

        worker._submit_locked = _submit_locked
        handle = worker.submit(lambda *_args, **_kwargs: None)
        assert handle.done is True
        assert closes == [0]
        assert threading.get_ident() not in worker._lease_depth

    def test_release_fatal_teardown_runs_after_wait_publication(self):
        worker = Worker(level=3, num_sub_workers=0)
        published = RuntimeError("published-wait")
        worker._latch_delegated_session_fatal(published)
        teardowns: list[tuple[bool, BaseException | None]] = []
        original = worker._teardown_delegated_fatal_if_safe

        def _teardown(primary_error=None):
            teardowns.append((handle._terminal, primary_error))
            original(primary_error)

        worker._teardown_delegated_fatal_if_safe = _teardown
        worker.close = lambda: None
        worker._wait_run_handle = lambda *_args, **_kwargs: True
        worker._finalize_run_handle = lambda *_args, **_kwargs: published
        handle = RunHandle(worker, run_id=1, keepalive=())
        with pytest.raises(RuntimeError) as excinfo:
            handle.wait()
        assert excinfo.value is published
        assert handle._terminal is True
        assert teardowns == [(True, published)]

    def test_explicit_close_drain_fatal_does_not_recurse_into_close(self):
        worker = Worker(level=3, num_sub_workers=0)
        attempt = _CloseAttempt()
        worker._close_completion = attempt
        closes: list[str] = []

        def _close():
            closes.append("close")
            raise AssertionError("recursive close")

        worker.close = _close
        worker._latch_delegated_session_fatal(RuntimeError("drain-fatal"))
        worker._teardown_delegated_fatal_if_safe()
        assert closes == []
        assert worker._delegated_session_fatal is not None

    def test_submit_primary_close_failure_keeps_primary_and_replays_close(self):
        worker = Worker(level=3, num_sub_workers=0)
        worker._lifecycle = _Lifecycle.READY
        primary = RuntimeError("submit-primary")
        close_error = RuntimeError("close-failed")

        def _close():
            prior = worker._close_completion
            if prior is not None and prior.done:
                if prior.error is not None:
                    raise prior.error
                return
            attempt = _CloseAttempt()
            worker._close_completion = attempt
            attempt.publish(close_error, False)
            raise close_error

        worker.close = _close

        def _submit_locked(*_args, **_kwargs):
            worker._latch_delegated_session_fatal(RuntimeError("session-fatal"))
            raise primary

        worker._submit_locked = _submit_locked
        with pytest.raises(RuntimeError) as excinfo:
            worker.submit(lambda *_args, **_kwargs: None)
        assert excinfo.value is primary
        assert any(
            "delegated-fatal teardown failed: RuntimeError: close-failed" in note
            for note in getattr(primary, "__notes__", [])
        )
        with pytest.raises(RuntimeError) as replay:
            worker.close()
        assert replay.value is close_error

    def test_waiter_published_close_failure_keeps_published_error_and_replays_close(self):
        worker = Worker(level=3, num_sub_workers=0)
        published = RuntimeError("published-wait")
        close_error = RuntimeError("close-failed")
        worker._latch_delegated_session_fatal(published)

        def _close():
            prior = worker._close_completion
            if prior is not None and prior.done:
                if prior.error is not None:
                    raise prior.error
                return
            attempt = _CloseAttempt()
            worker._close_completion = attempt
            attempt.publish(close_error, False)
            raise close_error

        worker.close = _close
        worker._wait_run_handle = lambda *_args, **_kwargs: True
        worker._finalize_run_handle = lambda *_args, **_kwargs: published
        handle = RunHandle(worker, run_id=1, keepalive=())
        with pytest.raises(RuntimeError) as excinfo:
            handle.wait()
        assert excinfo.value is published
        assert any(
            "delegated-fatal teardown failed: RuntimeError: close-failed" in note
            for note in getattr(published, "__notes__", [])
        )
        with pytest.raises(RuntimeError) as replay:
            worker.close()
        assert replay.value is close_error

    def test_teardown_without_primary_propagates_close_failure(self):
        worker = Worker(level=3, num_sub_workers=0)
        close_error = RuntimeError("close-failed")
        worker._latch_delegated_session_fatal(RuntimeError("session-fatal"))

        def _close():
            prior = worker._close_completion
            if prior is not None and prior.done:
                if prior.error is not None:
                    raise prior.error
                return
            attempt = _CloseAttempt()
            worker._close_completion = attempt
            attempt.publish(close_error, False)
            raise close_error

        worker.close = _close
        with pytest.raises(RuntimeError) as excinfo:
            worker._teardown_delegated_fatal_if_safe()
        assert excinfo.value is close_error
        with pytest.raises(RuntimeError) as replay:
            worker.close()
        assert replay.value is close_error

    def test_chip_teardown_only_sweeps_store(self):
        source = _WORKER_PY.read_text(encoding="utf-8")
        assert "provider_transaction_table = ProviderTransactionTable()" in source
        terminal = "_handle_ctrl_delegated_region_terminal(buf, provider_transaction_table, provider_region_store)"
        assert terminal in source
        teardown = source.split("def _teardown_chip_process_resources(")[1].split("\ndef ")[0]
        assert "provider_region_store.sweep()" in teardown
        assert "ProviderTransactionTable" not in teardown
        assert "table.execute" not in teardown


class _SetupOwnedWorker:
    def __init__(self, name: str, *, fail: BaseException | None = None) -> None:
        self.name = name
        self.fail = fail
        self.closes: list[str] = []

    def close(self) -> None:
        self.closes.append(self.name)
        if self.fail is not None:
            raise self.fail


class TestSetupCleanup:
    def test_register_setup_failure_closes_constructed_worker_and_reraises_primary(self):
        primary = RuntimeError("register failed")
        worker = _SetupOwnedWorker("l3")
        close_owned_workers(primary, worker)
        assert worker.closes == ["l3"]
        with pytest.raises(RuntimeError, match="register failed"):
            raise primary

    def test_add_worker_setup_failure_closes_l4_then_l3_and_keeps_primary(self):
        primary = RuntimeError("add_worker failed")
        l4 = _SetupOwnedWorker("l4")
        l3 = _SetupOwnedWorker("l3")
        close_owned_workers(primary, l4, l3)
        assert l4.closes == ["l4"]
        assert l3.closes == ["l3"]
        with pytest.raises(RuntimeError, match="add_worker failed"):
            raise primary

    def test_init_setup_failure_attaches_cleanup_note_and_keeps_primary(self):
        primary = RuntimeError("init failed")
        close_error = RuntimeError("close failed")
        worker = _SetupOwnedWorker("l4", fail=close_error)
        l3 = _SetupOwnedWorker("l3")
        close_owned_workers(primary, worker, l3)
        assert worker.closes == ["l4"]
        assert l3.closes == ["l3"]
        assert any("RuntimeError: close failed" in note for note in getattr(primary, "__notes__", []))
        with pytest.raises(RuntimeError, match="init failed"):
            raise primary
        assert primary is not close_error

    def test_attach_exception_note_without_add_note(self):
        class _LegacyError(RuntimeError):
            add_note = None

        primary = _LegacyError("init failed")
        attach_exception_note(primary, "RuntimeError: close failed")
        assert getattr(primary, "__notes__", []) == ["RuntimeError: close failed"]


class TestDelegatedIntegration:
    def test_canonical_control_module_does_not_keep_transition_path(self):
        delegated = _REPO_ROOT / "python" / "simpler" / "comm_delegated_region_control.py"
        canonical = _REPO_ROOT / "python" / "simpler" / "comm_provider_control.py"
        assert canonical.is_file()
        assert not delegated.exists()
        tree = ast.parse(canonical.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and "comm_delegated_region_control" in node.module:
                raise AssertionError("canonical control module must not import the transition path")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "comm_delegated_region_control" not in alias.name

    def test_compatibility_entry_uses_unified_materializer(self):
        tree = ast.parse(_WORKER_PY.read_text(encoding="utf-8"))
        create = None
        materialize = None
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == "Worker":
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "_create_worker_chip_region":
                        create = item
                    if isinstance(item, ast.FunctionDef) and item.name == "_materialize_region_instance":
                        materialize = item
                    assert not isinstance(item, ast.FunctionDef) or item.name not in {
                        "_create_delegated_worker_chip_region",
                        "_admitted_delegated_region_context",
                    }
        assert create is not None and materialize is not None

        def _names(node):
            found = set()
            for item in ast.walk(node):
                if isinstance(item, ast.Name):
                    found.add(item.id)
                elif isinstance(item, ast.Attribute):
                    found.add(item.attr)
            return found

        create_names = _names(create)
        materialize_names = _names(materialize)
        assert "_materialize_region_instance" in create_names
        assert "materialize_region_instance" in materialize_names
        assert "materialize_delegated_region_instance" not in create_names
        assert "materialize_delegated_region_instance" not in materialize_names
        assert callable(Worker._dispatch_delegated_allocate)
        assert callable(Worker._dispatch_delegated_release)

    def test_runtime_helpers_do_not_call_retired_region_handlers(self):
        tree = ast.parse(_WORKER_PY.read_text(encoding="utf-8"))
        names = {
            "_forward_delegated_region",
            "_handle_ctrl_delegated_region_hop",
            "_handle_ctrl_delegated_region_terminal",
            "_dispatch_delegated_allocate",
            "_dispatch_delegated_release",
            "_create_worker_chip_region",
            "_materialize_region_instance",
        }
        forbidden = {"handle_ctrl_region_allocate", "handle_ctrl_region_release"}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in names:
                used = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
                assert not (used & forbidden), node.name
