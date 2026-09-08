# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Managed transfer lifetimes, admission, and real fork/control round trips."""

import ctypes
import multiprocessing
import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest
import simpler.external_transfer as transfers
import simpler.worker as worker_mod
from simpler import ExternalBufferRange, ExternalTransferResult, Worker, register_external_transfer_provider
from simpler.buffer import AccessMode, CanonicalIdentity, create_host_shared_buffer, wrap_device_malloc
from simpler.external_transfer import ExternalTransferSubmissionError, ExternalTransferUnconfirmedError
from simpler.task_interface import ChipWorker, DataType

from ._harness import fake_chip_l3, requires_sim_binaries


@pytest.fixture(autouse=True)
def provider_registry(monkeypatch):
    monkeypatch.setattr(transfers, "_providers", {})


@pytest.fixture
def local_worker():
    worker = Worker(level=3, device_ids=[0, 1])
    worker._lifecycle = worker_mod._Lifecycle.READY
    worker._chip_shms = [object(), object()]
    block = ctypes.create_string_buffer(64)
    buffer = wrap_device_malloc(ctypes.addressof(block), 64, worker._owner_instance_id, 1, "L3", owner_worker_id=0)
    worker._record_device_alloc(buffer)
    worker._external_transfer_providers = {"test": lambda request, token: ExternalTransferResult()}
    worker._external_transfers = transfers._ExternalTransferPool(2, 64, worker._retire_external_transfer)

    def control(_kind, worker_id, opcode, payload, timeout):
        assert opcode == worker_mod._CTRL_EXTERNAL_TRANSFER
        assert worker_id == 0
        assert timeout == 30.0
        transfers._start_chip_transfer(
            payload,
            7,
            worker._external_transfer_providers,
            worker._external_transfers.signals,
            lambda: None,
        )
        return payload

    worker._worker = SimpleNamespace(control_payload=control, free=lambda *_args: None)
    yield worker, buffer, block
    # Only host-backed fake providers run here; tests join them before this fixture returns.
    for handle in tuple(worker._external_transfers.pending.values()):
        handle._shm.close()
        handle._shm.unlink()
    worker._external_transfers.pending.clear()
    worker._chip_shms = []
    worker._worker = None
    worker.close()


def _range(buffer, offset=0, nbytes=8, access=AccessMode.WRITE):
    return ExternalBufferRange(buffer, offset, nbytes, access)


def test_registration_and_snapshot(local_worker):
    worker, _, _ = local_worker

    def provider(_request, _token):
        return ExternalTransferResult()

    register_external_transfer_provider("one", provider)
    register_external_transfer_provider("one", provider)
    with pytest.raises(ValueError, match="already registered"):
        register_external_transfer_provider("one", lambda request, token: ExternalTransferResult())
    worker._snapshot_external_transfers()
    register_external_transfer_provider("late", provider)
    assert set(worker._external_transfer_providers) == {"one"}


@pytest.mark.parametrize("level", [2, 4])
def test_rejects_unsupported_levels(level):
    with pytest.raises(TypeError, match="level-3"):
        Worker(level=level).submit_external_transfer("test", ())


@pytest.mark.parametrize("offset,nbytes", [(-1, 8), (0, 0), (0, -1), (0.5, 8), (0, True)])
def test_range_validation(local_worker, offset, nbytes):
    _, buffer, _ = local_worker
    with pytest.raises(ValueError):
        _range(buffer, offset, nbytes)


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_control_timeout_validation(local_worker, timeout):
    worker, buffer, _ = local_worker
    with pytest.raises(ValueError, match="positive and finite"):
        worker.submit_external_transfer("test", (_range(buffer),), timeout_s=timeout)
    assert not worker._external_transfers.pending


def test_private_snapshot_defeats_mutated_pointer_and_extent(local_worker):
    worker, buffer, block = local_worker
    original = buffer.base
    observed = []

    def provider(request, _token):
        observed.append(request)
        ctypes.memset(request.buffers[0].address, 65, request.buffers[0].nbytes)
        return ExternalTransferResult(payload=b"written")

    worker._external_transfer_providers["test"] = provider
    buffer.base, buffer.nbytes, buffer.owner_worker_id = 1, 1024, 1
    handle = worker.submit_external_transfer("test", (_range(buffer, 4, 8),))
    assert handle.result(5) == b"written"
    assert handle.done()
    assert not handle.request_cancel()
    assert observed[0].buffers[0].address == original + 4
    assert observed[0].device_id == 7
    assert bytes(block)[4:12] == b"A" * 8
    with pytest.raises(ValueError, match="overruns"):
        worker.submit_external_transfer("test", (_range(buffer, 60, 8),))


def test_stale_foreign_closed_and_read_only_buffers(local_worker):
    worker, buffer, _ = local_worker
    with pytest.raises(ValueError, match="not a live"):
        identity = CanonicalIdentity(worker._owner_instance_id, 99, 1)
        worker.submit_external_transfer("test", (_range(replace(buffer, identity=identity)),))
    readonly = replace(buffer, access=AccessMode.READ)
    worker._record_device_alloc(readonly)
    with pytest.raises(ValueError, match="needs WRITE"):
        worker.submit_external_transfer("test", (_range(buffer),))
    worker._record_device_alloc(buffer)
    buffer.close()
    with pytest.raises(ValueError, match="closed"):
        worker.submit_external_transfer("test", (_range(buffer),))


def test_cancel_and_timeout_keep_memory_and_compute_excluded(local_worker):
    worker, buffer, _ = local_worker
    entered, finish = threading.Event(), threading.Event()

    def provider(_request, token):
        entered.set()
        assert finish.wait(5)
        return ExternalTransferResult(succeeded=False, error=f"cancelled={token.requested}")

    worker._external_transfer_providers["test"] = provider
    handle = worker.submit_external_transfer("test", (_range(buffer),))
    try:
        assert entered.wait(5)
        assert handle.request_cancel()
        assert not handle.done()
        with pytest.raises(TimeoutError):
            handle.wait(0.01)
        with pytest.raises(RuntimeError, match="external transfer"):
            worker.free(buffer)
        with pytest.raises(RuntimeError, match="external transfer"):
            worker.submit(lambda *_args: None)
        host = create_host_shared_buffer(8, worker._owner_instance_id, 99)
        try:
            with pytest.raises(RuntimeError, match="external transfer"):
                worker.copy_to(buffer, host)
        finally:
            host.close()
        assert buffer.identity in worker._child_alloc
    finally:
        finish.set()
        with pytest.raises(RuntimeError, match="cancelled=True"):
            handle.wait(5)
    assert handle.done()
    worker.free(buffer)
    assert buffer.identity not in worker._child_alloc


def test_backpressure_and_overlapping_writes(local_worker):
    worker, buffer, _ = local_worker
    finish = threading.Event()

    def provider(_request, _token):
        assert finish.wait(5)
        return ExternalTransferResult()

    worker._external_transfer_providers["test"] = provider
    first = worker.submit_external_transfer("test", (_range(buffer, nbytes=32),))
    second = None
    try:
        with pytest.raises(RuntimeError, match="conflicts"):
            worker.submit_external_transfer("test", (_range(buffer),))
        with pytest.raises(RuntimeError, match="byte limit"):
            worker.submit_external_transfer("test", (_range(buffer, nbytes=64),))
        second = worker.submit_external_transfer("test", (_range(buffer, 32, 8),))
        with pytest.raises(RuntimeError, match="count limit"):
            worker.submit_external_transfer("test", (_range(buffer, 48, 8),))
    finally:
        finish.set()
        first.wait(5)
        if second:
            second.wait(5)


def test_submission_ack_loss_retains_handle_and_late_completion(local_worker):
    worker, buffer, _ = local_worker
    control = worker._worker.control_payload

    def lost_ack(*args):
        control(*args)
        raise TimeoutError("ack lost")

    worker._worker.control_payload = lost_ack
    with pytest.raises(ExternalTransferSubmissionError) as caught:
        worker.submit_external_transfer("test", (_range(buffer),))
    assert worker._external_transfers.pending
    assert caught.value.handle.wait(5) == b""
    assert not worker._external_transfers.pending


def test_interrupted_completion_can_be_observed_again(local_worker, monkeypatch):
    worker, buffer, _ = local_worker
    handle = worker.submit_external_transfer("test", (_range(buffer),))
    observe = handle._observe_outcome

    def interrupt():
        raise KeyboardInterrupt("after wakeup")

    monkeypatch.setattr(handle, "_observe_outcome", interrupt)
    with pytest.raises(KeyboardInterrupt):
        handle.wait(5)
    assert worker._external_transfers.pending
    monkeypatch.setattr(handle, "_observe_outcome", observe)
    assert handle.wait(5) == b""
    assert not worker._external_transfers.pending


def test_interrupted_retirement_can_be_retried(local_worker, monkeypatch):
    worker, buffer, _ = local_worker
    handle = worker.submit_external_transfer("test", (_range(buffer),))
    retire = handle._pool.retire

    def interrupt(_handle):
        raise KeyboardInterrupt("after shared memory cleanup")

    monkeypatch.setattr(handle._pool, "retire", interrupt)
    with pytest.raises(KeyboardInterrupt):
        handle.wait(5)
    assert worker._external_transfers.pending
    monkeypatch.setattr(handle._pool, "retire", retire)
    assert handle.wait(5) == b""
    assert not worker._external_transfers.pending


def test_overlapping_reads_allowed_but_duplicate_writes_rejected(local_worker):
    worker, buffer, _ = local_worker
    with pytest.raises(RuntimeError, match="conflicts"):
        worker.submit_external_transfer("test", (_range(buffer), _range(buffer)))
    read = _range(buffer, access=AccessMode.READ)
    first = worker.submit_external_transfer("test", (read,))
    second = worker.submit_external_transfer("test", (read,))
    assert first.wait(5) == second.wait(5) == b""


def test_mixed_chip_request_rejected(local_worker):
    worker, buffer, _ = local_worker
    other = wrap_device_malloc(4096, 16, worker._owner_instance_id, 2, "L3", owner_worker_id=1)
    worker._record_device_alloc(other)
    with pytest.raises(ValueError, match="exactly one chip"):
        worker.submit_external_transfer("test", (_range(buffer), _range(other)))
    assert not worker._external_transfers.pending


def test_unexpected_exception_does_not_certify_quiescence(local_worker):
    worker, buffer, _ = local_worker

    def provider(_request, _token):
        raise RuntimeError("backend lost")

    worker._external_transfer_providers["test"] = provider
    handle = worker.submit_external_transfer("test", (_range(buffer),))
    with pytest.raises(ExternalTransferUnconfirmedError, match="backend lost"):
        handle.wait(5)
    assert not handle.done()
    assert worker._external_transfers.pending
    with pytest.raises(RuntimeError, match="external transfer"):
        worker.free(buffer)


@requires_sim_binaries
def test_forked_chip_direct_transfer_target_and_slot_reuse(monkeypatch):
    def provider(request, _token):
        span = request.buffers[0]
        ctypes.memset(span.address, request.payload[0], span.nbytes)
        return ExternalTransferResult(payload=str(request.device_id).encode())

    register_external_transfer_provider("fill", provider)
    with fake_chip_l3(monkeypatch, device_ids=(3, 5), external_transfer_max_pending=1) as worker:
        buffer = worker.alloc_child_tensor(1, (16,), DataType.UINT8)
        for value in (65, 66, 67):
            handle = worker.submit_external_transfer("fill", (_range(buffer, 4, 8),), bytes([value]))
            assert handle.wait(5) == b"5"
            output = worker.create_buffer(16)
            worker.copy_from(output, buffer)
            assert bytes(output.shm.buf)[4:12] == bytes([value]) * 8
            worker.release_buffer(output)
        worker.free(buffer)


@requires_sim_binaries
def test_close_timeout_defers_teardown_and_can_retry(monkeypatch):
    finish = multiprocessing.get_context("fork").Event()

    def provider(_request, _token):
        assert finish.wait(10)
        return ExternalTransferResult()

    register_external_transfer_provider("blocked", provider)
    with fake_chip_l3(monkeypatch) as worker:
        buffer = worker.alloc_child_tensor(0, (16,), DataType.UINT8)
        handle = worker.submit_external_transfer("blocked", (_range(buffer),))
        try:
            with monkeypatch.context() as context:
                context.setattr(worker_mod, "_ROLLBACK_GRACEFUL_TIMEOUT_S", 0.02)
                with pytest.raises(TimeoutError, match="buffers remain retained"):
                    worker.close()
            assert worker._lifecycle == worker_mod._Lifecycle.CLOSED
            assert not worker._teardown_attempted
            assert worker._worker is not None
            assert buffer.identity in worker._child_alloc
        finally:
            finish.set()
            handle.wait(5)
        worker.close()
        assert worker._worker is None


@requires_sim_binaries
def test_close_drains_unobserved_completed_transfers(monkeypatch):
    register_external_transfer_provider("done", lambda request, token: ExternalTransferResult())
    with fake_chip_l3(monkeypatch) as worker:
        buffer = worker.alloc_child_tensor(0, (8,), DataType.UINT8)
        handle = worker.submit_external_transfer("done", (_range(buffer),))
        worker.close()
        assert handle.done()
        assert not worker._external_transfers.pending


def test_native_thread_binding_requires_initialization():
    with pytest.raises(RuntimeError, match="not initialized"):
        ChipWorker()._bind_external_transfer_thread()


@requires_sim_binaries
def test_native_simulator_transfer():
    def provider(request, _token):
        span = request.buffers[0]
        ctypes.memset(span.address, 90, span.nbytes)
        return ExternalTransferResult()

    register_external_transfer_provider("native-sim", provider)
    worker = Worker(level=3, device_ids=[0], platform="a2a3sim", runtime="tensormap_and_ringbuffer")
    try:
        worker.init()
        buffer = worker.alloc_child_tensor(0, (16,), DataType.UINT8)
        worker.submit_external_transfer("native-sim", (_range(buffer, 4, 8),)).wait(5)
        output = worker.create_buffer(16)
        worker.copy_from(output, buffer)
        assert bytes(output.shm.buf)[4:12] == b"Z" * 8
        worker.release_buffer(output)
        worker.free(buffer)
    finally:
        worker.close()


@pytest.mark.requires_hardware
def test_onboard_hbm_transfer(st_platform, st_device_ids):
    def provider(request, _token):
        acl = ctypes.CDLL("libascendcl.so")
        get_device = acl.aclrtGetDevice
        get_device.argtypes, get_device.restype = [ctypes.POINTER(ctypes.c_int32)], ctypes.c_int32
        current = ctypes.c_int32(-1)
        status = get_device(ctypes.byref(current))
        if status != 0 or current.value != request.device_id:
            return ExternalTransferResult(False, error=f"thread device mismatch: {status}, {current.value}")
        memset = acl.aclrtMemset
        memset.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int32, ctypes.c_size_t]
        memset.restype = ctypes.c_int32
        span = request.buffers[0]
        status = memset(span.address, span.nbytes, 90, span.nbytes)
        return ExternalTransferResult(status == 0, error=f"aclrtMemset status={status}")

    register_external_transfer_provider("onboard", provider)
    worker = Worker(level=3, device_ids=st_device_ids, platform=st_platform, runtime="tensormap_and_ringbuffer")
    try:
        worker.init()
        buffer = worker.alloc_child_tensor(0, (4096,), DataType.UINT8)
        worker.submit_external_transfer("onboard", (_range(buffer, 256, 1024),)).wait(10)
        output = worker.create_buffer(4096)
        worker.copy_from(output, buffer)
        assert bytes(output.shm.buf)[256:1280] == b"Z" * 1024
        worker.release_buffer(output)
        worker.free(buffer)
    finally:
        worker.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
