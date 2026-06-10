# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import ctypes
import struct
from multiprocessing.shared_memory import SharedMemory

import pytest
from simpler.l3_l2_orch_comm import L3L2OrchCommCmd, L3L2OrchCommResponse, L3L2OrchRegionDesc
from simpler.task_interface import ContinuousTensor, DataType
from simpler.worker import (
    _IDLE,
    _OFF_STATE,
    _TASK_READY,
    Worker,
    _buffer_field_addr,
    _mailbox_store_i32,
)

from simpler_setup.runtime_builder import RuntimeBuilder


class _FakeCWorker:
    def __init__(self):
        self.bootstrap_calls: list[tuple[int, str]] = []

    def control_l3_l2_orch_comm_init(self, worker_id: int, control_shm_name: str) -> None:
        self.bootstrap_calls.append((int(worker_id), str(control_shm_name)))


class _FailingCWorker(_FakeCWorker):
    def control_l3_l2_orch_comm_init(self, worker_id: int, control_shm_name: str) -> None:
        super().control_l3_l2_orch_comm_init(worker_id, control_shm_name)
        raise RuntimeError("l3_l2_orch_comm_init is not supported by this platform/runtime")


class _EndpointFailingOrch:
    def _clear_error(self) -> None:
        pass

    def _scope_begin(self) -> None:
        pass

    def _scope_end(self) -> None:
        pass

    def _drain(self) -> None:
        raise RuntimeError(
            "child failed: L3-L2 endpoint error op=wait kind=3 region=2 seq=7 observed=0 msg=wait timed out"
        )


class _FakeClient:
    def __init__(self):
        self.requests = []
        self.next_region_id = 1

    def submit(self, request, timeout_s: float):
        self.requests.append((request, timeout_s))

        if request.cmd == L3L2OrchCommCmd.ALLOC_REGION:
            region_id = self.next_region_id
            self.next_region_id += 1
            return L3L2OrchCommResponse(
                status=0,
                error_kind=0,
                region_id=region_id,
                observed_signal=0,
                desc=L3L2OrchRegionDesc(
                    magic_version=0x4C334C3200010000,
                    region_id=region_id,
                    payload_base=0x100000 + region_id * 0x1000,
                    payload_bytes=request.nbytes,
                    l3_to_l2_signal_base=0x200000 + region_id * 0x1000,
                    l2_to_l3_signal_base=0x300000 + region_id * 0x1000,
                ),
                message="",
            )
        return L3L2OrchCommResponse(
            status=0,
            error_kind=0,
            region_id=request.region_id,
            observed_signal=request.seq,
            desc=None,
            message="",
        )


def _make_started_worker(mailbox_state: int = _IDLE) -> tuple[Worker, SharedMemory, _FakeCWorker, _FakeClient]:
    worker = Worker(level=3, device_ids=[0], platform="a2a3", runtime="tensormap_and_ringbuffer")
    shm = SharedMemory(create=True, size=4096)
    assert shm.buf is not None
    _mailbox_store_i32(_buffer_field_addr(shm.buf, _OFF_STATE), mailbox_state)
    fake_c_worker = _FakeCWorker()
    fake_client = _FakeClient()
    worker._initialized = True
    worker._hierarchical_started = True
    worker._worker = fake_c_worker
    worker._chip_shms = [shm]
    worker._make_l3_l2_orch_comm_client = lambda _shm: fake_client
    return worker, shm, fake_c_worker, fake_client


def test_first_region_bootstraps_and_second_region_reuses_ready_service():
    worker, shm, fake_c_worker, fake_client = _make_started_worker()
    try:
        first = worker._create_l3_l2_region(0, 128)
        _mailbox_store_i32(_buffer_field_addr(shm.buf, _OFF_STATE), _TASK_READY)
        second = worker._create_l3_l2_region(0, 256)

        assert len(fake_c_worker.bootstrap_calls) == 1
        assert first.descriptor_scalars()[1] == 1
        assert second.descriptor_scalars()[1] == 2
        assert [req.cmd for req, _timeout in fake_client.requests].count(1) == 2
    finally:
        worker._close_l3_l2_orch_comm()
        shm.close()
        shm.unlink()


def test_bootstrap_fails_immediately_when_worker_busy_and_service_not_ready():
    worker, shm, fake_c_worker, _fake_client = _make_started_worker(mailbox_state=_TASK_READY)
    try:
        with pytest.raises(RuntimeError, match="bootstrap.*busy"):
            worker._create_l3_l2_region(0, 128)
        assert fake_c_worker.bootstrap_calls == []
    finally:
        worker._close_l3_l2_orch_comm()
        shm.close()
        shm.unlink()


def test_bootstrap_unsupported_failure_leaves_no_partial_service_state():
    worker, shm, _fake_c_worker, _fake_client = _make_started_worker()
    failing_c_worker = _FailingCWorker()
    worker._worker = failing_c_worker
    try:
        for _ in range(2):
            with pytest.raises(RuntimeError, match="not supported"):
                worker._create_l3_l2_region(0, 128)
            assert worker._l3_l2_orch_comm_ready == set()
            assert worker._l3_l2_orch_comm_clients == {}
            assert worker._l3_l2_orch_comm_shms == {}
            assert worker._live_l3_l2_regions == []
        assert len(failing_c_worker.bootstrap_calls) == 2
    finally:
        worker._close_l3_l2_orch_comm()
        shm.close()
        shm.unlink()


def test_region_payload_and_signal_commands_use_service_client():
    worker, shm, _fake_c_worker, fake_client = _make_started_worker()
    try:
        region = worker._create_l3_l2_region(0, 16)

        payload = ContinuousTensor.make(0x1000, (16,), DataType.UINT8)
        worker._register_l3_l2_orch_comm_host_buffer(payload)

        region.payload_write(4, payload, nbytes=4)
        region.payload_read(8, payload, nbytes=4)
        region.notify(3)
        region.wait(3, timeout=0.001)

        assert [req.cmd for req, _timeout in fake_client.requests] == [
            L3L2OrchCommCmd.ALLOC_REGION,
            L3L2OrchCommCmd.PAYLOAD_WRITE,
            L3L2OrchCommCmd.PAYLOAD_READ,
            L3L2OrchCommCmd.SIGNAL_NOTIFY,
            L3L2OrchCommCmd.SIGNAL_WAIT,
        ]
    finally:
        worker._close_l3_l2_orch_comm()
        shm.close()
        shm.unlink()


def test_precommand_validation_failure_does_not_poison_region():
    worker, shm, _fake_c_worker, fake_client = _make_started_worker()
    try:
        region = worker._create_l3_l2_region(0, 4)
        with pytest.raises(ValueError, match="exceeds region size"):
            payload = ContinuousTensor.make(0x1000, (1,), DataType.UINT8)
            worker._register_l3_l2_orch_comm_host_buffer(payload)
            region.payload_write(8, payload)
        assert region.descriptor_scalars()[1] == 1
        assert len(fake_client.requests) == 1
    finally:
        worker._close_l3_l2_orch_comm()
        shm.close()
        shm.unlink()


def test_private_python_payload_buffer_fails_before_service_submission_without_poisoning():
    worker, shm, _fake_c_worker, fake_client = _make_started_worker()
    try:
        region = worker._create_l3_l2_region(0, 4)
        with pytest.raises(ValueError, match="ContinuousTensor.*orch.alloc"):
            region.payload_write(0, bytearray(struct.pack("<I", 0x12345678)))
        assert region.descriptor_scalars()[1] == 1
        assert len(fake_client.requests) == 1
    finally:
        worker._close_l3_l2_orch_comm()
        shm.close()
        shm.unlink()


def test_unregistered_continuous_tensor_fails_before_service_submission_without_poisoning():
    worker, shm, _fake_c_worker, fake_client = _make_started_worker()
    try:
        region = worker._create_l3_l2_region(0, 4)
        payload = ContinuousTensor.make(0x1000, (4,), DataType.UINT8)
        with pytest.raises(ValueError, match="not registered"):
            region.payload_write(0, payload)
        assert region.descriptor_scalars()[1] == 1
        assert len(fake_client.requests) == 1
    finally:
        worker._close_l3_l2_orch_comm()
        shm.close()
        shm.unlink()


def test_cleanup_expires_region_handles_after_physical_free():
    worker, shm, _fake_c_worker, fake_client = _make_started_worker()
    try:
        region = worker._create_l3_l2_region(0, 4)
        worker._cleanup_l3_l2_regions()
        with pytest.raises(RuntimeError, match="expired"):
            region.descriptor_scalars()

        assert [req.cmd for req, _timeout in fake_client.requests] == [
            L3L2OrchCommCmd.ALLOC_REGION,
            L3L2OrchCommCmd.FREE_REGION,
        ]
    finally:
        worker._close_l3_l2_orch_comm()
        shm.close()
        shm.unlink()


def test_endpoint_region_error_poisons_only_matching_live_region_during_drain():
    worker, shm, _fake_c_worker, fake_client = _make_started_worker()
    worker._orch = _EndpointFailingOrch()
    worker._start_hierarchical = lambda: None
    regions = []
    try:

        def orch(_orch_handle, _args, _cfg):
            regions.append(worker._create_l3_l2_region(0, 32))
            regions.append(worker._create_l3_l2_region(0, 64))

        with pytest.raises(RuntimeError, match="L3-L2 endpoint error.*region=2"):
            worker.run(orch)

        assert regions[0]._poisoned is False
        assert regions[1]._poisoned is True
        assert regions[0]._expired is True
        assert regions[1]._expired is True

        assert [req.cmd for req, _timeout in fake_client.requests] == [
            L3L2OrchCommCmd.ALLOC_REGION,
            L3L2OrchCommCmd.ALLOC_REGION,
            L3L2OrchCommCmd.FREE_REGION,
            L3L2OrchCommCmd.FREE_REGION,
        ]
    finally:
        worker._close_l3_l2_orch_comm()
        shm.close()
        shm.unlink()


@pytest.mark.parametrize("platform", ["a2a3sim", "a5sim"])
def test_sim_worker_region_payload_roundtrip(platform):
    try:
        RuntimeBuilder(platform=platform).get_binaries("tensormap_and_ringbuffer")
    except FileNotFoundError as e:
        pytest.skip(f"{platform} runtime binaries unavailable: {e}")

    worker = Worker(
        level=3,
        device_ids=[0],
        platform=platform,
        runtime="tensormap_and_ringbuffer",
        num_sub_workers=0,
    )
    worker.init()
    try:

        def orch(orch_handle, _args, _cfg):
            host = orch_handle.alloc([16], DataType.UINT8)
            buf_t = ctypes.c_uint8 * 16
            buf = buf_t.from_address(int(host.data))
            for i in range(16):
                buf[i] = (i + 41) & 0xFF
            region = orch_handle.create_l3_l2_region(worker_id=0, payload_bytes=16)
            region.payload_write(0, host)
            for i in range(16):
                buf[i] = 0
            region.payload_read(0, host)
            assert bytes(buf) == bytes((i + 41) & 0xFF for i in range(16))

        worker.run(orch)
    finally:
        worker.close()


@pytest.mark.parametrize("platform", ["a2a3sim", "a5sim"])
def test_sim_worker_wait_timeout_poisons_region_and_free_is_idempotent(platform):
    try:
        RuntimeBuilder(platform=platform).get_binaries("tensormap_and_ringbuffer")
    except FileNotFoundError as e:
        pytest.skip(f"{platform} runtime binaries unavailable: {e}")

    worker = Worker(
        level=3,
        device_ids=[0],
        platform=platform,
        runtime="tensormap_and_ringbuffer",
        num_sub_workers=0,
    )
    worker.init()
    try:

        def orch(orch_handle, _args, _cfg):
            region = orch_handle.create_l3_l2_region(worker_id=0, payload_bytes=16)
            with pytest.raises(RuntimeError, match="timed out"):
                region.wait(1, timeout=0.001)
            with pytest.raises(RuntimeError, match="poisoned"):
                region.descriptor_scalars()
            region.free()
            region.free()

        worker.run(orch)
    finally:
        worker.close()
