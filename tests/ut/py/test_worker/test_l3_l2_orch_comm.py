# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import ctypes
import threading
import time
from multiprocessing.shared_memory import SharedMemory
from typing import Optional

import pytest
from simpler import l3_l2_orch_comm
from simpler.l3_l2_orch_comm import (
    NotifyOp,
    SignalTestResult,
    WaitCmp,
)
from simpler.task_interface import DataType, Tensor
from simpler.worker import (
    _IDLE,
    _OFF_STATE,
    Worker,
    _buffer_field_addr,
    _mailbox_store_i32,
)

from simpler_setup.runtime_builder import RuntimeBuilder


class _FakeDirectCWorker:
    def __init__(
        self,
        *,
        payload_base: int = 0xDEAD_0000,
        access_profile: int = int(l3_l2_orch_comm.L3L2RegionAccessProfile.SIM_POSIX_SHM),
        device_id: int = 0,
        export_key: bytes = b"",
        magic_version: int = 0x4C334C3200020000,
        region_id: Optional[int] = None,
        mapping_bytes: Optional[int] = None,
    ):
        self.create_calls: list[tuple[int, str, str]] = []
        self.release_calls: list[tuple[int, int]] = []
        self.next_region_id = 1
        self.payload_base = int(payload_base)
        self.access_profile = int(access_profile)
        self.device_id = int(device_id)
        self.export_key = bytes(export_key)
        self.magic_version = int(magic_version)
        self.region_id = region_id
        self.mapping_bytes = mapping_bytes

    def control_l3_l2_region_create(self, worker_id: int, request_shm_name: str, reply_shm_name: str) -> None:
        self.create_calls.append((int(worker_id), str(request_shm_name), str(reply_shm_name)))
        req_shm = SharedMemory(name=request_shm_name)
        reply_shm = SharedMemory(name=reply_shm_name)
        req_buf = req_shm.buf
        reply_buf = reply_shm.buf
        assert req_buf is not None
        assert reply_buf is not None
        try:
            req = l3_l2_orch_comm._REGION_CREATE_REQUEST.unpack_from(req_buf, 0)
            payload_bytes = int(req[2])
            counter_bytes = int(req[3])
            counter_offset = ((payload_bytes + 63) // 64) * 64
            region_id = int(self.region_id) if self.region_id is not None else self.next_region_id
            if self.region_id is None:
                self.next_region_id += 1
            backing_name = f"sim-direct-{region_id}".encode()
            export_key = self.export_key or f"acl-ipc-key-{region_id}".encode()
            if self.access_profile == int(l3_l2_orch_comm.L3L2RegionAccessProfile.ONBOARD_ACL_IPC):
                backing_name = b""
            else:
                export_key = b""
            l3_l2_orch_comm._REGION_CREATE_REPLY.pack_into(
                reply_buf,
                0,
                self.magic_version,
                region_id,
                self.payload_base,
                payload_bytes,
                self.payload_base + counter_offset,
                counter_bytes,
                self.access_profile,
                0,
                self.device_id,
                export_key + b"\x00" * (l3_l2_orch_comm.ACL_IPC_EXPORT_KEY_BYTES - len(export_key)),
                backing_name + b"\x00" * (l3_l2_orch_comm.CTRL_SHM_TOKEN_BYTES - len(backing_name)),
                counter_offset + counter_bytes if self.mapping_bytes is None else int(self.mapping_bytes),
            )
        finally:
            del req_buf
            del reply_buf
            req_shm.close()
            reply_shm.close()

    def control_l3_l2_region_release(self, worker_id: int, region_id: int) -> None:
        self.release_calls.append((int(worker_id), int(region_id)))


class _EndpointFailingOrch:
    def _clear_error(self) -> None:
        pass

    def _scope_begin(self) -> None:
        pass

    def _scope_end(self) -> None:
        pass

    def _drain(self) -> None:
        raise RuntimeError(
            "child failed: L3-L2 endpoint error op=signal_wait kind=3 region=2 "
            "counter_addr=0x200000 counter_operand=7 observed_counter=0 msg=wait timed out"
        )


def _make_started_sim_worker() -> tuple[Worker, SharedMemory, _FakeDirectCWorker]:
    worker = Worker(level=3, device_ids=[0], platform="a2a3sim", runtime="tensormap_and_ringbuffer")
    shm = SharedMemory(create=True, size=4096)
    assert shm.buf is not None
    _mailbox_store_i32(_buffer_field_addr(shm.buf, _OFF_STATE), _IDLE)
    fake_c_worker = _FakeDirectCWorker()
    worker._initialized = True
    worker._hierarchical_started = True
    worker._worker = fake_c_worker
    worker._chip_shms = [shm]
    return worker, shm, fake_c_worker


def _make_started_onboard_worker(
    platform: str = "a2a3", export_key: bytes = b""
) -> tuple[Worker, SharedMemory, _FakeDirectCWorker]:
    worker = Worker(level=3, device_ids=[2], platform=platform, runtime="tensormap_and_ringbuffer")
    shm = SharedMemory(create=True, size=4096)
    assert shm.buf is not None
    _mailbox_store_i32(_buffer_field_addr(shm.buf, _OFF_STATE), _IDLE)
    fake_c_worker = _FakeDirectCWorker(
        access_profile=int(l3_l2_orch_comm.L3L2RegionAccessProfile.ONBOARD_ACL_IPC),
        device_id=2,
        export_key=export_key,
    )
    worker._initialized = True
    worker._hierarchical_started = True
    worker._worker = fake_c_worker
    worker._chip_shms = [shm]
    return worker, shm, fake_c_worker


def test_sim_direct_region_uses_lifecycle_control_and_l3_host_metadata(monkeypatch):
    worker, shm, fake_c_worker = _make_started_sim_worker()
    calls: list[tuple] = []
    try:
        monkeypatch.setattr(
            l3_l2_orch_comm,
            "_l3_host_mapped_region_import_sim",
            lambda token, mapping_bytes: calls.append(("import", token, mapping_bytes)) or 99,
        )
        monkeypatch.setattr(
            l3_l2_orch_comm,
            "_l3_host_mapped_payload_write",
            lambda handle, offset, src, nbytes: calls.append(("write", handle, offset, src, nbytes)),
        )
        monkeypatch.setattr(
            l3_l2_orch_comm,
            "_l3_host_mapped_payload_read",
            lambda handle, offset, dst, nbytes: calls.append(("read", handle, offset, dst, nbytes)),
        )
        monkeypatch.setattr(
            l3_l2_orch_comm,
            "_l3_host_mapped_counter_notify",
            lambda handle, offset, value, op: calls.append(("notify", handle, offset, value, op)),
        )
        monkeypatch.setattr(
            l3_l2_orch_comm,
            "_l3_host_mapped_counter_test",
            lambda handle, offset, value, cmp: (calls.append(("test", handle, offset, value, cmp)) or (True, 7)),
        )

        region = worker._create_l3_l2_region(0, 64, 128)
        payload = Tensor.make(0x1234_0000, (16,), DataType.UINT8)
        region.payload_write(0, payload, nbytes=8)
        region.payload_read(8, payload, nbytes=8)
        result = region.counter(64).test(7, WaitCmp.EQ)
        region.counter(64).notify(3, NotifyOp.Set)

        assert len(fake_c_worker.create_calls) == 1
        assert region.descriptor_scalars() == [0x4C334C3200020000, 1, 0xDEAD_0000, 64, 0xDEAD_0040, 128]
        assert 99 not in region.descriptor_scalars()
        l3_host_mapping = region._l3_host_mapping
        assert l3_host_mapping is not None
        assert l3_host_mapping.handle != region.descriptor.payload_base
        assert l3_host_mapping.counter_offset == 64
        assert calls[0] == ("import", "sim-direct-1", 192)
        assert calls[1][0:3] == ("write", 99, 0)
        assert calls[2][0:3] == ("read", 99, 8)
        assert calls[3] == ("test", 99, 128, 7, int(WaitCmp.EQ))
        assert calls[4] == ("notify", 99, 128, 3, int(NotifyOp.Set))
        assert result == SignalTestResult(matched=True, observed=7)
    finally:
        worker._close_l3_l2_orch_comm()
        shm.close()
        shm.unlink()


def test_onboard_direct_region_imports_acl_ipc_and_uses_l3_host_metadata(monkeypatch):
    worker, shm, fake_c_worker = _make_started_onboard_worker()
    calls: list[tuple] = []
    try:
        monkeypatch.setattr(
            l3_l2_orch_comm,
            "_l3_host_mapped_region_import_onboard",
            lambda device_id, export_key, mapping_bytes, enable_peer_access: calls.append(
                ("import_onboard", device_id, export_key, mapping_bytes, enable_peer_access)
            )
            or 123,
        )
        monkeypatch.setattr(
            l3_l2_orch_comm,
            "_l3_host_mapped_counter_notify",
            lambda handle, offset, value, op: calls.append(("notify", handle, offset, value, op)),
        )

        region = worker._create_l3_l2_region(0, 64, 128)
        region.counter(64).notify(9, NotifyOp.Set)

        assert len(fake_c_worker.create_calls) == 1
        assert region.descriptor_scalars() == [0x4C334C3200020000, 1, 0xDEAD_0000, 64, 0xDEAD_0040, 128]
        assert 123 not in region.descriptor_scalars()
        l3_host_mapping = region._l3_host_mapping
        assert l3_host_mapping is not None
        assert l3_host_mapping.access_profile == l3_l2_orch_comm.L3L2RegionAccessProfile.ONBOARD_ACL_IPC
        assert l3_host_mapping.counter_offset == 64
        assert calls[0] == ("import_onboard", 2, b"acl-ipc-key-1", 192, True)
        assert calls[1] == ("notify", 123, 128, 9, int(NotifyOp.Set))
    finally:
        worker._close_l3_l2_orch_comm()
        shm.close()
        shm.unlink()


def test_a5_onboard_direct_region_import_uses_acl_ipc_without_peer_access(monkeypatch):
    worker, shm, _fake_c_worker = _make_started_onboard_worker(platform="a5", export_key=b"acl-\xff-key")
    calls: list[tuple] = []
    try:
        monkeypatch.setattr(
            l3_l2_orch_comm,
            "_l3_host_mapped_region_import_onboard",
            lambda device_id, export_key, mapping_bytes, enable_peer_access: calls.append(
                ("import_onboard", device_id, export_key, mapping_bytes, enable_peer_access)
            )
            or 123,
        )

        worker._create_l3_l2_region(0, 64, 128)

        assert calls[0] == ("import_onboard", 2, b"acl-\xff-key", 192, False)
    finally:
        worker._close_l3_l2_orch_comm()
        shm.close()
        shm.unlink()


def test_sim_direct_create_import_failure_rolls_back_l2_host_region(monkeypatch):
    worker, shm, fake_c_worker = _make_started_sim_worker()
    try:
        monkeypatch.setattr(
            l3_l2_orch_comm,
            "_l3_host_mapped_region_import_sim",
            lambda _token, _mapping_bytes: (_ for _ in ()).throw(RuntimeError("import failed")),
        )

        with pytest.raises(RuntimeError, match="import failed"):
            worker._create_l3_l2_region(0, 64, 128)

        assert fake_c_worker.create_calls
        assert fake_c_worker.release_calls == [(0, 1)]
        assert worker._live_l3_l2_regions == []
    finally:
        worker._close_l3_l2_orch_comm()
        shm.close()
        shm.unlink()


def test_direct_create_decode_failure_rolls_back_l2_host_region():
    worker, shm, fake_c_worker = _make_started_sim_worker()
    fake_c_worker.access_profile = 99
    try:
        with pytest.raises(ValueError, match="99 is not a valid"):
            worker._create_l3_l2_region(0, 64, 128)

        assert fake_c_worker.release_calls == [(0, 1)]
        assert worker._live_l3_l2_regions == []
    finally:
        worker._close_l3_l2_orch_comm()
        shm.close()
        shm.unlink()


@pytest.mark.parametrize(
    ("reply_updates", "match"),
    [
        ({"magic_version": 0xBAD}, "magic_version is invalid"),
        ({"region_id": 0}, "region_id must be nonzero"),
        (
            {"access_profile": int(l3_l2_orch_comm.L3L2RegionAccessProfile.ONBOARD_ACL_IPC)},
            "access_profile must be sim_posix_shm",
        ),
    ],
)
def test_direct_create_validation_failure_rolls_back_l2_host_region(reply_updates, match):
    worker, shm, fake_c_worker = _make_started_sim_worker()
    for name, value in reply_updates.items():
        setattr(fake_c_worker, name, value)
    expected_region_id = int(reply_updates.get("region_id", 1))
    try:
        with pytest.raises(RuntimeError, match=match):
            worker._create_l3_l2_region(0, 64, 128)

        assert fake_c_worker.release_calls == ([(0, expected_region_id)] if expected_region_id else [])
        assert worker._live_l3_l2_regions == []
    finally:
        worker._close_l3_l2_orch_comm()
        shm.close()
        shm.unlink()


def test_onboard_direct_mapping_bytes_mismatch_rolls_back_l2_host_region(monkeypatch):
    worker, shm, fake_c_worker = _make_started_onboard_worker()
    fake_c_worker.mapping_bytes = 191
    calls: list[tuple] = []
    try:
        monkeypatch.setattr(
            l3_l2_orch_comm,
            "_l3_host_mapped_region_import_onboard",
            lambda *args: calls.append(args) or 123,
        )

        with pytest.raises(RuntimeError, match="onboard_acl_ipc reply mapping_bytes"):
            worker._create_l3_l2_region(0, 64, 128)

        assert calls == []
        assert fake_c_worker.release_calls == [(0, 1)]
        assert worker._live_l3_l2_regions == []
    finally:
        worker._close_l3_l2_orch_comm()
        shm.close()
        shm.unlink()


def test_l3_host_mapped_counter_wait_releases_gil_for_python_notifier():
    shm = SharedMemory(create=True, size=64)
    handle = 0
    notifier_error: list[BaseException] = []
    try:
        handle = l3_l2_orch_comm._l3_host_mapped_region_import_sim(shm.name, 64)

        def notify() -> None:
            try:
                time.sleep(0.05)
                l3_l2_orch_comm._l3_host_mapped_counter_notify(handle, 0, 1, int(NotifyOp.Set))
            except BaseException as exc:  # noqa: BLE001
                notifier_error.append(exc)

        thread = threading.Thread(target=notify)
        thread.start()
        status, error_kind, observed, matched, message = l3_l2_orch_comm._l3_host_mapped_counter_wait(
            handle, 0, 1, int(WaitCmp.EQ), 1_000_000_000
        )
        thread.join(timeout=1.0)

        assert not thread.is_alive()
        assert notifier_error == []
        assert (status, error_kind, observed, matched, message) == (0, 0, 1, True, "")
    finally:
        if handle:
            l3_l2_orch_comm._l3_host_mapped_region_close(handle)
        shm.close()
        shm.unlink()


def test_l3_host_mapped_sim_payload_and_counter_helpers_roundtrip():
    shm = SharedMemory(create=True, size=128)
    handle = 0
    try:
        handle = l3_l2_orch_comm._l3_host_mapped_region_import_sim(shm.name, 128)
        src_t = ctypes.c_uint8 * 8
        src = src_t(*range(10, 18))
        dst = src_t()

        l3_l2_orch_comm._l3_host_mapped_payload_write(handle, 16, ctypes.addressof(src), 8)
        l3_l2_orch_comm._l3_host_mapped_payload_read(handle, 16, ctypes.addressof(dst), 8)
        assert bytes(dst) == bytes(range(10, 18))

        l3_l2_orch_comm._l3_host_mapped_counter_notify(handle, 64, 3, int(NotifyOp.Set))
        assert l3_l2_orch_comm._l3_host_mapped_counter_test(handle, 64, 3, int(WaitCmp.EQ)) == (True, 3)
        l3_l2_orch_comm._l3_host_mapped_counter_notify(handle, 64, 4, int(NotifyOp.Add))
        assert l3_l2_orch_comm._l3_host_mapped_counter_test(handle, 64, 7, int(WaitCmp.GE)) == (True, 7)
        assert l3_l2_orch_comm._l3_host_mapped_counter_wait(handle, 64, 7, int(WaitCmp.EQ), 1_000_000) == (
            0,
            0,
            7,
            True,
            "",
        )

        l3_l2_orch_comm._l3_host_mapped_region_close(handle)
        with pytest.raises(RuntimeError, match="closed or unknown"):
            l3_l2_orch_comm._l3_host_mapped_payload_read(handle, 16, ctypes.addressof(dst), 8)
    finally:
        if handle:
            l3_l2_orch_comm._l3_host_mapped_region_close(handle)
        shm.close()
        shm.unlink()


def test_l3_host_mapped_region_close_makes_sim_handle_unusable():
    shm = SharedMemory(create=True, size=64)
    handle = 0
    try:
        handle = l3_l2_orch_comm._l3_host_mapped_region_import_sim(shm.name, 64)
        l3_l2_orch_comm._l3_host_mapped_region_close(handle)

        with pytest.raises(RuntimeError, match="closed or unknown"):
            l3_l2_orch_comm._l3_host_mapped_counter_test(handle, 0, 0, int(WaitCmp.EQ))
    finally:
        if handle:
            l3_l2_orch_comm._l3_host_mapped_region_close(handle)
        shm.close()
        shm.unlink()


def test_sim_direct_transfer_failure_poisons_only_region(monkeypatch):
    worker, shm, _fake_c_worker = _make_started_sim_worker()
    try:
        monkeypatch.setattr(l3_l2_orch_comm, "_l3_host_mapped_region_import_sim", lambda _token, _size: 55)
        monkeypatch.setattr(
            l3_l2_orch_comm,
            "_l3_host_mapped_payload_write",
            lambda _handle, _offset, _src, _nbytes: (_ for _ in ()).throw(RuntimeError("copy failed")),
        )

        region = worker._create_l3_l2_region(0, 64, 128)
        payload = Tensor.make(0x1234_0000, (16,), DataType.UINT8)
        with pytest.raises(RuntimeError, match="copy failed"):
            region.payload_write(0, payload, nbytes=8)
        with pytest.raises(RuntimeError, match="poisoned"):
            region.descriptor_scalars()
    finally:
        worker._close_l3_l2_orch_comm()
        shm.close()
        shm.unlink()


def test_sim_direct_cleanup_closes_l3_host_mapping_before_l2_host_release(monkeypatch):
    worker, shm, fake_c_worker = _make_started_sim_worker()
    events: list[tuple[str, int]] = []
    original_release = fake_c_worker.control_l3_l2_region_release

    def release(worker_id: int, region_id: int) -> None:
        events.append(("release", int(region_id)))
        original_release(worker_id, region_id)

    try:
        fake_c_worker.control_l3_l2_region_release = release
        monkeypatch.setattr(l3_l2_orch_comm, "_l3_host_mapped_region_import_sim", lambda _token, _size: 77)
        monkeypatch.setattr(
            l3_l2_orch_comm,
            "_l3_host_mapped_region_close",
            lambda handle: events.append(("close", int(handle))),
        )

        region = worker._create_l3_l2_region(0, 64, 128)
        region.free()
        worker._cleanup_l3_l2_regions()

        assert events == [("close", 77), ("release", 1)]
        with pytest.raises(RuntimeError, match="expired"):
            region.descriptor_scalars()
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
            region = orch_handle.create_l3_l2_region(worker_id=0, payload_bytes=16, counter_bytes=128)
            region.payload_write(0, host)
            for i in range(16):
                buf[i] = 0
            region.payload_read(0, host)
            assert bytes(buf) == bytes((i + 41) & 0xFF for i in range(16))

        worker.run(orch)
    finally:
        worker.close()


@pytest.mark.parametrize("platform", ["a2a3sim", "a5sim"])
def test_sim_worker_counter_wait_timeout_does_not_poison_region_and_free_is_idempotent(platform):
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
            region = orch_handle.create_l3_l2_region(worker_id=0, payload_bytes=16, counter_bytes=128)
            with pytest.raises(TimeoutError, match="observed=0"):
                region.counter(0).wait(1, WaitCmp.EQ, timeout=0.001)
            assert region.descriptor_scalars()[1] != 0
            region.free()
            region.free()

        worker.run(orch)
    finally:
        worker.close()
