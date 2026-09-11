# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: PLC0415

import contextlib
import ctypes
import gc
import importlib
import os
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Any, Optional, cast
from unittest.mock import MagicMock

import pytest
from simpler import comm_endpoints as ce
from simpler import comm_region, worker_chip_orch_comm
from simpler import worker as worker_module
from simpler.buffer import (
    AccessMode,
    AddressSpace,
    BackendKind,
    BufferDescriptor,
    CanonicalIdentity,
    mint_owner_instance_id,
    wrap_fork_inherited,
)
from simpler.comm_provider import (
    ProviderReleaseResult,
    ProviderReleaseStatus,
    RegionAllocationResult,
    RegionControlError,
    RegionExportDescriptor,
    RegionPartKind,
    RegionPartLocalView,
)
from simpler.orchestrator import Orchestrator
from simpler.task_interface import DataType
from simpler.worker import (
    _IDLE,
    _OFF_STATE,
    MAILBOX_SIZE,
    Worker,
    _buffer_field_addr,
    _mailbox_store_i32,
)
from simpler.worker_chip_orch_comm import (
    NotifyOp,
    SignalTestResult,
    WaitCmp,
)

from simpler_setup.runtime_builder import RuntimeBuilder

_task_interface_ext = cast(Any, importlib.import_module("_task_interface"))
_TASK_INTERFACE_CPP = Path(__file__).resolve().parents[4] / "python" / "bindings" / "task_interface.cpp"
_WORKER_PY = Path(__file__).resolve().parents[4] / "python" / "simpler" / "worker.py"
_ACCESS_ONBOARD_VMM = 1
_ACCESS_SIM_POSIX_SHM = 2
_TEST_WALL_BUDGET_S = 30.0
_TEST_WALL_BUDGET_NS = int(_TEST_WALL_BUDGET_S * 1_000_000_000)
_FAKE_POSIX_OBJECT_SIZES: dict[str, int] = {}


@pytest.fixture(autouse=True)
def _fake_region_runtime_facts(monkeypatch):
    real_posix_object_size = worker_module._posix_object_size

    def _posix_object_size(token: str) -> int:
        recorded = _FAKE_POSIX_OBJECT_SIZES.get(str(token))
        if recorded is not None:
            return int(recorded)
        return real_posix_object_size(token)

    monkeypatch.setattr(worker_module, "_posix_object_size", _posix_object_size)
    monkeypatch.setattr(worker_module, "_region_vmm_granularity", lambda _device_id: 1)


def _vmm_shareable_body(*, device_id: int, shareable_handle: int, mapping_bytes: int) -> bytes:
    return (
        int(device_id).to_bytes(4, "little", signed=True)
        + (0).to_bytes(4, "little")
        + int(shareable_handle).to_bytes(8, "little")
        + int(mapping_bytes).to_bytes(8, "little")
    )


def _region_part_descriptor(
    *,
    owner_nonce: bytes,
    buffer_id: int,
    nbytes: int,
    backend_kind: BackendKind,
    body: bytes,
) -> BufferDescriptor:
    space = AddressSpace.HOST if backend_kind is BackendKind.POSIX_SHM else AddressSpace.DEVICE
    return BufferDescriptor(
        CanonicalIdentity(bytes(owner_nonce), int(buffer_id), 1),
        space,
        AccessMode.READWRITE,
        backend_kind,
        int(nbytes),
        body,
    )


def _wait_until(predicate, failure_message: str) -> None:
    deadline = time.monotonic() + _TEST_WALL_BUDGET_S
    while not predicate():
        if time.monotonic() >= deadline:
            pytest.fail(failure_message)
        time.sleep(0.001)


def test_worker_host_mapped_backend_types_are_removed_from_native_implementation():
    source = _TASK_INTERFACE_CPP.read_text(encoding="utf-8")

    for legacy_type in (
        "WorkerHostMappedRegionCleanupErrors",
        "WorkerHostMappedRegionEntry",
        "WorkerHostMappedRegionLease",
        "WorkerHostMappedRegionRegistry",
        "WorkerHostMappedRegionHandle",
        "class WorkerHostMappedRegion",
        "ChipChildOnboardRegionExport",
        "region_vmm_legacy_create",
        "_l3_child_onboard_region_create",
        "_l3_child_onboard_region_close",
    ):
        assert legacy_type not in source


def test_combined_host_worker_chip_region_helpers_are_removed():
    source = _WORKER_PY.read_text(encoding="utf-8")
    for legacy_name in (
        "_HostWorkerChipRegion",
        "_HostWorkerChipRegionStore",
        "_release_host_worker_chip_region",
        "_create_sim_worker_chip_region",
        "_create_onboard_worker_chip_region",
        "_sweep_host_" + "worker_chip" + "_regions",
        "_close_worker_host_mapping",
    ):
        assert legacy_name not in source


def test_old_create_reply_codec_and_combined_mapping_remain_deleted():
    worker_source = _WORKER_PY.read_text(encoding="utf-8")
    region_source = Path(comm_region.__file__).read_text(encoding="utf-8")
    binding_source = _TASK_INTERFACE_CPP.read_text(encoding="utf-8")
    for source in (worker_source, region_source, binding_source):
        for legacy_name in (
            "validate_region_create_reply",
            "decode_region_create_reply",
            "WorkerHostRegionMapping",
            "worker_chip_orch_region_access.h",
            "ChipChildOnboardRegionExport",
            "region_vmm_legacy_create",
        ):
            assert legacy_name not in source
    access_header = (
        Path(__file__).resolve().parents[4]
        / "src"
        / "common"
        / "platform"
        / "include"
        / "host"
        / "worker_chip_orch_region_access.h"
    )
    assert not access_header.exists()


class _FakeDirectCWorker:
    def __init__(
        self,
        *,
        payload_base: int = 0xDEAD_0000,
        counter_base: int = 0xDEAD_1000,
        access_profile: int = _ACCESS_SIM_POSIX_SHM,
        device_id: int = 0,
        shareable_handle: int = 0xABCDEF,
        reply_magic: Optional[int] = None,
        region_id: Optional[int] = None,
        mapping_bytes: Optional[int] = None,
        corrupt_access_profile: bool = False,
        owner_nonce: Optional[bytes] = None,
    ):
        self.create_calls: list[tuple[int, int]] = []
        self.release_calls: list[tuple[int, int]] = []
        self.next_region_id = 1
        self._last_resource_id = 0
        self.payload_base = int(payload_base)
        self.counter_base = int(counter_base)
        self.access_profile = int(access_profile)
        self.device_id = int(device_id)
        self.shareable_handle = int(shareable_handle)
        self.reply_magic = reply_magic
        self.region_id = region_id
        self.mapping_bytes = mapping_bytes
        self.corrupt_access_profile = bool(corrupt_access_profile)
        self.owner_nonce = owner_nonce
        self.allocate_specs: list[Any] = []

    def close(self) -> None:
        return None

    def control_payload(self, _worker_type, worker_id, sub_cmd, payload, _timeout):
        from simpler.comm_provider_control import (
            DelegatedAllocateReply,
            DelegatedAllocateReplyTag,
            DelegatedRegionOperation,
            DelegatedReleaseReply,
            DelegatedReleaseReplyTag,
            encode_reply,
            parse_request,
            publish_reply,
        )
        from simpler.worker import _CTRL_DELEGATED_REGION

        assert int(sub_cmd) == _CTRL_DELEGATED_REGION
        staged = bytearray(payload)
        envelope = parse_request(staged)
        if envelope.operation is DelegatedRegionOperation.DELEGATED_ALLOCATE:
            self.create_calls.append((int(worker_id), int(sub_cmd)))
            request = envelope.decode_terminal()
            spec = request.spec
            self.allocate_specs.append(spec)
            region_id = int(self.region_id) if self.region_id is not None else self.next_region_id
            if self.region_id is None:
                self.next_region_id += 1
            sim = self.access_profile == _ACCESS_SIM_POSIX_SHM
            if self.corrupt_access_profile:
                sim = not sim
            payload_bytes = int(spec.payload.logical_bytes)
            counter_bytes = int(spec.counter.logical_bytes)
            payload_mapping = payload_bytes if self.mapping_bytes is None else int(self.mapping_bytes)
            counter_mapping = counter_bytes if self.mapping_bytes is None else int(self.mapping_bytes)
            owner_nonce = self.owner_nonce
            if owner_nonce is None:
                raise RuntimeError("FakeDirectCWorker requires owner_nonce")
            if sim:
                payload_token = f"sim-direct-{region_id}-p"
                counter_token = f"sim-direct-{region_id}-c"
                _FAKE_POSIX_OBJECT_SIZES[payload_token] = payload_bytes
                _FAKE_POSIX_OBJECT_SIZES[counter_token] = counter_bytes
                payload_desc = _region_part_descriptor(
                    owner_nonce=owner_nonce,
                    buffer_id=1,
                    nbytes=payload_bytes,
                    backend_kind=BackendKind.POSIX_SHM,
                    body=payload_token.encode("ascii"),
                )
                counter_desc = _region_part_descriptor(
                    owner_nonce=owner_nonce,
                    buffer_id=2,
                    nbytes=counter_bytes,
                    backend_kind=BackendKind.POSIX_SHM,
                    body=counter_token.encode("ascii"),
                )
            else:
                payload_desc = _region_part_descriptor(
                    owner_nonce=owner_nonce,
                    buffer_id=1,
                    nbytes=payload_bytes,
                    backend_kind=BackendKind.VMM_SHAREABLE,
                    body=_vmm_shareable_body(
                        device_id=self.device_id,
                        shareable_handle=self.shareable_handle,
                        mapping_bytes=max(payload_mapping, payload_bytes),
                    ),
                )
                counter_desc = _region_part_descriptor(
                    owner_nonce=owner_nonce,
                    buffer_id=2,
                    nbytes=counter_bytes,
                    backend_kind=BackendKind.VMM_SHAREABLE,
                    body=_vmm_shareable_body(
                        device_id=self.device_id,
                        shareable_handle=self.shareable_handle + 1,
                        mapping_bytes=max(counter_mapping, counter_bytes),
                    ),
                )
            result = RegionAllocationResult(
                provider_resource_id=max(region_id, 1),
                export_descriptor=RegionExportDescriptor(payload=payload_desc, counter=counter_desc),
            )
            committed = encode_reply(
                DelegatedAllocateReply(
                    tag=DelegatedAllocateReplyTag.ALLOCATED,
                    session_instance_id=envelope.session_instance_id,
                    transaction_id=envelope.transaction_id,
                    result=result,
                    payload_view=RegionPartLocalView(RegionPartKind.PAYLOAD, self.payload_base, payload_bytes),
                    counter_view=RegionPartLocalView(RegionPartKind.COUNTER, self.counter_base, counter_bytes),
                )
            )
            publish_reply(memoryview(staged), committed)
            if region_id == 0:
                struct.pack_into("<Q", staged, 40, 0)
            if self.reply_magic is not None:
                struct.pack_into("<Q", staged, 0, int(self.reply_magic))
            self._last_resource_id = max(region_id, 1) if region_id != 0 else 0
            return bytes(staged)
        resource_id = int(getattr(self, "_last_resource_id", 1))
        self.release_calls.append((int(worker_id), resource_id))
        committed = encode_reply(
            DelegatedReleaseReply(
                tag=DelegatedReleaseReplyTag.RELEASED,
                session_instance_id=envelope.session_instance_id,
                transaction_id=envelope.transaction_id,
                result=ProviderReleaseResult(
                    provider_resource_id=resource_id,
                    status=ProviderReleaseStatus.RELEASED,
                ),
            )
        )
        publish_reply(memoryview(staged), committed)
        return bytes(staged)


class _EndpointFailingOrch:
    def _begin_run(self) -> int:
        return 1

    def _scope_begin(self) -> None:
        pass

    def _scope_end(self) -> None:
        pass

    def _close_run_submission(self, run_id: int) -> None:
        assert run_id == 1

    def _fail_run_submission(self, run_id: int) -> None:
        assert run_id == 1

    def _wait_run(self, run_id: int) -> None:
        assert run_id == 1
        raise RuntimeError(
            "child failed: L3-L2 endpoint error op=signal_wait kind=3 region=2 "
            "counter_addr=0x200000 counter_operand=7 observed_counter=0 msg=wait timed out"
        )

    def _release_run(self, run_id: int) -> None:
        assert run_id == 1


def _make_started_sim_worker() -> tuple[Worker, SharedMemory, _FakeDirectCWorker]:
    worker = Worker(level=3, device_ids=[0], platform="a2a3sim", runtime="tensormap_and_ringbuffer")
    shm = SharedMemory(create=True, size=MAILBOX_SIZE)
    assert shm.buf is not None
    _mailbox_store_i32(_buffer_field_addr(shm.buf, _OFF_STATE), _IDLE)
    fake_c_worker = _FakeDirectCWorker()
    worker._lifecycle = worker_module._Lifecycle.READY
    worker._worker = fake_c_worker
    worker._next_level_worker_ids = [0]
    worker._chip_shms = [shm]
    worker._ensure_local_device_endpoint_identities()
    fake_c_worker.owner_nonce = worker._device_endpoint_identities[(0, ce.DEVICE_AICPU)][0]
    return worker, shm, fake_c_worker


def _make_started_onboard_worker(platform: str = "a2a3") -> tuple[Worker, SharedMemory, _FakeDirectCWorker]:
    worker = Worker(level=3, device_ids=[2], platform=platform, runtime="tensormap_and_ringbuffer")
    shm = SharedMemory(create=True, size=MAILBOX_SIZE)
    assert shm.buf is not None
    _mailbox_store_i32(_buffer_field_addr(shm.buf, _OFF_STATE), _IDLE)
    fake_c_worker = _FakeDirectCWorker(
        access_profile=_ACCESS_ONBOARD_VMM,
        device_id=2,
    )
    worker._lifecycle = worker_module._Lifecycle.READY
    worker._worker = fake_c_worker
    worker._next_level_worker_ids = [0]
    worker._chip_shms = [shm]
    worker._ensure_local_device_endpoint_identities()
    fake_c_worker.owner_nonce = worker._device_endpoint_identities[(0, ce.DEVICE_AICPU)][0]
    return worker, shm, fake_c_worker


def test_sim_direct_region_uses_lifecycle_control_and_worker_host_metadata(monkeypatch):
    worker, shm, fake_c_worker = _make_started_sim_worker()
    calls: list[tuple] = []
    try:
        monkeypatch.setattr(
            worker_module,
            "_worker_host_mapped_region_import_sim",
            lambda token, mapping_bytes, owner_token: calls.append(("import", token, mapping_bytes, owner_token)) or 99,
        )
        monkeypatch.setattr(
            comm_region,
            "_host_vmm_copy_to",
            lambda handle, offset, src, nbytes: calls.append(("write", handle, offset, src, nbytes)),
        )
        monkeypatch.setattr(
            comm_region,
            "_host_vmm_copy_from",
            lambda handle, offset, dst, nbytes: calls.append(("read", handle, offset, dst, nbytes)),
        )
        monkeypatch.setattr(
            comm_region,
            "_region_counter_notify",
            lambda handle, offset, value, op: calls.append(("notify", handle, offset, value, op)),
        )
        monkeypatch.setattr(
            comm_region,
            "_region_counter_test",
            lambda handle, offset, value, cmp: (calls.append(("test", handle, offset, value, cmp)) or (True, 7)),
        )

        region = worker._create_worker_chip_region(0, 64, 128)
        payload = wrap_fork_inherited(0x1234_0000, 16, mint_owner_instance_id(), 1, "L3")
        with worker._control_reservation("test_sim_direct_region"):
            region.payload_write(0, payload, nbytes=8)
            region.payload_read(8, payload, nbytes=8)
            result = region.counter(64).test(7, WaitCmp.EQ)
            region.counter(64).notify(3, NotifyOp.Set)

        assert len(fake_c_worker.create_calls) == 1
        assert region.descriptor_scalars() == [0x4C334C3200030000, 1, 0xDEAD_0000, 64, 0xDEAD_1000, 128]
        assert 99 not in region.descriptor_scalars()
        assert region._instance._payload_attachment is not None
        assert region._instance._counter_attachment is not None
        assert int(region._instance._payload_attachment.native_lease) == 99
        assert int(region._instance._counter_attachment.native_lease) == 99
        assert int(region._instance._payload_attachment.native_lease) != region.descriptor.payload_base
        assert calls[0] == ("import", "sim-direct-1-p", 64, worker._owner_id)
        assert calls[1] == ("import", "sim-direct-1-c", 128, worker._owner_id)
        assert calls[2][0:3] == ("write", 99, 0)
        assert calls[3][0:3] == ("read", 99, 8)
        assert calls[4] == ("test", 99, 64, 7, int(WaitCmp.EQ))
        assert calls[5] == ("notify", 99, 64, 3, int(NotifyOp.Set))
        assert result == SignalTestResult(matched=True, observed=7)
    finally:
        worker._close_worker_chip_orch_comm()
        shm.close()
        shm.unlink()


def test_onboard_direct_region_imports_vmm_shareable_handle_and_uses_worker_host_metadata(monkeypatch):
    worker, shm, fake_c_worker = _make_started_onboard_worker()
    calls: list[tuple] = []
    try:
        monkeypatch.setattr(
            worker_module,
            "_worker_host_mapped_region_import_onboard",
            lambda device_id, shareable_handle, mapping_bytes, owner_token: calls.append(
                ("import_onboard", device_id, shareable_handle, mapping_bytes, owner_token)
            )
            or 123,
        )
        monkeypatch.setattr(
            comm_region,
            "_region_counter_notify",
            lambda handle, offset, value, op: calls.append(("notify", handle, offset, value, op)),
        )

        region = worker._create_worker_chip_region(0, 64, 128)
        with worker._control_reservation("test_onboard_direct_region"):
            region.counter(64).notify(9, NotifyOp.Set)

        assert len(fake_c_worker.create_calls) == 1
        assert region.descriptor_scalars() == [0x4C334C3200030000, 1, 0xDEAD_0000, 64, 0xDEAD_1000, 128]
        assert 123 not in region.descriptor_scalars()
        assert region._instance._payload_attachment is not None
        assert int(region._instance._payload_attachment.native_lease) == 123
        assert calls[0] == ("import_onboard", 2, 0xABCDEF, 64, worker._owner_id)
        assert calls[1] == ("import_onboard", 2, 0xABCDEF + 1, 128, worker._owner_id)
        assert calls[2] == ("notify", 123, 64, 9, int(NotifyOp.Set))
    finally:
        worker._close_worker_chip_orch_comm()
        shm.close()
        shm.unlink()


def test_sim_direct_create_sends_projected_vmm_window_spec(monkeypatch):
    worker, shm, fake_c_worker = _make_started_sim_worker()
    try:
        monkeypatch.setattr(
            worker_module,
            "_worker_host_mapped_region_import_sim",
            lambda _token, _mapping_bytes, _owner_token: 99,
        )

        region = worker._create_worker_chip_region(0, 64, 128)

        assert len(fake_c_worker.allocate_specs) == 1
        spec = fake_c_worker.allocate_specs[0]
        assert spec.payload.planned_backing_kind is BackendKind.VMM_SHAREABLE
        assert spec.counter.planned_backing_kind is BackendKind.VMM_SHAREABLE
        assert spec.payload.logical_bytes == 64
        assert spec.counter.logical_bytes == 128
        assert region.region_id == 1
    finally:
        worker._close_worker_chip_orch_comm()
        shm.close()
        shm.unlink()


def test_sim_direct_create_import_failure_rolls_back_l2_host_region(monkeypatch):
    worker, shm, fake_c_worker = _make_started_sim_worker()
    try:
        monkeypatch.setattr(
            worker_module,
            "_worker_host_mapped_region_import_sim",
            lambda _token, _mapping_bytes, _owner_token: (_ for _ in ()).throw(RuntimeError("import failed")),
        )

        with pytest.raises(RuntimeError, match="import failed"):
            worker._create_worker_chip_region(0, 64, 128)

        assert fake_c_worker.create_calls
        assert fake_c_worker.release_calls == [(0, 1)]
        assert worker._region_instance_registry._instances == {}
        assert worker._delegated_session_fatal is None
    finally:
        worker._close_worker_chip_orch_comm()
        shm.close()
        shm.unlink()


def test_direct_create_decode_failure_rolls_back_l2_host_region():
    worker, shm, fake_c_worker = _make_started_sim_worker()
    fake_c_worker.access_profile = _ACCESS_ONBOARD_VMM
    try:
        with pytest.raises(RuntimeError, match="committed actual backend is not the legal lowering"):
            worker._create_worker_chip_region(0, 64, 128)

        assert fake_c_worker.release_calls == []
        assert worker._region_instance_registry._instances == {}
        assert worker._delegated_session_fatal is not None
    finally:
        worker._close_worker_chip_orch_comm()
        shm.close()
        shm.unlink()


@pytest.mark.parametrize(
    ("reply_updates", "exc_type", "match", "released_id"),
    [
        ({"reply_magic": 0xBAD}, RegionControlError, "unsupported delegated-region version", None),
        (
            {"reply_magic": 0x4C334C3200020000},
            RegionControlError,
            "unsupported delegated-region version",
            None,
        ),
        ({"region_id": 0}, RegionControlError, "ALLOCATED requires a resource id", None),
        (
            {"access_profile": _ACCESS_ONBOARD_VMM},
            RuntimeError,
            "committed actual backend is not the legal lowering",
            None,
        ),
    ],
)
def test_direct_create_validation_failure_rolls_back_l2_host_region(reply_updates, exc_type, match, released_id):
    worker, shm, fake_c_worker = _make_started_sim_worker()
    for name, value in reply_updates.items():
        setattr(fake_c_worker, name, value)
    try:
        with pytest.raises(exc_type, match=match):
            worker._create_worker_chip_region(0, 64, 128)

        assert fake_c_worker.release_calls == ([] if released_id is None else [(0, released_id)])
        assert worker._region_instance_registry._instances == {}
        assert worker._delegated_session_fatal is not None
    finally:
        worker._close_worker_chip_orch_comm()
        shm.close()
        shm.unlink()


def test_onboard_direct_mapping_bytes_cover_each_independent_part(monkeypatch):
    worker, shm, fake_c_worker = _make_started_onboard_worker()
    fake_c_worker.mapping_bytes = 191
    calls: list[tuple] = []
    try:
        monkeypatch.setattr(
            worker_module,
            "_worker_host_mapped_region_import_onboard",
            lambda *args: calls.append(args) or 123,
        )

        region = worker._create_worker_chip_region(0, 64, 128)

        assert calls == [
            (2, 0xABCDEF, 191, worker._owner_id),
            (2, 0xABCDEF + 1, 191, worker._owner_id),
        ]
        assert region._instance._payload_attachment is not None
        assert int(region._instance._payload_attachment.native_lease) == 123
        assert int(region._instance._counter_attachment.native_lease) == 123
    finally:
        worker._close_worker_chip_orch_comm()
        shm.close()
        shm.unlink()


def test_onboard_direct_mapping_allows_granularity_aligned_mapping(monkeypatch):
    worker, shm, fake_c_worker = _make_started_onboard_worker()
    fake_c_worker.mapping_bytes = 65536
    calls: list[tuple] = []
    try:
        monkeypatch.setattr(
            worker_module,
            "_worker_host_mapped_region_import_onboard",
            lambda *args: calls.append(args) or 123,
        )

        region = worker._create_worker_chip_region(0, 64, 128)

        assert calls == [
            (2, 0xABCDEF, 65536, worker._owner_id),
            (2, 0xABCDEF + 1, 65536, worker._owner_id),
        ]
        assert region._instance._payload_attachment is not None
        assert int(region._instance._payload_attachment.native_lease) == 123
        assert int(region._instance._counter_attachment.native_lease) == 123
    finally:
        worker._close_worker_chip_orch_comm()
        shm.close()
        shm.unlink()


def test_direct_region_create_rolls_back_when_compat_wrapper_fails(monkeypatch):
    worker, shm, fake_c_worker = _make_started_sim_worker()
    resources = worker_module._RunResources()
    worker._building_run_resources = resources
    close_calls: list[int] = []
    monkeypatch.setattr(worker_module, "_worker_host_mapped_region_import_sim", lambda _token, _size, _owner_token: 55)
    monkeypatch.setattr(
        comm_region,
        "_worker_host_mapped_region_close",
        lambda handle: close_calls.append(int(handle)),
    )
    monkeypatch.setattr(
        worker_module,
        "worker_chip_orch_region_desc_from_local_views",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt("interrupted wrapper publication")),
    )

    try:
        with pytest.raises(KeyboardInterrupt, match="interrupted wrapper publication"):
            worker._create_worker_chip_region(0, 64, 128)

        assert worker._region_instance_registry._instances == {}
        assert resources.requires_ordered_cleanup is False
        assert close_calls == [55, 55]
        assert fake_c_worker.release_calls == [(0, 1)]
    finally:
        worker._building_run_resources = None
        worker._close_worker_chip_orch_comm()
        shm.close()
        shm.unlink()


def test_direct_region_create_mapping_rollback_failure_poisons_worker(monkeypatch):
    worker, shm, fake_c_worker = _make_started_sim_worker()
    monkeypatch.setattr(worker_module, "_worker_host_mapped_region_import_sim", lambda _token, _size, _owner_token: 55)
    monkeypatch.setattr(
        comm_region,
        "_worker_host_mapped_region_close",
        lambda _handle: (_ for _ in ()).throw(RuntimeError("mapping close failed")),
    )
    monkeypatch.setattr(
        worker_module,
        "worker_chip_orch_region_desc_from_local_views",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt("interrupted wrapper publication")),
    )

    try:
        with pytest.raises(RuntimeError, match="rollback could not close the L3 Host mapping") as excinfo:
            worker._create_worker_chip_region(0, 64, 128)

        assert isinstance(excinfo.value.__cause__, RuntimeError)
        assert fake_c_worker.release_calls == [(0, 1)]
        assert worker._region_instance_registry._instances == {}
        with pytest.raises(RuntimeError, match="no further work is admitted"):
            worker._require_no_ordered_cleanup_failure("submit")
    finally:
        worker._close_worker_chip_orch_comm()
        shm.close()
        shm.unlink()


def test_unadopted_native_mapping_cleanup_failure_poisons_worker(monkeypatch):
    worker, shm, fake_c_worker = _make_started_sim_worker()
    cleanup_errors = iter(("", "native owner cleanup failed"))
    consumed_owner_tokens: list[str] = []
    acknowledgements: list[tuple[str, str]] = []

    def peek_cleanup_error(owner_token: str) -> str:
        consumed_owner_tokens.append(owner_token)
        return next(cleanup_errors)

    monkeypatch.setattr(worker_module, "_worker_host_mapped_region_peek_cleanup_error", peek_cleanup_error)
    monkeypatch.setattr(
        worker_module,
        "_worker_host_mapped_region_ack_cleanup_error",
        lambda owner_token, observed: acknowledgements.append((owner_token, observed)),
    )
    monkeypatch.setattr(
        worker_module,
        "_worker_host_mapped_region_import_sim",
        lambda _token, _size, _owner_token: (_ for _ in ()).throw(KeyboardInterrupt("interrupted native adoption")),
    )

    try:
        with pytest.raises(RuntimeError, match="rollback could not close the L3 Host mapping") as excinfo:
            worker._create_worker_chip_region(0, 64, 128)

        assert isinstance(excinfo.value.__cause__, RuntimeError)
        assert "native owner cleanup failed" in str(excinfo.value.__cause__)
        assert consumed_owner_tokens == [worker._owner_id, worker._owner_id]
        assert acknowledgements == [(worker._owner_id, "native owner cleanup failed")]
        assert fake_c_worker.release_calls == [(0, 1)]
        with pytest.raises(RuntimeError, match="no further work is admitted"):
            worker._require_no_ordered_cleanup_failure("submit")
    finally:
        worker._close_worker_chip_orch_comm()
        shm.close()
        shm.unlink()


def test_interrupted_cleanup_ack_happens_after_region_rollback(monkeypatch):
    worker, shm, fake_c_worker = _make_started_sim_worker()
    cleanup_errors = iter(("", "native owner cleanup failed"))
    ack_interrupt = KeyboardInterrupt("interrupted cleanup acknowledgement")

    monkeypatch.setattr(
        worker_module,
        "_worker_host_mapped_region_peek_cleanup_error",
        lambda _owner_token: next(cleanup_errors),
    )

    def interrupt_ack(_owner_token: str, _observed: str) -> None:
        assert fake_c_worker.release_calls == [(0, 1)]
        raise ack_interrupt

    monkeypatch.setattr(worker_module, "_worker_host_mapped_region_ack_cleanup_error", interrupt_ack)
    monkeypatch.setattr(
        worker_module,
        "_worker_host_mapped_region_import_sim",
        lambda _token, _size, _owner_token: (_ for _ in ()).throw(KeyboardInterrupt("interrupted native adoption")),
    )

    try:
        with pytest.raises(KeyboardInterrupt) as caught:
            worker._create_worker_chip_region(0, 64, 128)

        assert caught.value is ack_interrupt
        assert fake_c_worker.release_calls == [(0, 1)]
        assert worker._ordered_cleanup_error is not None
        assert worker._worker_host_mapped_cleanup_error is not None
    finally:
        worker._close_worker_chip_orch_comm()
        shm.close()
        shm.unlink()


def test_deferred_native_cleanup_error_only_poisons_owning_worker_on_admission(monkeypatch):
    owner = Worker(level=3, num_sub_workers=0)
    peer = Worker(level=3, num_sub_workers=0)
    owner._lifecycle = worker_module._Lifecycle.READY
    peer._lifecycle = worker_module._Lifecycle.READY
    errors = {owner._owner_id: "owner mapping cleanup failed"}
    consumed_owner_tokens: list[str] = []

    def peek_cleanup_error(owner_token: str) -> str:
        consumed_owner_tokens.append(owner_token)
        return errors.get(owner_token, "")

    def acknowledge_cleanup_error(owner_token: str, observed: str) -> None:
        if errors.get(owner_token) == observed:
            errors.pop(owner_token)

    monkeypatch.setattr(worker_module, "_worker_host_mapped_region_peek_cleanup_error", peek_cleanup_error)
    monkeypatch.setattr(worker_module, "_worker_host_mapped_region_ack_cleanup_error", acknowledge_cleanup_error)

    with peer._operation_lease("submit"):
        pass

    assert peer._ordered_cleanup_error is None
    with pytest.raises(RuntimeError, match="no further work is admitted") as excinfo:
        with owner._operation_lease("submit"):
            pass

    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert "owner mapping cleanup failed" in str(excinfo.value.__cause__.__cause__)
    assert consumed_owner_tokens == [peer._owner_id, owner._owner_id]


def test_close_consumes_only_its_deferred_native_cleanup_error(monkeypatch):
    owner = Worker(level=3, num_sub_workers=0)
    peer = Worker(level=3, num_sub_workers=0)
    errors = {owner._owner_id: "owner mapping cleanup failed"}
    consumed_owner_tokens: list[str] = []

    def peek_cleanup_error(owner_token: str) -> str:
        consumed_owner_tokens.append(owner_token)
        return errors.get(owner_token, "")

    def acknowledge_cleanup_error(owner_token: str, observed: str) -> None:
        if errors.get(owner_token) == observed:
            errors.pop(owner_token)

    monkeypatch.setattr(worker_module, "_worker_host_mapped_region_peek_cleanup_error", peek_cleanup_error)
    monkeypatch.setattr(worker_module, "_worker_host_mapped_region_ack_cleanup_error", acknowledge_cleanup_error)

    peer.close()
    with pytest.raises(RuntimeError, match="native L3 Host mapping") as excinfo:
        owner.close()

    assert "owner mapping cleanup failed" in str(excinfo.value.__cause__)
    assert consumed_owner_tokens == [peer._owner_id, peer._owner_id, owner._owner_id, owner._owner_id]


def test_cleanup_error_survives_interrupted_peek_boundary(monkeypatch):
    owner = Worker(level=3, num_sub_workers=0)
    peer = Worker(level=3, num_sub_workers=0)
    owner._lifecycle = worker_module._Lifecycle.READY
    peer._lifecycle = worker_module._Lifecycle.READY
    errors = {owner._owner_id: "owner mapping cleanup failed"}
    interrupt = KeyboardInterrupt("interrupted native cleanup-error lookup")
    interrupt_owner_once = True

    def peek_cleanup_error(owner_token: str) -> str:
        nonlocal interrupt_owner_once
        if owner_token == owner._owner_id and interrupt_owner_once:
            interrupt_owner_once = False
            raise interrupt
        return errors.get(owner_token, "")

    def acknowledge_cleanup_error(owner_token: str, observed: str) -> None:
        if errors.get(owner_token) == observed:
            errors.pop(owner_token)

    monkeypatch.setattr(worker_module, "_worker_host_mapped_region_peek_cleanup_error", peek_cleanup_error)
    monkeypatch.setattr(worker_module, "_worker_host_mapped_region_ack_cleanup_error", acknowledge_cleanup_error)

    with pytest.raises(KeyboardInterrupt) as caught:
        with owner._operation_lease("submit"):
            pass

    assert caught.value is interrupt
    assert owner._ordered_cleanup_error is None
    assert errors == {owner._owner_id: "owner mapping cleanup failed"}
    with peer._operation_lease("submit"):
        pass
    with pytest.raises(RuntimeError, match="no further work is admitted"):
        with owner._operation_lease("submit"):
            pass
    assert owner._ordered_cleanup_error is owner._worker_host_mapped_cleanup_error
    assert errors == {}


def test_cleanup_error_ack_interrupt_happens_after_poison_publication(monkeypatch):
    owner = Worker(level=3, num_sub_workers=0)
    owner._lifecycle = worker_module._Lifecycle.READY
    errors = {owner._owner_id: "owner mapping cleanup failed"}
    interrupt = KeyboardInterrupt("interrupted cleanup-error acknowledgement")
    interrupt_ack_once = True

    monkeypatch.setattr(
        worker_module,
        "_worker_host_mapped_region_peek_cleanup_error",
        lambda owner_token: errors.get(owner_token, ""),
    )

    def acknowledge_cleanup_error(owner_token: str, observed: str) -> None:
        nonlocal interrupt_ack_once
        if errors.get(owner_token) == observed:
            errors.pop(owner_token)
        if interrupt_ack_once:
            interrupt_ack_once = False
            raise interrupt

    monkeypatch.setattr(worker_module, "_worker_host_mapped_region_ack_cleanup_error", acknowledge_cleanup_error)

    with pytest.raises(KeyboardInterrupt) as caught:
        with owner._operation_lease("submit"):
            pass

    assert caught.value is interrupt
    assert owner._ordered_cleanup_error is owner._worker_host_mapped_cleanup_error
    assert errors == {}
    with pytest.raises(RuntimeError, match="no further work is admitted"):
        with owner._operation_lease("submit"):
            pass


def test_later_native_cleanup_error_is_retained_in_sticky_poison(monkeypatch):
    owner = Worker(level=3, num_sub_workers=0)
    owner._lifecycle = worker_module._Lifecycle.READY
    errors: dict[str, str] = {owner._owner_id: "cleanup A"}

    monkeypatch.setattr(
        worker_module,
        "_worker_host_mapped_region_peek_cleanup_error",
        lambda owner_token: errors.get(owner_token, ""),
    )

    def acknowledge_cleanup_error(owner_token: str, observed: str) -> None:
        if errors.get(owner_token) == observed:
            errors.pop(owner_token)

    monkeypatch.setattr(worker_module, "_worker_host_mapped_region_ack_cleanup_error", acknowledge_cleanup_error)

    with pytest.raises(RuntimeError, match="no further work is admitted"):
        with owner._operation_lease("submit"):
            pass

    errors[owner._owner_id] = "cleanup B"
    with pytest.raises(RuntimeError, match="no further work is admitted"):
        with owner._operation_lease("submit"):
            pass

    assert owner._worker_host_mapped_cleanup_error is not None
    assert owner._worker_host_mapped_cleanup_error.__cause__ is not None
    assert str(owner._worker_host_mapped_cleanup_error.__cause__) == "cleanup A; cleanup B"
    assert errors == {}


def test_interrupted_ack_replay_does_not_duplicate_cleanup_detail(monkeypatch):
    owner = Worker(level=3, num_sub_workers=0)
    owner._lifecycle = worker_module._Lifecycle.READY
    errors: dict[str, str] = {owner._owner_id: "cleanup A"}
    interrupt = KeyboardInterrupt("interrupted before native acknowledgement")
    interrupt_once = True

    monkeypatch.setattr(
        worker_module,
        "_worker_host_mapped_region_peek_cleanup_error",
        lambda owner_token: errors.get(owner_token, ""),
    )

    def acknowledge_cleanup_error(owner_token: str, observed: str) -> None:
        nonlocal interrupt_once
        if interrupt_once:
            interrupt_once = False
            raise interrupt
        if errors.get(owner_token) == observed:
            errors.pop(owner_token)

    monkeypatch.setattr(worker_module, "_worker_host_mapped_region_ack_cleanup_error", acknowledge_cleanup_error)

    with pytest.raises(KeyboardInterrupt) as caught:
        with owner._operation_lease("submit"):
            pass
    assert caught.value is interrupt

    with pytest.raises(RuntimeError, match="no further work is admitted"):
        with owner._operation_lease("submit"):
            pass

    assert owner._worker_host_mapped_cleanup_error is not None
    assert owner._worker_host_mapped_cleanup_error.__cause__ is not None
    assert str(owner._worker_host_mapped_cleanup_error.__cause__) == "cleanup A"
    assert errors == {}


def test_late_cleanup_error_after_successful_close_replays_stably(monkeypatch):
    worker = Worker(level=3, num_sub_workers=0)
    errors: dict[str, str] = {}

    monkeypatch.setattr(
        worker_module,
        "_worker_host_mapped_region_peek_cleanup_error",
        lambda owner_token: errors.get(owner_token, ""),
    )

    def acknowledge_cleanup_error(owner_token: str, observed: str) -> None:
        if errors.get(owner_token) == observed:
            errors.pop(owner_token)

    monkeypatch.setattr(worker_module, "_worker_host_mapped_region_ack_cleanup_error", acknowledge_cleanup_error)

    worker.close()
    errors[worker._owner_id] = "late owner mapping cleanup failed"

    with pytest.raises(RuntimeError, match="native L3 Host mapping") as first:
        worker.close()
    with pytest.raises(RuntimeError) as replayed:
        worker.close()

    assert replayed.value is first.value
    assert errors == {}


def test_close_replay_does_not_double_release_worker_chip_region_after_mapping_close_failure(monkeypatch):
    worker, shm, fake_c_worker = _make_started_sim_worker()
    worker._init_owner_thread = threading.current_thread()
    handles = iter((77, 78))
    monkeypatch.setattr(
        worker_module,
        "_worker_host_mapped_region_import_sim",
        lambda _token, _size, _owner_token: next(handles),
    )
    close_calls: list[int] = []
    fail_next_mapping_close = True

    def close_mapping(handle: int) -> None:
        nonlocal fail_next_mapping_close
        close_calls.append(int(handle))
        if fail_next_mapping_close:
            fail_next_mapping_close = False
            raise RuntimeError("mapping close failed")

    monkeypatch.setattr(comm_region, "_worker_host_mapped_region_close", close_mapping)

    try:
        region = worker._create_worker_chip_region(0, 64, 128)
        with pytest.raises(RuntimeError, match="mapping close failed"):
            worker.close()

        assert close_calls == [77, 78]
        assert fake_c_worker.release_calls == [(0, 1)]
        assert worker._region_instance_registry._instances == {}
        assert region.expired

        worker.close()

        assert close_calls == [77, 78]
        assert fake_c_worker.release_calls == [(0, 1)]
    finally:
        with contextlib.suppress(FileNotFoundError):
            shm.close()
            shm.unlink()


def test_concurrent_close_publishes_joiner_cleanup_error_to_every_caller(monkeypatch):
    worker = Worker(level=3, num_sub_workers=0)
    worker._lifecycle = worker_module._Lifecycle.READY
    worker._worker = cast(Any, object())
    worker._init_owner_thread = threading.current_thread()
    errors: dict[str, str] = {}
    teardown_started = threading.Event()
    joiner_errors: list[BaseException] = []

    monkeypatch.setattr(
        worker_module,
        "_worker_host_mapped_region_peek_cleanup_error",
        lambda owner_token: errors.get(owner_token, ""),
    )

    def acknowledge_cleanup_error(owner_token: str, observed: str) -> None:
        if errors.get(owner_token) == observed:
            errors.pop(owner_token)

    monkeypatch.setattr(worker_module, "_worker_host_mapped_region_ack_cleanup_error", acknowledge_cleanup_error)

    def teardown_tree() -> None:
        errors[worker._owner_id] = "joiner observed mapping cleanup failed"
        teardown_started.set()
        worker._worker = None

    monkeypatch.setattr(worker, "_teardown_ready_tree", teardown_tree)

    def join_close() -> None:
        assert teardown_started.wait(5.0)
        try:
            worker.close()
        except BaseException as exc:  # noqa: BLE001
            joiner_errors.append(exc)

    joiner = threading.Thread(target=join_close)
    joiner.start()
    try:
        with pytest.raises(RuntimeError, match="native L3 Host mapping") as owner_error:
            worker.close()
    finally:
        joiner.join(5.0)

    assert not joiner.is_alive()
    assert joiner_errors == [owner_error.value]


def test_cleanup_error_after_final_consume_waits_for_next_close_attempt(monkeypatch):
    worker = Worker(level=3, num_sub_workers=0)
    worker._lifecycle = worker_module._Lifecycle.READY
    worker._worker = cast(Any, object())
    worker._init_owner_thread = threading.current_thread()
    errors: dict[str, str] = {}
    late_recorded = threading.Event()
    joiner_waiting = threading.Event()
    joiner_errors: list[BaseException] = []
    has_live_calls = 0

    monkeypatch.setattr(
        worker_module,
        "_worker_host_mapped_region_peek_cleanup_error",
        lambda owner_token: errors.get(owner_token, ""),
    )

    def acknowledge_cleanup_error(owner_token: str, observed: str) -> None:
        if errors.get(owner_token) == observed:
            errors.pop(owner_token)

    monkeypatch.setattr(worker_module, "_worker_host_mapped_region_ack_cleanup_error", acknowledge_cleanup_error)
    monkeypatch.setattr(worker, "_teardown_ready_tree", lambda: setattr(worker, "_worker", None))

    def has_live_resources() -> bool:
        nonlocal has_live_calls
        has_live_calls += 1
        if has_live_calls == 1:
            return True
        if has_live_calls == 2:
            errors[worker._owner_id] = "late owner mapping cleanup failed"
            late_recorded.set()
            assert joiner_waiting.wait(5.0)
        return False

    monkeypatch.setattr(worker, "_has_live_resources", has_live_resources)
    real_close_wait = worker._hierarchical_start_cv.wait
    joiner: threading.Thread

    def close_wait(timeout=None):
        if threading.current_thread() is joiner:
            joiner_waiting.set()
        return real_close_wait(timeout=timeout)

    monkeypatch.setattr(worker._hierarchical_start_cv, "wait", close_wait)

    def join_close() -> None:
        assert late_recorded.wait(5.0)
        try:
            worker.close()
        except BaseException as exc:  # noqa: BLE001
            joiner_errors.append(exc)

    joiner = threading.Thread(target=join_close)
    joiner.start()
    try:
        worker.close()
    finally:
        joiner.join(5.0)

    assert not joiner.is_alive()
    assert joiner_errors == []
    assert worker._close_completion is not None
    assert worker._close_completion.error is None
    assert errors == {worker._owner_id: "late owner mapping cleanup failed"}

    with pytest.raises(RuntimeError, match="native L3 Host mapping") as first:
        worker.close()
    with pytest.raises(RuntimeError) as replayed:
        worker.close()
    assert replayed.value is first.value


def test_wrong_thread_close_does_not_consume_owner_cleanup_error(monkeypatch):
    worker = Worker(level=3, num_sub_workers=0)
    worker._lifecycle = worker_module._Lifecycle.READY
    worker._worker = cast(Any, object())
    worker._init_owner_thread = threading.current_thread()
    errors = {worker._owner_id: "owner mapping cleanup failed"}
    peeked: list[str] = []
    foreign_errors: list[BaseException] = []

    def peek_cleanup_error(owner_token: str) -> str:
        peeked.append(owner_token)
        return errors.get(owner_token, "")

    def acknowledge_cleanup_error(owner_token: str, observed: str) -> None:
        if errors.get(owner_token) == observed:
            errors.pop(owner_token)

    monkeypatch.setattr(worker_module, "_worker_host_mapped_region_peek_cleanup_error", peek_cleanup_error)
    monkeypatch.setattr(worker_module, "_worker_host_mapped_region_ack_cleanup_error", acknowledge_cleanup_error)
    monkeypatch.setattr(worker, "_teardown_ready_tree", lambda: setattr(worker, "_worker", None))

    def close_from_foreign_thread() -> None:
        try:
            worker.close()
        except BaseException as exc:  # noqa: BLE001
            foreign_errors.append(exc)

    foreign = threading.Thread(target=close_from_foreign_thread)
    foreign.start()
    foreign.join(5.0)

    assert len(foreign_errors) == 1
    assert "thread that init()'d it" in str(foreign_errors[0])
    assert peeked == []
    assert errors == {worker._owner_id: "owner mapping cleanup failed"}
    with pytest.raises(RuntimeError, match="native L3 Host mapping"):
        worker.close()
    assert errors == {}


def test_cleanup_error_survives_close_drain_timeout_retry(monkeypatch):
    worker = Worker(level=3, num_sub_workers=0)
    worker._lifecycle = worker_module._Lifecycle.READY
    worker._worker = cast(Any, object())
    worker._init_owner_thread = threading.current_thread()
    worker._active_ops = 1
    errors = {worker._owner_id: "owner mapping cleanup failed"}

    monkeypatch.setattr(worker_module, "_ROLLBACK_GRACEFUL_TIMEOUT_S", 0.001)
    monkeypatch.setattr(
        worker_module,
        "_worker_host_mapped_region_peek_cleanup_error",
        lambda owner_token: errors.get(owner_token, ""),
    )

    def acknowledge_cleanup_error(owner_token: str, observed: str) -> None:
        if errors.get(owner_token) == observed:
            errors.pop(owner_token)

    monkeypatch.setattr(worker_module, "_worker_host_mapped_region_ack_cleanup_error", acknowledge_cleanup_error)
    monkeypatch.setattr(worker, "_teardown_ready_tree", lambda: setattr(worker, "_worker", None))

    with pytest.raises(TimeoutError):
        worker.close()
    assert worker._worker_host_mapped_cleanup_error is not None
    assert errors == {}

    worker._active_ops = 0
    with pytest.raises(RuntimeError, match="native L3 Host mapping") as retry:
        worker.close()
    with pytest.raises(RuntimeError) as replayed:
        worker.close()

    assert replayed.value is retry.value


def test_native_mapping_cleanup_errors_are_keyed_by_owner_token():
    owner_token = "owner-a"
    peer_token = "owner-b"
    _task_interface_ext._worker_host_mapped_region_take_cleanup_error(owner_token)
    _task_interface_ext._worker_host_mapped_region_take_cleanup_error(peer_token)

    _task_interface_ext._worker_host_mapped_region_record_cleanup_error_for_test(
        owner_token, "owner mapping cleanup failed"
    )

    assert _task_interface_ext._worker_host_mapped_region_peek_cleanup_error(peer_token) == ""
    observed = _task_interface_ext._worker_host_mapped_region_peek_cleanup_error(owner_token)
    assert observed == "owner mapping cleanup failed"
    _task_interface_ext._worker_host_mapped_region_record_cleanup_error_for_test(owner_token, "later cleanup failed")
    _task_interface_ext._worker_host_mapped_region_ack_cleanup_error(owner_token, observed)
    assert _task_interface_ext._worker_host_mapped_region_peek_cleanup_error(owner_token) == "later cleanup failed"
    _task_interface_ext._worker_host_mapped_region_ack_cleanup_error(owner_token, "later cleanup failed")
    assert _task_interface_ext._worker_host_mapped_region_take_cleanup_error(owner_token) == ""


def test_worker_host_mapped_counter_wait_releases_gil_for_python_notifier():
    shm = SharedMemory(create=True, size=64)
    handle = 0
    try:
        owner = _task_interface_ext._worker_host_mapped_region_import_sim(shm.name, 64, "counter-wait-test")
        handle = int(owner)

        def notify() -> None:
            _wait_until(
                lambda: _task_interface_ext._worker_host_mapped_region_active_leases(handle) == 1,
                "counter wait never acquired its mapped-region lease",
            )
            _task_interface_ext._worker_host_mapped_counter_notify(handle, 0, 1, int(NotifyOp.Set))

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(notify)
            status, error_kind, observed, matched, message = _task_interface_ext._worker_host_mapped_counter_wait(
                handle, 0, 1, int(WaitCmp.EQ), _TEST_WALL_BUDGET_NS
            )
            future.result(timeout=_TEST_WALL_BUDGET_S)

        assert (status, error_kind, observed, matched, message) == (0, 0, 1, True, "")
    finally:
        if handle:
            _task_interface_ext._worker_host_mapped_region_close(handle)
        shm.close()
        shm.unlink()


def test_region_neutral_counter_wait_releases_gil_for_python_notifier():
    shm = SharedMemory(create=True, size=64)
    handle = 0
    try:
        owner = _task_interface_ext._region_import_sim(shm.name, 64, "neutral-counter-wait-test")
        handle = int(owner)

        def notify() -> None:
            _wait_until(
                lambda: _task_interface_ext._region_active_leases(handle) == 1,
                "counter wait never acquired its mapped-region lease",
            )
            _task_interface_ext._region_counter_notify(handle, 0, 1, int(NotifyOp.Set))

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(notify)
            status, error_kind, observed, matched, message = _task_interface_ext._region_counter_wait(
                handle, 0, 1, int(WaitCmp.EQ), _TEST_WALL_BUDGET_NS
            )
            future.result(timeout=_TEST_WALL_BUDGET_S)

        assert (status, error_kind, observed, matched, message) == (0, 0, 1, True, "")
    finally:
        if handle:
            _task_interface_ext._region_close(handle)
        shm.close()
        shm.unlink()


def test_region_neutral_sim_byte_copy_and_counter_helpers_roundtrip():
    shm = SharedMemory(create=True, size=128)
    handle = 0
    try:
        owner = _task_interface_ext._region_import_sim(shm.name, 128, "neutral-roundtrip-test")
        handle = int(owner)
        src_t = ctypes.c_uint8 * 8
        src = src_t(*range(10, 18))
        dst = src_t()

        _task_interface_ext._host_vmm_copy_to(handle, 16, ctypes.addressof(src), 8)
        _task_interface_ext._host_vmm_copy_from(handle, 16, ctypes.addressof(dst), 8)
        assert bytes(dst) == bytes(range(10, 18))

        _task_interface_ext._region_counter_notify(handle, 64, 3, int(NotifyOp.Set))
        assert _task_interface_ext._region_counter_test(handle, 64, 3, int(WaitCmp.EQ)) == (True, 3)
        _task_interface_ext._region_counter_notify(handle, 64, 4, int(NotifyOp.Add))
        assert _task_interface_ext._region_counter_test(handle, 64, 7, int(WaitCmp.GE)) == (True, 7)
        assert _task_interface_ext._region_counter_wait(handle, 64, 7, int(WaitCmp.EQ), 1_000_000) == (
            0,
            0,
            7,
            True,
            "",
        )

        _task_interface_ext._region_close(handle)
        with pytest.raises(RuntimeError, match="closed or unknown"):
            _task_interface_ext._host_vmm_copy_from(handle, 16, ctypes.addressof(dst), 8)
    finally:
        if handle:
            _task_interface_ext._region_close(handle)
        shm.close()
        shm.unlink()


def test_region_neutral_byte_copy_holds_active_lease_until_native_copy_returns():
    shm = SharedMemory(create=True, size=128)
    handle = 0
    try:
        owner = _task_interface_ext._region_import_sim(shm.name, 128, "neutral-byte-copy-lease-test")
        handle = int(owner)
        src_t = ctypes.c_uint8 * 8
        src = src_t(*range(1, 9))
        delayed_copy = _task_interface_ext._host_vmm_copy_to_with_delay_for_test

        def delayed_copy_to() -> None:
            delayed_copy(handle, 16, ctypes.addressof(src), 8, 200)

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(delayed_copy_to)
            _wait_until(
                lambda: _task_interface_ext._region_active_leases(handle) == 1,
                "byte copy never acquired its mapped-region lease",
            )

            assert _task_interface_ext._region_active_leases(handle) == 1
            future.result(timeout=_TEST_WALL_BUDGET_S)

        assert _task_interface_ext._region_active_leases(handle) == 0
        assert bytes(cast(memoryview, shm.buf)[16:24]) == bytes(range(1, 9))
    finally:
        if handle:
            _task_interface_ext._region_close(handle)
        shm.close()
        shm.unlink()


def test_region_neutral_counter_wait_timeout_reports_last_observed_value():
    shm = SharedMemory(create=True, size=64)
    handle = 0
    try:
        owner = _task_interface_ext._region_import_sim(shm.name, 64, "neutral-counter-timeout-test")
        handle = int(owner)
        _task_interface_ext._region_counter_notify(handle, 0, 0, int(NotifyOp.Set))
        assert _task_interface_ext._region_counter_wait(handle, 0, 1, int(WaitCmp.EQ), 1_000_000) == (
            -1,
            7,
            0,
            False,
            "SIGNAL_WAIT timed out",
        )
    finally:
        if handle:
            _task_interface_ext._region_close(handle)
        shm.close()
        shm.unlink()


def test_region_neutral_import_registry_failure_releases_pre_registry_mapping():
    if not os.path.exists("/proc/self/maps"):
        pytest.skip("requires Linux procfs resource accounting")

    shm = SharedMemory(create=True, size=64)
    shm_token = shm.name.lstrip("/")
    owner_token = "neutral-registry-failure-test"

    def mapped_resource_counts() -> tuple[int, int]:
        fd_count = 0
        for fd_name in os.listdir("/proc/self/fd"):
            try:
                target = os.readlink(f"/proc/self/fd/{fd_name}")
            except OSError:
                continue
            fd_count += shm_token in target
        with open("/proc/self/maps", encoding="utf-8") as maps_file:
            map_count = sum(shm_token in line for line in maps_file)
        return fd_count, map_count

    try:
        baseline = mapped_resource_counts()
        _task_interface_ext._region_take_cleanup_error(owner_token)
        _task_interface_ext._region_fail_next_registry_insert_for_test()

        with pytest.raises(RuntimeError, match="injected mapped-region registry insertion failure"):
            _task_interface_ext._region_import_sim(shm.name, 64, owner_token)

        gc.collect()
        assert mapped_resource_counts() == baseline
        assert _task_interface_ext._region_take_cleanup_error(owner_token) == ""
    finally:
        shm.close()
        shm.unlink()


def test_region_neutral_concurrent_closes_wait_for_in_flight_counter_wait():
    shm = SharedMemory(create=True, size=64)
    handle = 0
    try:
        owner = _task_interface_ext._region_import_sim(shm.name, 64, "neutral-concurrent-close-test")
        handle = int(owner)
        close_entered = [threading.Event(), threading.Event()]
        close_done = [threading.Event(), threading.Event()]

        def wait_for_counter():
            return _task_interface_ext._region_counter_wait(handle, 0, 1, int(WaitCmp.EQ), _TEST_WALL_BUDGET_NS)

        def close_mapping(index: int) -> None:
            close_entered[index].set()
            _task_interface_ext._region_close(handle)
            close_done[index].set()

        with ThreadPoolExecutor(max_workers=3) as executor:
            wait_future = executor.submit(wait_for_counter)
            _wait_until(
                lambda: _task_interface_ext._region_active_leases(handle) == 1,
                "counter wait never acquired its mapped-region lease",
            )

            close_futures = [executor.submit(close_mapping, index) for index in range(2)]
            assert all(event.wait(_TEST_WALL_BUDGET_S) for event in close_entered)
            _wait_until(
                lambda: _task_interface_ext._region_closing_for_test(handle),
                "mapped-region close never started waiting for the active lease",
            )
            assert not any(event.is_set() for event in close_done), (
                "a concurrent close returned while a native operation still held the region"
            )

            cast(memoryview, shm.buf)[:4] = b"\x01\x00\x00\x00"
            assert wait_future.result(timeout=_TEST_WALL_BUDGET_S) == (0, 0, 1, True, "")
            for close_future in close_futures:
                close_future.result(timeout=_TEST_WALL_BUDGET_S)

        with pytest.raises(RuntimeError, match="closed or unknown"):
            _task_interface_ext._region_counter_test(handle, 0, 1, int(WaitCmp.EQ))
    finally:
        if handle:
            _task_interface_ext._region_close(handle)
        shm.close()
        shm.unlink()


def test_region_neutral_cleanup_diagnostics_are_owner_scoped():
    owner_token = "neutral-cleanup-owner"
    peer_token = "neutral-cleanup-peer"
    _task_interface_ext._region_take_cleanup_error(owner_token)
    _task_interface_ext._region_take_cleanup_error(peer_token)

    _task_interface_ext._region_record_cleanup_error_for_test(owner_token, "cleanup failed")

    assert _task_interface_ext._region_peek_cleanup_error(peer_token) == ""
    observed = _task_interface_ext._region_peek_cleanup_error(owner_token)
    assert observed == "cleanup failed"
    _task_interface_ext._region_record_cleanup_error_for_test(owner_token, "later cleanup failed")
    _task_interface_ext._region_ack_cleanup_error(owner_token, observed)
    assert _task_interface_ext._region_peek_cleanup_error(owner_token) == "later cleanup failed"
    _task_interface_ext._region_ack_cleanup_error(owner_token, "later cleanup failed")
    assert _task_interface_ext._region_take_cleanup_error(owner_token) == ""


def test_worker_host_mapped_sim_payload_and_counter_helpers_roundtrip():
    shm = SharedMemory(create=True, size=128)
    handle = 0
    try:
        owner = _task_interface_ext._worker_host_mapped_region_import_sim(shm.name, 128, "roundtrip-test")
        handle = int(owner)
        src_t = ctypes.c_uint8 * 8
        src = src_t(*range(10, 18))
        dst = src_t()

        _task_interface_ext._worker_host_mapped_payload_write(handle, 16, ctypes.addressof(src), 8)
        _task_interface_ext._worker_host_mapped_payload_read(handle, 16, ctypes.addressof(dst), 8)
        assert bytes(dst) == bytes(range(10, 18))

        _task_interface_ext._worker_host_mapped_counter_notify(handle, 64, 3, int(NotifyOp.Set))
        assert _task_interface_ext._worker_host_mapped_counter_test(handle, 64, 3, int(WaitCmp.EQ)) == (True, 3)
        _task_interface_ext._worker_host_mapped_counter_notify(handle, 64, 4, int(NotifyOp.Add))
        assert _task_interface_ext._worker_host_mapped_counter_test(handle, 64, 7, int(WaitCmp.GE)) == (True, 7)
        assert _task_interface_ext._worker_host_mapped_counter_wait(handle, 64, 7, int(WaitCmp.EQ), 1_000_000) == (
            0,
            0,
            7,
            True,
            "",
        )

        _task_interface_ext._worker_host_mapped_region_close(handle)
        with pytest.raises(RuntimeError, match="closed or unknown"):
            _task_interface_ext._worker_host_mapped_payload_read(handle, 16, ctypes.addressof(dst), 8)
    finally:
        if handle:
            _task_interface_ext._worker_host_mapped_region_close(handle)
        shm.close()
        shm.unlink()


def test_worker_host_mapped_region_close_makes_sim_handle_unusable():
    shm = SharedMemory(create=True, size=64)
    handle = 0
    try:
        owner = _task_interface_ext._worker_host_mapped_region_import_sim(shm.name, 64, "closed-handle-test")
        handle = int(owner)
        _task_interface_ext._worker_host_mapped_region_close(handle)

        with pytest.raises(RuntimeError, match="closed or unknown"):
            _task_interface_ext._worker_host_mapped_counter_test(handle, 0, 0, int(WaitCmp.EQ))
    finally:
        if handle:
            _task_interface_ext._worker_host_mapped_region_close(handle)
        shm.close()
        shm.unlink()


def test_worker_host_mapped_import_owner_closes_unadopted_mapping():
    shm = SharedMemory(create=True, size=64)
    raw_handle = 0
    try:
        owner = _task_interface_ext._worker_host_mapped_region_import_sim(shm.name, 64, "unadopted-owner-test")
        raw_handle = int(owner)
        del owner
        gc.collect()

        with pytest.raises(RuntimeError, match="closed or unknown"):
            _task_interface_ext._worker_host_mapped_counter_test(raw_handle, 0, 0, int(WaitCmp.EQ))
    finally:
        if raw_handle:
            _task_interface_ext._worker_host_mapped_region_close(raw_handle)
        shm.close()
        shm.unlink()


def test_sim_import_registry_failure_releases_pre_registry_mapping():
    if not os.path.exists("/proc/self/maps"):
        pytest.skip("requires Linux procfs resource accounting")

    shm = SharedMemory(create=True, size=64)
    shm_token = shm.name.lstrip("/")
    owner_token = "registry-failure-test"

    def mapped_resource_counts() -> tuple[int, int]:
        fd_count = 0
        for fd_name in os.listdir("/proc/self/fd"):
            try:
                target = os.readlink(f"/proc/self/fd/{fd_name}")
            except OSError:
                continue
            fd_count += shm_token in target
        with open("/proc/self/maps", encoding="utf-8") as maps_file:
            map_count = sum(shm_token in line for line in maps_file)
        return fd_count, map_count

    try:
        baseline = mapped_resource_counts()
        _task_interface_ext._worker_host_mapped_region_take_cleanup_error(owner_token)
        _task_interface_ext._worker_host_mapped_region_fail_next_registry_insert_for_test()

        with pytest.raises(RuntimeError, match="injected mapped-region registry insertion failure"):
            _task_interface_ext._worker_host_mapped_region_import_sim(shm.name, 64, owner_token)

        gc.collect()
        assert mapped_resource_counts() == baseline
        assert _task_interface_ext._worker_host_mapped_region_take_cleanup_error(owner_token) == ""
    finally:
        shm.close()
        shm.unlink()


def test_worker_host_mapped_concurrent_closes_wait_for_in_flight_counter_wait():
    shm = SharedMemory(create=True, size=64)
    handle = 0
    try:
        owner = _task_interface_ext._worker_host_mapped_region_import_sim(shm.name, 64, "concurrent-close-test")
        handle = int(owner)
        close_entered = [threading.Event(), threading.Event()]
        close_done = [threading.Event(), threading.Event()]

        def wait_for_counter():
            return _task_interface_ext._worker_host_mapped_counter_wait(
                handle, 0, 1, int(WaitCmp.EQ), _TEST_WALL_BUDGET_NS
            )

        def close_mapping(index: int) -> None:
            close_entered[index].set()
            _task_interface_ext._worker_host_mapped_region_close(handle)
            close_done[index].set()

        with ThreadPoolExecutor(max_workers=3) as executor:
            wait_future = executor.submit(wait_for_counter)
            _wait_until(
                lambda: _task_interface_ext._worker_host_mapped_region_active_leases(handle) == 1,
                "counter wait never acquired its mapped-region lease",
            )

            close_futures = [executor.submit(close_mapping, index) for index in range(2)]
            assert all(event.wait(_TEST_WALL_BUDGET_S) for event in close_entered)
            _wait_until(
                lambda: _task_interface_ext._worker_host_mapped_region_closing_for_test(handle),
                "L3 Host mapped-region close never started waiting for the active lease",
            )
            assert not any(event.is_set() for event in close_done), (
                "a concurrent close returned while a native operation still held the region"
            )

            cast(memoryview, shm.buf)[:4] = b"\x01\x00\x00\x00"
            assert wait_future.result(timeout=_TEST_WALL_BUDGET_S) == (0, 0, 1, True, "")
            for close_future in close_futures:
                close_future.result(timeout=_TEST_WALL_BUDGET_S)

        with pytest.raises(RuntimeError, match="closed or unknown"):
            _task_interface_ext._worker_host_mapped_counter_test(handle, 0, 1, int(WaitCmp.EQ))
    finally:
        if handle:
            _task_interface_ext._worker_host_mapped_region_close(handle)
        shm.close()
        shm.unlink()


def test_sim_direct_transfer_failure_poisons_only_region(monkeypatch):
    worker, shm, _fake_c_worker = _make_started_sim_worker()
    try:
        monkeypatch.setattr(
            worker_module, "_worker_host_mapped_region_import_sim", lambda _token, _size, _owner_token: 55
        )
        monkeypatch.setattr(
            comm_region,
            "_host_vmm_copy_to",
            lambda _handle, _offset, _src, _nbytes: (_ for _ in ()).throw(RuntimeError("copy failed")),
        )

        region = worker._create_worker_chip_region(0, 64, 128)
        payload = wrap_fork_inherited(0x1234_0000, 16, mint_owner_instance_id(), 1, "L3")
        with worker._control_reservation("test_sim_direct_transfer_failure"):
            with pytest.raises(RuntimeError, match="copy failed"):
                region.payload_write(0, payload, nbytes=8)
        with pytest.raises(RuntimeError, match="poisoned"):
            region.descriptor_scalars()
    finally:
        worker._close_worker_chip_orch_comm()
        shm.close()
        shm.unlink()


def test_sim_direct_counter_failure_poisons_only_region(monkeypatch):
    worker, shm, _fake_c_worker = _make_started_sim_worker()
    try:
        monkeypatch.setattr(
            worker_module, "_worker_host_mapped_region_import_sim", lambda _token, _size, _owner_token: 55
        )
        monkeypatch.setattr(
            comm_region,
            "_region_counter_notify",
            lambda _handle, _offset, _value, _op: (_ for _ in ()).throw(RuntimeError("counter failed")),
        )

        region = worker._create_worker_chip_region(0, 64, 128)
        with worker._control_reservation("test_sim_direct_counter_failure"):
            with pytest.raises(RuntimeError, match="counter failed"):
                region.counter(0).notify(1, NotifyOp.Set)
        with pytest.raises(RuntimeError, match="poisoned"):
            region.descriptor_scalars()
    finally:
        worker._close_worker_chip_orch_comm()
        shm.close()
        shm.unlink()


def test_sim_direct_cleanup_closes_worker_host_mapping_before_l2_host_release(monkeypatch):
    worker, shm, fake_c_worker = _make_started_sim_worker()
    events: list[tuple[str, int]] = []
    original_payload = fake_c_worker.control_payload

    def payload(worker_type, worker_id, sub_cmd, request, timeout):
        from simpler.comm_provider_control import DelegatedRegionOperation, parse_request

        envelope = parse_request(request)
        if envelope.operation is DelegatedRegionOperation.DELEGATED_RELEASE:
            events.append(("release", int(getattr(fake_c_worker, "_last_resource_id", 1))))
        return original_payload(worker_type, worker_id, sub_cmd, request, timeout)

    try:
        fake_c_worker.control_payload = payload
        monkeypatch.setattr(
            worker_module, "_worker_host_mapped_region_import_sim", lambda _token, _size, _owner_token: 77
        )
        monkeypatch.setattr(
            comm_region,
            "_worker_host_mapped_region_close",
            lambda handle: events.append(("close", int(handle))),
        )

        region = worker._create_worker_chip_region(0, 64, 128)
        region.free()
        worker._sweep_region_instances()

        assert events == [("close", 77), ("close", 77), ("release", 1)]
        with pytest.raises(RuntimeError, match="expired"):
            region.descriptor_scalars()
        assert region._instance._close_attempted
    finally:
        worker._close_worker_chip_orch_comm()
        shm.close()
        shm.unlink()


def test_compat_wrapper_expires_after_run_cleanup(monkeypatch):
    worker, shm, fake_c_worker = _make_started_sim_worker()
    resources = worker_module._RunResources()
    worker._building_run_resources = resources
    try:
        monkeypatch.setattr(
            worker_module, "_worker_host_mapped_region_import_sim", lambda _token, _size, _owner_token: 55
        )
        region = worker._create_worker_chip_region(0, 64, 128)
        assert region.expired is False
        worker._building_run_resources = None
        worker._region_instance_registry.cleanup_run(resources)
        assert region.expired is True
        with pytest.raises(RuntimeError, match="expired"):
            region.descriptor_scalars()
        assert fake_c_worker.release_calls == [(0, 1)]
        assert worker._region_instance_registry._instances == {}
    finally:
        worker._building_run_resources = None
        worker._close_worker_chip_orch_comm()
        shm.close()
        shm.unlink()


def test_compat_free_is_logical_and_keeps_region_instance_live(monkeypatch):
    worker, shm, fake_c_worker = _make_started_sim_worker()
    try:
        monkeypatch.setattr(
            worker_module, "_worker_host_mapped_region_import_sim", lambda _token, _size, _owner_token: 55
        )
        region = worker._create_worker_chip_region(0, 64, 128)
        instance = region._instance
        region.free()
        region.free()

        with pytest.raises(RuntimeError, match="has been released"):
            region.descriptor_scalars()
        assert instance._state is comm_region.RegionInstanceState.LIVE
        assert fake_c_worker.release_calls == []
        assert worker._region_instance_registry._instances[id(instance)] is instance
    finally:
        worker._close_worker_chip_orch_comm()
        shm.close()
        shm.unlink()


def test_compat_descriptor_uses_independent_local_views_and_bumped_magic(monkeypatch):
    worker, shm, fake_c_worker = _make_started_sim_worker()
    fake_c_worker.payload_base = 0x1000
    fake_c_worker.counter_base = 0x9000
    try:
        monkeypatch.setattr(
            worker_module, "_worker_host_mapped_region_import_sim", lambda _token, _size, _owner_token: 55
        )
        region = worker._create_worker_chip_region(0, 64, 128)
        scalars = region.descriptor_scalars()

        assert len(scalars) == 6
        assert scalars[0] == worker_chip_orch_comm._REGION_MAGIC_VERSION
        assert scalars[0] != 0x4C334C3200020000
        assert scalars[1] == 1
        assert scalars[2] == 0x1000
        assert scalars[3] == 64
        assert scalars[4] == 0x9000
        assert scalars[5] == 128
        assert region._instance is not None
        assert region._instance._payload_part is not region._instance._counter_part
    finally:
        worker._close_worker_chip_orch_comm()
        shm.close()
        shm.unlink()


def test_public_create_worker_chip_region_uses_admitted_w2_plan_and_two_imports(monkeypatch):
    worker, shm, fake_c_worker = _make_started_sim_worker()
    imports: list[tuple[str, int]] = []
    monkeypatch.setattr(
        worker_module,
        "_worker_host_mapped_region_import_sim",
        lambda token, mapping_bytes, owner_token: imports.append((str(token), int(mapping_bytes)))
        or (100 + len(imports)),
    )
    try:
        orch = Orchestrator(MagicMock(), worker)
        region = orch.create_worker_chip_region(worker_id=0, payload_bytes=64, counter_bytes=128)
        plan = region._instance.plan
        assert isinstance(plan, ce.BackendPlan)
        assert isinstance(plan.topology_plan, ce.SingleOwnerPlan)
        registry = worker._get_endpoint_registry()
        provider = registry.record_for(plan.topology_plan.provider_endpoint)
        consumer = next(registry.record_for(member) for member in plan.ordered_members if member != provider.identity)
        assert provider.path == "L3/L2[0]"
        assert provider.deployment is ce.DEVICE_AICPU
        assert consumer.path == "L3"
        assert consumer.deployment is ce.HOST_CPU
        assert set(plan.ordered_members) == {provider.identity, consumer.identity}
        assert [item[0] for item in imports] == ["sim-direct-1-p", "sim-direct-1-c"]
        assert [item[1] for item in imports] == [64, 128]
        assert region._instance._payload_part is not region._instance._counter_part
        scalars = region.descriptor_scalars()
        assert len(scalars) == 6
        assert scalars[0] == worker_chip_orch_comm._REGION_MAGIC_VERSION
        assert scalars[0] != 0x4C334C3200020000
        region.free()
        assert fake_c_worker.release_calls == []
        assert region._instance._state is comm_region.RegionInstanceState.LIVE
        worker._sweep_region_instances()
        assert fake_c_worker.release_calls == [(0, 1)]
        assert worker._region_instance_registry._instances == {}
    finally:
        worker._close_worker_chip_orch_comm()
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
            buf = buf_t.from_address(int(host.base))
            for i in range(16):
                buf[i] = (i + 41) & 0xFF
            region = orch_handle.create_worker_chip_region(worker_id=0, payload_bytes=16, counter_bytes=128)
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
            region = orch_handle.create_worker_chip_region(worker_id=0, payload_bytes=16, counter_bytes=128)
            with pytest.raises(TimeoutError, match="observed=0"):
                region.counter(0).wait(1, WaitCmp.EQ, timeout=0.001)
            assert region.descriptor_scalars()[1] != 0
            region.free()
            region.free()

        worker.run(orch)
    finally:
        worker.close()


@pytest.mark.filterwarnings("error::pytest.PytestUnraisableExceptionWarning")
@pytest.mark.parametrize("platform", ["a2a3sim", "a5sim"])
def test_sim_public_api_two_shm_objects_and_two_consecutive_lifecycles(platform, monkeypatch):
    try:
        RuntimeBuilder(platform=platform).get_binaries("tensormap_and_ringbuffer")
    except FileNotFoundError as e:
        pytest.skip(f"{platform} runtime binaries unavailable: {e}")

    imports: list[tuple[str, int]] = []
    original_import = worker_module._worker_host_mapped_region_import_sim

    def spy_import(token, mapping_bytes, owner_token):
        imports.append((str(token), int(mapping_bytes)))
        return original_import(token, mapping_bytes, owner_token)

    monkeypatch.setattr(worker_module, "_worker_host_mapped_region_import_sim", spy_import)

    worker = Worker(
        level=3,
        device_ids=[0],
        platform=platform,
        runtime="tensormap_and_ringbuffer",
        num_sub_workers=0,
    )
    worker.init()
    leftover_names: list[str] = []
    try:

        def orch(orch_handle, _args, _cfg):
            host = orch_handle.alloc([16], DataType.UINT8)
            buf_t = ctypes.c_uint8 * 16
            buf = buf_t.from_address(int(host.base))
            for i in range(16):
                buf[i] = (i + 7) & 0xFF
            region = orch_handle.create_worker_chip_region(worker_id=0, payload_bytes=16, counter_bytes=128)
            region.payload_write(0, host)
            for i in range(16):
                buf[i] = 0
            region.payload_read(0, host)
            assert bytes(buf) == bytes((i + 7) & 0xFF for i in range(16))
            region.counter(0).notify(3, NotifyOp.Set)
            assert region.counter(0).test(3, WaitCmp.EQ).matched
            region.free()

        worker.run(orch)
        first_pair = imports[:2]
        leftover_names.extend(name for name, _size in first_pair)
        assert len(first_pair) == 2
        assert first_pair[0][0] != first_pair[1][0]
        assert {first_pair[0][1], first_pair[1][1]} == {16, 128}
        assert worker._region_instance_registry._instances == {}
        assert worker._region_instance_registry._instances == {}
        for name in leftover_names:
            with pytest.raises(FileNotFoundError):
                SharedMemory(name=name)

        worker.run(orch)
        second_pair = imports[2:]
        leftover_names.extend(name for name, _size in second_pair)
        assert len(second_pair) == 2
        assert second_pair[0][0] != second_pair[1][0]
        assert {name for name, _size in first_pair}.isdisjoint({name for name, _size in second_pair})
        assert worker._region_instance_registry._instances == {}
        assert worker._region_instance_registry._instances == {}
        for name in leftover_names:
            with pytest.raises(FileNotFoundError):
                SharedMemory(name=name)
    finally:
        worker.close()
        assert worker._region_instance_registry._instances == {}
        for name in leftover_names:
            with pytest.raises(FileNotFoundError):
                SharedMemory(name=name)
