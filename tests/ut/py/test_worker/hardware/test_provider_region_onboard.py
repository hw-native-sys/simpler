# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: PLC0415
"""Onboard proof that one provider region owns two independent VMM allocations."""

from __future__ import annotations

import ctypes

import pytest
from simpler import comm_region
from simpler import worker as worker_module
from simpler.task_interface import DataType
from simpler.worker import Worker
from simpler.worker_chip_orch_comm import NotifyOp, WaitCmp

from simpler_setup.runtime_builder import RuntimeBuilder

_RUNTIME = "tensormap_and_ringbuffer"
_PAYLOAD_BYTES = 16
_COUNTER_BYTES = 128
_HOST_MARKER = 0x11


def _require_runtime(platform: str) -> None:
    try:
        RuntimeBuilder(platform=platform).get_binaries(_RUNTIME)
    except (FileNotFoundError, RuntimeError) as e:
        pytest.skip(f"{platform} runtime binaries unavailable: {e}")


def _install_lifecycle_spies(monkeypatch) -> list[tuple]:
    events: list[tuple] = []
    original_import = worker_module._worker_host_mapped_region_import_onboard
    original_close = comm_region._worker_host_mapped_region_close
    original_release = Worker._dispatch_delegated_release

    def spy_import(device_id, shareable_handle, mapping_bytes, owner_token):
        handle = original_import(device_id, shareable_handle, mapping_bytes, owner_token)
        events.append(("import", int(handle), int(shareable_handle), int(mapping_bytes)))
        return handle

    def spy_close(handle):
        events.append(("close", int(handle)))
        return original_close(handle)

    def spy_release(self, *, session_instance_id, transaction_id, provider_path):
        result = original_release(
            self,
            session_instance_id=session_instance_id,
            transaction_id=transaction_id,
            provider_path=provider_path,
        )
        events.append(("release", int(result.provider_resource_id)))
        return result

    monkeypatch.setattr(worker_module, "_worker_host_mapped_region_import_onboard", spy_import)
    monkeypatch.setattr(comm_region, "_worker_host_mapped_region_close", spy_close)
    monkeypatch.setattr(Worker, "_dispatch_delegated_release", spy_release)
    return events


def _assert_independent_handles(imports: list[tuple[int, int, int]]) -> None:
    assert len(imports) == 2
    first_handle, first_shareable, first_bytes = imports[0]
    second_handle, second_shareable, second_bytes = imports[1]
    assert first_handle != second_handle
    assert first_shareable != second_shareable
    # Import order is PAYLOAD then COUNTER. mapping_bytes covers each logical
    # span plus VMM granularity and need not equal the requested sizes.
    assert first_bytes >= _PAYLOAD_BYTES
    assert second_bytes >= _COUNTER_BYTES


def _assert_run_teardown(events: list[tuple], resource_id: int) -> None:
    assert [item[0] for item in events] == ["import", "import", "close", "close", "release"]
    _assert_independent_handles([item[1:] for item in events[:2]])
    assert {events[2][1], events[3][1]} == {events[0][1], events[1][1]}
    assert events[4][1] == resource_id


@pytest.mark.requires_hardware
@pytest.mark.platforms(["a2a3", "a5"])
@pytest.mark.device_count(1)
@pytest.mark.runtime("tensormap_and_ringbuffer")
def test_onboard_provider_region_host_two_lifecycles(st_platform, st_device_ids, monkeypatch):
    _require_runtime(st_platform)
    events = _install_lifecycle_spies(monkeypatch)
    worker = Worker(
        level=3,
        device_ids=[int(st_device_ids[0])],
        num_sub_workers=0,
        platform=st_platform,
        runtime=_RUNTIME,
    )
    worker.init()
    resource_ids: list[int] = []
    try:

        def orch(orch_handle, _args, _cfg):
            host = orch_handle.alloc([_PAYLOAD_BYTES], DataType.UINT8)
            buf = (ctypes.c_uint8 * _PAYLOAD_BYTES).from_address(int(host.base))
            for i in range(_PAYLOAD_BYTES):
                buf[i] = _HOST_MARKER
            region = orch_handle.create_worker_chip_region(
                worker_id=0, payload_bytes=_PAYLOAD_BYTES, counter_bytes=_COUNTER_BYTES
            )
            resource_ids.append(int(region.descriptor_scalars()[1]))
            assert int(region._instance._payload_attachment.native_lease) != int(
                region._instance._counter_attachment.native_lease
            )
            assert int(region._instance._payload_local_view.logical_bytes) == _PAYLOAD_BYTES
            assert int(region._instance._counter_local_view.logical_bytes) == _COUNTER_BYTES
            region.payload_write(0, host, nbytes=_PAYLOAD_BYTES)
            region.counter(0).notify(1, NotifyOp.Set)
            assert region.counter(0).test(1, WaitCmp.EQ).matched
            for i in range(_PAYLOAD_BYTES):
                buf[i] = 0
            region.payload_read(0, host, nbytes=_PAYLOAD_BYTES)
            assert bytes(buf) == bytes([_HOST_MARKER] * _PAYLOAD_BYTES)
            assert [item[0] for item in events] == ["import", "import"]
            region.free()
            assert region._instance._state is comm_region.RegionInstanceState.LIVE
            assert [item[0] for item in events] == ["import", "import"]

        worker.run(orch)
        first = events[:]
        _assert_run_teardown(first, resource_ids[0])
        assert worker._region_instance_registry._instances == {}
        assert worker._region_instance_registry._instances == {}

        events.clear()
        worker.run(orch)
        second = events[:]
        _assert_run_teardown(second, resource_ids[1])
        assert {first[0][2], first[1][2]}.isdisjoint({second[0][2], second[1][2]})
        assert resource_ids[0] != resource_ids[1]
        assert worker._region_instance_registry._instances == {}
        assert worker._region_instance_registry._instances == {}
    finally:
        worker.close()
        assert worker._region_instance_registry._instances == {}
