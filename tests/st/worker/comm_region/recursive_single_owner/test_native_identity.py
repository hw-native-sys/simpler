#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Native Region helper failures return to the owning allocation."""

from __future__ import annotations

import ctypes
import multiprocessing

import pytest
from simpler.worker import Worker

from ._helpers import (
    _PAYLOAD_BYTES,
    _RUNTIME,
    FAULT_PLATFORMS,
    _install_release_probe,
    _stream_config,
    assert_sentinel_order,
    build_chip_callable,
    build_region_task_args,
    close_owned_workers,
    create_live_region,
    make_l3_forward,
    publish_channel_header,
    sentinel_task_args,
)

_REGION_MARKER = "Region endpoint error"
_INPUT_OFFSET_INDEX = 8


def _exception_text(exc: BaseException) -> str:
    parts: list[str] = []
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        parts.append(str(current))
        current = current.__cause__ if current.__cause__ is not None else current.__context__
    return "\n".join(parts)


def _trusted_marker(session: bytes, transaction_id: int) -> str:
    bits = int.from_bytes(bytes(session), "little")
    return f"{_REGION_MARKER} session=0x{bits:016x} transaction={int(transaction_id)}"


def _open_l3(platform: str, device_id: int):
    worker = Worker(
        level=3,
        device_ids=[int(device_id)],
        num_sub_workers=0,
        platform=platform,
        runtime=_RUNTIME,
    )
    try:
        handle = worker.register(build_chip_callable(platform))
        worker.init()
        return worker, handle
    except BaseException as primary:
        close_owned_workers(primary, worker)
        raise


def _open_l4(platform: str, device_id: int):
    l3 = Worker(
        level=3,
        device_ids=[int(device_id)],
        num_sub_workers=0,
        platform=platform,
        runtime=_RUNTIME,
    )
    worker = None
    try:
        chip = build_chip_callable(platform)
        chip_handle = l3.register(chip)
        worker = Worker(level=4, num_sub_workers=0, platform=platform, runtime=_RUNTIME)
        l3_worker_id = worker.add_worker(l3)
        worker.register(chip)
        l3_orch = worker.register(make_l3_forward(chip_handle))
        worker.init()
        return worker, l3, l3_orch, int(l3_worker_id)
    except BaseException as primary:
        close_owned_workers(primary, worker, l3)
        raise


def _run_region_failure(worker: Worker, orch) -> str:
    with pytest.raises(BaseException) as excinfo:  # noqa: PT011
        worker.run(orch, args=None, config=_stream_config())
    return _exception_text(excinfo.value)


@pytest.mark.platforms(FAULT_PLATFORMS)
@pytest.mark.device_count(1)
@pytest.mark.runtime(_RUNTIME)
def test_l3_view_failure_returns_trusted_identity(st_platform, st_device_ids):
    worker, chip_handle = _open_l3(st_platform, int(st_device_ids[0]))
    releases = _install_release_probe(worker)
    region_ref: dict[str, object] = {}
    try:

        def orch(orch_handle, _args, cfg):
            region = create_live_region(orch_handle, "L3/L2[0]")
            region_ref["region"] = region
            orch_handle.submit_next_level(
                chip_handle,
                build_region_task_args(region, {_INPUT_OFFSET_INDEX: _PAYLOAD_BYTES}),
                cfg,
                worker=0,
            )
            publish_channel_header(orch_handle, region, 1, 1)

        text = _run_region_failure(worker, orch)
        region = region_ref["region"]
        session, transaction_id = region._allocation_identity
        assert _trusted_marker(session, transaction_id) in text
        assert region._data_plane_error is not None
        assert len(releases) == 1
    finally:
        worker.close()
    assert len(releases) == 1


@pytest.mark.platforms(FAULT_PLATFORMS)
@pytest.mark.device_count(1)
@pytest.mark.runtime(_RUNTIME)
def test_l3_bad_transaction_is_untrusted(st_platform, st_device_ids):
    worker, chip_handle = _open_l3(st_platform, int(st_device_ids[0]))
    releases = _install_release_probe(worker)
    region_ref: dict[str, object] = {}
    try:

        def orch(orch_handle, _args, cfg):
            region = create_live_region(orch_handle, "L3/L2[0]")
            region_ref["region"] = region
            orch_handle.submit_next_level(chip_handle, build_region_task_args(region, {7: 0}), cfg, worker=0)

        text = _run_region_failure(worker, orch)
        region = region_ref["region"]
        assert f"{_REGION_MARKER} identity=untrusted" in text
        assert "session=0x" not in text
        assert region._data_plane_error is None
        assert len(releases) == 1
    finally:
        worker.close()


@pytest.mark.platforms(FAULT_PLATFORMS)
@pytest.mark.device_count(1)
@pytest.mark.runtime(_RUNTIME)
def test_l3_seq_mismatch_stays_generic(st_platform, st_device_ids):
    worker, chip_handle = _open_l3(st_platform, int(st_device_ids[0]))
    releases = _install_release_probe(worker)
    region_ref: dict[str, object] = {}
    try:

        def orch(orch_handle, _args, cfg):
            region = create_live_region(orch_handle, "L3/L2[0]")
            region_ref["region"] = region
            orch_handle.submit_next_level(chip_handle, build_region_task_args(region), cfg, worker=0)
            publish_channel_header(orch_handle, region, 99, 1)

        text = _run_region_failure(worker, orch)
        assert _REGION_MARKER not in text
        assert region_ref["region"]._data_plane_error is None
        assert len(releases) == 1
    finally:
        worker.close()


@pytest.mark.platforms(FAULT_PLATFORMS)
@pytest.mark.device_count(1)
@pytest.mark.runtime(_RUNTIME)
def test_l4_view_failure_returns_trusted_identity(st_platform, st_device_ids):
    worker, l3, l3_orch, l3_worker_id = _open_l4(st_platform, int(st_device_ids[0]))
    releases = _install_release_probe(worker)
    region_ref: dict[str, object] = {}
    try:

        def orch(orch_handle, _args, cfg):
            region = create_live_region(orch_handle, f"L4/L3[{l3_worker_id}]/L2[0]")
            region_ref["region"] = region
            orch_handle.submit_next_level(
                l3_orch,
                build_region_task_args(region, {_INPUT_OFFSET_INDEX: _PAYLOAD_BYTES}, sentinels=True),
                cfg,
                worker=l3_worker_id,
            )
            publish_channel_header(orch_handle, region, 1, 1)

        text = _run_region_failure(worker, orch)
        region = region_ref["region"]
        session, transaction_id = region._allocation_identity
        assert _trusted_marker(session, transaction_id) in text
        assert region._data_plane_error is not None
        assert l3._region_instance_registry._instances == {}
        assert len(releases) == 1
    finally:
        close_owned_workers(None, worker, l3)


@pytest.mark.platforms(FAULT_PLATFORMS)
@pytest.mark.device_count(1)
@pytest.mark.runtime(_RUNTIME)
def test_child_l3_view_failure_stays_on_child_scope(st_platform, st_device_ids):
    chip = build_chip_callable(st_platform)
    l3 = Worker(
        level=3,
        device_ids=[int(st_device_ids[0])],
        num_sub_workers=0,
        platform=st_platform,
        runtime=_RUNTIME,
    )
    worker = None
    matched = multiprocessing.Array(ctypes.c_uint64, 2, lock=True)
    try:
        chip_handle = l3.register(chip)
        original = l3._poison_region_from_endpoint_error

        def _watch(exc, resources=None):
            result = original(exc, resources)
            transaction_id = 0
            for instance in l3._region_instance_registry._instances.values():
                if instance._data_plane_error is not None:
                    transaction_id = int(instance._delegated_transaction_id)
            with matched.get_lock():
                matched[0] = 1 if result else 0
                matched[1] = transaction_id
            return result

        l3._poison_region_from_endpoint_error = _watch
        worker = Worker(level=4, num_sub_workers=0, platform=st_platform, runtime=_RUNTIME)
        l3_worker_id = int(worker.add_worker(l3))
        worker.register(chip)

        def l3_own(orch, args, cfg):
            assert_sentinel_order(args, 1)
            region = create_live_region(orch, "L3/L2[0]")
            orch.submit_next_level(
                chip_handle,
                build_region_task_args(region, {_INPUT_OFFSET_INDEX: _PAYLOAD_BYTES}),
                cfg,
                worker=0,
            )
            publish_channel_header(orch, region, 1, 1)

        l3_orch = worker.register(l3_own)
        worker.init()

        def l4_drive(orch_handle, _args, cfg):
            orch_handle.submit_next_level(l3_orch, sentinel_task_args(1), cfg, worker=l3_worker_id)

        text = _run_region_failure(worker, l4_drive)
        assert f"{_REGION_MARKER} session=0x" in text
        assert "transaction=1\n" in text or text.rstrip().endswith("transaction=1")
        assert int(matched[0]) == 1
        assert int(matched[1]) == 1
        assert worker._region_instance_registry._instances == {}
    finally:
        close_owned_workers(None, worker, l3)
