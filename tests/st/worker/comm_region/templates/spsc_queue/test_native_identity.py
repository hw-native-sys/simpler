#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Native SPSC helper failures return to the owning allocation."""

from __future__ import annotations

import ctypes
import multiprocessing
import struct

import pytest
from simpler.worker import Worker

from ._helpers import (
    _RUNTIME,
    FAULT_PLATFORMS,
    assert_driver_sentinels,
    build_chip_callable,
    close_owned_workers,
    create_queue,
    echo_payload,
    enqueue_bytes,
    forwarded_task_args,
    queue_task_args,
    sentinel_task_args,
    stream_config,
    submit_queue,
)

_QUEUE_MARKER = "SPSC queue endpoint error"


def _exception_text(exc: BaseException) -> str:
    parts: list[str] = []
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        parts.append(str(current))
        current = current.__cause__ if current.__cause__ is not None else current.__context__
    return "\n".join(parts)


def _trusted_marker(queue) -> str:
    session, transaction_id = queue._instance._allocation_identity
    bits = int.from_bytes(bytes(session), "little")
    return f"{_QUEUE_MARKER} session=0x{bits:016x} transaction={int(transaction_id)}"


def _smash(queue) -> None:
    enqueue_bytes(queue, echo_payload(b"bad-seq"))
    smashed = bytearray(8)
    struct.pack_into("<Q", smashed, 0, 99)
    queue._instance.payload_write(queue.layout.input_desc_offset, smashed, 8)


def _run_failure(worker: Worker, orch) -> str:
    with pytest.raises(BaseException) as excinfo:  # noqa: PT011
        worker.run(orch, args=None, config=stream_config())
    return _exception_text(excinfo.value)


@pytest.mark.platforms(FAULT_PLATFORMS)
@pytest.mark.device_count(1)
@pytest.mark.runtime(_RUNTIME)
def test_l4_descriptor_failure_returns_trusted_identity(st_platform, st_device_ids):
    chip = build_chip_callable(st_platform)
    l3 = Worker(
        level=3,
        device_ids=[int(st_device_ids[0])],
        num_sub_workers=0,
        platform=st_platform,
        runtime=_RUNTIME,
    )
    worker = None
    remembered: list = []
    primary = None
    try:
        chip_handle = l3.register(chip)
        worker = Worker(level=4, num_sub_workers=0, platform=st_platform, runtime=_RUNTIME)
        l3_worker_id = int(worker.add_worker(l3))
        worker.register(chip)

        def l3_forward(orch, args, cfg):
            orch.submit_next_level(chip_handle, forwarded_task_args(args), cfg, worker=0)

        l3_orch = worker.register(l3_forward)
        worker.init()

        def orch(orch_handle, _args, cfg):
            queue = create_queue(orch_handle, host_path="L4", peer_path=f"L4/L3[{l3_worker_id}]/L2[0]")
            remembered.append(queue)
            _smash(queue)
            orch_handle.submit_next_level(l3_orch, queue_task_args(queue, sentinels=True), cfg, worker=l3_worker_id)

        text = _run_failure(worker, orch)
        queue = remembered[0]
        assert _trusted_marker(queue) in text
        assert queue._instance._data_plane_error is not None
        assert l3._region_instance_registry._instances == {}
    except BaseException as exc:
        primary = exc
        raise
    finally:
        close_owned_workers(primary, worker, l3)


@pytest.mark.platforms(FAULT_PLATFORMS)
@pytest.mark.device_count(1)
@pytest.mark.runtime(_RUNTIME)
def test_child_l3_descriptor_failure_stays_on_child_scope(st_platform, st_device_ids):
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
    primary = None
    try:
        chip_handle = l3.register(chip)
        original = l3._poison_spsc_queue_from_endpoint_error

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

        l3._poison_spsc_queue_from_endpoint_error = _watch
        worker = Worker(level=4, num_sub_workers=0, platform=st_platform, runtime=_RUNTIME)
        l3_worker_id = int(worker.add_worker(l3))
        worker.register(chip)

        def l3_own(orch, args, cfg):
            assert_driver_sentinels(args, 1)
            queue = create_queue(orch)
            _smash(queue)
            submit_queue(orch, chip_handle, queue, cfg)

        l3_orch = worker.register(l3_own)
        worker.init()

        def l4_drive(orch_handle, _args, cfg):
            orch_handle.submit_next_level(l3_orch, sentinel_task_args(1), cfg, worker=l3_worker_id)

        text = _run_failure(worker, l4_drive)
        assert f"{_QUEUE_MARKER} session=0x" in text
        assert "transaction=1\n" in text or text.rstrip().endswith("transaction=1")
        assert int(matched[0]) == 1
        assert int(matched[1]) == 1
        assert worker._region_instance_registry._instances == {}
    except BaseException as exc:
        primary = exc
        raise
    finally:
        close_owned_workers(primary, worker, l3)
