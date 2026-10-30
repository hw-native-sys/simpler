#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""L4 tree drives a child L3 orch that creates its own region."""

from __future__ import annotations

import os
import tempfile

import pytest
from simpler.worker import Worker

from ._helpers import (
    _PAYLOAD_BYTES,
    _RUNTIME,
    SUCCESS_PLATFORMS,
    _handshake,
    _stream_config,
    assert_lifecycle_rounds,
    assert_sentinel_order,
    build_chip_callable,
    close_owned_workers,
    create_live_region,
    install_lifecycle_recorder,
    make_l3_submit,
    sentinel_task_args,
)


@pytest.mark.platforms(SUCCESS_PLATFORMS)
@pytest.mark.device_count(1)
@pytest.mark.runtime(_RUNTIME)
def test_child_l3_owns_region_inside_l4_tree(st_platform, st_device_ids):
    chip_callable = build_chip_callable(st_platform)
    l3 = Worker(
        level=3,
        device_ids=[int(st_device_ids[0])],
        num_sub_workers=0,
        platform=st_platform,
        runtime=_RUNTIME,
    )
    worker = None
    recorder = None
    tx_path = os.path.join(tempfile.mkdtemp(prefix="child-l3-region-"), "transactions")
    primary = None
    try:
        chip_handle = l3.register(chip_callable)
        recorder = install_lifecycle_recorder(l3)
        worker = Worker(level=4, num_sub_workers=0, platform=st_platform, runtime=_RUNTIME)
        l3_worker_id = worker.add_worker(l3)
        worker.register(chip_callable)
        submit_chip = make_l3_submit(chip_handle)

        def l3_own(orch, args, cfg):
            slot = int(args.scalar(1))
            assert_sentinel_order(args, slot)
            region = create_live_region(orch, "L3/L2[0]")
            assert int(orch._worker.level) == 3
            assert int(region._delegated_transaction_id) == slot
            assert int(region.layout.payload_bytes) == _PAYLOAD_BYTES
            with open(tx_path, "a", encoding="ascii") as record:
                record.write(f"{int(region._delegated_transaction_id)}\n")
            submit_chip(orch, region, cfg)
            _handshake(orch, region, 1)
            if slot == 1:
                return
            region.free()
            try:
                region.counter(0)
            except RuntimeError as exc:
                assert "released" in str(exc)
            else:
                raise AssertionError("child region counter access after free succeeded")

        l3_orch_handle = worker.register(l3_own)
        worker.init()

        def l4_drive(orch_handle, _args, cfg, slot: int):
            assert worker._region_instance_registry._instances == {}
            orch_handle.submit_next_level(l3_orch_handle, sentinel_task_args(slot), cfg, worker=int(l3_worker_id))

        worker.run(lambda orch, args, cfg: l4_drive(orch, args, cfg, 1), args=None, config=_stream_config())
        worker.run(lambda orch, args, cfg: l4_drive(orch, args, cfg, 2), args=None, config=_stream_config())
        assert [int(line) for line in open(tx_path, encoding="ascii")] == [1, 2]
        assert worker._region_instance_registry._instances == {}
        assert_lifecycle_rounds(recorder, 2)
    except BaseException as exc:
        primary = exc
        raise
    finally:
        try:
            close_owned_workers(primary, worker, l3)
        finally:
            if recorder is not None and recorder.restore is not None:
                recorder.restore()
