#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""L4 tree drives a child L3 orch that creates its own SPSC queue."""

from __future__ import annotations

import os
import tempfile

import pytest
from simpler.worker import Worker

from ._helpers import (
    _RUNTIME,
    SUCCESS_PLATFORMS,
    assert_driver_sentinels,
    build_chip_callable,
    close_owned_workers,
    create_queue,
    echo_once,
    region_lifecycle,
    sentinel_task_args,
    stream_config,
    submit_queue,
)


@pytest.mark.platforms(SUCCESS_PLATFORMS)
@pytest.mark.device_count(1)
@pytest.mark.runtime(_RUNTIME)
def test_child_l3_owns_queue_inside_l4_tree(st_platform, st_device_ids):
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
    tx_path = os.path.join(tempfile.mkdtemp(prefix="child-l3-queue-"), "transactions")
    primary = None
    try:
        chip_handle = l3.register(chip_callable)
        lifecycle = region_lifecycle()
        recorder = lifecycle.install_lifecycle_recorder(l3)
        worker = Worker(level=4, num_sub_workers=0, platform=st_platform, runtime=_RUNTIME)
        l3_worker_id = int(worker.add_worker(l3))
        worker.register(chip_callable)

        def l3_own(orch, args, cfg):
            slot = int(args.scalar(1))
            assert_driver_sentinels(args, slot)
            queue = create_queue(orch)
            assert int(orch._worker.level) == 3
            assert int(queue._endpoint_binding.transaction_id) == slot
            with open(tx_path, "a", encoding="ascii") as record:
                record.write(f"{int(queue._endpoint_binding.transaction_id)}\n")
            submit_queue(orch, chip_handle, queue, cfg)
            echo_once(queue, free=slot == 2)

        l3_orch = worker.register(l3_own)
        worker.init()

        def l4_drive(orch_handle, _args, cfg, slot: int):
            assert worker._region_instance_registry._instances == {}
            orch_handle.submit_next_level(l3_orch, sentinel_task_args(slot), cfg, worker=l3_worker_id)

        worker.run(lambda orch, args, cfg: l4_drive(orch, args, cfg, 1), args=None, config=stream_config())
        worker.run(lambda orch, args, cfg: l4_drive(orch, args, cfg, 2), args=None, config=stream_config())
        assert [int(line) for line in open(tx_path, encoding="ascii")] == [1, 2]
        assert worker._region_instance_registry._instances == {}
        lifecycle.assert_lifecycle_rounds(recorder, 2)
    except BaseException as exc:
        primary = exc
        raise
    finally:
        try:
            close_owned_workers(primary, worker, l3)
        finally:
            if recorder is not None and recorder.restore is not None:
                recorder.restore()
