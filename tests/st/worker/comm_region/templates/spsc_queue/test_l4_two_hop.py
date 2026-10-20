#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""L4→L3→L2 SPSC queue: public creation, ordered scalars, fresh allocation."""

from __future__ import annotations

import pytest
from simpler.worker import Worker

from ._helpers import (
    _RUNTIME,
    _SENTINEL_HEAD,
    _SENTINEL_TAIL,
    SUCCESS_PLATFORMS,
    build_chip_callable,
    close_owned_workers,
    create_queue,
    echo_once,
    forwarded_task_args,
    queue_task_args,
    region_lifecycle,
    stream_config,
)


@pytest.mark.platforms(SUCCESS_PLATFORMS)
@pytest.mark.device_count(1)
@pytest.mark.runtime(_RUNTIME)
def test_l4_two_hop(st_platform, st_device_ids):
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
    transactions: list[int] = []
    primary = None
    try:
        chip_handle = l3.register(chip_callable)
        worker = Worker(level=4, num_sub_workers=0, platform=st_platform, runtime=_RUNTIME)
        lifecycle = region_lifecycle()
        recorder = lifecycle.install_lifecycle_recorder(worker)
        l3_worker_id = int(worker.add_worker(l3))
        worker.register(chip_callable)

        def l3_forward(orch, args, cfg):
            assert int(args.scalar(0)) == _SENTINEL_HEAD
            assert int(args.scalar(args.scalar_count() - 1)) == _SENTINEL_TAIL
            forwarded = forwarded_task_args(args)
            assert int(forwarded.scalar_count()) == 10
            orch.submit_next_level(chip_handle, forwarded, cfg, worker=0)

        l3_orch = worker.register(l3_forward)
        worker.init()

        def drive(orch_handle, _args, cfg, *, free: bool):
            queue = create_queue(orch_handle, host_path="L4", peer_path=f"L4/L3[{l3_worker_id}]/L2[0]")
            transactions.append(int(queue._endpoint_binding.transaction_id))
            orch_handle.submit_next_level(l3_orch, queue_task_args(queue, sentinels=True), cfg, worker=l3_worker_id)
            echo_once(queue, free=free)

        worker.run(lambda orch, args, cfg: drive(orch, args, cfg, free=False), args=None, config=stream_config())
        worker.run(lambda orch, args, cfg: drive(orch, args, cfg, free=True), args=None, config=stream_config())
        assert transactions == [1, 2]
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
