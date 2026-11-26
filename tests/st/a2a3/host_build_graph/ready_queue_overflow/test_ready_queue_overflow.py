#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A2/A3 HBG rejects an oversized ready queue during Host bind, before launch."""

from pathlib import Path

import pytest
from simpler.task_interface import ArgDirection, ChipCallable, TaskArgs
from simpler.worker import Worker

from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.log_config import configure_logging
from simpler_setup.tools.strace_timing import parse_spans

RUNTIME = "host_build_graph"
ORCHESTRATION = Path(__file__).parent / "kernels/orchestration/ready_queue_overflow_orch.cpp"


@pytest.mark.platforms(["a2a3sim", "a2a3"])
@pytest.mark.device_count(1)
@pytest.mark.runtime(RUNTIME)
def test_bind_overflow_precedes_launch(st_platform, st_device_ids, capfd, drain_host_log):
    configure_logging("timing")
    binary = KernelCompiler(platform=st_platform).compile_orchestration(
        runtime_name=RUNTIME, source_path=str(ORCHESTRATION)
    )
    callable_obj = ChipCallable.build(
        signature=[ArgDirection.SCALAR], func_name="aicpu_orchestration_entry", binary=binary, children=[]
    )
    worker = Worker(level=2, platform=st_platform, runtime=RUNTIME, device_id=st_device_ids[0])
    try:
        handle = worker.register(callable_obj)
        worker.init()
        drain_host_log(capfd)
        # 257 replays at 128 ready tasks each cross the 32768 limit.
        args = TaskArgs()
        args.add_scalar(257)
        with pytest.raises(RuntimeError, match=r"failed with code -104\b"):
            worker.run(handle, args)
        log = drain_host_log(capfd)
        assert "orch_error_code=0 sched_error_code=104 runtime_status=-104" in log
        assert "sched_error_code=104 READY_QUEUE_OVERFLOW" in log
        spans = {span.name for span in parse_spans(log.splitlines())}
        post_bind_spans = {"chip.run.publish_image", "chip.run.runner_run"}
        assert "chip.run.bind" in spans
        assert post_bind_spans.isdisjoint(spans)

        # A valid run proves both post-bind markers are enabled for this worker.
        args = TaskArgs()
        args.add_scalar(1)
        worker.run(handle, args)
        spans = {span.name for span in parse_spans(drain_host_log(capfd).splitlines())}
        assert post_bind_spans <= spans
    finally:
        worker.close()
