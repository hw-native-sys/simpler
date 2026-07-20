#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Negative ST for the AICore op-execution timeout chain (regression for PR #718).

Hardware-only: dispatches an AIC kernel that spins forever. This test pins
the CI-tight timeout chain (AICPU scheduler 2 s, STARS op watchdog 3 s,
host stream sync 4 s) so the hang surfaces in single-digit seconds rather
than deadlocking. Sim variants are excluded
because the simulator has no STARS watchdog — a ``while(true)`` kernel
would wedge the sim.
"""

import os
import time

import pytest
from simpler.task_interface import CallConfig, ChipCallable, ChipStorageTaskArgs, CoreCallable
from simpler.worker import Worker

from simpler_setup.elf_parser import extract_text_section
from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.log_config import configure_logging
from simpler_setup.pto_isa import ensure_pto_isa_root

HERE = os.path.dirname(os.path.abspath(__file__))
RUNTIME = "tensormap_and_ringbuffer"
ORCH_SRC = os.path.join(HERE, "kernels/orchestration/aicore_op_timeout_orch.cpp")
AIC_SRC = os.path.join(HERE, "kernels/aic/kernel_hang.cpp")
FUNC_AIC_HANG = 0


def _build_chip_callable(platform: str) -> ChipCallable:
    kc = KernelCompiler(platform=platform)
    pto_isa_root = ensure_pto_isa_root()
    inc_dirs = kc.get_orchestration_include_dirs(RUNTIME)

    orch_bytes = kc.compile_orchestration(runtime_name=RUNTIME, source_path=ORCH_SRC)
    aic_bytes = kc.compile_incore(AIC_SRC, core_type="aic", pto_isa_root=pto_isa_root, extra_include_dirs=inc_dirs)
    # Onboard expects the .text section of the AIC ELF; sim consumes the full ELF.
    if not platform.endswith("sim"):
        aic_bytes = extract_text_section(aic_bytes)

    aic_core = CoreCallable.build(signature=[], binary=aic_bytes)
    return ChipCallable.build(
        signature=[],
        func_name="aicpu_orchestration_entry",
        binary=orch_bytes,
        children=[(FUNC_AIC_HANG, aic_core)],
    )


@pytest.mark.platforms(["a2a3", "a5"])
@pytest.mark.device_count(1)
@pytest.mark.runtime(RUNTIME)
@pytest.mark.timeout(60)
def test_aicore_op_timeout_surfaces_as_runtime_error(st_platform, st_device_ids, monkeypatch):
    configure_logging("error")
    monkeypatch.setenv("SIMPLER_SCHEDULER_TIMEOUT_MS", "2000")
    monkeypatch.setenv("SIMPLER_OP_EXECUTE_TIMEOUT_US", "3000000")
    monkeypatch.setenv("SIMPLER_STREAM_SYNC_TIMEOUT_MS", "4000")

    chip_callable = _build_chip_callable(st_platform)
    worker = Worker(level=2, platform=st_platform, runtime=RUNTIME, device_id=int(st_device_ids[0]))
    handle = worker.register(chip_callable)
    worker.init()
    try:
        config = CallConfig()
        config.block_dim = 1
        # >=2 so the orchestration thread and the scheduler thread don't fight
        # for a single AICPU; smaller configs may not dispatch the AIC task.
        config.aicpu_thread_num = 2

        t0 = time.monotonic()
        # Acceptable error codes for the STARS-killed AICore op. Device status
        # takes precedence when finalization can read it; otherwise the host
        # falls back to whichever stream-sync error surfaced first:
        #   -100   = PTO2 scheduler timeout — the device classified the stalled
        #            AIC task before finalization read the shared header.
        #   507046 = ACL_ERROR_RT_STREAM_SYNC_TIMEOUT — AICore stream's 4 s
        #            sync budget fires before AICPU sync notices.
        #   507018 = ACL_ERROR_RT_AICPU_EXCEPTION — AICPU stream sync surfaces
        #            the AICore failure as an AICPU exception when the
        #            orchestration kernel detects the dead AIC task first.
        #   507000 = ACL_ERROR_RT_INTERNAL_ERROR — same detection on a5,
        #            mapped through a different code path.
        # The generic codes are valid on both a2a3 and a5: the timing race is
        # between AICPU and AICore stream sync on host, not arch-specific. The
        # regression we care about is that the timeout chain reaps the hang in
        # single-digit seconds and surfaces either the device classification or
        # a valid host fallback rather than deadlocking.
        with pytest.raises(RuntimeError, match=r"run failed with code (-100|507(046|018|000))"):
            worker.run(handle, ChipStorageTaskArgs(), config)
        elapsed = time.monotonic() - t0

        # CI-tight env keeps the timeout chain short; default local values are
        # intentionally larger for production workloads.
        # If this fires, the timeout chain is broken (or absent).
        assert elapsed < 10, f"run() took {elapsed:.1f}s — timeout chain did not fire"
    finally:
        worker.close()
