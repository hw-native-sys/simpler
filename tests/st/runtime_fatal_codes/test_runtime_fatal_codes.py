#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Negative STs: deliberately trigger device error conditions and assert the host can tell them apart.

Closes the gap from issue #1180's comment — most ``orch_error_code`` /
``sched_error_code`` values had no end-to-end host coverage. Each case drives one
device error to its latch (orchestrator deadlocks via a tiny
``CallConfig.runtime_env``; the scheduler stall via a lowered scheduler timeout;
the async codes via a small AIV kernel that abuses the per-task completion slab).

Two host behaviours are exercised, because they genuinely differ:

* **sim** — ``worker.run`` returns the runtime's own status, so the host sees the
  real ``code -N`` directly (and the device error class on the failure line).
* **onboard** — for the cores-killed stalls the op-execute / stream-sync watchdog
  wins the race and masks the real code with a generic CANN ``507xxx``; the
  device-classified info (orchestrator code, the #1180 ``sub_class`` for the
  scheduler stall, or the async ``sched_error_code``) still reaches the host via
  the ``validate_runtime_impl`` log line. That race is the exact scenario #1180
  exists for and only hardware reproduces it; the onboard test therefore asserts
  on the host log rather than the masked exception code.
"""

import os

import pytest
from simpler.task_interface import ArgDirection, CallConfig, ChipCallable, ChipStorageTaskArgs, CoreCallable
from simpler.worker import Worker

from simpler_setup.elf_parser import extract_text_section
from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.log_config import configure_logging
from simpler_setup.pto_isa import ensure_pto_isa_root

HERE = os.path.dirname(os.path.abspath(__file__))
RUNTIME = "tensormap_and_ringbuffer"
KERNELS = os.path.join(HERE, "kernels")
ORCH_DIR = os.path.join(KERNELS, "orchestration")

# case -> dict(orch, code, runtime_env, kernel, marker)
#   code       : runtime status the host reports in sim (orch_error_code or sched_error_code)
#   runtime_env: CallConfig.runtime_env overrides that pin the offending resource small
#   kernel     : AIV kernel (rel to kernels/) for the async cases, else None
#   marker     : substring of the validate_runtime_impl host-log line proving the
#                device error class reached the host (the assertion that holds on
#                both sim and onboard, even when onboard masks the code as 507xxx)
CASES = {
    "scope_deadlock": dict(
        orch="scope_deadlock_orch.cpp",
        code=1,
        runtime_env={"ring_task_window": 4},
        kernel=None,
        marker="orch_error_code=1",
    ),
    "heap_ring_deadlock": dict(
        orch="heap_ring_deadlock_orch.cpp",
        code=2,
        runtime_env={"ring_heap": 1024},
        kernel=None,
        marker="orch_error_code=2",
    ),
    "flow_control_deadlock": dict(
        orch="flow_control_deadlock_orch.cpp",
        code=3,
        runtime_env={"ring_task_window": 4},
        kernel=None,
        marker="orch_error_code=3",
    ),
    "dep_pool_overflow": dict(
        orch="dep_pool_overflow_orch.cpp",
        code=4,
        runtime_env={"ring_dep_pool": 4},
        kernel=None,
        marker="orch_error_code=4",
    ),
    "invalid_args": dict(
        orch="invalid_args_orch.cpp",
        code=5,
        runtime_env={},
        kernel=None,
        marker="orch_error_code=5",
    ),
    "require_sync_start_invalid": dict(
        orch="require_sync_start_orch.cpp",
        code=7,
        runtime_env={},
        kernel="aiv/kernel_noop.cpp",
        marker="orch_error_code=7",
    ),
    "aicore_hang": dict(
        orch="aicore_hang_orch.cpp",
        code=100,
        runtime_env={},
        kernel="aic/kernel_hang.cpp",
        kernel_core="aic",
        onboard_only=True,  # a while(true) kernel would hang the simulator
        # Scheduler watchdog (2 s) classifies the running stall before STARS (3 s)
        # and host stream sync (4 s) reap the device. Mirrors aicore_op_timeout.
        env={
            "PTO2_SCHEDULER_TIMEOUT_MS": 2000,
            "PTO2_OP_EXECUTE_TIMEOUT_US": 3000000,
            "PTO2_STREAM_SYNC_TIMEOUT_MS": 4000,
        },
        marker="sub_class=S1",
    ),
    "tensor_wait_timeout": dict(
        orch="tensor_wait_timeout_orch.cpp",
        code=8,
        runtime_env={},
        kernel="aic/kernel_hang.cpp",
        kernel_core="aic",
        onboard_only=True,  # a while(true) kernel would hang the simulator
        # The data-wait timeout is 15 s on both arches now that it is frequency-
        # scaled (PTO2_TENSOR_DATA_TIMEOUT_MS, #1189) -- before that it was 15 s on
        # a5 but 300 s on a2a3, so this case used to be a5-only. Raise every other
        # watchdog above 15 s so the tensor-data wait wins the race and latches
        # code 8 before they reap the hung core.
        env={
            "PTO2_SCHEDULER_TIMEOUT_MS": 30000,
            "PTO2_OP_EXECUTE_TIMEOUT_US": 30000000,
            "PTO2_STREAM_SYNC_TIMEOUT_MS": 40000,
        },
        marker="orch_error_code=8",
    ),
    "async_completion_invalid": dict(
        orch="async_error_orch.cpp",
        code=101,
        runtime_env={},
        kernel="aiv/kernel_async_completion_invalid.cpp",
        marker="sched_error_code=101",
    ),
    "async_wait_overflow": dict(
        orch="async_error_orch.cpp",
        code=102,
        runtime_env={},
        kernel="aiv/kernel_async_wait_overflow.cpp",
        marker="sched_error_code=102",
    ),
    "explicit_fatal": dict(
        orch="explicit_fatal_orch.cpp",
        code=9,
        runtime_env={},
        kernel=None,
        marker="orch_error_code=9",
    ),
    # Error codes still without an e2e case, and why. Device-side cpput keeps the
    # structural ones covered; the sub-classes have unit coverage of the classifier.
    #
    # -- Structurally pre-empted (a different code always fires first) --
    # * ASYNC_REGISTRATION_FAILED (103): the AICore-side register_completion_
    #   condition caps at MAX_COMPLETIONS_PER_TASK (64) and latches ASYNC_WAIT_
    #   OVERFLOW (102) on the 65th condition — so it never emits enough CONDITION
    #   messages for the scheduler-side count check (> 64 -> 103) to fire. 103 is
    #   only reachable by corrupting the slab past the cap (UB; segfaults a5) or by
    #   an internally-malformed message kind — the scheduler-side guard that the
    #   AICore cap (102) always pre-empts.
    # * SCOPE_TASKS_OVERFLOW (10): scope_tasks_cap is the in-flight slot budget (sum
    #   of the per-ring windows, since #1188), but each ring physically holds only
    #   window-1 tasks (the ring's full/empty distinction), so all rings together
    #   hold at most sum(window-1) = cap - PTO2_MAX_RING_DEPTH, strictly below the
    #   cap. scope_tasks therefore tops out below its own cap: the rings fill first
    #   and latch SCOPE_DEADLOCK (1, single scope) or FLOW_CONTROL_DEADLOCK (3,
    #   same-ring nesting). Verified: no (depth, tasks/level, window) combo reaches
    #   code 10 -- they all latch 1 or 3. Shrinking the window shrinks the cap and
    #   the ring capacity together, so the constant gap holds.
    #
    # -- Design-unreachable from the public API (defensive classifier branches) --
    # * SCHEDULER_TIMEOUT sub-classes S3/S4/S5/UNKNOWN: the pure classifier is unit-
    #   tested for all of them (classify_stall_detail priority table in
    #   tests/ut/cpp/.../test_shared_memory.cpp), but the live states cannot be
    #   produced stably through the public API after Orch-side wiring. S3 used to
    #   be reachable by under-sizing the fanout dep_pool for a ready dummy consumer,
    #   but completed-producer fanins now bypass dep_pool entirely, while live
    #   producers block in Orch-side prewire and either recover or latch the
    #   orchestrator dep_pool error before the scheduler has an orch-done total to
    #   classify as ready-but-idle. S4 (dep-deadlock, pure WAIT) needs a dependency
    #   that never resolves with nothing running/ready; set_dependencies only
    #   references already-submitted tasks, so cycles are inexpressible, and a stuck
    #   producer is RUNNING (-> S1), never pure WAIT. S5 (orch-starvation) needs
    #   completed < total with empty rings, but total_tasks_ is the count of
    #   *submitted* tasks (sum of current_task_index), so once they retire completed
    #   == total and the watchdog never fires — a while-loop in the orch does not
    #   help. UNKNOWN is a bookkeeping-invariant violation (corruption). These are
    #   kept as defensive labels: if a future bug ever produces the state, the
    #   classifier names it correctly instead of mislabeling. S1 remains the
    #   reproducible scheduler-timeout e2e case (aicore_hang_orch.cpp).
    #
    # -- Reachable only by exhausting a fixed compile-time cap --
    # * TENSORMAP_OVERFLOW (11): the tensormap entry pool (PTO2_TENSORMAP_POOL_SIZE
    #   == 65536) is compile-time and not shrinkable via runtime_env, so wedging it
    #   needs a 65536-entry flood plus a stalled producer holding entries. Its latch
    #   is also orchestrator-level (a reclaim deadlock), so even a device UT would be
    #   a runtime-integration test, not a PTO2TensorMap unit test.
    #
    # TENSOR_WAIT_TIMEOUT (8) IS covered (onboard, both arches) — see the
    # tensor_wait_timeout case.
}


def _build_chip_callable(platform: str, case: dict) -> ChipCallable:
    kc = KernelCompiler(platform=platform)
    # AIV-kernel compilation needs the PTO-ISA headers (pto/pto-inst.hpp) on every
    # platform, sim included; onboard additionally needs them for orchestration.
    pto_isa_root = None
    if case["kernel"] is not None or not platform.endswith("sim"):
        pto_isa_root = ensure_pto_isa_root()
        os.environ["PTO_ISA_ROOT"] = pto_isa_root

    children = []
    if case["kernel"] is not None:
        core_type = case.get("kernel_core", "aiv")
        includes = list(kc.get_orchestration_include_dirs(RUNTIME)) + [str(kc.project_root / "src" / "common")]
        kernel = kc.compile_incore(
            source_path=os.path.join(KERNELS, case["kernel"]),
            core_type=core_type,
            pto_isa_root=pto_isa_root,
            extra_include_dirs=includes,
        )
        if not platform.endswith("sim"):
            kernel = extract_text_section(kernel)
        # The hang kernel binds no args (it just spins); the async kernels write output 0.
        if core_type == "aic":
            core_callable = CoreCallable.build(signature=[], binary=kernel)
        else:
            core_callable = CoreCallable.build(signature=[ArgDirection.OUT], binary=kernel)
        children.append((0, core_callable))

    orch_bytes = kc.compile_orchestration(runtime_name=RUNTIME, source_path=os.path.join(ORCH_DIR, case["orch"]))
    return ChipCallable.build(
        signature=[],
        func_name="aicpu_orchestration_entry",
        binary=orch_bytes,
        children=children,
    )


def _make_worker(platform: str, device_id: int, case_name: str, monkeypatch):
    case = CASES[case_name]
    # Per-case timeout-chain overrides (which watchdog must fire, and when).
    for key, value in case.get("env", {}).items():
        monkeypatch.setenv(key, str(value))
    chip_callable = _build_chip_callable(platform, case)
    worker = Worker(level=2, platform=platform, runtime=RUNTIME, device_id=device_id)
    handle = worker.register(chip_callable)
    worker.init()
    config = CallConfig()
    config.block_dim = 1
    config.aicpu_thread_num = 2
    for key, value in case["runtime_env"].items():
        setattr(config.runtime_env, key, value)
    return worker, handle, config


@pytest.mark.platforms(["a5sim", "a2a3sim"])
@pytest.mark.device_count(1)
@pytest.mark.runtime(RUNTIME)
@pytest.mark.parametrize("case_name", list(CASES))
def test_fatal_code_surfaces_on_sim(st_platform, st_device_ids, case_name, monkeypatch, capfd):
    """sim: the runtime status (``code -N``) reaches the host directly."""
    configure_logging("error")
    case = CASES[case_name]
    if case.get("onboard_only"):
        pytest.skip("hang kernel would spin the simulator forever (no STARS watchdog on sim)")
    worker, handle, config = _make_worker(st_platform, int(st_device_ids[0]), case_name, monkeypatch)
    try:
        with pytest.raises(RuntimeError, match=rf"(run_runtime|run) failed with code -{case['code']}\b"):
            worker.run(handle, ChipStorageTaskArgs(), config)
        captured = capfd.readouterr()
        assert case["marker"] in captured.err + captured.out, f"missing '{case['marker']}' in host log"
    finally:
        worker.close()


@pytest.mark.platforms(["a2a3"])
@pytest.mark.device_count(1)
@pytest.mark.runtime(RUNTIME)
@pytest.mark.timeout(90)  # hang cases hold a core ~15-20 s; bound a wedge
@pytest.mark.parametrize("case_name", list(CASES))
def test_device_error_class_reaches_host_log(st_platform, st_device_ids, case_name, monkeypatch, capfd):
    """onboard: the watchdog may mask the code as 507xxx, but the device class still reaches the host log."""
    configure_logging("error")
    case = CASES[case_name]
    worker, handle, config = _make_worker(st_platform, int(st_device_ids[0]), case_name, monkeypatch)
    try:
        # On hardware the op-execute / stream-sync watchdog can surface a generic
        # CANN 507xxx instead of the runtime's own -N; we only require that the
        # run fails. The point of the test is the device-classified host LOG.
        with pytest.raises(RuntimeError):
            worker.run(handle, ChipStorageTaskArgs(), config)
        captured = capfd.readouterr()
        assert case["marker"] in captured.err + captured.out, f"device error class '{case['marker']}' not in host log"
    finally:
        worker.close()
