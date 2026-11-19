# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host startup rollback and draining WSE workers after a failed Linx kernel."""

import subprocess
from pathlib import Path

import pytest
from simpler.task_interface import TaskArgs
from simpler.worker import Worker

from tests.st.wse.host_build_graph.scene_helpers import build_callable
from tests.ut.py.test_sim_startup_failure import _run_scene

pytestmark = [pytest.mark.platforms(["wsesim"]), pytest.mark.runtime("host_build_graph"), pytest.mark.device_count(1)]


@pytest.mark.parametrize("mode,thread", [("create", "wse-0-0"), ("create", "wse-1-2"), ("bind", "wse-0-1")])
def test_partial_startup_drains_created_threads(tmp_path, mode, thread):
    root = Path(__file__).resolve().parents[4]
    template = (root / "tests/ut/py/support/sim_startup_fault.cpp").read_text()
    source = tmp_path / "fault.cpp"
    source.write_text(
        template.replace("@THREAD@", thread).replace(
            "constexpr bool fail_creation = false;", f"constexpr bool fail_creation = {str(mode == 'create').lower()};"
        )
    )
    library = tmp_path / "fault.so"
    subprocess.run(
        ["c++", "-std=c++17", "-shared", "-fPIC", str(source), "-pthread", "-ldl", "-o", str(library)],
        check=True,
        capture_output=True,
        timeout=60,
    )
    code, output = _run_scene("wse", "host_build_graph", library)
    (tmp_path / "startup.log").write_text(output)
    marker = "SIM_STARTUP_CREATE_EAGAIN" if mode == "create" else "SIM_STARTUP_BIND_ENOMEM"
    assert output.count(marker) == 1, output
    assert code == 1 and "TIMEOUT:" not in output, output
    assert "WSE startup rollback: joined" in output and "device events=0" in output, output
    assert "launch_native_run failed with code -1" in output, output


@pytest.mark.timeout(60, method="thread")
def test_linx_kernel_failure_drains_other_cores(tmp_path, st_platform, capfd):
    (tmp_path / "orch.cpp").write_text("""
#include "orchestration_api.h"
extern "C" OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &) { return {.expected_arg_count = 0}; }
extern "C" void aicpu_orchestration_entry(const ChipTaskArgs &) {
    CoreTaskArgs task;
    task.launch_spec.set_block_num(2);
    WseLaunchSpec launch;
    launch.thread_num = 2;
    rt_submit_wse_task(0, task, launch);
}
""")
    (tmp_path / "failure.cpp").write_text("""
#include <pto/pto-inst.hpp>
#include "intrinsic.h"
#include <latch>
#include <cstdio>

static pto::linx_cpu::CoreContext *other_core;
static std::latch core_ready(1);
extern "C" void kernel_entry(int64_t *args) {
    if (get_core_id(args) == 1) {
        if (::get_thread_idx() == 1) {
            other_core = pto::linx_cpu::currentCore;
            core_ready.count_down();
            pto::SharedTile<pto::SharedMatrixRight<float, 32, 32>> tile;
            try {
                pto::linx_cpu::acquireSlot<0>(tile);
            } catch (const std::runtime_error &e) {
                std::fprintf(stderr, "slot waiter saw %s\\n", e.what());
                throw;
            }
            return;
        }
    } else if (::get_thread_idx() == 0) {
        core_ready.wait();
        std::unique_lock lock(other_core->mutex);
        other_core->changed.wait(lock, [] { return other_core->slotWaiters & (1u << 1); });
        throw std::runtime_error("injected Linx failure");
    }
    pto::linx_cpu::syncCore();
}
""")
    worker = Worker(level=2, platform=st_platform, runtime="host_build_graph", device_id=0)
    worker.init()
    try:
        handle = worker.register(build_callable(st_platform, tmp_path, "orch.cpp", [], [("failure.cpp", [])]))
        with pytest.raises(RuntimeError, match="finalize_native_run failed"):
            worker.run(handle, TaskArgs())
    finally:
        worker.close()
    assert "slot waiter saw injected Linx failure" in capfd.readouterr().err
