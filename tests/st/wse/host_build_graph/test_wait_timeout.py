# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A host wait deadline does not cancel a kernel held at an acknowledged gate."""

import ctypes
import hashlib
import json
import subprocess
import threading
from pathlib import Path

import pytest
import torch
from simpler.task_interface import (
    ArgDirection,
    CallConfig,
    ChipCallable,
    CoreCallable,
    DataType,
    TaskArgs,
    TensorArgType,
)
from simpler.worker import Worker

from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.runtime_builder import RuntimeBuilder
from tests.st.wse.host_build_graph.fixed_sram_layout import copy_from, layout_view, release_view

pytestmark = [pytest.mark.platforms(["wsesim"]), pytest.mark.runtime("host_build_graph"), pytest.mark.device_count(1)]

_GATE = r"""
#include <chrono>
#include <condition_variable>
#include <mutex>
namespace {
std::mutex mutex;
std::condition_variable changed;
bool entered = false;
bool released = false;
}
extern "C" void wse_test_gate() {
    std::unique_lock<std::mutex> lock(mutex);
    entered = true;
    changed.notify_all();
    changed.wait(lock, [] { return released; });
}
extern "C" bool await_entry() {
    std::unique_lock<std::mutex> lock(mutex);
    return changed.wait_for(lock, std::chrono::seconds(5), [] { return entered; });
}
extern "C" void release_gate() {
    std::lock_guard<std::mutex> lock(mutex);
    released = true;
    changed.notify_all();
}
"""
_KERNEL = r"""
#include <cstdint>
#include <cstdlib>
#include <dlfcn.h>
#include <pto/pto-inst.hpp>
#include "tensor.h"
extern "C" void kernel_entry(int64_t *args) {
    auto gate = reinterpret_cast<void(*)()>(dlsym(RTLD_DEFAULT, "wse_test_gate"));
    if (!gate) std::abort();
    gate();
    auto *tensor = reinterpret_cast<Tensor *>(args[0]);
    auto *out = reinterpret_cast<float *>(tensor->buffer.addr) + tensor->start_offset;
    out[0] = 42.0f;
}
"""
_ORCH = r"""
#include "orchestration_api.h"
extern "C" OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &) {
    return OrchestrationConfig{.expected_arg_count = 1};
}
extern "C" void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    CoreTaskArgs core;
    core.add_output(args.tensor(0).ref());
    rt_submit_aiv_task(0, core);
}
"""


@pytest.mark.parametrize("finish", ["wait", "close"])
def test_wait_timeout_retains_buffer_until_real_completion(tmp_path, st_platform, finish):
    gate_source = tmp_path / "gate.cpp"
    gate_source.write_text(
        _GATE.replace("wse_test_gate", "wse_test_gate_" + finish)
        .replace("await_entry", "await_entry_" + finish)
        .replace("release_gate", "release_gate_" + finish)
    )
    library = tmp_path / "gate.so"
    subprocess.run(
        ["c++", "-std=c++17", "-shared", "-fPIC", "-pthread", str(gate_source), "-o", str(library)],
        check=True,
        timeout=60,
    )
    gate = ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
    gate.await_entry = getattr(gate, "await_entry_" + finish)
    gate.release_gate = getattr(gate, "release_gate_" + finish)
    gate.await_entry.restype = ctypes.c_bool
    compiler = KernelCompiler(st_platform)
    kernel_source = tmp_path / "kernel.cpp"
    kernel_source.write_text(_KERNEL.replace("wse_test_gate", "wse_test_gate_" + finish))
    orchestration = tmp_path / "orch.cpp"
    orchestration.write_text(_ORCH)
    core = CoreCallable.build(
        signature=[ArgDirection.OUT],
        binary=compiler.compile_incore(
            str(kernel_source),
            core_type="",
            extra_include_dirs=compiler.get_orchestration_include_dirs("host_build_graph"),
        ),
    )
    chip = ChipCallable.build(
        signature=[ArgDirection.OUT],
        func_name="aicpu_orchestration_entry",
        binary=compiler.compile_orchestration("host_build_graph", str(orchestration)),
        children=[(0, core)],
    )
    maps_path = Path("/proc/self/maps")
    before_maps = {line.split()[-1] for line in maps_path.read_text().splitlines()} if maps_path.exists() else set()
    worker = Worker(level=2, platform=st_platform, runtime="host_build_graph", device_id=0)
    run = None
    try:
        worker.init()
        handle = worker.register(chip)
        buffer = layout_view(worker, 4, core=39)
        args = TaskArgs()
        args.add_tensor(buffer.tensor(shapes=(1,), dtype=DataType.FLOAT32), TensorArgType.OUTPUT_EXISTING)
        publications = worker.scheduler_mode_publish_count
        bad_config = CallConfig()
        bad_config.aicpu_thread_num = 1
        with pytest.raises(RuntimeError, match="failed with code"):
            worker.run(handle, args, bad_config)
        assert worker.scheduler_mode_publish_count == publications
        for option in ("enable_chip_swimlane", "enable_dump_args", "enable_pmu", "enable_scope_stats"):
            unsupported = CallConfig()
            unsupported.output_prefix = str(tmp_path / "unsupported")
            setattr(unsupported, option, True)
            with pytest.raises(RuntimeError):
                worker.run(handle, args, unsupported)
            assert worker.scheduler_mode_publish_count == publications
        run = worker.submit(handle, args)
        assert gate.await_entry(), "kernel did not acknowledge entering its gate"
        if maps_path.exists():
            after_maps = {line.split()[-1] for line in maps_path.read_text().splitlines()}
            added = after_maps - before_maps
            pe_modules = sorted(path for path in added if "/tmp/wse_pe_" in path)
            assert len(pe_modules) == 1, pe_modules
            assert not any("/tmp/aicpu_sim_" in path for path in added)
            binaries = RuntimeBuilder(st_platform).get_binaries("host_build_graph")
            assert str(binaries.host_path.resolve()) in after_maps
            assert binaries.aicpu_path is None
            loaded = [str(binaries.host_path.resolve()), *pe_modules]
            expected = {
                hashlib.sha256(path.read_bytes()).hexdigest() for path in (binaries.host_path, binaries.aicore_path)
            }
            assert {hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in loaded} == expected
            proof = {"loaded": {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in loaded}}
            (tmp_path / "loaded-modules.json").write_text(json.dumps(proof, indent=2))
        with pytest.raises(TimeoutError):
            run.wait(timeout=0.01)
        with pytest.raises(RuntimeError, match="in.flight"):
            worker.free(buffer.root)
        if finish == "close":
            closed = threading.Event()
            errors = []

            def observe_close():
                try:
                    # Native teardown is bound to the init-owner thread. Observe
                    # its published close attempt before releasing the Kernel.
                    with worker._hierarchical_start_cv:
                        assert worker._hierarchical_start_cv.wait_for(
                            lambda: worker._close_completion is not None, timeout=5
                        )
                    assert not closed.wait(0.05), "close returned before the held Kernel completed"
                except Exception as error:  # noqa: BLE001 -- forward the observer's failure
                    errors.append(error)
                finally:
                    gate.release_gate()

            observer = threading.Thread(target=observe_close)
            observer.start()
            try:
                worker.close()
            finally:
                closed.set()
                observer.join(15)
            assert not observer.is_alive() and not errors, errors
            run = None
            return
        gate.release_gate()
        run.wait(timeout=10)
        value = torch.zeros(1)
        copy_from(worker, value, buffer)
        assert value.item() == 42
        worker.run(handle, args)
        copy_from(worker, value, buffer)
        assert value.item() == 42
        release_view(worker, buffer)
        worker.unregister(handle)
    finally:
        gate.release_gate()
        if run is not None:
            run.wait(timeout=10)
        worker.close()
        if Path("/proc/self/task").exists():
            remaining = [path.read_text().strip() for path in Path("/proc/self/task").glob("*/comm")]
            assert not any(name.startswith("wse-") for name in remaining), remaining
