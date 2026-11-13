# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Prebuilt CPU kernels are admitted only by a matching simulator family."""

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.pto_isa import ensure_pto_isa_root
from simpler_setup.runtime_builder import RuntimeBuilder

_ROOT = Path(__file__).resolve().parents[4]
pytestmark = [pytest.mark.platforms(["wsesim"]), pytest.mark.runtime("host_build_graph")]
_RUN = r"""
import ctypes
import json
import sys
from pathlib import Path
from simpler.task_interface import ChipCallable, CoreCallable

root, arch, orch, marker, *kernels = map(Path, sys.argv[1:])
ctypes.CDLL(str(root / "build/lib/libcpu_sim_context.so"), mode=ctypes.RTLD_GLOBAL)
lib = ctypes.CDLL(str(root / "build/lib" / arch / "sim/host_build_graph/libhost_runtime.so"))
lib.create_device_context.restype = ctypes.c_void_p
lib.simpler_register_callable.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p]
lib.simpler_register_callable.restype = ctypes.c_int
lib.finalize_device.argtypes = [ctypes.c_void_p]
lib.finalize_device.restype = ctypes.c_int
lib.destroy_device_context.argtypes = [ctypes.c_void_p]
ctx = lib.create_device_context()
assert ctx
results = []
try:
    for index, kernel in enumerate(kernels):
        child = CoreCallable.build(signature=[], binary=kernel.read_bytes())
        chip = ChipCallable.build(signature=[], func_name="admission_orchestration",
                                  binary=orch.read_bytes(), children=[(0, child)])
        status = lib.simpler_register_callable(ctx, index, chip.buffer_ptr())
        results.append([status, marker.read_text() if marker.exists() else ""])
finally:
    assert lib.finalize_device(ctx) == 0
    lib.destroy_device_context(ctx)
print(json.dumps(results))
"""


@pytest.fixture(scope="module")
def admission_artifacts(tmp_path_factory):
    path = tmp_path_factory.mktemp("cpu-sim-admission")
    marker = path / "executed"
    source = path / "kernel.cpp"
    # Rejected artifacts must never register hooks or execute their kernel entry.
    source.write_text(
        "#include <cstdio>\n"
        f'static void record() {{ if (auto *f = std::fopen({json.dumps(str(marker))}, "a")) '
        '{ std::fputs("called", f); std::fclose(f); } }\n'
        'extern "C" void kernel_entry() { record(); }\n'
        'extern "C" void pto_sim_register_hooks(void*, void*) { record(); }\n'
    )
    isa_root = ensure_pto_isa_root()
    for arch in ("wse", "a5"):
        compiler = KernelCompiler(arch + "sim")
        (path / f"{arch}.so").write_bytes(
            compiler.compile_incore(str(source), core_type="" if arch == "wse" else "aiv", pto_isa_root=isa_root)
        )
        orch = path / "orch.cpp"
        orch.write_text('extern "C" void admission_orchestration() {}\n')
        (path / f"orch-{arch}.so").write_bytes(compiler.compile_orchestration("host_build_graph", str(orch)))
    # A legacy artifact is valid ELF with a kernel entry but has no family declaration.
    subprocess.run(["c++", "-shared", "-fPIC", str(source), "-o", str(path / "legacy.so")], check=True)
    unknown = path / "unknown.cpp"
    unknown.write_text('extern "C" const char *simpler_cpu_sim_target() { return "unknown"; }\n')
    subprocess.run(["c++", "-shared", "-fPIC", str(source), str(unknown), "-o", str(path / "unknown.so")], check=True)
    for arch in ("wse", "a5"):
        missing = path / f"missing-{arch}.cpp"
        missing.write_text(f'extern "C" const char *simpler_cpu_sim_target() {{ return "{arch}sim"; }}\n')
        subprocess.run(["c++", "-shared", "-fPIC", str(missing), "-o", str(path / f"missing-{arch}.so")], check=True)
    return path, marker


@pytest.mark.parametrize(
    "arch",
    [
        pytest.param("wse", marks=pytest.mark.platforms(["wsesim"])),
        pytest.param("a5", marks=pytest.mark.platforms(["a5sim"])),
    ],
)
def test_prebuilt_target_admission(arch, admission_artifacts):
    path, marker = admission_artifacts
    runtime = _ROOT / "build/lib" / arch / "sim/host_build_graph/libhost_runtime.so"
    assert runtime.exists(), f"required runtime not built: {runtime}"
    opposite = "a5" if arch == "wse" else "wse"
    marker.unlink(missing_ok=True)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _RUN,
            str(_ROOT),
            arch,
            str(path / f"orch-{arch}.so"),
            str(marker),
            *(str(path / f"{kind}.so") for kind in (opposite, "legacy", "unknown", f"missing-{arch}", arch, arch)),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    # Rejections precede hooks; a subsequent good image registers once and deduplicates.
    assert json.loads(result.stdout.splitlines()[-1]) == [
        [-1000, ""],
        [-1000, ""],
        [-1000, ""],
        [-1000, ""],
        [0, "called"],
        [0, "called"],
    ]


@pytest.mark.device_count(1)
def test_executor_rejects_missing_memory_sections_before_create(tmp_path, st_platform, monkeypatch):
    from simpler.task_interface import ChipCallable, TaskArgs  # noqa: PLC0415
    from simpler.worker import Worker  # noqa: PLC0415

    marker = tmp_path / "create-called"
    source = tmp_path / "incomplete_executor.cpp"
    source.write_text(
        "#include <cstddef>\n#include <cstdint>\n#include <cstdio>\n"
        'extern "C" {\n'
        "void *wse_create_run(const void *, uint32_t, uint64_t, const void *, uint32_t, "
        "const void *, uint32_t, char *, size_t) {\n"
        f'  if (auto *f = std::fopen({json.dumps(str(marker))}, "w")) std::fclose(f);\n'
        "  return nullptr;\n}\n"
        "void wse_destroy_run(void *) {}\n"
        "int wse_execute_pe(void *, uint32_t) { return 0; }\n"
        "const void *wse_run_events(const void *, uint32_t, uint64_t *count) { *count=0; return nullptr; }\n"
        "uint64_t wse_completed_tasks(const void *) { return 0; }\n"
        "uint64_t wse_runtime_used(const void *, uint32_t) { return 0; }\n"
        "uint64_t wse_control_address(const void *, uint32_t) { return 0; }\n}\n"
    )
    compiler = KernelCompiler(st_platform)
    library = tmp_path / "incomplete_executor.so"
    subprocess.run(
        [compiler.gxx15.cxx_path, "-shared", "-fPIC", str(source), "-o", str(library)],
        check=True,
        timeout=60,
    )
    binaries = RuntimeBuilder(st_platform).get_binaries("host_build_graph")
    incomplete = replace(binaries, aicore_path=library)
    monkeypatch.setattr(RuntimeBuilder, "get_binaries", lambda *_args, **_kwargs: incomplete)
    orchestration = Path(__file__).parent / "direct_launch/kernels/empty_orch.cpp"
    callable_ = ChipCallable.build(
        signature=[],
        func_name="aicpu_orchestration_entry",
        binary=compiler.compile_orchestration("host_build_graph", str(orchestration)),
        children=[],
    )
    worker = Worker(level=2, platform=st_platform, runtime="host_build_graph", device_id=0)
    try:
        worker.init()
        handle = worker.register(callable_)
        with pytest.raises(RuntimeError):
            worker.run(handle, TaskArgs())
        assert not marker.exists(), "an incomplete PE executor reached run creation"
    finally:
        worker.close()
