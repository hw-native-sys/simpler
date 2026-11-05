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
from pathlib import Path

import pytest

from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.pto_isa import ensure_pto_isa_root

_ROOT = Path(__file__).resolve().parents[3]
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
    for arch in ("a2a3", "a5"):
        compiler = KernelCompiler(arch + "sim")
        (path / f"{arch}.so").write_bytes(compiler.compile_incore(str(source), core_type="aiv", pto_isa_root=isa_root))
        orch = path / "orch.cpp"
        orch.write_text('extern "C" void admission_orchestration() {}\n')
        (path / f"orch-{arch}.so").write_bytes(compiler.compile_orchestration("host_build_graph", str(orch)))
    # A legacy artifact is valid ELF with a kernel entry but has no family declaration.
    subprocess.run(["c++", "-shared", "-fPIC", str(source), "-o", str(path / "legacy.so")], check=True)
    unknown = path / "unknown.cpp"
    unknown.write_text('extern "C" const char *simpler_cpu_sim_target() { return "unknown"; }\n')
    subprocess.run(["c++", "-shared", "-fPIC", str(source), str(unknown), "-o", str(path / "unknown.so")], check=True)
    return path, marker


@pytest.mark.parametrize("arch", ["a2a3", "a5"])
def test_prebuilt_target_admission(arch, admission_artifacts):
    path, marker = admission_artifacts
    runtime = _ROOT / "build/lib" / arch / "sim/host_build_graph/libhost_runtime.so"
    if not runtime.exists():
        pytest.skip(f"{runtime} not built")
    opposite = "a5" if arch == "a2a3" else "a2a3"
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
            *(str(path / f"{kind}.so") for kind in (opposite, "legacy", "unknown", arch, arch)),
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
        [0, "called"],
        [0, "called"],
    ]
