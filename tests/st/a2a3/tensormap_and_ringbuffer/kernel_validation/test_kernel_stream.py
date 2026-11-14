# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""AICore validation fails the native Kernel caller stream without a Host observer."""

import ctypes
import os
import subprocess
from pathlib import Path

import pytest
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import ChipCallable, CoreCallable

from simpler_setup.elf_parser import extract_text_section
from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.pto_isa import ensure_pto_isa_root
from simpler_setup.runtime_builder import RuntimeBuilder

HERE = Path(__file__).parent
ROOT = Path(__file__).resolve().parents[5]
RUNTIME = "tensormap_and_ringbuffer"
pytestmark = [
    pytest.mark.runtime_product("kernel"),
    pytest.mark.requires_hardware,
    pytest.mark.platforms(["a2a3"]),
    pytest.mark.runtime(RUNTIME),
    pytest.mark.device_count(1),
    pytest.mark.timeout(180),
]


def build_probe(build, core_type, code):
    compiler = KernelCompiler("a2a3")
    build.mkdir(parents=True, exist_ok=True)
    orch = build / "orchestration.cpp"
    orch.write_text(
        (HERE / "kernels/orchestration.cpp.in")
        .read_text()
        .replace("CORE_TYPE", core_type)
        .replace("ERROR_CODE", str(code))
    )
    kernel = compiler.compile_incore(
        str(HERE / "kernels/kernel.cpp"),
        core_type=core_type,
        pto_isa_root=ensure_pto_isa_root(),
        extra_include_dirs=compiler.get_orchestration_include_dirs(RUNTIME),
    )
    child = CoreCallable.build(signature=[D.OUT], binary=extract_text_section(kernel))
    chip = ChipCallable.build(
        signature=[D.OUT],
        func_name="aicpu_orchestration_entry",
        binary=compiler.compile_orchestration(RUNTIME, str(orch)),
        children=[(219, child)],
    )
    (build / "callable.bin").write_bytes(ctypes.string_at(int(chip.buffer_ptr()), int(chip.buffer_size())))
    cann = Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest"))
    runtime = RuntimeBuilder("a2a3").get_binaries(RUNTIME, build=False)
    command = [
        "c++",
        "-std=c++17",
        "-O2",
        str(HERE / "kernel_stream_driver.cpp"),
        *[f"-I{ROOT / path}" for path in ("src/common/worker", "src/common/task_interface", "src/common")],
        *[f"-I{path}" for path in compiler.get_orchestration_include_dirs(RUNTIME)],
        f"-I{cann / 'include'}",
        f"-L{runtime.host_path.parent}",
        f"-Wl,-rpath,{runtime.host_path.parent}",
        "-lhost_runtime",
        f"-L{cann / 'lib64'}",
        f"-Wl,-rpath,{cann / 'lib64'}",
        "-lascendcl",
        "-lruntime",
        "-ldl",
        "-o",
        str(build / "driver"),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120, check=False)
    (build / "build.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    return [
        str(build / "driver"),
        str(build / "callable.bin"),
        str(runtime.aicpu_path),
        str(runtime.aicore_path),
        str(runtime.dispatcher_path),
    ]


@pytest.mark.parametrize(
    "capture",
    [
        pytest.param(0, marks=pytest.mark.execution_mode("eager")),
        pytest.param(1, marks=pytest.mark.execution_mode("capture_replay")),
    ],
)
@pytest.mark.parametrize("core_type", ["aic", "aiv"])
@pytest.mark.parametrize("code", [0, 106])
def test_kernel_validation_fails_caller_stream(st_device_ids, tmp_path, capture, core_type, code):
    binaries = build_probe(tmp_path, core_type, code)
    run_probe(binaries, st_device_ids[0], tmp_path, capture, code)


def run_probe(binaries, device, tmp_path, capture, code):
    logs = tmp_path / "ascend"
    logs.mkdir()
    result = subprocess.run(
        [binaries[0], str(device), str(capture), str(code), *binaries[1:]],
        env={**os.environ, "ASCEND_PROCESS_LOG_PATH": str(logs)},
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    (tmp_path / "run.log").write_text(result.stdout + result.stderr)
    print(result.stdout, flush=True)
    assert result.returncode == 0, result.stdout + result.stderr
    if code:
        assert "KERNEL_STREAM_PASS owners=retained observer=caller-stream" in result.stdout
    else:
        assert "KERNEL_STREAM_PASS owners=released output=1" in result.stdout
