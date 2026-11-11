# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Onboard capture of both KernelExecutionState-owned streams (Probe B).

The native harness uses the production context ops and AICPU loader. Ordinary
AICore kernels surround the event chain on the borrowed caller stream. Both
internal kernels must run again for every replay, with fresh inputs and
poisoned outputs. No torch_npu dependency or complete kernel-mode binder is
needed for this CANN feasibility test.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path

import pytest

from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.runtime_builder import RuntimeBuilder

_ROOT = Path(__file__).resolve().parents[4]
_SOURCE = Path(__file__).with_name("native")
pytestmark = [pytest.mark.requires_hardware, pytest.mark.platforms(["a2a3", "a5"])]


def _run(command, **kwargs):
    result = subprocess.run(command, capture_output=True, text=True, timeout=180, check=False, **kwargs)
    assert result.returncode == 0, f"{command}\n{result.stdout}\n{result.stderr}"
    return result


def build_capture_probe(build: Path, device_platform: str):
    cann = Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest"))
    compiler = KernelCompiler(device_platform)
    assert compiler.ccec is not None and compiler.aarch64 is not None
    host_compiler = shutil.which("c++")
    assert host_compiler, "onboard capture probe requires a host C++ compiler"
    runtime = RuntimeBuilder(device_platform).get_binaries("tensormap_and_ringbuffer", build=False)
    dispatcher = runtime.dispatcher_path
    assert dispatcher is not None
    build.mkdir(parents=True, exist_ok=True)
    cpu = build / "cpu.so"
    core = build / "core.o"
    driver = build / "driver"
    _run(
        [
            compiler.aarch64.cxx_path,
            "-std=c++17",
            "-O2",
            "-shared",
            "-fPIC",
            "-Wl,--build-id",
            f"-I{_ROOT / 'src/common'}",
            str(_SOURCE / "cpu.cpp"),
            "-o",
            str(cpu),
        ]
    )
    # RTS registers this probe's named raw entry. Use the platform toolchain's
    # compile flags and keep the ELF intact for rtRegisterAllKernel.
    _run(
        [
            compiler.ccec.cxx_path,
            *compiler.ccec.get_compile_flags(core_type="aic"),
            "-O2",
            f"-I{cann / 'asc'}",
            *[f"-I{path}" for path in compiler.get_ascend_incore_include_dirs()],
            str(_SOURCE / "core.cpp"),
            "-o",
            str(core),
        ]
    )
    host_includes = [
        _ROOT / suffix
        for suffix in (
            "src/common",
            "src/common/platform/include",
            f"src/{device_platform}/platform/include",
            "src/common/task_interface",
            "src/common/worker",
            "src/common/log/include",
            "src/common/aicpu_loader/host",
            "src/common/platform/onboard/host",
        )
    ]
    pkg = cann / f"{platform.machine()}-linux/pkg_inc"
    host_includes += [
        cann / "include",
        pkg,
        pkg / "runtime",
        pkg / "runtime/runtime",
        pkg / "profiling",
        pkg / "driver",
    ]
    sources = [
        _SOURCE / "driver.cpp",
        _ROOT / "src/common/platform/shared/host/kernel_execution_state.cpp",
        _ROOT / "src/common/platform/onboard/host/kernel_platform_ops.cpp",
        _ROOT / "src/common/aicpu_loader/host/load_aicpu_op.cpp",
        _ROOT / "src/common/log/unified_log_host.cpp",
        _ROOT / "src/common/log/host_log.cpp",
    ]
    _run(
        [
            host_compiler,
            "-std=c++17",
            "-O2",
            "-pthread",
            *map(str, sources),
            *[f"-I{path}" for path in host_includes],
            f"-L{cann / 'lib64'}",
            f"-Wl,-rpath,{cann / 'lib64'}",
            "-lascendcl",
            "-lruntime",
            "-ldl",
            "-Wl,--wrap=rtStreamAddToModel",
            "-Wl,--wrap=rtStreamGetCaptureInfo",
            "-Wl,--wrap=aclmdlRICaptureGetInfo",
            "-o",
            str(driver),
        ]
    )
    return driver, dispatcher, cpu, core, build


@pytest.fixture(scope="module")
def capture_probe(tmp_path_factory, request):
    return build_capture_probe(tmp_path_factory.mktemp("kernel-capture"), request.config.getoption("--platform"))


def test_kernel_context_captures_both_internal_streams(capture_probe, request):
    driver, dispatcher, cpu, core, build = capture_probe
    device = str(request.config.getoption("--device")).split("-")[0].split(",")[0]
    env = dict(os.environ)
    logs = build / "ascend"
    logs.mkdir()
    env["ASCEND_PROCESS_LOG_PATH"] = str(logs)
    result = subprocess.run(
        [str(driver), device, str(dispatcher), str(cpu), str(core)],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    (build / "run.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}\nDevice logs: {logs}"
    assert "capture_probe PASS replays=100 internal_streams=2 forbidden_calls=0" in result.stdout
