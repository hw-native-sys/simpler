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

_ROOT = Path(__file__).resolve().parents[4]
_SOURCE = Path(__file__).with_name("native")
pytestmark = [pytest.mark.requires_hardware, pytest.mark.platforms(["a2a3"])]


def _run(command, **kwargs):
    result = subprocess.run(command, capture_output=True, text=True, timeout=180, check=False, **kwargs)
    assert result.returncode == 0, f"{command}\n{result.stdout}\n{result.stderr}"
    return result


@pytest.fixture(scope="module")
def capture_probe(tmp_path_factory):
    cann = Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest"))
    ccec = cann / "bin/ccec"
    cpu_compiler = cann / "tools/hcc/bin/aarch64-target-linux-gnu-g++"
    host_compiler = shutil.which("c++")
    for tool in (ccec, cpu_compiler):
        assert tool.is_file(), f"onboard capture probe requires {tool}"
    assert host_compiler, "onboard capture probe requires a host C++ compiler"
    dispatcher = _ROOT / "build/lib/a2a3/dispatcher/libsimpler_aicpu_dispatcher.so"
    assert dispatcher.is_file(), f"build the runtime first: missing {dispatcher}"
    build = tmp_path_factory.mktemp("kernel-capture")
    cpu = build / "cpu.so"
    core = build / "core.o"
    driver = build / "driver"
    _run(
        [
            str(cpu_compiler),
            "-std=c++17",
            "-O2",
            "-shared",
            "-fPIC",
            "-Wl,--build-id",
            str(_SOURCE / "cpu.cpp"),
            "-o",
            str(cpu),
        ]
    )
    core_includes = [
        cann / suffix
        for suffix in (
            "asc",
            "asc/include",
            "asc/include/basic_api",
            "asc/impl/basic_api",
            "include",
            "include/external",
            "include/c_api",
            "include/c_api/internal",
            "include/ascendc/basic_api",
            "include/ascendc/impl/basic_api",
        )
    ]
    _run(
        [
            str(ccec),
            "-c",
            "-O2",
            "-x",
            "cce",
            "-std=c++17",
            "--cce-aicore-only",
            "--cce-aicore-arch=dav-c220-cube",
            "-mllvm",
            "-cce-aicore-stack-size=0x8000",
            "-mllvm",
            "-cce-aicore-function-stack-size=0x8000",
            "-mllvm",
            "-cce-aicore-record-overflow=false",
            "-mllvm",
            "-cce-aicore-addr-transform",
            "-mllvm",
            "-cce-aicore-dcci-insert-for-scalar=false",
            *[f"-I{path}" for path in core_includes if path.is_dir()],
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
            "src/a2a3/platform/include",
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
