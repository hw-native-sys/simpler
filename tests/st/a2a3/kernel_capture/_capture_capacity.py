# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Build and validate the bounded native workspace probes."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

from simpler.task_interface import ArgDirection, ChipCallable, CoreCallable

from simpler_setup.elf_parser import extract_text_section
from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.pto_isa import ensure_pto_isa_root
from simpler_setup.runtime_builder import RuntimeBuilder
from tests.st.a2a3.kernel_capture.test_hbg_capture import _run_capture_probe


def build_probe(build: Path, platform: str, runtime: str, leaf: str) -> tuple[list[str], dict]:
    root = Path(__file__).resolve().parents[4]
    source = Path(__file__).with_name("native")
    build.mkdir(parents=True, exist_ok=True)
    compiler = KernelCompiler(platform)
    core = compiler.compile_incore(
        str(source / "hbg_step.cpp"),
        core_type="aiv",
        pto_isa_root=ensure_pto_isa_root(),
        extra_include_dirs=compiler.get_orchestration_include_dirs(runtime),
        build_dir=str(build),
    )
    orch = compiler.compile_orchestration(
        runtime, str(source / "capture_capacity_orchestration.cpp"), build_dir=str(build)
    )
    child = CoreCallable.build(
        signature=[ArgDirection.IN, ArgDirection.OUT, ArgDirection.SCALAR], binary=extract_text_section(core)
    )
    callable_ = ChipCallable.build(
        signature=[ArgDirection.IN, ArgDirection.OUT, ArgDirection.SCALAR],
        func_name="capture_capacity_orchestration",
        binary=orch,
        children=[(0, child)],
    )
    path = build / "callable.bin"
    path.write_bytes(ctypes.string_at(int(callable_.buffer_ptr()), int(callable_.buffer_size())))
    binaries = RuntimeBuilder(platform).get_binaries(runtime, build=False)
    assert binaries.dispatcher_path is not None
    cann = Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest"))
    command = [
        "c++",
        "-std=c++17",
        "-O2",
        str(source / f"{leaf}.cpp"),
        str(source / "capture_capacity_observer.cpp"),
        *[f"-I{root / item}" for item in ("src/common", "src/common/task_interface", "src/common/worker")],
        *[f"-I{item}" for item in compiler.get_orchestration_include_dirs(runtime)],
        *[
            f"-I{cann / item}"
            for item in ("include", "pkg_inc", "pkg_inc/runtime", "pkg_inc/profiling", "pkg_inc/runtime/runtime")
        ],
        f"-L{binaries.host_path.parent}",
        f"-Wl,-rpath,{binaries.host_path.parent}",
        "-lhost_runtime",
        f"-L{cann / 'lib64'}",
        f"-Wl,-rpath,{cann / 'lib64'}",
        "-lascendcl",
        "-lruntime",
        "-ldl",
        "-o",
        str(build / leaf),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=180, check=False)
    (build / f"{leaf}-command.json").write_text(json.dumps(command, indent=2))
    (build / f"{leaf}-build.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    inputs = [
        path,
        binaries.host_path,
        binaries.aicpu_path,
        binaries.aicore_path,
        binaries.dispatcher_path,
        build / leaf,
    ]
    identity = {str(item): hashlib.sha256(item.read_bytes()).hexdigest() for item in inputs}
    (build / f"{leaf}-identity.json").write_text(json.dumps(identity, indent=2))
    command = [
        str(build / leaf),
        str(path),
        str(binaries.aicpu_path),
        str(binaries.aicore_path),
        str(binaries.dispatcher_path),
        runtime,
    ]
    return command, identity


def probe_output(prefix: str) -> Path:
    output = Path(__file__).resolve().parents[4] / "outputs" / "default-ci" / "capture-capacity"
    output.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=output))


def write_input_identities(output: Path, identities: dict) -> str:
    root = Path(__file__).resolve().parents[4]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    (output / "input-identities.json").write_text(json.dumps({"source_head": head, "inputs": identities}, indent=2))
    return head


def run_probe(command: list[str], device: str, output: Path) -> str:
    output.mkdir(parents=True, exist_ok=True)
    logs = output / "ascend"
    logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, ASCEND_PROCESS_LOG_PATH=str(logs))
    result = _run_capture_probe([command[0], device, *command[1:]], output, env)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout
