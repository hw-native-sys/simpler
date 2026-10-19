# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exercise onboard execution state against the built host runtime, with RTS stubs.

The hardware marker selects CI jobs that build the onboard libraries and have
CANN headers. The probe does not initialize or execute on a device.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_CASES = [
    pytest.param(arch, runtime, id=f"{arch}-{runtime}", marks=pytest.mark.platforms([arch]))
    for arch in ("a2a3", "a5")
    for runtime in ("tensormap_and_ringbuffer", "host_build_graph")
]


@pytest.mark.parametrize("probe", ["runtime_publication_launch", "run_execution_streams", "run_caller_launch"])
@pytest.mark.requires_hardware
@pytest.mark.parametrize(("arch", "runtime"), _CASES)
def test_onboard_execution_contract(arch: str, runtime: str, probe: str, tmp_path: Path):
    cache = _ROOT / "build/cache" / arch / "onboard" / runtime / "host"
    library = _ROOT / "build/lib" / arch / "onboard" / runtime / "libhost_runtime.so"
    database = cache / "compile_commands.json"
    assert library.is_file(), f"Build the {arch} onboard runtimes before this test: {library}"
    assert database.is_file(), f"RuntimeBuilder's host compile database is required: {database}"
    entries = json.loads(database.read_text())
    entry = next(
        item for item in entries if Path(item["file"]) == _ROOT / f"src/{arch}/platform/onboard/host/device_runner.cpp"
    )
    args = entry.get("arguments") or shlex.split(entry["command"])
    # The same compiler, generated profiling headers and definitions must see
    # the same C++ object layouts as the library under test.
    command = []
    arguments = iter(args)
    for arg in arguments:
        if arg == "-o":
            next(arguments)
        elif arg != "-c" and arg != entry["file"]:
            command.append(arg)
    cann = Path(os.environ["ASCEND_HOME_PATH"])
    binary = tmp_path / probe
    if probe == "run_caller_launch" and arch == "a5":
        # Compile the real backend so this executable's strong scheduler-export
        # observer replaces its private weak leaf. The drain decision stays real.
        command.append(entry["file"])
    command += [
        "-UNDEBUG",
        str(_ROOT / f"tests/ut/py/support/{probe}_probe.cpp"),
        # The probe reaches its test fixture by path from the unit-test root, the
        # same way the C++ targets do, so a fixture that moves stays reachable
        # from both without this line changing.
        f"-I{_ROOT / 'tests/ut/cpp'}",
        str(library),
        f"-Wl,-rpath,{library.parent}",
        f"-L{cann / 'lib64'}",
        f"-L{cann / 'runtime/lib64'}",
        f"-Wl,-rpath,{cann / 'lib64'}",
        f"-Wl,-rpath,{cann / 'runtime/lib64'}",
        "-lruntime",
        "-lascendcl",
        "-pthread",
        "-ldl",
        "-o",
        str(binary),
    ]
    build = subprocess.run(command, cwd=entry["directory"], capture_output=True, text=True, timeout=120, check=False)
    assert build.returncode == 0, build.stdout + build.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS:" in result.stdout
