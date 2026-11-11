# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Verify real host-runtime service leases before final dlclose."""

import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
# The mapping check runs on every platform. The probe also compares thread
# counts where procfs exists; elsewhere it reports them as unavailable.
_LINUX = sys.platform.startswith("linux")


@pytest.fixture(scope="module")
def unload_probe(tmp_path_factory):
    executable = tmp_path_factory.mktemp("hbg-unload") / "probe"
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-pthread",
            "-I" + str(_ROOT / "src/common/worker"),
            str(Path(__file__).with_name("support") / "hbg_runtime_unload_probe.cpp"),
            str(_ROOT / "src/common/worker/runtime_module.cpp"),
            *(["-ldl"] if _LINUX else []),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )
    return executable


@pytest.mark.parametrize("arch", ["a2a3", "a5"])
@pytest.mark.parametrize("prewarm", [False, True])
def test_last_owner_unloads_real_runtime(arch, prewarm, unload_probe):
    libraries = _ROOT / "build/lib"
    runtime = libraries / arch / "sim/host_build_graph/libhost_runtime.so"
    assert runtime.exists(), f"Build runtimes before testing: {runtime}"
    result = subprocess.run(
        [str(unload_probe), str(libraries / "libcpu_sim_context.so"), str(runtime), str(int(prewarm))],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("mapped=0") == 3, result.stdout
