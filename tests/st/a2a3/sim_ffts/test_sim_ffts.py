# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Explicit FFTS simulator synchronization across separate kernel DSOs."""

import subprocess
import sys
from pathlib import Path

import pytest

from simpler_setup import SceneTestLevel, scene_level
from simpler_setup.environment import PROJECT_ROOT
from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.pto_isa import ensure_pto_isa_root


@scene_level(SceneTestLevel.CHIP)
@pytest.mark.platforms(["a2a3sim"])
@pytest.mark.runtime("host_build_graph")
def test_ffts_mode2_credits_across_kernel_dsos(tmp_path, st_platform):
    compiler = KernelCompiler(st_platform)
    isa = ensure_pto_isa_root()
    source_dir = Path(__file__).parent / "fixtures"
    libraries = []
    for core_type in ("aic", "aiv"):
        binary = compiler._compile_incore_sim(
            str(source_dir / "kernel.cpp"), core_type=core_type, pto_isa_root=isa, build_dir=str(tmp_path)
        )
        library = tmp_path / f"{core_type}.so"
        library.write_bytes(binary)
        libraries.append(library)

    context = PROJECT_ROOT / "build" / "lib" / "libcpu_sim_context.so"
    assert context.is_file(), "Install the simulator runtime before running FFTS integration tests"
    driver = tmp_path / "driver"
    command = [
        compiler.gxx15.cxx_path,
        "-std=c++23",
        "-O2",
        "-pthread",
        f"-I{PROJECT_ROOT / 'src/common/platform/sim/sim_context'}",
        str(source_dir / "driver.cpp"),
        str(context),
        f"-Wl,-rpath,{context.parent}",
        "-o",
        str(driver),
    ]
    if sys.platform != "darwin":
        command.append("-ldl")
    subprocess.run(command, check=True, capture_output=True, text=True)
    result = subprocess.run(
        [str(driver), *map(str, libraries)], check=False, capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS" in result.stdout
