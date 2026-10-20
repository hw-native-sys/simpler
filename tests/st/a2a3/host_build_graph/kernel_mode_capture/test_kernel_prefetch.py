# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""HBG kernel-mode SDMA ownership and real TPREFETCH_ASYNC eager/replay coverage."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from tests.st.a2a3.tensormap_and_ringbuffer.kernel_mode_capture.test_kernel_mode_capture import (
    ROOT,
    _fail,
    capture_observer,  # noqa: F401
)
from tests.st.a2a3.tensormap_and_ringbuffer.kernel_mode_capture.test_kernel_prefetch import (
    _run,
    _run_program_isolation,
)
from tests.ut.py.test_kernel_mode_c_api import _binaries

RUNTIME = "host_build_graph"
MODULE = "tests.st.a2a3.host_build_graph.kernel_mode_capture"


@pytest.mark.sdma
@pytest.mark.requires_hardware
@pytest.mark.platforms(["a2a3"])
@pytest.mark.runtime(RUNTIME)
@pytest.mark.device_count(1)
@pytest.mark.parametrize("scenario", ["eager", "replay", "recreate", "disabled", "program_isolation"])
def test_hbg_kernel_prefetch(st_platform, st_device_ids, scenario, capture_observer):  # noqa: F811
    _binaries(st_platform, RUNTIME)
    artifacts = ROOT / "outputs" / "kernel_mode"
    artifacts.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="prefetch-" + scenario + "-", dir=artifacts))
    logs = output / "ascend"
    logs.mkdir()
    env = dict(os.environ)
    env["ASCEND_PROCESS_LOG_PATH"] = str(logs)
    env["LD_PRELOAD"] = str(capture_observer) + (":" + env["LD_PRELOAD"] if env.get("LD_PRELOAD") else "")
    with (output / "run.log").open("w") as log:
        result = subprocess.run(
            [sys.executable, "-m", MODULE + ".test_kernel_prefetch", str(st_device_ids[0]), scenario, str(output)],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=300,
            check=False,
        )
    text = (output / "run.log").read_text()
    assert result.returncode == 0, f"{text}\nArtifacts: {output}"
    expected = f"PASS prefetch {scenario} rounds=100"
    if scenario != "program_isolation":
        expected += " forbidden_sync=0 allocations=0"
    assert expected in text


def _build_prefetch(build_dir):
    from simpler.task_interface import ArgDirection, ChipCallable, CoreCallable  # noqa: PLC0415

    from simpler_setup.elf_parser import extract_text_section  # noqa: PLC0415
    from simpler_setup.kernel_compiler import KernelCompiler  # noqa: PLC0415
    from simpler_setup.pto_isa import ensure_pto_isa_root  # noqa: PLC0415

    source = ROOT / "examples/a2a3/tensormap_and_ringbuffer/prefetch_async_demo/kernels"
    compiler = KernelCompiler("a2a3")
    orchestration = compiler.compile_orchestration(
        RUNTIME, str(Path(__file__).with_name("prefetch_orch.cpp")), build_dir=str(build_dir)
    )
    incore = compiler.compile_incore(
        str(source / "aiv/kernel_prefetch_copy.cpp"),
        core_type="aiv",
        pto_isa_root=ensure_pto_isa_root(),
        extra_include_dirs=compiler.get_orchestration_include_dirs(RUNTIME),
        build_dir=str(build_dir),
    )
    signature = [ArgDirection.IN, ArgDirection.OUT]
    child = CoreCallable.build(signature=signature, binary=extract_text_section(incore))
    return ChipCallable.build(
        signature=signature, func_name="prefetch_async_orchestration", binary=orchestration, children=[(0, child)]
    )


if __name__ == "__main__":
    try:
        options = {"runtime": RUNTIME, "build_prefetch": _build_prefetch}
        if sys.argv[2] == "program_isolation":
            _run_program_isolation(int(sys.argv[1]), Path(sys.argv[3]), **options)
        else:
            _run(int(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), **options)
    except BaseException:
        _fail()
