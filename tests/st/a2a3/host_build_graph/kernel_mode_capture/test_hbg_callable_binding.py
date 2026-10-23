# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Switch distinct child binaries sharing function ID zero across eager and graphs."""

import ctypes
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

import pytest

from tests.st.a2a3.host_build_graph.kernel_mode_capture.test_hbg_kernel_capture import _bind_capture
from tests.st.a2a3.tensormap_and_ringbuffer.kernel_mode_capture.kernel_capture_values import _check, _TensorIO
from tests.ut.py.test_kernel_mode_c_api import _binaries, kernel_close_faults  # noqa: F401
from tests.ut.py.test_worker.test_worker_kernel_mode_hw import _assert_closed_cleanly, _Caller, _launch_args

ROOT = Path(__file__).resolve().parents[5]
MODULE = "tests.st.a2a3.host_build_graph.kernel_mode_capture.test_hbg_callable_binding"
RUNTIME = "host_build_graph"


@pytest.mark.requires_hardware
@pytest.mark.platforms(["a2a3"])
@pytest.mark.runtime(RUNTIME)
@pytest.mark.device_count(1)
def test_hbg_callable_binding(st_platform, st_device_ids, kernel_close_faults):  # noqa: F811
    _binaries(st_platform, RUNTIME)
    artifacts = ROOT / "outputs" / "hbg_kernel_capture"
    artifacts.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="callable-binding-", dir=artifacts))
    (output / "ascend").mkdir()
    env = dict(os.environ, ASCEND_PROCESS_LOG_PATH=str(output / "ascend"))
    env["LD_PRELOAD"] = str(kernel_close_faults) + (":" + env["LD_PRELOAD"] if env.get("LD_PRELOAD") else "")
    with (output / "run.log").open("w") as log:
        result = subprocess.run(
            [sys.executable, "-m", MODULE, st_platform, str(st_device_ids[0])],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=300,
            check=False,
        )
    text = (output / "run.log").read_text()
    assert result.returncode == 0, f"{text}\nArtifacts: {output}"
    assert "PASS callable binding eager/graph ABA contexts=2" in text


def _build_callables(platform):
    from simpler.task_interface import ArgDirection, ChipCallable, CoreCallable  # noqa: PLC0415

    from simpler_setup.elf_parser import extract_text_section  # noqa: PLC0415
    from simpler_setup.kernel_compiler import KernelCompiler  # noqa: PLC0415
    from simpler_setup.pto_isa import ensure_pto_isa_root  # noqa: PLC0415

    compiler = KernelCompiler(platform)
    source = ROOT / f"examples/{platform}/tensormap_and_ringbuffer/vector_example/kernels/aiv/kernel_add_scalar.cpp"
    original = source.read_text()
    expression = "float scalar = from_u64<float>(static_cast<uint64_t>(args[2]));"
    assert original.count(expression) == 1
    with tempfile.TemporaryDirectory(prefix="hbg-callable-binding-") as build:
        subtract = Path(build) / "kernel_sub_scalar.cpp"
        subtract.write_text(original.replace(expression, expression.replace("= from_u64", "= -from_u64")))
        orchestration = compiler.compile_orchestration(
            RUNTIME, str(ROOT / "tests/ut/py/kernel_eager_orchestration.cpp"), build_dir=build
        )
        binaries = [
            extract_text_section(
                compiler.compile_incore(
                    str(kernel),
                    core_type="aiv",
                    pto_isa_root=ensure_pto_isa_root(),
                    extra_include_dirs=compiler.get_orchestration_include_dirs(RUNTIME),
                    build_dir=build,
                )
            )
            for kernel in (source, subtract)
        ]
    assert binaries[0] != binaries[1]
    signature = [ArgDirection.IN, ArgDirection.OUT, ArgDirection.SCALAR]
    return [
        ChipCallable.build(
            signature=signature,
            func_name="kernel_eager_orchestration",
            binary=orchestration,
            children=[(0, CoreCallable.build(signature=signature, binary=binary))],
        )
        for binary in binaries
    ]


def _run(platform, device):
    from simpler.task_interface import CallConfig  # noqa: PLC0415
    from simpler.worker import Worker  # noqa: PLC0415

    chips = _build_callables(platform)
    caller = _Caller(device)
    caller.open()
    lib = caller.acl
    _bind_capture(lib)
    for context in range(2):
        worker = Worker(level=2, execution_mode="kernel", device_id=device, platform=platform, runtime=RUNTIME)
        config = CallConfig()
        config.runtime_env.ring_task_window = 64
        worker.init(config=config)
        pin = worker._kernel_pin_finalizer
        ids = [worker.kernel_prepare_callable(chip) for chip in chips]
        assert ids[0] != ids[1]
        allocations = []
        io = _TensorIO(lib, allocations)
        source, output = io.allocate(), io.allocate()

        def launch(index, scalar):
            args = _launch_args(source.value, output.value, scalar)
            worker.kernel_launch(ids[index], args, caller_stream=caller.stream)
            args.clear()

        def seed_inputs(index, scalar, seed):
            values = [float(i % 127 + seed) for i in range(128 * 128)]
            io.write(source, values)
            io.write(output, [-999.0] * len(values))
            return [value + (scalar if index == 0 else -scalar) for value in values]

        # Neither graph has executed when B first replays. Use each capture mode
        # in a fresh context so no earlier eager call can publish its authority.
        cold_graphs = []
        for index in range(2):
            graph = ctypes.c_void_p()
            _check(lib.aclmdlRICaptureBegin(caller.stream, context), "cold capture begin")
            launch(index, 3.0)
            _check(lib.aclmdlRICaptureEnd(caller.stream, ctypes.byref(graph)), "cold capture end")
            assert graph.value
            cold_graphs.append(graph)
        for iteration, index in enumerate((1, 0, 1)):
            expected = seed_inputs(index, 3.0, context + iteration)
            _check(lib.aclmdlRIExecuteAsync(cold_graphs[index], caller.stream), "cold graph replay")
            _check(lib.aclrtSynchronizeStreamWithTimeout(caller.stream, 10000), "cold graph sync")
            io.verify(output, expected)
        for graph in cold_graphs:
            _check(lib.aclmdlRIDestroy(graph), "destroy cold graph")

        for index, scalar in ((0, 3.0), (1, 3.0), (0, 3.0), (0, 7.0)):
            expected = seed_inputs(index, scalar, context)
            launch(index, scalar)
            _check(lib.aclrtSynchronizeStreamWithTimeout(caller.stream, 10000), "eager sync")
            io.verify(output, expected)

        graphs = []
        for index in range(2):
            graph = ctypes.c_void_p()
            _check(lib.aclmdlRICaptureBegin(caller.stream, 0), "capture begin")
            launch(index, 3.0)
            _check(lib.aclmdlRICaptureEnd(caller.stream, ctypes.byref(graph)), "capture end")
            assert graph.value
            graphs.append(graph)
        for iteration, index in enumerate((0, 1, 0, 0, 1, 0)):
            expected = seed_inputs(index, 3.0, context + iteration + 1)
            _check(lib.aclmdlRIExecuteAsync(graphs[index], caller.stream), "graph replay")
            _check(lib.aclrtSynchronizeStreamWithTimeout(caller.stream, 10000), "graph sync")
            io.verify(output, expected)
            if index == 0:
                expected = seed_inputs(1, 7.0, iteration)
                launch(1, 7.0)
                _check(lib.aclrtSynchronizeStreamWithTimeout(caller.stream, 10000), "interleaved eager sync")
                io.verify(output, expected)
        for graph in graphs:
            _check(lib.aclmdlRIDestroy(graph), "destroy graph")
        worker.close()
        _assert_closed_cleanly(worker, pin)
        for address in reversed(allocations):
            _check(lib.aclrtFree(address), "free tensor")
    assert not caller.close()
    print("PASS callable binding eager/graph ABA contexts=2", flush=True)


if __name__ == "__main__":
    try:
        _run(sys.argv[1], int(sys.argv[2]))
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
