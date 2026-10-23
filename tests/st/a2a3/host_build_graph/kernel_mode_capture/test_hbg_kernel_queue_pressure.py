# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Kernel-mode queue capacity boundaries, SPMD requeue and sync fanout."""

import ctypes
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

import pytest

from tests.st.a2a3.host_build_graph.kernel_mode_capture.test_hbg_kernel_capture import _bind_capture
from tests.st.a2a3.tensormap_and_ringbuffer.kernel_mode_capture.kernel_capture_values import _check
from tests.ut.py.test_kernel_mode_c_api import _binaries, kernel_close_faults  # noqa: F401
from tests.ut.py.test_worker.test_worker_kernel_mode_hw import _assert_closed_cleanly, _Caller

ROOT = Path(__file__).resolve().parents[5]
MODULE = "tests.st.a2a3.host_build_graph.kernel_mode_capture.test_hbg_kernel_queue_pressure"
RUNTIME = "host_build_graph"


@pytest.mark.requires_hardware
@pytest.mark.platforms(["a2a3"])
@pytest.mark.runtime(RUNTIME)
@pytest.mark.device_count(1)
def test_hbg_kernel_queue_pressure(st_platform, st_device_ids, kernel_close_faults):  # noqa: F811
    _binaries(st_platform, RUNTIME)
    artifacts = ROOT / "outputs" / "hbg_kernel_capture"
    artifacts.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="queue-pressure-", dir=artifacts))
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
    assert "PASS queue pressure eager/replay window=512" in text


def _build_callable(platform):
    from simpler.task_interface import ArgDirection, ChipCallable, CoreCallable  # noqa: PLC0415

    from simpler_setup.elf_parser import extract_text_section  # noqa: PLC0415
    from simpler_setup.kernel_compiler import KernelCompiler  # noqa: PLC0415
    from simpler_setup.pto_isa import ensure_pto_isa_root  # noqa: PLC0415

    compiler = KernelCompiler(platform)
    kernels = Path(__file__).with_name("kernels")
    with tempfile.TemporaryDirectory(prefix="hbg-queue-pressure-") as build:
        orchestration = compiler.compile_orchestration(
            RUNTIME,
            str(kernels / "orchestration/queue_pressure.cpp"),
            build_dir=build,
        )
        incore = compiler.compile_incore(
            str(kernels / "aiv/kernel_queue_pressure.cpp"),
            core_type="aiv",
            pto_isa_root=ensure_pto_isa_root(),
            extra_include_dirs=compiler.get_orchestration_include_dirs(RUNTIME),
            build_dir=build,
        )
    signature = [ArgDirection.OUT] + [ArgDirection.SCALAR] * 4
    child = CoreCallable.build(
        signature=[ArgDirection.OUT] + [ArgDirection.SCALAR] * 3, binary=extract_text_section(incore)
    )
    return ChipCallable.build(
        signature=signature,
        func_name="queue_pressure",
        binary=orchestration,
        children=[(0, child)],
    )


def _run(platform, device, aicpu_threads=3, scenarios=None):
    from simpler.task_interface import CallConfig, ChipStorageTaskArgs, ChipTensor, DataType  # noqa: PLC0415
    from simpler.worker import Worker  # noqa: PLC0415

    chip = _build_callable(platform)
    caller = _Caller(device)
    caller.open()
    lib = caller.acl
    _bind_capture(lib)
    worker = Worker(level=2, execution_mode="kernel", device_id=device, platform=platform, runtime=RUNTIME)
    config = CallConfig()
    config.runtime_env.ring_task_window = 512
    config.aicpu_thread_num = aicpu_threads
    worker.init(config=config)
    pin = worker._kernel_pin_finalizer
    callable_id = worker.kernel_prepare_callable(chip)
    elements = 128 * 129 * 16
    output = caller.device_buffer([-2.0] * elements)
    committed = worker.committed_device_memory()
    graphs = []
    if scenarios is None:
        scenarios = ((64, 129, 0), (128, 129, 0), (65, 129, 0), (64, 2, 1), (1, 129, 0), (0, 129, 0))

    def launch(marker, scenario):
        args = ChipStorageTaskArgs()
        args.add_tensor(ChipTensor.make(output, (elements,), DataType.FLOAT32, child_memory=True))
        for scalar in (marker, *scenario):
            args.add_scalar(scalar)
        worker.kernel_launch(callable_id, args, caller_stream=caller.stream)
        args.clear()

    poison = (ctypes.c_float * elements)(*([-2.0] * elements))

    def clear():
        _check(lib.aclrtMemcpy(output, ctypes.sizeof(poison), poison, ctypes.sizeof(poison), 1), "poison output")

    def verify(marker, scenario):
        _check(caller.synchronize(), "sync queue pressure")
        actual = (ctypes.c_float * elements)()
        _check(lib.aclrtMemcpy(actual, ctypes.sizeof(actual), output, ctypes.sizeof(actual), 2), "read output")
        count, blocks, _ = scenario
        expected = [
            float(marker + task * 256 + block) for task in range(count) for block in range(blocks) for _ in range(16)
        ]
        expected += [-2.0] * (elements - len(expected))
        assert list(actual) == expected, (marker, scenario)

    for scenario in scenarios:
        print(f"PRESSURE_EAGER threads={aicpu_threads} scenario={scenario}", flush=True)
        clear()
        launch(11, scenario)
        verify(11, scenario)
        graph = ctypes.c_void_p()
        _check(lib.aclmdlRICaptureBegin(caller.stream, 0), "capture begin")
        launch(22, scenario)
        _check(lib.aclmdlRICaptureEnd(caller.stream, ctypes.byref(graph)), "capture end")
        assert graph.value
        graphs.append(graph)
    for _ in range(3):
        for graph, scenario in zip(graphs, scenarios, strict=True):
            clear()
            _check(lib.aclmdlRIExecuteAsync(graph, caller.stream), "replay queue pressure")
            verify(22, scenario)
            clear()
            launch(33, scenarios[0])
            verify(33, scenarios[0])
    assert worker.committed_device_memory() == committed
    for graph in graphs:
        _check(lib.aclmdlRIDestroy(graph), "destroy graph")
    worker.close()
    _assert_closed_cleanly(worker, pin)
    assert not caller.close()
    print(f"PASS queue pressure eager/replay window=512 threads={aicpu_threads} scenarios={scenarios}", flush=True)


if __name__ == "__main__":
    try:
        _run(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 3)
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
