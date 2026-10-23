# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Restore/replay publishes AICPU-only queues with valid deferred completions."""

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
MODULE = "tests.st.a2a3.host_build_graph.kernel_mode_capture.test_hbg_kernel_deferred_capture"
RUNTIME = "host_build_graph"


@pytest.mark.requires_hardware
@pytest.mark.platforms(["a2a3"])
@pytest.mark.runtime(RUNTIME)
@pytest.mark.device_count(1)
def test_hbg_kernel_deferred_capture(st_platform, st_device_ids, kernel_close_faults):  # noqa: F811
    _binaries(st_platform, RUNTIME)
    artifacts = ROOT / "outputs" / "hbg_kernel_capture"
    artifacts.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="deferred-", dir=artifacts))
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
    assert "PASS deferred completion eager/replay A-B-A" in text


def _build_callable(platform):
    from simpler.task_interface import ArgDirection, ChipCallable, CoreCallable  # noqa: PLC0415

    from simpler_setup.elf_parser import extract_text_section  # noqa: PLC0415
    from simpler_setup.kernel_compiler import KernelCompiler  # noqa: PLC0415
    from simpler_setup.pto_isa import ensure_pto_isa_root  # noqa: PLC0415

    compiler = KernelCompiler(platform)
    kernels = Path(__file__).with_name("kernels")
    with tempfile.TemporaryDirectory(prefix="hbg-deferred-") as build:
        orchestration = compiler.compile_orchestration(
            RUNTIME,
            str(kernels / "orchestration/deferred_counter.cpp"),
            build_dir=build,
        )
        incore = compiler.compile_incore(
            str(kernels / "aiv/kernel_deferred_counter.cpp"),
            core_type="aiv",
            pto_isa_root=ensure_pto_isa_root(),
            extra_include_dirs=compiler.get_orchestration_include_dirs(RUNTIME),
            build_dir=build,
        )
    signature = [ArgDirection.IN, ArgDirection.OUT, ArgDirection.SCALAR]
    child = CoreCallable.build(signature=signature, binary=extract_text_section(incore))
    return ChipCallable.build(
        signature=signature,
        func_name="deferred_counter",
        binary=orchestration,
        children=[(0, child)],
    )


def _run(platform, device):
    from simpler.task_interface import CallConfig, ChipStorageTaskArgs, ChipTensor, DataType  # noqa: PLC0415
    from simpler.worker import Worker  # noqa: PLC0415

    chip = _build_callable(platform)
    caller = _Caller(device)
    caller.open()
    lib = caller.acl
    _bind_capture(lib)
    worker = Worker(level=2, execution_mode="kernel", device_id=device, platform=platform, runtime=RUNTIME)
    config = CallConfig()
    config.runtime_env.ring_task_window = 64
    worker.init(config=config)
    pin = worker._kernel_pin_finalizer
    callable_id = worker.kernel_prepare_callable(chip)
    counter = caller.device_buffer([0.0] * 16)
    ready = (ctypes.c_uint32 * 16)(*([7] * 16))
    _check(lib.aclrtMemcpy(counter, ctypes.sizeof(ready), ready, ctypes.sizeof(ready), 1), "seed counter")
    output = caller.device_buffer([-2.0] * 128)
    committed = worker.committed_device_memory()
    graphs = []

    def launch(marker):
        args = ChipStorageTaskArgs()
        args.add_tensor(ChipTensor.make(counter, (16,), DataType.UINT32, child_memory=True))
        args.add_tensor(ChipTensor.make(output, (128,), DataType.FLOAT32, child_memory=True))
        args.add_scalar(marker)
        worker.kernel_launch(callable_id, args, caller_stream=caller.stream)
        args.clear()

    def verify(marker):
        _check(caller.synchronize(), "sync deferred completion")
        actual = (ctypes.c_float * 128)()
        _check(lib.aclrtMemcpy(actual, ctypes.sizeof(actual), output, ctypes.sizeof(actual), 2), "read registration")
        assert list(actual) == [float(marker)] * 128

    for marker in (11, 22, 11):
        launch(marker)
        verify(marker)
    for marker in (11, 22):
        graph = ctypes.c_void_p()
        _check(lib.aclmdlRICaptureBegin(caller.stream, 0), "capture begin")
        launch(marker)
        _check(lib.aclmdlRICaptureEnd(caller.stream, ctypes.byref(graph)), "capture end")
        assert graph.value
        graphs.append(graph)
    poison = (ctypes.c_float * 128)(*([-2.0] * 128))
    for _ in range(20):
        for graph, marker in ((graphs[0], 11), (graphs[1], 22), (graphs[0], 11)):
            _check(lib.aclrtMemcpy(output, ctypes.sizeof(poison), poison, ctypes.sizeof(poison), 1), "poison output")
            _check(lib.aclmdlRIExecuteAsync(graph, caller.stream), "replay deferred completion")
            verify(marker)
        launch(33)
        verify(33)
    assert worker.committed_device_memory() == committed
    for graph in graphs:
        _check(lib.aclmdlRIDestroy(graph), "destroy graph")
    worker.close()
    _assert_closed_cleanly(worker, pin)
    assert not caller.close()
    print("PASS deferred completion eager/replay A-B-A tasks=8 replays=60", flush=True)


if __name__ == "__main__":
    try:
        _run(sys.argv[1], int(sys.argv[2]))
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
