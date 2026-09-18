# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""HBG kernel-mode ACLGraph capture/replay through the public Worker interface."""

import ctypes
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

import pytest

from tests.st.a2a3.tensormap_and_ringbuffer.kernel_mode_capture.graph_cases import REPLAYS, run_graph_case
from tests.st.a2a3.tensormap_and_ringbuffer.kernel_mode_capture.kernel_capture_values import _check, _TensorIO
from tests.ut.py.test_kernel_mode_c_api import _binaries, _build_eager_callable
from tests.ut.py.test_worker.test_worker_kernel_mode_hw import _assert_closed_cleanly, _Caller, _launch_args

ROOT = Path(__file__).resolve().parents[5]
MODULE = "tests.st.a2a3.host_build_graph.kernel_mode_capture.test_hbg_kernel_capture"
RUNTIME = "host_build_graph"
SCENARIOS = ("fresh_inputs", "chain", "feedback_batch", "eager_replay", "graph_recreate", "long_chain")


@pytest.mark.requires_hardware
@pytest.mark.platforms(["a2a3"])
@pytest.mark.runtime(RUNTIME)
@pytest.mark.device_count(1)
@pytest.mark.parametrize("intermediate", [False, True], ids=["single_task", "internal_tensor"])
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_hbg_kernel_capture(st_platform, st_device_ids, scenario, intermediate):
    _binaries(st_platform, RUNTIME)
    artifacts = ROOT / "outputs" / "hbg_kernel_capture"
    artifacts.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix=f"{scenario}-{int(intermediate)}-", dir=artifacts))
    logs = output / "ascend"
    logs.mkdir()
    env = dict(os.environ, ASCEND_PROCESS_LOG_PATH=str(logs))
    with (output / "run.log").open("w") as log:
        try:
            result = subprocess.run(
                [sys.executable, "-m", MODULE, st_platform, str(st_device_ids[0]), scenario, str(int(intermediate))],
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=300,
                check=False,
            )
        except subprocess.TimeoutExpired:
            pytest.fail(f"HBG capture timed out; artifacts: {output}")
    text = (output / "run.log").read_text()
    assert result.returncode == 0, f"{text}\nArtifacts: {output}"
    assert f"PASS {scenario} intermediate={int(intermediate)} replays={REPLAYS}" in text


def _bind_capture(lib):
    for symbol, arguments in {
        "aclmdlRICaptureBegin": [ctypes.c_void_p, ctypes.c_int],
        "aclmdlRICaptureEnd": [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)],
        "aclmdlRIExecuteAsync": [ctypes.c_void_p, ctypes.c_void_p],
        "aclmdlRIDestroy": [ctypes.c_void_p],
    }.items():
        function = getattr(lib, symbol)
        function.argtypes = arguments
        function.restype = ctypes.c_int


def _run(platform, device, scenario, intermediate):
    from simpler.task_interface import CallConfig  # noqa: PLC0415
    from simpler.worker import Worker  # noqa: PLC0415

    chip = _build_eager_callable(platform, RUNTIME, intermediate=intermediate)
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
    committed = worker.committed_device_memory()
    assert committed > 0
    allocations = []
    io = _TensorIO(lib, allocations)
    graphs = []
    host_launches = 0
    replays = 0

    def sync():
        _check(lib.aclrtSynchronizeStreamWithTimeout(caller.stream, 10000), "caller sync")

    def launch(cid, tensors, scalar):
        nonlocal host_launches
        assert cid == 0
        source, destination = tensors
        # Both child tasks add the scalar; graph-level oracles use the total increment.
        args = _launch_args(source.value, destination.value, scalar / (2 if intermediate else 1))
        worker.kernel_launch(callable_id, args, caller_stream=caller.stream)
        args.clear()
        host_launches += 1

    def record(nodes):
        graph = ctypes.c_void_p()
        _check(lib.aclmdlRICaptureBegin(caller.stream, 0), "capture begin")
        for cid, tensors, scalar in nodes:
            launch(cid, tensors, scalar)
        _check(lib.aclmdlRICaptureEnd(caller.stream, ctypes.byref(graph)), "capture end")
        assert graph.value
        graphs.append(graph)
        return graph

    def replay(graph):
        nonlocal replays
        _check(lib.aclmdlRIExecuteAsync(graph, caller.stream), "replay")
        replays += 1

    def destroy(graph):
        _check(lib.aclmdlRIDestroy(graph), "destroy graph")
        graphs.remove(graph)

    run_graph_case(scenario, io, record, replay, launch, sync, destroy)
    assert replays == REPLAYS
    assert worker.committed_device_memory() == committed
    sync()
    for graph in list(graphs):
        destroy(graph)
    worker.close()
    _assert_closed_cleanly(worker, pin)
    for address in reversed(allocations):
        _check(lib.aclrtFree(address), "free tensor")
    assert not caller.close()
    print(
        f"PASS {scenario} intermediate={int(intermediate)} replays={replays} "
        f"host_launches={host_launches} committed_bytes={committed} memory_growth=0",
        flush=True,
    )


if __name__ == "__main__":
    try:
        _run(sys.argv[1], int(sys.argv[2]), sys.argv[3], bool(int(sys.argv[4])))
    except BaseException:
        # A failed capture can still own device resources; leave their retirement to process teardown.
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
