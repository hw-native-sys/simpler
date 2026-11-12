# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Actual A5 local Host mapping, gather and dependent consumer acceptance."""

import ctypes
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest


@pytest.mark.requires_hardware
@pytest.mark.platforms(["a5"])
@pytest.mark.parametrize("runtime", ["tensormap_and_ringbuffer", "host_build_graph"])
def test_local_host_ddr_gather(runtime, request):
    path = Path(__file__).resolve().parents[3] / "examples/a5/tensormap_and_ringbuffer/host_ddr_gather/run.py"
    device = int(str(request.config.getoption("--device")).split("-")[0].split(",")[0])
    # Runtime modules share the process's ACL context; isolate different runtimes.
    subprocess.run([sys.executable, str(path), "--device", str(device), "--runtime", runtime], check=True, timeout=180)


@pytest.mark.requires_hardware
@pytest.mark.platforms(["a5"])
@pytest.mark.parametrize("runtime", ["tensormap_and_ringbuffer", "host_build_graph"])
def test_local_host_mxfp8_with_device_scales(request, runtime):
    path = Path(__file__).resolve().parents[3] / "examples/a5/tensormap_and_ringbuffer/host_ddr_gather/run_mx.py"
    device = int(str(request.config.getoption("--device")).split("-")[0].split(",")[0])
    subprocess.run([sys.executable, str(path), "--device", str(device), "--runtime", runtime], check=True, timeout=180)


@pytest.mark.requires_hardware
@pytest.mark.platforms(["a5"])
@pytest.mark.parametrize("runtime", ["tensormap_and_ringbuffer", "host_build_graph"])
def test_local_host_ddr_writes(request, runtime):
    path = Path(__file__).resolve().parents[3] / "examples/a5/tensormap_and_ringbuffer/host_ddr_gather/run_write.py"
    device = int(str(request.config.getoption("--device")).split("-")[0].split(",")[0])
    subprocess.run([sys.executable, str(path), "--device", str(device), "--runtime", runtime], check=True, timeout=180)


def _build_a3_map_vector_add(runtime):
    from simpler.task_interface import ArgDirection  # noqa: PLC0415

    from simpler_setup.scene_test import compile_chip_callable_spec  # noqa: PLC0415

    kernels = Path(__file__).resolve().parents[3] / "examples/workers/l3/multi_chip_dispatch/kernels"
    signature = [ArgDirection.IN, ArgDirection.IN, ArgDirection.OUT]
    spec = {
        "orchestration": {
            "source": str(kernels / "orchestration/vector_add_orch.cpp"),
            "function_name": "vector_add_orchestration",
            "signature": signature,
        },
        "incores": [
            {
                "func_id": 0,
                "source": str(kernels / "aiv/vector_add_kernel.cpp"),
                "core_type": "aiv",
                "signature": signature,
            }
        ],
    }
    return compile_chip_callable_spec(spec, "a2a3", runtime, ("a3_host_map", runtime))


@pytest.mark.requires_hardware
@pytest.mark.platforms(["a2a3"])
def test_a3_map_reads_and_writes_shared_host_backing(request):
    device = str(request.config.getoption("--device")).split("-")[0].split(",")[0]
    # Release process-scoped ACL state before the next hardware test uses this device.
    subprocess.run([sys.executable, __file__, device], check=True, timeout=180)


def _run_a3_host_map(device: int) -> None:
    import numpy as np  # noqa: PLC0415
    from simpler.task_interface import DataType, TaskArgs, TensorArgType, TensorTransfer  # noqa: PLC0415
    from simpler.worker import Worker  # noqa: PLC0415

    runtime = "tensormap_and_ringbuffer"
    entry = _build_a3_map_vector_add(runtime)
    worker = Worker(level=2, platform="a2a3", runtime=runtime, device_id=device)
    try:
        worker.init()
        handle = worker.register(entry)
        source = worker.create_buffer(128 * 128 * 4)
        result = worker.create_buffer(source.nbytes)
        view = source.tensor((128 * 128,), DataType.FLOAT32)
        args = TaskArgs()
        args.add_tensor(view, TensorArgType.INPUT, transfer=TensorTransfer.H2D)
        args.add_tensor(view, TensorArgType.INPUT, transfer=TensorTransfer.MAP)
        args.add_tensor(
            result.tensor(view.shapes, view.dtype), TensorArgType.OUTPUT_EXISTING, transfer=TensorTransfer.MAP
        )
        for value in (2.0, 3.0):
            with source.host_view() as host:
                host[:] = np.full(view.shapes, value, dtype=np.float32).tobytes()
            worker.submit(handle, args).wait()
            with result.host_view() as host:
                np.testing.assert_array_equal(np.frombuffer(host, dtype=np.float32), 2 * value)
        get_context = ctypes.CDLL("libascendcl.so").aclrtGetCurrentContext
        get_context.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        get_context.restype = ctypes.c_int

        def release_on_unbound_thread():
            before = ctypes.c_void_p()
            rc = get_context(ctypes.byref(before))
            assert rc != 0 and before.value is None
            worker.release_buffer(result)
            worker.release_buffer(source)
            after = ctypes.c_void_p()
            assert get_context(ctypes.byref(after)) == rc
            assert after.value is None

        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(release_on_unbound_thread).result()
    finally:
        worker.close()


@pytest.mark.requires_hardware
@pytest.mark.platforms(["a2a3"])
@pytest.mark.device_count(2)
@pytest.mark.manual
def test_same_host_table_maps_to_two_a3_chips(st_device_ids):
    import numpy as np  # noqa: PLC0415
    from simpler.task_interface import DataType, TaskArgs, TensorArgType, TensorTransfer  # noqa: PLC0415
    from simpler.worker import Worker  # noqa: PLC0415

    runtime = "tensormap_and_ringbuffer"
    shape = (128 * 128,)
    nbytes = shape[0] * np.dtype(np.float32).itemsize
    worker = Worker(level=3, platform="a2a3", runtime=runtime, device_ids=st_device_ids)
    entry = worker.register(_build_a3_map_vector_add(runtime))
    try:
        worker.init()
        table = worker.create_buffer(nbytes)
        result = worker.create_buffer(2 * nbytes)
        table_view = table.tensor(shape, DataType.FLOAT32)
        outputs = [result.tensor(shape, DataType.FLOAT32, byte_offset=i * nbytes) for i in range(2)]
        declarations = TaskArgs()
        declarations.add_tensor(table_view, TensorArgType.INPUT, transfer=TensorTransfer.MAP)
        declarations.add_tensor(outputs[0], TensorArgType.OUTPUT_EXISTING, transfer=TensorTransfer.MAP)

        def dispatch(orch, _args, cfg):
            for i, output in enumerate(outputs):
                args = TaskArgs()
                args.add_tensor(table_view, TensorArgType.INPUT, transfer=TensorTransfer.MAP)
                args.add_tensor(table_view, TensorArgType.INPUT, transfer=TensorTransfer.H2D)
                args.add_tensor(output, TensorArgType.OUTPUT_EXISTING, transfer=TensorTransfer.MAP)
                orch.submit_next_level(entry, args, cfg, worker=i)

        for value in (2.0, 3.0):
            with table.host_view() as host:
                host[:] = np.full(shape, value, dtype=np.float32).tobytes()
            worker.run(dispatch, declarations)
            with result.host_view() as host:
                np.testing.assert_array_equal(np.frombuffer(host, dtype=np.float32), 2 * value)
        worker.release_buffer(table)
        worker.release_buffer(result)
    finally:
        worker.close()


if __name__ == "__main__":
    _run_a3_host_map(int(sys.argv[1]))
