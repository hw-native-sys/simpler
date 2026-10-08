# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A completed producer supplies the next run's native host scalar access."""

import ctypes
import struct
from contextlib import ExitStack
from pathlib import Path

import pytest
import torch
from simpler.buffer import Buffer, create_host_shared_buffer, mint_owner_instance_id
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import CallConfig, ChipWorker, DataType, TaskArgs, Tensor, TensorArgType, TensorTransfer
from simpler.worker import Worker

from simpler_setup.runtime_builder import RuntimeBuilder
from simpler_setup.scene_test import compile_chip_callable_spec, l3_compile_cache_key

_HERE = Path(__file__).resolve().parent
_SIZE = 128 * 128
_RUNTIME = "host_build_graph"


def _build_callable(platform):
    arch = "a5" if platform.startswith("a5") else "a2a3"
    kernel = _HERE.parent / arch / _RUNTIME / "vector_example/kernels/aiv/kernel_add_scalar.cpp"
    spec = {
        "orchestration": {
            "source": str(_HERE / "kernels/orchestration/host_readiness.cpp"),
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN, D.INOUT, D.OUT, D.INOUT],
        },
        "incores": [{"func_id": 0, "source": str(kernel), "core_type": "aiv", "signature": [D.IN, D.OUT]}],
    }
    return compile_chip_callable_spec(
        spec, platform, _RUNTIME, l3_compile_cache_key(__name__, "host_readiness", "scalar", platform, _RUNTIME)
    )


def _host_tensor(worker, tensor):
    return worker.make_tensor_arg(tensor, shapes=(_SIZE,), dtype=DataType.FLOAT32)


def _args(worker, source, control, output, host_control, mode, offset=0.0):
    args = TaskArgs()
    args.add_tensor(_host_tensor(worker, source), TensorArgType.INPUT)
    args.add_tensor(control, TensorArgType.INOUT)
    args.add_tensor(_host_tensor(worker, output), TensorArgType.OUTPUT_EXISTING)
    args.add_tensor(_host_tensor(worker, host_control), TensorArgType.INOUT, transfer=TensorTransfer.NONE)
    args.add_scalar(mode)
    args.add_scalar(struct.unpack("<I", struct.pack("<f", offset))[0])
    return args


@pytest.mark.platforms(["a2a3", "a5", "a2a3sim", "a5sim"])
@pytest.mark.device_count(1)
@pytest.mark.runtime(_RUNTIME)
@pytest.mark.parametrize("storage", ["host", "child", "source"])
@pytest.mark.parametrize("write_control", [False, True], ids=["get", "get-set"])
def test_completed_producer_supplies_native_host_access(st_platform, st_device_ids, storage, write_control):
    with ExitStack() as cleanup:
        worker = Worker(level=2, platform=st_platform, runtime=_RUNTIME, device_id=int(st_device_ids[0]))
        cleanup.callback(worker.close)
        handle = worker.register(_build_callable(st_platform))
        worker.init()
        source = torch.full((_SIZE,), 2.0)
        control = torch.full((_SIZE,), -91.0)
        output = torch.zeros(_SIZE)
        host_control = torch.zeros(_SIZE)
        device = None
        if storage == "child":
            device = worker.malloc(control.nbytes)
            cleanup.callback(worker.free, device)
            worker.copy_to(device, control)
            control_arg = device.tensor((_SIZE,), DataType.FLOAT32)
        elif storage == "source":
            if st_platform.endswith("sim"):
                # Not recorded by the native allocator: this arm exercises incomplete span coverage.
                external = ctypes.create_string_buffer(control.nbytes)
                address = ctypes.addressof(external)
                ctypes.memmove(address, control.data_ptr(), control.nbytes)
            else:
                # The low-level allocator owns this allocation; the public Worker does not.
                address = worker._chip_worker.malloc(control.nbytes)
                cleanup.callback(worker._chip_worker.free, address)
                worker._chip_worker.copy_to(address, control.data_ptr(), control.nbytes)
            device = Buffer.wrap(address=address, nbytes=control.nbytes, location=worker.device_location)
            cleanup.callback(device.close)
            control_arg = Tensor(device, shapes=(_SIZE,), dtype=DataType.FLOAT32)
        else:
            control_arg = _host_tensor(worker, control)

        for offset in (5.0, 11.0, -4.0):
            output.zero_()
            produced = 2.0 + offset
            producer = worker.submit(
                handle, _args(worker, source, control_arg, output, host_control, 0, offset), CallConfig()
            )
            producer.wait(30.0)
            if storage == "host":
                torch.testing.assert_close(control, torch.full_like(control, produced))

            # Explicit copy follows producer completion, before any host consumer.
            if storage == "source":
                if st_platform.endswith("sim"):
                    ctypes.memmove(control.data_ptr(), address, control.nbytes)
                else:
                    worker._chip_worker.copy_from(control.data_ptr(), address, control.nbytes)
            elif device is not None:
                worker.copy_from(control, device)
            host_control.copy_(control)
            mode = 2 if write_control else 1
            consumer = worker.submit(
                handle, _args(worker, source, control_arg, output, host_control, mode), CallConfig()
            )
            consumer.result(30.0)
            expected = torch.full_like(output, 2 * produced + 3 if write_control else 2 + produced)
            torch.testing.assert_close(output, expected)
            if storage == "source":
                if st_platform.endswith("sim"):
                    ctypes.memmove(control.data_ptr(), address, control.nbytes)
                else:
                    worker._chip_worker.copy_from(control.data_ptr(), address, control.nbytes)
            elif device is not None:
                worker.copy_from(control, device)
            assert control[0].item() == produced
            assert host_control[0].item() == produced + (3 if write_control else 0)


@pytest.mark.platforms(["a2a3", "a5", "a2a3sim", "a5sim"])
@pytest.mark.device_count(1)
@pytest.mark.runtime(_RUNTIME)
def test_chipworker_consumes_host_taskargs(st_platform, st_device_ids):
    with ExitStack() as cleanup:
        owner = mint_owner_instance_id()
        buffers = [create_host_shared_buffer((_SIZE + 2) * 4, owner, i + 1) for i in range(4)]
        for buffer in buffers:
            cleanup.callback(buffer.close)
        values = [torch.full((_SIZE + 2,), -999.0) for _ in buffers]
        for buffer, value, initial in zip(buffers, values, (2.0, -91.0, 0.0, 0.0), strict=True):
            value[1:-1] = initial
            ctypes.memmove(buffer.base, value.data_ptr(), buffer.nbytes)
        views = [Tensor(buffer, shapes=(_SIZE,), dtype=DataType.FLOAT32, byte_offset=4) for buffer in buffers]
        worker = ChipWorker()
        worker.init(int(st_device_ids[0]), RuntimeBuilder(st_platform).get_binaries(_RUNTIME))
        cleanup.callback(worker.finalize)
        handle = worker.register_callable(_build_callable(st_platform))

        def arguments(mode, offset=0.0):
            args = TaskArgs()
            for view, tag in zip(
                views[:3], (TensorArgType.INPUT, TensorArgType.INOUT, TensorArgType.OUTPUT_EXISTING), strict=True
            ):
                args.add_tensor(view, tag)
            args.add_tensor(views[3], TensorArgType.INOUT, transfer=TensorTransfer.NONE)
            args.add_scalar(mode)
            args.add_scalar(struct.unpack("<I", struct.pack("<f", offset))[0])
            return args

        for offset in (5.0, 11.0, -4.0):
            worker.run(handle, arguments(0, offset))
            ctypes.memmove(buffers[3].base, buffers[1].base, buffers[1].nbytes)
            worker.run(handle, arguments(2))
            for buffer, value in zip(buffers, values, strict=True):
                ctypes.memmove(value.data_ptr(), buffer.base, buffer.nbytes)
                assert value[0].item() == value[-1].item() == -999.0
            produced = 2.0 + offset
            expected = torch.full((_SIZE,), 2 * produced + 3)
            torch.testing.assert_close(values[2][1:-1], expected)
            assert values[1][1].item() == produced
            assert values[3][1].item() == produced + 3
