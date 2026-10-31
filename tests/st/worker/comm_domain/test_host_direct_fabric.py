# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Direct L2 Host Fabric views use the same native attachment as L3."""

import ctypes
import struct

import pytest
from simpler.task_interface import AccessMode, DataType, HostWindowSpec, TaskArgs, TensorArgType, TensorTransfer
from simpler.worker import Worker

from tests.st.worker.comm_domain.host_domain_kernels import HOST_RUNTIMES, build_chip_callable, require_a3_host_fabric

N = 128 * 128
SIZE = N * 4


@pytest.mark.platforms(["a2a3sim", "a2a3"])
@pytest.mark.parametrize("runtime", HOST_RUNTIMES)
@pytest.mark.device_count(1)
@pytest.mark.parametrize("mixed", [False, True], ids=["mapped", "native-and-mapped"])
def test_direct_host_fabric_two_live_buffers(st_platform, st_device_ids, mixed, runtime):
    worker = Worker(level=2, platform=st_platform, runtime=runtime, device_id=int(st_device_ids[0]))
    chip = worker.register(build_chip_callable(st_platform, runtime))
    try:
        worker.init()
        require_a3_host_fabric(st_platform)
        sources = [worker.create_buffer(SIZE, host_window=HostWindowSpec(SIZE)) for _ in range(2)]
        result = worker.create_buffer(SIZE)
        for iteration, source in enumerate(sources):
            payload = struct.pack(f"<{N}f", *([float(iteration + 2)] * N))
            ctypes.memmove(source.base, payload, SIZE)
            native = source.tensor((N,), DataType.FLOAT32)
            view = worker.map_view(native)
            assert native.mapping_ref == 0 and view.mapping_ref != 0
            assert native.buffer.identity == view.buffer.identity
            args = TaskArgs()
            args.add_tensor(
                native if mixed else view,
                TensorArgType.INPUT,
                transfer=TensorTransfer.H2D if mixed else TensorTransfer.NONE,
            )
            args.add_tensor(view, TensorArgType.INPUT, transfer=TensorTransfer.NONE)
            args.add_tensor(result.tensor((N,), DataType.FLOAT32), TensorArgType.OUTPUT_EXISTING)
            worker.run(chip, args)
            assert struct.unpack_from(f"<{N}f", result.shm.buf) == (float(2 * (iteration + 2)),) * N
            worker.unmap_view(view)
            with pytest.raises(ValueError, match="live explicit mapping"):
                worker.run(chip, args)
            remapped = worker.map_view(native, access=AccessMode.READ)
            assert remapped.mapping_ref != view.mapping_ref
            retry = TaskArgs()
            retry.add_tensor(
                native if mixed else remapped,
                TensorArgType.INPUT,
                transfer=TensorTransfer.H2D if mixed else TensorTransfer.NONE,
            )
            retry.add_tensor(remapped, TensorArgType.INPUT, transfer=TensorTransfer.NONE)
            retry.add_tensor(result.tensor((N,), DataType.FLOAT32), TensorArgType.OUTPUT_EXISTING)
            worker.run(chip, retry)
            assert struct.unpack_from(f"<{N}f", result.shm.buf) == (float(2 * (iteration + 2)),) * N
            worker.release_buffer(source)
            with pytest.raises(ValueError):
                worker.map_view(native)
    finally:
        worker.close()
