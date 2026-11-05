# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Hardware ST for examples/workers/l3/multi_chip_dispatch."""

import ctypes
import os
import struct
from pathlib import Path

import pytest
from simpler.chip_process import launch_chip
from simpler.task_interface import CallConfig, DataType, TaskArgs, TensorArgType, _bind_host_log_session_directory
from simpler.worker import Worker

from .main import N_COLS, N_ROWS, build_chip_callable, run


@pytest.mark.platforms(["a2a3sim", "a2a3", "a5sim", "a5"])
@pytest.mark.manual(["a2a3sim", "a5sim"])
@pytest.mark.runtime("tensormap_and_ringbuffer")
@pytest.mark.device_count(2)
def test_multi_chip_dispatch(st_platform, st_device_ids):
    rc = run(st_platform, [int(d) for d in st_device_ids])
    assert rc == 0


@pytest.mark.platforms(["a2a3sim", "a5sim"])
@pytest.mark.runtime("tensormap_and_ringbuffer")
@pytest.mark.device_count(2)
def test_exec_chips_compute_from_heap_slices(monkeypatch, st_platform, st_device_ids):
    monkeypatch.setattr(Worker, "_launch_chip_process", launch_chip)
    count = N_ROWS * N_COLS
    inputs = [struct.pack(f"{count}f", *(float(i + offset) for i in range(count))) for offset in (0, 3, 7)]
    expected = [struct.pack(f"{count}f", *(float(2 * i + offset) for i in range(count))) for offset in (3, 7)]
    guard = b"\xa5" * 64

    def verify(args):
        assert bytes(args[0].buffer) == expected[0]
        assert bytes(args[1].buffer) == expected[1]
        assert bytes(args[2].buffer) == bytes(args[3].buffer) == guard
        struct.pack_into("i", args[4].buffer, 0, 1)

    worker = Worker(
        level=3,
        platform=st_platform,
        runtime="tensormap_and_ringbuffer",
        device_ids=[int(device) for device in st_device_ids],
        num_sub_workers=1,
        heap_ring_size=512 * 1024,
    )
    chip_handle = worker.register(build_chip_callable(st_platform))
    verify_handle = worker.register(verify)
    pids = []
    try:
        worker.init()
        pids = list(worker._chip_pids) + list(worker._sub_pids)
        result = worker.create_buffer(4)
        assert result.shm is not None and result.shm.buf is not None

        def graph(orch, _args, config):
            before = orch.alloc((16,), DataType.FLOAT32)
            a, b0, b1, out0, out1 = [orch.alloc((N_ROWS, N_COLS), DataType.FLOAT32) for _ in range(5)]
            after = orch.alloc((16,), DataType.FLOAT32)
            for slab in (before, after):
                ctypes.memmove(slab.base, guard, len(guard))
            for slab, payload in zip((a, b0, b1), inputs):
                ctypes.memmove(slab.base, payload, len(payload))
            for index, (b, output) in enumerate(zip((b0, b1), (out0, out1))):
                assert int.from_bytes(output.body[8:16], "little") > 0
                ctypes.memset(output.base, 0, output.nbytes)
                args = TaskArgs()
                args.add_tensor(a.tensor((N_ROWS, N_COLS), DataType.FLOAT32), TensorArgType.INPUT)
                args.add_tensor(b.tensor((N_ROWS, N_COLS), DataType.FLOAT32), TensorArgType.INPUT)
                args.add_tensor(output.tensor((N_ROWS, N_COLS), DataType.FLOAT32), TensorArgType.OUTPUT_EXISTING)
                orch.submit_next_level(chip_handle, args, config, worker=index)
            args = TaskArgs()
            for slab in (out0, out1, before, after):
                args.add_tensor(slab.tensor((slab.nbytes // 4,), DataType.FLOAT32), TensorArgType.INPUT)
            args.add_tensor(result.tensor((1,), DataType.INT32), TensorArgType.OUTPUT_EXISTING)
            orch.submit_sub(verify_handle, args)

        for _ in range(3):
            result.shm.buf[:4] = b"\0" * 4
            worker.run(graph, config=CallConfig())
            assert struct.unpack_from("i", result.shm.buf)[0] == 1
    finally:
        worker.close()
    assert len(pids) == 3
    directory = Path(_bind_host_log_session_directory())
    for pid in pids[:2]:
        assert "name=chip.run " in (directory / f"host.{pid}.log").read_text()
    for pid in pids:
        with pytest.raises(ChildProcessError):
            os.waitpid(pid, os.WNOHANG)
