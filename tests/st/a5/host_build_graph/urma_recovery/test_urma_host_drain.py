# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Sample an unconsumed GET on the AIV, then sample the same queue after Host wait."""

import ctypes
import multiprocessing
import time
from pathlib import Path

import pytest
from simpler.task_interface import AddressSpace, ChipCallable, ChipStorageTaskArgs, ChipTensor, ChipWorker, DataType

from simpler_setup.runtime_builder import RuntimeBuilder
from simpler_setup.scene_test import _pto_isa_compile_cache_token, compile_chip_callable_spec, scene_level
from tests.st.a5.host_build_graph.urma_row_reuse.test_urma_row_reuse import _callable


class _QueueSample(ctypes.Structure):
    _fields_ = [
        (name, ctypes.c_uint32)
        for name in (
            "phase",
            "sender",
            "rows_per_peer",
            "row",
            "target_bb",
            "target_cqe",
            "registered",
            "head",
            "submitted",
            "sq_tail",
            "cq_tail",
            "doorbell",
            "cqe_status",
        )
    ]


def _run_rank(platform, rank, device, rootinfo, ready, image, observe):
    chip = ChipWorker()
    chip.init(device, RuntimeBuilder(platform).get_binaries("host_build_graph"))
    handle = chip.register_callable(ChipCallable.from_bytes(image))
    comm = chip.comm_init(rank, 2, rootinfo)
    committed = ctypes.c_uint64()
    ctx, window, _ = chip._impl.comm_alloc_domain_windows(
        comm, 1, [0, 1], rank, 4 * 1024 * 1024, ctypes.addressof(committed), workspace_mask=1 << 1
    )
    assert committed.value == 1
    count, guard, header = 8192, 32, 32
    total = header + count + 2 * guard
    data = ctypes.c_float * count

    def payload(sender):
        return [1000 * (sender + 1) + i % 251 for i in range(count)]

    source = data(*payload(rank))
    output = (ctypes.c_float * total)(*([0.0] * header + [-77.0] * (count + 2 * guard)))
    output_addr = chip.malloc(ctypes.sizeof(output))
    for addr, values in ((window, source), (output_addr, output)):
        chip.copy_to(addr, ctypes.addressof(values), ctypes.sizeof(values))
    ready.wait()

    args = ChipStorageTaskArgs()
    # terminal_get uses tensor 0 and tensor 2; the other two belong to its shared orchestration signature.
    for addr in (window, window + ctypes.sizeof(output), output_addr, window + 2 * ctypes.sizeof(output)):
        args.add_tensor(ChipTensor.make(addr, (total,), DataType.FLOAT32, address_space=AddressSpace.DEVICE))
    for scalar in (ctx, total, 0, 1):
        args.add_scalar(scalar)

    def execute(generation):
        run = chip._prepare_native_run_with_pipeline_lease(chip._resolve_handle(handle).slot_id, args, 0, generation)
        chip._launch_native_run(run)
        if generation == 1 and observe == "poll":
            deadline = time.monotonic() + 20
            while not chip._poll_native_run(run):
                assert time.monotonic() < deadline, "poll did not retire the posted GET"
                time.sleep(0.001)
        chip._wait_native_run(run)
        chip._finalize_native_run(run)
        sample = _QueueSample()
        chip.copy_from(ctypes.addressof(sample), output_addr, ctypes.sizeof(sample))
        return sample

    before = execute(1)
    assert before.phase == 1 and before.registered == 0
    assert before.row == (1 - rank) * before.rows_per_peer + before.sender
    assert before.head == before.target_bb > 0 and before.submitted == before.target_cqe == 1
    assert (before.cqe_status >> 2) & 1 and before.cqe_status >> 16 == 0
    assert (before.sq_tail, before.cq_tail, before.doorbell) == (0, 0, 0)

    # This invocation reopens the same domain and only samples the original row.
    after = execute(2)
    assert after.phase == 2 and after.row == before.row
    assert (after.head, after.submitted) == (before.target_bb, before.target_cqe)
    retired = (after.sq_tail, after.cq_tail, after.doorbell)
    assert retired == (before.target_bb, before.target_cqe, before.target_cqe & 0xFFFFFF)

    chip.copy_from(ctypes.addressof(output), output_addr, ctypes.sizeof(output))
    assert output[header:] == [-77.0] * guard + payload(1 - rank) + [-77.0] * guard
    chip.free(output_addr)
    ready.wait()
    chip._impl.comm_release_domain_windows(comm, 1, 2, rank)
    chip.comm_destroy(comm)
    chip.finalize()
    print(
        f"rank={rank} {observe=} row={before.row} BB={before.target_bb} CQE={before.target_cqe} "
        f"tails/db=(0, 0, 0)->{retired}",
        flush=True,
    )


@pytest.mark.platforms(["a5"])
@pytest.mark.runtime("host_build_graph")
@pytest.mark.device_count(2)
@pytest.mark.parametrize("observe", ["wait", "poll"])
@scene_level(3)
def test_host_reaps_posted_urma_get(st_platform, st_device_ids, tmp_path, observe):
    spec = _callable("terminal_get")["callables"][0]
    spec["incores"][0]["source"] = str(Path(__file__).with_name("post_get.cpp"))
    target = compile_chip_callable_spec(
        spec, st_platform, "host_build_graph", (__file__, _pto_isa_compile_cache_token())
    )
    image = ctypes.string_at(target.buffer_ptr(), target.buffer_size())
    context = multiprocessing.get_context("spawn")
    ready = context.Barrier(2, timeout=60)
    processes = []
    try:
        for rank, device in enumerate(st_device_ids[:2]):
            process = context.Process(
                target=_run_rank, args=(st_platform, rank, device, str(tmp_path / "rootinfo"), ready, image, observe)
            )
            process.start()
            processes.append(process)
        for process in processes:
            process.join(120)
            assert process.exitcode == 0, f"URMA rank process {process.pid} exited with {process.exitcode}"
    finally:
        for process in processes:
            if process.is_alive():
                process.kill()
            process.join()
