# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""CCU writes real allocated W bytes to the peer, with dynamic ranges and tokens."""

import struct
from pathlib import Path

import pytest
from simpler.task_interface import CommBufferSpec, DataType, TaskArgs, TensorArgType
from simpler.worker import Worker

from tests.st.a5.tensormap_and_ringbuffer.ccu_native_domain.test_ccu_native_domain import _build_callable


@pytest.mark.platforms(["a5"])
@pytest.mark.runtime("tensormap_and_ringbuffer")
@pytest.mark.device_count(2)
@pytest.mark.parametrize("workspace_engines", [("ccu",), ("ccu", "urma")])
def test_ccu_real_window_transfer(st_platform, st_device_ids, tmp_path, request, workspace_engines):
    _exercise_window_transfer(
        st_platform, st_device_ids[:2], tmp_path, request, (0, 1), workspace_engines=workspace_engines
    )


@pytest.mark.platforms(["a5"])
@pytest.mark.runtime("tensormap_and_ringbuffer")
@pytest.mark.device_count(3)
@pytest.mark.parametrize("members", [(0, 1, 2), (2, 0)])
def test_ccu_all_peer_channels_and_subset_rank_mapping(st_platform, st_device_ids, tmp_path, request, members):
    _exercise_window_transfer(st_platform, st_device_ids[:3], tmp_path, request, members)


@pytest.mark.platforms(["a5"])
@pytest.mark.runtime("tensormap_and_ringbuffer")
@pytest.mark.device_count(3)
def test_ccu_shared_context_across_distinct_window_members(st_platform, st_device_ids, tmp_path, request):
    _exercise_window_transfer(st_platform, st_device_ids[:3], tmp_path, request, (0, 1), member_sets=((0, 1), (0, 2)))


@pytest.mark.platforms(["a5"])
@pytest.mark.runtime("tensormap_and_ringbuffer")
@pytest.mark.device_count(2)
def test_ccu_partial_and_exit_only_requests_preserve_window(st_platform, st_device_ids, tmp_path, request):
    _exercise_window_transfer(st_platform, st_device_ids[:2], tmp_path, request, (0, 1), masks=(1, 2, 0, 3))


def _exercise_window_transfer(
    st_platform,
    devices,
    tmp_path,
    request,
    members,
    runtime="tensormap_and_ringbuffer",
    member_sets=None,
    masks=(3,),
    workspace_engines=("ccu",),
):
    rounds = request.config.getoption("--rounds")
    assert rounds > 0
    member_sets = member_sets or (members,)
    rounds = max(2 * len(member_sets), len(masks), rounds)
    target = _build_callable(tmp_path, source_root=Path(__file__).resolve().parent, runtime=runtime)
    worker = Worker(
        level=3,
        platform=st_platform,
        runtime=runtime,
        device_ids=[int(d) for d in devices],
        num_sub_workers=0,
    )
    retained = {}
    allocations = []
    registrations = {}
    try:
        chip = worker.register(target)
        worker.init()
        outputs = [[worker.create_buffer(64), worker.create_buffer(64), worker.create_buffer(32768)] for _ in members]
        for generation in range(1, rounds + 1):
            selection = (generation - 1) % len(member_sets)
            members = member_sets[selection]
            name = f"ccu_w_{selection}"
            mask = masks[(generation - 1) % len(masks)]
            payloads = [
                bytes((i * 17 + rank * 73 + generation * 11) % 256 for i in range(16384)) + bytes([0xA5]) * 16384
                for rank in range(len(members))
            ]

            def run(orch, _args, config):
                domain = orch.get_or_allocate_domain(
                    name=name,
                    members=list(members),
                    window_size=4 * 1024 * 1024,
                    buffers=[CommBufferSpec("data", "uint64", 4096, 32768)],
                    workspace_engines=workspace_engines,
                    retain_after_run=True,
                )
                if name in retained:
                    assert retained[name] is domain
                else:
                    retained[name] = domain
                    allocations.append(domain.allocation_id)
                for rank in range(len(members)):
                    orch.copy_to_domain(domain, rank, payloads[rank], buffer="data")
                bindings = {
                    chip_id: tuple(domain.get_or_register_ccu(chip, ccu_id=i, worker=chip_id) for i in range(2))
                    for chip_id in members
                }
                for rank, chip_id in enumerate(members):
                    args = TaskArgs()
                    for index, output in enumerate(outputs[rank]):
                        args.add_tensor(
                            output.tensor((4096 if index == 2 else 8,), DataType.UINT64), TensorArgType.OUTPUT_EXISTING
                        )
                    args.add_scalar(domain.device_context(chip_id, workspace_engines=("ccu",)))
                    args.add_scalar(generation | ((mask + 1) << 32))
                    orch.submit_next_level(chip, args, config, worker=chip_id, ccu_bindings=bindings[chip_id])

            worker.run(run)
            domain = retained[name]
            bindings = {
                chip_id: tuple(domain.get_or_register_ccu(chip, ccu_id=i, worker=chip_id) for i in range(2))
                for chip_id in members
            }
            key = domain.allocation_id
            if key in registrations:
                assert registrations[key] == bindings
            registrations[key] = bindings
            displacement = (generation % 8) * 64
            length = ((generation % 16) + 1) * 64
            for rank in range(len(members)):
                for index in range(2):
                    if (mask & (1 << (1 - index))) == 0:
                        continue
                    audit = struct.unpack("8Q", bytes(outputs[rank][index].shm.buf))
                    assert audit[4:] == (
                        length,
                        (rank - 1) % len(members),
                        (rank + 1) % len(members),
                        1,
                    )
                expected = bytearray(payloads[rank])
                for ccu_id in range(2):
                    if (mask & (1 << ccu_id)) == 0:
                        continue
                    src = ccu_id * 8192 + displacement
                    dst = 16384 + src
                    expected[dst : dst + length] = payloads[(rank + 1) % len(members)][src : src + length]
                actual = bytes(outputs[rank][2].shm.buf)
                assert actual == expected, f"W mismatch round={generation} rank={rank}"
            if generation in (1, rounds) or generation % 100 == 0:
                print(f"CCU W round={generation}/{rounds}: bidirectional bytes and untouched ranges passed", flush=True)
            if generation == rounds // 2:
                released = retained.pop("ccu_w_0")
                released.release()
                assert released.freed
        for domain in retained.values():
            domain.release()
            assert domain.freed
        assert len(allocations) == len(member_sets) + 1 and len(set(allocations)) == len(allocations)
        assert len(registrations) == len(allocations)
        print("CCU W retained reuse, release and reallocation passed; domain-owned native registrations", flush=True)
    finally:
        worker.close()
