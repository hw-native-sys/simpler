#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exercise transfers and independent release in two overlapping URMA domains."""

import struct

import pytest
from simpler.task_interface import CommBufferSpec, DataType, TaskArgs, TensorArgType
from simpler.worker import Worker

from tests.st.a5.tensormap_and_ringbuffer.urma_deferred_completion.test_urma_deferred_completion import (
    SMALL_ELEMS,
    URMA_DATA_OFFSET_NBYTES,
    _UrmaRegression,
)

_NBYTES = SMALL_ELEMS * 4
_RUNTIME = "tensormap_and_ringbuffer"
_DEVICE_COUNT = 3


def _payload(value):
    return struct.pack("<f", value) * SMALL_ELEMS


@pytest.mark.platforms(["a5"])
@pytest.mark.runtime(_RUNTIME)
@pytest.mark.device_count(_DEVICE_COUNT)
@pytest.mark.parametrize("bc_members", [(1, 2), (2, 1)], ids=["subset", "reordered"])
def test_retained_urma_domain_release_preserves_other_domain(st_platform, st_device_ids, bc_members):
    compiled = _UrmaRegression._compile_l3_callables(st_platform)
    worker = Worker(
        level=3,
        platform=st_platform,
        runtime=_RUNTIME,
        device_ids=[int(device) for device in st_device_ids[:_DEVICE_COUNT]],
        num_sub_workers=0,
    )
    chip = worker.register(compiled["urma_deferred_completion"])
    members = {"ab": (0, 1), "bc": bc_members}
    domains = {}
    receive_windows = {}
    sequence = 0

    def create(name):
        def stage(orch, _args, _cfg):
            domain = orch.get_or_allocate_domain(
                name=name,
                members=members[name],
                workspace_engines=("urma",),
                window_size=4 * 1024 * 1024,
                buffers=[
                    CommBufferSpec("urma_reserved", "int32", 64, URMA_DATA_OFFSET_NBYTES),
                    CommBufferSpec("input_window", "float32", SMALL_ELEMS, _NBYTES),
                    CommBufferSpec("receive_window", "float32", SMALL_ELEMS, _NBYTES),
                ],
                retain_after_run=True,
            )
            domains[name] = domain
            sentinel_base = -10 if name == "ab" else -20
            receive_windows[name] = tuple(_payload(sentinel_base - worker_id) for worker_id in members[name])
            for rank, worker_id in enumerate(members[name]):
                assert domain.member(rank).local_worker_id == worker_id
                assert domain.local_context(rank) is domain[worker_id]
                assert domain[worker_id].domain_rank == rank
                orch.copy_to_domain(domain, rank, receive_windows[name][rank], buffer="receive_window")

        worker.run(stage)

    def check_windows(orch, _args, _cfg):
        for name, domain in domains.items():
            for rank, expected in enumerate(receive_windows[name]):
                assert orch.copy_from_domain(domain, rank, _NBYTES, buffer="receive_window") == expected

    def transfer(name, mode):
        nonlocal sequence
        sequence += 1
        domain = domains[name]
        values = tuple(sequence * 10 + worker_id for worker_id in members[name])

        def launch(orch, _args, cfg):
            assert orch.get_domain(name) is domain
            for rank, value in enumerate(values):
                orch.copy_to_domain(domain, rank, _payload(value), buffer="input_window")
            for rank, worker_id in enumerate(members[name]):
                member = domain.local_context(rank)
                args = TaskArgs()
                args.add_tensor(
                    member.buffers["input_window"].tensor((SMALL_ELEMS,), DataType.FLOAT32), TensorArgType.INPUT
                )
                args.add_tensor(
                    member.buffers["receive_window"].tensor((SMALL_ELEMS,), DataType.FLOAT32), TensorArgType.INOUT
                )
                for output in (outputs[worker_id], results[worker_id]):
                    args.add_tensor(output.tensor((SMALL_ELEMS,), DataType.FLOAT32), TensorArgType.OUTPUT_EXISTING)
                for value in (domain.device_context(worker_id, workspace_engines=("urma",)), SMALL_ELEMS, mode, 1):
                    args.add_scalar(value)
                orch.submit_next_level(chip, args, cfg, worker=worker_id)

        worker.run(launch)
        for worker_id, expected in zip(members[name], values if mode else values[::-1]):
            assert outputs[worker_id].shm.buf[:_NBYTES] == _payload(expected)
            assert results[worker_id].shm.buf[:_NBYTES] == _payload(expected + 1)
        if mode:
            receive_windows[name] = tuple(_payload(value) for value in reversed(values))
        worker.run(check_windows)

    try:
        worker.init()
        outputs = [worker.create_buffer(_NBYTES) for _ in range(_DEVICE_COUNT)]
        results = [worker.create_buffer(_NBYTES) for _ in range(_DEVICE_COUNT)]
        create("ab")
        create("bc")
        for mode in (0, 1):
            transfer("ab", mode)
            transfer("bc", mode)

        for released, survivor in (("ab", "bc"), ("bc", "ab")):
            previous = domains.pop(released)
            previous.release()
            assert previous.released and previous.freed
            assert not domains[survivor].released and not domains[survivor].freed
            transfer(survivor, 0)
            create(released)
            assert domains[released].domain_id != previous.domain_id
            transfer(released, 1)
            transfer(survivor, 0)

        for domain in domains.values():
            domain.release()
            assert domain.freed
        assert worker.live_domains == {}
    finally:
        worker.close()
