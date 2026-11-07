#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Release one retained URMA domain while another continues transferring on the same Worker."""

import struct

import pytest
from simpler.task_interface import CommBufferSpec, DataType, TaskArgs, TensorArgType
from simpler.worker import Worker

from tests.st.a5.tensormap_and_ringbuffer.urma_deferred_completion.test_urma_deferred_completion import (
    NRANKS,
    SMALL_ELEMS,
    URMA_DATA_OFFSET_NBYTES,
    _UrmaRegression,
)

pytestmark = [
    pytest.mark.runtime_product("program"),
    pytest.mark.execution_mode("eager"),
]

_NBYTES = SMALL_ELEMS * 4
_RUNTIME = "tensormap_and_ringbuffer"


@pytest.mark.platforms(["a5"])
@pytest.mark.runtime(_RUNTIME)
@pytest.mark.device_count(NRANKS)
def test_retained_urma_domain_release_preserves_other_domain(st_platform, st_device_ids):
    compiled = _UrmaRegression._compile_l3_callables(st_platform)
    worker = Worker(
        level=3,
        platform=st_platform,
        runtime=_RUNTIME,
        device_ids=[int(device) for device in st_device_ids[:NRANKS]],
        num_sub_workers=0,
    )
    chip = worker.register(compiled["urma_deferred_completion"])
    domains = {}

    def allocate(orch, name):
        return orch.get_or_allocate_domain(
            name=name,
            members=list(range(NRANKS)),
            workspace_engines=("urma",),
            window_size=4 * 1024 * 1024,
            buffers=[
                CommBufferSpec("urma_reserved", "int32", 64, URMA_DATA_OFFSET_NBYTES),
                CommBufferSpec("input_window", "float32", SMALL_ELEMS, _NBYTES),
                CommBufferSpec("receive_window", "float32", SMALL_ELEMS, _NBYTES),
            ],
            retain_after_run=True,
        )

    def transfer(orch, domain, cfg, generation):
        for rank in range(NRANKS):
            value = float(generation * 10 + rank)
            orch.copy_to_domain(
                domain,
                rank,
                struct.pack(f"<{SMALL_ELEMS}f", *([value] * SMALL_ELEMS)),
                buffer="input_window",
            )
        for rank in range(NRANKS):
            member = domain.local_context(rank)
            args = TaskArgs()
            args.add_tensor(
                member.buffers["input_window"].tensor((SMALL_ELEMS,), DataType.FLOAT32), TensorArgType.INPUT
            )
            args.add_tensor(
                member.buffers["receive_window"].tensor((SMALL_ELEMS,), DataType.FLOAT32), TensorArgType.INOUT
            )
            for output in (outputs[rank], results[rank]):
                args.add_tensor(output.tensor((SMALL_ELEMS,), DataType.FLOAT32), TensorArgType.OUTPUT_EXISTING)
            for value in (domain.device_context(rank, workspace_engines=("urma",)), SMALL_ELEMS, 0, 1):
                args.add_scalar(value)
            orch.submit_next_level(chip, args, cfg, worker=rank)

    def verify(generation):
        for rank in range(NRANKS):
            peer_value = float(generation * 10 + (1 - rank))
            assert struct.unpack_from(f"<{SMALL_ELEMS}f", outputs[rank].shm.buf) == (peer_value,) * SMALL_ELEMS
            assert struct.unpack_from(f"<{SMALL_ELEMS}f", results[rank].shm.buf) == (peer_value + 1.0,) * SMALL_ELEMS

    try:
        worker.init()
        outputs = [worker.create_buffer(_NBYTES) for _ in range(NRANKS)]
        results = [worker.create_buffer(_NBYTES) for _ in range(NRANKS)]

        def first(orch, _args, cfg):
            domains["a"] = allocate(orch, "lifecycle_a")
            domains["b"] = allocate(orch, "lifecycle_b")
            transfer(orch, domains["a"], cfg, 1)

        worker.run(first)
        verify(1)
        a, b = domains["a"], domains["b"]
        assert not a.freed and not b.freed
        a.release()
        assert a.released and a.freed
        assert not b.released and not b.freed

        def second(orch, _args, cfg):
            assert orch.get_domain("lifecycle_b") is b
            transfer(orch, b, cfg, 2)

        worker.run(second)
        verify(2)
        assert not b.freed

        def third(orch, _args, cfg):
            domains["fresh_a"] = allocate(orch, "lifecycle_a")
            transfer(orch, domains["fresh_a"], cfg, 3)

        worker.run(third)
        verify(3)
        fresh_a = domains["fresh_a"]
        assert fresh_a.domain_id != a.domain_id
        fresh_a.release()
        assert fresh_a.freed and not b.freed
        b.release()
        assert b.freed and worker.live_domains == {}
    finally:
        worker.close()
