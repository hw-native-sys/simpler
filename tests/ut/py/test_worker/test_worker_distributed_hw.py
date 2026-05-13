# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: PLC0415
"""Hardware smoke test for `Worker(comm_plan=...)` on 2 Ascend devices.

End-to-end equivalent of ``test_bootstrap_context_hw.py`` but driven through
the top-level ``Worker`` class so the bootstrap happens inside forked chip
children and the parent observes it via ``worker.chip_contexts``.

Deliberately no ``comm_barrier`` — that path still trips HCCL 507018 on
some CANN builds (tracked separately).  The non-barrier invariants are
enough to prove each chip's communicator is up and both ranks carved a
GVA-visible window.
"""

from __future__ import annotations

import pytest


@pytest.mark.requires_hardware
@pytest.mark.platforms(["a2a3"])
@pytest.mark.device_count(2)
def test_worker_chip_bootstrap(st_device_ids):
    from simpler.task_interface import ChipBufferSpec, CommDomain, CommDomainPlan
    from simpler.worker import Worker

    assert len(st_device_ids) >= 2, "device_count(2) fixture must yield >= 2 ids"
    device_ids = [int(st_device_ids[0]), int(st_device_ids[1])]
    nranks = len(device_ids)
    window_size = 4096
    buffer_nbytes = 64

    comm_plan = CommDomainPlan(
        domains=[
            CommDomain(
                name="default",
                worker_indices=list(range(nranks)),
                window_size=window_size,
                buffers=[
                    ChipBufferSpec(
                        name="x",
                        dtype="float32",
                        count=buffer_nbytes // 4,
                        nbytes=buffer_nbytes,
                    )
                ],
            )
        ]
    )

    worker = Worker(
        level=3,
        platform="a2a3",
        runtime="tensormap_and_ringbuffer",
        device_ids=device_ids,
        num_sub_workers=0,
        comm_plan=comm_plan,
    )
    try:
        worker.init()

        ctxs = worker.chip_contexts
        assert len(ctxs) == nranks
        for rank, ctx in enumerate(ctxs):
            assert ctx.device_id == device_ids[rank]
            domain = ctx.domains["default"]
            assert domain.domain_rank == rank
            assert domain.domain_size == nranks
            assert domain.device_ctx != 0, f"rank {rank}: device_ctx is 0 (HCCL alloc failed)"
            assert domain.local_window_base != 0, f"rank {rank}: local_window_base is 0"
            assert domain.actual_window_size >= window_size, (
                f"rank {rank}: actual_window_size={domain.actual_window_size} < requested {window_size}"
            )
            # The single buffer spec carves offset 0, matching the
            # ChipContext.buffer_ptrs → local_window_base invariant.
            assert domain.buffer_ptrs == {"x": domain.local_window_base}
    finally:
        worker.close()
