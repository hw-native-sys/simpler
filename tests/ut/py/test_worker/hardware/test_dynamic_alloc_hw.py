# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: PLC0415
"""Hardware smoke test for `Orchestrator.allocate_domain` on HCCL.

Mirrors the happy-path case from `test_dynamic_alloc_sim.py` but drives the
real ``tensormap_and_ringbuffer`` runtime on Ascend.  The flow under test:

  1. Build an L3 ``Worker`` and call ``init()``.  No static comm_plan —
     base HCCL membership is established lazily on the first
     ``orch.allocate_domain`` call inside ``Worker.run``.
  2. Inside ``Worker.run(orch_fn)``, call ``orch.allocate_domain(...)``
     to drive ``comm_alloc_domain_windows`` (VMM allocation + Fabric V2
     handle exchange and import).
  3. Verify the returned per-chip ``ChipDomainContext`` carries a non-zero
     ``device_ctx`` + ``local_window_base`` on every participating chip.
  4. Exit the ``with`` block to mark the handle for release; the actual
     ``comm_release_domain_windows`` runs after Worker.run drains.

Deliberately no ``comm_barrier`` after alloc — HCCL 507018 surfaces only
on the explicit HcclBarrier path, not in our Fabric alloc/release.
Cross-rank sync is already enforced by the internal file barriers inside
the C ABI.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.mark.requires_hardware
@pytest.mark.platforms(["a2a3", "a5"])
@pytest.mark.device_count(2)
@pytest.mark.parametrize("runtime", ["tensormap_and_ringbuffer", "host_build_graph"])
def test_two_rank_allocate_release_round_trip(st_platform, st_device_ids, runtime):
    """End-to-end 2-rank hardware alloc + release round trip.

    Both onboard backends perform eight sequential allocations on one base
    communicator and retire each allocation before the next Worker.run.
    """
    from simpler.task_interface import CallConfig, CommBufferSpec
    from simpler.worker import Worker

    from simpler_setup.runtime_builder import RuntimeBuilder

    build = bool(os.environ.get("PTO_UT_BUILD"))
    _ = RuntimeBuilder(platform=st_platform).get_binaries(runtime, build=build)
    assert len(st_device_ids) >= 2, "device_count(2) fixture must yield >= 2 ids"
    device_ids = [int(d) for d in st_device_ids[:2]]
    nranks = len(device_ids)

    captures: list[dict[str, object]] = []
    marker_counts: list[int] = []

    def orch_fn(orch, _args, _cfg):
        captured: dict[str, object] = {}
        with orch.allocate_domain(
            name="tp",
            workers=list(range(nranks)),
            window_size=4 * 1024 * 1024,
            buffers=[
                CommBufferSpec(name="scratch", dtype="float32", count=16, nbytes=64),
                CommBufferSpec(name="signal", dtype="uint32", count=4, nbytes=64),
            ],
        ) as tp:
            rootinfo = Path(worker._comm_plan_rootinfo_path())
            announcements = list(rootinfo.parent.glob(f"barrier_{rootinfo.name}_alloc_{tp.allocation_id}_*announce*"))
            assert len(announcements) == nranks
            backend = "fabric" if "_fabric_announce_" in announcements[0].name else "ipc"
            captured["backend"] = backend
            print(f"runtime={runtime} backend={backend} allocation_id={tp.allocation_id}", flush=True)
            captured["workers"] = tuple(tp.workers)
            captured["alloc_id"] = tp.allocation_id
            captured["contexts"] = {
                chip_idx: {
                    "domain_rank": tp[chip_idx].domain_rank,
                    "domain_size": tp[chip_idx].domain_size,
                    "device_ctx": int(tp[chip_idx].device_ctx),
                    "local_window_base": int(tp[chip_idx].local_window_base),
                    "buffer_bases": {name: h.base for name, h in tp[chip_idx].buffers.items()},
                }
                for chip_idx in tp.workers
            }
        captured["released_after_with"] = tp.released
        captures.append(captured)

    worker = Worker(
        level=3,
        platform=st_platform,
        runtime=runtime,
        device_ids=device_ids,
        num_sub_workers=0,
    )
    try:
        worker.init()
        repetitions = 8
        for _ in range(repetitions):
            worker.run(orch_fn, args=None, config=CallConfig())
            root = Path(worker._comm_plan_rootinfo_path())
            count = len(list(root.parent.glob(f"barrier_{root.name}_alloc_*")))
            marker_counts.append(count)
            print(f"allocation_round={len(captures)} residual_markers={count}", flush=True)
    finally:
        worker.close()

    assert marker_counts == [0] * repetitions
    if st_platform in ("a2a3", "a5"):
        assert not root.parent.exists(), "closed Worker must reclaim its managed generation"
    assert len(captures) == repetitions
    assert len({captured["alloc_id"] for captured in captures}) == repetitions
    for captured in captures:
        assert captured["released_after_with"] is True
        assert captured["workers"] == tuple(range(nranks))

        contexts: dict[int, dict[str, object]] = captured["contexts"]  # type: ignore[assignment]
        # Dense domain ranks follow worker order.
        assert contexts[0]["domain_rank"] == 0
        assert contexts[1]["domain_rank"] == 1
        for chip_idx in range(nranks):
            ctx = contexts[chip_idx]
            assert ctx["device_ctx"] != 0, f"chip {chip_idx}: device_ctx is 0"
            assert ctx["local_window_base"] != 0, f"chip {chip_idx}: local_window_base is 0"
            # Buffers are carved sequentially from the local pool.
            ptrs = ctx["buffer_bases"]
            assert isinstance(ptrs, dict)
            assert ptrs["scratch"] == ctx["local_window_base"]
            assert ptrs["signal"] == ctx["local_window_base"] + 64
