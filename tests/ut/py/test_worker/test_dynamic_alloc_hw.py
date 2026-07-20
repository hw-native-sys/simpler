# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: PLC0415
"""Hardware smoke test for cached `Orchestrator.acquire_domain` on HCCL.

Mirrors the happy-path case from `test_dynamic_alloc_sim.py` but drives the
real ``tensormap_and_ringbuffer`` runtime on Ascend.  The flow under test:

  1. Build an L3 ``Worker`` and call ``init()``.  No static comm_plan —
     base HCCL membership is established lazily on the first
     ``orch.acquire_domain`` call inside ``Worker.run``.
  2. Inside ``Worker.run(orch_fn)``, call ``orch.acquire_domain(...)``
     to drive ``comm_alloc_domain_windows`` (aclrtMalloc + IPC announce +
     SetImportPid + ImportByKey).
  3. Verify the returned per-chip ``ChipDomainContext`` carries a non-zero
     ``device_ctx`` + ``local_window_base`` on every participating chip.
  4. Run the same orchestration again and verify the allocation identity and
     addresses are reused while the window contents are reset to zero.
  5. ``Worker.close`` performs the actual ``comm_release_domain_windows``.

Deliberately no ``comm_barrier`` after alloc — HCCL 507018 surfaces only
on the explicit HcclBarrier path, not in our IPC alloc/release.
Cross-rank sync is already enforced by the internal file barriers inside
the C ABI.
"""

from __future__ import annotations

import ctypes
import os
from multiprocessing.shared_memory import SharedMemory

import pytest


@pytest.mark.requires_hardware
@pytest.mark.platforms(["a2a3", "a5"])
@pytest.mark.device_count(2)
def test_two_rank_cached_domain_reuse_and_reset(st_platform, st_device_ids):
    """End-to-end 2-rank hardware allocation reuse and reset round trip.

    Single allocation, one CommBufferSpec, both chips participate.  Locks
    in the same fields the sim happy-path test does, on real HCCL.
    """
    from simpler.task_interface import CallConfig, CommBufferSpec
    from simpler.worker import Worker

    from simpler_setup.runtime_builder import RuntimeBuilder

    build = bool(os.environ.get("PTO_UT_BUILD"))
    _ = RuntimeBuilder(platform=st_platform).get_binaries("tensormap_and_ringbuffer", build=build)
    assert len(st_device_ids) >= 2, "device_count(2) fixture must yield >= 2 ids"
    device_ids = [int(d) for d in st_device_ids[:2]]
    nranks = len(device_ids)

    captured: dict[str, object] = {"runs": []}
    shared = SharedMemory(create=True, size=16)
    shared_buf = shared.buf
    assert shared_buf is not None
    shared_buf[:] = b"\x5a" * 16
    exported = ctypes.c_char.from_buffer(shared_buf)
    shared_addr = ctypes.addressof(exported)

    def orch_fn(orch, _args, _cfg):
        with orch.acquire_domain(
            name="tp",
            workers=list(range(nranks)),
            window_size=4 * 1024 * 1024,
            buffers=[
                CommBufferSpec(name="scratch", dtype="float32", count=16, nbytes=64),
                CommBufferSpec(name="signal", dtype="uint32", count=4, nbytes=16),
            ],
        ) as tp:
            run = {
                "workers": tuple(tp.workers),
                "alloc_id": tp.allocation_id,
                "contexts": {
                    chip_idx: {
                        "domain_rank": tp[chip_idx].domain_rank,
                        "domain_size": tp[chip_idx].domain_size,
                        "device_ctx": int(tp[chip_idx].device_ctx),
                        "local_window_base": int(tp[chip_idx].local_window_base),
                        "buffer_ptrs": dict(tp[chip_idx].buffer_ptrs),
                    }
                    for chip_idx in tp.workers
                },
            }
            captured["runs"].append(run)  # type: ignore[union-attr]
            if len(captured["runs"]) == 1:  # type: ignore[arg-type]
                for chip_idx in tp.workers:
                    orch.copy_to(
                        chip_idx,
                        dst=tp[chip_idx].buffer_ptrs["signal"],
                        src=shared_addr,
                        size=16,
                    )
            else:
                orch.copy_from(
                    0,
                    dst=shared_addr,
                    src=tp[0].buffer_ptrs["signal"],
                    size=16,
                )

    worker = Worker(
        level=3,
        platform=st_platform,
        runtime="tensormap_and_ringbuffer",
        device_ids=device_ids,
        num_sub_workers=0,
    )
    try:
        worker.init()
        worker.run(orch_fn, args=None, config=CallConfig())
        worker.run(orch_fn, args=None, config=CallConfig())
        reset_bytes = bytes(shared_buf)
    finally:
        worker.close()
        del exported
        shared_buf.release()
        shared.close()
        shared.unlink()

    runs: list[dict[str, object]] = captured["runs"]  # type: ignore[assignment]
    assert len(runs) == 2
    assert runs[0]["workers"] == tuple(range(nranks))
    assert runs[1]["workers"] == tuple(range(nranks))
    assert runs[0]["alloc_id"] == runs[1]["alloc_id"]

    contexts: dict[int, dict[str, object]] = runs[0]["contexts"]  # type: ignore[assignment]
    reused_contexts: dict[int, dict[str, object]] = runs[1]["contexts"]  # type: ignore[assignment]
    # Dense domain ranks follow worker order.
    assert contexts[0]["domain_rank"] == 0
    assert contexts[1]["domain_rank"] == 1
    for chip_idx in range(nranks):
        ctx = contexts[chip_idx]
        # aclrtMalloc returned non-null for the CommContext + the per-rank pool.
        assert ctx["device_ctx"] != 0, f"chip {chip_idx}: device_ctx is 0"
        assert ctx["local_window_base"] != 0, f"chip {chip_idx}: local_window_base is 0"
        # Buffers carved sequentially from the local pool: scratch at base + 0,
        # signal at base + 64 (scratch.nbytes).
        ptrs = ctx["buffer_ptrs"]
        assert isinstance(ptrs, dict)
        assert ptrs["scratch"] == ctx["local_window_base"]
        assert ptrs["signal"] == ctx["local_window_base"] + 64
        assert reused_contexts[chip_idx]["device_ctx"] == ctx["device_ctx"]
        assert reused_contexts[chip_idx]["local_window_base"] == ctx["local_window_base"]

    assert reset_bytes == b"\x00" * 16
