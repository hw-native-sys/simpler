# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: PLC0415
"""Sim-backend integration tests for dynamic CommDomain allocation.

Exercises the new ``Orchestrator.allocate_domain`` / ``release_domain`` path
end-to-end on the sim backend (POSIX shm + atomic counters; no real hardware).

The flow under test:

  1. ``Worker(comm_plan=...)`` declares one minimal "membership" domain so
     init runs the HCCL/sim ``comm_init`` handshake.  All chip children sit
     idle in their main loop after bootstrap.
  2. Inside ``Worker.run(orch_fn)``, ``orch.allocate_domain(...)`` issues
     CTRL_ALLOC_DOMAIN to each participating chip in parallel; each chip
     calls ``comm_alloc_domain_windows`` (sim: shm_open + ftruncate +
     ready_count barrier) and returns its ``(device_ctx, local_window_base,
     buffer_ptrs)`` via a reply shm.
  3. The orch function gets a ``CommDomainHandle`` whose
     ``contexts[chip_idx]`` exposes per-chip context.  ``release_domain``
     (or context-manager exit) drives CTRL_RELEASE_DOMAIN with a matching
     destroy barrier.
"""

from __future__ import annotations

import pytest


def _sim_binaries():
    """Resolve pre-built a2a3sim runtime binaries, or skip if unavailable."""
    from simpler_setup.runtime_builder import RuntimeBuilder

    try:
        bins = RuntimeBuilder(platform="a2a3sim").get_binaries("tensormap_and_ringbuffer")
    except FileNotFoundError as e:
        pytest.skip(f"a2a3sim runtime binaries unavailable: {e}")
    return bins


def _make_worker(nranks: int):
    """Build an L3 sim Worker.  No static `comm_plan` — base communicator is
    established lazily on the first ``orch.allocate_domain`` call.
    """
    from simpler.worker import Worker

    bins = _sim_binaries()
    # Sim binaries reuse the same RuntimeBinaries object; the Worker pulls
    # paths from there via RuntimeBuilder under the hood.  Fetched here only
    # to trigger the skip on absence.
    _ = bins
    return Worker(
        level=3,
        platform="a2a3sim",
        runtime="tensormap_and_ringbuffer",
        device_ids=list(range(nranks)),
        num_sub_workers=0,
    )


# ---------------------------------------------------------------------------
# 1. Round-trip: allocate, see independent per-chip contexts, release.
# ---------------------------------------------------------------------------


class TestDynamicAllocateBasic:
    def test_allocate_returns_distinct_per_chip_contexts(self):
        from simpler.task_interface import CallConfig, CommBufferSpec

        captured: dict[str, object] = {}

        def orch_fn(orch, _args, _cfg):
            tp = orch.allocate_domain(
                name="tp",
                workers=[0, 1],
                window_size=4096,
                buffers=[CommBufferSpec(name="scratch", dtype="float32", count=16, nbytes=64)],
            )
            try:
                captured["name"] = tp.name
                captured["workers"] = tuple(tp.workers)
                captured["alloc_id"] = tp.allocation_id
                captured["contexts"] = {
                    chip_idx: {
                        "domain_rank": tp[chip_idx].domain_rank,
                        "domain_size": tp[chip_idx].domain_size,
                        "device_ctx": int(tp[chip_idx].device_ctx),
                        "local_window_base": int(tp[chip_idx].local_window_base),
                        "buffer_ptrs": dict(tp[chip_idx].buffer_ptrs),
                    }
                    for chip_idx in tp.workers
                }
            finally:
                orch.release_domain(tp)
            captured["released"] = tp.released

        worker = _make_worker(nranks=2)
        try:
            worker.init()
            worker.run(orch_fn, args=None, config=CallConfig())
        finally:
            worker.close()

        assert captured["name"] == "tp"
        assert captured["workers"] == (0, 1)
        assert captured["released"] is True

        # Dense domain ranks follow workers order.
        contexts: dict[int, dict[str, object]] = captured["contexts"]  # type: ignore[assignment]
        assert contexts[0]["domain_rank"] == 0
        assert contexts[1]["domain_rank"] == 1
        assert contexts[0]["domain_size"] == 2

        # Each chip got its own per-allocation CommContext object on host.
        # Cross-chip device_ctx values are NOT comparable on the sim backend
        # because they're heap addresses in different processes (and HCCL's
        # GVA-style equality is not a sim invariant — see the existing
        # comm_sim.cpp "Cross-process addressing contract" comment).  We
        # check non-zero and intra-chip uniqueness across allocations
        # elsewhere.
        ctx0 = contexts[0]
        ctx1 = contexts[1]
        assert ctx0["device_ctx"] != 0
        assert ctx1["device_ctx"] != 0

        # Each rank's local_window_base equals its scratch buffer (1-buffer
        # carve), and the addresses are inside the freshly mmap'd region
        # (i.e. non-zero).
        assert ctx0["local_window_base"] != 0
        assert ctx1["local_window_base"] != 0
        scratch0 = ctx0["buffer_ptrs"]
        scratch1 = ctx1["buffer_ptrs"]
        assert isinstance(scratch0, dict) and scratch0["scratch"] == ctx0["local_window_base"]
        assert isinstance(scratch1, dict) and scratch1["scratch"] == ctx1["local_window_base"]


# ---------------------------------------------------------------------------
# 2. with-statement releases on exit.
# ---------------------------------------------------------------------------


class TestDynamicAllocateContextManager:
    def test_with_releases_on_exit(self):
        from simpler.task_interface import CallConfig, CommBufferSpec

        captured: dict[str, object] = {}

        def orch_fn(orch, _args, _cfg):
            with orch.allocate_domain(
                name="tp",
                workers=[0, 1],
                window_size=4096,
                buffers=[CommBufferSpec(name="scratch", dtype="float32", count=4, nbytes=16)],
            ) as tp:
                captured["alive_inside_with"] = tp.released
                captured["device_ctx_0"] = int(tp[0].device_ctx)
            captured["released_after_with"] = tp.released

        worker = _make_worker(nranks=2)
        try:
            worker.init()
            worker.run(orch_fn, args=None, config=CallConfig())
        finally:
            worker.close()

        assert captured["alive_inside_with"] is False
        assert captured["released_after_with"] is True
        assert captured["device_ctx_0"] != 0


# ---------------------------------------------------------------------------
# 3. Re-alloc with the same name after release succeeds (canonical "resize"
#    pattern: release, then allocate a bigger one under the same name).
# ---------------------------------------------------------------------------


class TestDynamicAllocateReuseName:
    def test_alloc_after_release_reuses_name(self):
        from simpler.task_interface import CallConfig

        alloc_ids: list[int] = []
        ctx0_pairs: list[tuple[int, int]] = []  # (device_ctx, local_window_base) per allocation

        def orch_fn(orch, _args, _cfg):
            a = orch.allocate_domain(name="tp", workers=[0, 1], window_size=4096)
            alloc_ids.append(a.allocation_id)
            ctx0_pairs.append((int(a[0].device_ctx), int(a[0].local_window_base)))
            a.release()

            b = orch.allocate_domain(name="tp", workers=[0, 1], window_size=4096)
            alloc_ids.append(b.allocation_id)
            ctx0_pairs.append((int(b[0].device_ctx), int(b[0].local_window_base)))
            b.release()

        worker = _make_worker(nranks=2)
        try:
            worker.init()
            worker.run(orch_fn, args=None, config=CallConfig())
        finally:
            worker.close()

        # The Worker's monotonic counter assigns distinct allocation_ids,
        # which scope IPC handshake filenames so the second alloc cannot
        # collide with stale state from the first.
        assert alloc_ids[0] != alloc_ids[1], alloc_ids
        # Both reuses produced valid (non-zero) pointers; addresses may or
        # may not coincide depending on malloc/mmap behaviour, which is not
        # part of the contract — just that the alloc actually succeeded.
        for device_ctx, base in ctx0_pairs:
            assert device_ctx != 0 and base != 0


# ---------------------------------------------------------------------------
# 4. Duplicate name while live raises.
# ---------------------------------------------------------------------------


class TestDynamicAllocateDuplicateName:
    def test_alloc_same_name_while_live_raises(self):
        from simpler.task_interface import CallConfig

        captured: dict[str, object] = {}

        def orch_fn(orch, _args, _cfg):
            handle = orch.allocate_domain(name="tp", workers=[0, 1], window_size=4096)
            try:
                with pytest.raises(ValueError, match="already live"):
                    orch.allocate_domain(name="tp", workers=[0, 1], window_size=4096)
                captured["reached_assertion"] = True
            finally:
                handle.release()

        worker = _make_worker(nranks=2)
        try:
            worker.init()
            worker.run(orch_fn, args=None, config=CallConfig())
        finally:
            worker.close()

        assert captured.get("reached_assertion") is True


# ---------------------------------------------------------------------------
# 5. orch_fn raises mid-DAG → live domains get auto-released by Worker.run.
# ---------------------------------------------------------------------------


class TestDynamicAllocateExceptionUnwind:
    def test_unhandled_exception_releases_live_domains(self):
        from simpler.task_interface import CallConfig

        sentinel = RuntimeError("orch boom")

        def orch_fn(orch, _args, _cfg):
            _ = orch.allocate_domain(name="tp", workers=[0, 1], window_size=4096)
            # Intentionally do NOT release.  Worker.run's finally should
            # call _release_all_live_domains before propagating.
            raise sentinel

        worker = _make_worker(nranks=2)
        try:
            worker.init()
            with pytest.raises(RuntimeError) as ei:
                worker.run(orch_fn, args=None, config=CallConfig())
            assert ei.value is sentinel
            # After the failed run, no live handle should remain.  Otherwise
            # the next Worker.run would see a name collision.
            assert worker.live_domains == {}
        finally:
            worker.close()


# ---------------------------------------------------------------------------
# 6. Interleaved / overlapping subsets — multiple concurrently-live domains
#    over non-contiguous worker subsets.  Each allocation has its own
#    `allocation_id`, which scopes the IPC handshake / barrier filenames,
#    so concurrent allocations over disjoint subsets must not collide.
# ---------------------------------------------------------------------------


class TestDynamicAllocateInterleavedSubsets:
    def test_disjoint_subsets_concurrent_alive(self):
        """Two simultaneously-live domains over disjoint interleaved subsets.

        Chip 4 ranks: 0,1,2,3.  Allocate `even` over [0, 2] and `odd` over
        [1, 3] inside the same orch_fn, both live at the same time.  Pins
        the contract that overlapping allocation_ids scope handshakes
        correctly across non-contiguous subsets.
        """
        from simpler.task_interface import CallConfig

        captured: dict[str, object] = {}

        def orch_fn(orch, _args, _cfg):
            even = orch.allocate_domain(name="even", workers=[0, 2], window_size=4096)
            odd = orch.allocate_domain(name="odd", workers=[1, 3], window_size=4096)
            try:
                captured["even_workers"] = even.workers
                captured["odd_workers"] = odd.workers
                captured["even_alloc_id"] = even.allocation_id
                captured["odd_alloc_id"] = odd.allocation_id
                # Dense domain ranks follow the user's worker order, not the
                # base-comm rank.  Chip 2 is `even`'s domain_rank 1; chip 3
                # is `odd`'s domain_rank 1.
                captured["even_chip0_rank"] = even[0].domain_rank
                captured["even_chip2_rank"] = even[2].domain_rank
                captured["odd_chip1_rank"] = odd[1].domain_rank
                captured["odd_chip3_rank"] = odd[3].domain_rank
                # Per-allocation device_ctx + local_window_base are non-zero
                # for every participating rank.
                captured["even_ctxs_nonzero"] = all(
                    even[w].device_ctx != 0 and even[w].local_window_base != 0 for w in even.workers
                )
                captured["odd_ctxs_nonzero"] = all(
                    odd[w].device_ctx != 0 and odd[w].local_window_base != 0 for w in odd.workers
                )
            finally:
                odd.release()
                even.release()

        worker = _make_worker(nranks=4)
        try:
            worker.init()
            worker.run(orch_fn, args=None, config=CallConfig())
        finally:
            worker.close()

        assert captured["even_workers"] == (0, 2)
        assert captured["odd_workers"] == (1, 3)
        assert captured["even_alloc_id"] != captured["odd_alloc_id"]
        assert captured["even_chip0_rank"] == 0 and captured["even_chip2_rank"] == 1
        assert captured["odd_chip1_rank"] == 0 and captured["odd_chip3_rank"] == 1
        assert captured["even_ctxs_nonzero"] is True
        assert captured["odd_ctxs_nonzero"] is True

    def test_overlapping_subsets_serial(self):
        """Two domains that overlap on one chip, allocated one at a time.

        Chip 4 ranks: 0,1,2,3.  Domain `left` over [0, 1, 2] and `right`
        over [1, 2, 3]; chip 1 and chip 2 participate in both.  Verifies
        that the same chip can hold multiple per-allocation records
        keyed by `allocation_id` and produce independent contexts.
        """
        from simpler.task_interface import CallConfig

        captured: dict[str, object] = {}

        def orch_fn(orch, _args, _cfg):
            left = orch.allocate_domain(name="left", workers=[0, 1, 2], window_size=4096)
            right = orch.allocate_domain(name="right", workers=[1, 2, 3], window_size=4096)
            try:
                # On the chips that participate in both, the two contexts
                # must be independent: different device_ctx + different
                # local_window_base (each allocation has its own shm).
                for chip_idx in (1, 2):
                    captured[f"chip{chip_idx}_left_ctx"] = int(left[chip_idx].device_ctx)
                    captured[f"chip{chip_idx}_right_ctx"] = int(right[chip_idx].device_ctx)
                    captured[f"chip{chip_idx}_left_base"] = int(left[chip_idx].local_window_base)
                    captured[f"chip{chip_idx}_right_base"] = int(right[chip_idx].local_window_base)
                # Dense domain ranks differ between the two views of the same chip.
                captured["chip1_left_rank"] = left[1].domain_rank
                captured["chip1_right_rank"] = right[1].domain_rank
            finally:
                right.release()
                left.release()

        worker = _make_worker(nranks=4)
        try:
            worker.init()
            worker.run(orch_fn, args=None, config=CallConfig())
        finally:
            worker.close()

        # Chip 1: domain_rank=1 in `left=[0,1,2]`, domain_rank=0 in `right=[1,2,3]`
        assert captured["chip1_left_rank"] == 1
        assert captured["chip1_right_rank"] == 0
        for chip_idx in (1, 2):
            # Distinct allocations → distinct local window bases (different shms).
            assert captured[f"chip{chip_idx}_left_base"] != captured[f"chip{chip_idx}_right_base"]
