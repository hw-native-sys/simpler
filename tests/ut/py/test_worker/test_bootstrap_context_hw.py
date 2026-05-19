# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: PLC0415
"""Hardware smoke test for ``ChipWorker.bootstrap_context``.

Drives the one-shot bring-up against the real ``tensormap_and_ringbuffer``
runtime on 2 Ascend devices.  The critical assertions are:

  1. ``bootstrap_context`` returns a non-null ``device_ctx`` and
     ``local_window_base`` (HCCL actually allocated GVA-visible windows).
  2. ``actual_window_size`` is at least the requested size.
  3. A single ``CommBufferSpec`` slices the window so
     ``buffer_ptrs[0] == local_window_base``.

Deliberately **no** ``comm_barrier``.  The paired ``comm_*`` UT
(``test_platform_comm.py``) already shows the known HCCL 507018 path fails
after ~52 s on some CANN builds; ``bootstrap_context`` does not issue a
barrier, so this test completes on any build.  Cross-rank synchronization
between the two ranks is already enforced inside
``HcclCommInitRootInfo`` / the root-info handshake that ``comm_init``
performs, so the non-barrier invariants above are enough to prove the
bring-up crossed both ranks.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import traceback

import pytest


def _bootstrap_rank_entry(  # noqa: PLR0913
    rank: int,
    nranks: int,
    device_id: int,
    bins,
    rootinfo_path: str,
    window_size: int,
    buffer_nbytes: int,
    result_queue: mp.Queue,  # type: ignore[type-arg]
) -> None:
    """Per-rank worker: drives bootstrap_context against HCCL and reports fields."""
    result: dict[str, object] = {"rank": rank, "stage": "start", "ok": False}
    try:
        from simpler.task_interface import (
            ChipBootstrapConfig,
            ChipWorker,
            CommBufferSpec,
            CommDomain,
            CommDomainPlan,
        )

        worker = ChipWorker()
        worker.init(device_id, bins)
        result["stage"] = "init"

        plan = CommDomainPlan(
            domains=[
                CommDomain(
                    name="default",
                    worker_indices=list(range(nranks)),
                    window_size=window_size,
                    buffers=[
                        CommBufferSpec(
                            name="x",
                            dtype="float32",
                            count=buffer_nbytes // 4,
                            nbytes=buffer_nbytes,
                        )
                    ],
                )
            ]
        )
        cfg = ChipBootstrapConfig(comm=plan.bootstrap_for_worker(rank))
        cfg.base_rank = rank
        cfg.base_size = nranks
        cfg.rootinfo_path = rootinfo_path
        cfg.base_window_size = plan.base_window_size()

        res = worker.bootstrap_context(device_id=device_id, cfg=cfg)
        domain = res.domains["default"]
        result["stage"] = "bootstrap"
        result["device_ctx"] = int(domain.device_ctx)
        result["local_window_base"] = int(domain.local_window_base)
        result["actual_window_size"] = int(domain.actual_window_size)
        result["buffer_ptrs"] = list(domain.buffer_ptrs.values())

        # Teardown mirrors the Worker bootstrap loop ordering: shutdown_bootstrap
        # (releases the HCCL comm handle) then finalize (releases ACL / unloads
        # runtime).
        worker.shutdown_bootstrap()
        worker.finalize()
        result["ok"] = True
    except Exception:  # noqa: BLE001
        result["error"] = traceback.format_exc()
    finally:
        result_queue.put(result)


@pytest.mark.requires_hardware
@pytest.mark.platforms(["a2a3"])
@pytest.mark.device_count(2)
def test_two_rank_bootstrap_context(st_device_ids):
    """End-to-end 2-rank hardware bootstrap_context smoke test.

    No barrier is issued — see the module docstring for why that dodges
    HCCL 507018.  The test still gates on every field ``bootstrap_context``
    is supposed to populate.
    """
    from simpler_setup.runtime_builder import RuntimeBuilder

    build = bool(os.environ.get("PTO_UT_BUILD"))
    bins = RuntimeBuilder(platform="a2a3").get_binaries("tensormap_and_ringbuffer", build=build)
    assert len(st_device_ids) >= 2, "device_count(2) fixture must yield >= 2 ids"
    nranks = 2
    rootinfo_path = f"/tmp/pto_bootstrap_hw_rootinfo_{os.getpid()}.bin"
    window_size = 4096
    buffer_nbytes = 64

    ctx = mp.get_context("fork")
    result_queue: mp.Queue = ctx.Queue()  # type: ignore[type-arg]
    procs = []
    for rank in range(nranks):
        p = ctx.Process(
            target=_bootstrap_rank_entry,
            args=(
                rank,
                nranks,
                int(st_device_ids[rank]),
                bins,
                rootinfo_path,
                window_size,
                buffer_nbytes,
                result_queue,
            ),
            daemon=False,
        )
        p.start()
        procs.append(p)

    results: dict[int, dict] = {}
    for _ in range(nranks):
        r = result_queue.get(timeout=180)
        results[int(r["rank"])] = r
    for p in procs:
        p.join(timeout=60)

    try:
        os.unlink(rootinfo_path)
    except FileNotFoundError:
        pass

    for rank in range(nranks):
        r = results.get(rank)
        if r is None:
            pytest.fail(f"rank {rank} never reported a result")
        if not r.get("ok"):
            pytest.fail(f"rank {rank} failed at {r.get('stage')!r}:\n{r.get('error', '(no traceback)')}")

        assert r["device_ctx"] != 0, f"rank {rank}: device_ctx is 0"
        assert r["local_window_base"] != 0, f"rank {rank}: local_window_base is 0"
        assert r["actual_window_size"] >= window_size, (
            f"rank {rank}: actual_window_size={r['actual_window_size']} < requested {window_size}"
        )
        # 1:1 buffer-to-spec invariant — the contract ChipContext relies on.
        assert r["buffer_ptrs"] == [r["local_window_base"]], (
            f"rank {rank}: buffer_ptrs={r['buffer_ptrs']} != [{r['local_window_base']}]"
        )


# ---------------------------------------------------------------------------
# Multi-domain hardware path — mirrors the sim integration test in
# test_bootstrap_context_sim.py::TestBootstrapContextMultiDomain.  3 ranks
# in two overlapping domains (`tp = [0, 1]`, `pp = [1, 2]`); chip 1
# participates in both.  Pins the wire format + base-window slicing on
# the HCCL backend so a CANN bump or `comm_derive_context` regression
# fails CI instead of only the manual `domain_rank_map` / `dual_domain_overlap`
# examples.
# ---------------------------------------------------------------------------


def _multi_domain_rank_entry(
    worker_idx: int,
    nranks: int,
    device_id: int,
    bins,
    rootinfo_path: str,
    result_queue: mp.Queue,  # type: ignore[type-arg]
) -> None:
    """Per-rank entry: declare two overlapping domains, report what bootstrap built.

    The plan is identical across ranks (same `CommDomainPlan` shape) — each
    rank's `bootstrap_for_worker(worker_idx)` selects only the domains it
    actually participates in, so chip 0 sees `{tp}`, chip 2 sees `{pp}`,
    and chip 1 sees both.  No barrier is issued (mirrors the single-domain
    test for the same HCCL 507018 reason).
    """
    result: dict[str, object] = {"rank": worker_idx, "ok": False}
    try:
        from simpler.task_interface import (
            ChipBootstrapConfig,
            ChipWorker,
            CommBufferSpec,
            CommDomain,
            CommDomainPlan,
        )

        plan = CommDomainPlan(
            domains=[
                CommDomain(
                    name="tp",
                    worker_indices=[0, 1],
                    window_size=4096,
                    buffers=[CommBufferSpec(name="scratch", dtype="float32", count=16, nbytes=64)],
                ),
                CommDomain(
                    name="pp",
                    worker_indices=[1, 2],
                    window_size=4096,
                    buffers=[CommBufferSpec(name="scratch", dtype="float32", count=16, nbytes=64)],
                ),
            ]
        )
        cfg = ChipBootstrapConfig(comm=plan.bootstrap_for_worker(worker_idx))
        cfg.base_rank = worker_idx
        cfg.base_size = nranks
        cfg.rootinfo_path = rootinfo_path

        worker = ChipWorker()
        worker.init(device_id, bins)
        try:
            res = worker.bootstrap_context(device_id=device_id, cfg=cfg)
            result["domains"] = {
                name: {
                    "domain_rank": domain.domain_rank,
                    "domain_size": domain.domain_size,
                    "device_ctx": int(domain.device_ctx),
                    "local_window_base": int(domain.local_window_base),
                    "actual_window_size": int(domain.actual_window_size),
                    "buffer_ptrs": dict(domain.buffer_ptrs),
                }
                for name, domain in res.domains.items()
            }
            result["ok"] = True
        finally:
            worker.shutdown_bootstrap()
            worker.finalize()
    except Exception:  # noqa: BLE001
        result["error"] = traceback.format_exc()
    finally:
        result_queue.put(result)


@pytest.mark.requires_hardware
@pytest.mark.platforms(["a2a3"])
@pytest.mark.device_count(3)
def test_overlapping_domains_create_independent_hw_windows(st_device_ids):
    """3 ranks, two overlapping domains; chip 1 gets independent views of both.

    Locks in on real hardware what the sim test already locks in: each domain
    materialises its own ``device_ctx`` and its own ``buffer_ptrs`` slice,
    even when the same chip participates in multiple domains, and
    ``comm_derive_context`` produces non-overlapping local addresses for the
    two domain windows.
    """
    from simpler_setup.runtime_builder import RuntimeBuilder

    build = bool(os.environ.get("PTO_UT_BUILD"))
    bins = RuntimeBuilder(platform="a2a3").get_binaries("tensormap_and_ringbuffer", build=build)
    assert len(st_device_ids) >= 3, "device_count(3) fixture must yield >= 3 ids"
    nranks = 3
    rootinfo_path = f"/tmp/pto_bootstrap_hw_multi_domain_{os.getpid()}.bin"

    ctx = mp.get_context("fork")
    result_queue: mp.Queue = ctx.Queue()  # type: ignore[type-arg]
    procs = []
    for worker_idx in range(nranks):
        p = ctx.Process(
            target=_multi_domain_rank_entry,
            args=(
                worker_idx,
                nranks,
                int(st_device_ids[worker_idx]),
                bins,
                rootinfo_path,
                result_queue,
            ),
            daemon=False,
        )
        p.start()
        procs.append(p)

    results: dict[int, dict] = {}
    for _ in range(nranks):
        r = result_queue.get(timeout=240)
        results[int(r["rank"])] = r
    for p in procs:
        p.join(timeout=60)

    try:
        os.unlink(rootinfo_path)
    except FileNotFoundError:
        pass

    for rank in range(nranks):
        r = results.get(rank)
        if r is None:
            pytest.fail(f"rank {rank} never reported a result")
        if not r.get("ok"):
            pytest.fail(f"rank {rank} failed:\n{r.get('error', '(no traceback)')}")

    # Membership: chip 0 in tp only, chip 2 in pp only, chip 1 in both.
    assert set(results[0]["domains"]) == {"tp"}, results[0]["domains"]
    assert set(results[1]["domains"]) == {"tp", "pp"}, results[1]["domains"]
    assert set(results[2]["domains"]) == {"pp"}, results[2]["domains"]

    # Dense domain ranks follow `worker_indices` order.
    assert results[0]["domains"]["tp"]["domain_rank"] == 0
    assert results[1]["domains"]["tp"]["domain_rank"] == 1
    assert results[1]["domains"]["pp"]["domain_rank"] == 0
    assert results[2]["domains"]["pp"]["domain_rank"] == 1

    # Chip 1's two domains must be independent: distinct device CommContext
    # pointers and distinct local buffer addresses, even though both live in
    # the same base symmetric pool.
    chip1 = results[1]["domains"]
    assert chip1["tp"]["device_ctx"] != chip1["pp"]["device_ctx"], chip1
    assert chip1["tp"]["buffer_ptrs"]["scratch"] != chip1["pp"]["buffer_ptrs"]["scratch"], chip1
    # And the buffer addresses fall inside their respective domain windows.
    for name in ("tp", "pp"):
        d = chip1[name]
        assert d["local_window_base"] <= d["buffer_ptrs"]["scratch"] < d["local_window_base"] + d["actual_window_size"]
