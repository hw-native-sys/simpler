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
  3. A single ``ChipBufferSpec`` slices the window so
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
    host_lib: str,
    aicpu_path: str,
    aicore_path: str,
    sim_context_path: str,
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
            ChipBufferSpec,
            ChipCommBootstrapConfig,
            ChipWorker,
        )

        worker = ChipWorker()
        worker.init(host_lib, aicpu_path, aicore_path, sim_context_path)
        result["stage"] = "init"

        cfg = ChipBootstrapConfig(
            comm=ChipCommBootstrapConfig(
                rank=rank,
                nranks=nranks,
                rootinfo_path=rootinfo_path,
                window_size=window_size,
            ),
            buffers=[
                ChipBufferSpec(
                    name="x",
                    dtype="float32",
                    count=buffer_nbytes // 4,
                    placement="window",
                    nbytes=buffer_nbytes,
                )
            ],
        )

        res = worker.bootstrap_context(device_id=device_id, cfg=cfg)
        result["stage"] = "bootstrap"
        result["device_ctx"] = int(res.device_ctx)
        result["local_window_base"] = int(res.local_window_base)
        result["actual_window_size"] = int(res.actual_window_size)
        result["buffer_ptrs"] = list(res.buffer_ptrs)

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
    host_lib = str(bins.host_path)
    aicpu_path = str(bins.aicpu_path)
    aicore_path = str(bins.aicore_path)
    sim_context_path = str(bins.sim_context_path) if bins.sim_context_path else ""

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
                host_lib,
                aicpu_path,
                aicore_path,
                sim_context_path,
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
