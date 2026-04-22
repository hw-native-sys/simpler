# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: PLC0415
"""Hardware UT for ChipWorker.comm_* wrappers (Python surface of the HCCL backend).

This is the Python twin of tests/ut/cpp/test_hccl_comm.cpp.  It drives the
full comm lifecycle entirely through ChipWorker's public Python API:

    ChipWorker.init → set_device → comm_init → comm_alloc_windows
    → comm_get_local_window_base → comm_get_window_size
    → copy_from (reads back CommContext) → comm_barrier (known-issue tolerant)
    → comm_destroy → finalize

ACL bring-up and aclrtStream lifetime are owned internally by
`ChipWorker.comm_init` / `comm_destroy` (matching the L2-boundary contract
in docs/hierarchical_level_runtime.md — device-side state stays in C++).
ctypes is used only to declare a `CommContext` layout mirror so the test
can inspect the struct returned by `comm_alloc_windows`; no CANN / libacl
symbols are loaded from Python.

Each rank runs in a forked subprocess so HCCL sees a distinct device context
per rank.  The parent only waits on exit codes plus a small result queue used
to surface CommContext field values.

Known issue inherited from the HCCL backend (HCCL 507018): on certain CANN builds
`HcclBarrier` + `aclrtSynchronizeStream` report 507018 after ~52s of timeout.
That is a CANN-coupling bug tracked separately; this test treats a barrier
failure as a warning and still asserts the non-barrier invariants (init/alloc
/ctx-fields/destroy) succeeded.
"""

from __future__ import annotations

import ctypes
import multiprocessing as mp
import os
import traceback
import warnings

import pytest

# ---------------------------------------------------------------------------
# CommContext layout — must stay byte-compatible with
# src/a2a3/platform/include/common/comm_context.h (static_asserts there).
# If CANN / HCCL ever shifts these offsets, comm_hccl.cpp's build-time asserts
# will fail first; this struct mirrors them so the Python side can read back a
# CommContext without rebuilding nanobind just to expose the layout.
# ---------------------------------------------------------------------------
_COMM_MAX_RANK_NUM = 64


class _CommContext(ctypes.Structure):
    _fields_ = [
        ("workSpace", ctypes.c_uint64),
        ("workSpaceSize", ctypes.c_uint64),
        ("rankId", ctypes.c_uint32),
        ("rankNum", ctypes.c_uint32),
        ("winSize", ctypes.c_uint64),
        ("windowsIn", ctypes.c_uint64 * _COMM_MAX_RANK_NUM),
        ("windowsOut", ctypes.c_uint64 * _COMM_MAX_RANK_NUM),
    ]


assert ctypes.sizeof(_CommContext) == 1056, "CommContext python mirror drifted from C++ header"


def _rank_entry(
    rank: int,
    nranks: int,
    device_id: int,
    host_lib: str,
    aicpu_path: str,
    aicore_path: str,
    sim_context_path: str,
    rootinfo_path: str,
    result_queue: mp.Queue,  # type: ignore[type-arg]
) -> None:
    """Worker-process body: runs comm_* lifecycle for one rank and reports results."""
    result: dict[str, object] = {"rank": rank, "stage": "start", "ok": False}
    try:
        from simpler.task_interface import ChipWorker

        worker = ChipWorker()
        worker.init(host_lib, aicpu_path, aicore_path, sim_context_path)
        result["stage"] = "init"

        worker.set_device(device_id)
        result["stage"] = "set_device"

        # ChipWorker.comm_init owns ACL bring-up and aclrtStream creation
        # internally — Python never touches aclInit / aclrtSetDevice /
        # aclrtCreateStream.  This matches the L2-boundary contract in
        # docs/hierarchical_level_runtime.md: device-side lifecycle stays
        # in C++.
        comm = worker.comm_init(rank, nranks, rootinfo_path)
        result["stage"] = "comm_init"

        # 4 KiB is the same window hint the C++ UT uses.  HCCL may round this
        # up; we cross-check the returned winSize against comm_get_window_size.
        device_ctx_ptr = worker.comm_alloc_windows(comm, 4096)
        if device_ctx_ptr == 0:
            raise RuntimeError("comm_alloc_windows returned null device_ctx")
        result["stage"] = "alloc"

        local_base = worker.comm_get_local_window_base(comm)
        win_size = worker.comm_get_window_size(comm)
        if local_base == 0:
            raise RuntimeError("comm_get_local_window_base returned 0")
        if win_size < 4096:
            raise RuntimeError(f"comm_get_window_size={win_size} < 4096")
        result["stage"] = "query"

        # ABI guard: copy the CommContext that HCCL populates back to host
        # and verify every field we consume in kernels matches what we asked
        # for.  This is the Python twin of EXIT_CTX_FIELDS from the C++ UT.
        # worker.copy_from() is the ChipWorker's device-to-host DMA; we hand
        # it the host address of a ctypes-backed _CommContext buffer.
        host_ctx = _CommContext()
        worker.copy_from(ctypes.addressof(host_ctx), device_ctx_ptr, ctypes.sizeof(host_ctx))
        result["stage"] = "memcpy"

        if host_ctx.rankId != rank:
            raise AssertionError(f"rankId={host_ctx.rankId}, expected {rank}")
        if host_ctx.rankNum != nranks:
            raise AssertionError(f"rankNum={host_ctx.rankNum}, expected {nranks}")
        if host_ctx.winSize != win_size:
            raise AssertionError(f"winSize={host_ctx.winSize}, expected {win_size}")
        if host_ctx.windowsIn[rank] != local_base:
            raise AssertionError(f"windowsIn[{rank}]=0x{host_ctx.windowsIn[rank]:x} != local_base=0x{local_base:x}")
        peer_windows = [int(host_ctx.windowsIn[i]) for i in range(nranks)]
        if any(w == 0 for w in peer_windows):
            raise AssertionError(f"peer windowsIn contains zero: {peer_windows}")
        result["stage"] = "ctx_fields_ok"
        result["peer_windows"] = peer_windows
        result["win_size"] = int(win_size)
        result["local_base"] = int(local_base)
        result["rank_id"] = int(host_ctx.rankId)
        result["rank_num"] = int(host_ctx.rankNum)

        # Barrier.  The C++ HCCL UT observed CANN error 507018 here on some
        # builds; that bug is tracked independently.  Surface the failure to
        # the parent as a warning and continue with teardown so the
        # non-barrier invariants above still gate this test.
        try:
            worker.comm_barrier(comm)
            result["barrier_ok"] = True
        except Exception as barrier_exc:  # noqa: BLE001
            result["barrier_ok"] = False
            result["barrier_error"] = str(barrier_exc)

        worker.comm_destroy(comm)
        result["stage"] = "destroy"

        worker.finalize()
        result["stage"] = "finalize"
        result["ok"] = True
    except Exception:  # noqa: BLE001
        result["error"] = traceback.format_exc()
    finally:
        result_queue.put(result)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.requires_hardware
@pytest.mark.platforms(["a2a3"])
@pytest.mark.device_count(2)
def test_two_rank_comm_lifecycle(st_device_ids):
    """End-to-end 2-rank hardware smoke test for ChipWorker.comm_* wrappers."""
    from simpler_setup.runtime_builder import RuntimeBuilder

    build = bool(os.environ.get("PTO_UT_BUILD"))
    bins = RuntimeBuilder(platform="a2a3").get_binaries("tensormap_and_ringbuffer", build=build)
    host_lib = str(bins.host_path)
    aicpu_path = str(bins.aicpu_path)
    aicore_path = str(bins.aicore_path)
    sim_context_path = str(bins.sim_context_path) if bins.sim_context_path else ""

    assert len(st_device_ids) >= 2, "device_count(2) fixture must yield >= 2 ids"
    nranks = 2
    rootinfo_path = f"/tmp/pto_comm_py_ut_rootinfo_{os.getpid()}.bin"

    ctx = mp.get_context("fork")
    result_queue: mp.Queue = ctx.Queue()  # type: ignore[type-arg]
    procs = []
    for rank in range(nranks):
        p = ctx.Process(
            target=_rank_entry,
            args=(
                rank,
                nranks,
                int(st_device_ids[rank]),
                host_lib,
                aicpu_path,
                aicore_path,
                sim_context_path,
                rootinfo_path,
                result_queue,
            ),
            daemon=False,
        )
        p.start()
        procs.append(p)

    # Drain the queue before joining — workers block on queue.put if the pipe
    # buffer fills, so pulling N results first avoids a fork+deadlock.
    results_by_rank: dict[int, dict] = {}
    for _ in range(nranks):
        r = result_queue.get(timeout=180)
        results_by_rank[int(r["rank"])] = r

    for p in procs:
        p.join(timeout=60)

    try:
        os.unlink(rootinfo_path)
    except FileNotFoundError:
        pass

    for rank in range(nranks):
        if rank not in results_by_rank:
            pytest.fail(f"rank {rank} never reported a result")
        r = results_by_rank[rank]
        if not r.get("ok"):
            pytest.fail(f"rank {rank} failed at stage {r.get('stage')!r}:\n{r.get('error', '(no traceback)')}")

    # Each rank's own-slot invariant (windowsIn[rank] == local_base) is
    # asserted inside _rank_entry; all peer slots are already checked to be
    # non-zero there.  We deliberately do NOT assert cross-rank address
    # agreement at this layer: under HCCL, windowsIn[i] holds the *current
    # rank's device-local view* of peer i's window (HBM pointer resolved via
    # remoteRes / MESH remapping), which is not required to numerically
    # equal peer i's own local_base.  The C++ hardware UT makes the same
    # weaker check for the same reason — anything stricter would fail on
    # every well-formed HCCL deployment.

    # Barrier is allowed to fail under the known 507018 regression; emit a
    # warning instead of failing the test.  The non-barrier invariants above
    # are the load-bearing assertions here.
    barrier_failures = [r for r in range(nranks) if not results_by_rank[r].get("barrier_ok")]
    if barrier_failures:
        msgs = "; ".join(f"rank {r}: {results_by_rank[r].get('barrier_error', '?')}" for r in barrier_failures)
        warnings.warn(
            f"comm_barrier failed on {len(barrier_failures)}/{nranks} ranks (known issue 507018): {msgs}",
            stacklevel=1,
        )
