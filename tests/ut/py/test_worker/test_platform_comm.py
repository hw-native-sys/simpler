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

    ChipWorker.init(device_id) → comm_init → comm_alloc_windows
    → comm_get_local_window_base → comm_get_window_size
    → copy_from (reads back CommContext) → comm_barrier (known-issue tolerant)
    → comm_destroy → finalize

ACL bring-up and aclrtStream lifetime are owned internally by
`ChipWorker.comm_init` / `comm_destroy` (matching the L2-boundary contract
in docs/hierarchical-level-runtime.md — device-side state stays in C++).
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
import signal
import sys
import time
import traceback
import warnings
from contextlib import contextmanager
from pathlib import Path

import pytest
from simpler.task_interface import _CommContextStruct as _CommContext

# ---------------------------------------------------------------------------
# CommContext layout — must stay byte-compatible with
# src/common/platform_comm/comm_context.h (static_asserts there).
# simpler owns this layout; its producers, kernels and Python mirrors are
# rebuilt together when the shared ABI changes.
# ---------------------------------------------------------------------------
assert ctypes.sizeof(_CommContext) == 1104, "CommContext python mirror drifted from C++ header"


@contextmanager
def _rank_phase(result, device_id, rootinfo_path, phase):
    """Flush a rank's native-call boundary before entering code that may stall."""
    result["stage"] = phase
    context = (
        f"[comm-lifecycle pid={os.getpid()} rank={result['rank']} "
        f"device={device_id} phase={phase} rootinfo={rootinfo_path}]"
    )
    print(f"{context} before", file=sys.stderr, flush=True)
    try:
        yield
    except BaseException:
        print(f"{context} failed", file=sys.stderr, flush=True)
        raise
    else:
        print(f"{context} after", file=sys.stderr, flush=True)


def _rank_entry(
    rank: int,
    nranks: int,
    device_id: int,
    bins,
    rootinfo_path: str,
    result_queue: mp.Queue,  # type: ignore[type-arg]
) -> None:
    """Worker-process body: runs comm_* lifecycle for one rank and reports results."""
    result: dict[str, object] = {"rank": rank, "stage": "start", "ok": False}
    try:
        from simpler.task_interface import ChipWorker

        with _rank_phase(result, device_id, rootinfo_path, "create"):
            worker = ChipWorker()
        with _rank_phase(result, device_id, rootinfo_path, "init"):
            worker.init(device_id, bins)

        # ChipWorker.comm_init owns ACL bring-up and aclrtStream creation
        # internally — Python never touches aclInit / aclrtSetDevice /
        # aclrtCreateStream.  This matches the L2-boundary contract in
        # docs/hierarchical-level-runtime.md: device-side lifecycle stays
        # in C++.
        with _rank_phase(result, device_id, rootinfo_path, "comm_init"):
            comm = worker.comm_init(rank, nranks, rootinfo_path)

        # 4 KiB is the same window hint the C++ UT uses.  HCCL may round this
        # up; we cross-check the returned winSize against comm_get_window_size.
        with _rank_phase(result, device_id, rootinfo_path, "alloc"):
            device_ctx_ptr = worker.comm_alloc_windows(comm, 4096)
        if device_ctx_ptr == 0:
            raise RuntimeError("comm_alloc_windows returned null device_ctx")
        result["stage"] = "alloc"

        with _rank_phase(result, device_id, rootinfo_path, "query_base"):
            local_base = worker.comm_get_local_window_base(comm)
        with _rank_phase(result, device_id, rootinfo_path, "query_size"):
            win_size = worker.comm_get_window_size(comm)
        if local_base == 0:
            raise RuntimeError("comm_get_local_window_base returned 0")
        if win_size < 4096:
            raise RuntimeError(f"comm_get_window_size={win_size} < 4096")
        result["stage"] = "query"

        # ABI guard: copy the CommContext published by simpler back to host
        # and verify every field we consume in kernels matches what we asked
        # for.  This is the Python twin of EXIT_CTX_FIELDS from the C++ UT.
        # worker.copy_from() is the ChipWorker's device-to-host DMA; we hand
        # it the host address of a ctypes-backed _CommContext buffer.
        host_ctx = _CommContext()
        with _rank_phase(result, device_id, rootinfo_path, "memcpy"):
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
            with _rank_phase(result, device_id, rootinfo_path, "barrier"):
                worker.comm_barrier(comm)
            result["barrier_ok"] = True
        except Exception as barrier_exc:  # noqa: BLE001
            result["barrier_ok"] = False
            result["barrier_error"] = str(barrier_exc)

        with _rank_phase(result, device_id, rootinfo_path, "destroy"):
            worker.comm_destroy(comm)

        with _rank_phase(result, device_id, rootinfo_path, "finalize"):
            worker.finalize()
        result["ok"] = True
    except Exception:  # noqa: BLE001
        result["error"] = traceback.format_exc()
    finally:
        result_queue.put(result)


def _rank_bootstrap(parent_channel, child_channel, args):
    """Publish this child's identity before any native/device operation."""
    parent_channel.close()
    try:
        child_channel.send(os.getpid())
        if child_channel.recv() is not True:
            return
    except EOFError:
        return
    finally:
        child_channel.close()
    _rank_entry(*args)


class _RankProcess:
    """Keep startup identity reachable even before Process publishes its Popen."""

    def __init__(self, ctx, args):
        self.parent_channel, self.child_channel = ctx.Pipe()
        try:
            self.process = ctx.Process(
                target=_rank_bootstrap,
                args=(self.parent_channel, self.child_channel, args),
                daemon=False,
            )
        except BaseException:
            self.parent_channel.close()
            self.child_channel.close()
            raise
        self.start_attempted = False
        self.startup_pid = None
        self.unpublished_exitcode = None

    def _read_startup_pid(self):
        if not self.parent_channel.poll(5):
            raise RuntimeError("rank startup PID is unconfirmed")
        pid = self.parent_channel.recv()
        if type(pid) is not int or pid <= 0:
            raise RuntimeError("rank startup reported an invalid PID")
        self.startup_pid = pid

    def start(self):
        self.start_attempted = True
        self.process.start()
        self.child_channel.close()
        self._read_startup_pid()
        if self.startup_pid != self.process.pid:
            raise RuntimeError("rank startup PID differs from Process.pid")
        self.parent_channel.send(True)
        self.parent_channel.close()

    def recover_startup(self):
        try:
            # The parent's child-end closes even when start() raises before fork.
            self.child_channel.close()
            if self.process.pid is None and self.start_attempted and self.startup_pid is None:
                try:
                    self._read_startup_pid()
                except EOFError as exc:
                    raise RuntimeError("rank startup ended without a confirmed PID") from exc
        finally:
            self.parent_channel.close()

    @property
    def pid(self):
        return self.process.pid if self.process.pid is not None else self.startup_pid

    @property
    def exitcode(self):
        return self.process.exitcode if self.process.pid is not None else self.unpublished_exitcode

    def is_alive(self):
        if self.process.pid is not None:
            return self.process.is_alive()
        if self.unpublished_exitcode is not None:
            return False
        if self.startup_pid is None:
            raise RuntimeError("rank startup has no confirmed child identity")
        # A private pipe establishes the claimed PID; waitpid additionally proves
        # this exact child still belongs to this parent before any signal.
        pid, status = os.waitpid(self.startup_pid, os.WNOHANG)
        if pid == 0:
            return True
        self.unpublished_exitcode = os.waitstatus_to_exitcode(status)
        return False

    def join(self, timeout):
        if self.process.pid is not None:
            self.process.join(timeout)
            return
        deadline = time.monotonic() + timeout
        while self.is_alive():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(remaining, 0.01))

    def _signal_unpublished(self, sig):
        if self.is_alive():
            pid = self.startup_pid
            if pid is None:
                raise RuntimeError("rank startup has no confirmed signal target")
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass

    def terminate(self):
        if self.process.pid is not None:
            self.process.terminate()
        else:
            self._signal_unpublished(signal.SIGTERM)

    def kill(self):
        if self.process.pid is not None:
            self.process.kill()
        else:
            self._signal_unpublished(signal.SIGKILL)


def _reap_rank_processes(procs):
    """Signal all owned children before bounded waits and survivor kills."""
    cleanup_errors = []
    started = []
    reaped = True
    for process in procs:
        try:
            process.recover_startup()
            if process.pid is not None:
                started.append(process)
        except BaseException as exc:  # noqa: BLE001
            reaped = False
            cleanup_errors.append(f"rank startup ownership is unconfirmed: {exc!r}")
    # Signal every owned child before waiting on any one of them.
    for process in started:
        try:
            if process.is_alive():
                process.terminate()
        except BaseException as exc:  # noqa: BLE001 -- cleanup must visit the other ranks
            cleanup_errors.append(f"terminate pid={process.pid}: {exc!r}")
    for process in started:
        try:
            process.join(timeout=5)
        except BaseException as exc:  # noqa: BLE001
            cleanup_errors.append(f"join pid={process.pid}: {exc!r}")
    for process in started:
        try:
            if process.is_alive():
                process.kill()
        except BaseException as exc:  # noqa: BLE001
            cleanup_errors.append(f"kill pid={process.pid}: {exc!r}")
    for process in started:
        try:
            process.join(timeout=5)
            if process.is_alive() or process.exitcode is None:
                reaped = False
                cleanup_errors.append(f"child is not reaped: pid={process.pid}")
        except BaseException as exc:  # noqa: BLE001
            reaped = False
            cleanup_errors.append(f"reap pid={process.pid}: {exc!r}")
    return reaped, cleanup_errors


def _run_rank_processes(bins, device_ids, rootinfo_path):
    """Own rank subprocesses until every started child has been reaped.

    Process exit is a test containment boundary, not proof of device drain.
    """
    ctx = mp.get_context("fork")
    result_queue = ctx.Queue()
    procs = []
    try:
        for rank, device_id in enumerate(device_ids):
            process = _RankProcess(ctx, (rank, len(device_ids), int(device_id), bins, rootinfo_path, result_queue))
            # start() can create a child before raising or being interrupted.
            procs.append(process)
            process.start()

        # The parent never puts on this queue, so it owns no feeder thread.
        results_by_rank = {}
        for _ in device_ids:
            result = result_queue.get(timeout=180)
            results_by_rank[int(result["rank"])] = result

        for rank in range(len(device_ids)):
            if rank not in results_by_rank:
                pytest.fail(f"rank {rank} never reported a result")
            r = results_by_rank[rank]
            if not r.get("ok"):
                pytest.fail(f"rank {rank} failed at stage {r.get('stage')!r}:\n{r.get('error', '(no traceback)')}")

        for process in procs:
            process.join(timeout=60)
        for rank, process in enumerate(procs):
            alive = process.is_alive()
            if alive or process.exitcode != 0:
                reason = "reported a result but did not exit" if alive else f"exited with code {process.exitcode}"
                raise RuntimeError(f"rank {rank} {reason} (pid={process.pid})")
        return results_by_rank
    finally:
        primary_error = sys.exc_info()[1]
        reaped, cleanup_errors = _reap_rank_processes(procs)
        if reaped:
            try:
                result_queue.close()
            except BaseException as exc:  # noqa: BLE001
                cleanup_errors.append(f"rank result queue cleanup: {exc!r}")
            try:
                Path(rootinfo_path).unlink(missing_ok=True)
            except BaseException as exc:  # noqa: BLE001
                cleanup_errors.append(f"rank rootinfo cleanup: {exc!r}")
        if cleanup_errors:
            message = "rank process cleanup failed: " + "; ".join(cleanup_errors)
            if primary_error is not None:
                print(message, file=sys.stderr, flush=True)
            else:
                raise RuntimeError(message)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.requires_hardware
@pytest.mark.platforms(["a2a3", "a5"])
@pytest.mark.device_count(2)
def test_two_rank_comm_lifecycle(st_platform, st_device_ids, tmp_path):
    """End-to-end 2-rank hardware smoke test for ChipWorker.comm_* wrappers."""
    from simpler_setup.runtime_builder import RuntimeBuilder

    build = bool(os.environ.get("PTO_UT_BUILD"))
    bins = RuntimeBuilder(platform=st_platform).get_binaries("tensormap_and_ringbuffer", build=build)
    assert len(st_device_ids) >= 2, "device_count(2) fixture must yield >= 2 ids"
    nranks = 2
    rootinfo_path = str(tmp_path / "rootinfo.bin")

    results_by_rank = _run_rank_processes(bins, st_device_ids[:nranks], rootinfo_path)

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
