# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: PLC0415
"""Simulation-backend tests for ``ChipWorker.bootstrap_context``.

These tests run without any Ascend NPU.  They drive the sim backend of the
``tensormap_and_ringbuffer`` runtime, whose ``comm_*`` lifecycle is backed by
POSIX shared memory + atomic counters.  The sim ``comm_alloc_windows`` has an
internal ready-count barrier: **all** ``nranks`` must call it before any
return.  So anything that exercises the communicator path is written as a
2-process fork with a small mp.Queue used to report results back to the test
runner.

The error-path case is deliberately single-process — it triggers a validation
error that raises *before* any communicator work, so no peer rank is needed.
"""

from __future__ import annotations

import ctypes
import multiprocessing as mp
import os
import struct
import traceback
from multiprocessing.shared_memory import SharedMemory

import pytest


def _shm_addr(shm: SharedMemory) -> int:
    """Return the raw address of a SharedMemory region (asserts buf is mapped)."""
    buf = shm.buf
    assert buf is not None
    return ctypes.addressof(ctypes.c_char.from_buffer(buf))


def _sim_binaries():
    """Resolve pre-built a2a3sim runtime binaries, or skip if unavailable.

    Respects ``PTO_UT_BUILD=1`` for local runs where the binaries have not
    been compiled yet — matches the pattern in ``test_platform_comm.py``.
    """
    from simpler_setup.runtime_builder import RuntimeBuilder

    build = bool(os.environ.get("PTO_UT_BUILD"))
    try:
        bins = RuntimeBuilder(platform="a2a3sim").get_binaries("tensormap_and_ringbuffer", build=build)
    except FileNotFoundError as e:
        pytest.skip(f"a2a3sim runtime binaries unavailable: {e}")
    return bins


def _rank_entry(  # noqa: PLR0913
    rank: int,
    nranks: int,
    rootinfo_path: str,
    window_size: int,
    host_lib: str,
    aicpu_path: str,
    aicore_path: str,
    simpler_log_path: str,
    sim_context_path: str,
    buffer_specs: list[dict],
    host_input_specs: list[dict],
    channel_shm_name: str | None,
    result_queue: mp.Queue,  # type: ignore[type-arg]
    readback_nbytes: int,
) -> None:
    """Forked-rank body: init ChipWorker, run bootstrap_context, report fields.

    ``buffer_specs`` / ``host_input_specs`` are plain dicts (picklable) that
    the child converts into the real dataclasses after import.  The test
    orchestrates everything through the result queue so a crashed child
    surfaces as a missing result (timeout) rather than a silent hang.
    """
    result: dict[str, object] = {"rank": rank, "stage": "start", "ok": False}
    try:
        from simpler.task_interface import (
            ChipBootstrapChannel,
            ChipBootstrapConfig,
            ChipBufferSpec,
            ChipCommBootstrapConfig,
            ChipWorker,
            HostBufferStaging,
        )

        worker = ChipWorker()
        worker.init(host_lib, aicpu_path, aicore_path, simpler_log_path, sim_context_path)
        result["stage"] = "init"

        cfg = ChipBootstrapConfig(
            comm=ChipCommBootstrapConfig(
                rank=rank,
                nranks=nranks,
                rootinfo_path=rootinfo_path,
                window_size=window_size,
            ),
            buffers=[ChipBufferSpec(**s) for s in buffer_specs],
            host_inputs=[HostBufferStaging(**s) for s in host_input_specs],
        )

        channel: ChipBootstrapChannel | None = None
        shm_attach: SharedMemory | None = None
        if channel_shm_name is not None:
            shm_attach = SharedMemory(name=channel_shm_name)
            channel = ChipBootstrapChannel(_shm_addr(shm_attach), max_buffer_count=376)

        try:
            res = worker.bootstrap_context(device_id=rank, cfg=cfg, channel=channel)
            result["stage"] = "bootstrap"
            result["device_ctx"] = int(res.device_ctx)
            result["local_window_base"] = int(res.local_window_base)
            result["actual_window_size"] = int(res.actual_window_size)
            result["buffer_ptrs"] = list(res.buffer_ptrs)

            # Read back the first buffer if the test asked for it.  Uses the
            # worker's device-to-host DMA so the test can assert on what
            # ``load_from_host`` actually wrote at ``buffer_ptrs[0]``.
            if readback_nbytes > 0 and res.buffer_ptrs:
                host_buf = (ctypes.c_char * readback_nbytes)()
                worker.copy_from(ctypes.addressof(host_buf), res.buffer_ptrs[0], readback_nbytes)
                result["readback"] = bytes(host_buf)

            # shutdown_bootstrap + finalize — matches the Worker bootstrap
            # loop's teardown order and leaves the sim shm segment clean for
            # the next test.
            worker.shutdown_bootstrap()
            worker.finalize()
            result["ok"] = True
        finally:
            if shm_attach is not None:
                shm_attach.close()
    except Exception:  # noqa: BLE001
        result["error"] = traceback.format_exc()
    finally:
        result_queue.put(result)


def _run_two_rank(
    *,
    window_size: int,
    buffer_specs: list[dict],
    host_inputs_for_rank: dict[int, tuple[list[dict], int]],
    rootinfo_suffix: str,
    channel_shm_names: dict[int, str] | None = None,
) -> dict[int, dict]:
    """Orchestrate a 2-rank fork test.

    ``host_inputs_for_rank[r]`` is a ``(staging_specs, readback_nbytes)`` pair
    so each rank can advertise its own inputs + ask for a device-to-host
    round-trip check.
    """
    bins = _sim_binaries()
    host_lib = str(bins.host_path)
    aicpu_path = str(bins.aicpu_path)
    aicore_path = str(bins.aicore_path)
    simpler_log_path = str(bins.simpler_log_path)

    sim_context_path = str(bins.sim_context_path) if bins.sim_context_path else ""

    nranks = 2
    rootinfo_path = f"/tmp/pto_bootstrap_sim_{os.getpid()}_{rootinfo_suffix}.bin"

    ctx = mp.get_context("fork")
    result_queue: mp.Queue = ctx.Queue()  # type: ignore[type-arg]
    procs = []
    for rank in range(nranks):
        staging, readback = host_inputs_for_rank.get(rank, ([], 0))
        channel_name = None if channel_shm_names is None else channel_shm_names.get(rank)
        p = ctx.Process(
            target=_rank_entry,
            args=(
                rank,
                nranks,
                rootinfo_path,
                window_size,
                host_lib,
                aicpu_path,
                aicore_path,
                simpler_log_path,
                sim_context_path,
                buffer_specs,
                staging,
                channel_name,
                result_queue,
                readback,
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

    return results


# ---------------------------------------------------------------------------
# 1. Happy path — bootstrap returns a populated result and window is carved.
# ---------------------------------------------------------------------------


class TestBootstrapContextHappyPath:
    def test_two_rank_no_host_inputs(self):
        buffer_specs = [
            {"name": "x", "dtype": "float32", "count": 16, "nbytes": 64},
        ]
        results = _run_two_rank(
            window_size=4096,
            buffer_specs=buffer_specs,
            host_inputs_for_rank={},
            rootinfo_suffix="happy",
        )
        for rank in (0, 1):
            r = results.get(rank)
            assert r is not None and r.get("ok"), f"rank {rank} failed: {r and r.get('error')}"
            assert r["local_window_base"] != 0, f"rank {rank} local_window_base is 0"
            assert r["actual_window_size"] >= 4096
            # Single buffer at window base — the 1:1 contract ChipContext relies on.
            assert r["buffer_ptrs"] == [r["local_window_base"]]


# ---------------------------------------------------------------------------
# 2. load_from_host — staged bytes end up at buffer_ptrs[0].
# ---------------------------------------------------------------------------


class TestBootstrapContextHostStaging:
    def test_load_from_host_round_trip(self):
        nbytes = 64
        payload = bytes(range(nbytes))

        shm = SharedMemory(create=True, size=nbytes)
        try:
            buf = shm.buf
            assert buf is not None
            buf[:nbytes] = payload

            buffer_specs = [
                {
                    "name": "x",
                    "dtype": "float32",
                    "count": 16,
                    "nbytes": nbytes,
                    "load_from_host": True,
                },
            ]
            # Only rank 0 consumes a host input; rank 1 still needs a buffer of
            # matching size so the two ranks carve identical windows.  Rank 1
            # is not asked to read back, which keeps the test focused on the
            # H2D staging path.
            host_inputs_by_rank = {
                0: ([{"name": "x", "shm_name": shm.name, "size": nbytes}], nbytes),
            }
            buffer_specs_r1 = [
                {
                    "name": "x",
                    "dtype": "float32",
                    "count": 16,
                    "nbytes": nbytes,
                    "load_from_host": False,
                },
            ]

            bins = _sim_binaries()
            host_lib = str(bins.host_path)
            aicpu_path = str(bins.aicpu_path)
            aicore_path = str(bins.aicore_path)
            simpler_log_path = str(bins.simpler_log_path)

            sim_context_path = str(bins.sim_context_path) if bins.sim_context_path else ""

            rootinfo_path = f"/tmp/pto_bootstrap_sim_{os.getpid()}_staging.bin"
            ctx = mp.get_context("fork")
            result_queue: mp.Queue = ctx.Queue()  # type: ignore[type-arg]
            procs = []
            for rank, specs in ((0, buffer_specs), (1, buffer_specs_r1)):
                staging, readback = host_inputs_by_rank.get(rank, ([], 0))
                p = ctx.Process(
                    target=_rank_entry,
                    args=(
                        rank,
                        2,
                        rootinfo_path,
                        4096,
                        host_lib,
                        aicpu_path,
                        aicore_path,
                        simpler_log_path,
                        sim_context_path,
                        specs,
                        staging,
                        None,
                        result_queue,
                        readback,
                    ),
                    daemon=False,
                )
                p.start()
                procs.append(p)

            results: dict[int, dict] = {}
            for _ in range(2):
                r = result_queue.get(timeout=180)
                results[int(r["rank"])] = r
            for p in procs:
                p.join(timeout=60)
            try:
                os.unlink(rootinfo_path)
            except FileNotFoundError:
                pass
        finally:
            shm.close()
            shm.unlink()

        assert results[0].get("ok"), f"rank 0 failed: {results[0].get('error')}"
        assert results[1].get("ok"), f"rank 1 failed: {results[1].get('error')}"
        assert results[0].get("readback") == payload, "round-trip payload mismatch"


# ---------------------------------------------------------------------------
# 2b. store_to_host — payload written by the child ends up in host_outputs shm.
# ---------------------------------------------------------------------------


def _store_rank_entry(  # noqa: PLR0913
    rank: int,
    nranks: int,
    rootinfo_path: str,
    window_size: int,
    host_lib: str,
    aicpu_path: str,
    aicore_path: str,
    simpler_log_path: str,
    sim_context_path: str,
    buffer_specs: list[dict],
    host_output_specs: list[dict],
    payload: bytes | None,
    result_queue: mp.Queue,  # type: ignore[type-arg]
) -> None:
    """Forked rank that exercises the store_to_host flush path.

    Mirrors the ``store_to_host=True`` flush that ``_chip_process_loop_with_bootstrap``
    runs after a successful task: write a known payload into the device buffer
    via ``copy_to``, then D2H-copy it into the parent's ``host_outputs``
    SharedMemory.  Leaves ``bootstrap_context`` to validate the
    store_to_host ↔ host_outputs pairing before any comm work runs.
    """
    result: dict[str, object] = {"rank": rank, "ok": False}
    try:
        from simpler.task_interface import (
            ChipBootstrapConfig,
            ChipBufferSpec,
            ChipCommBootstrapConfig,
            ChipWorker,
            HostBufferStaging,
        )

        worker = ChipWorker()
        worker.init(host_lib, aicpu_path, aicore_path, simpler_log_path, sim_context_path)

        cfg = ChipBootstrapConfig(
            comm=ChipCommBootstrapConfig(
                rank=rank,
                nranks=nranks,
                rootinfo_path=rootinfo_path,
                window_size=window_size,
            ),
            buffers=[ChipBufferSpec(**s) for s in buffer_specs],
            host_outputs=[HostBufferStaging(**s) for s in host_output_specs],
        )

        res = worker.bootstrap_context(device_id=rank, cfg=cfg)

        if payload is not None and res.buffer_ptrs:
            src = (ctypes.c_char * len(payload)).from_buffer_copy(payload)
            worker.copy_to(res.buffer_ptrs[0], ctypes.addressof(src), len(payload))

            # Manually run the same flush logic worker.py uses on TASK_DONE,
            # so this test covers the exact D2H handshake without needing a
            # full dispatch loop.
            for spec, ptr in zip(cfg.buffers, res.buffer_ptrs):
                if not spec.store_to_host or spec.nbytes == 0:
                    continue
                staging = cfg.output_staging(spec.name)
                shm = SharedMemory(name=staging.shm_name)
                try:
                    shm_buf = shm.buf
                    assert shm_buf is not None
                    host_ptr = ctypes.addressof(ctypes.c_char.from_buffer(shm_buf))
                    worker.copy_from(host_ptr, ptr, staging.size)
                finally:
                    shm.close()

        worker.shutdown_bootstrap()
        worker.finalize()
        result["ok"] = True
    except Exception:  # noqa: BLE001
        result["error"] = traceback.format_exc()
    finally:
        result_queue.put(result)


class TestBootstrapContextStoreToHost:
    def test_store_to_host_round_trip(self):
        """Round-trip a payload via the store_to_host + host_outputs pairing.

        Rank 0 writes a known pattern into its window buffer and flushes it to
        a parent-owned SharedMemory.  Rank 1 participates only so
        ``comm_alloc_windows`` can clear its internal barrier.  The parent
        reads the output shm after both children exit and asserts the payload
        round-tripped unchanged.
        """
        nbytes = 64
        payload = bytes(range(nbytes))

        shm = SharedMemory(create=True, size=nbytes)
        try:
            buf = shm.buf
            assert buf is not None
            buf[:nbytes] = b"\x00" * nbytes

            buffer_specs_r0 = [
                {
                    "name": "y",
                    "dtype": "float32",
                    "count": 16,
                    "nbytes": nbytes,
                    "store_to_host": True,
                },
            ]
            buffer_specs_r1 = [
                {
                    "name": "y",
                    "dtype": "float32",
                    "count": 16,
                    "nbytes": nbytes,
                    "store_to_host": False,
                },
            ]
            host_outputs_r0 = [{"name": "y", "shm_name": shm.name, "size": nbytes}]

            bins = _sim_binaries()
            host_lib = str(bins.host_path)
            aicpu_path = str(bins.aicpu_path)
            aicore_path = str(bins.aicore_path)
            simpler_log_path = str(bins.simpler_log_path)

            sim_context_path = str(bins.sim_context_path) if bins.sim_context_path else ""

            rootinfo_path = f"/tmp/pto_bootstrap_sim_{os.getpid()}_store.bin"
            ctx = mp.get_context("fork")
            result_queue: mp.Queue = ctx.Queue()  # type: ignore[type-arg]
            procs = []
            for rank, specs, outputs, pay in (
                (0, buffer_specs_r0, host_outputs_r0, payload),
                (1, buffer_specs_r1, [], None),
            ):
                p = ctx.Process(
                    target=_store_rank_entry,
                    args=(
                        rank,
                        2,
                        rootinfo_path,
                        4096,
                        host_lib,
                        aicpu_path,
                        aicore_path,
                        simpler_log_path,
                        sim_context_path,
                        specs,
                        outputs,
                        pay,
                        result_queue,
                    ),
                    daemon=False,
                )
                p.start()
                procs.append(p)

            results: dict[int, dict] = {}
            for _ in range(2):
                r = result_queue.get(timeout=180)
                results[int(r["rank"])] = r
            for p in procs:
                p.join(timeout=60)
            try:
                os.unlink(rootinfo_path)
            except FileNotFoundError:
                pass

            assert results[0].get("ok"), f"rank 0 failed: {results[0].get('error')}"
            assert results[1].get("ok"), f"rank 1 failed: {results[1].get('error')}"
            readback = bytes(shm.buf[:nbytes])  # type: ignore[index]
        finally:
            shm.close()
            shm.unlink()

        assert readback == payload, f"store_to_host round-trip mismatch: got {readback!r}"


# ---------------------------------------------------------------------------
# 3. Channel integration — parent reads SUCCESS fields from the mailbox.
# ---------------------------------------------------------------------------


class TestBootstrapContextChannel:
    def test_channel_publishes_success_fields(self):
        from _task_interface import (  # pyright: ignore[reportMissingImports]
            CHIP_BOOTSTRAP_MAILBOX_SIZE,
            ChipBootstrapChannel,
            ChipBootstrapMailboxState,
        )

        # One mailbox per rank — the parent owns both, forwards the shm name
        # to each child so the child can attach and publish its result.
        channels_shm = {rank: SharedMemory(create=True, size=CHIP_BOOTSTRAP_MAILBOX_SIZE) for rank in range(2)}
        try:
            buffer_specs = [
                {"name": "x", "dtype": "float32", "count": 16, "nbytes": 64},
            ]
            channel_shm_names = {rank: shm.name for rank, shm in channels_shm.items()}
            results = _run_two_rank(
                window_size=4096,
                buffer_specs=buffer_specs,
                host_inputs_for_rank={},
                rootinfo_suffix="channel",
                channel_shm_names=channel_shm_names,
            )

            for rank in (0, 1):
                r = results[rank]
                assert r.get("ok"), f"rank {rank} failed: {r.get('error')}"

                channel = ChipBootstrapChannel(_shm_addr(channels_shm[rank]), max_buffer_count=376)
                assert channel.state == ChipBootstrapMailboxState.SUCCESS
                assert channel.device_ctx == r["device_ctx"]
                assert channel.local_window_base == r["local_window_base"]
                assert channel.actual_window_size == r["actual_window_size"]
                assert channel.buffer_ptrs == r["buffer_ptrs"]
        finally:
            for shm in channels_shm.values():
                shm.close()
                shm.unlink()


# ---------------------------------------------------------------------------
# 4b. Error path — store_to_host=True without a matching host_outputs entry.
# ---------------------------------------------------------------------------


def _missing_output_staging_rank_entry(
    host_lib: str,
    aicpu_path: str,
    aicore_path: str,
    simpler_log_path: str,
    sim_context_path: str,
    channel_shm_name: str,
    result_queue: mp.Queue,  # type: ignore[type-arg]
) -> None:
    """Trip the store_to_host ↔ host_outputs symmetry check in bootstrap_context.

    Runs single-process: the new validation fires before any communicator
    work, so no peer rank is required.  Verifies both the child-side
    exception and the channel payload the parent will see.
    """
    result: dict[str, object] = {"raised": False, "state": None, "message": None}
    try:
        from simpler.task_interface import (
            ChipBootstrapChannel,
            ChipBootstrapConfig,
            ChipBufferSpec,
            ChipWorker,
        )

        worker = ChipWorker()
        worker.init(host_lib, aicpu_path, aicore_path, simpler_log_path, sim_context_path)

        shm = SharedMemory(name=channel_shm_name)
        try:
            channel = ChipBootstrapChannel(_shm_addr(shm), max_buffer_count=376)

            cfg = ChipBootstrapConfig(
                comm=None,
                buffers=[
                    ChipBufferSpec(
                        name="y",
                        dtype="float32",
                        count=1,
                        nbytes=4,
                        store_to_host=True,
                    )
                ],
                host_outputs=[],
            )
            try:
                worker.bootstrap_context(device_id=0, cfg=cfg, channel=channel)
            except ValueError as e:
                result["raised"] = True
                result["exc_msg"] = str(e)

            result["state"] = int(channel.state)
            result["message"] = channel.error_message
        finally:
            shm.close()
            worker.shutdown_bootstrap()
            worker.finalize()
    except Exception:  # noqa: BLE001
        result["error"] = traceback.format_exc()
    finally:
        result_queue.put(result)


class TestBootstrapContextMissingOutputStaging:
    def test_store_to_host_without_host_outputs_raises(self):
        from _task_interface import (  # pyright: ignore[reportMissingImports]
            CHIP_BOOTSTRAP_MAILBOX_SIZE,
            ChipBootstrapChannel,
            ChipBootstrapMailboxState,
        )

        bins = _sim_binaries()
        host_lib = str(bins.host_path)
        aicpu_path = str(bins.aicpu_path)
        aicore_path = str(bins.aicore_path)
        simpler_log_path = str(bins.simpler_log_path)

        sim_context_path = str(bins.sim_context_path) if bins.sim_context_path else ""

        shm = SharedMemory(create=True, size=CHIP_BOOTSTRAP_MAILBOX_SIZE)
        buf = shm.buf
        assert buf is not None
        for off in range(0, CHIP_BOOTSTRAP_MAILBOX_SIZE, 8):
            struct.pack_into("Q", buf, off, 0)
        try:
            ctx = mp.get_context("fork")
            result_queue: mp.Queue = ctx.Queue()  # type: ignore[type-arg]
            p = ctx.Process(
                target=_missing_output_staging_rank_entry,
                args=(host_lib, aicpu_path, aicore_path, simpler_log_path, sim_context_path, shm.name, result_queue),
                daemon=False,
            )
            p.start()
            r = result_queue.get(timeout=60)
            p.join(timeout=30)

            assert r.get("raised"), f"expected ValueError; got {r}"
            exc_msg = str(r.get("exc_msg", ""))
            assert "store_to_host=True" in exc_msg, f"exc_msg missing sentinel: {exc_msg!r}"
            assert "host_outputs" in exc_msg, f"exc_msg missing 'host_outputs': {exc_msg!r}"

            channel = ChipBootstrapChannel(_shm_addr(shm), max_buffer_count=376)
            assert channel.state == ChipBootstrapMailboxState.ERROR
            assert channel.error_code == 1
            assert channel.error_message.startswith("ValueError: ")
            assert "store_to_host=True" in channel.error_message
        finally:
            shm.close()
            shm.unlink()
