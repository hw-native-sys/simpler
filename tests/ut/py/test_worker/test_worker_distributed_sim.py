# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: PLC0415
"""Simulation tests for Worker-level chip bootstrap orchestration.

Covers the two externally-visible guarantees:

  1. Happy path — `Worker(level=3, chip_bootstrap_configs=...)` populates
     `worker.chip_contexts` with one per chip, and `close()` leaves no
     residue behind in `/dev/shm`.
  2. Error path — a bad `ChipBootstrapConfig` (placement="bogus") trips
     ValueError inside `bootstrap_context`; the channel publishes ERROR,
     the parent raises `RuntimeError`, and every forked child is reaped so
     the test process has no dangling descendants.

These tests drive the sim backend of `tensormap_and_ringbuffer`, so no
Ascend NPU is required.  `/dev/shm` only exists on Linux; the sweep
helpers short-circuit on other platforms.
"""

from __future__ import annotations

import os

import pytest

_SHM_DIR = "/dev/shm"


def _shm_supported() -> bool:
    return os.path.isdir(_SHM_DIR)


def _shm_snapshot() -> set[str]:
    """Return the set of ``SharedMemory``-created segment names in /dev/shm.

    Only tracks names with the ``psm_`` prefix (Python's `SharedMemory`
    default) so unrelated segments — most importantly the sim HCCL backend
    uses ``simpler_`` and may legitimately outlive a SIGKILLed rank — do
    not pollute the leak assertion.  Returns an empty set when /dev/shm
    is absent (non-Linux).
    """
    if not _shm_supported():
        return set()
    try:
        return {name for name in os.listdir(_SHM_DIR) if name.startswith("psm_")}
    except OSError:
        return set()


def _sim_binaries():
    """Resolve pre-built a2a3sim runtime binaries, or skip if unavailable."""
    from simpler_setup.runtime_builder import RuntimeBuilder

    build = bool(os.environ.get("PTO_UT_BUILD"))
    try:
        bins = RuntimeBuilder(platform="a2a3sim").get_binaries("tensormap_and_ringbuffer", build=build)
    except FileNotFoundError as e:
        pytest.skip(f"a2a3sim runtime binaries unavailable: {e}")
    return bins


def _make_configs(nranks: int, rootinfo_path: str, window_size: int = 4096):
    """Build a `[ChipBootstrapConfig] * nranks` with a single named buffer.

    The buffer carves the window at offset 0, so we get a deterministic
    `buffer_ptrs["x"] == local_window_base` invariant to assert on.
    """
    from simpler.task_interface import ChipBootstrapConfig, ChipBufferSpec, ChipCommBootstrapConfig

    return [
        ChipBootstrapConfig(
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
                    count=16,
                    placement="window",
                    nbytes=64,
                ),
            ],
        )
        for rank in range(nranks)
    ]


class TestWorkerBootstrapHappyPath:
    def test_init_populates_chip_contexts(self):
        from simpler.worker import Worker

        _sim_binaries()  # skip early if runtime binaries are missing
        rootinfo_path = f"/tmp/pto_worker_l6_sim_{os.getpid()}_happy.bin"
        nranks = 2

        before = _shm_snapshot()

        cfgs = _make_configs(nranks, rootinfo_path)
        worker = Worker(
            level=3,
            platform="a2a3sim",
            runtime="tensormap_and_ringbuffer",
            device_ids=list(range(nranks)),
            num_sub_workers=0,
            chip_bootstrap_configs=cfgs,
        )
        try:
            worker.init()

            ctxs = worker.chip_contexts
            assert len(ctxs) == nranks, f"expected {nranks} ChipContext, got {len(ctxs)}"
            for rank, ctx in enumerate(ctxs):
                assert ctx.device_id == rank
                assert ctx.rank == rank
                assert ctx.nranks == nranks
                assert ctx.actual_window_size >= 4096
                assert ctx.local_window_base != 0
                # buffer_ptrs is a name → device-ptr dict, and the single
                # "x" buffer lives at window base (offset 0).
                assert set(ctx.buffer_ptrs.keys()) == {"x"}
                assert ctx.buffer_ptrs["x"] == ctx.local_window_base
        finally:
            worker.close()
            try:
                os.unlink(rootinfo_path)
            except FileNotFoundError:
                pass

        after = _shm_snapshot()
        if _shm_supported():
            leaked = after - before
            assert not leaked, f"/dev/shm segments leaked after close(): {sorted(leaked)}"

    def test_chip_contexts_before_init_raises(self):
        """Accessing `chip_contexts` before `init()` must fail loudly."""
        from simpler.worker import Worker

        worker = Worker(
            level=3,
            platform="a2a3sim",
            runtime="tensormap_and_ringbuffer",
            device_ids=[0, 1],
            num_sub_workers=0,
            chip_bootstrap_configs=_make_configs(2, "/tmp/pto_unused.bin"),
        )
        with pytest.raises(RuntimeError, match="after init"):
            _ = worker.chip_contexts


class TestWorkerBootstrapErrorPath:
    def test_invalid_placement_fails_init_and_cleans_up(self):
        """A ValueError inside bootstrap_context → parent RuntimeError → clean teardown."""
        from simpler.task_interface import ChipBootstrapConfig, ChipBufferSpec, ChipCommBootstrapConfig
        from simpler.worker import Worker

        _sim_binaries()  # skip if runtime binaries are missing

        rootinfo_path = f"/tmp/pto_worker_l6_sim_{os.getpid()}_err.bin"
        nranks = 2

        # Rank 0 carries a bogus placement, which trips the `placement != 'window'`
        # guard inside `bootstrap_context` *before* any communicator work —
        # no peer rank is required to observe the failure.  Rank 1 uses a
        # valid config; it will either observe ERROR on rank 0 via the
        # shared sim segment or be reaped by the abort path before it
        # completes bootstrap.
        bad = ChipBootstrapConfig(
            comm=ChipCommBootstrapConfig(rank=0, nranks=nranks, rootinfo_path=rootinfo_path, window_size=4096),
            buffers=[
                ChipBufferSpec(name="x", dtype="float32", count=1, placement="bogus", nbytes=4),
            ],
        )
        good = ChipBootstrapConfig(
            comm=ChipCommBootstrapConfig(rank=1, nranks=nranks, rootinfo_path=rootinfo_path, window_size=4096),
            buffers=[
                ChipBufferSpec(name="x", dtype="float32", count=1, placement="window", nbytes=4),
            ],
        )

        before = _shm_snapshot()

        worker = Worker(
            level=3,
            platform="a2a3sim",
            runtime="tensormap_and_ringbuffer",
            device_ids=[0, 1],
            num_sub_workers=0,
            chip_bootstrap_configs=[bad, good],
        )
        with pytest.raises(RuntimeError, match="chip 0 bootstrap failed"):
            worker.init()

        # init() abort path must return the Worker to an uninitialised state.
        assert worker._initialized is False
        # close() on a failed init() is a no-op guard but must not raise.
        worker.close()

        try:
            os.unlink(rootinfo_path)
        except FileNotFoundError:
            pass

        after = _shm_snapshot()
        if _shm_supported():
            leaked = after - before
            assert not leaked, f"/dev/shm segments leaked after init() failure: {sorted(leaked)}"


class TestWorkerBootstrapValidation:
    def test_level_below_3_rejected(self):
        from simpler.worker import Worker

        with pytest.raises(ValueError, match="level >= 3"):
            Worker(
                level=2,
                platform="a2a3sim",
                runtime="tensormap_and_ringbuffer",
                device_id=0,
                chip_bootstrap_configs=_make_configs(1, "/tmp/pto_unused.bin"),
            )

    def test_length_mismatch_rejected(self):
        from simpler.worker import Worker

        with pytest.raises(ValueError, match="must equal device_ids length"):
            Worker(
                level=3,
                platform="a2a3sim",
                runtime="tensormap_and_ringbuffer",
                device_ids=[0, 1],
                num_sub_workers=0,
                chip_bootstrap_configs=_make_configs(1, "/tmp/pto_unused.bin"),
            )
