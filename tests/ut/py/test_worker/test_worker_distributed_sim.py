# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: PLC0415
"""Simulation tests for Worker-level communication-plan orchestration.

Covers the two externally-visible guarantees:

  1. Happy path — `Worker(level=3, comm_plan=...)` populates
     `worker.chip_contexts` with one per chip, and `close()` leaves no
     residue behind in `/dev/shm`.
  2. Error path — a bad communication plan (e.g. store_to_host without a
     matching host_outputs entry) trips ValueError inside `bootstrap_context`;
     the channel publishes ERROR, the parent raises `RuntimeError`, and every
     forked child is reaped so the test process has no dangling descendants.

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


def _make_comm_plan(nranks: int, window_size: int = 4096):
    """Build a `CommDomainPlan` with a single default domain and named buffer.

    The buffer carves the window at offset 0, so we get a deterministic
    `buffer_ptrs["x"] == local_window_base` invariant to assert on.
    """
    from simpler.task_interface import ChipBufferSpec, CommDomain, CommDomainPlan

    return CommDomainPlan(
        domains=[
            CommDomain(
                name="default",
                worker_indices=list(range(nranks)),
                window_size=window_size,
                buffers=[
                    ChipBufferSpec(
                        name="x",
                        dtype="float32",
                        count=16,
                        nbytes=64,
                    ),
                ],
            )
        ]
    )


def _make_bad_store_plan(nranks: int):
    from simpler.task_interface import ChipBufferSpec, CommDomain, CommDomainPlan

    return CommDomainPlan(
        domains=[
            CommDomain(
                name="default",
                worker_indices=list(range(nranks)),
                window_size=4096,
                # Missing matching host_outputs intentionally exercises the
                # bootstrap_context staging-symmetry error path.
                buffers=[
                    ChipBufferSpec(name="x", dtype="float32", count=1, nbytes=4, store_to_host=True),
                ],
            )
        ]
    )


def _make_store_plan(nranks: int):
    from simpler.task_interface import ChipBufferSpec, CommDomain, CommDomainPlan

    return CommDomainPlan(
        domains=[
            CommDomain(
                name="default",
                worker_indices=list(range(nranks)),
                window_size=4096,
                buffers=[
                    ChipBufferSpec(name="x", dtype="float32", count=1, nbytes=4, store_to_host=True),
                ],
            )
        ]
    )


def _make_out_of_range_plan():
    from simpler.task_interface import CommDomain, CommDomainPlan

    return CommDomainPlan(
        domains=[
            CommDomain(
                name="default",
                worker_indices=[0, 2],
                window_size=4096,
            )
        ]
    )


class TestWorkerBootstrapHappyPath:
    def test_init_populates_chip_contexts(self):
        from simpler.worker import Worker

        _sim_binaries()  # skip early if runtime binaries are missing
        nranks = 2

        before = _shm_snapshot()

        worker = Worker(
            level=3,
            platform="a2a3sim",
            runtime="tensormap_and_ringbuffer",
            device_ids=list(range(nranks)),
            num_sub_workers=0,
            comm_plan=_make_comm_plan(nranks),
        )
        try:
            worker.init()

            ctxs = worker.chip_contexts
            assert len(ctxs) == nranks, f"expected {nranks} ChipContext, got {len(ctxs)}"
            for rank, ctx in enumerate(ctxs):
                assert ctx.device_id == rank
                domain = ctx.domains["default"]
                assert domain.domain_rank == rank
                assert domain.domain_size == nranks
                assert domain.actual_window_size >= 4096
                assert domain.local_window_base != 0
                # buffer_ptrs is a name → device-ptr dict, and the single
                # "x" buffer lives at window base (offset 0).
                assert set(domain.buffer_ptrs.keys()) == {"x"}
                assert domain.buffer_ptrs["x"] == domain.local_window_base
        finally:
            worker.close()

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
            comm_plan=_make_comm_plan(2),
        )
        with pytest.raises(RuntimeError, match="after init"):
            _ = worker.chip_contexts


class TestWorkerBootstrapErrorPath:
    def test_bootstrap_value_error_fails_init_and_cleans_up(self):
        """A ValueError inside bootstrap_context → parent RuntimeError → clean teardown."""
        from simpler.worker import Worker

        _sim_binaries()  # skip if runtime binaries are missing

        nranks = 2

        before = _shm_snapshot()

        worker = Worker(
            level=3,
            platform="a2a3sim",
            runtime="tensormap_and_ringbuffer",
            device_ids=[0, 1],
            num_sub_workers=0,
            comm_plan=_make_bad_store_plan(nranks),
        )
        with pytest.raises(RuntimeError, match="chip 0 bootstrap failed"):
            worker.init()

        # init() abort path must return the Worker to an uninitialised state.
        assert worker._initialized is False
        # close() on a failed init() is a no-op guard but must not raise.
        worker.close()

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
                comm_plan=_make_comm_plan(1),
            )

    def test_out_of_range_worker_index_rejected(self):
        from simpler.worker import Worker

        with pytest.raises(ValueError, match="outside"):
            Worker(
                level=3,
                platform="a2a3sim",
                runtime="tensormap_and_ringbuffer",
                device_ids=[0, 1],
                num_sub_workers=0,
                comm_plan=_make_out_of_range_plan(),
            )

    def test_old_single_domain_chip_comm_bootstrap_config_is_not_public(self):
        import simpler.task_interface as ti

        assert not hasattr(ti, "ChipCommBootstrapConfig")

    def test_new_style_explicit_bootstrap_configs_are_accepted(self):
        from simpler.task_interface import ChipBootstrapConfig, HostBufferStaging
        from simpler.worker import Worker

        plan = _make_store_plan(2)
        cfgs = [
            ChipBootstrapConfig(
                comm=plan.bootstrap_for_worker(rank),
                host_outputs=[
                    HostBufferStaging(
                        domain_name="default",
                        name="x",
                        shm_name=f"psm_rank_{rank}",
                        size=4,
                    )
                ],
            )
            for rank in range(2)
        ]

        worker = Worker(
            level=3,
            platform="a2a3sim",
            runtime="tensormap_and_ringbuffer",
            device_ids=[0, 1],
            num_sub_workers=0,
            chip_bootstrap_configs=cfgs,
        )

        assert worker._chip_bootstrap_configs == cfgs
