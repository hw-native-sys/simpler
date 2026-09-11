# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: PLC0415
"""Sim-backend tests for ``Worker.detach_persistent_domain``.

Exercises the public retention API a caller uses to outlive one run's
completion fence with an allocated CommDomain — e.g. to dispatch many
requests against the same domain instead of allocating a fresh one per
request.
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
    _ = bins
    return Worker(
        level=3,
        platform="a2a3sim",
        runtime="tensormap_and_ringbuffer",
        device_ids=list(range(nranks)),
        num_sub_workers=0,
    )


class TestDetachTransfersOwnershipToWorker:
    def test_detached_handle_survives_run_and_frees_on_close(self):
        from simpler.task_interface import CallConfig

        captured: dict[str, object] = {}

        def orch_fn(orch, _args, _cfg):
            handle = orch.allocate_domain(name="tp", workers=[0, 1], window_size=4096)
            worker.detach_persistent_domain(handle)
            captured["handle"] = handle
            # Deliberately not released: detach is what makes this run-fence
            # complete without releasing the domain.

        worker = _make_worker(nranks=2)
        worker.init()
        worker.run(orch_fn, args=None, config=CallConfig())

        handle = captured["handle"]
        assert not handle.released
        # Detach removed the run-local claim but kept the Worker-level one.
        assert worker.live_domains == {"tp": handle}

        worker.close()
        assert handle.released
        assert handle.freed


class TestDetachRequiresAnInFlightRun:
    def test_raises_outside_a_run(self):
        worker = _make_worker(nranks=2)
        worker.init()
        try:
            with pytest.raises(RuntimeError, match="no run resources are being built"):
                worker.detach_persistent_domain(object())  # type: ignore[arg-type]
        finally:
            worker.close()


class TestDetachRejectsAStaleHandle:
    def test_second_detach_of_the_same_handle_raises(self):
        from simpler.task_interface import CallConfig

        captured: dict[str, object] = {}

        def orch_fn(orch, _args, _cfg):
            handle = orch.allocate_domain(name="tp", workers=[0, 1], window_size=4096)
            worker.detach_persistent_domain(handle)
            with pytest.raises(RuntimeError, match="not this run's live claim"):
                worker.detach_persistent_domain(handle)
            captured["reached_assertion"] = True

        worker = _make_worker(nranks=2)
        worker.init()
        try:
            worker.run(orch_fn, args=None, config=CallConfig())
        finally:
            worker.close()

        assert captured.get("reached_assertion") is True
