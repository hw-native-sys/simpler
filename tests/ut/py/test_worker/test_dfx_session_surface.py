# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: PLC0415
"""Public surface of cross-run diagnostic collection.

These assert the parts that hold without a device: the admission rules of
`Worker.flush_diagnostics`, and that the enable flag is a Worker-level option
travelling the same path `enable_sdma` already takes. What a retained run
actually collects is covered by the C++ retained-run cases, which drive the
collector, its threads and the device-side producer directly.
"""

from __future__ import annotations

import pytest


def _session_worker():
    from simpler.worker import Worker

    return Worker(
        3, device_ids=[0], num_sub_workers=0, platform="a2a3sim", runtime="host_build_graph", collect_across_runs=True
    )


def _make_worker(level: int):
    from simpler.worker import Worker

    common = {"platform": "a2a3sim", "runtime": "host_build_graph"}
    if level == 2:
        return Worker(level, device_id=0, **common)
    return Worker(level, device_ids=[0], num_sub_workers=0, **common)


def test_flush_diagnostics_requires_a_ready_worker():
    """The operation lease is the fence, so an uninitialized worker is rejected
    before any child is touched."""
    worker = _make_worker(3)
    with pytest.raises(RuntimeError, match=r"requires an initialized \(READY\) worker"):
        worker.flush_diagnostics()


def test_flush_diagnostics_is_level_three_only():
    """L2 owns its chip in process and has no chip children to flush; whether an
    in-process flush is safe against submission and close is untraced, so the
    first scope refuses rather than guessing."""
    worker = _make_worker(2)
    with pytest.raises(RuntimeError, match="level-3 worker"):
        worker.flush_diagnostics()


def test_dfx_session_is_a_worker_option_and_defaults_off():
    """The gate is a Worker-level config key, not a per-call one: `CallConfig` is
    packed with a size assertion feeding mailbox offsets and must not grow."""
    from simpler.task_interface import CallConfig

    default_worker = _make_worker(3)
    assert default_worker._config.get("collect_across_runs", False) is False

    enabled = _session_worker()
    assert enabled._config["collect_across_runs"] is True

    assert not hasattr(CallConfig(), "collect_across_runs"), "the gate must not be a per-task wire field"


def test_chip_worker_init_accepts_the_session_flag():
    """The flag reaches the native init through the same keyword path
    `enable_sdma` uses, so runs can be retained before the first one starts.
    The name it shipped under stays accepted beside the canonical one, and
    neither has a boolean default: `None` is what lets the two be told apart
    from each other and from not being given at all."""
    import inspect

    from simpler.task_interface import ChipWorker

    signature = inspect.signature(ChipWorker.init)
    assert "collect_across_runs" in signature.parameters
    assert signature.parameters["collect_across_runs"].default is None
    assert "dfx_session" in signature.parameters, "the name this option shipped under stopped being accepted"
    assert signature.parameters["dfx_session"].default is None
    assert hasattr(ChipWorker, "flush_diagnostics")


def test_close_flush_shares_one_budget_across_chips(monkeypatch):
    """The close-time flush phase holds one absolute budget for every chip.

    Time is advanced between children, so each share must be strictly smaller
    than the last and the budget must run out — assertions an implementation
    handing every chip a renewed full grace period cannot satisfy. What the
    endpoint then does with an exhausted share, refuse rather than substitute a
    default, is decided in C++ and is not reachable from here.
    """
    from simpler import worker as worker_mod

    grace = float(worker_mod._ROLLBACK_GRACEFUL_TIMEOUT_S)
    # Four chips and a step just over a third of the budget: the third share is
    # already spent, so both the decrease and the exhaustion are asserted.
    step = grace / 3.0 + 0.01
    clock = {"now": 1000.0}

    class _RecordingOrch:
        def __init__(self) -> None:
            self.budgets: list[float] = []

        def flush_diagnostics(self, worker_id: int, timeout_s: float) -> None:
            self.budgets.append(timeout_s)
            clock["now"] += step

    monkeypatch.setattr(worker_mod, "_monotonic", lambda: clock["now"])

    worker = _make_worker(3)
    orch = _RecordingOrch()
    worker._orch = orch
    # Only counted, never mapped: this phase reads the child count.
    worker._chip_shms = [object(), object(), object(), object()]
    worker._config["collect_across_runs"] = True

    assert worker._close_flush_diagnostics() is None
    assert len(orch.budgets) == 4, "a chip was skipped, so one child's artifacts would go unflushed"
    assert orch.budgets[0] == pytest.approx(grace), "the first chip did not get the whole phase budget"
    for earlier, later in zip(orch.budgets, orch.budgets[1:]):
        assert later < earlier, f"a later chip got {later} after {earlier}: the budget is not shared"
    assert orch.budgets[-1] == 0.0, "the budget never ran out, so an exhausted share is untested"
    assert all(budget >= 0.0 for budget in orch.budgets), "a negative share would read as an unbounded wait"


def test_close_flush_aggregates_every_chips_failure():
    """One chip's refusal must not skip the others, and the phase reports rather
    than raises so teardown, shutdown and reap still run."""

    class _FailingOrch:
        def __init__(self) -> None:
            self.calls = 0

        def flush_diagnostics(self, worker_id: int, timeout_s: float) -> None:
            self.calls += 1
            raise RuntimeError(f"chip {worker_id} refused")

    worker = _make_worker(3)
    orch = _FailingOrch()
    worker._orch = orch
    worker._chip_shms = [object(), object()]
    worker._config["collect_across_runs"] = True

    error = worker._close_flush_diagnostics()
    assert orch.calls == 2, "the first failure stopped the phase"
    assert isinstance(error, RuntimeError)
    assert "chip 0 refused" in str(error)
    assert "chip 1 refused" in str(error)
