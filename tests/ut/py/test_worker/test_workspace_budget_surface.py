# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: PLC0415
"""Public surface of the workspace budget: where it is refused, and what the
close-time guard does with each answer the native ledger can give.

These are the parts that hold without a device. Which regions the budget then
owns, and when their consumers retire, is covered by the C++ WorkspaceManager
cases that drive the ledger itself.
"""

from __future__ import annotations

import pytest

_COMMON = {"platform": "a2a3", "runtime": "host_build_graph"}


def _l2(**extra):
    from simpler.worker import Worker

    return Worker(2, device_id=0, **_COMMON, **extra)


class _FakeImpl:
    """Stands in for the nanobind accessor, which needs a real device context."""

    def __init__(self, answer):
        self._answer = answer

    def workspace_report(self):
        return self._answer


class _FakeChipWorker:
    def __init__(self, answer):
        self._impl = _FakeImpl(answer)


def _available(**fields):
    report = {"blocks_published": 1, "live_blocked": 0}
    report.update(fields)
    return ("available", report)


def test_a_non_integer_budget_is_refused_as_the_config_is_stored():
    """Checked where the key is stored, so a wrong type cannot reach the device
    attach or any later stage."""
    from simpler.worker import Worker

    with pytest.raises(TypeError, match="workspace_budget_bytes must be an int"):
        Worker(2, device_id=0, **_COMMON, workspace_budget_bytes="4mb")
    # bool is an int subclass, and True would silently mean one byte.
    with pytest.raises(TypeError, match="workspace_budget_bytes must be an int"):
        Worker(2, device_id=0, **_COMMON, workspace_budget_bytes=True)


def test_an_absent_budget_leaves_the_worker_unchanged():
    """The default: no key, nothing managed, and no new refusal anywhere."""
    worker = _l2()
    assert "workspace_budget_bytes" not in worker._config
    assert worker._validated_workspace_budget("a2a3") == 0


def test_a_budget_below_one_region_alignment_is_refused():
    """A budget smaller than a single base alignment could only ever refuse
    every request, so it is rejected rather than latched."""
    worker = _l2(workspace_budget_bytes=512)
    with pytest.raises(ValueError, match="at least 1024 bytes"):
        worker._validated_workspace_budget("a2a3")


def test_a_non_positive_budget_is_refused():
    worker = _l2(workspace_budget_bytes=0)
    with pytest.raises(ValueError, match="positive byte count"):
        worker._validated_workspace_budget("a2a3")


def test_the_simulation_platform_is_refused_before_any_device_attach():
    """That backend manages no device workspace, so asking for a budget there
    fails rather than running unmanaged."""
    worker = _l2(workspace_budget_bytes=1 << 20)
    with pytest.raises(ValueError, match="not supported on platform"):
        worker._validated_workspace_budget("a2a3sim")


def test_a_worker_owning_chip_children_refuses_the_budget_before_any_resource():
    """A chip child owns its own ChipWorker, and this process cannot ask that
    child whether its workspace still has a drainable consumer before it
    broadcasts shutdown and reaps it. Without that question the guard does not
    exist on this route, so the request fails instead of running unprotected —
    and it fails before a mailbox, a pre-fork worker or a fork exists."""
    from simpler.worker import Worker

    worker = Worker(3, device_ids=[0], num_sub_workers=0, **_COMMON, workspace_budget_bytes=1 << 20)
    with pytest.raises(ValueError, match="only supported on a same-process level-2 Worker"):
        worker._init_hierarchical()
    # Refused before it could allocate any startup resource.
    assert worker._chip_shms == []
    assert worker._sub_shms == []
    assert worker._worker is None


def test_the_close_guard_passes_when_no_budget_was_ever_latched():
    """The default path adds no close refusal at all."""
    worker = _l2()
    worker._chip_worker = _FakeChipWorker(("disabled", None))
    worker._check_workspace_live()


def test_the_close_guard_refuses_when_an_enabled_budget_cannot_be_read():
    """ "No budget" and "a budget whose accounting cannot be read" have opposite
    safety consequences: the second must not release the owner Buffers a live
    consumer may still be reading."""
    worker = _l2(workspace_budget_bytes=1 << 20)
    worker._chip_worker = _FakeChipWorker(("unavailable", None))
    with pytest.raises(RuntimeError, match="accounting could not be read"):
        worker._check_workspace_live()


def test_the_close_guard_refuses_while_a_consumer_can_still_be_drained():
    """The refusal a caller can act on: finalize those runs and close again."""
    worker = _l2(workspace_budget_bytes=1 << 20)
    worker._chip_worker = _FakeChipWorker(_available(live_blocked=2))
    with pytest.raises(RuntimeError, match=r"2 run\(s\) still hold workspace"):
        worker._check_workspace_live()


def test_the_close_guard_passes_once_no_consumer_remains():
    worker = _l2(workspace_budget_bytes=1 << 20)
    worker._chip_worker = _FakeChipWorker(_available(live_blocked=0))
    worker._check_workspace_live()


def test_a_context_that_published_no_block_is_exempt_on_that_fact_alone():
    """The only exemption during a partial-init rollback is the fact that this
    context never owned a block — not a query that failed, and not the rollback
    flag."""
    worker = _l2(workspace_budget_bytes=1 << 20)
    worker._chip_worker = _FakeChipWorker(_available(blocks_published=0, live_blocked=3))
    worker._check_workspace_live()


def test_the_guard_is_driven_alone_before_the_buffer_cleanup_batch():
    """`CleanupJournal.drive` attempts every matching entry even after one
    fails, so the guard cannot share a batch with the owner-Buffer release: it
    is registered and checked on its own, and its failure returns before that
    batch exists."""
    import inspect

    from simpler.worker import Worker

    source = inspect.getsource(Worker._teardown_worker_tree)
    gate = source.index("workspace live-consumer gate")
    batch = source.index("pre_transport_keys: set[")
    assert gate < batch, "the workspace guard must be driven before the pre-transport batch is built"
    # Its own single-entry drive, and a raise on both paths: the startup-abort
    # exemption is a fact about ownership, not about which path is running.
    gate_drive = source.index('drive({("native", "workspace live-consumer gate")})')
    assert gate_drive < batch
    assert "raise gate_err" in source[gate_drive:batch]
