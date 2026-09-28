# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: PLC0415
"""Which chip children own their workspace regions, and how the request reaches them.

Two halves, both driven through the production functions. The parent decides
once per Worker (`_chip_children_manage_workspace`) and the forked child entry
(`_chip_process_loop`) carries the answer into `ChipWorker.init`.

What the owner then does with the four regions — reuse, growth, retirement,
quarantine, the terminal sweep — is the C++ WorkspaceManager's, covered by the
cases that drive that ledger directly. These pin the routing, which is the only
thing this level can establish without a device.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

_COMMON = {"platform": "a2a3", "runtime": "host_build_graph"}


def _l3(**extra):
    from simpler.worker import Worker

    return Worker(3, num_sub_workers=0, **{**_COMMON, **extra})


class _RecordingChip:
    """Records what `init` received, then fails it.

    Failing is what keeps the child entry in this test's reach: it publishes
    INIT_FAILED and returns, so the assertion is about the real call the real
    function made, with no device, no serve loop and no second entry point.
    """

    seen: dict = {}

    def __init__(self):
        type(self).seen = {}

    def init(self, *_args, **kwargs):
        type(self).seen = dict(kwargs)
        raise RuntimeError("no device in this environment")

    def configure_launch_depth(self, *_a, **_k):  # pragma: no cover - depth 1 here
        pass


def _run_chip_entry(monkeypatch, **extra):
    """Drive the real `_chip_process_loop` up to its `ChipWorker.init` call."""
    import simpler.worker as worker_mod

    monkeypatch.setattr(worker_mod, "ChipWorker", _RecordingChip)
    buf = memoryview(bytearray(worker_mod.MAILBOX_SIZE))
    worker_mod._chip_process_loop(buf, object(), 0, {}, {}, {}, b"o", b"a", b"c", **extra)
    return _RecordingChip.seen, buf


def test_a_directly_closed_level_three_worker_manages_its_chip_children():
    """The approved route: a Worker the caller created, inits and closes itself.

    Its `close()` is the caller's call, so a child that fails its teardown is
    reported to somebody who can act on it.
    """
    worker = _l3(device_ids=[0, 1])
    assert worker._is_startup_root is True
    assert worker._chip_children_manage_workspace() is True


def test_a_worker_some_other_process_started_leaves_its_children_unmanaged():
    """`_is_startup_root` is False for every descendant Worker: `init()` sets it
    from `_startup_deadline is None`, and a nested level-3 inside an L4
    next-level child, a remote session worker and an MPI group worker are each
    given that deadline by whoever started them.

    Those are closed by the loop that owns them while their parent is already
    tearing down, so a refusal there reaches nobody.
    """
    worker = _l3(device_ids=[0, 1])
    worker._is_startup_root = False
    assert worker._chip_children_manage_workspace() is False


def test_a_worker_with_no_device_ids_has_no_chip_children_to_manage():
    """L4+ and a sub-worker-only Worker fork no chip child at all; `device_ids`
    is refused above level 3, so the predicate needs no level test of its own."""
    assert _l3()._chip_children_manage_workspace() is False


def test_the_simulation_platform_is_left_to_chip_worker_init():
    """The parent does not special-case sim, deliberately: `ChipWorker::init`
    resolves a simulated backend to the unsupported route before anything is
    requested, and a second test here could only drift from it."""
    worker = _l3(device_ids=[0], platform="a2a3sim")
    assert worker._chip_children_manage_workspace() is True


def test_the_chip_child_entry_carries_the_request_into_chip_worker_init(monkeypatch):
    """What the fork site passes is what `ChipWorker.init` is asked for."""
    seen, _buf = _run_chip_entry(monkeypatch, manage_workspace=True)
    assert seen["manage_workspace"] is True


def test_the_chip_child_entry_asks_for_nothing_when_the_parent_said_no(monkeypatch):
    """A descendant Worker's children keep exactly the path they had."""
    seen, _buf = _run_chip_entry(monkeypatch)
    assert seen["manage_workspace"] is False


def test_a_child_whose_init_failed_publishes_init_failed_and_returns(monkeypatch):
    """Management is requested before the device exists, so a failure to install
    it is an init failure like any other: the child publishes it and the
    parent's readiness barrier aborts startup, with no new recovery promised."""
    import simpler.worker as worker_mod

    _seen, buf = _run_chip_entry(monkeypatch, manage_workspace=True)
    state = worker_mod._mailbox_load_i32(worker_mod._buffer_field_addr(buf, worker_mod._OFF_STATE))
    assert state == worker_mod._INIT_FAILED


def test_a_failed_child_teardown_still_publishes_and_still_raises():
    """The close propagation this route relies on, unchanged: a managed child
    whose `finalize` reports a device teardown failure raises out of its one
    teardown call — the child then exits non-zero and the parent's existing reap
    reports it — and the observation is published from the `finally` either way.
    """
    import simpler.worker as worker_mod

    published: list[bool] = []

    class _FailingChip:
        def finalize(self):
            raise RuntimeError("device teardown failed (-7); keeping the context loaded")

        def teardown_report_bytes(self):
            # None is what a child with no captured observation returns, and
            # the publisher treats it as nothing to write.
            published.append(True)

    buf = memoryview(bytearray(worker_mod.MAILBOX_SIZE))
    with pytest.raises(RuntimeError, match="device teardown failed"):
        # A stand-in for the real ChipWorker, which needs a device context.
        worker_mod._finalize_chip_worker_and_publish(buf, cast("Any", _FailingChip()))
    assert published == [True]
