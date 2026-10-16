#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Which Workers let their endpoints size their own run concurrency, and which keep today's shape.

A Worker that configured none of the capacity knobs and owns its own admission asks its endpoints
to measure what they can hold: the capacity admitted then starts at one and grows only as a
resource commit proves a set. Every other shape is excluded, and each exclusion is a fact about
who has already decided rather than a preference — an explicit depth, queue bound or byte budget
is a caller's decision, and a Worker that is not the root of its own startup does not own the
admission its endpoints would be reporting to.

The identity directory is the separate number: how many runs can ever be *named*. It is finite,
internal, and much larger than the capacity, because a named-but-unproven identity costs one table
entry and no device memory.
"""

import pytest
from simpler.worker import (
    _AUTOMATIC_IDENTITY_DIRECTORY_SETS,
    _DEFAULT_PIPELINE_DEPTH_REQUEST,
    MAILBOX_REQUEST_FRAMES,
    MAILBOX_RUN_ENTRY_STRIDE,
    RemoteWorkerSpec,
    Worker,
    mailbox_size,
)


def _default_worker(**overrides):
    config = {"level": 3, "platform": "a2a3", "runtime": "host_build_graph", "device_ids": [0]}
    config.update(overrides)
    return Worker(**config)


def test_the_directory_is_a_64_kib_entry_table_not_a_device_budget():
    # Derived from the entry stride rather than written down, so it moves with the wire struct
    # instead of silently disagreeing with it. The table is the cost: ~384 B per identity.
    expected = (64 * 1024) // MAILBOX_RUN_ENTRY_STRIDE - MAILBOX_REQUEST_FRAMES
    assert _AUTOMATIC_IDENTITY_DIRECTORY_SETS == expected
    assert _AUTOMATIC_IDENTITY_DIRECTORY_SETS > _DEFAULT_PIPELINE_DEPTH_REQUEST
    # The whole directory's table stays inside the budget it was derived from.
    table_bytes = MAILBOX_RUN_ENTRY_STRIDE * (_AUTOMATIC_IDENTITY_DIRECTORY_SETS + MAILBOX_REQUEST_FRAMES)
    assert table_bytes <= 64 * 1024


def test_an_unconfigured_local_l3_worker_sizes_itself():
    worker = _default_worker()
    assert worker._pipeline_depth_scope_refusal() is None
    assert worker._automatic_admission_requested is True
    # The children are asked for the directory, not the old fixed default: the directory bounds
    # what can be named, and the capacity actually admitted is measured later.
    assert worker._requested_chip_pipeline_depth() == _AUTOMATIC_IDENTITY_DIRECTORY_SETS


@pytest.mark.parametrize(
    "overrides",
    [
        {"pipeline_depth": 4},
        {"pending_run_depth": 2},
        {"launch_depth": 2},
    ],
    ids=["pipeline_depth", "pending_run_depth", "launch_depth"],
)
def test_a_caller_who_set_a_capacity_knob_keeps_it(overrides):
    # Already decided: deriving a capacity here would override the caller's own figure.
    worker = _default_worker(**overrides)
    assert worker._automatic_admission_requested is False


def test_a_caller_who_set_a_byte_budget_keeps_todays_behaviour():
    # The byte policy is the caller's too, and that route refuses chip children anyway — so this
    # asserts the predicate rather than a startup that cannot happen.
    worker = _default_worker()
    worker._config["workspace_budget_bytes"] = 1 << 20
    assert worker._automatic_admission_requested is False


@pytest.mark.parametrize("platform", ["a2a3sim", "a5sim"])
def test_a_simulated_platform_is_excluded_here_and_not_inside_the_endpoint(platform):
    """The exclusion has to be the parent's, because the parent decides first.

    This answer sizes the mailbox entry table and seeds the FIFO bound before any child reports
    anything, so leaving it to the endpoint grants a simulated backend the whole directory — and
    an endpoint with no ledger to prove a set with then reports that grant straight back as what
    it can hold. Every simulated route therefore keeps the depth it had.
    """
    worker = _default_worker(platform=platform)
    assert worker._pipeline_depth_scope_refusal() is None, "the shape is in scope; the platform is what excludes it"
    assert worker._automatic_admission_requested is False
    assert worker._requested_chip_pipeline_depth() == _DEFAULT_PIPELINE_DEPTH_REQUEST


def test_an_explicit_depth_still_sizes_the_mailbox_to_that_depth():
    worker = _default_worker(pipeline_depth=3)
    assert worker._requested_chip_pipeline_depth() == 3
    assert worker._pipeline_depth_request == 3


def test_a_nested_worker_does_not_own_the_admission_it_would_report_to():
    parent = Worker(level=4, platform="a2a3", runtime="host_build_graph")
    child = _default_worker(device_ids=[1])
    parent.add_worker(child)
    assert child._automatic_admission_requested is False
    assert child._requested_chip_pipeline_depth() == _DEFAULT_PIPELINE_DEPTH_REQUEST


def test_a_worker_another_startup_initializes_does_not_size_itself():
    # What a remote or MPI session's inner Worker looks like: no parent object, only the record
    # that someone else's startup is driving it.
    worker = _default_worker()
    assert worker._automatic_admission_requested is True
    worker._is_startup_root = False
    assert worker._automatic_admission_requested is False
    assert worker._requested_chip_pipeline_depth() == _DEFAULT_PIPELINE_DEPTH_REQUEST


def test_a_parent_holding_a_remote_endpoint_does_not_size_itself():
    parent = Worker(level=4, platform="a2a3", runtime="host_build_graph", device_ids=[0])
    parent.add_remote_worker(RemoteWorkerSpec(endpoint="127.0.0.1:19077", platform="a2a3sim"))
    assert parent._automatic_admission_requested is False


@pytest.mark.parametrize("level", [2, 4])
def test_only_level_three_has_chip_endpoints_of_its_own_to_measure(level):
    worker = Worker(level=level, platform="a2a3", runtime="host_build_graph")
    assert worker._automatic_admission_requested is False


def test_the_directory_sized_mailbox_is_a_table_not_more_frames():
    # The whole point of the two numbers being allowed to differ: going from the old default to
    # the directory adds entry-table bytes only, and stays within one frame's worth of them.
    small = mailbox_size(_DEFAULT_PIPELINE_DEPTH_REQUEST)
    directory = mailbox_size(_AUTOMATIC_IDENTITY_DIRECTORY_SETS)
    added = directory - small
    assert added == MAILBOX_RUN_ENTRY_STRIDE * (_AUTOMATIC_IDENTITY_DIRECTORY_SETS - _DEFAULT_PIPELINE_DEPTH_REQUEST)
    assert added <= 64 * 1024
