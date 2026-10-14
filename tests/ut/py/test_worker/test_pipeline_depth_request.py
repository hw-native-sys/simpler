#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""`Worker(pipeline_depth=...)`: the request for native run-resource sets, and where it is honoured.

One set holds everything a run owns while it is in flight, so this bounds how many runs may be
prepared and accepted by the device at once. The default request is two, which is what every
backend granted before the key existed — raising a runtime's published maximum must not move it.

Three things are checked here: the constructor's validation, the default, and the provenance in
which a larger request is carried to the children at all. Most need no Worker started; the two
that do use a simulated backend, because a grant is decided in the native context and a
Worker's root-or-not role is decided by its own `init()`.
"""

import time

import pytest
from simpler.worker import (
    _DEFAULT_PIPELINE_DEPTH_REQUEST,
    _UINT32_MAX,
    RemoteWorkerSpec,
    Worker,
    _local_run_resource_set_count,
    _validated_pipeline_depth,
)

from ._harness import fake_chip_l3

# Above the default and above the three sets the old layout was built for, so a case that passes
# here is not passing on a width that happens to exist.
_LARGER_REQUEST = 6


def test_the_absent_key_requests_the_default_two():
    # Not whatever a runtime declares: a runtime declaring no limit must not hand an unconfigured
    # Worker a further resource set, which would move both its footprint and its behaviour.
    assert _DEFAULT_PIPELINE_DEPTH_REQUEST == 2
    assert Worker(level=3)._pipeline_depth_request == _DEFAULT_PIPELINE_DEPTH_REQUEST
    assert _validated_pipeline_depth({}, 3) == _DEFAULT_PIPELINE_DEPTH_REQUEST
    assert Worker(level=3)._pipeline_depth_requested_explicitly is False


@pytest.mark.parametrize("depth", [1, 2, 3, 8, 64, 1024])
def test_any_positive_request_is_carried_verbatim(depth):
    # No ceiling: every owner of a per-run resource is built to the grant, so a request is a
    # number this Worker carries to its child rather than one it clamps.
    worker = Worker(level=3, pipeline_depth=depth)
    assert worker._pipeline_depth_request == depth
    assert worker._pipeline_depth_requested_explicitly is True
    assert worker._mailbox_granted_sets == depth


@pytest.mark.parametrize("bad", [0, -1])
def test_a_request_below_one_is_refused(bad):
    with pytest.raises(ValueError, match="pipeline_depth must be >= 1"):
        Worker(level=3, pipeline_depth=bad)


@pytest.mark.parametrize("bad", [True, 2.0, "2", None])
def test_a_non_int_request_is_refused(bad):
    with pytest.raises(ValueError, match="pipeline_depth must be an int"):
        Worker(level=3, pipeline_depth=bad)


def test_the_key_is_refused_below_the_admission_fifo():
    # The runs being sized are the chip child's, and a Worker below the FIFO owns neither.
    with pytest.raises(ValueError, match="pipeline_depth requires a level >= 3 Worker"):
        Worker(level=2, pipeline_depth=2)


def _scoped_worker(**overrides):
    config = {
        "level": 3,
        "platform": "a2a3",
        "runtime": "host_build_graph",
        "device_ids": [0],
        "pipeline_depth": _LARGER_REQUEST,
    }
    config.update(overrides)
    return Worker(**config)


@pytest.mark.parametrize(
    "platform,runtime",
    [
        ("a2a3", "host_build_graph"),
        ("a5", "host_build_graph"),
        ("a2a3", "tensormap_and_ringbuffer"),
        ("a5", "tensormap_and_ringbuffer"),
        ("a2a3sim", "host_build_graph"),
        ("a5sim", "tensormap_and_ringbuffer"),
    ],
)
def test_every_backend_carries_the_larger_request(platform, runtime):
    # All four declare no limit of their own, so the request reaches the child whatever the
    # platform and runtime. What the child then grants is a separate answer, checked below.
    worker = _scoped_worker(platform=platform, runtime=runtime)
    assert worker._requested_chip_pipeline_depth() == _LARGER_REQUEST


def test_several_endpoints_and_sub_workers_still_carry_the_larger_request():
    # Each endpoint owns its own grant and its own mailbox, sized to this request, so neither a
    # second endpoint nor a sub-worker sharing this admission reduces what is asked for.
    assert _scoped_worker(device_ids=[0, 1])._requested_chip_pipeline_depth() == _LARGER_REQUEST
    assert _scoped_worker(num_sub_workers=1)._requested_chip_pipeline_depth() == _LARGER_REQUEST


@pytest.mark.parametrize("depth", [1, 2, 3, _LARGER_REQUEST, 64, 1024])
def test_the_grant_is_the_depth_the_child_resolved(depth):
    """No platform term, no runtime term, and no joined-launch term.

    The grant used to carry two meanings at once — the run-resource sets a child holds *and* the
    task frames a successor would occupy — so a route that could not order one native submission
    behind another was given a single set whatever it asked for. Those are separate counts now:
    request frames are a fixed pool that only carries hand-offs, and this one sizes the run-state
    entries a live run holds. Whether a backend joins launches is a different question with a
    different answer, and it no longer reaches this one.
    """
    assert _local_run_resource_set_count(depth) == depth


def test_a_grant_below_one_set_is_refused():
    # Every chip child holds at least the run it is executing, so zero is not a state to publish.
    with pytest.raises(ValueError, match="at least one run-resource set"):
        _local_run_resource_set_count(0)


def test_the_request_is_bounded_by_what_the_native_side_can_carry():
    """``Orchestrator::configure_pipeline_depth`` takes a ``uint32_t``, so the key is bounded by it.

    Checked at the boundary rather than at a round number: the largest accepted value and the
    first refused one are adjacent, so a bound that moved by one fails here.
    """
    assert _validated_pipeline_depth({"pipeline_depth": _UINT32_MAX}, 3) == _UINT32_MAX
    with pytest.raises(ValueError, match="pipeline_depth must be <="):
        _validated_pipeline_depth({"pipeline_depth": _UINT32_MAX + 1}, 3)


def test_an_unsupported_grant_fails_startup_with_both_numbers():
    # In scope and explicitly requested, so a child that granted only two is a shortfall the
    # caller is told about rather than silently served.
    worker = _scoped_worker(platform="a5")
    with pytest.raises(RuntimeError, match="was requested, but this configuration grants 2"):
        worker._granted_chip_pipeline_depth([2])


def test_a_default_request_accepts_whatever_the_children_granted():
    worker = Worker(level=3, platform="a2a3", runtime="host_build_graph", device_ids=[0])
    assert worker._granted_chip_pipeline_depth([2]) == 2


def test_an_invalid_published_depth_is_refused_whatever_was_requested():
    worker = Worker(level=3, platform="a2a3", runtime="host_build_graph", device_ids=[0])
    # Above what this Worker asked for: a grant can never exceed a request, so a child that
    # published one is reporting something this side cannot have mailbox frames for.
    with pytest.raises(RuntimeError, match="published invalid pipeline depths"):
        worker._granted_chip_pipeline_depth([worker._mailbox_granted_sets + 1])
    with pytest.raises(RuntimeError, match="published invalid pipeline depths"):
        worker._granted_chip_pipeline_depth([0])


# --------------------------------------------------------------------------------------------
# Routes that must not inherit the raised layout maximum
# --------------------------------------------------------------------------------------------


def test_a_level_two_worker_grants_the_standing_default_however_high_the_runtime_publishes():
    """The in-process level-2 route passes no request, so it must keep the capacity it had.

    Real init on a simulated backend, because the grant is decided in C++ from the runtime's own
    published contract: every runtime now declares no limit of its own, and a route that asks for
    nothing must still be granted two.
    """
    worker = Worker(level=2, platform="a2a3sim", runtime="host_build_graph", device_id=0)
    worker.init()
    try:
        chip_worker = worker._chip_worker
        assert chip_worker is not None
        granted = int(chip_worker.pipeline_depth)
    finally:
        worker.close()
    assert granted == _DEFAULT_PIPELINE_DEPTH_REQUEST, (
        f"a level-2 Worker was granted {granted} resource sets; it asks for nothing, so it must keep "
        f"the standing default of {_DEFAULT_PIPELINE_DEPTH_REQUEST}"
    )


@pytest.mark.parametrize(
    "overrides",
    [{"level": 4}],
    ids=["level4"],
)
def test_an_explicit_request_outside_the_supported_shape_is_refused_before_anything_commits(overrides):
    # Refused, not reduced: the caller asked for more runs in flight and this configuration does
    # not own the admission that would order them, so it is told rather than silently served two.
    worker = _scoped_worker(**overrides)
    with pytest.raises(RuntimeError, match=f"pipeline_depth={_LARGER_REQUEST} is not supported by this configuration"):
        worker._requested_chip_pipeline_depth()


def test_a_parent_holding_a_remote_endpoint_is_already_out_of_scope():
    # The parent of a remote endpoint is at level 4 or above, which the level test refuses.
    parent = Worker(
        level=4, platform="a2a3", runtime="host_build_graph", device_ids=[0], pipeline_depth=_LARGER_REQUEST
    )
    parent.add_remote_worker(RemoteWorkerSpec(endpoint="127.0.0.1:19077", platform="a2a3sim"))
    with pytest.raises(RuntimeError, match=f"pipeline_depth={_LARGER_REQUEST} is not supported by this configuration"):
        parent._requested_chip_pipeline_depth()


def test_a_parent_holding_a_nested_worker_is_already_out_of_scope():
    # `add_worker` refuses to share a Worker with `device_ids`, so a parent that holds a nested
    # child has no local chip endpoint of its own — and is out of scope on the level test either
    # way.
    parent = Worker(level=4, platform="a2a3", runtime="host_build_graph", pipeline_depth=_LARGER_REQUEST)
    parent.add_worker(Worker(level=3, platform="a2a3", runtime="host_build_graph", device_ids=[1]))
    with pytest.raises(RuntimeError, match=f"pipeline_depth={_LARGER_REQUEST} is not supported by this configuration"):
        parent._requested_chip_pipeline_depth()


def test_a_level_three_child_of_a_level_four_parent_is_refused():
    """The level test constrains the *parent* of a nested child, never the child itself.

    A level-3 Worker in the supported shape is a legal child: `add_worker` refuses a parent below
    level 4 and a parent that has `device_ids` of its own, and this construction satisfies both.
    The child then passes every configuration test — level, platform, runtime, one endpoint, no
    sub-workers — so being attached is what has to refuse it.
    """
    parent = Worker(level=4, platform="a2a3", runtime="host_build_graph")
    child = _scoped_worker(device_ids=[1])
    parent.add_worker(child)

    assert child._pipeline_depth_scope_refusal() == (
        "this Worker is attached to a parent Worker, so it is not the root of its tree"
    )
    with pytest.raises(RuntimeError, match=f"pipeline_depth={_LARGER_REQUEST} is not supported by this configuration"):
        child._requested_chip_pipeline_depth()


def test_a_worker_another_startup_initializes_is_refused():
    """A remote or MPI session's inner Worker holds no parent object, only a startup deadline.

    `remote_l3_session.py` and `mpi_l3_session.py` each call `inner_worker.init(...)` with the
    session's deadline, and `init()` records that as not being this Worker's own startup. The
    inner Worker is otherwise in the supported shape, so that record is the only thing between it
    and a third resource set nothing in that route negotiated.
    """
    worker = _scoped_worker()
    assert worker._topology_parent is None
    assert worker._pipeline_depth_scope_refusal() is None

    worker._is_startup_root = False

    assert worker._pipeline_depth_scope_refusal() == (
        "this Worker is initialized by another startup, so it is not the root of its tree"
    )
    with pytest.raises(RuntimeError, match=f"pipeline_depth={_LARGER_REQUEST} is not supported by this configuration"):
        worker._requested_chip_pipeline_depth()


def test_a_startup_with_a_deadline_is_not_its_own_root(monkeypatch):
    """What sets the flag the case above reads: an `init()` that another startup is driving.

    This is the call both session routes make, and the only difference from an ordinary `init()`
    is the deadline argument.
    """
    with fake_chip_l3(monkeypatch, init=False) as worker:
        assert worker._is_startup_root is True
        worker.init(_startup_deadline=time.monotonic() + 20.0)
        assert worker._is_startup_root is False


@pytest.mark.parametrize(
    "overrides",
    [{"level": 4}, {"num_sub_workers": 1}, {"device_ids": [0, 1]}, {"platform": "a2a3sim"}],
    ids=["level4", "sub_workers", "two_endpoints", "sim"],
)
def test_those_same_routes_ask_for_the_default_when_nothing_was_requested(overrides):
    # The unset key never fails a route: it asks for what that route always granted.
    config = {"level": 3, "platform": "a2a3", "runtime": "host_build_graph", "device_ids": [0]}
    config.update(overrides)
    worker = Worker(**config)
    assert worker._requested_chip_pipeline_depth() == _DEFAULT_PIPELINE_DEPTH_REQUEST


def test_an_attached_child_asks_for_the_default_when_nothing_was_requested():
    parent = Worker(level=4, platform="a2a3", runtime="host_build_graph")
    child = Worker(level=3, platform="a2a3", runtime="host_build_graph", device_ids=[1])
    parent.add_worker(child)
    assert child._requested_chip_pipeline_depth() == _DEFAULT_PIPELINE_DEPTH_REQUEST


def test_the_supported_shape_reports_no_refusal():
    assert _scoped_worker()._pipeline_depth_scope_refusal() is None
