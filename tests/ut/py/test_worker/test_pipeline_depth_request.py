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
from _task_interface import PTO_PIPELINE_MAX_DEPTH
from simpler.worker import _DEFAULT_PIPELINE_DEPTH_REQUEST, RemoteWorkerSpec, Worker, _validated_pipeline_depth

from ._harness import fake_chip_l3


def test_the_absent_key_requests_the_default_two():
    # Not the supported maximum: a raised ceiling must not hand an unconfigured Worker a third
    # resource set, which would move both its footprint and its behaviour.
    assert _DEFAULT_PIPELINE_DEPTH_REQUEST == 2
    assert Worker(level=3)._pipeline_depth_request == _DEFAULT_PIPELINE_DEPTH_REQUEST
    assert _validated_pipeline_depth({}, 3) == _DEFAULT_PIPELINE_DEPTH_REQUEST
    assert Worker(level=3)._pipeline_depth_requested_explicitly is False


@pytest.mark.parametrize("depth", [1, 2, PTO_PIPELINE_MAX_DEPTH])
def test_a_request_within_the_ceiling_is_carried_verbatim(depth):
    worker = Worker(level=3, pipeline_depth=depth)
    assert worker._pipeline_depth_request == depth
    assert worker._pipeline_depth_requested_explicitly is True


def test_a_request_above_the_ceiling_is_refused_by_the_constructor():
    with pytest.raises(ValueError, match="pipeline_depth must be <="):
        Worker(level=3, pipeline_depth=PTO_PIPELINE_MAX_DEPTH + 1)


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
        "pipeline_depth": PTO_PIPELINE_MAX_DEPTH,
    }
    config.update(overrides)
    return Worker(**config)


@pytest.mark.skipif(PTO_PIPELINE_MAX_DEPTH < 3, reason="a third set needs a ceiling above two")
@pytest.mark.parametrize("platform", ["a2a3", "a5"])
def test_the_supported_configuration_carries_the_larger_request(platform):
    # Both onboard host_build_graph platforms declare three sets, so both carry the request to
    # their child. What the child then grants is a separate answer, checked below.
    assert _scoped_worker(platform=platform)._requested_chip_pipeline_depth() == PTO_PIPELINE_MAX_DEPTH


@pytest.mark.skipif(PTO_PIPELINE_MAX_DEPTH < 3, reason="a third set needs a ceiling above two")
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
    with pytest.raises(RuntimeError, match="published invalid pipeline depths"):
        worker._granted_chip_pipeline_depth([PTO_PIPELINE_MAX_DEPTH + 1])
    with pytest.raises(RuntimeError, match="published invalid pipeline depths"):
        worker._granted_chip_pipeline_depth([0])


# --------------------------------------------------------------------------------------------
# Routes that must not inherit the raised layout maximum
# --------------------------------------------------------------------------------------------


def test_a_level_two_worker_grants_the_standing_default_however_high_the_runtime_publishes():
    """The in-process level-2 route passes no request, so it must keep the capacity it had.

    Real init on a simulated backend, because the grant is decided in C++ from the runtime's own
    published contract: a2a3 host_build_graph now advertises the ceiling as what it *supports*,
    and a route that asks for nothing must still be granted two.
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


@pytest.mark.skipif(PTO_PIPELINE_MAX_DEPTH < 3, reason="a third set needs a ceiling above two")
@pytest.mark.parametrize(
    "overrides",
    [
        {"level": 4},
        {"num_sub_workers": 1},
        {"device_ids": [0, 1]},
        {"platform": "a2a3sim"},
        {"platform": "a5sim"},
        {"runtime": "tensormap_and_ringbuffer"},
    ],
    ids=["level4", "sub_workers", "two_endpoints", "a2a3sim", "a5sim", "tmr"],
)
def test_an_explicit_request_outside_the_supported_shape_is_refused_before_anything_commits(overrides):
    # Refused, not reduced: the caller asked for three runs in flight and this configuration does
    # not negotiate them, so it is told rather than silently served two.
    worker = _scoped_worker(**overrides)
    with pytest.raises(RuntimeError, match="pipeline_depth=3 is not supported by this configuration"):
        worker._requested_chip_pipeline_depth()


@pytest.mark.skipif(PTO_PIPELINE_MAX_DEPTH < 3, reason="a third set needs a ceiling above two")
def test_a_parent_holding_a_remote_endpoint_is_already_out_of_scope():
    # The parent of a remote endpoint is at level 4 or above, which the level test refuses.
    parent = Worker(level=4, platform="a2a3", runtime="host_build_graph", device_ids=[0], pipeline_depth=3)
    parent.add_remote_worker(RemoteWorkerSpec(endpoint="127.0.0.1:19077", platform="a2a3sim"))
    with pytest.raises(RuntimeError, match="pipeline_depth=3 is not supported by this configuration"):
        parent._requested_chip_pipeline_depth()


@pytest.mark.skipif(PTO_PIPELINE_MAX_DEPTH < 3, reason="a third set needs a ceiling above two")
def test_a_parent_holding_a_nested_worker_is_already_out_of_scope():
    # `add_worker` refuses to share a Worker with `device_ids`, so a parent that holds a nested
    # child has no local chip endpoint of its own — and is out of scope on the level test either
    # way.
    parent = Worker(level=4, platform="a2a3", runtime="host_build_graph", pipeline_depth=3)
    parent.add_worker(Worker(level=3, platform="a2a3", runtime="host_build_graph", device_ids=[1]))
    with pytest.raises(RuntimeError, match="pipeline_depth=3 is not supported by this configuration"):
        parent._requested_chip_pipeline_depth()


@pytest.mark.skipif(PTO_PIPELINE_MAX_DEPTH < 3, reason="a third set needs a ceiling above two")
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
    with pytest.raises(RuntimeError, match="pipeline_depth=3 is not supported by this configuration"):
        child._requested_chip_pipeline_depth()


@pytest.mark.skipif(PTO_PIPELINE_MAX_DEPTH < 3, reason="a third set needs a ceiling above two")
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
    with pytest.raises(RuntimeError, match="pipeline_depth=3 is not supported by this configuration"):
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
