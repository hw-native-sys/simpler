# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""``DevicePool`` refuses rather than queues, so fixture lifetimes must not overlap.

A scene-test worker fixture allocates from this pool and holds its device for the
fixture's whole lifetime. The pool has no wait: a request it cannot satisfy comes
back empty, and the worker fixture turns that into an outright failure. So two
worker fixtures whose lifetimes overlap need two devices, and on the single-device
default (``--device 0``) the second one simply cannot be built.

That is why an intentional-failure scene case gets its own *class* rather than a
finer-scoped fixture inside a class that already holds a worker: class scope ends
before the next class begins, so the two lifetimes are sequential and one device
is enough.

The shared L2 fixture returns its DevicePool counter immediately while keeping
its Worker alive. A dedicated fixture must therefore retire that pooled owner
before initializing, rather than treating an available counter as cleanup proof.
The controls below invoke the real fixtures and scene method; only Worker
transport is substituted. Native cleanup proof is checked by the ChipWorker tests.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = Path(__file__).resolve().parents[3]


def _load_root_conftest():
    """Load the real root fixture definitions by path.

    Imported by location rather than by name: ``conftest`` is ambiguous with the
    per-directory conftests pytest has already imported.
    """
    conftest = _ROOT / "conftest.py"
    spec = importlib.util.spec_from_file_location("_root_conftest_for_pool_test", conftest)
    # Both are optional in the import protocol and both mean the same thing here:
    # the root conftest could not be prepared for execution, so there is nothing
    # to read DevicePool out of. Fail with the path rather than proceeding into
    # an AttributeError on None.
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load the root conftest for its DevicePool: {conftest}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ROOT_FIXTURES = _load_root_conftest()
DevicePool = _ROOT_FIXTURES.DevicePool


def test_one_device_serves_sequential_lifetimes():
    """acquire -> release -> acquire succeeds on a capacity of one."""
    pool = DevicePool([0])

    first = pool.allocate(1)
    assert first == [0]
    pool.release(first)

    second = pool.allocate(1)
    assert second == [0], "a released device has to be available to the next fixture"
    pool.release(second)


def test_one_device_refuses_overlapping_lifetimes():
    """A second allocation while the first is still held comes back empty.

    This is the shape a per-test worker fixture would take inside a class that
    already holds a class-scoped worker: the class fixture is still cached, so its
    device is still out, and the pool refuses instead of waiting.
    """
    pool = DevicePool([0])

    held = pool.allocate(1)
    assert held == [0]

    assert pool.allocate(1) == [], "the pool refuses rather than queueing, so overlap is not survivable"

    pool.release(held)
    assert pool.allocate(1) == [0]


def test_refusal_is_by_availability_not_by_capacity():
    """Two devices serve one overlap; a third request still comes back empty."""
    pool = DevicePool([0, 1])

    a = pool.allocate(1)
    b = pool.allocate(1)
    assert sorted(a + b) == [0, 1]

    assert pool.allocate(1) == []

    pool.release(a)
    assert pool.allocate(1) == a


def test_a_multi_device_request_is_all_or_nothing():
    """A request larger than what is free yields nothing, not a partial set."""
    pool = DevicePool([0, 1])

    held = pool.allocate(1)
    assert pool.allocate(2) == [], "a partial allocation would hand out a device twice"
    assert len(held) == 1

    pool.release(held)
    # Order is not part of the contract — availability is.
    assert sorted(pool.allocate(2)) == [0, 1]


def _load_scene_module(name, path):
    spec = importlib.util.spec_from_file_location(name, _ROOT / path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _fixture_value(fixture, values, lifetime):
    """Invoke real fixture generators; substitute only their explicit dependencies."""
    function = fixture.__wrapped__
    for name in inspect.signature(function).parameters:
        if name not in values:
            values[name] = _fixture_value(getattr(_ROOT_FIXTURES, name), values, lifetime)
    value = function(**{name: values[name] for name in inspect.signature(function).parameters})
    if inspect.isgenerator(value):

        def finish():
            try:
                next(value)
            except StopIteration:
                return
            finally:
                value.close()
            raise AssertionError("fixture yielded more than once")

        lifetime.callback(finish)
        return next(value)
    return value


@pytest.fixture
def worker_transport(monkeypatch):
    class WorkerTransport:
        owners = {}
        trace = []
        created = 0
        next_init_error = None
        next_close_error = None

        def __init__(self, level, **config):
            assert level == 2
            type(self).created += 1
            self.identity = type(self).created
            self._config = config
            self.device = config["device_id"]
            self.close_error = type(self).next_close_error
            self.init_error = type(self).next_init_error
            self.registrations = []
            self.runs = 0

        def init(self):
            self.trace.append(("init", self.identity))
            if self.device in self.owners:
                raise RuntimeError(f"device {self.device} already has an execution Worker in this process")
            self.owners[self.device] = self.identity
            if self.init_error is not None:
                raise self.init_error

        def close(self):
            self.trace.append(("close", self.identity))
            if self.close_error is not None:
                raise self.close_error
            if self.owners.get(self.device) == self.identity:
                del self.owners[self.device]

        def register(self, target):
            self.registrations.append(target)
            return target

        def unregister(self, handle):
            self.registrations.remove(handle)

    monkeypatch.setattr("simpler.worker.Worker", WorkerTransport)
    return WorkerTransport


def _pooled_fixture(lifetime, platform="a2a3", runtime="host_build_graph"):
    cls = type("Scene", (), {"_st_level": 2, "_st_runtime": runtime, "CASES": []})
    values = {
        "request": SimpleNamespace(node=SimpleNamespace(cls=cls), addfinalizer=lambda callback: None),
        "st_platform": platform,
        "device_pool": DevicePool([4]),
        "_l2_worker_pool": _ROOT_FIXTURES._L2WorkerPool(),
        "_l2_poisoned": set(),
    }
    return values, _fixture_value(_ROOT_FIXTURES.st_worker, values, lifetime)


_ISOLATED_FIXTURES = [
    f"tests/st/{arch}/{runtime}/prepared_callable/conftest.py"
    for arch in ("a2a3", "a5")
    for runtime in ("host_build_graph", "tensormap_and_ringbuffer")
] + [
    "tests/st/a2a3/host_build_graph/native_run_lifecycle/conftest.py",
    "tests/st/a2a3/host_build_graph/concurrent_prepare_stress/conftest.py",
]


@pytest.mark.parametrize("path", _ISOLATED_FIXTURES)
def test_isolated_scene_fixtures_retire_the_actual_pooled_owner(path, worker_transport):
    arch, runtime = path.split("/")[2:4]
    module = _load_scene_module("_isolated_fixture_" + path.replace("/", "_"), path)
    with ExitStack() as pooled_lifetime:
        values, pooled = _pooled_fixture(pooled_lifetime, arch, runtime)
        with ExitStack() as isolated_lifetime:
            isolated = _fixture_value(module.st_worker, values.copy(), isolated_lifetime)
            assert isolated is not pooled
            assert not isolated.registrations
            assert worker_transport.trace == [("init", 1), ("close", 1), ("init", 2)]
            assert values["device_pool"].allocate(1) == []
        assert worker_transport.trace[-1] == ("close", 2)
        assert not values["_l2_worker_pool"]
        assert not worker_transport.owners
        assert values["device_pool"]._available == [4]


def test_unleased_scene_keeps_first_generation_one_without_an_overlapping_owner(worker_transport, monkeypatch):
    scene = _load_scene_module(
        "_fresh_generation_scene", "tests/st/a2a3/host_build_graph/run_stream_reuse/test_run_stream_reuse.py"
    )
    monkeypatch.setattr(scene, "Worker", worker_transport, raising=False)
    case = scene.TestRunStreamReuseHbg()
    target = object()
    calls = []
    monkeypatch.setattr(case, "build_callable", lambda platform: target)
    monkeypatch.setattr(case, "_run_registered", lambda worker, handle, **kwargs: calls.append(("run", worker, handle)))
    monkeypatch.setattr(
        case,
        "_run_registered_with_lease",
        lambda worker, handle, **kwargs: calls.append(("lease", worker, handle, kwargs)),
    )
    with ExitStack() as lifetime:
        values, pooled = _pooled_fixture(lifetime)
        values["st_worker"] = pooled
        method = case.test_unleased_runs_do_not_consume_lease_generations
        method(**{name: values[name] for name in inspect.signature(method).parameters})
        assert worker_transport.trace == [("init", 1), ("close", 1), ("init", 2), ("close", 2)]
        assert [call[0] for call in calls] == ["run", "run", "run", "lease"]
        assert all(call[1] is calls[0][1] and call[2] is target for call in calls)
        assert calls[-1][3] == {"slot_id": 0, "generation": 1}
        assert not calls[0][1].registrations
        assert not worker_transport.owners
        # A later pooled method is free to acquire the device again.
        successor = _fixture_value(_ROOT_FIXTURES.st_worker, values, lifetime)
        assert successor is not pooled
        values["_l2_worker_pool"].retire(("host_build_graph", 4, False))


@pytest.mark.parametrize("entry", ["class_fixture", "scene_method", "pooled_runtime_switch"])
def test_close_refusal_stops_successor_init_and_retains_the_owner(entry, worker_transport):
    with ExitStack() as lifetime:
        values, pooled = _pooled_fixture(lifetime)
        failure = RuntimeError("close refused: native work is still outstanding")
        pooled.close_error = failure
        with pytest.raises(RuntimeError) as error:
            if entry == "class_fixture":
                module = _load_scene_module(
                    "_refused_native_fixture", "tests/st/a2a3/host_build_graph/native_run_lifecycle/conftest.py"
                )
                _fixture_value(module.st_worker, values, lifetime)
            elif entry == "scene_method":
                scene = _load_scene_module(
                    "_refused_generation_scene",
                    "tests/st/a2a3/host_build_graph/run_stream_reuse/test_run_stream_reuse.py",
                )
                method = scene.TestRunStreamReuseHbg().test_unleased_runs_do_not_consume_lease_generations
                method(**{name: values[name] for name in inspect.signature(method).parameters})
            else:
                values["request"].node.cls._st_runtime = "tensormap_and_ringbuffer"
                _fixture_value(_ROOT_FIXTURES.st_worker, values, lifetime)
        assert error.value is failure
        pool = values["_l2_worker_pool"]
        assert pool.retired == [pooled]
        assert worker_transport.created == 1
        assert worker_transport.owners == {4: 1}
        # Retained cleanup debt also blocks a later dedicated attempt before
        # constructing/initializing any replacement, even though the mapping
        # has already dropped the failed owner.
        with pytest.raises(RuntimeError, match="cleanup.*retained"):
            with pool.isolated("a2a3", "host_build_graph", values["device_pool"], values["_l2_poisoned"]):
                pytest.fail("failed cleanup admitted a replacement")
        assert worker_transport.created == 1
        pooled.close_error = None
        pooled.close()


@pytest.mark.parametrize("rollback_fails", [False, True])
def test_isolated_init_preserves_its_original_error_and_cleanup_owner(worker_transport, rollback_fails):
    pool = _ROOT_FIXTURES._L2WorkerPool()
    devices = DevicePool([4])
    original = RuntimeError("original native initialization error")
    cleanup = RuntimeError("rollback cleanup refusal")
    worker_transport.next_init_error = original
    worker_transport.next_close_error = cleanup if rollback_fails else None
    with pytest.raises(RuntimeError) as error:
        with pool.isolated("a2a3", "host_build_graph", devices, set()):
            pytest.fail("failed initialization yielded a Worker")
    assert error.value is original
    assert devices._available == [4]
    assert worker_transport.trace == [("init", 1), ("close", 1)]
    if rollback_fails:
        assert error.value.__cause__ is cleanup
        assert len(pool.retired) == 1
        assert worker_transport.owners == {4: 1}
        pool.retired[0].close_error = None
        pool.retired[0].close()
    else:
        assert not pool.retired
        assert not worker_transport.owners


def test_isolated_body_error_survives_a_failed_cleanup(worker_transport):
    pool = _ROOT_FIXTURES._L2WorkerPool()
    devices = DevicePool([4])
    original = AssertionError("original numerical mismatch")
    cleanup = RuntimeError("cleanup refused")
    with pytest.raises(BaseException) as error:
        with pool.isolated("a2a3", "host_build_graph", devices, set()) as worker:
            worker.close_error = cleanup
            raise original
    assert error.value is original
    assert error.value.__cause__ is cleanup
    assert pool.retired == [worker]
    assert worker_transport.owners == {4: 1}
    assert devices._available == [4]
    worker.close_error = None
    worker.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
