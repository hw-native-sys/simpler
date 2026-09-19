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
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]


def _device_pool_class():
    """Load the real ``DevicePool`` from the root conftest, by path.

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
    return module.DevicePool


DevicePool = _device_pool_class()


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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
