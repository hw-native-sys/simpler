# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Root conftest — CLI options, markers, ST platform filtering, and ST fixtures."""

from __future__ import annotations

import pytest


def _parse_device_range(s: str) -> list[int]:
    """Parse '4-7' -> [4,5,6,7] or '0' -> [0]."""
    if "-" in s:
        start, end = s.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return [int(s)]


class DevicePool:
    """Simple device allocator for pytest fixtures.

    On sim platforms, device IDs are virtual — allocate always succeeds.
    On real hardware, IDs are exclusive.
    """

    def __init__(self, device_ids: list[int], *, is_sim: bool = False):
        self._available = list(device_ids)
        self._is_sim = is_sim
        self._sim_next = 0

    def allocate(self, n: int = 1) -> list[int]:
        if self._is_sim:
            ids = list(range(self._sim_next, self._sim_next + n))
            self._sim_next += n
            return ids
        if n > len(self._available):
            return []
        allocated = self._available[:n]
        self._available = self._available[n:]
        return allocated

    def release(self, ids: list[int]) -> None:
        if not self._is_sim:
            self._available.extend(ids)


_device_pool: DevicePool | None = None


def pytest_addoption(parser):
    """Register CLI options."""
    parser.addoption("--platform", action="store", default=None, help="Target platform (e.g., a2a3sim, a2a3)")
    parser.addoption("--device", action="store", default="0", help="Device ID or range (e.g., 0, 4-7)")
    parser.addoption("--case", action="store", default=None, help="Run specific case name only")
    parser.addoption("--all-cases", action="store_true", default=False, help="Include manual cases")


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "platforms(list): supported platforms for standalone ST functions")
    config.addinivalue_line("markers", "requires_hardware: test needs Ascend toolchain and real device")
    config.addinivalue_line("markers", "device_count(n): number of NPU devices needed")


def pytest_collection_modifyitems(session, config, items):
    """Skip ST tests based on --platform filter."""
    platform = config.getoption("--platform")
    for item in items:
        # SceneTestCase subclass: skip if no case matches current platform
        cls = getattr(item, "cls", None)
        if cls and hasattr(cls, "CASES") and isinstance(cls.CASES, list):
            if not platform:
                item.add_marker(pytest.mark.skip(reason="--platform required"))
            elif not any(platform in c.get("platforms", []) for c in cls.CASES):
                item.add_marker(pytest.mark.skip(reason=f"No cases for {platform}"))
            continue
        # Standalone function with @pytest.mark.platforms([...])
        platforms_marker = item.get_closest_marker("platforms")
        if platforms_marker:
            if not platform:
                item.add_marker(pytest.mark.skip(reason="--platform required"))
            elif platform not in platforms_marker.args[0]:
                item.add_marker(pytest.mark.skip(reason=f"Not supported on {platform}"))


@pytest.fixture(scope="session")
def device_pool(request):
    """Session-scoped device pool parsed from --device."""
    global _device_pool  # noqa: PLW0603
    if _device_pool is None:
        raw = request.config.getoption("--device")
        platform = request.config.getoption("--platform")
        is_sim = platform is None or platform.endswith("sim")
        _device_pool = DevicePool(_parse_device_range(raw), is_sim=is_sim)
    return _device_pool


@pytest.fixture(scope="session")
def st_platform(request):
    """Platform from --platform CLI flag."""
    p = request.config.getoption("--platform")
    if not p:
        pytest.skip("--platform required for ST tests")
    return p


@pytest.fixture()
def st_worker(request, st_platform, device_pool):
    """Per-test Worker with devices allocated from pool.

    Reads _st_level and CASES from the test class to determine
    how many devices and sub-workers to allocate.
    """
    cls = request.node.cls
    if cls is None or not hasattr(cls, "_st_level"):
        pytest.skip("st_worker requires SceneTestCase")

    level = cls._st_level
    runtime = cls._st_runtime

    if level == 2:
        ids = device_pool.allocate(1)
        if not ids:
            pytest.fail("no devices available")

        from simpler.worker import Worker  # noqa: PLC0415

        w = Worker(level=2, device_id=ids[0], platform=st_platform, runtime=runtime)
        w.init()
        yield w
        w.close()
        device_pool.release(ids)

    elif level == 3:
        max_devices = max((c.get("config", {}).get("device_count", 1) for c in cls.CASES), default=1)
        max_subs = max((c.get("config", {}).get("num_sub_workers", 0) for c in cls.CASES), default=0)
        ids = device_pool.allocate(max_devices)
        if not ids:
            pytest.fail(f"need {max_devices} devices")

        from simpler.worker import Worker  # noqa: PLC0415

        w = Worker(level=3, device_ids=ids, num_sub_workers=max_subs, platform=st_platform, runtime=runtime)

        # Register SubCallable entries from cls.CALLABLE
        sub_ids = {}
        for entry in cls.CALLABLE.get("callables", []):
            if "callable" in entry:
                cid = w.register(entry["callable"])
                sub_ids[entry["name"]] = cid
        cls._st_sub_ids = sub_ids

        w.init()
        yield w
        w.close()
        device_pool.release(ids)


@pytest.fixture()
def st_device_ids(request, device_pool):
    """Allocate device IDs. Use @pytest.mark.device_count(n) to request multiple."""
    marker = request.node.get_closest_marker("device_count")
    n = marker.args[0] if marker else 1
    ids = device_pool.allocate(n)
    if not ids:
        pytest.fail(f"need {n} devices")
    yield ids
    device_pool.release(ids)
