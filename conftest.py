# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Root conftest — CLI options, markers, ST platform filtering, runtime isolation, and ST fixtures.

Runtime isolation: CANN's AICPU framework caches the user .so per device context.
Switching runtimes on the same device within one process causes hangs. When multiple
runtimes are collected and --runtime is not specified, pytest_runtestloop spawns a
subprocess per runtime so each gets a clean CANN context. See docs/testing.md.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest


def _parse_device_range(s: str) -> list[int]:
    """Parse '4-7' -> [4,5,6,7] or '0' -> [0]."""
    if "-" in s:
        start, end = s.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return [int(s)]


class DevicePool:
    """Device allocator for pytest fixtures.

    Manages a fixed set of device IDs. Tests allocate IDs before use
    and release them after. Works identically for sim and onboard.
    """

    def __init__(self, device_ids: list[int]):
        self._available = list(device_ids)

    def allocate(self, n: int = 1) -> list[int]:
        if n > len(self._available):
            return []
        allocated = self._available[:n]
        self._available = self._available[n:]
        return allocated

    def release(self, ids: list[int]) -> None:
        self._available.extend(ids)


_device_pool: DevicePool | None = None


def pytest_addoption(parser):
    """Register CLI options."""
    parser.addoption("--platform", action="store", default=None, help="Target platform (e.g., a2a3sim, a2a3)")
    parser.addoption("--device", action="store", default="0", help="Device ID or range (e.g., 0, 4-7)")
    parser.addoption("--case", action="store", default=None, help="Run specific case name only")
    parser.addoption("--all-cases", action="store_true", default=False, help="Include manual cases")
    parser.addoption("--runtime", action="store", default=None, help="Only run tests for this runtime")
    parser.addoption("--rounds", type=int, default=1, help="Run each case N times (default: 1)")
    parser.addoption(
        "--skip-golden", action="store_true", default=False, help="Skip golden comparison (benchmark mode)"
    )
    parser.addoption(
        "--enable-profiling", action="store_true", default=False, help="Enable profiling (first round only)"
    )
    parser.addoption("--build", action="store_true", default=False, help="Compile runtime from source")


def pytest_configure(config):
    """Register custom markers and apply global config."""
    config.addinivalue_line("markers", "platforms(list): supported platforms for standalone ST functions")
    config.addinivalue_line("markers", "requires_hardware: test needs Ascend toolchain and real device")
    config.addinivalue_line("markers", "device_count(n): number of NPU devices needed")

    log_level = config.getoption("--log-level", default=None)
    if log_level:
        os.environ["PTO_LOG_LEVEL"] = log_level


def pytest_collection_modifyitems(session, config, items):
    """Skip ST tests based on --platform and --runtime filters, and order L3 before L2."""
    platform = config.getoption("--platform")
    runtime_filter = config.getoption("--runtime")

    # Sort: L3 tests first (they fork child processes that inherit main process CANN state,
    # so they must run before L2 tests pollute the CANN context).
    def sort_key(item):
        cls = getattr(item, "cls", None)
        level = getattr(cls, "_st_level", 0) if cls else 0
        return (0 if level >= 3 else 1, item.nodeid)

    items.sort(key=sort_key)

    for item in items:
        cls = getattr(item, "cls", None)
        if cls and hasattr(cls, "CASES") and isinstance(cls.CASES, list):
            if not platform:
                item.add_marker(pytest.mark.skip(reason="--platform required"))
            elif not any(platform in c.get("platforms", []) for c in cls.CASES):
                item.add_marker(pytest.mark.skip(reason=f"No cases for {platform}"))
            elif runtime_filter and getattr(cls, "_st_runtime", None) != runtime_filter:
                item.add_marker(
                    pytest.mark.skip(reason=f"Runtime {getattr(cls, '_st_runtime', '?')} != {runtime_filter}")
                )
            continue
        platforms_marker = item.get_closest_marker("platforms")
        if platforms_marker:
            if not platform:
                item.add_marker(pytest.mark.skip(reason="--platform required"))
            elif platform not in platforms_marker.args[0]:
                item.add_marker(pytest.mark.skip(reason=f"Not supported on {platform}"))


# ---------------------------------------------------------------------------
# Runtime isolation: spawn subprocess per runtime
# ---------------------------------------------------------------------------


def _collect_st_runtimes(items):
    """Return sorted list of unique runtimes from collected SceneTestCase items."""
    runtimes = set()
    for item in items:
        cls = getattr(item, "cls", None)
        rt = getattr(cls, "_st_runtime", None) if cls else None
        if rt:
            runtimes.add(rt)
    return sorted(runtimes)


def pytest_runtestloop(session):
    """Override test execution to isolate runtimes in subprocesses.

    If --runtime is specified (or only one runtime collected), run normally.
    Otherwise, spawn one subprocess per runtime and aggregate results.
    """
    runtime_filter = session.config.getoption("--runtime")
    if runtime_filter:
        return  # single runtime — let pytest run normally

    runtimes = _collect_st_runtimes(session.items)
    if len(runtimes) <= 1:
        return  # zero or one runtime — no isolation needed

    # Multiple runtimes: spawn subprocess per runtime
    # Re-invoke pytest with the same args + --runtime <rt> for each runtime
    base_args = [sys.executable, "-m", "pytest"]
    for arg in session.config.invocation_params.args:
        base_args.append(str(arg))

    failed = False
    for rt in runtimes:
        # Build subprocess command: inject --runtime <rt>
        cmd = base_args + ["--runtime", rt]
        header = f"  Runtime: {rt}"
        print(f"\n{'=' * 60}\n{header}\n{'=' * 60}\n", flush=True)

        result = subprocess.run(cmd, check=False, cwd=session.config.invocation_params.dir)
        if result.returncode != 0:
            failed = True
            print(f"\n*** Runtime {rt}: FAILED ***\n", flush=True)
        else:
            print(f"\n--- Runtime {rt}: PASSED ---\n", flush=True)

    if failed:
        session.testsfailed = 1
    else:
        session.testscollected = sum(1 for _ in session.items)
        session.testsfailed = 0

    return True  # returning True prevents default runtestloop


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def device_pool(request):
    """Session-scoped device pool parsed from --device."""
    global _device_pool  # noqa: PLW0603
    if _device_pool is None:
        raw = request.config.getoption("--device")
        _device_pool = DevicePool(_parse_device_range(raw))
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
    build = request.config.getoption("--build", default=False)

    if level == 2:
        ids = device_pool.allocate(1)
        if not ids:
            pytest.fail("no devices available")

        from simpler.worker import Worker  # noqa: PLC0415

        w = Worker(level=2, device_id=ids[0], platform=st_platform, runtime=runtime, build=build)
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

        w = Worker(
            level=3,
            device_ids=ids,
            num_sub_workers=max_subs,
            platform=st_platform,
            runtime=runtime,
            build=build,
        )

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
