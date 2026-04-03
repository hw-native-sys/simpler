# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Pytest configuration for platform-aware testing."""

import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent

# ST files do module-level kernel compilation requiring g++/compiler toolchain.
# Exclude from default pytest collection; run ST via: python tests/st/<test>.py
collect_ignore_glob = ["st/*"]

# Make tools/ importable so we can use the shared discovery module
_tools_dir = str(PROJECT_ROOT / "tools")
if _tools_dir not in sys.path:
    sys.path.insert(0, _tools_dir)

from test_catalog import (  # noqa: E402
    arch_from_platform,
    discover_platforms,
    discover_runtimes_for_arch,
)


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "requires_hardware: test needs Ascend toolchain and real device")


def pytest_addoption(parser):
    """Add custom command-line options for pytest."""
    parser.addoption(
        "--platform",
        action="store",
        default=None,
        help="Platform to test (e.g., a2a3sim, a5sim). If not specified, tests all platforms.",
    )


@pytest.fixture
def default_test_platform(request):
    """Provide the default platform for discovery tests.

    Returns the platform specified via --platform, or defaults to a2a3sim.
    """
    platform = request.config.getoption("--platform")
    if platform:
        return platform

    # Default to a2a3sim for backward compatibility
    return "a2a3sim"


@pytest.fixture
def test_arch(default_test_platform):
    """Extract architecture name from platform (e.g., 'a2a3sim' -> 'a2a3')."""
    return arch_from_platform(default_test_platform)


def pytest_generate_tests(metafunc):
    """Dynamically parametrize integration tests based on available platforms and runtimes.

    This hook is called for each test function. If the test has 'platform' and 'runtime_name'
    parameters, we parametrize it with all valid platform×runtime combinations.
    """
    if "platform" in metafunc.fixturenames and "runtime_name" in metafunc.fixturenames:
        # Get platform filter from command line
        platform_filter = metafunc.config.getoption("--platform")

        # Discover available platforms
        if platform_filter:
            platforms = [platform_filter]
        else:
            platforms = discover_platforms()

        # Build platform×runtime combinations
        test_params = []
        for platform in platforms:
            arch = arch_from_platform(platform)
            runtimes = discover_runtimes_for_arch(arch)

            for runtime in runtimes:
                # Mark hardware platforms (non-sim) as requiring Ascend
                marks = []
                if not platform.endswith("sim"):
                    marks.append(
                        pytest.mark.skipif(
                            not os.getenv("ASCEND_HOME_PATH"),
                            reason=f"ASCEND_HOME_PATH not set; Ascend toolkit required for {platform}",
                        )
                    )

                test_params.append(pytest.param(platform, runtime, marks=marks, id=f"{platform}-{runtime}"))

        # Apply parametrization
        metafunc.parametrize("platform,runtime_name", test_params)


def pytest_collection_modifyitems(session, config, items):
    """Add skip markers to tests based on platform/architecture constraints.

    This hook runs after test collection and can dynamically add markers to tests.
    """
    platform_filter = config.getoption("--platform")

    # If no platform specified, use default
    if not platform_filter:
        platform_filter = "a2a3sim"

    arch = arch_from_platform(platform_filter)
    available_runtimes = discover_runtimes_for_arch(arch)

    for item in items:
        # Skip aicpu_build_graph tests for architectures that don't have it
        if "test_discovers_aicpu_build_graph" in item.nodeid:
            if "aicpu_build_graph" not in available_runtimes:
                item.add_marker(pytest.mark.skip(reason=f"aicpu_build_graph not available for {arch} architecture"))

        # Skip tensormap_and_ringbuffer tests for architectures that don't have it
        if "tensormap_and_ringbuffer" in item.nodeid:
            if "tensormap_and_ringbuffer" not in available_runtimes:
                item.add_marker(
                    pytest.mark.skip(reason=f"tensormap_and_ringbuffer not available for {arch} architecture")
                )
