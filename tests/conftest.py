"""Pytest configuration for platform-aware testing."""

import os
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).parent.parent


def discover_platforms() -> list[str]:
    """Discover available platforms by scanning src/*/platform/{onboard,sim}/ directories.

    Returns:
        List of platform names (e.g., ["a2a3", "a2a3sim", "a5", "a5sim"])
    """
    platforms = []
    src_dir = PROJECT_ROOT / "src"

    if not src_dir.exists():
        return platforms

    for arch_dir in sorted(src_dir.iterdir()):
        if not arch_dir.is_dir():
            continue

        arch_name = arch_dir.name
        platform_dir = arch_dir / "platform"

        if not platform_dir.exists():
            continue

        # Check for onboard (hardware) platform
        if (platform_dir / "onboard").exists():
            platforms.append(arch_name)

        # Check for sim (simulation) platform
        if (platform_dir / "sim").exists():
            platforms.append(f"{arch_name}sim")

    return platforms


def discover_runtimes_for_arch(arch: str) -> list[str]:
    """Discover available runtimes for a specific architecture.

    Args:
        arch: Architecture name (e.g., "a2a3", "a5")

    Returns:
        List of runtime names (e.g., ["host_build_graph", "aicpu_build_graph"])
    """
    runtime_dir = PROJECT_ROOT / "src" / arch / "runtime"

    if not runtime_dir.exists():
        return []

    runtimes = []
    for item in sorted(runtime_dir.iterdir()):
        if item.is_dir() and (item / "build_config.py").exists():
            runtimes.append(item.name)

    return runtimes


def pytest_addoption(parser):
    """Add custom command-line options for pytest."""
    parser.addoption(
        "--platform",
        action="store",
        default=None,
        help="Platform to test (e.g., a2a3sim, a5sim). If not specified, tests all platforms."
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
    platform = default_test_platform
    if platform.endswith("sim"):
        return platform[:-3]
    return platform


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
            # Extract architecture from platform name
            arch = platform[:-3] if platform.endswith("sim") else platform

            # Discover runtimes for this architecture
            runtimes = discover_runtimes_for_arch(arch)

            # Add all valid combinations
            for runtime in runtimes:
                # Mark hardware platforms (non-sim) as requiring Ascend
                marks = []
                if not platform.endswith("sim"):
                    marks.append(pytest.mark.skipif(
                        not os.getenv("ASCEND_HOME_PATH"),
                        reason=f"ASCEND_HOME_PATH not set; Ascend toolkit required for {platform}"
                    ))

                test_params.append(pytest.param(
                    platform,
                    runtime,
                    marks=marks,
                    id=f"{platform}-{runtime}"
                ))

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

    # Extract architecture from platform
    arch = platform_filter[:-3] if platform_filter.endswith("sim") else platform_filter

    # Get available runtimes for this architecture
    available_runtimes = discover_runtimes_for_arch(arch)

    for item in items:
        # Skip aicpu_build_graph tests for architectures that don't have it
        if "test_discovers_aicpu_build_graph" in item.nodeid:
            if "aicpu_build_graph" not in available_runtimes:
                item.add_marker(pytest.mark.skip(
                    reason=f"aicpu_build_graph not available for {arch} architecture"
                ))

        # Skip tensormap_and_ringbuffer tests for architectures that don't have it
        if "tensormap_and_ringbuffer" in item.nodeid:
            if "tensormap_and_ringbuffer" not in available_runtimes:
                item.add_marker(pytest.mark.skip(
                    reason=f"tensormap_and_ringbuffer not available for {arch} architecture"
                ))
