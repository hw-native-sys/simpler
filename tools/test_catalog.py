"""Unified platform, runtime, and test case discovery for tests and CI.

Single source of truth for scanning the project directory structure.
Used by both pytest (via conftest.py) and CI scripts (via CLI).

Test case sources:
  - examples/   : Small examples that run on both sim and onboard.
  - tests/st/   : Hardware-only scene tests for large-scale scenarios.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Platform / runtime discovery
# ---------------------------------------------------------------------------


def discover_platforms(project_root: Path | None = None) -> list[str]:
    """Discover available platforms by scanning src/*/platform/{onboard,sim}/ directories.

    Args:
        project_root: Root of the project tree. Defaults to the repo root.

    Returns:
        List of platform names (e.g., ["a2a3", "a2a3sim", "a5", "a5sim"])
    """
    root = project_root or PROJECT_ROOT
    platforms = []
    src_dir = root / "src"

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


def discover_runtimes_for_arch(arch: str, project_root: Path | None = None) -> list[str]:
    """Discover available runtimes for a specific architecture.

    Args:
        arch: Architecture name (e.g., "a2a3", "a5")
        project_root: Root of the project tree. Defaults to the repo root.

    Returns:
        List of runtime names (e.g., ["host_build_graph", "aicpu_build_graph"])
    """
    root = project_root or PROJECT_ROOT
    runtime_dir = root / "src" / arch / "runtime"

    if not runtime_dir.exists():
        return []

    runtimes = []
    for item in sorted(runtime_dir.iterdir()):
        if item.is_dir() and (item / "build_config.py").exists():
            runtimes.append(item.name)

    return runtimes


def arch_from_platform(platform: str) -> str:
    """Extract architecture name from platform string.

    Args:
        platform: Platform name (e.g., "a2a3sim", "a5")

    Returns:
        Architecture name (e.g., "a2a3", "a5")
    """
    if platform.endswith("sim"):
        return platform[:-3]
    return platform


# ---------------------------------------------------------------------------
# Test case discovery
# ---------------------------------------------------------------------------


@dataclass
class TestCase:
    """A discoverable test case (example or scene test)."""

    name: str       # Relative name: "a2a3/host_build_graph/vector_example"
    dir: str        # Full path to the case directory
    arch: str       # Architecture: "a2a3", "a5"
    runtime: str    # Runtime: "host_build_graph", etc.
    source: str     # "example" or "st"


def _scan_case_dir(base_dir: Path, source: str) -> list[TestCase]:
    """Scan a directory tree for valid test cases.

    A valid case is any directory containing both:
      - kernels/kernel_config.py
      - golden.py

    The directory structure must be: {base_dir}/{arch}/{runtime}/{case_name}/

    Args:
        base_dir: Root directory to scan (e.g., examples/ or tests/st/).
        source: Label for the source ("example" or "st").

    Returns:
        Sorted list of TestCase objects.
    """
    if not base_dir.is_dir():
        return []

    cases: list[TestCase] = []

    for arch_dir in sorted(base_dir.iterdir()):
        if not arch_dir.is_dir() or arch_dir.name == "scripts":
            continue
        arch = arch_dir.name

        for runtime_dir in sorted(arch_dir.iterdir()):
            if not runtime_dir.is_dir():
                continue
            runtime = runtime_dir.name

            for case_dir in sorted(runtime_dir.iterdir()):
                if not case_dir.is_dir():
                    continue
                kernel_config = case_dir / "kernels" / "kernel_config.py"
                golden = case_dir / "golden.py"
                if kernel_config.is_file() and golden.is_file():
                    name = f"{arch}/{runtime}/{case_dir.name}"
                    cases.append(TestCase(
                        name=name,
                        dir=str(case_dir),
                        arch=arch,
                        runtime=runtime,
                        source=source,
                    ))

    return cases


def discover_test_cases(
    project_root: Path | None = None,
    *,
    platform: str | None = None,
    runtime: str | None = None,
    source: str | None = None,
) -> list[TestCase]:
    """Discover all test cases from examples/ and tests/st/.

    Args:
        project_root: Root of the project tree. Defaults to the repo root.
        platform: Filter by platform (e.g., "a2a3sim", "a5"). Matches arch.
        runtime: Filter by runtime name.
        source: Filter by source ("example" or "st"). None returns both.

    Returns:
        Sorted list of TestCase objects matching the filters.
    """
    root = project_root or PROJECT_ROOT
    cases: list[TestCase] = []

    if source is None or source == "example":
        cases.extend(_scan_case_dir(root / "examples", "example"))
    if source is None or source == "st":
        cases.extend(_scan_case_dir(root / "tests" / "st", "st"))

    # Apply platform filter (match arch)
    if platform:
        target_arch = arch_from_platform(platform)
        platform_runtimes = discover_runtimes_for_arch(target_arch, root)
        cases = [
            c for c in cases
            if c.arch == target_arch and c.runtime in platform_runtimes
        ]

    # Apply runtime filter
    if runtime:
        cases = [c for c in cases if c.runtime == runtime]

    return cases


# ---------------------------------------------------------------------------
# CLI interface — usable by ci.sh and other scripts
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Discover platforms, runtimes, and test cases."
    )
    sub = parser.add_subparsers(dest="command")

    # --- platforms ---
    sub.add_parser("platforms", help="List available platforms")

    # --- runtimes ---
    rt = sub.add_parser("runtimes", help="List runtimes for an architecture")
    rt.add_argument("--arch", required=True)

    # --- cases ---
    cases = sub.add_parser("cases", help="List discoverable test cases")
    cases.add_argument("--platform", default=None, help="Filter by platform")
    cases.add_argument("--runtime", default=None, help="Filter by runtime")
    cases.add_argument("--source", default=None, choices=["example", "st"],
                       help="Filter by source (example or st)")
    cases.add_argument("--format", default="text", choices=["text", "json"],
                       help="Output format")

    args = parser.parse_args()

    if args.command == "platforms":
        for p in discover_platforms():
            print(p)

    elif args.command == "runtimes":
        for r in discover_runtimes_for_arch(args.arch):
            print(r)

    elif args.command == "cases":
        results = discover_test_cases(
            platform=args.platform,
            runtime=args.runtime,
            source=args.source,
        )
        if args.format == "json":
            print(json.dumps([asdict(c) for c in results], indent=2))
        else:
            for c in results:
                print(f"{c.source}\t{c.name}\t{c.dir}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
