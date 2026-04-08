#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Pre-build all runtime variants for available platforms.

Detects available toolchains and builds all runtime binaries using
persistent build directories (build/cache/) for incremental compilation.
Final binaries are placed in build/lib/{arch}/{variant}/{runtime}/.

Usage:
    python examples/scripts/build_runtimes.py                     # auto-detect platforms
    python examples/scripts/build_runtimes.py --platforms a2a3sim  # build specific platform
    python examples/scripts/build_runtimes.py --list               # list buildable platforms
"""

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

# Ensure examples/scripts/ and python/ are on sys.path for imports
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent.parent
sys.path.insert(0, str(_script_dir))
sys.path.insert(0, str(_project_root / "python"))

from platform_info import PROJECT_ROOT, discover_runtimes, parse_platform  # noqa: E402
from runtime_builder import RuntimeBuilder  # noqa: E402

logger = logging.getLogger(__name__)


def detect_buildable_platforms() -> list:
    """Detect which platforms can be built with available toolchains.

    Returns:
        List of platform strings, e.g. ["a2a3sim", "a5sim"] on x86_64.
    """
    platforms = []

    # Sim platforms always buildable (only need gcc/g++)
    if shutil.which("gcc") and shutil.which("g++"):
        platforms.extend(["a2a3sim", "a5sim"])

    # Onboard platforms need ccec + aarch64 cross-compiler from ASCEND_HOME_PATH
    has_ccec = shutil.which("ccec") is not None

    ascend_home = os.environ.get("ASCEND_HOME_PATH", "")
    cross_gxx = os.path.join(ascend_home, "tools", "hcc", "bin", "aarch64-target-linux-gnu-g++")
    has_cross = os.path.isfile(cross_gxx)

    if has_cross and has_ccec:
        local_arch = _detect_local_chip_arch()
        if local_arch:
            platforms.append(local_arch)

    return platforms


def _detect_local_chip_arch() -> str:
    """Detect chip architecture on aarch64 hardware.

    Returns "a2a3" or "a5" based on environment or device files.
    Falls back to "a2a3" if detection is ambiguous.
    """
    # Explicit override
    env_arch = os.environ.get("PTO_ARCH")
    if env_arch in ("a2a3", "a5"):
        return env_arch

    # Default to a2a3 on aarch64 hardware
    return "a2a3"


def build_all(
    lib_dir: Path,
    cache_dir: Path,
    platforms: Optional[list] = None,
) -> None:
    """Build all runtime variants for the given platforms.

    Args:
        lib_dir: Final binary output directory (lib/).
        cache_dir: Persistent cmake build directory (build/cache/).
        platforms: List of platform strings. None = auto-detect.
    """
    # Override default paths to respect CLI args
    RuntimeBuilder._LIB_DIR = lib_dir
    RuntimeBuilder._CACHE_DIR = cache_dir

    if platforms is None:
        platforms = detect_buildable_platforms()

    if not platforms:
        logger.warning("No buildable platforms detected (missing gcc/g++?)")
        return

    logger.info(f"Building for platforms: {', '.join(platforms)}")

    for platform in platforms:
        arch, variant = parse_platform(platform)
        runtimes = discover_runtimes(arch)

        if not runtimes:
            logger.warning(f"  {platform}: no runtimes found, skipping")
            continue

        try:
            builder = RuntimeBuilder(platform=platform)
        except (ValueError, FileNotFoundError) as e:
            logger.warning(f"  {platform}: cannot initialize builder: {e}")
            continue

        if variant == "sim":
            logger.info(f"  Building {platform}/sim_context...")
            try:
                builder.ensure_sim_context(build=True)
            except Exception as e:
                logger.error(f"  Failed to build {platform}/sim_context: {e}")
                raise

        for runtime_name in runtimes:
            logger.info(f"  Building {platform}/{runtime_name}...")
            try:
                builder.get_binaries(runtime_name, build=True)
            except Exception as e:
                logger.error(f"  Failed to build {platform}/{runtime_name}: {e}")
                raise


def main():
    parser = argparse.ArgumentParser(description="Pre-build runtime binaries for available platforms")
    parser.add_argument(
        "--lib-dir",
        type=Path,
        default=PROJECT_ROOT / "build" / "lib",
        help="Output directory for final binaries (default: build/lib/)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=PROJECT_ROOT / "build" / "cache",
        help="Persistent cmake build directory (default: build/cache/)",
    )
    parser.add_argument(
        "--platforms",
        nargs="*",
        help="Platforms to build (default: auto-detect)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List buildable platforms and exit",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    if args.list:
        platforms = detect_buildable_platforms()
        if platforms:
            print("Buildable platforms:")
            for p in platforms:
                arch, variant = parse_platform(p)
                runtimes = discover_runtimes(arch)
                print(f"  {p}: {', '.join(runtimes) or '(no runtimes)'}")
        else:
            print("No buildable platforms detected")
        return

    build_all(
        lib_dir=args.lib_dir,
        cache_dir=args.cache_dir,
        platforms=args.platforms,
    )


if __name__ == "__main__":
    main()
