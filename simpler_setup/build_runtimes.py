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
    python simpler_setup/build_runtimes.py                     # auto-detect platforms
    python simpler_setup/build_runtimes.py --platforms a2a3sim  # build specific platform
    python simpler_setup/build_runtimes.py --list               # list buildable platforms
"""

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

# Pre-install bootstrap: this script is invoked by CMake during `pip install .`
# before simpler_setup/simpler are on sys.path. Point at the source tree so
# `from simpler_setup...` (which eagerly imports `simpler` via its __init__.py)
# resolves.
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "python"))

from simpler_setup.platform_info import PROJECT_ROOT, discover_runtimes, parse_platform  # noqa: E402
from simpler_setup.runtime_builder import RuntimeBuilder  # noqa: E402

logger = logging.getLogger(__name__)


def detect_buildable_platforms() -> list:
    """Detect which platforms can be built with available toolchains.

    Returns:
        List of platform strings, e.g. ["a2a3sim", "a5sim"] when only gcc is available,
        or all four when the onboard cross-compiler is also present.
    """
    platforms = []

    # Sim platforms: only need gcc/g++
    if shutil.which("gcc") and shutil.which("g++"):
        platforms.extend(["a2a3sim", "a5sim"])

    # Onboard platforms: need ccec + cross-compiler from ASCEND_HOME_PATH.
    # a2a3 and a5 use the same toolchain and produce identical artifacts;
    # the difference is runtime-only, so always build both.
    has_ccec = shutil.which("ccec") is not None

    ascend_home = os.environ.get("ASCEND_HOME_PATH", "")
    cross_gxx = os.path.join(ascend_home, "tools", "hcc", "bin", "aarch64-target-linux-gnu-g++")
    has_cross = os.path.isfile(cross_gxx)

    if has_cross and has_ccec:
        platforms.extend(["a2a3", "a5"])

    return platforms


def build_all(
    lib_dir: Path,
    cache_dir: Path,
    platforms: Optional[list] = None,
    clone_protocol: str = "ssh",
) -> None:
    """Build all runtime variants for the given platforms.

    Args:
        lib_dir: Final binary output directory (lib/).
        cache_dir: Persistent cmake build directory (build/cache/).
        platforms: List of platform strings. None = auto-detect.
        clone_protocol: Protocol used by ensure_pto_isa_root() when an
            onboard platform needs the pto-isa headers and PTO_ISA_ROOT is
            not pre-set. Mirrors conftest's --clone-protocol flag.
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

    # a2a3 onboard host_runtime hard-depends on pto-isa headers + CANN-9.0
    # aclnn syms (cf. src/a2a3/platform/onboard/host/CMakeLists.txt
    # SIMPLER_ENABLE_PTO_SDMA_WORKSPACE marker). Resolve PTO_ISA_ROOT now so
    # the protocol declared on the CLI (and surfaced in the top-level
    # CMakeLists invocation) is the one actually used, instead of relying on
    # the fallback in RuntimeCompiler._init_a2a3. No-ops when PTO_ISA_ROOT
    # is already set. Skipped when only sim platforms are being built.
    if "a2a3" in platforms:
        from simpler_setup.pto_isa import ensure_pto_isa_root  # noqa: PLC0415

        os.environ["PTO_ISA_ROOT"] = ensure_pto_isa_root(clone_protocol=clone_protocol, verbose=True)

    # libsimpler_log.so and libcpu_sim_context.so are process-global (one per
    # host toolchain, not per arch/variant) — build them once before iterating
    # platforms. cpu_sim_context is only needed when building any sim platform.
    if platforms:
        logger.info("Building simpler_log (process-global)...")
        try:
            RuntimeBuilder(platform=platforms[0]).ensure_simpler_log(build=True)
        except Exception as e:
            logger.error(f"Failed to build simpler_log: {e}")
            raise

        sim_platforms = [p for p in platforms if parse_platform(p)[1] == "sim"]
        if sim_platforms:
            logger.info("Building cpu_sim_context (process-global)...")
            try:
                RuntimeBuilder(platform=sim_platforms[0]).ensure_sim_context(build=True)
            except Exception as e:
                logger.error(f"Failed to build cpu_sim_context: {e}")
                raise

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
    parser.add_argument(
        "--clone-protocol",
        choices=["ssh", "https"],
        default="ssh",
        help=(
            "Protocol for cloning pto-isa when an onboard a2a3 build needs it "
            "and PTO_ISA_ROOT is not pre-set (default: ssh, matching conftest)"
        ),
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
        clone_protocol=args.clone_protocol,
    )


if __name__ == "__main__":
    main()
