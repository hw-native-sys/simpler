# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import fcntl
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from platform_info import TARGETS, load_build_config, parse_platform
from simpler.runtime_compiler import RuntimeCompiler

logger = logging.getLogger(__name__)


@dataclass
class RuntimeBinaries:
    """Paths to the compiled runtime binaries."""

    host_path: Path
    aicpu_path: Path
    aicore_path: Path
    sim_context_path: Optional[Path] = None


class RuntimeBuilder:
    """Discovers and builds runtime implementations from src/runtime/.

    Accepts a platform selection to provide correctly configured
    RuntimeCompiler and KernelCompiler instances. Runtime and platform
    are orthogonal — the same runtime (e.g., host_build_graph) can
    be compiled for any platform (e.g., a2a3, a2a3sim).
    """

    _CACHE_DIR = Path(__file__).parent.parent.parent / "build" / "cache"
    _LIB_DIR = Path(__file__).parent.parent.parent / "build" / "lib"

    # Defaults for compile_commands.json placement (matches old gen_compile_commands.py).
    # Platform dirs get compdb from the most feature-rich runtime;
    # runtime dirs get compdb from the onboard (real hardware) variant.
    _COMPDB_RUNTIME = "tensormap_and_ringbuffer"
    _COMPDB_VARIANT = "onboard"

    def __init__(self, platform: str = "a2a3"):
        """
        Initialize RuntimeBuilder with platform selection.

        Args:
            platform: Target platform ("a2a3", "a2a3sim", "a5", or "a5sim")
        """
        self.platform = platform
        self._arch, self._variant = parse_platform(platform)

        runtime_root = Path(__file__).parent.parent.parent
        self.runtime_root = runtime_root

        self.runtime_dir = runtime_root / "src" / self._arch / "runtime"

        # Discover available runtime implementations
        self._runtimes = {}
        if self.runtime_dir.is_dir():
            for entry in sorted(self.runtime_dir.iterdir()):
                config_path = entry / "build_config.py"
                if entry.is_dir() and config_path.is_file():
                    self._runtimes[entry.name] = config_path

        # Create platform-configured compiler
        self._runtime_compiler = RuntimeCompiler.get_instance(platform=platform)

    def list_runtimes(self) -> list:
        """Return names of discovered runtime implementations."""
        return list(self._runtimes.keys())

    def _validate_runtime(self, name: str) -> None:
        if name not in self._runtimes:
            available = ", ".join(self._runtimes.keys()) or "(none)"
            raise ValueError(
                f"Runtime '{name}' is not available for platform '{self.platform}'.\n"
                f"Available runtimes for {self.platform}: {available}\n"
                f"Note: Different platforms may support different runtimes. "
                f"Check {self.runtime_dir} for available implementations."
            )

    def _resolve_target_dirs(self, config_dir: Path, build_config: dict, target: str):
        """Resolve include and source dirs for a target from build_config."""
        cfg = build_config[target]
        include_dirs = [str((config_dir / p).resolve()) for p in cfg["include_dirs"]]
        source_dirs = [str((config_dir / p).resolve()) for p in cfg["source_dirs"]]
        return include_dirs, source_dirs

    def _lookup_binaries(self, name: str, output_dir: Path) -> RuntimeBinaries:
        """Look up pre-built binaries from output_dir.

        Resolves binary names from the compiler's target configs, then
        checks that each file exists.

        Raises:
            FileNotFoundError: If any binary is missing.
        """
        compiler = self._runtime_compiler
        paths = {}
        missing = []
        for target in TARGETS:
            target_obj = getattr(compiler, f"{target}_target")
            binary = output_dir / target_obj.get_binary_name()
            paths[target] = binary
            if not binary.is_file():
                missing.append(str(binary))

        if missing:
            raise FileNotFoundError(
                f"Pre-built runtime binaries not found for '{name}' "
                f"(platform={self.platform}):\n"
                + "\n".join(f"  {m}" for m in missing)
                + "\nRun 'pip install .' or pass --build to compile them."
            )

        # Validate sim_context SO exists for sim platforms
        sim_context_path = self._resolve_sim_context_path()
        if sim_context_path is not None and not sim_context_path.is_file():
            raise FileNotFoundError(
                f"Pre-built libcpu_sim_context.so not found at {sim_context_path}.\n"
                "Run 'pip install .' or pass --build to compile it."
            )

        return RuntimeBinaries(
            host_path=paths["host"],
            aicpu_path=paths["aicpu"],
            aicore_path=paths["aicore"],
            sim_context_path=sim_context_path,
        )

    def get_binaries(self, name: str, build: bool = False) -> RuntimeBinaries:
        """Return paths to compiled runtime binaries.

        By default, looks up pre-built binaries from build/lib/. When
        build=True, runs cmake configure + make using persistent build
        directories under build/cache/ for incremental compilation.

        Args:
            name: Name of the runtime implementation (e.g. 'host_build_graph')
            build: If True, compile the runtime before returning paths.
                If False (default), return pre-built binary paths.

        Returns:
            RuntimeBinaries with paths to host, aicpu, and aicore binaries.

        Raises:
            FileNotFoundError: If build=False and pre-built binaries are missing.
        """
        self._validate_runtime(name)

        arch, variant = self._arch, self._variant
        output_dir = self._LIB_DIR / arch / variant / name

        if not build:
            return self._lookup_binaries(name, output_dir)

        config_path = self._runtimes[name]
        config_dir = config_path.parent
        build_config = load_build_config(config_path)

        compiler = self._runtime_compiler

        def _compile_target(target: str) -> Path:
            include_dirs, source_dirs = self._resolve_target_dirs(config_dir, build_config, target)
            # compile() adds a {target}/ subdirectory inside build_dir
            cache_dir = self._CACHE_DIR / arch / variant / name
            cache_dir.mkdir(parents=True, exist_ok=True)

            # File lock to prevent concurrent cmake runs in the same build dir.
            # Each target gets its own lock so host/aicpu/aicore build in parallel,
            # but two processes building the same target are serialized.
            lock_path = cache_dir / f".{target}.lock"
            with open(lock_path, "w") as lock_fd:
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
                return compiler.compile(  # type: ignore[return-value]
                    target,
                    include_dirs,
                    source_dirs,
                    build_dir=str(cache_dir),
                    output_dir=output_dir,
                )

        logger.info("Compiling AICore, AICPU, Host in parallel...")

        with ThreadPoolExecutor(max_workers=4) as executor:
            fut_host = executor.submit(_compile_target, "host")
            fut_aicpu = executor.submit(_compile_target, "aicpu")
            fut_aicore = executor.submit(_compile_target, "aicore")
            fut_sim_ctx = executor.submit(self.ensure_sim_context, build=True) if variant == "sim" else None

            host_path = fut_host.result()
            aicpu_path = fut_aicpu.result()
            aicore_path = fut_aicore.result()
            sim_context_path = fut_sim_ctx.result() if fut_sim_ctx else None

        self._place_compile_commands(name)
        logger.info("Build complete!")
        return RuntimeBinaries(
            host_path=host_path,
            aicpu_path=aicpu_path,
            aicore_path=aicore_path,
            sim_context_path=sim_context_path,
        )

    def _resolve_sim_context_path(self) -> Optional[Path]:
        """Return path to libcpu_sim_context.so for sim platforms, None for onboard."""
        if self._variant != "sim":
            return None
        return self._LIB_DIR / self._arch / self._variant / "libcpu_sim_context.so"

    def ensure_sim_context(self, build: bool = False) -> Optional[Path]:
        """Build or locate the sim context SO (sim platforms only)."""
        if self._variant != "sim":
            return None

        output_dir = self._LIB_DIR / self._arch / self._variant
        so_path = output_dir / "libcpu_sim_context.so"

        if not build and so_path.is_file():
            return so_path
        if not build:
            raise FileNotFoundError(
                f"Pre-built libcpu_sim_context.so not found at {so_path}.\n"
                "Run 'pip install .' or pass --build to compile it."
            )

        cache_dir = self._CACHE_DIR / self._arch / self._variant
        cache_dir.mkdir(parents=True, exist_ok=True)
        lock_path = cache_dir / ".sim_context.lock"
        with open(lock_path, "w") as lock_fd:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            result = self._runtime_compiler.compile_sim_context(
                build_dir=str(cache_dir),
                output_dir=output_dir,
            )
            return Path(result)  # type: ignore[arg-type]

    def _place_compile_commands(self, runtime_name: str) -> None:
        """Merge compile_commands.json from build/cache/ targets into source dirs.

        Placement follows the old gen_compile_commands.py defaults:
        - Runtime dirs get compdb only from the preferred variant (onboard),
          so clangd sees real-hardware compile flags.
        - Platform dirs get compdb only from the preferred runtime
          (tensormap_and_ringbuffer), the most feature-rich runtime.

        When the preferred variant/runtime isn't available for this arch,
        falls back to writing unconditionally.
        """
        arch, variant = self._arch, self._variant
        entries = []
        for target in TARGETS:
            cc = self._CACHE_DIR / arch / variant / runtime_name / target / "compile_commands.json"
            if cc.exists():
                try:
                    entries.extend(json.loads(cc.read_text()))
                except (json.JSONDecodeError, OSError):
                    pass

        if not entries:
            return

        merged = json.dumps(entries, indent=2) + "\n"

        # Place in runtime source directory (prefer onboard variant, fallback if unavailable)
        write_runtime = (
            variant == self._COMPDB_VARIANT
            or not (self.runtime_root / "src" / arch / "platform" / self._COMPDB_VARIANT).is_dir()
        )
        if write_runtime:
            runtime_dir = self.runtime_root / "src" / arch / "runtime" / runtime_name
            if runtime_dir.is_dir():
                (runtime_dir / "compile_commands.json").write_text(merged)

        # Place in platform variant source directory (prefer tensormap_and_ringbuffer, fallback if unavailable)
        write_platform = runtime_name == self._COMPDB_RUNTIME or self._COMPDB_RUNTIME not in self._runtimes
        if write_platform:
            platform_dir = self.runtime_root / "src" / arch / "platform" / variant
            if platform_dir.is_dir():
                (platform_dir / "compile_commands.json").write_text(merged)
