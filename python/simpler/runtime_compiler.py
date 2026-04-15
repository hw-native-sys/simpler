# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import logging
import multiprocessing
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Union

from . import env_manager
from .toolchain import Aarch64GxxToolchain, CCECToolchain, GxxToolchain, Toolchain

logger = logging.getLogger(__name__)


class BuildTarget:
    """CMake build target: composes a Toolchain with a source directory and output name.

    Used by RuntimeCompiler for CMake-based multi-file compilation.
    """

    def __init__(self, toolchain: Toolchain, root_dir: str, binary_name: str):
        self.toolchain = toolchain
        self._root_dir = os.path.abspath(root_dir)
        self._binary_name = binary_name

    def get_root_dir(self) -> str:
        return self._root_dir

    def get_binary_name(self) -> str:
        return self._binary_name

    def gen_cmake_args(self, include_dirs: list[str], source_dirs: list[str]) -> list[str]:
        """Generate CMake arguments list from toolchain args + custom directories."""
        inc = ";".join(os.path.abspath(d) for d in include_dirs)
        src = ";".join(os.path.abspath(d) for d in source_dirs)
        args = self.toolchain.get_cmake_args() + [
            f"-DCUSTOM_INCLUDE_DIRS={inc}",
            f"-DCUSTOM_SOURCE_DIRS={src}",
        ]
        if logger.isEnabledFor(logging.DEBUG):
            args.append("--log-level=VERBOSE")
        return args


class RuntimeCompiler:
    """
    Runtime compiler for compiling runtime binaries for multiple target platforms.

    Supports three target types:
    1. aicore - AICore accelerator kernels
    2. aicpu - AICPU device task scheduler
    3. host - Host runtime library

    Platform determines which toolchains and CMake directories are used:
    - "a2a3": ccec for aicore, aarch64 cross-compiler for aicpu, gcc for host
    - "a2a3sim": all use host gcc/g++ (builds host-compatible .so files)

    Use get_instance() to get a cached instance per platform.
    """

    _instances = {}

    @classmethod
    def get_instance(cls, platform: str = "a2a3") -> "RuntimeCompiler":
        """Get or create a RuntimeCompiler instance for the given platform."""
        if platform not in cls._instances:
            cls._instances[platform] = cls(platform)
        return cls._instances[platform]

    def __init__(self, platform: str = "a2a3"):
        self.platform = platform
        self.project_root = Path(__file__).parent.parent.parent

        # Map platform name to architecture path
        if platform == "a2a3":
            self.platform_dir = self.project_root / "src" / "a2a3" / "platform" / "onboard"
        elif platform == "a2a3sim":
            self.platform_dir = self.project_root / "src" / "a2a3" / "platform" / "sim"
        elif platform == "a5":
            self.platform_dir = self.project_root / "src" / "a5" / "platform" / "onboard"
        elif platform == "a5sim":
            self.platform_dir = self.project_root / "src" / "a5" / "platform" / "sim"
        else:
            raise ValueError(f"Unknown platform: {platform}. Supported: a2a3, a2a3sim, a5, a5sim")

        if not self.platform_dir.is_dir():
            raise ValueError(f"Platform '{platform}' not found at {self.platform_dir}")

        if platform == "a2a3":
            self._init_a2a3()
        elif platform == "a2a3sim":
            self._init_a2a3sim()
        elif platform == "a5":
            self._init_a5()
        elif platform == "a5sim":
            self._init_a5sim()
        else:
            raise ValueError(f"Unknown platform: {platform}. Supported: a2a3, a2a3sim, a5, a5sim")

    def _init_a2a3(self):
        """Initialize toolchains for real a2a3 hardware."""
        env_manager.ensure("ASCEND_HOME_PATH")

        # AICore: Bisheng CCE compiler
        ccec = CCECToolchain(platform="a2a3")
        self.aicore_target = BuildTarget(ccec, str(self.platform_dir / "aicore"), "aicore_kernel.o")

        # AICPU: aarch64 cross-compiler
        aarch64 = Aarch64GxxToolchain()
        self.aicpu_target = BuildTarget(aarch64, str(self.platform_dir / "aicpu"), "libaicpu_kernel.so")

        # Host: standard gcc/g++
        self._ensure_host_compilers()
        host_gxx = GxxToolchain()
        self.host_target = BuildTarget(host_gxx, str(self.platform_dir / "host"), "libhost_runtime.so")

    def _init_a2a3sim(self):
        """Initialize toolchains for simulation platform.
        All targets use host gcc/g++ with platform-specific CMake dirs.
        No Ascend SDK required.
        """
        self._ensure_host_compilers()
        gxx = GxxToolchain()

        self.aicore_target = BuildTarget(gxx, str(self.platform_dir / "aicore"), "libaicore_kernel.so")
        self.aicpu_target = BuildTarget(gxx, str(self.platform_dir / "aicpu"), "libaicpu_kernel.so")
        self.host_target = BuildTarget(gxx, str(self.platform_dir / "host"), "libhost_runtime.so")

    def _init_a5(self):
        """Initialize toolchains for real a5 hardware."""
        env_manager.ensure("ASCEND_HOME_PATH")

        # AICore: Bisheng CCE compiler with A5 platform
        ccec = CCECToolchain(platform="a5")
        self.aicore_target = BuildTarget(ccec, str(self.platform_dir / "aicore"), "aicore_kernel.o")

        # AICPU: aarch64 cross-compiler
        aarch64 = Aarch64GxxToolchain()
        self.aicpu_target = BuildTarget(aarch64, str(self.platform_dir / "aicpu"), "libaicpu_kernel.so")

        # Host: standard gcc/g++
        self._ensure_host_compilers()
        host_gxx = GxxToolchain()
        self.host_target = BuildTarget(host_gxx, str(self.platform_dir / "host"), "libhost_runtime.so")

    def _init_a5sim(self):
        """Initialize toolchains for A5 simulation platform.
        All targets use host gcc/g++ with platform-specific CMake dirs.
        No Ascend SDK required.
        """
        self._ensure_host_compilers()
        gxx = GxxToolchain()

        self.aicore_target = BuildTarget(gxx, str(self.platform_dir / "aicore"), "libaicore_kernel.so")
        self.aicpu_target = BuildTarget(gxx, str(self.platform_dir / "aicpu"), "libaicpu_kernel.so")
        self.host_target = BuildTarget(gxx, str(self.platform_dir / "host"), "libhost_runtime.so")

    def _ensure_host_compilers(self):
        if not self._find_executable("gcc"):
            raise FileNotFoundError("Host C compiler not found: gcc. Please install gcc.")
        if not self._find_executable("g++"):
            raise FileNotFoundError("Host C++ compiler not found: g++. Please install g++.")

    @staticmethod
    def _find_executable(name: str) -> bool:
        """Check if an executable exists (either as absolute path or in PATH)."""
        if os.path.isfile(name) and os.access(name, os.X_OK):
            return True
        result = subprocess.run(["which", name], check=False, capture_output=True, timeout=1)
        return result.returncode == 0

    def compile(
        self,
        target_platform: str,
        include_dirs: list[str],
        source_dirs: list[str],
        build_dir: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Union[bytes, Path]:
        """
        Compile binary for the specified target platform.

        Args:
            target_platform: Target platform ("aicore", "aicpu", or "host")
            include_dirs: List of include directory paths
            source_dirs: List of source directory paths
            build_dir: The directory path for compiling. When None, use a temporal path.
            output_dir: Directory to copy the final binary into. When set, returns Path.
                        When None, returns bytes (backward-compatible).

        Returns:
            If output_dir is set: Path to the compiled binary in output_dir.
            If output_dir is None: Compiled binary data as bytes.

        Raises:
            ValueError: If target platform is invalid
            RuntimeError: If CMake or Make fails
            FileNotFoundError: If output binary not found
        """
        if target_platform == "aicore":
            target = self.aicore_target
        elif target_platform == "aicpu":
            target = self.aicpu_target
        elif target_platform == "host":
            target = self.host_target
        else:
            raise ValueError(f"Invalid target platform: {target_platform}. Must be 'aicore', 'aicpu', or 'host'.")

        cmake_args = target.gen_cmake_args(include_dirs, source_dirs)
        cmake_source_dir = target.get_root_dir()
        binary_name = target.get_binary_name()
        platform = target_platform.upper()

        def _build(actual_build_dir: str) -> Union[bytes, Path]:
            binary_path = self._run_compilation(
                cmake_source_dir,
                cmake_args,
                binary_name,
                platform=platform,
                build_dir=actual_build_dir,
            )
            if output_dir is not None:
                od = Path(output_dir)
                od.mkdir(parents=True, exist_ok=True)
                dest = od / binary_name
                shutil.copy2(binary_path, dest)
                dispatcher_so = Path(actual_build_dir) / "libaicpu_dispatcher.so"
                if dispatcher_so.is_file():
                    dest_dispatcher = od / "libaicpu_dispatcher.so"
                    shutil.copy2(dispatcher_so, dest_dispatcher)
                    # Strip debug info to match CANN built-in SO format
                    subprocess.run(["strip", "-s", str(dest_dispatcher)], check=True)
                return dest
            else:
                with open(binary_path, "rb") as f:
                    return f.read()

        if build_dir is None:
            with tempfile.TemporaryDirectory(prefix=f"{platform.lower()}_build_", dir="/tmp") as tmp_dir:
                return _build(tmp_dir)
        else:
            platform_build_dir = Path(os.path.realpath(build_dir)) / f"{platform.lower()}"
            os.makedirs(platform_build_dir, exist_ok=True)
            return _build(str(platform_build_dir))

    def _run_build_step(
        self,
        cmd: list[str],
        cwd: str,
        platform: str,
        step_name: str,
    ) -> None:
        """Run a single build step (CMake or Make) with logging and error handling.

        Args:
            cmd: Command and arguments
            cwd: Working directory
            platform: Platform name for logging
            step_name: Step name for logging (e.g., "CMake", "Make")

        Raises:
            RuntimeError: If the step fails or executable not found
        """
        logger.info(f"[{platform}] Running {step_name}...")
        logger.debug(f"  Working directory: {cwd}")
        logger.debug(f"  Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, cwd=cwd, check=False, capture_output=True, text=True)

            if result.stdout and logger.isEnabledFor(10):  # DEBUG = 10
                logger.debug(f"[{platform}] {step_name} stdout:")
                logger.debug(result.stdout)
            if result.stderr and logger.isEnabledFor(10):
                logger.debug(f"[{platform}] {step_name} stderr:")
                logger.debug(result.stderr)

            if result.returncode != 0:
                self._log_failed_build_output(platform, step_name, result)
                raise RuntimeError(f"{step_name} failed for {platform} with exit code {result.returncode}")
        except FileNotFoundError:
            raise RuntimeError(f"{step_name} not found. Please install {step_name}.")

    @staticmethod
    def _log_failed_build_output(platform: str, step_name: str, result: subprocess.CompletedProcess) -> None:
        """Emit captured build output at ERROR level so failures are visible by default."""
        logger.error(f"[{platform}] {step_name} failed with exit code {result.returncode}")

        if result.stdout:
            logger.error(f"[{platform}] {step_name} stdout:\n{result.stdout.rstrip()}")
        if result.stderr:
            logger.error(f"[{platform}] {step_name} stderr:\n{result.stderr.rstrip()}")
        if not result.stdout and not result.stderr:
            logger.error(f"[{platform}] {step_name} produced no stdout/stderr output")

    def _run_compilation(
        self,
        cmake_source_dir: str,
        cmake_args: list[str],
        binary_name: str,
        platform: str = "AICore",
        build_dir: Optional[str] = None,
    ) -> Path:
        """
        Run CMake configuration and Make build.

        Args:
            cmake_source_dir: Path to CMake source directory
            cmake_args: CMake command-line arguments
            binary_name: Name of output binary
            platform: Platform name for logging
            build_dir: Build directory path

        Returns:
            Path to compiled binary within the build directory.

        Raises:
            RuntimeError: If CMake or Make fails
            FileNotFoundError: If output binary not found
        """
        if build_dir is None:
            raise ValueError("build_dir must be set")

        cmake_cmd = [
            "cmake",
            cmake_source_dir,
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
        ] + cmake_args
        try:
            self._run_build_step(cmake_cmd, build_dir, platform, "CMake configuration")
        except RuntimeError as exc:
            # Persistent CMake cache dirs under build/cache can become invalid when the
            # selected compiler or command-line cache entries change. In that case, wipe
            # the target build dir and retry once with a clean configure.
            build_path = Path(build_dir)
            if not any(build_path.iterdir()):
                raise
            logger.warning(
                "[%s] CMake configuration failed in cached build dir %s; clearing cache and retrying once",
                platform,
                build_dir,
            )
            shutil.rmtree(build_path)
            build_path.mkdir(parents=True, exist_ok=True)
            try:
                self._run_build_step(cmake_cmd, build_dir, platform, "CMake configuration")
            except RuntimeError:
                raise exc

        build_cmd = [
            "cmake",
            "--build",
            ".",
            "--parallel",
            str(min(multiprocessing.cpu_count(), 32)),
            "--verbose",
        ]
        self._run_build_step(build_cmd, build_dir, platform, "Build")

        # Return the path to the compiled binary
        binary_path = Path(build_dir) / binary_name
        if not binary_path.is_file():
            raise FileNotFoundError(
                f"Compiled binary not found: {binary_path}. Expected output file name: {binary_name}"
            )

        return binary_path

    def compile_sim_context(
        self,
        build_dir: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Union[bytes, Path]:
        """Compile the standalone libcpu_sim_context.so (sim platforms only).

        This library contains the per-device CPU simulation context functions
        (pto_cpu_sim_get_shared_storage, etc.) that must be loaded with
        RTLD_GLOBAL for PTO ISA kernel SOs to find via dlsym(RTLD_DEFAULT).
        """
        if not self.platform.endswith("sim"):
            raise ValueError(f"compile_sim_context is only for sim platforms, got {self.platform}")

        cmake_source_dir = str(self.project_root / "src" / "common" / "sim_context")
        binary_name = "libcpu_sim_context.so"
        cmake_args = self.host_target.toolchain.get_cmake_args()

        def _build(actual_build_dir: str) -> Union[bytes, Path]:
            binary_path = self._run_compilation(
                cmake_source_dir,
                cmake_args,
                binary_name,
                platform="SIM_CONTEXT",
                build_dir=actual_build_dir,
            )
            if output_dir is not None:
                od = Path(output_dir)
                od.mkdir(parents=True, exist_ok=True)
                dest = od / binary_name
                shutil.copy2(binary_path, dest)
                return dest
            else:
                with open(binary_path, "rb") as f:
                    return f.read()

        if build_dir is None:
            with tempfile.TemporaryDirectory(prefix="sim_context_build_", dir="/tmp") as tmp_dir:
                return _build(tmp_dir)
        else:
            ctx_build_dir = Path(os.path.realpath(build_dir)) / "sim_context"
            os.makedirs(ctx_build_dir, exist_ok=True)
            return _build(str(ctx_build_dir))
