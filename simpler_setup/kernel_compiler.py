# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import importlib.util
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional, Union

from simpler import env_manager

from .elf_parser import build_dist_orch_blob
from .environment import PROJECT_ROOT
from .toolchain import (
    Aarch64GxxToolchain,
    CCECToolchain,
    Gxx15Toolchain,
    GxxToolchain,
    ToolchainType,
)

logger = logging.getLogger(__name__)


class KernelCompiler:
    """
    Compiler for PTO kernels and orchestration functions.

    Public entry points:
    - compile_incore(): Compile a kernel source file for AICore/AIVector
    - compile_orchestration(): Compile an orchestration function for a given runtime

    Toolchain selection is determined by C++ via get_incore_compiler() and
    get_orchestration_compiler() (defined in runtime_compile_info.cpp).
    Falls back to platform-based logic if the library is not yet loaded.

    Available toolchains:
    - CCEC: ccec compiler for AICore kernels (real hardware)
    - HOST_GXX_15: g++-15 for simulation kernels (host execution)
    - HOST_GXX: g++ for orchestration .so (host dlopen)
    - AARCH64_GXX: aarch64 cross-compiler for device orchestration
    """

    # Comma-separated `-fsanitize` tokens, set once by conftest from the pytest
    # `--sanitizer` option (default "" = off). Only host toolchains (Gxx15 sim
    # incore, Gxx sim orchestration) honor it; ccec/aarch64 device builds never
    # do. Must match the runtime's install-time SIMPLER_SANITIZER.
    _sanitizers = ""

    def __init__(self, platform: str = "a2a3"):
        """
        Initialize KernelCompiler.

        Args:
            platform: Target platform ("a2a3" or "a2a3sim")

        Raises:
            ValueError: If platform is unknown
            EnvironmentError: If ASCEND_HOME_PATH is not set for a2a3 platform
            FileNotFoundError: If required compiler not found
        """
        self.platform = platform
        self.project_root = PROJECT_ROOT

        # Map platform to architecture directory
        if platform in ("a2a3", "a2a3sim"):
            self.platform_dir = self.project_root / "src" / "a2a3" / "platform"
        elif platform in ("a5", "a5sim"):
            self.platform_dir = self.project_root / "src" / "a5" / "platform"
        else:
            raise ValueError(f"Unknown platform: {platform}")

        # Create toolchain objects based on platform
        if platform in ("a2a3", "a5"):
            env_manager.ensure("ASCEND_HOME_PATH")
            self.ccec = CCECToolchain(platform)
            self.aarch64 = Aarch64GxxToolchain()
            self.host_gxx = GxxToolchain()
        else:
            self.ccec = None
            self.aarch64 = None
            # Sim orchestration must match the sim kernels' g++-15 under a
            # sanitizer (one runtime per process); see GxxToolchain prefer_g15.
            self.host_gxx = GxxToolchain(prefer_g15=bool(self._sanitizers))

        self.gxx15 = Gxx15Toolchain()

    def _sanitizer_flags(self, toolchain) -> list[str]:
        """Sanitizer flags for a host-compiled kernel / orchestration .so.

        No-op for device toolchains (ccec/aarch64) and when no sanitizer is
        selected. `-O1` + frame pointers mirror cmake/sanitizers.cmake so the
        sim kernel/orchestration match the sanitized runtime.
        """
        if not self._sanitizers or not toolchain.is_host:
            return []
        return [f"-fsanitize={self._sanitizers}", "-fno-omit-frame-pointer", "-O1"]

    def get_platform_include_dirs(self) -> list[str]:
        """
        Get platform-specific include directories for orchestration compilation.

        Returns:
            List of include directory paths (e.g., for device_runner.h, core_type.h)
        """
        return [
            str(self.platform_dir / "include"),  # For arch-specific headers
            # Shared platform headers (core_type.h, scope_stats.h, etc.) extracted
            # from per-arch copies into src/common/platform/include. Both arches
            # must see this on their include path so orchestration cpp can
            # resolve e.g. "common/core_type.h" the same way it did before.
            str(self.project_root / "src" / "common" / "platform" / "include"),
        ]

    def get_orchestration_include_dirs(self, runtime_name: str) -> list[str]:
        """
        Get all include directories needed for orchestration compilation.

        Combines the runtime-specific directory with platform include directories.

        Args:
            runtime_name: Name of the runtime (e.g., "host_build_graph")

        Returns:
            List of include directory paths:
            [runtime_dir, platform_host_dir, platform_include_dir]
        """
        # Map platform to runtime architecture
        if self.platform in ("a2a3", "a2a3sim"):
            arch = "a2a3"
        elif self.platform in ("a5", "a5sim"):
            arch = "a5"  # Phase 2: A5 uses A5 runtime
        else:
            arch = "a2a3"

        runtime_dir = str(self.project_root / "src" / arch / "runtime" / runtime_name / "runtime")
        runtime_common_dir = str(self.project_root / "src" / arch / "runtime" / runtime_name / "common")
        common_dir = str(self.project_root / "src" / "common" / "task_interface")
        return [runtime_dir, runtime_common_dir, common_dir] + self.get_platform_include_dirs()

    def get_incore_include_dirs(self) -> list[str]:
        """
        Include directories always on the incore (AICore/AIVector) kernel path.

        These hold convenience headers used by user kernels (tests, examples)
        — e.g. the pipe_sync helper at simpler_setup/incore/pipe_sync.h. They
        are not framework code and are colocated with the build tooling that
        exposes them. Both compile_incore and _compile_incore_sim prepend
        these regardless of what extra_include_dirs the caller passes, so
        kernels can include them without the call site knowing the dependency.
        """
        return [str(Path(__file__).resolve().parent / "incore")]

    def _get_orchestration_config(self, runtime_name: str) -> tuple[list[str], list[str]]:
        """
        Load the optional "orchestration" section from a runtime's build_config.py.

        If the runtime has an "orchestration" key in its BUILD_CONFIG, returns
        the resolved include dirs and discovered source files.  Otherwise returns
        empty lists (backward-compatible for runtimes without the section).

        Args:
            runtime_name: Name of the runtime (e.g., "tensormap_and_ringbuffer")

        Returns:
            (include_dirs, source_files) — both as absolute paths, or ([], [])
        """
        # Map platform to runtime architecture
        if self.platform in ("a2a3", "a2a3sim"):
            arch = "a2a3"
        elif self.platform in ("a5", "a5sim"):
            arch = "a5"  # Phase 2: A5 uses A5 runtime
        else:
            arch = "a2a3"

        config_path = self.project_root / "src" / arch / "runtime" / runtime_name / "build_config.py"
        if not config_path.is_file():
            return [], []

        spec = importlib.util.spec_from_file_location("build_config", str(config_path))
        if spec is None or spec.loader is None:
            return [], []
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        build_config = getattr(mod, "BUILD_CONFIG", {})

        orch_cfg = build_config.get("orchestration")
        if orch_cfg is None:
            return [], []

        config_dir = config_path.parent

        include_dirs = [str((config_dir / p).resolve()) for p in orch_cfg.get("include_dirs", [])]

        source_files = []
        for src_dir_rel in orch_cfg.get("source_dirs", []):
            src_dir = (config_dir / src_dir_rel).resolve()
            if src_dir.is_dir():
                for f in sorted(src_dir.iterdir()):
                    if f.suffix in (".cpp", ".c") and f.is_file():
                        source_files.append(str(f))

        return include_dirs, source_files

    def _run_subprocess(
        self, cmd: list[str], label: str, error_hint: str = "Compiler not found"
    ) -> subprocess.CompletedProcess:
        """Run a subprocess command with standardized logging and error handling."""
        logger.debug(f"[{label}] Command: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, check=False, capture_output=True, text=True)

            if result.stdout and logger.isEnabledFor(10):  # DEBUG = 10
                logger.debug(f"[{label}] stdout:\n{result.stdout}")
            if result.stderr and logger.isEnabledFor(10):
                logger.debug(f"[{label}] stderr:\n{result.stderr}")

            if result.returncode != 0:
                logger.error(f"[{label}] Compilation failed: {result.stderr}")
                raise RuntimeError(f"{label} compilation failed with exit code {result.returncode}:\n{result.stderr}")

            return result

        except FileNotFoundError:
            raise RuntimeError(error_hint)

    def _compile_to_bytes(
        self,
        cmd: list[str],
        output_path: str,
        label: str,
        error_hint: str = "Compiler not found",
        delete_output: bool = True,
    ) -> bytes:
        """Run compilation command, read output file, clean up, return bytes.

        Args:
            cmd: Compilation command and arguments
            output_path: Path to expected output file
            label: Label for log messages
            error_hint: Message for FileNotFoundError

        Returns:
            Binary contents of the compiled output file

        Raises:
            RuntimeError: If compilation fails or output file not found
        """
        self._run_subprocess(cmd, label, error_hint)

        if not os.path.isfile(output_path):
            raise RuntimeError(f"Compilation succeeded but output file not found: {output_path}")

        with open(output_path, "rb") as f:
            binary_data = f.read()

        if delete_output:
            os.remove(output_path)
        logger.info(f"[{label}] Compilation {output_path} successful: {len(binary_data)} bytes")
        return binary_data

    def _get_toolchain(self, toolchain_map: dict) -> ToolchainType:
        """Get toolchain for the current platform.

        Args:
            toolchain_map: Dict mapping platform name to ToolchainType

        Returns:
            ToolchainType for the current platform

        Raises:
            ValueError: If platform is not in the map
        """
        if self.platform not in toolchain_map:
            raise ValueError(f"No toolchain for platform: {self.platform}")
        return toolchain_map[self.platform]

    @staticmethod
    def _make_temp_path(prefix: str, suffix: str, build_dir: Optional[str] = None) -> str:
        """Create a unique temporary file path in /tmp via mkstemp.

        The file is created atomically to avoid races, then immediately
        closed so the caller can overwrite it with compiler output.
        """
        fd, path = tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=build_dir or "/tmp")
        os.close(fd)
        return path

    def compile_incore(
        self,
        source_path: str,
        core_type: str = "aiv",
        pto_isa_root: Optional[str] = None,
        extra_include_dirs: Optional[list[str]] = None,
        build_dir: Optional[str] = None,
    ) -> bytes:
        """
        Compile a kernel source file. Dispatches based on platform:
        - a2a3: Uses ccec compiler (requires pto_isa_root)
        - a2a3sim: Uses compile_incore_sim (g++-15)

        Args:
            source_path: Path to kernel source file (.cpp)
            core_type: Core type: "aic" (cube) or "aiv" (vector). Default: "aiv"
            pto_isa_root: Path to PTO-ISA root directory. Required for a2a3.
            extra_include_dirs: Additional include directories

        Returns:
            Binary contents of the compiled .o file

        Raises:
            FileNotFoundError: If source file or PTO-ISA headers not found
            ValueError: If pto_isa_root is not provided (for a2a3) or core_type is invalid
            RuntimeError: If compilation fails
        """
        incore_toolchain = self._get_toolchain(
            {
                "a2a3": ToolchainType.CCEC,
                "a2a3sim": ToolchainType.HOST_GXX_15,
                "a5": ToolchainType.CCEC,  # Phase 1: A5 uses same as A2A3
                "a5sim": ToolchainType.HOST_GXX_15,  # Phase 1: A5sim uses same as A2A3sim
            },
        )

        # Dispatch based on toolchain
        if incore_toolchain == ToolchainType.HOST_GXX_15:
            return self._compile_incore_sim(
                source_path,
                core_type=core_type,
                pto_isa_root=pto_isa_root,
                extra_include_dirs=extra_include_dirs,
                build_dir=build_dir,
            )

        # TOOLCHAIN_CCEC: continue with ccec compilation
        assert self.ccec is not None, "ccec toolchain is only available for hardware platforms"
        source_path = os.path.abspath(source_path)
        if not os.path.isfile(source_path):
            raise FileNotFoundError(f"Source file not found: {source_path}")

        if pto_isa_root is None:
            raise ValueError("pto_isa_root is required for incore compilation")

        pto_include = os.path.join(pto_isa_root, "include")
        pto_pto_include = os.path.join(pto_isa_root, "include", "pto")

        # Generate output path
        output_path = self._make_temp_path(
            prefix=f"{os.path.basename(source_path)}.incore_", suffix=".o", build_dir=build_dir
        )

        # Build command from toolchain
        cmd = [self.ccec.cxx_path] + self.ccec.get_compile_flags(core_type=core_type)
        cmd.extend([f"-I{pto_include}", f"-I{pto_pto_include}"])

        for inc_dir in self.get_incore_include_dirs():
            cmd.append(f"-I{os.path.abspath(inc_dir)}")

        if extra_include_dirs:
            for inc_dir in extra_include_dirs:
                cmd.append(f"-I{os.path.abspath(inc_dir)}")

        cmd.extend(["-o", output_path, source_path])

        # Execute compilation
        core_type_name = "AIV" if core_type == "aiv" else "AIC"
        logger.info(f"[Incore] Compiling ({core_type_name}): {source_path}")
        logger.debug(f"  Command: {' '.join(cmd)}")

        return self._compile_to_bytes(
            cmd,
            output_path,
            "Incore",
            error_hint=f"ccec compiler not found at {self.ccec.cxx_path}",
            delete_output=build_dir is None,
        )

    def compile_orchestration(
        self,
        runtime_name: str,
        source_path: str,
        extra_include_dirs: Optional[list[str]] = None,
        build_dir: Optional[str] = None,
    ) -> bytes:
        """Compile an orchestration function for the given runtime.

        Unified entry point that dispatches to the appropriate compilation
        strategy based on runtime_name.

        Args:
            runtime_name: Name of the runtime (e.g., "host_build_graph",
                         "tensormap_and_ringbuffer")
            source_path: Path to orchestration source file (.cpp)
            extra_include_dirs: Additional include directories (merged with
                               the runtime/platform include dirs)

        Returns:
            Binary contents of the compiled orchestration .so file

        Raises:
            FileNotFoundError: If source file not found
            RuntimeError: If compilation fails
            ValueError: If runtime_name is unknown
        """
        include_dirs = self.get_orchestration_include_dirs(runtime_name)
        if extra_include_dirs:
            include_dirs = include_dirs + list(extra_include_dirs)

        # Load optional orchestration config for extra sources/includes
        orch_includes, orch_sources = self._get_orchestration_config(runtime_name)
        if orch_includes:
            include_dirs = include_dirs + orch_includes

        # fully_distributed_within_core onboard: the orchestration is replayed on
        # the AICore itself (SPMD), so it must be CCEC-compiled + linked into a
        # position-independent AICore blob rather than a host-dlopen'd AArch64
        # .so. See _compile_orchestration_dist_blob and
        # docs/fully_distributed_within_core.md §16.
        if runtime_name == "fully_distributed_within_core" and self.platform == "a5":
            return self._compile_orchestration_dist_blob(
                source_path,
                extra_include_dirs=include_dirs,
                extra_sources=orch_sources or None,
                build_dir=build_dir,
            )

        # Resolve toolchain: HOST_GXX needs no runtime-specific extras
        toolchain_type = self._get_toolchain(
            {
                "a2a3": ToolchainType.AARCH64_GXX,
                "a2a3sim": ToolchainType.HOST_GXX,
                "a5": ToolchainType.AARCH64_GXX,
                "a5sim": ToolchainType.HOST_GXX,
            },
        )
        toolchain: Union[GxxToolchain, Aarch64GxxToolchain]
        if toolchain_type == ToolchainType.AARCH64_GXX:
            assert self.aarch64 is not None, "aarch64 toolchain is only available for hardware platforms"
            toolchain = self.aarch64
        else:
            toolchain = self.host_gxx

        # HOST_GXX: simulation build (host execution)
        # AARCH64_GXX: cross-compilation for supported runtimes
        #   Note: orchestration uses ops table via pto_orchestration_api.h (no extra runtime sources needed)
        return self._compile_orchestration_shared_lib(
            source_path,
            toolchain,
            extra_include_dirs=include_dirs,
            extra_sources=orch_sources or None,
            build_dir=build_dir,
        )

    def _compile_orchestration_shared_lib(
        self,
        source_path: str,
        toolchain: Union[GxxToolchain, Aarch64GxxToolchain],
        extra_include_dirs: Optional[list[str]] = None,
        extra_sources: Optional[list[str]] = None,
        build_dir: Optional[str] = None,
    ) -> bytes:
        """Compile an orchestration function to a shared library (.so).

        Prefer the unified compile_orchestration() entry point.

        Args:
            source_path: Path to orchestration source file (.cpp)
            toolchain: Resolved toolchain object (GxxToolchain or Aarch64GxxToolchain)
            extra_include_dirs: Additional include directories
            extra_sources: Additional source files to compile into the SO

        Returns:
            Binary contents of the compiled .so file
        """
        source_path = os.path.abspath(source_path)
        if not os.path.isfile(source_path):
            raise FileNotFoundError(f"Source file not found: {source_path}")

        # Generate output path
        output_path = self._make_temp_path(
            prefix=f"{os.path.basename(source_path)}.orch_", suffix=".so", build_dir=build_dir
        )

        cmd = [toolchain.cxx_path] + toolchain.get_compile_flags()
        cmd += self._sanitizer_flags(toolchain)

        # Force a deterministic ELF GNU Build-ID into every orchestration .so.
        # The host-side DeviceRunner reads `.note.gnu.build-id` to detect when
        # the same callable is being re-run (cache hit → skip device upload +
        # device dlopen). The compiler default already injects a Build-ID,
        # but pass it explicitly so the cache key remains stable across
        # toolchain versions. macOS/clang ld silently ignores this flag.
        if sys.platform != "darwin":
            cmd.append("-Wl,--build-id=sha1")

        if extra_sources:
            for src in extra_sources:
                src = os.path.abspath(src)
                if os.path.isfile(src):
                    cmd.append(src)
                    logger.debug(f"  Including extra source: {os.path.basename(src)}")

        # On macOS, allow undefined symbols to be resolved at dlopen time
        if sys.platform == "darwin":
            cmd.append("-undefined")
            cmd.append("dynamic_lookup")

        # Add include dirs
        if extra_include_dirs:
            for inc_dir in extra_include_dirs:
                cmd.append(f"-I{os.path.abspath(inc_dir)}")

        # Output and input
        cmd.extend(["-o", output_path, source_path])

        # Log compilation command
        logger.info(f"[Orchestration] Compiling: {source_path}")
        logger.debug(f"  Command: {' '.join(cmd)}")

        return self._compile_to_bytes(
            cmd,
            output_path,
            "Orchestration",
            error_hint=f"{toolchain.cxx_path} not found. Please install it.",
            delete_output=build_dir is None,
        )

    # Orchestration entry / bind symbols the AICore blob must export.
    _DIST_ORCH_ENTRY_SYMBOL = "aicpu_orchestration_entry"
    _DIST_ORCH_BIND_SYMBOL = "framework_bind_runtime"

    def _dist_blob_include_dirs(self, extra_include_dirs: Optional[list[str]]) -> list[str]:
        """Include roots for the CCEC orchestration blob.

        Mirrors the AICore runtime image build (src/{arch}/platform/onboard/
        aicore/CMakeLists.txt) so the blob's view of PTO2Runtime / PTO2RuntimeOps
        / Tensor layouts is byte-identical to the runtime it links against at
        run time.
        """
        arch = "a5" if self.platform in ("a5", "a5sim") else "a2a3"
        aicore_dir = self.platform_dir / "onboard" / "aicore"
        fdwc = self.project_root / "src" / arch / "runtime" / "fully_distributed_within_core"
        dirs = [
            str(aicore_dir),
            str(self.platform_dir / "include"),
            str(self.project_root / "src" / "common" / "platform" / "include"),
            str(self.project_root / "src" / "common" / "task_interface"),
            str(self.project_root / "src" / "common" / "log" / "include"),
            str(fdwc / "runtime"),
            str(fdwc / "common"),
            str(fdwc / "orchestration"),
            str(self.project_root / "src" / arch / "runtime"),
            str(self.project_root / "src" / "common"),
            str(self.project_root / "src" / "common" / "runtime" / "fully_distributed_within_core"),
        ]
        for inc_dir in extra_include_dirs or []:
            inc_abs = os.path.abspath(inc_dir)
            if inc_abs not in dirs:
                dirs.append(inc_abs)
        return dirs

    def _dist_blob_ccec_flags(self) -> list[str]:
        """CCEC flags matching the AICore runtime image build (single arch).

        AIC and AIV produce byte-identical code for the pure-scalar orchestration
        (verified), so one blob (vec) runs on every SPMD core.
        """
        arch = "dav-c310-vec" if self.platform in ("a5", "a5sim") else "dav-c220-vec"
        return [
            "-c",
            "-O3",
            "-g",
            "-x",
            "cce",
            "-Wall",
            "-std=c++17",
            "-DPTO2_PROFILING=0",
            "-DSIMPLER_DIST_AICORE_ONLY",
            "--cce-aicore-only",
            "-mllvm",
            "-cce-aicore-stack-size=0x8000",
            "-mllvm",
            "-cce-aicore-function-stack-size=0x8000",
            "-mllvm",
            "-cce-aicore-record-overflow=false",
            "-mllvm",
            "-cce-aicore-addr-transform",
            "-mllvm",
            "-cce-aicore-dcci-insert-for-scalar=false",
            f"--cce-aicore-arch={arch}",
        ]

    def _compile_orchestration_dist_blob(
        self,
        source_path: str,
        extra_include_dirs: Optional[list[str]] = None,
        extra_sources: Optional[list[str]] = None,
        build_dir: Optional[str] = None,
    ) -> bytes:
        """Compile the fully_distributed_within_core orchestration into a
        position-independent AICore blob for on-core SPMD replay.

        Steps: CCEC-compile the user orchestration (entry forced ``__aicore__``)
        and the shared orchestration/common.cpp, link them with
        ``cce-ld -m aicorelinux -Ttext=0 -static`` into one flat PC-relative
        AICore executable, then package the loadable image + entry/bind offsets
        (see build_dist_orch_blob). The AICPU stub copies the blob into device
        GM and dispatches the AICore entry directly (no host dlopen).
        """
        assert self.ccec is not None, "ccec toolchain is only available for hardware platforms"
        source_path = os.path.abspath(source_path)
        if not os.path.isfile(source_path):
            raise FileNotFoundError(f"Source file not found: {source_path}")

        include_flags = [f"-I{d}" for d in self._dist_blob_include_dirs(extra_include_dirs)]
        base_flags = self._dist_blob_ccec_flags()

        objects: list[str] = []
        try:
            orch_obj = self._make_temp_path(
                prefix=f"{os.path.basename(source_path)}.dist_orch_", suffix=".o", build_dir=build_dir
            )
            # The orchestration source self-attributes aicpu_orchestration_entry
            # with PTO_DEVICE_FUNC (→ __aicore__ under CCE), so no forced-include
            # prelude is needed: CCE compiles the entry as an __aicore__ symbol
            # here just as it does when the same source is linked into the
            # AICore kernel image (docs §16.1).
            cmd = [self.ccec.cxx_path] + base_flags + include_flags
            cmd.extend(["-o", orch_obj, source_path])
            logger.info(f"[DistOrch] Compiling orchestration: {source_path}")
            self._run_subprocess(cmd, "DistOrch", error_hint=f"ccec not found at {self.ccec.cxx_path}")
            objects.append(orch_obj)

            for src in extra_sources or []:
                src = os.path.abspath(src)
                if not os.path.isfile(src):
                    continue
                obj = self._make_temp_path(
                    prefix=f"{os.path.basename(src)}.dist_orch_", suffix=".o", build_dir=build_dir
                )
                cmd = [self.ccec.cxx_path] + base_flags + include_flags
                cmd.extend(["-o", obj, src])
                logger.info(f"[DistOrch] Compiling orchestration dep: {src}")
                self._run_subprocess(cmd, "DistOrch", error_hint=f"ccec not found at {self.ccec.cxx_path}")
                objects.append(obj)

            linked_path = self._make_temp_path(
                prefix=f"{os.path.basename(source_path)}.dist_orch_linked_", suffix=".elf", build_dir=build_dir
            )
            link_cmd = [
                self.ccec.linker_path,
                "-m",
                "aicorelinux",
                "-Ttext=0",
                "-static",
                "--no-undefined",
                "-o",
                linked_path,
            ] + objects
            logger.info("[DistOrch] Linking orchestration blob")
            self._run_subprocess(link_cmd, "DistOrch", error_hint=f"ccec linker not found at {self.ccec.linker_path}")

            with open(linked_path, "rb") as f:
                linked_bytes = f.read()
            blob = build_dist_orch_blob(
                linked_bytes, self._DIST_ORCH_ENTRY_SYMBOL, self._DIST_ORCH_BIND_SYMBOL
            )
            logger.info(f"[DistOrch] Blob ready: {len(blob)} bytes")

            if build_dir is None:
                for p in (*objects, linked_path):
                    if os.path.isfile(p):
                        os.remove(p)
            return blob
        except Exception:
            for p in objects:
                if os.path.isfile(p):
                    os.remove(p)
            raise

    def _compile_incore_sim(
        self,
        source_path: str,
        *,
        core_type: str,
        pto_isa_root: Optional[str] = None,
        extra_include_dirs: Optional[list[str]] = None,
        build_dir: Optional[str] = None,
    ) -> bytes:
        """
        Compile a simulation kernel to .so/.dylib using g++-15.

        Args:
            source_path: Path to kernel source file (.cpp)
            core_type: Core type: "aic" (cube) or "aiv" (vector).
            pto_isa_root: Path to PTO-ISA root directory (for PTO ISA headers)
            extra_include_dirs: Additional include directories

        Returns:
            Binary contents of the compiled .so/.dylib file

        Raises:
            FileNotFoundError: If source file not found
            RuntimeError: If compilation fails
        """
        source_path = os.path.abspath(source_path)
        if not os.path.isfile(source_path):
            raise FileNotFoundError(f"Source file not found: {source_path}")

        # Generate output path (use platform-appropriate extension)
        ext = ".dylib" if sys.platform == "darwin" else ".so"
        output_path = self._make_temp_path(
            prefix=f"{os.path.basename(source_path)}.sim_", suffix=ext, build_dir=build_dir
        )

        # Build command from toolchain
        cmd = [self.gxx15.cxx_path] + self.gxx15.get_compile_flags(core_type=core_type)
        cmd += self._sanitizer_flags(self.gxx15)

        # Add PTO ISA header paths if provided
        if pto_isa_root:
            pto_include = os.path.join(pto_isa_root, "include")
            pto_pto_include = os.path.join(pto_isa_root, "include", "pto")
            cmd.extend([f"-I{pto_include}", f"-I{pto_pto_include}"])

        for inc_dir in self.get_incore_include_dirs():
            cmd.append(f"-I{os.path.abspath(inc_dir)}")

        # Add extra include directories if provided
        if extra_include_dirs:
            for inc_dir in extra_include_dirs:
                cmd.append(f"-I{os.path.abspath(inc_dir)}")

        cmd.extend(["-o", output_path, source_path])

        # Log compilation command
        logger.info(f"[SimKernel] Compiling: {source_path}")
        logger.debug(f"  Command: {' '.join(cmd)}")

        return self._compile_to_bytes(
            cmd,
            output_path,
            "SimKernel",
            error_hint=f"{self.gxx15.cxx_path} not found. Please install g++-15.",
            delete_output=build_dir is None,
        )
