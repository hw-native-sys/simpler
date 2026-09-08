# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import hashlib
import importlib.util
import logging
import os
import subprocess
import sys
import tempfile
from functools import cache
from pathlib import Path
from typing import Optional, Union

from simpler import env_manager

from .compile_paths import compiler_visible_path
from .environment import PROJECT_ROOT
from .toolchain import (
    Aarch64GxxToolchain,
    CCECToolchain,
    Gxx15Toolchain,
    GxxToolchain,
    ToolchainType,
)

logger = logging.getLogger(__name__)

# Covers the part of the persistent scene-test cache key that nothing else
# fingerprints: how ``scene_test.compile_chip_callable_spec`` assembles
# compiled binaries into a ``ChipCallable``. The modules that decide the bytes
# of a single compiled artifact carry their own digest (see
# ``_artifact_logic_token``), and the binding layout carries an ABI token, so a
# manual bump here is only owed for a change in that assembly step.
_COMPILE_CACHE_SCHEMA = 1

# Modules whose logic turns kernel sources into artifact bytes: compiler
# invocation and flags, toolchain selection, and the ELF section extraction
# applied to every onboard incore.
_ARTIFACT_LOGIC_MODULES = ("kernel_compiler.py", "toolchain.py", "compile_paths.py", "elf_parser.py")


@cache
def _artifact_logic_token() -> str:
    """Digest the modules that decide a compiled artifact's bytes.

    Source content, compiler identity and compile flags are keyed separately;
    this token is what invalidates a cached artifact when the compilation logic
    itself changes, so editing one of these modules takes effect without a
    manual ``_COMPILE_CACHE_SCHEMA`` bump.
    """
    digest = hashlib.sha256()
    module_dir = Path(__file__).resolve().parent
    for name in _ARTIFACT_LOGIC_MODULES:
        digest.update(name.encode())
        try:
            digest.update(hashlib.sha256((module_dir / name).read_bytes()).digest())
        except OSError:
            digest.update(b"unreadable")
    return digest.hexdigest()


@cache
def _executable_cache_identity(executable: str) -> dict[str, object]:
    """Return stable compiler identity without embedding runner-local paths."""
    try:
        result = subprocess.run([executable, "--version"], check=False, capture_output=True, text=True, timeout=5)
        return {
            "name": os.path.basename(executable),
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except (OSError, subprocess.SubprocessError) as error:
        return {"name": os.path.basename(executable), "error": type(error).__name__}


def _cmake_bool_env_enabled(name: str, default: bool = False) -> bool:
    """Interpret a CMake-style boolean env var (ON/OFF/1/0/TRUE/FALSE/YES/NO)."""
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().upper()
    if normalized in {"1", "ON", "TRUE", "YES", "Y"}:
        return True
    if normalized in {"0", "OFF", "FALSE", "NO", "N"}:
        return False
    return default


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
    - HOST_GXX: g++ (g++-15 under sanitizers) for host orchestration .so
    - AARCH64_GXX: aarch64 cross-compiler for device orchestration
    """

    # Comma-separated `-fsanitize` tokens, set once by conftest from the pytest
    # `--sanitizer` option (default "" = off). Only host toolchains (Gxx15 sim
    # incore, Gxx host orchestration) honor it; ccec/aarch64 device builds never
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
            self.host_gxx = GxxToolchain(prefer_g15=bool(self._sanitizers))
        else:
            self.ccec = None
            self.aarch64 = None
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
        src_common_dir = str(self.project_root / "src" / "common")
        return [runtime_dir, runtime_common_dir, common_dir, src_common_dir] + self.get_platform_include_dirs()

    def get_incore_include_dirs(self) -> list[str]:
        """
        Include directories always on the incore (AICore/AIVector) kernel path.

        These hold convenience headers used by user kernels (tests, examples)
        — e.g. the pipe_sync helper at simpler_setup/incore/pipe_sync.h. They
        are not framework code and are colocated with the build tooling that
        exposes them. Both compile_incore and _compile_incore_sim prepend
        these regardless of what extra_include_dirs the caller passes, so
        kernels can include them without the call site knowing the dependency.

        Also carries ``src/common/platform/include`` so kernel-facing runtime
        headers (e.g. ``intrinsic.h``) can pull shared platform headers like
        ``common/dma_workspace.h`` regardless of what the call site passes.
        """
        return [
            str(Path(__file__).resolve().parent / "incore"),
            str(self.project_root / "src" / "common" / "platform" / "include"),
        ]

    def _incore_feature_defines(self) -> list[str]:
        """Feature macros forwarded to incore compiles for the RDMA overlay."""
        if self.platform == "a5" and _cmake_bool_env_enabled("SIMPLER_ENABLE_PTO_RDMA_WORKSPACE", default=False):
            return ["-DPTO_RDMA_SUPPORTED", "-DPTO_RDMA_BACKEND_HNS_1825_SUPPORTED"]
        return []

    @staticmethod
    def _pto_isa_include_dirs(pto_isa_root: str) -> list[str]:
        """PTO-ISA include roots; RDMA intrinsics live under pkg_inc."""
        include_dirs = [os.path.join(pto_isa_root, "include")]
        pkg_inc = os.path.join(pto_isa_root, "pkg_inc")
        if os.path.isdir(pkg_inc):
            include_dirs.append(pkg_inc)
        return include_dirs

    def get_ascend_incore_include_dirs(self) -> list[str]:
        """CANN device headers used by PTO-ISA backend-only includes."""
        ascend_home = env_manager.get("ASCEND_HOME_PATH")
        if not ascend_home:
            return []
        roots = [Path(ascend_home)]
        roots += [Path(ascend_home) / arch for arch in ("aarch64-linux", "x86_64-linux")]

        candidates: list[Path] = []
        for root in roots:
            candidates.extend(
                [
                    root / "include",
                    root / "include" / "external",
                    root / "include" / "c_api",
                    root / "include" / "c_api" / "internal",
                    root / "asc" / "include",
                    root / "asc" / "include" / "interface",
                    root / "asc" / "include" / "basic_api",
                    root / "asc" / "include" / "basic_api" / "interface",
                    root / "ascendc" / "include",
                    root / "ascendc" / "include" / "basic_api",
                    root / "ascendc" / "include" / "basic_api" / "interface",
                ]
            )
        for header in ("kernel_operator_sys_var_intf.h", "kernel_operator_sys_var_intf_impl.h"):
            try:
                candidates.extend(path.parent for path in Path(ascend_home).rglob(header))
            except OSError:
                continue

        include_dirs = []
        seen = set()
        for path in candidates:
            if path.is_dir() and path not in seen:
                include_dirs.append(str(path))
                seen.add(path)
        return include_dirs

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

    def _get_orchestration_platform_sources(self) -> list[str]:
        """Sources needed when public AICPU helper headers are used by orchestration SOs."""
        variant = "sim" if self.platform.endswith("sim") else "onboard"
        source_dir = self.project_root / "src" / "common" / "platform" / variant / "aicpu"
        return [
            str(source_dir / "cache_ops.cpp"),
            str(source_dir / "device_time.cpp"),
        ]

    def get_orchestration_cache_inputs(self, runtime_name: str) -> tuple[list[str], list[str]]:
        """Return the include directories and extra sources used by orchestration compilation.

        Keeping this calculation beside ``compile_orchestration`` ensures the
        persistent scene-test cache hashes the same inputs the compiler sees.
        """
        include_dirs = self.get_orchestration_include_dirs(runtime_name)
        config_include_dirs, config_sources = self._get_orchestration_config(runtime_name)
        platform_sources = self._get_orchestration_platform_sources()
        if self._orchestration_toolchain(runtime_name).is_host:
            include_dirs.append(str(self.project_root / "src" / "common" / "log" / "include"))
            host_log_dir = self.project_root / "src" / "common" / "log"
            platform_sources.extend([str(host_log_dir / "host_log.cpp"), str(host_log_dir / "unified_log_host.cpp")])
        return (
            [*include_dirs, *config_include_dirs],
            [*config_sources, *platform_sources],
        )

    def _orchestration_toolchain(self, runtime_name: str) -> Union[GxxToolchain, Aarch64GxxToolchain]:
        if runtime_name == "host_build_graph":
            return self.host_gxx
        if runtime_name == "tensormap_and_ringbuffer":
            if self.platform.endswith("sim"):
                return self.host_gxx
            assert self.aarch64 is not None, "aarch64 toolchain is only available for hardware platforms"
            return self.aarch64
        raise ValueError(f"Unknown runtime_name: {runtime_name!r}")

    def _orchestration_compile_flags(self, toolchain: Union[GxxToolchain, Aarch64GxxToolchain]) -> list[str]:
        return [*toolchain.get_compile_flags(), *self._sanitizer_flags(toolchain)]

    @staticmethod
    def _orchestration_link_flags(toolchain: Union[GxxToolchain, Aarch64GxxToolchain]) -> list[str]:
        """Return the link flags every orchestration ``.so`` carries on this host.

        macOS/clang resolves the runtime's symbols at dlopen time, so undefined
        symbols must be allowed. Elsewhere a deterministic ELF GNU Build-ID is
        forced in: the host-side DeviceRunner reads ``.note.gnu.build-id`` to
        recognize a callable it has already uploaded, and passing the flag
        explicitly keeps that id stable across toolchain versions even though
        the compiler default already injects one.
        """
        flags = ["-undefined", "dynamic_lookup"] if sys.platform == "darwin" else ["-Wl,--build-id=sha1"]
        if toolchain.is_host:
            flags.append("-pthread")
        return flags

    def compile_cache_token(self, runtime_name: str, core_types: list[str]) -> dict[str, object]:
        """Describe every compiler and fixed flag that affects kernel artifacts."""
        orchestration = self._orchestration_toolchain(runtime_name)
        incore_tokens = [self.incore_compile_cache_token(core_type) for core_type in sorted(set(core_types))]
        return {
            "schema": _COMPILE_CACHE_SCHEMA,
            "logic": _artifact_logic_token(),
            "orchestration": {
                "identity": _executable_cache_identity(orchestration.cxx_path),
                "compile_flags": self._orchestration_compile_flags(orchestration),
                "link_flags": self._orchestration_link_flags(orchestration),
            },
            "incore": {
                "variants": incore_tokens,
            },
        }

    def incore_compile_cache_token(self, core_type: str) -> dict[str, object]:
        """Describe the compiler inputs shared by identical incore sources."""
        if self.platform.endswith("sim"):
            incore = self.gxx15
            flags = [*incore.get_compile_flags(core_type=core_type), *self._sanitizer_flags(incore)]
            linker = None
        else:
            assert self.ccec is not None, "ccec toolchain is only available for hardware platforms"
            incore = self.ccec
            flags = incore.get_compile_flags(core_type=core_type)
            linker = {
                "identity": _executable_cache_identity(self.ccec.linker_path),
                "flags": ["-e", "kernel_entry"],
            }
        return {
            "schema": _COMPILE_CACHE_SCHEMA,
            "logic": _artifact_logic_token(),
            "identity": _executable_cache_identity(incore.cxx_path),
            "core_type": core_type,
            "flags": flags,
            "linker": linker,
        }

    def _run_subprocess(
        self, cmd: list[str], label: str, error_hint: str = "Compiler not found"
    ) -> subprocess.CompletedProcess:
        """Run a subprocess command with standardized logging and error handling."""
        logger.debug(f"[{label}] Command: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, check=False, capture_output=True, text=True, cwd=self.project_root)

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
        """Create a unique absolute temporary file path via mkstemp.

        The file is created atomically to avoid races, then immediately
        closed so the caller can overwrite it with compiler output.
        """
        directory = os.path.abspath(build_dir) if build_dir else "/tmp"
        fd, path = tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=directory)
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
        - a2a3: Uses ccec compiler (requires pto_isa_root), then links
        - a2a3sim: Uses compile_incore_sim (g++-15)

        Args:
            source_path: Path to kernel source file (.cpp)
            core_type: Core type: "aic" (cube) or "aiv" (vector). Default: "aiv"
            pto_isa_root: Path to PTO-ISA root directory. Required for a2a3.
            extra_include_dirs: Additional include directories

        Returns:
            On hardware platforms, the linked AICore image (see
            :meth:`_link_incore`) that ``elf_parser.extract_text_section``
            takes the loadable payload from; on sim, the compiled shared object.

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

        # Generate output path
        output_path = self._make_temp_path(
            prefix=f"{os.path.basename(source_path)}.incore_", suffix=".o", build_dir=build_dir
        )

        # Build command from toolchain
        cmd = [self.ccec.cxx_path, *self.ccec.get_compile_flags(core_type=core_type)]
        cmd += self._incore_feature_defines()
        for include_dir in self._pto_isa_include_dirs(pto_isa_root):
            cmd.append(f"-I{compiler_visible_path(include_dir)}")
            cmd.append(f"-I{compiler_visible_path(os.path.join(include_dir, 'pto'))}")

        for inc_dir in self.get_incore_include_dirs():
            cmd.append(f"-I{compiler_visible_path(inc_dir)}")

        for inc_dir in self.get_ascend_incore_include_dirs():
            cmd.append(f"-I{compiler_visible_path(inc_dir)}")

        if extra_include_dirs:
            for inc_dir in extra_include_dirs:
                cmd.append(f"-I{compiler_visible_path(inc_dir)}")

        cmd.extend(["-o", output_path, str(compiler_visible_path(source_path))])

        # Execute compilation
        core_type_name = "AIV" if core_type == "aiv" else "AIC"
        logger.info(f"[Incore] Compiling ({core_type_name}): {source_path}")
        logger.debug(f"  Command: {' '.join(cmd)}")

        self._compile_to_bytes(
            cmd,
            output_path,
            "Incore",
            error_hint=f"ccec compiler not found at {self.ccec.cxx_path}",
            delete_output=False,
        )
        try:
            return self._link_incore(output_path, build_dir=build_dir)
        finally:
            if build_dir is None:
                os.remove(output_path)

    def _link_incore(self, object_path: str, build_dir: Optional[str] = None) -> bytes:
        """Link a compiled incore object into a self-contained AICore image.

        The AICore loader copies the literal ``.text`` bytes and jumps to offset
        0 (``simpler_setup/elf_parser.py``), so the payload must carry no
        unapplied relocations and must begin at ``kernel_entry``. ``ld.lld``
        resolves ``.rela.text`` — notably the ``.bl.uninit.*`` block-local
        globals CANN AscendC headers declare, such as ``g_vecTPipePtr`` and
        ``g_kfcClient``, which share one merged block-local region and so cannot
        all sit at offset 0 — and folds ``.text.*`` COMDAT groups holding
        out-of-line template instantiations into the single output ``.text``.

        The linked image is position-independent: ``--image-base`` does not
        change the emitted ``.text`` bytes.
        """
        assert self.ccec is not None, "incore linking is only available for hardware platforms"
        linked_path = self._make_temp_path(
            prefix=f"{os.path.basename(object_path)}.linked_", suffix=".elf", build_dir=build_dir
        )
        cmd = [self.ccec.linker_path, "-e", "kernel_entry", "-o", linked_path, object_path]
        logger.debug(f"  Link command: {' '.join(cmd)}")
        return self._compile_to_bytes(
            cmd,
            linked_path,
            "Incore-link",
            error_hint=f"ccec linker not found at {self.ccec.linker_path}",
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
        include_dirs, orch_sources = self.get_orchestration_cache_inputs(runtime_name)
        if extra_include_dirs:
            include_dirs = include_dirs + list(extra_include_dirs)

        # host_build_graph dlopens the orchestration .so on the host, so it
        # compiles with the host g++ (x86_64) regardless of platform.
        # tensormap_and_ringbuffer dlopens it on the aarch64 AICPU onboard, so
        # it cross-compiles there and uses the host g++ only for sim.
        toolchain = self._orchestration_toolchain(runtime_name)

        # HOST_GXX: simulation build (host execution)
        # AARCH64_GXX: cross-compilation for supported runtimes
        # Runtime calls still go through orchestration_api.h; platform helper sources
        # are linked only for public AICPU utility headers used directly by orchestration code.
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

        cmd = [
            toolchain.cxx_path,
            *self._orchestration_compile_flags(toolchain),
            *self._orchestration_link_flags(toolchain),
        ]

        if extra_sources:
            for src in extra_sources:
                src = os.path.abspath(src)
                if os.path.isfile(src):
                    cmd.append(str(compiler_visible_path(src)))
                    logger.debug(f"  Including extra source: {os.path.basename(src)}")

        # Add include dirs
        if extra_include_dirs:
            for inc_dir in extra_include_dirs:
                cmd.append(f"-I{compiler_visible_path(inc_dir)}")

        # Output and input
        cmd.extend(["-o", output_path, str(compiler_visible_path(source_path))])

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
        cmd = [self.gxx15.cxx_path, *self.gxx15.get_compile_flags(core_type=core_type)]
        cmd += self._sanitizer_flags(self.gxx15)

        # Add PTO ISA header paths if provided. The path always comes from
        # ensure_pto_isa_root(), which has already verified HEAD == pto_isa.pin,
        # so no per-compile re-check is needed here.
        if pto_isa_root:
            pto_include = os.path.join(pto_isa_root, "include")
            pto_pto_include = os.path.join(pto_isa_root, "include", "pto")
            cmd.extend([f"-I{compiler_visible_path(pto_include)}", f"-I{compiler_visible_path(pto_pto_include)}"])

        for inc_dir in self.get_incore_include_dirs():
            cmd.append(f"-I{compiler_visible_path(inc_dir)}")

        # Add extra include directories if provided
        if extra_include_dirs:
            for inc_dir in extra_include_dirs:
                cmd.append(f"-I{compiler_visible_path(inc_dir)}")

        cmd.extend(["-o", output_path, str(compiler_visible_path(source_path))])

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
