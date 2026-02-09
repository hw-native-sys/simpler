import logging
import os
import subprocess
import sys
import tempfile

from pathlib import Path
from typing import List, Optional

from bindings import get_incore_compiler, get_orchestration_compiler
from toolchain import (
    ToolchainType, CCECToolchain, Gxx15Toolchain, GxxToolchain, Aarch64GxxToolchain,
)
import env_manager

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
        self.project_root = Path(__file__).parent.parent
        self.platform_dir = self.project_root / "src" / "platform" / platform

        if platform not in ("a2a3", "a2a3sim"):
            raise ValueError(
                f"Unknown platform: {platform}. Supported: a2a3, a2a3sim"
            )

        # Create toolchain objects based on platform
        if platform == "a2a3":
            env_manager.ensure("ASCEND_HOME_PATH")
            self.ccec = CCECToolchain()
            self.aarch64 = Aarch64GxxToolchain()
            self.host_gxx = GxxToolchain()
        else:
            self.ccec = None
            self.aarch64 = None
            self.host_gxx = GxxToolchain()

        self.gxx15 = Gxx15Toolchain()

    def get_platform_include_dirs(self) -> List[str]:
        """
        Get platform-specific include directories for orchestration compilation.

        Returns:
            List of include directory paths (e.g., for device_runner.h, core_type.h)
        """
        return [
            str(self.platform_dir / "host"),
            str(self.platform_dir.parent / "include"),  # For common headers like core_type.h
        ]

    def get_orchestration_include_dirs(self, runtime_name: str) -> List[str]:
        """
        Get all include directories needed for orchestration compilation.

        Combines the runtime-specific directory with platform include directories.

        Args:
            runtime_name: Name of the runtime (e.g., "host_build_graph")

        Returns:
            List of include directory paths:
            [runtime_dir, platform_host_dir, platform_include_dir]
        """
        runtime_dir = str(self.project_root / "src" / "runtime" / runtime_name / "runtime")
        return [runtime_dir] + self.get_platform_include_dirs()

    def _run_subprocess(
        self,
        cmd: List[str],
        label: str,
        error_hint: str = "Compiler not found"
    ) -> subprocess.CompletedProcess:
        """Run a subprocess command with standardized logging and error handling.

        Args:
            cmd: Command and arguments
            label: Label for log messages (e.g., "Incore", "Orchestration")
            error_hint: Message for FileNotFoundError

        Returns:
            CompletedProcess on success

        Raises:
            RuntimeError: If command fails or executable not found
        """
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.stdout and logger.isEnabledFor(10):  # DEBUG = 10
                logger.debug(f"[{label}] stdout:\n{result.stdout}")
            if result.stderr and logger.isEnabledFor(10):
                logger.debug(f"[{label}] stderr:\n{result.stderr}")

            if result.returncode != 0:
                logger.error(f"[{label}] Compilation failed: {result.stderr}")
                raise RuntimeError(
                    f"{label} compilation failed with exit code {result.returncode}:\n"
                    f"{result.stderr}"
                )

            return result

        except FileNotFoundError:
            raise RuntimeError(error_hint)

    def _compile_to_bytes(
        self,
        cmd: List[str],
        output_path: str,
        label: str,
        error_hint: str = "Compiler not found"
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
            raise RuntimeError(
                f"Compilation succeeded but output file not found: {output_path}"
            )

        with open(output_path, 'rb') as f:
            binary_data = f.read()

        os.remove(output_path)
        logger.info(f"[{label}] Compilation successful: {len(binary_data)} bytes")
        return binary_data

    def _get_toolchain(self, strategy_fn, fallback_map: dict) -> ToolchainType:
        """Get toolchain from C++ library, with platform-based fallback.

        Args:
            strategy_fn: Callable that queries C++ for the toolchain
                         (e.g., get_incore_compiler, get_orchestration_compiler)
            fallback_map: Dict mapping platform name to ToolchainType fallback

        Returns:
            ToolchainType for the current platform/runtime

        Raises:
            ValueError: If platform has no fallback and library is not loaded
        """
        try:
            return strategy_fn()
        except RuntimeError:
            logger.debug("C++ library not loaded, using platform-based fallback")
            if self.platform not in fallback_map:
                raise ValueError(f"No toolchain fallback for platform: {self.platform}")
            return fallback_map[self.platform]

    @staticmethod
    def _make_temp_path(prefix: str, suffix: str) -> str:
        """Create a unique temporary file path in /tmp via mkstemp.

        The file is created atomically to avoid races, then immediately
        closed so the caller can overwrite it with compiler output.
        """
        fd, path = tempfile.mkstemp(prefix=prefix, suffix=suffix, dir="/tmp")
        os.close(fd)
        return path

    def compile_incore(
        self,
        source_path: str,
        core_type: str = "aiv",
        pto_isa_root: Optional[str] = None,
        extra_include_dirs: Optional[List[str]] = None
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
        # Determine toolchain from C++ (with fallback to platform-based logic)
        incore_toolchain = self._get_toolchain(
            get_incore_compiler,
            {"a2a3": ToolchainType.CCEC, "a2a3sim": ToolchainType.HOST_GXX_15}
        )

        # Dispatch based on toolchain
        if incore_toolchain == ToolchainType.HOST_GXX_15:
            return self._compile_incore_sim(
                source_path,
                pto_isa_root=pto_isa_root,
                extra_include_dirs=extra_include_dirs
            )

        # TOOLCHAIN_CCEC: continue with ccec compilation
        source_path = os.path.abspath(source_path)
        if not os.path.isfile(source_path):
            raise FileNotFoundError(f"Source file not found: {source_path}")

        if pto_isa_root is None:
            raise ValueError("pto_isa_root is required for incore compilation")

        pto_include = os.path.join(pto_isa_root, "include")
        pto_pto_include = os.path.join(pto_isa_root, "include", "pto")

        # Generate output path
        output_path = self._make_temp_path(prefix="incore_", suffix=".o")

        # Build command from toolchain
        cmd = [self.ccec.compiler_path] + self.ccec.get_compile_flags(core_type=core_type)
        cmd.extend([f"-I{pto_include}", f"-I{pto_pto_include}"])

        if extra_include_dirs:
            for inc_dir in extra_include_dirs:
                cmd.append(f"-I{os.path.abspath(inc_dir)}")

        cmd.extend(["-o", output_path, source_path])

        # Execute compilation
        core_type_name = "AIV" if core_type == "aiv" else "AIC"
        logger.info(f"[Incore] Compiling ({core_type_name}): {source_path}")
        logger.debug(f"  Command: {' '.join(cmd)}")

        return self._compile_to_bytes(
            cmd, output_path, "Incore",
            error_hint=f"ccec compiler not found at {self.ccec.compiler_path}"
        )

    def compile_orchestration(
        self,
        runtime_name: str,
        source_path: str,
        extra_include_dirs: Optional[List[str]] = None,
    ) -> bytes:
        """Compile an orchestration function for the given runtime.

        Unified entry point that dispatches to the appropriate compilation
        strategy based on runtime_name.

        Args:
            runtime_name: Name of the runtime (e.g., "host_build_graph",
                         "tensormap_and_ringbuffer", "aicpu_build_graph")
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

        if runtime_name == "host_build_graph":
            return self._compile_orchestration_shared_lib(
                source_path,
                extra_include_dirs=include_dirs,
            )

        if runtime_name == "tensormap_and_ringbuffer":
            extra_sources = None
            extra_link_flags = None
            if self.platform == "a2a3":
                runtime_dir = (
                    self.project_root / "src" / "runtime"
                    / runtime_name / "runtime"
                )
                extra_sources = sorted(
                    str(p) for p in runtime_dir.glob("*.cpp")
                    if p.name != "runtime.cpp"
                )
                extra_link_flags = self.aarch64.get_device_link_flags()
            return self._compile_orchestration_shared_lib(
                source_path,
                extra_include_dirs=include_dirs,
                extra_sources=extra_sources,
                extra_link_flags=extra_link_flags,
            )

        if runtime_name == "aicpu_build_graph":
            extra_cxxflags = None
            if self.platform == "a2a3":
                extra_cxxflags = self.aarch64.get_aicpu_plugin_flags()
                include_dirs.extend(self.aarch64.get_aicpu_include_flags())
            return self._compile_orchestration_shared_lib(
                source_path,
                extra_include_dirs=include_dirs,
                extra_cxxflags=extra_cxxflags,
            )

        raise ValueError(
            f"Unknown runtime: {runtime_name}. "
            f"Supported: host_build_graph, tensormap_and_ringbuffer, aicpu_build_graph"
        )

    def _compile_orchestration_shared_lib(
        self,
        source_path: str,
        extra_include_dirs: Optional[List[str]] = None,
        extra_sources: Optional[List[str]] = None,
        extra_link_flags: Optional[List[str]] = None,
        extra_cxxflags: Optional[List[str]] = None,
    ) -> bytes:
        """Compile an orchestration function to a shared library (.so).

        Prefer the unified compile_orchestration() entry point.

        Args:
            source_path: Path to orchestration source file (.cpp)
            extra_include_dirs: Additional include directories
            extra_sources: Additional source files to compile into the SO
            extra_link_flags: Additional linker flags
            extra_cxxflags: Additional C++ compiler flags (appended after
                base flags, so can override e.g. -std=c++17)

        Returns:
            Binary contents of the compiled .so file
        """
        source_path = os.path.abspath(source_path)
        if not os.path.isfile(source_path):
            raise FileNotFoundError(f"Source file not found: {source_path}")

        # Generate output path
        output_path = self._make_temp_path(prefix="orch_", suffix=".so")

        orch_toolchain_type = self._get_toolchain(
            get_orchestration_compiler,
            {"a2a3": ToolchainType.AARCH64_GXX, "a2a3sim": ToolchainType.HOST_GXX}
        )

        # Select toolchain object
        if orch_toolchain_type == ToolchainType.AARCH64_GXX:
            toolchain = self.aarch64
        else:
            toolchain = self.host_gxx

        cmd = [toolchain.compiler_path] + toolchain.get_compile_flags()

        if extra_cxxflags:
            cmd.extend(extra_cxxflags)

        if extra_link_flags:
            cmd.extend(extra_link_flags)

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

        # Add toolchain-level includes (e.g. Ascend SDK)
        cmd.extend(toolchain.get_include_flags())

        # Output and input
        cmd.extend(["-o", output_path, source_path])

        # Log compilation command
        logger.info(f"[Orchestration] Compiling: {source_path}")
        logger.debug(f"  Command: {' '.join(cmd)}")

        return self._compile_to_bytes(
            cmd, output_path, "Orchestration",
            error_hint=f"{toolchain.compiler_path} not found. Please install it."
        )

    def _compile_incore_sim(
        self,
        source_path: str,
        pto_isa_root: Optional[str] = None,
        extra_include_dirs: Optional[List[str]] = None
    ) -> bytes:
        """
        Compile a simulation kernel to .so/.dylib using g++-15.

        Args:
            source_path: Path to kernel source file (.cpp)
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
        output_path = self._make_temp_path(prefix="sim_kernel_", suffix=ext)

        # Build command from toolchain
        cmd = [self.gxx15.compiler_path] + self.gxx15.get_compile_flags()

        # Add PTO ISA header paths if provided
        if pto_isa_root:
            pto_include = os.path.join(pto_isa_root, "include")
            pto_pto_include = os.path.join(pto_isa_root, "include", "pto")
            cmd.extend([f"-I{pto_include}", f"-I{pto_pto_include}"])

        # Add extra include directories if provided
        if extra_include_dirs:
            for inc_dir in extra_include_dirs:
                cmd.append(f"-I{os.path.abspath(inc_dir)}")

        cmd.extend(["-o", output_path, source_path])

        # Log compilation command
        logger.info(f"[SimKernel] Compiling: {source_path}")
        logger.debug(f"  Command: {' '.join(cmd)}")

        return self._compile_to_bytes(
            cmd, output_path, "SimKernel",
            error_hint=f"{self.gxx15.compiler_path} not found. Please install g++-15."
        )
