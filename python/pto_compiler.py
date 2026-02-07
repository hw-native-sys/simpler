import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


class PTOCompiler:
    """
    Compiler for PTO kernels and orchestration functions.

    Platform determines compilation method:
    - "a2a3": Uses ccec for incore kernels (real hardware)
    - "a2a3sim": Uses g++ for simulation kernels (host execution)

    Both platforms use g++ for orchestration compilation.
    """

    def __init__(self, platform: str = "a2a3", ascend_home_path: Optional[str] = None):
        """
        Initialize PTOCompiler.

        Args:
            platform: Target platform ("a2a3" or "a2a3sim")
            ascend_home_path: Path to Ascend toolkit. If None, reads from
                              ASCEND_HOME_PATH environment variable.

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

        if ascend_home_path is None:
            ascend_home_path = os.getenv("ASCEND_HOME_PATH")

        self.ascend_home_path = ascend_home_path

        if platform == "a2a3":
            if not self.ascend_home_path:
                raise EnvironmentError(
                    "ASCEND_HOME_PATH environment variable is not set. "
                    "Please `source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash`."
                )
            self.cc_path = os.path.join(self.ascend_home_path, "bin", "ccec")
            if not os.path.isfile(self.cc_path):
                raise FileNotFoundError(f"ccec compiler not found: {self.cc_path}")
        else:
            # a2a3sim uses g++ which is checked at compile time
            self.cc_path = None

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

    def _resolve_host_cxx(self) -> str:
        """Resolve host C++ compiler."""
        cxx = os.environ.get("CXX") or shutil.which("g++") or shutil.which("c++")
        if not cxx:
            raise RuntimeError("Host C++ compiler not found (g++). Please install g++ or set CXX.")
        return cxx

    def _resolve_aicpu_cxx(self) -> str:
        """
        Resolve the C++ compiler used to build AICPU-side orchestration plugins.

        For platform=a2a3, the plugin must be aarch64 ELF. We try, in order:
        1) Env override: PTO_AICPU_CXX
        2) Ascend toolkit hcc toolchain (ASCEND_HOME_PATH/tools/hcc/bin/aarch64-target-linux-gnu-g++)
        3) Common cross compiler names on PATH (aarch64-linux-gnu-g++)

        For platform=a2a3sim, we use the host compiler.
        """
        if self.platform != "a2a3":
            return self._resolve_host_cxx()

        override = os.environ.get("PTO_AICPU_CXX")
        if override:
            return override

        if self.ascend_home_path:
            hcc_cxx = os.path.join(self.ascend_home_path, "tools", "hcc", "bin", "aarch64-target-linux-gnu-g++")
            if os.path.isfile(hcc_cxx):
                return hcc_cxx

        for name in ("aarch64-linux-gnu-g++", "aarch64-linux-gnu-g++-12", "aarch64-linux-gnu-g++-11"):
            p = shutil.which(name)
            if p:
                return p

        raise FileNotFoundError(
            "AICPU cross-compiler not found. Set PTO_AICPU_CXX, or install an aarch64 cross toolchain, "
            "or ensure ASCEND_HOME_PATH is set and contains tools/hcc."
        )

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
        - a2a3sim: Uses compile_incore_sim (g++)

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
        # For simulation platform, dispatch to compile_incore_sim
        if self.platform == "a2a3sim":
            return self.compile_incore_sim(
                source_path,
                pto_isa_root=pto_isa_root,
                extra_include_dirs=extra_include_dirs
            )

        # For real hardware (a2a3), continue with ccec compilation
        # Validate source file exists
        source_path = os.path.abspath(source_path)
        if not os.path.isfile(source_path):
            raise FileNotFoundError(f"Source file not found: {source_path}")

        # Validate PTO-ISA root
        if pto_isa_root is None:
            raise ValueError("pto_isa_root is required for incore compilation")

        pto_include = os.path.join(pto_isa_root, "include")
        pto_pto_include = os.path.join(pto_isa_root, "include", "pto")

        if not os.path.isdir(pto_include):
            raise FileNotFoundError(f"PTO-ISA include directory not found: {pto_include}")

        # Generate output path
        timestamp = int(time.time() * 1000)
        output_path = f"/tmp/incore_{timestamp}_{os.getpid()}.o"

        # Build compilation command
        cmd = self._build_compile_command(
            source_path=source_path,
            output_path=output_path,
            core_type=core_type,
            pto_include=pto_include,
            pto_pto_include=pto_pto_include,
            extra_include_dirs=extra_include_dirs
        )

        # Execute compilation
        core_type_name = "AIV" if core_type == "aiv" else "AIC"
        logger.info(f"[Incore] Compiling ({core_type_name}): {source_path}")
        logger.debug(f"  Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )

            if result.stdout and logger.isEnabledFor(10):  # DEBUG = 10
                logger.debug(f"[Incore] stdout:\n{result.stdout}")
            if result.stderr and logger.isEnabledFor(10):
                logger.debug(f"[Incore] stderr:\n{result.stderr}")

            if result.returncode != 0:
                logger.error(f"[Incore] Compilation failed: {result.stderr}")
                raise RuntimeError(
                    f"Incore compilation failed with exit code {result.returncode}:\n"
                    f"{result.stderr}"
                )

        except FileNotFoundError:
            raise RuntimeError(f"ccec compiler not found at {self.cc_path}")

        # Verify output file exists and read binary data
        if not os.path.isfile(output_path):
            raise RuntimeError(f"Compilation succeeded but output file not found: {output_path}")

        with open(output_path, 'rb') as f:
            binary_data = f.read()

        # Clean up temp file
        os.remove(output_path)

        logger.info(f"[Incore] Compilation successful: {len(binary_data)} bytes")
        return binary_data

    def _build_compile_command(
        self,
        source_path: str,
        output_path: str,
        core_type: str,
        pto_include: str,
        pto_pto_include: str,
        extra_include_dirs: Optional[List[str]] = None
    ) -> List[str]:
        """
        Build the ccec compilation command.

        Args:
            source_path: Path to source file
            output_path: Path for output .o file
            core_type: "aic" (cube) or "aiv" (vector)
            pto_include: Path to PTO include directory
            pto_pto_include: Path to PTO/pto include directory
            extra_include_dirs: Additional include directories

        Returns:
            List of command arguments
        """
        arch = "dav-c220-vec" if core_type == "aiv" else "dav-c220-cube"
        define = "__AIV__" if core_type == "aiv" else "__AIC__"

        cmd = [
            self.cc_path,
            "-c", "-O3", "-g", "-x", "cce",
            "-Wall", "-std=c++17",
            "--cce-aicore-only",
            f"--cce-aicore-arch={arch}",
            f"-D{define}",
            "-mllvm", "-cce-aicore-stack-size=0x8000",
            "-mllvm", "-cce-aicore-function-stack-size=0x8000",
            "-mllvm", "-cce-aicore-record-overflow=false",
            "-mllvm", "-cce-aicore-addr-transform",
            "-mllvm", "-cce-aicore-dcci-insert-for-scalar=false",
            "-DMEMORY_BASE",
            f"-I{pto_include}",
            f"-I{pto_pto_include}",
        ]

        # Add extra include dirs
        if extra_include_dirs:
            for inc_dir in extra_include_dirs:
                cmd.append(f"-I{os.path.abspath(inc_dir)}")

        # Add output and input
        cmd.extend(["-o", output_path, source_path])

        return cmd

    def compile_orchestration(
        self,
        source_path: str,
        extra_include_dirs: Optional[List[str]] = None,
        runtime_name: str = "host_build_graph"
    ) -> bytes:
        """
        Compile an orchestration function to a shared library (.so).

        The orchestration function must have signature:
            int FuncName(Runtime* runtime, uint64_t* args, int arg_count);

        Note: Use get_platform_include_dirs() to get platform-specific includes
        (e.g., for device_runner.h) and add them to extra_include_dirs.

        Args:
            source_path: Path to orchestration source file (.cpp)
            extra_include_dirs: Additional include directories (must include
                               paths to runtime.h and device_runner.h)
            runtime_name: Runtime implementation name ("tensormap_and_ringbuffer" or "host_build_graph")

        Returns:
            Binary contents of the compiled .so file

        Raises:
            FileNotFoundError: If source file not found
            RuntimeError: If compilation fails
        """
        source_path = os.path.abspath(source_path)
        if not os.path.isfile(source_path):
            raise FileNotFoundError(f"Source file not found: {source_path}")

        # Generate output path
        timestamp = int(time.time() * 1000)
        output_path = f"/tmp/orch_{timestamp}_{os.getpid()}.so"

        # Build compilation command
        # For a2a3 (real hardware), use aarch64 cross-compiler since orchestration
        # runs on AICPU Thread 3 which is on the device (aarch64)
        if self.platform == "a2a3" and self.ascend_home_path:
            cxx_path = os.path.join(self.ascend_home_path, "tools", "hcc", "bin", "aarch64-target-linux-gnu-g++")
            if not os.path.isfile(cxx_path):
                print(f"Warning: aarch64 cross-compiler not found at {cxx_path}, falling back to g++")
                cxx_path = "g++"
        else:
            cxx_path = "g++"

        cmd = [
            cxx_path,
            "-shared", "-fPIC",
            "-O3", "-g",
            "-std=c++17",
        ]

        # For a2a3 + tensormap_and_ringbuffer (device orchestration), include static linking and PTO2 runtime sources
        # because dlopen'd SO cannot access symbols from libaicpu.so
        if self.platform == "a2a3" and self.ascend_home_path and runtime_name == "tensormap_and_ringbuffer":
            cmd.extend([
                "-static-libstdc++",
                "-static-libgcc",
                "-Wl,--export-dynamic",  # Ensure symbols are exported (needed for dlsym)
            ])
            # Include PTO2 runtime source files directly
            runtime_dir = os.path.join(os.path.dirname(__file__), "..", "src", "runtime", "tensormap_and_ringbuffer", "runtime")
            runtime_dir = os.path.abspath(runtime_dir)
            runtime_sources = [
                "pto_runtime2.c",
                "pto_orchestrator.c",
                "pto_shared_memory.c",
                "pto_scheduler.c",
                "pto_ring_buffer.c",
                "pto_tensormap.c",
                "pto_logical_tensor.cpp",
            ]
            for src in runtime_sources:
                src_path = os.path.join(runtime_dir, src)
                if os.path.isfile(src_path):
                    cmd.append(src_path)
                    print(f"  Including runtime source: {src}")

        # On macOS, allow undefined symbols to be resolved at dlopen time
        if sys.platform == "darwin":
            cmd.append("-undefined")
            cmd.append("dynamic_lookup")

        # Add include dirs
        if extra_include_dirs:
            for inc_dir in extra_include_dirs:
                cmd.append(f"-I{os.path.abspath(inc_dir)}")

        # Add Ascend runtime include if available
        if self.ascend_home_path:
            ascend_include = os.path.join(self.ascend_home_path, "include")
            cmd.append(f"-I{ascend_include}")

        # Output and input
        cmd.extend(["-o", output_path, source_path])

        # Log compilation command
        logger.info(f"[Orchestration] Compiling: {source_path}")
        logger.debug(f"  Command: {' '.join(cmd)}")

        # Execute
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )

            if result.stdout and logger.isEnabledFor(10):  # DEBUG = 10
                logger.debug(f"[Orchestration] stdout:\n{result.stdout}")
            if result.stderr and logger.isEnabledFor(10):
                logger.debug(f"[Orchestration] stderr:\n{result.stderr}")

            if result.returncode != 0:
                logger.error(f"[Orchestration] Compilation failed: {result.stderr}")
                raise RuntimeError(
                    f"Orchestration compilation failed with exit code {result.returncode}:\n"
                    f"{result.stderr}"
                )

        except FileNotFoundError:
            raise RuntimeError("g++ compiler not found. Please install g++.")

        # Verify output file exists and read binary data
        if not os.path.isfile(output_path):
            raise RuntimeError(f"Compilation succeeded but output file not found: {output_path}")

        with open(output_path, 'rb') as f:
            binary_data = f.read()

        # Clean up temp file
        os.remove(output_path)

        logger.info(f"[Orchestration] Compilation successful: {len(binary_data)} bytes")
        return binary_data

    def compile_aicpu_orchestration_plugin(
        self,
        source_path: str,
        *,
        output_path: Optional[str] = None,
        extra_include_dirs: Optional[List[str]] = None,
        extra_cxxflags: Optional[List[str]] = None,
    ) -> str:
        """
        Compile the AICPU-side orchestration plugin (.so) for `aicpu_build_graph`.

        - a2a3: compiled for aarch64 (AICPU) via cross compiler.
        - a2a3sim: compiled for host (runs in host threads).

        Returns:
            Path to the compiled plugin shared library on the host filesystem.
            (Caller owns lifecycle and may delete it after run.)
        """
        source_path = os.path.abspath(source_path)
        if not os.path.isfile(source_path):
            raise FileNotFoundError(f"Source file not found: {source_path}")

        if output_path is None:
            fd, out_s = tempfile.mkstemp(prefix="aicpu_orch_", suffix=".so", dir="/tmp")
            os.close(fd)
            output_path = out_s

        cxx = self._resolve_aicpu_cxx()

        # Build compilation command
        cmd = [cxx, "-shared", "-fPIC", "-O3", "-g"]

        # For a2a3, match the platform AICPU build include search paths (subset).
        extra_inc: List[str] = []
        if self.platform == "a2a3":
            cmd.append("-std=gnu++17")
            # Prefer a self-contained plugin on device to reduce runtime deps.
            cmd.extend(["-static-libstdc++", "-static-libgcc"])
            if self.ascend_home_path:
                extra_inc.extend([
                    os.path.join(self.ascend_home_path, "include"),
                    os.path.join(self.ascend_home_path, "include", "toolchain"),
                    os.path.join(self.ascend_home_path, "pkg_inc", "base"),
                ])
        else:
            cmd.append("-std=c++17")

        if extra_cxxflags:
            cmd.extend(extra_cxxflags)

        if extra_include_dirs:
            for inc_dir in extra_include_dirs:
                cmd.append(f"-I{os.path.abspath(inc_dir)}")
        for inc_dir in extra_inc:
            cmd.append(f"-I{os.path.abspath(inc_dir)}")

        cmd.extend(["-o", output_path, source_path])

        logger.info(f"[AICPU Orchestration Plugin] Compiling: {source_path}")
        logger.debug(f"  Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.stdout and logger.isEnabledFor(10):
                logger.debug(f"[AICPU Orchestration Plugin] stdout:\n{result.stdout}")
            if result.stderr and logger.isEnabledFor(10):
                logger.debug(f"[AICPU Orchestration Plugin] stderr:\n{result.stderr}")
            if result.returncode != 0:
                logger.error(f"[AICPU Orchestration Plugin] Compilation failed: {result.stderr}")
                raise RuntimeError(
                    f"AICPU orchestration plugin compilation failed with exit code {result.returncode}:\n{result.stderr}"
                )
        except FileNotFoundError:
            raise RuntimeError(f"AICPU C++ compiler not found: {cxx}")

        if not os.path.isfile(output_path):
            raise RuntimeError(f"AICPU orchestration plugin compilation succeeded but output not found: {output_path}")

        logger.info(f"[AICPU Orchestration Plugin] Compilation successful: {os.path.getsize(output_path)} bytes")
        return output_path

    def compile_incore_sim(
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
        timestamp = int(time.time() * 1000)
        ext = ".dylib" if sys.platform == "darwin" else ".so"
        output_path = f"/tmp/sim_kernel_{timestamp}_{os.getpid()}{ext}"

        # Build compilation command to create dynamic library
        cmd = [
            "g++-15", "-shared",
            "-O2", "-fPIC",
            "-std=c++23",
            "-fpermissive",                 # Allow extensions
            "-Wno-macro-redefined",         # Suppress macro redefinition warnings
            "-Wno-ignored-attributes",      # Suppress attribute warnings
            "-D__CPU_SIM",                  # CPU simulation mode
            "-DPTO_CPU_MAX_THREADS=1",      # Prevent multithreading in PTO-ISA simulation
            "-DNDEBUG",                     # Disable assert
        ]

        # Add PTO ISA header paths if provided
        if pto_isa_root:
            pto_include = os.path.join(pto_isa_root, "include")
            cmd.append(f"-I{pto_include}")

        # Add extra include directories if provided
        if extra_include_dirs:
            for inc_dir in extra_include_dirs:
                cmd.append(f"-I{os.path.abspath(inc_dir)}")

        cmd.extend(["-o", output_path, source_path])

        # Log compilation command
        logger.info(f"[SimKernel] Compiling: {source_path}")
        logger.debug(f"  Command: {' '.join(cmd)}")

        # Execute
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )

            if result.stdout and logger.isEnabledFor(10):  # DEBUG = 10
                logger.debug(f"[SimKernel] stdout:\n{result.stdout}")
            if result.stderr and logger.isEnabledFor(10):
                logger.debug(f"[SimKernel] stderr:\n{result.stderr}")

            if result.returncode != 0:
                logger.error(f"[SimKernel] Compilation failed: {result.stderr}")
                raise RuntimeError(
                    f"SimKernel compilation failed with exit code {result.returncode}:\n"
                    f"{result.stderr}"
                )

        except FileNotFoundError:
            raise RuntimeError("g++ compiler not found. Please install g++.")

        # Verify output file exists and read binary data
        if not os.path.isfile(output_path):
            raise RuntimeError(f"Compilation succeeded but output file not found: {output_path}")

        with open(output_path, 'rb') as f:
            binary_data = f.read()

        # Clean up temp files
        os.remove(output_path)

        logger.info(f"[SimKernel] Compilation successful: {len(binary_data)} bytes")
        return binary_data
