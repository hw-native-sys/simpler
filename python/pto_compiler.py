import os
import subprocess
import time
from pathlib import Path
from typing import List, Optional


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
            List of include directory paths (e.g., for device_runner.h)
        """
        return [str(self.platform_dir / "host")]

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
            return self.compile_incore_sim(source_path)

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
        print(f"\n{'='*80}")
        print(f"[Incore] Compiling ({core_type_name}): {source_path}")
        print(f"  Command: {' '.join(cmd)}")
        print(f"{'='*80}\n")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )

            if result.stdout:
                print(f"[Incore] stdout:\n{result.stdout}")
            if result.stderr:
                print(f"[Incore] stderr:\n{result.stderr}")

            if result.returncode != 0:
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

        print(f"[Incore] Compilation successful: {len(binary_data)} bytes")
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
        extra_include_dirs: Optional[List[str]] = None
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

        # Build compilation command (using g++)
        cmd = [
            "g++",
            "-shared", "-fPIC",
            "-O3", "-g",
            "-std=c++17",
        ]

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

        # Print compilation command
        print(f"\n{'='*80}")
        print(f"[Orchestration] Compiling: {source_path}")
        print(f"  Command: {' '.join(cmd)}")
        print(f"{'='*80}\n")

        # Execute
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )

            if result.stdout:
                print(f"[Orchestration] stdout:\n{result.stdout}")
            if result.stderr:
                print(f"[Orchestration] stderr:\n{result.stderr}")

            if result.returncode != 0:
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

        print(f"[Orchestration] Compilation successful: {len(binary_data)} bytes")
        return binary_data

    def compile_incore_sim(self, source_path: str) -> bytes:
        """
        Compile a simulation kernel to .o using g++.

        This compiles a simulation kernel (plain C++ code) to an object file,
        which can then have its .text section extracted for execution on host.

        Args:
            source_path: Path to kernel source file (.cpp)

        Returns:
            Binary contents of the compiled .o file

        Raises:
            FileNotFoundError: If source file not found
            RuntimeError: If compilation fails
        """
        source_path = os.path.abspath(source_path)
        if not os.path.isfile(source_path):
            raise FileNotFoundError(f"Source file not found: {source_path}")

        # Generate output path
        timestamp = int(time.time() * 1000)
        output_path = f"/tmp/sim_kernel_{timestamp}_{os.getpid()}.o"

        # Build compilation command
        cmd = [
            "g++", "-c",
            "-O2", "-fPIC", "-fno-plt",
            "-std=c++17",
            "-o", output_path,
            source_path
        ]

        # Print compilation command
        print(f"\n{'='*80}")
        print(f"[SimKernel] Compiling: {source_path}")
        print(f"  Command: {' '.join(cmd)}")
        print(f"{'='*80}\n")

        # Execute
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )

            if result.stdout:
                print(f"[SimKernel] stdout:\n{result.stdout}")
            if result.stderr:
                print(f"[SimKernel] stderr:\n{result.stderr}")

            if result.returncode != 0:
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

        # Clean up temp file
        os.remove(output_path)

        print(f"[SimKernel] Compilation successful: {len(binary_data)} bytes")
        return binary_data
