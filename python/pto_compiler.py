import os
import subprocess
import time
from typing import List, Optional


class PTOCompiler:
    """
    Compiler for PTO AICore kernels.

    Compiles single AICore kernel source files to .o using ccec compiler,
    suitable for runtime kernel compilation without CMake.
    """

    def __init__(self, ascend_home_path: Optional[str] = None):
        """
        Initialize PTOCompiler.

        Args:
            ascend_home_path: Path to Ascend toolkit. If None, reads from
                              ASCEND_HOME_PATH environment variable.

        Raises:
            EnvironmentError: If ASCEND_HOME_PATH is not set and not provided
            FileNotFoundError: If ccec compiler not found
        """
        if ascend_home_path is None:
            ascend_home_path = os.getenv("ASCEND_HOME_PATH")

        if not ascend_home_path:
            raise EnvironmentError(
                "ASCEND_HOME_PATH environment variable is not set. "
                "Please `source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash`."
            )

        self.ascend_home_path = ascend_home_path
        self.cc_path = os.path.join(self.ascend_home_path, "bin", "ccec")

        if not os.path.isfile(self.cc_path):
            raise FileNotFoundError(f"ccec compiler not found: {self.cc_path}")

    def compile_kernel(
        self,
        source_path: str,
        core_type: int = 1,
        pto_isa_root: Optional[str] = None,
        extra_include_dirs: Optional[List[str]] = None
    ) -> str:
        """
        Compile a single AICore kernel source file to .o using ccec.

        Args:
            source_path: Path to kernel source file (.cpp)
            core_type: Core type: 0=AIC (cube), 1=AIV (vector). Default: 1 (AIV)
            pto_isa_root: Path to PTO-ISA root directory. Required.
            extra_include_dirs: Additional include directories

        Returns:
            Path to compiled .o file (in /tmp)

        Raises:
            FileNotFoundError: If source file or PTO-ISA headers not found
            ValueError: If pto_isa_root is not provided
            RuntimeError: If compilation fails
        """
        # Validate source file exists
        source_path = os.path.abspath(source_path)
        if not os.path.isfile(source_path):
            raise FileNotFoundError(f"Source file not found: {source_path}")

        # Validate PTO-ISA root
        if pto_isa_root is None:
            raise ValueError("pto_isa_root is required for kernel compilation")

        pto_include = os.path.join(pto_isa_root, "include")
        pto_pto_include = os.path.join(pto_isa_root, "include", "pto")

        if not os.path.isdir(pto_include):
            raise FileNotFoundError(f"PTO-ISA include directory not found: {pto_include}")

        # Generate output path
        timestamp = int(time.time() * 1000)
        output_path = f"/tmp/kernel_{timestamp}_{os.getpid()}.o"

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
        core_type_name = "AIV" if core_type == 1 else "AIC"
        print(f"\n{'='*80}")
        print(f"[Kernel] Compiling ({core_type_name}): {source_path}")
        print(f"  Command: {' '.join(cmd)}")
        print(f"{'='*80}\n")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )

            if result.stdout:
                print(f"[Kernel] stdout:\n{result.stdout}")
            if result.stderr:
                print(f"[Kernel] stderr:\n{result.stderr}")

            if result.returncode != 0:
                raise RuntimeError(
                    f"Kernel compilation failed with exit code {result.returncode}:\n"
                    f"{result.stderr}"
                )

        except FileNotFoundError:
            raise RuntimeError(f"ccec compiler not found at {self.cc_path}")

        # Verify output file exists
        if not os.path.isfile(output_path):
            raise RuntimeError(f"Compilation succeeded but output file not found: {output_path}")

        print(f"[Kernel] Compilation successful: {output_path}")
        return output_path

    def _build_compile_command(
        self,
        source_path: str,
        output_path: str,
        core_type: int,
        pto_include: str,
        pto_pto_include: str,
        extra_include_dirs: Optional[List[str]] = None
    ) -> List[str]:
        """
        Build the ccec compilation command.

        Args:
            source_path: Path to source file
            output_path: Path for output .o file
            core_type: 0=AIC (cube), 1=AIV (vector)
            pto_include: Path to PTO include directory
            pto_pto_include: Path to PTO/pto include directory
            extra_include_dirs: Additional include directories

        Returns:
            List of command arguments
        """
        arch = "dav-c220-vec" if core_type == 1 else "dav-c220-cube"
        define = "__AIV__" if core_type == 1 else "__AIC__"

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
    ) -> str:
        """
        Compile an orchestration function to a shared library (.so).

        The orchestration function must have signature:
            int FuncName(Runtime* runtime, uint64_t* args, int arg_count);

        Args:
            source_path: Path to orchestration source file (.cpp)
            extra_include_dirs: Additional include directories (must include
                               paths to runtime.h and devicerunner.h)

        Returns:
            Path to compiled .so file (in /tmp)

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

        # Add Ascend runtime include
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

        # Verify output file exists
        if not os.path.isfile(output_path):
            raise RuntimeError(f"Compilation succeeded but output file not found: {output_path}")

        print(f"[Orchestration] Compilation successful: {output_path}")
        return output_path
