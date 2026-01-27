import os
import subprocess
import tempfile
from pathlib import Path
from typing import List
from toolchain import AICoreToolchain, AICPUToolchain, HostToolchain


class BinaryCompiler:
    """
    Binary compiler for compiling binaries for multiple target platforms.
    Singleton pattern - only one instance needed.

    Supports three platforms:
    1. aicore - AICore accelerator kernels (Bisheng CCE)
    2. aicpu - AICPU device task scheduler (aarch64 cross-compiler)
    3. host - Host executables (standard C/C++ compiler)
    """
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(BinaryCompiler, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not BinaryCompiler._initialized:
            self.ascend_home_path = os.getenv("ASCEND_HOME_PATH")
            if not self.ascend_home_path:
                raise EnvironmentError(
                    "ASCEND_HOME_PATH environment variable is not set. "
                    "Please `source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash`."
                )
            self.aicore_toolchain = self._gen_aicore_toolchain()
            self.aicpu_toolchain = self._gen_aicpu_toolchain()
            self.host_toolchain = self._gen_host_toolchain()
            BinaryCompiler._initialized = True

    def _gen_aicore_toolchain(self) -> AICoreToolchain:
        """
        Create AICore toolchain from ASCEND_HOME_PATH environment variable.

        Returns:
            AICoreToolchain instance

        Raises:
            EnvironmentError: If ASCEND_HOME_PATH is not set
            FileNotFoundError: If compiler or linker not found
        """
        cc_path = os.path.join(self.ascend_home_path, "bin", "ccec")
        ld_path = os.path.join(self.ascend_home_path, "bin", "ld.lld")

        if not os.path.isfile(cc_path):
            raise FileNotFoundError(f"Compiler not found: {cc_path}")
        if not os.path.isfile(ld_path):
            raise FileNotFoundError(f"Linker not found: {ld_path}")

        aicore_dir = Path(__file__).parent.parent / "src" / "platform" / "a2a3" / "aicore"

        return AICoreToolchain(cc=cc_path, ld=ld_path, aicore_dir=str(aicore_dir))

    def _gen_aicpu_toolchain(self) -> AICPUToolchain:
        """
        Create AICPU toolchain from ASCEND_HOME_PATH environment variable.
        Derives cross-compiler paths from ASCEND_HOME_PATH.

        Returns:
            AICPUToolchain instance

        Raises:
            EnvironmentError: If ASCEND_HOME_PATH is not set
            FileNotFoundError: If cross-compiler not found
        """
        cc_path = os.path.join(self.ascend_home_path, "tools", "hcc", "bin", "aarch64-target-linux-gnu-gcc")
        cxx_path = os.path.join(self.ascend_home_path, "tools", "hcc", "bin", "aarch64-target-linux-gnu-g++")
        if not os.path.isfile(cc_path):
            raise FileNotFoundError(f"AICPU C compiler not found: {cc_path}")
        if not os.path.isfile(cxx_path):
            raise FileNotFoundError(f"AICPU C++ compiler not found: {cxx_path}")

        aicpu_dir = Path(__file__).parent.parent / "src" / "platform" / "a2a3" / "aicpu"

        return AICPUToolchain(cc=cc_path, cxx=cxx_path, aicpu_dir=str(aicpu_dir), ascend_home_path=self.ascend_home_path)

    def _gen_host_toolchain(self) -> HostToolchain:
        """
        Create Host toolchain from standard compilers.

        Returns:
            HostToolchain instance

        Raises:
            FileNotFoundError: If standard compilers not found
        """
        cc_path = "gcc"
        cxx_path = "g++"

        if not self._find_executable(cc_path):
            raise FileNotFoundError(
                f"Host C compiler not found: {cc_path}. "
                "Please install gcc."
            )
        if not self._find_executable(cxx_path):
            raise FileNotFoundError(
                f"Host C++ compiler not found: {cxx_path}. "
                "Please install g++."
            )

        host_dir = Path(__file__).parent.parent / "src" / "platform" / "a2a3" / "host"

        return HostToolchain(cc=cc_path, cxx=cxx_path, host_dir=str(host_dir), ascend_home_path=self.ascend_home_path)

    @staticmethod
    def _find_executable(name: str) -> bool:
        """Check if an executable exists (either as absolute path or in PATH)."""
        if os.path.isfile(name) and os.access(name, os.X_OK):
            return True
        # Check if it's in PATH
        result = subprocess.run(
            ["which", name],
            capture_output=True,
            timeout=1
        )
        return result.returncode == 0

    def compile(
        self,
        target_platform: str,
        include_dirs: List[str],
        source_dirs: List[str]
    ) -> bytes:
        """
        Compile binary for the specified target platform.

        Args:
            target_platform: Target platform ("aicore", "aicpu", or "host")
            include_dirs: List of include directory paths
            source_dirs: List of source directory paths

        Returns:
            Compiled binary data as bytes for all platforms

        Raises:
            ValueError: If target platform is invalid
            RuntimeError: If CMake or Make fails
            FileNotFoundError: If output binary not found
        """
        if target_platform == "aicore":
            return self._compile_aicore(include_dirs, source_dirs)
        elif target_platform == "aicpu":
            return self._compile_aicpu(include_dirs, source_dirs)
        elif target_platform == "host":
            return self._compile_host(include_dirs, source_dirs)
        else:
            raise ValueError(
                f"Invalid target platform: {target_platform}. "
                "Must be 'aicore', 'aicpu', or 'host'."
            )

    def _compile_aicore(self, include_dirs: List[str], source_dirs: List[str]) -> bytes:
        """Compile AICore kernel."""
        toolchain = self.aicore_toolchain
        cmake_args = toolchain.gen_cmake_args(include_dirs, source_dirs)
        cmake_source_dir = toolchain.get_root_dir()
        binary_name = toolchain.get_binary_name()

        return self._run_compilation(
            cmake_source_dir, cmake_args, binary_name, platform="AICore"
        )

    def _compile_aicpu(self, include_dirs: List[str], source_dirs: List[str]) -> bytes:
        """Compile AICPU kernel."""
        toolchain = self.aicpu_toolchain
        cmake_args = toolchain.gen_cmake_args(include_dirs, source_dirs)
        cmake_source_dir = toolchain.get_root_dir()
        binary_name = toolchain.get_binary_name()

        return self._run_compilation(
            cmake_source_dir, cmake_args, binary_name, platform="AICPU"
        )

    def _compile_host(
        self,
        include_dirs: List[str],
        source_dirs: List[str],
    ) -> bytes:
        """Compile host executable.

        Returns:
            Compiled binary data as bytes
        """
        toolchain = self.host_toolchain
        cmake_args = toolchain.gen_cmake_args(include_dirs, source_dirs)
        cmake_source_dir = toolchain.get_root_dir()
        output_binary_name = toolchain.get_binary_name()

        binary_data = self._run_compilation(
            cmake_source_dir, cmake_args, output_binary_name, platform="Host"
        )

        return binary_data

    def _run_compilation(
        self,
        cmake_source_dir: str,
        cmake_args: str,
        binary_name: str,
        platform: str = "AICore"
    ) -> bytes:
        """
        Run CMake configuration and Make build in a temporary directory.

        Args:
            cmake_source_dir: Path to CMake source directory
            cmake_args: CMake command-line arguments string
            binary_name: Name of output binary
            platform: Platform name for logging

        Returns:
            Compiled binary data as bytes

        Raises:
            RuntimeError: If CMake or Make fails
            FileNotFoundError: If output binary not found
        """
        with tempfile.TemporaryDirectory(prefix=f"{platform.lower()}_build_", dir="/tmp") as build_dir:
            # Run CMake configuration
            cmake_cmd = ["cmake", cmake_source_dir] + cmake_args.split()

            # Print CMake command
            print(f"\n{'='*80}")
            print(f"[{platform}] CMake Command:")
            print(f"  Working directory: {build_dir}")
            print(f"  Command: {' '.join(cmake_cmd)}")
            print(f"{'='*80}\n")

            try:
                result = subprocess.run(
                    cmake_cmd,
                    cwd=build_dir,
                    check=False,
                    capture_output=True,
                    text=True
                )

                # Print CMake output
                if result.stdout:
                    print(f"[{platform}] CMake stdout:")
                    print(result.stdout)
                if result.stderr:
                    print(f"[{platform}] CMake stderr:")
                    print(result.stderr)

                if result.returncode != 0:
                    raise RuntimeError(
                        f"CMake configuration failed for {platform}: {result.stderr}"
                    )
            except FileNotFoundError:
                raise RuntimeError(f"CMake not found. Please install CMake.")

            # Run Make to build
            make_cmd = ["make", "VERBOSE=1"]

            # Print Make command
            print(f"\n{'='*80}")
            print(f"[{platform}] Make Command:")
            print(f"  Working directory: {build_dir}")
            print(f"  Command: {' '.join(make_cmd)}")
            print(f"{'='*80}\n")

            try:
                result = subprocess.run(
                    make_cmd,
                    cwd=build_dir,
                    check=False,
                    capture_output=True,
                    text=True
                )

                # Print Make output
                if result.stdout:
                    print(f"[{platform}] Make stdout:")
                    print(result.stdout)
                if result.stderr:
                    print(f"[{platform}] Make stderr:")
                    print(result.stderr)

                if result.returncode != 0:
                    raise RuntimeError(
                        f"Make build failed for {platform}: {result.stderr}"
                    )
            except FileNotFoundError:
                raise RuntimeError(f"Make not found. Please install Make.")

            # Read the compiled binary
            binary_path = os.path.join(build_dir, binary_name)
            if not os.path.isfile(binary_path):
                raise FileNotFoundError(
                    f"Compiled binary not found: {binary_path}. "
                    f"Expected output file name: {binary_name}"
                )

            with open(binary_path, "rb") as f:
                binary_data = f.read()

        return binary_data