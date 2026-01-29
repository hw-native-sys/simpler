
import os
from typing import List


class AICoreToolchain:
    """
    AICore toolchain for compiling AICore kernels.
    No path validation needed - caller ensures paths are valid.
    """
    def __init__(self, cc: str, ld: str, aicore_dir: str, aicore_binary: str = "aicore_kernel.o"):
        """
        Initialize the AICore toolchain.

        Args:
            cc: Path to the ccec C compiler (e.g., /opt/ascend/bin/ccec)
            ld: Path to the ld.lld linker (e.g., /opt/ascend/bin/ld.lld)
            aicore_dir: Path to the AICore source directory
            aicore_binary: Name of the AICore binary output, default is "aicore_kernel.o"
        """
        self.cc = cc
        self.ld = ld
        self.aicore_dir = os.path.abspath(aicore_dir)
        self.aicore_binary = aicore_binary

    def get_root_dir(self) -> str:
        """Get the AICore source root directory."""
        return self.aicore_dir

    def get_binary_name(self) -> str:
        """Get the output binary name."""
        return self.aicore_binary

    def gen_cmake_args(self, include_dirs: List[str], source_dirs: List[str]) -> str:
        """
        Generate CMake arguments for the AICore toolchain.

        Args:
            include_dirs: List of include directory paths
            source_dirs: List of source directory paths

        Returns:
            String of CMake command-line arguments
        """
        include_dirs = [os.path.abspath(d) for d in include_dirs]
        source_dirs = [os.path.abspath(d) for d in source_dirs]

        include_dirs_list = ";".join(include_dirs)
        source_dirs_list = ";".join(source_dirs)

        return " ".join([
            f"-DBISHENG_CC={self.cc}",
            f"-DBISHENG_LD={self.ld}",
            f"-DCUSTOM_INCLUDE_DIRS={include_dirs_list}",
            f"-DCUSTOM_SOURCE_DIRS={source_dirs_list}",
        ])


class AICPUToolchain:
    """
    AICPU toolchain for compiling AICPU kernels (ARM64 device task scheduler).
    No path validation needed - caller ensures paths are valid.
    """
    def __init__(self, cc: str, cxx: str, aicpu_dir: str, aicpu_binary: str = "libaicpu_kernel.so", ascend_home_path: str = None):
        """
        Initialize the AICPU toolchain.

        Args:
            cc: Path to the cross-compiler C compiler (e.g., aarch64-target-linux-gnu-gcc)
            cxx: Path to the cross-compiler C++ compiler (e.g., aarch64-target-linux-gnu-g++)
            aicpu_dir: Path to the AICPU source directory
            aicpu_binary: Name of the AICPU binary output, default is "libaicpu_kernel.so"
            ascend_home_path: Path to Ascend home directory, defaults to ASCEND_HOME_PATH environment variable
        """
        self.cc = cc
        self.cxx = cxx
        self.aicpu_dir = os.path.abspath(aicpu_dir)
        self.aicpu_binary = aicpu_binary
        self.ascend_home_path = ascend_home_path

    def get_root_dir(self) -> str:
        """Get the AICPU source root directory."""
        return self.aicpu_dir

    def get_binary_name(self) -> str:
        """Get the output binary name."""
        return self.aicpu_binary

    def gen_cmake_args(self, include_dirs: List[str], source_dirs: List[str]) -> str:
        """
        Generate CMake arguments for the AICPU toolchain.

        Args:
            include_dirs: List of include directory paths
            source_dirs: List of source directory paths

        Returns:
            String of CMake command-line arguments
        """
        include_dirs = [os.path.abspath(d) for d in include_dirs]
        source_dirs = [os.path.abspath(d) for d in source_dirs]

        include_dirs_list = ";".join(include_dirs)
        source_dirs_list = ";".join(source_dirs)

        return " ".join([
            f"-DCMAKE_C_COMPILER={self.cc}",
            f"-DCMAKE_CXX_COMPILER={self.cxx}",
            f"-DASCEND_HOME_PATH={self.ascend_home_path}",
            f"-DCUSTOM_INCLUDE_DIRS={include_dirs_list}",
            f"-DCUSTOM_SOURCE_DIRS={source_dirs_list}",
        ])


class HostToolchain:
    """
    Host toolchain for compiling host runtime library (ARM64 CPU shared library).
    No path validation needed - caller ensures paths are valid.
    """
    def __init__(self, cc: str, cxx: str, host_dir: str, binary_name: str = "libhost_runtime.so", ascend_home_path: str = None):
        """
        Initialize the Host toolchain.

        Args:
            cc: Path to the C compiler (e.g., gcc, arm-linux-gcc)
            cxx: Path to the C++ compiler (e.g., g++, arm-linux-g++)
            host_dir: Path to the host source directory
            binary_name: Name of the shared library output, default is "libhost_runtime.so"
            ascend_home_path: Path to Ascend home directory, defaults to ASCEND_HOME_PATH environment variable
        """
        self.cc = cc
        self.cxx = cxx
        self.host_dir = os.path.abspath(host_dir)
        self.binary_name = binary_name
        self.ascend_home_path = ascend_home_path or os.getenv("ASCEND_HOME_PATH", "")

    def get_root_dir(self) -> str:
        """Get the host source root directory."""
        return self.host_dir

    def get_binary_name(self) -> str:
        """Get the output binary name."""
        return self.binary_name

    def gen_cmake_args(self, include_dirs: List[str], source_dirs: List[str]) -> str:
        """
        Generate CMake arguments for the Host toolchain.

        Args:
            include_dirs: List of include directory paths
            source_dirs: List of source directory paths

        Returns:
            String of CMake command-line arguments
        """
        include_dirs = [os.path.abspath(d) for d in include_dirs]
        source_dirs = [os.path.abspath(d) for d in source_dirs]

        include_dirs_list = ";".join(include_dirs)
        source_dirs_list = ";".join(source_dirs)

        return " ".join([
            f"-DCMAKE_C_COMPILER={self.cc}",
            f"-DCMAKE_CXX_COMPILER={self.cxx}",
            f"-DASCEND_HOME_PATH={self.ascend_home_path}",
            f"-DCUSTOM_INCLUDE_DIRS={include_dirs_list}",
            f"-DCUSTOM_SOURCE_DIRS={source_dirs_list}",
        ])


class HostSimToolchain:
    """
    Host simulation toolchain for compiling simulation binaries.
    Uses host gcc/g++ without requiring Ascend SDK paths.
    """
    def __init__(self, cc: str, cxx: str, host_dir: str, binary_name: str = "libhost_runtime.so"):
        """
        Initialize the Host simulation toolchain.

        Args:
            cc: Path to the C compiler (e.g., gcc)
            cxx: Path to the C++ compiler (e.g., g++)
            host_dir: Path to the host source directory
            binary_name: Name of the shared library output
        """
        self.cc = cc
        self.cxx = cxx
        self.host_dir = os.path.abspath(host_dir)
        self.binary_name = binary_name

    def get_root_dir(self) -> str:
        """Get the host source root directory."""
        return self.host_dir

    def get_binary_name(self) -> str:
        """Get the output binary name."""
        return self.binary_name

    def gen_cmake_args(self, include_dirs: List[str], source_dirs: List[str]) -> str:
        """Generate CMake arguments without Ascend dependencies."""
        include_dirs = [os.path.abspath(d) for d in include_dirs]
        source_dirs = [os.path.abspath(d) for d in source_dirs]

        include_dirs_list = ";".join(include_dirs)
        source_dirs_list = ";".join(source_dirs)

        return " ".join([
            f"-DCMAKE_C_COMPILER={self.cc}",
            f"-DCMAKE_CXX_COMPILER={self.cxx}",
            f"-DCUSTOM_INCLUDE_DIRS={include_dirs_list}",
            f"-DCUSTOM_SOURCE_DIRS={source_dirs_list}",
        ])