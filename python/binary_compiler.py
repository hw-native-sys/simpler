import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Optional
from toolchain import AICoreToolchain, AICPUToolchain, HostToolchain, HostSimToolchain

logger = logging.getLogger(__name__)


# Runtime type to directory mapping
RUNTIME_DIRS = {
    "host_build_graph": "host_build_graph",
    "tensormap_and_ringbuffer": "tensormap_and_ringbuffer",
}


class BinaryCompiler:
    """
    Binary compiler for compiling binaries for multiple target platforms.
    Singleton-per-platform pattern - one instance cached per platform string.

    Supports three target types:
    1. aicore - AICore accelerator kernels
    2. aicpu - AICPU device task scheduler
    3. host - Host runtime library

    Platform determines which toolchains and CMake directories are used:
    - "a2a3": ccec for aicore, aarch64 cross-compiler for aicpu, gcc for host
    - "a2a3sim": all use host gcc/g++ (builds host-compatible .so files)
    """
    _instances = {}

    def __new__(cls, platform: str = "a2a3"):
        if platform not in cls._instances:
            instance = super(BinaryCompiler, cls).__new__(cls)
            instance._initialized = False
            cls._instances[platform] = instance
        return cls._instances[platform]

    def __init__(self, platform: str = "a2a3"):
        if self._initialized:
            return
        self.platform = platform
        self.project_root = Path(__file__).parent.parent
        self.platform_dir = self.project_root / "src" / "platform" / platform

        if not self.platform_dir.is_dir():
            raise ValueError(
                f"Platform '{platform}' not found at {self.platform_dir}"
            )

        if platform == "a2a3":
            self._init_a2a3()
        elif platform == "a2a3sim":
            self._init_a2a3sim()
        else:
            raise ValueError(
                f"Unknown platform: {platform}. Supported: a2a3, a2a3sim"
            )
        self._initialized = True

    def _init_a2a3(self):
        """Initialize toolchains for real a2a3 hardware."""
        self.ascend_home_path = os.environ.get("ASCEND_HOME_PATH")
        if self.ascend_home_path is None:
            raise EnvironmentError(
                "ASCEND_HOME_PATH environment variable not set. "
                "Please set it to your Ascend toolkit installation path, "
                "or use platform='a2a3sim' for simulation."
            )

        # AICore: Bisheng CCE compiler
        cc_path = os.path.join(self.ascend_home_path, "bin", "ccec")
        ld_path = os.path.join(self.ascend_home_path, "bin", "ld.lld")
        if not os.path.isfile(cc_path):
            raise FileNotFoundError(f"Compiler not found: {cc_path}")
        if not os.path.isfile(ld_path):
            raise FileNotFoundError(f"Linker not found: {ld_path}")
        self.aicore_toolchain = AICoreToolchain(
            cc=cc_path, ld=ld_path,
            aicore_dir=str(self.platform_dir / "aicore")
        )

        # AICPU: aarch64 cross-compiler
        cc_path = os.path.join(self.ascend_home_path, "tools", "hcc", "bin", "aarch64-target-linux-gnu-gcc")
        cxx_path = os.path.join(self.ascend_home_path, "tools", "hcc", "bin", "aarch64-target-linux-gnu-g++")
        if not os.path.isfile(cc_path):
            raise FileNotFoundError(f"AICPU C compiler not found: {cc_path}")
        if not os.path.isfile(cxx_path):
            raise FileNotFoundError(f"AICPU C++ compiler not found: {cxx_path}")
        self.aicpu_toolchain = AICPUToolchain(
            cc=cc_path, cxx=cxx_path,
            aicpu_dir=str(self.platform_dir / "aicpu"),
            ascend_home_path=self.ascend_home_path
        )

        # Host: standard gcc/g++
        self._ensure_host_compilers()
        self.host_toolchain = HostToolchain(
            cc="gcc", cxx="g++",
            host_dir=str(self.platform_dir / "host"),
            ascend_home_path=self.ascend_home_path
        )

    def _init_a2a3sim(self):
        """Initialize toolchains for simulation platform.
        All targets use host gcc/g++ with platform-specific CMake dirs.
        No Ascend SDK required.
        """
        self._ensure_host_compilers()

        # All three targets use HostSimToolchain (no ascend_home_path needed)
        self.aicore_toolchain = HostSimToolchain(
            cc="gcc", cxx="g++",
            host_dir=str(self.platform_dir / "aicore"),
            binary_name="libaicore_kernel.so",
        )
        self.aicpu_toolchain = HostSimToolchain(
            cc="gcc", cxx="g++",
            host_dir=str(self.platform_dir / "aicpu"),
            binary_name="libaicpu_kernel.so",
        )
        self.host_toolchain = HostSimToolchain(
            cc="gcc", cxx="g++",
            host_dir=str(self.platform_dir / "host"),
        )

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
        source_dirs: List[str],
    ) -> bytes:
        """
        Compile binary for the specified target platform.

        Args:
            target_platform: Target platform ("aicore", "aicpu", or "host")
            include_dirs: List of include directory paths
            source_dirs: List of source directory paths

        Returns:
            Compiled binary data as bytes

        Raises:
            ValueError: If target platform is invalid
            RuntimeError: If CMake or Make fails
            FileNotFoundError: If output binary not found
        """
        if target_platform == "aicore":
            toolchain = self.aicore_toolchain
        elif target_platform == "aicpu":
            toolchain = self.aicpu_toolchain
        elif target_platform == "host":
            toolchain = self.host_toolchain
        else:
            raise ValueError(
                f"Invalid target platform: {target_platform}. "
                "Must be 'aicore', 'aicpu', or 'host'."
            )

        cmake_args = toolchain.gen_cmake_args(include_dirs, source_dirs)
        cmake_source_dir = toolchain.get_root_dir()
        binary_name = toolchain.get_binary_name()

        return self._run_compilation(
            cmake_source_dir, cmake_args, binary_name, platform=target_platform.upper()
        )

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

            logger.info(f"[{platform}] Running CMake configuration...")
            logger.debug(f"  Working directory: {build_dir}")
            logger.debug(f"  Command: {' '.join(cmake_cmd)}")

            try:
                result = subprocess.run(
                    cmake_cmd,
                    cwd=build_dir,
                    check=False,
                    capture_output=True,
                    text=True
                )

                if result.stdout and logger.isEnabledFor(10):  # DEBUG = 10
                    logger.debug(f"[{platform}] CMake stdout:")
                    logger.debug(result.stdout)
                if result.stderr and logger.isEnabledFor(10):
                    logger.debug(f"[{platform}] CMake stderr:")
                    logger.debug(result.stderr)

                if result.returncode != 0:
                    logger.error(f"[{platform}] CMake configuration failed: {result.stderr}")
                    raise RuntimeError(
                        f"CMake configuration failed for {platform}: {result.stderr}"
                    )
            except FileNotFoundError:
                raise RuntimeError(f"CMake not found. Please install CMake.")

            # Run Make to build
            make_cmd = ["make", "VERBOSE=1"]

            logger.info(f"[{platform}] Running Make build...")
            logger.debug(f"  Working directory: {build_dir}")
            logger.debug(f"  Command: {' '.join(make_cmd)}")

            try:
                result = subprocess.run(
                    make_cmd,
                    cwd=build_dir,
                    check=False,
                    capture_output=True,
                    text=True
                )

                if result.stdout and logger.isEnabledFor(10):  # DEBUG = 10
                    logger.debug(f"[{platform}] Make stdout:")
                    logger.debug(result.stdout)
                if result.stderr and logger.isEnabledFor(10):
                    logger.debug(f"[{platform}] Make stderr:")
                    logger.debug(result.stderr)

                if result.returncode != 0:
                    logger.error(f"[{platform}] Make build failed: {result.stderr}")
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

    def get_runtime_dir(self, runtime_type: str = "host_build_graph") -> Path:
        """
        Get the runtime directory path for the specified runtime type.

        Args:
            runtime_type: Runtime type ("host_build_graph" or "tensormap_and_ringbuffer")

        Returns:
            Path to the runtime directory

        Raises:
            ValueError: If runtime type is invalid
        """
        if runtime_type not in RUNTIME_DIRS:
            raise ValueError(
                f"Invalid runtime type: {runtime_type}. "
                f"Supported: {list(RUNTIME_DIRS.keys())}"
            )
        return self.project_root / "src" / "runtime" / RUNTIME_DIRS[runtime_type]

    def compile_runtime(
        self,
        target_platform: str,
        runtime_type: str = "host_build_graph",
    ) -> bytes:
        """
        Compile runtime binary for the specified target platform and runtime type.

        This method loads the build configuration from the runtime's build_config.py
        and compiles the appropriate binary.

        Args:
            target_platform: Target platform ("aicore", "aicpu", or "host")
            runtime_type: Runtime type ("host_build_graph" or "tensormap_and_ringbuffer")

        Returns:
            Compiled binary data as bytes

        Raises:
            ValueError: If target platform or runtime type is invalid
            RuntimeError: If compilation fails
        """
        runtime_dir = self.get_runtime_dir(runtime_type)

        # Load build config from runtime directory
        import importlib.util
        config_path = runtime_dir / "build_config.py"
        if not config_path.exists():
            raise FileNotFoundError(f"Build config not found: {config_path}")

        spec = importlib.util.spec_from_file_location("build_config", config_path)
        build_config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(build_config_module)
        build_config = build_config_module.BUILD_CONFIG

        if target_platform not in build_config:
            raise ValueError(
                f"Invalid target platform: {target_platform}. "
                f"Available: {list(build_config.keys())}"
            )

        config = build_config[target_platform]

        # Convert relative paths to absolute paths
        include_dirs = [str(runtime_dir / d) for d in config["include_dirs"]]
        source_dirs = [str(runtime_dir / d) for d in config["source_dirs"]]

        print(f"\n[{runtime_type.upper()}] Compiling {target_platform} runtime...")
        print(f"  Runtime dir: {runtime_dir}")
        print(f"  Include dirs: {include_dirs}")
        print(f"  Source dirs: {source_dirs}")

        return self.compile(target_platform, include_dirs, source_dirs)

    def compile_all_runtime(
        self,
        runtime_type: str = "host_build_graph",
    ) -> Dict[str, bytes]:
        """
        Compile all runtime binaries (aicore, aicpu, host) for the specified runtime type.

        Args:
            runtime_type: Runtime type ("host_build_graph" or "tensormap_and_ringbuffer")

        Returns:
            Dictionary mapping target platform to compiled binary data
        """
        return {
            "aicore": self.compile_runtime("aicore", runtime_type),
            "aicpu": self.compile_runtime("aicpu", runtime_type),
            "host": self.compile_runtime("host", runtime_type),
        }

