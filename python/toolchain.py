
import os
from enum import IntEnum
from typing import List, Optional
import env_manager


# Must match compile_strategy.h
class ToolchainType(IntEnum):
    """Toolchain types matching the C enum in compile_strategy.h."""
    CCEC = 0           # ccec (Ascend AICore compiler)
    HOST_GXX_15 = 1    # g++-15 (host, simulation kernels)
    HOST_GXX = 2       # g++ (host, orchestration .so)
    AARCH64_GXX = 3    # aarch64-target-linux-gnu-g++ (cross-compile)


class Toolchain:
    """Base class for all compile toolchains.

    A Toolchain represents a compiler identity: which compiler binary to use,
    what flags to pass, and what CMake -D arguments to generate.

    The Ascend SDK path is managed by env_manager. Call
    env_manager.ensure("ASCEND_HOME_PATH") before creating toolchains that
    need the Ascend SDK (CCECToolchain, Aarch64GxxToolchain, GxxToolchain
    with Ascend includes).

    Used by:
    - KernelCompiler: calls get_compile_flags() for direct single-file invocation
    - BuildTarget (in runtime_compiler.py): calls get_cmake_args() for CMake builds
    """

    def __init__(self):
        self.ascend_home_path = env_manager.get("ASCEND_HOME_PATH")

    def get_compile_flags(self, **kwargs) -> List[str]:
        """Return base compiler flags for direct invocation."""
        raise NotImplementedError

    def get_cmake_args(self) -> List[str]:
        """Return compiler-specific CMake -D arguments."""
        raise NotImplementedError


class CCECToolchain(Toolchain):
    """Ascend ccec compiler for AICore kernels."""

    def __init__(self, platform: str = "a2a3"):
        super().__init__()
        self.platform = platform

        if platform in ("a5", "a5sim"):
            self.cxx_path = os.path.join(
                self.ascend_home_path, "tools", "bisheng_compiler", "bin", "ccec"
            )
            self.linker_path = os.path.join(
                self.ascend_home_path, "tools", "bisheng_compiler", "bin", "ld.lld"
            )
        elif platform in ("a2a3", "a2a3sim"):
            self.cxx_path = os.path.join(self.ascend_home_path, "bin", "ccec")
            self.linker_path = os.path.join(self.ascend_home_path, "bin", "ld.lld")
        else:
            raise ValueError(f"Unknown platform: {platform}. Supported: a2a3, a2a3sim, a5, a5sim")

        if not os.path.isfile(self.cxx_path):
            raise FileNotFoundError(
                f"ccec compiler not found: {self.cxx_path}"
            )
        if not os.path.isfile(self.linker_path):
            raise FileNotFoundError(
                f"ccec linker not found: {self.linker_path}"
            )

    def get_compile_flags(self, core_type: str = "aiv", **kwargs) -> List[str]:
        # A5 uses dav-c310 architecture, A2A3 uses dav-c220
        if self.platform in ("a5", "a5sim"):
            arch = "dav-c310-vec" if core_type == "aiv" else "dav-c310-cube"
        elif self.platform in ("a2a3", "a2a3sim"):
            arch = "dav-c220-vec" if core_type == "aiv" else "dav-c220-cube"
        else:
            raise ValueError(f"Unknown platform: {self.platform}. Supported: a2a3, a2a3sim, a5, a5sim")

        core_define = "__AIV__" if core_type == "aiv" else "__AIC__"
        return [
            "-c", "-O3", "-g", "-x", "cce",
            "-Wall", "-std=c++17",
            "--cce-aicore-only",
            f"--cce-aicore-arch={arch}",
            f"-D{core_define}",
            "-mllvm", "-cce-aicore-stack-size=0x8000",
            "-mllvm", "-cce-aicore-function-stack-size=0x8000",
            "-mllvm", "-cce-aicore-record-overflow=false",
            "-mllvm", "-cce-aicore-addr-transform",
            "-mllvm", "-cce-aicore-dcci-insert-for-scalar=false",
            "-DMEMORY_BASE",
        ]

    def get_cmake_args(self) -> List[str]:
        return [
            f"-DBISHENG_CC={self.cxx_path}",
            f"-DBISHENG_LD={self.linker_path}",
        ]


class AscendCToolchain(CCECToolchain):
    """ccec compiler configured for AscendC kernel compilation.

    AscendC uses a different dialect (--cce-aicore-lang) than PTO-ISA (-x cce),
    plus automatic synchronization (--cce-auto-sync) and the full AscendC SDK
    include tree.
    """

    def get_compile_flags(self, core_type: str = "aiv", **kwargs) -> List[str]:
        if self.platform in ("a5", "a5sim"):
            arch = "dav-c310-vec" if core_type == "aiv" else "dav-c310-cube"
        elif self.platform in ("a2a3", "a2a3sim"):
            arch = "dav-c220-vec" if core_type == "aiv" else "dav-c220-cube"
        else:
            raise ValueError(f"Unknown platform: {self.platform}")

        flags = [
            "-c", "-O3", "-std=c++17",
            "--cce-aicore-lang",
            "-DTILING_KEY_VAR=0",
            "--cce-aicore-only",
            f"--cce-aicore-arch={arch}",
            "--cce-auto-sync",
            "-mllvm", "-cce-aicore-stack-size=0x8000",
            "-mllvm", "-cce-aicore-function-stack-size=0x8000",
            "-mllvm", "-cce-aicore-record-overflow=false",
        ]

        # Force-include SDK version header if present
        version_header = os.path.join(
            self.ascend_home_path, "include", "ascendc", "asc_devkit_version.h"
        )
        if os.path.isfile(version_header):
            flags.extend(["-include", version_header])

        return flags

    def get_ascendc_include_dirs(self) -> List[str]:
        """Return AscendC SDK include directories (from bisheng_intf.cmake)."""
        dk = os.path.join(self.ascend_home_path, "aarch64-linux")
        asc = os.path.join(dk, "asc")
        tikcfw = os.path.join(dk, "tikcpp", "tikcfw")

        dirs = [
            os.path.join(asc, "impl", "adv_api"),
            os.path.join(asc, "impl", "basic_api"),
            os.path.join(asc, "impl", "c_api"),
            os.path.join(asc, "impl", "micro_api"),
            os.path.join(asc, "impl", "simt_api"),
            os.path.join(asc, "impl", "utils"),
            os.path.join(asc, "include"),
            os.path.join(asc, "include", "adv_api"),
            os.path.join(asc, "include", "basic_api"),
            os.path.join(asc, "include", "aicpu_api"),
            os.path.join(asc, "include", "c_api"),
            os.path.join(asc, "include", "micro_api"),
            os.path.join(asc, "include", "simt_api"),
            os.path.join(asc, "include", "utils"),
            tikcfw,
            os.path.join(tikcfw, "interface"),
            os.path.join(tikcfw, "impl"),
        ]
        return [d for d in dirs if os.path.isdir(d)]


class Gxx15Toolchain(Toolchain):
    """g++-15 compiler for simulation kernels."""

    def __init__(self):
        super().__init__()
        self.cxx_path = "g++-15"

    def get_compile_flags(self, **kwargs) -> List[str]:
        return [
            "-shared", "-O2", "-fPIC",
            "-std=c++23",
            "-fpermissive",
            "-Wno-macro-redefined",
            "-Wno-ignored-attributes",
            "-D__CPU_SIM",
            "-DPTO_CPU_MAX_THREADS=1",
            "-DNDEBUG",
        ]

    def get_cmake_args(self) -> List[str]:
        # Respect CC/CXX environment variables (e.g., CXX=g++-15 on macOS CI)
        cc = os.environ.get("CC", "gcc")
        cxx = os.environ.get("CXX", "g++")
        args = [
            f"-DCMAKE_C_COMPILER={cc}",
            f"-DCMAKE_CXX_COMPILER={cxx}",
        ]
        return args


class GxxToolchain(Toolchain):
    """g++ compiler for host compilation."""

    def __init__(self):
        super().__init__()
        self.cxx_path = "g++"

    def get_compile_flags(self, **kwargs) -> List[str]:
        return ["-shared", "-fPIC", "-O3", "-g", "-std=c++17"]

    def get_cmake_args(self) -> List[str]:
        # Respect CC/CXX environment variables (e.g., CXX=g++-15 on macOS CI)
        cc = os.environ.get("CC", "gcc")
        cxx = os.environ.get("CXX", "g++")
        args = [
            f"-DCMAKE_C_COMPILER={cc}",
            f"-DCMAKE_CXX_COMPILER={cxx}",
        ]
        if self.ascend_home_path:
            args.append(f"-DASCEND_HOME_PATH={self.ascend_home_path}")
        return args


class Aarch64GxxToolchain(Toolchain):
    """aarch64 cross-compiler for device code."""

    def __init__(self):
        super().__init__()
        self.cxx_path = os.path.join(
            self.ascend_home_path, "tools", "hcc", "bin",
            "aarch64-target-linux-gnu-g++",
        )
        self.cc_path = os.path.join(
            self.ascend_home_path, "tools", "hcc", "bin",
            "aarch64-target-linux-gnu-gcc",
        )
        if not os.path.isfile(self.cc_path):
            raise FileNotFoundError(
                f"aarch64 C compiler not found: {self.cc_path}"
            )
        if not os.path.isfile(self.cxx_path):
            raise FileNotFoundError(
                f"aarch64 C++ compiler not found: {self.cxx_path}"
            )

    def get_compile_flags(self, **kwargs) -> List[str]:
        return ["-shared", "-fPIC", "-O3", "-g", "-std=c++17"]

    def get_cmake_args(self) -> List[str]:
        return [
            f"-DCMAKE_C_COMPILER={self.cc_path}",
            f"-DCMAKE_CXX_COMPILER={self.cxx_path}",
            f"-DASCEND_HOME_PATH={self.ascend_home_path}",
        ]
