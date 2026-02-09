
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

    def get_include_flags(self) -> List[str]:
        """Return -I flags this toolchain always contributes (e.g. Ascend SDK)."""
        return []


class CCECToolchain(Toolchain):
    """Ascend ccec compiler for AICore kernels."""

    def __init__(self):
        super().__init__()
        self.compiler_path = os.path.join(self.ascend_home_path, "bin", "ccec")
        self.linker_path = os.path.join(self.ascend_home_path, "bin", "ld.lld")
        if not os.path.isfile(self.compiler_path):
            raise FileNotFoundError(
                f"ccec compiler not found: {self.compiler_path}"
            )
        if not os.path.isfile(self.linker_path):
            raise FileNotFoundError(
                f"ccec linker not found: {self.linker_path}"
            )

    def get_compile_flags(self, core_type: str = "aiv", **kwargs) -> List[str]:
        arch = "dav-c220-vec" if core_type == "aiv" else "dav-c220-cube"
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
            f"-DBISHENG_CC={self.compiler_path}",
            f"-DBISHENG_LD={self.linker_path}",
        ]


class Gxx15Toolchain(Toolchain):
    """g++-15 compiler for simulation kernels."""

    def __init__(self):
        super().__init__()
        self.compiler_path = "g++-15"

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
        return [
            "-DCMAKE_C_COMPILER=gcc",
            "-DCMAKE_CXX_COMPILER=g++",
        ]


class GxxToolchain(Toolchain):
    """g++ compiler for host compilation."""

    def __init__(self):
        super().__init__()
        self.compiler_path = "g++"

    def get_compile_flags(self, **kwargs) -> List[str]:
        return ["-shared", "-fPIC", "-O3", "-g", "-std=c++17"]

    def get_cmake_args(self) -> List[str]:
        args = [
            "-DCMAKE_C_COMPILER=gcc",
            "-DCMAKE_CXX_COMPILER=g++",
        ]
        if self.ascend_home_path:
            args.append(f"-DASCEND_HOME_PATH={self.ascend_home_path}")
        return args

    def get_include_flags(self) -> List[str]:
        if self.ascend_home_path:
            return [f"-I{os.path.join(self.ascend_home_path, 'include')}"]
        return []


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
        self.compiler_path = self.cxx_path
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

    def get_device_link_flags(self) -> List[str]:
        """Extra flags for device-side orchestration (symbol export for dlsym)."""
        return ["-Wl,--export-dynamic"]

    def get_cmake_args(self) -> List[str]:
        return [
            f"-DCMAKE_C_COMPILER={self.cc_path}",
            f"-DCMAKE_CXX_COMPILER={self.cxx_path}",
            f"-DASCEND_HOME_PATH={self.ascend_home_path}",
        ]

    def get_include_flags(self) -> List[str]:
        return [f"-I{os.path.join(self.ascend_home_path, 'include')}"]

    def get_aicpu_plugin_flags(self) -> List[str]:
        """Extra flags for AICPU orchestration plugin (GNU extensions + static linking)."""
        return ["-std=gnu++17", "-static-libstdc++", "-static-libgcc"]

    def get_aicpu_include_flags(self) -> List[str]:
        """Ascend SDK include paths needed by AICPU plugin compilation."""
        ascend = self.ascend_home_path
        if not ascend:
            return []
        return [
            f"-I{os.path.join(ascend, 'include')}",
            f"-I{os.path.join(ascend, 'include', 'toolchain')}",
            f"-I{os.path.join(ascend, 'pkg_inc', 'base')}",
        ]
