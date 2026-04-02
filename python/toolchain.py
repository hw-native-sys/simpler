# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import os
import shutil
from enum import IntEnum

import env_manager


def _resolve_compiler(env_var: str, default: str) -> str:
    """Resolve a compiler from an environment variable to a full path.

    Handles the case where the env var contains flags (e.g. sysconfig-derived
    'gcc -pthread -B /path') by extracting only the executable name and resolving
    it via PATH.
    """
    raw = os.environ.get(env_var, "")
    parts = raw.split()
    exe = parts[0] if parts else default
    return shutil.which(exe) or exe


# Must match compile_strategy.h
class ToolchainType(IntEnum):
    """Toolchain types matching the C enum in compile_strategy.h."""

    CCEC = 0  # ccec (Ascend AICore compiler)
    HOST_GXX_15 = 1  # g++-15 (host, simulation kernels)
    HOST_GXX = 2  # g++ (host, orchestration .so)
    AARCH64_GXX = 3  # aarch64-target-linux-gnu-g++ (cross-compile)


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

    cxx_path: str

    def __init__(self):
        self.ascend_home_path = env_manager.get("ASCEND_HOME_PATH")

    def get_compile_flags(self, **kwargs) -> list[str]:
        """Return base compiler flags for direct invocation."""
        raise NotImplementedError

    def get_cmake_args(self) -> list[str]:
        """Return compiler-specific CMake -D arguments."""
        raise NotImplementedError


class CCECToolchain(Toolchain):
    """Ascend ccec compiler for AICore kernels."""

    def __init__(self, platform: str = "a2a3"):
        super().__init__()
        self.platform = platform

        if self.ascend_home_path is None:
            raise RuntimeError("ASCEND_HOME_PATH is required for CCEC toolchain")

        self.cxx_path = os.path.join(self.ascend_home_path, "bin", "ccec")
        self.linker_path = os.path.join(self.ascend_home_path, "bin", "ld.lld")

        if not os.path.isfile(self.cxx_path):
            raise FileNotFoundError(f"ccec compiler not found: {self.cxx_path}")
        if not os.path.isfile(self.linker_path):
            raise FileNotFoundError(f"ccec linker not found: {self.linker_path}")

    def get_compile_flags(self, core_type: str = "aiv", **kwargs) -> list[str]:
        # A5 uses dav-c310 architecture, A2A3 uses dav-c220
        if self.platform in ("a5", "a5sim"):
            arch = "dav-c310-vec" if core_type == "aiv" else "dav-c310-cube"
        elif self.platform in ("a2a3", "a2a3sim"):
            arch = "dav-c220-vec" if core_type == "aiv" else "dav-c220-cube"
        else:
            raise ValueError(f"Unknown platform: {self.platform}. Supported: a2a3, a2a3sim, a5, a5sim")

        return [
            "-c",
            "-O3",
            "-g",
            "-x",
            "cce",
            "-Wall",
            "-std=c++17",
            "--cce-aicore-only",
            f"--cce-aicore-arch={arch}",
            "-mllvm",
            "-cce-aicore-stack-size=0x8000",
            "-mllvm",
            "-cce-aicore-function-stack-size=0x8000",
            "-mllvm",
            "-cce-aicore-record-overflow=false",
            "-mllvm",
            "-cce-aicore-addr-transform",
            "-mllvm",
            "-cce-aicore-dcci-insert-for-scalar=false",
            "-DMEMORY_BASE",
        ]

    def get_cmake_args(self) -> list[str]:
        return [
            f"-DBISHENG_CC={self.cxx_path}",
            f"-DBISHENG_LD={self.linker_path}",
        ]


class Gxx15Toolchain(Toolchain):
    """g++-15 compiler for simulation kernels."""

    def __init__(self):
        super().__init__()
        self.cc_path = _resolve_compiler("CC", "gcc-15")
        self.cxx_path = _resolve_compiler("CXX", "g++-15")

    def get_compile_flags(self, core_type: str = "", **kwargs) -> list[str]:
        flags = [
            "-shared",
            "-O2",
            "-fPIC",
            "-std=c++23",
            "-fpermissive",
            "-Wno-macro-redefined",
            "-Wno-ignored-attributes",
            "-D__CPU_SIM",
            "-DPTO_CPU_MAX_THREADS=1",
            "-DNDEBUG",
        ]
        # g++ does not define __DAV_VEC__/__DAV_CUBE__ like ccec does,
        # so we must add them explicitly based on core_type.
        if core_type == "aiv":
            flags.append("-D__DAV_VEC__")
        elif core_type == "aic":
            flags.append("-D__DAV_CUBE__")
        return flags

    def get_cmake_args(self) -> list[str]:
        return [
            f"-DCMAKE_C_COMPILER={self.cc_path}",
            f"-DCMAKE_CXX_COMPILER={self.cxx_path}",
        ]


class GxxToolchain(Toolchain):
    """g++ compiler for host compilation."""

    def __init__(self):
        super().__init__()
        self.cc_path = _resolve_compiler("CC", "gcc")
        self.cxx_path = _resolve_compiler("CXX", "g++")

    def get_compile_flags(self, **kwargs) -> list[str]:
        return ["-shared", "-fPIC", "-O3", "-g", "-std=c++17"]

    def get_cmake_args(self) -> list[str]:
        args = [
            f"-DCMAKE_C_COMPILER={self.cc_path}",
            f"-DCMAKE_CXX_COMPILER={self.cxx_path}",
        ]
        if self.ascend_home_path:
            args.append(f"-DASCEND_HOME_PATH={self.ascend_home_path}")
        return args


class Aarch64GxxToolchain(Toolchain):
    """aarch64 cross-compiler for device code."""

    def __init__(self):
        super().__init__()

        if self.ascend_home_path is None:
            raise RuntimeError("ASCEND_HOME_PATH is required for aarch64 toolchain")

        self.cxx_path = os.path.join(
            self.ascend_home_path,
            "tools",
            "hcc",
            "bin",
            "aarch64-target-linux-gnu-g++",
        )
        self.cc_path = os.path.join(
            self.ascend_home_path,
            "tools",
            "hcc",
            "bin",
            "aarch64-target-linux-gnu-gcc",
        )
        if not os.path.isfile(self.cc_path):
            raise FileNotFoundError(f"aarch64 C compiler not found: {self.cc_path}")
        if not os.path.isfile(self.cxx_path):
            raise FileNotFoundError(f"aarch64 C++ compiler not found: {self.cxx_path}")

    def get_compile_flags(self, **kwargs) -> list[str]:
        return ["-shared", "-fPIC", "-O3", "-g", "-std=c++17"]

    def get_cmake_args(self) -> list[str]:
        return [
            f"-DCMAKE_C_COMPILER={self.cc_path}",
            f"-DCMAKE_CXX_COMPILER={self.cxx_path}",
            f"-DASCEND_HOME_PATH={self.ascend_home_path}",
        ]
