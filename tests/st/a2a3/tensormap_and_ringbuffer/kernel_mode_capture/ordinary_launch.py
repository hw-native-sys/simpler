# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Caller-owned native AIV tasks, independent of Simpler submission."""

import ctypes
import os
import platform
import subprocess
from pathlib import Path

from simpler_setup.toolchain import CCECToolchain


def build_ordinary(build):
    toolchain = CCECToolchain("a2a3")
    cann = Path(os.environ["ASCEND_HOME_PATH"])
    source = Path(__file__).parent
    core = build / "ordinary.o"
    vector_object = build / "ordinary_vector.o"
    library = build / "ordinary.so"
    includes = [
        cann / suffix
        for suffix in (
            "asc",
            "asc/include",
            "asc/include/basic_api",
            "asc/impl/basic_api",
            "include",
            "include/external",
            "include/c_api",
            "include/c_api/internal",
            "include/ascendc/basic_api",
            "include/ascendc/impl/basic_api",
        )
    ]
    subprocess.run(
        [
            str(cann / "bin/ccec"),
            "-c",
            "-O2",
            "-x",
            "cce",
            "-std=c++17",
            "--cce-aicore-only",
            "--cce-aicore-arch=dav-c220-vec",
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
            *[f"-I{p}" for p in includes if p.is_dir()],
            str(source / "ordinary_core.cpp"),
            "-o",
            str(vector_object),
        ],
        check=True,
        timeout=180,
    )
    subprocess.run(
        [
            toolchain.linker_path,
            "-m",
            "aicorelinux",
            "-Ttext=0",
            "-static",
            "-n",
            "-o",
            str(core),
            str(vector_object),
        ],
        check=True,
        timeout=180,
    )
    pkg = cann / f"{platform.machine()}-linux/pkg_inc"
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-shared",
            "-fPIC",
            f"-I{cann / 'include'}",
            f"-I{pkg}",
            f"-I{pkg / 'runtime'}",
            f"-I{pkg / 'runtime/runtime'}",
            f"-I{pkg / 'profiling'}",
            str(source / "ordinary_launch.cpp"),
            f"-L{cann / 'lib64'}",
            f"-Wl,-rpath,{cann / 'lib64'}",
            "-lruntime",
            "-o",
            str(library),
        ],
        check=True,
        timeout=180,
    )
    lib = ctypes.CDLL(str(library))
    lib.ordinary_open.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
    lib.ordinary_launch.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_uint64, ctypes.c_float]
    lib.ordinary_close.argtypes = [ctypes.c_void_p]
    for name in ("ordinary_open", "ordinary_launch", "ordinary_close"):
        getattr(lib, name).restype = ctypes.c_int
    return lib, core
