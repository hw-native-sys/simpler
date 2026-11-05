# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Check onboard cache helper binding without executing device cache instructions."""

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.skipif(sys.platform != "linux", reason="ELF symbol interposition is Linux-specific")
@pytest.mark.parametrize("arch", ["a2a3", "a5"])
def test_onboard_cache_helpers_bind_to_each_shared_object(tmp_path: Path, arch: str):
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    assert shutil.which(compiler[0]), f"C++ compiler is required: {compiler[0]}"
    wrapper = tmp_path / "wrapper.cpp"
    wrapper.write_text(
        '#include "aicpu/cache_maintenance.h"\n'
        'extern "C" void exercise() {\n'
        "  cache_flush_range(nullptr, 0);\n"
        "  cache_invalidate_range(nullptr, 0);\n"
        "}\n"
    )
    interposer = tmp_path / "interposer.cpp"
    interposer.write_text(
        '#include "aicpu/cache_maintenance.h"\n'
        "static int calls;\n"
        "namespace aicpu_cache_maintenance {\n"
        "void flush_range_impl(const void *, size_t) { ++calls; }\n"
        "void invalidate_range_impl(const void *, size_t) { ++calls; }\n"
        "}\n"
        'extern "C" int count_calls() { return calls; }\n'
    )
    common = [
        *compiler,
        "-std=c++17",
        "-O2",
        "-shared",
        "-fPIC",
        "-I",
        str(_ROOT / "src/common/platform/include"),
    ]
    for name, sources in (
        ("interposer", [interposer]),
        ("dispatcher", [wrapper, _ROOT / "src" / arch / "platform/onboard/aicpu/cache_ops.cpp"]),
        ("runtime", [wrapper, _ROOT / "src" / arch / "platform/onboard/aicpu/cache_ops.cpp"]),
    ):
        result = subprocess.run(
            [*common, *map(str, sources), "-o", str(tmp_path / f"{name}.so")],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
    probe = """
import ctypes
import sys
from pathlib import Path
root = Path(sys.argv[1])
interposer = ctypes.CDLL(str(root / "interposer.so"), mode=ctypes.RTLD_GLOBAL)
libraries = [ctypes.CDLL(str(root / f"{name}.so")) for name in ("dispatcher", "runtime")]
for library in libraries:
    library.exercise()
assert interposer.count_calls() == 0, "cache helper escaped its shared object"
"""
    result = subprocess.run(
        [sys.executable, "-c", probe, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
