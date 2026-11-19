# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Counter contracts on real shared words, including separately loaded kernel DSOs."""

import ctypes
import subprocess
from pathlib import Path

import pytest

from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.scene_test_cache import _source_closure


@pytest.fixture(scope="module")
def counter_build(tmp_path_factory):
    directory = tmp_path_factory.mktemp("wse_counter")
    compiler = KernelCompiler("wsesim")
    root = compiler.project_root
    flags = [compiler.gxx15.cxx_path, "-std=c++23", "-O2", "-pthread"]
    flags += ["-I" + str(root / "src/wse/platform/include")]
    flags += ["-I" + str(p) for p in compiler.get_orchestration_include_dirs("host_build_graph")]
    binary = directory / "contracts"
    subprocess.run(
        flags + [str(Path(__file__).with_name("wse_counter_contracts.cpp")), "-o", str(binary)],
        check=True,
        timeout=90,
    )
    return directory, flags, binary


@pytest.mark.parametrize("case", ["bounds", "contention", "overflow", "publication", "tensor"])
def test_counter_contracts(counter_build, case):
    subprocess.run([str(counter_build[2]), case], check=True, timeout=20)


def test_counter_shared_across_dsos(counter_build):
    directory, flags, _ = counter_build
    functions = [
        "return wse::counter_signal({p,1},0,3).new_value;",
        "auto r=wse::counter_poll({p,1},0,8); return r.status==wse::CounterStatus::READY ? r.observed : 0;",
    ]
    libraries = []
    for index, body in enumerate(functions):
        source = directory / f"dso{index}.cpp"
        source.write_text('#include <wse/counter_ops.h>\nextern "C" uint64_t access_word(uint64_t *p){' + body + "}\n")
        library = directory / f"dso{index}.so"
        subprocess.run(flags + ["-shared", "-fPIC", str(source), "-o", str(library)], check=True, timeout=60)
        loaded = ctypes.CDLL(str(library), mode=ctypes.RTLD_LOCAL)
        loaded.access_word.argtypes = [ctypes.POINTER(ctypes.c_uint64)]
        loaded.access_word.restype = ctypes.c_uint64
        libraries.append(loaded)
    word, other = ctypes.c_uint64(5), ctypes.c_uint64(1)
    assert libraries[1].access_word(ctypes.byref(word)) == 0
    assert libraries[0].access_word(ctypes.byref(word)) == 8
    assert libraries[1].access_word(ctypes.byref(word)) == 8
    assert libraries[1].access_word(ctypes.byref(other)) == 0
    assert other.value == 1


def test_counter_header_cache_identity(tmp_path):
    compiler = KernelCompiler("wsesim")
    compiler.project_root = tmp_path
    compiler.platform_dir = tmp_path / "src/wse/platform"
    for relative in (
        "platform/include/wse/counter_ops.h",
        "platform/sim/operations/counter_ops.h",
        "runtime/host_build_graph/common/counter_args.h",
    ):
        header = tmp_path / "src/wse" / relative
        header.parent.mkdir(parents=True, exist_ok=True)
        header.write_text("first")
        source = tmp_path / "kernel.cpp"
        source.write_text('#include "' + header.name + '"\n')
        _, first, _ = _source_closure(source, [header.parent])
        header.write_text("updated counter semantics")
        _, second, _ = _source_closure(source, [header.parent])
        assert first != second
