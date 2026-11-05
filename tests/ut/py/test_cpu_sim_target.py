# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compile and execute the target-dependent PTO CPU paths through simpler."""

import json
import os
import subprocess
import sys

import pytest

from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.pto_isa import ensure_pto_isa_root

_KERNEL = r"""
#include <cstdint>
#include <pto/pto-inst.hpp>

extern "C" void kernel_entry(float* out) {
    using Input = pto::Tile<pto::TileType::Vec, float, 8, 64, pto::BLayout::RowMajor, 8, 64>;
    using Output = pto::Tile<pto::TileType::Vec, float, 8, 1, pto::BLayout::ColMajor, 8, 1>;
    Input input;
    Output output;
    Input scratch;
    pto::TASSIGN(input, 0);
    pto::TASSIGN(output, 4096);
    pto::TASSIGN(scratch, 8192);
    const float pattern[] = {1e8f, 1.0f, -1e8f, 1.0f};
    for (int i = 0; i < 8 * 64; ++i) {
        input.data()[i] = pattern[i % 4];
    }
    pto::TROWSUM(output, input, scratch);
    for (int i = 0; i < 8; ++i) {
        out[i] = output.data()[i];
    }
    auto& memory = pto::NPUMemoryModel::Instance();
    out[8] = memory.GetArch() == pto::NPUArch::A5 ? 5.0f : 2.0f;
    if (memory.GetArch() == pto::NPUArch::A5) {
        Input high;
        pto::TASSIGN(high, 200 * 1024);
        high.data()[0] = 17.0f;
        out[9] = high.data()[0];
    } else {
        out[9] = 0.0f;
    }
}
"""

_RUN = r"""
import ctypes
import json
import sys
from concurrent.futures import ThreadPoolExecutor

libraries = [ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL) for path in sys.argv[1:]]
for library in libraries:
    library.kernel_entry.argtypes = [ctypes.POINTER(ctypes.c_float)]
    library.kernel_entry.restype = None
    library.simpler_cpu_sim_target.restype = ctypes.c_char_p

def invoke(library):
    out = (ctypes.c_float * 10)()
    library.kernel_entry(out)
    return [library.simpler_cpu_sim_target().decode(), list(out)]

with ThreadPoolExecutor(max_workers=1) as worker:
    outputs = [worker.submit(invoke, library).result() for library in libraries * 2]
print(json.dumps(outputs))
"""


@pytest.mark.parametrize("core_type", ["aic", "aiv"])
def test_compiled_sim_targets_keep_semantics_on_reused_worker(tmp_path, core_type):
    isa_root = ensure_pto_isa_root()
    source = tmp_path / "target.cpp"
    source.write_text(_KERNEL)
    artifacts = {}
    tokens = {}
    for platform in ("a2a3sim", "a5sim"):
        compiler = KernelCompiler(platform)
        binary = compiler.compile_incore(str(source), core_type=core_type, pto_isa_root=isa_root)
        path = tmp_path / f"{platform}.so"
        path.write_bytes(binary)
        artifacts[platform] = str(path)
        tokens[platform] = compiler.incore_compile_cache_token(core_type)
    assert tokens["a2a3sim"] != tokens["a5sim"]

    expected = {"a2a3sim": [1.0] * 8 + [2.0, 0.0], "a5sim": [0.0] * 8 + [5.0, 17.0]}
    env = dict(os.environ)
    env.pop("PTO_CPU_SIM_ARCH", None)
    for order in (("a2a3sim", "a5sim"), ("a5sim", "a2a3sim")):
        result = subprocess.run(
            [sys.executable, "-c", _RUN, *(artifacts[p] for p in order)],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
        assert json.loads(result.stdout) == [[p, expected[p]] for p in order * 2]


def test_sim_compile_rejects_isa_without_target_interface(tmp_path):
    source = tmp_path / "kernel.cpp"
    source.write_text('extern "C" void kernel_entry() {}\n')
    (tmp_path / "include").mkdir()
    with pytest.raises(RuntimeError, match="target_config.hpp"):
        KernelCompiler("a5sim").compile_incore(str(source), core_type="aiv", pto_isa_root=str(tmp_path))


def test_sim_compile_defaults_to_managed_isa(tmp_path):
    source = tmp_path / "kernel.cpp"
    source.write_text('extern "C" void kernel_entry() {}\n')
    assert KernelCompiler("a5sim").compile_incore(str(source), core_type="aiv")
