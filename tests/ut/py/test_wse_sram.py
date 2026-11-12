# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Fixed SRAM configuration, whole-device claims and alias boundaries."""

import json
import subprocess

from simpler_setup.kernel_compiler import KernelCompiler


def test_sram_tensor_span_rejects_overflow(tmp_path):
    compiler = KernelCompiler("wsesim")
    root = compiler.project_root
    source = tmp_path / "span.cpp"
    source.write_text(r"""
#include "sram_access.h"
#include <cassert>
int main() {
    alignas(64) uint64_t data[16]{};
    Tensor tensor{};
    tensor.buffer.addr = reinterpret_cast<uint64_t>(data);
    tensor.buffer.size = sizeof(data);
    tensor.dtype = DataType::UINT64;
    tensor.ndims = 1;
    tensor.shapes[0] = 3;
    tensor.strides[0] = 2;
    tensor.start_offset = 1;
    uint64_t address = 0, bytes = 0;
    assert(wse::tensor_span(tensor, address, bytes));
    assert(address == reinterpret_cast<uint64_t>(data + 1));
    assert(bytes == 5 * sizeof(uint64_t));
    tensor.start_offset = 0;
    tensor.shapes[0] = 65537;
    tensor.strides[0] = 65536;
    assert(!wse::tensor_span(tensor, address, bytes));
    tensor.ndims = 2;
    tensor.shapes[0] = tensor.shapes[1] = UINT32_MAX;
    tensor.strides[0] = tensor.strides[1] = UINT32_MAX;
    assert(!wse::tensor_span(tensor, address, bytes));
    tensor.ndims = 1;
    tensor.shapes[0] = 2;
    tensor.strides[0] = 1;
    tensor.start_offset = UINT64_MAX;
    assert(!wse::tensor_span(tensor, address, bytes));
}
""")
    binary = tmp_path / "span"
    includes = [
        "src/common",
        "src/common/host_build_graph",
        "src/common/task_interface",
        "src/wse/platform/include",
        "src/wse/runtime/host_build_graph/runtime",
    ]
    subprocess.run(
        [compiler.gxx15.cxx_path, "-std=c++17", "-O2"]
        + ["-I" + str(root / path) for path in includes]
        + [str(source), "-o", str(binary)],
        check=True,
        timeout=90,
    )
    subprocess.run([str(binary)], check=True, timeout=20)


def test_sram_native_contracts(tmp_path):
    compiler = KernelCompiler("wsesim")
    root = compiler.project_root
    cores = 40
    capacities = ",".join(str(v) for v in [1024, 2048] + [1024] * 38)
    reserves = ",".join(str(v) for v in [256, 512] + [256] * 38)
    (tmp_path / "wse_hardware_defaults.h").write_text(
        "#pragma once\n#include <cstdint>\nnamespace wse {\n"
        f"constexpr unsigned DEFAULT_CORES = {cores};\n"
        "constexpr unsigned DEFAULT_PES = 4;\n"
        f"constexpr uint64_t DEFAULT_CAPACITIES[] = {{{capacities}}};\n"
        f"constexpr uint64_t DEFAULT_RESERVES[] = {{{reserves}}};\n}}\n"
    )
    source = tmp_path / "contract.cpp"
    source.write_text(r"""
#include "sram_storage.h"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <stdexcept>
#include <limits>
template <class F> void rejected(F f) {
 bool caught=false; try { f(); } catch(const std::exception &) { caught=true; } assert(caught);
}
int main() {
    wse::SramStorage storage;
    constexpr size_t app_bytes=41*768;
    assert(storage.core_count()==40 && storage.committed_bytes()==0);
    rejected([&]{storage.claim(app_bytes-1);});
    auto *whole=storage.claim(app_bytes);
    rejected([&]{storage.claim(app_bytes);});
    auto regions=storage.regions();
    assert(regions.size()==40 && regions[0].base==reinterpret_cast<uint64_t>(whole));
    assert(regions[1].base==regions[0].base+768);
    assert(regions[2].base==regions[1].base+1536);
    assert(storage.committed_bytes()==41*1024);
    assert(regions[0].runtime_base==regions[0].base+app_bytes);
    assert(regions[1].runtime_base==regions[0].runtime_base+256);
    storage.check_application(regions[0].base+app_bytes-1,1);
    rejected([&]{storage.check_application(regions[0].base+app_bytes-1,2);});
    rejected([&]{storage.check_application(regions[0].runtime_base,1);});
    rejected([&]{storage.release(static_cast<char *>(whole)+8);});
    std::memset(whole,37,768);
    std::memset(reinterpret_cast<void *>(regions[1].base),19,1536);
    assert(static_cast<unsigned char *>(whole)[0]==37);
    auto allocations=storage.allocations();
    assert(allocations.size()==40 && allocations[1].owner==1 && allocations[1].bytes==1536);
    storage.set_runtime_used(std::vector<uint64_t>(40,128));
    rejected([&]{storage.set_runtime_used({257,0});});
    auto excess=std::vector<uint64_t>(40,128);excess[0]=257;
    rejected([&]{storage.set_runtime_used(excess);});
    assert(storage.info(0)[6]==128);
    storage.reset_runtime_used();assert(storage.info(0)[6]==0);
    storage.release(whole);
    assert(storage.allocations().empty() && storage.committed_bytes()==41*1024);
    rejected([&]{storage.check_application(reinterpret_cast<uint64_t>(whole),1);});
    rejected([&]{storage.release(whole);});
    assert(storage.claim(app_bytes)==whole);
    wse::SramStorage other;other.freeze();
    assert(other.regions()[0].base!=regions[0].base);
    storage.clear();assert(storage.committed_bytes()==0);

}
""")
    binary = tmp_path / "contract"
    subprocess.run(
        [
            compiler.gxx15.cxx_path,
            "-std=c++17",
            "-O2",
            "-pthread",
            "-I" + str(tmp_path),
            "-I" + str(root / "src/wse/platform/include"),
            "-I" + str(root / "src/wse/platform/sim/host"),
            str(source),
            str(root / "src/wse/platform/sim/host/sram_storage.cpp"),
            "-o",
            str(binary),
        ],
        check=True,
        timeout=90,
    )
    subprocess.run([str(binary)], check=True, timeout=20)


def test_sram_default_config_generation(tmp_path):
    compiler = KernelCompiler("wsesim")
    root = compiler.project_root
    platform = tmp_path / "platform"
    source = platform / "sim/host"
    source.mkdir(parents=True)
    (platform / "config").mkdir()
    (platform / "include/common").mkdir(parents=True)
    (platform / "include/common/hardware_spec.h").write_bytes(
        (root / "src/wse/platform/include/common/hardware_spec.h").read_bytes()
    )
    (source / "dummy.cpp").write_text("int dummy;\n")
    (source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.19)\nproject(defaults CXX)\n"
        "add_library(probe STATIC dummy.cpp)\n"
        f'include("{root / "src/wse/platform/sim/host/hardware_defaults.cmake"}")\n'
        "simpler_configure_wse_hardware(probe)\n"
    )
    config = {
        "hardware": {"core_count": 40, "pe_per_core": 4, "sram_bytes_per_core": [4096, 8192] + [4096] * 38},
        "runtime": {"sram_reserve_bytes_per_core": 1024},
    }
    path = platform / "config/default_simulation.json"
    path.write_text(json.dumps(config))
    command = ["cmake", "-S", str(source), "-B", str(tmp_path / "build")]
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
    header = (tmp_path / "build/generated/wse_hardware_defaults.h").read_text()
    assert "4096ULL,8192ULL," in header and "1024ULL,1024ULL," in header
    config["hardware"]["unknown"] = 1
    path.write_text(json.dumps(config))
    rejected = subprocess.run(command, check=False, capture_output=True, text=True, timeout=60)
    assert rejected.returncode != 0 and "Unknown WSE hardware field" in rejected.stderr
    del config["hardware"]["unknown"]
    config["hardware"]["sram_bytes_per_core"] = [4096]
    path.write_text(json.dumps(config))
    rejected = subprocess.run(command, check=False, capture_output=True, text=True, timeout=60)
    assert rejected.returncode != 0 and "array must match core_count" in rejected.stderr
