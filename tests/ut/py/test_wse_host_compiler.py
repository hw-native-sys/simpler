# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""C++17 PE runtime and C++23 Linx kernel DSOs share a Block context."""

import os
import shlex
import subprocess
from pathlib import Path

from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.pto_isa import ensure_pto_isa_root

_ARGS = r"""
#pragma once
#include <wse/block_scope.h>
#include <wse/operation_trace.h>
struct Call {
    wse::BlockScope scope;
    wse::OperationTrace trace;
    float value[2];
    unsigned count;
};
"""

_LIBRARY = r"""
#include "call.h"
#include <wse/tile_ops.h>
#include <wse/counter_ops.h>
using TestSharedShape = pto::Tile<pto::Location::Vec, float, 4, 8, pto::BLayout::RowMajor, 1, 1>;
extern "C" void kernel_entry(int64_t *args) {
    auto &call = *reinterpret_cast<Call *>(args[0]);
    wse::SharedTileSlot<0, TestSharedShape> slot(3, 2, call.scope, call.trace);
    for (unsigned i = 0; i < call.count; ++i) {
        auto memory = wse::tile_memory<1, 1>(&call.value[i], 1);
        if (::get_thread_idx() == 3) {
            slot.publish(i + 1, memory);
        } else {
            if (::get_thread_idx() != 1) throw std::logic_error("wrong Linx PE");
            auto tile = slot.read(i + 1);
            TSTORE(memory, tile);
        }
    }
}
extern "C" void signal_word(uint64_t *words) {
    words[1] = 77;
    if (wse::counter_signal({words, 1}, 0).status != wse::CounterStatus::OK) std::terminate();
}
extern "C" bool poll_word(uint64_t *words) {
    return wse::counter_poll({words, 1}, 0, 1).status == wse::CounterStatus::READY;
}
"""

_MAIN = r"""
#include "call.h"
#include <wse/linx_kernel_abi.h>
#include <cassert>
#include <condition_variable>
#include <dlfcn.h>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
struct Gate {
    std::mutex mutex;
    std::condition_variable changed;
    bool waiting = false;
    static void trace(void *owner, wse::OperationEvent event, uint32_t, uint64_t) {
        if (event != wse::OperationEvent::WAIT) return;
        auto &gate = *static_cast<Gate *>(owner);
        std::lock_guard lock(gate.mutex);
        gate.waiting = true;
        gate.changed.notify_all();
    }
    void wait() {
        std::unique_lock lock(mutex);
        changed.wait(lock, [&] { return waiting; });
    }
};
template<class T> T symbol(void *lib, const char *name) {
    auto value = dlsym(lib, name);
    assert(value);
    return reinterpret_cast<T>(value);
}
int main(int argc, char **argv) {
    assert(argc == 4);
    int visibility = std::string(argv[3]) == "global" ? RTLD_GLOBAL : RTLD_LOCAL;
    void *libs[] = {dlopen(argv[1], RTLD_NOW | visibility), dlopen(argv[2], RTLD_NOW | RTLD_LOCAL)};
    assert(libs[0] && libs[1]);
    auto descriptor = [](void *lib) {
        auto entry = symbol<wse::KernelEntry>(lib, "kernel_entry");
        auto bind = symbol<wse::BindLinxKernel>(lib, "simpler_wse_linx_kernel");
        const auto *kernel = bind(entry);
        assert(kernel->entry == entry && kernel == bind(entry));
        return kernel;
    };
    const auto *producer = descriptor(libs[0]), *consumer = descriptor(libs[1]);
    assert(producer != consumer && producer->entry != consumer->entry);
    wse::SharedSlotTable slots[2]{};
    uint64_t epochs[] = {1, 1};
    wse::BlockScope scopes[] = {
        {&epochs[0], 1, nullptr, 0, &slots[0]}, {&epochs[1], 1, nullptr, 0, &slots[1]}};
    auto invoke = [](const wse::LinxKernel *kernel, void *core, unsigned pe, Call &call) {
        int64_t args[] = {reinterpret_cast<int64_t>(&call)};
        kernel->invoke(core, pe, kernel->entry, args);
    };
    auto *core = producer->create_core(0b1010);
    Gate gate;
    Call first{scopes[0], {&gate, Gate::trace}, {11, 22}, 2}, output{scopes[0], {}, {}, 2};
    std::thread publishing([&] { invoke(producer, core, 3, first); });
    gate.wait();
    // Another Core can publish and consume the same slot while Core 0 waits for reuse.
    auto *other = consumer->create_core(0b1010);
    Call second{scopes[1], {}, {37}, 1}, other_output{scopes[1], {}, {}, 1};
    invoke(producer, other, 3, second);
    invoke(consumer, other, 1, other_output);
    assert(other_output.value[0] == 37);
    consumer->destroy_core(other);
    invoke(consumer, core, 1, output);
    publishing.join();
    assert(output.value[0] == 11 && output.value[1] == 22);
    producer->destroy_core(core);

    ++epochs[0];
    slots[0] = {};
    core = producer->create_core(0b1010);
    bool rejected = false;
    try { invoke(consumer, core, 1, output); }
    catch (const std::invalid_argument &) { rejected = true; }
    assert(rejected);
    producer->destroy_core(core);

    alignas(8) uint64_t words[2]{};
    auto signal = symbol<void(*)(uint64_t *)>(libs[0], "signal_word");
    auto poll = symbol<bool(*)(uint64_t *)>(libs[1], "poll_word");
    std::thread reader([&] { while (!poll(words)) {} assert(words[1] == 77); });
    signal(words);
    reader.join();
    assert(dlclose(libs[0]) == 0 && dlclose(libs[1]) == 0);
}
"""


def test_host_cpp17_and_kernel_cpp23_share_slot_wakeups(tmp_path):
    root = Path(__file__).resolve().parents[3]
    host = shlex.split(os.environ.get("CXX", "g++"))
    kernel = KernelCompiler("wsesim")
    (tmp_path / "call.h").write_text(_ARGS)
    source = tmp_path / "kernel.cpp"
    source.write_text(_LIBRARY)
    binary = kernel.compile_incore(str(source), core_type="", pto_isa_root=ensure_pto_isa_root())
    libraries = [tmp_path / f"kernel-{index}.so" for index in range(2)]
    for library in libraries:
        library.write_bytes(binary)
    source = tmp_path / "main.cpp"
    source.write_text(_MAIN)
    program = tmp_path / "main"
    subprocess.run(
        [
            *host,
            "-std=c++17",
            "-O2",
            "-pthread",
            "-I" + str(root / "src/wse/platform/include"),
            str(source),
            "-ldl",
            "-o",
            str(program),
        ],
        check=True,
        timeout=90,
    )
    for visibility in ("local", "global"):
        subprocess.run([str(program), *libraries, visibility], check=True, timeout=30)
