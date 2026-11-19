# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""C++17 runtime storage interoperates with simulator kernels across DSOs."""

import os
import shlex
import subprocess
from pathlib import Path

from simpler_setup.kernel_compiler import KernelCompiler

_LIBRARY = r"""
#include <wse/tile_ops.h>
#include <wse/counter_ops.h>
using Tile = wse::Tile<1, 1>;
using Shared = wse::SharedTile<Tile>;
extern "C" void produce(wse::BlockScope scope, wse::OperationTrace trace) {
    Shared slot(0, 0, 0, 2, scope, trace);
    Tile tile; tile.at(0,0)=11; slot.publish(1,tile);
    tile.at(0,0)=22; slot.publish(2,tile);
}
extern "C" void publish_one(wse::BlockScope scope) {
    Shared slot(0,0,0,2,scope);
    Tile tile; tile.at(0,0)=37; slot.publish(1,tile);
}
extern "C" float consume(wse::BlockScope scope, uint64_t generation) {
    return Shared(1,0,0,2,scope).read(generation).at(0,0);
}
extern "C" void signal_word(uint64_t *words) {
    words[1]=77;
    if(wse::counter_signal({words,1},0).status!=wse::CounterStatus::OK) std::terminate();
}
extern "C" bool poll_word(uint64_t *words) {
    return wse::counter_poll({words,1},0,1).status==wse::CounterStatus::READY;
}
"""

_MAIN = r"""
#include <wse/tile_ops.h>
#include <wse/block_lifecycle.h>
#include <dlfcn.h>
#include <cassert>
#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <thread>

struct Gate {
    std::mutex mutex;
    std::condition_variable changed;
    bool waiting=false;
    static void trace(void *owner, wse::OperationEvent event, uint32_t, uint64_t generation) {
        if(event!=wse::OperationEvent::WAIT || generation!=2) return;
        auto &gate=*static_cast<Gate*>(owner);
        std::lock_guard<std::mutex> lock(gate.mutex);
        gate.waiting=true; gate.changed.notify_all();
    }
    void wait() {
        std::unique_lock<std::mutex> lock(mutex);
        changed.wait(lock,[&]{return waiting;});
    }
};
template<class T> T symbol(void *lib, const char *name) {
    auto value=dlsym(lib,name); assert(value); return reinterpret_cast<T>(value);
}
int main(int argc,char **argv) {
    assert(argc==3);
    void *libs[]={dlopen(argv[1],RTLD_NOW|RTLD_LOCAL),dlopen(argv[2],RTLD_NOW|RTLD_LOCAL)};
    assert(libs[0] && libs[1]);
    for(int producer=0;producer<2;++producer) {
        auto publish=symbol<void(*)(wse::BlockScope,wse::OperationTrace)>(libs[producer],"produce");
        auto publish_one=symbol<void(*)(wse::BlockScope)>(libs[producer],"publish_one");
        auto consume=symbol<float(*)(wse::BlockScope,uint64_t)>(libs[1-producer],"consume");
        wse::BlockOperationStorage storage[2];
        uint64_t epochs[]={1,1};
        wse::BlockScope scopes[]={
            {&epochs[0],1,nullptr,0,&storage[0],wse::BlockOperationStorage::get},
            {&epochs[1],1,nullptr,0,&storage[1],wse::BlockOperationStorage::get}};
        Gate gate;
        std::thread publishing([&]{publish(scopes[0],{&gate,Gate::trace});});
        gate.wait();
        // Core 0 is held at its real generation-reuse wait; Core 1 still progresses.
        publish_one(scopes[1]); assert(consume(scopes[1],1)==37);
        assert(consume(scopes[0],1)==11);
        assert(consume(scopes[0],2)==22);
        publishing.join();
        __atomic_store_n(&epochs[0],uint64_t{0},__ATOMIC_RELEASE);
        storage[0].clear();
        bool rejected=false;
        try {consume(scopes[0],2);} catch(const std::invalid_argument &) {rejected=true;}
        assert(rejected);
        __atomic_store_n(&epochs[0],uint64_t{2},__ATOMIC_RELEASE);
        scopes[0].epoch=2;
        publish_one(scopes[0]); assert(consume(scopes[0],1)==37);

        alignas(8) uint64_t words[2]{};
        auto signal=symbol<void(*)(uint64_t*)>(libs[producer],"signal_word");
        auto poll=symbol<bool(*)(uint64_t*)>(libs[1-producer],"poll_word");
        std::thread reader([&]{while(!poll(words)) {} assert(words[1]==77);});
        signal(words); reader.join();
    }
    assert(dlclose(libs[0])==0 && dlclose(libs[1])==0);
}
"""


def test_host_cpp17_and_kernel_cpp23_share_slot_wakeups(tmp_path):
    root = Path(__file__).resolve().parents[3]
    host = shlex.split(os.environ.get("CXX", "g++"))
    kernel = [KernelCompiler("wsesim").gxx15.cxx_path]
    common = ["-O2", "-pthread", "-I" + str(root / "src/wse/platform/include")]
    source = tmp_path / "kernel.cpp"
    source.write_text(_LIBRARY)
    libraries = []
    for index, (compiler, standard) in enumerate(((host, "c++17"), (kernel, "c++23"))):
        library = tmp_path / f"kernel-{index}.so"
        subprocess.run(
            [*compiler, "-std=" + standard, *common, "-shared", "-fPIC", str(source), "-o", str(library)],
            check=True,
            timeout=90,
        )
        libraries.append(str(library))
    source = tmp_path / "main.cpp"
    source.write_text(_MAIN)
    binary = tmp_path / "main"
    subprocess.run([*host, "-std=c++17", *common, str(source), "-ldl", "-o", str(binary)], check=True, timeout=90)
    subprocess.run([str(binary), *libraries], check=True, timeout=30)
