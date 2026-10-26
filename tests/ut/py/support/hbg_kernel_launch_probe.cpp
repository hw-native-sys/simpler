/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <dlfcn.h>
#include <cstdlib>
#include <new>
#include "host_build_graph/kernel_graph_slot_protocol.h"
#include "utils/elf_build_id.h"

namespace {
std::atomic<bool> guarded{false};
std::atomic<int> hits{0};
std::atomic<int> fail_record_after{0};
std::atomic<int> fail_free{0};
std::atomic<bool> fail_new{false};
std::atomic<bool> fail_sync{false};
std::atomic<bool> throw_sync{false};
std::atomic<int> fail_cpu_after{0};
std::atomic<bool> throw_cpu{false};
std::atomic<uint64_t> slot_binary_id{0};
std::atomic<bool> fail_slot{false};
}  // namespace
void *operator new(std::size_t bytes) {
    if (fail_new.exchange(false)) throw std::bad_alloc();
    if (void *p = std::malloc(bytes == 0 ? 1 : bytes)) return p;
    throw std::bad_alloc();
}
extern "C" void hbg_fail_host_new() { fail_new.store(true); }
extern "C" void hbg_fail_owned_sync() { fail_sync.store(true); }
extern "C" void hbg_throw_owned_sync() { throw_sync.store(true); }
extern "C" void hbg_guard_hot_path(int enable) {
    if (enable) hits.store(0);
    guarded.store(enable != 0);
}
extern "C" int hbg_forbidden_hot_calls() { return hits.load(); }
extern "C" void hbg_fail_record(int after) { fail_record_after.store(after); }
extern "C" void hbg_fail_free_once() { fail_free.store(1); }
extern "C" void hbg_fail_cpu_launch(int after) { fail_cpu_after.store(after); }
extern "C" void hbg_throw_cpu_launch() { throw_cpu.store(true); }
extern "C" uint64_t hbg_slot_binary_id() { return slot_binary_id.load(); }
extern "C" uint64_t hbg_expected_binary_id(const void *data, size_t bytes) {
    return simpler::common::utils::elf_build_id_64(data, bytes);
}
extern "C" void hbg_fail_slot_publish() { fail_slot.store(true); }
extern "C" int rtKernelLaunchWithHandleV2(
    void *handle, uint64_t key, uint32_t blocks, void *args, void *sm, void *stream, void *config
) {
    // The exception case tests Host recovery without leaving a core waiting for
    // an AICPU task that the injected exception prevents from being submitted.
    if (throw_cpu.load()) return 0;
    auto original =
        reinterpret_cast<decltype(&rtKernelLaunchWithHandleV2)>(dlsym(RTLD_NEXT, "rtKernelLaunchWithHandleV2"));
    return original(handle, key, blocks, args, sm, stream, config);
}
extern "C" int rtsLaunchCpuKernel(void *fn, uint32_t count, void *stream, void *config, void *args) {
    if (throw_cpu.exchange(false)) throw std::bad_alloc();
    // rtCpuKernelArgs_t begins with baseArgs, whose first field is args.
    void *payload = nullptr;
    std::memcpy(&payload, args, sizeof(payload));
    uint32_t magic = 0;
    std::memcpy(&magic, payload, sizeof(magic));
    if (magic == hbg::GRAPH_SLOT_MAGIC) {
        hbg::GraphSlotRegistration registration{};
        std::memcpy(&registration, payload, sizeof(registration));
        slot_binary_id.store(registration.runtime_binary_id);
        if (fail_slot.exchange(false)) return -4329;
    }
    const int remaining = fail_cpu_after.load();
    if (remaining > 0 && fail_cpu_after.fetch_sub(1) == 1) return -4329;
    auto original = reinterpret_cast<decltype(&rtsLaunchCpuKernel)>(dlsym(RTLD_NEXT, "rtsLaunchCpuKernel"));
    return original(fn, count, stream, config, args);
}
extern "C" int aclrtRecordEvent(void *event, void *stream) {
    const int remaining = fail_record_after.load();
    if (remaining > 0 && fail_record_after.fetch_sub(1) == 1) return -4326;
    auto original = reinterpret_cast<decltype(&aclrtRecordEvent)>(dlsym(RTLD_NEXT, "aclrtRecordEvent"));
    return original(event, stream);
}
extern "C" int rtMalloc(void **p, uint64_t n, uint32_t t, uint16_t m) {
    if (guarded.load()) {
        ++hits;
        return -4325;
    }
    using Fn = decltype(&rtMalloc);
    auto original = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "rtMalloc"));
    return original(p, n, t, m);
}
extern "C" int rtFree(void *p) {
    if (fail_free.exchange(0) != 0) return -4327;
    if (guarded.load()) {
        ++hits;
        return -4325;
    }
    using Fn = decltype(&rtFree);
    auto original = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "rtFree"));
    return original(p);
}
extern "C" int aclrtMalloc(void **p, size_t n, int t) {
    if (guarded.load()) {
        ++hits;
        return -4325;
    }
    using Fn = decltype(&aclrtMalloc);
    auto original = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "aclrtMalloc"));
    return original(p, n, t);
}
extern "C" int aclrtFree(void *p) {
    if (guarded.load()) {
        ++hits;
        return -4325;
    }
    using Fn = decltype(&aclrtFree);
    auto original = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "aclrtFree"));
    return original(p);
}
extern "C" int aclrtMemcpy(void *d, size_t c, const void *s, size_t n, int k) {
    if (guarded.load()) {
        ++hits;
        return -4325;
    }
    using Fn = decltype(&aclrtMemcpy);
    auto original = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "aclrtMemcpy"));
    return original(d, c, s, n, k);
}
extern "C" int aclrtMemcpyAsync(void *d, size_t c, const void *s, size_t n, int k, void *st) {
    if (guarded.load()) {
        ++hits;
        return -4325;
    }
    using Fn = decltype(&aclrtMemcpyAsync);
    auto original = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "aclrtMemcpyAsync"));
    return original(d, c, s, n, k, st);
}
extern "C" int aclrtSynchronizeStreamWithTimeout(void *s, int32_t t) {
    if (throw_sync.exchange(false)) throw std::bad_alloc();
    if (fail_sync.exchange(false)) return -4328;
    if (guarded.load()) {
        ++hits;
        return -4325;
    }
    using Fn = decltype(&aclrtSynchronizeStreamWithTimeout);
    auto original = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "aclrtSynchronizeStreamWithTimeout"));
    return original(s, t);
}
extern "C" int aclrtSynchronizeStream(void *s) {
    if (guarded.load()) {
        ++hits;
        return -4325;
    }
    using Fn = decltype(&aclrtSynchronizeStream);
    auto original = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "aclrtSynchronizeStream"));
    return original(s);
}
extern "C" int aclrtSynchronizeDevice() {
    if (guarded.load()) {
        ++hits;
        return -4325;
    }
    using Fn = decltype(&aclrtSynchronizeDevice);
    auto original = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "aclrtSynchronizeDevice"));
    return original();
}
extern "C" int aclmdlRICaptureGetInfo(void *s, int *status, void **model) {
    if (guarded.load()) {
        ++hits;
        return -4325;
    }
    using Fn = decltype(&aclmdlRICaptureGetInfo);
    auto original = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "aclmdlRICaptureGetInfo"));
    return original(s, status, model);
}
extern "C" int rtRegisterAllKernel(const void *b, void **h) {
    if (guarded.load()) {
        ++hits;
        return -4325;
    }
    using Fn = decltype(&rtRegisterAllKernel);
    auto original = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "rtRegisterAllKernel"));
    return original(b, h);
}
