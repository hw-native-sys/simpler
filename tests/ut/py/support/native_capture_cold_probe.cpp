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

// Preloaded transport boundary observer: host C++ preparation remains allowed,
// but device/pinned allocations and event/stream creation must be outside capture.
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>

namespace {
std::atomic<bool> watching{false};
std::atomic<uint64_t> violations{0}, observed{0};
bool refused() {
    ++observed;
    if (!watching.load()) return false;
    ++violations;
    return true;
}
template <typename Fn>
Fn resolve(const char *library, const char *symbol) {
    void *handle = dlopen(library, RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) std::abort();
    auto fn = reinterpret_cast<Fn>(dlsym(handle, symbol));
    if (fn == nullptr) std::abort();
    return fn;
}
}  // namespace
extern "C" void capture_cold_watch(int enabled) { watching.store(enabled != 0); }
extern "C" uint64_t capture_cold_violations() { return violations.load(); }
extern "C" uint64_t capture_cold_observed() { return observed.load(); }
extern "C" int rtMalloc(void **ptr, uint64_t bytes, uint32_t kind, uint16_t module) {
    if (refused()) return -778;
    static auto fn = resolve<decltype(&rtMalloc)>("libruntime.so", "rtMalloc");
    return fn(ptr, bytes, kind, module);
}
extern "C" int rtMallocHost(void **ptr, uint64_t bytes, uint16_t module) {
    if (refused()) return -778;
    static auto fn = resolve<decltype(&rtMallocHost)>("libruntime.so", "rtMallocHost");
    return fn(ptr, bytes, module);
}
extern "C" int aclrtMalloc(void **ptr, uint64_t bytes, int policy) {
    if (refused()) return -778;
    static auto fn = resolve<decltype(&aclrtMalloc)>("libascendcl.so", "aclrtMalloc");
    return fn(ptr, bytes, policy);
}
extern "C" int aclrtMallocHost(void **ptr, uint64_t bytes) {
    if (refused()) return -778;
    static auto fn = resolve<decltype(&aclrtMallocHost)>("libascendcl.so", "aclrtMallocHost");
    return fn(ptr, bytes);
}
extern "C" int aclrtCreateEventExWithFlag(void **event, uint32_t flag) {
    if (refused()) return -778;
    static auto fn = resolve<decltype(&aclrtCreateEventExWithFlag)>("libascendcl.so", "aclrtCreateEventExWithFlag");
    return fn(event, flag);
}
extern "C" int rtStreamCreate(void **stream, int32_t priority) {
    if (refused()) return -778;
    static auto fn = resolve<decltype(&rtStreamCreate)>("libruntime.so", "rtStreamCreate");
    return fn(stream, priority);
}
extern "C" int rtStreamCreateWithFlags(void **stream, int32_t priority, uint32_t flags) {
    if (refused()) return -778;
    static auto fn = resolve<decltype(&rtStreamCreateWithFlags)>("libruntime.so", "rtStreamCreateWithFlags");
    return fn(stream, priority, flags);
}

extern "C" int rtFree(void *ptr) {
    if (refused()) return -778;
    static auto fn = resolve<decltype(&rtFree)>("libruntime.so", "rtFree");
    return fn(ptr);
}
extern "C" int rtFreeHost(void *ptr) {
    if (refused()) return -778;
    static auto fn = resolve<decltype(&rtFreeHost)>("libruntime.so", "rtFreeHost");
    return fn(ptr);
}
extern "C" int aclrtFree(void *ptr) {
    if (refused()) return -778;
    static auto fn = resolve<decltype(&aclrtFree)>("libascendcl.so", "aclrtFree");
    return fn(ptr);
}
extern "C" int aclrtFreeHost(void *ptr) {
    if (refused()) return -778;
    static auto fn = resolve<decltype(&aclrtFreeHost)>("libascendcl.so", "aclrtFreeHost");
    return fn(ptr);
}
namespace {
using Unregister = int (*)(void *, uint32_t);
std::atomic<Unregister> original_unregister{nullptr};
int observe_unregister(void *ptr, uint32_t device) {
    if (refused()) return -778;
    return original_unregister.load()(ptr, device);
}
}  // namespace
// The production HAL hook resolves through its local library handle. Observe
// that lookup too: ordinary ELF interposition cannot see an explicit dlsym.
extern "C" void *dlsym(void *handle, const char *name) noexcept {
    using Lookup = void *(*)(void *, const char *);
    static auto lookup = [] {
        void *symbol = dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.17");
        if (symbol == nullptr) symbol = dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
        if (symbol == nullptr) std::abort();
        return reinterpret_cast<Lookup>(symbol);
    }();
    void *symbol = lookup(handle, name);
    if (symbol != nullptr && std::strcmp(name, "halHostUnregister") == 0) {
        original_unregister.store(reinterpret_cast<Unregister>(symbol));
        return reinterpret_cast<void *>(&observe_unregister);
    }
    return symbol;
}
