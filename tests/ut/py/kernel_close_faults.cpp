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

#include <cstdint>
#include <array>
#include <cstdio>
#include <dlfcn.h>
#include <unistd.h>

#include "common/host_log_state.h"

namespace {
int failure_kind = 0;
int attempts = 0;
bool pending = false;
int failures_to_skip = 0;
SimplerHostLogState test_log_state{};
bool guard_acl = false;
std::array<int, 6> forbidden_calls{};
bool fake_device_drain = false;
int device_drain_calls = 0;
// Targeting an allocation by identity rather than by a call index: a
// device-free failure has to land on a named resource for the test to claim
// anything about that resource's owner. The window is opened immediately
// before the call that allocates it.
bool watch_next_alloc = false;
void *watched_alloc_ptr = nullptr;
int watched_free_hits = 0;
int unload_failures_left = 0;
int unload_calls = 0;
int func_lookup_failures_left = 0;
// Independent of `guard_acl`, which refuses all six ACL/rt symbols at once and
// exists to assert that kernel mode calls none of them. This one fails a
// bounded number of resets and then lets the real call through, so a second
// lifecycle can still come up. It covers the two soft resets only —
// `aclrtResetDeviceForce` belongs to the fatal branch and stays uninjectable.
int reset_failures_left = 0;
int reset_calls = 0;

int forbidden(size_t index) {
    ++forbidden_calls[index];
    return -4322;
}

int destroy(const char *symbol, void *handle, int kind) {
    if (failure_kind == kind) {
        ++attempts;
        if (pending && attempts > failures_to_skip) {
            pending = false;
            return -4321;
        }
    }
    auto real_destroy = reinterpret_cast<int (*)(void *)>(dlsym(RTLD_NEXT, symbol));
    return real_destroy(handle);
}
}  // namespace

extern "C" void arm_destroy_failure(int kind) {
    failure_kind = kind;
    attempts = 0;
    pending = true;
    failures_to_skip = 0;
}

extern "C" void arm_destroy_failure_after(int kind, int skip) {
    failure_kind = kind;
    attempts = 0;
    pending = true;
    failures_to_skip = skip;
}

extern "C" int destroy_attempts() { return attempts; }

extern "C" void arm_acl_guard() {
    forbidden_calls.fill(0);
    guard_acl = true;
}
extern "C" int acl_call_count(int index) { return forbidden_calls.at(index); }
extern "C" void arm_fatal_drain() {
    fake_device_drain = true;
    device_drain_calls = 0;
}
extern "C" int fatal_drain_count() { return device_drain_calls; }

extern "C" int aclInit(const char *config) {
    if (guard_acl) return forbidden(0);
    return reinterpret_cast<int (*)(const char *)>(dlsym(RTLD_NEXT, "aclInit"))(config);
}
extern "C" int aclrtSetDevice(int device) {
    if (guard_acl) return forbidden(1);
    return reinterpret_cast<int (*)(int)>(dlsym(RTLD_NEXT, "aclrtSetDevice"))(device);
}
extern "C" void arm_reset_failures(int count) {
    reset_failures_left = count;
    reset_calls = 0;
}
extern "C" int reset_call_count() { return reset_calls; }

namespace {
bool reset_should_fail() {
    ++reset_calls;
    if (reset_failures_left <= 0) return false;
    --reset_failures_left;
    return true;
}
}  // namespace

extern "C" int aclrtResetDevice(int device) {
    if (guard_acl) return forbidden(2);
    if (reset_should_fail()) return -4321;
    return reinterpret_cast<int (*)(int)>(dlsym(RTLD_NEXT, "aclrtResetDevice"))(device);
}
extern "C" int aclrtResetDeviceForce(int device) {
    if (guard_acl) return forbidden(3);
    return reinterpret_cast<int (*)(int)>(dlsym(RTLD_NEXT, "aclrtResetDeviceForce"))(device);
}
extern "C" int aclFinalize() {
    if (guard_acl) return forbidden(4);
    return reinterpret_cast<int (*)()>(dlsym(RTLD_NEXT, "aclFinalize"))();
}
extern "C" int rtDeviceReset(int device) {
    if (guard_acl) return forbidden(5);
    if (reset_should_fail()) return -4321;
    return reinterpret_cast<int (*)(int)>(dlsym(RTLD_NEXT, "rtDeviceReset"))(device);
}
extern "C" int aclrtSynchronizeDeviceWithTimeout(int32_t timeout) {
    if (fake_device_drain) {
        ++device_drain_calls;
        return 0;
    }
    return reinterpret_cast<int (*)(int32_t)>(dlsym(RTLD_NEXT, "aclrtSynchronizeDeviceWithTimeout"))(timeout);
}

// The ctypes loader has no ChipWorker to bind a process-owned logger sink.
extern "C" int bind_test_log(void *runtime) {
    test_log_state.threshold = 40;
    test_log_state.sink_owner_pid = getpid();
    test_log_state.sink_process_pid = getpid();
    test_log_state.sink_context = &test_log_state;
    test_log_state.sink_enqueue = [](void *, SimplerHostLogState *, const char *record, uint32_t size, int32_t) {
        return std::fwrite(record, 1, size, stderr) == size ? 1 : 0;
    };
    auto bind = reinterpret_cast<SimplerHostLogBindStateFn>(dlsym(runtime, "simpler_host_log_bind_state"));
    return bind == nullptr ? -1 : bind(&test_log_state);
}

extern "C" int rtStreamDestroy(void *stream) { return destroy("rtStreamDestroy", stream, 1); }
extern "C" int aclrtDestroyEvent(void *event) { return destroy("aclrtDestroyEvent", event, 2); }

// Fail the device free of the next allocation to be made, once. `rtMalloc`
// below is what learns the address.
extern "C" void arm_free_failure_for_next_alloc() {
    watch_next_alloc = true;
    watched_alloc_ptr = nullptr;
    watched_free_hits = 0;
}
extern "C" int free_failure_hits() { return watched_free_hits; }
extern "C" int watched_alloc_found() { return watched_alloc_ptr != nullptr ? 1 : 0; }

// Signatures mirror CANN's exactly (rtMemType_t is uint32_t, moduleId is
// uint16_t, rtBinHandle and rtFuncHandle are void *), so interposing forwards
// the caller's arguments unchanged.
extern "C" int rtMalloc(void **ptr, uint64_t size, uint32_t type, uint16_t module_id) {
    auto real = reinterpret_cast<int (*)(void **, uint64_t, uint32_t, uint16_t)>(dlsym(RTLD_NEXT, "rtMalloc"));
    int rc = real(ptr, size, type, module_id);
    if (rc == 0 && watch_next_alloc) {
        watch_next_alloc = false;
        watched_alloc_ptr = *ptr;
    }
    return rc;
}

extern "C" int rtFree(void *ptr) {
    if (watched_alloc_ptr != nullptr && ptr == watched_alloc_ptr) {
        watched_alloc_ptr = nullptr;
        ++watched_free_hits;
        return -4321;
    }
    return destroy("rtFree", ptr, 4);
}

// `rtsBinaryUnload` and `rtsFuncGetByName` are the two calls that decide
// whether the AICPU loader keeps its handle: the first is the release, the
// second is what makes `Init` roll back after the binary is already loaded.
extern "C" void arm_unload_failures(int count) {
    unload_failures_left = count;
    unload_calls = 0;
}
extern "C" int unload_call_count() { return unload_calls; }

extern "C" int rtsBinaryUnload(const void *handle) {
    ++unload_calls;
    if (unload_failures_left > 0) {
        --unload_failures_left;
        return -4321;
    }
    return reinterpret_cast<int (*)(const void *)>(dlsym(RTLD_NEXT, "rtsBinaryUnload"))(handle);
}

extern "C" void arm_func_lookup_failure(int count) { func_lookup_failures_left = count; }

extern "C" int rtsFuncGetByName(const void *handle, const char *name, void **func) {
    if (func_lookup_failures_left > 0) {
        --func_lookup_failures_left;
        return -4321;
    }
    return reinterpret_cast<int (*)(const void *, const char *, void **)>(dlsym(RTLD_NEXT, "rtsFuncGetByName"))(
        handle, name, func
    );
}

extern "C" int aclrtCreateEventExWithFlag(void **event, uint32_t flag) {
    if (failure_kind == 3 && pending) {
        pending = false;
        return -4321;
    }
    auto create = reinterpret_cast<int (*)(void **, uint32_t)>(dlsym(RTLD_NEXT, "aclrtCreateEventExWithFlag"));
    return create(event, flag);
}
