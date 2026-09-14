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
extern "C" int aclrtResetDevice(int device) {
    if (guard_acl) return forbidden(2);
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
extern "C" int rtFree(void *ptr) { return destroy("rtFree", ptr, 4); }

extern "C" int aclrtCreateEventExWithFlag(void **event, uint32_t flag) {
    if (failure_kind == 3 && pending) {
        pending = false;
        return -4321;
    }
    auto create = reinterpret_cast<int (*)(void **, uint32_t)>(dlsym(RTLD_NEXT, "aclrtCreateEventExWithFlag"));
    return create(event, flag);
}
