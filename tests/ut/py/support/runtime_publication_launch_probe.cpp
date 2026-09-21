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

#include <acl/acl.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <utility>
#include "device_runner.h"
#include "common/support/native_run_execution_peer.h"

// These process-local RTS symbols interpose the library calls. No device is
// initialized; entering stream/event setup is observable and fails closed.
static int device_ops = 0;
static int copy_error = 0;
extern "C" rtError_t rtMalloc(void **p, uint64_t n, uint32_t, uint16_t) {
    *p = std::malloc(n);
    return *p ? 0 : -1;
}
extern "C" rtError_t rtFree(void *p) {
    std::free(p);
    return 0;
}
extern "C" rtError_t rtMemcpy(void *dst, uint64_t, const void *src, uint64_t n, rtMemcpyKind_t) {
    if (copy_error) return copy_error;
    std::memcpy(dst, src, n);
    return 0;
}
extern "C" rtError_t rtStreamCreate(rtStream_t *, int32_t) {
    ++device_ops;
    return -777;
}
extern "C" aclError aclrtCreateEventExWithFlag(aclrtEvent *, uint32_t) {
    ++device_ops;
    return -777;
}
int main() {
    DeviceRunner runner;
    Runtime runtime;
    MemoryAllocator allocator;
    SlotPersistentArgs slot;
    NativeRunIdentity id{1, 1, 1, 0};
    // Unprepared, skipped publish, failed publish, published then released.
    for (int arm = 0; arm != 4; ++arm) {
        auto prepared = std::make_unique<DeviceRunnerBase::PreparedExecution>(id, runtime, CallConfig{}, 0);
        if (arm) {
            assert(prepared->kernel_args.prepare_runtime_args(runtime, allocator, slot) == 0);
            if (arm == 2) {
                copy_error = -91;
                assert(prepared->kernel_args.publish_runtime_args() == -91);
                copy_error = 0;
            }
            if (arm == 3) {
                assert(prepared->kernel_args.publish_runtime_args() == 0);
                prepared->kernel_args.release_run_view();
            }
        }
        // Skipping publish retains a non-null destination over stale bytes.
        if (arm == 1) assert(prepared->kernel_args.args.runtime_args != nullptr);
        auto *owner = prepared.get();
        const int before = device_ops;
        auto outcome = runner.launch_execution(std::move(prepared), NativeRunExecutionTestPeer::mint(id));
        assert(outcome.rc == PTO_RUNTIME_ERR_INVALID_STATE);
        assert(outcome.progress == LaunchProgress::NotStarted);
        assert(outcome.prepared.get() == owner);
        assert(outcome.active == nullptr);
        assert(device_ops == before);
        assert(!outcome.poisoned());
        outcome.prepared->kernel_args.release_run_view();
    }
    assert(release_slot_persistent_args(slot, allocator) == 0);
    std::puts("PASS: four launch refusals; owner retained, NotStarted, zero stream/event creation");
}
extern "C" void unified_log_warn(const char *, const char *, ...) {}
extern "C" void unified_log_info(const char *, const char *, ...) {}

extern "C" void unified_log_error(const char *, const char *, ...) {}
