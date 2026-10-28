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
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "device_runner.h"
#include "aicpu_loader/bootstrap_protocol.h"
#include "aicpu_loader/host/load_aicpu_op.h"
#include "common/host_log_state.h"

namespace {
using namespace simpler_dispatcher;  // NOLINT(build/namespaces)
int arm;
int reads;
int allocations;
int frees;
int launches;
BootstrapResult *pending_result;
uint64_t pending_size;
SimplerHostLogState log_state{};
}  // namespace

extern "C" aclError aclrtMalloc(void **out, size_t bytes, aclrtMemMallocPolicy) {
    if (arm == 15 && bytes == sizeof(BootstrapResult)) {
        *out = nullptr;
        return -1515;
    }
    *out = std::aligned_alloc(64, (bytes + 63) / 64 * 64);
    ++allocations;
    return *out == nullptr ? -1 : 0;
}
extern "C" aclError aclrtFree(void *ptr) {
    std::free(ptr);
    ++frees;
    return 0;
}
extern "C" aclError aclrtMemcpy(void *dst, size_t, const void *src, size_t bytes, aclrtMemcpyKind kind) {
    if (arm == 16 && kind == ACL_MEMCPY_HOST_TO_DEVICE && bytes == sizeof(BootstrapResult)) return -1616;
    if (kind == ACL_MEMCPY_DEVICE_TO_HOST) {
        ++reads;
        if (arm == 6) return -606;
    }
    std::memcpy(dst, src, bytes);
    return 0;
}
extern "C" aclError aclrtSynchronizeStream(aclrtStream) { return arm == 5 ? 507018 : 0; }
extern "C" rtError_t rtAicpuKernelLaunchExWithArgs(
    uint32_t, const char *, uint32_t, const rtAicpuArgsEx_t *args, rtSmDesc_t *, rtStream_t, uint32_t
) {
    ++launches;

    uint64_t device_addr;
    std::memcpy(&device_addr, static_cast<const char *>(args->args) + 40, sizeof(device_addr));
    auto &device = *reinterpret_cast<BootstrapDeviceArgs *>(device_addr);
    auto &result = *reinterpret_cast<BootstrapResult *>(device.result_addr);
    assert(result.status == BootstrapStatus::Pending);
    pending_result = &result;
    pending_size = device.inner_so_len;
    if (arm == 4) return -404;
    if (arm == 1) return 0;
    result.status = BootstrapStatus::Success;
    result.stage = BootstrapStage::Complete;
    result.device_id = static_cast<int32_t>(device.device_id);
    result.requested_bytes = result.written_bytes = device.inner_so_len;
    if (arm == 2) ++result.version;
    if (arm == 3) ++result.device_id;
    if (arm == 7) std::memset(result.path, 'x', sizeof(result.path));
    if (arm == 8) ++result.written_bytes;
    if (arm == 9) result.stage = BootstrapStage::Write;
    if (arm == 10) ++result.magic;
    if (arm == 11) ++result.requested_bytes;
    if (arm == 12) {
        result.status = BootstrapStatus::Failed;
        result.stage = BootstrapStage::Write;
        result.error_number = ENOSPC;
        result.written_bytes = 1;
        result.capacity_valid = 1;
        result.block_bytes = 4096;
        std::strcpy(result.path, "/device/process/runtime.so");
    }
    if (arm == 13) {
        result.status = BootstrapStatus::Failed;
        result.stage = BootstrapStage::Context;
        result.context_error = -ENOSYS;
    }
    if (arm == 14) {
        result.status = BootstrapStatus::Failed;
        result.stage = BootstrapStage::Write;
        result.error_number = ENOSPC;
        result.capacity_errno = ENOSYS;
    }
    if (arm == 17) {
        result.stage = BootstrapStage::CacheHit;
        result.written_bytes = 0;
    }
    if (arm == 18) {
        result.stage = BootstrapStage::CacheHit;
        ++result.written_bytes;
    }
    return 0;
}

int main() {
    log_state.threshold = 40;
    assert(simpler_host_log_bind_state(&log_state) == 0);
    const char binary[] = "runtime";
    for (int test_arm = 0; test_arm != 19; ++test_arm) {
        arm = test_arm;
        pending_result = nullptr;
        reads = allocations = frees = launches = 0;
        host::LoadAicpuOp loader;
        const int expected = arm == 0 || arm == 17 ? 0 :
                             arm == 4              ? -404 :
                             arm == 5              ? 507018 :
                             arm == 6              ? -606 :
                             arm == 15             ? -1515 :
                             arm == 16             ? -1616 :
                                                     PTO_RUNTIME_ERR_INTERNAL;
        assert(loader.BootstrapDispatcher(binary, sizeof(binary), binary, sizeof(binary), nullptr, 12) == expected);
        assert(reads == (arm == 4 || arm == 5 || arm == 15 || arm == 16 ? 0 : 1));
        assert(launches == (arm == 15 || arm == 16 ? 0 : 1));
        if (loader.has_live_resources()) {
            assert(allocations - frees == 4);  // Including the device acknowledgement.
            assert(pending_result != nullptr);
            // The same task eventually publishes a valid result. A close retry
            // consumes it; it must not enqueue a replacement bootstrap.
            *pending_result = {};
            pending_result->status = BootstrapStatus::Success;
            pending_result->stage = BootstrapStage::Complete;
            pending_result->device_id = 12;
            pending_result->requested_bytes = pending_result->written_bytes = pending_size;
            arm = 0;
            assert(loader.Finalize() == 0);
            assert(launches == 1);
        }
        assert(allocations == frees);
    }
    std::puts("PASS: bootstrap result validation, original CANN errors, diagnostic failure and buffer cleanup");
}
