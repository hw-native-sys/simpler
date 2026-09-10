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
#include "kernel_launch_native.h"

#include <cstring>

namespace simpler::kernel_launch {

namespace {
struct NativeCall {
    const KernelNativeInvocation &native;
    const KernelLaunchGateOps &owner;
};
const KernelNativeInvocation &get(void *context) noexcept { return static_cast<NativeCall *>(context)->native; }

bool span(const void *ptr, size_t bytes) noexcept {
    return ptr && bytes && bytes <= UINTPTR_MAX - reinterpret_cast<uintptr_t>(ptr);
}
int validate(void *context, const KernelInvocationBinding &binding) noexcept {
    const auto &n = get(context);
    if (!n.aicore || !n.aicpu || !n.aicore_blocks || !n.aicpu_blocks || !span(n.aicore_args, n.aicore_args_bytes) ||
        n.aicpu_args != binding.packet || n.aicpu_args_bytes != binding.packet_bytes || !n.clear_regions ||
        !n.clear_region_count || !span(n.cancel.address, n.cancel.bytes) || n.cancel.bytes % sizeof(uint32_t) != 0 ||
        reinterpret_cast<uintptr_t>(n.cancel.address) % alignof(uint32_t) != 0 ||
        n.placeholder_count != binding.placeholder_count ||
        (n.placeholder_count && (!n.placeholders || !binding.placeholders)) ||
        n.placeholder_count > n.aicpu_args_bytes / sizeof(uint64_t))
        return PTO_RUNTIME_ERR_INTERNAL;
    bool cancel_cleared = false;
    for (size_t i = 0; i < n.clear_region_count; ++i) {
        const auto &r = n.clear_regions[i];
        if (!span(r.address, r.bytes)) return PTO_RUNTIME_ERR_INTERNAL;
        const auto base = reinterpret_cast<uintptr_t>(r.address);
        const auto cancel = reinterpret_cast<uintptr_t>(n.cancel.address);
        cancel_cleared |= cancel >= base && cancel - base <= r.bytes && n.cancel.bytes <= r.bytes - (cancel - base);
    }
    if (!cancel_cleared) return PTO_RUNTIME_ERR_INTERNAL;
    for (size_t i = 0; i < n.placeholder_count; ++i) {
        const auto &p = n.placeholders[i];
        if (p.addrOffset != binding.placeholders[i].address_offset ||
            p.dataOffset != binding.placeholders[i].data_offset || p.addrOffset > n.aicpu_args_bytes ||
            sizeof(uint64_t) > n.aicpu_args_bytes - p.addrOffset || p.dataOffset >= n.aicpu_args_bytes)
            return PTO_RUNTIME_ERR_INTERNAL;
        uint64_t address = 0;
        std::memcpy(&address, static_cast<const uint8_t *>(n.aicpu_args) + p.addrOffset, sizeof(address));
        if (address != 0) return PTO_RUNTIME_ERR_INTERNAL;
    }
    return 0;
}
int acquire(void *context, const KernelInvocationBinding &binding, void *caller, KernelLaunchAdmission *out) noexcept {
    const auto &owner = static_cast<NativeCall *>(context)->owner;
    const int admitted = owner.acquire(owner.context, binding, caller, out);
    if (admitted != 0) return admitted;
    const int rc = validate(context, binding);
    if (rc != 0) {
        KernelLaunchResult rejected;
        rejected.status = rc;
        owner.finish(owner.context, rejected);
    }
    return rc;
}
void finish(void *context, const KernelLaunchResult &result) noexcept {
    const auto &owner = static_cast<NativeCall *>(context)->owner;
    owner.finish(owner.context, result);
}
int query(void *, void *event, bool *complete) noexcept {
    aclrtEventRecordedStatus status{};
    const int rc = aclrtQueryEventStatus(event, &status);
    *complete = rc == 0 && status == ACL_EVENT_RECORDED_STATUS_COMPLETE;
    return rc;
}
int wait(void *, void *stream, void *event) noexcept { return aclrtStreamWaitEvent(stream, event); }
int record(void *, void *event, void *stream) noexcept { return aclrtRecordEvent(event, stream); }
int clear(void *context, void *stream) noexcept {
    const auto &n = get(context);
    for (size_t i = 0; i < n.clear_region_count; ++i) {
        const auto &r = n.clear_regions[i];
        const int rc = aclrtMemsetAsync(r.address, r.bytes, 0, r.bytes, stream);
        if (rc != 0) return rc;
    }
    return 0;
}
int cancel(void *context, void *stream) noexcept {
    const auto &r = get(context).cancel;
    // Every pre-window cancel word remains UINT32_MAX until the AICore tail.
    return aclrtMemsetAsync(r.address, r.bytes, 0xff, r.bytes, stream);
}
int core(void *context, void *stream) noexcept {
    const auto &n = get(context);
    return aclrtLaunchKernel(n.aicore, n.aicore_blocks, n.aicore_args, n.aicore_args_bytes, stream);
}
int cpu(void *context, void *stream) noexcept {
    const auto &n = get(context);
    const int rc = aclrtLaunchKernelWithHostArgs(
        n.aicpu, n.aicpu_blocks, stream, n.aicpu_config, n.aicpu_args, n.aicpu_args_bytes, n.placeholders,
        n.placeholder_count
    );
    // HostArgs has copied task-owned storage before returning. Writable staging
    // retains zero placeholders for reuse; the immutable template is untouched.
    for (size_t i = 0; i < n.placeholder_count; ++i) {
        const uint64_t zero = 0;
        std::memcpy(static_cast<uint8_t *>(n.aicpu_args) + n.placeholders[i].addrOffset, &zero, sizeof(zero));
    }
    return rc;
}
}  // namespace

KernelLaunchResult launch_bound_kernel_native(
    const KernelInvocationBinding &binding, const KernelNativeInvocation &native, void *caller_stream,
    const KernelLaunchGateOps &owner
) {
    if (!owner.acquire || !owner.finish) {
        KernelLaunchResult rejected;
        rejected.status = PTO_RUNTIME_ERR_INTERNAL;
        return rejected;
    }
    NativeCall call{native, owner};
    const KernelLaunchGateOps gate{&call, acquire, finish, query};
    const KernelLaunchOps ops{&call, wait, clear, record, cpu, core, cancel};
    return launch_bound_kernel(binding, caller_stream, gate, ops);
}

}  // namespace simpler::kernel_launch
