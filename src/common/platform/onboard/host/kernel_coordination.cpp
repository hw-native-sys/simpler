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
#include "device_runner_base.h"

#include <acl/acl.h>
#include <cstring>
#include <vector>

#include "host/capture_memcpy.h"
#include "tensormap_and_ringbuffer/kernel_clear_plan.h"

namespace {
using namespace simpler::tmr;

struct alignas(64) CoordinationImage {
    TmrKernelContextDescriptor descriptor;
    TmrLaunchControl control;
    TmrKernelAicoreArgs core;
    TmrContextRevokeReceipt receipt;
};
static_assert(sizeof(CoordinationImage) % alignof(TmrCoreReport) == 0);
}  // namespace

int DeviceRunnerBase::prepare_kernel_coordination() {
    if (kernel_coordination_ready_) return 0;
    if (kernel_coordination_block_ || kernel_revoke_host_receipt_) return PTO_RUNTIME_ERR_INVALID_STATE;
    if (!persistent_args_.is_prepared() || worker_count_ <= 0) return PTO_RUNTIME_ERR_INVALID_STATE;
    const size_t report_bytes = static_cast<size_t>(worker_count_) * sizeof(TmrCoreReport);
    if (report_bytes / sizeof(TmrCoreReport) != static_cast<size_t>(worker_count_) ||
        report_bytes > SIZE_MAX - sizeof(CoordinationImage))
        return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
    const size_t bytes = sizeof(CoordinationImage) + report_bytes;
    std::vector<uint8_t> image(bytes, 0);
    kernel_coordination_block_ = mem_alloc_.alloc(bytes);
    if (!kernel_coordination_block_) return PTO_RUNTIME_ERR_INTERNAL;
    const auto base = reinterpret_cast<uint64_t>(kernel_coordination_block_);
    if (base % alignof(CoordinationImage) != 0 || base > UINT64_MAX - bytes) return PTO_RUNTIME_ERR_INTERNAL;
    auto &d = kernel_descriptor_;
    auto &bank = *arena_banks_[0];
    d.version = kTmrKernelContextVersion;
    d.bytes = sizeof(d);
    d.context_generation = kernel_static_config_.generation();
    d.self_address = base + offsetof(CoordinationImage, descriptor);
    d.resident_runtime = reinterpret_cast<uint64_t>(persistent_args_.args().runtime_args);
    d.resident_kernel_args = reinterpret_cast<uint64_t>(persistent_args_.device_k_args());
    d.heap_base = reinterpret_cast<uint64_t>(bank.gm_heap.base());
    d.heap_capacity = bank.gm_heap.total_size();
    d.heap_required = bank.cached_gm_heap_size;
    d.sm_base = reinterpret_cast<uint64_t>(kernel_runtime_.get_gm_sm_ptr());
    d.sm_capacity = bank.gm_sm.total_size();
    d.sm_required = bank.cached_gm_sm_size;
    d.arena_base = reinterpret_cast<uint64_t>(kernel_runtime_.get_prebuilt_arena_base());
    d.arena_capacity = bank.runtime_pool.total_size();
    d.arena_required = bank.cached_runtime_arena_size;
    d.runtime_offset = kernel_runtime_.get_prebuilt_runtime_offset();
    d.control_address = base + offsetof(CoordinationImage, control);
    d.control_bytes = sizeof(TmrLaunchControl);
    d.reports_address = base + sizeof(CoordinationImage);
    d.reports_bytes = report_bytes;
    d.launch_threads = kernel_runtime_.get_aicpu_launch_count();
    d.execution_threads = kernel_runtime_.get_aicpu_thread_num();
    d.worker_count = worker_count_;
    const TmrKernelClearBinding clear{
        d.context_generation, {d.control_address, d.control_bytes}, {d.reports_address, d.reports_bytes}, d.worker_count
    };
    if (!valid_tmr_clear_binding(clear)) return PTO_RUNTIME_ERR_INTERNAL;
    kernel_core_envelope_ = reinterpret_cast<void *>(base + offsetof(CoordinationImage, core));
    kernel_revoke_device_receipt_ = reinterpret_cast<void *>(base + offsetof(CoordinationImage, receipt));
    CoordinationImage initial{};
    initial.descriptor = d;
    initial.core = {d.resident_kernel_args, d.self_address};
    initial.receipt.descriptor_address = d.self_address;
    initial.receipt.context_generation = d.context_generation;
    std::memcpy(image.data(), &initial, sizeof(initial));
    int rc = capture_memcpy_h2d(kernel_coordination_block_, bytes, image.data(), bytes);
    if (rc != 0) return rc;
    void *host_receipt = nullptr;
    rc = aclrtMallocHost(&host_receipt, sizeof(TmrContextRevokeReceipt));
    if (rc != 0) return rc;
    kernel_revoke_host_receipt_ = static_cast<TmrContextRevokeReceipt *>(host_receipt);
    std::memcpy(kernel_revoke_host_receipt_, &initial.receipt, sizeof(initial.receipt));
    TmrContextRegistrationArgs registration{d.self_address, d.context_generation};
    auto stream = kernel_exec_state_.hidden_stream(KernelStreamKind::Aicpu);
    kernel_revoke_.registration_may_exist();
    rc = launch_aicpu_payload(stream, &registration, sizeof(registration), "simpler_aicpu_prepare_tmr_context", 1);
    if (rc != 0) return rc;
    // Preparation runs this from init, which synchronizes by contract and is
    // outside any capture, so the handshake waits for its own result and a
    // device-side refusal is init's status rather than a later launch's.
    rc = aclrtSynchronizeStreamWithTimeout(stream, PLATFORM_STREAM_SYNC_TIMEOUT_MS);
    if (rc != 0) return rc;
    kernel_coordination_ready_ = true;
    return 0;
}

int DeviceRunnerBase::finalize_kernel_coordination() {
    kernel_coordination_ready_ = false;
    KernelRevokeOps ops;
    ops.context = this;
    ops.enqueue_revoke = [](void *context) noexcept -> int {
        auto &r = *static_cast<DeviceRunnerBase *>(context);
        TmrContextRevokeArgs args{
            r.kernel_descriptor_.self_address, r.kernel_descriptor_.context_generation,
            reinterpret_cast<uint64_t>(r.kernel_revoke_device_receipt_), sizeof(TmrContextRevokeReceipt)
        };
        return r.launch_aicpu_payload(
            r.kernel_exec_state_.hidden_stream(KernelStreamKind::Aicpu), &args, sizeof(args),
            "simpler_aicpu_revoke_tmr_context", 1
        );
    };
    ops.enqueue_receipt_copy = [](void *context) noexcept -> int {
        auto &r = *static_cast<DeviceRunnerBase *>(context);
        return aclrtMemcpyAsync(
            r.kernel_revoke_host_receipt_, sizeof(TmrContextRevokeReceipt), r.kernel_revoke_device_receipt_,
            sizeof(TmrContextRevokeReceipt), ACL_MEMCPY_DEVICE_TO_HOST,
            r.kernel_exec_state_.hidden_stream(KernelStreamKind::Aicpu)
        );
    };
    ops.record_completion = [](void *context) noexcept -> int {
        auto &r = *static_cast<DeviceRunnerBase *>(context);
        return aclrtRecordEvent(
            r.kernel_exec_state_.event(KernelEventKind::Revoke),
            r.kernel_exec_state_.hidden_stream(KernelStreamKind::Aicpu)
        );
    };
    ops.wait_completion = [](void *context) noexcept -> int {
        auto &r = *static_cast<DeviceRunnerBase *>(context);
        aclrtEventRecordedStatus status{};
        const int rc = aclrtQueryEventStatus(r.kernel_exec_state_.event(KernelEventKind::Revoke), &status);
        if (rc != 0 || status == ACL_EVENT_RECORDED_STATUS_COMPLETE) return rc;
        return aclrtSynchronizeStreamWithTimeout(
            r.kernel_exec_state_.hidden_stream(KernelStreamKind::Aicpu), PLATFORM_STREAM_SYNC_TIMEOUT_MS
        );
    };
    ops.validate_receipt = [](void *context) noexcept -> int {
        auto &r = *static_cast<DeviceRunnerBase *>(context);
        const TmrContextRevokeArgs args{
            r.kernel_descriptor_.self_address, r.kernel_descriptor_.context_generation,
            reinterpret_cast<uint64_t>(r.kernel_revoke_device_receipt_), sizeof(TmrContextRevokeReceipt)
        };
        return valid_tmr_context_revoke_receipt(*r.kernel_revoke_host_receipt_, args, TmrRevokeCompletion::Complete) ?
                   0 :
                   PTO_RUNTIME_ERR_INTERNAL;
    };
    int rc = kernel_revoke_.advance(ops);
    if (rc != 0) return rc;
    if (kernel_revoke_host_receipt_) {
        rc = aclrtFreeHost(kernel_revoke_host_receipt_);
        if (rc != 0) return rc;
        kernel_revoke_host_receipt_ = nullptr;
    }
    if (kernel_coordination_block_) {
        rc = mem_alloc_.free(kernel_coordination_block_);
        if (rc != 0) return rc;
        kernel_coordination_block_ = nullptr;
    }
    kernel_core_envelope_ = nullptr;
    kernel_revoke_device_receipt_ = nullptr;
    return 0;
}

void DeviceRunnerBase::abandon_kernel_coordination() {
    kernel_coordination_ready_ = false;
    kernel_coordination_block_ = nullptr;
    kernel_core_envelope_ = nullptr;
    kernel_revoke_device_receipt_ = nullptr;
    // Pinned receipt is retained: an interrupted D2H may still reference it.
}
