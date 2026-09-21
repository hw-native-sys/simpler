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
#include <runtime/rt.h>

#include "host/kernel_launch_binder.h"
#include "tensormap_and_ringbuffer/kernel_clear_plan.h"

int DeviceRunnerBase::launch_kernel_callable(
    int32_t callable_id, const ChipStorageTaskArgs &args, void *caller_stream
) {
    std::unique_lock<std::mutex> lease(kernel_submission_mutex_, std::try_to_lock);
    if (!lease.owns_lock() || !kernel_context_claim_.held() || !kernel_exec_state_.accepts_dispatch() ||
        !persistent_args_.is_prepared() || !kernel_coordination_ready_)
        return PTO_RUNTIME_ERR_INVALID_STATE;
    int rc = adopt_borrowed_device(device_id_);
    if (rc != 0) return rc;
    KernelCallableResidency residency;
    rc = kernel_callable_cache_.resolve(callable_id, residency);
    if (rc != 0) return rc;
    auto it = callables_.find(callable_id);
    if (it == callables_.end() || !kernel_aicpu_handle_ || !aicore_bin_handle_)
        return PTO_RUNTIME_ERR_CALLABLE_NOT_RESIDENT;
    auto &packet = it->second.kernel_packet;
    const simpler::tmr::TmrExecutionBindingView binding{
        reinterpret_cast<uint64_t>(persistent_args_.device_k_args()), kernel_static_config_.generation()
    };
    if (packet.encode(
            args, residency.device_address, residency.bytes, binding, arena_banks_[0]->cached_gm_sm_size,
            arena_banks_[0]->cached_runtime_arena_size
        ) != simpler::kernel::InvocationStatus::Ok)
        return PTO_RUNTIME_ERR_INTERNAL;

    namespace kl = simpler::kernel_launch;
    struct Submission {
        DeviceRunnerBase *runner;
        void *caller;
        const uint8_t *packet;
        size_t bytes;
        simpler::tmr::TmrKernelClearPlan clear;
    };
    const auto &d = kernel_descriptor_;
    simpler::tmr::TmrKernelClearPlan clear;
    if (!simpler::tmr::build_tmr_kernel_clear_plan(
            {d.context_generation,
             {d.control_address, d.control_bytes},
             {d.reports_address, d.reports_bytes},
             d.worker_count},
            &clear
        ))
        return PTO_RUNTIME_ERR_INTERNAL;
    Submission submission{this, caller_stream, packet.packet().data, packet.packet().size, clear};
    kl::KernelLaunchGateOps gate;
    gate.context = &submission;
    gate.acquire = [](void *context, const kl::KernelInvocationBinding &, void *,
                      kl::KernelLaunchAdmission *out) noexcept -> int {
        auto &s = *static_cast<Submission *>(context);
        auto &r = *s.runner;
        if (!r.kernel_exec_state_.accepts_dispatch()) return PTO_RUNTIME_ERR_INVALID_STATE;
        auto &h = out->handles;
        h.aicpu = r.kernel_exec_state_.hidden_stream(KernelStreamKind::Aicpu);
        h.aicore = r.kernel_exec_state_.hidden_stream(KernelStreamKind::Aicore);
        h.start = r.kernel_exec_state_.event(KernelEventKind::Start);
        h.aicore_start = r.kernel_exec_state_.event(KernelEventKind::AicoreStart);
        h.aicore_done = r.kernel_exec_state_.event(KernelEventKind::AicoreDone);
        h.aicpu_done = r.kernel_exec_state_.event(KernelEventKind::AicpuDone);
        h.serial_tail = r.kernel_exec_state_.event(KernelEventKind::SerialTail);
        out->previous_caller_identity = r.kernel_previous_caller_;
        return 0;
    };
    gate.finish = [](void *context, const kl::KernelLaunchResult &result) noexcept {
        auto &s = *static_cast<Submission *>(context);
        if (result.status == 0) {
            s.runner->kernel_previous_caller_ = reinterpret_cast<uintptr_t>(s.caller);
        } else if (result.enqueue_started) {
            s.runner->kernel_exec_state_.poison(result.status);
        }
    };
    gate.query_tail = [](void *, void *event, bool *complete) noexcept {
        aclrtEventRecordedStatus status{};
        const int result = aclrtQueryEventStatus(event, &status);
        *complete = result == 0 && status == ACL_EVENT_RECORDED_STATUS_COMPLETE;
        return result;
    };
    kl::KernelLaunchOps ops;
    ops.context = &submission;
    ops.wait_event = [](void *, void *stream, void *event) noexcept {
        return aclrtStreamWaitEvent(stream, event);
    };
    ops.record_event = [](void *, void *event, void *stream) noexcept {
        return aclrtRecordEvent(event, stream);
    };
    ops.memset_handshake = [](void *context, void *stream) noexcept {
        const auto &s = *static_cast<Submission *>(context);
        for (const auto &region : s.clear.regions) {
            const int result =
                aclrtMemsetAsync(reinterpret_cast<void *>(region.address), region.bytes, 0, region.bytes, stream);
            if (result != 0) return result;
        }
        return 0;
    };
    ops.launch_aicore = [](void *context, void *stream) noexcept {
        auto &r = *static_cast<Submission *>(context)->runner;
        void *device_args = r.kernel_core_envelope_;
        rtArgsEx_t native{};
        native.args = &device_args;
        native.argsSize = sizeof(device_args);
        rtTaskCfgInfo_t config{};
        config.schemMode = RT_SCHEM_MODE_BATCH;
        return rtKernelLaunchWithHandleV2(r.aicore_bin_handle_, 0, r.block_dim_, &native, nullptr, stream, &config);
    };
    ops.launch_aicpu = [](void *context, void *stream) noexcept {
        const auto &s = *static_cast<Submission *>(context);
        rtCpuKernelArgs_t native{};
        native.baseArgs.args = const_cast<uint8_t *>(s.packet);
        native.baseArgs.argsSize = static_cast<uint32_t>(s.bytes);
        // CANN requires non-null attribute storage even when numAttrs is zero.
        rtLaunchKernelAttr_t attribute{};
        rtKernelLaunchCfg_t config{&attribute, 0U};
        return rtsLaunchCpuKernel(
            s.runner->kernel_aicpu_handle_, s.runner->kernel_runtime_.get_aicpu_launch_count(), stream, &config, &native
        );
    };
    return kl::launch_bound_kernel({submission.packet, submission.bytes, nullptr, 0}, caller_stream, gate, ops).status;
}
