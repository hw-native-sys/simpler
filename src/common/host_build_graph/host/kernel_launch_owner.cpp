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
#if defined(SIMPLER_HBG_KERNEL_MODE)

#include "../../platform/onboard/host/device_runner_base.h"

#include <acl/acl.h>
#include <runtime/rt.h>

#include <new>

#include "aicpu_loader/host/kernel_graph_launch.h"
#include "common/host_api.h"
#include "host/kernel_launch_binder.h"
#include "host_build_graph/kernel_argument_snapshot.h"
#include "host_build_graph/kernel_external_tensor.h"
#include "host_build_graph/kernel_graph_owner.h"
#include "host_build_graph/kernel_launch_state.h"

int DeviceRunnerBase::launch_hbg_kernel_callable(
    int32_t callable_id, const ChipStorageTaskArgs &args, void *caller_stream, const HostApi *api
) {
    auto it = callables_.find(callable_id);
    if (it == callables_.end() || api == nullptr) return PTO_RUNTIME_ERR_CALLABLE_NOT_RESIDENT;
    auto &state = it->second;
    if (hbg_kernel_state_ == nullptr || !hbg_kernel_state_->resources_prepared || !hbg_kernel_state_->slot_registered ||
        state.host_orch_func_ptr == nullptr)
        return PTO_RUNTIME_ERR_INVALID_STATE;

    simpler::kernel::PreparedInvocationView callable{callable_id, 0, 0};
    if (simpler::kernel::derive_invocation_counts(
            state.signature.data(), static_cast<int32_t>(state.signature.size()), &callable.tensor_count,
            &callable.scalar_count
        ) != simpler::kernel::InvocationStatus::Ok)
        return PTO_RUNTIME_ERR_INTERNAL;
    int32_t host_copy_tensor_count = 0;
    for (int32_t i = args.tensor_count() - 1; i >= 0 && args.tensor(i).address_space == AddressSpace::HOST; --i)
        ++host_copy_tensor_count;
    if (args.scalar_count() != callable.scalar_count ||
        args.tensor_count() != callable.tensor_count + host_copy_tensor_count)
        return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
    if (hbg::validate_kernel_external_tensors(args, host_copy_tensor_count) != hbg::KernelExternalTensorStatus::Ok)
        return PTO_RUNTIME_ERR_INVALID_ARGUMENT;

    const uint64_t argument_hash = hbg::kernel_argument_snapshot_hash(args);
    if (argument_hash == 0) return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
    const auto &cached = state.hbg_launch_state;
    // Host-copy tensors are tiling-data inputs to Host build. Their contents may
    // change while the pointer and shape stay stable, so eager launches rebuild
    // them. ACLGraph replay still reuses the HostArgs snapshot captured by its
    // node and therefore never enters this host launch path again.
    const bool cache_hit = host_copy_tensor_count == 0 && cached != nullptr && cached->argument_snapshot != nullptr &&
                           cached->graph_template.size() != 0 && cached->argument_hash == argument_hash &&
                           hbg::same_kernel_argument_snapshot(*cached->argument_snapshot, args);
    if (!cache_hit) {
        hbg::GraphInvocationIdentity identity{
            callable_id,   args.tensor_count(),     args.scalar_count(),    state.chip_buffer_hash,
            argument_hash, state.aicore_image_hash, host_copy_tensor_count,
        };
        hbg::GraphLaunchTemplate candidate;
        int rc = hbg::build_kernel_graph_template(
            kernel_runtime_, *api, args, state.host_orch_func_ptr, kernel_exec_state_, device_id_,
            kernel_static_config_.generation(), kernel_runtime_binary_id_,
            hbg_kernel_state_->resource_plan.capacity().layout.task_capacity, identity, candidate
        );
        if (rc != 0) return rc;
        std::unique_ptr<ChipStorageTaskArgs> snapshot(new (std::nothrow) ChipStorageTaskArgs(args));
        if (snapshot == nullptr) return PTO_RUNTIME_ERR_INTERNAL;
        auto next = std::make_shared<hbg::KernelCallableLaunchState>();
        next->graph_template = std::move(candidate);
        next->argument_snapshot = std::move(snapshot);
        next->argument_hash = argument_hash;
        state.hbg_launch_state = std::move(next);
    }

    namespace kl = simpler::kernel_launch;
    struct Submission {
        DeviceRunnerBase *runner;
        void *caller;
        const hbg::GraphLaunchTemplate *graph;
        uint64_t clear_address;
        size_t clear_bytes;
        uint64_t cancel_address;
    };
    const uint64_t runtime_address = reinterpret_cast<uint64_t>(persistent_args_.args().runtime_args);
    const auto *host_runtime_base = reinterpret_cast<const uint8_t *>(&kernel_runtime_);
    const uint64_t prelaunch_offset =
        reinterpret_cast<const uint8_t *>(&kernel_runtime_.kernel_prelaunch) - host_runtime_base;
    const uint64_t workers_offset =
        reinterpret_cast<const uint8_t *>(kernel_runtime_.get_workers()) - host_runtime_base;
    const uint64_t clear_address = runtime_address + prelaunch_offset;
    const uint64_t workers_end = runtime_address + workers_offset +
                                 static_cast<uint64_t>(kernel_runtime_.get_worker_count()) * sizeof(Handshake);
    if (runtime_address == 0 || workers_end <= clear_address || kernel_runtime_.get_worker_count() <= 0)
        return PTO_RUNTIME_ERR_INVALID_STATE;
    Submission submission{
        this,
        caller_stream,
        &state.hbg_launch_state->graph_template,
        clear_address,
        static_cast<size_t>(workers_end - clear_address),
        clear_address
    };
    kl::KernelLaunchGateOps gate;
    gate.context = &submission;
    gate.acquire = [](void *context, const kl::KernelInvocationBinding &, void *,
                      kl::KernelLaunchAdmission *out) noexcept -> int {
        auto &s = *static_cast<Submission *>(context);
        auto &r = *s.runner;
        if (!r.kernel_exec_state_.accepts_dispatch() || r.hbg_kernel_state_ == nullptr ||
            !r.hbg_kernel_state_->slot_registered)
            return PTO_RUNTIME_ERR_INVALID_STATE;
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
        return aclrtMemsetAsync(reinterpret_cast<void *>(s.clear_address), s.clear_bytes, 0, s.clear_bytes, stream);
    };
    ops.cancel_waiting_aicore = [](void *context, void *stream) noexcept {
        const auto &s = *static_cast<Submission *>(context);
        return aclrtMemsetAsync(
            reinterpret_cast<void *>(s.cancel_address), sizeof(HbgKernelPrelaunchControl), 0xff,
            sizeof(HbgKernelPrelaunchControl), stream
        );
    };
    ops.launch_aicore = [](void *context, void *stream) noexcept {
        auto &r = *static_cast<Submission *>(context)->runner;
        return r.launch_aicore_kernel(static_cast<rtStream_t>(stream), r.persistent_args_.device_k_args());
    };
    ops.launch_aicpu = [](void *context, void *stream) noexcept {
        const auto &s = *static_cast<Submission *>(context);
        return hbg::launch_graph_template(
            *s.graph, s.runner->kernel_aicpu_handle_,
            static_cast<uint32_t>(s.runner->kernel_runtime_.get_aicpu_launch_count()), static_cast<aclrtStream>(stream),
            nullptr, s.runner->kernel_exec_state_.event(KernelEventKind::AicoreStart)
        );
    };
    return kl::launch_bound_kernel(
               {static_cast<const uint8_t *>(state.hbg_launch_state->graph_template.data()),
                state.hbg_launch_state->graph_template.size(), nullptr, 0},
               caller_stream, gate, ops
    )
        .status;
}

#endif  // SIMPLER_HBG_KERNEL_MODE
