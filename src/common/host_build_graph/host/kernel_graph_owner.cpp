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

#include "host_build_graph/kernel_graph_owner.h"

#include <algorithm>
#include <memory>
#include <vector>

#include <acl/acl.h>
#include <runtime/rt.h>

#include "common/host_api.h"
#include "call_config.h"
#include "host/kernel_execution_state.h"
#include "host/memory_allocator.h"
#include "host_build_graph/host_tensor_access.h"
#include "host_build_graph/kernel_external_tensor.h"
#include "host_build_graph/kernel_callable_registration.h"
#include "host_build_graph/kernel_graph_slot.h"
#include "host_build_graph/kernel_launch_state.h"
#include "host_build_graph/runtime_core.h"
#include "runtime.h"
#include "../../platform/onboard/host/device_runner_base.h"
#include "host/kernel_pipeline_contract.h"

namespace hbg {

int prepare_kernel_graph_resources(
    KernelExecutionState &context, MemoryAllocator &allocator, const CallConfig &config,
    RuntimeArchitecture architecture, KernelResourcePlan *plan_out
) {
    if (plan_out == nullptr) return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
    const uint64_t task_window =
        config.runtime_env.ring_task_window[0] == 0 ? CHIP_DEFAULT_GRAPH_TASKS : config.runtime_env.ring_task_window[0];
    RuntimeArenaLayout layout{};
    int rc = make_kernel_graph_layout(task_window, layout);
    if (rc != 0) return rc;
    const auto maximum = sm_layout::segment_offsets(sm_layout::mirror_extents(task_window));
    constexpr uint64_t kDefaultHeap = UINT64_C(256) * 1024 * 1024;
    const uint64_t heap = config.runtime_env.ring_heap[0] == 0 ? kDefaultHeap : config.runtime_env.ring_heap[0];
    GraphResourceRequirements capacity{};
    capacity.layout = {HBG_RUNTIME_LAYOUT_ABI_VERSION, architecture,         task_window, layout.arena_size,
                       layout.off_copied_begin,        layout.off_copied_end};
    capacity.gm_heap_bytes = heap;
    capacity.runtime_arena_bytes = layout.off_copied_end + maximum.end;
    // Definitions and the optional A5 scheduler state are bounded by the same
    // explicit kernel capacity knob. They remain distinct frozen destinations.
    capacity.graph_definition_bytes = heap;
    capacity.scheduler_state_bytes = architecture == RuntimeArchitecture::A5 ? heap : 0;
    KernelResourcePlan plan;
    rc = KernelResourcePlan::create(&capacity, 1, plan);
    if (rc != 0) return rc;
    rc = plan.prepare(context, KernelResourceOps::from_allocator(allocator));
    if (rc != 0) return rc;
    rc = context.freeze_resources();
    if (rc != 0) return rc;
    *plan_out = plan;
    return 0;
}

int build_kernel_graph_template(
    Runtime &runtime, const HostApi &api, const ChipStorageTaskArgs &args, void *host_orch_func_ptr,
    const KernelExecutionState &context, int device_id, uint64_t generation, uint64_t runtime_binary_id,
    uint64_t task_window, const GraphInvocationIdentity &identity, GraphLaunchTemplate &out
) try {
    if (host_orch_func_ptr == nullptr || task_window == 0) return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
    const auto &entry_points = *static_cast<const HostOrchEntryPoints *>(host_orch_func_ptr);
    HostTensorAccessor tensor_access(&api, HostTensorAccessMode::KernelHostCopiesOnly);
    const auto external =
        prepare_kernel_external_tensors(args, identity.host_copy_tensor_count, entry_points, tensor_access);
    if (external != KernelExternalTensorStatus::Ok) return PTO_RUNTIME_ERR_INVALID_ARGUMENT;

    runtime.set_orch_args(args);
    ChipTaskArgs orch_args;
    orch_args.create_from_entry_storage(runtime.get_orch_args());
    DeviceArena host_arena;
    uint64_t ready_capacity = 64;
    while (ready_capacity < task_window)
        ready_capacity <<= 1;
    RuntimeArenaLayout layout = runtime_reserve_layout(host_arena, task_window, ready_capacity);
    if (host_arena.commit(DeviceArena::kDefaultBaseAlign) == nullptr) return PTO_RUNTIME_ERR_INTERNAL;
    const uint64_t sm_bytes = SharedMemoryHandle::calculate_size(task_window);
    RuntimeContext *rt = runtime_init_data_from_layout(host_arena, layout, MODE_EXECUTE, nullptr, sm_bytes);
    if (rt == nullptr) return PTO_RUNTIME_ERR_INTERNAL;
    runtime_wire_arena_pointers(host_arena, layout, rt);
    rt->prebuilt_layout = layout;
    rt->active_callable_hash = identity.callable_hash;
    std::vector<std::byte> sm_storage(sm_bytes + CHIP_ALIGN_SIZE);
    void *sm = reinterpret_cast<void *>(
        (reinterpret_cast<uintptr_t>(sm_storage.data()) + CHIP_ALIGN_SIZE - 1) &
        ~(static_cast<uintptr_t>(CHIP_ALIGN_SIZE) - 1)
    );
    GraphBuild build;
    int rc = build_graph(&runtime, tensor_access, {rt, sm, sm_bytes, task_window, {}}, entry_points, orch_args, build);
    if (rc < 0) return rc;
    return make_graph_launch_template(build, *rt, context, device_id, generation, runtime_binary_id, identity, out);
} catch (const std::bad_alloc &) {
    return PTO_RUNTIME_ERR_INTERNAL;
}

}  // namespace hbg

int DeviceRunnerBase::prepare_hbg_kernel_runtime(const HostApi *) {
    const auto architecture = static_cast<hbg::RuntimeArchitecture>(runtime_hbg_kernel_architecture_impl());
    if (architecture != hbg::RuntimeArchitecture::A2A3 && architecture != hbg::RuntimeArchitecture::A5)
        return PTO_RUNTIME_ERR_INTERNAL;
    auto state = std::make_shared<hbg::KernelContextLaunchState>();
    int rc = hbg::prepare_kernel_graph_resources(
        kernel_exec_state_, mem_alloc_, kernel_static_config_.request(), architecture, &state->resource_plan
    );
    if (rc != 0) return rc;
    state->resources_prepared = true;

    const auto &capacity = state->resource_plan.capacity();
    const uint64_t required[] = {
        capacity.gm_heap_bytes, capacity.runtime_arena_bytes, capacity.graph_definition_bytes,
        capacity.scheduler_state_bytes, sizeof(hbg::GraphSlotRegistry)
    };
    KernelResourceBinding binding;
    rc = kernel_exec_state_.inspect_frozen_resources(
        device_id_, kernel_static_config_.generation(), hbg::KernelResourcePlan::resource_schema, required, 5, binding
    );
    if (rc != 0) return rc;
    RuntimeArenaLayout layout{};
    rc = hbg::make_kernel_graph_layout(capacity.layout.task_capacity, layout);
    if (rc != 0) return rc;
    kernel_runtime_.set_gm_sm_ptr(reinterpret_cast<void *>(binding.regions[1].address + layout.off_copied_end));
    kernel_runtime_.set_prebuilt_arena(reinterpret_cast<void *>(binding.regions[1].address), layout.off_runtime);
    kernel_runtime_.host_total_tasks = 0;
    kernel_runtime_.sm_image_bytes = binding.regions[1].capacity - layout.off_copied_end;
    hbg_kernel_state_ = std::move(state);
    return 0;
}

int DeviceRunnerBase::prepare_hbg_kernel_callable_registration(
    int32_t callable_id, size_t callable_bytes, const simpler::kernel::PreparedInvocationView &callable,
    void *control_stream_raw
) {
    auto *control_stream = static_cast<rtStream_t>(control_stream_raw);
    auto state_it = callables_.find(callable_id);
    if (control_stream == nullptr || state_it == callables_.end() || hbg_kernel_state_ == nullptr ||
        !hbg_kernel_state_->resources_prepared)
        return PTO_RUNTIME_ERR_INVALID_STATE;
    auto &context = *hbg_kernel_state_;
    int rc = 0;
    if (!context.slot_registered) {
        rc = hbg::seal_graph_execution_slot(
            kernel_exec_state_, device_id_, kernel_static_config_.generation(), kernel_runtime_binary_id_,
            context.slot_registration
        );
        if (rc != 0) return rc;
        rc = launch_aicpu_payload(
            control_stream, &context.slot_registration, sizeof(context.slot_registration),
            "simpler_aicpu_l1_hbg_register_execution_slot", 1
        );
        if (rc != 0) return rc;
        rc = aclrtSynchronizeStreamWithTimeout(control_stream, PLATFORM_STREAM_SYNC_TIMEOUT_MS);
        if (rc != 0) return rc;
        context.slot_registered = true;
    }

    const auto &callable_state = state_it->second;
    hbg::HbgCallableRegistration registration{};
    registration.context_generation = kernel_static_config_.generation();
    registration.runtime_address = reinterpret_cast<uint64_t>(persistent_args_.args().runtime_args);
    registration.register_table_address = persistent_args_.args().regs;
    registration.callable_address = kernel_callable_cache_.pending_uploaded_address();
    registration.callable_bytes = callable_bytes;
    registration.callable_hash = callable_state.chip_buffer_hash;
    registration.function_hash = callable_state.aicore_image_hash;
    registration.callable_id = callable_id;
    registration.tensor_count = callable.tensor_count;
    registration.scalar_count = callable.scalar_count;
    registration.device_id = device_id_;
    registration.runtime_binary_id = kernel_runtime_binary_id_;
    if (!hbg::valid_hbg_callable_registration(registration)) return PTO_RUNTIME_ERR_INTERNAL;
    rc = launch_aicpu_payload(
        control_stream, &registration, sizeof(registration), "simpler_aicpu_l1_hbg_register_callable", 1
    );
    if (rc != 0) return rc;
    rc = aclrtSynchronizeStreamWithTimeout(control_stream, PLATFORM_STREAM_SYNC_TIMEOUT_MS);
    if (rc != 0) return rc;
    rc = commit_device_register(callable_id);
    if (rc != 0) return rc;
    return kernel_exec_state_.mark_ready_enqueued();
}

int DeviceRunnerBase::finalize_hbg_kernel_registration() {
    if (hbg_kernel_state_ == nullptr || !hbg_kernel_state_->slot_registered) return 0;
    rtStream_t control_stream = static_cast<rtStream_t>(kernel_exec_state_.hidden_stream(KernelStreamKind::Aicpu));
    if (control_stream == nullptr) return PTO_RUNTIME_ERR_INVALID_STATE;
    hbg::GraphSlotDetach detach{
        hbg_kernel_state_->slot_registration.registry.address,
        kernel_static_config_.generation(),
        kernel_runtime_binary_id_,
        device_id_,
        0,
    };
    int rc =
        launch_aicpu_payload(control_stream, &detach, sizeof(detach), "simpler_aicpu_l1_hbg_detach_execution_slot", 1);
    if (rc != 0) return rc;
    rc = aclrtSynchronizeStreamWithTimeout(control_stream, PLATFORM_STREAM_SYNC_TIMEOUT_MS);
    if (rc != 0) return rc;
    hbg_kernel_state_->slot_registered = false;
    return 0;
}

#endif  // SIMPLER_HBG_KERNEL_MODE
