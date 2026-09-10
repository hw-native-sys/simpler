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

#pragma once

#include <limits>

#include "callable_table_view.h"
#include "kernel_invocation_args.h"
#include "runtime.h"
#include "runtime_core.h"

namespace simpler::tmr {

struct ExecutionInputs {
    int32_t callable_id{-1};
    const EntryArgsStorage *args{nullptr};
    CallableTableView functions{};
    void *sm{nullptr};
    void *arena{nullptr};
    size_t runtime_offset{0};
};

inline ExecutionInputs program_execution_inputs(const Runtime &runtime) {
    return {
        runtime.get_active_callable_id(),
        &runtime.get_orch_args(),
        {runtime.dev.func_id_to_addr_, RUNTIME_MAX_FUNC_ID},
        runtime.get_gm_sm_ptr(),
        runtime.get_prebuilt_arena_base(),
        runtime.get_prebuilt_runtime_offset()
    };
}

// Trusted, process-local views; none of these C++ objects is a transport wire.
// required_bytes and the initialized arena layout come from the resource provider.
struct ExecutionRegionView {
    void *base{nullptr};
    size_t capacity{0};
    size_t required_bytes{0};
};

struct KernelBindingView {
    TmrExecutionBindingView identity{};
    Runtime *resident{nullptr};
    ExecutionRegionView sm{};
    ExecutionRegionView arena{};
    size_t runtime_offset{0};
};

struct KernelCallableView {
    PreparedInvocationView identity{};
    CallableTableView functions{};
};

inline bool valid_execution_region(const ExecutionRegionView &region, size_t alignment) noexcept {
    const uintptr_t base = reinterpret_cast<uintptr_t>(region.base);
    return base != 0 && base % alignment == 0 && region.required_bytes != 0 &&
           region.required_bytes <= region.capacity && region.capacity <= std::numeric_limits<uintptr_t>::max() - base;
}

inline InvocationStatus validate_execution_binding(const KernelBindingView &binding) noexcept {
    if (binding.identity.device_binding_addr == 0 || binding.identity.context_generation == 0 ||
        binding.resident == nullptr || reinterpret_cast<uintptr_t>(binding.resident) % alignof(Runtime) != 0 ||
        !valid_execution_region(binding.sm, alignof(SharedMemoryHeader)) ||
        binding.sm.required_bytes < sizeof(SharedMemoryHeader) ||
        !valid_execution_region(binding.arena, DeviceArena::kDefaultBaseAlign) ||
        binding.runtime_offset > binding.arena.required_bytes ||
        sizeof(RuntimeContext) > binding.arena.required_bytes - binding.runtime_offset ||
        (reinterpret_cast<uintptr_t>(binding.arena.base) + binding.runtime_offset) % alignof(RuntimeContext) != 0)
        return InvocationStatus::InvalidBinding;
    return InvocationStatus::Ok;
}

// One execution's storage, owned by the existing executor. Admission and clear
// require exclusive ownership; readers join before clear or a following admission.
// The provider has already published the arena image and pins all borrowed views.
class KernelInvocationState {
public:
    InvocationStatus
    admit(ByteSpan packet, const KernelCallableView &callable, const KernelBindingView &binding) noexcept {
        if (active_) return InvocationStatus::InvalidArgument;
        const auto status = validate_execution_binding(binding);
        if (status != InvocationStatus::Ok) return status;
        if (callable.functions.count > RUNTIME_MAX_FUNC_ID ||
            reinterpret_cast<uintptr_t>(callable.functions.entries) % alignof(uint64_t) != 0 ||
            (callable.functions.count != 0 && callable.functions.entries == nullptr))
            return InvocationStatus::InvalidBinding;
        const auto decoded = consume_tmr_invocation(packet, callable.identity, binding.identity, &storage_);
        if (decoded != InvocationStatus::Ok) return decoded;
        inputs_ = {callable.identity.callable_id, &storage_, callable.functions, binding.sm.base, binding.arena.base,
                   binding.runtime_offset};
        resident_ = binding.resident;
        active_ = true;
        return InvocationStatus::Ok;
    }

    bool active() const { return active_; }
    Runtime *resident() const { return resident_; }
    const ExecutionInputs &inputs() const { return inputs_; }
    // All ChipTaskArgs/TensorRef borrowers must be reset before this call.
    void clear() {
        inputs_ = {};
        resident_ = nullptr;
        storage_.clear();
        active_ = false;
    }

private:
    EntryArgsStorage storage_{};
    ExecutionInputs inputs_{};
    Runtime *resident_{nullptr};
    bool active_{false};
};

inline bool configure_orchestration_args(
    const ExecutionInputs &inputs, ChipTaskArgs &args, OrchestrationConfig (*config)(const ChipTaskArgs &)
) {
    args.create_from_entry_storage(*inputs.args);
    if (config == nullptr) return true;
    const auto cfg = config(args);
    return cfg.expected_arg_count <= 0 ||
           inputs.args->tensor_count() + inputs.args->scalar_count() >= cfg.expected_arg_count;
}

// Internal device consumption phases. Admission/release have one owner;
// init/run execute on the admitted affinity group. No launch symbol is registered.
// Caller supplies publication, cancellation, and an all-consumers-complete barrier.
InvocationStatus
admit_kernel_execution(ByteSpan packet, const KernelCallableView &, const KernelBindingView &) noexcept;
int32_t init_kernel_execution();
int32_t run_kernel_execution();
int32_t kernel_execution_status();
void release_kernel_execution();

}  // namespace simpler::tmr
