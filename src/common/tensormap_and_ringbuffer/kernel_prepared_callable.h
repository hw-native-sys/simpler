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

#include <algorithm>
#include <vector>

#include "kernel_execution_inputs.h"
#include "task_interface/kernel_callable_residency.h"
#include "task_interface/kernel_callable_validation.h"

namespace simpler::tmr {

// Metadata attached to the existing orchestration SO slot. Only the table of
// borrowed CoreCallable addresses is owned here; code/storage remain K10's.
// Prepare builds this once, under the same quiescent registration protocol as
// the SO. Launch never allocates or replaces it.
struct PreparedKernelCallable {
    uint64_t residency_address{0};
    KernelCallableDeviceResidency residency{};
    PreparedInvocationView identity{};
    std::vector<uint64_t> functions;

    KernelCallableView view() const noexcept { return {identity, {functions.data(), functions.size()}}; }
};

inline bool make_prepared_kernel_callable(
    uint64_t residency_address, const KernelCallableDeviceResidency &resident, PreparedKernelCallable *out
) {
    if (out == nullptr || residency_address == 0 || resident.callable_id < 0 ||
        resident.callable_id >= MAX_REGISTERED_CALLABLE_IDS || resident.generation == 0 || resident.reserved != 0 ||
        resident.bytes > SIZE_MAX || resident.bytes > UINTPTR_MAX - resident.device_address)
        return false;
    const auto *image = reinterpret_cast<const ChipCallable *>(resident.device_address);
    if (!simpler::kernel::valid_kernel_callable_image(image, static_cast<size_t>(resident.bytes))) return false;
    PreparedKernelCallable candidate;
    candidate.residency_address = residency_address;
    candidate.residency = resident;
    candidate.identity.callable_id = resident.callable_id;
    candidate.identity.slot_generation = resident.generation;
    if (simpler::kernel::derive_invocation_counts(
            image->signature_, image->sig_count(), image->scalar_count(), &candidate.identity.tensor_count,
            &candidate.identity.scalar_count
        ) != InvocationStatus::Ok)
        return false;
    size_t count = 0;
    for (int32_t i = 0; i < image->child_count(); ++i) {
        const int32_t id = image->child_func_id(i);
        if (id < 0 || id >= RUNTIME_MAX_FUNC_ID || image->child(i).resolved_addr() == 0) return false;
        count = std::max(count, static_cast<size_t>(id) + 1);
    }
    candidate.functions.resize(count);
    for (int32_t i = 0; i < image->child_count(); ++i) {
        auto &address = candidate.functions[image->child_func_id(i)];
        if (address != 0) return false;
        address = reinterpret_cast<uint64_t>(&image->child(i));
    }
    *out = std::move(candidate);
    return true;
}

}  // namespace simpler::tmr
