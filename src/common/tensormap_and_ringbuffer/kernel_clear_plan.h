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

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "task_interface/tmr_kernel_control.h"

namespace simpler::tmr {

struct TmrClearRegion {
    uint64_t address{0};
    uint64_t bytes{0};
};

// Borrowed, trusted resource-provider view, never derived from the launch packet.
// Sizes describe the exact control/report regions, not their enclosing allocation.
struct TmrKernelClearBinding {
    uint64_t context_generation{0};
    TmrClearRegion control{};
    TmrClearRegion reports{};
    int32_t worker_count{0};
};

struct TmrKernelClearPlan {
    uint64_t context_generation{0};
    std::array<TmrClearRegion, 2> regions{};
    TmrClearRegion cancel{};
};

inline bool valid_tmr_clear_binding(const TmrKernelClearBinding &binding) noexcept {
    if (binding.context_generation == 0 || binding.worker_count <= 0) return false;
    // A positive int32_t worker count times a 128-byte report fits uint64_t.
    const uint64_t report_bytes = static_cast<uint64_t>(binding.worker_count) * sizeof(TmrCoreReport);
    if (binding.control.bytes != sizeof(TmrLaunchControl) || binding.reports.bytes != report_bytes) return false;
    const auto valid_region = [](const TmrClearRegion &region) {
        return region.address != 0 && region.address % alignof(TmrLaunchControl) == 0 &&
               region.bytes <= std::numeric_limits<uint64_t>::max() - region.address;
    };
    if (!valid_region(binding.control) || !valid_region(binding.reports)) return false;
    return binding.control.address + binding.control.bytes <= binding.reports.address ||
           binding.reports.address + binding.reports.bytes <= binding.control.address;
}

// The owner pins these regions and serializes invocations/replays before clear.
// This plan performs no allocation, device access, or concurrency admission.
inline bool build_tmr_kernel_clear_plan(const TmrKernelClearBinding &binding, TmrKernelClearPlan *out) noexcept {
    if (out == nullptr || !valid_tmr_clear_binding(binding)) return false;
    TmrKernelClearPlan candidate;
    candidate.context_generation = binding.context_generation;
    candidate.regions = {binding.control, binding.reports};
    candidate.cancel = {binding.control.address + offsetof(TmrLaunchControl, host_cancel), sizeof(uint32_t)};
    *out = candidate;
    return true;
}

inline bool
validate_tmr_kernel_clear_plan(const TmrKernelClearPlan &plan, const TmrKernelClearBinding &trusted_binding) noexcept {
    TmrKernelClearPlan expected;
    if (!build_tmr_kernel_clear_plan(trusted_binding, &expected) ||
        plan.context_generation != expected.context_generation)
        return false;
    for (size_t i = 0; i < expected.regions.size(); ++i) {
        if (plan.regions[i].address != expected.regions[i].address ||
            plan.regions[i].bytes != expected.regions[i].bytes)
            return false;
    }
    return plan.cancel.address == expected.cancel.address && plan.cancel.bytes == expected.cancel.bytes;
}

}  // namespace simpler::tmr
