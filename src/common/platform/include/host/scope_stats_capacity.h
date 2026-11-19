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
#include <cstdint>
#include <optional>

#include "common/scope_stats.h"

namespace simpler::dfx::scope_stats_runs {

/** Immutable capacities of one bound TMR arena; heap values are bytes. */
struct Capacities {
    std::array<int32_t, SCOPE_STATS_MAX_RING_DEPTH> task_window{};
    std::array<int32_t, SCOPE_STATS_MAX_RING_DEPTH> dep_pool{};
    std::array<uint64_t, SCOPE_STATS_MAX_RING_DEPTH> heap{};
    int32_t tensormap{0};
};

/** Host-only metadata. A runtime without TMR rings states `required=false`. */
struct CapacityMetadata {
    bool required{true};
    std::optional<Capacities> values;

    bool complete() const { return !required || values.has_value(); }
};

}  // namespace simpler::dfx::scope_stats_runs
