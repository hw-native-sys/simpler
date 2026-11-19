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
#include <cstdint>
#include <array>
#include <stdexcept>

namespace wse {
// Linx owns the tile payload. Each absolute Shared slot has one Block-local publication contract.
struct SharedSlotState {
    uint32_t producer{};
    uint32_t consumers{};
    const char *shape_name{};
    uint32_t consumed{};
    uint64_t generation{};
};
using SharedSlotTable = std::array<SharedSlotState, 64>;

struct BlockScope {
    uint64_t *active_epoch;
    uint64_t epoch;
    const uint64_t *active_run_epoch{nullptr};
    uint64_t run_epoch{0};
    SharedSlotTable *slots{nullptr};
    void validate() const {
        if (!active_epoch || !epoch || __atomic_load_n(active_epoch, __ATOMIC_ACQUIRE) != epoch ||
            (active_run_epoch && __atomic_load_n(active_run_epoch, __ATOMIC_ACQUIRE) != run_epoch))
            throw std::invalid_argument("WSE SharedTile belongs to an ended Kernel Block");
    }
};
}  // namespace wse
