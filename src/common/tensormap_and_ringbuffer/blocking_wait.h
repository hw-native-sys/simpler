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

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "submit_types.h"
#include "types.h"

struct RuntimeContext;

enum class BlockingWaitMode : uint32_t { ProducersComplete = 0, Resident = 1 };

// Producer ids belong to this orchestration run. The borrowed array lasts
// through submit; the runtime does not retain it or verify notification semantics.
struct BlockingWait {
    BlockingWaitMode mode{BlockingWaitMode::ProducersComplete};
    uint32_t producer_count{0};
    const TaskId *producers{nullptr};

    static BlockingWait after_producers(const TaskId *producers, uint32_t count) {
        return {BlockingWaitMode::ProducersComplete, count, producers};
    }

    static BlockingWait resident() { return {BlockingWaitMode::Resident, 0, nullptr}; }
};

inline constexpr uint32_t BLOCKING_WAIT_OPS_VERSION = 1;

// Version 1 defines the BlockingWait layout and this synchronous submit entry.
// Only the optional binding passes this table; legacy RuntimeOps has no tail.
struct BlockingWaitOps {
    uint32_t version;
    uint32_t size;
    TaskOutputTensors (*submit_task)(
        RuntimeContext *rt, const MixedKernels &kernels, const CoreTaskArgs &args, const BlockingWait &wait
    );
};

static_assert(std::is_standard_layout_v<BlockingWait> && std::is_trivially_copyable_v<BlockingWait>);
static_assert(offsetof(BlockingWait, mode) == 0 && offsetof(BlockingWait, producer_count) == 4);
static_assert(offsetof(BlockingWait, producers) == 8 && sizeof(BlockingWait) == 16);
static_assert(offsetof(BlockingWaitOps, submit_task) == 8 && sizeof(BlockingWaitOps) == 16);
