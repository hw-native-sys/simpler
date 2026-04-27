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

/**
 * CallConfig — per-NEXT_LEVEL-task config. Carries execution knobs
 * (block_dim, aicpu_thread_num) plus the three parallel diagnostics
 * sub-features under the profiling umbrella: `enable_l2_swimlane` (swimlane),
 * `enable_dump_tensor`, and `enable_pmu`.
 *
 * Lives here (rather than chip_worker.h) so distributed task slot state
 * can store it directly without pulling in the full ChipWorker header
 * (which depends on types.h).
 *
 * Wire-compatible POD — packed and laid out so that one memcpy moves the
 * whole struct between the parent and the forked child via the shared-memory
 * mailbox. `bool` fields are stored as int32 to keep the layout deterministic
 * across compilers (sizeof(bool) is implementation-defined).
 */

#pragma once

#include <cstdint>

#pragma pack(push, 1)
struct CallConfig {
    int32_t block_dim = 24;
    int32_t aicpu_thread_num = 3;
    int32_t enable_l2_swimlane = 0;
    int32_t enable_dump_tensor = 0;
    int32_t enable_pmu = 0;  // 0 = disabled; >0 = enabled, value selects event type
};
#pragma pack(pop)
static_assert(sizeof(CallConfig) == 5 * sizeof(int32_t), "CallConfig wire layout drift");
