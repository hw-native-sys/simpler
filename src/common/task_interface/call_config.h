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
 */

#pragma once

struct CallConfig {
    int block_dim = 24;
    int aicpu_thread_num = 3;
    bool enable_l2_swimlane = false;
    bool enable_dump_tensor = false;
    int enable_pmu = 0;  // 0 = disabled; >0 = enabled, value selects event type
};
