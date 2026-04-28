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
 *
 * `output_prefix` is a NUL-terminated directory path under which all
 * diagnostic artifacts (l2_perf_records.json / tensor_dump/ / pmu.csv) are
 * written. The caller is responsible for filling it whenever any diagnostic
 * flag is enabled — `validate()` enforces this contract at every submit/run
 * entry point so the runtime never has to invent a path.
 */

#pragma once

#include <cstdint>
#include <cstring>
#include <stdexcept>

#pragma pack(push, 1)
struct CallConfig {
    int32_t block_dim = 24;
    int32_t aicpu_thread_num = 3;
    int32_t enable_l2_swimlane = 0;
    int32_t enable_dump_tensor = 0;
    int32_t enable_pmu = 0;  // 0 = disabled; >0 = enabled, value selects event type
    char output_prefix[1024] = {};

    bool diagnostics_any() const noexcept {
        return enable_l2_swimlane != 0 || enable_dump_tensor != 0 || enable_pmu != 0;
    }

    bool output_prefix_set() const noexcept { return output_prefix[0] != '\0'; }

    // Throws if any diagnostic flag is enabled but `output_prefix` is empty.
    // Called at every submit/run entry point so the failure surfaces as close
    // to the user's call site as possible (no IPC round-trip).
    void validate() const {
        if (diagnostics_any() && !output_prefix_set()) {
            throw std::invalid_argument(
                "CallConfig: output_prefix must be set whenever any of "
                "enable_l2_swimlane / enable_dump_tensor / enable_pmu is enabled"
            );
        }
    }
};
#pragma pack(pop)
static_assert(sizeof(CallConfig) == 5 * sizeof(int32_t) + 1024, "CallConfig wire layout drift");
