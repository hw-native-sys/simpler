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
 * (block_dim, aicpu_thread_num), per-task runtime-environment overrides
 * (`runtime_env.ring_task_window` / `.ring_heap` / `.ring_dep_pool`, each a per-ring array) plus the five parallel
 * diagnostics sub-features under the profiling umbrella: `enable_l2_swimlane` (swimlane), `enable_dump_tensor`,
 * `enable_pmu`, `enable_dep_gen`, and `enable_scope_stats`. All five require `output_prefix` because they each write
 * sibling artifacts into that directory
 * (`l2_swimlane_records.json` / `tensor_dump/` / `pmu.csv` / `deps.json` /
 * `scope_stats/scope_stats.jsonl`).
 *
 * `block_dim == 0` is a sentinel for "auto" — DeviceRunner resolves it at
 * run() time to the max block_dim the AICore stream allows
 * (aclrtGetStreamResLimit on onboard; PLATFORM_MAX_BLOCKDIM on sim).
 * Any positive value is taken as an explicit cap and validated against
 * the same stream-resource limits.
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
 * diagnostic artifacts (l2_swimlane_records.json / tensor_dump/ / pmu.csv /
 * deps.json / scope_stats/scope_stats.jsonl) are written. The caller is
 * responsible for filling it whenever any diagnostic flag is enabled — `validate()` enforces
 * this contract at every submit/run entry point so the runtime never has to
 * invent a path.
 */

#pragma once

#include <cstdint>
#include <cstring>
#include <stdexcept>

inline constexpr int RUNTIME_ENV_RING_COUNT = 4;
inline constexpr int RUNTIME_ENV_FIELD_GROUPS = 3;  // ring_task_window, ring_heap, ring_dep_pool
inline constexpr int RUNTIME_ENV_UINT64_FIELD_COUNT = RUNTIME_ENV_FIELD_GROUPS * RUNTIME_ENV_RING_COUNT;

#pragma pack(push, 1)
// Per-task runtime-environment overrides — the programmatic equivalent of the
// `PTO2_RING_*` env vars, grouped under their own sub-struct so they read as a
// distinct configuration tier from the top-level execution knobs (block_dim,
// aicpu_thread_num). Consumed by tensormap_and_ringbuffer only; other runtimes
// ignore them.
//
// Each resource is a per-scope-depth-ring array (index 0..3). A 0 entry is
// unset and falls through to the next precedence tier: per-ring entry >
// PTO2_RING_* env var > compile-time default. ring_heap is bytes per ring.
// The Python/scene-test layer accepts a scalar and broadcasts it to every ring
// before populating these arrays, so a one-value "size all rings the same"
// request arrives here as [v, v, v, v].
struct RuntimeEnv {
    uint64_t ring_task_window[RUNTIME_ENV_RING_COUNT] = {};
    uint64_t ring_heap[RUNTIME_ENV_RING_COUNT] = {};
    uint64_t ring_dep_pool[RUNTIME_ENV_RING_COUNT] = {};

    bool any() const noexcept {
        for (int i = 0; i < RUNTIME_ENV_RING_COUNT; ++i) {
            if (ring_task_window[i] != 0 || ring_heap[i] != 0 || ring_dep_pool[i] != 0) {
                return true;
            }
        }
        return false;
    }

    // Throws if a ring sizing override violates the ring buffer's constraints.
    void validate() const {
        auto pow2 = [](uint64_t v) {
            return (v & (v - 1)) == 0;
        };
        auto validate_task_window = [&](uint64_t value) {
            if (value != 0 && (value < 4 || value > INT32_MAX || !pow2(value))) {
                throw std::invalid_argument("RuntimeEnv: ring_task_window must be a power of 2 in [4, INT32_MAX]");
            }
        };
        auto validate_heap = [&](uint64_t value) {
            if (value != 0 && value < 1024) {
                throw std::invalid_argument("RuntimeEnv: ring_heap must be >= 1024 (bytes per ring)");
            }
        };
        auto validate_dep_pool = [&](uint64_t value) {
            if (value != 0 && (value < 4 || value > INT32_MAX)) {
                throw std::invalid_argument("RuntimeEnv: ring_dep_pool must be in [4, INT32_MAX]");
            }
        };
        for (int i = 0; i < RUNTIME_ENV_RING_COUNT; ++i) {
            validate_task_window(ring_task_window[i]);
            validate_heap(ring_heap[i]);
            validate_dep_pool(ring_dep_pool[i]);
        }
    }
};

struct CallConfig {
    int32_t block_dim = 0;  // 0 = auto; resolved by DeviceRunner at run() time
    int32_t aicpu_thread_num = 3;
    int32_t enable_l2_swimlane = 0;
    int32_t enable_dump_tensor = 0;
    int32_t enable_pmu = 0;  // 0 = disabled; >0 = enabled, value selects event type
    int32_t enable_dep_gen = 0;
    int32_t enable_scope_stats = 0;  // writes <output_prefix>/scope_stats/scope_stats.jsonl
    RuntimeEnv runtime_env;          // per-task PTO2_RING_* overrides
    char output_prefix[1024] = {};

    bool diagnostics_any() const noexcept {
        return enable_l2_swimlane != 0 || enable_dump_tensor != 0 || enable_pmu != 0 || enable_dep_gen != 0 ||
               enable_scope_stats != 0;
    }

    bool output_prefix_set() const noexcept { return output_prefix[0] != '\0'; }

    // Throws if any diagnostic flag is enabled but `output_prefix` is empty,
    // or if a ring sizing override violates the ring buffer's constraints.
    // Called at every submit/run entry point so the failure surfaces as close
    // to the user's call site as possible (no IPC round-trip).
    void validate() const {
        if (diagnostics_any() && !output_prefix_set()) {
            throw std::invalid_argument(
                "CallConfig: output_prefix must be set whenever any of "
                "enable_l2_swimlane / enable_dump_tensor / enable_pmu / enable_dep_gen / "
                "enable_scope_stats is enabled"
            );
        }
        runtime_env.validate();
    }
};
#pragma pack(pop)
static_assert(sizeof(RuntimeEnv) == RUNTIME_ENV_UINT64_FIELD_COUNT * sizeof(uint64_t), "RuntimeEnv wire layout drift");
static_assert(
    sizeof(CallConfig) == 7 * sizeof(int32_t) + RUNTIME_ENV_UINT64_FIELD_COUNT * sizeof(uint64_t) + 1024,
    "CallConfig wire layout drift"
);
