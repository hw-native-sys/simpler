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

// Upper bound on AICPU gate threads / slots, and the size of the
// `aicpu_allowed_cpus[]` table the host writes into Runtime. Single source of
// truth: the filter gate barriers/classifies at most this many threads, and
// `allowed_count` (the number of survivor cpu_ids) is bounded by it — so the
// Runtime ABI array must be exactly this long. Keep the runtime structs and
// the gate in lockstep by referencing this constant instead of a literal.
constexpr int32_t MAX_GATE_THREADS = 16;

// Returns true if this thread should call aicpu_execute().
// Returns false if this thread should exit (dropped).
// logical_count: desired active threads (from runtime.aicpu_thread_num)
// total_launched: actual threads launched (PLATFORM_MAX_AICPU_THREADS_JUST_FOR_LAUNCH)
//
// Used by sim platforms. a2a3/a5 onboard use the _filter variant below
// instead.
bool platform_aicpu_affinity_gate(int32_t logical_count, int32_t total_launched);

// Filter-style affinity gate. Every launched thread reads sched_getcpu(),
// barriers, and the gate keeps exactly those whose cpu_id appears in
// `allowed_cpus[0..allowed_count-1]`. The deterministic survivor index
// (its position in `allowed_cpus`) is exposed through
// platform_aicpu_affinity_thread_idx() and drives role assignment
// downstream (sched / orch).
//
// Convention used by tensormap_and_ringbuffer: indices 0..allowed_count-2
// are scheduler slots, index allowed_count-1 is the orchestrator slot. The
// host builds `allowed_cpus` from platform topology/OCCUPY and passes the
// array down through the Runtime struct.
//
// total_launched is the number of AICPU threads CANN actually launched
// (PLATFORM_MAX_AICPU_THREADS_JUST_FOR_LAUNCH on the target arch); the
// gate stops exactly that many threads on its barrier and lets only the
// allowed_count survivors continue.
bool platform_aicpu_affinity_gate_filter(const int32_t *allowed_cpus, int32_t allowed_count, int32_t total_launched);

// Deterministic position (0..allowed_count-1) of the calling thread in the
// `allowed_cpus[]` array that was passed to the most recent
// platform_aicpu_affinity_gate_filter() invocation on this thread.
// Returns -1 if this thread was dropped or did not go through the filter
// gate.
//
// Stable across the lifetime of the surviving thread — safe to call from
// aicpu_executor as the per-launch role identifier (sched 0..N-2 / orch
// N-1).
int32_t platform_aicpu_affinity_thread_idx();

// Publish this thread's resolved exec index into the affinity TLS.
//
// Onboard, the filter gate already assigns the index, so this is a redundant
// (same-value) store. Sim's basic gate does NOT assign an exec index, so
// platform_aicpu_affinity_thread_idx() would return -1 inside the AICPU `.so`;
// the executor resolves the real index (via its own fallback counter) and
// publishes it here so every per-thread reader in the `.so` agrees — in
// particular the per-thread AICPU phase-record slot
// (aicpu_phase_self_records()), which otherwise has no valid index on sim and
// drops every sub-phase stamp. Must be called on the executing thread, before
// any phase stamping.
void platform_aicpu_affinity_set_thread_idx(int32_t idx);
