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

#include "common/scope_stats.h"

// Scope-stats collector — platform-owned, runtime-agnostic.
//
// Platform owns all collector state. Runtime calls pure-value APIs to report
// the task/heap ring start/end and tensormap usage at scope boundaries; no
// runtime types cross the boundary.
//
// Streaming transport (mirrors dep_gen): one ScopeStatsRecord is appended per
// scope boundary (begin and end) into a pooled ScopeStatsBuffer popped from the
// host-fed free_queue; full buffers are pushed onto the orchestrator thread's
// ready_queue. The host
// collector drains them in real time, so the effective record capacity is
// disk-bounded rather than the old fixed 16 384-entry ring.
//
// Setter symbols (set_scope_stats_enabled, set_platform_scope_stats_base)
// are exported unconditionally so the host-side sim DeviceRunner's dlsym
// always resolves.

extern "C" {

// --- Scope lifecycle probes (called by orchestrator begin_scope/end_scope) ---
//
// Each emits one record for the scope's current ring, carrying the task ring's
// and heap ring's start/end plus the tensormap pool usage, sampled at that
// boundary: begin tags it SCOPE_STATS_PHASE_BEGIN, end tags it
// SCOPE_STATS_PHASE_END. task_start/end = task ring tail/head, heap_start/end =
// heap reclaim/allocation pointers.

void scope_stats_begin(
    int ring_id, int32_t task_start, int32_t task_end, uint64_t heap_start, uint64_t heap_end, int32_t tensormap_used
);
void scope_stats_end(
    int ring_id, int32_t task_start, int32_t task_end, uint64_t heap_start, uint64_t heap_end, int32_t tensormap_used
);

// Per-task probe (scope_stats_task). Emits one PHASE_TASK record carrying the
// submitted task_id plus the current ring/heap occupancy, attributed to the
// enclosing scope. No-op unless per-task sampling is enabled and the enclosing
// scope passes the site filter. Called from submit_task; the orchestrator gates
// on is_scope_stats_task_enabled() first so a disabled run pays one bool load.
void scope_stats_record_task(
    uint64_t task_id, int ring_id, int32_t task_start, int32_t task_end, uint64_t heap_start, uint64_t heap_end,
    int32_t tensormap_used
);

// Per-task sampling enabled-state predicate (host-written run-constant cached at
// set_platform_scope_stats_base). Queried by the orchestrator as its submit-time
// gate; the orchestrator declares a weak `false` fallback for host builds.
bool is_scope_stats_task_enabled();

void scope_stats_on_fatal();

// --- Site tracking ---

void scope_stats_set_pending_site(const char *file, int line);

// --- Setter symbols (always exported) ---

void set_scope_stats_enabled(bool enable);

// Enabled-state predicate, queried by the orchestrator as its begin/end gate
// (same idiom as is_dep_gen_enabled). The collector owns the strong
// definition; the orchestrator declares a weak `false` fallback for host
// builds.
bool is_scope_stats_enabled();

// Map the shared region and reset collector-local state. Doubles as the init
// entry point — no separate init symbol; the current buffer is popped lazily
// on the first scope_end append. Host owns the shared header (free_queue,
// num_instances, capacities), so this no longer zeroes shared memory.
void set_platform_scope_stats_base(uint64_t scope_stats_data_base);

// --- Streaming transport plumbing (orchestrator thread) ---

// Record which AICPU thread runs the orchestrator (selects the per-thread
// ready_queue when buffers fill / on flush). Mirrors
// dep_gen_aicpu_set_orch_thread_idx.
void scope_stats_aicpu_set_orch_thread_idx(int thread_idx);

// Push the current (partially-filled) buffer onto the ready_queue so the host
// gets the final records. Idempotent: a no-op when there is no current buffer.
// Called from the orchestrator-thread exit path and from scope_stats_on_fatal.
void scope_stats_aicpu_flush_buffers();

// --- Capacity registration (called by runtime at init) ---

void scope_stats_set_ring_capacity(int ring_id, int32_t window_cap, uint64_t heap_cap);
void scope_stats_set_tensormap_capacity(int32_t cap);

}  // extern "C"
