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
 * @file dep_gen_host_graph.h
 * @brief dep_gen capture for host_build_graph — the dependency graph as the
 *        host orchestrator actually builds it.
 *
 * host_build_graph runs the whole orchestration on the host before any
 * scheduler thread starts, so every submit_task and every tensormap hit is
 * observable in-process, in order, with nothing to lose: the graph is recorded
 * from the real dependency path (`compute_task_fanin`'s emit/annotate hooks),
 * not reconstructed from a captured input stream. That is the difference from
 * tensormap_and_ringbuffer, where the orchestrator runs on the AICPU and the
 * host can only replay a ring of captured submits.
 *
 * Capture surface:
 *   begin_capture()         — once per orchestration, from simpler::hbg::build_graph
 *   begin_task()            — one per submit, before its dependency steps
 *   add_explicit_edge()     — STEP 1, per declared dependency
 *   add_creator_edge()      — STEP 3 Step A, per creator-retention producer
 *   add_tensormap_edge()    — STEP 3 Step B, per tensormap producer
 *   end_task()              — closes the task, after its last dependency step
 *
 * Control surface, called from the device runner (same host_runtime.so):
 *   set_enabled() / active() / emit()
 *
 * Every runtime build links this .cpp. A unit test that takes the orchestrator
 * without it resolves to the no-ops in tests/ut/cpp/stubs/hbg_orch_stubs.cpp.
 *
 * The graph lives in thread-local state, so capture is lock-free and two
 * prepared contexts on different threads cannot overwrite one another. Emit reads
 * the calling thread's state, and the caller keeps that read on the capturing
 * thread by emitting at the end of the orchestration's own bind — see
 * `emit_host_dep_gen_graph` in each platform's c_api_shared.cpp. Emitting later
 * would not be safe: the run lane serializes with a mutex, which guarantees
 * mutual exclusion but not thread affinity, so a drain can land on another
 * thread, and a successor's bind resets the state. Emit returns -3 if a caller
 * ever reads a thread that did not capture.
 *
 * Per-task producer dedup mirrors append_fanin_or_fail, which keys on the producer's
 * local id; this keys on producer task id. The two agree only because
 * host_build_graph is whole-graph-resident and never reuses a task slot at build time
 * (see append_fanin_or_fail in orchestrator.cpp). A runtime that recycles slots
 * mid-build would need this key revisited.
 *
 * Output is `deps.json` in the schema documented in docs/dfx/dep-gen.md — the
 * same schema the tensormap_and_ringbuffer replay emits, so every downstream
 * consumer (deps viewer, swimlane join) reads both runtimes' output the same way.
 */

#pragma once

#include <cstdint>

#include "host_build_graph/tensormap.h"
#include "host_build_graph/types.h"  // TensorRef
#include "tensor.h"

// ---------------------------------------------------------------------------
// Capture surface (orchestrator side)
// ---------------------------------------------------------------------------

/** True while a run is capturing; every other capture call is gated on this. */
bool dep_gen_host_graph_enabled();

/**
 * Start a fresh graph for the orchestration about to run. The graph a run emits
 * is the one that run built, so the reset belongs to the orchestration entry —
 * not to set_enabled(), which the runner may call again after the graph is
 * already built.
 */
void dep_gen_host_graph_begin_capture();

/**
 * Open a task's graph entry: its identity, launch shape, and arg slots. Any
 * tensor arg not yet materialized (OUTPUT) is recorded without tensor info,
 * matching what the runtime knows at submit time.
 */
void dep_gen_host_graph_begin_task(
    uint64_t task_id_raw, bool in_manual_scope, bool early_dispatch, const int32_t kernel_ids[3], int32_t block_num,
    int32_t tensor_count, const TensorRef *tensors, const TensorArgType *arg_types
);

/**
 * Close the task opened by begin_task(). Edges are attributed to the open task,
 * so closing it keeps a stray edge — one raised outside any submit — out of the
 * graph instead of silently attaching it to the previous task.
 */
void dep_gen_host_graph_end_task();

/** STEP 1: a dependency the caller declared via Arg::set_dependencies. */
void dep_gen_host_graph_add_explicit_edge(uint64_t producer_raw);

/** STEP 3 Step A: the producer that created the tensor this task consumes. */
void dep_gen_host_graph_add_creator_edge(uint64_t producer_raw, int32_t arg_idx, const simpler::hbg::Tensor &consumer);

/** STEP 3 Step B: a tensormap producer whose written slice this task reads. */
void dep_gen_host_graph_add_tensormap_edge(
    uint64_t producer_raw, int32_t arg_idx, const simpler::hbg::Tensor &consumer, const ChipTensorMapEntry &entry,
    OverlapStatus overlap
);

// ---------------------------------------------------------------------------
// Control surface (device-runner side)
// ---------------------------------------------------------------------------

extern "C" {

/** Enable/disable capture. The graph itself is cleared by begin_capture(). */
void dep_gen_host_graph_set_enabled(bool enable);

/**
 * True when this runtime captures the graph on the host, i.e. the runner must
 * not stand up the device-side dep_gen collector. The runner links a weak
 * `false` for runtimes that capture on the device instead.
 */
bool dep_gen_host_graph_active();

/**
 * Write the captured graph to `deps_json_path`. Returns 0 on success, non-zero
 * if capture was off/empty or the file could not be written.
 */
int dep_gen_host_graph_emit(const char *deps_json_path);
}
