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
 * DistOrchestrator — DAG builder.
 *
 * Public API (called by the user's orch fn during Worker::run):
 *   - submit_next_level(callable, TaskArgs, ChipCallConfig)
 *   - submit_next_level_group(callable, vector<TaskArgs>, ChipCallConfig)
 *   - submit_sub(callable_id, TaskArgs)
 *   - submit_sub_group(callable_id, vector<TaskArgs>)
 *
 * Each TaskArgs carries per-tensor TensorArgType tags. The Orchestrator
 * walks those tags to drive dependency inference (INPUT/INOUT → tensormap
 * lookup; OUTPUT/INOUT/OUTPUT_EXISTING → tensormap insert; NO_DEP → skip).
 *
 * Internal:
 *   - scope_begin / scope_end / drain — invoked only by Worker::run
 */

#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include "../task_interface/chip_call_config.h"
#include "../task_interface/data_type.h"
#include "../task_interface/task_args.h"
#include "../task_interface/tensor_arg.h"
#include "dist_ring.h"
#include "dist_scope.h"
#include "dist_tensormap.h"
#include "dist_types.h"

// ---------------------------------------------------------------------------
// SubmitResult — just the slot id
// ---------------------------------------------------------------------------
//
// Downstream consumers reference outputs by their own tensor pointers (the
// tensors live in shm/heap allocated by the user), and tensormap.lookup
// finds the producer slot from the data pointer. No outputs[] field needed.

struct DistSubmitResult {
    DistTaskSlot task_slot{DIST_INVALID_SLOT};
};

// ---------------------------------------------------------------------------
// DistOrchestrator
// ---------------------------------------------------------------------------

class DistOrchestrator {
public:
    void init(
        DistTensorMap *tensormap, DistRing *ring, DistScope *scope, DistReadyQueue *ready_queue,
        DistTaskSlotState *slots, int32_t num_slots
    );

    // Allocate an intermediate buffer (mmap MAP_SHARED|MAP_ANONYMOUS so child
    // workers see it after fork). Returns a ContinuousTensor with .data
    // pointing at the buffer.
    //
    // Lifetime: aligned with the slot lifecycle. alloc creates a synthetic
    // task slot in COMPLETED state that owns the buffer. Downstream tasks
    // that tag the buffer as INPUT/INOUT/OUTPUT_EXISTING wire a fanout edge
    // on this slot via TensorMap; the buffer is munmap'd in on_consumed
    // once all consumers have released their fanout refs and scope_end has
    // released the scope ref.
    ContinuousTensor alloc(const std::vector<uint32_t> &shape, DataType dtype);

    // Submit a NEXT_LEVEL task. `callable` is the chip callable buffer pointer
    // (uint64_t handle from Python — typically ChipCallable.buffer_ptr()).
    // Tags inside `args` drive dependency inference.
    DistSubmitResult submit_next_level(uint64_t callable, const TaskArgs &args, const ChipCallConfig &config);

    // Submit a group of NEXT_LEVEL tasks: N args -> N workers, 1 DAG node.
    DistSubmitResult
    submit_next_level_group(uint64_t callable, const std::vector<TaskArgs> &args_list, const ChipCallConfig &config);

    // Submit a SUB task by registered callable id.
    DistSubmitResult submit_sub(int32_t callable_id, const TaskArgs &args);

    // Submit a group of SUB tasks: N args -> N workers, 1 DAG node.
    DistSubmitResult submit_sub_group(int32_t callable_id, const std::vector<TaskArgs> &args_list);

    // Internal — invoked by Worker::run only.
    void scope_begin();
    void scope_end();

    // Block until every submitted task has reached CONSUMED. Invoked by
    // Worker::run after scope_end; not part of the user-facing orch-fn API.
    void drain();

    // Called by Scheduler (via DistWorker) when a task becomes CONSUMED:
    // erases TensorMap entries, frees alloc'd buffers, releases the ring slot.
    // Returns true iff this call performed the COMPLETED -> CONSUMED transition.
    // Idempotent: concurrent callers (release_ref vs try_consume) race on a
    // CAS — only the winner returns true and runs cleanup; losers return false.
    bool on_consumed(DistTaskSlot slot);

private:
    DistTensorMap *tensormap_ = nullptr;
    DistRing *ring_ = nullptr;
    DistScope *scope_ = nullptr;
    DistReadyQueue *ready_queue_ = nullptr;
    DistTaskSlotState *slots_ = nullptr;
    int32_t num_slots_ = 0;

    // --- Drain support (owned here, not on Worker) ---
    std::atomic<int32_t> active_tasks_{0};
    std::mutex drain_mu_;
    std::condition_variable drain_cv_;

    DistTaskSlotState &slot_state(DistTaskSlot s) { return slots_[s]; }

    // Shared submit machinery — installs slot, walks tags for deps, dispatches
    // ready transitions. `callable_ptr` and `callable_id` are mutually
    // exclusive depending on `worker_type`.
    DistSubmitResult submit_impl(
        WorkerType worker_type, uint64_t callable_ptr, int32_t callable_id, const ChipCallConfig &config,
        const std::vector<TaskArgs> &args_list
    );

    // Walk the tags of each TaskArgs in `args_list`, accumulating producer
    // slots (for INPUT/INOUT tags) and registering outputs in the tensormap
    // (for OUTPUT/INOUT/OUTPUT_EXISTING tags). NO_DEP tags are skipped.
    void infer_deps(
        DistTaskSlot slot, const std::vector<TaskArgs> &args_list, std::vector<DistTaskSlot> &producers,
        std::vector<uint64_t> &output_keys
    );

    // Release one fanout reference on 'slot'.
    // If all references are released → transition to CONSUMED.
    void release_ref(DistTaskSlot slot);
};
