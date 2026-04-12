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
 * DistOrchestrator — 7-step submit() flow.
 *
 * The Orchestrator runs exclusively on the main (Orch) thread and owns:
 *   - DistTensorMap  (no locking needed)
 *   - DistScope      (no locking needed)
 *
 * It shares with the Scheduler (via pointers / atomics):
 *   - DistRing       (alloc orch-only; release Scheduler-only)
 *   - DistReadyQueue (push Orch; pop Scheduler)
 *   - DistTaskSlotState[] (fanin/fanout fields protected per-task)
 *
 * submit() 7-step flow (mirrors L2 pto2_submit_mixed_task):
 *   1. Alloc slot from ring (back-pressure blocks here)
 *   2. Allocate output buffers (malloc per output)
 *   3. TensorMap lookup for each input → collect producer slots
 *   4. TensorMap insert for each output
 *   5. Write task slot: state=PENDING, fanin_count, payload, outputs
 *   6. Finalize fanin: for each producer, lock fanout_mu, append consumer;
 *      if producer is already COMPLETED/CONSUMED skip (already released)
 *   7. If fanin_count == 0 (no live producers): state=READY, push ready_queue
 *      Also push if within scope (scope ref counted in fanout_total)
 */

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "dist_ring.h"
#include "dist_scope.h"
#include "dist_tensormap.h"
#include "dist_types.h"

// ---------------------------------------------------------------------------
// Submit API types
// ---------------------------------------------------------------------------

struct DistInputSpec {
    DistTensorKey key;
};

enum class DistOutputOwnership : int32_t {
    ALLOCATED = 0,
    EXTERNAL = 1,
};

struct DistOutputSpec {
    DistOutputOwnership ownership{DistOutputOwnership::ALLOCATED};
    size_t size{0};
    DistTensorKey key{};
    void *external_ptr{nullptr};
};

struct DistSubmitOutput {
    void *ptr{nullptr};
    size_t size{0};
};

struct DistSubmitResult {
    DistTaskSlot task_slot{DIST_INVALID_SLOT};
    std::vector<DistSubmitOutput> outputs;
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

    // Submit a task.  Returns allocated slot + output buffer pointers.
    DistSubmitResult submit(
        WorkerType worker_type, const WorkerPayload &base_payload, const std::vector<DistInputSpec> &inputs,
        const std::vector<DistOutputSpec> &outputs
    );

    // Submit a group task: N args → N workers, 1 DAG node.
    // All args' input/output tensors are unioned for dependency tracking.
    // The task only reaches COMPLETED when all N workers finish.
    DistSubmitResult submit_group(
        WorkerType worker_type, const WorkerPayload &base_payload, const std::vector<const void *> &args_list,
        const std::vector<DistInputSpec> &inputs, const std::vector<DistOutputSpec> &outputs
    );

    void scope_begin();
    void scope_end();

    // Called by Scheduler (via DistWorker) when a task becomes CONSUMED:
    // erases TensorMap entries and releases the ring slot.
    void on_consumed(DistTaskSlot slot);

private:
    DistTensorMap *tensormap_ = nullptr;
    DistRing *ring_ = nullptr;
    DistScope *scope_ = nullptr;
    DistReadyQueue *ready_queue_ = nullptr;
    DistTaskSlotState *slots_ = nullptr;
    int32_t num_slots_ = 0;

    DistTaskSlotState &slot_state(DistTaskSlot s) { return slots_[s]; }

    // Release one fanout reference on 'slot'.
    // If all references are released → transition to CONSUMED.
    void release_ref(DistTaskSlot slot);
};
