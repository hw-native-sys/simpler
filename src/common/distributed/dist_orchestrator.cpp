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

#include "dist_orchestrator.h"

#include <stdexcept>

void DistOrchestrator::init(
    DistTensorMap *tensormap, DistRing *ring, DistScope *scope, DistReadyQueue *ready_queue, DistTaskSlotState *slots,
    int32_t num_slots
) {
    tensormap_ = tensormap;
    ring_ = ring;
    scope_ = scope;
    ready_queue_ = ready_queue;
    slots_ = slots;
    num_slots_ = num_slots;
}

// =============================================================================
// submit() — delegates to submit_group with a single-element args_list
// =============================================================================

DistSubmitResult DistOrchestrator::submit(
    WorkerType worker_type, const WorkerPayload &base_payload, const std::vector<DistInputSpec> &inputs,
    const std::vector<DistOutputSpec> &output_specs
) {
    return submit_group(worker_type, base_payload, {base_payload.args}, inputs, output_specs);
}

// =============================================================================
// submit_group() — N args → N workers, 1 DAG node
// =============================================================================

DistSubmitResult DistOrchestrator::submit_group(
    WorkerType worker_type, const WorkerPayload &base_payload, const std::vector<const void *> &args_list,
    const std::vector<DistInputSpec> &inputs, const std::vector<DistOutputSpec> &output_specs
) {
    if (args_list.empty()) throw std::invalid_argument("DistOrchestrator: args_list must not be empty");

    // --- Step 1: Alloc slot (blocks if ring full) ---
    DistTaskSlot slot = ring_->alloc();
    if (slot == DIST_INVALID_SLOT) throw std::runtime_error("DistOrchestrator: ring shutdown");

    DistTaskSlotState &s = slot_state(slot);
    s.reset();

    // --- Store per-worker args list ---
    s.args_list = args_list;

    // --- Step 2: Allocate output buffers ---
    DistSubmitResult result;
    result.task_slot = slot;
    result.outputs.reserve(output_specs.size());

    s.output_bufs.reserve(output_specs.size());
    s.output_sizes.reserve(output_specs.size());
    s.output_ownerships.reserve(output_specs.size());
    s.output_keys.reserve(output_specs.size());

    for (const DistOutputSpec &spec : output_specs) {
        void *buf = nullptr;
        DistTensorKey key = spec.key;
        if (spec.ownership == DistOutputOwnership::ALLOCATED) {
            buf = spec.size > 0 ? ::operator new(spec.size) : nullptr;
            key = DistTensorKey{-1, reinterpret_cast<uint64_t>(buf)};
        } else {
            buf = spec.external_ptr;
        }
        s.output_bufs.push_back(buf);
        s.output_sizes.push_back(spec.size);
        s.output_ownerships.push_back(static_cast<int32_t>(spec.ownership));
        result.outputs.push_back({buf, spec.size});
        s.output_keys.push_back(key);
    }

    // --- Step 3: TensorMap lookup — collect producer slots ---
    // Inputs are unioned across all args (specified via DistInputSpec)
    std::vector<DistTaskSlot> producers;
    producers.reserve(inputs.size());
    for (const DistInputSpec &inp : inputs) {
        DistTaskSlot prod = tensormap_->lookup(inp.key);
        if (prod != DIST_INVALID_SLOT) {
            bool found = false;
            for (DistTaskSlot p : producers) {
                if (p == prod) {
                    found = true;
                    break;
                }
            }
            if (!found) producers.push_back(prod);
        }
    }

    // --- Step 4: TensorMap insert — register outputs ---
    for (size_t i = 0; i < output_specs.size(); ++i) {
        if (s.output_keys[i].base_ptr != 0) {
            tensormap_->insert(s.output_keys[i], slot);
        }
    }

    // --- Step 5: Write task slot initial state ---
    WorkerPayload payload = base_payload;
    payload.task_slot = slot;
    payload.worker_type = worker_type;
    s.payload = payload;

    // --- Step 6: Finalize fanin — lock each producer's fanout_mu, attach ---
    int32_t live_fanins = 0;
    for (DistTaskSlot prod : producers) {
        DistTaskSlotState &ps = slot_state(prod);
        std::lock_guard<std::mutex> lk(ps.fanout_mu);

        TaskState ps_state = ps.state.load(std::memory_order_acquire);
        if (ps_state == TaskState::COMPLETED || ps_state == TaskState::CONSUMED) {
            continue;
        }
        ps.fanout_consumers.push_back(slot);
        ps.fanout_total++;
        live_fanins++;
        s.fanin_producers.push_back(prod);
    }

    s.fanin_count = live_fanins;
    s.fanin_released.store(0, std::memory_order_relaxed);

    int32_t scope_ref = (scope_->depth() > 0) ? 1 : 0;
    {
        std::lock_guard<std::mutex> lk(s.fanout_mu);
        s.fanout_total = scope_ref;
    }
    s.fanout_released.store(0, std::memory_order_relaxed);

    if (scope_ref > 0) scope_->register_task(slot);

    // --- Step 7: If no live fanins → READY ---
    if (live_fanins == 0) {
        s.state.store(TaskState::READY, std::memory_order_release);
        ready_queue_->push(slot);
    } else {
        s.state.store(TaskState::PENDING, std::memory_order_release);
    }

    return result;
}

// =============================================================================
// Scope
// =============================================================================

void DistOrchestrator::scope_begin() { scope_->scope_begin(); }

void DistOrchestrator::scope_end() {
    scope_->scope_end([this](DistTaskSlot slot) {
        release_ref(slot);
    });
}

// =============================================================================
// Reference release helpers
// =============================================================================

void DistOrchestrator::release_ref(DistTaskSlot slot) {
    DistTaskSlotState &s = slot_state(slot);
    int32_t released = s.fanout_released.fetch_add(1, std::memory_order_acq_rel) + 1;
    int32_t total;
    {
        std::lock_guard<std::mutex> lk(s.fanout_mu);
        total = s.fanout_total;
    }
    // Only consume COMPLETED tasks — never RUNNING (task may still be executing).
    // Threshold is `total` (not total+1): release_ref counts explicit refs (scope,
    // downstream consumers) whereas try_consume adds its own self-ref on top.
    if (released >= total && s.state.load(std::memory_order_acquire) == TaskState::COMPLETED) {
        on_consumed(slot);
    }
}

void DistOrchestrator::on_consumed(DistTaskSlot slot) {
    DistTaskSlotState &s = slot_state(slot);
    s.state.store(TaskState::CONSUMED, std::memory_order_release);
    tensormap_->erase_task_outputs(s.output_keys);
    ring_->release(slot);
}
