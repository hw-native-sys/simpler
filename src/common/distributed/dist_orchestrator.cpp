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

#include <sys/mman.h>

#include <cstdint>
#include <cstring>
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
    active_tasks_.store(0, std::memory_order_relaxed);
}

ContinuousTensor DistOrchestrator::alloc(const std::vector<uint32_t> &shape, DataType dtype) {
    if (shape.size() > CONTINUOUS_TENSOR_MAX_DIMS) {
        throw std::invalid_argument("DistOrchestrator::alloc: shape exceeds CONTINUOUS_TENSOR_MAX_DIMS");
    }

    // --- Compute size and mmap a MAP_SHARED|MAP_ANONYMOUS region ---
    // Page-align so munmap on this exact size is valid.
    size_t numel = 1;
    for (uint32_t d : shape)
        numel *= static_cast<size_t>(d);
    size_t bytes = numel * get_element_size(dtype);
    static constexpr size_t PAGE = 4096;
    size_t mmap_bytes = (bytes + PAGE - 1) & ~(PAGE - 1);
    if (mmap_bytes == 0) mmap_bytes = PAGE;

    void *buf = mmap(nullptr, mmap_bytes, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (buf == MAP_FAILED) {
        throw std::runtime_error("DistOrchestrator::alloc: mmap failed");
    }

    // --- Synthetic task slot to own the buffer's lifecycle ---
    DistTaskSlot slot = ring_->alloc();
    if (slot == DIST_INVALID_SLOT) {
        munmap(buf, mmap_bytes);
        throw std::runtime_error("DistOrchestrator::alloc: ring shutdown");
    }
    DistTaskSlotState &s = slot_state(slot);
    s.reset();
    s.alloc_bufs.push_back(buf);
    s.alloc_sizes.push_back(mmap_bytes);

    // Register the buffer as this slot's output in the TensorMap so any
    // downstream task tagging it as INPUT/INOUT lookups this slot as producer.
    uint64_t key = reinterpret_cast<uint64_t>(buf);
    tensormap_->insert(key, slot);
    s.output_keys.push_back(key);

    // No fanin — alloc has no work to wait on.
    s.fanin_count = 0;
    s.fanin_released.store(0, std::memory_order_relaxed);

    // Initial fanout_total = scope_ref. Consumers that wire on this slot
    // will increment fanout_total in infer_deps.
    int32_t scope_ref = (scope_->depth() > 0) ? 1 : 0;
    {
        std::lock_guard<std::mutex> lk(s.fanout_mu);
        s.fanout_total = scope_ref;
    }
    // Simulate the self try_consume that on_task_complete would normally
    // contribute for a slot that ran through the scheduler. Without this
    // bump, the fanout-release threshold (`>= total + 1`) would be one
    // short and the slot would never reach CONSUMED.
    s.fanout_released.store(1, std::memory_order_relaxed);
    if (scope_ref > 0) scope_->register_task(slot);

    // Mark COMPLETED — alloc has no work, so it's "done" immediately.
    // Downstream consumers in infer_deps see this state and skip live_fanin
    // wiring (consumer is immediately ready) but still wire fanout (so this
    // slot waits for them before being consumed and freeing its buffer).
    s.state.store(TaskState::COMPLETED, std::memory_order_release);

    active_tasks_.fetch_add(1, std::memory_order_relaxed);

    ContinuousTensor t{};
    t.data = key;
    t.dtype = dtype;
    t.ndims = static_cast<uint32_t>(shape.size());
    for (size_t i = 0; i < shape.size(); ++i)
        t.shapes[i] = shape[i];
    return t;
}

// =============================================================================
// User-facing submit_* — thin wrappers around submit_impl
// =============================================================================

DistSubmitResult
DistOrchestrator::submit_next_level(uint64_t callable, const TaskArgs &args, const ChipCallConfig &config) {
    return submit_impl(WorkerType::NEXT_LEVEL, callable, /*callable_id=*/-1, config, {args});
}

DistSubmitResult DistOrchestrator::submit_next_level_group(
    uint64_t callable, const std::vector<TaskArgs> &args_list, const ChipCallConfig &config
) {
    return submit_impl(WorkerType::NEXT_LEVEL, callable, /*callable_id=*/-1, config, args_list);
}

DistSubmitResult DistOrchestrator::submit_sub(int32_t callable_id, const TaskArgs &args) {
    return submit_impl(WorkerType::SUB, /*callable_ptr=*/0, callable_id, ChipCallConfig{}, {args});
}

DistSubmitResult DistOrchestrator::submit_sub_group(int32_t callable_id, const std::vector<TaskArgs> &args_list) {
    return submit_impl(WorkerType::SUB, /*callable_ptr=*/0, callable_id, ChipCallConfig{}, args_list);
}

// =============================================================================
// submit_impl — shared 7-step submit machinery
// =============================================================================

DistSubmitResult DistOrchestrator::submit_impl(
    WorkerType worker_type, uint64_t callable_ptr, int32_t callable_id, const ChipCallConfig &config,
    const std::vector<TaskArgs> &args_list
) {
    if (args_list.empty()) throw std::invalid_argument("DistOrchestrator: args_list must not be empty");

    // Track this submission for drain() before any allocations so the count
    // is incremented exactly once per submitted DAG node, regardless of the
    // group_size N.
    active_tasks_.fetch_add(1, std::memory_order_relaxed);

    // --- Step 1: Alloc slot (blocks if ring full) ---
    DistTaskSlot slot = ring_->alloc();
    if (slot == DIST_INVALID_SLOT) throw std::runtime_error("DistOrchestrator: ring shutdown");

    DistTaskSlotState &s = slot_state(slot);
    s.reset();

    s.worker_type = worker_type;
    s.callable_ptr = callable_ptr;
    s.callable_id = callable_id;
    s.config = config;

    // --- Step 2: Per-worker chip storage (one ChipStorageTaskArgs per group member) ---
    s.chip_storage_list.reserve(args_list.size());
    for (const TaskArgs &a : args_list) {
        s.chip_storage_list.push_back(view_to_chip_storage(make_view(a)));
    }

    // --- Step 3 + 4: Walk tags → tensormap.lookup (deps) + tensormap.insert (outputs) ---
    std::vector<DistTaskSlot> producers;
    infer_deps(slot, args_list, producers, s.output_keys);

    // --- Step 5: Finalize fanin — lock each producer's fanout_mu, attach ---
    //
    // For COMPLETED producers (notably alloc-created synthetic slots), we
    // still wire the fanout edge so the producer waits for this consumer
    // before being CONSUMED (and freeing any owned buffers). The consumer
    // itself doesn't gain a live fanin — it can run immediately because the
    // producer is already done. CONSUMED producers are gone (resources freed),
    // so we skip them entirely.
    int32_t live_fanins = 0;
    for (DistTaskSlot prod : producers) {
        DistTaskSlotState &ps = slot_state(prod);
        std::lock_guard<std::mutex> lk(ps.fanout_mu);

        TaskState ps_state = ps.state.load(std::memory_order_acquire);
        if (ps_state == TaskState::CONSUMED) {
            continue;
        }
        ps.fanout_consumers.push_back(slot);
        ps.fanout_total++;
        s.fanin_producers.push_back(prod);
        if (ps_state != TaskState::COMPLETED) {
            live_fanins++;
        }
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

    // --- Step 6: If no live fanins → READY ---
    if (live_fanins == 0) {
        s.state.store(TaskState::READY, std::memory_order_release);
        ready_queue_->push(slot);
    } else {
        s.state.store(TaskState::PENDING, std::memory_order_release);
    }

    return DistSubmitResult{slot};
}

// =============================================================================
// infer_deps — tag-driven dependency inference
// =============================================================================

void DistOrchestrator::infer_deps(
    DistTaskSlot slot, const std::vector<TaskArgs> &args_list, std::vector<DistTaskSlot> &producers,
    std::vector<uint64_t> &output_keys
) {
    auto add_unique_producer = [&](DistTaskSlot p) {
        for (DistTaskSlot existing : producers) {
            if (existing == p) return;
        }
        producers.push_back(p);
    };

    // Inputs (and INOUT) → lookup producer; outputs (and INOUT, OUTPUT_EXISTING)
    // → insert as producer of `slot`. NO_DEP tags are skipped.
    for (const TaskArgs &a : args_list) {
        for (int32_t i = 0; i < a.tensor_count(); ++i) {
            uint64_t key = a.tensor(i).data;
            if (key == 0) continue;  // null tensor — nothing to track
            TensorArgType tag = a.tag(i);
            switch (tag) {
            case TensorArgType::INPUT: {
                DistTaskSlot prod = tensormap_->lookup(key);
                if (prod != DIST_INVALID_SLOT) add_unique_producer(prod);
                break;
            }
            case TensorArgType::INOUT: {
                DistTaskSlot prod = tensormap_->lookup(key);
                if (prod != DIST_INVALID_SLOT) add_unique_producer(prod);
                tensormap_->insert(key, slot);
                output_keys.push_back(key);
                break;
            }
            case TensorArgType::OUTPUT:
            case TensorArgType::OUTPUT_EXISTING: {
                tensormap_->insert(key, slot);
                output_keys.push_back(key);
                break;
            }
            case TensorArgType::NO_DEP:
            default:
                break;
            }
        }
    }
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
    // Threshold matches DistScheduler::try_consume: total contributors are
    // 1 (self try_consume from on_task_complete, or the alloc-time sim) +
    // N (per consumer's deferred try_consume) + 1 (this scope_end release)
    // = N + 2 = total + 1 where total = scope_ref + N.
    // Using `>= total + 1` keeps scope_end from prematurely consuming when
    // a consumer is still running — important once on_consumed actually
    // frees runtime-owned buffers (orch.alloc).
    if (released >= total + 1 && s.state.load(std::memory_order_acquire) == TaskState::COMPLETED) {
        on_consumed(slot);
    }
}

bool DistOrchestrator::on_consumed(DistTaskSlot slot) {
    DistTaskSlotState &s = slot_state(slot);

    // Idempotent: the threshold can be hit by either release_ref (scope_end,
    // Orch thread) or try_consume (consumer's deferred release, scheduler
    // thread). Whichever fires last wins; subsequent callers see CONSUMED
    // and bail.
    TaskState expected = TaskState::COMPLETED;
    if (!s.state.compare_exchange_strong(
            expected, TaskState::CONSUMED, std::memory_order_acq_rel, std::memory_order_acquire
        )) {
        return false;
    }

    tensormap_->erase_task_outputs(s.output_keys);

    // Free any runtime-owned intermediate buffers (orch.alloc).
    for (size_t i = 0; i < s.alloc_bufs.size(); ++i) {
        munmap(s.alloc_bufs[i], s.alloc_sizes[i]);
    }
    s.alloc_bufs.clear();
    s.alloc_sizes.clear();

    ring_->release(slot);

    // Decrement active-task counter so drain() observes completion. Gated
    // on the CAS win so both consume paths — release_ref (Orch thread,
    // scope_end) and try_consume (scheduler thread, consumer's deferred
    // release) — decrement exactly once. Notify drain_cv when the count
    // hits zero.
    int32_t remaining = active_tasks_.fetch_sub(1, std::memory_order_acq_rel) - 1;
    if (remaining == 0) {
        std::lock_guard<std::mutex> lk(drain_mu_);
        drain_cv_.notify_all();
    }
    return true;
}

void DistOrchestrator::drain() {
    std::unique_lock<std::mutex> lk(drain_mu_);
    drain_cv_.wait(lk, [this] {
        return active_tasks_.load(std::memory_order_acquire) == 0;
    });
}
