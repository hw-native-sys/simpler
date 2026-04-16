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

#include <cstdint>
#include <cstring>
#include <stdexcept>

void DistOrchestrator::init(
    DistTensorMap *tensormap, DistRing *allocator, DistScope *scope, DistReadyQueue *ready_next_level_queue,
    DistReadyQueue *ready_sub_queue
) {
    tensormap_ = tensormap;
    allocator_ = allocator;
    scope_ = scope;
    ready_next_level_queue_ = ready_next_level_queue;
    ready_sub_queue_ = ready_sub_queue;
    active_tasks_.store(0, std::memory_order_relaxed);
}

DistTaskSlotState &DistOrchestrator::slot_state(DistTaskSlot s) {
    DistTaskSlotState *p = allocator_->slot_state(s);
    if (!p) throw std::runtime_error("DistOrchestrator::slot_state: invalid slot id");
    return *p;
}

// ---------------------------------------------------------------------------
// alloc(shape, dtype) — user-facing intermediate buffer from the HeapRing
// ---------------------------------------------------------------------------

uint64_t DistOrchestrator::output_alloc_bytes(const ContinuousTensor &t) {
    return dist_align_up(t.nbytes(), DIST_HEAP_ALIGN);
}

ContinuousTensor DistOrchestrator::alloc(const std::vector<uint32_t> &shape, DataType dtype) {
    if (shape.size() > CONTINUOUS_TENSOR_MAX_DIMS) {
        throw std::invalid_argument("DistOrchestrator::alloc: shape exceeds CONTINUOUS_TENSOR_MAX_DIMS");
    }

    uint64_t numel = 1;
    for (uint32_t d : shape)
        numel *= static_cast<uint64_t>(d);
    uint64_t bytes = numel * get_element_size(dtype);
    uint64_t aligned = dist_align_up(bytes, DIST_HEAP_ALIGN);

    // 0-byte request (e.g. shape with a zero dim) flows straight through the
    // allocator as a slot-only claim — matches reserve_outputs_and_slot.
    // Skip tensormap registration when the returned heap_ptr is nullptr,
    // since 0 is the sentinel for "no tensor" in infer_deps.
    //
    // Inherit the caller's scope depth so alloc buffers land in the same
    // ring as any tasks submitted inside that scope — an alloc inside a
    // nested `with orch.scope():` uses the nested ring and reclaims
    // independently of the outer ring (Strict-1).
    DistAllocResult ar = allocator_->alloc(aligned, scope_->current_depth());
    if (ar.slot == DIST_INVALID_SLOT) {
        throw std::runtime_error("DistOrchestrator::alloc: allocator shutdown");
    }

    DistTaskSlotState &s = slot_state(ar.slot);
    s.reset();

    uint64_t key = reinterpret_cast<uint64_t>(ar.heap_ptr);
    if (key != 0) {
        tensormap_->insert(key, ar.slot);
        s.output_keys.push_back(key);
    }

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
    if (scope_ref > 0) scope_->register_task(ar.slot);

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
    std::vector<TaskArgs> args_list
) {
    if (args_list.empty()) throw std::invalid_argument("DistOrchestrator: args_list must not be empty");

    // Track this submission for drain() before any allocations so the count
    // is incremented exactly once per submitted DAG node, regardless of the
    // group_size N.
    active_tasks_.fetch_add(1, std::memory_order_relaxed);

    // --- Step 1: Atomically claim slot + auto-alloc any OUTPUT tensors that
    // arrived with a null data pointer. Both resources come from the same
    // merged allocator (Strict-2) so there is no partial-failure rollback
    // path.
    DistAllocResult ar = reserve_outputs_and_slot(args_list);
    if (ar.slot == DIST_INVALID_SLOT) {
        active_tasks_.fetch_sub(1, std::memory_order_relaxed);
        throw std::runtime_error("DistOrchestrator: allocator shutdown");
    }
    DistTaskSlot slot = ar.slot;

    DistTaskSlotState &s = slot_state(slot);
    s.reset();

    s.worker_type = worker_type;
    s.callable = callable_ptr;
    s.callable_id = callable_id;
    s.config = config;

    // --- Step 2: Walk tags → tensormap.lookup (deps) + tensormap.insert
    // (outputs). Must happen before we move args_list into the slot because
    // infer_deps reads tensor data pointers and tags from it.
    std::vector<DistTaskSlot> producers;
    infer_deps(slot, args_list, producers, s.output_keys);

    // --- Step 3: Store TaskArgs directly (no chip-storage pre-build) ---
    // Dispatch builds a TaskArgsView on demand via `slot.args_view(i)`
    // (THREAD mode) or write_blob → read_blob (PROCESS mode). The L2 ABI
    // ChipStorageTaskArgs conversion now runs inside ChipWorker::run
    // rather than at submit time.
    if (args_list.size() == 1) {
        s.is_group_ = false;
        s.task_args = std::move(args_list.front());
    } else {
        s.is_group_ = true;
        s.task_args_list = std::move(args_list);
    }

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
    // Strict-4: push to the queue dedicated to this task's worker type so a
    // saturated sub pool cannot stall next-level dispatch (and vice versa).
    if (live_fanins == 0) {
        s.state.store(TaskState::READY, std::memory_order_release);
        ready_queue_for(worker_type)->push(slot);
    } else {
        s.state.store(TaskState::PENDING, std::memory_order_release);
    }

    return DistSubmitResult{slot};
}

// =============================================================================
// reserve_outputs_and_slot — atomic slot + heap carve-up for this submit
// =============================================================================
//
// Walks every OUTPUT-tagged tensor that arrived with `data == 0` and reserves
// aligned slabs out of a single contiguous HeapRing allocation. OUTPUT tensors
// with a user-supplied data pointer are left untouched (that's the
// OUTPUT_EXISTING-equivalent back-compat path for callers that pre-fill
// OUTPUT.data themselves). The single allocator call owns both the slot and
// the heap range, so there is no partial-failure rollback.

DistAllocResult DistOrchestrator::reserve_outputs_and_slot(std::vector<TaskArgs> &args_list) {
    uint64_t total_bytes = 0;
    for (const TaskArgs &a : args_list) {
        for (int32_t i = 0; i < a.tensor_count(); ++i) {
            if (a.tag(i) != TensorArgType::OUTPUT) continue;
            if (a.tensor(i).data != 0) continue;  // user supplied a pointer — leave alone
            total_bytes += output_alloc_bytes(a.tensor(i));
        }
    }

    DistAllocResult ar = allocator_->alloc(total_bytes, scope_->current_depth());
    if (ar.slot == DIST_INVALID_SLOT) return ar;

    // Hand slabs out in the same order we counted them.
    uint64_t off = 0;
    char *base = static_cast<char *>(ar.heap_ptr);
    for (TaskArgs &a : args_list) {
        for (int32_t i = 0; i < a.tensor_count(); ++i) {
            if (a.tag(i) != TensorArgType::OUTPUT) continue;
            ContinuousTensor &t = a.tensor(i);
            if (t.data != 0) continue;
            uint64_t slab = output_alloc_bytes(t);
            t.data = reinterpret_cast<uint64_t>(base + off);
            off += slab;
        }
    }
    return ar;
}

// =============================================================================
// infer_deps — tag-driven dependency inference
// =============================================================================

void DistOrchestrator::infer_deps(
    DistTaskSlot slot, const std::vector<TaskArgs> &args_list, std::vector<DistTaskSlot> &producers,
    std::vector<uint64_t> &output_keys
) {
    auto add_unique_producer = [&](DistTaskSlot p) {
        // Group submits walk many TaskArgs under one slot: if two entries in
        // the same group tag the same buffer (e.g. both OUTPUT 0xCAFE), the
        // second-pass lookup would return the slot that the first pass just
        // inserted — a self-loop. Skip it.
        if (p == slot) return;
        for (DistTaskSlot existing : producers) {
            if (existing == p) return;
        }
        producers.push_back(p);
    };

    // Tag-driven dependency inference — mirrors L2
    // (src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.cpp
    //  steps B and 4):
    //   INPUT            → lookup only (RaW)
    //   INOUT            → lookup + insert (RaW + WaW)
    //   OUTPUT_EXISTING  → insert only (user-provided buffer; any WaW dep on
    //                      the creator must be expressed via INOUT instead)
    //   OUTPUT           → insert only (pure overwrite; if auto-alloc is
    //                      needed, the data ptr is assigned in
    //                      reserve_outputs_and_slot before this step)
    //   NO_DEP           → skip
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
    // Using `>= total + 1` keeps scope_end from prematurely consuming while
    // a consumer (or a HeapRing peer) is still live.
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

    // HeapRing-owned OUTPUT slabs are reclaimed implicitly when the allocator
    // advances last_alive past this slot — no per-slot munmap needed.
    allocator_->release(slot);

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
    {
        std::unique_lock<std::mutex> lk(drain_mu_);
        drain_cv_.wait(lk, [this] {
            return active_tasks_.load(std::memory_order_acquire) == 0;
        });
    }
    // Every slot is CONSUMED (active_tasks_ == 0 ⇒ allocator last_alive_ ==
    // next_task_id_). Drop all per-slot state so the next Worker.run()
    // starts from task_id = 0 with no accumulated memory.
    allocator_->reset_to_empty();
}
