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

#include "orchestrator.h"

#include <cstdint>
#include <stdexcept>
#include <unordered_set>

#include "worker_manager.h"

void Orchestrator::init(
    TensorMap *tensormap, Ring *allocator, Scope *scope, ReadyQueue *ready_next_level_queue,
    ReadyQueue *ready_sub_queue, WorkerManager *manager, std::function<void()> ready_notify_cb
) {
    tensormap_ = tensormap;
    allocator_ = allocator;
    scope_ = scope;
    ready_next_level_queue_ = ready_next_level_queue;
    ready_sub_queue_ = ready_sub_queue;
    manager_ = manager;
    ready_notify_cb_ = std::move(ready_notify_cb);
    active_tasks_.store(0, std::memory_order_relaxed);
}

uint64_t Orchestrator::malloc(int worker_id, size_t size) {
    auto *wt = manager_->get_worker_by_id(WorkerType::NEXT_LEVEL, worker_id);
    if (!wt) throw std::runtime_error("Orchestrator::malloc: invalid worker_id");
    return wt->control_malloc(size);
}

void Orchestrator::free(int worker_id, uint64_t ptr) {
    auto *wt = manager_->get_worker_by_id(WorkerType::NEXT_LEVEL, worker_id);
    if (!wt) throw std::runtime_error("Orchestrator::free: invalid worker_id");
    wt->control_free(ptr);
}

void Orchestrator::copy_to(int worker_id, uint64_t dst, uint64_t src, size_t size) {
    auto *wt = manager_->get_worker_by_id(WorkerType::NEXT_LEVEL, worker_id);
    if (!wt) throw std::runtime_error("Orchestrator::copy_to: invalid worker_id");
    wt->control_copy_to(dst, src, size);
}

void Orchestrator::copy_from(int worker_id, uint64_t dst, uint64_t src, size_t size) {
    auto *wt = manager_->get_worker_by_id(WorkerType::NEXT_LEVEL, worker_id);
    if (!wt) throw std::runtime_error("Orchestrator::copy_from: invalid worker_id");
    wt->control_copy_from(dst, src, size);
}

TaskSlotState &Orchestrator::slot_state(TaskSlot s) {
    TaskSlotState *p = allocator_->slot_state(s);
    if (!p) throw std::runtime_error("Orchestrator::slot_state: invalid slot id");
    return *p;
}

// ---------------------------------------------------------------------------
// alloc(shape, dtype) — user-facing intermediate buffer from the HeapRing
// ---------------------------------------------------------------------------

uint64_t Orchestrator::output_alloc_bytes(const Tensor &t) { return align_up(t.nbytes(), HEAP_ALIGN); }

Tensor Orchestrator::alloc(const std::vector<uint32_t> &shape, DataType dtype) {
    if (shape.empty()) {
        // Rank-0 tensors are not supported across the ABI (Tensor enforces
        // ndims > 0). Reject here so we never allocate + register a buffer in
        // the tensormap only to hand back an unusable addr==0 sentinel.
        throw std::invalid_argument("Orchestrator::alloc: shape must have at least one dimension");
    }
    if (shape.size() > MAX_TENSOR_DIMS) {
        throw std::invalid_argument("Orchestrator::alloc: shape exceeds MAX_TENSOR_DIMS");
    }

    uint64_t numel = 1;
    for (uint32_t d : shape)
        numel *= static_cast<uint64_t>(d);
    uint64_t bytes = numel * get_element_size(dtype);
    uint64_t aligned = align_up(bytes, HEAP_ALIGN);

    // 0-byte request (e.g. shape with a zero dim) flows straight through the
    // allocator as a slot-only claim — matches reserve_outputs_and_slot.
    // Skip tensormap registration when the returned heap_ptr is nullptr,
    // since 0 is the sentinel for "no tensor" in infer_deps.
    //
    // Inherit the caller's scope depth so alloc buffers land in the same
    // ring as any tasks submitted inside that scope — an alloc inside a
    // nested `with orch.scope():` uses the nested ring and reclaims
    // independently of the outer ring (Strict-1).
    AllocResult ar = allocator_->alloc(aligned, scope_->current_depth());
    if (ar.slot == INVALID_SLOT) {
        throw std::runtime_error("Orchestrator::alloc: allocator shutdown");
    }

    TaskSlotState &s = slot_state(ar.slot);
    s.reset();

    uint64_t ptr = reinterpret_cast<uint64_t>(ar.heap_ptr);
    if (ptr != 0) {
        TensorKey key = TensorKey::local_host(ptr);
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

    // Build a contiguous external Tensor over the allocated buffer. ptr may be
    // 0 for a 0-byte request (a shape with a zero dim), in which case
    // init_external sets buffer.addr == 0 — the "no tensor" sentinel honored by
    // infer_deps; buffer.size carries numel*elem. shape is non-empty (rejected
    // at entry), so ndims >= 1 holds for init_external's assertion.
    Tensor t{};
    t.init_external(
        reinterpret_cast<void *>(ptr), bytes, shape.data(), static_cast<uint32_t>(shape.size()), dtype,
        /*version=*/0
    );
    return t;
}

// =============================================================================
// User-facing submit_* — thin wrappers around submit_impl
// =============================================================================

SubmitResult Orchestrator::submit_next_level(
    const CallableIdentity &callable, const TaskArgs &args, const CallConfig &config, int32_t worker_id,
    const std::vector<int32_t> &eligible_worker_ids, const RemoteTaskArgsSidecar &remote_sidecar
) {
    std::vector<int32_t> affinities;
    if (worker_id >= 0) affinities = {worker_id};
    std::vector<std::vector<int32_t>> worker_id_sets;
    if (!eligible_worker_ids.empty()) worker_id_sets = {eligible_worker_ids};
    std::vector<RemoteTaskArgsSidecar> sidecars;
    if (!remote_sidecar.tensors.empty() || !remote_sidecar.inline_payload.empty()) sidecars = {remote_sidecar};
    return submit_impl(
        WorkerType::NEXT_LEVEL, callable, config, {args}, std::move(affinities), std::move(worker_id_sets),
        std::move(sidecars)
    );
}

SubmitResult Orchestrator::submit_next_level_group(
    const CallableIdentity &callable, const std::vector<TaskArgs> &args_list, const CallConfig &config,
    const std::vector<int32_t> &worker_ids, const std::vector<std::vector<int32_t>> &eligible_worker_ids,
    const std::vector<RemoteTaskArgsSidecar> &remote_sidecars
) {
    return submit_impl(
        WorkerType::NEXT_LEVEL, callable, config, args_list, worker_ids, eligible_worker_ids, remote_sidecars
    );
}

SubmitResult Orchestrator::submit_sub(const CallableIdentity &callable, const TaskArgs &args) {
    return submit_impl(WorkerType::SUB, callable, CallConfig{}, {args});
}

SubmitResult Orchestrator::submit_sub_group(const CallableIdentity &callable, const std::vector<TaskArgs> &args_list) {
    return submit_impl(WorkerType::SUB, callable, CallConfig{}, args_list);
}

// =============================================================================
// submit_impl — shared 7-step submit machinery
// =============================================================================

SubmitResult Orchestrator::submit_impl(
    WorkerType worker_type, const CallableIdentity &callable, const CallConfig &config, std::vector<TaskArgs> args_list,
    std::vector<int32_t> affinities, std::vector<std::vector<int32_t>> eligible_worker_ids,
    std::vector<RemoteTaskArgsSidecar> remote_sidecars
) {
    if (args_list.empty()) throw std::invalid_argument("Orchestrator: args_list must not be empty");
    config.validate();
    validate_worker_eligibility(worker_type, args_list.size(), affinities, eligible_worker_ids);
    validate_remote_sidecars(args_list, remote_sidecars, eligible_worker_ids);

    // Fail-fast: if a previously-dispatched task has already failed, abort
    // this submit before any bookkeeping so the orch fn unwinds promptly
    // and no further work is queued. Tasks already in flight run to
    // completion; drain() picks up any remaining bookkeeping and rethrows
    // at the finally: _drain() site in Worker.run.
    if (manager_ && manager_->has_error()) {
        std::rethrow_exception(manager_->take_error());
    }

    // Track this submission for drain() before any allocations so the count
    // is incremented exactly once per submitted DAG node, regardless of the
    // group_size N.
    active_tasks_.fetch_add(1, std::memory_order_relaxed);

    // --- Step 1: Atomically claim slot + auto-alloc any OUTPUT tensors that
    // arrived with a null data pointer. Both resources come from the same
    // merged allocator (Strict-2) so there is no partial-failure rollback
    // path.
    AllocResult ar = reserve_outputs_and_slot(args_list, remote_sidecars);
    if (ar.slot == INVALID_SLOT) {
        active_tasks_.fetch_sub(1, std::memory_order_relaxed);
        throw std::runtime_error("Orchestrator: allocator shutdown");
    }
    TaskSlot slot = ar.slot;

    TaskSlotState &s = slot_state(slot);
    s.reset();

    s.worker_type = worker_type;
    s.callable = callable;
    s.config = config;
    s.eligible_worker_ids = std::move(eligible_worker_ids);

    // --- Step 2: Walk tags → tensormap.lookup (deps) + tensormap.insert
    // (outputs). Must happen before we move args_list into the slot because
    // infer_deps reads tensor data pointers and tags from it.
    std::vector<TaskSlot> producers;
    infer_deps(slot, args_list, affinities, remote_sidecars, producers, s.output_keys);

    // --- Step 3: Store TaskArgs directly (no chip-storage pre-build) ---
    // Dispatch builds a TaskArgsView on demand via `slot.args_view(i)`
    // (THREAD mode) or write_blob → read_blob (PROCESS mode). The L2 ABI
    // ChipStorageTaskArgs conversion now runs inside ChipWorker::run
    // rather than at submit time.
    if (args_list.size() == 1) {
        s.is_group_ = false;
        s.task_args = std::move(args_list.front());
        if (!remote_sidecars.empty()) s.remote_sidecar = std::move(remote_sidecars.front());
    } else {
        s.is_group_ = true;
        s.task_args_list = std::move(args_list);
        s.remote_sidecars = std::move(remote_sidecars);
    }
    s.affinities = std::move(affinities);

    // --- Step 5: Finalize fanin — lock each producer's fanout_mu, attach ---
    //
    // For COMPLETED producers (notably alloc-created synthetic slots), we
    // still wire the fanout edge so the producer waits for this consumer
    // before being CONSUMED (and freeing any owned buffers). The consumer
    // itself doesn't gain a live fanin — it can run immediately because the
    // producer is already done. CONSUMED producers are gone (resources freed),
    // so we skip them entirely.
    int32_t live_fanins = 0;
    bool poisoned_by_failed_producer = false;
    std::string poison_message;
    for (TaskSlot prod : producers) {
        TaskSlotState &ps = slot_state(prod);
        std::lock_guard<std::mutex> lk(ps.fanout_mu);

        TaskState ps_state = ps.state.load(std::memory_order_acquire);
        if (ps_state == TaskState::CONSUMED) {
            continue;
        }
        ps.fanout_consumers.push_back(slot);
        ps.fanout_total++;
        s.fanin_producers.push_back(prod);
        if (ps_state == TaskState::FAILED) {
            poisoned_by_failed_producer = true;
            if (poison_message.empty()) poison_message = ps.failure_message;
        } else if (ps_state != TaskState::COMPLETED) {
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

    if (poisoned_by_failed_producer) {
        if (poison_message.empty()) poison_message = "producer task failed";
        s.failure_message = poison_message;
        s.state.store(TaskState::FAILED, std::memory_order_release);
        std::vector<TaskSlot> fanin_producers = s.fanin_producers;
        try_consume(slot);
        for (TaskSlot prod : fanin_producers) {
            try_consume(prod);
        }
        return SubmitResult{slot};
    }

    // --- Step 6: If no live fanins → READY ---
    // Strict-4: push to the queue dedicated to this task's worker type so a
    // saturated sub pool cannot stall next-level dispatch (and vice versa).
    if (live_fanins == 0) {
        s.state.store(TaskState::READY, std::memory_order_release);
        ready_queue_for(worker_type)->push(slot);
        if (ready_notify_cb_) ready_notify_cb_();
    } else {
        s.state.store(TaskState::PENDING, std::memory_order_release);
    }

    return SubmitResult{slot};
}

void Orchestrator::validate_worker_eligibility(
    WorkerType worker_type, size_t args_count, const std::vector<int32_t> &affinities,
    const std::vector<std::vector<int32_t>> &eligible_worker_ids
) const {
    if (!affinities.empty() && affinities.size() != args_count) {
        throw std::invalid_argument(
            "Orchestrator: affinity length " + std::to_string(affinities.size()) + " does not match args length " +
            std::to_string(args_count)
        );
    }
    if (!eligible_worker_ids.empty() && eligible_worker_ids.size() != args_count) {
        throw std::invalid_argument(
            "Orchestrator: eligible worker-id set length " + std::to_string(eligible_worker_ids.size()) +
            " does not match args length " + std::to_string(args_count)
        );
    }

    const std::vector<int32_t> empty_eligible;
    for (size_t i = 0; i < args_count; ++i) {
        const auto &eligible = eligible_worker_ids.empty() ? empty_eligible : eligible_worker_ids[i];
        if (!eligible_worker_ids.empty() && eligible.empty()) {
            throw std::invalid_argument(
                "Orchestrator: final eligible worker-id set is empty for member " + std::to_string(i)
            );
        }
        if (manager_ != nullptr && !eligible_worker_ids.empty()) {
            for (int32_t worker_id : eligible) {
                if (manager_->get_worker_by_id(worker_type, worker_id) == nullptr) {
                    throw std::invalid_argument(
                        "Orchestrator: eligible worker-id " + std::to_string(worker_id) + " is not a registered worker"
                    );
                }
            }
        }
        int32_t affinity = affinities.empty() ? -1 : affinities[i];
        if (affinity < 0) continue;

        if (manager_ != nullptr) {
            auto *wt = worker_type == WorkerType::NEXT_LEVEL ? manager_->get_worker_by_id(worker_type, affinity) :
                                                               manager_->get_worker_by_index(worker_type, affinity);
            if (wt == nullptr) {
                throw std::invalid_argument(
                    "Orchestrator: worker affinity " + std::to_string(affinity) + " is not a registered worker"
                );
            }
            int32_t worker_id = wt->worker_id();
            bool allowed = eligible_worker_ids.empty();
            for (int32_t id : eligible) {
                if (id == worker_id) {
                    allowed = true;
                    break;
                }
            }
            if (!allowed) {
                throw std::invalid_argument(
                    "Orchestrator: worker affinity " + std::to_string(affinity) +
                    " is not in the slot's final eligible worker-id set"
                );
            }
        } else if (affinity >= 0 && !eligible_worker_ids.empty()) {
            bool allowed = false;
            for (int32_t id : eligible) {
                if (id == affinity) {
                    allowed = true;
                    break;
                }
            }
            if (!allowed) {
                throw std::invalid_argument(
                    "Orchestrator: worker affinity " + std::to_string(affinity) +
                    " is not in the slot's final eligible worker-id set"
                );
            }
        }
    }
}

void Orchestrator::validate_remote_sidecars(
    const std::vector<TaskArgs> &args_list, const std::vector<RemoteTaskArgsSidecar> &remote_sidecars,
    const std::vector<std::vector<int32_t>> &eligible_worker_ids
) const {
    if (remote_sidecars.empty()) return;
    if (remote_sidecars.size() != args_list.size()) {
        throw std::invalid_argument(
            "Orchestrator: remote sidecar length " + std::to_string(remote_sidecars.size()) +
            " does not match args length " + std::to_string(args_list.size())
        );
    }
    if (eligible_worker_ids.empty()) {
        throw std::invalid_argument("Orchestrator: remote sidecars require an explicit eligible worker-id set");
    }
    for (size_t g = 0; g < args_list.size(); ++g) {
        const TaskArgs &args = args_list[g];
        const RemoteTaskArgsSidecar &sidecar = remote_sidecars[g];
        if (sidecar.empty() && args.tensor_count() == 0) continue;
        if (sidecar.tensors.size() != static_cast<size_t>(args.tensor_count())) {
            throw std::invalid_argument("Orchestrator: remote sidecar tensor count does not match TaskArgs");
        }
        for (int32_t worker_id : eligible_worker_ids[g]) {
            if (manager_ == nullptr) continue;
            WorkerThread *wt = manager_->get_worker_by_id(WorkerType::NEXT_LEVEL, worker_id);
            if (wt == nullptr) {
                throw std::invalid_argument(
                    "Orchestrator: remote sidecar names an unknown worker " + std::to_string(worker_id)
                );
            }
            if (!wt->caps().remote) {
                throw std::invalid_argument(
                    "Orchestrator: remote sidecar cannot be submitted to local worker " + std::to_string(worker_id)
                );
            }
        }
        for (int32_t i = 0; i < args.tensor_count(); ++i) {
            const Tensor &tensor = args.tensor(i);
            const RemoteTensorSidecar &tensor_sidecar = sidecar.tensors[static_cast<size_t>(i)];
            if (tensor_sidecar.present && tensor.buffer.addr != 0) {
                throw std::invalid_argument("Orchestrator: remote tensor metadata data field must be zero");
            }
            if (!tensor_sidecar.present && tensor.buffer.addr != 0) {
                throw std::invalid_argument("Orchestrator: remote tensor uses a bare host pointer without sidecar");
            }
            if (args.tag(i) == TensorArgType::OUTPUT && tensor.buffer.addr == 0 && !tensor_sidecar.present) {
                throw std::invalid_argument("Orchestrator: remote OUTPUT tensor requires a RemoteTensorRef sidecar");
            }
            if (!tensor_sidecar.present && tensor.nbytes() != 0) {
                throw std::invalid_argument("Orchestrator: remote tensor payload requires a RemoteTensorRef sidecar");
            }
            if (tensor.is_child_memory() && !tensor_sidecar.present) {
                throw std::invalid_argument("Orchestrator: remote child-memory tensor requires a sidecar");
            }
            if (tensor_sidecar.present && tensor_sidecar.desc.address_space != RemoteAddressSpace::HOST_INLINE) {
                if (tensor_sidecar.desc.owner_worker_id < 0) {
                    throw std::invalid_argument("Orchestrator: remote tensor sidecar has invalid owner worker");
                }
                bool has_allowed_worker = false;
                for (int32_t worker_id : eligible_worker_ids[g]) {
                    has_allowed_worker = true;
                    if (tensor_sidecar.desc.address_space == RemoteAddressSpace::REMOTE_DEVICE &&
                        worker_id != tensor_sidecar.desc.owner_worker_id) {
                        throw std::invalid_argument(
                            "Orchestrator: remote tensor sidecar requires IMPORT_BUFFER before submitting to worker " +
                            std::to_string(worker_id)
                        );
                    }
                }
                if (!has_allowed_worker) {
                    throw std::invalid_argument("Orchestrator: remote tensor has no final eligible worker");
                }
            }
        }
    }
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

AllocResult Orchestrator::reserve_outputs_and_slot(
    std::vector<TaskArgs> &args_list, const std::vector<RemoteTaskArgsSidecar> &remote_sidecars
) {
    uint64_t total_bytes = 0;
    for (size_t g = 0; g < args_list.size(); ++g) {
        const TaskArgs &a = args_list[g];
        for (int32_t i = 0; i < a.tensor_count(); ++i) {
            if (a.tag(i) != TensorArgType::OUTPUT) continue;
            if (a.tensor(i).buffer.addr != 0) continue;  // user supplied a pointer — leave alone
            bool remote_output = !remote_sidecars.empty() &&
                                 static_cast<size_t>(i) < remote_sidecars[g].tensors.size() &&
                                 remote_sidecars[g].tensors[static_cast<size_t>(i)].present;
            if (remote_output) continue;
            total_bytes += output_alloc_bytes(a.tensor(i));
        }
    }

    AllocResult ar = allocator_->alloc(total_bytes, scope_->current_depth());
    if (ar.slot == INVALID_SLOT) return ar;

    // Hand slabs out in the same order we counted them.
    uint64_t off = 0;
    char *base = static_cast<char *>(ar.heap_ptr);
    for (size_t g = 0; g < args_list.size(); ++g) {
        TaskArgs &a = args_list[g];
        for (int32_t i = 0; i < a.tensor_count(); ++i) {
            if (a.tag(i) != TensorArgType::OUTPUT) continue;
            Tensor &t = a.tensor(i);
            if (t.buffer.addr != 0) continue;
            bool remote_output = !remote_sidecars.empty() &&
                                 static_cast<size_t>(i) < remote_sidecars[g].tensors.size() &&
                                 remote_sidecars[g].tensors[static_cast<size_t>(i)].present;
            if (remote_output) continue;
            uint64_t slab = output_alloc_bytes(t);
            t.buffer.addr = reinterpret_cast<uint64_t>(base + off);
            off += slab;
        }
    }
    return ar;
}

// =============================================================================
// infer_deps — tag-driven dependency inference
// =============================================================================

void Orchestrator::infer_deps(
    TaskSlot slot, const std::vector<TaskArgs> &args_list, const std::vector<int32_t> &affinities,
    const std::vector<RemoteTaskArgsSidecar> &remote_sidecars, std::vector<TaskSlot> &producers,
    std::vector<TensorKey> &output_keys
) {
    std::unordered_set<TaskSlot> producer_seen;
    size_t tensor_count_hint = 0;
    for (const TaskArgs &args : args_list) {
        tensor_count_hint += static_cast<size_t>(args.tensor_count());
    }
    producer_seen.reserve(tensor_count_hint);

    auto add_unique_producer = [&](TaskSlot p) {
        // Group submits walk many TaskArgs under one slot: if two entries in
        // the same group tag the same buffer (e.g. both OUTPUT 0xCAFE), the
        // second-pass lookup would return the slot that the first pass just
        // inserted — a self-loop. Skip it.
        if (p == slot) return;
        if (producer_seen.insert(p).second) {
            producers.push_back(p);
        }
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
    for (size_t g = 0; g < args_list.size(); ++g) {
        int32_t worker_id = (g < affinities.size()) ? affinities[g] : -1;
        const TaskArgs &a = args_list[g];
        for (int32_t i = 0; i < a.tensor_count(); ++i) {
            const Tensor &t = a.tensor(i);
            TensorKey key{};
            bool has_key = false;
            if (!remote_sidecars.empty()) {
                const auto &sidecar = remote_sidecars[g];
                if (static_cast<size_t>(i) < sidecar.tensors.size() &&
                    sidecar.tensors[static_cast<size_t>(i)].present) {
                    const RemoteTensorDesc &desc = sidecar.tensors[static_cast<size_t>(i)].desc;
                    TensorAddressKind kind = desc.address_space == RemoteAddressSpace::HOST_INLINE ?
                                                 TensorAddressKind::HOST_INLINE :
                                                 TensorAddressKind::REMOTE_BUFFER;
                    key = TensorKey::remote_buffer(
                        kind, desc.owner_worker_id, desc.buffer_id, desc.generation, desc.offset
                    );
                    has_key = true;
                }
            }
            if (!has_key) {
                if (t.buffer.addr == 0) continue;  // null tensor — nothing to track
                key = t.is_child_memory() ? TensorKey::local_child(t.buffer.addr, worker_id) :
                                            TensorKey::local_host(t.buffer.addr);
                has_key = true;
            }
            TensorArgType tag = a.tag(i);
            switch (tag) {
            case TensorArgType::INPUT: {
                TaskSlot prod = tensormap_->lookup(key);
                if (prod != INVALID_SLOT) add_unique_producer(prod);
                break;
            }
            case TensorArgType::INOUT: {
                TaskSlot prod = tensormap_->lookup(key);
                if (prod != INVALID_SLOT) add_unique_producer(prod);
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

void Orchestrator::scope_begin() { scope_->scope_begin(); }

void Orchestrator::scope_end() {
    scope_->scope_end([this](TaskSlot slot) {
        release_ref(slot);
    });
}

// =============================================================================
// Reference release helpers
// =============================================================================

void Orchestrator::release_ref(TaskSlot slot) { try_consume(slot); }

void Orchestrator::try_consume(TaskSlot slot) {
    TaskSlotState &s = slot_state(slot);
    int32_t released = s.fanout_released.fetch_add(1, std::memory_order_acq_rel) + 1;
    int32_t total;
    {
        std::lock_guard<std::mutex> lk(s.fanout_mu);
        total = s.fanout_total;
    }
    // Threshold matches Scheduler::try_consume. fanout_total counts
    // scope_ref + N consumer refs; the extra +1 is the terminal self
    // release. These refs can be released from completion, poison, or
    // scope_end paths in different orders.
    TaskState state = s.state.load(std::memory_order_acquire);
    if (released >= total + 1 && (state == TaskState::COMPLETED || state == TaskState::FAILED)) {
        on_consumed(slot);
    }
}

bool Orchestrator::on_consumed(TaskSlot slot) {
    TaskSlotState &s = slot_state(slot);

    // Idempotent: the threshold can be hit by either release_ref (scope_end,
    // Orch thread) or try_consume (consumer's deferred release, scheduler
    // thread). Whichever fires last wins; subsequent callers see CONSUMED
    // and bail.
    TaskState expected = TaskState::COMPLETED;
    if (!s.state.compare_exchange_strong(
            expected, TaskState::CONSUMED, std::memory_order_acq_rel, std::memory_order_acquire
        )) {
        expected = TaskState::FAILED;
        if (!s.state.compare_exchange_strong(
                expected, TaskState::CONSUMED, std::memory_order_acq_rel, std::memory_order_acquire
            )) {
            return false;
        }
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

void Orchestrator::drain() {
    {
        std::unique_lock<std::mutex> lk(drain_mu_);
        drain_cv_.wait(lk, [this] {
            return active_tasks_.load(std::memory_order_acquire) == 0;
        });
    }
    // Every slot is CONSUMED (active_tasks_ == 0 ⇒ allocator last_alive_ ==
    // next_task_id_). Drop all per-slot state so the next Worker.run()
    // starts from task_id = 0 with no accumulated memory.
    //
    // Hold the scheduler's loop mutex across the reset: active_tasks_ can reach
    // zero while the scheduler thread is still inside on_task_complete (it reads
    // the slot after the consume that drives the count to 0). Freeing the slots
    // here without this guard is a heap-use-after-free. The scheduler releases
    // loop_mu_ only between iterations, and with active_tasks_ == 0 it has no
    // further slots to touch, so this blocks at most one in-flight iteration.
    if (sched_loop_mu_ != nullptr) {
        std::lock_guard<std::mutex> sched_lk(*sched_loop_mu_);
        allocator_->reset_to_empty();
    } else {
        allocator_->reset_to_empty();
    }

    // Rethrow the first dispatch failure seen during this run. Deferred to
    // after allocator reset so the next Worker.run() can proceed cleanly
    // once clear_error() is called.
    if (manager_ && manager_->has_error()) {
        std::rethrow_exception(manager_->take_error());
    }
}

void Orchestrator::clear_error() {
    if (manager_) manager_->clear_error();
}
