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

#include <chrono>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_set>

#include "worker_manager.h"

void Orchestrator::init(
    TensorMap *tensormap, Ring *allocator, Scope *scope, ReadyQueue *ready_sub_queue,
    NextLevelReadyQueues *ready_next_level_queues, WorkerManager *manager, std::function<void()> ready_notify_cb
) {
    tensormap_ = tensormap;
    allocator_ = allocator;
    scope_ = scope;
    ready_sub_queue_ = ready_sub_queue;
    ready_next_level_queues_ = ready_next_level_queues;
    manager_ = manager;
    ready_notify_cb_ = std::move(ready_notify_cb);
}

bool Orchestrator::is_terminal(RunPhase phase) { return phase == RunPhase::COMPLETED || phase == RunPhase::FAILED; }

bool Orchestrator::acceptance_ready(const std::shared_ptr<RunState> &run) {
    return run->submission_closed && (run->pending_accepts.load(std::memory_order_acquire) == 0 ||
                                      is_terminal(run->phase.load(std::memory_order_acquire)));
}

std::shared_ptr<RunState> Orchestrator::get_run(RunId run_id) const {
    std::lock_guard<std::mutex> lk(runs_mu_);
    auto it = runs_.find(run_id);
    if (it == runs_.end()) throw std::invalid_argument("Orchestrator: unknown run id");
    return it->second;
}

std::shared_ptr<RunState> Orchestrator::current_building_run() const {
    std::lock_guard<std::mutex> lk(runs_mu_);
    auto it = runs_.find(building_run_id_);
    if (building_run_id_ == INVALID_RUN_ID || it == runs_.end()) {
        throw std::logic_error("Orchestrator: task submission requires an active run");
    }
    return it->second;
}

RunId Orchestrator::begin_run() {
    std::lock_guard<std::mutex> lk(runs_mu_);
    if (building_run_id_ != INVALID_RUN_ID) {
        throw std::logic_error("Orchestrator::begin_run: another run is still building");
    }
    if (next_run_id_ == INVALID_RUN_ID || next_run_id_ == std::numeric_limits<RunId>::max()) {
        throw std::overflow_error("Orchestrator::begin_run: run id space exhausted");
    }
    RunId run_id = next_run_id_++;
    runs_.emplace(run_id, std::make_shared<RunState>(run_id));
    building_run_id_ = run_id;
    return run_id;
}

void Orchestrator::finish_run_if_ready(const std::shared_ptr<RunState> &run) {
    bool notify = false;
    {
        std::lock_guard<std::mutex> lk(run->completion_mu);
        if (!run->submission_closed || run->active_tasks.load(std::memory_order_acquire) != 0 ||
            is_terminal(run->phase.load(std::memory_order_acquire))) {
            return;
        }
        bool failed = run->submission_failed || static_cast<bool>(run->first_error);
        run->phase.store(failed ? RunPhase::FAILED : RunPhase::COMPLETED, std::memory_order_release);
        notify = true;
    }
    if (notify) run->completion_cv.notify_all();
}

void Orchestrator::close_run_submission(RunId run_id) {
    auto run = get_run(run_id);
    {
        std::lock_guard<std::mutex> runs_lk(runs_mu_);
        if (building_run_id_ != run_id) {
            throw std::logic_error("Orchestrator::close_run_submission: run is not building");
        }
        building_run_id_ = INVALID_RUN_ID;
    }
    {
        std::lock_guard<std::mutex> lk(run->completion_mu);
        if (run->submission_closed) {
            throw std::logic_error("Orchestrator::close_run_submission: submission already closed");
        }
        run->submission_closed = true;
        if (run->active_tasks.load(std::memory_order_acquire) != 0) {
            run->phase.store(RunPhase::EXECUTING, std::memory_order_release);
        }
    }
    run->completion_cv.notify_all();
    finish_run_if_ready(run);
}

void Orchestrator::fail_run_submission(RunId run_id, std::exception_ptr error) {
    auto run = get_run(run_id);
    if (error) record_run_error(run_id, std::move(error));
    {
        std::lock_guard<std::mutex> runs_lk(runs_mu_);
        if (building_run_id_ != run_id) {
            throw std::logic_error("Orchestrator::fail_run_submission: run is not building");
        }
        building_run_id_ = INVALID_RUN_ID;
    }
    {
        std::lock_guard<std::mutex> lk(run->completion_mu);
        run->submission_failed = true;
        run->submission_closed = true;
        if (run->active_tasks.load(std::memory_order_acquire) != 0) {
            run->phase.store(RunPhase::EXECUTING, std::memory_order_release);
        }
    }
    run->completion_cv.notify_all();
    finish_run_if_ready(run);
}

void Orchestrator::wait_run_accepted(RunId run_id) {
    auto run = get_run(run_id);
    std::unique_lock<std::mutex> lk(run->completion_mu);
    run->completion_cv.wait(lk, [&run] {
        return acceptance_ready(run);
    });
}

bool Orchestrator::run_accepted(RunId run_id) const {
    auto run = get_run(run_id);
    std::lock_guard<std::mutex> lk(run->completion_mu);
    return acceptance_ready(run);
}

void Orchestrator::wait_run(RunId run_id) {
    auto run = get_run(run_id);
    std::exception_ptr error;
    {
        std::unique_lock<std::mutex> lk(run->completion_mu);
        run->completion_cv.wait(lk, [&run] {
            return is_terminal(run->phase.load(std::memory_order_acquire));
        });
        error = run->first_error;
    }
    if (error) std::rethrow_exception(error);
}

bool Orchestrator::wait_run_for(RunId run_id, double timeout_seconds) {
    auto run = get_run(run_id);
    std::exception_ptr error;
    {
        std::unique_lock<std::mutex> lk(run->completion_mu);
        if (!run->completion_cv.wait_for(lk, std::chrono::duration<double>(timeout_seconds), [&run] {
                return is_terminal(run->phase.load(std::memory_order_acquire));
            })) {
            return false;
        }
        error = run->first_error;
    }
    if (error) std::rethrow_exception(error);
    return true;
}

bool Orchestrator::run_done(RunId run_id) const {
    return is_terminal(get_run(run_id)->phase.load(std::memory_order_acquire));
}

bool Orchestrator::run_failed(RunId run_id) const {
    auto run = get_run(run_id);
    std::lock_guard<std::mutex> lk(run->completion_mu);
    return run->submission_failed || static_cast<bool>(run->first_error);
}

void Orchestrator::release_run(RunId run_id) {
    bool globally_quiescent = false;
    {
        std::lock_guard<std::mutex> lk(runs_mu_);
        auto it = runs_.find(run_id);
        if (it == runs_.end()) throw std::invalid_argument("Orchestrator::release_run: unknown run id");
        if (!is_terminal(it->second->phase.load(std::memory_order_acquire))) {
            throw std::logic_error("Orchestrator::release_run: run is not terminal");
        }
        runs_.erase(it);
        globally_quiescent = runs_.empty() && building_run_id_ == INVALID_RUN_ID;
    }

    if (!globally_quiescent || allocator_->active_count() != 0) return;
    if (sched_loop_mu_ != nullptr) {
        std::lock_guard<std::mutex> sched_lk(*sched_loop_mu_);
        allocator_->reset_to_empty();
    } else {
        allocator_->reset_to_empty();
    }
}

void Orchestrator::increment_run_tasks(RunId run_id) {
    get_run(run_id)->active_tasks.fetch_add(1, std::memory_order_relaxed);
}

void Orchestrator::decrement_run_tasks(RunId run_id) {
    auto run = get_run(run_id);
    int32_t remaining = run->active_tasks.fetch_sub(1, std::memory_order_acq_rel) - 1;
    if (remaining < 0) throw std::logic_error("Orchestrator: run task count underflow");
    if (remaining == 0) finish_run_if_ready(run);
}

void Orchestrator::increment_run_accepts(RunId run_id, int32_t count) {
    if (count <= 0) throw std::logic_error("Orchestrator: run acceptance count must be positive");
    get_run(run_id)->pending_accepts.fetch_add(count, std::memory_order_relaxed);
}

void Orchestrator::decrement_run_accepts(RunId run_id) {
    auto run = get_run(run_id);
    int32_t remaining = run->pending_accepts.fetch_sub(1, std::memory_order_acq_rel) - 1;
    if (remaining < 0) throw std::logic_error("Orchestrator: run acceptance count underflow");
    if (remaining == 0) run->completion_cv.notify_all();
}

void Orchestrator::record_run_error(RunId run_id, std::exception_ptr error) {
    if (!error) return;
    auto run = get_run(run_id);
    std::lock_guard<std::mutex> lk(run->completion_mu);
    if (!run->first_error) run->first_error = std::move(error);
}

void Orchestrator::report_task_error(TaskSlot slot, const std::string &message) {
    TaskSlotState &task = slot_state(slot);
    record_run_error(task.run_id, std::make_exception_ptr(std::runtime_error(message)));
}

void Orchestrator::mark_task_accepted(TaskSlot slot) {
    TaskSlotState &task = slot_state(slot);
    decrement_run_accepts(task.run_id);
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
    auto run = current_building_run();
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
    s.run_id = run->id;

    uint64_t ptr = reinterpret_cast<uint64_t>(ar.heap_ptr);
    if (ptr != 0) {
        TensorKey key = TensorKey::local_host(ptr);
        tensormap_->insert(run->id, key, ar.slot);
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

    increment_run_tasks(run->id);

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
    std::vector<int32_t> target_worker_ids{worker_id};
    std::vector<std::vector<int32_t>> worker_id_sets;
    if (!eligible_worker_ids.empty()) worker_id_sets = {eligible_worker_ids};
    std::vector<RemoteTaskArgsSidecar> sidecars;
    if (!remote_sidecar.tensors.empty() || !remote_sidecar.inline_payload.empty()) sidecars = {remote_sidecar};
    return submit_impl(
        WorkerType::NEXT_LEVEL, callable, config, {args}, std::move(target_worker_ids), std::move(worker_id_sets),
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
    std::vector<int32_t> target_worker_ids, std::vector<std::vector<int32_t>> eligible_worker_ids,
    std::vector<RemoteTaskArgsSidecar> remote_sidecars
) {
    auto run = current_building_run();
    if (args_list.empty()) throw std::invalid_argument("Orchestrator: args_list must not be empty");
    config.validate();
    validate_worker_eligibility(worker_type, args_list.size(), target_worker_ids, eligible_worker_ids);
    validate_remote_sidecars(args_list, remote_sidecars, eligible_worker_ids);

    {
        std::lock_guard<std::mutex> lk(run->completion_mu);
        if (run->first_error) std::rethrow_exception(run->first_error);
    }

    // --- Step 1: Atomically claim slot + auto-alloc any OUTPUT tensors that
    // arrived with a null data pointer. Both resources come from the same
    // merged allocator (Strict-2) so there is no partial-failure rollback
    // path.
    AllocResult ar = reserve_outputs_and_slot(args_list, remote_sidecars);
    if (ar.slot == INVALID_SLOT) {
        throw std::runtime_error("Orchestrator: allocator shutdown");
    }
    TaskSlot slot = ar.slot;

    TaskSlotState &s = slot_state(slot);
    s.reset();
    s.run_id = run->id;

    s.worker_type = worker_type;
    s.callable = callable;
    s.config = config;
    // --- Step 2: Walk tags → tensormap.lookup (deps) + tensormap.insert
    // (outputs). Must happen before we move args_list into the slot because
    // infer_deps reads tensor data pointers and tags from it.
    std::vector<TaskSlot> producers;
    infer_deps(slot, args_list, target_worker_ids, remote_sidecars, producers, s.output_keys);

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
    s.target_worker_ids = std::move(target_worker_ids);

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
    increment_run_tasks(run->id);
    increment_run_accepts(run->id, s.group_size());

    if (poisoned_by_failed_producer) {
        if (poison_message.empty()) poison_message = "producer task failed";
        s.failure_message = poison_message;
        record_run_error(run->id, std::make_exception_ptr(std::runtime_error(poison_message)));
        s.state.store(TaskState::FAILED, std::memory_order_release);
        std::vector<TaskSlot> fanin_producers = s.fanin_producers;
        try_consume(slot);
        for (TaskSlot prod : fanin_producers) {
            try_consume(prod);
        }
        return SubmitResult{slot};
    }

    // --- Step 6: If no live fanins → READY and route by final placement ---
    if (live_fanins == 0) {
        s.state.store(TaskState::READY, std::memory_order_release);
        enqueue_ready(slot);
        if (ready_notify_cb_) ready_notify_cb_();
    } else {
        s.state.store(TaskState::PENDING, std::memory_order_release);
    }

    return SubmitResult{slot};
}

void Orchestrator::enqueue_ready(TaskSlot slot) {
    TaskSlotState &s = slot_state(slot);
    if (s.worker_type == WorkerType::NEXT_LEVEL) {
        if (ready_next_level_queues_ == nullptr)
            throw std::runtime_error("Orchestrator::enqueue_ready: NEXT_LEVEL queues are not initialized");
        if (s.is_group()) {
            ready_next_level_queues_->push_group(slot);
        } else {
            ready_next_level_queues_->push_single(s.target_worker_id(0), slot);
        }
        return;
    }
    ready_sub_queue_->push(slot);
}

void Orchestrator::validate_worker_eligibility(
    WorkerType worker_type, size_t args_count, const std::vector<int32_t> &target_worker_ids,
    const std::vector<std::vector<int32_t>> &eligible_worker_ids
) const {
    if (worker_type == WorkerType::SUB) {
        if (!target_worker_ids.empty() || !eligible_worker_ids.empty()) {
            throw std::invalid_argument("Orchestrator: SUB tasks do not accept worker-selection metadata");
        }
        return;
    }

    if (target_worker_ids.size() != args_count) {
        throw std::invalid_argument(
            "Orchestrator: NEXT_LEVEL target count " + std::to_string(target_worker_ids.size()) +
            " does not match args length " + std::to_string(args_count)
        );
    }
    if (!eligible_worker_ids.empty() && eligible_worker_ids.size() != args_count) {
        throw std::invalid_argument(
            "Orchestrator: eligible worker-id set length " + std::to_string(eligible_worker_ids.size()) +
            " does not match args length " + std::to_string(args_count)
        );
    }

    std::unordered_set<int32_t> unique_targets;
    for (int32_t worker_id : target_worker_ids) {
        if (worker_id < 0) {
            throw std::invalid_argument("Orchestrator: NEXT_LEVEL worker id must be non-negative");
        }
        if (!unique_targets.insert(worker_id).second) {
            throw std::invalid_argument("Orchestrator: duplicate NEXT_LEVEL worker id " + std::to_string(worker_id));
        }
    }

    const std::vector<int32_t> empty_eligible;
    for (size_t i = 0; i < args_count; ++i) {
        const auto &eligible = eligible_worker_ids.empty() ? empty_eligible : eligible_worker_ids[i];
        if (!eligible_worker_ids.empty() && eligible.empty()) {
            throw std::invalid_argument(
                "Orchestrator: final eligible worker-id set is empty for member " + std::to_string(i)
            );
        }
        if (manager_ != nullptr) {
            for (int32_t worker_id : eligible) {
                if (manager_->get_worker_by_id(WorkerType::NEXT_LEVEL, worker_id) == nullptr) {
                    throw std::invalid_argument(
                        "Orchestrator: eligible worker-id " + std::to_string(worker_id) + " is not a registered worker"
                    );
                }
            }
            if (manager_->get_worker_by_id(WorkerType::NEXT_LEVEL, target_worker_ids[i]) == nullptr) {
                throw std::invalid_argument(
                    "Orchestrator: target worker " + std::to_string(target_worker_ids[i]) +
                    " is not a registered worker"
                );
            }
        }

        if (!eligible_worker_ids.empty()) {
            bool allowed = false;
            for (int32_t id : eligible) {
                if (id == target_worker_ids[i]) {
                    allowed = true;
                    break;
                }
            }
            if (!allowed) {
                throw std::invalid_argument(
                    "Orchestrator: target worker " + std::to_string(target_worker_ids[i]) +
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
    TaskSlot slot, const std::vector<TaskArgs> &args_list, const std::vector<int32_t> &target_worker_ids,
    const std::vector<RemoteTaskArgsSidecar> &remote_sidecars, std::vector<TaskSlot> &producers,
    std::vector<TensorKey> &output_keys
) {
    RunId run_id = slot_state(slot).run_id;
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
        int32_t worker_id = (g < target_worker_ids.size()) ? target_worker_ids[g] : -1;
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
                TaskSlot prod = tensormap_->lookup(run_id, key);
                if (prod != INVALID_SLOT) add_unique_producer(prod);
                break;
            }
            case TensorArgType::INOUT: {
                TaskSlot prod = tensormap_->lookup(run_id, key);
                if (prod != INVALID_SLOT) add_unique_producer(prod);
                tensormap_->insert(run_id, key, slot);
                output_keys.push_back(key);
                break;
            }
            case TensorArgType::OUTPUT:
            case TensorArgType::OUTPUT_EXISTING: {
                tensormap_->insert(run_id, key, slot);
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

void Orchestrator::scope_begin() {
    (void)current_building_run();
    scope_->scope_begin();
}

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

    RunId run_id = s.run_id;
    tensormap_->erase_task_outputs(run_id, s.output_keys);

    // HeapRing-owned OUTPUT slabs are reclaimed implicitly when the allocator
    // advances last_alive past this slot — no per-slot munmap needed.
    allocator_->release(slot);

    decrement_run_tasks(run_id);
    return true;
}
