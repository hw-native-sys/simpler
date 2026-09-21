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
 * Orchestrator — DAG builder.
 *
 * Public API (called by the user's orch fn during Worker::run):
 *   - submit_next_level(CallableIdentity, TaskArgs, CallConfig, worker_id)
 *   - submit_next_level_group(CallableIdentity, vector<TaskArgs>, CallConfig, worker_ids)
 *   - submit_sub(CallableIdentity, TaskArgs)
 *   - submit_sub_group(CallableIdentity, vector<TaskArgs>)
 *   - alloc(shape, dtype, identity) — runtime-owned intermediate buffer (returns its VA)
 *
 * Each TaskArgs carries per-tensor TensorArgType tags. The Orchestrator
 * walks those tags to drive dependency inference and — for OUTPUT tags with
 * a null data pointer — automatically assigns a slab from the HeapRing
 * (see docs/orchestrator.md §8b).
 *
 * Internal:
 *   - scope_begin / scope_end / drain — invoked only by Worker::run
 */

#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../task_interface/call_config.h"
#include "../task_interface/data_type.h"
#include "../task_interface/task_args_wire.h"
#include "../task_interface/tensor.h"
#include "../worker/pipeline_slot_pool.h"
#include "../worker/device_memory_info.h"
#include "ring.h"
#include "scope.h"
#include "tensormap.h"
#include "types.h"

class WorkerManager;

// ---------------------------------------------------------------------------
// TaskHandle — run-scoped producer handle
// ---------------------------------------------------------------------------
//
// Downstream consumers reference outputs by their own tensor pointers (the
// tensors live in the HeapRing allocated by the Worker), and tensormap.lookup
// finds the producer slot from the data pointer. No outputs[] field needed.
// The Python facade exposes this as an opaque handle for TaskArgs.add_dep()
// and TaskArgs.add_dep_wait().

using SubmitResult = TaskHandle;

// Deterministic seams for exception-safety unit tests. Production workers
// leave the callback unset.
enum class OrchestratorTestPoint : int32_t {
    SCOPE_REGISTERED = 0,
    PRODUCER_FORWARD_EDGE_PUBLISHED = 1,
    FAILURE_FANIN_SNAPSHOT = 2,
    GROUP_MEMBER_STATES_PREPARED = 3,
    ALLOC_RUN_SLOT_REGISTERING = 4,
    ALLOC_OUTPUT_KEY_PREPARED = 5,
    SUBMIT_RUN_SLOT_REGISTERING = 6,
    SUBMIT_OUTPUT_KEY_PREPARED = 7,
    BEGIN_RUN_MAP_PUBLISHED = 8,
    BEGIN_RUN_FIFO_PUBLISHED = 9,
};

// ---------------------------------------------------------------------------
// Orchestrator
// ---------------------------------------------------------------------------

class Orchestrator {
public:
    void init(
        TensorMap *tensormap, Ring *allocator, Scope *scope, ReadyQueue *ready_sub_queue,
        NextLevelReadyQueues *ready_next_level_queues, WorkerManager *manager = nullptr,
        std::function<void()> ready_notify_cb = {}
    );

    // Allocate an intermediate buffer from the Worker's HeapRing (MAP_SHARED, visible to forked child
    // workers) and return its VA. Registered in the tensormap under the identity's canonical hash
    // (matching how infer_deps keys a Tensor), so the caller can wrap the VA as a FORK_SHM
    // Buffer carrying `identity` and a ref over it dependency-wires to this slot. Backs
    // Worker.alloc_shared_tensor / Orchestrator.alloc (Python).
    //
    // Lifetime: aligned with a synthetic task slot. The buffer is reclaimed (FIFO, via last_alive) once
    // every downstream consumer tagging the ref has reached CONSUMED and scope_end has released the
    // scope ref.
    uint64_t alloc(const std::vector<uint32_t> &shape, DataType dtype, const CanonicalIdentity &identity);

    // Bytes currently committed by a specific next-level worker's device
    // allocator. Thread-safe: can be called from the orch thread while the
    // target worker is running a task (MemoryAllocator is mutex-protected).
    uint64_t committed_device_memory(int worker_id);
    DeviceMemoryInfo device_memory_info(int worker_id);

    // Submit a NEXT_LEVEL task. `callable` is the stable identity returned
    // by Worker.register(); the child resolves its digest to a private slot.
    // Tags inside `args` drive dependency inference; OUTPUT tensors with
    // null data are auto-allocated from the HeapRing.
    // `worker_id`: exact stable NEXT_LEVEL worker id that runs this task.
    SubmitResult submit_next_level(
        const CallableIdentity &callable, const TaskArgs &args, const CallConfig &config, int32_t worker_id,
        const std::vector<int32_t> &eligible_worker_ids = {}, const RemoteTaskArgsSidecar &remote_sidecar = {}
    );

    // Submit a group of NEXT_LEVEL tasks: N args -> N worker selections, 1 DAG node.
    // `worker_ids`: one exact stable NEXT_LEVEL worker id per member.
    SubmitResult submit_next_level_group(
        const CallableIdentity &callable, const std::vector<TaskArgs> &args_list, const CallConfig &config,
        const std::vector<int32_t> &worker_ids, const std::vector<std::vector<int32_t>> &eligible_worker_ids = {},
        const std::vector<RemoteTaskArgsSidecar> &remote_sidecars = {}
    );

    // Submit a SUB task by registered callable identity.
    SubmitResult submit_sub(const CallableIdentity &callable, const TaskArgs &args);

    // Submit a group of SUB tasks: N args -> N workers, 1 DAG node.
    SubmitResult submit_sub_group(const CallableIdentity &callable, const std::vector<TaskArgs> &args_list);

    // Only the calling orchestration thread builds a run at a time.
    RunId begin_run();

    /**
     * Set how many runs may hold a native pipeline-slot lease at once
     * (`depth`) and how many non-terminal logical runs may be admitted
     * (`pending_depth`).
     *
     * The two are separate budgets. `depth` bounds a device resource: the
     * leases handed to the active run and to the first eligible preparable
     * successor. `pending_depth` bounds only the count of entries in the
     * admission FIFO, which is host bookkeeping — it is not a byte budget, and
     * it does not bound terminal runs whose `RunState` a caller has not yet
     * released.
     *
     * `pending_depth == 0` derives the FIFO bound from `depth`, which is the
     * behaviour every caller had before the two budgets were separable. A
     * positive value below `depth` is legal and simply leaves the successor
     * role unfillable, because that role needs a second FIFO entry to exist.
     *
     * `launch_depth` bounds how many runs may have their device work launched
     * at once. One — the default — is the serial behaviour: a successor's work
     * reaches a child only once its predecessor is terminal. Two authorizes the
     * staged successor of a run whose every dispatch has already reached an
     * endpoint outcome, so its work reaches the device while the predecessor is
     * still executing. It cannot exceed `depth`, since a launched run holds a
     * pipeline lease for its whole lifetime.
     */
    void configure_pipeline_depth(uint32_t depth, uint32_t pending_depth = 0, uint32_t launch_depth = 1);
    void close_run_submission(RunId run_id);
    void fail_run_submission(RunId run_id, std::exception_ptr error = nullptr);
    void wait_run_accepted(RunId run_id);
    bool run_accepted(RunId run_id) const;
    void wait_run(RunId run_id);
    bool wait_run_for(RunId run_id, double timeout_seconds);
    bool run_done(RunId run_id) const;
    bool run_failed(RunId run_id) const;
    bool can_dispatch_run(RunId run_id) const;

    /**
     * The run the scheduler may dispatch from, or INVALID_RUN_ID when none is.
     *
     * Stricter than `active_run_id()`: the FIFO head must also be EXECUTING and
     * still own its pipeline lease, so a head whose lease was released cannot
     * have further work handed down under it.
     */
    RunId dispatchable_run_id() const;

    /**
     * Block until `run_id` holds the whole-run FIFO head, or is terminal.
     *
     * Direct device control bypasses the ready-queue FIFO, so a caller acting
     * on behalf of a run must wait here before touching a child. The run is
     * explicit: only the thread actually inside that run's graph callback may
     * be ordered against it. INVALID_RUN_ID returns immediately.
     */
    void await_run_admission(RunId run_id);
    RunId active_run_id() const;
    RunId preparable_run_id() const;

    /**
     * The staged successor authorized to launch its device work while the
     * active run is still executing, or INVALID_RUN_ID when there is none.
     *
     * Authorization needs all of: a configured launch depth above one; an
     * active, dispatchable, error-free FIFO head whose every dispatch has
     * reached an endpoint outcome — so no further dispatch of it can be issued;
     * and a second FIFO entry that is PREPARED and owns a pipeline lease.
     *
     * "Every dispatch reached an outcome" is `submission_closed` *and* a zero
     * acceptance count *and* not terminal. Neither of the first two alone says
     * it: the flag closes only the build declaration, and the count reads zero
     * before anything is published. Terminal is excluded because a failed head
     * must not open the gate.
     */
    RunId early_launch_run_id() const;

    /**
     * A run's dispatch reached a child and was staged there.
     *
     * The reverse order of the early-launch predicate's two halves: the active
     * head may already have had every dispatch accepted when the successor's
     * frame is staged, and nothing else announces that. Non-throwing by
     * contract — it is reached from the worker progress driver.
     */
    void notify_run_staged(RunId run_id);
    void release_run(RunId run_id);

    // Open a nested scope. Every task submitted between this call and the
    // matching `scope_end()` picks a heap ring based on the current scope
    // depth (`min(depth, MAX_RING_DEPTH - 1)`) so its slab reclaims
    // independently of the outer scope's slabs (Strict-1). `Worker::run`
    // opens the outermost scope automatically; user orch fns may nest up
    // to `MAX_SCOPE_DEPTH` additional scopes.
    //
    // Non-blocking: `scope_end` walks the scope's tasks and releases one
    // ref per task, returning immediately. Actual CONSUMED transitions
    // happen asynchronously as each task's consumer count reaches
    // threshold (mirrors the chip runtime's `rt_scope_end`). The owning run fence
    // provides the synchronous completion boundary.
    void scope_begin();
    void scope_end();

    // Wire the Scheduler's loop mutex so release_run() can safely perform an
    // optional allocator compaction when the whole worker is quiescent.
    void set_scheduler_loop_mutex(std::mutex *m) { sched_loop_mu_ = m; }

    // Attach a scheduler/endpoint failure to the task's originating run.
    void report_task_error(TaskSlot slot, const std::string &message);

    // Called once for each dispatched group member after its endpoint has
    // accepted the launch, or conservatively at endpoint completion.
    void mark_task_accepted(TaskSlot slot);

    // Called by Scheduler (via Worker) when a task becomes CONSUMED:
    // erases TensorMap entries, releases the allocator slot (and implicitly
    // the slot's heap slab via last_alive).
    // Returns true iff this call performed the COMPLETED/FAILED -> CONSUMED transition.
    // Idempotent: concurrent callers (release_ref vs try_consume) race on a
    // CAS — only the winner returns true and runs cleanup; losers return false.
    bool on_consumed(TaskSlot slot);

    // Route a slot whose state is already READY to the queue that owns it.
    // Scheduler uses the same path after releasing the final dependency.
    void enqueue_ready(TaskSlot slot);

    void set_test_hook(std::function<void(OrchestratorTestPoint)> hook) { test_hook_ = std::move(hook); }

    // Deterministic observation seam for tests that must issue another submit
    // only after an asynchronous task failure has reached the current run.
    bool current_building_run_failed_for_test() const;
    size_t begin_run_waiter_count_for_test() const;

private:
    TensorMap *tensormap_ = nullptr;
    Ring *allocator_ = nullptr;
    Scope *scope_ = nullptr;
    WorkerManager *manager_ = nullptr;
    std::function<void()> ready_notify_cb_;
    std::function<void(OrchestratorTestPoint)> test_hook_;
    ReadyQueue *ready_sub_queue_ = nullptr;
    NextLevelReadyQueues *ready_next_level_queues_ = nullptr;

    mutable std::mutex runs_mu_;
    std::condition_variable runs_cv_;
    std::unordered_map<RunId, std::shared_ptr<RunState>> runs_;
    std::deque<RunId> run_fifo_;
    PipelineSlotPool pipeline_slots_{PTO_PIPELINE_MAX_DEPTH};
    uint32_t admission_depth_{PTO_PIPELINE_MAX_DEPTH};
    // How many non-terminal runs `run_fifo_` may hold. Derived from
    // `admission_depth_` unless a caller configured it, so the default admits
    // exactly the runs a lease was available for before the two budgets split.
    uint32_t pending_run_limit_{PTO_PIPELINE_MAX_DEPTH};
    // How many runs may have their device work launched at once. One is the
    // serial behaviour, and at one nothing below reads
    // `early_launch_run_id_`, which stays invalid.
    uint32_t launch_depth_{1};
    // The staged successor currently authorized to launch early, latched so
    // the transition into that state can be announced once rather than
    // re-derived by every reader.
    RunId early_launch_run_id_{INVALID_RUN_ID};
    RunId next_run_id_{1};
    RunId building_run_id_{INVALID_RUN_ID};
    RunId active_run_id_{INVALID_RUN_ID};
    size_t begin_run_waiters_{0};

    // Scheduler's loop mutex (not owned). Held across optional quiescent
    // compaction so the scheduler cannot retain a slot pointer being removed.
    std::mutex *sched_loop_mu_{nullptr};

    // Returns nullptr for an unknown id. Bookkeeping reached from the
    // scheduler and worker threads uses this instead of `get_run` — an
    // exception escaping those threads terminates the process.
    std::shared_ptr<RunState> find_run(RunId run_id) const;
    std::shared_ptr<RunState> get_run(RunId run_id) const;
    std::shared_ptr<RunState> current_building_run() const;
    void finish_run_if_ready(const std::shared_ptr<RunState> &run);
    static bool is_terminal(RunPhase phase);
    static bool acceptance_ready(const std::shared_ptr<RunState> &run);
    // Callers hold runs_mu_.
    bool quiescent_locked() const;
    bool dispatchable_locked(RunId run_id) const;

    /**
     * Give `run` a native pipeline-slot lease if it does not already hold one,
     * reporting whether it holds one on return. `*assigned` is set only when
     * this call is the one that acquired it.
     *
     * Non-blocking: a depleted pool leaves the run lease-less and the caller
     * unblocked. Blocking here would deadlock, because this runs under
     * `runs_mu_` and that is the mutex a retiring run needs to return the very
     * slot this call would wait for.
     */
    bool acquire_lease_locked(const std::shared_ptr<RunState> &run, bool *assigned = nullptr);

    /**
     * Hand leases to the two roles that may hold one: the FIFO head and the
     * first eligible preparable successor. Returns whether any lease was newly
     * assigned, which is what a caller uses to decide it owes the scheduler a
     * wake.
     */
    bool refresh_leases_locked();

    /**
     * Whether every dispatch this run will ever issue has reached an endpoint
     * outcome, so no further dispatch of it can be issued.
     *
     * One decrement of the acceptance count certifies that *that* dispatch
     * reached an outcome — an ACCEPTED progress, or a terminal that never saw
     * one — and not that it succeeded, which is the property a successor needs.
     * The count is complete because the increment happens synchronously inside
     * `submit_impl`, in the same step that publishes the slot, and
     * `close_run_submission` runs afterwards on the same building thread.
     *
     * Takes the run's `completion_mu` under `runs_mu_`, the order
     * `acquire_lease_locked` already establishes.
     */
    static bool all_dispatches_accepted(const std::shared_ptr<RunState> &run);

    /**
     * Re-derive `early_launch_run_id_`, reporting whether this call is the one
     * that authorized a run. The caller owes the scheduler a wake on true,
     * issued outside `runs_mu_`.
     *
     * Called at every transition that can change the answer: the last
     * acceptance arriving, submission closing, a lease refresh at retirement,
     * and a head becoming active. A staged run whose predecessor was already
     * fully accepted is the reverse order, and reaches this through the
     * endpoint's staging event.
     */
    bool refresh_early_launch_locked();
    void activate_fifo_head();
    void retire_terminal_run(const std::shared_ptr<RunState> &run);
    void cancel_unstarted_run(const std::shared_ptr<RunState> &run, const std::string &message);
    void register_run_slot(const std::shared_ptr<RunState> &run, TaskSlot slot);
    void clear_run_ready_queues(RunId run_id);
    void compact_if_quiescent();
    void decrement_run_tasks(RunId run_id);
    void decrement_run_accepts(RunId run_id);
    static void record_run_error(const std::shared_ptr<RunState> &run, std::exception_ptr error);
    void record_run_error(RunId run_id, std::exception_ptr error);

    // Slot state lives in the Ring; the pointer stays stable for the
    // slot's lifetime. Throws if the id is out of range — callers that
    // hold a recently-allocated slot id should always get a valid pointer.
    TaskSlotState &slot_state(TaskSlot s);

    // Shared submit machinery. Takes `args_list` by value so the Orchestrator
    // can patch `tensor.data` on OUTPUT tensors flagged for auto-allocation.
    SubmitResult submit_impl(
        WorkerType worker_type, const CallableIdentity &callable, const CallConfig &config,
        std::vector<TaskArgs> args_list, std::vector<int32_t> target_worker_ids = {},
        std::vector<std::vector<int32_t>> eligible_worker_ids = {},
        std::vector<RemoteTaskArgsSidecar> remote_sidecars = {}
    );

    // Size, in aligned bytes, an OUTPUT tensor should occupy in the HeapRing.
    static uint64_t output_alloc_bytes(const ChipTensor &t);

    // Rewrite any OUTPUT tensors with a null data pointer to point into a
    // freshly-allocated HeapRing slab. Returns the total aligned byte span
    // consumed, and populates `slot` / `heap_ptr` / `heap_end_offset` via the
    // output params (reused for book-keeping on the slot state). Throws on
    // back-pressure timeout.
    AllocResult reserve_outputs_and_slot(
        std::vector<TaskArgs> &args_list, const std::vector<RemoteTaskArgsSidecar> &remote_sidecars
    );

    struct ProducerDependency {
        TaskSlot slot{INVALID_SLOT};
        bool retain{false};
    };

    // Walk the tags of each TaskArgs in `args_list`, accumulating producer
    // edges (explicit wait/retain deps plus retained INPUT/INOUT deps) and
    // registering outputs in the tensormap (for
    // OUTPUT/INOUT/OUTPUT_EXISTING tags). NO_DEP tags are skipped.
    // `target_worker_ids` maps NEXT_LEVEL args_list[i] to its exact worker for
    // TensorKey construction. It is empty for SUB tasks.
    void infer_deps(
        TaskSlot slot, const std::vector<TaskArgs> &args_list, const std::vector<int32_t> &target_worker_ids,
        const std::vector<RemoteTaskArgsSidecar> &remote_sidecars, std::vector<ProducerDependency> &producers,
        std::vector<TensorKey> &output_keys
    );
    void validate_explicit_deps(RunId run_id, const std::vector<TaskArgs> &args_list) const;
    void validate_worker_eligibility(
        WorkerType worker_type, size_t args_count, const std::vector<int32_t> &target_worker_ids,
        const std::vector<std::vector<int32_t>> &eligible_worker_ids
    ) const;
    void validate_remote_sidecars(
        const std::vector<TaskArgs> &args_list, const std::vector<RemoteTaskArgsSidecar> &remote_sidecars,
        const std::vector<std::vector<int32_t>> &eligible_worker_ids
    ) const;

    // Release one fanout reference on 'slot'.
    // If all references are released → transition to CONSUMED.
    void release_ref(TaskSlot slot);
    void try_consume(TaskSlot slot);
};
