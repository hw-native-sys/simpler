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
 * Onboard host `DeviceRunnerBase` — common base class for a2a3 and a5
 * onboard `DeviceRunner`s.
 *
 * This module owns the host-side state and methods that are identical
 * between the two onboard arches today:
 *   - The `MemoryAllocator` and the three `DeviceArena`s (gm heap, shared memory
 *     SM, runtime arena) backing the per-Worker pooled regions.
 *   - The trivial tensor-memory wrappers (`allocate_tensor`,
 *     `free_tensor`, `copy_*_device`).
 *   - The arena-pool accessors (`acquire_pooled_gm_heap`, etc.).
 *   - Device lifecycle: `bind_current_thread`, `attach_current_thread`,
 *     `adopt_borrowed_device`, `configure_aicore_op_timeout`,
 *     `ensure_device_initialized`,
 *     `ensure_binaries_loaded`, persistent AICPU/AICore streams,
 *     dispatcher/executor bytes, `LoadAicpuOp`, `KernelArgsHelper`.
 *   - block_dim resolution: `query_max_block_dim`, `resolve_block_dim`.
 *   - Debug: `print_handshake_results`, `create_thread`.
 *
 * Subclasses (`{a2a3,a5}::DeviceRunner`) add arch-specific state
 * (callable registry, profiling collectors, ACL/HCCL plumbing on a2a3,
 * `enable_*` flags) and the divergent methods (`prepare_execution`,
 * `launch_execution`, `poll_execution`, `drain_execution`, `finalize`,
 * `setup_static_arena`, the kernel launch /
 * chip-callable upload, the per-callable registration helpers, and the
 * per-diagnostic `init_*`).
 */

#pragma once

#include <runtime/rt.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "arg_direction.h"
#include "call_config.h"
#include "callable.h"
#include "common/device_phase.h"
#include "common/device_run_result.h"
#include "common/dma_workspace.h"
#include "common/chip_swimlane_profiling.h"
#include "utils/device_arena.h"
#include "utils/retained_scheduler_storage.h"
#include "device_phase_capture.h"
#include "device_runner_helpers.h"
#include "aicpu_loader/host/load_aicpu_op.h"
#include "host/arena_replacement_transaction.h"
#include "host/chip_swimlane_collector.h"
#include "host/device_fault_monitor.h"
#include "host/device_health_state.h"
#include "host/dfx_run_config.h"
#include "host/caller_device_buffers.h"
#include "host/child_memory_host_view.h"
#include "host/execution_mode_latch.h"
#include "host/host_phase_records.h"
#include "host/host_phase_run_state.h"
#include "host/kernel_entry_validation.h"
#include "host/kernel_execution_state.h"
#include "host/memory_allocator.h"
#include "host/workspace_manager.h"
#include "host/pmu_collector.h"
#include "host/queued_stream_waits.h"
#include "host/run_boundary_marks.h"
#include "host/run_evidence_retention.h"
#include "host/run_completion_fence.h"
#include "host/teardown_recorder.h"
#include "host/run_outcome_decision.h"
#include "host/runtime_timeout_config.h"
#include "host/scope_stats_collector.h"
#include "host/args_dump_collector.h"
#include "prepare_callable_common.h"
#include "runtime_c_api.h"
#include "native_run_execution.h"
#include "kernel_persistent_args.h"

struct HostApi;  // common/host_api.h — fwd-declared to keep task_interface headers out

/**
 * Common base class for both a2a3 and a5 onboard `DeviceRunner`s.
 *
 * Ctor + dtor are `protected` so this class can only be used as a base;
 * direct instantiation and `delete` through a base pointer are both
 * compile errors. The arch subclass's `DeviceRunner` is what
 * `destroy_device_context` sees, so the non-virtual `~DeviceRunnerBase`
 * is safe — it never runs as a virtual base destructor.
 */
class DeviceRunnerBase {
    // #2267's late-read retention probe reads this run's completion fence and
    // result region directly, without the slot-gated poll/drain entries that
    // a successor's launch takes over. Fixture access only — nothing in the
    // product reaches these through the peer.
    friend class RunRetentionProbePeer;

public:
    // Public virtual dtor so the shared c_api can `delete` a polymorphic
    // `DeviceRunnerBase *` (the `destroy_device_context` entrypoint). Each
    // arch's `DeviceRunner` defaults this through the compiler-generated dtor.
    virtual ~DeviceRunnerBase() = default;
    DeviceRunnerBase(const DeviceRunnerBase &) = delete;
    DeviceRunnerBase &operator=(const DeviceRunnerBase &) = delete;
    DeviceRunnerBase(DeviceRunnerBase &&) = delete;
    DeviceRunnerBase &operator=(DeviceRunnerBase &&) = delete;

    /**
     * Claim the runner for one native execution. The opaque owner and
     * runner-owned timing and diagnostic state remain exclusive through
     * validation/finalize.
     *
     * `join` is null for every ordinary launch, and then the claim is refused
     * while any other run holds it. A non-null join is the caller's statement
     * that this run has been ordered behind the named predecessor on the device;
     * it is admitted only while that predecessor is the newest claim holder and
     * its recorded identity matches, so a second launched run is reachable
     * exclusively through the joined path.
     */
    bool try_acquire_native_run(
        const void *owner, const NativeRunIdentity &identity, LaunchPermit *permit, const NativeRunJoin *join = nullptr
    );
    void release_native_run(const void *owner);
    bool native_run_active() const;
    bool native_run_owned_by(const void *owner) const;
    /** How many runs currently hold the native execution claim. */
    size_t native_run_claim_count() const;
    /** How many stream waits queued on a run completion boundary are live. */
    size_t queued_boundary_wait_count() const;
    /**
     * Whether the predecessor a join names still has its whole-operator
     * boundary unfired, as observed now. Reports the query's own failure rather
     * than folding it into the answer.
     */
    int predecessor_boundary_unfired(const NativeRunJoin &join, bool *unfired) const;

    /**
     * Record this run's passive device-timestamp marker for one stream position.
     *
     * Adds no stream ordering and no device wait — see host/run_boundary_marks.h for the two
     * positions, why each latches its own recording run, and why a record failure is remembered
     * against the position instead of failing the run.
     */
    int mark_run_boundary(RunBoundaryMarks::Position position, const NativeRunIdentity &identity, void *stream);

    /**
     * What one position's marker says for one run, or unavailable with the reason.
     *
     * Call once that run's own completion has been established, which is where every caller reads
     * it: retrieval establishes the marker event's own completion first, a host-blocking step
     * bounded by the configured stream-synchronize timeout. Whether the reading belongs to the run
     * that asked is `RunBoundaryMarks`' decision, and it answers unavailable when it cannot tell.
     * Not `const` because a read advances the attribution state that keeps one run's reading from
     * being returned as another's.
     */
    RunBoundaryMarks::Mark run_boundary_mark(RunBoundaryMarks::Position position, const NativeRunIdentity &identity);

    /**
     * Whether every stream that could have recorded a boundary marker is proven retired.
     *
     * Asked at teardown, immediately before the markers would be destroyed: a stream that kept
     * its handle because its own destroy failed may still hold a queued record naming one of
     * them, and destroying an event in that state is the one thing the markers must not do. A
     * subclass that records markers answers from the pair it recorded them on; one that records
     * none has nothing to wait for. This is not about the bootstrap stream pair, which records no
     * markers and is destroyed later in the same teardown.
     */
    virtual bool marker_recording_streams_retired() const { return true; }

    /**
     * One successor's ordering observation, belonging to an exact pair.
     *
     * Both identities are carried because the claim is about a pair, and
     * `observed` is separate from `predecessor_unfired` so a failed query reads
     * as a failed query rather than as an absence of ordering.
     */
    struct JoinedLaunchRecord {
        NativeRunIdentity successor{};
        NativeRunIdentity predecessor{};
        /** The query returned an answer. False means `query_rc` says why not. */
        bool observed{false};
        /** The predecessor's whole-operator boundary had not fired. */
        bool predecessor_unfired{false};
        int query_rc{0};
    };

    /**
     * Record what the predecessor's whole-operator boundary read *after* this
     * successor's native submission completed successfully, and return it.
     *
     * This ordering is the whole point. A query taken before the submission
     * proves nothing — the boundary may fire in the window between the query
     * and the submission returning — whereas a not-ready read taken after a
     * submission that has already succeeded places the completed enqueue
     * strictly before the predecessor's completion. Only the not-ready case is
     * evidence; a fired boundary means the submission and the completion raced
     * and nothing is claimed, and a failed query is recorded as a failed query.
     *
     * Safe at that point, and only at that point: the successor holds a
     * committed reference on that boundary, so the event is alive and the
     * predecessor's fence cannot have retired the identity out from under the
     * read.
     */
    JoinedLaunchRecord note_joined_launch(const NativeRunJoin &join, const NativeRunIdentity &successor);
    /**
     * Whether that run's AICPU completion boundary is known to cover its whole
     * operator, which is what makes it something another run can be ordered
     * behind. False for every run launched without the construction, and false
     * once the run that published it ends.
     */
    bool has_whole_operator_boundary(const NativeRunIdentity &identity) const;

    /**
     * Reserve caller-owned native-run storage before binding starts. A
     * concurrent reservation is admitted only while the first reservation
     * owns the execution claim and selects a distinct pipeline slot. A backend
     * that shares an arena bank must reject or defer incompatible preparation
     * before mutating that bank.
     */
    bool try_reserve_native_run(
        const void *owner, uint32_t pipeline_slot, uint32_t arena_bank, bool allow_prepared_successor
    );
    void release_native_run_reservation(const void *owner);
    bool native_runs_outstanding() const;

    /**
     * Whether a live reservation other than `owner`'s holds `arena_bank`.
     *
     * Two runs collide over the pooled device regions only when the slot lease
     * hands them the same bank, and that is a property of the contract rather
     * than of the depth: `pipeline_resource_slot` gives every slot its own bank
     * for a `HOST_PER_RUN` arena and bank 0 to all of them for a shared one. So
     * a successor whose bank differs cannot reach its predecessor's regions —
     * `setup_static_arena` only ever touches the bank it is given — while one
     * that shares a bank can grow or release regions the predecessor is
     * executing against. The prepare gate uses this to decide whether the
     * runtime's compatibility probe has to answer before the successor may
     * prepare concurrently.
     */
    bool arena_bank_shared_with_other_run(const void *owner, uint32_t arena_bank) const;

    /**
     * Committed GM heap base of one arena bank, or 0 while that bank has never
     * been committed. Two banks that have both served a run hold distinct
     * device allocations; tests read this to prove the depth-two split is real
     * rather than two names for one region.
     */
    uint64_t arena_bank_gm_heap_base(uint32_t bank_id) const;

    /**
     * Retained temporary-buffer address held for one pipeline slot, or 0 while
     * that slot holds none. Two slots that have both staged arguments hold
     * distinct buffers; tests read this to prove the split is real.
     */
    uint64_t retained_temp_addr(uint32_t slot_id) const;

    /**
     * This context's execution identity, latched once by whichever init entry
     * constructs it. Every kernel-mode guard on the ACL-lifecycle and arena
     * paths keys on is_kernel(); `attach_current_thread` refuses outright on a
     * kernel latch, which is what keeps the program-mode entries and the
     * per-thread device bind off a borrowed device.
     */
    ExecutionModeLatch &execution_mode_latch() { return execution_mode_latch_; }

    /** Context-lifetime streams and events, live only in kernel mode. */
    KernelExecutionState &kernel_execution_state() { return kernel_exec_state_; }

    /**
     * Whether any kernel-context owner still holds a device resource: the
     * streams and events in `KernelExecutionState`, the argument blocks in
     * `PersistentKernelArgs`, the loaded AICPU binary in `LoadAicpuOp`, or a
     * retained callable image in `chip_callable_buffers_`. The destruction
     * guard reads this aggregate, so a close that succeeded for some owners and
     * failed for another is not a close.
     *
     * False on a program context: `chip_callable_buffers_` and the loader are
     * shared with program registration, and a program context resets its
     * device at finalize, which ends the generation its addresses and handles
     * belonged to.
     */
    bool kernel_resources_live() const {
        if (!execution_mode_latch_.is_kernel()) return false;
        return kernel_exec_state_.has_live_resources() || persistent_args_.has_live_resources() ||
               load_aicpu_op_.has_live_resources() || !chip_callable_buffers_.empty();
    }

    /**
     * Whether a failed release of a callable image keeps its map entry as the
     * retained owner of the block.
     *
     * True only on a kernel context, the only one that can act on such an
     * entry: `kernel_resources_live()` refuses destruction while one exists,
     * and `finalize_common_impl` stops before `MemoryAllocator::finalize()` so
     * an explicit close retries the free. A program context erases the entry
     * even when the free fails — it resets its device at finalize, so a
     * retained entry would outlive the generation its address belonged to and
     * still answer the next generation's dedup lookup.
     */
    bool retains_failed_callable_release() const { return execution_mode_latch_.is_kernel(); }

    int init_kernel_context(int device_id);
    int prepare_kernel_callable(int32_t callable_id);

    /** Allocate / free / copy on the per-Worker `MemoryAllocator` + CANN runtime. */
    void *allocate_tensor(std::size_t bytes);
    /** Total device HBM (bytes) currently committed by this runner's MemoryAllocator. */
    std::size_t committed_device_memory() const { return mem_alloc_.committed_bytes(); }

    /**
     * Whether a collector may hold a run past its own boundary.
     * Latched once at device init.
     *
     * Default false: every collector behaves exactly as it does today, each
     * artifact keeps its name and location, and `run()` returning still implies
     * the file is written.
     *
     * Both retaining collectors are configured in the same breath, because this
     * is the one point at which the choice is known and it is before their lazy
     * `initialize()`. PMU retention is independent of swimlane's: a run may
     * enable either or both.
     */
    void set_retain_runs(bool enabled) {
        chip_swimlane_collector_.configure_retained_runs(enabled, simpler::dfx::runs::kDefaultBudgetBytes);
        pmu_collector_.configure_retained_runs(enabled);
    }
    bool retains_runs() const { return chip_swimlane_collector_.retains_runs() || pmu_collector_.retains_runs(); }

    /**
     * Publish every run closed up to now, then report.
     *
     * Returns 0 when each promised file exists; a published partial counts as
     * promised, and anything that left no file does not. The collector's writer
     * publishes on its own, so this is a barrier, not the only publisher.
     */
    int flush_diagnostics(int timeout_ms, std::string *error);

    /** Stop admitting runs and publish what is still retained. */
    void finish_retained_runs();

    void free_tensor(void *dev_ptr);
    int copy_to_device(void *dev_ptr, const void *host_ptr, std::size_t bytes);
    int copy_from_device(void *host_ptr, const void *dev_ptr, std::size_t bytes);

    /**
     * Allocate for a caller and record the allocation as theirs.
     *
     * The caller-facing mint. Separate from `allocate_tensor` because only what a caller minted may
     * be named by a run's arguments: this runner's own regions — workspace, retained temporaries,
     * arena banks — go through that one and stay unrecorded, so they cannot be borrowed.
     */
    void *allocate_caller_buffer(std::size_t bytes);

    /**
     * Release a caller's allocation, unless a run may still be using it.
     *
     * The caller keeps the right to free throughout; this only answers *now*. A refusal names the
     * borrow that is outstanding and changes nothing, so the caller retries once its run is done.
     *
     * @return 0 when the allocation is gone, `PTO_RUNTIME_ERR_INVALID_STATE` when a borrow still
     *         holds it. Nothing else: the platform free underneath is `free_tensor`, which reports
     *         no status, so a failure inside it is not visible here and 0 means "this path did not
     *         refuse", not "the pages are provably returned".
     */
    int free_caller_buffer(void *dev_ptr);

    /**
     * Take `identity`'s borrow over the caller allocations covering `spans`.
     *
     * All or nothing — see host/caller_device_buffers.h. A span that names no recorded caller
     * allocation is what makes this refuse, which is the proof-of-owner the joined path needs.
     */
    bool borrow_caller_buffers(uint64_t identity, const CallerDeviceBuffers::Span *spans, std::size_t count);

    /** Drop `identity`'s borrow, or keep it for good when its last consumer is unproven. */
    void release_caller_buffers(uint64_t identity, bool keep);

    /**
     * Declare which caller allocations `identity` produces; see host/caller_device_buffers.h.
     *
     * @return false when the statement could not be recorded, which the declaring run's bind has
     *         to treat as its own failure: an undeclared producer reads as no producer.
     */
    [[nodiscard]] bool declare_caller_buffer_writes(
        uint64_t identity, const CallerDeviceBuffers::Span *spans, std::size_t count,
        std::size_t *unresolved_out = nullptr
    ) {
        return caller_device_buffers_.declare_writes(identity, spans, count, unresolved_out);
    }

    /** Whether `[addr, addr + bytes)` has no readable content for `identity` yet. */
    bool caller_buffer_written_by_other_run(uint64_t identity, uint64_t addr, uint64_t bytes) const {
        return caller_device_buffers_.written_by_other_run(identity, addr, bytes);
    }

    /** The caller allocations this context has recorded, for teardown reporting and tests. */
    std::size_t caller_buffer_count() const { return caller_device_buffers_.allocation_count(); }
    std::size_t caller_buffer_borrow_count() const { return caller_device_buffers_.borrow_count(); }
    std::size_t caller_buffer_retained_count() const { return caller_device_buffers_.retained_count(); }

    int device_memset(void *dev_ptr, int value, std::size_t bytes);
    void get_retained_temp_buffer(uint32_t pipeline_slot, void **addr, std::size_t *size);
    void set_retained_temp_buffer(uint32_t pipeline_slot, void *addr, std::size_t size);
    int acquire_graph_definition_block(
        uint32_t pipeline_slot, std::size_t bytes, std::size_t alignment, void **device_out, void **staging_out
    );
    void get_graph_definition_staging(uint32_t pipeline_slot, void **addr, std::size_t *size);
    /**
     * Hand one pipeline slot its retained scheduler-state storage, both sides.
     *
     * `bytes` is what this run uses; the retained capacity may exceed it, and
     * only the caller's own length is ever initialized, uploaded or read back.
     * Both outputs are the aligned base, never the allocation it sits in.
     * Neither block is cleared: the caller writes the whole range it uses
     * before shipping it.
     *
     * Growth prepares the host side first — the side that can fail without
     * touching the device — and keeps the previously handed-out device block
     * when the device allocation fails, so a failed growth leaves the slot the
     * storage its last successful bind published into. A device block whose
     * release fails is retained in the slot's single failed-release record,
     * which then refuses any further growth for that slot.
     *
     * Refused while the runner cannot accept a run: a quarantined device is one
     * whose previous writer was never proven stopped, so its slot's storage is
     * not handed back for overwriting.
     */
    int acquire_scheduler_state_storage(
        uint32_t pipeline_slot, std::size_t bytes, std::size_t alignment, void **device_out, void **host_out
    );
    int acquire_sm_mirror(uint32_t pipeline_slot, std::size_t bytes, std::size_t alignment, void **addr_out);
    /**
     * Retain the host buffer a run assembles its device execution image in.
     *
     * Same retention contract as the shared-memory mirror above, and for the
     * same reason a run needs it: the publication that ships these bytes is a
     * separate step, so the source has to outlive the preparation that wrote
     * it rather than dying with the caller's frame.
     */
    int acquire_run_image_staging(uint32_t pipeline_slot, std::size_t bytes, std::size_t alignment, void **addr_out);
    void clear_temporary_buffer();

    /**
     * Latch this context's finite workspace budget, once.
     *
     * @return 0 on success; PTO_RUNTIME_ERR_INVALID_ARGUMENT for a zero budget
     *         or a second call.
     */
    int set_workspace_budget(std::uint64_t limit_bytes);

    /** Fill one workspace report. False when no budget is latched. */
    bool workspace_report(SimplerWorkspaceReport *out) const;

    /** Whether a workspace budget is latched on this context. */
    bool workspace_enabled() const { return workspace_.enabled(); }

    /**
     * Report one run fact at the boundary that produced it.
     *
     * Called from the run phase entries, never derived from a phase word or
     * from the code a caller received.
     */
    void note_workspace_run_fact(uint32_t pipeline_slot, std::uint64_t run_epoch, WorkspaceManager::RunFact fact) {
        workspace_.note_run_fact(pipeline_slot, run_epoch, fact);
    }

    /**
     * Publish the run identity this thread's workspace requests belong to.
     *
     * The arena callbacks and the retained-temp grow carry no identity of their
     * own, so the prepare driving them names one around the call. Thread-scoped
     * because a prepared successor may be built on another thread while this
     * one still holds its own plan.
     */
    static void begin_workspace_plan(uint32_t pipeline_slot, std::uint64_t run_epoch) noexcept;
    static void end_workspace_plan() noexcept;

    /**
     * Name the consumer region this thread's next workspace request serves.
     *
     * The arena allocation callback receives only a byte count, so the setup
     * driving it announces each region just before that region's backing is
     * staged. A block belongs to exactly one region, which is what stops a
     * growing GM heap from being handed the block a still-attached
     * shared-memory region is published at.
     */
    static void set_workspace_plan_region(const WorkspaceManager::RegionKey &region) noexcept;
    static WorkspaceManager::RegionKey workspace_plan_region() noexcept;

    /** Scopes one thread's workspace plan identity to a prepare. */
    class WorkspacePlanScope {
    public:
        WorkspacePlanScope(uint32_t pipeline_slot, std::uint64_t run_epoch) noexcept {
            begin_workspace_plan(pipeline_slot, run_epoch);
        }
        ~WorkspacePlanScope() { end_workspace_plan(); }
        WorkspacePlanScope(const WorkspacePlanScope &) = delete;
        WorkspacePlanScope &operator=(const WorkspacePlanScope &) = delete;
    };

    /** Runs still holding workspace whose completion a caller can still prove. */
    std::uint32_t workspace_live_consumers() const { return workspace_.live_drainable_consumers(); }

    /**
     * Acquire this slot's retained temporary staging buffer.
     *
     * Managed contexts answer from the ledger; unmanaged ones allocate exactly
     * as `RetainedTempBump` did on its own. Either way the slot ends up naming
     * the returned block, and a failure leaves the previous block named and
     * intact.
     *
     * @return 0 on success, non-zero when no block of `bytes` could be had
     */
    int acquire_retained_temp(uint32_t pipeline_slot, std::size_t bytes, void **addr_out, std::size_t *size_out);
    /**
     * Map a device buffer into the host address space and return a
     * host-readable VA (or nullptr on failure); the paired unregister releases
     * it. The returned VA may differ from dev_ptr, so callers must use it, not
     * dev_ptr, for host access. Register/unregister must be paired (unregister
     * before free_tensor). On a2a3 onboard this wraps
     * halHostRegister(DEV_SVM_MAP_HOST); a5 onboard has no host-map path and
     * uses the base default. Base default: unsupported (returns nullptr /
     * no-op); a2a3 overrides.
     */
    virtual void *register_device_memory_to_host(void *dev_ptr, std::size_t bytes) {
        (void)dev_ptr;
        (void)bytes;
        return nullptr;
    }
    /**
     * Release a mapping established above.
     *
     * @return 0 once the range is no longer mapped into this process, non-zero
     *         when it still is. A caller that owns the storage must keep it: the
     *         mapping covers the whole allocation, so releasing the bytes behind
     *         one would hand a live host address to the next allocation.
     */
    virtual int unregister_device_memory_from_host(void *dev_ptr) {
        (void)dev_ptr;
        return 0;
    }

    /**
     * Host view of a child-memory address for a host-side orchestrator, with
     * the mapping owned by this runner rather than by the caller.
     *
     * The mapping covers the whole tracked allocation containing `dev_ptr` —
     * that is the unit `free_tensor` invalidates, and several tensors or views
     * inside one child buffer then share it. Established on the first request
     * and kept until that allocation is freed, because establishing one costs
     * ~5.2 µs plus ~7.0 ms/GiB and a bind would otherwise pay it every run
     * (docs/investigations/2026-09-hbg-per-run-host-view-rebuild.md).
     *
     * @return a host address carrying `dev_ptr`'s offset within the allocation,
     *         or nullptr when `dev_ptr` is not inside a tracked allocation, or
     *         this backend has no host-map path (a5 onboard), or the platform
     *         refused the mapping (issue #1531).
     */
    void *acquire_child_memory_host_view(void *dev_ptr, std::size_t bytes);

    /**
     * Unregister every child-memory mapping still held.
     *
     * Runs before `mem_alloc_.finalize()`, which frees the allocations these
     * map: past that point the pages are gone and the mapping cannot be named.
     */
    void release_child_memory_host_views();

    /**
     * Unregister the one mapping over `alloc_base`, if it has one.
     *
     * The record is dropped only once the platform confirms the range is
     * unmapped, so a failure leaves this runner still naming the mapping it
     * still holds instead of forgetting it.
     *
     * @return 0 when `alloc_base` is no longer mapped into this process —
     *         including the common case of never having been mapped
     */
    int drop_child_memory_host_view(void *alloc_base);

    /**
     * Commit the three per-Worker pooled regions (GM heap, shared
     * shared memory, runtime arena) as three independent
     * device allocations. Must be called before any `acquire_pooled_*`.
     * Idempotent on identical (or smaller) sizes; an equal-or-smaller
     * follow-up request leaves the arena untouched. A region asked for 0 bytes
     * stays uncommitted, which is the shape hbg uses for its shared memory —
     * it calls `(heap_bytes, 0, device_arena_bytes)`, so its runtime arena
     * carries bytes like trb's does.
     *
     * A region that must grow is replaced through a staged transaction: every
     * replacement is allocated before any of them is installed, so a failed
     * allocation leaves each region with the base and capacity this call found
     * — including a region whose request was 0, whose release is deferred to
     * publication for the same reason. The call still fails, and the caller
     * that asked for the larger layout does not proceed on the old capacity.
     *
     * The transaction holds a growing region's old and new backing at the same
     * time, so it can be refused where a free-first sequence would have fitted.
     *
     * @return 0 on success, -1 on failure.
     */
    int setup_static_arena(uint32_t arena_bank, size_t gm_heap_size, size_t gm_sm_size, size_t runtime_arena_size);

    /**
     * Return the pooled GM heap / shared memory / runtime arena base pointer of the
     * selected arena bank. `setup_static_arena` (arch subclass) must have
     * already committed the relevant region on that bank; otherwise returns
     * nullptr.
     *
     * Which regions carry bytes is the caller's shape, and the two runtimes
     * differ. trb commits all three. hbg calls
     * `setup_static_arena(heap_bytes, 0, device_arena_bytes)`, so its shared
     * memory stays uncommitted and `acquire_pooled_gm_sm` returns nullptr for
     * it, while its runtime arena is committed and is a region it acquires.
     */
    void *acquire_pooled_gm_heap(uint32_t arena_bank);
    void *acquire_pooled_gm_sm(uint32_t arena_bank);
    void *acquire_pooled_runtime_arena(uint32_t arena_bank);
    bool lookup_prebuilt_runtime_arena_cache(
        uint32_t arena_bank, uint64_t hash, const void *key_data, size_t key_size, void **gm_heap_base, void **sm_base,
        void **runtime_arena_base, size_t *runtime_off, const void **image_data, size_t *image_size
    ) const;
    void mark_prebuilt_runtime_arena_cached(
        uint32_t arena_bank, uint64_t hash, const void *key_data, size_t key_size, void *gm_heap_base, void *sm_base,
        void *runtime_arena_base, size_t runtime_off, const void *image_data, size_t image_size
    );

    /**
     * Create a thread bound to this device. The thread calls
     * rtSetDevice(device_id) on entry.
     */
    std::thread create_thread(std::function<void()> fn);

    /**
     * Bind the calling thread to a device, taking nothing else from it.
     *
     * Idempotent for the same id; errors if called with a different id after
     * a prior adopt. Creates no streams and records no identity.
     *
     * @param device_id  Device ID (0-15)
     * @return 0 on success, error code on failure.
     */
    int bind_current_thread(int device_id);

    /**
     * Bind the current host thread and adopt the device for a program
     * context: on the first call it also resolves the timeout config, writes
     * the card's op-execute watchdog, and records device_id_.
     *
     * Required before host-side runtime initialization may allocate or free
     * device memory on the current thread. Refuses with
     * PTO_RUNTIME_ERR_INVALID_STATE on a context latched to kernel mode, which
     * owns neither the bind nor the watchdog.
     *
     * @param device_id  Device ID (0-15)
     * @return 0 on success, error code on failure.
     */
    int attach_current_thread(int device_id);
    /**
     * Record which device a kernel-mode context runs on. Takes no device side
     * effect: the caller already holds the device current, so this neither
     * binds the thread nor touches the card's op-execute watchdog. Requires
     * the execution-mode latch to already read kernel.
     */
    int adopt_borrowed_device(int device_id);

    /**
     * One-shot device initialization. Performs, in order:
     *   1. attach_current_thread on device_id_
     *   2. rtStreamCreate for AICPU + AICore streams (persistent, freed
     *      by the subclass `finalize()`).
     *   3. Bootstrap the dispatcher + register the inner AICPU SO via
     *      `ensure_binaries_loaded()`.
     *   4. Provision the requested async-DMA workspaces via
     *      `ensure_dma_workspace_provisioned()`.
     *   5. Launch `simpler_aicpu_init` via `ensure_aicpu_init_launched()`,
     *      which publishes step 4's addresses along with the other resident
     *      invariants. Step 4 precedes it so that publication is one launch.
     *   6. Warm the SDMA control path via `ensure_dma_workspace_warmed()`,
     *      which needs both the live workspace and the AICore stream.
     *
     * Called from `simpler_init` after executor + dispatcher bytes and the
     * async-DMA request have been cached on the runner. Idempotent: each step
     * short-circuits on its own guard.
     *
     * @return 0 on success, error code on failure.
     */
    int ensure_device_initialized();

    /**
     * Print handshake results from device. Reads the per-core
     * `Handshake` array out of device memory and logs it at DEBUG. Must
     * be called after `drain_execution()` and before `finalize()`.
     */
    void print_handshake_results(const KernelArgsHelper &kernel_args);

    /**
     * Take ownership of the AICPU + AICore executor binaries. Called
     * once by simpler_init at ChipWorker::init time; subsequent
     * enqueue invocations read from `aicpu_so_binary_` /
     * `aicore_kernel_binary_`.
     */
    void set_executors(std::vector<uint8_t> aicpu_so_binary, std::vector<uint8_t> aicore_kernel_binary) {
        aicpu_so_binary_ = std::move(aicpu_so_binary);
        aicore_kernel_binary_ = std::move(aicore_kernel_binary);
    }

    /**
     * Take ownership of the dispatcher SO bytes. Called by simpler_init
     * when the caller provided a dispatcher path; the eager
     * `ensure_device_initialized()` in simpler_init hands the buffer to
     * `LoadAicpuOp::BootstrapDispatcher` at init time. Leaving this
     * unset (empty buffer) makes `ensure_binaries_loaded()` fail with a
     * clear message — callers that drive `_ChipWorker.init` directly
     * without a dispatcher path get a deterministic error at
     * `simpler_init` time rather than a confusing dladdr-derived path.
     */
    void set_dispatcher_binary(std::vector<uint8_t> dispatcher_so_binary) {
        dispatcher_so_binary_ = std::move(dispatcher_so_binary);
    }

    /**
     * Record this Worker's async-DMA request. Called by simpler_init before its
     * `ensure_device_initialized()`, which is where the request is acted on —
     * the workspace has to exist before the one-shot `simpler_aicpu_init` launch
     * that publishes its addresses.
     *
     * `enable_sdma` opts into the SDMA workspace; every other engine
     * `dma_workspace_supported_mask()` names is provisioned regardless. SDMA is
     * declinable because its workspace cannot be obtained without also creating
     * 48 CP-process STARS streams, which halves this Worker's post-fault reset
     * budget. Opting in where SDMA is unsupported fails device init.
     *
     * `sdma_warmup_binary` is the vector-only ELF that walks the SDMA control
     * path once per channel once the workspace is live. An empty buffer only
     * costs first-call latency.
     */
    void set_dma_workspace_request(bool enable_sdma, std::vector<uint8_t> sdma_warmup_binary) {
        sdma_requested_ = enable_sdma;
        sdma_warmup_binary_ = std::move(sdma_warmup_binary);
    }

    /** Which device this context is on; -1 until an init entry adopts one. */
    int device_id() const { return device_id_; }

    /**
     * One run's device-timing readback, owned by that run's pipeline slot.
     *
     * Every field here is a *result of one execution*, so it must survive until
     * that run's finalize has emitted it. Keeping them on the runner instead
     * would let a successor's drain overwrite a predecessor's unread values —
     * silently, because nothing reads a generation. `device_id` and `sys_cnt_hz`
     * travel with the result rather than being read off the runner at emit time,
     * so the emitted bounds always name the device they were stamped on.
     */
    struct DeviceRunTiming {
        uint64_t wall_ns{0};
        uint64_t phase_ns[NUM_AICPU_PHASES]{};
        uint64_t phase_start_ns[NUM_AICPU_PHASES]{};
        uint64_t task_slot_dispatch_ns[NUM_TASK_TIMING_SLOTS]{};
        uint64_t task_slot_finish_ns[NUM_TASK_TIMING_SLOTS]{};
        uint64_t run_wall_start_cycles{0};
        uint64_t run_wall_end_cycles{0};
        uint64_t sys_cnt_hz{0};
        int device_id{-1};
    };

    /**
     * This slot's last completed run's device timing. A slot's result is written
     * by `read_device_wall_ns(slot)` at drain and consumed by finalize; the slot
     * is not handed to another run in between, which is what makes the result
     * this run's own. Returns a zeroed record before the slot has run, or when
     * capture is off.
     */
    const DeviceRunTiming &device_run_timing(uint32_t pipeline_slot) const;

    /**
     * Mark this slot's timing result consumed, so the slot may be armed again.
     * Called by finalize on every path that ends a run, emitted or not: a run
     * that never launched still owns its slot. Re-arming a slot that is still
     * armed means a successor took storage whose result nobody read, and
     * `arm_device_wall_buffer` reports that rather than overwriting it.
     */
    void release_device_run_timing(uint32_t pipeline_slot);

    /**
     * Device-side wall (ns) of the run that last used `pipeline_slot`,
     * written by the platform AICPU entry. Returns 0 before that slot has
     * completed a run. Independent of any profiling / swimlane subsystem.
     */
    uint64_t last_device_wall_ns(uint32_t pipeline_slot) const { return device_run_timing(pipeline_slot).wall_ns; }

    /**
     * Per-phase AICPU wall (ns) for that slot's run, reduced across threads as
     * max(end) - min(start). Returns 0 for a phase that was never stamped
     * (e.g. a platform whose AICPU does not emit that phase).
     * AicpuPhase::RunWall aliases last_device_wall_ns(). Used by the host to
     * emit device-phase trace markers; see simpler_run in c_api_shared.
     */
    uint64_t last_device_phase_ns(uint32_t pipeline_slot, AicpuPhase phase) const {
        return device_run_timing(pipeline_slot).phase_ns[static_cast<int>(phase)];
    }

    /**
     * Per-phase start offset (ns) on a common device-clock timeline shared by
     * all sub-phases of the run (origin = the earliest sub-phase start). Lets
     * the host emit each device span with a device-domain `ts` so the
     * orchestrator/scheduler windows are comparable (their union is the
     * "Effective" window) and the sub-phases nest correctly. 0 for RunWall (the
     * origin) and for any phase never stamped.
     */
    uint64_t last_device_phase_start_ns(uint32_t pipeline_slot, AicpuPhase phase) const {
        return device_run_timing(pipeline_slot).phase_start_ns[static_cast<int>(phase)];
    }

    /**
     * RunWall's raw device-clock bounds in `get_sys_cnt_aicpu()` ticks —
     * `min_start` and `max_end` across threads, unconverted and not rebased on
     * this run's origin. That counter is CNTVCT_EL0 rescaled into the
     * PLATFORM_PROF_SYS_CNT_FREQ unit, so it is a monotone function of a
     * free-running counter that is never reset per run: these ticks stay
     * comparable **across runs on one device within one counter epoch** — which
     * is what makes the interval between one run's device end and the next
     * run's device start computable. Difference the ticks first and convert
     * afterwards against the record's own `sys_cnt_hz` (the unit they are
     * already in, not `cntfrq_el0`); converting each bound first would round
     * both ends of a sub-microsecond gap away. Not comparable to the host
     * clock. Both 0 when unstamped or capture is off.
     */
    uint64_t last_device_run_wall_start_cycles(uint32_t pipeline_slot) const {
        return device_run_timing(pipeline_slot).run_wall_start_cycles;
    }
    uint64_t last_device_run_wall_end_cycles(uint32_t pipeline_slot) const {
        return device_run_timing(pipeline_slot).run_wall_end_cycles;
    }

    /** Tick rate the two bounds above are expressed in (50 MHz a2a3, 1 GHz a5). */
    static uint64_t device_sys_cnt_frequency_hz() { return PLATFORM_PROF_SYS_CNT_FREQ; }

    /**
     * This slot's last run's device-published result payload, or `nullptr` when
     * that run published none.
     *
     * `run_epoch` is the epoch of the run whose result is wanted: a region still
     * holding an earlier run's epoch is reported as absent rather than returned,
     * so a caller cannot read a predecessor's payload as this run's. `*bytes_out`
     * receives the published length.
     *
     * Absent carries no verdict. A producer attaches a payload only to a
     * failure, so a successful run, a run that reported no detail and a run
     * that never reached its publish point — one the op-execute watchdog
     * reaped, say — all read as absent here. Whether the run succeeded is
     * `device_run_terminal`'s answer, and it must not be inferred from the
     * presence or absence of this payload, nor answered by reading shared
     * device state, which by then may belong to a successor.
     */
    const uint8_t *device_run_result(uint32_t pipeline_slot, uint64_t run_epoch, size_t *bytes_out) const;

    /**
     * Copy this slot's result region into the host-side copy `device_run_result`
     * and `device_run_terminal` read. Call after the run's own completion
     * boundaries: what makes the record this run's rather than a successor's is
     * that its device side published it before its kernel returned, so this read
     * itself races nothing — the slot is not handed on until the run holding it
     * finalizes.
     *
     * One read per run. A repeat call for an epoch already read is a no-op, so
     * every later consumer sees the same bytes and no path pays a second D2H —
     * in particular a read must not be retried after a device recovery, which
     * would sample a generation this run never wrote.
     *
     * Leaves the host copy empty when there is no region or the copy fails; a
     * failed copy is recorded as such, so it reads as undecided rather than as
     * an absent record, and a slot with no region records no attempt at all.
     *
     * Returns the status the transfer itself reported for this run: zero when
     * the bytes landed, when there was no region, and when no read is owned
     * here. A non-zero answer is an error this host thread has already
     * observed from the SDK, and the same value is returned by every later
     * call for the same run — so a caller may act on it without a
     * deduplicated call looking like a transfer that succeeded. Pair it with
     * `device_run_result_read_status` rather than reading zero as proof a
     * transfer happened.
     */
    int read_device_run_result(uint32_t pipeline_slot, uint64_t run_epoch);

    /**
     * What this slot's cached region says about the run whose epoch is
     * `run_epoch`: succeeded, failed with the runtime's own signed code, or
     * undecided with the reason.
     *
     * This is the run's execution outcome only. It does not say the device is
     * healthy and it does not say the run's resources are retirable — a device
     * can publish a failure and keep tearing down. Quiescence comes from the
     * run's completion boundaries and its outstanding wait references.
     */
    DeviceRunTerminal device_run_terminal(uint32_t pipeline_slot, uint64_t run_epoch) const;

    /**
     * Which of the three read states this slot holds for `run_epoch`.
     *
     * The three answers are distinct evidence, which is why the caller gets
     * them rather than a bool: a read that never happened, a copy that failed,
     * and a copy that succeeded onto a region no run published into all leave
     * the same empty bytes behind. A slot holding no region is the first of
     * those, not the second — nothing was copied, so nothing was lost.
     */
    RunRecordRead device_run_result_read_status(uint32_t pipeline_slot, uint64_t run_epoch) const;

    /**
     * The boundary completion this run's own drain or poll observed, or
     * `Pending` when none did.
     *
     * Retained from the observation rather than re-derived: by the time a
     * caller asks, the drain's cleanup has retired the fence's arming, so the
     * fence can no longer answer for this run. Reads no device state.
     */
    RunCompletionFence::Completion observed_run_boundaries(const NativeRunIdentity &identity) const;

    /**
     * Consume every notification reported since this runner last looked, and
     * return how many named a stream **this runner's runs submit on**.
     *
     * A notice that matches refuses this runner's *future* admission, through
     * the suspicion `accepts_new_run` reads. It decides no run: the notice names
     * a device and a stream, carries no run identity, and can arrive late (16 s
     * is the longest lag measured, not a bound), so it can name a fault from an
     * earlier run than the one being finalized. It starts no drain, reset or
     * recovery either, and reclaims nothing.
     *
     * The match is membership, not provenance. It says a fault landed on a
     * stream this runner's runs use — on a5 that stream also carries binary
     * load, AICPU init and callable registration — not that a run caused it and
     * not that a run was impaired. So a control-plane failure the host already
     * reported synchronously can still refuse later admission.
     *
     * Attribution is per stream, not merely per device: the notice's `device_id`
     * is logical (the same space this runner names its device in) and its
     * `stream_id` is a driver id compared against the ids this device's runs
     * were recorded on at launch. A notice matching none of them is unattributed
     * while that history is complete, and *undecided* once an id is missing —
     * evicted by capacity, or never obtained because the query failed. Neither
     * refuses anything, and neither does a lost or dropped notice; the ring is
     * process-wide, and this runner speaks for its own streams.
     */
    uint64_t consume_device_fault_notices() noexcept;

    /** Evidence this device's fault channel has produced in the live generation. */
    const DeviceHealthState &device_health() const { return device_health_; }

    /**
     * Per-slot task-timing dispatch/finish (ns) on the same device-clock timeline
     * as the phases. Both 0 for an untagged or incomplete slot. `slot` is 0..15
     * — a *task* timing slot, unrelated to `pipeline_slot`.
     */
    uint64_t last_task_slot_dispatch_ns(uint32_t pipeline_slot, int slot) const {
        return device_run_timing(pipeline_slot).task_slot_dispatch_ns[slot];
    }
    uint64_t last_task_slot_finish_ns(uint32_t pipeline_slot, int slot) const {
        return device_run_timing(pipeline_slot).task_slot_finish_ns[slot];
    }

    /**
     * Upload an entire ChipCallable buffer to device memory in one shot.
     * Walks child_offsets_ to compute total byte size, allocates device
     * GM once, fixes up each child's resolved_addr_ in an internal host
     * scratch (= device-side address of that child's binary code),
     * H2D's once, and returns the device-side address of the
     * ChipCallable header.
     *
     * Pool-managed: identical buffer bytes (FNV-1a 64-bit content hash)
     * hit the dedup cache and return the cached chip_dev without
     * reallocating. Each successful upload retains one reference; ownership is
     * transferred into a CallableState or released on registration failure.
     *
     * Callers compute child addresses as
     *     chip_dev + offsetof(ChipCallable, storage_) + child_offset(i)
     * for their own validation. The same arithmetic also builds the callable's
     * function tables here, in the aligned tail of this one allocation, so a run
     * binds a reference to them instead of rebuilding them.
     *
     * @param callable  Host-side ChipCallable pointer.
     * @return Device GM address of the ChipCallable header, or 0 on failure.
     */
    uint64_t upload_chip_callable_buffer(const ChipCallable *callable);
    int release_chip_callable_buffer(uint64_t hash);

    /**
     * Stage a per-callable_id orchestration SO from the retained ChipCallable and
     * remember the supporting metadata (entry/config symbol names). The
     * orchestration SO is the leading slice of ChipCallable::storage_ inside the
     * retained chip buffer, whose hash is also how a bind reaches that
     * callable's function tables.
     *
     * @param callable_id   Caller-stable id, must be in [0, MAX_REGISTERED_CALLABLE_IDS).
     * @param chip_buffer_hash  FNV-1a hash of the retained ChipCallable buffer.
     * @param chip_dev      Device GM address of the retained ChipCallable header.
     * @param orch_so_data  Host pointer to orchestration SO bytes (owned by caller).
     * @param orch_so_size  Size of orchestration SO in bytes.
     * @param func_name     Entry symbol name (copied).
     * @param config_name   Config symbol name (copied).
     * @return 0 on success, negative on failure.
     */
    int record_device_orch_callable(
        int32_t callable_id, uint64_t chip_buffer_hash, uint64_t aicore_image_hash, uint64_t chip_dev,
        const void *orch_so_data, size_t orch_so_size, const char *func_name, const char *config_name,
        std::vector<ArgDirection> signature
    );

    /**
     * Host-orchestration variant of record_device_orch_callable: stores a dlopen
     * handle + entry-symbol pointer that runtime_maker resolved on the
     * host (host_build_graph variant). Mutually exclusive with the
     * trb-shaped overload — exactly one is invoked for a given
     * callable_id, picked by the C ABI based on which staging fields
     * the runtime carries after register_callable_impl. dlopen handle
     * is owned by `DeviceRunnerBase` from this call onward and
     * dlclose'd by `unregister_callable`. Increments `host_dlopen_total_`.
     */
    int record_host_orch_callable(
        int32_t callable_id, uint64_t chip_buffer_hash, uint64_t aicore_image_hash, void *host_dlopen_handle,
        void *host_orch_func_ptr, std::vector<ArgDirection> signature
    );

    /**
     * Drop the registered state for `callable_id`. Decrements the retained
     * chip buffer's hash-keyed refcount and frees when it hits zero. hbg path
     * also dlcloses the host dlopen handle.
     *
     * @return 0 on success or if the id was not registered.
     */
    int unregister_callable(int32_t callable_id);

    /**
     * True iff `callable_id` has registered state staged via
     * `record_device_orch_callable*`. Lets the c_api layer reject `simpler_run`
     * calls without a matching `simpler_register_callable`.
     */
    bool has_callable(int32_t callable_id) const;

    /**
     * Content-derived stable identity for a registered callable: the
     * ELF Build-ID 64-bit hash of its orchestration SO (CallableState::hash,
     * computed at record_device_orch_callable time via elf_build_id_64). Returns 0 when
     * the callable_id is not registered.
     *
     * Stable across slot reuse (unlike callable_id, which is a recyclable
     * slot index) and across processes / runs (same SO bytes → same hash),
     * so DFX trace markers use it as the `hid` grouping key to attribute
     * per-stage timing to a specific callable.
     */
    uint64_t callable_hash(int32_t callable_id) const;

    /**
     * Publish this run's core geometry onto `Runtime` before the graph is
     * built: resolves `block_dim`, derives `num_aicore = block_dim *
     * cores_per_blockdim_`, range-checks against `RUNTIME_MAX_WORKER`,
     * publishes the Runtime's `worker_count` / `aicpu_thread_num`,
     * and zero-initializes the handshake worker array with AIC/AIV core
     * typing (first `block_dim` cores are AIC, remaining are AIV).
     *
     * Callers run this before `bind_callable_to_runtime` so a host-side
     * orchestrator sees the real core count while it submits, rather than
     * the zeros a freshly constructed `Runtime` carries. Needs
     * `ensure_device_initialized()` to have latched `max_block_dim_`.
     *
     * Returns 0 on success, -1 on a bad `block_dim` / `aicpu_thread_num`.
     */
    int prepare_launch_shape(Runtime &runtime, const CallConfig &config);

    /** Latch a prepared Runtime's geometry immediately before execution. */
    void activate_launch_shape(const Runtime &runtime);

    /**
     * Point a fresh Runtime at a previously-registered callable and complete
     * the per-run binding in one step. Installs the reference to that
     * callable's registration-owned function tables and its
     * active_callable_id, then calls the runtime's bind_callable_to_runtime_impl
     * with the CallableState-derived host_orch_func_ptr + signature (kept
     * internal to the runner rather than returned across the c_api boundary).
     *
     * @param api               Context-bound platform device-memory hooks.
     * @param orch_args         const ChipStorageTaskArgs* for this run (void* to
     *                          keep task_interface headers out of this header).
     * @param ring_task_window  Per-ring overrides (trb); ignored by hbg.
     * @return 0 on success, non-zero on failure (unregistered id, a callable
     *         whose registration block is gone, or the underlying
     *         bind_callable_to_runtime_impl rc).
     */
    int bind_callable_to_runtime(
        Runtime &runtime, int32_t callable_id, const HostApi *api, const void *orch_args,
        const uint64_t *ring_task_window, const uint64_t *ring_heap, const uint64_t *ring_dep_pool
    );

    /**
     * Number of distinct callable_ids the AICPU has been asked to
     * dlopen for. Monotonically increases when an AICPU load succeeds
     * during prepare prewarm or first-run fallback; `unregister_callable`
     * does NOT decrement it. So a `prepare → unregister → re-prepare`
     * sequence reports 2 (each AICPU dlopen counted once), even though one cid is
     * currently registered.
     */
    size_t aicpu_dlopen_count() const { return aicpu_dlopen_total_; }

    /**
     * Number of host-side dlopen() invocations triggered by
     * `record_host_orch_callable`. Mirrors `aicpu_dlopen_count` but
     * counts the host_build_graph variant's host-side dlopens; it
     * never decrements.
     */
    size_t host_dlopen_count() const { return host_dlopen_total_; }

    /**
     * Number of run stream generations this runner has created. AICPU streams
     * belong to pipeline slots, while an AICore stream is reused only for the
     * same AICore image. Arches whose runs use the persistent pair report 0.
     */
    virtual size_t run_stream_set_create_count() const { return 0; }

    /**
     * Copy out what this runner's `finalize()` observed about its own device
     * teardown, if it recorded one.
     *
     * A runner that records nothing answers false, which is how a backend
     * outside the recording scope reports "no observation" rather than a
     * teardown that did not happen.
     */
    bool copy_teardown_report(SimplerTeardownReport *out) const { return teardown_recorder_.copy_to(out); }

    /**
     * Device-orchestration callable registration used internally by
     * simpler_register_callable(): launches `simpler_aicpu_register_callable` with a
     * RegisterCallableArgs descriptor so the AICPU dlopens the callable's
     * orch SO. Host-orchestration callables are a no-op. On success, AICPU has
     * populated orch_so_table_[callable_id] and subsequent runs only need to
     * stamp the active callable_id.
     */
    int launch_device_register(int32_t callable_id);

    /**
     * Commit host-side AICPU seen/counting state after a device-side SO load
     * has returned success. Calling this before the device helper succeeds can
     * make a later run advertise a false cache hit.
     */
    int commit_device_register(int32_t callable_id);

    // ---- Virtual entry points called by the shared c_api ----------------
    //
    // The shared `runtime_c_api` glue (`src/common/platform/onboard/host/
    // c_api_shared.cpp`) works through `DeviceRunnerBase *` and dispatches
    // through these virtuals. Each arch's `DeviceRunner` overrides the
    // enqueue/poll/drain lifecycle and `finalize`; a2a3 and a5 both override
    // `set_dep_gen_enabled` (an arch without dep_gen keeps the base no-op
    // default).

    /**
     * Whether this runner may start another run without first being finalized.
     * The shared c_api checks this at every run boundary (prepare / launch /
     * finalize), so a poisoned runner fails admission ahead of the arch-specific
     * enqueue fail-fast guard.
     */
    virtual bool can_accept_run() const = 0;

    /**
     * Whether the shared c_api may admit a new run on this runner.
     *
     * Composes the arch's own quarantine with the device-fault channel's
     * generation-scoped suspicion; `device_admits_new_run` states what each
     * refusal means and how each clears. Admission sites ask this. The two
     * diagnostic readers of `can_accept_run()` — the clock-correlation session's
     * abandon flag — deliberately keep asking the arch flag alone, because a
     * matched notice says nothing about whether this run's own DFX resources are
     * safe to release normally.
     */
    bool accepts_new_run() const { return device_admits_new_run(!can_accept_run(), device_health_); }

    /**
     * An AICore launch or stream sync failed outside the per-run path. The arch
     * runner drains what it can and flips its device-unusable flag, so the next
     * admission fails fast and finalize() takes its fatal teardown path instead of
     * per-resource release on a faulted card. The base default is a no-op for
     * runners that track no such state.
     */
    virtual void recover_device_or_mark_unusable(int /*aicore_rc*/) {}

    /** Invalidate retained run streams after new AICore code is published. */
    virtual void mark_run_streams_stale() {}

    /**
     * Whether this runner can order one run's submission behind another's right
     * now.
     *
     * Base answer: no. The ordering edge is queued onto per-run streams, so a
     * backend with no such pair has nothing to queue it into, and one whose
     * AICore stream is awaiting replacement cannot both keep a live run on it
     * and give a joining run a stream free of the previous code image.
     */
    virtual bool ready_to_join_launch() const { return false; }

    /** Provision/abandon platform resources owned by one prepared native run. */
    virtual int provision_native_run_resources(uint32_t /*pipeline_slot*/) { return 0; }
    virtual int abandon_native_run_resources(uint32_t /*pipeline_slot*/) { return 0; }

    struct PreparedExecution {
        PreparedExecution(
            const NativeRunIdentity &identity_in, Runtime &runtime_in, const CallConfig &config_in,
            uint32_t pipeline_slot_in
        ) :
            identity(identity_in),
            runtime(&runtime_in),
            config(config_in),
            dfx(DfxRunConfig::from(config_in)),
            pipeline_slot(pipeline_slot_in) {}
        PreparedExecution(const PreparedExecution &) = delete;
        PreparedExecution &operator=(const PreparedExecution &) = delete;
        PreparedExecution(PreparedExecution &&other) noexcept :
            identity(other.identity),
            runtime(std::exchange(other.runtime, nullptr)),
            config(other.config),
            dfx(std::move(other.dfx)),
            pipeline_slot(other.pipeline_slot),
            num_aicore(other.num_aicore),
            launch_aicpu_num(other.launch_aicpu_num),
            kernel_args(std::move(other.kernel_args)),
            resources_owned(std::exchange(other.resources_owned, false)),
            aicore_retirement_attempted(std::exchange(other.aicore_retirement_attempted, false)),
            joinable_boundary(other.joinable_boundary),
            join(other.join) {}
        PreparedExecution &operator=(PreparedExecution &&) = delete;

        NativeRunIdentity identity{};
        Runtime *runtime{nullptr};
        CallConfig config{};
        /**
         * This run's diagnostics configuration, resolved from its own config.
         *
         * Every phase of the run reads its DFX configuration from here rather
         * than from the runner's members: prepare because it can overlap a
         * predecessor whose configuration is still the one bound on the runner,
         * and launch/drain so that one run answers for its whole lifetime from a
         * single value. A run that degrades a channel (a5 disables PMU when its
         * init fails) writes that here, where it reaches this run's arming and
         * teardown and no other run's.
         */
        DfxRunConfig dfx{};
        uint32_t pipeline_slot{PTO_PIPELINE_MAX_DEPTH};
        int num_aicore{0};
        int launch_aicpu_num{0};
        KernelArgsHelper kernel_args{};
        bool resources_owned{false};
        bool aicore_retirement_attempted{false};
        /**
         * Whether this run constructs a whole-operator completion boundary: an
         * intra-run wait for its own AICore boundary queued ahead of its AICPU
         * boundary record, so that the AICPU boundary covers both kernels.
         *
         * A property of the launching Worker's configured depth rather than of
         * this run, because a predecessor is launched before any successor can
         * be authorized to join it — a boundary constructed only once a join is
         * known would never exist when it is needed. At depth one it stays
         * false and the launch path is the one that shipped without it.
         */
        bool joinable_boundary{false};
        /**
         * The predecessor this run was ordered behind, or an empty join for an
         * ordinary launch. Set at launch, not at prepare: a prepared successor
         * whose predecessor retires first launches ordinarily, and a join
         * decided at prepare would by then name a run that is gone.
         */
        NativeRunJoin join{};
    };

    struct ActiveExecution {
        explicit ActiveExecution(std::unique_ptr<PreparedExecution> prepared_in, LaunchProgress progress_in) :
            prepared(std::move(prepared_in)),
            progress(progress_in) {}
        ActiveExecution(const ActiveExecution &) = delete;
        ActiveExecution &operator=(const ActiveExecution &) = delete;
        ActiveExecution(ActiveExecution &&) noexcept = default;
        ActiveExecution &operator=(ActiveExecution &&) noexcept = default;

        std::unique_ptr<PreparedExecution> prepared;
        LaunchProgress progress{LaunchProgress::NotStarted};
    };

    struct LaunchOutcome {
        int rc{-1};
        LaunchProgress progress{LaunchProgress::NotStarted};
        std::unique_ptr<PreparedExecution> prepared{};
        std::unique_ptr<ActiveExecution> active{};
        LaunchReceipt receipt{};

        bool poisoned() const { return progress == LaunchProgress::Partial; }
    };
    /**
     * Prepare host-owned execution state without crossing the device launch
     * boundary. The returned object owns everything needed by launch.
     */
    virtual int prepare_execution(
        Runtime &runtime, const CallConfig &config, uint32_t pipeline_slot, const NativeRunIdentity &identity,
        std::unique_ptr<PreparedExecution> *prepared
    ) = 0;
    virtual LaunchOutcome launch_execution(std::unique_ptr<PreparedExecution> prepared, LaunchPermit permit) = 0;
    virtual void abandon_prepared_execution(PreparedExecution &prepared) noexcept = 0;

    /**
     * Query the active run without waiting. Returns one of the
     * SIMPLER_NATIVE_RUN_POLL_* values.
     */
    virtual int poll_execution(const ActiveExecution &active) = 0;

    /**
     * Wait for the launched run, publish DFX, and release its execution
     * resources. Called on the child progress path that performed launch.
     */
    virtual int drain_execution(ActiveExecution &active) = 0;

    /**
     * Cleanup all resources. Each arch's `finalize()` wraps
     * `finalize_common()` with arch-specific device-reset behaviour:
     * a2a3 has the ACL-ready branch + dep_gen collector teardown;
     * a5 does straight `rtDeviceReset`. See the subclass docs for the
     * per-arch contract.
     */
    virtual int finalize() = 0;

    virtual int fill_persistent_arch_fields(KernelArgs *args, uint64_t device_id) = 0;

    /**
     * Fill the `InitArgs` fields only one architecture defines, before the init
     * entry uploads them. Default: nothing, which is what an architecture whose
     * `InitArgs` carries no such field needs — the base fills the common ones.
     */
    virtual void fill_init_arch_fields(InitArgs & /*init_args*/) {}

    /**
     * Arm or disarm this thread's host-side dep_gen capture, from the run's own
     * config, before it binds.
     *
     * A host-orchestrating runtime holds the captured graph in thread-local
     * state between orchestration and emit, so this has to run on the thread
     * that is about to bind, for every prepare — including one that overlaps an
     * active predecessor, which skips `apply_call_config`. It writes nothing the
     * runner shares between runs. An arch without dep_gen keeps the no-op.
     */
    virtual void arm_host_dep_gen_capture(bool /*enable*/) {}

    /**
     * Launch an AICPU entry with an arbitrary launch-arg payload.
     *
     * Every AICPU launch goes through here: the run entry, whose payload is
     * `KernelArgs` alone or that header followed by this run's entry values,
     * and the non-exec entries whose payload is neither — `simpler_aicpu_init`
     * (InitArgs) and `simpler_aicpu_register_callable` (RegisterCallableArgs).
     * `args_size` is what reaches `rtsLaunchCpuKernel` as `argsSize`, so it is
     * how far past the header RTS copies.
     *
     * @param stream       AICPU stream
     * @param args         Payload pointer (host memory; CANN copies it in)
     * @param args_size    Payload size in bytes
     * @param kernel_name  Name of the kernel to launch
     * @param aicpu_num    Number of AICPU instances to launch
     * @return 0 on success, error code on failure
     */
    int launch_aicpu_payload(rtStream_t stream, void *args, size_t args_size, const char *kernel_name, int aicpu_num);

    /**
     * Launch an AICore kernel. Lazy-registers the kernel binary
     * (`aicore_kernel_binary_`) on first call via `rtRegisterAllKernel`
     * and caches the resulting `aicore_bin_handle_`; subsequent calls
     * reuse the cached handle. CANN has no public
     * `rtUnregisterAllKernel`, so re-registering on every run would pin
     * another device-side copy of the ELF and quickly exhaust HBM —
     * manifested in CI as 207001 at `rtKernelLaunchWithHandleV2` with a
     * 507899 cascade at `rtStreamCreate`.
     *
     * `k_args` is projected into `AicoreLaunchArgs` and reaches the AICore
     * kernel as the `rtArgsEx_t` parameter block itself — by value, with no
     * device-resident copy. The projection carries this run's final values, so
     * the call must follow collector arming.
     */
    int launch_aicore_kernel(rtStream_t stream, const KernelArgs &k_args);

    /**
     * Walk the SDMA control path once per channel, so the first TPREFETCH_ASYNC
     * of a run does not pay it. Called from ensure_dma_workspace_warmed() once
     * the workspace is live, on `stream_aicore_`, and synchronized before
     * returning.
     *
     * `binary` is a vector-only ELF, registered separately from the executor
     * (`RT_DEV_BINARY_MAGIC_ELF_AIVEC`, its own handle) because the executor is a
     * resident loop launched per run with a `block_dim_` that is still 0 here.
     *
     * Unavailability is best-effort and returns 0: an absent binary, no channels,
     * a failed registration or allocation, and a channel that declines to warm all
     * cost only first-call latency. A failed launch or stream sync is not, because
     * it means an AICore operation faulted on this card; that marks the runner
     * unusable and returns the error so the caller does not hand a poisoned device
     * to the first run.
     *
     * @return 0 when the warmup ran or was unavailable, the device error otherwise.
     */
    int launch_sdma_warmup_kernel(const void *binary, size_t size);

    /**
     * Read back the warmup kernel's per-channel status slots and report how many
     * channels came up warm, splitting the remainder into channels that declined
     * the warmup's preconditions and channels no core reached. Takes ownership of
     * `status_dev` and frees it. `elapsed_ms` is the launch-to-sync wall time,
     * reported alongside the count because it is the init-time cost being traded
     * for first-run latency.
     */
    void report_sdma_warmup_status(void *status_dev, uint32_t channel_count, double elapsed_ms);

    /**
     * Enablement setters for the four shared diagnostics sub-features.
     * Applied from the per-run CallConfig by `apply_call_config()` before
     * prepare. Execution paths do not read these: a run's own diagnostics
     * configuration reaches them on its `PreparedExecution::dfx`, and the runner
     * keeps only what a device-context query answers from.
     *
     * `set_dep_gen_enabled` is a2a3-only and lives on the subclass.
     */
    void set_chip_swimlane_enabled(int level) { chip_swimlane_level_ = static_cast<ChipSwimlaneLevel>(level); }
    uint32_t chip_swimlane_level() const { return static_cast<uint32_t>(chip_swimlane_level_); }
    bool
    publish_chip_swimlane_extension(ChipSwimlaneExtensionSection section, const char *json_value, size_t json_size) {
        return json_value != nullptr &&
               chip_swimlane_collector_.set_json_extension(section, std::string(json_value, json_size));
    }
    /**
     * Hand one pipeline slot's host-phase state to the run about to bind into
     * it, from that run's own config. Called before every bind, because the
     * runner's members describe whichever run last held the execution claim.
     */
    void begin_host_phase_run(uint32_t pipeline_slot, const DfxRunConfig &dfx);
    HostPhaseRecordPool *host_phase_pool_arm(uint32_t pipeline_slot, bool producer_wants_records) noexcept;
    void host_phase_pool_finish(uint32_t pipeline_slot, uint64_t submitted_tasks, uint64_t invocation_id) noexcept {
        if (pipeline_slot >= host_phase_runs_.size()) return;
        host_phase_runs_[pipeline_slot].records.finish(submitted_tasks, invocation_id);
    }
    /** Hand this pass's records to the swimlane reader, just before its export. */
    bool host_clock_alignment_log_required(uint32_t pipeline_slot) const {
        return pipeline_slot < host_phase_runs_.size() && host_phase_runs_[pipeline_slot].needs_clock_alignment();
    }
    void publish_host_phase_records_to_swimlane(uint32_t pipeline_slot);
    /**
     * Hand this run's host-phase state to the resident collector, under the
     * execution claim — the first point at which writing the collector cannot
     * land on a predecessor's. `set_host_orchestrated` rides along because the
     * collector's initialize() reads it when it sizes the orch phase pool, and
     * that now runs from the same launch arming.
     */
    void publish_host_phase_run_to_collector(uint32_t pipeline_slot) noexcept;
    /**
     * Write this pass's per-event host phase records under `output_prefix`.
     *
     * Host-only: every phase recorded is produced on the host during bind and the
     * store is finished before launch, so a caller that never reaches the device
     * can still produce the artifact. The store writes a pass at most once, so
     * every path that can end a run may call this unconditionally.
     */
    void write_host_phase_records_artifact(const std::string &output_prefix, uint32_t pipeline_slot);

    /**
     * Latch the part of this run's config a device-context query answers from.
     * The c_api applies it only when no active run can observe the runner-global
     * state. Defined in the .cpp so this header does not need the full CallConfig
     * definition.
     */
    void apply_call_config(const CallConfig &config);

    /**
     * Directory under which all diagnostic artifacts
     * (chip_swimlane_records.json / args_dump/ / pmu.csv) land. Required
     * (non-empty) when any diagnostic is enabled; `CallConfig::validate()`
     * enforces this contract upstream.
     */
    void set_output_prefix(const char *prefix) { output_prefix_ = (prefix != nullptr) ? prefix : ""; }
    const std::string &output_prefix() const { return output_prefix_; }

protected:
    // Ctor is protected: this class is for inheritance only — direct
    // instantiation (`new DeviceRunnerBase()`) is a compile error. The
    // public virtual dtor above lets the shared c_api delete through a
    // base pointer safely.
    DeviceRunnerBase();

    /**
     * `DeviceArena` callback trampolines bridging from C-style
     * `void *(void *ctx, size_t)` / `void (void *ctx, void *)` to this runner.
     * The `ctx` opaque pointer passed at arena construction time is the runner,
     * not the allocator: a context with a latched workspace budget routes its
     * arena backing through the ledger that owns those blocks, and one without
     * a budget reaches the same allocator calls it always did.
     */
    static void *arena_alloc_trampoline(void *ctx, std::size_t size) {
        return static_cast<DeviceRunnerBase *>(ctx)->acquire_arena_backing(size);
    }
    static void arena_free_trampoline(void *ctx, void *p) {
        static_cast<DeviceRunnerBase *>(ctx)->release_arena_backing(p);
    }

    /**
     * Arena backing acquisition and hand-back for one bank region.
     *
     * Unmanaged: the allocator, as before. Managed: the workspace ledger, which
     * may hand back a block an earlier generation still fits and no consumer
     * references, and which treats the hand-back as dropping this run's
     * reference rather than as permission to free.
     */
    void *acquire_arena_backing(std::size_t size);
    void release_arena_backing(void *p);

    /** The runner and bank one arena setup announces its regions against. */
    struct ArenaRegionAnnounce {
        DeviceRunnerBase *runner;
        uint32_t bank;
    };

    /**
     * Record that a bank region has given up the block at `base`.
     *
     * A managed block outlives the arena that was using it, so the arena's own
     * free callback cannot end its ownership — and the same callback serves a
     * stage abort, a superseded backing and a teardown, which mean different
     * things here. This is the transaction telling the ledger which of them
     * happened, so a claim that ends stops holding budget: an aborted staging
     * never became a generation anybody named, and a detached region publishes
     * no address at all. Neither drops a run reference or lifts a quarantine,
     * so the bytes still wait for their last true consumer.
     */
    void note_arena_region_disposition(uint32_t arena_bank, ArenaRegionDisposition what, void *base);

    /**
     * Register this plan as a consumer of every region the bank now publishes.
     *
     * A region whose existing capacity was enough allocates nothing, so it
     * reaches no allocation callback — and a plan that read and wrote it
     * without registering would let a later growth treat those bytes as free
     * to overwrite.
     *
     * @return 0, or PTO_RUNTIME_ERR_INTERNAL when a reference could not be
     *         recorded — refused rather than silently unprotected
     */
    int reference_bank_arenas(uint32_t arena_bank, const ArenaRegionRequest *requests, std::size_t count);

    /**
     * Configure STARS op execution timeout (once per DeviceRunner lifetime).
     *
     * Called on first device attach to set the hardware-level AICore op
     * execution timeout via `aclrtSetOpExecuteTimeOutV2`. The actual
     * timeout may differ from the requested value due to hardware timer
     * granularity.
     */
    void configure_aicore_op_timeout();

    PersistentArgsOps persistent_args_ops();
    int register_callable_on_device(int32_t callable_id, rtStream_t control_stream);

    /**
     * Load AICPU SO and initialize device args. Called from
     * `ensure_device_initialized()` after the persistent streams are
     * created. Reads `aicpu_so_binary_` / `dispatcher_so_binary_` off
     * the runner; releases both host buffers on success.
     *
     * @return 0 on success, error code on failure.
     */
    int ensure_binaries_loaded(rtStream_t control_stream);

    /**
     * Initial launch of `simpler_aicpu_init`, latching the invariants (orch
     * device id, log config, provisioned async-DMA workspace addresses) into the
     * resident AICPU SO globals. Idempotent via `aicpu_init_launched_`; called
     * from `ensure_device_initialized()` after the binaries are loaded and the
     * workspaces are provisioned, so one launch publishes everything.
     *
     * @return 0 on success, error code on failure.
     */
    int ensure_aicpu_init_launched(rtStream_t control_stream);

    /**
     * Provision the async-DMA workspaces this Worker asked for (see
     * `set_dma_workspace_request`) and record their device addresses in
     * `dma_workspace_addr_`, ready for `ensure_aicpu_init_launched()` to
     * publish. Idempotent: a runner that already holds a provider handle, and
     * one whose request declined the only declinable engine on a device that
     * supports nothing else, both no-op. The handle is released by
     * `finalize_common()`, including on a later step's failure.
     *
     * @return 0 on success, negative on unsupported/failed provisioning.
     */
    int ensure_dma_workspace_provisioned();

    /**
     * Walk the SDMA control path once, after `ensure_aicpu_init_launched()` has
     * published the workspace addresses. No-op without a provisioned workspace.
     * One-shot via `sdma_warmed_`, which also releases the warmup ELF bytes.
     *
     * @return 0 on success or a skipped warmup, error code when the warmup
     *         faulted the card.
     */
    int ensure_dma_workspace_warmed();

    /**
     * Query the maximum block_dim the stream can host.
     *
     * Uses `aclrtGetStreamResLimit(CUBE_CORE / VECTOR_CORE)` and
     * returns `min(cube / AIC_PER_BLOCKDIM, vector / AIV_PER_BLOCKDIM)`,
     * capped by `PLATFORM_MAX_BLOCKDIM`. Falls back to the static cap
     * when the query is unavailable or reports no cores.
     *
     * If non-null, `out_cube` / `out_vector` receive the raw ACL limits
     * when the query succeeded, or 0 when it failed. Callers use this
     * to distinguish the ACL-unavailable fallback path from the
     * success path in error logs.
     */
    int query_max_block_dim(rtStream_t stream, uint32_t *out_cube = nullptr, uint32_t *out_vector = nullptr);

    // ---- execution sub-sequence helpers ---------------------------------
    //
    // Each arch keeps the heavily-divergent middle (register
    // address setup, profiling flag building, init_*, collector start /
    // teardown, dep_gen, ffts setup, kernel launches). These helpers
    // cover the byte-identical sub-sequences at the head and tail.

    /**
     * Validate the caller's `launch_aicpu_num` against
     * `PLATFORM_MAX_AICPU_THREADS`. Returns 0 on success, -1 on
     * out-of-range with a logged error.
     */
    int validate_launch_aicpu_num(int launch_aicpu_num);

    /**
     * Resolve the active AICPU thread count for partial-good tolerance.
     * requested == 0 means auto (use arch_default = 1 orch + N sched); the
     * result is clamped to `usable` (the probed AICPU count — PG/OS cores are
     * absent from OCCUPY) so a degraded die runs with fewer schedulers, and
     * returns <0 if usable < 2 (need >=1 orch + >=1 sched). Returns the active
     * total otherwise.
     */
    int resolve_aicpu_thread_num(int requested, int usable, int arch_default);

    /**
     * Prepare the device-phase/task-timing buffer for one run, in that run's
     * pipeline slot. Capture-disabled runs publish a null device base.
     * Capture-enabled runs allocate the slot's buffer lazily, reset every
     * record, and publish the base for AICPU stamping. Allocation or reset
     * failure is non-fatal; the base stays null and timing reads as 0.
     */
    void ensure_device_wall_buffer(uint32_t pipeline_slot, KernelArgsHelper &kernel_args);
    int arm_device_wall_buffer(uint32_t pipeline_slot, KernelArgsHelper &kernel_args);

    /**
     * Point this run's KernelArgs at its slot's result region and stamp the run
     * epoch the device must publish. Allocated lazily per slot and, unlike the
     * timing buffer, never gated on diagnostics. Returns non-zero when the
     * region could not be provided, which the caller must treat as a prepare
     * failure: continuing would launch a run whose device side has nowhere to
     * put its result, and whose region still holds a predecessor's payload.
     */
    int ensure_device_run_result_region(uint32_t pipeline_slot, uint64_t run_epoch, KernelArgsHelper &kernel_args);

    /**
     * Resolve and reserve this run's chip-swimlane terminal-snapshot bank, and
     * return its device address for KernelArgs.
     *
     * The bank is the slice of the collector's retained region into which each
     * producer copies its settled record totals at its last flush, so those
     * totals survive the next run's counter reset. Indexed by the run's actual
     * pipeline slot; returns 0 whenever no bank can be resolved (swimlane off,
     * collector not initialized, slot out of range, or no run identity), which
     * the device reads as "publish no snapshot".
     *
     * Diagnostic-only and never a prepare failure: a run with no bank simply
     * reports no retained snapshot, and the existing reconcile remains the
     * authoritative accounting either way.
     */
    uint64_t arm_chip_swimlane_run_terminal_bank(uint32_t pipeline_slot, uint64_t run_epoch);

    /**
     * Resolve this run's block_dim: every cluster the device has, i.e.
     * the cached `max_block_dim_`. A run is never narrower than the
     * device — orchestration sizes its cohorts from
     * `rt_available_cluster_count()` instead.
     *
     * Reads the cached ceiling only — no ACL call, so it is safe to run
     * at bind time, before any stream work for the run.
     *
     * Returns the resolved block_dim on success, -1 if the ceiling was
     * never latched. The value is latched into runner execution state only
     * when the prepared Runtime is launched.
     */
    int resolve_block_dim();

    /**
     * Wait for an explicit AICPU/AICore stream pair (AICPU first) with the
     * resolved stream-sync timeout. Distinguishes the timeout sentinel
     * `ACL_ERROR_RT_STREAM_SYNC_TIMEOUT` with a stream-id and (device,
     * block_dim) context in the log. Returns the first non-zero rc encountered.
     *
     * Waits for everything queued on the pair, so a run's own completion is
     * established by `wait_run_fence` below instead. This stays the bounded
     * wait for work no boundary covers, and the call `wait_run_fence` reads the
     * device's verdict with once completion is settled.
     */
    int sync_stream_pair(rtStream_t aicpu_stream, rtStream_t aicore_stream);

    // ---- Per-run completion fences ---------------------------------------
    //
    // A stream query or stream wait answers a question about a queue, so it
    // covers everything queued on it. These helpers answer the same question
    // about one run, from the two boundary events that run recorded after its
    // own kernels. See host/run_completion_fence.h for the ownership model and
    // docs/design/run-completion-fence.md for why each fallback below exists.

    /** One pipeline slot's fence. Slots are indexed as everywhere else. */
    RunCompletionFence &run_fence(uint32_t pipeline_slot) { return *run_fences_[pipeline_slot]; }

    /**
     * Take this run's slot fence, committing its events. Belongs in the launch
     * arming prologue: creation can fail, and there it fails while the run has
     * still submitted nothing and can roll back.
     */
    int arm_run_fence(const PreparedExecution &prepared);

    /**
     * Record one stream's completion boundary, immediately after that stream's
     * kernel submission was accepted. Notes the submission first, so a record
     * failure cannot be mistaken for a kernel that never launched.
     */
    int record_run_boundary(const PreparedExecution &prepared, RunCompletionFence::StreamRole role, rtStream_t stream);

    /**
     * Query this run's boundaries without waiting, as one of the
     * SIMPLER_NATIVE_RUN_POLL_* values.
     *
     * A run holding submitted work that no boundary covers cannot be decided
     * from its own events, so it falls back to querying the streams — the
     * queues are the only remaining evidence, and a partial launch is not a
     * path that has a successor queued behind it.
     *
     * A run the boundaries prove complete is additionally checked against the
     * streams' sticky error state, which keeps what a whole-pair query used to
     * report about a stream left in error. That check is not a device-exception
     * detector; see `wait_run_fence`.
     */
    int poll_run_fence(const PreparedExecution &prepared, rtStream_t aicpu_stream, rtStream_t aicore_stream);

    /**
     * Establish that this run finished and what the device made of it.
     *
     * Completion comes from the run's own boundaries, waited with the resolved
     * stream-sync timeout and the event-timeout sentinel logged with the same
     * (device, block_dim) context `sync_stream_pair` logs. A run no boundary
     * covers has no such proof and falls back to the bounded whole-stream wait;
     * expiry proves nothing about quiescence either way, so the caller's
     * recover-or-mark-unusable policy still owns the non-zero rc.
     *
     * The device's verdict is then read from this run's own published record
     * where that record decides it, and from a stream synchronize on every other
     * shape — the only call on this SDK that produces one, see the measurement
     * in the definition. So the branch a successful run takes touches neither
     * stream of the pair, which is what lets a successor stay queued behind it.
     */
    int wait_run_fence(const PreparedExecution &prepared, rtStream_t aicpu_stream, rtStream_t aicore_stream);

    /**
     * Give up this run's arming. Idempotent and a no-op for a run that never
     * armed, so every teardown path may call it.
     */
    void retire_run_fence(const PreparedExecution &prepared) noexcept;

    // ---- Queued waits on this run's boundaries ---------------------------
    //
    // A run whose whole operator is one boundary queues a wait for its own
    // AICore boundary ahead of recording its AICPU one, and a joined successor
    // queues a wait for that AICPU boundary. Both are references on this run's
    // fence, and both must be discharged before the fence may retire. See
    // host/queued_stream_waits.h for the evidence each one takes.

    /**
     * Reserve this run's own intra-run wait and hand back its AICore boundary
     * event. The wait is queued by the caller into this run's AICPU stream,
     * ahead of the AICPU boundary record that then covers the whole operator.
     */
    int open_own_boundary_wait(const PreparedExecution &prepared, void **core_done_out);

    /**
     * Reserve a joined successor's cross-run wait and hand back the
     * predecessor's whole-operator boundary event.
     *
     * Refuses unless this run carries a join naming a predecessor whose fence
     * still owns that identity and has recorded the boundary, which is what
     * makes a queued wait unable to name an event nothing will record. It also
     * refuses unless that predecessor published a whole-operator boundary: its
     * AICPU boundary alone leaves the AICore kernel's tail uncovered, so a wait
     * on it would order the successor ahead of work still running.
     */
    int open_cross_run_wait(const PreparedExecution &prepared, void **predecessor_boundary_out);

    /**
     * Publish that this run's AICPU boundary covers its whole operator, which is
     * what makes the run joinable. Called once the intra-run wait is committed
     * and the boundary behind it recorded — never on a launch that skipped or
     * failed either step, so a run whose construction degraded is simply not
     * joined rather than joined unsafely.
     */
    void note_whole_operator_boundary(const PreparedExecution &prepared);
    /** Withdraw that publication as the run ends. */
    void clear_whole_operator_boundary(const PreparedExecution &prepared) noexcept;

    /** Promote a queued wait, or drop one that was never queued. */
    int commit_boundary_wait(QueuedStreamWaits::Shape shape, const PreparedExecution &prepared);
    int revoke_boundary_wait(QueuedStreamWaits::Shape shape, const PreparedExecution &prepared);

    /**
     * Record the proof event of this run's cross-run wait into the stream that
     * holds the wait. A failure costs only the cheap proof: the wait is queued,
     * and the reference falls back to the quiescence rung at discharge.
     */
    int record_cross_run_proof(const PreparedExecution &prepared, rtStream_t waiter_stream);

    /**
     * Retire every wait naming a boundary of this run, before anything it owns
     * is released.
     *
     * `boundaries_complete` is the caller's own observation that this run's two
     * boundaries completed. The rungs are tried in order: the run's own
     * boundaries, then a cross-run wait's proof event, then a stream-pair
     * synchronize this call itself issues. Exhausting them leaves a queued wait
     * naming an event that must not be destroyed, so the device is recovered or
     * marked unusable and the first error is returned.
     */
    int discharge_boundary_waits(
        const PreparedExecution &prepared, bool boundaries_complete, rtStream_t aicpu_stream, rtStream_t aicore_stream
    );

    /** The same ladder from a path that cannot report, poisoning on failure. */
    void discharge_boundary_waits_noexcept(
        const PreparedExecution &prepared, bool boundaries_complete, rtStream_t aicpu_stream, rtStream_t aicore_stream
    ) noexcept;

    /**
     * Take this runner's reference on the process's exception-notification
     * callback, once. Idempotent: the slot is process-global and refcounted
     * elsewhere, so a runner holds at most one reference no matter how often
     * its device bring-up runs.
     */
    int acquire_device_fault_monitor();

    /** Drop it. A no-op for a runner that never took one. */
    void release_device_fault_monitor() noexcept;

    /**
     * Retire this runner's per-generation fault evidence after a **confirmed**
     * device reset, and re-register the callback if this runner holds the monitor.
     *
     * Named for the retirement because that is the part this always does. A
     * runner whose `acquire` failed still runs work and still accumulates a
     * generation's stream ids, so the reset has to invalidate them whether or not
     * a callback was ever installed — the monitor's own fence is the conditional
     * half. Whether a registration survives a force reset is unmeasured, so it is
     * remade rather than assumed either way; registering twice is harmless.
     *
     * Returns the monitor's re-install rc, or 0 when no monitor is held.
     */
    int retire_device_generation_after_confirmed_reset() noexcept;

    /**
     * Read and reduce this slot's device-phase/task-timing records after stream
     * sync, into that slot's `DeviceRunTiming`. A D2H failure is a soft warning
     * and leaves the record zeroed, as do a capture-disabled run, a missing
     * buffer, and a run whose arming failed — the launch path continues after a
     * failed arm, and that run has no stamps of its own to read.
     */
    void read_device_wall_ns(uint32_t pipeline_slot);

    /**
     * H2D the Runtime struct via the supplied per-execution kernel arguments. Log config
     * and device ordinal are NOT published here: they are per-device invariants
     * latched once into the AICPU SO globals by `simpler_aicpu_init`
     * (`ensure_aicpu_init_launched`) at device init, not carried per-run on
     * KernelArgs.
     *
     * @return 0 on success, the underlying prepare/publish rc on failure.
     */
    int init_runtime_args_with_metadata(Runtime &runtime, KernelArgsHelper &kernel_args, SlotPersistentArgs &slot);

    /**
     * Open this run's collection window on the four shared diagnostics
     * collectors (`chip_swimlane_collector_`, `dump_collector_`,
     * `pmu_collector_`, `scope_stats_collector_`) that it enables, and start
     * their mgmt + poll threads. Each block is gated on `dfx`, this run's own
     * configuration, not on the runner's members.
     *
     * The collectors are resident and serve every run, so opening the window is
     * destructive: it drops the previous run's records and counters and
     * republishes the level the device reads. That is why it happens here, at
     * launch, rather than during preparation — this is the first point the run
     * holds the execution claim, so it is the first point at which no other run
     * is executing against those collectors.
     *
     * Each spawned thread is bound to `device_id_` via `create_thread`.
     *
     * Subclasses with arch-specific collectors (`dep_gen_collector_`) call
     * this helper and then open and start their own. The sim base carries the
     * same split.
     *
     * Returns non-zero when a collector that retains runs would not admit this
     * one. Nothing execution-visible has been submitted at that point, so the
     * caller propagates the rc and the run is rolled back rather than collected
     * by the destructive single-run path.
     */
    int start_shared_collectors_for_run(const DfxRunConfig &dfx, uint64_t run_epoch);

    /**
     * Give back what `start_shared_collectors_for_run` admitted for a run that
     * ended up submitting nothing.
     *
     * Call this on, and only on, a launch transaction that reached
     * `LaunchProgress::NotStarted`: a partial or ambiguous submission may have
     * left a device-side producer writing into that run's slot, and nothing may
     * be freed under it. A no-op with retention off, and a no-op for a run that
     * was never admitted, so a failure before admission cannot reach a
     * predecessor's records.
     */
    void withdraw_unlaunched_collectors_for_run(const DfxRunConfig &dfx, uint64_t run_epoch) noexcept;

    /**
     * Close one run's PMU window: either today's drain and reconcile, or, when
     * PMU retains runs, the claim-time snapshot that hands the epoch to its
     * background writer.
     */
    void close_pmu_run_boundary(const DfxRunConfig &dfx, uint64_t run_epoch, bool device_execution_complete);

    /**
     * Tear down the four shared diagnostics collectors after the launched
     * kernels have synced. Each block is gated on `dfx`, this run's own
     * configuration, and does: stop() → reconcile_counters() →
     * export step (`chip_swimlane` writes swimlane JSON via
     * `read_phase_header_metadata` + `export_swimlane_json`; `dump`
     * writes dump files; `pmu` has no export step beyond reconcile;
     * `scope_stats` writes JSONL).
     *
     * Subclasses with arch-specific collectors (`dep_gen_collector_` + its
     * `dep_gen_replay_emit_deps_json` export) inline their own teardown after
     * calling this helper. The sim base carries the same split.
     *
     * `run_epoch` identifies the run whose retained terminal snapshot is read
     * back, which happens only when `device_execution_complete` says the caller
     * observed this run's completion fence.
     */
    void teardown_shared_collectors_after_run(
        const DfxRunConfig &dfx, uint32_t pipeline_slot, uint64_t run_epoch, bool device_execution_complete
    );

    /**
     * The core and AICPU-thread counts a resident collector's pools were built
     * for.
     *
     * Collector pool topology is derived from those counts: buffer seeding
     * covers pools [0, aicpu_thread_num), and a core's recycled lane is
     * `(core / PLATFORM_CORES_PER_BLOCKDIM) % aicpu_thread_num`. A collector
     * that stays initialized across runs therefore holds pools shaped for the
     * run that built them, so a later run with different counts must rebuild
     * them rather than reuse pools whose lanes it maps differently.
     */
    struct CollectorShape {
        bool latched{false};
        int num_aicore{0};
        int aicpu_thread_num{0};
        int launch_aicpu_num{0};
    };

    /**
     * True once collectors are built and this run's counts differ from theirs,
     * i.e. their pools must be released and rebuilt before this run seeds them.
     *
     * The release frees device memory the collectors are holding, so it is only
     * safe while no other run is executing against them. Nothing here enforces
     * that. What guarantees it is the caller: `arm_collectors_for_run()` is the
     * sole user, and it runs from the launch arming, under the execution claim.
     * Do not move the call back into preparation — a prepared successor overlaps
     * its predecessor's device window, so this would free pools that predecessor
     * is still writing into.
     */
    bool collector_shape_is_stale(int num_aicore, int aicpu_thread_num, int launch_aicpu_num) const {
        return collector_shape_.latched &&
               (collector_shape_.num_aicore != num_aicore || collector_shape_.aicpu_thread_num != aicpu_thread_num ||
                collector_shape_.launch_aicpu_num != launch_aicpu_num);
    }

    void latch_collector_shape(int num_aicore, int aicpu_thread_num, int launch_aicpu_num) {
        collector_shape_ = CollectorShape{true, num_aicore, aicpu_thread_num, launch_aicpu_num};
    }

    /** Called by the subclass's finalize_collectors(): no pools are built now. */
    void clear_collector_shape() { collector_shape_ = CollectorShape{}; }

    CollectorShape collector_shape_{};

    /**
     * Shared body of `finalize()`. Each arch subclass's `finalize()`
     * handles: (a) the early-return + thread attach prologue, (b) any
     * arch-specific collector teardown (e.g. a2a3's `dep_gen_collector_`),
     * and (c) the arch-specific device reset (a2a3's ACL/rt branch vs
     * a5's `rtDeviceReset`). Everything else lives here:
     *
     *   - rtStreamDestroy for both persistent streams
     *   - aicore_bin_handle_ + binaries_loaded_ reset
     *   - chip_callable_buffers_ free + clear
     *   - callables_ dlclose-on-hbg + clear + aicpu counter reset
     *   - 3 arenas release + cached size reset
     *   - device_wall_dev_ptr_ free (before mem_alloc_.finalize)
     *   - mem_alloc_.finalize
     *   - block_dim_, worker_count_, aicore_kernel_binary_ reset
     *
     * Device-wall free order is normalized to "before mem_alloc_.finalize"
     * (matching the prior a5 ordering). The prior a2a3 ordering freed it
     * AFTER `mem_alloc_.finalize` + `rtDeviceReset`, which routed through
     * an already-finalized allocator on a torn-down device context — a
     * latent UAF / no-op. This refactor fixes that.
     *
     * @return 0 on success, first nonzero rc encountered otherwise.
     */
    int finalize_common();

    /**
     * Retire loader state that a program close's device teardown has left
     * unreachable, whichever way that teardown went.
     *
     * `LoadAicpuOp::Finalize()` keeps its binary handle when `rtsBinaryUnload`
     * fails, so that an owner able to retry it survives. Neither outcome of a
     * program close leaves that retry reachable, so both forget the handle:
     *
     *   - `reset_confirmed` — the reset ended the generation the handle
     *     belonged to, so it names a binary that no longer exists. Forgetting
     *     it is what keeps `init -> finalize -> init` working on one context.
     *   - otherwise — the device's state is unconfirmed, and this close has
     *     already released every other owner and is about to clear
     *     `device_id_`. Forgetting the handle is what keeps `~LoadAicpuOp`
     *     from issuing an unreported unload against that device.
     *
     * Neither is a successful release and neither is reported as one: the
     * error this close already returned to the caller is the last word on it.
     *
     * A no-op on a kernel context, which resets nothing — its retained handle
     * stays valid and its explicit close is the only thing that may retire it.
     * That close cannot reach here today, because a retained handle makes
     * `finalize_common()` return non-zero and the arch tail returns early on a
     * kernel latch; the guard enforces the invariant rather than resting on
     * that.
     */
    void retire_loader_after_device_teardown(bool reset_confirmed) {
        if (execution_mode_latch_.is_kernel()) return;
        if (!load_aicpu_op_.has_live_resources()) return;
        if (reset_confirmed) {
            LOG_WARN("finalize: device reset ended the generation of the retained AICPU binary handle; forgetting it");
        } else {
            LOG_ERROR(
                "finalize: device reset did not complete and the AICPU binary handle is still retained; forgetting it "
                "without unloading — the binary may still be resident, and no further call is issued against this "
                "device"
            );
        }
        load_aicpu_op_.ForgetWithoutUnload();
    }

    void release_graph_definition_blocks();

    /**
     * Release every slot's retained scheduler-state storage.
     *
     * Returns the first failing free's code, having attempted every block —
     * the live one and any failed-release record — so one failure cannot
     * strand the rest. Each outcome is recorded before the slot's entry is
     * cleared, and an address the allocator could not free stays in its
     * tracking map with its bytes still committed, which is what hands that
     * block to the allocator's own terminal sweep rather than dropping it.
     * Clearing the entry afterwards is what makes a second close a no-op; it
     * is not a claim that the bytes went back.
     */
    int release_scheduler_state_storage();

    /**
     * Drop every slot's retained scheduler-state storage without a device call.
     *
     * The fatal counterpart of release_scheduler_state_storage(): a force reset
     * has already invalidated the device generation these addresses belong to,
     * so the host-side bookkeeping and staging are dropped and no allocator or
     * device function is entered.
     */
    void abandon_scheduler_state_storage();

    /** Drop every retained host SM mirror, returning its pages to the allocator. */
    void release_sm_mirrors();
    void release_run_image_stagings();

    /**
     * Drop the retained graph-definition blocks without freeing the device side.
     *
     * The fatal counterpart of release_graph_definition_blocks(): a force reset
     * has already invalidated every device allocation, so only the host-side
     * bookkeeping and staging are dropped.
     */
    void abandon_graph_definition_blocks();

    /**
     * Clear host-side ownership after a fatal device failure without issuing
     * per-resource RTS calls. The caller must first attempt a force reset.
     */
    int abandon_common_after_device_failure();

    int finalize_common_impl(bool abandon_device_resources);

    /**
     * Stamp the active callable_id onto a Runtime so the AICPU knows which
     * orch_so_table_ slot to dispatch. The orch SO itself was already delivered
     * device-side at register time (launch_device_register), so nothing else
     * needs rewriting per run.
     *
     * @param runtime  Runtime whose active callable_id will be set.
     * @return 0 on success, non-zero on failure.
     */
    int prepare_orch_so(Runtime &runtime);
    int stamp_orch_so(Runtime &runtime, int32_t callable_id);

    // ---- Group D state shared by both a2a3 and a5 -------------------------
    //
    // Chip-callable buffer pool. Keyed by FNV-1a 64-bit content hash of
    // the ChipCallable bytes. Each entry owns one device GM allocation
    // holding the entire ChipCallable buffer (header + storage_, with
    // each child's resolved_addr_ fixed up to its post-H2D device
    // address) followed by that callable's function tables. Identical buffer
    // bytes share one entry across cids; refcount drops on unregister and
    // finalize bulk-frees any leftovers.
    struct ChipCallableBuffer {
        uint64_t chip_dev{0};  // device GM address of the ChipCallable header
        size_t total_size{0};  // byte size of the device allocation, tables included
        int refcount{0};
        // The entry exists only so a release whose free failed still has an
        // owner to retry through, and names no callable a caller may use: its
        // contents either never reached the device or are already released. The
        // dedup lookup skips it, and finalize retries the free.
        bool release_pending{false};
        // The callable's func_id -> CoreCallable object address table, dense
        // over [0, table_len) and the host source the device copy was made
        // from. `table_len` is one past the callable's largest child func_id,
        // never RUNTIME_MAX_FUNC_ID: the content hash covers the child func_ids
        // and offsets, so every cid sharing this entry has exactly this table.
        std::vector<uint64_t> object_table;
        uint64_t object_table_dev{0};
        // The same domain resolved to kernel-entry addresses, present only on a
        // runtime whose device scheduler dispatches from them.
        uint64_t entry_table_dev{0};
        uint32_t table_len{0};
    };
    std::unordered_map<uint64_t, ChipCallableBuffer> chip_callable_buffers_;

    // Per-callable_id registered state.
    //
    // `callables_` maps the caller-stable callable_id to the chip buffer
    // lease, orch SO slice + symbol names needed to launch it.
    // `aicpu_seen_callable_ids_` tracks which ids have completed a successful
    // AICPU SO load for the monotonic dlopen counter.
    struct CallableState {
        // trb path (AICPU dlopens orch SO from device buffer)
        // Orchestration ELF Build-ID returned by callable_hash(); distinct from
        // chip_buffer_hash, which keys the retained buffer.
        uint64_t hash{0};
        uint64_t chip_buffer_hash{0};
        uint64_t aicore_image_hash{0};
        uint64_t dev_orch_so_addr{0};
        size_t dev_orch_so_size{0};
        std::string func_name;
        std::string config_name;
        // common
        std::vector<ArgDirection> signature;
        // hbg path (host already dlopen'd the orch SO)
        void *host_dlopen_handle{nullptr};
        void *host_orch_func_ptr{nullptr};
    };
    std::unordered_map<int32_t, CallableState> callables_;
    // Opaque provider handle from dma_workspace_provision(), owned for the
    // Worker's life and released by finalize_common(). Null unless the Worker
    // was created with SDMA enabled.
    void *dma_workspace_handle_{nullptr};
    // Provisioned async-DMA workspace device addresses, indexed by
    // DmaWorkspaceKind. Published into InitArgs by ensure_aicpu_init_launched()
    // so the resident AICPU SO latches them into g_dma_workspace_addr; the
    // scheduler prefills each core's GlobalContext from there. All-zero until a
    // Worker opts into SDMA via set_dma_workspace_request().
    uint64_t dma_workspace_addr_[DMA_WORKSPACE_KIND_COUNT]{};
    // This Worker's async-DMA request, recorded by set_dma_workspace_request()
    // before device bring-up. `sdma_warmup_binary_` is released once
    // ensure_dma_workspace_warmed() has consumed it.
    bool sdma_requested_{false};
    bool sdma_warmed_{false};
    std::vector<uint8_t> sdma_warmup_binary_;
    std::unordered_set<int32_t> aicpu_seen_callable_ids_;
    // Monotonic count of successful AICPU dlopens (incremented after prewarm
    // or first-run fallback succeeds; never decremented). Diverges from
    // aicpu_seen_callable_ids_.size() once any cid is unregistered and
    // re-registered. Exposed via `aicpu_dlopen_count()` for tests.
    size_t aicpu_dlopen_total_{0};
    // Monotonic count of host-side dlopens triggered (incremented on
    // every `record_host_orch_callable` call; never decremented).
    // Same re-register semantics as `aicpu_dlopen_total_`, but for hbg
    // variants.
    size_t host_dlopen_total_{0};
    struct NativeRunReservation {
        const void *owner{nullptr};
        uint32_t pipeline_slot{0};
        uint32_t arena_bank{0};
        bool permits_prepared_successor{false};
    };
    mutable std::mutex native_run_mu_;
    std::array<NativeRunReservation, PTO_PIPELINE_MAX_DEPTH> native_run_reservations_{};
    struct NativeRunClaim {
        const void *owner{nullptr};
        NativeRunIdentity identity{};
    };
    // The runs holding the native execution claim, in the order they took it.
    //
    // The claim stays exclusive by default. A second holder is admitted only to
    // a caller that presents a `NativeRunJoin` naming the newest holder and its
    // identity, which is why the identity is recorded here rather than only the
    // owner pointer: the array bounds the resource, and the recorded identity is
    // what makes "this successor was ordered behind exactly that predecessor" a
    // checked fact instead of an inference from a free slot.
    //
    // A predecessor keeps its claim until it finalizes, and a release names the
    // run that is leaving, so an out-of-order release — a fast successor
    // finishing first — removes only its own entry and preserves the order of
    // the rest.
    std::array<NativeRunClaim, PTO_PIPELINE_MAX_DEPTH> active_native_runs_{};
    size_t active_native_run_count_{0};
    /** Index of `owner` among the claim holders, or the array size when absent. */
    size_t native_run_claim_index(const void *owner) const;

    // ---- State shared by both a2a3 and a5 ---------------------------------
    //
    // Which device this context is on — not a claim of ownership, which the
    // execution-mode latch carries instead. Written once before any prepare,
    // execution or collector thread attaches: `attach_current_thread` writes
    // it for a program context and `adopt_borrowed_device` for a kernel one, both
    // guarded on the still-unset value, so repeated same-value writes from
    // later-attaching threads cannot race.
    int device_id_{-1};
    // This context's execution identity. Write-once: the first init entry to
    // run latches it, and it never changes afterwards.
    ExecutionModeLatch execution_mode_latch_;
    KernelExecutionState kernel_exec_state_;
    PersistentKernelArgs persistent_args_;
    Runtime kernel_runtime_;
    int block_dim_{0};
    int cores_per_blockdim_{PLATFORM_CORES_PER_BLOCKDIM};
    int worker_count_{0};  // Stored for print_handshake_results

    // This device's block_dim ceiling and the raw ACL core limits behind it,
    // resolved once against the persistent AICore stream in
    // ensure_device_initialized(). Nothing in this codebase calls
    // aclrtSetStreamResLimit, so the limits hold for that stream's lifetime;
    // finalize_common() clears them along with the stream.
    int max_block_dim_{0};
    uint32_t max_cube_cores_{0};
    uint32_t max_vector_cores_{0};
    HostRuntimeTimeoutConfig timeout_config_{PLATFORM_OP_EXECUTE_TIMEOUT_US, PLATFORM_STREAM_SYNC_TIMEOUT_MS};

    // Executor binaries — populated once via `set_executors()` during
    // simpler_init. `aicore_kernel_binary_` is consumed once by
    // `launch_aicore_kernel()` (`rtRegisterAllKernel` returns
    // `aicore_bin_handle_`, cached and reused on every subsequent
    // launch). Caching is required: CANN has no public
    // `rtUnregisterAllKernel`, so re-registering on every run would pin
    // another device-side copy of the ELF and quickly exhaust HBM
    // (manifested in CI as 207001 at `rtKernelLaunchWithHandleV2` with
    // a 507899 cascade at `rtStreamCreate`). `aicpu_so_binary_` is
    // released by `ensure_binaries_loaded()` after bootstrap;
    // bootstrap is the only consumer and per-task launches go through
    // the cached `rtFuncHandle` on `LoadAicpuOp`, not the host bytes.
    std::vector<uint8_t> aicpu_so_binary_;
    std::vector<uint8_t> aicore_kernel_binary_;
    // AICore kernel handle from `rtRegisterAllKernel` — lazily
    // populated by the subclass's `launch_aicore_kernel()` and reused
    // across all runs. `nullptr` means not yet registered. Reset to
    // `nullptr` in `finalize()`; CANN releases the device-side state
    // implicitly when the device context tears down.
    void *aicore_bin_handle_{nullptr};
    // SDMA warmup ELF handle from `rtRegisterAllKernel`, kept separate from
    // `aicore_bin_handle_` because it is a different (vector-only) binary. Only
    // ever registered once, during provisioning. Reset the same way in
    // `finalize()`.
    void *sdma_warmup_bin_handle_{nullptr};
    // Dispatcher SO bytes — populated once via `set_dispatcher_binary()`
    // during simpler_init. Consumed exclusively by
    // `BootstrapDispatcher` on the first run and released by
    // `ensure_binaries_loaded()` right after. Empty buffer is permitted
    // at init time (callers that drive `ChipWorker.init` without a
    // dispatcher path); `ensure_binaries_loaded()` then fails fast
    // with a clear message if/when bootstrap is actually attempted.
    std::vector<uint8_t> dispatcher_so_binary_;

    // AICPU op loader — handles dispatcher bootstrap and per-task launches.
    host::LoadAicpuOp load_aicpu_op_;

    MemoryAllocator mem_alloc_;

    // The device allocations a caller minted through this context, and which
    // runs still hold them — see host/caller_device_buffers.h. Only the
    // caller-facing mint records here, so an address this runner allocated for
    // itself is absent and cannot be named by a run's arguments.
    CallerDeviceBuffers caller_device_buffers_;

    // One budget and one ownership ledger for this context's workspace
    // regions. Off unless a caller latches a budget.
    WorkspaceManager workspace_;

    // Host mappings of child-memory allocations a host-side orchestrator has
    // touched — see HostApi acquire_child_memory_host_view. Keyed by allocation
    // base and dropped by that allocation's free, which is what keeps a cached
    // host VA from outliving its pages.
    ChildMemoryHostViewCache child_memory_host_views_;
    // Retained temporary buffer for device arguments, one per pipeline slot
    // (see HostApi get/set_retained_temp_buffer). Just a remembered
    // {addr, size} that the slot reuses across its runs and finalize frees;
    // the grow/slice logic lives in utils/retained_temp_bump.h.
    std::array<void *, PTO_PIPELINE_MAX_DEPTH> retained_temp_addrs_{};
    std::array<std::size_t, PTO_PIPELINE_MAX_DEPTH> retained_temp_sizes_{};
    // Graph Definition storage, one retained block per pipeline slot — see
    // HostApi acquire_graph_definition_block. `staging` is the host block the
    // run's Definition objects are packed into and stays allocated across runs,
    // so a bind neither acquires nor returns host memory for them; the device
    // side is the raw allocation plus the aligned address handed out. One block
    // per slot rather than one per Definition: every Definition of a run is
    // packed end to end and shipped by a single H2D, and every submission
    // references the device-resident copy of its own Definition.
    struct RetainedGraphBlock {
        void *allocation{nullptr};
        void *aligned_addr{nullptr};
        std::size_t capacity{0};
        std::vector<std::byte> staging;
    };
    std::array<RetainedGraphBlock, PTO_PIPELINE_MAX_DEPTH> graph_definition_blocks_{};
    // Scheduler-state storage, one retained pair per pipeline slot — see
    // HostApi acquire_scheduler_state_storage and utils/retained_scheduler_storage.h,
    // which holds the grow, alignment and failure rules. One pair per slot
    // because a slot's runs are serialized while two slots' are not.
    std::array<RetainedSchedulerStorage, PTO_PIPELINE_MAX_DEPTH> scheduler_state_storage_{};
    // Host mirror of the runtime shared memory, one retained buffer per pipeline
    // slot — see HostApi acquire_sm_mirror. A host-side orchestrator writes its
    // whole shared-memory image here and the bind ships the live prefix, so the
    // buffer is capacity-sized (tens of MB) and stays mapped across binds: one
    // buffer per slot rather than one per bind, because two binds in different
    // slots are in flight at once. `capacity` counts the raw block, which is over-allocated
    // by the requested alignment so the aligned address handed out has the
    // requested bytes behind it.
    //
    // The block is never value-initialized. The caller's layout is init-on-write
    // and it ships only the prefixes it wrote, so the resident set is the pages a
    // bind touches rather than the whole capacity — which is also why this is not a
    // std::vector: `resize` would zero every page of a capacity the caller writes
    // a fraction of, and would copy the old bytes on growth for a buffer whose
    // contents mean nothing between binds.
    struct RetainedSmMirror {
        std::unique_ptr<std::byte[]> storage;
        std::size_t capacity{0};
    };
    std::array<RetainedSmMirror, PTO_PIPELINE_MAX_DEPTH> sm_mirrors_{};

    // Host staging for the device execution image, one retained buffer per
    // pipeline slot — see HostApi acquire_run_image_staging. Same block shape
    // and the same grow-only retention as the mirror above; what differs is
    // what it holds and how long it has to hold it. A bind assembles the
    // bytes here and records where they go; the publication reads them
    // afterwards, so this buffer is what makes the source outlive the
    // preparation. Sized to the image a bind ships rather than to the
    // mirror's capacity.
    std::array<RetainedSmMirror, PTO_PIPELINE_MAX_DEPTH> run_image_stagings_{};

    // One independently committed set of the three pooled device regions. A
    // run reaches its set through the arena bank its lease selects, so
    // preparing one bank never mutates a region the active run is executing
    // out of. `cached_*` back `setup_static_arena`'s "fits" check: a later
    // init asking for an equal-or-smaller layout on an already-committed
    // arena reuses it instead of re-allocating.
    struct ArenaBank {
        ArenaBank(DeviceArena::AllocFn alloc, DeviceArena::FreeFn free_fn, void *ctx) :
            gm_heap(alloc, free_fn, ctx),
            gm_sm(alloc, free_fn, ctx),
            runtime_pool(alloc, free_fn, ctx) {}

        DeviceArena gm_heap;
        DeviceArena gm_sm;
        DeviceArena runtime_pool;
        size_t cached_gm_heap_size{0};
        size_t cached_gm_sm_size{0};
        size_t cached_runtime_arena_size{0};
    };
    // Held by pointer because DeviceArena is non-copyable and non-movable, so
    // the array cannot be brace-initialised without naming every bank.
    std::array<std::unique_ptr<ArenaBank>, PTO_PIPELINE_MAX_DEPTH> arena_banks_;
    ArenaBank &arena_bank(uint32_t bank_id) { return *arena_banks_[bank_id]; }

    // The device blocks each pipeline slot reuses across its runs. Committed on
    // a slot's first prepare and released in finalize(), so a steady-state run
    // rewrites their contents instead of reallocating them.
    std::array<SlotPersistentArgs, PTO_PIPELINE_MAX_DEPTH> slot_persistent_args_;

    // One completion fence per pipeline slot. The events are runner-owned for
    // the same reason the blocks above are: a reuse-capable event re-records
    // without a reset, so creating a pair per run would add device calls to
    // every dispatch. The per-run facts they carry are identity-bound, so a
    // slot's next run cannot read the previous one's completion. Held by
    // pointer because the fence is non-copyable, so the array cannot be
    // brace-initialised without naming every slot.
    std::array<std::unique_ptr<RunCompletionFence>, PTO_PIPELINE_MAX_DEPTH> run_fences_;

    // Every stream wait queued on one of those boundaries, across all slots.
    // One table rather than one per slot: a cross-run wait names two runs, and
    // the entry has to be reachable from the predecessor's drain, which is the
    // side that must not release anything while the wait is live.
    std::unique_ptr<QueuedStreamWaits> queued_waits_;

    // Passive device-timestamp markers at two stream positions the fences bracket. Separate
    // handles with a separate creation flag, because a timestamp needs a capability the
    // completion flag does not promise and the fence contract must not change to borrow it.
    std::unique_ptr<RunBoundaryMarks> boundary_marks_;

    // Which run of each slot has published a whole-operator AICPU boundary, or
    // an empty identity while that slot's run has not. Held beside the claim
    // because it is read by a *successor's* launch to decide whether the run it
    // was ordered behind can be joined at all.
    mutable std::mutex whole_operator_mu_;
    std::array<NativeRunIdentity, PTO_PIPELINE_MAX_DEPTH> whole_operator_boundaries_{};

    // What this runner's finalize() observed about its own device teardown.
    // Outlives nothing: `copy_teardown_report` must be called while the runner
    // is alive, which is why the C entry sits before context destruction.
    TeardownRecorder teardown_recorder_;

    // Whether this runner holds a reference on the process's fault-notification
    // callback, which process took it, and where this runner has read up to.
    // The read position is per runner rather than per run because the
    // notification carries nothing that could place it on a run.
    //
    // The pid is what a fork makes necessary. The monitor resets itself in the
    // child, so a runner that carried an inherited `held` across the fork would
    // never reacquire — no callback installed for the child, and a read
    // position sitting past the child's freshly zeroed stream, which reports
    // nothing for the rest of that process's life. Every entry point therefore
    // goes through `fault_monitor_if_held()`.
    bool fault_monitor_held_{false};
    long fault_monitor_pid_{-1};
    DeviceFaultNoticeCursor fault_notices_;
    // What the fault channel has said about *this* device, per generation. Its
    // suspicion is one of the two refusals `accepts_new_run` composes; the other
    // is each arch's `device_unusable_`, which `recover_device_or_mark_unusable`
    // sets. The two clear by different routes, and only a confirmed reset retires
    // this one.
    DeviceHealthState device_health_;
    // Driver ids of the streams this device's runs were launched on, captured at
    // boundary-record time. The fault filter reads these rather than asking a
    // handle: it runs at teardown, where a force reset may already have
    // invalidated the handles, and a stream a run used may since have been
    // replaced. Retired with the generation.
    RunStreamIdentities run_stream_ids_;

    /**
     * The process monitor this runner holds a reference on, or `nullptr`.
     *
     * Answers `nullptr` for a reference inherited across a fork, and drops the
     * inherited bookkeeping on the way out so the next `acquire` takes a real
     * reference for this process.
     */
    DeviceFaultMonitor *fault_monitor_if_held() noexcept;

public:
    /** The persistent device blocks belonging to one pipeline slot. */
    SlotPersistentArgs &slot_persistent_args(uint32_t pipeline_slot) { return slot_persistent_args_[pipeline_slot]; }

protected:
    // The one prebuilt runtime-arena image entry, describing bank 0's region
    // bases. `setup_static_arena` settles it alongside the bank's regions, so a
    // publication that moved a base leaves it invalid and a transaction that
    // changed nothing leaves it answerable.
    PrebuiltRuntimeArenaCache prebuilt_runtime_arena_cache_;

    // Persistent AICPU / AICore streams created in
    // `ensure_device_initialized()` and torn down in the subclass's
    // `finalize()`. A2A3 reserves these for bootstrap/control operations and
    // submits runs on its own per-slot stream sets; A5 submits runs on these.
    // `nullptr` before init.
    rtStream_t stream_aicpu_{nullptr};
    rtStream_t stream_aicore_{nullptr};
    // Device-constant AICore MMIO register-address tables: one 8-byte entry per
    // physical sub-core, queried from the driver for `device_id_` and copied to
    // device once per device context. The addresses are a property of the card,
    // not of a run or a pipeline slot, so every run on both slots reads the same
    // table. Committed lazily on the prepare path by the subclass's
    // `ensure_aicore_reg_table` — only the driver query is arch-specific, a2a3
    // mapping two MMIO pages and a5 one — and released in `finalize_common()`,
    // the same window as `device_wall_dev_ptr_`.
    //
    // `*_dev_` is the block's address and `*_committed_` says whether its
    // contents reached the device. They are separate because
    // `init_aicore_register_addresses` records the address before the copy and
    // clears it on failure only when the rollback release succeeded: a retained
    // address is owned but unwritten, so a non-zero address alone does not mean
    // "usable". Release keys on the address; reuse keys on the flag.
    //
    // Ctrl backs `KernelArgs::regs` on both arches. Pmu backs a2a3's
    // `KernelArgs::pmu_reg_addrs` and stays unset on a5, which has no separate
    // PMU register page and reads the Ctrl table instead.
    //
    // Kernel mode owns the same table through `PersistentKernelArgs`
    // (`kernel_persistent_args.h`) instead; a context latches one mode for its
    // whole lifetime, so exactly one of the two owners is ever live.
    uint64_t aicore_ctrl_reg_table_dev_{0};
    uint64_t aicore_pmu_reg_table_dev_{0};
    bool aicore_ctrl_reg_table_committed_{false};
    bool aicore_pmu_reg_table_committed_{false};
    // Platform-level device phase buffer: a header, thread-major phase records,
    // and the optional task-timing tail. Its address rides on
    // `KernelArgs.device_wall_data_base`. AICPU stamps raw sys-counter cycles;
    // subclass drain always pulls back the header + phases after stream sync,
    // and only pulls the tail when the header marks it used.
    //
    // One buffer per pipeline slot, allocated lazily on that slot's first
    // capture-enabled run and freed in subclass `finalize()`. Per slot rather
    // than per run so the hot path does no device malloc/free, and rather than
    // one shared buffer because the device writes it for the whole of a run
    // while the host reads it only at that run's drain — a successor armed into
    // the same storage would corrupt both.
    std::array<void *, PTO_PIPELINE_MAX_DEPTH> device_wall_dev_ptrs_{};
    // Per-slot readback results; see device_run_timing().
    std::array<DeviceRunTiming, PTO_PIPELINE_MAX_DEPTH> device_run_timing_{};
    // Set when a slot's buffer is successfully reset for a launch, cleared when
    // finalize has finished with the result. It answers two questions from the
    // one fact, because both are "this slot's buffer is reset and unread":
    // `arm_device_wall_buffer` refuses while it is set, so a successor cannot
    // overwrite an unconsumed result; `read_device_wall_ns` skips while it is
    // clear, so a run whose reset failed does not publish the storage's
    // previous contents as its own timing.
    std::array<bool, PTO_PIPELINE_MAX_DEPTH> device_timing_armed_{};
    // One result region per pipeline slot: the device address handed to that
    // slot's runs, and the host's copy of what the last such run published. Not
    // gated on diagnostics — an error result must survive with capture off.
    std::array<void *, PTO_PIPELINE_MAX_DEPTH> device_run_result_dev_ptrs_{};
    std::array<DeviceRunResultRegion, PTO_PIPELINE_MAX_DEPTH> device_run_results_{};
    // Which run each cached copy was read for, and what that read left behind.
    // Together they make the read once-per-run and keep the three read states
    // apart: an empty copy read successfully is an absent record, an empty copy
    // left by a failed D2H is no observation at all, and a slot with no region
    // attempted no copy to lose.
    RunRecordReadLedgerT<PTO_PIPELINE_MAX_DEPTH> device_run_result_reads_;
    // The boundary completion each run's own drain or poll observed, retained
    // because the drain's cleanup retires the fence that could otherwise be
    // asked. See host/run_evidence_retention.h.
    RunBoundaryLedgerT<PTO_PIPELINE_MAX_DEPTH> run_boundaries_observed_;
    // Whether a slot's region has had `published` zeroed since it was
    // allocated. `allocate_tensor` is an `rtMalloc`, so a fresh region holds
    // whatever the device left there — which cannot be assumed to differ from
    // the epoch of the run about to use it. Steady-state reuse needs no clear
    // because epochs distinguish runs, but the first use of an allocation does.
    std::array<bool, PTO_PIPELINE_MAX_DEPTH> device_run_result_initialized_{};

    // True after AICPU SO loaded; reset by the subclass's `finalize()`.
    bool binaries_loaded_{false};
    // Per-device guard for the initial simpler_aicpu_init launch.
    bool aicpu_init_launched_{false};
    // Shared diagnostics collectors. Each subclass initializes its own
    // (a2a3 wraps `halHostRegister`/`Unregister` callbacks, a5 uses
    // direct `rtMalloc`/`rtFree`), but the storage and lifetime live
    // on the base. `DepGenCollector` is not shared — each arch that
    // implements dep_gen (a2a3, a5) keeps it on its own subclass.
    ChipSwimlaneCollector chip_swimlane_collector_;
    // Not a collector: the state the runtime's bind writes into, read by
    // whichever per-event views the run enabled. Its two readers are gated
    // independently, so it belongs to neither. One per pipeline slot, because a
    // bind is preparation and a prepared successor prepares while its
    // predecessor still owns the collectors — see host_phase_run_state.h.
    std::array<HostPhaseRunState, PTO_PIPELINE_MAX_DEPTH> host_phase_runs_{};
    ArgsDumpCollector dump_collector_;
    PmuCollector pmu_collector_;
    ScopeStatsCollector scope_stats_collector_;

    // Enablement for the four shared diagnostics sub-features.
    ChipSwimlaneLevel chip_swimlane_level_{ChipSwimlaneLevel::DISABLED};  // resolved from set_chip_swimlane_enabled()
    std::string output_prefix_{};                                         // diagnostic artifact root directory
};
