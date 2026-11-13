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
// WSE Host execution, callable ownership and retained storage.
// The concrete DeviceRunner supplies SRAM policy and PE execution.

#pragma once
#include "wse_dso.h"
#include "host/aligned_image_storage.h"

#include <dlfcn.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "runtime_c_api.h"
#include "wse_launch_transaction.h"
#include "run_resource_arena.h"

#include "callable.h"
#include "call_config.h"
#include "prepare_callable_common.h"
#include "utils/device_arena.h"
#include "utils/retained_scheduler_storage.h"
#include "common/kernel_args.h"
#include "common/device_phase.h"
#include "common/device_run_result.h"
#include "common/native_invocation_evidence.h"
#include "common/chip_swimlane_profiling.h"
#include "common/dma_workspace.h"
#include "common/platform_config.h"
#include "common/unified_log.h"
#include "platform_comm/comm.h"
#include "host/execution_mode_latch.h"
#include "host/caller_device_buffers.h"
#include "host/memory_allocator.h"
#include "host/chip_swimlane_collector.h"
#include "host/diagnostic_credit_signal.h"
#include "host/dfx_run_config.h"
#include "host/host_phase_records.h"
#include "host/host_phase_run_state.h"
#include "host/run_resource_slots.h"
#include "host/args_dump_collector.h"
#include "host/host_graph_exporter.h"
#include "host/pmu_collector.h"
#include "host/scope_stats_collector.h"
#include "runtime.h"

struct HostApi;  // common/host_api.h — fwd-declared to keep task_interface headers out

class WseHostRunner {
public:
    WseHostRunner() { commit_run_resources(kDefaultRunResourceSets, build_run_resources(kDefaultRunResourceSets)); }

    // The Worker grants resource-set count before use. Refused after any arena bank
    // is committed or a native run owns the runner.
    bool set_run_resource_grant(size_t granted);

    /** How many run-resource sets this runner currently holds. */
    size_t run_resource_grant() const { return granted_run_resource_sets_; }

    uint64_t arena_bank_gm_heap_base(uint32_t bank_id) const;

    /**
     * Retained temporary-buffer address held for one pipeline slot, or 0 while
     * that slot holds none. Two slots that have both staged arguments hold
     * distinct buffers; tests read this to prove the split is real.
     */
    uint64_t retained_temp_addr(uint32_t slot_id) const;

    // Public virtual dtor so c_api_shared can `delete` a WseHostRunner *
    // (destroy_device_context entrypoint).
    virtual ~WseHostRunner() = default;

    // Unspecified runners cannot admit a fixed-family kernel image.
    virtual const char *cpu_sim_target_family() const { return nullptr; }

    // --- Pure / no-op virtuals dispatched from the shared c_api glue ----
    struct PreparedExecution {
        PreparedExecution(
            const NativeRunIdentity &identity_in, Runtime &runtime_in, const CallConfig &config_in,
            uint32_t pipeline_slot_in
        ) :
            identity(identity_in),
            runtime(&runtime_in),
            config(config_in),
            dfx(DfxRunConfig::from(config_in)),
            scope_stats_capacity(runtime_in.scope_stats_capacity()),
            pipeline_slot(pipeline_slot_in) {
            // Device publication and host export share one resolved selection.
            runtime_in.dev.pmu_event_type = dfx.pmu_enabled ? static_cast<uint32_t>(dfx.pmu_event_type) : 0;
        }
        virtual ~PreparedExecution() = default;
        PreparedExecution(const PreparedExecution &) = delete;
        PreparedExecution &operator=(const PreparedExecution &) = delete;
        PreparedExecution(PreparedExecution &&other) noexcept :
            identity(other.identity),
            runtime(std::exchange(other.runtime, nullptr)),
            config(other.config),
            dfx(std::move(other.dfx)),
            scope_stats_capacity(other.scope_stats_capacity),
            pipeline_slot(other.pipeline_slot),
            num_aicore(other.num_aicore),
            launch_aicpu_num(other.launch_aicpu_num),
            artifact_identity(other.artifact_identity) {}
        PreparedExecution &operator=(PreparedExecution &&) = delete;

        NativeRunIdentity identity{};
        Runtime *runtime{nullptr};
        CallConfig config{};
        // Per-run diagnostic selection, fixed through prepare, execution and retirement.
        DfxRunConfig dfx{};
        simpler::dfx::scope_stats_runs::CapacityMetadata scope_stats_capacity;
        uint32_t pipeline_slot{UINT32_MAX};
        int num_aicore{0};
        int launch_aicpu_num{0};
        // Artifact identity includes run_id and Worker provenance; NativeRunIdentity does not.
        // An all-zero owner_instance_id denotes unspecified provenance.
        NativeRunArtifactIdentity artifact_identity{};
    };

    struct ActiveExecution {
        explicit ActiveExecution(std::unique_ptr<PreparedExecution> prepared_in, LaunchProgress progress_in) noexcept :
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
        WseLaunchReceipt receipt{};

        bool poisoned() const { return progress == LaunchProgress::Partial; }
    };
    /** Submit a Runtime and retain all state needed to query and drain it. */
    virtual int prepare_execution(
        Runtime &runtime, const CallConfig &config, uint32_t pipeline_slot, const NativeRunIdentity &identity,
        std::unique_ptr<PreparedExecution> *prepared
    ) = 0;
    /**
     * Allocate the active owner before entering the backend. An escaping backend
     * exception returns Partial with that same prepared state still owned; only
     * an explicit NotStarted result returns it for rollback.
     */
    LaunchOutcome launch_execution(std::unique_ptr<PreparedExecution> prepared, WseLaunchPermit permit) noexcept;
    virtual void abandon_prepared_execution(PreparedExecution &prepared) noexcept = 0;
    /** Return one of the SIMPLER_NATIVE_RUN_POLL_* values without waiting. */
    virtual int poll_execution(const ActiveExecution &active) = 0;
    /**
     * Wait for completion, publish DFX, and release per-run resources.
     *
     * Returns the device result and this run's diagnostics result separately;
     * see `DrainOutcome`.
     */
    virtual DrainOutcome drain_execution(ActiveExecution &active) = 0;
    virtual int finalize() = 0;

    // Device DepGen hooks default to no-ops. Host-built dependency graphs have
    // a separate exporter owned by this runner.
    virtual void configure_dep_gen_retention(bool /*enabled*/, std::size_t /*budget_bytes*/) {}
    virtual bool dep_gen_retains_runs() const { return false; }
    // Nonzero refuses launch; any already-admitted collectors must be withdrawn.
    virtual int admit_dep_gen_run(
        const DfxRunConfig & /*dfx*/, uint64_t /*run_epoch*/, simpler::dfx::runs::AdmissionRefusal * /*why*/ = nullptr
    ) {
        return 0;
    }
    // A device DepGen producer must support joined admission. Host graph capture
    // does not use this hook.
    virtual bool dep_gen_can_admit_joined_run(uint64_t /*run_epoch*/) const { return false; }
    /** Resolve this run's dep_gen terminal bank, 0 when it holds none. */
    virtual uint64_t dep_gen_run_terminal_bank(uint64_t /*run_epoch*/) const { return 0; }
    /** Release an admitted run whose launch submitted nothing. */
    virtual void withdraw_dep_gen_run(uint64_t /*run_epoch*/) noexcept {}
    virtual bool dep_gen_flush_retained(int /*timeout_ms*/, std::string * /*error*/) { return true; }
    virtual void dep_gen_finish_retained() {}

    // Transfer this thread's completed dependency graph to retained export during prepare.
    // False means the graph was dropped and diagnostics failed; never falls back to sync export.
    bool seal_host_dep_gen_graph(const DfxRunConfig &dfx, uint64_t run_epoch) noexcept;
    bool host_graph_retains_runs() const { return host_graph_exporter_.retains_runs(); }

    /** Reserve the runner's single active native execution through finalize. */
    bool try_acquire_native_run(const void *owner, const NativeRunIdentity &identity, WseLaunchPermit *permit);

    // Reissue only for the same owner/identity after a proven NotStarted launch.
    // The runner claim remains held; a changed owner is refused.
    bool reissue_launch_permit(const void *owner, const NativeRunIdentity &identity, WseLaunchPermit *permit);
    void release_native_run(const void *owner);
    bool native_run_active() const;
    bool native_runs_outstanding() const;
    void retain_run_result(uint32_t pipeline_slot, uint64_t run_epoch);
    void release_run_result(uint64_t run_epoch);
    DeviceRunTerminal device_run_terminal(uint32_t pipeline_slot, uint64_t run_epoch) const;
    /** Cleanup-only terminal storage keeps this context alive for close retry. */
    virtual bool terminal_backing_retained() const {
        return dump_collector_.terminal_backing_retained() || pmu_collector_.terminal_backing_retained() ||
               scope_stats_collector_.terminal_backing_retained();
    }

    bool native_run_owned_by(const void *owner) const;
    bool can_accept_run() const { return !launch_poisoned_.load(std::memory_order_acquire); }
    void poison_launch() noexcept { launch_poisoned_.store(true, std::memory_order_release); }

    // --- Shared methods --------------------------------------------------

    int setup_static_arena(uint32_t arena_bank, size_t gm_heap_size, size_t gm_sm_size, size_t runtime_arena_size);

    void *acquire_pooled_gm_heap(uint32_t arena_bank);
    void *acquire_pooled_gm_sm(uint32_t arena_bank);
    void *acquire_pooled_runtime_arena(uint32_t arena_bank);

    // Bind the child to this runner before invoking fn; inherited device identity is not trusted.
    // Returns after successful binding, or joins the failed child and throws std::system_error.
    // Linux thread names are truncated to the kernel limit.
    std::thread create_thread(std::function<void()> fn, std::string name = {});
    int attach_current_thread(int device_id);

    virtual void *allocate_tensor(size_t bytes);
    virtual void *allocate_caller_storage(size_t bytes) { return allocate_tensor(bytes); }
    virtual void *allocate_graph_storage(size_t bytes) { return allocate_tensor(bytes); }
    virtual void free_graph_storage(void *ptr) { free_tensor(ptr); }
    /** Total physical device backing committed by this runner. */
    virtual size_t committed_device_memory() const { return mem_alloc_.committed_bytes(); }
    virtual void free_tensor(void *dev_ptr);
    virtual int copy_to_device(void *dev_ptr, const void *host_ptr, size_t bytes);
    virtual int copy_from_device(void *host_ptr, const void *dev_ptr, size_t bytes);

    /**
     * Allocate for a caller and record the allocation as theirs; release it unless a run may still
     * be using it. Same contract as the onboard runner's pair — see
     * host/caller_device_buffers.h. A run's arguments may name only what a caller minted, so this
     * runner's own regions keep going through `allocate_tensor` and stay unrecorded.
     */
    void *allocate_caller_buffer(size_t bytes);
    int free_caller_buffer(void *dev_ptr);

    bool borrow_caller_buffers(uint64_t identity, const CallerDeviceBuffers::Span *spans, size_t count);
    void release_caller_buffers(uint64_t identity, bool keep);
    /**
     * Declare which caller allocations `identity` produces; see host/caller_device_buffers.h.
     *
     * @return false when the statement could not be recorded, which the declaring run's bind has
     *         to treat as its own failure: an undeclared producer reads as no producer.
     */
    [[nodiscard]] bool declare_caller_buffer_writes(
        uint64_t identity, const CallerDeviceBuffers::Span *spans, size_t count, size_t *unresolved_out = nullptr
    ) {
        return caller_device_buffers_.declare_writes(identity, spans, count, unresolved_out);
    }

    /** Whether `[addr, addr + bytes)` has no readable content for `identity` yet. */
    bool caller_buffer_written_by_other_run(uint64_t identity, uint64_t addr, uint64_t bytes) const {
        return caller_device_buffers_.written_by_other_run(identity, addr, bytes);
    }
    size_t caller_buffer_count() const { return caller_device_buffers_.allocation_count(); }
    size_t caller_buffer_borrow_count() const { return caller_device_buffers_.borrow_count(); }
    size_t caller_buffer_retained_count() const { return caller_device_buffers_.retained_count(); }

    virtual int device_memset(void *dev_ptr, int value, size_t bytes);

    // Per-slot result storage. Zero denotes an invalid slot; epoch checks prevent
    // a later run from consuming an earlier publication.
    uint64_t run_result_region_base(uint32_t pipeline_slot);

    // Returns null unless the result matches the requested slot and epoch and is published.
    // On success bytes_out receives the complete result-region size.
    const uint8_t *device_run_result(uint32_t pipeline_slot, uint64_t run_epoch, size_t *bytes_out) const;

    void get_retained_temp_buffer(uint32_t pipeline_slot, void **addr, size_t *size);
    void set_retained_temp_buffer(uint32_t pipeline_slot, void *addr, size_t size);
    int acquire_graph_staging(uint32_t pipeline_slot, size_t bytes, size_t alignment, void **staging_out);
    // Retain a packed Definition section for this slot. Source storage may be reused
    // after return; release occurs with the slot's graph staging, not the prepare stack.
    int publish_graph_section(
        uint32_t pipeline_slot, const void *bytes, size_t length, uint32_t *source_out, uint64_t *base_out
    );
    void get_graph_definition_staging(uint32_t pipeline_slot, void **addr, size_t *size);
    // Grow-only per-slot scheduler storage; the Host allocation is aligned and retained.
    // Reuse never initializes bytes: the runtime owns initialization and publication.
    int acquire_scheduler_state_storage(
        uint32_t pipeline_slot, size_t bytes, size_t alignment, void **device_out, void **host_out
    );
    std::shared_ptr<void> get_orchestration_scratch(uint32_t slot) const {
        return slot < orchestration_scratch_.size() ? orchestration_scratch_[slot] : std::shared_ptr<void>{};
    }
    void set_orchestration_scratch(uint32_t slot, std::shared_ptr<void> scratch) {
        if (slot < orchestration_scratch_.size()) orchestration_scratch_[slot] = std::move(scratch);
    }
    int acquire_sm_mirror(uint32_t pipeline_slot, size_t bytes, size_t alignment, void **addr_out);
    // Per-slot, grow-only aligned execution-image storage, independent of the shared-memory mirror.
    int acquire_run_image_staging(uint32_t pipeline_slot, size_t bytes, size_t alignment, void **addr_out);
    void clear_temporary_buffer();

    // On sim, allocate_tensor returns a plain host pointer, so the "device"
    // address is already host-readable — register is identity, unregister a
    // no-op. Mirrors the onboard DeviceRunnerBase API (separate class trees).
    void *register_device_memory_to_host(void *dev_ptr, size_t bytes) {
        (void)bytes;
        return dev_ptr;
    }
    void unregister_device_memory_from_host(void *dev_ptr) { (void)dev_ptr; }

    // Host view of child storage. WSE overrides this to enforce fixed application SRAM bounds.
    virtual void *acquire_child_memory_host_view(void *dev_ptr, size_t bytes);

    int record_host_orch_callable(
        int32_t callable_id, uint64_t chip_buffer_hash, void *host_dlopen_handle, void *host_orch_func_ptr,
        std::vector<ArgDirection> signature
    );
    int unregister_callable(int32_t callable_id);
    bool has_callable(int32_t callable_id) const;

    /**
     * Serializes the registration, unregistration and finalize entry points,
     * so an auto-ID scan and the ID it claims are one atomic step with respect
     * to explicit-ID registration and unregistration. Runs, initialization and
     * destroy are still caller-serialized.
     */
    std::mutex &callable_registration_mutex() { return callable_registration_mu_; }

    // One-step bind: install the reference to the callable's
    // registration-owned function tables and its active_callable_id, then run
    // the per-run bind_callable_to_runtime_impl with the state's
    // host_orch_func_ptr + signature. `api` is bound to this run; `orch_args` is a
    // const ChipStorageTaskArgs* (void* keeps task_interface headers out of this
    // header). Returns 0 on success, non-zero on failure.
    int bind_callable_to_runtime(
        Runtime &runtime, int32_t callable_id, const HostApi *api, const void *orch_args,
        const uint64_t *ring_task_window, const uint64_t *ring_heap, const uint64_t *ring_dep_pool
    );

    // Publish PE geometry from the fixed SRAM Core count and PEs per Core before
    // bind_callable_to_runtime builds the graph. WSE requires aicpu_thread_num == 0
    // and rejects legacy device diagnostics. Returns zero on success, an error on refusal.
    virtual int prepare_launch_shape(Runtime &runtime, const CallConfig &config) = 0;
    uint64_t upload_chip_callable_buffer(const ChipCallable *callable);
    int release_chip_callable_buffer(uint64_t hash);

    void set_executors(std::vector<uint8_t> aicpu_so_binary, std::vector<uint8_t> aicore_kernel_binary) {
        aicpu_so_binary_ = std::move(aicpu_so_binary);
        aicore_kernel_binary_ = std::move(aicore_kernel_binary);
    }

    /**
     * Record whether this Worker asked for an async-DMA workspace.
     */
    void set_dma_workspace_request(bool enable_sdma) { sdma_requested_ = enable_sdma; }
    int ensure_dma_workspace_provisioned();
    int device_id() const { return device_id_; }

    /**
     * This context's execution identity, latched once by whichever init entry
     * constructs it. Simulation supports program mode only: it has no device
     * streams for a caller to lend.
     */
    ExecutionModeLatch &execution_mode_latch() { return execution_mode_latch_; }

    /** True only after simpler_init succeeds and before teardown begins. */
    bool program_initialized() const { return program_initialized_; }
    void set_program_initialized(bool initialized) { program_initialized_ = initialized; }

    uint64_t last_device_wall_ns() const { return device_wall_ns_; }
    // Per-phase AICPU wall (ns) from the most recent run; RunWall aliases
    // last_device_wall_ns(). 0 for a phase that was never stamped. Used to emit
    // device-phase trace markers from the sim c_api, mirroring onboard.
    uint64_t last_device_phase_ns(AicpuPhase phase) const { return device_phase_ns_[static_cast<int>(phase)]; }
    // Per-phase start offset (ns) on a common device-clock timeline (origin =
    // earliest sub-phase start), so device spans carry a device-domain `ts` and
    // the orch∪sched "Effective" window is computable. 0 for RunWall / unstamped.
    uint64_t last_device_phase_start_ns(AicpuPhase phase) const {
        return device_phase_start_ns_[static_cast<int>(phase)];
    }
    // Per-slot task-timing dispatch/finish (ns) on the same device-clock timeline
    // as the phases. Both 0 for an untagged or incomplete slot. `slot` is 0..15.
    uint64_t last_task_slot_dispatch_ns(int slot) const { return task_slot_dispatch_ns_[slot]; }
    uint64_t last_task_slot_finish_ns(int slot) const { return task_slot_finish_ns_[slot]; }

    void set_chip_swimlane_enabled(int level) { chip_swimlane_level_ = static_cast<ChipSwimlaneLevel>(level); }
    uint32_t chip_swimlane_level() const { return static_cast<uint32_t>(chip_swimlane_level_); }
    void set_pmu_enabled(bool enabled) { pmu_enabled_ = enabled; }
    /** The runner's applied PMU selection, used by legacy HostApi callers; see the onboard base. */
    bool pmu_enabled() const { return pmu_enabled_; }
    /** One run's host-side JSON section, addressed by that run's epoch; see the onboard base. */
    bool publish_chip_swimlane_extension(
        uint64_t run_epoch, ChipSwimlaneExtensionSection section, const char *json_value, size_t json_size
    ) {
        // Handed on as the caller's bytes: the collector copies them once,
        // inside the charge its per-run budget admitted, so no payload-sized
        // temporary is materialized on a path that may refuse it.
        return chip_swimlane_collector_.set_json_extension(run_epoch, section, json_value, json_size);
    }
    /** Hand one slot's host-phase state to the run about to bind into it. */
    void begin_host_phase_run(uint32_t pipeline_slot, const DfxRunConfig &dfx);
    HostPhaseRecordPool *host_phase_pool_arm(uint32_t pipeline_slot, bool producer_wants_records) noexcept;
    void host_phase_pool_finish(uint32_t pipeline_slot, uint64_t submitted_tasks, uint64_t invocation_id) noexcept {
        if (pipeline_slot >= host_phase_runs_.size()) return;
        host_phase_runs_[pipeline_slot].records.finish(submitted_tasks, invocation_id);
    }
    /** Hand this run's host-phase state to the resident collector, at launch. */
    void publish_host_phase_run_to_collector(uint32_t pipeline_slot) noexcept;
    bool host_clock_alignment_log_required(uint32_t pipeline_slot) const {
        return pipeline_slot < host_phase_runs_.size() && host_phase_runs_[pipeline_slot].needs_clock_alignment();
    }
    /**
     * This slot's own host-orchestration state, for a run that must not read
     * the collector's single resident copy.
     */
    bool host_orchestrated_for_slot(uint32_t pipeline_slot) const noexcept {
        const auto *state = host_phase_runs_.find(pipeline_slot);
        return state != nullptr && state->host_orchestrated;
    }
    /** Hand this pass's records to the swimlane reader, just before its export. */
    void publish_host_phase_records_to_swimlane(uint32_t pipeline_slot);
    /**
     * Publish arch-specific runtime metadata into the swimlane export, between
     * the host-phase handoff and the export itself — the only point at which the
     * collector holds this run's records but has not yet serialized them. a5
     * overrides it; every other arch has nothing to add.
     */
    virtual void publish_chip_swimlane_runtime_extensions() {}
    // Diagnostic admission is all-or-nothing for the run's DfxRunConfig.
    // Refusal rolls back every earlier admission and reports the observed credit generation.
    /** The outcome of one all-or-nothing admission attempt. */
    struct DiagnosticAdmission {
        simpler::dfx::runs::AdmissionRefusal refusal{simpler::dfx::runs::AdmissionRefusal::None};
        const char *owner{nullptr};
        uint64_t credit_generation{0};
    };

    int start_shared_collectors_for_run(
        const DfxRunConfig &dfx, uint64_t run_epoch, const RunLocalDfxMetadata &run_metadata,
        DiagnosticAdmission *admission = nullptr
    );

    // Undo admission without publishing restored capacity as new progress.
    // Quarantined or failed owners still notify waiters of genuine capacity loss.
    void undo_admission_attempt(const DfxRunConfig &dfx, uint64_t run_epoch) noexcept;

    // Withdraw only after NotStarted, never after a partial launch.
    // Returns true only when every admitted owner restored its capacity.
    bool withdraw_unlaunched_collectors_for_run(const DfxRunConfig &dfx, uint64_t run_epoch) noexcept;

    // Close the PMU window by quiesce/reconcile or retained run_close.
    // Only joined producers justify device_producers_stopped; completion is a separate proof.
    void close_pmu_run_boundary(
        const DfxRunConfig &dfx, uint64_t run_epoch, bool device_execution_complete, bool device_producers_stopped
    );
    void close_scope_stats_run_boundary(
        const DfxRunConfig &dfx, uint64_t run_epoch, bool device_execution_complete, bool device_producers_stopped
    );

    // Close ordinary or retained argument capture. An unproved ownership release
    // returns an error; no unprocessed payload is acknowledged.
    int close_args_dump_run_boundary(
        const DfxRunConfig &dfx, uint64_t run_epoch, bool device_execution_complete, bool device_producers_stopped
    );

    /**
     * Whether a collector may hold a run past its boundary. Default off, and
     * both retaining collectors are configured here for the reason the onboard
     * base gives. PMU retention is independent of swimlane's.
     */
    void set_retain_runs(bool enabled) {
        chip_swimlane_collector_.configure_retained_runs(enabled, simpler::dfx::runs::kDefaultBudgetBytes);
        dump_collector_.configure_retained_runs(enabled, simpler::dfx::runs::kDefaultBudgetBytes);
        pmu_collector_.configure_retained_runs(enabled);
        scope_stats_collector_.configure_retained_runs(enabled, simpler::dfx::runs::kDefaultBudgetBytes);
        configure_dep_gen_retention(enabled, simpler::dfx::runs::kDefaultBudgetBytes);
        // A host-orchestrating runtime's graph is finished inside `prepare` and
        // never reaches a device collector, so it has an owner of its own.
        host_graph_exporter_.configure_retained_runs(enabled, simpler::dfx::runs::kDefaultBudgetBytes);
    }
    bool retains_runs() const {
        return chip_swimlane_collector_.retains_runs() || dump_collector_.retains_runs() ||
               pmu_collector_.retains_runs();
    }
    int flush_diagnostics(int timeout_ms, std::string *error);
    void finish_retained_runs();
    // Resolve this epoch's retained terminal bank, or zero when none exists.
    // A previous run's terminal totals must survive successor counter resets.
    uint64_t arm_chip_swimlane_run_terminal_bank(uint64_t run_epoch);

    // Credit reads and cancellation are nonblocking. wait_for_credit must not hold
    // a lock needed by collector writers, which publish capacity progress.
    uint64_t credit_generation() const { return credit_signal_.generation(); }
    int wait_for_credit(uint64_t seen, int32_t timeout_ms) {
        return static_cast<int>(credit_signal_.wait_for(seen, timeout_ms));
    }
    void cancel_credit_waits() { credit_signal_.cancel(); }

    /** Record that a diagnostic output slot or terminal bank came back. */
    void note_diagnostic_credit_released() { credit_signal_.advance(); }

    // Joined admission requires each enabled owner to distinguish epochs and have
    // both retained admission capacity and a free terminal bank.
    bool every_enabled_owner_can_admit(const DfxRunConfig &dfx, uint64_t run_epoch) const;

    // Write every terminal-bank address, including zero for disabled owners,
    // so reused KernelArgs cannot retain a predecessor's bank.
    void resolve_run_terminal_banks(const DfxRunConfig &dfx, uint64_t run_epoch, KernelArgs &kernel_args) const;

    // Open retained terminal arrays before admission. An enabled owner that cannot
    // allocate them fails; live counters cannot substitute for per-run terminal state.
    int open_run_terminal_banks_for_enabled_owners(const DfxRunConfig &dfx);

    /** Wire every collector's capacity release to this context's signal. */
    void install_credit_release_notifiers();

    // True while an unpublished retained consumer may still read its pools.
    // No active kernel does not imply that these pools can be freed.
    bool retained_consumers_outstanding() const;

    /** The dep_gen half, on the arch runner that owns that collector. */
    virtual bool dep_gen_retained_runs_outstanding() const { return false; }

    /** The dep_gen half, on the arch runner that owns that collector. */
    virtual void install_dep_gen_credit_release_notifier(const std::function<void()> & /*notify*/) {}

    /** The dep_gen half, on the arch runner that owns that collector. */
    virtual int open_dep_gen_run_terminal_banks(const DfxRunConfig & /*dfx*/) { return 0; }
    /** Write this pass's per-event host phase records, if it collected any. */
    void write_host_phase_records_artifact(const std::string &output_prefix, uint32_t pipeline_slot);
    // After kernel synchronization, close clock correlation, then quiesce/reconcile/export
    // collectors. Only observed completion allows reading terminal snapshots; joined
    // producers independently allow failed-run backing to be withdrawn.
    /**
     * Returns non-zero when a collector's close reported an ownership failure
     * the caller must surface — today only retained ArgsDump. A caller folds it
     * into its own rc behind any device error, which stays the more useful
     * diagnosis.
     */
    int teardown_shared_collectors_after_run(
        const DfxRunConfig &dfx, uint32_t pipeline_slot, uint64_t run_epoch, bool device_execution_complete,
        bool device_producers_stopped
    );

    // Join the run's producers before recording its stopped epoch. An exception
    // leaves no proof. Epoch identity prevents one run from settling another's banks.
    template <typename StopProducers>
    void stop_run_producers(uint64_t run_epoch, StopProducers &&stop) {
        producers_stopped_epoch_ = 0;
        stop();
        producers_stopped_epoch_ = run_epoch;
    }

    /** Whether *this* run's producers were stopped by the call above. */
    bool run_producers_stopped(uint64_t run_epoch) const {
        return run_epoch != 0 && producers_stopped_epoch_ == run_epoch;
    }

    // Settle using the recorded stopped epoch. Producer stop does not imply
    // successful completion or permit reading an unpublished terminal snapshot.
    int settle_run_after_producer_stop(
        const DfxRunConfig &dfx, uint32_t pipeline_slot, uint64_t run_epoch, bool device_execution_complete
    ) {
        return teardown_shared_collectors_after_run(
            dfx, pipeline_slot, run_epoch, device_execution_complete, run_producers_stopped(run_epoch)
        );
    }
    // The run whose producers `stop_run_producers` stopped, or zero. Zero is
    // the identity no run carries, so every run that has not reached its stop
    // reads as unstopped.
    uint64_t producers_stopped_epoch_ = 0;

    // Diagnostic artifact root directory (CallConfig::validate() enforces non-empty
    // upstream when any diagnostic is enabled).
    void set_output_prefix(const char *prefix) { output_prefix_ = (prefix != nullptr) ? prefix : ""; }
    const std::string &output_prefix() const { return output_prefix_; }

    // Latch this run's per-run diagnostic config onto the runner's enable_*_
    // members before prepare_execution() uses them. Each arch calls this at prepare;
    // the c_api threads the CallConfig through instead of calling setters.
    // Defined in the .cpp so this header does not need the full CallConfig.
    void apply_call_config(const CallConfig &config);

    size_t aicpu_dlopen_count() const { return 0; }
    size_t host_dlopen_count() const { return host_dlopen_total_; }

    // Last scheduler mode selected by bind, retained after release.
    // The publication count distinguishes repeated selection of the same mode.
    uint32_t last_scheduler_runtime_mode() const {
        return last_scheduler_runtime_mode_.load(std::memory_order_relaxed);
    }
    size_t scheduler_mode_publish_count() const {
        return scheduler_mode_publish_total_.load(std::memory_order_relaxed);
    }
    void record_scheduler_runtime_mode(uint32_t mode) {
        last_scheduler_runtime_mode_.store(mode, std::memory_order_relaxed);
        scheduler_mode_publish_total_.fetch_add(1, std::memory_order_relaxed);
    }

protected:
    virtual WseLaunchTransactionResult launch_execution_impl(PreparedExecution &prepared, WseLaunchPermit permit) = 0;

    // --- Helpers usable by subclass execution / finalize -----------------
    int ensure_device_initialized();
    virtual int ensure_binaries_loaded() = 0;
    // Registration owns orchestration code; validation leaves Runtime unchanged.
    int validate_active_callable(const Runtime &runtime) const;

    // Bulk-free the shared callable / chip-callable / orch-SO state. Subclass
    // finalize() calls this before mem_alloc_.finalize(). Idempotent.
    void release_callable_state();
    void release_graph_definition_staging();

    /**
     * Release every slot's retained scheduler-state storage.
     *
     * Returns the first failing free's code, having attempted every block, and
     * records each outcome before the slot's entry is cleared.
     */
    int release_scheduler_state_storage();

    /** Drop every retained host SM mirror, returning its pages to the allocator. */
    void release_sm_mirrors();
    void release_run_image_stagings();

    // --- Shared state (protected so subclass execution / init_* / finalize()
    // can read or write directly) ----------------------------------------

    // Configuration. device_id_ is which device this context is on — not a
    // claim of ownership, which the execution-mode latch carries instead. Set
    // once in attach_current_thread() during simpler_init and read afterwards;
    // the user's call sequence is single-threaded with respect to it so plain
    // int is sufficient.
    int device_id_{-1};
    int block_dim_{0};
    int worker_count_{0};

    // This context's execution identity. Write-once: the first init entry to
    // run latches it, and it never changes afterwards.
    ExecutionModeLatch execution_mode_latch_;
    bool program_initialized_{false};

    // Executor binaries — populated once via set_executors() during simpler_init,
    // owned for the rest of the runner's lifetime.
    std::vector<uint8_t> aicpu_so_binary_;
    std::vector<uint8_t> aicore_kernel_binary_;

    bool sdma_requested_{false};
    void *dma_workspace_handle_{nullptr};
    uint64_t dma_workspace_addr_[DMA_WORKSPACE_KIND_COUNT]{};

    MemoryAllocator mem_alloc_;
    // The device allocations a caller minted through this context, and which runs still hold them
    // — see host/caller_device_buffers.h.
    CallerDeviceBuffers caller_device_buffers_;
    RunResourceSlots<void *> retained_temp_addrs_;
    RunResourceSlots<size_t> retained_temp_sizes_;
    // Per-slot Definition staging is grow-only; the separate snapshot remains stable
    // while a later bind grows staging and a previous run still reads the snapshot.
    struct RetainedGraphStaging {
        std::vector<std::byte> bytes;
        std::vector<std::byte> snapshot;
    };
    RunResourceSlots<RetainedGraphStaging> graph_definition_staging_;
    // Scheduler-state storage, one retained pair per pipeline slot — see
    // HostApi acquire_scheduler_state_storage and
    // utils/retained_scheduler_storage.h, which holds the grow, alignment and
    // failure rules.
    RunResourceSlots<RetainedSchedulerStorage> scheduler_state_storage_;
    // Per-slot CHIP_ALIGN_SIZE-aligned mirror, retained across binds.
    // Only live segment prefixes are initialized and published; untouched capacity is not read.
    using RetainedSmMirror = AlignedImageStorage;
    RunResourceSlots<std::shared_ptr<void>> orchestration_scratch_;
    RunResourceSlots<RetainedSmMirror> sm_mirrors_;

    // Per-slot grow-only execution-image staging, retained beyond bind until publication.
    RunResourceSlots<RetainedSmMirror> run_image_stagings_;

    // Each arena bank backs the three pooled regions (GM heap / shared
    // shared memory / trb prebuilt runtime arena) for one pipeline slot. They
    // are separate allocations because the combined size can exceed the device
    // allocator's largest contiguous block. Released explicitly in finalize()
    // before mem_alloc_.finalize().
    //
    // A bank's runtime pool stays unreserved when setup_static_arena was
    // invoked with runtime_arena_size == 0 (hbg path).
    static void *arena_alloc_trampoline(void *ctx, size_t size) {
        return static_cast<WseHostRunner *>(ctx)->allocate_tensor(size);
    }
    static void arena_free_trampoline(void *ctx, void *p) { static_cast<WseHostRunner *>(ctx)->free_tensor(p); }
    static void *graph_alloc_trampoline(void *ctx, size_t size) {
        return static_cast<WseHostRunner *>(ctx)->allocate_graph_storage(size);
    }
    static void graph_free_trampoline(void *ctx, void *p) { static_cast<WseHostRunner *>(ctx)->free_graph_storage(p); }
    // Independent pooled regions per resource bank. Equal or smaller requests reuse backing;
    // DeviceArena is nonmovable, so bank storage owns pointers.
    struct ArenaBank {
        ArenaBank(void *ctx) :
            gm_heap(&arena_alloc_trampoline, &arena_free_trampoline, ctx),
            gm_sm(&graph_alloc_trampoline, &graph_free_trampoline, ctx),
            runtime_pool(&graph_alloc_trampoline, &graph_free_trampoline, ctx) {}

        DeviceArena gm_heap;
        DeviceArena gm_sm;
        DeviceArena runtime_pool;
        size_t cached_gm_heap_size{0};
        size_t cached_gm_sm_size{0};
        size_t cached_runtime_arena_size{0};
    };
    RunResourceSlots<std::unique_ptr<ArenaBank>> arena_banks_;
    ArenaBank &arena_bank(uint32_t bank_id) { return *arena_banks_[bank_id]; }
    // How many run-resource sets every per-set container holds. Written last by
    // `commit_run_resources`, so until a grant is published every bound check
    // still answers for the previous one.
    size_t granted_run_resource_sets_{0};

    /** One built, unpublished grant: every per-set container's new storage. */
    struct RunResourceStaging {
        RunResourceSlots<void *>::Storage retained_temp_addrs;
        RunResourceSlots<size_t>::Storage retained_temp_sizes;
        RunResourceSlots<RetainedGraphStaging>::Storage graph_definition_staging;
        RunResourceSlots<RetainedSchedulerStorage>::Storage scheduler_state_storage;
        RunResourceSlots<std::shared_ptr<void>>::Storage orchestration_scratch;
        RunResourceSlots<RetainedSmMirror>::Storage sm_mirrors;
        RunResourceSlots<RetainedSmMirror>::Storage run_image_stagings;
        RunResourceSlots<std::unique_ptr<ArenaBank>>::Storage arena_banks;
        RunResourceSlots<HostPhaseRunState>::Storage host_phase_runs;
        RunResourceSlots<NativeInvocationResultStorage>::Storage device_run_results;
    };

    /** Allocate every per-set resource for `granted`; publishes nothing. */
    RunResourceStaging build_run_resources(size_t granted);
    /** Adopt a built grant by swapping every container. Never throws. */
    void commit_run_resources(size_t granted, RunResourceStaging &&staged) noexcept;

    // Simulation state — written by enqueue/init and read by the AICPU /
    // AICore execute functions via the platform-regs setter functions.
    KernelArgs kernel_args_;

    // Platform-level device wall buffer: 8-byte device-resident slot whose
    // address rides on KernelArgs.device_wall_data_base. AICPU writes the
    // run wall (ns) through that pointer; this DeviceRunner pulls it back
    // via copy_from_device after stream sync and caches it for
    // last_device_wall_ns(). Allocated lazily at enqueue, freed in finalize().
    void *device_wall_dev_ptr_{nullptr};
    uint64_t device_wall_ns_{0};
    uint64_t device_phase_ns_[NUM_AICPU_PHASES] = {0};
    // Per-phase start offset (ns) from the earliest sub-phase start; see
    // last_device_phase_start_ns().
    uint64_t device_phase_start_ns_[NUM_AICPU_PHASES] = {0};
    // Per-slot task-timing dispatch/finish (ns), offset from the same origin as
    // the phases; see last_task_slot_dispatch_ns() / last_task_slot_finish_ns().
    uint64_t task_slot_dispatch_ns_[NUM_TASK_TIMING_SLOTS] = {0};
    uint64_t task_slot_finish_ns_[NUM_TASK_TIMING_SLOTS] = {0};

    // Chip-callable buffer pool (sim path). Keyed by FNV-1a 64-bit content
    // hash. Each entry owns a host scratch holding the ChipCallable with each
    // child's resolved_addr_ fixed up to the dlopen'd function pointer;
    // chip_dev == (uint64_t)host_scratch. The dlopen handles in
    // dlopen_handles are bulk-dlclose'd in finalize().
    struct ChipCallableBuffer {
        uint64_t chip_dev{0};  // (uint64_t)host_scratch
        uint8_t *host_scratch{nullptr};
        size_t total_size{0};
        int refcount{0};
        std::vector<void *> dlopen_handles;
        // The callable's registration-owned function tables, dense over
        // [0, table_len). Sim performs no upload, so the "device" addresses
        // below are the vectors' own host addresses and the AICPU reads them in
        // place. The object view holds the CoreCallable objects inside
        // `host_scratch`; the entry view holds the host function pointers this
        // registration's dlopen resolved, which is what a device consumer of
        // resolved entries would dispatch through.
        std::vector<uint64_t> object_table;
        std::vector<uint64_t> entry_table;
        uint64_t object_table_dev{0};
        uint64_t entry_table_dev{0};
        uint32_t table_len{0};
    };
    std::unordered_map<uint64_t, ChipCallableBuffer> chip_callable_buffers_;

    // Per-callable_id prepared state. Mirrors onboard.
    struct CallableState {
        uint64_t chip_buffer_hash{0};
        std::vector<ArgDirection> signature;
        void *host_dlopen_handle{nullptr};
        void *host_orch_func_ptr{nullptr};
    };
    std::unordered_map<int32_t, CallableState> callables_;
    // Held by every registration/unregistration entry point and by
    // finalize_device. It makes an auto-ID scan and its claim atomic against
    // an explicit-ID claim of the same slot; runs, initialization and destroy
    // are serialized by their callers instead.
    std::mutex callable_registration_mu_;
    size_t host_dlopen_total_{0};
    // Written on the bind path, read by the caller's thread.
    std::atomic<uint32_t> last_scheduler_runtime_mode_{0};
    std::atomic<size_t> scheduler_mode_publish_total_{0};
    // AICPU executor SO: load-once, matching onboard's aicpu_runtime_registered_ pattern.
    // The aicpu_executor g_aicpu_executor static lives inside the dlopen'd DSO;
    // reloading it destroys orch_so_handle_ and breaks the orch-SO cache-hit path.
    bool aicpu_so_loaded_{false};

    std::atomic<const void *> active_native_run_{nullptr};
    std::atomic<bool> launch_poisoned_{false};

    // Dynamically loaded executor libraries (shared infra; the dlsym'd function-
    // pointer table itself lives on the subclass since signatures diverge
    // per-arch — a2a3 vs a5 differ on aicore_execute and several setters).
    void *aicpu_so_handle_{nullptr};
    void *aicore_so_handle_{nullptr};
    std::string aicpu_so_path_;
    std::string aicore_so_path_;

    // Performance / diagnostics collectors shared across arches.
    // Context-owned signal for blocked diagnostic admission; only integer tokens
    // and opaque handles cross the runtime C ABI.
    simpler::dfx::credit::DiagnosticCreditSignal credit_signal_;

    // Callbacks are installed once before any collector thread starts;
    // concurrent replacement would race a writer invoking its callback.
    bool credit_notifiers_installed_{false};

    ChipSwimlaneCollector chip_swimlane_collector_;
    // Per-slot Host phase records are populated during prepare and shared by independent
    // diagnostic readers. A WSE runner holds the exclusive claim throughout a run.
    RunResourceSlots<HostPhaseRunState> host_phase_runs_;
    // Epoch-tagged per-slot result storage, initialized once and published by the device.
    // Simulation reads the Host backing directly. Never clear a live earlier run's result.
    RunResourceSlots<NativeInvocationResultStorage> device_run_results_;
    mutable std::mutex retained_result_mu_;
    std::unordered_map<uint64_t, DeviceRunResultRegion> retained_run_results_;
    ArgsDumpCollector dump_collector_;
    PmuCollector pmu_collector_;
    ScopeStatsCollector scope_stats_collector_;
    // The host-built graph's owner. Not a device collector: nothing it holds
    // is written by the device, and no run boundary reads it.
    simpler::dfx::host_graph::HostGraphExporter host_graph_exporter_;

    // Collector pools are shaped by core/thread counts. A changed topology requires
    // rebuilding pools only after all retained readers have released them.
    struct CollectorShape {
        bool latched{false};
        int num_aicore{0};
        int aicpu_thread_num{0};
        int launch_aicpu_num{0};
    };

    // A changed shape requires rebuilding under the exclusive execution claim.
    // No other run may execute against the pools during that rebuild.
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

    // Enablement flags. A run's own diagnostics configuration travels on its
    // PreparedExecution::dfx; the runner keeps only what a device-context query
    // answers from, plus the artifact root the host-phase pool arms against.
    ChipSwimlaneLevel chip_swimlane_level_{ChipSwimlaneLevel::DISABLED};  // resolved from set_chip_swimlane_enabled()
    bool pmu_enabled_{false};                                             // resolved from set_pmu_enabled()
    std::string output_prefix_{};                                         // diagnostic artifact root directory
};
