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
 * Device Runner - Ascend Device Execution Utilities
 *
 * This module provides utilities for launching and managing AICPU and AICore
 * kernels on Ascend devices using CANN runtime APIs.
 *
 * Key Components:
 * - KernelArgsHelper: Helper for managing kernel arguments with device memory
 * - DeviceRunner: kernel launching and execution
 */

#pragma once

#include <runtime/rt.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <map>
#include <string>
#include <thread>
#include <unordered_map>

#include "host/run_progress_slots.h"
#include <unordered_set>
#include <vector>

#include "callable.h"
#include "prepare_callable_common.h"
#include "runtime_c_api.h"
#include "utils/device_arena.h"
#include "device_runner_base.h"     // common DeviceRunnerBase
#include "device_runner_helpers.h"  // common KernelArgsHelper
#include "common/kernel_args.h"
#include "common/memory_barrier.h"
#include "common/chip_swimlane_profiling.h"
#include "common/platform_config.h"
#include "common/unified_log.h"
#include "host/function_cache.h"
#include "host/memory_allocator.h"
#include "host/chip_swimlane_collector.h"
#include "host/pmu_collector.h"
#include "host/dep_gen_collector.h"
#include "host/scope_stats_collector.h"
#include "host/args_dump_collector.h"
#include "aicpu_loader/host/load_aicpu_op.h"
#include "runtime.h"
#include "aicpu_topology_probe.h"

// KernelArgsHelper is defined in
// src/common/platform/onboard/host/device_runner_helpers.h (included above).

/**
 * Device runner for kernel execution
 *
 * This class provides a unified interface for launching AICPU and AICore
 * kernels on Ascend devices. It handles:
 * - Device initialization and resource management
 * - ChipTensor memory allocation and data transfer
 * - AICPU kernel launching with dynamic arguments
 * - AICore kernel registration and launching
 * - Coordinated execution of both kernel types
 * - Runtime execution workflow
 */
class DeviceRunner : public DeviceRunnerBase {
    // #2267's retention probe. There is no stream pair here to retire — every
    // run submits on the persistent bootstrap streams — so what the probe needs
    // instead is the single-run poll slot, which a successor's launch takes
    // over and no path restores.
    friend class RunRetentionProbePeer;

public:
    DeviceRunner() = default;
    ~DeviceRunner();

    // `setup_static_arena`, `allocate_tensor`, `free_tensor`,
    // `copy_to_device`, `copy_from_device`,
    // `acquire_pooled_{gm_heap,gm_sm,runtime_arena}`, `create_thread`,
    // `attach_current_thread`, `ensure_device_initialized`,
    // `print_handshake_results`, `set_executors`, `set_dispatcher_binary`,
    // `device_id`, `last_device_wall_ns`, `launch_aicpu_payload`, and
    // `launch_aicore_kernel` are inherited from `DeviceRunnerBase`.

    // The blocking entry point composes these operations. enqueue owns rollback
    // until the AICPU launch marker; drain takes that ownership on success.
    int prepare_execution(
        Runtime &runtime, const CallConfig &config, uint32_t pipeline_slot, const NativeRunIdentity &identity,
        std::unique_ptr<PreparedExecution> *prepared
    ) override;
    LaunchOutcome launch_execution(std::unique_ptr<PreparedExecution> prepared, LaunchPermit permit) override;
    void abandon_prepared_execution(PreparedExecution &prepared) noexcept override;
    int poll_execution(const ActiveExecution &active) override;
    DrainOutcome drain_execution(ActiveExecution &active) override;
    bool can_accept_run() const override { return !device_unusable_.load(std::memory_order_acquire); }

    /**
     * Whether a run may be ordered behind another on this runner right now.
     *
     * There is no stream ownership to wait for: every run submits on the two
     * persistent bootstrap streams, which this runner holds for its lifetime.
     * What does disqualify the runner is poison — a context that may not accept
     * a new submission must not have ordering edges built into it either, and
     * the boundary events such an edge would name are exactly the resources a
     * poisoned runner has quarantined rather than proved retired.
     */
    bool ready_to_join_launch() const override { return !device_unusable_.load(std::memory_order_acquire); }

    // `set_chip_swimlane_enabled`, `set_dump_args_enabled`,
    // `set_pmu_enabled`, `set_scope_stats_enabled`, `set_output_prefix`,
    // `output_prefix()`, and `launch_aicpu_payload` live on
    // `DeviceRunnerBase`.

    /**
     * a5 `dep_gen` enablement setter, overriding the base no-op. Also arms the
     * loaded runtime's host-side graph capture, which a host-orch runtime uses
     * instead of the device collector. Defined in the .cpp so this header stays
     * free of the runtime-provided capture symbols.
     */
    void arm_host_dep_gen_capture(bool enable) override;
    void configure_dep_gen_retention(bool enabled, std::size_t budget_bytes) override;
    bool dep_gen_retains_runs() const override;
    int admit_dep_gen_run(const DfxRunConfig &dfx, uint64_t run_epoch) override;
    void withdraw_dep_gen_run(uint64_t run_epoch) noexcept override;
    bool dep_gen_flush_retained(int timeout_ms, std::string *error) override;
    void dep_gen_finish_retained() override;

    /**
     * a5 fills the per-core register table only: it has no ffts_base_addr
     * field. The table is allocated on `mem_alloc_`.
     */
    int fill_persistent_arch_fields(KernelArgs *args, uint64_t device_id) override;

    // `upload_chip_callable_buffer`, `register_callable`,
    // `record_host_orch_callable`, `unregister_callable`, `has_callable`,
    // `bind_callable_to_runtime`, `aicpu_dlopen_count`, and
    // `host_dlopen_count` are inherited from `DeviceRunnerBase`.

    /**
     * Make the ACL context ready on the current thread.
     *
     * Calls aclInit() once per process (subsequent calls are idempotent and
     * tolerate the ACL_ERROR_REPEAT_INITIALIZE sentinel) and aclrtSetDevice()
     * on the current thread. This is the entry point for consumers that need
     * to call acl* / Hccl* APIs (for example the comm_hccl backend) but
     * intentionally do not want those modules to own ACL lifecycle themselves.
     *
     * Symmetric with finalize(): aclrtResetDevice + aclFinalize run there.
     *
     * @param device_id  Device ID to bind on the current thread.
     * @return 0 on success, error code on failure.
     */
    int ensure_acl_ready(int device_id);

    /**
     * Create a caller-owned aclrtStream for comm_* usage.
     *
     * Intended to back the ChipWorker Python wrapper's internal stream
     * ownership for distributed comm — callers pair it with
     * destroy_comm_stream() at teardown.  The ACL context must already be
     * ready on the calling thread (ensure_acl_ready()).
     *
     * @return aclrtStream pointer on success, NULL on failure.
     */
    void *create_comm_stream();

    /**
     * Destroy a stream previously returned by create_comm_stream().
     * Tolerates a nullptr stream (returns 0).
     *
     * @return 0 on success, error code on failure.
     */
    int destroy_comm_stream(void *stream);

private:
    int finalize_impl() override;

    // Most lifecycle state (device_id_, block_dim_, cores_per_blockdim_,
    // executor + dispatcher bytes, aicore_bin_handle_,
    // load_aicpu_op_, mem_alloc_, the three DeviceArenas + their cached
    // sizes, persistent AICPU/AICore streams, device_wall_*,
    // binaries_loaded_) is inherited from `DeviceRunnerBase`.

    // Group D state (`chip_callable_buffers_`, `callables_`,
    // `aicpu_seen_callable_ids_`, `aicpu_dlopen_total_`,
    // `host_dlopen_total_`) and inner struct types
    // (`ChipCallableBuffer`, `CallableState`) are
    // inherited from `DeviceRunnerBase`.

    // Shared collectors (`chip_swimlane_collector_`, `dump_collector_`,
    // `pmu_collector_`, `scope_stats_collector_`) live on `DeviceRunnerBase`.

    // dep_gen collector — captures orchestrator submit_task inputs for
    // offline replay. a5-specific (the base keeps dep_gen as a virtual hook).
    DepGenCollector dep_gen_collector_;

    // `query_max_block_dim`, `validate_block_dim`, `ensure_binaries_loaded`,
    // `configure_aicore_op_timeout`, and `prepare_orch_so` are inherited
    // (protected) from `DeviceRunnerBase`.

    // ACL lifecycle (process-wide). aclInit must run exactly once; ensure_acl_ready
    // gates it behind this flag. finalize() drives aclFinalize only if we observed
    // acl_ready_, so runtimes that never ask for ACL (e.g. pure rt-layer) stay unaffected.
    bool acl_ready_{false};

    // Set true when an AICore launch/sync error (e.g. an op-timeout reaped by
    // STARS, surfaced as 507000/507018 at stream sync, or a 207001 launch
    // failure) left the device context in a sticky-error state that an
    // in-place drain could not clear. Once set, admission/enqueue fail fast
    // instead of cascading into the confusing downstream failures (halResMap rc=62 at
    // init_aicore_register_addresses, or rtMalloc 507899) that a poisoned
    // context produces. On a5 the poison survives a close()+soft-reset for the
    // life of the process (an in-process re-init fails with rtStreamCreate
    // 507899), but a *force* reset clears it: finalize() calls
    // force_reset_device() on this path so the next Worker re-inits clean in the
    // same process (see force_reset_device()). This flag drives admission and
    // recovery. See launch_execution() and recover_device_or_mark_unusable().
    // Admission and recovery execute on different host threads.
    std::atomic<bool> device_unusable_{false};

    // Where each run-resource set is in its run, and which run that is. Per set
    // rather than per runner: with two runs launched a predecessor reaching
    // `Drained` says nothing about a successor still executing. See
    // `host/run_progress_slots.h` for the single-owner precondition the epoch
    // check relies on.
    RunProgressSlots run_progress_;

    // Queue this run's own AICore boundary into the AICPU stream, ahead of the
    // AICPU boundary record that then covers the whole operator. The reservation
    // is taken by the caller, so a refusal costs only joinability; this is the
    // step that touches the stream, and its failure grades the run Partial.
    int queue_own_boundary_wait(PreparedExecution &prepared, rtStream_t aicpu_stream, void *core_done);

    // Order this run behind the whole-operator boundary of the predecessor its
    // join names, on the AICore stream it submits first, and record the event
    // that proves the wait consumed. Its AICPU side needs no edge: that stream
    // already holds the predecessor's kernel and boundary record ahead of this
    // run's payload, and it is one FIFO.
    int queue_cross_run_wait(PreparedExecution &prepared, rtStream_t waiter_stream, LaunchProgressSink &sink);

    // Release execution-owned per-run resources. Idempotent so prepare rollback
    // and drain share one path. `launched` publishes this set's terminal state,
    // which only a run that reached the streams may claim. Discharges the queued
    // waits naming this run's boundaries first, and reads the poison only after
    // that, because the discharge is what may raise it. Collectors are not
    // released here: their device resources belong to the worker's lifetime and
    // are released in finalize().
    void cleanup_execution(PreparedExecution &prepared, bool launched) noexcept;

    // On an AICore launch/sync error, best-effort drain the device so a later
    // enqueue on the same DeviceRunner can recover in place; if the drain itself
    // errors the context is unrecoverable without a full reset, so flip
    // device_unusable_ and let admission/enqueue fail fast.
    void recover_device_or_mark_unusable(int aicore_rc) override;

    // Force-reset the card via aclrtResetDeviceForce to clear an op-timeout
    // sticky-error that the soft rtDeviceReset cannot (verified: a soft reset
    // + fresh in-process Worker.init still fails at rtStreamCreate 507899,
    // whereas a force reset lets the next init succeed in the same process).
    // Called from finalize() only on the device-poison path (device_unusable_).
    // Safe because onboard work always holds an exclusive task-submit lock on
    // the card (.claude/rules/running-onboard.md) and the reset is verified to
    // scope to this card only (does not disturb other devices). Returns 0 on
    // success, non-zero if the reset did not run or failed, so finalize() can
    // keep a still-poisoned card flagged instead of clearing device_unusable_
    // unconditionally.
    int force_reset_device();

    /**
     * Initialize performance profiling device buffers
     *
     * Allocates ChipSwimlaneSetupHeader and per-core/per-thread buffers on device;
     * caller publishes the device pointer via kernel_args.chip_swimlane_data_base
     * (AICPU reads it through get_platform_chip_swimlane_base()).
     *
     * @param runtime Runtime instance to configure
     * @param num_aicore Number of AICore instances
     * @param device_id Device ID
     * @return 0 on success, error code on failure
     */
    /**
     * Build this run's collector pools, profiling flag and device KernelArgs
     * refresh, under the execution claim.
     *
     * The collectors are resident and shared by every run on this runner, and a
     * run whose core / AICPU-thread counts differ from the pools' releases and
     * rebuilds them. Neither is safe while another run is executing against
     * them, which is why none of it happens during preparation.
     */
    int arm_collectors_for_run(const Runtime &runtime, PreparedExecution &prepared);

    /**
     * Commit this device's AICore register-address table on first use.
     *
     * Storage lives on DeviceRunnerBase and is released in finalize_common();
     * only the driver query is arch-specific, which is why this is not a base
     * method — a5 maps one register window per physical core, a2a3 two MMIO
     * pages selected by an AicoreRegKind.
     *
     * Guarded on the committed flag rather than the address: a failed
     * host-to-device copy whose rollback release also failed retains the address
     * for teardown, and that block is owned but unwritten. The retained address
     * is passed back in, which is what lets the driver entry reuse the block
     * instead of stranding it.
     *
     * Failure propagates. The AICPU handshake and the AICore PMU base both
     * dereference these addresses, so handing the device an uncommitted table
     * would deadlock the next task on a stream-sync timeout rather than fail the
     * prepare (see host_regs.cpp).
     */
    int ensure_aicore_reg_table();

    int init_chip_swimlane(
        int num_aicore, int aicpu_thread_num, int device_id, KernelArgsHelper &kernel_args,
        ChipSwimlaneLevel chip_swimlane_level
    );

    /**
     * Initialize args dump device buffers.
     *
     * @param runtime Runtime instance to configure
     * @param num_aicore Number of AICore instances (unused)
     * @param device_id Device ID for allocations
     * @return 0 on success, error code on failure
     */
    int
    init_args_dump(const Runtime &runtime, int device_id, KernelArgsHelper &kernel_args, DumpArgsLevel dump_args_level);

    /**
     * Initialize PMU profiling device buffers.
     *
     * Allocates a PmuDataHeader and one PmuBuffer per core on device, then
     * publishes the data-header pointer into kernel_args.pmu_data_base.
     * Signature matches a2a3 for cross-platform consistency.
     */
    // Shared enable flags (`enable_chip_swimlane_`, `enable_dump_args_`,
    // `enable_pmu_`, `enable_scope_stats_`, `chip_swimlane_level_`,
    // `pmu_event_type_`, `output_prefix_`) live on `DeviceRunnerBase`.
    //
    // dep_gen enablement is a5-specific (a2a3 carries its own copy).

    int query_aicpu_device_occupancy(pto::a5::AicpuDeviceOccupancy &out);
    int query_aicpu_topology(pto::a5::AicpuTopology &out);
    void clear_aicpu_topology_cache();
    // Device-side occupancy and the merged Host topology are immutable during
    // one DeviceRunner attach/reset lifetime. Cache successful probes only;
    // allowed CPU selection still runs per call because the requested active
    // count may change. Recovery, reset, and finalize clear both values.
    bool aicpu_device_occupancy_cached_{false};
    pto::a5::AicpuDeviceOccupancy aicpu_device_occupancy_{};
    bool aicpu_topology_cached_{false};
    pto::a5::AicpuTopology aicpu_topology_{};

    int init_pmu(int num_cores, int num_threads, int device_id, KernelArgsHelper &kernel_args);
    int init_scope_stats(int num_threads, int device_id, KernelArgsHelper &kernel_args);

    /**
     * Initialize dep_gen capture shared memory.
     *
     * Allocates a DepGenDataHeader + 1 DepGenBufferState + N DepGenBuffers,
     * stores the device pointer to the data header into
     * kernel_args.dep_gen_data_base.
     */
    int init_dep_gen(int num_threads, int device_id, KernelArgsHelper &kernel_args);

    // Emit the device-orchestration dep_gen graph, on both the success and the
    // error return of drain_execution: the device flushes its dep_gen buffers
    // during emergency_shutdown, so a failed run's graph is recoverable. Its own
    // reconcile is the completeness gate — see the definition.
    /**
     * Emit this run's dependency graph.
     *
     * `device_execution_complete` is the caller's own fence observation and the
     * whole of the completion proof: without it the retained path reads nothing
     * shared and publishes nothing.
     */
    void emit_device_dep_gen_graph(const DfxRunConfig &dfx, uint64_t run_epoch, bool device_execution_complete);

    // Per-run collector teardown: stops mgmt + poll threads on every collector
    // whose init succeeded, in the only safe order (stop() joins mgmt before
    // poll). Idempotent — collectors that never initialized are skipped.
    // Does not release device memory; full release happens in finalize().
    /**
     * Release the diagnostics collectors' shared memory.
     *
     * Returns non-zero when a collector's own finalize reported a failure —
     * today only retained ArgsDump, whose last host sealing happens there,
     * after the caller's diagnostic flush has already run. A caller folds this
     * into its own rc only where no device error has been recorded: a device
     * failure is the more useful diagnosis and keeps priority.
     */
    int finalize_collectors(bool abandon_device_resources = false);
};
