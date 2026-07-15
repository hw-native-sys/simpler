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

#ifndef RUNTIME_DEVICERUNNER_H
#define RUNTIME_DEVICERUNNER_H

#include <runtime/rt.h>

#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <map>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "callable.h"
#include "prepare_callable_common.h"
#include "utils/device_arena.h"
#include "device_runner_base.h"     // common DeviceRunnerBase
#include "device_runner_helpers.h"  // common KernelArgsHelper
#include "common/kernel_args.h"
#include "common/memory_barrier.h"
#include "common/l2_swimlane_profiling.h"
#include "common/platform_config.h"
#include "common/unified_log.h"
#include "host/function_cache.h"
#include "host/memory_allocator.h"
#include "host/l2_swimlane_collector.h"
#include "host/pmu_collector.h"
#include "host/dep_gen_collector.h"
#include "host/scope_stats_collector.h"
#include "host/args_dump_collector.h"
#include "aicpu_loader/host/load_aicpu_op.h"
#include "runtime.h"

// KernelArgsHelper is defined in
// src/common/platform/onboard/host/device_runner_helpers.h (included above).

/**
 * Device runner for kernel execution
 *
 * This class provides a unified interface for launching AICPU and AICore
 * kernels on Ascend devices. It handles:
 * - Device initialization and resource management
 * - Tensor memory allocation and data transfer
 * - AICPU kernel launching with dynamic arguments
 * - AICore kernel registration and launching
 * - Coordinated execution of both kernel types
 * - Runtime execution workflow
 */
class DeviceRunner : public DeviceRunnerBase {
public:
    DeviceRunner() = default;
    ~DeviceRunner();

    // `setup_static_arena`, `allocate_tensor`, `free_tensor`,
    // `copy_to_device`, `copy_from_device`,
    // `acquire_pooled_{gm_heap,gm_sm,runtime_arena}`, `create_thread`,
    // `attach_current_thread`, `ensure_device_initialized`,
    // `print_handshake_results`, `set_executors`, `set_dispatcher_binary`,
    // `device_id`, `last_device_wall_ns`, `launch_aicpu_kernel`, and
    // `launch_aicore_kernel` are inherited from `DeviceRunnerBase`.

    /**
     * Execute a runtime
     *
     * This method:
     * 1. Initializes device if not already done (lazy initialization)
     * 2. Initializes worker handshake buffers in the runtime based on block_dim
     * 3. Transfers runtime to device memory
     * 4. Launches AICPU main kernel
     * 5. Launches AICore kernel
     * 6. Synchronizes streams
     * 7. Cleans up runtime memory
     *
     * @param runtime             Runtime to execute (will be modified to
     * initialize workers)
     * @param block_dim            Number of blocks (1 block = 1 AIC + 2 AIV)
     * @param launch_aicpu_num     Number of AICPU instances (default: 1)
     * @return 0 on success, error code on failure
     *
     * The bound device id, AICPU/AICore executor binaries, and log filter
     * are captured once by simpler_init (binaries) / libsimpler_log.so (log)
     * and read off DeviceRunner state / HostLogger here — no per-run args.
     */
    int run(Runtime &runtime, const CallConfig &config) override;

    // `set_l2_swimlane_enabled`, `set_dump_args_enabled`,
    // `set_pmu_enabled`, `set_scope_stats_enabled`, `set_output_prefix`,
    // `output_prefix()`, and `launch_aicpu_kernel` live on
    // `DeviceRunnerBase`.

    /**
     * a5 `dep_gen` enablement setter, overriding the base no-op. Captures
     * orchestrator submit_task inputs for offline replay into deps.json.
     */
    void set_dep_gen_enabled(bool enable) override { enable_dep_gen_ = enable; }

    /**
     * Cleanup all resources
     *
     * Frees all device memory, destroys streams, and resets state.
     * Use this for final cleanup when no more tests will run.
     *
     * @return 0 on success, error code on failure
     */
    int finalize() override;

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
    // Most lifecycle state (device_id_, block_dim_, cores_per_blockdim_,
    // worker_count_, executor + dispatcher bytes, aicore_bin_handle_,
    // load_aicpu_op_, mem_alloc_, the three DeviceArenas + their cached
    // sizes, persistent AICPU/AICore streams, kernel_args_, device_wall_*,
    // binaries_loaded_) is inherited from `DeviceRunnerBase`.

    // Group D state (`chip_callable_buffers_`, `callables_`,
    // `aicpu_seen_callable_ids_`, `aicpu_dlopen_total_`,
    // `host_dlopen_total_`) and inner struct types
    // (`ChipCallableBuffer`, `CallableState`) are
    // inherited from `DeviceRunnerBase`.

    // Shared collectors (`l2_swimlane_collector_`, `dump_collector_`,
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
    // in-place drain could not clear. Once set, run() fails fast instead of
    // cascading into the confusing downstream failures (halResMap rc=62 at
    // init_aicore_register_addresses, or rtMalloc 507899) that a poisoned
    // context produces. On a5 the poison survives a close()+soft-reset for the
    // life of the process (an in-process re-init fails with rtStreamCreate
    // 507899), but a *force* reset clears it: finalize() calls
    // force_reset_device() on this path so the next Worker re-inits clean in the
    // same process (see force_reset_device()). This flag fails run() fast and
    // drives that recovery. See run() and recover_device_or_mark_unusable().
    bool device_unusable_{false};

    // On an AICore launch/sync error, best-effort drain the device so a later
    // run() on the same DeviceRunner can recover in place; if the drain itself
    // errors the context is unrecoverable without a full reset, so flip
    // device_unusable_ and let run() fail fast.
    void recover_device_or_mark_unusable(int aicore_rc);

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
     * Allocates L2SwimlaneSetupHeader and per-core/per-thread buffers on device;
     * caller publishes the device pointer via kernel_args.l2_swimlane_data_base
     * (AICPU reads it through get_platform_l2_swimlane_base()).
     *
     * @param runtime Runtime instance to configure
     * @param num_aicore Number of AICore instances
     * @param device_id Device ID
     * @return 0 on success, error code on failure
     */
    int init_l2_swimlane(int num_aicore, int aicpu_thread_num, int device_id);

    /**
     * Initialize args dump device buffers.
     *
     * @param runtime Runtime instance to configure
     * @param num_aicore Number of AICore instances (unused)
     * @param device_id Device ID for allocations
     * @return 0 on success, error code on failure
     */
    int init_args_dump(Runtime &runtime, int device_id);

    /**
     * Initialize PMU profiling device buffers.
     *
     * Allocates a PmuDataHeader and one PmuBuffer per core on device, then
     * publishes the data-header pointer into kernel_args.pmu_data_base.
     * Signature matches a2a3 for cross-platform consistency.
     */
    // Shared enable flags (`enable_l2_swimlane_`, `enable_dump_args_`,
    // `enable_pmu_`, `enable_scope_stats_`, `l2_swimlane_level_`,
    // `pmu_event_type_`, `output_prefix_`) live on `DeviceRunnerBase`.
    //
    // dep_gen enablement is a5-specific (a2a3 carries its own copy).
    bool enable_dep_gen_{false};

    int init_pmu(int num_cores, int num_threads, const std::string &csv_path, PmuEventType event_type, int device_id);
    int init_scope_stats(int num_threads, int device_id);

    /**
     * Initialize dep_gen capture shared memory.
     *
     * Allocates a DepGenDataHeader + 1 DepGenBufferState + N DepGenBuffers,
     * stores the device pointer to the data header into
     * kernel_args.dep_gen_data_base.
     */
    int init_dep_gen(int num_threads, int device_id);

    // Per-run collector teardown: stops mgmt + poll threads on every collector
    // whose init succeeded, in the only safe order (stop() joins mgmt before
    // poll). Idempotent — collectors that never initialized are skipped.
    // Does not release device memory; full release happens in finalize().
    void finalize_collectors();
};

#endif  // RUNTIME_DEVICERUNNER_H
