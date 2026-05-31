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
 * - DeviceArgs: AICPU device argument structure
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
#include "device_arena.h"
#include "device_runner_base.h"     // common DeviceRunnerBase
#include "device_runner_helpers.h"  // common DeviceArgs + KernelArgsHelper
#include "common/kernel_args.h"
#include "common/memory_barrier.h"
#include "common/l2_perf_profiling.h"
#include "common/platform_config.h"
#include "common/unified_log.h"
#include "host/function_cache.h"
#include "host/memory_allocator.h"
#include "host/l2_perf_collector.h"
#include "host/pmu_collector.h"
#include "host/scope_stats_collector.h"
#include "host/tensor_dump_collector.h"
#include "load_aicpu_op.h"
#include "runtime.h"

// DeviceArgs + KernelArgsHelper are defined in
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
     * 4. Launches AICPU init kernel
     * 5. Launches AICPU main kernel
     * 6. Launches AICore kernel
     * 7. Synchronizes streams
     * 8. Cleans up runtime memory
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
    int run(Runtime &runtime, int block_dim, int launch_aicpu_num = 1) override;

    // `set_l2_swimlane_enabled`, `set_dump_tensor_enabled`,
    // `set_pmu_enabled`, `set_scope_stats_enabled`, `set_output_prefix`,
    // `output_prefix()`, and `launch_aicpu_kernel` live on
    // `DeviceRunnerBase`.

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
    // `register_callable_host_orch`, `unregister_callable`, `has_callable`,
    // `bind_callable_to_runtime`, `aicpu_dlopen_count`, and
    // `host_dlopen_count` are inherited from `DeviceRunnerBase`.

private:
    // Most lifecycle state (device_id_, block_dim_, cores_per_blockdim_,
    // worker_count_, executor + dispatcher bytes, aicore_bin_handle_,
    // load_aicpu_op_, mem_alloc_, the three DeviceArenas + their cached
    // sizes, persistent AICPU/AICore streams, kernel_args_, device_wall_*,
    // device_args_, binaries_loaded_) is inherited from `DeviceRunnerBase`.

    // Group D state (`chip_callable_buffers_`, `callables_`,
    // `orch_so_dedup_`, `aicpu_seen_callable_ids_`, `aicpu_dlopen_total_`,
    // `host_dlopen_total_`) and inner struct types
    // (`ChipCallableBuffer`, `CallableState`, `OrchSoBuffer`) are
    // inherited from `DeviceRunnerBase`.

    // Shared collectors (`l2_perf_collector_`, `dump_collector_`,
    // `pmu_collector_`, `scope_stats_collector_`) live on `DeviceRunnerBase`.

    // `query_max_block_dim`, `validate_block_dim`, `ensure_binaries_loaded`,
    // `configure_aicore_op_timeout`, and `prepare_orch_so` are inherited
    // (protected) from `DeviceRunnerBase`.

    /**
     * Initialize performance profiling device buffers
     *
     * Allocates L2PerfSetupHeader and per-core/per-thread buffers on device;
     * caller publishes the device pointer via kernel_args.l2_perf_data_base
     * (AICPU reads it through get_platform_l2_perf_base()).
     *
     * @param runtime Runtime instance to configure
     * @param num_aicore Number of AICore instances
     * @param device_id Device ID
     * @return 0 on success, error code on failure
     */
    int init_l2_perf(int num_aicore, int device_id);

    /**
     * Initialize tensor dump device buffers.
     *
     * @param runtime Runtime instance to configure
     * @param num_aicore Number of AICore instances (unused)
     * @param device_id Device ID for allocations
     * @return 0 on success, error code on failure
     */
    int init_tensor_dump(Runtime &runtime, int device_id);

    /**
     * Initialize PMU profiling device buffers.
     *
     * Allocates a PmuDataHeader and one PmuBuffer per core on device, then
     * publishes the data-header pointer into kernel_args.pmu_data_base.
     * Signature matches a2a3 for cross-platform consistency.
     */
    // Shared enable flags (`enable_l2_swimlane_`, `enable_dump_tensor_`,
    // `enable_pmu_`, `enable_scope_stats_`, `l2_perf_level_`,
    // `pmu_event_type_`, `output_prefix_`) live on `DeviceRunnerBase`.

    int init_pmu(int num_cores, int num_threads, const std::string &csv_path, PmuEventType event_type, int device_id);
    int init_scope_stats(int num_threads, int device_id);

    // Per-run collector teardown: stops mgmt + poll threads on every collector
    // whose init succeeded, in the only safe order (stop() joins mgmt before
    // poll). Idempotent — collectors that never initialized are skipped.
    // Does not release device memory; full release happens in finalize().
    void finalize_collectors();
};

#endif  // RUNTIME_DEVICERUNNER_H
