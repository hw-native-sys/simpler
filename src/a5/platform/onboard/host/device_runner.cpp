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
 * Device Runner Implementation
 *
 * This file implements the device execution utilities for launching and
 * managing AICPU and AICore kernels on Ascend devices.
 */

#include "device_runner.h"

#include "acl/acl.h"
#include "host_log.h"

#include <dlfcn.h>

#include "load_aicpu_op.h"

#include <cassert>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "callable.h"
#include "callable_protocol.h"
#include "utils/elf_build_id.h"
#include "utils/fnv1a_64.h"
#include "host/host_regs.h"  // Register address retrieval
#include "host/raii_scope_guard.h"

// =============================================================================
// DeviceRunner Implementation
// =============================================================================

// rtMalloc / rtFree wrappers shared by all three profiling subsystems.
// a5 onboard goes directly through CANN runtime — no per-allocation tracking,
// so the framework's std::function alloc / free shapes match plain function
// pointers here.
static void *prof_alloc_cb(size_t size) {
    void *ptr = nullptr;
    int rc = rtMalloc(&ptr, size, RT_MEMORY_HBM, 0);
    return (rc == 0) ? ptr : nullptr;
}

static int prof_free_cb(void *dev_ptr) { return rtFree(dev_ptr); }

DeviceRunner::~DeviceRunner() { finalize(); }

// `setup_static_arena`, `create_thread`, `attach_current_thread`,
// `configure_aicore_op_timeout`, `ensure_device_initialized`,
// `ensure_binaries_loaded`, `query_max_block_dim`, and `validate_block_dim`
// live on `DeviceRunnerBase` — see
// `src/common/platform/onboard/host/device_runner_base.cpp`.

int DeviceRunner::run(Runtime &runtime, int block_dim, int launch_aicpu_num) {
    if (launch_aicpu_num < 1 || launch_aicpu_num > PLATFORM_MAX_AICPU_THREADS) {
        LOG_ERROR("launch_aicpu_num (%d) must be in range [1, %d]", launch_aicpu_num, PLATFORM_MAX_AICPU_THREADS);
        return -1;
    }

    // Ensure device is initialized (lazy initialization)
    int rc = ensure_device_initialized();
    if (rc != 0) {
        LOG_ERROR("ensure_device_initialized failed: %d", rc);
        return rc;
    }

    // Lazy-allocate the 8-byte device_wall buffer on first run. See the a2a3
    // onboard equivalent for the rationale.
    if (device_wall_dev_ptr_ == nullptr) {
        device_wall_dev_ptr_ = allocate_tensor(sizeof(uint64_t));
        if (device_wall_dev_ptr_ != nullptr) {
            kernel_args_.args.device_wall_data_base = reinterpret_cast<uint64_t>(device_wall_dev_ptr_);
            uint64_t zero = 0;
            (void)copy_to_device(device_wall_dev_ptr_, &zero, sizeof(uint64_t));
        }
    }

    // Auto sentinel (block_dim == 0) is resolved directly from
    // query_max_block_dim; explicit values still go through validate. The
    // auto branch skips validate so we don't pay the ACL syscalls twice.
    if (block_dim == 0) {
        block_dim = query_max_block_dim(stream_aicore_);
        LOG_INFO_V0("block_dim auto-resolved to %d", block_dim);
        if (block_dim < 1) {
            LOG_ERROR("block_dim auto-resolved to invalid value %d", block_dim);
            return -1;
        }
    } else {
        rc = validate_block_dim(stream_aicore_, block_dim);
        if (rc != 0) {
            return rc;
        }
    }
    block_dim_ = block_dim;

    int num_aicore = block_dim * cores_per_blockdim_;
    // Initialize handshake buffers in runtime
    if (num_aicore > RUNTIME_MAX_WORKER) {
        LOG_ERROR("block_dim (%d) exceeds RUNTIME_MAX_WORKER (%d)", block_dim, RUNTIME_MAX_WORKER);
        return -1;
    }

    runtime.worker_count = num_aicore;
    worker_count_ = num_aicore;  // Store for print_handshake_results in destructor
    runtime.aicpu_thread_num = launch_aicpu_num;

    // Get AICore register addresses for register-based task dispatch
    rc = init_aicore_register_addresses(&kernel_args_.args.regs, static_cast<uint64_t>(device_id_), mem_alloc_);
    if (rc != 0) {
        LOG_ERROR("init_aicore_register_addresses failed: %d", rc);
        return rc;
    }

    // Calculate number of AIC cores (1/3 of total)
    int num_aic = block_dim;  // Round up for 1/3
    uint32_t enable_profiling_flag = PROFILING_FLAG_NONE;
    if (enable_dump_tensor_) {
        SET_PROFILING_FLAG(enable_profiling_flag, PROFILING_FLAG_DUMP_TENSOR);
    }
    if (enable_l2_swimlane_) {
        SET_PROFILING_FLAG(enable_profiling_flag, PROFILING_FLAG_L2_SWIMLANE);
    }
    if (enable_pmu_) {
        SET_PROFILING_FLAG(enable_profiling_flag, PROFILING_FLAG_PMU);
    }
    if (enable_scope_stats_) {
        SET_PROFILING_FLAG(enable_profiling_flag, PROFILING_FLAG_SCOPE_STATS);
    }

    for (int i = 0; i < num_aicore; i++) {
        runtime.workers[i].aicpu_ready = 0;
        runtime.workers[i].aicore_done = 0;
        runtime.workers[i].task = 0;
        // Set core type: first 1/3 are AIC, remaining 2/3 are AIV
        runtime.workers[i].core_type = (i < num_aic) ? CoreType::AIC : CoreType::AIV;
    }
    // Profiling enablement lives on KernelArgs (no longer mirrored into Handshake).
    kernel_args_.args.enable_profiling_flag = enable_profiling_flag;

    // Set function_bin_addr for all tasks: Runtime::func_id_to_addr_[] stores
    // a CoreCallable device address; the binary code address is one
    // compile-time offset further in.
    LOG_DEBUG("Setting function_bin_addr for Tasks");
    for (int i = 0; i < runtime.get_task_count(); i++) {
        Task *task = runtime.get_task(i);
        if (task != nullptr) {
            uint64_t callable_addr = runtime.get_function_bin_addr(task->func_id);
            task->function_bin_addr = callable_addr + CoreCallable::binary_data_offset();
            LOG_DEBUG("Task %d (func_id=%d) -> function_bin_addr=0x%lx", i, task->func_id, task->function_bin_addr);
        }
    }
    LOG_DEBUG("");

    // Scope guards for cleanup on all exit paths
    auto regs_cleanup = RAIIScopeGuard([this]() {
        if (kernel_args_.args.regs != 0) {
            mem_alloc_.free(reinterpret_cast<void *>(kernel_args_.args.regs));
            kernel_args_.args.regs = 0;
        }
    });

    auto runtime_args_cleanup = RAIIScopeGuard([this]() {
        kernel_args_.finalize_device_kernel_args();
        kernel_args_.finalize_runtime_args();
    });

    // Initialize per-subsystem shared memory.
    if (enable_l2_swimlane_) {
        rc = init_l2_perf(num_aicore, device_id_);
        if (rc != 0) {
            LOG_ERROR("init_l2_perf failed: %d", rc);
            return rc;
        }
    }

    if (enable_dump_tensor_) {
        rc = init_tensor_dump(runtime, device_id_);
        if (rc != 0) {
            LOG_ERROR("init_tensor_dump failed: %d", rc);
            return rc;
        }
    }

    if (enable_pmu_) {
        rc = init_pmu(num_aicore, launch_aicpu_num, make_pmu_csv_path(output_prefix_), pmu_event_type_, device_id_);
        if (rc != 0) {
            LOG_ERROR("PMU init failed: %d, disabling PMU for this run", rc);
            kernel_args_.args.pmu_data_base = 0;
            enable_pmu_ = false;
        }
    }

    if (enable_scope_stats_) {
        rc = init_scope_stats(launch_aicpu_num, device_id_);
        if (rc != 0) {
            LOG_ERROR("init_scope_stats failed: %d", rc);
            return rc;
        }
    }

    // Cleanup guard for early returns: stops all started collectors so
    // their mgmt + poll threads exit cleanly. stop() is idempotent and a
    // no-op on collectors that never started.
    auto perf_cleanup = RAIIScopeGuard([this]() {
        finalize_collectors();
    });

    LOG_INFO_V0("=== Initialize runtime args ===");
    rc = prepare_orch_so(runtime);
    if (rc != 0) {
        LOG_ERROR("prepare_orch_so failed: %d", rc);
        return rc;
    }
    rc = kernel_args_.init_runtime_args(runtime, mem_alloc_);
    if (rc != 0) {
        LOG_ERROR("init_runtime_args failed: %d", rc);
        return rc;
    }

    // Publish log config to AICPU via KernelArgs (severity floor + INFO verbosity).
    // HostLogger is the single source of truth for log config (seeded by
    // libsimpler_log.so via simpler_log_init before host_runtime.so was even
    // dlopen'd). Read it directly when populating KernelArgs.
    kernel_args_.args.log_level = static_cast<uint32_t>(HostLogger::get_instance().level());
    kernel_args_.args.log_info_v = static_cast<uint32_t>(HostLogger::get_instance().info_v());
    // Device ordinal for the AICPU executor's per-device orchestration-SO name.
    kernel_args_.args.device_id = static_cast<uint32_t>(device_id_);

    // Start collector mgmt + poll threads now, just before kernels launch.
    // Starting earlier wastes CPU on empty queues and risks tripping
    // ProfilerBase's poll-loop idle-timeout if device-side init is slow.
    auto thread_factory = [this](std::function<void()> fn) {
        return create_thread(std::move(fn));
    };
    if (enable_l2_swimlane_) {
        l2_perf_collector_.start(thread_factory);
    }
    if (enable_dump_tensor_) {
        dump_collector_.start(thread_factory);
    }
    if (enable_pmu_) {
        pmu_collector_.start(thread_factory);
    }
    if (enable_scope_stats_) {
        scope_stats_collector_.start(thread_factory);
    }

    LOG_INFO_V0("=== launch_aicpu_kernel %s ===", host::KernelNames::InitName);
    rc = launch_aicpu_kernel(stream_aicpu_, &kernel_args_.args, host::KernelNames::InitName, 1);
    if (rc != 0) {
        LOG_ERROR("launch_aicpu_kernel (init) failed: %d", rc);
        return rc;
    }

    LOG_INFO_V0("=== launch_aicpu_kernel %s ===", host::KernelNames::RunName);
    rc = launch_aicpu_kernel(
        stream_aicpu_, &kernel_args_.args, host::KernelNames::RunName, PLATFORM_MAX_AICPU_THREADS_JUST_FOR_LAUNCH
    );
    if (rc != 0) {
        LOG_ERROR("launch_aicpu_kernel (main) failed: %d", rc);
        return rc;
    }

    LOG_INFO_V0("=== launch_aicore_kernel ===");
    rc = kernel_args_.init_device_kernel_args(mem_alloc_);
    if (rc != 0) {
        LOG_ERROR("init_device_kernel_args failed: %d", rc);
        return rc;
    }
    rc = launch_aicore_kernel(stream_aicore_, kernel_args_.device_k_args_);
    if (rc != 0) {
        LOG_ERROR("launch_aicore_kernel failed: %d", rc);
        return rc;
    }

    LOG_INFO_V0("=== aclrtSynchronizeStreamWithTimeout stream_aicpu_ ===");
    rc = aclrtSynchronizeStreamWithTimeout(stream_aicpu_, PLATFORM_STREAM_SYNC_TIMEOUT_MS);
    if (rc == ACL_ERROR_RT_STREAM_SYNC_TIMEOUT) {
        LOG_ERROR(
            "Stream sync timeout: stream=AICPU timeout_ms=%d device_id=%d block_dim=%d",
            PLATFORM_STREAM_SYNC_TIMEOUT_MS, device_id_, block_dim_
        );
        return rc;
    }
    if (rc != 0) {
        LOG_ERROR("aclrtSynchronizeStreamWithTimeout (AICPU) failed: %d", rc);
        return rc;
    }

    LOG_INFO_V0("=== aclrtSynchronizeStreamWithTimeout stream_aicore_ ===");
    rc = aclrtSynchronizeStreamWithTimeout(stream_aicore_, PLATFORM_STREAM_SYNC_TIMEOUT_MS);
    if (rc == ACL_ERROR_RT_STREAM_SYNC_TIMEOUT) {
        LOG_ERROR(
            "Stream sync timeout: stream=AICore timeout_ms=%d device_id=%d block_dim=%d",
            PLATFORM_STREAM_SYNC_TIMEOUT_MS, device_id_, block_dim_
        );
        return rc;
    }
    if (rc != 0) {
        LOG_ERROR("aclrtSynchronizeStreamWithTimeout (AICore) failed: %d", rc);
        return rc;
    }

    // Pull the platform-level device wall (ns) back from the dedicated
    // 8-byte device buffer (AICPU writes via KernelArgs::device_wall_data_base
    // — CANN copies args at launch so the inline field would be unreachable).
    device_wall_ns_ = 0;
    if (device_wall_dev_ptr_ != nullptr) {
        int wall_rc = rtMemcpy(
            &device_wall_ns_, sizeof(uint64_t), device_wall_dev_ptr_, sizeof(uint64_t), RT_MEMCPY_DEVICE_TO_HOST
        );
        if (wall_rc != 0) {
            LOG_WARN("rtMemcpy(device_wall_ns) D2H failed: %d", wall_rc);
            device_wall_ns_ = 0;
        }
    }

    // Tear down collectors. stop() joins mgmt then collector in the only
    // safe order (mgmt's final-drain pass into L2 has poll as its
    // consumer). Diagnostic exports use the per-task `output_prefix_`
    // directory the user set on CallConfig (validate() enforces non-empty
    // upstream).
    if (enable_l2_swimlane_) {
        l2_perf_collector_.stop();
        l2_perf_collector_.read_phase_header_metadata();
        l2_perf_collector_.reconcile_counters();
        l2_perf_collector_.export_swimlane_json();
    }

    if (enable_dump_tensor_) {
        dump_collector_.stop();
        dump_collector_.reconcile_counters();
        dump_collector_.export_dump_files();
    }

    if (enable_pmu_) {
        pmu_collector_.stop();
        pmu_collector_.reconcile_counters();
    }

    if (enable_scope_stats_) {
        scope_stats_collector_.stop();
        scope_stats_collector_.reconcile_counters();
        scope_stats_collector_.write_jsonl(output_prefix_);
    }

    // Print handshake results (reads from device memory, must be before free)
    print_handshake_results();

    return 0;
}

// `print_handshake_results`, `prepare_orch_so`, `register_callable`,
// `register_callable_host_orch`, `unregister_callable`, `has_callable`,
// `bind_callable_to_runtime`, and `upload_chip_callable_buffer` live on
// `DeviceRunnerBase`.

int DeviceRunner::finalize() {
    if (device_id_ == -1) {
        return 0;
    }

    int rc = attach_current_thread(device_id_);
    if (rc != 0) {
        LOG_ERROR("Failed to attach finalize thread to device %d: %d", device_id_, rc);
        return rc;
    }

    // Cleanup all profiling subsystems (free shm + per-buffer dev/host
    // shadows). All four shared collectors use the same alloc/free shape
    // on a5: no unregister callback (a5 doesn't use halHostRegister) +
    // prof_free_cb (rtFree directly).
    if (l2_perf_collector_.is_initialized()) {
        l2_perf_collector_.finalize(/*unregister_cb=*/nullptr, prof_free_cb);
    }
    if (dump_collector_.is_initialized()) {
        dump_collector_.finalize(/*unregister_cb=*/nullptr, prof_free_cb);
    }
    if (pmu_collector_.is_initialized()) {
        pmu_collector_.finalize(/*unregister_cb=*/nullptr, prof_free_cb);
    }
    if (scope_stats_collector_.is_initialized()) {
        scope_stats_collector_.finalize(/*unregister_cb=*/nullptr, prof_free_cb);
        kernel_args_.args.scope_stats_data_base = 0;
    }

    // Shared cleanup body — streams, kernel_args, callable/orch maps,
    // chip-callable buffer pool, the three arenas, device_wall, and
    // mem_alloc_.finalize(). The a5-specific cached arena sizes still need
    // clearing here.
    rc = finalize_common();
    cached_gm_heap_size_ = 0;
    cached_gm_sm_size_ = 0;
    cached_runtime_arena_size_ = 0;

    int reset_rc = rtDeviceReset(device_id_);
    if (reset_rc != 0) {
        LOG_ERROR("rtDeviceReset(%d) failed during finalize: %d", device_id_, reset_rc);
        if (rc == 0) rc = reset_rc;
    }

    device_id_ = -1;
    LOG_INFO_V0("DeviceRunner finalized");
    return rc;
}

// `launch_aicpu_kernel` and `launch_aicore_kernel` live on `DeviceRunnerBase`.

void DeviceRunner::finalize_collectors() {
    if (l2_perf_collector_.is_initialized()) {
        l2_perf_collector_.stop();
    }
    if (dump_collector_.is_initialized()) {
        dump_collector_.stop();
    }
    if (pmu_collector_.is_initialized()) {
        pmu_collector_.stop();
    }
}

int DeviceRunner::init_l2_perf(int num_aicore, int device_id) {
    int rc = l2_perf_collector_.initialize(
        num_aicore, device_id, l2_perf_level_, prof_alloc_cb, /*register_cb=*/nullptr, prof_free_cb, output_prefix_
    );
    if (rc == 0) {
        kernel_args_.args.l2_perf_data_base =
            reinterpret_cast<uint64_t>(l2_perf_collector_.get_l2_perf_setup_device_ptr());
        kernel_args_.args.aicore_l2_perf_ring_addrs =
            reinterpret_cast<uint64_t>(l2_perf_collector_.get_aicore_ring_addrs_device_ptr());
    }
    return rc;
}

int DeviceRunner::init_tensor_dump(Runtime &runtime, int device_id) {
    int num_dump_threads = runtime.aicpu_thread_num;

    int rc = dump_collector_.initialize(
        num_dump_threads, device_id, prof_alloc_cb, /*register_cb=*/nullptr, prof_free_cb, output_prefix_
    );
    if (rc != 0) {
        return rc;
    }

    kernel_args_.args.dump_data_base = reinterpret_cast<uint64_t>(dump_collector_.get_dump_shm_device_ptr());
    return 0;
}

int DeviceRunner::init_pmu(
    int num_cores, int num_threads, const std::string &csv_path, PmuEventType event_type, int device_id
) {
    int rc = pmu_collector_.init(
        num_cores, num_threads, csv_path, event_type, prof_alloc_cb, /*register_cb=*/nullptr, prof_free_cb, device_id
    );
    if (rc == 0) {
        kernel_args_.args.pmu_data_base = reinterpret_cast<uint64_t>(pmu_collector_.get_pmu_shm_device_ptr());
        kernel_args_.args.aicore_pmu_ring_addrs =
            reinterpret_cast<uint64_t>(pmu_collector_.get_aicore_ring_addrs_device_ptr());
    }
    return rc;
}

int DeviceRunner::init_scope_stats(int num_threads, int device_id) {
    // a5: register_cb=nullptr, so the collector mallocs a host shadow per
    // device buffer + rtMemcpy's the zeroed shadow to device (see
    // ScopeStatsCollector::alloc_single_buffer). No halHostRegister on a5.
    int rc = scope_stats_collector_.init(num_threads, prof_alloc_cb, /*register_cb=*/nullptr, prof_free_cb, device_id);
    if (rc != 0) {
        return rc;
    }
    kernel_args_.args.scope_stats_data_base =
        reinterpret_cast<uint64_t>(scope_stats_collector_.get_scope_stats_shm_device_ptr());
    return 0;
}
