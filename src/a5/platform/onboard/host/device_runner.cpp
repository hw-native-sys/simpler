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

int DeviceRunner::setup_static_arena(size_t gm_heap_size, size_t gm_sm_size, size_t runtime_arena_size) {
    // Three independent device_malloc'd buffers: GM heap, PTO2 SM, prebuilt
    // runtime arena. Split out from a single large allocation because the
    // combined size can exceed the device allocator's largest contiguous
    // block. Each arena commits exactly one region, so its base() is the
    // pooled pointer the caller wants.
    //
    // Idempotent for the production case (sizes do not change across a
    // worker's lifetime). If a caller asks for a larger layout on any
    // region, redo just that region — already-committed peers stay alive
    // so their callers don't have to re-acquire.
    auto commit_region = [](DeviceArena &arena, size_t &cached_size, size_t requested_size) -> int {
        if (requested_size == 0) {
            // hbg's runtime_arena path: caller passed 0 and never reserved
            // a region. Leave the arena uncommitted; acquire_pooled_* will
            // return nullptr.
            if (arena.is_committed() && cached_size != 0) {
                arena.release();
                cached_size = 0;
            }
            return 0;
        }
        if (arena.is_committed() && requested_size <= cached_size) {
            return 0;
        }
        arena.release();
        cached_size = 0;
        arena.reserve(requested_size, DeviceArena::kDefaultBaseAlign);
        if (arena.commit(DeviceArena::kDefaultBaseAlign) == nullptr) {
            // commit() failure leaves committed_=false, so the next entry's
            // is_committed() guard skips the release branch. release() is
            // idempotent on a never-committed arena (zeroes cursor_).
            arena.release();
            return -1;
        }
        cached_size = requested_size;
        return 0;
    };
    if (commit_region(gm_heap_arena_, cached_gm_heap_size_, gm_heap_size) != 0) return -1;
    if (commit_region(gm_sm_arena_, cached_gm_sm_size_, gm_sm_size) != 0) {
        gm_heap_arena_.release();
        cached_gm_heap_size_ = 0;
        return -1;
    }
    if (commit_region(runtime_arena_pool_, cached_runtime_arena_size_, runtime_arena_size) != 0) {
        gm_heap_arena_.release();
        gm_sm_arena_.release();
        cached_gm_heap_size_ = 0;
        cached_gm_sm_size_ = 0;
        return -1;
    }
    return 0;
}

std::thread DeviceRunner::create_thread(std::function<void()> fn) {
    int dev_id = device_id_;
    return std::thread([dev_id, fn = std::move(fn)]() {
        rtSetDevice(dev_id);
        fn();
    });
}

int DeviceRunner::ensure_device_initialized() {
    // Attach the current thread to the device (device_id_ was set in
    // attach_current_thread() during simpler_init) and create the persistent
    // AICPU/AICore streams. Streams live for the DeviceRunner's lifetime and
    // are destroyed in finalize().
    int rc = attach_current_thread(device_id_);
    if (rc != 0) {
        return rc;
    }

    bool aicpu_created_here = false;
    bool aicore_created_here = false;
    if (stream_aicpu_ == nullptr) {
        rc = rtStreamCreate(&stream_aicpu_, 0);
        if (rc != 0) {
            LOG_ERROR("rtStreamCreate (AICPU) failed: %d", rc);
            return rc;
        }
        aicpu_created_here = true;
    }
    if (stream_aicore_ == nullptr) {
        rc = rtStreamCreate(&stream_aicore_, 0);
        if (rc != 0) {
            LOG_ERROR("rtStreamCreate (AICore) failed: %d", rc);
            // Roll back only the AICPU stream we just created, not a
            // pre-existing persistent one.
            if (aicpu_created_here) {
                rtStreamDestroy(stream_aicpu_);
                stream_aicpu_ = nullptr;
            }
            return rc;
        }
        aicore_created_here = true;
    }
    if (aicpu_created_here || aicore_created_here) {
        LOG_INFO_V0("DeviceRunner: device=%d set, streams created", device_id_);
    }

    return ensure_binaries_loaded();
}

int DeviceRunner::attach_current_thread(int device_id) {
    if (device_id < 0) {
        LOG_ERROR("Invalid device_id: %d", device_id);
        return -1;
    }
    if (device_id_ != -1 && device_id_ != device_id) {
        LOG_ERROR(
            "DeviceRunner already initialized on device %d; reset/finalize before switching to device %d", device_id_,
            device_id
        );
        return -1;
    }

    // CANN device context is per-thread, so every caller must attach explicitly.
    int rc = rtSetDevice(device_id);
    if (rc != 0) {
        LOG_ERROR("rtSetDevice(%d) failed: %d", device_id, rc);
        return rc;
    }

    if (device_id_ == -1) {
        configure_aicore_op_timeout();
    }

    device_id_ = device_id;
    return 0;
}

void DeviceRunner::configure_aicore_op_timeout() {
    uint64_t actual_timeout = 0;
    int rc = aclrtSetOpExecuteTimeOutV2(PLATFORM_OP_EXECUTE_TIMEOUT_US, &actual_timeout);
    if (rc != 0) {
        LOG_ERROR(
            "aclrtSetOpExecuteTimeOutV2(%llu us) failed: %d", (unsigned long long)PLATFORM_OP_EXECUTE_TIMEOUT_US, rc
        );
    } else {
        LOG_INFO_V0(
            "aclrtSetOpExecuteTimeOutV2: requested=%llu us, actual=%llu us",
            (unsigned long long)PLATFORM_OP_EXECUTE_TIMEOUT_US, (unsigned long long)actual_timeout
        );
    }
}

int DeviceRunner::ensure_binaries_loaded() {
    // Check if already loaded (binaries owned by the runner via set_executors).
    if (binaries_loaded_) {
        return 0;
    }

    // Device must be set first
    if (stream_aicpu_ == nullptr) {
        LOG_ERROR("Device not set before loading binaries");
        return -1;
    }

    if (dispatcher_so_binary_.empty()) {
        LOG_ERROR(
            "DeviceRunner: dispatcher SO bytes not provided; pass dispatcher_path through ChipWorker.init "
            "(RuntimeBinaries.dispatcher_path)"
        );
        return -1;
    }

    // One-shot bootstrap: libaicpu_extend_kernels invokes our dispatcher,
    // which writes the runtime AICPU SO bytes to simpler_inner_<fp>_<device_id>.so in
    // the device-side preinstall path. The dispatcher SO itself is never
    // persisted. Subsequent per-task AICPU launches resolve symbols via
    // rtsBinaryLoadFromFile + rtsFuncGetByName + rtsLaunchCpuKernel
    // directly against that preinstall file.
    int rc = load_aicpu_op_.BootstrapDispatcher(
        dispatcher_so_binary_.data(), dispatcher_so_binary_.size(), aicpu_so_binary_.data(), aicpu_so_binary_.size(),
        stream_aicpu_, device_id_
    );
    if (rc != 0) {
        LOG_ERROR("LoadAicpuOp::BootstrapDispatcher failed: %d", rc);
        return rc;
    }
    LOG_INFO_V2("DeviceRunner: inner SO uploaded to preinstall via dispatcher bootstrap");

    rc = load_aicpu_op_.Init();
    if (rc != 0) {
        LOG_ERROR("LoadAicpuOp::Init failed: %d", rc);
        return rc;
    }
    LOG_INFO_V2("DeviceRunner: inner SO registered (simpler_aicpu_init/exec handles ready)");

    // H2D the per-task DeviceArgs struct itself. device_args_.aicpu_so_bin/len
    // stay zero — our own per-task AICPU code (launched via rtsLaunchCpuKernel
    // against the cached rtFuncHandle on LoadAicpuOp) never reads them, and
    // the dispatcher-bootstrap KernelArgs (KERNEL_TYPE_AICPU_KFC) builds its
    // own DeviceArgs view inside BootstrapDispatcher rather than reading
    // ours. The "load-bearing on a5" finding documented prior to #864/#870
    // no longer reproduces against current HEAD — see PR removing
    // AicpuSoInfo (CI on both archs green).
    rc = kernel_args_.init_device_args(device_args_, mem_alloc_);
    if (rc != 0) {
        LOG_ERROR("init_device_args failed: %d", rc);
        return rc;
    }

    // Release host bytes — bootstrap is done. Per-task launches go through
    // the cached rtFuncHandle owned by LoadAicpuOp; dispatcher SO bytes are
    // never referenced again; the aicpu kernel SO's host buffer is no longer
    // needed either (we used to H2D it through AicpuSoInfo as a CANN-internal
    // bookkeeping workaround; that's gone).
    dispatcher_so_binary_.clear();
    dispatcher_so_binary_.shrink_to_fit();
    aicpu_so_binary_.clear();
    aicpu_so_binary_.shrink_to_fit();

    binaries_loaded_ = true;
    LOG_INFO_V0("DeviceRunner: binaries loaded");
    return 0;
}

int DeviceRunner::query_max_block_dim(rtStream_t stream, uint32_t *out_cube, uint32_t *out_vector) {
    uint32_t cube_limit = 0, vector_limit = 0;
    bool got_limits = (aclrtGetStreamResLimit(stream, ACL_RT_DEV_RES_CUBE_CORE, &cube_limit) == ACL_ERROR_NONE) &&
                      (aclrtGetStreamResLimit(stream, ACL_RT_DEV_RES_VECTOR_CORE, &vector_limit) == ACL_ERROR_NONE) &&
                      cube_limit > 0 && vector_limit > 0;
    if (out_cube != nullptr) *out_cube = got_limits ? cube_limit : 0;
    if (out_vector != nullptr) *out_vector = got_limits ? vector_limit : 0;
    if (got_limits) {
        // Cap by PLATFORM_MAX_BLOCKDIM as well: runtime handshake/scheduler
        // arrays are statically sized to RUNTIME_MAX_WORKER (= PLATFORM_MAX_BLOCKDIM
        // * PLATFORM_CORES_PER_BLOCKDIM), so even if ACL reports more cores
        // than the platform cap we must not exceed it.
        int from_stream = static_cast<int>(
            std::min(cube_limit / PLATFORM_AIC_CORES_PER_BLOCKDIM, vector_limit / PLATFORM_AIV_CORES_PER_BLOCKDIM)
        );
        return std::min(from_stream, PLATFORM_MAX_BLOCKDIM);
    }
    return PLATFORM_MAX_BLOCKDIM;
}

int DeviceRunner::validate_block_dim(rtStream_t stream, int block_dim) {
    if (block_dim < 1) {
        LOG_ERROR("block_dim (%d) must be >= 1", block_dim);
        return -1;
    }
    uint32_t cube_limit = 0, vector_limit = 0;
    int max_bd = query_max_block_dim(stream, &cube_limit, &vector_limit);
    if (block_dim > max_bd) {
        if (cube_limit > 0 && vector_limit > 0) {
            LOG_ERROR(
                "block_dim (%d) exceeds available cores (max_block_dim=%d, cube=%u, vector=%u)", block_dim, max_bd,
                cube_limit, vector_limit
            );
        } else {
            LOG_ERROR(
                "aclrtGetStreamResLimit unavailable; block_dim (%d) exceeds static cap PLATFORM_MAX_BLOCKDIM (%d)",
                block_dim, PLATFORM_MAX_BLOCKDIM
            );
        }
        return -1;
    }
    return 0;
}

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

void DeviceRunner::print_handshake_results() {
    if (stream_aicpu_ == nullptr || worker_count_ == 0 || kernel_args_.args.runtime_args == nullptr) {
        return;
    }

    // Allocate temporary buffer to read handshake data from device
    std::vector<Handshake> workers(worker_count_);
    size_t total_size = sizeof(Handshake) * worker_count_;
    rtMemcpy(workers.data(), total_size, kernel_args_.args.runtime_args->workers, total_size, RT_MEMCPY_DEVICE_TO_HOST);

    LOG_DEBUG("Handshake results for %d cores:", worker_count_);
    for (int i = 0; i < worker_count_; i++) {
        LOG_DEBUG(
            "  Core %d: aicore_done=%d aicpu_ready=%d task=%d", i, workers[i].aicore_done, workers[i].aicpu_ready,
            workers[i].task
        );
    }
}

int DeviceRunner::prepare_orch_so(Runtime &runtime) {
    // Prepared-callable flow only: bytes were H2D'd at prepare_callable
    // time. Stamp dev_orch_so on the runtime and resolve `is_new` from
    // first sighting.
    const int32_t cid = runtime.get_active_callable_id();
    if (cid < 0) {
        LOG_ERROR("prepare_orch_so: no active callable_id; prepared-callable flow required");
        return -1;
    }
    auto it = prepared_callables_.find(cid);
    if (it == prepared_callables_.end()) {
        LOG_ERROR("prepare_orch_so: callable_id=%d not registered", cid);
        return -1;
    }
    const auto &state = it->second;
    // hbg variant: orch SO never crosses host/device, so AICPU does no
    // per-cid dlopen. Skip orch_so_table_ bookkeeping and clear metadata.
    if (state.host_dlopen_handle != nullptr) {
        runtime.set_dev_orch_so(0, 0);
        runtime.set_active_callable_id(cid, /*is_new=*/false);
        return 0;
    }
    const bool first_sighting = aicpu_seen_callable_ids_.insert(cid).second;
    if (first_sighting) {
        ++aicpu_dlopen_total_;
    }
    runtime.set_dev_orch_so(state.dev_orch_so_addr, state.dev_orch_so_size);
    // The c_api caller passed is_new=false; refresh with the authoritative
    // first_sighting flag before AICPU consumes register_new_callable_id_.
    runtime.set_active_callable_id(cid, first_sighting);
    LOG_INFO_V0(
        "Orch SO prepared cid=%d hash=0x%lx %zu bytes (is_new=%d)", cid, state.hash, state.dev_orch_so_size,
        first_sighting ? 1 : 0
    );
    return 0;
}

int DeviceRunner::register_prepared_callable(
    int32_t callable_id, const void *orch_so_data, size_t orch_so_size, const char *func_name, const char *config_name,
    std::vector<std::pair<int, uint64_t>> kernel_addrs, std::vector<ArgDirection> signature
) {
    // The AICPU executor reserves `orch_so_table_[MAX_REGISTERED_CALLABLE_IDS]`
    // (declared in src/common/task_interface/callable_protocol.h) and indexes
    // it by callable_id; rejecting an out-of-range id here keeps host and AICPU
    // in sync and avoids an OOB access at run time.
    if (callable_id < 0 || callable_id >= MAX_REGISTERED_CALLABLE_IDS) {
        LOG_ERROR(
            "register_prepared_callable: callable_id=%d out of range [0, %d)", callable_id, MAX_REGISTERED_CALLABLE_IDS
        );
        return -1;
    }
    if (orch_so_data == nullptr || orch_so_size == 0) {
        LOG_ERROR("register_prepared_callable: empty orch SO for callable_id=%d", callable_id);
        return -1;
    }
    if (prepared_callables_.count(callable_id) != 0) {
        LOG_ERROR("register_prepared_callable: callable_id=%d already registered", callable_id);
        return -1;
    }

    const uint64_t hash = simpler::common::utils::elf_build_id_64(orch_so_data, orch_so_size);

    // Hash dedup: share device buffer across callable_ids that carry the same
    // SO bytes. Refcount drops in unregister_prepared_callable; we only free
    // when the count hits zero.
    auto buf_it = orch_so_dedup_.find(hash);
    uint64_t dev_addr = 0;
    if (buf_it == orch_so_dedup_.end()) {
        void *buf = mem_alloc_.alloc(orch_so_size);
        if (buf == nullptr) {
            LOG_ERROR("register_prepared_callable: alloc %zu bytes failed", orch_so_size);
            return -1;
        }
        int rc = rtMemcpy(buf, orch_so_size, orch_so_data, orch_so_size, RT_MEMCPY_HOST_TO_DEVICE);
        if (rc != 0) {
            LOG_ERROR("register_prepared_callable: rtMemcpy failed: %d", rc);
            mem_alloc_.free(buf);
            return rc;
        }
        OrchSoBuffer entry;
        entry.dev_addr = buf;
        entry.capacity = orch_so_size;
        entry.refcount = 1;
        orch_so_dedup_.emplace(hash, entry);
        dev_addr = reinterpret_cast<uint64_t>(buf);
        LOG_INFO_V0("register_prepared_callable: hash=0x%lx new buffer %zu bytes", hash, orch_so_size);
    } else {
        buf_it->second.refcount++;
        dev_addr = reinterpret_cast<uint64_t>(buf_it->second.dev_addr);
        LOG_INFO_V0(
            "register_prepared_callable: hash=0x%lx shared buffer (refcount=%d)", hash, buf_it->second.refcount
        );
    }

    PreparedCallableState state;
    state.hash = hash;
    state.dev_orch_so_addr = dev_addr;
    state.dev_orch_so_size = orch_so_size;
    state.func_name = (func_name != nullptr) ? func_name : "";
    state.config_name = (config_name != nullptr) ? config_name : "";
    state.kernel_addrs = std::move(kernel_addrs);
    state.signature = std::move(signature);
    prepared_callables_.emplace(callable_id, std::move(state));
    return 0;
}

int DeviceRunner::register_prepared_callable_host_orch(
    int32_t callable_id, void *host_dlopen_handle, void *host_orch_func_ptr,
    std::vector<std::pair<int, uint64_t>> kernel_addrs, std::vector<ArgDirection> signature
) {
    if (callable_id < 0 || callable_id >= MAX_REGISTERED_CALLABLE_IDS) {
        LOG_ERROR(
            "register_prepared_callable_host_orch: callable_id=%d out of range [0, %d)", callable_id,
            MAX_REGISTERED_CALLABLE_IDS
        );
        return -1;
    }
    if (host_dlopen_handle == nullptr || host_orch_func_ptr == nullptr) {
        LOG_ERROR("register_prepared_callable_host_orch: null handle/fn for callable_id=%d", callable_id);
        return -1;
    }
    if (prepared_callables_.count(callable_id) != 0) {
        LOG_ERROR("register_prepared_callable_host_orch: callable_id=%d already registered", callable_id);
        return -1;
    }

    PreparedCallableState state;
    state.host_dlopen_handle = host_dlopen_handle;
    state.host_orch_func_ptr = host_orch_func_ptr;
    state.kernel_addrs = std::move(kernel_addrs);
    state.signature = std::move(signature);
    prepared_callables_.emplace(callable_id, std::move(state));
    ++host_dlopen_total_;
    LOG_INFO_V0("register_prepared_callable_host_orch: cid=%d (host dlopen #%zu)", callable_id, host_dlopen_total_);
    return 0;
}

int DeviceRunner::unregister_prepared_callable(int32_t callable_id) {
    auto it = prepared_callables_.find(callable_id);
    if (it == prepared_callables_.end()) {
        return 0;
    }
    PreparedCallableState state = std::move(it->second);
    prepared_callables_.erase(it);
    aicpu_seen_callable_ids_.erase(callable_id);

    if (state.host_dlopen_handle != nullptr) {
        // hbg path: dlclose the host handle; no orch SO refcount to decrement.
        dlclose(state.host_dlopen_handle);
        return 0;
    }

    auto buf_it = orch_so_dedup_.find(state.hash);
    if (buf_it != orch_so_dedup_.end()) {
        if (--buf_it->second.refcount <= 0) {
            mem_alloc_.free(buf_it->second.dev_addr);
            orch_so_dedup_.erase(buf_it);
        }
    }
    return 0;
}

bool DeviceRunner::has_prepared_callable(int32_t callable_id) const {
    return prepared_callables_.count(callable_id) != 0;
}

BindPreparedCallableResult DeviceRunner::bind_prepared_callable_to_runtime(Runtime &runtime, int32_t callable_id) {
    auto it = prepared_callables_.find(callable_id);
    if (it == prepared_callables_.end()) {
        LOG_ERROR("bind_prepared_callable_to_runtime: callable_id=%d not registered", callable_id);
        return {-1, nullptr, nullptr, 0};
    }
    const auto &state = it->second;

    // Replay kernel addresses directly into runtime.func_id_to_addr_ without
    // going through set_function_bin_addr — the latter would record func_ids
    // in registered_kernel_func_ids_, which validate_runtime_impl iterates to
    // free kernel binaries. Prepared kernels must survive across runs and only
    // be freed by finalize().
    for (const auto &kv : state.kernel_addrs) {
        if (kv.first < 0 || kv.first >= RUNTIME_MAX_FUNC_ID) {
            LOG_ERROR("bind_prepared_callable_to_runtime: func_id=%d out of range", kv.first);
            return {-1, nullptr, nullptr, 0};
        }
        runtime.replay_function_bin_addr(kv.first, kv.second);
    }
    runtime.set_device_orch_func_name(state.func_name.c_str());
    runtime.set_device_orch_config_name(state.config_name.c_str());
    // Stamp callable_id with is_new=false; prepare_orch_so refreshes the flag
    // with the authoritative first_sighting answer right before launch.
    runtime.set_active_callable_id(callable_id, /*is_new=*/false);
    return {
        0, state.host_orch_func_ptr, state.signature.empty() ? nullptr : state.signature.data(),
        static_cast<int>(state.signature.size())
    };
}

int DeviceRunner::finalize() {
    if (device_id_ == -1) {
        return 0;
    }

    int rc = attach_current_thread(device_id_);
    if (rc != 0) {
        LOG_ERROR("Failed to attach finalize thread to device %d: %d", device_id_, rc);
        return rc;
    }

    // Streams are persistent for the DeviceRunner's lifetime; destroy them here.
    // Intentionally no pre-destroy sync: when a run hits the AICore op-timeout
    // chain (PR #718), the AICPU stream surfaces ACL_ERROR_RT_AICPU_EXCEPTION
    // (507018) at run-path sync; calling aclrtSynchronizeStream* again on the
    // error-state stream at finalize wedges subsequent tests (observed: 507018
    // / 507899 / 507901 cascade across the whole st-onboard-a2a3 suite).
    // rtStreamDestroy on an error-state stream is the supported teardown path.
    if (stream_aicpu_ != nullptr) {
        rtStreamDestroy(stream_aicpu_);
        stream_aicpu_ = nullptr;
    }
    if (stream_aicore_ != nullptr) {
        rtStreamDestroy(stream_aicore_);
        stream_aicore_ = nullptr;
    }

    // Cleanup kernel args (deviceArgs); device-side KernelArgs + runtime args
    // are released by runtime_args_cleanup RAII so they also unwind on errors.
    kernel_args_.finalize_device_args();

    // load_aicpu_op_ has no per-task host-side state to release —
    // rtsLaunchCpuKernel does not hand back any per-launch handle, and the
    // dispatcher itself was a transient libaicpu_extend_kernels dlopen.
    // aicore_bin_handle_ was registered once via rtRegisterAllKernel; CANN
    // releases its device-side state when the device context tears down.
    aicore_bin_handle_ = nullptr;
    binaries_loaded_ = false;

    // Release any chip callable buffers uploaded via upload_chip_callable_buffer.
    // Pool semantics mirror per-fid binaries: never freed until finalize.
    for (auto &kv : chip_callable_buffers_) {
        mem_alloc_.free(reinterpret_cast<void *>(kv.second.chip_dev));
        LOG_DEBUG(
            "Freed chip callable buffer: chip_dev=0x%lx, size=%zu, hash=0x%lx", kv.second.chip_dev,
            kv.second.total_size, kv.first
        );
    }
    chip_callable_buffers_.clear();

    // Release any prepared-callable orch SO buffers that callers forgot to
    // unregister. Refcounts no longer matter at this point — the device is
    // about to be reset.
    for (auto &kv : orch_so_dedup_) {
        if (kv.second.dev_addr != nullptr) {
            mem_alloc_.free(kv.second.dev_addr);
        }
    }
    orch_so_dedup_.clear();
    // hbg path: dlclose any host orch handles callers forgot to unregister.
    // finalize() is the last chance; Worker.close() does not auto-unregister
    // each callable_id, so without this loop the host process leaks one
    // dlopen handle per (re)created Worker — observable in long-running
    // pytest sessions.
    for (auto &kv : prepared_callables_) {
        if (kv.second.host_dlopen_handle != nullptr) {
            dlclose(kv.second.host_dlopen_handle);
        }
    }
    prepared_callables_.clear();
    aicpu_seen_callable_ids_.clear();
    aicpu_dlopen_total_ = 0;

    // Cleanup all profiling subsystems (free shm + per-buffer dev/host
    // shadows). All three collectors share the same alloc/free shape.
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

    // Release the three per-Worker pooled arenas (GM heap, PTO2 SM, optional
    // trb prebuilt runtime arena — each its own device_malloc). Must precede
    // mem_alloc_.finalize() so the arenas free through the still-live
    // allocator, not after it.
    gm_heap_arena_.release();
    gm_sm_arena_.release();
    runtime_arena_pool_.release();
    cached_gm_heap_size_ = 0;
    cached_gm_sm_size_ = 0;
    cached_runtime_arena_size_ = 0;

    // Free the 8-byte device_wall buffer (allocated lazily in run()) while
    // mem_alloc_ and the device context are still live. free_tensor() routes
    // through mem_alloc_.free(), so it must run before finalize() and before
    // rtDeviceReset() tears down the device runtime.
    if (device_wall_dev_ptr_ != nullptr) {
        free_tensor(device_wall_dev_ptr_);
        device_wall_dev_ptr_ = nullptr;
    }

    // Free all remaining allocations (including handshake buffer and binGmAddr)
    mem_alloc_.finalize();

    rc = rtDeviceReset(device_id_);
    if (rc != 0) {
        LOG_ERROR("rtDeviceReset(%d) failed during finalize: %d", device_id_, rc);
        return rc;
    }

    device_id_ = -1;
    block_dim_ = 0;
    worker_count_ = 0;
    aicore_kernel_binary_.clear();

    LOG_INFO_V0("DeviceRunner finalized");
    return 0;
}

int DeviceRunner::launch_aicpu_kernel(rtStream_t stream, KernelArgs *k_args, const char *kernel_name, int aicpu_num) {
    // kernel_name is host::KernelNames::InitName / RunName — the runtime SO's
    // actual exported symbol (simpler_aicpu_init / simpler_aicpu_exec).
    // LaunchBuiltInOp dispatches via rtsLaunchCpuKernel on the cached
    // rtFuncHandle resolved by LoadAicpuOp::Init at first-time bootstrap.
    return load_aicpu_op_.LaunchBuiltInOp(stream, k_args, aicpu_num, kernel_name);
}

int DeviceRunner::launch_aicore_kernel(rtStream_t stream, KernelArgs *k_args) {
    // Lazy-register the AICore binary on first call; reuse cached handle
    // thereafter. CANN has no public rtUnregisterAllKernel, so re-registering
    // every run would pin another device-side copy of the ELF (~365KB on a5)
    // and quickly exhaust HBM — surfaced in CI as 207001 at
    // rtKernelLaunchWithHandleV2 with a 507899 cascade at rtStreamCreate.
    if (aicore_bin_handle_ == nullptr) {
        if (aicore_kernel_binary_.empty()) {
            LOG_ERROR("AICore kernel binary is empty");
            return -1;
        }
        rtDevBinary_t binary;
        std::memset(&binary, 0, sizeof(binary));
        binary.magic = RT_DEV_BINARY_MAGIC_ELF;
        binary.version = 0;
        binary.data = aicore_kernel_binary_.data();
        binary.length = aicore_kernel_binary_.size();
        int rc = rtRegisterAllKernel(&binary, &aicore_bin_handle_);
        if (rc != RT_ERROR_NONE) {
            LOG_ERROR("rtRegisterAllKernel failed: %d", rc);
            aicore_bin_handle_ = nullptr;
            return rc;
        }
    }

    struct Args {
        KernelArgs *k_args;
    };
    // Pass device address of KernelArgs to AICore (KERNEL_ENTRY signature).
    Args args = {k_args};
    rtArgsEx_t rt_args;
    std::memset(&rt_args, 0, sizeof(rt_args));
    rt_args.args = &args;
    rt_args.argsSize = sizeof(args);

    rtTaskCfgInfo_t cfg = {};
    cfg.schemMode = RT_SCHEM_MODE_BATCH;

    int rc = rtKernelLaunchWithHandleV2(aicore_bin_handle_, 0, block_dim_, &rt_args, nullptr, stream, &cfg);
    if (rc != RT_ERROR_NONE) {
        LOG_ERROR("rtKernelLaunchWithHandleV2 failed: %d", rc);
        return rc;
    }

    return rc;
}

// =============================================================================
// Chip Callable Buffer Upload (returns device address of ChipCallable header)
// =============================================================================

uint64_t DeviceRunner::upload_chip_callable_buffer(const ChipCallable *callable) {
    if (callable == nullptr || callable->child_count() == 0) {
        return 0;
    }
    if (stream_aicpu_ == nullptr) {
        LOG_ERROR("Run context not prepared before upload_chip_callable_buffer()");
        return 0;
    }

    constexpr size_t kHeaderSize = offsetof(ChipCallable, storage_);
    size_t storage_used = static_cast<size_t>(callable->binary_size());
    for (int32_t i = 0; i < callable->child_count(); ++i) {
        const CoreCallable &c = callable->child(i);
        size_t child_total = CoreCallable::binary_data_offset() + static_cast<size_t>(c.binary_size());
        size_t end = static_cast<size_t>(callable->child_offset(i)) + child_total;
        if (end > storage_used) storage_used = end;
    }
    const size_t total_size = kHeaderSize + storage_used;

    const auto *raw_bytes = reinterpret_cast<const uint8_t *>(callable);
    const uint64_t hash = simpler::common::utils::fnv1a_64(raw_bytes, total_size);
    auto it = chip_callable_buffers_.find(hash);
    if (it != chip_callable_buffers_.end()) {
        LOG_DEBUG(
            "Chip callable dedup hit: chip_dev=0x%lx, size=%zu, hash=0x%lx", it->second.chip_dev, it->second.total_size,
            hash
        );
        return it->second.chip_dev;
    }

    void *gm_addr = mem_alloc_.alloc(total_size);
    if (gm_addr == nullptr) {
        LOG_ERROR("Failed to allocate device GM for ChipCallable buffer (size=%zu)", total_size);
        return 0;
    }
    const uint64_t chip_dev = reinterpret_cast<uint64_t>(gm_addr);
    assert((chip_dev & (CALLABLE_ALIGN - 1)) == 0 && "device alloc must be CALLABLE_ALIGN-byte aligned");

    std::vector<uint8_t> scratch(total_size);
    std::memcpy(scratch.data(), raw_bytes, total_size);
    for (int32_t i = 0; i < callable->child_count(); ++i) {
        const uint32_t off = callable->child_offset(i);
        auto *child_in_scratch = reinterpret_cast<CoreCallable *>(scratch.data() + kHeaderSize + off);
        uint64_t child_dev = chip_dev + kHeaderSize + off;
        child_in_scratch->set_resolved_addr(child_dev + CoreCallable::binary_data_offset());
    }

    int rc = rtMemcpy(gm_addr, total_size, scratch.data(), total_size, RT_MEMCPY_HOST_TO_DEVICE);
    if (rc != 0) {
        LOG_ERROR("rtMemcpy chip callable H2D failed: %d", rc);
        mem_alloc_.free(gm_addr);
        return 0;
    }

    chip_callable_buffers_.emplace(hash, ChipCallableBuffer{chip_dev, total_size});
    LOG_DEBUG(
        "Uploaded chip callable: chip_dev=0x%lx, size=%zu, child_count=%d, hash=0x%lx", chip_dev, total_size,
        callable->child_count(), hash
    );
    return chip_dev;
}

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
