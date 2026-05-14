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

#include <cassert>
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
// KernelArgsHelper Implementation
// =============================================================================

int KernelArgsHelper::init_device_args(const DeviceArgs &host_device_args, MemoryAllocator &allocator) {
    allocator_ = &allocator;

    // Allocate device memory for device_args
    if (args.device_args == nullptr) {
        uint64_t device_args_size = sizeof(DeviceArgs);
        void *device_args_dev = allocator_->alloc(device_args_size);
        if (device_args_dev == nullptr) {
            LOG_ERROR("Alloc for device_args failed");
            return -1;
        }
        args.device_args = reinterpret_cast<DeviceArgs *>(device_args_dev);
    }
    // Copy host_device_args to device memory via device_args
    int rc =
        rtMemcpy(args.device_args, sizeof(DeviceArgs), &host_device_args, sizeof(DeviceArgs), RT_MEMCPY_HOST_TO_DEVICE);
    if (rc != 0) {
        LOG_ERROR("rtMemcpy failed: %d", rc);
        allocator_->free(args.device_args);
        args.device_args = nullptr;
        return rc;
    }
    return 0;
}

int KernelArgsHelper::finalize_device_args() {
    if (args.device_args != nullptr && allocator_ != nullptr) {
        int rc = allocator_->free(args.device_args);
        args.device_args = nullptr;
        return rc;
    }
    return 0;
}

int KernelArgsHelper::init_runtime_args(const Runtime &host_runtime, MemoryAllocator &allocator) {
    allocator_ = &allocator;

    if (args.runtime_args == nullptr) {
        uint64_t runtime_size = sizeof(Runtime);
        void *runtime_dev = allocator_->alloc(runtime_size);
        if (runtime_dev == nullptr) {
            LOG_ERROR("Alloc for runtime_args failed");
            return -1;
        }
        args.runtime_args = reinterpret_cast<Runtime *>(runtime_dev);
    }
    int rc = rtMemcpy(args.runtime_args, sizeof(Runtime), &host_runtime, sizeof(Runtime), RT_MEMCPY_HOST_TO_DEVICE);
    if (rc != 0) {
        LOG_ERROR("rtMemcpy for runtime failed: %d", rc);
        allocator_->free(args.runtime_args);
        args.runtime_args = nullptr;
        return rc;
    }
    return 0;
}

int KernelArgsHelper::finalize_runtime_args() {
    if (args.runtime_args != nullptr && allocator_ != nullptr) {
        int rc = allocator_->free(args.runtime_args);
        args.runtime_args = nullptr;
        return rc;
    }
    return 0;
}

// =============================================================================
// AicpuSoInfo Implementation
// =============================================================================

int AicpuSoInfo::init(const std::vector<uint8_t> &aicpu_so_binary, MemoryAllocator &allocator) {
    allocator_ = &allocator;

    if (aicpu_so_binary.empty()) {
        LOG_ERROR("AICPU binary is empty");
        return -1;
    }

    size_t file_size = aicpu_so_binary.size();
    void *d_aicpu_data = allocator_->alloc(file_size);
    if (d_aicpu_data == nullptr) {
        LOG_ERROR("Alloc failed for AICPU SO");
        return -1;
    }

    int rc = rtMemcpy(d_aicpu_data, file_size, aicpu_so_binary.data(), file_size, RT_MEMCPY_HOST_TO_DEVICE);
    if (rc != 0) {
        LOG_ERROR("rtMemcpy failed: %d", rc);
        allocator_->free(d_aicpu_data);
        d_aicpu_data = nullptr;
        return rc;
    }

    aicpu_so_bin = reinterpret_cast<uint64_t>(d_aicpu_data);
    aicpu_so_len = file_size;
    return 0;
}

int AicpuSoInfo::finalize() {
    if (aicpu_so_bin != 0 && allocator_ != nullptr) {
        int rc = allocator_->free(reinterpret_cast<void *>(aicpu_so_bin));
        aicpu_so_bin = 0;
        return rc;
    }
    return 0;
}

// =============================================================================
// DeviceRunner Implementation
// =============================================================================

DeviceRunner::~DeviceRunner() { finalize(); }

std::thread DeviceRunner::create_thread(std::function<void()> fn) {
    int dev_id = device_id_;
    return std::thread([dev_id, fn = std::move(fn)]() {
        rtSetDevice(dev_id);
        fn();
    });
}

int DeviceRunner::ensure_device_initialized() {
    // First attach the current thread and create fresh run-scoped streams.
    // device_id_ was set in attach_current_thread() during simpler_init.
    int rc = prepare_run_context(device_id_);
    if (rc != 0) {
        return rc;
    }

    // Then ensure binaries are loaded
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

    device_id_ = device_id;
    return 0;
}

int DeviceRunner::prepare_run_context(int device_id) {
    int rc = attach_current_thread(device_id);
    if (rc != 0) {
        return rc;
    }

    if (stream_aicpu_ != nullptr && stream_aicore_ != nullptr) {
        return 0;
    }

    release_run_context();

    // Create streams
    rc = rtStreamCreate(&stream_aicpu_, 0);
    if (rc != 0) {
        LOG_ERROR("rtStreamCreate (AICPU) failed: %d", rc);
        return rc;
    }

    rc = rtStreamCreate(&stream_aicore_, 0);
    if (rc != 0) {
        LOG_ERROR("rtStreamCreate (AICore) failed: %d", rc);
        rtStreamDestroy(stream_aicpu_);
        stream_aicpu_ = nullptr;
        return rc;
    }

    LOG_INFO_V0("DeviceRunner: device=%d set, streams created", device_id);
    return 0;
}

void DeviceRunner::release_run_context() {
    if (stream_aicpu_ != nullptr) {
        rtStreamDestroy(stream_aicpu_);
        stream_aicpu_ = nullptr;
    }
    if (stream_aicore_ != nullptr) {
        rtStreamDestroy(stream_aicore_);
        stream_aicore_ = nullptr;
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

    // Load AICPU SO
    int rc = so_info_.init(aicpu_so_binary_, mem_alloc_);
    if (rc != 0) {
        LOG_ERROR("AicpuSoInfo::init failed: %d", rc);
        return rc;
    }

    // Initialize device args
    device_args_.aicpu_so_bin = so_info_.aicpu_so_bin;
    device_args_.aicpu_so_len = so_info_.aicpu_so_len;
    rc = kernel_args_.init_device_args(device_args_, mem_alloc_);
    if (rc != 0) {
        LOG_ERROR("init_device_args failed: %d", rc);
        so_info_.finalize();
        return rc;
    }

    binaries_loaded_ = true;
    LOG_INFO_V0("DeviceRunner: binaries loaded");
    return 0;
}

void *DeviceRunner::allocate_tensor(size_t bytes) { return mem_alloc_.alloc(bytes); }

void DeviceRunner::free_tensor(void *dev_ptr) {
    if (dev_ptr != nullptr) {
        mem_alloc_.free(dev_ptr);
    }
}

int DeviceRunner::copy_to_device(void *dev_ptr, const void *host_ptr, size_t bytes) {
    return rtMemcpy(dev_ptr, bytes, host_ptr, bytes, RT_MEMCPY_HOST_TO_DEVICE);
}

int DeviceRunner::copy_from_device(void *host_ptr, const void *dev_ptr, size_t bytes) {
    return rtMemcpy(host_ptr, bytes, dev_ptr, bytes, RT_MEMCPY_DEVICE_TO_HOST);
}

int DeviceRunner::validate_block_dim(rtStream_t stream, int block_dim) {
    if (block_dim < 1) {
        LOG_ERROR("block_dim (%d) must be >= 1", block_dim);
        return -1;
    }

    uint32_t cube_limit = 0, vector_limit = 0;
    bool got_limits = (aclrtGetStreamResLimit(stream, ACL_RT_DEV_RES_CUBE_CORE, &cube_limit) == ACL_ERROR_NONE) &&
                      (aclrtGetStreamResLimit(stream, ACL_RT_DEV_RES_VECTOR_CORE, &vector_limit) == ACL_ERROR_NONE) &&
                      cube_limit > 0 && vector_limit > 0;

    if (got_limits) {
        int max_bd = static_cast<int>(
            std::min(cube_limit / PLATFORM_AIC_CORES_PER_BLOCKDIM, vector_limit / PLATFORM_AIV_CORES_PER_BLOCKDIM)
        );
        if (block_dim > max_bd) {
            LOG_ERROR(
                "block_dim (%d) exceeds available cores (max_block_dim=%d, cube=%u, vector=%u)", block_dim, max_bd,
                cube_limit, vector_limit
            );
            return -1;
        }
    } else if (block_dim > PLATFORM_MAX_BLOCKDIM) {
        // aclrtGetStreamResLimit unavailable (or reported no cores) — fall back
        // to the static platform capacity so block_dim stays bounded.
        LOG_ERROR(
            "aclrtGetStreamResLimit unavailable; block_dim (%d) exceeds static cap PLATFORM_MAX_BLOCKDIM (%d)",
            block_dim, PLATFORM_MAX_BLOCKDIM
        );
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

    // Validate block_dim against stream resource limits
    rc = validate_block_dim(stream_aicore_, block_dim);
    if (rc != 0) {
        return rc;
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
    runtime.sche_cpu_num = launch_aicpu_num;

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

    for (int i = 0; i < num_aicore; i++) {
        runtime.workers[i].aicpu_ready = 0;
        runtime.workers[i].aicore_done = 0;
        runtime.workers[i].task = 0;
        // Set core type: first 1/3 are AIC, remaining 2/3 are AIV
        runtime.workers[i].core_type = (i < num_aic) ? CoreType::AIC : CoreType::AIV;
        runtime.workers[i].enable_profiling_flag = enable_profiling_flag;
        runtime.workers[i].l2_perf_records_addr = static_cast<uint64_t>(0);
    }

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
        kernel_args_.finalize_runtime_args();
    });

    // Initialize performance profiling if enabled
    if (enable_l2_swimlane_) {
        rc = init_l2_perf_collection(num_aicore, device_id_);
        if (rc != 0) {
            LOG_ERROR("init_l2_perf_collection failed: %d", rc);
            return rc;
        }
        // Start memory management thread (needs device context)
        l2_perf_collector_.start_memory_manager([this](std::function<void()> fn) {
            return create_thread(std::move(fn));
        });
    }

    // Initialize tensor dump if enabled
    if (enable_dump_tensor_) {
        rc = init_tensor_dump(runtime, num_aicore, device_id_);
        if (rc != 0) {
            LOG_ERROR("init_tensor_dump failed: %d", rc);
            return rc;
        }
        dump_collector_.start_memory_manager();
    }

    if (enable_pmu_) {
        rc = init_pmu_buffers(
            num_aicore, launch_aicpu_num, make_pmu_csv_path(output_prefix_), pmu_event_type_, device_id_
        );
        if (rc != 0) {
            LOG_ERROR("PMU init failed: %d, disabling PMU for this run", rc);
            kernel_args_.args.pmu_data_base = 0;
            enable_pmu_ = false;
        }
    }

    LOG_INFO_V0("=== Initialize runtime args ===");
    rc = prepare_orch_so(runtime);
    if (rc != 0) {
        LOG_ERROR("prepare_orch_so failed: %d", rc);
        return rc;
    }
    // Initialize runtime args
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

    LOG_INFO_V0("=== launch_aicpu_kernel DynTileFwkKernelServerInit ===");
    // Launch AICPU init kernel
    rc = launch_aicpu_kernel(stream_aicpu_, &kernel_args_.args, "DynTileFwkKernelServerInit", 1);
    if (rc != 0) {
        LOG_ERROR("launch_aicpu_kernel (init) failed: %d", rc);
        return rc;
    }

    LOG_INFO_V0("=== launch_aicpu_kernel DynTileFwkKernelServer ===");
    // Launch AICPU main kernel (over-launch for affinity gate)
    rc = launch_aicpu_kernel(
        stream_aicpu_, &kernel_args_.args, "DynTileFwkKernelServer", PLATFORM_MAX_AICPU_THREADS_JUST_FOR_LAUNCH
    );
    if (rc != 0) {
        LOG_ERROR("launch_aicpu_kernel (main) failed: %d", rc);
        return rc;
    }

    LOG_INFO_V0("=== launch_aicore_kernel ===");
    // Launch AICore kernel
    rc = launch_aicore_kernel(stream_aicore_, kernel_args_.args.runtime_args);
    if (rc != 0) {
        LOG_ERROR("launch_aicore_kernel failed: %d", rc);
        return rc;
    }

    // Launch collector threads before synchronization.
    // The enclosing scope ensures guards signal + join on any early return.
    {
        std::thread collector_thread;
        if (enable_l2_swimlane_) {
            collector_thread = create_thread([this]() {
                poll_and_collect_performance_data(0);  // auto-detect task count
            });
        }
        auto perf_thread_guard = RAIIScopeGuard([&]() {
            if (collector_thread.joinable()) {
                collector_thread.join();
            }
        });
        auto perf_signal_guard = RAIIScopeGuard([this]() {
            if (enable_l2_swimlane_) {
                l2_perf_collector_.signal_execution_complete();
            }
        });

        std::thread dump_collector_thread;
        if (enable_dump_tensor_) {
            dump_collector_thread = create_thread([this]() {
                dump_collector_.poll_and_collect(output_prefix_);
            });
        }
        auto dump_thread_guard = RAIIScopeGuard([&]() {
            if (dump_collector_thread.joinable()) {
                dump_collector_thread.join();
            }
        });
        auto dump_signal_guard = RAIIScopeGuard([this]() {
            if (enable_dump_tensor_) {
                dump_collector_.signal_execution_complete();
            }
        });

        std::thread pmu_collector_thread;
        if (enable_pmu_ && pmu_collector_.is_initialized()) {
            pmu_collector_thread = create_thread([this]() {
                pmu_collector_.poll_and_collect();
            });
        }
        auto pmu_thread_guard = RAIIScopeGuard([&]() {
            if (pmu_collector_thread.joinable()) {
                pmu_collector_thread.join();
            }
        });
        auto pmu_signal_guard = RAIIScopeGuard([&]() {
            if (enable_pmu_ && pmu_collector_.is_initialized()) {
                pmu_collector_.signal_execution_complete();
            }
        });

        LOG_INFO_V0("=== rtStreamSynchronize stream_aicpu_ ===");
        rc = rtStreamSynchronize(stream_aicpu_);
        if (rc != 0) {
            LOG_ERROR("rtStreamSynchronize (AICPU) failed: %d", rc);
            return rc;
        }

        LOG_INFO_V0("=== rtStreamSynchronize stream_aicore_ ===");
        rc = rtStreamSynchronize(stream_aicore_);
        if (rc != 0) {
            LOG_ERROR("rtStreamSynchronize (AICore) failed: %d", rc);
            return rc;
        }
    }

    // After streams are synchronized and guards have signal+joined all threads,
    // stop memory managers, drain remaining buffers, and export.
    // All three collectors write under `output_prefix_`, the per-task directory
    // the user must set on CallConfig (CallConfig::validate() enforces non-empty).
    if (enable_l2_swimlane_) {
        l2_perf_collector_.stop_memory_manager();
        l2_perf_collector_.drain_remaining_buffers();
        l2_perf_collector_.scan_remaining_perf_buffers();
        l2_perf_collector_.collect_phase_data();
        l2_perf_collector_.export_swimlane_json(output_prefix_);
    }

    if (enable_dump_tensor_) {
        dump_collector_.stop_memory_manager();
        dump_collector_.drain_remaining_buffers();
        dump_collector_.scan_remaining_dump_buffers();
        dump_collector_.export_dump_files();
    }

    if (enable_pmu_ && pmu_collector_.is_initialized()) {
        pmu_collector_.drain_remaining_buffers();
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
    std::vector<std::pair<int, uint64_t>> kernel_addrs
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
    prepared_callables_.emplace(callable_id, std::move(state));
    return 0;
}

int DeviceRunner::register_prepared_callable_host_orch(
    int32_t callable_id, void *host_dlopen_handle, void *host_orch_func_ptr,
    std::vector<std::pair<int, uint64_t>> kernel_addrs
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
        return {-1, nullptr};
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
            return {-1, nullptr};
        }
        runtime.replay_function_bin_addr(kv.first, kv.second);
    }
    runtime.set_device_orch_func_name(state.func_name.c_str());
    runtime.set_device_orch_config_name(state.config_name.c_str());
    // Stamp callable_id with is_new=false; prepare_orch_so refreshes the flag
    // with the authoritative first_sighting answer right before launch.
    runtime.set_active_callable_id(callable_id, /*is_new=*/false);
    return {0, state.host_orch_func_ptr};
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

    release_run_context();

    // Cleanup kernel args (deviceArgs)
    kernel_args_.finalize_device_args();

    // Cleanup AICPU SO
    so_info_.finalize();

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

    // Cleanup performance profiling (frees L2PerfSetupHeader + all per-core/per-thread buffers)
    if (l2_perf_collector_.is_initialized()) {
        auto free_cb = [](void *dev_ptr) -> int {
            return rtFree(dev_ptr);
        };
        l2_perf_collector_.finalize(nullptr, free_cb);
    }

    // Cleanup tensor dump
    if (dump_collector_.is_initialized()) {
        auto free_cb = [](void *dev_ptr) -> int {
            return rtFree(dev_ptr);
        };
        dump_collector_.finalize(nullptr, free_cb);
    }

    // Cleanup PMU profiling
    if (pmu_collector_.is_initialized()) {
        auto free_cb = [](void *dev_ptr, void * /*user_data*/) -> int {
            return rtFree(dev_ptr);
        };
        pmu_collector_.finalize(nullptr, free_cb, nullptr);
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
    struct Args {
        KernelArgs k_args;
        char kernel_name[32];
        const char so_name[32] = {"libaicpu_extend_kernels.so"};
        const char op_name[32] = {""};
    } args;

    args.k_args = *k_args;
    std::strncpy(args.kernel_name, kernel_name, sizeof(args.kernel_name) - 1);
    args.kernel_name[sizeof(args.kernel_name) - 1] = '\0';

    rtAicpuArgsEx_t rt_args;
    std::memset(&rt_args, 0, sizeof(rt_args));
    rt_args.args = &args;
    rt_args.argsSize = sizeof(args);
    rt_args.kernelNameAddrOffset = offsetof(struct Args, kernel_name);
    rt_args.soNameAddrOffset = offsetof(struct Args, so_name);

    return rtAicpuKernelLaunchExWithArgs(
        rtKernelType_t::KERNEL_TYPE_AICPU_KFC, "AST_DYN_AICPU", aicpu_num, &rt_args, nullptr, stream, 0
    );
}

int DeviceRunner::launch_aicore_kernel(rtStream_t stream, Runtime *runtime) {
    if (aicore_kernel_binary_.empty()) {
        LOG_ERROR("AICore kernel binary is empty");
        return -1;
    }

    size_t bin_size = aicore_kernel_binary_.size();
    const void *bin_data = aicore_kernel_binary_.data();

    rtDevBinary_t binary;
    std::memset(&binary, 0, sizeof(binary));
    binary.magic = RT_DEV_BINARY_MAGIC_ELF;
    binary.version = 0;
    binary.data = bin_data;
    binary.length = bin_size;
    void *bin_handle = nullptr;
    int rc = rtRegisterAllKernel(&binary, &bin_handle);
    if (rc != RT_ERROR_NONE) {
        LOG_ERROR("rtRegisterAllKernel failed: %d", rc);
        return rc;
    }

    struct Args {
        Runtime *runtime;
    };
    // Pass device address of Runtime to AICore
    Args args = {runtime};
    rtArgsEx_t rt_args;
    std::memset(&rt_args, 0, sizeof(rt_args));
    rt_args.args = &args;
    rt_args.argsSize = sizeof(args);

    rtTaskCfgInfo_t cfg = {};
    cfg.schemMode = RT_SCHEM_MODE_BATCH;

    rc = rtKernelLaunchWithHandleV2(bin_handle, 0, block_dim_, &rt_args, nullptr, stream, &cfg);
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

int DeviceRunner::init_l2_perf_collection(int num_aicore, int device_id) {
    // Device memory allocation via rtMalloc directly
    auto alloc_cb = [](size_t size) -> void * {
        void *ptr = nullptr;
        int rc = rtMalloc(&ptr, size, RT_MEMORY_HBM, 0);
        return (rc == 0) ? ptr : nullptr;
    };

    auto free_cb = [](void *dev_ptr) -> int {
        return rtFree(dev_ptr);
    };

    int rc = l2_perf_collector_.initialize(num_aicore, device_id, alloc_cb, nullptr, free_cb);
    if (rc == 0) {
        kernel_args_.args.l2_perf_data_base =
            reinterpret_cast<uint64_t>(l2_perf_collector_.get_l2_perf_setup_device_ptr());
    }
    return rc;
}

void DeviceRunner::poll_and_collect_performance_data(int expected_tasks) {
    l2_perf_collector_.poll_and_collect(expected_tasks);
}

int DeviceRunner::init_tensor_dump(Runtime &runtime, int num_aicore, int device_id) {
    (void)num_aicore;
    int num_dump_threads = runtime.sche_cpu_num;

    auto alloc_cb = [](size_t size) -> void * {
        void *ptr = nullptr;
        int rc = rtMalloc(&ptr, size, RT_MEMORY_HBM, 0);
        return (rc == 0) ? ptr : nullptr;
    };

    auto free_cb = [](void *dev_ptr) -> int {
        return rtFree(dev_ptr);
    };

    auto set_device_cb = [](int dev_id) -> int {
        return rtSetDevice(dev_id);
    };

    int rc = dump_collector_.initialize(num_dump_threads, device_id, alloc_cb, nullptr, free_cb, set_device_cb);
    if (rc != 0) {
        return rc;
    }

    kernel_args_.args.dump_data_base = reinterpret_cast<uint64_t>(dump_collector_.get_dump_shm_device_ptr());
    return 0;
}

int DeviceRunner::init_pmu_buffers(
    int num_cores, int num_threads, const std::string &csv_path, PmuEventType event_type, int device_id
) {
    auto alloc_cb = [](size_t size, void * /*user_data*/) -> void * {
        void *ptr = nullptr;
        int rc = rtMalloc(&ptr, size, RT_MEMORY_HBM, 0);
        return (rc == 0) ? ptr : nullptr;
    };

    auto free_cb = [](void *dev_ptr, void * /*user_data*/) -> int {
        return rtFree(dev_ptr);
    };

    int rc = pmu_collector_.init(
        num_cores, num_threads, &kernel_args_.args.pmu_data_base, csv_path, event_type, alloc_cb, free_cb, nullptr,
        device_id
    );
    return rc;
}
