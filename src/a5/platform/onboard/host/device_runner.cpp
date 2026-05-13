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

int DeviceRunner::run(Runtime &runtime, int block_dim, int launch_aicpu_num) {
    if (launch_aicpu_num < 1 || launch_aicpu_num > PLATFORM_MAX_AICPU_THREADS) {
        LOG_ERROR("launch_aicpu_num (%d) must be in range [1, %d]", launch_aicpu_num, PLATFORM_MAX_AICPU_THREADS);
        return -1;
    }

    // Validate block_dim
    if (block_dim < 1 || block_dim > PLATFORM_MAX_BLOCKDIM) {
        LOG_ERROR("block_dim (%d) must be in range [1, %d]", block_dim, PLATFORM_MAX_BLOCKDIM);
        return -1;
    }

    int scheduler_thread_num = runtime.get_orch_built_on_host() ? launch_aicpu_num : launch_aicpu_num - 1;

    // Validate even core distribution for initial scheduler threads
    if (scheduler_thread_num > 0) {
        if (block_dim % scheduler_thread_num != 0) {
            LOG_ERROR(
                "block_dim (%d) not evenly divisible by scheduler_thread_num (%d)", block_dim, scheduler_thread_num
            );
            return -1;
        }
    } else {
        LOG_INFO_V0(
            "All %d threads are orchestrators, cores will be assigned after orchestration completes", launch_aicpu_num
        );
        // Post-transition: all threads become schedulers
        if (block_dim % launch_aicpu_num != 0) {
            LOG_WARN(
                "block_dim (%d) not evenly divisible by aicpu_thread_num (%d), "
                "some threads will have different core counts after transition",
                block_dim, launch_aicpu_num
            );
        }
    }

    // Ensure device is initialized (lazy initialization)
    int rc = ensure_device_initialized();
    if (rc != 0) {
        LOG_ERROR("ensure_device_initialized failed: %d", rc);
        return rc;
    }

    // Calculate execution parameters
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

    // Set function_bin_addr for all tasks: func_id_to_addr_[] stores CoreCallable
    // device address; compute binary code address using compile-time offset
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
    // Per-callable_id path: when run_prepared bound a known callable_id,
    // the SO bytes were already H2D'd at prepare_callable time.
    // We just stamp dev_orch_so on the runtime, plus mark `is_new` based on
    // whether the AICPU has seen this id since registration.
    const int32_t cid = runtime.get_active_callable_id();
    if (cid >= 0) {
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
        // Pending fields must be empty in the prepared path — runtime_maker's
        // bind_prepared_to_runtime_impl never stages them. Defensive clear:
        runtime.pending_orch_so_data_ = nullptr;
        runtime.pending_orch_so_size_ = 0;
        LOG_INFO_V0(
            "Orch SO prepared cid=%d hash=0x%lx %zu bytes (is_new=%d)", cid, state.hash, state.dev_orch_so_size,
            first_sighting ? 1 : 0
        );
        return 0;
    }

    const void *host_so_data = runtime.pending_orch_so_data_;
    const size_t host_so_size = runtime.pending_orch_so_size_;
    runtime.pending_orch_so_data_ = nullptr;
    runtime.pending_orch_so_size_ = 0;

    if (host_so_data == nullptr || host_so_size == 0) {
        runtime.set_dev_orch_so(0, 0);
        return 0;
    }

    const uint64_t new_hash = simpler::common::utils::elf_build_id_64(host_so_data, host_so_size);

    if (new_hash == cached_orch_so_hash_ && dev_orch_so_buffer_ != nullptr) {
        LOG_INFO_V0("Orch SO cache hit (hash=0x%lx, %zu bytes)", new_hash, host_so_size);
        runtime.set_dev_orch_so(reinterpret_cast<uint64_t>(dev_orch_so_buffer_), host_so_size);
        return 0;
    }

    if (host_so_size > dev_orch_so_capacity_) {
        if (dev_orch_so_buffer_ != nullptr) {
            mem_alloc_.free(dev_orch_so_buffer_);
            dev_orch_so_buffer_ = nullptr;
            dev_orch_so_capacity_ = 0;
        }
        dev_orch_so_buffer_ = mem_alloc_.alloc(host_so_size);
        if (dev_orch_so_buffer_ == nullptr) {
            LOG_ERROR("Failed to allocate %zu bytes for orchestration SO buffer", host_so_size);
            cached_orch_so_hash_ = 0;
            return -1;
        }
        dev_orch_so_capacity_ = host_so_size;
    }

    host_orch_so_copy_.assign(
        static_cast<const uint8_t *>(host_so_data), static_cast<const uint8_t *>(host_so_data) + host_so_size
    );
    int rc = rtMemcpy(
        dev_orch_so_buffer_, dev_orch_so_capacity_, host_orch_so_copy_.data(), host_so_size, RT_MEMCPY_HOST_TO_DEVICE
    );
    if (rc != 0) {
        LOG_ERROR("rtMemcpy for orchestration SO failed: %d", rc);
        cached_orch_so_hash_ = 0;
        return rc;
    }

    cached_orch_so_hash_ = new_hash;
    runtime.set_dev_orch_so(reinterpret_cast<uint64_t>(dev_orch_so_buffer_), host_so_size);
    LOG_INFO_V0("Orch SO cache miss (hash=0x%lx, %zu bytes uploaded)", new_hash, host_so_size);
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
    prepared_callable_path_used_ = true;
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
    prepared_callable_path_used_ = true;
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

int DeviceRunner::bind_prepared_callable_to_runtime(Runtime &runtime, int32_t callable_id) {
    auto it = prepared_callables_.find(callable_id);
    if (it == prepared_callables_.end()) {
        LOG_ERROR("bind_prepared_callable_to_runtime: callable_id=%d not registered", callable_id);
        return -1;
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
            return -1;
        }
        runtime.replay_function_bin_addr(kv.first, kv.second);
    }
    runtime.pending_host_dlopen_handle_ = state.host_dlopen_handle;
    runtime.pending_host_orch_func_ptr_ = state.host_orch_func_ptr;
    runtime.set_device_orch_func_name(state.func_name.c_str());
    runtime.set_device_orch_config_name(state.config_name.c_str());
    // Stamp callable_id with is_new=false; prepare_orch_so refreshes the flag
    // with the authoritative first_sighting answer right before launch.
    runtime.set_active_callable_id(callable_id, /*is_new=*/false);
    return 0;
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

    // Kernel binaries are normally released by validate_runtime_impl on the
    // legacy run() path. The prepared-callable path intentionally leaves
    // them resident across runs (shared by func_id) and relies on
    // finalize() to reclaim them; that is not a leak. Emit at DEBUG so the
    // legacy regression signal is preserved for callers that never went
    // through prepare_callable.
    if (!func_id_to_addr_.empty()) {
        const bool prepared_path_used = prepared_callable_path_used_;
        if (prepared_path_used) {
            LOG_DEBUG("finalize() releasing %zu kernel binaries staged by prepare_callable", func_id_to_addr_.size());
        } else {
            LOG_ERROR("finalize() called with %zu kernel binaries still cached (memory leak)", func_id_to_addr_.size());
        }
        for (const auto &pair : func_id_to_addr_) {
            void *gm_addr = reinterpret_cast<void *>(pair.second);
            mem_alloc_.free(gm_addr);
            LOG_DEBUG("Freed kernel binary: func_id=%d, addr=0x%lx", pair.first, pair.second);
        }
    }
    func_id_to_addr_.clear();
    func_id_to_hash_.clear();
    binaries_loaded_ = false;

    if (dev_orch_so_buffer_ != nullptr) {
        mem_alloc_.free(dev_orch_so_buffer_);
        dev_orch_so_buffer_ = nullptr;
    }
    dev_orch_so_capacity_ = 0;
    cached_orch_so_hash_ = 0;
    host_orch_so_copy_.clear();
    host_orch_so_copy_.shrink_to_fit();

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
    cfg.localMemorySize = PLATFORM_AICORE_LOCAL_MEMORY_SIZE;

    rc = rtKernelLaunchWithHandleV2(bin_handle, 0, block_dim_, &rt_args, nullptr, stream, &cfg);
    if (rc != RT_ERROR_NONE) {
        LOG_ERROR("rtKernelLaunchWithHandleV2 failed: %d", rc);
        return rc;
    }

    return rc;
}

// =============================================================================
// Kernel Binary Upload (returns device address for caller to store in Runtime)
// =============================================================================

uint64_t DeviceRunner::upload_kernel_binary(int func_id, const uint8_t *bin_data, size_t bin_size) {
    if (bin_data == nullptr || bin_size == 0) {
        LOG_ERROR("Invalid kernel binary data");
        return 0;
    }

    // Run context (streams) must be prepared first.
    if (stream_aicpu_ == nullptr) {
        LOG_ERROR("Run context not prepared before upload_kernel_binary()");
        return 0;
    }

    // Return cached callable address if already uploaded *and* the new bytes
    // match. With the prepared-callable path, multiple ChipCallables share a
    // single ChipWorker (and DeviceRunner) and can pick distinct kernel
    // binaries for the same func_id. Naively reusing the cached entry hands
    // the AICore the previous callable's kernel: dispatch never completes
    // the new task and the AICPU spins forever.
    const uint64_t new_hash = simpler::common::utils::elf_build_id_64(bin_data, bin_size);
    auto it = func_id_to_addr_.find(func_id);
    if (it != func_id_to_addr_.end()) {
        auto hash_it = func_id_to_hash_.find(func_id);
        if (hash_it != func_id_to_hash_.end() && hash_it->second == new_hash) {
            LOG_INFO_V0("Kernel func_id=%d already uploaded (matching hash), returning cached address", func_id);
            return it->second;
        }
        LOG_INFO_V0("Kernel func_id=%d binary changed, evicting cached entry", func_id);
        mem_alloc_.free(reinterpret_cast<void *>(it->second));
        func_id_to_addr_.erase(it);
        func_id_to_hash_.erase(func_id);
    }

    LOG_DEBUG("Uploading kernel binary: func_id=%d, size=%zu bytes", func_id, bin_size);

    // Allocate device GM memory for kernel binary
    void *gm_addr = mem_alloc_.alloc(bin_size);
    if (gm_addr == nullptr) {
        LOG_ERROR("Failed to allocate device GM memory for kernel func_id=%d", func_id);
        return 0;
    }

    // Set resolved_addr_ in host buffer before copying to device:
    // AICPU will read this field to get the binary code address for dispatch
    uint64_t callable_addr = reinterpret_cast<uint64_t>(gm_addr);
    assert((callable_addr & (CALLABLE_ALIGN - 1)) == 0 && "device alloc must be CALLABLE_ALIGN-byte aligned");
    uint64_t binary_code_addr = callable_addr + CoreCallable::binary_data_offset();
    // Write resolved_addr_ into the host-side buffer (the field lives at a fixed offset)
    CoreCallable *host_callable = reinterpret_cast<CoreCallable *>(const_cast<uint8_t *>(bin_data));
    host_callable->set_resolved_addr(binary_code_addr);

    // Copy the full CoreCallable (header + binary) to device
    int rc = rtMemcpy(gm_addr, bin_size, bin_data, bin_size, RT_MEMCPY_HOST_TO_DEVICE);
    if (rc != 0) {
        LOG_ERROR("rtMemcpy to device failed: %d", rc);
        mem_alloc_.free(gm_addr);
        return 0;
    }

    func_id_to_addr_[func_id] = callable_addr;
    func_id_to_hash_[func_id] = new_hash;

    LOG_DEBUG("  func_id=%d -> callable_addr=0x%lx, binary_code_addr=0x%lx", func_id, callable_addr, binary_code_addr);

    return callable_addr;
}

void DeviceRunner::remove_kernel_binary(int func_id) {
    auto it = func_id_to_addr_.find(func_id);
    if (it == func_id_to_addr_.end()) {
        return;
    }

    uint64_t function_bin_addr = it->second;
    void *gm_addr = reinterpret_cast<void *>(function_bin_addr);

    mem_alloc_.free(gm_addr);
    func_id_to_addr_.erase(it);
    func_id_to_hash_.erase(func_id);

    LOG_DEBUG("Removed kernel binary: func_id=%d, addr=0x%lx", func_id, function_bin_addr);
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
