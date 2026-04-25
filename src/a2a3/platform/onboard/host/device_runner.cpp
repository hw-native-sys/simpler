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

#include <dlfcn.h>

#include <cassert>
#include <iostream>
#include <string>
#include <vector>
#include "acl/acl.h"

// Include HAL constants from CANN (header only, library loaded dynamically)
#include "ascend_hal.h"
#include "callable.h"
#include "utils/elf_build_id.h"
#include "host/host_regs.h"  // Register address retrieval
#include "host/raii_scope_guard.h"

// =============================================================================
// Lazy-loaded HAL (ascend_hal) for profiling host-register only
// =============================================================================

namespace {
void *g_hal_handle = nullptr;

using HalHostRegisterFn = int (*)(void *dev_ptr, size_t size, unsigned int flags, int device_id, void **host_ptr);
using HalHostUnregisterFn = int (*)(void *host_ptr, int device_id);

int load_hal_if_needed() {
    if (g_hal_handle != nullptr) {
        return 0;
    }
    g_hal_handle = dlopen("libascend_hal.so", RTLD_NOW | RTLD_LOCAL);
    if (g_hal_handle == nullptr) {
        return -1;
    }
    return 0;
}

HalHostRegisterFn get_halHostRegister() {
    if (g_hal_handle == nullptr) {
        return nullptr;
    }
    return reinterpret_cast<HalHostRegisterFn>(dlsym(g_hal_handle, "halHostRegister"));
}

HalHostUnregisterFn get_halHostUnregister() {
    if (g_hal_handle == nullptr) {
        return nullptr;
    }
    return reinterpret_cast<HalHostUnregisterFn>(dlsym(g_hal_handle, "halHostUnregister"));
}

}  // namespace

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

int KernelArgsHelper::init_ffts_base_addr() {
    uint64_t ffts_base_addr{0};
    uint32_t ffts_len{0};
    int rc = rtGetC2cCtrlAddr(&ffts_base_addr, &ffts_len);
    if (rc != 0) {
        LOG_ERROR("rtGetC2cCtrlAddr failed: %d", rc);
        return rc;
    }
    args.ffts_base_addr = ffts_base_addr;
    return 0;
}

int KernelArgsHelper::init_device_kernel_args(MemoryAllocator &allocator) {
    allocator_ = &allocator;

    if (device_k_args_ == nullptr) {
        void *dev_ptr = allocator_->alloc(sizeof(KernelArgs));
        if (dev_ptr == nullptr) {
            LOG_ERROR("Alloc for device KernelArgs failed");
            return -1;
        }
        device_k_args_ = reinterpret_cast<KernelArgs *>(dev_ptr);
    }
    int rc = rtMemcpy(device_k_args_, sizeof(KernelArgs), &args, sizeof(KernelArgs), RT_MEMCPY_HOST_TO_DEVICE);
    if (rc != 0) {
        LOG_ERROR("rtMemcpy for KernelArgs failed: %d", rc);
        allocator_->free(device_k_args_);
        device_k_args_ = nullptr;
        return rc;
    }
    return 0;
}

int KernelArgsHelper::finalize_device_kernel_args() {
    if (device_k_args_ != nullptr && allocator_ != nullptr) {
        int rc = allocator_->free(device_k_args_);
        device_k_args_ = nullptr;
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

int DeviceRunner::ensure_device_initialized(
    int device_id, const std::vector<uint8_t> &aicpu_so_binary, const std::vector<uint8_t> &aicore_kernel_binary
) {
    // First attach the current thread and create fresh run-scoped streams
    int rc = prepare_run_context(device_id);
    if (rc != 0) {
        return rc;
    }

    // Then ensure binaries are loaded
    return ensure_binaries_loaded(aicpu_so_binary, aicore_kernel_binary);
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

int DeviceRunner::ensure_acl_ready(int device_id) {
    if (device_id < 0) {
        LOG_ERROR("ensure_acl_ready: invalid device_id %d", device_id);
        return -1;
    }

    // aclInit is process-wide; CANN returns 100002 if it has already been
    // initialized (possibly by another owner), which we treat as success.
    constexpr int kAclRepeatInit = 100002;
    aclError aRet = aclInit(nullptr);
    if (aRet != ACL_SUCCESS && static_cast<int>(aRet) != kAclRepeatInit) {
        LOG_ERROR("aclInit failed: %d", static_cast<int>(aRet));
        return static_cast<int>(aRet);
    }

    // ACL device binding is per-thread; every caller must still hit it.
    aRet = aclrtSetDevice(device_id);
    if (aRet != ACL_SUCCESS) {
        LOG_ERROR("aclrtSetDevice(%d) failed: %d", device_id, static_cast<int>(aRet));
        return static_cast<int>(aRet);
    }

    // Record that we are responsible for aclFinalize at teardown.
    acl_ready_ = true;
    if (device_id_ < 0) device_id_ = device_id;
    return 0;
}

void *DeviceRunner::create_comm_stream() {
    aclrtStream stream = nullptr;
    aclError aRet = aclrtCreateStream(&stream);
    if (aRet != ACL_SUCCESS) {
        LOG_ERROR("aclrtCreateStream failed: %d", static_cast<int>(aRet));
        return nullptr;
    }
    return stream;
}

int DeviceRunner::destroy_comm_stream(void *stream) {
    if (stream == nullptr) return 0;

    // Best-effort teardown.  HcclBarrier submits async work on the stream;
    // if the caller never blocked for completion (or hit the HCCL 507018
    // barrier regression), aclrtDestroyStream will refuse with 507901
    // ("stream still has pending tasks").  We try to drain first, then
    // destroy anyway, and log failures without propagating them — leaking
    // a stream at teardown is strictly better than failing the teardown
    // itself, which would block device finalization.  This matches the
    // cleanup behavior of the HCCL C++ hardware UT.
    aclError sync_rc = aclrtSynchronizeStream(static_cast<aclrtStream>(stream));
    if (sync_rc != ACL_SUCCESS) {
        LOG_ERROR("aclrtSynchronizeStream during stream teardown failed: %d", static_cast<int>(sync_rc));
    }
    aclError destroy_rc = aclrtDestroyStream(static_cast<aclrtStream>(stream));
    if (destroy_rc != ACL_SUCCESS) {
        LOG_ERROR("aclrtDestroyStream failed (leaking stream): %d", static_cast<int>(destroy_rc));
    }
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

    LOG_INFO("DeviceRunner: device=%d set, streams created", device_id);
    return 0;
}

void DeviceRunner::release_run_context() {
    // Destroy streams (they belong to the current thread's CANN context)
    if (stream_aicpu_ != nullptr) {
        rtStreamDestroy(stream_aicpu_);
        stream_aicpu_ = nullptr;
    }
    if (stream_aicore_ != nullptr) {
        rtStreamDestroy(stream_aicore_);
        stream_aicore_ = nullptr;
    }
}

int DeviceRunner::ensure_binaries_loaded(
    const std::vector<uint8_t> &aicpu_so_binary, const std::vector<uint8_t> &aicore_kernel_binary
) {
    // Check if already loaded
    if (binaries_loaded_) {
        // Just update kernel binary if different
        if (aicore_kernel_binary_ != aicore_kernel_binary) {
            aicore_kernel_binary_ = aicore_kernel_binary;
        }
        return 0;
    }

    // Device must be set first
    if (stream_aicpu_ == nullptr) {
        LOG_ERROR("Device not set before loading binaries");
        return -1;
    }

    aicore_kernel_binary_ = aicore_kernel_binary;

    // Load AICPU SO
    int rc = so_info_.init(aicpu_so_binary, mem_alloc_);
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
    LOG_INFO("DeviceRunner: binaries loaded");
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

int DeviceRunner::run(
    Runtime &runtime, int block_dim, int device_id, const std::vector<uint8_t> &aicpu_so_binary,
    const std::vector<uint8_t> &aicore_kernel_binary, int launch_aicpu_num
) {
    // Validate launch_aicpu_num
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
        LOG_INFO(
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
    int rc = ensure_device_initialized(device_id, aicpu_so_binary, aicore_kernel_binary);
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
    rc = init_aicore_register_addresses(
        &kernel_args_.args.regs, static_cast<uint64_t>(device_id), mem_alloc_, AicoreRegKind::Ctrl
    );
    if (rc != 0) {
        LOG_ERROR("init_aicore_register_addresses(Ctrl) failed: %d", rc);
        return rc;
    }

    // Get AICore PMU register addresses (distinct MMIO page from AIC_CTRL).
    // Failure is non-fatal: PMU will be disabled if this query fails.
    if (enable_pmu_) {
        int pmu_rc = init_aicore_register_addresses(
            &kernel_args_.args.pmu_reg_addrs, static_cast<uint64_t>(device_id), mem_alloc_, AicoreRegKind::Pmu
        );
        if (pmu_rc != 0) {
            LOG_ERROR("init_aicore_register_addresses(Pmu) failed: %d, disabling PMU", pmu_rc);
            kernel_args_.args.pmu_reg_addrs = 0;
            enable_pmu_ = false;
        }
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

    auto pmu_regs_cleanup = RAIIScopeGuard([this]() {
        if (kernel_args_.args.pmu_reg_addrs != 0) {
            mem_alloc_.free(reinterpret_cast<void *>(kernel_args_.args.pmu_reg_addrs));
            kernel_args_.args.pmu_reg_addrs = 0;
        }
    });

    auto runtime_args_cleanup = RAIIScopeGuard([this]() {
        kernel_args_.finalize_device_kernel_args();
        kernel_args_.finalize_runtime_args();
    });

    // Initialize performance profiling if enabled
    if (enable_l2_swimlane_) {
        rc = init_l2_perf_collection(num_aicore, device_id);
        if (rc != 0) {
            LOG_ERROR("init_l2_perf_collection failed: %d", rc);
            return rc;
        }
        // Start memory management thread
        l2_perf_collector_.start_memory_manager([this](std::function<void()> fn) {
            return create_thread(std::move(fn));
        });
    }

    if (enable_dump_tensor_) {
        // Initialize tensor dump (independent from profiling)
        rc = init_tensor_dump(runtime, num_aicore, device_id);
        if (rc != 0) {
            LOG_ERROR("init_tensor_dump failed: %d", rc);
            return rc;
        }
        dump_collector_.start_memory_manager();
    }

    if (enable_pmu_) {
        rc = init_pmu_buffers(num_aicore, launch_aicpu_num, make_pmu_csv_path(), pmu_event_type_, device_id);
        if (rc != 0) {
            LOG_ERROR("PMU init failed: %d, disabling PMU for this run", rc);
            kernel_args_.args.pmu_data_base = 0;
            enable_pmu_ = false;
        }
    }

    auto perf_cleanup = RAIIScopeGuard([this]() {
        bool was_initialized = l2_perf_collector_.is_initialized();
        if (was_initialized) {
            l2_perf_collector_.stop_memory_manager();
        }
    });

    LOG_INFO("=== Initialize runtime args ===");
    // Resolve the orchestration SO into a device-resident buffer and refresh
    // runtime metadata before the Runtime struct is uploaded to device.
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

    rc = kernel_args_.init_ffts_base_addr();
    if (rc != 0) {
        LOG_ERROR("init_ffts_base_addr failed: %d", rc);
        return rc;
    }

    // Copy KernelArgs to device memory for AICore
    rc = kernel_args_.init_device_kernel_args(mem_alloc_);
    if (rc != 0) {
        LOG_ERROR("init_device_kernel_args failed: %d", rc);
        return rc;
    }

    LOG_INFO("=== launch_aicpu_kernel DynTileFwkKernelServerInit ===");
    // Launch AICPU init kernel
    rc = launch_aicpu_kernel(stream_aicpu_, &kernel_args_.args, "DynTileFwkKernelServerInit", 1);
    if (rc != 0) {
        LOG_ERROR("launch_aicpu_kernel (init) failed: %d", rc);
        return rc;
    }

    LOG_INFO("=== launch_aicpu_kernel DynTileFwkKernelServer ===");
    // Launch AICPU main kernel (over-launch for affinity gate)
    rc = launch_aicpu_kernel(
        stream_aicpu_, &kernel_args_.args, "DynTileFwkKernelServer", PLATFORM_MAX_AICPU_THREADS_JUST_FOR_LAUNCH
    );
    if (rc != 0) {
        LOG_ERROR("launch_aicpu_kernel (main) failed: %d", rc);
        return rc;
    }

    LOG_INFO("=== launch_aicore_kernel ===");
    // Launch AICore kernel (pass device copy of KernelArgs)
    rc = launch_aicore_kernel(stream_aicore_, kernel_args_.device_k_args_);
    if (rc != 0) {
        LOG_ERROR("launch_aicore_kernel failed: %d", rc);
        return rc;
    }

    {
        std::thread collector_thread;
        if (enable_l2_swimlane_) {
            collector_thread = create_thread([this, &runtime]() {
                l2_perf_collector_.poll_and_collect(runtime.get_task_count());
            });
        }
        auto thread_guard = RAIIScopeGuard([&]() {
            if (collector_thread.joinable()) {
                collector_thread.join();
            }
        });
        auto collector_signal_guard = RAIIScopeGuard([this, &runtime]() {
            if (enable_l2_swimlane_) {
                l2_perf_collector_.signal_execution_complete();
            }
        });

        std::thread dump_collector_thread;
        if (enable_dump_tensor_) {
            dump_collector_thread = std::thread([this]() {
                dump_collector_.poll_and_collect();
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
        if (enable_pmu_) {
            pmu_collector_thread = std::thread([this]() {
                pmu_collector_.poll_and_collect();
            });
        }
        auto pmu_thread_guard = RAIIScopeGuard([&]() {
            if (pmu_collector_thread.joinable()) {
                pmu_collector_thread.join();
            }
        });
        auto pmu_signal_guard = RAIIScopeGuard([this]() {
            if (enable_pmu_) {
                pmu_collector_.signal_execution_complete();
            }
        });

        LOG_INFO("=== rtStreamSynchronize stream_aicpu_ ===");
        // Synchronize streams
        rc = rtStreamSynchronize(stream_aicpu_);
        if (rc != 0) {
            LOG_ERROR("rtStreamSynchronize (AICPU) failed: %d", rc);
            return rc;
        }

        LOG_INFO("=== rtStreamSynchronize stream_aicore_ ===");
        rc = rtStreamSynchronize(stream_aicore_);
        if (rc != 0) {
            LOG_ERROR("rtStreamSynchronize (AICore) failed: %d", rc);
            return rc;
        }
    }

    // Stop memory management, drain remaining buffers, collect phase data, export
    if (enable_l2_swimlane_) {
        l2_perf_collector_.stop_memory_manager();
        l2_perf_collector_.drain_remaining_buffers();
        l2_perf_collector_.scan_remaining_perf_buffers();
        l2_perf_collector_.collect_phase_data();
        l2_perf_collector_.export_swimlane_json();
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
    const void *host_so_data = runtime.pending_orch_so_data_;
    const size_t host_so_size = runtime.pending_orch_so_size_;
    runtime.pending_orch_so_data_ = nullptr;
    runtime.pending_orch_so_size_ = 0;

    if (host_so_data == nullptr || host_so_size == 0) {
        // Host-orchestration mode (no device SO needed).
        runtime.set_dev_orch_so(0, 0, false);
        return 0;
    }

    const uint64_t new_hash = simpler::common::utils::elf_build_id_64(host_so_data, host_so_size);

    if (new_hash == cached_orch_so_hash_ && dev_orch_so_buffer_ != nullptr) {
        LOG_INFO("Orch SO cache hit (hash=0x%lx, %zu bytes)", new_hash, host_so_size);
        runtime.set_dev_orch_so(reinterpret_cast<uint64_t>(dev_orch_so_buffer_), host_so_size, /*is_new=*/false);
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

    // Persist a host-side copy so the rtMemcpy source is independent from
    // any Python ctypes buffer the caller may release as soon as run()
    // returns. This is also what runtime_maker hands us by reference.
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
    runtime.set_dev_orch_so(reinterpret_cast<uint64_t>(dev_orch_so_buffer_), host_so_size, /*is_new=*/true);
    LOG_INFO("Orch SO cache miss (hash=0x%lx, %zu bytes uploaded)", new_hash, host_so_size);
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

    // Kernel binaries should have been removed by validate_runtime_impl()
    if (!func_id_to_addr_.empty()) {
        LOG_ERROR("finalize() called with %zu kernel binaries still cached (memory leak)", func_id_to_addr_.size());
        // Cleanup leaked binaries to prevent memory leaks
        for (const auto &pair : func_id_to_addr_) {
            void *gm_addr = reinterpret_cast<void *>(pair.second);
            mem_alloc_.free(gm_addr);
            LOG_DEBUG("Freed leaked kernel binary: func_id=%d, addr=0x%lx", pair.first, pair.second);
        }
    }
    func_id_to_addr_.clear();
    binaries_loaded_ = false;

    // Release the cached orchestration SO buffer.
    if (dev_orch_so_buffer_ != nullptr) {
        mem_alloc_.free(dev_orch_so_buffer_);
        dev_orch_so_buffer_ = nullptr;
    }
    dev_orch_so_capacity_ = 0;
    cached_orch_so_hash_ = 0;
    host_orch_so_copy_.clear();
    host_orch_so_copy_.shrink_to_fit();

    // Cleanup performance profiling
    if (l2_perf_collector_.is_initialized()) {
        auto unregister_cb = [](void *dev_ptr, int device_id) -> int {
            HalHostUnregisterFn fn = get_halHostUnregister();
            if (fn != nullptr) {
                return fn(dev_ptr, device_id);
            }
            return 0;
        };

        auto free_cb = [](void *dev_ptr) -> int {
            return rtFree(dev_ptr);
        };

        l2_perf_collector_.finalize(unregister_cb, free_cb);
    }

    if (dump_collector_.is_initialized()) {
        auto unregister_cb = [](void *dev_ptr, int device_id, void *user_data) -> int {
            (void)user_data;
            HalHostUnregisterFn fn = get_halHostUnregister();
            if (fn != nullptr) {
                return fn(dev_ptr, device_id);
            }
            return 0;
        };

        auto free_cb = [](void *dev_ptr, void *user_data) -> int {
            auto *allocator = static_cast<MemoryAllocator *>(user_data);
            return allocator->free(dev_ptr);
        };

        dump_collector_.finalize(unregister_cb, free_cb, &mem_alloc_);
    }

    if (pmu_collector_.is_initialized()) {
        auto unregister_cb = [](void *dev_ptr, int device_id, void *user_data) -> int {
            (void)user_data;
            HalHostUnregisterFn fn = get_halHostUnregister();
            if (fn != nullptr) {
                return fn(dev_ptr, device_id);
            }
            return 0;
        };

        auto free_cb = [](void *dev_ptr, void *user_data) -> int {
            auto *allocator = static_cast<MemoryAllocator *>(user_data);
            return allocator->free(dev_ptr);
        };

        pmu_collector_.finalize(unregister_cb, free_cb, &mem_alloc_);
    }

    // Free all remaining allocations (including handshake buffer and binGmAddr)
    mem_alloc_.finalize();

    // Reset device and finalize ACL AFTER all device memory is freed.
    // Gated on acl_ready_ so rt-only runtimes that never called
    // ensure_acl_ready() do not try to aclFinalize an un-init'd ACL state.
    if (acl_ready_ && device_id_ >= 0) {
        int reset_rc = aclrtResetDevice(device_id_);
        if (reset_rc != 0) {
            LOG_ERROR("aclrtResetDevice(%d) failed during finalize: %d", device_id_, reset_rc);
            rc = reset_rc;
        }
        int finalize_rc = aclFinalize();
        if (finalize_rc != 0) {
            LOG_ERROR("aclFinalize failed during finalize: %d", finalize_rc);
            if (rc == 0) rc = finalize_rc;
        }
        acl_ready_ = false;
    }

    device_id_ = -1;
    block_dim_ = 0;
    worker_count_ = 0;
    aicore_kernel_binary_.clear();

    LOG_INFO("DeviceRunner finalized");
    return rc;
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

int DeviceRunner::launch_aicore_kernel(rtStream_t stream, KernelArgs *k_args) {
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
        KernelArgs *k_args;
    };
    Args args = {k_args};
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
// Kernel Binary Upload (returns device address for caller to store in Runtime)
// =============================================================================

uint64_t DeviceRunner::upload_kernel_binary(int func_id, const uint8_t *bin_data, size_t bin_size) {
    if (bin_data == nullptr || bin_size == 0) {
        LOG_ERROR("Invalid kernel binary data");
        return 0;
    }

    // Device must be set first (set_device() must be called before upload_kernel_binary())
    if (stream_aicpu_ == nullptr) {
        LOG_ERROR("Run context not prepared before upload_kernel_binary()");
        return 0;
    }

    // Return cached callable address if already uploaded
    auto it = func_id_to_addr_.find(func_id);
    if (it != func_id_to_addr_.end()) {
        LOG_INFO("Kernel func_id=%d already uploaded, returning cached address", func_id);
        return it->second;
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

    LOG_DEBUG("Removed kernel binary: func_id=%d, addr=0x%lx", func_id, function_bin_addr);
}

int DeviceRunner::init_l2_perf_collection(int num_aicore, int device_id) {
    // Define allocation callback (a2a3: use rtMalloc directly)
    auto alloc_cb = [](size_t size) -> void * {
        void *ptr = nullptr;
        int rc = rtMalloc(&ptr, size, RT_MEMORY_HBM, 0);
        return (rc == 0) ? ptr : nullptr;
    };

    // Define registration callback (a2a3: use halHostRegister for shared memory)
    auto register_cb = [](void *dev_ptr, size_t size, int device_id, void **host_ptr) -> int {
        if (load_hal_if_needed() != 0) {
            LOG_ERROR("Failed to load ascend_hal for profiling: %s", dlerror());
            return -1;
        }
        HalHostRegisterFn fn = get_halHostRegister();
        if (fn == nullptr) {
            LOG_ERROR("halHostRegister symbol not found: %s", dlerror());
            return -1;
        }
        return fn(dev_ptr, size, DEV_SVM_MAP_HOST, device_id, host_ptr);
    };

    auto free_cb = [](void *dev_ptr) -> int {
        return rtFree(dev_ptr);
    };

    int rc = l2_perf_collector_.initialize(num_aicore, device_id, alloc_cb, register_cb, free_cb);
    if (rc == 0) {
        kernel_args_.args.l2_perf_data_base =
            reinterpret_cast<uint64_t>(l2_perf_collector_.get_l2_perf_setup_device_ptr());
    }
    return rc;
}

int DeviceRunner::init_tensor_dump(Runtime &runtime, int num_aicore, int device_id) {
    int num_dump_threads = runtime.sche_cpu_num;

    auto alloc_cb = [](size_t size, void *user_data) -> void * {
        auto *allocator = static_cast<MemoryAllocator *>(user_data);
        return allocator->alloc(size);
    };

    auto register_cb = [](void *dev_ptr, size_t size, int device_id, void *user_data, void **host_ptr) -> int {
        (void)user_data;
        if (load_hal_if_needed() != 0) {
            LOG_ERROR("Failed to load ascend_hal for tensor dump: %s", dlerror());
            return -1;
        }
        HalHostRegisterFn fn = get_halHostRegister();
        if (fn == nullptr) {
            LOG_ERROR("halHostRegister symbol not found: %s", dlerror());
            return -1;
        }
        return fn(dev_ptr, size, DEV_SVM_MAP_HOST, device_id, host_ptr);
    };

    auto free_cb = [](void *dev_ptr, void *user_data) -> int {
        auto *allocator = static_cast<MemoryAllocator *>(user_data);
        return allocator->free(dev_ptr);
    };

    auto set_device_cb = [](int device_id, void * /*user_data*/) -> int {
        return rtSetDevice(device_id);
    };

    int rc = dump_collector_.initialize(
        num_dump_threads, device_id, alloc_cb, register_cb, free_cb, &mem_alloc_, set_device_cb
    );
    if (rc != 0) {
        return rc;
    }

    kernel_args_.args.dump_data_base = reinterpret_cast<uint64_t>(dump_collector_.get_dump_shm_device_ptr());
    return 0;
}

int DeviceRunner::init_pmu_buffers(
    int num_cores, int num_threads, const std::string &csv_path, PmuEventType event_type, int device_id
) {
    auto alloc_cb = [](size_t size, void *user_data) -> void * {
        auto *allocator = static_cast<MemoryAllocator *>(user_data);
        return allocator->alloc(size);
    };

    auto register_cb = [](void *dev_ptr, size_t size, int device_id, void *user_data, void **host_ptr) -> int {
        (void)user_data;
        if (load_hal_if_needed() != 0) {
            LOG_ERROR("Failed to load ascend_hal for PMU: %s", dlerror());
            return -1;
        }
        HalHostRegisterFn fn = get_halHostRegister();
        if (fn == nullptr) {
            LOG_ERROR("halHostRegister symbol not found: %s", dlerror());
            return -1;
        }
        return fn(dev_ptr, size, DEV_SVM_MAP_HOST, device_id, host_ptr);
    };

    auto free_cb = [](void *dev_ptr, void *user_data) -> int {
        auto *allocator = static_cast<MemoryAllocator *>(user_data);
        return allocator->free(dev_ptr);
    };

    int rc = pmu_collector_.init(
        num_cores, num_threads, &kernel_args_.args.pmu_data_base, csv_path, event_type, alloc_cb, register_cb, free_cb,
        &mem_alloc_, device_id
    );
    return rc;
}
