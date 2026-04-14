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
 * Device Runner Implementation - Thread-Based Simulation
 *
 * This file implements the simulated device execution using host threads.
 * It provides the same API as the real a2a3 implementation but uses
 * std::thread instead of CANN runtime APIs.
 *
 * aicpu_execute and aicore_execute_wrapper are loaded dynamically via dlopen from
 * the binaries passed to launch_runtime.
 *
 * Cross-platform notes:
 * - Linux: Uses MAP_ANONYMOUS for anonymous memory mapping
 * - macOS: Uses MAP_ANON (aliased) and MAP_JIT for executable memory on Apple Silicon
 *   which requires W^X (write xor execute) protection toggling via pthread_jit_write_protect_np
 */

#include "device_runner.h"

#include <stdlib.h>
#include <sys/stat.h>

#include <atomic>
#include <cstdio>
#include <string>
#include <vector>

#include "aicpu/platform_aicpu_affinity.h"
#include "callable.h"
#include "cpu_sim_context.h"
#include "host/raii_scope_guard.h"

// Function pointer types for dynamically loaded executors
typedef int (*aicpu_execute_func_t)(Runtime *runtime);
typedef void (*aicore_execute_func_t)(
    Runtime *runtime, int block_idx, CoreType core_type, uint32_t physical_core_id, uint64_t regs
);
typedef void (*set_platform_regs_func_t)(uint64_t regs);
typedef void (*set_platform_dump_base_func_t)(uint64_t dump_data_base);
typedef void (*set_enable_dump_tensor_func_t)(bool enable);

namespace {

bool write_all_bytes(int fd, const uint8_t *data, size_t size) {
    size_t total_written = 0;
    while (total_written < size) {
        ssize_t written = write(fd, data + total_written, size - total_written);
        if (written <= 0) {
            return false;
        }
        total_written += static_cast<size_t>(written);
    }
    return true;
}

bool create_temp_so_file(const std::string &path_template, const uint8_t *data, size_t size, std::string *out_path) {
    std::vector<char> path_buf(path_template.begin(), path_template.end());
    path_buf.push_back('\0');

    int fd = mkstemp(path_buf.data());
    if (fd < 0) {
        return false;
    }

    // dlopen requires the file to be executable; mkstemp creates 0600 (no exec bit)
    if (fchmod(fd, 0755) != 0) {
        close(fd);
        unlink(path_buf.data());
        return false;
    }

    bool ok = write_all_bytes(fd, data, size);
    if (close(fd) != 0) {
        ok = false;
    }
    if (!ok) {
        unlink(path_buf.data());
        return false;
    }

    *out_path = path_buf.data();
    return true;
}

}  // namespace

// =============================================================================
// DeviceRunner Implementation
// =============================================================================

DeviceRunner::~DeviceRunner() { finalize(); }

std::thread DeviceRunner::create_thread(std::function<void()> fn) {
    int dev_id = device_id_;
    return std::thread([dev_id, fn = std::move(fn)]() {
        pto_cpu_sim_bind_device(dev_id);
        fn();
        pto_cpu_sim_bind_device(-1);
    });
}

int DeviceRunner::ensure_device_initialized(
    int device_id, const std::vector<uint8_t> &aicpu_so_binary, const std::vector<uint8_t> &aicore_kernel_binary
) {
    device_id_ = device_id;
    return ensure_binaries_loaded(aicpu_so_binary, aicore_kernel_binary);
}

int DeviceRunner::ensure_binaries_loaded(
    const std::vector<uint8_t> &aicpu_so_binary, const std::vector<uint8_t> &aicore_kernel_binary
) {
    // Close any previously loaded binaries before reloading
    unload_executor_binaries();

    // Write AICPU binary to temp file and dlopen
    if (!aicpu_so_binary.empty()) {
        if (!create_temp_so_file(
                "/tmp/aicpu_sim_XXXXXX", aicpu_so_binary.data(), aicpu_so_binary.size(), &aicpu_so_path_
            )) {
            LOG_ERROR("Failed to create temp file for AICPU SO");
            return -1;
        }

        aicpu_so_handle_ = dlopen(aicpu_so_path_.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (aicpu_so_handle_ == nullptr) {
            LOG_ERROR("dlopen failed for AICPU SO: %s", dlerror());
            return -1;
        }

        aicpu_execute_func_ = reinterpret_cast<int (*)(Runtime *)>(dlsym(aicpu_so_handle_, "aicpu_execute"));
        if (aicpu_execute_func_ == nullptr) {
            LOG_ERROR("dlsym failed for aicpu_execute: %s", dlerror());
            return -1;
        }

        set_platform_regs_func_ = reinterpret_cast<void (*)(uint64_t)>(dlsym(aicpu_so_handle_, "set_platform_regs"));
        if (set_platform_regs_func_ == nullptr) {
            LOG_ERROR("dlsym failed for set_platform_regs: %s", dlerror());
            return -1;
        }
        set_platform_dump_base_func_ =
            reinterpret_cast<void (*)(uint64_t)>(dlsym(aicpu_so_handle_, "set_platform_dump_base"));
        if (set_platform_dump_base_func_ == nullptr) {
            LOG_ERROR("dlsym failed for set_platform_dump_base: %s", dlerror());
            return -1;
        }
        set_enable_dump_tensor_func_ =
            reinterpret_cast<void (*)(bool)>(dlsym(aicpu_so_handle_, "set_enable_dump_tensor"));
        if (set_enable_dump_tensor_func_ == nullptr) {
            LOG_ERROR("dlsym failed for set_enable_dump_tensor: %s", dlerror());
            return -1;
        }

        LOG_INFO("DeviceRunner(sim): Loaded aicpu_execute from %s", aicpu_so_path_.c_str());
    }

    // Write AICore binary to temp file and dlopen
    if (!aicore_kernel_binary.empty()) {
        if (!create_temp_so_file(
                "/tmp/aicore_sim_XXXXXX", aicore_kernel_binary.data(), aicore_kernel_binary.size(), &aicore_so_path_
            )) {
            LOG_ERROR("Failed to create temp file for AICore SO");
            return -1;
        }

        aicore_so_handle_ = dlopen(aicore_so_path_.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (aicore_so_handle_ == nullptr) {
            LOG_ERROR("dlopen failed for AICore SO: %s", dlerror());
            return -1;
        }

        aicore_execute_func_ = reinterpret_cast<void (*)(Runtime *, int, CoreType, uint32_t, uint64_t)>(
            dlsym(aicore_so_handle_, "aicore_execute_wrapper")
        );
        if (aicore_execute_func_ == nullptr) {
            LOG_ERROR("dlsym failed for aicore_execute_wrapper: %s", dlerror());
            return -1;
        }
        LOG_INFO("DeviceRunner(sim): Loaded aicore_execute_wrapper from %s", aicore_so_path_.c_str());

        // Pass core identity setter function pointers to the AICore SO so it can
        // set per-thread subblock_id and cluster_id for pto-isa's TPUSH/TPOP hooks.
        auto set_identity_helpers =
            reinterpret_cast<void (*)(void *, void *)>(dlsym(aicore_so_handle_, "set_sim_core_identity_helpers"));
        if (set_identity_helpers != nullptr) {
            set_identity_helpers(
                reinterpret_cast<void *>(sim_context_set_subblock_id),
                reinterpret_cast<void *>(sim_context_set_cluster_id)
            );
        }
    }

    return 0;
}

void *DeviceRunner::allocate_tensor(size_t bytes) { return mem_alloc_.alloc(bytes); }

void DeviceRunner::free_tensor(void *dev_ptr) {
    if (dev_ptr != nullptr) {
        mem_alloc_.free(dev_ptr);
    }
}

int DeviceRunner::copy_to_device(void *dev_ptr, const void *host_ptr, size_t bytes) {
    // In simulation, this is just a memcpy
    std::memcpy(dev_ptr, host_ptr, bytes);
    return 0;
}

int DeviceRunner::copy_from_device(void *host_ptr, const void *dev_ptr, size_t bytes) {
    // In simulation, this is just a memcpy
    std::memcpy(host_ptr, dev_ptr, bytes);
    return 0;
}

int DeviceRunner::run(
    Runtime &runtime, int block_dim, int device_id, const std::vector<uint8_t> &aicpu_so_binary,
    const std::vector<uint8_t> &aicore_kernel_binary, int launch_aicpu_num, bool enable_dump_tensor
) {
    clear_cpu_sim_shared_storage();
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

    // Ensure device is initialized
    int rc = ensure_device_initialized(device_id, aicpu_so_binary, aicore_kernel_binary);
    if (rc != 0) {
        LOG_ERROR("ensure_device_initialized failed: %d", rc);
        return rc;
    }

    // Calculate execution parameters
    block_dim_ = block_dim;
    int num_aicore = block_dim * cores_per_blockdim_;

    if (num_aicore > RUNTIME_MAX_WORKER) {
        LOG_ERROR("num_aicore (%d) exceeds RUNTIME_MAX_WORKER (%d)", num_aicore, RUNTIME_MAX_WORKER);
        return -1;
    }

    // Initialize handshake buffers
    runtime.worker_count = num_aicore;
    worker_count_ = num_aicore;
    runtime.sche_cpu_num = launch_aicpu_num;

    // Calculate number of AIC cores
    int num_aic = block_dim;

    for (int i = 0; i < num_aicore; i++) {
        runtime.workers[i].aicpu_ready = 0;
        runtime.workers[i].aicore_done = 0;
        runtime.workers[i].control = 0;
        runtime.workers[i].task = 0;
        runtime.workers[i].task_status = 0;
        // First 1/3 are AIC, remaining 2/3 are AIV
        runtime.workers[i].core_type = (i < num_aic) ? CoreType::AIC : CoreType::AIV;
    }

    // Set function_bin_addr for each task: func_id_to_addr_[] stores CoreCallable
    // host address; dereference resolved_addr_ for the dlsym function pointer
    LOG_DEBUG("Setting function_bin_addr for Tasks (Simulation)");
    for (int i = 0; i < runtime.get_task_count(); i++) {
        Task *task = runtime.get_task(i);
        if (task != nullptr) {
            uint64_t callable_addr = runtime.get_function_bin_addr(task->func_id);
            const CoreCallable *c = reinterpret_cast<const CoreCallable *>(callable_addr);
            task->function_bin_addr = c->resolved_addr();
            LOG_DEBUG("Task %d (func_id=%d) -> function_bin_addr=0x%lx", i, task->func_id, task->function_bin_addr);
        }
    }

    // Store runtime pointer for print_handshake_results
    last_runtime_ = &runtime;

    // Initialize performance profiling if enabled
    if (runtime.enable_profiling) {
        rc = init_performance_profiling(runtime, num_aicore, device_id);
        if (rc != 0) {
            LOG_ERROR("init_performance_profiling failed: %d", rc);
            return rc;
        }
        // Start memory management thread
        perf_collector_.start_memory_manager([this](std::function<void()> fn) {
            return create_thread(std::move(fn));
        });
    }

    if (enable_dump_tensor) {
        // Initialize tensor dump (independent from profiling)
        rc = init_tensor_dump(runtime, num_aicore, device_id);
        if (rc != 0) {
            LOG_ERROR("init_tensor_dump failed: %d", rc);
            return rc;
        }
        dump_collector_.start_memory_manager();
    }

    auto perf_cleanup = RAIIScopeGuard([this]() {
        bool was_initialized = perf_collector_.is_initialized();
        if (was_initialized) {
            perf_collector_.stop_memory_manager();
        }
    });

    // Allocate simulated register blocks for all AICore cores
    size_t total_reg_size = num_aicore * SIM_REG_BLOCK_SIZE;
    void *reg_blocks = mem_alloc_.alloc(total_reg_size);
    if (reg_blocks == nullptr) {
        LOG_ERROR("Failed to allocate simulated register memory (%zu bytes)", total_reg_size);
        return -1;
    }
    std::memset(reg_blocks, 0, total_reg_size);

    auto reg_blocks_cleanup = RAIIScopeGuard([this, reg_blocks]() {
        mem_alloc_.free(reg_blocks);
    });

    // Build array of per-core register base addresses
    size_t regs_array_size = num_aicore * sizeof(uint64_t);
    uint64_t *regs_array = reinterpret_cast<uint64_t *>(mem_alloc_.alloc(regs_array_size));
    if (regs_array == nullptr) {
        LOG_ERROR("Failed to allocate register address array");
        return -1;
    }
    for (int i = 0; i < num_aicore; i++) {
        regs_array[i] = reinterpret_cast<uint64_t>(static_cast<uint8_t *>(reg_blocks) + i * SIM_REG_BLOCK_SIZE);
    }
    kernel_args_.regs = reinterpret_cast<uint64_t>(regs_array);

    auto regs_array_cleanup = RAIIScopeGuard([this]() {
        if (kernel_args_.regs != 0) {
            mem_alloc_.free(reinterpret_cast<void *>(kernel_args_.regs));
            kernel_args_.regs = 0;
        }
    });

    LOG_INFO("Allocated simulated registers: %d cores x 0x%x bytes", num_aicore, SIM_REG_BLOCK_SIZE);

    // Check if executors are loaded
    if (aicpu_execute_func_ == nullptr || aicore_execute_func_ == nullptr) {
        LOG_ERROR("Executor functions not loaded. Call ensure_binaries_loaded first.");
        return -1;
    }

    // Set platform regs in the AICPU .so before launching threads
    set_platform_regs_func_(kernel_args_.regs);
    set_platform_dump_base_func_(kernel_args_.dump_data_base);
    set_enable_dump_tensor_func_(enable_dump_tensor);

    // Launch AICPU threads (over-launch for affinity gate)
    constexpr int over_launch = PLATFORM_MAX_AICPU_THREADS_JUST_FOR_LAUNCH;
    LOG_INFO("Launching %d AICPU threads (logical=%d)", over_launch, launch_aicpu_num);
    std::vector<std::thread> aicpu_threads;
    aicpu_threads.reserve(over_launch);
    std::atomic<int> aicpu_rc{0};
    for (int i = 0; i < over_launch; i++) {
        aicpu_threads.push_back(create_thread([this, &runtime, launch_aicpu_num, over_launch, &aicpu_rc]() {
            if (!platform_aicpu_affinity_gate(launch_aicpu_num, over_launch)) {
                return;
            }
            int rc = aicpu_execute_func_(&runtime);
            if (rc != 0) {
                int expected = 0;
                aicpu_rc.compare_exchange_strong(expected, rc, std::memory_order_acq_rel);
            }
        }));
    }

    // Launch AICore threads
    LOG_INFO("Launching %d AICore thread(s)", num_aicore);
    std::vector<std::thread> aicore_threads;
    for (int i = 0; i < num_aicore; i++) {
        CoreType core_type = runtime.workers[i].core_type;
        uint32_t physical_core_id = static_cast<uint32_t>(i);
        aicore_threads.push_back(create_thread([this, &runtime, i, core_type, physical_core_id]() {
            aicore_execute_func_(&runtime, i, core_type, physical_core_id, kernel_args_.regs);
        }));
    }

    // Poll and collect performance data during execution (if enabled)
    std::thread collector_thread;
    if (runtime.enable_profiling) {
        collector_thread = create_thread([this, &runtime]() {
            poll_and_collect_performance_data(runtime.get_task_count());
        });
    }

    if (enable_dump_tensor) {
        // Poll and collect dump data in a separate collector thread
        std::thread dump_collector_thread([this]() {
            dump_collector_.poll_and_collect();
        });

        // Wait for all threads to complete
        LOG_INFO("Waiting for threads to complete");
        for (auto &t : aicpu_threads) {
            t.join();
        }
        for (auto &t : aicore_threads) {
            t.join();
        }

        // Signal collector that device execution is complete
        if (runtime.enable_profiling) {
            perf_collector_.signal_execution_complete();
        }
        dump_collector_.signal_execution_complete();

        // Wait for collector thread if it was launched
        if (runtime.enable_profiling && collector_thread.joinable()) {
            collector_thread.join();
        }
        if (dump_collector_thread.joinable()) {
            dump_collector_thread.join();
        }
    } else {
        // Wait for all threads to complete
        LOG_INFO("Waiting for threads to complete");
        for (auto &t : aicpu_threads) {
            t.join();
        }
        for (auto &t : aicore_threads) {
            t.join();
        }

        // Signal collector that device execution is complete
        if (runtime.enable_profiling) {
            perf_collector_.signal_execution_complete();
        }

        // Wait for collector thread if it was launched
        if (runtime.enable_profiling && collector_thread.joinable()) {
            collector_thread.join();
        }
    }

    LOG_INFO("All threads completed");

    int runtime_rc = aicpu_rc.load(std::memory_order_acquire);
    if (runtime_rc != 0) {
        LOG_ERROR("AICPU execution failed with rc=%d", runtime_rc);
        return runtime_rc;
    }

    // Stop memory management, drain remaining buffers, collect phase data, export
    if (runtime.enable_profiling) {
        perf_collector_.stop_memory_manager();
        perf_collector_.drain_remaining_buffers();
        perf_collector_.scan_remaining_perf_buffers();
        perf_collector_.collect_phase_data();
        export_swimlane_json();
    }

    if (enable_dump_tensor) {
        dump_collector_.stop_memory_manager();
        dump_collector_.drain_remaining_buffers();
        dump_collector_.scan_remaining_dump_buffers();
        dump_collector_.export_dump_files();
    }

    // Print handshake results at end of run
    print_handshake_results();

    // Close executor .so files now while the process is healthy, rather than
    // deferring to finalize()/destructor where static destruction order on
    // macOS can cause segfaults (aicpu.so contains file-scope statics with
    // std::mutex that crash when dlclosed during process teardown).
    unload_executor_binaries();

    return 0;
}

void DeviceRunner::print_handshake_results() {
    if (worker_count_ == 0 || last_runtime_ == nullptr) {
        return;
    }

    LOG_DEBUG("Handshake results for %d cores:", worker_count_);
    for (int i = 0; i < worker_count_; i++) {
        LOG_DEBUG(
            "  Core %d: aicore_done=%d aicpu_ready=%d control=%d task=%d", i, last_runtime_->workers[i].aicore_done,
            last_runtime_->workers[i].aicpu_ready, last_runtime_->workers[i].control, last_runtime_->workers[i].task
        );
    }
}

void DeviceRunner::unload_executor_binaries() {
    if (aicpu_so_handle_ != nullptr) {
        dlclose(aicpu_so_handle_);
        aicpu_so_handle_ = nullptr;
        aicpu_execute_func_ = nullptr;
        set_platform_regs_func_ = nullptr;
        set_platform_dump_base_func_ = nullptr;
        set_enable_dump_tensor_func_ = nullptr;
    }
    if (!aicpu_so_path_.empty()) {
        std::remove(aicpu_so_path_.c_str());
        aicpu_so_path_.clear();
    }

    if (aicore_so_handle_ != nullptr) {
        dlclose(aicore_so_handle_);
        aicore_so_handle_ = nullptr;
        aicore_execute_func_ = nullptr;
    }
    if (!aicore_so_path_.empty()) {
        std::remove(aicore_so_path_.c_str());
        aicore_so_path_.clear();
    }
}

int DeviceRunner::finalize() {
    // Skip if already finalized
    if (device_id_ == -1 && aicpu_so_handle_ == nullptr && aicore_so_handle_ == nullptr) {
        return 0;
    }

    // Cleanup performance profiling
    if (perf_collector_.is_initialized()) {
        auto free_cb = [](void *dev_ptr) -> int {
            free(dev_ptr);
            return 0;
        };

        perf_collector_.finalize(nullptr, free_cb);
    }

    if (dump_collector_.is_initialized()) {
        auto free_cb = [](void *dev_ptr, void *user_data) -> int {
            (void)user_data;
            free(dev_ptr);
            return 0;
        };

        dump_collector_.finalize(nullptr, free_cb, nullptr);
    }

    // Kernel binaries should have been removed by validate_runtime_impl()
    if (!func_id_to_addr_.empty()) {
        LOG_ERROR("finalize() called with %zu kernel binaries still cached", func_id_to_addr_.size());
        // Cleanup leaked handles and host copies
        for (auto &pair : func_id_to_addr_) {
            MappedKernel &kernel = pair.second;
            if (kernel.dl_handle != nullptr) {
                dlclose(kernel.dl_handle);
                LOG_DEBUG("Closed leaked kernel: func_id=%d", pair.first);
            }
            delete[] kernel.callable_buf;
        }
    }
    func_id_to_addr_.clear();

    // Close executor .so files (typically already closed by run(), this is a safety net)
    unload_executor_binaries();

    // Free all remaining allocations
    mem_alloc_.finalize();
    clear_cpu_sim_shared_storage();

    device_id_ = -1;
    worker_count_ = 0;
    last_runtime_ = nullptr;

    LOG_INFO("DeviceRunner(sim) finalized");
    return 0;
}

// =============================================================================
// Kernel Binary Upload (returns function address for caller to store in Runtime)
// =============================================================================

uint64_t DeviceRunner::upload_kernel_binary(int func_id, const uint8_t *bin_data, size_t bin_size) {
    if (bin_data == nullptr || bin_size == 0) {
        LOG_ERROR("Invalid kernel data");
        return 0;
    }

    // Return cached callable address if already uploaded
    auto it = func_id_to_addr_.find(func_id);
    if (it != func_id_to_addr_.end()) {
        LOG_INFO("Kernel func_id=%d already uploaded, returning cached address", func_id);
        return reinterpret_cast<uint64_t>(it->second.callable_buf);
    }

    // Extract binary from CoreCallable envelope
    const CoreCallable *callable = reinterpret_cast<const CoreCallable *>(bin_data);
    const void *kernel_binary = callable->binary_data();
    size_t kernel_size = callable->binary_size();

    // 1. Generate temp file path
    std::string tmpfile;
    if (!create_temp_so_file(
            "/tmp/kernel_" + std::to_string(func_id) + "_XXXXXX", reinterpret_cast<const uint8_t *>(kernel_binary),
            kernel_size, &tmpfile
        )) {
        LOG_ERROR("Failed to create temp file for kernel func_id=%d", func_id);
        return 0;
    }

    LOG_DEBUG("Uploading kernel .so: %s (size=%zu bytes)", tmpfile.c_str(), kernel_size);

    // 3. dlopen to load .so (RTLD_NOW ensures all symbols resolved immediately)
    void *handle = dlopen(tmpfile.c_str(), RTLD_NOW | RTLD_LOCAL);

    // 4. Remove temp file immediately (.so is already in memory)
    std::remove(tmpfile.c_str());

    if (!handle) {
        LOG_ERROR("dlopen failed: %s", dlerror());
        return 0;
    }

    // 5. dlsym to get kernel function address (unified entry point: "kernel_entry")
    void *func = dlsym(handle, "kernel_entry");
    if (!func) {
        LOG_ERROR("dlsym failed for 'kernel_entry': %s", dlerror());
        dlclose(handle);
        return 0;
    }

    // 6. Inject pto-isa simulation hooks into the kernel SO.
    //    Each kernel SO has its own copy of the inline static function pointers
    //    in cpu_stub.hpp, so every SO must be registered after dlopen.
    auto register_hooks = reinterpret_cast<void (*)(void *, void *)>(dlsym(handle, "pto_sim_register_hooks"));
    if (register_hooks != nullptr) {
        register_hooks(
            reinterpret_cast<void *>(pto_sim_get_subblock_id), reinterpret_cast<void *>(pto_sim_get_pipe_shared_state)
        );
    }

    // 6. Create host-memory copy of CoreCallable with resolved_addr_ = func_ptr
    uint8_t *copy = new uint8_t[bin_size];
    std::memcpy(copy, bin_data, bin_size);
    CoreCallable *callable_copy = reinterpret_cast<CoreCallable *>(copy);
    callable_copy->set_resolved_addr(reinterpret_cast<uint64_t>(func));

    // 7. Store mapping info for cleanup
    MappedKernel kernel;
    kernel.dl_handle = handle;
    kernel.callable_buf = copy;
    func_id_to_addr_[func_id] = kernel;

    LOG_DEBUG(
        "Registered kernel (dlopen): func_id=%d -> callable=0x%lx, func_addr=0x%lx, handle=%p", func_id,
        reinterpret_cast<uint64_t>(copy), reinterpret_cast<uint64_t>(func), handle
    );

    return reinterpret_cast<uint64_t>(copy);
}

void DeviceRunner::remove_kernel_binary(int func_id) {
    auto it = func_id_to_addr_.find(func_id);
    if (it == func_id_to_addr_.end()) {
        return;
    }

    MappedKernel &kernel = it->second;
    if (kernel.dl_handle != nullptr) {
        dlclose(kernel.dl_handle);
        LOG_DEBUG("Removed kernel binary (dlclose): func_id=%d, handle=%p", func_id, kernel.dl_handle);
    }
    delete[] kernel.callable_buf;

    func_id_to_addr_.erase(it);
}

// =============================================================================
// Performance Profiling Implementation
// =============================================================================

int DeviceRunner::init_performance_profiling(Runtime &runtime, int num_aicore, int device_id) {
    // Define allocation callback (a2a3sim: use malloc)
    auto alloc_cb = [](size_t size) -> void * {
        return malloc(size);
    };

    // Define free callback (a2a3sim: use free)
    auto free_cb = [](void *dev_ptr) -> int {
        free(dev_ptr);
        return 0;
    };

    // Simulation: no registration needed (pass nullptr)
    return perf_collector_.initialize(runtime, num_aicore, device_id, alloc_cb, nullptr, free_cb);
}

void DeviceRunner::poll_and_collect_performance_data(int expected_tasks) {
    perf_collector_.poll_and_collect(expected_tasks);
}

int DeviceRunner::export_swimlane_json(const std::string &output_path) {
    return perf_collector_.export_swimlane_json(output_path);
}

int DeviceRunner::init_tensor_dump(Runtime &runtime, int num_aicore, int device_id) {
    (void)num_aicore;
    int num_dump_threads = runtime.sche_cpu_num;

    auto alloc_cb = [](size_t size, void * /*user_data*/) -> void * {
        return malloc(size);
    };

    auto free_cb = [](void *dev_ptr, void * /*user_data*/) -> int {
        free(dev_ptr);
        return 0;
    };

    // Simulation: no registration needed (dev == host)
    int rc = dump_collector_.initialize(num_dump_threads, device_id, alloc_cb, nullptr, free_cb, nullptr, nullptr);
    if (rc != 0) {
        return rc;
    }

    kernel_args_.dump_data_base = reinterpret_cast<uint64_t>(dump_collector_.get_dump_shm_device_ptr());
    return 0;
}
