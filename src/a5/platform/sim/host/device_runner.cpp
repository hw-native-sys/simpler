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
 * It provides the same API as the real a5 implementation but uses
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
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#include "aicpu/platform_aicpu_affinity.h"
#include "callable.h"
#include "callable_protocol.h"
#include "utils/elf_build_id.h"
#include "utils/fnv1a_64.h"
#include "cpu_sim_context.h"
#include "host/raii_scope_guard.h"

// Function pointer types for dynamically loaded executors
typedef int (*aicpu_execute_func_t)(Runtime *runtime);
typedef void (*aicore_execute_func_t)(
    Runtime *runtime, int block_idx, CoreType core_type, uint32_t physical_core_id, uint64_t regs,
    uint32_t enable_profiling_flag, uint64_t aicore_l2_swimlane_ring_addrs, uint64_t aicore_pmu_ring_addrs
);
typedef void (*set_platform_regs_func_t)(uint64_t regs);

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

// malloc / free wrappers shared by all three profiling subsystems. Plain
// function pointers convert implicitly into the framework's std::function
// alloc / free shapes.
static void *prof_alloc_cb(size_t size) { return std::malloc(size); }

static int prof_free_cb(void *dev_ptr) {
    std::free(dev_ptr);
    return 0;
}

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
    // Failure of a later region leaves earlier peers committed on purpose:
    // pooled pointers previously returned to callers must stay valid even if
    // this resize attempt aborts.
    if (commit_region(gm_heap_arena_, cached_gm_heap_size_, gm_heap_size) != 0) return -1;
    if (commit_region(gm_sm_arena_, cached_gm_sm_size_, gm_sm_size) != 0) return -1;
    if (commit_region(runtime_arena_pool_, cached_runtime_arena_size_, runtime_arena_size) != 0) return -1;
    return 0;
}

void *DeviceRunner::acquire_pooled_gm_heap() {
    if (!gm_heap_arena_.is_committed()) return nullptr;
    return gm_heap_arena_.base();
}

void *DeviceRunner::acquire_pooled_gm_sm() {
    if (!gm_sm_arena_.is_committed()) return nullptr;
    return gm_sm_arena_.base();
}

void *DeviceRunner::acquire_pooled_runtime_arena() {
    if (!runtime_arena_pool_.is_committed()) return nullptr;
    return runtime_arena_pool_.base();
}

std::thread DeviceRunner::create_thread(std::function<void()> fn) {
    int dev_id = device_id_;
    return std::thread([dev_id, fn = std::move(fn)]() {
        pto_cpu_sim_bind_device(dev_id);
        fn();
        pto_cpu_sim_bind_device(-1);
    });
}

int DeviceRunner::attach_current_thread(int device_id) {
    if (device_id < 0) {
        LOG_ERROR("Invalid device_id: %d", device_id);
        return -1;
    }
    if (device_id_ != -1 && device_id_ != device_id) {
        LOG_ERROR(
            "DeviceRunner already initialized on device %d; finalize before switching to device %d", device_id_,
            device_id
        );
        return -1;
    }

    pto_cpu_sim_bind_device(device_id);
    pto_cpu_sim_acquire_device(device_id);
    device_id_ = device_id;
    return 0;
}

int DeviceRunner::ensure_device_initialized() {
    // device_id_ was set in attach_current_thread() during simpler_init.
    int rc = attach_current_thread(device_id_);
    if (rc != 0) return rc;
    return ensure_binaries_loaded();
}

int DeviceRunner::ensure_binaries_loaded() {
    // AICPU .so: load-once, matching onboard's binaries_loaded_ pattern.
    // Keeping the DSO alive across runs preserves g_aicpu_executor state
    // (orch_so_handle_ etc.), which is required for the orch-SO cache-hit path.
    if (!aicpu_so_loaded_ && !aicpu_so_binary_.empty()) {
        if (!create_temp_so_file(
                "/tmp/aicpu_sim_XXXXXX", aicpu_so_binary_.data(), aicpu_so_binary_.size(), &aicpu_so_path_
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

        set_dump_tensor_enabled_func_ =
            reinterpret_cast<void (*)(bool)>(dlsym(aicpu_so_handle_, "set_dump_tensor_enabled"));
        if (set_dump_tensor_enabled_func_ == nullptr) {
            LOG_ERROR("dlsym failed for set_dump_tensor_enabled: %s", dlerror());
            return -1;
        }

        set_platform_l2_swimlane_base_func_ =
            reinterpret_cast<void (*)(uint64_t)>(dlsym(aicpu_so_handle_, "set_platform_l2_swimlane_base"));
        if (set_platform_l2_swimlane_base_func_ == nullptr) {
            LOG_ERROR("dlsym failed for set_platform_l2_swimlane_base: %s", dlerror());
            return -1;
        }

        set_l2_swimlane_enabled_func_ =
            reinterpret_cast<void (*)(bool)>(dlsym(aicpu_so_handle_, "set_l2_swimlane_enabled"));
        if (set_l2_swimlane_enabled_func_ == nullptr) {
            LOG_ERROR("dlsym failed for set_l2_swimlane_enabled: %s", dlerror());
            return -1;
        }

        set_platform_pmu_base_func_ =
            reinterpret_cast<void (*)(uint64_t)>(dlsym(aicpu_so_handle_, "set_platform_pmu_base"));
        if (set_platform_pmu_base_func_ == nullptr) {
            LOG_ERROR("dlsym failed for set_platform_pmu_base: %s", dlerror());
            return -1;
        }

        set_pmu_enabled_func_ = reinterpret_cast<void (*)(bool)>(dlsym(aicpu_so_handle_, "set_pmu_enabled"));
        if (set_pmu_enabled_func_ == nullptr) {
            LOG_ERROR("dlsym failed for set_pmu_enabled: %s", dlerror());
            return -1;
        }

        set_scope_stats_enabled_func_ =
            reinterpret_cast<void (*)(bool)>(dlsym(aicpu_so_handle_, "set_scope_stats_enabled"));
        if (set_scope_stats_enabled_func_ == nullptr) {
            LOG_ERROR("dlsym failed for set_scope_stats_enabled: %s", dlerror());
            return -1;
        }

        set_platform_scope_stats_base_func_ =
            reinterpret_cast<void (*)(uint64_t)>(dlsym(aicpu_so_handle_, "set_platform_scope_stats_base"));
        if (set_platform_scope_stats_base_func_ == nullptr) {
            LOG_ERROR("dlsym failed for set_platform_scope_stats_base: %s", dlerror());
            return -1;
        }

        // Log config travels via the RTLD_GLOBAL HostLogger singleton in
        // libsimpler_log.so — already seeded by simpler_log_init() before the
        // AICPU sim SO was dlopen'd, so no per-SO setter forwarding is needed.

        aicpu_so_loaded_ = true;
        LOG_INFO_V0("DeviceRunner(sim): Loaded aicpu_execute from %s", aicpu_so_path_.c_str());
    }

    // AICore kernel .so: reload every run — kernel binary varies per case and
    // the AICore DSO holds no cross-run state that needs preserving.
    if (aicore_so_handle_ != nullptr) {
        dlclose(aicore_so_handle_);
        aicore_so_handle_ = nullptr;
        aicore_execute_func_ = nullptr;
    }
    if (!aicore_so_path_.empty()) {
        std::remove(aicore_so_path_.c_str());
        aicore_so_path_.clear();
    }

    // Write AICore binary to temp file and dlopen
    if (!aicore_kernel_binary_.empty()) {
        if (!create_temp_so_file(
                "/tmp/aicore_sim_XXXXXX", aicore_kernel_binary_.data(), aicore_kernel_binary_.size(), &aicore_so_path_
            )) {
            LOG_ERROR("Failed to create temp file for AICore SO");
            return -1;
        }

        aicore_so_handle_ = dlopen(aicore_so_path_.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (aicore_so_handle_ == nullptr) {
            LOG_ERROR("dlopen failed for AICore SO: %s", dlerror());
            return -1;
        }

        aicore_execute_func_ =
            reinterpret_cast<aicore_execute_func_t>(dlsym(aicore_so_handle_, "aicore_execute_wrapper"));
        if (aicore_execute_func_ == nullptr) {
            LOG_ERROR("dlsym failed for aicore_execute_wrapper: %s", dlerror());
            return -1;
        }
        LOG_INFO_V0("DeviceRunner(sim): Loaded aicore_execute_wrapper from %s", aicore_so_path_.c_str());

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

int DeviceRunner::run(Runtime &runtime, int block_dim, int launch_aicpu_num) {
    clear_cpu_sim_shared_storage();
    // Validate launch_aicpu_num
    if (launch_aicpu_num < 1 || launch_aicpu_num > PLATFORM_MAX_AICPU_THREADS) {
        LOG_ERROR("launch_aicpu_num (%d) must be in range [1, %d]", launch_aicpu_num, PLATFORM_MAX_AICPU_THREADS);
        return -1;
    }

    // Validate block_dim. Sim has no stream resource query, so the static
    // platform capacity is the bound (mirrors the onboard fallback in
    // DeviceRunner::validate_block_dim). The scheduler assigns cores to
    // threads cluster-aligned round-robin, so block_dim need not be evenly
    // divisible by the scheduler thread count.
    //
    // block_dim == 0 is the CallConfig "auto" sentinel — resolve to the
    // static max since sim has no per-stream resource query.
    if (block_dim == 0) {
        block_dim = PLATFORM_MAX_BLOCKDIM;
        LOG_INFO_V0("block_dim auto-resolved to %d", block_dim);
    }
    if (block_dim < 1 || block_dim > PLATFORM_MAX_BLOCKDIM) {
        LOG_ERROR("block_dim (%d) must be in range [1, %d]", block_dim, PLATFORM_MAX_BLOCKDIM);
        return -1;
    }

    // Ensure device is initialized
    int rc = ensure_device_initialized();
    if (rc != 0) {
        LOG_ERROR("ensure_device_initialized failed: %d", rc);
        return rc;
    }

    // Lazy-allocate the 8-byte device_wall buffer on first run.
    if (device_wall_dev_ptr_ == nullptr) {
        device_wall_dev_ptr_ = allocate_tensor(sizeof(uint64_t));
        if (device_wall_dev_ptr_ != nullptr) {
            kernel_args_.device_wall_data_base = reinterpret_cast<uint64_t>(device_wall_dev_ptr_);
            *static_cast<uint64_t *>(device_wall_dev_ptr_) = 0;
        }
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
    runtime.aicpu_thread_num = launch_aicpu_num;

    // Calculate number of AIC cores
    int num_aic = block_dim;
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
        // First 1/3 are AIC, remaining 2/3 are AIV
        runtime.workers[i].core_type = (i < num_aic) ? CoreType::AIC : CoreType::AIV;
    }
    // Profiling state lives on KernelArgs now (no longer mirrored into Handshake).
    kernel_args_.enable_profiling_flag = enable_profiling_flag;

    // Set function_bin_addr for each task: Runtime::func_id_to_addr_[] stores
    // a CoreCallable host address (chip buffer + offset); dereference
    // resolved_addr_ for the dlsym function pointer.
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

    rc = prepare_orch_so(runtime);
    if (rc != 0) {
        LOG_ERROR("prepare_orch_so failed: %d", rc);
        return rc;
    }

    // Store runtime pointer for print_handshake_results
    last_runtime_ = &runtime;

    // Initialize per-subsystem shared memory.
    if (enable_l2_swimlane_) {
        rc = init_l2_swimlane(num_aicore, device_id_);
        if (rc != 0) {
            LOG_ERROR("init_l2_swimlane failed: %d", rc);
            return rc;
        }
    }

    if (enable_dump_tensor_) {
        // Initialize tensor dump (independent from profiling)
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
            kernel_args_.pmu_data_base = 0;
            enable_pmu_ = false;
        }
    }

    if (enable_scope_stats_) {
        rc = init_scope_stats(launch_aicpu_num);
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

    // Allocate simulated register blocks for all AICore cores
    // Using sparse mapping: 2 x 4KB pages per core instead of 24KB contiguous block
    size_t total_reg_size = num_aicore * SIM_REG_TOTAL_SIZE;
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
        regs_array[i] = reinterpret_cast<uint64_t>(static_cast<uint8_t *>(reg_blocks) + i * SIM_REG_TOTAL_SIZE);
    }
    kernel_args_.regs = reinterpret_cast<uint64_t>(regs_array);

    auto regs_array_cleanup = RAIIScopeGuard([this]() {
        if (kernel_args_.regs != 0) {
            mem_alloc_.free(reinterpret_cast<void *>(kernel_args_.regs));
            kernel_args_.regs = 0;
        }
    });

    LOG_INFO_V0(
        "Allocated simulated registers: %d cores x 0x%x bytes (sparse: 3 pages)", num_aicore, SIM_REG_TOTAL_SIZE
    );

    // Check if executors are loaded
    if (aicpu_execute_func_ == nullptr || aicore_execute_func_ == nullptr || set_platform_regs_func_ == nullptr ||
        set_platform_dump_base_func_ == nullptr || set_dump_tensor_enabled_func_ == nullptr ||
        set_platform_pmu_base_func_ == nullptr || set_pmu_enabled_func_ == nullptr ||
        set_scope_stats_enabled_func_ == nullptr || set_platform_scope_stats_base_func_ == nullptr) {
        LOG_ERROR("Executor functions not loaded. Call ensure_binaries_loaded first.");
        return -1;
    }

    set_platform_regs_func_(kernel_args_.regs);
    set_platform_dump_base_func_(kernel_args_.dump_data_base);
    set_dump_tensor_enabled_func_(enable_dump_tensor_);
    set_platform_l2_swimlane_base_func_(kernel_args_.l2_swimlane_data_base);
    set_l2_swimlane_enabled_func_(enable_l2_swimlane_);
    set_platform_pmu_base_func_(kernel_args_.pmu_data_base);
    set_pmu_enabled_func_(enable_pmu_);
    set_scope_stats_enabled_func_(enable_scope_stats_);
    set_platform_scope_stats_base_func_(kernel_args_.scope_stats_data_base);

    // No per-SO log-config push: HostLogger lives in libsimpler_log.so
    // (RTLD_GLOBAL singleton) and the AICPU sim SO reads it directly via the
    // same global lookup.

    // Start collector mgmt + poll threads now, just before kernels launch.
    // Starting earlier wastes CPU on empty queues and risks tripping
    // ProfilerBase's poll-loop idle-timeout if the AICPU SO is slow to come
    // up.
    auto thread_factory = [this](std::function<void()> fn) {
        return create_thread(std::move(fn));
    };
    if (enable_l2_swimlane_) {
        l2_swimlane_collector_.start(thread_factory);
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

    // Launch AICPU threads (over-launch for affinity gate)
    constexpr int over_launch = PLATFORM_MAX_AICPU_THREADS_JUST_FOR_LAUNCH;
    LOG_INFO_V0("Launching %d AICPU threads (logical=%d)", over_launch, launch_aicpu_num);
    std::vector<std::thread> aicpu_threads;
    aicpu_threads.reserve(over_launch);
    std::atomic<int> aicpu_rc{0};

    // Sim "device wall" capture — see a2a3 sim/onboard kernel.cpp for the
    // shape. Writes via device_wall_data_base (sim's "device pointer" is a
    // host malloc'd uint64).
    if (kernel_args_.device_wall_data_base != 0) {
        *reinterpret_cast<uint64_t *>(kernel_args_.device_wall_data_base) = 0;
    }
    const auto sim_t0 = std::chrono::steady_clock::now();

    for (int i = 0; i < over_launch; i++) {
        aicpu_threads.push_back(create_thread([this, &runtime, launch_aicpu_num, over_launch, &aicpu_rc, sim_t0]() {
            if (!platform_aicpu_affinity_gate(launch_aicpu_num, over_launch)) {
                return;
            }
            int rc = aicpu_execute_func_(&runtime);
            if (rc != 0) {
                int expected = 0;
                aicpu_rc.compare_exchange_strong(expected, rc, std::memory_order_acq_rel);
            }
            if (kernel_args_.device_wall_data_base != 0) {
                const auto t1 = std::chrono::steady_clock::now();
                *reinterpret_cast<uint64_t *>(kernel_args_.device_wall_data_base) =
                    static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - sim_t0).count());
            }
        }));
    }

    // Launch AICore threads
    LOG_INFO_V0("Launching %d AICore thread(s)", num_aicore);
    std::vector<std::thread> aicore_threads;
    for (int i = 0; i < num_aicore; i++) {
        CoreType core_type = runtime.workers[i].core_type;
        uint32_t physical_core_id = static_cast<uint32_t>(i);
        aicore_threads.push_back(create_thread([this, &runtime, i, core_type, physical_core_id]() {
            aicore_execute_func_(
                &runtime, i, core_type, physical_core_id, kernel_args_.regs, kernel_args_.enable_profiling_flag,
                kernel_args_.aicore_l2_swimlane_ring_addrs, kernel_args_.aicore_pmu_ring_addrs
            );
        }));
    }

    LOG_INFO_V0("Waiting for threads to complete");
    for (auto &t : aicpu_threads) {
        t.join();
    }
    for (auto &t : aicore_threads) {
        t.join();
    }

    LOG_INFO_V0("All threads completed");

    // Snapshot device_wall buffer into device_wall_ns_ (sim: single deref,
    // no DMA — mirrors onboard's post-sync rtMemcpy D2H).
    device_wall_ns_ = 0;
    if (device_wall_dev_ptr_ != nullptr) {
        device_wall_ns_ = *static_cast<uint64_t *>(device_wall_dev_ptr_);
    }

    int runtime_rc = aicpu_rc.load(std::memory_order_acquire);
    if (runtime_rc != 0) {
        LOG_ERROR("AICPU execution failed with rc=%d", runtime_rc);
        return runtime_rc;
    }

    // Tear down collectors. stop() joins mgmt then collector in the only
    // safe order (mgmt's final-drain pass into L2 has poll as its
    // consumer). Diagnostic exports use the per-task `output_prefix_`
    // directory the user set on CallConfig (validate() enforces non-empty
    // upstream).
    if (enable_l2_swimlane_) {
        l2_swimlane_collector_.stop();
        l2_swimlane_collector_.read_phase_header_metadata();
        l2_swimlane_collector_.reconcile_counters();
        l2_swimlane_collector_.export_swimlane_json();
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

    // Print handshake results at end of run
    print_handshake_results();

    // Close AICore kernel .so now while the process is healthy.
    // AICPU .so is kept alive (load-once) so that g_aicpu_executor state
    // (orch_so_handle_ etc.) survives across runs for the orch-SO cache-hit path.
    // It will be closed in finalize() / unload_executor_binaries().
    if (aicore_so_handle_ != nullptr) {
        dlclose(aicore_so_handle_);
        aicore_so_handle_ = nullptr;
        aicore_execute_func_ = nullptr;
    }
    if (!aicore_so_path_.empty()) {
        std::remove(aicore_so_path_.c_str());
        aicore_so_path_.clear();
    }

    return 0;
}

void DeviceRunner::print_handshake_results() {
    if (worker_count_ == 0 || last_runtime_ == nullptr) {
        return;
    }

    LOG_DEBUG("Handshake results for %d cores:", worker_count_);
    for (int i = 0; i < worker_count_; i++) {
        LOG_DEBUG(
            "  Core %d: aicore_done=%d aicpu_ready=%d task=%d", i, last_runtime_->workers[i].aicore_done,
            last_runtime_->workers[i].aicpu_ready, last_runtime_->workers[i].task
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
        set_dump_tensor_enabled_func_ = nullptr;
        set_platform_l2_swimlane_base_func_ = nullptr;
        set_l2_swimlane_enabled_func_ = nullptr;
        set_platform_pmu_base_func_ = nullptr;
        set_pmu_enabled_func_ = nullptr;
        set_scope_stats_enabled_func_ = nullptr;
        set_platform_scope_stats_base_func_ = nullptr;
        aicpu_so_loaded_ = false;
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

int DeviceRunner::prepare_orch_so(Runtime &runtime) {
    // Prepared-callable flow only — bytes were staged at
    // register_callable time; here we only stamp metadata onto
    // the runtime and resolve `register_new_callable_id_` from first sighting.
    const int32_t cid = runtime.get_active_callable_id();
    if (cid < 0) {
        LOG_ERROR("prepare_orch_so: no active callable_id; prepared-callable flow required");
        return -1;
    }
    auto it = callables_.find(cid);
    if (it == callables_.end()) {
        LOG_ERROR("prepare_orch_so: callable_id=%d not registered", cid);
        return -1;
    }
    const auto &state = it->second;
    // hbg variant: orch SO never crosses host/device boundary.
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
    runtime.set_active_callable_id(cid, first_sighting);
    LOG_INFO_V0(
        "Orch SO prepared cid=%d hash=0x%lx %zu bytes (is_new=%d)", cid, state.hash, state.dev_orch_so_size,
        first_sighting ? 1 : 0
    );
    return 0;
}

int DeviceRunner::register_callable(
    int32_t callable_id, const void *orch_so_data, size_t orch_so_size, const char *func_name, const char *config_name,
    std::vector<std::pair<int, uint64_t>> kernel_addrs, std::vector<ArgDirection> signature
) {
    if (callable_id < 0 || callable_id >= MAX_REGISTERED_CALLABLE_IDS) {
        LOG_ERROR("register_callable: callable_id=%d out of range [0, %d)", callable_id, MAX_REGISTERED_CALLABLE_IDS);
        return -1;
    }
    if (orch_so_data == nullptr || orch_so_size == 0) {
        LOG_ERROR("register_callable: empty orch SO for callable_id=%d", callable_id);
        return -1;
    }
    if (callables_.count(callable_id) != 0) {
        LOG_ERROR("register_callable: callable_id=%d already registered", callable_id);
        return -1;
    }

    const uint64_t hash = simpler::common::utils::elf_build_id_64(orch_so_data, orch_so_size);

    auto buf_it = orch_so_dedup_.find(hash);
    uint64_t dev_addr = 0;
    if (buf_it == orch_so_dedup_.end()) {
        void *buf = mem_alloc_.alloc(orch_so_size);
        if (buf == nullptr) {
            LOG_ERROR("register_callable: alloc %zu bytes failed", orch_so_size);
            return -1;
        }
        // Sim shares an address space with the simulated AICPU thread, so a
        // plain memcpy is the moral equivalent of rtMemcpy on hardware.
        std::memcpy(buf, orch_so_data, orch_so_size);
        OrchSoBuffer entry;
        entry.dev_addr = buf;
        entry.capacity = orch_so_size;
        entry.refcount = 1;
        orch_so_dedup_.emplace(hash, entry);
        dev_addr = reinterpret_cast<uint64_t>(buf);
        LOG_INFO_V0("register_callable: hash=0x%lx new buffer %zu bytes", hash, orch_so_size);
    } else {
        buf_it->second.refcount++;
        dev_addr = reinterpret_cast<uint64_t>(buf_it->second.dev_addr);
        LOG_INFO_V0("register_callable: hash=0x%lx shared buffer (refcount=%d)", hash, buf_it->second.refcount);
    }

    CallableState state;
    state.hash = hash;
    state.dev_orch_so_addr = dev_addr;
    state.dev_orch_so_size = orch_so_size;
    state.func_name = (func_name != nullptr) ? func_name : "";
    state.config_name = (config_name != nullptr) ? config_name : "";
    state.kernel_addrs = std::move(kernel_addrs);
    state.signature = std::move(signature);
    callables_.emplace(callable_id, std::move(state));
    return 0;
}

int DeviceRunner::register_callable_host_orch(
    int32_t callable_id, void *host_dlopen_handle, void *host_orch_func_ptr,
    std::vector<std::pair<int, uint64_t>> kernel_addrs, std::vector<ArgDirection> signature
) {
    if (callable_id < 0 || callable_id >= MAX_REGISTERED_CALLABLE_IDS) {
        LOG_ERROR(
            "register_callable_host_orch: callable_id=%d out of range [0, %d)", callable_id, MAX_REGISTERED_CALLABLE_IDS
        );
        return -1;
    }
    if (host_dlopen_handle == nullptr || host_orch_func_ptr == nullptr) {
        LOG_ERROR("register_callable_host_orch: null handle/fn for callable_id=%d", callable_id);
        return -1;
    }
    if (callables_.count(callable_id) != 0) {
        LOG_ERROR("register_callable_host_orch: callable_id=%d already registered", callable_id);
        return -1;
    }

    CallableState state;
    state.host_dlopen_handle = host_dlopen_handle;
    state.host_orch_func_ptr = host_orch_func_ptr;
    state.kernel_addrs = std::move(kernel_addrs);
    state.signature = std::move(signature);
    callables_.emplace(callable_id, std::move(state));
    ++host_dlopen_total_;
    LOG_INFO_V0("register_callable_host_orch: cid=%d (host dlopen #%zu)", callable_id, host_dlopen_total_);
    return 0;
}

int DeviceRunner::unregister_callable(int32_t callable_id) {
    auto it = callables_.find(callable_id);
    if (it == callables_.end()) {
        return 0;
    }
    CallableState state = std::move(it->second);
    callables_.erase(it);
    aicpu_seen_callable_ids_.erase(callable_id);

    if (state.host_dlopen_handle != nullptr) {
        // hbg path: dlclose host handle; no orch SO refcount.
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

bool DeviceRunner::has_callable(int32_t callable_id) const { return callables_.count(callable_id) != 0; }

BindCallableResult DeviceRunner::bind_callable_to_runtime(Runtime &runtime, int32_t callable_id) {
    auto it = callables_.find(callable_id);
    if (it == callables_.end()) {
        LOG_ERROR("bind_callable_to_runtime: callable_id=%d not registered", callable_id);
        return {-1, nullptr, nullptr, 0};
    }
    const auto &state = it->second;
    for (const auto &kv : state.kernel_addrs) {
        if (kv.first < 0 || kv.first >= RUNTIME_MAX_FUNC_ID) {
            LOG_ERROR("bind_callable_to_runtime: func_id=%d out of range", kv.first);
            return {-1, nullptr, nullptr, 0};
        }
        runtime.replay_function_bin_addr(kv.first, kv.second);
    }
    runtime.set_device_orch_func_name(state.func_name.c_str());
    runtime.set_device_orch_config_name(state.config_name.c_str());
    runtime.set_active_callable_id(callable_id, /*is_new=*/false);
    return {
        0, state.host_orch_func_ptr, state.signature.empty() ? nullptr : state.signature.data(),
        static_cast<int>(state.signature.size())
    };
}

int DeviceRunner::finalize() {
    // Skip if already finalized
    if (device_id_ == -1 && aicpu_so_handle_ == nullptr && aicore_so_handle_ == nullptr) {
        return 0;
    }

    // Cleanup all profiling subsystems.
    if (l2_swimlane_collector_.is_initialized()) {
        l2_swimlane_collector_.finalize(/*unregister_cb=*/nullptr, prof_free_cb);
    }
    if (dump_collector_.is_initialized()) {
        dump_collector_.finalize(/*unregister_cb=*/nullptr, prof_free_cb);
    }
    if (pmu_collector_.is_initialized()) {
        pmu_collector_.finalize(/*unregister_cb=*/nullptr, prof_free_cb);
    }
    if (scope_stats_collector_.is_initialized()) {
        scope_stats_collector_.finalize(/*unregister_cb=*/nullptr, prof_free_cb);
        kernel_args_.scope_stats_data_base = 0;
    }

    // Release any chip callable buffers uploaded via upload_chip_callable_buffer.
    // Pool semantics mirror per-fid binaries: never freed until finalize.
    for (auto &kv : chip_callable_buffers_) {
        for (void *h : kv.second.dlopen_handles) {
            if (h != nullptr) dlclose(h);
        }
        delete[] kv.second.host_scratch;
        LOG_DEBUG(
            "Freed chip callable buffer (sim): chip_dev=0x%lx, size=%zu, hash=0x%lx", kv.second.chip_dev,
            kv.second.total_size, kv.first
        );
    }
    chip_callable_buffers_.clear();

    // Release any prepared-callable orch SO buffers callers forgot to drop.
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
    for (auto &kv : callables_) {
        if (kv.second.host_dlopen_handle != nullptr) {
            dlclose(kv.second.host_dlopen_handle);
        }
    }
    callables_.clear();
    aicpu_seen_callable_ids_.clear();
    aicpu_dlopen_total_ = 0;

    // Close executor .so files (typically already closed by run(), this is a safety net)
    unload_executor_binaries();

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

    // Free all remaining allocations
    mem_alloc_.finalize();
    clear_cpu_sim_shared_storage();

    // Free the 8-byte device_wall buffer (allocated lazily in run()).
    if (device_wall_dev_ptr_ != nullptr) {
        free_tensor(device_wall_dev_ptr_);
        device_wall_dev_ptr_ = nullptr;
    }
    device_id_ = -1;
    worker_count_ = 0;
    last_runtime_ = nullptr;

    LOG_INFO_V0("DeviceRunner(sim) finalized");
    return 0;
}

// =============================================================================
// Chip Callable Buffer Upload (returns host address of ChipCallable header)
// =============================================================================

uint64_t DeviceRunner::upload_chip_callable_buffer(const ChipCallable *callable) {
    if (callable == nullptr || callable->child_count() == 0) {
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
            "Chip callable dedup hit (sim): chip_dev=0x%lx, size=%zu, hash=0x%lx", it->second.chip_dev,
            it->second.total_size, hash
        );
        return it->second.chip_dev;
    }

    auto *scratch = new uint8_t[total_size];
    std::memcpy(scratch, raw_bytes, total_size);

    // Per-child dlopen + dlsym kernel_entry + register pto-sim hooks, then
    // patch the child's resolved_addr_ to the function pointer. A scope guard
    // owns scratch and any dlopen'd handles until the success path dismisses
    // it; every early return unwinds cleanly.
    std::vector<void *> dlopen_handles;
    dlopen_handles.reserve(callable->child_count());
    auto cleanup = RAIIScopeGuard([&]() {
        for (void *h : dlopen_handles)
            dlclose(h);
        delete[] scratch;
    });

    for (int32_t i = 0; i < callable->child_count(); ++i) {
        const uint32_t off = callable->child_offset(i);
        auto *child_in_scratch = reinterpret_cast<CoreCallable *>(scratch + kHeaderSize + off);
        const void *kernel_binary = child_in_scratch->binary_data();
        size_t kernel_size = static_cast<size_t>(child_in_scratch->binary_size());

        std::string tmpfile;
        if (!create_temp_so_file(
                "/tmp/kernel_" + std::to_string(callable->child_func_id(i)) + "_XXXXXX",
                reinterpret_cast<const uint8_t *>(kernel_binary), kernel_size, &tmpfile
            )) {
            LOG_ERROR("Failed to create temp file for child kernel #%d", i);
            return 0;
        }

        void *handle = dlopen(tmpfile.c_str(), RTLD_NOW | RTLD_LOCAL);
        std::remove(tmpfile.c_str());
        if (!handle) {
            LOG_ERROR("dlopen failed for child kernel #%d: %s", i, dlerror());
            return 0;
        }
        dlopen_handles.push_back(handle);

        void *func = dlsym(handle, "kernel_entry");
        if (!func) {
            LOG_ERROR("dlsym failed for child kernel #%d 'kernel_entry': %s", i, dlerror());
            return 0;
        }

        auto register_hooks = reinterpret_cast<void (*)(void *, void *)>(dlsym(handle, "pto_sim_register_hooks"));
        if (register_hooks != nullptr) {
            register_hooks(
                reinterpret_cast<void *>(pto_sim_get_subblock_id),
                reinterpret_cast<void *>(pto_sim_get_pipe_shared_state)
            );
        }

        child_in_scratch->set_resolved_addr(reinterpret_cast<uint64_t>(func));
    }

    cleanup.dismiss();
    const uint64_t chip_dev = reinterpret_cast<uint64_t>(scratch);
    chip_callable_buffers_.emplace(hash, ChipCallableBuffer{chip_dev, scratch, total_size, std::move(dlopen_handles)});
    LOG_DEBUG(
        "Uploaded chip callable (sim): chip_dev=0x%lx, size=%zu, child_count=%d, hash=0x%lx", chip_dev, total_size,
        callable->child_count(), hash
    );
    return chip_dev;
}

// =============================================================================
// Performance Profiling Implementation
// =============================================================================

void DeviceRunner::finalize_collectors() {
    if (l2_swimlane_collector_.is_initialized()) {
        l2_swimlane_collector_.stop();
    }
    if (dump_collector_.is_initialized()) {
        dump_collector_.stop();
    }
    if (pmu_collector_.is_initialized()) {
        pmu_collector_.stop();
    }
}

int DeviceRunner::init_l2_swimlane(int num_aicore, int device_id) {
    int rc = l2_swimlane_collector_.initialize(
        num_aicore, device_id, l2_swimlane_level_, prof_alloc_cb, /*register_cb=*/nullptr, prof_free_cb, output_prefix_
    );
    if (rc == 0) {
        kernel_args_.l2_swimlane_data_base =
            reinterpret_cast<uint64_t>(l2_swimlane_collector_.get_l2_swimlane_setup_device_ptr());
        kernel_args_.aicore_l2_swimlane_ring_addrs =
            reinterpret_cast<uint64_t>(l2_swimlane_collector_.get_aicore_ring_addrs_device_ptr());
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

    kernel_args_.dump_data_base = reinterpret_cast<uint64_t>(dump_collector_.get_dump_shm_device_ptr());
    return 0;
}

int DeviceRunner::init_pmu(
    int num_cores, int num_threads, const std::string &csv_path, PmuEventType event_type, int /*device_id*/
) {
    int rc = pmu_collector_.init(
        num_cores, num_threads, csv_path, event_type, prof_alloc_cb, /*register_cb=*/nullptr, prof_free_cb,
        /*device_id=*/-1
    );
    if (rc == 0) {
        kernel_args_.pmu_data_base = reinterpret_cast<uint64_t>(pmu_collector_.get_pmu_shm_device_ptr());
        kernel_args_.aicore_pmu_ring_addrs =
            reinterpret_cast<uint64_t>(pmu_collector_.get_aicore_ring_addrs_device_ptr());
    }
    return rc;
}

int DeviceRunner::init_scope_stats(int num_threads) {
    // a5 sim: register_cb=nullptr, so the collector mallocs a host shadow per
    // device buffer; sim's profiling_copy_* are plain memcpys, so the dev/host
    // shadow path collapses to one allocation pair without address-space tricks.
    int rc = scope_stats_collector_.init(
        num_threads, prof_alloc_cb, /*register_cb=*/nullptr, prof_free_cb, /*device_id=*/-1
    );
    if (rc != 0) {
        return rc;
    }
    kernel_args_.scope_stats_data_base =
        reinterpret_cast<uint64_t>(scope_stats_collector_.get_scope_stats_shm_device_ptr());
    return 0;
}
