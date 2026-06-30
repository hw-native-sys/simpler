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
 * a2a3 sim DeviceRunner implementation — wired against a2a3's aicore_execute
 * signature + dep_gen support. Shared arena/tensor/callable lifecycle lives
 * on SimDeviceRunnerBase; see device_runner_base.cpp.
 */

#include "device_runner.h"

#include <dlfcn.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "aicpu/device_phase_aicpu.h"
#include "aicpu/platform_aicpu_affinity.h"
#include "callable_protocol.h"
#include "common/memory_barrier.h"
#include "common/platform_config.h"
#include "common/unified_log.h"
#include "cpu_sim_context.h"
#include "host/raii_scope_guard.h"
#include "runtime.h"

// dep_gen_replay_emit_deps_json: strong symbol provided by
// runtime/tensormap_and_ringbuffer/host/dep_gen_replay.cpp when that runtime is
// linked into host_runtime.so. host_build_graph has no replay implementation
// today, so its host_runtime.so falls through to this weak stub. Hidden
// visibility keeps the stub off the global symbol table so RTLD_GLOBAL can't
// let it shadow the strong symbol in cross-.so loads.
// LOG_DEBUG (not WARN): runtimes that don't link dep_gen never enable it in
// practice, so this path is unreachable for end users — the symbol exists
// purely to keep the .so loadable.
extern "C" __attribute__((weak, visibility("hidden"))) int dep_gen_replay_emit_deps_json(
    const struct DepGenRecord * /*records*/, size_t /*num_records*/, const char * /*deps_json_path*/
) {
    LOG_DEBUG("dep_gen replay not implemented for this runtime — deps.json skipped");
    return -1;
}

DeviceRunner::~DeviceRunner() { finalize(); }

int DeviceRunner::ensure_binaries_loaded() {
    // AICPU .so: load-once, matching onboard's binaries_loaded_ pattern.
    // Keeping the DSO alive across runs preserves g_aicpu_executor state
    // (orch_so_handle_ etc.), which is required for the orch-SO cache-hit path.
    if (!aicpu_so_loaded_ && !aicpu_so_binary_.empty()) {
        if (!simpler::common::sim_host::create_temp_so_file(
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

        auto load_sym = [this](const char *name, void **out) -> bool {
            void *sym = dlsym(aicpu_so_handle_, name);
            if (sym == nullptr) {
                LOG_ERROR("dlsym failed for %s: %s", name, dlerror());
                return false;
            }
            *out = sym;
            return true;
        };
        auto load_optional_sym = [this](const char *name, void **out) {
            dlerror();
            void *sym = dlsym(aicpu_so_handle_, name);
            if (sym == nullptr) {
                LOG_DEBUG("Optional dlsym skipped for %s: %s", name, dlerror());
            }
            *out = sym;
        };

        if (!load_sym("aicpu_execute", reinterpret_cast<void **>(&aicpu_execute_func_))) return -1;
        load_optional_sym("aicpu_register_callable", reinterpret_cast<void **>(&aicpu_register_callable_func_));
        if (!load_sym("set_platform_regs", reinterpret_cast<void **>(&set_platform_regs_func_))) return -1;
        load_optional_sym("set_orch_device_id", reinterpret_cast<void **>(&set_orch_device_id_func_));
        if (!load_sym("set_platform_dump_base", reinterpret_cast<void **>(&set_platform_dump_base_func_))) return -1;
        if (!load_sym("set_platform_phase_base", reinterpret_cast<void **>(&set_platform_phase_base_func_))) return -1;
        if (!load_sym("set_dump_args_enabled", reinterpret_cast<void **>(&set_dump_args_enabled_func_))) return -1;
        if (!load_sym("set_platform_l2_swimlane_base", reinterpret_cast<void **>(&set_platform_l2_swimlane_base_func_)))
            return -1;
        if (!load_sym(
                "set_platform_l2_swimlane_aicore_rotation_table",
                reinterpret_cast<void **>(&set_platform_l2_swimlane_aicore_rotation_table_func_)
            ))
            return -1;
        if (!load_sym("set_l2_swimlane_enabled", reinterpret_cast<void **>(&set_l2_swimlane_enabled_func_))) return -1;
        if (!load_sym("set_platform_pmu_base", reinterpret_cast<void **>(&set_platform_pmu_base_func_))) return -1;
        if (!load_sym("set_platform_pmu_reg_addrs", reinterpret_cast<void **>(&set_platform_pmu_reg_addrs_func_)))
            return -1;
        if (!load_sym("set_pmu_enabled", reinterpret_cast<void **>(&set_pmu_enabled_func_))) return -1;
        if (!load_sym("set_platform_dep_gen_base", reinterpret_cast<void **>(&set_platform_dep_gen_base_func_)))
            return -1;
        if (!load_sym("set_dep_gen_enabled", reinterpret_cast<void **>(&set_dep_gen_enabled_func_))) return -1;
        if (!load_sym("set_scope_stats_enabled", reinterpret_cast<void **>(&set_scope_stats_enabled_func_))) return -1;
        if (!load_sym("set_platform_scope_stats_base", reinterpret_cast<void **>(&set_platform_scope_stats_base_func_)))
            return -1;

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

    if (!aicore_kernel_binary_.empty()) {
        if (!simpler::common::sim_host::create_temp_so_file(
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
            reinterpret_cast<void (*)(Runtime *, int, CoreType, uint32_t, uint64_t, uint32_t, uint64_t)>(
                dlsym(aicore_so_handle_, "aicore_execute_wrapper")
            );
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

int DeviceRunner::invoke_aicpu_register_callable(Runtime &runtime) {
    if (aicpu_register_callable_func_ == nullptr || set_orch_device_id_func_ == nullptr) {
        LOG_ERROR("Register-callable functions not loaded. Call ensure_binaries_loaded first.");
        return -1;
    }
    set_orch_device_id_func_(device_id_);
    // Extract the orch-SO descriptor from the stamped Runtime into the small
    // RegisterCallableArgs the renamed entry expects (sim shares process
    // memory, so the name pointers stay valid for the synchronous call).
    RegisterCallableArgs reg_args{};
    reg_args.active_callable_id = runtime.get_active_callable_id();
    reg_args.register_new = runtime.register_new_callable_id() ? 1 : 0;
    reg_args.dev_orch_so_addr = runtime.get_dev_orch_so_addr();
    reg_args.dev_orch_so_size = runtime.get_dev_orch_so_size();
    const char *func_name = runtime.get_device_orch_func_name();
    const char *config_name = runtime.get_device_orch_config_name();
    snprintf(reg_args.device_orch_func_name, sizeof(reg_args.device_orch_func_name), "%s", func_name ? func_name : "");
    snprintf(
        reg_args.device_orch_config_name, sizeof(reg_args.device_orch_config_name), "%s", config_name ? config_name : ""
    );
    return aicpu_register_callable_func_(&reg_args);
}

int DeviceRunner::run(Runtime &runtime, int block_dim, int launch_aicpu_num) {
    clear_cpu_sim_shared_storage();
    if (launch_aicpu_num < 1 || launch_aicpu_num > PLATFORM_MAX_AICPU_THREADS) {
        LOG_ERROR("launch_aicpu_num (%d) must be in range [1, %d]", launch_aicpu_num, PLATFORM_MAX_AICPU_THREADS);
        return -1;
    }

    // Validate block_dim. Sim has no stream resource query, so the static
    // platform capacity is the bound. block_dim == 0 is the CallConfig "auto"
    // sentinel.
    if (block_dim == 0) {
        block_dim = PLATFORM_MAX_BLOCKDIM;
        LOG_INFO_V0("block_dim auto-resolved to %d", block_dim);
    }
    if (block_dim < 1 || block_dim > PLATFORM_MAX_BLOCKDIM) {
        LOG_ERROR("block_dim (%d) must be in range [1, %d]", block_dim, PLATFORM_MAX_BLOCKDIM);
        return -1;
    }

    int rc = ensure_device_initialized();
    if (rc != 0) {
        LOG_ERROR("ensure_device_initialized failed: %d", rc);
        return rc;
    }

    // Lazy-allocate the 8-byte device_wall buffer on first run. Sim's "device
    // pointer" is a host malloc returned by allocate_tensor; the sim AICPU
    // thread writes through this pointer just like onboard's AICPU does.
    if (device_wall_dev_ptr_ == nullptr) {
        device_wall_dev_ptr_ = allocate_tensor(sizeof(uint64_t));
        if (device_wall_dev_ptr_ != nullptr) {
            kernel_args_.device_wall_data_base = reinterpret_cast<uint64_t>(device_wall_dev_ptr_);
            *static_cast<uint64_t *>(device_wall_dev_ptr_) = 0;
        }
    }

    block_dim_ = block_dim;
    int num_aicore = block_dim * cores_per_blockdim_;

    if (num_aicore > RUNTIME_MAX_WORKER) {
        LOG_ERROR("num_aicore (%d) exceeds RUNTIME_MAX_WORKER (%d)", num_aicore, RUNTIME_MAX_WORKER);
        return -1;
    }

    runtime.worker_count = num_aicore;
    worker_count_ = num_aicore;
    runtime.aicpu_thread_num = launch_aicpu_num;

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
    if (enable_dep_gen_) {
        SET_PROFILING_FLAG(enable_profiling_flag, PROFILING_FLAG_DEP_GEN);
    }
    if (enable_scope_stats_) {
        SET_PROFILING_FLAG(enable_profiling_flag, PROFILING_FLAG_SCOPE_STATS);
    }
    kernel_args_.enable_profiling_flag = enable_profiling_flag;

    for (int i = 0; i < num_aicore; i++) {
        runtime.workers[i].aicpu_ready = 0;
        runtime.workers[i].aicore_done = 0;
        runtime.workers[i].task = 0;
        runtime.workers[i].core_type = (i < num_aic) ? CoreType::AIC : CoreType::AIV;
    }

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

    last_runtime_ = &runtime;

    if (enable_l2_swimlane_) {
        rc = init_l2_swimlane(num_aicore, runtime.aicpu_thread_num, device_id_);
        if (rc != 0) {
            LOG_ERROR("init_l2_swimlane failed: %d", rc);
            return rc;
        }
        // Publish per-core core_type to the collector so the level=1 host
        // emit path can label lanes without an AICPU record. Sim already set
        // workers[i].core_type via the (i < num_aic) rule at line ~240.
        std::vector<CoreType> core_types(num_aicore);
        for (int i = 0; i < num_aicore; i++) {
            core_types[i] = runtime.workers[i].core_type;
        }
        l2_swimlane_collector_.set_core_types(core_types.data(), num_aicore);
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
            LOG_ERROR("init_pmu failed: %d", rc);
            return rc;
        }
    }

    if (enable_dep_gen_) {
        rc = init_dep_gen(launch_aicpu_num, device_id_);
        if (rc != 0) {
            LOG_ERROR("init_dep_gen failed: %d", rc);
            return rc;
        }
    }

    if (enable_scope_stats_) {
        rc = init_scope_stats(launch_aicpu_num);
        if (rc != 0) {
            LOG_ERROR("init_scope_stats failed: %d", rc);
            return rc;
        }
    }

    // On any exit from run() — success or early error — release the diagnostics
    // collectors' shared memory.
    auto perf_cleanup = RAIIScopeGuard([this]() {
        finalize_collectors();
    });

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

    LOG_INFO_V0("Allocated simulated registers: %d cores x 0x%x bytes", num_aicore, SIM_REG_BLOCK_SIZE);

    // Allocate simulated PMU register blocks. PMU MMIO is a separate address
    // region from the general AICore regs on hardware, so sim mirrors that with
    // its own backing memory; otherwise the AICPU PMU collector would early-out
    // on every core.
    size_t total_pmu_reg_size = num_aicore * SIM_REG_BLOCK_SIZE;
    void *pmu_reg_blocks = mem_alloc_.alloc(total_pmu_reg_size);
    if (pmu_reg_blocks == nullptr) {
        LOG_ERROR("Failed to allocate simulated PMU register memory (%zu bytes)", total_pmu_reg_size);
        return -1;
    }
    std::memset(pmu_reg_blocks, 0, total_pmu_reg_size);

    auto pmu_reg_blocks_cleanup = RAIIScopeGuard([this, pmu_reg_blocks]() {
        mem_alloc_.free(pmu_reg_blocks);
    });

    size_t pmu_regs_array_size = num_aicore * sizeof(uint64_t);
    uint64_t *pmu_regs_array = reinterpret_cast<uint64_t *>(mem_alloc_.alloc(pmu_regs_array_size));
    if (pmu_regs_array == nullptr) {
        LOG_ERROR("Failed to allocate PMU register address array");
        return -1;
    }
    for (int i = 0; i < num_aicore; i++) {
        pmu_regs_array[i] = reinterpret_cast<uint64_t>(static_cast<uint8_t *>(pmu_reg_blocks) + i * SIM_REG_BLOCK_SIZE);
    }
    kernel_args_.pmu_reg_addrs = reinterpret_cast<uint64_t>(pmu_regs_array);

    auto pmu_regs_array_cleanup = RAIIScopeGuard([this]() {
        if (kernel_args_.pmu_reg_addrs != 0) {
            mem_alloc_.free(reinterpret_cast<void *>(kernel_args_.pmu_reg_addrs));
            kernel_args_.pmu_reg_addrs = 0;
        }
    });

    LOG_INFO_V0("Allocated simulated PMU registers: %d cores x 0x%x bytes", num_aicore, SIM_REG_BLOCK_SIZE);

    const bool needs_orch_device_id = runtime.register_new_callable_id();
    if (aicpu_execute_func_ == nullptr || aicore_execute_func_ == nullptr || set_platform_regs_func_ == nullptr ||
        (needs_orch_device_id && set_orch_device_id_func_ == nullptr) || set_platform_dump_base_func_ == nullptr ||
        set_platform_phase_base_func_ == nullptr || set_dump_args_enabled_func_ == nullptr ||
        set_platform_pmu_base_func_ == nullptr || set_platform_pmu_reg_addrs_func_ == nullptr ||
        set_pmu_enabled_func_ == nullptr || set_platform_dep_gen_base_func_ == nullptr ||
        set_dep_gen_enabled_func_ == nullptr || set_scope_stats_enabled_func_ == nullptr ||
        set_platform_scope_stats_base_func_ == nullptr) {
        LOG_ERROR("Executor functions not loaded. Call ensure_binaries_loaded first.");
        return -1;
    }

    set_platform_regs_func_(kernel_args_.regs);
    if (set_orch_device_id_func_ != nullptr) {
        set_orch_device_id_func_(device_id_);
    }
    set_platform_dump_base_func_(kernel_args_.dump_data_base);
    set_dump_args_enabled_func_(enable_dump_tensor_);
    set_platform_l2_swimlane_base_func_(kernel_args_.l2_swimlane_data_base);
    set_platform_l2_swimlane_aicore_rotation_table_func_(kernel_args_.l2_swimlane_aicore_rotation_table);
    set_l2_swimlane_enabled_func_(enable_l2_swimlane_);
    set_platform_pmu_base_func_(kernel_args_.pmu_data_base);
    set_platform_pmu_reg_addrs_func_(kernel_args_.pmu_reg_addrs);
    set_pmu_enabled_func_(enable_pmu_);
    set_platform_dep_gen_base_func_(kernel_args_.dep_gen_data_base);
    set_dep_gen_enabled_func_(enable_dep_gen_);
    set_scope_stats_enabled_func_(enable_scope_stats_);
    set_platform_scope_stats_base_func_(kernel_args_.scope_stats_data_base);

    // Start collector mgmt + poll threads now, just before kernels launch.
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
    if (enable_dep_gen_) {
        dep_gen_collector_.start(thread_factory);
    }
    if (enable_scope_stats_) {
        scope_stats_collector_.start(thread_factory);
    }

    constexpr int over_launch = PLATFORM_MAX_AICPU_THREADS_JUST_FOR_LAUNCH;
    LOG_INFO_V0("Launching %d AICPU threads (logical=%d)", over_launch, launch_aicpu_num);
    std::vector<std::thread> aicpu_threads;
    aicpu_threads.reserve(over_launch);
    std::atomic<int> aicpu_rc{0};

    // Sim "device wall" capture: RunWall via host steady_clock through the
    // single-uint64 device_wall_data_base buffer (sim has no kernel.cpp wrapper
    // to stamp it). The finer preamble/so_load/graph_build/post_orch + orch/sched
    // phases are stamped by the AICPU `.so` (real sim get_sys_cnt_aicpu clock)
    // into a separate AicpuPhaseRecord buffer whose base is published into the SO
    // via the dlsym'd set_platform_phase_base — crossing the dlopen boundary the
    // same way set_platform_dump_base etc. do. Local buffer: sim runs in-process,
    // its lifetime spans the join below; reduced into device_phase_ns_ after.
    if (kernel_args_.device_wall_data_base != 0) {
        *reinterpret_cast<uint64_t *>(kernel_args_.device_wall_data_base) = 0;
    }
    constexpr int kPhaseThreads = PLATFORM_MAX_AICPU_THREADS_JUST_FOR_LAUNCH;
    std::vector<AicpuPhaseRecord> phase_buf(
        static_cast<size_t>(kPhaseThreads) * NUM_AICPU_PHASES, AicpuPhaseRecord{kPhaseUnset, 0}
    );
    set_platform_phase_base_func_(reinterpret_cast<uint64_t>(phase_buf.data()));
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

    LOG_INFO_V0("Launching %d AICore thread(s)", num_aicore);
    std::vector<std::thread> aicore_threads;
    for (int i = 0; i < num_aicore; i++) {
        CoreType core_type = runtime.workers[i].core_type;
        uint32_t physical_core_id = static_cast<uint32_t>(i);
        aicore_threads.push_back(create_thread([this, &runtime, i, core_type, physical_core_id]() {
            aicore_execute_func_(
                &runtime, i, core_type, physical_core_id, kernel_args_.regs, kernel_args_.enable_profiling_flag,
                kernel_args_.l2_swimlane_aicore_rotation_table
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

    // Snapshot the device_wall buffer into device_wall_ns_.
    device_wall_ns_ = 0;
    if (device_wall_dev_ptr_ != nullptr) {
        device_wall_ns_ = *static_cast<uint64_t *>(device_wall_dev_ptr_);
    }

    // Reduce the AICPU phase records (cycles → ns via the sim sys-cnt freq).
    // RunWall keeps the steady_clock device_wall above for a single source.
    // Per-phase start offsets (from the earliest sub-phase start) give the
    // device spans a device-domain `ts`, so the orch∪sched "Effective" window
    // is computable and the sub-phases nest correctly.
    uint64_t phase_start[NUM_AICPU_PHASES];
    uint64_t phase_cycles[NUM_AICPU_PHASES];
    reduce_aicpu_phase_windows(phase_buf.data(), kPhaseThreads, phase_start, phase_cycles);
    auto cyc_to_ns = [](uint64_t c) {
        return static_cast<uint64_t>(c * 1'000'000'000.0 / static_cast<double>(PLATFORM_PROF_SYS_CNT_FREQ));
    };
    uint64_t origin = kPhaseUnset;
    for (int p = static_cast<int>(AicpuPhase::Preamble); p < NUM_AICPU_PHASES; ++p) {
        if (phase_start[p] != kPhaseUnset && phase_start[p] < origin) origin = phase_start[p];
    }
    for (int p = 0; p < NUM_AICPU_PHASES; ++p) {
        device_phase_ns_[p] = cyc_to_ns(phase_cycles[p]);
        device_phase_start_ns_[p] = 0;
        if (p != static_cast<int>(AicpuPhase::RunWall) && phase_start[p] != kPhaseUnset && origin != kPhaseUnset &&
            phase_start[p] >= origin) {
            device_phase_start_ns_[p] = cyc_to_ns(phase_start[p] - origin);
        }
    }
    device_phase_ns_[static_cast<int>(AicpuPhase::RunWall)] = device_wall_ns_;

    int runtime_rc = aicpu_rc.load(std::memory_order_acquire);
    if (runtime_rc != 0) {
        LOG_ERROR("AICPU execution failed with rc=%d", runtime_rc);
        return runtime_rc;
    }

    // Tear down collectors. stop() joins mgmt then collector in the only safe
    // order (mgmt's final-drain pass into L2 has poll as its consumer).
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

    if (enable_dep_gen_) {
        dep_gen_collector_.stop();
        if (dep_gen_collector_.reconcile_counters()) {
            const auto &records = dep_gen_collector_.records();
            const std::string deps = make_deps_json_path(output_prefix_);
            int rc = dep_gen_replay_emit_deps_json(records.data(), records.size(), deps.c_str());
            if (rc != 0) {
                LOG_ERROR("dep_gen replay failed (%d) — deps.json not produced", rc);
            }
        }
    }

    if (enable_scope_stats_) {
        scope_stats_collector_.stop();
        scope_stats_collector_.reconcile_counters();
        scope_stats_collector_.write_jsonl(output_prefix_);
    }

    print_handshake_results();

    // Close AICore kernel .so now while the process is healthy. AICPU .so is
    // kept alive (load-once) so that g_aicpu_executor state survives across
    // runs for the orch-SO cache-hit path.
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

void DeviceRunner::unload_executor_binaries() {
    if (aicpu_so_handle_ != nullptr) {
        dlclose(aicpu_so_handle_);
        aicpu_so_handle_ = nullptr;
        aicpu_execute_func_ = nullptr;
        aicpu_register_callable_func_ = nullptr;
        set_platform_regs_func_ = nullptr;
        set_orch_device_id_func_ = nullptr;
        set_platform_dump_base_func_ = nullptr;
        set_dump_args_enabled_func_ = nullptr;
        set_platform_l2_swimlane_base_func_ = nullptr;
        set_platform_l2_swimlane_aicore_rotation_table_func_ = nullptr;
        set_l2_swimlane_enabled_func_ = nullptr;
        set_platform_pmu_base_func_ = nullptr;
        set_platform_pmu_reg_addrs_func_ = nullptr;
        set_pmu_enabled_func_ = nullptr;
        set_platform_dep_gen_base_func_ = nullptr;
        set_dep_gen_enabled_func_ = nullptr;
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

int DeviceRunner::finalize() {
    if (device_id_ == -1 && aicpu_so_handle_ == nullptr && aicore_so_handle_ == nullptr) {
        return 0;
    }

    // Cleanup performance profiling. Normally already done by run()'s
    // perf_cleanup guard; this is the backstop for the no-run-since-init case.
    finalize_collectors();

    release_callable_state();

    unload_executor_binaries();

    // Release the three per-Worker pooled arenas. Must precede mem_alloc_.finalize()
    // so the arenas free through the still-live allocator, not after it.
    gm_heap_arena_.release();
    gm_sm_arena_.release();
    runtime_arena_pool_.release();
    cached_gm_heap_size_ = 0;
    cached_gm_sm_size_ = 0;
    cached_runtime_arena_size_ = 0;

    // Free the 8-byte device_wall buffer (allocated lazily in run()) before
    // mem_alloc_.finalize().
    if (device_wall_dev_ptr_ != nullptr) {
        free_tensor(device_wall_dev_ptr_);
        device_wall_dev_ptr_ = nullptr;
    }

    mem_alloc_.finalize();
    clear_cpu_sim_shared_storage();

    device_id_ = -1;
    worker_count_ = 0;
    last_runtime_ = nullptr;

    LOG_INFO_V0("DeviceRunner(sim) finalized");
    return 0;
}

// =============================================================================
// Performance Profiling Implementation
// =============================================================================

int DeviceRunner::init_l2_swimlane(int num_aicore, int aicpu_thread_num, int device_id) {
    auto alloc_cb = [this](size_t size) -> void * {
        return mem_alloc_.alloc(size);
    };
    auto free_cb = [this](void *dev_ptr) -> int {
        return mem_alloc_.free(dev_ptr);
    };

    int rc = l2_swimlane_collector_.initialize(
        num_aicore, aicpu_thread_num, device_id, l2_swimlane_level_, alloc_cb, nullptr, free_cb, output_prefix_
    );
    if (rc != 0) {
        return rc;
    }

    kernel_args_.l2_swimlane_data_base =
        reinterpret_cast<uint64_t>(l2_swimlane_collector_.get_l2_swimlane_setup_device_ptr());
    kernel_args_.l2_swimlane_aicore_rotation_table =
        reinterpret_cast<uint64_t>(l2_swimlane_collector_.get_aicore_ring_addr_table_device_ptr());
    return 0;
}

int DeviceRunner::init_tensor_dump(Runtime &runtime, int device_id) {
    int num_dump_threads = runtime.aicpu_thread_num;

    auto alloc_cb = [this](size_t size) -> void * {
        return mem_alloc_.alloc(size);
    };
    auto free_cb = [this](void *dev_ptr) -> int {
        return mem_alloc_.free(dev_ptr);
    };

    int rc = dump_collector_.initialize(
        num_dump_threads, device_id, alloc_cb, nullptr, free_cb, output_prefix_, dump_tensor_level_
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
    auto alloc_cb = [this](size_t size) -> void * {
        return mem_alloc_.alloc(size);
    };
    auto free_cb = [this](void *dev_ptr) -> int {
        return mem_alloc_.free(dev_ptr);
    };

    int rc = pmu_collector_.init(num_cores, num_threads, csv_path, event_type, alloc_cb, nullptr, free_cb, -1);
    if (rc != 0) {
        return rc;
    }

    kernel_args_.pmu_data_base = reinterpret_cast<uint64_t>(pmu_collector_.get_pmu_shm_device_ptr());
    return 0;
}

int DeviceRunner::init_dep_gen(int num_threads, int /*device_id*/) {
    auto alloc_cb = [this](size_t size) -> void * {
        return mem_alloc_.alloc(size);
    };
    auto free_cb = [this](void *dev_ptr) -> int {
        return mem_alloc_.free(dev_ptr);
    };

    int rc = dep_gen_collector_.init(num_threads, alloc_cb, nullptr, free_cb, -1);
    if (rc != 0) {
        return rc;
    }

    kernel_args_.dep_gen_data_base = reinterpret_cast<uint64_t>(dep_gen_collector_.get_dep_gen_shm_device_ptr());
    return 0;
}

int DeviceRunner::init_scope_stats(int num_threads) {
    auto alloc_cb = [this](size_t size) -> void * {
        return mem_alloc_.alloc(size);
    };
    auto free_cb = [this](void *dev_ptr) -> int {
        return mem_alloc_.free(dev_ptr);
    };

    int rc = scope_stats_collector_.init(num_threads, alloc_cb, /*register_cb=*/nullptr, free_cb, /*device_id=*/-1);
    if (rc != 0) {
        return rc;
    }

    kernel_args_.scope_stats_data_base =
        reinterpret_cast<uint64_t>(scope_stats_collector_.get_scope_stats_shm_device_ptr());
    return 0;
}

void DeviceRunner::finalize_collectors() {
    auto free_cb = [this](void *dev_ptr) -> int {
        return mem_alloc_.free(dev_ptr);
    };

    if (l2_swimlane_collector_.is_initialized()) {
        l2_swimlane_collector_.finalize(nullptr, free_cb);
    }
    if (dump_collector_.is_initialized()) {
        dump_collector_.finalize(nullptr, free_cb);
    }
    if (pmu_collector_.is_initialized()) {
        pmu_collector_.finalize(nullptr, free_cb);
    }
    if (dep_gen_collector_.is_initialized()) {
        dep_gen_collector_.finalize(nullptr, free_cb);
    }
    if (scope_stats_collector_.is_initialized()) {
        scope_stats_collector_.finalize(nullptr, free_cb);
        kernel_args_.scope_stats_data_base = 0;
    }
}
