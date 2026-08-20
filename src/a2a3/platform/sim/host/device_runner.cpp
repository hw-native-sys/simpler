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

#include <chrono>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "aicpu/device_phase_aicpu.h"
#include "aicpu/platform_aicpu_affinity.h"
#include "call_config.h"
#include "callable_protocol.h"
#include "common/host_log_binding.h"
#include "common/memory_barrier.h"
#include "common/platform_config.h"
#include "common/unified_log.h"
#include "cpu_sim_context.h"
#include "host_log.h"
#include "host/host_phase_records_artifact.h"
#include "host/raii_scope_guard.h"
#include "host/runtime_timeout_config.h"
#include "runtime.h"

// dep_gen has two shapes, one per orchestration site, and each runtime provides
// the strong symbols for the one it uses:
//   - device orchestration (tensormap_and_ringbuffer): the AICPU writes a ring
//     of captured submits, the host collector drains it, and
//     `dep_gen_replay_emit_deps_json` (runtime/.../host/dep_gen_replay.cpp)
//     replays them into deps.json.
//   - host orchestration (host_build_graph): the graph is captured from the
//     orchestrator's own dependency path as it runs on the host, and
//     `dep_gen_host_graph_*` (runtime/.../host/dep_gen_host_graph.cpp) writes it
//     out directly — no ring, no collector, nothing to reconcile.
// A runtime links only its own half, so each half needs a weak fallback here.
// Hidden visibility keeps the stubs off the global symbol table so RTLD_GLOBAL
// can't let them shadow a strong symbol in cross-.so loads.
// LOG_DEBUG (not WARN): the runner picks the shape via
// `dep_gen_host_graph_active()`, so neither stub is reachable when dep_gen is on
// — they exist purely to keep the .so loadable.
extern "C" __attribute__((weak, visibility("hidden"))) int dep_gen_replay_emit_deps_json(
    const struct DepGenRecord * /*records*/, size_t /*num_records*/, const char * /*deps_json_path*/
) {
    LOG_DEBUG("dep_gen replay not implemented for this runtime — deps.json skipped");
    return -1;
}

extern "C" __attribute__((weak, visibility("hidden"))) bool dep_gen_host_graph_active() { return false; }
extern "C" __attribute__((weak, visibility("hidden"))) void dep_gen_host_graph_set_enabled(bool /*enable*/) {}
extern "C" __attribute__((weak, visibility("hidden"))) int dep_gen_host_graph_emit(const char * /*deps_json_path*/) {
    LOG_DEBUG("dep_gen host graph not implemented for this runtime — deps.json skipped");
    return -1;
}

struct DeviceRunner::ActiveRun {
    Runtime *runtime{nullptr};
    void *reg_blocks{nullptr};
    void *pmu_reg_blocks{nullptr};
    std::vector<std::thread> aicpu_threads;
    std::vector<std::thread> aicore_threads;
    DevicePhaseBufferStorage<PLATFORM_MAX_AICPU_THREADS_JUST_FOR_LAUNCH> phase_buf{};

    void join() noexcept {
        for (auto &thread : aicpu_threads) {
            if (thread.joinable()) thread.join();
        }
        for (auto &thread : aicore_threads) {
            if (thread.joinable()) thread.join();
        }
    }

    ~ActiveRun() { join(); }
};

DeviceRunner::DeviceRunner() = default;
DeviceRunner::~DeviceRunner() { finalize(); }

void DeviceRunner::cleanup_active_run() noexcept {
    if (active_run_ == nullptr) return;
    active_run_->join();

    if (kernel_args_.pmu_reg_addrs != 0) {
        mem_alloc_.free(reinterpret_cast<void *>(kernel_args_.pmu_reg_addrs));
        kernel_args_.pmu_reg_addrs = 0;
    }
    if (active_run_->pmu_reg_blocks != nullptr) {
        mem_alloc_.free(active_run_->pmu_reg_blocks);
        active_run_->pmu_reg_blocks = nullptr;
    }
    if (kernel_args_.regs != 0) {
        mem_alloc_.free(reinterpret_cast<void *>(kernel_args_.regs));
        kernel_args_.regs = 0;
    }
    if (active_run_->reg_blocks != nullptr) {
        mem_alloc_.free(active_run_->reg_blocks);
        active_run_->reg_blocks = nullptr;
    }
    finalize_collectors();
    active_run_.reset();
}

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
            *out = sym;
        };

        if (!load_sym("aicpu_execute", reinterpret_cast<void **>(&aicpu_execute_func_))) return -1;
        load_optional_sym("simpler_aicpu_register_callable", reinterpret_cast<void **>(&aicpu_register_callable_func_));
        if (!load_sym("set_platform_regs", reinterpret_cast<void **>(&set_platform_regs_func_))) return -1;
        load_optional_sym("set_orch_device_id", reinterpret_cast<void **>(&set_orch_device_id_func_));
        load_optional_sym("set_scheduler_timeout_ms", reinterpret_cast<void **>(&set_scheduler_timeout_ms_func_));
        if (set_scheduler_timeout_ms_func_ != nullptr) {
            // Per-device one-shot latch (mirrors the onboard InitArgs path):
            // honor SIMPLER_SCHEDULER_TIMEOUT_MS once at SO load, not per run. 0 ->
            // the scheduler keeps its compile-time default. Sim skips the
            // op/stream ordering check (validate_runtime_timeout_order is onboard).
            RuntimeTimeoutParseStatus sched_status;
            RuntimeTimeoutConfig sched_cfg =
                resolve_runtime_timeout_config(RuntimeTimeoutConfig{1, 1, 0}, &sched_status);
            set_scheduler_timeout_ms_func_(
                (sched_status.scheduler_env_set && sched_status.scheduler_valid) ? sched_cfg.scheduler_timeout_ms : 0
            );
        }
        if (!load_sym("set_platform_dump_base", reinterpret_cast<void **>(&set_platform_dump_base_func_))) return -1;
        if (!load_sym("set_platform_phase_base", reinterpret_cast<void **>(&set_platform_phase_base_func_))) return -1;
        if (!load_sym("set_dump_args_enabled", reinterpret_cast<void **>(&set_dump_args_enabled_func_))) return -1;
        if (!load_sym(
                "set_platform_chip_swimlane_base", reinterpret_cast<void **>(&set_platform_chip_swimlane_base_func_)
            ))
            return -1;
        if (!load_sym(
                "set_platform_chip_swimlane_aicore_rotation_table",
                reinterpret_cast<void **>(&set_platform_chip_swimlane_aicore_rotation_table_func_)
            ))
            return -1;
        if (!load_sym("set_chip_swimlane_enabled", reinterpret_cast<void **>(&set_chip_swimlane_enabled_func_)))
            return -1;
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

        // The AICPU sim SO owns its level flags because it is RTLD_LOCAL.
        // Forward the process-wide HostLogger threshold explicitly.
        using SetLogLevelFunc = void (*)(int);
        SetLogLevelFunc set_log_level_func = nullptr;
        if (!load_sym("set_log_level", reinterpret_cast<void **>(&set_log_level_func))) return -1;
        set_log_level_func(HostLogger::get_instance().level());
        using SetHostLogStateFunc = void (*)(SimplerHostLogState *);
        SetHostLogStateFunc set_host_log_state_func = nullptr;
        if (!load_sym("set_host_log_state", reinterpret_cast<void **>(&set_host_log_state_func))) return -1;
        set_host_log_state_func(HostLogger::get_instance().state());

        aicpu_so_loaded_ = true;
        LOG_INFO("DeviceRunner(sim): Loaded aicpu_execute from %s", aicpu_so_path_.c_str());
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

        const char *bind_log_error = nullptr;
        if (simpler::log::bind_loaded_host_log_state(
                aicore_so_handle_, HostLogger::get_instance().state(), &bind_log_error
            ) != 0) {
            LOG_ERROR(
                "AICore SO failed to bind host-log state: %s",
                bind_log_error != nullptr ? bind_log_error : "unknown error"
            );
            dlclose(aicore_so_handle_);
            aicore_so_handle_ = nullptr;
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

int DeviceRunner::invoke_device_register(const RegisterCallableArgs &reg_args) {
    if (aicpu_register_callable_func_ == nullptr || set_orch_device_id_func_ == nullptr) {
        LOG_ERROR("Register-callable functions not loaded. Call ensure_binaries_loaded first.");
        return -1;
    }
    set_orch_device_id_func_(device_id_);
    // The descriptor was assembled from CallableState by the base
    // launch_device_register; sim shares process memory so the name pointers
    // in reg_args stay valid for this synchronous call. The AICPU entry's C ABI
    // takes void* and only reads the args, so the const_cast is safe.
    return aicpu_register_callable_func_(const_cast<RegisterCallableArgs *>(&reg_args));
}

void DeviceRunner::set_dep_gen_enabled(bool enable) {
    enable_dep_gen_ = enable;
    // Arms host-side capture for a host-orch runtime (no-op weak stub for the
    // device-orch one). The c_api latches the CallConfig before bind, and the
    // orchestration entry resets the graph before recording it.
    dep_gen_host_graph_set_enabled(enable);
}

int DeviceRunner::prepare_execution(
    Runtime &runtime, const CallConfig &config, uint32_t pipeline_slot, const NativeRunIdentity &identity,
    std::unique_ptr<PreparedExecution> *prepared
) {
    if (prepared == nullptr || *prepared != nullptr) return -1;
    if (active_run_ != nullptr) {
        LOG_ERROR("prepare_execution called while another simulated run still owns execution state");
        return -1;
    }
    auto execution = std::make_unique<PreparedExecution>(identity, runtime, config, pipeline_slot);
    active_run_ = std::make_unique<ActiveRun>();
    active_run_->runtime = &runtime;
    run_completion_.reset(1);
    auto prepare_cleanup = RAIIScopeGuard([this]() {
        run_completion_.abandon();
        cleanup_active_run();
    });

    apply_call_config(config);
    const int block_dim = runtime.get_worker_count() / cores_per_blockdim_;
    int launch_aicpu_num = config.aicpu_thread_num;
    clear_cpu_sim_shared_storage();
    // Sim has no hardware topology to probe, so auto uses the architecture
    // default directly. Publish the effective count so AICPU init and DFX
    // setup match the launch gate.
    if (launch_aicpu_num == 0) launch_aicpu_num = PLATFORM_DEFAULT_AICPU_THREAD_NUM;
    runtime.set_aicpu_thread_num(launch_aicpu_num);
    if (block_dim < 1) {
        LOG_ERROR("prepare_execution computed block_dim < 1 from worker_count=%d", runtime.get_worker_count());
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

    int num_aicore = block_dim * cores_per_blockdim_;
    uint32_t enable_profiling_flag = SIMPLER_DFX_FLAG_NONE;
    if (enable_dump_args_) {
        SIMPLER_SET_DFX_FLAG(enable_profiling_flag, SIMPLER_DFX_FLAG_DUMP_ARGS);
    }
    if (enable_chip_swimlane_) {
        SIMPLER_SET_DFX_FLAG(enable_profiling_flag, SIMPLER_DFX_FLAG_CHIP_SWIMLANE);
    }
    if (enable_pmu_) {
        SIMPLER_SET_DFX_FLAG(enable_profiling_flag, SIMPLER_DFX_FLAG_PMU);
    }
    // The device flag drives the AICPU writer only; a host-orch runtime has no
    // device-side dep_gen to switch on.
    if (enable_dep_gen_ && !dep_gen_host_graph_active()) {
        SIMPLER_SET_DFX_FLAG(enable_profiling_flag, SIMPLER_DFX_FLAG_DEP_GEN);
    }
    if (enable_scope_stats_) {
        SIMPLER_SET_DFX_FLAG(enable_profiling_flag, SIMPLER_DFX_FLAG_SCOPE_STATS);
    }
    kernel_args_.enable_profiling_flag = enable_profiling_flag;

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

    if (enable_chip_swimlane_) {
        rc = init_chip_swimlane(num_aicore, runtime.get_aicpu_thread_num(), device_id_);
        if (rc != 0) {
            LOG_ERROR("init_chip_swimlane failed: %d", rc);
            return rc;
        }
        // Publish per-core core_type to the collector so the level=1 host
        // emit path can label lanes without an AICPU record. prepare_launch_shape
        // already typed workers[i].core_type (first block_dim cores are AIC).
        std::vector<CoreType> core_types(num_aicore);
        for (int i = 0; i < num_aicore; i++) {
            core_types[i] = runtime.get_workers()[i].core_type;
        }
        chip_swimlane_collector_.set_core_types(core_types.data(), num_aicore);
    }

    if (enable_dump_args_) {
        rc = init_args_dump(runtime, device_id_);
        if (rc != 0) {
            LOG_ERROR("init_args_dump failed: %d", rc);
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

    // A host-orch runtime already holds the graph in host memory; standing up
    // the device ring and its collector would allocate shared memory and a
    // drain thread for a stream that never produces a record.
    if (enable_dep_gen_ && !dep_gen_host_graph_active()) {
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

    size_t total_reg_size = num_aicore * SIM_REG_BLOCK_SIZE;
    active_run_->reg_blocks = mem_alloc_.alloc(total_reg_size);
    if (active_run_->reg_blocks == nullptr) {
        LOG_ERROR("Failed to allocate simulated register memory (%zu bytes)", total_reg_size);
        return -1;
    }
    std::memset(active_run_->reg_blocks, 0, total_reg_size);

    size_t regs_array_size = num_aicore * sizeof(uint64_t);
    uint64_t *regs_array = reinterpret_cast<uint64_t *>(mem_alloc_.alloc(regs_array_size));
    if (regs_array == nullptr) {
        LOG_ERROR("Failed to allocate register address array");
        return -1;
    }
    for (int i = 0; i < num_aicore; i++) {
        regs_array[i] =
            reinterpret_cast<uint64_t>(static_cast<uint8_t *>(active_run_->reg_blocks) + i * SIM_REG_BLOCK_SIZE);
    }
    kernel_args_.regs = reinterpret_cast<uint64_t>(regs_array);

    // Allocate simulated PMU register blocks. PMU MMIO is a separate address
    // region from the general AICore regs on hardware, so sim mirrors that with
    // its own backing memory; otherwise the AICPU PMU collector would early-out
    // on every core.
    size_t total_pmu_reg_size = num_aicore * SIM_REG_BLOCK_SIZE;
    active_run_->pmu_reg_blocks = mem_alloc_.alloc(total_pmu_reg_size);
    if (active_run_->pmu_reg_blocks == nullptr) {
        LOG_ERROR("Failed to allocate simulated PMU register memory (%zu bytes)", total_pmu_reg_size);
        return -1;
    }
    std::memset(active_run_->pmu_reg_blocks, 0, total_pmu_reg_size);

    size_t pmu_regs_array_size = num_aicore * sizeof(uint64_t);
    uint64_t *pmu_regs_array = reinterpret_cast<uint64_t *>(mem_alloc_.alloc(pmu_regs_array_size));
    if (pmu_regs_array == nullptr) {
        LOG_ERROR("Failed to allocate PMU register address array");
        return -1;
    }
    for (int i = 0; i < num_aicore; i++) {
        pmu_regs_array[i] =
            reinterpret_cast<uint64_t>(static_cast<uint8_t *>(active_run_->pmu_reg_blocks) + i * SIM_REG_BLOCK_SIZE);
    }
    kernel_args_.pmu_reg_addrs = reinterpret_cast<uint64_t>(pmu_regs_array);

    if (aicpu_execute_func_ == nullptr || aicore_execute_func_ == nullptr || set_platform_regs_func_ == nullptr ||
        set_platform_dump_base_func_ == nullptr || set_platform_phase_base_func_ == nullptr ||
        set_dump_args_enabled_func_ == nullptr || set_platform_pmu_base_func_ == nullptr ||
        set_platform_pmu_reg_addrs_func_ == nullptr || set_pmu_enabled_func_ == nullptr ||
        set_platform_dep_gen_base_func_ == nullptr || set_dep_gen_enabled_func_ == nullptr ||
        set_scope_stats_enabled_func_ == nullptr || set_platform_scope_stats_base_func_ == nullptr ||
        set_platform_chip_swimlane_base_func_ == nullptr ||
        set_platform_chip_swimlane_aicore_rotation_table_func_ == nullptr ||
        set_chip_swimlane_enabled_func_ == nullptr) {
        LOG_ERROR("Executor functions not loaded. Call ensure_binaries_loaded first.");
        return -1;
    }

    constexpr int over_launch = PLATFORM_MAX_AICPU_THREADS_JUST_FOR_LAUNCH;
    active_run_->aicpu_threads.reserve(over_launch);
    active_run_->aicore_threads.reserve(num_aicore);
    execution->num_aicore = num_aicore;
    execution->launch_aicpu_num = launch_aicpu_num;
    prepare_cleanup.dismiss();
    *prepared = std::move(execution);
    return 0;
}

SimDeviceRunnerBase::LaunchOutcome
DeviceRunner::launch_execution(std::unique_ptr<PreparedExecution> prepared, LaunchPermit permit) {
    LaunchOutcome outcome;
    if (prepared == nullptr) return outcome;
    Runtime &runtime = *prepared->runtime;
    const int num_aicore = prepared->num_aicore;
    const int launch_aicpu_num = prepared->launch_aicpu_num;
    constexpr int over_launch = PLATFORM_MAX_AICPU_THREADS_JUST_FOR_LAUNCH;
    ActiveRun *run = active_run_.get();
    if (run == nullptr) {
        outcome.prepared = std::move(prepared);
        return outcome;
    }

    std::chrono::steady_clock::time_point sim_t0;

    LaunchTransactionResult result = exact_launch_transaction(
        prepared->identity, std::move(permit),
        [&]() {
            // Arming precedes any simulated-core thread, so its failures — including
            // a thread-spawn or allocation throw — are reported as an rc and leave
            // the run safely rollback-able.
            try {
                set_platform_regs_func_(kernel_args_.regs);
                if (set_orch_device_id_func_ != nullptr) set_orch_device_id_func_(device_id_);
                set_platform_dump_base_func_(kernel_args_.dump_data_base);
                set_dump_args_enabled_func_(enable_dump_args_);
                set_platform_chip_swimlane_base_func_(kernel_args_.chip_swimlane_data_base);
                set_platform_chip_swimlane_aicore_rotation_table_func_(
                    kernel_args_.chip_swimlane_aicore_rotation_table
                );
                set_chip_swimlane_enabled_func_(enable_chip_swimlane_);
                set_platform_pmu_base_func_(kernel_args_.pmu_data_base);
                set_platform_pmu_reg_addrs_func_(kernel_args_.pmu_reg_addrs);
                set_pmu_enabled_func_(enable_pmu_);
                set_platform_dep_gen_base_func_(kernel_args_.dep_gen_data_base);
                set_dep_gen_enabled_func_(enable_dep_gen_ && !dep_gen_host_graph_active());
                set_scope_stats_enabled_func_(enable_scope_stats_);
                set_platform_scope_stats_base_func_(kernel_args_.scope_stats_data_base);

                auto thread_factory = [this](std::function<void()> fn) {
                    return create_thread(std::move(fn));
                };
                if (enable_chip_swimlane_) chip_swimlane_collector_.start(thread_factory);
                if (enable_dump_args_) dump_collector_.start(thread_factory);
                if (enable_pmu_) pmu_collector_.start(thread_factory);
                if (enable_dep_gen_ && !dep_gen_host_graph_active()) dep_gen_collector_.start(thread_factory);
                if (enable_scope_stats_) scope_stats_collector_.start(thread_factory);

                if (kernel_args_.device_wall_data_base != 0) {
                    *reinterpret_cast<uint64_t *>(kernel_args_.device_wall_data_base) = 0;
                }
                reset_device_phase_buffer(&run->phase_buf, over_launch);
                set_platform_phase_base_func_(reinterpret_cast<uint64_t>(&run->phase_buf));
                sim_t0 = std::chrono::steady_clock::now();
                run_completion_.reset(static_cast<size_t>(over_launch) + static_cast<size_t>(num_aicore));
            } catch (...) {
                LOG_ERROR("launch_execution: arming failed before any simulated core started");
                return -1;
            }

            LOG_INFO("Launching %d AICore thread(s)", num_aicore);
            for (int i = 0; i < num_aicore; i++) {
                CoreType core_type = runtime.get_workers()[i].core_type;
                uint32_t physical_core_id = static_cast<uint32_t>(i);
                run->aicore_threads.push_back(create_thread([this, run, i, core_type, physical_core_id]() {
                    aicore_execute_func_(
                        run->runtime, i, core_type, physical_core_id, kernel_args_.regs,
                        kernel_args_.enable_profiling_flag, kernel_args_.chip_swimlane_aicore_rotation_table
                    );
                    run_completion_.task_finished();
                }));
            }
            return 0;
        },
        [&]() {
            LOG_INFO("Launching %d AICPU threads (logical=%d)", over_launch, launch_aicpu_num);
            for (int i = 0; i < over_launch; i++) {
                run->aicpu_threads.push_back(create_thread([this, run, launch_aicpu_num, over_launch, sim_t0]() {
                    if (!platform_aicpu_affinity_gate(launch_aicpu_num, over_launch)) {
                        run_completion_.task_finished();
                        return;
                    }
                    int rc = aicpu_execute_func_(run->runtime);
                    if (kernel_args_.device_wall_data_base != 0) {
                        const auto t1 = std::chrono::steady_clock::now();
                        *reinterpret_cast<uint64_t *>(kernel_args_.device_wall_data_base) = static_cast<uint64_t>(
                            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - sim_t0).count()
                        );
                    }
                    run_completion_.task_finished(rc);
                }));
            }
            return 0;
        }
    );
    if (result.poisoned()) poison_launch();
    outcome.rc = result.rc;
    outcome.progress = result.progress;
    outcome.receipt = std::move(result.receipt);
    if (result.progress == LaunchProgress::NotStarted) {
        outcome.prepared = std::move(prepared);
    } else {
        outcome.active = std::make_unique<ActiveExecution>(std::move(prepared), result.progress);
    }
    return outcome;
}

int DeviceRunner::poll_execution(const ActiveExecution &) { return run_completion_.poll(); }

int DeviceRunner::drain_execution(ActiveExecution &) {
    if (active_run_ == nullptr) {
        LOG_ERROR("drain_execution called without a launched simulated run");
        return -1;
    }
    auto run_cleanup = RAIIScopeGuard([this]() {
        cleanup_active_run();
    });
    active_run_->join();
    LOG_INFO("All threads completed");

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
    constexpr int kPhaseThreads = PLATFORM_MAX_AICPU_THREADS_JUST_FOR_LAUNCH;
    const void *phase_buffer = &active_run_->phase_buf;
    reduce_aicpu_phase_windows(device_phase_records(phase_buffer), kPhaseThreads, phase_start, phase_cycles);
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

    std::fill_n(task_slot_dispatch_ns_, NUM_TASK_TIMING_SLOTS, uint64_t{0});
    std::fill_n(task_slot_finish_ns_, NUM_TASK_TIMING_SLOTS, uint64_t{0});
    if (device_phase_buffer_header(phase_buffer)->task_timing_tail_used != 0) {
        const TaskTimingRecord *tail = task_timing_tail_records(phase_buffer, kPhaseThreads);
        resolve_task_timing_slots_ns(
            tail, kPhaseThreads, origin, cyc_to_ns, task_slot_dispatch_ns_, task_slot_finish_ns_
        );
    }

    int runtime_rc = run_completion_.first_error();
    if (runtime_rc != 0) {
        LOG_ERROR("AICPU execution failed with rc=%d", runtime_rc);
        finish_clock_correlation_session(false);
        return runtime_rc;
    }

    // Tear down collectors. stop() joins mgmt then collector in the only safe
    // order (mgmt's final-drain pass into L2 has poll as its consumer).
    finish_clock_correlation_session(true);
    if (enable_chip_swimlane_) {
        chip_swimlane_collector_.stop();
        chip_swimlane_collector_.read_phase_header_metadata();
        chip_swimlane_collector_.reconcile_counters();
        publish_host_phase_records_to_swimlane();
        chip_swimlane_collector_.export_swimlane_json();
    }

    if (enable_dump_args_) {
        dump_collector_.stop();
        dump_collector_.reconcile_counters();
        dump_collector_.export_dump_files();
    }

    if (enable_pmu_) {
        pmu_collector_.stop();
        pmu_collector_.reconcile_counters();
    }

    // Host-orch emits the graph its orchestration built on this same thread;
    // device-orch stops the collector, reconciles the ring, and replays.
    if (enable_dep_gen_) {
        const std::string deps = make_deps_json_path(output_prefix_);
        if (dep_gen_host_graph_active()) {
            int rc = dep_gen_host_graph_emit(deps.c_str());
            if (rc != 0) {
                LOG_ERROR("dep_gen host graph emit failed (%d) — deps.json not produced", rc);
            }
        } else {
            dep_gen_collector_.stop();
            if (dep_gen_collector_.reconcile_counters()) {
                const auto &records = dep_gen_collector_.records();
                int rc = dep_gen_replay_emit_deps_json(records.data(), records.size(), deps.c_str());
                if (rc != 0) {
                    LOG_ERROR("dep_gen replay failed (%d) — deps.json not produced", rc);
                }
            }
        }
    }

    // Per-event view of the prepare path. Keyed on output_prefix_ rather than a
    // flag because it is non-empty exactly when this run produces diagnostic
    // artifacts, and on the store having finished a pass, which is what a
    // host-orchestrating runtime leaves behind.
    if (!output_prefix_.empty() && host_phase_records_.finished()) {
        (void)host_phase_records_.write_records_jsonl(make_host_phase_records_path(output_prefix_));
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

void DeviceRunner::abandon_prepared_execution(PreparedExecution &) noexcept {
    run_completion_.abandon();
    cleanup_active_run();
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
        set_platform_chip_swimlane_base_func_ = nullptr;
        set_platform_chip_swimlane_aicore_rotation_table_func_ = nullptr;
        set_chip_swimlane_enabled_func_ = nullptr;
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
    cleanup_active_run();
    if (device_id_ == -1 && aicpu_so_handle_ == nullptr && aicore_so_handle_ == nullptr) {
        return 0;
    }

    // cleanup_active_run() normally stops active collectors; this is the
    // backstop for the initialized-but-never-enqueued case.
    finalize_collectors();

    release_callable_state();

    unload_executor_binaries();

    // Release the arena-bank regions. Must precede mem_alloc_.finalize()
    // so the arenas free through the still-live allocator, not after it.
    for (auto &bank : arena_banks_) {
        bank->gm_heap.release();
        bank->gm_sm.release();
        bank->runtime_pool.release();
        bank->graph_block.release();
    }
    clear_temporary_buffer();
    for (auto &bank : arena_banks_) {
        bank->cached_gm_heap_size = 0;
        bank->cached_gm_sm_size = 0;
        bank->cached_runtime_arena_size = 0;
        bank->cached_graph_block_size = 0;
    }
    prebuilt_runtime_arena_cache_valid_ = false;
    prebuilt_runtime_arena_cache_key_.clear();
    prebuilt_runtime_arena_cache_gm_heap_base_ = nullptr;
    prebuilt_runtime_arena_cache_sm_base_ = nullptr;
    prebuilt_runtime_arena_cache_runtime_arena_base_ = nullptr;
    prebuilt_runtime_arena_cache_image_.clear();

    // Free the 8-byte device_wall buffer (allocated lazily in prepare_execution()) before
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

    return 0;
}

// =============================================================================
// Performance Profiling Implementation
// =============================================================================

int DeviceRunner::init_chip_swimlane(int num_aicore, int aicpu_thread_num, int device_id) {
    auto alloc_cb = [this](size_t size) -> void * {
        return mem_alloc_.alloc(size);
    };
    auto free_cb = [this](void *dev_ptr) -> int {
        return mem_alloc_.free(dev_ptr);
    };

    int rc = chip_swimlane_collector_.initialize(
        num_aicore, aicpu_thread_num, device_id, chip_swimlane_level_, alloc_cb, nullptr, free_cb, output_prefix_
    );
    if (rc != 0) {
        return rc;
    }

    kernel_args_.chip_swimlane_data_base =
        reinterpret_cast<uint64_t>(chip_swimlane_collector_.get_chip_swimlane_setup_device_ptr());
    kernel_args_.chip_swimlane_aicore_rotation_table =
        reinterpret_cast<uint64_t>(chip_swimlane_collector_.get_aicore_ring_addr_table_device_ptr());
    return 0;
}

int DeviceRunner::init_args_dump(Runtime &runtime, int device_id) {
    int num_dump_threads = runtime.get_aicpu_thread_num();

    auto alloc_cb = [this](size_t size) -> void * {
        return mem_alloc_.alloc(size);
    };
    auto free_cb = [this](void *dev_ptr) -> int {
        return mem_alloc_.free(dev_ptr);
    };

    int rc = dump_collector_.initialize(
        num_dump_threads, device_id, alloc_cb, nullptr, free_cb, output_prefix_, dump_args_level_
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

    if (chip_swimlane_collector_.is_initialized()) {
        chip_swimlane_collector_.finalize(nullptr, free_cb);
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
