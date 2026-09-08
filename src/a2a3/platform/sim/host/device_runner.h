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
 * a2a3 sim DeviceRunner — thread-based simulation of the Ascend AICPU/AICore
 * execution model. The shared base (`SimDeviceRunnerBase`) hosts the arena /
 * tensor-copy / callable-registry / chip-callable-buffer pool. This subclass
 * adds the a2a3-specific dlsym'd function-pointer table, dep_gen collector +
 * gating, and the enqueue/poll/drain sequence wired to a2a3's aicore_execute
 * signature.
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "common/core_type.h"
#include "device_runner_base.h"
#include "host/dep_gen_collector.h"
#include "sim_run_completion.h"

class DeviceRunner : public SimDeviceRunnerBase {
public:
    DeviceRunner();
    ~DeviceRunner() override;

    int prepare_execution(
        Runtime &runtime, const CallConfig &config, uint32_t pipeline_slot, const NativeRunIdentity &identity,
        std::unique_ptr<PreparedExecution> *prepared
    ) override;
    LaunchOutcome launch_execution(std::unique_ptr<PreparedExecution> prepared, LaunchPermit permit) override;
    void abandon_prepared_execution(PreparedExecution &prepared) noexcept override;
    int poll_execution(const ActiveExecution &active) override;
    int drain_execution(ActiveExecution &active) override;
    int finalize() override;
    // Also arms the loaded runtime's host-side graph capture, which a host-orch
    // runtime uses instead of the device collector. Defined in the .cpp so this
    // header stays free of the runtime-provided capture symbols.
    void set_dep_gen_enabled(bool enable) override;

private:
    struct ActiveRun;

    int ensure_binaries_loaded() override;
    int invoke_device_register(const RegisterCallableArgs &reg_args) override;
    void unload_executor_binaries();
    void cleanup_active_run() noexcept;

    int init_chip_swimlane(
        int num_aicore, int aicpu_thread_num, int device_id, const std::string &output_prefix,
        ChipSwimlaneLevel chip_swimlane_level
    );
    int
    init_args_dump(Runtime &runtime, int device_id, const std::string &output_prefix, DumpArgsLevel dump_args_level);
    int init_pmu(int num_cores, int num_threads, const std::string &csv_path, PmuEventType event_type, int device_id);
    int init_dep_gen(int num_threads, int device_id);
    int init_scope_stats(int num_threads);

    // Per-run collector teardown: releases shared memory back to mem_alloc_.
    // Idempotent. Mirrors the onboard helper.
    void finalize_collectors();

    // a2a3 sim's dlsym'd function-pointer table. Loaded once via
    // ensure_binaries_loaded(), nulled on unload_executor_binaries().
    int (*aicpu_execute_func_)(Runtime *){nullptr};
    // The runtime exports simpler_aicpu_register_callable(void*) directly (TMARB
    // only; hbg does not export it). Optional dlsym: null on the hbg SO.
    int (*aicpu_register_callable_func_)(void *){nullptr};
    void (*aicore_execute_func_)(Runtime *, int, CoreType, uint32_t, uint64_t, uint32_t, uint64_t){nullptr};
    void (*set_platform_regs_func_)(uint64_t){nullptr};
    void (*set_orch_device_id_func_)(int){nullptr};
    void (*set_scheduler_timeout_ms_func_)(int){nullptr};
    void (*set_platform_dump_base_func_)(uint64_t){nullptr};
    void (*set_platform_phase_base_func_)(uint64_t){nullptr};
    void (*set_dump_args_enabled_func_)(bool){nullptr};
    void (*set_platform_chip_swimlane_base_func_)(uint64_t){nullptr};
    void (*set_platform_chip_swimlane_aicore_rotation_table_func_)(uint64_t){nullptr};
    void (*set_chip_swimlane_enabled_func_)(bool){nullptr};
    void (*set_platform_pmu_base_func_)(uint64_t){nullptr};
    void (*set_platform_pmu_reg_addrs_func_)(uint64_t){nullptr};
    void (*set_pmu_enabled_func_)(bool){nullptr};
    void (*set_platform_dep_gen_base_func_)(uint64_t){nullptr};
    void (*set_dep_gen_enabled_func_)(bool){nullptr};
    void (*set_scope_stats_enabled_func_)(bool){nullptr};
    void (*set_platform_scope_stats_base_func_)(uint64_t){nullptr};

    // dep_gen collector — captures orchestrator submit_task inputs for offline replay.
    // a2a3-only; a5 has no dep_gen.
    DepGenCollector dep_gen_collector_;
    bool enable_dep_gen_{false};
    std::unique_ptr<ActiveRun> active_run_;
    simpler::common::sim_host::SimRunCompletion run_completion_;
};
