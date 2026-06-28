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
 * a5 sim DeviceRunner — thread-based simulation of the Ascend AICPU/AICore
 * execution model. The shared base (`SimDeviceRunnerBase`) hosts the arena /
 * tensor-copy / callable-registry / chip-callable-buffer pool. This subclass
 * adds the a5-specific dlsym'd function-pointer table (different aicore_execute
 * signature than a2a3 — extra aicore_pmu_ring_addrs arg) and the run() /
 * init_* / ensure_binaries_loaded sequence wired to a5's contract.
 */

#ifndef SRC_A5_PLATFORM_SIM_HOST_DEVICE_RUNNER_H_
#define SRC_A5_PLATFORM_SIM_HOST_DEVICE_RUNNER_H_

#include <cstdint>
#include <string>

#include "common/core_type.h"
#include "device_runner_base.h"
#include "host/dep_gen_collector.h"

class DeviceRunner : public SimDeviceRunnerBase {
public:
    DeviceRunner() = default;
    ~DeviceRunner() override;

    int run(Runtime &runtime, int block_dim, int launch_aicpu_num = 1) override;
    int finalize() override;
    // a5 dep_gen enablement setter, overriding the base no-op (the c_api
    // unconditionally calls it).
    void set_dep_gen_enabled(bool enable) override { enable_dep_gen_ = enable; }

private:
    int ensure_binaries_loaded() override;
    int invoke_aicpu_prewarm(Runtime &runtime) override;
    void unload_executor_binaries();

    int init_l2_swimlane(int num_aicore, int aicpu_thread_num, int device_id);
    int init_tensor_dump(Runtime &runtime, int device_id);
    int init_pmu(int num_cores, int num_threads, const std::string &csv_path, PmuEventType event_type, int device_id);
    int init_scope_stats(int num_threads);
    int init_dep_gen(int num_threads, int device_id);

    // Per-run collector teardown: stops mgmt + poll threads on every collector
    // whose init succeeded. Idempotent. a5 stops only (does not finalize); the
    // shm release happens later in finalize().
    void stop_collectors();

    // a5 sim's dlsym'd function-pointer table. Loaded once via
    // ensure_binaries_loaded(), nulled on unload_executor_binaries().
    int (*aicpu_execute_func_)(Runtime *){nullptr};
    int (*aicpu_prewarm_func_)(Runtime *){nullptr};
    void (*aicore_execute_func_)(Runtime *, int, CoreType, uint32_t, uint64_t, uint32_t, uint64_t, uint64_t){nullptr};
    void (*set_platform_regs_func_)(uint64_t){nullptr};
    void (*set_orch_device_id_func_)(int){nullptr};
    void (*set_platform_dump_base_func_)(uint64_t){nullptr};
    void (*set_platform_phase_base_func_)(uint64_t){nullptr};
    void (*set_platform_pmu_base_func_)(uint64_t){nullptr};
    void (*set_dump_args_enabled_func_)(bool){nullptr};
    void (*set_platform_l2_swimlane_base_func_)(uint64_t){nullptr};
    void (*set_l2_swimlane_enabled_func_)(bool){nullptr};
    void (*set_pmu_enabled_func_)(bool){nullptr};
    void (*set_platform_dep_gen_base_func_)(uint64_t){nullptr};
    void (*set_dep_gen_enabled_func_)(bool){nullptr};
    void (*set_scope_stats_enabled_func_)(bool){nullptr};
    void (*set_platform_scope_stats_base_func_)(uint64_t){nullptr};

    // dep_gen collector — captures orchestrator submit_task inputs for offline replay.
    DepGenCollector dep_gen_collector_;
    bool enable_dep_gen_{false};
};

#endif  // SRC_A5_PLATFORM_SIM_HOST_DEVICE_RUNNER_H_
