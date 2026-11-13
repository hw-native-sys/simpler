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
#pragma once
#include "wse_host_runner.h"
#include "sim_run_completion.h"
#include "wse_launch.h"
#include "sram_storage.h"

class DeviceRunner : public WseHostRunner {
public:
    DeviceRunner();
    void *allocate_tensor(size_t bytes) override;
    void *allocate_caller_storage(size_t bytes) override;
    void *allocate_graph_storage(size_t bytes) override;
    void free_graph_storage(void *ptr) override;
    size_t committed_device_memory() const override;
    void *acquire_child_memory_host_view(void *ptr, size_t bytes) override;
    int device_memset(void *ptr, int value, size_t bytes) override;
    void free_tensor(void *ptr) override;
    int copy_to_device(void *ptr, const void *host, size_t bytes) override;
    int copy_from_device(void *host, const void *ptr, size_t bytes) override;
    ~DeviceRunner() override;
    const char *cpu_sim_target_family() const override { return "wsesim"; }
    int prepare_launch_shape(Runtime &, const CallConfig &) override;
    int prepare_execution(
        Runtime &, const CallConfig &, uint32_t, const NativeRunIdentity &, std::unique_ptr<PreparedExecution> *
    ) override;
    void abandon_prepared_execution(PreparedExecution &) noexcept override;
    int poll_execution(const ActiveExecution &) override;
    DrainOutcome drain_execution(ActiveExecution &) override;
    int finalize() override;

private:
    WseLaunchTransactionResult launch_execution_impl(PreparedExecution &, WseLaunchPermit) override;
    int ensure_binaries_loaded() override;
    void cleanup_run() noexcept;
    void unload_executor();
    wse::SramStorage sram_;
    MemoryAllocator host_graph_storage_;
    struct ActiveRun;
    std::unique_ptr<ActiveRun> active_run_;
    simpler::common::sim_host::SimRunCompletion completion_;
    decltype(&wse_create_run) create_run_{nullptr};
    decltype(&wse_destroy_run) destroy_run_{nullptr};
    decltype(&wse_execute_pe) execute_pe_{nullptr};
    decltype(&wse_run_events) run_events_{nullptr};
    decltype(&wse_completed_tasks) completed_tasks_{nullptr};
    decltype(&wse_runtime_used) runtime_used_{nullptr};
    decltype(&wse_memory_sections) memory_sections_{nullptr};
    decltype(&wse_control_address) control_address_{nullptr};
};
