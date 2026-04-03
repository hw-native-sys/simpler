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
 * DistWorker — top-level distributed worker node.
 *
 * DistWorker is the implementation of one level in the hierarchy (L3, L4, …).
 * From the level above it looks like an IWorker; internally it contains the full
 * scheduling engine (TensorMap, Ring, Scope, Orchestrator, Scheduler) and a set
 * of sub-IWorkers it dispatches to.
 *
 * Usage (L3 host worker, instantiated from Python via nanobind):
 *
 *   DistWorker dw(level=3);
 *   dw.add_worker(WorkerType::CHIP, chip_worker_ptr);
 *   dw.add_worker(WorkerType::SUB,  sub_worker_ptr);
 *   dw.init();
 *
 *   // Orchestrator side (main thread):
 *   auto result = dw.submit(CHIP, payload, inputs, outputs);
 *   dw.scope_begin();
 *   dw.submit(...);
 *   dw.scope_end();
 *   dw.execute();   // blocks until all submitted tasks complete
 *
 *   // When used as an IWorker by a higher-level DistWorker (L4+):
 *   parent.add_worker(WorkerType::DIST, &dw);
 *   // parent scheduler calls dw.dispatch() / dw.poll()
 */

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "dist_orchestrator.h"
#include "dist_ring.h"
#include "dist_scheduler.h"
#include "dist_scope.h"
#include "dist_tensormap.h"
#include "dist_types.h"

class DistWorker : public IWorker {
public:
    explicit DistWorker(int32_t level);
    ~DistWorker() override;

    DistWorker(const DistWorker &) = delete;
    DistWorker &operator=(const DistWorker &) = delete;

    // Register sub-workers before calling init().
    void add_worker(WorkerType type, IWorker *worker);

    // Initialise the engine and start the Scheduler thread.
    void init();

    // Shut down the Scheduler thread and release resources.
    void close();

    // Submit a task (Orch thread only).
    DistSubmitResult submit(
        WorkerType worker_type, const WorkerPayload &base_payload, const std::vector<DistInputSpec> &inputs,
        const std::vector<DistOutputSpec> &outputs
    );

    // Submit a group task: N args → N workers, 1 DAG node.
    DistSubmitResult submit_group(
        WorkerType worker_type, const WorkerPayload &base_payload, const std::vector<const void *> &args_list,
        const std::vector<DistInputSpec> &inputs, const std::vector<DistOutputSpec> &outputs
    );

    void scope_begin();
    void scope_end();

    // Block until all submitted tasks have reached CONSUMED.
    // Called at the end of execute() or from the parent Scheduler.
    void drain();

    // ------------------------------------------------------------------
    // IWorker — used when this DistWorker is itself a sub-worker of L4+.
    // run() executes the stored HostTask orch + drains (placeholder for now).
    // ------------------------------------------------------------------
    void run(const WorkerPayload &payload) override;

    int32_t level() const { return level_; }
    bool idle() const { return active_tasks_.load(std::memory_order_acquire) == 0; }

private:
    int32_t level_;
    bool initialized_{false};

    // --- Scheduling engine components ---
    std::unique_ptr<DistTaskSlotState[]> slots_;
    DistTensorMap tensormap_;
    DistRing ring_;
    DistScope scope_;
    DistReadyQueue ready_queue_;
    DistOrchestrator orchestrator_;
    DistScheduler scheduler_;

    std::vector<IWorker *> chip_workers_;
    std::vector<IWorker *> sub_workers_;

    // --- Drain support ---
    std::mutex drain_mu_;
    std::condition_variable drain_cv_;
    std::atomic<int32_t> active_tasks_{0};

    void on_consumed(DistTaskSlot slot);
};
