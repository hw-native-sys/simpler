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
 * From the level above it looks like an IWorker; internally it contains the
 * full scheduling engine (TensorMap, Ring, Scope, Orchestrator, Scheduler)
 * and a set of sub-IWorkers it dispatches to.
 *
 * Public surface:
 *   - add_worker(type, IWorker*)  — register sub-workers (before init)
 *   - init() / close()             — lifecycle
 *   - get_orchestrator()           — accessor used by Worker::run / Python facade
 *                                    (scope_begin / drain / scope_end live on the
 *                                     Orchestrator, not here)
 *   - run(payload)                 — IWorker entry (placeholder for L4+ recursion)
 *
 * Worker holds no submit / scope / drain / active-task bookkeeping — those
 * concepts belong to Orchestrator.
 */

#pragma once

#include <cstdint>
#include <memory>

#include "dist_orchestrator.h"
#include "dist_ring.h"
#include "dist_scheduler.h"
#include "dist_scope.h"
#include "dist_tensormap.h"
#include "dist_types.h"
#include "dist_worker_manager.h"

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

    // Accessor: the Orchestrator handle used by the user's orch fn. Valid
    // only between init() and close().
    DistOrchestrator &get_orchestrator() { return orchestrator_; }

    // IWorker — used when this DistWorker is itself a sub-worker of L4+.
    // Placeholder for recursive composition; filled in by plan step F.
    void run(const WorkerPayload &payload) override;

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
    DistWorkerManager manager_;
};
