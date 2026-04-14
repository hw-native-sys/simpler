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

#include "dist_worker.h"

#include <stdexcept>

DistWorker::DistWorker(int32_t level) :
    level_(level) {
    slots_ = std::make_unique<DistTaskSlotState[]>(DIST_TASK_WINDOW_SIZE);
}

DistWorker::~DistWorker() {
    if (initialized_) close();
}

void DistWorker::add_worker(WorkerType type, IWorker *worker) {
    if (initialized_) throw std::runtime_error("DistWorker: add_worker after init");
    if (type == WorkerType::NEXT_LEVEL) manager_.add_next_level(worker);
    else manager_.add_sub(worker);
}

void DistWorker::init() {
    if (initialized_) throw std::runtime_error("DistWorker: already initialized");

    ring_.init(DIST_TASK_WINDOW_SIZE);
    orchestrator_.init(&tensormap_, &ring_, &scope_, &ready_queue_, slots_.get(), DIST_TASK_WINDOW_SIZE);

    // Start WorkerManager first — creates WorkerThreads.
    // The on_complete callback routes through the Scheduler's worker_done().
    manager_.start([this](DistTaskSlot slot) {
        scheduler_.worker_done(slot);
    });

    DistScheduler::Config cfg;
    cfg.slots = slots_.get();
    cfg.num_slots = DIST_TASK_WINDOW_SIZE;
    cfg.ready_queue = &ready_queue_;
    cfg.manager = &manager_;
    cfg.on_consumed_cb = [this](DistTaskSlot slot) {
        orchestrator_.on_consumed(slot);
    };

    scheduler_.start(cfg);
    initialized_ = true;
}

void DistWorker::close() {
    if (!initialized_) return;
    scheduler_.stop();
    manager_.stop();
    ring_.shutdown();
    initialized_ = false;
}

// =============================================================================
// IWorker::run() — DistWorker as sub-worker of a higher level (placeholder)
// =============================================================================

void DistWorker::run(const WorkerPayload & /*payload*/) {
    // Full L4+ support: payload would carry a HostTask* to execute.
    // Placeholder for plan step F.
}
