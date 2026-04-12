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
        on_consumed(slot);
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
// Orchestrator-facing API
// =============================================================================

DistSubmitResult DistWorker::submit(
    WorkerType worker_type, const WorkerPayload &base_payload, const std::vector<DistInputSpec> &inputs,
    const std::vector<DistOutputSpec> &outputs
) {
    active_tasks_.fetch_add(1, std::memory_order_relaxed);
    return orchestrator_.submit(worker_type, base_payload, inputs, outputs);
}

DistSubmitResult DistWorker::submit_group(
    WorkerType worker_type, const WorkerPayload &base_payload, const std::vector<const void *> &args_list,
    const std::vector<DistInputSpec> &inputs, const std::vector<DistOutputSpec> &outputs
) {
    active_tasks_.fetch_add(1, std::memory_order_relaxed);
    return orchestrator_.submit_group(worker_type, base_payload, args_list, inputs, outputs);
}

void DistWorker::scope_begin() { orchestrator_.scope_begin(); }
void DistWorker::scope_end() { orchestrator_.scope_end(); }

void DistWorker::drain() {
    std::unique_lock<std::mutex> lk(drain_mu_);
    drain_cv_.wait(lk, [this] {
        return active_tasks_.load(std::memory_order_acquire) == 0;
    });
}

// =============================================================================
// on_consumed callback (called from Scheduler thread)
// =============================================================================

void DistWorker::on_consumed(DistTaskSlot slot) {
    orchestrator_.on_consumed(slot);

    int32_t remaining = active_tasks_.fetch_sub(1, std::memory_order_acq_rel) - 1;
    if (remaining == 0) {
        std::lock_guard<std::mutex> lk(drain_mu_);
        drain_cv_.notify_all();
    }
}

// =============================================================================
// IWorker::run() — DistWorker as sub-worker of a higher level (placeholder)
// =============================================================================

void DistWorker::run(const WorkerPayload & /*payload*/) {
    // Full L4+ support: payload would carry a HostTask* to execute.
    // For now this is a placeholder; drain() returns immediately when idle.
}
