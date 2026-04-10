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

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>
#include <vector>

#include "dist_orchestrator.h"
#include "dist_ring.h"
#include "dist_scheduler.h"
#include "dist_scope.h"
#include "dist_tensormap.h"
#include "dist_types.h"

// ---------------------------------------------------------------------------
// MockWorker: run() blocks until complete() is called by the test thread.
// WorkerThread wraps it, so the Scheduler calls WorkerThread.dispatch() and
// WorkerThread calls MockWorker.run() in its own thread.
// ---------------------------------------------------------------------------

struct MockWorker : public IWorker {
    struct Record {
        DistTaskSlot slot;
        WorkerType type;
        const void *args;
    };

    std::vector<Record> dispatched;
    std::mutex dispatched_mu;

    std::mutex run_mu;
    std::condition_variable run_cv;
    std::atomic<bool> should_complete{false};
    std::atomic<bool> is_running{false};

    void run(const WorkerPayload &p) override {
        {
            std::lock_guard<std::mutex> lk(dispatched_mu);
            dispatched.push_back({p.task_slot, p.worker_type, p.args});
        }
        is_running.store(true, std::memory_order_release);

        std::unique_lock<std::mutex> lk(run_mu);
        run_cv.wait(lk, [this] {
            return should_complete.load(std::memory_order_acquire);
        });
        should_complete.store(false, std::memory_order_relaxed);
        is_running.store(false, std::memory_order_release);
    }

    void complete() {
        std::lock_guard<std::mutex> lk(run_mu);
        should_complete.store(true, std::memory_order_release);
        run_cv.notify_one();
    }

    // Wait until run() starts (dispatched and executing)
    void wait_running(int timeout_ms = 500) {
        auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        while (!is_running.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    int dispatched_count() {
        std::lock_guard<std::mutex> lk(dispatched_mu);
        return static_cast<int>(dispatched.size());
    }
};

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

struct SchedulerFixture : public ::testing::Test {
    static constexpr int32_t N = DIST_TASK_WINDOW_SIZE;

    std::unique_ptr<DistTaskSlotState[]> slots;
    DistTensorMap tm;
    DistRing ring;
    DistScope scope;
    DistReadyQueue rq;
    DistOrchestrator orch;
    MockWorker mock_worker;
    DistScheduler sched;

    std::vector<DistTaskSlot> consumed_slots;
    std::mutex consumed_mu;

    void SetUp() override {
        slots = std::make_unique<DistTaskSlotState[]>(N);
        ring.init(N);
        orch.init(&tm, &ring, &scope, &rq, slots.get(), N);

        DistScheduler::Config cfg;
        cfg.slots = slots.get();
        cfg.num_slots = N;
        cfg.ready_queue = &rq;
        cfg.next_level_workers = {&mock_worker};
        cfg.on_consumed_cb = [this](DistTaskSlot s) {
            orch.on_consumed(s);
            std::lock_guard<std::mutex> lk(consumed_mu);
            consumed_slots.push_back(s);
        };
        sched.start(cfg);
    }

    void TearDown() override {
        sched.stop();
        ring.shutdown();
    }

    DistSubmitResult
    submit_next_level(const std::vector<DistInputSpec> &inputs, const std::vector<DistOutputSpec> &outputs) {
        WorkerPayload p;
        p.worker_type = WorkerType::NEXT_LEVEL;
        return orch.submit(WorkerType::NEXT_LEVEL, p, inputs, outputs);
    }

    void wait_consumed(DistTaskSlot slot, int timeout_ms = 500) {
        auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        while (std::chrono::steady_clock::now() < deadline) {
            {
                std::lock_guard<std::mutex> lk(consumed_mu);
                for (DistTaskSlot s : consumed_slots)
                    if (s == slot) return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        FAIL() << "Timed out waiting for slot " << slot << " to be consumed";
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST_F(SchedulerFixture, IndependentTaskDispatchedAndConsumed) {
    auto res = submit_next_level({}, {{64}});
    DistTaskSlot slot = res.task_slot;

    // WorkerThread calls MockWorker.run() — wait for it to start
    mock_worker.wait_running();
    ASSERT_GE(mock_worker.dispatched_count(), 1);
    EXPECT_EQ(mock_worker.dispatched[0].slot, slot);

    // Signal completion → WorkerThread pushes to completion_queue → Scheduler consumes
    mock_worker.complete();
    wait_consumed(slot);
}

TEST_F(SchedulerFixture, DependentTaskDispatchedAfterProducerCompletes) {
    auto a = submit_next_level({}, {{128}});
    uint64_t a_key = reinterpret_cast<uint64_t>(a.outputs[0].ptr);

    auto b = submit_next_level({{a_key}}, {{64}});
    EXPECT_EQ(slots[b.task_slot].state.load(), TaskState::PENDING);

    // Complete A → B should become ready
    mock_worker.wait_running();
    EXPECT_EQ(mock_worker.dispatched[0].slot, a.task_slot);
    mock_worker.complete();  // A done

    // Wait for B to be dispatched
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(300);
    while (mock_worker.dispatched_count() < 2 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_GE(mock_worker.dispatched_count(), 2);
    EXPECT_EQ(mock_worker.dispatched[1].slot, b.task_slot);

    mock_worker.complete();  // B done
    wait_consumed(b.task_slot);
}

// ===========================================================================
// Group task tests — fixture with 2 MockWorkers
// ===========================================================================

struct GroupSchedulerFixture : public ::testing::Test {
    static constexpr int32_t N = DIST_TASK_WINDOW_SIZE;

    std::unique_ptr<DistTaskSlotState[]> slots;
    DistTensorMap tm;
    DistRing ring;
    DistScope scope;
    DistReadyQueue rq;
    DistOrchestrator orch;
    MockWorker worker_a;
    MockWorker worker_b;
    DistScheduler sched;

    std::vector<DistTaskSlot> consumed_slots;
    std::mutex consumed_mu;

    void SetUp() override {
        slots = std::make_unique<DistTaskSlotState[]>(N);
        ring.init(N);
        orch.init(&tm, &ring, &scope, &rq, slots.get(), N);

        DistScheduler::Config cfg;
        cfg.slots = slots.get();
        cfg.num_slots = N;
        cfg.ready_queue = &rq;
        cfg.next_level_workers = {&worker_a, &worker_b};
        cfg.on_consumed_cb = [this](DistTaskSlot s) {
            orch.on_consumed(s);
            std::lock_guard<std::mutex> lk(consumed_mu);
            consumed_slots.push_back(s);
        };
        sched.start(cfg);
    }

    void TearDown() override {
        sched.stop();
        ring.shutdown();
    }

    void wait_consumed(DistTaskSlot slot, int timeout_ms = 1000) {
        auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        while (std::chrono::steady_clock::now() < deadline) {
            {
                std::lock_guard<std::mutex> lk(consumed_mu);
                for (DistTaskSlot s : consumed_slots)
                    if (s == slot) return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        FAIL() << "Timed out waiting for slot " << slot << " to be consumed";
    }
};

TEST_F(GroupSchedulerFixture, GroupDispatchesToNWorkers) {
    // Two distinct args pointers — one per worker
    int dummy_args_0 = 0;
    int dummy_args_1 = 1;

    WorkerPayload p;
    p.worker_type = WorkerType::NEXT_LEVEL;
    std::vector<const void *> args_list = {&dummy_args_0, &dummy_args_1};

    auto res = orch.submit_group(WorkerType::NEXT_LEVEL, p, args_list, {}, {{64}});
    DistTaskSlot slot = res.task_slot;

    // Both workers should receive dispatches
    worker_a.wait_running();
    worker_b.wait_running();

    EXPECT_EQ(worker_a.dispatched_count(), 1);
    EXPECT_EQ(worker_b.dispatched_count(), 1);
    EXPECT_EQ(worker_a.dispatched[0].slot, slot);
    EXPECT_EQ(worker_b.dispatched[0].slot, slot);

    // Each worker got a different args pointer
    EXPECT_EQ(worker_a.dispatched[0].args, &dummy_args_0);
    EXPECT_EQ(worker_b.dispatched[0].args, &dummy_args_1);

    worker_a.complete();
    worker_b.complete();
    wait_consumed(slot);
}

TEST_F(GroupSchedulerFixture, GroupCompletesOnlyWhenAllDone) {
    int d0 = 0, d1 = 1;
    WorkerPayload p;
    p.worker_type = WorkerType::NEXT_LEVEL;

    auto res = orch.submit_group(WorkerType::NEXT_LEVEL, p, {&d0, &d1}, {}, {});
    DistTaskSlot slot = res.task_slot;

    worker_a.wait_running();
    worker_b.wait_running();

    // Complete only worker A — task should still be RUNNING
    worker_a.complete();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_EQ(slots[slot].state.load(), TaskState::RUNNING);

    // Complete worker B — now the task should reach COMPLETED → CONSUMED
    worker_b.complete();
    wait_consumed(slot);
}

TEST_F(GroupSchedulerFixture, GroupDependencyChain) {
    // Group task A (2 workers) produces an output.
    // Task B depends on A's output — B stays PENDING until group A finishes.
    int d0 = 0, d1 = 1;
    WorkerPayload pa;
    pa.worker_type = WorkerType::NEXT_LEVEL;

    auto a = orch.submit_group(WorkerType::NEXT_LEVEL, pa, {&d0, &d1}, {}, {{128}});
    uint64_t a_out = reinterpret_cast<uint64_t>(a.outputs[0].ptr);

    // Submit B depending on A's output
    WorkerPayload pb;
    pb.worker_type = WorkerType::NEXT_LEVEL;
    auto b = orch.submit(WorkerType::NEXT_LEVEL, pb, {{a_out}}, {});
    EXPECT_EQ(slots[b.task_slot].state.load(), TaskState::PENDING);

    // Complete group A
    worker_a.wait_running();
    worker_b.wait_running();
    worker_a.complete();
    worker_b.complete();

    // B should become ready and get dispatched
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
    while (worker_a.dispatched_count() + worker_b.dispatched_count() < 3 &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    int total = worker_a.dispatched_count() + worker_b.dispatched_count();
    EXPECT_GE(total, 3);  // 2 from group A + 1 from B

    // Complete B
    if (worker_a.is_running.load()) worker_a.complete();
    if (worker_b.is_running.load()) worker_b.complete();
    wait_consumed(b.task_slot);
}
