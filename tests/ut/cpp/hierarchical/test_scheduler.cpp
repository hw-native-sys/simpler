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

#include <algorithm>
#include <atomic>
#include <chrono>
#include <iterator>
#include <mutex>
#include <thread>
#include <vector>

#include "call_config.h"
#include "orchestrator.h"
#include "ring.h"
#include "scheduler.h"
#include "scope.h"
#include "tensormap.h"
#include "types.h"
#include "worker_manager.h"
#include "task_args.h"

// ---------------------------------------------------------------------------
// MockWorker: run() blocks until complete() is called by the test thread.
// ---------------------------------------------------------------------------

struct MockWorker : public IWorker {
    struct Record {
        uint64_t callable;
        uint64_t tensor_key;              // tensors[0].data (unique per submit in tests)
        const ContinuousTensor *tensors;  // backing pointer (distinct per group member)
    };

    std::vector<Record> dispatched;
    std::mutex dispatched_mu;

    std::mutex run_mu;
    std::condition_variable run_cv;
    std::atomic<bool> should_complete{false};
    std::atomic<bool> is_running{false};

    void run(uint64_t callable, TaskArgsView args, const CallConfig & /*cfg*/) override {
        {
            std::lock_guard<std::mutex> lk(dispatched_mu);
            uint64_t key = (args.tensor_count > 0) ? args.tensors[0].data : 0;
            dispatched.push_back({callable, key, args.tensors});
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
// Helper: build a TaskArgs whose only tensor has the given (data, tag).
// ---------------------------------------------------------------------------

static TaskArgs single_tensor_args(uint64_t data_ptr, TensorArgType tag) {
    TaskArgs a;
    ContinuousTensor t{};
    t.data = data_ptr;
    t.ndims = 1;
    t.shapes[0] = 1;
    t.dtype = DataType::UINT8;
    a.add_tensor(t, tag);
    return a;
}

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

struct SchedulerFixture : public ::testing::Test {
    TensorMap tm;
    Ring allocator;
    Scope scope;
    // Strict-4: per-type ready queues.
    ReadyQueue rq_next_level;
    ReadyQueue rq_sub;
    Orchestrator orch;
    MockWorker mock_worker;
    WorkerManager manager;
    Scheduler sched;
    CallConfig cfg;

    std::vector<TaskSlot> consumed_slots;
    std::mutex consumed_mu;

    TaskSlotState &S(TaskSlot id) { return *allocator.slot_state(id); }

    void SetUp() override {
        allocator.init(/*heap_bytes=*/1ULL << 20);
        orch.init(&tm, &allocator, &scope, &rq_next_level, &rq_sub);

        manager.add_next_level(&mock_worker);
        manager.start(&allocator, [this](TaskSlot slot) {
            sched.worker_done(slot);
        });

        Scheduler::Config c;
        c.ring = &allocator;
        c.ready_next_level_queue = &rq_next_level;
        c.ready_sub_queue = &rq_sub;
        c.manager = &manager;
        c.on_consumed_cb = [this](TaskSlot s) {
            orch.on_consumed(s);
            std::lock_guard<std::mutex> lk(consumed_mu);
            consumed_slots.push_back(s);
        };
        sched.start(c);
    }

    void TearDown() override {
        sched.stop();
        manager.stop();
        allocator.shutdown();
    }

    void wait_consumed(TaskSlot slot, int timeout_ms = 500) {
        auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        while (std::chrono::steady_clock::now() < deadline) {
            {
                std::lock_guard<std::mutex> lk(consumed_mu);
                for (TaskSlot s : consumed_slots)
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
    auto args_a = single_tensor_args(0xCAFE, TensorArgType::OUTPUT);
    auto res = orch.submit_next_level(0xDEAD, args_a, cfg);
    TaskSlot slot = res.task_slot;

    mock_worker.wait_running();
    ASSERT_GE(mock_worker.dispatched_count(), 1);
    EXPECT_EQ(mock_worker.dispatched[0].tensor_key, 0xCAFEu);
    EXPECT_EQ(mock_worker.dispatched[0].callable, 0xDEADu);

    mock_worker.complete();
    wait_consumed(slot);
}

TEST_F(SchedulerFixture, DependentTaskDispatchedAfterProducerCompletes) {
    auto args_a = single_tensor_args(0xBEEF, TensorArgType::OUTPUT);
    auto a = orch.submit_next_level(0xAA, args_a, cfg);

    auto args_b = single_tensor_args(0xBEEF, TensorArgType::INPUT);
    auto b = orch.submit_next_level(0xBB, args_b, cfg);
    EXPECT_EQ(S(b.task_slot).state.load(), TaskState::PENDING);

    mock_worker.wait_running();
    EXPECT_EQ(mock_worker.dispatched[0].callable, 0xAAu);
    mock_worker.complete();  // A done

    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(300);
    while (mock_worker.dispatched_count() < 2 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_GE(mock_worker.dispatched_count(), 2);
    EXPECT_EQ(mock_worker.dispatched[1].callable, 0xBBu);

    mock_worker.complete();  // B done
    wait_consumed(b.task_slot);
    (void)a;
}

// ===========================================================================
// Group task tests -- fixture with 2 MockWorkers
// ===========================================================================

struct GroupSchedulerFixture : public ::testing::Test {
    TensorMap tm;
    Ring allocator;
    Scope scope;
    // Strict-4: per-type ready queues.
    ReadyQueue rq_next_level;
    ReadyQueue rq_sub;
    Orchestrator orch;
    MockWorker worker_a;
    MockWorker worker_b;
    WorkerManager manager;
    Scheduler sched;
    CallConfig cfg;

    std::vector<TaskSlot> consumed_slots;
    std::mutex consumed_mu;

    TaskSlotState &S(TaskSlot id) { return *allocator.slot_state(id); }

    void SetUp() override {
        allocator.init(/*heap_bytes=*/1ULL << 20);
        orch.init(&tm, &allocator, &scope, &rq_next_level, &rq_sub);

        manager.add_next_level(&worker_a);
        manager.add_next_level(&worker_b);
        manager.start(&allocator, [this](TaskSlot slot) {
            sched.worker_done(slot);
        });

        Scheduler::Config c;
        c.ring = &allocator;
        c.ready_next_level_queue = &rq_next_level;
        c.ready_sub_queue = &rq_sub;
        c.manager = &manager;
        c.on_consumed_cb = [this](TaskSlot s) {
            orch.on_consumed(s);
            std::lock_guard<std::mutex> lk(consumed_mu);
            consumed_slots.push_back(s);
        };
        sched.start(c);
    }

    void TearDown() override {
        sched.stop();
        manager.stop();
        allocator.shutdown();
    }

    void wait_consumed(TaskSlot slot, int timeout_ms = 1000) {
        auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        while (std::chrono::steady_clock::now() < deadline) {
            {
                std::lock_guard<std::mutex> lk(consumed_mu);
                for (TaskSlot s : consumed_slots)
                    if (s == slot) return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        FAIL() << "Timed out waiting for slot " << slot << " to be consumed";
    }
};

TEST_F(GroupSchedulerFixture, GroupDispatchesToNWorkers) {
    TaskArgs a0 = single_tensor_args(0xA0, TensorArgType::OUTPUT);
    TaskArgs a1 = single_tensor_args(0xA1, TensorArgType::OUTPUT);

    auto res = orch.submit_next_level_group(0xDEAD, {a0, a1}, cfg);
    TaskSlot slot = res.task_slot;

    worker_a.wait_running();
    worker_b.wait_running();

    EXPECT_EQ(worker_a.dispatched_count(), 1);
    EXPECT_EQ(worker_b.dispatched_count(), 1);

    // Each worker got a different TaskArgs from the slot's task_args_list.
    uint64_t keys[2] = {worker_a.dispatched[0].tensor_key, worker_b.dispatched[0].tensor_key};
    std::sort(std::begin(keys), std::end(keys));
    EXPECT_EQ(keys[0], 0xA0u);
    EXPECT_EQ(keys[1], 0xA1u);
    EXPECT_NE(worker_a.dispatched[0].tensors, worker_b.dispatched[0].tensors);
    (void)slot;

    worker_a.complete();
    worker_b.complete();
    wait_consumed(slot);
}

TEST_F(GroupSchedulerFixture, GroupCompletesOnlyWhenAllDone) {
    TaskArgs a0 = single_tensor_args(0xB0, TensorArgType::OUTPUT);
    TaskArgs a1 = single_tensor_args(0xB1, TensorArgType::OUTPUT);
    auto res = orch.submit_next_level_group(0xDEAD, {a0, a1}, cfg);
    TaskSlot slot = res.task_slot;

    worker_a.wait_running();
    worker_b.wait_running();

    worker_a.complete();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_EQ(S(slot).state.load(), TaskState::RUNNING);

    worker_b.complete();
    wait_consumed(slot);
}

// ===========================================================================
// Strict-4: per-worker-type ready queues (no head-of-line blocking across
// types). Covered here with one NEXT_LEVEL worker + one SUB worker: with a
// saturated NEXT_LEVEL pool, a SUB task submitted afterwards must still
// dispatch immediately instead of waiting behind the stuck next-level task.
// ===========================================================================

struct MixedTypeSchedulerFixture : public ::testing::Test {
    TensorMap tm;
    Ring allocator;
    Scope scope;
    ReadyQueue rq_next_level;
    ReadyQueue rq_sub;
    Orchestrator orch;
    MockWorker next_level_worker;
    MockWorker sub_worker;
    WorkerManager manager;
    Scheduler sched;
    CallConfig cfg;

    std::vector<TaskSlot> consumed_slots;
    std::mutex consumed_mu;

    TaskSlotState &S(TaskSlot id) { return *allocator.slot_state(id); }

    void SetUp() override {
        allocator.init(/*heap_bytes=*/1ULL << 20);
        orch.init(&tm, &allocator, &scope, &rq_next_level, &rq_sub);

        manager.add_next_level(&next_level_worker);
        manager.add_sub(&sub_worker);
        manager.start(&allocator, [this](TaskSlot slot) {
            sched.worker_done(slot);
        });

        Scheduler::Config c;
        c.ring = &allocator;
        c.ready_next_level_queue = &rq_next_level;
        c.ready_sub_queue = &rq_sub;
        c.manager = &manager;
        c.on_consumed_cb = [this](TaskSlot s) {
            orch.on_consumed(s);
            std::lock_guard<std::mutex> lk(consumed_mu);
            consumed_slots.push_back(s);
        };
        sched.start(c);
    }

    void TearDown() override {
        sched.stop();
        manager.stop();
        allocator.shutdown();
    }

    bool is_consumed(TaskSlot slot) {
        std::lock_guard<std::mutex> lk(consumed_mu);
        for (TaskSlot s : consumed_slots)
            if (s == slot) return true;
        return false;
    }

    void wait_consumed(TaskSlot slot, int timeout_ms = 500) {
        auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        while (std::chrono::steady_clock::now() < deadline) {
            if (is_consumed(slot)) return;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        FAIL() << "Timed out waiting for slot " << slot << " to be consumed";
    }
};

TEST_F(MixedTypeSchedulerFixture, SubTaskDispatchesWhileNextLevelPoolSaturated) {
    // Submit a next-level task; the only chip worker begins running it and
    // stays blocked until we call complete() on it.
    auto chip_args = single_tensor_args(0xAAA, TensorArgType::OUTPUT);
    auto chip = orch.submit_next_level(0xCDCD, chip_args, cfg);
    next_level_worker.wait_running();
    ASSERT_TRUE(next_level_worker.is_running.load());

    // Now submit a sub task while the chip pool is saturated. With a single
    // shared ready queue this would block behind any next-level task sitting
    // at the queue head waiting for a free chip worker. With per-type
    // queues (Strict-4) it must dispatch immediately to the idle sub
    // worker.
    auto sub_args = single_tensor_args(0xBBB, TensorArgType::OUTPUT);
    auto sub = orch.submit_sub(/*callable_id=*/7, sub_args);

    sub_worker.wait_running();
    EXPECT_TRUE(sub_worker.is_running.load());
    EXPECT_TRUE(next_level_worker.is_running.load()) << "chip worker must still be busy";

    // Complete the sub task first; it reaches CONSUMED while the chip task
    // is still running -- demonstrating independent per-type dispatch.
    sub_worker.complete();
    wait_consumed(sub.task_slot);
    EXPECT_FALSE(is_consumed(chip.task_slot));

    next_level_worker.complete();
    wait_consumed(chip.task_slot);
}

TEST_F(GroupSchedulerFixture, GroupDependencyChain) {
    // Group A (2 workers) produces an OUTPUT at key 0xCAFE.
    // Task B reads INPUT at the same key -- depends on group A.
    TaskArgs a0 = single_tensor_args(0xCAFE, TensorArgType::OUTPUT);
    TaskArgs a1 = single_tensor_args(0xCAFE, TensorArgType::OUTPUT);
    auto a = orch.submit_next_level_group(0xDEAD, {a0, a1}, cfg);

    auto args_b = single_tensor_args(0xCAFE, TensorArgType::INPUT);
    auto b = orch.submit_next_level(0xDEAD, args_b, cfg);
    EXPECT_EQ(S(b.task_slot).state.load(), TaskState::PENDING);

    worker_a.wait_running();
    worker_b.wait_running();
    worker_a.complete();
    worker_b.complete();

    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
    while (worker_a.dispatched_count() + worker_b.dispatched_count() < 3 &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    int total = worker_a.dispatched_count() + worker_b.dispatched_count();
    EXPECT_GE(total, 3);  // 2 from group A + 1 from B

    if (worker_a.is_running.load()) worker_a.complete();
    if (worker_b.is_running.load()) worker_b.complete();
    wait_consumed(b.task_slot);
    (void)a;  // suppress unused
}
