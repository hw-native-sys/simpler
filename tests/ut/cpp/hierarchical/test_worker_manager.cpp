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
 * WorkerManager / WorkerThread independent UT.
 *
 * Tests the worker-pool lifecycle, idle selection, dispatch, and group dispatch
 * in THREAD mode using a lightweight MockWorker.  PROCESS mode requires fork +
 * shared-memory children and is covered by the Python integration tests.
 *
 * Follows UT development guidelines:
 *   - AAA pattern (Arrange / Act / Assert)
 *   - FIRST (Fast / Independent / Repeatable / Self-validating / Timely)
 *   - Single responsibility per test
 *   - Naming: Method_Scenario_ExpectedResult
 *   - Mock/Stub for external deps (IWorker)
 */

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

#include "ring.h"
#include "types.h"
#include "worker.h"
#include "worker_manager.h"

namespace {

// ---------------------------------------------------------------------------
// CountingWorker: records run() calls, optionally blocks until released.
// ---------------------------------------------------------------------------

struct CountingWorker : public IWorker {
    std::atomic<int> run_count{0};
    std::atomic<bool> is_running{false};

    std::mutex mu;
    std::condition_variable cv;
    bool should_complete{false};
    bool blocking{false};

    explicit CountingWorker(bool blocking_ = false) :
        blocking(blocking_) {}

    void run(uint64_t /*callable*/, TaskArgsView /*args*/, const ChipCallConfig & /*config*/) override {
        run_count.fetch_add(1, std::memory_order_relaxed);
        if (blocking) {
            is_running.store(true, std::memory_order_release);
            std::unique_lock<std::mutex> lk(mu);
            cv.wait(lk, [this] {
                return should_complete;
            });
            should_complete = false;
            is_running.store(false, std::memory_order_release);
        }
    }

    void complete() {
        std::lock_guard<std::mutex> lk(mu);
        should_complete = true;
        cv.notify_one();
    }

    void wait_running(int timeout_ms = 500) {
        auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        while (!is_running.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
};

// ---------------------------------------------------------------------------
// Fixture: creates a Ring and provides start/stop helpers.
// ---------------------------------------------------------------------------

class WorkerManagerTest : public ::testing::Test {
protected:
    Ring ring_;
    WorkerManager manager_;
    std::vector<TaskSlot> completed_slots_;
    std::mutex completed_mu_;

    void SetUp() override { ring_.init(/*heap_bytes=*/1ULL << 16); }

    void TearDown() override {
        manager_.stop();
        ring_.shutdown();
    }

    void start_manager() {
        manager_.start(&ring_, [this](TaskSlot slot) {
            std::lock_guard<std::mutex> lk(completed_mu_);
            completed_slots_.push_back(slot);
        });
    }

    void wait_completed(int expected, int timeout_ms = 500) {
        auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        while (std::chrono::steady_clock::now() < deadline) {
            {
                std::lock_guard<std::mutex> lk(completed_mu_);
                if (static_cast<int>(completed_slots_.size()) >= expected) return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
};

}  // namespace

// =============================================================================
// WorkerManager registration
// =============================================================================

TEST_F(WorkerManagerTest, AddNextLevel_BeforeStart_PickIdleReturnsWorkerAfterStart) {
    // Arrange
    CountingWorker w;
    manager_.add_next_level(&w);

    // Act
    start_manager();

    // Assert
    WorkerThread *idle = manager_.pick_idle(WorkerType::NEXT_LEVEL);
    ASSERT_NE(idle, nullptr);
    EXPECT_TRUE(idle->idle());
}

TEST_F(WorkerManagerTest, AddSub_BeforeStart_PickIdleReturnsSubWorker) {
    CountingWorker w;
    manager_.add_sub(&w);
    start_manager();

    EXPECT_NE(manager_.pick_idle(WorkerType::SUB), nullptr);
    EXPECT_EQ(manager_.pick_idle(WorkerType::NEXT_LEVEL), nullptr);
}

TEST_F(WorkerManagerTest, NoWorkers_PickIdleReturnsNull) {
    start_manager();
    EXPECT_EQ(manager_.pick_idle(WorkerType::NEXT_LEVEL), nullptr);
    EXPECT_EQ(manager_.pick_idle(WorkerType::SUB), nullptr);
}

// =============================================================================
// WorkerManager::start -- null ring
// =============================================================================

TEST_F(WorkerManagerTest, Start_NullRing_Throws) {
    CountingWorker w;
    manager_.add_next_level(&w);
    EXPECT_THROW(manager_.start(nullptr, [](TaskSlot) {}), std::invalid_argument);
}

// =============================================================================
// Dispatch (THREAD mode)
// =============================================================================

TEST_F(WorkerManagerTest, Dispatch_SingleTask_WorkerRunsAndCompletes) {
    // Arrange: one blocking worker
    CountingWorker w(/*blocking=*/true);
    manager_.add_next_level(&w);
    start_manager();

    // Allocate a slot in the ring so there is a valid TaskSlotState.
    AllocResult ar = ring_.alloc(0);
    TaskSlot slot = ar.slot;
    ASSERT_NE(slot, INVALID_SLOT);
    TaskSlotState &s = *ring_.slot_state(slot);
    s.worker_type = WorkerType::NEXT_LEVEL;
    s.callable = 0xABC;

    // Act: dispatch the task
    WorkerThread *wt = manager_.pick_idle(WorkerType::NEXT_LEVEL);
    ASSERT_NE(wt, nullptr);
    wt->dispatch({slot, 0});

    // Assert: worker is running
    w.wait_running();
    EXPECT_TRUE(w.is_running.load());
    EXPECT_FALSE(wt->idle());

    // Release the worker
    w.complete();
    wait_completed(1);

    // Worker is idle again
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(200);
    while (!wt->idle() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_TRUE(wt->idle());
    EXPECT_EQ(w.run_count.load(), 1);
}

TEST_F(WorkerManagerTest, Dispatch_MultipleWorkers_BothExecute) {
    // Arrange: two blocking workers
    CountingWorker w1(true), w2(true);
    manager_.add_next_level(&w1);
    manager_.add_next_level(&w2);
    start_manager();

    // Two slots
    AllocResult ar1 = ring_.alloc(0);
    AllocResult ar2 = ring_.alloc(0);
    TaskSlot s1 = ar1.slot;
    TaskSlot s2 = ar2.slot;
    ring_.slot_state(s1)->worker_type = WorkerType::NEXT_LEVEL;
    ring_.slot_state(s1)->callable = 1;
    ring_.slot_state(s2)->worker_type = WorkerType::NEXT_LEVEL;
    ring_.slot_state(s2)->callable = 2;

    // Dispatch to both
    auto *t1 = manager_.get_worker(WorkerType::NEXT_LEVEL, 0);
    auto *t2 = manager_.get_worker(WorkerType::NEXT_LEVEL, 1);
    ASSERT_NE(t1, nullptr);
    ASSERT_NE(t2, nullptr);
    t1->dispatch({s1, 0});
    t2->dispatch({s2, 0});

    w1.wait_running();
    w2.wait_running();

    // Both are running concurrently
    EXPECT_TRUE(w1.is_running.load());
    EXPECT_TRUE(w2.is_running.load());

    w1.complete();
    w2.complete();
    wait_completed(2);
}

// =============================================================================
// pick_idle / pick_n_idle / pick_idle_excluding
// =============================================================================

TEST_F(WorkerManagerTest, PickNIdle_ReturnsUpToN) {
    CountingWorker w1, w2, w3;
    manager_.add_next_level(&w1);
    manager_.add_next_level(&w2);
    manager_.add_next_level(&w3);
    start_manager();

    auto idle2 = manager_.pick_n_idle(WorkerType::NEXT_LEVEL, 2);
    EXPECT_EQ(static_cast<int>(idle2.size()), 2);

    auto idle10 = manager_.pick_n_idle(WorkerType::NEXT_LEVEL, 10);
    EXPECT_EQ(static_cast<int>(idle10.size()), 3);
}

TEST_F(WorkerManagerTest, PickIdleExcluding_SkipsExcludedWorkers) {
    CountingWorker w1, w2;
    manager_.add_next_level(&w1);
    manager_.add_next_level(&w2);
    start_manager();

    auto *t0 = manager_.get_worker(WorkerType::NEXT_LEVEL, 0);
    auto *t1 = manager_.get_worker(WorkerType::NEXT_LEVEL, 1);

    // Exclude t0 -> should get t1
    auto *picked = manager_.pick_idle_excluding(WorkerType::NEXT_LEVEL, {t0});
    EXPECT_EQ(picked, t1);

    // Exclude both -> nullptr
    auto *none = manager_.pick_idle_excluding(WorkerType::NEXT_LEVEL, {t0, t1});
    EXPECT_EQ(none, nullptr);
}

// =============================================================================
// get_worker -- index bounds
// =============================================================================

TEST_F(WorkerManagerTest, GetWorker_ValidIndex_ReturnsThread) {
    CountingWorker w;
    manager_.add_next_level(&w);
    start_manager();

    EXPECT_NE(manager_.get_worker(WorkerType::NEXT_LEVEL, 0), nullptr);
}

TEST_F(WorkerManagerTest, GetWorker_NegativeIndex_ReturnsNull) {
    CountingWorker w;
    manager_.add_next_level(&w);
    start_manager();

    EXPECT_EQ(manager_.get_worker(WorkerType::NEXT_LEVEL, -1), nullptr);
}

TEST_F(WorkerManagerTest, GetWorker_OutOfBoundsIndex_ReturnsNull) {
    CountingWorker w;
    manager_.add_next_level(&w);
    start_manager();

    EXPECT_EQ(manager_.get_worker(WorkerType::NEXT_LEVEL, 99), nullptr);
}

// =============================================================================
// any_busy
// =============================================================================

TEST_F(WorkerManagerTest, AnyBusy_AllIdle_ReturnsFalse) {
    CountingWorker w;
    manager_.add_next_level(&w);
    start_manager();

    EXPECT_FALSE(manager_.any_busy());
}

TEST_F(WorkerManagerTest, AnyBusy_OneRunning_ReturnsTrue) {
    CountingWorker w(/*blocking=*/true);
    manager_.add_next_level(&w);
    start_manager();

    AllocResult ar = ring_.alloc(0);
    TaskSlot slot = ar.slot;
    ring_.slot_state(slot)->worker_type = WorkerType::NEXT_LEVEL;

    manager_.pick_idle(WorkerType::NEXT_LEVEL)->dispatch({slot, 0});
    w.wait_running();

    EXPECT_TRUE(manager_.any_busy());

    w.complete();
    wait_completed(1);
}

// =============================================================================
// Worker lifecycle (top-level Worker class)
// =============================================================================

TEST(WorkerLifecycleTest, Construct_InitClose_NoCrash) {
    // Arrange
    CountingWorker mock;

    // Act: full lifecycle without dispatching anything
    Worker worker(/*level=*/3, /*heap_ring_size=*/1ULL << 16);
    worker.add_worker(WorkerType::NEXT_LEVEL, &mock);
    worker.init();
    worker.close();

    // Assert: reaching here without crash/hang is the observable signal.
    SUCCEED();
}

TEST(WorkerLifecycleTest, AddWorker_AfterInit_Throws) {
    CountingWorker mock;
    Worker worker(3, 1ULL << 16);
    worker.add_worker(WorkerType::NEXT_LEVEL, &mock);
    worker.init();

    CountingWorker extra;
    EXPECT_THROW(worker.add_worker(WorkerType::NEXT_LEVEL, &extra), std::runtime_error);

    worker.close();
}

TEST(WorkerLifecycleTest, DoubleInit_Throws) {
    CountingWorker mock;
    Worker worker(3, 1ULL << 16);
    worker.add_worker(WorkerType::NEXT_LEVEL, &mock);
    worker.init();

    EXPECT_THROW(worker.init(), std::runtime_error);

    worker.close();
}

TEST(WorkerLifecycleTest, CloseWithoutInit_Noop) {
    Worker worker(3, 1ULL << 16);
    worker.close();
    SUCCEED();
}

TEST(WorkerLifecycleTest, DestructorCallsClose) {
    CountingWorker mock;
    {
        Worker worker(3, 1ULL << 16);
        worker.add_worker(WorkerType::NEXT_LEVEL, &mock);
        worker.init();
        // destructor should call close() without hang/crash
    }
    SUCCEED();
}

TEST(WorkerLifecycleTest, RunCallback_InvokedWhenSet) {
    Worker worker(3, 1ULL << 16);

    int callback_count = 0;
    worker.set_run_callback([&](uint64_t, TaskArgsView, const ChipCallConfig &) {
        callback_count++;
    });

    // Worker::run delegates to run_callback_
    ChipCallConfig cfg{};
    TaskArgsView view{};
    worker.run(42, view, cfg);

    EXPECT_EQ(callback_count, 1);
}

TEST(WorkerLifecycleTest, RunWithoutCallback_Noop) {
    Worker worker(3, 1ULL << 16);

    ChipCallConfig cfg{};
    TaskArgsView view{};
    worker.run(42, view, cfg);
    // No crash when run_callback_ is empty.
    SUCCEED();
}
