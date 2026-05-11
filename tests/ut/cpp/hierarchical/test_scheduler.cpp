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
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
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
// MockMailboxWorker: in-process stand-in for the forked Python child loop.
//
// The production dispatch path writes (callable, config, args_blob) into a
// MAILBOX_SIZE-byte shared region and spin-polls TASK_DONE; the real child
// (`_chip_process_loop` in python/simpler/worker.py) decodes the mailbox and
// runs an IWorker. For unit testing the Scheduler / WorkerManager state
// machine in isolation, we replace the forked child with a thread inside
// the test process that mimics the same handshake but blocks until the
// test thread releases it via `complete()`.
//
// API parity with the previous IWorker-based MockWorker:
//   - dispatched[i].callable / .tensor_key — recorded on TASK_READY
//   - is_running                            — atomic flag the test polls
//   - wait_running()                        — spin-wait until is_running flips
//   - complete()                            — release the parked dispatch so
//                                             the loop writes TASK_DONE
// ---------------------------------------------------------------------------

struct MockMailboxWorker {
    struct Record {
        uint64_t callable;
        uint64_t tensor_key;  // first tensor's `data` field (unique per submit in tests)
    };

    alignas(8) std::array<char, MAILBOX_SIZE> mailbox{};
    std::vector<Record> dispatched;
    std::mutex dispatched_mu;

    std::mutex run_mu;
    std::condition_variable run_cv;
    std::atomic<bool> should_complete{false};
    std::atomic<bool> is_running{false};
    std::atomic<bool> stop_flag{false};
    std::thread loop_thread;

    void start() {
        // SharedMemory zero-fills, but std::array does not — explicitly
        // store IDLE (=0) to mirror production parity and keep the polling
        // loop's first read deterministic.
        write_state(MailboxState::IDLE);
        loop_thread = std::thread(&MockMailboxWorker::loop, this);
    }

    ~MockMailboxWorker() {
        // Defensive teardown — if a test fails before completing every
        // dispatch, set stop_flag and wake the parked loop so the thread
        // joins instead of leaking. The loop's TASK_READY branch always
        // publishes TASK_DONE before checking stop_flag, so any in-flight
        // WorkerThread::dispatch_process completes its spin-poll cleanly.
        stop_flag.store(true, std::memory_order_release);
        {
            std::lock_guard<std::mutex> lk(run_mu);
            should_complete.store(true, std::memory_order_release);
            run_cv.notify_one();
        }
        if (loop_thread.joinable()) loop_thread.join();
    }

    void *mailbox_ptr() { return mailbox.data(); }

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

private:
    // Mirror the acquire/release semantics in
    // worker_manager.cpp::read_mailbox_state / write_mailbox_state. Plain
    // memcpy on the mailbox state would let the parent observe the state
    // flip before the preceding error-field stores are visible.
    MailboxState read_state() const {
        const auto *ptr = reinterpret_cast<const volatile int32_t *>(mailbox.data() + MAILBOX_OFF_STATE);
        int32_t v = __atomic_load_n(ptr, __ATOMIC_ACQUIRE);
        return static_cast<MailboxState>(v);
    }

    void write_state(MailboxState s) {
        auto *ptr = reinterpret_cast<volatile int32_t *>(mailbox.data() + MAILBOX_OFF_STATE);
        __atomic_store_n(ptr, static_cast<int32_t>(s), __ATOMIC_RELEASE);
    }

    void loop() {
        while (true) {
            if (stop_flag.load(std::memory_order_acquire)) return;
            MailboxState s = read_state();
            if (s == MailboxState::TASK_READY) {
                uint64_t callable = 0;
                std::memcpy(&callable, mailbox.data() + MAILBOX_OFF_CALLABLE, sizeof(uint64_t));
                int32_t t_count = 0;
                std::memcpy(&t_count, mailbox.data() + MAILBOX_OFF_ARGS, sizeof(int32_t));
                uint64_t tensor_key = 0;
                if (t_count > 0) {
                    ContinuousTensor first{};
                    std::memcpy(
                        &first, mailbox.data() + MAILBOX_OFF_ARGS + TASK_ARGS_BLOB_HEADER_SIZE, sizeof(ContinuousTensor)
                    );
                    tensor_key = first.data;
                }
                {
                    std::lock_guard<std::mutex> lk(dispatched_mu);
                    dispatched.push_back({callable, tensor_key});
                }
                is_running.store(true, std::memory_order_release);

                {
                    std::unique_lock<std::mutex> lk(run_mu);
                    run_cv.wait(lk, [this] {
                        return should_complete.load(std::memory_order_acquire);
                    });
                    should_complete.store(false, std::memory_order_relaxed);
                }
                is_running.store(false, std::memory_order_release);

                int32_t zero_err = 0;
                std::memcpy(mailbox.data() + MAILBOX_OFF_ERROR, &zero_err, sizeof(int32_t));
                std::memset(mailbox.data() + MAILBOX_OFF_ERROR_MSG, 0, MAILBOX_ERROR_MSG_SIZE);
                write_state(MailboxState::TASK_DONE);
            } else if (s == MailboxState::CONTROL_REQUEST) {
                // Acknowledge the control request so a future test using
                // WorkerThread::control_* doesn't hang on the spin-poll.
                // No memory operation is simulated — result stays zero.
                int32_t zero_err = 0;
                std::memcpy(mailbox.data() + MAILBOX_OFF_ERROR, &zero_err, sizeof(int32_t));
                std::memset(mailbox.data() + MAILBOX_OFF_ERROR_MSG, 0, MAILBOX_ERROR_MSG_SIZE);
                uint64_t zero_result = 0;
                std::memcpy(mailbox.data() + CTRL_OFF_RESULT, &zero_result, sizeof(uint64_t));
                write_state(MailboxState::CONTROL_DONE);
            } else if (s == MailboxState::SHUTDOWN) {
                return;
            } else {
                std::this_thread::sleep_for(std::chrono::microseconds(50));
            }
        }
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
    MockMailboxWorker mock_worker;
    WorkerManager manager;
    Scheduler sched;
    CallConfig cfg;

    std::vector<TaskSlot> consumed_slots;
    std::mutex consumed_mu;

    TaskSlotState &S(TaskSlot id) { return *allocator.slot_state(id); }

    void SetUp() override {
        allocator.init(/*heap_bytes=*/1ULL << 20);
        orch.init(&tm, &allocator, &scope, &rq_next_level, &rq_sub);

        mock_worker.start();
        manager.add_next_level(mock_worker.mailbox_ptr());
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
    auto res = orch.submit_next_level(42, args_a, cfg);
    TaskSlot slot = res.task_slot;

    mock_worker.wait_running();
    ASSERT_GE(mock_worker.dispatched_count(), 1);
    EXPECT_EQ(mock_worker.dispatched[0].tensor_key, 0xCAFEu);
    EXPECT_EQ(mock_worker.dispatched[0].callable, 42u);

    mock_worker.complete();
    wait_consumed(slot);
}

TEST_F(SchedulerFixture, DependentTaskDispatchedAfterProducerCompletes) {
    auto args_a = single_tensor_args(0xBEEF, TensorArgType::OUTPUT);
    auto a = orch.submit_next_level(10, args_a, cfg);

    auto args_b = single_tensor_args(0xBEEF, TensorArgType::INPUT);
    auto b = orch.submit_next_level(11, args_b, cfg);
    EXPECT_EQ(S(b.task_slot).state.load(), TaskState::PENDING);

    mock_worker.wait_running();
    EXPECT_EQ(mock_worker.dispatched[0].callable, 10u);
    mock_worker.complete();  // A done

    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(300);
    while (mock_worker.dispatched_count() < 2 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_GE(mock_worker.dispatched_count(), 2);
    EXPECT_EQ(mock_worker.dispatched[1].callable, 11u);

    mock_worker.complete();  // B done
    wait_consumed(b.task_slot);
    (void)a;
}

// ===========================================================================
// Group task tests -- fixture with 2 MockMailboxWorkers
// ===========================================================================

struct GroupSchedulerFixture : public ::testing::Test {
    TensorMap tm;
    Ring allocator;
    Scope scope;
    // Strict-4: per-type ready queues.
    ReadyQueue rq_next_level;
    ReadyQueue rq_sub;
    Orchestrator orch;
    MockMailboxWorker worker_a;
    MockMailboxWorker worker_b;
    WorkerManager manager;
    Scheduler sched;
    CallConfig cfg;

    std::vector<TaskSlot> consumed_slots;
    std::mutex consumed_mu;

    TaskSlotState &S(TaskSlot id) { return *allocator.slot_state(id); }

    void SetUp() override {
        allocator.init(/*heap_bytes=*/1ULL << 20);
        orch.init(&tm, &allocator, &scope, &rq_next_level, &rq_sub);

        worker_a.start();
        worker_b.start();
        manager.add_next_level(worker_a.mailbox_ptr());
        manager.add_next_level(worker_b.mailbox_ptr());
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

    auto res = orch.submit_next_level_group(42, {a0, a1}, cfg);
    TaskSlot slot = res.task_slot;

    worker_a.wait_running();
    worker_b.wait_running();

    EXPECT_EQ(worker_a.dispatched_count(), 1);
    EXPECT_EQ(worker_b.dispatched_count(), 1);

    // Each worker got a different TaskArgs from the slot's task_args_list —
    // proven by the keys 0xA0 and 0xA1 each landing on exactly one worker.
    uint64_t keys[2] = {worker_a.dispatched[0].tensor_key, worker_b.dispatched[0].tensor_key};
    std::sort(std::begin(keys), std::end(keys));
    EXPECT_EQ(keys[0], 0xA0u);
    EXPECT_EQ(keys[1], 0xA1u);
    (void)slot;

    worker_a.complete();
    worker_b.complete();
    wait_consumed(slot);
}

TEST_F(GroupSchedulerFixture, GroupCompletesOnlyWhenAllDone) {
    TaskArgs a0 = single_tensor_args(0xB0, TensorArgType::OUTPUT);
    TaskArgs a1 = single_tensor_args(0xB1, TensorArgType::OUTPUT);
    auto res = orch.submit_next_level_group(42, {a0, a1}, cfg);
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
    MockMailboxWorker next_level_worker;
    MockMailboxWorker sub_worker;
    WorkerManager manager;
    Scheduler sched;
    CallConfig cfg;

    std::vector<TaskSlot> consumed_slots;
    std::mutex consumed_mu;

    TaskSlotState &S(TaskSlot id) { return *allocator.slot_state(id); }

    void SetUp() override {
        allocator.init(/*heap_bytes=*/1ULL << 20);
        orch.init(&tm, &allocator, &scope, &rq_next_level, &rq_sub);

        next_level_worker.start();
        sub_worker.start();
        manager.add_next_level(next_level_worker.mailbox_ptr());
        manager.add_sub(sub_worker.mailbox_ptr());
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
    auto chip = orch.submit_next_level(20, chip_args, cfg);
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
    auto a = orch.submit_next_level_group(42, {a0, a1}, cfg);

    auto args_b = single_tensor_args(0xCAFE, TensorArgType::INPUT);
    auto b = orch.submit_next_level(42, args_b, cfg);
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
