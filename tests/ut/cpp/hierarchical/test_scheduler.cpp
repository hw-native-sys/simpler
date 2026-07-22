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
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
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
// The production dispatch path writes (callable digest, config, args_blob) into a
// MAILBOX_SIZE-byte shared region and spin-polls TASK_DONE; the real child
// (`_chip_process_loop` in python/simpler/worker.py) decodes the mailbox and
// dispatches to a `ChipWorker`. For unit testing the Scheduler / WorkerManager
// state machine in isolation, we replace the forked child with a thread inside
// the test process that mimics the same handshake but blocks until the
// test thread releases it via `complete()`.
//
// API parity with the previous MockWorker:
//   - dispatched[i].callable_hash0 / .tensor_key — recorded on TASK_READY
//   - is_running                            — atomic flag the test polls
//   - wait_running()                        — spin-wait until is_running flips
//   - complete()                            — release the parked dispatch so
//                                             the loop writes TASK_DONE
// ---------------------------------------------------------------------------

struct MockMailboxWorker {
    struct Record {
        uint8_t callable_hash0;
        uint64_t tensor_key;  // first tensor's `data` field (unique per submit in tests)
    };

    alignas(8) std::array<char, MAILBOX_SIZE> mailbox{};
    std::vector<Record> dispatched;
    std::mutex dispatched_mu;

    std::mutex run_mu;
    std::condition_variable run_cv;
    std::atomic<bool> should_complete{false};
    int32_t next_error_code{0};
    std::string next_error_msg;
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
        // LocalMailboxEndpoint::run completes its spin-poll cleanly.
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
        next_error_code = 0;
        next_error_msg.clear();
        should_complete.store(true, std::memory_order_release);
        run_cv.notify_one();
    }

    void complete_with_error(std::string msg) {
        std::lock_guard<std::mutex> lk(run_mu);
        next_error_code = 1;
        next_error_msg = std::move(msg);
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
                uint8_t callable_hash0 = static_cast<uint8_t>(mailbox[MAILBOX_OFF_TASK_CALLABLE_HASH]);
                int32_t t_count = 0;
                std::memcpy(&t_count, mailbox.data() + MAILBOX_OFF_TASK_ARGS_BLOB, sizeof(int32_t));
                uint64_t tensor_key = 0;
                if (t_count > 0) {
                    Tensor first{};
                    std::memcpy(
                        &first, mailbox.data() + MAILBOX_OFF_TASK_ARGS_BLOB + TASK_ARGS_BLOB_HEADER_SIZE, sizeof(Tensor)
                    );
                    tensor_key = first.buffer.addr;
                }
                {
                    std::lock_guard<std::mutex> lk(dispatched_mu);
                    dispatched.push_back({callable_hash0, tensor_key});
                }
                is_running.store(true, std::memory_order_release);

                {
                    std::unique_lock<std::mutex> lk(run_mu);
                    run_cv.wait(lk, [this] {
                        return should_complete.load(std::memory_order_acquire);
                    });
                    should_complete.store(false, std::memory_order_relaxed);
                }
                int32_t error_code = 0;
                std::string error_msg;
                {
                    std::lock_guard<std::mutex> lk(run_mu);
                    error_code = next_error_code;
                    error_msg = std::move(next_error_msg);
                    next_error_code = 0;
                    next_error_msg.clear();
                }
                is_running.store(false, std::memory_order_release);

                std::memcpy(mailbox.data() + MAILBOX_OFF_ERROR, &error_code, sizeof(int32_t));
                std::memset(mailbox.data() + MAILBOX_OFF_ERROR_MSG, 0, MAILBOX_ERROR_MSG_SIZE);
                if (!error_msg.empty()) {
                    size_t n = std::min(error_msg.size(), MAILBOX_ERROR_MSG_SIZE - 1);
                    std::memcpy(mailbox.data() + MAILBOX_OFF_ERROR_MSG, error_msg.data(), n);
                }
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

class FakeEndpoint final : public WorkerEndpoint {
public:
    explicit FakeEndpoint(int32_t worker_id, std::atomic<int> *prepare_count = nullptr) :
        prepare_count_(prepare_count) {
        caps_.kind = WorkerEndpointKind::REMOTE_L3;
        caps_.worker_id = worker_id;
        caps_.remote = true;
        caps_.transport = "test-remote";
    }

    const WorkerEndpointCaps &caps() const override { return caps_; }

    WorkerCompletion run(Ring *ring, const WorkerDispatch &dispatch) override {
        (void)ring;
        WorkerCompletion completion;
        completion.task_slot = dispatch.task_slot;
        completion.group_index = dispatch.group_index;
        completion.outcome = EndpointOutcome::SUCCESS;
        return completion;
    }

    void control_prepare(const uint8_t *) override {
        if (prepare_count_ != nullptr) prepare_count_->fetch_add(1, std::memory_order_relaxed);
    }

private:
    WorkerEndpointCaps caps_;
    std::atomic<int> *prepare_count_{nullptr};
};

// ---------------------------------------------------------------------------
// Helper: build a TaskArgs whose only tensor has the given (data, tag).
// ---------------------------------------------------------------------------

static TaskArgs single_tensor_args(uint64_t data_ptr, TensorArgType tag) {
    TaskArgs a;
    Tensor t{};
    t.buffer.addr = data_ptr;
    t.ndims = 1;
    t.shapes[0] = 1;
    t.dtype = DataType::UINT8;
    a.add_tensor(t, tag);
    return a;
}

static CallableIdentity C(uint8_t seed) {
    CallableIdentity c;
    c.digest.fill(seed);
    return c;
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
    PerWorkerReadyQueues rq_next_level_single;
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

        mock_worker.start();
        manager.add_next_level(mock_worker.mailbox_ptr());
        manager.start(&allocator, [this](WorkerCompletion completion) {
            sched.worker_done(std::move(completion));
        });
        rq_next_level_single.reset(manager.next_level_worker_ids());
        orch.init(
            &tm, &allocator, &scope, &rq_next_level, &rq_sub, &rq_next_level_single, &manager,
            [this] { sched.notify_ready(); }
        );

        Scheduler::Config c;
        c.ring = &allocator;
        c.ready_next_level_queue = &rq_next_level;
        c.ready_sub_queue = &rq_sub;
        c.ready_next_level_single_queues = &rq_next_level_single;
        c.manager = &manager;
        c.enqueue_ready_cb = [this](TaskSlot slot) {
            orch.enqueue_ready(slot);
        };
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

TEST(WorkerManagerTest, StartRejectsDuplicateNextLevelWorkerId) {
    alignas(8) std::array<char, MAILBOX_SIZE> mailbox{};
    Ring allocator;
    allocator.init(/*heap_bytes=*/0);
    WorkerManager manager;

    manager.add_next_level(mailbox.data());
    manager.add_next_level_endpoint(std::make_unique<FakeEndpoint>(0));

    bool threw = false;
    try {
        manager.start(&allocator, [](WorkerCompletion) {});
    } catch (const std::runtime_error &e) {
        threw = true;
        EXPECT_NE(std::string(e.what()).find("duplicate NEXT_LEVEL worker_id 0"), std::string::npos);
    }

    manager.stop();
    allocator.shutdown();
    EXPECT_TRUE(threw);
}

TEST(WorkerManagerTest, ControlPrepareUsesStableNextLevelWorkerId) {
    Ring allocator;
    allocator.init(/*heap_bytes=*/0);
    WorkerManager manager;
    std::atomic<int> worker7_prepares{0};
    std::atomic<int> worker3_prepares{0};

    manager.add_next_level_endpoint(std::make_unique<FakeEndpoint>(7, &worker7_prepares));
    manager.add_next_level_endpoint(std::make_unique<FakeEndpoint>(3, &worker3_prepares));
    manager.start(&allocator, [](WorkerCompletion) {});

    std::array<uint8_t, CALLABLE_HASH_DIGEST_SIZE> digest{};
    manager.control_prepare(3, digest.data());

    manager.stop();
    allocator.shutdown();
    EXPECT_EQ(worker7_prepares.load(std::memory_order_relaxed), 0);
    EXPECT_EQ(worker3_prepares.load(std::memory_order_relaxed), 1);
}

TEST_F(SchedulerFixture, IndependentTaskDispatchedAndConsumed) {
    auto args_a = single_tensor_args(0xCAFE, TensorArgType::OUTPUT);
    auto res = orch.submit_next_level(C(42), args_a, cfg, 0);
    TaskSlot slot = res.task_slot;

    mock_worker.wait_running();
    ASSERT_GE(mock_worker.dispatched_count(), 1);
    EXPECT_EQ(mock_worker.dispatched[0].tensor_key, 0xCAFEu);
    EXPECT_EQ(mock_worker.dispatched[0].callable_hash0, 42u);

    mock_worker.complete();
    wait_consumed(slot);
}

TEST_F(SchedulerFixture, DependentTaskDispatchedAfterProducerCompletes) {
    auto args_a = single_tensor_args(0xBEEF, TensorArgType::OUTPUT);
    auto a = orch.submit_next_level(C(10), args_a, cfg, 0);

    auto args_b = single_tensor_args(0xBEEF, TensorArgType::INPUT);
    auto b = orch.submit_next_level(C(11), args_b, cfg, 0);
    EXPECT_EQ(S(b.task_slot).state.load(), TaskState::PENDING);

    mock_worker.wait_running();
    EXPECT_EQ(mock_worker.dispatched[0].callable_hash0, 10u);
    mock_worker.complete();  // A done

    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(300);
    while (mock_worker.dispatched_count() < 2 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_GE(mock_worker.dispatched_count(), 2);
    EXPECT_EQ(mock_worker.dispatched[1].callable_hash0, 11u);

    mock_worker.complete();  // B done
    wait_consumed(b.task_slot);
    (void)a;
}

// Issue #1024: composed child kernels can carry far more tensor args than a
// top-level entry (repro: 76 tensors + 2 scalars = 3064-byte blob). The
// mailbox must hold any blob the runtime itself accepts, i.e. up to
// CHIP_MAX_TENSOR_ARGS / CHIP_MAX_SCALAR_ARGS.
TEST_F(SchedulerFixture, ComposedKernelArgsBlobFitsMailbox) {
    constexpr size_t max_blob = TASK_ARGS_BLOB_HEADER_SIZE +
                                static_cast<size_t>(CHIP_MAX_TENSOR_ARGS) * sizeof(Tensor) +
                                static_cast<size_t>(CHIP_MAX_SCALAR_ARGS) * sizeof(uint64_t);
    EXPECT_GE(MAILBOX_ARGS_CAPACITY, max_blob);

    TaskArgs args;
    for (int i = 0; i < 76; ++i) {
        Tensor t{};
        t.buffer.addr = 0x1000u + static_cast<uint64_t>(i) * 0x100u;
        t.ndims = 1;
        t.shapes[0] = 1;
        t.dtype = DataType::UINT8;
        args.add_tensor(t, TensorArgType::OUTPUT);
    }
    args.add_scalar(1);
    args.add_scalar(2);

    auto res = orch.submit_next_level(C(76), args, cfg, 0);

    mock_worker.wait_running();
    ASSERT_GE(mock_worker.dispatched_count(), 1)
        << "dispatch never reached the child: args blob exceeds mailbox capacity";
    EXPECT_EQ(mock_worker.dispatched[0].callable_hash0, 76u);

    mock_worker.complete();
    wait_consumed(res.task_slot);
}

TEST_F(SchedulerFixture, FailedProducerPoisonsDependentTask) {
    auto args_a = single_tensor_args(0xD00D, TensorArgType::OUTPUT);
    auto a = orch.submit_next_level(C(21), args_a, cfg, 0);

    auto args_b = single_tensor_args(0xD00D, TensorArgType::INPUT);
    auto b = orch.submit_next_level(C(22), args_b, cfg, 0);
    EXPECT_EQ(S(b.task_slot).state.load(), TaskState::PENDING);

    mock_worker.wait_running();
    ASSERT_EQ(mock_worker.dispatched_count(), 1);
    EXPECT_EQ(mock_worker.dispatched[0].callable_hash0, 21u);

    mock_worker.complete_with_error("root task boom");

    wait_consumed(a.task_slot);
    wait_consumed(b.task_slot);
    EXPECT_TRUE(manager.has_error());
    EXPECT_EQ(mock_worker.dispatched_count(), 1) << "poisoned consumer must not dispatch";
    EXPECT_EQ(S(a.task_slot).state.load(), TaskState::CONSUMED);
    EXPECT_EQ(S(b.task_slot).state.load(), TaskState::CONSUMED);
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
    PerWorkerReadyQueues rq_next_level_single;
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

        worker_a.start();
        worker_b.start();
        manager.add_next_level(worker_a.mailbox_ptr());
        manager.add_next_level(worker_b.mailbox_ptr());
        manager.start(&allocator, [this](WorkerCompletion completion) {
            sched.worker_done(std::move(completion));
        });
        rq_next_level_single.reset(manager.next_level_worker_ids());
        orch.init(
            &tm, &allocator, &scope, &rq_next_level, &rq_sub, &rq_next_level_single, &manager,
            [this] { sched.notify_ready(); }
        );

        Scheduler::Config c;
        c.ring = &allocator;
        c.ready_next_level_queue = &rq_next_level;
        c.ready_sub_queue = &rq_sub;
        c.ready_next_level_single_queues = &rq_next_level_single;
        c.manager = &manager;
        c.enqueue_ready_cb = [this](TaskSlot slot) {
            orch.enqueue_ready(slot);
        };
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

    auto res = orch.submit_next_level_group(C(42), {a0, a1}, cfg, {0, 1});
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
    auto res = orch.submit_next_level_group(C(42), {a0, a1}, cfg, {0, 1});
    TaskSlot slot = res.task_slot;

    worker_a.wait_running();
    worker_b.wait_running();

    worker_a.complete();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_EQ(S(slot).state.load(), TaskState::RUNNING);

    worker_b.complete();
    wait_consumed(slot);
}

TEST_F(GroupSchedulerFixture, GroupFailureWaitsForRunningMembersThenConsumes) {
    TaskArgs a0 = single_tensor_args(0xC0, TensorArgType::OUTPUT);
    TaskArgs a1 = single_tensor_args(0xC1, TensorArgType::OUTPUT);
    auto res = orch.submit_next_level_group(C(42), {a0, a1}, cfg, {0, 1});
    TaskSlot slot = res.task_slot;

    worker_a.wait_running();
    worker_b.wait_running();

    worker_a.complete_with_error("member boom");
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(300);
    while (!manager.has_error() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_TRUE(manager.has_error());
    EXPECT_EQ(S(slot).state.load(), TaskState::RUNNING);

    worker_b.complete();
    wait_consumed(slot);
    EXPECT_EQ(S(slot).state.load(), TaskState::CONSUMED);
}

TEST_F(GroupSchedulerFixture, InvalidGroupIndexFailsAndConsumesGroup) {
    TaskArgs a0 = single_tensor_args(0xD0, TensorArgType::OUTPUT);
    TaskArgs a1 = single_tensor_args(0xD1, TensorArgType::OUTPUT);
    auto res = orch.submit_next_level_group(C(42), {a0, a1}, cfg, {0, 1});
    TaskSlot slot = res.task_slot;

    worker_a.wait_running();
    worker_b.wait_running();

    WorkerCompletion bad;
    bad.task_slot = slot;
    bad.group_index = 99;
    bad.outcome = EndpointOutcome::ENDPOINT_FAILURE;
    bad.error_message = "bad completion index";
    sched.worker_done(std::move(bad));

    wait_consumed(slot);
    EXPECT_EQ(S(slot).state.load(), TaskState::CONSUMED);

    worker_a.complete();
    worker_b.complete();
}

TEST_F(GroupSchedulerFixture, ExplicitTargetWithinEligibilityIsUsed) {
    TaskArgs args = single_tensor_args(0xE0, TensorArgType::OUTPUT);
    auto res = orch.submit_next_level(C(55), args, cfg, 1, {1});
    TaskSlot slot = res.task_slot;

    worker_b.wait_running();
    EXPECT_FALSE(worker_a.is_running.load());
    EXPECT_TRUE(worker_b.is_running.load());
    EXPECT_EQ(worker_a.dispatched_count(), 0);
    EXPECT_EQ(worker_b.dispatched_count(), 1);

    worker_b.complete();
    wait_consumed(slot);
}

TEST_F(GroupSchedulerFixture, BusyTargetDoesNotBlockAnotherWorkerQueue) {
    auto running_args = single_tensor_args(0xE4, TensorArgType::OUTPUT);
    auto running = orch.submit_next_level(C(62), running_args, cfg, 0);
    worker_a.wait_running();
    ASSERT_TRUE(worker_a.is_running.load());

    auto blocked_args = single_tensor_args(0xE5, TensorArgType::OUTPUT);
    auto blocked = orch.submit_next_level(C(63), blocked_args, cfg, 0);
    auto blocked_second_args = single_tensor_args(0xE8, TensorArgType::OUTPUT);
    auto blocked_second = orch.submit_next_level(C(67), blocked_second_args, cfg, 0);
    auto independent_args = single_tensor_args(0xE6, TensorArgType::OUTPUT);
    auto independent = orch.submit_next_level(C(64), independent_args, cfg, 1);

    worker_b.wait_running();
    ASSERT_TRUE(worker_b.is_running.load());
    EXPECT_EQ(worker_b.dispatched_count(), 1);
    EXPECT_EQ(worker_b.dispatched[0].callable_hash0, 64u);

    worker_b.complete();
    wait_consumed(independent.task_slot);
    worker_a.complete();

    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
    while (worker_a.dispatched_count() < 2 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_EQ(worker_a.dispatched_count(), 2);
    EXPECT_EQ(worker_a.dispatched[1].callable_hash0, 63u);
    worker_a.complete();

    deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
    while (worker_a.dispatched_count() < 3 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_EQ(worker_a.dispatched_count(), 3);
    EXPECT_EQ(worker_a.dispatched[2].callable_hash0, 67u);
    worker_a.complete();
    wait_consumed(running.task_slot);
    wait_consumed(blocked.task_slot);
    wait_consumed(blocked_second.task_slot);
}

TEST_F(GroupSchedulerFixture, DependencyReleaseUsesConsumerWorkerQueue) {
    auto producer_args = single_tensor_args(0xE7, TensorArgType::OUTPUT);
    auto producer = orch.submit_next_level(C(65), producer_args, cfg, 0);
    auto consumer_args = single_tensor_args(0xE7, TensorArgType::INPUT);
    auto consumer = orch.submit_next_level(C(66), consumer_args, cfg, 1);
    EXPECT_EQ(S(consumer.task_slot).state.load(), TaskState::PENDING);

    worker_a.wait_running();
    ASSERT_TRUE(worker_a.is_running.load());
    EXPECT_FALSE(worker_b.is_running.load());
    worker_a.complete();

    worker_b.wait_running();
    ASSERT_TRUE(worker_b.is_running.load());
    EXPECT_EQ(worker_b.dispatched[0].callable_hash0, 66u);
    worker_b.complete();
    wait_consumed(producer.task_slot);
    wait_consumed(consumer.task_slot);
}

TEST_F(GroupSchedulerFixture, AffinityMustBeInEligibleEndpointSet) {
    TaskArgs args = single_tensor_args(0xE1, TensorArgType::OUTPUT);
    EXPECT_THROW((void)orch.submit_next_level(C(56), args, cfg, 0, {1}), std::invalid_argument);
}

TEST_F(GroupSchedulerFixture, UnknownEligibleWorkerIdIsRejectedBeforeScheduling) {
    TaskArgs args = single_tensor_args(0xE3, TensorArgType::OUTPUT);
    EXPECT_THROW((void)orch.submit_next_level(C(59), args, cfg, 99, {99}), std::invalid_argument);
}

TEST(SchedulerWorkerAffinityTest, NextLevelAffinityUsesWorkerIdNotVectorIndex) {
    TensorMap tm;
    Ring allocator;
    Scope scope;
    ReadyQueue rq_next_level;
    ReadyQueue rq_sub;
    PerWorkerReadyQueues rq_next_level_single;
    Orchestrator orch;
    MockMailboxWorker worker_a;
    MockMailboxWorker worker_b;
    WorkerManager manager;
    Scheduler sched;
    CallConfig cfg;
    std::vector<TaskSlot> consumed_slots;
    std::mutex consumed_mu;

    allocator.init(/*heap_bytes=*/1ULL << 20);
    worker_a.start();
    worker_b.start();
    manager.add_next_level_at(7, worker_a.mailbox_ptr());
    manager.add_next_level_at(9, worker_b.mailbox_ptr());
    manager.start(&allocator, [&sched](WorkerCompletion completion) {
        sched.worker_done(std::move(completion));
    });
    rq_next_level_single.reset(manager.next_level_worker_ids());
    orch.init(
        &tm, &allocator, &scope, &rq_next_level, &rq_sub, &rq_next_level_single, &manager,
        [&sched] { sched.notify_ready(); }
    );

    Scheduler::Config c;
    c.ring = &allocator;
    c.ready_next_level_queue = &rq_next_level;
    c.ready_sub_queue = &rq_sub;
    c.ready_next_level_single_queues = &rq_next_level_single;
    c.manager = &manager;
    c.enqueue_ready_cb = [&orch](TaskSlot slot) {
        orch.enqueue_ready(slot);
    };
    c.on_consumed_cb = [&orch, &consumed_slots, &consumed_mu](TaskSlot s) {
        orch.on_consumed(s);
        std::lock_guard<std::mutex> lk(consumed_mu);
        consumed_slots.push_back(s);
    };
    sched.start(c);

    auto wait_consumed_slot = [&consumed_slots, &consumed_mu](TaskSlot slot) {
        bool consumed = false;
        auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(1000);
        while (std::chrono::steady_clock::now() < deadline) {
            {
                std::lock_guard<std::mutex> lk(consumed_mu);
                consumed = std::find(consumed_slots.begin(), consumed_slots.end(), slot) != consumed_slots.end();
            }
            if (consumed) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        EXPECT_TRUE(consumed);
    };

    TaskArgs args = single_tensor_args(0xE2, TensorArgType::OUTPUT);
    auto res = orch.submit_next_level(C(58), args, cfg, 9);

    worker_b.wait_running();
    EXPECT_FALSE(worker_a.is_running.load());
    EXPECT_TRUE(worker_b.is_running.load());
    EXPECT_EQ(worker_a.dispatched_count(), 0);
    EXPECT_EQ(worker_b.dispatched_count(), 1);

    if (worker_a.is_running.load()) worker_a.complete();
    if (worker_b.is_running.load()) worker_b.complete();

    wait_consumed_slot(res.task_slot);

    TaskArgs a0 = single_tensor_args(0xE6, TensorArgType::OUTPUT);
    TaskArgs a1 = single_tensor_args(0xE7, TensorArgType::OUTPUT);
    auto group_res = orch.submit_next_level_group(C(61), {a0, a1}, cfg, {7, 9}, {{7}, {9}});

    worker_a.wait_running();
    worker_b.wait_running();
    EXPECT_TRUE(worker_a.is_running.load());
    EXPECT_TRUE(worker_b.is_running.load());
    EXPECT_EQ(worker_a.dispatched_count(), 1);
    EXPECT_EQ(worker_b.dispatched_count(), 2);

    worker_a.complete();
    worker_b.complete();
    wait_consumed_slot(group_res.task_slot);

    sched.stop();
    manager.stop();
    allocator.shutdown();
}

TEST_F(GroupSchedulerFixture, RemoteSidecarRejectsLocalEndpointEligibility) {
    TaskArgs args;
    Tensor tensor{};
    tensor.buffer.addr = 0;
    tensor.ndims = 1;
    tensor.shapes[0] = 1;
    tensor.dtype = DataType::UINT8;
    args.add_tensor(tensor, TensorArgType::OUTPUT);

    RemoteTaskArgsSidecar sidecar;
    sidecar.tensors.resize(1);
    sidecar.tensors[0].present = true;
    sidecar.tensors[0].desc.address_space = RemoteAddressSpace::REMOTE_DEVICE;
    sidecar.tensors[0].desc.owner_worker_id = 7;
    sidecar.tensors[0].desc.buffer_id = 11;
    sidecar.tensors[0].desc.generation = 1;
    sidecar.tensors[0].desc.nbytes = 1;

    EXPECT_THROW((void)orch.submit_next_level(C(57), args, cfg, 0, {0}, sidecar), std::invalid_argument);
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
    PerWorkerReadyQueues rq_next_level_single;
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

        next_level_worker.start();
        sub_worker.start();
        manager.add_next_level(next_level_worker.mailbox_ptr());
        manager.add_sub(sub_worker.mailbox_ptr());
        manager.start(&allocator, [this](WorkerCompletion completion) {
            sched.worker_done(std::move(completion));
        });
        rq_next_level_single.reset(manager.next_level_worker_ids());
        orch.init(
            &tm, &allocator, &scope, &rq_next_level, &rq_sub, &rq_next_level_single, &manager,
            [this] { sched.notify_ready(); }
        );

        Scheduler::Config c;
        c.ring = &allocator;
        c.ready_next_level_queue = &rq_next_level;
        c.ready_sub_queue = &rq_sub;
        c.ready_next_level_single_queues = &rq_next_level_single;
        c.manager = &manager;
        c.enqueue_ready_cb = [this](TaskSlot slot) {
            orch.enqueue_ready(slot);
        };
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
    auto chip = orch.submit_next_level(C(20), chip_args, cfg, 0);
    next_level_worker.wait_running();
    ASSERT_TRUE(next_level_worker.is_running.load());

    // Now submit a sub task while the chip pool is saturated. With a single
    // shared ready queue this would block behind any next-level task sitting
    // at the queue head waiting for a free chip worker. With per-type
    // queues (Strict-4) it must dispatch immediately to the idle sub
    // worker.
    auto sub_args = single_tensor_args(0xBBB, TensorArgType::OUTPUT);
    auto sub = orch.submit_sub(C(7), sub_args);

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
    auto a = orch.submit_next_level_group(C(42), {a0, a1}, cfg, {0, 1});

    auto args_b = single_tensor_args(0xCAFE, TensorArgType::INPUT);
    auto b = orch.submit_next_level(C(42), args_b, cfg, 0);
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
