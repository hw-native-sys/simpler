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
#include <poll.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <future>
#include <iterator>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "call_config.h"
#include "common/host_span.h"
#include "host_trace.h"
#include "orchestrator.h"
#include "ring.h"
#include "scheduler.h"
#include "scope.h"
#include "tensormap.h"
#include "types.h"
#include "worker_manager.h"
#include "task_args.h"

namespace {

std::mutex captured_host_spans_mu;
std::vector<std::string> captured_host_span_names;

void reset_captured_host_spans() {
    std::lock_guard<std::mutex> lk(captured_host_spans_mu);
    captured_host_span_names.clear();
}

bool captured_host_span(const std::string &name) {
    std::lock_guard<std::mutex> lk(captured_host_spans_mu);
    return std::find(captured_host_span_names.begin(), captured_host_span_names.end(), name) !=
           captured_host_span_names.end();
}

}  // namespace

extern "C" void simpler_log_emit_host_span(const SimplerHostSpan *span) {
    if (span == nullptr || span->name == nullptr) return;
    std::lock_guard<std::mutex> lk(captured_host_spans_mu);
    captured_host_span_names.emplace_back(span->name);
}

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
    std::atomic<bool> drain_mode{false};
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

    // Persistent teardown mode: every dispatch — including one arriving after
    // this call — completes itself, so Scheduler::stop() can always join.
    void drain() {
        drain_mode.store(true, std::memory_order_release);
        std::lock_guard<std::mutex> lk(run_mu);
        should_complete.store(true, std::memory_order_release);
        run_cv.notify_one();
    }

    // The child publishes acceptance into the sticky word, not the state.
    void write_task_accepted() {
        auto *ptr = reinterpret_cast<int32_t *>(static_cast<char *>(mailbox_ptr()) + MAILBOX_OFF_ACCEPTED);
        int32_t v = MAILBOX_TASK_ACCEPTED;
        __atomic_store(ptr, &v, __ATOMIC_RELEASE);
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
                    ChipTensor first{};
                    std::memcpy(
                        &first, mailbox.data() + MAILBOX_OFF_TASK_ARGS_BLOB + TASK_ARGS_BLOB_HEADER_SIZE,
                        sizeof(ChipTensor)
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
                        return should_complete.load(std::memory_order_acquire) ||
                               drain_mode.load(std::memory_order_acquire);
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

class DeterministicProgressEndpoint final : public WorkerEndpoint {
public:
    explicit DeterministicProgressEndpoint(int32_t worker_id = 0, uint32_t max_inflight_tasks = 2) {
        caps_.worker_id = worker_id;
        caps_.max_inflight_tasks = max_inflight_tasks;
        caps_.supports_frame_staging = true;
    }

    const WorkerEndpointCaps &caps() const override { return caps_; }
    bool progressable() const override { return true; }

    WorkerCompletion run(Ring *, const WorkerDispatch &) override {
        throw std::logic_error("progress endpoint must not use blocking run");
    }

    void submit_progress(Ring *ring, const WorkerDispatch &dispatch) override {
        ProgressCall call(*this);
        RunId run_id = INVALID_RUN_ID;
        if (ring != nullptr) {
            TaskSlotState *slot = ring->slot_state(dispatch.task_slot);
            if (slot != nullptr) run_id = slot->run_id;
        }
        std::lock_guard<std::mutex> lk(mu_);
        submitted_.push_back(dispatch);
        outstanding_.emplace(dispatch.dispatch_id, Outstanding{dispatch, run_id});
        cv_.notify_all();
    }

    bool poll_progress(WorkerEndpointProgress &progress) override {
        ProgressCall call(*this);
        std::lock_guard<std::mutex> lk(mu_);
        if (throw_poll_once_) {
            throw_poll_once_ = false;
            throw std::runtime_error("injected poll failure");
        }
        if (events_.empty()) return false;
        progress = std::move(events_.front());
        events_.pop_front();
        if (progress.kind == WorkerProgressKind::COMPLETED) {
            outstanding_.erase(progress.dispatch.dispatch_id);
        }
        return true;
    }

    bool activate_progress(RunId run_id) override {
        ProgressCall call(*this);
        std::lock_guard<std::mutex> lk(mu_);
        for (const auto &[dispatch_id, outstanding] : outstanding_) {
            (void)dispatch_id;
            if (outstanding.dispatch.prepare_only && outstanding.run_id == run_id) {
                activated_runs_.insert(run_id);
                cv_.notify_all();
                return true;
            }
        }
        return false;
    }

    void request_progress_stop() noexcept override {
        ProgressCall call(*this);
        std::lock_guard<std::mutex> lk(mu_);
        stop_requested_ = true;
        ++stop_request_count_;
        if (stop_request_count_ >= stop_terminalization_request_) terminalize_outstanding_locked();
        cv_.notify_all();
    }

    void report_progress_error(const std::string &reason) override {
        ProgressCall call(*this);
        std::lock_guard<std::mutex> lk(mu_);
        progress_error_ = reason;
        terminalize_outstanding_locked();
        cv_.notify_all();
    }

    bool wait_submitted(size_t count, std::chrono::milliseconds timeout = std::chrono::seconds(3)) {
        std::unique_lock<std::mutex> lk(mu_);
        return cv_.wait_for(lk, timeout, [this, count] {
            return submitted_.size() >= count;
        });
    }

    bool wait_activated(RunId run_id, std::chrono::milliseconds timeout = std::chrono::seconds(3)) {
        std::unique_lock<std::mutex> lk(mu_);
        return cv_.wait_for(lk, timeout, [this, run_id] {
            return activated_runs_.count(run_id) != 0;
        });
    }

    bool wait_stop_requested(std::chrono::milliseconds timeout = std::chrono::seconds(3)) {
        std::unique_lock<std::mutex> lk(mu_);
        return cv_.wait_for(lk, timeout, [this] {
            return stop_requested_;
        });
    }

    void terminalize_after_stop_request(size_t request_count) {
        std::lock_guard<std::mutex> lk(mu_);
        stop_terminalization_request_ = request_count;
    }

    void throw_on_next_poll() {
        std::lock_guard<std::mutex> lk(mu_);
        throw_poll_once_ = true;
    }

    bool wait_progress_error(std::chrono::milliseconds timeout = std::chrono::seconds(3)) {
        std::unique_lock<std::mutex> lk(mu_);
        return cv_.wait_for(lk, timeout, [this] {
            return !progress_error_.empty();
        });
    }

    std::string progress_error() const {
        std::lock_guard<std::mutex> lk(mu_);
        return progress_error_;
    }

    size_t stop_request_count() const {
        std::lock_guard<std::mutex> lk(mu_);
        return stop_request_count_;
    }

    void force_stop_terminalization() {
        std::lock_guard<std::mutex> lk(mu_);
        terminalize_outstanding_locked();
        cv_.notify_all();
    }

    std::vector<WorkerDispatch> submitted() const {
        std::lock_guard<std::mutex> lk(mu_);
        return submitted_;
    }

    void emit(WorkerProgressKind kind, const WorkerDispatch &dispatch) {
        std::lock_guard<std::mutex> lk(mu_);
        WorkerEndpointProgress progress;
        progress.kind = kind;
        progress.dispatch = dispatch;
        if (kind == WorkerProgressKind::COMPLETED) {
            progress.completion =
                WorkerCompletion{dispatch.task_slot, dispatch.group_index, EndpointOutcome::SUCCESS, {}};
        }
        events_.push_back(std::move(progress));
        cv_.notify_all();
    }

    int max_concurrent_progress_calls() const { return max_concurrent_calls_.load(std::memory_order_acquire); }
    bool progress_owner_changed() const {
        std::lock_guard<std::mutex> lk(owner_mu_);
        return progress_owner_changed_;
    }

private:
    struct Outstanding {
        WorkerDispatch dispatch;
        RunId run_id{INVALID_RUN_ID};
    };

    class ProgressCall {
    public:
        explicit ProgressCall(DeterministicProgressEndpoint &endpoint) :
            endpoint_(endpoint) {
            endpoint_.enter_progress_call();
        }
        ~ProgressCall() { endpoint_.leave_progress_call(); }

    private:
        DeterministicProgressEndpoint &endpoint_;
    };

    void enter_progress_call() {
        int current = concurrent_calls_.fetch_add(1, std::memory_order_acq_rel) + 1;
        int observed = max_concurrent_calls_.load(std::memory_order_acquire);
        while (current > observed &&
               !max_concurrent_calls_.compare_exchange_weak(observed, current, std::memory_order_acq_rel)) {}
        std::lock_guard<std::mutex> lk(owner_mu_);
        if (progress_owner_ == std::thread::id{}) {
            progress_owner_ = std::this_thread::get_id();
        } else if (progress_owner_ != std::this_thread::get_id()) {
            progress_owner_changed_ = true;
        }
    }

    void leave_progress_call() { concurrent_calls_.fetch_sub(1, std::memory_order_acq_rel); }

    void terminalize_outstanding_locked() {
        if (stop_terminalized_) return;
        stop_terminalized_ = true;
        for (const auto &[dispatch_id, outstanding] : outstanding_) {
            (void)dispatch_id;
            WorkerEndpointProgress progress;
            progress.kind = WorkerProgressKind::COMPLETED;
            progress.dispatch = outstanding.dispatch;
            progress.completion = WorkerCompletion{
                outstanding.dispatch.task_slot, outstanding.dispatch.group_index, EndpointOutcome::ENDPOINT_FAILURE,
                "test endpoint stopped"
            };
            events_.push_back(std::move(progress));
        }
    }

    WorkerEndpointCaps caps_;
    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::vector<WorkerDispatch> submitted_;
    std::unordered_map<uint64_t, Outstanding> outstanding_;
    std::deque<WorkerEndpointProgress> events_;
    std::set<RunId> activated_runs_;
    bool stop_requested_{false};
    bool stop_terminalized_{false};
    size_t stop_request_count_{0};
    size_t stop_terminalization_request_{1};
    bool throw_poll_once_{false};
    std::string progress_error_;
    std::atomic<int> concurrent_calls_{0};
    std::atomic<int> max_concurrent_calls_{0};
    mutable std::mutex owner_mu_;
    std::thread::id progress_owner_{};
    bool progress_owner_changed_{false};
};

static char *test_task_frame(std::array<char, MAILBOX_SIZE> &mailbox, size_t index) {
    return mailbox.data() + (MAILBOX_FIRST_TASK_FRAME + index) * MAILBOX_FRAME_SIZE;
}

static MailboxState test_frame_state(const char *frame) {
    const auto *state = reinterpret_cast<const int32_t *>(frame + MAILBOX_OFF_STATE);
    return static_cast<MailboxState>(__atomic_load_n(state, __ATOMIC_ACQUIRE));
}

static void set_test_frame_state(char *frame, MailboxState state) {
    auto *wire_state = reinterpret_cast<int32_t *>(frame + MAILBOX_OFF_STATE);
    __atomic_store_n(wire_state, static_cast<int32_t>(state), __ATOMIC_RELEASE);
}

static void set_test_frame_accepted(char *frame) {
    auto *accepted = reinterpret_cast<int32_t *>(frame + MAILBOX_OFF_ACCEPTED);
    __atomic_store_n(accepted, MAILBOX_TASK_ACCEPTED, __ATOMIC_RELEASE);
}

static uint64_t test_frame_dispatch_id(const char *frame) {
    uint64_t dispatch_id = 0;
    std::memcpy(&dispatch_id, frame + MAILBOX_OFF_FRAME_DISPATCH_ID, sizeof(dispatch_id));
    return dispatch_id;
}

class ScopedChildProcess {
public:
    explicit ScopedChildProcess(pid_t pid) :
        pid_(pid) {}
    ScopedChildProcess(const ScopedChildProcess &) = delete;
    ScopedChildProcess &operator=(const ScopedChildProcess &) = delete;

    ~ScopedChildProcess() {
        if (pid_ <= 0) return;
        int status = 0;
        pid_t result = 0;
        do {
            result = waitpid(pid_, &status, WNOHANG);
        } while (result < 0 && errno == EINTR);
        if (result != 0) return;

        (void)kill(pid_, SIGKILL);
        do {
            result = waitpid(pid_, &status, 0);
        } while (result < 0 && errno == EINTR);
    }

private:
    pid_t pid_{-1};
};

static TaskSlot make_progress_slot(Ring &ring, RunId run_id, uint32_t pipeline_slot, uint64_t generation) {
    AllocResult allocation = ring.alloc(/*heap_bytes=*/0, /*depth=*/0);
    if (allocation.slot == INVALID_SLOT) return INVALID_SLOT;
    TaskSlotState *slot = ring.slot_state(allocation.slot);
    if (slot == nullptr) return INVALID_SLOT;
    slot->reset();
    slot->run_id = run_id;
    slot->pipeline_lease = PipelineSlotLease{pipeline_slot, 0, generation};
    return allocation.slot;
}

// ---------------------------------------------------------------------------
// Helper: build a TaskArgs whose only tensor has the given (data, tag).
// ---------------------------------------------------------------------------

static TaskArgs single_tensor_args(uint64_t data_ptr, TensorArgType tag) {
    TaskArgs a;
    ChipTensor t{};
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

// The claim is what makes "pop a READY slot" and "dispatch it" one decision.
// A cancelling run moves its unstarted slots out of READY and consumes them;
// anything the scheduler had already popped must lose the race rather than
// overwrite that state with RUNNING.
TEST(ClaimForDispatch, OnlyAReadySlotCanBeClaimed) {
    TaskSlotState s;

    s.state.store(TaskState::READY, std::memory_order_release);
    EXPECT_TRUE(claim_for_dispatch(s));
    EXPECT_EQ(s.state.load(std::memory_order_acquire), TaskState::RUNNING);

    // Already claimed by this scheduler: a second claim must not re-dispatch.
    EXPECT_FALSE(claim_for_dispatch(s));
    EXPECT_EQ(s.state.load(std::memory_order_acquire), TaskState::RUNNING);

    // Cancelled between the pop and here — the cancelling path owns it now.
    // BUILDING is in the list for a different reason: a slot whose submit has
    // not published it has no final args or fanin count to dispatch on.
    for (TaskState taken :
         {TaskState::FAILED, TaskState::COMPLETED, TaskState::CONSUMED, TaskState::PENDING, TaskState::BUILDING}) {
        s.state.store(taken, std::memory_order_release);
        EXPECT_FALSE(claim_for_dispatch(s)) << "claimed a slot in state " << static_cast<int>(taken);
        EXPECT_EQ(s.state.load(std::memory_order_acquire), taken) << "claim overwrote a state it did not own";
    }
}

// Every failing path — a device completion poisoning its consumers, a run
// cancellation, and a submit that wired onto a producer that had already
// failed — moves a task to FAILED through this one exchange, so exactly one of
// them writes the message and runs the propagation.
TEST(ClaimTaskFailure, ReportsThePriorStateAndIsWonOnce) {
    TaskSlotState s;

    for (TaskState claimable : {TaskState::PENDING, TaskState::READY, TaskState::BUILDING}) {
        s.state.store(claimable, std::memory_order_release);
        s.failure_message.clear();

        std::optional<TaskState> won = claim_task_failure(s, "first");
        ASSERT_TRUE(won.has_value()) << "refused a claimable slot in state " << static_cast<int>(claimable);
        EXPECT_EQ(*won, claimable) << "claim did not report the state it took the slot from";
        EXPECT_EQ(s.state.load(std::memory_order_acquire), TaskState::FAILED);
        EXPECT_EQ(s.failure_message, "first");
        EXPECT_EQ(s.failure_propagation_pending.load(std::memory_order_acquire), claimable == TaskState::BUILDING)
            << "only a mid-wiring claim may advertise propagation takeover debt";

        // A second path reaching the same slot must not overwrite the reason
        // the first one recorded.
        EXPECT_FALSE(claim_task_failure(s, "second").has_value());
        EXPECT_EQ(s.failure_message, "first");

        s.failure_propagation_pending.store(false, std::memory_order_release);
    }
}

TEST(MarkGroupMembersSkipped, RepairsBothBookkeepingVectorsAsOneTransaction) {
    TaskSlotState s;
    s.is_group_ = true;
    s.task_args_list.resize(2);
    s.group_member_states.assign(2, GroupMemberState::NOT_DISPATCHED);
    s.group_member_outcomes.clear();

    mark_group_members_skipped(s, "cancelled");

    ASSERT_EQ(s.group_member_states.size(), 2u);
    ASSERT_EQ(s.group_member_outcomes.size(), 2u);
    EXPECT_EQ(s.group_member_states[0], GroupMemberState::SKIPPED);
    EXPECT_EQ(s.group_member_states[1], GroupMemberState::SKIPPED);
    EXPECT_EQ(s.group_member_outcomes[0], EndpointOutcome::SKIPPED);
    EXPECT_EQ(s.group_member_outcomes[1], EndpointOutcome::SKIPPED);
    EXPECT_EQ(s.group_terminal_count.load(std::memory_order_acquire), 2);
}

// Readiness is one decision over two fields that different threads own.
// Judging the count outside the transition lets a producer pass a comparison
// against a count submit has not published, then act on it after submit has —
// dispatching a task whose remaining producers are still running.
TEST(TryMarkReady, JudgesThePublishedCountNotTheOneItArrivedWith) {
    TaskSlotState s;
    s.state.store(TaskState::BUILDING, std::memory_order_release);

    // A producer completes while the slot is still building. The count it can
    // see is zero, which any release count passes — but nothing is readiable
    // yet, and the release is not lost either.
    s.fanin_released.store(1, std::memory_order_release);
    EXPECT_FALSE(try_mark_ready(s));

    // Submit publishes two live producers alongside the transition.
    {
        std::lock_guard<std::mutex> lk(s.fanout_mu);
        s.fanin_count.store(2, std::memory_order_release);
        s.state.store(TaskState::PENDING, std::memory_order_release);
    }
    EXPECT_FALSE(try_mark_ready(s)) << "one of two producers released and the task was marked ready";

    s.fanin_released.store(2, std::memory_order_release);
    EXPECT_TRUE(try_mark_ready(s));
    EXPECT_EQ(s.state.load(std::memory_order_acquire), TaskState::READY);

    // Exactly one caller owns the enqueue.
    EXPECT_FALSE(try_mark_ready(s));
}

// The publication is not a moment another thread can slip through. A producer
// that has already completed is held outside it for its whole duration, so
// there is no instant at which the slot is PENDING with a count only half the
// deciders have seen — which is the state that dispatches a task whose
// remaining producers are still running.
TEST(TryMarkReady, NoProducerTransitionsTheSlotWhileThePublicationHoldsIt) {
    TaskSlotState s;
    s.state.store(TaskState::BUILDING, std::memory_order_release);
    s.fanin_count.store(0, std::memory_order_release);

    std::atomic<bool> producer_marked_ready{false};
    std::unique_lock<std::mutex> publishing(s.fanout_mu);

    std::thread producer([&] {
        s.fanin_released.fetch_add(1, std::memory_order_acq_rel);
        producer_marked_ready.store(try_mark_ready(s), std::memory_order_release);
    });

    // The producer's release has landed and its decision is now in flight.
    while (s.fanin_released.load(std::memory_order_acquire) == 0)
        std::this_thread::yield();

    // Publish two live producers alongside the transition, then stay in the
    // critical section: a decider that judged the count before entering it
    // would take PENDING to READY right here.
    s.fanin_count.store(2, std::memory_order_release);
    s.state.store(TaskState::PENDING, std::memory_order_release);
    for (int i = 0; i < 1000; ++i)
        std::this_thread::yield();
    EXPECT_EQ(s.state.load(std::memory_order_acquire), TaskState::PENDING)
        << "a producer transitioned the slot from inside the publication";

    publishing.unlock();
    producer.join();
    EXPECT_FALSE(producer_marked_ready.load(std::memory_order_acquire));
    EXPECT_EQ(s.state.load(std::memory_order_acquire), TaskState::PENDING)
        << "a task was made ready with a live producer still running";
}

TEST(ClaimTaskFailure, RefusesASlotItDoesNotOwn) {
    TaskSlotState s;

    // RUNNING is the device's until its completion arrives; the rest are
    // already terminal, and resurrecting one would release its dependency
    // references a second time.
    for (TaskState owned :
         {TaskState::RUNNING, TaskState::COMPLETED, TaskState::FAILED, TaskState::CONSUMED, TaskState::FREE}) {
        s.state.store(owned, std::memory_order_release);
        s.failure_message.clear();
        EXPECT_FALSE(claim_task_failure(s, "cancelled").has_value())
            << "claimed a slot in state " << static_cast<int>(owned);
        EXPECT_EQ(s.state.load(std::memory_order_acquire), owned);
        EXPECT_TRUE(s.failure_message.empty());
    }
}

struct SchedulerFixture : public ::testing::Test {
    TensorMap tm;
    Ring allocator;
    Scope scope;
    ReadyQueue rq_sub;
    NextLevelReadyQueues rq_next_level;
    Orchestrator orch;
    MockMailboxWorker mock_worker;
    WorkerManager manager;
    Scheduler sched;
    CallConfig cfg;
    RunId run_id{INVALID_RUN_ID};

    std::vector<TaskSlot> consumed_slots;
    std::mutex consumed_mu;

    // Set by a test before the Scheduler reaches a claim; see
    // Scheduler::Config::before_claim_cb.
    std::function<void(TaskSlot)> before_claim_hook;

    TaskSlotState &S(TaskSlot id) { return *allocator.slot_state(id); }

    void SetUp() override {
        allocator.init(/*heap_bytes=*/1ULL << 20);

        mock_worker.start();
        manager.add_next_level(mock_worker.mailbox_ptr());
        manager.start(
            &allocator,
            [this](WorkerCompletion completion) {
                sched.worker_done(std::move(completion));
            },
            [this](WorkerDispatch dispatch) {
                orch.mark_task_accepted(dispatch.task_slot);
            },
            [this] {
                sched.notify_ready();
            }
        );
        rq_next_level.reset(manager.next_level_worker_ids());
        orch.init(&tm, &allocator, &scope, &rq_sub, &rq_next_level, &manager, [this] {
            sched.notify_ready();
        });
        run_id = orch.begin_run();

        Scheduler::Config c;
        c.ring = &allocator;
        c.ready_sub_queue = &rq_sub;
        c.ready_next_level_queues = &rq_next_level;
        c.manager = &manager;
        c.enqueue_ready_cb = [this](TaskSlot slot) {
            orch.enqueue_ready(slot);
        };
        // Same gate Worker::start installs: an active run that is also the
        // EXECUTING FIFO head and still owns its pipeline lease. Testing
        // against the weaker active_run_id() would let a slot dispatch here
        // that production refuses.
        c.active_run_cb = [this] {
            return orch.dispatchable_run_id();
        };
        c.on_consumed_cb = [this](TaskSlot s) {
            orch.on_consumed(s);
            std::lock_guard<std::mutex> lk(consumed_mu);
            consumed_slots.push_back(s);
        };
        c.on_task_failed_cb = [this](TaskSlot s, const std::string &message) {
            orch.report_task_error(s, message);
        };
        c.before_claim_cb = [this](TaskSlot slot) {
            if (before_claim_hook) before_claim_hook(slot);
        };
        sched.start(c);
    }

    void TearDown() override {
        mock_worker.drain();
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
        manager.start(&allocator, [](WorkerCompletion) {}, {});
    } catch (const std::runtime_error &e) {
        threw = true;
        EXPECT_NE(std::string(e.what()).find("duplicate NEXT_LEVEL worker_id 0"), std::string::npos);
    }

    manager.stop();
    allocator.shutdown();
    EXPECT_TRUE(threw);
}

TEST(WorkerManagerTest, SingleFrameSuccessorIsNotReportedAsStageable) {
    Ring allocator;
    allocator.init(/*heap_bytes=*/0);
    WorkerManager manager;
    manager.add_next_level_endpoint(std::make_unique<FakeEndpoint>(0));
    manager.start(&allocator, [](WorkerCompletion) {}, [](WorkerDispatch) {});
    NextLevelReadyQueues ready;
    ready.reset(manager.next_level_worker_ids());
    constexpr RunId successor_run = 17;
    ready.push_single(/*worker_id=*/0, successor_run, /*slot=*/3);

    EXPECT_FALSE(ready.singles_empty(successor_run));
    WorkerThread *worker = manager.get_worker_by_id(WorkerType::NEXT_LEVEL, 0);
    ASSERT_NE(worker, nullptr);
    EXPECT_FALSE(worker->can_stage());

    manager.stop();
    allocator.shutdown();
}

// The scheduler's wait is edge-triggered, so the wake it consumes for a
// finished dispatch has to imply the worker can take the next one. A
// completion cannot carry that: it is published before the lane state, on
// purpose, so a stopping scheduler never reads a worker as no longer busy
// while its last completion is still unqueued. These two pin the ordering the
// idle edge restores — one per endpoint driving mode.
TEST(WorkerManagerTest, IdleCallbackFollowsTheLanePublicationOnBlockingEndpoints) {
    Ring allocator;
    allocator.init(/*heap_bytes=*/0);
    TaskSlot slot = make_progress_slot(allocator, /*run_id=*/71, /*pipeline_slot=*/0, /*generation=*/1);
    ASSERT_NE(slot, INVALID_SLOT);

    WorkerManager manager;
    manager.add_next_level_endpoint(std::make_unique<FakeEndpoint>(0));

    std::mutex mu;
    std::condition_variable cv;
    int completions = 0;
    int idle_calls = 0;
    bool completion_seen_first = false;
    bool worker_readable_as_idle = false;

    manager.start(
        &allocator,
        [&](WorkerCompletion) {
            std::lock_guard<std::mutex> lk(mu);
            ++completions;
        },
        [](WorkerDispatch) {},
        [&] {
            WorkerThread *worker = manager.get_worker_by_id(WorkerType::NEXT_LEVEL, 0);
            std::lock_guard<std::mutex> lk(mu);
            ++idle_calls;
            completion_seen_first = completions == 1;
            worker_readable_as_idle = worker != nullptr && worker->idle() && !worker->busy();
            cv.notify_all();
        }
    );

    WorkerThread *worker = manager.get_worker_by_id(WorkerType::NEXT_LEVEL, 0);
    ASSERT_NE(worker, nullptr);
    worker->dispatch(WorkerDispatch{slot, 0});

    {
        std::unique_lock<std::mutex> lk(mu);
        ASSERT_TRUE(cv.wait_for(lk, std::chrono::seconds(3), [&] {
            return idle_calls == 1;
        })) << "the worker never signalled that its lane freed up";
        EXPECT_TRUE(completion_seen_first) << "the idle edge must not precede the completion it belongs to";
        EXPECT_TRUE(worker_readable_as_idle) << "a dispatch placed on this edge would find the worker occupied";
    }

    manager.stop();
    allocator.shutdown();
}

TEST(WorkerManagerTest, SingleFramePathEmitsFrameAndCompletionSpans) {
    ASSERT_TRUE(simpler::host_trace::bind_process_sink());
    reset_captured_host_spans();

    Ring allocator;
    allocator.init(/*heap_bytes=*/0);
    TaskSlot slot = make_progress_slot(allocator, /*run_id=*/73, /*pipeline_slot=*/0, /*generation=*/1);
    ASSERT_NE(slot, INVALID_SLOT);

    MockMailboxWorker child;
    child.start();
    WorkerThread worker;
    std::mutex completion_mu;
    std::condition_variable completion_cv;
    bool completed = false;
    worker.start(
        &allocator,
        [&](WorkerCompletion) {
            std::lock_guard<std::mutex> lk(completion_mu);
            completed = true;
            completion_cv.notify_all();
        },
        [](WorkerDispatch) {}, {}, std::make_unique<LocalMailboxEndpoint>(0, child.mailbox_ptr())
    );

    worker.dispatch(WorkerDispatch{slot, 0});
    child.wait_running();
    EXPECT_TRUE(child.is_running.load(std::memory_order_acquire));
    child.complete();
    {
        std::unique_lock<std::mutex> lk(completion_mu);
        EXPECT_TRUE(completion_cv.wait_for(lk, std::chrono::seconds(3), [&] {
            return completed;
        }));
    }

    worker.stop();
    EXPECT_TRUE(captured_host_span("l3.frame_submit"));
    EXPECT_TRUE(captured_host_span("l3.complete"));
    allocator.shutdown();
}

TEST(WorkerManagerTest, IdleCallbackFollowsTheLanePublicationOnProgressEndpoints) {
    Ring allocator;
    allocator.init(/*heap_bytes=*/0);
    TaskSlot slot = make_progress_slot(allocator, /*run_id=*/72, /*pipeline_slot=*/0, /*generation=*/1);
    ASSERT_NE(slot, INVALID_SLOT);

    WorkerThread worker;
    auto endpoint = std::make_unique<DeterministicProgressEndpoint>();
    DeterministicProgressEndpoint *endpoint_ptr = endpoint.get();

    std::mutex mu;
    std::condition_variable cv;
    int completions = 0;
    int idle_calls = 0;
    bool completion_seen_first = false;
    bool worker_readable_as_idle = false;

    worker.start(
        &allocator,
        [&](WorkerCompletion) {
            std::lock_guard<std::mutex> lk(mu);
            ++completions;
        },
        [](WorkerDispatch) {},
        [&] {
            std::lock_guard<std::mutex> lk(mu);
            ++idle_calls;
            completion_seen_first = completions == 1;
            worker_readable_as_idle = worker.idle() && !worker.busy();
            cv.notify_all();
        },
        std::move(endpoint)
    );

    worker.dispatch(WorkerDispatch{slot, 0});
    ASSERT_TRUE(endpoint_ptr->wait_submitted(1));
    endpoint_ptr->force_stop_terminalization();

    {
        std::unique_lock<std::mutex> lk(mu);
        ASSERT_TRUE(cv.wait_for(lk, std::chrono::seconds(3), [&] {
            return idle_calls == 1;
        })) << "the worker never signalled that its lane freed up";
        EXPECT_TRUE(completion_seen_first) << "the idle edge must not precede the completion it belongs to";
        EXPECT_TRUE(worker_readable_as_idle) << "a dispatch placed on this edge would find the worker occupied";
    }

    worker.stop();
    allocator.shutdown();
}

TEST(WorkerManagerTest, WorkerThreadUsesOneProgressOwnerForActiveAndStagedLanes) {
    Ring allocator;
    allocator.init(/*heap_bytes=*/0);
    TaskSlot active_slot = make_progress_slot(allocator, /*run_id=*/11, /*pipeline_slot=*/0, /*generation=*/1);
    TaskSlot staged_slot = make_progress_slot(allocator, /*run_id=*/12, /*pipeline_slot=*/1, /*generation=*/1);
    ASSERT_NE(active_slot, INVALID_SLOT);
    ASSERT_NE(staged_slot, INVALID_SLOT);

    WorkerThread worker;
    auto endpoint = std::make_unique<DeterministicProgressEndpoint>();
    DeterministicProgressEndpoint *endpoint_ptr = endpoint.get();
    std::mutex callback_mu;
    std::condition_variable callback_cv;
    std::vector<WorkerDispatch> accepted;
    std::vector<WorkerCompletion> completed;
    worker.start(
        &allocator,
        [&](WorkerCompletion completion) {
            std::lock_guard<std::mutex> lk(callback_mu);
            completed.push_back(std::move(completion));
            callback_cv.notify_all();
        },
        [&](WorkerDispatch dispatch) {
            std::lock_guard<std::mutex> lk(callback_mu);
            accepted.push_back(dispatch);
            callback_cv.notify_all();
        },
        {}, std::move(endpoint)
    );

    worker.dispatch(WorkerDispatch{active_slot, 0});
    worker.dispatch_prepared(WorkerDispatch{staged_slot, 0});
    EXPECT_TRUE(endpoint_ptr->wait_submitted(2));
    std::vector<WorkerDispatch> submitted = endpoint_ptr->submitted();
    ASSERT_EQ(submitted.size(), 2u);
    EXPECT_FALSE(submitted[0].prepare_only);
    EXPECT_TRUE(submitted[1].prepare_only);
    EXPECT_FALSE(worker.idle());
    EXPECT_FALSE(worker.can_stage());
    EXPECT_TRUE(worker.busy());

    endpoint_ptr->emit(WorkerProgressKind::ACCEPTED, submitted[0]);
    endpoint_ptr->emit(WorkerProgressKind::COMPLETED, submitted[0]);
    {
        std::unique_lock<std::mutex> lk(callback_mu);
        EXPECT_TRUE(callback_cv.wait_for(lk, std::chrono::seconds(3), [&] {
            return completed.size() >= 1;
        }));
    }
    EXPECT_TRUE(worker.idle());
    EXPECT_TRUE(worker.busy());
    EXPECT_TRUE(worker.activate_prepared(/*run_id=*/12));
    EXPECT_TRUE(endpoint_ptr->wait_activated(/*run_id=*/12));

    endpoint_ptr->emit(WorkerProgressKind::ACCEPTED, submitted[1]);
    endpoint_ptr->emit(WorkerProgressKind::COMPLETED, submitted[1]);
    {
        std::unique_lock<std::mutex> lk(callback_mu);
        EXPECT_TRUE(callback_cv.wait_for(lk, std::chrono::seconds(3), [&] {
            return accepted.size() == 2 && completed.size() == 2;
        }));
    }
    EXPECT_TRUE(worker.idle());
    EXPECT_FALSE(worker.busy());
    EXPECT_EQ(endpoint_ptr->max_concurrent_progress_calls(), 1);
    EXPECT_FALSE(endpoint_ptr->progress_owner_changed());
    worker.stop();
    allocator.shutdown();
}

TEST(WorkerManagerTest, AdmissionRejectionsCompleteClaimedDispatchesWithoutThrowing) {
    Ring allocator;
    allocator.init(/*heap_bytes=*/0);
    TaskSlot active_slot = make_progress_slot(allocator, /*run_id=*/13, /*pipeline_slot=*/0, /*generation=*/1);
    TaskSlot capacity_rejected = make_progress_slot(allocator, /*run_id=*/14, /*pipeline_slot=*/1, /*generation=*/1);
    TaskSlot lane_rejected = make_progress_slot(allocator, /*run_id=*/15, /*pipeline_slot=*/1, /*generation=*/2);
    TaskSlot stopping_rejected = make_progress_slot(allocator, /*run_id=*/16, /*pipeline_slot=*/0, /*generation=*/2);
    ASSERT_NE(active_slot, INVALID_SLOT);
    ASSERT_NE(capacity_rejected, INVALID_SLOT);
    ASSERT_NE(lane_rejected, INVALID_SLOT);
    ASSERT_NE(stopping_rejected, INVALID_SLOT);

    WorkerThread worker;
    auto endpoint = std::make_unique<DeterministicProgressEndpoint>(/*worker_id=*/0, /*max_inflight_tasks=*/1);
    DeterministicProgressEndpoint *endpoint_ptr = endpoint.get();
    std::mutex callback_mu;
    std::condition_variable callback_cv;
    std::vector<WorkerDispatch> accepted;
    std::vector<WorkerCompletion> completed;
    worker.start(
        &allocator,
        [&](WorkerCompletion completion) {
            std::lock_guard<std::mutex> lk(callback_mu);
            completed.push_back(std::move(completion));
            callback_cv.notify_all();
        },
        [&](WorkerDispatch dispatch) {
            std::lock_guard<std::mutex> lk(callback_mu);
            accepted.push_back(dispatch);
            callback_cv.notify_all();
        },
        {}, std::move(endpoint)
    );

    worker.dispatch(WorkerDispatch{active_slot, 0});
    ASSERT_TRUE(endpoint_ptr->wait_submitted(1));

    EXPECT_NO_THROW(worker.dispatch_prepared(WorkerDispatch{capacity_rejected, 0}));
    EXPECT_TRUE(worker.can_stage());
    EXPECT_NO_THROW(worker.dispatch(WorkerDispatch{lane_rejected, 0}));
    {
        std::unique_lock<std::mutex> lk(callback_mu);
        ASSERT_TRUE(callback_cv.wait_for(lk, std::chrono::seconds(3), [&] {
            return accepted.size() == 2 && completed.size() == 2;
        }));
        std::set<TaskSlot> accepted_slots;
        std::set<TaskSlot> completed_slots;
        for (const WorkerDispatch &dispatch : accepted) {
            accepted_slots.insert(dispatch.task_slot);
            EXPECT_EQ(dispatch.prepare_only, dispatch.task_slot == capacity_rejected);
        }
        for (const WorkerCompletion &completion : completed) {
            EXPECT_EQ(completion.outcome, EndpointOutcome::ENDPOINT_FAILURE);
            completed_slots.insert(completion.task_slot);
        }
        EXPECT_EQ(accepted_slots, (std::set<TaskSlot>{capacity_rejected, lane_rejected}));
        EXPECT_EQ(completed_slots, accepted_slots);
    }
    EXPECT_EQ(endpoint_ptr->submitted().size(), 1u);

    WorkerDispatch active = endpoint_ptr->submitted().front();
    endpoint_ptr->emit(WorkerProgressKind::ACCEPTED, active);
    endpoint_ptr->emit(WorkerProgressKind::COMPLETED, active);
    {
        std::unique_lock<std::mutex> lk(callback_mu);
        ASSERT_TRUE(callback_cv.wait_for(lk, std::chrono::seconds(3), [&] {
            return accepted.size() == 3 && completed.size() == 3;
        }));
    }
    worker.stop();

    EXPECT_NO_THROW(worker.dispatch(WorkerDispatch{stopping_rejected, 0}));
    {
        std::lock_guard<std::mutex> lk(callback_mu);
        ASSERT_EQ(accepted.size(), 4u);
        ASSERT_EQ(completed.size(), 4u);
        EXPECT_EQ(completed.back().outcome, EndpointOutcome::ENDPOINT_FAILURE);
        EXPECT_EQ(completed.back().task_slot, stopping_rejected);
    }
    allocator.shutdown();
}

TEST(WorkerManagerTest, TwoFrameLeaseSlotsDoNotDefineFifoOrAcceptance) {
    ASSERT_TRUE(simpler::host_trace::bind_process_sink());
    reset_captured_host_spans();

    alignas(8) std::array<char, MAILBOX_SIZE> mailbox{};
    Ring allocator;
    allocator.init(/*heap_bytes=*/0);
    TaskSlot active_slot = make_progress_slot(allocator, /*run_id=*/21, /*pipeline_slot=*/1, /*generation=*/7);
    TaskSlot staged_slot = make_progress_slot(allocator, /*run_id=*/22, /*pipeline_slot=*/0, /*generation=*/9);
    ASSERT_NE(active_slot, INVALID_SLOT);
    ASSERT_NE(staged_slot, INVALID_SLOT);

    LocalMailboxEndpoint endpoint(/*worker_id=*/0, mailbox.data(), /*child_pid=*/-1, /*task_frame_count=*/2);
    WorkerDispatch active{active_slot, 0, /*dispatch_id=*/41, /*prepare_only=*/false};
    WorkerDispatch staged{staged_slot, 0, /*dispatch_id=*/42, /*prepare_only=*/true};
    endpoint.submit_progress(&allocator, active);
    endpoint.submit_progress(&allocator, staged);

    char *lower_frame = test_task_frame(mailbox, 0);
    char *upper_frame = test_task_frame(mailbox, 1);
    EXPECT_EQ(test_frame_state(lower_frame), MailboxState::PREPARE_READY);
    EXPECT_EQ(test_frame_state(upper_frame), MailboxState::TASK_READY);
    EXPECT_EQ(test_frame_dispatch_id(lower_frame), 42u);
    EXPECT_EQ(test_frame_dispatch_id(upper_frame), 41u);

    EXPECT_TRUE(endpoint.activate_progress(/*run_id=*/22));
    EXPECT_EQ(test_frame_state(lower_frame), MailboxState::PREPARE_READY);
    WorkerEndpointProgress progress;
    EXPECT_FALSE(endpoint.poll_progress(progress));

    set_test_frame_state(lower_frame, MailboxState::FRAME_STAGED);
    ASSERT_TRUE(endpoint.poll_progress(progress));
    EXPECT_EQ(progress.kind, WorkerProgressKind::FRAME_STAGED);
    EXPECT_EQ(progress.dispatch.dispatch_id, 42u);
    EXPECT_EQ(test_frame_state(lower_frame), MailboxState::ACTIVATE);

    set_test_frame_state(lower_frame, MailboxState::TASK_LAUNCHED);
    EXPECT_FALSE(endpoint.poll_progress(progress));
    set_test_frame_accepted(lower_frame);
    ASSERT_TRUE(endpoint.poll_progress(progress));
    EXPECT_EQ(progress.kind, WorkerProgressKind::ACCEPTED);
    EXPECT_EQ(progress.dispatch.dispatch_id, 42u);
    EXPECT_TRUE(captured_host_span("l3.frame_submit"));
    EXPECT_TRUE(captured_host_span("l3.activate"));
    allocator.shutdown();
}

TEST(WorkerManagerTest, StaleActivationObservationCannotOverwriteTerminalState) {
    alignas(8) std::array<char, MAILBOX_FRAME_SIZE> frame{};
    set_test_frame_state(frame.data(), MailboxState::FRAME_STAGED);
    MailboxState stale_observation = test_frame_state(frame.data());
    ASSERT_EQ(stale_observation, MailboxState::FRAME_STAGED);

    set_test_frame_state(frame.data(), MailboxState::TASK_DONE);
    EXPECT_FALSE(mailbox_compare_exchange_state(frame.data(), stale_observation, MailboxState::ACTIVATE));
    EXPECT_EQ(test_frame_state(frame.data()), MailboxState::TASK_DONE);
}

TEST(WorkerManagerTest, ThirdDispatchCannotMutateTwoOccupiedFrames) {
    alignas(8) std::array<char, MAILBOX_SIZE> mailbox{};
    Ring allocator;
    allocator.init(/*heap_bytes=*/0);
    TaskSlot active_slot = make_progress_slot(allocator, /*run_id=*/31, /*pipeline_slot=*/0, /*generation=*/1);
    TaskSlot staged_slot = make_progress_slot(allocator, /*run_id=*/32, /*pipeline_slot=*/1, /*generation=*/1);
    TaskSlot third_slot = make_progress_slot(allocator, /*run_id=*/33, /*pipeline_slot=*/0, /*generation=*/1);
    TaskSlot fourth_slot = make_progress_slot(allocator, /*run_id=*/34, /*pipeline_slot=*/1, /*generation=*/2);
    ASSERT_NE(active_slot, INVALID_SLOT);
    ASSERT_NE(staged_slot, INVALID_SLOT);
    ASSERT_NE(third_slot, INVALID_SLOT);
    ASSERT_NE(fourth_slot, INVALID_SLOT);

    WorkerThread worker;
    std::mutex completion_mu;
    std::condition_variable completion_cv;
    int completion_count = 0;
    worker.start(
        &allocator,
        [&](WorkerCompletion) {
            std::lock_guard<std::mutex> lk(completion_mu);
            ++completion_count;
            completion_cv.notify_all();
        },
        [](WorkerDispatch) {}, {},
        std::make_unique<LocalMailboxEndpoint>(
            /*worker_id=*/0, mailbox.data(), /*child_pid=*/-1, /*task_frame_count=*/2
        )
    );
    worker.dispatch(WorkerDispatch{active_slot, 0});
    worker.dispatch_prepared(WorkerDispatch{staged_slot, 0});

    char *frame0 = test_task_frame(mailbox, 0);
    char *frame1 = test_task_frame(mailbox, 1);
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while ((test_frame_state(frame0) != MailboxState::TASK_READY ||
            test_frame_state(frame1) != MailboxState::PREPARE_READY) &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_EQ(test_frame_state(frame0), MailboxState::TASK_READY);
    EXPECT_EQ(test_frame_state(frame1), MailboxState::PREPARE_READY);
    const uint64_t first_dispatch_id = test_frame_dispatch_id(frame0);
    const uint64_t second_dispatch_id = test_frame_dispatch_id(frame1);

    EXPECT_NO_THROW(worker.dispatch(WorkerDispatch{third_slot, 0}));
    EXPECT_NO_THROW(worker.dispatch_prepared(WorkerDispatch{fourth_slot, 0}));
    EXPECT_EQ(test_frame_state(frame0), MailboxState::TASK_READY);
    EXPECT_EQ(test_frame_state(frame1), MailboxState::PREPARE_READY);
    EXPECT_EQ(test_frame_dispatch_id(frame0), first_dispatch_id);
    EXPECT_EQ(test_frame_dispatch_id(frame1), second_dispatch_id);
    {
        std::lock_guard<std::mutex> lk(completion_mu);
        EXPECT_EQ(completion_count, 2);
    }

    set_test_frame_accepted(frame0);
    set_test_frame_accepted(frame1);
    set_test_frame_state(frame0, MailboxState::TASK_DONE);
    set_test_frame_state(frame1, MailboxState::TASK_DONE);
    {
        std::unique_lock<std::mutex> lk(completion_mu);
        EXPECT_TRUE(completion_cv.wait_for(lk, std::chrono::seconds(3), [&] {
            return completion_count == 4;
        }));
    }
    worker.stop();
    allocator.shutdown();
}

TEST(WorkerManagerTest, StaleFrameIdentityWithholdsCompletionUntilChildQuiesces) {
    alignas(8) std::array<char, MAILBOX_SIZE> mailbox{};
    Ring allocator;
    allocator.init(/*heap_bytes=*/0);
    TaskSlot slot0 = make_progress_slot(allocator, /*run_id=*/41, /*pipeline_slot=*/0, /*generation=*/3);
    TaskSlot slot1 = make_progress_slot(allocator, /*run_id=*/42, /*pipeline_slot=*/1, /*generation=*/4);
    ASSERT_NE(slot0, INVALID_SLOT);
    ASSERT_NE(slot1, INVALID_SLOT);

    LocalMailboxEndpoint endpoint(/*worker_id=*/0, mailbox.data(), /*child_pid=*/-1, /*task_frame_count=*/2);
    endpoint.submit_progress(&allocator, WorkerDispatch{slot0, 0, /*dispatch_id=*/51, /*prepare_only=*/false});
    endpoint.submit_progress(&allocator, WorkerDispatch{slot1, 0, /*dispatch_id=*/52, /*prepare_only=*/true});

    char *frame1 = test_task_frame(mailbox, 1);
    uint64_t stale_generation = 99;
    std::memcpy(frame1 + MAILBOX_OFF_FRAME_GENERATION, &stale_generation, sizeof(stale_generation));
    set_test_frame_accepted(frame1);
    set_test_frame_state(frame1, MailboxState::TASK_LAUNCHED);

    WorkerEndpointProgress first;
    WorkerEndpointProgress second;
    EXPECT_FALSE(endpoint.poll_progress(first));
    EXPECT_EQ(test_frame_state(frame1), MailboxState::TASK_LAUNCHED);

    // Endpoint poison requests child shutdown, but parent-side completion must
    // not release the run lease until every native task has terminalized. A
    // real child publishes these states from its shutdown/finalize path.
    set_test_frame_state(test_task_frame(mailbox, 0), MailboxState::TASK_FAILED);
    set_test_frame_state(frame1, MailboxState::TASK_FAILED);

    ASSERT_TRUE(endpoint.poll_progress(first));
    ASSERT_TRUE(endpoint.poll_progress(second));
    EXPECT_EQ(first.kind, WorkerProgressKind::COMPLETED);
    EXPECT_EQ(second.kind, WorkerProgressKind::COMPLETED);
    EXPECT_EQ(first.completion.outcome, EndpointOutcome::ENDPOINT_FAILURE);
    EXPECT_EQ(second.completion.outcome, EndpointOutcome::ENDPOINT_FAILURE);
    EXPECT_NE(first.completion.error_message.find("stale frame identity"), std::string::npos);
    EXPECT_NE(second.completion.error_message.find("stale frame identity"), std::string::npos);
    EXPECT_EQ(
        std::set<TaskSlot>({first.dispatch.task_slot, second.dispatch.task_slot}), std::set<TaskSlot>({slot0, slot1})
    );
    EXPECT_THROW(
        endpoint.submit_progress(&allocator, WorkerDispatch{slot0, 0, /*dispatch_id=*/53, false}), std::runtime_error
    );
    allocator.shutdown();
}

TEST(WorkerManagerTest, PoisonedProgressWithholdsCompletionUntilTheChildExits) {
    Ring allocator;
    allocator.init(/*heap_bytes=*/0);
    TaskSlot slot0 = make_progress_slot(allocator, /*run_id=*/43, /*pipeline_slot=*/0, /*generation=*/5);
    TaskSlot slot1 = make_progress_slot(allocator, /*run_id=*/44, /*pipeline_slot=*/1, /*generation=*/6);
    ASSERT_NE(slot0, INVALID_SLOT);
    ASSERT_NE(slot1, INVALID_SLOT);

    void *mailbox =
        mmap(nullptr, MAILBOX_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, /*fd=*/-1, /*offset=*/0);
    ASSERT_NE(mailbox, MAP_FAILED);
    std::memset(mailbox, 0, MAILBOX_SIZE);

    int terminal_pipe[2] = {-1, -1};
    int release_pipe[2] = {-1, -1};
    ASSERT_EQ(pipe(terminal_pipe), 0);
    ASSERT_EQ(pipe(release_pipe), 0);

    pid_t child = fork();
    ASSERT_GE(child, 0);
    if (child == 0) {
        close(terminal_pipe[0]);
        close(release_pipe[1]);
        alarm(15);

        auto *base_state = reinterpret_cast<int32_t *>(static_cast<char *>(mailbox) + MAILBOX_OFF_STATE);
        while (static_cast<MailboxState>(__atomic_load_n(base_state, __ATOMIC_ACQUIRE)) != MailboxState::SHUTDOWN) {
            usleep(100);
        }

        char *frame0 = static_cast<char *>(mailbox) + MAILBOX_FIRST_TASK_FRAME * MAILBOX_FRAME_SIZE;
        char *frame1 = frame0 + MAILBOX_FRAME_SIZE;
        set_test_frame_state(frame0, MailboxState::TASK_FAILED);
        set_test_frame_state(frame1, MailboxState::TASK_FAILED);

        char terminal = 1;
        ssize_t written = 0;
        do {
            written = write(terminal_pipe[1], &terminal, sizeof(terminal));
        } while (written < 0 && errno == EINTR);
        if (written != sizeof(terminal)) _exit(2);

        char release = 0;
        ssize_t received = 0;
        do {
            received = read(release_pipe[0], &release, sizeof(release));
        } while (received < 0 && errno == EINTR);
        _exit(received == sizeof(release) ? 0 : 3);
    }

    ScopedChildProcess child_guard(child);
    close(terminal_pipe[1]);
    close(release_pipe[0]);

    {
        LocalMailboxEndpoint endpoint(/*worker_id=*/0, mailbox, static_cast<int>(child), /*task_frame_count=*/2);
        endpoint.submit_progress(&allocator, WorkerDispatch{slot0, 0, /*dispatch_id=*/53, /*prepare_only=*/false});
        endpoint.submit_progress(&allocator, WorkerDispatch{slot1, 0, /*dispatch_id=*/54, /*prepare_only=*/true});

        char *frame0 = static_cast<char *>(mailbox) + MAILBOX_FIRST_TASK_FRAME * MAILBOX_FRAME_SIZE;
        char *frame1 = frame0 + MAILBOX_FRAME_SIZE;
        uint64_t stale_generation = 100;
        std::memcpy(frame1 + MAILBOX_OFF_FRAME_GENERATION, &stale_generation, sizeof(stale_generation));
        set_test_frame_accepted(frame1);
        set_test_frame_state(frame1, MailboxState::TASK_LAUNCHED);

        WorkerEndpointProgress progress;
        EXPECT_FALSE(endpoint.poll_progress(progress));

        pollfd terminal_ready{terminal_pipe[0], POLLIN, 0};
        int poll_result = 0;
        do {
            poll_result = poll(&terminal_ready, 1, 3000);
        } while (poll_result < 0 && errno == EINTR);
        EXPECT_EQ(poll_result, 1) << "child did not terminalize both task frames after SHUTDOWN";
        char terminal = 0;
        ssize_t received = 0;
        if (poll_result == 1) {
            do {
                received = read(terminal_pipe[0], &terminal, sizeof(terminal));
            } while (received < 0 && errno == EINTR);
        }
        EXPECT_EQ(received, sizeof(terminal));
        EXPECT_EQ(test_frame_state(frame0), MailboxState::TASK_FAILED);
        EXPECT_EQ(test_frame_state(frame1), MailboxState::TASK_FAILED);
        EXPECT_FALSE(endpoint.poll_progress(progress))
            << "terminal frame states are insufficient while the native child remains alive";

        char release = 1;
        ssize_t written = 0;
        do {
            written = write(release_pipe[1], &release, sizeof(release));
        } while (written < 0 && errno == EINTR);
        EXPECT_EQ(written, sizeof(release));
        close(release_pipe[1]);
        release_pipe[1] = -1;

        std::vector<WorkerEndpointProgress> completions;
        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
        while (completions.size() < 2 && std::chrono::steady_clock::now() < deadline) {
            if (endpoint.poll_progress(progress)) {
                completions.push_back(progress);
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
        ASSERT_EQ(completions.size(), 2u);
        EXPECT_EQ(completions[0].kind, WorkerProgressKind::COMPLETED);
        EXPECT_EQ(completions[1].kind, WorkerProgressKind::COMPLETED);
        EXPECT_EQ(completions[0].completion.outcome, EndpointOutcome::ENDPOINT_FAILURE);
        EXPECT_EQ(completions[1].completion.outcome, EndpointOutcome::ENDPOINT_FAILURE);
        EXPECT_EQ(
            std::set<TaskSlot>({completions[0].dispatch.task_slot, completions[1].dispatch.task_slot}),
            std::set<TaskSlot>({slot0, slot1})
        );
    }

    close(terminal_pipe[0]);
    if (release_pipe[1] >= 0) close(release_pipe[1]);
    EXPECT_EQ(munmap(mailbox, MAILBOX_SIZE), 0);
    allocator.shutdown();
}

TEST(WorkerManagerTest, StopTerminalizesOutstandingProgress) {
    Ring allocator;
    allocator.init(/*heap_bytes=*/0);
    TaskSlot active_slot = make_progress_slot(allocator, /*run_id=*/51, /*pipeline_slot=*/0, /*generation=*/1);
    TaskSlot staged_slot = make_progress_slot(allocator, /*run_id=*/52, /*pipeline_slot=*/1, /*generation=*/1);
    ASSERT_NE(active_slot, INVALID_SLOT);
    ASSERT_NE(staged_slot, INVALID_SLOT);

    WorkerThread worker;
    auto endpoint = std::make_unique<DeterministicProgressEndpoint>();
    DeterministicProgressEndpoint *endpoint_ptr = endpoint.get();
    std::mutex completion_mu;
    std::condition_variable completion_cv;
    std::vector<WorkerCompletion> completed;
    worker.start(
        &allocator,
        [&](WorkerCompletion completion) {
            std::lock_guard<std::mutex> lk(completion_mu);
            completed.push_back(std::move(completion));
            completion_cv.notify_all();
        },
        [](WorkerDispatch) {}, {}, std::move(endpoint)
    );
    worker.dispatch(WorkerDispatch{active_slot, 0});
    worker.dispatch_prepared(WorkerDispatch{staged_slot, 0});
    EXPECT_TRUE(endpoint_ptr->wait_submitted(2));

    std::promise<void> stopped;
    std::future<void> stop_done = stopped.get_future();
    std::thread stopper([&] {
        worker.stop();
        stopped.set_value();
    });
    EXPECT_TRUE(endpoint_ptr->wait_stop_requested());
    std::future_status stop_status = stop_done.wait_for(std::chrono::seconds(3));
    EXPECT_EQ(stop_status, std::future_status::ready);
    if (stop_status != std::future_status::ready) endpoint_ptr->force_stop_terminalization();
    stopper.join();
    {
        std::lock_guard<std::mutex> lk(completion_mu);
        ASSERT_EQ(completed.size(), 2u);
        EXPECT_EQ(completed[0].outcome, EndpointOutcome::ENDPOINT_FAILURE);
        EXPECT_EQ(completed[1].outcome, EndpointOutcome::ENDPOINT_FAILURE);
    }
    EXPECT_FALSE(worker.busy());
    allocator.shutdown();
}

TEST(WorkerManagerTest, ProgressStopRepeatsUntilOutstandingWorkTerminalizes) {
    Ring allocator;
    allocator.init(/*heap_bytes=*/0);
    TaskSlot slot = make_progress_slot(allocator, /*run_id=*/53, /*pipeline_slot=*/0, /*generation=*/1);
    ASSERT_NE(slot, INVALID_SLOT);

    WorkerThread worker;
    auto endpoint = std::make_unique<DeterministicProgressEndpoint>();
    DeterministicProgressEndpoint *endpoint_ptr = endpoint.get();
    endpoint_ptr->terminalize_after_stop_request(2);
    std::mutex completion_mu;
    std::vector<WorkerCompletion> completed;
    worker.start(
        &allocator,
        [&](WorkerCompletion completion) {
            std::lock_guard<std::mutex> lk(completion_mu);
            completed.push_back(std::move(completion));
        },
        [](WorkerDispatch) {}, {}, std::move(endpoint)
    );
    worker.dispatch(WorkerDispatch{slot, 0});
    EXPECT_TRUE(endpoint_ptr->wait_submitted(1));

    std::promise<void> stopped;
    std::future<void> stop_done = stopped.get_future();
    std::thread stopper([&] {
        worker.stop();
        stopped.set_value();
    });
    std::future_status stop_status = stop_done.wait_for(std::chrono::seconds(3));
    EXPECT_EQ(stop_status, std::future_status::ready);
    if (stop_status != std::future_status::ready) endpoint_ptr->force_stop_terminalization();
    stopper.join();

    EXPECT_EQ(endpoint_ptr->stop_request_count(), 2u)
        << "the first stop request may be overwritten by an in-flight control completion";
    {
        std::lock_guard<std::mutex> lk(completion_mu);
        ASSERT_EQ(completed.size(), 1u);
        EXPECT_EQ(completed[0].outcome, EndpointOutcome::ENDPOINT_FAILURE);
    }
    EXPECT_FALSE(worker.busy());
    allocator.shutdown();
}

TEST(WorkerManagerTest, PollExceptionTerminalizesOutstandingProgressAndStopsTheDriver) {
    Ring allocator;
    allocator.init(/*heap_bytes=*/0);
    TaskSlot slot = make_progress_slot(allocator, /*run_id=*/54, /*pipeline_slot=*/0, /*generation=*/1);
    ASSERT_NE(slot, INVALID_SLOT);

    WorkerThread worker;
    auto endpoint = std::make_unique<DeterministicProgressEndpoint>();
    DeterministicProgressEndpoint *endpoint_ptr = endpoint.get();
    endpoint_ptr->throw_on_next_poll();
    std::mutex completion_mu;
    std::condition_variable completion_cv;
    std::vector<WorkerCompletion> completed;
    worker.start(
        &allocator,
        [&](WorkerCompletion completion) {
            std::lock_guard<std::mutex> lk(completion_mu);
            completed.push_back(std::move(completion));
            completion_cv.notify_all();
        },
        [](WorkerDispatch) {}, {}, std::move(endpoint)
    );
    worker.dispatch(WorkerDispatch{slot, 0});
    EXPECT_TRUE(endpoint_ptr->wait_submitted(1));
    EXPECT_TRUE(endpoint_ptr->wait_progress_error());

    bool completion_ready = false;
    {
        std::unique_lock<std::mutex> lk(completion_mu);
        completion_ready = completion_cv.wait_for(lk, std::chrono::seconds(3), [&] {
            return completed.size() == 1;
        });
    }
    EXPECT_TRUE(completion_ready);
    if (!completion_ready) endpoint_ptr->force_stop_terminalization();

    std::promise<void> stopped;
    std::future<void> stop_done = stopped.get_future();
    std::thread stopper([&] {
        worker.stop();
        stopped.set_value();
    });
    std::future_status stop_status = stop_done.wait_for(std::chrono::seconds(3));
    EXPECT_EQ(stop_status, std::future_status::ready);
    if (stop_status != std::future_status::ready) endpoint_ptr->force_stop_terminalization();
    stopper.join();

    EXPECT_EQ(endpoint_ptr->progress_error(), "poll_progress failed: injected poll failure");
    {
        std::lock_guard<std::mutex> lk(completion_mu);
        ASSERT_EQ(completed.size(), 1u);
        EXPECT_EQ(completed[0].outcome, EndpointOutcome::ENDPOINT_FAILURE);
    }
    EXPECT_FALSE(worker.busy());
    allocator.shutdown();
}

TEST(WorkerManagerTest, StopKeepsWorkerBusyUntilItsLastCompletionIsPublished) {
    Ring allocator;
    allocator.init(/*heap_bytes=*/0);
    TaskSlot slot = make_progress_slot(allocator, /*run_id=*/61, /*pipeline_slot=*/0, /*generation=*/1);
    ASSERT_NE(slot, INVALID_SLOT);
    allocator.slot_state(slot)->state.store(TaskState::RUNNING, std::memory_order_release);

    WorkerManager manager;
    auto endpoint = std::make_unique<DeterministicProgressEndpoint>();
    DeterministicProgressEndpoint *endpoint_ptr = endpoint.get();
    manager.add_next_level_endpoint(std::move(endpoint));
    Scheduler scheduler;
    std::promise<void> completion_entered;
    std::future<void> completion_entered_future = completion_entered.get_future();
    std::promise<void> allow_completion;
    std::shared_future<void> allow_completion_future = allow_completion.get_future().share();
    manager.start(
        &allocator,
        [&](WorkerCompletion completion) {
            completion_entered.set_value();
            allow_completion_future.wait();
            scheduler.worker_done(std::move(completion));
        },
        [](WorkerDispatch) {},
        [&scheduler] {
            scheduler.notify_ready();
        }
    );

    ReadyQueue ready_sub;
    NextLevelReadyQueues ready_next;
    ready_next.reset(manager.next_level_worker_ids());
    Scheduler::Config scheduler_config;
    scheduler_config.ring = &allocator;
    scheduler_config.ready_sub_queue = &ready_sub;
    scheduler_config.ready_next_level_queues = &ready_next;
    scheduler_config.manager = &manager;
    scheduler_config.enqueue_ready_cb = [](TaskSlot) {};
    scheduler.start(scheduler_config);

    WorkerThread *worker = manager.get_worker_by_id(WorkerType::NEXT_LEVEL, 0);
    ASSERT_NE(worker, nullptr);
    worker->dispatch(WorkerDispatch{slot, 0});
    ASSERT_TRUE(endpoint_ptr->wait_submitted(1));
    scheduler.request_stop();

    std::thread manager_stopper([&] {
        manager.stop_workers();
    });
    EXPECT_EQ(completion_entered_future.wait_for(std::chrono::seconds(3)), std::future_status::ready);

    std::promise<void> scheduler_stopped;
    std::future<void> scheduler_stopped_future = scheduler_stopped.get_future();
    std::thread scheduler_stopper([&] {
        scheduler.stop();
        scheduler_stopped.set_value();
    });
    EXPECT_EQ(scheduler_stopped_future.wait_for(std::chrono::milliseconds(50)), std::future_status::timeout)
        << "scheduler exited before the worker published its terminal completion";

    allow_completion.set_value();
    manager_stopper.join();
    scheduler_stopper.join();
    EXPECT_EQ(scheduler_stopped_future.wait_for(std::chrono::seconds(0)), std::future_status::ready);
    EXPECT_EQ(allocator.slot_state(slot)->state.load(std::memory_order_acquire), TaskState::FAILED);
    manager.stop();
    allocator.shutdown();
}

struct ProgressSchedulerFixture : public ::testing::Test {
    TensorMap tensor_map;
    Ring allocator;
    Scope scope;
    ReadyQueue ready_sub;
    NextLevelReadyQueues ready_next;
    Orchestrator orchestrator;
    WorkerManager manager;
    Scheduler scheduler;
    CallConfig config;
    DeterministicProgressEndpoint *endpoint0{nullptr};
    DeterministicProgressEndpoint *endpoint1{nullptr};

    virtual uint32_t endpoint0_capacity() const { return 2; }

    void SetUp() override {
        allocator.init(/*heap_bytes=*/1ULL << 20);
        auto first_endpoint = std::make_unique<DeterministicProgressEndpoint>(0, endpoint0_capacity());
        auto second_endpoint = std::make_unique<DeterministicProgressEndpoint>(1);
        endpoint0 = first_endpoint.get();
        endpoint1 = second_endpoint.get();
        manager.add_next_level_endpoint(std::move(first_endpoint));
        manager.add_next_level_endpoint(std::move(second_endpoint));
        manager.start(
            &allocator,
            [this](WorkerCompletion completion) {
                scheduler.worker_done(std::move(completion));
            },
            [this](WorkerDispatch dispatch) {
                orchestrator.mark_task_accepted(dispatch.task_slot);
            },
            [this] {
                scheduler.notify_ready();
            }
        );
        ready_next.reset(manager.next_level_worker_ids());
        orchestrator.init(&tensor_map, &allocator, &scope, &ready_sub, &ready_next, &manager, [this] {
            scheduler.notify_ready();
        });
        orchestrator.configure_pipeline_depth(2);

        Scheduler::Config scheduler_config;
        scheduler_config.ring = &allocator;
        scheduler_config.ready_sub_queue = &ready_sub;
        scheduler_config.ready_next_level_queues = &ready_next;
        scheduler_config.manager = &manager;
        scheduler_config.enqueue_ready_cb = [this](TaskSlot task_slot) {
            orchestrator.enqueue_ready(task_slot);
        };
        scheduler_config.active_run_cb = [this] {
            return orchestrator.dispatchable_run_id();
        };
        scheduler_config.preparable_run_cb = [this] {
            return orchestrator.preparable_run_id();
        };
        scheduler_config.on_consumed_cb = [this](TaskSlot task_slot) {
            orchestrator.on_consumed(task_slot);
        };
        scheduler_config.on_task_failed_cb = [this](TaskSlot task_slot, const std::string &message) {
            orchestrator.report_task_error(task_slot, message);
        };
        scheduler.start(scheduler_config);
    }

    void TearDown() override {
        scheduler.request_stop();
        manager.stop_workers();
        scheduler.stop();
        manager.stop();
        allocator.shutdown();
    }
};

struct CapacityOneProgressSchedulerFixture : public ProgressSchedulerFixture {
    uint32_t endpoint0_capacity() const override { return 1; }
};

TEST_F(ProgressSchedulerFixture, SuccessorStagesButActivatesOnlyAfterFifoPromotion) {
    RunId first_run = orchestrator.begin_run();
    SubmitResult first =
        orchestrator.submit_next_level(C(1), single_tensor_args(0x1000, TensorArgType::OUTPUT), config, 0);
    orchestrator.close_run_submission(first_run);
    // Pin the production ordering: the predecessor is already active before
    // closing the successor publishes the edge that makes it preparable.
    ASSERT_TRUE(endpoint0->wait_submitted(1));
    RunId second_run = orchestrator.begin_run();
    SubmitResult second =
        orchestrator.submit_next_level(C(2), single_tensor_args(0x2000, TensorArgType::OUTPUT), config, 0);
    orchestrator.close_run_submission(second_run);

    EXPECT_TRUE(endpoint0->wait_submitted(2));
    std::vector<WorkerDispatch> submitted = endpoint0->submitted();
    ASSERT_EQ(submitted.size(), 2u);
    auto first_dispatch = std::find_if(submitted.begin(), submitted.end(), [&](const WorkerDispatch &dispatch) {
        return dispatch.task_slot == first.task_slot;
    });
    auto second_dispatch = std::find_if(submitted.begin(), submitted.end(), [&](const WorkerDispatch &dispatch) {
        return dispatch.task_slot == second.task_slot;
    });
    ASSERT_NE(first_dispatch, submitted.end());
    ASSERT_NE(second_dispatch, submitted.end());
    EXPECT_FALSE(first_dispatch->prepare_only);
    EXPECT_TRUE(second_dispatch->prepare_only);
    EXPECT_EQ(orchestrator.active_run_id(), first_run);
    EXPECT_EQ(orchestrator.preparable_run_id(), second_run);
    EXPECT_FALSE(endpoint0->wait_activated(second_run, std::chrono::milliseconds(20)));

    endpoint0->emit(WorkerProgressKind::ACCEPTED, *first_dispatch);
    endpoint0->emit(WorkerProgressKind::COMPLETED, *first_dispatch);
    EXPECT_TRUE(endpoint0->wait_activated(second_run));
    EXPECT_EQ(orchestrator.active_run_id(), second_run);

    endpoint0->emit(WorkerProgressKind::ACCEPTED, *second_dispatch);
    endpoint0->emit(WorkerProgressKind::COMPLETED, *second_dispatch);
    EXPECT_TRUE(orchestrator.wait_run_for(first_run, 3.0));
    EXPECT_TRUE(orchestrator.wait_run_for(second_run, 3.0));
    if (orchestrator.run_done(first_run)) orchestrator.release_run(first_run);
    if (orchestrator.run_done(second_run)) orchestrator.release_run(second_run);
}

TEST_F(ProgressSchedulerFixture, DiagnosticSuccessorWaitsForActiveLaneInsteadOfStaging) {
    RunId first_run = orchestrator.begin_run();
    SubmitResult first =
        orchestrator.submit_next_level(C(1), single_tensor_args(0x1100, TensorArgType::OUTPUT), config, 0);
    orchestrator.close_run_submission(first_run);
    ASSERT_TRUE(endpoint0->wait_submitted(1));

    CallConfig diagnostic_config;
    diagnostic_config.enable_dump_args = 1;
    std::snprintf(
        diagnostic_config.output_prefix, sizeof(diagnostic_config.output_prefix), "%s",
        "/tmp/simpler-diagnostic-successor"
    );
    RunId second_run = orchestrator.begin_run();
    SubmitResult second =
        orchestrator.submit_next_level(C(2), single_tensor_args(0x2200, TensorArgType::OUTPUT), diagnostic_config, 0);
    orchestrator.close_run_submission(second_run);

    EXPECT_FALSE(endpoint0->wait_submitted(2, std::chrono::milliseconds(50)));
    std::vector<WorkerDispatch> submitted = endpoint0->submitted();
    ASSERT_EQ(submitted.size(), 1u);
    EXPECT_EQ(submitted[0].task_slot, first.task_slot);
    EXPECT_FALSE(submitted[0].prepare_only);

    endpoint0->emit(WorkerProgressKind::ACCEPTED, submitted[0]);
    endpoint0->emit(WorkerProgressKind::COMPLETED, submitted[0]);
    ASSERT_TRUE(endpoint0->wait_submitted(2));
    submitted = endpoint0->submitted();
    ASSERT_EQ(submitted.size(), 2u);
    EXPECT_EQ(submitted[1].task_slot, second.task_slot);
    EXPECT_FALSE(submitted[1].prepare_only);

    endpoint0->emit(WorkerProgressKind::ACCEPTED, submitted[1]);
    endpoint0->emit(WorkerProgressKind::COMPLETED, submitted[1]);
    EXPECT_TRUE(orchestrator.wait_run_for(first_run, 3.0));
    EXPECT_TRUE(orchestrator.wait_run_for(second_run, 3.0));
    if (orchestrator.run_done(first_run)) orchestrator.release_run(first_run);
    if (orchestrator.run_done(second_run)) orchestrator.release_run(second_run);
}

TEST_F(ProgressSchedulerFixture, ActivatedRunReleasesTheStagingLaneForItsSuccessor) {
    RunId first_run = orchestrator.begin_run();
    SubmitResult first =
        orchestrator.submit_next_level(C(7), single_tensor_args(0x8000, TensorArgType::OUTPUT), config, 0);
    orchestrator.close_run_submission(first_run);
    RunId second_run = orchestrator.begin_run();
    SubmitResult second =
        orchestrator.submit_next_level(C(8), single_tensor_args(0x9000, TensorArgType::OUTPUT), config, 0);
    orchestrator.close_run_submission(second_run);

    ASSERT_TRUE(endpoint0->wait_submitted(2));
    std::vector<WorkerDispatch> submitted = endpoint0->submitted();
    auto first_dispatch = std::find_if(submitted.begin(), submitted.end(), [&](const WorkerDispatch &dispatch) {
        return dispatch.task_slot == first.task_slot;
    });
    auto second_dispatch = std::find_if(submitted.begin(), submitted.end(), [&](const WorkerDispatch &dispatch) {
        return dispatch.task_slot == second.task_slot;
    });
    ASSERT_NE(first_dispatch, submitted.end());
    ASSERT_NE(second_dispatch, submitted.end());
    const WorkerDispatch first_dispatch_copy = *first_dispatch;
    const WorkerDispatch second_dispatch_copy = *second_dispatch;
    endpoint0->emit(WorkerProgressKind::ACCEPTED, first_dispatch_copy);
    endpoint0->emit(WorkerProgressKind::COMPLETED, first_dispatch_copy);
    ASSERT_TRUE(endpoint0->wait_activated(second_run));
    ASSERT_TRUE(orchestrator.wait_run_for(first_run, 3.0));

    RunId third_run = orchestrator.begin_run();
    SubmitResult third =
        orchestrator.submit_next_level(C(9), single_tensor_args(0xA000, TensorArgType::OUTPUT), config, 0);
    orchestrator.close_run_submission(third_run);
    ASSERT_TRUE(endpoint0->wait_submitted(3));
    submitted = endpoint0->submitted();
    auto third_dispatch = std::find_if(submitted.begin(), submitted.end(), [&](const WorkerDispatch &dispatch) {
        return dispatch.task_slot == third.task_slot;
    });
    ASSERT_NE(third_dispatch, submitted.end());
    EXPECT_TRUE(third_dispatch->prepare_only);
    EXPECT_FALSE(endpoint0->wait_activated(third_run, std::chrono::milliseconds(20)));

    endpoint0->emit(WorkerProgressKind::ACCEPTED, second_dispatch_copy);
    endpoint0->emit(WorkerProgressKind::COMPLETED, second_dispatch_copy);
    ASSERT_TRUE(endpoint0->wait_activated(third_run));
    endpoint0->emit(WorkerProgressKind::ACCEPTED, *third_dispatch);
    endpoint0->emit(WorkerProgressKind::COMPLETED, *third_dispatch);

    EXPECT_TRUE(orchestrator.wait_run_for(second_run, 3.0));
    EXPECT_TRUE(orchestrator.wait_run_for(third_run, 3.0));
    if (orchestrator.run_done(first_run)) orchestrator.release_run(first_run);
    if (orchestrator.run_done(second_run)) orchestrator.release_run(second_run);
    if (orchestrator.run_done(third_run)) orchestrator.release_run(third_run);
}

TEST_F(CapacityOneProgressSchedulerFixture, PublicationFailureCompletesTheClaimedSuccessor) {
    RunId first_run = orchestrator.begin_run();
    orchestrator.submit_next_level(C(5), single_tensor_args(0x6000, TensorArgType::OUTPUT), config, 0);
    orchestrator.close_run_submission(first_run);
    EXPECT_TRUE(endpoint0->wait_submitted(1));

    RunId second_run = orchestrator.begin_run();
    orchestrator.submit_next_level(C(6), single_tensor_args(0x7000, TensorArgType::OUTPUT), config, 0);
    orchestrator.close_run_submission(second_run);

    // Wait for the scheduler to claim and reject the successor while the first
    // dispatch still owns the endpoint's only inflight slot. This successor
    // remains PREPARED after its task error and is not terminal until FIFO
    // promotion, so observe its recorded failure rather than wait_run_for().
    // Completing the first dispatch before this fence races promotion and can
    // turn the successor into an ordinary active dispatch.
    const auto publication_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (!orchestrator.run_failed(second_run) && std::chrono::steady_clock::now() < publication_deadline) {
        std::this_thread::yield();
    }
    ASSERT_TRUE(orchestrator.run_failed(second_run));
    std::vector<WorkerDispatch> submitted = endpoint0->submitted();
    ASSERT_EQ(submitted.size(), 1u) << "the rejected successor must not reach the endpoint";
    endpoint0->emit(WorkerProgressKind::ACCEPTED, submitted[0]);
    endpoint0->emit(WorkerProgressKind::COMPLETED, submitted[0]);

    EXPECT_TRUE(orchestrator.wait_run_for(first_run, 3.0));
    EXPECT_THROW((void)orchestrator.wait_run_for(second_run, 3.0), std::runtime_error);
    EXPECT_TRUE(scheduler.running());
    if (orchestrator.run_done(first_run)) orchestrator.release_run(first_run);
    if (orchestrator.run_done(second_run)) orchestrator.release_run(second_run);
}

TEST_F(ProgressSchedulerFixture, PreparedSuccessorGroupRemainsQueuedUntilPromotion) {
    RunId first_run = orchestrator.begin_run();
    orchestrator.submit_next_level(C(3), single_tensor_args(0x3000, TensorArgType::OUTPUT), config, 0);
    orchestrator.close_run_submission(first_run);
    RunId second_run = orchestrator.begin_run();
    orchestrator.submit_next_level_group(
        C(4), {single_tensor_args(0x4000, TensorArgType::OUTPUT), single_tensor_args(0x5000, TensorArgType::OUTPUT)},
        config, {0, 1}
    );
    orchestrator.close_run_submission(second_run);

    EXPECT_TRUE(endpoint0->wait_submitted(1));
    EXPECT_TRUE(ready_next.singles_empty(second_run));
    EXPECT_FALSE(ready_next.groups_empty(second_run));
    EXPECT_FALSE(manager.has_staged_run(second_run));
    EXPECT_TRUE(endpoint1->submitted().empty());
    std::vector<WorkerDispatch> first_submissions = endpoint0->submitted();
    ASSERT_EQ(first_submissions.size(), 1u);

    endpoint0->emit(WorkerProgressKind::ACCEPTED, first_submissions[0]);
    endpoint0->emit(WorkerProgressKind::COMPLETED, first_submissions[0]);
    EXPECT_TRUE(endpoint0->wait_submitted(2));
    EXPECT_TRUE(endpoint1->wait_submitted(1));
    std::vector<WorkerDispatch> worker0_submissions = endpoint0->submitted();
    std::vector<WorkerDispatch> worker1_submissions = endpoint1->submitted();
    ASSERT_EQ(worker0_submissions.size(), 2u);
    ASSERT_EQ(worker1_submissions.size(), 1u);
    EXPECT_FALSE(worker0_submissions[1].prepare_only);
    EXPECT_FALSE(worker1_submissions[0].prepare_only);
    EXPECT_EQ(orchestrator.active_run_id(), second_run);
    EXPECT_EQ(orchestrator.preparable_run_id(), INVALID_RUN_ID);

    endpoint0->emit(WorkerProgressKind::ACCEPTED, worker0_submissions[1]);
    endpoint1->emit(WorkerProgressKind::ACCEPTED, worker1_submissions[0]);
    endpoint0->emit(WorkerProgressKind::COMPLETED, worker0_submissions[1]);
    endpoint1->emit(WorkerProgressKind::COMPLETED, worker1_submissions[0]);
    EXPECT_TRUE(orchestrator.wait_run_for(first_run, 3.0));
    EXPECT_TRUE(orchestrator.wait_run_for(second_run, 3.0));
    if (orchestrator.run_done(first_run)) orchestrator.release_run(first_run);
    if (orchestrator.run_done(second_run)) orchestrator.release_run(second_run);
}

TEST_F(ProgressSchedulerFixture, PreparedSuccessorSingleCannotBypassItsReadyGroup) {
    RunId first_run = orchestrator.begin_run();
    SubmitResult first =
        orchestrator.submit_next_level(C(10), single_tensor_args(0xB000, TensorArgType::OUTPUT), config, 0);
    orchestrator.close_run_submission(first_run);

    RunId second_run = orchestrator.begin_run();
    SubmitResult group = orchestrator.submit_next_level_group(
        C(11), {single_tensor_args(0xC000, TensorArgType::OUTPUT), single_tensor_args(0xD000, TensorArgType::OUTPUT)},
        config, {0, 1}
    );
    SubmitResult single =
        orchestrator.submit_next_level(C(12), single_tensor_args(0xE000, TensorArgType::OUTPUT), config, 0);
    orchestrator.close_run_submission(second_run);

    ASSERT_TRUE(endpoint0->wait_submitted(1));
    EXPECT_EQ(endpoint0->submitted()[0].task_slot, first.task_slot);
    EXPECT_TRUE(endpoint1->submitted().empty());
    EXPECT_FALSE(manager.has_staged_run(second_run));

    endpoint0->emit(WorkerProgressKind::ACCEPTED, endpoint0->submitted()[0]);
    endpoint0->emit(WorkerProgressKind::COMPLETED, endpoint0->submitted()[0]);

    ASSERT_TRUE(endpoint0->wait_submitted(2));
    ASSERT_TRUE(endpoint1->wait_submitted(1));
    std::vector<WorkerDispatch> worker0_submissions = endpoint0->submitted();
    std::vector<WorkerDispatch> worker1_submissions = endpoint1->submitted();
    ASSERT_EQ(worker0_submissions.size(), 2u);
    ASSERT_EQ(worker1_submissions.size(), 1u);
    EXPECT_EQ(worker0_submissions[1].task_slot, group.task_slot);
    EXPECT_EQ(worker1_submissions[0].task_slot, group.task_slot);
    EXPECT_FALSE(worker0_submissions[1].prepare_only);
    EXPECT_FALSE(worker1_submissions[0].prepare_only);

    endpoint0->emit(WorkerProgressKind::ACCEPTED, worker0_submissions[1]);
    endpoint1->emit(WorkerProgressKind::ACCEPTED, worker1_submissions[0]);
    endpoint0->emit(WorkerProgressKind::COMPLETED, worker0_submissions[1]);
    endpoint1->emit(WorkerProgressKind::COMPLETED, worker1_submissions[0]);

    ASSERT_TRUE(endpoint0->wait_submitted(3));
    worker0_submissions = endpoint0->submitted();
    ASSERT_EQ(worker0_submissions.size(), 3u);
    EXPECT_EQ(worker0_submissions[2].task_slot, single.task_slot);
    EXPECT_FALSE(worker0_submissions[2].prepare_only);
    endpoint0->emit(WorkerProgressKind::ACCEPTED, worker0_submissions[2]);
    endpoint0->emit(WorkerProgressKind::COMPLETED, worker0_submissions[2]);

    EXPECT_TRUE(orchestrator.wait_run_for(first_run, 3.0));
    EXPECT_TRUE(orchestrator.wait_run_for(second_run, 3.0));
    if (orchestrator.run_done(first_run)) orchestrator.release_run(first_run);
    if (orchestrator.run_done(second_run)) orchestrator.release_run(second_run);
}

TEST(WorkerManagerTest, LocalMailboxPublishesAcceptanceBeforeCompletion) {
    MockMailboxWorker child;
    child.start();

    Ring allocator;
    allocator.init(/*heap_bytes=*/0);
    AllocResult ar = allocator.alloc(/*heap_bytes=*/0, /*depth=*/0);
    ASSERT_NE(ar.slot, INVALID_SLOT);
    TaskSlotState *slot = allocator.slot_state(ar.slot);
    ASSERT_NE(slot, nullptr);
    slot->reset();
    slot->callable.digest[0] = 0x42;
    slot->pipeline_lease = PipelineSlotLease{1, 0, 7};

    LocalMailboxEndpoint endpoint(/*worker_id=*/0, child.mailbox_ptr());
    std::promise<WorkerCompletion> result;
    auto done = result.get_future();
    std::atomic<bool> accepted{false};
    std::thread caller([&] {
        result.set_value(endpoint.run_with_accept(&allocator, WorkerDispatch{ar.slot, 0}, [&] {
            accepted.store(true, std::memory_order_release);
        }));
    });

    child.wait_running();
    EXPECT_TRUE(child.is_running.load(std::memory_order_acquire));
    PipelineSlotLease wire_lease{};
    std::memcpy(
        &wire_lease, static_cast<char *>(child.mailbox_ptr()) + MAILBOX_OFF_PIPELINE_LEASE, sizeof(PipelineSlotLease)
    );
    EXPECT_EQ(wire_lease.slot_id, 1u);
    EXPECT_EQ(wire_lease.generation, 7u);
    child.write_task_accepted();
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (!accepted.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_TRUE(accepted.load(std::memory_order_acquire));
    EXPECT_EQ(done.wait_for(std::chrono::milliseconds(0)), std::future_status::timeout);

    // Non-fatal from here on: a fatal assertion would return with `caller`
    // joinable, and ~std::thread would terminate the whole test binary.
    child.complete();
    EXPECT_EQ(done.wait_for(std::chrono::seconds(3)), std::future_status::ready);
    if (done.valid() && done.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        EXPECT_EQ(done.get().outcome, EndpointOutcome::SUCCESS);
    }
    caller.join();
    allocator.shutdown();
}

// The ACK must not be carried by anything TASK_DONE can overwrite. The child
// publishes acceptance only into the sticky word — never into the state — and
// then completes immediately, so an endpoint that looks for acceptance in the
// state word observes none at all.
//
// This does not force the parent to skip a poll between the two writes: the
// parent is already spinning by then and nothing here can stop it. What it
// pins is the property that makes that interleaving harmless — acceptance is
// readable after TASK_DONE, so losing a poll cannot lose the ACK.
TEST(WorkerManagerTest, AcceptanceIsReadableAfterTaskDone) {
    MockMailboxWorker child;
    child.start();

    Ring allocator;
    allocator.init(/*heap_bytes=*/0);
    AllocResult ar = allocator.alloc(/*heap_bytes=*/0, /*depth=*/0);
    ASSERT_NE(ar.slot, INVALID_SLOT);
    TaskSlotState *slot = allocator.slot_state(ar.slot);
    ASSERT_NE(slot, nullptr);
    slot->reset();
    slot->callable.digest[0] = 0x42;

    LocalMailboxEndpoint endpoint(/*worker_id=*/0, child.mailbox_ptr());
    std::promise<WorkerCompletion> result;
    auto done = result.get_future();
    std::atomic<bool> accepted{false};

    std::thread caller([&] {
        result.set_value(endpoint.run_with_accept(&allocator, WorkerDispatch{ar.slot, 0}, [&] {
            accepted.store(true, std::memory_order_release);
        }));
    });

    child.wait_running();
    EXPECT_TRUE(child.is_running.load(std::memory_order_acquire));
    // Back to back, with no parent poll in between.
    child.write_task_accepted();
    child.complete();

    // Non-fatal: a fatal assertion would return with `caller` joinable, and
    // ~std::thread would terminate the whole test binary.
    EXPECT_EQ(done.wait_for(std::chrono::seconds(3)), std::future_status::ready);
    if (done.valid() && done.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        EXPECT_EQ(done.get().outcome, EndpointOutcome::SUCCESS);
    }
    EXPECT_TRUE(accepted.load(std::memory_order_acquire))
        << "the endpoint lost the launch ACK to a task that completed first";
    caller.join();
    allocator.shutdown();
}

// A child that dies without publishing CONTROL_DONE must be reported, not
// waited on forever. The mailbox stays at CONTROL_REQUEST exactly as it would
// if the real `_chip_process_loop` had crashed mid-command. Run in a worker
// thread with a bounded join so a regression fails the test instead of
// hanging the suite.
TEST(WorkerManagerTest, ControlCommandFailsWhenChildExitsBeforeCompletion) {
    void *mailbox =
        mmap(nullptr, MAILBOX_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, /*fd=*/-1, /*offset=*/0);
    ASSERT_NE(mailbox, MAP_FAILED);
    std::memset(mailbox, 0, MAILBOX_SIZE);

    pid_t child = fork();
    ASSERT_GE(child, 0);
    if (child == 0) {
        _exit(3);
    }

    LocalMailboxEndpoint endpoint(/*worker_id=*/0, mailbox, static_cast<int>(child));

    std::promise<std::string> result;
    auto done = result.get_future();
    std::thread caller([&] {
        try {
            endpoint.control_malloc(64);
            result.set_value("");
        } catch (const std::runtime_error &e) {
            result.set_value(e.what());
        }
    });

    ASSERT_EQ(done.wait_for(std::chrono::seconds(10)), std::future_status::ready)
        << "control_malloc did not observe the dead child; it is spinning on CONTROL_DONE";
    std::string message = done.get();
    caller.join();

    EXPECT_NE(message.find("child process pid=" + std::to_string(child)), std::string::npos) << message;
    EXPECT_NE(message.find("exit_status=3"), std::string::npos) << message;

    // The endpoint is poisoned once the child is gone: a later command reports
    // rather than resuming the spin.
    EXPECT_THROW(endpoint.control_free(0), std::runtime_error);

    ASSERT_EQ(munmap(mailbox, MAILBOX_SIZE), 0);
}

TEST(WorkerManagerTest, ControlPrepareUsesStableNextLevelWorkerId) {
    Ring allocator;
    allocator.init(/*heap_bytes=*/0);
    WorkerManager manager;
    std::atomic<int> worker7_prepares{0};
    std::atomic<int> worker3_prepares{0};

    manager.add_next_level_endpoint(std::make_unique<FakeEndpoint>(7, &worker7_prepares));
    manager.add_next_level_endpoint(std::make_unique<FakeEndpoint>(3, &worker3_prepares));
    manager.start(&allocator, [](WorkerCompletion) {}, {});

    std::array<uint8_t, CALLABLE_HASH_DIGEST_SIZE> digest{};
    manager.control_prepare(3, digest.data());

    manager.stop();
    allocator.shutdown();
    EXPECT_EQ(worker7_prepares.load(std::memory_order_relaxed), 0);
    EXPECT_EQ(worker3_prepares.load(std::memory_order_relaxed), 1);
}

// The losing side of the dispatch claim, driven by the real cancellation path
// rather than a simulated state write. `before_claim_cb` is the only point that
// can observe the window: everything else is either before the queue pop or
// after the launch.
TEST_F(SchedulerFixture, ACancellationThatWinsTheClaimStopsTheDispatch) {
    std::atomic<int> hook_calls{0};
    before_claim_hook = [this, &hook_calls](TaskSlot) {
        if (hook_calls.fetch_add(1) != 0) return;
        orch.fail_run_submission(run_id, std::make_exception_ptr(std::runtime_error("cancelled mid-dispatch")));
    };

    auto task = orch.submit_next_level(C(60), single_tensor_args(0x6000, TensorArgType::OUTPUT), cfg, 0);

    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (hook_calls.load() == 0 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_GT(hook_calls.load(), 0) << "the Scheduler never reached a dispatch claim";

    // Give the Scheduler its whole loop iteration; a lost claim must leave the
    // slot alone rather than continue into the launch.
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    EXPECT_NE(S(task.task_slot).state.load(std::memory_order_acquire), TaskState::RUNNING)
        << "the dispatch overwrote a slot the cancellation already owned";
    {
        std::lock_guard<std::mutex> lk(mock_worker.dispatched_mu);
        EXPECT_TRUE(mock_worker.dispatched.empty()) << "a cancelled task was still handed to a worker";
    }
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
                                static_cast<size_t>(CHIP_MAX_TENSOR_ARGS) * sizeof(ChipTensor) +
                                static_cast<size_t>(CHIP_MAX_SCALAR_ARGS) * sizeof(uint64_t);
    EXPECT_GE(MAILBOX_ARGS_CAPACITY, max_blob);

    TaskArgs args;
    for (int i = 0; i < 76; ++i) {
        ChipTensor t{};
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
    EXPECT_TRUE(orch.run_failed(run_id));
    EXPECT_EQ(mock_worker.dispatched_count(), 1) << "poisoned consumer must not dispatch";
    EXPECT_EQ(S(a.task_slot).state.load(), TaskState::CONSUMED);
    EXPECT_EQ(S(b.task_slot).state.load(), TaskState::CONSUMED);
}

// ===========================================================================
// Group task tests -- fixture with 3 MockMailboxWorkers
// ===========================================================================

struct GroupSchedulerFixture : public ::testing::Test {
    TensorMap tm;
    Ring allocator;
    Scope scope;
    ReadyQueue rq_sub;
    NextLevelReadyQueues rq_next_level;
    Orchestrator orch;
    MockMailboxWorker worker_a;
    MockMailboxWorker worker_b;
    MockMailboxWorker worker_c;
    MockMailboxWorker sub_worker_a;
    MockMailboxWorker sub_worker_b;
    WorkerManager manager;
    Scheduler sched;
    CallConfig cfg;
    RunId run_id{INVALID_RUN_ID};
    std::chrono::milliseconds reservation_stall_warn_after{std::chrono::seconds(5)};
    Scheduler::ReservationStallSink reservation_stall_sink{nullptr};
    void *reservation_stall_sink_context{nullptr};

    std::vector<TaskSlot> consumed_slots;
    std::mutex consumed_mu;

    TaskSlotState &S(TaskSlot id) { return *allocator.slot_state(id); }

    void SetUp() override {
        allocator.init(/*heap_bytes=*/1ULL << 20);

        worker_a.start();
        worker_b.start();
        worker_c.start();
        sub_worker_a.start();
        sub_worker_b.start();
        manager.add_next_level(worker_a.mailbox_ptr());
        manager.add_next_level(worker_b.mailbox_ptr());
        manager.add_next_level(worker_c.mailbox_ptr());
        manager.add_sub(sub_worker_a.mailbox_ptr());
        manager.add_sub(sub_worker_b.mailbox_ptr());
        manager.start(
            &allocator,
            [this](WorkerCompletion completion) {
                sched.worker_done(std::move(completion));
            },
            [this](WorkerDispatch dispatch) {
                orch.mark_task_accepted(dispatch.task_slot);
            },
            [this] {
                sched.notify_ready();
            }
        );
        rq_next_level.reset(manager.next_level_worker_ids());
        orch.init(&tm, &allocator, &scope, &rq_sub, &rq_next_level, &manager, [this] {
            sched.notify_ready();
        });
        run_id = orch.begin_run();

        Scheduler::Config c;
        c.ring = &allocator;
        c.ready_sub_queue = &rq_sub;
        c.ready_next_level_queues = &rq_next_level;
        c.manager = &manager;
        c.enqueue_ready_cb = [this](TaskSlot slot) {
            orch.enqueue_ready(slot);
        };
        // Same gate Worker::start installs. Without it the scheduler takes the
        // unpartitioned branch, which is not the one #1565's group reservation
        // and placement run through.
        c.active_run_cb = [this] {
            return orch.dispatchable_run_id();
        };
        c.on_consumed_cb = [this](TaskSlot s) {
            orch.on_consumed(s);
            std::lock_guard<std::mutex> lk(consumed_mu);
            consumed_slots.push_back(s);
        };
        c.on_task_failed_cb = [this](TaskSlot s, const std::string &message) {
            orch.report_task_error(s, message);
        };
        c.reservation_stall_warn_after = reservation_stall_warn_after;
        c.reservation_stall_sink = reservation_stall_sink;
        c.reservation_stall_sink_context = reservation_stall_sink_context;
        sched.start(c);
    }

    void TearDown() override {
        worker_a.drain();
        worker_b.drain();
        worker_c.drain();
        sub_worker_a.drain();
        sub_worker_b.drain();
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

struct ReservationStallCapture {
    std::atomic<int> report_count{0};
    TaskSlot group_slot{INVALID_SLOT};
    std::array<int32_t, 3> busy_target_worker_ids{};
    size_t busy_target_count{0};
    std::array<int32_t, 3> idle_queued_target_worker_ids{};
    std::array<TaskSlot, 3> idle_queued_single_head_slots{};
    size_t idle_queued_target_count{0};
};

void capture_reservation_stall(void *context, const Scheduler::ReservationStallDiagnostic &diagnostic) noexcept {
    auto *capture = static_cast<ReservationStallCapture *>(context);
    capture->group_slot = diagnostic.group_slot;
    capture->busy_target_count = std::min(diagnostic.busy_target_count, capture->busy_target_worker_ids.size());
    if (capture->busy_target_count > 0) {
        std::copy_n(
            diagnostic.busy_target_worker_ids, capture->busy_target_count, capture->busy_target_worker_ids.begin()
        );
    }
    capture->idle_queued_target_count =
        std::min(diagnostic.idle_queued_target_count, capture->idle_queued_target_worker_ids.size());
    if (capture->idle_queued_target_count > 0) {
        std::copy_n(
            diagnostic.idle_queued_target_worker_ids, capture->idle_queued_target_count,
            capture->idle_queued_target_worker_ids.begin()
        );
        std::copy_n(
            diagnostic.idle_queued_single_head_slots, capture->idle_queued_target_count,
            capture->idle_queued_single_head_slots.begin()
        );
    }
    capture->report_count.fetch_add(1, std::memory_order_release);
}

struct ReservationStallSchedulerFixture : public GroupSchedulerFixture {
    ReservationStallCapture stall_capture;

    ReservationStallSchedulerFixture() {
        reservation_stall_warn_after = std::chrono::milliseconds(20);
        reservation_stall_sink = capture_reservation_stall;
        reservation_stall_sink_context = &stall_capture;
    }
};

TEST_F(ReservationStallSchedulerFixture, ReportsStructuralStallOncePerEpisode) {
    auto running_a = orch.submit_next_level(C(88), single_tensor_args(0x110, TensorArgType::OUTPUT), cfg, 0);
    auto running_b = orch.submit_next_level(C(89), single_tensor_args(0x111, TensorArgType::OUTPUT), cfg, 1);
    worker_a.wait_running();
    worker_b.wait_running();
    EXPECT_TRUE(worker_a.is_running.load(std::memory_order_acquire));
    EXPECT_TRUE(worker_b.is_running.load(std::memory_order_acquire));

    auto group = orch.submit_next_level_group(
        C(90), {single_tensor_args(0x112, TensorArgType::OUTPUT), single_tensor_args(0x113, TensorArgType::OUTPUT)},
        cfg, {0, 1}
    );
    auto single_a = orch.submit_next_level(C(91), single_tensor_args(0x114, TensorArgType::OUTPUT), cfg, 0);
    auto single_b = orch.submit_next_level(C(92), single_tensor_args(0x115, TensorArgType::OUTPUT), cfg, 1);

    std::this_thread::sleep_for(std::chrono::milliseconds(40));
    EXPECT_EQ(stall_capture.report_count.load(std::memory_order_acquire), 0);

    worker_a.complete();
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(200);
    while (stall_capture.report_count.load(std::memory_order_acquire) == 0 &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_EQ(stall_capture.report_count.load(std::memory_order_acquire), 1);
    EXPECT_EQ(stall_capture.group_slot, group.task_slot);
    EXPECT_EQ(stall_capture.busy_target_count, 1u);
    EXPECT_EQ(stall_capture.busy_target_worker_ids[0], 1);
    EXPECT_EQ(stall_capture.idle_queued_target_count, 1u);
    EXPECT_EQ(stall_capture.idle_queued_target_worker_ids[0], 0);
    EXPECT_EQ(stall_capture.idle_queued_single_head_slots[0], single_a.task_slot);
    EXPECT_EQ(worker_a.dispatched_count(), 1);

    for (int i = 0; i < 3; ++i)
        sched.notify_ready();
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    EXPECT_EQ(stall_capture.report_count.load(std::memory_order_acquire), 1);

    worker_b.complete();
    deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
    while ((worker_a.dispatched_count() < 2 || worker_b.dispatched_count() < 2) &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_EQ(worker_a.dispatched_count(), 2);
    EXPECT_EQ(worker_b.dispatched_count(), 2);
    worker_a.complete();
    worker_b.complete();

    deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
    while ((worker_a.dispatched_count() < 3 || worker_b.dispatched_count() < 3) &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_EQ(worker_a.dispatched_count(), 3);
    EXPECT_EQ(worker_b.dispatched_count(), 3);
    worker_a.complete();
    worker_b.complete();

    wait_consumed(running_a.task_slot);
    wait_consumed(running_b.task_slot);
    wait_consumed(group.task_slot);
    wait_consumed(single_a.task_slot);
    wait_consumed(single_b.task_slot);
}

TEST_F(GroupSchedulerFixture, GroupDispatchesToNWorkers) {
    TaskArgs a0 = single_tensor_args(0xA0, TensorArgType::OUTPUT);
    TaskArgs a1 = single_tensor_args(0xA1, TensorArgType::OUTPUT);

    auto res = orch.submit_next_level_group(C(42), {a0, a1}, cfg, {0, 1});
    TaskSlot slot = res.task_slot;

    worker_a.wait_running();
    worker_b.wait_running();

    EXPECT_EQ(worker_a.dispatched_count(), 1);
    EXPECT_EQ(worker_b.dispatched_count(), 1);

    EXPECT_EQ(worker_a.dispatched[0].tensor_key, 0xA0u);
    EXPECT_EQ(worker_b.dispatched[0].tensor_key, 0xA1u);
    (void)slot;

    worker_a.complete();
    worker_b.complete();
    wait_consumed(slot);
}

TEST_F(GroupSchedulerFixture, SubGroupUsesTheAllocationFreeGroupCommitPath) {
    TaskArgs a0 = single_tensor_args(0xA2, TensorArgType::OUTPUT);
    TaskArgs a1 = single_tensor_args(0xA3, TensorArgType::OUTPUT);
    auto res = orch.submit_sub_group(C(44), {a0, a1});

    sub_worker_a.wait_running();
    sub_worker_b.wait_running();
    EXPECT_EQ(sub_worker_a.dispatched_count(), 1);
    EXPECT_EQ(sub_worker_b.dispatched_count(), 1);
    EXPECT_EQ(S(res.task_slot).state.load(std::memory_order_acquire), TaskState::RUNNING);

    sub_worker_a.complete();
    sub_worker_b.complete();
    wait_consumed(res.task_slot);
}

TEST_F(GroupSchedulerFixture, GroupMapsEachMemberToItsTargetWorkerIdNotIndex) {
    // Reversed target order: member 0 -> worker id 1 (worker_b), member 1 ->
    // worker id 0 (worker_a). A map-by-registration-index bug would instead
    // send member 0 (a0) to worker_a; the reversed keys catch it.
    TaskArgs a0 = single_tensor_args(0xA0, TensorArgType::OUTPUT);
    TaskArgs a1 = single_tensor_args(0xA1, TensorArgType::OUTPUT);

    auto res = orch.submit_next_level_group(C(42), {a0, a1}, cfg, {1, 0});
    TaskSlot slot = res.task_slot;

    worker_a.wait_running();
    worker_b.wait_running();

    EXPECT_EQ(worker_a.dispatched_count(), 1);
    EXPECT_EQ(worker_b.dispatched_count(), 1);

    EXPECT_EQ(worker_b.dispatched[0].tensor_key, 0xA0u);
    EXPECT_EQ(worker_a.dispatched[0].tensor_key, 0xA1u);
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

TEST_F(GroupSchedulerFixture, BlockedGroupReservesTargetsThatBecomeIdleOneAtATime) {
    auto running_a = orch.submit_next_level(C(70), single_tensor_args(0xF0, TensorArgType::OUTPUT), cfg, 0);
    auto running_b = orch.submit_next_level(C(71), single_tensor_args(0xF1, TensorArgType::OUTPUT), cfg, 1);
    worker_a.wait_running();
    worker_b.wait_running();
    ASSERT_TRUE(worker_a.is_running.load());
    ASSERT_TRUE(worker_b.is_running.load());

    TaskArgs group_a = single_tensor_args(0xF2, TensorArgType::OUTPUT);
    TaskArgs group_b = single_tensor_args(0xF3, TensorArgType::OUTPUT);
    auto group = orch.submit_next_level_group(C(72), {group_a, group_b}, cfg, {1, 0});
    auto single_a = orch.submit_next_level(C(73), single_tensor_args(0xF4, TensorArgType::OUTPUT), cfg, 0);
    auto single_b = orch.submit_next_level(C(74), single_tensor_args(0xF5, TensorArgType::OUTPUT), cfg, 1);
    auto unrelated = orch.submit_next_level(C(75), single_tensor_args(0xF6, TensorArgType::OUTPUT), cfg, 2);

    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
    while (worker_c.dispatched_count() == 0 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_EQ(worker_c.dispatched_count(), 1);
    EXPECT_EQ(worker_c.dispatched[0].callable_hash0, 75u);
    worker_c.complete();

    worker_a.complete();
    deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(50);
    while (worker_a.dispatched_count() == 1 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_EQ(worker_a.dispatched_count(), 1) << "blocked group must neither dispatch partially nor yield to singles";
    EXPECT_EQ(S(group.task_slot).state.load(), TaskState::READY);

    worker_b.complete();
    deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
    while ((worker_a.dispatched_count() < 2 || worker_b.dispatched_count() < 2) &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_GE(worker_a.dispatched_count(), 2);
    ASSERT_GE(worker_b.dispatched_count(), 2);
    EXPECT_EQ(worker_a.dispatched[1].callable_hash0, 72u);
    EXPECT_EQ(worker_b.dispatched[1].callable_hash0, 72u);

    // Completing one group member makes only worker A idle. Its queued single
    // must progress from the incomplete-member wake while worker B still runs
    // the other member and the group remains aggregate-incomplete.
    worker_a.complete();
    deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
    while (worker_a.dispatched_count() < 3 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_GE(worker_a.dispatched_count(), 3);
    EXPECT_EQ(worker_b.dispatched_count(), 2);
    EXPECT_EQ(S(group.task_slot).state.load(), TaskState::RUNNING);
    EXPECT_EQ(worker_a.dispatched[2].callable_hash0, 73u);
    worker_a.complete();

    worker_b.complete();
    deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
    while (worker_b.dispatched_count() < 3 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_GE(worker_b.dispatched_count(), 3);
    EXPECT_EQ(worker_b.dispatched[2].callable_hash0, 74u);
    worker_b.complete();

    wait_consumed(running_a.task_slot);
    wait_consumed(running_b.task_slot);
    wait_consumed(group.task_slot);
    wait_consumed(single_a.task_slot);
    wait_consumed(single_b.task_slot);
    wait_consumed(unrelated.task_slot);
}

TEST(SchedulerDispatchPassTest, ActiveRunSwitchCannotBypassSuccessorGroupReservation) {
    constexpr RunId run_a = 41;
    constexpr RunId run_b = 42;

    Ring allocator;
    ReadyQueue rq_sub;
    NextLevelReadyQueues rq_next_level;
    MockMailboxWorker worker_a;
    MockMailboxWorker worker_b;
    WorkerManager manager;
    Scheduler sched;

    allocator.init(/*heap_bytes=*/0);
    worker_a.start();
    worker_b.start();
    manager.add_next_level(worker_a.mailbox_ptr());
    manager.add_next_level(worker_b.mailbox_ptr());
    manager.start(
        &allocator,
        [&sched](WorkerCompletion completion) {
            sched.worker_done(std::move(completion));
        },
        [](WorkerDispatch) {},
        [&sched] {
            sched.notify_ready();
        }
    );
    rq_next_level.reset(manager.next_level_worker_ids());

    auto allocate_slot = [&](RunId run_id, uint8_t callable_seed, int32_t worker_id, bool group) {
        AllocResult allocation = allocator.alloc(/*heap_bytes=*/0, /*scope_depth=*/0);
        TaskSlotState &state = *allocator.slot_state(allocation.slot);
        state.reset();
        state.run_id = run_id;
        state.worker_type = WorkerType::NEXT_LEVEL;
        state.callable = C(callable_seed);
        state.target_worker_ids.push_back(worker_id);
        if (group) {
            state.is_group_ = true;
            state.task_args_list.push_back(single_tensor_args(callable_seed, TensorArgType::OUTPUT));
            rq_next_level.push_group(run_id, allocation.slot);
        } else {
            state.task_args = single_tensor_args(callable_seed, TensorArgType::OUTPUT);
            rq_next_level.push_single(worker_id, run_id, allocation.slot);
        }
        state.state.store(TaskState::READY, std::memory_order_release);
        return allocation.slot;
    };

    // Run A's group occupies worker 1. Run B's group and following single
    // both target worker 0, so the group head owns that worker reservation.
    allocate_slot(run_a, /*callable_seed=*/70, /*worker_id=*/1, /*group=*/true);
    allocate_slot(run_b, /*callable_seed=*/71, /*worker_id=*/0, /*group=*/true);
    allocate_slot(run_b, /*callable_seed=*/72, /*worker_id=*/0, /*group=*/false);

    std::atomic<RunId> active_run{run_a};
    Scheduler::Config config;
    config.ring = &allocator;
    config.ready_sub_queue = &rq_sub;
    config.ready_next_level_queues = &rq_next_level;
    config.manager = &manager;
    config.enqueue_ready_cb = [&](TaskSlot slot) {
        TaskSlotState &state = *allocator.slot_state(slot);
        if (state.is_group()) {
            rq_next_level.push_group(state.run_id, slot);
        } else {
            rq_next_level.push_single(state.target_worker_id(0), state.run_id, slot);
        }
    };
    config.active_run_cb = [&] {
        return active_run.load(std::memory_order_acquire);
    };
    // The run switch lands after the group phase selected A and before the
    // singles phase can select a queue partition. Production run admission
    // advances the scheduler wake generation (ready_notify_cb_), so the seam
    // must signal the switch the same way: in the edge-triggered wake model
    // the switch itself is the wake event.
    config.before_claim_cb = [&](TaskSlot slot) {
        if (allocator.slot_state(slot)->run_id == run_a) {
            active_run.store(run_b, std::memory_order_release);
            sched.notify_ready();
        }
    };
    config.on_consumed_cb = [&](TaskSlot slot) {
        allocator.slot_state(slot)->state.store(TaskState::CONSUMED, std::memory_order_release);
        allocator.release(slot);
    };
    config.on_task_failed_cb = [](TaskSlot, const std::string &) {};
    sched.start(config);

    worker_a.wait_running();
    worker_b.wait_running();
    EXPECT_EQ(worker_a.dispatched_count(), 1);
    EXPECT_EQ(worker_b.dispatched_count(), 1);
    if (worker_a.dispatched_count() == 1) {
        std::lock_guard<std::mutex> lock(worker_a.dispatched_mu);
        EXPECT_EQ(worker_a.dispatched[0].callable_hash0, 71u)
            << "run B's group must dispatch before its single on the same target";
    }
    if (worker_b.dispatched_count() == 1) {
        std::lock_guard<std::mutex> lock(worker_b.dispatched_mu);
        EXPECT_EQ(worker_b.dispatched[0].callable_hash0, 70u);
    }

    if (worker_a.is_running.load(std::memory_order_acquire)) worker_a.complete();
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
    while (worker_a.dispatched_count() < 2 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_EQ(worker_a.dispatched_count(), 2);
    worker_a.wait_running();
    if (worker_a.dispatched_count() == 2) {
        std::lock_guard<std::mutex> lock(worker_a.dispatched_mu);
        EXPECT_EQ(worker_a.dispatched[1].callable_hash0, 72u);
    }

    if (worker_a.is_running.load(std::memory_order_acquire)) worker_a.complete();
    if (worker_b.is_running.load(std::memory_order_acquire)) worker_b.complete();
    sched.stop();
    manager.stop();
    allocator.shutdown();
}

TEST_F(GroupSchedulerFixture, ConsecutiveGroupsReserveOnlyBlockedHeadTargets) {
    SubmitResult first_group;
    SubmitResult second_group;
    SubmitResult single_a;
    SubmitResult single_c;
    {
        std::lock_guard<std::mutex> scheduler_pause(sched.loop_mutex());
        first_group = orch.submit_next_level_group(
            C(80), {single_tensor_args(0x100, TensorArgType::OUTPUT), single_tensor_args(0x101, TensorArgType::OUTPUT)},
            cfg, {0, 1}
        );
        second_group = orch.submit_next_level_group(
            C(81), {single_tensor_args(0x102, TensorArgType::OUTPUT), single_tensor_args(0x103, TensorArgType::OUTPUT)},
            cfg, {1, 2}
        );
        single_a = orch.submit_next_level(C(82), single_tensor_args(0x104, TensorArgType::OUTPUT), cfg, 0);
        single_c = orch.submit_next_level(C(83), single_tensor_args(0x105, TensorArgType::OUTPUT), cfg, 2);
    }

    worker_a.wait_running();
    worker_b.wait_running();
    ASSERT_EQ(worker_a.dispatched_count(), 1);
    ASSERT_EQ(worker_b.dispatched_count(), 1);
    EXPECT_EQ(worker_a.dispatched[0].callable_hash0, 80u);
    EXPECT_EQ(worker_b.dispatched[0].callable_hash0, 80u);
    EXPECT_EQ(worker_c.dispatched_count(), 0);

    worker_a.complete();
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
    while (worker_a.dispatched_count() < 2 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_EQ(worker_a.dispatched_count(), 2);
    EXPECT_EQ(worker_a.dispatched[1].callable_hash0, 82u);
    EXPECT_EQ(worker_c.dispatched_count(), 0);
    EXPECT_EQ(S(second_group.task_slot).state.load(), TaskState::READY);

    worker_a.complete();
    worker_b.complete();
    deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
    while ((worker_b.dispatched_count() < 2 || worker_c.dispatched_count() < 1) &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_EQ(worker_b.dispatched_count(), 2);
    ASSERT_EQ(worker_c.dispatched_count(), 1);
    EXPECT_EQ(worker_b.dispatched[1].callable_hash0, 81u);
    EXPECT_EQ(worker_c.dispatched[0].callable_hash0, 81u);

    worker_b.complete();
    worker_c.complete();
    deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
    while (worker_c.dispatched_count() < 2 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_EQ(worker_c.dispatched_count(), 2);
    EXPECT_EQ(worker_c.dispatched[1].callable_hash0, 83u);
    worker_c.complete();

    wait_consumed(first_group.task_slot);
    wait_consumed(second_group.task_slot);
    wait_consumed(single_a.task_slot);
    wait_consumed(single_c.task_slot);
}

TEST_F(GroupSchedulerFixture, TearDownDrainsCurrentAndQueuedDispatches) {
    // The verification is the teardown itself: work is left deliberately in
    // both states — running and queued-but-undispatched — across NEXT_LEVEL
    // and SUB workers, so every registered worker type must drain cleanly.
    {
        std::lock_guard<std::mutex> scheduler_pause(sched.loop_mutex());
        (void)orch.submit_next_level_group(
            C(84), {single_tensor_args(0x106, TensorArgType::OUTPUT), single_tensor_args(0x107, TensorArgType::OUTPUT)},
            cfg, {0, 1}
        );
        (void)orch.submit_next_level_group(
            C(85), {single_tensor_args(0x108, TensorArgType::OUTPUT), single_tensor_args(0x109, TensorArgType::OUTPUT)},
            cfg, {1, 2}
        );
        (void)orch.submit_next_level(C(86), single_tensor_args(0x10A, TensorArgType::OUTPUT), cfg, 0);
        (void)orch.submit_next_level(C(87), single_tensor_args(0x10B, TensorArgType::OUTPUT), cfg, 2);
        (void)orch.submit_sub_group(
            C(88), {single_tensor_args(0x10C, TensorArgType::OUTPUT), single_tensor_args(0x10D, TensorArgType::OUTPUT)}
        );
    }

    worker_a.wait_running();
    worker_b.wait_running();
    sub_worker_a.wait_running();
    sub_worker_b.wait_running();
    EXPECT_TRUE(worker_a.is_running.load(std::memory_order_acquire));
    EXPECT_TRUE(worker_b.is_running.load(std::memory_order_acquire));
    EXPECT_TRUE(sub_worker_a.is_running.load(std::memory_order_acquire));
    EXPECT_TRUE(sub_worker_b.is_running.load(std::memory_order_acquire));
}

// The shape that hangs when the idle edge is missing: a second task queued for
// the only worker that can run it. Its wake is the first task's completion,
// which a dispatch pass can consume while that worker still reads as occupied.
TEST_F(GroupSchedulerFixture, QueuedSingleRunsAfterItsWorkerFreesUp) {
    auto first = orch.submit_next_level(C(81), single_tensor_args(0x1A, TensorArgType::OUTPUT), cfg, 0);
    worker_a.wait_running();
    EXPECT_EQ(worker_a.dispatched_count(), 1);

    auto queued = orch.submit_next_level(C(82), single_tensor_args(0x1B, TensorArgType::OUTPUT), cfg, 0);
    EXPECT_EQ(worker_a.dispatched_count(), 1);

    worker_a.complete();
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
    while (worker_a.dispatched_count() < 2 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_EQ(worker_a.dispatched_count(), 2) << "the queued single never reached the worker that freed up";
    EXPECT_EQ(worker_a.dispatched[1].callable_hash0, 82u);
    worker_a.complete();

    wait_consumed(first.task_slot);
    wait_consumed(queued.task_slot);
}

TEST_F(GroupSchedulerFixture, BlockedGroupSleepsUntilWorkerCompletion) {
    auto running = orch.submit_next_level(C(78), single_tensor_args(0xFA, TensorArgType::OUTPUT), cfg, 0);
    worker_a.wait_running();
    EXPECT_TRUE(worker_a.is_running.load(std::memory_order_acquire));

    const uint64_t rounds_before_blocked_group = sched.dispatch_round_count();
    auto blocked = orch.submit_next_level_group(
        C(79), {single_tensor_args(0xFB, TensorArgType::OUTPUT), single_tensor_args(0xFC, TensorArgType::OUTPUT)}, cfg,
        {0, 1}
    );

    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
    while (sched.dispatch_round_count() == rounds_before_blocked_group && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    const uint64_t settled_rounds = sched.dispatch_round_count();
    ASSERT_GT(settled_rounds, rounds_before_blocked_group);
    // The fix's property: with the group blocked, dispatch rounds stop
    // advancing once the scheduler parks. Poll for a quiet window instead of
    // a fixed sleep so a loaded runner cannot fail the check spuriously.
    uint64_t quiet_rounds = settled_rounds;
    bool parked = false;
    deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
    while (std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        const uint64_t rounds_now = sched.dispatch_round_count();
        if (rounds_now == quiet_rounds) {
            parked = true;
            break;
        }
        quiet_rounds = rounds_now;
    }
    EXPECT_TRUE(parked) << "scheduler did not park while the group head was blocked";
    EXPECT_EQ(S(blocked.task_slot).state.load(std::memory_order_acquire), TaskState::READY);

    worker_a.complete();
    deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
    while ((worker_a.dispatched_count() < 2 || worker_b.dispatched_count() < 1) &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_EQ(worker_a.dispatched_count(), 2);
    EXPECT_EQ(worker_b.dispatched_count(), 1);
    worker_a.complete();
    worker_b.complete();

    wait_consumed(running.task_slot);
    wait_consumed(blocked.task_slot);
}

TEST_F(GroupSchedulerFixture, LaunchableGroupPrecedesConflictingSingles) {
    auto running_a = orch.submit_next_level(C(73), single_tensor_args(0xF4, TensorArgType::OUTPUT), cfg, 0);
    auto running_b = orch.submit_next_level(C(74), single_tensor_args(0xF5, TensorArgType::OUTPUT), cfg, 1);
    worker_a.wait_running();
    worker_b.wait_running();
    ASSERT_TRUE(worker_a.is_running.load());
    ASSERT_TRUE(worker_b.is_running.load());

    TaskArgs group_a = single_tensor_args(0xF6, TensorArgType::OUTPUT);
    TaskArgs group_b = single_tensor_args(0xF7, TensorArgType::OUTPUT);
    SubmitResult group;
    SubmitResult single_a;
    SubmitResult single_b;
    {
        std::lock_guard<std::mutex> scheduler_pause(sched.loop_mutex());
        group = orch.submit_next_level_group(C(75), {group_a, group_b}, cfg, {0, 1});
        single_a = orch.submit_next_level(C(76), single_tensor_args(0xF8, TensorArgType::OUTPUT), cfg, 0);
        single_b = orch.submit_next_level(C(77), single_tensor_args(0xF9, TensorArgType::OUTPUT), cfg, 1);
        worker_a.complete();
        worker_b.complete();

        auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
        while (manager.any_busy() && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        ASSERT_FALSE(manager.any_busy());
    }

    worker_a.wait_running();
    worker_b.wait_running();
    ASSERT_EQ(worker_a.dispatched_count(), 2);
    ASSERT_EQ(worker_b.dispatched_count(), 2);
    EXPECT_EQ(worker_a.dispatched[1].callable_hash0, 75u);
    EXPECT_EQ(worker_b.dispatched[1].callable_hash0, 75u);

    worker_a.complete();
    worker_b.complete();
    wait_consumed(group.task_slot);

    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
    while ((worker_a.dispatched_count() < 3 || worker_b.dispatched_count() < 3) &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_EQ(worker_a.dispatched_count(), 3);
    ASSERT_EQ(worker_b.dispatched_count(), 3);
    EXPECT_EQ(worker_a.dispatched[2].callable_hash0, 76u);
    EXPECT_EQ(worker_b.dispatched[2].callable_hash0, 77u);
    worker_a.complete();
    worker_b.complete();

    wait_consumed(running_a.task_slot);
    wait_consumed(running_b.task_slot);
    wait_consumed(single_a.task_slot);
    wait_consumed(single_b.task_slot);
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
    while (!orch.run_failed(run_id) && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_TRUE(orch.run_failed(run_id));
    EXPECT_EQ(S(slot).state.load(), TaskState::RUNNING);

    worker_b.complete();
    wait_consumed(slot);
    EXPECT_EQ(S(slot).state.load(), TaskState::CONSUMED);
}

TEST_F(GroupSchedulerFixture, CompletionRepairPreservesRunningPeersWhenOutcomesAreMissing) {
    TaskArgs a0 = single_tensor_args(0xC2, TensorArgType::OUTPUT);
    TaskArgs a1 = single_tensor_args(0xC3, TensorArgType::OUTPUT);
    auto res = orch.submit_next_level_group(C(43), {a0, a1}, cfg, {0, 1});
    TaskSlot slot = res.task_slot;

    worker_a.wait_running();
    worker_b.wait_running();
    {
        std::lock_guard<std::mutex> lk(S(slot).group_mu);
        ASSERT_EQ(S(slot).group_member_states.size(), 2u);
        ASSERT_EQ(S(slot).group_member_outcomes.size(), 2u);
        S(slot).group_member_outcomes.clear();
    }

    worker_a.complete_with_error("first member failed");
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
    bool repaired = false;
    while (!repaired && std::chrono::steady_clock::now() < deadline) {
        {
            std::lock_guard<std::mutex> lk(S(slot).group_mu);
            repaired = S(slot).group_member_states.size() == 2u && S(slot).group_member_outcomes.size() == 2u &&
                       S(slot).group_member_states[0] == GroupMemberState::FAILED &&
                       S(slot).group_member_states[1] == GroupMemberState::RUNNING;
        }
        if (!repaired) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_TRUE(repaired) << "worker_done indexed mismatched group bookkeeping instead of repairing both vectors";
    EXPECT_EQ(S(slot).state.load(std::memory_order_acquire), TaskState::RUNNING)
        << "repair discarded a live peer and let group failure consume its slot early";

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

    auto single_a = orch.submit_next_level(C(43), single_tensor_args(0xD2, TensorArgType::OUTPUT), cfg, 0);
    auto single_b = orch.submit_next_level(C(44), single_tensor_args(0xD3, TensorArgType::OUTPUT), cfg, 1);

    WorkerCompletion bad;
    bad.task_slot = slot;
    bad.group_index = 99;
    bad.outcome = EndpointOutcome::ENDPOINT_FAILURE;
    bad.error_message = "bad completion index";
    sched.worker_done(std::move(bad));

    wait_consumed(slot);
    EXPECT_EQ(S(slot).state.load(), TaskState::CONSUMED);

    {
        // Wait for the invalid completion's dispatch round to finish, then
        // make both terminalized group members idle while the scheduler loop
        // is paused. Their completion callbacks must provide the next wake.
        std::lock_guard<std::mutex> scheduler_pause(sched.loop_mutex());
        worker_a.complete();
        worker_b.complete();
        auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
        WorkerThread *manager_worker_a = manager.get_worker_by_id(WorkerType::NEXT_LEVEL, 0);
        WorkerThread *manager_worker_b = manager.get_worker_by_id(WorkerType::NEXT_LEVEL, 1);
        while ((!manager_worker_a->idle() || !manager_worker_b->idle()) &&
               std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        EXPECT_TRUE(manager_worker_a->idle());
        EXPECT_TRUE(manager_worker_b->idle());
    }

    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
    while ((worker_a.dispatched_count() < 2 || worker_b.dispatched_count() < 2) &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_EQ(worker_a.dispatched_count(), 2);
    EXPECT_EQ(worker_b.dispatched_count(), 2);

    // The EXPECTs above already recorded the failure; this retry exists only
    // to unblock cleanup so a missing wake fails the test instead of hanging
    // the fixture in Scheduler::stop().
    if (worker_a.dispatched_count() < 2 || worker_b.dispatched_count() < 2) {
        sched.notify_ready();
        deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
        while ((worker_a.dispatched_count() < 2 || worker_b.dispatched_count() < 2) &&
               std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    if (worker_a.dispatched_count() >= 2) worker_a.complete();
    if (worker_b.dispatched_count() >= 2) worker_b.complete();
    wait_consumed(single_a.task_slot);
    wait_consumed(single_b.task_slot);
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

// A producer that fails while its consumer is still being submitted. Wiring
// happens under each producer's fanout_mu, so the consumer is reachable from
// the failing producer's fanout list well before its own fanin/fanout counters
// are final — which is exactly what BUILDING marks. The poison must stop at the
// claim and leave the propagation to the submitting thread; running it from
// both sides releases every producer reference the consumer holds twice.
//
// The window is opened by holding the *second* producer's fanout_mu: submit
// wires the first producer, then parks on the second, and the failure is
// injected in between.
TEST_F(GroupSchedulerFixture, APoisonThatLandsMidSubmitLeavesThePropagationToSubmit) {
    auto failing = orch.submit_next_level(C(70), single_tensor_args(0xF100, TensorArgType::OUTPUT), cfg, 0);
    auto blocking = orch.submit_next_level(C(71), single_tensor_args(0xB200, TensorArgType::OUTPUT), cfg, 1);
    worker_a.wait_running();
    worker_b.wait_running();

    TaskArgs consumer_args;
    for (uint64_t key : {0xF100ULL, 0xB200ULL}) {
        ChipTensor t{};
        t.buffer.addr = key;
        t.ndims = 1;
        t.shapes[0] = 1;
        t.dtype = DataType::UINT8;
        consumer_args.add_tensor(t, TensorArgType::INPUT);
    }

    std::unique_lock<std::mutex> parked(S(blocking.task_slot).fanout_mu);
    std::thread submitter([&] {
        (void)orch.submit_next_level(C(72), consumer_args, cfg, 2);
    });

    // Wired into `failing` and now parked on `blocking`: the exact window.
    TaskSlot consumer = INVALID_SLOT;
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (consumer == INVALID_SLOT && std::chrono::steady_clock::now() < deadline) {
        std::lock_guard<std::mutex> lk(S(failing.task_slot).fanout_mu);
        if (!S(failing.task_slot).fanout_consumers.empty()) consumer = S(failing.task_slot).fanout_consumers[0];
    }
    ASSERT_NE(consumer, INVALID_SLOT) << "submit never reached the failing producer's fanout list";
    ASSERT_EQ(S(consumer).state.load(std::memory_order_acquire), TaskState::BUILDING);

    worker_a.complete_with_error("producer boom");
    while (S(consumer).state.load(std::memory_order_acquire) == TaskState::BUILDING &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_EQ(S(consumer).state.load(std::memory_order_acquire), TaskState::FAILED)
        << "the poison did not claim the consumer while it was building";

    parked.unlock();
    submitter.join();

    // One release per producer reference the consumer holds, from the one
    // thread that knows the wiring is final. `failing` reaches its threshold
    // (fanout_total 1 + the terminal self release) and no further.
    EXPECT_EQ(S(consumer).state.load(std::memory_order_acquire), TaskState::CONSUMED);
    EXPECT_EQ(S(failing.task_slot).fanout_released.load(std::memory_order_acquire), 2);
    EXPECT_EQ(S(blocking.task_slot).fanout_released.load(std::memory_order_acquire), 1);

    worker_b.complete();
    wait_consumed(blocking.task_slot);
}

TEST_F(GroupSchedulerFixture, TargetMustBeInEligibleEndpointSet) {
    TaskArgs args = single_tensor_args(0xE1, TensorArgType::OUTPUT);
    EXPECT_THROW((void)orch.submit_next_level(C(56), args, cfg, 0, {1}), std::invalid_argument);
}

TEST_F(GroupSchedulerFixture, UnknownEligibleWorkerIdIsRejectedBeforeScheduling) {
    TaskArgs args = single_tensor_args(0xE3, TensorArgType::OUTPUT);
    EXPECT_THROW((void)orch.submit_next_level(C(59), args, cfg, 99, {99}), std::invalid_argument);
}

TEST(SchedulerWorkerTargetTest, NextLevelTargetUsesWorkerIdNotVectorIndex) {
    TensorMap tm;
    Ring allocator;
    Scope scope;
    ReadyQueue rq_sub;
    NextLevelReadyQueues rq_next_level;
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
    manager.start(
        &allocator,
        [&sched](WorkerCompletion completion) {
            sched.worker_done(std::move(completion));
        },
        [&orch](WorkerDispatch dispatch) {
            orch.mark_task_accepted(dispatch.task_slot);
        },
        [&sched] {
            sched.notify_ready();
        }
    );
    rq_next_level.reset(manager.next_level_worker_ids());
    orch.init(&tm, &allocator, &scope, &rq_sub, &rq_next_level, &manager, [&sched] {
        sched.notify_ready();
    });
    (void)orch.begin_run();

    Scheduler::Config c;
    c.ring = &allocator;
    c.ready_sub_queue = &rq_sub;
    c.ready_next_level_queues = &rq_next_level;
    c.manager = &manager;
    c.enqueue_ready_cb = [&orch](TaskSlot slot) {
        orch.enqueue_ready(slot);
    };
    c.on_consumed_cb = [&orch, &consumed_slots, &consumed_mu](TaskSlot s) {
        orch.on_consumed(s);
        std::lock_guard<std::mutex> lk(consumed_mu);
        consumed_slots.push_back(s);
    };
    c.on_task_failed_cb = [&orch](TaskSlot s, const std::string &message) {
        orch.report_task_error(s, message);
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
    ChipTensor tensor{};
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
// Directed NEXT_LEVEL and shared SUB queues do not block each other. Covered
// here with one worker of each type: a SUB task submitted while the exact
// NEXT_LEVEL target is busy must still dispatch immediately.
// ===========================================================================

struct MixedTypeSchedulerFixture : public ::testing::Test {
    TensorMap tm;
    Ring allocator;
    Scope scope;
    ReadyQueue rq_sub;
    NextLevelReadyQueues rq_next_level;
    Orchestrator orch;
    MockMailboxWorker next_level_worker;
    MockMailboxWorker sub_worker;
    WorkerManager manager;
    Scheduler sched;
    CallConfig cfg;
    RunId run_id{INVALID_RUN_ID};

    std::vector<TaskSlot> consumed_slots;
    std::mutex consumed_mu;

    TaskSlotState &S(TaskSlot id) { return *allocator.slot_state(id); }

    void SetUp() override {
        allocator.init(/*heap_bytes=*/1ULL << 20);

        next_level_worker.start();
        sub_worker.start();
        manager.add_next_level(next_level_worker.mailbox_ptr());
        manager.add_sub(sub_worker.mailbox_ptr());
        manager.start(
            &allocator,
            [this](WorkerCompletion completion) {
                sched.worker_done(std::move(completion));
            },
            [this](WorkerDispatch dispatch) {
                orch.mark_task_accepted(dispatch.task_slot);
            },
            [this] {
                sched.notify_ready();
            }
        );
        rq_next_level.reset(manager.next_level_worker_ids());
        orch.init(&tm, &allocator, &scope, &rq_sub, &rq_next_level, &manager, [this] {
            sched.notify_ready();
        });
        run_id = orch.begin_run();

        Scheduler::Config c;
        c.ring = &allocator;
        c.ready_sub_queue = &rq_sub;
        c.ready_next_level_queues = &rq_next_level;
        c.manager = &manager;
        c.enqueue_ready_cb = [this](TaskSlot slot) {
            orch.enqueue_ready(slot);
        };
        // Same gate Worker::start installs: an active run that is also the
        // EXECUTING FIFO head and still owns its pipeline lease. Testing
        // against the weaker active_run_id() would let a slot dispatch here
        // that production refuses.
        c.active_run_cb = [this] {
            return orch.dispatchable_run_id();
        };
        c.on_consumed_cb = [this](TaskSlot s) {
            orch.on_consumed(s);
            std::lock_guard<std::mutex> lk(consumed_mu);
            consumed_slots.push_back(s);
        };
        c.on_task_failed_cb = [this](TaskSlot s, const std::string &message) {
            orch.report_task_error(s, message);
        };
        sched.start(c);
    }

    void TearDown() override {
        next_level_worker.drain();
        sub_worker.drain();
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
    // in worker 0's directed FIFO. The independent shared SUB queue must
    // dispatch immediately to the idle SUB worker.
    auto sub_args = single_tensor_args(0xBBB, TensorArgType::OUTPUT);
    auto sub = orch.submit_sub(C(7), sub_args);

    sub_worker.wait_running();
    EXPECT_TRUE(sub_worker.is_running.load());
    EXPECT_TRUE(next_level_worker.is_running.load()) << "chip worker must still be busy";

    // Complete the sub task first; it reaches CONSUMED while the chip task
    // is still running -- demonstrating independent queue dispatch.
    sub_worker.complete();
    wait_consumed(sub.task_slot);
    EXPECT_FALSE(is_consumed(chip.task_slot));

    next_level_worker.complete();
    wait_consumed(chip.task_slot);
}

TEST_F(MixedTypeSchedulerFixture, BusySubWorkerRequeuesWithinTheActiveRun) {
    auto first = orch.submit_sub(C(8), single_tensor_args(0xC01, TensorArgType::OUTPUT));
    sub_worker.wait_running();
    ASSERT_TRUE(sub_worker.is_running.load());

    auto second = orch.submit_sub(C(9), single_tensor_args(0xC02, TensorArgType::OUTPUT));
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    EXPECT_EQ(sub_worker.dispatched_count(), 1);

    sub_worker.complete();
    wait_consumed(first.task_slot);
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
    while (sub_worker.dispatched_count() < 2 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_EQ(sub_worker.dispatched_count(), 2);
    EXPECT_TRUE(sub_worker.is_running.load());

    sub_worker.complete();
    wait_consumed(second.task_slot);
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
    EXPECT_EQ(total, 3);  // 2 from group A + exactly 1 downstream dispatch

    if (worker_a.is_running.load()) worker_a.complete();
    if (worker_b.is_running.load()) worker_b.complete();
    wait_consumed(b.task_slot);
    (void)a;  // suppress unused
}
