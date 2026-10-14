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
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "call_config.h"
#include "common/host_span.h"
#include "common/host_span_names.h"
#include "common/host_span_scope.h"
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
std::vector<std::string> captured_host_span_attributes;
std::atomic<bool> captured_host_spans_enabled{true};

class ScopedHostSpanCapture {
public:
    explicit ScopedHostSpanCapture(bool enabled = true) {
        captured_host_spans_enabled.store(enabled, std::memory_order_release);
        std::lock_guard<std::mutex> lk(captured_host_spans_mu);
        captured_host_span_names.clear();
        captured_host_span_attributes.clear();
    }

    ~ScopedHostSpanCapture() { captured_host_spans_enabled.store(true, std::memory_order_release); }

    ScopedHostSpanCapture(const ScopedHostSpanCapture &) = delete;
    ScopedHostSpanCapture &operator=(const ScopedHostSpanCapture &) = delete;
};

bool captured_host_span(const std::string &name) {
    std::lock_guard<std::mutex> lk(captured_host_spans_mu);
    return std::find(captured_host_span_names.begin(), captured_host_span_names.end(), name) !=
           captured_host_span_names.end();
}

bool captured_host_spans_empty() {
    std::lock_guard<std::mutex> lk(captured_host_spans_mu);
    return captured_host_span_names.empty();
}

// The attributes of the first span emitted under `name`, or "" if none was.
std::string captured_host_span_attrs(const std::string &name) {
    std::lock_guard<std::mutex> lk(captured_host_spans_mu);
    for (size_t i = 0; i < captured_host_span_names.size(); ++i) {
        if (captured_host_span_names[i] == name) return captured_host_span_attributes[i];
    }
    return "";
}

}  // namespace

extern "C" int unified_log_host_span_enabled() {
    return captured_host_spans_enabled.load(std::memory_order_acquire) ? 1 : 0;
}

extern "C" void unified_log_host_span(const SimplerHostSpan *span) {
    if (span == nullptr || span->name == nullptr) return;
    std::lock_guard<std::mutex> lk(captured_host_spans_mu);
    captured_host_span_names.emplace_back(span->name);
    captured_host_span_attributes.emplace_back(span->attributes == nullptr ? "" : span->attributes);
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
        uint64_t tensor_key;  // first tensor's identity buffer_id (unique per submit in tests)
        // Which request frame carried this payload, which run-state entry its header named, and
        // whether the parent asked for a preparation rather than an ordinary dispatch. Defaulted
        // so the two-field aggregate initializer the older cases use still compiles.
        size_t frame_index{0};
        size_t entry_index{0};
        bool prepare_only{false};
    };

    /** A payload this child has copied out but whose hand-back it is deliberately withholding. */
    struct HeldFrame {
        size_t frame_index;
        size_t entry_index;
        uint64_t frame_generation;
        bool prepare_only;
    };

    /** No run is staged, so acceptance has no entry to reach. */
    static constexpr size_t kNoEntry = static_cast<size_t>(-1);

    alignas(8) std::array<char, MAILBOX_SIZE> mailbox{};
    // What this child's parent granted it, which is what bounds the entry table it may address.
    // `MAILBOX_SIZE` is sized for `MAILBOX_DEFAULT_GRANTED_SETS`, and every endpoint in this file
    // is constructed over that region, so this is the width of the table that exists.
    size_t granted_sets{MAILBOX_DEFAULT_GRANTED_SETS};
    // The entry the run currently staged occupies, taken from the header of the frame its payload
    // arrived on. Read by `write_task_accepted` from the test's thread.
    std::atomic<size_t> current_entry{kNoEntry};
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
    // Whether this child writes its own acceptance word. Off by default, because the older cases
    // publish acceptance from the test thread at a point of their own choosing.
    std::atomic<bool> auto_accept{false};
    std::thread loop_thread;

    // Frame withholding. A child owns a request frame from the instant it reads TASK_READY until
    // its acknowledgement, and the protocol puts no bound on how long that is — so a fixture that
    // wants the parent's transport full simply stops before the hand-back. Guarded by `hold_mu`.
    std::mutex hold_mu;
    std::condition_variable hold_cv;
    size_t hold_budget{0};
    std::vector<HeldFrame> held;
    std::vector<HeldFrame> releasing;
    // Request frames this child has acknowledged, counted at the acknowledgement itself.
    std::atomic<size_t> acknowledged{0};
    // Entries this child has staged and that are waiting for the parent's ACTIVATE.
    std::vector<size_t> staged;

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
        release_all_held_frames();
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
    // this call — completes itself, so Scheduler::stop() can always join. Any
    // frame still withheld is released too, since a frame this child never
    // hands back is one the parent can never reclaim.
    void drain() {
        drain_mode.store(true, std::memory_order_release);
        release_all_held_frames();
        std::lock_guard<std::mutex> lk(run_mu);
        should_complete.store(true, std::memory_order_release);
        run_cv.notify_one();
    }

    /** Withhold the hand-back of the next `count` payloads this child copies out. */
    void hold_next_request_frames(size_t count) {
        std::lock_guard<std::mutex> lk(hold_mu);
        hold_budget = count;
    }

    size_t held_frame_count() {
        std::lock_guard<std::mutex> lk(hold_mu);
        return held.size();
    }

    bool wait_held_frames(size_t count, int timeout_ms = 3000) {
        std::unique_lock<std::mutex> lk(hold_mu);
        return hold_cv.wait_for(lk, std::chrono::milliseconds(timeout_ms), [this, count] {
            return held.size() >= count;
        });
    }

    /**
     * Hand back the oldest withheld frame.
     *
     * Only queues it: every store the protocol defines is made on this child's own loop thread,
     * so a release from the test thread cannot reorder itself against the payload copy that
     * preceded it.
     */
    void release_one_held_frame() {
        std::lock_guard<std::mutex> lk(hold_mu);
        if (held.empty()) return;
        releasing.push_back(held.front());
        held.erase(held.begin());
    }

    void release_all_held_frames() {
        std::lock_guard<std::mutex> lk(hold_mu);
        releasing.insert(releasing.end(), held.begin(), held.end());
        held.clear();
        hold_budget = 0;
    }

    /** Wait until this child has acknowledged at least `count` request frames. */
    bool wait_acknowledged(size_t count, int timeout_ms = 3000) {
        auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        while (std::chrono::steady_clock::now() < deadline) {
            if (acknowledged.load(std::memory_order_acquire) >= count) return true;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return acknowledged.load(std::memory_order_acquire) >= count;
    }

    // The child publishes acceptance into its run's entry, not the frame that
    // delivered the payload: the device writes this word after the frame has
    // already been handed back. Which entry that is comes from the header of the
    // frame the payload arrived on, which `loop` recorded when it staged the run
    // — never a fixed index, because the parent allocates entries in submission
    // order and a later dispatch owns a later one.
    void write_task_accepted() {
        const size_t entry_index = current_entry.load(std::memory_order_acquire);
        if (entry_index == kNoEntry) return;
        int32_t v = MAILBOX_TASK_ACCEPTED;
        __atomic_store(&mailbox_run_entry(mailbox.data(), entry_index)->accepted, &v, __ATOMIC_RELEASE);
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

    bool wait_dispatched(size_t count, int timeout_ms = 3000) {
        auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        while (std::chrono::steady_clock::now() < deadline) {
            {
                std::lock_guard<std::mutex> lk(dispatched_mu);
                if (dispatched.size() >= count) return true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        std::lock_guard<std::mutex> lk(dispatched_mu);
        return dispatched.size() >= count;
    }

    std::vector<Record> dispatched_records() {
        std::lock_guard<std::mutex> lk(dispatched_mu);
        return dispatched;
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

    char *task_frame(size_t index) { return mailbox.data() + (MAILBOX_FIRST_TASK_FRAME + index) * MAILBOX_FRAME_SIZE; }

    MailboxState read_task_state(size_t index) const {
        const auto *ptr = reinterpret_cast<const volatile int32_t *>(
            mailbox.data() + (MAILBOX_FIRST_TASK_FRAME + index) * MAILBOX_FRAME_SIZE + MAILBOX_OFF_STATE
        );
        int32_t v = __atomic_load_n(ptr, __ATOMIC_ACQUIRE);
        return static_cast<MailboxState>(v);
    }

    void write_task_state(size_t index, MailboxState state) {
        auto *ptr = reinterpret_cast<volatile int32_t *>(task_frame(index) + MAILBOX_OFF_STATE);
        __atomic_store_n(ptr, static_cast<int32_t>(state), __ATOMIC_RELEASE);
    }

    /**
     * Hand one request frame back, in the order the protocol requires.
     *
     * The frame's own state word first and the consumption ACK last: the parent treats that ACK
     * alone as permission to refill the frame, so an `IDLE` sequenced after it would erase a
     * payload the parent had already published. A level, not a pulse — the parent compares the
     * echoed value against the generation it published.
     */
    void return_request_frame(size_t frame_index, MailboxRunEntry *entry, uint64_t frame_generation) {
        write_task_state(frame_index, MailboxState::IDLE);
        __atomic_store_n(&entry->frame_consumed_generation, frame_generation, __ATOMIC_RELEASE);
        // Published after the acknowledgement it counts, so a test that observes this count has
        // also observed the `IDLE` that preceded it — which is what lets it tell a republication
        // apart from the publication that was already on the frame.
        acknowledged.fetch_add(1, std::memory_order_release);
    }

    /** What this child resolved a preparation to, read by the parent with the staged state. */
    void set_entry_disposition(MailboxRunEntry *entry, MailboxPreparationDisposition disposition) {
        __atomic_store_n(&entry->preparation_disposition, static_cast<int32_t>(disposition), __ATOMIC_RELEASE);
    }

    /** Publish one run's terminal into its entry: rc and text first, the state last. */
    void publish_terminal(MailboxRunEntry *entry, int32_t error_code, const std::string &error_msg) {
        entry->error = error_code;
        std::memset(entry->error_text, 0, MAILBOX_ERROR_MSG_SIZE);
        if (!error_msg.empty()) {
            const size_t n = std::min(error_msg.size(), static_cast<size_t>(MAILBOX_ERROR_MSG_SIZE - 1));
            std::memcpy(entry->error_text, error_msg.data(), n);
        }
        __atomic_store_n(&entry->state, static_cast<int32_t>(MailboxState::TASK_DONE), __ATOMIC_RELEASE);
    }

    /** Take one hand-off: resolve its entry from the header, copy the payload out, give the frame back. */
    bool serve_one_request(size_t frame_index, bool prepare_only) {
        char *frame = task_frame(frame_index);
        uint64_t entry_index = 0;
        uint64_t entry_generation = 0;
        uint64_t frame_generation = 0;
        std::memcpy(&entry_index, frame + MAILBOX_OFF_FRAME_ENTRY_INDEX, sizeof(entry_index));
        std::memcpy(&entry_generation, frame + MAILBOX_OFF_FRAME_ENTRY_GENERATION, sizeof(entry_generation));
        std::memcpy(&frame_generation, frame + MAILBOX_OFF_FRAME_GENERATION, sizeof(frame_generation));
        if (entry_index >= mailbox_run_entry_count(granted_sets)) {
            // Nothing in this header names a live owner, so no entry is touched; the frame comes
            // back so the loop stops re-reading it.
            write_task_state(frame_index, MailboxState::IDLE);
            return false;
        }
        MailboxRunEntry *entry = mailbox_run_entry(mailbox.data(), static_cast<size_t>(entry_index));
        if (__atomic_load_n(&entry->entry_generation, __ATOMIC_ACQUIRE) != entry_generation) {
            // The entry moved on; this frame is left over from a previous occupant of it and the
            // parent has already reclaimed it from its own record.
            write_task_state(frame_index, MailboxState::IDLE);
            return false;
        }

        uint8_t callable_hash0 = static_cast<uint8_t>(frame[MAILBOX_OFF_TASK_CALLABLE_HASH]);
        // The frame carries the wire blob, so decode it the way a real child does.
        TaskArgsView view =
            read_blob(reinterpret_cast<const uint8_t *>(frame) + MAILBOX_OFF_TASK_ARGS_BLOB, MAILBOX_ARGS_CAPACITY);
        uint64_t tensor_key = 0;
        if (view.tensor_count > 0) {
            tensor_key = view.tensors(0).buffer.identity.buffer_id;
        }
        {
            std::lock_guard<std::mutex> lk(dispatched_mu);
            dispatched.push_back(
                {callable_hash0, tensor_key, frame_index, static_cast<size_t>(entry_index), prepare_only}
            );
        }
        const HeldFrame copied{frame_index, static_cast<size_t>(entry_index), frame_generation, prepare_only};
        {
            // The payload is out and nothing about this run has been published yet, which is the
            // one point where a child may simply stop. A fixture that put a budget here wants the
            // parent's transport full; every store the protocol defines still happens on this
            // thread, when the frame is released.
            std::lock_guard<std::mutex> lk(hold_mu);
            if (hold_budget > 0) {
                --hold_budget;
                held.push_back(copied);
                hold_cv.notify_all();
                return true;
            }
        }
        publish_consumption(copied);
        return true;
    }

    /**
     * Everything a child does after the payload is out: hand the frame back, then either stage the
     * run or execute it.
     *
     * The hand-back comes first in both shapes, which is the whole point of transport being
     * separate from the run — holding the frame across the execution below would stop the parent
     * admitting anything else into this endpoint for as long as the run takes.
     */
    void publish_consumption(const HeldFrame &frame) {
        MailboxRunEntry *entry = mailbox_run_entry(mailbox.data(), frame.entry_index);
        if (frame.prepare_only) {
            set_entry_disposition(entry, MailboxPreparationDisposition::NATIVE_PREPARED);
            return_request_frame(frame.frame_index, entry, frame.frame_generation);
            __atomic_store_n(&entry->state, static_cast<int32_t>(MailboxState::FRAME_STAGED), __ATOMIC_RELEASE);
            staged.push_back(frame.entry_index);
            return;
        }
        return_request_frame(frame.frame_index, entry, frame.frame_generation);
        execute_entry(frame.entry_index, entry);
    }

    /** Run one entry to its terminal: the parked wait, then the acceptance and the terminal. */
    void execute_entry(size_t entry_index, MailboxRunEntry *entry) {
        current_entry.store(entry_index, std::memory_order_release);
        is_running.store(true, std::memory_order_release);
        if (auto_accept.load(std::memory_order_acquire)) {
            __atomic_store_n(&entry->state, static_cast<int32_t>(MailboxState::TASK_LAUNCHED), __ATOMIC_RELEASE);
            int32_t accepted = MAILBOX_TASK_ACCEPTED;
            __atomic_store(&entry->accepted, &accepted, __ATOMIC_RELEASE);
        }

        {
            std::unique_lock<std::mutex> lk(run_mu);
            run_cv.wait(lk, [this] {
                return should_complete.load(std::memory_order_acquire) || drain_mode.load(std::memory_order_acquire);
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

        // The run's own terminal, into its own entry, and this side's last access to it.
        publish_terminal(entry, error_code, error_msg);
    }

    /** Hand back one frame the fixture released, and carry on from where the hold stopped. */
    bool resume_one_released_frame() {
        HeldFrame frame{};
        {
            std::lock_guard<std::mutex> lk(hold_mu);
            if (releasing.empty()) return false;
            frame = releasing.front();
            releasing.erase(releasing.begin());
        }
        publish_consumption(frame);
        return true;
    }

    /** Execute the first staged run the parent has promoted to ACTIVATE. */
    bool serve_one_activation() {
        for (size_t position = 0; position < staged.size(); ++position) {
            const size_t entry_index = staged[position];
            MailboxRunEntry *entry = mailbox_run_entry(mailbox.data(), entry_index);
            if (static_cast<MailboxState>(__atomic_load_n(&entry->state, __ATOMIC_ACQUIRE)) != MailboxState::ACTIVATE) {
                continue;
            }
            staged.erase(staged.begin() + static_cast<long>(position));
            execute_entry(entry_index, entry);
            return true;
        }
        return false;
    }

    bool frame_is_withheld(size_t frame_index) {
        std::lock_guard<std::mutex> lk(hold_mu);
        for (const HeldFrame &frame : held) {
            if (frame.frame_index == frame_index) return true;
        }
        for (const HeldFrame &frame : releasing) {
            if (frame.frame_index == frame_index) return true;
        }
        return false;
    }

    void loop() {
        while (true) {
            if (stop_flag.load(std::memory_order_acquire)) return;
            if (resume_one_released_frame()) continue;
            if (serve_one_activation()) continue;
            MailboxState s = read_state();
            size_t ready_frame = MAILBOX_REQUEST_FRAMES;
            bool ready_prepare_only = false;
            for (size_t index = 0; index < MAILBOX_REQUEST_FRAMES; ++index) {
                if (frame_is_withheld(index)) continue;
                const MailboxState frame_state = read_task_state(index);
                if (frame_state == MailboxState::TASK_READY || frame_state == MailboxState::PREPARE_READY) {
                    ready_frame = index;
                    ready_prepare_only = frame_state == MailboxState::PREPARE_READY;
                    break;
                }
            }
            if (ready_frame < MAILBOX_REQUEST_FRAMES) {
                (void)serve_one_request(ready_frame, ready_prepare_only);
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

    // Immediate success: submit_progress queues the completion the very next
    // poll_progress reports, so a dispatch through this endpoint finishes
    // without any test-side stepping.
    void submit_progress(Ring *, const WorkerDispatch &dispatch) override {
        std::lock_guard<std::mutex> lk(mu_);
        WorkerEndpointProgress progress;
        progress.kind = WorkerProgressKind::COMPLETED;
        progress.dispatch = dispatch;
        progress.completion.task_slot = dispatch.task_slot;
        progress.completion.group_index = dispatch.group_index;
        progress.completion.outcome = EndpointOutcome::SUCCESS;
        events_.push_back(progress);
    }

    bool poll_progress(WorkerEndpointProgress &progress) override {
        std::lock_guard<std::mutex> lk(mu_);
        if (events_.empty()) return false;
        progress = events_.front();
        events_.pop_front();
        return true;
    }

    bool activate_progress(RunId) override { return true; }

    void control_prepare(const uint8_t *) override {
        if (prepare_count_ != nullptr) prepare_count_->fetch_add(1, std::memory_order_relaxed);
    }

private:
    WorkerEndpointCaps caps_;
    std::atomic<int> *prepare_count_{nullptr};
    std::mutex mu_;
    std::deque<WorkerEndpointProgress> events_;
};

class DeterministicProgressEndpoint final : public WorkerEndpoint {
public:
    explicit DeterministicProgressEndpoint(int32_t worker_id = 0, uint32_t max_inflight_tasks = 2) {
        caps_.worker_id = worker_id;
        caps_.max_inflight_tasks = max_inflight_tasks;
        caps_.supports_frame_staging = true;
    }

    const WorkerEndpointCaps &caps() const override {
        caps_call_count_.fetch_add(1, std::memory_order_relaxed);
        return caps_;
    }

    size_t caps_call_count() const { return caps_call_count_.load(std::memory_order_relaxed); }

    void submit_progress(Ring *ring, const WorkerDispatch &dispatch) override {
        ProgressCall call(*this);
        {
            std::unique_lock<std::mutex> gate_lk(submit_gate_mu_);
            if (block_submit_) {
                submit_entered_ = true;
                submit_gate_cv_.notify_all();
                submit_gate_cv_.wait(gate_lk, [this] {
                    return release_submit_;
                });
                block_submit_ = false;
            }
        }
        RunId run_id = INVALID_RUN_ID;
        if (ring != nullptr) {
            TaskSlotState *slot = ring->slot_state(dispatch.task_slot);
            if (slot != nullptr) run_id = slot->run_id;
        }
        std::lock_guard<std::mutex> lk(mu_);
        submitted_.push_back(dispatch);
        outstanding_.emplace(dispatch.dispatch_id, Outstanding{dispatch, run_id});
        cv_.notify_all();
        if (throw_submit_after_publication_) {
            throw_submit_after_publication_ = false;
            throw std::runtime_error("injected submit failure after publication");
        }
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

    bool report_submission_error(const WorkerDispatch &dispatch, const std::string &reason) override {
        ProgressCall call(*this);
        std::lock_guard<std::mutex> lk(mu_);
        const bool endpoint_owned = outstanding_.count(dispatch.dispatch_id) != 0;
        progress_error_ = reason;
        terminalize_outstanding_locked();
        cv_.notify_all();
        return endpoint_owned;
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

    void block_next_submit() {
        std::lock_guard<std::mutex> lk(submit_gate_mu_);
        block_submit_ = true;
        submit_entered_ = false;
        release_submit_ = false;
    }

    bool wait_submit_entered(std::chrono::milliseconds timeout = std::chrono::seconds(3)) {
        std::unique_lock<std::mutex> lk(submit_gate_mu_);
        return submit_gate_cv_.wait_for(lk, timeout, [this] {
            return submit_entered_;
        });
    }

    void release_blocked_submit() {
        std::lock_guard<std::mutex> lk(submit_gate_mu_);
        release_submit_ = true;
        submit_gate_cv_.notify_all();
    }

    void throw_after_next_submit_publication() {
        std::lock_guard<std::mutex> lk(mu_);
        throw_submit_after_publication_ = true;
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
    std::thread::id progress_owner() const {
        std::lock_guard<std::mutex> lk(owner_mu_);
        return progress_owner_;
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
    mutable std::atomic<size_t> caps_call_count_{0};
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
    bool throw_submit_after_publication_{false};
    std::string progress_error_;
    std::mutex submit_gate_mu_;
    std::condition_variable submit_gate_cv_;
    bool block_submit_{false};
    bool submit_entered_{false};
    bool release_submit_{false};
    std::atomic<int> concurrent_calls_{0};
    std::atomic<int> max_concurrent_calls_{0};
    mutable std::mutex owner_mu_;
    std::thread::id progress_owner_{};
    bool progress_owner_changed_{false};
};

static char *test_task_frame(std::array<char, MAILBOX_SIZE> &mailbox, size_t index) {
    return mailbox.data() + (MAILBOX_FIRST_TASK_FRAME + index) * MAILBOX_FRAME_SIZE;
}

static MailboxState mailbox_state_of(const char *frame) {
    const auto *state = reinterpret_cast<const int32_t *>(frame + MAILBOX_OFF_STATE);
    return static_cast<MailboxState>(__atomic_load_n(state, __ATOMIC_ACQUIRE));
}

// A run's state lives in its own entry, not in the frame that delivered its
// payload, so everything a child publishes about a run is written here. Entries
// are allocated in submission order, which is what `index` means.
static MailboxRunEntry *test_entry(std::array<char, MAILBOX_SIZE> &mailbox, size_t index) {
    return mailbox_run_entry(mailbox.data(), index);
}

static MailboxState test_entry_state(const MailboxRunEntry *entry) {
    return static_cast<MailboxState>(__atomic_load_n(&entry->state, __ATOMIC_ACQUIRE));
}

static void set_test_entry_state(MailboxRunEntry *entry, MailboxState state) {
    __atomic_store_n(&entry->state, static_cast<int32_t>(state), __ATOMIC_RELEASE);
}

static void set_test_entry_accepted(MailboxRunEntry *entry) {
    __atomic_store_n(&entry->accepted, MAILBOX_TASK_ACCEPTED, __ATOMIC_RELEASE);
}

static void set_test_entry_disposition(MailboxRunEntry *entry, MailboxPreparationDisposition disposition) {
    __atomic_store_n(&entry->preparation_disposition, static_cast<int32_t>(disposition), __ATOMIC_RELEASE);
}

static void set_test_entry_error(MailboxRunEntry *entry, int32_t code, const char *text) {
    entry->error = code;
    std::snprintf(entry->error_text, MAILBOX_ERROR_MSG_SIZE, "%s", text == nullptr ? "" : text);
}

/**
 * Hand one request frame back without staging the run it carried.
 *
 * Where a child sits once it has copied a payload out and before it publishes anything about the
 * run: the frame is the parent's again and the entry is still this side's. The frame's own state
 * word goes first and the ACK last, because the parent treats that ACK alone as permission to
 * refill the frame — an `IDLE` after it would erase whatever was published in between.
 */
static void test_child_return_frame(
    std::array<char, MAILBOX_SIZE> &mailbox, size_t frame_index, MailboxRunEntry *entry, uint64_t frame_generation
) {
    auto *frame_state = reinterpret_cast<volatile int32_t *>(test_task_frame(mailbox, frame_index) + MAILBOX_OFF_STATE);
    __atomic_store_n(frame_state, static_cast<int32_t>(MailboxState::IDLE), __ATOMIC_RELEASE);
    __atomic_store_n(&entry->frame_consumed_generation, frame_generation, __ATOMIC_RELEASE);
}

/**
 * The same hand-back for a raw mapping, which the forked-child fixtures use.
 *
 * Which entry this frame belongs to comes from the frame's own header, and the generation there is
 * checked against the entry's before anything is written: that is the rule the child loop follows,
 * and it is what makes the ACK a statement about this run rather than about whichever run holds
 * that entry now. A header that does not match leaves the entry untouched and returns false.
 *
 * The payload is read before the ACK, because the hand-back is what a child does *after* copying
 * the bytes out — acknowledging without reading is a hand-back with nothing behind it. This is a
 * simulated consumption: it touches the digest and the blob's length prefix at the offsets the
 * real child reads, and does not decode the args. `entry_count` bounds the table this mapping has.
 */
static bool test_child_consume_and_return_frame(void *mailbox, size_t frame_index, size_t entry_count) {
    char *frame = static_cast<char *>(mailbox) + (MAILBOX_FIRST_TASK_FRAME + frame_index) * MAILBOX_FRAME_SIZE;
    uint64_t entry_index = 0;
    uint64_t header_generation = 0;
    uint64_t frame_generation = 0;
    std::memcpy(&entry_index, frame + MAILBOX_OFF_FRAME_ENTRY_INDEX, sizeof(entry_index));
    std::memcpy(&header_generation, frame + MAILBOX_OFF_FRAME_ENTRY_GENERATION, sizeof(header_generation));
    std::memcpy(&frame_generation, frame + MAILBOX_OFF_FRAME_GENERATION, sizeof(frame_generation));
    if (entry_index >= entry_count) return false;
    MailboxRunEntry *entry = mailbox_run_entry(mailbox, static_cast<size_t>(entry_index));
    if (__atomic_load_n(&entry->entry_generation, __ATOMIC_ACQUIRE) != header_generation) return false;

    // The payload, at the offsets the child reads it from. Consumed into locals and discarded:
    // what this fixture is about is the hand-back, not the decode.
    volatile uint8_t digest0 = static_cast<uint8_t>(frame[MAILBOX_OFF_TASK_CALLABLE_HASH]);
    (void)digest0;
    int32_t blob_bytes = 0;
    std::memcpy(&blob_bytes, frame + MAILBOX_OFF_TASK_ARGS_BLOB, sizeof(blob_bytes));
    (void)blob_bytes;

    auto *frame_state = reinterpret_cast<volatile int32_t *>(frame + MAILBOX_OFF_STATE);
    __atomic_store_n(frame_state, static_cast<int32_t>(MailboxState::IDLE), __ATOMIC_RELEASE);
    __atomic_store_n(&entry->frame_consumed_generation, frame_generation, __ATOMIC_RELEASE);
    return true;
}

/**
 * The child's side of one hand-off: take the payload and say so.
 *
 * Echoes the frame's own generation into the entry it names, which is what lets
 * the parent reclaim that frame while the run is still executing, and leaves the
 * entry staged. Faithful to the real loop's order — the echo before the state —
 * so a fixture that drives this reaches the same states a child would.
 */
static void test_child_consume_frame(
    std::array<char, MAILBOX_SIZE> &mailbox, size_t frame_index, MailboxPreparationDisposition disposition
) {
    char *frame = test_task_frame(mailbox, frame_index);
    uint64_t entry_index = 0;
    uint64_t frame_generation = 0;
    std::memcpy(&entry_index, frame + MAILBOX_OFF_FRAME_ENTRY_INDEX, sizeof(entry_index));
    std::memcpy(&frame_generation, frame + MAILBOX_OFF_FRAME_GENERATION, sizeof(frame_generation));
    MailboxRunEntry *entry = test_entry(mailbox, static_cast<size_t>(entry_index));
    set_test_entry_disposition(entry, disposition);
    test_child_return_frame(mailbox, frame_index, entry, frame_generation);
    set_test_entry_state(entry, MailboxState::FRAME_STAGED);
}

static uint64_t test_entry_dispatch_id(const MailboxRunEntry *entry) { return entry->dispatch_id; }
static uint64_t test_entry_task_slot(const MailboxRunEntry *entry) { return entry->task_slot; }
static uint64_t test_entry_group_index(const MailboxRunEntry *entry) { return entry->group_index; }
static uint64_t test_entry_group_size(const MailboxRunEntry *entry) { return entry->group_size; }

/** Which entry the payload in request frame `frame_index` belongs to. */
static size_t test_frame_entry_index(std::array<char, MAILBOX_SIZE> &mailbox, size_t frame_index) {
    uint64_t entry_index = 0;
    std::memcpy(
        &entry_index, test_task_frame(mailbox, frame_index) + MAILBOX_OFF_FRAME_ENTRY_INDEX, sizeof(entry_index)
    );
    return static_cast<size_t>(entry_index);
}

/**
 * Put every one of `slots` in flight at once on a granted endpoint.
 *
 * Transport is narrower than the grant, so this is the sequence a child and parent actually run
 * to reach `M` live runs: each hand-off's payload is copied out and its frame acknowledged while
 * the run it carried keeps going, which is what frees the frame for the next one. Returns the
 * run-state entry each dispatch landed on, in submission order, or an empty vector if any
 * hand-back was refused.
 */
static std::vector<size_t> test_fill_every_entry(
    LocalMailboxEndpoint &endpoint, Ring &allocator, std::array<char, MAILBOX_SIZE> &mailbox,
    const std::vector<TaskSlot> &slots, uint32_t granted_sets, uint64_t first_dispatch_id
) {
    const size_t entry_count = mailbox_run_entry_count(granted_sets);
    std::vector<size_t> entry_of;
    for (size_t index = 0; index < slots.size(); ++index) {
        // A pure query that reclaims first, so this is the parent noticing the previous
        // acknowledgement rather than a frame being handed back twice.
        if (endpoint.available_request_transport() == 0) return {};
        endpoint.submit_progress(
            &allocator, WorkerDispatch{slots[index], 0, first_dispatch_id + index, /*prepare_only=*/false}
        );
        // Exactly one frame can be unconsumed here, because the previous one was acknowledged
        // before this publication.
        size_t carrier = MAILBOX_REQUEST_FRAMES;
        for (size_t frame = 0; frame < MAILBOX_REQUEST_FRAMES; ++frame) {
            if (mailbox_state_of(test_task_frame(mailbox, frame)) == MailboxState::TASK_READY) {
                carrier = frame;
                break;
            }
        }
        if (carrier == MAILBOX_REQUEST_FRAMES) return {};
        entry_of.push_back(test_frame_entry_index(mailbox, carrier));
        if (!test_child_consume_and_return_frame(mailbox.data(), carrier, entry_count)) return {};
    }
    return entry_of;
}

/** A distinct message that fills the entry's inline text to its last usable byte. */
static std::string test_saturating_error_text(char fill, const std::string &prefix) {
    std::string text = prefix;
    text.resize(MAILBOX_ERROR_MSG_SIZE - 1, fill);
    return text;
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

// Poll an endpoint until it reports one progress event, or the budget expires.
static bool poll_until(WorkerEndpoint &endpoint, WorkerEndpointProgress &progress) {
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (std::chrono::steady_clock::now() < deadline) {
        if (endpoint.poll_progress(progress)) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return false;
}

// ---------------------------------------------------------------------------
// Helper: build a TaskArgs whose only tensor has the given (data, tag).
// ---------------------------------------------------------------------------

// A valid wire Tensor over a 1-byte POSIX_SHM backing, keyed by `buffer_id`. The dispatch path
// decodes it with TaskArgsView::tensors, which validates every element, so the descriptor must be
// well-formed (magic, non-zero generation, strides, an extent that fits the backing).
static Tensor wire_tensor(uint64_t buffer_id) {
    Tensor t{};
    t.buffer.magic = BUFFER_DESCRIPTOR_MAGIC;
    t.buffer.address_space = static_cast<uint8_t>(AddressSpace::HOST);
    t.buffer.access = static_cast<uint8_t>(AccessMode::READWRITE);
    t.buffer.backend_kind = static_cast<uint8_t>(BackendKind::POSIX_SHM);
    t.buffer.nbytes = 1;
    t.buffer.identity.buffer_id = buffer_id;
    t.buffer.identity.generation = 1;
    // A POSIX_SHM descriptor names its backing; nothing here opens it, but the wire schema requires
    // the name to be present and printable, and every element a TaskArgsView hands out is validated.
    const std::string shm_name = "psm_sched_" + std::to_string(buffer_id);
    t.buffer.body_len = static_cast<uint16_t>(shm_name.size());
    std::memcpy(t.buffer.body, shm_name.data(), shm_name.size());
    t.ndims = 1;
    t.shapes[0] = 1;
    t.strides[0] = 1;
    t.dtype = DataType::UINT8;
    return t;
}

static TaskArgs single_tensor_args(uint64_t data_ptr, TensorArgType tag) {
    TaskArgs a;
    Tensor t = wire_tensor(data_ptr);
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

TEST(WorkerManagerTest, SchedulerOwnsProgressAcrossWorkerEndpoints) {
    Ring allocator;
    ReadyQueue ready_sub;
    NextLevelReadyQueues ready_next;
    WorkerManager manager;
    Scheduler scheduler;
    allocator.init(/*heap_bytes=*/0);

    auto endpoint0 = std::make_unique<DeterministicProgressEndpoint>(0, 1);
    auto endpoint1 = std::make_unique<DeterministicProgressEndpoint>(1, 1);
    DeterministicProgressEndpoint *endpoint0_ptr = endpoint0.get();
    DeterministicProgressEndpoint *endpoint1_ptr = endpoint1.get();
    manager.add_next_level_endpoint(std::move(endpoint0));
    manager.add_next_level_endpoint(std::move(endpoint1));
    manager.start(
        &allocator,
        [&scheduler](WorkerCompletion completion) {
            scheduler.worker_done(std::move(completion));
        },
        [](WorkerDispatch) {}
    );
    ready_next.reset(manager.next_level_worker_ids());

    auto enqueue = [&](int32_t worker_id, RunId run_id) {
        TaskSlot slot = make_progress_slot(allocator, run_id, /*pipeline_slot=*/0, /*generation=*/run_id);
        TaskSlotState &state = *allocator.slot_state(slot);
        state.worker_type = WorkerType::NEXT_LEVEL;
        state.target_worker_ids.push_back(worker_id);
        state.state.store(TaskState::READY, std::memory_order_release);
        ready_next.push_single(worker_id, slot);
        return slot;
    };
    TaskSlot slot0 = enqueue(/*worker_id=*/0, /*run_id=*/81);
    TaskSlot slot1 = enqueue(/*worker_id=*/1, /*run_id=*/82);

    Scheduler::Config config;
    config.ring = &allocator;
    config.ready_sub_queue = &ready_sub;
    config.ready_next_level_queues = &ready_next;
    config.manager = &manager;
    config.enqueue_ready_cb = [&](TaskSlot slot) {
        TaskSlotState &state = *allocator.slot_state(slot);
        ready_next.push_single(state.target_worker_id(0), slot);
    };
    scheduler.start(config);

    ASSERT_TRUE(endpoint0_ptr->wait_submitted(1));
    ASSERT_TRUE(endpoint1_ptr->wait_submitted(1));
    WorkerDispatch dispatch0 = endpoint0_ptr->submitted().front();
    WorkerDispatch dispatch1 = endpoint1_ptr->submitted().front();
    endpoint0_ptr->emit(WorkerProgressKind::COMPLETED, dispatch0);
    endpoint1_ptr->emit(WorkerProgressKind::COMPLETED, dispatch1);

    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while ((allocator.slot_state(slot0)->state.load(std::memory_order_acquire) != TaskState::COMPLETED ||
            allocator.slot_state(slot1)->state.load(std::memory_order_acquire) != TaskState::COMPLETED) &&
           std::chrono::steady_clock::now() < deadline) {}
    EXPECT_EQ(allocator.slot_state(slot0)->state.load(std::memory_order_acquire), TaskState::COMPLETED);
    EXPECT_EQ(allocator.slot_state(slot1)->state.load(std::memory_order_acquire), TaskState::COMPLETED);
    EXPECT_EQ(endpoint0_ptr->progress_owner(), endpoint1_ptr->progress_owner());

    scheduler.request_stop();
    scheduler.stop();
    EXPECT_EQ(endpoint0_ptr->progress_owner(), endpoint1_ptr->progress_owner());
    EXPECT_FALSE(endpoint0_ptr->progress_owner_changed());
    EXPECT_FALSE(endpoint1_ptr->progress_owner_changed());
    manager.stop();
    allocator.shutdown();
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
        std::move(endpoint)
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
    worker.progress();
    worker.progress();
    {
        std::unique_lock<std::mutex> lk(callback_mu);
        EXPECT_TRUE(callback_cv.wait_for(lk, std::chrono::seconds(3), [&] {
            return completed.size() >= 1;
        }));
    }
    EXPECT_TRUE(worker.idle());
    EXPECT_TRUE(worker.busy());
    EXPECT_TRUE(worker.activate_prepared(/*run_id=*/12));
    worker.progress();
    EXPECT_TRUE(endpoint_ptr->wait_activated(/*run_id=*/12));

    endpoint_ptr->emit(WorkerProgressKind::ACCEPTED, submitted[1]);
    endpoint_ptr->emit(WorkerProgressKind::COMPLETED, submitted[1]);
    worker.progress();
    worker.progress();
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

TEST(WorkerManagerTest, DispatchAndCompletionEmitHostSpans) {
    ScopedHostSpanCapture host_span_capture;

    Ring allocator;
    allocator.init(/*heap_bytes=*/0);
    TaskSlot slot = make_progress_slot(allocator, /*run_id=*/73, /*pipeline_slot=*/0, /*generation=*/1);
    ASSERT_NE(slot, INVALID_SLOT);

    WorkerThread worker;
    auto endpoint = std::make_unique<DeterministicProgressEndpoint>();
    DeterministicProgressEndpoint *endpoint_ptr = endpoint.get();
    std::vector<WorkerCompletion> completed;
    worker.start(
        &allocator,
        [&](WorkerCompletion completion) {
            completed.push_back(std::move(completion));
        },
        [](WorkerDispatch) {}, std::move(endpoint)
    );

    worker.dispatch(WorkerDispatch{slot, 0});
    ASSERT_TRUE(endpoint_ptr->wait_submitted(1));
    EXPECT_TRUE(captured_host_span(simpler::host_trace::host_span_name(simpler::host_trace::HostSpan::Dispatch)));

    const WorkerDispatch submitted = endpoint_ptr->submitted().front();
    endpoint_ptr->emit(WorkerProgressKind::COMPLETED, submitted);
    worker.progress();
    ASSERT_EQ(completed.size(), 1u);
    EXPECT_TRUE(captured_host_span(simpler::host_trace::host_span_name(simpler::host_trace::HostSpan::Complete)));

    worker.stop();
    allocator.shutdown();
}

TEST(WorkerManagerTest, DisabledHostSpansSkipDispatchAndCompletionTraceWork) {
    ScopedHostSpanCapture host_span_capture(/*enabled=*/false);

    Ring allocator;
    allocator.init(/*heap_bytes=*/0);
    TaskSlot slot = make_progress_slot(allocator, /*run_id=*/73, /*pipeline_slot=*/0, /*generation=*/1);
    ASSERT_NE(slot, INVALID_SLOT);

    WorkerThread worker;
    auto endpoint = std::make_unique<DeterministicProgressEndpoint>();
    DeterministicProgressEndpoint *endpoint_ptr = endpoint.get();
    std::vector<WorkerCompletion> completed;
    worker.start(
        &allocator,
        [&](WorkerCompletion completion) {
            completed.push_back(std::move(completion));
        },
        [](WorkerDispatch) {}, std::move(endpoint)
    );

    const size_t caps_calls_before_dispatch = endpoint_ptr->caps_call_count();
    worker.dispatch(WorkerDispatch{slot, 0});
    ASSERT_TRUE(endpoint_ptr->wait_submitted(1));
    EXPECT_EQ(endpoint_ptr->caps_call_count(), caps_calls_before_dispatch + 1)
        << "disabled dispatch should query caps only for admission";

    const WorkerDispatch submitted = endpoint_ptr->submitted().front();
    endpoint_ptr->emit(WorkerProgressKind::COMPLETED, submitted);
    const size_t caps_calls_before_completion = endpoint_ptr->caps_call_count();
    worker.progress();
    ASSERT_EQ(completed.size(), 1u);
    EXPECT_EQ(endpoint_ptr->caps_call_count(), caps_calls_before_completion)
        << "disabled completion should not query caps for trace attributes";
    EXPECT_TRUE(captured_host_spans_empty());

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
        std::move(endpoint)
    );

    worker.dispatch(WorkerDispatch{active_slot, 0});
    ASSERT_TRUE(endpoint_ptr->wait_submitted(1));

    EXPECT_NO_THROW(worker.dispatch_prepared(WorkerDispatch{capacity_rejected, 0}));
    EXPECT_FALSE(worker.can_stage());
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
    worker.progress();
    worker.progress();
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

TEST(WorkerManagerTest, StopLinearizesWithEndpointSubmission) {
    Ring allocator;
    allocator.init(/*heap_bytes=*/0);
    TaskSlot submitted_slot = make_progress_slot(allocator, /*run_id=*/17, /*pipeline_slot=*/0, /*generation=*/1);
    TaskSlot rejected_slot = make_progress_slot(allocator, /*run_id=*/18, /*pipeline_slot=*/0, /*generation=*/2);
    ASSERT_NE(submitted_slot, INVALID_SLOT);
    ASSERT_NE(rejected_slot, INVALID_SLOT);

    WorkerThread worker;
    auto endpoint = std::make_unique<DeterministicProgressEndpoint>(/*worker_id=*/0, /*max_inflight_tasks=*/1);
    DeterministicProgressEndpoint *endpoint_ptr = endpoint.get();
    endpoint_ptr->block_next_submit();
    std::mutex completion_mu;
    std::vector<WorkerCompletion> completed;
    worker.start(
        &allocator,
        [&](WorkerCompletion completion) {
            std::lock_guard<std::mutex> lk(completion_mu);
            completed.push_back(std::move(completion));
        },
        [](WorkerDispatch) {}, std::move(endpoint)
    );

    std::thread submitter([&] {
        worker.dispatch(WorkerDispatch{submitted_slot, 0});
    });
    ASSERT_TRUE(endpoint_ptr->wait_submit_entered());

    std::promise<void> stop_returned;
    std::future<void> stop_future = stop_returned.get_future();
    std::thread stopper([&] {
        worker.stop();
        stop_returned.set_value();
    });
    EXPECT_EQ(stop_future.wait_for(std::chrono::milliseconds(50)), std::future_status::timeout)
        << "stop must not linearize while endpoint publication is still in progress";

    endpoint_ptr->release_blocked_submit();
    submitter.join();
    ASSERT_EQ(stop_future.wait_for(std::chrono::seconds(3)), std::future_status::ready);
    stopper.join();
    ASSERT_TRUE(endpoint_ptr->wait_submitted(1));

    worker.dispatch(WorkerDispatch{rejected_slot, 0});
    EXPECT_EQ(endpoint_ptr->submitted().size(), 1u) << "nothing may publish after stop returns";

    worker.progress();
    worker.progress();
    {
        std::lock_guard<std::mutex> lk(completion_mu);
        EXPECT_EQ(completed.size(), 2u);
    }
    EXPECT_FALSE(worker.busy());
    allocator.shutdown();
}

TEST(WorkerManagerTest, StopBeforeProgressDoesNotActivatePreparedSuccessor) {
    Ring allocator;
    allocator.init(/*heap_bytes=*/0);
    TaskSlot active_slot = make_progress_slot(allocator, /*run_id=*/19, /*pipeline_slot=*/0, /*generation=*/1);
    TaskSlot staged_slot = make_progress_slot(allocator, /*run_id=*/20, /*pipeline_slot=*/1, /*generation=*/1);
    ASSERT_NE(active_slot, INVALID_SLOT);
    ASSERT_NE(staged_slot, INVALID_SLOT);

    WorkerThread worker;
    auto endpoint = std::make_unique<DeterministicProgressEndpoint>();
    DeterministicProgressEndpoint *endpoint_ptr = endpoint.get();
    std::vector<WorkerCompletion> completed;
    worker.start(
        &allocator,
        [&](WorkerCompletion completion) {
            completed.push_back(std::move(completion));
        },
        [](WorkerDispatch) {}, std::move(endpoint)
    );
    worker.dispatch(WorkerDispatch{active_slot, 0});
    worker.dispatch_prepared(WorkerDispatch{staged_slot, 0});
    ASSERT_TRUE(endpoint_ptr->wait_submitted(2));
    std::vector<WorkerDispatch> submitted = endpoint_ptr->submitted();
    endpoint_ptr->emit(WorkerProgressKind::COMPLETED, submitted[0]);
    worker.progress();
    ASSERT_TRUE(worker.activate_prepared(/*run_id=*/20));

    worker.stop();
    worker.progress();
    EXPECT_FALSE(endpoint_ptr->wait_activated(/*run_id=*/20, std::chrono::milliseconds(20)))
        << "stop must fence activation even after the staged lane moved to active";
    worker.progress();
    EXPECT_EQ(completed.size(), 2u);
    EXPECT_FALSE(worker.busy());
    allocator.shutdown();
}

TEST(WorkerManagerTest, SubmitExceptionQuiescesEndpointOwnedPublication) {
    Ring allocator;
    allocator.init(/*heap_bytes=*/0);
    TaskSlot slot = make_progress_slot(allocator, /*run_id=*/21, /*pipeline_slot=*/0, /*generation=*/1);
    ASSERT_NE(slot, INVALID_SLOT);

    WorkerThread worker;
    auto endpoint = std::make_unique<DeterministicProgressEndpoint>();
    DeterministicProgressEndpoint *endpoint_ptr = endpoint.get();
    endpoint_ptr->throw_after_next_submit_publication();
    std::vector<WorkerCompletion> completed;
    worker.start(
        &allocator,
        [&](WorkerCompletion completion) {
            completed.push_back(std::move(completion));
        },
        [](WorkerDispatch) {}, std::move(endpoint)
    );

    worker.dispatch(WorkerDispatch{slot, 0});
    EXPECT_TRUE(endpoint_ptr->wait_progress_error());
    EXPECT_TRUE(completed.empty()) << "endpoint-owned work must terminalize through endpoint progress";
    EXPECT_TRUE(worker.busy());

    worker.progress();
    ASSERT_EQ(completed.size(), 1u);
    EXPECT_EQ(completed[0].outcome, EndpointOutcome::ENDPOINT_FAILURE);
    EXPECT_FALSE(worker.busy());
    allocator.shutdown();
}

TEST(WorkerManagerTest, TwoFrameLeaseSlotsDoNotDefineFifoOrAcceptance) {
    ScopedHostSpanCapture host_span_capture;

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

    // Entries are allocated in submission order and the payload frames are a
    // separate pool, so neither index is the run's lease slot any more — which
    // is a stronger form of what this case has always asserted.
    MailboxRunEntry *active_entry = test_entry(mailbox, 0);
    MailboxRunEntry *staged_entry = test_entry(mailbox, 1);
    EXPECT_EQ(test_entry_state(active_entry), MailboxState::IDLE);
    EXPECT_EQ(test_entry_state(staged_entry), MailboxState::IDLE);
    EXPECT_EQ(test_frame_entry_index(mailbox, 0), 0u);
    EXPECT_EQ(test_frame_entry_index(mailbox, 1), 1u);
    EXPECT_EQ(mailbox_state_of(test_task_frame(mailbox, 0)), MailboxState::TASK_READY);
    EXPECT_EQ(mailbox_state_of(test_task_frame(mailbox, 1)), MailboxState::PREPARE_READY);
    EXPECT_EQ(test_entry_dispatch_id(active_entry), 41u);
    EXPECT_EQ(test_entry_dispatch_id(staged_entry), 42u);
    EXPECT_EQ(test_entry_task_slot(active_entry), static_cast<uint64_t>(active_slot));
    EXPECT_EQ(test_entry_task_slot(staged_entry), static_cast<uint64_t>(staged_slot));
    EXPECT_EQ(test_entry_group_index(active_entry), 0u);
    EXPECT_EQ(test_entry_group_index(staged_entry), 0u);
    EXPECT_EQ(test_entry_group_size(active_entry), 1u);
    EXPECT_EQ(test_entry_group_size(staged_entry), 1u);
    EXPECT_EQ(active_entry->slot_id, 1u);
    EXPECT_EQ(staged_entry->slot_id, 0u);

    EXPECT_TRUE(endpoint.activate_progress(/*run_id=*/22));
    EXPECT_EQ(test_entry_state(staged_entry), MailboxState::IDLE);
    WorkerEndpointProgress progress;
    EXPECT_FALSE(endpoint.poll_progress(progress));

    // The child takes the staged run's payload and says so, which both stages
    // the entry and hands its frame back while the run is still to come.
    test_child_consume_frame(mailbox, 1, MailboxPreparationDisposition::NATIVE_PREPARED);
    ASSERT_TRUE(endpoint.poll_progress(progress));
    EXPECT_EQ(progress.kind, WorkerProgressKind::FRAME_STAGED);
    EXPECT_EQ(progress.dispatch.dispatch_id, 42u);
    EXPECT_EQ(progress.preparation_disposition, MailboxPreparationDisposition::NATIVE_PREPARED);
    EXPECT_EQ(test_entry_state(staged_entry), MailboxState::ACTIVATE);

    set_test_entry_state(staged_entry, MailboxState::TASK_LAUNCHED);
    EXPECT_FALSE(endpoint.poll_progress(progress));
    set_test_entry_accepted(staged_entry);
    ASSERT_TRUE(endpoint.poll_progress(progress));
    EXPECT_EQ(progress.kind, WorkerProgressKind::ACCEPTED);
    EXPECT_EQ(progress.dispatch.dispatch_id, 42u);
    EXPECT_TRUE(captured_host_span(simpler::host_trace::host_span_name(simpler::host_trace::HostSpan::FrameSubmit)));
    EXPECT_TRUE(captured_host_span(simpler::host_trace::host_span_name(simpler::host_trace::HostSpan::Activate)));
    allocator.shutdown();
}

TEST(WorkerManagerTest, CapacityOneMailboxUsesTheProgressTaskFrame) {
    alignas(8) std::array<char, MAILBOX_SIZE> mailbox{};
    Ring allocator;
    allocator.init(/*heap_bytes=*/0);
    TaskSlot task_slot = make_progress_slot(allocator, /*run_id=*/21, /*pipeline_slot=*/1, /*generation=*/7);
    ASSERT_NE(task_slot, INVALID_SLOT);
    TaskSlotState *state = allocator.slot_state(task_slot);
    ASSERT_NE(state, nullptr);
    state->is_group_ = true;
    state->task_args_list.resize(3);

    LocalMailboxEndpoint endpoint(/*worker_id=*/0, mailbox.data(), /*child_pid=*/-1, /*task_frame_count=*/1);
    EXPECT_FALSE(endpoint.caps().supports_frame_staging);

    WorkerDispatch dispatch{task_slot, 1, /*dispatch_id=*/41, /*prepare_only=*/false};
    endpoint.submit_progress(&allocator, dispatch);

    EXPECT_EQ(mailbox_state_of(test_task_frame(mailbox, 0)), MailboxState::TASK_READY);
    MailboxRunEntry *frame = test_entry(mailbox, 0);
    EXPECT_EQ(test_entry_dispatch_id(frame), 41u);
    EXPECT_EQ(test_entry_task_slot(frame), static_cast<uint64_t>(task_slot));
    EXPECT_EQ(test_entry_group_index(frame), 1u);
    EXPECT_EQ(test_entry_group_size(frame), 3u);

    set_test_entry_accepted(frame);
    WorkerEndpointProgress progress;
    ASSERT_TRUE(endpoint.poll_progress(progress));
    EXPECT_EQ(progress.kind, WorkerProgressKind::ACCEPTED);

    set_test_entry_state(frame, MailboxState::TASK_DONE);
    ASSERT_TRUE(endpoint.poll_progress(progress));
    EXPECT_EQ(progress.kind, WorkerProgressKind::COMPLETED);
    EXPECT_EQ(progress.completion.outcome, EndpointOutcome::SUCCESS);
    allocator.shutdown();
}

TEST(WorkerManagerTest, StaleActivationObservationCannotOverwriteTerminalState) {
    // The exchange is over the state word itself, wherever it lives; an entry's
    // state is at the same offset as a frame's, so this drives the word
    // directly rather than through either owner.
    alignas(8) std::array<char, MAILBOX_RUN_ENTRY_STRIDE> word{};
    auto *state = reinterpret_cast<int32_t *>(word.data() + MAILBOX_OFF_STATE);
    __atomic_store_n(state, static_cast<int32_t>(MailboxState::FRAME_STAGED), __ATOMIC_RELEASE);
    MailboxState stale_observation = static_cast<MailboxState>(__atomic_load_n(state, __ATOMIC_ACQUIRE));
    ASSERT_EQ(stale_observation, MailboxState::FRAME_STAGED);

    __atomic_store_n(state, static_cast<int32_t>(MailboxState::TASK_DONE), __ATOMIC_RELEASE);
    EXPECT_FALSE(mailbox_compare_exchange_state(word.data(), stale_observation, MailboxState::ACTIVATE));
    EXPECT_EQ(static_cast<MailboxState>(__atomic_load_n(state, __ATOMIC_ACQUIRE)), MailboxState::TASK_DONE);
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
        [](WorkerDispatch) {},
        std::make_unique<LocalMailboxEndpoint>(
            /*worker_id=*/0, mailbox.data(), /*child_pid=*/-1, /*task_frame_count=*/2
        )
    );
    worker.dispatch(WorkerDispatch{active_slot, 0});
    worker.dispatch_prepared(WorkerDispatch{staged_slot, 0});

    // Readiness is the request frame's word and identity is the entry's: the parent writes `IDLE`
    // to a run's entry and the ready state to the frame whose header names it, so a wait on the
    // entry's state would never see either one.
    char *request0 = test_task_frame(mailbox, 0);
    char *request1 = test_task_frame(mailbox, 1);
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while ((mailbox_state_of(request0) != MailboxState::TASK_READY ||
            mailbox_state_of(request1) != MailboxState::PREPARE_READY) &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_EQ(mailbox_state_of(request0), MailboxState::TASK_READY);
    EXPECT_EQ(mailbox_state_of(request1), MailboxState::PREPARE_READY);
    MailboxRunEntry *frame0 = test_entry(mailbox, test_frame_entry_index(mailbox, 0));
    MailboxRunEntry *frame1 = test_entry(mailbox, test_frame_entry_index(mailbox, 1));
    const uint64_t first_dispatch_id = test_entry_dispatch_id(frame0);
    const uint64_t second_dispatch_id = test_entry_dispatch_id(frame1);

    EXPECT_NO_THROW(worker.dispatch(WorkerDispatch{third_slot, 0}));
    EXPECT_NO_THROW(worker.dispatch_prepared(WorkerDispatch{fourth_slot, 0}));
    // Neither frame was refilled and neither entry was re-pointed: a third dispatch that found no
    // free transport is refused before it claims anything, so the two occupied hand-offs stand.
    EXPECT_EQ(mailbox_state_of(request0), MailboxState::TASK_READY);
    EXPECT_EQ(mailbox_state_of(request1), MailboxState::PREPARE_READY);
    EXPECT_EQ(test_frame_entry_index(mailbox, 0), static_cast<size_t>(0));
    EXPECT_EQ(test_frame_entry_index(mailbox, 1), static_cast<size_t>(1));
    EXPECT_EQ(test_entry_dispatch_id(frame0), first_dispatch_id);
    EXPECT_EQ(test_entry_dispatch_id(frame1), second_dispatch_id);
    {
        std::lock_guard<std::mutex> lk(completion_mu);
        EXPECT_EQ(completion_count, 2);
    }

    set_test_entry_accepted(frame0);
    set_test_entry_accepted(frame1);
    set_test_entry_state(frame0, MailboxState::TASK_DONE);
    set_test_entry_state(frame1, MailboxState::TASK_DONE);
    worker.progress();
    worker.progress();
    worker.progress();
    worker.progress();
    {
        std::unique_lock<std::mutex> lk(completion_mu);
        EXPECT_TRUE(completion_cv.wait_for(lk, std::chrono::seconds(3), [&] {
            return completion_count == 4;
        }));
    }
    worker.stop();
    allocator.shutdown();
}

// Transport is bounded apart from the grant, so a grant wider than the frame pool
// makes saturation an ordinary state. What it must not be is lossy: the request
// held back by a full pool is admitted once a frame comes home, and neither the
// payload that was in flight nor the one that follows it is overwritten.
//
// Driven synchronously against a real endpoint — the child side is this thread,
// so every step is ordered by the program rather than by a device or a sleep.
TEST(WorkerManagerTest, TransportSaturationResumesOnTheAcknowledgedFrame) {
    alignas(8) std::array<char, MAILBOX_SIZE> mailbox{};
    Ring allocator;
    allocator.init(/*heap_bytes=*/0);
    std::array<TaskSlot, 3> slots{};
    for (size_t index = 0; index < slots.size(); ++index) {
        slots[index] = make_progress_slot(allocator, 71 + index, index, 9);
        ASSERT_NE(slots[index], INVALID_SLOT);
    }

    // Granted three sets against two request frames: the widths differ on purpose, which is the
    // state row 28 is about.
    LocalMailboxEndpoint endpoint(/*worker_id=*/0, mailbox.data(), /*child_pid=*/-1, /*task_frame_count=*/3);
    ASSERT_EQ(endpoint.available_request_transport(), MAILBOX_REQUEST_FRAMES);

    endpoint.submit_progress(&allocator, WorkerDispatch{slots[0], 0, /*dispatch_id=*/81, /*prepare_only=*/false});
    endpoint.submit_progress(&allocator, WorkerDispatch{slots[1], 0, /*dispatch_id=*/82, /*prepare_only=*/true});
    // Both frames are out and the pool reports empty, so an admitter asking now is told to wait.
    EXPECT_EQ(endpoint.available_request_transport(), 0u);

    const uint64_t first_entry = test_frame_entry_index(mailbox, 0);
    const uint64_t second_entry = test_frame_entry_index(mailbox, 1);
    EXPECT_NE(first_entry, second_entry) << "two hand-offs landed on one entry";
    const uint64_t first_dispatch = test_entry_dispatch_id(test_entry(mailbox, first_entry));
    const uint64_t second_dispatch = test_entry_dispatch_id(test_entry(mailbox, second_entry));

    // The child copies the first payload out and hands that frame back: IDLE, then the ACK.
    ASSERT_TRUE(test_child_consume_and_return_frame(
        mailbox.data(), /*frame_index=*/0, mailbox_run_entry_count(/*granted_sets=*/3)
    ));
    // The level the parent reads is the echo, so the frame is available again from this instant.
    EXPECT_EQ(endpoint.available_request_transport(), 1u)
        << "an acknowledged frame did not come back, so a saturated endpoint would never resume";

    // The request that saturation held back now goes out, and lands on the reclaimed frame.
    endpoint.submit_progress(&allocator, WorkerDispatch{slots[2], 0, /*dispatch_id=*/83, /*prepare_only=*/false});
    EXPECT_EQ(mailbox_state_of(test_task_frame(mailbox, 0)), MailboxState::TASK_READY);
    EXPECT_EQ(endpoint.available_request_transport(), 0u);

    // Nothing was lost and nothing was overwritten. The second hand-off is untouched — its frame
    // still carries PREPARE_READY for the same entry and dispatch — and the third has an entry of
    // its own rather than displacing either.
    EXPECT_EQ(mailbox_state_of(test_task_frame(mailbox, 1)), MailboxState::PREPARE_READY);
    EXPECT_EQ(test_frame_entry_index(mailbox, 1), second_entry);
    EXPECT_EQ(test_entry_dispatch_id(test_entry(mailbox, second_entry)), second_dispatch);
    const uint64_t third_entry = test_frame_entry_index(mailbox, 0);
    EXPECT_NE(third_entry, first_entry) << "the third run reused the entry the first still holds";
    EXPECT_NE(third_entry, second_entry);
    EXPECT_EQ(test_entry_dispatch_id(test_entry(mailbox, third_entry)), 83u);
    // The first run's own entry still names it: its frame was reclaimed, its run was not.
    EXPECT_EQ(test_entry_dispatch_id(test_entry(mailbox, first_entry)), first_dispatch);

    // Every one of the three reaches a terminal, so none was dropped on the way.
    for (const uint64_t entry_index : {first_entry, second_entry, third_entry}) {
        MailboxRunEntry *entry = test_entry(mailbox, entry_index);
        set_test_entry_accepted(entry);
        set_test_entry_state(entry, MailboxState::TASK_DONE);
    }
    std::set<TaskSlot> completed;
    for (int attempt = 0; attempt < 16 && completed.size() < slots.size(); ++attempt) {
        WorkerEndpointProgress progress;
        if (!endpoint.poll_progress(progress)) break;
        if (progress.kind == WorkerProgressKind::COMPLETED) {
            EXPECT_EQ(progress.completion.outcome, EndpointOutcome::SUCCESS);
            completed.insert(progress.completion.task_slot);
        }
    }
    EXPECT_EQ(completed, std::set<TaskSlot>(slots.begin(), slots.end()))
        << "a run admitted across the saturation did not reach its own completion";
    allocator.shutdown();
}

// Contract row 32: error-text pressure. Every one of the `M` runs an endpoint can hold at once
// fails, each with its own message filled to the last byte the entry's inline text has room for,
// and each must reach the parent whole and attributed to its own run.
//
// `M` is `granted + MAILBOX_REQUEST_FRAMES` and the texts are per-entry, so this is the widest
// simultaneous text pressure the layout can produce — one text per entry, all live together,
// none of them short enough to fit beside another.
TEST(WorkerManagerTest, EveryConcurrentFailureKeepsItsOwnFullText) {
    alignas(8) std::array<char, MAILBOX_SIZE> mailbox{};
    Ring allocator;
    allocator.init(/*heap_bytes=*/0);
    constexpr uint32_t kGrantedSets = 3;
    const size_t entry_count = mailbox_run_entry_count(kGrantedSets);
    std::vector<TaskSlot> slots;
    for (size_t index = 0; index < entry_count; ++index) {
        slots.push_back(make_progress_slot(allocator, 301 + index, index % kGrantedSets, 5));
        ASSERT_NE(slots.back(), INVALID_SLOT);
    }

    LocalMailboxEndpoint endpoint(/*worker_id=*/0, mailbox.data(), /*child_pid=*/-1, kGrantedSets);
    const std::vector<size_t> entry_of =
        test_fill_every_entry(endpoint, allocator, mailbox, slots, kGrantedSets, /*first_dispatch_id=*/901);
    ASSERT_EQ(entry_of.size(), entry_count) << "the endpoint could not hold one run per run-state entry";
    EXPECT_EQ(std::set<size_t>(entry_of.begin(), entry_of.end()).size(), entry_count)
        << "two runs were given the same run-state entry";

    // Each run fails with a text of its own, published the way a child publishes one: rc and text
    // before the terminal state.
    std::map<TaskSlot, std::string> expected_text;
    for (size_t index = 0; index < entry_of.size(); ++index) {
        const std::string text =
            test_saturating_error_text(static_cast<char>('a' + index), "run" + std::to_string(index) + ":");
        expected_text[slots[index]] = text;
        MailboxRunEntry *entry = test_entry(mailbox, entry_of[index]);
        set_test_entry_error(entry, static_cast<int32_t>(index + 1), text.c_str());
        set_test_entry_state(entry, MailboxState::TASK_FAILED);
    }

    std::map<TaskSlot, std::string> reported;
    for (size_t attempt = 0; attempt < 4 * entry_count && reported.size() < entry_count; ++attempt) {
        WorkerEndpointProgress progress;
        if (!endpoint.poll_progress(progress)) break;
        if (progress.kind != WorkerProgressKind::COMPLETED) continue;
        EXPECT_EQ(progress.completion.outcome, EndpointOutcome::TASK_FAILURE);
        EXPECT_EQ(reported.count(progress.completion.task_slot), 0u) << "one run reported twice";
        reported[progress.completion.task_slot] = progress.completion.error_message;
    }
    ASSERT_EQ(reported.size(), entry_count) << "not every concurrent failure was reported";
    for (size_t index = 0; index < slots.size(); ++index) {
        const std::string &message = reported[slots[index]];
        EXPECT_NE(message.find(expected_text[slots[index]]), std::string::npos)
            << "run " << index << " did not keep its own full text; got: " << message;
        EXPECT_NE(message.find("code=" + std::to_string(index + 1)), std::string::npos)
            << "run " << index << " was reported under another run's code: " << message;
    }

    // Late identity: the entries are free again, so a new run takes one of them. Its completion
    // must carry its own text rather than whatever the previous occupant left in that entry.
    const TaskSlot reused_slot = make_progress_slot(allocator, 399, 0, 5);
    ASSERT_NE(reused_slot, INVALID_SLOT);
    ASSERT_GT(endpoint.available_request_transport(), 0u);
    endpoint.submit_progress(&allocator, WorkerDispatch{reused_slot, 0, /*dispatch_id=*/999, /*prepare_only=*/false});
    size_t carrier = MAILBOX_REQUEST_FRAMES;
    for (size_t frame = 0; frame < MAILBOX_REQUEST_FRAMES; ++frame) {
        if (mailbox_state_of(test_task_frame(mailbox, frame)) == MailboxState::TASK_READY) carrier = frame;
    }
    ASSERT_LT(carrier, MAILBOX_REQUEST_FRAMES);
    const size_t reused_entry = test_frame_entry_index(mailbox, carrier);
    ASSERT_TRUE(test_child_consume_and_return_frame(mailbox.data(), carrier, entry_count));
    MailboxRunEntry *entry = test_entry(mailbox, reused_entry);
    set_test_entry_error(entry, /*code=*/77, "the reused entry's own run");
    set_test_entry_state(entry, MailboxState::TASK_FAILED);

    WorkerEndpointProgress reused_progress;
    ASSERT_TRUE(endpoint.poll_progress(reused_progress));
    EXPECT_EQ(reused_progress.kind, WorkerProgressKind::COMPLETED);
    EXPECT_EQ(reused_progress.completion.task_slot, reused_slot);
    EXPECT_NE(reused_progress.completion.error_message.find("the reused entry's own run"), std::string::npos);
    EXPECT_EQ(reused_progress.completion.error_message.find("run0:"), std::string::npos)
        << "a recycled entry reported a previous occupant's text";
    allocator.shutdown();
}

// Contract row 29, the successor half: a failure in one of `M` concurrently live runs is that
// run's outcome and nobody else's. Every other run was published before the failure was
// observable, so each of them is a successor that is already out at the endpoint when the
// failure lands — the state where a shared error field, or a completion attributed by position
// rather than identity, would collapse five outcomes into one.
//
// The lane-side half of this row — that the failure poisons admission while the runs already out
// keep their own errors — is `test_chip_run_lane.cpp`'s
// `EarlierFailedRunRetainsItsErrorAfterLaterRunPoisonsLane`; this is the endpoint's half.
TEST(WorkerManagerTest, AFailureAmongLiveRunsIsNotItsSuccessorsOutcome) {
    alignas(8) std::array<char, MAILBOX_SIZE> mailbox{};
    Ring allocator;
    allocator.init(/*heap_bytes=*/0);
    constexpr uint32_t kGrantedSets = 3;
    const size_t entry_count = mailbox_run_entry_count(kGrantedSets);
    std::vector<TaskSlot> slots;
    for (size_t index = 0; index < entry_count; ++index) {
        slots.push_back(make_progress_slot(allocator, 401 + index, index % kGrantedSets, 11));
        ASSERT_NE(slots.back(), INVALID_SLOT);
    }

    LocalMailboxEndpoint endpoint(/*worker_id=*/0, mailbox.data(), /*child_pid=*/-1, kGrantedSets);
    const std::vector<size_t> entry_of =
        test_fill_every_entry(endpoint, allocator, mailbox, slots, kGrantedSets, /*first_dispatch_id=*/701);
    ASSERT_EQ(entry_of.size(), entry_count) << "the endpoint could not hold one run per run-state entry";

    // The first run fails; every run behind it was already published and succeeds.
    constexpr size_t kFailingIndex = 0;
    for (size_t index = 0; index < entry_of.size(); ++index) {
        MailboxRunEntry *entry = test_entry(mailbox, entry_of[index]);
        set_test_entry_accepted(entry);
        if (index == kFailingIndex) {
            set_test_entry_error(entry, /*code=*/13, "the one run that failed");
            set_test_entry_state(entry, MailboxState::TASK_FAILED);
        } else {
            set_test_entry_state(entry, MailboxState::TASK_DONE);
        }
    }

    std::map<TaskSlot, WorkerCompletion> completions;
    for (size_t attempt = 0; attempt < 6 * entry_count && completions.size() < entry_count; ++attempt) {
        WorkerEndpointProgress progress;
        if (!endpoint.poll_progress(progress)) break;
        if (progress.kind != WorkerProgressKind::COMPLETED) continue;
        completions.emplace(progress.completion.task_slot, progress.completion);
    }
    ASSERT_EQ(completions.size(), entry_count) << "a live run did not reach its own completion past the failure";
    for (size_t index = 0; index < slots.size(); ++index) {
        const WorkerCompletion &completion = completions.at(slots[index]);
        if (index == kFailingIndex) {
            EXPECT_EQ(completion.outcome, EndpointOutcome::TASK_FAILURE);
            EXPECT_NE(completion.error_message.find("the one run that failed"), std::string::npos);
            continue;
        }
        EXPECT_EQ(completion.outcome, EndpointOutcome::SUCCESS)
            << "run " << index << " took its predecessor's failure as its own outcome";
        EXPECT_TRUE(completion.error_message.empty())
            << "run " << index << " carried a message it never published: " << completion.error_message;
    }
    allocator.shutdown();
}

class MailboxChildExitTest : public ::testing::TestWithParam<std::pair<int, bool>> {};

TEST_P(MailboxChildExitTest, PreservesOnlyMatchingTerminalResults) {
    const auto [exit_code, stale_identity] = GetParam();
    Ring allocator;
    allocator.init(/*heap_bytes=*/0);
    std::array<TaskSlot, 3> slots{};
    for (size_t index = 0; index < slots.size(); ++index) {
        slots[index] = make_progress_slot(allocator, 41 + index, index, 3);
        ASSERT_NE(slots[index], INVALID_SLOT);
    }

    void *mailbox = mmap(nullptr, MAILBOX_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    ASSERT_NE(mailbox, MAP_FAILED);
    std::memset(mailbox, 0, MAILBOX_SIZE);
    int start_pipe[2];
    ASSERT_EQ(pipe(start_pipe), 0);
    const pid_t child = fork();
    ASSERT_GE(child, 0);
    if (child == 0) {
        close(start_pipe[1]);
        char start = 0;
        ssize_t received;
        do {
            received = read(start_pipe[0], &start, sizeof(start));
        } while (received < 0 && errno == EINTR);
        if (received != sizeof(start)) _exit(99);
        MailboxRunEntry *success = mailbox_run_entry(mailbox, 0);
        MailboxRunEntry *failure = mailbox_run_entry(mailbox, 1);
        if (stale_identity) {
            // The entry's own generation: what an identity check compares first, and what tells
            // this run apart from any later occupant of the entry.
            __atomic_store_n(&success->entry_generation, static_cast<uint64_t>(99), __ATOMIC_RELEASE);
        }
        set_test_entry_accepted(success);
        set_test_entry_state(success, MailboxState::TASK_DONE);
        set_test_entry_error(failure, /*code=*/17, "successor's own failure");
        set_test_entry_accepted(failure);
        set_test_entry_state(failure, MailboxState::TASK_FAILED);
        _exit(exit_code);
    }

    ScopedChildProcess child_guard(child);
    close(start_pipe[0]);
    {
        LocalMailboxEndpoint endpoint(0, mailbox, child, /*task_frame_count=*/3);
        for (size_t index = 0; index < slots.size(); ++index) {
            // Three runs, two frames: the grant and the transport are separate widths, so a third
            // publication needs a frame back first. This is the hand-back a child makes once it
            // has copied the payload out — the frame returns while the run it carried stays this
            // endpoint's, which is the whole reason the two are separate.
            if (index == slots.size() - 1) {
                ASSERT_TRUE(test_child_consume_and_return_frame(
                    mailbox, /*frame_index=*/0, mailbox_run_entry_count(/*granted_sets=*/3)
                )) << "the first hand-off's frame header did not name a live entry";
            }
            endpoint.submit_progress(&allocator, WorkerDispatch{slots[index], 0, 51 + index, false});
        }
        const char start = 1;
        ASSERT_EQ(write(start_pipe[1], &start, sizeof(start)), sizeof(start));
        close(start_pipe[1]);

        // WNOWAIT leaves the exit status for the endpoint's own waitpid.
        siginfo_t status{};
        int waited;
        do {
            waited = waitid(P_PID, child, &status, WEXITED | WNOWAIT);
        } while (waited < 0 && errno == EINTR);
        ASSERT_EQ(waited, 0);
        ASSERT_EQ(status.si_status, exit_code);
        // No progress poll occurs before the 10 ms child-liveness deadline.
        std::this_thread::sleep_for(std::chrono::milliseconds(20));

        for (size_t index = 0; index < slots.size(); ++index) {
            WorkerEndpointProgress progress;
            ASSERT_TRUE(endpoint.poll_progress(progress));
            ASSERT_EQ(progress.kind, WorkerProgressKind::COMPLETED);
            EXPECT_EQ(progress.completion.task_slot, slots[index]);
            const auto expected = index == 0 && !stale_identity ? EndpointOutcome::SUCCESS :
                                  index == 1                    ? EndpointOutcome::TASK_FAILURE :
                                                                  EndpointOutcome::ENDPOINT_FAILURE;
            EXPECT_EQ(progress.completion.outcome, expected);
            if (index == 1) {
                EXPECT_NE(progress.completion.error_message.find("code=17"), std::string::npos);
                EXPECT_NE(progress.completion.error_message.find("successor's own failure"), std::string::npos);
            }
        }
        WorkerEndpointProgress extra;
        EXPECT_FALSE(endpoint.poll_progress(extra));
        EXPECT_FALSE(endpoint.activate_progress(41));
        EXPECT_THROW(endpoint.submit_progress(&allocator, WorkerDispatch{slots[0], 0, 54, false}), std::runtime_error);
    }
    EXPECT_EQ(munmap(mailbox, MAILBOX_SIZE), 0);
    allocator.shutdown();
}

INSTANTIATE_TEST_SUITE_P(
    ExitStatusAndIdentity, MailboxChildExitTest,
    ::testing::Values(
        std::make_pair(0, false), std::make_pair(7, false), std::make_pair(0, true), std::make_pair(7, true)
    )
);

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

    MailboxRunEntry *frame1 = test_entry(mailbox, 1);
    // The entry's own generation, through its member: it is the first thing an identity check
    // compares, and it is what distinguishes this run from any later occupant of the entry.
    // Written as a member store rather than a byte copy at a frame offset — a frame offset added
    // to a `MailboxRunEntry *` strides whole entries, not bytes.
    __atomic_store_n(&frame1->entry_generation, static_cast<uint64_t>(99), __ATOMIC_RELEASE);
    set_test_entry_accepted(frame1);
    set_test_entry_state(frame1, MailboxState::TASK_LAUNCHED);

    WorkerEndpointProgress first;
    WorkerEndpointProgress second;
    EXPECT_FALSE(endpoint.poll_progress(first));
    EXPECT_EQ(test_entry_state(frame1), MailboxState::TASK_LAUNCHED);

    // Endpoint poison requests child shutdown, but parent-side completion must
    // not release the run lease until every native task has terminalized. A
    // real child publishes these states from its shutdown/finalize path.
    set_test_entry_state(test_entry(mailbox, 0), MailboxState::TASK_FAILED);
    set_test_entry_state(frame1, MailboxState::TASK_FAILED);

    ASSERT_TRUE(endpoint.poll_progress(first));
    ASSERT_TRUE(endpoint.poll_progress(second));
    EXPECT_EQ(first.kind, WorkerProgressKind::COMPLETED);
    EXPECT_EQ(second.kind, WorkerProgressKind::COMPLETED);
    EXPECT_EQ(first.completion.outcome, EndpointOutcome::ENDPOINT_FAILURE);
    EXPECT_EQ(second.completion.outcome, EndpointOutcome::ENDPOINT_FAILURE);
    EXPECT_NE(first.completion.error_message.find("stale run-state entry identity"), std::string::npos);
    EXPECT_NE(second.completion.error_message.find("stale run-state entry identity"), std::string::npos);
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

        MailboxRunEntry *frame0 = mailbox_run_entry(mailbox, 0);
        MailboxRunEntry *frame1 = mailbox_run_entry(mailbox, 1);
        set_test_entry_state(frame0, MailboxState::TASK_FAILED);
        set_test_entry_state(frame1, MailboxState::TASK_FAILED);

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

        MailboxRunEntry *frame0 = mailbox_run_entry(mailbox, 0);
        MailboxRunEntry *frame1 = mailbox_run_entry(mailbox, 1);
        __atomic_store_n(&frame1->entry_generation, static_cast<uint64_t>(100), __ATOMIC_RELEASE);
        set_test_entry_accepted(frame1);
        set_test_entry_state(frame1, MailboxState::TASK_LAUNCHED);

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
        EXPECT_EQ(test_entry_state(frame0), MailboxState::TASK_FAILED);
        EXPECT_EQ(test_entry_state(frame1), MailboxState::TASK_FAILED);
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
        [](WorkerDispatch) {}, std::move(endpoint)
    );
    worker.dispatch(WorkerDispatch{active_slot, 0});
    worker.dispatch_prepared(WorkerDispatch{staged_slot, 0});
    EXPECT_TRUE(endpoint_ptr->wait_submitted(2));

    worker.stop();
    worker.progress();
    EXPECT_TRUE(endpoint_ptr->wait_stop_requested());
    worker.progress();
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
        [](WorkerDispatch) {}, std::move(endpoint)
    );
    worker.dispatch(WorkerDispatch{slot, 0});
    EXPECT_TRUE(endpoint_ptr->wait_submitted(1));

    worker.stop();
    worker.progress();
    worker.progress();

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
        [](WorkerDispatch) {}, std::move(endpoint)
    );
    worker.dispatch(WorkerDispatch{slot, 0});
    EXPECT_TRUE(endpoint_ptr->wait_submitted(1));
    worker.progress();
    worker.progress();
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

    worker.stop();

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
        [](WorkerDispatch) {}
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
    std::vector<DeterministicProgressEndpoint *> endpoints;
    std::atomic<uint64_t> preparable_run_query_count{0};

    virtual uint32_t endpoint0_capacity() const { return 2; }
    virtual std::vector<uint32_t> endpoint_capacities() const { return {endpoint0_capacity(), 2}; }
    virtual uint32_t pipeline_depth() const { return 2; }

    void SetUp() override {
        allocator.init(/*heap_bytes=*/1ULL << 20);
        const std::vector<uint32_t> capacities = endpoint_capacities();
        for (size_t worker_id = 0; worker_id < capacities.size(); ++worker_id) {
            auto endpoint =
                std::make_unique<DeterministicProgressEndpoint>(static_cast<int32_t>(worker_id), capacities[worker_id]);
            endpoints.push_back(endpoint.get());
            manager.add_next_level_endpoint(std::move(endpoint));
        }
        ASSERT_GE(endpoints.size(), 2u);
        endpoint0 = endpoints[0];
        endpoint1 = endpoints[1];
        manager.start(
            &allocator,
            [this](WorkerCompletion completion) {
                scheduler.worker_done(std::move(completion));
            },
            [this](WorkerDispatch dispatch) {
                orchestrator.mark_task_accepted(dispatch.task_slot);
            }
        );
        ready_next.reset(manager.next_level_worker_ids());
        orchestrator.init(&tensor_map, &allocator, &scope, &ready_sub, &ready_next, &manager, [this] {
            scheduler.notify_ready();
        });
        orchestrator.configure_pipeline_depth(pipeline_depth());

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
            preparable_run_query_count.fetch_add(1, std::memory_order_relaxed);
            return orchestrator.preparable_run_id();
        };
        if (pipeline_depth() > 2) {
            scheduler_config.preparable_runs_cb = [this] {
                preparable_run_query_count.fetch_add(1, std::memory_order_relaxed);
                return orchestrator.preparable_run_ids();
            };
        }
        scheduler_config.can_stage_run_on_worker_cb = [this](RunId run_id, int32_t worker_id) {
            return orchestrator.can_stage_run_on_worker(run_id, worker_id);
        };
        scheduler_config.on_consumed_cb = [this](TaskSlot task_slot) {
            orchestrator.on_consumed(task_slot);
        };
        scheduler_config.on_task_failed_cb = [this](TaskSlot task_slot, const std::string &message) {
            orchestrator.report_task_error(task_slot, message);
        };
        scheduler.start(scheduler_config);
    }

    bool wait_for_preparable_scan(uint64_t before) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
        while (preparable_run_query_count.load() <= before) {
            if (std::chrono::steady_clock::now() >= deadline) return false;
            std::this_thread::yield();
        }
        std::scoped_lock lock(scheduler.loop_mutex());
        return true;
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

struct PartialStagingProgressSchedulerFixture : public ProgressSchedulerFixture {
    std::vector<uint32_t> endpoint_capacities() const override { return {2, 1, 2}; }
};

struct ThreeRunProgressSchedulerFixture : public ProgressSchedulerFixture {
    uint32_t endpoint0_capacity() const override { return 3; }
    uint32_t pipeline_depth() const override { return 3; }
};

TEST_F(ThreeRunProgressSchedulerFixture, ColdQueuePublishesThreeRunsInFifoOrder) {
    std::vector<RunId> runs;
    std::vector<SubmitResult> tasks;
    {
        std::scoped_lock lock(scheduler.loop_mutex());
        for (int index = 0; index < 3; ++index) {
            runs.push_back(orchestrator.begin_run());
            tasks.push_back(orchestrator.submit_next_level(
                C(91 + index), single_tensor_args(0x9100 + index * 0x100, TensorArgType::OUTPUT), config, 0
            ));
            orchestrator.close_run_submission(runs.back());
        }
    }

    ASSERT_TRUE(endpoint0->wait_submitted(3));
    const auto submitted = endpoint0->submitted();
    ASSERT_EQ(submitted.size(), 3u);
    for (size_t index = 0; index < submitted.size(); ++index) {
        EXPECT_EQ(submitted[index].task_slot, tasks[index].task_slot);
        EXPECT_EQ(submitted[index].prepare_only, index != 0);
        if (index != 0) EXPECT_LT(submitted[index - 1].dispatch_id, submitted[index].dispatch_id);
        endpoint0->emit(WorkerProgressKind::ACCEPTED, submitted[index]);
        endpoint0->emit(WorkerProgressKind::COMPLETED, submitted[index]);
        ASSERT_TRUE(orchestrator.wait_run_for(runs[index], 3.0));
        orchestrator.release_run(runs[index]);
    }
}

// The same three-run pipeline over a *real* mailbox endpoint and a real child loop, which is what
// gives it a bounded request-frame pool to saturate. `DeterministicProgressEndpoint` has no
// transport at all, so no fixture built on it can reach the state below.
struct MailboxPipelineSchedulerFixture : public ::testing::Test {
    TensorMap tensor_map;
    Ring allocator;
    Scope scope;
    ReadyQueue ready_sub;
    NextLevelReadyQueues ready_next;
    Orchestrator orchestrator;
    MockMailboxWorker child;
    WorkerManager manager;
    Scheduler scheduler;
    CallConfig config;

    static constexpr uint32_t kGrantedSets = 3;

    void SetUp() override {
        allocator.init(/*heap_bytes=*/1ULL << 20);
        // The child publishes its own terminal states here, including the acceptance word a
        // staged run reaches only after the parent activates it.
        child.auto_accept.store(true, std::memory_order_release);
        child.granted_sets = kGrantedSets;
        child.start();
        manager.add_next_level(child.mailbox_ptr(), /*child_pid=*/-1, kGrantedSets);
        manager.start(
            &allocator,
            [this](WorkerCompletion completion) {
                scheduler.worker_done(std::move(completion));
            },
            [this](WorkerDispatch dispatch) {
                orchestrator.mark_task_accepted(dispatch.task_slot);
            }
        );
        ready_next.reset(manager.next_level_worker_ids());
        orchestrator.init(&tensor_map, &allocator, &scope, &ready_sub, &ready_next, &manager, [this] {
            scheduler.notify_ready();
        });
        orchestrator.configure_pipeline_depth(kGrantedSets);

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
        scheduler_config.preparable_runs_cb = [this] {
            return orchestrator.preparable_run_ids();
        };
        scheduler_config.can_stage_run_on_worker_cb = [this](RunId run_id, int32_t worker_id) {
            return orchestrator.can_stage_run_on_worker(run_id, worker_id);
        };
        scheduler_config.on_consumed_cb = [this](TaskSlot task_slot) {
            orchestrator.on_consumed(task_slot);
        };
        scheduler_config.on_task_failed_cb = [this](TaskSlot task_slot, const std::string &message) {
            std::lock_guard<std::mutex> lk(failure_mu);
            failures.emplace_back(task_slot, message);
            orchestrator.report_task_error(task_slot, message);
        };
        scheduler.start(scheduler_config);
    }

    void TearDown() override {
        child.drain();
        scheduler.request_stop();
        manager.stop_workers();
        scheduler.stop();
        manager.stop();
        allocator.shutdown();
    }

    std::vector<std::pair<TaskSlot, std::string>> reported_failures() {
        std::lock_guard<std::mutex> lk(failure_mu);
        return failures;
    }

    WorkerThread *worker() { return manager.get_worker_by_id(WorkerType::NEXT_LEVEL, 0); }

    std::mutex failure_mu;
    std::vector<std::pair<TaskSlot, std::string>> failures;
};

// Contract row 28 / finding F6, at the layer that admits: a request the scheduler has already
// made ready must *wait* for a frame rather than being published into a full pool or dropped.
//
// Transport is `MAILBOX_REQUEST_FRAMES` frames and the grant here is three sets, so the two
// widths differ on purpose — saturation is an ordinary state rather than a fault. The child
// withholds both hand-backs, which is a thing the protocol lets a child do at any time: the
// acknowledgement is its last store to a frame, and nothing bounds when it makes it.
//
// What separates this from the endpoint-level regression below it: nothing here publishes
// anything by hand. The scheduler owns every admission decision, and the third run is admitted
// only because its own gate was asked again after a frame came home.
TEST_F(MailboxPipelineSchedulerFixture, AStagedRequestParkedByFullTransportResumesOnTheAcknowledgedFrame) {
    child.hold_next_request_frames(MAILBOX_REQUEST_FRAMES);

    std::vector<RunId> runs;
    std::vector<SubmitResult> tasks;
    {
        std::scoped_lock lock(scheduler.loop_mutex());
        for (int index = 0; index < 3; ++index) {
            runs.push_back(orchestrator.begin_run());
            tasks.push_back(orchestrator.submit_next_level(
                C(91 + index), single_tensor_args(0x9100 + index * 0x100, TensorArgType::OUTPUT), config, 0
            ));
            orchestrator.close_run_submission(runs.back());
        }
    }

    // Both frames are out and the child is sitting on them, so the pool is empty.
    ASSERT_TRUE(child.wait_held_frames(MAILBOX_REQUEST_FRAMES))
        << "the child never took both request frames, so transport was never full";

    // The parked state, read while the loop is still running: two payloads reached the child and
    // the third did not, and its task is still queued rather than claimed or failed. `loop_mutex`
    // is held so this is one instant of the scheduler's own state, not a walk across several of
    // its iterations.
    //
    // The third run is a staged successor, so `can_stage` is the gate that refused it — and with
    // `M = granted + MAILBOX_REQUEST_FRAMES` entries against two dispatches out, its inflight and
    // staged-lane terms both have headroom, leaving transport as the term that can be false.
    // `can_dispatch` is false too, but for the ordinary lane's own reason as well, so it is not
    // the isolating reading; what isolates transport is that the release below changes nothing
    // except the frame.
    std::vector<MockMailboxWorker::Record> parked;
    {
        std::scoped_lock lock(scheduler.loop_mutex());
        parked = child.dispatched_records();
        EXPECT_EQ(parked.size(), static_cast<size_t>(MAILBOX_REQUEST_FRAMES))
            << "more payloads reached the child than there are request frames to carry them";
        EXPECT_EQ(allocator.slot_state(tasks[2].task_slot)->state.load(std::memory_order_acquire), TaskState::READY)
            << "the third run's task left the ready queue while no frame could carry it";
        EXPECT_FALSE(worker()->can_stage()) << "the endpoint offered to stage a run while the child held both frames";
        EXPECT_FALSE(worker()->can_dispatch());
    }
    ASSERT_EQ(parked.size(), static_cast<size_t>(MAILBOX_REQUEST_FRAMES));
    EXPECT_EQ(parked[0].tensor_key, 0x9100u);
    EXPECT_FALSE(parked[0].prepare_only) << "the first run of the pipeline was staged rather than made active";
    EXPECT_EQ(parked[1].tensor_key, 0x9200u);
    EXPECT_TRUE(parked[1].prepare_only) << "the second run was not a staged successor";
    EXPECT_TRUE(reported_failures().empty()) << "a full request pool was reported as a task failure";

    // One frame is released. The release only queues the hand-back: the child's own loop performs
    // it, and then carries straight on into the run that frame carried — so from here until this
    // test says otherwise, that run is still executing and cannot be the reason anything else is
    // admitted.
    const size_t reclaimed_frame = parked[0].frame_index;
    const size_t executing_entry = parked[0].entry_index;
    child.release_one_held_frame();

    // The acknowledgement itself, observed where it is made. This matters for what follows: the
    // child's hand-back writes `IDLE` *before* the acknowledgement it counts here, so once this
    // returns the frame reads `IDLE` unless the parent has already published something new onto
    // it. Without this step the frame still carries the previous publication's `TASK_READY`, and
    // a poll for "the frame is ready" would accept the payload that was already there.
    ASSERT_TRUE(child.wait_acknowledged(1)) << "the released frame was never handed back";

    // So from here, any state other than `IDLE` is a *new* publication. The load is an acquire,
    // and the parent writes the entry and the frame's header before the state word it releases
    // them with, so everything read below belongs to the publication this observed.
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    bool republished = false;
    while (std::chrono::steady_clock::now() < deadline) {
        if (mailbox_state_of(test_task_frame(child.mailbox, reclaimed_frame)) != MailboxState::IDLE) {
            republished = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_TRUE(republished) << "the request a full pool held back was never published after the acknowledgement "
                                "that freed a frame";
    EXPECT_NE(test_entry_state(test_entry(child.mailbox, executing_entry)), MailboxState::TASK_DONE)
        << "the run that acknowledged the frame had already finished, so its completion could be what admitted "
           "the third rather than the acknowledgement";

    // And it is the third run: the entry the republished frame names carries that task's own
    // slot, not a reused header from either run still in flight.
    const size_t resumed_entry = test_frame_entry_index(child.mailbox, reclaimed_frame);
    EXPECT_NE(resumed_entry, executing_entry) << "the admitted request took the entry a live run still holds";
    EXPECT_NE(resumed_entry, parked[1].entry_index) << "the admitted request took the entry of the withheld frame";
    EXPECT_EQ(test_entry_task_slot(test_entry(child.mailbox, resumed_entry)), static_cast<uint64_t>(tasks[2].task_slot))
        << "the entry the reclaimed frame names is not the task that was held back";
    EXPECT_EQ(mailbox_state_of(test_task_frame(child.mailbox, reclaimed_frame)), MailboxState::PREPARE_READY)
        << "the third run of the pipeline was published as an active dispatch rather than a staged successor";
    EXPECT_TRUE(reported_failures().empty()) << "resuming a parked request reported a failure";
    // The frame the child is still withholding is untouched: nothing was republished over it.
    EXPECT_EQ(test_frame_entry_index(child.mailbox, parked[1].frame_index), parked[1].entry_index);

    // Now the child is let go. It copies the resumed payload out, and every run reaches its own
    // completion — which is what makes the park a delay rather than a loss.
    child.drain();
    ASSERT_TRUE(child.wait_dispatched(3)) << "the resumed request was published but never consumed";
    const std::vector<MockMailboxWorker::Record> resumed = child.dispatched_records();
    ASSERT_EQ(resumed.size(), 3u) << "a payload was delivered more than once across the saturation";
    EXPECT_EQ(resumed[2].tensor_key, 0x9300u) << "the admitted request was not the one that had been held back";
    EXPECT_EQ(resumed[2].callable_hash0, 93u);
    EXPECT_EQ(resumed[2].frame_index, reclaimed_frame) << "the resumed request did not land on the acknowledged frame";
    // Neither earlier payload was re-delivered or altered by the admission behind it.
    EXPECT_EQ(resumed[0].tensor_key, parked[0].tensor_key);
    EXPECT_EQ(resumed[1].tensor_key, parked[1].tensor_key);

    for (size_t index = 0; index < runs.size(); ++index) {
        ASSERT_TRUE(orchestrator.wait_run_for(runs[index], 10.0)) << "run " << index << " never retired";
        orchestrator.release_run(runs[index]);
    }
    EXPECT_TRUE(reported_failures().empty()) << "a run of this pipeline failed: " << reported_failures().size();
}

TEST_F(ProgressSchedulerFixture, StagedSuccessorCannotOvertakePendingPredecessorOnSameEndpoint) {
    const RunId first_run = orchestrator.begin_run();
    const auto first =
        orchestrator.submit_next_level(C(93), single_tensor_args(0x9300, TensorArgType::OUTPUT), config, 0);
    auto tail_args = single_tensor_args(0x9400, TensorArgType::OUTPUT);
    tail_args.add_dep_wait(first);
    const auto tail = orchestrator.submit_next_level(C(94), tail_args, config, 0);
    orchestrator.close_run_submission(first_run);
    ASSERT_TRUE(endpoint0->wait_submitted(1));

    uint64_t before;
    RunId second_run;
    SubmitResult successor;
    {
        std::scoped_lock lock(scheduler.loop_mutex());
        before = preparable_run_query_count.load();
        second_run = orchestrator.begin_run();
        successor = orchestrator.submit_next_level(C(95), single_tensor_args(0x9500, TensorArgType::OUTPUT), config, 0);
        orchestrator.close_run_submission(second_run);
    }
    ASSERT_TRUE(wait_for_preparable_scan(before));
    {
        std::scoped_lock lock(scheduler.loop_mutex());
        EXPECT_EQ(endpoint0->submitted().size(), 1u);
        EXPECT_EQ(allocator.slot_state(tail.task_slot)->state.load(), TaskState::PENDING);
    }

    const auto first_dispatch = endpoint0->submitted().front();
    endpoint0->emit(WorkerProgressKind::ACCEPTED, first_dispatch);
    endpoint0->emit(WorkerProgressKind::COMPLETED, first_dispatch);
    ASSERT_TRUE(endpoint0->wait_submitted(3));
    const auto submitted = endpoint0->submitted();
    ASSERT_EQ(submitted.size(), 3u);
    EXPECT_EQ(submitted[1].task_slot, tail.task_slot);
    EXPECT_FALSE(submitted[1].prepare_only);
    EXPECT_EQ(submitted[2].task_slot, successor.task_slot);
    EXPECT_TRUE(submitted[2].prepare_only);
    EXPECT_LT(submitted[1].dispatch_id, submitted[2].dispatch_id);
    for (size_t index = 1; index < submitted.size(); ++index) {
        endpoint0->emit(WorkerProgressKind::ACCEPTED, submitted[index]);
        endpoint0->emit(WorkerProgressKind::COMPLETED, submitted[index]);
    }
    ASSERT_TRUE(orchestrator.wait_run_for(first_run, 3.0));
    ASSERT_TRUE(orchestrator.wait_run_for(second_run, 3.0));
    orchestrator.release_run(first_run);
    orchestrator.release_run(second_run);
}

TEST_F(ProgressSchedulerFixture, StagedSuccessorCannotOvertakePendingPredecessorGroup) {
    const RunId first_run = orchestrator.begin_run();
    const auto first =
        orchestrator.submit_next_level(C(96), single_tensor_args(0x9600, TensorArgType::OUTPUT), config, 0);
    auto group0 = single_tensor_args(0x9700, TensorArgType::OUTPUT);
    group0.add_dep_wait(first);
    const auto group = orchestrator.submit_next_level_group(
        C(97), {group0, single_tensor_args(0x9800, TensorArgType::OUTPUT)}, config, {0, 1}
    );
    orchestrator.close_run_submission(first_run);
    ASSERT_TRUE(endpoint0->wait_submitted(1));

    uint64_t before;
    RunId second_run;
    SubmitResult successor;
    {
        std::scoped_lock lock(scheduler.loop_mutex());
        before = preparable_run_query_count.load();
        second_run = orchestrator.begin_run();
        successor = orchestrator.submit_next_level(C(99), single_tensor_args(0x9900, TensorArgType::OUTPUT), config, 1);
        orchestrator.close_run_submission(second_run);
    }
    ASSERT_TRUE(wait_for_preparable_scan(before));
    EXPECT_TRUE(endpoint1->submitted().empty());

    const auto first_dispatch = endpoint0->submitted().front();
    endpoint0->emit(WorkerProgressKind::ACCEPTED, first_dispatch);
    endpoint0->emit(WorkerProgressKind::COMPLETED, first_dispatch);
    ASSERT_TRUE(endpoint0->wait_submitted(2));
    ASSERT_TRUE(endpoint1->wait_submitted(2));
    const auto worker0 = endpoint0->submitted();
    const auto worker1 = endpoint1->submitted();
    EXPECT_EQ(worker0.back().task_slot, group.task_slot);
    EXPECT_EQ(worker1[0].task_slot, group.task_slot);
    EXPECT_FALSE(worker1[0].prepare_only);
    EXPECT_EQ(worker1[1].task_slot, successor.task_slot);
    EXPECT_TRUE(worker1[1].prepare_only);
    EXPECT_LT(worker1[0].dispatch_id, worker1[1].dispatch_id);
    endpoint0->emit(WorkerProgressKind::ACCEPTED, worker0.back());
    endpoint0->emit(WorkerProgressKind::COMPLETED, worker0.back());
    for (const auto &dispatch : worker1) {
        endpoint1->emit(WorkerProgressKind::ACCEPTED, dispatch);
        endpoint1->emit(WorkerProgressKind::COMPLETED, dispatch);
    }
    ASSERT_TRUE(orchestrator.wait_run_for(first_run, 3.0));
    ASSERT_TRUE(orchestrator.wait_run_for(second_run, 3.0));
    orchestrator.release_run(first_run);
    orchestrator.release_run(second_run);
}

TEST_F(ProgressSchedulerFixture, GroupSubmitReportsNoSingleWorkerAndNoSingleIndex) {
    ScopedHostSpanCapture host_span_capture;

    RunId run = orchestrator.begin_run();
    SubmitResult group = orchestrator.submit_next_level_group(
        C(9), {single_tensor_args(0x9000, TensorArgType::OUTPUT), single_tensor_args(0xA000, TensorArgType::OUTPUT)},
        config, {0, 1}
    );
    orchestrator.close_run_submission(run);

    ASSERT_TRUE(captured_host_span(simpler::host_trace::host_span_name(simpler::host_trace::HostSpan::Submit)));
    std::ostringstream expected;
    expected << "run_id=" << run << " task_slot=" << group.task_slot << " group_index=-1 group_size=2 role=facade";
    EXPECT_EQ(
        captured_host_span_attrs(simpler::host_trace::host_span_name(simpler::host_trace::HostSpan::Submit)),
        expected.str()
    );
}

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

TEST_F(ProgressSchedulerFixture, PreparedSuccessorStagesEveryEligibleWorker) {
    const RunId first_run = orchestrator.begin_run();
    const SubmitResult first =
        orchestrator.submit_next_level(C(21), single_tensor_args(0x2100, TensorArgType::OUTPUT), config, 0);
    orchestrator.close_run_submission(first_run);
    ASSERT_TRUE(endpoint0->wait_submitted(1));

    const RunId second_run = orchestrator.begin_run();
    const SubmitResult worker0 =
        orchestrator.submit_next_level(C(22), single_tensor_args(0x2200, TensorArgType::OUTPUT), config, 0);
    const SubmitResult worker1 =
        orchestrator.submit_next_level(C(23), single_tensor_args(0x2300, TensorArgType::OUTPUT), config, 1);
    orchestrator.close_run_submission(second_run);

    ASSERT_TRUE(endpoint0->wait_submitted(2));
    ASSERT_TRUE(endpoint1->wait_submitted(1));
    const std::vector<WorkerDispatch> worker0_submitted = endpoint0->submitted();
    const std::vector<WorkerDispatch> worker1_submitted = endpoint1->submitted();
    ASSERT_EQ(worker0_submitted.size(), 2u);
    ASSERT_EQ(worker1_submitted.size(), 1u);
    EXPECT_EQ(worker0_submitted[0].task_slot, first.task_slot);
    EXPECT_FALSE(worker0_submitted[0].prepare_only);
    EXPECT_EQ(worker0_submitted[1].task_slot, worker0.task_slot);
    EXPECT_TRUE(worker0_submitted[1].prepare_only);
    EXPECT_EQ(worker1_submitted[0].task_slot, worker1.task_slot);
    EXPECT_TRUE(worker1_submitted[0].prepare_only);
    EXPECT_EQ(orchestrator.active_run_id(), first_run);

    endpoint0->emit(WorkerProgressKind::ACCEPTED, worker0_submitted[0]);
    endpoint0->emit(WorkerProgressKind::COMPLETED, worker0_submitted[0]);
    ASSERT_TRUE(endpoint0->wait_activated(second_run));
    ASSERT_TRUE(endpoint1->wait_activated(second_run));

    endpoint0->emit(WorkerProgressKind::ACCEPTED, worker0_submitted[1]);
    endpoint1->emit(WorkerProgressKind::ACCEPTED, worker1_submitted[0]);
    endpoint0->emit(WorkerProgressKind::COMPLETED, worker0_submitted[1]);
    endpoint1->emit(WorkerProgressKind::COMPLETED, worker1_submitted[0]);
    ASSERT_TRUE(orchestrator.wait_run_for(first_run, 3.0));
    ASSERT_TRUE(orchestrator.wait_run_for(second_run, 3.0));
    orchestrator.release_run(first_run);
    orchestrator.release_run(second_run);
}

TEST_F(PartialStagingProgressSchedulerFixture, UnstageableWorkerDoesNotBlockLaterEligibleWorker) {
    DeterministicProgressEndpoint *endpoint2 = endpoints[2];
    const RunId first_run = orchestrator.begin_run();
    const SubmitResult first =
        orchestrator.submit_next_level(C(24), single_tensor_args(0x2400, TensorArgType::OUTPUT), config, 1);
    orchestrator.close_run_submission(first_run);
    ASSERT_TRUE(endpoint1->wait_submitted(1));

    const RunId second_run = orchestrator.begin_run();
    const SubmitResult worker0 =
        orchestrator.submit_next_level(C(25), single_tensor_args(0x2500, TensorArgType::OUTPUT), config, 0);
    const SubmitResult worker1 =
        orchestrator.submit_next_level(C(26), single_tensor_args(0x2600, TensorArgType::OUTPUT), config, 1);
    const SubmitResult worker2 =
        orchestrator.submit_next_level(C(27), single_tensor_args(0x2700, TensorArgType::OUTPUT), config, 2);
    orchestrator.close_run_submission(second_run);

    ASSERT_TRUE(endpoint0->wait_submitted(1));
    ASSERT_TRUE(endpoint2->wait_submitted(1));
    ASSERT_EQ(endpoint1->submitted().size(), 1u);
    const WorkerDispatch worker0_dispatch = endpoint0->submitted().front();
    const WorkerDispatch worker2_dispatch = endpoint2->submitted().front();
    EXPECT_EQ(worker0_dispatch.task_slot, worker0.task_slot);
    EXPECT_TRUE(worker0_dispatch.prepare_only);
    EXPECT_EQ(worker2_dispatch.task_slot, worker2.task_slot);
    EXPECT_TRUE(worker2_dispatch.prepare_only);
    EXPECT_EQ(orchestrator.active_run_id(), first_run);

    const WorkerDispatch first_dispatch = endpoint1->submitted().front();
    EXPECT_EQ(first_dispatch.task_slot, first.task_slot);
    endpoint1->emit(WorkerProgressKind::ACCEPTED, first_dispatch);
    endpoint1->emit(WorkerProgressKind::COMPLETED, first_dispatch);

    ASSERT_TRUE(endpoint0->wait_activated(second_run));
    ASSERT_TRUE(endpoint2->wait_activated(second_run));
    ASSERT_TRUE(endpoint1->wait_submitted(2));
    const WorkerDispatch worker1_dispatch = endpoint1->submitted().back();
    EXPECT_EQ(worker1_dispatch.task_slot, worker1.task_slot);
    EXPECT_FALSE(worker1_dispatch.prepare_only);

    endpoint0->emit(WorkerProgressKind::ACCEPTED, worker0_dispatch);
    endpoint1->emit(WorkerProgressKind::ACCEPTED, worker1_dispatch);
    endpoint2->emit(WorkerProgressKind::ACCEPTED, worker2_dispatch);
    endpoint0->emit(WorkerProgressKind::COMPLETED, worker0_dispatch);
    endpoint1->emit(WorkerProgressKind::COMPLETED, worker1_dispatch);
    endpoint2->emit(WorkerProgressKind::COMPLETED, worker2_dispatch);
    ASSERT_TRUE(orchestrator.wait_run_for(first_run, 3.0));
    ASSERT_TRUE(orchestrator.wait_run_for(second_run, 3.0));
    EXPECT_FALSE(orchestrator.run_failed(second_run));
    orchestrator.release_run(first_run);
    orchestrator.release_run(second_run);
}

TEST_F(ProgressSchedulerFixture, ReadySuccessorCannotExecuteBeforePendingPredecessorTail) {
    auto wait_for_activation_progress = [this] {
        for (int pass = 0; pass < 2; ++pass) {
            const uint64_t before = scheduler.dispatch_round_count();
            scheduler.notify_ready();
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
            while (scheduler.dispatch_round_count() <= before) {
                if (std::chrono::steady_clock::now() >= deadline) return false;
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            // Dispatch requests activation; the following loop's progress
            // delivers it. The mutex brackets each entire dispatch pass.
            std::scoped_lock lock(scheduler.loop_mutex());
        }
        return true;
    };
    const RunId first_run = orchestrator.begin_run();
    const auto first =
        orchestrator.submit_next_level(C(81), single_tensor_args(0x8100, TensorArgType::OUTPUT), config, 0);
    auto tail_args = single_tensor_args(0x8200, TensorArgType::OUTPUT);
    tail_args.add_dep_wait(first);
    const auto tail = orchestrator.submit_next_level(C(82), tail_args, config, 0);
    orchestrator.close_run_submission(first_run);
    ASSERT_TRUE(endpoint0->wait_submitted(1));
    EXPECT_EQ(allocator.slot_state(tail.task_slot)->state.load(), TaskState::PENDING);

    const RunId second_run = orchestrator.begin_run();
    const auto successor =
        orchestrator.submit_next_level(C(83), single_tensor_args(0x8200, TensorArgType::INPUT), config, 1);
    orchestrator.close_run_submission(second_run);
    ASSERT_TRUE(endpoint1->wait_submitted(1));
    const auto staged = endpoint1->submitted().front();
    EXPECT_EQ(staged.task_slot, successor.task_slot);
    ASSERT_TRUE(staged.prepare_only);
    ASSERT_TRUE(wait_for_activation_progress());
    EXPECT_EQ(orchestrator.dispatchable_run_id(), first_run);
    EXPECT_FALSE(endpoint1->wait_activated(second_run, std::chrono::milliseconds(0)));

    const auto first_dispatch = endpoint0->submitted().front();
    EXPECT_EQ(first_dispatch.task_slot, first.task_slot);
    endpoint0->emit(WorkerProgressKind::ACCEPTED, first_dispatch);
    endpoint0->emit(WorkerProgressKind::COMPLETED, first_dispatch);
    ASSERT_TRUE(endpoint0->wait_submitted(2));
    const auto tail_dispatch = endpoint0->submitted().back();
    EXPECT_EQ(tail_dispatch.task_slot, tail.task_slot);
    EXPECT_FALSE(tail_dispatch.prepare_only);
    endpoint0->emit(WorkerProgressKind::ACCEPTED, tail_dispatch);
    ASSERT_TRUE(wait_for_activation_progress());
    ASSERT_TRUE(orchestrator.run_accepted(first_run));
    EXPECT_FALSE(orchestrator.run_done(first_run));
    EXPECT_EQ(orchestrator.dispatchable_run_id(), first_run);
    EXPECT_FALSE(endpoint1->wait_activated(second_run, std::chrono::milliseconds(0)));

    endpoint0->emit(WorkerProgressKind::COMPLETED, tail_dispatch);
    ASSERT_TRUE(endpoint1->wait_activated(second_run));
    EXPECT_TRUE(orchestrator.run_done(first_run));
    EXPECT_EQ(orchestrator.dispatchable_run_id(), second_run);
    endpoint1->emit(WorkerProgressKind::ACCEPTED, staged);
    endpoint1->emit(WorkerProgressKind::COMPLETED, staged);
    ASSERT_TRUE(orchestrator.wait_run_for(second_run, 3.0));
    EXPECT_FALSE(orchestrator.run_failed(first_run));
    EXPECT_FALSE(orchestrator.run_failed(second_run));
    orchestrator.release_run(first_run);
    orchestrator.release_run(second_run);
}

// A diagnostics config once took the active FIFO lane instead of the prepared
// one, because collector setup wrote runner-global state during preparation.
// That state is now built and reset under the execution claim, so a diagnostic
// successor stages like any other. Nothing user-visible reports the difference —
// the run still completes and its artifacts still land — so without this the
// special case could come back unnoticed.
TEST_F(ProgressSchedulerFixture, DiagnosticSuccessorStagesLikeAnyOther) {
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

    ASSERT_TRUE(endpoint0->wait_submitted(2));
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

    endpoint0->emit(WorkerProgressKind::ACCEPTED, *first_dispatch);
    endpoint0->emit(WorkerProgressKind::COMPLETED, *first_dispatch);
    ASSERT_TRUE(endpoint0->wait_activated(second_run));
    endpoint0->emit(WorkerProgressKind::ACCEPTED, *second_dispatch);
    endpoint0->emit(WorkerProgressKind::COMPLETED, *second_dispatch);
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

TEST_F(CapacityOneProgressSchedulerFixture, CapacityShortageFallsBackToOrdinarySuccessorDispatch) {
    const RunId first_run = orchestrator.begin_run();
    orchestrator.submit_next_level(C(5), single_tensor_args(0x6000, TensorArgType::OUTPUT), config, 0);
    orchestrator.close_run_submission(first_run);
    ASSERT_TRUE(endpoint0->wait_submitted(1));

    const RunId second_run = orchestrator.begin_run();
    const SubmitResult second =
        orchestrator.submit_next_level(C(6), single_tensor_args(0x7000, TensorArgType::OUTPUT), config, 0);
    orchestrator.close_run_submission(second_run);

    uint64_t rounds_before_fence = 0;
    {
        std::scoped_lock lock(scheduler.loop_mutex());
        rounds_before_fence = scheduler.dispatch_round_count();
        scheduler.notify_ready();
    }
    const auto scan_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (scheduler.dispatch_round_count() < rounds_before_fence + 2 &&
           std::chrono::steady_clock::now() < scan_deadline) {
        std::this_thread::yield();
    }
    { std::scoped_lock lock(scheduler.loop_mutex()); }
    ASSERT_GE(scheduler.dispatch_round_count(), rounds_before_fence + 2);
    ASSERT_FALSE(orchestrator.run_failed(second_run));
    ASSERT_EQ(endpoint0->submitted().size(), 1u);
    const uint64_t preparable_queries_after_fence = preparable_run_query_count.load(std::memory_order_relaxed);
    ASSERT_GT(preparable_queries_after_fence, 0u);

    const uint64_t rounds_before_spin = scheduler.dispatch_round_count();
    const auto spin_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (scheduler.dispatch_round_count() < rounds_before_spin + 100 &&
           std::chrono::steady_clock::now() < spin_deadline) {
        std::this_thread::yield();
    }
    { std::scoped_lock lock(scheduler.loop_mutex()); }
    EXPECT_GE(scheduler.dispatch_round_count(), rounds_before_spin + 100);
    EXPECT_FALSE(orchestrator.run_failed(second_run));
    EXPECT_EQ(endpoint0->submitted().size(), 1u)
        << "busy-loop iterations must not repeatedly claim an unstageable successor";
    EXPECT_EQ(preparable_run_query_count.load(std::memory_order_relaxed), preparable_queries_after_fence)
        << "busy-loop iterations without a state change must not repeat the prepared-worker scan";

    const WorkerDispatch first_dispatch = endpoint0->submitted().front();
    endpoint0->emit(WorkerProgressKind::ACCEPTED, first_dispatch);
    endpoint0->emit(WorkerProgressKind::COMPLETED, first_dispatch);
    ASSERT_TRUE(endpoint0->wait_submitted(2));
    { std::scoped_lock lock(scheduler.loop_mutex()); }
    EXPECT_GT(preparable_run_query_count.load(std::memory_order_relaxed), preparable_queries_after_fence)
        << "a completion must trigger another prepared-worker scan";
    const WorkerDispatch second_dispatch = endpoint0->submitted().back();
    EXPECT_EQ(second_dispatch.task_slot, second.task_slot);
    EXPECT_FALSE(second_dispatch.prepare_only);
    endpoint0->emit(WorkerProgressKind::ACCEPTED, second_dispatch);
    endpoint0->emit(WorkerProgressKind::COMPLETED, second_dispatch);

    ASSERT_TRUE(orchestrator.wait_run_for(first_run, 3.0));
    ASSERT_TRUE(orchestrator.wait_run_for(second_run, 3.0));
    EXPECT_FALSE(orchestrator.run_failed(second_run));
    EXPECT_TRUE(scheduler.running());
    orchestrator.release_run(first_run);
    orchestrator.release_run(second_run);
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

// The endpoint must report acceptance as its own progress event, ahead of the
// completion event, so a waiter on wait_run_accepted is released while the task
// is still running rather than only once it finishes.
TEST(WorkerManagerTest, LocalMailboxPublishesAcceptanceBeforeCompletion) {
    MockMailboxWorker child;
    child.start();

    Ring allocator;
    allocator.init(/*heap_bytes=*/0);
    TaskSlot task_slot = make_progress_slot(allocator, /*run_id=*/31, /*pipeline_slot=*/1, /*generation=*/7);
    ASSERT_NE(task_slot, INVALID_SLOT);
    TaskSlotState *slot = allocator.slot_state(task_slot);
    ASSERT_NE(slot, nullptr);
    slot->callable.digest[0] = 0x42;

    LocalMailboxEndpoint endpoint(/*worker_id=*/0, child.mailbox_ptr());
    WorkerDispatch dispatch{task_slot, 0, /*dispatch_id=*/71, /*prepare_only=*/false};
    endpoint.submit_progress(&allocator, dispatch);

    child.wait_running();
    EXPECT_TRUE(child.is_running.load(std::memory_order_acquire));
    // The lease travels in the task frame the dispatch was published to, not
    // the control base frame.
    const char *published_frame =
        static_cast<const char *>(child.mailbox_ptr()) + MAILBOX_FIRST_TASK_FRAME * MAILBOX_FRAME_SIZE;
    PipelineSlotLease wire_lease{};
    std::memcpy(&wire_lease, published_frame + MAILBOX_OFF_PIPELINE_LEASE, sizeof(PipelineSlotLease));
    EXPECT_EQ(wire_lease.slot_id, 1u);
    EXPECT_EQ(wire_lease.generation, 7u);

    child.write_task_accepted();
    WorkerEndpointProgress progress;
    ASSERT_TRUE(poll_until(endpoint, progress));
    EXPECT_EQ(progress.kind, WorkerProgressKind::ACCEPTED);
    EXPECT_EQ(progress.dispatch.dispatch_id, dispatch.dispatch_id);

    // Acceptance is a distinct earlier event: the task is not complete yet.
    WorkerEndpointProgress not_yet;
    EXPECT_FALSE(endpoint.poll_progress(not_yet));

    child.complete();
    ASSERT_TRUE(poll_until(endpoint, progress));
    EXPECT_EQ(progress.kind, WorkerProgressKind::COMPLETED);
    EXPECT_EQ(progress.completion.outcome, EndpointOutcome::SUCCESS);
    allocator.shutdown();
}

// The ACK must not be carried by anything TASK_DONE can overwrite. The child
// publishes acceptance only into the sticky word — never into the state — and
// then completes immediately, so an endpoint that looks for acceptance in the
// state word observes none at all.
//
// This does not force the parent to skip a poll between the two writes: the
// parent is already polling by then and nothing here can stop it. What it pins
// is the property that makes that interleaving harmless — acceptance is
// readable after TASK_DONE, so losing a poll cannot lose the ACK.
TEST(WorkerManagerTest, AcceptanceIsReadableAfterTaskDone) {
    MockMailboxWorker child;
    child.start();

    Ring allocator;
    allocator.init(/*heap_bytes=*/0);
    TaskSlot task_slot = make_progress_slot(allocator, /*run_id=*/33, /*pipeline_slot=*/0, /*generation=*/1);
    ASSERT_NE(task_slot, INVALID_SLOT);
    TaskSlotState *slot = allocator.slot_state(task_slot);
    ASSERT_NE(slot, nullptr);
    slot->callable.digest[0] = 0x42;

    LocalMailboxEndpoint endpoint(/*worker_id=*/0, child.mailbox_ptr());
    WorkerDispatch dispatch{task_slot, 0, /*dispatch_id=*/73, /*prepare_only=*/false};
    endpoint.submit_progress(&allocator, dispatch);

    child.wait_running();
    EXPECT_TRUE(child.is_running.load(std::memory_order_acquire));
    // Back to back, with no parent poll in between.
    child.write_task_accepted();
    child.complete();

    bool saw_accepted = false;
    bool saw_completed = false;
    WorkerEndpointProgress progress;
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (!saw_completed && std::chrono::steady_clock::now() < deadline) {
        if (!endpoint.poll_progress(progress)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        if (progress.kind == WorkerProgressKind::ACCEPTED) saw_accepted = true;
        if (progress.kind == WorkerProgressKind::COMPLETED) {
            saw_completed = true;
            EXPECT_EQ(progress.completion.outcome, EndpointOutcome::SUCCESS);
        }
    }
    EXPECT_TRUE(saw_completed);
    EXPECT_TRUE(saw_accepted) << "the endpoint lost the launch ACK to a task that completed first";
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

// `node.submit` is what a dispatch arrow is drawn back to, so the swimlane can
// only pair the two if these attributes carry the same run/slot the dispatch
// reports. A SUB submission is deliberately silent: it has no NEXT_LEVEL worker
// to hand off to.
TEST_F(SchedulerFixture, NextLevelSubmitEmitsAPairableHostSpan) {
    ScopedHostSpanCapture host_span_capture;

    auto submitted = orch.submit_next_level(C(0x42), single_tensor_args(0xCAFE, TensorArgType::OUTPUT), cfg, 0);

    ASSERT_TRUE(captured_host_span(simpler::host_trace::host_span_name(simpler::host_trace::HostSpan::Submit)));
    std::ostringstream expected;
    expected << "run_id=" << run_id << " task_slot=" << submitted.task_slot
             << " group_index=0 group_size=1 worker_id=0 role=facade";
    EXPECT_EQ(
        captured_host_span_attrs(simpler::host_trace::host_span_name(simpler::host_trace::HostSpan::Submit)),
        expected.str()
    );

    mock_worker.wait_running();
    mock_worker.complete();
    wait_consumed(submitted.task_slot);
}

// A dispatch pass the scheduler thread has completed since `before`, so a test
// can assert on what an iteration did without a wall-clock budget deciding the
// verdict: a loaded box waits longer here rather than failing.
void wait_dispatch_round_after(Scheduler &sched, uint64_t before) {
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (sched.dispatch_round_count() == before) {
        if (std::chrono::steady_clock::now() > deadline) {
            FAIL() << "Scheduler ran no dispatch pass after being notified";
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

TEST_F(SchedulerFixture, ADispatchingLoopIterationEmitsOneSpanCountingWhatItDid) {
    ScopedHostSpanCapture host_span_capture;

    auto submitted = orch.submit_next_level(C(0x42), single_tensor_args(0xCAFE, TensorArgType::OUTPUT), cfg, 0);
    mock_worker.wait_running();
    mock_worker.complete();
    wait_consumed(submitted.task_slot);

    const std::string loop_span = simpler::host_trace::host_span_name(simpler::host_trace::HostSpan::SchedulerLoop);
    ASSERT_TRUE(captured_host_span(loop_span));
    // The first one is the iteration that dispatched this task: nothing had
    // completed yet, so it drained none, and `spins` counts the iterations
    // before it that did neither.
    const std::string attrs = captured_host_span_attrs(loop_span);
    EXPECT_EQ(attrs.rfind("role=scheduler drained=0 dispatched=1 drain_ns=", 0), 0u) << attrs;
    EXPECT_NE(attrs.find(" spins="), std::string::npos) << attrs;
    EXPECT_EQ(sched.dispatched_total(), 1u);
}

TEST_F(SchedulerFixture, AnIterationThatNeitherDrainedNorDispatchedEmitsNothing) {
    // A wake with nothing ready is an ordinary event — the loop also skips its
    // wait entirely while any worker is busy — so an unbounded number of these
    // iterations can run per unit of work. The round counter is the positive
    // control: without it, a loop that never woke would pass this test.
    ScopedHostSpanCapture host_span_capture;

    const uint64_t rounds_before = sched.dispatch_round_count();
    sched.notify_ready();
    wait_dispatch_round_after(sched, rounds_before);

    EXPECT_TRUE(captured_host_spans_empty());
    EXPECT_EQ(sched.dispatched_total(), 0u);
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

TEST_F(SchedulerFixture, ExplicitTaskDependencyOrdersIndependentTaskWithoutRetainingProducer) {
    auto producer = orch.submit_next_level(C(12), single_tensor_args(0xA100, TensorArgType::OUTPUT), cfg, 0);
    EXPECT_EQ(producer.run_id, run_id);

    TaskArgs consumer_args = single_tensor_args(0xB200, TensorArgType::OUTPUT);
    consumer_args.add_dep_wait(producer);
    auto consumer = orch.submit_next_level(C(13), consumer_args, cfg, 0);

    EXPECT_EQ(S(consumer.task_slot).state.load(std::memory_order_acquire), TaskState::PENDING);
    EXPECT_EQ(S(consumer.task_slot).fanin_count.load(std::memory_order_acquire), 1);
    EXPECT_TRUE(S(consumer.task_slot).fanin_producers.empty());
    {
        std::lock_guard<std::mutex> lk(S(producer.task_slot).fanout_mu);
        EXPECT_EQ(S(producer.task_slot).fanout_consumers, (std::vector<TaskSlot>{consumer.task_slot}));
        EXPECT_EQ(S(producer.task_slot).fanout_total, 0);
    }

    mock_worker.wait_running();
    ASSERT_EQ(mock_worker.dispatched_count(), 1);
    EXPECT_EQ(mock_worker.dispatched[0].callable_hash0, 12u);
    mock_worker.complete();

    wait_consumed(producer.task_slot);
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(300);
    while (mock_worker.dispatched_count() < 2 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_EQ(mock_worker.dispatched_count(), 2);
    EXPECT_EQ(mock_worker.dispatched[1].callable_hash0, 13u);

    mock_worker.complete();
    wait_consumed(consumer.task_slot);
}

TEST_F(SchedulerFixture, ExplicitWaitAndRetainDependenciesDeduplicateAndKeepProducerAlive) {
    auto producer = orch.submit_next_level(C(22), single_tensor_args(0xFA00, TensorArgType::OUTPUT), cfg, 0);

    TaskArgs consumer_args = single_tensor_args(0xFB00, TensorArgType::OUTPUT);
    consumer_args.add_dep_wait(producer);
    consumer_args.add_dep(producer);
    auto consumer = orch.submit_next_level(C(23), consumer_args, cfg, 0);

    EXPECT_EQ(S(consumer.task_slot).state.load(std::memory_order_acquire), TaskState::PENDING);
    EXPECT_EQ(S(consumer.task_slot).fanin_count.load(std::memory_order_acquire), 1);
    EXPECT_EQ(S(consumer.task_slot).fanin_producers, (std::vector<TaskSlot>{producer.task_slot}));
    {
        std::lock_guard<std::mutex> lk(S(producer.task_slot).fanout_mu);
        EXPECT_EQ(S(producer.task_slot).fanout_consumers, (std::vector<TaskSlot>{consumer.task_slot}));
        EXPECT_EQ(S(producer.task_slot).fanout_total, 1);
    }

    mock_worker.wait_running();
    mock_worker.complete();

    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(300);
    while (mock_worker.dispatched_count() < 2 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_EQ(mock_worker.dispatched_count(), 2);
    EXPECT_EQ(mock_worker.dispatched[1].callable_hash0, 23u);
    EXPECT_EQ(S(producer.task_slot).state.load(std::memory_order_acquire), TaskState::COMPLETED);

    mock_worker.complete();
    wait_consumed(producer.task_slot);
    wait_consumed(consumer.task_slot);
}

TEST_F(SchedulerFixture, ExplicitTaskDependencyOnAnAlreadyConsumedProducerIsSatisfied) {
    auto producer = orch.submit_next_level(C(18), single_tensor_args(0xF600, TensorArgType::OUTPUT), cfg, 0);
    mock_worker.wait_running();
    mock_worker.complete();
    wait_consumed(producer.task_slot);

    TaskArgs consumer_args = single_tensor_args(0xF700, TensorArgType::OUTPUT);
    consumer_args.add_dep_wait(producer);
    auto consumer = orch.submit_next_level(C(19), consumer_args, cfg, 0);

    EXPECT_EQ(S(consumer.task_slot).fanin_count.load(std::memory_order_acquire), 0);
    EXPECT_NE(S(consumer.task_slot).state.load(std::memory_order_acquire), TaskState::PENDING);
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(300);
    while (mock_worker.dispatched_count() < 2 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_EQ(mock_worker.dispatched_count(), 2);
    EXPECT_EQ(mock_worker.dispatched[1].callable_hash0, 19u);

    mock_worker.complete();
    wait_consumed(consumer.task_slot);
}

TEST_F(SchedulerFixture, FailedExplicitTaskDependencyPoisonsConsumer) {
    auto producer = orch.submit_next_level(C(20), single_tensor_args(0xF800, TensorArgType::OUTPUT), cfg, 0);
    TaskArgs consumer_args = single_tensor_args(0xF900, TensorArgType::OUTPUT);
    consumer_args.add_dep_wait(producer);
    auto consumer = orch.submit_next_level(C(21), consumer_args, cfg, 0);

    EXPECT_EQ(S(consumer.task_slot).state.load(std::memory_order_acquire), TaskState::PENDING);
    mock_worker.wait_running();
    mock_worker.complete_with_error("explicit producer boom");

    wait_consumed(producer.task_slot);
    wait_consumed(consumer.task_slot);
    EXPECT_TRUE(orch.run_failed(run_id));
    EXPECT_EQ(mock_worker.dispatched_count(), 1);
}

TEST_F(SchedulerFixture, ExplicitTaskDependencyDeduplicatesAnInferredLifetimeEdge) {
    auto producer = orch.submit_next_level(C(14), single_tensor_args(0xC300, TensorArgType::OUTPUT), cfg, 0);

    TaskArgs consumer_args = single_tensor_args(0xC300, TensorArgType::INPUT);
    consumer_args.add_dep_wait(producer);
    consumer_args.add_dep_wait(producer);
    auto consumer = orch.submit_next_level(C(15), consumer_args, cfg, 0);

    EXPECT_EQ(S(consumer.task_slot).fanin_count.load(std::memory_order_acquire), 1);
    EXPECT_EQ(S(consumer.task_slot).fanin_producers, (std::vector<TaskSlot>{producer.task_slot}));
    {
        std::lock_guard<std::mutex> lk(S(producer.task_slot).fanout_mu);
        EXPECT_EQ(S(producer.task_slot).fanout_consumers, (std::vector<TaskSlot>{consumer.task_slot}));
        EXPECT_EQ(S(producer.task_slot).fanout_total, 1);
    }

    mock_worker.wait_running();
    mock_worker.complete();
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(300);
    while (mock_worker.dispatched_count() < 2 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_EQ(mock_worker.dispatched_count(), 2);
    mock_worker.complete();
    wait_consumed(producer.task_slot);
    wait_consumed(consumer.task_slot);
}

TEST_F(SchedulerFixture, ExplicitTaskDependencyRejectsAHandleFromAnotherRunBeforeAllocating) {
    auto producer = orch.submit_next_level(C(16), single_tensor_args(0xD400, TensorArgType::OUTPUT), cfg, 0);
    TaskHandle foreign = producer;
    foreign.run_id++;

    TaskArgs consumer_args = single_tensor_args(0xE500, TensorArgType::OUTPUT);
    consumer_args.add_dep_wait(foreign);
    int32_t next_task_id = allocator.next_task_id();
    EXPECT_THROW((void)orch.submit_next_level(C(17), consumer_args, cfg, 0), std::invalid_argument);
    EXPECT_EQ(allocator.next_task_id(), next_task_id);

    mock_worker.wait_running();
    mock_worker.complete();
    wait_consumed(producer.task_slot);
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
        Tensor t = wire_tensor(0x1000u + static_cast<uint64_t>(i) * 0x100u);
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
    std::function<void()> after_group_phase_hook;

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
        c.after_group_phase_cb = [this] {
            if (after_group_phase_hook) after_group_phase_hook();
        };
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
        [](WorkerDispatch) {}
    );
    rq_next_level.reset(manager.next_level_worker_ids());

    auto allocate_slot = [&](RunId run_id, uint8_t callable_seed, int32_t worker_id, bool group) {
        AllocResult allocation = allocator.alloc(/*heap_bytes=*/0, /*scope_depth=*/0);
        TaskSlotState &state = *allocator.slot_state(allocation.slot);
        state.reset();
        state.run_id = run_id;
        state.pipeline_lease = PipelineSlotLease{/*slot_id=*/0, /*reserved=*/0, /*generation=*/run_id};
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

TEST_F(GroupSchedulerFixture, GroupPublishedBetweenGroupAndSinglePhasesStillOwnsItsTargets) {
    SubmitResult second_group;
    SubmitResult single_a;
    SubmitResult single_c;
    std::atomic<bool> published{false};
    std::promise<void> publication_done;
    std::future<void> publication_future = publication_done.get_future();
    after_group_phase_hook = [&] {
        // The scheduler runs a group phase every dispatch pass, including an
        // empty one it may reach before this thread's first_group submission is
        // visible. Publishing then would place second_group while cores 0/1 are
        // still idle, so it would dispatch to them instead of blocking behind
        // first_group. Fire only once first_group has claimed its targets — the
        // scheduler occupies each worker's active lane synchronously during that
        // dispatch, so idle() here reflects it on this same thread.
        if (manager.get_worker_by_id(WorkerType::NEXT_LEVEL, 0)->idle() ||
            manager.get_worker_by_id(WorkerType::NEXT_LEVEL, 1)->idle()) {
            return;
        }
        if (published.exchange(true, std::memory_order_acq_rel)) return;
        second_group = orch.submit_next_level_group(
            C(91), {single_tensor_args(0x202, TensorArgType::OUTPUT), single_tensor_args(0x203, TensorArgType::OUTPUT)},
            cfg, {1, 2}
        );
        single_a = orch.submit_next_level(C(92), single_tensor_args(0x204, TensorArgType::OUTPUT), cfg, 0);
        single_c = orch.submit_next_level(C(93), single_tensor_args(0x205, TensorArgType::OUTPUT), cfg, 2);
        publication_done.set_value();
    };

    SubmitResult first_group = orch.submit_next_level_group(
        C(90), {single_tensor_args(0x200, TensorArgType::OUTPUT), single_tensor_args(0x201, TensorArgType::OUTPUT)},
        cfg, {0, 1}
    );
    ASSERT_EQ(publication_future.wait_for(std::chrono::seconds(3)), std::future_status::ready);
    worker_a.wait_running();
    worker_b.wait_running();
    EXPECT_EQ(worker_c.dispatched_count(), 0)
        << "the later single must not cross a group that appeared between dispatch phases";

    worker_a.complete();
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
    while (worker_a.dispatched_count() < 2 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_EQ(worker_a.dispatched_count(), 2);
    EXPECT_EQ(worker_a.dispatched[1].callable_hash0, 92u);
    worker_a.complete();

    worker_b.complete();
    deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
    while ((worker_b.dispatched_count() < 2 || worker_c.dispatched_count() < 1) &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_EQ(worker_b.dispatched_count(), 2);
    ASSERT_EQ(worker_c.dispatched_count(), 1);
    EXPECT_EQ(worker_b.dispatched[1].callable_hash0, 91u);
    EXPECT_EQ(worker_c.dispatched[0].callable_hash0, 91u);
    worker_b.complete();
    worker_c.complete();

    deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
    while (worker_c.dispatched_count() < 2 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_EQ(worker_c.dispatched_count(), 2);
    EXPECT_EQ(worker_c.dispatched[1].callable_hash0, 93u);
    worker_c.complete();

    wait_consumed(first_group.task_slot);
    wait_consumed(second_group.task_slot);
    wait_consumed(single_a.task_slot);
    wait_consumed(single_c.task_slot);
}

TEST_F(GroupSchedulerFixture, BlockedGroupRemainsReadyUntilWorkerCompletion) {
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
    }

    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
    while ((worker_a.dispatched_count() < 2 || worker_b.dispatched_count() < 2) &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_EQ(worker_a.dispatched_count(), 2);
    ASSERT_EQ(worker_b.dispatched_count(), 2);
    EXPECT_EQ(worker_a.dispatched[1].callable_hash0, 75u);
    EXPECT_EQ(worker_b.dispatched[1].callable_hash0, 75u);

    worker_a.complete();
    worker_b.complete();
    wait_consumed(group.task_slot);

    deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
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

    worker_a.complete();
    worker_b.complete();

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
        Tensor t = wire_tensor(key);
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
    Tensor tensor = wire_tensor(0);
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
