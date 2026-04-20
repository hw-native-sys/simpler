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

#include "worker_manager.h"

#include <cstring>
#include <stdexcept>
#include <string>

#include "../worker/chip_worker.h"
#include "ring.h"

namespace {

// Read the child-written error message from the mailbox, guaranteeing
// NUL-termination even if the child wrote exactly MAILBOX_ERROR_MSG_SIZE
// bytes without a terminator.
std::string read_error_msg(const char *mbox) {
    char buf[MAILBOX_ERROR_MSG_SIZE + 1] = {};
    std::memcpy(buf, mbox + MAILBOX_OFF_ERROR_MSG, MAILBOX_ERROR_MSG_SIZE);
    buf[MAILBOX_ERROR_MSG_SIZE] = '\0';
    return std::string(buf);
}

}  // namespace

// =============================================================================
// WorkerThread — mailbox helpers (PROCESS mode)
// =============================================================================

MailboxState WorkerThread::read_mailbox_state() const {
    volatile int32_t *ptr = reinterpret_cast<volatile int32_t *>(mbox() + MAILBOX_OFF_STATE);
    int32_t v;
#if defined(__aarch64__)
    __asm__ volatile("ldar %w0, [%1]" : "=r"(v) : "r"(ptr) : "memory");
#elif defined(__x86_64__)
    v = *ptr;
    __asm__ volatile("" ::: "memory");
#else
    __atomic_load(ptr, &v, __ATOMIC_ACQUIRE);
#endif
    return static_cast<MailboxState>(v);
}

void WorkerThread::write_mailbox_state(MailboxState s) {
    volatile int32_t *ptr = reinterpret_cast<volatile int32_t *>(mbox() + MAILBOX_OFF_STATE);
    int32_t v = static_cast<int32_t>(s);
#if defined(__aarch64__)
    __asm__ volatile("stlr %w0, [%1]" : : "r"(v), "r"(ptr) : "memory");
#elif defined(__x86_64__)
    __asm__ volatile("" ::: "memory");
    *ptr = v;
#else
    __atomic_store(ptr, &v, __ATOMIC_RELEASE);
#endif
}

// =============================================================================
// WorkerThread — lifecycle
// =============================================================================

void WorkerThread::start(
    Mode mode, IWorker *worker, Ring *ring, WorkerManager *manager, const std::function<void(TaskSlot)> &on_complete,
    void *mailbox
) {
    mode_ = mode;
    worker_ = worker;
    ring_ = ring;
    manager_ = manager;
    on_complete_ = on_complete;
    mailbox_ = mailbox;
    shutdown_ = false;
    idle_.store(true, std::memory_order_relaxed);
    thread_ = std::thread(&WorkerThread::loop, this);
}

void WorkerThread::dispatch(WorkerDispatch d) {
    idle_.store(false, std::memory_order_release);
    std::lock_guard<std::mutex> lk(mu_);
    queue_.push(d);
    cv_.notify_one();
}

void WorkerThread::stop() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        shutdown_ = true;
    }
    cv_.notify_all();
    if (thread_.joinable()) thread_.join();
}

void WorkerThread::shutdown_child() {
    if (mode_ == Mode::PROCESS && mailbox_) {
        write_mailbox_state(MailboxState::SHUTDOWN);
    }
}

// =============================================================================
// WorkerThread — main loop + per-mode dispatch
// =============================================================================

void WorkerThread::loop() {
    while (true) {
        WorkerDispatch d;
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [this] {
                return !queue_.empty() || shutdown_;
            });
            if (queue_.empty()) break;
            d = queue_.front();
            queue_.pop();
        }

        TaskSlotState &s = *ring_->slot_state(d.task_slot);

        // Dispatch may throw from THREAD mode (worker_->run raised) or
        // PROCESS mode (dispatch_process saw non-zero ERROR from the
        // child). An uncaught exception escaping loop() would terminate
        // the std::thread via std::terminate — instead, capture it and
        // let the orch thread observe it at the next submit_*/drain.
        // on_complete_ still fires so the scheduler releases consumers
        // and active_tasks_ eventually reaches zero; otherwise drain()
        // would hang.
        try {
            if (mode_ == Mode::THREAD) {
                dispatch_thread(s, d.group_index);
            } else {
                dispatch_process(s, d.group_index);
            }
        } catch (...) {
            if (manager_) manager_->report_error(std::current_exception());
        }

        idle_.store(true, std::memory_order_release);
        on_complete_(d.task_slot);
    }
}

void WorkerThread::dispatch_thread(TaskSlotState &s, int32_t group_index) {
    uint64_t callable = (s.worker_type == WorkerType::SUB) ? static_cast<uint64_t>(s.callable_id) : s.callable;
    TaskArgsView view = s.args_view(group_index);
    worker_->run(callable, view, s.config);
}

void WorkerThread::dispatch_process(TaskSlotState &s, int32_t group_index) {
    uint64_t callable = (s.worker_type == WorkerType::SUB) ? static_cast<uint64_t>(s.callable_id) : s.callable;
    TaskArgsView view = s.args_view(group_index);

    // Clear the child-writable error fields so stale bytes from a prior
    // dispatch cannot masquerade as a fresh failure.
    int32_t zero_err = 0;
    std::memcpy(mbox() + MAILBOX_OFF_ERROR, &zero_err, sizeof(int32_t));
    std::memset(mbox() + MAILBOX_OFF_ERROR_MSG, 0, MAILBOX_ERROR_MSG_SIZE);

    // Write callable.
    std::memcpy(mbox() + MAILBOX_OFF_CALLABLE, &callable, sizeof(uint64_t));

    // Write config fields individually to avoid struct-layout portability issues.
    int32_t block_dim = s.config.block_dim;
    int32_t aicpu_tn = s.config.aicpu_thread_num;
    int32_t profiling = s.config.enable_profiling ? 1 : 0;
    int32_t dump_tensor = s.config.enable_dump_tensor ? 1 : 0;
    std::memcpy(mbox() + MAILBOX_OFF_BLOCK_DIM, &block_dim, sizeof(int32_t));
    std::memcpy(mbox() + MAILBOX_OFF_AICPU_THREAD_NUM, &aicpu_tn, sizeof(int32_t));
    std::memcpy(mbox() + MAILBOX_OFF_ENABLE_PROFILING, &profiling, sizeof(int32_t));
    std::memcpy(mbox() + MAILBOX_OFF_ENABLE_DUMP_TENSOR, &dump_tensor, sizeof(int32_t));

    // Write length-prefixed TaskArgs blob: [T][S][tensors][scalars].
    size_t blob_bytes = TASK_ARGS_BLOB_HEADER_SIZE + static_cast<size_t>(view.tensor_count) * sizeof(ContinuousTensor) +
                        static_cast<size_t>(view.scalar_count) * sizeof(uint64_t);
    if (blob_bytes > MAILBOX_ARGS_CAPACITY) {
        throw std::runtime_error("WorkerThread::dispatch_process: args blob exceeds mailbox capacity");
    }
    uint8_t *d = reinterpret_cast<uint8_t *>(mbox() + MAILBOX_OFF_ARGS);
    std::memcpy(d + 0, &view.tensor_count, sizeof(int32_t));
    std::memcpy(d + 4, &view.scalar_count, sizeof(int32_t));
    if (view.tensor_count > 0) {
        std::memcpy(
            d + TASK_ARGS_BLOB_HEADER_SIZE, view.tensors,
            static_cast<size_t>(view.tensor_count) * sizeof(ContinuousTensor)
        );
    }
    if (view.scalar_count > 0) {
        std::memcpy(
            d + TASK_ARGS_BLOB_HEADER_SIZE + static_cast<size_t>(view.tensor_count) * sizeof(ContinuousTensor),
            view.scalars, static_cast<size_t>(view.scalar_count) * sizeof(uint64_t)
        );
    }

    // Signal child process.
    write_mailbox_state(MailboxState::TASK_READY);

    // Spin-poll until child signals TASK_DONE.
    while (read_mailbox_state() != MailboxState::TASK_DONE) {
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }

    // Inspect the child's error report before releasing the mailbox back
    // to IDLE. Non-zero error_code means the child-side Python loop
    // caught an exception and filled OFF_ERROR_MSG with
    // `f"{type(e).__name__}: {e}"` (truncated to MAILBOX_ERROR_MSG_SIZE).
    int32_t error_code = 0;
    std::memcpy(&error_code, mbox() + MAILBOX_OFF_ERROR, sizeof(int32_t));
    if (error_code != 0) {
        std::string msg = read_error_msg(mbox());
        write_mailbox_state(MailboxState::IDLE);
        throw std::runtime_error(
            "WorkerThread::dispatch_process: child failed (code=" + std::to_string(error_code) + "): " + msg
        );
    }

    write_mailbox_state(MailboxState::IDLE);
}

// =============================================================================
// WorkerManager
// =============================================================================

void WorkerManager::add_next_level(IWorker *worker) {
    next_level_entries_.push_back({worker, WorkerThread::Mode::THREAD, nullptr});
}

void WorkerManager::add_sub(IWorker *worker) { sub_entries_.push_back({worker, WorkerThread::Mode::THREAD, nullptr}); }

void WorkerManager::add_next_level_process(void *mailbox) {
    next_level_entries_.push_back({nullptr, WorkerThread::Mode::PROCESS, mailbox});
}

void WorkerManager::add_sub_process(void *mailbox) {
    sub_entries_.push_back({nullptr, WorkerThread::Mode::PROCESS, mailbox});
}

void WorkerManager::start(Ring *ring, const OnCompleteFn &on_complete) {
    if (ring == nullptr) throw std::invalid_argument("WorkerManager::start: null ring");
    auto make_threads = [&](const std::vector<WorkerEntry> &entries,
                            std::vector<std::unique_ptr<WorkerThread>> &threads) {
        for (const WorkerEntry &e : entries) {
            auto wt = std::make_unique<WorkerThread>();
            wt->start(e.mode, e.worker, ring, this, on_complete, e.mailbox);
            threads.push_back(std::move(wt));
        }
    };
    make_threads(next_level_entries_, next_level_threads_);
    make_threads(sub_entries_, sub_threads_);
}

void WorkerManager::report_error(std::exception_ptr e) {
    if (!e) return;
    std::lock_guard<std::mutex> lk(err_mu_);
    if (first_error_) return;  // first-error-wins
    first_error_ = std::move(e);
    has_error_.store(true, std::memory_order_release);
}

std::exception_ptr WorkerManager::take_error() {
    std::lock_guard<std::mutex> lk(err_mu_);
    return first_error_;
}

void WorkerManager::clear_error() {
    std::lock_guard<std::mutex> lk(err_mu_);
    first_error_ = nullptr;
    has_error_.store(false, std::memory_order_release);
}

void WorkerManager::stop() {
    for (auto &wt : next_level_threads_)
        wt->stop();
    for (auto &wt : sub_threads_)
        wt->stop();
    next_level_threads_.clear();
    sub_threads_.clear();
}

void WorkerManager::shutdown_children() {
    for (auto &wt : next_level_threads_)
        wt->shutdown_child();
    for (auto &wt : sub_threads_)
        wt->shutdown_child();
}

WorkerThread *WorkerManager::pick_idle(WorkerType type) const {
    auto &threads = (type == WorkerType::NEXT_LEVEL) ? next_level_threads_ : sub_threads_;
    for (auto &wt : threads) {
        if (wt->idle()) return wt.get();
    }
    return nullptr;
}

std::vector<WorkerThread *> WorkerManager::pick_n_idle(WorkerType type, int n) const {
    auto &threads = (type == WorkerType::NEXT_LEVEL) ? next_level_threads_ : sub_threads_;
    std::vector<WorkerThread *> result;
    result.reserve(n);
    for (auto &wt : threads) {
        if (wt->idle()) {
            result.push_back(wt.get());
            if (static_cast<int>(result.size()) >= n) break;
        }
    }
    return result;
}

WorkerThread *WorkerManager::get_worker(WorkerType type, int logical_id) const {
    auto &threads = (type == WorkerType::NEXT_LEVEL) ? next_level_threads_ : sub_threads_;
    if (logical_id < 0 || static_cast<size_t>(logical_id) >= threads.size()) return nullptr;
    return threads[static_cast<size_t>(logical_id)].get();
}

WorkerThread *WorkerManager::pick_idle_excluding(WorkerType type, const std::vector<WorkerThread *> &exclude) const {
    auto &threads = (type == WorkerType::NEXT_LEVEL) ? next_level_threads_ : sub_threads_;
    for (auto &wt : threads) {
        if (!wt->idle()) continue;
        bool excluded = false;
        for (auto *ex : exclude) {
            if (ex == wt.get()) {
                excluded = true;
                break;
            }
        }
        if (!excluded) return wt.get();
    }
    return nullptr;
}

// =============================================================================
// WorkerThread — memory control (orch thread, concurrent with worker thread)
// =============================================================================

static void write_control_args(char *mbox, uint64_t sub_cmd, uint64_t a0 = 0, uint64_t a1 = 0, uint64_t a2 = 0) {
    std::memcpy(mbox + MAILBOX_OFF_CALLABLE, &sub_cmd, sizeof(uint64_t));
    std::memcpy(mbox + CTRL_OFF_ARG0, &a0, sizeof(uint64_t));
    std::memcpy(mbox + CTRL_OFF_ARG1, &a1, sizeof(uint64_t));
    std::memcpy(mbox + CTRL_OFF_ARG2, &a2, sizeof(uint64_t));
}

static uint64_t read_control_result(const char *mbox) {
    uint64_t r;
    std::memcpy(&r, mbox + CTRL_OFF_RESULT, sizeof(uint64_t));
    return r;
}

uint64_t WorkerThread::control_malloc(size_t size) {
    if (mode_ == Mode::THREAD) {
        auto *cw = dynamic_cast<ChipWorker *>(worker_);
        if (!cw) throw std::runtime_error("control_malloc: worker is not a ChipWorker");
        return cw->malloc(size);
    }
    int32_t zero_err = 0;
    std::memcpy(mbox() + MAILBOX_OFF_ERROR, &zero_err, sizeof(int32_t));
    std::memset(mbox() + MAILBOX_OFF_ERROR_MSG, 0, MAILBOX_ERROR_MSG_SIZE);
    write_control_args(mbox(), CTRL_MALLOC, static_cast<uint64_t>(size));
    write_mailbox_state(MailboxState::CONTROL_REQUEST);
    while (read_mailbox_state() != MailboxState::CONTROL_DONE) {}
    int32_t err;
    std::memcpy(&err, mbox() + MAILBOX_OFF_ERROR, sizeof(int32_t));
    if (err != 0) {
        std::string msg = read_error_msg(mbox());
        write_mailbox_state(MailboxState::IDLE);
        throw std::runtime_error("control_malloc failed on child: " + msg);
    }
    uint64_t result = read_control_result(mbox());
    write_mailbox_state(MailboxState::IDLE);
    return result;
}

void WorkerThread::control_free(uint64_t ptr) {
    if (mode_ == Mode::THREAD) {
        auto *cw = dynamic_cast<ChipWorker *>(worker_);
        if (!cw) throw std::runtime_error("control_free: worker is not a ChipWorker");
        cw->free(ptr);
        return;
    }
    write_control_args(mbox(), CTRL_FREE, ptr);
    write_mailbox_state(MailboxState::CONTROL_REQUEST);
    while (read_mailbox_state() != MailboxState::CONTROL_DONE) {}
    write_mailbox_state(MailboxState::IDLE);
}

void WorkerThread::control_copy_to(uint64_t dst, uint64_t src, size_t size) {
    if (mode_ == Mode::THREAD) {
        auto *cw = dynamic_cast<ChipWorker *>(worker_);
        if (!cw) throw std::runtime_error("control_copy_to: worker is not a ChipWorker");
        cw->copy_to(dst, src, size);
        return;
    }
    write_control_args(mbox(), CTRL_COPY_TO, dst, src, static_cast<uint64_t>(size));
    write_mailbox_state(MailboxState::CONTROL_REQUEST);
    while (read_mailbox_state() != MailboxState::CONTROL_DONE) {}
    write_mailbox_state(MailboxState::IDLE);
}

void WorkerThread::control_copy_from(uint64_t dst, uint64_t src, size_t size) {
    if (mode_ == Mode::THREAD) {
        auto *cw = dynamic_cast<ChipWorker *>(worker_);
        if (!cw) throw std::runtime_error("control_copy_from: worker is not a ChipWorker");
        cw->copy_from(dst, src, size);
        return;
    }
    write_control_args(mbox(), CTRL_COPY_FROM, dst, src, static_cast<uint64_t>(size));
    write_mailbox_state(MailboxState::CONTROL_REQUEST);
    while (read_mailbox_state() != MailboxState::CONTROL_DONE) {}
    write_mailbox_state(MailboxState::IDLE);
}

bool WorkerManager::any_busy() const {
    for (auto &wt : next_level_threads_)
        if (!wt->idle()) return true;
    for (auto &wt : sub_threads_)
        if (!wt->idle()) return true;
    return false;
}
