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

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

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
// WorkerThread — mailbox helpers
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
    Ring *ring, WorkerManager *manager, const std::function<void(TaskSlot)> &on_complete, void *mailbox
) {
    if (mailbox == nullptr) throw std::invalid_argument("WorkerThread::start: null mailbox");
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
    if (mailbox_) {
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

        // dispatch_process may throw on a non-zero ERROR from the child.
        // An uncaught exception escaping loop() would terminate the
        // std::thread via std::terminate — instead, capture it and let
        // the orch thread observe it at the next submit_*/drain.
        // on_complete_ still fires so the scheduler releases consumers
        // and active_tasks_ eventually reaches zero; otherwise drain()
        // would hang.
        try {
            dispatch_process(s, d.group_index);
        } catch (...) {
            if (manager_) manager_->report_error(std::current_exception());
        }

        idle_.store(true, std::memory_order_release);
        on_complete_(d.task_slot);
    }
}

void WorkerThread::dispatch_process(TaskSlotState &s, int32_t group_index) {
    uint64_t callable = static_cast<uint64_t>(static_cast<uint32_t>(s.callable_id));
    TaskArgsView view = s.args_view(group_index);

    // Hold mailbox_mu_ for the entire round trip (write payload + state +
    // spin-poll TASK_DONE + reset to IDLE). Any control_* request from the
    // orch thread waits for the dispatch to finish before claiming the
    // mailbox; without this they would race on MAILBOX_OFF_STATE.
    std::lock_guard<std::mutex> lk(mailbox_mu_);

    // Clear the child-writable error fields so stale bytes from a prior
    // dispatch cannot masquerade as a fresh failure.
    int32_t zero_err = 0;
    std::memcpy(mbox() + MAILBOX_OFF_ERROR, &zero_err, sizeof(int32_t));
    std::memset(mbox() + MAILBOX_OFF_ERROR_MSG, 0, MAILBOX_ERROR_MSG_SIZE);

    // Write callable.
    std::memcpy(mbox() + MAILBOX_OFF_CALLABLE, &callable, sizeof(uint64_t));

    // Write config as a single packed POD block (see call_config.h).
    std::memcpy(mbox() + MAILBOX_OFF_CONFIG, &s.config, sizeof(CallConfig));

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

void WorkerManager::add_next_level(void *mailbox) { next_level_entries_.push_back(mailbox); }

void WorkerManager::add_sub(void *mailbox) { sub_entries_.push_back(mailbox); }

void WorkerManager::start(Ring *ring, const OnCompleteFn &on_complete) {
    if (ring == nullptr) throw std::invalid_argument("WorkerManager::start: null ring");
    auto make_threads = [&](const std::vector<void *> &entries, std::vector<std::unique_ptr<WorkerThread>> &threads) {
        for (void *mailbox : entries) {
            auto wt = std::make_unique<WorkerThread>();
            wt->start(ring, this, on_complete, mailbox);
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

// Issue a control sub-command and block until the child publishes
// CONTROL_DONE. Caller must hold `mailbox_mu_`. On a non-zero error code
// from the child, throws and leaves the mailbox in IDLE before unwinding
// (so the next claim starts from a clean state). The `op_name` is used
// only for the exception message.
void WorkerThread::run_control_command(const char *op_name) {
    int32_t zero_err = 0;
    std::memcpy(mbox() + MAILBOX_OFF_ERROR, &zero_err, sizeof(int32_t));
    std::memset(mbox() + MAILBOX_OFF_ERROR_MSG, 0, MAILBOX_ERROR_MSG_SIZE);
    write_mailbox_state(MailboxState::CONTROL_REQUEST);
    while (read_mailbox_state() != MailboxState::CONTROL_DONE) {}
    int32_t err = 0;
    std::memcpy(&err, mbox() + MAILBOX_OFF_ERROR, sizeof(int32_t));
    if (err != 0) {
        std::string msg = read_error_msg(mbox());
        write_mailbox_state(MailboxState::IDLE);
        throw std::runtime_error(std::string(op_name) + " failed on child: " + msg);
    }
    write_mailbox_state(MailboxState::IDLE);
}

uint64_t WorkerThread::control_malloc(size_t size) {
    std::lock_guard<std::mutex> lk(mailbox_mu_);
    write_control_args(mbox(), CTRL_MALLOC, static_cast<uint64_t>(size));
    run_control_command("control_malloc");
    return read_control_result(mbox());
}

void WorkerThread::control_prepare(int32_t cid) {
    std::lock_guard<std::mutex> lk(mailbox_mu_);
    write_control_args(mbox(), CTRL_PREPARE, static_cast<uint64_t>(static_cast<uint32_t>(cid)));
    run_control_command("control_prepare");
}

void WorkerThread::control_register(int32_t cid, const char *shm_name) {
    std::lock_guard<std::mutex> lk(mailbox_mu_);
    // OFF_ERROR / OFF_ERROR_MSG are cleared by run_control_command — no
    // prelude memset needed (matches the other control_* methods).
    uint64_t sub_cmd = CTRL_REGISTER;
    std::memcpy(mbox() + MAILBOX_OFF_CALLABLE, &sub_cmd, sizeof(uint64_t));
    uint64_t cid_v = static_cast<uint32_t>(cid);
    std::memcpy(mbox() + CTRL_OFF_ARG0, &cid_v, sizeof(uint64_t));
    // Stage the NUL-terminated shm name in the args region. Pad with zeros so
    // stale bytes from a prior control op cannot leak into the child's decode.
    size_t name_len = std::strlen(shm_name);
    if (name_len + 1 > CTRL_SHM_NAME_BYTES) {
        throw std::runtime_error(std::string("control_register: shm name too long: ") + shm_name);
    }
    std::memcpy(mbox() + MAILBOX_OFF_ARGS, shm_name, name_len);
    std::memset(mbox() + MAILBOX_OFF_ARGS + name_len, 0, CTRL_SHM_NAME_BYTES - name_len);
    run_control_command("control_register");
}

void WorkerThread::control_unregister(int32_t cid) {
    std::lock_guard<std::mutex> lk(mailbox_mu_);
    write_control_args(mbox(), CTRL_UNREGISTER, static_cast<uint64_t>(static_cast<uint32_t>(cid)));
    run_control_command("control_unregister");
}

void WorkerThread::control_free(uint64_t ptr) {
    std::lock_guard<std::mutex> lk(mailbox_mu_);
    write_control_args(mbox(), CTRL_FREE, ptr);
    run_control_command("control_free");
}

void WorkerThread::control_copy_to(uint64_t dst, uint64_t src, size_t size) {
    std::lock_guard<std::mutex> lk(mailbox_mu_);
    write_control_args(mbox(), CTRL_COPY_TO, dst, src, static_cast<uint64_t>(size));
    run_control_command("control_copy_to");
}

void WorkerThread::control_copy_from(uint64_t dst, uint64_t src, size_t size) {
    std::lock_guard<std::mutex> lk(mailbox_mu_);
    write_control_args(mbox(), CTRL_COPY_FROM, dst, src, static_cast<uint64_t>(size));
    run_control_command("control_copy_from");
}

bool WorkerManager::any_busy() const {
    for (auto &wt : next_level_threads_)
        if (!wt->idle()) return true;
    for (auto &wt : sub_threads_)
        if (!wt->idle()) return true;
    return false;
}

// =============================================================================
// Dynamic register/unregister broadcast (POSIX shm staging + parallel fan-out)
// =============================================================================

namespace {

// Process-wide monotonic counter so concurrent broadcasts to the same cid
// don't collide on shm name. Atomic increment is enough — no need to lock.
std::atomic<uint64_t> g_shm_counter{0};

// Build the per-broadcast POSIX shm name. The name itself does NOT carry the
// leading '/' that shm_open requires (Python's multiprocessing.SharedMemory
// uses the same convention, so the child Python side reads the field as a
// plain name). Caller adds '/' when opening.
std::string make_shm_name(int32_t cid) {
    char buf[CTRL_SHM_NAME_BYTES];
    int pid = static_cast<int>(getpid());
    uint64_t counter = g_shm_counter.fetch_add(1, std::memory_order_relaxed);
    int n = std::snprintf(
        buf, sizeof(buf), "simpler-cb-%d-%d-%llu", pid, static_cast<int>(cid), static_cast<unsigned long long>(counter)
    );
    if (n < 0 || static_cast<size_t>(n) >= sizeof(buf)) {
        throw std::runtime_error("broadcast_register: shm name overflow");
    }
    return std::string(buf);
}

// Strip the outer "<op_name> failed on child: " prefix that
// run_control_command prepends to every control failure, so the broadcast
// caller can surface the child-side message (`register cid=<N>
// chip=<id>: <reason>` per docs §6) directly under its own one-line
// Worker.{register,unregister}(cid=N) prefix.
std::string strip_control_prefix(const std::string &msg, const std::string &op_name) {
    const std::string needle = op_name + " failed on child: ";
    if (msg.compare(0, needle.size(), needle) == 0) {
        return msg.substr(needle.size());
    }
    return msg;
}

// RAII guard for a POSIX shm segment: create on construction, unlink on
// destruction. mmaps the region so the staged blob can be memcpy'd in
// place; the mmap is released in the destructor as well. The shm is only
// unlinked once — children open by name *before* this guard is destroyed.
class PosixShmHolder {
public:
    PosixShmHolder(const std::string &name, size_t size) :
        name_(name),
        size_(size) {
        std::string full_name = "/" + name_;
        fd_ = shm_open(full_name.c_str(), O_CREAT | O_RDWR | O_EXCL, 0600);
        if (fd_ < 0) {
            throw std::runtime_error(
                std::string("broadcast_register: shm_open(") + full_name + ") failed: " + std::strerror(errno)
            );
        }
        if (ftruncate(fd_, static_cast<off_t>(size)) != 0) {
            int err = errno;
            ::close(fd_);
            shm_unlink(full_name.c_str());
            throw std::runtime_error(std::string("broadcast_register: ftruncate failed: ") + std::strerror(err));
        }
        addr_ = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
        if (addr_ == MAP_FAILED) {
            int err = errno;
            ::close(fd_);
            shm_unlink(full_name.c_str());
            addr_ = nullptr;
            throw std::runtime_error(std::string("broadcast_register: mmap failed: ") + std::strerror(err));
        }
    }
    ~PosixShmHolder() {
        if (addr_ != nullptr) munmap(addr_, size_);
        if (fd_ >= 0) ::close(fd_);
        std::string full_name = "/" + name_;
        shm_unlink(full_name.c_str());
    }
    PosixShmHolder(const PosixShmHolder &) = delete;
    PosixShmHolder &operator=(const PosixShmHolder &) = delete;

    void *addr() { return addr_; }
    const std::string &name() const { return name_; }

private:
    std::string name_;
    size_t size_{0};
    int fd_{-1};
    void *addr_{nullptr};
};

}  // namespace

void WorkerManager::control_prepare(int worker_id, int32_t cid) {
    auto *wt = get_worker(WorkerType::NEXT_LEVEL, worker_id);
    if (wt == nullptr) {
        throw std::runtime_error("control_prepare: invalid worker_id " + std::to_string(worker_id));
    }
    wt->control_prepare(cid);
}

void WorkerManager::broadcast_register_all(int32_t cid, const void *blob_ptr, size_t blob_size) {
    if (next_level_threads_.empty()) return;

    std::string shm_name = make_shm_name(cid);
    PosixShmHolder shm(shm_name, blob_size);
    std::memcpy(shm.addr(), blob_ptr, blob_size);

    // Fan out to every WorkerThread in parallel. Per-WorkerThread mailbox_mu_
    // is independent, so N control_register calls run concurrently — latency
    // is 1 × prepare_cost instead of N × prepare_cost.
    std::vector<std::exception_ptr> errors(next_level_threads_.size(), nullptr);
    std::vector<std::thread> workers;
    workers.reserve(next_level_threads_.size());
    for (size_t i = 0; i < next_level_threads_.size(); ++i) {
        workers.emplace_back([this, i, cid, name = shm.name(), &errors]() {
            try {
                next_level_threads_[i]->control_register(cid, name.c_str());
            } catch (...) {
                errors[i] = std::current_exception();
            }
        });
    }
    for (auto &t : workers)
        t.join();

    // shm is unlinked when `shm` goes out of scope. Children opened it by
    // name during control_register and have already closed their mappings
    // before publishing CONTROL_DONE — see python/simpler/worker.py.

    for (size_t i = 0; i < errors.size(); ++i) {
        if (errors[i]) {
            try {
                std::rethrow_exception(errors[i]);
            } catch (const std::exception &e) {
                std::string msg = strip_control_prefix(e.what(), "control_register");
                throw std::runtime_error(
                    std::string("Worker.register(cid=") + std::to_string(cid) + ") failed on next_level " +
                    std::to_string(i) + ": " + msg
                );
            }
        }
    }
}

std::vector<std::string> WorkerManager::broadcast_unregister_all(int32_t cid) {
    std::vector<std::string> errors;
    if (next_level_threads_.empty()) return errors;

    std::vector<std::string> per_worker(next_level_threads_.size());
    std::vector<std::thread> workers;
    workers.reserve(next_level_threads_.size());
    for (size_t i = 0; i < next_level_threads_.size(); ++i) {
        workers.emplace_back([this, i, cid, &per_worker]() {
            try {
                next_level_threads_[i]->control_unregister(cid);
            } catch (const std::exception &e) {
                std::string msg = strip_control_prefix(e.what(), "control_unregister");
                per_worker[i] = std::string("next_level ") + std::to_string(i) + ": " + msg;
            }
        });
    }
    for (auto &t : workers)
        t.join();

    for (auto &msg : per_worker) {
        if (!msg.empty()) errors.push_back(std::move(msg));
    }
    return errors;
}
