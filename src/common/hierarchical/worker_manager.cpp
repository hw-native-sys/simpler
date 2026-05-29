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

std::string format_digest(const uint8_t *digest) {
    if (digest == nullptr) return "sha256:<null>";
    static constexpr char kHex[] = "0123456789abcdef";
    std::string out = "sha256:";
    out.reserve(71);
    for (size_t i = 0; i < CALLABLE_HASH_DIGEST_SIZE; ++i) {
        uint8_t v = digest[i];
        out.push_back(kHex[v >> 4]);
        out.push_back(kHex[v & 0x0F]);
    }
    return out;
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
    TaskArgsView view = s.args_view(group_index);

    // Hold mailbox_mu_ for the entire round trip (write payload + state +
    // spin-poll TASK_DONE + reset to IDLE). Any control_* request from the
    // orch thread waits for the dispatch to finish before claiming the
    // mailbox; without this they would race on MAILBOX_OFF_STATE.
    std::lock_guard<std::mutex> lk(mailbox_mu_);
    if (mailbox_control_timed_out_) {
        throw std::runtime_error("WorkerThread::dispatch_process: mailbox has an unresolved timed-out control command");
    }

    // Clear the child-writable error fields so stale bytes from a prior
    // dispatch cannot masquerade as a fresh failure.
    int32_t zero_err = 0;
    std::memcpy(mbox() + MAILBOX_OFF_ERROR, &zero_err, sizeof(int32_t));
    std::memset(mbox() + MAILBOX_OFF_ERROR_MSG, 0, MAILBOX_ERROR_MSG_SIZE);

    uint64_t reserved_callable = 0;
    std::memcpy(mbox() + MAILBOX_OFF_CALLABLE, &reserved_callable, sizeof(uint64_t));

    // Write config as a single packed POD block (see call_config.h).
    std::memcpy(mbox() + MAILBOX_OFF_CONFIG, &s.config, sizeof(CallConfig));

    // Write length-prefixed TaskArgs blob: [T][S][tensors][scalars].
    size_t blob_bytes = TASK_ARGS_BLOB_HEADER_SIZE + static_cast<size_t>(view.tensor_count) * sizeof(ContinuousTensor) +
                        static_cast<size_t>(view.scalar_count) * sizeof(uint64_t);
    if (blob_bytes > MAILBOX_ARGS_CAPACITY) {
        throw std::runtime_error("WorkerThread::dispatch_process: args blob exceeds mailbox capacity");
    }
    uint8_t *hash_dst = reinterpret_cast<uint8_t *>(mbox() + MAILBOX_OFF_TASK_CALLABLE_HASH);
    std::memcpy(hash_dst, s.callable.digest.data(), CALLABLE_HASH_DIGEST_SIZE);

    uint8_t *d = reinterpret_cast<uint8_t *>(mbox() + MAILBOX_OFF_TASK_ARGS_BLOB);
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

static void write_control_digest(char *mbox, const uint8_t *digest) {
    if (digest == nullptr) {
        std::memset(mbox + MAILBOX_OFF_CONTROL_CALLABLE_HASH, 0, CALLABLE_HASH_DIGEST_SIZE);
        return;
    }
    std::memcpy(mbox + MAILBOX_OFF_CONTROL_CALLABLE_HASH, digest, CALLABLE_HASH_DIGEST_SIZE);
}

// Issue a control sub-command and block until the child publishes
// CONTROL_DONE. Caller must hold `mailbox_mu_`. On a non-zero error code
// from the child, throws and leaves the mailbox in IDLE before unwinding
// (so the next claim starts from a clean state). The `op_name` is used
// only for the exception message.
void WorkerThread::run_control_command(const char *op_name, double timeout_s) {
    if (mailbox_control_timed_out_) {
        throw std::runtime_error(std::string(op_name) + " failed: mailbox has an unresolved timed-out control command");
    }
    int32_t zero_err = 0;
    std::memcpy(mbox() + MAILBOX_OFF_ERROR, &zero_err, sizeof(int32_t));
    std::memset(mbox() + MAILBOX_OFF_ERROR_MSG, 0, MAILBOX_ERROR_MSG_SIZE);
    write_mailbox_state(MailboxState::CONTROL_REQUEST);
    auto deadline = std::chrono::steady_clock::time_point::max();
    if (timeout_s >= 0.0) {
        deadline =
            std::chrono::steady_clock::now() +
            std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(timeout_s));
    }
    while (read_mailbox_state() != MailboxState::CONTROL_DONE) {
        if (std::chrono::steady_clock::now() >= deadline) {
            mailbox_control_timed_out_ = true;
            throw std::runtime_error(std::string(op_name) + " timed out waiting for CONTROL_DONE");
        }
    }
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

void WorkerThread::control_prepare(const uint8_t *digest) {
    std::lock_guard<std::mutex> lk(mailbox_mu_);
    write_control_args(mbox(), CTRL_PREPARE);
    write_control_digest(mbox(), digest);
    run_control_command("control_prepare");
}

void WorkerThread::control_register(const char *shm_name, const uint8_t *digest) {
    std::lock_guard<std::mutex> lk(mailbox_mu_);
    // OFF_ERROR / OFF_ERROR_MSG are cleared by run_control_command — no
    // prelude memset needed (matches the other control_* methods).
    uint64_t sub_cmd = CTRL_REGISTER;
    std::memcpy(mbox() + MAILBOX_OFF_CALLABLE, &sub_cmd, sizeof(uint64_t));
    uint64_t reserved = 0;
    std::memcpy(mbox() + CTRL_OFF_ARG0, &reserved, sizeof(uint64_t));
    write_control_digest(mbox(), digest);
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

void WorkerThread::control_unregister(const uint8_t *digest) {
    std::lock_guard<std::mutex> lk(mailbox_mu_);
    write_control_args(mbox(), CTRL_UNREGISTER);
    write_control_digest(mbox(), digest);
    run_control_command("control_unregister");
}

void WorkerThread::control_generic(uint64_t sub_cmd, const char *shm_name, double timeout_s, const uint8_t *digest) {
    std::lock_guard<std::mutex> lk(mailbox_mu_);
    std::memcpy(mbox() + MAILBOX_OFF_CALLABLE, &sub_cmd, sizeof(uint64_t));
    uint64_t reserved = 0;
    std::memcpy(mbox() + CTRL_OFF_ARG0, &reserved, sizeof(uint64_t));
    write_control_digest(mbox(), digest);
    const char *name = shm_name ? shm_name : "";
    size_t name_len = std::strlen(name);
    if (name_len + 1 > CTRL_SHM_NAME_BYTES) {
        throw std::runtime_error(std::string("control_generic: shm name too long: ") + name);
    }
    if (name_len > 0) std::memcpy(mbox() + MAILBOX_OFF_ARGS, name, name_len);
    std::memset(mbox() + MAILBOX_OFF_ARGS + name_len, 0, CTRL_SHM_NAME_BYTES - name_len);
    run_control_command("control_generic", timeout_s);
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

// Stage two NUL-terminated shm names at MAILBOX_OFF_ARGS: request first
// (CTRL_SHM_NAME_BYTES wide) then reply (CTRL_SHM_NAME_BYTES wide).  Pads each
// slot with zeros so stale bytes from a prior op cannot leak into the child's
// decode.  `reply_shm_name` may be empty (NUL byte) for release.
static void write_shm_name_pair(char *mbox, const char *request_shm_name, const char *reply_shm_name) {
    auto write_one = [&](char *dst, const char *name) {
        size_t n = name ? std::strlen(name) : 0;
        if (n + 1 > CTRL_SHM_NAME_BYTES) {
            throw std::runtime_error(std::string("control: shm name too long: ") + (name ? name : "(null)"));
        }
        if (n > 0) std::memcpy(dst, name, n);
        std::memset(dst + n, 0, CTRL_SHM_NAME_BYTES - n);
    };
    write_one(mbox + MAILBOX_OFF_ARGS, request_shm_name);
    write_one(mbox + MAILBOX_OFF_ARGS + CTRL_SHM_NAME_BYTES, reply_shm_name);
}

void WorkerThread::control_alloc_domain(const char *request_shm_name, const char *reply_shm_name) {
    if (!request_shm_name || !*request_shm_name || !reply_shm_name || !*reply_shm_name) {
        throw std::runtime_error("control_alloc_domain: request and reply shm names must be non-empty");
    }
    std::lock_guard<std::mutex> lk(mailbox_mu_);
    uint64_t sub_cmd = CTRL_ALLOC_DOMAIN;
    std::memcpy(mbox() + MAILBOX_OFF_CALLABLE, &sub_cmd, sizeof(uint64_t));
    write_shm_name_pair(mbox(), request_shm_name, reply_shm_name);
    run_control_command("control_alloc_domain");
}

void WorkerThread::control_release_domain(const char *request_shm_name) {
    if (!request_shm_name || !*request_shm_name) {
        throw std::runtime_error("control_release_domain: request shm name must be non-empty");
    }
    std::lock_guard<std::mutex> lk(mailbox_mu_);
    uint64_t sub_cmd = CTRL_RELEASE_DOMAIN;
    std::memcpy(mbox() + MAILBOX_OFF_CALLABLE, &sub_cmd, sizeof(uint64_t));
    write_shm_name_pair(mbox(), request_shm_name, "");
    run_control_command("control_release_domain");
}

void WorkerThread::control_comm_init(const char *request_shm_name) {
    if (!request_shm_name || !*request_shm_name) {
        throw std::runtime_error("control_comm_init: request shm name must be non-empty");
    }
    std::lock_guard<std::mutex> lk(mailbox_mu_);
    uint64_t sub_cmd = CTRL_COMM_INIT;
    std::memcpy(mbox() + MAILBOX_OFF_CALLABLE, &sub_cmd, sizeof(uint64_t));
    write_shm_name_pair(mbox(), request_shm_name, "");
    run_control_command("control_comm_init");
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

// Process-wide monotonic counter so concurrent broadcasts do not collide on shm
// name. Atomic increment is enough — no need to lock.
std::atomic<uint64_t> g_shm_counter{0};

// Build the per-broadcast POSIX shm name. The name itself does NOT carry the
// leading '/' that shm_open requires (Python's multiprocessing.SharedMemory
// uses the same convention, so the child Python side reads the field as a
// plain name). Caller adds '/' when opening.
std::string make_shm_name() {
    char buf[CTRL_SHM_NAME_BYTES];
    int pid = static_cast<int>(getpid());
    uint64_t counter = g_shm_counter.fetch_add(1, std::memory_order_relaxed);
    int n = std::snprintf(buf, sizeof(buf), "simpler-cb-%d-%llu", pid, static_cast<unsigned long long>(counter));
    if (n < 0 || static_cast<size_t>(n) >= sizeof(buf)) {
        throw std::runtime_error("broadcast_register: shm name overflow");
    }
    return std::string(buf);
}

// Strip the outer "<op_name> failed on child: " prefix that
// run_control_command prepends to every control failure, so the broadcast
// caller can surface the child-side message (`register hash=sha256:...
// chip=<id>: <reason>`) directly under its own one-line Worker.register prefix.
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

void WorkerManager::control_prepare(int worker_id, const uint8_t *digest) {
    auto *wt = get_worker(WorkerType::NEXT_LEVEL, worker_id);
    if (wt == nullptr) {
        throw std::runtime_error("control_prepare: invalid worker_id " + std::to_string(worker_id));
    }
    wt->control_prepare(digest);
}

void WorkerManager::control_alloc_domain(int worker_id, const char *request_shm_name, const char *reply_shm_name) {
    auto *wt = get_worker(WorkerType::NEXT_LEVEL, worker_id);
    if (wt == nullptr) {
        throw std::runtime_error("control_alloc_domain: invalid worker_id " + std::to_string(worker_id));
    }
    wt->control_alloc_domain(request_shm_name, reply_shm_name);
}

void WorkerManager::control_release_domain(int worker_id, const char *request_shm_name) {
    auto *wt = get_worker(WorkerType::NEXT_LEVEL, worker_id);
    if (wt == nullptr) {
        throw std::runtime_error("control_release_domain: invalid worker_id " + std::to_string(worker_id));
    }
    wt->control_release_domain(request_shm_name);
}

void WorkerManager::control_comm_init(int worker_id, const char *request_shm_name) {
    auto *wt = get_worker(WorkerType::NEXT_LEVEL, worker_id);
    if (wt == nullptr) {
        throw std::runtime_error("control_comm_init: invalid worker_id " + std::to_string(worker_id));
    }
    wt->control_comm_init(request_shm_name);
}

ControlResult WorkerManager::control_digest_only(
    WorkerType type, int worker_id, uint64_t sub_cmd, const uint8_t *digest, double timeout_s
) {
    auto &threads = (type == WorkerType::NEXT_LEVEL) ? next_level_threads_ : sub_threads_;
    const char *type_name = (type == WorkerType::NEXT_LEVEL) ? "NEXT_LEVEL" : "SUB";
    ControlResult result{type_name, static_cast<int32_t>(worker_id), false, ""};
    if (worker_id < 0 || static_cast<size_t>(worker_id) >= threads.size()) {
        result.error_message = "invalid worker_id " + std::to_string(worker_id);
        return result;
    }
    try {
        threads[static_cast<size_t>(worker_id)]->control_generic(sub_cmd, nullptr, timeout_s, digest);
        result.ok = true;
    } catch (const std::exception &e) {
        result.error_message = strip_control_prefix(e.what(), "control_generic");
    }
    return result;
}

std::vector<ControlResult>
WorkerManager::broadcast_register_all(const void *blob_ptr, size_t blob_size, const uint8_t *digest) {
    std::vector<ControlResult> results;
    results.reserve(next_level_threads_.size());
    for (size_t i = 0; i < next_level_threads_.size(); ++i) {
        results.push_back(ControlResult{"NEXT_LEVEL", static_cast<int32_t>(i), true, ""});
    }
    if (next_level_threads_.empty()) return results;

    std::string shm_name = make_shm_name();
    PosixShmHolder shm(shm_name, blob_size);
    std::memcpy(shm.addr(), blob_ptr, blob_size);

    // Fan out to every WorkerThread in parallel. Per-WorkerThread mailbox_mu_
    // is independent, so N control_register calls run concurrently — latency
    // is 1 × prepare_cost instead of N × prepare_cost.
    std::vector<std::thread> workers;
    workers.reserve(next_level_threads_.size());
    for (size_t i = 0; i < next_level_threads_.size(); ++i) {
        workers.emplace_back([this, i, digest, name = shm.name(), &results]() {
            try {
                next_level_threads_[i]->control_register(name.c_str(), digest);
            } catch (const std::exception &e) {
                results[i].ok = false;
                results[i].error_message = strip_control_prefix(e.what(), "control_register");
            }
        });
    }
    for (auto &t : workers)
        t.join();

    // shm is unlinked when `shm` goes out of scope. Children opened it by
    // name during control_register and have already closed their mappings
    // before publishing CONTROL_DONE — see python/simpler/worker.py.

    std::string hash = format_digest(digest);
    for (auto &result : results) {
        if (!result.ok && result.error_message.find("hash=") == std::string::npos) {
            result.error_message = "Worker.register(hash=" + hash + ") failed on next_level " +
                                   std::to_string(result.worker_index) + ": " + result.error_message;
        }
    }
    return results;
}

std::vector<std::string> WorkerManager::broadcast_unregister_all(const uint8_t *digest) {
    std::vector<std::string> errors;
    if (next_level_threads_.empty()) return errors;

    std::vector<std::string> per_worker(next_level_threads_.size());
    std::vector<std::thread> workers;
    workers.reserve(next_level_threads_.size());
    for (size_t i = 0; i < next_level_threads_.size(); ++i) {
        workers.emplace_back([this, i, digest, &per_worker]() {
            try {
                next_level_threads_[i]->control_unregister(digest);
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

std::vector<ControlResult> WorkerManager::broadcast_control_all(
    WorkerType type, uint64_t sub_cmd, const void *payload, size_t payload_size, const uint8_t *digest, double timeout_s
) {
    auto &threads = (type == WorkerType::NEXT_LEVEL) ? next_level_threads_ : sub_threads_;
    const char *type_name = (type == WorkerType::NEXT_LEVEL) ? "NEXT_LEVEL" : "SUB";

    std::vector<ControlResult> results;
    results.reserve(threads.size());
    for (size_t i = 0; i < threads.size(); ++i) {
        results.push_back(ControlResult{type_name, static_cast<int32_t>(i), true, ""});
    }
    if (threads.empty()) return results;

    std::unique_ptr<PosixShmHolder> shm;
    std::string shm_name;
    if (payload != nullptr || payload_size != 0) {
        if (payload == nullptr || payload_size == 0) {
            throw std::runtime_error("broadcast_control_all: payload pointer and size must both be set");
        }
        shm_name = make_shm_name();
        shm = std::make_unique<PosixShmHolder>(shm_name, payload_size);
        std::memcpy(shm->addr(), payload, payload_size);
    }

    std::vector<std::thread> workers;
    workers.reserve(threads.size());
    for (size_t i = 0; i < threads.size(); ++i) {
        workers.emplace_back([&, i]() {
            try {
                threads[i]->control_generic(sub_cmd, shm_name.empty() ? nullptr : shm_name.c_str(), timeout_s, digest);
            } catch (const std::exception &e) {
                results[i].ok = false;
                results[i].error_message = strip_control_prefix(e.what(), "control_generic");
            }
        });
    }
    for (auto &t : workers)
        t.join();

    return results;
}
