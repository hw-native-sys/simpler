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
 * @file host_log.cpp
 * @brief Implementation of Unified Host Logging System
 */

#include "host_log.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cinttypes>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <limits.h>
#include <mutex>
#include <new>
#include <pthread.h>
#include <semaphore.h>
#include <string>
#include <thread>

#include <unistd.h>

#if defined(__linux__)
#include <sys/syscall.h>
#endif

#include "common/host_span.h"
#include "common/log_clock.h"

using simpler::log::LogLevel;

namespace {

// Every queued record fits the portable atomic pipe-write floor. The writer
// therefore preserves one physical write per record when it falls back to a
// stderr pipe shared by several processes.
constexpr size_t kRecordCapacity = _POSIX_PIPE_BUF;
// About 2 MiB per logging process: large enough to absorb ordinary bursts,
// fixed enough that a permanently blocked destination cannot grow memory use.
constexpr size_t kQueueCapacity = 4096;
// Producers receive a fixed CPU budget for the lock-free MPSC position claim.
// Exhausting it is a counted drop, never an I/O wait or condition-variable sleep.
constexpr size_t kProducerClaimAttempts = 1024;
constexpr uint64_t kProducerStopFlag = UINT64_C(1) << 63;
static_assert(kQueueCapacity + 1 <= _POSIX_SEM_VALUE_MAX);

#if defined(SIMPLER_HOST_LOG_TEST_HOOKS)
std::atomic<void (*)(size_t)> g_after_queue_claim_hook{nullptr};
std::atomic<void (*)()> g_before_gap_wait_hook{nullptr};
#endif

// The two text-field widths live in common/host_span.h, where a producer that
// formats a field can size its buffer to the same number.
constexpr size_t kHostSpanNameCapacity = SIMPLER_HOST_SPAN_NAME_CAPACITY;
constexpr size_t kHostSpanAttributesCapacity = SIMPLER_HOST_SPAN_ATTRIBUTES_CAPACITY;
static_assert(kHostSpanNameCapacity + kHostSpanAttributesCapacity <= _POSIX_PIPE_BUF - 256);
static_assert(kRecordCapacity >= 2);

struct QueuedRecord {
    uint32_t size;
    char data[kRecordCapacity];
};

struct QueueSlot {
    std::atomic<size_t> sequence;
    QueuedRecord record;
};

std::string encode_host_span_field(const char *value, size_t capacity, bool attributes) {
    static constexpr char kHex[] = "0123456789ABCDEF";
    std::string encoded;
    encoded.reserve(capacity);
    bool truncated = false;
    size_t last_unit_size = 0;
    for (const unsigned char *p = reinterpret_cast<const unsigned char *>(value); *p != '\0'; ++p) {
        const unsigned char c = *p;
        const bool printable = c >= 0x20 && c <= 0x7E;
        const bool grammar_character = attributes && (c == ' ' || c == '=');
        const bool safe =
            printable && c != '%' && c != '[' && c != ']' && (grammar_character || (c != ' ' && c != '='));
        const size_t required = safe ? 1 : 3;
        if (encoded.size() + required > capacity) {
            truncated = true;
            break;
        }
        if (safe) {
            encoded.push_back(static_cast<char>(c));
        } else {
            encoded.push_back('%');
            encoded.push_back(kHex[c >> 4]);
            encoded.push_back(kHex[c & 0x0F]);
        }
        last_unit_size = required;
    }
    // A `%XX` escape is one indivisible unit, so a marker written over the tail
    // of a full field drops that whole unit rather than its last byte — which
    // would leave `%0A` as the undecodable `%0~`.
    if (truncated && capacity != 0) {
        if (encoded.size() == capacity) encoded.resize(encoded.size() - last_unit_size);
        encoded.push_back('~');
    }
    return encoded;
}

// Renders the timestamp/thread/level prefix, the caller's message, and an
// optional trailing newline into `buffer`, and returns the length of the whole
// record. A return value of `capacity` or more means `buffer` holds a truncated
// record and the caller must replace its tail with the truncation marker.
size_t format_record(
    char *buffer, size_t capacity, int64_t monotonic_ns, unsigned long tid, const char *level_tag, const char *func,
    const char *fmt, va_list args, bool append_newline
) {
    size_t length = 0;
    auto tail = [&]() -> char * {
        return length < capacity ? buffer + length : nullptr;
    };
    auto remaining = [&]() -> size_t {
        return length < capacity ? capacity - length : 0;
    };

    const int prefix = snprintf(
        tail(), remaining(), "[mono_ns=%lld][T0x%lx][%s] %s: ", static_cast<long long>(monotonic_ns), tid, level_tag,
        func
    );
    if (prefix < 0) {
        return 0;
    }
    length += static_cast<size_t>(prefix);

    va_list formatting_args;
    va_copy(formatting_args, args);
    const int body = vsnprintf(tail(), remaining(), fmt, formatting_args);
    va_end(formatting_args);
    if (body > 0) {
        length += static_cast<size_t>(body);
    }

    if (append_newline) {
        if (length + 1 < capacity) {
            buffer[length] = '\n';
            buffer[length + 1] = '\0';
        }
        length += 1;
    }
    return length;
}

bool write_fd(int fd, const char *record, size_t size) {
    size_t offset = 0;
    while (offset < size) {
        const ssize_t written = ::write(fd, record + offset, size - offset);
        if (written > 0) {
            offset += static_cast<size_t>(written);
        } else if (written < 0 && errno == EINTR) {
            continue;
        } else {
            return false;
        }
    }
    return true;
}

bool write_stderr(const char *record, size_t size) { return write_fd(STDERR_FILENO, record, size); }

uint64_t atomic_load_u64(const uint64_t *value) { return __atomic_load_n(value, __ATOMIC_ACQUIRE); }

void atomic_add_u64(uint64_t *value, uint64_t count) { __atomic_fetch_add(value, count, __ATOMIC_RELAXED); }

void atomic_sub_u64(uint64_t *value, uint64_t count) { __atomic_fetch_sub(value, count, __ATOMIC_RELEASE); }

void atomic_store_u64(uint64_t *value, uint64_t desired) { __atomic_store_n(value, desired, __ATOMIC_RELEASE); }

// Every drop is one increment of the total plus one of its reason. Keeping them
// in step is why no site touches dropped_record_count directly: a total that
// disagrees with the breakdown is worse than either alone, because it makes both
// unusable.
void count_drop(SimplerHostLogState *state, SimplerHostLogDropReason reason) {
    atomic_add_u64(&state->dropped_record_count, 1);
    atomic_add_u64(&state->dropped_by_reason[reason], 1);
}

long host_trace_tid() {
#if defined(__linux__) && defined(SYS_gettid)
    return static_cast<long>(syscall(SYS_gettid));
#else
    return static_cast<long>(getpid());
#endif
}

struct HostLogFileSink {
    HostLogFileSink();

    std::mutex mutex;
    int fd = -1;
    pid_t pid = -1;
    std::string directory;
};

HostLogFileSink &host_log_file_sink() {
    // Process-lifetime allocation avoids static-destruction order racing the
    // writer thread. The kernel closes the fd when the process exits.
    static auto *sink = new HostLogFileSink;
    return *sink;
}

void host_log_sink_before_fork() { host_log_file_sink().mutex.lock(); }

void host_log_sink_after_fork() { host_log_file_sink().mutex.unlock(); }

HostLogFileSink::HostLogFileSink() {
    // A child inherits only the thread that called fork(). Locking here makes
    // that thread the mutex owner at the fork boundary, so both the parent and
    // child handlers can release it without inheriting an owner that vanished.
    (void)pthread_atfork(host_log_sink_before_fork, host_log_sink_after_fork, host_log_sink_after_fork);
}

// Blocking I/O is confined to the process writer thread. A raw append fd makes
// completion and loss accounting exact: successful write(2) means the whole
// record reached the kernel, with no hidden stdio buffer left to fail later.
bool write_log_file(const char *directory, const char *record, size_t size) {
    HostLogFileSink &sink = host_log_file_sink();
    std::scoped_lock lock(sink.mutex);
    const pid_t pid = getpid();
    const bool inherited = sink.fd >= 0 && sink.pid != pid;
    const bool changed_directory = sink.fd >= 0 && sink.directory != directory;
    if (inherited) {
        (void)::close(sink.fd);
        sink.fd = -1;
        sink.pid = -1;
        sink.directory.clear();
    } else if (changed_directory) {
        (void)::close(sink.fd);
        sink.fd = -1;
        sink.pid = -1;
        sink.directory.clear();
    }
    if (sink.fd < 0) {
        char path[PATH_MAX];
        const int length = std::snprintf(path, sizeof(path), "%s/host.%d.log", directory, pid);
        if (length <= 0 || static_cast<size_t>(length) >= sizeof(path)) return false;
        int flags = O_WRONLY | O_CREAT | O_APPEND;
#ifdef O_CLOEXEC
        flags |= O_CLOEXEC;
#endif
        sink.fd = ::open(path, flags, 0666);
        if (sink.fd < 0) return false;
        sink.pid = pid;
        sink.directory = directory;
    }
    if (write_fd(sink.fd, record, size)) return true;
    (void)::close(sink.fd);
    sink.fd = -1;
    sink.pid = -1;
    sink.directory.clear();
    return false;
}

const char *bound_log_directory(const SimplerHostLogState *state) {
    if (__atomic_load_n(&state->log_directory_bound, __ATOMIC_ACQUIRE) != 1) return nullptr;
    return state->log_directory;
}

// The one place a destination is chosen, and it chooses exactly one: a bound
// directory is the destination, not a preference. A record the file cannot take
// is dropped and counted rather than relocated to stderr, so "the log is
// complete" and "dropped_record_count is zero" stay the same statement. The
// alternative loses that: a bound directory that cannot be opened would send
// every record to stderr while the counter still read zero, which is a silent
// failure wearing a healthy counter.
bool write_record_now(const SimplerHostLogState *state, const char *record, size_t size) {
    const char *directory = bound_log_directory(state);
    if (directory != nullptr) return write_log_file(directory, record, size);
    return write_stderr(record, size);
}

struct HostLogAsyncSink {
    explicit HostLogAsyncSink(SimplerHostLogState *shared_state) :
        state(shared_state),
        pid(getpid()) {
        for (size_t index = 0; index < kQueueCapacity; ++index) {
            queue[index].sequence.store(index, std::memory_order_relaxed);
        }
        // macOS deliberately does not implement unnamed semaphores. A named
        // semaphore works on both supported host OSes; unlink it immediately so
        // the kernel object has this process's handles as its only lifetime.
        char name[32];
        const int length = snprintf(
            name, sizeof(name), "/sl-%x-%llx", static_cast<unsigned int>(pid),
            static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(this))
        );
        if (length > 0 && static_cast<size_t>(length) < sizeof(name)) {
            ready = sem_open(name, O_CREAT | O_EXCL, 0600, 0);
            if (ready != SEM_FAILED && sem_unlink(name) != 0) {
                (void)sem_close(ready);
                ready = SEM_FAILED;
            }
        }
    }

    ~HostLogAsyncSink() {
        if (ready != SEM_FAILED) (void)sem_close(ready);
    }

    bool start() {
        if (ready == SEM_FAILED) return false;
        try {
            writer = std::thread([this] {
                run();
            });
            return true;
        } catch (...) {
            return false;
        }
    }

    bool stop_when_empty(uint32_t timeout_ms) {
        if (!wait_until_empty(timeout_ms)) return false;

        stopping.store(true, std::memory_order_release);
        (void)sem_post(ready);
        if (writer.joinable()) writer.join();
        return true;
    }

    bool wait_until_empty(uint32_t timeout_ms) {
        std::unique_lock<std::mutex> lock(completion_mutex);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        for (;;) {
            if ((atomic_load_u64(&state->sink_producer_state) & ~kProducerStopFlag) == 0 &&
                atomic_load_u64(&state->pending_record_count) == 0) {
                return true;
            }
            const auto now = std::chrono::steady_clock::now();
            if (now >= deadline) return false;
            completion_cv.wait_until(lock, std::min(deadline, now + std::chrono::milliseconds(1)));
        }
    }

    SimplerHostLogState *state;
    pid_t pid;

    static int enqueue(void *context, SimplerHostLogState *state, const char *record, uint32_t size) {
        auto *sink = static_cast<HostLogAsyncSink *>(context);
        if (sink == nullptr) {
            atomic_sub_u64(&state->sink_producer_state, 1);
            count_drop(state, SIMPLER_HOST_LOG_DROP_NOT_ADMITTED);
            return 0;
        }
        if (sink->state != state || sink->pid != getpid() || size > kRecordCapacity) {
            sink->producer_done();
            count_drop(state, SIMPLER_HOST_LOG_DROP_NOT_ADMITTED);
            return 0;
        }

        size_t position = sink->enqueue_position.load(std::memory_order_relaxed);
        QueueSlot *slot = nullptr;
        for (size_t attempt = 0; attempt < kProducerClaimAttempts; ++attempt) {
            slot = &sink->queue[position % kQueueCapacity];
            const size_t sequence = slot->sequence.load(std::memory_order_acquire);
            const auto difference = static_cast<intptr_t>(sequence) - static_cast<intptr_t>(position);
            if (difference == 0) {
                if (sink->enqueue_position.compare_exchange_weak(
                        position, position + 1, std::memory_order_relaxed, std::memory_order_relaxed
                    )) {
                    break;
                }
            } else if (difference < 0) {
                sink->producer_done();
                count_drop(state, SIMPLER_HOST_LOG_DROP_QUEUE_FULL);
                return 0;  // The consumer has not freed this lap: queue full.
            } else {
                position = sink->enqueue_position.load(std::memory_order_relaxed);
            }
            slot = nullptr;
        }
        if (slot == nullptr) {
            sink->producer_done();
            count_drop(state, SIMPLER_HOST_LOG_DROP_CLAIM_EXHAUSTED);
            return 0;
        }

#if defined(SIMPLER_HOST_LOG_TEST_HOOKS)
        if (auto hook = g_after_queue_claim_hook.load(std::memory_order_acquire); hook != nullptr) hook(position);
#endif

        slot->record.size = size;
        std::memcpy(slot->record.data, record, size);
        atomic_add_u64(&state->pending_record_count, 1);
        slot->sequence.store(position + 1, std::memory_order_release);
        sink->queue_size.fetch_add(1, std::memory_order_release);
        (void)sem_post(sink->ready);
        sink->producer_done();
        return 1;
    }

private:
    void producer_done() {
        // Producers never take the drain waiter's mutex. A waiter polls the
        // atomic producer count at a bounded interval, so it cannot miss this
        // transition permanently even though no condition-variable signal is
        // needed on this hot path.
        atomic_sub_u64(&state->sink_producer_state, 1);
    }

    bool pop(QueuedRecord *record) {
        if (queue_size.load(std::memory_order_acquire) == 0) return false;
        QueueSlot &slot = queue[dequeue_position % kQueueCapacity];
        if (slot.sequence.load(std::memory_order_acquire) != dequeue_position + 1) return false;
        *record = slot.record;
        slot.sequence.store(dequeue_position + kQueueCapacity, std::memory_order_release);
        ++dequeue_position;
        queue_size.fetch_sub(1, std::memory_order_release);
        return true;
    }

    void run() {
        size_t ready_tokens = 0;
        for (;;) {
            if (ready_tokens == 0) {
                int result;
                do {
                    result = sem_wait(ready);
                } while (result != 0 && errno == EINTR);
                if (result != 0) return;
                if (stopping.load(std::memory_order_acquire) && queue_size.load(std::memory_order_acquire) == 0) return;
                ++ready_tokens;
            }

            QueuedRecord record;
            // Producers can publish adjacent MPSC positions out of order. A
            // later token may wake us before the next position is visible, so
            // retain every later token and sleep for the earlier producer's
            // token instead of burning a CPU in an unbounded yield loop.
            if (!pop(&record)) {
#if defined(SIMPLER_HOST_LOG_TEST_HOOKS)
                if (auto hook = g_before_gap_wait_hook.load(std::memory_order_acquire); hook != nullptr) hook();
#endif
                int result;
                do {
                    result = sem_wait(ready);
                } while (result != 0 && errno == EINTR);
                if (result != 0) return;
                ++ready_tokens;
                continue;
            }
            --ready_tokens;

            if (!write_record_now(state, record.data, record.size)) {
                count_drop(state, SIMPLER_HOST_LOG_DROP_OUTPUT_FAILED);
            }
            atomic_sub_u64(&state->pending_record_count, 1);
            completion_cv.notify_all();
        }
    }

    std::array<QueueSlot, kQueueCapacity> queue;
    std::atomic<size_t> queue_size{0};
    std::atomic<size_t> enqueue_position{0};
    size_t dequeue_position = 0;
    sem_t *ready = SEM_FAILED;
    std::atomic<bool> stopping{false};
    std::mutex completion_mutex;
    std::condition_variable completion_cv;
    std::thread writer;
};

// A private logger stays silent until its owner seeds this state or its loader
// binds the process-owned state. Missing binding is therefore observable as an
// absent module stream rather than output filtered at the wrong threshold.
SimplerHostLogState g_module_log_state{
    static_cast<int32_t>(LogLevel::NUL), 0, {}, 0, 0, nullptr, nullptr, 0, 0, 0, {}, 0,
};

int32_t atomic_load_i32(const int32_t *value) { return __atomic_load_n(value, __ATOMIC_ACQUIRE); }

void atomic_store_i32(int32_t *value, int32_t desired) { __atomic_store_n(value, desired, __ATOMIC_RELEASE); }

bool atomic_compare_exchange_i32(int32_t *value, int32_t *expected, int32_t desired) {
    return __atomic_compare_exchange_n(value, expected, desired, false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
}

void *atomic_load_pointer(void *const *value) { return __atomic_load_n(value, __ATOMIC_ACQUIRE); }

void atomic_store_pointer(void **value, void *desired) { __atomic_store_n(value, desired, __ATOMIC_RELEASE); }

SimplerHostLogEnqueueFn atomic_load_enqueue(SimplerHostLogEnqueueFn const *value) {
    return __atomic_load_n(value, __ATOMIC_ACQUIRE);
}

void atomic_store_enqueue(SimplerHostLogEnqueueFn *value, SimplerHostLogEnqueueFn desired) {
    __atomic_store_n(value, desired, __ATOMIC_RELEASE);
}

bool acquire_sink_producer(SimplerHostLogState *state) {
    // Admission and stop are ordered by one atomic RMW. If stop won first, undo
    // our transient producer reference and reject the record. If admission won
    // first, prepare_to_fork() observes the reference and keeps sink_context
    // alive until this producer returns. Unlike a CAS retry loop, this gives
    // every producer a fixed admission cost under contention.
    const uint64_t previous = __atomic_fetch_add(&state->sink_producer_state, 1, __ATOMIC_ACQ_REL);
    if ((previous & kProducerStopFlag) != 0) {
        atomic_sub_u64(&state->sink_producer_state, 1);
        return false;
    }
    return true;
}

void release_sink_producer(SimplerHostLogState *state) { atomic_sub_u64(&state->sink_producer_state, 1); }

}  // namespace

HostLogger &HostLogger::get_instance() {
    static HostLogger instance;
    return instance;
}

HostLogger::HostLogger() :
    state_(&g_module_log_state) {}

HostLogger::~HostLogger() {
    // Normal executable/module teardown gets a bounded chance to preserve
    // accepted records. os._exit() call sites flush explicitly because static
    // destructors do not run there.
    (void)prepare_to_fork(100);
}

int HostLogger::bind_state(SimplerHostLogState *state) {
    if (state == nullptr || !simpler::log::is_valid_level(atomic_load_i32(&state->threshold))) {
        return -1;
    }
    state_owner_.store(false, std::memory_order_release);
    state_.store(state, std::memory_order_release);
    return 0;
}

int HostLogger::adopt_state(SimplerHostLogState *state) {
    if (state == nullptr || !simpler::log::is_valid_level(atomic_load_i32(&state->threshold))) {
        return -1;
    }
    state_.store(state, std::memory_order_release);
    state_owner_.store(true, std::memory_order_release);
    return 0;
}

SimplerHostLogState *HostLogger::state() const { return state_.load(std::memory_order_acquire); }

void HostLogger::set_level(LogLevel level, bool defer_writer) {
    atomic_store_i32(&state()->threshold, static_cast<int32_t>(level));
    if (!defer_writer) (void)start_writer();
}

bool HostLogger::start_writer() {
    SimplerHostLogState *shared = state();
    const int32_t pid = static_cast<int32_t>(getpid());
    int32_t owner = atomic_load_i32(&shared->sink_owner_pid);
    const auto enqueue = atomic_load_enqueue(&shared->sink_enqueue);
    void *context = atomic_load_pointer(&shared->sink_context);
    if (owner == pid && enqueue != nullptr && context != nullptr) {
        return true;
    }

    // A bound DSO may submit to an already-published process sink, but only the
    // module that owns the process state may create that sink. Otherwise the
    // callback and writer thread could outlive a transient dlopen() consumer.
    if (!state_owner_.load(std::memory_order_acquire)) return false;

    // Another caller is already constructing this process's sink. Treat that
    // as a failed duplicate start instead of letting -pid claim itself.
    if (owner == -pid) return false;
    if (!atomic_compare_exchange_i32(&shared->sink_owner_pid, &owner, -pid)) return false;
    __atomic_fetch_or(&shared->sink_producer_state, kProducerStopFlag, __ATOMIC_ACQ_REL);

    const int32_t process_pid = atomic_load_i32(&shared->sink_process_pid);
    if (process_pid != 0 && process_pid != pid) {
        // A fork child inherits the parent's counters and callback addresses,
        // but none of its threads or accepted records. Its new sink starts clean,
        // even when the parent had already quiesced to owner=0 before fork.
        atomic_store_u64(&shared->dropped_record_count, 0);
        atomic_store_u64(&shared->pending_record_count, 0);
        for (int reason = 0; reason < SIMPLER_HOST_LOG_DROP_REASON_COUNT; ++reason)
            atomic_store_u64(&shared->dropped_by_reason[reason], 0);
        atomic_store_u64(&shared->reported_drop_count, 0);
        atomic_store_u64(&shared->sink_producer_state, kProducerStopFlag);
    }
    atomic_store_i32(&shared->sink_process_pid, pid);

    auto *candidate = new (std::nothrow) HostLogAsyncSink(shared);
    if (candidate == nullptr || !candidate->start()) {
        delete candidate;
        atomic_store_enqueue(&shared->sink_enqueue, nullptr);
        atomic_store_pointer(&shared->sink_context, nullptr);
        int32_t claim = -pid;
        (void)atomic_compare_exchange_i32(&shared->sink_owner_pid, &claim, 0);
        __atomic_fetch_and(&shared->sink_producer_state, ~kProducerStopFlag, __ATOMIC_RELEASE);
        return false;
    }

    // The active sink normally has process lifetime. prepare_to_fork() is the
    // explicit quiescent boundary that can join and reclaim it before a later
    // hierarchical Worker forks in this process.
    sink_.store(candidate, std::memory_order_release);
    atomic_store_pointer(&shared->sink_context, candidate);
    atomic_store_enqueue(&shared->sink_enqueue, &HostLogAsyncSink::enqueue);
    atomic_store_i32(&shared->sink_owner_pid, pid);
    __atomic_fetch_and(&shared->sink_producer_state, ~kProducerStopFlag, __ATOMIC_RELEASE);
    return true;
}

bool HostLogger::prepare_to_fork(uint32_t timeout_ms) {
    SimplerHostLogState *shared = state();
    const int32_t pid = static_cast<int32_t>(getpid());
    int32_t owner = atomic_load_i32(&shared->sink_owner_pid);
    if (owner == 0) return true;
    if (owner == -pid) return false;
    if (owner != pid) return true;

    auto *sink = static_cast<HostLogAsyncSink *>(sink_.load(std::memory_order_acquire));
    if (sink == nullptr || sink->state != shared || sink->pid != getpid()) return false;

    if (!atomic_compare_exchange_i32(&shared->sink_owner_pid, &owner, -pid)) return false;
    __atomic_fetch_or(&shared->sink_producer_state, kProducerStopFlag, __ATOMIC_ACQ_REL);
    if (!sink->stop_when_empty(timeout_ms)) {
        __atomic_fetch_and(&shared->sink_producer_state, ~kProducerStopFlag, __ATOMIC_RELEASE);
        atomic_store_i32(&shared->sink_owner_pid, pid);
        return false;
    }

    atomic_store_enqueue(&shared->sink_enqueue, nullptr);
    atomic_store_pointer(&shared->sink_context, nullptr);
    sink_.store(nullptr, std::memory_order_release);
    atomic_store_i32(&shared->sink_owner_pid, 0);
    delete sink;
    // Last, once no queue remains to admit anyone into: the quiesced state is
    // the same "no sink" state a process starts in, and emit()'s synchronous
    // fallback is gated on the flag being clear. Leaving it set here would make
    // every record between this call and the next start_writer() a silent drop —
    // exactly the initialization window the fallback exists to cover. A producer
    // that observes the pre-clear state still drops, as it must; one that
    // observes the post-clear state re-reads sink_enqueue and finds it null, so
    // it takes the fallback rather than a freed sink.
    __atomic_fetch_and(&shared->sink_producer_state, ~kProducerStopFlag, __ATOMIC_RELEASE);
    // Last, with the fallback usable again, so the summary reaches the same
    // destination as everything it is accounting for.
    report_drops_if_grown();
    return true;
}

// The drop counters live in process memory and die with the process, so a reader
// holding only host.<pid>.log has no other way to learn that records are missing.
// This is the one place that can still write: every path that stops logging —
// each fork boundary and ~HostLogger — passes through prepare_to_fork().
//
// Reported as a growth rather than a total, so a process that quiesces several
// times does not restate the same losses at every boundary. ERROR, not WARN:
// losing diagnostics is itself the failure, and a run whose threshold is ERROR
// is exactly the one whose dropped records mattered most.
void HostLogger::report_drops_if_grown() {
    SimplerHostLogState *shared = state();
    const uint64_t total = atomic_load_u64(&shared->dropped_record_count);
    const uint64_t reported = atomic_load_u64(&shared->reported_drop_count);
    if (total <= reported) return;
    atomic_store_u64(&shared->reported_drop_count, total);
    if (!is_enabled(LogLevel::ERROR)) return;
    unsigned long long by_reason[SIMPLER_HOST_LOG_DROP_REASON_COUNT];
    for (int reason = 0; reason < SIMPLER_HOST_LOG_DROP_REASON_COUNT; ++reason) {
        by_reason[reason] = static_cast<unsigned long long>(atomic_load_u64(&shared->dropped_by_reason[reason]));
    }
    (void)emit_ungated(
        level_name(LogLevel::ERROR), "host_log_drops",
        "[HOSTLOG_DROPS] v=1 pid=%d new=%llu total=%llu queue_full=%llu claim_exhausted=%llu output_failed=%llu "
        "not_admitted=%llu\n",
        static_cast<int>(getpid()), static_cast<unsigned long long>(total - reported),
        static_cast<unsigned long long>(total), by_reason[SIMPLER_HOST_LOG_DROP_QUEUE_FULL],
        by_reason[SIMPLER_HOST_LOG_DROP_CLAIM_EXHAUSTED], by_reason[SIMPLER_HOST_LOG_DROP_OUTPUT_FAILED],
        by_reason[SIMPLER_HOST_LOG_DROP_NOT_ADMITTED]
    );
}

bool HostLogger::flush(uint32_t timeout_ms) {
    SimplerHostLogState *shared = state();
    if ((atomic_load_u64(&shared->sink_producer_state) & ~kProducerStopFlag) == 0 &&
        atomic_load_u64(&shared->pending_record_count) == 0) {
        return true;
    }
    auto *sink = static_cast<HostLogAsyncSink *>(sink_.load(std::memory_order_acquire));
    if (sink != nullptr && sink->state == shared && sink->pid == getpid()) return sink->wait_until_empty(timeout_ms);

    // A bound runtime DSO borrows the process writer. Shared counters reach zero
    // after accepted records are processed; write failures count as drops.
    // No owner-local sink pointer is needed.
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    do {
        if (atomic_load_i32(&shared->sink_owner_pid) != getpid() ||
            atomic_load_i32(&shared->sink_process_pid) != getpid())
            return false;
        if ((atomic_load_u64(&shared->sink_producer_state) & ~kProducerStopFlag) == 0 &&
            atomic_load_u64(&shared->pending_record_count) == 0)
            return true;
    } while (std::chrono::steady_clock::now() < deadline);
    return false;
}

uint64_t HostLogger::dropped_records() const { return atomic_load_u64(&state()->dropped_record_count); }

uint64_t HostLogger::pending_records() const { return atomic_load_u64(&state()->pending_record_count); }

uint64_t HostLogger::dropped_records(SimplerHostLogDropReason reason) const {
    if (reason < 0 || reason >= SIMPLER_HOST_LOG_DROP_REASON_COUNT) return 0;
    return atomic_load_u64(&state()->dropped_by_reason[reason]);
}

int HostLogger::level() const { return atomic_load_i32(&state()->threshold); }

int HostLogger::cann_level() const { return simpler::log::to_cann_log_level(static_cast<LogLevel>(level())); }

void HostLogger::configure_cann_log_level(int (*set_level)(int, int, int)) const {
    if (std::getenv("ASCEND_GLOBAL_LOG_LEVEL") == nullptr) {
        set_level(-1, cann_level(), 0);
    }
}

bool HostLogger::is_enabled(LogLevel level) const {
    // threshold is the floor: messages with severity >= floor are kept.
    const int current_level = this->level();
    return static_cast<int>(level) >= current_level && current_level != static_cast<int>(LogLevel::NUL);
}

const char *HostLogger::level_name(LogLevel level) const {
    switch (level) {
    case LogLevel::DEBUG:
        return "DEBUG";
    case LogLevel::INFO:
        return "INFO";
    case LogLevel::TIMING:
        return "TIMING";
    case LogLevel::WARN:
        return "WARN";
    case LogLevel::ERROR:
        return "ERROR";
    case LogLevel::NUL:
        return "NUL";
    }
    return "?";
}

bool HostLogger::emit(const char *level_tag, const char *func, const char *fmt, va_list args) {
    const int64_t monotonic_ns = simpler::log::monotonic_now_ns();
    auto tid = static_cast<unsigned long>(reinterpret_cast<uintptr_t>(pthread_self()));

    const bool append_newline = fmt[0] != '\0' && fmt[strlen(fmt) - 1] != '\n';

    char record[kRecordCapacity];
    const size_t length =
        format_record(record, sizeof(record), monotonic_ns, tid, level_tag, func, fmt, args, append_newline);
    size_t size = length;
    if (length >= sizeof(record)) {
        // A bounded record is part of the producer non-blocking contract. Keep
        // a visible truncation marker and a complete physical log line.
        record[sizeof(record) - 2] = '~';
        record[sizeof(record) - 1] = '\n';
        size = sizeof(record);
    }

    SimplerHostLogState *shared = state();
    const int32_t pid = static_cast<int32_t>(getpid());
    const int32_t owner = atomic_load_i32(&shared->sink_owner_pid);
    const auto current_enqueue = atomic_load_enqueue(&shared->sink_enqueue);
    void *current_context = atomic_load_pointer(&shared->sink_context);
    const uint64_t producer_state = atomic_load_u64(&shared->sink_producer_state);
    if (owner == 0 && current_enqueue == nullptr && current_context == nullptr &&
        (producer_state & kProducerStopFlag) == 0) {
        // Hierarchical processes deliberately have no writer until their final
        // local fork. Preserve initialization records with the synchronous
        // path; steady-state producers below remain bounded and do no output I/O.
        if (write_record_now(shared, record, size)) return true;
        count_drop(shared, SIMPLER_HOST_LOG_DROP_OUTPUT_FAILED);
        return false;
    }
    if (!acquire_sink_producer(shared)) {
        count_drop(shared, SIMPLER_HOST_LOG_DROP_NOT_ADMITTED);
        return false;
    }
    if (atomic_load_i32(&shared->sink_owner_pid) != pid) {
        release_sink_producer(shared);
        count_drop(shared, SIMPLER_HOST_LOG_DROP_NOT_ADMITTED);
        return false;
    }
    const auto enqueue = atomic_load_enqueue(&shared->sink_enqueue);
    void *context = atomic_load_pointer(&shared->sink_context);
    if (enqueue == nullptr || context == nullptr) {
        release_sink_producer(shared);
        count_drop(shared, SIMPLER_HOST_LOG_DROP_NOT_ADMITTED);
        return false;
    }
    // A rejection is attributed by the sink, which is the only side that knows
    // whether the queue was full or the claim budget ran out.
    return enqueue(context, shared, record, size) != 0;
}

bool HostLogger::emit_ungated(const char *level_tag, const char *func, const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    const bool written = emit(level_tag, func, fmt, args);
    va_end(args);
    return written;
}

void HostLogger::vlog(LogLevel level, const char *func, const char *fmt, va_list args) {
    if (!is_enabled(level)) {
        return;
    }
    (void)emit(level_name(level), func, fmt, args);
}

void HostLogger::log(LogLevel level, const char *func, const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vlog(level, func, fmt, args);
    va_end(args);
}

void HostLogger::set_log_directory(const char *path) {
    if (path == nullptr || path[0] == '\0') return;
    SimplerHostLogState *shared = state();
    if (shared == nullptr) return;
    // Claim 0 -> -1, fill, publish -1 -> 1, mirroring the anchor claim above. A
    // reader accepts only 1, so it never observes a half-written path; and a
    // later caller with a different path is refused rather than moving a file
    // some thread may already hold open.
    int32_t unclaimed = 0;
    if (!atomic_compare_exchange_i32(&shared->log_directory_bound, &unclaimed, -1)) return;
    (void)std::snprintf(shared->log_directory, sizeof(shared->log_directory), "%s", path);
    int32_t claim = -1;
    (void)atomic_compare_exchange_i32(&shared->log_directory_bound, &claim, 1);
}

const char *HostLogger::log_directory() const {
    SimplerHostLogState *shared = state();
    if (shared == nullptr || atomic_load_i32(&shared->log_directory_bound) != 1) return nullptr;
    return shared->log_directory;
}

void HostLogger::log_host_span(const SimplerHostSpan *span) {
    if (!is_enabled(LogLevel::TIMING)) return;
    if (span == nullptr || span->name == nullptr) {
        return;
    }
    const std::string name = encode_host_span_field(span->name, kHostSpanNameCapacity, false);
    const std::string attributes =
        encode_host_span_field(span->attributes == nullptr ? "" : span->attributes, kHostSpanAttributesCapacity, true);

    // One record grammar, in one place. Where it lands is the logger's business,
    // not this emitter's.
    char record[kRecordCapacity];
    (void)std::snprintf(
        record, sizeof(record),
        "[STRACE] v=1 pid=%d tid=%ld inv=%" PRIu64 " hid=%" PRIx64 " depth=%d name=%s ts=%" PRId64 " dur=%" PRId64
        " %s",
        static_cast<int>(getpid()), host_trace_tid(), span->invocation_id, span->callable_hash, span->depth,
        name.c_str(), span->timestamp_ns, span->duration_ns, attributes.c_str()
    );

    log(LogLevel::TIMING, "emit_host_span", "%s", record);
}

extern "C" __attribute__((visibility("default"))) int simpler_host_log_bind_state(SimplerHostLogState *state) {
    return HostLogger::get_instance().bind_state(state);
}

#if defined(SIMPLER_HOST_LOG_TEST_HOOKS)
extern "C" void simpler_host_log_set_queue_hooks_for_test(void (*after_claim)(size_t), void (*before_gap_wait)()) {
    g_after_queue_claim_hook.store(after_claim, std::memory_order_release);
    g_before_gap_wait_hook.store(before_gap_wait, std::memory_order_release);
}
#endif
