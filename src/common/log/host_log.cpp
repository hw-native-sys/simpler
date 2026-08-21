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

#include <cerrno>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits.h>
#include <pthread.h>
#include <string>
#include <vector>

#include <unistd.h>

#if defined(__linux__)
#include <sys/syscall.h>
#endif

#include "common/host_span.h"
#include "common/log_clock.h"

using simpler::log::LogLevel;

namespace {

// Every STRACE marker renders well inside this allocation bound, so the heap
// fallback below stays off the path a traced run pays per span. This capacity
// is not an atomic-write guarantee.
constexpr size_t kRecordStackCapacity = 2048;

// POSIX guarantees atomic pipe writes up to _POSIX_PIPE_BUF (512 bytes). A
// conservative bound for the logger prefix, fixed-width STRACE fields, and
// newline is 256 bytes, leaving the other half for the encoded text fields.
constexpr size_t kHostSpanNameCapacity = 64;
constexpr size_t kHostSpanAttributesCapacity = 192;
static_assert(kHostSpanNameCapacity + kHostSpanAttributesCapacity <= _POSIX_PIPE_BUF - 256);

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
// record. A return value of `capacity` or more means `buffer` holds only a
// truncated prefix and the caller must re-render into that many bytes.
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

bool write_stderr(const char *record, size_t size) {
    size_t offset = 0;
    while (offset < size) {
        const ssize_t written = ::write(STDERR_FILENO, record + offset, size - offset);
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

long host_trace_tid() {
#if defined(__linux__) && defined(SYS_gettid)
    return static_cast<long>(syscall(SYS_gettid));
#else
    return static_cast<long>(getpid());
#endif
}

struct HostSpanFileSink {
    std::mutex mutex;
    FILE *stream = nullptr;
    pid_t pid = -1;
    std::string directory;
};

HostSpanFileSink &host_span_file_sink() {
    static HostSpanFileSink sink;
    return sink;
}

bool write_host_span_file(
    const char *directory, const SimplerHostSpan &span, long tid, const std::string &name, const std::string &attributes
) {
    HostSpanFileSink &sink = host_span_file_sink();
    std::scoped_lock lock(sink.mutex);
    const pid_t pid = getpid();
    const bool inherited = sink.stream != nullptr && sink.pid != pid;
    const bool changed_directory = sink.stream != nullptr && sink.directory != directory;
    if (inherited) {
        sink.stream = nullptr;
        sink.pid = -1;
        sink.directory.clear();
    } else if (changed_directory) {
        std::fclose(sink.stream);
        sink.stream = nullptr;
        sink.pid = -1;
        sink.directory.clear();
    }
    if (sink.stream == nullptr) {
        char path[PATH_MAX];
        const int length = std::snprintf(path, sizeof(path), "%s/host-strace.%d.log", directory, pid);
        if (length <= 0 || static_cast<size_t>(length) >= sizeof(path)) return false;
        sink.stream = std::fopen(path, "a");
        if (sink.stream == nullptr) return false;
        std::setvbuf(sink.stream, nullptr, _IOFBF, 1U << 20U);
        sink.pid = pid;
        sink.directory = directory;
    }
    const int written = std::fprintf(
        sink.stream,
        "[STRACE] v=1 pid=%d tid=%ld inv=%" PRIu64 " hid=%" PRIx64 " depth=%d name=%s ts=%" PRId64 " dur=%" PRId64
        " %s\n",
        static_cast<int>(pid), tid, span.invocation_id, span.callable_hash, span.depth, name.c_str(), span.timestamp_ns,
        span.duration_ns, attributes.c_str()
    );
    if (written < 0) return false;
    return span.depth != 0 || std::fflush(sink.stream) == 0;
}

}  // namespace

namespace {

// A private logger stays silent until its owner seeds this state or its loader
// binds the process-owned state. Missing binding is therefore observable as an
// absent module stream rather than output filtered at the wrong threshold.
SimplerHostLogState g_module_log_state{
    SIMPLER_HOST_LOG_STATE_ABI_VERSION,
    sizeof(SimplerHostLogState),
    static_cast<int32_t>(LogLevel::NUL),
    0,
};

int32_t atomic_load_i32(const int32_t *value) { return __atomic_load_n(value, __ATOMIC_ACQUIRE); }

void atomic_store_i32(int32_t *value, int32_t desired) { __atomic_store_n(value, desired, __ATOMIC_RELEASE); }

bool atomic_compare_exchange_i32(int32_t *value, int32_t *expected, int32_t desired) {
    return __atomic_compare_exchange_n(value, expected, desired, false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
}

}  // namespace

HostLogger &HostLogger::get_instance() {
    static HostLogger instance;
    return instance;
}

HostLogger::HostLogger() :
    state_(&g_module_log_state) {}

int HostLogger::bind_state(SimplerHostLogState *state) {
    if (state == nullptr || state->abi_version != SIMPLER_HOST_LOG_STATE_ABI_VERSION ||
        state->struct_size < sizeof(SimplerHostLogState) ||
        !simpler::log::is_valid_level(atomic_load_i32(&state->threshold))) {
        return -1;
    }
    state_.store(state, std::memory_order_release);
    return 0;
}

SimplerHostLogState *HostLogger::state() const { return state_.load(std::memory_order_acquire); }

void HostLogger::set_level(LogLevel level) {
    atomic_store_i32(&state()->threshold, static_cast<int32_t>(level));
    emit_clock_anchor_if_needed();
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

    // One write per record avoids thread interleaving under mutex_. On a shared
    // pipe, only records no larger than that pipe's PIPE_BUF are indivisible
    // across forked writers. Machine-readable host spans are separately
    // budgeted to the portable _POSIX_PIPE_BUF floor (512 bytes); longer human
    // log records use this same best-effort write path without that promise.
    char stack_buffer[kRecordStackCapacity];
    const size_t length = format_record(
        stack_buffer, sizeof(stack_buffer), monotonic_ns, tid, level_tag, func, fmt, args, append_newline
    );
    if (length < sizeof(stack_buffer)) {
        std::scoped_lock lock(mutex_);
        return write_stderr(stack_buffer, length);
    }

    std::vector<char> heap_buffer(length + 1);
    const size_t heap_length = format_record(
        heap_buffer.data(), heap_buffer.size(), monotonic_ns, tid, level_tag, func, fmt, args, append_newline
    );
    std::scoped_lock lock(mutex_);
    return write_stderr(heap_buffer.data(), heap_length < heap_buffer.size() ? heap_length : length);
}

bool HostLogger::emit_ungated(const char *level_tag, const char *func, const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    const bool written = emit(level_tag, func, fmt, args);
    va_end(args);
    return written;
}

void HostLogger::emit_clock_anchor_if_needed() {
    if (!is_enabled(LogLevel::TIMING)) return;

    const pid_t pid = getpid();
    SimplerHostLogState *shared = state();
    const int32_t pid_value = static_cast<int32_t>(pid);
    int32_t observed = atomic_load_i32(&shared->clock_anchor_pid);
    if (observed == pid_value || observed == -pid_value) return;
    if (!atomic_compare_exchange_i32(&shared->clock_anchor_pid, &observed, -pid_value)) {
        return;
    }

    const int64_t monotonic_ns = simpler::log::monotonic_now_ns();
    const int64_t wall_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch())
            .count();
    const bool written = emit_ungated(
        level_name(LogLevel::TIMING), "clock_anchor", "[CLOCK_ANCHOR] v=1 pid=%d mono_ns=%lld wall_ns=%lld",
        static_cast<int>(pid), static_cast<long long>(monotonic_ns), static_cast<long long>(wall_ns)
    );
    int32_t claim = -pid_value;
    (void)atomic_compare_exchange_i32(&shared->clock_anchor_pid, &claim, written ? pid_value : 0);
}

void HostLogger::vlog(LogLevel level, const char *func, const char *fmt, va_list args) {
    if (!is_enabled(level)) {
        return;
    }
    emit_clock_anchor_if_needed();
    emit(level_name(level), func, fmt, args);
}

void HostLogger::log(LogLevel level, const char *func, const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vlog(level, func, fmt, args);
    va_end(args);
}

void HostLogger::log_host_span(const SimplerHostSpan *span) {
    if (!is_enabled(LogLevel::TIMING)) return;
    if (span == nullptr || span->abi_version != SIMPLER_HOST_SPAN_ABI_VERSION ||
        span->struct_size < sizeof(SimplerHostSpan) || span->name == nullptr) {
        return;
    }
    const std::string name = encode_host_span_field(span->name, kHostSpanNameCapacity, false);
    const std::string attributes =
        encode_host_span_field(span->attributes == nullptr ? "" : span->attributes, kHostSpanAttributesCapacity, true);
    const char *directory = std::getenv("SIMPLER_HOST_STRACE_DIR");
    if (directory != nullptr && directory[0] != '\0' &&
        write_host_span_file(directory, *span, host_trace_tid(), name, attributes)) {
        return;
    }
    log(LogLevel::TIMING, "emit_host_span",
        "[STRACE] v=1 pid=%d tid=%ld inv=%" PRIu64 " hid=%" PRIx64 " depth=%d name=%s ts=%" PRId64 " dur=%" PRId64
        " %s",
        static_cast<int>(getpid()), host_trace_tid(), span->invocation_id, span->callable_hash, span->depth,
        name.c_str(), span->timestamp_ns, span->duration_ns, attributes.c_str());
}

extern "C" __attribute__((visibility("default"))) int simpler_host_log_bind_state(SimplerHostLogState *state) {
    return HostLogger::get_instance().bind_state(state);
}
