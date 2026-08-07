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
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <pthread.h>
#include <vector>

#include <unistd.h>

#if defined(__linux__)
#include <sys/syscall.h>
#endif

#include "common/host_span.h"

using simpler::log::LogLevel;

namespace {

// Every STRACE marker renders well inside this, so the heap fallback below
// stays off the path a traced run pays per span.
constexpr size_t kRecordStackCapacity = 2048;

// Renders the timestamp/thread/level prefix, the caller's message, and an
// optional trailing newline into `buffer`, and returns the length of the whole
// record. A return value of `capacity` or more means `buffer` holds only a
// truncated prefix and the caller must re-render into that many bytes.
size_t format_record(
    char *buffer, size_t capacity, const char *ts, unsigned long tid, const char *level_tag, const char *func,
    const char *fmt, va_list args, bool append_newline
) {
    size_t length = 0;
    auto tail = [&]() -> char * {
        return length < capacity ? buffer + length : nullptr;
    };
    auto remaining = [&]() -> size_t {
        return length < capacity ? capacity - length : 0;
    };

    const int prefix = snprintf(tail(), remaining(), "[%s][T0x%lx][%s] %s: ", ts, tid, level_tag, func);
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

void write_stderr(const char *record, size_t size) {
    size_t offset = 0;
    while (offset < size) {
        const ssize_t written = ::write(STDERR_FILENO, record + offset, size - offset);
        if (written > 0) {
            offset += static_cast<size_t>(written);
        } else if (written < 0 && errno == EINTR) {
            continue;
        } else {
            break;
        }
    }
}

long host_trace_tid() {
#if defined(__linux__) && defined(SYS_gettid)
    return static_cast<long>(syscall(SYS_gettid));
#else
    return static_cast<long>(getpid());
#endif
}

}  // namespace

HostLogger &HostLogger::get_instance() {
    static HostLogger instance;
    return instance;
}

HostLogger::HostLogger() :
    current_level_(LogLevel::TIMING) {}

void HostLogger::set_level(LogLevel level) {
    std::scoped_lock lock(mutex_);
    current_level_ = level;
}

int HostLogger::level() const { return static_cast<int>(current_level_); }

int HostLogger::cann_level() const { return simpler::log::to_cann_log_level(current_level_); }

void HostLogger::configure_cann_log_level(int (*set_level)(int, int, int)) const {
    if (std::getenv("ASCEND_GLOBAL_LOG_LEVEL") == nullptr) {
        set_level(-1, cann_level(), 0);
    }
}

bool HostLogger::is_enabled(LogLevel level) const {
    // current_level_ is the floor: messages with severity >= floor are kept.
    return static_cast<int>(level) >= static_cast<int>(current_level_) && current_level_ != LogLevel::NUL;
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

void HostLogger::emit(const char *level_tag, const char *func, const char *fmt, va_list args) {
    using namespace std::chrono;
    auto now = system_clock::now();
    auto t = system_clock::to_time_t(now);
    auto us = duration_cast<microseconds>(now.time_since_epoch()) % 1'000'000;
    struct tm tm_buf;
    localtime_r(&t, &tm_buf);
    char ts[40];
    size_t n = strftime(ts, sizeof(ts), "%Y-%m-%d %H:%M:%S", &tm_buf);
    snprintf(ts + n, sizeof(ts) - n, ".%06lld", static_cast<long long>(us.count()));

    auto tid = static_cast<unsigned long>(reinterpret_cast<uintptr_t>(pthread_self()));

    const bool append_newline = fmt[0] != '\0' && fmt[strlen(fmt) - 1] != '\n';

    // One write per record: a record of at most PIPE_BUF bytes reaches a shared
    // pipe indivisibly, so forked workers writing a captured stderr cannot
    // interleave inside it. A longer record — a large LOG_ERROR dump, say — has
    // no such guarantee; STRACE markers stay well below the limit.
    char stack_buffer[kRecordStackCapacity];
    const size_t length =
        format_record(stack_buffer, sizeof(stack_buffer), ts, tid, level_tag, func, fmt, args, append_newline);
    if (length < sizeof(stack_buffer)) {
        std::scoped_lock lock(mutex_);
        write_stderr(stack_buffer, length);
        return;
    }

    std::vector<char> heap_buffer(length + 1);
    const size_t heap_length =
        format_record(heap_buffer.data(), heap_buffer.size(), ts, tid, level_tag, func, fmt, args, append_newline);
    std::scoped_lock lock(mutex_);
    write_stderr(heap_buffer.data(), heap_length < heap_buffer.size() ? heap_length : length);
}

void HostLogger::vlog(LogLevel level, const char *func, const char *fmt, va_list args) {
    if (!is_enabled(level)) {
        return;
    }
    emit(level_name(level), func, fmt, args);
}

void HostLogger::log(LogLevel level, const char *func, const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vlog(level, func, fmt, args);
    va_end(args);
}

// ---------------------------------------------------------------------------
// C ABI entry — resolved by ChipWorker via dlsym from libsimpler_log.so.
//
// Called once early in ChipWorker::init (before host_runtime.so is even
// dlopen'd) to seed the process-wide HostLogger from the user's
// `simpler` Python logger snapshot. Consumers that need the current value
// later (host_runtime.so populating InitArgs.log_level at device init) read it
// via HostLogger::get_instance().level() directly; the value never
// has to travel through any other SO's C ABI.
//
// Level values match Python logging thresholds.
// Returns 0 on success, negative for an unsupported threshold.
// ---------------------------------------------------------------------------
extern "C" int simpler_log_init(int log_level) {
    if (!simpler::log::is_valid_level(log_level)) {
        return -1;
    }
    HostLogger::get_instance().set_level(static_cast<LogLevel>(log_level));
    return 0;
}

extern "C" void simpler_log_emit_host_span(const SimplerHostSpan *span) {
    if (span == nullptr || span->abi_version != SIMPLER_HOST_SPAN_ABI_VERSION ||
        span->struct_size < sizeof(SimplerHostSpan) || span->name == nullptr) {
        return;
    }
    HostLogger::get_instance().log(
        LogLevel::TIMING, "emit_host_span",
        "[STRACE] v=1 pid=%d tid=%ld inv=%llu hid=%llx depth=%d name=%s ts=%lld dur=%lld %s",
        static_cast<int>(getpid()), host_trace_tid(), static_cast<unsigned long long>(span->invocation_id),
        static_cast<unsigned long long>(span->callable_hash), span->depth, span->name,
        static_cast<long long>(span->timestamp_ns), static_cast<long long>(span->duration_ns),
        span->attributes == nullptr ? "" : span->attributes
    );
}
