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

#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <pthread.h>

using simpler::log::LogLevel;

HostLogger &HostLogger::get_instance() {
    static HostLogger instance;
    return instance;
}

HostLogger::HostLogger() :
    current_level_(LogLevel::INFO),
    current_info_v_(simpler::log::kDefaultInfoV) {}

void HostLogger::set_level(LogLevel level) {
    std::scoped_lock lock(mutex_);
    current_level_ = level;
}

void HostLogger::set_info_v(int v) {
    if (v < 0) v = 0;
    if (v > 9) v = 9;
    std::scoped_lock lock(mutex_);
    current_info_v_ = v;
}

int HostLogger::level() const { return static_cast<int>(current_level_); }

int HostLogger::info_v() const { return current_info_v_; }

bool HostLogger::is_severity_enabled(LogLevel level) const {
    // current_level_ is the floor: messages with severity >= floor are kept.
    return static_cast<int>(level) >= static_cast<int>(current_level_) && current_level_ != LogLevel::NUL;
}

bool HostLogger::is_info_v_enabled(int v) const { return is_severity_enabled(LogLevel::INFO) && v >= current_info_v_; }

const char *HostLogger::level_name(LogLevel level) const {
    switch (level) {
    case LogLevel::DEBUG:
        return "DEBUG";
    case LogLevel::INFO:
        return "INFO";
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

    std::scoped_lock lock(mutex_);
    fprintf(stderr, "[%s][T0x%lx][%s] %s: ", ts, tid, level_tag, func);
    vfprintf(stderr, fmt, args);
    if (fmt[0] != '\0' && fmt[strlen(fmt) - 1] != '\n') {
        fputc('\n', stderr);
    }
    fflush(stderr);
}

void HostLogger::vlog(LogLevel level, const char *func, const char *fmt, va_list args) {
    if (!is_severity_enabled(level)) {
        return;
    }
    emit(level_name(level), func, fmt, args);
}

void HostLogger::vlog_info_v(int v, const char *func, const char *fmt, va_list args) {
    if (!is_info_v_enabled(v)) {
        return;
    }
    char tag[8];
    snprintf(tag, sizeof(tag), "INFO_V%d", v);
    emit(tag, func, fmt, args);
}

void HostLogger::log(LogLevel level, const char *func, const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vlog(level, func, fmt, args);
    va_end(args);
}

void HostLogger::log_info_v(int v, const char *func, const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vlog_info_v(v, func, fmt, args);
    va_end(args);
}

// ---------------------------------------------------------------------------
// C ABI entry — resolved by ChipWorker via dlsym from libsimpler_log.so.
//
// Called once early in ChipWorker::init (before host_runtime.so is even
// dlopen'd) to seed the process-wide HostLogger from the user's
// `simpler` Python logger snapshot. Consumers that need the current value
// later (host_runtime.so populating KernelArgs.log_level) read it via
// HostLogger::get_instance().level() / .info_v() directly; the value never
// has to travel through any other SO's C ABI.
//
// Severity layout matches CANN dlog (0=DEBUG..4=NUL); info_v ∈ [0,9].
// Returns 0 on success, negative on out-of-range input.
// ---------------------------------------------------------------------------
extern "C" int simpler_log_init(int log_level, int log_info_v) {
    if (log_level < static_cast<int>(LogLevel::DEBUG) || log_level > static_cast<int>(LogLevel::NUL)) {
        return -1;
    }
    if (log_info_v < 0 || log_info_v > 9) {
        return -1;
    }
    HostLogger::get_instance().set_level(static_cast<LogLevel>(log_level));
    HostLogger::get_instance().set_info_v(log_info_v);
    return 0;
}
