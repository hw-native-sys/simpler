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
 * @file host_log.h
 * @brief Unified Host Logging System
 *
 * Two orthogonal axes:
 *   - Severity: DEBUG/INFO/WARN/ERROR/NUL (matches CANN dlog 1:1)
 *   - INFO verbosity: integer 0..9 (only meaningful when severity == INFO)
 *
 * Configuration is pushed in from Python via nanobind binding
 * `set_host_log_config(severity, info_v)`; this module never reads env vars.
 * The Python-facing integer level layout (V0=15..V9=24, INFO=20=V5, etc.)
 * lives in the Python module — C++ only stores the two axes separately.
 */

#ifndef PLATFORM_HOST_LOG_H_
#define PLATFORM_HOST_LOG_H_

#include <cstdarg>
#include <cstdio>
#include <mutex>

namespace simpler::log {

// Severity (matches CANN dlog enum 1:1)
enum class LogLevel : int {
    DEBUG = 0,
    INFO = 1,
    WARN = 2,
    ERROR = 3,
    NUL = 4,
};

// Defaults — single source of truth shared with Python via nanobind binding.
// The full integer level layout (DEBUG=10, V0..V9=15..24, WARN=30, etc.)
// is Python-side; C++ only stores severity + verbosity as separate axes.
constexpr int kDefaultInfoV = 5;       // V5
constexpr int kDefaultThreshold = 20;  // V5 = Python INFO

}  // namespace simpler::log

class HostLogger {
public:
    static HostLogger &get_instance();

    // Severity-only entry (DEBUG/WARN/ERROR). NOTE: caller must NOT pass INFO here;
    // INFO goes through log_info_v with a verbosity tier.
    void log(simpler::log::LogLevel level, const char *func, const char *fmt, ...);

    // INFO with verbosity tier (v ∈ [0, 9]).
    void log_info_v(int v, const char *func, const char *fmt, ...);

    // va_list-taking primitives — used by unified_log_* adapters to forward
    // a caller's variadic args without an intermediate vsnprintf-to-buffer
    // round-trip. Caller is responsible for `va_start` / `va_end`.
    void vlog(simpler::log::LogLevel level, const char *func, const char *fmt, va_list args);
    void vlog_info_v(int v, const char *func, const char *fmt, va_list args);

    void set_level(simpler::log::LogLevel level);
    void set_info_v(int v);

    bool is_severity_enabled(simpler::log::LogLevel level) const;
    bool is_info_v_enabled(int v) const;

private:
    HostLogger();
    ~HostLogger() = default;

    HostLogger(const HostLogger &) = delete;
    HostLogger &operator=(const HostLogger &) = delete;
    HostLogger(HostLogger &&) = delete;
    HostLogger &operator=(HostLogger &&) = delete;

    const char *level_name(simpler::log::LogLevel level) const;
    void emit(const char *level_tag, const char *func, const char *fmt, va_list args);

    simpler::log::LogLevel current_level_;
    int current_info_v_;
    std::mutex mutex_;
};

#endif  // PLATFORM_HOST_LOG_H_
