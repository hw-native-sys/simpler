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
 * @file unified_log_host.cpp
 * @brief Unified logging - Host implementation.
 *
 * Adapter that forwards the unified C ABI to HostLogger via va_list, avoiding
 * an intermediate vsnprintf-to-buffer round-trip. HostLogger::vlog is the
 * single authority for level gating; this adapter does not re-check.
 */

#include "common/unified_log.h"
#include "host_log.h"

#include <cstdarg>

#include "common/host_span.h"

using simpler::log::LogLevel;

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC visibility push(hidden)
#endif

int unified_log_host_span_enabled() { return HostLogger::get_instance().is_enabled(LogLevel::TIMING) ? 1 : 0; }

void unified_log_host_span(const SimplerHostSpan *span) { HostLogger::get_instance().log_host_span(span); }

SimplerHostTraceState *unified_log_host_trace_state() { return HostLogger::get_instance().trace_state(); }

uint64_t unified_log_host_trace_allocate_inv() { return HostLogger::get_instance().allocate_trace_invocation(); }

void unified_log_error(const char *func, const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    HostLogger::get_instance().vlog(LogLevel::ERROR, func, fmt, args);
    va_end(args);
}

void unified_log_warn(const char *func, const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    HostLogger::get_instance().vlog(LogLevel::WARN, func, fmt, args);
    va_end(args);
}

void unified_log_timing(const char *func, const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    HostLogger::get_instance().vlog(LogLevel::TIMING, func, fmt, args);
    va_end(args);
}

void unified_log_info(const char *func, const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    HostLogger::get_instance().vlog(LogLevel::INFO, func, fmt, args);
    va_end(args);
}

void unified_log_debug(const char *func, const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    HostLogger::get_instance().vlog(LogLevel::DEBUG, func, fmt, args);
    va_end(args);
}

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC visibility pop
#endif
