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
 * an intermediate vsnprintf-to-buffer round-trip.
 */

#include "common/unified_log.h"
#include "host_log.h"

#include <cstdarg>

using simpler::log::LogLevel;

void unified_log_error(const char *func, const char *fmt, ...) {
    if (!HostLogger::get_instance().is_severity_enabled(LogLevel::ERROR)) {
        return;
    }
    va_list args;
    va_start(args, fmt);
    HostLogger::get_instance().vlog(LogLevel::ERROR, func, fmt, args);
    va_end(args);
}

void unified_log_warn(const char *func, const char *fmt, ...) {
    if (!HostLogger::get_instance().is_severity_enabled(LogLevel::WARN)) {
        return;
    }
    va_list args;
    va_start(args, fmt);
    HostLogger::get_instance().vlog(LogLevel::WARN, func, fmt, args);
    va_end(args);
}

void unified_log_debug(const char *func, const char *fmt, ...) {
    if (!HostLogger::get_instance().is_severity_enabled(LogLevel::DEBUG)) {
        return;
    }
    va_list args;
    va_start(args, fmt);
    HostLogger::get_instance().vlog(LogLevel::DEBUG, func, fmt, args);
    va_end(args);
}

void unified_log_info_v(const char *func, int v, const char *fmt, ...) {
    if (!HostLogger::get_instance().is_info_v_enabled(v)) {
        return;
    }
    va_list args;
    va_start(args, fmt);
    HostLogger::get_instance().vlog_info_v(v, func, fmt, args);
    va_end(args);
}
