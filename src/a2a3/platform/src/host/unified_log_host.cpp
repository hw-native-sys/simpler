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
 * @brief Unified logging - Host implementation
 */

#include "common/unified_log.h"
#include "host_log.h"

#include <cstdarg>
#include <cstdio>

void unified_log_error(const char *func, const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    char buffer[2048];
    vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);
    HostLogger::get_instance().log(HostLogLevel::ERROR, "%s: %s", func, buffer);
}

void unified_log_warn(const char *func, const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);

    char buffer[2048];
    vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);

    HostLogger::get_instance().log(HostLogLevel::WARN, "%s: %s", func, buffer);
}

void unified_log_info(const char *func, const char *fmt, ...) {
    if (!HostLogger::get_instance().is_enabled(HostLogLevel::INFO)) {
        return;
    }
    va_list args;
    va_start(args, fmt);
    char buffer[2048];
    vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);
    HostLogger::get_instance().log(HostLogLevel::INFO, "%s: %s", func, buffer);
}

void unified_log_debug(const char *func, const char *fmt, ...) {
    if (!HostLogger::get_instance().is_enabled(HostLogLevel::DEBUG)) {
        return;
    }
    va_list args;
    va_start(args, fmt);
    char buffer[2048];
    vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);
    HostLogger::get_instance().log(HostLogLevel::DEBUG, "%s: %s", func, buffer);
}

void unified_log_always(const char *func, const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    char buffer[2048];
    vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);
    HostLogger::get_instance().log(HostLogLevel::ALWAYS, "%s: %s", func, buffer);
}
