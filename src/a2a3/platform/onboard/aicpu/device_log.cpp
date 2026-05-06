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
 * @file device_log.cpp (onboard)
 * @brief Onboard AICPU log implementation backed by CANN dlog.
 *
 * Severity is owned by CANN: init_log_switch() snapshots CheckLogLevel() and
 * set_log_level() is therefore a no-op on this backend (the host's pushed
 * value would be overwritten by CANN at the next init anyway, and CANN is the
 * only authoritative source on the AICPU).
 *
 * Verbosity (V0..V9) is simpler-managed: g_log_info_v is set by
 * set_log_info_v() from the host-published KernelArgs.log_info_v before each
 * kernel run.
 */

#include "aicpu/device_log.h"
#include "dlog_pub.h"

#include <cstdarg>
#include <cstdio>

bool g_is_log_enable_debug = false;
bool g_is_log_enable_info = false;
bool g_is_log_enable_warn = false;
bool g_is_log_enable_error = false;

int g_log_info_v = 5;

void init_log_switch() {
    g_is_log_enable_debug = CheckLogLevel(AICPU, DLOG_DEBUG);
    g_is_log_enable_info = CheckLogLevel(AICPU, DLOG_INFO);
    g_is_log_enable_warn = CheckLogLevel(AICPU, DLOG_WARN);
    g_is_log_enable_error = CheckLogLevel(AICPU, DLOG_ERROR);
}

extern "C" void set_log_level(int /*level*/) {
    // No-op on onboard: CANN dlog is the only authoritative severity source.
    // Severity flags are populated by init_log_switch() via CheckLogLevel.
}

extern "C" void set_log_info_v(int v) {
    if (v < 0) v = 0;
    if (v > 9) v = 9;
    g_log_info_v = v;
}

extern "C" int get_log_info_v() { return g_log_info_v; }

// =============================================================================
// Low-level dev_log_* (onboard: route through CANN dlog)
// =============================================================================

void dev_log_debug(const char *func, const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    char buffer[2048];
    vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);
    dlog_debug(AICPU, "%lu %s\n\"%s\"", GET_TID(), func, buffer);
}

void dev_log_warn(const char *func, const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    char buffer[2048];
    vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);
    dlog_warn(AICPU, "%lu %s\n\"%s\"", GET_TID(), func, buffer);
}

void dev_log_error(const char *func, const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    char buffer[2048];
    vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);
    dlog_error(AICPU, "%lu %s\n\"%s\"", GET_TID(), func, buffer);
}

void dev_log_info_v(int v, const char *func, const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    char buffer[2048];
    vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);
    // Tag the verbosity tier in-message so it's grep-able alongside CANN's
    // own [INFO] prefix.
    dlog_info(AICPU, "%lu %s [V%d]\n\"%s\"", GET_TID(), func, v, buffer);
}
