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
 * CANN owns DEBUG, INFO, WARN, and ERROR through CheckLogLevel(). TIMING uses
 * CANN WARN for emission but keeps the finer host-provided threshold because
 * CANN has no intermediate level.
 */

#include "aicpu/device_log.h"
#include "common/log_level.h"
#include "dlog_pub.h"

#include <cstdarg>
#include <cstdio>

namespace {
bool g_is_log_enable_debug = false;
bool g_is_log_enable_info = false;
bool g_is_log_enable_timing = false;
bool g_is_log_enable_warn = false;
bool g_is_log_enable_error = false;
}  // namespace

bool is_log_enable_debug() { return g_is_log_enable_debug; }

bool is_log_enable_info() { return g_is_log_enable_info; }

bool is_log_enable_timing() { return g_is_log_enable_timing && g_is_log_enable_warn; }

bool is_log_enable_warn() { return g_is_log_enable_warn; }

bool is_log_enable_error() { return g_is_log_enable_error; }

void init_log_switch() {
    g_is_log_enable_debug = CheckLogLevel(AICPU, DLOG_DEBUG);
    g_is_log_enable_info = CheckLogLevel(AICPU, DLOG_INFO);
    g_is_log_enable_warn = CheckLogLevel(AICPU, DLOG_WARN);
    g_is_log_enable_error = CheckLogLevel(AICPU, DLOG_ERROR);
}

extern "C" void set_log_level(int level) {
    g_is_log_enable_timing = level <= static_cast<int>(simpler::log::LogLevel::TIMING);
}

int bind_orchestration_host_log_state(void * /*handle*/, const char **error) {
    if (error != nullptr) *error = nullptr;
    return 0;
}

// =============================================================================
// Low-level dev_log_* / dev_vlog_* (onboard: route through CANN dlog)
//
// CANN's dlog API is variadic only (no va_list variant), so the va_list path
// still buffers via vsnprintf — same total cost as before, just moved one
// frame deeper so unified_log_device.cpp can call dev_vlog_* uniformly.
// =============================================================================

void dev_vlog_debug(const char *func, const char *fmt, va_list args) {
    char buffer[2048];
    vsnprintf(buffer, sizeof(buffer), fmt, args);
    dlog_debug(AICPU, "%lu %s\n\"%s\"", GET_TID(), func, buffer);
}

void dev_vlog_info(const char *func, const char *fmt, va_list args) {
    char buffer[2048];
    vsnprintf(buffer, sizeof(buffer), fmt, args);
    dlog_info(AICPU, "%lu %s\n\"%s\"", GET_TID(), func, buffer);
}

void dev_vlog_timing(const char *func, const char *fmt, va_list args) {
    char buffer[2048];
    vsnprintf(buffer, sizeof(buffer), fmt, args);
    dlog_warn(AICPU, "%lu %s [TIMING]\n\"%s\"", GET_TID(), func, buffer);
}

void dev_vlog_warn(const char *func, const char *fmt, va_list args) {
    char buffer[2048];
    vsnprintf(buffer, sizeof(buffer), fmt, args);
    dlog_warn(AICPU, "%lu %s\n\"%s\"", GET_TID(), func, buffer);
}

void dev_vlog_error(const char *func, const char *fmt, va_list args) {
    char buffer[2048];
    vsnprintf(buffer, sizeof(buffer), fmt, args);
    dlog_error(AICPU, "%lu %s\n\"%s\"", GET_TID(), func, buffer);
}
