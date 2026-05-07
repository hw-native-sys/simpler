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
 * @file unified_log_device.cpp
 * @brief Unified logging - Device implementation.
 *
 * Forwards the unified C ABI to dev_vlog_* primitives via va_list — no
 * intermediate vsnprintf-to-buffer round-trip in this layer. On sim,
 * dev_vlog_* is a single vfprintf (buffer-free); on onboard, it still
 * buffers internally because CANN's dlog has no va_list variant.
 *
 * Severity flags and verbosity threshold come from device_log.cpp's globals
 * (set at init time from KernelArgs).
 */

#include "common/unified_log.h"
#include "aicpu/device_log.h"

#include <cstdarg>

void unified_log_error(const char *func, const char *fmt, ...) {
    if (!is_log_enable_error()) {
        return;
    }
    va_list args;
    va_start(args, fmt);
    dev_vlog_error(func, fmt, args);
    va_end(args);
}

void unified_log_warn(const char *func, const char *fmt, ...) {
    if (!is_log_enable_warn()) {
        return;
    }
    va_list args;
    va_start(args, fmt);
    dev_vlog_warn(func, fmt, args);
    va_end(args);
}

void unified_log_debug(const char *func, const char *fmt, ...) {
    if (!is_log_enable_debug()) {
        return;
    }
    va_list args;
    va_start(args, fmt);
    dev_vlog_debug(func, fmt, args);
    va_end(args);
}

void unified_log_info_v(const char *func, int v, const char *fmt, ...) {
    if (!is_log_enable_info() || v < g_log_info_v) {
        return;
    }
    va_list args;
    va_start(args, fmt);
    dev_vlog_info_v(v, func, fmt, args);
    va_end(args);
}
