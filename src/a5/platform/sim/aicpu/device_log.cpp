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
 * @file device_log.cpp (sim)
 * @brief Simulation Platform Log Implementation
 *
 * Severity and verbosity flags are populated by host via set_log_level() /
 * set_log_info_v() at AICPU kernel init (see kernel.cpp / aicpu_executor.cpp);
 * this file does not read env vars.
 */

#include "aicpu/device_log.h"

#include <cstdarg>
#include <cstdio>

// =============================================================================
// Severity enable flags + verbosity threshold (mutated by setters below)
// =============================================================================

bool g_is_log_enable_debug = false;
bool g_is_log_enable_info = true;  // default INFO+ on
bool g_is_log_enable_warn = true;
bool g_is_log_enable_error = true;

// Default V5 (matches HostLogger / Python defaults).
int g_log_info_v = 5;

const char *TILE_FWK_DEVICE_MACHINE = "SIM_CPU";

// =============================================================================
// Setters (called by AICPU init from KernelArgs)
// =============================================================================

extern "C" void set_log_level(int level) {
    // CANN-aligned: DEBUG=0, INFO=1, WARN=2, ERROR=3, NUL=4. Floor semantics:
    // messages with severity >= floor are kept.
    g_is_log_enable_debug = (level <= 0) && (level != 4);
    g_is_log_enable_info = (level <= 1) && (level != 4);
    g_is_log_enable_warn = (level <= 2) && (level != 4);
    g_is_log_enable_error = (level <= 3) && (level != 4);
}

extern "C" void set_log_info_v(int v) {
    if (v < 0) v = 0;
    if (v > 9) v = 9;
    g_log_info_v = v;
}

extern "C" int get_log_info_v() { return g_log_info_v; }

// =============================================================================
// init_log_switch: sim respects host-pushed config — this is now a no-op
// (kept for ABI compatibility with onboard, where it queries CANN dlog).
// =============================================================================

void init_log_switch() {
    // Sim has no env / dlog to consult. Defaults already applied at static
    // init; host overrides via set_log_level()/set_log_info_v() before this
    // is called.
}

// =============================================================================
// Low-level dev_log_* (sim: fprintf to stderr; severity-by-prefix)
// =============================================================================

void dev_log_debug(const char *func, const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "[DEBUG] %s: ", func);
    vfprintf(stderr, fmt, args);
    fputc('\n', stderr);
    va_end(args);
}

void dev_log_warn(const char *func, const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "[WARN] %s: ", func);
    vfprintf(stderr, fmt, args);
    fputc('\n', stderr);
    va_end(args);
}

void dev_log_error(const char *func, const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "[ERROR] %s: ", func);
    vfprintf(stderr, fmt, args);
    fputc('\n', stderr);
    va_end(args);
}

void dev_log_info_v(int v, const char *func, const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "[INFO_V%d] %s: ", v, func);
    vfprintf(stderr, fmt, args);
    fputc('\n', stderr);
    va_end(args);
}
