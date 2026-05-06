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
 * @file device_log.h
 * @brief Unified Device Logging Interface for AICPU
 *
 * Layered design:
 *   - Low-level dev_log_*() functions are platform-specific (CANN dlog on
 *     real hardware, fprintf(stderr,...) in simulation).
 *   - Severity gating uses the same flag table on both backends:
 *     onboard fills it from CheckLogLevel(AICPU,...) (CANN-managed),
 *     sim fills it from set_log_level() called by the host (dlsym path).
 *   - INFO verbosity gating (V0..V9) is simpler-managed on both backends:
 *     g_log_info_v populated from set_log_info_v(); onboard receives the
 *     value via KernelArgs.log_info_v at kernel entry, sim receives it via
 *     dlsym from the host runner.
 *
 * Platform Support:
 * - a5     : Real hardware with CANN dlog API
 * - a5sim  : Host-based simulation using fprintf(stderr,...)
 */

#ifndef PLATFORM_DEVICE_LOG_H_
#define PLATFORM_DEVICE_LOG_H_

#include <cstdio>
#include <cstdint>

// =============================================================================
// Platform Detection and Thread ID
// =============================================================================

#ifdef __linux__
#include <sys/syscall.h>
#include <unistd.h>
#define GET_TID() syscall(SYS_gettid)
#else
#define GET_TID() 0
#endif

// =============================================================================
// Severity enable flags (defined in platform-specific device_log.cpp)
// =============================================================================

extern bool g_is_log_enable_debug;
extern bool g_is_log_enable_info;
extern bool g_is_log_enable_warn;
extern bool g_is_log_enable_error;

// INFO verbosity threshold (0..9). Default 5.
extern int g_log_info_v;

// =============================================================================
// Configuration setters (called by AICPU kernel init from KernelArgs)
// =============================================================================

// Severity. Levels are CANN-aligned ints: DEBUG=0, INFO=1, WARN=2, ERROR=3, NUL=4.
// Onboard ignores this (CANN dlog is the source); sim uses it to set the flag table.
extern "C" void set_log_level(int level);
extern "C" void set_log_info_v(int v);
extern "C" int get_log_info_v();

// =============================================================================
// Platform-specific logging functions (low-level layer)
// =============================================================================

void dev_log_debug(const char *func, const char *fmt, ...);
void dev_log_warn(const char *func, const char *fmt, ...);
void dev_log_error(const char *func, const char *fmt, ...);
void dev_log_info_v(int v, const char *func, const char *fmt, ...);

// =============================================================================
// Helper Functions
// =============================================================================

inline bool is_log_enable_debug() { return g_is_log_enable_debug; }
inline bool is_log_enable_info() { return g_is_log_enable_info; }
inline bool is_log_enable_warn() { return g_is_log_enable_warn; }
inline bool is_log_enable_error() { return g_is_log_enable_error; }

// Initialize log switch (platform-specific implementation)
void init_log_switch();

#endif  // PLATFORM_DEVICE_LOG_H_
