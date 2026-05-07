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
 * @file unified_log.h
 * @brief Unified logging interface using link-time polymorphism.
 *
 * One adapter ABI for both host and device. Host build links unified_log_host.cpp,
 * device build links unified_log_device.cpp.
 *
 * Severity macros (LOG_DEBUG/WARN/ERROR) plus 10 INFO verbosity tiers
 * (LOG_INFO_V0..V9). v=0 is the most verbose (sub-INFO), v=9 is the most
 * must-see (above-INFO). v=5 is the default threshold and aliases Python's
 * standard INFO level.
 */

#ifndef PLATFORM_UNIFIED_LOG_H_
#define PLATFORM_UNIFIED_LOG_H_

#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

void unified_log_error(const char *func, const char *fmt, ...);
void unified_log_warn(const char *func, const char *fmt, ...);
void unified_log_debug(const char *func, const char *fmt, ...);
void unified_log_info_v(const char *func, int v, const char *fmt, ...);

#ifdef __cplusplus
}
#endif

#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

// Severity-only macros
#define LOG_ERROR(fmt, ...) unified_log_error(__FUNCTION__, "[%s:%d] " fmt, __FILENAME__, __LINE__, ##__VA_ARGS__)
#define LOG_WARN(fmt, ...) unified_log_warn(__FUNCTION__, "[%s:%d] " fmt, __FILENAME__, __LINE__, ##__VA_ARGS__)
#define LOG_DEBUG(fmt, ...) unified_log_debug(__FUNCTION__, "[%s:%d] " fmt, __FILENAME__, __LINE__, ##__VA_ARGS__)

// INFO verbosity tiers (v ∈ [0,9]). 0=most verbose, 9=must-see, 5=default threshold.
#define LOG_INFO_V0(fmt, ...) unified_log_info_v(__FUNCTION__, 0, "[%s:%d] " fmt, __FILENAME__, __LINE__, ##__VA_ARGS__)
#define LOG_INFO_V1(fmt, ...) unified_log_info_v(__FUNCTION__, 1, "[%s:%d] " fmt, __FILENAME__, __LINE__, ##__VA_ARGS__)
#define LOG_INFO_V2(fmt, ...) unified_log_info_v(__FUNCTION__, 2, "[%s:%d] " fmt, __FILENAME__, __LINE__, ##__VA_ARGS__)
#define LOG_INFO_V3(fmt, ...) unified_log_info_v(__FUNCTION__, 3, "[%s:%d] " fmt, __FILENAME__, __LINE__, ##__VA_ARGS__)
#define LOG_INFO_V4(fmt, ...) unified_log_info_v(__FUNCTION__, 4, "[%s:%d] " fmt, __FILENAME__, __LINE__, ##__VA_ARGS__)
#define LOG_INFO_V5(fmt, ...) unified_log_info_v(__FUNCTION__, 5, "[%s:%d] " fmt, __FILENAME__, __LINE__, ##__VA_ARGS__)
#define LOG_INFO_V6(fmt, ...) unified_log_info_v(__FUNCTION__, 6, "[%s:%d] " fmt, __FILENAME__, __LINE__, ##__VA_ARGS__)
#define LOG_INFO_V7(fmt, ...) unified_log_info_v(__FUNCTION__, 7, "[%s:%d] " fmt, __FILENAME__, __LINE__, ##__VA_ARGS__)
#define LOG_INFO_V8(fmt, ...) unified_log_info_v(__FUNCTION__, 8, "[%s:%d] " fmt, __FILENAME__, __LINE__, ##__VA_ARGS__)
#define LOG_INFO_V9(fmt, ...) unified_log_info_v(__FUNCTION__, 9, "[%s:%d] " fmt, __FILENAME__, __LINE__, ##__VA_ARGS__)

#endif  // PLATFORM_UNIFIED_LOG_H_
