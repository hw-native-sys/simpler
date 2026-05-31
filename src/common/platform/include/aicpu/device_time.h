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
 * @file device_time.h
 * @brief AICPU Device Timestamping Interface
 *
 * Provides get_sys_cnt_aicpu() for AICPU-side timestamping on both
 * real hardware and simulation.
 *
 * Platform Support (same shape for both arches):
 * - onboard: Real Ascend hardware (reads CNTVCT_EL0)
 * - sim: Host-based simulation using std::chrono
 */

#ifndef SRC_COMMON_PLATFORM_INCLUDE_AICPU_DEVICE_TIME_H_
#define SRC_COMMON_PLATFORM_INCLUDE_AICPU_DEVICE_TIME_H_

#include <cstdint>

/**
 * AICPU system counter for performance profiling.
 *
 * Returns a monotonic counter value compatible with AICore's get_sys_cnt().
 * Implementation is platform-specific (hardware counter or chrono simulation).
 *
 * @return Counter ticks
 */
uint64_t get_sys_cnt_aicpu();

#endif  // SRC_COMMON_PLATFORM_INCLUDE_AICPU_DEVICE_TIME_H_
