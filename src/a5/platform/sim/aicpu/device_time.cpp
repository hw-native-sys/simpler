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
#include "aicpu/device_time.h"

#include <chrono>

#include "common/platform_config.h"

uint64_t get_sys_cnt_aicpu() {
    auto now = std::chrono::high_resolution_clock::now();
    uint64_t elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
    constexpr uint64_t kNsPerSec = std::nano::den;
    uint64_t seconds = elapsed_ns / kNsPerSec;
    uint64_t remaining_ns = elapsed_ns % kNsPerSec;
    uint64_t ticks = seconds * PLATFORM_PROF_SYS_CNT_FREQ + (remaining_ns * PLATFORM_PROF_SYS_CNT_FREQ) / kNsPerSec;
    return ticks;
}
