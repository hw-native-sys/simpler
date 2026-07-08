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

#ifndef SRC_COMMON_PLATFORM_SIM_AICPU_DEVICE_TIME_FAST_H_
#define SRC_COMMON_PLATFORM_SIM_AICPU_DEVICE_TIME_FAST_H_

#include <cstdint>

#include "aicpu/device_time.h"

static inline uint64_t fast_sys_cnt_aicpu() { return get_sys_cnt_aicpu(); }

#endif  // SRC_COMMON_PLATFORM_SIM_AICPU_DEVICE_TIME_FAST_H_
