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

#ifndef SRC_A2A3_PLATFORM_INCLUDE_AICPU_CPU_SIM_TASK_COOKIE_H_
#define SRC_A2A3_PLATFORM_INCLUDE_AICPU_CPU_SIM_TASK_COOKIE_H_

#include <cstdint>

// Publish the logical task cookie for a dispatched CPU-sim task.
void platform_set_cpu_sim_task_cookie(uint32_t core_id, uint32_t reg_task_id, uint64_t task_cookie);

#endif  // SRC_A2A3_PLATFORM_INCLUDE_AICPU_CPU_SIM_TASK_COOKIE_H_
