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
 * @file spin_hint.h
 * @brief Platform-specific spin-wait hint for AICPU (real hardware)
 *
 * On real Ascend hardware, AICPU runs on dedicated ARM A55 cores with sufficient
 * resources. No spin-wait hint is needed — the macro expands to a no-op.
 */

#ifndef PLATFORM_A2A3_AICPU_SPIN_HINT_H_
#define PLATFORM_A2A3_AICPU_SPIN_HINT_H_

#include <cstdint>

#define SPIN_WAIT_HINT() ((void)0)

// Consecutive idle scheduler iterations (no task progress) before the dispatch
// loop aborts with PTO2_ERROR_SCHEDULER_TIMEOUT. On real hardware each idle
// iteration is a cheap no-op spin, so this bounds a genuine deadlock. The
// runtime consumes it as MAX_IDLE_ITERATIONS (see scheduler_types.h).
constexpr int32_t PLATFORM_MAX_IDLE_ITERATIONS = 800000;

#endif  // PLATFORM_A2A3_AICPU_SPIN_HINT_H_
