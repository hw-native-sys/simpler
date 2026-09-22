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
 * @brief Platform-specific spin-wait policy for AICPU (real hardware)
 *
 * On real Ascend hardware, AICPU runs on dedicated ARM A55 cores with sufficient
 * resources. No spin-wait hint is needed — the macro expands to a no-op.
 */

#pragma once

#include <cstdint>

#include "common/platform_config.h"

#define SPIN_WAIT_HINT() ((void)0)

constexpr int32_t PLATFORM_TENSOR_DATA_WAIT_TIMEOUT_MS = PLATFORM_ONBOARD_TENSOR_DATA_WAIT_TIMEOUT_MS;

// Onboard, the tensor-data wait reaps before the scheduler no-progress budget,
// so a hung producer latches code 8 naming the stuck tensor rather than a
// generic scheduler timeout. Simulation deliberately inverts this order: a
// legitimately slow sim kernel needs the larger budget, and independent task
// completions keep the scheduler watchdog fed meanwhile.
static_assert(
    PLATFORM_TENSOR_DATA_WAIT_TIMEOUT_MS < PLATFORM_SCHEDULER_TIMEOUT_MS,
    "onboard tensor-data wait must stay below the scheduler no-progress budget"
);

// The no-progress budget PLATFORM_SCHEDULER_TIMEOUT_MS is not defined here: it
// is one value across every platform variant and lives in platform_config.h,
// which the host also reads for timeout-ordering validation. On real hardware
// it must sit below the STARS AICore op-execution timeout
// (PLATFORM_OP_EXECUTE_TIMEOUT_US, 45 s) so the AICPU detects the hang and
// flushes its diagnostics (args dump, in-flight partial output) before STARS
// reaps the op and poisons the context. Chain: scheduler < op-exec < host
// stream-sync, all three in platform_config.h.
