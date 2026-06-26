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

#include "common/platform_config.h"

#define SPIN_WAIT_HINT() ((void)0)

// Wall-clock budget (ms) of no task progress before the dispatch loop aborts
// with PTO2_ERROR_SCHEDULER_TIMEOUT. On real hardware this must sit *below* the
// STARS AICore op-execution timeout (PLATFORM_OP_EXECUTE_TIMEOUT_US, 3 s) so the
// AICPU detects the hang and flushes its diagnostics (tensor dump, in-flight
// partial output) before STARS reaps the op and poisons the context. Chain:
// this < op-exec < host stream-sync (platform_config.h). Trade-off: 2 s is
// shorter than the worst distributed-init / HCCL skew #897 sized 5 s for, so a
// slow distributed startup can false-latch; if that bites, raise this together
// with the op-exec / stream-sync timeouts. The sim build keeps the full 5 s (no
// STARS to race). The runtime consumes it as SCHEDULER_TIMEOUT_MS (see
// scheduler_types.h). Host may override this per run via
// PTO2_SCHEDULER_TIMEOUT_MS after validating the timeout ordering.
constexpr int32_t PLATFORM_SCHEDULER_TIMEOUT_MS = PLATFORM_ONBOARD_SCHEDULER_TIMEOUT_MS;

#endif  // PLATFORM_A2A3_AICPU_SPIN_HINT_H_
