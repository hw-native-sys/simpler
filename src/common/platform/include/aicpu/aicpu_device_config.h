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
 * Per-device AICPU runtime config.
 *
 * Home for run-invariant per-device knobs that simpler_aicpu_init latches once
 * at worker init (from InitArgs) into resident-SO globals surviving every
 * subsequent per-task launch. The runtime consumes these read-only; they do
 * NOT ride the per-run KernelArgs or the arena layout. Add fields here as new
 * per-device config appears rather than threading it through the per-run path.
 *
 * Kept separate from platform_regs (which is strictly per-core register
 * addressing) so neither file accretes the other's concern.
 */

#ifndef PLATFORM_COMMON_AICPU_DEVICE_CONFIG_H_
#define PLATFORM_COMMON_AICPU_DEVICE_CONFIG_H_

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Set the ACL device ordinal. Latched once per device by simpler_aicpu_init
 * (from InitArgs.device_id) into this resident-SO global; the AICPU executor
 * reads it to make the staged orchestration SO filename unique per device so
 * paired dies sharing the preinstall filesystem never collide.
 */
void set_orch_device_id(int device_id);

/** Get the ACL device ordinal set for the current run (0 if unset). */
int get_orch_device_id();

/**
 * Set the AICPU scheduler no-progress watchdog timeout (ms). Latched once per
 * device by simpler_aicpu_init (from InitArgs.scheduler_timeout_ms); read by
 * the scheduler dispatch loop each run. 0 means "no override" — the scheduler
 * keeps its compile-time SCHEDULER_TIMEOUT_CYCLES.
 */
void set_scheduler_timeout_ms(int timeout_ms);

/** Get the scheduler watchdog timeout override in ms (0 if unset). */
int get_scheduler_timeout_ms();

#ifdef __cplusplus
}
#endif

#endif  // PLATFORM_COMMON_AICPU_DEVICE_CONFIG_H_
