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
 * Home for per-device knobs that simpler_aicpu_init latches at worker init
 * (from InitArgs) into resident-SO globals surviving every subsequent per-task
 * launch. Async-DMA slots may be republished after first-use provisioning; the
 * runtime otherwise consumes these read-only. They do NOT ride the per-run
 * KernelArgs or arena layout.
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

/**
 * Set the orchestration tensor-data wait timeout (ms). Latched once per device
 * by simpler_aicpu_init (from InitArgs.tensor_data_timeout_ms); read by
 * get_tensor_data / set_tensor_data when they block on a producer or its
 * consumers. 0 means "no override" — the wait keeps its compile-time
 * TENSOR_DATA_TIMEOUT_MS. Consumed by tensormap_and_ringbuffer only;
 * host_build_graph resolves tensor data on the host and never waits.
 */
void set_tensor_data_timeout_ms(int timeout_ms);

/** Get the tensor-data wait timeout override in ms (0 if unset). */
int get_tensor_data_timeout_ms();

/**
 * Set the device address of the per-device async-DMA workspace for one engine
 * kind (see DmaWorkspaceKind). Published by simpler_aicpu_init (from
 * InitArgs.dma_workspace_addr[]) into a resident-SO array; the scheduler
 * copies each slot into every core's GlobalContext, so kernels read it via
 * get_dma_workspace(args, kind). 0 means this Worker did not provision that
 * engine — the platform does not support it, or the Worker declined it. A
 * kernel that reads 0 is responsible for its own fallback; nothing rejects it
 * before launch. Out-of-range kinds are ignored.
 */
void set_dma_workspace_addr(int kind, unsigned long long addr);

/** Get the async-DMA workspace device address for one engine kind (0 if unavailable). */
unsigned long long get_dma_workspace_addr(int kind);

#ifdef __cplusplus
}
#endif

#endif  // PLATFORM_COMMON_AICPU_DEVICE_CONFIG_H_
