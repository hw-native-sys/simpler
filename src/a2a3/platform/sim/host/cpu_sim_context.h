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
 * @file cpu_sim_context.h
 * @brief Internal API for CPU simulation context lifecycle management
 *
 * Declares clear_cpu_sim_shared_storage() for use by DeviceRunner at
 * run start/finalize to reset shared CPU simulation state between runs.
 */

#pragma once

#include <cstdint>

/**
 * Free all entries in the CPU simulation shared storage map and reset the
 * pthread-backed per-thread execution context store.
 * Called by DeviceRunner::run() at start and DeviceRunner::finalize() at end.
 */
void clear_cpu_sim_shared_storage();

#ifdef __cplusplus
extern "C" {
#endif

void pto_cpu_sim_set_execution_context(uint32_t block_idx, uint32_t subblock_id, uint32_t subblock_dim);
void pto_cpu_sim_set_task_cookie(uint64_t task_cookie);
uint64_t platform_get_cpu_sim_task_cookie(uint32_t core_id, uint32_t reg_task_id);

#ifdef __cplusplus
}
#endif
