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

#pragma once
#include <cstdint>

// Returns true if this thread should call aicpu_execute().
// Returns false if this thread should exit (dropped).
// logical_count: desired active threads (from runtime.aicpu_thread_num)
// total_launched: actual threads launched (PLATFORM_MAX_AICPU_THREADS_JUST_FOR_LAUNCH)
bool platform_aicpu_affinity_gate(int32_t logical_count, int32_t total_launched);

// Deterministic executor index assigned to the surviving thread by the gate.
// Range [0, ALLOWED_CPU_COUNT-1]; -1 if this thread was dropped or the gate
// was bypassed (ALLOWED_CPU_COUNT >= total_launched).
// Highest index (ALLOWED_CPU_COUNT-1) is the orchestrator slot; lower indices
// are scheduler slots in declaration order of ALLOWED_CPUS.
int32_t platform_aicpu_affinity_thread_idx();
