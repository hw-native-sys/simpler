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

/**
 * Per-launch carrier for KernelArgs::sched_thread_die_bits.
 *
 * The AICPU entry receives KernelArgs, the scheduler receives only Runtime, and
 * the die vector belongs to neither: it is a5-only, so it cannot live in the
 * host_build_graph Runtime that a2a3 shares. `simpler_aicpu_exec` latches it
 * here for the scheduler's cold path to read, the same shape the profiling
 * enable bits use.
 *
 * Written once per launch by the entry before any scheduler thread reaches
 * cluster assignment, then read-only, so no synchronisation is needed.
 */

// 0 means no scheduler slot's die is known; ownership stays round-robin.
extern "C" void set_sched_thread_die_bits(uint64_t bits);
extern "C" uint64_t get_sched_thread_die_bits();
