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

#include "aicpu/sched_die_config.h"

namespace {
// Resident across launches like the other latched per-launch config; the entry
// overwrites it on every launch, so a stale value cannot survive into a run
// whose host published a different vector.
uint64_t g_sched_thread_die_bits = 0;
}  // namespace

extern "C" void set_sched_thread_die_bits(uint64_t bits) { g_sched_thread_die_bits = bits; }

extern "C" uint64_t get_sched_thread_die_bits() { return g_sched_thread_die_bits; }
