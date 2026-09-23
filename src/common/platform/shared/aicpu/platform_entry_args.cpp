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
#include "aicpu/platform_entry_args.h"

#include "aicpu/platform_aicpu_affinity.h"  // MAX_GATE_THREADS

namespace {
// One slot per gate survivor. Plain objects, because each slot has exactly one
// writer and one reader and they are the same thread — see the header for why a
// shared slot cannot be made correct by making it atomic.
PlatformEntryArgs g_entry_args[MAX_GATE_THREADS];

bool owns_slot(int32_t exec_idx) { return exec_idx >= 0 && exec_idx < MAX_GATE_THREADS; }
}  // namespace

void set_platform_entry_args(int32_t exec_idx, const PlatformEntryArgs &view) {
    if (!owns_slot(exec_idx)) return;
    g_entry_args[exec_idx] = view;
}

PlatformEntryArgs get_platform_entry_args(int32_t exec_idx) {
    if (!owns_slot(exec_idx)) return PlatformEntryArgs{};
    return g_entry_args[exec_idx];
}
