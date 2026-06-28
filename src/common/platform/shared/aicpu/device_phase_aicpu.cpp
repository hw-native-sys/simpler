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
 * @file device_phase_aicpu.cpp
 * @brief Published phase-buffer base for the AICPU device-phase stamps.
 *
 * Mirrors the dump / l2_swimlane / pmu / scope_stats base setters: a plain
 * global living inside the AICPU SO, written once by the host before the inner
 * threads run (onboard: kernel.cpp from KernelArgs; sim: dlsym'd setter), read
 * by the stamp helpers in device_phase_aicpu.h. Deliberately NOT a C++
 * `thread_local` (per docs/dynamic-linking.md) so it survives the host↔dlopen'd
 * runtime SO boundary on sim; per-thread slotting is done via the affinity
 * pthread-key index, not here.
 */

#include "aicpu/device_phase_aicpu.h"

namespace {
uint64_t g_platform_phase_base = 0;
}  // namespace

extern "C" void set_platform_phase_base(uint64_t phase_data_base) { g_platform_phase_base = phase_data_base; }

extern "C" uint64_t get_platform_phase_base() { return g_platform_phase_base; }
