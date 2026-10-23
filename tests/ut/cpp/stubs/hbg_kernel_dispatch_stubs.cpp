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
#include <cstdlib>
#include <cstdint>

#include "aicpu/platform_regs.h"
#include "host_build_graph/runtime.h"

// Failure-path tests run the production consumer and sim participant gate, but
// must never reach an executor or perform hardware initialization.
void prepare_kernel_aicpu_thread() {}

int32_t aicpu_execute_restored_graph(Runtime *) { std::abort(); }

static uint64_t dispatch_test_regs = 0;

void set_platform_regs(uint64_t regs) { dispatch_test_regs = regs; }

uint64_t get_platform_regs() { return dispatch_test_regs; }

namespace {
uint64_t swimlane_base = 0;
uint64_t rotation_table = 0;
uint64_t dep_gen_base = 0;
uint64_t pmu_base = 0;
uint64_t phase_base = 0;
bool swimlane_enabled = false;
bool dep_gen_enabled = false;
bool pmu_enabled = false;
}  // namespace

extern "C" void set_platform_chip_swimlane_base(uint64_t value) { swimlane_base = value; }
extern "C" uint64_t get_platform_chip_swimlane_base() { return swimlane_base; }
extern "C" void set_platform_chip_swimlane_aicore_rotation_table(uint64_t value) { rotation_table = value; }
extern "C" uint64_t get_platform_chip_swimlane_aicore_rotation_table() { return rotation_table; }
extern "C" void set_chip_swimlane_enabled(bool value) { swimlane_enabled = value; }
extern "C" bool is_chip_swimlane_enabled() { return swimlane_enabled; }
extern "C" void set_platform_dep_gen_base(uint64_t value) { dep_gen_base = value; }
extern "C" uint64_t get_platform_dep_gen_base() { return dep_gen_base; }
extern "C" void set_dep_gen_enabled(bool value) { dep_gen_enabled = value; }
extern "C" bool is_dep_gen_enabled() { return dep_gen_enabled; }
extern "C" void set_platform_pmu_base(uint64_t value) { pmu_base = value; }
extern "C" uint64_t get_platform_pmu_base() { return pmu_base; }
extern "C" void set_pmu_enabled(bool value) { pmu_enabled = value; }
extern "C" bool is_pmu_enabled() { return pmu_enabled; }
extern "C" void set_platform_phase_base(uint64_t value) { phase_base = value; }
uint64_t get_test_kernel_phase_base() { return phase_base; }
