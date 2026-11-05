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

#include <pto/cpu/target_config.hpp>

static_assert(
    pto::CPU_SIM_TARGET == pto::CpuSimTarget::A2A3 || pto::CPU_SIM_TARGET == pto::CpuSimTarget::A5,
    "simpler CPU kernels require a supported fixed target family"
);

// Every translation unit has the same fixed target; inline permits one export per linked image.
extern "C" __attribute__((used, visibility("default"))) inline const char *simpler_cpu_sim_target() {
    return pto::CPU_SIM_TARGET == pto::CpuSimTarget::A5 ? "a5sim" : "a2a3sim";
}
