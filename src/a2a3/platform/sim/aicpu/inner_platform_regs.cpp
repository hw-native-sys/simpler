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
 * @file inner_platform_regs.cpp
 * @brief Variant-specific platform_regs hooks for simulation (a2a3sim)
 *
 * a2a3 sim and onboard share the same register layout, so read_reg /
 * write_reg live in the shared src/aicpu/platform_regs.cpp. This file
 * exists only for the deinit-timeout split where sim wants a much wider
 * budget than onboard — see platform_regs.h for the rationale.
 */

#include <cstdint>
#include "aicpu/platform_regs.h"
#include "common/platform_config.h"

// 10 s budget on sim — AICore is a host CPU thread and "no response" can
// just mean the OS scheduler hasn't given it a slice on a CPU-starved CI
// runner. See declaration in platform_regs.h for the design rationale.
uint64_t inner_get_deinit_timeout_ticks() { return 10 * PLATFORM_PROF_SYS_CNT_FREQ; }
