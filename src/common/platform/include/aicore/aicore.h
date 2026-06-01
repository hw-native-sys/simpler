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
 * @file aicore.h
 * @brief AICore Platform Abstraction Layer
 *
 * Provides unified AICore qualifiers and macros for both real hardware
 * and simulation environments. Uses conditional compilation to select
 * the appropriate implementation.
 *
 * Platform Support (same shape for both arches):
 * - onboard: Real Ascend hardware with CANN compiler
 * - sim: Host-based simulation using standard C++
 */

#ifndef SRC_COMMON_PLATFORM_INCLUDE_AICORE_AICORE_H_
#define SRC_COMMON_PLATFORM_INCLUDE_AICORE_AICORE_H_

// =============================================================================
// Common Memory Qualifiers (All Platforms)
// =============================================================================

#include "common/qualifier.h"

// =============================================================================
// Platform-Specific Definitions
// =============================================================================
// Platform-specific macros (__aicore__, dcci) are defined in inner_kernel.h.
// The build system selects the correct implementation based on platform:
//   src/{a2a3,a5}/platform/onboard/aicore/inner_kernel.h (real hardware)
//   src/{a2a3,a5}/platform/sim/aicore/inner_kernel.h     (simulation)

#include "inner_kernel.h"

#endif  // SRC_COMMON_PLATFORM_INCLUDE_AICORE_AICORE_H_
