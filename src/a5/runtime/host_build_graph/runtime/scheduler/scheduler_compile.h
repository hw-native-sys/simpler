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

// Device objects for AIC and AIV share a final ELF. Outlined helpers must have
// internal linkage so architecture-specific implementations are not coalesced.
#if defined(SIMPLER_AICORE_DEV_OUTLINE)
#define SCHEDULER_HEAVY_INLINE static __attribute__((noinline))
#else
#define SCHEDULER_HEAVY_INLINE inline
#endif
