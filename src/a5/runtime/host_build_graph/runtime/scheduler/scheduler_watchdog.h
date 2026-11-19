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

#ifndef __aicore__
#define __aicore__
#endif

inline __aicore__ bool scheduler_watchdog_expired(uint64_t start, uint64_t now, uint64_t timeout_cycles) {
    const uint64_t elapsed = now - start;
    // Forward elapsed intervals occupy the lower half of the counter range.
    return timeout_cycles != 0 && elapsed < (uint64_t{1} << 63) && elapsed >= timeout_cycles;
}
