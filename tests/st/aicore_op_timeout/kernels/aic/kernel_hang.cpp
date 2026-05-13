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
 * AIC kernel that never returns. The STARS op-execution watchdog
 * (PLATFORM_OP_EXECUTE_TIMEOUT_US, ~1 s) must reap it; this kernel exists
 * to exercise the 3-layer host timeout chain added in PR #718.
 */

#include <cstdint>

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]  // NOLINT(whitespace/braces)
#endif

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    // Touch args[0] once so the compiler cannot prove the parameter dead and
    // drop the call frame; the volatile sink keeps the load alive across
    // optimisation. The kernel then spins forever — STARS reaps it.
    volatile int64_t sink = args[0];
    (void)sink;
    while (true) {}
}
