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
 * AIV (VECTOR) 50us busy-wait kernel — body matches the AIC variant exactly,
 * compiled here under the AIV toolchain. Used for both aiv0_kernel_id and
 * aiv1_kernel_id slots of the MixedKernels submit (one binary, two slots).
 */

#include <cstdint>
#include <pto/pto-inst.hpp>

using namespace pto;

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

constexpr uint64_t kSpinTicks = 50ULL * 50ULL;  // 50us at 50 MHz

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    (void)args;
    uint64_t start = static_cast<uint64_t>(get_sys_cnt());
    while ((static_cast<uint64_t>(get_sys_cnt()) - start) < kSpinTicks) {
        // busy spin
    }
}
