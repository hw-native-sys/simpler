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

#include "aicpu/platform_regs.h"
#include "common/kernel_args.h"

void corrupt_kernel_arch_argument(KernelArgs &args, int fault) {
    if (fault == 0) args.force_simt_anchor = 1;
    else args.aicore_pmu_ring_addrs = 64;
}

void write_reg(uint64_t base, RegId reg, uint64_t value) {
    reg_store_release(reinterpret_cast<volatile uint32_t *>(base + reg_offset(reg)), static_cast<uint32_t>(value));
}
int32_t platform_deinit_aicore_regs(uint64_t) { return 0; }
