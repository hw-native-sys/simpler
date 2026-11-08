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
#include <cstdint>

// Standalone AIV binary, independent of user kernel_entry and runtime kernels.
// 私有 AIV 触发器不依赖用户 kernel_entry，也不写完成记录。
extern "C" __global__[aicore] void simpler_ccu_trigger(uint64_t gate_va, uint64_t mask) {
    auto *gate = reinterpret_cast<volatile __gm__ uint64_t *>(gate_va);
    auto *address = reinterpret_cast<__gm__ void *>(gate_va);
    __asm__ __volatile__("" ::: "memory");
    dcci(address, cache_line_t::ENTIRE_DATA_CACHE);
    *gate = mask | (1ULL << 63);
    dcci(address, cache_line_t::ENTIRE_DATA_CACHE);
    dsb(DSB_DDR);
    pipe_barrier(PIPE_ALL);
    __asm__ __volatile__("" ::: "memory");
}
