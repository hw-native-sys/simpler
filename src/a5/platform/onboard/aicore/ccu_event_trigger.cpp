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
#include "../urma_queue_command.h"

extern "C" __global__[aicore] void
simpler_ccu_event_trigger(uint64_t gate_address, uint64_t value, uint64_t result_address) {
    auto *gate = reinterpret_cast<volatile __gm__ uint64_t *>(gate_address);
    auto *cache_line = reinterpret_cast<__gm__ void *>(gate_address);
    dcci(cache_line, cache_line_t::ENTIRE_DATA_CACHE);
    uint64_t result = *gate;
    if (value != UINT64_MAX) {
        *gate = value;
        dcci(cache_line, cache_line_t::ENTIRE_DATA_CACHE);
        dsb(DSB_DDR);
        result = *gate;
    }
    if (result_address != 0) {
        *reinterpret_cast<volatile __gm__ uint64_t *>(result_address) = result;
        dcci(reinterpret_cast<__gm__ void *>(result_address), cache_line_t::ENTIRE_DATA_CACHE);
    }
    dsb(DSB_DDR);
    pipe_barrier(PIPE_ALL);
}

namespace {
struct AivQueueMemory {
    static inline[aicore] bool read(uint64_t address, uint32_t &word) {
        word = static_cast<uint32_t>(__builtin_cce_ld_dev(reinterpret_cast<__gm__ uint32_t *>(address), 0));
        return true;
    }
    static inline[aicore] bool write(uint64_t address, uint32_t word) {
        __builtin_cce_st_dev(word, reinterpret_cast<__gm__ uint32_t *>(address), 0);
        dsb(DSB_DDR);
        return true;
    }
};
}  // namespace

// HCCL's AIV channel exports its CQE and doorbell in the device's address space.
extern "C" __global__[aicore] void simpler_urma_queue_control(
    uint64_t action, uint64_t address, uint64_t depth, uint64_t completed, uint64_t target, uint64_t result_address
) {
    using namespace simpler::urma;
    const QueueCommand command{static_cast<QueueAction>(action), address, depth, completed, target, result_address};
    AivQueueMemory::write(result_address, static_cast<uint32_t>(execute_queue_command(AivQueueMemory{}, command)));
}
