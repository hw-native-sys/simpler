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
#include <pto/pto-inst.hpp>
#ifdef PTO_CPUSTUB_HPP
#include <thread>
#endif
#include "intrinsic.h"
#include "tensor.h"

struct alignas(128) RendezvousFlag {
    volatile int64_t value;
    int64_t padding[15];
};

static_assert(sizeof(RendezvousFlag) == 128);

static __aicore__ void rendezvous(__gm__ int64_t *args, uint32_t lane) {
    auto *tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    auto *state = reinterpret_cast<__gm__ RendezvousFlag *>(
        reinterpret_cast<__gm__ int64_t *>(tensor->buffer.addr) + tensor->start_offset
    );
    const uint32_t mask = static_cast<uint32_t>(args[1]);
    state[lane].value = 1;
    dcci(&state[lane], cache_line_t::SINGLE_CACHE_LINE, dcci_dst_t::CACHELINE_OUT);
    dsb((mem_dsb_t)0);
    for (uint32_t poll = 0; poll < 10000000; ++poll) {
        bool arrived = true;
        for (uint32_t peer = 0; peer < 3; ++peer) {
            if (peer == lane || (mask & (1U << peer)) == 0) continue;
            dcci(&state[peer], cache_line_t::SINGLE_CACHE_LINE);
            dsb((mem_dsb_t)0);
            arrived = arrived && state[peer].value >= 1;
        }
        if (!arrived) {
#ifdef PTO_CPUSTUB_HPP
            std::this_thread::yield();
#endif
            continue;
        }
        if (lane == 1) {
            for (volatile uint32_t delay = 0; delay < 2048; ++delay) {}
        }
        state[lane].value = 2;
        dcci(&state[lane], cache_line_t::SINGLE_CACHE_LINE, dcci_dst_t::CACHELINE_OUT);
        dsb((mem_dsb_t)0);
        return;
    }
    state[lane].value = -1;
    dcci(&state[lane], cache_line_t::SINGLE_CACHE_LINE, dcci_dst_t::CACHELINE_OUT);
    dsb((mem_dsb_t)0);
}
