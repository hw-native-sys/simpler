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
#include <chrono>
#include <thread>
#endif
#include "common/platform_config.h"
#include "intrinsic.h"
#include "tensor.h"

struct alignas(128) RendezvousFlag {
    volatile int64_t value;
    int64_t padding[15];
};

static_assert(sizeof(RendezvousFlag) == 128);

// How long a lane waits for the rest of its 1C2V before giving up.
//
// Bounded by elapsed time rather than by a poll count. A poll costs a
// different amount in the CPU-stub build than on silicon, and it also costs
// more when the machine is busy -- so a count that is generous when this case
// runs by itself can run out when it runs beside other work. It did: adding
// the cohort cases to the same suite left two of this case's 256 tasks
// reporting a timeout that a longer wait would not have hit.
constexpr uint64_t RENDEZVOUS_WAIT_MS = 2000;

#ifdef PTO_CPUSTUB_HPP
static uint64_t rendezvous_now_ticks() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
            .count()
    );
}
#else
static __aicore__ uint64_t rendezvous_now_ticks() { return get_sys_cnt(); }
#endif

static __aicore__ void rendezvous(__gm__ int64_t *args, uint32_t lane) {
    auto *tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    auto *state = reinterpret_cast<__gm__ RendezvousFlag *>(
        reinterpret_cast<__gm__ int64_t *>(tensor->buffer.addr) + tensor->start_offset
    );
    const uint32_t mask = static_cast<uint32_t>(args[1]);
    state[lane].value = 1;
    dcci(&state[lane], cache_line_t::SINGLE_CACHE_LINE, dcci_dst_t::CACHELINE_OUT);
    dsb((mem_dsb_t)0);
    const uint64_t deadline_ticks =
        RENDEZVOUS_WAIT_MS * (static_cast<uint64_t>(PLATFORM_PROF_SYS_CNT_FREQ) / 1000u);
    const uint64_t started = rendezvous_now_ticks();
    while (rendezvous_now_ticks() - started < deadline_ticks) {
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
