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

#ifdef PTO_CPUSTUB_HPP
#include <chrono>
#endif

#include "common/platform_config.h"

struct DeviceGateWord {
    volatile int64_t value;
    int64_t padding[15];
};
static_assert(sizeof(DeviceGateWord) == 128);

static __aicore__ uint64_t device_gate_ticks() {
#ifdef PTO_CPUSTUB_HPP
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
            .count()
    );
#else
    return get_sys_cnt();
#endif
}

static __aicore__ bool wait_device_gate(uint64_t address) {
    if (address == 0) return true;
    auto *gate = reinterpret_cast<__gm__ DeviceGateWord *>(address);
    gate[0].value = 1;
    dcci(&gate[0], cache_line_t::SINGLE_CACHE_LINE, dcci_dst_t::CACHELINE_OUT);
    dsb((mem_dsb_t)0);
#ifdef PTO_CPUSTUB_HPP
    constexpr uint64_t kDeadline = 10ULL * 1000000000;
#else
    constexpr uint64_t kDeadline = 10ULL * PLATFORM_PROF_SYS_CNT_FREQ;
#endif
    const uint64_t started = device_gate_ticks();
    while (device_gate_ticks() - started < kDeadline) {
        dcci(&gate[1], cache_line_t::SINGLE_CACHE_LINE);
        dsb((mem_dsb_t)0);
        if (gate[1].value == 1) return true;
    }
    gate[0].value = -1;
    dcci(&gate[0], cache_line_t::SINGLE_CACHE_LINE, dcci_dst_t::CACHELINE_OUT);
    dsb((mem_dsb_t)0);
    return false;
}
