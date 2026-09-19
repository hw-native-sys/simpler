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
#include <pto/pto-inst.hpp>
#ifdef PTO_CPUSTUB_HPP
#include <chrono>
#include <thread>
#endif
#include "common/platform_config.h"
#include "tensor.h"
#ifndef __gm__
#define __gm__
#endif
#ifndef __aicore__
#define __aicore__ [aicore]  // NOLINT(whitespace/braces)
#endif
#include "intrinsic.h"

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    auto *tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
#ifdef PTO_CPUSTUB_HPP
    std::this_thread::sleep_for(std::chrono::milliseconds(args[1]));
#else
    uint64_t start = get_sys_cnt();
    uint64_t ticks = static_cast<uint64_t>(args[1]) * PLATFORM_PROF_SYS_CNT_FREQ / 1000;
    while (get_sys_cnt() - start < ticks) {}
#endif
    auto *out = reinterpret_cast<__gm__ int32_t *>(tensor->buffer.addr) + tensor->start_offset;
    *out = 7;
#ifndef PTO_CPUSTUB_HPP
    dcci(out, SINGLE_CACHE_LINE, CACHELINE_OUT);
#endif
}
