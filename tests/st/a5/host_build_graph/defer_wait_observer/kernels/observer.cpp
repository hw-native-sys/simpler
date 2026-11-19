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

#include "scheduler/scheduler_async_control.h"
#include "tensor.h"

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    const auto *counter_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    const auto *post_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    const auto *result_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    auto *counters = reinterpret_cast<__gm__ uint32_t *>(counter_tensor->buffer.addr) + counter_tensor->start_offset;
    auto *posts = reinterpret_cast<__gm__ uint64_t *>(post_tensor->buffer.addr) + post_tensor->start_offset;
    auto *results = reinterpret_cast<__gm__ int32_t *>(result_tensor->buffer.addr) + result_tensor->start_offset;
    const uint32_t counter_count = counter_tensor->shapes[0];
    const uint32_t post_count = post_tensor->shapes[0];
#if defined(__DAV_CUBE__)
    constexpr uint32_t observer_mask = 1;
#else
    constexpr uint32_t observer_mask = 2;
#endif
    const SchedulerAsyncCompletionObserver observer;
    for (uint32_t row = 0; row < counter_count; ++row) {
        const CompletionToken token{
            reinterpret_cast<uint64_t>(counters + 2 * row), counters[2 * row + 1], COMPLETION_ENGINE_SDMA,
            COMPLETION_TYPE_COUNTER, 0
        };
        const auto result = observer(token, observer_mask);
        results[row] = result.error_code == 0 ? static_cast<int32_t>(result.state) : -1;
    }
    for (uint32_t row = 0; row < post_count; ++row) {
        const CompletionToken token{
            reinterpret_cast<uint64_t>(posts + 2 * row), 0, COMPLETION_ENGINE_SDMA, COMPLETION_TYPE_SDMA_EVENT_RECORD,
            posts[2 * row + 1]
        };
        const auto result = observer(token, observer_mask);
        results[counter_count + row] = result.error_code == 0 ? static_cast<int32_t>(result.state) : -1;
    }
#ifndef PTO_CPUSTUB_HPP
    for (uint32_t word = 0; word < counter_count + post_count; word += 16)
        dcci(results + word, cache_line_t::SINGLE_CACHE_LINE, dcci_dst_t::CACHELINE_OUT);
    dsb((mem_dsb_t)0);
#endif
}
