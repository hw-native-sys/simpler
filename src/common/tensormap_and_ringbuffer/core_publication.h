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

#include <atomic>
#include <cstdint>
#include "spin_hint.h"
#include "common/memory_barrier.h"

namespace simpler::tmr {
// Task tokens, early-dispatch doorbells and EXIT share DATA_MAIN_BASE.
// Close admission per core and drain its current writer before sending EXIT.
class alignas(64) CorePublication {
    static constexpr uint32_t WRITING = 1;
    static constexpr uint32_t CLOSED = 2;
    std::atomic<uint32_t> state{0};

public:
    void reset() { state.store(0, std::memory_order_relaxed); }

    template <typename Write>
    bool publish(Write write) {
        uint32_t expected = 0;
        while (!state.compare_exchange_weak(expected, WRITING, std::memory_order_acquire)) {
            if ((expected & CLOSED) != 0) return false;
            expected = 0;
            SPIN_WAIT_HINT();
        }
        write();
        wmb();  // Complete the device store before a different AICPU can send EXIT.
        state.fetch_and(~WRITING, std::memory_order_release);
        return true;
    }

    void close() {
        if ((state.fetch_or(CLOSED, std::memory_order_acq_rel) & WRITING) != 0) {
            while ((state.load(std::memory_order_acquire) & WRITING) != 0) {
                SPIN_WAIT_HINT();
            }
        }
    }
};
}  // namespace simpler::tmr
