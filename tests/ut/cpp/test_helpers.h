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
 * Shared test helper utilities for C++ unit tests.
 *
 * Provides convenience functions that initialize internal data structures
 * from user-supplied buffers, avoiding direct field manipulation in tests.
 */
#pragma once

#include "pto_scheduler.h"

/**
 * Initialize a ReadyQueue with a caller-provided slot buffer and start sequence.
 *
 * Unlike pto2_ready_queue_init() which malloc's its own buffer and starts at 0,
 * this helper uses a stack-allocated buffer and supports arbitrary start sequences
 * (needed for sequence-wrap tests).
 */
inline void
test_ready_queue_init(PTO2ReadyQueue *queue, PTO2ReadyQueueSlot *slots, uint64_t capacity, int64_t start_seq = 0) {
    queue->slots = slots;
    queue->capacity = capacity;
    queue->mask = capacity - 1;
    queue->enqueue_pos.store(start_seq, std::memory_order_relaxed);
    queue->dequeue_pos.store(start_seq, std::memory_order_relaxed);
    for (uint64_t i = 0; i < capacity; i++) {
        int64_t pos = start_seq + (int64_t)i;
        uint64_t idx = (uint64_t)pos & (capacity - 1);
        slots[idx].sequence.store(pos, std::memory_order_relaxed);
        slots[idx].slot_state = nullptr;
    }
}
