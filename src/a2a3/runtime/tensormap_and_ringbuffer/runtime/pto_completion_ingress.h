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

#ifndef SRC_A2A3_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_PTO_COMPLETION_INGRESS_H_
#define SRC_A2A3_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_PTO_COMPLETION_INGRESS_H_

#include <stdint.h>

#include "pto_constants.h"
#include "pto_task_id.h"

#define PTO2_COMPLETION_INGRESS_CAPACITY 4096u
#define PTO2_COMPLETION_INGRESS_MASK (PTO2_COMPLETION_INGRESS_CAPACITY - 1u)

static_assert(
    (PTO2_COMPLETION_INGRESS_CAPACITY & (PTO2_COMPLETION_INGRESS_CAPACITY - 1u)) == 0,
    "PTO2_COMPLETION_INGRESS_CAPACITY must be a power of two"
);

inline constexpr int32_t PTO2_MAX_COMPLETIONS_PER_TASK = 64;

#define PTO2_COMPLETION_ENGINE_SDMA 0u
#define PTO2_COMPLETION_ENGINE_ROCE 1u
#define PTO2_COMPLETION_ENGINE_URMA 2u
#define PTO2_COMPLETION_ENGINE_CCU 3u

#define PTO2_COMPLETION_TYPE_COUNTER 0

struct PTO2CompletionIngressEntry {
    volatile uint64_t seq;
    PTO2TaskId task_token;
    uint64_t addr;
    uint32_t expected_value;
    uint32_t engine;
    int32_t completion_type;
    uint32_t _pad[6];
};

static_assert(sizeof(PTO2CompletionIngressEntry) == PTO2_ALIGN_SIZE, "PTO2CompletionIngressEntry layout drift");

struct PTO2DeferredCompletionEntry {
    uint64_t addr;
    uint32_t expected_value;
    uint32_t engine;
    int32_t completion_type;
    uint32_t _pad;
};

static_assert(sizeof(PTO2DeferredCompletionEntry) == 24, "PTO2DeferredCompletionEntry layout drift");

struct PTO2DeferredCompletionIngressBuffer {
    alignas(PTO2_ALIGN_SIZE) volatile uint32_t count;
    volatile int32_t error_code;
    uint8_t _header_pad[PTO2_ALIGN_SIZE - sizeof(uint32_t) - sizeof(int32_t)];
    alignas(PTO2_ALIGN_SIZE) PTO2DeferredCompletionEntry entries[PTO2_MAX_COMPLETIONS_PER_TASK];
};

static_assert(
    sizeof(PTO2DeferredCompletionIngressBuffer) % PTO2_ALIGN_SIZE == 0,
    "PTO2DeferredCompletionIngressBuffer size must be cache-line aligned"
);

struct PTO2CompletionIngressQueue {
    alignas(PTO2_ALIGN_SIZE) volatile uint64_t head;
    uint8_t _head_pad[PTO2_ALIGN_SIZE - sizeof(uint64_t)];
    alignas(PTO2_ALIGN_SIZE) volatile uint64_t tail;
    uint8_t _tail_pad[PTO2_ALIGN_SIZE - sizeof(uint64_t)];
    alignas(PTO2_ALIGN_SIZE) PTO2CompletionIngressEntry entries[PTO2_COMPLETION_INGRESS_CAPACITY];
};

static_assert(
    sizeof(PTO2CompletionIngressQueue) % PTO2_ALIGN_SIZE == 0,
    "PTO2CompletionIngressQueue size must be cache-line aligned"
);

#endif  // SRC_A2A3_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_PTO_COMPLETION_INGRESS_H_
