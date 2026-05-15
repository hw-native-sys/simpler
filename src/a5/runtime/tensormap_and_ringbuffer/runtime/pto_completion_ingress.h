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

#ifndef SRC_A5_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_PTO_COMPLETION_INGRESS_H_
#define SRC_A5_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_PTO_COMPLETION_INGRESS_H_

#include <stdint.h>

#include "pto_constants.h"
#include "pto_task_id.h"

#define COMPLETION_INGRESS_CAPACITY 4096u
#define COMPLETION_INGRESS_MASK (COMPLETION_INGRESS_CAPACITY - 1u)

static_assert(
    (COMPLETION_INGRESS_CAPACITY & (COMPLETION_INGRESS_CAPACITY - 1u)) == 0,
    "COMPLETION_INGRESS_CAPACITY must be a power of two"
);

inline constexpr int32_t MAX_COMPLETIONS_PER_TASK = 64;

#define COMPLETION_ENGINE_SDMA 0u
#define COMPLETION_ENGINE_ROCE 1u
#define COMPLETION_ENGINE_URMA 2u
#define COMPLETION_ENGINE_CCU 3u

#define COMPLETION_TYPE_COUNTER 0

struct CompletionIngressEntry {
    volatile uint64_t seq;
    PTO2TaskId task_token;
    uint64_t addr;
    uint32_t expected_value;
    uint32_t engine;
    int32_t completion_type;
    uint32_t _pad[6];
};

static_assert(sizeof(CompletionIngressEntry) == PTO2_ALIGN_SIZE, "CompletionIngressEntry layout drift");

struct DeferredCompletionEntry {
    uint64_t addr;
    uint32_t expected_value;
    uint32_t engine;
    int32_t completion_type;
    uint32_t _pad;
};

static_assert(sizeof(DeferredCompletionEntry) == 24, "DeferredCompletionEntry layout drift");

struct alignas(PTO2_ALIGN_SIZE) DeferredCompletionIngressBuffer {
    volatile uint32_t count;
    volatile int32_t error_code;
    DeferredCompletionEntry entries[MAX_COMPLETIONS_PER_TASK];
};

static_assert(
    sizeof(DeferredCompletionIngressBuffer) % PTO2_ALIGN_SIZE == 0,
    "DeferredCompletionIngressBuffer size must preserve array element cache-line boundaries"
);

struct CompletionIngressQueue {
    alignas(PTO2_ALIGN_SIZE) volatile uint64_t head;
    uint8_t _head_pad[PTO2_ALIGN_SIZE - sizeof(uint64_t)];
    alignas(PTO2_ALIGN_SIZE) volatile uint64_t tail;
    uint8_t _tail_pad[PTO2_ALIGN_SIZE - sizeof(uint64_t)];
    alignas(PTO2_ALIGN_SIZE) CompletionIngressEntry entries[COMPLETION_INGRESS_CAPACITY];
};

static_assert(
    sizeof(CompletionIngressQueue) % PTO2_ALIGN_SIZE == 0,
    "CompletionIngressQueue size must be cache-line aligned"
);

#endif  // SRC_A5_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_PTO_COMPLETION_INGRESS_H_
