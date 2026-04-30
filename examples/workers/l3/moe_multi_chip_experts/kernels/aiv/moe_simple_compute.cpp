/*
 * Copyright (c) PyPTO Contributors.
 * Simple Compute Kernel for MoE
 *
 * Adds 1.0 to all elements in recv[:][:4][:]
 *
 * args layout:
 *   tensor(0) = recv [num_cards][NUM_TOKENS][HIDDEN_DIM]
 *   scalar(0) = unused (for compatibility)
 *   scalar(1) = unused (for compatibility)
 *   scalar(2) = unused (for compatibility)
 */

#include <cstdint>
#include <pto/pto-inst.hpp>
#include "tensor.h"

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

static constexpr size_t NUM_TOKENS = 10;
static constexpr size_t HIDDEN_DIM = 16;
static constexpr size_t COUNT = 4;
static constexpr int kMaxSupportedCards = 16;

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *recv_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ float *recv = reinterpret_cast<__gm__ float *>(recv_tensor->buffer.addr) + recv_tensor->start_offset;

    // Add 1.0 to first COUNT tokens for all cards
    // recv layout: [num_cards][NUM_TOKENS][HIDDEN_DIM]
    for (int card = 0; card < kMaxSupportedCards; ++card) {
        for (size_t t = 0; t < COUNT; ++t) {
            for (size_t d = 0; d < HIDDEN_DIM; ++d) {
                size_t offset = card * NUM_TOKENS * HIDDEN_DIM + t * HIDDEN_DIM + d;
                recv[offset] += 1.0f;
            }
        }
    }

    pipe_barrier(PIPE_ALL);
}
