/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */
/**
 * MoE Combine All-to-All Kernel (Direct Store Version)
 *
 * This kernel implements the combine phase of distributed MoE:
 * Each card i sends recv[i][card_j] (expert_i's result for card_j) to card j,
 * then directly stores all received results to output (one expert per output row).
 *
 * Data flow:
 *   Phase 1 (stage-in):  recv[:][:][:COUNT][:] → scratch[my_rank][:][:][:]
 *   Phase 2 (barrier):   signal matrix + TWAIT cross-rank sync
 *   Phase 3 (store):     for expert_i in num_cards: copy scratch[expert_i][my_rank][:][:] to output[expert_i][:][:]
 *
 * Output layout:
 *   output[expert_i][token_t][:] = data from expert_i for this card, token t
 *
 * args layout:
 *   tensor(0) = recv_local       [num_cards][num_tokens][hidden_dim]
 *   tensor(1) = output_local     [num_cards][count][hidden_dim] - stores all experts' data
 *   tensor(2) = scratch          HCCL window buffer
 *   tensor(3) = scratch_print    Debug output buffer (Phase 1 stage-in mirror)
 *   scalar(0) = card_id          which card this is
 *   scalar(1) = num_cards        total number of cards
 *   scalar(2) = CommContext      device pointer for cross-card communication
 */

#include <cstdint>
#include <pto/pto-inst.hpp>
#include "pto/comm/comm_types.hpp"
#include "pto/comm/pto_comm_inst.hpp"
#include "platform_comm/comm_context.h"
#include "tensor.h"

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

// Configuration matching golden.py
static constexpr size_t NUM_TOKENS = 10;
static constexpr size_t HIDDEN_DIM = 16;
static constexpr size_t COUNT = 4;  // tokens to process per (card, expert) pair
static constexpr int kMaxSupportedCards = 16;

template <typename T>
AICORE inline __gm__ T *CommRemotePtr(__gm__ CommContext *ctx, __gm__ T *localPtr, int pe) {
    uint64_t localBase = ctx->windowsIn[ctx->rankId];
    uint64_t offset = (uint64_t)localPtr - localBase;
    return (__gm__ T *)(ctx->windowsIn[pe] + offset);
}

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    // Unpack tensors
    __gm__ Tensor *recv_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *output_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ Tensor *scratch_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ Tensor *scratch_print_tensor = reinterpret_cast<__gm__ Tensor *>(args[3]);

    // Unpack scalars
    int64_t card_id = static_cast<int64_t>(args[4]);
    int num_cards = static_cast<int>(args[5]);
    __gm__ CommContext *commCtx = reinterpret_cast<__gm__ CommContext *>(args[6]);

    // Get base pointers
    __gm__ float *recv = reinterpret_cast<__gm__ float *>(recv_tensor->buffer.addr) + recv_tensor->start_offset;
    __gm__ float *output = reinterpret_cast<__gm__ float *>(output_tensor->buffer.addr) + output_tensor->start_offset;
    __gm__ float *scratch = reinterpret_cast<__gm__ float *>(scratch_tensor->buffer.addr) + scratch_tensor->start_offset;
    __gm__ float *scratch_print = reinterpret_cast<__gm__ float *>(scratch_print_tensor->buffer.addr) + scratch_print_tensor->start_offset;

    // Signal area at tail of scratch: num_cards int32 slots
    // Must be placed AFTER all data slots to avoid corruption
    size_t total_data_size = num_cards * num_cards * NUM_TOKENS * HIDDEN_DIM;
    __gm__ int32_t *signal_base = reinterpret_cast<__gm__ int32_t *>(scratch + total_data_size);

    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<float, ShapeDyn, StrideDyn, pto::Layout::ND>;

    int my_rank = static_cast<int>(commCtx->rankId);

    if (num_cards <= 0 || num_cards > kMaxSupportedCards) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    // ------------------------------------------------------------------
    // Phase 1: stage-in — copy recv to scratch
    // This card's expert result for all cards (as destination)
    //
    //
    // For card_i with expert_id, copy recv[card_j][:][:] to scratch[expert_id][card_j][:][:]
    // ------------------------------------------------------------------
    for (int card_j = 0; card_j < num_cards; ++card_j) {
        for (size_t t = 0; t < COUNT; ++t) {
            // Source: recv[card_j][t][:HIDDEN_DIM] (expert_id's processed data from card_j)
            // recv layout: [num_cards][NUM_TOKENS][HIDDEN_DIM]
            // Base points to current (card_j, t), stride should keep access within current token
            ShapeDyn src_shape(1, 1, 1, 1, HIDDEN_DIM);
            StrideDyn src_stride(NUM_TOKENS * HIDDEN_DIM, NUM_TOKENS * HIDDEN_DIM,
                                 NUM_TOKENS * HIDDEN_DIM, HIDDEN_DIM, 1);
            Global srcG(recv + card_j * NUM_TOKENS * HIDDEN_DIM + t * HIDDEN_DIM,
                       src_shape, src_stride);

            // Destination: scratch[my_rank][card_j][t][:HIDDEN_DIM]
            // Offset = my_rank * (num_cards * NUM_TOKENS * HIDDEN_DIM)
            //        + card_j * (NUM_TOKENS * HIDDEN_DIM)
            //        + t * HIDDEN_DIM
            size_t dst_offset = my_rank * num_cards * NUM_TOKENS * HIDDEN_DIM
                              + card_j * NUM_TOKENS * HIDDEN_DIM
                              + t * HIDDEN_DIM;

            ShapeDyn dst_shape(1, 1, 1, 1, HIDDEN_DIM);
            StrideDyn dst_stride(num_cards * NUM_TOKENS * HIDDEN_DIM,
                                 num_cards * NUM_TOKENS * HIDDEN_DIM,
                                 NUM_TOKENS * HIDDEN_DIM, HIDDEN_DIM, 1);
            Global dstG(scratch + dst_offset,
                       dst_shape, dst_stride);
            Global dstG_print(scratch_print + dst_offset,
                             dst_shape, dst_stride);

            using TileType = pto::Tile<pto::TileType::Vec, float, 1, HIDDEN_DIM,
                                       pto::BLayout::RowMajor, -1, -1>;
            TileType tile(1, HIDDEN_DIM);
            TASSIGN(tile, 0);

            TLOAD(tile, srcG);
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            TSTORE(dstG, tile);
            TSTORE(dstG_print, tile);
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        }
    }
    pipe_barrier(PIPE_ALL);

    // ------------------------------------------------------------------
    // Phase 2: device barrier — each card notifies peers that its
    // recv[:][my_card] data is visible in scratch, then waits for all peers.
    // ------------------------------------------------------------------
    for (int peer = 0; peer < num_cards; ++peer) {
        if (peer == my_rank) continue;
        __gm__ int32_t *remote_signal = CommRemotePtr(commCtx, signal_base + my_rank, peer);
        pto::comm::Signal sig(remote_signal);
        pto::comm::TNOTIFY(sig, (int32_t)1, pto::comm::NotifyOp::AtomicAdd);
    }
    for (int peer = 0; peer < num_cards; ++peer) {
        if (peer == my_rank) continue;
        pto::comm::Signal sig(signal_base + peer);
        pto::comm::TWAIT(sig, (int32_t)1, pto::comm::WaitCmp::GE);
    }
    pipe_barrier(PIPE_ALL);

    // ------------------------------------------------------------------
    // Phase 3: direct store — copy each expert's data to output
    // Read scratch[expert_i][my_rank][t][:HIDDEN_DIM] from each expert i
    // and store to output[expert_i][t][:HIDDEN_DIM]
    //
    // For card_id with my_rank:
    //   output[expert_0][t][:] = scratch[expert_0][my_rank][t][:]
    //   output[expert_1][t][:] = scratch[expert_1][my_rank][t][:]
    //   etc.
    // ------------------------------------------------------------------
    for (int expert_i = 0; expert_i < num_cards; ++expert_i) {
        for (size_t t = 0; t < COUNT; ++t) {
            // Source: scratch[expert_i][my_rank][t][:HIDDEN_DIM]
            // Offset = expert_i * (num_cards * NUM_TOKENS * HIDDEN_DIM)
            //        + my_rank * (NUM_TOKENS * HIDDEN_DIM)
            //        + t * HIDDEN_DIM
            __gm__ float *src_base = (expert_i == my_rank) ? scratch :
                                     CommRemotePtr(commCtx, scratch, expert_i);
            size_t src_offset = expert_i * num_cards * NUM_TOKENS * HIDDEN_DIM
                              + my_rank * NUM_TOKENS * HIDDEN_DIM
                              + t * HIDDEN_DIM;

            ShapeDyn src_shape(1, 1, 1, 1, HIDDEN_DIM);
            StrideDyn src_stride(num_cards * NUM_TOKENS * HIDDEN_DIM,
                                 num_cards * NUM_TOKENS * HIDDEN_DIM,
                                 NUM_TOKENS * HIDDEN_DIM, HIDDEN_DIM, 1);
            Global srcG(src_base + src_offset, src_shape, src_stride);

            // Destination: output[expert_i][t][:HIDDEN_DIM]
            // Offset = expert_i * (COUNT * HIDDEN_DIM) + t * HIDDEN_DIM
            size_t dst_offset = expert_i * COUNT * HIDDEN_DIM + t * HIDDEN_DIM;

            ShapeDyn dst_shape(1, 1, 1, 1, HIDDEN_DIM);
            StrideDyn dst_stride(COUNT * HIDDEN_DIM, HIDDEN_DIM, HIDDEN_DIM, HIDDEN_DIM, 1);
            Global dstG(output + dst_offset, dst_shape, dst_stride);

            using TileType = pto::Tile<pto::TileType::Vec, float, 1, HIDDEN_DIM,
                                       pto::BLayout::RowMajor, -1, -1>;
            TileType tile(1, HIDDEN_DIM);
            TASSIGN(tile, 0);

            // Load from scratch
            TLOAD(tile, srcG);
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);

            // Store to output
            TSTORE(dstG, tile);
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        }
    }

    pipe_barrier(PIPE_ALL);
}
