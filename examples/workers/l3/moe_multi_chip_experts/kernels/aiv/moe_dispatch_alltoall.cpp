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
 * MoE Dispatch All-to-All Kernel
 *
 * This kernel implements the dispatch phase of distributed MoE:
 * Each card i sends send[i][expert_i] to all other cards, and receives
 * send[j][expert_i] from card j.
 *
 * Data flow:
 *   Phase 1 (stage-in):  send[expert_i][:][:] → my scratch slot
 *   Phase 2 (barrier):   signal matrix + TWAIT cross-rank sync
 *   Phase 3 (gather):    for card_j in num_cards: TLOAD(card_j_scratch), TSTORE(recv[card_j][:][:])
 *
 * args layout:
 *   tensor(0) = send_local    [num_experts][num_tokens][hidden_dim]
 *   tensor(1) = recv_local    [num_cards][num_tokens][hidden_dim]
 *   tensor(2) = scratch       HCCL window buffer
 *   scalar(0) = expert_id      which expert this card processes
 *   scalar(1) = num_cards      total number of cards
 *   scalar(2) = CommContext    device pointer for cross-card communication
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

// Configuration matching the in-test golden references
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
    __gm__ Tensor *send_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *recv_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ Tensor *scratch_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);

    // Unpack scalars
    int64_t expert_id = static_cast<int64_t>(args[3]);
    int num_cards = static_cast<int>(args[4]);
    __gm__ CommContext *commCtx = reinterpret_cast<__gm__ CommContext *>(args[5]);

    // Get base pointers
    __gm__ float *send = reinterpret_cast<__gm__ float *>(send_tensor->buffer.addr) + send_tensor->start_offset;
    __gm__ float *recv = reinterpret_cast<__gm__ float *>(recv_tensor->buffer.addr) + recv_tensor->start_offset;
    __gm__ float *scratch =
        reinterpret_cast<__gm__ float *>(scratch_tensor->buffer.addr) + scratch_tensor->start_offset;

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
    // Phase 1: stage-in — copy ALL experts' data to my scratch slot
    // Each card contributes ALL of its send[:] (all experts) to enable all-to-all
    //
    // Data layout in scratch: scratch[card_j][expert_i][:][:]
    // where card_j = my_rank (the card sending the data)
    //       expert_i = expert index (0..num_cards-1)
    //       t = token index (0..COUNT-1)
    //
    // This allows combine phase to access:
    //   "expert_i's data from card_j" at scratch[card_j][expert_i]
    // ------------------------------------------------------------------
    for (int expert_i = 0; expert_i < num_cards; ++expert_i) {
        for (size_t t = 0; t < COUNT; ++t) {
            // Load from send[expert_i][t][:HIDDEN_DIM] (ALL experts, not just expert_id)
            ShapeDyn send_shape(1, 1, 1, 1, HIDDEN_DIM);
            StrideDyn send_stride(NUM_TOKENS * HIDDEN_DIM, NUM_TOKENS * HIDDEN_DIM, HIDDEN_DIM, HIDDEN_DIM, 1);
            Global sendG(send + expert_i * NUM_TOKENS * HIDDEN_DIM + t * HIDDEN_DIM, send_shape, send_stride);

            // Store to scratch[my_rank][expert_i][t][:HIDDEN_DIM]
            // Index = my_rank * (num_cards * NUM_TOKENS * HIDDEN_DIM)
            //       + expert_i * (NUM_TOKENS * HIDDEN_DIM)
            //       + t * HIDDEN_DIM
            size_t scratch_offset =
                my_rank * num_cards * NUM_TOKENS * HIDDEN_DIM + expert_i * NUM_TOKENS * HIDDEN_DIM + t * HIDDEN_DIM;

            ShapeDyn scratch_shape(1, 1, 1, 1, HIDDEN_DIM);
            StrideDyn scratch_stride(
                num_cards * NUM_TOKENS * HIDDEN_DIM, num_cards * NUM_TOKENS * HIDDEN_DIM, NUM_TOKENS * HIDDEN_DIM,
                HIDDEN_DIM, 1
            );
            Global scratchG(scratch + scratch_offset, scratch_shape, scratch_stride);

            // Use tile for data movement
            using TileType = pto::Tile<pto::TileType::Vec, float, 1, HIDDEN_DIM, pto::BLayout::RowMajor, -1, -1>;
            TileType tile(1, HIDDEN_DIM);
            TASSIGN(tile, 0);

            TLOAD(tile, sendG);
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            TSTORE(scratchG, tile);
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        }
    }
    pipe_barrier(PIPE_ALL);

    // ------------------------------------------------------------------
    // Phase 2: device barrier — each card notifies peers that its
    // send[expert_i] data is visible in scratch, then waits for all peers.
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
    // Phase 3: gather — read send[j][expert_id] from each card j's scratch
    // and store to recv[card_j][:COUNT][:HIDDEN_DIM]
    //
    // For expert_id on this card, gather data from ALL cards:
    //   recv[card_j][:][:] = scratch[card_j][expert_id][:][:]
    // ------------------------------------------------------------------
    for (int card_j = 0; card_j < num_cards; ++card_j) {
        for (size_t t = 0; t < COUNT; ++t) {
            // Source: scratch[card_j][expert_id][t][:HIDDEN_DIM]
            // Offset = card_j * (num_cards * NUM_TOKENS * HIDDEN_DIM)
            //        + expert_id * (NUM_TOKENS * HIDDEN_DIM)
            //        + t * HIDDEN_DIM
            __gm__ float *src_base = (card_j == my_rank) ? scratch : CommRemotePtr(commCtx, scratch, card_j);
            size_t src_offset =
                card_j * num_cards * NUM_TOKENS * HIDDEN_DIM + expert_id * NUM_TOKENS * HIDDEN_DIM + t * HIDDEN_DIM;

            ShapeDyn src_shape(1, 1, 1, 1, HIDDEN_DIM);
            StrideDyn src_stride(
                num_cards * NUM_TOKENS * HIDDEN_DIM, num_cards * NUM_TOKENS * HIDDEN_DIM, NUM_TOKENS * HIDDEN_DIM,
                HIDDEN_DIM, 1
            );
            Global srcG(src_base + src_offset, src_shape, src_stride);

            // Destination: recv[card_j][t][:HIDDEN_DIM]
            ShapeDyn dst_shape(1, 1, 1, 1, HIDDEN_DIM);
            StrideDyn dst_stride(NUM_TOKENS * HIDDEN_DIM, NUM_TOKENS * HIDDEN_DIM, HIDDEN_DIM, HIDDEN_DIM, 1);
            Global dstG(recv + card_j * NUM_TOKENS * HIDDEN_DIM + t * HIDDEN_DIM, dst_shape, dst_stride);

            using TileType = pto::Tile<pto::TileType::Vec, float, 1, HIDDEN_DIM, pto::BLayout::RowMajor, -1, -1>;
            TileType tile(1, HIDDEN_DIM);
            TASSIGN(tile, 0);

            TLOAD(tile, srcG);
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            TSTORE(dstG, tile);
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        }
    }

    pipe_barrier(PIPE_ALL);
}
