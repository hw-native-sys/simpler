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
 * Combine kernel — two-phase pipeline within a single AIV task:
 *
 *   push:   for each (dst, e):
 *             n       = pub_counts[dst][my_rank][e]
 *             src_off = sum_{s<dst} pub_counts[s][my_rank][e]
 *             for row in [0, n):
 *               r = recv_idx_out[e * R + src_off + row]   // = t * TOPK + k
 *               TPUT recv_y[e * R + src_off + row, :]
 *                 to peer dst's routed_y_buf[r * D : (r+1) * D]
 *           then combine_done barrier.
 *   reduce: reduce_sum along the TOPK axis of routed_y_buf into
 *           routed_y[T, D] FP32.
 *
 *   Inputs:
 *     recv_y          BF16  [L, R, D]   (local_expert OUTPUT_EXISTING)
 *     recv_idx_out    INT32 [L, R]      (dispatch OUTPUT_EXISTING)
 *   Output:
 *     routed_y        FP32  [T, D]      (verification ground truth)
 *   Scratch:
 *     window slot — pub_counts (read-only) / routed_y_buf (push dest) /
 *     combine_done_sig
 *
 * Single-write invariant: dispatch routes are injective (src, t, k) →
 * (dst, e, slot); combine reverses one-to-one. Each routed_y_buf[t, k]
 * slot at a given dst is written by AT MOST one sender per call.
 *
 * recv_count_out is intentionally NOT consumed here — the (dst, e) slab
 * length comes from pub_counts[dst][me][e], which is what dispatch left
 * in the window.
 */

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

#include <cstdint>

#include <pto/pto-inst.hpp>
#include "pto/comm/comm_types.hpp"
#include "pto/comm/pto_comm_inst.hpp"
#include "platform_comm/comm_context.h"
#include "tensor.h"

using namespace pto;

// Demo dimensions — must match dispatch.cpp / main.py.
static constexpr int N = 2;
static constexpr int T = 16;
static constexpr int TOPK = 2;
static constexpr int D = 4096;
static constexpr int L = 8;
static constexpr int R = 32;
static constexpr int W_PAD = 8;
static constexpr int IDX_PAD = 8;

// Window offsets — must mirror dispatch.cpp.
static constexpr int kPubCountsBytes = N * N * L * 4;  // N*N*L INT32
static constexpr int kSignalBytes = 64;
static constexpr int kRecvXBytes = L * R * D * 2;
static constexpr int kRecvWBytes = L * R * W_PAD * 4;
static constexpr int kRecvIdxBytes = L * R * IDX_PAD * 4;
static constexpr int kRoutedYBufBytes = T * TOPK * D * 2;

static constexpr int kOffPubCounts = 0;
static constexpr int kOffCountDone = kOffPubCounts + kPubCountsBytes;
static constexpr int kOffRecvX = kOffCountDone + kSignalBytes;
static constexpr int kOffRecvW = kOffRecvX + kRecvXBytes;
static constexpr int kOffRecvIdx = kOffRecvW + kRecvWBytes;
static constexpr int kOffDataDone = kOffRecvIdx + kRecvIdxBytes;
static constexpr int kOffRoutedYBuf = kOffDataDone + kSignalBytes;
static constexpr int kOffCombineDone = kOffRoutedYBuf + kRoutedYBufBytes;

template <typename T_>
AICORE inline __gm__ T_ *CommRemotePtr(__gm__ CommContext *ctx, __gm__ T_ *local_ptr, int peer_rank) {
    uint64_t local_base = ctx->windowsIn[ctx->rankId];
    uint64_t offset = reinterpret_cast<uint64_t>(local_ptr) - local_base;
    return reinterpret_cast<__gm__ T_ *>(ctx->windowsIn[peer_rank] + offset);
}

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *recv_y_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *recv_idx_out_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ Tensor *routed_y_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ Tensor *scratch_tensor = reinterpret_cast<__gm__ Tensor *>(args[3]);
    int nranks = static_cast<int>(args[4]);
    __gm__ CommContext *comm_ctx = reinterpret_cast<__gm__ CommContext *>(args[5]);

    if (nranks != N) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    __gm__ bfloat16_t *recv_y =
        reinterpret_cast<__gm__ bfloat16_t *>(recv_y_tensor->buffer.addr) + recv_y_tensor->start_offset;
    __gm__ int32_t *recv_idx_out =
        reinterpret_cast<__gm__ int32_t *>(recv_idx_out_tensor->buffer.addr) + recv_idx_out_tensor->start_offset;
    __gm__ float *routed_y =
        reinterpret_cast<__gm__ float *>(routed_y_tensor->buffer.addr) + routed_y_tensor->start_offset;

    __gm__ uint8_t *scratch_base =
        reinterpret_cast<__gm__ uint8_t *>(scratch_tensor->buffer.addr) + scratch_tensor->start_offset * sizeof(float);
    __gm__ int32_t *pub_counts_local = reinterpret_cast<__gm__ int32_t *>(scratch_base + kOffPubCounts);
    __gm__ bfloat16_t *routed_y_buf_local = reinterpret_cast<__gm__ bfloat16_t *>(scratch_base + kOffRoutedYBuf);
    __gm__ int32_t *combine_done_sig_local = reinterpret_cast<__gm__ int32_t *>(scratch_base + kOffCombineDone);

    int my_rank = static_cast<int>(comm_ctx->rankId);

    // ------------------------------------------------------------------
    // Tile types (all use STATIC valid rows/cols — TCVT seems to interact
    // poorly with `Tile<..., -1, -1>` dynamic-valid form here).
    // ------------------------------------------------------------------
    using RowBfG = GlobalTensor<bfloat16_t, Shape<1, 1, 1, 1, D>, Stride<D, D, D, D, 1>>;
    using RowFpG = GlobalTensor<float, Shape<1, 1, 1, 1, D>, Stride<D, D, D, D, 1>>;
    using RowBfTile = Tile<TileType::Vec, bfloat16_t, 1, D, BLayout::RowMajor, 1, D>;
    using RowFpTile = Tile<TileType::Vec, float, 1, D, BLayout::RowMajor, 1, D>;

    // routed_y_buf and combine_done_sig start zero by virtue of HCCL's window
    // zero-init at allocation time. We deliberately do NOT clear them per
    // call here: a per-call zero-init would race with the peer's push-phase
    // TPUT (peer pushes into our routed_y_buf; our zero-init can clobber an
    // already-arrived push). Multi-step decode would need a cross-rank
    // barrier between zero-init and the push phase to make the clear safe.

    // ------------------------------------------------------------------
    // push: TPUT recv_y rows to dst's routed_y_buf[t, k, :].
    // ------------------------------------------------------------------
    RowBfTile push_tile;
    TASSIGN(push_tile, 0x20000);

    for (int dst = 0; dst < N; ++dst) {
        for (int e = 0; e < L; ++e) {
            int n = pub_counts_local[(dst * N + my_rank) * L + e];
            if (n == 0) continue;

            int src_off = 0;
            for (int s = 0; s < dst; ++s) {
                src_off += pub_counts_local[(s * N + my_rank) * L + e];
            }

            for (int row = 0; row < n; ++row) {
                int idx_lin = e * R + src_off + row;
                int r = recv_idx_out[idx_lin];

                __gm__ bfloat16_t *src_row = recv_y + idx_lin * D;
                __gm__ bfloat16_t *dst_row_local = routed_y_buf_local + r * D;
                __gm__ bfloat16_t *dst_row_remote = CommRemotePtr(comm_ctx, dst_row_local, dst);

#if defined(__CPU_SIM)
                for (int i = 0; i < D; ++i) {
                    dst_row_remote[i] = src_row[i];
                }
#else
                RowBfG src_g(src_row);
                RowBfG dst_g(dst_row_remote);
                pto::comm::TPUT(dst_g, src_g, push_tile);
#endif
            }
        }
    }
    pipe_barrier(PIPE_ALL);

    // combine_done barrier — same form as dispatch's data_done.
    for (int peer = 0; peer < N; ++peer) {
        if (peer == my_rank) continue;
        __gm__ int32_t *remote_done = CommRemotePtr(comm_ctx, combine_done_sig_local + my_rank, peer);
        pto::comm::Signal sig(remote_done);
        pto::comm::TNOTIFY(sig, (int32_t)1, pto::comm::NotifyOp::AtomicAdd);
    }
    for (int src = 0; src < N; ++src) {
        if (src == my_rank) continue;
        pto::comm::Signal sig(combine_done_sig_local + src);
        pto::comm::TWAIT(sig, (int32_t)1, pto::comm::WaitCmp::GE);
    }
    pipe_barrier(PIPE_ALL);

    // ------------------------------------------------------------------
    // reduce: reduce_sum along TOPK -> routed_y[T, D] FP32.
    //
    // For each token t:
    //   acc_fp32 = 0
    //   for k in 0..TOPK-1:
    //     bf16 = TLOAD routed_y_buf[(t * TOPK + k) * D : ...]
    //     acc_fp32 += cast(bf16, fp32)
    //   TSTORE routed_y[t * D : (t+1) * D] <- acc_fp32   (FP32 directly)
    // ------------------------------------------------------------------
    RowFpTile acc_tile;
    RowBfTile add_bf_tile;
    RowFpTile add_fp_tile;
    // Reuse push-phase UB slots — barriers above ensure they have drained.
    TASSIGN(acc_tile, 0x0);
    TASSIGN(add_bf_tile, 0x10000);
    TASSIGN(add_fp_tile, 0x20000);

    for (int t = 0; t < T; ++t) {
        TEXPANDS(acc_tile, 0.0f);
        pipe_barrier(PIPE_V);

        for (int k = 0; k < TOPK; ++k) {
            int r = t * TOPK + k;
            __gm__ bfloat16_t *src_row = routed_y_buf_local + r * D;
            RowBfG src_g(src_row);

            TLOAD(add_bf_tile, src_g);
            pipe_barrier(PIPE_ALL);

            TCVT(add_fp_tile, add_bf_tile, RoundMode::CAST_RINT);
            pipe_barrier(PIPE_V);
            TADD(acc_tile, acc_tile, add_fp_tile);
            pipe_barrier(PIPE_V);
        }

        RowFpG out_g(routed_y + t * D);
        TSTORE(out_g, acc_tile);
        pipe_barrier(PIPE_ALL);
    }
    pipe_barrier(PIPE_ALL);
}
