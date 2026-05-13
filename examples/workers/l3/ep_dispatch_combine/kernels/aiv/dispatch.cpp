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
 * EP dispatch kernel — 2-rank demo. Publishes per-token routing payload
 * (x BF16 / weight FP32 / idx INT32) to peer ranks' receive areas keyed by
 * `(local_expert, slot)`, where slot is computed from a globally consistent
 * pub_counts table. Idx encodes `r = t * TOPK + k` so combine can later
 * address routed_y_buf[t, k, :] directly.
 *
 * Pipeline:
 *   histogram     (scalar)   build local histogram + (dst, loc_e)-sorted route
 *                            table from indices via scalar GM reads
 *   publish       (TNOTIFY)  publish full send_counts table to every peer via
 *                            TNOTIFY(AtomicAdd) + count_done barrier
 *   prefix_sum    (scalar)   local prefix sums over global pub_counts (no comm);
 *                            writes recv_count_out[e] = sum_s pub_counts[s][me][e]
 *   payload_push  (TPUT)     for each route: TPUT three independent payload tiles
 *                            (x BF16 / weight FP32 / idx INT32) to peer's
 *                            recv_x[loc_e][slot, :] / recv_w[...] / recv_idx[...]
 *                            + data_done barrier
 *   stage_out     (TLOAD/TSTORE) stage out window -> host outputs.
 *                            recv_x_out is [L, R, D] (per-row 1xD copy);
 *                            recv_w_out is [L, R]    (TROWSUM over [L,R,W_PAD]
 *                                                      wide window — sum-along-PAD
 *                                                      recovers column-0 values
 *                                                      because columns [1, W_PAD)
 *                                                      are zero by design);
 *                            recv_idx_out is [L, R]  (scalar copy of column 0)
 *
 * Design notes:
 *   - All cross-rank GM writes go through tile primitives (TPUT). No AIV
 *     scalar GM stores on the cross-rank push path.
 *   - x is BF16; weight stays FP32; idx is INT32. The three channels use
 *     independent tiles, window regions, and host-backed outputs so dtype
 *     changes on any channel stay local.
 *   - Weight uses TROWSUM along the W_PAD axis to compact the wide window
 *     [L, R, W_PAD] → [L, R] FP32: sum-of-row recovers slot [0] because the
 *     other lanes are zero. One TLOAD + TROWSUM + TSTORE per expert.
 *   - Idx uses scalar GM copy of column 0 to compact [L, R, IDX_PAD] →
 *     [L, R] INT32. INT32 TROWSUM exists in pto-isa but hangs on a2a3 in
 *     this configuration; the L*R = 128 scalar stores are negligible.
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

// Demo dimensions — must match main.py.
static constexpr int N = 2;
static constexpr int T = 8;
static constexpr int TOPK = 2;
static constexpr int D = 64;
static constexpr int L = 4;
static constexpr int R = 32;
static constexpr int N_ROUTES = T * TOPK;  // 16

// Weight payload tile width. The protocol contract is one FP32 weight per
// (e, slot) — recv_w[L, R] FP32. AIV vector tiles have a hardware minimum
// granularity of 1x8 FP32 (32 B = one MTE burst), so the on-window /
// staged-out layout is [L, R, W_PAD] FP32 with the actual weight at
// slot [0] and zeros in [1, W_PAD). Host extracts column 0 to recover the
// [L, R] grid; production with proper stride-1 ops or a scalar-write path
// can drop W_PAD entirely.
static constexpr int W_PAD = 8;
// Same minimum-tile rationale for the idx channel — INT32 [L, R] in spirit,
// but materialized as [L, R, IDX_PAD] with the actual r=t*TOPK+k at slot [0].
static constexpr int IDX_PAD = 8;

// Window region byte sizes — mirror *_BYTES in main.py.
//
// Layout:
//   pub_counts[N][N][L]            INT32   (64 B)
//   count_done_sig[N]              INT32   (padded slot, 64 B)
//   recv_x[L][R][D]                BF16    (16 KB)
//   recv_w[L][R][W_PAD]            FP32    (4  KB; weight at slot [0], rest = 0)
//   recv_idx[L][R][IDX_PAD]        INT32   (4  KB; r=t*TOPK+k at slot [0], rest = 0)
//   data_done_sig[N]               INT32   (padded slot, 64 B)
// ---- Cross-rank visible regions consumed by combine.cpp ----
//   routed_y_buf[T][TOPK][D]       BF16    (2  KB demo; combine push destination,
//                                            addressed directly by (t, k))
//   combine_done_sig[N]            INT32   (padded slot, 64 B)
//
// recv_y does NOT live in this window — local_expert and combine pass it
// as a host-backed device tensor via the orch.
static constexpr int kPubCountsBytes = N * N * L * 4;  // N*N*L INT32
static constexpr int kSignalBytes = 64;
static constexpr int kRecvXBytes = L * R * D * 2;  // BF16
static constexpr int kRecvWBytes = L * R * W_PAD * 4;
static constexpr int kRecvIdxBytes = L * R * IDX_PAD * 4;
static constexpr int kRoutedYBufBytes = T * TOPK * D * 2;  // BF16

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
    __gm__ Tensor *indices_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *x_norm_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ Tensor *w_padded_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ Tensor *idx_padded_tensor = reinterpret_cast<__gm__ Tensor *>(args[3]);
    __gm__ Tensor *recv_x_out_tensor = reinterpret_cast<__gm__ Tensor *>(args[4]);
    __gm__ Tensor *recv_w_out_tensor = reinterpret_cast<__gm__ Tensor *>(args[5]);
    __gm__ Tensor *recv_idx_out_tensor = reinterpret_cast<__gm__ Tensor *>(args[6]);
    __gm__ Tensor *recv_count_out_tensor = reinterpret_cast<__gm__ Tensor *>(args[7]);
    __gm__ Tensor *scratch_tensor = reinterpret_cast<__gm__ Tensor *>(args[8]);
    int nranks = static_cast<int>(args[9]);
    __gm__ CommContext *comm_ctx = reinterpret_cast<__gm__ CommContext *>(args[10]);

    if (nranks != N) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    __gm__ int32_t *indices =
        reinterpret_cast<__gm__ int32_t *>(indices_tensor->buffer.addr) + indices_tensor->start_offset;
    __gm__ bfloat16_t *x_norm =
        reinterpret_cast<__gm__ bfloat16_t *>(x_norm_tensor->buffer.addr) + x_norm_tensor->start_offset;
    __gm__ float *w_padded =
        reinterpret_cast<__gm__ float *>(w_padded_tensor->buffer.addr) + w_padded_tensor->start_offset;
    __gm__ int32_t *idx_padded =
        reinterpret_cast<__gm__ int32_t *>(idx_padded_tensor->buffer.addr) + idx_padded_tensor->start_offset;
    __gm__ bfloat16_t *recv_x_out =
        reinterpret_cast<__gm__ bfloat16_t *>(recv_x_out_tensor->buffer.addr) + recv_x_out_tensor->start_offset;
    __gm__ float *recv_w_out =
        reinterpret_cast<__gm__ float *>(recv_w_out_tensor->buffer.addr) + recv_w_out_tensor->start_offset;
    __gm__ int32_t *recv_idx_out =
        reinterpret_cast<__gm__ int32_t *>(recv_idx_out_tensor->buffer.addr) + recv_idx_out_tensor->start_offset;
    // recv_count_out is shape [L, 1] INT32 (1-element-per-expert grid);
    // storage is L contiguous int32 slots either way.
    __gm__ int32_t *recv_count_out =
        reinterpret_cast<__gm__ int32_t *>(recv_count_out_tensor->buffer.addr) + recv_count_out_tensor->start_offset;

    // scratch_tensor's `start_offset` is in element units of its declared dtype (FP32).
    __gm__ uint8_t *scratch_base =
        reinterpret_cast<__gm__ uint8_t *>(scratch_tensor->buffer.addr) + scratch_tensor->start_offset * sizeof(float);
    __gm__ int32_t *pub_counts_local = reinterpret_cast<__gm__ int32_t *>(scratch_base + kOffPubCounts);
    __gm__ int32_t *count_done_sig_local = reinterpret_cast<__gm__ int32_t *>(scratch_base + kOffCountDone);
    __gm__ bfloat16_t *recv_x_local = reinterpret_cast<__gm__ bfloat16_t *>(scratch_base + kOffRecvX);
    __gm__ float *recv_w_local = reinterpret_cast<__gm__ float *>(scratch_base + kOffRecvW);
    __gm__ int32_t *recv_idx_local = reinterpret_cast<__gm__ int32_t *>(scratch_base + kOffRecvIdx);
    __gm__ int32_t *data_done_sig_local = reinterpret_cast<__gm__ int32_t *>(scratch_base + kOffDataDone);

    int my_rank = static_cast<int>(comm_ctx->rankId);

    // ------------------------------------------------------------------
    // histogram: scalar histogram + (dst, loc_e)-sorted route table.
    //
    // Scalar GM reads of indices[t * TOPK + k] are fine on AIV.
    // Bucket each route by (dst, loc_e) and stable-sort so the payload_push
    // cursor matches each peer's src-major slot_offset rule.
    // ------------------------------------------------------------------
    int send_counts[N][L];
    for (int d = 0; d < N; ++d) {
        for (int e = 0; e < L; ++e) {
            send_counts[d][e] = 0;
        }
    }

    int route_dst[N_ROUTES];
    int route_loc_e[N_ROUTES];
    int route_r[N_ROUTES];

    for (int r = 0; r < N_ROUTES; ++r) {
        int eid = indices[r];
        int dst = eid / L;
        int loc_e = eid - dst * L;
        send_counts[dst][loc_e] += 1;
        route_dst[r] = dst;
        route_loc_e[r] = loc_e;
        route_r[r] = r;
    }

    for (int i = 1; i < N_ROUTES; ++i) {
        int kd = route_dst[i], kl = route_loc_e[i], kr = route_r[i];
        int j = i - 1;
        while (j >= 0) {
            int cd = route_dst[j], cl = route_loc_e[j];
            bool greater = (cd > kd) || (cd == kd && cl > kl);
            if (!greater) break;
            route_dst[j + 1] = cd;
            route_loc_e[j + 1] = cl;
            route_r[j + 1] = route_r[j];
            --j;
        }
        route_dst[j + 1] = kd;
        route_loc_e[j + 1] = kl;
        route_r[j + 1] = kr;
    }

    pipe_barrier(PIPE_ALL);

    // ------------------------------------------------------------------
    // publish: publish my full send_counts table to every peer.
    //
    // Goal: every rank's local pub_counts[N_src=N][N_dst=N][L] becomes the
    // global view — pub_counts[s][d][e] = how many rows sender s sends to
    // dst d's local expert e. The prefix_sum phase then computes prefix sums
    // purely from local reads.
    //
    // Self-rank is included in the publish loop so pub_counts[my_rank][:][:]
    // gets populated locally — recv_count_out (= sum_s pub_counts[s][me][e])
    // depends on the s = me term too. CommRemotePtr returns the local addr
    // for peer == my_rank, so the same TNOTIFY path works.
    //
    // pub_counts is assumed zero-init by HCCL on first window allocation;
    // AtomicAdd-from-zero is equivalent to a store.
    //
    // ⚠ The pipe_barrier between the two TNOTIFY groups is load-bearing:
    // without it, count_done can become visible on a peer before the
    // corresponding pub_counts writes drain, so the peer's TWAIT passes
    // early and the prefix_sum phase reads stale pub_counts → wrong slot
    // offsets.
    // ------------------------------------------------------------------
    for (int peer = 0; peer < N; ++peer) {
        for (int d = 0; d < N; ++d) {
            for (int e = 0; e < L; ++e) {
                int v = send_counts[d][e];
                if (v == 0) continue;
                int idx = (my_rank * N + d) * L + e;
                __gm__ int32_t *remote = CommRemotePtr(comm_ctx, pub_counts_local + idx, peer);
                pto::comm::Signal sig(remote);
                pto::comm::TNOTIFY(sig, (int32_t)v, pto::comm::NotifyOp::AtomicAdd);
            }
        }
    }
    pipe_barrier(PIPE_ALL);

    for (int peer = 0; peer < N; ++peer) {
        if (peer == my_rank) continue;
        __gm__ int32_t *remote_done = CommRemotePtr(comm_ctx, count_done_sig_local + my_rank, peer);
        pto::comm::Signal sig(remote_done);
        pto::comm::TNOTIFY(sig, (int32_t)1, pto::comm::NotifyOp::AtomicAdd);
    }
    for (int src = 0; src < N; ++src) {
        if (src == my_rank) continue;
        pto::comm::Signal sig(count_done_sig_local + src);
        pto::comm::TWAIT(sig, (int32_t)1, pto::comm::WaitCmp::GE);
    }
    pipe_barrier(PIPE_ALL);

    // ------------------------------------------------------------------
    // prefix_sum: local prefix sums (scalar).
    //
    // Two outputs from the same pub_counts scan:
    //   my_slot_at_dst[dst][e] = sum_{s<my_rank} pub_counts[s][dst][e]
    //     — sender's slot offset on each peer's recv area (payload_push uses this)
    //   recv_expert_count[e]   = sum_{s<N}       pub_counts[s][my_rank][e]
    //     — total rows arriving at THIS rank's local expert e (host output)
    //
    // recv_count_out is what the production moe_expert kernel consumes to
    // decide tile counts; emitting it here lets the demo verify the
    // protocol contract without standing up the expert kernel.
    // ------------------------------------------------------------------
    int my_slot_at_dst[N][L];
    for (int dst = 0; dst < N; ++dst) {
        for (int e = 0; e < L; ++e) {
            int sum = 0;
            for (int s = 0; s < my_rank; ++s) {
                sum += pub_counts_local[(s * N + dst) * L + e];
            }
            my_slot_at_dst[dst][e] = sum;
        }
    }
    for (int e = 0; e < L; ++e) {
        int sum = 0;
        for (int s = 0; s < N; ++s) {
            sum += pub_counts_local[(s * N + my_rank) * L + e];
        }
        recv_count_out[e] = sum;
    }

    // ------------------------------------------------------------------
    // payload_push: push x / weight / idx payloads via TPUT.
    //
    // Each route emits three independent 1xC tiles:
    //   - x   (BF16, 1 x D)         x_norm[t, :]                       -> peer.recv_x[loc_e][slot, :]
    //   - w   (FP32, 1 x W_PAD)     w_padded[r, :] = [w, 0, …, 0]      -> peer.recv_w[loc_e][slot, :]
    //   - idx (INT32, 1 x IDX_PAD)  idx_padded[r, :] = [r, 0, …, 0]    -> peer.recv_idx[loc_e][slot, :]
    //   where r = t * TOPK + k.
    //
    // Each channel has its own tile register so dtype changes stay local.
    //
    // Self-rank is *not* skipped: the sender pushes its own routes into its
    // local recv_x / recv_w / recv_idx windows so the stage_out phase can
    // stage everything out uniformly. CommRemotePtr returns the local addr
    // for peer == my_rank.
    // ------------------------------------------------------------------
    using XShape = Shape<1, 1, 1, 1, D>;
    using XStride = Stride<D, D, D, D, 1>;
    using XGlobal = GlobalTensor<bfloat16_t, XShape, XStride>;
    using XTile = Tile<TileType::Vec, bfloat16_t, 1, D, BLayout::RowMajor, -1, -1>;

    using WShape = Shape<1, 1, 1, 1, W_PAD>;
    using WStride = Stride<W_PAD, W_PAD, W_PAD, W_PAD, 1>;
    using WGlobal = GlobalTensor<float, WShape, WStride>;
    using WTile = Tile<TileType::Vec, float, 1, W_PAD, BLayout::RowMajor, -1, -1>;

    using IShape = Shape<1, 1, 1, 1, IDX_PAD>;
    using IStride = Stride<IDX_PAD, IDX_PAD, IDX_PAD, IDX_PAD, 1>;
    using IGlobal = GlobalTensor<int32_t, IShape, IStride>;
    using ITile = Tile<TileType::Vec, int32_t, 1, IDX_PAD, BLayout::RowMajor, -1, -1>;

    // Reuse three tile registers across payload_push (TPUT staging) and
    // stage_out (xfer). The pipe_barrier between phases serializes the two
    // uses, so sharing UB slots is safe.
    XTile x_tile(1, D);
    WTile w_tile(1, W_PAD);
    ITile idx_tile(1, IDX_PAD);
    TASSIGN(x_tile, 0x0);
    TASSIGN(w_tile, 0x10000);
    TASSIGN(idx_tile, 0x20000);

    int cursor[N][L];
    for (int d = 0; d < N; ++d) {
        for (int e = 0; e < L; ++e) {
            cursor[d][e] = 0;
        }
    }

    for (int route_i = 0; route_i < N_ROUTES; ++route_i) {
        int dst = route_dst[route_i];
        int loc_e = route_loc_e[route_i];
        int r = route_r[route_i];
        int t = r / TOPK;

        int off = my_slot_at_dst[dst][loc_e] + cursor[dst][loc_e];
        cursor[dst][loc_e] += 1;
        int row = loc_e * R + off;

        // Channel 1: x  (BF16, 1xD)
        __gm__ bfloat16_t *x_src = x_norm + t * D;
        __gm__ bfloat16_t *x_dst_local = recv_x_local + row * D;
        __gm__ bfloat16_t *x_dst_remote = CommRemotePtr(comm_ctx, x_dst_local, dst);

#if defined(__CPU_SIM)
        for (int i = 0; i < D; ++i) {
            x_dst_remote[i] = x_src[i];
        }
#else
        XGlobal x_src_g(x_src);
        XGlobal x_dst_g(x_dst_remote);
        pto::comm::TPUT(x_dst_g, x_src_g, x_tile);
#endif

        // Channel 2: weight (FP32, 1xW_PAD); host pre-packed [w, 0, …, 0]
        __gm__ float *w_src = w_padded + r * W_PAD;
        __gm__ float *w_dst_local = recv_w_local + row * W_PAD;
        __gm__ float *w_dst_remote = CommRemotePtr(comm_ctx, w_dst_local, dst);

#if defined(__CPU_SIM)
        for (int i = 0; i < W_PAD; ++i) {
            w_dst_remote[i] = w_src[i];
        }
#else
        WGlobal w_src_g(w_src);
        WGlobal w_dst_g(w_dst_remote);
        pto::comm::TPUT(w_dst_g, w_src_g, w_tile);
#endif

        // Channel 3: idx (INT32, 1xIDX_PAD); host pre-packed [r, 0, …, 0]
        __gm__ int32_t *idx_src = idx_padded + r * IDX_PAD;
        __gm__ int32_t *idx_dst_local = recv_idx_local + row * IDX_PAD;
        __gm__ int32_t *idx_dst_remote = CommRemotePtr(comm_ctx, idx_dst_local, dst);

#if defined(__CPU_SIM)
        for (int i = 0; i < IDX_PAD; ++i) {
            idx_dst_remote[i] = idx_src[i];
        }
#else
        IGlobal idx_src_g(idx_src);
        IGlobal idx_dst_g(idx_dst_remote);
        pto::comm::TPUT(idx_dst_g, idx_src_g, idx_tile);
#endif
    }
    pipe_barrier(PIPE_ALL);

    // Data-done barrier — payload visibility across ranks.
    //
    // Self-rank is skipped here even though payload_push includes a self-push
    // (loopback into our own recv_x window): the pipe_barrier above already
    // orders that local TPUT vs. the upcoming stage_out reads on this rank,
    // so a self-signal would be redundant. Cross-rank payloads are what need
    // explicit done signals.
    for (int peer = 0; peer < N; ++peer) {
        if (peer == my_rank) continue;
        __gm__ int32_t *remote_done = CommRemotePtr(comm_ctx, data_done_sig_local + my_rank, peer);
        pto::comm::Signal sig(remote_done);
        pto::comm::TNOTIFY(sig, (int32_t)1, pto::comm::NotifyOp::AtomicAdd);
    }
    for (int src = 0; src < N; ++src) {
        if (src == my_rank) continue;
        pto::comm::Signal sig(data_done_sig_local + src);
        pto::comm::TWAIT(sig, (int32_t)1, pto::comm::WaitCmp::GE);
    }
    pipe_barrier(PIPE_ALL);

    // ------------------------------------------------------------------
    // stage_out: stage out window -> host outputs.
    //
    // recv_x_out  is [L, R, D]  BF16  : per-row 1xD TLOAD/TSTORE, L*R rows.
    // recv_w_out  is [L, R]     FP32  : compacted from [L, R, W_PAD] window.
    // recv_idx_out is [L, R]    INT32 : compacted from [L, R, IDX_PAD].
    //
    // Compaction trick (no AIV scalar GM stores, no strided MTE2 needed):
    // each W_PAD/IDX_PAD-wide row was filled by the sender as
    //   [real_value, 0, 0, …, 0]
    // so summing along the wide axis recovers the real value at slot [0].
    // We TLOAD one expert's wide grid as an RxW_PAD vec tile and TROWSUM
    // it to an Rx1 ColMajor tile, then TSTORE compactly to recv_*_out[e, :].
    // One TROWSUM per expert × L experts × 2 channels = 2*L = 8 ops total.
    //
    // Padding rows (slot >= cnt[e]) carry stale window contents — host
    // verification slices by `expected_count` and ignores them.
    // ------------------------------------------------------------------

    // Wide-window tile types (full RxPAD grid TLOADed in one shot).
    // Sum-output tiles use Layout::DN — TROWSUM produces a column tile
    // (Rx1 ColMajor); writing it back to GM needs DN layout.
    using WWideShape = Shape<1, 1, 1, R, W_PAD>;
    using WWideStride = Stride<R * W_PAD, R * W_PAD, R * W_PAD, W_PAD, 1>;
    using WWideG = GlobalTensor<float, WWideShape, WWideStride>;
    using WWideTile = Tile<TileType::Vec, float, R, W_PAD, BLayout::RowMajor, R, W_PAD>;
    using WSumShape = Shape<1, 1, 1, R, 1>;
    using WSumStride = Stride<1, 1, 1, 1, 1>;
    using WSumG = GlobalTensor<float, WSumShape, WSumStride, Layout::DN>;
    using WSumTile = Tile<TileType::Vec, float, R, 1, BLayout::ColMajor, R, 1>;

    using IWideShape = Shape<1, 1, 1, R, IDX_PAD>;
    using IWideStride = Stride<R * IDX_PAD, R * IDX_PAD, R * IDX_PAD, IDX_PAD, 1>;
    using IWideG = GlobalTensor<int32_t, IWideShape, IWideStride>;
    using IWideTile = Tile<TileType::Vec, int32_t, R, IDX_PAD, BLayout::RowMajor, R, IDX_PAD>;
    using ISumShape = Shape<1, 1, 1, R, 1>;
    using ISumStride = Stride<1, 1, 1, 1, 1>;
    using ISumG = GlobalTensor<int32_t, ISumShape, ISumStride, Layout::DN>;
    using ISumTile = Tile<TileType::Vec, int32_t, R, 1, BLayout::ColMajor, R, 1>;

    // UB allocation. The payload_push phase's x_tile/w_tile/idx_tile have
    // drained, so we can reuse those slots. Slot pitch is 64 KB; the largest
    // tile here is R*W_PAD FP32 = 1 KB or R*IDX_PAD INT32 = 1 KB.
    //
    // TROWSUM's tmp tile must be the SAME SHAPE as the source (RxPAD), not
    // the destination (Rx1) — pto-isa uses it as scratch for partial
    // reductions.
    WWideTile w_wide_tile;
    WSumTile w_sum_tile;
    WWideTile w_tmp_tile;
    IWideTile idx_wide_tile;
    ISumTile idx_sum_tile;
    IWideTile idx_tmp_tile;
    TASSIGN(w_wide_tile, 0x10000);
    TASSIGN(w_sum_tile, 0x20000);
    TASSIGN(w_tmp_tile, 0x21000);
    TASSIGN(idx_wide_tile, 0x30000);
    TASSIGN(idx_sum_tile, 0x40000);
    TASSIGN(idx_tmp_tile, 0x41000);

    // Stage out x: per-row 1xD copies.
    for (int e = 0; e < L; ++e) {
        for (int slot = 0; slot < R; ++slot) {
            int row = e * R + slot;
            __gm__ bfloat16_t *x_win = recv_x_local + row * D;
            __gm__ bfloat16_t *x_out = recv_x_out + row * D;
            XGlobal x_win_g(x_win);
            XGlobal x_out_g(x_out);
            TLOAD(x_tile, x_win_g);
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            TSTORE(x_out_g, x_tile);
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        }
    }

    // Stage out compact w / idx: per-expert TLOAD + TROWSUM + TSTORE.
    // pipe_barrier(PIPE_V) brackets the TROWSUM, plus MTE2/MTE3 fences for
    // the GM legs.
    for (int e = 0; e < L; ++e) {
        // weight channel
        __gm__ float *w_win = recv_w_local + e * R * W_PAD;
        __gm__ float *w_out = recv_w_out + e * R;
        WWideG w_win_g(w_win);
        WSumG w_out_g(w_out);
        TLOAD(w_wide_tile, w_win_g);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        pipe_barrier(PIPE_V);
        TROWSUM(w_sum_tile, w_wide_tile, w_tmp_tile);
        pipe_barrier(PIPE_V);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
        TSTORE(w_out_g, w_sum_tile);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    }

    // Stage out idx: scalar copy of column 0 from the wide window.
    //
    // ⚠ The same TROWSUM compaction used above for the FP32 weight channel
    // does NOT work reliably for INT32 on a2a3: pto-isa declares INT32
    // TROWSUM support, but with the same [R, IDX_PAD] / Layout::DN setup
    // the kernel hangs on hardware. Until that path is stabilized, fall
    // back to a scalar copy here. Volume is small (L*R = 128 INT32 stores)
    // so the perf cost is negligible.
    for (int e = 0; e < L; ++e) {
        for (int slot = 0; slot < R; ++slot) {
            recv_idx_out[e * R + slot] = recv_idx_local[(e * R + slot) * IDX_PAD];
        }
    }
    pipe_barrier(PIPE_ALL);
}
