// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// Packed-cache gather, dual KV rings, and sink-aware sparse attention.
//
// Deep pipeline:
//   * 512 KiB L1: double-buffered Q [32, 384] and shared Q [32, 128] tails per
//     head half, two depth-3 64-row KV rings, and one P [64, 128] slot.
//   * 240 KiB UB: two PV result channels (accumulator + partial halves), one QK score
//     tile, P staging, three raw-quantized slots, two NZ17 slots, gather IDs.
//   * AIC issues QK(next, 0) before PV(current, last); KV stays live through PV.
//   * AIV taps: gather t, dequant t-1, softmax t-3, update t-4, with four drain ticks.
#include <cstdint>
#if defined(__CPU_SIM)
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
#endif
#include <pto/pto-inst.hpp>
#include "tensor.h"
#include "intrinsic.h"
#include "sparse_dequant_vf.hpp"
#include "sparse_softmax_vf.hpp"
#include "sparse_merge_vf.hpp"

using namespace pto;

namespace {
template <typename T, int R, int C>
using Vec = Tile<TileType::Vec, T, R, C>;
template <typename T, int R, int C>
using GM = GlobalTensor<T, TileShape2D<T, R, C>, BaseShape2D<T, R, C>>;

template <typename T>
__aicore__ inline __gm__ T* data(__gm__ int64_t* args, int index) {
    auto tensor = reinterpret_cast<__gm__ Tensor*>(args[index]);
    return reinterpret_cast<__gm__ T*>(tensor->buffer.addr) + tensor->start_offset;
}

constexpr int ARG_OUTPUT = 0;
constexpr int ARG_QUERY = 1;
constexpr int ARG_SINK = 2;
constexpr int ARG_WINDOW = 3;
constexpr int ARG_COMPRESSED = 4;
constexpr int ARG_WINDOW_INDICES = 5;
constexpr int ARG_COMPRESSED_INDICES = 6;
constexpr int ARG_PARTIALS = 7;
constexpr int ARG_TOKENS = 8;
constexpr int ARG_HEADS = 9;
constexpr int ARG_WINDOW_KEYS = 10;
constexpr int ARG_COMPRESSED_KEYS = 11;
constexpr int ARG_BALANCED = 12;
constexpr int PARTIAL_STRIDE = 64 * (512 + 2);

#if !defined(__CPU_SIM)
__aicore__ inline __gm__ uint32_t* ready_flags(__gm__ int64_t* args) {
    return reinterpret_cast<__gm__ uint32_t*>(data<float>(args, ARG_PARTIALS) +
        (int64_t)2 * get_block_num(args) * PARTIAL_STRIDE);
}
#endif


constexpr int DIM = 512;
constexpr int CUBE_HEADS = 64;
constexpr int VEC_HEADS = 32;
constexpr int TILE_KEYS = 128;
constexpr int KEY_HALF = 64;
constexpr int VEC_KEYS = 32;
constexpr int KEY_CHUNK = 64;
constexpr int QK_CHUNKS = 8;
constexpr int PV_CHUNKS = 8;
constexpr int KV_SLOTS = 3;
constexpr int RAW_SLOTS = 3;
constexpr int NZ_SLOTS = 2;
constexpr int AIV_BANK = 16;

// Affine offsets avoid read-only tables outside the runtime-loaded text section.
// L1 map: wide Q double slots plus one shared tail per head half.
constexpr int Q_WIDE_LOWER_BASE = 0;
constexpr int Q_WIDE_STRIDE = 24576;
constexpr int Q_TAIL_LOWER = 49152;
constexpr int KV_LOWER_BASE = 57344;
constexpr int KV_STRIDE = 65536;
constexpr int P_L1 = 253952;
constexpr int KV_UPPER_BASE = 270336;
constexpr int Q_WIDE_UPPER_BASE = 466944;
constexpr int Q_TAIL_UPPER = 516096;
constexpr int L1_SIZE = 524288;

// L0 geometry.
constexpr int PV_ACC_STRIDE = 16384;
constexpr int QK_ACC_BASE = 131072;

// UB map per AIV: 240 KiB used, 16 KiB free.
constexpr int PV_CHANNEL_BYTES = 65536;
constexpr int PV_ROW_HALF_BYTES = VEC_HEADS * 256 * sizeof(float);
constexpr int SCORE_UB = 131072;
constexpr int SCORE_BYTES = VEC_HEADS * TILE_KEYS * sizeof(float);
constexpr int P_STAGING_UB = 147456;
constexpr int P_STAGING_BYTES = (VEC_HEADS + 1) * TILE_KEYS * sizeof(bfloat16_t);
constexpr int MAX_STATE_UB = 155904;
constexpr int DEN_STATE_UB = 156416;
constexpr int ALPHA_STATE_UB = 156928;
constexpr int STATE_SLOT_STRIDE = 256;
constexpr int SINK_UB = 157440;
constexpr int FINAL_MAX_UB = 157568;
// Raw slots use 32 packed rows with a 544-byte UB row stride.
constexpr int RAW_UB = 157696;
constexpr int RAW_STRIDE = 17408;
constexpr int NZ_UB = 209920;
constexpr int NZ_STRIDE = 17408;
constexpr int GATHER_ID_UB = 244736;
constexpr int GATHER_ID_STRIDE = VEC_KEYS * sizeof(int32_t);
constexpr int SOFTMAX_ID_UB = 245120;
constexpr int FINAL_CORRECTION_UB = 245632;
constexpr int UB_USED = 245760;
constexpr int UB_SIZE = 262144;

// Cross-core (intra-block) semaphores: AIC addresses the second AIV bank at id + 16.
// A5 intra-block IDs are plain 4-bit semaphores: one counter per id, no separate
// producer/consumer direction (see pto-isa include/pto/npu/a5/custom/TSync_Custom.hpp,
// which spends four ids -- ready/free for each direction -- on one bidirectional FIFO,
// and the device-proven b931df2 kernel, which keeps separate ready/free ids).  Every
// FIFO below therefore uses distinct ready (producer -> consumer) and free
// (consumer -> producer) ids; sharing one id for both directions deadlocks as soon as
// the consumer races ahead of the producer.
constexpr int FLAG_KV_FREE = 0;    // +slot (0..2): AIC -> AIV, ring slot writable
constexpr int FLAG_KV_READY = 4;   // +slot (0..2): AIV -> AIC, ring slot readable
constexpr int FLAG_P_FREE = 3;     // AIC -> AIV, P staging writable
constexpr int FLAG_P_READY = 7;    // AIV -> AIC, P tile readable
constexpr int FLAG_PV_FREE = 8;    // +channel: AIV -> AIC, PV UB half writable
constexpr int FLAG_PV_READY = 10;  // +channel: AIC -> AIV, PV UB half readable
constexpr int FLAG_SCORE_FREE = 12;   // AIV -> AIC, score tile writable
constexpr int FLAG_SCORE_READY = 13;  // AIC -> AIV, score tile readable
constexpr int FINISH_READY = 14;
constexpr int FINISH_ACK = 15;

// AIC events: Q wide uses 0/1; L0A uses stages 0..3; L0B uses 4..7. Q uses a separate MTE2<->MTE1 flag domain.
constexpr int EV_Q_TAIL = 2;
constexpr int EV_L0B_BASE = 4;
constexpr int EV_FIX_QK = 0;
constexpr int PV_ACC_LOCK_BASE = 14;

// AIV in-pipe events.
constexpr int EV_SOFTMAX_ID = 0;
constexpr int EV_SINK = 1;
constexpr int EV_RAW0 = 2;
constexpr int EV_RAW1 = 3;
constexpr int EV_RAW2 = 4;
constexpr int EV_NZ0 = 5;
constexpr int EV_NZ1 = 6;
constexpr int EV_P_STAGE = 7;

static_assert(L1_SIZE == 512 * 1024, "L1 map must cover the 512 KiB L1");
static_assert((Q_WIDE_LOWER_BASE + Q_WIDE_STRIDE) + 24576 == Q_TAIL_LOWER, "lower wide Q must precede its tail");
static_assert(Q_TAIL_LOWER + 8192 == KV_LOWER_BASE, "lower Q tail must precede lower KV");
static_assert((KV_LOWER_BASE + 2 * KV_STRIDE) + 65536 == P_L1, "lower KV ring must precede P");
static_assert(P_L1 + 16384 == KV_UPPER_BASE, "P must precede upper KV");
static_assert((KV_UPPER_BASE + 2 * KV_STRIDE) + 65536 == Q_WIDE_UPPER_BASE, "upper KV must precede upper wide Q");
static_assert((Q_WIDE_UPPER_BASE + Q_WIDE_STRIDE) + 24576 == Q_TAIL_UPPER, "upper wide Q must precede its tail");
static_assert(Q_TAIL_UPPER + 8192 == L1_SIZE, "Q tail must close the exact 512 KiB map");
static_assert(UB_USED == 240 * 1024, "UB map must use 240 KiB");
static_assert(UB_USED + 16384 == UB_SIZE, "UB map must leave exactly 16 KiB free");
static_assert(SCORE_UB + SCORE_BYTES == P_STAGING_UB, "score tile must precede P staging");
static_assert(P_STAGING_UB + P_STAGING_BYTES == MAX_STATE_UB, "P staging must precede state slots");
static_assert(ALPHA_STATE_UB + 2 * STATE_SLOT_STRIDE == SINK_UB, "alpha slots must precede sink");
static_assert(SINK_UB + 128 == FINAL_MAX_UB, "sink must precede final max");
static_assert(FINAL_MAX_UB + 128 == RAW_UB, "final max must precede raw slots");
static_assert(RAW_UB + RAW_SLOTS * RAW_STRIDE == NZ_UB, "raw slots must precede NZ slots");
static_assert(NZ_UB + NZ_SLOTS * NZ_STRIDE == GATHER_ID_UB, "NZ slots must precede gather IDs");
static_assert(GATHER_ID_UB + RAW_SLOTS * GATHER_ID_STRIDE == SOFTMAX_ID_UB, "gather IDs must precede softmax IDs");
static_assert(SOFTMAX_ID_UB + TILE_KEYS * sizeof(int32_t) == FINAL_CORRECTION_UB, "softmax IDs must precede final correction");
static_assert(FINAL_CORRECTION_UB + VEC_HEADS * sizeof(float) == UB_USED, "final correction must close the used UB");
static_assert(PV_CHANNEL_BYTES + PV_CHANNEL_BYTES == SCORE_UB, "PV channels must precede the score tile");

template <int R, int C>
using Mat = Tile<TileType::Mat, bfloat16_t, R, C, BLayout::ColMajor, R, C, SLayout::RowMajor>;
template <int R, int C>
using TransposedMat = Tile<TileType::Mat, bfloat16_t, R, C, BLayout::RowMajor, R, C, SLayout::ColMajor>;
template <int R, int C>
using Left = Tile<TileType::Left, bfloat16_t, R, C, BLayout::ColMajor, R, C, SLayout::RowMajor>;
template <int R, int C>
using Right = Tile<TileType::Right, bfloat16_t, R, C, BLayout::RowMajor, R, C, SLayout::ColMajor>;
template <int R, int C>
using Acc = Tile<TileType::Acc, float, R, C, BLayout::ColMajor, R, C, SLayout::RowMajor, 1024>;
template <int H>
using HeadColumn = Tile<TileType::Vec, float, H, 1, BLayout::ColMajor>;

template <int C>
using QLoad = Tile<TileType::Mat, bfloat16_t, VEC_HEADS, C, BLayout::ColMajor, DYNAMIC, C, SLayout::RowMajor,
                   512, PadValue::Zero>;
using QWideView = Mat<VEC_HEADS, 384>;
using QTailView = Mat<VEC_HEADS, 128>;
using KVTile = Mat<KEY_HALF, DIM>;
using KVTranspose = TransposedMat<DIM, KEY_HALF>;
using PTile = Mat<CUBE_HEADS, TILE_KEYS>;
using QKLeft = Left<CUBE_HEADS, 64>;
using QKRight = Right<64, TILE_KEYS>;
using QKAcc = Acc<CUBE_HEADS, TILE_KEYS>;
using PVLeft = Left<CUBE_HEADS, TILE_KEYS>;
using PVRightHalf = Right<KEY_HALF, KEY_CHUNK>;
using PVRight = Right<TILE_KEYS, KEY_CHUNK>;
using PVAcc = Acc<CUBE_HEADS, KEY_CHUNK>;
using Score = Vec<float, VEC_HEADS, TILE_KEYS>;
using Channel = Vec<float, CUBE_HEADS, 256>;
using ChannelHalf = Vec<float, VEC_HEADS, 256>;
using State = Vec<float, 1, VEC_HEADS>;
using PStaging = Tile<TileType::Vec, bfloat16_t, VEC_HEADS, TILE_KEYS, BLayout::ColMajor, VEC_HEADS, TILE_KEYS,
                      SLayout::RowMajor, 512, PadValue::Null, CompactMode::RowPlusOne>;
using SoftmaxIds = Vec<int32_t, 1, TILE_KEYS>;
using GatherIds = Vec<int32_t, 1, VEC_KEYS>;
using NzChunk = Tile<TileType::Vec, bfloat16_t, 16, DIM, BLayout::ColMajor, 16, DIM, SLayout::RowMajor, 512,
                     PadValue::Null, CompactMode::RowPlusOne>;
using ChunkIds = Vec<int32_t, 1, 16>;
using ResultBf16 = Vec<bfloat16_t, VEC_HEADS, 256>;

__aicore__ inline int64_t row_stride(__gm__ int64_t* args, int index) {
    return reinterpret_cast<__gm__ Tensor*>(args[index])->strides[0];
}

__aicore__ inline int valid_heads_of(__gm__ int64_t* args) {
    return args[ARG_HEADS] < CUBE_HEADS ? (int)args[ARG_HEADS] : CUBE_HEADS;
}

struct TileTask {
    int ordinal;  // -1 only for a drain sentinel
    int token;
    int part;
    int query;
    int query_slot;   // query & 1
    bool first;       // part == 0
    bool last;        // part + 1 == parts
    bool compressed;  // part * TILE_KEYS >= Wkeys
    int partial_slot;
};

struct WorkPlan {
    int workers;
    int block;
    int begin;
    int end;
    int first_token;
    int tiles;
    int queries;
};

template <int Parts, bool Balanced>
__aicore__ inline WorkPlan work_plan(__gm__ int64_t* args) {
    const int workers = get_block_num(args);
    const int block = get_block_idx(args);
    const int tokens = args[ARG_TOKENS];
    const int parts = Parts > 0 ? Parts : (args[ARG_WINDOW_KEYS] + args[ARG_COMPRESSED_KEYS]) / TILE_KEYS;
    WorkPlan plan;
    plan.workers = workers;
    plan.block = block;
    if constexpr (Balanced) {
        plan.begin = (int64_t)tokens * parts * block / workers;
        plan.end = (int64_t)tokens * parts * (block + 1) / workers;
        plan.first_token = plan.begin / parts;
        plan.tiles = plan.end - plan.begin;
        plan.queries = plan.tiles ? (plan.end - 1) / parts - plan.first_token + 1 : 0;
    } else {
        plan.queries = block < tokens ? (tokens - block + workers - 1) / workers : 0;
        plan.begin = 0;
        plan.end = plan.queries * parts;
        plan.first_token = block;
        plan.tiles = plan.end;
    }
    return plan;
}

template <int Parts, bool Balanced>
__aicore__ inline TileTask make_task(__gm__ int64_t* args, const WorkPlan& plan, int ordinal) {
    const int parts = Parts > 0 ? Parts : (args[ARG_WINDOW_KEYS] + args[ARG_COMPRESSED_KEYS]) / TILE_KEYS;
    const int position = Balanced ? plan.begin + ordinal : ordinal;
    const int query = parts == 1 ? position : (parts == 5 ? position / 5 : position / parts);
    TileTask task;
    task.ordinal = ordinal;
    task.part = position - query * parts;
    task.query = Balanced ? query - plan.first_token : query;
    task.token = Balanced ? query : plan.block + query * plan.workers;
    task.query_slot = task.query & 1;
    task.first = ordinal == 0 || task.part == 0;
    task.last = ordinal + 1 == plan.tiles || task.part + 1 == parts;
    task.compressed = task.part * TILE_KEYS >= args[ARG_WINDOW_KEYS];
    task.partial_slot = -1;
    if constexpr (Balanced) {
        if (task.token == plan.first_token && plan.begin % parts != 0) task.partial_slot = 2 * plan.block;
        else if (task.token == (plan.end - 1) / parts && plan.end % parts != 0) task.partial_slot = 2 * plan.block + 1;
    }
    return task;
}

#if defined(__DAV_CUBE__) && !defined(__CPU_SIM)
template <pipe_t Pipe>
__aicore__ inline void signal_both(int flag) {
    set_intra_block(Pipe, flag);
    set_intra_block(Pipe, flag + AIV_BANK);
}
template <pipe_t Pipe>
__aicore__ inline void wait_both(int flag) {
    wait_intra_block(Pipe, flag);
    wait_intra_block(Pipe, flag + AIV_BANK);
}

template <int Width>
__aicore__ inline void load_q_component(__gm__ int64_t* args, int token, int column, int lower_base,
                                        int upper_base, int event) {
    const int valid = valid_heads_of(args);
    const int lower = valid < VEC_HEADS ? valid : VEC_HEADS;
    const int upper = valid > VEC_HEADS ? valid - VEC_HEADS : 0;
    using QShape = TileShape2D<bfloat16_t, DYNAMIC, Width>;
    using QBase = BaseShape2D<bfloat16_t, CUBE_HEADS, DIM>;
    __gm__ bfloat16_t* q = data<bfloat16_t>(args, ARG_QUERY) + (int64_t)token * row_stride(args, ARG_QUERY) + column;
    wait_flag(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(event));
    if (lower > 0) {
        QLoad<Width> tile(lower);
        TASSIGN(tile, lower_base);
        GlobalTensor<bfloat16_t, QShape, QBase> tensor(q, QShape(lower, Width));
        TLOAD(tile, tensor);
    }
    if (upper > 0) {
        QLoad<Width> tile(upper);
        TASSIGN(tile, upper_base);
        GlobalTensor<bfloat16_t, QShape, QBase> tensor(q + (int64_t)VEC_HEADS * DIM, QShape(upper, Width));
        TLOAD(tile, tensor);
    }
    set_flag(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(event));
}
__aicore__ inline void load_q_wide(__gm__ int64_t* args, int query, int slot) {
    load_q_component<384>(args, query, 0, (Q_WIDE_LOWER_BASE + slot * Q_WIDE_STRIDE), (Q_WIDE_UPPER_BASE + slot * Q_WIDE_STRIDE), slot);
}
__aicore__ inline void load_q_tail(__gm__ int64_t* args, int query) {
    load_q_component<128>(args, query, 384, Q_TAIL_LOWER, Q_TAIL_UPPER, EV_Q_TAIL);
}

// Preserve the destination [64,64] pitch: TEXTRACT(Left<32,64>) would compact at dstStride=2.
template <typename DstTile, typename SrcTile>
__tf__ PTO_INTERNAL void extract_qk_left_half(DstTile& dst, SrcTile& src, int head_half, int q_chunk) {
    __ca__ bfloat16_t* dst_addr = (__ca__ bfloat16_t*)__cce_get_tile_ptr(dst.data()) + head_half * 512;
    __cbuf__ bfloat16_t* src_addr = (__cbuf__ bfloat16_t*)__cce_get_tile_ptr(src.data());
    load_cbuf_to_ca(dst_addr, src_addr, 0, q_chunk * 4, 2, 4, 2, 4, 0);
}
// Each 64-key half retains the destination Right<64,128> N-panel pitch (dstStride=8).
template <typename DstTile, typename SrcTile>
__tf__ PTO_INTERNAL void extract_qk_right_half(DstTile& dst, SrcTile& src, int key_half, int d_chunk) {
    __cb__ bfloat16_t* dst_addr = (__cb__ bfloat16_t*)__cce_get_tile_ptr(dst.data()) + key_half * 1024;
    __cbuf__ bfloat16_t* src_addr = (__cbuf__ bfloat16_t*)__cce_get_tile_ptr(src.data());
    pto_load_cbuf_to_cb<false>(dst_addr, src_addr, 0, d_chunk * 4, 4, 4, 4, 8);
}

__tf__ __aicore__ inline void store_qk_final(
    Score::TileDType __out__ dst, QKAcc::TileDType __in__ src) {
    SetLoop3Para();
    auto output = (__ubuf__ float*)__cce_get_tile_ptr(dst);
    auto input = (__cc__ float*)__cce_get_tile_ptr(src);
    constexpr uint8_t dual = GetDualDstCtl<Score, QKAcc, AccToVecMode::DualModeSplitM, QuantMode_t::NoQuant>();
    constexpr uint32_t stride = GetTmovAccDstStride<Score, QKAcc>();
    pto_copy_matrix_cc_to_ub(
        output, input, 0, TILE_KEYS, CUBE_HEADS, stride, CUBE_HEADS, dual, false, 0, 3,
        QuantMode_t::NoQuant, 0, false, true, 0, 0, false, false, 0, false, false, false, false, false, false);
}

__aicore__ inline void compute_qk(__gm__ int64_t* args, const TileTask& task) {
    set_ctrl(sbitset0(get_ctrl(), 51));
    const int slot = task.ordinal % KV_SLOTS;
    const int q_slot = task.query_slot;
    QWideView wide_lo, wide_hi;
    QTailView tail_lo, tail_hi;
    TASSIGN(wide_lo, (Q_WIDE_LOWER_BASE + q_slot * Q_WIDE_STRIDE));
    TASSIGN(wide_hi, (Q_WIDE_UPPER_BASE + q_slot * Q_WIDE_STRIDE));
    TASSIGN(tail_lo, Q_TAIL_LOWER);
    TASSIGN(tail_hi, Q_TAIL_UPPER);
    KVTranspose key_lo, key_hi;
    TASSIGN(key_lo, (KV_LOWER_BASE + slot * KV_STRIDE));
    TASSIGN(key_hi, (KV_UPPER_BASE + slot * KV_STRIDE));
    QKLeft a[4];
    QKRight b[4];
    for (int stage = 0; stage < 4; ++stage) {
        TASSIGN(a[stage], stage * 16384);
        TASSIGN(b[stage], stage * 16384);
    }
    QKAcc acc;
    TASSIGN(acc, QK_ACC_BASE);
    if (task.first) {
        wait_flag(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(q_slot));
        wait_flag(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(EV_Q_TAIL));
    }
    wait_intra_block(PIPE_MTE1, FLAG_KV_READY + slot);
    wait_intra_block(PIPE_MTE1, FLAG_KV_READY + slot + AIV_BANK);
    wait_flag(PIPE_FIX, PIPE_M, static_cast<event_t>(EV_FIX_QK));
    for (int chunk = 0; chunk < QK_CHUNKS; ++chunk) {
        const int stage = chunk & 3;
        wait_flag(PIPE_M, PIPE_MTE1, static_cast<event_t>(stage));
        if (chunk < 6) {
            extract_qk_left_half(a[stage], wide_lo, 0, chunk);
            extract_qk_left_half(a[stage], wide_hi, 1, chunk);
        } else {
            extract_qk_left_half(a[stage], tail_lo, 0, chunk - 6);
            extract_qk_left_half(a[stage], tail_hi, 1, chunk - 6);
        }
        set_flag(PIPE_MTE1, PIPE_M, static_cast<event_t>(stage));
        wait_flag(PIPE_M, PIPE_MTE1, static_cast<event_t>((EV_L0B_BASE + stage)));
        extract_qk_right_half(b[stage], key_lo, 0, chunk);
        extract_qk_right_half(b[stage], key_hi, 1, chunk);
        set_flag(PIPE_MTE1, PIPE_M, static_cast<event_t>((EV_L0B_BASE + stage)));
        wait_flag(PIPE_MTE1, PIPE_M, static_cast<event_t>(stage));
        wait_flag(PIPE_MTE1, PIPE_M, static_cast<event_t>((EV_L0B_BASE + stage)));
        if (chunk == 0) {
            TMATMUL<AccPhase::Partial>(acc, a[stage], b[stage]);
        } else if (chunk + 1 == QK_CHUNKS) {
            TMATMUL_ACC<AccPhase::Final>(acc, a[stage], b[stage]);
        } else {
            TMATMUL_ACC<AccPhase::Partial>(acc, a[stage], b[stage]);
        }
        set_flag(PIPE_M, PIPE_MTE1, static_cast<event_t>(stage));
        set_flag(PIPE_M, PIPE_MTE1, static_cast<event_t>((EV_L0B_BASE + stage)));
    }
    if (task.last) {
        set_flag(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(q_slot));
        set_flag(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(EV_Q_TAIL));
    }
    wait_both<PIPE_FIX>(FLAG_SCORE_FREE);
    Score score;
    TASSIGN(score, SCORE_UB);
    store_qk_final(score.data(), acc.data());
    signal_both<PIPE_FIX>(FLAG_SCORE_READY);
    set_flag(PIPE_FIX, PIPE_M, static_cast<event_t>(EV_FIX_QK));
}

__aicore__ inline void compute_pv(__gm__ int64_t* args, const TileTask& task) {
    set_ctrl(sbitset1(get_ctrl(), 51));
    const int slot = task.ordinal % KV_SLOTS;
    const int p_stage = task.ordinal & 3;
    PTile p_l1;
    KVTile kv_lo, kv_hi;
    TASSIGN(p_l1, P_L1);
    TASSIGN(kv_lo, (KV_LOWER_BASE + slot * KV_STRIDE));
    TASSIGN(kv_hi, (KV_UPPER_BASE + slot * KV_STRIDE));
    PVLeft p;
    TASSIGN(p, p_stage * 16384);
    PVRightHalf v_lo[4], v_hi[4];
    for (int stage = 0; stage < 4; ++stage) {
        TASSIGN(v_lo[stage], stage * 16384);
        TASSIGN(v_hi[stage], stage * 16384 + 8192);
    }
    PVAcc acc[PV_CHUNKS];
    for (int i = 0; i < PV_CHUNKS; ++i) TASSIGN(acc[i], i * PV_ACC_STRIDE);
    wait_both<PIPE_MTE1>(FLAG_P_READY);
    wait_flag(PIPE_M, PIPE_MTE1, static_cast<event_t>(p_stage));
    TEXTRACT(p, p_l1, 0, 0);
    set_flag(PIPE_MTE1, PIPE_M, static_cast<event_t>(p_stage));
    signal_both<PIPE_MTE1>(FLAG_P_FREE);
    wait_flag(PIPE_MTE1, PIPE_M, static_cast<event_t>(p_stage));
    const int rows = task.first ? 0 : VEC_HEADS;
    for (int group = 0; group < 2; ++group) {
        Channel channel;
        TASSIGN(channel, group * PV_CHANNEL_BYTES);
        for (int index = 0; index < 4; ++index) {
            const int chunk = group * 4 + index;
            const int stage = chunk & 3;
            wait_flag(PIPE_M, PIPE_MTE1, static_cast<event_t>((EV_L0B_BASE + stage)));
            TEXTRACT(v_lo[stage], kv_lo, 0, chunk * KEY_CHUNK);
            TEXTRACT(v_hi[stage], kv_hi, 0, chunk * KEY_CHUNK);
            set_flag(PIPE_MTE1, PIPE_M, static_cast<event_t>((EV_L0B_BASE + stage)));
            if (chunk == PV_CHUNKS - 1) signal_both<PIPE_MTE1>(FLAG_KV_FREE + slot);
            wait_flag(PIPE_MTE1, PIPE_M, static_cast<event_t>((EV_L0B_BASE + stage)));
            PVRight v;
            TASSIGN(v, stage * 16384);
            const int accumulator_lock = PV_ACC_LOCK_BASE + chunk;
            get_buf(PIPE_M, accumulator_lock, false);
            TMATMUL(acc[chunk], p, v);
            rls_buf(PIPE_M, accumulator_lock, false);
            set_flag(PIPE_M, PIPE_MTE1, static_cast<event_t>((EV_L0B_BASE + stage)));
            if (index == 0) wait_both<PIPE_FIX>(FLAG_PV_FREE + group);
            get_buf(PIPE_FIX, accumulator_lock, false);
            TINSERT<Channel, PVAcc, AccToVecMode::DualModeSplitM>(channel, acc[chunk], rows,
                                                                  index * KEY_CHUNK);
            rls_buf(PIPE_FIX, accumulator_lock, false);
        }
        signal_both<PIPE_FIX>(FLAG_PV_READY + group);
    }
    // P is live through all eight K=128 MMADs.
    set_flag(PIPE_M, PIPE_MTE1, static_cast<event_t>(p_stage));
}

template <int Parts, bool Balanced>
__aicore__ inline void attend_cube(__gm__ int64_t* args) {
    const WorkPlan plan = work_plan<Parts, Balanced>(args);
    const int queries = plan.queries;
    const int tiles = plan.tiles;
    if (tiles == 0) return;
    for (int stage = 0; stage < 2; ++stage) {
        set_flag(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(stage));
    }
    set_flag(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(EV_Q_TAIL));
    for (int stage = 0; stage < 4; ++stage) {
        set_flag(PIPE_M, PIPE_MTE1, static_cast<event_t>(stage));
        set_flag(PIPE_M, PIPE_MTE1, static_cast<event_t>((EV_L0B_BASE + stage)));
    }
    set_flag(PIPE_FIX, PIPE_M, static_cast<event_t>(EV_FIX_QK));
    for (int slot = 0; slot < KV_SLOTS; ++slot) signal_both<PIPE_MTE1>(FLAG_KV_FREE + slot);
    signal_both<PIPE_MTE1>(FLAG_P_FREE);

    // Prologue QK(0), then QK(next) before PV(current), including query boundaries.
    load_q_wide(args, plan.first_token, 0);
    load_q_tail(args, plan.first_token);
    compute_qk(args, make_task<Parts, Balanced>(args, plan, 0));
    for (int ordinal = 0; ordinal < tiles; ++ordinal) {
        const TileTask task = make_task<Parts, Balanced>(args, plan, ordinal);
        if (task.first && task.query + 1 < queries) load_q_wide(args, task.token + (Balanced ? 1 : plan.workers), (task.query + 1) & 1);
        if (!task.last) {
            compute_qk(args, make_task<Parts, Balanced>(args, plan, ordinal + 1));
        } else if (task.query + 1 < queries) {
            load_q_tail(args, task.token + (Balanced ? 1 : plan.workers));
            compute_qk(args, make_task<Parts, Balanced>(args, plan, ordinal + 1));
        }
        compute_pv(args, task);
    }
    for (int stage = 0; stage < 4; ++stage) {
        wait_flag(PIPE_M, PIPE_MTE1, static_cast<event_t>(stage));
        wait_flag(PIPE_M, PIPE_MTE1, static_cast<event_t>((EV_L0B_BASE + stage)));
    }
    for (int stage = 0; stage < 2; ++stage) {
        wait_flag(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(stage));
    }
    wait_flag(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(EV_Q_TAIL));
    wait_flag(PIPE_FIX, PIPE_M, static_cast<event_t>(EV_FIX_QK));
    wait_both<PIPE_FIX>(FLAG_SCORE_FREE);
    wait_both<PIPE_FIX>(FLAG_PV_FREE);
    wait_both<PIPE_FIX>(FLAG_PV_FREE + 1);
    pipe_barrier(PIPE_ALL);
    wait_both<PIPE_MTE1>(FINISH_ACK);
    pipe_barrier(PIPE_ALL);
    signal_both<PIPE_FIX>(FINISH_READY);
}
#endif

#if defined(__DAV_VEC__) && !defined(__CPU_SIM)
template <bool FP4>
__aicore__ inline void gather_rows(__gm__ uint8_t* cache, __gm__ const int32_t* indices, int raw_base) {
    constexpr int row_bytes = FP4 ? 320 : 544;
    constexpr uint64_t cache_line_bytes = 64;
    const uint64_t first_line = reinterpret_cast<uint64_t>(indices) & ~(cache_line_bytes - 1);
    const uint64_t last_line = reinterpret_cast<uint64_t>(indices + VEC_KEYS - 1) & ~(cache_line_bytes - 1);
    // Refresh every scalar index cache line, including unaligned tensor views.
    for (uint64_t line = first_line; line <= last_line; line += cache_line_bytes) {
        dcci(reinterpret_cast<__gm__ void*>(line), cache_line_t::SINGLE_CACHE_LINE);
    }
    dsb(DSB_DDR);
    for (int row = 0; row < VEC_KEYS; row += 2) {
        const int a = indices[row];
        const int b = indices[row + 1];
        const int first = a < b ? a : b;
        const int second = a < b ? b : a;
        if (first >= 0 && second > first) {
            Tile<TileType::Vec, uint8_t, 2, 544, BLayout::RowMajor, 2, row_bytes> pair;
            TASSIGN(pair, raw_base + row * 544);
            using PairStride = pto::Stride<1, 1, 1, DYNAMIC, 1>;
            using PairShape = Shape<1, 1, 1, 2, row_bytes>;
            GlobalTensor<uint8_t, PairShape, PairStride> input(
                cache + (int64_t)first * row_bytes, PairShape(),
                PairStride(1, 1, 1, (int64_t)(second - first) * row_bytes, 1));
            TLOAD(pair, input);
        } else {
            for (int index = 0; index < 2; ++index) {
                const int physical = index == 0 ? a : b;
                if (physical < 0) continue;
                Vec<uint8_t, 1, row_bytes> value;
                TASSIGN(value, raw_base + (row + index) * 544);
                GM<uint8_t, 1, row_bytes> input(cache + (int64_t)physical * row_bytes);
                TLOAD(value, input);
            }
        }
    }
}

__aicore__ inline void gather_half(__gm__ int64_t* args, const TileTask& task, int half) {
    const int lane = get_sub_block_id(args);
    const int slot = (task.ordinal * 2 + half) % RAW_SLOTS;
    const int raw_base = RAW_UB + slot * RAW_STRIDE;
    const int column = task.part * TILE_KEYS + half * KEY_HALF + lane * VEC_KEYS;
    const int source = task.compressed ? ARG_COMPRESSED_INDICES : ARG_WINDOW_INDICES;
    const int index_column = task.compressed ? column - args[ARG_WINDOW_KEYS] : column;
    GatherIds ids;
    TASSIGN(ids, GATHER_ID_UB + slot * GATHER_ID_STRIDE);
    auto indices = data<int32_t>(args, source) + (int64_t)task.token * row_stride(args, source) + index_column;
    GM<int32_t, 1, VEC_KEYS> ids_gm(indices);
    wait_flag(PIPE_V, PIPE_MTE2, static_cast<event_t>((EV_RAW0 + slot)));
    TLOAD(ids, ids_gm);
    if (task.compressed) {
        gather_rows<true>(data<uint8_t>(args, ARG_COMPRESSED), indices, raw_base);
    } else {
        gather_rows<false>(data<uint8_t>(args, ARG_WINDOW), indices, raw_base);
    }
    set_flag(PIPE_MTE2, PIPE_V, static_cast<event_t>((EV_RAW0 + slot)));
}

__aicore__ inline void dequant_half(__gm__ int64_t* args, const TileTask& task, int half) {
    const int lane = get_sub_block_id(args);
    const int ring = (task.ordinal * 2 + half) % RAW_SLOTS;
    const int raw_base = RAW_UB + ring * RAW_STRIDE;
    const int id_base = GATHER_ID_UB + ring * GATHER_ID_STRIDE;
    const int kv_slot = task.ordinal % KV_SLOTS;
    const int kv_base = half == 0 ? (KV_LOWER_BASE + kv_slot * KV_STRIDE) : (KV_UPPER_BASE + kv_slot * KV_STRIDE);
    NzChunk nz[2];
    TASSIGN(nz[0], NZ_UB + 0 * NZ_STRIDE);
    TASSIGN(nz[1], NZ_UB + 1 * NZ_STRIDE);
    wait_flag(PIPE_MTE2, PIPE_V, static_cast<event_t>((EV_RAW0 + ring)));
    for (int chunk = 0; chunk < 2; ++chunk) {
        ChunkIds ids;
        Vec<uint8_t, 16, 544> packed_rows;
        TASSIGN(ids, id_base + chunk * 16 * (int)sizeof(int32_t));
        TASSIGN(packed_rows, raw_base + chunk * 16 * 544);
        wait_flag(PIPE_MTE3, PIPE_V, static_cast<event_t>((EV_NZ0 + chunk)));
        if (task.compressed) {
            sparse_dequant::decode_nz<true, NzChunk, decltype(packed_rows), ChunkIds>(
                nz[chunk].data(), packed_rows.data(), ids.data());
        } else {
            sparse_dequant::decode_nz<false, NzChunk, decltype(packed_rows), ChunkIds>(
                nz[chunk].data(), packed_rows.data(), ids.data());
        }
        set_flag(PIPE_V, PIPE_MTE3, static_cast<event_t>((EV_NZ0 + chunk)));
    }
    set_flag(PIPE_V, PIPE_MTE2, static_cast<event_t>((EV_RAW0 + ring)));
    Mat<KEY_HALF, DIM> kv;
    TASSIGN(kv, kv_base);
    // The KV slot free/ready handshake is per (slot, tile), not per half: the caller
    // waits FLAG_KV_FREE once before the two half inserts and publishes
    // FLAG_KV_READY once after both, so the AIC sees a whole 128-key slot.
    for (int chunk = 0; chunk < 2; ++chunk) {
        wait_flag(PIPE_V, PIPE_MTE3, static_cast<event_t>((EV_NZ0 + chunk)));
        TINSERT(kv, nz[chunk], lane * VEC_KEYS + chunk * 16, 0);
        set_flag(PIPE_MTE3, PIPE_V, static_cast<event_t>((EV_NZ0 + chunk)));
    }
}

__aicore__ inline void softmax_publish(__gm__ int64_t* args, const TileTask& task, int tick) {
    const int lane = get_sub_block_id(args);
    const int column = task.compressed ? task.part * TILE_KEYS - args[ARG_WINDOW_KEYS] : task.part * TILE_KEYS;
    const int source = task.compressed ? ARG_COMPRESSED_INDICES : ARG_WINDOW_INDICES;
    SoftmaxIds ids;
    TASSIGN(ids, SOFTMAX_ID_UB);
    GM<int32_t, 1, TILE_KEYS> ids_gm(
        data<int32_t>(args, source) + (int64_t)task.token * row_stride(args, source) + column);
    wait_flag(PIPE_V, PIPE_MTE2, static_cast<event_t>(EV_SOFTMAX_ID));
    TLOAD(ids, ids_gm);
    set_flag(PIPE_MTE2, PIPE_V, static_cast<event_t>(EV_SOFTMAX_ID));
    wait_flag(PIPE_MTE2, PIPE_V, static_cast<event_t>(EV_SOFTMAX_ID));
    wait_intra_block(PIPE_V, FLAG_SCORE_READY);
    Score score;
    TASSIGN(score, SCORE_UB);
    State maximum, denominator, next_max, correction;
    TASSIGN(maximum, MAX_STATE_UB + task.query_slot * STATE_SLOT_STRIDE);
    TASSIGN(next_max, MAX_STATE_UB + task.query_slot * STATE_SLOT_STRIDE + VEC_HEADS * (int)sizeof(float));
    TASSIGN(denominator, DEN_STATE_UB + task.query_slot * STATE_SLOT_STRIDE);
    TASSIGN(correction, ALPHA_STATE_UB + (tick & 1) * STATE_SLOT_STRIDE);
    if (task.first) {
        // Seed the online-softmax state of this query before its first tile.
        TEXPANDS(maximum, -1e30f);
        TEXPANDS(denominator, 0.0f);
    }
    wait_flag(PIPE_MTE3, PIPE_V, static_cast<event_t>(EV_P_STAGE));
    PStaging weights;
    TASSIGN(weights, P_STAGING_UB);
    sparse_softmax::update<decltype(weights), decltype(score), decltype(ids), decltype(maximum)>(
        weights.data(), next_max.data(), denominator.data(), correction.data(), score.data(), ids.data(),
        maximum.data(), denominator.data());
    TMOV(maximum, next_max);
    set_intra_block(PIPE_V, FLAG_SCORE_FREE);
    set_flag(PIPE_V, PIPE_MTE2, static_cast<event_t>(EV_SOFTMAX_ID));
    set_flag(PIPE_V, PIPE_MTE3, static_cast<event_t>(EV_P_STAGE));
    PTile p_l1;
    TASSIGN(p_l1, P_L1);
    wait_flag(PIPE_V, PIPE_MTE3, static_cast<event_t>(EV_P_STAGE));
    wait_intra_block(PIPE_MTE3, FLAG_P_FREE);
    TINSERT(p_l1, weights, lane * VEC_HEADS, 0);
    set_intra_block(PIPE_MTE3, FLAG_P_READY);
    set_flag(PIPE_MTE3, PIPE_V, static_cast<event_t>(EV_P_STAGE));
}

__aicore__ inline void store_channel(__gm__ int64_t* args, const TileTask& task, int channel) {
    set_flag(PIPE_V, PIPE_MTE3, static_cast<event_t>(channel));
    wait_flag(PIPE_V, PIPE_MTE3, static_cast<event_t>(channel));
    const int valid = valid_heads_of(args);
    const int head = get_sub_block_id(args) * VEC_HEADS;
    const int valid_local = valid > head ? (valid - head < VEC_HEADS ? valid - head : VEC_HEADS) : 0;
    if (valid_local > 0) {
        Tile<TileType::Vec, bfloat16_t, VEC_HEADS, 256, BLayout::RowMajor, DYNAMIC, 256> active(valid_local);
        TASSIGN(active, channel * PV_CHANNEL_BYTES + PV_ROW_HALF_BYTES);
        using OutShape = TileShape2D<bfloat16_t, DYNAMIC, 256>;
        using OutBase = BaseShape2D<bfloat16_t, CUBE_HEADS, DIM>;
        __gm__ bfloat16_t* out = data<bfloat16_t>(args, ARG_OUTPUT) +
                                (int64_t)task.token * row_stride(args, ARG_OUTPUT) + (int64_t)head * DIM +
                                channel * 256;
        GlobalTensor<bfloat16_t, OutShape, OutBase> out_gm(out, OutShape(valid_local, 256));
        TSTORE(out_gm, active);
    }
    set_flag(PIPE_MTE3, PIPE_V, static_cast<event_t>(channel));
    wait_flag(PIPE_MTE3, PIPE_V, static_cast<event_t>(channel));
    set_intra_block(PIPE_V, FLAG_PV_FREE + channel);
}

__aicore__ inline void store_partial_channel(
    __gm__ int64_t* args, const TileTask& task, int channel, ChannelHalf& acc,
    State& maximum, State& denominator) {
    const int head = get_sub_block_id(args) * VEC_HEADS;
    auto base = data<float>(args, ARG_PARTIALS) + (int64_t)task.partial_slot * PARTIAL_STRIDE;
    set_flag(PIPE_V, PIPE_MTE3, static_cast<event_t>(channel));
    wait_flag(PIPE_V, PIPE_MTE3, static_cast<event_t>(channel));
    using Shape = TileShape2D<float, VEC_HEADS, 256>;
    using Base = BaseShape2D<float, CUBE_HEADS, DIM>;
    GlobalTensor<float, Shape, Base> numerator(base + head * DIM + channel * 256);
    TSTORE(numerator, acc);
    if (channel == 0) {
        GM<float, 1, VEC_HEADS> max_gm(base + CUBE_HEADS * DIM + head);
        GM<float, 1, VEC_HEADS> den_gm(base + CUBE_HEADS * (DIM + 1) + head);
        TSTORE(max_gm, maximum);
        TSTORE(den_gm, denominator);
    }
    set_flag(PIPE_MTE3, PIPE_V, static_cast<event_t>(channel));
    wait_flag(PIPE_MTE3, PIPE_V, static_cast<event_t>(channel));
    set_intra_block(PIPE_V, FLAG_PV_FREE + channel);
    if (channel == 1 && task.partial_slot % 2 == 0) {
        // Publish the leading fragment after both numerator channels and statistics.
        pipe_barrier(PIPE_ALL);
        const int half = get_sub_block_id(args);
        st_dev(uint32_t(1), ready_flags(args) + task.partial_slot + half, 0);
    }
}

__aicore__ inline void update_pv(__gm__ int64_t* args, const TileTask& task, int tick) {
    const int alpha_slot = (tick - 1) & 1;
    State maximum, denominator, sink, alpha, factor;
    TASSIGN(maximum, MAX_STATE_UB + task.query_slot * STATE_SLOT_STRIDE);
    TASSIGN(denominator, DEN_STATE_UB + task.query_slot * STATE_SLOT_STRIDE);
    TASSIGN(sink, SINK_UB);
    TASSIGN(alpha, ALPHA_STATE_UB + alpha_slot * STATE_SLOT_STRIDE);
    TASSIGN(factor, FINAL_CORRECTION_UB);
    if (task.last && task.partial_slot < 0) {
        sparse_softmax::normalization<State>(factor.data(), maximum.data(), denominator.data(), sink.data());
    }
    for (int channel = 0; channel < 2; ++channel) {
        ChannelHalf acc, partial;
        ResultBf16 result;
        TASSIGN(acc, channel * PV_CHANNEL_BYTES);
        TASSIGN(partial, channel * PV_CHANNEL_BYTES + PV_ROW_HALF_BYTES);
        TASSIGN(result, channel * PV_CHANNEL_BYTES + PV_ROW_HALF_BYTES);
        wait_intra_block(PIPE_V, FLAG_PV_READY + channel);
        if (task.last && task.partial_slot >= 0) {
            if (!task.first) {
                sparse_softmax::accumulate<ChannelHalf, State>(acc.data(), acc.data(), partial.data(), alpha.data());
            }
            store_partial_channel(args, task, channel, acc, maximum, denominator);
        } else if (task.last) {
            if (task.first) {
                sparse_softmax::finish<true, ResultBf16, ChannelHalf, State>(
                    result.data(), acc.data(), acc.data(), alpha.data(), factor.data());
            } else {
                sparse_softmax::finish<false, ResultBf16, ChannelHalf, State>(
                    result.data(), acc.data(), partial.data(), alpha.data(), factor.data());
            }
            store_channel(args, task, channel);
        } else {
            if (!task.first) {
                sparse_softmax::accumulate<ChannelHalf, State>(acc.data(), acc.data(), partial.data(), alpha.data());
            }
            set_intra_block(PIPE_V, FLAG_PV_FREE + channel);
        }
    }
}

__aicore__ inline void load_sink(__gm__ int64_t* args) {
    const int head = get_sub_block_id(args) * VEC_HEADS;
    const int valid = valid_heads_of(args);
    const int valid_local = valid > head ? (valid - head < VEC_HEADS ? valid - head : VEC_HEADS) : 0;
    State sink;
    TASSIGN(sink, SINK_UB);
    auto sink_data = data<float>(args, ARG_SINK);
    const int64_t sink_stride = row_stride(args, ARG_SINK);
    if (sink_stride == 1) {
        set_flag(PIPE_V, PIPE_MTE2, static_cast<event_t>(EV_SINK));
        wait_flag(PIPE_V, PIPE_MTE2, static_cast<event_t>(EV_SINK));
        if (valid_local > 0) {
            Tile<TileType::Vec, float, 1, VEC_HEADS, BLayout::RowMajor, 1, DYNAMIC> load(valid_local);
            TASSIGN(load, SINK_UB);
            GlobalTensor<float, TileShape2D<float, 1, DYNAMIC>, BaseShape2D<float, 1, CUBE_HEADS>> sink_gm(
                sink_data + head, TileShape2D<float, 1, DYNAMIC>(1, valid_local));
            TLOAD(load, sink_gm);
        }
        set_flag(PIPE_MTE2, PIPE_V, static_cast<event_t>(EV_SINK));
        wait_flag(PIPE_MTE2, PIPE_V, static_cast<event_t>(EV_SINK));
    } else {
        set_flag(PIPE_V, PIPE_S, static_cast<event_t>(EV_SINK));
        wait_flag(PIPE_V, PIPE_S, static_cast<event_t>(EV_SINK));
        for (int lane = 0; lane < VEC_HEADS; ++lane) {
            sink.SetValue(lane, lane < valid_local ? sink_data[(int64_t)(head + lane) * sink_stride] : 0.0f);
        }
        set_flag(PIPE_S, PIPE_V, static_cast<event_t>(EV_SINK));
        wait_flag(PIPE_S, PIPE_V, static_cast<event_t>(EV_SINK));
    }
}

template <int Parts, bool Balanced>
__aicore__ inline void attend_vector(__gm__ int64_t* args) {
    const WorkPlan plan = work_plan<Parts, Balanced>(args);
    const int queries = plan.queries;
    const int tiles = plan.tiles;
    if (tiles == 0) return;
    set_flag(PIPE_V, PIPE_MTE2, static_cast<event_t>(EV_SOFTMAX_ID));
    set_flag(PIPE_V, PIPE_MTE2, static_cast<event_t>(EV_RAW0));
    set_flag(PIPE_V, PIPE_MTE2, static_cast<event_t>(EV_RAW1));
    set_flag(PIPE_V, PIPE_MTE2, static_cast<event_t>(EV_RAW2));
    set_flag(PIPE_MTE3, PIPE_V, static_cast<event_t>(EV_NZ0));
    set_flag(PIPE_MTE3, PIPE_V, static_cast<event_t>(EV_NZ1));
    set_flag(PIPE_MTE3, PIPE_V, static_cast<event_t>(EV_P_STAGE));
    set_intra_block(PIPE_V, FLAG_SCORE_FREE);
    set_intra_block(PIPE_V, FLAG_PV_FREE);
    set_intra_block(PIPE_V, FLAG_PV_FREE + 1);
    load_sink(args);
    for (int tick = 0; tick < tiles + 4; ++tick) {
        const bool has_current = tick < tiles;
        const bool has_previous = tick >= 1 && tick - 1 < tiles;
        const bool has_softmax = tick >= 3 && tick - 3 < tiles;
        const bool has_update = tick >= 4 && tick - 4 < tiles;
        // One KV ring slot is filled by both halves of both AIVs per tile, so the slot
        // free/ready handshake is hoisted out of the two dequant_half calls.
        if (has_previous) wait_intra_block(PIPE_MTE3, FLAG_KV_FREE + (tick - 1) % KV_SLOTS);
        if (has_current) gather_half(args, make_task<Parts, Balanced>(args, plan, tick), 0);
        if (has_previous) dequant_half(args, make_task<Parts, Balanced>(args, plan, tick - 1), 0);
        if (has_softmax) softmax_publish(args, make_task<Parts, Balanced>(args, plan, tick - 3), tick);
        if (has_current) gather_half(args, make_task<Parts, Balanced>(args, plan, tick), 1);
        if (has_previous) dequant_half(args, make_task<Parts, Balanced>(args, plan, tick - 1), 1);
        if (has_previous) set_intra_block(PIPE_MTE3, FLAG_KV_READY + (tick - 1) % KV_SLOTS);
        if (has_update) update_pv(args, make_task<Parts, Balanced>(args, plan, tick - 4), tick);
    }
    // Retirement: consume exactly one credit per still-outstanding flag, no more.
    //   MTE3->V P staging : 1 seed  - 1 per softmax consume + 1 per softmax publish = 1
    //   MTE3->V NZ0/NZ1   : 2 seeds - 2 per dequant consume + 2 per dequant publish = 2
    // (the V->MTE3 ready sides of P staging and NZ are consumed once per tile and carry
    //  no seed, so they are already balanced and must not be waited again.)
    wait_flag(PIPE_MTE3, PIPE_V, static_cast<event_t>(EV_P_STAGE));
    wait_flag(PIPE_MTE3, PIPE_V, static_cast<event_t>(EV_NZ0));
    wait_flag(PIPE_MTE3, PIPE_V, static_cast<event_t>(EV_NZ1));
    // Same rule for the two rings whose free credits the AIC never consumes:
    //   V->MTE2 softmax ID : 1 seed  - 1 per softmax acquire + 1 per softmax publish = 1
    //   V->MTE2 RAW0/1/2   : 3 seeds - 1 per gather acquire  + 1 per dequant publish = 3
    // RAW free is published by dequant_half and consumed by the NEXT gather of that slot,
    // so the acquire/publish multisets are identical over any full tile count.
    wait_flag(PIPE_V, PIPE_MTE2, static_cast<event_t>(EV_SOFTMAX_ID));
    wait_flag(PIPE_V, PIPE_MTE2, static_cast<event_t>(EV_RAW0));
    wait_flag(PIPE_V, PIPE_MTE2, static_cast<event_t>(EV_RAW1));
    wait_flag(PIPE_V, PIPE_MTE2, static_cast<event_t>(EV_RAW2));
    wait_intra_block(PIPE_MTE3, FLAG_P_FREE);
    for (int slot = 0; slot < KV_SLOTS; ++slot) {
        wait_intra_block(PIPE_MTE3, FLAG_KV_FREE + slot);
    }
    pipe_barrier(PIPE_ALL);
    set_intra_block(PIPE_MTE3, FINISH_ACK);
    wait_intra_block(PIPE_V, FINISH_READY);
    pipe_barrier(PIPE_ALL);
}
#endif

#if defined(__CPU_SIM)
float decode_byte(uint8_t code, bool fp4) {
    if (fp4) {
        const float magnitude[8] = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6};
        return magnitude[code & 7] * (code & 8 ? -1.0f : 1.0f);
    }
    int exponent = (code >> 3) & 15;
    int mantissa = code & 7;
    float value = exponent == 0 ? mantissa / 512.0f : std::ldexp(1.0f + mantissa / 8.0f, exponent - 7);
    if (exponent == 15 && mantissa == 7) value = std::numeric_limits<float>::quiet_NaN();
    return code & 128 ? -value : value;
}

void simulate_attention(__gm__ int64_t* args) {
    int heads = args[ARG_HEADS];
    int keys = args[ARG_WINDOW_KEYS] + args[ARG_COMPRESSED_KEYS];
    auto query = data<bfloat16_t>(args, ARG_QUERY);
    auto output = data<bfloat16_t>(args, ARG_OUTPUT);
    auto sink = data<float>(args, ARG_SINK);
    int64_t sink_stride = row_stride(args, ARG_SINK);
    for (int token = get_block_idx(args); token < args[ARG_TOKENS]; token += get_block_num(args)) {
        std::vector<float> numerator(heads * 512, 0.0f), maximum(heads, -1e30f), denominator(heads, 0.0f);
        std::vector<float> kv(128 * 512), scores(128);
        std::vector<bool> valid(128);
        for (int part = 0; part < keys; part += 128) {
            bool fp4 = part >= args[ARG_WINDOW_KEYS];
            int payload_arg = fp4 ? ARG_COMPRESSED : ARG_WINDOW;
            int index_arg = fp4 ? ARG_COMPRESSED_INDICES : ARG_WINDOW_INDICES;
            int width = fp4 ? 320 : 544;
            int groups = width / sizeof(bfloat16_t);
            auto payload = data<uint8_t>(args, payload_arg);
            auto scales = reinterpret_cast<__gm__ bfloat16_t*>(payload + (fp4 ? 256 : 512));
            auto indices = data<int32_t>(args, index_arg) + token * row_stride(args, index_arg);
            for (int k = 0; k < 128; ++k) {
                int physical = indices[part + k - (fp4 ? args[ARG_WINDOW_KEYS] : 0)];
                valid[k] = physical >= 0;
                for (int d = 0; d < 512; ++d) {
                    float value = 0;
                    if (valid[k]) {
                        uint8_t code = payload[(int64_t)physical * width + (fp4 ? d / 2 : d)];
                        if (fp4) code = (code >> ((d % 2) * 4)) & 15;
                        float factor = static_cast<float>(scales[(int64_t)physical * groups + d / (fp4 ? 16 : 32)]);
                        value = static_cast<float>(static_cast<bfloat16_t>(decode_byte(code, fp4) * factor));
                    }
                    kv[k * 512 + d] = value;
                }
            }
            for (int h = 0; h < heads; ++h) {
                float next = maximum[h];
                for (int k = 0; k < 128; ++k) {
                    float dot = 0;
                    if (valid[k]) {
                        for (int d = 0; d < 512; ++d) {
                            dot += static_cast<float>(query[token * row_stride(args, ARG_QUERY) + h * 512 + d]) *
                                   kv[k * 512 + d];
                        }
                    }
                    scores[k] = valid[k] ? dot * 0.0441941731f : -1e30f;
                    next = std::max(next, scores[k]);
                }
                float correction = std::exp(maximum[h] - next);
                denominator[h] *= correction;
                for (int d = 0; d < 512; ++d) numerator[h * 512 + d] *= correction;
                for (int k = 0; k < 128; ++k) {
                    float probability = valid[k] ? std::exp(scores[k] - next) : 0;
                    denominator[h] += probability;
                    float rounded = static_cast<float>(static_cast<bfloat16_t>(probability));
                    for (int d = 0; d < 512; ++d) numerator[h * 512 + d] += rounded * kv[k * 512 + d];
                }
                maximum[h] = next;
            }
        }
        for (int h = 0; h < heads; ++h) {
            float sink_value = sink[h * sink_stride];
            float final_max = std::max(maximum[h], sink_value);
            float correction = std::exp(maximum[h] - final_max);
            float factor = correction / (denominator[h] * correction + std::exp(sink_value - final_max));
            for (int d = 0; d < 512; ++d) {
                output[token * row_stride(args, ARG_OUTPUT) + h * 512 + d] =
                    static_cast<bfloat16_t>(numerator[h * 512 + d] * factor);
            }
        }
    }
}
#endif
}  // namespace

extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
    // ABI: O, Q, sink, W, C, Wi, Ci, partials; tokens, heads, Wkeys, Ckeys, balanced.
#if defined(__CPU_SIM) && defined(__DAV_VEC__)
    if (get_sub_block_id(args) == 0) {
        simulate_attention(args);
    }
#endif
#if defined(__DAV_CUBE__) && !defined(__CPU_SIM)
    const uint64_t saved_ctrl = get_ctrl();
    const int parts = (args[ARG_WINDOW_KEYS] + args[ARG_COMPRESSED_KEYS]) / TILE_KEYS;
    if (args[ARG_BALANCED]) {
        if (parts == 5) attend_cube<5, true>(args);
        else attend_cube<0, true>(args);
    } else {
        if (parts == 5) attend_cube<5, false>(args);
        else if (parts == 1) attend_cube<1, false>(args);
        else attend_cube<0, false>(args);
    }
    set_ctrl(saved_ctrl);
#elif defined(__DAV_VEC__) && !defined(__CPU_SIM)
    set_mask_norm();
    set_vector_mask(-1, -1);
    const int parts = (args[ARG_WINDOW_KEYS] + args[ARG_COMPRESSED_KEYS]) / TILE_KEYS;
    if (args[ARG_BALANCED]) {
        if (parts == 5) attend_vector<5, true>(args);
        else attend_vector<0, true>(args);
        const int boundary = get_block_idx(args) + 1;
        const int workers = get_block_num(args);
        const int cut = (int64_t)args[ARG_TOKENS] * parts * boundary / workers;
        const int head = get_sub_block_id(args) * VEC_HEADS;
        if (boundary < workers && cut % parts != 0 && head < args[ARG_HEADS]) {
            auto ready = ready_flags(args) + 2 * boundary + get_sub_block_id(args);
            while (ld_dev(ready, 0) != 1) {
                for (int pause = 0; pause < 50; ++pause) asm volatile("nop");
            }
            // Main-loop users are drained; the reducer reuses their UB storage.
            sparse_merge::reduce(
                data<float>(args, ARG_PARTIALS), data<bfloat16_t>(args, ARG_OUTPUT), data<float>(args, ARG_SINK),
                row_stride(args, ARG_OUTPUT), row_stride(args, ARG_SINK), args[ARG_HEADS], boundary, head, cut / parts);
        }
    } else {
        if (parts == 5) attend_vector<5, false>(args);
        else if (parts == 1) attend_vector<1, false>(args);
        else attend_vector<0, false>(args);
    }
#endif
}
