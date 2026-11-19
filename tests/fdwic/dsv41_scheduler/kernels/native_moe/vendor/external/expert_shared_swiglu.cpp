// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// Shared-expert SwiGLU epilogue and grouped MXFP8 requantization.
//
// One work item owns a 16-row x 64-column tile: 16 rows is the packed MX_A_ZZ
// scale granularity and 64 columns is one MX scale-group pair, so an item writes
// exactly one 32-byte packed scale word group and one 16 x 64 FP8 payload tile.
// Two independent UB slots carry gate, up, payload and scale bytes, and the
// MTE2 -> VF -> MTE3 stages are chained by readyIn/freeIn/readyOut/freeOut
// events so the next item's TLOAD overlaps the current item's VF.
//
#include <cstdint>
#if defined(__CPU_SIM)
#include <algorithm>
#include <cmath>
#include <cstring>
#endif
#include <pto/pto-inst.hpp>
#include "tensor.h"
#include "intrinsic.h"
using namespace pto;
namespace {
template <typename T, int R, int C> using Vec = Tile<TileType::Vec, T, R, C>;
template <typename T> __aicore__ inline __gm__ T* data(__gm__ int64_t* args, int i) {
    auto t = reinterpret_cast<__gm__ Tensor*>(args[i]);
    return reinterpret_cast<__gm__ T*>(t->buffer.addr) + t->start_offset;
}
__aicore__ inline int64_t stride(__gm__ int64_t* args, int i) {
    return reinterpret_cast<__gm__ Tensor*>(args[i])->strides[0];
}
// The extent a dynamic shape symbol stands for is the runtime shape of the
// declaring tensor, so the kernel reads it off the operand instead of taking a
// scalar argument per symbol: fewer payload words to publish per block.
__aicore__ inline int64_t dim(__gm__ int64_t* args, int i, int d) {
    return reinterpret_cast<__gm__ Tensor*>(args[i])->shapes[d];
}
#if defined(__DAV_VEC__) && !defined(__CPU_SIM)
constexpr int ROW_TILE = 16;
constexpr int GROUP = 32;
constexpr int COL_TILE = 2 * GROUP;                   // one MX scale-group pair
constexpr int PAIRS = COL_TILE / GROUP;               // scale bytes per row and item
constexpr int INPUT_BYTES = ROW_TILE * COL_TILE * 2;  // one BF16 tile
constexpr int PAYLOAD_BYTES = ROW_TILE * COL_TILE;    // one FP8 tile
constexpr int SCALE_BYTES = ROW_TILE * PAIRS;         // one packed MX_A_ZZ word group
// Two independent slots. Each holds gate, up, payload and scale bytes so a slot
// is reused only after both of its consumers (VF and TSTORE) released it.
constexpr int GATE_BASE = 0;
constexpr int UP_BASE = GATE_BASE + 2 * INPUT_BYTES;
constexpr int PAYLOAD_BASE = UP_BASE + 2 * INPUT_BYTES;
constexpr int SCALE_BASE = PAYLOAD_BASE + 2 * PAYLOAD_BYTES;
constexpr float SWIGLU_LIMIT = 10.0f;
constexpr float MIN_GROUP_MAX = 1e-4f;
constexpr float FP8_E4M3_MAX = 448.0f;

// Event ids per slot: readyIn(MTE2->V), freeIn(V->MTE2), readyOut(V->MTE3),
// freeOut(MTE3->V). CCEC only accepts an event literal, so the ids travel as
// template value parameters instead of an indexed table.

using GateTile = Vec<bfloat16_t, ROW_TILE, COL_TILE>;
using UpTile = Vec<bfloat16_t, ROW_TILE, COL_TILE>;
using PayloadTile = Vec<uint8_t, ROW_TILE, COL_TILE>;
using ScaleTile = Vec<uint8_t, 1, SCALE_BYTES>;

// Upward MXFP8: gate is clamped from above only, up from both sides, the gated
// value stays FP32, and every 32-value group takes its own E8M0 scale.
//
// The 16 rows of a tile are independent, so four rows are in flight at once and
// each opcode is issued across all four chains. The vector unit is deeply
// pipelined but a single row is a ~25 op serial dependency chain; interleaving
// four of them is what keeps the pipe busy.
constexpr int CHAINS = 4;
constexpr int ROW_GROUPS = ROW_TILE / CHAINS;

// Upcast a BF16 row, clamp it, and evaluate SiLU(gate) * up in FP32.
__aicore__ inline void activate4(__ubuf__ bfloat16_t* gsrc, __ubuf__ bfloat16_t* usrc, uint16_t base,
                                 RegTensor<float>& h0, RegTensor<float>& h1, RegTensor<float>& h2,
                                 RegTensor<float>& h3, MaskReg all, MaskReg all16) {
    RegTensor<bfloat16_t> b0, b1, b2, b3;
    RegTensor<float> g0, g1, g2, g3, u0, u1, u2, u3, s0, s1, s2, s3;
    vlds(b0, gsrc, (base + 0) * COL_TILE, UNPK_B16);
    vlds(b1, gsrc, (base + 1) * COL_TILE, UNPK_B16);
    vlds(b2, gsrc, (base + 2) * COL_TILE, UNPK_B16);
    vlds(b3, gsrc, (base + 3) * COL_TILE, UNPK_B16);
    vcvt(g0, b0, all16, PART_EVEN);
    vcvt(g1, b1, all16, PART_EVEN);
    vcvt(g2, b2, all16, PART_EVEN);
    vcvt(g3, b3, all16, PART_EVEN);
    vlds(b0, usrc, (base + 0) * COL_TILE, UNPK_B16);
    vlds(b1, usrc, (base + 1) * COL_TILE, UNPK_B16);
    vlds(b2, usrc, (base + 2) * COL_TILE, UNPK_B16);
    vlds(b3, usrc, (base + 3) * COL_TILE, UNPK_B16);
    vcvt(u0, b0, all16, PART_EVEN);
    vcvt(u1, b1, all16, PART_EVEN);
    vcvt(u2, b2, all16, PART_EVEN);
    vcvt(u3, b3, all16, PART_EVEN);
    vmins(g0, g0, SWIGLU_LIMIT, all);
    vmins(g1, g1, SWIGLU_LIMIT, all);
    vmins(g2, g2, SWIGLU_LIMIT, all);
    vmins(g3, g3, SWIGLU_LIMIT, all);
    vmins(u0, u0, SWIGLU_LIMIT, all);
    vmins(u1, u1, SWIGLU_LIMIT, all);
    vmins(u2, u2, SWIGLU_LIMIT, all);
    vmins(u3, u3, SWIGLU_LIMIT, all);
    vmaxs(u0, u0, -SWIGLU_LIMIT, all);
    vmaxs(u1, u1, -SWIGLU_LIMIT, all);
    vmaxs(u2, u2, -SWIGLU_LIMIT, all);
    vmaxs(u3, u3, -SWIGLU_LIMIT, all);
    vmuls(s0, g0, -1.0f, all);
    vmuls(s1, g1, -1.0f, all);
    vmuls(s2, g2, -1.0f, all);
    vmuls(s3, g3, -1.0f, all);
    vexp(s0, s0, all);
    vexp(s1, s1, all);
    vexp(s2, s2, all);
    vexp(s3, s3, all);
    vadds(s0, s0, 1.0f, all);
    vadds(s1, s1, 1.0f, all);
    vadds(s2, s2, 1.0f, all);
    vadds(s3, s3, 1.0f, all);
    vdiv(h0, g0, s0, all);
    vdiv(h1, g1, s1, all);
    vdiv(h2, g2, s2, all);
    vdiv(h3, g3, s3, all);
    vmul(h0, h0, u0, all);
    vmul(h1, h1, u1, all);
    vmul(h2, h2, u2, all);
    vmul(h3, h3, u3, all);
}

// Per-group upward E8M0 scale: one MX group is one half of a row, so each half is
// reduced to lane 0 and broadcast back over both halves. Emits the packed MX_A_ZZ
// scale word for the row and the per-group reciprocal as a float bit pattern.
__aicore__ inline void scale4(RegTensor<float>& h0, RegTensor<float>& h1, RegTensor<float>& h2,
                              RegTensor<float>& h3, __ubuf__ uint16_t* codes, uint16_t base,
                              MaskReg all, MaskReg lower, MaskReg upper, MaskReg one16,
                              RegTensor<uint32_t>& lo, RegTensor<uint32_t>& hi,
                              RegTensor<int32_t>& bias) {
    RegTensor<float> m0, m1, m2, m3, x0, x1, x2, x3, r0, r1, r2, r3;
    RegTensor<uint32_t> l0, l1, l2, l3, g0, g1, g2, g3, y0, y1, y2, y3;
    RegTensor<int32_t> e0, e1, e2, e3;
    vabs(m0, h0, all);
    vabs(m1, h1, all);
    vabs(m2, h2, all);
    vabs(m3, h3, all);
    vcmax(x0, m0, lower);
    vcmax(x1, m1, lower);
    vcmax(x2, m2, lower);
    vcmax(x3, m3, lower);
    vcmax(r0, m0, upper);
    vcmax(r1, m1, upper);
    vcmax(r2, m2, upper);
    vcmax(r3, m3, upper);
    vselr(g0, (RegTensor<uint32_t>&)x0, lo);
    vselr(g1, (RegTensor<uint32_t>&)x1, lo);
    vselr(g2, (RegTensor<uint32_t>&)x2, lo);
    vselr(g3, (RegTensor<uint32_t>&)x3, lo);
    vselr(y0, (RegTensor<uint32_t>&)r0, lo);
    vselr(y1, (RegTensor<uint32_t>&)r1, lo);
    vselr(y2, (RegTensor<uint32_t>&)r2, lo);
    vselr(y3, (RegTensor<uint32_t>&)r3, lo);
    vsel(x0, (RegTensor<float>&)g0, (RegTensor<float>&)y0, lower);
    vsel(x1, (RegTensor<float>&)g1, (RegTensor<float>&)y1, lower);
    vsel(x2, (RegTensor<float>&)g2, (RegTensor<float>&)y2, lower);
    vsel(x3, (RegTensor<float>&)g3, (RegTensor<float>&)y3, lower);
    vmaxs(x0, x0, MIN_GROUP_MAX, all);
    vmaxs(x1, x1, MIN_GROUP_MAX, all);
    vmaxs(x2, x2, MIN_GROUP_MAX, all);
    vmaxs(x3, x3, MIN_GROUP_MAX, all);
    vmuls(x0, x0, 1.0f / FP8_E4M3_MAX, all);
    vmuls(x1, x1, 1.0f / FP8_E4M3_MAX, all);
    vmuls(x2, x2, 1.0f / FP8_E4M3_MAX, all);
    vmuls(x3, x3, 1.0f / FP8_E4M3_MAX, all);
    // ceil(log2(x)) for positive finite x, offset by the E8M0 bias.
    vadds(e0, (RegTensor<int32_t>&)x0, 8388607, all);
    vadds(e1, (RegTensor<int32_t>&)x1, 8388607, all);
    vadds(e2, (RegTensor<int32_t>&)x2, 8388607, all);
    vadds(e3, (RegTensor<int32_t>&)x3, 8388607, all);
    vshrs(e0, e0, 23u, all);
    vshrs(e1, e1, 23u, all);
    vshrs(e2, e2, 23u, all);
    vshrs(e3, e3, 23u, all);
    vselr(l0, (RegTensor<uint32_t>&)e0, lo);
    vselr(l1, (RegTensor<uint32_t>&)e1, lo);
    vselr(l2, (RegTensor<uint32_t>&)e2, lo);
    vselr(l3, (RegTensor<uint32_t>&)e3, lo);
    vselr(g0, (RegTensor<uint32_t>&)e0, hi);
    vselr(g1, (RegTensor<uint32_t>&)e1, hi);
    vselr(g2, (RegTensor<uint32_t>&)e2, hi);
    vselr(g3, (RegTensor<uint32_t>&)e3, hi);
    vshls(g0, g0, 8u, all, MODE_ZEROING);
    vshls(g1, g1, 8u, all, MODE_ZEROING);
    vshls(g2, g2, 8u, all, MODE_ZEROING);
    vshls(g3, g3, 8u, all, MODE_ZEROING);
    vor(l0, l0, g0, all);
    vor(l1, l1, g1, all);
    vor(l2, l2, g2, all);
    vor(l3, l3, g3, all);
    vsts((RegTensor<uint16_t>&)l0, codes, base + 0, ONEPT_B16, one16);
    vsts((RegTensor<uint16_t>&)l1, codes, base + 1, ONEPT_B16, one16);
    vsts((RegTensor<uint16_t>&)l2, codes, base + 2, ONEPT_B16, one16);
    vsts((RegTensor<uint16_t>&)l3, codes, base + 3, ONEPT_B16, one16);
    // 2^-exponent as a float bit pattern, already broadcast per group.
    vsub(e0, bias, e0, all);
    vsub(e1, bias, e1, all);
    vsub(e2, bias, e2, all);
    vsub(e3, bias, e3, all);
    vshls(e0, e0, 23u, all);
    vshls(e1, e1, 23u, all);
    vshls(e2, e2, 23u, all);
    vshls(e3, e3, 23u, all);
    vmul(h0, h0, (RegTensor<float>&)e0, all);
    vmul(h1, h1, (RegTensor<float>&)e1, all);
    vmul(h2, h2, (RegTensor<float>&)e2, all);
    vmul(h3, h3, (RegTensor<float>&)e3, all);
}

// Clamp to +-448 and convert four rows to E4M3, storing 64 packed bytes per row.
__aicore__ inline void quantize4(RegTensor<float>& h0, RegTensor<float>& h1, RegTensor<float>& h2,
                                 RegTensor<float>& h3, __ubuf__ uint8_t* dst, uint16_t base,
                                 MaskReg all) {
    RegTensor<float8_e4m3_t> q0, q1, q2, q3;
    vmaxs(h0, h0, -FP8_E4M3_MAX, all);
    vmaxs(h1, h1, -FP8_E4M3_MAX, all);
    vmaxs(h2, h2, -FP8_E4M3_MAX, all);
    vmaxs(h3, h3, -FP8_E4M3_MAX, all);
    vmins(h0, h0, FP8_E4M3_MAX, all);
    vmins(h1, h1, FP8_E4M3_MAX, all);
    vmins(h2, h2, FP8_E4M3_MAX, all);
    vmins(h3, h3, FP8_E4M3_MAX, all);
    vcvt(q0, h0, all, ROUND_R, RS_ENABLE, PART_P0);
    vcvt(q1, h1, all, ROUND_R, RS_ENABLE, PART_P0);
    vcvt(q2, h2, all, ROUND_R, RS_ENABLE, PART_P0);
    vcvt(q3, h3, all, ROUND_R, RS_ENABLE, PART_P0);
    vsts((RegTensor<uint8_t>&)q0, dst, (base + 0) * COL_TILE, PK4_B32, all);
    vsts((RegTensor<uint8_t>&)q1, dst, (base + 1) * COL_TILE, PK4_B32, all);
    vsts((RegTensor<uint8_t>&)q2, dst, (base + 2) * COL_TILE, PK4_B32, all);
    vsts((RegTensor<uint8_t>&)q3, dst, (base + 3) * COL_TILE, PK4_B32, all);
}

template <typename Payload, typename Scales, typename Gate, typename Up>
__tf__ __aicore__ inline void swiglu_quant_vf(
    typename Payload::TileDType __out__ payload, typename Scales::TileDType __out__ scales,
    typename Gate::TileDType __in__ gate, typename Up::TileDType __in__ up) {
    auto gsrc = (__ubuf__ bfloat16_t*)__cce_get_tile_ptr(gate);
    auto usrc = (__ubuf__ bfloat16_t*)__cce_get_tile_ptr(up);
    auto dst = (__ubuf__ uint8_t*)__cce_get_tile_ptr(payload);
    auto codes = (__ubuf__ uint16_t*)__cce_get_tile_ptr(scales);
    __VEC_SCOPE__ {
        MaskReg all = pset_b32(PAT_ALL), all16 = pset_b16(PAT_ALL), lower = pset_b32(PAT_VL32);
        MaskReg upper;
        pnot(upper, lower, all);
        uint32_t count1 = 1;
        MaskReg one16 = CreatePredicate<uint16_t>(count1);
        RegTensor<uint32_t> lo, hi;
        vdup(lo, 0u, all, MODE_ZEROING);
        vdup(hi, 32u, all, MODE_ZEROING);
        RegTensor<int32_t> bias;
        vbr(bias, 254);
        for (uint16_t group = 0; group < ROW_GROUPS; ++group) {
            const uint16_t base = group * CHAINS;
            RegTensor<float> h0, h1, h2, h3;
            activate4(gsrc, usrc, base, h0, h1, h2, h3, all, all16);
            scale4(h0, h1, h2, h3, codes, base, all, lower, upper, one16, lo, hi, bias);
            quantize4(h0, h1, h2, h3, dst, base, all);
        }
    }
}

// Item index -> tile origin. Items run row tile major, so consecutive items of
// one block stay inside the same 16-row band.
__aicore__ inline int64_t item_row(int64_t work, int64_t pairsPerTile) {
    return work / pairsPerTile * ROW_TILE;
}
__aicore__ inline int64_t item_column(int64_t work, int64_t pairsPerTile) {
    return work % pairsPerTile * COL_TILE;
}

// Uniform contiguous work ranges with the original Q16 rounding.
__aicore__ inline int64_t range_start(int64_t index, int64_t tasks, int64_t base16) {
    const int64_t start = (index * base16 + 32768) >> 16;
    return start < 0 ? 0 : (start > tasks ? tasks : start);
}

__aicore__ inline void block_range(int64_t block, int64_t tasks, int64_t blocks, int64_t& first,
                                  int64_t& last) {
    const int64_t base16 = (tasks << 16) / blocks;
    first = block == 0 ? 0 : range_start(block, tasks, base16);
    last = block + 1 >= blocks ? tasks : range_start(block + 1, tasks, base16);
    if (last < first) last = first;
}

template <event_t ReadyIn, int Slot>
__aicore__ inline void prefetch(__gm__ bfloat16_t* gate, __gm__ bfloat16_t* up, int64_t row, int64_t column,
                                int64_t rowStride) {
    using InputShape = TileShape2D<bfloat16_t, ROW_TILE, COL_TILE>;
    using DynamicStride = pto::Stride<1, 1, 1, DYNAMIC, 1>;
    GateTile gateTile;
    UpTile upTile;
    TASSIGN(gateTile, GATE_BASE + Slot * INPUT_BYTES);
    TASSIGN(upTile, UP_BASE + Slot * INPUT_BYTES);
    GlobalTensor<bfloat16_t, InputShape, DynamicStride> gateGlobal(
        gate + row * rowStride + column, InputShape(), DynamicStride(1, 1, 1, rowStride, 1));
    GlobalTensor<bfloat16_t, InputShape, DynamicStride> upGlobal(
        up + row * rowStride + column, InputShape(), DynamicStride(1, 1, 1, rowStride, 1));
    TLOAD(gateTile, gateGlobal);
    TLOAD(upTile, upGlobal);
    set_flag(PIPE_MTE2, PIPE_V, ReadyIn);
}

template <int Slot>
__aicore__ inline void compute() {
    GateTile gateTile;
    UpTile upTile;
    PayloadTile payloadTile;
    ScaleTile scaleTile;
    TASSIGN(gateTile, GATE_BASE + Slot * INPUT_BYTES);
    TASSIGN(upTile, UP_BASE + Slot * INPUT_BYTES);
    TASSIGN(payloadTile, PAYLOAD_BASE + Slot * PAYLOAD_BYTES);
    TASSIGN(scaleTile, SCALE_BASE + Slot * SCALE_BYTES);
    swiglu_quant_vf<PayloadTile, ScaleTile, GateTile, UpTile>(
        payloadTile.data(), scaleTile.data(), gateTile.data(), upTile.data());
}

template <event_t FreeOut, int Slot>
__aicore__ inline void store(__gm__ uint8_t* payload, __gm__ uint8_t* scales, int64_t row, int64_t column,
                             int64_t rowStride, int64_t width) {
    using PayloadShape = TileShape2D<uint8_t, ROW_TILE, COL_TILE>;
    using ScaleShape = TileShape2D<uint8_t, 1, SCALE_BYTES>;
    using DynamicStride = pto::Stride<1, 1, 1, DYNAMIC, 1>;
    PayloadTile payloadTile;
    ScaleTile scaleTile;
    TASSIGN(payloadTile, PAYLOAD_BASE + Slot * PAYLOAD_BYTES);
    TASSIGN(scaleTile, SCALE_BASE + Slot * SCALE_BYTES);
    GlobalTensor<uint8_t, PayloadShape, DynamicStride> payloadGlobal(
        payload + row * rowStride + column, PayloadShape(), DynamicStride(1, 1, 1, rowStride, 1));
    // MX_A_ZZ: [row tile][scale pair][16 rows][2 groups], so one row tile owns
    // (width/64) * 32 scale bytes and one work item owns exactly one such word group.
    const int64_t pairsPerTile = width / COL_TILE;
    GlobalTensor<uint8_t, ScaleShape, BaseShape2D<uint8_t, 1, SCALE_BYTES>> scaleGlobal(
        scales + (row / ROW_TILE * pairsPerTile + column / COL_TILE) * SCALE_BYTES);
    TSTORE(payloadGlobal, payloadTile);
    TSTORE(scaleGlobal, scaleTile);
    set_flag(PIPE_MTE3, PIPE_V, FreeOut);
}

// One pipelined work item: consume the prefetched slot, relaunch its load for the
// item two slots ahead, then drain this slot's output. A slot's output region is
// reused only from the third iteration, so its FIRST wait is skipped.
template <event_t Event, int Slot>
__aicore__ inline void step(int64_t work, int64_t last, int64_t width, int64_t pairsPerTile,
                            __gm__ bfloat16_t* gate, __gm__ bfloat16_t* up, __gm__ uint8_t* payload,
                            __gm__ uint8_t* scales, int64_t gateStride, int64_t payloadStride, bool reuse) {
    wait_flag(PIPE_MTE2, PIPE_V, Event);
    if (reuse) {
        wait_flag(PIPE_MTE3, PIPE_V, Event);
    }
    compute<Slot>();
    set_flag(PIPE_V, PIPE_MTE2, Event);
    set_flag(PIPE_V, PIPE_MTE3, Event);
    const int64_t next = work + 2;
    if (next < last) {
        // The slot is reusable once this item's VF retires.
        wait_flag(PIPE_V, PIPE_MTE2, Event);
        prefetch<Event, Slot>(gate, up, item_row(next, pairsPerTile), item_column(next, pairsPerTile), gateStride);
    }
    wait_flag(PIPE_V, PIPE_MTE3, Event);
    store<Event, Slot>(payload, scales, item_row(work, pairsPerTile), item_column(work, pairsPerTile), payloadStride,
                       width);
}

__aicore__ inline void run_native(__gm__ int64_t* args) {
    const int64_t rows = dim(args, 0, 0);
    const int64_t width = dim(args, 1, 1);
    auto* gate = data<bfloat16_t>(args, 0);
    auto* up = gate + width;
    auto* payload = data<uint8_t>(args, 1);
    auto* scales = data<uint8_t>(args, 2);
    const int64_t gateStride = stride(args, 0), payloadStride = stride(args, 1);
    const int64_t pairsPerTile = width / COL_TILE;
    const int64_t tasks = rows / ROW_TILE * pairsPerTile;
    const int64_t block = get_block_idx(args), blocks = get_block_num(args);
    int64_t first = 0, last = 0;
    block_range(block, tasks, blocks, first, last);
    if (first >= last) {
        // An empty range owns no slot, so it never waits on a slot event.
        pipe_barrier(PIPE_ALL);
        return;
    }
    // Two slots: one event id per slot, reused by every stage of that slot. Each
    // pipe pair carries the id independently, so one id per slot is enough.
    prefetch<EVENT_ID0, 0>(gate, up, item_row(first, pairsPerTile), item_column(first, pairsPerTile), gateStride);
    if (first + 1 < last) {
        prefetch<EVENT_ID1, 1>(gate, up, item_row(first + 1, pairsPerTile),
                               item_column(first + 1, pairsPerTile), gateStride);
    }
    for (int64_t work = first, iteration = 0; work < last; ++work, ++iteration) {
        const bool reuse = iteration >= 2;
        if ((iteration & 1) == 0) {
            step<EVENT_ID0, 0>(work, last, width, pairsPerTile, gate, up, payload, scales, gateStride,
                               payloadStride, reuse);
        } else {
            step<EVENT_ID1, 1>(work, last, width, pairsPerTile, gate, up, payload, scales, gateStride,
                               payloadStride, reuse);
        }
    }
    // Drain the final input/output credits before another kernel reuses these events.
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    if (first + 1 < last) {
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    }
    pipe_barrier(PIPE_ALL);
}
#endif
} // namespace
extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
#if defined(__CPU_SIM)
    (void)args;
#elif defined(__DAV_VEC__)
    set_mask_norm(); set_vector_mask(-1, -1); run_native(args);
#endif
}
