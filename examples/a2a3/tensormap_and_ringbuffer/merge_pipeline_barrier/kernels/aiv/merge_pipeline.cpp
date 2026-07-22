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
 * The whole 3-stage pipeline (x+1 -> *2 -> +1) as ONE block_num=8 SPMD task,
 * with a uniform cross-core barrier between segments instead of inter-task
 * deps: stage A does 8 tiles on block 0 only, stage B 2 tiles each on blocks
 * 0..3, stage C 1 tile each on all 8. Every block hits every barrier (idle
 * blocks still arrive), so each barrier waits for block_num. Bulk data uses
 * tile DMA; the barrier uses ld_dev/st_dev (see MERGE_BARRIER_COUNTER below).
 *
 * args: [0]=x, [1]=sync (block_num int32 slots, zero-init), [2]=s1, [3]=s2,
 *       [4]=out, [5]=timing (per-block get_sys_cnt gaps, 32-int32 stride)
 */

#include <cstdint>
#include <pto/pto-inst.hpp>

#include "tensor.h"
#include "intrinsic.h"

using namespace pto;

#include "pipe_sync.h"

#ifndef __gm__
#define __gm__
#endif
#ifndef __aicore__
#define __aicore__ [aicore]
#endif

#if defined(__CCE_AICORE__) && !defined(__CPU_SIM) && !defined(__COSTMODEL)
#define MERGE_USE_DEV_INTRIN 1
#else
#define MERGE_USE_DEV_INTRIN 0
#endif

static __aicore__ inline void st_i32_dev(__gm__ int32_t *p, int32_t v) {
#if MERGE_USE_DEV_INTRIN
    st_dev(static_cast<uint32_t>(v), reinterpret_cast<__gm__ uint32_t *>(p), 0);
#else
    // CPU_SIM fallback is thread-based; volatile stops the spin loop hoisting.
    *reinterpret_cast<volatile int32_t *>(p) = v;
#endif
}

static __aicore__ inline int32_t ld_i32_dev(__gm__ int32_t *p) {
#if MERGE_USE_DEV_INTRIN
    return static_cast<int32_t>(ld_dev(reinterpret_cast<__gm__ uint32_t *>(p), 0));
#else
    return *reinterpret_cast<volatile int32_t *>(p);
#endif
}

static __aicore__ inline void ddr_fence() {
#if MERGE_USE_DEV_INTRIN
    dsb(DSB_DDR);
#endif
}

// Per-slot cross-core barrier over `n` blocks; slot advances monotonically per
// epoch. Single-writer per slot (no atomic), ld_dev poll (no dcci). Reader polls
// all n slots -> O(n) non-cacheable reads per barrier.
static __aicore__ inline void barrier_slots(__gm__ int32_t *sync, int32_t n, int32_t i, int32_t epoch) {
    ddr_fence();
    st_i32_dev(&sync[i], epoch);
    for (int32_t j = 0; j < n; ++j) {
        while (ld_i32_dev(&sync[j]) < epoch) {}
    }
    ddr_fence();
}

// Scalar hardware atomic-add to a GM int32 (no UB/DMA). The dcci writes the line
// out atomically; the reader polls with ld_dev, so no dcci on the hot path.
static __aicore__ inline void atomic_add_i32(__gm__ int32_t *p, int32_t v) {
#if MERGE_USE_DEV_INTRIN
    set_st_atomic_cfg(ATOMIC_S32, ATOMIC_SUM);
    dcci(reinterpret_cast<__gm__ void *>(p), SINGLE_CACHE_LINE);
    st_atomic<int32_t>(v, p);
    dcci(reinterpret_cast<__gm__ void *>(p), SINGLE_CACHE_LINE);
    dsb(DSB_DDR);
    set_st_atomic_cfg(ATOMIC_NONE, ATOMIC_SUM);
#else
    volatile int32_t *vp = reinterpret_cast<volatile int32_t *>(p);
    *vp = *vp + v;
#endif
}

// Single shared counter barrier: atomic-add arrival, poll ONE counter with
// ld_dev until it reaches epoch*n -> O(1) read per barrier.
static __aicore__ inline void barrier_counter(__gm__ int32_t *counter, int32_t n, int32_t epoch) {
    ddr_fence();
    atomic_add_i32(counter, 1);
    while (ld_i32_dev(counter) < epoch * n) {}
    ddr_fence();
}

// Select the barrier at compile time: 0 = per-slot st_dev, 1 = atomic counter.
#ifndef MERGE_BARRIER_COUNTER
#define MERGE_BARRIER_COUNTER 1
#endif
static __aicore__ inline void barrier(__gm__ int32_t *sync, int32_t n, int32_t i, int32_t epoch) {
#if MERGE_BARRIER_COUNTER
    (void)i;
    barrier_counter(sync, n, epoch);
#else
    barrier_slots(sync, n, i, epoch);
#endif
}

constexpr int32_t kR = 1;
constexpr int32_t kC = 128;
using GData = GlobalTensor<float, Shape<1, 1, 1, kR, kC>, Stride<1, 1, 1, kC, 1>>;
using VTile = Tile<TileType::Vec, float, kR, kC, BLayout::RowMajor, -1, -1>;

static __aicore__ inline void wait_store_done() {
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
}

// tile compute: dst[t] = op0 ? (src[t] + b) : (src[t] + src[t])
static __aicore__ inline void
stage_tile(__gm__ float *src, __gm__ float *dst, int32_t t, int32_t op, float b, VTile &t0, VTile &t1) {
    GData ig(src + t * kC);
    GData og(dst + t * kC);
    TLOAD(t0, ig);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    if (op == 0) {
        TADDS(t1, t0, b);
    } else {
        TADD(t1, t0, t0);
    }
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(og, t1);
    // Drain this store before the caller reuses t0/t1 for the next tile.
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID6);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID6);
}

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *x_t = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *sync_t = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ Tensor *s1_t = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ Tensor *s2_t = reinterpret_cast<__gm__ Tensor *>(args[3]);
    __gm__ Tensor *out_t = reinterpret_cast<__gm__ Tensor *>(args[4]);
    __gm__ Tensor *tm_t = reinterpret_cast<__gm__ Tensor *>(args[5]);

    __gm__ float *x = reinterpret_cast<__gm__ float *>(x_t->buffer.addr) + x_t->start_offset;
    __gm__ int32_t *sync = reinterpret_cast<__gm__ int32_t *>(sync_t->buffer.addr) + sync_t->start_offset;
    __gm__ float *s1 = reinterpret_cast<__gm__ float *>(s1_t->buffer.addr) + s1_t->start_offset;
    __gm__ float *s2 = reinterpret_cast<__gm__ float *>(s2_t->buffer.addr) + s2_t->start_offset;
    __gm__ float *out = reinterpret_cast<__gm__ float *>(out_t->buffer.addr) + out_t->start_offset;
    __gm__ int32_t *tm = reinterpret_cast<__gm__ int32_t *>(tm_t->buffer.addr) + tm_t->start_offset;

    int32_t N = get_block_num(args);  // 8
    int32_t i = get_block_idx(args);

    VTile t0(kR, kC), t1(kR, kC);
    TASSIGN(t0, 0x0);
    TASSIGN(t1, 0x10000);

    // get_sys_cnt() = 50 MHz counter (20 ns/tick). Deltas below are in ticks.
    int64_t c0 = get_sys_cnt();

    // Segment A (stage 1): block 0 does all 8 tiles: s1 = x + 1
    if (i == 0) {
        for (int32_t t = 0; t < 8; ++t) {
            stage_tile(x, s1, t, 0, 1.0f, t0, t1);
        }
        wait_store_done();
    }
    int64_t c1 = get_sys_cnt();  // arrived at barrier 1
    barrier(sync, N, i, 1);
    int64_t c2 = get_sys_cnt();  // released from barrier 1

    // Segment B (stage 2): blocks 0..3 do 2 tiles each: s2 = s1 * 2
    if (i < 4) {
        for (int32_t t = i * 2; t < i * 2 + 2; ++t) {
            stage_tile(s1, s2, t, 1, 0.0f, t0, t1);
        }
        wait_store_done();
    }
    int64_t c3 = get_sys_cnt();  // arrived at barrier 2
    barrier(sync, N, i, 2);
    int64_t c4 = get_sys_cnt();  // released from barrier 2

    // Segment C (stage 3): all 8 blocks do 1 tile each: out = s2 + 1
    stage_tile(s2, out, i, 0, 1.0f, t0, t1);
    int64_t c5 = get_sys_cnt();  // end

    // Per-block timing (ticks): [segA, barrier1_gap, segB, barrier2_gap, segC, total].
    // Stride each block by 32 int32 (one 128B cacheline): scalar st_dev only
    // commits the first 8 int32 of a cacheline, so keep all 6 writes in that head.
    st_i32_dev(&tm[i * 32 + 0], static_cast<int32_t>(c1 - c0));
    st_i32_dev(&tm[i * 32 + 1], static_cast<int32_t>(c2 - c1));
    st_i32_dev(&tm[i * 32 + 2], static_cast<int32_t>(c3 - c2));
    st_i32_dev(&tm[i * 32 + 3], static_cast<int32_t>(c4 - c3));
    st_i32_dev(&tm[i * 32 + 4], static_cast<int32_t>(c5 - c4));
    st_i32_dev(&tm[i * 32 + 5], static_cast<int32_t>(c5 - c0));
    ddr_fence();  // flush timing writes to HBM before the task completes / copy-back

    pipe_sync();
}
