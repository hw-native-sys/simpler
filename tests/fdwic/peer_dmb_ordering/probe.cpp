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
#include "cce_aicore_intrinsics.h"
#include <pto/common/kernel_meta.hpp>
#include "probe.h"

using namespace peer_dmb;
namespace {
using Word = __gm__ uint32_t;
__aicore__ inline uint32_t read_word(Word *p) { return __builtin_cce_ld_dev(p, 0); }
__aicore__ inline void write_word(Word *p, uint32_t v) { __builtin_cce_st_dev(v, p, 0); }
__aicore__ inline uint32_t local_dmb() {
    uint64_t v;
    asm volatile("MOV %0, DATA_MAIN_BASE" : "=l"(v) : : "memory");
    return static_cast<uint32_t>(v);
}
__aicore__ inline bool expired(uint64_t begin) {
    return get_sys_cnt() - begin > 10000000;  // 200 ms at 50 MHz
}
__aicore__ inline bool await_word(Word *p, uint32_t value) {
    const uint64_t start = get_sys_cnt();
    while (read_word(p) != value) {
        if (expired(start)) return false;
    }
    return true;
}
__aicore__ inline void receive(__gm__ Config *cfg, __gm__ State *state, uint32_t core) {
    auto &r = state->receiver;
    auto *cnt = reinterpret_cast<Word *>(cfg->set);
    const bool peer = cfg->mode == PEER_DMB;
    const uint32_t initial = local_dmb();
    const uint32_t token = initial == 0x1357U ? 0x2468U : 0x1357U;
    write_word(&r.core, core);
    write_word(&r.initial, initial);
    write_word(&r.token, token);
    write_word(cnt, 1);  // One predecessor has already completed.
    dsb(DSB_DDR);
    if (!await_word(cnt, 1)) {
        write_word(&r.status, 11);
        return;
    }
    write_word(&r.armed, 1);
    const uint64_t begin = get_sys_cnt();
    uint32_t observed;
    do {
        observed = peer ? local_dmb() : read_word(&state->sender.doorbell);
        if (observed == token) break;
        if (expired(begin)) {
            write_word(&r.observed, observed);
            write_word(&r.eventual, read_word(cnt));
            write_word(&r.status, 12);
            return;
        }
    } while (true);
    // The very first device load after observing this token is the measurement.
    asm volatile("" ::: "memory");
    const uint32_t first = read_word(cnt);
    write_word(&r.first, first);
    write_word(&r.observed, observed);
    dsb(DSB_DDR);
    write_word(&r.sampled, 1);
    const bool settled = await_word(cnt, 2);
    write_word(&r.eventual, read_word(cnt));
    write_word(&r.status, settled ? 1 : 13);
}
__aicore__ inline void send(__gm__ Config *cfg, __gm__ State *state, uint32_t core) {
    auto &s = state->sender;
    auto &r = state->receiver;
    write_word(&s.core, core);
    if (!await_word(&r.armed, 1)) {
        write_word(&s.stage, 21);
        return;
    }
    dsb(DSB_DDR);
    const uint32_t target = read_word(&r.core);
    if (target >= kSlots || target == core || cfg->dmb[target] == 0) {
        write_word(&s.stage, 22);
        return;
    }
    Word *wakeup = cfg->mode == PEER_DMB ? reinterpret_cast<Word *>(cfg->dmb[target]) : &s.doorbell;
    const uint32_t token = read_word(&r.token);
    auto *inc = reinterpret_cast<Word *>(cfg->inc);
    if (cfg->mode == GM_EARLY) {
        write_word(wakeup, token);
        if (!await_word(&r.sampled, 1)) {
            write_word(&s.stage, 23);
            return;
        }
        write_word(inc, 1);
    } else {
        const uint32_t fence = cfg->fence;
        write_word(&s.stage, 1);
        dsb(DSB_DDR);
        write_word(inc, 1);
        asm volatile("" ::: "memory");
        if (fence == DDR) dsb(DSB_DDR);
        if (fence == ALL) dsb(DSB_ALL);
        // Only one 32-bit store to the enrolled receiver's low DMB word.
        write_word(wakeup, token);
    }
    dsb(DSB_DDR);
    write_word(&s.stage, 2);
}
}  // namespace
PTO_SYNCALL_AIC_KERNEL_META(peer_dmb_ordering);
extern "C" __global__ __aicore__ void peer_dmb_ordering(__gm__ Config *cfg, __gm__ State *state) {
    const uint32_t core = static_cast<uint32_t>(get_coreid()) & 0xfffU;
    if (get_block_idx() == 0) send(cfg, state, core);
    else receive(cfg, state, core);
    dsb(DSB_DDR);
}
