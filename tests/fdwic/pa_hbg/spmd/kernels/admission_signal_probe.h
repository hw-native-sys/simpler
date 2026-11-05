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

#pragma once

#include "execution_record.h"
#include "inner_kernel.h"

static inline __aicore__ uint64_t admission_signal_pattern(uint64_t epoch, uint32_t index, uint32_t word) {
    return epoch ^ (static_cast<uint64_t>(index) << 32) ^ (UINT64_C(0x1234000010000100) + word);
}

static inline __aicore__ int64_t admission_signal_log_value(uint32_t index) {
    return (index & 1) == 0 ? -static_cast<int64_t>(index) - 7 : static_cast<int64_t>(index) + 7;
}

static inline __aicore__ bool admission_signal_wait(__gm__ volatile uint64_t &notice, uint64_t expected) {
    const uint64_t started = get_sys_cnt_aicore();
    while (true) {
        const uint64_t observed = scheduler_gm_fetch_add(notice, UINT64_C(0));
        if (observed >= expected) return observed == expected;
        if (get_sys_cnt_aicore() - started > PLATFORM_PROF_SYS_CNT_FREQ * 5) return false;
    }
}

static inline __aicore__ bool admission_signal_record_visible(
    __gm__ int64_t *log, __gm__ uint64_t *payloads, __gm__ uint64_t *states, uint64_t epoch, uint32_t index
) {
    // Published phase and payload observations do not use an RMW on either signal line.
    bool valid = scheduler_gm_query(log[index]) == admission_signal_log_value(index);
    auto *payload = payloads + index * 16;
    for (uint32_t offset = 0; offset < 128; offset += 64)
        scheduler_invalidate_cache_line(reinterpret_cast<__gm__ uint8_t *>(payload) + offset);
    scheduler_cache_barrier();
    for (uint32_t word = 0; word < 16; ++word)
        if (payload[word] != admission_signal_pattern(epoch, index, word)) valid = false;
    if (scheduler_gm_query(states[index * 16]) != 3) valid = false;
    return valid;
}

static inline __aicore__ void admission_signal_probe(__gm__ int64_t *args, uint32_t lane) {
    static_assert(sizeof(SchedulerAdmissionControl) == 256, "probe admission control layout mismatch");
    const uint32_t block = static_cast<uint32_t>(get_block_idx(args));
    const uint32_t blocks = static_cast<uint32_t>(args[2]);
    const uint32_t rounds = static_cast<uint32_t>(args[3]);
    const bool shared = args[4] != 0;
    const uint64_t epoch = static_cast<uint64_t>(args[5]);
    const int64_t delayed_rank = args[6];
    if (block >= blocks) return;
    const uint32_t participants = shared ? blocks * 3 : 3;
    const uint32_t rank = shared ? block * 3 + lane : lane;
    const uint32_t arena = shared ? 0 : block;
    const uint32_t entries = participants * rounds;
    const uint32_t log_stride = (entries + 15) / 16 * 16;
    const uint32_t arena_words = 48 + log_stride + entries * 32 + 16;
    auto *base = reinterpret_cast<__gm__ uint64_t *>(tensor_data(args[0])) + arena * arena_words;
    auto *control = reinterpret_cast<__gm__ SchedulerAdmissionControl *>(base);
    auto &notice = base[32];
    auto *log = reinterpret_cast<__gm__ int64_t *>(base + 48);
    auto *payloads = base + 48 + log_stride;
    auto *states = payloads + entries * 16;
    auto *check = reinterpret_cast<__gm__ ExecutionRecord *>(tensor_data(args[1])) + block * 3 + lane;
    scheduler_gm_fetch_add(check->visits, UINT64_C(1));
    uint64_t errors = 0;
    uint32_t completed = 0;
    for (uint32_t round = 0; round < rounds; ++round) {
        const uint32_t index = round * participants + rank;
        if (!admission_signal_wait(notice, index)) {
            errors |= 1;
            break;
        }
        bool locked = false;
        const uint64_t started = get_sys_cnt_aicore();
        while (get_sys_cnt_aicore() - started <= PLATFORM_PROF_SYS_CNT_FREQ * 5) {
            if (scheduler_gm_compare_exchange(control->commit_lock, UINT64_C(0), static_cast<uint64_t>(rank + 1)) ==
                0) {
                locked = true;
                break;
            }
        }
        if (!locked) {
            errors |= 2;
            break;
        }
        // The preceding publisher returned before its atomic notice and lock
        // release. This first bypass observation must see its entire prefix.
        if (scheduler_gm_query(control->commit_tail) != index) errors |= 4;
        if (index != 0 && !admission_signal_record_visible(log, payloads, states, epoch, index - 1)) errors |= 8;
        if (static_cast<int64_t>(rank) == delayed_rank) {
            const uint64_t delay_start = get_sys_cnt_aicore();
            while (get_sys_cnt_aicore() - delay_start < 2048) {}
        }
        scheduler_publish_signal(states[index * 16], UINT64_C(1));
        if (scheduler_gm_query(states[index * 16]) != 1) errors |= 256;
        auto *payload = payloads + index * 16;
        for (uint32_t word = 0; word < 16; ++word)
            payload[word] = admission_signal_pattern(epoch, index, word);
        for (uint32_t offset = 0; offset < 128; offset += 64)
            scheduler_writeback_cache_line(reinterpret_cast<__gm__ uint8_t *>(payload) + offset);
        scheduler_cache_barrier();
        scheduler_publish_signal(states[index * 16], UINT64_C(2));
        if (scheduler_gm_query(states[index * 16]) != 2) errors |= 512;
        scheduler_publish_signal(log[index], admission_signal_log_value(index));
        scheduler_publish_signal(states[index * 16], UINT64_C(3));
        scheduler_cache_barrier();
        scheduler_publish_signal(control->commit_tail, static_cast<uint64_t>(index + 1));
        scheduler_cache_barrier();
        scheduler_gm_store(notice, static_cast<uint64_t>(index + 1));
        scheduler_gm_store(control->commit_lock, UINT64_C(0));
        ++completed;
    }
    if (completed == rounds) {
        if (!admission_signal_wait(notice, entries)) {
            errors |= 16;
        } else {
            if (scheduler_gm_query(control->commit_tail) != entries) errors |= 32;
            for (uint32_t round = 0; round < rounds; ++round)
                if (!admission_signal_record_visible(log, payloads, states, epoch, round * participants + rank))
                    errors |= 64;
            if (rank == 0 && !admission_signal_record_visible(log, payloads, states, epoch, entries - 1)) errors |= 128;
        }
    }
    check->block_plus_one = static_cast<int64_t>(errors);
    check->block_num = completed;
    check->slot_plus_one = rank + 1;
    check->task_plus_one = arena + 1;
    check->metadata_padding[0] = participants;
    scheduler_publish_cache_line(&check->block_plus_one);
}
