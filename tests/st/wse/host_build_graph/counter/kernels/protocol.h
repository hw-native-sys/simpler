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
#include "counter_args.h"
#include "intrinsic.h"
#include <chrono>
#include <dlfcn.h>

inline bool await_counter(wse::CounterView view, size_t index, uint64_t target) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(15);
    do {
        auto r = wse::counter_poll(view, index, target);
        if (r.status == wse::CounterStatus::READY) return true;
        if (r.status != wse::CounterStatus::NOT_READY) return false;
    } while (std::chrono::steady_clock::now() < deadline);
    return false;
}

inline void counter_protocol(int64_t *args, int role) {
    const bool producer = role == 0 || (role < 0 && get_block_idx(args) == 0);
    const int pe = get_pe_id();
    const int producers = args[3], consumers = args[4], rounds = args[5];
    const uint64_t base = args[6];
    // Producers and consumers each own one result row per physical PE.
    const size_t result_row = producer ? pe : wse::PE_PER_CORE + pe;
    auto &out = *reinterpret_cast<Tensor *>(args[2]);
    auto *record = reinterpret_cast<uint64_t *>(out.buffer.addr) + out.start_offset + result_row * rounds * 6;
    if (uint64_t(rounds) > (UINT64_MAX - base) / producers) {
        record[0] = UINT64_MAX - 3;
        return;
    }
    wse::CounterView counters;
    if (wse::make_counter_view(*reinterpret_cast<Tensor *>(args[0]), counters) != wse::CounterStatus::OK) {
        record[0] = UINT64_MAX - 1;
        return;
    }
    auto &data = *reinterpret_cast<Tensor *>(args[1]);
    auto *payload = reinterpret_cast<uint64_t *>(data.buffer.addr) + data.start_offset;
    auto hook = args[7] ? reinterpret_cast<void (*)(int)>(dlsym(RTLD_DEFAULT, "wse_counter_probe")) : nullptr;
    if (args[7] && !hook) {
        record[0] = UINT64_MAX - 2;
        return;
    }
    if (!producer) {
        record[2] = wse::counter_poll(counters, 0, base + producers).status == wse::CounterStatus::NOT_READY;
        if (hook) hook(1 + pe);
        if (wse::counter_signal(counters, 2).status != wse::CounterStatus::OK) return;
    } else {
        if (hook) hook(0);
        if (!await_counter(counters, 2, consumers)) return;
    }
    for (int g = 1; g <= rounds; ++g) {
        auto *row = record + (g - 1) * 6;
        row[0] = get_core_id();
        row[1] = pe;
        if (producer) {
            if (!await_counter(counters, 1, (g - 1) * consumers)) return;
            payload[pe] = base + g * 100 + pe;
            auto signal = wse::counter_signal(counters, 0);
            if (signal.status != wse::CounterStatus::OK) return;
            row[3] = signal.old_value;
            row[4] = signal.new_value;
            row[5] = payload[pe];
            if (!await_counter(counters, 1, g * consumers)) return;
        } else {
            if (!await_counter(counters, 0, base + g * producers)) return;
            uint64_t sum = 0;
            for (uint32_t p = 0; p < wse::PE_PER_CORE; ++p)
                sum += payload[p];
            row[5] = sum;
            auto signal = wse::counter_signal(counters, 1);
            if (signal.status != wse::CounterStatus::OK) return;
            row[3] = signal.old_value;
            row[4] = signal.new_value;
        }
    }
}
