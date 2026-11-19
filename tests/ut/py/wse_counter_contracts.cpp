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
#include "counter_args.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

using S = wse::CounterStatus;

void bounds() {
    alignas(64) uint64_t words[] = {17, 9, 123};
    wse::CounterView v{words, 3};
    auto r = wse::counter_signal(v, 1, 3);
    assert(r.status == S::OK && r.old_value == 9 && r.new_value == 12);
    assert(words[0] == 17 && words[2] == 123);
    for (auto target : {0u, 11u, 12u})
        assert(wse::counter_poll(v, 1, target).status == S::READY);
    auto p = wse::counter_poll(v, 1, 13, 7);
    assert(p.status == S::NOT_READY && p.observed == 12 && words[1] == 12);
    assert(wse::counter_poll(v, 1, 0, 0).status == S::INVALID_ARGUMENT);
    assert(wse::counter_signal(v, 1, 0).status == S::INVALID_ARGUMENT);
    for (auto bad :
         {wse::CounterView{}, wse::CounterView{words, 0},
          wse::CounterView{reinterpret_cast<uint64_t *>(reinterpret_cast<char *>(words) + 1), 1},
          wse::CounterView{words, SIZE_MAX}}) {
        assert(wse::counter_signal(bad, 0).status == S::INVALID_ARGUMENT);
        assert(wse::counter_poll(bad, 0, 0).status == S::INVALID_ARGUMENT);
    }
    assert(wse::counter_signal(v, 3).status == S::INVALID_ARGUMENT);
    assert(wse::counter_poll(v, SIZE_MAX, 0).status == S::INVALID_ARGUMENT);
    assert(words[1] == 12);
}

void contention(bool overflow) {
    alignas(64) uint64_t word = overflow ? UINT64_MAX - 4 : 23;
    constexpr size_t workers = 8, count = 1000;
    std::array<std::vector<uint64_t>, workers> results;
    std::vector<std::thread> threads;
    for (size_t t = 0; t < workers; ++t)
        threads.emplace_back([&, t] {
            for (size_t i = 0; i < (overflow ? 1 : count); ++i) {
                auto r = wse::counter_signal({&word, 1}, 0);
                if (r.status == S::OK) {
                    assert(r.new_value == r.old_value + 1);
                    results[t].push_back(r.new_value);
                } else
                    assert(
                        overflow && r.status == S::OVERFLOW && r.old_value == UINT64_MAX && r.new_value == r.old_value
                    );
            }
        });
    for (auto &t : threads)
        t.join();
    std::vector<uint64_t> values;
    for (auto &v : results)
        values.insert(values.end(), v.begin(), v.end());
    std::sort(values.begin(), values.end());
    assert(values.size() == (overflow ? 4 : workers * count));
    const uint64_t first = overflow ? UINT64_MAX - 3 : 24;
    for (size_t i = 0; i < values.size(); ++i)
        assert(values[i] == first + i);
    assert(word == (overflow ? UINT64_MAX : 23 + workers * count));
    if (overflow) assert(wse::counter_signal({&word, 1}, 0, 2).status == S::OVERFLOW && word == UINT64_MAX);
}

void publication() {
    constexpr size_t producers = 3, consumers = 2, rounds = 20;
    alignas(64) uint64_t words[3] = {7, 11, 0};
    wse::CounterView v{words, 3};
    std::array<uint64_t, producers> payload{};
    std::vector<std::thread> threads;
    for (size_t p = 0; p < producers; ++p)
        threads.emplace_back([&, p] {
            while (wse::counter_poll(v, 2, consumers).status != S::READY) {}
            for (size_t g = 1; g <= rounds; ++g) {
                while (wse::counter_poll(v, 1, 11 + (g - 1) * consumers).status != S::READY) {}
                payload[p] = g * 100 + p;
                assert(wse::counter_signal(v, 0).status == S::OK);
            }
        });
    for (size_t c = 0; c < consumers; ++c)
        threads.emplace_back([&] {
            assert(wse::counter_poll(v, 0, 7 + producers).status == S::NOT_READY);
            assert(wse::counter_signal(v, 2).status == S::OK);
            for (size_t g = 1; g <= rounds; ++g) {
                while (wse::counter_poll(v, 0, 7 + g * producers).status != S::READY) {}
                for (size_t p = 0; p < producers; ++p)
                    assert(payload[p] == g * 100 + p);
                assert(wse::counter_signal(v, 1).status == S::OK);
            }
        });
    for (auto &t : threads)
        t.join();
    assert(words[0] == 7 + rounds * producers && words[1] == 11 + rounds * consumers);
    assert(wse::counter_poll(v, 0, 7 + (rounds + 1) * producers).status == S::NOT_READY);
    words[0] = 3;
    assert(wse::counter_signal(v, 0).new_value == 4);
}

void tensor_args() {
    alignas(64) uint64_t words[8]{};
    Tensor t{};
    t.buffer = {reinterpret_cast<uintptr_t>(words), sizeof(words)};
    t.dtype = DataType::UINT64;
    t.ndims = 1;
    t.shapes[0] = 3;
    t.strides[0] = 1;
    t.start_offset = 2;
    wse::CounterView v;
    assert(wse::make_counter_view(t, v) == S::OK && v.data == words + 2 && v.size == 3);
    assert(wse::counter_signal(v, 2, 9).new_value == 9 && words[4] == 9);
    auto reject = [&](const Tensor &bad) {
        v = {words, 8};
        assert(wse::make_counter_view(bad, v) == S::INVALID_ARGUMENT && !v.data && !v.size);
    };
    auto bad = t;
    bad.dtype = DataType::FLOAT32;
    reject(bad);
    bad = t;
    bad.ndims = 2;
    reject(bad);
    bad = t;
    bad.ndims = 0;
    reject(bad);
    bad = t;
    bad.shapes[0] = 0;
    reject(bad);
    bad = t;
    bad.strides[0] = 2;
    reject(bad);
    bad = t;
    bad.strides[0] = 0;
    reject(bad);
    bad = t;
    bad.start_offset = UINT64_MAX;
    reject(bad);
    bad = t;
    bad.start_offset = 6;
    reject(bad);
    bad = t;
    bad.buffer.addr += 1;
    reject(bad);
    bad = t;
    bad.buffer.addr = 0;
    reject(bad);
    bad = t;
    bad.buffer.size = UINT64_MAX;
    reject(bad);
    bad = t;
    bad.buffer.size = 39;
    reject(bad);
    assert(words[4] == 9);
}

int main(int argc, char **argv) {
    assert(argc == 2);
    std::string test = argv[1];
    if (test == "bounds") bounds();
    else if (test == "contention") contention(false);
    else if (test == "overflow") contention(true);
    else if (test == "publication") publication();
    else if (test == "tensor") tensor_args();
    else return 2;
}
