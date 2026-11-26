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

#include <array>
#include <cstdint>
#include <pto/comm/async/urma/urma_types.hpp>

namespace simpler::test {

using pto::comm::urma::UrmaCqCtx;
using pto::comm::urma::UrmaInfo;
using pto::comm::urma::UrmaWQCtx;

constexpr uint32_t kRows = 8;
constexpr uint32_t kDepth = 8;
constexpr uint32_t kCqeWords = 16;

struct alignas(64) TestCqe {
    std::array<uint32_t, kCqeWords> words{};
};

struct UrmaQueueFixture {
    UrmaInfo info{};
    std::array<UrmaWQCtx, kRows> sq{};
    std::array<UrmaCqCtx, kRows> cq{};
    std::array<uint32_t, kRows> sq_head{};
    std::array<uint32_t, kRows> sq_tail{};
    std::array<uint32_t, kRows> sq_doorbell{};
    std::array<uint32_t, kRows> cq_tail{};
    std::array<uint32_t, kRows> cq_doorbell{};
    std::array<std::array<TestCqe, kDepth>, kRows> entries{};

    UrmaQueueFixture() {
        info.rowsPerPeer = 4;
        info.rankCount = 2;
        info.sqPtr = reinterpret_cast<uint64_t>(sq.data());
        info.scqPtr = reinterpret_cast<uint64_t>(cq.data());
        info.layout = static_cast<decltype(info.layout)>(0);
        info.jettyCount = kRows;
        info.jettiesPerCore = 2;
        for (uint32_t row = 0; row < kRows; ++row) {
            sq[row].bufAddr = reinterpret_cast<uint64_t>(entries[row].data());
            sq[row].wqeShiftSize = 6;
            sq[row].depth = kDepth;
            sq[row].headAddr = reinterpret_cast<uint64_t>(&sq_head[row]);
            sq[row].tailAddr = reinterpret_cast<uint64_t>(&sq_tail[row]);
            sq[row].dbAddr = reinterpret_cast<uint64_t>(&sq_doorbell[row]);
            cq[row].bufAddr = reinterpret_cast<uint64_t>(entries[row].data());
            cq[row].cqeShiftSize = 6;
            cq[row].dbMode = pto::comm::urma::UrmaDbMode::SW_DB;
            cq[row].depth = kDepth;
            cq[row].tailAddr = reinterpret_cast<uint64_t>(&cq_tail[row]);
            cq[row].dbAddr = reinterpret_cast<uint64_t>(&cq_doorbell[row]);
        }
    }

    uint64_t addr() { return reinterpret_cast<uint64_t>(&info); }

    void set_cqe(uint32_t row, uint32_t sequence, bool ready, uint32_t status = 0, uint32_t substatus = 0) {
        const uint32_t lap_owner = (sequence / kDepth) & 1u;
        const uint32_t owner = ready ? lap_owner ^ 1u : lap_owner;
        entries[row][sequence & (kDepth - 1)].words[0] = (owner << 2) | (substatus << 16) | (status << 24);
    }
};

}  // namespace simpler::test
