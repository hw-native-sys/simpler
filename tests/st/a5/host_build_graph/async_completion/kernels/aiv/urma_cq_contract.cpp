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

#include "../urma.h"
#include "backend/urma/urma_completion_scheduler.h"
using namespace async_case;
using namespace pto::comm::urma;
extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    auto *meta = data<uint64_t>(args[3]);
    auto *info = reinterpret_cast<__gm__ UrmaInfo *>(meta + 512);
    auto *sq = reinterpret_cast<__gm__ UrmaWQCtx *>(meta + 1024);
    auto *cq = reinterpret_cast<__gm__ UrmaCqCtx *>(meta + 1536);
    auto *words = reinterpret_cast<__gm__ uint32_t *>(meta + 2048);
    auto *counts = reinterpret_cast<__gm__ uint32_t *>(meta + 4096);
    info->layout = UrmaLayout::PER_PEER;
    info->rankCount = 2;
    info->rowsPerPeer = 1;
    info->jettyCount = 2;
    info->jettiesPerCore = 1;
    info->sqPtr = reinterpret_cast<uint64_t>(sq);
    info->scqPtr = reinterpret_cast<uint64_t>(cq);
    sq[1].depth = 4;
    sq[1].headAddr = reinterpret_cast<uint64_t>(counts);
    sq[1].tailAddr = reinterpret_cast<uint64_t>(counts + 1);
    sq[1].dbAddr = reinterpret_cast<uint64_t>(counts + 2);
    cq[1].depth = 4;
    cq[1].cqeShiftSize = 6;
    cq[1].bufAddr = reinterpret_cast<uint64_t>(words);
    cq[1].tailAddr = reinterpret_cast<uint64_t>(counts + 3);
    cq[1].dbAddr = reinterpret_cast<uint64_t>(counts + 4);
    for (int i = 0; i < 5; ++i)
        counts[i] = 0;
    for (int i = 0; i < 4; ++i)
        words[i * 16] = 0;
    DcciCachelines(reinterpret_cast<__gm__ uint8_t *>(meta + 512), (4096 - 512 + 16) * 8);
    dsb(DSB_DDR);
    CompletionToken token{
        uint64_t{1} << 32 | 2, 2, COMPLETION_ENGINE_URMA, COMPLETION_TYPE_URMA_EVENT_HANDLE,
        reinterpret_cast<uint64_t>(info)
    };
    auto check = [&](CompletionPollState expected) __aicore__ {
        const auto result = pto2::urma_backend::poll_owned_queue(token, 0, pto2::urma_backend::AicoreCqPoll{});
        if (result.state != expected) add(meta + 32, 1);
        else add(meta, 1);
    };
    check(CompletionPollState::PENDING);
    words[0] = 4;
    words[16] = 4;
    flush(words);
    flush(words + 16);
    check(CompletionPollState::READY);
    if (ld_dev(counts + 1, 0) != 2 || ld_dev(counts + 3, 0) != 2 || ld_dev(counts + 4, 0) != 2) add(meta + 32, 1);
    token.addr = (uint64_t{1} << 32) | 1;
    token.expected_value = 1;
    check(CompletionPollState::READY);
    if (ld_dev(counts + 1, 0) != 2 || ld_dev(counts + 3, 0) != 2) add(meta + 32, 1);
    token.addr = (uint64_t{1} << 32) | 3;
    token.expected_value = 3;
    words[32] = 4 | (1u << 16);
    flush(words + 32);
    check(CompletionPollState::FAILED);
    if (ld_dev(counts + 1, 0) != 2 || ld_dev(counts + 3, 0) != 2) add(meta + 32, 1);
    st_dev(UINT32_MAX, counts + 1, 0);
    st_dev(UINT32_MAX, counts + 3, 0);
    words[48] = 0;
    words[0] = 4;
    flush(words + 48);
    flush(words);
    token.addr = (uint64_t{1} << 32) | 1;
    token.expected_value = 1;
    check(CompletionPollState::READY);
    if (ld_dev(counts + 1, 0) != 1 || ld_dev(counts + 3, 0) != 1) add(meta + 32, 1);
}
