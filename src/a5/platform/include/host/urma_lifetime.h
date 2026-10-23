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

#include <cstddef>
#include <cstdint>
#include <vector>
#include "common/urma_workspace_abi.h"

namespace urma_lifetime {

// A successful device stop and equal submitted/completed counters are both
// required. A barrier or a timeout alone is not evidence of either condition.
template <typename Read>
bool queues_drained(uint64_t workspace, Read read, uint32_t expected_ranks) {
    using namespace pto2::urma_backend;
    UrmaInfo info{};
    if (!workspace || !read(&info, workspace, sizeof(info)) || info.layout != 0 || info.qp_num != 1 ||
        info.jetties_per_core != 1 || info.rank_count < 2 || info.rank_count != expected_ranks ||
        info.jetty_count != info.rank_count || !info.sq_ptr || !info.scq_ptr)
        return false;
    std::vector<UrmaWqCtx> sq(info.rank_count);
    std::vector<UrmaCqCtx> cq(info.rank_count);
    if (!read(sq.data(), info.sq_ptr, sq.size() * sizeof(UrmaWqCtx)) ||
        !read(cq.data(), info.scq_ptr, cq.size() * sizeof(UrmaCqCtx)))
        return false;
    uint32_t empty_rows = 0;
    for (uint32_t peer = 0; peer < info.rank_count; ++peer) {
        // PTO leaves the self row empty.
        if (!sq[peer].head_addr && !sq[peer].tail_addr && !cq[peer].tail_addr) {
            ++empty_rows;
            continue;
        }
        uint32_t head = 0, tail = 0, completed = 0;
        if (!sq[peer].head_addr || !sq[peer].tail_addr || !cq[peer].tail_addr ||
            !read(&head, sq[peer].head_addr, sizeof(head)) || !read(&tail, sq[peer].tail_addr, sizeof(tail)) ||
            !read(&completed, cq[peer].tail_addr, sizeof(completed)) || head != tail ||
            completed != sq[peer].submitted_wqe_count)
            return false;
    }
    return empty_rows == 1;
}

template <typename Stop, typename Drain, typename Destroy, typename Reclaim>
bool shutdown(Stop stop, Drain drain, Destroy destroy, Reclaim reclaim) {
    if (!stop() || !drain() || !destroy()) return false;
    reclaim();
    return true;
}

}  // namespace urma_lifetime
