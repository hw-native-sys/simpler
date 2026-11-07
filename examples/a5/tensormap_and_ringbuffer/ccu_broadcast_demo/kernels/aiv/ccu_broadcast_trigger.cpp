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
#include <pto/pto-inst.hpp>

#include "intrinsic.h"
#include "pto/comm/comm_types.hpp"

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    if (get_block_idx(args) != 0) return;
    const uint64_t gate = static_cast<uint64_t>(args[0]);
    const uint32_t mask = static_cast<uint32_t>(args[1]);
    auto *status = reinterpret_cast<volatile __gm__ uint32_t *>(args[2]);
    const uint32_t root = static_cast<uint32_t>(args[3]);
    const uint32_t rank = static_cast<uint32_t>(args[4]);
    if (gate == 0 || mask == 0 || status == nullptr || root >= 2 || rank >= 2) return;

    using Shape = pto::Shape<1, 1, 1, 1, 1024>;
    using Stride = pto::Stride<1024, 1024, 1024, 1024, 1>;
    using Global = pto::GlobalTensor<float, Shape, Stride, pto::Layout::ND>;
    using Tile = pto::Tile<pto::TileType::Vec, float, 1, 1024, pto::BLayout::RowMajor, -1, -1>;
    Global rank_tensors[2];
    Global source;
    Tile scratch;
    pto::comm::ParallelGroup<Global> group(rank_tensors, 2, root);
    // HostManaged inputs and all data addresses belong to the Host-prepared CCU task.
    const pto::comm::CcuTriggerContext trigger{gate, mask, rank, pto::comm::CcuInputSource::HostManaged};
    pto::comm::TBROADCAST<pto::comm::CollEngine::CCU>(group, source, scratch, trigger);
    *status = 0x43435531;
    dcci(reinterpret_cast<__gm__ void *>(const_cast<__gm__ uint32_t *>(status)), cache_line_t::ENTIRE_DATA_CACHE);
    dsb(DSB_DDR);
    pipe_barrier(PIPE_ALL);
}
