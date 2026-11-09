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
#include "../device.h"

inline __aicore__ void copy_and_add(__gm__ float *source, __gm__ float *output, uint32_t count) {
    using Global = pto::GlobalTensor<float, pto::Shape<1, 1, 1, 1, 256>, pto::Stride<256, 256, 256, 256, 1>>;
    pto::Tile<pto::TileType::Vec, float, 1, 256> input, result;
    pto::TASSIGN(input, 0);
    pto::TASSIGN(result, 0x10000);
    for (uint32_t offset = 0; offset < count; offset += 256) {
        Global src(source + offset), dst(output + offset);
        pto::TLOAD(input, src);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        pto::TADDS(result, input, 7.0F);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        pto::TSTORE(dst, result);
        set_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
        wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
    }
}
extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    using namespace mixed_async;
    auto *comm = reinterpret_cast<__gm__ CommContext *>(args[9]);
    const uint32_t elements = args[10], mask = args[11];
    const uint32_t e = get_block_idx(args), lane = e % 3;
    if (!active(e, mask)) return;
    auto *meta = data<uint64_t>(args, 8) + e * kMeta;
    meta[1] = 1;
    if (meta[0] != 1) meta[6] |= 1;
    uint32_t record = 0;
    for (uint32_t peer = 0; peer < comm->rankNum; ++peer) {
        if (peer == comm->rankId) continue;
        const uint32_t at = (e * comm->rankNum + peer) * kLine;
        if (data<int32_t>(args, 1)[at] != 1) meta[6] |= 2;
        if (lane != 0) {
            const uint64_t address = meta[8 + record * 2], expected = meta[9 + record * 2];
            if (address == 0 || expected == 0 || read_done(address) < expected) meta[6] |= 4;
            ++record;
        }
    }
    if (meta[6] != 0) {
        publish(meta, kMeta * sizeof(uint64_t));
        return;
    }
    for (uint32_t peer = 0; peer < comm->rankNum; ++peer) {
        if (peer == comm->rankId) continue;
        const uint32_t at = (e * comm->rankNum + peer) * kLine;
        // No test-local payload invalidate: visibility belongs to the runtime's consumer acquire.
        for (uint32_t i = 0; i < kLine; ++i)
            data<int32_t>(args, 6)[at + i] = data<int32_t>(args, 2)[at + i];
        publish(data<int32_t>(args, 6) + at, kLine * sizeof(int32_t));
        if (lane != 0) {
            const uint32_t offset = (e * comm->rankNum + peer) * (elements + kLine);
            copy_and_add(data<float>(args, 4) + offset, data<float>(args, 5) + offset, elements);
        }
    }
    publish(meta, kMeta * sizeof(uint64_t));
}
