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

#include <cstdint>

#include <pto/pto-inst.hpp>

#include "common/platform_config.h"
#include "platform_comm/comm_context.h"
#include "tensor.h"

namespace {
struct DeviceGateWord {
    volatile int64_t value;
    int64_t padding[15];
};
static_assert(sizeof(DeviceGateWord) == 128);

inline __aicore__ bool wait_device_gate(uint64_t address) {
    if (address == 0) return true;
    auto *gate = reinterpret_cast<__gm__ DeviceGateWord *>(address);
    gate[0].value = 1;
    dcci(&gate[0], cache_line_t::SINGLE_CACHE_LINE, dcci_dst_t::CACHELINE_OUT);
    dsb(static_cast<mem_dsb_t>(0));
    const uint64_t started = get_sys_cnt();
    while (get_sys_cnt() - started < 10ULL * PLATFORM_PROF_SYS_CNT_FREQ) {
        dcci(&gate[1], cache_line_t::SINGLE_CACHE_LINE);
        dsb(static_cast<mem_dsb_t>(0));
        if (gate[1].value == 1) return true;
    }
    gate[0].value = -1;
    dcci(&gate[0], cache_line_t::SINGLE_CACHE_LINE, dcci_dst_t::CACHELINE_OUT);
    dsb(static_cast<mem_dsb_t>(0));
    return false;
}
}  // namespace

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    auto *local = reinterpret_cast<__gm__ Tensor *>(args[0]);
    auto *output = reinterpret_cast<__gm__ Tensor *>(args[1]);
    auto *comm = reinterpret_cast<__gm__ CommContext *>(args[2]);
    auto *dst = reinterpret_cast<__gm__ float *>(output->buffer.addr) + output->start_offset;
    bool valid = comm != nullptr && comm->rankNum == 2 && comm->rankId < 2;
    if (valid) {
        for (uint32_t i = 0; i < COMM_WORKSPACE_COUNT; ++i)
            valid = valid && comm->workspaces[i].addr == 0 && comm->workspaces[i].size == 0;
        valid = valid && comm->windowsIn[comm->rankId] != 0 && comm->windowsIn[1u - comm->rankId] != 0;
    }
    if (!valid) {
        dst[0] = -1.0f;
        pipe_barrier(PIPE_ALL);
        return;
    }

    const auto local_address = local->buffer.addr + local->start_offset * sizeof(float);
    const uint64_t offset = local_address - comm->windowsIn[comm->rankId];
    auto *peer = reinterpret_cast<__gm__ float *>(comm->windowsIn[1u - comm->rankId] + offset);
    using Shape = pto::Shape<1, 1, 1, 8, 128>;
    using Stride = pto::Stride<1024, 1024, 1024, 128, 1>;
    using GlobalData = pto::GlobalTensor<float, Shape, Stride>;
    using Tile = pto::Tile<pto::TileType::Vec, float, 8, 128>;
    Tile tile;
    pto::TASSIGN(tile, 0);
    for (uint32_t chunk = 0; chunk < 16; ++chunk) {
        if (chunk == 8 && !wait_device_gate(static_cast<uint64_t>(args[3]))) return;
        GlobalData source(peer + chunk * 1024);
        GlobalData destination(dst + chunk * 1024);
        pto::TLOAD(tile, source);
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        pto::TSTORE(destination, tile);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    }
    pipe_barrier(PIPE_ALL);
}
