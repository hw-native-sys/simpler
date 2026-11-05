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

#include "backend/urma/urma_completion_kernel.h"
#include "platform_comm/comm_context.h"
#include "tensor.h"

namespace {
constexpr uint32_t kWidth = 128;
using Shape = pto::Shape<1, 1, 1, 1, kWidth>;
using Stride = pto::Stride<kWidth, kWidth, kWidth, kWidth, 1>;
using GlobalRow = pto::GlobalTensor<float, Shape, Stride>;
using RowTile = pto::Tile<pto::TileType::Vec, float, 1, kWidth>;

inline __aicore__ __gm__ float *data(__gm__ Tensor *tensor) {
    return reinterpret_cast<__gm__ float *>(tensor->buffer.addr) + tensor->start_offset;
}
template <uint32_t Request, uint32_t Row>
inline __aicore__ bool gather_row(
    RowTile &tile, AsyncCtx &async, __gm__ float *table, __gm__ float *peer_table, __gm__ float *local_out,
    __gm__ float *peer_out, __gm__ uint8_t *workspace, uint32_t peer, uint32_t block, uint32_t blocks
) {
    if (Request % blocks != block) return true;
    GlobalRow source(table + Row * kWidth);
    GlobalRow destination(local_out + Request * kWidth);
    pto::TLOAD(tile, source);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    pto::TSTORE(destination, tile);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    return send_request_entry(
        async, UrmaTget(GlobalRow(peer_out + Request * kWidth), GlobalRow(peer_table + Row * kWidth), workspace, peer)
    );
}
}  // namespace

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    auto *table = data(reinterpret_cast<__gm__ Tensor *>(args[0]));
    auto *local_out = data(reinterpret_cast<__gm__ Tensor *>(args[1]));
    auto *peer_out = data(reinterpret_cast<__gm__ Tensor *>(args[2]));
    auto *comm = reinterpret_cast<__gm__ CommContext *>(args[3]);
    AsyncCtx async = get_async_ctx(args);
    const uint32_t block = get_block_idx(args);
    const uint32_t blocks = get_block_num(args);
    if (comm == nullptr || comm->rankNum != 2 || comm->rankId >= 2 || (blocks != 1 && blocks != 4) || block >= blocks ||
        comm_workspace<CommWorkspaceKind::URMA>(*comm).addr == 0 || comm->windowsIn[comm->rankId] == 0) {
        pto2::detail::defer_error(async, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
        pto2::detail::defer_flush(async);
        return;
    }
    auto *workspace = reinterpret_cast<__gm__ uint8_t *>(comm_workspace<CommWorkspaceKind::URMA>(*comm).addr);
    const uint32_t peer = 1u - comm->rankId;
    const uint64_t offset = reinterpret_cast<uint64_t>(table) - comm->windowsIn[comm->rankId];
    auto *peer_table = pto2::urma_backend::peer_mr_ptr<float>(workspace, peer, offset);

    RowTile tile;
    pto::TASSIGN(tile, 0);
    if (!gather_row<0, 0>(tile, async, table, peer_table, local_out, peer_out, workspace, peer, block, blocks)) return;
    if (!gather_row<1, 63>(tile, async, table, peer_table, local_out, peer_out, workspace, peer, block, blocks)) return;
    if (!gather_row<2, 7>(tile, async, table, peer_table, local_out, peer_out, workspace, peer, block, blocks)) return;
    if (!gather_row<3, 7>(tile, async, table, peer_table, local_out, peer_out, workspace, peer, block, blocks)) return;
    if (!gather_row<4, 1>(tile, async, table, peer_table, local_out, peer_out, workspace, peer, block, blocks)) return;
    if (!gather_row<5, 62>(tile, async, table, peer_table, local_out, peer_out, workspace, peer, block, blocks)) return;
    if (!gather_row<6, 31>(tile, async, table, peer_table, local_out, peer_out, workspace, peer, block, blocks)) return;
    if (!gather_row<7, 32>(tile, async, table, peer_table, local_out, peer_out, workspace, peer, block, blocks)) return;
    pipe_barrier(PIPE_ALL);
}
