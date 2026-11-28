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

#include "async_kernel_api.h"
#include "backend/urma/urma_completion_kernel.h"
#include "platform_comm/comm_context.h"
#include "tensor.h"

namespace {
using Shape = pto::Shape<1, 1, 1, 1, pto::DYNAMIC>;
using Stride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1>;
using Data = pto::GlobalTensor<float, Shape, Stride>;

inline __aicore__ Data data(__gm__ float *ptr, uint32_t count) {
    return Data(ptr, Shape(1, 1, 1, 1, count), Stride(count, count, count, count, 1));
}

inline __aicore__ __gm__ float *base(int64_t arg) {
    const auto *tensor = reinterpret_cast<const __gm__ Tensor *>(arg);
    return reinterpret_cast<__gm__ float *>(tensor->buffer.addr) + tensor->start_offset;
}
}  // namespace

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    const auto *comm = reinterpret_cast<const __gm__ CommContext *>(args[3]);
    const uint32_t count = static_cast<uint32_t>(args[4]);
    const uint32_t mode = static_cast<uint32_t>(args[5]);
    const bool mixed = mode == 1;
    const uint32_t block = get_block_idx(args);
    const uint32_t lane = get_sub_block_id(args);
    const uint32_t slot = mixed ? 2 * block + lane : block;
    const uint32_t slots = (mixed ? 2 : 1) * static_cast<uint32_t>(get_block_num(args));
    const uint32_t segment = count / slots;
    const uint32_t guard = 32;
    auto ctx = get_async_ctx(args);
    if (!comm || comm->rankNum != 2 || comm->rankId >= 2 || count % slots != 0 || segment <= 2 * guard) {
        pto2::detail::defer_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
        return;
    }
    const auto &workspace = comm_workspace<CommWorkspaceKind::URMA>(*comm);
    const uint32_t peer = 1 - comm->rankId;
    const uint64_t offset = reinterpret_cast<uint64_t>(base(args[0]) + slot * segment) - comm->windowsIn[comm->rankId];
    auto *remote =
        pto2::urma_backend::peer_mr_ptr<float>(reinterpret_cast<__gm__ uint8_t *>(workspace.addr), peer, offset);
    auto *out = base(args[2]) + slot * segment;
    const uint32_t payload = segment - 2 * guard;
    if (!send_request_entry(ctx, UrmaTget(data(out + guard, payload), data(remote + guard, payload), workspace, peer)))
        return;
    const auto &entry = ctx.completion_entries[0];
    const auto *info = reinterpret_cast<const __gm__ pto::comm::urma::UrmaInfo *>(workspace.addr);
    for (uint32_t i = 0; i < guard; ++i) {
        out[i] = -77.0f;
        out[segment - guard + i] = -77.0f;
    }
    out[0] = static_cast<float>(ctx.urma_sender_id);
    out[1] = static_cast<float>(entry.addr >> 32);
    out[2] = static_cast<float>(static_cast<uint32_t>(entry.addr));
    out[3] = static_cast<float>(entry.expected_value);
    out[4] = static_cast<float>(block);
    out[5] = static_cast<float>(lane);
    out[6] = static_cast<float>(load_global_context(args).async_group_index);
    out[7] = static_cast<float>(info->rowsPerPeer);
    out[8] = static_cast<float>(info->jettiesPerCore);
    out[9] = entry.backend_cookie == workspace.addr && *ctx.completion_count == 1 ? 1.0f : 0.0f;
    pto2::detail::defer_flush_range(out, guard * sizeof(float));
    pto2::detail::defer_flush_range(out + segment - guard, guard * sizeof(float));
    dsb(DSB_DDR);
    if (mode >= 2) save_expected_notification_counter(ctx, reinterpret_cast<__gm__ int32_t *>(base(args[1])), 1);
}
