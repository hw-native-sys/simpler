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
#include "backend/urma/urma_completion_kernel.h"
#include "platform_comm/comm_context.h"
using namespace pto;
using namespace async_case;
extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    auto *meta = data<uint64_t>(args[3]);
    auto *comm = reinterpret_cast<__gm__ CommContext *>(args[4]);
    auto *workspace = reinterpret_cast<__gm__ uint8_t *>(comm_workspace<CommWorkspaceKind::URMA>(*comm).addr);
    auto ctx = get_async_ctx(args);
    const uint32_t peer = 1 - comm->rankId;
    const bool put = args[6] != 0;
    const int64_t n = args[5];
    const uint64_t offset = reinterpret_cast<uint64_t>(data<float>(args[put ? 1 : 0])) - comm->windowsIn[comm->rankId];
    auto *remote = simpler::urma::peer_mr_ptr<float>(workspace, peer, offset);
    using S = Shape<1, 1, 1, 1, DYNAMIC>;
    using R = pto::Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, 1>;
    using G = GlobalTensor<float, S, R>;
    G dst(put ? remote : data<float>(args[1]), S(1, 1, 1, 1, n), R(n, n, n, n, 1));
    G src(put ? data<float>(args[0]) : remote, S(1, 1, 1, 1, n), R(n, n, n, n, 1));
    bool ok = ctx.resident_enabled == 1;
    if (!ok) add(meta + 16, 1);
    if (ok)
        ok = send_request_entry(
            ctx, UrmaRequestDescriptor<G, G>{put ? UrmaOp::TPUT : UrmaOp::TGET, dst, src, workspace, peer}
        );
    if (!ok) {
        add(meta + 32, 1);
        return;
    }
    if (n != 0) {
        const auto &entry = ctx.completion_entries[0];
        const uint32_t row = static_cast<uint32_t>(entry.addr >> 32);
        auto *info = reinterpret_cast<__gm__ pto::comm::urma::UrmaInfo *>(workspace);
        meta[128] = info->scqPtr + row * sizeof(pto::comm::urma::UrmaCqCtx);
        meta[144] = entry.expected_value;
        meta[192] = info->sqPtr + row * sizeof(pto::comm::urma::UrmaWQCtx);
        meta[208] = static_cast<uint32_t>(entry.addr);
        meta[224] = row;
        flush(meta + 192);
        flush(meta + 208);
        flush(meta + 224);
        flush(meta + 128);
        flush(meta + 144);
        if (args[7] != 0) {
            bool done = false;
            pto::comm::AsyncSession session;
            ok = simpler::urma::build_urma_session(ctx.urma_sender_id, workspace, peer, session);
            if (ok)
                ok = pto::comm::urma::detail::UrmaPollCqAt(
                    session, peer, ctx.urma_sender_id, static_cast<uint32_t>(entry.addr), entry.expected_value, true,
                    done
                );
            if (!ok || !done) add(meta + 32, 1);
        }
    }
    meta[176] = ticks_per_ms();
    flush(meta + 176);
    add(meta, 1);
}
