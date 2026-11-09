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

#include "../sdma.h"
#include "async_kernel_api.h"
#include "backend/sdma/sdma_completion_kernel.h"
#include "platform_comm/comm_context.h"

using namespace pto;
using namespace async_case;

namespace {
template <int N>
__aicore__ void submit(__gm__ int64_t *args, AsyncCtx &ctx, __gm__ uint64_t *meta) {
    auto *comm = reinterpret_cast<__gm__ CommContext *>(args[4]);
    auto *local = data<float>(args[0]);
    const uint64_t offset = reinterpret_cast<uint64_t>(local) - comm->windowsIn[comm->rankId];
    auto *remote = reinterpret_cast<__gm__ float *>(comm->windowsIn[1 - comm->rankId] + offset);
    using ShapeType = Shape<1, 1, 1, 1, N>;
    using StrideType = pto::Stride<N, N, N, N, 1>;
    using Global = GlobalTensor<float, ShapeType, StrideType>;
    Tile<TileType::Vec, uint8_t, 1, 32> scratch;
    TASSIGN(scratch, 0);
    pto::comm::AsyncSession session;
    auto *workspace = reinterpret_cast<__gm__ uint8_t *>(comm_workspace<CommWorkspaceKind::SDMA>(*comm).addr);
    pto2::detail::SdmaSubmissionScope submission;
    pto2::resident::AsyncReservation reservation{};
    bool failed = workspace == nullptr || comm->rankNum != 2 || ctx.resident_enabled != 1;
    if (!failed) failed = !submission.acquire(ctx);
    if (!failed) failed = !pto::comm::BuildAsyncSession(scratch, workspace, session, 0);
    if (!failed) {
        reservation = pto2::resident::reserve_conditions(ctx.resident, session.cfg.sdma.queueNum);
        failed = reservation.serial == 0;
    }
    if (!failed) {
        Global destination(data<float>(args[1]));
        Global source(remote);
        const auto event = pto::comm::TGET_ASYNC(destination, source, session);
        const uint32_t count = event.CompletionRecordCount(session);
        failed = event.postFailed || !event.valid() || count == 0 || count > kSdmaRecords;
        if (!failed) {
            for (uint32_t i = 0; i < count; ++i) {
                const auto record = event.CompletionRecordAt(session, i);
                if (record.addr == nullptr || record.expected == 0 ||
                    record.kind != pto::comm::CompletionKind::SDMA_POST_DONE) {
                    failed = true;
                    break;
                }
                const CompletionToken token{
                    reinterpret_cast<uint64_t>(record.addr), 0, COMPLETION_ENGINE_SDMA,
                    COMPLETION_TYPE_SDMA_EVENT_RECORD, record.expected
                };
                if (!pto2::resident::commit_condition(ctx.resident, reservation, token)) {
                    failed = true;
                    break;
                }
                meta[kSdmaAddresses + 2 * i] = reinterpret_cast<uint64_t>(record.addr);
                meta[kSdmaAddresses + 2 * i + 1] = record.expected;
                flush(meta + kSdmaAddresses + 2 * i);
            }
            if (!failed) failed = !pto2::resident::finish_reservation(ctx.resident, reservation, count);
            if (!failed) add(meta + 112, count);
        }
    }
    if (!submission.release(!failed, workspace)) failed = true;
    if (failed) {
        add(meta + 32, 1);
        pto2::detail::defer_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
    }
}
}  // namespace

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    auto *meta = data<uint64_t>(args[3]);
    auto ctx = get_async_ctx(args);
    if (!async_ctx_is_deferred(ctx)) {
        add(meta + 16, 1);
    } else if (args[5] == 8) {
        submit<8>(args, ctx, meta);
    } else if (args[5] == 1024 * 1024) {
        submit<1024 * 1024>(args, ctx, meta);
    } else if (args[5] != 0) {
        add(meta + 32, 1);
    }
    add(meta, 1);
}
