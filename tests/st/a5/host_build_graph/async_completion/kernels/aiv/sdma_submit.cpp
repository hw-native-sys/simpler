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
#include "backend/sdma/sdma_submission_contract.h"
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
    Global destination(data<float>(args[1]));
    Global source(remote);
    uint64_t transfer_bytes = 0;
    bool failed =
        workspace == nullptr || comm->rankNum != 2 || !async_sdma_transfer_bytes(destination, source, &transfer_bytes);
    if (!failed) failed = !pto::comm::BuildAsyncSession(scratch, workspace, session, 0);
    AsyncCompletionAllowance allowance;
    if (!failed)
        failed = session.cfg.sdma.queueNum > kSdmaRecords ||
                 !async_completion_reserve(ctx, session.cfg.sdma.queueNum, &allowance);
    if (!failed) {
        const auto event = pto::comm::TGET_ASYNC(destination, source, session);
        const uint32_t count = event.CompletionRecordCount(session);
        failed = !async_sdma_event_valid(
                     SdmaOp::TGET, transfer_bytes, event.handle, event.postFailed, static_cast<uint32_t>(event.engine),
                     static_cast<uint32_t>(pto::comm::DmaEngine::SDMA)
                 ) ||
                 count == 0 || count > kSdmaRecords || !async_completion_accept_count(ctx, allowance, count);
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
                if (!register_completion_condition(ctx, token)) {
                    failed = true;
                    break;
                }
                meta[kSdmaAddresses + 2 * i] = reinterpret_cast<uint64_t>(record.addr);
                meta[kSdmaAddresses + 2 * i + 1] = record.expected;
                flush(meta + kSdmaAddresses + 2 * i);
            }
            if (!failed) add(meta + 112, count);
        }
    }
    if (failed) {
        add(meta + 32, 1);
        // AsyncCtx exposes this writer error channel; no scheduler state is inspected or patched.
        if (ctx.completion_error_code != nullptr) {
            *ctx.completion_error_code = SIMPLER_ERROR_ASYNC_COMPLETION_INVALID;
            flush(const_cast<__gm__ int32_t *>(ctx.completion_error_code));
        }
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
