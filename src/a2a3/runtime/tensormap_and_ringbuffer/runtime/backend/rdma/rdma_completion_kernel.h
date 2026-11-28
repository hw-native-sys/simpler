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

#include <pto/pto-inst.hpp>
#include <pto/comm/async_common/async_event_impl.hpp>
#include <pto/comm/async/rdma/rdma_async_intrin.hpp>
#include <pto/comm/async/rdma/backends/in_die/in_die_types.hpp>

#if !defined(PTO_RDMA_SUPPORTED) || !defined(PTO_RDMA_BACKEND_IN_DIE_SUPPORTED)
#error "A3 RDMA kernels require the PTO IN_DIE backend"
#endif

#include "async_kernel_api.h"
#include "platform_comm/comm_context.h"
#include "rdma/rdma_request.h"

namespace simpler::rdma {

inline __aicore__ bool fail_request(AsyncCtx &ctx) {
    if (ctx.completion_error_code != nullptr && *ctx.completion_error_code == SIMPLER_ERROR_NONE)
        pto2::detail::defer_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
    pto2::detail::defer_flush(ctx);
    return false;
}

template <bool Put, typename Scratch>
inline __aicore__ bool
submit_checked(AsyncCtx &ctx, const RdmaTransferArgsV1 &request, Scratch &scratch, uint32_t sync_id) {
    if (!async_ctx_is_deferred(ctx) || ctx.completion_count == nullptr || ctx.completion_entries == nullptr ||
        ctx.completion_error_code == nullptr || ctx.completion_capacity > MAX_COMPLETIONS_PER_TASK ||
        request.comm_context == 0 || request.comm_context % alignof(CommContext) != 0 || sync_id > 7)
        return fail_request(ctx);
    if (*ctx.completion_error_code != SIMPLER_ERROR_NONE) return fail_request(ctx);
    if (*ctx.completion_count >= ctx.completion_capacity) {
        pto2::detail::defer_error(ctx, SIMPLER_ERROR_ASYNC_WAIT_OVERFLOW);
        pto2::detail::defer_flush(ctx);
        return false;
    }
    auto *context = reinterpret_cast<__gm__ CommContext *>(request.comm_context);
    const auto &workspace = comm_workspace<CommWorkspaceKind::RDMA>(*context);
    if (context->rankNum == 0 || context->rankNum > COMM_MAX_RANK_NUM || workspace.addr == 0 ||
        workspace.addr % alignof(pto::comm::rdma::RdmaInfo) != 0 ||
        workspace.size < sizeof(pto::comm::rdma::RdmaInfo) || workspace.addr > UINT64_MAX - workspace.size)
        return fail_request(ctx);
    auto *info = reinterpret_cast<__gm__ pto::comm::rdma::RdmaInfo *>(workspace.addr);
    if (info->magic != pto::comm::rdma::kRdmaWorkspaceMagic ||
        info->version != pto::comm::rdma::kRdmaWorkspaceVersion || info->backend != pto::comm::RdmaBackend::IN_DIE ||
        info->rankCount != context->rankNum || info->qpNum == 0 || info->qpNum > 8 ||
        !rdma_request_in_bounds(request, context->rankId, context->rankNum, info->qpNum, context->winSize))
        return fail_request(ctx);
    const uint64_t queue_bytes = uint64_t(info->rankCount) * info->qpNum * sizeof(pto::comm::rdma::in_die::Queue);
    if (workspace.size <
            sizeof(*info) + queue_bytes + uint64_t(info->rankCount + 1) * sizeof(pto::comm::rdma::RdmaMemInfo) ||
        info->sqPtr != workspace.addr + sizeof(*info) || info->scqPtr != info->sqPtr ||
        info->memPtr != info->sqPtr + queue_bytes)
        return fail_request(ctx);
    const uint64_t local = context->windowsIn[context->rankId];
    const uint64_t remote = context->windowsOut[request.peer_rank];
    auto *memory = reinterpret_cast<__gm__ pto::comm::rdma::RdmaMemInfo *>(info->memPtr);
    if (memory[context->rankId].addr != local || memory[request.peer_rank].addr != remote ||
        memory[context->rankId].size < context->winSize || memory[request.peer_rank].size < context->winSize)
        return fail_request(ctx);
    if (local == 0 || remote == 0 || local > UINT64_MAX - context->winSize || remote > UINT64_MAX - context->winSize)
        return fail_request(ctx);
    auto *workspace_gm = reinterpret_cast<__gm__ uint8_t *>(workspace.addr);
    pto::comm::AsyncSession session;
    if (!pto::comm::BuildAsyncSession<pto::comm::DmaEngine::RDMA>(
            scratch, workspace_gm, context->rankId, session, sync_id, static_cast<uint32_t>(request.qp_index)
        ))
        return fail_request(ctx);
    using Shape = pto::Shape<1, 1, 1, 1, pto::DYNAMIC>;
    using Stride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1>;
    using Tensor = pto::GlobalTensor<uint8_t, Shape, Stride, pto::Layout::ND>;
    const auto bytes = static_cast<int64_t>(request.nbytes);
    const Shape shape(1, 1, 1, 1, bytes);
    const Stride stride(bytes, bytes, bytes, bytes, 1);
    Tensor local_tensor(reinterpret_cast<__gm__ uint8_t *>(local + request.local_offset), shape, stride);
    Tensor remote_tensor(reinterpret_cast<__gm__ uint8_t *>(remote + request.remote_offset), shape, stride);
    pto::comm::AsyncEvent event;
    if constexpr (Put) {
        event = pto::comm::TPUT_ASYNC<pto::comm::DmaEngine::RDMA>(
            remote_tensor, local_tensor, session, static_cast<uint32_t>(request.peer_rank)
        );
    } else {
        event = pto::comm::TGET_ASYNC<pto::comm::DmaEngine::RDMA>(
            local_tensor, remote_tensor, session, static_cast<uint32_t>(request.peer_rank)
        );
    }
    if (event.postFailed || event.handle == 0 || static_cast<uint32_t>(event.handle >> 32) == UINT32_MAX ||
        event.CompletionRecordCount(session) != 1 ||
        static_cast<uint32_t>(event.handle >> 32) != request.peer_rank * info->qpNum + request.qp_index ||
        static_cast<uint32_t>(event.handle) == 0)
        return fail_request(ctx);
    const auto record = event.CompletionRecordAt(session, 0);
    auto *queues = reinterpret_cast<__gm__ pto::comm::rdma::in_die::Queue *>(info->sqPtr);
    const auto &cq = queues[static_cast<uint32_t>(event.handle >> 32)].cq;
    const auto index = static_cast<uint32_t>(event.handle) - 1;
    if (cq.depth < 2 || (cq.depth & (cq.depth - 1)) != 0 || cq.stride != 32 || record.addr == nullptr ||
        record.kind != pto::comm::CompletionKind::RDMA_IN_DIE_CQE ||
        reinterpret_cast<uint64_t>(record.addr) != cq.buffer + uint64_t(index & (cq.depth - 1)) * cq.stride ||
        record.expected != ((index & cq.depth) == 0 ? 1U : 0U))
        return fail_request(ctx);
    const CompletionToken token{
        event.handle, 0, COMPLETION_ENGINE_ROCE, COMPLETION_TYPE_RDMA_IN_DIE_EVENT_HANDLE, workspace.addr
    };
    if (!register_completion_condition(ctx, token)) return fail_request(ctx);
    pto2::detail::defer_flush(ctx);
    return true;
}

template <typename Scratch>
inline __aicore__ bool
rdma_put_async_checked(AsyncCtx &ctx, const RdmaTransferArgsV1 &request, Scratch &scratch, uint32_t sync_id) {
    return submit_checked<true>(ctx, request, scratch, sync_id);
}

template <typename Scratch>
inline __aicore__ bool
rdma_get_async_checked(AsyncCtx &ctx, const RdmaTransferArgsV1 &request, Scratch &scratch, uint32_t sync_id) {
    return submit_checked<false>(ctx, request, scratch, sync_id);
}

}  // namespace simpler::rdma
