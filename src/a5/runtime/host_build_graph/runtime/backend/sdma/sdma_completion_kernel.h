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

#include <stdint.h>

#include "async_kernel_api.h"
#include "sdma_submission_contract.h"
#include "sdma_submission_guard.h"
#include "aicore_completion_mailbox_types.h"
#include "host_build_graph/runtime_status.h"

#ifndef __aicore__
#define __aicore__
#endif
#ifndef __gm__
#define __gm__
#endif

// Re-exposed PTO-ISA constant so examples / callers don't need to include
// <pto/comm/async/sdma/sdma_types.hpp> just to spell their scratch tile.
inline constexpr uint32_t SDMA_SCRATCH_ALIGNMENT = pto::comm::sdma::UB_ALIGN_SIZE;

// SdmaRequestDescriptor bundles everything send_request_entry needs to drive
// one SDMA transfer + completion registration. It is a template because the
// destination / source / scratch types carry tensor shape & stride at compile
// time; the SdmaTget() / SdmaTput() helpers below let callers skip the
// template arguments.
//
// sync_id is forwarded to PTO-ISA's session builder for its internal pipe
// synchronization. Today every caller submits one request per kernel
// invocation, so passing 0 is safe.
template <typename DstTensor, typename SrcTensor, typename ScratchTileT>
struct SdmaRequestDescriptor {
    SdmaOp op;
    DstTensor dst;
    SrcTensor src;
    ScratchTileT scratch;
    __gm__ uint8_t *workspace;
    uint32_t sync_id;
};

template <typename DstTensor, typename SrcTensor, typename ScratchTileT>
inline __aicore__ SdmaRequestDescriptor<DstTensor, SrcTensor, ScratchTileT> SdmaTget(
    const DstTensor &dst, const SrcTensor &src, const ScratchTileT &scratch, __gm__ uint8_t *workspace,
    uint32_t sync_id = 0
) {
    return SdmaRequestDescriptor<DstTensor, SrcTensor, ScratchTileT>{SdmaOp::TGET, dst,       src,
                                                                     scratch,      workspace, sync_id};
}

template <typename DstTensor, typename SrcTensor, typename ScratchTileT>
inline __aicore__ SdmaRequestDescriptor<DstTensor, SrcTensor, ScratchTileT> SdmaTput(
    const DstTensor &dst, const SrcTensor &src, const ScratchTileT &scratch, __gm__ uint8_t *workspace,
    uint32_t sync_id = 0
) {
    return SdmaRequestDescriptor<DstTensor, SrcTensor, ScratchTileT>{SdmaOp::TPUT, dst,       src,
                                                                     scratch,      workspace, sync_id};
}

namespace pto2::detail {

inline __aicore__ bool
register_sdma_post_done_record(AsyncCtx &ctx, volatile __gm__ void *record_addr, uint64_t post_id) {
    CompletionToken token{
        reinterpret_cast<uint64_t>(record_addr), 0, COMPLETION_ENGINE_SDMA, COMPLETION_TYPE_SDMA_EVENT_RECORD, post_id
    };
    return register_completion_condition(ctx, token);
}

template <typename PtoAsyncEvent, typename PtoAsyncSession>
inline __aicore__ bool register_pto_async_event(
    AsyncCtx &ctx, const PtoAsyncEvent &event, const PtoAsyncSession &session, const AsyncCompletionAllowance &allowance
) {
    const uint32_t engine = static_cast<uint32_t>(event.engine);
    if (event.postFailed || engine != static_cast<uint32_t>(::pto::comm::DmaEngine::SDMA)) {
        defer_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
        return false;
    }
    if (!ctx.task_token.is_valid()) return event.Wait(session);
    if (event.handle == 0) return async_completion_accept_count(ctx, allowance, 0);

    const uint32_t record_count = event.CompletionRecordCount(session);
    if (record_count == 0 || record_count == ::pto::comm::kCompletionRecordCountError ||
        !async_completion_accept_count(ctx, allowance, record_count)) {
        defer_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
        return false;
    }
    for (uint32_t index = 0; index < record_count; ++index) {
        const auto record = event.CompletionRecordAt(session, index);
        if (record.kind != ::pto::comm::CompletionKind::SDMA_POST_DONE || record.addr == nullptr ||
            (reinterpret_cast<uint64_t>(record.addr) & 7u) != 0 || record.expected == 0) {
            defer_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
            return false;
        }
        if (!register_sdma_post_done_record(ctx, record.addr, record.expected)) return false;
    }
    return true;
}

}  // namespace pto2::detail

// SDMA overload of the runtime's send_request_entry. Submits the descriptor
// to PTO-ISA, then registers the resulting AsyncEvent's GM post-done records into the
// AsyncCtx deferred-wait slab and flushes. Returns false on submit/session
// failure (also records the error in ctx.completion_error_code).
template <typename DstTensor, typename SrcTensor, typename ScratchTileT>
inline __aicore__ bool
send_request_entry(AsyncCtx &ctx, SdmaRequestDescriptor<DstTensor, SrcTensor, ScratchTileT> desc) {
    if (ctx.task_token.is_valid() && !async_completion_writer_valid(ctx)) {
        pto2::detail::defer_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
        pto2::detail::defer_flush(ctx);
        return false;
    }
    if (ctx.task_token.is_valid() && *ctx.completion_error_code != 0) {
        pto2::detail::defer_flush(ctx);
        return false;
    }
    if (desc.op != SdmaOp::TGET && desc.op != SdmaOp::TPUT) {
        pto2::detail::defer_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
        pto2::detail::defer_flush(ctx);
        return false;
    }
    uint64_t transfer_bytes = 0;
    if (!async_sdma_transfer_bytes(desc.dst, desc.src, &transfer_bytes)) {
        pto2::detail::defer_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
        pto2::detail::defer_flush(ctx);
        return false;
    }
    if (transfer_bytes == 0) return true;
    pto2::detail::SdmaSubmissionScope submission;
    if (!submission.acquire(ctx, static_cast<uint32_t>(get_block_idx()))) {
        pto2::detail::defer_flush(ctx);
        return false;
    }
    pto::comm::AsyncSession session;
    if (!pto::comm::BuildAsyncSession(desc.scratch, desc.workspace, session, desc.sync_id)) {
        pto2::detail::defer_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
        pto2::detail::defer_flush(ctx);
        return false;
    }

    // A post may produce a completion record for every queue in this session.
    // The context is single-writer; hold that entire suffix until the submit
    // returns, then accept only an actual record count within this allowance.
    AsyncCompletionAllowance allowance;
    if (ctx.task_token.is_valid() && !async_completion_reserve(ctx, session.cfg.sdma.queueNum, &allowance)) {
        pto2::detail::defer_flush(ctx);
        return false;
    }
    pto::comm::AsyncEvent event;
    if (desc.op == SdmaOp::TGET) {
        event = pto::comm::TGET_ASYNC(desc.dst, desc.src, session);
    } else {
        event = pto::comm::TPUT_ASYNC(desc.dst, desc.src, session);
    }
    if (!async_sdma_event_valid(
            desc.op, transfer_bytes, event.handle, event.postFailed, static_cast<uint32_t>(event.engine),
            static_cast<uint32_t>(pto::comm::DmaEngine::SDMA)
        )) {
        pto2::detail::defer_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
        pto2::detail::defer_flush(ctx);
        return false;
    }
    if (!ctx.task_token.is_valid()) return submission.release(true, desc.workspace) && event.Wait(session);
    const bool registered = pto2::detail::register_pto_async_event(ctx, event, session, allowance);
    if (!submission.release(registered, desc.workspace)) {
        pto2::detail::defer_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
        pto2::detail::defer_flush(ctx);
        return false;
    }
    pto2::detail::defer_flush(ctx);
    return registered;
}

// The args-aware path retains its scheduler-granted channel and resource checks.
#include "sdma_qualified_kernel.h"
