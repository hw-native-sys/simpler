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

#ifndef SRC_A2A3_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_BACKEND_SDMA_SDMA_COMPLETION_KERNEL_H_
#define SRC_A2A3_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_BACKEND_SDMA_SDMA_COMPLETION_KERNEL_H_

#include <stdint.h>

#include <pto/comm/async_common/async_event_impl.hpp>
#include <pto/npu/comm/async/sdma/sdma_async_intrin.hpp>

#include "pto_async_kernel_api.h"
#include "pto_completion_ingress.h"
#include "pto_runtime_status.h"

#ifndef __aicore__
#define __aicore__
#endif
#ifndef __gm__
#define __gm__
#endif

// Re-exposed PTO-ISA constant so examples / callers don't need to include
// <pto/npu/comm/async/sdma/sdma_types.hpp> just to spell their scratch tile.
inline constexpr uint32_t SDMA_SCRATCH_ALIGNMENT = pto::comm::sdma::UB_ALIGN_SIZE;

enum class SdmaOp : uint8_t {
    TGET = 0,
    TPUT = 1,
};

// SdmaRequestDescriptor bundles everything send_request_entry needs to drive
// one SDMA transfer + completion registration. It is a template because the
// destination / source / scratch types carry tensor shape & stride at compile
// time; the SdmaTget() / SdmaTput() helpers below let callers skip the
// template arguments.
//
// sync_id selects which event-record slot inside the workspace the engine
// writes into. Concurrent dispatches must use distinct sync_ids; today every
// caller submits one request per kernel invocation so passing 0 is safe.
// Future work (see .docs/25.comm-api-refactor/03.implementation-plan.md §5.2)
// will fold sync_id allocation into the adapter.
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
    return SdmaRequestDescriptor<DstTensor, SrcTensor, ScratchTileT>{SdmaOp::TGET, dst, src, scratch, workspace,
                                                                     sync_id};
}

template <typename DstTensor, typename SrcTensor, typename ScratchTileT>
inline __aicore__ SdmaRequestDescriptor<DstTensor, SrcTensor, ScratchTileT> SdmaTput(
    const DstTensor &dst, const SrcTensor &src, const ScratchTileT &scratch, __gm__ uint8_t *workspace,
    uint32_t sync_id = 0
) {
    return SdmaRequestDescriptor<DstTensor, SrcTensor, ScratchTileT>{SdmaOp::TPUT, dst, src, scratch, workspace,
                                                                     sync_id};
}

namespace pto2::detail {

inline __aicore__ void register_sdma_event_record(AsyncCtx &ctx, volatile __gm__ void *record_addr) {
    CompletionToken token{reinterpret_cast<uint64_t>(record_addr), 0, COMPLETION_ENGINE_SDMA,
                          COMPLETION_TYPE_SDMA_EVENT_RECORD, 0};
    (void)register_completion_condition(ctx, token);
}

template <typename PtoAsyncEvent, typename PtoAsyncSession>
inline __aicore__ void
register_pto_async_event(AsyncCtx &ctx, const PtoAsyncEvent &event, const PtoAsyncSession &session) {
    if (ctx.task_token.is_invalid() || ctx.completion_count == nullptr || ctx.completion_entries == nullptr) {
        (void)event.Wait(session);
        return;
    }
    if (event.handle == 0) {
        return;
    }

    const uint32_t engine = static_cast<uint32_t>(event.engine);
    if (engine != static_cast<uint32_t>(::pto::comm::DmaEngine::SDMA)) {
        defer_error(ctx, PTO2_ERROR_ASYNC_COMPLETION_INVALID);
        return;
    }

    ::pto::comm::sdma::detail::UbTmpBuf tmp_buf;
    uint32_t sync_id = 0;
    __gm__ uint8_t *recv_workspace = nullptr;
    uint32_t queue_num = 0;
    if (!::pto::comm::sdma::detail::PrepareEventCheck(
            session.sdmaSession, tmp_buf, sync_id, recv_workspace, queue_num
        )) {
        defer_error(ctx, PTO2_ERROR_ASYNC_COMPLETION_INVALID);
        return;
    }
    for (uint32_t queue_id = 0; queue_id < queue_num; ++queue_id) {
        register_sdma_event_record(ctx, ::pto::comm::sdma::detail::GetEventRecord(recv_workspace, queue_id));
    }
}

}  // namespace pto2::detail

// SDMA overload of the runtime's send_request_entry. Submits the descriptor
// to PTO-ISA, then registers the resulting AsyncEvent's GM flag(s) into the
// AsyncCtx deferred-wait slab and flushes. Returns false on submit/session
// failure (also records the error in ctx.completion_error_code).
template <typename DstTensor, typename SrcTensor, typename ScratchTileT>
inline __aicore__ bool send_request_entry(
    AsyncCtx &ctx, const SdmaRequestDescriptor<DstTensor, SrcTensor, ScratchTileT> &desc
) {
    ScratchTileT scratch = desc.scratch;
    pto::comm::AsyncSession session;
    if (!pto::comm::BuildAsyncSession(scratch, desc.workspace, session, desc.sync_id)) {
        pto2::detail::defer_error(ctx, PTO2_ERROR_ASYNC_COMPLETION_INVALID);
        return false;
    }

    DstTensor dst = desc.dst;
    SrcTensor src = desc.src;
    pto::comm::AsyncEvent event;
    if (desc.op == SdmaOp::TGET) {
        event = pto::comm::TGET_ASYNC(dst, src, session);
    } else {
        event = pto::comm::TPUT_ASYNC(dst, src, session);
    }
    pto2::detail::register_pto_async_event(ctx, event, session);
    pto2::detail::defer_flush(ctx);
    return true;
}

#endif  // SRC_A2A3_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_BACKEND_SDMA_SDMA_COMPLETION_KERNEL_H_
