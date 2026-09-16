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
#ifndef SRC_A5_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_BACKEND_RDMA_RDMA_COMPLETION_KERNEL_H_
#define SRC_A5_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_BACKEND_RDMA_RDMA_COMPLETION_KERNEL_H_

#include <stdint.h>

#if defined(__CPU_SIM)
#include <type_traits>
#else
#include "async_kernel_api.h"

#include <pto/comm/async_common/async_event_impl.hpp>
#if defined(PTO_RDMA_SUPPORTED)
#include <pto/comm/async/rdma/rdma_async_intrin.hpp>
#endif
#endif

#include "aicore_completion_mailbox_types.h"
#include "intrinsic.h"
#include "completion_token.h"
#include "runtime_status.h"

#ifndef __aicore__
#define __aicore__
#endif
#ifndef __gm__
#define __gm__
#endif

#if defined(__CPU_SIM)
namespace pto {
namespace comm {
enum class DmaEngine : uint8_t {
    SDMA = 0,
    URMA = 1,
    RDMA = 3,
};
}  // namespace comm
}  // namespace pto
#endif

enum class RdmaOp : uint8_t {
    TGET = 0,
    TPUT = 1,
};

template <typename DstTensor, typename SrcTensor, typename ScratchTileT>
struct RdmaRequestDescriptor {
    RdmaOp op;
    DstTensor dst;
    SrcTensor src;
    ScratchTileT scratch;
    __gm__ uint8_t *workspace;
    uint32_t peer_rank;
    uint32_t local_rank;
    uint32_t sync_id;
};

template <typename DstTensor, typename SrcTensor, typename ScratchTileT>
inline __aicore__ RdmaRequestDescriptor<DstTensor, SrcTensor, ScratchTileT> RdmaTget(
    const DstTensor &dst, const SrcTensor &src, const ScratchTileT &scratch, __gm__ uint8_t *workspace,
    uint32_t peer_rank, uint32_t local_rank, uint32_t sync_id = 0
) {
    return RdmaRequestDescriptor<DstTensor, SrcTensor, ScratchTileT>{RdmaOp::TGET, dst,       src,        scratch,
                                                                     workspace,    peer_rank, local_rank, sync_id};
}

template <typename DstTensor, typename SrcTensor, typename ScratchTileT>
inline __aicore__ RdmaRequestDescriptor<DstTensor, SrcTensor, ScratchTileT> RdmaTput(
    const DstTensor &dst, const SrcTensor &src, const ScratchTileT &scratch, __gm__ uint8_t *workspace,
    uint32_t peer_rank, uint32_t local_rank, uint32_t sync_id = 0
) {
    return RdmaRequestDescriptor<DstTensor, SrcTensor, ScratchTileT>{RdmaOp::TPUT, dst,       src,        scratch,
                                                                     workspace,    peer_rank, local_rank, sync_id};
}

namespace pto2::detail {

enum class RdmaEventRegistrationResult : int32_t {
    OK = 0,
    INVALID_EVENT = -31,
    REGISTRATION_FAILED = -32,
};

template <typename PtoAsyncEvent, typename PtoAsyncSession>
inline __aicore__ RdmaEventRegistrationResult register_rdma_async_event_status(
    AsyncCtx &ctx, const PtoAsyncEvent &event, const PtoAsyncSession &session, __gm__ uint8_t *workspace
);

template <typename PtoAsyncEvent, typename PtoAsyncSession>
inline __aicore__ bool register_rdma_async_event(
    AsyncCtx &ctx, const PtoAsyncEvent &event, const PtoAsyncSession &session, __gm__ uint8_t *workspace
);

}  // namespace pto2::detail

namespace pto2::rdma_backend {

enum class RdmaSubmitStatus : int32_t {
    OK = 0,
    BUILD_SESSION_FAILED = -30,
    INVALID_EVENT = -31,
    REGISTRATION_FAILED = -32,
    UNSUPPORTED = -33,
};

inline __aicore__ uint64_t peer_mr_base_addr(__gm__ uint8_t *workspace, uint32_t peer) {
#if defined(PTO_RDMA_SUPPORTED)
    return pto::comm::rdma::PeerMrBaseAddr(workspace, peer);
#else
    (void)workspace;
    (void)peer;
    return 0;
#endif
}

template <typename T>
inline __aicore__ __gm__ T *peer_mr_ptr(__gm__ uint8_t *workspace, uint32_t peer, uint64_t local_offset) {
    return reinterpret_cast<__gm__ T *>(peer_mr_base_addr(workspace, peer) + local_offset);
}

template <typename DstTensor, typename SrcTensor, typename ScratchTileT>
inline __aicore__ RdmaSubmitStatus
submit_rdma_request_status(AsyncCtx &ctx, RdmaRequestDescriptor<DstTensor, SrcTensor, ScratchTileT> desc) {
#if defined(PTO_RDMA_SUPPORTED)
    pto::comm::AsyncSession session;
    if (!pto::comm::BuildAsyncSession<pto::comm::DmaEngine::RDMA>(
            desc.scratch, desc.workspace, desc.local_rank, session, desc.sync_id
        )) {
        pto2::detail::defer_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
        return RdmaSubmitStatus::BUILD_SESSION_FAILED;
    }

    pto::comm::AsyncEvent event;
    if (desc.op == RdmaOp::TGET) {
        event = pto::comm::TGET_ASYNC<pto::comm::DmaEngine::RDMA>(desc.dst, desc.src, session, desc.peer_rank);
    } else {
        event = pto::comm::TPUT_ASYNC<pto::comm::DmaEngine::RDMA>(desc.dst, desc.src, session, desc.peer_rank);
    }
    const pto2::detail::RdmaEventRegistrationResult reg_result =
        pto2::detail::register_rdma_async_event_status(ctx, event, session, desc.workspace);
    if (reg_result != pto2::detail::RdmaEventRegistrationResult::OK) {
        return reg_result == pto2::detail::RdmaEventRegistrationResult::INVALID_EVENT ?
                   RdmaSubmitStatus::INVALID_EVENT :
                   RdmaSubmitStatus::REGISTRATION_FAILED;
    }
    pto2::detail::defer_flush(ctx);
    return RdmaSubmitStatus::OK;
#else
    (void)desc;
    pto2::detail::defer_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
    return RdmaSubmitStatus::UNSUPPORTED;
#endif
}

template <typename DstTensor, typename SrcTensor, typename ScratchTileT>
inline __aicore__ bool
submit_rdma_request(AsyncCtx &ctx, RdmaRequestDescriptor<DstTensor, SrcTensor, ScratchTileT> desc) {
    return submit_rdma_request_status(ctx, desc) == RdmaSubmitStatus::OK;
}

}  // namespace pto2::rdma_backend

namespace pto2::detail {

template <typename PtoAsyncEvent, typename PtoAsyncSession>
inline __aicore__ RdmaEventRegistrationResult register_rdma_async_event_status(
    AsyncCtx &ctx, const PtoAsyncEvent &event, const PtoAsyncSession &session, __gm__ uint8_t *workspace
) {
    if (!ctx.task_token.is_valid() || ctx.completion_count == nullptr || ctx.completion_entries == nullptr) {
        (void)event.Wait(session);
        return RdmaEventRegistrationResult::OK;
    }
    if (event.handle == 0) {
        return RdmaEventRegistrationResult::OK;
    }

    const uint32_t engine = static_cast<uint32_t>(event.engine);
    if (engine != static_cast<uint32_t>(::pto::comm::DmaEngine::RDMA) || workspace == nullptr) {
        defer_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
        (void)event.Wait(session);
        return RdmaEventRegistrationResult::INVALID_EVENT;
    }

    CompletionToken token{
        event.handle,
        0,
        COMPLETION_ENGINE_ROCE,
        COMPLETION_TYPE_RDMA_EVENT_HANDLE,
        reinterpret_cast<uint64_t>(workspace),
    };
    if (!register_completion_condition(ctx, token)) {
        defer_error(ctx, SIMPLER_ERROR_ASYNC_REGISTRATION_FAILED);
        (void)event.Wait(session);
        return RdmaEventRegistrationResult::REGISTRATION_FAILED;
    }
    return RdmaEventRegistrationResult::OK;
}

template <typename PtoAsyncEvent, typename PtoAsyncSession>
inline __aicore__ bool register_rdma_async_event(
    AsyncCtx &ctx, const PtoAsyncEvent &event, const PtoAsyncSession &session, __gm__ uint8_t *workspace
) {
    return register_rdma_async_event_status(ctx, event, session, workspace) == RdmaEventRegistrationResult::OK;
}

}  // namespace pto2::detail

template <typename DstTensor, typename SrcTensor, typename ScratchTileT>
inline __aicore__ bool
send_request_entry(AsyncCtx &ctx, RdmaRequestDescriptor<DstTensor, SrcTensor, ScratchTileT> desc) {
    return pto2::rdma_backend::submit_rdma_request(ctx, desc);
}

#endif  // SRC_A5_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_BACKEND_RDMA_RDMA_COMPLETION_KERNEL_H_
