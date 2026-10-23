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

#include <type_traits>

#if defined(__CPU_SIM)
#include <type_traits>
#else
#include "async_kernel_api.h"

#include <pto/comm/async_common/async_event_impl.hpp>
#if defined(PTO_URMA_SUPPORTED)
#include <pto/comm/async/urma/urma_async_intrin.hpp>
#include "common/urma_workspace_contract.h"
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

struct GlobalTensorDim {
    static constexpr int DIM_0 = 0;
    static constexpr int DIM_1 = 1;
    static constexpr int DIM_2 = 2;
    static constexpr int DIM_3 = 3;
    static constexpr int DIM_4 = 4;
};

enum class Layout : uint8_t {
    ND = 0,
    DN = 1,
};

namespace comm {
enum class DmaEngine : uint8_t {
    SDMA = 0,
    URMA = 1,
};
}  // namespace comm

}  // namespace pto
#endif

enum class UrmaOp : uint8_t {
    TGET = 0,
    TPUT = 1,
};

template <typename DstTensor, typename SrcTensor>
struct UrmaRequestDescriptor {
    UrmaOp op;
    DstTensor dst;
    SrcTensor src;
    __gm__ uint8_t *workspace;
    uint32_t remote_rank;
};

template <typename DstTensor, typename SrcTensor>
inline __aicore__ UrmaRequestDescriptor<DstTensor, SrcTensor>
UrmaTget(const DstTensor &dst, const SrcTensor &src, __gm__ uint8_t *workspace, uint32_t src_rank) {
    return UrmaRequestDescriptor<DstTensor, SrcTensor>{UrmaOp::TGET, dst, src, workspace, src_rank};
}

template <typename DstTensor, typename SrcTensor>
inline __aicore__ UrmaRequestDescriptor<DstTensor, SrcTensor>
UrmaTput(const DstTensor &dst, const SrcTensor &src, __gm__ uint8_t *workspace, uint32_t dest_rank) {
    return UrmaRequestDescriptor<DstTensor, SrcTensor>{UrmaOp::TPUT, dst, src, workspace, dest_rank};
}

namespace pto2::detail {

inline __aicore__ void defer_error(AsyncCtx &ctx, int32_t error_code);

template <typename PtoAsyncEvent, typename PtoAsyncSession>
inline __aicore__ bool register_urma_async_event(
    AsyncCtx &ctx, const PtoAsyncEvent &event, const PtoAsyncSession &session, __gm__ uint8_t *workspace
);

}  // namespace pto2::detail

namespace pto2::urma_backend {

inline constexpr uint64_t kUrmaMaxTransferBytes = 256ULL * 1024ULL * 1024ULL;

#if defined(PTO_URMA_SUPPORTED)
static_assert(kUrmaMaxTransferBytes == pto::comm::urma::kUrmaMaxWqeTransferBytes, "URMA transfer limit drift");
#endif

inline __aicore__ uint64_t peer_mr_base_addr(__gm__ uint8_t *workspace, uint32_t peer) {
#if defined(__CPU_SIM)
    (void)workspace;
    (void)peer;
    return 0;
#elif defined(PTO_URMA_SUPPORTED)
    return pto::comm::urma::UrmaPeerMrBaseAddr(workspace, peer);
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

template <typename TensorT>
inline __aicore__ uint64_t tensor_element_count(TensorT &tensor) {
    return static_cast<uint64_t>(tensor.GetShape(pto::GlobalTensorDim::DIM_0)) *
           static_cast<uint64_t>(tensor.GetShape(pto::GlobalTensorDim::DIM_1)) *
           static_cast<uint64_t>(tensor.GetShape(pto::GlobalTensorDim::DIM_2)) *
           static_cast<uint64_t>(tensor.GetShape(pto::GlobalTensorDim::DIM_3)) *
           static_cast<uint64_t>(tensor.GetShape(pto::GlobalTensorDim::DIM_4));
}

inline __aicore__ uint64_t chunk_count(uint64_t total_bytes) {
    return (total_bytes + kUrmaMaxTransferBytes - 1) / kUrmaMaxTransferBytes;
}

template <typename TensorT>
inline __aicore__ bool is_flat_contiguous_1d(TensorT &tensor) {
    const int64_t shp0 = tensor.GetShape(pto::GlobalTensorDim::DIM_0);
    const int64_t shp1 = tensor.GetShape(pto::GlobalTensorDim::DIM_1);
    const int64_t shp2 = tensor.GetShape(pto::GlobalTensorDim::DIM_2);
    const int64_t shp3 = tensor.GetShape(pto::GlobalTensorDim::DIM_3);
    const int64_t shp4 = tensor.GetShape(pto::GlobalTensorDim::DIM_4);

    const int64_t step0 = tensor.GetStride(pto::GlobalTensorDim::DIM_0);
    const int64_t step1 = tensor.GetStride(pto::GlobalTensorDim::DIM_1);
    const int64_t step2 = tensor.GetStride(pto::GlobalTensorDim::DIM_2);
    const int64_t step3 = tensor.GetStride(pto::GlobalTensorDim::DIM_3);
    const int64_t step4 = tensor.GetStride(pto::GlobalTensorDim::DIM_4);

    const bool packed_layout = (step4 == 1) && (step3 == shp4) && (step2 == shp3 * step3) && (step1 == shp2 * step2) &&
                               (step0 == shp1 * step1);
    const bool one_dim_logical = (shp0 == 1 && shp1 == 1 && shp2 == 1 && shp3 == 1);
    return packed_layout && one_dim_logical;
}

template <typename TensorT>
inline __aicore__ TensorT make_tensor_slice(TensorT &tensor, uint64_t elem_offset, uint64_t elem_count) {
    using ShapeT = typename TensorT::Shape;
    using StrideT = typename TensorT::Stride;
    ShapeT shape(
        tensor.GetShape(pto::GlobalTensorDim::DIM_0), tensor.GetShape(pto::GlobalTensorDim::DIM_1),
        tensor.GetShape(pto::GlobalTensorDim::DIM_2), tensor.GetShape(pto::GlobalTensorDim::DIM_3),
        static_cast<int64_t>(elem_count)
    );
    StrideT stride(
        tensor.GetStride(pto::GlobalTensorDim::DIM_0), tensor.GetStride(pto::GlobalTensorDim::DIM_1),
        tensor.GetStride(pto::GlobalTensorDim::DIM_2), tensor.GetStride(pto::GlobalTensorDim::DIM_3),
        tensor.GetStride(pto::GlobalTensorDim::DIM_4)
    );
    return TensorT(tensor.data() + elem_offset, shape, stride);
}

template <typename DstTensor, typename SrcTensor>
inline __aicore__ bool submit_urma_request_once(AsyncCtx &ctx, UrmaRequestDescriptor<DstTensor, SrcTensor> desc) {
#if defined(PTO_URMA_SUPPORTED)
    if (!ctx.task_token.is_valid() || !ctx.completion_count || !ctx.completion_entries ||
        *ctx.completion_count >= ctx.completion_capacity || !desc.workspace) {
        pto2::detail::defer_error(ctx, SIMPLER_ERROR_ASYNC_REGISTRATION_FAILED);
        return false;
    }
    auto *info = reinterpret_cast<__gm__ pto::comm::urma::UrmaInfo *>(desc.workspace);
    if (info->layout != pto::comm::urma::UrmaLayout::PER_PEER || info->rowsPerPeer != 1 || info->jettiesPerCore != 1 ||
        info->jettyCount != info->rankCount || desc.remote_rank >= info->rankCount || !info->sqPtr || !info->scqPtr) {
        pto2::detail::defer_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
        return false;
    }
    auto *wq = reinterpret_cast<__gm__ pto::comm::urma::UrmaWQCtx *>(info->sqPtr) + desc.remote_rank;
    auto *cq = reinterpret_cast<__gm__ pto::comm::urma::UrmaCqCtx *>(info->scqPtr) + desc.remote_rank;
    if (!wq->bufAddr || !wq->dbAddr || !wq->headAddr || !wq->tailAddr || !cq->tailAddr || !cq->bufAddr || !cq->dbAddr ||
        cq->cqeShiftSize != 6 || wq->wqeShiftSize != 6 || !wq->depth || !cq->depth || wq->depth >= (1u << 31) ||
        cq->depth >= (1u << 31) || (wq->depth & (wq->depth - 1)) || (cq->depth & (cq->depth - 1))) {
        pto2::detail::defer_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
        return false;
    }
    // The scheduler is the only CQ consumer; PTO's full-ring synchronous drain
    // is unreachable for accepted submissions.
    const uint32_t head = ld_dev(reinterpret_cast<__gm__ uint32_t *>(wq->headAddr), 0);
    if (!pto::comm::urma::detail::UrmaRingHasRoom(wq, cq, head, wq->submittedWqeCount, 1, 1)) {
        pto2::detail::defer_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
        return false;
    }
    pto::comm::AsyncSession session;
    if (!pto::comm::BuildAsyncSession<pto::comm::DmaEngine::URMA>(desc.workspace, desc.remote_rank, session)) {
        pto2::detail::defer_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
        return false;
    }

    pto::comm::AsyncEvent event;
    if (desc.op == UrmaOp::TGET) {
        event = pto::comm::TGET_ASYNC<pto::comm::DmaEngine::URMA>(desc.dst, desc.src, session);
    } else {
        event = pto::comm::TPUT_ASYNC<pto::comm::DmaEngine::URMA>(desc.dst, desc.src, session);
    }
    if (!pto2::detail::register_urma_async_event(ctx, event, session, desc.workspace)) {
        return false;
    }
    pto2::detail::defer_flush(ctx);
    return true;
#else
    (void)desc;
    pto2::detail::defer_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
    return false;
#endif
}

template <typename DstTensor, typename SrcTensor>
inline __aicore__ bool submit_chunked_urma_request(AsyncCtx &ctx, UrmaRequestDescriptor<DstTensor, SrcTensor> desc) {
    using RawDType = typename DstTensor::RawDType;
    static_assert(std::is_same_v<RawDType, typename SrcTensor::RawDType>, "URMA transfer requires matching dtypes");

    if (!is_flat_contiguous_1d(desc.dst) || !is_flat_contiguous_1d(desc.src)) {
        pto2::detail::defer_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
        return false;
    }

    const uint64_t elem_count = tensor_element_count(desc.dst);
    if (elem_count != tensor_element_count(desc.src)) {
        pto2::detail::defer_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
        return false;
    }
    if (elem_count == 0) return true;
    const uint64_t total_bytes = elem_count * sizeof(RawDType);
    const uint64_t max_bytes = kUrmaMaxTransferBytes;
    if (total_bytes <= max_bytes) {
        return submit_urma_request_once(ctx, desc);
    }

    const uint64_t chunk_elems = max_bytes / sizeof(RawDType);
    if (chunk_elems == 0 || (max_bytes % sizeof(RawDType)) != 0) {
        pto2::detail::defer_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
        return false;
    }

    uint64_t offset = 0;
    while (offset < elem_count) {
        const uint64_t remaining = elem_count - offset;
        const uint64_t current_elems = (remaining < chunk_elems) ? remaining : chunk_elems;
        auto chunk_desc = desc;
        chunk_desc.dst = make_tensor_slice(desc.dst, offset, current_elems);
        chunk_desc.src = make_tensor_slice(desc.src, offset, current_elems);
        if (!submit_urma_request_once(ctx, chunk_desc)) {
            return false;
        }
        offset += current_elems;
    }
    return true;
}

}  // namespace pto2::urma_backend

#if defined(__CPU_SIM)
inline __aicore__ AsyncCtx get_async_ctx(__gm__ int64_t *args) {
    __gm__ LocalContext *lc =
        reinterpret_cast<__gm__ LocalContext *>(static_cast<uintptr_t>(args[PAYLOAD_LOCAL_CONTEXT_INDEX]));
    AsyncCtx ctx{};
    ctx.completion_count = lc->async_ctx.completion_count;
    ctx.completion_error_code = lc->async_ctx.completion_error_code;
    ctx.completion_entries = lc->async_ctx.completion_entries;
    ctx.completion_capacity = lc->async_ctx.completion_capacity;
    TaskId::assign(ctx.task_token, lc->async_ctx.task_token);
    return ctx;
}

inline __aicore__ bool register_completion_condition(AsyncCtx &ctx, const CompletionToken &token) {
    if (!ctx.task_token.is_valid() || ctx.completion_count == nullptr || ctx.completion_entries == nullptr) {
        return false;
    }

    uint32_t idx = *ctx.completion_count;
    if (idx >= ctx.completion_capacity) {
        if (ctx.completion_error_code != nullptr) {
            *ctx.completion_error_code = SIMPLER_ERROR_ASYNC_WAIT_OVERFLOW;
        }
        return false;
    }

    volatile __gm__ DeferredCompletionEntry *slot = &ctx.completion_entries[idx];
    slot->addr = token.addr;
    slot->backend_cookie = token.backend_cookie;
    slot->expected_value = token.expected_value;
    slot->engine = token.engine;
    slot->completion_type = token.completion_type;
    slot->_pad = 0;
    *ctx.completion_count = idx + 1;
    return true;
}
#endif

namespace pto2::detail {

#if defined(__CPU_SIM)
inline __aicore__ void defer_load_slab(AsyncCtx & /*ctx*/) { __asm__ __volatile__("" ::: "memory"); }

inline __aicore__ void defer_error(AsyncCtx &ctx, int32_t error_code) {
    if (ctx.task_token.is_valid() && ctx.completion_error_code != nullptr) {
        *ctx.completion_error_code = error_code;
    }
}

inline __aicore__ void defer_flush(AsyncCtx & /*ctx*/) { __asm__ __volatile__("" ::: "memory"); }
#endif

template <typename PtoAsyncEvent, typename PtoAsyncSession>
inline __aicore__ bool register_urma_async_event(
    AsyncCtx &ctx, const PtoAsyncEvent &event, const PtoAsyncSession &session, __gm__ uint8_t *workspace
) {
    if (!ctx.task_token.is_valid() || ctx.completion_count == nullptr || ctx.completion_entries == nullptr ||
        workspace == nullptr || session.qpIdxBase != 0 || session.qpCount != 1 ||
        static_cast<uint32_t>(event.engine) != static_cast<uint32_t>(::pto::comm::DmaEngine::URMA) ||
        event.urmaJettyBase != 0 || event.urmaJettyCount != 1 ||
        static_cast<uint32_t>(event.handle) != event.urmaTargetBbPerJetty[0]) {
        defer_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
        return false;
    }
    CompletionToken token{
        event.handle,
        event.urmaTargetCqePerJetty[0],
        COMPLETION_ENGINE_URMA,
        COMPLETION_TYPE_URMA_EVENT_HANDLE,
        reinterpret_cast<uint64_t>(workspace),
    };
    if (!register_completion_condition(ctx, token)) {
        defer_error(ctx, SIMPLER_ERROR_ASYNC_REGISTRATION_FAILED);
        return false;
    }
    return true;
}

}  // namespace pto2::detail

template <typename DstTensor, typename SrcTensor>
inline __aicore__ bool send_request_entry(AsyncCtx &ctx, UrmaRequestDescriptor<DstTensor, SrcTensor> desc) {
    const bool submitted = pto2::urma_backend::submit_chunked_urma_request(ctx, desc);
    if (!submitted) pto2::detail::defer_flush(ctx);
    return submitted;
}
