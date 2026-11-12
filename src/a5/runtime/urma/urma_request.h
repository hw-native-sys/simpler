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

#if !defined(__CPU_SIM)
#include <pto/comm/async_common/async_event_impl.hpp>
#if defined(PTO_URMA_SUPPORTED)
#include <pto/comm/async/urma/urma_async_intrin.hpp>
#endif
#endif
#include <pto/comm/async/urma/urma_types.hpp>

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

namespace simpler::urma {

enum class RequestError : uint8_t { INVALID, REGISTRATION_FAILED };

struct CompletionTarget {
    __gm__ uint8_t *workspace;
    uint32_t queue_row;
    uint32_t target_bb;
    uint32_t target_cqe;
};

template <typename Runtime, typename Event, typename Session>
inline __aicore__ bool
register_urma_async_event(Runtime &runtime, const Event &event, const Session &session, __gm__ uint8_t *workspace);

using pto::comm::urma::UrmaCqCtx;
using pto::comm::urma::UrmaInfo;
using pto::comm::urma::UrmaLayout;
using pto::comm::urma::UrmaWQCtx;

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
    return total_bytes == 0 ? 0 : 1 + (total_bytes - 1) / pto::comm::urma::kUrmaMaxWqeTransferBytes;
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

#if defined(PTO_URMA_SUPPORTED)
inline __aicore__ bool
build_urma_session(uint32_t sender_id, __gm__ uint8_t *workspace, uint32_t peer, pto::comm::AsyncSession &session) {
    session = pto::comm::AsyncSession{};
    // Activate both URMA union alternatives without consulting the CCE block identity.
    pto::comm::detail::ActivateEngineState<pto::comm::DmaEngine::URMA>(session);
    session.valid = false;
    session.contextGm = workspace;
    session.destRankId = peer;
    if (workspace == nullptr || sender_id == UINT32_MAX) return false;
    auto *info = reinterpret_cast<__gm__ UrmaInfo *>(workspace);
    const uint32_t per_sender = info->jettiesPerCore;
    if (peer >= info->rankCount || per_sender == 0 || per_sender > pto::comm::urma::kUrmaMaxJettiesPerCore)
        return false;
    uint32_t bound = 0;
    if (info->layout == UrmaLayout::PER_PEER) {
        if (static_cast<uint64_t>(info->rankCount) * info->rowsPerPeer != info->jettyCount) return false;
        bound = info->rowsPerPeer;
    } else if (info->layout == UrmaLayout::SHARED_POOL) {
        if (info->rowsPerPeer != 1) return false;
        bound = info->jettyCount;
    } else {
        return false;
    }
    if (bound < per_sender || bound % per_sender != 0 || sender_id >= bound / per_sender) return false;
    session.cfg.urma.qpCount = per_sender;
    session.qpIdxBase = sender_id * per_sender;
    session.valid = true;
    return true;
}

template <typename TensorT>
inline __aicore__ auto make_tensor_slice(TensorT &tensor, uint64_t elem_offset, uint64_t elem_count) {
    using Shape = pto::Shape<1, 1, 1, 1, pto::DYNAMIC>;
    using Stride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1>;
    using Slice = pto::GlobalTensor<typename TensorT::RawDType, Shape, Stride>;
    return Slice(
        tensor.data() + elem_offset, Shape(1, 1, 1, 1, elem_count),
        Stride(elem_count, elem_count, elem_count, elem_count, 1)
    );
}
#endif

inline __aicore__ bool queue_row_for(__gm__ uint8_t *workspace, uint32_t peer, uint32_t jetty, uint32_t &row) {
    if (workspace == nullptr) return false;
    auto *info = reinterpret_cast<__gm__ UrmaInfo *>(workspace);
    if (peer >= info->rankCount || info->jettiesPerCore == 0 ||
        info->jettiesPerCore > pto::comm::urma::kUrmaMaxJettiesPerCore || info->rowsPerPeer == 0)
        return false;
    if (info->layout == UrmaLayout::PER_PEER) {
        if (jetty >= info->rowsPerPeer ||
            static_cast<uint64_t>(info->rankCount) * info->rowsPerPeer != info->jettyCount)
            return false;
        row = peer * info->rowsPerPeer + jetty;
    } else if (info->layout == UrmaLayout::SHARED_POOL) {
        row = jetty;
    } else {
        return false;
    }
    return row < info->jettyCount && info->sqPtr != 0 && info->scqPtr != 0;
}

inline __aicore__ uint32_t load_urma_counter(uint64_t addr) {
    auto *ptr = reinterpret_cast<__gm__ uint32_t *>(addr);
#if defined(__CPU_SIM)
    return *reinterpret_cast<volatile uint32_t *>(ptr);
#else
    return ld_dev(ptr, 0);
#endif
}

template <typename Runtime, typename Session>
inline __aicore__ bool preflight_urma_request(Runtime &runtime, const Session &session, uint64_t chunks) {
    const uint32_t jetties = session.cfg.urma.qpCount;
    if (jetties == 0 || jetties > pto::comm::urma::kUrmaMaxJettiesPerCore || chunks > UINT32_MAX ||
        !runtime.capacity_available(chunks)) {
        runtime.set_error(RequestError::REGISTRATION_FAILED);
        return false;
    }
    if (!session.valid || session.engine != pto::comm::DmaEngine::URMA || session.contextGm == nullptr) {
        runtime.set_error(RequestError::INVALID);
        return false;
    }
    auto *info = reinterpret_cast<__gm__ UrmaInfo *>(session.contextGm);
    const bool per_peer = info->layout == UrmaLayout::PER_PEER;
    // At most UINT32_MAX chunks and eight jetties keep the product below 2^35.
    if (!per_peer && !runtime.capacity_available(chunks * jetties)) {
        runtime.set_error(RequestError::REGISTRATION_FAILED);
        return false;
    }
    if (per_peer && info->railCycleLen != 0) {
        runtime.set_error(RequestError::INVALID);
        return false;
    }
    // Producers are serialized per sender; the runtime completion observer alone advances tails.
    for (uint32_t j = 0; j < jetties; ++j) {
        const uint64_t required = per_peer ? chunks / jetties + (j < chunks % jetties ? 1 : 0) : chunks;
        uint32_t row = 0;
        if (session.qpIdxBase > UINT32_MAX - j ||
            !queue_row_for(session.contextGm, session.destRankId, session.qpIdxBase + j, row)) {
            runtime.set_error(RequestError::INVALID);
            return false;
        }
        auto *wq = reinterpret_cast<__gm__ UrmaWQCtx *>(info->sqPtr) + row;
        auto *cq = reinterpret_cast<__gm__ UrmaCqCtx *>(info->scqPtr) + row;
        if (!wq->bufAddr || !wq->headAddr || !wq->tailAddr || !wq->dbAddr || !cq->bufAddr || !cq->tailAddr ||
            !cq->dbAddr || wq->wqeShiftSize != 6 || cq->cqeShiftSize != 6 || wq->depth == 0 || cq->depth == 0 ||
            (wq->depth & (wq->depth - 1)) != 0 || (cq->depth & (cq->depth - 1)) != 0 || wq->depth >= (1u << 31) ||
            cq->depth >= (1u << 31) || required > wq->depth || required > cq->depth) {
            runtime.set_error(RequestError::INVALID);
            return false;
        }
        const uint32_t head = load_urma_counter(wq->headAddr);
        const uint32_t bb_tail = load_urma_counter(wq->tailAddr);
        const uint32_t cqe_tail = load_urma_counter(cq->tailAddr);
        if (static_cast<uint64_t>(head - bb_tail) + required > wq->depth ||
            static_cast<uint64_t>(wq->submittedWqeCount - cqe_tail) + required > cq->depth) {
            runtime.set_error(RequestError::INVALID);
            return false;
        }
    }
    return true;
}

template <typename Runtime, typename DstTensor, typename SrcTensor, typename Session>
inline __aicore__ bool
submit_urma_request_once(Runtime &runtime, UrmaRequestDescriptor<DstTensor, SrcTensor> desc, const Session &session) {
#if defined(PTO_URMA_SUPPORTED)
    pto::comm::AsyncEvent event;
    if (desc.op == UrmaOp::TGET) {
        event = pto::comm::TGET_ASYNC<pto::comm::DmaEngine::URMA>(desc.dst, desc.src, session);
    } else {
        event = pto::comm::TPUT_ASYNC<pto::comm::DmaEngine::URMA>(desc.dst, desc.src, session);
    }
    if (event.urmaJettyCount == 0 && !event.postFailed) {
        runtime.set_error(RequestError::INVALID);
        return false;
    }
    return register_urma_async_event(runtime, event, session, desc.workspace);
#else
    (void)desc;
    (void)session;
    runtime.set_error(RequestError::INVALID);
    return false;
#endif
}

template <typename Runtime, typename DstTensor, typename SrcTensor>
inline __aicore__ bool submit_chunked_urma_request(Runtime &runtime, UrmaRequestDescriptor<DstTensor, SrcTensor> desc) {
    using RawDType = typename DstTensor::RawDType;
    static_assert(std::is_same_v<RawDType, typename SrcTensor::RawDType>, "URMA transfer requires matching dtypes");
    if (!is_flat_contiguous_1d(desc.dst) || !is_flat_contiguous_1d(desc.src) ||
        tensor_element_count(desc.dst) != tensor_element_count(desc.src)) {
        runtime.set_error(RequestError::INVALID);
        return false;
    }
    const uint64_t elem_count = tensor_element_count(desc.dst);
    if (elem_count == 0) return true;
    if (elem_count > UINT64_MAX / sizeof(RawDType)) {
        runtime.set_error(RequestError::INVALID);
        return false;
    }
#if defined(PTO_URMA_SUPPORTED)
    pto::comm::AsyncSession session;
    if (!build_urma_session(runtime.sender_id(), desc.workspace, desc.remote_rank, session)) {
        runtime.set_error(RequestError::INVALID);
        return false;
    }
    if (!preflight_urma_request(runtime, session, chunk_count(elem_count * sizeof(RawDType)))) return false;
    const bool per_peer = reinterpret_cast<__gm__ UrmaInfo *>(desc.workspace)->layout == UrmaLayout::PER_PEER;
    const uint64_t chunks = chunk_count(elem_count * sizeof(RawDType));
    const uint64_t capacity = per_peer ? chunks : chunks * session.cfg.urma.qpCount;
    if (!runtime.reserve(capacity)) {
        runtime.set_error(RequestError::REGISTRATION_FAILED);
        return false;
    }
    const uint32_t sender_base = session.qpIdxBase;
    const uint32_t jetties = session.cfg.urma.qpCount;
    if (per_peer) session.cfg.urma.qpCount = 1;
    uint32_t jetty = 0;
    const uint64_t chunk_elems = pto::comm::urma::kUrmaMaxWqeTransferBytes / sizeof(RawDType);
    for (uint64_t offset = 0; offset < elem_count;) {
        const uint64_t remaining = elem_count - offset;
        const uint64_t current_elems = remaining < chunk_elems ? remaining : chunk_elems;
        auto dst_slice = make_tensor_slice(desc.dst, offset, current_elems);
        auto src_slice = make_tensor_slice(desc.src, offset, current_elems);
        UrmaRequestDescriptor<decltype(dst_slice), decltype(src_slice)> chunk_desc{
            desc.op, dst_slice, src_slice, desc.workspace, desc.remote_rank
        };
        // Each PER_PEER call targets one queue; the non-rail ISA cursor restarts on every call.
        if (per_peer) session.qpIdxBase = sender_base + jetty;
        if (!submit_urma_request_once(runtime, chunk_desc, session)) {
            runtime.finish();
            return false;
        }
        jetty = (jetty + 1) % jetties;
        offset += current_elems;
    }
    return runtime.finish();
#else
    runtime.set_error(RequestError::INVALID);
    return false;
#endif
}

template <typename Runtime, typename PtoAsyncEvent, typename PtoAsyncSession>
inline __aicore__ bool register_urma_async_event(
    Runtime &runtime, const PtoAsyncEvent &event, const PtoAsyncSession &session, __gm__ uint8_t *workspace
) {
    const uint32_t count = event.urmaJettyCount;
    if (count == 0) {
        if (event.postFailed || event.handle != 0) {
            runtime.set_error(RequestError::INVALID);
            return false;
        }
        return true;
    }
    if (!session.valid || session.contextGm != workspace || workspace == nullptr ||
        static_cast<uint32_t>(event.engine) != static_cast<uint32_t>(::pto::comm::DmaEngine::URMA) ||
        event.engine != session.engine || count > pto::comm::urma::kUrmaMaxJettiesPerCore ||
        event.urmaJettyBase < session.qpIdxBase ||
        static_cast<uint32_t>(event.urmaJettyBase) - session.qpIdxBase + count > session.cfg.urma.qpCount) {
        runtime.set_error(RequestError::INVALID);
        return false;
    }
    if (!runtime.capacity_available(count)) {
        runtime.set_error(RequestError::REGISTRATION_FAILED);
        return false;
    }
    uint32_t rows[pto::comm::urma::kUrmaMaxJettiesPerCore];
    for (uint32_t j = 0; j < count; ++j) {
        if (!queue_row_for(workspace, session.destRankId, event.urmaJettyBase + j, rows[j])) {
            runtime.set_error(RequestError::INVALID);
            return false;
        }
    }
    for (uint32_t j = 0; j < count; ++j) {
        CompletionTarget target{workspace, rows[j], event.urmaTargetBbPerJetty[j], event.urmaTargetCqePerJetty[j]};
        if (!runtime.register_target(target)) {
            runtime.set_error(RequestError::REGISTRATION_FAILED);
            return false;
        }
    }
    if (event.postFailed) {
        runtime.set_error(RequestError::INVALID);
        return false;
    }
    return true;
}

}  // namespace simpler::urma
