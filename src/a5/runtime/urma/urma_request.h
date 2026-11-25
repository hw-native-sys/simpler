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
#include <pto/comm/pto_comm_inst.hpp>
#endif
#endif
#include <pto/comm/async/urma/urma_types.hpp>

#include "platform_comm/comm_context.h"

#if defined(PTO_URMA_SUPPORTED)
#include "intrinsic.h"
#include "platform_comm/urma_workspace_layout.h"
#endif

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
    const __gm__ CommWorkspace *domain{};
};

#if defined(PTO_URMA_SUPPORTED)
// The remote descriptor identifies the peer. The local segment is a source for
// Gather PUT and a destination for Scatter GET.
template <typename Remote, typename Local>
struct UrmaSegmentedRequest {
    __gm__ const Remote *remote;
    uint32_t remote_count;
    __gm__ const Local *local;
    uint32_t local_count;
};
#endif

template <typename DstTensor, typename SrcTensor>
inline __aicore__ UrmaRequestDescriptor<DstTensor, SrcTensor>
UrmaTget(const DstTensor &dst, const SrcTensor &src, __gm__ uint8_t *workspace, uint32_t src_rank) {
    return UrmaRequestDescriptor<DstTensor, SrcTensor>{UrmaOp::TGET, dst, src, workspace, src_rank};
}

template <typename DstTensor, typename SrcTensor>
inline __aicore__ auto
UrmaTget(const DstTensor &dst, const SrcTensor &src, const __gm__ CommWorkspace &urma, uint32_t src_rank) {
    auto *root = reinterpret_cast<__gm__ uint8_t *>(urma.addr);
    return UrmaRequestDescriptor<DstTensor, SrcTensor>{UrmaOp::TGET, dst, src, root, src_rank, &urma};
}

#if defined(PTO_URMA_SUPPORTED)
inline __aicore__ auto UrmaTget(
    __gm__ const pto::comm::ScatterSegment *segments, uint32_t segment_count,
    __gm__ const pto::comm::ScatterSource *sources, uint32_t source_count
) {
    return UrmaSegmentedRequest<pto::comm::ScatterSource, pto::comm::ScatterSegment>{
        sources, source_count, segments, segment_count
    };
}
#endif

template <typename DstTensor, typename SrcTensor>
inline __aicore__ UrmaRequestDescriptor<DstTensor, SrcTensor>
UrmaTput(const DstTensor &dst, const SrcTensor &src, __gm__ uint8_t *workspace, uint32_t dest_rank) {
    return UrmaRequestDescriptor<DstTensor, SrcTensor>{UrmaOp::TPUT, dst, src, workspace, dest_rank};
}

template <typename DstTensor, typename SrcTensor>
inline __aicore__ auto
UrmaTput(const DstTensor &dst, const SrcTensor &src, const __gm__ CommWorkspace &urma, uint32_t dest_rank) {
    auto *root = reinterpret_cast<__gm__ uint8_t *>(urma.addr);
    return UrmaRequestDescriptor<DstTensor, SrcTensor>{UrmaOp::TPUT, dst, src, root, dest_rank, &urma};
}

#if defined(PTO_URMA_SUPPORTED)
inline __aicore__ auto UrmaTput(
    __gm__ const pto::comm::GatherDestination *destinations, uint32_t destination_count,
    __gm__ const pto::comm::GatherSegment *segments, uint32_t segment_count
) {
    return UrmaSegmentedRequest<pto::comm::GatherDestination, pto::comm::GatherSegment>{
        destinations, destination_count, segments, segment_count
    };
}
#endif

namespace simpler::urma {

enum class RequestError : uint8_t { INVALID, REGISTRATION_FAILED };

struct CompletionTarget {
    __gm__ uint8_t *workspace;
    uint32_t queue_row;
    uint32_t target_bb;
    uint32_t target_cqe;
};

template <typename Runtime, typename Event, typename Session>
inline __aicore__ bool register_urma_async_event(
    Runtime &runtime, const Event &event, const Session &session, __gm__ uint8_t *workspace, uint32_t peer
);

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

struct UrmaSession;

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

// The ISA reads the sender header. AICPU completion tokens retain the domain root.
struct UrmaSession : pto::comm::AsyncSession {
    __gm__ uint8_t *domain_workspace{nullptr};

    __aicore__ UrmaSession() = default;
    UrmaSession(const UrmaSession &) = delete;
    UrmaSession &operator=(const UrmaSession &) = delete;
};

namespace detail {
inline __aicore__ bool
bind_urma_session(uint32_t sender_id, const __gm__ CommWorkspace &urma, UrmaSession &session, uint32_t peer) {
    const uint64_t addr = urma.addr;
    const uint64_t size = urma.size;
    session.valid = false;
    session.domain_workspace = nullptr;
    auto *workspace = reinterpret_cast<__gm__ uint8_t *>(addr);
    if (size < sizeof(UrmaInfo) || addr % CHIP_ALIGN_SIZE != 0 ||
        !build_urma_session(sender_id, workspace, peer, session))
        return false;

    auto *info = reinterpret_cast<__gm__ const UrmaInfo *>(workspace);
    const uint32_t block = static_cast<uint32_t>(get_block_idx());
    const uint64_t stride = info->gatherRuntimeStride;
    if (info->gatherRuntimeBase == 0 || stride == 0) {
        session.valid = false;
        return false;
    }
    if (block != sender_id) {
        const uint64_t offset = sender_view_offset(size, sender_id);
        if (offset > UINT64_MAX - addr || sizeof(UrmaInfo) > UINT64_MAX - addr - offset) {
            session.valid = false;
            return false;
        }
        auto *view = reinterpret_cast<__gm__ UrmaInfo *>(addr + offset);
        // ISA adds block * stride. The slot is private to this physical sender
        // and shared by all of its sessions in this domain.
        const uint64_t base = info->gatherRuntimeBase + stride * sender_id - stride * block;
        if (view->gatherRuntimeBase != base) {
            __builtin_memcpy(view, info, sizeof(UrmaInfo));
            view->gatherRuntimeBase = base;
#if !defined(__CPU_SIM)
            pto::comm::urma::DcciCachelines(reinterpret_cast<__gm__ uint8_t *>(view), sizeof(UrmaInfo));
            dsb(DSB_DDR);
#endif
        }
        session.contextGm = reinterpret_cast<__gm__ uint8_t *>(view);
    }
    session.cfg.urma.cqePolicy = pto::comm::urma::UrmaCqePolicy::OneCqe;
    session.domain_workspace = workspace;
    return true;
}

}  // namespace detail

inline __aicore__ bool
build_urma_session(const AsyncCtx &ctx, const __gm__ CommWorkspace &urma, UrmaSession &session, uint32_t peer = 0) {
    return detail::bind_urma_session(ctx.urma_sender_id, urma, session, peer);
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

inline __aicore__ bool urma_queue_has_room(__gm__ UrmaInfo *info, uint32_t row, uint64_t required) {
    auto *wq = reinterpret_cast<__gm__ UrmaWQCtx *>(info->sqPtr) + row;
    auto *cq = reinterpret_cast<__gm__ UrmaCqCtx *>(info->scqPtr) + row;
    if (!wq->bufAddr || !wq->headAddr || !wq->tailAddr || !wq->dbAddr || !cq->bufAddr || !cq->tailAddr || !cq->dbAddr ||
        wq->wqeShiftSize != 6 || cq->cqeShiftSize != 6 || wq->depth == 0 || cq->depth == 0 ||
        (wq->depth & (wq->depth - 1)) != 0 || (cq->depth & (cq->depth - 1)) != 0 || wq->depth >= (1u << 31) ||
        cq->depth >= (1u << 31) || required > wq->depth || required > cq->depth) {
        return false;
    }
    const uint32_t head = load_urma_counter(wq->headAddr);
    const uint32_t bb_tail = load_urma_counter(wq->tailAddr);
    const uint32_t cqe_tail = load_urma_counter(cq->tailAddr);
    if (static_cast<uint64_t>(head - bb_tail) + required > wq->depth ||
        static_cast<uint64_t>(wq->submittedWqeCount - cqe_tail) + required > cq->depth) {
        return false;
    }
    return true;
}

// Number of WQE sequence numbers in [0, end) selecting this jetty.
inline __aicore__ uint64_t jetty_prefix_count(uint64_t end, uint32_t jetties, uint32_t jetty) {
    return end / jetties + (jetty < end % jetties);
}

template <typename Runtime, typename Session>
inline __aicore__ bool
preflight_urma_request(Runtime &runtime, const Session &session, uint64_t chunks, uint32_t peer) {
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
    // The ISA round-robin cursor wraps at 2^32, including for non-power-of-two widths.
    const uint64_t first = session.rt.urma.nextWqeSeq;
    const uint64_t end = first + chunks;
    constexpr uint64_t wrap = uint64_t{1} << 32;
    const uint64_t split = end < wrap ? end : wrap;
    for (uint32_t j = 0; j < jetties; ++j) {
        const uint64_t required = per_peer ?
                                      jetty_prefix_count(split, jetties, j) - jetty_prefix_count(first, jetties, j) +
                                          jetty_prefix_count(end - split, jetties, j) :
                                      chunks;
        uint32_t row = 0;
        if (session.qpIdxBase > UINT32_MAX - j || !queue_row_for(session.contextGm, peer, session.qpIdxBase + j, row)) {
            runtime.set_error(RequestError::INVALID);
            return false;
        }
        if (!urma_queue_has_room(info, row, required)) {
            runtime.set_error(RequestError::INVALID);
            return false;
        }
    }
    return true;
}

#if defined(PTO_URMA_SUPPORTED)
#if !defined(__CPU_SIM)
template <typename Remote, typename Local>
inline __aicore__ pto::comm::AsyncEvent
post_urma_segment(__gm__ const Remote *remote, __gm__ const Local *local, pto::comm::AsyncSession &session) {
    using namespace pto::comm;
    if constexpr (std::is_same_v<Remote, GatherDestination>) {
        return TPUT_ASYNC<DmaEngine::URMA>(remote, 1, local, 1, session);
    } else {
        static_assert(std::is_same_v<Remote, ScatterSource>);
        return TGET_ASYNC<DmaEngine::URMA>(local, 1, remote, 1, session);
    }
}
#endif

inline __aicore__ bool can_submit(const UrmaSession &session, uint32_t sender) {
    if (!session.valid || session.engine != pto::comm::DmaEngine::URMA || session.domain_workspace == nullptr ||
        session.contextGm == nullptr || session.submitMode != pto::comm::AsyncSubmitMode::IMMEDIATE ||
        session.rt.urma.batchWqeCount != 0 || session.cfg.urma.cqePolicy != pto::comm::urma::UrmaCqePolicy::OneCqe)
        return false;
    const uint32_t width = reinterpret_cast<__gm__ const UrmaInfo *>(session.contextGm)->jettiesPerCore;
    return width != 0 && width == session.cfg.urma.qpCount &&
           static_cast<uint64_t>(sender) * width == session.qpIdxBase;
}

template <typename Runtime, typename Remote, typename Local>
inline __aicore__ bool
submit_segmented_request(Runtime &runtime, UrmaSegmentedRequest<Remote, Local> desc, UrmaSession &session) {
    if (!can_submit(session, runtime.sender_id()) || desc.remote_count != 1 || desc.local_count != 1 ||
        desc.remote == nullptr || desc.local == nullptr) {
        runtime.set_error(RequestError::INVALID);
        return false;
    }
    const uint32_t peer = load_urma_counter(reinterpret_cast<uint64_t>(&desc.remote->peer));
    const uint32_t segments = load_urma_counter(reinterpret_cast<uint64_t>(&desc.remote->segCount));
    auto *info = reinterpret_cast<__gm__ UrmaInfo *>(session.contextGm);
    const uint32_t width = info->jettiesPerCore;
    if (segments != 1) {
        runtime.set_error(RequestError::INVALID);
        return false;
    }
    // A single segment produces one SQE on the session's next round-robin queue.
    // Check ring space before posting so the ISA never needs to drain the CQ
    // while the AICPU may still own earlier completions on this queue.
    const uint32_t jetty = session.qpIdxBase + session.rt.urma.nextWqeSeq % width;
    uint32_t row = 0;
    if (jetty < session.qpIdxBase || !queue_row_for(session.contextGm, peer, jetty, row) ||
        !urma_queue_has_room(info, row, 1)) {
        runtime.set_error(RequestError::INVALID);
        return false;
    }
    if (!runtime.reserve(1)) {
        runtime.set_error(RequestError::REGISTRATION_FAILED);
        return false;
    }

    const auto event = post_urma_segment(desc.remote, desc.local, session);
    if (event.postFailed || event.urmaJettyCount != pto::comm::urma::kUrmaGatherInlineEvent ||
        event.urmaJettyBase != jetty) {
        runtime.set_error(RequestError::INVALID);
        return false;
    }
    // In an inline descriptor event the per-jetty CQ field is the *begin*
    // sequence. ISA waits on urmaTargetCqe and the end BB sequence.
    return runtime.register_target({session.domain_workspace, row, event.urmaTargetBbPerJetty[0], event.urmaTargetCqe});
}
#endif

template <typename Runtime, typename DstTensor, typename SrcTensor, typename Session>
inline __aicore__ bool
submit_urma_request_once(Runtime &runtime, UrmaRequestDescriptor<DstTensor, SrcTensor> desc, const Session &session) {
#if defined(PTO_URMA_SUPPORTED)
    pto::comm::AsyncEvent event;
    if (desc.op == UrmaOp::TGET) {
        event = pto::comm::TGET_ASYNC<pto::comm::DmaEngine::URMA>(desc.dst, desc.src, session, desc.remote_rank);
    } else {
        event = pto::comm::TPUT_ASYNC<pto::comm::DmaEngine::URMA>(desc.dst, desc.src, session, desc.remote_rank);
    }
    if (event.urmaJettyCount == 0 && !event.postFailed) {
        runtime.set_error(RequestError::INVALID);
        return false;
    }
    return register_urma_async_event(runtime, event, session, desc.workspace, desc.remote_rank);
#else
    (void)desc;
    (void)session;
    runtime.set_error(RequestError::INVALID);
    return false;
#endif
}

#if defined(PTO_URMA_SUPPORTED)
template <typename Runtime, typename DstTensor, typename SrcTensor, typename Session>
inline __aicore__ bool submit_tensor_chunks(
    Runtime &runtime, UrmaRequestDescriptor<DstTensor, SrcTensor> desc, Session &session, uint64_t elem_count
) {
    using RawDType = typename DstTensor::RawDType;
    const bool per_peer = reinterpret_cast<__gm__ UrmaInfo *>(desc.workspace)->layout == UrmaLayout::PER_PEER;
    const uint64_t chunks = chunk_count(elem_count * sizeof(RawDType));
    if (!preflight_urma_request(runtime, session, chunks, desc.remote_rank)) return false;
    const uint64_t capacity = per_peer ? chunks : chunks * session.cfg.urma.qpCount;
    // A native Final queued by another session on this sender must never be
    // published using a narrowed queue group.
    if (per_peer && !pto::comm::urma::detail::ValidateUrmaSubmissionTransition(
                        session, pto::comm::urma::UrmaBatchKind::Continuous, pto::comm::urma::UrmaCqePolicy::OneCqe
                    )) {
        runtime.set_error(RequestError::INVALID);
        return false;
    }
    if (!runtime.reserve(capacity)) {
        runtime.set_error(RequestError::REGISTRATION_FAILED);
        return false;
    }
    const uint32_t sender_base = session.qpIdxBase;
    const uint32_t jetties = session.cfg.urma.qpCount;
    if (per_peer) session.cfg.urma.qpCount = 1;
    uint32_t sequence = session.rt.urma.nextWqeSeq;
    bool submitted = true;
    const uint64_t chunk_elems = pto::comm::urma::kUrmaMaxWqeTransferBytes / sizeof(RawDType);
    for (uint64_t offset = 0; offset < elem_count;) {
        const uint64_t remaining = elem_count - offset;
        const uint64_t current_elems = remaining < chunk_elems ? remaining : chunk_elems;
        auto dst_slice = make_tensor_slice(desc.dst, offset, current_elems);
        auto src_slice = make_tensor_slice(desc.src, offset, current_elems);
        UrmaRequestDescriptor<decltype(dst_slice), decltype(src_slice)> chunk_desc{
            desc.op, dst_slice, src_slice, desc.workspace, desc.remote_rank
        };
        // A PER_PEER chunk registers only the queue receiving its single SQE.
        if (per_peer) session.qpIdxBase = sender_base + sequence % jetties;
        if (!submit_urma_request_once(runtime, chunk_desc, session)) {
            submitted = false;
            break;
        }
        ++sequence;
        offset += current_elems;
    }
    session.qpIdxBase = sender_base;
    session.cfg.urma.qpCount = jetties;
    return runtime.finish() && submitted;
}
#endif

template <typename Runtime, typename DstTensor, typename SrcTensor>
inline __aicore__ bool submit_chunked_urma_request(
    Runtime &runtime, UrmaRequestDescriptor<DstTensor, SrcTensor> desc, [[maybe_unused]] UrmaSession *bound = nullptr
) {
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
    if (bound == nullptr) {
        UrmaSession session;
        // The raw-pointer overload predates domain-owned sender headers.
        const bool built = desc.domain ?
                               detail::bind_urma_session(runtime.sender_id(), *desc.domain, session, desc.remote_rank) :
                               build_urma_session(runtime.sender_id(), desc.workspace, desc.remote_rank, session);
        if (built && (!desc.domain || session.domain_workspace == desc.workspace)) {
            session.domain_workspace = desc.workspace;
            return submit_tensor_chunks(runtime, desc, session, elem_count);
        }
    } else if (bound->domain_workspace == desc.workspace && can_submit(*bound, runtime.sender_id())) {
        return submit_tensor_chunks(runtime, desc, *bound, elem_count);
    }
#endif
    runtime.set_error(RequestError::INVALID);
    return false;
}

template <typename Session>
inline __aicore__ __gm__ uint8_t *completion_workspace(const Session &session) {
    return session.contextGm;
}

#if defined(PTO_URMA_SUPPORTED)
inline __aicore__ __gm__ uint8_t *completion_workspace(const UrmaSession &session) { return session.domain_workspace; }
#endif

template <typename Runtime, typename PtoAsyncEvent, typename PtoAsyncSession>
inline __aicore__ bool register_urma_async_event(
    Runtime &runtime, const PtoAsyncEvent &event, const PtoAsyncSession &session, __gm__ uint8_t *workspace,
    uint32_t peer
) {
    const uint32_t count = event.urmaJettyCount;
    if (count == 0) {
        if (event.postFailed || event.handle != 0) {
            runtime.set_error(RequestError::INVALID);
            return false;
        }
        return true;
    }
    if (!session.valid || completion_workspace(session) != workspace || workspace == nullptr ||
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
        if (!queue_row_for(workspace, peer, event.urmaJettyBase + j, rows[j])) {
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

// Existing event consumers use the peer bound at session creation.
template <typename Runtime, typename Event, typename Session>
inline __aicore__ bool
register_urma_async_event(Runtime &runtime, const Event &event, const Session &session, __gm__ uint8_t *workspace) {
    return register_urma_async_event(runtime, event, session, workspace, session.destRankId);
}

}  // namespace simpler::urma
