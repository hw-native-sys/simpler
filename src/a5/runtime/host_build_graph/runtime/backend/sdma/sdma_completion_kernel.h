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
#include "sdma_submission.h"
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
    uint32_t queue_num{1};
};

template <typename DstTensor, typename SrcTensor, typename ScratchTileT>
inline __aicore__ SdmaRequestDescriptor<DstTensor, SrcTensor, ScratchTileT> SdmaTget(
    const DstTensor &dst, const SrcTensor &src, const ScratchTileT &scratch, __gm__ uint8_t *workspace,
    uint32_t sync_id = 0, uint32_t queue_num = 1
) {
    return SdmaRequestDescriptor<DstTensor, SrcTensor, ScratchTileT>{SdmaOp::TGET, dst,     src,      scratch,
                                                                     workspace,    sync_id, queue_num};
}

template <typename DstTensor, typename SrcTensor, typename ScratchTileT>
inline __aicore__ SdmaRequestDescriptor<DstTensor, SrcTensor, ScratchTileT> SdmaTput(
    const DstTensor &dst, const SrcTensor &src, const ScratchTileT &scratch, __gm__ uint8_t *workspace,
    uint32_t sync_id = 0, uint32_t queue_num = 1
) {
    return SdmaRequestDescriptor<DstTensor, SrcTensor, ScratchTileT>{SdmaOp::TPUT, dst,     src,      scratch,
                                                                     workspace,    sync_id, queue_num};
}

namespace pto2::detail {
template <typename Descriptor>
class SdmaPtoBackend {
public:
    inline __aicore__ explicit SdmaPtoBackend(Descriptor &descriptor) :
        descriptor_(descriptor) {}
    inline __aicore__ bool preflight() const {
        using Scratch = std::remove_reference_t<decltype(descriptor_.scratch)>;
        static_assert(Scratch::Loc == pto::TileType::Vec);
        static_assert(pto::comm::sdma::kSdmaMaxChannel == ASYNC_SDMA_CHANNELS);
        static_assert(sizeof(pto::comm::sdma::BatchWriteFlagInfo) == 64);
        static_assert(sizeof(pto::comm::sdma::BatchWriteChannelInfo) == 64);
        constexpr uint64_t bytes = static_cast<uint64_t>(Scratch::Numel) * sizeof(typename Scratch::DType);
        return sdma_scratch_valid(
            reinterpret_cast<uint64_t>(descriptor_.scratch.data()), bytes, pto::comm::sdma::UB_ALIGN_SIZE,
            descriptor_.sync_id
        );
    }
    inline __aicore__ bool build_session(uint32_t queues, uint32_t group) {
        const pto::comm::sdma::SdmaBaseConfig config{pto::comm::sdma::kDefaultSdmaBlockBytes, 0, queues};
        return pto::comm::BuildAsyncSession(
            descriptor_.scratch, descriptor_.workspace, session_, descriptor_.sync_id, config, group
        );
    }
    inline __aicore__ bool session_valid(uint32_t queues, uint32_t group) const {
        return session_.valid && session_.engine == pto::comm::DmaEngine::SDMA &&
               session_.contextGm == descriptor_.workspace && session_.cfg.sdma.queueNum == queues &&
               session_.cfg.sdma.channelGroupIdx == group;
    }
    inline __aicore__ bool submit() {
        if (descriptor_.op == SdmaOp::TGET) event_ = pto::comm::TGET_ASYNC(descriptor_.dst, descriptor_.src, session_);
        else event_ = pto::comm::TPUT_ASYNC(descriptor_.dst, descriptor_.src, session_);
        return true;
    }
    inline __aicore__ bool event_valid(uint64_t bytes) const {
        return async_sdma_event_valid(
            descriptor_.op, bytes, event_.handle, event_.postFailed, static_cast<uint32_t>(event_.engine),
            static_cast<uint32_t>(pto::comm::DmaEngine::SDMA)
        );
    }
    inline __aicore__ bool has_records() const { return event_.handle != 0; }
    inline __aicore__ uint32_t record_count() const { return event_.CompletionRecordCount(session_); }
    inline __aicore__ bool record(uint32_t index, SdmaCompletionReceiptRecord *out) const {
        const auto record = event_.CompletionRecordAt(session_, index);
        if (out == nullptr || record.kind != pto::comm::CompletionKind::SDMA_POST_DONE || record.addr == nullptr)
            return false;
        *out = {reinterpret_cast<uint64_t>(record.addr), record.expected};
        return true;
    }
    inline __aicore__ bool wait() const { return event_.Wait(session_); }

private:
    Descriptor &descriptor_;
    pto::comm::AsyncSession session_{};
    pto::comm::AsyncEvent event_{};
};

template <typename Descriptor, typename Sink>
inline __aicore__ bool sdma_kernel_submit(AsyncCtx &ctx, Descriptor &descriptor, Sink &sink) {
    CompletionCache cache;
#if defined(__DAV_CUBE__) && !defined(__DAV_VEC__)
    return sdma_submit_error(ctx, cache);
#else
    uint64_t bytes = 0;
    const bool valid_shape = async_sdma_transfer_bytes(descriptor.dst, descriptor.src, &bytes);
    SdmaPtoBackend<Descriptor> backend(descriptor);
    return sdma_submit(
        ctx, descriptor.op, valid_shape, reinterpret_cast<uint64_t>(descriptor.workspace),
        reinterpret_cast<uint64_t>(descriptor.src.data()), reinterpret_cast<uint64_t>(descriptor.dst.data()), bytes,
        static_cast<uint32_t>(get_block_idx()), backend, cache, sink, SdmaSubmissionMetadata{}, descriptor.queue_num
    );
#endif
}
}  // namespace pto2::detail

template <typename DstTensor, typename SrcTensor, typename ScratchTileT>
inline __aicore__ bool
send_request_entry(AsyncCtx &ctx, SdmaRequestDescriptor<DstTensor, SrcTensor, ScratchTileT> descriptor) {
    SdmaIgnoreCompletionRecords sink;
    return pto2::detail::sdma_kernel_submit(ctx, descriptor, sink);
}

// Receipts are emitted after SDK metadata publication and queue-guard release.
// This adapter uses the same canonical context as the AsyncCtx overload.
template <typename DstTensor, typename SrcTensor, typename ScratchTileT, typename Sink>
inline __aicore__ bool send_request_entry(
    __gm__ int64_t *args, SdmaRequestDescriptor<DstTensor, SrcTensor, ScratchTileT> descriptor, Sink sink
) {
    AsyncCtx ctx{};
    if (!async_execution_context_load(args, &ctx)) return false;
    return pto2::detail::sdma_kernel_submit(ctx, descriptor, sink);
}

template <typename DstTensor, typename SrcTensor, typename ScratchTileT>
inline __aicore__ bool
send_request_entry(__gm__ int64_t *args, SdmaRequestDescriptor<DstTensor, SrcTensor, ScratchTileT> descriptor) {
    return send_request_entry(args, descriptor, SdmaIgnoreCompletionRecords{});
}
