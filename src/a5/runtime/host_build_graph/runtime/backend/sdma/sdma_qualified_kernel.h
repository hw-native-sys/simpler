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

#include "sdma_completion_kernel.h"
#include "sdma_metadata_pto_bridge.h"
#include "sdma_qualified_submission.h"

namespace pto2::detail {
template <typename Descriptor>
class SdmaQualifiedPtoBackend {
public:
    inline __aicore__ explicit SdmaQualifiedPtoBackend(Descriptor &descriptor) :
        descriptor_(descriptor) {}
    inline __aicore__ bool preflight() const {
        using Scratch = std::remove_reference_t<decltype(descriptor_.scratch)>;
        static_assert(Scratch::Loc == pto::TileType::Vec);
        constexpr uint64_t scratch_bytes = static_cast<uint64_t>(Scratch::Numel) * sizeof(typename Scratch::DType);
        const uint64_t address = reinterpret_cast<uint64_t>(descriptor_.scratch.data());
        // Host currently exports a remote-read (TGET) use. The old TPUT helper
        // keeps its synchronous MTE fallback; it cannot consume this authority.
        return descriptor_.op == SdmaOp::TGET &&
               sdma_qualified_scratch_valid(
                   address, scratch_bytes, pto::comm::sdma::UB_ALIGN_SIZE, descriptor_.sync_id
               );
    }
    inline __aicore__ bool build_session(uint32_t queues, uint32_t group) {
        const pto::comm::sdma::SdmaBaseConfig config{pto::comm::sdma::kDefaultSdmaBlockBytes, 0, queues};
        return pto::comm::BuildAsyncSession(
            descriptor_.scratch, descriptor_.workspace, session_, descriptor_.sync_id, config, group
        );
    }
    inline __aicore__ SdmaMetadataSession metadata_session() const { return sdma_metadata_session(session_); }
    inline __aicore__ bool submit() {
        event_ = pto::comm::TGET_ASYNC(descriptor_.dst, descriptor_.src, session_);
        return true;
    }
    inline __aicore__ bool event_valid(uint64_t bytes) const {
        return async_sdma_event_valid(
            SdmaOp::TGET, bytes, event_.handle, event_.postFailed, static_cast<uint32_t>(event_.engine),
            static_cast<uint32_t>(pto::comm::DmaEngine::SDMA)
        );
    }
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
}  // namespace pto2::detail

// Qualified SDK entry: TaskArgs identifies the actual resident frame. A caller
// cannot upgrade the old AsyncCtx-only path by supplying a workspace address.
// Receipt sink(index, count, record) runs only after successful publication and
// copies public CompletionRecordAt values; it neither observes READY nor grants
// access to private scheduler rows. No callback is invoked for an empty transfer.
template <typename DstTensor, typename SrcTensor, typename ScratchTileT, typename Sink>
inline __aicore__ bool
send_request_entry(__gm__ int64_t *args, SdmaRequestDescriptor<DstTensor, SrcTensor, ScratchTileT> desc, Sink sink) {
    using namespace pto2::detail;
    SdmaMetadataHandoffCache cache;
#if defined(__DAV_CUBE__) && !defined(__DAV_VEC__)
    // The pinned SDMA session uses UB/MTE3. AIC may observe its completion, but
    // does not execute this AIV submission backend.
    return sdma_qualified_reject(args, cache);
#else
    uint64_t bytes = 0;
    const bool valid_shape = async_sdma_transfer_bytes(desc.dst, desc.src, &bytes);
    SdmaQualifiedPtoBackend<decltype(desc)> backend(desc);
    return sdma_qualified_entry(
        args, valid_shape, reinterpret_cast<uint64_t>(desc.workspace), reinterpret_cast<uint64_t>(desc.src.data()),
        reinterpret_cast<uint64_t>(desc.dst.data()), bytes, SDMA_PTO_METADATA_LAYOUT, backend, cache, sink
    );
#endif
}

template <typename DstTensor, typename SrcTensor, typename ScratchTileT>
inline __aicore__ bool
send_request_entry(__gm__ int64_t *args, SdmaRequestDescriptor<DstTensor, SrcTensor, ScratchTileT> desc) {
    return send_request_entry(args, desc, SdmaIgnoreCompletionRecords{});
}
