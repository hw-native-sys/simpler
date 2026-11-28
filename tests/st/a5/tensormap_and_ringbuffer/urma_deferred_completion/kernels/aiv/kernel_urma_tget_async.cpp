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

#include <cstdint>

#ifndef __gm__
#define __gm__
#endif
#ifndef __aicore__
#define __aicore__ [aicore]
#endif

#ifdef MEMORY_BASE
#undef MEMORY_BASE
#endif
#ifndef REGISTER_BASE
#define REGISTER_BASE
#endif

#include <pto/pto-inst.hpp>

#include "backend/urma/urma_completion_kernel.h"
#include "common/platform_config.h"
#include "platform_comm/comm_context.h"
#include "tensor.h"

namespace {

using Shape = pto::Shape<1, 1, 1, 1, pto::DYNAMIC>;
using Stride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1>;
using GlobalData = pto::GlobalTensor<float, Shape, Stride>;
constexpr uint32_t kChunkedElems = 256 * 1024 * 1024 / sizeof(float) + 8192;

inline __aicore__ GlobalData global_data(__gm__ float *ptr, uint32_t elems) {
    return GlobalData(ptr, Shape(1, 1, 1, 1, elems), Stride(elems, elems, elems, elems, 1));
}

inline __aicore__ __gm__ float *tensor_data(__gm__ Tensor *tensor) {
    return reinterpret_cast<__gm__ float *>(tensor->buffer.addr) + tensor->start_offset;
}

template <typename Tensor, typename Workspace>
inline __aicore__ void submit_repeated(
    AsyncCtx &ctx, Tensor dst, Tensor src, const Workspace &workspace, uint32_t peer, bool put, uint32_t repeats
) {
    const uint32_t before = ctx.completion_count == nullptr ? 0 : *ctx.completion_count;
    uint32_t entries_per_request = 0;
    for (uint32_t i = 0; i < repeats; ++i) {
        const bool ok = put ? send_request_entry(ctx, UrmaTput(dst, src, workspace, peer)) :
                              send_request_entry(ctx, UrmaTget(dst, src, workspace, peer));
        if (!ok) return;
        if (i == 0) entries_per_request = *ctx.completion_count - before;
    }
    if (entries_per_request == 0 || ctx.completion_count == nullptr ||
        static_cast<uint64_t>(*ctx.completion_count - before) != static_cast<uint64_t>(entries_per_request) * repeats) {
        pto2::detail::defer_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
        pto2::detail::defer_flush(ctx);
    }
}

#if defined(PTO_URMA_SUPPORTED)
struct alignas(64) DescriptorScratch {
    alignas(64) uint8_t range[64];
    alignas(64) uint8_t segment[64];
    alignas(64) uint32_t ready;
};
static_assert(sizeof(DescriptorScratch) == 3 * 64);

inline __aicore__ bool rendezvous(__gm__ DescriptorScratch *scratch, uint32_t lane, uint32_t phase) {
    scratch[lane].ready = phase;
    pto::comm::urma::DcciCachelines(reinterpret_cast<__gm__ uint8_t *>(&scratch[lane].ready), sizeof(uint32_t));
    dsb(DSB_DDR);
    const uint64_t begin = get_sys_cnt();
    while (ld_dev(&scratch[1 - lane].ready, 0) < phase) {
        if (get_sys_cnt() - begin > PLATFORM_PROF_SYS_CNT_FREQ) return false;
    }
    return true;
}

template <bool Put>
inline __aicore__ bool submit_descriptor_transfer(
    AsyncCtx &ctx, __gm__ DescriptorScratch *scratch, uint32_t lane, const __gm__ CommWorkspace &urma, uint32_t peer,
    simpler::urma::UrmaSession &session, __gm__ float *local, __gm__ float *remote, uint32_t elems, uint32_t repeats,
    bool native_wait
) {
    auto *workspace = reinterpret_cast<__gm__ uint8_t *>(urma.addr);
    const auto *info = reinterpret_cast<__gm__ const pto::comm::urma::UrmaInfo *>(workspace);
    using Remote = std::conditional_t<Put, pto::comm::GatherDestination, pto::comm::ScatterSource>;
    using Local = std::conditional_t<Put, pto::comm::GatherSegment, pto::comm::ScatterSegment>;
    auto *range = reinterpret_cast<__gm__ Remote *>(scratch[lane].range);
    auto *segment = reinterpret_cast<__gm__ Local *>(scratch[lane].segment);
    if (native_wait && (session.cfg.urma.qpCount != 1 || repeats != 2)) return false;
    const uint32_t before = ctx.completion_count == nullptr ? 0 : *ctx.completion_count;
    const uint32_t part = elems / repeats;
    for (uint32_t i = 0; i < repeats; ++i) {
        const uint32_t offset = i * part;
        const uint32_t length = (i + 1 == repeats ? elems - offset : part) * sizeof(float);
        range->addr = reinterpret_cast<__gm__ uint8_t *>(remote + offset);
        range->segCount = 1;
        range->peer = peer;
        segment->addr = reinterpret_cast<__gm__ uint8_t *>(local + offset);
        segment->length = length;
        pto::comm::urma::DcciCachelines(scratch[lane].range, 64 + sizeof(Local));
        dsb(DSB_DDR);
        if (native_wait && i == 1) {
            // Both siblings have registered A before either can enter native B.
            const auto *bound = reinterpret_cast<__gm__ const pto::comm::urma::UrmaInfo *>(session.contextGm);
            const auto stride = info->gatherRuntimeStride;
            bool ok = bound->gatherRuntimeBase + static_cast<uint64_t>(get_block_idx()) * stride ==
                      info->gatherRuntimeBase + static_cast<uint64_t>(ctx.urma_sender_id) * stride;
            ok = ok && rendezvous(scratch, lane, Put ? 1 : 2);
            if (ok) {
                pto::comm::AsyncEvent event;
                if constexpr (Put) {
                    event = pto::comm::TPUT_ASYNC<pto::comm::DmaEngine::URMA>(range, 1, segment, 1, session);
                } else {
                    event = pto::comm::TGET_ASYNC<pto::comm::DmaEngine::URMA>(segment, 1, range, 1, session);
                }
                ok = !event.postFailed && event.Wait(session);
                // Consecutive CQ sequence numbers on the single jetty establish that Wait(B) covered A.
                ok = ok && event.urmaJettyBase == session.qpIdxBase &&
                     event.urmaTargetCqe == ctx.completion_entries[before].expected_value + 1U;
            }
            if (!ok) {
                return false;
            }
            continue;
        }
        if constexpr (Put) {
            const auto request = UrmaTput(range, 1, segment, 1);
            if (!(native_wait ? send_request_entry(ctx, request, session) : send_request_entry(ctx, request, urma)))
                return false;
        } else {
            const auto request = UrmaTget(segment, 1, range, 1);
            if (!(native_wait ? send_request_entry(ctx, request, session) : send_request_entry(ctx, request, urma)))
                return false;
        }
    }
    uint32_t expected_row = 0;
    const auto first_jetty = ctx.urma_sender_id * info->jettiesPerCore;
    return ctx.completion_count != nullptr && *ctx.completion_count - before == repeats - (native_wait ? 1 : 0) &&
           simpler::urma::queue_row_for(workspace, peer, first_jetty, expected_row) &&
           (ctx.completion_entries[before].addr >> 32) == expected_row;
}
#endif

}  // namespace

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    auto *window_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    auto *remote_window_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    auto *out_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    auto *comm = reinterpret_cast<__gm__ CommContext *>(args[4]);
    const uint32_t elems = static_cast<uint32_t>(args[5]);
    const bool put = args[6] == 1;
    const uint32_t repeats = static_cast<uint32_t>(args[7]);
    AsyncCtx ctx = get_async_ctx(args);
    const uint32_t ranks = args[6] == 6 ? 3 : 2;
    if (comm == nullptr || comm->rankNum != ranks || comm->rankId >= ranks ||
        comm_workspace<CommWorkspaceKind::URMA>(*comm).addr == 0 || comm->windowsIn[comm->rankId] == 0 || elems == 0 ||
        repeats == 0) {
        pto2::detail::defer_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
        pto2::detail::defer_flush(ctx);
        return;
    }
    const auto &urma = comm_workspace<CommWorkspaceKind::URMA>(*comm);
    auto *workspace = reinterpret_cast<__gm__ uint8_t *>(urma.addr);
    const uint32_t peer = (comm->rankId + 1) % ranks;
    auto *local = tensor_data(window_tensor);
    const uint64_t remote_offset =
        reinterpret_cast<uint64_t>(tensor_data(remote_window_tensor)) - comm->windowsIn[comm->rankId];
    auto *remote = pto2::urma_backend::peer_mr_ptr<float>(workspace, peer, remote_offset);
    if (args[6] >= 2) {
#if defined(PTO_URMA_SUPPORTED)
        const uint32_t lane = get_sub_block_id(args);
        const uint32_t lane_elems = elems / 2;
        const uint32_t offset = lane * lane_elems;
        auto *scratch = reinterpret_cast<__gm__ DescriptorScratch *>(comm->windowsIn[comm->rankId]);
        const bool native_wait = args[6] >= 4;
        remote += offset;
        auto *output = tensor_data(out_tensor) + offset;
        simpler::urma::UrmaSession session;
        bool ok = !native_wait || (ranks == 3 ? simpler::urma::build_urma_session(ctx, urma, session) :
                                                simpler::urma::build_urma_session(ctx, urma, session, peer));
        if (ok && ranks == 3) {
            const uint32_t half = lane_elems / 2;
            const uint32_t quarter = half / 2;
            const uint32_t other_peer = (comm->rankId + 2) % ranks;
            const uint64_t input_offset = reinterpret_cast<uint64_t>(local) - comm->windowsIn[comm->rankId];
            auto *other_input = pto2::urma_backend::peer_mr_ptr<float>(workspace, other_peer, input_offset);
            // The first half of the readback contains disjoint tensor and Gather PUTs.
            auto put_dst = global_data(remote, quarter);
            auto put_src = global_data(local + offset, quarter);
            auto get_dst = global_data(output, half);
            auto get_src = global_data(remote, half);
            ok = send_request_entry(ctx, UrmaTput(put_dst, put_src, workspace, peer), session) &&
                 submit_descriptor_transfer<true>(
                     ctx, scratch, lane, urma, peer, session, local + offset + quarter, remote + quarter, quarter,
                     repeats, true
                 ) &&
                 submit_descriptor_transfer<false>(
                     ctx, scratch, lane, urma, other_peer, session, output + half, other_input + offset + half, half,
                     repeats, true
                 ) &&
                 send_request_entry(ctx, UrmaTget(get_dst, get_src, workspace, peer), session);
        } else if (ok && (args[6] == 2 || args[6] == 4)) {
            ok = submit_descriptor_transfer<true>(
                ctx, scratch, lane, urma, peer, session, local + offset, remote, lane_elems, repeats, native_wait
            );
        } else if (ok) {
            ok = submit_descriptor_transfer<false>(
                ctx, scratch, lane, urma, peer, session, output, remote, lane_elems, repeats, native_wait
            );
        }
        if (!ok) {
            if (ctx.completion_error_code && *ctx.completion_error_code == SIMPLER_ERROR_NONE)
                pto2::detail::defer_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
            pto2::detail::defer_flush(ctx);
        }
#else
        pto2::detail::defer_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
        pto2::detail::defer_flush(ctx);
#endif
        return;
    }
    if (!put && elems == kChunkedElems) {
        using StaticShape = pto::Shape<1, 1, 1, 1, kChunkedElems>;
        using StaticStride = pto::Stride<kChunkedElems, kChunkedElems, kChunkedElems, kChunkedElems, 1>;
        using StaticData = pto::GlobalTensor<float, StaticShape, StaticStride>;
        submit_repeated(ctx, StaticData(tensor_data(out_tensor)), StaticData(remote), workspace, peer, false, repeats);
        return;
    }
    auto src = global_data(put ? local : remote, elems);
    auto dst = global_data(put ? remote : tensor_data(out_tensor), elems);
    submit_repeated(ctx, dst, src, urma, peer, put, repeats);
}
