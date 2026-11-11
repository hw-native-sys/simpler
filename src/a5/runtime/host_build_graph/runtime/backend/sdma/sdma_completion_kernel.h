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

#include <pto/comm/async_common/async_event_impl.hpp>
#include <pto/comm/async/sdma/sdma_async_intrin.hpp>

#include "async_kernel_api.h"
#include "sdma_completion_registration.h"
#include "sdma_submission_guard.h"
#include "aicore_completion_mailbox_types.h"
#include "common/platform_config.h"
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

namespace pto2::detail {

namespace {

class SdmaSubmissionScope {
public:
    __aicore__ SdmaSubmissionScope() = default;
    SdmaSubmissionScope(const SdmaSubmissionScope &) = delete;
    SdmaSubmissionScope &operator=(const SdmaSubmissionScope &) = delete;

    inline __aicore__ bool acquire(AsyncCtx &ctx) {
        if (ctx.resident_enabled == 0) return true;
        guard_ = resident::sdma_submission_guard(ctx.resident, static_cast<uint32_t>(get_block_idx()));
        if (guard_ == nullptr) return false;
        const uint64_t start = get_sys_cnt();
        constexpr uint64_t timeout = UINT64_C(5) * PLATFORM_PROF_SYS_CNT_FREQ;
        for (;;) {
            const auto result = resident::sdma_try_acquire(*guard_);
            if (result == resident::SdmaGuardResult::ACQUIRED) {
                owned_ = true;
                return true;
            }
            if (result == resident::SdmaGuardResult::FAILED || get_sys_cnt() - start >= timeout) {
                defer_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
                return false;
            }
        }
    }

    inline __aicore__ bool release(bool success, __gm__ uint8_t *workspace) {
        if (!owned_) return guard_ == nullptr;
        if (success) {
            // SDK PersistSqTails mirrors its MTE3 metadata update with scalar
            // stores. Clean those lines before another AIV changes the same SQ.
            // A later session's full-cache maintenance must not write back an
            // older owner's dirty tail over a newer committed value.
            pipe_barrier(PIPE_ALL);
            dcci(workspace, cache_line_t::ENTIRE_DATA_CACHE, dcci_dst_t::CACHELINE_OUT);
            dsb((mem_dsb_t)0);
        }
        owned_ = false;
        return resident::sdma_release(*guard_, success);
    }

    inline __aicore__ ~SdmaSubmissionScope() {
        if (owned_) (void)resident::sdma_release(*guard_, false);
    }

private:
    __gm__ resident::AsyncSdmaSubmissionGuard *guard_{nullptr};
    bool owned_{false};
};

}  // namespace

}  // namespace pto2::detail

// SdmaRequestDescriptor bundles everything send_request_entry needs to drive
// one SDMA transfer + completion registration. It is a template because the
// destination / source / scratch types carry tensor shape & stride at compile
// time; the SdmaTget() / SdmaTput() helpers below let callers skip the
// template arguments.
//
// sync_id is forwarded to PTO-ISA's session builder for its internal pipe
// synchronization. Each call finishes temporary-pipe use before returning;
// the SDMA transfer itself is tracked by the registered completion records.
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

// SDMA overload of the runtime's send_request_entry. Submits the descriptor
// to PTO-ISA, then registers the resulting AsyncEvent's GM post-done records into the
// AsyncCtx deferred-wait slab and flushes. Returns false on submit/session
// failure (also records the error in ctx.completion_error_code).
template <typename DstTensor, typename SrcTensor, typename ScratchTileT>
static inline __aicore__ bool
send_request_entry(AsyncCtx &ctx, SdmaRequestDescriptor<DstTensor, SrcTensor, ScratchTileT> desc) {
    if (!async_ctx_is_deferred(ctx) ||
        (ctx.resident_enabled == 0 && (ctx.completion_count == nullptr || ctx.completion_entries == nullptr ||
                                       (ctx.completion_error_code != nullptr && *ctx.completion_error_code != 0))))
        return false;
    uint64_t source_bytes = 0;
    uint64_t destination_bytes = 0;
    if ((desc.op != SdmaOp::TGET && desc.op != SdmaOp::TPUT) ||
        !pto2::detail::flat_transfer_bytes(desc.src, source_bytes) ||
        !pto2::detail::flat_transfer_bytes(desc.dst, destination_bytes) || destination_bytes < source_bytes) {
        pto2::detail::defer_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
        return false;
    }
    if (source_bytes == 0) return true;
    pto2::detail::SdmaSubmissionScope submission;
    if (!submission.acquire(ctx)) return false;
    pto::comm::AsyncSession session;
    if (!pto::comm::BuildAsyncSession(desc.scratch, desc.workspace, session, desc.sync_id)) {
        pto2::detail::defer_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
        return false;
    }

    const uint32_t record_limit = session.cfg.sdma.queueNum;
    pto2::resident::AsyncReservation reservation{};
    if (ctx.resident_enabled != 0) {
        reservation = pto2::resident::reserve_conditions(ctx.resident, record_limit);
        if (reservation.serial == 0) return false;
    } else if (record_limit == 0 || *ctx.completion_count > ctx.completion_capacity ||
               record_limit > ctx.completion_capacity - *ctx.completion_count) {
        pto2::detail::defer_error(ctx, SIMPLER_ERROR_ASYNC_WAIT_OVERFLOW);
        return false;
    }

    pto::comm::AsyncEvent event;
    if (desc.op == SdmaOp::TGET) {
        event = pto::comm::TGET_ASYNC(desc.dst, desc.src, session);
    } else {
        event = pto::comm::TPUT_ASYNC(desc.dst, desc.src, session);
    }
    const bool registered =
        pto2::detail::register_pto_async_event(ctx, event, session, reservation, record_limit, desc.op);
    if (!submission.release(registered, desc.workspace)) {
        pto2::detail::defer_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
        return false;
    }
    pto2::detail::defer_flush(ctx);
    return registered;
}
