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

#ifndef SRC_A5_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_BACKEND_URMA_URMA_COMPLETION_KERNEL_H_
#define SRC_A5_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_BACKEND_URMA_URMA_COMPLETION_KERNEL_H_

#include <stdint.h>

#if defined(__CPU_SIM)
#include <type_traits>
#else
#include "pto_async_kernel_api.h"

#include <pto/comm/async_common/async_event_impl.hpp>
#if defined(PTO_URMA_SUPPORTED)
#include <pto/comm/async/urma/urma_async_intrin.hpp>
#endif
#endif

#include "aicore_completion_mailbox_types.h"
#include "backend/urma/urma_completion_abi.h"
#include "intrinsic.h"
#include "pto_completion_token.h"
#include "pto_runtime_status.h"

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

#if defined(__CPU_SIM)
inline __aicore__ AsyncCtx get_async_ctx(__gm__ int64_t *args) {
    __gm__ LocalContext *lc =
        reinterpret_cast<__gm__ LocalContext *>(static_cast<uintptr_t>(args[PAYLOAD_LOCAL_CONTEXT_INDEX]));
    AsyncCtx ctx{};
    ctx.completion_count = lc->async_ctx.completion_count;
    ctx.completion_error_code = lc->async_ctx.completion_error_code;
    ctx.completion_entries = lc->async_ctx.completion_entries;
    ctx.completion_capacity = lc->async_ctx.completion_capacity;
    ctx.task_token.raw = lc->async_ctx.task_token.raw;
    return ctx;
}

inline __aicore__ bool register_completion_condition(AsyncCtx &ctx, const CompletionToken &token) {
    if (ctx.task_token.is_invalid() || ctx.completion_count == nullptr || ctx.completion_entries == nullptr) {
        return false;
    }

    uint32_t idx = *ctx.completion_count;
    if (idx >= ctx.completion_capacity) {
        if (ctx.completion_error_code != nullptr) {
            *ctx.completion_error_code = PTO2_ERROR_ASYNC_WAIT_OVERFLOW;
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
inline __aicore__ void register_urma_async_event(
    AsyncCtx &ctx, const PtoAsyncEvent &event, const PtoAsyncSession &session, __gm__ uint8_t *workspace
) {
    if (ctx.task_token.is_invalid() || ctx.completion_count == nullptr || ctx.completion_entries == nullptr) {
        (void)event.Wait(session);
        return;
    }
    if (event.handle == 0) {
        return;
    }

    const uint32_t engine = static_cast<uint32_t>(event.engine);
    if (engine != static_cast<uint32_t>(::pto::comm::DmaEngine::URMA) || workspace == nullptr) {
        defer_error(ctx, PTO2_ERROR_ASYNC_COMPLETION_INVALID);
        return;
    }

    CompletionToken token{
        event.handle,
        0,
        COMPLETION_ENGINE_URMA,
        COMPLETION_TYPE_URMA_EVENT_HANDLE,
        reinterpret_cast<uint64_t>(workspace),
    };
    if (!register_completion_condition(ctx, token)) {
        defer_error(ctx, PTO2_ERROR_ASYNC_REGISTRATION_FAILED);
    }
}

}  // namespace pto2::detail

#if defined(__CPU_SIM)
namespace pto2::urma_backend {

struct FakeUrmaAsyncSession {
    __gm__ uint8_t *workspace{nullptr};
    uint32_t remote_rank{0};
};

inline __aicore__ bool
build_fake_urma_session(__gm__ uint8_t *workspace, uint32_t remote_rank, FakeUrmaAsyncSession &session) {
    if (workspace == nullptr || remote_rank >= kFakeUrmaMaxRanks) {
        return false;
    }
    session.workspace = workspace;
    session.remote_rank = remote_rank;
    return true;
}

inline __aicore__ FakeUrmaWorkspace *fake_workspace(__gm__ uint8_t *workspace) {
    return reinterpret_cast<FakeUrmaWorkspace *>(workspace);
}

inline __aicore__ void ensure_fake_workspace_initialized(__gm__ uint8_t *workspace) {
    FakeUrmaWorkspace *ws = fake_workspace(workspace);
    uint32_t magic = __atomic_load_n(&ws->magic, __ATOMIC_ACQUIRE);
    if (magic == kFakeUrmaWorkspaceMagic) {
        return;
    }
    if (magic == kFakeUrmaWorkspaceInitializing) {
        while (__atomic_load_n(&ws->magic, __ATOMIC_ACQUIRE) != kFakeUrmaWorkspaceMagic) {}
        return;
    }
    uint32_t expected = magic;
    if (!__atomic_compare_exchange_n(
            &ws->magic, &expected, kFakeUrmaWorkspaceInitializing, false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE
        )) {
        while (__atomic_load_n(&ws->magic, __ATOMIC_ACQUIRE) != kFakeUrmaWorkspaceMagic) {}
        return;
    }

    ws->info.qp_num = 1;
    ws->info.local_token_id = 0;
    ws->info.rank_count = kFakeUrmaMaxRanks;
    ws->info.sq_ptr = reinterpret_cast<uint64_t>(&ws->sq[0]);
    ws->info.rq_ptr = 0;
    ws->info.scq_ptr = reinterpret_cast<uint64_t>(&ws->scq[0]);
    ws->info.rcq_ptr = 0;
    ws->info.mem_ptr = 0;

    for (uint32_t rank = 0; rank < kFakeUrmaMaxRanks; ++rank) {
        ws->sq_head[rank] = 0;
        ws->sq_tail[rank] = 0;
        ws->cq_tail[rank] = 0;
        ws->cq_doorbell[rank] = 0;
        ws->sq_submit_lock[rank] = 0;

        ws->sq[rank].wqn = rank;
        ws->sq[rank].buf_addr = 0;
        ws->sq[rank].wqe_shift_size = 0;
        ws->sq[rank].depth = kFakeUrmaCqDepth;
        ws->sq[rank].head_addr = reinterpret_cast<uint64_t>(&ws->sq_head[rank]);
        ws->sq[rank].tail_addr = reinterpret_cast<uint64_t>(&ws->sq_tail[rank]);
        ws->sq[rank].db_mode = UrmaDbMode::SW_DB;
        ws->sq[rank].db_addr = 0;
        ws->sq[rank].sl = 0;

        ws->scq[rank].cqn = rank;
        ws->scq[rank].buf_addr = reinterpret_cast<uint64_t>(&ws->scq_entries[rank][0]);
        ws->scq[rank].cqe_shift_size = kFakeUrmaCqeShift;
        ws->scq[rank].depth = kFakeUrmaCqDepth;
        ws->scq[rank].head_addr = 0;
        ws->scq[rank].tail_addr = reinterpret_cast<uint64_t>(&ws->cq_tail[rank]);
        ws->scq[rank].db_mode = UrmaDbMode::SW_DB;
        ws->scq[rank].db_addr = reinterpret_cast<uint64_t>(&ws->cq_doorbell[rank]);

        for (uint32_t slot = 0; slot < kFakeUrmaCqDepth; ++slot) {
            ws->scq_entries[rank][slot].dw[0] = encode_fake_pending_cqe_dw0(slot);
        }
    }
    __atomic_store_n(&ws->magic, kFakeUrmaWorkspaceMagic, __ATOMIC_RELEASE);
}

template <typename GlobalData>
inline __aicore__ bool fake_is_flat_contiguous_1d(GlobalData &global_data) {
    const int64_t shp0 = global_data.GetShape(pto::GlobalTensorDim::DIM_0);
    const int64_t shp1 = global_data.GetShape(pto::GlobalTensorDim::DIM_1);
    const int64_t shp2 = global_data.GetShape(pto::GlobalTensorDim::DIM_2);
    const int64_t shp3 = global_data.GetShape(pto::GlobalTensorDim::DIM_3);
    const int64_t shp4 = global_data.GetShape(pto::GlobalTensorDim::DIM_4);

    const int64_t step0 = global_data.GetStride(pto::GlobalTensorDim::DIM_0);
    const int64_t step1 = global_data.GetStride(pto::GlobalTensorDim::DIM_1);
    const int64_t step2 = global_data.GetStride(pto::GlobalTensorDim::DIM_2);
    const int64_t step3 = global_data.GetStride(pto::GlobalTensorDim::DIM_3);
    const int64_t step4 = global_data.GetStride(pto::GlobalTensorDim::DIM_4);

    const bool packed_layout = (step4 == 1) && (step3 == shp4) && (step2 == shp3 * step3) && (step1 == shp2 * step2) &&
                               (step0 == shp1 * step1);
    const bool one_dim_logical = (shp0 == 1 && shp1 == 1 && shp2 == 1 && shp3 == 1);
    return packed_layout && one_dim_logical;
}

template <typename GlobalData>
inline __aicore__ uint32_t fake_total_elem_count(GlobalData &global_data) {
    const uint32_t d0 = static_cast<uint32_t>(global_data.GetShape(pto::GlobalTensorDim::DIM_0));
    const uint32_t d1 = static_cast<uint32_t>(global_data.GetShape(pto::GlobalTensorDim::DIM_1));
    const uint32_t d2 = static_cast<uint32_t>(global_data.GetShape(pto::GlobalTensorDim::DIM_2));
    const uint32_t d3 = static_cast<uint32_t>(global_data.GetShape(pto::GlobalTensorDim::DIM_3));
    const uint32_t d4 = static_cast<uint32_t>(global_data.GetShape(pto::GlobalTensorDim::DIM_4));
    return (((d0 * d1) * d2) * d3) * d4;
}

template <typename DstTensor, typename SrcTensor>
inline __aicore__ bool fake_copy_tensor(DstTensor &dst, SrcTensor &src) {
    using SrcElem = typename SrcTensor::RawDType;
    using DstElem = typename DstTensor::RawDType;
    static_assert(std::is_same_v<SrcElem, DstElem>, "URMA fake submitter requires matching src/dst element types");
    static_assert(SrcTensor::layout == DstTensor::layout, "URMA fake submitter requires matching src/dst layouts");

    if (dst.data() == nullptr || src.data() == nullptr) {
        return false;
    }
    if (!fake_is_flat_contiguous_1d(src) || !fake_is_flat_contiguous_1d(dst)) {
        return false;
    }

    const uint32_t src_elems = fake_total_elem_count(src);
    const uint32_t dst_elems = fake_total_elem_count(dst);
    if (dst_elems < src_elems) {
        return false;
    }

    SrcElem *src_ptr = src.data();
    DstElem *dst_ptr = dst.data();
    for (uint32_t i = 0; i < src_elems; ++i) {
        dst_ptr[i] = src_ptr[i];
    }
    return true;
}

inline __aicore__ bool mark_fake_urma_cqe_ready(
    __gm__ uint8_t *workspace, uint32_t remote_rank, uint32_t target_head, uint8_t status = 0, uint8_t substatus = 0
) {
    if (workspace == nullptr || remote_rank >= kFakeUrmaMaxRanks) {
        return false;
    }
    if (target_head == 0) {
        return true;
    }

    ensure_fake_workspace_initialized(workspace);
    FakeUrmaWorkspace *ws = fake_workspace(workspace);
    const uint32_t cqe_seq = target_head - 1u;
    FakeUrmaCqe *cqe = &ws->scq_entries[remote_rank][cqe_seq & (kFakeUrmaCqDepth - 1u)];
    for (uint32_t i = 1; i < 16; ++i) {
        cqe->dw[i] = 0;
    }
    __atomic_store_n(&cqe->dw[0], encode_fake_cqe_dw0(cqe_seq, status, substatus), __ATOMIC_RELEASE);
    return true;
}

struct FakeUrmaAsyncEvent {
    uint64_t handle{0};
    pto::comm::DmaEngine engine{pto::comm::DmaEngine::URMA};

    inline __aicore__ bool Wait(const FakeUrmaAsyncSession &session) const {
        uint32_t remote_rank = 0;
        uint32_t target_head = 0;
        decode_urma_event_handle(handle, remote_rank, target_head);
        return mark_fake_urma_cqe_ready(session.workspace, remote_rank, target_head);
    }

    inline __aicore__ bool Test(const FakeUrmaAsyncSession & /*session*/) const { return handle == 0; }
};

template <typename DstTensor, typename SrcTensor>
inline __aicore__ bool
submit_fake_urma_request(UrmaRequestDescriptor<DstTensor, SrcTensor> &desc, FakeUrmaAsyncEvent &event) {
    if (desc.workspace == nullptr || desc.remote_rank >= kFakeUrmaMaxRanks) {
        return false;
    }
    if (!fake_copy_tensor(desc.dst, desc.src)) {
        return false;
    }

    ensure_fake_workspace_initialized(desc.workspace);
    FakeUrmaWorkspace *ws = fake_workspace(desc.workspace);
    uint32_t unlocked = 0;
    while (!__atomic_compare_exchange_n(
        &ws->sq_submit_lock[desc.remote_rank], &unlocked, 1u, false, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED
    )) {
        unlocked = 0;
    }
    const uint32_t target_head = __atomic_load_n(&ws->sq_head[desc.remote_rank], __ATOMIC_ACQUIRE) + 1u;
    const uint32_t cqe_seq = target_head - 1u;
    FakeUrmaCqe *cqe = &ws->scq_entries[desc.remote_rank][cqe_seq & (kFakeUrmaCqDepth - 1u)];
    for (uint32_t i = 1; i < 16; ++i) {
        cqe->dw[i] = 0;
    }
    __atomic_store_n(&cqe->dw[0], encode_fake_pending_cqe_dw0(cqe_seq), __ATOMIC_RELEASE);
    __atomic_store_n(&ws->sq_head[desc.remote_rank], target_head, __ATOMIC_RELEASE);
    __atomic_store_n(&ws->sq_submit_lock[desc.remote_rank], 0u, __ATOMIC_RELEASE);

    event.handle = encode_urma_event_handle(desc.remote_rank, target_head);
    event.engine = pto::comm::DmaEngine::URMA;
    return true;
}

}  // namespace pto2::urma_backend

inline __aicore__ bool UrmaFakeComplete(
    __gm__ uint8_t *workspace, uint32_t remote_rank, uint32_t target_head, uint8_t status = 0, uint8_t substatus = 0
) {
    return pto2::urma_backend::mark_fake_urma_cqe_ready(workspace, remote_rank, target_head, status, substatus);
}

inline __aicore__ void UrmaFakeReset(__gm__ uint8_t *workspace) {
    auto *ws = pto2::urma_backend::fake_workspace(workspace);
    __atomic_store_n(&ws->magic, 0u, __ATOMIC_RELEASE);
    pto2::urma_backend::ensure_fake_workspace_initialized(workspace);
}
#endif

template <typename DstTensor, typename SrcTensor>
inline __aicore__ bool send_request_entry(AsyncCtx &ctx, UrmaRequestDescriptor<DstTensor, SrcTensor> desc) {
#if defined(__CPU_SIM)
    pto2::urma_backend::FakeUrmaAsyncSession session;
    if (!pto2::urma_backend::build_fake_urma_session(desc.workspace, desc.remote_rank, session)) {
        pto2::detail::defer_error(ctx, PTO2_ERROR_ASYNC_COMPLETION_INVALID);
        return false;
    }

    pto2::urma_backend::FakeUrmaAsyncEvent event;
    if (!pto2::urma_backend::submit_fake_urma_request(desc, event)) {
        pto2::detail::defer_error(ctx, PTO2_ERROR_ASYNC_COMPLETION_INVALID);
        return false;
    }
    pto2::detail::register_urma_async_event(ctx, event, session, desc.workspace);
    pto2::detail::defer_flush(ctx);
    return true;
#elif defined(PTO_URMA_SUPPORTED)
    pto::comm::AsyncSession session;
    if (!pto::comm::BuildAsyncSession<pto::comm::DmaEngine::URMA>(desc.workspace, desc.remote_rank, session)) {
        pto2::detail::defer_error(ctx, PTO2_ERROR_ASYNC_COMPLETION_INVALID);
        return false;
    }

    pto::comm::AsyncEvent event;
    if (desc.op == UrmaOp::TGET) {
        event = pto::comm::TGET_ASYNC<pto::comm::DmaEngine::URMA>(desc.dst, desc.src, session);
    } else {
        event = pto::comm::TPUT_ASYNC<pto::comm::DmaEngine::URMA>(desc.dst, desc.src, session);
    }
    pto2::detail::register_urma_async_event(ctx, event, session, desc.workspace);
    pto2::detail::defer_flush(ctx);
    return true;
#else
    (void)desc;
    pto2::detail::defer_error(ctx, PTO2_ERROR_ASYNC_COMPLETION_INVALID);
    return false;
#endif
}

#endif  // SRC_A5_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_BACKEND_URMA_URMA_COMPLETION_KERNEL_H_
