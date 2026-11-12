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
#include <pto/pto-inst.hpp>

#include "backend/urma/urma_completion_kernel.h"
#include "common/platform_config.h"
#include "platform_comm/comm_context.h"
#include "tensor.h"
#include "../records.h"

namespace {
using Shape = pto::Shape<1, 1, 1, 1, pto::DYNAMIC>;
using Stride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1>;
using Data = pto::GlobalTensor<float, Shape, Stride>;

inline __aicore__ Data data(__gm__ float *ptr, uint32_t elems) {
    return Data(ptr, Shape(1, 1, 1, 1, elems), Stride(elems, elems, elems, elems, 1));
}

template <typename T>
inline __aicore__ __gm__ T *ptr(int64_t arg) {
    auto *tensor = reinterpret_cast<__gm__ Tensor *>(arg);
    return reinterpret_cast<__gm__ T *>(tensor->buffer.addr) + tensor->start_offset;
}

inline __aicore__ void flush(__gm__ urma_test::TransferRecord *record) {
    pto2::detail::defer_flush_range(record, sizeof(*record));
    dsb(DSB_DDR);
}

inline __aicore__ void fail(AsyncCtx &ctx, __gm__ urma_test::TransferRecord *record, int64_t status) {
    record->status = status;
    pto2::detail::set_urma_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
    record->error = *ctx.completion_error_code;
    flush(record);
    pto2::detail::defer_flush(ctx);
}
}  // namespace

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    AsyncCtx ctx = get_async_ctx(args);
    const uint32_t logical = get_block_idx(args);
    auto *record = reinterpret_cast<__gm__ urma_test::TransferRecord *>(ptr<int64_t>(args[3])) + logical;
    auto *comm = reinterpret_cast<__gm__ CommContext *>(args[4]);
    const uint32_t elems = args[5];
    const uint32_t slots = args[6];
    const uint32_t repeats = args[7];
    const uint32_t mode = args[8];
    if (comm == nullptr || comm->rankId != 0 || comm->rankNum != 2 || logical >= slots || mode > 2) {
        fail(ctx, record, -1);
        return;
    }
    const auto &workspace_desc = comm_workspace<CommWorkspaceKind::URMA>(*comm);
    auto *workspace = reinterpret_cast<__gm__ uint8_t *>(workspace_desc.addr);
    pto::comm::AsyncSession session;
    if (!simpler::urma::build_urma_session(ctx.urma_sender_id, workspace, 1, session)) {
        fail(ctx, record, -2);
        return;
    }
    auto *info = reinterpret_cast<__gm__ pto::comm::urma::UrmaInfo *>(workspace);
    const uint32_t jetties = info->jettiesPerCore;
    if (info->layout != pto::comm::urma::UrmaLayout::PER_PEER || jetties > urma_test::kMaxJetties) {
        fail(ctx, record, -3);
        return;
    }
    const uint32_t row_base = info->rowsPerPeer + ctx.urma_sender_id * jetties;
    auto *wq = reinterpret_cast<__gm__ pto::comm::urma::UrmaWQCtx *>(info->sqPtr) + row_base;
    auto *cq = reinterpret_cast<__gm__ pto::comm::urma::UrmaCqCtx *>(info->scqPtr) + row_base;
    record->sender = ctx.urma_sender_id;
    record->jetties = jetties;
    record->peer = 1;
    record->rows_per_peer = info->rowsPerPeer;
    record->cycle_hz = static_cast<int64_t>(PLATFORM_PROF_SYS_CNT_FREQ);
    record->workspace_bytes = workspace_desc.size;
    record->queue_rows = info->jettyCount;
    record->mode = mode;
    record->chunks = simpler::urma::chunk_count(static_cast<uint64_t>(elems) * sizeof(float));
    record->repeats = repeats;
    record->logical_sender = logical;
    for (uint32_t j = 0; j < jetties; ++j) {
        auto &queue = record->queues[j];
        queue.row = row_base + j;
        queue.sq_depth = wq[j].depth;
        queue.cq_depth = cq[j].depth;
        queue.head_before = simpler::urma::load_urma_counter(wq[j].headAddr);
        queue.bb_tail_before = simpler::urma::load_urma_counter(wq[j].tailAddr);
        queue.submitted_before = wq[j].submittedWqeCount;
        queue.cqe_tail_before = simpler::urma::load_urma_counter(cq[j].tailAddr);
        queue.target_bb = queue.head_before;
        queue.target_cqe = queue.submitted_before;
        queue.completion_count = 0;
    }
    auto *local_source = ptr<float>(args[0]) + static_cast<uint64_t>(logical) * elems;
    auto *local_peer_region = ptr<float>(mode == 0 ? args[0] : args[1]) + static_cast<uint64_t>(logical) * elems;
    const uint64_t remote_offset = reinterpret_cast<uint64_t>(local_peer_region) - comm->windowsIn[0];
    auto *remote = pto2::urma_backend::peer_mr_ptr<float>(workspace, 1, remote_offset);
    auto *out = ptr<float>(args[2]) + static_cast<uint64_t>(logical) * elems;
    auto src = data(mode == 1 ? local_source : remote, elems);
    auto dst = data(mode == 1 ? remote : out, elems);
    const uint32_t before = *ctx.completion_count;
    record->begin_cycles = get_sys_cnt();
    for (uint32_t rep = 0; rep < repeats; ++rep) {
        const bool ok = mode == 1 ? send_request_entry(ctx, UrmaTput(dst, src, workspace, 1)) :
                                    send_request_entry(ctx, UrmaTget(dst, src, workspace, 1));
        if (!ok) {
            fail(ctx, record, -4);
            return;
        }
    }
    record->post_cycles = get_sys_cnt();
    record->completion_count = *ctx.completion_count - before;
    for (uint32_t i = before; i < *ctx.completion_count; ++i) {
        const uint32_t row = ctx.completion_entries[i].addr >> 32;
        if (row < row_base || row >= row_base + jetties ||
            ctx.completion_entries[i].backend_cookie != reinterpret_cast<uint64_t>(workspace)) {
            fail(ctx, record, -5);
            return;
        }
        auto &queue = record->queues[row - row_base];
        queue.target_bb = static_cast<uint32_t>(ctx.completion_entries[i].addr);
        queue.target_cqe = ctx.completion_entries[i].expected_value;
        queue.completion_count += 1;
    }
    for (uint32_t j = 0; j < jetties; ++j) {
        record->queues[j].head_after = simpler::urma::load_urma_counter(wq[j].headAddr);
        record->queues[j].submitted_after = wq[j].submittedWqeCount;
    }
    record->error = *ctx.completion_error_code;
    record->status = 1;
    flush(record);
}
