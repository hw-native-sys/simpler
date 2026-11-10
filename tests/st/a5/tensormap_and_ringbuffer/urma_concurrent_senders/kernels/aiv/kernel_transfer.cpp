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

namespace {
constexpr uint32_t kRecordWords = 32;
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

inline __aicore__ void flush(__gm__ int64_t *record) {
    pto2::detail::defer_flush_range(record, kRecordWords * sizeof(int64_t));
    dsb(DSB_DDR);
}
}  // namespace

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    AsyncCtx ctx = get_async_ctx(args);
    auto *comm = reinterpret_cast<__gm__ CommContext *>(args[4]);
    const uint32_t elems = args[5];
    const uint32_t slots = args[6];
    const uint32_t peers = args[7];
    const uint32_t repeats = args[8];
    const bool put = args[9] != 0;
    const bool mixed = args[14] == 2;
    const uint32_t local_block = mixed ? 2 * get_block_idx(args) + get_sub_block_id(args) : get_block_idx(args);
    const uint32_t logical = static_cast<uint32_t>(args[10]) + local_block;
    const uint32_t width = args[11];
    auto *barrier = reinterpret_cast<__gm__ int64_t *>(args[12]);
    const uint32_t wave = args[13];
    const bool independent = args[14] != 0;
    auto *record = ptr<int64_t>(args[3]) + local_block * kRecordWords;
    record[8] = get_sys_cnt();
    if (comm == nullptr || comm->rankId != 0 || peers == 0 || peers > 2 || comm->rankNum != peers + 1 ||
        logical >= slots) {
        record[0] = -1;
        flush(record);
        pto2::detail::set_urma_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
        pto2::detail::defer_flush(ctx);
        return;
    }
    auto *workspace = reinterpret_cast<__gm__ uint8_t *>(comm_workspace<CommWorkspaceKind::URMA>(*comm).addr);
    auto *info = reinterpret_cast<__gm__ pto::comm::urma::UrmaInfo *>(workspace);
    const uint32_t jetties = info->jettiesPerCore;
    if (info->layout != pto::comm::urma::UrmaLayout::PER_PEER || jetties == 0 || info->rankCount != comm->rankNum ||
        info->rowsPerPeer < jetties || info->rowsPerPeer % jetties != 0 ||
        ctx.urma_sender_id >= info->rowsPerPeer / jetties ||
        static_cast<uint64_t>(info->rankCount) * info->rowsPerPeer != info->jettyCount) {
        record[0] = -2;
        flush(record);
        pto2::detail::set_urma_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
        pto2::detail::defer_flush(ctx);
        return;
    }
    const uint32_t sender_base = ctx.urma_sender_id * jetties;
    const uint32_t row_base = info->rowsPerPeer + sender_base;
    auto *wq = reinterpret_cast<__gm__ pto::comm::urma::UrmaWQCtx *>(info->sqPtr) + row_base;
    auto *cq = reinterpret_cast<__gm__ pto::comm::urma::UrmaCqCtx *>(info->scqPtr) + row_base;
    auto *second_wq = peers == 2 ? wq + info->rowsPerPeer : nullptr;
    record[1] = ctx.urma_sender_id;
    record[2] = logical;
    record[3] = get_block_idx();
    record[4] = get_subblockid();
    record[5] = row_base;
    record[6] = jetties;
    record[10] = wq[0].depth;
    record[11] = cq[0].depth;
    record[12] = wq[0].submittedWqeCount;
    record[22] = second_wq ? second_wq->submittedWqeCount : 0;
    record[23] = info->rowsPerPeer;
    record[15] = static_cast<uint32_t>(info->layout);
    record[16] = pto2::urma_backend::load_urma_counter(wq[0].headAddr);
    record[19] = pto2::urma_backend::load_urma_counter(cq[0].tailAddr);
    record[20] = peers;
    record[0] = 2;
    flush(record);
    if (put && wave == 0 && width > 1) {
        const uint64_t timeout = static_cast<uint64_t>(PLATFORM_PROF_SYS_CNT_FREQ);
        const uint64_t begin = get_sys_cnt();
        bool arrived = false;
        while (!arrived && get_sys_cnt() - begin < timeout) {
            arrived = true;
            for (uint32_t i = 0; i < width; ++i) {
                if (ld_dev(reinterpret_cast<__gm__ uint64_t *>(barrier + i * kRecordWords), 0) == 0) {
                    arrived = false;
                }
            }
        }
        if (!arrived) {
            record[0] = -3;
            flush(record);
            pto2::detail::set_urma_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
            pto2::detail::defer_flush(ctx);
            return;
        }
    }

    auto *local = ptr<float>(args[0]);
    const uint64_t receive_offset = reinterpret_cast<uint64_t>(ptr<float>(args[1])) - comm->windowsIn[0] +
                                    static_cast<uint64_t>(logical) * elems * sizeof(float);
    auto *out = ptr<float>(args[2]);
    const uint32_t before = *ctx.completion_count;
    for (uint32_t rep = 0; rep < repeats; ++rep) {
        for (uint32_t peer = 1; peer <= peers; ++peer) {
            auto *remote = pto2::urma_backend::peer_mr_ptr<float>(workspace, peer, receive_offset);
            record[25 + peer] = reinterpret_cast<uint64_t>(remote);
            record[28 + peer] = reinterpret_cast<uint64_t>(pto2::urma_backend::peer_mr_ptr<float>(workspace, peer, 0));
            auto *peer_source = local + (static_cast<uint64_t>(peer - 1) * slots + logical) * elems;
            auto source = data(put ? peer_source : remote, elems);
            auto dest = data(put ? remote : out + (static_cast<uint64_t>(peer - 1) * slots + logical) * elems, elems);
            const uint32_t request_begin = *ctx.completion_count;
            const bool ok = put ? send_request_entry(ctx, UrmaTput(dest, source, workspace, peer)) :
                                  send_request_entry(ctx, UrmaTget(dest, source, workspace, peer));
            if (!ok) {
                record[0] = -4;
                flush(record);
                return;
            }
            const uint32_t peer_row = peer * info->rowsPerPeer + sender_base;
            for (uint32_t i = request_begin; i < *ctx.completion_count; ++i) {
                const uint32_t actual_row = ctx.completion_entries[i].addr >> 32;
                if (actual_row < peer_row || actual_row >= peer_row + jetties ||
                    ctx.completion_entries[i].backend_cookie != reinterpret_cast<uint64_t>(workspace)) {
                    record[0] = -5;
                    flush(record);
                    pto2::detail::set_urma_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
                    pto2::detail::defer_flush(ctx);
                    return;
                }
                if (i == request_begin) record[23 + peer] = actual_row;
            }
        }
    }
    record[7] = *ctx.completion_count - before;
    record[13] = wq[0].submittedWqeCount;
    record[14] = second_wq ? second_wq->submittedWqeCount : 0;
    record[17] = pto2::urma_backend::load_urma_counter(wq[0].headAddr);
    record[18] = get_sub_block_id(args);
    record[21] = independent;
    record[28] = receive_offset;
    record[31] = *ctx.completion_error_code;
    record[9] = get_sys_cnt();
    record[0] = 1;
    flush(record);
}
