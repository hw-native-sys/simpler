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

#include "backend/urma/urma_completion_kernel.h"

#include <pto/pto-inst.hpp>

#include "backend/urma/urma_completion.h"
#include "common/platform_config.h"
#include "intrinsic.h"
#include "platform_comm/comm_context.h"
#include "tensor.h"

namespace {

using namespace simpler::urma;
using simpler::percounter::CoreCqAccess;

struct QueueSample {
    uint32_t phase, sender, rows_per_peer, row, target_bb, target_cqe, registered;
    uint32_t head, submitted, sq_tail, cq_tail, doorbell, cqe_status;
};
constexpr uint32_t header_floats = 32;
constexpr uint32_t guard_floats = 32;
static_assert(sizeof(QueueSample) == 13 * sizeof(uint32_t) && sizeof(QueueSample) <= header_floats * sizeof(float));

// Capture the same PTO target as the deferred writer, without installing a Core completion condition.
struct Capture {
    CompletionTarget target{};
    inline __aicore__ bool capacity_available(uint64_t count) const { return count == 1; }
    inline __aicore__ bool register_target(const CompletionTarget &posted) {
        target = posted;
        return true;
    }
    inline __aicore__ void set_error(RequestError) {}
};

}  // namespace

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    auto ctx = get_async_ctx(args);
    auto *output = reinterpret_cast<__gm__ Tensor *>(args[2]);
    auto *comm = reinterpret_cast<__gm__ CommContext *>(args[4]);
    const auto &workspace = comm_workspace<CommWorkspaceKind::URMA>(*comm);
    const uint32_t peer = 1 - comm->rankId;
    const uint32_t count = static_cast<uint32_t>(args[5]) - header_floats - 2 * guard_floats;
    auto *root = reinterpret_cast<__gm__ uint8_t *>(workspace.addr);
    auto *remote = pto2::urma_backend::peer_mr_ptr<float>(root, peer, 0);
    auto *local = reinterpret_cast<__gm__ float *>(output->buffer.addr) + output->start_offset;
    auto *sample = reinterpret_cast<__gm__ QueueSample *>(local);
    const bool after_wait = CoreCqAccess::load(reinterpret_cast<uint64_t>(&sample->phase)) == 1;
    using Shape = pto::Shape<1, 1, 1, 1, pto::DYNAMIC>;
    using Stride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1>;
    using Data = pto::GlobalTensor<float, Shape, Stride>;
    const Shape shape(1, 1, 1, 1, count);
    const Stride stride(count, count, count, count, 1);
    if (!after_wait) {
        UrmaSession session;
        Capture capture;
        const auto desc = UrmaTget(
            Data(local + header_floats + guard_floats, shape, stride), Data(remote, shape, stride), workspace, peer
        );
        if (!build_urma_session(ctx, workspace, session, peer) || !preflight_urma_request(capture, session, 1, peer) ||
            !submit_urma_request_once(capture, desc, session))
            return;
        sample->sender = ctx.urma_sender_id;
        sample->row = capture.target.queue_row;
        sample->target_bb = capture.target.target_bb;
        sample->target_cqe = capture.target.target_cqe;
        sample->registered = *ctx.completion_count;
    }
    const auto *info = reinterpret_cast<const __gm__ pto::comm::urma::UrmaInfo *>(root);
    if (sample->row >= info->jettyCount) return;
    const auto &sq = reinterpret_cast<const __gm__ pto::comm::urma::UrmaWQCtx *>(info->sqPtr)[sample->row];
    const auto &cq = reinterpret_cast<const __gm__ pto::comm::urma::UrmaCqCtx *>(info->scqPtr)[sample->row];
    sample->rows_per_peer = info->rowsPerPeer;
    if (!after_wait) {
        // CQ memory belongs to the AIV channel. Copy observations to ordinary HBM.
        const uint32_t seq = sample->target_cqe - 1;
        const auto addr = cq.bufAddr + 64 * (seq & (cq.depth - 1));
        const uint64_t start = get_sys_cnt();
        do {
            sample->cqe_status = CoreCqAccess::load(addr);
            if (((sample->cqe_status >> 2) & 1u) != ((seq / cq.depth) & 1u)) break;
        } while (get_sys_cnt() - start < 2 * uint64_t{PLATFORM_PROF_SYS_CNT_FREQ});
    }
    sample->head = CoreCqAccess::load(sq.headAddr);
    sample->submitted = CoreCqAccess::load(reinterpret_cast<uint64_t>(&sq.submittedWqeCount));
    sample->sq_tail = CoreCqAccess::load(sq.tailAddr);
    sample->cq_tail = CoreCqAccess::load(cq.tailAddr);
    sample->doorbell = CoreCqAccess::load(cq.dbAddr);
    sample->phase = after_wait ? 2 : 1;
    pto2::detail::defer_flush_range(sample, sizeof(*sample));
    dsb(DSB_DDR);
}
