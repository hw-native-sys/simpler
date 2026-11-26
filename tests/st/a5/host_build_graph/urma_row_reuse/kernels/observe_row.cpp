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

#include <pto/pto-inst.hpp>

#include "backend/urma/urma_completion.h"
#include "intrinsic.h"
#include "scheduler/wire_types.h"
#include "tensor.h"

namespace {

using namespace simpler::percounter;

struct GroupSample {
    uint32_t ref, row, bb, cqe, claimed, retired, pending;
};

struct RowSample {
    uint32_t phase, worker, sender_plus_one, rows_per_peer, registered, settled;
    GroupSample older, newer;
    uint32_t sq_tail, cq_tail, cq_doorbell;
};

// Each observer owns a separate 128-byte slot in the ordinary output tensor.
constexpr size_t sample_stride = 128;
static_assert(sizeof(RowSample) == 23 * sizeof(uint32_t) && sizeof(RowSample) <= sample_stride);

inline __aicore__ const __gm__ UrmaRunBinding *sample_group(
    __gm__ PercounterControl *control, uint32_t task, __gm__ GroupSample &sample, __gm__ uint32_t &sender_plus_one
) {
    auto *image = reinterpret_cast<__gm__ uint8_t *>(control);
    auto *counters = reinterpret_cast<__gm__ PercounterCounter *>(image + control->counters_offset);
    const uint64_t ref = percounter_gm_fetch_add(counters[task].calendar_defer_head, uint64_t{0});
    if (ref == 0 || ref > control->async_completion_capacity) return nullptr;
    auto *groups = reinterpret_cast<__gm__ PercounterAsyncCompletion *>(image + control->async_completions_offset);
    auto &group = groups[ref - 1];
    const auto owner = percounter_gm_fetch_add(group.calendar_defer_owner, uint64_t{0});
    auto &entry = group.entries[0];
    percounter_observe_cache_line(&entry);
    if (entry.engine != COMPLETION_ENGINE_URMA || entry.completion_type != COMPLETION_TYPE_URMA_EVENT_HANDLE)
        return nullptr;
    sender_plus_one = percounter_gm_read(group.urma_sender_plus_one);
    sample.ref = static_cast<uint32_t>(ref);
    sample.row = static_cast<uint32_t>(entry.addr >> 32);
    sample.bb = static_cast<uint32_t>(entry.addr);
    sample.cqe = entry.expected_value;
    sample.claimed = owner != 0;
    sample.retired = owner == UINT64_MAX;
    sample.pending = static_cast<uint32_t>(percounter_gm_fetch_add(group.pending_mask, uint64_t{0}));
    return find_urma_binding(control->urma, entry.backend_cookie);
}

}  // namespace

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    const uint32_t phase = static_cast<uint32_t>(args[1]);
    if (phase == 0) return;
    const auto global = load_global_context(args);
    auto *control = reinterpret_cast<__gm__ simpler::percounter::PercounterControl *>(global.async_image_address);
    auto *tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    auto *output = reinterpret_cast<__gm__ uint8_t *>(tensor->buffer.addr) + tensor->start_offset * sizeof(float);
    auto &sample = *reinterpret_cast<__gm__ RowSample *>(output + (phase - 1) * sample_stride);
    sample.worker = global.async_group_index;
    sample.registered = static_cast<uint32_t>(percounter_gm_fetch_add(control->async_registered, uint64_t{0}));
    sample.settled = static_cast<uint32_t>(percounter_gm_fetch_add(control->async_settled, uint64_t{0}));
    const auto *domain = sample_group(control, static_cast<uint32_t>(args[2]), sample.older, sample.sender_plus_one);
    if (!domain) return;
    if (phase > 1 &&
        domain != sample_group(control, static_cast<uint32_t>(args[3]), sample.newer, sample.sender_plus_one))
        return;
    sample.rows_per_peer = domain->rows_per_peer;
    using namespace pto::comm::urma;
    const auto *info = reinterpret_cast<const __gm__ UrmaInfo *>(domain->workspace);
    if (sample.older.row >= info->jettyCount) return;
    const auto &sq = reinterpret_cast<const __gm__ UrmaWQCtx *>(info->sqPtr)[sample.older.row];
    const auto &cq = reinterpret_cast<const __gm__ UrmaCqCtx *>(info->scqPtr)[sample.older.row];
    using simpler::percounter::CoreCqAccess;
    sample.sq_tail = CoreCqAccess::load(sq.tailAddr);
    sample.cq_tail = CoreCqAccess::load(cq.tailAddr);
    sample.cq_doorbell = CoreCqAccess::load(cq.dbAddr);
    sample.phase = phase;
    dcci(&sample, SINGLE_CACHE_LINE, CACHELINE_OUT);
    dcci(reinterpret_cast<__gm__ uint8_t *>(&sample) + 64, SINGLE_CACHE_LINE, CACHELINE_OUT);
    dsb(DSB_DDR);
}
