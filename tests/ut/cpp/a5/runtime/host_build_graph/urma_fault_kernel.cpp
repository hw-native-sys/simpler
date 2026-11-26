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

#define __CPU_SIM 1
#include "inner_kernel.h"
#include "runtime_types.h"

uint64_t get_aicore_report_epoch();
#include "lane_loop.h"
#include "async_context.h"
#include "../urma_fault_domain.h"
#include "../urma_request_fixture.h"
#include "backend/urma/urma_completion_kernel.h"

namespace simpler::test {
using namespace percounter;

void FaultDomain::run_kernel(uint64_t operand, bool put, UrmaFault fault) {
    struct Image {
        PercounterControl control{};
        pto2::resident::AsyncGroupData scratch{};
        DeferredCompletionEntry entries[4]{};
    } image;
    AsyncCtx ctx{};
    ctx.urma_sender_id = 0;
    ctx.resident_enabled = 1;
    ctx.resident = {reinterpret_cast<uint8_t *>(&image), sizeof(image), 1, offsetof(Image, scratch), 0};
    image.control.urma = {reinterpret_cast<uint64_t>(&binding), 1};
    image.scratch.run_epoch = 1;
    image.scratch.capacity = 4;
    image.scratch.entries_offset = offsetof(Image, entries);
    image.scratch.entries_bytes = sizeof(image.entries);
    info.jettiesPerCore = 1;
    info.gatherRuntimeBase = addr();
    info.gatherRuntimeStride = 128;

    using Shape = pto::Shape<1, 1, 1, 1, pto::DYNAMIC>;
    using Stride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1>;
    using Tensor = pto::GlobalTensor<uint8_t, Shape, Stride>;
    const Shape shape(1, 1, 1, 1, 16);
    const Stride stride(16, 16, 16, 16, 1);
    const Tensor local({operand}, shape, stride), peer({0x80000}, shape, stride);
    const CommWorkspace workspace{binding.workspace, binding.workspace_bytes};
    PostScript script{};
    script.post_from_queue = true;
    active_script = &script;
    const bool posted =
        send_request_entry(ctx, put ? UrmaTput(peer, local, workspace, 1) : UrmaTget(local, peer, workspace, 1));
    active_script = nullptr;
    ASSERT_TRUE(posted);
    ASSERT_EQ(script.submissions.size(), 1u);
    const auto &submission = script.submissions.front();
    ASSERT_EQ(submission.put, put);
    posted_operand = put ? submission.src : submission.dst;

    alignas(32) uint32_t done = 0;
    PercounterCounter counter{};
    counter.completion_notify_address = reinterpret_cast<uint64_t>(&done);
    counter.calendar_defer_bit = 1U << 24;
    CalendarExecutionExtra extra{};
    PercounterAsyncCompletion group{};
    LaneConfig lane{};
    lane.control = &image.control;
    lane.task_count = 1;
    lane.counters = &counter;
    lane.async_group = &image.scratch;
    lane.async_entries = image.entries;
    lane.async_completions = &group;
    lane.async_completion_capacity = 1;
    alignas(32) uint32_t lifecycle = 0;
    lane.lifecycle_notify = &lifecycle;
    lane.urma_sender_plus_one = 1;

    // The transport-only board probe leaves one actual GET for the Host to reap after chip exit.
    if (fault == UrmaFault::HostReceipt) return;
    bool deferred = false;
    if (fault == UrmaFault::Finish) {
        // The WQE above was accepted. The next reservation is unfinished at kernel return.
        const auto reservation = pto2::resident::reserve_conditions(ctx.resident, 1);
        ASSERT_FALSE(pto2::resident::finish_reservation(ctx.resident, reservation, 1));
        ASSERT_FALSE(percounter_calendar_seal(lane, &extra, false, nullptr, 0, &deferred));
        EXPECT_EQ(image.control.async_registered, 0u);
    } else {
        ASSERT_TRUE(percounter_calendar_seal(lane, &extra, false, nullptr, 0, &deferred));
        ASSERT_TRUE(deferred);
        if (fault == UrmaFault::BadCqe) set_cqe(row, 0, true, 1);
        uint64_t cursor = 0;
        if (fault == UrmaFault::BadCqe) {
            ASSERT_FALSE(percounter_calendar_service_defer(lane, 0, &cursor));
        } else {
            // A deterministic supervisor deadline after repeated pending observations.
            for (int poll = 0; poll < 4; ++poll) {
                ASSERT_TRUE(percounter_calendar_service_defer(lane, 0, &cursor));
                ASSERT_EQ(group.pending_mask, 1u);
            }
            percounter_fail(lane, SchedulerGraphResult::TIMEOUT);
        }
        EXPECT_EQ(image.control.async_settled, 0u);
    }
    done_count = done & CALENDAR_COUNT_MASK;
    lane_error = image.control.lane_error;
}

}  // namespace simpler::test
