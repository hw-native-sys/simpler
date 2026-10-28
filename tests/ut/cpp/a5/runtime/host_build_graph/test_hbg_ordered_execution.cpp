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

#include "inner_kernel.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <thread>
#include <tuple>
#include <vector>

#include "hbg_scheduler_test_support.h"
#include "scheduler/scheduler_admission.h"
#include "scheduler/scheduler_dispatch.h"
#include "scheduler/scheduler_protocol_execute.h"

namespace {

struct OrderedShape {
    uint16_t blocks;
    LaneMask mask;
    bool sync_start{false};
};

constexpr LaneMask kAicAiv0{/*aic=*/true, /*aiv0=*/true, /*aiv1=*/false};
constexpr LaneMask kAicAiv1{/*aic=*/true, /*aiv0=*/false, /*aiv1=*/true};
// The single-lane sets: one cube lane, then each vector lane.
constexpr std::array<LaneMask, 2> kPureCAndV{LaneMask::aic(), LaneMask::aiv0()};
constexpr std::array<LaneMask, 3> kPureMasks{LaneMask::aic(), LaneMask::aiv0(), LaneMask::aiv1()};

uint64_t maximum_blocks(const std::vector<OrderedShape> &shapes) {
    uint64_t maximum = 1;
    for (const auto &shape : shapes)
        maximum = std::max<uint64_t>(maximum, shape.blocks);
    return maximum;
}

SchedulerSpmdRequirements ordered_requirements(const std::vector<OrderedShape> &shapes) {
    SchedulerSpmdRequirements requirements{6, 2, 0, 0};
    for (const auto &shape : shapes)
        if (shape.blocks > 1) ++requirements.cooperative_task_count;
    return requirements;
}

struct OrderedExecution {
    int64_t task{SCHEDULER_TASK_REF_INVALID};
    uint32_t block{0};
    uint32_t blocks{0};
    uint32_t worker{0};
    uint32_t slot{0};
    uint32_t generation{0};
    uint64_t function{0};
};

struct OrderedFixture {
    explicit OrderedFixture(const std::vector<OrderedShape> &shapes, bool protocol_profiling = false) :
        storage(shapes.size(), 6, maximum_blocks(shapes), ordered_requirements(shapes), protocol_profiling),
        graph(shapes.size()) {
        uint32_t trace = 0;
        for (uint64_t task = 0; task < shapes.size(); ++task) {
            const auto shape = shapes[task];
            graph.executable(task, shape.mask == LaneMask::aic() ? 0 : 1);
            auto &metadata = storage.metadata[task];
            metadata.active_mask = shape.mask;
            metadata.logical_block_num = shape.blocks;
            metadata.flags = SCHEDULER_TASK_EXECUTABLE | (shape.blocks > 1 ? SCHEDULER_TASK_SPMD : 0);
            if (shape.sync_start) metadata.flags |= SCHEDULER_TASK_SYNC_START;
            metadata.trace_index_base = trace;
            trace += shape.blocks;
            metadata.timing_slot = -1;
            for (uint32_t slot = 0; slot < 3; ++slot)
                metadata.kernel_ids[slot] = shape.mask.has(static_cast<SubtaskSlot>(slot)) ? 1 : UINT16_MAX;
        }
        EXPECT_TRUE(scheduler_init_spmd_task_records(base(), storage.layout, 37));
        storage.run_control->active_worker_count = 6;
        storage.run_control->aic_active_worker_count = 2;
        storage.run_control->aiv_active_worker_count = 4;
        for (uint32_t worker = 0; worker < 6; ++worker) {
            auto &context = storage.contexts[worker];
            context.is_scheduler = 1;
            context.scheduler_count = 6;
            for (uint32_t lane = 0; lane < 3; ++lane)
                context.cluster_worker_ids[lane] = (worker / 3) * 3 + lane;
            storage.test_graphs[worker] = graph.graph();
        }
        for (uint32_t worker = 0; worker < 6; ++worker) {
            locals[worker] = storage.local_context(&storage.contexts[worker]);
            EXPECT_TRUE(scheduler_initialize_admission(base(), locals[worker], &admissions[worker]));
            locals[worker]->admission = &admissions[worker];
        }
    }

    void *base() const { return storage.scheduler_state->base(); }
    SchedulerRunControl *run() const { return storage.run_control; }
    SchedulerTaskControl *control(uint64_t task) const {
        return scheduler_state_at<SchedulerTaskControl>(base(), storage.layout.task_controls_offset) + task;
    }
    uint32_t publisher(uint64_t task) const { return storage.metadata[task].active_mask == LaneMask::aic() ? 0 : 1; }
    void dependency(uint64_t consumer, uint64_t producer) {
        graph.executable(
            consumer, storage.metadata[consumer].active_mask == LaneMask::aic() ? 0 : 1,
            {static_cast<int32_t>(producer)}
        );
        storage.metadata[consumer].flags |= SCHEDULER_TASK_HAS_FANIN;
        EXPECT_TRUE(scheduler_init_spmd_task_records(base(), storage.layout, 37));
    }
    bool enqueue(uint64_t task, uint32_t worker) {
        auto *local = locals[worker];
        if (scheduler_route_task(graph.graph(), base(), local, run(), task, nullptr) !=
            SchedulerRouteResult::READY_TO_ENQUEUE)
            return false;
        SchedulerReadyBatch batch{};
        const uint32_t queue =
            scheduler_task_ready_queue(storage.metadata[task].flags, storage.metadata[task].active_mask);
        return scheduler_ready_batch_append(base(), local, task, &batch, &ready_stats[worker]) &&
               scheduler_ready_batch_push(base(), local, queue, &batch, &ready_stats[worker]);
    }
    bool pop(uint64_t task, uint32_t worker, SchedulerReadyClaim *ready) {
        if (!enqueue(task, worker)) return false;
        const uint32_t queue =
            scheduler_task_ready_queue(storage.metadata[task].flags, storage.metadata[task].active_mask);
        *ready = {};
        ready->inbox_index = scheduler_ready_push_target(locals[worker], queue, task);
        return scheduler_ready_pop_from_inbox(
                   graph.graph(), base(), locals[worker], run(), queue, ready->inbox_index, &ready->task_ref,
                   &ready_stats[worker]
               ) &&
               ready->task_ref == static_cast<int64_t>(task);
    }
    bool admit(uint64_t task, uint32_t worker) {
        SchedulerReadyClaim ready{};
        return pop(task, worker, &ready) &&
               scheduler_try_claim_admission(
                   graph.graph(), base(), locals[worker], run(), &admissions[worker], ready
               ) == SchedulerAdmissionResult::PROGRESS &&
               scheduler_prepare_admission_record(graph.graph(), base(), locals[worker], run(), &admissions[worker]) &&
               scheduler_try_commit_admission(graph.graph(), base(), locals[worker], run(), &admissions[worker]) ==
                   SchedulerAdmissionResult::PROGRESS;
    }
    bool fill(uint32_t worker, uint64_t profiling = 0) {
        bool failed = false;
        scheduler_fill_own_slots(
            graph.graph(), base(), locals[worker], run(), cursors[worker].data(), &ready_stats[worker], profiling, 0,
            &failed, storage.ssbuf_region
        );
        return !failed;
    }
    bool stage_one(uint32_t worker, uint32_t slot) {
        auto *local = locals[worker];
        if (!scheduler_refresh_local_obligation(graph.graph(), base(), local, run(), &admissions[worker]) ||
            scheduler_claim_ordered_block(graph.graph(), base(), local, run(), slot) !=
                SchedulerBlockClaimResult::CLAIMED)
            return false;
        SchedulerReadyClaim ready{};
        ready.task_ref = local->slots[slot].task_ref;
        ready.source = SchedulerReadySource::COOPERATIVE;
        return scheduler_fill_dispatch_slot(
            graph.graph(), base(), local, run(), {worker, slot, local->slots[slot].generation, local->config.self_lane},
            ready, 0, storage.ssbuf_region
        );
    }
    bool begin(uint32_t worker, OrderedExecution *execution) {
        auto *local = locals[worker];
        uint32_t slot = UINT32_MAX;
        uint64_t publication = 0;
        if (!scheduler_local_ready_pop(local, &slot, &publication)) return false;
        bool permit = false;
        if (!scheduler_ordered_can_enter_slot(graph.graph(), base(), local, run(), slot, &permit)) return false;
        if (!permit) {
            local->local_ready_mask |= static_cast<uint8_t>(1U << slot);
            return false;
        }
        auto *payload =
            scheduler_state_at<DispatchPayload>(base(), local->dispatch_payload_offset(local->config.self_lane, slot));
        if (scheduler_protocol_enabled(local)) scheduler_execute_protocol_task(base(), local, slot, payload);
        *execution = {
            local->slots[slot].task_ref,
            static_cast<uint32_t>(payload->local_context.block_idx),
            static_cast<uint32_t>(payload->local_context.block_num),
            worker,
            slot,
            scheduler_dispatch_generation(publication),
            payload->function_bin_addr
        };
        return true;
    }
    bool complete(const OrderedExecution &execution) {
        auto *local = locals[execution.worker];
        local->local_completed_generations[execution.slot] = execution.generation;
        uint64_t refilled = 0;
        return scheduler_service_cluster_completions(
            graph.graph(), base(), local, run(), nullptr, &ready_stats[execution.worker], nullptr, 0, &refilled,
            storage.ssbuf_region
        );
    }
    bool close(uint32_t worker, bool *closed) {
        return scheduler_close_ordered_obligation(graph.graph(), base(), locals[worker], run(), closed);
    }

    scheduler_test::FixtureStorage storage;
    scheduler_test::GraphBuffer graph;
    std::array<SchedulerAdmissionLocalState, 6> admissions{};
    std::array<SchedulerLocalState *, 6> locals{};
    std::array<std::array<uint64_t, SCHEDULER_READY_QUEUE_COUNT>, 6> cursors{};
    std::array<SchedulerReadyStats, 6> ready_stats{};
};

class HbgOrderedFirstReady : public ::testing::TestWithParam<LaneMask> {};

TEST_P(HbgOrderedFirstReady, FirstBlockIsExecutableWithoutClaimingOrPublishingASecondBlock) {
    OrderedFixture fixture({{8, GetParam()}}, true);
    const uint32_t worker = fixture.publisher(0);
    ASSERT_TRUE(fixture.admit(0, worker));
    ASSERT_TRUE(fixture.fill(worker));
    EXPECT_EQ(fixture.control(0)->claim_tickets, 1u);
    EXPECT_EQ(fixture.control(0)->next_block, 1u);
    EXPECT_EQ(fixture.locals[worker]->local_ready_mask, 1u);
    EXPECT_EQ(fixture.locals[worker]->slots[1].state, SchedulerDispatchSlotState::FREE);
    const auto *blocks =
        scheduler_state_at<SchedulerProtocolBlockTrace>(fixture.base(), fixture.storage.layout.protocol.blocks_offset);
    const auto *events = scheduler_state_at<SchedulerProtocolExecutionTrace>(
        fixture.base(), fixture.storage.layout.protocol.executions_offset
    );
    EXPECT_NE(blocks[0].owner.claim_after.sequence, 0u);
    EXPECT_EQ(blocks[1].owner.claim_before.sequence, 0u);
    EXPECT_EQ(events[1].publication_count, 0u);
    OrderedExecution execution{};
    ASSERT_TRUE(fixture.begin(worker, &execution));
    EXPECT_EQ(execution.block, 0u);
    EXPECT_NE(events[0].permit.sequence, 0u);
    EXPECT_EQ(blocks[1].owner.claim_before.sequence, 0u);
    EXPECT_EQ(events[1].publication_count, 0u);
}

TEST_P(HbgOrderedFirstReady, ReadySecondSlotRunsBeforeTheFreeFirstSlotIsRefilled) {
    OrderedFixture fixture({{8, GetParam()}}, true);
    const uint32_t worker = fixture.publisher(0);
    auto *local = fixture.locals[worker];
    ASSERT_TRUE(fixture.admit(0, worker));
    ASSERT_TRUE(fixture.stage_one(worker, 1));
    const uint32_t generation = local->slots[1].generation;
    for (uint32_t poll = 0; poll < 8; ++poll) {
        ASSERT_TRUE(fixture.fill(worker));
        EXPECT_EQ(fixture.control(0)->claim_tickets, 1u);
        EXPECT_EQ(fixture.control(0)->next_block, 1u);
        EXPECT_EQ(local->local_ready_mask, 2u);
        EXPECT_EQ(local->slots[0].state, SchedulerDispatchSlotState::FREE);
        EXPECT_EQ(local->slots[1].generation, generation);
    }
    OrderedExecution execution{};
    ASSERT_TRUE(fixture.begin(worker, &execution));
    EXPECT_EQ(execution.slot, 1u);
    EXPECT_EQ(execution.block, 0u);
    ASSERT_TRUE(fixture.complete(execution));
    ASSERT_TRUE(fixture.fill(worker));
    EXPECT_EQ(fixture.control(0)->next_block, 2u);
    EXPECT_EQ(fixture.control(0)->completed_blocks, 1u);
    EXPECT_EQ(local->local_ready_mask, 1u);
    EXPECT_EQ(local->slots[0].state, SchedulerDispatchSlotState::READY);
    EXPECT_EQ(local->slots[0].block_idx, 1u);
}

TEST_P(HbgOrderedFirstReady, CompletionAllowsTheNextClaimWithoutLosingOrDuplicatingBlocks) {
    constexpr uint16_t count = 8;
    OrderedFixture fixture({{count, GetParam()}});
    const uint32_t worker = fixture.publisher(0);
    ASSERT_TRUE(fixture.admit(0, worker));
    for (uint32_t block = 0; block < count; ++block) {
        ASSERT_TRUE(fixture.fill(worker));
        EXPECT_EQ(fixture.control(0)->next_block, block + 1);
        OrderedExecution execution{};
        ASSERT_TRUE(fixture.begin(worker, &execution));
        EXPECT_EQ(execution.task, 0);
        EXPECT_EQ(execution.block, block);
        EXPECT_EQ(execution.blocks, count);
        ASSERT_TRUE(fixture.complete(execution));
        EXPECT_EQ(fixture.control(0)->completed_blocks, block + 1);
    }
    EXPECT_EQ(scheduler_gm_load(fixture.control(0)->state), static_cast<int64_t>(SchedulerTaskState::DONE));
    ASSERT_TRUE(fixture.fill(worker));
    OrderedExecution extra{};
    EXPECT_FALSE(fixture.begin(worker, &extra));
    EXPECT_EQ(fixture.control(0)->next_block, count);
    EXPECT_EQ(fixture.control(0)->completed_blocks, count);
    EXPECT_EQ(fixture.run()->scheduler_error, 0u);
}

TEST_P(HbgOrderedFirstReady, ReadyBlockDoesNotSkipDiscoveryOrPublicationOfALaterAdmission) {
    OrderedFixture fixture({{8, GetParam()}, {2, GetParam()}}, true);
    const uint32_t worker = fixture.publisher(0);
    ASSERT_TRUE(fixture.admit(0, worker));
    ASSERT_TRUE(fixture.stage_one(worker, 1));
    ASSERT_TRUE(fixture.enqueue(1, worker));
    ASSERT_TRUE(fixture.fill(worker));
    const auto *shared =
        scheduler_state_at<SchedulerAdmissionControl>(fixture.base(), fixture.storage.layout.admission_control_offset);
    EXPECT_EQ(shared->commit_tail, 2u);
    EXPECT_EQ(
        scheduler_gm_query(fixture.control(1)->admission_state),
        static_cast<uint64_t>(SchedulerAdmissionState::ADMITTED)
    );
    EXPECT_EQ(fixture.admissions[worker].pending_task_ref, SCHEDULER_TASK_REF_INVALID);
    EXPECT_EQ(fixture.admissions[worker].current_task_ref, 0);
    EXPECT_EQ(fixture.control(0)->next_block, 1u);
    EXPECT_EQ(fixture.control(1)->next_block, 0u);
    OrderedExecution execution{};
    ASSERT_TRUE(fixture.begin(worker, &execution));
    EXPECT_EQ(execution.task, 0);
    EXPECT_EQ(execution.slot, 1u);
}

INSTANTIATE_TEST_SUITE_P(PureCAndV, HbgOrderedFirstReady, ::testing::ValuesIn(kPureCAndV));

class HbgOrderedExhaustionSafety : public ::testing::TestWithParam<LaneMask> {};

TEST_P(HbgOrderedExhaustionSafety, ZeroWorkMemberClosesWhileRemoteClaimedBlocksRemainLive) {
    const LaneMask mask = GetParam();
    OrderedFixture fixture({{2, mask}}, true);
    const uint32_t owner = fixture.publisher(0);
    const uint32_t zero_worker = mask == LaneMask::aic() ? 3 : 2;
    ASSERT_TRUE(fixture.admit(0, owner));
    ASSERT_TRUE(fixture.stage_one(owner, 0));
    ASSERT_TRUE(fixture.stage_one(owner, 1));
    ASSERT_EQ(fixture.control(0)->next_block, 2u);
    const uint64_t tickets = fixture.control(0)->claim_tickets;
    ASSERT_TRUE(fixture.fill(zero_worker));
    EXPECT_EQ(fixture.admissions[zero_worker].current_task_ref, SCHEDULER_TASK_REF_INVALID);
    EXPECT_EQ(fixture.admissions[zero_worker].scan_cursor, 1u);
    EXPECT_EQ(fixture.locals[zero_worker]->local_ready_mask, 0u);
    EXPECT_EQ(fixture.control(0)->claim_tickets, tickets);
    EXPECT_EQ(fixture.control(0)->completed_blocks, 0u);
    EXPECT_EQ(scheduler_gm_load(fixture.control(0)->state), static_cast<int64_t>(SchedulerTaskState::BLOCKED));
    const auto *members = scheduler_state_at<SchedulerProtocolMemberTrace>(
        fixture.base(), fixture.storage.layout.protocol.members_offset
    );
    EXPECT_NE(members[zero_worker].closed.sequence, 0u);
    EXPECT_EQ(members[zero_worker].closed_cursor, 2u);
    EXPECT_EQ(members[zero_worker].pending_mask, 0u);
    EXPECT_EQ(fixture.run()->scheduler_error, 0u);
}

TEST_P(HbgOrderedExhaustionSafety, ExhaustionCannotCloseAnActiveSlotWithItsReadyBitAlreadyConsumed) {
    const LaneMask mask = GetParam();
    OrderedFixture fixture({{2, mask}});
    const uint32_t worker = fixture.publisher(0);
    const uint32_t peer = mask == LaneMask::aic() ? 3 : 2;
    ASSERT_TRUE(fixture.admit(0, worker));
    ASSERT_TRUE(fixture.stage_one(worker, 1));
    ASSERT_TRUE(fixture.stage_one(peer, 0));
    OrderedExecution execution{};
    ASSERT_TRUE(fixture.begin(worker, &execution));
    ASSERT_EQ(fixture.locals[worker]->local_ready_mask, 0u);
    ASSERT_EQ(fixture.locals[worker]->slots[1].state, SchedulerDispatchSlotState::READY);
    ASSERT_TRUE(fixture.fill(worker));
    EXPECT_EQ(fixture.admissions[worker].current_task_ref, 0);
    EXPECT_EQ(fixture.admissions[worker].scan_cursor, 0u);
    EXPECT_EQ(fixture.control(0)->completed_blocks, 0u);
    ASSERT_TRUE(fixture.complete(execution));
    ASSERT_TRUE(fixture.fill(worker));
    EXPECT_EQ(fixture.admissions[worker].current_task_ref, SCHEDULER_TASK_REF_INVALID);
    EXPECT_EQ(fixture.admissions[worker].scan_cursor, 1u);
    EXPECT_EQ(fixture.control(0)->completed_blocks, 1u);
    EXPECT_EQ(scheduler_gm_load(fixture.control(0)->state), static_cast<int64_t>(SchedulerTaskState::BLOCKED));
}

TEST_P(HbgOrderedExhaustionSafety, PositiveHintCannotBypassCursorShapeOrErrorValidation) {
    const LaneMask mask = GetParam();
    for (uint32_t fault = 0; fault < 4; ++fault) {
        SCOPED_TRACE(fault);
        OrderedFixture fixture({{2, mask}});
        const uint32_t worker = fixture.publisher(0);
        ASSERT_TRUE(fixture.admit(0, worker));
        ASSERT_TRUE(scheduler_refresh_local_obligation(
            fixture.graph.graph(), fixture.base(), fixture.locals[worker], fixture.run(), &fixture.admissions[worker]
        ));
        scheduler_gm_store(fixture.control(0)->next_block, fault == 0 ? UINT64_C(3) : UINT64_C(2));
        if (fault == 1) fixture.admissions[worker].current_active_mask = LaneMask::none();
        if (fault == 2)
            fixture.admissions[worker].current_active_mask =
                mask == LaneMask::aic() ? LaneMask::aiv0() : LaneMask::aic();
        if (fault == 3)
            scheduler_record_error(
                fixture.run(), 0, SchedulerGraphResult::TIMEOUT, nullptr, fixture.locals[3],
                SchedulerErrorSite::ORDERED_EXIT_TIMEOUT
            );
        EXPECT_FALSE(fixture.fill(worker));
        EXPECT_EQ(fixture.admissions[worker].current_task_ref, 0);
        EXPECT_EQ(fixture.admissions[worker].scan_cursor, 0u);
        EXPECT_EQ(fixture.control(0)->completed_blocks, 0u);
        EXPECT_EQ(fixture.locals[worker]->local_ready_mask, 0u);
        EXPECT_EQ(
            fixture.run()->error_site,
            static_cast<uint64_t>(
                fault == 3 ? SchedulerErrorSite::ORDERED_EXIT_TIMEOUT : SchedulerErrorSite::BLOCK_CURSOR_INVALID
            )
        );
    }
}

TEST_P(HbgOrderedExhaustionSafety, InvalidTailAndPartiallyClaimedSyncStartKeepDifferentCloseRules) {
    const LaneMask mask = GetParam();
    {
        OrderedFixture fixture({{3, mask}});
        const uint32_t worker = fixture.publisher(0);
        ASSERT_TRUE(fixture.admit(0, worker));
        ASSERT_TRUE(scheduler_refresh_local_obligation(
            fixture.graph.graph(), fixture.base(), fixture.locals[worker], fixture.run(), &fixture.admissions[worker]
        ));
        scheduler_gm_store(fixture.control(0)->claim_tickets, UINT64_C(3));
        scheduler_gm_store(fixture.control(0)->next_block, UINT64_C(2));
        uint32_t block = UINT32_MAX;
        ASSERT_EQ(
            scheduler_claim_ordered_cursor(
                fixture.graph.graph(), fixture.base(), fixture.locals[worker], fixture.run(), &block
            ),
            SchedulerBlockClaimResult::EXHAUSTED
        );
        ASSERT_TRUE(fixture.admissions[worker].ticket_exhausted);
        ASSERT_TRUE(fixture.fill(worker));
        EXPECT_EQ(fixture.admissions[worker].current_task_ref, 0);
        EXPECT_EQ(fixture.admissions[worker].scan_cursor, 0u);
        scheduler_gm_fetch_add(fixture.control(0)->next_block, UINT64_C(1));
        ASSERT_TRUE(fixture.fill(worker));
        EXPECT_EQ(fixture.admissions[worker].current_task_ref, SCHEDULER_TASK_REF_INVALID);
        EXPECT_EQ(fixture.admissions[worker].scan_cursor, 1u);
        EXPECT_EQ(fixture.control(0)->completed_blocks, 0u);
    }
    {
        const uint16_t count = mask == LaneMask::aic() ? 2 : 4;
        OrderedFixture fixture({{count, mask, true}});
        const uint32_t worker = fixture.publisher(0);
        ASSERT_TRUE(fixture.admit(0, worker));
        ASSERT_TRUE(fixture.fill(worker));
        OrderedExecution execution{};
        ASSERT_TRUE(fixture.begin(worker, &execution));
        ASSERT_TRUE(fixture.complete(execution));
        ASSERT_TRUE(fixture.fill(worker));
        EXPECT_EQ(fixture.admissions[worker].current_task_ref, SCHEDULER_TASK_REF_INVALID);
        EXPECT_EQ(fixture.admissions[worker].scan_cursor, 1u);
        EXPECT_EQ(fixture.control(0)->next_block, 1u);
        EXPECT_EQ(fixture.control(0)->claim_tickets, 0u);
        EXPECT_EQ(fixture.control(0)->completed_blocks, 1u);
        EXPECT_EQ(scheduler_gm_load(fixture.control(0)->state), static_cast<int64_t>(SchedulerTaskState::BLOCKED));
    }
}

INSTANTIATE_TEST_SUITE_P(PureCAndV, HbgOrderedExhaustionSafety, ::testing::ValuesIn(kPureCAndV));

class HbgOrderedSyncStart : public ::testing::TestWithParam<LaneMask> {};

TEST_P(HbgOrderedSyncStart, TwoPendingSlotsAndEarlyReturnDoNotClaimAnotherPosition) {
    const LaneMask mask = GetParam();
    const uint32_t first = mask == LaneMask::aic() ? 0 : 1;
    OrderedFixture fixture({{2, mask, true}});
    ASSERT_TRUE(fixture.admit(0, first));
    ASSERT_TRUE(fixture.fill(first));
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.control(0)->next_block, UINT64_C(0)), 1u);
    OrderedExecution execution{};
    ASSERT_TRUE(fixture.begin(first, &execution));
    EXPECT_EQ(execution.block, 0u);
    EXPECT_EQ(execution.blocks, 2u);
    bool closed = true;
    ASSERT_TRUE(fixture.close(first, &closed));
    EXPECT_FALSE(closed);
    ASSERT_TRUE(fixture.complete(execution));
    uint32_t block = UINT32_MAX;
    EXPECT_EQ(
        scheduler_claim_ordered_cursor(
            fixture.graph.graph(), fixture.base(), fixture.locals[first], fixture.run(), &block
        ),
        SchedulerBlockClaimResult::RETRY
    );
    EXPECT_EQ(block, UINT32_MAX);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.control(0)->next_block, UINT64_C(0)), 1u);
    ASSERT_TRUE(fixture.close(first, &closed));
    EXPECT_TRUE(closed);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.control(0)->completed_blocks, UINT64_C(0)), 1u);
    EXPECT_EQ(scheduler_gm_load(fixture.control(0)->state), static_cast<int64_t>(SchedulerTaskState::BLOCKED));
}

TEST_P(HbgOrderedSyncStart, FastMemberEntersNextFullTaskWithoutStealingLateMembersPosition) {
    const LaneMask mask = GetParam();
    const std::vector<uint32_t> members =
        mask == LaneMask::aic() ? std::vector<uint32_t>{0, 3} : std::vector<uint32_t>{1, 2, 4, 5};
    const auto blocks = static_cast<uint16_t>(members.size());
    OrderedFixture fixture({{blocks, mask, true}, {blocks, mask, true}});
    ASSERT_TRUE(fixture.admit(0, members[0]));
    ASSERT_TRUE(fixture.fill(members[0]));
    // Publishing another task does not retire the current task's position.
    ASSERT_TRUE(fixture.admit(1, members[0]));
    OrderedExecution first{};
    ASSERT_TRUE(fixture.begin(members[0], &first));
    ASSERT_EQ(first.task, 0);
    ASSERT_TRUE(fixture.complete(first));
    bool closed = false;
    ASSERT_TRUE(fixture.close(members[0], &closed));
    ASSERT_TRUE(closed);
    ASSERT_TRUE(fixture.fill(members[0]));
    OrderedExecution next{};
    ASSERT_TRUE(fixture.begin(members[0], &next));
    EXPECT_EQ(next.task, 1);
    EXPECT_EQ(next.block, 0u);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.control(0)->next_block, UINT64_C(0)), 1u);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.control(0)->completed_blocks, UINT64_C(0)), 1u);
    for (uint32_t index = 1; index < members.size(); ++index) {
        ASSERT_TRUE(fixture.fill(members[index]));
        OrderedExecution late{};
        ASSERT_TRUE(fixture.begin(members[index], &late));
        EXPECT_EQ(late.task, 0);
        EXPECT_EQ(late.block, index);
        ASSERT_TRUE(fixture.complete(late));
        ASSERT_TRUE(fixture.close(members[index], &closed));
        ASSERT_TRUE(closed);
        ASSERT_TRUE(fixture.fill(members[index]));
        OrderedExecution successor{};
        ASSERT_TRUE(fixture.begin(members[index], &successor));
        EXPECT_EQ(successor.task, 1);
        EXPECT_EQ(successor.block, index);
        ASSERT_TRUE(fixture.complete(successor));
    }
    EXPECT_EQ(scheduler_gm_load(fixture.control(0)->state), static_cast<int64_t>(SchedulerTaskState::DONE));
    EXPECT_EQ(scheduler_gm_load(fixture.control(1)->state), static_cast<int64_t>(SchedulerTaskState::BLOCKED));
    ASSERT_TRUE(fixture.complete(next));
    EXPECT_EQ(scheduler_gm_load(fixture.control(1)->state), static_cast<int64_t>(SchedulerTaskState::DONE));
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.run()->scheduler_error, UINT64_C(0)), 0u);
}

TEST_P(HbgOrderedSyncStart, OrdinarySuccessorContinuesClaimingAfterSyncPositionRetires) {
    const LaneMask mask = GetParam();
    const uint32_t worker = mask == LaneMask::aic() ? 0 : 1;
    OrderedFixture fixture({{2, mask, true}, {3, mask}});
    ASSERT_TRUE(fixture.admit(0, worker));
    ASSERT_TRUE(fixture.admit(1, worker));
    ASSERT_TRUE(fixture.fill(worker));
    OrderedExecution execution{};
    ASSERT_TRUE(fixture.begin(worker, &execution));
    ASSERT_TRUE(fixture.complete(execution));
    bool closed = false;
    ASSERT_TRUE(fixture.close(worker, &closed));
    ASSERT_TRUE(closed);
    ASSERT_TRUE(fixture.fill(worker));
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.control(1)->next_block, UINT64_C(0)), 1u);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.control(0)->next_block, UINT64_C(0)), 1u);
    for (uint32_t index = 0; index < 3; ++index) {
        ASSERT_TRUE(fixture.fill(worker));
        ASSERT_TRUE(fixture.begin(worker, &execution));
        EXPECT_EQ(execution.task, 1);
        EXPECT_EQ(execution.block, index);
        ASSERT_TRUE(fixture.complete(execution));
    }
}

INSTANTIATE_TEST_SUITE_P(PureCAndV, HbgOrderedSyncStart, ::testing::ValuesIn(kPureCAndV));

TEST(HbgOrderedExecution, SyncStartZeroWorkMemberWaitsForAllPositionsToBeClaimedNotCompleted) {
    OrderedFixture fixture({{2, LaneMask::aiv0(), true}});
    ASSERT_TRUE(fixture.admit(0, 1));
    ASSERT_TRUE(fixture.fill(1));
    ASSERT_TRUE(scheduler_refresh_local_obligation(
        fixture.graph.graph(), fixture.base(), fixture.locals[5], fixture.run(), &fixture.admissions[5]
    ));
    bool closed = true;
    ASSERT_TRUE(fixture.close(5, &closed));
    EXPECT_FALSE(closed);
    ASSERT_TRUE(fixture.fill(2));
    ASSERT_TRUE(fixture.close(5, &closed));
    EXPECT_TRUE(closed);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.control(0)->next_block, UINT64_C(0)), 2u);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.control(0)->completed_blocks, UINT64_C(0)), 0u);
}

class HbgOrderedBlockIdentity : public ::testing::TestWithParam<std::tuple<LaneMask, uint16_t>> {};

TEST_P(HbgOrderedBlockIdentity, RealDispatchAndCompletionExecuteEveryOriginalBlockExactlyOnce) {
    const auto [mask, blocks] = GetParam();
    OrderedFixture fixture({{blocks, mask}});
    ASSERT_TRUE(fixture.enqueue(0, fixture.publisher(0)));
    std::vector<uint32_t> visits(blocks, 0);
    uint64_t executions = 0;
    for (uint64_t step = 0; step < static_cast<uint64_t>(blocks) * 16 + 32 && executions < blocks; ++step) {
        for (uint32_t worker = 0; worker < 6; ++worker) {
            ASSERT_TRUE(fixture.fill(worker));
            OrderedExecution execution{};
            if (!fixture.begin(worker, &execution)) continue;
            ASSERT_EQ(execution.task, 0);
            ASSERT_EQ(execution.blocks, blocks);
            ASSERT_LT(execution.block, blocks);
            ASSERT_NE(execution.function, 0u);
            ++visits[execution.block];
            ++executions;
            ASSERT_TRUE(fixture.complete(execution));
        }
    }
    EXPECT_EQ(executions, blocks);
    for (uint32_t visits_for_block : visits)
        EXPECT_EQ(visits_for_block, 1u);
    EXPECT_EQ(scheduler_gm_load(fixture.control(0)->state), static_cast<int64_t>(SchedulerTaskState::DONE));
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.control(0)->next_block, UINT64_C(0)), blocks > 1 ? blocks : 0);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.control(0)->completed_blocks, UINT64_C(0)), blocks > 1 ? blocks : 0);
    for (uint32_t worker = 0; worker < 6; ++worker) {
        ASSERT_TRUE(fixture.fill(worker));
        scheduler_flush_completions(fixture.run(), fixture.locals[worker]);
    }
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.run()->resolved_task_count, UINT64_C(0)), 1u);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.run()->scheduler_error, UINT64_C(0)), 0u);
}

INSTANTIATE_TEST_SUITE_P(
    PureCAndVCapacityBoundaries, HbgOrderedBlockIdentity,
    ::testing::Combine(
        ::testing::ValuesIn(kPureCAndV), ::testing::Values<uint16_t>(1, 2, 3, 7, 31, 32, 33, 63, 64, 65, 129, 512)
    )
);

TEST(HbgOrderedExecution, PartialCompletionDoesNotReleaseFaninAndLastGenerationReleasesExactlyOnce) {
    OrderedFixture fixture({{3, LaneMask::aic()}, {1, LaneMask::aic()}});
    fixture.dependency(1, 0);
    ASSERT_EQ(
        scheduler_route_task(fixture.graph.graph(), fixture.base(), fixture.locals[0], fixture.run(), 1, nullptr),
        SchedulerRouteResult::WAITING
    );
    ASSERT_TRUE(fixture.admit(0, 0));
    ASSERT_TRUE(fixture.stage_one(0, 0));
    OrderedExecution slow{};
    ASSERT_TRUE(fixture.begin(0, &slow));
    ASSERT_TRUE(fixture.fill(3));
    for (uint32_t index = 0; index < 2; ++index) {
        ASSERT_TRUE(fixture.fill(3));
        OrderedExecution fast{};
        ASSERT_TRUE(fixture.begin(3, &fast));
        ASSERT_EQ(fast.task, 0);
        ASSERT_TRUE(fixture.complete(fast));
    }
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.control(0)->completed_blocks, UINT64_C(0)), 2u);
    EXPECT_EQ(scheduler_gm_load(fixture.control(0)->state), static_cast<int64_t>(SchedulerTaskState::BLOCKED));
    EXPECT_EQ(scheduler_gm_load(fixture.control(0)->wake_list_head), 1);
    bool closed = false;
    ASSERT_TRUE(fixture.close(3, &closed));
    EXPECT_TRUE(closed);
    ASSERT_TRUE(fixture.fill(3));
    OrderedExecution premature{};
    EXPECT_FALSE(fixture.begin(3, &premature));
    EXPECT_EQ(fixture.locals[3]->pending_completed, 0u);
    ASSERT_TRUE(fixture.complete(slow));
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.control(0)->completed_blocks, UINT64_C(0)), 3u);
    EXPECT_EQ(scheduler_gm_load(fixture.control(0)->state), static_cast<int64_t>(SchedulerTaskState::DONE));
    EXPECT_EQ(scheduler_gm_load(fixture.control(0)->wake_list_head), SCHEDULER_WAKE_LIST_CLOSED);
    EXPECT_EQ(fixture.locals[0]->pending_completed, 1u);
    EXPECT_FALSE(scheduler_service_cluster_completion_slot(
        fixture.graph.graph(), fixture.base(), fixture.locals[0], fixture.run(), 0, slow.slot, slow.generation, nullptr,
        nullptr, nullptr, 0, nullptr, nullptr, fixture.storage.ssbuf_region
    ));
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.control(0)->completed_blocks, UINT64_C(0)), 3u);
    EXPECT_EQ(fixture.locals[0]->pending_completed, 1u);
    uint32_t retained = 0;
    for (const auto &candidate : fixture.admissions[0].candidates)
        if (candidate.ready.task_ref == 1) ++retained;
    EXPECT_EQ(retained, 1u);
    ASSERT_TRUE(fixture.fill(0));
    ASSERT_TRUE(fixture.fill(0));
    OrderedExecution successor{};
    ASSERT_TRUE(fixture.begin(0, &successor));
    EXPECT_EQ(successor.task, 1);
    ASSERT_TRUE(fixture.complete(successor));
    EXPECT_EQ(fixture.locals[0]->pending_completed, 2u);
}

TEST(HbgOrderedExecution, LocalDoneAllowsIndependentBWhileARemoteLastBlockIsStillExecuting) {
    OrderedFixture fixture({{3, LaneMask::aic()}, {2, LaneMask::aic()}});
    ASSERT_TRUE(fixture.admit(0, 0));
    ASSERT_TRUE(fixture.admit(1, 3));
    ASSERT_TRUE(fixture.stage_one(0, 0));
    OrderedExecution slow{};
    ASSERT_TRUE(fixture.begin(0, &slow));
    ASSERT_TRUE(fixture.fill(3));
    for (uint32_t index = 0; index < 2; ++index) {
        ASSERT_TRUE(fixture.fill(3));
        OrderedExecution fast{};
        ASSERT_TRUE(fixture.begin(3, &fast));
        ASSERT_EQ(fast.task, 0);
        ASSERT_TRUE(fixture.complete(fast));
    }
    bool closed = false;
    ASSERT_TRUE(fixture.close(0, &closed));
    EXPECT_FALSE(closed);
    ASSERT_TRUE(fixture.close(3, &closed));
    EXPECT_TRUE(closed);
    EXPECT_EQ(fixture.admissions[3].scan_cursor, 1u);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.control(0)->completed_blocks, UINT64_C(0)), 2u);
    ASSERT_TRUE(fixture.fill(3));
    OrderedExecution next{};
    ASSERT_TRUE(fixture.begin(3, &next));
    EXPECT_EQ(next.task, 1);
    EXPECT_EQ(scheduler_gm_load(fixture.control(0)->state), static_cast<int64_t>(SchedulerTaskState::BLOCKED));
    ASSERT_TRUE(fixture.complete(next));
    ASSERT_TRUE(fixture.fill(3));
    ASSERT_TRUE(fixture.begin(3, &next));
    EXPECT_EQ(next.task, 1);
    ASSERT_TRUE(fixture.complete(next));
    ASSERT_TRUE(fixture.complete(slow));
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.control(0)->completed_blocks, UINT64_C(0)), 3u);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.control(1)->completed_blocks, UINT64_C(0)), 2u);
}

TEST(HbgOrderedExecution, CursorExhaustionIsBoundedAndDoesNotCloseLiveLocalSlots) {
    OrderedFixture fixture({{2, LaneMask::aic()}});
    ASSERT_TRUE(fixture.admit(0, 0));
    ASSERT_TRUE(fixture.stage_one(0, 0));
    ASSERT_TRUE(fixture.stage_one(3, 0));
    for (uint32_t attempt = 0; attempt < 64; ++attempt)
        EXPECT_EQ(
            scheduler_claim_ordered_block(fixture.graph.graph(), fixture.base(), fixture.locals[3], fixture.run(), 1),
            SchedulerBlockClaimResult::EXHAUSTED
        );
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.control(0)->next_block, UINT64_C(0)), 2u);
    bool closed = true;
    ASSERT_TRUE(fixture.close(0, &closed));
    EXPECT_FALSE(closed);
    ASSERT_TRUE(fixture.close(3, &closed));
    EXPECT_FALSE(closed);
    EXPECT_EQ(fixture.admissions[0].current_task_ref, 0);
    EXPECT_EQ(fixture.admissions[3].current_task_ref, 0);
}

TEST(HbgOrderedExecution, EmptyFillStillStopsOnAnAlreadyPublishedRunError) {
    OrderedFixture fixture({{2, LaneMask::aic()}, {1, LaneMask::aic()}});
    for (uint32_t poll = 0; poll < 8; ++poll)
        ASSERT_TRUE(fixture.fill(0));
    EXPECT_EQ(fixture.locals[0]->local_ready_mask, 0u);
    EXPECT_EQ(fixture.admissions[0].current_task_ref, SCHEDULER_TASK_REF_INVALID);
    scheduler_record_error(
        fixture.run(), 0, SchedulerGraphResult::TIMEOUT, nullptr, fixture.locals[3],
        SchedulerErrorSite::ORDERED_EXIT_TIMEOUT
    );
    EXPECT_FALSE(fixture.fill(0));
    EXPECT_EQ(fixture.locals[0]->local_ready_mask, 0u);
    for (const auto &candidate : fixture.admissions[0].candidates)
        EXPECT_EQ(candidate.ready.task_ref, SCHEDULER_TASK_REF_INVALID);
    for (const auto &slot : fixture.locals[0]->slots)
        EXPECT_EQ(slot.state, SchedulerDispatchSlotState::FREE);
    EXPECT_EQ(fixture.run()->error_worker_index, 3u);
}

TEST(HbgOrderedExecution, IdlePollIsOnlyAHintAndDoesNotConsumeNewWorkOrEntryPermission) {
    OrderedFixture fixture({{2, LaneMask::aic()}, {1, LaneMask::aic()}});
    for (uint32_t worker = 0; worker < 6; ++worker)
        EXPECT_FALSE(
            scheduler_ordered_poll_may_progress(fixture.graph.graph(), fixture.base(), fixture.locals[worker])
        );
    ASSERT_TRUE(fixture.admit(0, 3));
    EXPECT_TRUE(scheduler_ordered_poll_may_progress(fixture.graph.graph(), fixture.base(), fixture.locals[0]));
    EXPECT_EQ(fixture.admissions[0].current_task_ref, SCHEDULER_TASK_REF_INVALID);
    EXPECT_EQ(fixture.control(0)->next_block, 0u);
    ASSERT_TRUE(fixture.fill(0));
    EXPECT_TRUE(scheduler_ordered_poll_may_progress(fixture.graph.graph(), fixture.base(), fixture.locals[0]));
    scheduler_record_error(
        fixture.run(), 0, SchedulerGraphResult::TIMEOUT, nullptr, fixture.locals[3],
        SchedulerErrorSite::ORDERED_EXIT_TIMEOUT
    );
    OrderedExecution execution{};
    EXPECT_FALSE(fixture.begin(0, &execution));
    EXPECT_FALSE(fixture.fill(0));
    EXPECT_EQ(fixture.control(0)->completed_blocks, 0u);
}

TEST(HbgOrderedExecution, IdlePollFindsOwnWorkEvenWhenItsDirectoryBitIsMissing) {
    for (LaneMask mask : kPureCAndV) {
        SCOPED_TRACE(mask.count());
        OrderedFixture fixture({{2, mask}});
        const uint32_t worker = mask == LaneMask::aic() ? 0 : 1;
        auto *local = fixture.locals[worker];
        ASSERT_TRUE(fixture.enqueue(0, worker));
        scheduler_ready_directory_clear(
            scheduler_ready_directory_at(fixture.base(), local), local->config.self_core_type, local->self_type_rank()
        );
        ASSERT_FALSE(scheduler_ready_directory_nonempty(
            fixture.base(), local, local->config.scheduler_count, local->config.self_core_type
        ));
        EXPECT_TRUE(scheduler_ordered_poll_may_progress(fixture.graph.graph(), fixture.base(), local));
        ASSERT_TRUE(fixture.fill(worker));
        OrderedExecution execution{};
        ASSERT_TRUE(fixture.begin(worker, &execution));
        EXPECT_EQ(execution.task, 0);
        EXPECT_EQ(execution.blocks, 2u);
    }
}

TEST(HbgOrderedExecution, EmptyPollDoesNotGrantLaterOrdinaryWorkAPermitPastANewAdmission) {
    OrderedFixture fixture({{1, LaneMask::aic()}, {2, LaneMask::aic()}});
    for (uint32_t poll = 0; poll < 8; ++poll)
        ASSERT_TRUE(fixture.fill(0));
    ASSERT_TRUE(fixture.admit(1, 3));
    ASSERT_TRUE(fixture.enqueue(0, 0));
    ASSERT_TRUE(fixture.fill(0));
    for (uint32_t block = 0; block < 2; ++block) {
        ASSERT_TRUE(fixture.fill(0));
        OrderedExecution execution{};
        ASSERT_TRUE(fixture.begin(0, &execution));
        ASSERT_EQ(execution.task, 1);
        ASSERT_TRUE(fixture.complete(execution));
    }
    ASSERT_TRUE(fixture.fill(0));
    ASSERT_TRUE(fixture.fill(0));
    OrderedExecution ordinary{};
    ASSERT_TRUE(fixture.begin(0, &ordinary));
    EXPECT_EQ(ordinary.task, 0);
    EXPECT_EQ(fixture.admissions[0].scan_cursor, 1u);
    EXPECT_EQ(fixture.control(1)->completed_blocks, 2u);
    ASSERT_TRUE(fixture.complete(ordinary));
    EXPECT_FALSE(fixture.begin(0, &ordinary));
}

TEST(HbgOrderedExecution, SecondPreparedOrdinarySlotRechecksAdmissionAfterTheFirstEntered) {
    OrderedFixture fixture({{1, LaneMask::aic()}, {1, LaneMask::aic()}, {2, LaneMask::aic()}}, true);
    const auto *events = scheduler_state_at<SchedulerProtocolExecutionTrace>(
        fixture.base(), fixture.storage.layout.protocol.executions_offset
    );
    ASSERT_TRUE(fixture.enqueue(0, 0));
    ASSERT_TRUE(fixture.enqueue(1, 0));
    ASSERT_TRUE(fixture.fill(0));
    ASSERT_EQ(fixture.locals[0]->local_ready_mask, 3u);
    for (uint32_t poll = 0; poll < 8; ++poll)
        ASSERT_TRUE(fixture.fill(0));
    OrderedExecution first{};
    ASSERT_TRUE(fixture.begin(0, &first));
    ASSERT_GE(first.task, 0);
    ASSERT_LT(first.task, 2);
    const int64_t other = 1 - first.task;
    EXPECT_EQ(events[first.task].permit_tail, 0u);
    ASSERT_TRUE(fixture.admit(2, 3));
    ASSERT_TRUE(fixture.complete(first));
    OrderedExecution denied{};
    EXPECT_FALSE(fixture.begin(0, &denied));
    EXPECT_EQ(events[other].permit.sequence, 0u);
    ASSERT_TRUE(fixture.fill(0));
    for (uint32_t block = 0; block < 2; ++block) {
        ASSERT_TRUE(fixture.fill(0));
        OrderedExecution execution{};
        ASSERT_TRUE(fixture.begin(0, &execution));
        ASSERT_EQ(execution.task, 2);
        ASSERT_TRUE(fixture.complete(execution));
    }
    ASSERT_TRUE(fixture.fill(0));
    ASSERT_TRUE(fixture.fill(0));
    OrderedExecution second{};
    ASSERT_TRUE(fixture.begin(0, &second));
    EXPECT_EQ(second.task, other);
    EXPECT_EQ(events[other].permit_tail, 1u);
    EXPECT_EQ(events[first.task].permit_tail, 0u);
    ASSERT_TRUE(fixture.complete(second));
    EXPECT_FALSE(fixture.begin(0, &second));
}

TEST(HbgOrderedExecution, TwoPrefilledOrdinaryCandidatesYieldSlotsAndPreservePredicateDecisions) {
    OrderedFixture fixture({{1, LaneMask::aic()}, {1, LaneMask::aic()}, {2, LaneMask::aic()}}, true);
    const auto *events = scheduler_state_at<SchedulerProtocolExecutionTrace>(
        fixture.base(), fixture.storage.layout.protocol.executions_offset
    );
    alignas(64) uint64_t predicates[2]{1, 0};
    for (uint64_t task = 0; task < 2; ++task) {
        fixture.storage.metadata[task].flags |= SCHEDULER_TASK_HAS_PREDICATE;
        fixture.graph.predicate(
            task, reinterpret_cast<uint64_t>(&predicates[task]), sizeof(uint64_t),
            static_cast<uint8_t>(PredicateOp::EQ), 1
        );
        ASSERT_TRUE(fixture.enqueue(task, 0));
    }
    ASSERT_TRUE(fixture.fill(0));
    ASSERT_EQ(fixture.locals[0]->local_ready_mask, 3u);
    const uint64_t enqueue_count = fixture.ready_stats[0].enqueue_count;
    std::array<uint32_t, 2> old_generations{};
    for (uint32_t slot = 0; slot < 2; ++slot) {
        EXPECT_FALSE(fixture.locals[0]->slots[slot].cooperative);
        EXPECT_TRUE(fixture.locals[0]->slots[slot].predicate_cached);
        old_generations[slot] = fixture.locals[0]->slots[slot].generation;
    }
    ASSERT_TRUE(fixture.admit(2, 3));
    OrderedExecution denied{};
    EXPECT_FALSE(fixture.begin(0, &denied));
    EXPECT_EQ(fixture.locals[0]->local_ready_mask, 3u);
    EXPECT_EQ(events[0].permit.sequence, 0u);
    EXPECT_EQ(events[1].permit.sequence, 0u);
    predicates[0] = 0;
    predicates[1] = 1;
    ASSERT_TRUE(fixture.fill(0));
    for (uint32_t candidate = 0; candidate < 2; ++candidate) {
        EXPECT_EQ(fixture.admissions[0].candidates[candidate].ready.task_ref, candidate);
        EXPECT_EQ(fixture.admissions[0].candidates[candidate].slot_index, UINT8_MAX);
        EXPECT_TRUE(fixture.admissions[0].candidates[candidate].predicate_cached);
        EXPECT_EQ(fixture.admissions[0].candidates[candidate].predicate_pass, candidate == 0);
    }
    uint32_t ready_cooperative = 0;
    for (const auto &slot : fixture.locals[0]->slots) {
        if (slot.state == SchedulerDispatchSlotState::READY) {
            ++ready_cooperative;
            EXPECT_TRUE(slot.cooperative);
            EXPECT_EQ(slot.task_ref, 2);
        } else {
            EXPECT_EQ(slot.state, SchedulerDispatchSlotState::FREE);
        }
    }
    EXPECT_EQ(ready_cooperative, 1u);
    for (uint32_t block = 0; block < 2; ++block) {
        ASSERT_TRUE(fixture.fill(0));
        OrderedExecution cooperative{};
        ASSERT_TRUE(fixture.begin(0, &cooperative));
        ASSERT_EQ(cooperative.task, 2);
        ASSERT_TRUE(fixture.complete(cooperative));
    }
    ASSERT_TRUE(fixture.fill(0));
    ASSERT_TRUE(fixture.fill(0));
    std::array<uint32_t, 2> resumed{};
    for (uint32_t index = 0; index < 2; ++index) {
        OrderedExecution ordinary{};
        ASSERT_TRUE(fixture.begin(0, &ordinary));
        ASSERT_GE(ordinary.task, 0);
        ASSERT_LT(ordinary.task, 2);
        ++resumed[ordinary.task];
        EXPECT_EQ(ordinary.function != 0, ordinary.task == 0);
        EXPECT_GT(ordinary.generation, old_generations[ordinary.slot]);
        ASSERT_TRUE(fixture.complete(ordinary));
    }
    EXPECT_EQ(resumed, (std::array<uint32_t, 2>{1, 1}));
    EXPECT_NE(events[0].returned.sequence, 0u);
    EXPECT_EQ(events[1].returned.sequence, 0u);
    EXPECT_EQ(events[1].visible.sequence, 0u);
    EXPECT_EQ(events[1].flags & SCHEDULER_PROTOCOL_EXECUTED, 0u);
    for (uint32_t task = 0; task < 2; ++task) {
        EXPECT_EQ(events[task].publication_count, 2u);
        EXPECT_EQ(events[task].permit_tail, 1u);
        EXPECT_NE(events[task].flags & SCHEDULER_PROTOCOL_PERMIT_TAIL, 0u);
        EXPECT_GT(events[task].completion_consumed.sequence, events[task].entry.sequence);
    }
    EXPECT_EQ(fixture.ready_stats[0].enqueue_count, enqueue_count);
    const auto *protocol_workers = scheduler_state_at<SchedulerProtocolWorkerTrace>(
        fixture.base(), fixture.storage.layout.protocol.workers_offset
    );
    EXPECT_EQ(protocol_workers[0].error_flags, 0u);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.run()->scheduler_error, UINT64_C(0)), 0u);
}

TEST(HbgOrderedExecution, NonmemberVectorCanExecuteOrdinaryWorkWhileCubeObligationIsLive) {
    OrderedFixture fixture({{2, LaneMask::aic()}, {1, LaneMask::aiv0()}});
    ASSERT_TRUE(fixture.admit(0, 0));
    ASSERT_TRUE(fixture.stage_one(0, 0));
    OrderedExecution cube{};
    ASSERT_TRUE(fixture.begin(0, &cube));
    ASSERT_TRUE(fixture.enqueue(1, 1));
    ASSERT_TRUE(fixture.fill(1));
    OrderedExecution vector{};
    ASSERT_TRUE(fixture.begin(1, &vector));
    EXPECT_EQ(vector.task, 1);
    EXPECT_EQ(fixture.admissions[1].current_task_ref, SCHEDULER_TASK_REF_INVALID);
    EXPECT_EQ(fixture.admissions[1].scan_cursor, 1u);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.control(0)->completed_blocks, UINT64_C(0)), 0u);
    ASSERT_TRUE(fixture.complete(vector));
}

TEST(HbgOrderedExecution, NonmemberCubeCanExecuteOrdinaryWorkWhileVectorObligationIsLive) {
    OrderedFixture fixture({{2, LaneMask::aiv0()}, {1, LaneMask::aic()}});
    ASSERT_TRUE(fixture.admit(0, 1));
    ASSERT_TRUE(fixture.stage_one(1, 0));
    OrderedExecution vector{};
    ASSERT_TRUE(fixture.begin(1, &vector));
    ASSERT_TRUE(fixture.enqueue(1, 0));
    ASSERT_TRUE(fixture.fill(0));
    OrderedExecution cube{};
    ASSERT_TRUE(fixture.begin(0, &cube));
    EXPECT_EQ(cube.task, 1);
    EXPECT_EQ(fixture.admissions[0].current_task_ref, SCHEDULER_TASK_REF_INVALID);
    EXPECT_EQ(fixture.admissions[0].scan_cursor, 1u);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.control(0)->completed_blocks, UINT64_C(0)), 0u);
    ASSERT_TRUE(fixture.complete(cube));
}

TEST(HbgOrderedExecution, PendingAdmissionPreventsEvenAnOlderCooperativeKernelEntry) {
    OrderedFixture fixture({{2, LaneMask::aic()}, {2, LaneMask::aic()}});
    ASSERT_TRUE(fixture.admit(0, 0));
    ASSERT_TRUE(fixture.stage_one(0, 0));
    SchedulerReadyClaim later{};
    ASSERT_TRUE(fixture.pop(1, 0, &later));
    ASSERT_EQ(
        scheduler_try_claim_admission(
            fixture.graph.graph(), fixture.base(), fixture.locals[0], fixture.run(), &fixture.admissions[0], later
        ),
        SchedulerAdmissionResult::PROGRESS
    );
    OrderedExecution denied{};
    EXPECT_FALSE(fixture.begin(0, &denied));
    EXPECT_EQ(fixture.locals[0]->local_ready_mask, 1u);
    ASSERT_TRUE(scheduler_prepare_admission_record(
        fixture.graph.graph(), fixture.base(), fixture.locals[0], fixture.run(), &fixture.admissions[0]
    ));
    ASSERT_EQ(
        scheduler_try_commit_admission(
            fixture.graph.graph(), fixture.base(), fixture.locals[0], fixture.run(), &fixture.admissions[0]
        ),
        SchedulerAdmissionResult::PROGRESS
    );
    OrderedExecution permitted{};
    ASSERT_TRUE(fixture.begin(0, &permitted));
    EXPECT_EQ(permitted.task, 0);
}

TEST(HbgOrderedExecution, PersistentCandidateCreditsRejectDuplicatesAndDoNotOverwriteFullPool) {
    OrderedFixture fixture({{1, LaneMask::aic()}, {1, LaneMask::aic()}, {1, LaneMask::aic()}});
    std::array<SchedulerReadyClaim, 3> claims{};
    for (uint32_t task = 0; task < 3; ++task)
        ASSERT_TRUE(fixture.pop(task, 0, &claims[task]));
    ASSERT_TRUE(scheduler_ordered_retain_candidate(&fixture.admissions[0], claims[0]));
    EXPECT_FALSE(scheduler_ordered_retain_candidate(&fixture.admissions[0], claims[0]));
    ASSERT_TRUE(scheduler_ordered_retain_candidate(&fixture.admissions[0], claims[1]));
    EXPECT_FALSE(scheduler_ordered_retain_candidate(&fixture.admissions[0], claims[2]));
    EXPECT_EQ(scheduler_ordered_direct_sink(&fixture.admissions[0]), nullptr);
    EXPECT_EQ(fixture.admissions[0].candidates[0].ready.task_ref, 0);
    EXPECT_EQ(fixture.admissions[0].candidates[1].ready.task_ref, 1);
    EXPECT_EQ(claims[2].task_ref, 2);
}

class HbgOrderedWithoutSsbuf : public ::testing::TestWithParam<LaneMask> {};

TEST_P(HbgOrderedWithoutSsbuf, MultiBlockAndDependentOrdinaryTaskCompleteWithoutSsbufStorage) {
    const LaneMask mask = GetParam();
    OrderedFixture fixture({{3, mask}, {1, mask}});
    fixture.dependency(1, 0);
    fixture.storage.ssbuf_region = nullptr;
    const uint32_t publisher = fixture.publisher(0);
    ASSERT_EQ(
        scheduler_route_task(
            fixture.graph.graph(), fixture.base(), fixture.locals[publisher], fixture.run(), 1, nullptr
        ),
        SchedulerRouteResult::WAITING
    );
    ASSERT_TRUE(fixture.enqueue(0, publisher));
    std::array<uint32_t, 3> producer_visits{};
    uint32_t consumer_visits = 0;
    uint32_t executions = 0;
    for (uint32_t step = 0; step < 128 && executions < 4; ++step) {
        for (uint32_t worker = 0; worker < 6; ++worker) {
            ASSERT_TRUE(fixture.fill(worker));
            OrderedExecution execution{};
            if (!fixture.begin(worker, &execution)) continue;
            ASSERT_GE(execution.task, 0);
            ASSERT_LT(execution.task, 2);
            ASSERT_NE(execution.function, 0u);
            if (execution.task == 0) {
                ASSERT_EQ(execution.blocks, 3u);
                ASSERT_LT(execution.block, producer_visits.size());
                ++producer_visits[execution.block];
            } else {
                EXPECT_EQ(execution.blocks, 1u);
                EXPECT_EQ(execution.block, 0u);
                EXPECT_EQ(scheduler_gm_fetch_add(fixture.control(0)->completed_blocks, UINT64_C(0)), 3u);
                EXPECT_EQ(scheduler_gm_load(fixture.control(0)->state), static_cast<int64_t>(SchedulerTaskState::DONE));
                ++consumer_visits;
            }
            ++executions;
            ASSERT_TRUE(fixture.complete(execution));
        }
    }
    EXPECT_EQ(executions, 4u);
    EXPECT_EQ(producer_visits, (std::array<uint32_t, 3>{1, 1, 1}));
    EXPECT_EQ(consumer_visits, 1u);
    for (uint64_t task = 0; task < 2; ++task)
        EXPECT_EQ(scheduler_gm_load(fixture.control(task)->state), static_cast<int64_t>(SchedulerTaskState::DONE));
    for (uint32_t worker = 0; worker < 6; ++worker) {
        ASSERT_TRUE(fixture.fill(worker));
        scheduler_flush_completions(fixture.run(), fixture.locals[worker]);
    }
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.run()->resolved_task_count, UINT64_C(0)), 2u);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.run()->scheduler_error, UINT64_C(0)), 0u);
}

INSTANTIATE_TEST_SUITE_P(PureCAndV, HbgOrderedWithoutSsbuf, ::testing::ValuesIn(kPureCAndV));

class HbgOrderedQueueProbe : public ::testing::TestWithParam<std::tuple<LaneMask, LaneMask>> {};

TEST_P(HbgOrderedQueueProbe, OrdinaryLocalQueueClaimRecordsRealProbeTimesAtPhaseProfilingLevel) {
    const auto [cooperative_mask, ordinary_mask] = GetParam();
    OrderedFixture fixture({{2, cooperative_mask}, {1, ordinary_mask}});
    const uint32_t worker = ordinary_mask == LaneMask::aic() ? 0 : 2;
    ASSERT_EQ(fixture.admissions[worker].capacity, 1u);
    ASSERT_TRUE(fixture.enqueue(1, worker));
    ASSERT_TRUE(fixture.fill(worker, SCHEDULER_PROFILING_SCHED_PHASES_LEVEL));
    const auto *traces =
        scheduler_state_at<SchedulerTaskTrace>(fixture.base(), fixture.storage.layout.trace_cells_offset);
    const auto &trace = traces[fixture.storage.metadata[1].trace_index_base];
    EXPECT_EQ(trace.task_ref, 1u);
    EXPECT_EQ(trace.worker_id, worker);
    EXPECT_EQ(trace.ready_source, static_cast<uint64_t>(SchedulerReadySource::LOCAL));
    EXPECT_EQ(trace.state_probe_scheduler_worker_id, worker);
    EXPECT_GT(trace.state_probe_start_cycles, 0u);
    EXPECT_GE(trace.state_probe_end_cycles, trace.state_probe_start_cycles);
    EXPECT_GT(trace.dispatch_start_cycles, 0u);
    EXPECT_EQ(
        scheduler_gm_query(fixture.control(0)->admission_state),
        static_cast<uint64_t>(SchedulerAdmissionState::UNCLAIMED)
    );
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.run()->scheduler_error, UINT64_C(0)), 0u);
}

INSTANTIATE_TEST_SUITE_P(
    SameAndDifferentCoreTypes, HbgOrderedQueueProbe,
    ::testing::Combine(::testing::ValuesIn(kPureCAndV), ::testing::ValuesIn(kPureCAndV))
);

class HbgOrderedCooperativeProbe : public ::testing::TestWithParam<LaneMask> {};

TEST_P(HbgOrderedCooperativeProbe, CooperativeBlockDispatchDoesNotInventQueueProbeTimes) {
    OrderedFixture fixture({{2, GetParam()}});
    const uint32_t worker = fixture.publisher(0);
    ASSERT_TRUE(fixture.enqueue(0, worker));
    ASSERT_TRUE(fixture.fill(worker, SCHEDULER_PROFILING_SCHED_PHASES_LEVEL));
    const auto *traces =
        scheduler_state_at<SchedulerTaskTrace>(fixture.base(), fixture.storage.layout.trace_cells_offset);
    for (uint32_t block = 0; block < 2; ++block) {
        SCOPED_TRACE(block);
        ASSERT_TRUE(fixture.fill(worker, SCHEDULER_PROFILING_SCHED_PHASES_LEVEL));
        const auto &trace = traces[fixture.storage.metadata[0].trace_index_base + block];
        EXPECT_EQ(trace.task_ref, 0u);
        EXPECT_EQ(trace.worker_id, worker);
        EXPECT_EQ(trace.ready_source, static_cast<uint64_t>(SchedulerReadySource::COOPERATIVE));
        EXPECT_EQ(trace.state_probe_scheduler_worker_id, UINT64_MAX);
        EXPECT_EQ(trace.state_probe_start_cycles, 0u);
        EXPECT_EQ(trace.state_probe_end_cycles, 0u);
        EXPECT_GT(trace.dispatch_start_cycles, 0u);
        OrderedExecution execution{};
        ASSERT_TRUE(fixture.begin(worker, &execution));
        EXPECT_EQ(execution.block, block);
        ASSERT_TRUE(fixture.complete(execution));
    }
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.run()->scheduler_error, UINT64_C(0)), 0u);
}

INSTANTIATE_TEST_SUITE_P(PureCAndV, HbgOrderedCooperativeProbe, ::testing::ValuesIn(kPureCAndV));

class HbgOrderedDeferredPublication : public ::testing::TestWithParam<LaneMask> {};

TEST_P(HbgOrderedDeferredPublication, DirectSuccessorPublishedLaterIsDispatchNotAnInlineRefill) {
    const LaneMask mask = GetParam();
    OrderedFixture fixture({{2, mask}, {1, mask}});
    fixture.dependency(1, 0);
    const uint32_t worker = fixture.publisher(0);
    auto *local = fixture.locals[worker];
    ASSERT_EQ(
        scheduler_route_task(fixture.graph.graph(), fixture.base(), local, fixture.run(), 1, nullptr),
        SchedulerRouteResult::WAITING
    );
    ASSERT_TRUE(fixture.enqueue(0, worker));
    ASSERT_TRUE(fixture.fill(worker, SCHEDULER_PROFILING_SCHED_PHASES_LEVEL));
    for (uint32_t block = 0; block < 2; ++block) {
        ASSERT_TRUE(fixture.fill(worker, SCHEDULER_PROFILING_SCHED_PHASES_LEVEL));
        OrderedExecution producer{};
        ASSERT_TRUE(fixture.begin(worker, &producer));
        ASSERT_EQ(producer.task, 0);
        ASSERT_TRUE(fixture.complete(producer));
    }
    ASSERT_EQ(scheduler_gm_load(fixture.control(0)->state), static_cast<int64_t>(SchedulerTaskState::DONE));
    EXPECT_EQ(local->local_ready_mask, 0u);
    for (const auto &slot : local->slots)
        EXPECT_EQ(slot.state, SchedulerDispatchSlotState::FREE);
    uint32_t retained = 0;
    for (const auto &candidate : fixture.admissions[worker].candidates) {
        if (candidate.ready.task_ref != 1) continue;
        ++retained;
        EXPECT_EQ(candidate.ready.source, SchedulerReadySource::DIRECT_RESOLVE);
        EXPECT_EQ(candidate.slot_index, UINT8_MAX);
    }
    ASSERT_EQ(retained, 1u);

    ASSERT_TRUE(fixture.fill(worker, SCHEDULER_PROFILING_SCHED_PHASES_LEVEL));
    ASSERT_TRUE(fixture.fill(worker, SCHEDULER_PROFILING_SCHED_PHASES_LEVEL));
    const auto *traces =
        scheduler_state_at<SchedulerTaskTrace>(fixture.base(), fixture.storage.layout.trace_cells_offset);
    const auto &trace = traces[fixture.storage.metadata[1].trace_index_base];
    EXPECT_EQ(trace.task_ref, 1u);
    EXPECT_EQ(trace.worker_id, worker);
    EXPECT_EQ(trace.ready_source, static_cast<uint64_t>(SchedulerReadySource::DIRECT_RESOLVE));
    EXPECT_EQ(trace.publication_mode, static_cast<uint64_t>(SchedulerPublicationMode::DISPATCH));
    EXPECT_GT(trace.dispatch_start_cycles, 0u);
    EXPECT_GE(trace.dispatch_end_cycles, trace.dispatch_start_cycles);
    EXPECT_EQ(trace.state_probe_scheduler_worker_id, UINT64_MAX);
    EXPECT_EQ(trace.state_probe_start_cycles, 0u);
    EXPECT_EQ(trace.state_probe_end_cycles, 0u);
    OrderedExecution successor{};
    ASSERT_TRUE(fixture.begin(worker, &successor));
    EXPECT_EQ(successor.task, 1);
    ASSERT_TRUE(fixture.complete(successor));
    EXPECT_EQ(scheduler_gm_load(fixture.control(1)->state), static_cast<int64_t>(SchedulerTaskState::DONE));
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.run()->scheduler_error, UINT64_C(0)), 0u);
}

INSTANTIATE_TEST_SUITE_P(PureCAndV, HbgOrderedDeferredPublication, ::testing::ValuesIn(kPureCAndV));

TEST(HbgOrderedProtocol, RecordsClaimOwnerAndActualCompletionTicketRatherThanBlockOrder) {
    for (LaneMask mask : kPureCAndV) {
        OrderedFixture fixture({{2, mask}}, true);
        const uint32_t first = mask == LaneMask::aic() ? 0 : 1;
        const uint32_t second = mask == LaneMask::aic() ? 3 : 2;
        ASSERT_TRUE(fixture.admit(0, first));
        ASSERT_TRUE(fixture.stage_one(first, 0));
        ASSERT_TRUE(fixture.stage_one(second, 0));
        auto *blocks = scheduler_state_at<SchedulerProtocolBlockTrace>(
            fixture.base(), fixture.storage.layout.protocol.blocks_offset
        );
        for (uint32_t block = 0; block < 2; ++block) {
            const auto &owner = blocks[block].owner;
            EXPECT_NE(owner.claim_before.sequence, 0u);
            EXPECT_GT(owner.claim_after.sequence, owner.claim_before.sequence);
            EXPECT_LE(owner.claim_before.cycles, owner.claim_after.cycles);
            EXPECT_EQ(owner.run_epoch, 37u);
            EXPECT_EQ(owner.task_ref, 0);
            EXPECT_EQ(owner.block_idx, block);
            EXPECT_EQ(owner.worker_index, block == 0 ? first : second);
            EXPECT_EQ(owner.logical_block_num, 2u);
            EXPECT_EQ(owner.active_mask, mask);
            EXPECT_EQ(owner.group_seq, 0u);
            EXPECT_EQ(owner.completion_before.sequence, 0u);
        }
        OrderedExecution first_execution{}, second_execution{};
        ASSERT_TRUE(fixture.begin(first, &first_execution));
        ASSERT_TRUE(fixture.begin(second, &second_execution));
        ASSERT_TRUE(fixture.complete(second_execution));
        EXPECT_EQ(blocks[1].owner.completion_ticket, 0u);
        EXPECT_GT(blocks[1].owner.completion_before.sequence, blocks[1].owner.claim_after.sequence);
        EXPECT_GT(blocks[1].owner.completion_after.sequence, blocks[1].owner.completion_before.sequence);
        EXPECT_EQ(blocks[0].owner.completion_after.sequence, 0u);
        ASSERT_TRUE(fixture.complete(first_execution));
        EXPECT_EQ(blocks[0].owner.completion_ticket, 1u);
        EXPECT_GT(blocks[0].owner.completion_after.sequence, blocks[0].owner.completion_before.sequence);
        EXPECT_EQ(scheduler_gm_fetch_add(fixture.control(0)->completed_blocks, UINT64_C(0)), 2u);
    }
}

class HbgOrderedTicketCursor : public ::testing::TestWithParam<LaneMask> {};

TEST(HbgOrderedTicketPolicy, AllOrdinaryMultiblockTasksUseTicketsButSingleAndSyncStartDoNot) {
    for (LaneMask mask :
         {LaneMask::aic(), LaneMask::aiv0(), kAicAiv0, LaneMask::aiv1(), kAicAiv1, LaneMask::both_aiv(),
          LaneMask::all()}) {
        for (uint64_t clusters : {UINT64_C(1), UINT64_C(2), UINT64_C(8), UINT64_C(32)}) {
            const uint64_t positions = clusters * (mask.to_shape() == ResourceShape::AIV ? 2 : 1);
            EXPECT_FALSE(scheduler_task_uses_ticket_claim(0, mask, 1, clusters));
            EXPECT_FALSE(scheduler_task_uses_ticket_claim(0, mask, 0, clusters));
            EXPECT_TRUE(scheduler_task_uses_ticket_claim(0, mask, 2, clusters));
            EXPECT_TRUE(scheduler_task_uses_ticket_claim(0, mask, 3, clusters));
            EXPECT_EQ(scheduler_task_uses_ticket_claim(0, mask, positions * 2 - 1, clusters), positions * 2 - 1 > 1);
            EXPECT_TRUE(scheduler_task_uses_ticket_claim(0, mask, positions * 2, clusters));
            EXPECT_TRUE(scheduler_task_uses_ticket_claim(0, mask, positions * 2 + 1, clusters));
            EXPECT_FALSE(scheduler_task_uses_ticket_claim(SCHEDULER_TASK_SYNC_START, mask, positions * 2, clusters));
        }
        EXPECT_FALSE(scheduler_task_uses_ticket_claim(0, mask, 32767, 0));
    }
}

TEST_P(HbgOrderedTicketCursor, SmallCountsAndTwoWaveNeighboursKeepExactBlockOwnership) {
    const LaneMask mask = GetParam();
    const uint32_t positions = mask == LaneMask::aic() ? 2 : 4;
    for (uint32_t count : {2u, 3u, positions * 2 - 1, positions * 2, positions * 2 + 1}) {
        OrderedFixture fixture({{static_cast<uint16_t>(count), mask}}, true);
        const uint32_t worker = fixture.publisher(0);
        ASSERT_TRUE(fixture.admit(0, worker));
        ASSERT_TRUE(scheduler_refresh_local_obligation(
            fixture.graph.graph(), fixture.base(), fixture.locals[worker], fixture.run(), &fixture.admissions[worker]
        ));
        for (uint32_t expected = 0; expected < count; ++expected) {
            uint32_t block = UINT32_MAX;
            ASSERT_EQ(
                scheduler_claim_ordered_cursor(
                    fixture.graph.graph(), fixture.base(), fixture.locals[worker], fixture.run(), &block
                ),
                SchedulerBlockClaimResult::CLAIMED
            );
            EXPECT_EQ(block, expected);
            EXPECT_EQ(fixture.control(0)->claim_tickets, expected + 1);
            EXPECT_EQ(fixture.control(0)->next_block, expected + 1);
            EXPECT_EQ(fixture.control(0)->completed_blocks, 0u);
        }
        uint32_t block = UINT32_MAX;
        EXPECT_EQ(
            scheduler_claim_ordered_cursor(
                fixture.graph.graph(), fixture.base(), fixture.locals[worker], fixture.run(), &block
            ),
            SchedulerBlockClaimResult::EXHAUSTED
        );
        EXPECT_EQ(block, UINT32_MAX);
        EXPECT_EQ(fixture.control(0)->next_block, count);
        EXPECT_LE(fixture.control(0)->claim_tickets, count + positions);
    }
}

TEST_P(HbgOrderedTicketCursor, LargeOrdinaryClaimsKeepExactValidCountAndBoundTheTail) {
    const LaneMask mask = GetParam();
    const uint32_t positions = mask == LaneMask::aic() ? 2 : 4;
    const uint32_t count = positions * 2 + 1;
    OrderedFixture fixture({{static_cast<uint16_t>(count), mask}}, true);
    ASSERT_TRUE(fixture.admit(0, fixture.publisher(0)));
    const uint32_t worker = fixture.publisher(0);
    ASSERT_TRUE(scheduler_refresh_local_obligation(
        fixture.graph.graph(), fixture.base(), fixture.locals[worker], fixture.run(), &fixture.admissions[worker]
    ));
    for (uint32_t expected = 0; expected < count; ++expected) {
        uint32_t block = UINT32_MAX;
        ASSERT_EQ(
            scheduler_claim_ordered_cursor(
                fixture.graph.graph(), fixture.base(), fixture.locals[worker], fixture.run(), &block
            ),
            SchedulerBlockClaimResult::CLAIMED
        );
        EXPECT_EQ(block, expected);
        EXPECT_EQ(fixture.control(0)->claim_tickets, expected + 1);
        EXPECT_EQ(fixture.control(0)->next_block, expected + 1);
        EXPECT_EQ(fixture.control(0)->completed_blocks, 0u);
    }
    for (uint32_t attempt = 0; attempt < 20; ++attempt) {
        uint32_t block = UINT32_MAX;
        EXPECT_EQ(
            scheduler_claim_ordered_cursor(
                fixture.graph.graph(), fixture.base(), fixture.locals[worker], fixture.run(), &block
            ),
            SchedulerBlockClaimResult::EXHAUSTED
        );
        EXPECT_EQ(block, UINT32_MAX);
    }
    EXPECT_EQ(fixture.control(0)->next_block, count);
    EXPECT_LE(fixture.control(0)->claim_tickets, count + positions);
}

TEST_P(HbgOrderedTicketCursor, AReservedLastTicketCannotBecomeAnEarlyCloseOrAnotherBlock) {
    const LaneMask mask = GetParam();
    for (uint32_t count : {2u, 3u, mask == LaneMask::aic() ? 4u : 8u}) {
        OrderedFixture fixture({{static_cast<uint16_t>(count), mask}, {2, mask}}, true);
        const uint32_t worker = fixture.publisher(0);
        ASSERT_TRUE(fixture.admit(0, worker));
        ASSERT_TRUE(fixture.admit(1, worker));
        ASSERT_TRUE(scheduler_refresh_local_obligation(
            fixture.graph.graph(), fixture.base(), fixture.locals[worker], fixture.run(), &fixture.admissions[worker]
        ));
        // The last ticket's owner is between reservation and valid-claim publication.
        scheduler_gm_store(fixture.control(0)->claim_tickets, count);
        scheduler_gm_store(fixture.control(0)->next_block, count - 1);
        scheduler_gm_store(fixture.control(0)->completed_blocks, count - 1);
        uint32_t block = UINT32_MAX;
        ASSERT_EQ(
            scheduler_claim_ordered_cursor(
                fixture.graph.graph(), fixture.base(), fixture.locals[worker], fixture.run(), &block
            ),
            SchedulerBlockClaimResult::EXHAUSTED
        );
        EXPECT_EQ(block, UINT32_MAX);
        EXPECT_EQ(fixture.control(0)->claim_tickets, count + 1);
        EXPECT_TRUE(fixture.admissions[worker].ticket_exhausted);
        const auto *trace = scheduler_state_at<SchedulerProtocolWorkerTrace>(
            fixture.base(), fixture.storage.layout.protocol.workers_offset
        );
        const uint64_t sequence = trace[worker].next_sequence;
        for (uint32_t attempt = 0; attempt < 20; ++attempt)
            EXPECT_EQ(
                scheduler_claim_ordered_cursor(
                    fixture.graph.graph(), fixture.base(), fixture.locals[worker], fixture.run(), &block
                ),
                SchedulerBlockClaimResult::EXHAUSTED
            );
        EXPECT_EQ(fixture.control(0)->claim_tickets, count + 1);
        EXPECT_EQ(trace[worker].next_sequence, sequence);
        bool closed = true;
        ASSERT_TRUE(scheduler_close_ordered_obligation(
            fixture.graph.graph(), fixture.base(), fixture.locals[worker], fixture.run(), &closed
        ));
        EXPECT_FALSE(closed);
        EXPECT_EQ(fixture.admissions[worker].current_task_ref, 0);
        scheduler_gm_fetch_add(fixture.control(0)->next_block, UINT64_C(1));
        ASSERT_TRUE(scheduler_close_ordered_obligation(
            fixture.graph.graph(), fixture.base(), fixture.locals[worker], fixture.run(), &closed
        ));
        EXPECT_TRUE(closed);
        EXPECT_FALSE(fixture.admissions[worker].ticket_exhausted);
        EXPECT_EQ(fixture.control(0)->completed_blocks, count - 1);
        ASSERT_TRUE(scheduler_refresh_local_obligation(
            fixture.graph.graph(), fixture.base(), fixture.locals[worker], fixture.run(), &fixture.admissions[worker]
        ));
        EXPECT_EQ(fixture.admissions[worker].current_task_ref, 1);
    }
}

TEST_P(HbgOrderedTicketCursor, ErrorAndPoisonedCountersCannotPublishAnotherValidClaim) {
    const LaneMask mask = GetParam();
    const uint64_t positions = mask == LaneMask::aic() ? 2 : 4;
    for (uint32_t fault = 0; fault < 4; ++fault) {
        OrderedFixture fixture({{17, mask}});
        const uint32_t worker = fixture.publisher(0);
        ASSERT_TRUE(fixture.admit(0, worker));
        ASSERT_TRUE(scheduler_refresh_local_obligation(
            fixture.graph.graph(), fixture.base(), fixture.locals[worker], fixture.run(), &fixture.admissions[worker]
        ));
        const uint64_t ticket = fault == 1 ? 17 + positions : fault == 2 ? UINT64_MAX : 0;
        if (fault == 0) scheduler_record_error(fixture.run(), 0, SchedulerGraphResult::TIMEOUT);
        scheduler_gm_store(fixture.control(0)->claim_tickets, ticket);
        if (fault == 3) scheduler_gm_store(fixture.control(0)->next_block, UINT64_C(18));
        uint32_t block = UINT32_MAX;
        EXPECT_EQ(
            scheduler_claim_ordered_cursor(
                fixture.graph.graph(), fixture.base(), fixture.locals[worker], fixture.run(), &block
            ),
            SchedulerBlockClaimResult::ERROR
        );
        EXPECT_EQ(block, UINT32_MAX);
        EXPECT_EQ(fixture.control(0)->claim_tickets, ticket);
        EXPECT_EQ(fixture.control(0)->next_block, fault == 3 ? 18u : 0u);
        EXPECT_EQ(fixture.control(0)->completed_blocks, 0u);
        EXPECT_NE(scheduler_read_error(fixture.run()), 0u);
    }
}

TEST_P(HbgOrderedTicketCursor, InconsistentReservationAndConfirmationStopBeforeReturningABlock) {
    const uint64_t positions = GetParam() == LaneMask::aic() ? 2 : 4;
    OrderedFixture fixture({{17, GetParam()}}, true);
    const uint32_t worker = fixture.publisher(0);
    ASSERT_TRUE(fixture.admit(0, worker));
    ASSERT_TRUE(scheduler_refresh_local_obligation(
        fixture.graph.graph(), fixture.base(), fixture.locals[worker], fixture.run(), &fixture.admissions[worker]
    ));
    scheduler_gm_store(fixture.control(0)->claim_tickets, positions + 1);
    const auto *trace = scheduler_state_at<SchedulerProtocolWorkerTrace>(
        fixture.base(), fixture.storage.layout.protocol.workers_offset
    );
    const uint64_t sequence = trace[worker].next_sequence;
    uint32_t block = UINT32_MAX;
    EXPECT_EQ(
        scheduler_claim_ordered_cursor(
            fixture.graph.graph(), fixture.base(), fixture.locals[worker], fixture.run(), &block
        ),
        SchedulerBlockClaimResult::ERROR
    );
    EXPECT_EQ(block, UINT32_MAX);
    EXPECT_EQ(fixture.control(0)->claim_tickets, positions + 2);
    EXPECT_EQ(fixture.control(0)->next_block, 1u);
    EXPECT_EQ(fixture.control(0)->completed_blocks, 0u);
    EXPECT_EQ(trace[worker].next_sequence, sequence);
    EXPECT_NE(scheduler_read_error(fixture.run()), 0u);
    for (const auto &slot : fixture.locals[worker]->slots)
        EXPECT_EQ(slot.state, SchedulerDispatchSlotState::FREE);
}

TEST_P(HbgOrderedTicketCursor, ConcurrentTailLosersStopWithoutHidingTheUnpublishedValidTicket) {
    const LaneMask mask = GetParam();
    const std::vector<uint32_t> workers =
        mask == LaneMask::aic() ? std::vector<uint32_t>{0, 3} : std::vector<uint32_t>{1, 2, 4, 5};
    for (uint16_t count : {uint16_t{2}, uint16_t{3}, static_cast<uint16_t>(workers.size() * 2 + 1)}) {
        OrderedFixture fixture({{count, mask}});
        ASSERT_TRUE(fixture.admit(0, workers.front()));
        for (uint32_t worker : workers)
            ASSERT_TRUE(scheduler_refresh_local_obligation(
                fixture.graph.graph(), fixture.base(), fixture.locals[worker], fixture.run(),
                &fixture.admissions[worker]
            ));
        scheduler_gm_store(fixture.control(0)->claim_tickets, count);
        scheduler_gm_store(fixture.control(0)->next_block, count - 1);
        std::atomic<uint32_t> arrived{0};
        std::atomic<bool> failed{false};
        std::vector<std::thread> competitors;
        for (size_t index = 1; index < workers.size(); ++index)
            competitors.emplace_back([&, index] {
                arrived.fetch_add(1);
                while (arrived.load() != workers.size() - 1) {}
                for (uint32_t attempt = 0; attempt < 32; ++attempt) {
                    uint32_t block = UINT32_MAX;
                    if (scheduler_claim_ordered_cursor(
                            fixture.graph.graph(), fixture.base(), fixture.locals[workers[index]], fixture.run(), &block
                        ) != SchedulerBlockClaimResult::EXHAUSTED ||
                        block != UINT32_MAX)
                        failed.store(true);
                }
            });
        for (auto &competitor : competitors)
            competitor.join();
        ASSERT_FALSE(failed.load());
        EXPECT_EQ(fixture.control(0)->claim_tickets, count + workers.size() - 1);
        EXPECT_EQ(fixture.control(0)->next_block, count - 1);
        for (size_t index = 1; index < workers.size(); ++index) {
            bool closed = true;
            ASSERT_TRUE(scheduler_close_ordered_obligation(
                fixture.graph.graph(), fixture.base(), fixture.locals[workers[index]], fixture.run(), &closed
            ));
            EXPECT_FALSE(closed);
        }
    }
}

TEST_P(HbgOrderedTicketCursor, LargestPublicBlockCountKeepsFullWidthTicketAndValidCount) {
    OrderedFixture fixture({{32767, GetParam()}});
    const uint32_t worker = fixture.publisher(0);
    ASSERT_TRUE(fixture.admit(0, worker));
    ASSERT_TRUE(scheduler_refresh_local_obligation(
        fixture.graph.graph(), fixture.base(), fixture.locals[worker], fixture.run(), &fixture.admissions[worker]
    ));
    scheduler_gm_store(fixture.control(0)->claim_tickets, UINT64_C(32766));
    scheduler_gm_store(fixture.control(0)->next_block, UINT64_C(32766));
    uint32_t block = UINT32_MAX;
    ASSERT_EQ(
        scheduler_claim_ordered_cursor(
            fixture.graph.graph(), fixture.base(), fixture.locals[worker], fixture.run(), &block
        ),
        SchedulerBlockClaimResult::CLAIMED
    );
    EXPECT_EQ(block, 32766u);
    EXPECT_EQ(fixture.control(0)->claim_tickets, 32767u);
    EXPECT_EQ(fixture.control(0)->next_block, 32767u);
}

TEST_P(HbgOrderedTicketCursor, StaleTicketsRejectBothPreparationAndCommit) {
    for (bool prepared : {false, true}) {
        OrderedFixture fixture({{17, GetParam()}});
        const uint32_t worker = fixture.publisher(0);
        SchedulerReadyClaim ready{};
        ASSERT_TRUE(fixture.pop(0, worker, &ready));
        ASSERT_EQ(
            scheduler_try_claim_admission(
                fixture.graph.graph(), fixture.base(), fixture.locals[worker], fixture.run(),
                &fixture.admissions[worker], ready
            ),
            SchedulerAdmissionResult::PROGRESS
        );
        if (prepared)
            ASSERT_TRUE(scheduler_prepare_admission_record(
                fixture.graph.graph(), fixture.base(), fixture.locals[worker], fixture.run(),
                &fixture.admissions[worker]
            ));
        scheduler_gm_store(fixture.control(0)->claim_tickets, UINT64_C(1));
        if (prepared) {
            EXPECT_EQ(
                scheduler_try_commit_admission(
                    fixture.graph.graph(), fixture.base(), fixture.locals[worker], fixture.run(),
                    &fixture.admissions[worker]
                ),
                SchedulerAdmissionResult::ERROR
            );
        } else {
            EXPECT_FALSE(scheduler_prepare_admission_record(
                fixture.graph.graph(), fixture.base(), fixture.locals[worker], fixture.run(),
                &fixture.admissions[worker]
            ));
        }
        const auto *admission = scheduler_state_at<SchedulerAdmissionControl>(
            fixture.base(), fixture.storage.layout.admission_control_offset
        );
        EXPECT_EQ(admission->commit_tail, 0u);
        EXPECT_EQ(fixture.control(0)->next_block, 0u);
        EXPECT_NE(scheduler_read_error(fixture.run()), 0u);
    }
}

TEST_P(HbgOrderedTicketCursor, SyncStartTasksKeepTheBoundedCasPath) {
    const LaneMask mask = GetParam();
    const uint16_t positions = mask == LaneMask::aic() ? 2 : 4;
    OrderedFixture fixture({{positions, mask, true}});
    const uint32_t worker = fixture.publisher(0);
    ASSERT_TRUE(fixture.admit(0, worker));
    ASSERT_TRUE(scheduler_refresh_local_obligation(
        fixture.graph.graph(), fixture.base(), fixture.locals[worker], fixture.run(), &fixture.admissions[worker]
    ));
    uint32_t block = UINT32_MAX;
    ASSERT_EQ(
        scheduler_claim_ordered_cursor(
            fixture.graph.graph(), fixture.base(), fixture.locals[worker], fixture.run(), &block
        ),
        SchedulerBlockClaimResult::CLAIMED
    );
    EXPECT_EQ(fixture.control(0)->claim_tickets, 0u);
    EXPECT_EQ(
        scheduler_claim_ordered_cursor(
            fixture.graph.graph(), fixture.base(), fixture.locals[worker], fixture.run(), &block
        ),
        SchedulerBlockClaimResult::RETRY
    );
}

INSTANTIATE_TEST_SUITE_P(PureMasks, HbgOrderedTicketCursor, ::testing::ValuesIn(kPureMasks));

TEST(HbgOrderedProtocol, ExhaustedCursorDoesNotInventAnotherClaim) {
    OrderedFixture fixture({{2, LaneMask::aic()}}, true);
    ASSERT_TRUE(fixture.admit(0, 0));
    ASSERT_TRUE(fixture.stage_one(0, 0));
    ASSERT_TRUE(fixture.stage_one(0, 1));
    const auto *workers = scheduler_state_at<SchedulerProtocolWorkerTrace>(
        fixture.base(), fixture.storage.layout.protocol.workers_offset
    );
    const uint64_t sequence = workers[0].next_sequence;
    uint32_t block = UINT32_MAX;
    EXPECT_EQ(
        scheduler_claim_ordered_cursor(fixture.graph.graph(), fixture.base(), fixture.locals[0], fixture.run(), &block),
        SchedulerBlockClaimResult::EXHAUSTED
    );
    EXPECT_EQ(workers[0].next_sequence, sequence);
    EXPECT_EQ(block, UINT32_MAX);
    const auto *blocks =
        scheduler_state_at<SchedulerProtocolBlockTrace>(fixture.base(), fixture.storage.layout.protocol.blocks_offset);
    EXPECT_NE(blocks[0].owner.claim_after.sequence, 0u);
    EXPECT_NE(blocks[1].owner.claim_after.sequence, 0u);
}

TEST(HbgOrderedProtocol, OutOfRangeTracePrefixDoesNotWriteOutsideThePlannedBlockRegion) {
    OrderedFixture fixture({{2, LaneMask::aic()}}, true);
    ASSERT_TRUE(fixture.admit(0, 0));
    fixture.storage.metadata[0].trace_index_base = UINT32_MAX;
    ASSERT_TRUE(scheduler_refresh_local_obligation(
        fixture.graph.graph(), fixture.base(), fixture.locals[0], fixture.run(), &fixture.admissions[0]
    ));
    uint32_t block = UINT32_MAX;
    ASSERT_EQ(
        scheduler_claim_ordered_cursor(fixture.graph.graph(), fixture.base(), fixture.locals[0], fixture.run(), &block),
        SchedulerBlockClaimResult::CLAIMED
    );
    EXPECT_EQ(block, 0u);
    const auto *records =
        scheduler_state_at<SchedulerProtocolBlockTrace>(fixture.base(), fixture.storage.layout.protocol.blocks_offset);
    EXPECT_EQ(records[0].owner.claim_after.sequence, 0u);
    EXPECT_EQ(records[1].owner.claim_after.sequence, 0u);
    const auto *worker = scheduler_state_at<SchedulerProtocolWorkerTrace>(
        fixture.base(), fixture.storage.layout.protocol.workers_offset
    );
    EXPECT_EQ(worker->error_flags, SCHEDULER_PROTOCOL_INVALID_IDENTITY);
    EXPECT_EQ(worker->finished, 0u);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.run()->scheduler_error, UINT64_C(0)), 0u);
}

class HbgOrderedConcurrentProtocol : public ::testing::TestWithParam<std::tuple<LaneMask, uint16_t>> {};

TEST_P(HbgOrderedConcurrentProtocol, ConcurrentWorkersKeepUniqueOwnersAndContiguousLocalSequences) {
    const auto [mask, count] = GetParam();
    const std::vector<uint32_t> participants =
        mask == LaneMask::aic() ? std::vector<uint32_t>{0, 3} : std::vector<uint32_t>{1, 2, 4, 5};
    OrderedFixture fixture({{count, mask}}, true);
    ASSERT_TRUE(fixture.admit(0, participants.front()));
    for (uint32_t worker : participants)
        ASSERT_TRUE(scheduler_refresh_local_obligation(
            fixture.graph.graph(), fixture.base(), fixture.locals[worker], fixture.run(), &fixture.admissions[worker]
        ));
    std::vector<std::vector<uint32_t>> claimed(participants.size());
    std::atomic<uint32_t> arrived{0};
    std::atomic<bool> failed{false};
    const auto claim = [&](uint32_t participant) {
        arrived.fetch_add(1, std::memory_order_acq_rel);
        while (arrived.load(std::memory_order_acquire) != participants.size()) {}
        while (true) {
            uint32_t block = UINT32_MAX;
            const auto result = scheduler_claim_ordered_cursor(
                fixture.graph.graph(), fixture.base(), fixture.locals[participants[participant]], fixture.run(), &block
            );
            if (result == SchedulerBlockClaimResult::EXHAUSTED) break;
            if (result == SchedulerBlockClaimResult::ERROR) {
                failed.store(true);
                break;
            }
            if (result == SchedulerBlockClaimResult::CLAIMED) claimed[participant].push_back(block);
        }
    };
    std::vector<std::thread> threads;
    for (uint32_t participant = 0; participant < participants.size(); ++participant)
        threads.emplace_back(claim, participant);
    for (auto &thread : threads)
        thread.join();
    ASSERT_FALSE(failed.load());
    size_t claimed_count = 0;
    for (const auto &claims : claimed)
        claimed_count += claims.size();
    EXPECT_EQ(claimed_count, count);
    EXPECT_GE(fixture.control(0)->claim_tickets, count);
    EXPECT_LE(fixture.control(0)->claim_tickets, count + participants.size());
    EXPECT_EQ(fixture.control(0)->next_block, count);
    std::vector<bool> seen(count, false);
    const auto *records =
        scheduler_state_at<SchedulerProtocolBlockTrace>(fixture.base(), fixture.storage.layout.protocol.blocks_offset);
    const auto *workers = scheduler_state_at<SchedulerProtocolWorkerTrace>(
        fixture.base(), fixture.storage.layout.protocol.workers_offset
    );
    for (uint32_t participant = 0; participant < participants.size(); ++participant) {
        uint64_t sequence = participant == 0 ? 3 : 1;
        for (uint32_t block : claimed[participant]) {
            ASSERT_LT(block, count);
            EXPECT_FALSE(seen[block]);
            seen[block] = true;
            const auto &owner = records[block].owner;
            EXPECT_EQ(owner.worker_index, participants[participant]);
            EXPECT_EQ(owner.block_idx, block);
            EXPECT_EQ(owner.claim_before.sequence, ++sequence);
            EXPECT_EQ(owner.claim_after.sequence, ++sequence);
        }
        EXPECT_EQ(workers[participants[participant]].next_sequence, sequence);
        EXPECT_EQ(workers[participants[participant]].error_flags, 0u);
    }
    EXPECT_TRUE(std::all_of(seen.begin(), seen.end(), [](bool value) {
        return value;
    }));
}

INSTANTIATE_TEST_SUITE_P(
    PureMasksAndCounts, HbgOrderedConcurrentProtocol,
    ::testing::Combine(::testing::ValuesIn(kPureMasks), ::testing::Values<uint16_t>(2, 3, 128))
);

TEST(HbgOrderedProtocol, AdmissionAfterAnOrdinaryPermitCannotRewriteItsObservedTail) {
    OrderedFixture fixture({{1, LaneMask::aic()}, {2, LaneMask::aic()}}, true);
    ASSERT_TRUE(fixture.enqueue(0, 0));
    ASSERT_TRUE(fixture.fill(0));
    auto *local = fixture.locals[0];
    uint32_t slot = UINT32_MAX;
    uint64_t publication = 0;
    ASSERT_TRUE(scheduler_local_ready_pop(local, &slot, &publication));
    bool permit = false;
    ASSERT_TRUE(
        scheduler_ordered_can_enter_slot(fixture.graph.graph(), fixture.base(), local, fixture.run(), slot, &permit)
    );
    ASSERT_TRUE(permit);
    ASSERT_TRUE(fixture.admit(1, 3));
    const auto *shared =
        scheduler_state_at<SchedulerAdmissionControl>(fixture.base(), fixture.storage.layout.admission_control_offset);
    EXPECT_EQ(shared->commit_tail, 1u);
    auto *payload = scheduler_state_at<DispatchPayload>(fixture.base(), local->dispatch_payload_offset(0, slot));
    scheduler_execute_protocol_task(fixture.base(), local, slot, payload);
    const auto *events = scheduler_state_at<SchedulerProtocolExecutionTrace>(
        fixture.base(), fixture.storage.layout.protocol.executions_offset
    );
    EXPECT_EQ(events[0].permit_tail, 0u);
    EXPECT_NE(events[0].flags & SCHEDULER_PROTOCOL_PERMIT_TAIL, 0u);
    EXPECT_GT(events[0].entry.sequence, events[0].permit.sequence);
    EXPECT_GT(events[0].returned.sequence, events[0].entry.sequence);
    OrderedExecution execution{
        0, 0, 1, 0, slot, scheduler_dispatch_generation(publication), payload->function_bin_addr
    };
    ASSERT_TRUE(fixture.complete(execution));
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.control(1)->completed_blocks, UINT64_C(0)), 0u);
    const auto *worker = scheduler_state_at<SchedulerProtocolWorkerTrace>(
        fixture.base(), fixture.storage.layout.protocol.workers_offset
    );
    EXPECT_EQ(worker[0].error_flags, 0u);
}

TEST(HbgOrderedProtocol, DuplicateReturnKeepsTheFirstReturnAndVisibilityPoints) {
    OrderedFixture fixture({{2, LaneMask::aic()}}, true);
    ASSERT_TRUE(fixture.admit(0, 0));
    ASSERT_TRUE(fixture.stage_one(0, 0));
    OrderedExecution execution{};
    ASSERT_TRUE(fixture.begin(0, &execution));
    const auto *events = scheduler_state_at<SchedulerProtocolExecutionTrace>(
        fixture.base(), fixture.storage.layout.protocol.executions_offset
    );
    const auto returned = events[0].returned;
    const auto visible = events[0].visible;
    scheduler_protocol_execution_returned(fixture.base(), fixture.locals[0], execution.slot, 123, 456);
    EXPECT_EQ(events[0].returned.cycles, returned.cycles);
    EXPECT_EQ(events[0].returned.sequence, returned.sequence);
    EXPECT_EQ(events[0].visible.cycles, visible.cycles);
    EXPECT_EQ(events[0].visible.sequence, visible.sequence);
    const auto *worker = scheduler_state_at<SchedulerProtocolWorkerTrace>(
        fixture.base(), fixture.storage.layout.protocol.workers_offset
    );
    EXPECT_EQ(worker->error_flags, SCHEDULER_PROTOCOL_DUPLICATE);
}

TEST(HbgOrderedProtocol, InvalidReturnSlotLeavesExecutionRecordsUntouched) {
    OrderedFixture fixture({{1, LaneMask::aic()}}, true);
    scheduler_protocol_execution_returned(fixture.base(), fixture.locals[0], UINT32_MAX, 0, 0);
    const auto *event = scheduler_state_at<SchedulerProtocolExecutionTrace>(
        fixture.base(), fixture.storage.layout.protocol.executions_offset
    );
    EXPECT_EQ(event->returned.sequence, 0u);
    EXPECT_EQ(event->visible.sequence, 0u);
    const auto *worker = scheduler_state_at<SchedulerProtocolWorkerTrace>(
        fixture.base(), fixture.storage.layout.protocol.workers_offset
    );
    EXPECT_EQ(worker->next_sequence, 0u);
    EXPECT_EQ(worker->error_flags, SCHEDULER_PROTOCOL_INVALID_IDENTITY);
}

TEST(HbgOrderedProtocol, ConsumedPermitPreventsPayloadSnapshotFromBeingReplaced) {
    OrderedFixture fixture({{1, LaneMask::aic()}}, true);
    ASSERT_TRUE(fixture.enqueue(0, 0));
    ASSERT_TRUE(fixture.fill(0));
    OrderedExecution execution{};
    ASSERT_TRUE(fixture.begin(0, &execution));
    const auto *events = scheduler_state_at<SchedulerProtocolExecutionTrace>(
        fixture.base(), fixture.storage.layout.protocol.executions_offset
    );
    const auto first = events[0];
    SchedulerReadyClaim ready{};
    ready.task_ref = 0;
    SchedulerFreeSlotClaim claim{0, execution.slot, execution.generation, 0};
    scheduler_protocol_payload_publication(fixture.base(), fixture.locals[0], claim, ready, true);
    EXPECT_EQ(events[0].publication_count, first.publication_count);
    EXPECT_EQ(events[0].publication_before_cycles, first.publication_before_cycles);
    EXPECT_EQ(events[0].publication_after_cycles, first.publication_after_cycles);
    EXPECT_EQ(events[0].entry.sequence, first.entry.sequence);
    const auto *workers = scheduler_state_at<SchedulerProtocolWorkerTrace>(
        fixture.base(), fixture.storage.layout.protocol.workers_offset
    );
    EXPECT_EQ(workers[0].error_flags, SCHEDULER_PROTOCOL_DUPLICATE);
}

TEST(HbgOrderedProtocol, FinishedWorkerCannotPublishEvenAnUnsequencedPayloadSnapshot) {
    OrderedFixture fixture({{1, LaneMask::aic()}}, true);
    auto *local = fixture.locals[0];
    scheduler_protocol_finish(fixture.base(), local, true);
    SchedulerReadyClaim ready{};
    ready.task_ref = 0;
    SchedulerFreeSlotClaim claim{0, 0, 0, 0};
    scheduler_protocol_payload_publication(fixture.base(), local, claim, ready, true);
    const auto *events = scheduler_state_at<SchedulerProtocolExecutionTrace>(
        fixture.base(), fixture.storage.layout.protocol.executions_offset
    );
    EXPECT_EQ(events[0].publication_count, 0u);
    const auto *workers = scheduler_state_at<SchedulerProtocolWorkerTrace>(
        fixture.base(), fixture.storage.layout.protocol.workers_offset
    );
    EXPECT_EQ(workers[0].next_sequence, 0u);
    EXPECT_EQ(workers[0].error_flags, SCHEDULER_PROTOCOL_AFTER_FINISH);
}

TEST(HbgOrderedProtocol, CooperativePayloadCannotBeRepublishedBeforeItsFirstPermit) {
    OrderedFixture fixture({{2, LaneMask::aic()}}, true);
    ASSERT_TRUE(fixture.admit(0, 0));
    ASSERT_TRUE(fixture.stage_one(0, 0));
    const auto *events = scheduler_state_at<SchedulerProtocolExecutionTrace>(
        fixture.base(), fixture.storage.layout.protocol.executions_offset
    );
    ASSERT_EQ(events[0].permit.sequence, 0u);
    const auto first = events[0];
    ++fixture.locals[0]->slots[0].generation;
    SchedulerReadyClaim ready{};
    ready.task_ref = 0;
    SchedulerFreeSlotClaim claim{0, 0, fixture.locals[0]->slots[0].generation, 0};
    scheduler_protocol_payload_publication(fixture.base(), fixture.locals[0], claim, ready, true);
    EXPECT_EQ(events[0].publication_count, first.publication_count);
    EXPECT_EQ(events[0].generation, first.generation);
    const auto *workers = scheduler_state_at<SchedulerProtocolWorkerTrace>(
        fixture.base(), fixture.storage.layout.protocol.workers_offset
    );
    EXPECT_EQ(workers[0].error_flags, SCHEDULER_PROTOCOL_DUPLICATE);
}

}  // namespace
