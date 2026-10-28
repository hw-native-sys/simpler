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
#include <map>
#include <set>
#include <tuple>
#include <vector>

#include "hbg_scheduler_test_support.h"
#include "scheduler/scheduler_gm_mix.h"
#include "scheduler/scheduler_dispatch.h"
#include "scheduler/scheduler_protocol_execute.h"

namespace {

struct GmShape {
    uint16_t blocks;
    uint8_t mask;
    bool sync_start{false};
};

uint64_t gm_trace_capacity(const std::vector<GmShape> &shapes) {
    uint64_t maximum = 1;
    for (const auto &shape : shapes)
        maximum = std::max<uint64_t>(maximum, static_cast<uint64_t>(shape.blocks) * 3);
    return maximum;
}

SchedulerSpmdRequirements gm_requirements(const std::vector<GmShape> &shapes, uint32_t clusters) {
    SchedulerSpmdRequirements result{clusters * 3, clusters, 0, 0};
    for (const auto &shape : shapes) {
        const bool mix = __builtin_popcount(shape.mask) > 1;
        if (mix || shape.blocks > 1) ++result.cooperative_task_count;
        if (mix) ++result.mix_task_count;
    }
    return result;
}

struct GmExecution {
    int64_t task{SCHEDULER_TASK_REF_INVALID};
    uint32_t block{0};
    uint32_t blocks{0};
    uint32_t worker{0};
    uint32_t slot{0};
    uint32_t generation{0};
    uint8_t subtask{UINT8_MAX};
    uint32_t physical_subblock{0};
};

struct GmMixFixture {
    explicit GmMixFixture(const std::vector<GmShape> &shapes, bool protocol_profiling = false, uint32_t clusters = 2) :
        storage(
            shapes.size(), clusters * 3, gm_trace_capacity(shapes), gm_requirements(shapes, clusters),
            protocol_profiling
        ),
        graph(shapes.size()),
        admissions(clusters * 3),
        locals(clusters * 3),
        cursors(clusters * 3) {
        uint32_t trace = 0;
        for (uint64_t task = 0; task < shapes.size(); ++task) {
            const auto shape = shapes[task];
            graph.mixed(task, shape.mask);
            auto &metadata = storage.metadata[task];
            metadata.active_mask = shape.mask;
            metadata.logical_block_num = shape.blocks;
            metadata.flags = SCHEDULER_TASK_EXECUTABLE;
            if (shape.blocks > 1) metadata.flags |= SCHEDULER_TASK_SPMD;
            if (shape.sync_start) metadata.flags |= SCHEDULER_TASK_SYNC_START;
            if (__builtin_popcount(shape.mask) > 1) metadata.flags |= SCHEDULER_TASK_MIX;
            metadata.trace_index_base = trace;
            trace += shape.blocks * static_cast<uint32_t>(__builtin_popcount(shape.mask));
            metadata.timing_slot = -1;
            for (uint8_t subtask = 0; subtask < 3; ++subtask)
                metadata.kernel_ids[subtask] = (shape.mask & (1U << subtask)) != 0 ? 1 : UINT16_MAX;
        }
        EXPECT_TRUE(scheduler_init_spmd_task_records(base(), storage.layout, 47));
        storage.ssbuf_region = nullptr;
        storage.run_control->active_worker_count = clusters * 3;
        storage.run_control->aic_active_worker_count = clusters;
        storage.run_control->aiv_active_worker_count = clusters * 2;
        storage.run_control->expected_task_count = shapes.size();
        for (uint32_t worker = 0; worker < locals.size(); ++worker) {
            auto &context = storage.contexts[worker];
            context.is_scheduler = 1;
            context.scheduler_count = clusters * 3;
            for (uint32_t lane = 0; lane < 3; ++lane)
                context.cluster_worker_ids[lane] = (worker / 3) * 3 + lane;
            storage.test_graphs[worker] = graph.graph();
        }
        for (uint32_t worker = 0; worker < locals.size(); ++worker) {
            locals[worker] = storage.local_context(&storage.contexts[worker]);
            locals[worker]->has_mix = storage.layout.spmd.mix_task_count != 0;
            EXPECT_TRUE(scheduler_initialize_admission(base(), locals[worker], &admissions[worker]));
            locals[worker]->admission = &admissions[worker];
        }
    }

    void *base() const { return storage.scheduler_state->base(); }
    SchedulerRunControl *run() const { return storage.run_control; }
    SchedulerTaskControl *control(uint64_t task) const {
        return scheduler_state_at<SchedulerTaskControl>(base(), storage.layout.task_controls_offset) + task;
    }
    SchedulerGroupMailbox *mailbox(uint32_t cluster) const {
        return scheduler_state_at<SchedulerGroupMailbox>(base(), storage.layout.group_mailboxes_offset) + cluster;
    }
    uint64_t &seal(uint64_t task, uint32_t cluster) const {
        const auto *record = scheduler_state_at<SchedulerAdmissionRecord>(
            base(), storage.layout.admission_records_offset + task * storage.layout.admission_record_stride
        );
        return scheduler_state_at<uint64_t>(base(), record->mix_close_seals_offset)[cluster];
    }
    bool enqueue(uint64_t task, uint32_t worker = 0) {
        auto *local = locals[worker];
        if (scheduler_route_task(graph.graph(), base(), local, run(), task, nullptr) !=
            SchedulerRouteResult::READY_TO_ENQUEUE)
            return false;
        SchedulerReadyBatch batch{};
        const auto &metadata = storage.metadata[task];
        return scheduler_ready_batch_append(base(), local, task, &batch, nullptr) &&
               scheduler_ready_batch_push(
                   base(), local, scheduler_task_ready_queue(metadata.flags, metadata.active_mask), &batch, nullptr
               );
    }
    bool admit(uint64_t task, uint32_t worker = 0) {
        if (!enqueue(task, worker)) return false;
        auto *local = locals[worker];
        const auto &metadata = storage.metadata[task];
        const uint32_t queue = scheduler_task_ready_queue(metadata.flags, metadata.active_mask);
        SchedulerReadyClaim ready{};
        ready.inbox_index = scheduler_ready_push_target(local, queue, task);
        return scheduler_ready_pop_from_inbox(
                   graph.graph(), base(), local, run(), queue, ready.inbox_index, &ready.task_ref, nullptr
               ) &&
               ready.task_ref == static_cast<int64_t>(task) &&
               scheduler_try_claim_admission(graph.graph(), base(), local, run(), &admissions[worker], ready) ==
                   SchedulerAdmissionResult::PROGRESS &&
               scheduler_prepare_admission_record(graph.graph(), base(), local, run(), &admissions[worker]) &&
               scheduler_try_commit_admission(graph.graph(), base(), local, run(), &admissions[worker]) ==
                   SchedulerAdmissionResult::PROGRESS;
    }
    bool refresh(uint32_t worker) {
        return scheduler_refresh_local_obligation(graph.graph(), base(), locals[worker], run(), &admissions[worker]);
    }
    bool refresh_all() {
        for (uint32_t worker = 0; worker < locals.size(); ++worker)
            if (!refresh(worker)) return false;
        return true;
    }
    SchedulerBlockClaimResult claim(uint32_t cluster) {
        return scheduler_gm_mix_claim(graph.graph(), base(), locals[cluster * 3], run());
    }
    bool publish(uint32_t cluster) {
        return scheduler_gm_mix_publish(graph.graph(), base(), locals[cluster * 3], run());
    }
    bool observe(uint32_t worker, bool *progress) {
        return scheduler_gm_mix_observe(graph.graph(), base(), locals[worker], run(), progress);
    }
    bool arrive(uint32_t worker, bool *progress) {
        return scheduler_gm_mix_arrive(graph.graph(), base(), locals[worker], run(), progress);
    }
    bool prepare(uint32_t worker, bool *progress) {
        return scheduler_gm_mix_prepare_lane(graph.graph(), base(), locals[worker], run(), 0, progress);
    }
    bool rendezvous(uint32_t cluster) {
        bool progress = false;
        for (uint32_t lane = 1; lane < 3; ++lane)
            if (!observe(cluster * 3 + lane, &progress)) return false;
        for (uint32_t lane = 0; lane < 3; ++lane)
            if (!arrive(cluster * 3 + lane, &progress)) return false;
        for (uint32_t lane = 0; lane < 3; ++lane)
            if (!prepare(cluster * 3 + lane, &progress)) return false;
        return true;
    }
    SchedulerGmMixResult retire(uint32_t cluster) {
        return scheduler_gm_mix_retire(graph.graph(), base(), locals[cluster * 3], run());
    }
    bool complete_task(uint64_t task, uint32_t cluster) {
        return scheduler_complete_ordered_task(
            graph.graph(), base(), locals[cluster * 3], run(), task, 0, nullptr, nullptr, nullptr
        );
    }
    bool close(uint32_t worker, bool *closed) {
        return scheduler_gm_mix_close(graph.graph(), base(), locals[worker], run(), closed);
    }
    bool fill(uint32_t worker) {
        bool failed = false;
        scheduler_fill_own_slots(
            graph.graph(), base(), locals[worker], run(), cursors[worker].data(), nullptr, 0, 0, &failed, nullptr
        );
        return !failed;
    }
    bool begin(uint32_t worker, GmExecution *execution) {
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
            local->slots[slot].subtask_slot,
            static_cast<uint32_t>(payload->global_context.sub_block_id)
        };
        return true;
    }
    bool complete(const GmExecution &execution) {
        auto *local = locals[execution.worker];
        local->local_completed_generations[execution.slot] = execution.generation;
        return scheduler_service_cluster_completions(
            graph.graph(), base(), local, run(), nullptr, nullptr, nullptr, 0, nullptr, nullptr
        );
    }
    bool finish_lanes(uint32_t cluster) {
        const auto descriptor = mailbox(cluster)->descriptor;
        for (uint32_t lane = 0; lane < 3; ++lane) {
            GmExecution execution{};
            const bool active = descriptor.lane_subtasks[lane] != UINT8_MAX;
            const bool entered = begin(cluster * 3 + lane, &execution);
            if (active != entered || (entered && !complete(execution))) return false;
        }
        return true;
    }
    bool drain(uint32_t worker, bool *done) {
        return scheduler_drain_ordered_exit(graph.graph(), base(), locals[worker], run(), done);
    }

    scheduler_test::FixtureStorage storage;
    scheduler_test::GraphBuffer graph;
    std::vector<SchedulerAdmissionLocalState> admissions;
    std::vector<SchedulerLocalState *> locals;
    std::vector<std::array<uint64_t, SCHEDULER_READY_QUEUE_COUNT>> cursors;
};

class HbgGmMixContinuous : public ::testing::TestWithParam<uint8_t> {};

TEST_P(HbgGmMixContinuous, TicketExhaustionDoesNotSealAnUnpublishedRemoteReservation) {
    for (uint16_t count : {2u, 3u, 5u}) {
        GmMixFixture fixture({{count, GetParam()}, {1, GetParam()}}, true);
        ASSERT_TRUE(fixture.admit(0));
        ASSERT_TRUE(fixture.admit(1));
        ASSERT_TRUE(fixture.refresh_all());
        scheduler_gm_store(fixture.control(0)->claim_tickets, static_cast<uint64_t>(count));
        scheduler_gm_store(fixture.control(0)->next_block, static_cast<uint64_t>(count - 1));
        scheduler_gm_store(fixture.control(0)->completed_blocks, static_cast<uint64_t>(count - 1));
        for (uint32_t repeat = 0; repeat < 16; ++repeat)
            ASSERT_EQ(fixture.claim(1), SchedulerBlockClaimResult::EXHAUSTED);
        EXPECT_EQ(fixture.control(0)->claim_tickets, count + 1u);
        EXPECT_EQ(fixture.mailbox(1)->published_seq, 0u);
        for (uint32_t worker : {3u, 4u, 5u}) {
            bool closed = true;
            ASSERT_TRUE(fixture.close(worker, &closed));
            EXPECT_FALSE(closed);
        }
        EXPECT_EQ(fixture.seal(0, 1), 0u);
        scheduler_gm_fetch_add(fixture.control(0)->next_block, UINT64_C(1));
        for (uint32_t worker : {3u, 4u, 5u}) {
            bool closed = false;
            ASSERT_TRUE(fixture.close(worker, &closed));
            EXPECT_TRUE(closed);
            EXPECT_FALSE(fixture.admissions[worker].ticket_exhausted);
            ASSERT_TRUE(fixture.refresh(worker));
            EXPECT_EQ(fixture.admissions[worker].current_task_ref, 1);
        }
        EXPECT_EQ(fixture.control(0)->completed_blocks, count - 1u);
        EXPECT_EQ(fixture.run()->resolved_task_count, 0u);
    }
}

TEST_P(HbgGmMixContinuous, ReleasedGroupPublishesNextBlockInTheSameFill) {
    GmMixFixture fixture({{3, GetParam()}});
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(0));
    ASSERT_TRUE(fixture.rendezvous(0));
    ASSERT_TRUE(fixture.finish_lanes(0));

    ASSERT_TRUE(fixture.fill(0));
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.control(0)->completed_blocks, UINT64_C(0)), 1u);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.control(0)->next_block, UINT64_C(0)), 2u);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.mailbox(0)->published_seq, UINT64_C(0)), 2u);
    EXPECT_EQ(fixture.admissions[0].gm_mix.phase, SchedulerGmMixPhase::ARRIVED);
    EXPECT_EQ(fixture.mailbox(0)->descriptor.block_idx, 1u);
    GmExecution execution{};
    EXPECT_FALSE(fixture.begin(0, &execution));
}

TEST_P(HbgGmMixContinuous, LastLocalGroupPublishesSuccessorWhileRemotePredecessorStillRuns) {
    GmMixFixture fixture({{2, GetParam()}, {1, GetParam()}});
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.admit(1, 3));
    ASSERT_TRUE(fixture.refresh_all());
    for (uint32_t cluster = 0; cluster < 2; ++cluster) {
        ASSERT_EQ(fixture.claim(cluster), SchedulerBlockClaimResult::CLAIMED);
        ASSERT_TRUE(fixture.publish(cluster));
        ASSERT_TRUE(fixture.rendezvous(cluster));
    }
    ASSERT_TRUE(fixture.finish_lanes(0));
    ASSERT_TRUE(fixture.fill(0));
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.control(0)->completed_blocks, UINT64_C(0)), 1u);
    EXPECT_NE(scheduler_gm_load(fixture.control(0)->state), static_cast<int64_t>(SchedulerTaskState::DONE));
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.seal(0, 0), UINT64_C(0)), 2u);
    EXPECT_EQ(fixture.admissions[0].current_task_ref, 1);
    EXPECT_EQ(fixture.mailbox(0)->descriptor.task_ref, 1);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.mailbox(0)->published_seq, UINT64_C(0)), 2u);

    for (uint32_t worker : {1u, 2u}) {
        ASSERT_EQ(fixture.admissions[worker].current_task_ref, 0);
        ASSERT_TRUE(fixture.fill(worker));
        EXPECT_EQ(fixture.admissions[worker].current_task_ref, 1);
        EXPECT_EQ(fixture.admissions[worker].gm_mix.group_seq, 2u);
    }
    EXPECT_EQ(fixture.admissions[3].current_task_ref, 0);
}

TEST_P(HbgGmMixContinuous, MissingReleaseStopsContinuousProgressWithoutTakingAnotherBlock) {
    GmMixFixture fixture({{3, GetParam()}});
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(0));
    ASSERT_TRUE(fixture.rendezvous(0));
    const uint32_t late_lane = (GetParam() & 4) != 0 ? 2 : 1;
    GmExecution late{};
    ASSERT_TRUE(fixture.begin(late_lane, &late));
    for (uint32_t lane = 0; lane < 3; ++lane) {
        if (lane == late_lane || (GetParam() & (1U << lane)) == 0) continue;
        GmExecution execution{};
        ASSERT_TRUE(fixture.begin(lane, &execution));
        ASSERT_TRUE(fixture.complete(execution));
    }
    ASSERT_TRUE(fixture.fill(0));
    EXPECT_EQ(fixture.admissions[0].gm_mix.phase, SchedulerGmMixPhase::RELEASED);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.control(0)->completed_blocks, UINT64_C(0)), 0u);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.control(0)->next_block, UINT64_C(0)), 1u);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.mailbox(0)->published_seq, UINT64_C(0)), 1u);
    ASSERT_TRUE(fixture.complete(late));
    ASSERT_TRUE(fixture.fill(0));
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.control(0)->next_block, UINT64_C(0)), 2u);
}

TEST_P(HbgGmMixContinuous, ErrorBeforeRetirementDoesNotPublishOrContribute) {
    GmMixFixture fixture({{3, GetParam()}});
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(0));
    ASSERT_TRUE(fixture.rendezvous(0));
    ASSERT_TRUE(fixture.finish_lanes(0));
    scheduler_gm_store(fixture.run()->scheduler_error, UINT64_C(1));
    EXPECT_FALSE(fixture.fill(0));
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.control(0)->completed_blocks, UINT64_C(0)), 0u);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.control(0)->next_block, UINT64_C(0)), 1u);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.mailbox(0)->published_seq, UINT64_C(0)), 1u);
}

TEST_P(HbgGmMixContinuous, DirectSuccessorFromRetirementKeepsItsAdmissionRequirement) {
    for (uint16_t blocks : {1, 2, 8, 65}) {
        SCOPED_TRACE(blocks);
        GmMixFixture fixture({{1, GetParam()}, {blocks, 1}});
        fixture.graph.executable(1, 0, {0});
        fixture.storage.metadata[1].flags |= SCHEDULER_TASK_HAS_FANIN;
        ASSERT_TRUE(scheduler_init_spmd_task_records(fixture.base(), fixture.storage.layout, 47));
        ASSERT_EQ(
            scheduler_route_task(fixture.graph.graph(), fixture.base(), fixture.locals[0], fixture.run(), 1, nullptr),
            SchedulerRouteResult::WAITING
        );
        ASSERT_TRUE(fixture.admit(0));
        ASSERT_TRUE(fixture.refresh_all());
        ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
        ASSERT_TRUE(fixture.publish(0));
        ASSERT_TRUE(fixture.rendezvous(0));
        ASSERT_TRUE(fixture.finish_lanes(0));
        ASSERT_TRUE(fixture.fill(0));
        ASSERT_EQ(scheduler_gm_load(fixture.control(0)->state), static_cast<int64_t>(SchedulerTaskState::DONE));
        if (blocks > 1) {
            for (const auto &slot : fixture.locals[0]->slots)
                ASSERT_FALSE(slot.task_ref == 1 && slot.state == SchedulerDispatchSlotState::READY);
        }
        std::set<uint32_t> executed;
        for (uint32_t step = 0; step < 200; ++step) {
            for (uint32_t worker = 0; worker < 6; ++worker) {
                ASSERT_TRUE(fixture.fill(worker));
                GmExecution execution{};
                if (!fixture.begin(worker, &execution)) continue;
                ASSERT_EQ(execution.task, 1);
                EXPECT_EQ(execution.blocks, blocks);
                EXPECT_EQ(fixture.locals[worker]->slots[execution.slot].cooperative, blocks > 1);
                EXPECT_TRUE(executed.insert(execution.block).second);
                ASSERT_TRUE(fixture.complete(execution));
            }
            if (scheduler_gm_load(fixture.control(1)->state) == static_cast<int64_t>(SchedulerTaskState::DONE)) break;
        }
        EXPECT_EQ(executed.size(), blocks);
        EXPECT_EQ(scheduler_gm_load(fixture.control(1)->state), static_cast<int64_t>(SchedulerTaskState::DONE));
    }
}

INSTANTIATE_TEST_SUITE_P(AllMixMasks, HbgGmMixContinuous, ::testing::Values<uint8_t>(3, 5, 6, 7));

TEST(HbgGmMix, ZeroWorkContinuousProgressHasABoundedNumberOfObligations) {
    GmMixFixture fixture({{1, 7}, {1, 7}, {1, 7}});
    for (uint32_t task = 0; task < 3; ++task) {
        ASSERT_TRUE(fixture.admit(task));
        ASSERT_TRUE(fixture.refresh(3));
        ASSERT_EQ(fixture.claim(1), SchedulerBlockClaimResult::CLAIMED);
        ASSERT_TRUE(fixture.publish(1));
        for (uint32_t worker : {4u, 5u})
            ASSERT_TRUE(fixture.refresh(worker));
        ASSERT_TRUE(fixture.rendezvous(1));
        ASSERT_TRUE(fixture.finish_lanes(1));
        ASSERT_EQ(fixture.retire(1), SchedulerGmMixResult::TASK_LAST);
        ASSERT_TRUE(fixture.complete_task(task, 1));
        for (uint32_t worker : {3u, 4u, 5u}) {
            bool closed = false;
            ASSERT_TRUE(fixture.close(worker, &closed));
            ASSERT_TRUE(closed);
        }
    }
    ASSERT_TRUE(fixture.fill(0));
    EXPECT_EQ(fixture.admissions[0].scan_cursor, SCHEDULER_PENDING_SLOT_COUNT);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.seal(2, 0), UINT64_C(0)), 0u);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.mailbox(0)->published_seq, UINT64_C(0)), 0u);
    ASSERT_TRUE(fixture.fill(0));
    EXPECT_EQ(fixture.admissions[0].scan_cursor, 3u);
}

class HbgGmMixSyncStart : public ::testing::TestWithParam<uint8_t> {};

class HbgGmMixIdlePoll : public ::testing::TestWithParam<std::tuple<uint8_t, uint32_t>> {};

void clear_drained_ready_directories(GmMixFixture &fixture) {
    for (auto *local : fixture.locals) {
        for (uint32_t type = 0; type < SCHEDULER_READY_QUEUE_COUNT; ++type) {
            if (scheduler_ready_queue_holder(type, local->config.self_core_type))
                ASSERT_TRUE(scheduler_refresh_ready_inbox_type(fixture.base(), local, type));
        }
    }
}

TEST_P(HbgGmMixIdlePoll, UnpublishedClaimIsIdleButDescriptorAndRetainedSealWakeTheVector) {
    const auto [mask, lane] = GetParam();
    GmMixFixture fixture({{1, mask}});
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_NO_FATAL_FAILURE(clear_drained_ready_directories(fixture));
    auto *local = fixture.locals[lane];
    EXPECT_FALSE(scheduler_ordered_poll_may_progress(fixture.graph.graph(), fixture.base(), local));
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    EXPECT_FALSE(scheduler_ordered_poll_may_progress(fixture.graph.graph(), fixture.base(), local));
    ASSERT_TRUE(fixture.fill(lane));
    EXPECT_EQ(fixture.admissions[lane].current_task_ref, 0);
    EXPECT_EQ(fixture.admissions[lane].gm_mix.phase, SchedulerGmMixPhase::IDLE);
    ASSERT_TRUE(fixture.publish(0));
    EXPECT_TRUE(scheduler_ordered_poll_may_progress(fixture.graph.graph(), fixture.base(), local));
    ASSERT_TRUE(fixture.rendezvous(0));
    ASSERT_TRUE(fixture.finish_lanes(0));
    EXPECT_FALSE(scheduler_ordered_poll_may_progress(fixture.graph.graph(), fixture.base(), local));
    ASSERT_EQ(fixture.retire(0), SchedulerGmMixResult::TASK_LAST);
    bool closed = false;
    ASSERT_TRUE(fixture.close(0, &closed));
    ASSERT_TRUE(closed);
    EXPECT_TRUE(scheduler_ordered_poll_may_progress(fixture.graph.graph(), fixture.base(), local));
    ASSERT_TRUE(fixture.fill(lane));
    EXPECT_EQ(fixture.admissions[lane].current_task_ref, SCHEDULER_TASK_REF_INVALID);
    EXPECT_FALSE(scheduler_ordered_poll_may_progress(fixture.graph.graph(), fixture.base(), local));
}

TEST_P(HbgGmMixIdlePoll, WaitingForPartnersDoesNotHideAnOrdinaryReadyTask) {
    const auto [mask, lane] = GetParam();
    GmMixFixture fixture({{1, mask}, {1, 2}});
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_NO_FATAL_FAILURE(clear_drained_ready_directories(fixture));
    auto *local = fixture.locals[lane];
    EXPECT_FALSE(scheduler_ordered_poll_may_progress(fixture.graph.graph(), fixture.base(), local));
    ASSERT_TRUE(fixture.enqueue(1, lane));
    EXPECT_TRUE(scheduler_ordered_poll_may_progress(fixture.graph.graph(), fixture.base(), local));
    ASSERT_TRUE(fixture.fill(lane));
    GmExecution execution{};
    EXPECT_FALSE(fixture.begin(lane, &execution));
    EXPECT_EQ(fixture.control(1)->completed_blocks, 0u);
}

INSTANTIATE_TEST_SUITE_P(
    MasksAndVectors, HbgGmMixIdlePoll,
    ::testing::Combine(::testing::Values<uint8_t>(3, 5, 6, 7), ::testing::Values<uint32_t>(1, 2))
);

TEST_P(HbgGmMixSyncStart, IdlePollRequiresAllArrivalAndReleaseAcknowledgements) {
    GmMixFixture fixture({{1, GetParam()}});
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_NO_FATAL_FAILURE(clear_drained_ready_directories(fixture));
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(0));
    bool progress = false;
    ASSERT_TRUE(fixture.arrive(0, &progress));
    EXPECT_FALSE(scheduler_ordered_poll_may_progress(fixture.graph.graph(), fixture.base(), fixture.locals[0]));
    ASSERT_TRUE(fixture.observe(1, &progress));
    ASSERT_TRUE(fixture.arrive(1, &progress));
    EXPECT_FALSE(scheduler_ordered_poll_may_progress(fixture.graph.graph(), fixture.base(), fixture.locals[0]));
    ASSERT_TRUE(fixture.observe(2, &progress));
    ASSERT_TRUE(fixture.arrive(2, &progress));
    EXPECT_TRUE(scheduler_ordered_poll_may_progress(fixture.graph.graph(), fixture.base(), fixture.locals[0]));
    ASSERT_TRUE(fixture.rendezvous(0));
    if ((GetParam() & 1U) != 0) {
        GmExecution execution{};
        ASSERT_TRUE(fixture.begin(0, &execution));
        ASSERT_TRUE(fixture.complete(execution));
    }
    EXPECT_FALSE(scheduler_ordered_poll_may_progress(fixture.graph.graph(), fixture.base(), fixture.locals[0]));
    for (uint32_t lane = 1; lane < 3; ++lane) {
        if ((GetParam() & (1U << lane)) == 0) continue;
        GmExecution execution{};
        ASSERT_TRUE(fixture.begin(lane, &execution));
        ASSERT_TRUE(fixture.complete(execution));
    }
    EXPECT_TRUE(scheduler_ordered_poll_may_progress(fixture.graph.graph(), fixture.base(), fixture.locals[0]));
    ASSERT_TRUE(fixture.fill(0));
    EXPECT_EQ(fixture.control(0)->completed_blocks, 1u);
}

struct GmServiceCase {
    uint32_t worker;
    SchedulerGmMixPhase phase;
};

void prepare_service_phase(GmMixFixture &fixture, const GmServiceCase &scenario) {
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    if (scenario.phase == SchedulerGmMixPhase::IDLE) return;
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    if (scenario.phase == SchedulerGmMixPhase::CLAIMED) return;
    ASSERT_TRUE(fixture.publish(0));
    bool progress = false;
    if (scenario.worker != 0) ASSERT_TRUE(fixture.observe(scenario.worker, &progress));
    if (scenario.phase == SchedulerGmMixPhase::PUBLISHED) return;
    if (scenario.phase == SchedulerGmMixPhase::ARRIVED) {
        ASSERT_TRUE(fixture.arrive(scenario.worker, &progress));
        return;
    }
    ASSERT_TRUE(fixture.rendezvous(0));
    if (scenario.phase == SchedulerGmMixPhase::RELEASED) {
        GmExecution execution{};
        ASSERT_TRUE(fixture.begin(scenario.worker, &execution));
        ASSERT_TRUE(fixture.complete(execution));
    }
}

class HbgGmMixServiceError : public ::testing::TestWithParam<GmServiceCase> {};

TEST_P(HbgGmMixServiceError, ServiceRejectsPublishedErrorBeforeAnyStateSpecificWork) {
    GmMixFixture fixture({{2, 7}});
    const auto scenario = GetParam();
    ASSERT_NO_FATAL_FAILURE(prepare_service_phase(fixture, scenario));
    ASSERT_EQ(fixture.admissions[scenario.worker].gm_mix.phase, scenario.phase);
    const auto before = fixture.admissions[scenario.worker].gm_mix;
    const uint64_t cursor = fixture.control(0)->next_block;
    const uint64_t published = fixture.mailbox(0)->published_seq;
    const uint8_t ready_mask = fixture.locals[scenario.worker]->local_ready_mask;
    scheduler_record_error(
        fixture.run(), 0, SchedulerGraphResult::TIMEOUT, nullptr, fixture.locals[4],
        SchedulerErrorSite::ORDERED_EXIT_TIMEOUT
    );
    bool progress = true;
    EXPECT_FALSE(scheduler_service_gm_mix(
        fixture.graph.graph(), fixture.base(), fixture.locals[scenario.worker], fixture.run(), 0, nullptr, nullptr,
        nullptr, &progress
    ));
    EXPECT_FALSE(progress);
    EXPECT_EQ(fixture.admissions[scenario.worker].gm_mix.phase, before.phase);
    EXPECT_EQ(fixture.admissions[scenario.worker].gm_mix.group_seq, before.group_seq);
    EXPECT_EQ(fixture.admissions[scenario.worker].gm_mix.slot_index, before.slot_index);
    EXPECT_EQ(fixture.control(0)->next_block, cursor);
    EXPECT_EQ(fixture.control(0)->completed_blocks, 0u);
    EXPECT_EQ(fixture.mailbox(0)->published_seq, published);
    EXPECT_EQ(fixture.locals[scenario.worker]->local_ready_mask, ready_mask);
    EXPECT_EQ(fixture.seal(0, 0), 0u);
    EXPECT_EQ(fixture.run()->error_worker_index, 4u);
    EXPECT_EQ(fixture.run()->error_site, static_cast<uint64_t>(SchedulerErrorSite::ORDERED_EXIT_TIMEOUT));
}

INSTANTIATE_TEST_SUITE_P(
    CoordinatorAndVectorPhases, HbgGmMixServiceError,
    ::testing::Values(
        GmServiceCase{0, SchedulerGmMixPhase::IDLE}, GmServiceCase{0, SchedulerGmMixPhase::CLAIMED},
        GmServiceCase{0, SchedulerGmMixPhase::PUBLISHED}, GmServiceCase{0, SchedulerGmMixPhase::ARRIVED},
        GmServiceCase{0, SchedulerGmMixPhase::RUNNING}, GmServiceCase{0, SchedulerGmMixPhase::RELEASED},
        GmServiceCase{1, SchedulerGmMixPhase::IDLE}, GmServiceCase{1, SchedulerGmMixPhase::PUBLISHED},
        GmServiceCase{1, SchedulerGmMixPhase::ARRIVED}, GmServiceCase{1, SchedulerGmMixPhase::RUNNING},
        GmServiceCase{1, SchedulerGmMixPhase::RELEASED}
    )
);

TEST(HbgGmMix, ServiceCarriesNewPublicationThroughArrivalWithoutAnExtraPoll) {
    for (uint8_t mask : {3, 5, 6, 7}) {
        SCOPED_TRACE(mask);
        GmMixFixture fixture({{2, mask}});
        ASSERT_TRUE(fixture.admit(0));
        ASSERT_TRUE(fixture.refresh_all());
        for (uint32_t worker = 0; worker < 3; ++worker) {
            bool progress = false;
            ASSERT_TRUE(scheduler_service_gm_mix(
                fixture.graph.graph(), fixture.base(), fixture.locals[worker], fixture.run(), 0, nullptr, nullptr,
                nullptr, &progress
            ));
            EXPECT_TRUE(progress);
            EXPECT_EQ(fixture.mailbox(0)->published_seq, 1u);
            EXPECT_EQ(fixture.mailbox(0)->lanes[worker].arrived_seq, 1u);
            EXPECT_EQ(
                fixture.admissions[worker].gm_mix.phase, worker < 2 ? SchedulerGmMixPhase::ARRIVED :
                                                         (mask & 4) ? SchedulerGmMixPhase::RUNNING :
                                                                      SchedulerGmMixPhase::RELEASED
            );
        }
        for (uint32_t worker = 0; worker < 2; ++worker) {
            bool progress = false;
            ASSERT_TRUE(scheduler_service_gm_mix(
                fixture.graph.graph(), fixture.base(), fixture.locals[worker], fixture.run(), 0, nullptr, nullptr,
                nullptr, &progress
            ));
            EXPECT_TRUE(progress);
            EXPECT_EQ(
                fixture.admissions[worker].gm_mix.phase,
                (mask & (1U << worker)) != 0 ? SchedulerGmMixPhase::RUNNING : SchedulerGmMixPhase::RELEASED
            );
        }
        EXPECT_EQ(fixture.control(0)->next_block, 1u);
        EXPECT_EQ(fixture.control(0)->completed_blocks, 0u);
    }
}

TEST(HbgGmMix, ServiceRetiresOnlyAfterAllReleasesAndPublishesTheNextGroupWithoutWaiting) {
    GmMixFixture fixture({{2, 7}});
    ASSERT_NO_FATAL_FAILURE(prepare_service_phase(fixture, {0, SchedulerGmMixPhase::RELEASED}));
    bool progress = true;
    ASSERT_TRUE(scheduler_service_gm_mix(
        fixture.graph.graph(), fixture.base(), fixture.locals[0], fixture.run(), 0, nullptr, nullptr, nullptr, &progress
    ));
    EXPECT_FALSE(progress);
    EXPECT_EQ(fixture.admissions[0].gm_mix.phase, SchedulerGmMixPhase::RELEASED);
    EXPECT_EQ(fixture.control(0)->completed_blocks, 0u);
    for (uint32_t worker : {1u, 2u}) {
        GmExecution execution{};
        ASSERT_TRUE(fixture.begin(worker, &execution));
        ASSERT_TRUE(fixture.complete(execution));
    }
    ASSERT_TRUE(scheduler_service_gm_mix(
        fixture.graph.graph(), fixture.base(), fixture.locals[0], fixture.run(), 0, nullptr, nullptr, nullptr, &progress
    ));
    EXPECT_TRUE(progress);
    EXPECT_EQ(fixture.admissions[0].gm_mix.phase, SchedulerGmMixPhase::ARRIVED);
    EXPECT_EQ(fixture.control(0)->next_block, 2u);
    EXPECT_EQ(fixture.control(0)->completed_blocks, 1u);
    EXPECT_EQ(fixture.mailbox(0)->published_seq, 2u);
    ASSERT_TRUE(scheduler_service_gm_mix(
        fixture.graph.graph(), fixture.base(), fixture.locals[0], fixture.run(), 0, nullptr, nullptr, nullptr, &progress
    ));
    EXPECT_FALSE(progress);
    EXPECT_EQ(fixture.admissions[0].gm_mix.phase, SchedulerGmMixPhase::ARRIVED);
    EXPECT_EQ(fixture.control(0)->next_block, 2u);
    EXPECT_EQ(fixture.mailbox(0)->published_seq, 2u);
}

TEST(HbgGmMix, PublishedRunErrorRejectsNewClaimAndDescriptorPublication) {
    for (uint8_t mask : {3, 5, 6, 7}) {
        SCOPED_TRACE(mask);
        GmMixFixture fixture({{2, mask}});
        ASSERT_TRUE(fixture.admit(0));
        ASSERT_TRUE(fixture.refresh_all());
        ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
        scheduler_record_error(
            fixture.run(), 0, SchedulerGraphResult::TIMEOUT, nullptr, fixture.locals[4],
            SchedulerErrorSite::ORDERED_EXIT_TIMEOUT
        );
        EXPECT_EQ(fixture.claim(1), SchedulerBlockClaimResult::ERROR);
        EXPECT_FALSE(fixture.publish(0));
        EXPECT_EQ(fixture.control(0)->next_block, 1u);
        EXPECT_EQ(fixture.mailbox(0)->published_seq, 0u);
        EXPECT_EQ(fixture.admissions[0].gm_mix.phase, SchedulerGmMixPhase::CLAIMED);
        EXPECT_EQ(fixture.admissions[3].gm_mix.phase, SchedulerGmMixPhase::IDLE);
    }
}

TEST(HbgGmMix, PublishedRunErrorRejectsAlreadyPreparedKernelEntryOnEveryActiveLane) {
    for (uint8_t mask : {3, 5, 6, 7}) {
        SCOPED_TRACE(mask);
        GmMixFixture fixture({{2, mask}});
        ASSERT_TRUE(fixture.admit(0));
        ASSERT_TRUE(fixture.refresh_all());
        ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
        ASSERT_TRUE(fixture.publish(0));
        ASSERT_TRUE(fixture.rendezvous(0));
        scheduler_record_error(
            fixture.run(), 0, SchedulerGraphResult::TIMEOUT, nullptr, fixture.locals[4],
            SchedulerErrorSite::ORDERED_EXIT_TIMEOUT
        );
        for (uint32_t lane = 0; lane < 3; ++lane) {
            if ((mask & (1U << lane)) == 0) continue;
            auto *local = fixture.locals[lane];
            const auto slot = fixture.admissions[lane].gm_mix.slot_index;
            ASSERT_EQ(local->slots[slot].state, SchedulerDispatchSlotState::READY);
            bool permit = true;
            EXPECT_FALSE(scheduler_ordered_can_enter_slot(
                fixture.graph.graph(), fixture.base(), local, fixture.run(), slot, &permit
            ));
            EXPECT_FALSE(permit);
            EXPECT_EQ(local->slots[slot].state, SchedulerDispatchSlotState::READY);
        }
        EXPECT_EQ(fixture.control(0)->completed_blocks, 0u);
        EXPECT_EQ(fixture.run()->resolved_task_count, 0u);
    }
}

TEST(HbgGmMix, MixPrefixSuffixErrorRejectsNewAdmissionAndPreviouslyPermittedKernelEntry) {
    GmMixFixture fixture({{2, 7}, {1, 1}, {1, 7}, {1, 7}, {1, 7}});
    for (uint32_t task = 2; task <= 4; ++task) {
        fixture.graph.executable(task, 0, {1});
        fixture.storage.metadata[task].flags |= SCHEDULER_TASK_HAS_FANIN;
    }
    ASSERT_TRUE(scheduler_init_spmd_task_records(fixture.base(), fixture.storage.layout, 47));
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(0));
    ASSERT_TRUE(fixture.rendezvous(0));
    for (uint32_t lane = 0; lane < 3; ++lane) {
        const auto slot = fixture.admissions[lane].gm_mix.slot_index;
        bool permit = false;
        ASSERT_TRUE(scheduler_ordered_can_enter_slot(
            fixture.graph.graph(), fixture.base(), fixture.locals[lane], fixture.run(), slot, &permit
        ));
        ASSERT_TRUE(permit);
    }

    auto *resolver = fixture.locals[3];
    for (uint32_t task : {4u, 3u, 2u})
        ASSERT_EQ(
            scheduler_route_task(fixture.graph.graph(), fixture.base(), resolver, fixture.run(), task, nullptr),
            SchedulerRouteResult::WAITING
        );
    fixture.graph.executable(4, 0, {1, 4});
    ASSERT_EQ(fixture.graph.payload(4).fanin_count, 2);
    scheduler_gm_store(fixture.control(1)->state, static_cast<int64_t>(SchedulerTaskState::DONE));
    SchedulerReadyStats ready_stats{};
    ASSERT_FALSE(scheduler_resolve_completion(
        fixture.graph.graph(), fixture.base(), resolver, fixture.run(), 1, nullptr, &ready_stats, nullptr, 0, true,
        nullptr, UINT32_MAX
    ));
    ASSERT_EQ(ready_stats.batch_count, 1u);
    ASSERT_EQ(scheduler_read_error(fixture.run()), static_cast<uint64_t>(SchedulerGraphResult::INVALID_FANIN_ID));
    const uint64_t first_error = scheduler_read_error(fixture.run());
    const uint64_t first_site = fixture.run()->error_site;
    const uint64_t first_task = fixture.run()->error_task_ref;
    ASSERT_EQ(first_task, 4u);

    // A published prefix is not rolled back. A consumer that already owns it
    // must still pass the real admission gate after a later suffix failure.
    SchedulerReadyClaim candidate{};
    candidate.inbox_index = 1;
    ASSERT_TRUE(scheduler_ready_pop_from_inbox(
        fixture.graph.graph(), fixture.base(), resolver, fixture.run(), SCHEDULER_MIX_QUEUE, candidate.inbox_index,
        &candidate.task_ref, nullptr
    ));
    ASSERT_EQ(candidate.task_ref, 2);
    EXPECT_EQ(
        scheduler_try_claim_admission(
            fixture.graph.graph(), fixture.base(), resolver, fixture.run(), &fixture.admissions[3], candidate
        ),
        SchedulerAdmissionResult::ERROR
    );
    EXPECT_EQ(fixture.admissions[3].pending_task_ref, SCHEDULER_TASK_REF_INVALID);
    EXPECT_EQ(
        scheduler_gm_query(fixture.control(2)->admission_state),
        static_cast<uint64_t>(SchedulerAdmissionState::UNCLAIMED)
    );
    EXPECT_EQ(scheduler_gm_query(fixture.control(2)->admission_claim_lock), 0u);
    const auto *shared =
        scheduler_state_at<SchedulerAdmissionControl>(fixture.base(), fixture.storage.layout.admission_control_offset);
    EXPECT_EQ(shared->commit_tail, 1u);
    EXPECT_EQ(shared->commit_lock, 0u);
    for (uint32_t lane = 0; lane < 3; ++lane) {
        const auto slot = fixture.admissions[lane].gm_mix.slot_index;
        bool permit = true;
        EXPECT_FALSE(scheduler_ordered_can_enter_slot(
            fixture.graph.graph(), fixture.base(), fixture.locals[lane], fixture.run(), slot, &permit
        ));
        EXPECT_FALSE(permit);
        EXPECT_EQ(fixture.locals[lane]->slots[slot].state, SchedulerDispatchSlotState::READY);
    }
    EXPECT_EQ(fixture.control(0)->completed_blocks, 0u);
    EXPECT_EQ(fixture.run()->resolved_task_count, 0u);
    EXPECT_EQ(scheduler_read_error(fixture.run()), first_error);
    EXPECT_EQ(fixture.run()->error_site, first_site);
    EXPECT_EQ(fixture.run()->error_task_ref, first_task);
}

TEST(HbgGmMix, PublishedRunErrorDoesNotTurnReleasedLanesIntoSuccessfulBlockCompletion) {
    for (uint8_t mask : {3, 5, 6, 7}) {
        SCOPED_TRACE(mask);
        GmMixFixture fixture({{2, mask}});
        ASSERT_TRUE(fixture.admit(0));
        ASSERT_TRUE(fixture.refresh_all());
        ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
        ASSERT_TRUE(fixture.publish(0));
        ASSERT_TRUE(fixture.rendezvous(0));
        ASSERT_TRUE(fixture.finish_lanes(0));
        scheduler_record_error(
            fixture.run(), 0, SchedulerGraphResult::TIMEOUT, nullptr, fixture.locals[4],
            SchedulerErrorSite::ORDERED_EXIT_TIMEOUT
        );
        EXPECT_EQ(fixture.retire(0), SchedulerGmMixResult::ERROR);
        EXPECT_EQ(fixture.control(0)->completed_blocks, 0u);
        EXPECT_EQ(fixture.run()->resolved_task_count, 0u);
        EXPECT_EQ(fixture.seal(0, 0), 0u);
    }
}

TEST(HbgGmMix, PublishedRunErrorDoesNotSealALocallyCompletedSyncTask) {
    for (uint8_t mask : {3, 5, 6, 7}) {
        SCOPED_TRACE(mask);
        GmMixFixture fixture({{2, mask, true}});
        ASSERT_TRUE(fixture.admit(0));
        ASSERT_TRUE(fixture.refresh_all());
        ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
        ASSERT_TRUE(fixture.publish(0));
        ASSERT_TRUE(fixture.rendezvous(0));
        ASSERT_TRUE(fixture.finish_lanes(0));
        ASSERT_EQ(fixture.retire(0), SchedulerGmMixResult::PROGRESS);
        scheduler_record_error(
            fixture.run(), 0, SchedulerGraphResult::TIMEOUT, nullptr, fixture.locals[4],
            SchedulerErrorSite::ORDERED_EXIT_TIMEOUT
        );
        bool closed = true;
        EXPECT_FALSE(fixture.close(0, &closed));
        EXPECT_FALSE(closed);
        EXPECT_EQ(fixture.seal(0, 0), 0u);
        EXPECT_EQ(fixture.admissions[0].current_task_ref, 0);
        EXPECT_EQ(fixture.admissions[0].scan_cursor, 0u);
        EXPECT_EQ(fixture.control(0)->completed_blocks, 1u);
    }
}

TEST_P(HbgGmMixSyncStart, LocalSealAndNextTaskDoNotWaitForRemoteClusterClaim) {
    const uint8_t mask = GetParam();
    GmMixFixture fixture({{2, mask, true}, {2, mask, true}});
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.admit(1));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    bool closed = true;
    ASSERT_TRUE(fixture.close(0, &closed));
    EXPECT_FALSE(closed);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.seal(0, 0), UINT64_C(0)), 0u);
    ASSERT_TRUE(fixture.publish(0));
    ASSERT_TRUE(fixture.rendezvous(0));
    ASSERT_TRUE(fixture.finish_lanes(0));
    ASSERT_EQ(fixture.retire(0), SchedulerGmMixResult::PROGRESS);
    EXPECT_EQ(fixture.claim(0), SchedulerBlockClaimResult::RETRY);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.control(0)->next_block, UINT64_C(0)), 1u);
    ASSERT_TRUE(fixture.close(0, &closed));
    ASSERT_TRUE(closed);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.seal(0, 0), UINT64_C(0)), 2u);
    // The retained seal remains valid after the next task replaces the mailbox.
    ASSERT_TRUE(fixture.refresh(0));
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(0));
    EXPECT_EQ(fixture.mailbox(0)->descriptor.task_ref, 1);
    EXPECT_EQ(fixture.mailbox(0)->descriptor.group_seq, 2u);
    for (uint32_t worker : {1u, 2u}) {
        ASSERT_TRUE(fixture.close(worker, &closed));
        ASSERT_TRUE(closed);
        ASSERT_TRUE(fixture.refresh(worker));
    }
    ASSERT_TRUE(fixture.rendezvous(0));
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.control(0)->completed_blocks, UINT64_C(0)), 1u);
    ASSERT_EQ(fixture.claim(1), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(1));
    EXPECT_EQ(fixture.mailbox(1)->descriptor.task_ref, 0);
    EXPECT_EQ(fixture.mailbox(1)->descriptor.block_idx, 1u);
    ASSERT_TRUE(fixture.rendezvous(1));
    ASSERT_TRUE(fixture.finish_lanes(1));
    ASSERT_EQ(fixture.retire(1), SchedulerGmMixResult::TASK_LAST);
    ASSERT_TRUE(fixture.complete_task(0, 1));
    ASSERT_TRUE(fixture.finish_lanes(0));
    ASSERT_EQ(fixture.retire(0), SchedulerGmMixResult::PROGRESS);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.run()->scheduler_error, UINT64_C(0)), 0u);
}

TEST_P(HbgGmMixSyncStart, ZeroWorkClusterClosesOnlyAfterExhaustionWhileLastPublisherStillOwesItsGroup) {
    GmMixFixture fixture({{2, GetParam(), true}}, false, 3);
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    bool closed = true;
    ASSERT_TRUE(fixture.close(6, &closed));
    EXPECT_FALSE(closed);
    ASSERT_EQ(fixture.claim(1), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.close(6, &closed));
    ASSERT_TRUE(closed);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.seal(0, 2), UINT64_C(0)), 1u);
    for (uint32_t worker : {7u, 8u}) {
        ASSERT_TRUE(fixture.close(worker, &closed));
        EXPECT_TRUE(closed);
    }
    for (uint32_t worker : {0u, 1u, 2u, 3u, 4u, 5u}) {
        ASSERT_TRUE(fixture.close(worker, &closed));
        EXPECT_FALSE(closed);
    }
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.control(0)->completed_blocks, UINT64_C(0)), 0u);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.run()->scheduler_error, UINT64_C(0)), 0u);
}

INSTANTIATE_TEST_SUITE_P(AllMixMasks, HbgGmMixSyncStart, ::testing::Values<uint8_t>(3, 5, 6, 7));

class HbgGmMixSingleVectorAbi : public ::testing::TestWithParam<uint8_t> {};

TEST_P(HbgGmMixSingleVectorAbi, EvenAndOddBlocksKeepTheLegacySubtaskToPhysicalLaneMapping) {
    const uint8_t mask = GetParam();
    const uint8_t vector_lane = mask == 3 ? 1 : 2;
    GmMixFixture fixture({{2, mask}});
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    for (uint32_t block = 0; block < 2; ++block) {
        SCOPED_TRACE(block);
        ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
        ASSERT_TRUE(fixture.publish(0));
        const auto &descriptor = fixture.mailbox(0)->descriptor;
        EXPECT_EQ(descriptor.block_idx, block);
        EXPECT_EQ(descriptor.lane_subtasks[0], 0u);
        EXPECT_EQ(descriptor.lane_subtasks[vector_lane], vector_lane);
        EXPECT_EQ(descriptor.lane_subtasks[3 - vector_lane], UINT8_MAX);
        ASSERT_TRUE(fixture.rendezvous(0));
        for (uint32_t worker = 0; worker < 3; ++worker) {
            GmExecution execution{};
            if (!fixture.begin(worker, &execution)) continue;
            if (execution.subtask != 0) {
                EXPECT_EQ(worker, vector_lane);
                EXPECT_EQ(execution.subtask, vector_lane);
                EXPECT_EQ(execution.physical_subblock, vector_lane - 1u);
            }
            ASSERT_TRUE(fixture.complete(execution));
        }
        EXPECT_EQ(fixture.retire(0), block == 0 ? SchedulerGmMixResult::PROGRESS : SchedulerGmMixResult::TASK_LAST);
    }
}

TEST_P(HbgGmMixSingleVectorAbi, VectorObserverRejectsAValidMaskRelocatedToTheOtherPhysicalLane) {
    const uint8_t mask = GetParam();
    const uint8_t vector_lane = mask == 3 ? 1 : 2;
    GmMixFixture fixture({{1, mask}});
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(0));
    auto &descriptor = fixture.mailbox(0)->descriptor;
    descriptor.lane_subtasks[vector_lane] = UINT8_MAX;
    descriptor.lane_subtasks[3 - vector_lane] = vector_lane;
    bool progress = false;
    EXPECT_FALSE(fixture.observe(1, &progress));
    EXPECT_FALSE(progress);
    EXPECT_NE(scheduler_gm_fetch_add(fixture.run()->scheduler_error, UINT64_C(0)), 0u);
    EXPECT_EQ(fixture.locals[1]->local_ready_mask, 0u);
}

INSTANTIATE_TEST_SUITE_P(SingleVectorMasks, HbgGmMixSingleVectorAbi, ::testing::Values<uint8_t>(3, 5));

class HbgGmMixShapes : public ::testing::TestWithParam<std::tuple<uint8_t, uint16_t>> {};

TEST_P(HbgGmMixShapes, RealOwnDispatchAndCompletionKeepEveryLogicalBlockAndSubtaskExactlyOnce) {
    const auto [mask, blocks] = GetParam();
    GmMixFixture fixture({{blocks, mask}});
    ASSERT_EQ(fixture.storage.ssbuf_region, nullptr);
    ASSERT_TRUE(fixture.enqueue(0));
    std::map<std::pair<uint32_t, uint8_t>, uint32_t> visits;
    std::set<uint32_t> single_vector_lanes;
    const uint64_t expected = blocks * static_cast<uint32_t>(__builtin_popcount(mask));
    uint64_t executions = 0;
    for (uint32_t step = 0; step < blocks * 100 + 100; ++step) {
        for (uint32_t worker = 0; worker < 6; ++worker) {
            ASSERT_TRUE(fixture.fill(worker));
            GmExecution execution{};
            if (!fixture.begin(worker, &execution)) continue;
            ASSERT_EQ(execution.task, 0);
            ASSERT_EQ(execution.blocks, blocks);
            ASSERT_LT(execution.block, blocks);
            ASSERT_LT(execution.subtask, 3);
            ASSERT_NE(mask & (1U << execution.subtask), 0u);
            EXPECT_TRUE(fixture.locals[worker]->slots[execution.slot].gm_mix);
            EXPECT_EQ(fixture.mailbox(worker / 3)->descriptor.lane_subtasks[worker % 3], execution.subtask);
            if (execution.subtask != 0) {
                EXPECT_EQ(worker % 3, execution.subtask);
                EXPECT_EQ(execution.physical_subblock, worker % 3 - 1);
                if (mask == 3 || mask == 5) single_vector_lanes.insert(worker % 3);
            }
            ++visits[{execution.block, execution.subtask}];
            ++executions;
            ASSERT_TRUE(fixture.complete(execution));
        }
        if (scheduler_gm_load(fixture.control(0)->state) == static_cast<int64_t>(SchedulerTaskState::DONE)) break;
    }
    EXPECT_EQ(executions, expected);
    for (uint32_t block = 0; block < blocks; ++block)
        for (uint8_t subtask = 0; subtask < 3; ++subtask)
            EXPECT_EQ(visits[std::make_pair(block, subtask)], (mask & (1U << subtask)) != 0 ? 1u : 0u);
    if (mask == 3 || mask == 5) EXPECT_EQ(single_vector_lanes, (std::set<uint32_t>{mask == 3 ? 1U : 2U}));
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.control(0)->next_block, UINT64_C(0)), blocks);
    if (blocks > 1) {
        EXPECT_GE(fixture.control(0)->claim_tickets, blocks);
        EXPECT_LE(fixture.control(0)->claim_tickets, blocks + 2);
    } else {
        EXPECT_EQ(fixture.control(0)->claim_tickets, 0u);
    }
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.control(0)->completed_blocks, UINT64_C(0)), blocks);
    EXPECT_EQ(scheduler_gm_load(fixture.control(0)->state), static_cast<int64_t>(SchedulerTaskState::DONE));
    std::array<bool, 6> done{};
    for (uint32_t step = 0; step < 16; ++step)
        for (uint32_t worker = 0; worker < 6; ++worker)
            ASSERT_TRUE(fixture.drain(worker, &done[worker]));
    EXPECT_TRUE(std::all_of(done.begin(), done.end(), [](bool value) {
        return value;
    }));
    for (uint32_t worker = 0; worker < 6; ++worker)
        scheduler_flush_completions(fixture.run(), fixture.locals[worker]);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.run()->resolved_task_count, UINT64_C(0)), 1u);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.run()->scheduler_error, UINT64_C(0)), 0u);
}

INSTANTIATE_TEST_SUITE_P(
    MasksAndClusterCapacity, HbgGmMixShapes,
    ::testing::Combine(::testing::Values<uint8_t>(3, 5, 6, 7), ::testing::Values<uint16_t>(1, 2, 3, 4, 5))
);

TEST(HbgGmMix, ClaimedLastBlockWithoutDescriptorCannotLetItsVectorMembersClose) {
    GmMixFixture fixture({{1, 7}});
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    EXPECT_EQ(fixture.admissions[0].gm_mix.phase, SchedulerGmMixPhase::CLAIMED);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.control(0)->next_block, UINT64_C(0)), 1u);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.mailbox(0)->published_seq, UINT64_C(0)), 0u);
    for (uint32_t worker : {1u, 2u}) {
        bool progress = true;
        ASSERT_TRUE(fixture.observe(worker, &progress));
        EXPECT_FALSE(progress);
        bool closed = true;
        ASSERT_TRUE(fixture.close(worker, &closed));
        EXPECT_FALSE(closed);
        EXPECT_EQ(fixture.admissions[worker].current_task_ref, 0);
        EXPECT_EQ(fixture.locals[worker]->local_ready_mask, 0u);
    }
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.seal(0, 0), UINT64_C(0)), 0u);
    ASSERT_TRUE(fixture.publish(0));
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.mailbox(0)->published_seq, UINT64_C(0)), 1u);
}

class HbgGmMixInactiveLane : public ::testing::TestWithParam<uint8_t> {};

TEST_P(HbgGmMixInactiveLane, InactiveResourceLaneMustArriveButNeverRunsAKernel) {
    GmMixFixture fixture({{1, GetParam()}});
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(0));
    const auto &descriptor = fixture.mailbox(0)->descriptor;
    ASSERT_EQ(descriptor.resource_mask, 7u);
    uint32_t inactive = UINT32_MAX;
    for (uint32_t lane = 0; lane < 3; ++lane)
        if (descriptor.lane_subtasks[lane] == UINT8_MAX) inactive = lane;
    ASSERT_LT(inactive, 3u);
    bool progress = false;
    for (uint32_t lane = 1; lane < 3; ++lane)
        ASSERT_TRUE(fixture.observe(lane, &progress));
    for (uint32_t lane = 0; lane < 3; ++lane)
        if (lane != inactive) ASSERT_TRUE(fixture.arrive(lane, &progress));
    for (uint32_t lane = 0; lane < 3; ++lane) {
        if (lane == inactive) continue;
        ASSERT_TRUE(fixture.prepare(lane, &progress));
        EXPECT_EQ(fixture.locals[lane]->local_ready_mask, 0u);
    }
    ASSERT_TRUE(fixture.arrive(inactive, &progress));
    for (uint32_t lane = 0; lane < 3; ++lane)
        ASSERT_TRUE(fixture.prepare(lane, &progress));
    EXPECT_EQ(fixture.admissions[inactive].gm_mix.phase, SchedulerGmMixPhase::RELEASED);
    EXPECT_EQ(fixture.locals[inactive]->local_ready_mask, 0u);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.mailbox(0)->lanes[inactive].arrived_seq, UINT64_C(0)), 1u);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.mailbox(0)->lanes[inactive].released_seq, UINT64_C(0)), 1u);
    for (uint32_t lane = 0; lane < 3; ++lane) {
        GmExecution execution{};
        if (lane == inactive) {
            EXPECT_FALSE(fixture.begin(lane, &execution));
            continue;
        }
        ASSERT_TRUE(fixture.begin(lane, &execution));
        EXPECT_EQ(execution.subtask, descriptor.lane_subtasks[lane]);
        ASSERT_TRUE(fixture.complete(execution));
    }
    ASSERT_EQ(fixture.retire(0), SchedulerGmMixResult::TASK_LAST);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.control(0)->completed_blocks, UINT64_C(0)), 1u);
}

INSTANTIATE_TEST_SUITE_P(MasksWithInactiveResourceMember, HbgGmMixInactiveLane, ::testing::Values<uint8_t>(3, 5, 6));

class HbgGmMixOrdinaryResource : public ::testing::TestWithParam<std::tuple<uint8_t, uint32_t, uint32_t>> {};

TEST_P(HbgGmMixOrdinaryResource, ArrivesOnlyAfterRealOrdinarySlotCompletionIsConsumed) {
    const auto [mask, lane, pending_slot] = GetParam();
    const uint8_t ordinary_mask = lane == 0 ? 1 : 2;
    GmMixFixture fixture({{1, ordinary_mask}, {1, ordinary_mask}, {1, mask}});
    auto *local = fixture.locals[lane];
    const uint32_t core_type = lane == 0 ? 0 : 1;
    const uint32_t first_task = lane == 2 ? 1 : 0;
    SchedulerReadyBatch batch{};
    for (uint32_t task : {first_task, 1U - first_task}) {
        ASSERT_EQ(
            scheduler_route_task(fixture.graph.graph(), fixture.base(), local, fixture.run(), task, nullptr),
            SchedulerRouteResult::READY_TO_ENQUEUE
        );
        ASSERT_TRUE(scheduler_ready_batch_append(fixture.base(), local, task, &batch, nullptr));
    }
    ASSERT_TRUE(scheduler_ready_batch_push(fixture.base(), local, core_type, &batch, nullptr));
    ASSERT_TRUE(fixture.fill(lane));
    ASSERT_EQ(local->local_ready_mask, 3u);
    GmExecution ordinary{};
    ASSERT_TRUE(fixture.begin(lane, &ordinary));
    ASSERT_EQ(ordinary.slot, 0u);
    if (pending_slot == 1) {
        ASSERT_TRUE(fixture.complete(ordinary));
        ASSERT_TRUE(fixture.begin(lane, &ordinary));
    }
    ASSERT_EQ(ordinary.slot, pending_slot);
    ASSERT_FALSE(local->slots[pending_slot].cooperative);
    ASSERT_EQ(local->slots[pending_slot].state, SchedulerDispatchSlotState::READY);
    const uint32_t consumed = scheduler_consumed_completion_generation(local, lane, pending_slot);
    ASSERT_LT(consumed, ordinary.generation);

    ASSERT_TRUE(fixture.admit(2, 3));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_TRUE(scheduler_ordered_park_ordinary(fixture.graph.graph(), local, fixture.run()));
    ASSERT_EQ(local->local_ready_mask, 0u);
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(0));
    bool progress = false;
    for (uint32_t member : {1U, 2U})
        ASSERT_TRUE(fixture.observe(member, &progress));
    for (uint32_t member : {0U, 1U, 2U}) {
        if (member == lane) continue;
        ASSERT_TRUE(fixture.arrive(member, &progress));
        EXPECT_TRUE(progress);
    }

    for (bool returned : {false, true}) {
        SCOPED_TRACE(returned);
        if (returned) local->local_completed_generations[pending_slot] = ordinary.generation;
        EXPECT_EQ(local->local_completed_generations[pending_slot] == ordinary.generation, returned);
        EXPECT_EQ(scheduler_consumed_completion_generation(local, lane, pending_slot), consumed);
        progress = true;
        ASSERT_TRUE(fixture.arrive(lane, &progress));
        EXPECT_FALSE(progress);
        EXPECT_EQ(fixture.admissions[lane].gm_mix.phase, SchedulerGmMixPhase::PUBLISHED);
        EXPECT_EQ(scheduler_gm_fetch_add(fixture.mailbox(0)->lanes[lane].arrived_seq, UINT64_C(0)), 0u);
        EXPECT_EQ(local->slots[pending_slot].task_ref, ordinary.task);
        EXPECT_EQ(local->slots[pending_slot].generation, ordinary.generation);
        EXPECT_EQ(local->slots[pending_slot].state, SchedulerDispatchSlotState::READY);
        for (uint32_t member : {0U, 1U, 2U}) {
            progress = true;
            ASSERT_TRUE(fixture.prepare(member, &progress));
            EXPECT_FALSE(progress);
            EXPECT_EQ(fixture.locals[member]->local_ready_mask, 0u);
        }
        EXPECT_EQ(scheduler_gm_fetch_add(fixture.control(2)->completed_blocks, UINT64_C(0)), 0u);
    }

    ASSERT_TRUE(fixture.complete(ordinary));
    ASSERT_EQ(local->slots[pending_slot].state, SchedulerDispatchSlotState::FREE);
    EXPECT_EQ(scheduler_consumed_completion_generation(local, lane, pending_slot), ordinary.generation);
    ASSERT_TRUE(fixture.arrive(lane, &progress));
    EXPECT_TRUE(progress);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.mailbox(0)->lanes[lane].arrived_seq, UINT64_C(0)), 1u);
    for (uint32_t member : {0U, 1U, 2U}) {
        ASSERT_TRUE(fixture.prepare(member, &progress));
        EXPECT_TRUE(progress);
        const bool active = (mask & (1U << member)) != 0;
        EXPECT_EQ(
            fixture.admissions[member].gm_mix.phase,
            active ? SchedulerGmMixPhase::RUNNING : SchedulerGmMixPhase::RELEASED
        );
        EXPECT_EQ(fixture.locals[member]->local_ready_mask != 0, active);
    }
    ASSERT_TRUE(fixture.finish_lanes(0));
    ASSERT_EQ(fixture.retire(0), SchedulerGmMixResult::TASK_LAST);
    ASSERT_TRUE(fixture.complete_task(2, 0));
    for (uint32_t worker = 0; worker < fixture.locals.size(); ++worker) {
        bool closed = false;
        ASSERT_TRUE(fixture.close(worker, &closed));
        EXPECT_TRUE(closed);
    }
    ASSERT_TRUE(fixture.fill(lane));
    GmExecution resumed{};
    if (pending_slot == 0) {
        ASSERT_TRUE(fixture.begin(lane, &resumed));
        EXPECT_EQ(resumed.task, 1 - ordinary.task);
        EXPECT_FALSE(local->slots[resumed.slot].cooperative);
        ASSERT_TRUE(fixture.complete(resumed));
    }
    EXPECT_FALSE(fixture.begin(lane, &resumed));
    for (uint32_t task : {0U, 1U, 2U})
        EXPECT_EQ(scheduler_gm_load(fixture.control(task)->state), static_cast<int64_t>(SchedulerTaskState::DONE));
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.control(2)->completed_blocks, UINT64_C(0)), 1u);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.run()->scheduler_error, UINT64_C(0)), 0u);
}

INSTANTIATE_TEST_SUITE_P(
    EveryMaskLaneAndOwnSlot, HbgGmMixOrdinaryResource,
    ::testing::Combine(
        ::testing::Values<uint8_t>(3, 5, 6, 7), ::testing::Values<uint32_t>(0, 1, 2), ::testing::Values<uint32_t>(0, 1)
    )
);

TEST(HbgGmMix, KernelReturnWithoutCompletionReleaseDoesNotRetireTheGroup) {
    GmMixFixture fixture({{1, 7}});
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(0));
    ASSERT_TRUE(fixture.rendezvous(0));
    std::array<GmExecution, 3> executions{};
    for (uint32_t worker = 0; worker < 3; ++worker)
        ASSERT_TRUE(fixture.begin(worker, &executions[worker]));
    ASSERT_TRUE(fixture.complete(executions[0]));
    ASSERT_TRUE(fixture.complete(executions[1]));
    const auto &late = executions[2];
    fixture.locals[2]->local_completed_generations[late.slot] = late.generation;
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.mailbox(0)->lanes[2].released_seq, UINT64_C(0)), 0u);
    EXPECT_EQ(fixture.retire(0), SchedulerGmMixResult::WAIT);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.control(0)->completed_blocks, UINT64_C(0)), 0u);
    ASSERT_TRUE(fixture.complete(late));
    ASSERT_EQ(fixture.retire(0), SchedulerGmMixResult::TASK_LAST);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.control(0)->completed_blocks, UINT64_C(0)), 1u);
    EXPECT_EQ(scheduler_gm_load(fixture.control(0)->state), static_cast<int64_t>(SchedulerTaskState::BLOCKED));
    ASSERT_TRUE(fixture.complete_task(0, 0));
    EXPECT_EQ(scheduler_gm_load(fixture.control(0)->state), static_cast<int64_t>(SchedulerTaskState::DONE));
    EXPECT_EQ(fixture.locals[0]->pending_completed, 1u);
}

TEST(HbgGmMixProtocol, ZeroWorkMembersRecordTheActualSealWithoutInventingCursorReads) {
    GmMixFixture fixture({{1, 6}}, true);
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    auto *members = scheduler_state_at<SchedulerProtocolMemberTrace>(
        fixture.base(), fixture.storage.layout.protocol.members_offset
    );
    bool closed = false;
    ASSERT_TRUE(fixture.close(4, &closed));
    EXPECT_FALSE(closed);
    EXPECT_EQ(members[4].closed.sequence, 0u);
    for (uint32_t worker = 3; worker < 6; ++worker) {
        ASSERT_TRUE(fixture.close(worker, &closed));
        EXPECT_TRUE(closed);
        EXPECT_EQ(members[worker].observed.sequence, 1u);
        EXPECT_EQ(members[worker].closed.sequence, 2u);
        EXPECT_EQ(members[worker].close_seal, 1u);
        EXPECT_EQ(members[worker].last_group_seq, 0u);
        EXPECT_EQ(members[worker].closed_cursor, worker == 3 ? UINT64_C(1) : UINT64_MAX);
        EXPECT_EQ(members[worker].pending_mask, 0u);
        EXPECT_EQ(members[worker].worker_index, worker);
    }
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.control(0)->completed_blocks, UINT64_C(0)), 0u);
    EXPECT_EQ(members[0].closed.sequence, 0u);
    EXPECT_EQ(members[1].closed.sequence, 0u);
}

TEST(HbgGmMix, AZeroWorkClusterPublishesSealOneWhileAnotherClusterStillOwnsTheLastBlock) {
    GmMixFixture fixture({{1, 7}});
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_EQ(fixture.claim(1), SchedulerBlockClaimResult::EXHAUSTED);
    bool closed = false;
    ASSERT_TRUE(fixture.close(3, &closed));
    EXPECT_TRUE(closed);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.seal(0, 1), UINT64_C(0)), 1u);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.seal(0, 0), UINT64_C(0)), 0u);
    for (uint32_t worker : {4u, 5u}) {
        ASSERT_TRUE(fixture.close(worker, &closed));
        EXPECT_TRUE(closed);
        EXPECT_EQ(fixture.admissions[worker].current_task_ref, SCHEDULER_TASK_REF_INVALID);
        EXPECT_EQ(fixture.locals[worker]->local_ready_mask, 0u);
    }
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.control(0)->completed_blocks, UINT64_C(0)), 0u);
}

TEST(HbgGmMix, FastClusterPublishesBWhileRemoteAIsRunningAndLateVectorConsumesAsRetainedSeal) {
    GmMixFixture fixture({{2, 7}, {1, 3}});
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.admit(1, 3));
    ASSERT_TRUE(fixture.refresh_all());
    for (uint32_t cluster = 0; cluster < 2; ++cluster) {
        ASSERT_EQ(fixture.claim(cluster), SchedulerBlockClaimResult::CLAIMED);
        ASSERT_TRUE(fixture.publish(cluster));
        ASSERT_TRUE(fixture.rendezvous(cluster));
    }
    GmExecution remote{};
    ASSERT_TRUE(fixture.begin(5, &remote));
    ASSERT_TRUE(fixture.finish_lanes(0));
    ASSERT_EQ(fixture.retire(0), SchedulerGmMixResult::PROGRESS);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.control(0)->completed_blocks, UINT64_C(0)), 1u);
    bool closed = false;
    ASSERT_TRUE(fixture.close(0, &closed));
    ASSERT_TRUE(closed);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.seal(0, 0), UINT64_C(0)), 2u);
    ASSERT_TRUE(fixture.refresh(0));
    ASSERT_EQ(fixture.admissions[0].current_task_ref, 1);
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(0));
    EXPECT_EQ(fixture.mailbox(0)->descriptor.task_ref, 1);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.mailbox(0)->published_seq, UINT64_C(0)), 2u);
    EXPECT_EQ(scheduler_gm_load(fixture.control(0)->state), static_cast<int64_t>(SchedulerTaskState::BLOCKED));
    EXPECT_EQ(fixture.admissions[5].current_task_ref, 0);
    EXPECT_EQ(fixture.admissions[5].gm_mix.phase, SchedulerGmMixPhase::RUNNING);
    for (uint32_t worker : {1u, 2u}) {
        bool progress = false;
        ASSERT_EQ(fixture.admissions[worker].current_task_ref, 0);
        ASSERT_TRUE(fixture.observe(worker, &progress));
        EXPECT_TRUE(progress);
        EXPECT_EQ(fixture.admissions[worker].current_task_ref, SCHEDULER_TASK_REF_INVALID);
        EXPECT_EQ(fixture.admissions[worker].gm_mix.group_seq, 1u);
        EXPECT_EQ(fixture.admissions[worker].gm_mix.task_last_seq, 0u);
        ASSERT_TRUE(fixture.refresh(worker));
        ASSERT_EQ(fixture.admissions[worker].current_task_ref, 1);
        ASSERT_TRUE(fixture.observe(worker, &progress));
        EXPECT_TRUE(progress);
        EXPECT_EQ(fixture.admissions[worker].gm_mix.group_seq, 2u);
    }
    ASSERT_TRUE(fixture.rendezvous(0));
    ASSERT_TRUE(fixture.finish_lanes(0));
    ASSERT_EQ(fixture.retire(0), SchedulerGmMixResult::TASK_LAST);
    ASSERT_TRUE(fixture.complete_task(1, 0));
    EXPECT_EQ(scheduler_gm_load(fixture.control(1)->state), static_cast<int64_t>(SchedulerTaskState::DONE));
    EXPECT_EQ(scheduler_gm_load(fixture.control(0)->state), static_cast<int64_t>(SchedulerTaskState::BLOCKED));
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.seal(0, 0), UINT64_C(0)), 2u);
}

TEST(HbgGmMix, ReleasedAcknowledgementsFromThePreviousGroupCannotRetireTheNextGroup) {
    GmMixFixture fixture({{2, 7}});
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(0));
    ASSERT_TRUE(fixture.rendezvous(0));
    ASSERT_TRUE(fixture.finish_lanes(0));
    ASSERT_EQ(fixture.retire(0), SchedulerGmMixResult::PROGRESS);
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(0));
    ASSERT_TRUE(fixture.rendezvous(0));
    GmExecution cube{};
    ASSERT_TRUE(fixture.begin(0, &cube));
    ASSERT_TRUE(fixture.complete(cube));
    for (uint32_t lane : {1u, 2u})
        EXPECT_EQ(scheduler_gm_fetch_add(fixture.mailbox(0)->lanes[lane].released_seq, UINT64_C(0)), 1u);
    EXPECT_EQ(fixture.retire(0), SchedulerGmMixResult::WAIT);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.control(0)->completed_blocks, UINT64_C(0)), 1u);
    for (uint32_t lane : {1u, 2u}) {
        GmExecution vector{};
        ASSERT_TRUE(fixture.begin(lane, &vector));
        EXPECT_EQ(vector.block, 1u);
        ASSERT_TRUE(fixture.complete(vector));
    }
    ASSERT_EQ(fixture.retire(0), SchedulerGmMixResult::TASK_LAST);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.control(0)->completed_blocks, UINT64_C(0)), 2u);
}

TEST(HbgGmMix, DuplicateGenerationCannotReleaseOrContributeTwice) {
    GmMixFixture fixture({{1, 7}});
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(0));
    ASSERT_TRUE(fixture.rendezvous(0));
    GmExecution first{};
    ASSERT_TRUE(fixture.begin(1, &first));
    ASSERT_TRUE(fixture.complete(first));
    EXPECT_FALSE(scheduler_service_cluster_completion_slot(
        fixture.graph.graph(), fixture.base(), fixture.locals[1], fixture.run(), 1, first.slot, first.generation,
        nullptr, nullptr, nullptr, 0, nullptr, nullptr, nullptr
    ));
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.mailbox(0)->lanes[1].released_seq, UINT64_C(0)), 1u);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.control(0)->completed_blocks, UINT64_C(0)), 0u);
    for (uint32_t worker : {0u, 2u}) {
        GmExecution execution{};
        ASSERT_TRUE(fixture.begin(worker, &execution));
        ASSERT_TRUE(fixture.complete(execution));
    }
    ASSERT_EQ(fixture.retire(0), SchedulerGmMixResult::TASK_LAST);
    EXPECT_EQ(fixture.retire(0), SchedulerGmMixResult::WAIT);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.control(0)->completed_blocks, UINT64_C(0)), 1u);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.run()->scheduler_error, UINT64_C(0)), 0u);
}

TEST(HbgGmMix, ReleaseBeforeRealSlotRetirementFailsWithoutPublishingAnAcknowledgement) {
    GmMixFixture fixture({{1, 7}});
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(0));
    ASSERT_TRUE(fixture.rendezvous(0));
    GmExecution execution{};
    ASSERT_TRUE(fixture.begin(1, &execution));
    EXPECT_FALSE(scheduler_gm_mix_release_lane(
        fixture.graph.graph(), fixture.base(), fixture.locals[1], fixture.run(), execution.task, execution.slot,
        execution.generation
    ));
    EXPECT_EQ(fixture.locals[1]->slots[execution.slot].state, SchedulerDispatchSlotState::READY);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.mailbox(0)->lanes[1].released_seq, UINT64_C(0)), 0u);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.control(0)->completed_blocks, UINT64_C(0)), 0u);
}

TEST(HbgGmMix, CoordinatorPreparesOnlyItsOwnPayloadAndVectorsMaterializeTheirOwnSlots) {
    GmMixFixture fixture({{1, 7}});
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(0));
    bool progress = false;
    for (uint32_t worker : {1u, 2u})
        ASSERT_TRUE(fixture.observe(worker, &progress));
    for (uint32_t worker = 0; worker < 3; ++worker)
        ASSERT_TRUE(fixture.arrive(worker, &progress));
    ASSERT_TRUE(fixture.prepare(0, &progress));
    EXPECT_TRUE(progress);
    for (uint32_t worker : {1u, 2u}) {
        const auto *payload = scheduler_state_at<DispatchPayload>(
            fixture.base(), fixture.locals[worker]->dispatch_payload_offset(worker, 0)
        );
        EXPECT_EQ(payload->function_bin_addr, 0u);
        EXPECT_EQ(fixture.locals[worker]->local_ready_mask, 0u);
        ASSERT_TRUE(fixture.prepare(worker, &progress));
        EXPECT_TRUE(progress);
        EXPECT_NE(payload->function_bin_addr, 0u);
        EXPECT_EQ(payload->local_context.block_num, 1);
        EXPECT_EQ(payload->global_context.sub_block_id, worker - 1);
    }
}

class HbgGmMixHighSequence : public ::testing::TestWithParam<uint64_t> {};

TEST_P(HbgGmMixHighSequence, GroupSequenceIsIndependentOfTheNarrowOwnSlotGeneration) {
    const uint64_t previous = GetParam();
    GmMixFixture fixture({{1, 7}});
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    scheduler_gm_store(fixture.mailbox(0)->published_seq, previous);
    for (uint32_t worker = 0; worker < 3; ++worker) {
        fixture.admissions[worker].gm_mix.group_seq = previous;
        scheduler_gm_store(fixture.mailbox(0)->lanes[worker].arrived_seq, previous);
        scheduler_gm_store(fixture.mailbox(0)->lanes[worker].released_seq, previous);
    }
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(0));
    ASSERT_TRUE(fixture.rendezvous(0));
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.mailbox(0)->published_seq, UINT64_C(0)), previous + 1);
    for (uint32_t worker = 0; worker < 3; ++worker) {
        EXPECT_EQ(fixture.admissions[worker].gm_mix.group_seq, previous + 1);
        EXPECT_EQ(fixture.admissions[worker].gm_mix.slot_generation, 1u);
    }
    ASSERT_TRUE(fixture.finish_lanes(0));
    ASSERT_EQ(fixture.retire(0), SchedulerGmMixResult::TASK_LAST);
    ASSERT_TRUE(fixture.complete_task(0, 0));
    bool closed = false;
    for (uint32_t worker = 0; worker < 3; ++worker) {
        ASSERT_TRUE(fixture.close(worker, &closed));
        EXPECT_TRUE(closed);
        EXPECT_EQ(fixture.admissions[worker].gm_mix.group_seq, previous + 1);
        EXPECT_EQ(fixture.admissions[worker].gm_mix.task_last_seq, 0u);
    }
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.seal(0, 0), UINT64_C(0)), previous + 2);
}

INSTANTIATE_TEST_SUITE_P(
    WideCounterBoundaries, HbgGmMixHighSequence, ::testing::Values(UINT64_C(1) << 32, UINT64_MAX - 2)
);

TEST(HbgGmMix, SequenceExhaustionFailsBeforeClaimingAnUnpublishableBlock) {
    for (uint64_t sequence : {UINT64_MAX - 1, UINT64_MAX}) {
        GmMixFixture fixture({{1, 7}});
        ASSERT_TRUE(fixture.admit(0));
        ASSERT_TRUE(fixture.refresh_all());
        fixture.admissions[0].gm_mix.group_seq = sequence;
        EXPECT_EQ(fixture.claim(0), SchedulerBlockClaimResult::ERROR);
        EXPECT_EQ(scheduler_gm_fetch_add(fixture.control(0)->next_block, UINT64_C(0)), 0u);
        EXPECT_EQ(scheduler_gm_fetch_add(fixture.mailbox(0)->published_seq, UINT64_C(0)), 0u);
        EXPECT_EQ(fixture.admissions[0].gm_mix.group_seq, sequence);
        EXPECT_EQ(fixture.admissions[0].gm_mix.phase, SchedulerGmMixPhase::IDLE);
    }
}

enum class DescriptorFault {
    EPOCH,
    TASK,
    ORDER,
    SEQUENCE,
    BLOCK,
    BLOCK_COUNT,
    CLUSTER,
    ACTIVE_MASK,
    RESOURCE_MASK,
    FLAGS,
    WRONG_CORE_TYPE,
    DUPLICATED_SUBTASK,
    MISSING_SUBTASK,
    WRONG_VECTOR_PAIR,
    SKIPPED_SEQUENCE,
};

class HbgGmMixDescriptorFault : public ::testing::TestWithParam<DescriptorFault> {};

TEST_P(HbgGmMixDescriptorFault, CorruptPublishedDescriptorCannotAcquireAVectorLane) {
    GmMixFixture fixture({{1, 7}});
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(0));
    auto &descriptor = fixture.mailbox(0)->descriptor;
    switch (GetParam()) {
    case DescriptorFault::EPOCH:
        ++descriptor.run_epoch;
        break;
    case DescriptorFault::TASK:
        descriptor.task_ref = SCHEDULER_TASK_REF_INVALID;
        break;
    case DescriptorFault::ORDER:
        ++descriptor.order_key;
        break;
    case DescriptorFault::SEQUENCE:
        ++descriptor.group_seq;
        break;
    case DescriptorFault::BLOCK:
        descriptor.block_idx = descriptor.logical_block_num;
        break;
    case DescriptorFault::BLOCK_COUNT:
        ++descriptor.logical_block_num;
        break;
    case DescriptorFault::CLUSTER:
        ++descriptor.cluster_index;
        break;
    case DescriptorFault::ACTIVE_MASK:
        descriptor.active_mask = 3;
        break;
    case DescriptorFault::RESOURCE_MASK:
        descriptor.resource_mask = 3;
        break;
    case DescriptorFault::FLAGS:
        descriptor.flags ^= SCHEDULER_TASK_SPMD;
        break;
    case DescriptorFault::WRONG_CORE_TYPE:
        std::swap(descriptor.lane_subtasks[0], descriptor.lane_subtasks[1]);
        break;
    case DescriptorFault::DUPLICATED_SUBTASK:
        descriptor.lane_subtasks[2] = 1;
        break;
    case DescriptorFault::MISSING_SUBTASK:
        descriptor.lane_subtasks[2] = UINT8_MAX;
        break;
    case DescriptorFault::WRONG_VECTOR_PAIR:
        std::swap(descriptor.lane_subtasks[1], descriptor.lane_subtasks[2]);
        break;
    case DescriptorFault::SKIPPED_SEQUENCE:
        descriptor.group_seq = 3;
        scheduler_gm_store(fixture.mailbox(0)->published_seq, UINT64_C(3));
        break;
    }
    bool progress = true;
    EXPECT_FALSE(fixture.observe(1, &progress));
    EXPECT_FALSE(progress);
    EXPECT_EQ(fixture.admissions[1].gm_mix.group_seq, 0u);
    EXPECT_EQ(fixture.admissions[1].gm_mix.phase, SchedulerGmMixPhase::IDLE);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.mailbox(0)->lanes[1].arrived_seq, UINT64_C(0)), 0u);
    EXPECT_EQ(fixture.locals[1]->local_ready_mask, 0u);
    EXPECT_NE(scheduler_gm_fetch_add(fixture.run()->scheduler_error, UINT64_C(0)), 0u);
}

INSTANTIATE_TEST_SUITE_P(
    IdentityAndPlacement, HbgGmMixDescriptorFault,
    ::testing::Values(
        DescriptorFault::EPOCH, DescriptorFault::TASK, DescriptorFault::ORDER, DescriptorFault::SEQUENCE,
        DescriptorFault::BLOCK, DescriptorFault::BLOCK_COUNT, DescriptorFault::CLUSTER, DescriptorFault::ACTIVE_MASK,
        DescriptorFault::RESOURCE_MASK, DescriptorFault::FLAGS, DescriptorFault::WRONG_CORE_TYPE,
        DescriptorFault::DUPLICATED_SUBTASK, DescriptorFault::MISSING_SUBTASK, DescriptorFault::WRONG_VECTOR_PAIR,
        DescriptorFault::SKIPPED_SEQUENCE
    )
);

struct AckPrefixCase {
    bool releases;
    uint64_t sequence;
    std::array<int8_t, 3> offsets;
};

std::vector<AckPrefixCase> ack_prefix_cases() {
    std::vector<AckPrefixCase> result;
    for (bool releases : {false, true})
        for (uint64_t sequence : {UINT64_C(1), (UINT64_C(1) << 32) + 9, UINT64_MAX - 1})
            for (int8_t first : {-1, 0, 1})
                for (int8_t second : {-1, 0, 1})
                    for (int8_t third : {-1, 0, 1})
                        result.push_back({releases, sequence, {first, second, third}});
    return result;
}

class HbgGmAckPrefix : public ::testing::TestWithParam<AckPrefixCase> {};

TEST_P(HbgGmAckPrefix, WaitsOnOldPrefixAndRejectsFuturePrefixBeforePublishingEvidence) {
    const auto scenario = GetParam();
    GmMixFixture fixture({{1, 7}});
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    fixture.admissions[0].gm_mix.group_seq = scenario.sequence;
    int8_t first_difference = 0;
    for (uint32_t lane = 0; lane < 3; ++lane) {
        const int8_t delta = scenario.offsets[lane];
        if (first_difference == 0 && delta != 0) first_difference = delta;
        const uint64_t value = delta < 0 ? scenario.sequence - 1 : scenario.sequence + delta;
        auto &ack = scenario.releases ? fixture.mailbox(0)->lanes[lane].released_seq :
                                        fixture.mailbox(0)->lanes[lane].arrived_seq;
        scheduler_gm_store(ack, value);
    }
    bool all = true;
    const bool valid = scheduler_gm_mix_check_acks(
        fixture.graph.graph(), fixture.base(), fixture.locals[0], fixture.run(), scenario.releases, &all
    );
    EXPECT_EQ(valid, first_difference <= 0);
    EXPECT_EQ(all, first_difference == 0);
    EXPECT_EQ(scheduler_read_error(fixture.run()) != 0, first_difference > 0);
    if (first_difference > 0)
        EXPECT_EQ(fixture.run()->error_site, static_cast<uint64_t>(SchedulerErrorSite::GROUP_ACK_INVALID));
    EXPECT_EQ(fixture.locals[0]->local_ready_mask, 0u);
    EXPECT_EQ(fixture.control(0)->completed_blocks, 0u);
    EXPECT_EQ(fixture.admissions[0].scan_cursor, 0u);
}

INSTANTIATE_TEST_SUITE_P(OldCurrentFutureAndFullWidth, HbgGmAckPrefix, ::testing::ValuesIn(ack_prefix_cases()));

class HbgGmLateAckReadyCandidate : public ::testing::TestWithParam<std::tuple<uint8_t, uint32_t>> {};

TEST_P(HbgGmLateAckReadyCandidate, RepeatedReadyDiscoveryCannotPrepareBeforeEveryResourceLaneArrives) {
    const auto [mask, delayed] = GetParam();
    const uint32_t observer = (delayed + 1) % 3;
    const uint8_t ordinary_mask = observer == 0 ? 1 : 2;
    GmMixFixture fixture({{1, mask}, {1, ordinary_mask}}, true);
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(0));
    bool progress = false;
    for (uint32_t lane : {1u, 2u})
        ASSERT_TRUE(fixture.observe(lane, &progress));
    for (uint32_t lane = 0; lane < 3; ++lane)
        if (lane != delayed) ASSERT_TRUE(fixture.arrive(lane, &progress));
    ASSERT_TRUE(fixture.enqueue(1, observer));
    auto *block =
        scheduler_state_at<SchedulerProtocolBlockTrace>(fixture.base(), fixture.storage.layout.protocol.blocks_offset);
    for (uint32_t poll = 0; poll < 8; ++poll) {
        EXPECT_TRUE(
            scheduler_ordered_poll_may_progress(fixture.graph.graph(), fixture.base(), fixture.locals[observer])
        );
        ASSERT_TRUE(fixture.fill(observer));
        EXPECT_EQ(fixture.admissions[observer].gm_mix.phase, SchedulerGmMixPhase::ARRIVED);
        EXPECT_EQ(fixture.locals[observer]->local_ready_mask, 0u);
        EXPECT_EQ(block->lanes[observer].arrivals_observed.sequence, 0u);
        EXPECT_EQ(block->lanes[observer].release_before.sequence, 0u);
        EXPECT_EQ(fixture.control(0)->completed_blocks, 0u);
        EXPECT_EQ(fixture.control(1)->state, static_cast<int64_t>(SchedulerTaskState::BLOCKED));
    }
    ASSERT_TRUE(fixture.arrive(delayed, &progress));
    ASSERT_TRUE(fixture.fill(observer));
    EXPECT_NE(block->lanes[observer].arrivals_observed.sequence, 0u);
    for (uint32_t lane = 0; lane < 3; ++lane)
        EXPECT_EQ(block->lanes[observer].arrived_seq[lane], 1u);
    EXPECT_EQ(scheduler_read_error(fixture.run()), 0u);
}

INSTANTIATE_TEST_SUITE_P(
    AllMasksAndDelayedLanes, HbgGmLateAckReadyCandidate,
    ::testing::Combine(::testing::Values<uint8_t>(3, 5, 6, 7), ::testing::Values<uint32_t>(0, 1, 2))
);

TEST(HbgGmMix, FutureArrivedAcknowledgementIsRejectedInsteadOfSatisfyingTheBarrier) {
    GmMixFixture fixture({{1, 7}}, true);
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(0));
    bool progress = false;
    for (uint32_t worker : {1u, 2u})
        ASSERT_TRUE(fixture.observe(worker, &progress));
    for (uint32_t worker = 0; worker < 3; ++worker)
        ASSERT_TRUE(fixture.arrive(worker, &progress));
    scheduler_gm_store(fixture.mailbox(0)->lanes[2].arrived_seq, UINT64_C(2));
    EXPECT_FALSE(fixture.prepare(0, &progress));
    EXPECT_EQ(fixture.locals[0]->local_ready_mask, 0u);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.control(0)->completed_blocks, UINT64_C(0)), 0u);
    const auto *record =
        scheduler_state_at<SchedulerProtocolBlockTrace>(fixture.base(), fixture.storage.layout.protocol.blocks_offset);
    EXPECT_EQ(record->lanes[0].arrivals_observed.sequence, 0u);
    EXPECT_EQ(record->owner.completion_before.sequence, 0u);
}

TEST(HbgGmMix, FutureReleasedAcknowledgementIsRejectedBeforeBlockContribution) {
    GmMixFixture fixture({{1, 7}}, true);
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(0));
    ASSERT_TRUE(fixture.rendezvous(0));
    ASSERT_TRUE(fixture.finish_lanes(0));
    scheduler_gm_store(fixture.mailbox(0)->lanes[2].released_seq, UINT64_C(2));
    EXPECT_EQ(fixture.retire(0), SchedulerGmMixResult::ERROR);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.control(0)->completed_blocks, UINT64_C(0)), 0u);
    const auto *record =
        scheduler_state_at<SchedulerProtocolBlockTrace>(fixture.base(), fixture.storage.layout.protocol.blocks_offset);
    EXPECT_EQ(record->owner.releases_observed.sequence, 0u);
    EXPECT_EQ(record->owner.completion_before.sequence, 0u);
    EXPECT_NE(record->lanes[2].release_after.sequence, 0u);
}

TEST(HbgGmMix, WrongRetainedSealCannotAdvanceALateVectorFrontier) {
    GmMixFixture fixture({{1, 7}});
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(0));
    ASSERT_TRUE(fixture.rendezvous(0));
    ASSERT_TRUE(fixture.finish_lanes(0));
    ASSERT_EQ(fixture.retire(0), SchedulerGmMixResult::TASK_LAST);
    bool closed = false;
    ASSERT_TRUE(fixture.close(0, &closed));
    ASSERT_TRUE(closed);
    scheduler_gm_store(fixture.seal(0, 0), UINT64_C(1));
    EXPECT_FALSE(fixture.close(1, &closed));
    EXPECT_FALSE(closed);
    EXPECT_EQ(fixture.admissions[1].current_task_ref, 0);
    EXPECT_EQ(fixture.admissions[1].scan_cursor, 0u);
}

TEST(HbgGmMix, ExitDrainNeverClaimsOrPreparesUnexecutedWork) {
    GmMixFixture fixture({{2, 7}});
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    bool done = true;
    ASSERT_TRUE(fixture.drain(0, &done));
    EXPECT_FALSE(done);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.control(0)->next_block, UINT64_C(0)), 0u);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.mailbox(0)->published_seq, UINT64_C(0)), 0u);
    EXPECT_EQ(fixture.admissions[0].gm_mix.phase, SchedulerGmMixPhase::IDLE);
    EXPECT_EQ(fixture.locals[0]->local_ready_mask, 0u);
}

TEST(HbgGmMix, ExitDrainConsumesOutstandingSealsWithoutPublishingAnotherGroup) {
    GmMixFixture fixture({{1, 7}});
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(0));
    ASSERT_TRUE(fixture.rendezvous(0));
    ASSERT_TRUE(fixture.finish_lanes(0));
    ASSERT_EQ(fixture.retire(0), SchedulerGmMixResult::TASK_LAST);
    ASSERT_TRUE(fixture.complete_task(0, 0));
    bool done = true;
    ASSERT_TRUE(fixture.drain(1, &done));
    EXPECT_FALSE(done);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.seal(0, 0), UINT64_C(0)), 0u);
    ASSERT_TRUE(fixture.drain(0, &done));
    EXPECT_FALSE(done);
    for (uint32_t worker = 0; worker < 6; ++worker) {
        done = false;
        for (uint32_t step = 0; step < 3 && !done; ++step)
            ASSERT_TRUE(fixture.drain(worker, &done));
        EXPECT_TRUE(done);
    }
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.mailbox(0)->published_seq, UINT64_C(0)), 1u);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.mailbox(1)->published_seq, UINT64_C(0)), 0u);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.control(0)->next_block, UINT64_C(0)), 1u);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.seal(0, 0), UINT64_C(0)), 2u);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.seal(0, 1), UINT64_C(0)), 1u);
}

TEST(HbgGmMix, MixFaninWaitsForLastGroupAndMaskSixCoordinatorReleasesItOnce) {
    GmMixFixture fixture({{2, 6}, {1, 3}});
    fixture.graph.executable(1, 0, {0});
    fixture.storage.metadata[1].flags |= SCHEDULER_TASK_HAS_FANIN;
    ASSERT_TRUE(scheduler_init_spmd_task_records(fixture.base(), fixture.storage.layout, 47));
    ASSERT_EQ(
        scheduler_route_task(fixture.graph.graph(), fixture.base(), fixture.locals[0], fixture.run(), 1, nullptr),
        SchedulerRouteResult::WAITING
    );
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(0));
    ASSERT_TRUE(fixture.rendezvous(0));
    ASSERT_TRUE(fixture.finish_lanes(0));
    ASSERT_EQ(fixture.retire(0), SchedulerGmMixResult::PROGRESS);
    EXPECT_EQ(scheduler_gm_load(fixture.control(0)->state), static_cast<int64_t>(SchedulerTaskState::BLOCKED));
    EXPECT_EQ(scheduler_gm_load(fixture.control(0)->wake_list_head), 1);
    EXPECT_EQ(
        scheduler_gm_query(fixture.control(1)->admission_state),
        static_cast<uint64_t>(SchedulerAdmissionState::UNCLAIMED)
    );
    EXPECT_EQ(fixture.locals[0]->pending_completed, 0u);
    ASSERT_EQ(fixture.claim(1), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(1));
    ASSERT_TRUE(fixture.rendezvous(1));
    ASSERT_TRUE(fixture.finish_lanes(1));
    ASSERT_EQ(fixture.retire(1), SchedulerGmMixResult::TASK_LAST);
    ASSERT_TRUE(fixture.complete_task(0, 1));
    EXPECT_EQ(scheduler_gm_load(fixture.control(0)->wake_list_head), SCHEDULER_WAKE_LIST_CLOSED);
    EXPECT_EQ(fixture.locals[3]->pending_completed, 1u);
    auto *inbox = scheduler_ready_inbox_at(fixture.base(), fixture.locals[3], SCHEDULER_MIX_QUEUE, 1);
    EXPECT_EQ(scheduler_ready_endpoints_head(scheduler_gm_query(inbox->endpoints)), 1);
    uint32_t successor_executions = 0;
    for (uint32_t step = 0; step < 100; ++step) {
        for (uint32_t worker = 0; worker < 6; ++worker) {
            ASSERT_TRUE(fixture.fill(worker));
            GmExecution execution{};
            if (!fixture.begin(worker, &execution)) continue;
            EXPECT_EQ(execution.task, 1);
            EXPECT_EQ(execution.block, 0u);
            EXPECT_EQ(execution.blocks, 1u);
            ++successor_executions;
            ASSERT_TRUE(fixture.complete(execution));
        }
        if (scheduler_gm_load(fixture.control(1)->state) == static_cast<int64_t>(SchedulerTaskState::DONE)) break;
    }
    EXPECT_EQ(successor_executions, 2u);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.control(0)->completed_blocks, UINT64_C(0)), 2u);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.control(1)->completed_blocks, UINT64_C(0)), 1u);
    for (auto *local : fixture.locals)
        scheduler_flush_completions(fixture.run(), local);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.run()->resolved_task_count, UINT64_C(0)), 2u);
}

class HbgGmMixExitInFlight : public ::testing::TestWithParam<SchedulerGmMixPhase> {};

TEST_P(HbgGmMixExitInFlight, ExitDrainRejectsAnOutstandingGroupWithoutClaimingMoreWork) {
    GmMixFixture fixture({{2, 7}});
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    if (GetParam() != SchedulerGmMixPhase::CLAIMED) ASSERT_TRUE(fixture.publish(0));
    if (GetParam() == SchedulerGmMixPhase::ARRIVED || GetParam() == SchedulerGmMixPhase::RUNNING) {
        bool progress = false;
        for (uint32_t worker : {1u, 2u})
            ASSERT_TRUE(fixture.observe(worker, &progress));
        for (uint32_t worker = 0; worker < 3; ++worker)
            ASSERT_TRUE(fixture.arrive(worker, &progress));
        if (GetParam() == SchedulerGmMixPhase::RUNNING)
            for (uint32_t worker = 0; worker < 3; ++worker)
                ASSERT_TRUE(fixture.prepare(worker, &progress));
    }
    ASSERT_EQ(fixture.admissions[0].gm_mix.phase, GetParam());
    bool done = true;
    EXPECT_FALSE(fixture.drain(0, &done));
    EXPECT_FALSE(done);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.control(0)->next_block, UINT64_C(0)), 1u);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.control(0)->completed_blocks, UINT64_C(0)), 0u);
    EXPECT_EQ(fixture.admissions[0].gm_mix.phase, GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    UnretiredGroupPhases, HbgGmMixExitInFlight,
    ::testing::Values(
        SchedulerGmMixPhase::CLAIMED, SchedulerGmMixPhase::PUBLISHED, SchedulerGmMixPhase::ARRIVED,
        SchedulerGmMixPhase::RUNNING
    )
);

class HbgGmMixOwnPrepare : public ::testing::TestWithParam<uint8_t> {};

TEST_P(HbgGmMixOwnPrepare, HasMixDoesNotRequireSsbufForAnOwnPayloadPreparation) {
    const uint8_t mask = GetParam();
    GmMixFixture fixture({{1, mask}, {1, 7}});
    for (uint8_t subtask = 0; subtask < 3; ++subtask) {
        if ((mask & (1U << subtask)) == 0) continue;
        auto *local = fixture.locals[subtask];
        ASSERT_TRUE(local->has_mix);
        local->slots[0].state = SchedulerDispatchSlotState::FILLING;
        SchedulerReadyClaim ready{};
        ready.task_ref = 0;
        ASSERT_TRUE(scheduler_prepare_dispatch_slot(
            fixture.graph.graph(), fixture.base(), local, fixture.run(),
            {subtask, 0, local->slots[0].generation, subtask}, ready, fixture.storage.metadata[0], 0, nullptr, subtask
        ));
        const auto *payload =
            scheduler_state_at<DispatchPayload>(fixture.base(), local->dispatch_payload_offset(subtask, 0));
        EXPECT_NE(payload->function_bin_addr, 0u);
        EXPECT_EQ(payload->local_context.block_num, 1);
    }
}

INSTANTIATE_TEST_SUITE_P(PureAndMixedPayloads, HbgGmMixOwnPrepare, ::testing::Values<uint8_t>(1, 2, 3, 5, 6, 7));

class HbgGmProtocolEvents : public ::testing::TestWithParam<uint8_t> {};

TEST_P(HbgGmProtocolEvents, CapturesPublicationDescriptorActualAcksAndUniqueBlockCompletion) {
    const uint8_t mask = GetParam();
    GmMixFixture fixture({{1, mask}}, true);
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    auto *blocks =
        scheduler_state_at<SchedulerProtocolBlockTrace>(fixture.base(), fixture.storage.layout.protocol.blocks_offset);
    auto &record = blocks[0];
    ASSERT_NE(record.owner.claim_after.sequence, 0u);
    EXPECT_EQ(record.owner.publish_before.sequence, 0u);
    EXPECT_EQ(record.lanes[1].descriptor_ready.sequence, 0u);
    bool progress = true;
    ASSERT_TRUE(fixture.observe(1, &progress));
    EXPECT_FALSE(progress);
    ASSERT_TRUE(fixture.publish(0));
    EXPECT_GT(record.owner.publish_before.sequence, record.owner.claim_after.sequence);
    EXPECT_GT(record.owner.publish_after.sequence, record.owner.publish_before.sequence);
    EXPECT_EQ(record.owner.group_seq, 1u);
    EXPECT_EQ(record.owner.active_mask, mask);
    EXPECT_EQ(record.owner.logical_block_num, 1u);
    EXPECT_GT(record.lanes[0].descriptor_ready.sequence, record.owner.publish_after.sequence);
    EXPECT_EQ(record.lanes[1].descriptor_ready.sequence, 0u);
    ASSERT_TRUE(fixture.rendezvous(0));
    for (uint32_t lane = 0; lane < 3; ++lane) {
        const auto &event = record.lanes[lane];
        EXPECT_EQ(event.worker_index, lane);
        EXPECT_EQ(event.lane, lane);
        EXPECT_EQ(event.run_epoch, 47u);
        EXPECT_EQ(event.group_seq, 1u);
        EXPECT_EQ(event.active_mask, mask);
        EXPECT_EQ(event.resource_mask, 7u);
        EXPECT_EQ(event.logical_block_num, 1u);
        EXPECT_GT(event.arrive_before.sequence, event.descriptor_ready.sequence);
        EXPECT_GT(event.arrive_after.sequence, event.arrive_before.sequence);
        EXPECT_GT(event.arrivals_observed.sequence, event.arrive_after.sequence);
        for (uint32_t peer = 0; peer < 3; ++peer) {
            EXPECT_EQ(event.arrived_seq[peer], 1u);
            EXPECT_EQ(event.lane_subtasks[peer], (mask & (1U << peer)) != 0 ? peer : UINT8_MAX);
        }
        if ((mask & (1U << lane)) != 0) EXPECT_EQ(event.release_before.sequence, 0u);
        else EXPECT_GT(event.release_before.sequence, event.arrivals_observed.sequence);
    }
    EXPECT_EQ(fixture.retire(0), SchedulerGmMixResult::WAIT);
    EXPECT_EQ(record.owner.releases_observed.sequence, 0u);
    EXPECT_EQ(record.owner.completion_before.sequence, 0u);
    ASSERT_TRUE(fixture.finish_lanes(0));
    ASSERT_EQ(fixture.retire(0), SchedulerGmMixResult::TASK_LAST);
    for (uint32_t lane = 0; lane < 3; ++lane) {
        EXPECT_GT(record.lanes[lane].release_after.sequence, record.lanes[lane].release_before.sequence);
        EXPECT_EQ(record.owner.released_seq[lane], 1u);
    }
    EXPECT_GT(record.owner.releases_observed.sequence, record.lanes[0].release_after.sequence);
    EXPECT_GT(record.owner.completion_before.sequence, record.owner.releases_observed.sequence);
    EXPECT_GT(record.owner.completion_after.sequence, record.owner.completion_before.sequence);
    EXPECT_EQ(record.owner.completion_ticket, 0u);
    EXPECT_EQ(record.owner.worker_index, 0u);
    ASSERT_TRUE(fixture.complete_task(0, 0));
    const auto *tasks =
        scheduler_state_at<SchedulerProtocolTaskTrace>(fixture.base(), fixture.storage.layout.protocol.tasks_offset);
    EXPECT_GT(tasks[0].completion.before.sequence, record.owner.completion_after.sequence);
    EXPECT_GT(tasks[0].completion.after.sequence, tasks[0].completion.before.sequence);
    EXPECT_EQ(tasks[0].completion.order_key, 0u);
    EXPECT_EQ(tasks[0].completion.worker_index, 0u);
    EXPECT_EQ(blocks[1].owner.claim_after.sequence, 0u);
    const auto *workers = scheduler_state_at<SchedulerProtocolWorkerTrace>(
        fixture.base(), fixture.storage.layout.protocol.workers_offset
    );
    for (uint32_t lane = 0; lane < 3; ++lane)
        EXPECT_EQ(workers[lane].error_flags, 0u);
}

INSTANTIATE_TEST_SUITE_P(AllMixMasks, HbgGmProtocolEvents, ::testing::Values<uint8_t>(3, 5, 6, 7));

TEST(HbgGmMixProtocol, OldAcksCannotCreateNewGroupEvidenceAndMailboxReuseKeepsEarlierRecords) {
    GmMixFixture fixture({{2, 7}}, true);
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(0));
    ASSERT_TRUE(fixture.rendezvous(0));
    ASSERT_TRUE(fixture.finish_lanes(0));
    ASSERT_EQ(fixture.retire(0), SchedulerGmMixResult::PROGRESS);
    auto *blocks =
        scheduler_state_at<SchedulerProtocolBlockTrace>(fixture.base(), fixture.storage.layout.protocol.blocks_offset);
    const auto first = blocks[0];
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(0));
    bool progress = false;
    for (uint32_t lane : {1u, 2u})
        ASSERT_TRUE(fixture.observe(lane, &progress));
    ASSERT_TRUE(fixture.arrive(0, &progress));
    ASSERT_TRUE(fixture.prepare(0, &progress));
    EXPECT_FALSE(progress);
    EXPECT_EQ(blocks[3].lanes[0].arrivals_observed.sequence, 0u);
    EXPECT_EQ(blocks[3].lanes[0].release_before.sequence, 0u);
    for (uint32_t lane : {1u, 2u})
        ASSERT_TRUE(fixture.arrive(lane, &progress));
    for (uint32_t lane = 0; lane < 3; ++lane)
        ASSERT_TRUE(fixture.prepare(lane, &progress));
    const uint64_t checked_sequence = blocks[3].lanes[0].arrivals_observed.sequence;
    ASSERT_NE(checked_sequence, 0u);
    bool all = false;
    ASSERT_TRUE(scheduler_gm_mix_check_acks(
        fixture.graph.graph(), fixture.base(), fixture.locals[0], fixture.run(), false, &all
    ));
    EXPECT_TRUE(all);
    EXPECT_EQ(blocks[3].lanes[0].arrivals_observed.sequence, checked_sequence);
    for (uint32_t lane = 0; lane < 3; ++lane)
        for (uint32_t peer = 0; peer < 3; ++peer)
            EXPECT_EQ(blocks[3].lanes[lane].arrived_seq[peer], 2u);
    for (uint32_t lane : {0u, 1u}) {
        GmExecution execution{};
        ASSERT_TRUE(fixture.begin(lane, &execution));
        ASSERT_TRUE(fixture.complete(execution));
    }
    EXPECT_EQ(fixture.retire(0), SchedulerGmMixResult::WAIT);
    EXPECT_EQ(blocks[3].owner.releases_observed.sequence, 0u);
    EXPECT_EQ(blocks[3].owner.completion_before.sequence, 0u);
    GmExecution last{};
    ASSERT_TRUE(fixture.begin(2, &last));
    ASSERT_TRUE(fixture.complete(last));
    ASSERT_EQ(fixture.retire(0), SchedulerGmMixResult::TASK_LAST);
    for (uint32_t peer = 0; peer < 3; ++peer)
        EXPECT_EQ(blocks[3].owner.released_seq[peer], 2u);
    EXPECT_EQ(blocks[3].owner.completion_ticket, 1u);
    EXPECT_EQ(blocks[0].owner.claim_before.sequence, first.owner.claim_before.sequence);
    EXPECT_EQ(blocks[0].owner.completion_after.sequence, first.owner.completion_after.sequence);
    for (uint32_t lane = 0; lane < 3; ++lane) {
        EXPECT_EQ(blocks[0].lanes[lane].descriptor_ready.sequence, first.lanes[lane].descriptor_ready.sequence);
        EXPECT_EQ(blocks[0].lanes[lane].release_after.sequence, first.lanes[lane].release_after.sequence);
        EXPECT_EQ(blocks[0].lanes[lane].group_seq, 1u);
    }
}

TEST(HbgGmMixProtocol, SameSequenceInDifferentClustersDoesNotAliasBlockOrLaneRecords) {
    GmMixFixture fixture({{2, 6}}, true);
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    for (uint32_t cluster = 0; cluster < 2; ++cluster) {
        ASSERT_EQ(fixture.claim(cluster), SchedulerBlockClaimResult::CLAIMED);
        ASSERT_TRUE(fixture.publish(cluster));
        ASSERT_TRUE(fixture.rendezvous(cluster));
    }
    const auto *blocks =
        scheduler_state_at<SchedulerProtocolBlockTrace>(fixture.base(), fixture.storage.layout.protocol.blocks_offset);
    for (uint32_t cluster = 0; cluster < 2; ++cluster) {
        ASSERT_TRUE(fixture.finish_lanes(cluster));
        ASSERT_EQ(
            fixture.retire(cluster), cluster == 0 ? SchedulerGmMixResult::PROGRESS : SchedulerGmMixResult::TASK_LAST
        );
        const auto &record = blocks[cluster * 2];
        EXPECT_EQ(record.owner.cluster_index, cluster);
        EXPECT_EQ(record.owner.worker_index, cluster * 3);
        EXPECT_EQ(record.owner.block_idx, cluster);
        EXPECT_EQ(record.owner.group_seq, 1u);
        EXPECT_EQ(record.owner.completion_ticket, cluster);
        for (uint32_t lane = 0; lane < 3; ++lane) {
            EXPECT_EQ(record.lanes[lane].worker_index, cluster * 3 + lane);
            EXPECT_EQ(record.lanes[lane].cluster_index, cluster);
            EXPECT_EQ(record.lanes[lane].block_idx, cluster);
            EXPECT_EQ(record.lanes[lane].group_seq, 1u);
            EXPECT_NE(record.lanes[lane].release_after.sequence, 0u);
        }
    }
    EXPECT_EQ(blocks[1].owner.claim_after.sequence, 0u);
}

TEST(HbgGmMixProtocol, FullWidthGroupSequenceIsPreservedInObservedAckValues) {
    GmMixFixture fixture({{1, 7}}, true);
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    const uint64_t previous = UINT64_MAX - 2;
    for (uint32_t lane = 0; lane < 3; ++lane) {
        fixture.admissions[lane].gm_mix.group_seq = previous;
        scheduler_gm_store(fixture.mailbox(0)->lanes[lane].arrived_seq, previous);
        scheduler_gm_store(fixture.mailbox(0)->lanes[lane].released_seq, previous);
    }
    scheduler_gm_store(fixture.mailbox(0)->published_seq, previous);
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(0));
    ASSERT_TRUE(fixture.rendezvous(0));
    ASSERT_TRUE(fixture.finish_lanes(0));
    ASSERT_EQ(fixture.retire(0), SchedulerGmMixResult::TASK_LAST);
    const auto *record =
        scheduler_state_at<SchedulerProtocolBlockTrace>(fixture.base(), fixture.storage.layout.protocol.blocks_offset);
    EXPECT_EQ(record->owner.group_seq, UINT64_MAX - 1);
    for (uint32_t lane = 0; lane < 3; ++lane) {
        EXPECT_EQ(record->owner.released_seq[lane], UINT64_MAX - 1);
        EXPECT_EQ(record->lanes[lane].group_seq, UINT64_MAX - 1);
        for (uint32_t peer = 0; peer < 3; ++peer)
            EXPECT_EQ(record->lanes[lane].arrived_seq[peer], UINT64_MAX - 1);
    }
}

TEST(HbgGmMixProtocol, ReplayedPublicationCannotOverwriteItsFirstInterval) {
    GmMixFixture fixture({{1, 7}}, true);
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(0));
    const auto *record =
        scheduler_state_at<SchedulerProtocolBlockTrace>(fixture.base(), fixture.storage.layout.protocol.blocks_offset);
    const auto first = record->owner;
    scheduler_protocol_group_publish(fixture.base(), fixture.locals[0], true);
    scheduler_protocol_group_publish(fixture.base(), fixture.locals[0], false);
    EXPECT_EQ(record->owner.publish_before.sequence, first.publish_before.sequence);
    EXPECT_EQ(record->owner.publish_after.sequence, first.publish_after.sequence);
    EXPECT_EQ(record->owner.group_seq, first.group_seq);
    const auto *worker = scheduler_state_at<SchedulerProtocolWorkerTrace>(
        fixture.base(), fixture.storage.layout.protocol.workers_offset
    );
    EXPECT_EQ(worker->error_flags, SCHEDULER_PROTOCOL_DUPLICATE);
}

}  // namespace
