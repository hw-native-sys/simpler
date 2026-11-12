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
#include <memory>
#include <string>
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
        storage.prepare_ready_pool();
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
    bool bootstrap_admission() {
        if (run()->bootstrap_scan_complete != 0) return false;
        for (uint64_t task = 0; task < graph.graph().task_count; ++task) {
            if (!scheduler_task_requires_admission(scheduler_load_dispatch_metadata(base(), locals[0], task, 0)))
                continue;
            if (!scheduler_bootstrap_prepare_admission(
                    graph.graph(), base(), locals[0], run(), &admissions[0], static_cast<int64_t>(task)
                ))
                return false;
        }
        scheduler_gm_store(run()->bootstrap_scan_complete, UINT64_C(1));
        return true;
    }
    bool enqueue(uint64_t task, uint32_t worker) {
        auto *local = locals[worker];
        if (scheduler_route_task(graph.graph(), base(), local, run(), task) != SchedulerRouteResult::READY_TO_ENQUEUE)
            return false;
        SchedulerReadyBatch batch{};
        const uint32_t queue =
            scheduler_task_ready_queue(storage.metadata[task].flags, storage.metadata[task].active_mask);
        return scheduler_ready_batch_append(base(), local, task, &batch) &&
               scheduler_ready_batch_push(base(), local, queue, &batch);
    }
    bool pop(uint64_t task, uint32_t worker, SchedulerReadyClaim *ready) {
        if (!enqueue(task, worker)) return false;
        const uint32_t queue =
            scheduler_task_ready_queue(storage.metadata[task].flags, storage.metadata[task].active_mask);
        return scheduler_claim_ready_for_slot(
                   graph.graph(), base(), locals[worker], run(), locals[worker]->config.scheduler_count, queue, ready
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
        scheduler_fill_own_slots(graph.graph(), base(), locals[worker], run(), profiling, &failed);
        return !failed;
    }
    std::vector<uint32_t> members(uint64_t task) {
        const auto *record = scheduler_state_at<SchedulerAdmissionRecord>(
            base(), storage.layout.admission_records_offset + task * storage.layout.admission_record_stride
        );
        uint64_t words[(SCHEDULER_WORKER_CAPACITY + 63) / 64]{};
        uint32_t count = 0;
        EXPECT_TRUE(scheduler_admission_expected_members(base(), locals[0], &admissions[0], record, words, &count));
        std::vector<uint32_t> result;
        for (uint32_t worker = 0; worker < locals.size(); ++worker)
            if ((words[worker / 64] & (UINT64_C(1) << (worker % 64))) != 0) result.push_back(worker);
        EXPECT_EQ(result.size(), count);
        return result;
    }
    uint32_t member(uint64_t task, uint32_t preferred) {
        const auto workers = members(task);
        EXPECT_FALSE(workers.empty());
        if (workers.empty()) return preferred;
        return std::find(workers.begin(), workers.end(), preferred) != workers.end() ? preferred : workers.front();
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
        return scheduler_fill_dispatch_slot(graph.graph(), base(), local, run(), {slot}, ready, 0);
    }
    bool begin(uint32_t worker, OrderedExecution *execution) {
        auto *local = locals[worker];
        uint32_t slot = UINT32_MAX;
        if (!scheduler_local_ready_peek(local, &slot)) return false;
        bool permit = false;
        if (!scheduler_ordered_can_enter_slot(graph.graph(), base(), local, run(), slot, &permit)) return false;
        if (!permit) return false;
        if (!scheduler_local_ready_execute(local, slot)) return false;
        auto *payload =
            scheduler_state_at<DispatchPayload>(base(), local->dispatch_payload_offset(local->config.self_lane, slot));
        if (scheduler_protocol_enabled(local)) scheduler_execute_protocol_task(base(), local, slot, payload);
        *execution = {
            local->slots[slot].task_ref,
            static_cast<uint32_t>(payload->local_context.block_idx),
            static_cast<uint32_t>(payload->local_context.block_num),
            worker,
            slot,
            payload->function_bin_addr
        };
        return true;
    }
    bool complete(const OrderedExecution &execution) {
        auto *local = locals[execution.worker];
        return scheduler_complete_own_slot(graph.graph(), base(), local, run(), execution.slot, 0);
    }
    bool close(uint32_t worker, bool *closed) {
        return scheduler_close_ordered_obligation(graph.graph(), base(), locals[worker], run(), closed);
    }

    scheduler_test::FixtureStorage storage;
    scheduler_test::GraphBuffer graph;
    std::array<SchedulerAdmissionLocalState, 6> admissions{};
    std::array<SchedulerLocalState *, 6> locals{};
};

std::vector<SchedulerScalarRecord> scalar_rows(const SchedulerScalarTrace &trace) {
    std::vector<SchedulerScalarRecord> rows;
    uint32_t cursor = 0;
    for (uint32_t i = 0; i < trace.committed; ++i) {
        SchedulerScalarRecord row{};
        EXPECT_TRUE(scheduler_decode_scalar_storage(trace.records, trace.storage_slots, cursor, row));
        rows.push_back(row);
    }
    EXPECT_EQ(cursor, trace.storage_slots);
    return rows;
}

class HbgOrderedScalarDetail : public ::testing::TestWithParam<bool> {};

TEST_P(HbgOrderedScalarDetail, FillPublishesExecutableWorkAndBalancesNestedRecording) {
    OrderedFixture fixture({{4, LaneMask::aic()}});
    ASSERT_TRUE(fixture.bootstrap_admission());
    ASSERT_TRUE(fixture.enqueue(0, 0));
    auto buffer = std::make_unique<SchedulerActivityBuffer>();
    auto records = std::make_unique<SchedulerScalarStorageSlot[]>(2 * SCHEDULER_SCALAR_RECORD_CAPACITY);
    auto &trace = fixture.locals[0]->profiling->scalar;
    trace.buffer = GetParam() ? buffer.get() : nullptr;
    trace.records = records.get();
    trace.window_start = scheduler_cycles();
    ASSERT_TRUE(fixture.fill(0));
    EXPECT_EQ(trace.depth, 0u);
    EXPECT_EQ(fixture.control(0)->next_block, 1u);
    EXPECT_NE(fixture.locals[0]->local_ready_mask, 0u);
    trace.finish(scheduler_cycles());
    const auto decoded = scalar_rows(trace);
    if (!GetParam()) {
        EXPECT_EQ(trace.committed, 0u);
        EXPECT_EQ(trace.storage_slots, 0u);
        return;
    }
    ASSERT_EQ(buffer->scalar_dropped, 0u);
    auto find = [&](const std::string &name) -> const SchedulerScalarRecord * {
        for (uint32_t i = 0; i < trace.committed; ++i)
            if (name == sched_phase_kind_name(static_cast<SchedPhaseKind>(decoded[i].kind & 0xffffU)))
                return &decoded[i];
        return nullptr;
    };
    const auto *fill = find("scalar_fill_slots");
    const auto *ordered = find("scalar_ordered_fill");
    ASSERT_NE(fill, nullptr);
    ASSERT_NE(ordered, nullptr);
    EXPECT_EQ(ordered->kind >> 16, (fill->kind >> 16) + 1);
    for (const char *name :
         {"scalar_ordered_poll", "scalar_admission_claim", "scalar_admission_publish", "scalar_admission_prepare",
          "scalar_admission_commit", "scalar_admission_refresh", "scalar_ordinary_park", "scalar_block_claim",
          "scalar_admission_close", "scalar_ready_claim", "scalar_dispatch_metadata", "scalar_dispatch_fill",
          "scalar_dispatch_publish"}) {
        const auto *child = find(name);
        ASSERT_NE(child, nullptr) << name;
        EXPECT_GT(child->kind >> 16, ordered->kind >> 16) << name;
        EXPECT_GE(child->start_time, ordered->start_time) << name;
        EXPECT_LE(child->end_time, ordered->end_time) << name;
    }
}

INSTANTIATE_TEST_SUITE_P(Recording, HbgOrderedScalarDetail, ::testing::Bool());

TEST(HbgOrderedScalarDetail, EmptyPollStillRecordsItsEarlyReturn) {
    OrderedFixture fixture({{4, LaneMask::aic()}});
    auto buffer = std::make_unique<SchedulerActivityBuffer>();
    auto records = std::make_unique<SchedulerScalarStorageSlot[]>(2 * SCHEDULER_SCALAR_RECORD_CAPACITY);
    auto &trace = fixture.locals[0]->profiling->scalar;
    trace.buffer = buffer.get();
    trace.records = records.get();
    trace.window_start = scheduler_cycles();
    bool failed = false;
    EXPECT_FALSE(
        scheduler_fill_own_slots(fixture.graph.graph(), fixture.base(), fixture.locals[0], fixture.run(), 0, &failed)
    );
    EXPECT_FALSE(failed);
    EXPECT_EQ(trace.depth, 0u);
    trace.finish(scheduler_cycles());
    const auto decoded = scalar_rows(trace);
    uint32_t polls = 0;
    for (uint32_t i = 0; i < trace.committed; ++i)
        if (std::string(sched_phase_kind_name(static_cast<SchedPhaseKind>(decoded[i].kind & 0xffffU))) ==
            "scalar_ordered_poll")
            ++polls;
    EXPECT_EQ(polls, 1u);
    EXPECT_EQ(buffer->scalar_dropped, 0u);
}

TEST(HbgOrderedScalarDetail, NonAdmissionFillExposesReadyAndDispatchWork) {
    OrderedFixture fixture({{1, LaneMask::aic()}});
    ASSERT_TRUE(fixture.bootstrap_admission());
    ASSERT_TRUE(fixture.enqueue(0, 0));
    fixture.locals[0]->admission = nullptr;
    auto buffer = std::make_unique<SchedulerActivityBuffer>();
    auto records = std::make_unique<SchedulerScalarStorageSlot[]>(2 * SCHEDULER_SCALAR_RECORD_CAPACITY);
    auto &trace = fixture.locals[0]->profiling->scalar;
    trace.buffer = buffer.get();
    trace.records = records.get();
    trace.window_start = scheduler_cycles();
    ASSERT_TRUE(fixture.fill(0));
    EXPECT_NE(fixture.locals[0]->local_ready_mask, 0u);
    EXPECT_EQ(trace.depth, 0u);
    trace.finish(scheduler_cycles());
    const auto decoded = scalar_rows(trace);
    std::vector<std::string> names;
    for (uint32_t i = 0; i < trace.committed; ++i)
        names.emplace_back(sched_phase_kind_name(static_cast<SchedPhaseKind>(decoded[i].kind & 0xffffU)));
    for (const char *name :
         {"scalar_fill_slots", "scalar_ready_claim", "scalar_dispatch_metadata", "scalar_dispatch_fill",
          "scalar_dispatch_publish"})
        EXPECT_NE(std::find(names.begin(), names.end(), name), names.end()) << name;
    EXPECT_EQ(std::find(names.begin(), names.end(), "scalar_ordered_fill"), names.end());
    EXPECT_EQ(buffer->scalar_dropped, 0u);
}

TEST(HbgOrderedScalarDetail, ClaimCursorDetailPreservesSealAndFallbackPaths) {
    for (const auto &[seal, next, valid, used_seal] :
         {std::tuple<uint64_t, uint64_t, bool, bool>{0, 1, true, false}, {2, 2, true, true}, {2, 1, false, false}}) {
        SCOPED_TRACE(seal);
        SCOPED_TRACE(next);
        OrderedFixture fixture({{2, LaneMask::aic()}});
        ASSERT_TRUE(fixture.bootstrap_admission());
        ASSERT_TRUE(fixture.admit(0, 0));
        ASSERT_TRUE(scheduler_refresh_local_obligation(
            fixture.graph.graph(), fixture.base(), fixture.locals[0], fixture.run(), &fixture.admissions[0]
        ));
        fixture.control(0)->claim_seal = seal;
        fixture.control(0)->next_block = next;
        auto buffer = std::make_unique<SchedulerActivityBuffer>();
        auto records = std::make_unique<SchedulerScalarStorageSlot[]>(2 * SCHEDULER_SCALAR_RECORD_CAPACITY);
        auto &trace = fixture.locals[0]->profiling->scalar;
        trace.buffer = buffer.get();
        trace.records = records.get();
        trace.window_start = scheduler_cycles();
        uint64_t cursor = UINT64_MAX;
        bool observed_seal = false;
        EXPECT_EQ(
            scheduler_read_claim_cursor(
                fixture.graph.graph(), fixture.base(), fixture.locals[0], fixture.run(), &cursor, &observed_seal
            ),
            valid
        );
        if (valid) EXPECT_EQ(cursor, next);
        EXPECT_EQ(observed_seal, used_seal);
        EXPECT_EQ(fixture.admissions[0].scan_cursor, 0u);
        EXPECT_EQ(fixture.control(0)->next_block, next);
        EXPECT_EQ(trace.depth, 0u);
        trace.finish(scheduler_cycles());
        const auto decoded = scalar_rows(trace);
        std::vector<std::string> names;
        for (uint32_t i = 0; i < trace.committed; ++i)
            names.emplace_back(sched_phase_kind_name(static_cast<SchedPhaseKind>(decoded[i].kind & 0xffffU)));
        EXPECT_EQ(std::count(names.begin(), names.end(), "scalar_claim_seal_probe"), 1);
        EXPECT_EQ(std::count(names.begin(), names.end(), "scalar_claim_seal_confirm"), seal != 0 ? 1 : 0);
        EXPECT_EQ(std::count(names.begin(), names.end(), "scalar_next_block_probe"), seal == 2 ? 1 : 0);
        EXPECT_EQ(std::count(names.begin(), names.end(), "scalar_next_block_confirm"), used_seal ? 0 : 1);
        EXPECT_EQ(buffer->scalar_dropped, 0u);
    }
}

TEST(HbgOrderedScalarDetail, PollDetailKeepsTheReadyProbeShortCircuit) {
    for (bool ready : {false, true}) {
        OrderedFixture fixture({{2, LaneMask::aic()}});
        ASSERT_TRUE(fixture.bootstrap_admission());
        if (ready) ASSERT_TRUE(fixture.enqueue(0, 0));
        auto buffer = std::make_unique<SchedulerActivityBuffer>();
        auto records = std::make_unique<SchedulerScalarStorageSlot[]>(2 * SCHEDULER_SCALAR_RECORD_CAPACITY);
        auto &trace = fixture.locals[0]->profiling->scalar;
        trace.buffer = buffer.get();
        trace.records = records.get();
        trace.window_start = scheduler_cycles();
        EXPECT_EQ(scheduler_ordered_poll_may_progress(fixture.graph.graph(), fixture.base(), fixture.locals[0]), ready);
        EXPECT_EQ(trace.depth, 0u);
        EXPECT_EQ(fixture.control(0)->next_block, 0u);
        trace.finish(scheduler_cycles());
        const auto decoded = scalar_rows(trace);
        std::vector<std::string> names;
        for (uint32_t i = 0; i < trace.committed; ++i)
            names.emplace_back(sched_phase_kind_name(static_cast<SchedPhaseKind>(decoded[i].kind & 0xffffU)));
        EXPECT_EQ(std::count(names.begin(), names.end(), "scalar_poll_local_state"), 1);
        EXPECT_EQ(std::count(names.begin(), names.end(), "scalar_poll_ready_queue"), 1);
        EXPECT_EQ(std::count(names.begin(), names.end(), "scalar_poll_ready_pool"), 1);
        EXPECT_EQ(std::count(names.begin(), names.end(), "scalar_poll_admission_tail"), ready ? 0 : 1);
        EXPECT_EQ(buffer->scalar_dropped, 0u);
    }
}

class HbgOrderedFirstReady : public ::testing::TestWithParam<LaneMask> {};

TEST_P(HbgOrderedFirstReady, FirstBlockIsExecutableWithoutClaimingOrPublishingASecondBlock) {
    OrderedFixture fixture({{8, GetParam()}}, true);
    const uint32_t worker = fixture.publisher(0);
    ASSERT_TRUE(fixture.bootstrap_admission());
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
    ASSERT_TRUE(fixture.bootstrap_admission());
    ASSERT_TRUE(fixture.admit(0, worker));
    ASSERT_TRUE(fixture.stage_one(worker, 1));
    const auto staged = local->slots[1];
    for (uint32_t poll = 0; poll < 8; ++poll) {
        ASSERT_TRUE(fixture.fill(worker));
        EXPECT_EQ(fixture.control(0)->claim_tickets, 1u);
        EXPECT_EQ(fixture.control(0)->next_block, 1u);
        EXPECT_EQ(local->local_ready_mask, 2u);
        EXPECT_EQ(local->slots[0].state, SchedulerDispatchSlotState::FREE);
        EXPECT_EQ(local->slots[1].task_ref, staged.task_ref);
        EXPECT_EQ(local->slots[1].block_idx, staged.block_idx);
        EXPECT_EQ(local->ready_head, 1u);
    }
    OrderedExecution execution{};
    ASSERT_TRUE(fixture.begin(worker, &execution));
    EXPECT_EQ(execution.slot, 1u);
    EXPECT_EQ(execution.block, 0u);
    ASSERT_TRUE(fixture.complete(execution));
    ASSERT_TRUE(fixture.fill(worker));
    EXPECT_EQ(fixture.control(0)->next_block, 2u);
    EXPECT_EQ(scheduler_completion_progress(fixture.control(0)->completion_state), 1u);
    EXPECT_EQ(local->local_ready_mask, 1u);
    EXPECT_EQ(local->slots[0].state, SchedulerDispatchSlotState::READY);
    EXPECT_EQ(local->slots[0].block_idx, 1u);
}

TEST_P(HbgOrderedFirstReady, CompletionAllowsTheNextClaimWithoutLosingOrDuplicatingBlocks) {
    constexpr uint16_t count = 8;
    OrderedFixture fixture({{count, GetParam()}});
    const uint32_t worker = fixture.publisher(0);
    ASSERT_TRUE(fixture.bootstrap_admission());
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
        EXPECT_EQ(scheduler_completion_progress(fixture.control(0)->completion_state), block + 1);
    }
    EXPECT_EQ(scheduler_gm_load(fixture.control(0)->state), static_cast<int64_t>(SchedulerTaskState::DONE));
    ASSERT_TRUE(fixture.fill(worker));
    OrderedExecution extra{};
    EXPECT_FALSE(fixture.begin(worker, &extra));
    EXPECT_EQ(fixture.control(0)->next_block, count);
    EXPECT_EQ(scheduler_completion_progress(fixture.control(0)->completion_state), count);
    EXPECT_EQ(fixture.run()->scheduler_error, 0u);
}

TEST_P(HbgOrderedFirstReady, ReadyBlockDoesNotSkipDiscoveryOrPublicationOfALaterAdmission) {
    OrderedFixture fixture({{8, GetParam()}, {2, GetParam()}}, true);
    const uint32_t worker = fixture.publisher(0);
    ASSERT_TRUE(fixture.bootstrap_admission());
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

class HbgOrderedSharedOrdinaryWork : public ::testing::TestWithParam<LaneMask> {};

TEST_P(HbgOrderedSharedOrdinaryWork, OrdinaryDiscoveryLeavesWorkForAnotherWorker) {
    OrderedFixture fixture({{1, GetParam()}, {1, GetParam()}});
    const uint32_t worker = fixture.publisher(0);
    const uint32_t peer = worker + 3;
    ASSERT_TRUE(fixture.bootstrap_admission());
    ASSERT_TRUE(fixture.enqueue(0, worker));
    ASSERT_TRUE(fixture.enqueue(1, worker));
    ASSERT_TRUE(fixture.fill(worker));
    for (uint32_t poll = 0; poll < 8; ++poll)
        ASSERT_TRUE(fixture.fill(worker));
    ASSERT_TRUE(fixture.fill(peer));
    OrderedExecution first{};
    OrderedExecution second{};
    ASSERT_TRUE(fixture.begin(worker, &first));
    ASSERT_TRUE(fixture.begin(peer, &second));
    EXPECT_NE(first.task, second.task);
    ASSERT_TRUE(fixture.complete(first));
    ASSERT_TRUE(fixture.complete(second));
    EXPECT_EQ(fixture.run()->scheduler_error, 0u);
}

TEST_P(HbgOrderedSharedOrdinaryWork, APreviouslyRetainedCooperativeCandidateStillPublishes) {
    for (uint32_t ordinary_index = 0; ordinary_index < 2; ++ordinary_index) {
        OrderedFixture fixture({{1, GetParam()}, {2, GetParam()}}, true);
        const uint32_t worker = fixture.member(1, fixture.publisher(0));
        auto &candidates = fixture.admissions[worker].candidates;
        ASSERT_TRUE(fixture.bootstrap_admission());
        ASSERT_TRUE(fixture.pop(0, worker, &candidates[ordinary_index].ready));
        ASSERT_TRUE(fixture.pop(1, worker, &candidates[1 - ordinary_index].ready));
        ASSERT_TRUE(fixture.fill(worker));
        EXPECT_EQ(candidates[ordinary_index].ready.task_ref, 0);
        OrderedExecution execution{};
        ASSERT_TRUE(fixture.begin(worker, &execution));
        EXPECT_EQ(execution.task, 1);
        EXPECT_EQ(candidates[ordinary_index].ready.task_ref, 0);
        EXPECT_EQ(fixture.run()->scheduler_error, 0u);
    }
}

INSTANTIATE_TEST_SUITE_P(PureCAndV, HbgOrderedSharedOrdinaryWork, ::testing::ValuesIn(kPureCAndV));

class HbgOrderedExhaustionSafety : public ::testing::TestWithParam<LaneMask> {};

TEST_P(HbgOrderedExhaustionSafety, ZeroWorkMemberClosesWhileRemoteClaimedBlocksRemainLive) {
    const LaneMask mask = GetParam();
    OrderedFixture fixture({{2, mask}}, true);
    const uint32_t owner = fixture.publisher(0);
    const uint32_t zero_worker = mask == LaneMask::aic() ? 3 : 2;
    ASSERT_TRUE(fixture.bootstrap_admission());
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
    EXPECT_EQ(scheduler_completion_progress(fixture.control(0)->completion_state), 0u);
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
    ASSERT_TRUE(fixture.bootstrap_admission());
    ASSERT_TRUE(fixture.admit(0, worker));
    ASSERT_TRUE(fixture.stage_one(worker, 1));
    ASSERT_TRUE(fixture.stage_one(peer, 0));
    OrderedExecution execution{};
    ASSERT_TRUE(fixture.begin(worker, &execution));
    ASSERT_EQ(fixture.locals[worker]->local_ready_mask, 0u);
    ASSERT_EQ(fixture.locals[worker]->slots[1].state, SchedulerDispatchSlotState::EXECUTING);
    ASSERT_TRUE(fixture.fill(worker));
    EXPECT_EQ(fixture.admissions[worker].current_task_ref, 0);
    EXPECT_EQ(fixture.admissions[worker].scan_cursor, 0u);
    EXPECT_EQ(scheduler_completion_progress(fixture.control(0)->completion_state), 0u);
    ASSERT_TRUE(fixture.complete(execution));
    ASSERT_TRUE(fixture.fill(worker));
    EXPECT_EQ(fixture.admissions[worker].current_task_ref, SCHEDULER_TASK_REF_INVALID);
    EXPECT_EQ(fixture.admissions[worker].scan_cursor, 1u);
    EXPECT_EQ(scheduler_completion_progress(fixture.control(0)->completion_state), 1u);
    EXPECT_EQ(scheduler_gm_load(fixture.control(0)->state), static_cast<int64_t>(SchedulerTaskState::BLOCKED));
}

TEST_P(HbgOrderedExhaustionSafety, PositiveHintCannotBypassCursorShapeOrErrorValidation) {
    const LaneMask mask = GetParam();
    for (uint32_t fault = 0; fault < 4; ++fault) {
        SCOPED_TRACE(fault);
        OrderedFixture fixture({{2, mask}});
        const uint32_t worker = fixture.publisher(0);
        ASSERT_TRUE(fixture.bootstrap_admission());
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
        EXPECT_EQ(scheduler_completion_progress(fixture.control(0)->completion_state), 0u);
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
        ASSERT_TRUE(fixture.bootstrap_admission());
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
        EXPECT_EQ(scheduler_completion_progress(fixture.control(0)->completion_state), 0u);
    }
    {
        const uint16_t count = mask == LaneMask::aic() ? 2 : 4;
        OrderedFixture fixture({{count, mask, true}});
        const uint32_t worker = fixture.publisher(0);
        ASSERT_TRUE(fixture.bootstrap_admission());
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
        EXPECT_EQ(scheduler_completion_progress(fixture.control(0)->completion_state), 1u);
        EXPECT_EQ(scheduler_gm_load(fixture.control(0)->state), static_cast<int64_t>(SchedulerTaskState::BLOCKED));
    }
}

INSTANTIATE_TEST_SUITE_P(PureCAndV, HbgOrderedExhaustionSafety, ::testing::ValuesIn(kPureCAndV));

class HbgOrderedSyncStart : public ::testing::TestWithParam<LaneMask> {};

TEST_P(HbgOrderedSyncStart, TwoPendingSlotsAndEarlyReturnDoNotClaimAnotherPosition) {
    const LaneMask mask = GetParam();
    const uint32_t first = mask == LaneMask::aic() ? 0 : 1;
    OrderedFixture fixture({{2, mask, true}});
    ASSERT_TRUE(fixture.bootstrap_admission());
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
    EXPECT_EQ(
        scheduler_completion_progress(scheduler_gm_fetch_add(fixture.control(0)->completion_state, UINT64_C(0))), 1u
    );
    EXPECT_EQ(scheduler_gm_load(fixture.control(0)->state), static_cast<int64_t>(SchedulerTaskState::BLOCKED));
}

TEST_P(HbgOrderedSyncStart, FastMemberEntersNextFullTaskWithoutStealingLateMembersPosition) {
    const LaneMask mask = GetParam();
    const std::vector<uint32_t> members =
        mask == LaneMask::aic() ? std::vector<uint32_t>{0, 3} : std::vector<uint32_t>{1, 2, 4, 5};
    const auto blocks = static_cast<uint16_t>(members.size());
    OrderedFixture fixture({{blocks, mask, true}, {blocks, mask, true}});
    ASSERT_TRUE(fixture.bootstrap_admission());
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
    EXPECT_EQ(
        scheduler_completion_progress(scheduler_gm_fetch_add(fixture.control(0)->completion_state, UINT64_C(0))), 1u
    );
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
    ASSERT_TRUE(fixture.bootstrap_admission());
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
    ASSERT_TRUE(fixture.bootstrap_admission());
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
    EXPECT_EQ(
        scheduler_completion_progress(scheduler_gm_fetch_add(fixture.control(0)->completion_state, UINT64_C(0))), 0u
    );
}

class HbgOrderedBlockIdentity : public ::testing::TestWithParam<std::tuple<LaneMask, uint16_t>> {};

TEST_P(HbgOrderedBlockIdentity, RealDispatchAndCompletionExecuteEveryOriginalBlockExactlyOnce) {
    const auto [mask, blocks] = GetParam();
    OrderedFixture fixture({{blocks, mask}});
    ASSERT_TRUE(fixture.bootstrap_admission());
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
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.control(0)->next_block, UINT64_C(0)), blocks > 1 ? blocks : 0u);
    EXPECT_EQ(
        scheduler_completion_progress(scheduler_gm_fetch_add(fixture.control(0)->completion_state, UINT64_C(0))), blocks
    );
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

TEST(HbgOrderedExecution, PartialCompletionDoesNotReleaseFaninAndLastSubtaskReleasesExactlyOnce) {
    OrderedFixture fixture({{3, LaneMask::aic()}, {1, LaneMask::aic()}});
    fixture.dependency(1, 0);
    ASSERT_EQ(
        scheduler_route_task(fixture.graph.graph(), fixture.base(), fixture.locals[0], fixture.run(), 1),
        SchedulerRouteResult::WAITING
    );
    ASSERT_TRUE(fixture.bootstrap_admission());
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
    EXPECT_EQ(
        scheduler_completion_progress(scheduler_gm_fetch_add(fixture.control(0)->completion_state, UINT64_C(0))), 2u
    );
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
    EXPECT_EQ(
        scheduler_completion_progress(scheduler_gm_fetch_add(fixture.control(0)->completion_state, UINT64_C(0))), 3u
    );
    EXPECT_EQ(scheduler_gm_load(fixture.control(0)->state), static_cast<int64_t>(SchedulerTaskState::DONE));
    EXPECT_EQ(scheduler_gm_load(fixture.control(0)->wake_list_head), SCHEDULER_WAKE_LIST_CLOSED);
    EXPECT_EQ(fixture.locals[0]->pending_completed, 1u);
    EXPECT_EQ(scheduler_completion_count(fixture.control(0)->completion_state), 3u);
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
    EXPECT_FALSE(scheduler_complete_own_slot(
        fixture.graph.graph(), fixture.base(), fixture.locals[0], fixture.run(), successor.slot, 0
    ));
    EXPECT_EQ(fixture.locals[0]->pending_completed, 2u);
    EXPECT_EQ(fixture.run()->error_site, static_cast<uint64_t>(SchedulerErrorSite::COMPLETION_SLOT_BINDING_INVALID));
}

TEST(HbgOrderedExecution, LocalDoneAllowsIndependentBWhileARemoteLastBlockIsStillExecuting) {
    OrderedFixture fixture({{3, LaneMask::aic()}, {2, LaneMask::aic()}});
    ASSERT_TRUE(fixture.bootstrap_admission());
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
    EXPECT_EQ(
        scheduler_completion_progress(scheduler_gm_fetch_add(fixture.control(0)->completion_state, UINT64_C(0))), 2u
    );
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
    EXPECT_EQ(
        scheduler_completion_progress(scheduler_gm_fetch_add(fixture.control(0)->completion_state, UINT64_C(0))), 3u
    );
    EXPECT_EQ(
        scheduler_completion_progress(scheduler_gm_fetch_add(fixture.control(1)->completion_state, UINT64_C(0))), 2u
    );
}

TEST(HbgOrderedExecution, CursorExhaustionIsBoundedAndDoesNotCloseLiveLocalSlots) {
    OrderedFixture fixture({{2, LaneMask::aic()}});
    ASSERT_TRUE(fixture.bootstrap_admission());
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

class HbgOrderedAdmissionProgress : public ::testing::TestWithParam<std::tuple<LaneMask, bool, bool>> {};

TEST_P(HbgOrderedAdmissionProgress, BoundedRetirementReportsProgressAndCaughtUpPollingIsIdle) {
    const auto [mask, record_protocol, pending_task] = GetParam();
    OrderedFixture fixture(std::vector<OrderedShape>(pending_task ? 9 : 8, {2, mask}), record_protocol);
    const uint32_t publisher = fixture.publisher(0);
    const auto first_members = fixture.members(0);
    ASSERT_GE(first_members.size(), 2u);
    const uint32_t late_worker = first_members.back();
    ASSERT_TRUE(fixture.bootstrap_admission());
    for (uint64_t task = 0; task < 8; ++task) {
        ASSERT_TRUE(fixture.admit(task, publisher));
        // Keep the delayed worker idle while an actual member consumes each
        // record. Ordinary cohorts can select different workers across tasks.
        const auto members = fixture.members(task);
        const auto executor = std::find_if(members.begin(), members.end(), [&](uint32_t worker) {
            return worker != late_worker;
        });
        ASSERT_NE(executor, members.end());
        for (uint32_t block = 0; block < 2; ++block) {
            ASSERT_TRUE(fixture.stage_one(*executor, 0));
            OrderedExecution execution{};
            ASSERT_TRUE(fixture.begin(*executor, &execution));
            ASSERT_EQ(execution.task, task);
            ASSERT_EQ(execution.block, block);
            ASSERT_TRUE(fixture.complete(execution));
        }
        bool closed = false;
        ASSERT_TRUE(fixture.close(*executor, &closed));
        ASSERT_TRUE(closed);
    }
    if (pending_task) ASSERT_TRUE(fixture.admit(8, publisher));
    ASSERT_EQ(fixture.admissions[late_worker].scan_cursor, 0u);
    const auto fill_slots = record_protocol ? scheduler_fill_own_slots<true> : scheduler_fill_own_slots<false>;
    auto fill = [&](bool *failed) {
        return fill_slots(fixture.graph.graph(), fixture.base(), fixture.locals[late_worker], fixture.run(), 0, failed);
    };
    bool failed = false;
    EXPECT_TRUE(fill(&failed));
    ASSERT_FALSE(failed);
    EXPECT_EQ(fixture.admissions[late_worker].scan_cursor, 8u);
    EXPECT_EQ(fixture.admissions[late_worker].current_task_ref, SCHEDULER_TASK_REF_INVALID);
    EXPECT_EQ(fixture.locals[late_worker]->local_ready_mask, 0u);
    if (pending_task) {
        EXPECT_EQ(fixture.control(8)->next_block, 0u);
        EXPECT_TRUE(fill(&failed));
        ASSERT_FALSE(failed);
        EXPECT_EQ(fixture.control(8)->next_block, 1u);
        OrderedExecution execution{};
        ASSERT_TRUE(fixture.begin(late_worker, &execution));
        EXPECT_EQ(execution.task, 8);
        EXPECT_EQ(execution.block, 0u);
    } else {
        EXPECT_FALSE(fill(&failed));
        ASSERT_FALSE(failed);
        EXPECT_EQ(fixture.admissions[late_worker].scan_cursor, 8u);
    }
}

INSTANTIATE_TEST_SUITE_P(
    PureCAndV, HbgOrderedAdmissionProgress,
    ::testing::Combine(::testing::ValuesIn(kPureCAndV), ::testing::Bool(), ::testing::Bool())
);

TEST(HbgOrderedExecution, EmptyFillStillStopsOnAnAlreadyPublishedRunError) {
    OrderedFixture fixture({{2, LaneMask::aic()}, {1, LaneMask::aic()}});
    ASSERT_TRUE(fixture.bootstrap_admission());
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
    ASSERT_TRUE(fixture.bootstrap_admission());
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
    EXPECT_EQ(scheduler_completion_progress(fixture.control(0)->completion_state), 0u);
}

TEST(HbgOrderedExecution, IdlePollFindsWorkPublishedByAnotherWorker) {
    for (LaneMask mask : kPureCAndV) {
        SCOPED_TRACE(mask.count());
        OrderedFixture fixture({{2, mask}});
        const uint32_t worker = mask == LaneMask::aic() ? 0 : 1;
        auto *local = fixture.locals[worker];
        ASSERT_TRUE(fixture.bootstrap_admission());
        ASSERT_TRUE(fixture.enqueue(0, worker + 3));
        EXPECT_TRUE(scheduler_ready_pool_may_have_work(fixture.base(), local, local->config.self_core_type));
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
    ASSERT_TRUE(fixture.bootstrap_admission());
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
    EXPECT_EQ(scheduler_completion_progress(fixture.control(1)->completion_state), 2u);
    ASSERT_TRUE(fixture.complete(ordinary));
    EXPECT_FALSE(fixture.begin(0, &ordinary));
}

TEST(HbgOrderedExecution, SecondPreparedOrdinarySlotRechecksAdmissionAfterTheFirstEntered) {
    OrderedFixture fixture({{1, LaneMask::aic()}, {1, LaneMask::aic()}, {2, LaneMask::aic()}}, true);
    const auto *events = scheduler_state_at<SchedulerProtocolExecutionTrace>(
        fixture.base(), fixture.storage.layout.protocol.executions_offset
    );
    // Direct retention can fill both credits even when queue discovery is bounded.
    ASSERT_TRUE(fixture.bootstrap_admission());
    for (uint32_t task = 0; task < 2; ++task) {
        SchedulerReadyClaim claim{};
        ASSERT_TRUE(fixture.pop(task, 0, &claim));
        ASSERT_TRUE(scheduler_ordered_retain_candidate(&fixture.admissions[0], claim));
    }
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
    ASSERT_TRUE(fixture.bootstrap_admission());
    for (uint64_t task = 0; task < 2; ++task) {
        fixture.storage.metadata[task].flags |= SCHEDULER_TASK_HAS_PREDICATE;
        fixture.graph.predicate(
            task, reinterpret_cast<uint64_t>(&predicates[task]), sizeof(uint64_t),
            static_cast<uint8_t>(PredicateOp::EQ), 1
        );
        SchedulerReadyClaim claim{};
        ASSERT_TRUE(fixture.pop(task, 0, &claim));
        ASSERT_TRUE(scheduler_ordered_retain_candidate(&fixture.admissions[0], claim));
    }
    ASSERT_TRUE(fixture.fill(0));
    ASSERT_EQ(fixture.locals[0]->local_ready_mask, 3u);
    for (uint32_t slot = 0; slot < 2; ++slot) {
        EXPECT_FALSE(scheduler_slot_uses_admission(fixture.locals[0], fixture.locals[0]->slots[slot]));
        EXPECT_NE(fixture.locals[0]->slots[slot].predicate, SchedulerPredicateState::UNEVALUATED);
        EXPECT_EQ(events[slot].publication_count, 1u);
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
        EXPECT_EQ(
            fixture.admissions[0].candidates[candidate].predicate,
            candidate == 0 ? SchedulerPredicateState::PASS : SchedulerPredicateState::FAIL
        );
    }
    uint32_t ready_cooperative = 0;
    for (const auto &slot : fixture.locals[0]->slots) {
        if (slot.state == SchedulerDispatchSlotState::READY) {
            ++ready_cooperative;
            EXPECT_TRUE(scheduler_slot_uses_admission(fixture.locals[0], slot));
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
        EXPECT_EQ(events[ordinary.task].publication_count, 2u);
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

    const auto *protocol_workers = scheduler_state_at<SchedulerProtocolWorkerTrace>(
        fixture.base(), fixture.storage.layout.protocol.workers_offset
    );
    EXPECT_EQ(protocol_workers[0].error_flags, 0u);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.run()->scheduler_error, UINT64_C(0)), 0u);
}

TEST(HbgOrderedExecution, NonmemberVectorCanExecuteOrdinaryWorkWhileCubeObligationIsLive) {
    OrderedFixture fixture({{2, LaneMask::aic()}, {1, LaneMask::aiv0()}});
    ASSERT_TRUE(fixture.bootstrap_admission());
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
    EXPECT_EQ(
        scheduler_completion_progress(scheduler_gm_fetch_add(fixture.control(0)->completion_state, UINT64_C(0))), 0u
    );
    ASSERT_TRUE(fixture.complete(vector));
}

TEST(HbgOrderedExecution, NonmemberCubeCanExecuteOrdinaryWorkWhileVectorObligationIsLive) {
    OrderedFixture fixture({{2, LaneMask::aiv0()}, {1, LaneMask::aic()}});
    ASSERT_TRUE(fixture.bootstrap_admission());
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
    EXPECT_EQ(
        scheduler_completion_progress(scheduler_gm_fetch_add(fixture.control(0)->completion_state, UINT64_C(0))), 0u
    );
    ASSERT_TRUE(fixture.complete(cube));
}

TEST(HbgOrderedExecution, PendingLaterAdmissionDoesNotPreemptTheCurrentCooperativeKernel) {
    OrderedFixture fixture({{2, LaneMask::aic()}, {2, LaneMask::aic()}});
    ASSERT_TRUE(fixture.bootstrap_admission());
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
    OrderedExecution first{};
    ASSERT_TRUE(fixture.begin(0, &first));
    EXPECT_EQ(first.task, 0);
    EXPECT_EQ(first.block, 0u);
    EXPECT_EQ(fixture.admissions[0].pending_task_ref, 1);
    EXPECT_EQ(fixture.control(1)->admission_state, static_cast<uint64_t>(SchedulerAdmissionState::PREPARED));
    EXPECT_EQ(fixture.control(1)->next_block, 0u);
    EXPECT_EQ(fixture.locals[0]->local_ready_mask, 0u);
    ASSERT_TRUE(fixture.complete(first));
    ASSERT_TRUE(scheduler_prepare_admission_record(
        fixture.graph.graph(), fixture.base(), fixture.locals[0], fixture.run(), &fixture.admissions[0]
    ));
    ASSERT_EQ(
        scheduler_try_commit_admission(
            fixture.graph.graph(), fixture.base(), fixture.locals[0], fixture.run(), &fixture.admissions[0]
        ),
        SchedulerAdmissionResult::PROGRESS
    );
    ASSERT_TRUE(fixture.stage_one(0, 0));
    OrderedExecution second{};
    ASSERT_TRUE(fixture.begin(0, &second));
    EXPECT_EQ(second.task, 0);
    EXPECT_EQ(second.block, 1u);
    EXPECT_EQ(fixture.control(1)->next_block, 0u);
    ASSERT_TRUE(fixture.complete(second));
}

struct OrderedPendingEntryCase {
    uint32_t worker;
    uint16_t blocks;
    bool sync;
    bool prepared;
    bool protocol;
    bool reversed_task_ids{false};
};

class HbgOrderedPendingEarlierEntry : public ::testing::TestWithParam<OrderedPendingEntryCase> {};

TEST_P(HbgOrderedPendingEarlierEntry, CurrentBlockRetiresWhileLaterPublicationRemainsPending) {
    auto scenario = GetParam();
    const LaneMask mask = scenario.worker % 3 == 0 ? LaneMask::aic() : LaneMask::aiv0();
    const uint32_t current_task = scenario.reversed_task_ids ? 1 : 0;
    const uint32_t later_task = current_task ^ 1U;
    std::vector<OrderedShape> shapes{{2, mask}, {2, mask}, {2, mask}};
    shapes[current_task] = {scenario.blocks, mask, scenario.sync};
    OrderedFixture fixture(shapes, scenario.protocol);
    scenario.worker = fixture.member(current_task, scenario.worker);
    if (!scenario.protocol)
        for (auto *local : fixture.locals)
            local->profiling = nullptr;
    ASSERT_TRUE(fixture.bootstrap_admission());
    ASSERT_TRUE(fixture.admit(current_task, scenario.worker));
    ASSERT_TRUE(fixture.stage_one(scenario.worker, 0));
    const auto claim = [&](uint32_t task, uint32_t worker) {
        SchedulerReadyClaim ready{};
        return fixture.pop(task, worker, &ready) && scheduler_try_claim_admission(
                                                        fixture.graph.graph(), fixture.base(), fixture.locals[worker],
                                                        fixture.run(), &fixture.admissions[worker], ready
                                                    ) == SchedulerAdmissionResult::PROGRESS;
    };
    ASSERT_TRUE(claim(later_task, scenario.worker));
    auto *shared =
        scheduler_state_at<SchedulerAdmissionControl>(fixture.base(), fixture.storage.layout.admission_control_offset);
    const uint32_t peer = (scenario.worker + 3) % 6;
    if (scenario.prepared) {
        ASSERT_TRUE(scheduler_prepare_admission_record(
            fixture.graph.graph(), fixture.base(), fixture.locals[scenario.worker], fixture.run(),
            &fixture.admissions[scenario.worker]
        ));
        ASSERT_TRUE(claim(2, peer));
        ASSERT_TRUE(scheduler_prepare_admission_record(
            fixture.graph.graph(), fixture.base(), fixture.locals[peer], fixture.run(), &fixture.admissions[peer]
        ));
        ASSERT_EQ(scheduler_gm_compare_exchange(shared->commit_lock, UINT64_C(0), static_cast<uint64_t>(peer + 1)), 0u);
        ASSERT_EQ(
            scheduler_try_commit_admission(
                fixture.graph.graph(), fixture.base(), fixture.locals[scenario.worker], fixture.run(),
                &fixture.admissions[scenario.worker]
            ),
            SchedulerAdmissionResult::RETRY
        );
    }
    OrderedExecution execution{};
    ASSERT_TRUE(fixture.begin(scenario.worker, &execution));
    EXPECT_EQ(execution.task, current_task);
    EXPECT_EQ(execution.block, 0u);
    EXPECT_EQ(execution.blocks, scenario.blocks);
    ASSERT_TRUE(fixture.complete(execution));
    EXPECT_EQ(scheduler_completion_progress(fixture.control(current_task)->completion_state), 1u);
    EXPECT_EQ(fixture.control(later_task)->next_block, 0u);
    EXPECT_EQ(fixture.control(2)->next_block, 0u);
    EXPECT_EQ(shared->commit_tail, 1u);
    EXPECT_EQ(shared->commit_lock, scenario.prepared ? peer + 1u : 0u);
    EXPECT_EQ(fixture.admissions[scenario.worker].pending_task_ref, later_task);
    EXPECT_EQ(fixture.run()->scheduler_error, 0u);
}

std::vector<OrderedPendingEntryCase> ordered_pending_entry_cases() {
    std::vector<OrderedPendingEntryCase> cases;
    for (uint32_t worker = 0; worker < 6; ++worker)
        for (uint16_t blocks : {2, 65})
            for (bool sync : {false, true}) {
                if (sync && blocks != 2) continue;
                for (bool prepared : {false, true})
                    for (bool protocol : {false, true})
                        cases.push_back({worker, blocks, sync, prepared, protocol});
            }
    for (uint32_t worker = 0; worker < 6; ++worker)
        for (bool protocol : {false, true})
            cases.push_back({worker, 2, false, true, protocol, true});
    return cases;
}

INSTANTIATE_TEST_SUITE_P(
    CooperativeTypesAndPendingPhases, HbgOrderedPendingEarlierEntry, ::testing::ValuesIn(ordered_pending_entry_cases())
);

TEST(HbgOrderedExecution, PersistentCandidateCreditsRejectDuplicatesAndDoNotOverwriteFullPool) {
    OrderedFixture fixture({{1, LaneMask::aic()}, {1, LaneMask::aic()}, {1, LaneMask::aic()}});
    std::array<SchedulerReadyClaim, 3> claims{};
    ASSERT_TRUE(fixture.bootstrap_admission());
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

    const uint32_t publisher = fixture.publisher(0);
    ASSERT_EQ(
        scheduler_route_task(fixture.graph.graph(), fixture.base(), fixture.locals[publisher], fixture.run(), 1),
        SchedulerRouteResult::WAITING
    );
    ASSERT_TRUE(fixture.bootstrap_admission());
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
                EXPECT_EQ(
                    scheduler_completion_progress(
                        scheduler_gm_fetch_add(fixture.control(0)->completion_state, UINT64_C(0))
                    ),
                    3u
                );
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
    ASSERT_TRUE(fixture.bootstrap_admission());
    ASSERT_TRUE(fixture.enqueue(1, worker));
    ASSERT_TRUE(fixture.fill(worker, SCHEDULER_PROFILING_SCHED_PHASES_LEVEL));
    const auto *traces =
        scheduler_state_at<SchedulerTaskTrace>(fixture.base(), fixture.storage.layout.trace_cells_offset);
    const auto &trace = traces[fixture.storage.metadata[1].trace_index_base];
    EXPECT_EQ(trace.task_ref, 1u);
    EXPECT_EQ(trace.worker_id, worker);
    EXPECT_EQ(trace.ready_source, static_cast<uint64_t>(SchedulerReadySource::POOL));
    EXPECT_EQ(trace.state_probe_scheduler_worker_id, worker);
    EXPECT_GT(trace.state_probe_start_cycles, 0u);
    EXPECT_GE(trace.state_probe_end_cycles, trace.state_probe_start_cycles);
    EXPECT_GT(trace.dispatch_start_cycles, 0u);
    EXPECT_EQ(
        scheduler_gm_query(fixture.control(0)->admission_state),
        static_cast<uint64_t>(SchedulerAdmissionState::INITIALIZED)
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
    ASSERT_TRUE(fixture.bootstrap_admission());
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
        scheduler_route_task(fixture.graph.graph(), fixture.base(), local, fixture.run(), 1),
        SchedulerRouteResult::WAITING
    );
    ASSERT_TRUE(fixture.bootstrap_admission());
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
        ASSERT_TRUE(fixture.bootstrap_admission());
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
        EXPECT_EQ(
            scheduler_completion_progress(scheduler_gm_fetch_add(fixture.control(0)->completion_state, UINT64_C(0))), 2u
        );
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
        ASSERT_TRUE(fixture.bootstrap_admission());
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
            EXPECT_EQ(scheduler_completion_progress(fixture.control(0)->completion_state), 0u);
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
    ASSERT_TRUE(fixture.bootstrap_admission());
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
        EXPECT_EQ(scheduler_completion_progress(fixture.control(0)->completion_state), 0u);
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
        const uint16_t later_blocks = mask.has_aic() ? 2 : 4;
        OrderedFixture fixture({{static_cast<uint16_t>(count), mask}, {later_blocks, mask}}, true);
        const uint32_t worker = fixture.publisher(0);
        ASSERT_TRUE(fixture.bootstrap_admission());
        ASSERT_TRUE(fixture.admit(0, worker));
        ASSERT_TRUE(fixture.admit(1, worker));
        ASSERT_TRUE(scheduler_refresh_local_obligation(
            fixture.graph.graph(), fixture.base(), fixture.locals[worker], fixture.run(), &fixture.admissions[worker]
        ));
        // The last ticket's owner is between reservation and valid-claim publication.
        scheduler_gm_store(fixture.control(0)->claim_tickets, count);
        scheduler_gm_store(fixture.control(0)->next_block, count - 1);
        scheduler_gm_store(
            fixture.control(0)->completion_state,
            (fixture.control(0)->completion_state & SCHEDULER_COMPLETION_TOTAL_MASK) | (count - 1)
        );
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
        EXPECT_EQ(scheduler_completion_progress(fixture.control(0)->completion_state), count - 1);
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
        ASSERT_TRUE(fixture.bootstrap_admission());
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
        EXPECT_EQ(scheduler_completion_progress(fixture.control(0)->completion_state), 0u);
        EXPECT_NE(scheduler_read_error(fixture.run()), 0u);
    }
}

TEST_P(HbgOrderedTicketCursor, InconsistentReservationAndConfirmationStopBeforeReturningABlock) {
    const uint64_t positions = GetParam() == LaneMask::aic() ? 2 : 4;
    OrderedFixture fixture({{17, GetParam()}}, true);
    const uint32_t worker = fixture.publisher(0);
    ASSERT_TRUE(fixture.bootstrap_admission());
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
    EXPECT_EQ(scheduler_completion_progress(fixture.control(0)->completion_state), 0u);
    EXPECT_EQ(trace[worker].next_sequence, sequence);
    EXPECT_NE(scheduler_read_error(fixture.run()), 0u);
    for (const auto &slot : fixture.locals[worker]->slots)
        EXPECT_EQ(slot.state, SchedulerDispatchSlotState::FREE);
}

TEST_P(HbgOrderedTicketCursor, ConcurrentTailLosersStopWithoutHidingTheUnpublishedValidTicket) {
    const LaneMask mask = GetParam();
    const std::vector<uint32_t> eligible =
        mask == LaneMask::aic() ? std::vector<uint32_t>{0, 3} : std::vector<uint32_t>{1, 2, 4, 5};
    for (uint16_t count : {uint16_t{2}, uint16_t{3}, static_cast<uint16_t>(eligible.size() * 2 + 1)}) {
        OrderedFixture fixture({{count, mask}});
        const auto workers = fixture.members(0);
        ASSERT_TRUE(fixture.bootstrap_admission());
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
    ASSERT_TRUE(fixture.bootstrap_admission());
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

TEST_P(HbgOrderedTicketCursor, StaleTicketsRejectBootstrapBeforeReadyOwnership) {
    OrderedFixture fixture({{17, GetParam()}});
    scheduler_gm_store(fixture.control(0)->claim_tickets, UINT64_C(1));
    EXPECT_FALSE(fixture.bootstrap_admission());
    const auto *admission =
        scheduler_state_at<SchedulerAdmissionControl>(fixture.base(), fixture.storage.layout.admission_control_offset);
    EXPECT_EQ(admission->commit_tail, 0u);
    EXPECT_EQ(admission->commit_lock, 0u);
    EXPECT_EQ(fixture.control(0)->admission_state, static_cast<uint64_t>(SchedulerAdmissionState::UNCLAIMED));
    EXPECT_EQ(fixture.control(0)->next_block, 0u);
    EXPECT_EQ(fixture.run()->bootstrap_scan_complete, 0u);
    EXPECT_NE(scheduler_read_error(fixture.run()), 0u);
}

TEST_P(HbgOrderedTicketCursor, ExactFullResidencySyncClaimsThroughTheFetchAdd) {
    const LaneMask mask = GetParam();
    const uint16_t positions = mask == LaneMask::aic() ? 2 : 4;
    OrderedFixture fixture({{positions, mask, true}});
    const uint32_t worker = fixture.publisher(0);
    ASSERT_TRUE(fixture.bootstrap_admission());
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
    ASSERT_TRUE(fixture.bootstrap_admission());
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

TEST(HbgOrderedProtocol, EachBlockContributesBeforePublishingItsTrace) {
    OrderedFixture fixture({{2, LaneMask::aic()}}, true);
    ASSERT_TRUE(fixture.bootstrap_admission());
    ASSERT_TRUE(fixture.admit(0, 0));
    for (const uint32_t worker : {0u, 3u}) {
        ASSERT_TRUE(fixture.stage_one(worker, 0));
        OrderedExecution execution;
        ASSERT_TRUE(fixture.begin(worker, &execution));
        auto *local = fixture.locals[worker];
        ASSERT_TRUE(
            scheduler_complete_own_slot(fixture.graph.graph(), fixture.base(), local, fixture.run(), execution.slot, 4)
        );
        const auto *blocks = scheduler_state_at<SchedulerProtocolBlockTrace>(
            fixture.base(), fixture.storage.layout.protocol.blocks_offset
        );
        const auto *executions = scheduler_state_at<SchedulerProtocolExecutionTrace>(
            fixture.base(), fixture.storage.layout.protocol.executions_offset
        );
        EXPECT_NE(executions[execution.block].trace_published.sequence, 0u);
        EXPECT_LT(
            blocks[execution.block].owner.completion_after.sequence,
            executions[execution.block].trace_published.sequence
        );
        EXPECT_EQ(local->slots[execution.slot].state, SchedulerDispatchSlotState::FREE);
    }
    EXPECT_EQ(scheduler_completion_progress(fixture.control(0)->completion_state), 2u);
}

TEST(HbgOrderedProtocol, OutOfRangeTracePrefixDoesNotWriteOutsideThePlannedBlockRegion) {
    OrderedFixture fixture({{2, LaneMask::aic()}}, true);
    ASSERT_TRUE(fixture.bootstrap_admission());
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
    OrderedFixture fixture({{count, mask}}, true);
    const auto participants = fixture.members(0);
    ASSERT_TRUE(fixture.bootstrap_admission());
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
    std::vector<bool> confirmations(count, false);
    uint32_t seal_publishers = 0;
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
            const auto &certificate = records[block].claim;
            EXPECT_EQ(certificate.worker_index, owner.worker_index);
            EXPECT_EQ(certificate.block_idx, block);
            EXPECT_EQ(certificate.run_epoch, owner.run_epoch);
            ASSERT_GE(certificate.confirmed_count, 1u);
            ASSERT_LE(certificate.confirmed_count, count);
            EXPECT_FALSE(confirmations[certificate.confirmed_count - 1]);
            confirmations[certificate.confirmed_count - 1] = true;
            if (certificate.confirmed_count == count) {
                ++seal_publishers;
                EXPECT_EQ(certificate.seal_value, count);
                EXPECT_EQ(certificate.published_before.sequence, ++sequence);
                EXPECT_EQ(certificate.published_after.sequence, ++sequence);
            } else {
                EXPECT_EQ(certificate.seal_value, 0u);
                EXPECT_EQ(certificate.published_before.sequence, 0u);
                EXPECT_EQ(certificate.published_after.sequence, 0u);
            }
        }
        EXPECT_EQ(workers[participants[participant]].next_sequence, sequence);
        EXPECT_EQ(workers[participants[participant]].error_flags, 0u);
    }
    EXPECT_EQ(seal_publishers, 1u);
    EXPECT_EQ(fixture.control(0)->claim_seal, count);
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
    ASSERT_TRUE(fixture.bootstrap_admission());
    ASSERT_TRUE(fixture.enqueue(0, 0));
    ASSERT_TRUE(fixture.fill(0));
    auto *local = fixture.locals[0];
    uint32_t slot = UINT32_MAX;
    ASSERT_TRUE(scheduler_local_ready_peek(local, &slot));
    bool permit = false;
    ASSERT_TRUE(
        scheduler_ordered_can_enter_slot(fixture.graph.graph(), fixture.base(), local, fixture.run(), slot, &permit)
    );
    ASSERT_TRUE(permit);
    ASSERT_TRUE(scheduler_local_ready_execute(local, slot));
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
    OrderedExecution execution{0, 0, 1, 0, slot, payload->function_bin_addr};
    ASSERT_TRUE(fixture.complete(execution));
    EXPECT_EQ(
        scheduler_completion_progress(scheduler_gm_fetch_add(fixture.control(1)->completion_state, UINT64_C(0))), 0u
    );
    const auto *worker = scheduler_state_at<SchedulerProtocolWorkerTrace>(
        fixture.base(), fixture.storage.layout.protocol.workers_offset
    );
    EXPECT_EQ(worker[0].error_flags, 0u);
}

TEST(HbgOrderedProtocol, DuplicateReturnKeepsTheFirstReturnAndVisibilityPoints) {
    OrderedFixture fixture({{2, LaneMask::aic()}}, true);
    ASSERT_TRUE(fixture.bootstrap_admission());
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
    ASSERT_TRUE(fixture.bootstrap_admission());
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
    SchedulerFreeSlotClaim claim{execution.slot};
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
    SchedulerFreeSlotClaim claim{0};
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
    ASSERT_TRUE(fixture.bootstrap_admission());
    ASSERT_TRUE(fixture.admit(0, 0));
    ASSERT_TRUE(fixture.stage_one(0, 0));
    const auto *events = scheduler_state_at<SchedulerProtocolExecutionTrace>(
        fixture.base(), fixture.storage.layout.protocol.executions_offset
    );
    ASSERT_EQ(events[0].permit.sequence, 0u);
    const auto first = events[0];
    SchedulerReadyClaim ready{};
    ready.task_ref = 0;
    SchedulerFreeSlotClaim claim{0};
    scheduler_protocol_payload_publication(fixture.base(), fixture.locals[0], claim, ready, true);
    EXPECT_EQ(events[0].publication_count, first.publication_count);
    EXPECT_EQ(events[0].pending_slot, first.pending_slot);
    const auto *workers = scheduler_state_at<SchedulerProtocolWorkerTrace>(
        fixture.base(), fixture.storage.layout.protocol.workers_offset
    );
    EXPECT_EQ(workers[0].error_flags, SCHEDULER_PROTOCOL_DUPLICATE);
}

}  // namespace
