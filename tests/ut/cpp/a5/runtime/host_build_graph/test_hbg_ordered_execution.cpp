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

#include "scheduler/scheduler_scalar_policy.h"

namespace {
template <>
struct SchedulerScalarCompilation<void> {
    static constexpr bool enabled = true;
};
}  // namespace

#include "inner_kernel.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include "hbg_scheduler_test_support.h"
#include "hbg_gm_mix_test_support.h"
#include "scheduler/scheduler_dispatch.h"
#include "scheduler/scheduler_graph_control.h"
#include "scheduler/scheduler_protocol_execute.h"
#include "test_hbg_owned_execution.h"

namespace {

struct OrderedShape {
    uint16_t blocks;
    LaneMask mask;
    bool sync_start{true};
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
        if (shape.blocks > 1 && shape.sync_start) ++requirements.cooperative_task_count;
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
            if (shape.blocks > 1 && shape.sync_start) metadata.flags |= SCHEDULER_TASK_SYNC_START;
            metadata.trace_index_base = trace;
            trace += shape.blocks;
            metadata.timing_slot = -1;
            for (uint32_t slot = 0; slot < 3; ++slot)
                metadata.kernel_ids[slot] = shape.mask.has(static_cast<SubtaskSlot>(slot)) ? 1 : UINT16_MAX;
        }
        storage.callable_addresses[1] = reinterpret_cast<uint64_t>(&scheduler_test::protocol_noop_kernel);
        storage.prepare_catalog();
        EXPECT_TRUE(scheduler_init_spmd_task_records(base(), storage.layout, 37));
        storage.run_control->active_worker_count = 6;
        for (uint32_t worker = 0; worker < 6; ++worker) {
            auto &context = storage.contexts[worker];
            for (uint32_t lane = 0; lane < 3; ++lane)
                context.cluster_worker_ids[lane] = (worker / 3) * 3 + lane;
            storage.test_graphs[worker] = graph.graph();
        }
        for (uint32_t worker = 0; worker < 6; ++worker) {
            locals[worker] = storage.local_context(&storage.contexts[worker]);
            locals[worker]->run_epoch = 37;
            if (protocol_profiling) EXPECT_TRUE(scheduler_protocol_bind(base(), locals[worker]));
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
    }
    bool prepare() { return storage.prepare_calls(graph.graph()); }
    bool stage_pending_token(uint32_t worker, uint64_t profiling = 0) {
        bool progress = false;
        uint32_t direct = UINT32_MAX;
        if (!scheduler_progress_execution_tokens(
                graph.graph(), base(), locals[worker], run(), profiling, &direct, &progress
            ) ||
            direct == UINT32_MAX)
            return false;
        return true;
    }
    bool fill(uint32_t worker, uint64_t profiling = 0) {
        bool failed = false;
        scheduler_find_local_execution(graph.graph(), base(), locals[worker], run(), profiling, &failed);
        return !failed;
    }
    bool fill_until_ready(uint32_t worker, uint64_t profiling = 0) {
        for (uint64_t visit = 0; visit < graph.graph().task_count + SCHEDULER_EXECUTION_TOKEN_COUNT; ++visit) {
            if (hbg_test_eligible_mask(locals[worker]) != 0) return true;
            if (!fill(worker, profiling)) return false;
        }
        return hbg_test_eligible_mask(locals[worker]) != 0;
    }
    bool prefetch(uint64_t task, uint32_t worker, uint64_t profiling = 0) {
        if (!prepare()) return false;
        auto *local = locals[worker];
        const auto &pool = *scheduler_state_at<SchedulerTaskCatalogHeader>(base(), local->config.task_catalog_offset);
        const auto catalog = scheduler_task_catalog(base(), pool, local->config.self_core_type);
        for (uint64_t index = 0; index < catalog.count; ++index) {
            if (catalog.refs[index] != static_cast<int64_t>(task)) continue;
            for (uint64_t visit = 0; visit <= catalog.count; ++visit) {
                bool progress = false;
                if (!scheduler_refill_execution_tokens(graph.graph(), base(), local, run(), profiling, &progress))
                    return false;
                if (scheduler_execution_token_has_task(local, task)) return true;
                if (!progress) return false;
            }
            return false;
        }
        return false;
    }
    std::vector<uint32_t> members(uint64_t task) {
        const auto &metadata = storage.metadata[task];
        const auto mask = metadata.active_mask;
        const auto resources = scheduler_task_claim_lanes(metadata.flags, mask).merged_with(mask);
        std::vector<uint32_t> result;
        for (uint32_t worker = 0; worker < locals.size(); ++worker) {
            const auto &config = locals[worker]->config;
            const bool member = mask.is_mix() ? resources.has(static_cast<SubtaskSlot>(config.self_lane)) :
                                                config.self_core_type == (mask == LaneMask::aic() ? 0u : 1u);
            if (member) result.push_back(worker);
        }
        return result;
    }
    uint32_t member(uint64_t task, uint32_t preferred) {
        const auto workers = members(task);
        EXPECT_FALSE(workers.empty());
        if (workers.empty()) return preferred;
        return std::find(workers.begin(), workers.end(), preferred) != workers.end() ? preferred : workers.front();
    }
    bool stage_one(uint32_t worker, uint32_t index) {
        uint32_t selected = UINT32_MAX;
        bool failed = false;
        for (uint64_t n = 0; n <= graph.graph().task_count; ++n) {
            scheduler_find_local_execution(graph.graph(), base(), locals[worker], run(), 0, &failed, &selected);
            if (failed || selected != UINT32_MAX) break;
        }
        return !failed && selected == index;
    }
    bool begin(uint32_t worker, OrderedExecution *execution) {
        auto *local = locals[worker];
        uint32_t slot = UINT32_MAX;
        if (!hbg_test_find_eligible(local, &slot)) return false;
        bool permit = false;
        if (!scheduler_check_entry_permit(graph.graph(), base(), local, run(), slot, &permit) || !permit) return false;
        if (!hbg_test_begin_execution(local, slot)) return false;
        const auto index = local->execution_at(slot).prepared.execution_index;
        auto *call = scheduler_prepared_call_at(base(), local->config.prepared, index);
        auto *binding = scheduler_execution_binding_at(base(), local->config.prepared, index);
        const uint64_t function =
            local->execution_at(slot).predicate == SchedulerPredicateState::FAIL ? 0 : call->function_bin_addr;
        *execution = {
            local->execution_at(slot).task_ref,
            static_cast<uint32_t>(binding->local_context.block_idx),
            static_cast<uint32_t>(binding->local_context.block_num),
            worker,
            slot,
            function
        };
        scheduler_execute_protocol_task(base(), local, slot, function, call->args, binding->local_context);
        return true;
    }
    bool complete(const OrderedExecution &execution) {
        auto *local = locals[execution.worker];
        return scheduler_complete_lane_execution(graph.graph(), base(), local, run(), execution.slot, 0);
    }
    scheduler_test::FixtureStorage storage;
    scheduler_test::GraphBuffer graph;
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

class HbgOrderedBlockIdentity : public ::testing::TestWithParam<std::tuple<LaneMask, uint16_t>> {};
class HbgOrderedCooperativeProbe : public ::testing::TestWithParam<LaneMask> {};
class HbgOrderedFirstReady : public ::testing::TestWithParam<LaneMask> {};
class HbgOrderedSharedOrdinaryWork : public ::testing::TestWithParam<LaneMask> {};
class HbgOrderedWithoutSsbuf : public ::testing::TestWithParam<LaneMask> {};
class HbgOrdinaryTokenProbe : public ::testing::TestWithParam<std::tuple<LaneMask, LaneMask>> {};

TEST(HbgOrderedControl, NotifyBeforeAccountingStillCountsExactlyOnceAndRejectsInvalidGate) {
    using scheduler_test::gm_mix::GmMixFixture;
    using scheduler_test::gm_mix::GmShape;
    for (bool invalid : {false, true}) {
        GmShape shape{1, LaneMask::aic()};
        shape.inline_task = true;
        GmMixFixture fixture({shape}, false, 1);
        uint32_t notification = 0;
        scheduler_state_at<uint64_t>(fixture.base(), fixture.storage.layout.done_notifications_offset)[0] =
            reinterpret_cast<uint64_t>(&notification);
        if (invalid) fixture.control(0)->completion_state = UINT64_MAX;
        bool progress = false;
        const bool succeeded = scheduler_progress_graph_controls(
            fixture.graph.graph(), fixture.base(), fixture.locals[0], fixture.run(), 0, &progress
        );
        EXPECT_EQ(succeeded, !invalid);
        EXPECT_EQ(notification, invalid ? 0u : 1u);
        EXPECT_EQ(scheduler_test::resolved_count(fixture.run()), invalid ? 0u : 1u);
        if (invalid) continue;
        ASSERT_TRUE(scheduler_progress_graph_controls(
            fixture.graph.graph(), fixture.base(), fixture.locals[1], fixture.run(), 0, &progress
        ));
        EXPECT_EQ(scheduler_test::resolved_count(fixture.run()), 1u);
        EXPECT_TRUE(scheduler_completion_done(fixture.control(0)->completion_state));
    }
}

TEST(HbgOrderedControl, ConfirmedFaninPrefixSurvivesProofCacheCollisionAndOtherScannedNodes) {
    using scheduler_test::gm_mix::GmMixFixture;
    using scheduler_test::gm_mix::GmShape;
    std::vector<GmShape> shapes(18, {1, LaneMask::aic()});
    shapes[2].inline_task = true;
    shapes[3].inline_task = true;
    GmMixFixture fixture(shapes, false, 1);
    auto &fanin = fixture.graph.payload(2);
    fanin.fanin_count = 2;
    fanin.fanin_data()[0] = 0;
    fanin.fanin_data()[1] = 1;
    auto *local = fixture.locals[0];
    fixture.control(0)->completion_state = scheduler_completion_completed_state(1);
    fixture.control(3)->completion_state = scheduler_completion_completed_state(1);
    fixture.control(16)->completion_state = scheduler_completion_completed_state(1);
    bool progress = false;
    ASSERT_TRUE(
        scheduler_progress_graph_controls(fixture.graph.graph(), fixture.base(), local, fixture.run(), 0, &progress)
    );
    EXPECT_FALSE(progress);
    ASSERT_EQ(scheduler_cached_task_proof(local, 0), SchedulerTaskProof::DONE);
    ASSERT_EQ(scheduler_observe_cached_completion(fixture.base(), local, 16), SchedulerCompletionObservation::DONE);
    ASSERT_EQ(scheduler_cached_task_proof(local, 0), SchedulerTaskProof::NONE);

    auto buffer = std::make_unique<SchedulerActivityBuffer>();
    auto records = std::make_unique<SchedulerScalarStorageSlot[]>(2 * SCHEDULER_SCALAR_RECORD_CAPACITY);
    auto &trace = local->profiling->scalar;
    trace.buffer = buffer.get();
    trace.records = records.get();
    trace.window_start = scheduler_cycles();
    const uint32_t cursor = local->control_peer_cursor;
    {
        SchedulerScalarBinding binding;
        binding.attach(&trace);
        for (uint32_t poll = 0; poll < 9; ++poll) {
            // Task 16's already confirmed DONE collides with task 0's cache entry.
            scheduler_remember_task_proof(local, 16, SchedulerTaskProof::DONE);
            ASSERT_TRUE(scheduler_progress_graph_controls(
                fixture.graph.graph(), fixture.base(), local, fixture.run(), 0, &progress
            ));
            EXPECT_FALSE(progress);
        }
    }
    trace.finish(scheduler_cycles());
    const auto rows = scalar_rows(trace);
    EXPECT_EQ(
        std::count_if(
            rows.begin(), rows.end(),
            [](const auto &row) {
                return static_cast<SchedPhaseKind>(row.kind & 0xffffU) == SchedPhaseKind::ScalarAtomicFetchAdd;
            }
        ),
        0
    );
    EXPECT_EQ(buffer->scalar_dropped, 0u);
    trace.buffer = nullptr;
    EXPECT_EQ(local->control_peer_cursor, cursor + 9 * SCHEDULER_CONTROL_NODE_BUDGET);
    EXPECT_FALSE(scheduler_completion_done(fixture.control(2)->completion_state));
    EXPECT_EQ(scheduler_test::resolved_count(fixture.run()), 0u);
    fixture.control(1)->completion_state = scheduler_completion_completed_state(1);
    for (uint32_t slice = 0; slice < 3; ++slice)
        ASSERT_TRUE(
            scheduler_progress_graph_controls(fixture.graph.graph(), fixture.base(), local, fixture.run(), 0, &progress)
        );
    EXPECT_TRUE(scheduler_completion_done(fixture.control(2)->completion_state));
    EXPECT_EQ(scheduler_test::resolved_count(fixture.run()), 1u);
    EXPECT_EQ(fixture.run()->scheduler_error, 0u);
}

TEST(HbgOrderedControl, InterleavedControlsKeepPrefixesAndCollisionsOnlyForgetProgress) {
    using scheduler_test::gm_mix::GmMixFixture;
    using scheduler_test::gm_mix::GmShape;
    std::vector<GmShape> shapes(20, {1, LaneMask::aic()});
    for (uint32_t task : {2u, 3u, 5u, 18u})
        shapes[task].inline_task = true;
    GmMixFixture fixture(shapes, false, 1);
    for (uint32_t task : {2u, 3u}) {
        fixture.graph.payload(task).fanin_count = 2;
        fixture.graph.payload(task).fanin_data()[0] = 0;
        fixture.graph.payload(task).fanin_data()[1] = 1;
    }
    fixture.graph.payload(5).fanin_count = 1;
    fixture.graph.payload(5).fanin_data()[0] = 0;
    fixture.graph.payload(18).fanin_count = 1;
    fixture.graph.payload(18).fanin_data()[0] = 17;
    fixture.control(0)->completion_state = scheduler_completion_completed_state(1);
    fixture.control(16)->completion_state = scheduler_completion_completed_state(1);
    auto *local = fixture.locals[0];
    bool progress = false;
    ASSERT_TRUE(
        scheduler_progress_graph_controls(fixture.graph.graph(), fixture.base(), local, fixture.run(), 0, &progress)
    );
    EXPECT_TRUE(progress);
    EXPECT_EQ(local->control_peer_cursor, SCHEDULER_CONTROL_NODE_BUDGET);
    EXPECT_TRUE(scheduler_completion_done(fixture.control(5)->completion_state));
    EXPECT_EQ(scheduler_test::resolved_count(fixture.run()), 1u);
    ASSERT_EQ(scheduler_observe_cached_completion(fixture.base(), local, 16), SchedulerCompletionObservation::DONE);

    auto buffer = std::make_unique<SchedulerActivityBuffer>();
    auto records = std::make_unique<SchedulerScalarStorageSlot[]>(2 * SCHEDULER_SCALAR_RECORD_CAPACITY);
    auto &trace = local->profiling->scalar;
    trace.buffer = buffer.get();
    trace.records = records.get();
    trace.window_start = scheduler_cycles();
    {
        SchedulerScalarBinding binding;
        binding.attach(&trace);
        for (uint32_t poll = 0; poll < 9; ++poll) {
            scheduler_remember_task_proof(local, 16, SchedulerTaskProof::DONE);
            ASSERT_TRUE(scheduler_progress_graph_controls(
                fixture.graph.graph(), fixture.base(), local, fixture.run(), 0, &progress
            ));
            EXPECT_FALSE(progress);
        }
    }
    trace.finish(scheduler_cycles());
    const auto rows = scalar_rows(trace);
    EXPECT_EQ(
        std::count_if(
            rows.begin(), rows.end(),
            [](const auto &row) {
                return static_cast<SchedPhaseKind>(row.kind & 0xffffU) == SchedPhaseKind::ScalarAtomicFetchAdd;
            }
        ),
        0
    );
    EXPECT_EQ(buffer->scalar_dropped, 0u);
    trace.buffer = nullptr;
    EXPECT_EQ(local->control_peer_cursor, 10 * SCHEDULER_CONTROL_NODE_BUDGET);
    for (uint32_t task : {2u, 3u, 18u})
        EXPECT_FALSE(scheduler_completion_done(fixture.control(task)->completion_state));
    EXPECT_EQ(scheduler_test::resolved_count(fixture.run()), 1u);

    // Task 18 hashes with task 2; only its own predecessor can complete it.
    fixture.control(17)->completion_state = scheduler_completion_completed_state(1);
    for (uint32_t slice = 0; slice < 3; ++slice)
        ASSERT_TRUE(
            scheduler_progress_graph_controls(fixture.graph.graph(), fixture.base(), local, fixture.run(), 0, &progress)
        );
    EXPECT_TRUE(scheduler_completion_done(fixture.control(18)->completion_state));
    EXPECT_FALSE(scheduler_completion_done(fixture.control(2)->completion_state));
    EXPECT_FALSE(scheduler_completion_done(fixture.control(3)->completion_state));
    EXPECT_EQ(scheduler_test::resolved_count(fixture.run()), 2u);
    fixture.control(1)->completion_state = scheduler_completion_completed_state(1);
    for (uint32_t slice = 0; slice < 3; ++slice)
        ASSERT_TRUE(
            scheduler_progress_graph_controls(fixture.graph.graph(), fixture.base(), local, fixture.run(), 0, &progress)
        );
    EXPECT_TRUE(scheduler_completion_done(fixture.control(2)->completion_state));
    EXPECT_TRUE(scheduler_completion_done(fixture.control(3)->completion_state));
    EXPECT_EQ(scheduler_test::resolved_count(fixture.run()), 4u);
    EXPECT_EQ(fixture.run()->scheduler_error, 0u);
}

TEST(HbgOrderedScalarDetail, NonAdmissionFillExposesClaimReceiveAndDirectTransfer) {
    OrderedFixture fixture({{1, LaneMask::aic()}});
    ASSERT_TRUE(fixture.prepare());
    auto buffer = std::make_unique<SchedulerActivityBuffer>();
    auto records = std::make_unique<SchedulerScalarStorageSlot[]>(2 * SCHEDULER_SCALAR_RECORD_CAPACITY);
    auto &trace = fixture.locals[0]->profiling->scalar;
    trace.buffer = buffer.get();
    trace.records = records.get();
    trace.window_start = scheduler_cycles();
    {
        SchedulerScalarBinding binding;
        binding.attach(&trace);
        ASSERT_TRUE(fixture.fill(0));
    }
    EXPECT_NE(hbg_test_eligible_mask(fixture.locals[0]), 0u);
    EXPECT_EQ(trace.depth, 0u);
    trace.finish(scheduler_cycles());
    const auto decoded = scalar_rows(trace);
    std::vector<std::string> names;
    for (uint32_t i = 0; i < trace.committed; ++i)
        names.emplace_back(sched_phase_kind_name(static_cast<SchedPhaseKind>(decoded[i].kind & 0xffffU)));
    for (const char *name :
         {"scalar_fill_own_slots", "scalar_execution_token_refill", "scalar_dispatch_publish", "scalar_prepared_claim",
          "scalar_execution_token_progress", "scalar_dispatch_metadata", "scalar_select_owned_execution",
          "scalar_cache_publish"})
        EXPECT_NE(std::find(names.begin(), names.end(), name), names.end()) << name;
    EXPECT_EQ(std::find(names.begin(), names.end(), "scalar_fanin_observe"), names.end());
    EXPECT_EQ(std::find(names.begin(), names.end(), "scalar_ordered_fill"), names.end());
    EXPECT_EQ(buffer->scalar_dropped, 0u);
}

TEST(HbgOrderedScalarDetail, CompletionWriterReusesItsOwnDoneProofForDependencies) {
    for (const auto mask : kPureCAndV) {
        OrderedFixture fixture({{1, mask}, {1, mask}});
        fixture.dependency(1, 0);
        ASSERT_TRUE(fixture.prepare());
        const uint32_t worker = fixture.publisher(0);
        SCOPED_TRACE(worker);
        ASSERT_TRUE(fixture.fill_until_ready(worker));
        OrderedExecution execution{};
        ASSERT_TRUE(fixture.begin(worker, &execution));
        ASSERT_EQ(execution.task, 0);
        ASSERT_TRUE(fixture.complete(execution));
        auto *local = fixture.locals[worker];
        EXPECT_EQ(scheduler_cached_task_proof(local, 0), SchedulerTaskProof::DONE);
        SchedulerFaninView fanin{};
        ASSERT_EQ(scheduler_node_fanin_view(fixture.graph.graph(), 1, &fanin), SchedulerGraphResult::OK);
        auto buffer = std::make_unique<SchedulerActivityBuffer>();
        auto records = std::make_unique<SchedulerScalarStorageSlot[]>(2 * SCHEDULER_SCALAR_RECORD_CAPACITY);
        auto &trace = local->profiling->scalar;
        trace.buffer = buffer.get();
        trace.records = records.get();
        trace.window_start = scheduler_cycles();
        uint32_t prefix = 0;
        bool ready = false;
        {
            SchedulerScalarBinding binding;
            binding.attach(&trace);
            ASSERT_TRUE(scheduler_observe_fanin_prefix(
                fixture.graph.graph(), fixture.base(), local, fixture.run(), 1, fanin, prefix, &ready
            ));
        }
        EXPECT_TRUE(ready);
        EXPECT_EQ(prefix, 1u);
        trace.finish(scheduler_cycles());
        const auto rows = scalar_rows(trace);
        EXPECT_EQ(
            std::count_if(
                rows.begin(), rows.end(),
                [](const auto &row) {
                    return static_cast<SchedPhaseKind>(row.kind & 0xffffU) == SchedPhaseKind::ScalarAtomicFetchAdd;
                }
            ),
            0
        );
        EXPECT_EQ(buffer->scalar_dropped, 0u);
        trace.buffer = nullptr;
    }
}

TEST_P(HbgOrderedFirstReady, FirstBlockIsExecutableWithoutClaimingOrPublishingASecondBlock) {
    OrderedFixture fixture({{8, GetParam(), false}}, true);
    const uint32_t worker = fixture.publisher(0);
    ASSERT_TRUE(fixture.prepare());

    ASSERT_TRUE(fixture.fill(worker));
    EXPECT_EQ(fixture.control(0)->next_block, 0u);
    EXPECT_EQ(fixture.control(0)->claim_sectors[worker & 3].next, 1u);
    EXPECT_EQ(hbg_test_eligible_mask(fixture.locals[worker]), 1u);
    EXPECT_EQ(fixture.locals[worker]->execution_at(1).state, SchedulerDispatchSlotState::FREE);
    const auto *blocks =
        scheduler_state_at<SchedulerProtocolBlockTrace>(fixture.base(), fixture.storage.layout.protocol.blocks_offset);
    const auto *events = scheduler_state_at<SchedulerProtocolExecutionTrace>(
        fixture.base(), fixture.storage.layout.protocol.executions_offset
    );
    const uint32_t first = worker & 3;
    const uint32_t other = (first + 1) % 8;
    EXPECT_EQ(blocks[first].owner.claim_after.sequence, 0u);
    EXPECT_EQ(events[first].publication_count, 1u);
    EXPECT_EQ(blocks[other].owner.claim_before.sequence, 0u);
    EXPECT_EQ(events[other].publication_count, 0u);
    OrderedExecution execution{};
    ASSERT_TRUE(fixture.begin(worker, &execution));
    EXPECT_EQ(execution.block, first);
    EXPECT_NE(events[first].permit.sequence, 0u);
    EXPECT_EQ(blocks[other].owner.claim_before.sequence, 0u);
    EXPECT_EQ(events[other].publication_count, 0u);
}

TEST_P(HbgOrderedFirstReady, CompletionAllowsTheNextClaimWithoutLosingOrDuplicatingBlocks) {
    constexpr uint16_t count = 8;
    OrderedFixture fixture({{count, GetParam(), false}});
    const uint32_t worker = fixture.publisher(0);
    ASSERT_TRUE(fixture.prepare());

    std::array<bool, count> seen{};
    for (uint32_t block = 0; block < count; ++block) {
        ASSERT_TRUE(fixture.fill(worker));
        uint64_t distributed = 0;
        for (const auto &sector : fixture.control(0)->claim_sectors)
            distributed += sector.next;
        EXPECT_EQ(distributed, block + 1);
        OrderedExecution execution{};
        ASSERT_TRUE(fixture.begin(worker, &execution));
        EXPECT_EQ(execution.task, 0);
        ASSERT_LT(execution.block, count);
        EXPECT_FALSE(seen[execution.block]);
        seen[execution.block] = true;
        EXPECT_EQ(execution.blocks, count);
        ASSERT_TRUE(fixture.complete(execution));
        EXPECT_EQ(scheduler_completion_count(fixture.control(0)->completion_state), block + 1);
    }
    EXPECT_EQ(
        scheduler_completion_task_state(scheduler_gm_query(fixture.control(0)->completion_state)),
        static_cast<int64_t>(SchedulerTaskState::DONE)
    );
    ASSERT_TRUE(fixture.fill(worker));
    OrderedExecution extra{};
    EXPECT_FALSE(fixture.begin(worker, &extra));
    EXPECT_EQ(fixture.control(0)->next_block, count);
    EXPECT_EQ(scheduler_completion_count(fixture.control(0)->completion_state), count);
    EXPECT_EQ(fixture.run()->scheduler_error, 0u);
}

TEST_P(HbgOrderedFirstReady, ExhaustedOrdinarySupplyStopsRetriesWithoutReplacingTheSeal) {
    OrderedFixture fixture({{2, GetParam(), false}, {2, GetParam(), false}});
    ASSERT_TRUE(fixture.prepare());
    auto *local = fixture.locals[fixture.publisher(0)];
    bool progress = false;
    int64_t task = SCHEDULER_TASK_REF_INVALID;
    SchedulerIssueSnapshot snapshot{};
    ASSERT_TRUE(
        scheduler_issue_head(fixture.graph.graph(), fixture.base(), local, fixture.run(), &task, &progress, &snapshot)
    );
    ASSERT_EQ(task, 0);
    // Both peers own tickets, but the last claimant has not published its seal.
    fixture.control(0)->next_block = 2;
    auto buffer = std::make_unique<SchedulerActivityBuffer>();
    auto records = std::make_unique<SchedulerScalarStorageSlot[]>(2 * SCHEDULER_SCALAR_RECORD_CAPACITY);
    auto &trace = local->profiling->scalar;
    trace.buffer = buffer.get();
    trace.records = records.get();
    trace.window_start = scheduler_cycles();
    {
        SchedulerScalarBinding binding;
        binding.attach(&trace);
        ASSERT_TRUE(scheduler_refill_execution_tokens(
            fixture.graph.graph(), fixture.base(), local, fixture.run(), 0, &progress, &snapshot
        ));
        EXPECT_FALSE(snapshot.evaluated);
        for (uint32_t poll = 0; poll < 8; ++poll) {
            ASSERT_TRUE(scheduler_refill_execution_tokens(
                fixture.graph.graph(), fixture.base(), local, fixture.run(), 0, &progress
            ));
            local->continuation_task_ref = 0;
            ASSERT_TRUE(scheduler_refill_execution_tokens(
                fixture.graph.graph(), fixture.base(), local, fixture.run(), 0, &progress
            ));
            EXPECT_FALSE(progress);
        }
    }
    trace.finish(scheduler_cycles());
    const auto rows = scalar_rows(trace);
    EXPECT_EQ(
        std::count_if(
            rows.begin(), rows.end(),
            [](const auto &row) {
                return static_cast<SchedPhaseKind>(row.kind & 0xffffU) == SchedPhaseKind::ScalarAtomicFetchAdd;
            }
        ),
        0
    );
    EXPECT_EQ(buffer->scalar_dropped, 0u);
    trace.buffer = nullptr;
    EXPECT_EQ(fixture.control(0)->next_block, 2u);
    EXPECT_EQ(fixture.control(0)->claim_seal, 0u);
    EXPECT_EQ(local->issue_cursor, 0u);
    EXPECT_EQ(scheduler_cached_task_proof(local, 0), SchedulerTaskProof::NONE);
    EXPECT_TRUE(local->issue_ticket_exhausted);
    EXPECT_EQ(scheduler_execution_token_count(local), 0u);
    EXPECT_EQ(fixture.control(1)->next_block, 0u);

    fixture.control(0)->claim_seal = 2;
    ASSERT_TRUE(
        scheduler_refill_execution_tokens(fixture.graph.graph(), fixture.base(), local, fixture.run(), 0, &progress)
    );
    EXPECT_TRUE(progress);
    EXPECT_EQ(local->issue_cursor, 1u);
    EXPECT_EQ(local->issue_task_ref, 1);
    EXPECT_FALSE(local->issue_ticket_exhausted);
    EXPECT_EQ(fixture.control(1)->next_block, 1u);
    EXPECT_TRUE(scheduler_execution_token_has_task(local, 1));
    EXPECT_EQ(scheduler_completion_count(fixture.control(0)->completion_state), 0u);
    EXPECT_EQ(fixture.run()->scheduler_error, 0u);
}

TEST_P(HbgOrderedSharedOrdinaryWork, OrdinaryDiscoveryLeavesWorkForAnotherWorker) {
    OrderedFixture fixture({{1, GetParam()}, {1, GetParam()}});
    const uint32_t worker = fixture.publisher(0);
    const uint32_t peer = worker + 3;
    ASSERT_TRUE(fixture.prepare());
    ASSERT_TRUE(fixture.prepare());
    ASSERT_TRUE(fixture.prepare());
    ASSERT_TRUE(fixture.fill(worker));
    for (uint32_t poll = 0; poll < 8; ++poll)
        ASSERT_TRUE(fixture.fill(worker));
    ASSERT_TRUE(fixture.fill_until_ready(peer));
    OrderedExecution first{};
    OrderedExecution second{};
    ASSERT_TRUE(fixture.begin(worker, &first));
    ASSERT_TRUE(fixture.begin(peer, &second));
    EXPECT_NE(first.task, second.task);
    ASSERT_TRUE(fixture.complete(first));
    ASSERT_TRUE(fixture.complete(second));
    EXPECT_EQ(fixture.run()->scheduler_error, 0u);
}

thread_local std::vector<uint32_t> kernel_block_visits;
void record_block_entry(__gm__ int64_t *args) {
    const auto *context = reinterpret_cast<const LocalContext *>(args[SPMD_LOCAL_CONTEXT_INDEX]);
    ASSERT_LT(static_cast<uint64_t>(context->block_idx), kernel_block_visits.size());
    ++kernel_block_visits[context->block_idx];
}

TEST_P(HbgOrderedBlockIdentity, RealDispatchAndCompletionExecuteEveryOriginalBlockExactlyOnce) {
    const auto [mask, blocks] = GetParam();
    OrderedFixture fixture({{blocks, mask, false}});
    kernel_block_visits.assign(blocks, 0);
    fixture.storage.callable_addresses[1] = reinterpret_cast<uint64_t>(&record_block_entry);
    ASSERT_TRUE(fixture.prepare());
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
            ++executions;
            ASSERT_TRUE(fixture.complete(execution));
        }
    }
    EXPECT_EQ(executions, blocks);
    for (uint32_t visits_for_block : kernel_block_visits)
        EXPECT_EQ(visits_for_block, 1u);
    EXPECT_EQ(
        scheduler_completion_task_state(scheduler_gm_query(fixture.control(0)->completion_state)),
        static_cast<int64_t>(SchedulerTaskState::DONE)
    );
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.control(0)->next_block, UINT64_C(0)), blocks);
    EXPECT_EQ(
        scheduler_completion_count(scheduler_gm_fetch_add(fixture.control(0)->completion_state, UINT64_C(0))), blocks
    );
    for (uint32_t worker = 0; worker < 6; ++worker) {
        ASSERT_TRUE(fixture.fill(worker));
    }
    EXPECT_EQ(scheduler_test::resolved_count(fixture.run()), 1u);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.run()->scheduler_error, UINT64_C(0)), 0u);
}

TEST(HbgOrderedExecution, IndependentTaskFinishesWhileARemoteBlockIsStillExecuting) {
    OrderedFixture fixture({{3, LaneMask::aic(), false}, {2, LaneMask::aic(), false}});
    ASSERT_TRUE(fixture.prepare());

    ASSERT_TRUE(fixture.stage_one(0, 0));
    OrderedExecution slow{};
    ASSERT_TRUE(fixture.begin(0, &slow));
    ASSERT_EQ(slow.task, 0);
    std::set<std::pair<int64_t, uint32_t>> seen{{slow.task, slow.block}};
    std::array<uint32_t, 2> fast_visits{};
    for (uint32_t index = 0; index < 4; ++index) {
        ASSERT_TRUE(fixture.fill_until_ready(3));
        OrderedExecution fast{};
        ASSERT_TRUE(fixture.begin(3, &fast));
        ASSERT_GE(fast.task, 0);
        ASSERT_LT(fast.task, 2);
        ASSERT_TRUE(seen.emplace(fast.task, fast.block).second);
        EXPECT_EQ(fast.blocks, fast.task == 0 ? 3u : 2u);
        ASSERT_LT(fast.block, fast.blocks);
        ++fast_visits[fast.task];
        EXPECT_EQ(fixture.locals[0]->execution_at(slow.slot).state, SchedulerDispatchSlotState::EXECUTING);
        EXPECT_EQ(
            scheduler_completion_task_state(scheduler_gm_query(fixture.control(0)->completion_state)),
            static_cast<int64_t>(SchedulerTaskState::BLOCKED)
        );
        ASSERT_TRUE(fixture.complete(fast));
    }
    EXPECT_EQ(fast_visits, (std::array<uint32_t, 2>{2, 2}));
    EXPECT_EQ(fixture.locals[0]->execution_at(slow.slot).state, SchedulerDispatchSlotState::EXECUTING);
    EXPECT_EQ(scheduler_execution_token_count(fixture.locals[3]), 0u);

    EXPECT_EQ(scheduler_completion_count(fixture.control(0)->completion_state), 2u);
    EXPECT_EQ(scheduler_completion_count(fixture.control(1)->completion_state), 2u);
    EXPECT_EQ(
        scheduler_completion_task_state(scheduler_gm_query(fixture.control(1)->completion_state)),
        static_cast<int64_t>(SchedulerTaskState::DONE)
    );
    ASSERT_TRUE(fixture.complete(slow));
    EXPECT_EQ(scheduler_completion_count(fixture.control(0)->completion_state), 3u);
    EXPECT_EQ(
        scheduler_completion_task_state(scheduler_gm_query(fixture.control(0)->completion_state)),
        static_cast<int64_t>(SchedulerTaskState::DONE)
    );
    EXPECT_EQ(fixture.run()->scheduler_error, 0u);
}

TEST_P(HbgOrderedWithoutSsbuf, MultiBlockAndDependentOrdinaryTaskCompleteWithoutSsbufStorage) {
    const LaneMask mask = GetParam();
    OrderedFixture fixture({{3, mask, false}, {1, mask}});
    fixture.dependency(1, 0);

    const uint32_t publisher = fixture.publisher(0);
    ASSERT_TRUE(fixture.prepare());
    ASSERT_TRUE(fixture.prepare());
    ASSERT_TRUE(fixture.prepare());
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
                    scheduler_completion_count(
                        scheduler_gm_fetch_add(fixture.control(0)->completion_state, UINT64_C(0))
                    ),
                    3u
                );
                EXPECT_EQ(
                    scheduler_completion_task_state(scheduler_gm_query(fixture.control(0)->completion_state)),
                    static_cast<int64_t>(SchedulerTaskState::DONE)
                );
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
        EXPECT_EQ(
            scheduler_completion_task_state(scheduler_gm_query(fixture.control(task)->completion_state)),
            static_cast<int64_t>(SchedulerTaskState::DONE)
        );
    for (uint32_t worker = 0; worker < 6; ++worker) {
        ASSERT_TRUE(fixture.fill(worker));
    }
    EXPECT_EQ(scheduler_test::resolved_count(fixture.run()), 2u);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.run()->scheduler_error, UINT64_C(0)), 0u);
}

TEST_P(HbgOrdinaryTokenProbe, OrdinaryTokenRecordsDiscoveryAndAcquireBeforeDispatch) {
    const auto [cooperative_mask, ordinary_mask] = GetParam();
    OrderedFixture fixture({{1, ordinary_mask}, {2, cooperative_mask}});
    const uint32_t worker = ordinary_mask == LaneMask::aic() ? 0 : 2;

    ASSERT_TRUE(fixture.prepare());
    ASSERT_TRUE(fixture.prefetch(0, worker, SCHEDULER_PROFILING_SCHED_PHASES_LEVEL));
    ASSERT_TRUE(fixture.stage_pending_token(worker, SCHEDULER_PROFILING_SCHED_PHASES_LEVEL));
    const auto *traces =
        scheduler_state_at<SchedulerTaskTrace>(fixture.base(), fixture.storage.layout.trace_cells_offset);
    const auto &trace = traces[fixture.storage.metadata[0].trace_index_base];
    EXPECT_EQ(trace.task_ref, 0u);
    EXPECT_EQ(trace.worker_id, worker);
    EXPECT_GT(trace.state_probe_start_cycles, 0u);
    EXPECT_GE(trace.state_probe_end_cycles, trace.state_probe_start_cycles);
    EXPECT_GT(trace.dispatch_start_cycles, 0u);
    uint32_t owner = UINT32_MAX;
    ASSERT_TRUE(hbg_test_find_eligible(fixture.locals[worker], &owner));
    EXPECT_EQ(
        fixture.locals[worker]->execution_at(owner).rank,
        scheduler_order_for(fixture.base(), fixture.locals[worker], 0).rank
    );
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.run()->scheduler_error, UINT64_C(0)), 0u);
}

TEST_P(HbgOrderedCooperativeProbe, PureSyncUsesTheObservedSelfClaimAndAcquisitionTimes) {
    OrderedFixture fixture({{2, GetParam()}});
    const uint32_t worker = fixture.publisher(0);
    ASSERT_TRUE(fixture.prepare());
    ASSERT_TRUE(fixture.prepare());
    ASSERT_TRUE(fixture.fill(worker, SCHEDULER_PROFILING_SCHED_PHASES_LEVEL));
    const auto *traces =
        scheduler_state_at<SchedulerTaskTrace>(fixture.base(), fixture.storage.layout.trace_cells_offset);
    for (uint32_t block = 0; block < 2; ++block) {
        SCOPED_TRACE(block);
        const uint32_t executor = worker + block * 3;
        ASSERT_TRUE(fixture.fill(executor, SCHEDULER_PROFILING_SCHED_PHASES_LEVEL));
        uint32_t selected = UINT32_MAX;
        ASSERT_TRUE(hbg_test_find_eligible(fixture.locals[executor], &selected));
        const auto &token = fixture.locals[executor]->execution_tokens[selected];
        const auto &trace = traces[fixture.storage.metadata[0].trace_index_base + token.block_idx];
        EXPECT_EQ(trace.task_ref, 0u);
        EXPECT_EQ(trace.worker_id, executor);
        EXPECT_GT(token.discovery_cycles, 0u);
        EXPECT_GE(token.acquired_cycles, token.discovery_cycles);
        EXPECT_EQ(trace.state_probe_start_cycles, token.discovery_cycles);
        EXPECT_EQ(trace.state_probe_end_cycles, token.acquired_cycles);
        EXPECT_GT(trace.dispatch_start_cycles, 0u);
        OrderedExecution execution{};
        const auto selected_block = token.block_idx;
        ASSERT_TRUE(fixture.begin(executor, &execution));
        EXPECT_EQ(execution.block, selected_block);
        ASSERT_TRUE(fixture.complete(execution));
    }
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.run()->scheduler_error, UINT64_C(0)), 0u);
}

TEST(HbgExecutionTokens, ClaimUnreadyBlocksOnceAndExecuteAfterTheDependencyGate) {
    OrderedFixture fixture({{1, LaneMask::aic(), false}, {3, LaneMask::aiv0(), false}});
    fixture.dependency(1, 0);
    ASSERT_TRUE(fixture.prepare());
    EXPECT_FALSE(fixture.prefetch(1, 1));
    EXPECT_EQ(fixture.control(1)->next_block, 0u);
    ASSERT_TRUE(fixture.prefetch(0, 0));
    EXPECT_EQ(fixture.control(0)->claim_seal, 1u);
    EXPECT_EQ(fixture.locals[0]->issue_cursor, 1u);
    const std::array<uint32_t, 3> workers{1, 2, 4};
    for (auto worker : workers) {
        auto *local = fixture.locals[worker];
        bool progress = false;
        ASSERT_TRUE(
            scheduler_refill_execution_tokens(fixture.graph.graph(), fixture.base(), local, fixture.run(), 0, &progress)
        );
        ASSERT_TRUE(progress);
        EXPECT_EQ(fixture.control(1)->claim_seal, worker == workers.back() ? 3u : 0u);
        EXPECT_EQ(local->issue_cursor, worker == workers.back() ? 1u : 0u);
        EXPECT_EQ(scheduler_execution_token_count(local), 1u);
        uint32_t direct = UINT32_MAX;
        ASSERT_TRUE(scheduler_progress_execution_tokens(
            fixture.graph.graph(), fixture.base(), local, fixture.run(), 0, &direct, &progress
        ));
        EXPECT_EQ(direct, UINT32_MAX);
        EXPECT_EQ(hbg_test_eligible_mask(local), 0u);
        for (const auto &owned : local->execution_tokens)
            if (owned.phase == SchedulerExecutionTokenPhase::WAIT_FANIN)
                EXPECT_EQ(owned.state, SchedulerDispatchSlotState::FILLING);
    }
    EXPECT_EQ(fixture.control(1)->next_block, 3u);
    EXPECT_NE(
        scheduler_completion_task_state(fixture.control(0)->completion_state),
        static_cast<int64_t>(SchedulerTaskState::DONE)
    );
    ASSERT_TRUE(fixture.stage_pending_token(0));
    OrderedExecution producer{};
    ASSERT_TRUE(fixture.begin(0, &producer));
    ASSERT_TRUE(fixture.complete(producer));
    std::set<uint32_t> blocks;
    for (auto worker : workers) {
        auto *local = fixture.locals[worker];
        bool progress = false;
        uint32_t direct = UINT32_MAX;
        ASSERT_TRUE(scheduler_progress_execution_tokens(
            fixture.graph.graph(), fixture.base(), local, fixture.run(), 0, &direct, &progress
        ));
        ASSERT_LT(direct, SCHEDULER_LOCAL_EXECUTION_COUNT);
        EXPECT_EQ(fixture.control(1)->next_block, 3u);
        EXPECT_EQ(hbg_test_eligible_mask(local), 1u << direct);
        const auto &slot = local->execution_at(direct);
        EXPECT_TRUE(blocks.insert(slot.block_idx).second);
        OrderedExecution execution{};
        ASSERT_TRUE(fixture.begin(worker, &execution));
        ASSERT_EQ(execution.task, 1);
        ASSERT_TRUE(fixture.complete(execution));
        EXPECT_FALSE(scheduler_has_execution_tokens(local));
    }
    EXPECT_EQ(blocks, (std::set<uint32_t>{0, 1, 2}));
    EXPECT_EQ(
        scheduler_completion_task_state(fixture.control(1)->completion_state),
        static_cast<int64_t>(SchedulerTaskState::DONE)
    );
}

TEST(HbgOrderedProtocol, RecordsClaimOwnerAndActualCompletionTicketRatherThanBlockOrder) {
    for (LaneMask mask : kPureCAndV) {
        OrderedFixture fixture({{2, mask}}, true);
        const uint32_t first = mask == LaneMask::aic() ? 0 : 1;
        const uint32_t second = mask == LaneMask::aic() ? 3 : 2;
        ASSERT_TRUE(fixture.prepare());

        ASSERT_TRUE(fixture.stage_one(first, 0));
        ASSERT_TRUE(fixture.stage_one(second, 0));
        auto *blocks = scheduler_state_at<SchedulerProtocolBlockTrace>(
            fixture.base(), fixture.storage.layout.protocol.blocks_offset
        );
        const uint32_t first_block = fixture.locals[first]->execution_at(0).block_idx;
        const uint32_t second_block = fixture.locals[second]->execution_at(0).block_idx;
        ASSERT_NE(first_block, second_block);
        for (uint32_t block = 0; block < 2; ++block) {
            const auto &owner = blocks[block].owner;
            EXPECT_NE(owner.claim_before.sequence, 0u);
            EXPECT_GT(owner.claim_after.sequence, owner.claim_before.sequence);
            EXPECT_LE(owner.claim_before.cycles, owner.claim_after.cycles);
            EXPECT_EQ(owner.run_epoch, 37u);
            EXPECT_EQ(owner.task_ref, 0);
            EXPECT_EQ(owner.block_idx, block);
            EXPECT_EQ(owner.worker_index, block == first_block ? first : second);
            EXPECT_EQ(owner.logical_block_num, 2u);
            EXPECT_EQ(owner.active_mask, mask);
            EXPECT_EQ(owner.group_seq, 0u);
            EXPECT_EQ(owner.completion_before.sequence, 0u);
        }
        OrderedExecution first_execution{}, second_execution{};
        ASSERT_TRUE(fixture.begin(first, &first_execution));
        ASSERT_TRUE(fixture.begin(second, &second_execution));
        ASSERT_TRUE(fixture.complete(second_execution));
        EXPECT_EQ(blocks[second_block].owner.completion_ticket, 0u);
        EXPECT_GT(
            blocks[second_block].owner.completion_before.sequence, blocks[second_block].owner.claim_after.sequence
        );
        EXPECT_GT(
            blocks[second_block].owner.completion_after.sequence, blocks[second_block].owner.completion_before.sequence
        );
        EXPECT_EQ(blocks[first_block].owner.completion_after.sequence, 0u);
        ASSERT_TRUE(fixture.complete(first_execution));
        EXPECT_EQ(blocks[first_block].owner.completion_ticket, 1u);
        EXPECT_GT(
            blocks[first_block].owner.completion_after.sequence, blocks[first_block].owner.completion_before.sequence
        );
        EXPECT_EQ(
            scheduler_completion_count(scheduler_gm_fetch_add(fixture.control(0)->completion_state, UINT64_C(0))), 2u
        );
    }
}

TEST(HbgOrderedProtocol, EachBlockContributesBeforePublishingItsTrace) {
    OrderedFixture fixture({{2, LaneMask::aic()}}, true);
    ASSERT_TRUE(fixture.prepare());

    for (const uint32_t worker : {0u, 3u}) {
        ASSERT_TRUE(fixture.stage_one(worker, 0));
        OrderedExecution execution;
        ASSERT_TRUE(fixture.begin(worker, &execution));
        auto *local = fixture.locals[worker];
        ASSERT_TRUE(scheduler_complete_lane_execution(
            fixture.graph.graph(), fixture.base(), local, fixture.run(), execution.slot, 4
        ));
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
        EXPECT_EQ(local->execution_at(execution.slot).state, SchedulerDispatchSlotState::FREE);
    }
    EXPECT_EQ(scheduler_completion_count(fixture.control(0)->completion_state), 2u);
}

TEST(HbgOrderedProtocol, DuplicateReturnKeepsTheFirstReturnAndVisibilityPoints) {
    OrderedFixture fixture({{2, LaneMask::aic()}}, true);
    ASSERT_TRUE(fixture.prepare());

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
    ASSERT_TRUE(fixture.prepare());
    ASSERT_TRUE(fixture.prepare());
    ASSERT_TRUE(fixture.fill(0));
    OrderedExecution execution{};
    ASSERT_TRUE(fixture.begin(0, &execution));
    const auto *events = scheduler_state_at<SchedulerProtocolExecutionTrace>(
        fixture.base(), fixture.storage.layout.protocol.executions_offset
    );
    const auto first = events[0];
    SchedulerReadyClaim ready{};
    ready.task_ref = 0;
    scheduler_protocol_payload_publication(fixture.base(), fixture.locals[0], execution.slot, ready, true);
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
    scheduler_protocol_payload_publication(fixture.base(), local, 0, ready, true);
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
    ASSERT_TRUE(fixture.prepare());

    ASSERT_TRUE(fixture.stage_one(0, 0));
    const auto *events = scheduler_state_at<SchedulerProtocolExecutionTrace>(
        fixture.base(), fixture.storage.layout.protocol.executions_offset
    );
    ASSERT_EQ(events[0].permit.sequence, 0u);
    const auto first = events[0];
    SchedulerReadyClaim ready{};
    ready.task_ref = 0;
    scheduler_protocol_payload_publication(fixture.base(), fixture.locals[0], 0, ready, true);
    EXPECT_EQ(events[0].publication_count, first.publication_count);
    EXPECT_EQ(events[0].pending_slot, first.pending_slot);
    const auto *workers = scheduler_state_at<SchedulerProtocolWorkerTrace>(
        fixture.base(), fixture.storage.layout.protocol.workers_offset
    );
    EXPECT_EQ(workers[0].error_flags, SCHEDULER_PROTOCOL_DUPLICATE);
}

INSTANTIATE_TEST_SUITE_P(PureCAndV, HbgOrderedFirstReady, ::testing::ValuesIn(kPureCAndV));

INSTANTIATE_TEST_SUITE_P(PureCAndV, HbgOrderedSharedOrdinaryWork, ::testing::ValuesIn(kPureCAndV));

INSTANTIATE_TEST_SUITE_P(
    PureCAndVCapacityBoundaries, HbgOrderedBlockIdentity,
    ::testing::Combine(
        ::testing::ValuesIn(kPureCAndV), ::testing::Values<uint16_t>(1, 2, 3, 7, 31, 32, 33, 63, 64, 65, 129, 512)
    )
);

INSTANTIATE_TEST_SUITE_P(PureCAndV, HbgOrderedWithoutSsbuf, ::testing::ValuesIn(kPureCAndV));

INSTANTIATE_TEST_SUITE_P(
    SameAndDifferentCoreTypes, HbgOrdinaryTokenProbe,
    ::testing::Combine(::testing::ValuesIn(kPureCAndV), ::testing::ValuesIn(kPureCAndV))
);

INSTANTIATE_TEST_SUITE_P(PureCAndV, HbgOrderedCooperativeProbe, ::testing::ValuesIn(kPureCAndV));

}  // namespace
