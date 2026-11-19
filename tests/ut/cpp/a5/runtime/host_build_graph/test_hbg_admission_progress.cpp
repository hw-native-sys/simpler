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

#include <array>
#include <set>

#include "hbg_scheduler_test_support.h"
#include "scheduler/scheduler_dispatch.h"
#include "test_hbg_owned_execution.h"

namespace {

struct Task {
    LaneMask lanes;
    uint16_t blocks;
    bool sync;
};

class Ranked {
public:
    explicit Ranked(std::initializer_list<Task> tasks) :
        store(tasks.size(), 6, 12, requirements(tasks)),
        graph(tasks.size()) {
        uint32_t id = 0;
        uint32_t trace = 0;
        for (const auto &task : tasks) {
            auto &meta = store.metadata[id];
            meta.active_mask = task.lanes;
            meta.logical_block_num = task.blocks;
            meta.flags = SCHEDULER_TASK_EXECUTABLE;
            if (task.blocks > 1) meta.flags |= SCHEDULER_TASK_SPMD;
            if (task.sync) meta.flags |= SCHEDULER_TASK_SYNC_START;
            if (task.lanes.is_mix()) meta.flags |= SCHEDULER_TASK_MIX;
            meta.trace_index_base = trace;
            trace += task.blocks * task.lanes.count();
            graph.mixed(id, task.lanes);
            for (uint32_t lane = 0; lane < 3; ++lane)
                meta.kernel_ids[lane] = task.lanes.has(static_cast<SubtaskSlot>(lane)) ? 1 : UINT16_MAX;
            ++id;
        }
        store.prepare_catalog();
        EXPECT_TRUE(scheduler_init_spmd_task_records(base(), store.layout, 31));
        EXPECT_TRUE(store.prepare_calls(graph.graph()));
        for (uint32_t worker = 0; worker < locals.size(); ++worker) {
            auto &c = store.contexts[worker];
            for (uint32_t lane = 0; lane < 3; ++lane)
                c.cluster_worker_ids[lane] = worker / 3 * 3 + lane;
            locals[worker] = store.local_context(&c);
            locals[worker]->cluster_signal = &signals[worker / 3];
            locals[worker]->run_epoch = 31;
        }
        for (auto &s : signals)
            scheduler_cluster_signal_clear(&s, 31);
    }

    void *base() { return store.scheduler_state->base(); }
    SchedulerRunControl *run() { return store.run_control; }
    SchedulerTaskControl *ctrl(uint32_t ref) { return scheduler_task_control_at(base(), locals[0], ref); }
    const SchedulerTaskCatalogOrder &order(int64_t ref) { return scheduler_order_for(base(), locals[0], ref); }

    bool next(uint32_t worker, uint32_t *selected) {
        bool failed = false;
        for (uint32_t n = 0; n <= 2 * graph.graph().task_count + SCHEDULER_EXECUTION_TOKEN_COUNT; ++n) {
            scheduler_fill_own_slots(graph.graph(), base(), locals[worker], run(), 0, &failed, selected);
            if (failed || *selected != UINT32_MAX) break;
        }
        return !failed;
    }
    bool finish(uint32_t worker, uint32_t selected) {
        bool allow = false;
        auto *local = locals[worker];
        return scheduler_can_enter_execution(graph.graph(), base(), local, run(), selected, &allow) && allow &&
               hbg_test_begin_execution(local, selected) &&
               scheduler_complete_own_slot(graph.graph(), base(), local, run(), selected, 0);
    }
    static SchedulerSpmdRequirements requirements(std::initializer_list<Task> tasks) {
        SchedulerSpmdRequirements req{6, 2};
        for (auto t : tasks) {
            if (t.sync) ++req.cooperative_task_count;
            if (t.lanes.is_mix()) ++req.mix_task_count;
            if (t.sync && t.lanes.is_mix()) ++req.sync_mix_task_count;
        }
        return req;
    }
    scheduler_test::FixtureStorage store;
    scheduler_test::BasicGraphBuffer<16> graph;
    std::array<SchedulerLocalState *, 6> locals{};
    std::array<SchedulerClusterSignal, 2> signals{};
};

TEST(HbgResourceOrder, PureSyncMemberCanReturnBeforeAnotherCoreClaimsAndNeverTakesItsPeerBlock) {
    Ranked f({{LaneMask::aic(), 2, true}, {LaneMask::aic(), 1, false}});
    uint32_t first = UINT32_MAX;
    ASSERT_TRUE(f.next(0, &first));
    ASSERT_LT(first, SCHEDULER_EXECUTION_TOKEN_COUNT);
    ASSERT_EQ(f.locals[0]->execution_at(first).task_ref, 0);
    const auto first_block = f.locals[0]->execution_at(first).block_idx;
    ASSERT_TRUE(f.finish(0, first));
    EXPECT_FALSE(scheduler_completion_done(scheduler_gm_query(f.ctrl(0)->completion_state)));
    EXPECT_EQ(scheduler_gm_query(f.ctrl(0)->claim_seal), 0u);
    EXPECT_EQ(f.locals[0]->last_sync_self_ref, 0);
    uint32_t occupied = UINT32_MAX;
    ASSERT_TRUE(f.next(0, &occupied));
    EXPECT_EQ(occupied, UINT32_MAX);
    uint32_t other = UINT32_MAX;
    ASSERT_TRUE(f.next(3, &other));
    ASSERT_LT(other, SCHEDULER_EXECUTION_TOKEN_COUNT);
    EXPECT_EQ(f.locals[3]->execution_at(other).task_ref, 0);
    EXPECT_NE(f.locals[3]->execution_at(other).block_idx, first_block);
    EXPECT_EQ(scheduler_gm_query(f.ctrl(0)->claim_seal), 2u);
    ASSERT_TRUE(f.finish(3, other));
    ASSERT_TRUE(f.next(0, &first));
    ASSERT_LT(first, SCHEDULER_EXECUTION_TOKEN_COUNT);
    EXPECT_EQ(f.locals[0]->execution_at(first).task_ref, 1);
    EXPECT_TRUE(f.finish(0, first));
    EXPECT_EQ(scheduler_test::resolved_count(f.run()), 2u);
}

TEST(HbgResourceOrder, OrdinaryBlockOwnerCanRefillAfterItsOwnReturn) {
    Ranked f({{LaneMask::aic(), 3, false}});
    std::set<uint32_t> seen;
    for (uint32_t n = 0; n < 3; ++n) {
        uint32_t selected = UINT32_MAX;
        ASSERT_TRUE(f.next(0, &selected));
        ASSERT_LT(selected, SCHEDULER_EXECUTION_TOKEN_COUNT);
        ASSERT_TRUE(seen.insert(f.locals[0]->execution_at(selected).block_idx).second);
        ASSERT_TRUE(f.finish(0, selected));
    }
    EXPECT_EQ(seen, (std::set<uint32_t>{0, 1, 2}));
    EXPECT_EQ(scheduler_test::resolved_count(f.run()), 1u);
}

TEST(HbgResourceOrder, PureCohortWaitsForBothPublicationFloorAndItsOwnEarlierLaneObligation) {
    Ranked f({{LaneMask::all(), 1, false}, {LaneMask::aiv0(), 2, true}});
    auto *lane = f.locals[2];
    auto &signal = f.signals[0];
    const auto &self = f.order(1);
    const auto required = self.required_floor[lane->config.self_lane];
    ASSERT_EQ(required, 1u);
    bool ready = true;
    const auto allowed = [&] {
        return scheduler_self_order_ready(f.graph.graph(), f.base(), lane, f.run(), self.rank, required, true, &ready);
    };
    ASSERT_TRUE(allowed());
    EXPECT_FALSE(ready);
    scheduler_cluster_store_field(&signal.rank, f.order(0).rank);
    scheduler_cluster_store_field(&signal.active_mask, UINT32_C(7));
    scheduler_cluster_store_release(&signal.armed_generation, UINT32_C(3));
    scheduler_cluster_store_release(&signal.mix_floor, required);
    ASSERT_TRUE(allowed());
    EXPECT_FALSE(ready);
    scheduler_cluster_store_release(&signal.finished[2], UINT32_C(1));
    ASSERT_TRUE(allowed());
    EXPECT_FALSE(ready);
    scheduler_cluster_store_release(&signal.finished[2], UINT32_C(3));
    ASSERT_TRUE(allowed());
    EXPECT_TRUE(ready);
}

TEST(HbgResourceOrder, DisjointCubeDoesNotNeedAVectorOnlyMixFloor) {
    Ranked f({{LaneMask::both_aiv(), 1, false}, {LaneMask::aic(), 2, true}});
    auto *cube = f.locals[0];
    cube->cluster_signal = nullptr;
    const auto &self = f.order(1);
    ASSERT_EQ(self.required_floor[0], 0u);
    uint32_t selected = UINT32_MAX;
    ASSERT_TRUE(f.next(0, &selected));
    ASSERT_LT(selected, SCHEDULER_EXECUTION_TOKEN_COUNT);
    ASSERT_EQ(cube->execution_at(selected).task_ref, 1);
    EXPECT_TRUE(f.finish(0, selected));
}

TEST(HbgResourceOrder, EarlierSelfClaimRemainsAnObligationWhenItsDirectoryAdvances) {
    Ranked f({{LaneMask::aiv0(), 2, true}, {LaneMask::all(), 1, false}});
    auto *lane = f.locals[2];
    const auto &later = f.order(1);
    ASSERT_GT(later.required_floor[2], 0u);
    bool ready = true;
    const auto check = [&] {
        return scheduler_mix_lane_order_ready(f.graph.graph(), f.base(), lane, f.run(), 1, &ready);
    };
    ASSERT_TRUE(check());
    EXPECT_FALSE(ready);
    uint32_t first;
    ASSERT_TRUE(f.next(2, &first));
    ASSERT_LT(first, SCHEDULER_EXECUTION_TOKEN_COUNT);
    uint32_t second;
    ASSERT_TRUE(f.next(4, &second));
    ASSERT_LT(second, SCHEDULER_EXECUTION_TOKEN_COUNT);
    bool progress = false;
    int64_t head = SCHEDULER_TASK_REF_INVALID;
    ASSERT_TRUE(scheduler_issue_head(f.graph.graph(), f.base(), lane, f.run(), &head, &progress));
    ASSERT_GT(lane->issue_cursor, 0u);
    ASSERT_EQ(lane->execution_at(first).task_ref, 0);
    ASSERT_TRUE(check());
    EXPECT_FALSE(ready);
    ASSERT_TRUE(f.finish(2, first));
    ASSERT_TRUE(check());
    EXPECT_TRUE(ready);
}

TEST(HbgResourceOrder, AClosedSupplyCannotAdvancePastTheProxysUnpublishedGroup) {
    Ranked f({{LaneMask::all(), 1, false}, {LaneMask::aiv0(), 2, true}});
    auto *proxy = f.locals[1];
    ASSERT_TRUE(proxy->is_mix_coordinator());
    ASSERT_EQ(proxy->next_mix_ref, 0u);
    scheduler_gm_store(f.ctrl(0)->claim_seal, UINT64_C(1));
    proxy->cluster_pending.task_ref = 0;
    bool progress = false;
    ASSERT_TRUE(scheduler_advance_mix_floor(f.graph.graph(), f.base(), proxy, f.run(), &progress));
    EXPECT_EQ(scheduler_cluster_load_acquire(&f.signals[0].mix_floor), 0u);
    EXPECT_FALSE(progress);
    proxy->cluster_pending.task_ref = SCHEDULER_TASK_REF_INVALID;
    ASSERT_TRUE(scheduler_advance_mix_floor(f.graph.graph(), f.base(), proxy, f.run(), &progress));
    EXPECT_TRUE(progress);
    EXPECT_EQ(scheduler_cluster_load_acquire(&f.signals[0].mix_floor), 2u);
}

}  // namespace
