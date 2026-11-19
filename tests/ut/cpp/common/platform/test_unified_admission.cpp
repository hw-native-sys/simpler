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
#include "support/mix_lane_priority_fixture.h"

class MixAdmissionTestPeer {
public:
    static void owners(SchedulerContext &c, uint64_t *regs, size_t words) {
        c.active_sched_threads_ = 3;
        c.cores_total_num_ = 9;
        c.lane_pool_.reset();
        for (int t = 0; t < 3; ++t) {
            c.core_trackers_[t].init(1);
            c.local_lanes_[t] = {};
            c.core_trackers_[t].set_cluster(0, t * 3, t * 3 + 1, t * 3 + 2);
            for (int lane = 0; lane < 3; ++lane) {
                auto &core = c.core_exec_states_[t * 3 + lane];
                core = {};
                core.reg_addr = reinterpret_cast<uint64_t>(regs + (t * 3 + lane) * words);
                core.running_reg_task_id = AICPU_TASK_INVALID;
                core.pending_reg_task_id = AICPU_TASK_INVALID;
            }
        }
        for (int t = 0; t < 3; ++t)
            c.lane_pool_.initialize(t, 1);
    }
    static void acknowledge_running_payload(SchedulerContext &c, int owner, int offset) {
        const auto &core = c.core_exec_states_[c.core_trackers_[owner].get_core_id_by_offset(offset)];
        ASSERT_NE(core.running_reg_task_id, AICPU_TASK_INVALID);
        ASSERT_EQ(core.pending_reg_task_id, AICPU_TASK_INVALID);
        c.core_trackers_[owner].clear_pending_occupied(offset);
    }
    static void admit(SchedulerContext &c, int owner = 2) { c.admit_ready_sync(owner); }
    static bool consume(SchedulerContext &c, int owner) { return c.consume_ready_sync(owner); }
    static int remaining(SchedulerContext &c) { return c.ready_sync_.remaining.load(); }
    static uint64_t generation(SchedulerContext &c) { return c.ready_sync_.state.load() >> 3; }
    static bool idle(SchedulerContext &c) {
        return c.admission_mode(c.ready_sync_.state.load()) == SchedulerContext::AdmissionMode::IDLE;
    }
    static bool waiting(SchedulerContext &c) {
        return c.admission_mode(c.ready_sync_.state.load()) == SchedulerContext::AdmissionMode::WAIT_CAPACITY;
    }
    static uint64_t free(SchedulerContext &c) {
        uint64_t result = 0;
        for (int t = 0; t < c.active_sched_threads_; ++t) {
            for (int i = 0; i < c.owner_clusters(t); ++i) {
                auto v = c.lane_pool_.snapshot(t, i / 10);
                if (((v >> (3 * (i % 10))) & 7) == 7) result |= uint64_t{1} << (t + i * c.active_sched_threads_);
            }
        }
        return result;
    }
    static bool scan_pending(SchedulerContext &c, int owner) { return c.local_lanes_[owner].needs_idle_scan; }
    static uint32_t free_word(SchedulerContext &c, int owner) { return uint32_t(c.lane_pool_.snapshot(owner, 0)); }
    static uint32_t protected_word(SchedulerContext &c, int owner) { return c.lane_pool_.snapshot(owner, 0) >> 32; }
    static int ordinary_early(SchedulerContext &c, int owner) {
        bool progress = false, pushed = false;
        return c.try_early_dispatch(owner, c.core_trackers_[owner], false, progress, pushed);
    }
    static void regular_vector(SchedulerContext &c, int owner) {
        bool stop = false, progress = false, pushed = false;
        LaneMask observed = LaneMask::none();
        c.dispatch_shape(
            owner, c.sched_->ready_queues, ResourceShape::AIV, CoreTracker::DispatchPhase::IDLE,
            c.core_trackers_[owner], stop, progress, pushed, SchedulerContext::MixPriority::Ready, observed
        );
    }
    static void wide(SchedulerContext &c, uint64_t *regs, size_t words) {
        ASSERT_TRUE(assign_serial(c, 3, 24));
        for (int i = 0; i < 72; ++i) {
            auto &core = c.core_exec_states_[i];
            core = {};
            core.reg_addr = reinterpret_cast<uint64_t>(regs + i * words);
            core.cond_ptr = reinterpret_cast<volatile uint32_t *>(core.reg_addr + reg_offset(RegId::COND));
            core.running_reg_task_id = core.pending_reg_task_id = AICPU_TASK_INVALID;
        }
    }
    static void poll_one(SchedulerContext &c, int owner, int id, bool fin) {
        auto &core = c.core_exec_states_[id];
        *core.cond_ptr = fin ? MAKE_FIN_VALUE(core.running_reg_task_id) : uint32_t(core.running_reg_task_id);
        int completed = 0, core_completions = 0, count = 0;
        bool progress = false;
        ChipTaskSlotState *deferred[64]{};
        c.check_running_cores_for_completion(owner, nullptr, completed, core_completions, progress, deferred, count);
        c.release_idle_lanes(owner);
        EXPECT_EQ(core_completions, fin ? 1 : 0);
    }
    static int plan_whole_chip(SchedulerContext &c) {
        return c.collect_ready_positions(LaneMask::all(), c.cores_total_num_ / 3, false, false);
    }
    static bool reserve_plan(SchedulerContext &c) { return c.reserve_ready_plans(); }
    static bool acquire_one(SchedulerContext &c, int owner, LaneMask lanes, int offset) {
        return c.acquire_local_lanes(owner, CoreTracker::BitStates::bit(offset), lanes.to_shape(), lanes, 1)
            .has_value();
    }
    static void rollback_race(SchedulerContext &c) {
        c.lane_pool_.reset();
        c.active_sched_threads_ = 2;
        c.ready_sync_.plans[0] = {};
        c.ready_sync_.plans[1] = {};
        c.ready_sync_.plans[0].grant[0] = 7;
        c.ready_sync_.plans[1].grant[0] = 7;
        for (unsigned a = 0; a <= 7; ++a)
            for (unsigned b = 0; b <= 7; ++b) {
                c.lane_pool_.reset();
                c.lane_pool_.release(0, 0, a);
                c.lane_pool_.release(1, 0, b);
                c.lane_pool_.protect(0, 0, 5);
                c.lane_pool_.protect(1, 0, 3);
                bool got = c.reserve_ready_plans();
                ASSERT_EQ(got, a == 7 && b == 7);
                EXPECT_EQ(c.lane_pool_.snapshot(0, 0), (uint64_t(5) << 32) | (got ? 0 : a));
                EXPECT_EQ(c.lane_pool_.snapshot(1, 0), (uint64_t(3) << 32) | (got ? 0 : b));
            }
    }
    static bool hold(SchedulerContext &c, int owner) {
        return c.acquire_local_lanes(owner, CoreTracker::BitStates::bit(0), ResourceShape::MIX, LaneMask::all(), 1)
            .has_value();
    }
    static void release_unused(SchedulerContext &c, int owner) { c.release_idle_lanes(owner); }
    static void retire_fixture(SchedulerContext &c, int owner) {
        auto &tracker = c.core_trackers_[owner];
        for (int lane = 0; lane < 3; ++lane) {
            tracker.change_core_state(lane);
            tracker.clear_pending_occupied(lane);
            auto &core = c.core_exec_states_[owner * 3 + lane];
            core.running_reg_task_id = AICPU_TASK_INVALID;
            core.running_slot_state = nullptr;
        }
        c.local_lanes_[owner].needs_idle_scan = true;
        c.release_idle_lanes(owner);
    }
    static bool legacy(SchedulerContext &c, ChipTaskSlotState &slot) { return c.enter_drain_mode(&slot, 1); }
    static ChipTaskSlotState *placed(SchedulerContext &c, int core) {
        return c.core_exec_states_[core].running_slot_state;
    }
    static uint64_t published_token(SchedulerContext &c, int core) {
        return *reinterpret_cast<const uint64_t *>(
            c.core_exec_states_[core].reg_addr + reg_offset(RegId::DATA_MAIN_BASE)
        );
    }
    static void fatal(SchedulerContext &c) { c.completed_.store(true, std::memory_order_release); }
    static void reset_joined(SchedulerContext &c) { c.deinit(); }
    static bool assign_serial(SchedulerContext &c, int owners, int clusters) {
        c.lane_pool_.reset();
        for (auto &local : c.local_lanes_)
            local = {};
        c.aicpu_thread_num_ = owners;
        c.sched_thread_num_ = owners;
        c.cores_total_num_ = clusters * 3;
        c.aic_count_ = clusters;
        c.aiv_count_ = clusters * 2;
        for (int ci = 0; ci < clusters; ++ci) {
            c.aic_worker_ids_[ci] = ci;
            c.aiv_worker_ids_[2 * ci] = clusters + 2 * ci;
            c.aiv_worker_ids_[2 * ci + 1] = clusters + 2 * ci + 1;
        }
        return c.assign_cores_to_threads();
    }
    static int claim_assigned(SchedulerContext &c, int owner) {
        return c
            .acquire_local_lanes(
                owner, c.core_trackers_[owner].get_cluster_offset_states(), ResourceShape::MIX, LaneMask::all(), 24
            )
            .count();
    }
    static void toggle_running(SchedulerContext &c, int owner, int offset) {
        c.core_trackers_[owner].change_core_state(offset);
        c.local_lanes_[owner].needs_idle_scan = true;
    }
    static void pending(SchedulerContext &c, int owner, int offset, bool occupied) {
        if (occupied) c.core_trackers_[owner].set_pending_occupied(offset);
        else c.core_trackers_[owner].clear_pending_occupied(offset);
        c.local_lanes_[owner].needs_idle_scan = true;
    }
    static void single_owner(SchedulerContext &c, uint64_t *regs, size_t words, int clusters) {
        c.active_sched_threads_ = 1;
        c.cores_total_num_ = clusters * 3;
        c.lane_pool_.reset();
        c.lane_pool_.initialize(0, clusters);
        c.local_lanes_[0] = {};
        c.core_trackers_[0].init(clusters);
        for (int b = 0; b < clusters; ++b) {
            c.core_trackers_[0].set_cluster(b, b * 3, b * 3 + 1, b * 3 + 2);
            for (int lane = 0; lane < 3; ++lane) {
                auto &core = c.core_exec_states_[b * 3 + lane];
                core = {};
                core.reg_addr = reinterpret_cast<uint64_t>(regs + (b * 3 + lane) * words);
                core.running_reg_task_id = AICPU_TASK_INVALID;
                core.pending_reg_task_id = AICPU_TASK_INVALID;
            }
        }
    }
};

class UnifiedAdmission : public LanePriority {
    void SetUp() override {
        LanePriority::SetUp();
        const size_t words = (reg_offset(RegId::COND) + sizeof(uint64_t)) / sizeof(uint64_t);
        registers.resize(9 * words);
        MixAdmissionTestPeer::owners(*context, registers.data(), words);
        mixed.init(7, 3);
        mixed.slot.task_attrs.set_sync_start();
    }

protected:
    void enqueue_ss() { ASSERT_TRUE(sched->ready_sync_queues[2].push_mix(&mixed.slot, mixed.descriptor.task_id)); }
};

TEST_P(UnifiedAdmission, ParallelOwnersRetireWithoutWaitingForPeers) {
    enqueue_ss();
    MixAdmissionTestPeer::admit(*context);
    EXPECT_EQ(SchedulerContextTestPeer::drain_pending(*context), 0);
    EXPECT_EQ(mixed.slot.next_block_idx.load(), 3);
    EXPECT_EQ(MixAdmissionTestPeer::remaining(*context), 3);
    EXPECT_TRUE(MixAdmissionTestPeer::consume(*context, 1));
    EXPECT_EQ(MixAdmissionTestPeer::remaining(*context), 2);
    EXPECT_FALSE(MixAdmissionTestPeer::consume(*context, 1));
    EXPECT_EQ(MixAdmissionTestPeer::placed(*context, 0), nullptr);
    EXPECT_EQ(MixAdmissionTestPeer::placed(*context, 3), &mixed.slot);
    EXPECT_FALSE(MixAdmissionTestPeer::legacy(*context, pure.slot));
    EXPECT_TRUE(MixAdmissionTestPeer::consume(*context, 0));
    EXPECT_FALSE(MixAdmissionTestPeer::idle(*context));
    EXPECT_TRUE(MixAdmissionTestPeer::consume(*context, 2));
    EXPECT_TRUE(MixAdmissionTestPeer::idle(*context));
    for (int c = 0; c < 9; ++c)
        EXPECT_EQ(MixAdmissionTestPeer::placed(*context, c), &mixed.slot);
}

TEST_P(UnifiedAdmission, NonparticipantBuilderDoesNotHoldRetirement) {
    mixed.init(7, 1);
    enqueue_ss();
    MixAdmissionTestPeer::admit(*context, 2);
    EXPECT_EQ(MixAdmissionTestPeer::remaining(*context), 1);
    EXPECT_FALSE(MixAdmissionTestPeer::consume(*context, 2));
    EXPECT_TRUE(MixAdmissionTestPeer::consume(*context, 0));
    EXPECT_TRUE(MixAdmissionTestPeer::idle(*context));
    EXPECT_EQ(MixAdmissionTestPeer::free(*context), 6);
}

TEST_P(UnifiedAdmission, ShortageClaimsNothingAndCurrentFreeStateRestartsAdmission) {
    ASSERT_TRUE(MixAdmissionTestPeer::hold(*context, 0));
    enqueue_ss();
    MixAdmissionTestPeer::admit(*context);
    EXPECT_TRUE(MixAdmissionTestPeer::waiting(*context));
    EXPECT_EQ(mixed.slot.next_block_idx.load(), 0);
    EXPECT_EQ(MixAdmissionTestPeer::free(*context), 6);
    for (int t = 0; t < 3; ++t)
        EXPECT_FALSE(MixAdmissionTestPeer::hold(*context, t));
    MixAdmissionTestPeer::release_unused(*context, 0);
    EXPECT_EQ(MixAdmissionTestPeer::free(*context), 7);
    MixAdmissionTestPeer::admit(*context, 0);
    EXPECT_EQ(mixed.slot.next_block_idx.load(), 3);
    for (int t = 0; t < 3; ++t)
        EXPECT_TRUE(MixAdmissionTestPeer::consume(*context, t));
    EXPECT_TRUE(MixAdmissionTestPeer::idle(*context));
}

TEST_P(UnifiedAdmission, ReservedIdleTrackerCannotBeUsedByOrdinaryPlacement) {
    enqueue_ss();
    MixAdmissionTestPeer::admit(*context);
    for (int t = 0; t < 3; ++t)
        EXPECT_FALSE(MixAdmissionTestPeer::hold(*context, t));
    pure.init(7, 1, 2);
    ASSERT_TRUE(sched->ready_queues[2].push_mix(&pure.slot, pure.descriptor.task_id));
    LaneMask observed = LaneMask::none();
    SchedulerContextTestPeer::leaf(*context, ResourceShape::MIX, false, false, observed);
    EXPECT_EQ(pure.slot.next_block_idx.load(), 0);
    EXPECT_EQ(MixAdmissionTestPeer::placed(*context, 0), nullptr);
    for (int t = 0; t < 3; ++t)
        EXPECT_TRUE(MixAdmissionTestPeer::consume(*context, t));
}

TEST_P(UnifiedAdmission, ReuseRequiresAllOwnersToDropOldRecord) {
    enqueue_ss();
    MixAdmissionTestPeer::admit(*context);
    uint64_t generation = MixAdmissionTestPeer::generation(*context);
    for (int t = 0; t < 2; ++t) {
        ASSERT_TRUE(MixAdmissionTestPeer::consume(*context, t));
        MixAdmissionTestPeer::retire_fixture(*context, t);
    }
    pure.init(7, 2, 2);
    pure.slot.task_attrs.set_sync_start();
    ASSERT_TRUE(sched->ready_sync_queues[2].push_mix(&pure.slot, pure.descriptor.task_id));
    MixAdmissionTestPeer::admit(*context);
    EXPECT_EQ(pure.slot.next_block_idx.load(), 0);
    ASSERT_TRUE(MixAdmissionTestPeer::consume(*context, 2));
    MixAdmissionTestPeer::admit(*context);
    EXPECT_GT(MixAdmissionTestPeer::generation(*context), generation);
    EXPECT_EQ(pure.slot.next_block_idx.load(), 2);
    EXPECT_FALSE(MixAdmissionTestPeer::consume(*context, 2));
    EXPECT_TRUE(MixAdmissionTestPeer::consume(*context, 0));
    EXPECT_TRUE(MixAdmissionTestPeer::consume(*context, 1));
    EXPECT_TRUE(MixAdmissionTestPeer::idle(*context));
}

TEST_P(UnifiedAdmission, PartialMixOnlyReservesAndPublishesItsTwoLanes) {
    mixed.init(3, 3);
    enqueue_ss();
    MixAdmissionTestPeer::admit(*context);
    ASSERT_EQ(mixed.slot.next_block_idx.load(), 3);
    EXPECT_EQ(SchedulerContextTestPeer::drain_pending(*context), 0);
    for (int t = 0; t < 3; ++t) {
        EXPECT_EQ(MixAdmissionTestPeer::free_word(*context, t), 4u);
        ASSERT_TRUE(MixAdmissionTestPeer::consume(*context, t));
        EXPECT_EQ(MixAdmissionTestPeer::placed(*context, 3 * t), &mixed.slot);
        EXPECT_EQ(MixAdmissionTestPeer::placed(*context, 3 * t + 1), &mixed.slot);
        EXPECT_EQ(MixAdmissionTestPeer::placed(*context, 3 * t + 2), nullptr);
    }
}
TEST_P(UnifiedAdmission, PartialLaneLocalFitPublishesWithoutDrain) {
    mixed.init(3, 1);
    enqueue_ss();
    MixAdmissionTestPeer::admit(*context, 2);
    ASSERT_TRUE(MixAdmissionTestPeer::consume(*context, 0));
    EXPECT_EQ(mixed.slot.next_block_idx.load(), 1);
    EXPECT_EQ(SchedulerContextTestPeer::drain_pending(*context), 0);
    EXPECT_EQ(MixAdmissionTestPeer::placed(*context, 0), &mixed.slot);
    EXPECT_EQ(MixAdmissionTestPeer::placed(*context, 1), &mixed.slot);
    EXPECT_EQ(MixAdmissionTestPeer::placed(*context, 2), nullptr);
    EXPECT_NE(MixAdmissionTestPeer::published_token(*context, 0), 0u);
    EXPECT_TRUE(MixAdmissionTestPeer::idle(*context));
}

TEST_P(UnifiedAdmission, ConcurrentOwnersPublishExactlyOnce) {
    enqueue_ss();
    MixAdmissionTestPeer::admit(*context);
    std::atomic<int> arrived{0};
    std::atomic<bool> go{false};
    bool consumed[3]{};
    std::thread owners[3];
    for (int t = 0; t < 3; ++t)
        owners[t] = std::thread([&, t] {
            arrived.fetch_add(1, std::memory_order_release);
            while (!go.load(std::memory_order_acquire)) {}
            consumed[t] = MixAdmissionTestPeer::consume(*context, t);
        });
    while (arrived.load(std::memory_order_acquire) != 3) {}
    EXPECT_EQ(MixAdmissionTestPeer::remaining(*context), 3);
    go.store(true, std::memory_order_release);
    for (auto &t : owners)
        t.join();
    for (bool result : consumed)
        EXPECT_TRUE(result);
    EXPECT_TRUE(MixAdmissionTestPeer::idle(*context));
    EXPECT_EQ(MixAdmissionTestPeer::remaining(*context), 0);
    for (int t = 0; t < 3; ++t)
        EXPECT_FALSE(MixAdmissionTestPeer::consume(*context, t));
}

TEST_P(UnifiedAdmission, FatalRetainsUnconsumedRequestUntilJoinedReset) {
    enqueue_ss();
    MixAdmissionTestPeer::admit(*context);
    ASSERT_TRUE(MixAdmissionTestPeer::consume(*context, 0));
    MixAdmissionTestPeer::fatal(*context);
    EXPECT_EQ(MixAdmissionTestPeer::remaining(*context), 2);
    EXPECT_FALSE(MixAdmissionTestPeer::idle(*context));
    EXPECT_EQ(MixAdmissionTestPeer::free(*context), 0);
    MixAdmissionTestPeer::reset_joined(*context);
    EXPECT_TRUE(MixAdmissionTestPeer::idle(*context));
    EXPECT_EQ(MixAdmissionTestPeer::remaining(*context), 0);
    EXPECT_EQ(MixAdmissionTestPeer::free(*context), 0);
    for (int t = 0; t < 3; ++t)
        EXPECT_FALSE(MixAdmissionTestPeer::consume(*context, t));
}

TEST_P(UnifiedAdmission, SingleOwnerFullChipPreservesOffsetsAbove63) {
    const size_t words = (reg_offset(RegId::COND) + sizeof(uint64_t)) / sizeof(uint64_t);
    registers.resize(72 * words);
    MixAdmissionTestPeer::single_owner(*context, registers.data(), words, 24);
    mixed.init(7, 24);
    enqueue_ss();
    MixAdmissionTestPeer::admit(*context, 0);
    EXPECT_EQ(mixed.slot.next_block_idx.load(), 24);
    ASSERT_TRUE(MixAdmissionTestPeer::consume(*context, 0));
    for (int c = 0; c < 72; ++c)
        EXPECT_EQ(MixAdmissionTestPeer::placed(*context, c), &mixed.slot);
    EXPECT_TRUE(MixAdmissionTestPeer::idle(*context));
}

TEST_P(UnifiedAdmission, SerialAssignmentPublishesExactUnevenOwnerPools) {
    for (int owners : {1, 3, MAX_AICPU_THREADS}) {
        for (int clusters : {23, 24}) {
            ASSERT_TRUE(MixAdmissionTestPeer::assign_serial(*context, owners, clusters));
            uint64_t free = (uint64_t{1} << clusters) - 1;
            EXPECT_EQ(MixAdmissionTestPeer::free(*context), free);
            for (int owner = 0; owner < owners; ++owner) {
                int count = 0;
                for (int ci = owner; ci < clusters; ci += owners) {
                    free &= ~(uint64_t{1} << ci);
                    ++count;
                }
                EXPECT_EQ(MixAdmissionTestPeer::claim_assigned(*context, owner), count);
                EXPECT_EQ(MixAdmissionTestPeer::free(*context), free);
            }
        }
    }
}

TEST_P(UnifiedAdmission, OrdinaryCubeAndVectorShareAnOwnedCluster) {
    mixed.init(1, 1);
    mixed.slot.task_attrs = {};
    pure.init(2, 1, 2);
    ASSERT_TRUE(sched->ready_queues[0].push_shaped(&mixed.slot, ResourceShape::AIC, mixed.descriptor.task_id));
    LaneMask observed = LaneMask::none();
    SchedulerContextTestPeer::leaf(*context, ResourceShape::AIC, false, false, observed);
    EXPECT_EQ(MixAdmissionTestPeer::placed(*context, 0), &mixed.slot);
    ASSERT_TRUE(sched->ready_queues[1].push_shaped(&pure.slot, ResourceShape::AIV, pure.descriptor.task_id));
    SchedulerContextTestPeer::leaf(*context, ResourceShape::AIV, false, false, observed);
    EXPECT_EQ(MixAdmissionTestPeer::placed(*context, 1), &pure.slot);
    MixAdmissionTestPeer::release_unused(*context, 0);
    EXPECT_EQ(MixAdmissionTestPeer::free(*context), 6);
}

TEST_P(UnifiedAdmission, IdleScanReturnsOnlyWhollyIdleOwnedClusters) {
    ASSERT_TRUE(MixAdmissionTestPeer::assign_serial(*context, 1, 3));
    ASSERT_EQ(MixAdmissionTestPeer::claim_assigned(*context, 0), 3);
    MixAdmissionTestPeer::toggle_running(*context, 0, 0);
    MixAdmissionTestPeer::pending(*context, 0, 4, true);
    MixAdmissionTestPeer::release_unused(*context, 0);
    EXPECT_EQ(MixAdmissionTestPeer::free(*context), 4);
    MixAdmissionTestPeer::release_unused(*context, 0);
    EXPECT_EQ(MixAdmissionTestPeer::free(*context), 4);
    MixAdmissionTestPeer::toggle_running(*context, 0, 0);
    MixAdmissionTestPeer::release_unused(*context, 0);
    EXPECT_EQ(MixAdmissionTestPeer::free(*context), 5);
    MixAdmissionTestPeer::pending(*context, 0, 4, false);
    MixAdmissionTestPeer::release_unused(*context, 0);
    EXPECT_EQ(MixAdmissionTestPeer::free(*context), 7);
}

TEST_P(UnifiedAdmission, EmptyReadyScanDoesNotPreventLaterProducerPublication) {
    SchedulerContextTestPeer::ready(*context, false);
    EXPECT_EQ(MixAdmissionTestPeer::free(*context), 7);
    mixed.init(7, 1);
    mixed.slot.task_attrs = {};
    std::atomic<bool> publish{false};
    std::atomic<bool> published{false};
    bool pushed = false;
    std::thread producer([&] {
        while (!publish.load(std::memory_order_acquire)) {}
        pushed = sched->ready_queues[2].push_mix(&mixed.slot, mixed.descriptor.task_id);
        published.store(true, std::memory_order_release);
    });
    publish.store(true, std::memory_order_release);
    while (!published.load(std::memory_order_acquire)) {}
    producer.join();
    ASSERT_TRUE(pushed);
    SchedulerContextTestPeer::ready(*context, false);
    EXPECT_EQ(mixed.slot.next_block_idx.load(), 1);
    for (int lane = 0; lane < 3; ++lane)
        EXPECT_EQ(MixAdmissionTestPeer::placed(*context, lane), &mixed.slot);
}

TEST_P(UnifiedAdmission, EmptyRegularQueuesStillAdmitSsMix) {
    enqueue_ss();
    SchedulerContextTestPeer::ready(*context, false);
    EXPECT_EQ(mixed.slot.next_block_idx.load(), 3);
    for (int lane = 0; lane < 3; ++lane)
        EXPECT_EQ(MixAdmissionTestPeer::placed(*context, lane), &mixed.slot);
    ASSERT_TRUE(MixAdmissionTestPeer::consume(*context, 1));
    ASSERT_TRUE(MixAdmissionTestPeer::consume(*context, 2));
    EXPECT_TRUE(MixAdmissionTestPeer::idle(*context));
}

TEST_P(UnifiedAdmission, OrdinaryVectorStillUsesPendingAfterSsPublication) {
    const size_t words = (reg_offset(RegId::COND) + sizeof(uint64_t)) / sizeof(uint64_t);
    registers.resize(9 * words);
    MixAdmissionTestPeer::single_owner(*context, registers.data(), words, 3);
    enqueue_ss();
    SchedulerContextTestPeer::ready(*context, false);
    pure.init(2, 1, 2);
    ASSERT_TRUE(sched->ready_queues[1].push_shaped(&pure.slot, ResourceShape::AIV, pure.descriptor.task_id));
    SchedulerContextTestPeer::ready(*context, false);
    EXPECT_EQ(pure.slot.next_block_idx.load(), 0);
    EXPECT_EQ(SchedulerContextTestPeer::placed(*context, 1, true), nullptr);
    MixAdmissionTestPeer::acknowledge_running_payload(*context, 0, 1);
    SchedulerContextTestPeer::ready(*context, false);
    EXPECT_EQ(pure.slot.next_block_idx.load(), 1);
    EXPECT_EQ(SchedulerContextTestPeer::placed(*context, 1, true), &pure.slot);
    EXPECT_EQ(MixAdmissionTestPeer::placed(*context, 1), &mixed.slot);
}

TEST_P(UnifiedAdmission, EmptyRegularQueuesStillDispatchSsCube) {
    mixed.init(1, 1);
    ASSERT_TRUE(sched->ready_sync_queues[0].push_shaped(&mixed.slot, ResourceShape::AIC, mixed.descriptor.task_id));
    SchedulerContextTestPeer::ready(*context, false);
    EXPECT_EQ(mixed.slot.next_block_idx.load(), 1);
    EXPECT_EQ(MixAdmissionTestPeer::placed(*context, 0), &mixed.slot);
    EXPECT_EQ(SchedulerContextTestPeer::drain_pending(*context), 0);
}

TEST_P(UnifiedAdmission, ReservedUnpublishedQueueEntryIsNotDispatched) {
    mixed.init(1, 1);
    mixed.slot.task_attrs = {};
    auto &queue = sched->ready_queues[0];
    queue.enqueue_pos.store(1, std::memory_order_relaxed);
    queue.slots[0].slot_value = CapturedReadySlot(&mixed.slot, LaneMask::none());
    queue.slots[0].task_id_snapshot = mixed.descriptor.task_id;
    SchedulerContextTestPeer::ready(*context, false);
    EXPECT_EQ(mixed.slot.next_block_idx.load(), 0);
    EXPECT_EQ(queue.dequeue_pos.load(), 0);
    queue.slots[0].sequence.store(1, std::memory_order_release);
    SchedulerContextTestPeer::ready(*context, false);
    EXPECT_EQ(mixed.slot.next_block_idx.load(), 1);
    EXPECT_EQ(queue.dequeue_pos.load(), 1);
    EXPECT_EQ(MixAdmissionTestPeer::placed(*context, 0), &mixed.slot);
}

TEST_P(UnifiedAdmission, NonemptyContinuationRepopRequeuesBlockedMix) {
    mixed.init(3, 1);
    mixed.slot.task_attrs = {};
    pure.init(5, 1, 2);
    auto &queue = sched->ready_queues[2];
    ASSERT_TRUE(queue.push_mix(&mixed.slot, mixed.descriptor.task_id));
    ASSERT_TRUE(queue.push_mix(&pure.slot, pure.descriptor.task_id));
    LaneMask observed = LaneMask::none();
    SchedulerContextTestPeer::leaf(*context, ResourceShape::MIX, false, false, observed);
    EXPECT_EQ(mixed.slot.next_block_idx.load(), 1);
    EXPECT_EQ(pure.slot.next_block_idx.load(), 0);
    EXPECT_EQ(MixAdmissionTestPeer::placed(*context, 0), &mixed.slot);
    EXPECT_EQ(MixAdmissionTestPeer::placed(*context, 1), &mixed.slot);
    EXPECT_EQ(MixAdmissionTestPeer::placed(*context, 2), nullptr);
    EXPECT_EQ(queue.dequeue_pos.load(), 2);
    EXPECT_EQ(queue.size(), 1);
    EXPECT_EQ(queue.enqueue_pos.load(), 3);
}

TEST_P(UnifiedAdmission, NoCubeCandidateLeavesQueueHeadUntouched) {
    mixed.init(1, 1);
    mixed.slot.task_attrs = {};
    auto &queue = sched->ready_queues[0];
    ASSERT_TRUE(queue.push_shaped(&mixed.slot, ResourceShape::AIC, mixed.descriptor.task_id));
    SchedulerContextTestPeer::block_lane(*context, 0, &pure.slot);
    LaneMask observed = LaneMask::none();
    SchedulerContextTestPeer::leaf(*context, ResourceShape::AIC, false, false, observed);
    EXPECT_EQ(queue.dequeue_pos.load(), 0);
    EXPECT_EQ(mixed.slot.next_block_idx.load(), 0);
}

TEST_P(UnifiedAdmission, ZeroClaimStopsScanBeforeNextBatch) {
    mixed.init(1, 1);
    mixed.slot.task_attrs = {};
    mixed.slot.next_block_idx.store(1);
    pure.init(1, 1, 2);
    auto &queue = sched->ready_queues[0];
    ASSERT_TRUE(queue.push_shaped(&mixed.slot, ResourceShape::AIC, mixed.descriptor.task_id));
    ASSERT_TRUE(queue.push_shaped(&pure.slot, ResourceShape::AIC, pure.descriptor.task_id));
    LaneMask observed = LaneMask::none();
    SchedulerContextTestPeer::leaf(*context, ResourceShape::AIC, false, false, observed);
    EXPECT_EQ(queue.dequeue_pos.load(), 1);
    EXPECT_EQ(pure.slot.next_block_idx.load(), 0);
    SchedulerContextTestPeer::leaf(*context, ResourceShape::AIC, false, false, observed);
    EXPECT_EQ(queue.dequeue_pos.load(), 2);
    EXPECT_EQ(pure.slot.next_block_idx.load(), 1);
    EXPECT_EQ(MixAdmissionTestPeer::placed(*context, 0), &pure.slot);
}

TEST_P(UnifiedAdmission, PublishedPrefixSkipsExhaustedClaimAndKeepsRemainder) {
    mixed.init(2, 1);
    mixed.slot.task_attrs = {};
    mixed.slot.task_attrs.set_early_resolve(true);
    mixed.slot.next_block_idx.store(1);
    pure.init(2, 3, 2);
    pure.slot.task_attrs.set_early_resolve(true);
    auto &queue = sched->ready_queues[1];
    ASSERT_TRUE(queue.push_shaped(&mixed.slot, ResourceShape::AIV, mixed.descriptor.task_id));
    ASSERT_TRUE(queue.push_shaped(&pure.slot, ResourceShape::AIV, pure.descriptor.task_id));
    LaneMask observed = LaneMask::none();
    SchedulerContextTestPeer::leaf(*context, ResourceShape::AIV, false, false, observed);
    EXPECT_EQ(mixed.payload.published_block_count.load(), 0);
    EXPECT_EQ(pure.payload.published_block_count.load(), 2);
    EXPECT_EQ(pure.slot.next_block_idx.load(), 2);
    EXPECT_EQ(queue.size(), 1);
    EXPECT_EQ(MixAdmissionTestPeer::placed(*context, 1), &pure.slot);
    EXPECT_EQ(MixAdmissionTestPeer::placed(*context, 2), &pure.slot);
}

TEST_P(UnifiedAdmission, PublishedPrefixKeepsBlockedMixAndTaggedTail) {
    const size_t words = (reg_offset(RegId::COND) + sizeof(uint64_t)) / sizeof(uint64_t);
    registers.resize(9 * words);
    MixAdmissionTestPeer::single_owner(*context, registers.data(), words, 3);
    mixed.init(7, 1);
    mixed.slot.task_attrs = {};
    pure.init(6, 1, 2);
    pure.slot.task_attrs.set_early_resolve(true);
    TestTask tail, blocker;
    tail.init(6, 3, 3);
    tail.slot.task_attrs.set_early_resolve(true);
    blocker.init(1, 3, 4);
    for (int core : {0, 3, 6})
        SchedulerContextTestPeer::block_lane(*context, core, &blocker.slot);
    auto &queue = sched->ready_queues[2];
    ASSERT_TRUE(queue.push_mix(&mixed.slot, mixed.descriptor.task_id));
    ASSERT_TRUE(queue.push_mix(&pure.slot, pure.descriptor.task_id));
    ASSERT_TRUE(queue.push_mix(&tail.slot, tail.descriptor.task_id));
    LaneMask observed = LaneMask::none();
    SchedulerContextTestPeer::leaf(*context, ResourceShape::MIX, false, false, observed);
    EXPECT_EQ(mixed.slot.next_block_idx.load(), 0);
    EXPECT_EQ(pure.payload.published_block_count.load(), 1);
    EXPECT_EQ(tail.payload.published_block_count.load(), 2);
    EXPECT_EQ(tail.slot.next_block_idx.load(), 2);
    CapturedReadySlot remaining[2];
    ASSERT_EQ(queue.pop_shaped_batch(remaining, nullptr, 2, ResourceShape::MIX), 2);
    EXPECT_EQ(remaining[0].ptr(), &mixed.slot);
    EXPECT_TRUE(remaining[0].tag().is_full());
    EXPECT_EQ(remaining[1].ptr(), &tail.slot);
    EXPECT_FALSE(remaining[1].tag().has_aic());
    EXPECT_TRUE(remaining[1].tag().has_both_aiv());
}

TEST_P(UnifiedAdmission, PureSsUsesOneQueueHeadPerGeneration) {
    mixed.init(2, 3);
    pure.init(2, 1, 2);
    pure.slot.task_attrs.set_sync_start();
    sched->push_ready_routed(&mixed.slot);
    sched->push_ready_routed(&pure.slot);
    MixAdmissionTestPeer::admit(*context);
    EXPECT_EQ(mixed.slot.next_block_idx.load(), 3);
    EXPECT_EQ(pure.slot.next_block_idx.load(), 0);
    EXPECT_EQ(sched->ready_sync_queues[1].size(), 1);
    EXPECT_EQ(sched->ready_sync_queues[1].dequeue_pos.load(), 1);
    EXPECT_EQ(SchedulerContextTestPeer::drain_pending(*context), 0);
}

TEST_P(UnifiedAdmission, ReadyRoutingPartitionsSyncAttribute) {
    mixed.init(1, 1);
    pure.init(1, 1, 2);
    sched->push_ready_routed(&mixed.slot);
    sched->push_ready_routed(&pure.slot);
    EXPECT_EQ(sched->ready_sync_queues[0].size(), 1);
    EXPECT_EQ(sched->ready_queues[0].size(), 1);
}
TEST_P(UnifiedAdmission, TwoSyncVectorTasksPublishInSuccessiveGenerations) {
    mixed.init(2, 1);
    pure.init(2, 1, 2);
    pure.slot.task_attrs.set_sync_start();
    sched->push_ready_routed(&mixed.slot);
    sched->push_ready_routed(&pure.slot);
    LaneMask observed = LaneMask::none();
    SchedulerContextTestPeer::sync_leaf(*context, observed);
    EXPECT_EQ(pure.slot.next_block_idx.load(), 0);
    SchedulerContextTestPeer::sync_leaf(*context, observed);
    EXPECT_EQ(mixed.slot.next_block_idx.load(), 1);
    EXPECT_EQ(pure.slot.next_block_idx.load(), 1);
    EXPECT_EQ(sched->ready_sync_queues[1].size(), 0);
    EXPECT_EQ(SchedulerContextTestPeer::drain_pending(*context), 0);
}
TEST_P(UnifiedAdmission, FirstPublicationSurvivesSecondReservation) {
    mixed.init(2, 1);
    pure.init(2, 3, 2);
    pure.slot.task_attrs.set_sync_start();
    sched->push_ready_routed(&mixed.slot);
    sched->push_ready_routed(&pure.slot);
    LaneMask observed = LaneMask::none();
    SchedulerContextTestPeer::sync_leaf(*context, observed);
    EXPECT_EQ(mixed.slot.next_block_idx.load(), 1);
    EXPECT_EQ(pure.slot.next_block_idx.load(), 0);
    SchedulerContextTestPeer::sync_leaf(*context, observed);
    EXPECT_EQ(pure.slot.next_block_idx.load(), 3);
    EXPECT_EQ(SchedulerContextTestPeer::drain_pending(*context), 0);
    EXPECT_EQ(MixAdmissionTestPeer::placed(*context, 1), &mixed.slot);
    EXPECT_NE(MixAdmissionTestPeer::published_token(*context, 1), 0u);
}

INSTANTIATE_TEST_SUITE_P(
    ThreeOwners, UnifiedAdmission, testing::Combine(testing::Values(7), testing::Values(false), testing::Values(false))
);

TEST_P(UnifiedAdmission, OneVectorLeavesTwentyThreeCompletePositionsBeforeItsFin) {
    const size_t words = (reg_offset(RegId::COND) + sizeof(uint64_t)) / sizeof(uint64_t);
    registers.resize(72 * words);
    MixAdmissionTestPeer::wide(*context, registers.data(), words);
    pure.init(2, 1, 1);
    sched->push_ready_routed(&pure.slot);
    LaneMask observed = LaneMask::none();
    SchedulerContextTestPeer::leaf(*context, ResourceShape::AIV, false, false, observed);
    ASSERT_EQ(pure.slot.next_block_idx.load(), 1);
    ASSERT_EQ(MixAdmissionTestPeer::placed(*context, 24), &pure.slot);
    EXPECT_FALSE(MixAdmissionTestPeer::scan_pending(*context, 0));
    EXPECT_EQ(__builtin_popcount(MixAdmissionTestPeer::free_word(*context, 0)), 23);
    EXPECT_EQ(__builtin_popcountll(MixAdmissionTestPeer::free(*context)), 23);
    mixed.init(7, 23);
    enqueue_ss();
    MixAdmissionTestPeer::admit(*context, 1);
    ASSERT_EQ(mixed.slot.next_block_idx.load(), 23);
    for (int t = 0; t < 3; ++t)
        ASSERT_TRUE(MixAdmissionTestPeer::consume(*context, t));
    int count = 0;
    for (int core = 0; core < 72; ++core)
        count += MixAdmissionTestPeer::placed(*context, core) == &mixed.slot;
    EXPECT_EQ(count, 69);
    EXPECT_EQ(MixAdmissionTestPeer::placed(*context, 24), &pure.slot);
    EXPECT_TRUE(MixAdmissionTestPeer::idle(*context));
}

TEST_P(UnifiedAdmission, OrdinaryBatchReturnsUnusedGrantAfterExhaustedTail) {
    const size_t words = (reg_offset(RegId::COND) + sizeof(uint64_t)) / sizeof(uint64_t);
    registers.resize(72 * words);
    MixAdmissionTestPeer::wide(*context, registers.data(), words);
    pure.init(2, 1, 1);
    mixed.init(2, 1, 2);
    mixed.slot.task_attrs = {};
    mixed.slot.next_block_idx.store(1);
    auto &queue = sched->ready_queues[static_cast<int>(ResourceShape::AIV)];
    ASSERT_TRUE(queue.push_shaped(&pure.slot, ResourceShape::AIV, pure.descriptor.task_id));
    ASSERT_TRUE(queue.push_shaped(&mixed.slot, ResourceShape::AIV, mixed.descriptor.task_id));
    MixAdmissionTestPeer::regular_vector(*context, 0);
    EXPECT_EQ(pure.slot.next_block_idx.load(), 1);
    EXPECT_TRUE(MixAdmissionTestPeer::scan_pending(*context, 0));
    EXPECT_EQ(__builtin_popcount(MixAdmissionTestPeer::free_word(*context, 0)), 22);
    MixAdmissionTestPeer::release_unused(*context, 0);
    EXPECT_EQ(__builtin_popcount(MixAdmissionTestPeer::free_word(*context, 0)), 23);
    EXPECT_FALSE(MixAdmissionTestPeer::scan_pending(*context, 0));
}

TEST_P(UnifiedAdmission, FullWidthWaitsForFinAndPublishesAllSeventyTwoSubtasksOnce) {
    const size_t words = (reg_offset(RegId::COND) + sizeof(uint64_t)) / sizeof(uint64_t);
    registers.resize(72 * words);
    MixAdmissionTestPeer::wide(*context, registers.data(), words);
    pure.init(2, 1, 1);
    sched->push_ready_routed(&pure.slot);
    LaneMask observed = LaneMask::none();
    SchedulerContextTestPeer::leaf(*context, ResourceShape::AIV, false, false, observed);
    ASSERT_EQ(MixAdmissionTestPeer::placed(*context, 24), &pure.slot);
    mixed.init(7, 24);
    enqueue_ss();
    MixAdmissionTestPeer::admit(*context, 1);
    EXPECT_TRUE(MixAdmissionTestPeer::waiting(*context));
    EXPECT_EQ(mixed.slot.next_block_idx.load(), 0);
    for (int t = 0; t < 3; ++t)
        EXPECT_FALSE(MixAdmissionTestPeer::consume(*context, t));
    MixAdmissionTestPeer::poll_one(*context, 0, 24, false);
    MixAdmissionTestPeer::admit(*context, 1);
    EXPECT_EQ(mixed.slot.next_block_idx.load(), 0);
    MixAdmissionTestPeer::poll_one(*context, 0, 24, true);
    MixAdmissionTestPeer::admit(*context, 0);
    ASSERT_EQ(mixed.slot.next_block_idx.load(), 24);
    for (int t : {2, 0, 1}) {
        ASSERT_TRUE(MixAdmissionTestPeer::consume(*context, t));
        EXPECT_FALSE(MixAdmissionTestPeer::consume(*context, t));
    }
    for (int core = 0; core < 72; ++core)
        EXPECT_EQ(MixAdmissionTestPeer::placed(*context, core), &mixed.slot);
    EXPECT_TRUE(MixAdmissionTestPeer::idle(*context));
}

TEST_P(UnifiedAdmission, FailedReservationRestoresExactlyTheBitsTakenAcrossOwners) {
    MixAdmissionTestPeer::rollback_race(*context);
}

TEST_P(UnifiedAdmission, OrdinaryOwnerRacesReservationRollbackAndWaitProtection) {
    const size_t words = (reg_offset(RegId::COND) + sizeof(uint64_t)) / sizeof(uint64_t);
    registers.resize(72 * words);
    MixAdmissionTestPeer::wide(*context, registers.data(), words);
    ASSERT_EQ(MixAdmissionTestPeer::plan_whole_chip(*context), 0);
    const uint32_t initially_free = MixAdmissionTestPeer::free_word(*context, 0);

    // Owner 1 takes a vector after the SS snapshot. Its ACK opens the pending slot.
    pure.init(2, 1, 1);
    sched->push_ready_routed(&pure.slot);
    MixAdmissionTestPeer::regular_vector(*context, 1);
    ASSERT_EQ(MixAdmissionTestPeer::placed(*context, 26), &pure.slot);
    MixAdmissionTestPeer::acknowledge_running_payload(*context, 1, 1);
    ASSERT_TRUE(MixAdmissionTestPeer::acquire_one(*context, 1, LaneMask::aiv0(), 1));
    mixed.init(7, 24);
    enqueue_ss();

    std::atomic<bool> at_start{false}, go{false};
    bool cube_acquired = false;
    std::thread ordinary([&] {
        at_start.store(true, std::memory_order_release);
        while (!go.load(std::memory_order_acquire))
            std::this_thread::yield();
        cube_acquired = MixAdmissionTestPeer::acquire_one(*context, 1, LaneMask::aic(), 0);
    });
    while (!at_start.load(std::memory_order_acquire))
        std::this_thread::yield();
    go.store(true, std::memory_order_release);
    bool stale_plan_reserved = MixAdmissionTestPeer::reserve_plan(*context);
    MixAdmissionTestPeer::admit(*context, 2);
    ordinary.join();

    ASSERT_FALSE(stale_plan_reserved);
    ASSERT_TRUE(MixAdmissionTestPeer::waiting(*context));
    ASSERT_EQ(mixed.slot.next_block_idx.load(), 0);
    EXPECT_EQ(MixAdmissionTestPeer::free_word(*context, 0), initially_free);
    EXPECT_EQ(MixAdmissionTestPeer::free_word(*context, 1), initially_free & ~(cube_acquired ? 3u : 2u));
    EXPECT_EQ(MixAdmissionTestPeer::protected_word(*context, 1), initially_free);
    EXPECT_FALSE(MixAdmissionTestPeer::acquire_one(*context, 1, LaneMask::aiv0(), 1));
    EXPECT_FALSE(MixAdmissionTestPeer::acquire_one(*context, 1, LaneMask::aic(), 0));

    MixAdmissionTestPeer::release_unused(*context, 1);
    MixAdmissionTestPeer::admit(*context, 1);
    EXPECT_EQ(mixed.slot.next_block_idx.load(), 0);
    MixAdmissionTestPeer::poll_one(*context, 1, 26, true);
    MixAdmissionTestPeer::admit(*context, 1);
    ASSERT_EQ(mixed.slot.next_block_idx.load(), 24);
    for (int t = 0; t < 3; ++t)
        ASSERT_TRUE(MixAdmissionTestPeer::consume(*context, t));
    for (int core = 0; core < 72; ++core)
        EXPECT_EQ(MixAdmissionTestPeer::placed(*context, core), &mixed.slot);
    EXPECT_TRUE(MixAdmissionTestPeer::idle(*context));
}

TEST_P(UnifiedAdmission, EmptyEarlyPassSeesLaterPublicationAndHonorsRegularReady) {
    EXPECT_EQ(MixAdmissionTestPeer::ordinary_early(*context, 0), 0);
    pure.init(2, 1, 1);
    pure.payload.early_dispatch_state.store(EARLY_DISPATCH_STAGING);
    auto &early = sched->early_dispatch_queues[static_cast<int>(ResourceShape::AIV)];
    ASSERT_TRUE(early.push_shaped(&pure.slot, ResourceShape::AIV, pure.descriptor.task_id));
    TestTask regular;
    regular.init(1, 1, 2);
    auto &ready = sched->ready_queues[static_cast<int>(ResourceShape::AIC)];
    ASSERT_TRUE(ready.push_shaped(&regular.slot, ResourceShape::AIC, regular.descriptor.task_id));
    EXPECT_EQ(MixAdmissionTestPeer::ordinary_early(*context, 0), 0);
    EXPECT_EQ(early.size(), 1u);
    CapturedReadySlot removed;
    ASSERT_EQ(ready.pop_shaped_batch(&removed, nullptr, 1, ResourceShape::AIC), 1);
    EXPECT_EQ(MixAdmissionTestPeer::ordinary_early(*context, 0), 1);
    EXPECT_EQ(early.size(), 0u);
    EXPECT_EQ(pure.slot.next_block_idx.load(), 1);
}

TEST_P(UnifiedAdmission, HiddenReadyRequestKeepsAnEarlyCandidateInItsQueue) {
    pure.init(2, 1, 1);
    pure.payload.early_dispatch_state.store(EARLY_DISPATCH_STAGING);
    auto &early = sched->early_dispatch_queues[static_cast<int>(ResourceShape::AIV)];
    ASSERT_TRUE(early.push_shaped(&pure.slot, ResourceShape::AIV, pure.descriptor.task_id));
    ASSERT_TRUE(MixAdmissionTestPeer::hold(*context, 2));
    enqueue_ss();
    MixAdmissionTestPeer::admit(*context, 1);
    ASSERT_TRUE(MixAdmissionTestPeer::waiting(*context));
    ASSERT_EQ(sched->ready_sync_queues[2].size(), 0);
    EXPECT_EQ(MixAdmissionTestPeer::ordinary_early(*context, 0), 0);
    EXPECT_EQ(early.dequeue_pos.load(), 0);
    MixAdmissionTestPeer::release_unused(*context, 2);
    MixAdmissionTestPeer::admit(*context, 2);
    ASSERT_EQ(mixed.slot.next_block_idx.load(), 3);
    EXPECT_EQ(MixAdmissionTestPeer::ordinary_early(*context, 0), 0);
    EXPECT_EQ(early.dequeue_pos.load(), 0);
    EXPECT_EQ(pure.slot.next_block_idx.load(), 0);
}

TEST_P(UnifiedAdmission, WaitCompletesOnAnUnprotectedPeerFinWhileFirstChoiceIsStillRunning) {
    const size_t words = (reg_offset(RegId::COND) + sizeof(uint64_t)) / sizeof(uint64_t);
    registers.resize(72 * words);
    MixAdmissionTestPeer::wide(*context, registers.data(), words);
    TestTask other;
    pure.init(2, 1, 1);
    other.init(2, 1, 2);
    for (auto *slot : {&pure.slot, &other.slot}) {
        sched->push_ready_routed(slot);
        MixAdmissionTestPeer::regular_vector(*context, slot == &pure.slot ? 0 : 1);
    }
    ASSERT_EQ(MixAdmissionTestPeer::placed(*context, 24), &pure.slot);
    ASSERT_EQ(MixAdmissionTestPeer::placed(*context, 26), &other.slot);
    mixed.init(7, 23);
    enqueue_ss();
    MixAdmissionTestPeer::admit(*context, 2);
    EXPECT_EQ(mixed.slot.next_block_idx.load(), 0);
    EXPECT_EQ(MixAdmissionTestPeer::protected_word(*context, 0) & 7, 7u);
    EXPECT_EQ(MixAdmissionTestPeer::protected_word(*context, 1) & 7, 0u);
    MixAdmissionTestPeer::poll_one(*context, 1, 26, true);
    MixAdmissionTestPeer::admit(*context, 1);
    ASSERT_EQ(mixed.slot.next_block_idx.load(), 23);
    for (int t = 0; t < 3; ++t)
        ASSERT_TRUE(MixAdmissionTestPeer::consume(*context, t));
    EXPECT_EQ(MixAdmissionTestPeer::placed(*context, 24), &pure.slot);
    EXPECT_EQ(MixAdmissionTestPeer::placed(*context, 26), &mixed.slot);
}

TEST(ReadyLanePool, DelayedOwnerInitializationPreservesPublishedProtection) {
    ReadyLanePool<3, 24> pool;
    pool.initialize(0, 1);
    ASSERT_EQ(pool.take_for_ss(0, 0, 3), 3u);
    pool.protect(1, 0, 3);
    pool.initialize(1, 1);
    EXPECT_EQ(pool.snapshot(1, 0), (uint64_t{3} << 32) | 7);
    EXPECT_EQ(pool.snapshot(0, 0), 4u);
    EXPECT_EQ(pool.snapshot(2, 0), 0u);
}
