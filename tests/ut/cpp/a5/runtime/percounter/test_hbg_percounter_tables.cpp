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

#include <gtest/gtest.h>

#include <cstdint>
#include <ios>
#include <utility>
#include <vector>

#include "percounter/percounter_plan.h"
#include "scheduler/scheduler_layout.h"

namespace {

// --- the claim ------------------------------------------------------------

namespace pcx = simpler::hbg::percounter;

// A group's ticket hands out its share of a list, and a draw yields a range of
// the share's indices. These tests stand in for the atomic the device uses:
// `scheduler_gm_fetch_add(ticket, batch)` supplies bases `batch` apart, and what
// has to hold is that the ranges tile the share in order and that none reaches
// past its end.

// Successive bases must give ascending, non-overlapping ranges. If they ever
// went backwards a lane could reach a task whose producers are still pending
// and spin on a counter that will be set only after it -- a hang, with the
// blame landing on the fanin spin.
TEST(PercounterClaim, SuccessiveDrawsGiveAscendingDisjointRanges) {
    const uint64_t order_count = 37;
    for (uint64_t batch : {uint64_t{1}, uint64_t{4}, uint64_t{8}}) {
        uint64_t previous_end = 0;
        for (uint64_t base = 0;; base += batch) {
            uint64_t first = 0;
            uint64_t last = 0;
            if (!pcx::percounter_claim_range(base, batch, order_count, &first, &last)) break;
            EXPECT_EQ(first, previous_end) << "batch=" << batch << " base=" << base;
            EXPECT_GT(last, first);
            previous_end = last;
        }
        EXPECT_EQ(previous_end, order_count) << "batch=" << batch << " left positions unclaimed";
    }
}

// Every position is covered by exactly one range. A gap strands a task forever;
// an overlap runs one twice. Neither shows up as anything but a hang or a wrong
// result much later.
TEST(PercounterClaim, RangesCoverEveryPositionExactlyOnce) {
    for (uint64_t batch : {uint64_t{1}, uint64_t{3}, uint64_t{8}, uint64_t{64}}) {
        for (uint64_t order_count : {uint64_t{0}, uint64_t{1}, uint64_t{7}, uint64_t{36}, uint64_t{100}}) {
            std::vector<int> claimed(static_cast<size_t>(order_count), 0);
            for (uint64_t base = 0;; base += batch) {
                uint64_t first = 0;
                uint64_t last = 0;
                if (!pcx::percounter_claim_range(base, batch, order_count, &first, &last)) break;
                ASSERT_LE(last, order_count);
                for (uint64_t i = first; i < last; ++i)
                    ++claimed[static_cast<size_t>(i)];
            }
            for (uint64_t i = 0; i < order_count; ++i) {
                EXPECT_EQ(claimed[static_cast<size_t>(i)], 1)
                    << "batch=" << batch << " order_count=" << order_count << " index=" << i;
            }
        }
    }
}

// The last real range straddles the end of the list and must be cut to it; the
// draw counter keeps rising afterwards, so every later base has to be refused.
// Without either, a lane takes a task id out of an uninitialised slot.
TEST(PercounterClaim, ClampsTheStraddlingRangeAndRefusesEveryBasePastTheEnd) {
    uint64_t first = 0;
    uint64_t last = 0;
    ASSERT_TRUE(pcx::percounter_claim_range(4, 8, 6, &first, &last));
    EXPECT_EQ(first, 4u);
    EXPECT_EQ(last, 6u) << "range must be cut to the list, not run past it";

    EXPECT_FALSE(pcx::percounter_claim_range(6, 8, 6, &first, &last));
    EXPECT_FALSE(pcx::percounter_claim_range(~uint64_t{0}, 8, 6, &first, &last));
    // An empty list admits nothing, including the first draw.
    EXPECT_FALSE(pcx::percounter_claim_range(0, 8, 0, &first, &last));
    // A zero batch would make a lane draw forever without advancing.
    EXPECT_FALSE(pcx::percounter_claim_range(0, 0, 6, &first, &last));
    EXPECT_FALSE(pcx::percounter_claim_range(0, 8, 6, nullptr, &last));
    EXPECT_FALSE(pcx::percounter_claim_range(0, 8, 6, &first, nullptr));
}

// The groups' shares must partition the list: a position no group owns strands
// its task, one two groups own runs twice. Within a share the positions must
// rise, or a group's ticket would hand out a task before one it depends on.
TEST(PercounterClaim, GroupSharesPartitionTheListInOrder) {
    for (uint32_t groups = 1; groups <= pcx::PERCOUNTER_TICKET_CAPACITY; ++groups) {
        for (uint64_t order_count : {uint64_t{0}, uint64_t{1}, uint64_t{3}, uint64_t{4}, uint64_t{37}, uint64_t{257}}) {
            std::vector<int> owned(static_cast<size_t>(order_count), 0);
            for (uint32_t group = 0; group < groups; ++group) {
                const uint64_t size = pcx::percounter_group_size(order_count, group, groups);
                uint64_t previous = 0;
                for (uint64_t index = 0; index < size; ++index) {
                    const uint64_t position = pcx::percounter_group_position(index, group, groups);
                    ASSERT_LT(position, order_count) << "groups=" << groups << " group=" << group << " index=" << index;
                    if (index != 0) EXPECT_GT(position, previous);
                    previous = position;
                    ++owned[static_cast<size_t>(position)];
                }
            }
            for (uint64_t p = 0; p < order_count; ++p) {
                EXPECT_EQ(owned[static_cast<size_t>(p)], 1)
                    << "groups=" << groups << " order_count=" << order_count << " position=" << p;
            }
        }
    }
}

// --- placement from the active mask ---------------------------------------

// A single-slot mask names a core type and goes in that type's list; any
// multi-slot mask is a MIX task and goes in neither. The four legal MIX shapes
// are every mask with two or three of the three slots set, and the enumeration
// below is exhaustive over the 3-bit space, so a shape cannot be added here
// without the rest of the pipeline being asked about it.
TEST(PercounterPlacement, ClassifiesEveryThreeBitMask) {
    uint8_t type = 0xFF;
    EXPECT_EQ(pcx::task_placement_from_active_mask(0x0, &type), pcx::TaskPlacement::UNSUPPORTED) << "empty";
    EXPECT_EQ(pcx::task_placement_from_active_mask(0x4, &type), pcx::TaskPlacement::UNSUPPORTED) << "AIV1 alone";

    ASSERT_EQ(pcx::task_placement_from_active_mask(0x1, &type), pcx::TaskPlacement::AIC);
    EXPECT_EQ(type, 0);
    ASSERT_EQ(pcx::task_placement_from_active_mask(0x2, &type), pcx::TaskPlacement::AIV);
    EXPECT_EQ(type, 1);

    for (const uint8_t mask : {uint8_t{0x3}, uint8_t{0x5}, uint8_t{0x6}, uint8_t{0x7}}) {
        EXPECT_EQ(pcx::task_placement_from_active_mask(mask, &type), pcx::TaskPlacement::MIX)
            << "mask 0x" << std::hex << static_cast<unsigned>(mask);
    }

    EXPECT_EQ(pcx::task_placement_from_active_mask(0x1, nullptr), pcx::TaskPlacement::UNSUPPORTED);
}

// --- layout --------------------------------------------------------------

namespace pc = simpler::hbg::percounter;

namespace {

// Aligned like the real region, which acquire_scheduler_state_storage hands out
// at SCHEDULER_STATE_ALIGNMENT.
class Region {
public:
    explicit Region(uint64_t size) :
        bytes_(size + SCHEDULER_STATE_ALIGNMENT) {
        storage_.resize(static_cast<size_t>(bytes_), 0);
        auto raw = reinterpret_cast<uintptr_t>(storage_.data());
        auto aligned = (raw + SCHEDULER_STATE_ALIGNMENT - 1) & ~(uintptr_t{SCHEDULER_STATE_ALIGNMENT} - 1);
        base_ = reinterpret_cast<void *>(aligned);
    }
    void *base() const { return base_; }

private:
    uint64_t bytes_;
    std::vector<uint8_t> storage_;
    void *base_ = nullptr;
};

bool is_aligned(uint64_t offset, uint64_t alignment) { return (offset & (alignment - 1)) == 0; }

}  // namespace

TEST(PercounterLayout, SegmentsAreAlignedAndOrdered) {
    const uint64_t order_count[2] = {3, 5};
    pc::PercounterLayout layout{};
    ASSERT_TRUE(pc::plan_layout(/*task_count=*/8, /*block_total=*/8, /*edge_count=*/11, order_count, &layout));

    EXPECT_EQ(layout.task_count, 8u);
    EXPECT_EQ(layout.edge_count, 11u);
    EXPECT_TRUE(is_aligned(layout.control_offset, alignof(pc::PercounterControl)));
    EXPECT_TRUE(is_aligned(layout.tickets_offset, alignof(pc::PercounterTicket)));
    EXPECT_TRUE(is_aligned(layout.counters_offset, alignof(pc::PercounterCounter)));
    EXPECT_TRUE(is_aligned(layout.total_size, SCHEDULER_STATE_ALIGNMENT));

    // Every segment must fit before the next one starts.
    EXPECT_GE(layout.tickets_offset, layout.control_offset + sizeof(pc::PercounterControl));
    EXPECT_GE(layout.counters_offset + 8 * sizeof(pc::PercounterCounter), layout.counters_offset);
    EXPECT_LE(layout.counters_offset + 8 * sizeof(pc::PercounterCounter), layout.total_size);
}

// The measured reason the whole array is padded: the atomic unit serialises per
// cache line, so two counters in one line cost double. A change that packs them
// stays correct and quietly runs 15x slower, which is exactly the kind of
// regression a test has to catch.
TEST(PercounterLayout, CountersAreOnePerCacheLine) {
    const uint64_t order_count[2] = {2, 2};
    pc::PercounterLayout layout{};
    ASSERT_TRUE(pc::plan_layout(4, 4, 0, order_count, &layout));
    EXPECT_EQ(sizeof(pc::PercounterCounter), 64u);
    EXPECT_EQ(layout.counters_offset % 64u, 0u);
}

// Every reader of the progress counter's line queues behind the atomics the
// lanes aim at it: `lane_error` read beside it on paged_attention Case1 cost 919
// cycles a task against 94 uncontended. Packing them together costs nothing a
// correctness test can see, which is why this one exists.
TEST(PercounterControl, TheOnlyFieldLanesWriteOwnsItsCacheLine) {
    EXPECT_EQ(offsetof(pc::PercounterControl, resolved_task_count) % 64u, 0u);
    // The high-frequency readers that must stay off its line.
    for (size_t reader : {
             offsetof(pc::PercounterControl, lane_error),
             offsetof(pc::PercounterControl, fanin_offset),
             offsetof(pc::PercounterControl, fanin_addr_offset),
         }) {
        EXPECT_NE(reader / 64u, offsetof(pc::PercounterControl, resolved_task_count) / 64u);
    }
}

TEST(PercounterLayout, RejectsOrderCountsLargerThanTheGraph) {
    pc::PercounterLayout layout{};
    const uint64_t too_many[2] = {3, 3};  // 6 placements for 4 tasks
    EXPECT_FALSE(pc::plan_layout(4, 4, 0, too_many, &layout));
    const uint64_t one_too_many[2] = {5, 0};
    EXPECT_FALSE(pc::plan_layout(4, 4, 0, one_too_many, &layout));
}

TEST(PercounterLayout, RejectsNullOut) {
    const uint64_t order_count[2] = {0, 0};
    EXPECT_FALSE(pc::plan_layout(0, 0, 0, order_count, nullptr));
}

TEST(PercounterLayout, HandlesAnEmptyGraph) {
    const uint64_t order_count[2] = {0, 0};
    pc::PercounterLayout layout{};
    ASSERT_TRUE(pc::plan_layout(0, 0, 0, order_count, &layout));
    EXPECT_GT(layout.total_size, 0u);  // the control header still exists
}

// --- table build ---------------------------------------------------------

namespace {

constexpr uint64_t kDeviceBase = 0x7000'0000'0000ull;

// An order entry packs (task id, block index, block count). The placement
// tests below are about which task landed where, so this reads the id back out
// and lets them index a list as if it still held bare ids. The encoding itself
// is tested separately.
struct OrderList {
    const uint64_t *entries;
    int64_t operator[](size_t i) const { return pc::percounter_order_task_id(entries[i]); }
    uint64_t raw(size_t i) const { return entries[i]; }
};

OrderList order_list(void *base, uint64_t offset) { return OrderList{pc::region_at<uint64_t>(base, offset)}; }

// A chain 0 -> 1 -> 2 -> 3, alternating core type, with task 0 already done.
struct ChainGraph {
    std::vector<std::vector<int32_t>> fanin{{}, {0}, {1}, {2}};
    std::vector<pc::TaskInput> tasks;

    ChainGraph() {
        for (size_t i = 0; i < fanin.size(); ++i) {
            tasks.push_back(
                pc::TaskInput{
                    static_cast<uint8_t>(i % 2), /*is_mix=*/false, /*inline_completed=*/i == 0,
                    static_cast<int32_t>(fanin[i].size()), fanin[i].empty() ? nullptr : fanin[i].data()
                }
            );
        }
    }
    uint64_t edges() const { return 3; }
};

}  // namespace

TEST(PercounterTables, PlacesEachTaskInItsCoreTypeListInAscendingOrder) {
    ChainGraph graph;
    // task 0 is inline-completed, so only 1,2,3 are placed: AIV gets 1 and 3,
    // AIC gets 2.
    const uint64_t order_count[2] = {1, 2};
    pc::PercounterLayout layout{};
    ASSERT_TRUE(pc::plan_layout(4, 4, graph.edges(), order_count, &layout));
    Region region(layout.total_size);

    pc::BuildResult result{};
    ASSERT_TRUE(pc::build_tables(graph.tasks.data(), 4, layout, region.base(), kDeviceBase, {}, &result))
        << pc::build_status_name(result.status);

    const OrderList aic = order_list(region.base(), layout.orders_offset[0]);
    const OrderList aiv = order_list(region.base(), layout.orders_offset[1]);
    EXPECT_EQ(aic[0], 2);
    EXPECT_EQ(aiv[0], 1);
    EXPECT_EQ(aiv[1], 3);
    // A chain's b-levels descend with its ids, so level order and id order
    // coincide here. The next two tests are the ones that tell them apart.
    EXPECT_LT(aiv[0], aiv[1]);
}

// Two independent chains emitted chain-major, which is how the orchestrator
// emits them: 0 -> 1 and 2 -> 3. Ids ascend 0,1,2,3 but the b-levels are
// 1,0,1,0, so id order walks one chain to its end before reaching the other
// chain's first task. That is what cost `paged_attention` Case1 2.1x against a
// scheduler whose dispatch order is dynamic. Level order must interleave them.
TEST(PercounterTables, OrdersByLevelRatherThanByChainMajorId) {
    std::vector<int32_t> chain_a{0};
    std::vector<int32_t> chain_b{2};
    const pc::TaskInput tasks[4] = {
        {0, false, false, 0, nullptr},         // chain A head, b-level 1
        {0, false, false, 1, chain_a.data()},  // chain A tail, b-level 0
        {0, false, false, 0, nullptr},         // chain B head, b-level 1
        {0, false, false, 1, chain_b.data()},  // chain B tail, b-level 0
    };
    const uint64_t order_count[2] = {4, 0};
    pc::PercounterLayout layout{};
    ASSERT_TRUE(pc::plan_layout(4, 4, 2, order_count, &layout));
    Region region(layout.total_size);

    pc::BuildResult result{};
    ASSERT_TRUE(pc::build_tables(tasks, 4, layout, region.base(), kDeviceBase, {}, &result))
        << pc::build_status_name(result.status);

    const OrderList aic = order_list(region.base(), layout.orders_offset[0]);
    // Both heads first, ties by id; then both tails.
    EXPECT_EQ(aic[0], 0);
    EXPECT_EQ(aic[1], 2);
    EXPECT_EQ(aic[2], 1);
    EXPECT_EQ(aic[3], 3);
}

// What the deadlock argument needs once the list is no longer in id order: a
// producer still precedes its consumer *within a list*. The b-level gives it,
// since A -> B implies blevel(A) >= blevel(B) + 1, but a comparator that lost
// the stability or compared the wrong key would not show up as anything except
// a lane spinning on a counter nobody will set.
TEST(PercounterTables, KeepsEveryProducerBeforeItsConsumerAfterTheLevelSort) {
    // Chain 0->1->2 beside chain 3->4. Levels are 2,1,0,1,0, so the long
    // chain's head outranks the short chain's head, and id order, depth order
    // and b-level order are all different here.
    std::vector<int32_t> p1{0};
    std::vector<int32_t> p2{1};
    std::vector<int32_t> p4{3};
    const pc::TaskInput tasks[5] = {
        {0, false, false, 0, nullptr},    // b-level 2
        {0, false, false, 1, p1.data()},  // b-level 1
        {0, false, false, 1, p2.data()},  // b-level 0
        {0, false, false, 0, nullptr},    // b-level 1, and a higher id
        {0, false, false, 1, p4.data()},  // b-level 0
    };
    const uint64_t order_count[2] = {5, 0};
    pc::PercounterLayout layout{};
    ASSERT_TRUE(pc::plan_layout(5, 5, 3, order_count, &layout));
    Region region(layout.total_size);

    pc::BuildResult result{};
    ASSERT_TRUE(pc::build_tables(tasks, 5, layout, region.base(), kDeviceBase, {}, &result))
        << pc::build_status_name(result.status);

    const OrderList aic = order_list(region.base(), layout.orders_offset[0]);
    EXPECT_EQ(aic[0], 0);
    EXPECT_EQ(aic[1], 1);
    EXPECT_EQ(aic[2], 3);
    EXPECT_EQ(aic[3], 2);
    EXPECT_EQ(aic[4], 4);

    // The invariant itself, read off the list rather than assumed from the
    // expectations above.
    std::vector<int> position(5, -1);
    for (int i = 0; i < 5; ++i)
        position[static_cast<size_t>(aic[i])] = i;
    const std::pair<int, int> edges[3] = {{0, 1}, {1, 2}, {3, 4}};  // producer, consumer
    for (const auto &edge : edges) {
        EXPECT_LT(position[static_cast<size_t>(edge.first)], position[static_cast<size_t>(edge.second)])
            << "producer " << edge.first << " must precede consumer " << edge.second;
    }
}

// --- the MIX list ---------------------------------------------------------

// A MIX task occupies two or three lanes of one cluster at once, so it belongs
// to neither core type's list. It gets a list and a ticket of its own, and the
// two per-core-type lists must come out exactly as they would if the MIX tasks
// were not in the graph at all -- a lane of either type never sees them.
TEST(PercounterTables, PlacesMixTasksInTheirOwnListAndNotInEither) {
    std::vector<int32_t> p0{0};
    std::vector<int32_t> p1{1};
    std::vector<int32_t> p2{2};
    // 0 -> 1 -> 2 -> 3, plus a lone sink 4. b-levels: 3, 2, 1, 0, 0.
    const pc::TaskInput tasks[5] = {
        {0, false, false, 0, nullptr},    // AIC, b-level 3
        {1, true, false, 1, p0.data()},   // MIX, b-level 2
        {1, false, false, 1, p1.data()},  // AIV, b-level 1
        {0, true, false, 1, p2.data()},   // MIX, b-level 0
        {1, false, false, 0, nullptr},    // AIV, b-level 0
    };
    const uint64_t order_count[2] = {1, 2};
    pc::PercounterLayout layout{};
    ASSERT_TRUE(pc::plan_layout(5, 5, 3, order_count, &layout, /*mix_order_count=*/2));
    Region region(layout.total_size);

    pc::BuildResult result{};
    ASSERT_TRUE(pc::build_tables(tasks, 5, layout, region.base(), kDeviceBase, {}, &result))
        << pc::build_status_name(result.status);

    const OrderList mix = order_list(region.base(), layout.mix_order_offset);
    EXPECT_EQ(mix[0], 1);  // b-level 2 before b-level 0
    EXPECT_EQ(mix[1], 3);

    // Neither per-core-type list learned about them, and the task whose core
    // type index a MIX task carries is unaffected by it.
    const OrderList aic = order_list(region.base(), layout.orders_offset[0]);
    const OrderList aiv = order_list(region.base(), layout.orders_offset[1]);
    EXPECT_EQ(aic[0], 0);
    EXPECT_EQ(aiv[0], 2);  // b-level 1 before b-level 0
    EXPECT_EQ(aiv[1], 4);

    // All three lists are drawn from, so all three count toward the exit test.
    const auto *control = pc::region_at<pc::PercounterControl>(region.base(), layout.control_offset);
    EXPECT_EQ(control->executable_task_count, 5u);
    EXPECT_EQ(control->mix_order_count, 2u);
    EXPECT_EQ(control->mix_order_offset, layout.mix_order_offset);
    EXPECT_EQ(control->mix_ticket_offset, layout.mix_ticket_offset);
}

// A MIX task with several blocks is several rendezvous of the same task, so it
// takes that many consecutive entries in the MIX list -- the same expansion the
// per-core-type lists get.
TEST(PercounterTables, ExpandsAMixTaskIntoOneEntryPerBlock) {
    // 0: MIX, 3 slots, 2 blocks.  1: AIC, 1 block, consuming it.
    std::vector<int32_t> p1{0};
    const pc::TaskInput tasks[2] = {
        {0, true, false, 0, nullptr, 2, 3},
        {0, false, false, 1, p1.data(), 1, 1},
    };
    const uint64_t order_count[2] = {1, 0};
    pc::PercounterLayout layout{};
    ASSERT_TRUE(pc::plan_layout(2, 3, 1, order_count, &layout, /*mix_order_count=*/2));
    Region region(layout.total_size);
    pc::BuildResult result{};
    ASSERT_TRUE(pc::build_tables(tasks, 2, layout, region.base(), kDeviceBase, {}, &result))
        << pc::build_status_name(result.status);

    const OrderList mix = order_list(region.base(), layout.mix_order_offset);
    EXPECT_EQ(mix[0], 0);
    EXPECT_EQ(mix[1], 0);
    EXPECT_EQ(pc::percounter_order_block_idx(mix.raw(0)), 0);
    EXPECT_EQ(pc::percounter_order_block_idx(mix.raw(1)), 1);
    EXPECT_EQ(pc::percounter_order_block_num(mix.raw(0)), 2);

    // Six subtasks: two blocks of three slots, so six adds reach published.
    const auto *counters = pc::region_at<pc::PercounterCounter>(region.base(), layout.counters_offset);
    EXPECT_EQ(counters[0].value, pc::percounter_counter_preset(6));

    // Every entry counts toward the exit test, MIX included.
    const auto *control = pc::region_at<pc::PercounterControl>(region.base(), layout.control_offset);
    EXPECT_EQ(control->executable_task_count, 3u);
}

// The one place a list entry and a unit of work are not the same thing. An
// order list hands out work that may be run one after another, and a cohort's
// blocks must be running at the same time, so they are not in it: the task
// holds a single arming entry and the blocks come from the cohort record.
TEST(PercounterTables, ASyncStartTaskTakesOneArmingEntryRatherThanOnePerBlock) {
    const pc::TaskInput tasks[1] = {{0, false, false, 0, nullptr, 4, 1, /*sync_start=*/true}};
    const uint64_t order_count[2] = {1, 0};
    pc::PercounterLayout layout{};
    ASSERT_TRUE(pc::plan_layout(1, 4, 0, order_count, &layout, 0, /*sync_cohort_count=*/1));
    EXPECT_NE(layout.sync_cohort_offset, 0u);
    Region region(layout.total_size);
    pc::BuildResult result{};
    ASSERT_TRUE(pc::build_tables(tasks, 1, layout, region.base(), kDeviceBase, {}, &result))
        << pc::build_status_name(result.status);

    const OrderList aic = order_list(region.base(), layout.orders_offset[0]);
    EXPECT_EQ(aic[0], 0);
    EXPECT_TRUE(pc::percounter_order_is_arming(aic.raw(0)));
    // The width travels on the entry: it is what the lane that arms the record
    // writes into it, and nothing else on the device knows it.
    EXPECT_EQ(pc::percounter_order_block_num(aic.raw(0)), 4);

    // Four blocks still publish four completions, so the counter is preset
    // against them exactly as an ordinary SPMD task's is.
    const auto *counters = pc::region_at<pc::PercounterCounter>(region.base(), layout.counters_offset);
    EXPECT_EQ(counters[0].value, pc::percounter_counter_preset(4));

    // **Blocks, not entries.** A lane folds its progress once per block it
    // runs, and the arming entry runs nothing. Counting the entry here would
    // end the run three blocks short of its own graph.
    const auto *control = pc::region_at<pc::PercounterControl>(region.base(), layout.control_offset);
    EXPECT_EQ(control->executable_task_count, 4u);
    EXPECT_EQ(control->sync_cohort_count, 1u);
    EXPECT_EQ(control->sync_cohort_offset, layout.sync_cohort_offset);

    // The queue starts with nothing open, and the open slot's task id starts
    // at a value no task has. A lane gates on that id to know the payload
    // behind an open slot is the one that belongs to it, and the region-wide
    // memset leaves a 0 here -- which is a real task id, and would make the
    // first cohort readable between taking the slot and filling it.
    const auto *queue = pc::region_at<pc::PercounterSyncQueue>(region.base(), layout.sync_cohort_offset);
    EXPECT_EQ(queue->open_generation, 0u);
    EXPECT_EQ(queue->open_task_id, -1);
    EXPECT_EQ(queue->released, 0u);
    EXPECT_EQ(queue->tail, 0u);

    // **Both regions have to reach the control block, and a zero offset is not
    // merely missing data.** `scheduler_state_at` resolves an offset against
    // the region base, and the control block itself sits at that base -- so a
    // zero offset silently aliases the queue, or the descriptor ring, onto the
    // control block, where `task_count` reads back as an odd generation and
    // the first store corrupts the header.
    EXPECT_NE(layout.sync_pending_offset, 0u);
    EXPECT_NE(layout.sync_cohort_offset, layout.sync_pending_offset);
    EXPECT_EQ(control->sync_cohort_offset, layout.sync_cohort_offset);
    EXPECT_EQ(control->sync_pending_offset, layout.sync_pending_offset);

    // The ring holds one descriptor and one claim count per sync-start task, so
    // a sequence number can never wrap, an enqueue can never find it full, and
    // no two cohorts share a claim count. Every count starts at zero.
    EXPECT_GE(layout.total_size, layout.sync_pending_offset + 2 * layout.sync_cohort_count * sizeof(uint64_t));
    const auto *ring = pc::region_at<uint64_t>(region.base(), layout.sync_pending_offset);
    for (uint64_t seq = 0; seq < layout.sync_cohort_count; ++seq)
        EXPECT_EQ(ring[layout.sync_cohort_count + seq], 0u);
}

// A sync-start MIX task is placed by its mask like every other task, so its
// arming entry sits in the MIX list -- but its rendezvous report to the cohort
// record rather than to the MIX done ticket, and the exit rule reads a count
// that must not include them.
TEST(PercounterTables, ASyncStartMixTaskArmsFromTheMixListAndCountsNoRendezvous) {
    const pc::TaskInput tasks[2] = {
        {0, true, false, 0, nullptr, 3, 3, /*sync_start=*/true},
        {0, true, false, 0, nullptr, 2, 3},
    };
    const uint64_t order_count[2] = {0, 0};
    pc::PercounterLayout layout{};
    ASSERT_TRUE(pc::plan_layout(2, 5, 0, order_count, &layout, /*mix_order_count=*/3, 1));
    Region region(layout.total_size);
    pc::BuildResult result{};
    ASSERT_TRUE(pc::build_tables(tasks, 2, layout, region.base(), kDeviceBase, {}, &result))
        << pc::build_status_name(result.status);

    const OrderList mix = order_list(region.base(), layout.mix_order_offset);
    EXPECT_EQ(mix[0], 0);
    EXPECT_TRUE(pc::percounter_order_is_arming(mix.raw(0)));
    EXPECT_EQ(pc::percounter_order_block_num(mix.raw(0)), 3);
    EXPECT_EQ(mix[1], 1);
    EXPECT_EQ(pc::percounter_order_block_idx(mix.raw(1)), 0);
    EXPECT_EQ(mix[2], 1);
    EXPECT_EQ(pc::percounter_order_block_idx(mix.raw(2)), 1);

    const auto *control = pc::region_at<pc::PercounterControl>(region.base(), layout.control_offset);
    // Three entries drawn, but only task 1's two rendezvous are counted by the
    // ticket the exit rule tests.
    EXPECT_EQ(control->mix_order_count, 3u);
    EXPECT_EQ(control->mix_rendezvous_count, 2u);
    EXPECT_EQ(control->executable_task_count, 5u);
}

// Without a sync-start task the two counts are the same number; they are
// separate fields because a sync-start MIX task makes them differ.
TEST(PercounterTables, WithoutSyncStartTheRendezvousCountIsTheMixListLength) {
    const pc::TaskInput tasks[1] = {{0, true, false, 0, nullptr, 2, 3}};
    const uint64_t order_count[2] = {0, 0};
    pc::PercounterLayout layout{};
    ASSERT_TRUE(pc::plan_layout(1, 2, 0, order_count, &layout, /*mix_order_count=*/2));
    EXPECT_EQ(layout.sync_cohort_count, 0u);
    EXPECT_EQ(layout.sync_cohort_offset, 0u);
    Region region(layout.total_size);
    ASSERT_TRUE(pc::build_tables(tasks, 1, layout, region.base(), kDeviceBase));

    const auto *control = pc::region_at<pc::PercounterControl>(region.base(), layout.control_offset);
    EXPECT_EQ(control->mix_rendezvous_count, control->mix_order_count);
    EXPECT_EQ(control->sync_cohort_count, 0u);
    EXPECT_EQ(control->sync_cohort_offset, 0u);
}

// A cohort of one would be released by its own commitment, and the orchestrator
// sets the attribute only above one block. The device code is written on that,
// so a shape that broke it must fail the bind rather than reach a lane.
TEST(PercounterTables, RejectsAOneBlockCohort) {
    const pc::TaskInput tasks[1] = {{0, false, false, 0, nullptr, 1, 1, /*sync_start=*/true}};
    const uint64_t order_count[2] = {1, 0};
    pc::PercounterLayout layout{};
    ASSERT_TRUE(pc::plan_layout(1, 1, 0, order_count, &layout, 0, 1));
    Region region(layout.total_size);
    pc::BuildResult result{};
    EXPECT_FALSE(pc::build_tables(tasks, 1, layout, region.base(), kDeviceBase, {}, &result));
    EXPECT_EQ(result.status, pc::BuildStatus::INVALID_ARGUMENT);
}

// The plan's cohort count is what sizes the record and what the exit rule
// compares against, so a disagreement with the tasks has to fail the bind.
TEST(PercounterTables, RejectsACohortCountThatDisagreesWithTheTasks) {
    const pc::TaskInput tasks[1] = {{0, false, false, 0, nullptr, 4, 1, /*sync_start=*/true}};
    const uint64_t order_count[2] = {1, 0};
    pc::PercounterLayout layout{};
    ASSERT_TRUE(pc::plan_layout(1, 4, 0, order_count, &layout, 0, /*sync_cohort_count=*/0));
    Region region(layout.total_size);
    pc::BuildResult result{};
    EXPECT_FALSE(pc::build_tables(tasks, 1, layout, region.base(), kDeviceBase, {}, &result));
    EXPECT_EQ(result.status, pc::BuildStatus::ORDER_COUNT_MISMATCH);
}

// The sentinel has to be a value no real block index can take, which is what
// the orchestrator's own INT16_MAX cap on `block_num * popcount(mask)` makes
// true -- and the widest legal entry has to stay distinguishable from it.
TEST(PercounterOrderEntry, TheArmingSentinelIsNoRealBlockIndex) {
    const uint64_t widest = pc::percounter_order_entry(0x7fffffff, 32766, 32767);
    EXPECT_FALSE(pc::percounter_order_is_arming(widest));
    const uint64_t arming = pc::percounter_order_entry(0x7fffffff, pc::PERCOUNTER_ORDER_BLOCK_IDX_ARM, 32767);
    EXPECT_TRUE(pc::percounter_order_is_arming(arming));
    EXPECT_EQ(pc::percounter_order_task_id(arming), 0x7fffffff);
    EXPECT_EQ(pc::percounter_order_block_num(arming), 32767);
}

// The ordinary case. `mix_order_count == 0` is what makes the claimer branch on
// the device fall through at no cost, so the zero has to reach the control
// block -- and nothing may be reserved for a list that does not exist.
TEST(PercounterTables, AMixFreeGraphReservesNothingForMix) {
    const uint64_t order_count[2] = {1, 1};
    pc::PercounterLayout layout_without{};
    ASSERT_TRUE(pc::plan_layout(2, 2, 0, order_count, &layout_without));
    EXPECT_EQ(layout_without.mix_order_count, 0u);
    EXPECT_EQ(layout_without.mix_order_offset, 0u);
    EXPECT_EQ(layout_without.mix_ticket_offset, 0u);
    EXPECT_EQ(layout_without.mix_signal_ready_offset, 0u);

    pc::PercounterLayout layout_with{};
    ASSERT_TRUE(pc::plan_layout(2, 2, 0, order_count, &layout_with, /*mix_order_count=*/0));
    EXPECT_EQ(layout_with.total_size, layout_without.total_size);

    const pc::TaskInput tasks[2] = {{0, false, false, 0, nullptr}, {1, false, false, 0, nullptr}};
    Region region(layout_without.total_size);
    ASSERT_TRUE(pc::build_tables(tasks, 2, layout_without, region.base(), kDeviceBase));
    const auto *control = pc::region_at<pc::PercounterControl>(region.base(), layout_without.control_offset);
    EXPECT_EQ(control->mix_order_count, 0u);
    EXPECT_EQ(control->mix_order_offset, 0u);
}

// The MIX ticket is separate storage, not a third slot in the per-core-type
// ticket array: a lane draws from its own type's ticket on every iteration and
// from the MIX one only when it is the claimer, so sharing a line would put the
// claimer's CAS in the way of every other lane's draw.
TEST(PercounterLayout, TheMixTicketOwnsItsOwnCacheLine) {
    const uint64_t order_count[2] = {1, 1};
    pc::PercounterLayout layout{};
    ASSERT_TRUE(pc::plan_layout(3, 3, 0, order_count, &layout, /*mix_order_count=*/1));

    EXPECT_TRUE(is_aligned(layout.mix_ticket_offset, alignof(pc::PercounterTicket)));
    EXPECT_TRUE(is_aligned(layout.mix_order_offset, alignof(int64_t)));
    EXPECT_GE(
        layout.mix_ticket_offset, layout.tickets_offset + pc::PERCOUNTER_CORE_TYPE_COUNT *
                                                              pc::PERCOUNTER_TICKET_CAPACITY *
                                                              sizeof(pc::PercounterTicket)
    );
    EXPECT_LE(
        layout.mix_ticket_offset + pc::PERCOUNTER_MIX_TICKET_COUNT * sizeof(pc::PercounterTicket), layout.total_size
    );
    // One word per cluster, sized by the capacity rather than by the cluster
    // count: the AICPU discovers that at bring-up, long after this plan.
    EXPECT_TRUE(is_aligned(layout.mix_signal_ready_offset, alignof(uint64_t)));
    EXPECT_LE(layout.mix_signal_ready_offset + SCHEDULER_CLUSTER_CAPACITY * sizeof(uint64_t), layout.total_size);
}

// The same guard the per-core-type lists get, and it has to catch both
// directions. An undercount leaves the last MIX tasks in no list at all; an
// overcount leaves a tail of zeros in the list, which reads as task 0 and would
// have a cluster gather for a task that has already run.
TEST(PercounterTables, RejectsAMixCountThatDisagreesWithTheTasks) {
    const uint64_t order_count[2] = {0, 0};
    // Two MIX tasks against a plan for one; then one MIX task against a plan
    // for two, where the inline-completed task leaves room for the second.
    const pc::TaskInput two_mix[2] = {{0, true, false, 0, nullptr}, {0, true, false, 0, nullptr}};
    const pc::TaskInput one_mix[2] = {{0, true, false, 0, nullptr}, {0, true, true, 0, nullptr}};
    const pc::TaskInput *const cases[2] = {two_mix, one_mix};
    const uint64_t planned_mix[2] = {1, 2};

    for (int c = 0; c < 2; ++c) {
        pc::PercounterLayout layout{};
        ASSERT_TRUE(pc::plan_layout(2, 2, 0, order_count, &layout, planned_mix[c])) << c;
        Region region(layout.total_size);
        pc::BuildResult result{};
        EXPECT_FALSE(pc::build_tables(cases[c], 2, layout, region.base(), kDeviceBase, {}, &result)) << c;
        EXPECT_EQ(result.status, pc::BuildStatus::ORDER_COUNT_MISMATCH) << c;
    }
}

// Inline-completed wins over is_mix: the task is done before the run starts, so
// no cluster may be told to gather for it.
TEST(PercounterTables, AnInlineCompletedMixTaskEntersNoList) {
    const pc::TaskInput tasks[2] = {{0, true, true, 0, nullptr}, {0, true, false, 0, nullptr}};
    const uint64_t order_count[2] = {0, 0};
    pc::PercounterLayout layout{};
    ASSERT_TRUE(pc::plan_layout(2, 2, 0, order_count, &layout, /*mix_order_count=*/1));
    Region region(layout.total_size);

    pc::BuildResult result{};
    ASSERT_TRUE(pc::build_tables(tasks, 2, layout, region.base(), kDeviceBase, {}, &result))
        << pc::build_status_name(result.status);

    const OrderList mix = order_list(region.base(), layout.mix_order_offset);
    EXPECT_EQ(mix[0], 1);
    const auto *counters = pc::region_at<pc::PercounterCounter>(region.base(), layout.counters_offset);
    EXPECT_EQ(counters[0].value, 1);
    const auto *control = pc::region_at<pc::PercounterControl>(region.base(), layout.control_offset);
    EXPECT_EQ(control->executable_task_count, 1u);
}

TEST(PercounterTables, FaninAddressesPointAtTheProducersCounter) {
    ChainGraph graph;
    const uint64_t order_count[2] = {1, 2};
    pc::PercounterLayout layout{};
    ASSERT_TRUE(pc::plan_layout(4, 4, graph.edges(), order_count, &layout));
    Region region(layout.total_size);
    ASSERT_TRUE(pc::build_tables(graph.tasks.data(), 4, layout, region.base(), kDeviceBase));

    const auto *fanin = pc::region_at<pc::PercounterFanin>(region.base(), layout.fanin_offset);
    const auto *addr = pc::region_at<uint64_t>(region.base(), layout.fanin_addr_offset);

    EXPECT_EQ(fanin[0].count, 0);
    for (int32_t consumer = 1; consumer <= 3; ++consumer) {
        ASSERT_EQ(fanin[consumer].count, 1);
        const uint64_t want =
            kDeviceBase + layout.counters_offset + static_cast<uint64_t>(consumer - 1) * sizeof(pc::PercounterCounter);
        EXPECT_EQ(addr[fanin[consumer].begin], want) << "consumer " << consumer;
    }
}

TEST(PercounterTables, PresetsInlineCompletedCountersAndLeavesTheRestAtZero) {
    ChainGraph graph;
    const uint64_t order_count[2] = {1, 2};
    pc::PercounterLayout layout{};
    ASSERT_TRUE(pc::plan_layout(4, 4, graph.edges(), order_count, &layout));
    Region region(layout.total_size);
    ASSERT_TRUE(pc::build_tables(graph.tasks.data(), 4, layout, region.base(), kDeviceBase));

    const auto *counters = pc::region_at<pc::PercounterCounter>(region.base(), layout.counters_offset);
    EXPECT_EQ(counters[0].value, 1u) << "an inline-completed task must not be waited on";
    EXPECT_EQ(counters[1].value, 0u);
    EXPECT_EQ(counters[2].value, 0u);
    EXPECT_EQ(counters[3].value, 0u);
}

// The arithmetic the whole completion path now rests on: a counter preset to
// `1 - total` is walked to published by exactly `total` adds of one, and reads
// as unpublished at every step before that. The ramp passes through 0, which
// is why the test cannot be "non-zero".
TEST(PercounterCounters, PresetIsWalkedToPublishedByExactlyTheSubtaskTotal) {
    for (int64_t total : {int64_t{1}, int64_t{2}, int64_t{3}, int64_t{24}, int64_t{96}}) {
        uint64_t value = pc::percounter_counter_preset(total);
        for (int64_t step = 0; step < total; ++step) {
            EXPECT_FALSE(pc::percounter_counter_published(value))
                << "total=" << total << " after " << step << " of " << total;
            value += 1;
        }
        EXPECT_TRUE(pc::percounter_counter_published(value)) << "total=" << total;
        EXPECT_EQ(value, pc::PERCOUNTER_COUNTER_PUBLISHED) << "total=" << total;
    }
}

// An overshoot must read published rather than hang. An equality test would
// turn a subtask that somehow ran twice into a consumer waiting forever, and a
// hang is the failure mode with nothing to read.
TEST(PercounterCounters, AnOvershootStillReadsPublished) {
    uint64_t value = pc::percounter_counter_preset(3);
    for (int i = 0; i < 5; ++i)
        value += 1;
    EXPECT_TRUE(pc::percounter_counter_published(value));
}

// A task of `block_num` blocks and `subtask_count` slots is preset against
// their product, so no lane has to know the total to publish it.
TEST(PercounterTables, PresetsEveryCounterAgainstItsSubtaskTotal) {
    // 0: AIC, 4 blocks.  1: AIV, 1 block.  2: MIX, 1 block, 3 slots.
    const pc::TaskInput tasks[3] = {
        {0, false, false, 0, nullptr, 4, 1},
        {1, false, false, 0, nullptr, 1, 1},
        {0, true, false, 0, nullptr, 1, 3},
    };
    const uint64_t order_count[2] = {4, 1};
    pc::PercounterLayout layout{};
    ASSERT_TRUE(pc::plan_layout(3, 6, 0, order_count, &layout, /*mix_order_count=*/1));
    Region region(layout.total_size);
    pc::BuildResult result{};
    ASSERT_TRUE(pc::build_tables(tasks, 3, layout, region.base(), kDeviceBase, {}, &result))
        << pc::build_status_name(result.status);

    const auto *counters = pc::region_at<pc::PercounterCounter>(region.base(), layout.counters_offset);
    EXPECT_EQ(counters[0].value, pc::percounter_counter_preset(4));
    EXPECT_EQ(counters[1].value, pc::percounter_counter_preset(1));
    EXPECT_EQ(counters[2].value, pc::percounter_counter_preset(3));
    for (int i = 0; i < 3; ++i)
        EXPECT_FALSE(pc::percounter_counter_published(counters[i].value)) << i;
}

// The three fields of an order entry survive the round trip, and the AIC list
// holds one entry per block with ascending indices.
TEST(PercounterTables, OrderEntriesCarryTheBlockIndexAndCount) {
    const pc::TaskInput tasks[1] = {{0, false, false, 0, nullptr, 4, 1}};
    const uint64_t order_count[2] = {4, 0};
    pc::PercounterLayout layout{};
    ASSERT_TRUE(pc::plan_layout(1, 4, 0, order_count, &layout));
    Region region(layout.total_size);
    ASSERT_TRUE(pc::build_tables(tasks, 1, layout, region.base(), kDeviceBase));

    const OrderList aic = order_list(region.base(), layout.orders_offset[0]);
    for (int32_t b = 0; b < 4; ++b) {
        EXPECT_EQ(aic[static_cast<size_t>(b)], 0) << "block " << b;
        EXPECT_EQ(pc::percounter_order_block_idx(aic.raw(static_cast<size_t>(b))), b);
        EXPECT_EQ(pc::percounter_order_block_num(aic.raw(static_cast<size_t>(b))), 4);
    }
}

// The widths the orchestrator's own INT16_MAX guard makes safe.
TEST(PercounterOrderEntry, PacksTheWidestLegalTripleWithoutCollision) {
    const int64_t task_id = 0x7fffffff;
    const int32_t block_num = 32767;
    const uint64_t entry = pc::percounter_order_entry(task_id, block_num - 1, block_num);
    EXPECT_EQ(pc::percounter_order_task_id(entry), task_id);
    EXPECT_EQ(pc::percounter_order_block_idx(entry), block_num - 1);
    EXPECT_EQ(pc::percounter_order_block_num(entry), block_num);
}

TEST(PercounterTables, ControlHeaderMirrorsThePlan) {
    ChainGraph graph;
    const uint64_t order_count[2] = {1, 2};
    pc::PercounterLayout layout{};
    ASSERT_TRUE(pc::plan_layout(4, 4, graph.edges(), order_count, &layout));
    Region region(layout.total_size);
    ASSERT_TRUE(pc::build_tables(graph.tasks.data(), 4, layout, region.base(), kDeviceBase));

    const auto *control = pc::region_at<pc::PercounterControl>(region.base(), layout.control_offset);
    EXPECT_EQ(control->task_count, 4u);
    EXPECT_EQ(control->edge_count, graph.edges());
    EXPECT_EQ(control->order_count[0], 1u);
    EXPECT_EQ(control->order_count[1], 2u);
    EXPECT_EQ(control->counters_offset, layout.counters_offset);
    EXPECT_EQ(control->tickets_offset, layout.tickets_offset);
}

// A lane reaches the task table and the callable table through these and
// nothing else: neither lives in the percounter region. A zero here is a lane
// that cannot materialize a payload.
TEST(PercounterTables, CarriesTheAddressesThatLiveOutsideTheRegion) {
    ChainGraph graph;
    const uint64_t order_count[2] = {1, 2};
    pc::PercounterLayout layout{};
    ASSERT_TRUE(pc::plan_layout(4, 4, graph.edges(), order_count, &layout));
    Region region(layout.total_size);

    pc::ExternalAddresses external{};
    external.graph_storage = 0x1234'5000ull;
    external.callable_table = 0x9876'0000ull;
    external.callable_count = 17;
    ASSERT_TRUE(pc::build_tables(graph.tasks.data(), 4, layout, region.base(), kDeviceBase, external));

    const auto *control = pc::region_at<pc::PercounterControl>(region.base(), layout.control_offset);
    EXPECT_EQ(control->graph_storage_address, external.graph_storage);
    EXPECT_EQ(control->callable_addresses_address, external.callable_table);
    EXPECT_EQ(control->callable_addresses_count, external.callable_count);
    EXPECT_EQ(control->dispatch_payloads_offset, layout.dispatch_payloads_offset);
    // The AICPU fills these after the handshake, so a freshly built region must
    // not claim a topology it cannot know.
    EXPECT_EQ(control->cluster_count, 0u);
    EXPECT_EQ(control->aiv_per_cluster, 0u);
}

TEST(PercounterLayout, ReservesOneDispatchPayloadPerLane) {
    const uint64_t order_count[2] = {1, 1};
    pc::PercounterLayout layout{};
    ASSERT_TRUE(pc::plan_layout(2, 2, 0, order_count, &layout));
    EXPECT_TRUE(is_aligned(layout.dispatch_payloads_offset, alignof(DispatchPayload)));
    EXPECT_LE(layout.dispatch_payloads_offset + SCHEDULER_WORKER_CAPACITY * sizeof(DispatchPayload), layout.total_size);
}

// The load-bearing assumption of the whole design: ascending task id is already
// a topological order, so a producer's id is always lower than its consumer's.
// If that ever stops holding, a lane spins forever on a counter nobody will set
// -- a device hang with no diagnosis. Checking it per edge turns that into a
// bind failure.
TEST(PercounterTables, RejectsAnEdgeThatDoesNotPointBackwards) {
    std::vector<int32_t> forward{2};  // task 1 waiting on task 2
    std::vector<int32_t> none{};
    const pc::TaskInput tasks[3] = {
        {0, false, false, 0, nullptr},
        {0, false, false, 1, forward.data()},
        {0, false, false, 0, nullptr},
    };
    const uint64_t order_count[2] = {3, 0};
    pc::PercounterLayout layout{};
    ASSERT_TRUE(pc::plan_layout(3, 3, 1, order_count, &layout));
    Region region(layout.total_size);

    pc::BuildResult result{};
    EXPECT_FALSE(pc::build_tables(tasks, 3, layout, region.base(), kDeviceBase, {}, &result));
    EXPECT_EQ(result.status, pc::BuildStatus::PRODUCER_NOT_BEFORE_CONSUMER);
    EXPECT_EQ(result.task_id, 1);
    EXPECT_EQ(result.fanin_index, 0);
}

TEST(PercounterTables, RejectsSelfEdges) {
    std::vector<int32_t> self{0};
    const pc::TaskInput tasks[1] = {{0, false, false, 1, self.data()}};
    const uint64_t order_count[2] = {1, 0};
    pc::PercounterLayout layout{};
    ASSERT_TRUE(pc::plan_layout(1, 1, 1, order_count, &layout));
    Region region(layout.total_size);

    pc::BuildResult result{};
    EXPECT_FALSE(pc::build_tables(tasks, 1, layout, region.base(), kDeviceBase, {}, &result));
    EXPECT_EQ(result.status, pc::BuildStatus::PRODUCER_NOT_BEFORE_CONSUMER);
}

TEST(PercounterTables, RejectsAProducerOutsideTheGraph) {
    std::vector<int32_t> bad{7};
    const pc::TaskInput tasks[2] = {{0, false, false, 0, nullptr}, {0, false, false, 1, bad.data()}};
    const uint64_t order_count[2] = {2, 0};
    pc::PercounterLayout layout{};
    ASSERT_TRUE(pc::plan_layout(2, 2, 1, order_count, &layout));
    Region region(layout.total_size);

    pc::BuildResult result{};
    EXPECT_FALSE(pc::build_tables(tasks, 2, layout, region.base(), kDeviceBase, {}, &result));
    EXPECT_EQ(result.status, pc::BuildStatus::PRODUCER_OUT_OF_RANGE);
}

TEST(PercounterTables, RejectsABadCoreType) {
    const pc::TaskInput tasks[1] = {{9, false, false, 0, nullptr}};
    const uint64_t order_count[2] = {1, 0};
    pc::PercounterLayout layout{};
    ASSERT_TRUE(pc::plan_layout(1, 1, 0, order_count, &layout));
    Region region(layout.total_size);

    pc::BuildResult result{};
    EXPECT_FALSE(pc::build_tables(tasks, 1, layout, region.base(), kDeviceBase, {}, &result));
    EXPECT_EQ(result.status, pc::BuildStatus::BAD_CORE_TYPE);
}

// A plan that disagrees with the graph must not half-fill the region: the
// device would read a list whose tail is still zeroed and dispatch task 0
// repeatedly.
TEST(PercounterTables, RejectsAPlanThatDisagreesWithTheTasks) {
    const pc::TaskInput tasks[2] = {{0, false, false, 0, nullptr}, {0, false, false, 0, nullptr}};
    const uint64_t too_few[2] = {1, 0};  // two AIC tasks, room for one
    pc::PercounterLayout layout{};
    ASSERT_TRUE(pc::plan_layout(2, 2, 0, too_few, &layout));
    Region region(layout.total_size);

    pc::BuildResult result{};
    EXPECT_FALSE(pc::build_tables(tasks, 2, layout, region.base(), kDeviceBase, {}, &result));
    EXPECT_EQ(result.status, pc::BuildStatus::ORDER_COUNT_MISMATCH);

    // A plan too LARGE passes plan_layout (it only bounds the total against the
    // graph) and has to be caught by the tail check, after the walk has placed
    // fewer tasks than promised. An inline-completed task is placed in no list,
    // which is the easy way to get there.
    const pc::TaskInput with_inline[2] = {{0, false, true, 0, nullptr}, {0, false, false, 0, nullptr}};
    const uint64_t too_many[2] = {2, 0};  // one placeable AIC task, room for two
    ASSERT_TRUE(pc::plan_layout(2, 2, 0, too_many, &layout));
    Region region2(layout.total_size);
    EXPECT_FALSE(pc::build_tables(with_inline, 2, layout, region2.base(), kDeviceBase, {}, &result));
    EXPECT_EQ(result.status, pc::BuildStatus::ORDER_COUNT_MISMATCH);
}

// plan_layout owns the "cannot possibly fit" check; build_tables owns the
// "does not actually match" one. Keeping them separate is why the case above
// has to construct its mismatch so carefully.
TEST(PercounterLayout, BoundsTheTotalPlacementNotEachList) {
    pc::PercounterLayout layout{};
    const uint64_t split[2] = {2, 2};  // 4 placements, 4 tasks: fine
    EXPECT_TRUE(pc::plan_layout(4, 4, 0, split, &layout));
    const uint64_t over[2] = {3, 2};  // 5 placements, 4 tasks: impossible
    EXPECT_FALSE(pc::plan_layout(4, 4, 0, over, &layout));
}

TEST(PercounterTables, RejectsAnEdgeCountThatDisagrees) {
    std::vector<int32_t> one{0};
    const pc::TaskInput tasks[2] = {{0, false, false, 0, nullptr}, {0, false, false, 1, one.data()}};
    const uint64_t order_count[2] = {2, 0};
    pc::PercounterLayout layout{};
    ASSERT_TRUE(pc::plan_layout(2, 2, /*edge_count=*/0, order_count, &layout));
    Region region(layout.total_size);

    pc::BuildResult result{};
    EXPECT_FALSE(pc::build_tables(tasks, 2, layout, region.base(), kDeviceBase, {}, &result));
    EXPECT_EQ(result.status, pc::BuildStatus::EDGE_COUNT_MISMATCH);
}

TEST(PercounterTables, BuildsAnEmptyGraph) {
    const uint64_t order_count[2] = {0, 0};
    pc::PercounterLayout layout{};
    ASSERT_TRUE(pc::plan_layout(0, 0, 0, order_count, &layout));
    Region region(layout.total_size);
    pc::BuildResult result{};
    EXPECT_TRUE(pc::build_tables(nullptr, 0, layout, region.base(), kDeviceBase, {}, &result))
        << pc::build_status_name(result.status);
}

}  // namespace
