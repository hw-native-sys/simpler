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

#include <numeric>
#include <vector>

#include "common/scheduler_die_partition.h"

using pto::a5::compute_cluster_owners;
using pto::a5::sched_die_bits_get;
using pto::a5::sched_die_bits_with;

namespace {

uint64_t pack(const std::vector<int32_t> &thread_die) {
    uint64_t bits = 0;
    for (size_t t = 0; t < thread_die.size(); ++t)
        bits = sched_die_bits_with(bits, static_cast<int32_t>(t), thread_die[t]);
    return bits;
}

// die0 first, then die1 — the shape physical_core_id yields, since CANN hands
// out block indices in ascending pcid order and the two AIC windows are
// [0, AICORE_PER_DIE) and [SUB_CORES_PER_DIE, ...).
std::vector<int8_t> split(int32_t die0, int32_t die1) {
    std::vector<int8_t> d(die0, 0);
    d.insert(d.end(), die1, 1);
    return d;
}

struct Outcome {
    std::vector<int32_t> owner;
    std::vector<int32_t> per_thread;  // clusters owned
    int32_t same_die{0};
};

Outcome run(const std::vector<int8_t> &cluster_die, const std::vector<int32_t> &thread_die) {
    Outcome o;
    o.owner.assign(cluster_die.size(), -1);
    o.per_thread.assign(thread_die.size(), 0);
    EXPECT_TRUE(compute_cluster_owners(
        static_cast<int32_t>(cluster_die.size()), static_cast<int32_t>(thread_die.size()), cluster_die.data(),
        pack(thread_die), o.owner.data()
    ));
    for (size_t ci = 0; ci < cluster_die.size(); ++ci) {
        const int32_t t = o.owner[ci];
        o.per_thread[t]++;
        if (cluster_die[ci] == thread_die[t]) o.same_die++;
    }
    return o;
}

}  // namespace

TEST(A5SchedulerDiePartition, PacksAndReadsBackPerThreadDie) {
    const uint64_t bits = pack({1, 1, 1, 0});
    EXPECT_EQ(sched_die_bits_get(bits, 0), 1);
    EXPECT_EQ(sched_die_bits_get(bits, 3), 0);
    // Never published: distinct from "die 0".
    EXPECT_EQ(sched_die_bits_get(bits, 4), -1);
    EXPECT_EQ(sched_die_bits_get(0, 0), -1);
}

// The measured Ascend950DT_9581 FG shape: 32 clusters split 16/16 by
// physical_core_id, ALLOWED_CPUS = [7, 5, 6, 3] -> phy [6, 4, 5, 2] -> die
// [1, 1, 1, 0].
TEST(A5SchedulerDiePartition, MatchesMeasuredFgShape) {
    const Outcome o = run(split(16, 16), {1, 1, 1, 0});
    EXPECT_EQ(o.per_thread, (std::vector<int32_t>{8, 8, 8, 8}));
    EXPECT_EQ(o.same_die, 24);  // 3 threads share 16 die-1 clusters; one goes cross

    // S3 is the only die-0 thread, so it holds die-0 clusters exclusively.
    for (int32_t ci = 0; ci < 8; ++ci)
        EXPECT_EQ(o.owner[ci], 3) << "cluster " << ci;
}

TEST(A5SchedulerDiePartition, KeepsQuotaExactAcrossUnevenDieSplits) {
    for (const auto &[d0, d1] :
         std::vector<std::pair<int32_t, int32_t>>{{16, 16}, {20, 12}, {12, 20}, {24, 8}, {8, 24}, {31, 1}, {1, 31}}) {
        const Outcome o = run(split(d0, d1), {1, 1, 1, 0});
        EXPECT_EQ(o.per_thread, (std::vector<int32_t>{8, 8, 8, 8})) << "split " << d0 << ":" << d1;
    }
}

TEST(A5SchedulerDiePartition, DegradesToBalancedSplitWhenEveryThreadSharesOneDie) {
    // No thread can be served cross-die-free for more than half the clusters,
    // so the result must merely stay balanced and no worse than round-robin.
    const Outcome o = run(split(16, 16), {0, 0, 0, 0});
    EXPECT_EQ(o.per_thread, (std::vector<int32_t>{8, 8, 8, 8}));
    EXPECT_EQ(o.same_die, 16);
}

TEST(A5SchedulerDiePartition, RejectsUnusableInput) {
    std::vector<int32_t> owner(8, -1);
    const std::vector<int8_t> die = split(4, 4);

    // One thread's die never published -> no partition, caller keeps round-robin.
    uint64_t partial = sched_die_bits_with(0, 0, 1);
    partial = sched_die_bits_with(partial, 1, 0);
    EXPECT_FALSE(compute_cluster_owners(8, 4, die.data(), partial, owner.data()));
    EXPECT_FALSE(compute_cluster_owners(8, 4, die.data(), 0, owner.data()));

    // Degenerate shapes.
    EXPECT_FALSE(compute_cluster_owners(0, 4, die.data(), pack({0, 0, 0, 0}), owner.data()));
    EXPECT_FALSE(compute_cluster_owners(8, 0, die.data(), pack({0, 0, 0, 0}), owner.data()));
    EXPECT_FALSE(compute_cluster_owners(8, 4, nullptr, pack({0, 0, 0, 0}), owner.data()));

    // A cluster die outside {0, 1} is a corrupt handshake report, not a hint.
    std::vector<int8_t> bad = die;
    bad[2] = 7;
    EXPECT_FALSE(compute_cluster_owners(8, 4, bad.data(), pack({0, 0, 0, 0}), owner.data()));
}

TEST(A5SchedulerDiePartition, HandlesFewerClustersThanThreads) {
    // Threads beyond the cluster count get a zero quota and must not steal one.
    std::vector<int32_t> owner(3, -1);
    const std::vector<int8_t> die = split(2, 1);
    ASSERT_TRUE(compute_cluster_owners(3, 4, die.data(), pack({0, 0, 1, 1}), owner.data()));
    std::vector<int32_t> count(4, 0);
    for (int32_t t : owner)
        count[t]++;
    EXPECT_EQ(std::accumulate(count.begin(), count.end(), 0), 3);
    for (int32_t c : count)
        EXPECT_LE(c, 1);
}

// Every scheduler thread runs this independently on the barrier-free init path
// and must land on the same partition without exchanging anything.
TEST(A5SchedulerDiePartition, IsDeterministic) {
    const std::vector<int8_t> die = split(16, 16);
    std::vector<int32_t> first(32, -1);
    ASSERT_TRUE(compute_cluster_owners(32, 4, die.data(), pack({1, 1, 1, 0}), first.data()));
    for (int i = 0; i < 4; ++i) {
        std::vector<int32_t> again(32, -1);
        ASSERT_TRUE(compute_cluster_owners(32, 4, die.data(), pack({1, 1, 1, 0}), again.data()));
        EXPECT_EQ(again, first);
    }
}
