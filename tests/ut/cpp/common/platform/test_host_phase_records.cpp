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
#include <unistd.h>

#include <atomic>
#include <cstddef>
#include <cstdio>
#include <fstream>
#include <new>
#include <string>
#include <thread>
#include <vector>

#include "host/host_phase_records.h"

using simpler::dfx::HostPhaseRecordStore;

namespace {

size_t count_lines(const std::string &path) {
    std::ifstream in(path);
    size_t n = 0;
    for (std::string line; std::getline(in, line);) {
        if (!line.empty()) n++;
    }
    return n;
}

void record_n(
    HostPhaseRecordPool *pool, HostPhaseKind kind, uint32_t n, uint64_t first_start = 0, uint32_t thread_id = 0
) {
    for (uint32_t i = 0; i < n; ++i) {
        const uint64_t start = first_start + i;
        host_phase_pool_append(pool, kind, start, start + 1, i, i, thread_id);
    }
}

TEST(HostPhaseRecords, DisarmedPoolCollectsNothing) {
    HostPhaseRecordStore store;
    HostPhaseRecordPool *pool = store.arm(/*collect_records=*/false);
    EXPECT_EQ(pool, nullptr);
    EXPECT_FALSE(store.armed());

    // The producer's record call must tolerate a null pool — that is the common
    // configuration, where only the per-kind counters run.
    record_n(pool, HostPhaseKind::OrchRecordSubTask, 10);
    store.finish(0, /*invocation_id=*/9);
    EXPECT_TRUE(store.records().empty());
    EXPECT_EQ(store.total_records(), 0u);
    EXPECT_EQ(store.dropped_records(), 0u);
}

TEST(HostPhaseRecords, RecordsSurviveInOrderWithinOneBuffer) {
    HostPhaseRecordStore store;
    HostPhaseRecordPool *pool = store.arm(true);
    ASSERT_NE(pool, nullptr);

    record_n(pool, HostPhaseKind::OrchRecordSubTask, 337, /*first_start=*/1000);
    store.finish(0, /*invocation_id=*/9);

    const auto records = store.records();
    ASSERT_EQ(records.size(), 337u);
    EXPECT_EQ(store.total_records(), 337u);
    EXPECT_EQ(store.dropped_records(), 0u);
    for (size_t i = 0; i < records.size(); ++i) {
        EXPECT_EQ(records[i].start_ns, 1000u + i) << "record " << i << " out of rotation order";
        EXPECT_EQ(records[i].kind, static_cast<uint32_t>(HostPhaseKind::OrchRecordSubTask));
    }
}

TEST(HostPhaseRecords, SlotsSpanEveryBufferInOrder) {
    HostPhaseRecordStore store;
    HostPhaseRecordPool *pool = store.arm(true);
    ASSERT_NE(pool, nullptr);

    // Exactly fills the pool, so the last record lands without a drop and every
    // buffer reports itself full.
    const uint32_t capacity = static_cast<uint32_t>(HostPhaseRecordStore::capacity());
    record_n(pool, HostPhaseKind::OrchGraphSubmit, capacity, /*first_start=*/1);
    store.finish(0, /*invocation_id=*/9);

    const auto records = store.records();
    ASSERT_EQ(records.size(), capacity);
    EXPECT_EQ(store.dropped_records(), 0u);
    EXPECT_EQ(store.total_records(), capacity);
    for (uint32_t i = 0; i < capacity; ++i) {
        EXPECT_EQ(records[i].start_ns, 1u + i) << "record " << i << " out of start-time order";
    }
}

TEST(HostPhaseRecords, ConcurrentProducersLoseNothingAndDoNotOverlapSlots) {
    // The pool has one writer per producer thread of a pass -- a Graph workload
    // records one thread per concurrent Definition plus the submitting one. Each
    // record must land exactly once, which is what makes the slot claim, rather
    // than a lock, sufficient.
    constexpr uint32_t kThreads = 8;
    constexpr uint32_t kPerThread = 500;
    HostPhaseRecordStore store;
    HostPhaseRecordPool *pool = store.arm(true);
    ASSERT_NE(pool, nullptr);
    ASSERT_GE(HostPhaseRecordStore::capacity(), static_cast<size_t>(kThreads) * kPerThread);

    std::atomic<bool> go{false};
    std::vector<std::thread> producers;
    for (uint32_t t = 0; t < kThreads; ++t) {
        producers.emplace_back([&, t]() {
            while (!go.load(std::memory_order_acquire)) {}
            // Disjoint start_ns ranges, so a lost or duplicated record is visible
            // as a hole or a repeat rather than needing per-thread bookkeeping.
            record_n(pool, HostPhaseKind::OrchRecordSubTask, kPerThread, /*first_start=*/1 + t * kPerThread, t);
        });
    }
    go.store(true, std::memory_order_release);
    for (auto &producer : producers)
        producer.join();
    store.finish(0, /*invocation_id=*/9);

    const auto records = store.records();
    ASSERT_EQ(records.size(), static_cast<size_t>(kThreads) * kPerThread);
    EXPECT_EQ(store.dropped_records(), 0u);
    std::vector<uint64_t> starts;
    starts.reserve(records.size());
    for (const auto &record : records)
        starts.push_back(record.start_ns);
    // records() sorts, so every value in the union must appear exactly once.
    for (size_t i = 0; i < starts.size(); ++i) {
        EXPECT_EQ(starts[i], 1u + i) << "record " << i << " lost, duplicated, or out of order";
    }
}

TEST(HostPhaseRecords, OverflowDropsAndCountsWithoutLosingEarlierRecords) {
    HostPhaseRecordStore store;
    HostPhaseRecordPool *pool = store.arm(true);
    ASSERT_NE(pool, nullptr);

    const uint32_t capacity = static_cast<uint32_t>(HostPhaseRecordStore::capacity());
    const uint32_t overshoot = 25;
    record_n(pool, HostPhaseKind::OrchRecordSubTask, capacity + overshoot, /*first_start=*/1);
    store.finish(0, /*invocation_id=*/9);

    const auto records = store.records();
    EXPECT_EQ(records.size(), capacity);
    EXPECT_EQ(store.total_records(), capacity + overshoot);
    EXPECT_EQ(store.dropped_records(), overshoot);
    // Dropping is tail-loss, not corruption: the retained prefix is intact.
    ASSERT_FALSE(records.empty());
    EXPECT_EQ(records.front().start_ns, 1u);
    EXPECT_EQ(records.back().start_ns, static_cast<uint64_t>(capacity));
}

TEST(HostPhaseRecords, AStoreReusingAnAddressDoesNotInheritACachedLane) {
    // A producer caches its buffer and offset against the identity of the pass it
    // claimed them for. That identity must be unique across the process, not
    // counted per pool: there is a store per DeviceRunner, so a new store can
    // occupy a destroyed one's address, and a per-pool counter would then hand it
    // an identity the cached lane already matches. The producer would resume the
    // dead store's offset -- appending past the records it wrote, or, once that
    // offset is at a buffer boundary, dropping everything.
    //
    // Placement-new over one buffer is what makes the address reuse certain rather
    // than left to the stack allocator, which is why this reproduced only in CI.
    alignas(HostPhaseRecordStore) static std::byte storage[sizeof(HostPhaseRecordStore)];
    auto *at_address = reinterpret_cast<HostPhaseRecordStore *>(storage);

    new (at_address) HostPhaseRecordStore();
    HostPhaseRecordPool *pool = at_address->arm(true);
    ASSERT_NE(pool, nullptr);
    record_n(pool, HostPhaseKind::OrchRecordSubTask, 100, /*first_start=*/1);
    at_address->finish(0, /*invocation_id=*/9);
    ASSERT_EQ(at_address->records().size(), 100u);
    at_address->~HostPhaseRecordStore();

    // Same address, brand-new store, its first pass.
    new (at_address) HostPhaseRecordStore();
    pool = at_address->arm(true);
    ASSERT_NE(pool, nullptr);
    record_n(pool, HostPhaseKind::OrchRecordSubTask, 100, /*first_start=*/1000);
    at_address->finish(0, /*invocation_id=*/9);

    const auto records = at_address->records();
    EXPECT_EQ(records.size(), 100u) << "the new store inherited the dead one's producer lane";
    EXPECT_EQ(at_address->dropped_records(), 0u);
    for (size_t i = 0; i < records.size(); ++i) {
        EXPECT_EQ(records[i].start_ns, 1000u + i) << "record " << i << " is not from this store's pass";
    }
    at_address->~HostPhaseRecordStore();
}

TEST(HostPhaseRecords, ReArmClearsThePreviousPass) {
    HostPhaseRecordStore store;
    HostPhaseRecordPool *pool = store.arm(true);
    ASSERT_NE(pool, nullptr);
    record_n(pool, HostPhaseKind::OrchRecordSubTask, 50);
    store.finish(7, /*invocation_id=*/9);
    EXPECT_EQ(store.records().size(), 50u);
    EXPECT_EQ(store.submitted_tasks(), 7u);

    pool = store.arm(true);
    ASSERT_NE(pool, nullptr);
    EXPECT_TRUE(store.records().empty());
    EXPECT_EQ(store.total_records(), 0u);
    EXPECT_FALSE(store.finished());
    record_n(pool, HostPhaseKind::OrchGraphSubmit, 3);
    store.finish(3, /*invocation_id=*/9);
    EXPECT_EQ(store.records().size(), 3u);
}

TEST(HostPhaseRecords, ProducerThreadIdSurvivesTheArtifactStore) {
    HostPhaseRecordStore store;
    HostPhaseRecordPool *pool = store.arm(true);
    ASSERT_NE(pool, nullptr);

    record_n(pool, HostPhaseKind::OrchRecordSubTask, 1, /*first_start=*/10, /*thread_id=*/101);
    record_n(pool, HostPhaseKind::OrchGraphSubmit, 1, /*first_start=*/20, /*thread_id=*/202);
    store.finish(1, /*invocation_id=*/9);

    const auto records = store.records();
    ASSERT_EQ(records.size(), 2u);
    EXPECT_EQ(records[0].thread_id, 101u);
    EXPECT_EQ(records[1].thread_id, 202u);
}

TEST(HostPhaseRecords, SubmitProjectionKeepsOnlyTaskSubmittingKinds) {
    HostPhaseRecordStore store;
    HostPhaseRecordPool *pool = store.arm(true);
    ASSERT_NE(pool, nullptr);

    // Mirrors one qwen decode pass: the three task-submitting kinds sum to the
    // pass's total_tasks, and the sub-operations do not count.
    record_n(pool, HostPhaseKind::OrchSubmitTask, 5);
    record_n(pool, HostPhaseKind::OrchAllocTensors, 2);
    record_n(pool, HostPhaseKind::OrchRecordSubTask, 277);
    record_n(pool, HostPhaseKind::OrchGraphSubmit, 40);
    record_n(pool, HostPhaseKind::OrchBuildDefinition, 1);
    record_n(pool, HostPhaseKind::BindHostOrch, 1);
    store.finish(47, /*invocation_id=*/9);

    EXPECT_EQ(store.records().size(), 326u);
    EXPECT_EQ(store.submit_records().size(), 47u);
    EXPECT_EQ(store.submitted_tasks(), 47u);
    for (const auto &record : store.submit_records()) {
        EXPECT_TRUE(host_phase_kind_submits_task(static_cast<HostPhaseKind>(record.kind)));
    }
}

TEST(HostPhaseRecords, EveryKindHasItsOwnName) {
    for (uint32_t i = 0; i < kHostPhaseKindCount; ++i) {
        const char *name = host_phase_kind_name(static_cast<HostPhaseKind>(i));
        EXPECT_STRNE(name, "unknown") << "kind " << i << " has no name";
        for (uint32_t j = i + 1; j < kHostPhaseKindCount; ++j) {
            EXPECT_STRNE(name, host_phase_kind_name(static_cast<HostPhaseKind>(j)))
                << "kinds " << i << " and " << j << " share a name";
        }
    }
    EXPECT_STREQ(host_phase_kind_name(HostPhaseKind::Count), "unknown");
}

TEST(HostPhaseRecords, APassIsWrittenAtMostOnceAndTheNextPassAppends) {
    // Both the device-run teardown and a run that stops before launch write the
    // artifact unconditionally, so the store -- not its callers -- is what keeps a
    // pass from appearing twice.
    // This test source is compiled into one executable per runtime, and ctest
    // runs them concurrently; the pid keeps their scratch files from colliding.
    const std::string path =
        std::string(testing::TempDir()) + "/host_phase_records_write_once." + std::to_string(getpid()) + ".jsonl";
    std::remove(path.c_str());

    HostPhaseRecordStore store;
    HostPhaseRecordPool *pool = store.arm(true);
    ASSERT_NE(pool, nullptr);
    record_n(pool, HostPhaseKind::OrchRecordSubTask, 4);
    store.finish(4, /*invocation_id=*/11);

    EXPECT_EQ(store.write_records_jsonl(path), 0);
    EXPECT_EQ(store.write_records_jsonl(path), 0);
    EXPECT_EQ(count_lines(path), 1u) << "a second write of the same pass must append nothing";

    pool = store.arm(true);
    ASSERT_NE(pool, nullptr);
    record_n(pool, HostPhaseKind::OrchGraphSubmit, 2);
    store.finish(2, /*invocation_id=*/12);
    EXPECT_EQ(store.write_records_jsonl(path), 0);
    EXPECT_EQ(count_lines(path), 2u) << "a --rounds run must still get one line per pass";

    std::remove(path.c_str());
}

}  // namespace
