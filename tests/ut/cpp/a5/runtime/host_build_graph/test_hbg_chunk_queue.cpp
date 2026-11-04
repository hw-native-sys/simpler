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

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <numeric>
#include <thread>
#include <vector>

#include "scheduler/scheduler_chunk_queue.h"

namespace {
struct ChunkFixture {
    SchedulerLinkedQueue control{};
    std::vector<int64_t> links;
    std::vector<SchedulerReadyChunk> chunks;
    SchedulerChunkQueueView queue;

    explicit ChunkFixture(uint32_t count) :
        links(count + 1, -1),
        chunks(count),
        queue{{&control, links.data(), count, count + 1, count}, chunks.data()} {
        reset();
    }
    void reset() {
        control.head = control.tail = queue.chain.sentinel;
        std::fill(links.begin(), links.end(), SCHEDULER_TASK_REF_INVALID);
        for (auto &chunk : chunks) {
            chunk.cursor = chunk.count = 0;
            for (auto &ref : chunk.refs)
                ref = SCHEDULER_TASK_REF_INVALID;
        }
    }
    SchedulerLinkedBatch prepare(const std::vector<int64_t> &refs) {
        SchedulerLinkedBatch batch{};
        EXPECT_EQ(scheduler_chunk_prepare(queue, refs.data(), refs.size(), &batch), SchedulerLinkedResult::SUCCESS);
        return batch;
    }
    void push(const std::vector<int64_t> &refs) {
        auto batch = prepare(refs);
        EXPECT_EQ(scheduler_linked_try_push(queue.chain, &batch, 32), SchedulerLinkedResult::SUCCESS);
    }
    SchedulerLinkedResult pop(int64_t *ref) {
        int64_t window[1]{};
        uint32_t count = 0;
        auto result = scheduler_chunk_window(queue, window, &count);
        if (result != SchedulerLinkedResult::SUCCESS) return result;
        SchedulerChunkCandidate candidate{};
        result = scheduler_chunk_peek(queue, window[0], &candidate);
        if (result != SchedulerLinkedResult::SUCCESS) return result;
        int64_t refs[1]{};
        result = scheduler_chunk_try_claim(queue, candidate, 1, refs, &count);
        *ref = refs[0];
        return result;
    }
};

TEST(SchedulerChunkQueueTest, PreservesOrderWithOneChunkWindowAndAppendAfterEmpty) {
    ChunkFixture fixture(6);
    int64_t ref = -1;
    EXPECT_EQ(fixture.pop(&ref), SchedulerLinkedResult::NOT_READY);
    fixture.push({4, 1, 2});
    fixture.push({0, 3});
    for (int64_t expected : {4, 1, 2, 0, 3}) {
        ASSERT_EQ(fixture.pop(&ref), SchedulerLinkedResult::SUCCESS);
        EXPECT_EQ(ref, expected);
    }
    EXPECT_EQ(fixture.pop(&ref), SchedulerLinkedResult::NOT_READY);
    fixture.push({5});
    ASSERT_EQ(fixture.pop(&ref), SchedulerLinkedResult::SUCCESS);
    EXPECT_EQ(ref, 5);
}

TEST(SchedulerChunkQueueTest, PrivatePreparedChunkDoesNotBlockOtherPublishers) {
    ChunkFixture fixture(4);
    auto paused = fixture.prepare({0, 1});
    fixture.push({2, 3});
    int64_t ref = -1;
    ASSERT_EQ(fixture.pop(&ref), SchedulerLinkedResult::SUCCESS);
    EXPECT_EQ(ref, 2);
    ASSERT_EQ(scheduler_linked_try_push(fixture.queue.chain, &paused), SchedulerLinkedResult::SUCCESS);
    for (int64_t expected : {3, 0, 1}) {
        ASSERT_EQ(fixture.pop(&ref), SchedulerLinkedResult::SUCCESS);
        EXPECT_EQ(ref, expected);
    }
}

TEST(SchedulerChunkQueueTest, ConsumersPassLaggingTailAndPublishersHelpAcrossExhaustedChunks) {
    ChunkFixture fixture(6);
    auto paused = fixture.prepare({0, 1, 2, 3});
    ASSERT_EQ(scheduler_gm_compare_exchange(fixture.links[6], INT64_C(-1), paused.first), -1);
    int64_t ref = -1;
    for (int64_t expected = 0; expected < 4; ++expected) {
        ASSERT_EQ(fixture.pop(&ref), SchedulerLinkedResult::SUCCESS);
        EXPECT_EQ(ref, expected);
    }
    EXPECT_EQ(fixture.control.tail, 6);
    fixture.push({4, 5});
    ASSERT_EQ(fixture.pop(&ref), SchedulerLinkedResult::SUCCESS);
    EXPECT_EQ(ref, 4);
}

TEST(SchedulerChunkQueueTest, WindowDistributesClaimsWithoutTakingOwnershipOfWholeChunks) {
    ChunkFixture fixture(8);
    for (int64_t id : {0, 2, 4, 6})
        fixture.push({id, id + 1});
    int64_t window[4]{};
    uint32_t count = 0;
    ASSERT_EQ(scheduler_chunk_window(fixture.queue, window, &count), SchedulerLinkedResult::SUCCESS);
    ASSERT_EQ(count, 4u);
    for (uint32_t index : {3, 1, 2, 0}) {
        SchedulerChunkCandidate candidate{};
        ASSERT_EQ(scheduler_chunk_peek(fixture.queue, window[index], &candidate), SchedulerLinkedResult::SUCCESS);
        int64_t refs[2]{};
        ASSERT_EQ(scheduler_chunk_try_claim(fixture.queue, candidate, 1, refs, &count), SchedulerLinkedResult::SUCCESS);
        EXPECT_EQ(count, 1u);
        EXPECT_EQ(refs[0], index * 2);
        EXPECT_EQ(fixture.chunks[index * 2].cursor, 1u);
    }
    EXPECT_EQ(fixture.control.head, 0);
    int64_t ref = -1;
    for (int64_t expected : {1, 3, 5, 7}) {
        ASSERT_EQ(fixture.pop(&ref), SchedulerLinkedResult::SUCCESS);
        EXPECT_EQ(ref, expected);
    }
}

TEST(SchedulerChunkQueueTest, StalePeekCannotReclaimAfterHeadAdvances) {
    ChunkFixture fixture(2);
    fixture.push({0});
    fixture.push({1});
    SchedulerChunkCandidate stale{};
    ASSERT_EQ(scheduler_chunk_peek(fixture.queue, 0, &stale), SchedulerLinkedResult::SUCCESS);
    int64_t ref = -1;
    ASSERT_EQ(fixture.pop(&ref), SchedulerLinkedResult::SUCCESS);
    ASSERT_EQ(fixture.pop(&ref), SchedulerLinkedResult::SUCCESS);
    EXPECT_EQ(fixture.control.head, 1);
    int64_t refs[1]{};
    uint32_t count = 99;
    EXPECT_EQ(scheduler_chunk_try_claim(fixture.queue, stale, 1, refs, &count), SchedulerLinkedResult::RETRY);
    EXPECT_EQ(count, 0u);
    EXPECT_EQ(refs[0], SCHEDULER_TASK_REF_INVALID);
}

TEST(SchedulerChunkQueueTest, CreditBoundsClaimsAndPlacementCanLeaveCandidateShared) {
    ChunkFixture fixture(4);
    fixture.push({0, 1, 2, 3});
    SchedulerChunkCandidate candidate{};
    ASSERT_EQ(scheduler_chunk_peek(fixture.queue, 0, &candidate), SchedulerLinkedResult::SUCCESS);
    EXPECT_EQ(fixture.chunks[0].cursor, 0u);
    int64_t refs[2]{};
    uint32_t count = 99;
    EXPECT_EQ(scheduler_chunk_try_claim(fixture.queue, candidate, 0, refs, &count), SchedulerLinkedResult::NOT_READY);
    EXPECT_EQ(fixture.chunks[0].cursor, 0u);
    ASSERT_EQ(scheduler_chunk_try_claim(fixture.queue, candidate, 2, refs, &count), SchedulerLinkedResult::SUCCESS);
    EXPECT_EQ(count, 2u);
    EXPECT_EQ(refs[0], 0);
    EXPECT_EQ(refs[1], 1);
    int64_t ref = -1;
    ASSERT_EQ(fixture.pop(&ref), SchedulerLinkedResult::SUCCESS);
    EXPECT_EQ(ref, 2);
}

TEST(SchedulerChunkQueueTest, SkippingExhaustedChunksHasFiniteBudget) {
    ChunkFixture fixture(12);
    for (int64_t ref = 0; ref < 12; ++ref) {
        fixture.push({ref});
        fixture.chunks[ref].cursor = 1;
    }
    int64_t window[4]{};
    uint32_t count = 99;
    EXPECT_EQ(scheduler_chunk_window(fixture.queue, window, &count, 2), SchedulerLinkedResult::RETRY);
    EXPECT_EQ(count, 0u);
    EXPECT_EQ(fixture.control.head, 1);
    EXPECT_EQ(scheduler_chunk_window(fixture.queue, window, &count, 16), SchedulerLinkedResult::NOT_READY);
    EXPECT_EQ(fixture.control.head, 11);
}

TEST(SchedulerChunkQueueTest, InvalidInputNeverPublishesPayloadOrClaimsWork) {
    ChunkFixture fixture(3);
    SchedulerLinkedBatch batch{};
    const int64_t duplicate[] = {0, 0};
    EXPECT_EQ(scheduler_chunk_prepare(fixture.queue, duplicate, 2, &batch), SchedulerLinkedResult::INVALID);
    EXPECT_EQ(fixture.chunks[0].count, 0u);
    EXPECT_EQ(batch.count, 0u);
    fixture.push({0, 1});
    fixture.chunks[0].count = 9;
    int64_t ref = -1;
    EXPECT_EQ(fixture.pop(&ref), SchedulerLinkedResult::INVALID);
    EXPECT_EQ(fixture.chunks[0].cursor, 0u);
}

TEST(SchedulerChunkQueueTest, BackingIsReusableOnlyAfterQuiescence) {
    ChunkFixture fixture(2);
    for (uint32_t epoch = 0; epoch < 3; ++epoch) {
        fixture.reset();
        fixture.push({1, 0});
        int64_t ref = -1;
        ASSERT_EQ(fixture.pop(&ref), SchedulerLinkedResult::SUCCESS);
        EXPECT_EQ(ref, 1);
        ASSERT_EQ(fixture.pop(&ref), SchedulerLinkedResult::SUCCESS);
        EXPECT_EQ(ref, 0);
    }
}

TEST(SchedulerChunkQueueTest, PrivateChunkChainsMergeWithoutRewritingSealedPayload) {
    ChunkFixture fixture(6);
    auto pending = fixture.prepare({0, 1});
    auto incoming = fixture.prepare({2, 3, 4, 5});
    ASSERT_EQ(
        scheduler_linked_append_private(fixture.queue.chain, &pending, &incoming), SchedulerLinkedResult::SUCCESS
    );
    EXPECT_EQ(pending.count, 2u);
    EXPECT_EQ(incoming.count, 0u);
    EXPECT_EQ(fixture.chunks[0].count, 2u);
    EXPECT_EQ(fixture.chunks[2].count, 4u);
    EXPECT_EQ(scheduler_linked_try_push(fixture.queue.chain, &pending, 0), SchedulerLinkedResult::RETRY);
    EXPECT_EQ(pending.count, 2u);
    ASSERT_EQ(scheduler_linked_try_push(fixture.queue.chain, &pending), SchedulerLinkedResult::SUCCESS);
    int64_t ref = -1;
    for (int64_t expected = 0; expected < 6; ++expected) {
        ASSERT_EQ(fixture.pop(&ref), SchedulerLinkedResult::SUCCESS);
        EXPECT_EQ(ref, expected);
    }
}

TEST(SchedulerChunkQueueTest, ConcurrentPublicationAndWindowClaimsPreservePayloadAndUniqueTasks) {
    for (uint32_t credit : {1, 2}) {
        constexpr uint32_t task_count = 512;
        ChunkFixture fixture(task_count);
        std::vector<uint64_t> payload(task_count);
        std::vector<std::atomic<uint32_t>> seen(task_count);
        for (auto &entry : seen)
            entry.store(0);
        std::atomic<bool> start{false};
        std::atomic<uint32_t> consumed{0}, failures{0};
        std::vector<std::thread> threads;
        for (uint32_t producer = 0; producer < 8; ++producer)
            threads.emplace_back([&, producer] {
                while (!start.load()) {}
                const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
                for (uint32_t offset = 0; offset < 64; offset += 8) {
                    std::vector<int64_t> refs(8);
                    std::iota(refs.begin(), refs.end(), producer * 64 + offset);
                    for (auto ref : refs)
                        payload[ref] = static_cast<uint64_t>(ref) * 17 + 23;
                    SchedulerLinkedBatch pending{};
                    if (scheduler_chunk_prepare(fixture.queue, refs.data(), 8, &pending) !=
                        SchedulerLinkedResult::SUCCESS) {
                        ++failures;
                        return;
                    }
                    while (pending.count && failures.load() == 0) {
                        auto result = scheduler_linked_try_push(fixture.queue.chain, &pending);
                        if ((result != SchedulerLinkedResult::SUCCESS && result != SchedulerLinkedResult::RETRY) ||
                            std::chrono::steady_clock::now() >= deadline) {
                            ++failures;
                            return;
                        }
                    }
                }
            });
        for (uint32_t consumer = 0; consumer < 12; ++consumer)
            threads.emplace_back([&, consumer] {
                while (!start.load()) {}
                const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
                uint32_t poll = consumer;
                while (consumed.load() < task_count && failures.load() == 0) {
                    if (std::chrono::steady_clock::now() >= deadline) {
                        ++failures;
                        break;
                    }
                    int64_t window[4]{};
                    uint32_t visible = 0;
                    auto result = scheduler_chunk_window(fixture.queue, window, &visible);
                    if (result == SchedulerLinkedResult::INVALID) {
                        ++failures;
                        break;
                    }
                    if (result != SchedulerLinkedResult::SUCCESS) continue;
                    const uint32_t preferred = (++poll % 4 == 0) ? 0 : (consumer + poll) % visible;
                    for (uint32_t index = 0; index < visible; ++index) {
                        SchedulerChunkCandidate candidate{};
                        result = scheduler_chunk_peek(fixture.queue, window[(preferred + index) % visible], &candidate);
                        if (result == SchedulerLinkedResult::INVALID) {
                            ++failures;
                            break;
                        }
                        if (result != SchedulerLinkedResult::SUCCESS) continue;
                        int64_t refs[2]{};
                        uint32_t count = 0;
                        result = scheduler_chunk_try_claim(fixture.queue, candidate, credit, refs, &count);
                        if (result == SchedulerLinkedResult::INVALID) {
                            ++failures;
                            break;
                        }
                        if (result != SchedulerLinkedResult::SUCCESS) continue;
                        for (uint32_t item = 0; item < count; ++item) {
                            const auto ref = refs[item];
                            if (ref < 0 || ref >= task_count || ++seen[ref] != 1 ||
                                payload[ref] != static_cast<uint64_t>(ref) * 17 + 23) {
                                ++failures;
                                break;
                            }
                            ++consumed;
                        }
                        break;
                    }
                }
            });
        start.store(true);
        for (auto &thread : threads)
            thread.join();
        EXPECT_EQ(failures.load(), 0u);
        EXPECT_EQ(consumed.load(), task_count);
        for (auto &entry : seen)
            EXPECT_EQ(entry.load(), 1u);
    }
}
}  // namespace
