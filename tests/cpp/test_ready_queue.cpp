/**
 * Unit tests for PTO2ReadyQueue — lock-free bounded MPMC queue.
 *
 * Tests FIFO ordering, empty/full, wrap-around, size query,
 * and concurrent push/pop.
 */

#include <gtest/gtest.h>
#include <atomic>
#include <thread>
#include <vector>
#include <set>
#include <cstring>
#include "pto_scheduler.h"

// =============================================================================
// Test fixture
// =============================================================================

class ReadyQueueTest : public ::testing::Test {
protected:
    static constexpr uint64_t QUEUE_CAP = 64;

    PTO2ReadyQueueSlot slots[QUEUE_CAP]{};
    PTO2ReadyQueue queue{};

    // Dummy slot states for pushing into the queue
    PTO2TaskSlotState dummy_slots[QUEUE_CAP]{};

    void SetUp() override {
        memset(slots, 0, sizeof(slots));
        queue.slots = slots;
        queue.capacity = QUEUE_CAP;
        queue.mask = QUEUE_CAP - 1;
        queue.enqueue_pos.store(0, std::memory_order_relaxed);
        queue.dequeue_pos.store(0, std::memory_order_relaxed);

        // Initialize per-slot sequence numbers (Vyukov pattern)
        for (uint64_t i = 0; i < QUEUE_CAP; i++) {
            slots[i].sequence.store((int64_t)i, std::memory_order_relaxed);
            slots[i].slot_state = nullptr;
        }
    }
};

// =============================================================================
// FIFO ordering
// =============================================================================

TEST_F(ReadyQueueTest, PushPop_FIFO) {
    bool ok;
    ok = queue.push(&dummy_slots[0]);
    EXPECT_TRUE(ok);
    ok = queue.push(&dummy_slots[1]);
    EXPECT_TRUE(ok);
    ok = queue.push(&dummy_slots[2]);
    EXPECT_TRUE(ok);

    PTO2TaskSlotState* a = queue.pop();
    PTO2TaskSlotState* b = queue.pop();
    PTO2TaskSlotState* c = queue.pop();

    EXPECT_EQ(a, &dummy_slots[0]);
    EXPECT_EQ(b, &dummy_slots[1]);
    EXPECT_EQ(c, &dummy_slots[2]);
}

// =============================================================================
// Empty queue pop
// =============================================================================

TEST_F(ReadyQueueTest, EmptyPop) {
    PTO2TaskSlotState* result = queue.pop();
    EXPECT_EQ(result, nullptr);
}

// =============================================================================
// Full queue push
// =============================================================================

TEST_F(ReadyQueueTest, FullPush) {
    // Fill the queue to capacity
    for (uint64_t i = 0; i < QUEUE_CAP; i++) {
        bool ok = queue.push(&dummy_slots[i % QUEUE_CAP]);
        if (!ok) {
            // Queue is full — this should happen at capacity
            EXPECT_GE(i, QUEUE_CAP - 1);
            break;
        }
    }

    // Next push should fail
    PTO2TaskSlotState extra{};
    bool ok = queue.push(&extra);
    EXPECT_FALSE(ok);
}

// =============================================================================
// Wrap-around
// =============================================================================

TEST_F(ReadyQueueTest, WrapAround) {
    // Push and pop more than capacity to exercise wrap-around
    for (int round = 0; round < 3; round++) {
        for (uint64_t i = 0; i < QUEUE_CAP / 2; i++) {
            bool ok = queue.push(&dummy_slots[i]);
            EXPECT_TRUE(ok);
        }
        for (uint64_t i = 0; i < QUEUE_CAP / 2; i++) {
            PTO2TaskSlotState* s = queue.pop();
            EXPECT_NE(s, nullptr);
        }
    }

    // Queue should be empty at the end
    EXPECT_EQ(queue.pop(), nullptr);
    EXPECT_EQ(queue.size(), 0u);
}

// =============================================================================
// Size query
// =============================================================================

TEST_F(ReadyQueueTest, SizeQuery) {
    EXPECT_EQ(queue.size(), 0u);

    for (int i = 0; i < 10; i++) {
        queue.push(&dummy_slots[i]);
    }
    EXPECT_EQ(queue.size(), 10u);

    for (int i = 0; i < 5; i++) {
        queue.pop();
    }
    EXPECT_EQ(queue.size(), 5u);
}

// =============================================================================
// Concurrent push/pop stress test
// =============================================================================

TEST_F(ReadyQueueTest, ConcurrentPushPop) {
    constexpr int NUM_ITEMS = 1000;
    constexpr int NUM_PRODUCERS = 2;
    constexpr int NUM_CONSUMERS = 2;

    // Allocate slot states for all items
    std::vector<PTO2TaskSlotState> items(NUM_ITEMS);

    std::atomic<int> pushed{0};
    std::atomic<int> popped{0};

    // Producers
    auto producer = [&](int start) {
        for (int i = start; i < NUM_ITEMS; i += NUM_PRODUCERS) {
            while (!queue.push(&items[i])) {
                // Retry
            }
            pushed.fetch_add(1);
        }
    };

    // Consumers
    std::vector<PTO2TaskSlotState*> consumed[NUM_CONSUMERS];
    auto consumer = [&](int id) {
        while (popped.load() < NUM_ITEMS) {
            PTO2TaskSlotState* s = queue.pop();
            if (s != nullptr) {
                consumed[id].push_back(s);
                popped.fetch_add(1);
            }
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < NUM_PRODUCERS; i++) {
        threads.emplace_back(producer, i);
    }
    for (int i = 0; i < NUM_CONSUMERS; i++) {
        threads.emplace_back(consumer, i);
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(pushed.load(), NUM_ITEMS);
    EXPECT_EQ(popped.load(), NUM_ITEMS);

    // Verify no duplicates
    std::set<PTO2TaskSlotState*> unique_items;
    for (int i = 0; i < NUM_CONSUMERS; i++) {
        for (auto* s : consumed[i]) {
            unique_items.insert(s);
        }
    }
    EXPECT_EQ(unique_items.size(), (size_t)NUM_ITEMS);
}
