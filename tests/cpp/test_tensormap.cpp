/**
 * Unit tests for PTO2TensorMap — hash table for automatic dependency discovery.
 *
 * Tests hash function, insert/lookup, overlap detection integration,
 * entry validity, cleanup, and collision chain integrity.
 */

#include <gtest/gtest.h>
#include <cstring>
#include <cstdlib>
#include <set>
#include "pto_tensormap.h"

// =============================================================================
// Test fixture
// =============================================================================

class TensorMapTest : public ::testing::Test {
protected:
    static constexpr int32_t NUM_BUCKETS = 64;
    static constexpr int32_t POOL_SIZE = 256;

    PTO2TensorMap tmap{};
    int32_t window_sizes[PTO2_MAX_RING_DEPTH]{};

    void SetUp() override {
        for (int i = 0; i < PTO2_MAX_RING_DEPTH; i++) {
            window_sizes[i] = 64;
        }
        bool ok = tmap.init(NUM_BUCKETS, POOL_SIZE, window_sizes);
        ASSERT_TRUE(ok);
    }

    void TearDown() override {
        tmap.destroy();
    }

    // Helper: create a simple 1D tensor
    Tensor make_tensor(uint64_t addr, uint32_t shape, uint32_t offset = 0,
                       int32_t version = 0) {
        Tensor t{};
        uint32_t shapes[] = {shape, 0, 0, 0, 0};
        uint32_t raw_shapes[] = {shape, 0, 0, 0, 0};
        uint32_t offsets[] = {offset, 0, 0, 0, 0};
        bool all_zero = (offset == 0);
        t.init((void*)addr, shape * 4, raw_shapes, shapes, offsets, 1,
               DataType::FLOAT32, version, all_zero, true);
        return t;
    }
};

// =============================================================================
// Hash function tests
// =============================================================================

TEST_F(TensorMapTest, HashDistribution) {
    // Test that different addresses hash to different buckets
    // Use large address spread to avoid alignment-caused collisions
    std::set<uint32_t> buckets;
    for (uint64_t i = 0; i < 100; i++) {
        uint64_t addr = 0x1000 + i * 0x10000;  // Large stride to get different hash bits
        uint32_t bucket = tmap.hash(addr);
        EXPECT_LT(bucket, (uint32_t)NUM_BUCKETS);
        buckets.insert(bucket);
    }
    // At least a few different buckets (hash should spread across buckets)
    EXPECT_GE(buckets.size(), 3u);
}

TEST_F(TensorMapTest, SameAddrSameBucket) {
    uint64_t addr = 0x5000;
    uint32_t b1 = tmap.hash(addr);
    uint32_t b2 = tmap.hash(addr);
    EXPECT_EQ(b1, b2);
}

TEST_F(TensorMapTest, PowerOf2Buckets) {
    // Trying to init with non-power-of-2 should fail
    PTO2TensorMap bad{};
    int32_t ws[PTO2_MAX_RING_DEPTH] = {64, 64, 64, 64};
    bool ok = bad.init(7, 128, ws);  // 7 is not power of 2
    EXPECT_FALSE(ok);
}

// =============================================================================
// Insert and lookup
// =============================================================================

TEST_F(TensorMapTest, InsertAndLookup) {
    // Task A writes tensor at addr 0x1000
    Tensor output = make_tensor(0x1000, 100, 0, 0);
    PTO2TaskId task_a = pto2_make_task_id(0, 0);
    tmap.insert(output, task_a, true);

    // Task B reads the same tensor — lookup should find it
    Tensor input = make_tensor(0x1000, 100, 0, 0);
    PTO2LookupResult result;
    result.count = 0;
    tmap.lookup(input, result);

    EXPECT_GE(result.count, 1);
    EXPECT_EQ(result.entries[0].entry->producer_task_id.raw, task_a.raw);
}

TEST_F(TensorMapTest, MultipleProducers) {
    Tensor t = make_tensor(0x2000, 100, 0, 0);

    // Two tasks write to same address
    PTO2TaskId task_a = pto2_make_task_id(0, 0);
    PTO2TaskId task_b = pto2_make_task_id(0, 1);
    tmap.insert(t, task_a, true);
    tmap.insert(t, task_b, true);

    // Lookup should find both producers
    PTO2LookupResult result;
    result.count = 0;
    tmap.lookup(t, result);
    EXPECT_GE(result.count, 2);
}

// =============================================================================
// Stale entry filtering
// =============================================================================

TEST_F(TensorMapTest, StaleEntryFiltering) {
    Tensor t = make_tensor(0x3000, 100, 0, 0);
    PTO2TaskId task_old = pto2_make_task_id(0, 0);
    tmap.insert(t, task_old, true);

    // Advance validity — task 0 is now stale
    tmap.sync_validity(0, 1);

    // Lookup should filter out the stale entry
    PTO2LookupResult result;
    result.count = 0;
    tmap.lookup(t, result);
    EXPECT_EQ(result.count, 0);
}

// =============================================================================
// No overlap — different address
// =============================================================================

TEST_F(TensorMapTest, NoOverlapDifferentAddr) {
    Tensor output = make_tensor(0x4000, 100, 0, 0);
    PTO2TaskId task_a = pto2_make_task_id(0, 0);
    tmap.insert(output, task_a, true);

    // Lookup with a different address — should find nothing
    Tensor input = make_tensor(0x5000, 100, 0, 0);
    PTO2LookupResult result;
    result.count = 0;
    tmap.lookup(input, result);
    EXPECT_EQ(result.count, 0);
}

// =============================================================================
// Collision chain integrity — insert, remove, re-insert
// =============================================================================

TEST_F(TensorMapTest, CollisionChainIntegrity) {
    // Insert multiple entries that hash to same bucket
    // (use same address, different task IDs)
    Tensor t = make_tensor(0x6000, 100, 0, 0);

    PTO2TaskId ids[5];
    for (int i = 0; i < 5; i++) {
        ids[i] = pto2_make_task_id(0, i);
        tmap.insert(t, ids[i], true);
    }

    // Verify all 5 can be found
    PTO2LookupResult result;
    result.count = 0;
    tmap.lookup(t, result);
    EXPECT_EQ(result.count, 5);

    // Clean up tasks 0-2
    tmap.cleanup_retired(0, 0, 3);

    // Re-lookup — should only find tasks 3,4
    result.count = 0;
    tmap.sync_validity(0, 3);
    tmap.lookup(t, result);
    EXPECT_EQ(result.count, 2);

    // Re-insert new tasks
    PTO2TaskId new_id = pto2_make_task_id(0, 5);
    tmap.insert(t, new_id, true);

    result.count = 0;
    tmap.lookup(t, result);
    EXPECT_EQ(result.count, 3);
}

// =============================================================================
// Valid count tracking
// =============================================================================

TEST_F(TensorMapTest, ValidCountTracking) {
    EXPECT_EQ(tmap.valid_count(), 0);

    Tensor t = make_tensor(0x7000, 50, 0, 0);
    for (int i = 0; i < 10; i++) {
        tmap.insert(t, pto2_make_task_id(0, i), true);
    }
    EXPECT_EQ(tmap.valid_count(), 10);
}
