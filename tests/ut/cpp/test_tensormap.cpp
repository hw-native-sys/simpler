/**
 * Unit tests for PTO2TensorMap and check_overlap from pto_tensormap.h
 */

#include <gtest/gtest.h>

#include <cstring>

#include "pto_orchestration_api.h"
#include "pto_tensormap.h"

// =============================================================================
// TensorMapEntry::check_overlap tests
// =============================================================================

class CheckOverlapTest : public ::testing::Test {
protected:
    alignas(64) PTO2TensorMapEntry entry;
    char buf[256];

    void SetUp() override {
        memset(&entry, 0, sizeof(entry));
        memset(buf, 0, sizeof(buf));
    }

    Tensor make_input(uint32_t shapes[], uint32_t ndims, int32_t version = 0) {
        return make_tensor_external(buf, shapes, ndims, DataType::FLOAT32, false, version);
    }

    void setup_entry(uint32_t shapes[], uint32_t ndims, int32_t version = 0) {
        entry.buffer_addr = reinterpret_cast<uint64_t>(buf);
        entry.ndims = ndims;
        entry.version = version;
        entry.is_all_offset_zero = true;
        for (uint32_t i = 0; i < ndims; i++) {
            entry.shapes[i] = shapes[i];
        }
    }
};

TEST_F(CheckOverlapTest, IdenticalShapesZeroOffsets) {
    uint32_t shapes[] = {10, 20};
    setup_entry(shapes, 2);
    auto input = make_input(shapes, 2);
    EXPECT_EQ(entry.check_overlap(input), OverlapStatus::COVERED);
}

TEST_F(CheckOverlapTest, InputLargerThanOutput) {
    uint32_t entry_shapes[] = {5, 10};
    uint32_t input_shapes[] = {10, 20};
    setup_entry(entry_shapes, 2);
    auto input = make_input(input_shapes, 2);
    EXPECT_EQ(entry.check_overlap(input), OverlapStatus::COVERED);
}

TEST_F(CheckOverlapTest, InputSmallerThanOutput) {
    uint32_t entry_shapes[] = {10, 20};
    uint32_t input_shapes[] = {5, 10};
    setup_entry(entry_shapes, 2);
    auto input = make_input(input_shapes, 2);
    EXPECT_EQ(entry.check_overlap(input), OverlapStatus::OTHER);
}

TEST_F(CheckOverlapTest, VersionMismatch) {
    // input.version > entry.version -> returns OTHER (not NO_OVERLAP)
    // This means version bumps create dependencies (intentional)
    uint32_t shapes[] = {10};
    setup_entry(shapes, 1, /*version=*/0);
    auto input = make_input(shapes, 1, /*version=*/1);
    EXPECT_EQ(entry.check_overlap(input), OverlapStatus::OTHER);
}

TEST_F(CheckOverlapTest, DisjointOffsetsWithNonZeroEntry) {
    // Entry covers [10,20), input covers [0,5) -> NO_OVERLAP
    uint32_t entry_shapes[] = {10};
    uint32_t input_shapes[] = {5};
    setup_entry(entry_shapes, 1);
    entry.is_all_offset_zero = false;
    entry.offsets[0] = 10;

    auto input = make_input(input_shapes, 1);
    // Input is [0,5), entry is [10,20)
    EXPECT_EQ(entry.check_overlap(input), OverlapStatus::NO_OVERLAP);
}

// =============================================================================
// TensorMap lifecycle tests
// =============================================================================

class TensorMapTest : public ::testing::Test {
protected:
    PTO2TensorMap tmap;
    int32_t window_sizes[PTO2_MAX_RING_DEPTH] = {16, 16, 16, 16};

    void SetUp() override {
        bool ok = tmap.init(64, 256, window_sizes);
        ASSERT_TRUE(ok);
        // Initialize last_task_alives to 0 for all rings
        for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
            tmap.last_task_alives[r] = 0;
        }
    }

    void TearDown() override {
        tmap.destroy();
    }
};

TEST_F(TensorMapTest, InitSucceeds) {
    EXPECT_EQ(tmap.num_buckets, 64);
    EXPECT_EQ(tmap.pool_size, 256);
}

TEST_F(TensorMapTest, HashDistribution) {
    // Aligned addresses should distribute across buckets
    uint64_t addr1 = 0x1000;
    uint64_t addr2 = 0x2000;
    uint64_t addr3 = 0x3000;
    uint32_t h1 = tmap.hash(addr1);
    uint32_t h2 = tmap.hash(addr2);
    uint32_t h3 = tmap.hash(addr3);
    // At least some should be different
    EXPECT_TRUE(h1 != h2 || h2 != h3);
    // All within bucket range
    EXPECT_LT(h1, 64u);
    EXPECT_LT(h2, 64u);
    EXPECT_LT(h3, 64u);
}

TEST_F(TensorMapTest, InsertAndLookupExact) {
    char buf[256];
    uint32_t shapes[] = {10, 20};
    auto tensor = make_tensor_external(buf, shapes, 2);
    auto task_id = pto2_make_task_id(0, 5);

    tmap.insert(tensor, task_id, false);

    PTO2LookupResult result;
    tmap.lookup(tensor, result);
    EXPECT_GE(result.count, 1);
}

TEST_F(TensorMapTest, LookupNoMatch) {
    char buf1[256], buf2[256];
    uint32_t shapes[] = {10};
    auto tensor1 = make_tensor_external(buf1, shapes, 1);
    auto tensor2 = make_tensor_external(buf2, shapes, 1);
    auto task_id = pto2_make_task_id(0, 0);

    tmap.insert(tensor1, task_id, false);

    PTO2LookupResult result;
    tmap.lookup(tensor2, result);
    EXPECT_EQ(result.count, 0);
}

TEST_F(TensorMapTest, LookupStaleEntrySkipped) {
    char buf[256];
    uint32_t shapes[] = {10};
    auto tensor = make_tensor_external(buf, shapes, 1);
    auto task_id = pto2_make_task_id(0, 0);

    tmap.insert(tensor, task_id, false);

    // Invalidate: advance last_task_alives past this task
    tmap.sync_validity(0, 5);

    PTO2LookupResult result;
    tmap.lookup(tensor, result);
    EXPECT_EQ(result.count, 0);
}

TEST_F(TensorMapTest, MultipleSameBucket) {
    // Insert multiple entries for the same address
    char buf[256];
    uint32_t shapes[] = {10};
    auto tensor = make_tensor_external(buf, shapes, 1);

    tmap.insert(tensor, pto2_make_task_id(0, 0), false);
    tmap.insert(tensor, pto2_make_task_id(0, 1), false);
    tmap.insert(tensor, pto2_make_task_id(0, 2), false);

    PTO2LookupResult result;
    tmap.lookup(tensor, result);
    EXPECT_EQ(result.count, 3);
}

TEST_F(TensorMapTest, CleanupRetired) {
    char buf[256];
    uint32_t shapes[] = {10};
    auto tensor = make_tensor_external(buf, shapes, 1);

    // Insert entries for tasks 0..4
    for (int i = 0; i < 5; i++) {
        tmap.insert(tensor, pto2_make_task_id(0, i), false);
    }

    // Retire tasks 0..3
    tmap.cleanup_retired(0, 0, 4);
    tmap.sync_validity(0, 4);

    PTO2LookupResult result;
    tmap.lookup(tensor, result);
    EXPECT_EQ(result.count, 1);  // Only task 4 remains
}

TEST_F(TensorMapTest, NewEntryFreeListPriority) {
    // Allocate, free, allocate again -> should reuse freed entry
    PTO2TensorMapEntry* e1 = tmap.new_entry();
    ASSERT_NE(e1, nullptr);
    // Link entry so we can free it
    e1->bucket_index = 0;
    e1->prev_in_bucket = nullptr;
    e1->next_in_bucket = nullptr;
    e1->next_in_task = nullptr;
    e1->prev_in_task = nullptr;
    tmap.buckets[0] = e1;

    tmap.free_entry(*e1);

    PTO2TensorMapEntry* e2 = tmap.new_entry();
    EXPECT_EQ(e1, e2);  // Reused from free list
}

TEST_F(TensorMapTest, EntryValidBoundary) {
    alignas(64) PTO2TensorMapEntry entry;
    memset(&entry, 0, sizeof(entry));

    // local_id == last_task_alive -> valid (not yet retired)
    entry.producer_task_id = pto2_make_task_id(0, 5);
    tmap.last_task_alives[0] = 5;
    EXPECT_TRUE(tmap.entry_valid(entry));

    // local_id < last_task_alive -> stale
    tmap.last_task_alives[0] = 6;
    EXPECT_FALSE(tmap.entry_valid(entry));
}

TEST_F(TensorMapTest, MultiRingInterleaving) {
    char buf[256];
    uint32_t shapes[] = {10};
    auto tensor = make_tensor_external(buf, shapes, 1);

    // Insert entries from ring 0 and ring 1
    tmap.insert(tensor, pto2_make_task_id(0, 0), false);
    tmap.insert(tensor, pto2_make_task_id(1, 0), false);
    tmap.insert(tensor, pto2_make_task_id(0, 1), false);

    // Retire ring 0 tasks
    tmap.cleanup_retired(0, 0, 2);
    tmap.sync_validity(0, 2);

    // Ring 1 entry should still be valid
    PTO2LookupResult result;
    tmap.lookup(tensor, result);
    EXPECT_EQ(result.count, 1);
    EXPECT_EQ(result.entries[0].entry->producer_task_id.ring(), 1);
}

// =============================================================================
// Static assertions (compile-time checks)
// =============================================================================

TEST(TensorMapLayout, EntrySizeIs128) {
    EXPECT_EQ(sizeof(PTO2TensorMapEntry), 128u);
}

TEST(TensorMapLayout, CacheLine2StartsAt64) {
    EXPECT_EQ(offsetof(PTO2TensorMapEntry, prev_in_bucket), 64u);
}
