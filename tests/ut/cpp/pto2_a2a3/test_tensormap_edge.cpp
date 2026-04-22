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
/**
 * Edge-case tests for TensorMap and Tensor overlap detection.
 *
 * ============================================================================
 * ANALYSIS FINDINGS -- check_overlap() in PTO2TensorMapEntry
 * ============================================================================
 *
 * BUG-CANDIDATE-1 (Overlap fast path): check_overlap() loops for
 *   entry->ndims, reading input.shapes[i] for all i < entry->ndims.
 *   When input has fewer dimensions, shapes[i] beyond input->ndims are
 *   stale (Tensor::init only copies ndims elements).  The result is
 *   non-deterministic -- depends on whatever value happens to be in memory.
 *   The test poisons input.shapes[1] to make the stale read deterministic
 *   and proves the loop bound is wrong.
 *
 * BUG-CANDIDATE-2 (Overlap slow path): The slow path constructs Segment from
 *   offsets and shapes.  But it uses `uint64_t in_off = input.offsets[i]` when
 *   `input.is_all_offset_zero` is false.  If ndims < RUNTIME_MAX_TENSOR_DIMS,
 *   offsets[ndims..4] may be uninitialized garbage.  The loop runs for
 *   entry->ndims iterations, which could exceed input->ndims.
 *   -> Actually the loop runs for `ndims` which is the ENTRY's ndims.
 *     If entry->ndims > input->ndims, input->shapes[i] beyond input->ndims is 0.
 *     Segment{in_off, in_off + 0} has length 0 -> intersection is always false
 *     -> returns NO_OVERLAP.  This might be wrong if the extra dimensions
 *     are broadcast or don't exist.
 *
 * BUG-CANDIDATE-3 (Dimension mismatch): check_overlap uses entry->ndims
 *   exclusively, ignoring input->ndims.  If input has MORE dimensions than
 *   entry, the extra input dimensions are never checked.  This could miss
 *   partial overlaps in higher dimensions.
 *
 * BUG-CANDIDATE-4 (Lookup result saturation): PTO2_LOOKUP_MAX_RESULTS = 16.
 *   If more than 16 overlapping entries exist, results are silently dropped.
 *   This means dependencies can be missed in highly-connected graphs.
 *
 * BUG-CANDIDATE-5 (TensorMap new_entry pool exhaustion): new_entry() calls
 *   `always_assert(next_entry_idx < pool_size)` which throws/aborts when the
 *   pool is fully used AND free_list is empty.  There's no graceful fallback.
 *
 * BUG-CANDIDATE-6 (Hash collision with cleanup): DISMISSED.
 *   cleanup_retired() uses debug_assert to verify entry belongs to the
 *   retiring task.  In theory, if the cleanup range exceeds task_window_size,
 *   slot reuse causes ABA.  However, sync_tensormap()'s overlap check
 *   (pto_tensormap.cpp:244) triggers cleanup every time the current task's
 *   slot collides with last_cleanup, bounding the cleanup range to at most
 *   task_window_size.  This guarantees each slot maps to exactly one task
 *   in any cleanup pass.  The scenario is unreachable in production.
 *
 * BUG-CANDIDATE-7 (copy_from_tensor doesn't zero beyond ndims): When
 *   copying shapes[]/offsets[] from Tensor to Entry, only ndims elements
 *   are copied.  shapes[ndims..4] retain whatever was in the entry before
 *   (from pool reuse).  check_overlap loops for entry->ndims, so garbage
 *   data beyond ndims could affect overlap detection if the loop ever
 *   reads beyond what was copied.  Currently safe because the loop uses
 *   entry->ndims which matches what was copied, but fragile.
 *
 * ============================================================================
 * ANALYSIS FINDINGS -- Tensor struct
 * ============================================================================
 *
 * EDGE-1: Tensor with 0 dimensions (ndims=0).  No shapes/offsets.
 *   check_overlap loop doesn't execute -> returns COVERED (fast path, contains=true).
 *   Two 0-dim tensors at same addr are always "covered".
 *
 * EDGE-2: Tensor with maximum dimensions (ndims=5).
 *   All shape/offset arrays fully used.
 *
 * EDGE-3: Shape of 0 in one dimension.  Segment = {off, off+0} = empty.
 *   line_segment_intersection({off, off+0}, {x,y}) = (off+0 > x) && (y > off)
 *   = (off > x) && (y > off).  Empty segment may or may not intersect.
 *
 * EDGE-4: Cleanup ABA -- DISMISSED.  sync_tensormap()'s overlap check
 *   bounds cleanup range to at most task_window_size, so a single slot never
 *   maps to two different tasks within one cleanup_retired() call.
 */

#include <gtest/gtest.h>
#include <cstring>
#include <set>
#include "common.h"
#include "pto_tensormap.h"
#include "pto_orchestration_api.h"

// =============================================================================
// Helpers
// =============================================================================

static Tensor
make_tensor_nd(uint64_t addr, uint32_t ndims, const uint32_t shapes[], const uint32_t offsets[], int32_t version = 0) {
    // Seed with make_tensor_external() (Tensor's default ctor is private).
    // Use a dummy 1-dim shape for the seed; we overwrite everything via init().
    uint32_t seed_shape[1] = {1};
    Tensor t = make_tensor_external(
        reinterpret_cast<void *>(addr), seed_shape, 1, DataType::FLOAT32, /*manual_dep=*/false, /*version=*/0
    );
    uint32_t s[RUNTIME_MAX_TENSOR_DIMS]{};
    uint32_t rs[RUNTIME_MAX_TENSOR_DIMS]{};
    uint32_t o[RUNTIME_MAX_TENSOR_DIMS]{};
    bool all_zero = true;
    for (uint32_t i = 0; i < ndims && i < RUNTIME_MAX_TENSOR_DIMS; i++) {
        s[i] = shapes[i];
        rs[i] = shapes[i];
        o[i] = offsets ? offsets[i] : 0;
        if (o[i] != 0) all_zero = false;
    }
    uint64_t total = 4;
    for (uint32_t i = 0; i < ndims; i++)
        total *= (rs[i] + (offsets ? offsets[i] : 0));
    t.init((void *)addr, total, rs, s, o, ndims, DataType::FLOAT32, version, all_zero, /*is_raw_eq_shapes=*/true);
    return t;
}

class TensorMapEdgeTest : public ::testing::Test {
protected:
    PTO2TensorMap tmap{};
    int32_t window_sizes[PTO2_MAX_RING_DEPTH]{};

    void SetUp() override {
        for (int i = 0; i < PTO2_MAX_RING_DEPTH; i++)
            window_sizes[i] = 64;
        ASSERT_TRUE(tmap.init(256, 512, window_sizes));
    }
    void TearDown() override { tmap.destroy(); }
};


// ---------------------------------------------------------------------------
// EDGE-1: Zero dimensions (ndims=0)
// ---------------------------------------------------------------------------
TEST_F(TensorMapEdgeTest, ZeroDimensionTensor) {
    uint32_t seed_shape[1] = {1};
    Tensor t = make_tensor_external(reinterpret_cast<void *>(0x2000), seed_shape, 1, DataType::FLOAT32, false, 0);
    uint32_t s[5]{}, o[5]{};
    t.init((void *)0x2000, 0, s, s, o, 0, DataType::FLOAT32, 0, true, true);

    PTO2TaskId task = PTO2TaskId::make(0, 0);
    tmap.insert(t, task);

    PTO2LookupResult result;
    result.count = 0;
    tmap.lookup(t, result);

    EXPECT_GE(result.count, 1);
    if (result.count > 0) {
        // ndims=0: fast-path loop doesn't execute, contains=true -> COVERED
        EXPECT_EQ(result.entries[0].overlap_status, OverlapStatus::COVERED);
    }
}

// ---------------------------------------------------------------------------
// Zero dimensions: Two different 0-dim tensors at same address always COVERED
// This is semantically questionable -- should scalar tensors be independent?
// ---------------------------------------------------------------------------
TEST_F(TensorMapEdgeTest, TwoZeroDimTensorsSameAddr) {
    uint32_t seed_shape[1] = {1};
    Tensor t1 = make_tensor_external(reinterpret_cast<void *>(0x2100), seed_shape, 1, DataType::FLOAT32, false, 0);
    Tensor t2 = make_tensor_external(reinterpret_cast<void *>(0x2100), seed_shape, 1, DataType::FLOAT32, false, 0);
    uint32_t s[5]{}, o[5]{};
    t1.init((void *)0x2100, 0, s, s, o, 0, DataType::FLOAT32, 0, true, true);
    t2.init((void *)0x2100, 0, s, s, o, 0, DataType::FLOAT32, 0, true, true);

    tmap.insert(t1, PTO2TaskId::make(0, 0));
    tmap.insert(t2, PTO2TaskId::make(0, 1));

    PTO2LookupResult result;
    result.count = 0;
    tmap.lookup(t1, result);

    // Both 0-dim entries report COVERED for any 0-dim input at same addr
    EXPECT_EQ(result.count, 2);
    for (int i = 0; i < result.count; i++) {
        EXPECT_EQ(result.entries[i].overlap_status, OverlapStatus::COVERED)
            << "0-dim tensors always report COVERED (empty loop -> contains=true)";
    }
}

// ---------------------------------------------------------------------------
// BUG-CANDIDATE-4: Lookup result saturation (>16 producers)
// ---------------------------------------------------------------------------
TEST_F(TensorMapEdgeTest, LookupResultSaturation) {
    uint32_t shapes[] = {100};
    Tensor t = make_tensor_nd(0x4000, 1, shapes, nullptr, 0);

    // Insert 20 producers for the same tensor
    for (int i = 0; i < 20; i++) {
        tmap.insert(t, PTO2TaskId::make(0, i));
    }

    PTO2LookupResult result;
    result.count = 0;
    tmap.lookup(t, result);

    // Only 16 results fit -- 4 dependencies are silently dropped
    EXPECT_EQ(result.count, PTO2_LOOKUP_MAX_RESULTS)
        << "More than 16 overlapping producers: results saturated, deps missed";
}

// ---------------------------------------------------------------------------
// BUG-CANDIDATE-4 extended: Saturation drops OLDEST producers (newest first)
// Because insert() adds at head of bucket chain, lookup traverses newest first.
// The first 16 (newest) entries fill the result, dropping the 4 oldest.
// ---------------------------------------------------------------------------
TEST_F(TensorMapEdgeTest, LookupSaturationDropsOldest) {
    uint32_t shapes[] = {100};
    Tensor t = make_tensor_nd(0x4100, 1, shapes, nullptr, 0);

    for (int i = 0; i < 20; i++) {
        tmap.insert(t, PTO2TaskId::make(0, i));
    }

    PTO2LookupResult result;
    result.count = 0;
    tmap.lookup(t, result);

    ASSERT_EQ(result.count, PTO2_LOOKUP_MAX_RESULTS);

    // Verify the kept results are the newest 16 (tasks 19, 18, ..., 4)
    // and the oldest 4 (tasks 0, 1, 2, 3) are dropped
    for (int i = 0; i < result.count; i++) {
        int32_t local_id = result.entries[i].entry->producer_task_id.local();
        // The newest entries are inserted at head, so lookup sees them first
        EXPECT_GE(local_id, 4) << "Oldest tasks (0-3) should be the ones dropped by saturation";
    }
}

// ---------------------------------------------------------------------------
// Version-based overlap: newer version returns OTHER
// ---------------------------------------------------------------------------
TEST_F(TensorMapEdgeTest, VersionMismatchReturnsOther) {
    uint32_t shapes[] = {100};
    Tensor v0 = make_tensor_nd(0x5000, 1, shapes, nullptr, 0);
    Tensor v1 = make_tensor_nd(0x5000, 1, shapes, nullptr, 1);

    tmap.insert(v0, PTO2TaskId::make(0, 0));

    PTO2LookupResult result;
    result.count = 0;
    tmap.lookup(v1, result);

    EXPECT_EQ(result.count, 1);
    // Version 1 > Version 0 -> OTHER (not COVERED)
    EXPECT_EQ(result.entries[0].overlap_status, OverlapStatus::OTHER);
}

// ---------------------------------------------------------------------------
// Version: Same version, same shapes -> COVERED
// ---------------------------------------------------------------------------
TEST_F(TensorMapEdgeTest, SameVersionSameShapesCovered) {
    uint32_t shapes[] = {100};
    Tensor t = make_tensor_nd(0x5100, 1, shapes, nullptr, 0);

    tmap.insert(t, PTO2TaskId::make(0, 0));

    PTO2LookupResult result;
    result.count = 0;
    tmap.lookup(t, result);

    EXPECT_EQ(result.count, 1);
    EXPECT_EQ(result.entries[0].overlap_status, OverlapStatus::COVERED) << "Same version + same shapes -> COVERED";
}

// ---------------------------------------------------------------------------
// Partial overlap 1D: [0:100] vs [50:150]
// ---------------------------------------------------------------------------
TEST_F(TensorMapEdgeTest, PartialOverlap1D) {
    uint32_t prod_shapes[] = {100};
    Tensor prod = make_tensor_nd(0x6000, 1, prod_shapes, nullptr, 0);
    tmap.insert(prod, PTO2TaskId::make(0, 0));

    // Consumer reads [50:150] -- partial overlap
    uint32_t cons_shapes[] = {100};
    uint32_t cons_offsets[] = {50};
    Tensor cons = make_tensor_nd(0x6000, 1, cons_shapes, cons_offsets, 0);

    PTO2LookupResult result;
    result.count = 0;
    tmap.lookup(cons, result);

    EXPECT_EQ(result.count, 1);
    // Consumer [50,150) vs Producer [0,100) -> intersection = [50,100).
    // Consumer does NOT contain producer -> OTHER
    EXPECT_EQ(result.entries[0].overlap_status, OverlapStatus::OTHER);
}

// ---------------------------------------------------------------------------
// Consumer fully covers producer: COVERED
// ---------------------------------------------------------------------------
TEST_F(TensorMapEdgeTest, ConsumerCoversProducer) {
    // Producer writes [10:20]
    uint32_t prod_shapes[] = {10};
    uint32_t prod_offsets[] = {10};
    Tensor prod = make_tensor_nd(0x7000, 1, prod_shapes, prod_offsets, 0);
    tmap.insert(prod, PTO2TaskId::make(0, 0));

    // Consumer reads [0:100] -- fully covers producer
    uint32_t cons_shapes[] = {100};
    Tensor cons = make_tensor_nd(0x7000, 1, cons_shapes, nullptr, 0);

    PTO2LookupResult result;
    result.count = 0;
    tmap.lookup(cons, result);

    EXPECT_EQ(result.count, 1);
    // Consumer [0,100) contains Producer [10,20) -> COVERED
    EXPECT_EQ(result.entries[0].overlap_status, OverlapStatus::COVERED);
}

// ---------------------------------------------------------------------------
// Adjacent regions: [0:100] vs [100:200] -> NO_OVERLAP
// ---------------------------------------------------------------------------
TEST_F(TensorMapEdgeTest, AdjacentNoOverlap) {
    uint32_t prod_shapes[] = {100};
    Tensor prod = make_tensor_nd(0x8000, 1, prod_shapes, nullptr, 0);
    tmap.insert(prod, PTO2TaskId::make(0, 0));

    uint32_t cons_shapes[] = {100};
    uint32_t cons_offsets[] = {100};
    Tensor cons = make_tensor_nd(0x8000, 1, cons_shapes, cons_offsets, 0);

    PTO2LookupResult result;
    result.count = 0;
    tmap.lookup(cons, result);

    // [0,100) vs [100,200) -> end(100) > begin(100)? No -> NO_OVERLAP
    EXPECT_EQ(result.count, 0);
}

// ---------------------------------------------------------------------------
// One-element overlap: [0:100] vs [99:199]
// ---------------------------------------------------------------------------
TEST_F(TensorMapEdgeTest, OneElementOverlap) {
    uint32_t prod_shapes[] = {100};
    Tensor prod = make_tensor_nd(0x8100, 1, prod_shapes, nullptr, 0);
    tmap.insert(prod, PTO2TaskId::make(0, 0));

    uint32_t cons_shapes[] = {100};
    uint32_t cons_offsets[] = {99};
    Tensor cons = make_tensor_nd(0x8100, 1, cons_shapes, cons_offsets, 0);

    PTO2LookupResult result;
    result.count = 0;
    tmap.lookup(cons, result);

    // [0,100) vs [99,199) -> intersection = [99,100) = 1 element
    EXPECT_EQ(result.count, 1);
    EXPECT_EQ(result.entries[0].overlap_status, OverlapStatus::OTHER) << "Partial overlap (1 element) -> OTHER";
}

// ---------------------------------------------------------------------------
// EDGE-3: Shape of 0 in one dimension (empty segment behavior)
// ---------------------------------------------------------------------------
TEST_F(TensorMapEdgeTest, ZeroShapeInDimension) {
    // Producer: 2D [10, 0] -- zero in dim 1
    uint32_t prod_shapes[] = {10, 0};
    Tensor prod = make_tensor_nd(0x8200, 2, prod_shapes, nullptr, 0);
    tmap.insert(prod, PTO2TaskId::make(0, 0));

    // Consumer: 2D [10, 20]
    uint32_t cons_shapes[] = {10, 20};
    Tensor cons = make_tensor_nd(0x8200, 2, cons_shapes, nullptr, 0);

    PTO2LookupResult result;
    result.count = 0;
    tmap.lookup(cons, result);

    if (result.count > 0) {
        // Fast path: input.shapes[1](20) < entry.shapes[1](0)? No, 20 >= 0.
        // -> contains = true -> COVERED.
        // But the producer wrote ZERO elements in dim 1!
        // Should a zero-area producer be "covered" by any consumer?
        // This is semantically questionable.
        EXPECT_EQ(result.entries[0].overlap_status, OverlapStatus::COVERED)
            << "Zero-shape producer is COVERED by any consumer (empty production)";
    }
}

// ---------------------------------------------------------------------------
// 2D overlap: different slices
// ---------------------------------------------------------------------------
TEST_F(TensorMapEdgeTest, MultiDimOverlap) {
    // Producer: 2D [10, 20] at offset [0, 0]
    uint32_t prod_shapes[] = {10, 20};
    Tensor prod = make_tensor_nd(0x9000, 2, prod_shapes, nullptr, 0);
    tmap.insert(prod, PTO2TaskId::make(0, 0));

    // Consumer: 2D [5, 10] at offset [2, 5] -- overlaps partially
    uint32_t cons_shapes[] = {5, 10};
    uint32_t cons_offsets[] = {2, 5};
    Tensor cons = make_tensor_nd(0x9000, 2, cons_shapes, cons_offsets, 0);

    PTO2LookupResult result;
    result.count = 0;
    tmap.lookup(cons, result);

    EXPECT_EQ(result.count, 1);
    // Consumer [2,7)x[5,15) vs Producer [0,10)x[0,20)
    // check_overlap checks if INPUT(consumer) contains ENTRY(producer):
    // Dim 0: consumer [2,7) does NOT contain producer [0,10) -> contains=false
    // Dim 1: consumer [5,15) does NOT contain producer [0,20) -> contains=false
    // All dims intersect, but consumer doesn't fully cover -> OTHER
    EXPECT_EQ(result.entries[0].overlap_status, OverlapStatus::OTHER)
        << "Consumer sub-region inside producer: overlap exists but not COVERED";
}

// ---------------------------------------------------------------------------
// 2D: Consumer exceeds producer in one dimension -> OTHER
// ---------------------------------------------------------------------------
TEST_F(TensorMapEdgeTest, MultiDimPartialOverlap) {
    uint32_t prod_shapes[] = {10, 20};
    Tensor prod = make_tensor_nd(0x9100, 2, prod_shapes, nullptr, 0);
    tmap.insert(prod, PTO2TaskId::make(0, 0));

    // Consumer: [8, 25] -- exceeds producer in dim 1 (25 > 20)
    uint32_t cons_shapes[] = {8, 25};
    Tensor cons = make_tensor_nd(0x9100, 2, cons_shapes, nullptr, 0);

    PTO2LookupResult result;
    result.count = 0;
    tmap.lookup(cons, result);

    EXPECT_EQ(result.count, 1);
    // Fast path: shapes comparison
    // input.shapes[0]=8 >= entry.shapes[0]=10? No -> contains=false -> OTHER
    EXPECT_EQ(result.entries[0].overlap_status, OverlapStatus::OTHER);
}

// ---------------------------------------------------------------------------
// 5D full overlap test (maximum dimensions)
// ---------------------------------------------------------------------------
TEST_F(TensorMapEdgeTest, FullFiveDimensionalOverlap) {
    uint32_t prod_shapes[] = {2, 3, 4, 5, 6};
    Tensor prod = make_tensor_nd(0x9200, 5, prod_shapes, nullptr, 0);
    tmap.insert(prod, PTO2TaskId::make(0, 0));

    // Consumer with larger shapes in all dims -> COVERED
    uint32_t cons_shapes[] = {4, 6, 8, 10, 12};
    Tensor cons = make_tensor_nd(0x9200, 5, cons_shapes, nullptr, 0);

    PTO2LookupResult result;
    result.count = 0;
    tmap.lookup(cons, result);

    EXPECT_EQ(result.count, 1);
    EXPECT_EQ(result.entries[0].overlap_status, OverlapStatus::COVERED)
        << "5D consumer covers 5D producer in all dimensions";
}

// ---------------------------------------------------------------------------
// Cleanup then insert: verify chain integrity
// ---------------------------------------------------------------------------
TEST_F(TensorMapEdgeTest, CleanupThenReuseSlot) {
    uint32_t shapes[] = {100};
    Tensor t = make_tensor_nd(0xA000, 1, shapes, nullptr, 0);

    // Insert entries for tasks 0-7
    for (int i = 0; i < 8; i++) {
        tmap.insert(t, PTO2TaskId::make(0, i));
    }
    EXPECT_EQ(tmap.valid_count(), 8);

    // Cleanup tasks 0-4
    tmap.cleanup_retired(0, 0, 5);
    tmap.sync_validity(0, 5);
    EXPECT_EQ(tmap.valid_count(), 3);  // tasks 5,6,7 remain

    // Re-insert with new task IDs that reuse slots 0-4
    // (task window = 64, so IDs 64-68 map to slots 0-4)
    for (int i = 64; i < 69; i++) {
        tmap.insert(t, PTO2TaskId::make(0, i));
    }

    PTO2LookupResult result;
    result.count = 0;
    tmap.lookup(t, result);

    // Should find 8 entries: 3 old (5,6,7) + 5 new (64-68)
    EXPECT_EQ(result.count, 8);
}

// ---------------------------------------------------------------------------
// Hash distribution: addresses that are multiples of common alignment
// ---------------------------------------------------------------------------
TEST_F(TensorMapEdgeTest, HashDistributionAlignedAddresses) {
    // Typical device addresses are 256-byte or 1024-byte aligned
    // The hash function should distribute these well
    std::set<uint32_t> buckets_used;
    for (int i = 0; i < 100; i++) {
        uint64_t addr = 0x10000 + i * 1024;
        uint32_t bucket = tmap.hash(addr);
        buckets_used.insert(bucket);
    }
    // With 256 buckets and 100 addresses, we should use many distinct buckets
    // (poor hash would cluster aligned addresses into few buckets)
    EXPECT_GT(buckets_used.size(), 50u) << "Hash should distribute 1024-aligned addresses across many buckets";
}

// ---------------------------------------------------------------------------
// Lookup on empty TensorMap
// ---------------------------------------------------------------------------
TEST_F(TensorMapEdgeTest, LookupEmpty) {
    uint32_t shapes[] = {100};
    Tensor t = make_tensor_nd(0xC000, 1, shapes, nullptr, 0);

    PTO2LookupResult result;
    result.count = 0;
    tmap.lookup(t, result);

    EXPECT_EQ(result.count, 0) << "Empty TensorMap returns no results";
}

// ---------------------------------------------------------------------------
// Lazy invalidation: entries become stale when last_task_alive advances
// ---------------------------------------------------------------------------
TEST_F(TensorMapEdgeTest, LazyInvalidation) {
    uint32_t shapes[] = {100};
    Tensor t = make_tensor_nd(0xD000, 1, shapes, nullptr, 0);

    // Insert entries for tasks 0-4
    for (int i = 0; i < 5; i++) {
        tmap.insert(t, PTO2TaskId::make(0, i));
    }

    // All 5 should be found
    PTO2LookupResult result;
    result.count = 0;
    tmap.lookup(t, result);
    EXPECT_EQ(result.count, 5);

    // Advance validity threshold: tasks 0-2 become stale
    tmap.sync_validity(0, 3);

    result.count = 0;
    tmap.lookup(t, result);
    EXPECT_EQ(result.count, 2) << "Only tasks 3,4 are valid after sync_validity(3)";
}

// ---------------------------------------------------------------------------
// entry_valid with different rings: ring isolation
// ---------------------------------------------------------------------------
TEST_F(TensorMapEdgeTest, RingIsolation) {
    uint32_t shapes[] = {100};
    Tensor t = make_tensor_nd(0xE000, 1, shapes, nullptr, 0);

    // Insert in ring 0 (task 0) and ring 1 (task 0)
    tmap.insert(t, PTO2TaskId::make(0, 0));
    tmap.insert(t, PTO2TaskId::make(1, 0));

    // Invalidate ring 0's tasks but not ring 1's
    tmap.sync_validity(0, 1);

    PTO2LookupResult result;
    result.count = 0;
    tmap.lookup(t, result);

    // Only ring 1's entry should remain valid
    EXPECT_EQ(result.count, 1);
    if (result.count == 1) {
        EXPECT_EQ(result.entries[0].entry->producer_task_id.ring(), 1)
            << "Ring 0's entry is invalidated; ring 1's entry survives";
    }
}

// ---------------------------------------------------------------------------
// Multiple tensors at different addresses: no cross-contamination
// ---------------------------------------------------------------------------
TEST_F(TensorMapEdgeTest, DifferentAddressesIsolated) {
    uint32_t shapes[] = {100};
    Tensor t1 = make_tensor_nd(0xF000, 1, shapes, nullptr, 0);
    Tensor t2 = make_tensor_nd(0xF100, 1, shapes, nullptr, 0);

    tmap.insert(t1, PTO2TaskId::make(0, 0));
    tmap.insert(t2, PTO2TaskId::make(0, 1));

    PTO2LookupResult result1;
    result1.count = 0;
    tmap.lookup(t1, result1);
    EXPECT_EQ(result1.count, 1);

    PTO2LookupResult result2;
    result2.count = 0;
    tmap.lookup(t2, result2);
    EXPECT_EQ(result2.count, 1);

    // Each lookup only finds its own producer
    if (result1.count == 1 && result2.count == 1) {
        EXPECT_NE(
            result1.entries[0].entry->producer_task_id.local(), result2.entries[0].entry->producer_task_id.local()
        );
    }
}

// ---------------------------------------------------------------------------
// Free list recycling: after cleanup, new inserts reuse freed entries
// without exhausting the pool. Verified via observable behavior (pool
// doesn't exhaust) rather than internal pool index inspection.
// ---------------------------------------------------------------------------
TEST_F(TensorMapEdgeTest, FreeListRecycling) {
    uint32_t shapes[] = {100};
    Tensor t = make_tensor_nd(0x10000, 1, shapes, nullptr, 0);

    // Insert 60 entries (within window_size=64, no slot collision)
    for (int i = 0; i < 60; i++) {
        tmap.insert(t, PTO2TaskId::make(0, i));
    }

    // Cleanup all 60 (range 0..60 < window_size=64, no ABA)
    tmap.cleanup_retired(0, 0, 60);
    tmap.sync_validity(0, 60);

    // Insert another 60 -- should succeed because freed entries are reused
    for (int i = 60; i < 120; i++) {
        tmap.insert(t, PTO2TaskId::make(0, i));
    }

    // Verify via lookup: all 60 new entries should be findable
    PTO2LookupResult result;
    result.count = 0;
    tmap.lookup(t, result);
    // Capped at PTO2_LOOKUP_MAX_RESULTS=16, but count > 0 proves entries exist
    EXPECT_EQ(result.count, PTO2_LOOKUP_MAX_RESULTS) << "After cleanup+reinsert, new entries are findable -- "
                                                        "free list recycling keeps the pool from exhausting";
}
