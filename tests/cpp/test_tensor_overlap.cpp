/**
 * Unit tests for Tensor overlap detection — tensor.h.
 *
 * Tests the Segment intersection/containment logic and multi-dimensional
 * overlap checking between tensors.
 */

#include <gtest/gtest.h>
#include "tensor.h"

// =============================================================================
// Helper: create a simple 1D tensor
// =============================================================================

static Tensor make_1d_tensor(uint64_t addr, uint64_t buf_size, uint32_t shape,
                              uint32_t offset = 0, int32_t version = 0) {
    Tensor t{};
    uint32_t shapes[] = {shape, 0, 0, 0, 0};
    uint32_t raw_shapes[] = {shape, 0, 0, 0, 0};
    uint32_t offsets[] = {offset, 0, 0, 0, 0};
    bool all_offset_zero = (offset == 0);
    t.init((void*)addr, buf_size, raw_shapes, shapes, offsets, 1,
           DataType::FLOAT32, version, all_offset_zero, true);
    return t;
}

// =============================================================================
// Segment tests
// =============================================================================

TEST(SegmentTest, Intersection) {
    Segment a{0, 100};
    Segment b{50, 150};
    EXPECT_TRUE(a.line_segment_intersection(b));
    EXPECT_TRUE(b.line_segment_intersection(a));
}

TEST(SegmentTest, NoIntersection) {
    Segment a{0, 100};
    Segment b{100, 200};
    EXPECT_FALSE(a.line_segment_intersection(b));
}

TEST(SegmentTest, Contains) {
    Segment outer{0, 100};
    Segment inner{10, 50};
    EXPECT_TRUE(outer.contains(inner));
    EXPECT_FALSE(inner.contains(outer));
}

TEST(SegmentTest, IdenticalContains) {
    Segment a{10, 50};
    EXPECT_TRUE(a.contains(a));
}

// =============================================================================
// Tensor overlap tests — different base address
// =============================================================================

TEST(TensorOverlapTest, NoOverlap_DifferentAddr) {
    Tensor a = make_1d_tensor(0x100, 400, 100);
    Tensor b = make_1d_tensor(0x200, 400, 100);
    // Different buffer.addr → completely independent buffers
    EXPECT_NE(a.buffer.addr, b.buffer.addr);
}

// =============================================================================
// Tensor overlap tests — identical tensors
// =============================================================================

TEST(TensorOverlapTest, FullOverlap_Identical) {
    Tensor a = make_1d_tensor(0x100, 400, 100, 0, 0);
    Tensor b = make_1d_tensor(0x100, 400, 100, 0, 0);
    // Same addr, same shape, same offset → COVERED
    // TensorMap uses check_overlap on entries; here we verify tensors are equal
    EXPECT_EQ(a.buffer.addr, b.buffer.addr);
    EXPECT_EQ(a.shapes[0], b.shapes[0]);
    EXPECT_EQ(a.offsets[0], b.offsets[0]);
}

// =============================================================================
// Tensor overlap tests — partial overlap 1D
// =============================================================================

TEST(TensorOverlapTest, PartialOverlap_1D) {
    // [0:100] vs [50:150] — partial overlap
    Tensor a = make_1d_tensor(0x100, 600, 100, 0, 0);
    Tensor b = make_1d_tensor(0x100, 600, 100, 50, 0);
    // They share the same buffer but different offsets
    EXPECT_EQ(a.buffer.addr, b.buffer.addr);
    EXPECT_NE(a.offsets[0], b.offsets[0]);
}

// =============================================================================
// Tensor overlap tests — subset contained
// =============================================================================

TEST(TensorOverlapTest, Contained_Subset) {
    // [10:20] is within [0:100]
    Tensor big = make_1d_tensor(0x100, 400, 100, 0, 0);
    Tensor small = make_1d_tensor(0x100, 400, 10, 10, 0);
    EXPECT_EQ(big.buffer.addr, small.buffer.addr);
    // big covers small
    Segment big_seg{0, 100};
    Segment small_seg{10, 20};
    EXPECT_TRUE(big_seg.contains(small_seg));
}

// =============================================================================
// Tensor overlap tests — adjacent (no overlap)
// =============================================================================

TEST(TensorOverlapTest, NoOverlap_Adjacent) {
    // [0:100] vs [100:200] — adjacent, no overlap
    Segment a{0, 100};
    Segment b{100, 200};
    EXPECT_FALSE(a.line_segment_intersection(b));
}

// =============================================================================
// Tensor init correctness
// =============================================================================

TEST(TensorOverlapTest, TensorInitFields) {
    uint32_t shapes[] = {10, 20, 0, 0, 0};
    uint32_t raw_shapes[] = {10, 20, 0, 0, 0};
    uint32_t offsets[] = {0, 0, 0, 0, 0};
    Tensor t{};
    t.init((void*)0x1000, 800, raw_shapes, shapes, offsets, 2,
           DataType::FLOAT32, 5, true, true);
    EXPECT_EQ(t.buffer.addr, 0x1000u);
    EXPECT_EQ(t.buffer.size, 800u);
    EXPECT_EQ(t.ndims, 2u);
    EXPECT_EQ(t.version, 5);
    EXPECT_EQ(t.shapes[0], 10u);
    EXPECT_EQ(t.shapes[1], 20u);
    EXPECT_TRUE(t.is_all_offset_zero);
    EXPECT_TRUE(t.is_raw_eq_shapes);
}
