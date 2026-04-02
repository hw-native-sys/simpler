/**
 * Unit tests for Tensor and related types in tensor.h
 *
 * Tests Tensor operations, TensorCreateInfo, Segment intersection,
 * and boundary conditions in cache-line layout coupling.
 */

#include <gtest/gtest.h>

#include <cstring>

#include "pto_orchestration_api.h"

// Helper: create a Tensor via make_tensor_external (the public factory)
static Tensor make_test_tensor(void* addr, uint64_t size, const uint32_t shapes[],
                               uint32_t ndims, DataType dtype = DataType::FLOAT32,
                               bool manual_dep = false, int32_t version = 0) {
    return make_tensor_external(addr, shapes, ndims, dtype, manual_dep, version);
}

// =============================================================================
// Segment intersection
// =============================================================================

TEST(Segment, OverlappingIntersects) {
    Segment a{0, 10};
    Segment b{5, 15};
    EXPECT_TRUE(a.line_segment_intersection(b));
    EXPECT_TRUE(b.line_segment_intersection(a));
}

TEST(Segment, TouchingDoesNotIntersect) {
    Segment a{0, 10};
    Segment b{10, 20};
    EXPECT_FALSE(a.line_segment_intersection(b));
    EXPECT_FALSE(b.line_segment_intersection(a));
}

TEST(Segment, DisjointDoesNotIntersect) {
    Segment a{0, 5};
    Segment b{10, 20};
    EXPECT_FALSE(a.line_segment_intersection(b));
    EXPECT_FALSE(b.line_segment_intersection(a));
}

TEST(Segment, ZeroLengthAtBoundary) {
    // Zero-length segment at position 10 touching [0,10)
    Segment a{10, 10};
    Segment b{0, 10};
    EXPECT_FALSE(a.line_segment_intersection(b));
}

TEST(Segment, ZeroLengthInsideRange) {
    // Zero-length segment at position 5 inside [0,10)
    // end(5) > other.begin(0) && other.end(10) > begin(5) => true
    // KNOWN BEHAVIOR: zero-length segments report intersection.
    // This could cause spurious dependencies in TensorMap overlap detection.
    Segment a{5, 5};
    Segment b{0, 10};
    EXPECT_TRUE(a.line_segment_intersection(b));
}

TEST(Segment, IdenticalRanges) {
    Segment a{0, 10};
    EXPECT_TRUE(a.line_segment_intersection(a));
}

TEST(Segment, ContainsFull) {
    Segment outer{0, 20};
    Segment inner{5, 10};
    EXPECT_TRUE(outer.contains(inner));
}

TEST(Segment, ContainsIdentical) {
    Segment a{0, 10};
    EXPECT_TRUE(a.contains(a));
}

TEST(Segment, DoesNotContainPartial) {
    Segment a{0, 10};
    Segment b{5, 15};
    EXPECT_FALSE(a.contains(b));
}

TEST(Segment, ContainsAtBoundary) {
    Segment a{0, 10};
    Segment b{0, 10};
    EXPECT_TRUE(a.contains(b));
}

// =============================================================================
// TensorCreateInfo
// =============================================================================

TEST(TensorCreateInfo, BufferSizeBytes) {
    uint32_t shapes[] = {4, 8};
    TensorCreateInfo ci(shapes, 2, DataType::FLOAT32);
    EXPECT_EQ(ci.buffer_size_bytes(), 4u * 8u * 4u);  // 4*8 elements * 4 bytes
}

TEST(TensorCreateInfo, BufferSizeBytesInt8) {
    uint32_t shapes[] = {10, 20, 30};
    TensorCreateInfo ci(shapes, 3, DataType::INT8);
    EXPECT_EQ(ci.buffer_size_bytes(), 10u * 20u * 30u * 1u);
}

TEST(TensorCreateInfo, SizeIs64Bytes) {
    EXPECT_EQ(sizeof(TensorCreateInfo), 64u);
}

TEST(TensorCreateInfo, InitialValueDefault) {
    uint32_t shapes[] = {4};
    TensorCreateInfo ci(shapes, 1);
    EXPECT_FALSE(ci.has_initial_value);
}

TEST(TensorCreateInfo, SetInitialValue) {
    uint32_t shapes[] = {4};
    TensorCreateInfo ci(shapes, 1);
    ci.set_initial_value<float>(3.14f);
    EXPECT_TRUE(ci.has_initial_value);
}

// =============================================================================
// Tensor basic operations
// =============================================================================

TEST(Tensor, SizeIs128Bytes) {
    EXPECT_EQ(sizeof(Tensor), 128u);
}

TEST(Tensor, RawShapesAtOffset64) {
    EXPECT_EQ(offsetof(Tensor, raw_shapes), 64u);
}

TEST(Tensor, MakeExternal) {
    char buf[256];
    uint32_t shapes[] = {4, 8};
    auto t = make_test_tensor(buf, sizeof(buf), shapes, 2);
    EXPECT_EQ(t.buffer.addr, reinterpret_cast<uint64_t>(buf));
    EXPECT_EQ(t.ndims, 2u);
    EXPECT_EQ(t.shapes[0], 4u);
    EXPECT_EQ(t.shapes[1], 8u);
}

TEST(Tensor, Numel) {
    char buf[256];
    uint32_t shapes[] = {4, 8, 2};
    auto t = make_test_tensor(buf, sizeof(buf), shapes, 3);
    EXPECT_EQ(t.numel(), 64u);
}

TEST(Tensor, NumelZeroDim) {
    char buf[256];
    uint32_t shapes[] = {};
    auto t = make_test_tensor(buf, sizeof(buf), shapes, 0);
    EXPECT_EQ(t.numel(), 0u);
}

TEST(Tensor, IsContiguousWhenRawEqShapes) {
    char buf[256];
    uint32_t shapes[] = {4, 8};
    auto t = make_test_tensor(buf, sizeof(buf), shapes, 2);
    EXPECT_TRUE(t.is_raw_eq_shapes);
    EXPECT_TRUE(t.is_contiguous());
}

TEST(Tensor, IsSameMemref) {
    char buf1[256], buf2[256];
    uint32_t shapes[] = {4};
    auto t1 = make_test_tensor(buf1, sizeof(buf1), shapes, 1);
    auto t2 = make_test_tensor(buf1, sizeof(buf1), shapes, 1);
    auto t3 = make_test_tensor(buf2, sizeof(buf2), shapes, 1);
    EXPECT_TRUE(t1.is_same_memref(t2));
    EXPECT_FALSE(t1.is_same_memref(t3));
}

// =============================================================================
// View
// =============================================================================

TEST(Tensor, ViewWithZeroOffsets) {
    char buf[256];
    uint32_t shapes[] = {10, 20};
    auto parent = make_test_tensor(buf, sizeof(buf), shapes, 2);

    uint32_t view_shapes[] = {5, 10};
    uint32_t view_offsets[] = {0, 0};
    auto v = parent.view(view_shapes, view_offsets);

    EXPECT_EQ(v.shapes[0], 5u);
    EXPECT_EQ(v.shapes[1], 10u);
    EXPECT_TRUE(v.is_all_offset_zero);
    EXPECT_EQ(v.buffer.addr, parent.buffer.addr);
}

TEST(Tensor, ViewWithNonZeroOffsets) {
    char buf[256];
    uint32_t shapes[] = {10, 20};
    auto parent = make_test_tensor(buf, sizeof(buf), shapes, 2);

    uint32_t view_shapes[] = {5, 10};
    uint32_t view_offsets[] = {2, 3};
    auto v = parent.view(view_shapes, view_offsets);

    EXPECT_EQ(v.shapes[0], 5u);
    EXPECT_EQ(v.shapes[1], 10u);
    EXPECT_FALSE(v.is_all_offset_zero);
    EXPECT_EQ(v.offsets[0], 2u);
    EXPECT_EQ(v.offsets[1], 3u);
}

TEST(Tensor, ViewOffsetAccumulation) {
    char buf[256];
    uint32_t shapes[] = {20, 30};
    auto parent = make_test_tensor(buf, sizeof(buf), shapes, 2);

    // First view with offsets
    uint32_t v1_shapes[] = {10, 15};
    uint32_t v1_offsets[] = {5, 10};
    auto v1 = parent.view(v1_shapes, v1_offsets);

    // Second view on top of first
    uint32_t v2_shapes[] = {3, 4};
    uint32_t v2_offsets[] = {1, 2};
    auto v2 = v1.view(v2_shapes, v2_offsets);

    EXPECT_EQ(v2.offsets[0], 6u);  // 5 + 1
    EXPECT_EQ(v2.offsets[1], 12u);  // 10 + 2
}

// =============================================================================
// Reshape
// =============================================================================

TEST(Tensor, ReshapeContiguous) {
    char buf[256];
    uint32_t shapes[] = {4, 8};
    auto t = make_test_tensor(buf, sizeof(buf), shapes, 2);

    uint32_t new_shapes[] = {32};
    auto r = t.reshape(new_shapes, 1);

    EXPECT_EQ(r.numel(), 32u);
    EXPECT_EQ(r.ndims, 1u);
    EXPECT_EQ(r.shapes[0], 32u);
    EXPECT_TRUE(r.is_raw_eq_shapes);
    EXPECT_TRUE(r.is_all_offset_zero);
}

TEST(Tensor, ReshapePreservesBuffer) {
    char buf[256];
    uint32_t shapes[] = {4, 8};
    auto t = make_test_tensor(buf, sizeof(buf), shapes, 2);

    uint32_t new_shapes[] = {2, 16};
    auto r = t.reshape(new_shapes, 2);

    EXPECT_EQ(r.buffer.addr, t.buffer.addr);
}

// =============================================================================
// Transpose
// =============================================================================

TEST(Tensor, TransposeSwapsDims) {
    char buf[256];
    uint32_t shapes[] = {4, 8, 2};
    auto t = make_test_tensor(buf, sizeof(buf), shapes, 3);

    auto tr = t.transpose(0, 2);

    EXPECT_EQ(tr.shapes[0], 2u);
    EXPECT_EQ(tr.shapes[1], 8u);
    EXPECT_EQ(tr.shapes[2], 4u);
    EXPECT_EQ(tr.numel(), t.numel());
}

// =============================================================================
// compute_flat_offset
// =============================================================================

TEST(Tensor, ComputeFlatOffsetZeroDim) {
    char buf[256];
    uint32_t shapes[] = {};
    auto t = make_test_tensor(buf, sizeof(buf), shapes, 0);
    uint32_t indices[] = {};
    EXPECT_EQ(t.compute_flat_offset(indices, 0), 0u);
}

TEST(Tensor, ComputeFlatOffset1D) {
    char buf[256];
    uint32_t shapes[] = {10};
    auto t = make_test_tensor(buf, sizeof(buf), shapes, 1);
    uint32_t indices[] = {7};
    EXPECT_EQ(t.compute_flat_offset(indices, 1), 7u);
}

TEST(Tensor, ComputeFlatOffset2D) {
    char buf[256];
    uint32_t shapes[] = {4, 8};
    auto t = make_test_tensor(buf, sizeof(buf), shapes, 2);
    // Row-major: offset = i0 * 8 + i1 = 2*8+3 = 19
    uint32_t indices[] = {2, 3};
    EXPECT_EQ(t.compute_flat_offset(indices, 2), 19u);
}

// =============================================================================
// update_start_offset
// =============================================================================

TEST(Tensor, UpdateStartOffsetZeroOffsets) {
    char buf[256];
    uint32_t shapes[] = {4, 8};
    auto t = make_test_tensor(buf, sizeof(buf), shapes, 2);
    t.update_start_offset();
    EXPECT_EQ(t.start_offset, 0u);
}

// =============================================================================
// fill_initial_value
// =============================================================================

TEST(Tensor, FillInitialValue) {
    alignas(64) char buf[128];
    memset(buf, 0, sizeof(buf));

    uint32_t shapes[] = {32};
    TensorCreateInfo ci(shapes, 1, DataType::FLOAT32);
    ci.set_initial_value<float>(1.0f);

    // Use make_tensor_external then overwrite with init_from_create_info
    auto t = make_tensor_external(buf, shapes, 1);
    t.init_from_create_info(ci, buf, sizeof(buf));

    // Check that the buffer was filled with 1.0f
    float* data = reinterpret_cast<float*>(buf);
    for (int i = 0; i < 32; i++) {
        EXPECT_FLOAT_EQ(data[i], 1.0f) << "Mismatch at index " << i;
    }
}

// =============================================================================
// Layout coupling: TensorCreateInfo <-> Tensor cacheline 1
// =============================================================================

TEST(LayoutCoupling, TensorCreateInfoMatchesTensor) {
    // These static_asserts are in tensor.h but we verify they compile here
    static_assert(offsetof(TensorCreateInfo, version) == offsetof(Tensor, version));
    static_assert(offsetof(TensorCreateInfo, dtype) == offsetof(Tensor, dtype));
    static_assert(offsetof(TensorCreateInfo, ndims) == offsetof(Tensor, ndims));
    static_assert(offsetof(TensorCreateInfo, is_all_offset_zero) == offsetof(Tensor, is_all_offset_zero));
    SUCCEED();
}
