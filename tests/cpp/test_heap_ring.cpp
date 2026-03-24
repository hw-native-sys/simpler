/**
 * Unit tests for PTO2HeapRing — GM output buffer ring allocator.
 *
 * Tests allocation correctness, alignment, wrap-around, back-pressure,
 * and reclamation logic.
 */

#include <gtest/gtest.h>
#include <atomic>
#include <cstring>
#include "pto_ring_buffer.h"

// =============================================================================
// Test fixture — sets up a small HeapRing for testing
// =============================================================================

class HeapRingTest : public ::testing::Test {
protected:
    static constexpr uint64_t HEAP_SIZE = 1024;

    alignas(64) uint8_t heap_buf[HEAP_SIZE]{};
    std::atomic<uint64_t> top{0};
    std::atomic<uint64_t> tail{0};
    std::atomic<int32_t> error_code{PTO2_ERROR_NONE};
    PTO2HeapRing ring{};

    void SetUp() override {
        top.store(0);
        tail.store(0);
        error_code.store(PTO2_ERROR_NONE);
        pto2_heap_ring_init(&ring, heap_buf, HEAP_SIZE, &tail, &top);
        ring.error_code_ptr = &error_code;
    }
};

// =============================================================================
// Basic allocation
// =============================================================================

TEST_F(HeapRingTest, BasicAlloc) {
    void* ptr = ring.pto2_heap_ring_try_alloc(128);
    ASSERT_NE(ptr, nullptr);
    // Pointer should be within the heap buffer
    EXPECT_GE((uintptr_t)ptr, (uintptr_t)heap_buf);
    EXPECT_LT((uintptr_t)ptr, (uintptr_t)(heap_buf + HEAP_SIZE));
    // top should have advanced
    EXPECT_GE(top.load(), 128u);
}

// =============================================================================
// Alignment enforcement
// =============================================================================

TEST_F(HeapRingTest, AlignmentEnforcement) {
    // Request 13 bytes — should be rounded up to PTO2_ALIGN_SIZE (64)
    void* ptr = ring.pto2_heap_ring_try_alloc(13);
    ASSERT_NE(ptr, nullptr);
    uint64_t allocated = top.load();
    EXPECT_EQ(allocated % PTO2_ALIGN_SIZE, 0u);
    EXPECT_GE(allocated, 64u);  // At least 64 bytes (aligned from 13)
}

// =============================================================================
// Wrap-around
// =============================================================================

TEST_F(HeapRingTest, WrapAround) {
    // Allocate most of the heap (leaving < 128 at end)
    uint64_t first_alloc = HEAP_SIZE - 128;  // 896 bytes
    void* p1 = ring.pto2_heap_ring_try_alloc(first_alloc);
    ASSERT_NE(p1, nullptr);

    // Advance tail past the first allocation to free it
    tail.store(first_alloc);

    // Now request 256 bytes — won't fit at end (only 128 left), should wrap
    void* p2 = ring.pto2_heap_ring_try_alloc(256);
    ASSERT_NE(p2, nullptr);
    // The wrapped allocation should start from the beginning
    EXPECT_EQ((uintptr_t)p2, (uintptr_t)heap_buf);
}

// =============================================================================
// Exact fit at end
// =============================================================================

TEST_F(HeapRingTest, ExactFitAtEnd) {
    // Allocate to leave exactly 128 bytes at end
    uint64_t first_alloc = HEAP_SIZE - 128;
    void* p1 = ring.pto2_heap_ring_try_alloc(first_alloc);
    ASSERT_NE(p1, nullptr);

    // Advance tail to free space
    tail.store(first_alloc);

    // Request exactly 128 bytes — should fit at end without wrapping
    void* p2 = ring.pto2_heap_ring_try_alloc(128);
    ASSERT_NE(p2, nullptr);
    // Should be allocated at end, not wrapped
    EXPECT_EQ((uintptr_t)p2, (uintptr_t)(heap_buf + first_alloc));
}

// =============================================================================
// Full — try_alloc returns nullptr
// =============================================================================

TEST_F(HeapRingTest, FullReturnsNull) {
    // Fill the heap
    void* p1 = ring.pto2_heap_ring_try_alloc(HEAP_SIZE - 64);
    ASSERT_NE(p1, nullptr);

    // Try to allocate more — should fail (non-blocking)
    void* p2 = ring.pto2_heap_ring_try_alloc(128);
    EXPECT_EQ(p2, nullptr);
}

// =============================================================================
// Reclaim and reuse
// =============================================================================

TEST_F(HeapRingTest, ReclaimAndReuse) {
    // Allocate 512 bytes
    void* p1 = ring.pto2_heap_ring_try_alloc(512);
    ASSERT_NE(p1, nullptr);

    // Advance tail to reclaim first allocation
    tail.store(512);

    // Now should be able to allocate again
    void* p2 = ring.pto2_heap_ring_try_alloc(512);
    ASSERT_NE(p2, nullptr);
}

// =============================================================================
// Zero size allocation
// =============================================================================

TEST_F(HeapRingTest, ZeroSizeAlloc) {
    // Request 0 bytes — implementation may return NULL or allocate minimum unit
    void* ptr = ring.pto2_heap_ring_try_alloc(0);
    // Either behavior is acceptable: NULL (reject 0-size) or valid pointer
    // Just verify no crash occurred
    (void)ptr;
}

// =============================================================================
// Available space query
// =============================================================================

TEST_F(HeapRingTest, AvailableSpace) {
    uint64_t avail_before = ring.pto2_heap_ring_available();
    EXPECT_EQ(avail_before, HEAP_SIZE);

    ring.pto2_heap_ring_try_alloc(256);
    uint64_t avail_after = ring.pto2_heap_ring_available();
    EXPECT_LT(avail_after, avail_before);
}

// =============================================================================
// Multiple sequential allocations
// =============================================================================

TEST_F(HeapRingTest, SequentialAllocations) {
    // Allocate several chunks
    void* p1 = ring.pto2_heap_ring_try_alloc(64);
    void* p2 = ring.pto2_heap_ring_try_alloc(64);
    void* p3 = ring.pto2_heap_ring_try_alloc(64);
    ASSERT_NE(p1, nullptr);
    ASSERT_NE(p2, nullptr);
    ASSERT_NE(p3, nullptr);

    // Allocations should be non-overlapping and sequential
    EXPECT_LT((uintptr_t)p1, (uintptr_t)p2);
    EXPECT_LT((uintptr_t)p2, (uintptr_t)p3);
}
