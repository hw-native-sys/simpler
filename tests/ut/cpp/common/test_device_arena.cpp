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
// Unit tests for src/common/device_comm/device_arena.h.
//
// DeviceArena lays out multiple sub-regions inside a single backing
// allocation. Its contract:
//   - reserve(size, align) returns the offset of a region, bumping a cursor
//     while honoring per-region alignment. Calls before commit() only.
//   - commit(base_align) performs exactly one backend alloc, padded so the
//     visible base can be forward-aligned. Idempotent.
//   - region_ptr(off) returns base + off after commit.
//   - release() frees the backing buffer and resets to pre-commit state.
//
// Tests inject mock alloc/free function pointers (no std::function — the
// header is also compiled into the AICPU binary).

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <unordered_set>

#include <gtest/gtest.h>

#include "device_arena.h"

namespace {

// Backend that hands out fresh aligned buffers and records every alloc/free
// it sees. Unique addresses make pointer-identity assertions trustworthy.
struct MockBackend {
    int alloc_count = 0;
    int free_count = 0;
    std::unordered_set<void *> live;

    // Last alloc size — tests assert it equals cursor + (base_align - 1).
    size_t last_alloc_size = 0;

    void *alloc(size_t size) {
        ++alloc_count;
        last_alloc_size = size;
        // posix_memalign for predictable alignment so the forward-alignment
        // math in commit() has a well-defined input.
        void *p = nullptr;
        if (posix_memalign(&p, 64, size) != 0) return nullptr;
        live.insert(p);
        return p;
    }

    void free(void *p) {
        ++free_count;
        EXPECT_EQ(live.count(p), 1u) << "free called on pointer not currently live";
        live.erase(p);
        std::free(p);
    }
};

void *mock_alloc(void *ctx, size_t size) { return static_cast<MockBackend *>(ctx)->alloc(size); }
void mock_free(void *ctx, void *p) { static_cast<MockBackend *>(ctx)->free(p); }

// Backend whose alloc returns an explicitly under-aligned address by
// over-allocating and shifting forward by a fixed amount. Used to exercise
// the forward-alignment path inside commit().
struct UnalignedBackend {
    // Raw blocks owned by us so we can free them via destructor.
    std::unordered_set<void *> raw_blocks;
    // Map shifted ptr -> raw block, so free() finds the right pointer.
    std::unordered_set<void *> live_shifted;
    void *(*shifted_from)(void *) = nullptr;  // unused; we track via map below

    // shift_bytes makes returned pointer mis-aligned w.r.t. base_align=1024.
    static constexpr size_t kShift = 16;
    int alloc_count = 0;
    int free_count = 0;
    size_t last_alloc_size = 0;

    // shifted -> raw map (kept small; tests use 1-2 entries).
    void *raw_for_shifted[8]{};
    void *shifted_keys[8]{};
    size_t entries = 0;

    void *alloc(size_t size) {
        ++alloc_count;
        last_alloc_size = size;
        void *raw = nullptr;
        if (posix_memalign(&raw, 4096, size + kShift) != 0) return nullptr;
        raw_blocks.insert(raw);
        void *shifted = static_cast<char *>(raw) + kShift;
        live_shifted.insert(shifted);
        shifted_keys[entries] = shifted;
        raw_for_shifted[entries] = raw;
        ++entries;
        return shifted;
    }

    void free(void *p) {
        ++free_count;
        EXPECT_EQ(live_shifted.count(p), 1u) << "unknown shifted pointer";
        live_shifted.erase(p);
        for (size_t i = 0; i < entries; ++i) {
            if (shifted_keys[i] == p) {
                std::free(raw_for_shifted[i]);
                raw_blocks.erase(raw_for_shifted[i]);
                shifted_keys[i] = nullptr;
                raw_for_shifted[i] = nullptr;
                return;
            }
        }
        FAIL() << "no raw block recorded for shifted pointer";
    }
};

void *unaligned_alloc(void *ctx, size_t size) { return static_cast<UnalignedBackend *>(ctx)->alloc(size); }
void unaligned_free(void *ctx, void *p) { static_cast<UnalignedBackend *>(ctx)->free(p); }

bool is_aligned(const void *p, size_t align) { return (reinterpret_cast<uintptr_t>(p) & (align - 1)) == 0; }

TEST(DeviceArenaTest, SingleRegionCommitAllocatesOnce) {
    MockBackend m;
    DeviceArena arena(mock_alloc, mock_free, &m);

    const size_t off = arena.reserve(100, 64);
    EXPECT_EQ(off, 0u);

    void *base = arena.commit(1024);
    ASSERT_NE(base, nullptr);
    EXPECT_EQ(m.alloc_count, 1);
    EXPECT_EQ(arena.alloc_count(), 1u);
    EXPECT_TRUE(arena.is_committed());
    EXPECT_EQ(arena.region_ptr(off), base);
    EXPECT_EQ(arena.region_size(off), 100u);
    EXPECT_TRUE(is_aligned(base, 1024));
}

TEST(DeviceArenaTest, MultipleRegionsAlignedSequentially) {
    MockBackend m;
    DeviceArena arena(mock_alloc, mock_free, &m);

    const size_t off_a = arena.reserve(1000, 64);
    const size_t off_b = arena.reserve(2000, 1024);
    const size_t off_c = arena.reserve(7, 8);

    // off_a is at 0; off_b must be align_up(1000, 1024) = 1024;
    // off_c is at align_up(1024 + 2000, 8) = align_up(3024, 8) = 3024.
    EXPECT_EQ(off_a, 0u);
    EXPECT_EQ(off_b, 1024u);
    EXPECT_EQ(off_c, 3024u);
    // total_size includes the trailing region exactly (no extra rounding).
    EXPECT_EQ(arena.total_size(), 3024u + 7u);

    ASSERT_NE(arena.commit(1024), nullptr);

    char *base = static_cast<char *>(arena.base());
    EXPECT_EQ(arena.region_ptr(off_a), base + 0);
    EXPECT_EQ(arena.region_ptr(off_b), base + 1024);
    EXPECT_EQ(arena.region_ptr(off_c), base + 3024);
    EXPECT_TRUE(is_aligned(arena.region_ptr(off_b), 1024));
    EXPECT_TRUE(is_aligned(arena.region_ptr(off_c), 8));
}

TEST(DeviceArenaTest, CommitPadsBackingAllocByBaseAlignMinusOne) {
    MockBackend m;
    DeviceArena arena(mock_alloc, mock_free, &m);
    arena.reserve(500, 64);
    arena.reserve(700, 64);

    ASSERT_NE(arena.commit(1024), nullptr);
    EXPECT_EQ(m.last_alloc_size, arena.total_size() + 1024 - 1);
}

TEST(DeviceArenaTest, CommitForwardAlignsUnalignedBase) {
    UnalignedBackend ub;
    DeviceArena arena(unaligned_alloc, unaligned_free, &ub);

    const size_t off = arena.reserve(256, 64);
    void *base = arena.commit(1024);
    ASSERT_NE(base, nullptr);
    EXPECT_TRUE(is_aligned(base, 1024));
    EXPECT_EQ(arena.region_ptr(off), base);
}

TEST(DeviceArenaTest, CommitIsIdempotent) {
    MockBackend m;
    DeviceArena arena(mock_alloc, mock_free, &m);
    arena.reserve(64, 64);

    void *b1 = arena.commit(1024);
    ASSERT_NE(b1, nullptr);
    void *b2 = arena.commit(1024);
    EXPECT_EQ(b2, b1) << "repeated commit must return the same base";
    EXPECT_EQ(m.alloc_count, 1);
}

TEST(DeviceArenaTest, ReleaseFreesAndAllowsReuse) {
    MockBackend m;
    DeviceArena arena(mock_alloc, mock_free, &m);

    arena.reserve(128, 64);
    ASSERT_NE(arena.commit(1024), nullptr);
    EXPECT_EQ(m.alloc_count, 1);

    arena.release();
    EXPECT_EQ(m.free_count, 1);
    EXPECT_FALSE(arena.is_committed());
    EXPECT_EQ(arena.base(), nullptr);
    EXPECT_EQ(arena.total_size(), 0u);

    // release() on empty arena is a no-op.
    arena.release();
    EXPECT_EQ(m.free_count, 1);

    // Re-use: a fresh reserve+commit allocates a new buffer.
    arena.reserve(256, 64);
    ASSERT_NE(arena.commit(1024), nullptr);
    EXPECT_EQ(m.alloc_count, 2);
}

TEST(DeviceArenaTest, ZeroSizedRegionDoesNotAdvanceCursor) {
    MockBackend m;
    DeviceArena arena(mock_alloc, mock_free, &m);

    const size_t off_a = arena.reserve(100, 64);
    const size_t off_zero = arena.reserve(0, 64);
    const size_t off_b = arena.reserve(50, 64);

    EXPECT_EQ(off_a, 0u);
    // align_up(100, 64) = 128, region of size 0 lives at 128, then cursor
    // remains 128 -> next region also starts at 128.
    EXPECT_EQ(off_zero, 128u);
    EXPECT_EQ(off_b, 128u);
}

TEST(DeviceArenaTest, DestructorReleasesOutstandingBuffer) {
    MockBackend m;
    {
        DeviceArena arena(mock_alloc, mock_free, &m);
        arena.reserve(128, 64);
        arena.commit(1024);
        EXPECT_EQ(m.alloc_count, 1);
        EXPECT_EQ(m.free_count, 0);
    }
    EXPECT_EQ(m.free_count, 1);
    EXPECT_TRUE(m.live.empty());
}

TEST(DeviceArenaTest, BackendAllocFailureReturnsNull) {
    auto failing_alloc = [](void *, size_t) -> void * {
        return nullptr;
    };
    auto noop_free = [](void *, void *) {};
    DeviceArena arena(static_cast<DeviceArena::AllocFn>(failing_alloc), noop_free, nullptr);

    arena.reserve(128, 64);
    void *base = arena.commit(1024);
    EXPECT_EQ(base, nullptr);
    EXPECT_FALSE(arena.is_committed());
}

}  // namespace
