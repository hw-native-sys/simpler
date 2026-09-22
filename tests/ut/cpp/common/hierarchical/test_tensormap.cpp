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

#include "tensormap.h"

// Helper: host key (worker_id=-1)
static TensorKey hk(uint64_t ptr) { return TensorKey::local_host(ptr); }

// Helper: child key scoped by NEXT_LEVEL worker id.
static TensorKey ck(uint64_t ptr, int32_t worker_id) { return TensorKey::local_child(ptr, worker_id); }
static constexpr RunId RUN1 = 1;
static constexpr RunId RUN2 = 2;

// Helper: a contiguous UINT8 view spanning [begin, end) of a `backing`-byte buffer — the simplest
// footprint whose bytes are exactly the range named.
static TensorFootprint br(uint64_t begin, uint64_t end, uint64_t backing = 4096) {
    TensorFootprint f{};
    f.byte_offset = begin;
    f.backing_nbytes = backing;
    f.ndims = 1;
    f.shapes[0] = static_cast<uint32_t>(end - begin);
    f.strides[0] = 1;
    f.dtype = DataType::UINT8;
    return f;
}

// Helper: rows [r0, r0+rows) x cols [c0, c0+cols) of a `R`x`C` UINT8 matrix. Two of these can be
// disjoint while their bounding boxes intersect, which is what the per-dimension stage is for.
static TensorFootprint tile(uint32_t r0, uint32_t rows, uint32_t c0, uint32_t cols, uint32_t R, uint32_t C) {
    TensorFootprint f{};
    f.byte_offset = static_cast<uint64_t>(r0) * C + c0;
    f.backing_nbytes = static_cast<uint64_t>(R) * C;
    f.ndims = 2;
    f.shapes[0] = rows;
    f.shapes[1] = cols;
    f.strides[0] = C;
    f.strides[1] = 1;
    f.dtype = DataType::UINT8;
    return f;
}

// Helper: the whole backing — what an arg with no narrower footprint claims.
static TensorFootprint whole() { return TensorFootprint{}; }

// Helper: producers overlapping `view`, in insertion order.
static std::vector<TaskSlot> hits(const TensorMap &tm, RunId run_id, TensorKey key, const TensorFootprint &view) {
    std::vector<TaskSlot> out;
    tm.lookup_overlapping(run_id, key, view, out);
    return out;
}

// Helper: the single producer of the whole backing, or INVALID_SLOT when there is none. Asserts
// there is at most one — a test that uses this is stating the key holds one live writer.
static TaskSlot sole(const TensorMap &tm, RunId run_id, TensorKey key) {
    std::vector<TaskSlot> out = hits(tm, run_id, key, whole());
    EXPECT_LE(out.size(), 1u);
    return out.empty() ? INVALID_SLOT : out[0];
}

TEST(TensorMap, LookupEmptyReturnsInvalid) {
    TensorMap tm;
    EXPECT_EQ(sole(tm, RUN1, hk(0xDEADBEEF)), INVALID_SLOT);
}

TEST(TensorMap, InsertAndLookup) {
    TensorMap tm;
    tm.insert(RUN1, hk(0x1000), whole(), 5);
    EXPECT_EQ(sole(tm, RUN1, hk(0x1000)), 5);
    EXPECT_EQ(sole(tm, RUN1, hk(0x2000)), INVALID_SLOT);
    EXPECT_EQ(tm.size(), 1);
}

TEST(TensorMap, OverwriteExistingEntry) {
    TensorMap tm;
    tm.insert(RUN1, hk(0x1000), whole(), 3);
    tm.insert(RUN1, hk(0x1000), whole(), 7);  // new producer reuses same buffer
    EXPECT_EQ(sole(tm, RUN1, hk(0x1000)), 7);
    EXPECT_EQ(tm.size(), 1);
}

TEST(TensorMap, EraseTaskOutputs) {
    TensorMap tm;
    tm.insert(RUN1, hk(0x1000), whole(), 0);
    tm.insert(RUN1, hk(0x2000), whole(), 0);
    tm.insert(RUN1, hk(0x3000), whole(), 1);

    tm.erase_task_outputs(RUN1, 0, {hk(0x1000), hk(0x2000)});

    EXPECT_EQ(sole(tm, RUN1, hk(0x1000)), INVALID_SLOT);
    EXPECT_EQ(sole(tm, RUN1, hk(0x2000)), INVALID_SLOT);
    EXPECT_EQ(sole(tm, RUN1, hk(0x3000)), 1);
    EXPECT_EQ(tm.size(), 1);
}

TEST(TensorMap, EraseWithEmptyKeyList) {
    TensorMap tm;
    tm.insert(RUN1, hk(0x1000), whole(), 2);
    tm.erase_task_outputs(RUN1, 2, {});
    EXPECT_EQ(sole(tm, RUN1, hk(0x1000)), 2);
}

TEST(TensorMap, MultipleEntries) {
    TensorMap tm;
    for (int i = 0; i < 100; ++i)
        tm.insert(RUN1, hk(static_cast<uint64_t>(i) * 0x1000), whole(), i % 16);
    EXPECT_EQ(tm.size(), 100);
    for (int i = 0; i < 100; ++i)
        EXPECT_EQ(sole(tm, RUN1, hk(static_cast<uint64_t>(i) * 0x1000)), i % 16);
}

// --- TensorKey compound key tests ---

TEST(TensorMap, SamePtrDifferentEndpointAreDistinct) {
    TensorMap tm;
    tm.insert(RUN1, ck(0xABC, 0), whole(), 10);
    tm.insert(RUN1, ck(0xABC, 1), whole(), 20);
    EXPECT_EQ(sole(tm, RUN1, ck(0xABC, 0)), 10);
    EXPECT_EQ(sole(tm, RUN1, ck(0xABC, 1)), 20);
    EXPECT_EQ(tm.size(), 2);
}

TEST(TensorMap, HostAndChildKeyAreDistinct) {
    TensorMap tm;
    tm.insert(RUN1, hk(0x1000), whole(), 5);
    tm.insert(RUN1, ck(0x1000, 0), whole(), 6);
    EXPECT_EQ(sole(tm, RUN1, hk(0x1000)), 5);
    EXPECT_EQ(sole(tm, RUN1, ck(0x1000, 0)), 6);
    EXPECT_EQ(tm.size(), 2);
}

TEST(TensorMap, EraseChildKeyLeavesHostKey) {
    TensorMap tm;
    tm.insert(RUN1, hk(0x1000), whole(), 5);
    tm.insert(RUN1, ck(0x1000, 0), whole(), 6);
    tm.erase_task_outputs(RUN1, 6, {ck(0x1000, 0)});
    EXPECT_EQ(sole(tm, RUN1, hk(0x1000)), 5);
    EXPECT_EQ(sole(tm, RUN1, ck(0x1000, 0)), INVALID_SLOT);
    EXPECT_EQ(tm.size(), 1);
}

TEST(TensorMap, SameAddressIsNamespacedByRun) {
    TensorMap tm;
    tm.insert(RUN1, hk(0x1000), whole(), 5);
    tm.insert(RUN2, hk(0x1000), whole(), 9);

    EXPECT_EQ(sole(tm, RUN1, hk(0x1000)), 5);
    EXPECT_EQ(sole(tm, RUN2, hk(0x1000)), 9);
    EXPECT_EQ(tm.size(), 2);

    tm.erase_task_outputs(RUN1, 5, {hk(0x1000)});
    EXPECT_EQ(sole(tm, RUN1, hk(0x1000)), INVALID_SLOT);
    EXPECT_EQ(sole(tm, RUN2, hk(0x1000)), 9);
}

TEST(TensorMap, EraseKeepsNewerSameRunProducer) {
    TensorMap tm;
    tm.insert(RUN1, hk(0x1000), whole(), 3);
    tm.insert(RUN1, hk(0x1000), whole(), 7);  // unordered second writer of the same key

    // Slot 3 no longer owns the key; its cleanup must not evict slot 7.
    tm.erase_task_outputs(RUN1, 3, {hk(0x1000)});
    EXPECT_EQ(sole(tm, RUN1, hk(0x1000)), 7);
    EXPECT_EQ(tm.size(), 1);

    tm.erase_task_outputs(RUN1, 7, {hk(0x1000)});
    EXPECT_EQ(sole(tm, RUN1, hk(0x1000)), INVALID_SLOT);
    EXPECT_EQ(tm.size(), 0);
}

// --- Byte-range overlap: which same-key entries actually conflict ---

TEST(TensorMap, DisjointRangesUnderOneKeyAreIndependentProducers) {
    // The rank-major case: two workers write `x[0]` and `x[1]` of one backing. They share a key
    // by construction, and neither may resolve as the other's producer.
    TensorMap tm;
    tm.insert(RUN1, hk(0x1000), br(0, 64), 1);
    tm.insert(RUN1, hk(0x1000), br(64, 128), 2);

    EXPECT_EQ(tm.size(), 2);
    EXPECT_EQ(hits(tm, RUN1, hk(0x1000), br(0, 64)), (std::vector<TaskSlot>{1}));
    EXPECT_EQ(hits(tm, RUN1, hk(0x1000), br(64, 128)), (std::vector<TaskSlot>{2}));
}

TEST(TensorMap, OverlappingRangeResolvesEveryProducerItTouches) {
    TensorMap tm;
    tm.insert(RUN1, hk(0x1000), br(0, 64), 1);
    tm.insert(RUN1, hk(0x1000), br(64, 128), 2);

    // A reader of the whole backing depends on both half-writers.
    EXPECT_EQ(hits(tm, RUN1, hk(0x1000), whole()), (std::vector<TaskSlot>{1, 2}));
    // A reader straddling the boundary depends on both; one strictly inside, on one.
    EXPECT_EQ(hits(tm, RUN1, hk(0x1000), br(32, 96)), (std::vector<TaskSlot>{1, 2}));
    EXPECT_EQ(hits(tm, RUN1, hk(0x1000), br(8, 16)), (std::vector<TaskSlot>{1}));
}

TEST(TensorMap, TouchingRangesDoNotOverlap) {
    // Ranges are half-open, so `end == begin` of the next is adjacency, not a shared byte.
    TensorMap tm;
    tm.insert(RUN1, hk(0x1000), br(0, 64), 1);
    EXPECT_TRUE(hits(tm, RUN1, hk(0x1000), br(64, 128)).empty());
    EXPECT_EQ(hits(tm, RUN1, hk(0x1000), br(63, 128)), (std::vector<TaskSlot>{1}));
}

TEST(TensorMap, CoveringWriteSupersedesTheEntriesUnderIt) {
    TensorMap tm;
    tm.insert(RUN1, hk(0x1000), br(0, 64), 1);
    tm.insert(RUN1, hk(0x1000), br(64, 128), 2);
    tm.insert(RUN1, hk(0x1000), br(0, 128), 3);

    // Neither half-writer is the last word on any byte any more.
    EXPECT_EQ(tm.size(), 1);
    EXPECT_EQ(hits(tm, RUN1, hk(0x1000), whole()), (std::vector<TaskSlot>{3}));
}

TEST(TensorMap, PartiallyCoveredProducerSurvivesForItsOwnBytes) {
    TensorMap tm;
    tm.insert(RUN1, hk(0x1000), br(0, 128), 1);
    tm.insert(RUN1, hk(0x1000), br(0, 64), 2);  // rewrites only the first half

    EXPECT_EQ(tm.size(), 2);
    // Slot 1 still owns [64, 128), so a reader of that half must not lose the edge to it.
    EXPECT_EQ(hits(tm, RUN1, hk(0x1000), br(64, 128)), (std::vector<TaskSlot>{1}));
    EXPECT_EQ(hits(tm, RUN1, hk(0x1000), br(0, 64)), (std::vector<TaskSlot>{1, 2}));
}

TEST(TensorMap, EraseDropsEveryRangeATaskStillOwns) {
    TensorMap tm;
    tm.insert(RUN1, hk(0x1000), br(0, 64), 1);
    tm.insert(RUN1, hk(0x1000), br(64, 128), 1);  // one task writing two disjoint views
    tm.insert(RUN1, hk(0x1000), br(128, 192), 2);

    // output_keys records the key once per arg, so cleanup replays it twice; both of slot 1's
    // ranges go, and the unrelated writer stays.
    tm.erase_task_outputs(RUN1, 1, {hk(0x1000), hk(0x1000)});
    EXPECT_EQ(tm.size(), 1);
    EXPECT_EQ(hits(tm, RUN1, hk(0x1000), whole()), (std::vector<TaskSlot>{2}));
}

TEST(TensorMap, AWholeBackingWriteSupersedesEveryRangeUnderIt) {
    // What Orchestrator::alloc registers: no narrower footprint, so it owns the whole backing.
    TensorMap tm;
    tm.insert(RUN1, hk(0x1000), br(0, 64), 1);
    tm.insert(RUN1, hk(0x1000), whole(), 2);
    EXPECT_EQ(tm.size(), 1);
    EXPECT_EQ(hits(tm, RUN1, hk(0x1000), br(0, 8)), (std::vector<TaskSlot>{2}));
}

// --- Per-dimension refinement: bounding boxes intersect, the views do not ---

TEST(TensorMap, DisjointColumnBlocksAreIndependentProducers) {
    // `x[:, 0:4]` and `x[:, 8:12]` of a 16x16 matrix. Every row of the first sits between two rows
    // of the second, so their bounding ranges interleave -- [0, 244) and [8, 252) -- and stage 1
    // alone would wire an edge. The per-axis test rejects on the column axis.
    TensorMap tm;
    tm.insert(RUN1, hk(0x1000), tile(0, 16, 0, 4, 16, 16), 1);
    tm.insert(RUN1, hk(0x1000), tile(0, 16, 8, 4, 16, 16), 2);

    EXPECT_EQ(tm.size(), 2);
    EXPECT_EQ(hits(tm, RUN1, hk(0x1000), tile(0, 16, 0, 4, 16, 16)), (std::vector<TaskSlot>{1}));
    EXPECT_EQ(hits(tm, RUN1, hk(0x1000), tile(0, 16, 8, 4, 16, 16)), (std::vector<TaskSlot>{2}));
}

TEST(TensorMap, DisjointTilesAreIndependentProducers) {
    // A 2x2 grid of 8x8 tiles over one 16x16 matrix: four writers, no edge between any pair.
    TensorMap tm;
    tm.insert(RUN1, hk(0x1000), tile(0, 8, 0, 8, 16, 16), 1);
    tm.insert(RUN1, hk(0x1000), tile(0, 8, 8, 8, 16, 16), 2);
    tm.insert(RUN1, hk(0x1000), tile(8, 8, 0, 8, 16, 16), 3);
    tm.insert(RUN1, hk(0x1000), tile(8, 8, 8, 8, 16, 16), 4);

    EXPECT_EQ(tm.size(), 4);
    EXPECT_EQ(hits(tm, RUN1, hk(0x1000), tile(0, 8, 0, 8, 16, 16)), (std::vector<TaskSlot>{1}));
    EXPECT_EQ(hits(tm, RUN1, hk(0x1000), tile(8, 8, 8, 8, 16, 16)), (std::vector<TaskSlot>{4}));
}

TEST(TensorMap, TilesSharingABlockStillDepend) {
    // The converse: tiles that do share elements keep their edge.
    TensorMap tm;
    tm.insert(RUN1, hk(0x1000), tile(0, 8, 0, 8, 16, 16), 1);
    EXPECT_EQ(hits(tm, RUN1, hk(0x1000), tile(4, 8, 4, 8, 16, 16)), (std::vector<TaskSlot>{1}));
    EXPECT_EQ(hits(tm, RUN1, hk(0x1000), tile(0, 16, 0, 16, 16, 16)), (std::vector<TaskSlot>{1}));
}

TEST(TensorMap, ATileWriteSupersedesOnlyTheTilesItContains) {
    TensorMap tm;
    tm.insert(RUN1, hk(0x1000), tile(0, 4, 0, 4, 16, 16), 1);
    tm.insert(RUN1, hk(0x1000), tile(0, 4, 8, 4, 16, 16), 2);
    // Rewrites the top-left quadrant: covers slot 1 whole, does not touch slot 2's columns.
    tm.insert(RUN1, hk(0x1000), tile(0, 8, 0, 8, 16, 16), 3);

    EXPECT_EQ(tm.size(), 2);
    EXPECT_EQ(hits(tm, RUN1, hk(0x1000), tile(0, 4, 0, 4, 16, 16)), (std::vector<TaskSlot>{3}));
    EXPECT_EQ(hits(tm, RUN1, hk(0x1000), tile(0, 4, 8, 4, 16, 16)), (std::vector<TaskSlot>{2}));
}

TEST(TensorMap, ARowBlockAndAColumnBlockCross) {
    // A row block and a column block of one matrix always share a corner, whatever their offsets.
    TensorMap tm;
    tm.insert(RUN1, hk(0x1000), tile(0, 4, 0, 16, 16, 16), 1);
    EXPECT_EQ(hits(tm, RUN1, hk(0x1000), tile(0, 16, 12, 4, 16, 16)), (std::vector<TaskSlot>{1}));
}

TEST(TensorMap, MismatchedAxisLayoutsFallBackToConservative) {
    // Stage 2 models only pairs sharing one canonical row-major layout. A 16x16 view and a flat
    // 256-element view of the same backing do not, so they are assumed to conflict even where a
    // per-element answer would say otherwise.
    TensorMap tm;
    tm.insert(RUN1, hk(0x1000), tile(0, 4, 0, 4, 16, 16), 1);
    EXPECT_EQ(hits(tm, RUN1, hk(0x1000), br(0, 256, 256)), (std::vector<TaskSlot>{1}));
    EXPECT_EQ(hits(tm, RUN1, hk(0x1000), br(200, 256, 256)), (std::vector<TaskSlot>{}));
}
