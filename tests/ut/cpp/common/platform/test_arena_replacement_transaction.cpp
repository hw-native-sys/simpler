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

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <set>
#include <vector>

#include "host/arena_replacement_transaction.h"
#include "utils/device_arena.h"

namespace {

/**
 * Stands in for the host MemoryAllocator: it can refuse a chosen allocation,
 * and it can refuse a free, which the real allocator answers by keeping the
 * block's map entry so the existing finalize path still owns it.
 */
struct InjectingBackend {
    // 1-based index of the allocation to refuse; 0 refuses none.
    int fail_alloc_at = 0;
    bool fail_frees = false;
    int allocs = 0;
    int frees = 0;
    // Blocks the backend still owns. A refused free keeps its entry.
    std::set<void *> live;
    // Where the destructor records how many blocks it had to reclaim, for a
    // test that outlives this backend.
    size_t *reclaimed_out = nullptr;

    static void *alloc(void *ctx, size_t size) {
        auto *self = static_cast<InjectingBackend *>(ctx);
        ++self->allocs;
        if (self->allocs == self->fail_alloc_at) return nullptr;
        void *p = std::malloc(size);
        if (p != nullptr) self->live.insert(p);
        return p;
    }

    static void free_fn(void *ctx, void *ptr) {
        auto *self = static_cast<InjectingBackend *>(ctx);
        ++self->frees;
        if (self->fail_frees) return;
        self->live.erase(ptr);
        std::free(ptr);
    }

    // Final owner of whatever a refused free left behind. The production
    // allocator keeps such a block's map entry so its finalize path still owns
    // it, and this backend has to be the equivalent terminal owner: it is
    // declared ahead of every arena that borrows it, so it is destroyed last
    // and reclaims the retained blocks however the test ends, assertion
    // failures included.
    ~InjectingBackend() {
        if (reclaimed_out != nullptr) *reclaimed_out = live.size();
        for (void *ptr : live)
            std::free(ptr);
        live.clear();
    }
};

/** The three pooled regions of one arena bank, in the order the runner lists them. */
struct BankFixture {
    InjectingBackend backend;
    DeviceArena gm_heap{&InjectingBackend::alloc, &InjectingBackend::free_fn, &backend};
    DeviceArena gm_sm{&InjectingBackend::alloc, &InjectingBackend::free_fn, &backend};
    DeviceArena runtime_pool{&InjectingBackend::alloc, &InjectingBackend::free_fn, &backend};
    size_t cached_gm_heap = 0;
    size_t cached_gm_sm = 0;
    size_t cached_runtime_arena = 0;

    ArenaRegionRequest requests(size_t heap, size_t sm, size_t runtime, ArenaRegionRequest (&out)[3]) {
        out[0] = ArenaRegionRequest{&gm_heap, &cached_gm_heap, heap, "gm_heap"};
        out[1] = ArenaRegionRequest{&gm_sm, &cached_gm_sm, sm, "gm_sm"};
        out[2] = ArenaRegionRequest{&runtime_pool, &cached_runtime_arena, runtime, "runtime_pool"};
        return out[0];
    }

    ArenaTransactionResult run(size_t heap, size_t sm, size_t runtime) {
        ArenaRegionRequest reqs[3];
        (void)requests(heap, sm, runtime, reqs);
        return run_arena_replacement_transaction(reqs, 3, DeviceArena::kDefaultBaseAlign);
    }

    DeviceArena *arena(size_t i) {
        DeviceArena *all[] = {&gm_heap, &gm_sm, &runtime_pool};
        return all[i];
    }

    size_t cached(size_t i) const {
        const size_t all[] = {cached_gm_heap, cached_gm_sm, cached_runtime_arena};
        return all[i];
    }
};

struct BankSnapshot {
    void *bases[3];
    size_t cached[3];
    bool committed[3];

    static BankSnapshot of(BankFixture &bank) {
        BankSnapshot snap{};
        for (size_t i = 0; i < 3; ++i) {
            snap.bases[i] = bank.arena(i)->base();
            snap.cached[i] = bank.cached(i);
            snap.committed[i] = bank.arena(i)->is_committed();
        }
        return snap;
    }
};

void expect_bank_unchanged(BankFixture &bank, const BankSnapshot &before) {
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_EQ(bank.arena(i)->base(), before.bases[i]) << "region " << i << " base moved";
        EXPECT_EQ(bank.cached(i), before.cached[i]) << "region " << i << " capacity changed";
        EXPECT_EQ(bank.arena(i)->is_committed(), before.committed[i]) << "region " << i << " commit state changed";
        EXPECT_FALSE(bank.arena(i)->has_staged_replacement()) << "region " << i << " kept a staged block";
    }
}

// The trb caller's shape: all three regions carry bytes.
constexpr size_t kHeap = 4096;
constexpr size_t kSm = 2048;
constexpr size_t kRuntime = 1024;

void commit_initial_bank(BankFixture &bank) {
    const ArenaTransactionResult first = bank.run(kHeap, kSm, kRuntime);
    ASSERT_TRUE(first.published);
    ASSERT_EQ(first.changed_regions, 3u);
    for (size_t i = 0; i < 3; ++i) {
        ASSERT_TRUE(bank.arena(i)->is_committed());
        ASSERT_NE(bank.arena(i)->base(), nullptr);
    }
}

/**
 * What an owner that keeps its own ledger of these allocations is told.
 *
 * Such an owner refuses the arena's frees, so the free callback cannot tell it
 * that a claim ended; these reports are the only thing that can.
 */
struct DispositionLog {
    struct Entry {
        size_t region;
        ArenaRegionDisposition what;
        void *base;
    };
    std::vector<Entry> entries;

    static void record(void *ctx, size_t region, ArenaRegionDisposition what, void *base) {
        static_cast<DispositionLog *>(ctx)->entries.push_back(Entry{region, what, base});
    }

    size_t count(ArenaRegionDisposition what) const {
        size_t n = 0;
        for (const Entry &entry : entries) {
            if (entry.what == what) ++n;
        }
        return n;
    }
};

void fill_reporting_requests(
    BankFixture &bank, size_t heap, size_t sm, size_t runtime, DispositionLog *log, ArenaRegionRequest (&out)[3]
) {
    (void)bank.requests(heap, sm, runtime, out);
    for (ArenaRegionRequest &request : out) {
        request.disposition = &DispositionLog::record;
        request.disposition_ctx = log;
    }
}

TEST(ArenaReplacementTransaction, FirstRegionStageFailureKeepsEveryCommittedRegion) {
    BankFixture bank;
    ASSERT_NO_FATAL_FAILURE(commit_initial_bank(bank));
    const BankSnapshot before = BankSnapshot::of(bank);
    const int frees_before = bank.backend.frees;

    bank.backend.fail_alloc_at = bank.backend.allocs + 1;
    const ArenaTransactionResult grow = bank.run(kHeap * 2, kSm * 2, kRuntime * 2);

    EXPECT_FALSE(grow.published);
    EXPECT_EQ(grow.failed_region, 0);
    EXPECT_EQ(grow.changed_regions, 0u);
    EXPECT_EQ(bank.backend.frees, frees_before);
    expect_bank_unchanged(bank, before);
}

TEST(ArenaReplacementTransaction, SecondRegionStageFailureKeepsEveryCommittedRegion) {
    BankFixture bank;
    ASSERT_NO_FATAL_FAILURE(commit_initial_bank(bank));
    const BankSnapshot before = BankSnapshot::of(bank);

    bank.backend.fail_alloc_at = bank.backend.allocs + 2;
    const ArenaTransactionResult grow = bank.run(kHeap * 2, kSm * 2, kRuntime * 2);

    EXPECT_FALSE(grow.published);
    EXPECT_EQ(grow.failed_region, 1);
    expect_bank_unchanged(bank, before);
    // The first region's staged block is the only one allocated, and it is gone.
    EXPECT_EQ(bank.backend.live.size(), 3u);
}

TEST(ArenaReplacementTransaction, ThirdRegionStageFailureKeepsBothSuccessfulPeers) {
    // The case a release-then-allocate sequence cannot preserve: two regions
    // have already been replaced by the time the third one fails.
    BankFixture bank;
    ASSERT_NO_FATAL_FAILURE(commit_initial_bank(bank));
    const BankSnapshot before = BankSnapshot::of(bank);

    bank.backend.fail_alloc_at = bank.backend.allocs + 3;
    const ArenaTransactionResult grow = bank.run(kHeap * 2, kSm * 2, kRuntime * 2);

    EXPECT_FALSE(grow.published);
    EXPECT_EQ(grow.failed_region, 2);
    expect_bank_unchanged(bank, before);
    EXPECT_EQ(bank.backend.live.size(), 3u);
}

TEST(ArenaReplacementTransaction, ZeroRequestSurvivesAPeersStageFailure) {
    // A release driven by a zero request must not run before the transaction
    // can publish, or a peer's allocation failure would take the region with
    // it. Asked for on the last region here; hbg's actual shape asks on the
    // middle one, which HbgShapedGrowthFailureKeepsTheUncommittedSharedMemory
    // covers.
    BankFixture bank;
    ASSERT_NO_FATAL_FAILURE(commit_initial_bank(bank));
    const BankSnapshot before = BankSnapshot::of(bank);

    bank.backend.fail_alloc_at = bank.backend.allocs + 1;
    const ArenaTransactionResult grow = bank.run(kHeap * 2, kSm, /*runtime=*/0);

    EXPECT_FALSE(grow.published);
    EXPECT_EQ(grow.failed_region, 0);
    EXPECT_TRUE(bank.runtime_pool.is_committed());
    EXPECT_EQ(bank.cached_runtime_arena, kRuntime);
    expect_bank_unchanged(bank, before);
}

TEST(ArenaReplacementTransaction, PublicationMovesBasesUpdatesCapacitiesAndFreesSuperseded) {
    BankFixture bank;
    ASSERT_NO_FATAL_FAILURE(commit_initial_bank(bank));
    const BankSnapshot before = BankSnapshot::of(bank);
    const int frees_before = bank.backend.frees;

    const ArenaTransactionResult grow = bank.run(kHeap * 2, kSm * 2, kRuntime * 2);

    EXPECT_TRUE(grow.published);
    EXPECT_EQ(grow.failed_region, -1);
    EXPECT_EQ(grow.changed_regions, 3u);
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_TRUE(bank.arena(i)->is_committed());
        EXPECT_NE(bank.arena(i)->base(), before.bases[i]);
        EXPECT_EQ(bank.cached(i), before.cached[i] * 2);
        EXPECT_FALSE(bank.arena(i)->has_staged_replacement());
    }
    // Exactly the three superseded blocks, freed after every region published.
    EXPECT_EQ(bank.backend.frees, frees_before + 3);
    EXPECT_EQ(bank.backend.live.size(), 3u);
}

TEST(ArenaReplacementTransaction, ZeroRequestReleaseLandsOnPublication) {
    BankFixture bank;
    ASSERT_NO_FATAL_FAILURE(commit_initial_bank(bank));
    const int frees_before = bank.backend.frees;

    const ArenaTransactionResult release = bank.run(kHeap, kSm, /*runtime=*/0);

    EXPECT_TRUE(release.published);
    EXPECT_EQ(release.changed_regions, 1u);
    EXPECT_FALSE(bank.runtime_pool.is_committed());
    EXPECT_EQ(bank.cached_runtime_arena, 0u);
    EXPECT_EQ(bank.backend.frees, frees_before + 1);
    // The two regions whose request was unchanged were not touched.
    EXPECT_TRUE(bank.gm_heap.is_committed());
    EXPECT_TRUE(bank.gm_sm.is_committed());
}

TEST(ArenaReplacementTransaction, SmallerRequestAfterAFailedGrowthReusesTheKeptBlocks) {
    // What the preservation is for: the run that asked for more fails, and a
    // later compatible request still runs on the regions this bank already has.
    BankFixture bank;
    ASSERT_NO_FATAL_FAILURE(commit_initial_bank(bank));
    const BankSnapshot before = BankSnapshot::of(bank);

    bank.backend.fail_alloc_at = bank.backend.allocs + 1;
    ASSERT_FALSE(bank.run(kHeap * 2, kSm, kRuntime).published);

    bank.backend.fail_alloc_at = 0;
    const int allocs_before = bank.backend.allocs;
    const int frees_before = bank.backend.frees;
    const ArenaTransactionResult smaller = bank.run(kHeap / 2, kSm, kRuntime);

    EXPECT_TRUE(smaller.published);
    EXPECT_EQ(smaller.changed_regions, 0u);
    EXPECT_EQ(bank.backend.allocs, allocs_before);
    EXPECT_EQ(bank.backend.frees, frees_before);
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_EQ(bank.arena(i)->base(), before.bases[i]);
        EXPECT_EQ(bank.cached(i), before.cached[i]);
    }
}

TEST(ArenaReplacementTransaction, WarmRepeatOfTheSameLayoutAllocatesNothing) {
    BankFixture bank;
    ASSERT_NO_FATAL_FAILURE(commit_initial_bank(bank));
    const int allocs_before = bank.backend.allocs;
    const int frees_before = bank.backend.frees;

    const ArenaTransactionResult warm = bank.run(kHeap, kSm, kRuntime);

    EXPECT_TRUE(warm.published);
    EXPECT_EQ(warm.changed_regions, 0u);
    EXPECT_EQ(bank.backend.allocs, allocs_before);
    EXPECT_EQ(bank.backend.frees, frees_before);
}

TEST(ArenaReplacementTransaction, StagedBlockWhoseReleaseFailsStaysWithTheAllocator) {
    // A rollback free that fails must not lose the block's only owner, and it
    // must not turn the staging failure into a different error.
    BankFixture bank;
    ASSERT_NO_FATAL_FAILURE(commit_initial_bank(bank));
    const BankSnapshot before = BankSnapshot::of(bank);

    bank.backend.fail_frees = true;
    bank.backend.fail_alloc_at = bank.backend.allocs + 2;
    const ArenaTransactionResult grow = bank.run(kHeap * 2, kSm * 2, kRuntime);

    EXPECT_FALSE(grow.published);
    EXPECT_EQ(grow.failed_region, 1);
    expect_bank_unchanged(bank, before);
    // Three committed regions plus the staged block the refused free retained:
    // the allocator still owns it, for its existing finalize path to retire.
    EXPECT_EQ(bank.backend.live.size(), 4u);
}

TEST(ArenaReplacementTransaction, FirstCommitOfAnEmptyBankNeedsNoPriorBacking) {
    // Two regions to establish and one asked for 0, on an empty bank.
    BankFixture bank;
    const ArenaTransactionResult first = bank.run(kHeap, kSm, /*runtime=*/0);

    EXPECT_TRUE(first.published);
    EXPECT_EQ(first.changed_regions, 2u);
    EXPECT_TRUE(bank.gm_heap.is_committed());
    EXPECT_TRUE(bank.gm_sm.is_committed());
    EXPECT_FALSE(bank.runtime_pool.is_committed());
    EXPECT_EQ(bank.cached_runtime_arena, 0u);
    // Nothing was superseded, so nothing was freed.
    EXPECT_EQ(bank.backend.frees, 0);
}

TEST(ArenaReplacementTransaction, PublishedRegionCarriesTheRequestedCapacityAtOffsetZero) {
    // The published layout must be what reserve+commit produced, since
    // acquire_pooled_* hands base() out and the region table backs region_ptr.
    BankFixture bank;
    ASSERT_NO_FATAL_FAILURE(commit_initial_bank(bank));

    ASSERT_TRUE(bank.run(kHeap * 2, kSm, kRuntime).published);

    EXPECT_EQ(bank.gm_heap.total_size(), kHeap * 2);
    EXPECT_EQ(bank.gm_heap.region_size(0), kHeap * 2);
    EXPECT_EQ(bank.gm_heap.region_ptr(0), bank.gm_heap.base());
    EXPECT_EQ(reinterpret_cast<uintptr_t>(bank.gm_heap.base()) % DeviceArena::kDefaultBaseAlign, 0u);
}

TEST(ArenaReplacementTransaction, ArenaStagedBlockIsFreedWhenTheArenaIsReleased) {
    // release() owns the staged block too, so an abandoned transaction cannot
    // outlive its arena with a block nobody frees.
    InjectingBackend backend;
    {
        DeviceArena arena(&InjectingBackend::alloc, &InjectingBackend::free_fn, &backend);
        arena.reserve(kHeap, 64);
        ASSERT_NE(arena.commit(), nullptr);
        ASSERT_NE(arena.stage_replacement(kHeap * 2), nullptr);
        EXPECT_TRUE(arena.has_staged_replacement());
        EXPECT_EQ(backend.live.size(), 2u);
    }
    EXPECT_TRUE(backend.live.empty());
}

TEST(ArenaReplacementTransaction, RejectsARegionCountItCannotPlan) {
    BankFixture bank;
    ArenaRegionRequest reqs[3];
    (void)bank.requests(kHeap, kSm, kRuntime, reqs);

    const ArenaTransactionResult none = run_arena_replacement_transaction(reqs, 0);
    EXPECT_FALSE(none.published);
    EXPECT_EQ(none.failed_region, -1) << "an argument error must not name a region as having failed";
    const ArenaTransactionResult too_many = run_arena_replacement_transaction(reqs, kMaxArenaTransactionRegions + 1);
    EXPECT_FALSE(too_many.published);
    EXPECT_EQ(too_many.failed_region, -1);
    EXPECT_FALSE(bank.gm_heap.is_committed());
}

TEST(ArenaReplacementTransaction, StagingRefusesASizeItCannotAlign) {
    // A size within base_align-1 of SIZE_MAX would wrap the forward-alignment
    // arithmetic into a small request the allocator can satisfy, after which
    // publication would advertise the unwrapped capacity against that small
    // block. The refusal happens before the allocator is called, so it is a
    // staging failure like any other and the peers are preserved.
    BankFixture bank;
    ASSERT_NO_FATAL_FAILURE(commit_initial_bank(bank));
    const BankSnapshot before = BankSnapshot::of(bank);
    const int allocs_before = bank.backend.allocs;

    const size_t unalignable = SIZE_MAX - (DeviceArena::kDefaultBaseAlign - 2);
    const ArenaTransactionResult grow = bank.run(unalignable, kSm, kRuntime);

    EXPECT_FALSE(grow.published);
    EXPECT_EQ(grow.failed_region, 0);
    EXPECT_EQ(bank.backend.allocs, allocs_before) << "the allocator was asked for a wrapped size";
    expect_bank_unchanged(bank, before);

    // One byte less is the largest size the guard still lets through, so the
    // boundary is a size_t limit rather than a new capacity cap. The backend
    // refuses that allocation on injection: whether a real allocator returns
    // null or aborts for an absurd request is its business, and the assertion
    // here is only that the guard handed the size on rather than rejecting it.
    const size_t largest_alignable = SIZE_MAX - (DeviceArena::kDefaultBaseAlign - 1);
    bank.backend.fail_alloc_at = bank.backend.allocs + 1;
    EXPECT_EQ(bank.gm_heap.stage_replacement(largest_alignable), nullptr);
    EXPECT_EQ(bank.backend.allocs, allocs_before + 1) << "the guard rejected a size it can align";
    EXPECT_FALSE(bank.gm_heap.has_staged_replacement());
    expect_bank_unchanged(bank, before);
}

// ---------------------------------------------------------------------------
// The seam setup_static_arena itself runs: the transaction over a bank's three
// regions plus the disposition of the prebuilt-image entry, and the rc the
// caller propagates. `DeviceRunnerBase` cannot be constructed here — its
// translation unit leaves ~190 symbols undefined, including the CANN rt/acl
// driver entries, MemoryAllocator, and the collector subsystems — so these
// drive the same `run_bank_arena_setup` the runner calls, against the same
// `PrebuiltRuntimeArenaCache` type it holds as a member.
// ---------------------------------------------------------------------------

constexpr uint64_t kCacheHash = 0xA5A5'1234'0000'0001ULL;
const unsigned char kCacheKey[] = {'r', 'i', 'n', 'g'};
const unsigned char kCacheImage[] = {1, 2, 3, 4, 5, 6, 7, 8};

struct CacheProbe {
    void *gm_heap_base = nullptr;
    void *sm_base = nullptr;
    void *runtime_arena_base = nullptr;
    size_t runtime_off = 0;
    const void *image_data = nullptr;
    size_t image_size = 0;

    bool lookup(const PrebuiltRuntimeArenaCache &cache) {
        return cache.lookup(
            kCacheHash, kCacheKey, sizeof(kCacheKey), &gm_heap_base, &sm_base, &runtime_arena_base, &runtime_off,
            &image_data, &image_size
        );
    }
};

void store_entry_for(BankFixture &bank, PrebuiltRuntimeArenaCache &cache) {
    cache.store(
        kCacheHash, kCacheKey, sizeof(kCacheKey), bank.gm_heap.base(), bank.gm_sm.base(), bank.runtime_pool.base(),
        /*runtime_off=*/64, kCacheImage, sizeof(kCacheImage)
    );
    ASSERT_TRUE(cache.is_valid());
}

BankArenaSetupOutcome setup_bank(
    BankFixture &bank, size_t heap, size_t sm, size_t runtime, uint32_t bank_id, PrebuiltRuntimeArenaCache *cache
) {
    ArenaRegionRequest reqs[3];
    (void)bank.requests(heap, sm, runtime, reqs);
    return run_bank_arena_setup(reqs, 3, /*owns_prebuilt_cache=*/bank_id == 0, cache, DeviceArena::kDefaultBaseAlign);
}

TEST(ArenaBankSetupSeam, FailedGrowthReturnsErrorAndLeavesTheCachedImageAnswerable) {
    // The two halves the caller depends on: the rc it propagates is an error
    // rather than a success at the old capacity, and the entry describing this
    // bank's bases is still answerable because no base moved.
    BankFixture bank;
    ASSERT_NO_FATAL_FAILURE(commit_initial_bank(bank));
    PrebuiltRuntimeArenaCache cache;
    ASSERT_NO_FATAL_FAILURE(store_entry_for(bank, cache));
    const BankSnapshot before = BankSnapshot::of(bank);

    bank.backend.fail_alloc_at = bank.backend.allocs + 3;
    const BankArenaSetupOutcome outcome = setup_bank(bank, kHeap * 2, kSm * 2, kRuntime * 2, /*bank_id=*/0, &cache);

    EXPECT_NE(outcome.rc, 0) << "a failed growth must not report success";
    EXPECT_FALSE(outcome.transaction.published);
    EXPECT_FALSE(outcome.cache_invalidated);
    expect_bank_unchanged(bank, before);

    EXPECT_TRUE(cache.is_valid());
    CacheProbe probe;
    ASSERT_TRUE(probe.lookup(cache)) << "the entry stopped answering after a failure that moved no base";
    EXPECT_EQ(probe.gm_heap_base, before.bases[0]);
    EXPECT_EQ(probe.sm_base, before.bases[1]);
    EXPECT_EQ(probe.runtime_arena_base, before.bases[2]);
    EXPECT_EQ(probe.runtime_off, 64u);
    EXPECT_EQ(probe.image_size, sizeof(kCacheImage));
    EXPECT_EQ(std::memcmp(probe.image_data, kCacheImage, sizeof(kCacheImage)), 0);
}

TEST(ArenaBankSetupSeam, PublishedGrowthSucceedsAndRetiresTheCachedImage) {
    BankFixture bank;
    ASSERT_NO_FATAL_FAILURE(commit_initial_bank(bank));
    PrebuiltRuntimeArenaCache cache;
    ASSERT_NO_FATAL_FAILURE(store_entry_for(bank, cache));

    const BankArenaSetupOutcome outcome = setup_bank(bank, kHeap * 2, kSm, kRuntime, /*bank_id=*/0, &cache);

    EXPECT_EQ(outcome.rc, 0);
    EXPECT_TRUE(outcome.transaction.published);
    EXPECT_EQ(outcome.transaction.changed_regions, 1u);
    EXPECT_TRUE(outcome.cache_invalidated);
    EXPECT_FALSE(cache.is_valid()) << "a moved base left the entry advertising an address it no longer owns";
    CacheProbe probe;
    EXPECT_FALSE(probe.lookup(cache));
}

TEST(ArenaBankSetupSeam, WarmReuseSucceedsAndKeepsTheCachedImage) {
    BankFixture bank;
    ASSERT_NO_FATAL_FAILURE(commit_initial_bank(bank));
    PrebuiltRuntimeArenaCache cache;
    ASSERT_NO_FATAL_FAILURE(store_entry_for(bank, cache));
    const int allocs_before = bank.backend.allocs;

    const BankArenaSetupOutcome outcome = setup_bank(bank, kHeap, kSm, kRuntime, /*bank_id=*/0, &cache);

    EXPECT_EQ(outcome.rc, 0);
    EXPECT_EQ(outcome.transaction.changed_regions, 0u);
    EXPECT_FALSE(outcome.cache_invalidated);
    EXPECT_EQ(bank.backend.allocs, allocs_before);
    CacheProbe probe;
    EXPECT_TRUE(probe.lookup(cache)) << "a repeat of a covered layout must keep skipping the rebuild";
}

TEST(ArenaBankSetupSeam, ANonZeroBankDoesNotRetireBankZerosCachedImage) {
    // The entry describes bank 0's bases, so another bank's publication has no
    // business retiring it.
    BankFixture bank_zero;
    ASSERT_NO_FATAL_FAILURE(commit_initial_bank(bank_zero));
    PrebuiltRuntimeArenaCache cache;
    ASSERT_NO_FATAL_FAILURE(store_entry_for(bank_zero, cache));

    BankFixture bank_one;
    const BankArenaSetupOutcome outcome = setup_bank(bank_one, kHeap, kSm, kRuntime, /*bank_id=*/1, &cache);

    EXPECT_EQ(outcome.rc, 0);
    EXPECT_EQ(outcome.transaction.changed_regions, 3u);
    EXPECT_FALSE(outcome.cache_invalidated);
    EXPECT_TRUE(cache.is_valid());
    CacheProbe probe;
    EXPECT_TRUE(probe.lookup(cache));
}

TEST(ArenaBankSetupSeam, HbgShapedGrowthFailureKeepsTheUncommittedSharedMemory) {
    // hbg's actual request shape, both arches: setup_static_arena(heap, 0,
    // device_arena_bytes) — the zero is the middle region, and the runtime
    // arena carries bytes and is a region hbg goes on to acquire.
    BankFixture bank;
    const BankArenaSetupOutcome first = setup_bank(bank, kHeap, /*sm=*/0, kRuntime, /*bank_id=*/0, nullptr);
    ASSERT_EQ(first.rc, 0);
    ASSERT_TRUE(bank.gm_heap.is_committed());
    ASSERT_FALSE(bank.gm_sm.is_committed()) << "hbg leaves shared memory uncommitted";
    ASSERT_TRUE(bank.runtime_pool.is_committed()) << "hbg's runtime arena carries bytes";
    ASSERT_EQ(bank.cached_gm_sm, 0u);
    const BankSnapshot before = BankSnapshot::of(bank);

    // A later bind whose graph needs more heap, with the heap allocation refused.
    bank.backend.fail_alloc_at = bank.backend.allocs + 1;
    const BankArenaSetupOutcome grow = setup_bank(bank, kHeap * 4, /*sm=*/0, kRuntime, /*bank_id=*/0, nullptr);

    EXPECT_NE(grow.rc, 0);
    expect_bank_unchanged(bank, before);
    EXPECT_FALSE(bank.gm_sm.is_committed());

    // And the heap this bind could still run on is the one it already had.
    bank.backend.fail_alloc_at = 0;
    const int allocs_before = bank.backend.allocs;
    const BankArenaSetupOutcome retry = setup_bank(bank, kHeap, /*sm=*/0, kRuntime, /*bank_id=*/0, nullptr);
    EXPECT_EQ(retry.rc, 0);
    EXPECT_EQ(retry.transaction.changed_regions, 0u);
    EXPECT_EQ(bank.backend.allocs, allocs_before);
    EXPECT_EQ(bank.gm_heap.base(), before.bases[0]);
}

TEST(ArenaBankSetupSeam, TrbShapedGrowthPublishesAllThreeRegions) {
    // trb's actual request shape: all three regions carry bytes, and they are
    // the bank the prebuilt entry describes.
    BankFixture bank;
    PrebuiltRuntimeArenaCache cache;
    const BankArenaSetupOutcome first = setup_bank(bank, kHeap, kSm, kRuntime, /*bank_id=*/0, &cache);

    EXPECT_EQ(first.rc, 0);
    EXPECT_EQ(first.transaction.changed_regions, 3u);
    EXPECT_TRUE(bank.gm_heap.is_committed());
    EXPECT_TRUE(bank.gm_sm.is_committed());
    EXPECT_TRUE(bank.runtime_pool.is_committed());
    // The bases moved, so the owning bank runs the entry's invalidation; there
    // was nothing stored for it to retire, and it stays unanswerable.
    EXPECT_TRUE(first.cache_invalidated);
    EXPECT_FALSE(cache.is_valid());
}

TEST(ArenaReplacementTransaction, RefusedFreesAreReclaimedByTheBackendAfterItsArenas) {
    // The retained-block simulation must not become a fixture leak: while the
    // arenas live the refused frees stay tracked, and once they are gone the
    // backend is the terminal owner that reclaims them — the same division the
    // production allocator has with its finalize path.
    size_t reclaimed = 0;
    {
        BankFixture bank;
        bank.backend.reclaimed_out = &reclaimed;
        ASSERT_NO_FATAL_FAILURE(commit_initial_bank(bank));

        bank.backend.fail_frees = true;
        bank.backend.fail_alloc_at = bank.backend.allocs + 2;
        const ArenaTransactionResult grow = bank.run(kHeap * 2, kSm * 2, kRuntime);

        ASSERT_FALSE(grow.published);
        // Three committed regions plus the staged block whose free was refused:
        // all four are still tracked while an owner exists.
        EXPECT_EQ(bank.backend.live.size(), 4u);
        EXPECT_EQ(reclaimed, 0u) << "cleanup ran before the arenas were done with the blocks";
    }
    EXPECT_EQ(reclaimed, 4u) << "the backend left blocks behind after its arenas were destroyed";
}

TEST(ArenaReplacementTransaction, ClaimEndingBoundariesAreReportedToTheAllocationsOwner) {
    BankFixture bank;
    ASSERT_NO_FATAL_FAILURE(commit_initial_bank(bank));
    void *committed[3];
    for (size_t i = 0; i < 3; ++i)
        committed[i] = bank.arena(i)->raw_backing();

    // A later region's staging failure drops the blocks its peers already
    // staged. Those never became the generation their region publishes, so an
    // owner has to hear that the claim ended — otherwise it keeps protecting a
    // block nothing will ever name.
    DispositionLog aborted;
    ArenaRegionRequest grow[3];
    fill_reporting_requests(bank, kHeap * 2, kSm * 2, kRuntime * 2, &aborted, grow);
    bank.backend.fail_alloc_at = bank.backend.allocs + 2;
    const ArenaTransactionResult failed = run_arena_replacement_transaction(grow, 3, DeviceArena::kDefaultBaseAlign);

    ASSERT_FALSE(failed.published);
    ASSERT_EQ(aborted.entries.size(), 1u);
    EXPECT_EQ(aborted.entries[0].region, 0u);
    EXPECT_EQ(aborted.entries[0].what, ArenaRegionDisposition::StageAborted);
    ASSERT_NE(aborted.entries[0].base, nullptr);
    // The block reported is the staged one, never a committed backing: giving up
    // a claim on one of those would abandon storage its region is still using.
    for (size_t i = 0; i < 3; ++i)
        EXPECT_NE(aborted.entries[0].base, committed[i]) << "reported region " << i << "'s live backing";
    EXPECT_EQ(aborted.count(ArenaRegionDisposition::Detached), 0u);

    // A region reduced to nothing publishes no address at all, so its old block
    // is reported too — there is no successor for an owner to settle it
    // against.
    DispositionLog detached;
    ArenaRegionRequest release[3];
    fill_reporting_requests(bank, kHeap, 0, kRuntime, &detached, release);
    const ArenaTransactionResult published =
        run_arena_replacement_transaction(release, 3, DeviceArena::kDefaultBaseAlign);

    ASSERT_TRUE(published.published);
    ASSERT_EQ(detached.entries.size(), 1u);
    EXPECT_EQ(detached.entries[0].region, 1u);
    EXPECT_EQ(detached.entries[0].what, ArenaRegionDisposition::Detached);
    EXPECT_EQ(detached.entries[0].base, committed[1]);
    EXPECT_FALSE(bank.arena(1)->is_committed());
    // The regions that kept their backing report nothing, and a publication
    // that merely replaced one would not either: its successor is the fact an
    // owner settles the old generation against.
    EXPECT_EQ(detached.count(ArenaRegionDisposition::StageAborted), 0u);
    EXPECT_EQ(bank.arena(0)->raw_backing(), committed[0]);
    EXPECT_EQ(bank.arena(2)->raw_backing(), committed[2]);
}

}  // namespace
