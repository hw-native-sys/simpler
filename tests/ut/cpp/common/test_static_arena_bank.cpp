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
// Capacity rule for an arena bank's three pooled regions, shared verbatim by
// the onboard and simulation runners' setup_static_arena.
//
// A kernel-mode context is the case under test: a captured graph replays the
// base addresses the regions held when it was captured, so the assertions are
// about addresses and about the underlying allocator, not about return codes
// alone. DeviceArena defaults to a libc backend and counts its own alloc/free
// calls, so an unwanted re-base is observable here with no device present.

#include <cstddef>
#include <new>

#include <gtest/gtest.h>

#include "host/static_arena_bank.h"
#include "worker/runtime_c_api.h"

namespace {

constexpr size_t kGmHeapBytes = 64 * 1024;
constexpr size_t kGmSmBytes = 16 * 1024;
constexpr size_t kRuntimePoolBytes = 8 * 1024;

struct FailingAllocator {
    int calls{0};
    int fail_on{0};
    bool throw_on_failure{false};

    static void *allocate(void *context, size_t bytes) {
        auto &self = *static_cast<FailingAllocator *>(context);
        if (++self.calls == self.fail_on) {
            if (self.throw_on_failure) throw std::bad_alloc();
            return nullptr;
        }
        return DeviceArena::default_alloc(nullptr, bytes);
    }
};

// One bank's regions plus the sizes it remembers for them: the state a runner
// keeps between calls.
struct Bank {
    DeviceArena gm_heap;
    DeviceArena gm_sm;
    DeviceArena runtime_pool;
    size_t cached_gm_heap_size{0};
    size_t cached_gm_sm_size{0};
    size_t cached_runtime_pool_size{0};

    Bank() = default;
    explicit Bank(FailingAllocator &allocator) :
        gm_heap(FailingAllocator::allocate, DeviceArena::default_free, &allocator),
        gm_sm(FailingAllocator::allocate, DeviceArena::default_free, &allocator),
        runtime_pool(FailingAllocator::allocate, DeviceArena::default_free, &allocator) {}

    StaticArenaBankRequest request(size_t gm_heap_size, size_t gm_sm_size, size_t runtime_pool_size) {
        return StaticArenaBankRequest{
            {&gm_heap, &cached_gm_heap_size, gm_heap_size},
            {&gm_sm, &cached_gm_sm_size, gm_sm_size},
            {&runtime_pool, &cached_runtime_pool_size, runtime_pool_size},
        };
    }

    StaticArenaBankOutcome commit(size_t gm_heap_size, size_t gm_sm_size, size_t runtime_pool_size, bool kernel_mode) {
        return commit_static_arena_bank(request(gm_heap_size, gm_sm_size, runtime_pool_size), kernel_mode);
    }

    size_t alloc_calls() const { return gm_heap.alloc_count() + gm_sm.alloc_count() + runtime_pool.alloc_count(); }
    size_t free_calls() const { return gm_heap.free_count() + gm_sm.free_count() + runtime_pool.free_count(); }
};

// The three base addresses a caller may have handed to a captured graph.
struct Bases {
    void *gm_heap;
    void *gm_sm;
    void *runtime_pool;

    bool operator==(const Bases &other) const {
        return gm_heap == other.gm_heap && gm_sm == other.gm_sm && runtime_pool == other.runtime_pool;
    }
};

Bases bases_of(const Bank &bank) { return Bases{bank.gm_heap.base(), bank.gm_sm.base(), bank.runtime_pool.base()}; }

}  // namespace

// The first commit is the context's own, and kernel mode must let it through:
// a rule that refused it would leave a kernel context with no arena at all.
TEST(StaticArenaBankKernelMode, FirstCommitIsPermitted) {
    Bank bank;
    const StaticArenaBankOutcome outcome = bank.commit(kGmHeapBytes, kGmSmBytes, kRuntimePoolBytes, true);

    EXPECT_EQ(outcome.rc, 0);
    EXPECT_TRUE(outcome.bases_changed);
    EXPECT_TRUE(bank.gm_heap.is_committed());
    EXPECT_TRUE(bank.gm_sm.is_committed());
    EXPECT_TRUE(bank.runtime_pool.is_committed());
    EXPECT_EQ(bank.cached_gm_heap_size, kGmHeapBytes);
    EXPECT_EQ(bank.cached_gm_sm_size, kGmSmBytes);
    EXPECT_EQ(bank.cached_runtime_pool_size, kRuntimePoolBytes);
}

// Acceptance 1 and 2: base and capacity per region survive repeated calls at
// the context's own sizes, and the allocator is not entered at all.
TEST(StaticArenaBankKernelMode, RepeatedCommitsMoveNoBaseAndCallNoAllocator) {
    Bank bank;
    ASSERT_EQ(bank.commit(kGmHeapBytes, kGmSmBytes, kRuntimePoolBytes, true).rc, 0);
    const Bases pinned = bases_of(bank);
    const size_t allocs_after_commit = bank.alloc_calls();
    const size_t frees_after_commit = bank.free_calls();

    for (int call = 0; call < 8; ++call) {
        const StaticArenaBankOutcome outcome = bank.commit(kGmHeapBytes, kGmSmBytes, kRuntimePoolBytes, true);
        EXPECT_EQ(outcome.rc, 0);
        EXPECT_FALSE(outcome.bases_changed);
        EXPECT_TRUE(bases_of(bank) == pinned);
    }

    EXPECT_EQ(bank.alloc_calls(), allocs_after_commit);
    EXPECT_EQ(bank.free_calls(), frees_after_commit);
    EXPECT_EQ(bank.cached_gm_heap_size, kGmHeapBytes);
    EXPECT_EQ(bank.cached_gm_sm_size, kGmSmBytes);
    EXPECT_EQ(bank.cached_runtime_pool_size, kRuntimePoolBytes);
}

// Acceptance 3: a request landing exactly on the committed capacity succeeds,
// and a smaller one is served from the same buffer.
TEST(StaticArenaBankKernelMode, ExactAndSmallerRequestsAreServedInPlace) {
    Bank bank;
    ASSERT_EQ(bank.commit(kGmHeapBytes, kGmSmBytes, kRuntimePoolBytes, true).rc, 0);
    const Bases pinned = bases_of(bank);
    const size_t allocs = bank.alloc_calls();

    EXPECT_EQ(bank.commit(kGmHeapBytes, kGmSmBytes, kRuntimePoolBytes, true).rc, 0);
    EXPECT_EQ(bank.commit(kGmHeapBytes - 1, kGmSmBytes - 1, kRuntimePoolBytes - 1, true).rc, 0);

    EXPECT_TRUE(bases_of(bank) == pinned);
    EXPECT_EQ(bank.alloc_calls(), allocs);
    // A request under capacity does not shrink what the bank remembers.
    EXPECT_EQ(bank.cached_gm_heap_size, kGmHeapBytes);
}

// Acceptance 4 and 5: one byte over capacity is refused with
// PTO_RUNTIME_ERR_INTERNAL, and the refusal leaves the bank able to serve the
// layout it already holds.
TEST(StaticArenaBankKernelMode, OneByteOverCapacityIsRefusedAndChangesNothing) {
    Bank bank;
    ASSERT_EQ(bank.commit(kGmHeapBytes, kGmSmBytes, kRuntimePoolBytes, true).rc, 0);
    const Bases pinned = bases_of(bank);
    const size_t allocs = bank.alloc_calls();
    const size_t frees = bank.free_calls();

    const StaticArenaBankOutcome refused = bank.commit(kGmHeapBytes + 1, kGmSmBytes, kRuntimePoolBytes, true);

    EXPECT_EQ(refused.rc, PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_FALSE(refused.bases_changed);
    EXPECT_TRUE(bases_of(bank) == pinned);
    EXPECT_TRUE(bank.gm_heap.is_committed());
    EXPECT_TRUE(bank.gm_sm.is_committed());
    EXPECT_TRUE(bank.runtime_pool.is_committed());
    EXPECT_EQ(bank.alloc_calls(), allocs);
    EXPECT_EQ(bank.free_calls(), frees);
    EXPECT_EQ(bank.cached_gm_heap_size, kGmHeapBytes);

    // The plan the bank held before the refusal is still the plan it serves.
    const StaticArenaBankOutcome after = bank.commit(kGmHeapBytes, kGmSmBytes, kRuntimePoolBytes, true);
    EXPECT_EQ(after.rc, 0);
    EXPECT_FALSE(after.bases_changed);
    EXPECT_TRUE(bases_of(bank) == pinned);
    EXPECT_EQ(bank.alloc_calls(), allocs);
    EXPECT_EQ(bank.free_calls(), frees);
}

// A refusal naming a later region must not roll the earlier ones back. This is
// the shape the whole rule protects: the rollback releases every region, so a
// refusal routed through it would drop the addresses it just declined to move.
TEST(StaticArenaBankKernelMode, RefusalOnALaterRegionKeepsItsPeersCommitted) {
    Bank bank;
    ASSERT_EQ(bank.commit(kGmHeapBytes, kGmSmBytes, kRuntimePoolBytes, true).rc, 0);
    const Bases pinned = bases_of(bank);
    const size_t frees = bank.free_calls();

    const StaticArenaBankOutcome refused = bank.commit(kGmHeapBytes, kGmSmBytes, kRuntimePoolBytes + 1, true);

    EXPECT_EQ(refused.rc, PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_TRUE(bases_of(bank) == pinned);
    EXPECT_TRUE(bank.gm_heap.is_committed());
    EXPECT_TRUE(bank.gm_sm.is_committed());
    EXPECT_TRUE(bank.runtime_pool.is_committed());
    EXPECT_EQ(bank.free_calls(), frees);
    EXPECT_EQ(bank.cached_gm_heap_size, kGmHeapBytes);
    EXPECT_EQ(bank.cached_gm_sm_size, kGmSmBytes);
    EXPECT_EQ(bank.cached_runtime_pool_size, kRuntimePoolBytes);
}

// Releasing is the other way a base goes away, and hbg's runtime-arena path
// asks for it by passing zero on every bind.
TEST(StaticArenaBankKernelMode, ReleasingACommittedRegionIsRefused) {
    Bank bank;
    ASSERT_EQ(bank.commit(kGmHeapBytes, kGmSmBytes, kRuntimePoolBytes, true).rc, 0);
    const Bases pinned = bases_of(bank);
    const size_t frees = bank.free_calls();

    const StaticArenaBankOutcome refused = bank.commit(kGmHeapBytes, kGmSmBytes, 0, true);

    EXPECT_EQ(refused.rc, PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_TRUE(bank.runtime_pool.is_committed());
    EXPECT_TRUE(bases_of(bank) == pinned);
    EXPECT_EQ(bank.free_calls(), frees);
    EXPECT_EQ(bank.cached_runtime_pool_size, kRuntimePoolBytes);
}

// A region the bank never committed carries no address anyone can hold, so a
// zero request for it is the ordinary case of hbg passing zero, not a release.
TEST(StaticArenaBankKernelMode, ZeroRequestForAnUncommittedRegionIsPermitted) {
    Bank bank;
    const StaticArenaBankOutcome outcome = bank.commit(kGmHeapBytes, kGmSmBytes, 0, true);

    EXPECT_EQ(outcome.rc, 0);
    EXPECT_TRUE(bank.gm_heap.is_committed());
    EXPECT_FALSE(bank.runtime_pool.is_committed());
    EXPECT_EQ(bank.cached_runtime_pool_size, 0u);

    // And it stays permitted on every later bind.
    EXPECT_EQ(bank.commit(kGmHeapBytes, kGmSmBytes, 0, true).rc, 0);
}

// Program mode keeps the growth and rollback behavior it has always had: the
// rule is selected by the context's identity, not applied to every caller.
TEST(StaticArenaBankProgramMode, GrowthRebasesTheRegion) {
    Bank bank;
    ASSERT_EQ(bank.commit(kGmHeapBytes, kGmSmBytes, kRuntimePoolBytes, false).rc, 0);
    const size_t frees_before = bank.free_calls();

    const StaticArenaBankOutcome grown = bank.commit(kGmHeapBytes * 2, kGmSmBytes, kRuntimePoolBytes, false);

    EXPECT_EQ(grown.rc, 0);
    EXPECT_TRUE(grown.bases_changed);
    EXPECT_EQ(bank.cached_gm_heap_size, kGmHeapBytes * 2);
    EXPECT_GT(bank.free_calls(), frees_before);
}

TEST(StaticArenaBankProgramMode, ZeroRequestReleasesACommittedRegion) {
    Bank bank;
    ASSERT_EQ(bank.commit(kGmHeapBytes, kGmSmBytes, kRuntimePoolBytes, false).rc, 0);

    const StaticArenaBankOutcome outcome = bank.commit(kGmHeapBytes, kGmSmBytes, 0, false);

    EXPECT_EQ(outcome.rc, 0);
    EXPECT_TRUE(outcome.bases_changed);
    EXPECT_FALSE(bank.runtime_pool.is_committed());
    EXPECT_EQ(bank.cached_runtime_pool_size, 0u);
}

TEST(StaticArenaBankKernelMode, RefusalPrecedesEveryNewAllocation) {
    Bank bank;
    ASSERT_EQ(bank.commit(0, kGmSmBytes, kRuntimePoolBytes, true).rc, 0);
    const auto pinned = bases_of(bank);
    const auto allocs = bank.alloc_calls();
    const auto frees = bank.free_calls();
    const auto refused = bank.commit(kGmHeapBytes, kGmSmBytes + 1, kRuntimePoolBytes, true);
    EXPECT_EQ(refused.rc, PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_FALSE(refused.bases_changed);
    EXPECT_TRUE(bases_of(bank) == pinned);
    EXPECT_EQ(bank.alloc_calls(), allocs);
    EXPECT_EQ(bank.free_calls(), frees);
    EXPECT_EQ(bank.cached_gm_heap_size, 0u);
}

TEST(StaticArenaBankKernelMode, AllocationFailureRollsBackOnlyNewRegions) {
    for (bool previously_committed : {false, true}) {
        for (bool throw_on_failure : {false, true}) {
            FailingAllocator allocator;
            Bank bank(allocator);
            const size_t old_heap = previously_committed ? kGmHeapBytes : 0;
            ASSERT_EQ(bank.commit(old_heap, 0, 0, true).rc, 0);
            const auto pinned = bases_of(bank);
            allocator.fail_on = allocator.calls + (previously_committed ? 2 : 3);
            allocator.throw_on_failure = throw_on_failure;
            const auto failed = bank.commit(kGmHeapBytes, kGmSmBytes, kRuntimePoolBytes, true);
            EXPECT_EQ(failed.rc, PTO_RUNTIME_ERR_INTERNAL);
            EXPECT_FALSE(failed.bases_changed);
            EXPECT_TRUE(bases_of(bank) == pinned);
            EXPECT_EQ(bank.cached_gm_heap_size, old_heap);
            EXPECT_EQ(bank.cached_gm_sm_size, 0u);
            EXPECT_EQ(bank.cached_runtime_pool_size, 0u);
            allocator.fail_on = 0;
            EXPECT_EQ(bank.commit(kGmHeapBytes, kGmSmBytes, kRuntimePoolBytes, true).rc, 0);
            if (previously_committed) EXPECT_EQ(bank.gm_heap.base(), pinned.gm_heap);
        }
    }
}

TEST(StaticArenaBankProgramMode, AllocationFailureStillReleasesCommittedPeers) {
    FailingAllocator allocator;
    Bank bank(allocator);
    ASSERT_EQ(bank.commit(kGmHeapBytes, kGmSmBytes, 0, false).rc, 0);
    allocator.fail_on = allocator.calls + 1;
    EXPECT_EQ(bank.commit(kGmHeapBytes, kGmSmBytes, kRuntimePoolBytes, false).rc, PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_EQ(bank.gm_heap.base(), nullptr);
    EXPECT_EQ(bank.gm_sm.base(), nullptr);
    EXPECT_EQ(bank.runtime_pool.base(), nullptr);
    EXPECT_EQ(bank.cached_gm_heap_size, 0u);
    EXPECT_EQ(bank.cached_gm_sm_size, 0u);
}
