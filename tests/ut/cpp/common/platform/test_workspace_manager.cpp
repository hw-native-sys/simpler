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

#include <algorithm>
#include <map>
#include <vector>

#include "host/workspace_manager.h"

namespace {

/**
 * A backend whose allocations are distinct addresses, so a test can tell one
 * generation from another, and whose failures are injectable — the production
 * one fails on device pressure, which a unit test cannot produce.
 */
class FakeBackend {
public:
    WorkspaceManager::Backend ops() {
        WorkspaceManager::Backend b;
        b.ctx = this;
        b.acquire = [](void *ctx, size_t bytes) -> void * {
            return static_cast<FakeBackend *>(ctx)->acquire(bytes);
        };
        b.release = [](void *ctx, void *base, int *platform_rc) {
            return static_cast<FakeBackend *>(ctx)->release(base, platform_rc);
        };
        return b;
    }

    void *acquire(size_t bytes) {
        ++acquire_calls;
        if (fail_next_acquire) {
            fail_next_acquire = false;
            return nullptr;
        }
        if (next_forced != nullptr) {
            // Hands back an address the platform has already reclaimed, which
            // is what a real allocator is free to do.
            void *forced = next_forced;
            next_forced = nullptr;
            live_bytes[forced] = bytes;
            return forced;
        }
        next_ += 0x10000;
        live_bytes[reinterpret_cast<void *>(next_)] = bytes;
        return reinterpret_cast<void *>(next_);
    }

    WorkspaceManager::ReleaseOutcome release(void *base, int *platform_rc) {
        released.push_back(base);
        if (fail_release_of == base) {
            if (platform_rc != nullptr) *platform_rc = -7;
            return WorkspaceManager::ReleaseOutcome::FreeFailed;
        }
        if (keep_mapped_of == base) {
            if (platform_rc != nullptr) *platform_rc = -9;
            return WorkspaceManager::ReleaseOutcome::StillMapped;
        }
        live_bytes.erase(base);
        return WorkspaceManager::ReleaseOutcome::Freed;
    }

    bool fail_next_acquire{false};
    void *fail_release_of{nullptr};
    // The unmap half of the pair: the range stays mapped, so the bytes are
    // never freed and the block can never be offered again.
    void *keep_mapped_of{nullptr};
    // The next acquire returns exactly this address, once.
    void *next_forced{nullptr};
    int acquire_calls{0};
    std::vector<void *> released;
    std::map<void *, size_t> live_bytes;

private:
    uintptr_t next_{0x100000};
};

// The facts an ordinary launched run reports, in the order its phases produce
// them. Split so a case can stop partway and assert what that leaves.
void report_launched_and_drained(WorkspaceManager &m, uint32_t slot, uint64_t epoch) {
    m.note_run_fact(slot, epoch, WorkspaceManager::RunFact::Launched);
    m.note_run_fact(slot, epoch, WorkspaceManager::RunFact::DrainAttempted);
    m.note_run_fact(slot, epoch, WorkspaceManager::RunFact::DrainProvedComplete);
}

void report_host_side_done(WorkspaceManager &m, uint32_t slot, uint64_t epoch) {
    m.note_run_fact(slot, epoch, WorkspaceManager::RunFact::CopybackReturned);
    m.note_run_fact(slot, epoch, WorkspaceManager::RunFact::BindingsReleased);
}

constexpr uint64_t kBudget = 1u << 20;

TEST(WorkspaceManager, DisabledUntilAFiniteBudgetIsLatched) {
    FakeBackend backend;
    WorkspaceManager m;
    EXPECT_FALSE(m.enabled());
    // Nothing is served and nothing is charged while off, which is what keeps
    // an unconfigured context on the path it always had.
    EXPECT_EQ(m.acquire(WorkspaceManager::staging_region(0), 1, 4096), nullptr);
    EXPECT_EQ(backend.acquire_calls, 0);
    SimplerWorkspaceReport report{};
    EXPECT_FALSE(m.report(&report));

    EXPECT_FALSE(m.limit_enforced());
    EXPECT_FALSE(m.set_limit(kBudget));  // a limit needs management first
    EXPECT_TRUE(m.configure(backend.ops()));
    EXPECT_TRUE(m.enabled());
    EXPECT_FALSE(m.configure(backend.ops()));  // latched once
    EXPECT_FALSE(m.set_limit(0));              // a zero budget is not a budget
    EXPECT_TRUE(m.set_limit(kBudget));
    EXPECT_TRUE(m.limit_enforced());
    EXPECT_FALSE(m.set_limit(kBudget));  // latched once
}

TEST(WorkspaceManager, ReuseNeedsBothCapacityAndNoRemainingConsumer) {
    FakeBackend backend;
    WorkspaceManager m;
    ASSERT_TRUE(m.configure(backend.ops()));
    ASSERT_TRUE(m.set_limit(kBudget));

    void *first = m.acquire(WorkspaceManager::staging_region(0), 1, 4096);
    ASSERT_NE(first, nullptr);
    EXPECT_EQ(backend.acquire_calls, 1);

    // Same slot, a request the block fits — but run 1 has reported nothing, so
    // its contents may still be read and the block is not reusable.
    void *while_held = m.acquire(WorkspaceManager::staging_region(0), 2, 2048);
    EXPECT_NE(while_held, first);
    EXPECT_EQ(backend.acquire_calls, 2);

    // Retire both runs, and the first block becomes reusable at its capacity.
    report_launched_and_drained(m, 0, 1);
    report_host_side_done(m, 0, 1);
    report_launched_and_drained(m, 0, 2);
    report_host_side_done(m, 0, 2);
    void *reused = m.acquire(WorkspaceManager::staging_region(0), 3, 2048);
    EXPECT_TRUE(reused == first || reused == while_held);
    EXPECT_EQ(backend.acquire_calls, 2);  // nothing new was allocated
}

TEST(WorkspaceManager, ADrainThatDidNotProveCompletionDoesNotRetire) {
    FakeBackend backend;
    WorkspaceManager m;
    ASSERT_TRUE(m.configure(backend.ops()));
    ASSERT_TRUE(m.set_limit(kBudget));
    void *block = m.acquire(WorkspaceManager::arena_region(0, WorkspaceManager::ArenaRegion::GmHeap), 9, 4096);
    ASSERT_NE(block, nullptr);

    // Everything except the drain's own success: an attempted drain, a returned
    // copy-back and released bindings. The device side is not settled, so the
    // block keeps its consumer.
    m.note_run_fact(0, 9, WorkspaceManager::RunFact::Launched);
    m.note_run_fact(0, 9, WorkspaceManager::RunFact::DrainAttempted);
    report_host_side_done(m, 0, 9);
    EXPECT_EQ(m.block_state(block), WorkspaceManager::BlockState::Referenced);

    m.note_run_fact(0, 9, WorkspaceManager::RunFact::DrainProvedComplete);
    EXPECT_EQ(m.block_state(block), WorkspaceManager::BlockState::ProvenUnused);
}

TEST(WorkspaceManager, ARunThatSubmittedNothingRetiresWithoutADrain) {
    FakeBackend backend;
    WorkspaceManager m;
    ASSERT_TRUE(m.configure(backend.ops()));
    ASSERT_TRUE(m.set_limit(kBudget));
    void *block = m.acquire(WorkspaceManager::staging_region(1), 4, 4096);
    ASSERT_NE(block, nullptr);

    // A prepare that failed, or a run cancelled before launch: no device
    // consumer ever existed, so demanding a drain would demand one that cannot
    // happen. The host-side facts still have to arrive.
    m.note_run_fact(1, 4, WorkspaceManager::RunFact::NoDeviceSubmission);
    EXPECT_EQ(m.block_state(block), WorkspaceManager::BlockState::Referenced);
    EXPECT_EQ(m.live_drainable_consumers(), 1u);  // its host facts are still outstanding
    report_host_side_done(m, 1, 4);
    EXPECT_EQ(m.block_state(block), WorkspaceManager::BlockState::ProvenUnused);
    EXPECT_EQ(m.live_drainable_consumers(), 0u);
}

TEST(WorkspaceManager, ThreeRetainedGenerationsAllStayAccountedFor) {
    FakeBackend backend;
    WorkspaceManager m;
    ASSERT_TRUE(m.configure(backend.ops()));
    ASSERT_TRUE(m.set_limit(kBudget));

    // One region growing across three runs whose earlier consumers never
    // retire: every generation stays charged and none is dropped from the
    // ledger, which is what a fixed two-entry list would have lost.
    void *g1 = m.acquire(WorkspaceManager::staging_region(0), 1, 1024);
    void *g2 = m.acquire(WorkspaceManager::staging_region(0), 2, 2048);
    void *g3 = m.acquire(WorkspaceManager::staging_region(0), 3, 4096);
    ASSERT_NE(g1, nullptr);
    ASSERT_NE(g2, nullptr);
    ASSERT_NE(g3, nullptr);
    EXPECT_NE(g1, g2);
    EXPECT_NE(g2, g3);
    EXPECT_EQ(m.reserved_bytes(), 1024u + 2048u + 4096u);
    EXPECT_EQ(m.block_count(), 3u);

    SimplerWorkspaceReport report{};
    ASSERT_TRUE(m.report(&report));
    EXPECT_EQ(report.blocks_published, 3u);
    EXPECT_EQ(report.coverage_is_partial, 1u);

    // Only the runs that reported everything release their hold.
    report_launched_and_drained(m, 0, 1);
    report_host_side_done(m, 0, 1);
    EXPECT_EQ(m.block_state(g1), WorkspaceManager::BlockState::ProvenUnused);
    EXPECT_EQ(m.block_state(g2), WorkspaceManager::BlockState::Referenced);
    EXPECT_EQ(m.block_state(g3), WorkspaceManager::BlockState::Referenced);
}

TEST(WorkspaceManager, AnOverBudgetRequestFailsAndChangesNothing) {
    FakeBackend backend;
    WorkspaceManager m;
    ASSERT_TRUE(m.configure(backend.ops()));
    ASSERT_TRUE(m.set_limit(8192));
    void *held = m.acquire(WorkspaceManager::arena_region(0, WorkspaceManager::ArenaRegion::GmHeap), 1, 4096);
    ASSERT_NE(held, nullptr);
    const uint64_t charged = m.reserved_bytes();

    // A second generation would exceed the budget, and the first is still held,
    // so it cannot be reused to make room. The refusal leaves the published
    // address, its charge and the block count exactly as they were.
    EXPECT_EQ(m.acquire(WorkspaceManager::arena_region(0, WorkspaceManager::ArenaRegion::GmHeap), 2, 8192), nullptr);
    EXPECT_EQ(m.reserved_bytes(), charged);
    EXPECT_EQ(m.block_count(), 1u);
    EXPECT_EQ(m.block_state(held), WorkspaceManager::BlockState::Referenced);
    EXPECT_EQ(backend.acquire_calls, 1);  // the device was never asked
}

TEST(WorkspaceManager, ADeviceAllocationFailureLeavesPublishedStateIntact) {
    FakeBackend backend;
    WorkspaceManager m;
    ASSERT_TRUE(m.configure(backend.ops()));
    ASSERT_TRUE(m.set_limit(kBudget));
    void *held = m.acquire(WorkspaceManager::staging_region(0), 1, 4096);
    ASSERT_NE(held, nullptr);
    const uint64_t charged = m.reserved_bytes();

    backend.fail_next_acquire = true;
    EXPECT_EQ(m.acquire(WorkspaceManager::staging_region(0), 2, 8192), nullptr);
    EXPECT_EQ(m.reserved_bytes(), charged);
    EXPECT_EQ(m.block_count(), 1u);
    EXPECT_EQ(m.block_bytes(held), 4096u);
}

TEST(WorkspaceManager, AFailedReleaseKeepsItsOwnerAndItsCharge) {
    FakeBackend backend;
    WorkspaceManager m;
    ASSERT_TRUE(m.configure(backend.ops()));
    ASSERT_TRUE(m.set_limit(kBudget));
    void *block = m.acquire(WorkspaceManager::arena_region(0, WorkspaceManager::ArenaRegion::GmHeap), 5, 4096);
    ASSERT_NE(block, nullptr);
    report_launched_and_drained(m, 0, 5);
    report_host_side_done(m, 0, 5);

    backend.fail_release_of = block;
    EXPECT_NE(m.release_unreferenced(), 0);
    // Not reported as released, not dropped: the block keeps its owner and its
    // bytes keep their charge.
    EXPECT_EQ(m.block_state(block), WorkspaceManager::BlockState::ReleaseUnconfirmed);
    EXPECT_TRUE(m.owns(block));
    EXPECT_EQ(m.reserved_bytes(), 4096u);
    SimplerWorkspaceReport report{};
    ASSERT_TRUE(m.report(&report));
    EXPECT_EQ(report.release_unconfirmed_blocks, 1u);
    EXPECT_EQ(report.proof_unavailable, 1u);
}

TEST(WorkspaceManager, ASuccessfulReleaseDropsItsChargeAndIsNotReused) {
    FakeBackend backend;
    WorkspaceManager m;
    ASSERT_TRUE(m.configure(backend.ops()));
    ASSERT_TRUE(m.set_limit(kBudget));
    void *block = m.acquire(WorkspaceManager::arena_region(0, WorkspaceManager::ArenaRegion::GmHeap), 5, 4096);
    ASSERT_NE(block, nullptr);
    report_launched_and_drained(m, 0, 5);
    report_host_side_done(m, 0, 5);

    EXPECT_EQ(m.release_unreferenced(), 0);
    EXPECT_EQ(m.reserved_bytes(), 0u);
    EXPECT_FALSE(m.owns(block));
    EXPECT_FALSE(m.must_keep(block));
}

TEST(WorkspaceManager, ADestroyedContextQuarantinesTheWholeBlockItHeld) {
    FakeBackend backend;
    WorkspaceManager m;
    ASSERT_TRUE(m.configure(backend.ops()));
    ASSERT_TRUE(m.set_limit(kBudget));
    void *block = m.acquire(WorkspaceManager::arena_region(0, WorkspaceManager::ArenaRegion::GmHeap), 11, 4096);
    ASSERT_NE(block, nullptr);

    // The run owned device work, its drain never proved anything, and its
    // context is gone: no further fact can arrive, so the block is excluded
    // from every release path instead of being refused on forever.
    m.note_run_fact(0, 11, WorkspaceManager::RunFact::Launched);
    m.note_run_fact(0, 11, WorkspaceManager::RunFact::DrainAttempted);
    m.note_run_fact(0, 11, WorkspaceManager::RunFact::ContextDestroyed);

    EXPECT_EQ(m.block_state(block), WorkspaceManager::BlockState::Quarantined);
    EXPECT_TRUE(m.must_keep(block));
    EXPECT_EQ(m.live_drainable_consumers(), 0u);  // a refusal here would never clear
    EXPECT_EQ(m.release_unreferenced(), 0);
    EXPECT_TRUE(backend.released.empty());  // never individually released

    // Nor reused: the contents behind it may still be written.
    void *again = m.acquire(WorkspaceManager::arena_region(0, WorkspaceManager::ArenaRegion::GmHeap), 12, 1024);
    EXPECT_NE(again, block);

    SimplerWorkspaceReport report{};
    ASSERT_TRUE(m.report(&report));
    EXPECT_EQ(report.quarantined_blocks, 1u);
    EXPECT_EQ(report.proof_unavailable, 1u);
}

TEST(WorkspaceManager, ALiveDrainableConsumerIsCountedAndClearsOnRetirement) {
    FakeBackend backend;
    WorkspaceManager m;
    ASSERT_TRUE(m.configure(backend.ops()));
    ASSERT_TRUE(m.set_limit(kBudget));
    void *block = m.acquire(WorkspaceManager::arena_region(0, WorkspaceManager::ArenaRegion::GmHeap), 21, 4096);
    ASSERT_NE(block, nullptr);

    // Holds a reference with facts still outstanding: exactly the state a
    // caller can resolve by finalizing the run, which is why a teardown
    // refusal on it is a refusal the caller can act on.
    m.note_run_fact(0, 21, WorkspaceManager::RunFact::Launched);
    EXPECT_EQ(m.live_drainable_consumers(), 1u);
    SimplerWorkspaceReport report{};
    ASSERT_TRUE(m.report(&report));
    EXPECT_EQ(report.live_blocked, 1u);

    // A drain that has been entered, and even one that proved the device side
    // finished, still leaves the host consumers to report: the run keeps
    // counting until every fact its retirement needs has arrived.
    m.note_run_fact(0, 21, WorkspaceManager::RunFact::DrainAttempted);
    EXPECT_EQ(m.live_drainable_consumers(), 1u);
    m.note_run_fact(0, 21, WorkspaceManager::RunFact::DrainProvedComplete);
    EXPECT_EQ(m.live_drainable_consumers(), 1u);
    report_host_side_done(m, 0, 21);
    EXPECT_EQ(m.live_drainable_consumers(), 0u);
    EXPECT_EQ(m.block_state(block), WorkspaceManager::BlockState::ProvenUnused);
}

TEST(WorkspaceManager, TheSweepKeepsQuarantinedBlocksAndRecordsEveryOutcome) {
    FakeBackend backend;
    WorkspaceManager m;
    ASSERT_TRUE(m.configure(backend.ops()));
    ASSERT_TRUE(m.set_limit(kBudget));
    void *kept = m.acquire(WorkspaceManager::arena_region(0, WorkspaceManager::ArenaRegion::GmHeap), 31, 4096);
    void *freed = m.acquire(WorkspaceManager::staging_region(1), 32, 2048);
    ASSERT_NE(kept, nullptr);
    ASSERT_NE(freed, nullptr);
    m.note_run_fact(0, 31, WorkspaceManager::RunFact::Launched);
    m.note_run_fact(0, 31, WorkspaceManager::RunFact::DrainAttempted);
    m.note_run_fact(0, 31, WorkspaceManager::RunFact::ContextDestroyed);

    EXPECT_TRUE(m.must_keep(kept));
    EXPECT_FALSE(m.must_keep(freed));

    // What the allocator's terminal sweep reports back, including an address
    // outside this ledger — which goes into the fixed aggregate rather than a
    // buffer built during cleanup.
    m.note_sweep_result(kept, 0, true);
    m.note_sweep_result(freed, 0, false);
    m.note_sweep_result(reinterpret_cast<void *>(0xdeadbeef), -3, false);

    SimplerWorkspaceReport report{};
    ASSERT_TRUE(m.report(&report));
    EXPECT_EQ(report.quarantined_blocks, 1u);
    EXPECT_EQ(report.foreign_release_failures, 1u);
    EXPECT_EQ(report.last_foreign_release_rc, -3);
}

TEST(WorkspaceManager, ARetainedMappingIsAccountedSeparately) {
    FakeBackend backend;
    WorkspaceManager m;
    ASSERT_TRUE(m.configure(backend.ops()));
    ASSERT_TRUE(m.set_limit(kBudget));
    void *block = m.acquire(WorkspaceManager::arena_region(0, WorkspaceManager::ArenaRegion::GmHeap), 41, 4096);
    ASSERT_NE(block, nullptr);
    m.note_run_fact(0, 41, WorkspaceManager::RunFact::Launched);
    m.note_run_fact(0, 41, WorkspaceManager::RunFact::DrainAttempted);
    m.note_run_fact(0, 41, WorkspaceManager::RunFact::ContextDestroyed);

    m.note_mapping_retained(block, m.block_bytes(block));
    SimplerWorkspaceReport report{};
    ASSERT_TRUE(m.report(&report));
    EXPECT_EQ(report.quarantined_mapped_bytes, 4096u);
}

TEST(WorkspaceManager, ClosedAdmissionServesNoNewRequest) {
    FakeBackend backend;
    WorkspaceManager m;
    ASSERT_TRUE(m.configure(backend.ops()));
    ASSERT_TRUE(m.set_limit(kBudget));
    // A drain still admits the work already accepted, which is what lets an
    // in-progress prepare finish; a closed manager admits nothing.
    m.enter_drain_only();
    EXPECT_NE(m.acquire(WorkspaceManager::arena_region(0, WorkspaceManager::ArenaRegion::GmHeap), 51, 1024), nullptr);
    m.enter_closed();
    EXPECT_EQ(m.acquire(WorkspaceManager::arena_region(0, WorkspaceManager::ArenaRegion::GmHeap), 52, 1024), nullptr);
}

TEST(WorkspaceManager, TheReportNamesItsOwnCoverageAsPartial) {
    FakeBackend backend;
    WorkspaceManager m;
    ASSERT_TRUE(m.configure(backend.ops()));
    ASSERT_TRUE(m.set_limit(kBudget));
    SimplerWorkspaceReport report{};
    ASSERT_TRUE(m.report(&report));
    // The one field that stops a reader treating limit_bytes as a device-wide
    // ceiling: external tensors, results, diagnostics, code and provider
    // memory are all outside this budget.
    EXPECT_EQ(report.coverage_is_partial, 1u);
    EXPECT_EQ(report.budget_enforced, 1u);
    EXPECT_EQ(report.limit_bytes, kBudget);
    EXPECT_EQ(report.schema, static_cast<uint32_t>(WORKSPACE_REPORT_SCHEMA));
}

// --- Regressions for the corrections the exact-head review asked for ---

TEST(WorkspaceManagerOwnership, AReleasedBlockIsNeverHandedOutAgain) {
    FakeBackend backend;
    WorkspaceManager m;
    ASSERT_TRUE(m.configure(backend.ops()));
    ASSERT_TRUE(m.set_limit(kBudget));
    const WorkspaceManager::RegionKey region = WorkspaceManager::staging_region(0);

    void *first = m.acquire(region, 1, 4096);
    ASSERT_NE(first, nullptr);
    report_launched_and_drained(m, 0, 1);
    report_host_side_done(m, 0, 1);
    ASSERT_EQ(m.release_unreferenced(), 0);
    ASSERT_EQ(backend.released.size(), 1u);

    // The address belongs to the platform again, so the record is history: a
    // matching request must allocate rather than hand back freed memory.
    void *second = m.acquire(region, 2, 4096);
    ASSERT_NE(second, nullptr);
    EXPECT_NE(second, first);
    EXPECT_EQ(backend.acquire_calls, 2);
    EXPECT_EQ(m.reserved_bytes(), 4096u);
}

TEST(WorkspaceManagerOwnership, OneRegionsBlockIsNeverGivenToAnother) {
    FakeBackend backend;
    WorkspaceManager m;
    ASSERT_TRUE(m.configure(backend.ops()));
    ASSERT_TRUE(m.set_limit(kBudget));
    const WorkspaceManager::RegionKey heap = WorkspaceManager::arena_region(0, WorkspaceManager::ArenaRegion::GmHeap);
    const WorkspaceManager::RegionKey sm = WorkspaceManager::arena_region(0, WorkspaceManager::ArenaRegion::GmSm);

    void *heap_block = m.acquire(heap, 1, 4096);
    void *sm_block = m.acquire(sm, 1, 16384);
    ASSERT_NE(heap_block, nullptr);
    ASSERT_NE(sm_block, nullptr);
    report_launched_and_drained(m, 0, 1);
    report_host_side_done(m, 0, 1);

    // Run 2 needs a bigger heap and the same shared memory. The SM block now
    // has no reference and is large enough, so a pool keyed by domain alone
    // would hand it to the heap — while the SM region is still published at it,
    // leaving two live regions overlapping.
    void *grown_heap = m.acquire(heap, 2, 8192);
    ASSERT_NE(grown_heap, nullptr);
    EXPECT_NE(grown_heap, sm_block);
    EXPECT_NE(grown_heap, heap_block);
    // The unchanged SM region keeps the very block it is published at, and run
    // 2 registers as its consumer rather than being handed something else.
    EXPECT_TRUE(m.reference(sm_block, 2));
    EXPECT_EQ(m.block_bytes(sm_block), 16384u);
    EXPECT_EQ(m.block_state(sm_block), WorkspaceManager::BlockState::Referenced);
    // Three distinct blocks, so the two live regions cannot overlap.
    EXPECT_EQ(m.block_count(), 3u);
}

TEST(WorkspaceManagerOwnership, AReusedBlockRegistersItsNewConsumer) {
    FakeBackend backend;
    WorkspaceManager m;
    ASSERT_TRUE(m.configure(backend.ops()));
    ASSERT_TRUE(m.set_limit(kBudget));
    const WorkspaceManager::RegionKey region = WorkspaceManager::staging_region(0);

    void *block = m.acquire(region, 1, 4096);
    ASSERT_NE(block, nullptr);
    report_launched_and_drained(m, 0, 1);
    report_host_side_done(m, 0, 1);
    ASSERT_EQ(m.block_state(block), WorkspaceManager::BlockState::ProvenUnused);

    // The second run's request fits, so it allocates nothing — but it is about
    // to read and write those bytes, so registering is what stops a later
    // growth from treating them as free.
    EXPECT_TRUE(m.reference(block, 2));
    EXPECT_EQ(m.block_state(block), WorkspaceManager::BlockState::Referenced);
    EXPECT_EQ(m.live_drainable_consumers(), 1u);

    report_launched_and_drained(m, 0, 2);
    report_host_side_done(m, 0, 2);
    EXPECT_EQ(m.block_state(block), WorkspaceManager::BlockState::ProvenUnused);
}

TEST(WorkspaceManagerOwnership, AQuarantinedBlockRefusesNewConsumers) {
    FakeBackend backend;
    WorkspaceManager m;
    ASSERT_TRUE(m.configure(backend.ops()));
    ASSERT_TRUE(m.set_limit(kBudget));
    void *block = m.acquire(WorkspaceManager::staging_region(0), 1, 4096);
    ASSERT_NE(block, nullptr);
    m.note_run_fact(0, 1, WorkspaceManager::RunFact::Launched);
    m.note_run_fact(0, 1, WorkspaceManager::RunFact::DrainAttempted);
    m.note_run_fact(0, 1, WorkspaceManager::RunFact::ContextDestroyed);
    ASSERT_TRUE(m.must_keep(block));

    // Registering a consumer of a block nobody can prove is finished would be
    // handing out the very bytes the quarantine exists to withhold.
    EXPECT_FALSE(m.reference(block, 2));
    EXPECT_EQ(m.block_state(block), WorkspaceManager::BlockState::Quarantined);
}

TEST(WorkspaceManagerAccounting, TheTerminalSweepMovesEveryChargeExactlyOnce) {
    FakeBackend backend;
    WorkspaceManager m;
    ASSERT_TRUE(m.configure(backend.ops()));
    ASSERT_TRUE(m.set_limit(kBudget));
    void *kept = m.acquire(WorkspaceManager::arena_region(0, WorkspaceManager::ArenaRegion::GmHeap), 1, 4096);
    void *freed = m.acquire(WorkspaceManager::staging_region(1), 2, 2048);
    void *unconfirmed = m.acquire(WorkspaceManager::staging_region(0), 3, 1024);
    ASSERT_NE(kept, nullptr);
    ASSERT_NE(freed, nullptr);
    ASSERT_NE(unconfirmed, nullptr);
    ASSERT_EQ(m.reserved_bytes(), 4096u + 2048u + 1024u);

    m.note_run_fact(0, 1, WorkspaceManager::RunFact::Launched);
    m.note_run_fact(0, 1, WorkspaceManager::RunFact::DrainAttempted);
    m.note_run_fact(0, 1, WorkspaceManager::RunFact::ContextDestroyed);
    ASSERT_TRUE(m.must_keep(kept));

    {
        WorkspaceManager::TerminalSweep sweep = m.begin_terminal_sweep();
        sweep.note_result(kept, 0, true);
        sweep.note_result(freed, 0, false);
        sweep.note_result(unconfirmed, -5, false);
    }

    SimplerWorkspaceReport report{};
    ASSERT_TRUE(m.report(&report));
    // Kept bytes move to relinquished, a successful free leaves the budget, and
    // a failed free stays charged — it was not reclaimed.
    EXPECT_EQ(report.relinquished_bytes, 4096u);
    EXPECT_EQ(report.reserved_bytes, 1024u);
    EXPECT_EQ(report.quarantined_blocks, 1u);
    EXPECT_EQ(report.release_unconfirmed_blocks, 1u);
    EXPECT_EQ(report.proof_unavailable, 1u);

    // A second close finds the same records and must not account again.
    {
        WorkspaceManager::TerminalSweep sweep = m.begin_terminal_sweep();
        sweep.note_result(kept, 0, true);
        sweep.note_result(freed, 0, false);
    }
    SimplerWorkspaceReport again{};
    ASSERT_TRUE(m.report(&again));
    EXPECT_EQ(again.relinquished_bytes, 4096u);
    EXPECT_EQ(again.reserved_bytes, 1024u);
}

TEST(WorkspaceManagerAccounting, AnOrdinaryReleaseIsNotCountedTwiceByTheSweep) {
    FakeBackend backend;
    WorkspaceManager m;
    ASSERT_TRUE(m.configure(backend.ops()));
    ASSERT_TRUE(m.set_limit(kBudget));
    void *block = m.acquire(WorkspaceManager::staging_region(0), 1, 4096);
    ASSERT_NE(block, nullptr);
    report_launched_and_drained(m, 0, 1);
    report_host_side_done(m, 0, 1);
    ASSERT_EQ(m.release_unreferenced(), 0);
    ASSERT_EQ(m.reserved_bytes(), 0u);

    {
        WorkspaceManager::TerminalSweep sweep = m.begin_terminal_sweep();
        sweep.note_result(block, 0, false);
    }
    SimplerWorkspaceReport report{};
    ASSERT_TRUE(m.report(&report));
    EXPECT_EQ(report.reserved_bytes, 0u);
    EXPECT_EQ(report.relinquished_bytes, 0u);
}

TEST(WorkspaceManagerOwnership, ARunIsLiveFromItsFirstReferenceUntilItRetires) {
    FakeBackend backend;
    WorkspaceManager m;
    ASSERT_TRUE(m.configure(backend.ops()));
    ASSERT_TRUE(m.set_limit(kBudget));
    void *block = m.acquire(WorkspaceManager::staging_region(0), 7, 4096);
    ASSERT_NE(block, nullptr);

    // Prepared and holding workspace, with no launch reported yet: a close here
    // must still see it, because its bytes are already named by a plan.
    EXPECT_EQ(m.live_drainable_consumers(), 1u);
    m.note_run_fact(0, 7, WorkspaceManager::RunFact::Launched);
    EXPECT_EQ(m.live_drainable_consumers(), 1u);
    m.note_run_fact(0, 7, WorkspaceManager::RunFact::DrainAttempted);
    m.note_run_fact(0, 7, WorkspaceManager::RunFact::DrainProvedComplete);
    // Device side proved, host side not: still live.
    EXPECT_EQ(m.live_drainable_consumers(), 1u);
    m.note_run_fact(0, 7, WorkspaceManager::RunFact::CopybackReturned);
    EXPECT_EQ(m.live_drainable_consumers(), 1u);
    m.note_run_fact(0, 7, WorkspaceManager::RunFact::BindingsReleased);
    EXPECT_EQ(m.live_drainable_consumers(), 0u);
}

TEST(WorkspaceManagerOwnership, ALaunchedRunKeepsThatFactAgainstALateNoSubmission) {
    FakeBackend backend;
    WorkspaceManager m;
    ASSERT_TRUE(m.configure(backend.ops()));
    ASSERT_TRUE(m.set_limit(kBudget));
    void *block = m.acquire(WorkspaceManager::staging_region(0), 5, 4096);
    ASSERT_NE(block, nullptr);

    // The launch reported ownership; a partial unwind can then leave the
    // caller's pointer null, and finalize would report "never submitted". That
    // must not retire a run whose device work was real.
    m.note_run_fact(0, 5, WorkspaceManager::RunFact::Launched);
    m.note_run_fact(0, 5, WorkspaceManager::RunFact::NoDeviceSubmission);
    report_host_side_done(m, 0, 5);
    EXPECT_EQ(m.block_state(block), WorkspaceManager::BlockState::Referenced);
    EXPECT_EQ(m.live_drainable_consumers(), 1u);

    m.note_run_fact(0, 5, WorkspaceManager::RunFact::DrainAttempted);
    m.note_run_fact(0, 5, WorkspaceManager::RunFact::DrainProvedComplete);
    EXPECT_EQ(m.block_state(block), WorkspaceManager::BlockState::ProvenUnused);
}

// One region's generations under a finite budget, sized so the arithmetic of a
// reclaim is visible: 1 + 2 fits, 1 + 2 + 4 does not, and 2 + 4 does.
constexpr uint64_t kSixMiB = 6u << 20;
constexpr size_t kOneMiB = 1u << 20;
constexpr size_t kTwoMiB = 2u << 20;
constexpr size_t kFourMiB = 4u << 20;

void retire(WorkspaceManager &m, uint32_t slot, uint64_t epoch) {
    report_launched_and_drained(m, slot, epoch);
    report_host_side_done(m, slot, epoch);
}

TEST(WorkspaceManagerLifecycle, TheContextsOwnBackingIsNoConsumerACloseMustWaitFor) {
    FakeBackend backend;
    WorkspaceManager m;
    ASSERT_TRUE(m.configure(backend.ops()));
    ASSERT_TRUE(m.set_limit(kBudget));
    const WorkspaceManager::RegionKey region =
        WorkspaceManager::arena_region(0, WorkspaceManager::ArenaRegion::RuntimePool);

    // The eager device initialization allocates before any run exists, so its
    // requests carry the context identity rather than a run epoch.
    void *block = m.acquire(region, WorkspaceManager::kContextEpoch, 4096);
    ASSERT_NE(block, nullptr);
    m.note_published(region, block);

    // No run can ever report this identity finished, so counting it as a
    // consumer would refuse every close for the life of the context.
    EXPECT_EQ(m.live_drainable_consumers(), 0u);
    SimplerWorkspaceReport report{};
    ASSERT_TRUE(m.report(&report));
    EXPECT_EQ(report.live_blocked, 0u);
    EXPECT_EQ(report.blocks_published, 1u);

    // And it is the context's to release when the context tears down.
    EXPECT_EQ(m.release_unreferenced(), 0);
    EXPECT_EQ(backend.released, std::vector<void *>{block});
    EXPECT_EQ(m.reserved_bytes(), 0u);
}

TEST(WorkspaceManagerLifecycle, ARunOnTheContextsBackingLeavesNothingBehindWhenItRetires) {
    FakeBackend backend;
    WorkspaceManager m;
    ASSERT_TRUE(m.configure(backend.ops()));
    ASSERT_TRUE(m.set_limit(kBudget));
    const WorkspaceManager::RegionKey region = WorkspaceManager::arena_region(0, WorkspaceManager::ArenaRegion::GmHeap);

    void *block = m.acquire(region, WorkspaceManager::kContextEpoch, 4096);
    ASSERT_NE(block, nullptr);
    m.note_published(region, block);

    // A run whose request that block already covers registers against it and is
    // then the only consumer a close has to wait for.
    ASSERT_TRUE(m.reference(block, 7));
    EXPECT_EQ(m.live_drainable_consumers(), 1u);
    retire(m, 0, 7);
    EXPECT_EQ(m.live_drainable_consumers(), 0u);
    EXPECT_EQ(m.block_state(block), WorkspaceManager::BlockState::ProvenUnused);
    EXPECT_EQ(backend.acquire_calls, 1);

    // A second run over the same block leaves the same state behind.
    ASSERT_TRUE(m.reference(block, 8));
    EXPECT_EQ(m.live_drainable_consumers(), 1u);
    retire(m, 0, 8);
    EXPECT_EQ(m.live_drainable_consumers(), 0u);

    EXPECT_EQ(m.release_unreferenced(), 0);
    EXPECT_EQ(m.reserved_bytes(), 0u);
}

TEST(WorkspaceManagerBudget, GrowthReclaimsAnObsoleteGenerationRatherThanRefusing) {
    FakeBackend backend;
    WorkspaceManager m;
    ASSERT_TRUE(m.configure(backend.ops()));
    ASSERT_TRUE(m.set_limit(kSixMiB));
    const WorkspaceManager::RegionKey region = WorkspaceManager::staging_region(0);

    void *first = m.acquire(region, 1, kOneMiB);
    ASSERT_NE(first, nullptr);
    m.note_published(region, first);
    retire(m, 0, 1);

    // The region republishes bigger, which makes the 1 MiB generation obsolete:
    // nothing names it and nothing can come to name it again.
    void *second = m.acquire(region, 2, kTwoMiB);
    ASSERT_NE(second, first);
    m.note_published(region, second);
    retire(m, 0, 2);
    EXPECT_EQ(m.reserved_bytes(), kOneMiB + kTwoMiB);

    // 3 MiB charged and 4 MiB asked for exceeds the budget, and refusing here
    // would be wrong: the obsolete 1 MiB is reclaimable, and 2 + 4 fits.
    void *third = m.acquire(region, 3, kFourMiB);
    ASSERT_NE(third, nullptr);
    EXPECT_EQ(m.reserved_bytes(), kTwoMiB + kFourMiB);
    EXPECT_EQ(backend.released, std::vector<void *>{first});
    // The generation the region is published at was not touched to fund it.
    EXPECT_EQ(m.block_bytes(second), kTwoMiB);
    EXPECT_TRUE(m.owns(second));
}

TEST(WorkspaceManagerBudget, GrowthNeverReclaimsTheAddressAFailedPlanPreserved) {
    FakeBackend backend;
    WorkspaceManager m;
    ASSERT_TRUE(m.configure(backend.ops()));
    ASSERT_TRUE(m.set_limit(kSixMiB));
    const WorkspaceManager::RegionKey region = WorkspaceManager::staging_region(0);

    void *published = m.acquire(region, 1, kOneMiB);
    ASSERT_NE(published, nullptr);
    m.note_published(region, published);
    retire(m, 0, 1);

    // A plan that took a bigger block and then failed before publication. The
    // region still uses the old address, which is therefore still current.
    void *abandoned = m.acquire(region, 2, kTwoMiB);
    ASSERT_NE(abandoned, nullptr);
    retire(m, 0, 2);
    EXPECT_EQ(m.reserved_bytes(), kOneMiB + kTwoMiB);

    // Until the plan's abort is reported, this manager cannot tell the two
    // apart: both are current, so neither funds the request. That is the state
    // between the two events, not a resting state — the block is still charged
    // because nothing has yet said the staging ended.
    EXPECT_EQ(m.acquire(region, 3, kFourMiB), nullptr);
    EXPECT_TRUE(backend.released.empty());
    EXPECT_EQ(m.reserved_bytes(), kOneMiB + kTwoMiB);

    // The abort ends that claim, and only that one. The published address is
    // the one a failure has to preserve, so it is still current and still
    // unreclaimable; the abandoned block is now the obsolete generation the
    // request may reclaim.
    EXPECT_TRUE(m.note_unpublished(abandoned));
    void *grown = m.acquire(region, 3, kFourMiB);
    ASSERT_NE(grown, nullptr);
    EXPECT_EQ(backend.released, std::vector<void *>{abandoned});
    EXPECT_EQ(m.reserved_bytes(), kOneMiB + kFourMiB);
    EXPECT_TRUE(m.owns(published));
    EXPECT_EQ(m.block_bytes(published), kOneMiB);
}

TEST(WorkspaceManagerBudget, AnIdleCurrentBackingIsNotEvictableForAnotherRegion) {
    FakeBackend backend;
    WorkspaceManager m;
    ASSERT_TRUE(m.configure(backend.ops()));
    ASSERT_TRUE(m.set_limit(kSixMiB));
    const WorkspaceManager::RegionKey heap = WorkspaceManager::arena_region(0, WorkspaceManager::ArenaRegion::GmHeap);
    const WorkspaceManager::RegionKey sm = WorkspaceManager::arena_region(0, WorkspaceManager::ArenaRegion::GmSm);

    void *heap_block = m.acquire(heap, 1, kFourMiB);
    ASSERT_NE(heap_block, nullptr);
    m.note_published(heap, heap_block);
    retire(m, 0, 1);
    // Idle, not abandoned: no run holds it, and the heap is published at it.
    EXPECT_EQ(m.block_state(heap_block), WorkspaceManager::BlockState::ProvenUnused);

    EXPECT_EQ(m.acquire(sm, 2, kFourMiB), nullptr);
    EXPECT_TRUE(backend.released.empty());
    EXPECT_EQ(m.reserved_bytes(), kFourMiB);
    EXPECT_TRUE(m.owns(heap_block));
}

TEST(WorkspaceManagerOwnership, AnUnmappableBlockIsKeptRatherThanFreedUnderItsMapping) {
    FakeBackend backend;
    WorkspaceManager m;
    ASSERT_TRUE(m.configure(backend.ops()));
    ASSERT_TRUE(m.set_limit(kSixMiB));
    const WorkspaceManager::RegionKey region = WorkspaceManager::staging_region(0);

    void *mapped = m.acquire(region, 1, kOneMiB);
    ASSERT_NE(mapped, nullptr);
    m.note_published(region, mapped);
    retire(m, 0, 1);

    // The unmap failed, so a host address still covers this whole allocation.
    // Releasing the bytes would hand that range to the next allocation.
    EXPECT_TRUE(m.note_mapping_unregister_failed(mapped));
    EXPECT_EQ(m.block_state(mapped), WorkspaceManager::BlockState::Quarantined);
    EXPECT_TRUE(m.must_keep(mapped));
    EXPECT_FALSE(m.note_mapping_unregister_failed(reinterpret_cast<void *>(0xdead)));

    SimplerWorkspaceReport report{};
    ASSERT_TRUE(m.report(&report));
    EXPECT_EQ(report.quarantined_blocks, 1u);
    EXPECT_EQ(report.quarantined_mapped_bytes, kOneMiB);
    EXPECT_EQ(report.proof_unavailable, 1u);

    // Not by an ordinary release: a block a host address still covers is
    // excluded from every release path.
    EXPECT_EQ(m.release_unreferenced(), 0);
    EXPECT_TRUE(backend.released.empty());

    // The region cannot republish past it either, and for a stronger reason
    // than this case once asserted. It used to publish a successor and check
    // that growth's reclamation stepped over the mapped block; now ownership
    // of that block is unprovable, so no new block is published at all and
    // the mapped one can never even become an obsolete generation.
    WorkspaceManager::AcquireRefusal why = WorkspaceManager::AcquireRefusal::None;
    EXPECT_EQ(m.acquire(region, 2, kTwoMiB, &why), nullptr);
    EXPECT_EQ(why, WorkspaceManager::AcquireRefusal::Degraded);

    // Through all of it the bytes stay owned here and unfreed, which is the
    // property the mapping made necessary.
    EXPECT_TRUE(backend.released.empty());
    EXPECT_TRUE(m.owns(mapped));
    EXPECT_TRUE(m.must_keep(mapped));
    EXPECT_EQ(m.block_state(mapped), WorkspaceManager::BlockState::Quarantined);
    ASSERT_TRUE(m.report(&report));
    EXPECT_EQ(report.quarantined_mapped_bytes, kOneMiB);
    EXPECT_EQ(m.reserved_bytes(), kOneMiB);
}

TEST(WorkspaceManagerBudget, AGivenUpClaimStillWaitsForItsLastConsumer) {
    FakeBackend backend;
    WorkspaceManager m;
    ASSERT_TRUE(m.configure(backend.ops()));
    ASSERT_TRUE(m.set_limit(kSixMiB));
    const WorkspaceManager::RegionKey region = WorkspaceManager::staging_region(0);
    const WorkspaceManager::RegionKey peer = WorkspaceManager::staging_region(1);

    void *given_up = m.acquire(region, 4, kFourMiB);
    ASSERT_NE(given_up, nullptr);
    m.note_published(region, given_up);

    // The region gives up its claim while a run is still using the block — a
    // detach ordered mid-flight, or a staging aborted after its run had already
    // registered. The claim is the region's to end; whether the bytes may go is
    // still the consumer's to answer.
    EXPECT_TRUE(m.note_unpublished(given_up));
    EXPECT_EQ(m.live_drainable_consumers(), 1u);
    EXPECT_EQ(m.block_state(given_up), WorkspaceManager::BlockState::Referenced);
    EXPECT_EQ(m.acquire(peer, 5, kFourMiB), nullptr);
    EXPECT_TRUE(backend.released.empty());

    // Once that consumer retires the bytes are reclaimable, and a peer region's
    // growth takes them.
    retire(m, 0, 4);
    void *peer_block = m.acquire(peer, 5, kFourMiB);
    ASSERT_NE(peer_block, nullptr);
    EXPECT_EQ(backend.released, std::vector<void *>{given_up});
    EXPECT_EQ(m.reserved_bytes(), kFourMiB);
}

TEST(WorkspaceManagerOwnership, AGivenUpClaimLiftsNoQuarantine) {
    FakeBackend backend;
    WorkspaceManager m;
    ASSERT_TRUE(m.configure(backend.ops()));
    ASSERT_TRUE(m.set_limit(kSixMiB));
    const WorkspaceManager::RegionKey region = WorkspaceManager::arena_region(0, WorkspaceManager::ArenaRegion::GmSm);
    const WorkspaceManager::RegionKey peer = WorkspaceManager::arena_region(0, WorkspaceManager::ArenaRegion::GmHeap);

    void *unprovable = m.acquire(region, 6, kFourMiB);
    ASSERT_NE(unprovable, nullptr);
    m.note_published(region, unprovable);
    // Its context went away, so no further fact about its consumer can arrive.
    m.note_run_fact(0, 6, WorkspaceManager::RunFact::ContextDestroyed);
    ASSERT_EQ(m.block_state(unprovable), WorkspaceManager::BlockState::Quarantined);

    // The region may still stop publishing it. What that must not do is turn an
    // unprovable last consumer into a reclaimable block: nothing here knows
    // whether those bytes are still being written.
    EXPECT_TRUE(m.note_unpublished(unprovable));
    EXPECT_EQ(m.block_state(unprovable), WorkspaceManager::BlockState::Quarantined);
    EXPECT_TRUE(m.must_keep(unprovable));
    EXPECT_EQ(m.acquire(peer, 7, kFourMiB), nullptr);
    EXPECT_TRUE(backend.released.empty());
    EXPECT_EQ(m.reserved_bytes(), kFourMiB);

    // And an address this manager never owned is not a claim it can end.
    EXPECT_FALSE(m.note_unpublished(reinterpret_cast<void *>(0xfeed)));
}

// Management with no byte limit — the default this contract adds. A separate
// region per case so nothing here depends on another's leftovers.
void publish_and_retire(
    WorkspaceManager &m, const WorkspaceManager::RegionKey &region, uint64_t epoch, size_t bytes, void **out
) {
    void *block = m.acquire(region, epoch, bytes);
    ASSERT_NE(block, nullptr);
    m.note_published(region, block);
    report_launched_and_drained(m, 0, epoch);
    report_host_side_done(m, 0, epoch);
    if (out != nullptr) *out = block;
}

TEST(WorkspaceManagerDefault, ManagementIsOnWithoutAnyByteLimit) {
    FakeBackend backend;
    WorkspaceManager m;
    ASSERT_TRUE(m.configure(backend.ops()));
    EXPECT_TRUE(m.enabled());
    EXPECT_FALSE(m.limit_enforced());

    const WorkspaceManager::RegionKey region = WorkspaceManager::staging_region(0);
    void *block = m.acquire(region, 1, 4096);
    ASSERT_NE(block, nullptr);
    EXPECT_EQ(m.reserved_bytes(), 4096u);

    // Nothing caps it, so a request no budget would have allowed still lands.
    void *huge = m.acquire(WorkspaceManager::staging_region(1), 2, size_t{1} << 40);
    EXPECT_NE(huge, nullptr);

    SimplerWorkspaceReport report{};
    ASSERT_TRUE(m.report(&report));
    // The two questions are answered separately: this record exists because
    // the context is managed, and this field says no limit is enforced.
    EXPECT_EQ(report.budget_enforced, 0u);
    EXPECT_EQ(report.limit_bytes, 0u);
    EXPECT_EQ(report.coverage_is_partial, 1u);
    EXPECT_EQ(report.blocks_published, 2u);

    WorkspaceManager::AcquireRefusal why = WorkspaceManager::AcquireRefusal::None;
    EXPECT_EQ(m.acquire(region, 3, 0, &why), nullptr);
    EXPECT_EQ(why, WorkspaceManager::AcquireRefusal::NotServing);
}

TEST(WorkspaceManagerDefault, ObsoleteGenerationsAreReclaimedWithNoBudgetPressure) {
    FakeBackend backend;
    WorkspaceManager m;
    ASSERT_TRUE(m.configure(backend.ops()));
    const WorkspaceManager::RegionKey region = WorkspaceManager::staging_region(0);

    void *first = nullptr;
    ASSERT_NO_FATAL_FAILURE(publish_and_retire(m, region, 1, kOneMiB, &first));
    void *second = nullptr;
    ASSERT_NO_FATAL_FAILURE(publish_and_retire(m, region, 2, kTwoMiB, &second));
    ASSERT_NE(second, first);
    EXPECT_EQ(m.reserved_bytes(), kOneMiB + kTwoMiB);

    // No limit is set, so nothing is under pressure. The first generation is
    // reclaimed because its region republished and nothing references it.
    EXPECT_TRUE(m.has_reclaimable());
    EXPECT_EQ(m.reclaim_obsolete(), 0);
    EXPECT_EQ(backend.released, std::vector<void *>{first});
    EXPECT_EQ(m.reserved_bytes(), kTwoMiB);
    EXPECT_TRUE(m.owns(second));
    // Its record went with it, so the ledger tracks live ownership only.
    EXPECT_EQ(m.block_count(), 1u);
    EXPECT_FALSE(m.has_reclaimable());
}

TEST(WorkspaceManagerDefault, ACurrentBackingIsNeverReclaimed) {
    FakeBackend backend;
    WorkspaceManager m;
    ASSERT_TRUE(m.configure(backend.ops()));
    const WorkspaceManager::RegionKey region = WorkspaceManager::arena_region(0, WorkspaceManager::ArenaRegion::GmHeap);

    void *warm = nullptr;
    ASSERT_NO_FATAL_FAILURE(publish_and_retire(m, region, 1, kOneMiB, &warm));
    // Idle, not abandoned: the region is still published at it, so the next
    // same-sized call reuses it rather than paying another allocation.
    EXPECT_FALSE(m.has_reclaimable());
    EXPECT_EQ(m.reclaim_obsolete(), 0);
    EXPECT_TRUE(backend.released.empty());
    EXPECT_TRUE(m.owns(warm));
    EXPECT_EQ(m.acquire(region, 2, kOneMiB), warm);
}

TEST(WorkspaceManagerLedger, CompactionBoundsTheLedgerAndSurvivesAReissuedAddress) {
    FakeBackend backend;
    WorkspaceManager m;
    ASSERT_TRUE(m.configure(backend.ops()));
    const WorkspaceManager::RegionKey region = WorkspaceManager::staging_region(0);

    void *freed = nullptr;
    for (uint64_t gen = 1; gen <= 6; ++gen) {
        void *block = nullptr;
        ASSERT_NO_FATAL_FAILURE(publish_and_retire(m, region, gen, kOneMiB * static_cast<size_t>(gen), &block));
        if (gen == 1) freed = block;
        EXPECT_EQ(m.reclaim_obsolete(), 0);
    }
    // Six generations, one live block: history does not accumulate.
    EXPECT_EQ(m.block_count(), 1u);
    EXPECT_EQ(backend.released.size(), 5u);
    SimplerWorkspaceReport report{};
    ASSERT_TRUE(m.report(&report));
    // Compaction removes records, never the cumulative audit.
    EXPECT_EQ(report.blocks_published, 6u);

    // The platform hands back an address whose record was compacted away. It
    // must be matched to the new block only, not resurrected as the old one.
    ASSERT_NE(freed, nullptr);
    backend.next_forced = freed;
    void *reissued = m.acquire(WorkspaceManager::staging_region(1), 7, kOneMiB);
    ASSERT_EQ(reissued, freed);
    EXPECT_EQ(m.block_state(reissued), WorkspaceManager::BlockState::Referenced);
    EXPECT_EQ(m.block_count(), 2u);
}

TEST(WorkspaceManagerDegraded, AFailedFreeStopsNewPublicationButNotReuse) {
    FakeBackend backend;
    WorkspaceManager m;
    ASSERT_TRUE(m.configure(backend.ops()));
    const WorkspaceManager::RegionKey region = WorkspaceManager::staging_region(0);

    void *stale = nullptr;
    ASSERT_NO_FATAL_FAILURE(publish_and_retire(m, region, 1, kOneMiB, &stale));
    void *current = nullptr;
    ASSERT_NO_FATAL_FAILURE(publish_and_retire(m, region, 2, kTwoMiB, &current));

    backend.fail_release_of = stale;
    EXPECT_NE(m.reclaim_obsolete(), 0);
    EXPECT_EQ(m.block_state(stale), WorkspaceManager::BlockState::ReleaseUnconfirmed);
    EXPECT_TRUE(m.owns(stale));
    EXPECT_EQ(m.reserved_bytes(), kOneMiB + kTwoMiB);

    // Ownership of one block is now unprovable, so no new block is published —
    // which is what keeps the ledger from growing one failed record per round.
    WorkspaceManager::AcquireRefusal why = WorkspaceManager::AcquireRefusal::None;
    EXPECT_EQ(m.acquire(WorkspaceManager::staging_region(1), 3, kOneMiB, &why), nullptr);
    EXPECT_EQ(why, WorkspaceManager::AcquireRefusal::Degraded);

    // Blocks already proven safe stay usable, so accepted work continues.
    void *reused = m.acquire(region, 4, kOneMiB);
    EXPECT_EQ(reused, current);
    EXPECT_TRUE(m.reference(current, 5));

    SimplerWorkspaceReport report{};
    ASSERT_TRUE(m.report(&report));
    EXPECT_EQ(report.release_unconfirmed_blocks, 1u);
    EXPECT_EQ(report.quarantined_blocks, 0u);
    EXPECT_EQ(report.proof_unavailable, 1u);
}

TEST(WorkspaceManagerDegraded, AFailedUnmapAndAFailedFreeAreDifferentDispositions) {
    FakeBackend backend;
    WorkspaceManager m;
    ASSERT_TRUE(m.configure(backend.ops()));
    const WorkspaceManager::RegionKey mapped_region = WorkspaceManager::staging_region(0);
    const WorkspaceManager::RegionKey freed_region = WorkspaceManager::staging_region(1);

    void *mapped = nullptr;
    ASSERT_NO_FATAL_FAILURE(publish_and_retire(m, mapped_region, 1, kOneMiB, &mapped));
    ASSERT_NO_FATAL_FAILURE(publish_and_retire(m, mapped_region, 2, kTwoMiB, nullptr));
    void *unfreed = nullptr;
    ASSERT_NO_FATAL_FAILURE(publish_and_retire(m, freed_region, 3, kOneMiB, &unfreed));
    ASSERT_NO_FATAL_FAILURE(publish_and_retire(m, freed_region, 4, kTwoMiB, nullptr));

    backend.keep_mapped_of = mapped;
    backend.fail_release_of = unfreed;
    EXPECT_NE(m.reclaim_obsolete(), 0);

    // Still mapped: the bytes can never be handed to another allocation, so
    // the block is quarantined and excluded from the terminal sweep.
    EXPECT_EQ(m.block_state(mapped), WorkspaceManager::BlockState::Quarantined);
    EXPECT_TRUE(m.must_keep(mapped));
    // Unmapped but not freed: the charge stays and the block is never offered
    // again, but this is a reclamation cost rather than a live mapping.
    EXPECT_EQ(m.block_state(unfreed), WorkspaceManager::BlockState::ReleaseUnconfirmed);
    EXPECT_FALSE(m.must_keep(unfreed));

    SimplerWorkspaceReport report{};
    ASSERT_TRUE(m.report(&report));
    EXPECT_EQ(report.quarantined_blocks, 1u);
    EXPECT_EQ(report.quarantined_mapped_bytes, kOneMiB);
    EXPECT_EQ(report.release_unconfirmed_blocks, 1u);
}

TEST(WorkspaceManagerDegraded, AnUnprovableRunStopsNewPublication) {
    FakeBackend backend;
    WorkspaceManager m;
    ASSERT_TRUE(m.configure(backend.ops()));
    const WorkspaceManager::RegionKey region = WorkspaceManager::staging_region(0);

    void *held = m.acquire(region, 11, kOneMiB);
    ASSERT_NE(held, nullptr);
    // Its context went away with facts outstanding, so its last consumer can
    // never be proved. That record is permanent, and publishing more blocks
    // would only add more of them.
    m.note_run_fact(0, 11, WorkspaceManager::RunFact::ContextDestroyed);
    EXPECT_EQ(m.block_state(held), WorkspaceManager::BlockState::Quarantined);

    WorkspaceManager::AcquireRefusal why = WorkspaceManager::AcquireRefusal::None;
    EXPECT_EQ(m.acquire(WorkspaceManager::staging_region(1), 12, kOneMiB, &why), nullptr);
    EXPECT_EQ(why, WorkspaceManager::AcquireRefusal::Degraded);
    // Nothing was freed to get there.
    EXPECT_TRUE(backend.released.empty());
    EXPECT_EQ(m.reserved_bytes(), kOneMiB);
}

TEST(WorkspaceManagerOwnership, ALateFactFromAReleasedSlotDoesNotReplaceItsSuccessor) {
    FakeBackend backend;
    WorkspaceManager m;
    ASSERT_TRUE(m.configure(backend.ops()));
    const WorkspaceManager::RegionKey old_region = WorkspaceManager::staging_region(0);
    const WorkspaceManager::RegionKey new_region = WorkspaceManager::staging_region(1);

    void *predecessor_block = m.acquire(old_region, 7, kOneMiB);
    ASSERT_NE(predecessor_block, nullptr);
    // The slot is handed to a successor, which reports against it.
    void *successor_block = m.acquire(new_region, 9, kOneMiB);
    ASSERT_NE(successor_block, nullptr);
    m.note_run_fact(0, 9, WorkspaceManager::RunFact::Launched);

    // The predecessor's terminal fact arrives late, against a slot that is no
    // longer its own. Epochs only increase, so this is recognisable.
    m.note_run_fact(0, 7, WorkspaceManager::RunFact::ContextDestroyed);

    // Its own references are still disposed of — a late destruction must not
    // leave those blocks looking reclaimable.
    EXPECT_EQ(m.block_state(predecessor_block), WorkspaceManager::BlockState::Quarantined);

    // And the successor's record survived: a launch it reported cannot be
    // undone by a stale "never submitted", so it does not retire.
    m.note_run_fact(0, 9, WorkspaceManager::RunFact::NoDeviceSubmission);
    report_host_side_done(m, 0, 9);
    EXPECT_EQ(m.block_state(successor_block), WorkspaceManager::BlockState::Referenced);
    EXPECT_EQ(m.live_drainable_consumers(), 1u);
}

TEST(WorkspaceManagerOwnership, AReleaseUnconfirmedBlockIsNotHandedBackByReference) {
    FakeBackend backend;
    WorkspaceManager m;
    ASSERT_TRUE(m.configure(backend.ops()));
    const WorkspaceManager::RegionKey region = WorkspaceManager::staging_region(0);

    void *stale = nullptr;
    ASSERT_NO_FATAL_FAILURE(publish_and_retire(m, region, 1, kOneMiB, &stale));
    ASSERT_NO_FATAL_FAILURE(publish_and_retire(m, region, 2, kTwoMiB, nullptr));
    backend.fail_release_of = stale;
    EXPECT_NE(m.reclaim_obsolete(), 0);
    ASSERT_EQ(m.block_state(stale), WorkspaceManager::BlockState::ReleaseUnconfirmed);

    // A caller arriving from a capacity hit has not re-derived the block's
    // disposition. Handing back storage whose free was attempted and failed
    // would let a run write where the platform may already have reclaimed.
    EXPECT_FALSE(m.reference(stale, 8));
}

TEST(WorkspaceManagerDegraded, ASweepThatProvesTheFreeClearsTheEarlierDoubt) {
    FakeBackend backend;
    WorkspaceManager m;
    ASSERT_TRUE(m.configure(backend.ops()));
    const WorkspaceManager::RegionKey region = WorkspaceManager::staging_region(0);

    void *stale = nullptr;
    ASSERT_NO_FATAL_FAILURE(publish_and_retire(m, region, 1, kOneMiB, &stale));
    ASSERT_NO_FATAL_FAILURE(publish_and_retire(m, region, 2, kTwoMiB, nullptr));
    backend.fail_release_of = stale;
    EXPECT_NE(m.reclaim_obsolete(), 0);
    ASSERT_EQ(m.block_state(stale), WorkspaceManager::BlockState::ReleaseUnconfirmed);

    // The ordinary close path does not try this block again — that failure is
    // the run boundary's and is already recorded. It still releases the other,
    // unreferenced block, so what must not grow is this block's own attempt
    // count rather than the total.
    const auto attempts_on = [&backend](void *base) {
        return std::count(backend.released.begin(), backend.released.end(), base);
    };
    ASSERT_EQ(attempts_on(stale), 1);
    EXPECT_EQ(m.release_unreferenced(), 0);
    EXPECT_EQ(attempts_on(stale), 1);

    // The terminal sweep gets the one further attempt that can settle it, and
    // when it succeeds the doubt is over: a freed block reported as
    // unconfirmed would stay that way for the rest of the context's life and
    // never leave the ledger.
    {
        WorkspaceManager::TerminalSweep sweep = m.begin_terminal_sweep();
        EXPECT_FALSE(sweep.must_keep(stale));
        sweep.note_result(stale, 0, /*kept=*/false);
    }
    EXPECT_FALSE(m.owns(stale));
    SimplerWorkspaceReport report{};
    ASSERT_TRUE(m.report(&report));
    EXPECT_EQ(report.release_unconfirmed_blocks, 0u);
    EXPECT_EQ(report.proof_unavailable, 0u);
}

}  // namespace
