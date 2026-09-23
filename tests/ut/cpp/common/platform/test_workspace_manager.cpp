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
        b.release = [](void *ctx, void *base) {
            return static_cast<FakeBackend *>(ctx)->release(base);
        };
        return b;
    }

    void *acquire(size_t bytes) {
        ++acquire_calls;
        if (fail_next_acquire) {
            fail_next_acquire = false;
            return nullptr;
        }
        next_ += 0x10000;
        live_bytes[reinterpret_cast<void *>(next_)] = bytes;
        return reinterpret_cast<void *>(next_);
    }

    int release(void *base) {
        released.push_back(base);
        if (fail_release_of == base) return -7;
        live_bytes.erase(base);
        return 0;
    }

    bool fail_next_acquire{false};
    void *fail_release_of{nullptr};
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

    EXPECT_FALSE(m.configure(0, backend.ops()));  // a zero budget is not a budget
    EXPECT_FALSE(m.enabled());
    EXPECT_TRUE(m.configure(kBudget, backend.ops()));
    EXPECT_TRUE(m.enabled());
    EXPECT_FALSE(m.configure(kBudget, backend.ops()));  // latched once
}

TEST(WorkspaceManager, ReuseNeedsBothCapacityAndNoRemainingConsumer) {
    FakeBackend backend;
    WorkspaceManager m;
    ASSERT_TRUE(m.configure(kBudget, backend.ops()));

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
    ASSERT_TRUE(m.configure(kBudget, backend.ops()));
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
    ASSERT_TRUE(m.configure(kBudget, backend.ops()));
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
    ASSERT_TRUE(m.configure(kBudget, backend.ops()));

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
    ASSERT_TRUE(m.configure(8192, backend.ops()));
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
    ASSERT_TRUE(m.configure(kBudget, backend.ops()));
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
    ASSERT_TRUE(m.configure(kBudget, backend.ops()));
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
    ASSERT_TRUE(m.configure(kBudget, backend.ops()));
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
    ASSERT_TRUE(m.configure(kBudget, backend.ops()));
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
    ASSERT_TRUE(m.configure(kBudget, backend.ops()));
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
    ASSERT_TRUE(m.configure(kBudget, backend.ops()));
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
    ASSERT_TRUE(m.configure(kBudget, backend.ops()));
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
    ASSERT_TRUE(m.configure(kBudget, backend.ops()));
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
    ASSERT_TRUE(m.configure(kBudget, backend.ops()));
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
    ASSERT_TRUE(m.configure(kBudget, backend.ops()));
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
    ASSERT_TRUE(m.configure(kBudget, backend.ops()));
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
    ASSERT_TRUE(m.configure(kBudget, backend.ops()));
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
    ASSERT_TRUE(m.configure(kBudget, backend.ops()));
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
    ASSERT_TRUE(m.configure(kBudget, backend.ops()));
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
    ASSERT_TRUE(m.configure(kBudget, backend.ops()));
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
    ASSERT_TRUE(m.configure(kBudget, backend.ops()));
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
    ASSERT_TRUE(m.configure(kBudget, backend.ops()));
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
    ASSERT_TRUE(m.configure(kBudget, backend.ops()));
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
    ASSERT_TRUE(m.configure(kBudget, backend.ops()));
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
    ASSERT_TRUE(m.configure(kSixMiB, backend.ops()));
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
    ASSERT_TRUE(m.configure(kSixMiB, backend.ops()));
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
    ASSERT_TRUE(m.configure(kSixMiB, backend.ops()));
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
    ASSERT_TRUE(m.configure(kSixMiB, backend.ops()));
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

    // Not by an ordinary release, and not by growth's reclamation either — even
    // once the region has republished elsewhere and left it obsolete.
    EXPECT_EQ(m.release_unreferenced(), 0);
    EXPECT_TRUE(backend.released.empty());
    void *successor = m.acquire(region, 2, kTwoMiB);
    ASSERT_NE(successor, nullptr);
    m.note_published(region, successor);
    retire(m, 0, 2);
    EXPECT_EQ(m.acquire(region, 3, kFourMiB), nullptr);
    EXPECT_TRUE(backend.released.empty());
    EXPECT_TRUE(m.owns(mapped));
}

TEST(WorkspaceManagerBudget, AGivenUpClaimStillWaitsForItsLastConsumer) {
    FakeBackend backend;
    WorkspaceManager m;
    ASSERT_TRUE(m.configure(kSixMiB, backend.ops()));
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
    ASSERT_TRUE(m.configure(kSixMiB, backend.ops()));
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

}  // namespace
