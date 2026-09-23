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
    EXPECT_EQ(m.acquire(WorkspaceManager::Domain::HostStaging, 0, 1, 4096), nullptr);
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

    void *first = m.acquire(WorkspaceManager::Domain::HostStaging, 0, 1, 4096);
    ASSERT_NE(first, nullptr);
    EXPECT_EQ(backend.acquire_calls, 1);

    // Same slot, a request the block fits — but run 1 has reported nothing, so
    // its contents may still be read and the block is not reusable.
    void *while_held = m.acquire(WorkspaceManager::Domain::HostStaging, 0, 2, 2048);
    EXPECT_NE(while_held, first);
    EXPECT_EQ(backend.acquire_calls, 2);

    // Retire both runs, and the first block becomes reusable at its capacity.
    report_launched_and_drained(m, 0, 1);
    report_host_side_done(m, 0, 1);
    report_launched_and_drained(m, 0, 2);
    report_host_side_done(m, 0, 2);
    void *reused = m.acquire(WorkspaceManager::Domain::HostStaging, 0, 3, 2048);
    EXPECT_TRUE(reused == first || reused == while_held);
    EXPECT_EQ(backend.acquire_calls, 2);  // nothing new was allocated
}

TEST(WorkspaceManager, ADrainThatDidNotProveCompletionDoesNotRetire) {
    FakeBackend backend;
    WorkspaceManager m;
    ASSERT_TRUE(m.configure(kBudget, backend.ops()));
    void *block = m.acquire(WorkspaceManager::Domain::ExecScratch, 0, 9, 4096);
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
    void *block = m.acquire(WorkspaceManager::Domain::HostStaging, 1, 4, 4096);
    ASSERT_NE(block, nullptr);

    // A prepare that failed, or a run cancelled before launch: no device
    // consumer ever existed, so demanding a drain would demand one that cannot
    // happen. The host-side facts still have to arrive.
    m.note_run_fact(1, 4, WorkspaceManager::RunFact::NoDeviceSubmission);
    EXPECT_EQ(m.block_state(block), WorkspaceManager::BlockState::Referenced);
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
    void *g1 = m.acquire(WorkspaceManager::Domain::HostStaging, 0, 1, 1024);
    void *g2 = m.acquire(WorkspaceManager::Domain::HostStaging, 0, 2, 2048);
    void *g3 = m.acquire(WorkspaceManager::Domain::HostStaging, 0, 3, 4096);
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
    void *held = m.acquire(WorkspaceManager::Domain::ExecScratch, 0, 1, 4096);
    ASSERT_NE(held, nullptr);
    const uint64_t charged = m.reserved_bytes();

    // A second generation would exceed the budget, and the first is still held,
    // so it cannot be reused to make room. The refusal leaves the published
    // address, its charge and the block count exactly as they were.
    EXPECT_EQ(m.acquire(WorkspaceManager::Domain::ExecScratch, 0, 2, 8192), nullptr);
    EXPECT_EQ(m.reserved_bytes(), charged);
    EXPECT_EQ(m.block_count(), 1u);
    EXPECT_EQ(m.block_state(held), WorkspaceManager::BlockState::Referenced);
    EXPECT_EQ(backend.acquire_calls, 1);  // the device was never asked
}

TEST(WorkspaceManager, ADeviceAllocationFailureLeavesPublishedStateIntact) {
    FakeBackend backend;
    WorkspaceManager m;
    ASSERT_TRUE(m.configure(kBudget, backend.ops()));
    void *held = m.acquire(WorkspaceManager::Domain::HostStaging, 0, 1, 4096);
    ASSERT_NE(held, nullptr);
    const uint64_t charged = m.reserved_bytes();

    backend.fail_next_acquire = true;
    EXPECT_EQ(m.acquire(WorkspaceManager::Domain::HostStaging, 0, 2, 8192), nullptr);
    EXPECT_EQ(m.reserved_bytes(), charged);
    EXPECT_EQ(m.block_count(), 1u);
    EXPECT_EQ(m.block_bytes(held), 4096u);
}

TEST(WorkspaceManager, AFailedReleaseKeepsItsOwnerAndItsCharge) {
    FakeBackend backend;
    WorkspaceManager m;
    ASSERT_TRUE(m.configure(kBudget, backend.ops()));
    void *block = m.acquire(WorkspaceManager::Domain::ExecScratch, 0, 5, 4096);
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
    void *block = m.acquire(WorkspaceManager::Domain::ExecScratch, 0, 5, 4096);
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
    void *block = m.acquire(WorkspaceManager::Domain::ExecScratch, 0, 11, 4096);
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
    void *again = m.acquire(WorkspaceManager::Domain::ExecScratch, 0, 12, 1024);
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
    void *block = m.acquire(WorkspaceManager::Domain::ExecScratch, 0, 21, 4096);
    ASSERT_NE(block, nullptr);

    // Owns device work and no drain has been entered: exactly the state a
    // caller can resolve by finalizing the run, which is why a teardown
    // refusal on it is a refusal the caller can act on.
    m.note_run_fact(0, 21, WorkspaceManager::RunFact::Launched);
    EXPECT_EQ(m.live_drainable_consumers(), 1u);
    SimplerWorkspaceReport report{};
    ASSERT_TRUE(m.report(&report));
    EXPECT_EQ(report.live_blocked, 1u);

    // Once the drain is entered the proof either arrives or never will, so the
    // run stops counting as drainable either way.
    m.note_run_fact(0, 21, WorkspaceManager::RunFact::DrainAttempted);
    EXPECT_EQ(m.live_drainable_consumers(), 0u);
    m.note_run_fact(0, 21, WorkspaceManager::RunFact::DrainProvedComplete);
    report_host_side_done(m, 0, 21);
    EXPECT_EQ(m.block_state(block), WorkspaceManager::BlockState::ProvenUnused);
}

TEST(WorkspaceManager, TheSweepKeepsQuarantinedBlocksAndRecordsEveryOutcome) {
    FakeBackend backend;
    WorkspaceManager m;
    ASSERT_TRUE(m.configure(kBudget, backend.ops()));
    void *kept = m.acquire(WorkspaceManager::Domain::ExecScratch, 0, 31, 4096);
    void *freed = m.acquire(WorkspaceManager::Domain::HostStaging, 1, 32, 2048);
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
    void *block = m.acquire(WorkspaceManager::Domain::ExecScratch, 0, 41, 4096);
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
    EXPECT_NE(m.acquire(WorkspaceManager::Domain::ExecScratch, 0, 51, 1024), nullptr);
    m.enter_closed();
    EXPECT_EQ(m.acquire(WorkspaceManager::Domain::ExecScratch, 0, 52, 1024), nullptr);
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

}  // namespace
