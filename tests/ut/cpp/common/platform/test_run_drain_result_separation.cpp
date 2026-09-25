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

/**
 * Which half of a drain's result may decide a device lifecycle fact.
 *
 * `DrainOutcome` carries two results because they answer different questions.
 * `combined()` is what the caller is told, so a diagnostics ownership failure
 * still fails the run; `device_rc` is the only one that is a physical fact
 * about the device, and it alone may record
 * `WorkspaceManager::RunFact::DrainProvedComplete`.
 *
 * Reading the composed value there is not a conservative approximation. A run
 * whose device work finished but whose diagnostics could not prove ownership
 * would never be `retired()`, so at `ContextDestroyed` every block it
 * referenced is quarantined — excluded from reuse and from release for the
 * manager's life, with `proof_unavailable` set. These cases pin the split at
 * the two things that depend on it: the caller's error, and the block's fate.
 *
 * The manager is the production one and the composition is `DrainOutcome`'s
 * own; what a case supplies is the pair of results a drain returned, in the
 * order the two onboard c_api sites report them.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <map>
#include <vector>

#include "host/workspace_manager.h"
#include "worker/native_run_execution.h"

namespace {

constexpr uint64_t kBudget = 1u << 20;
constexpr uint32_t kSlot = 0;
constexpr uint64_t kEpoch = 11;
// A diagnostics result with no device meaning: the retained-ArgsDump close
// returns this when it cannot prove it owns everything the run collected.
constexpr int kDiagnosticsError = PTO_RUNTIME_ERR_INTERNAL;
// The code an a2a3 stream synchronize produced when the device did not finish.
constexpr int kDeviceError = 507018;

/** Allocations at distinct addresses; the production backend needs a device. */
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
        next_ += 0x10000;
        live_bytes[reinterpret_cast<void *>(next_)] = bytes;
        return reinterpret_cast<void *>(next_);
    }

    int release(void *base) {
        released.push_back(base);
        live_bytes.erase(base);
        return 0;
    }

    std::vector<void *> released;
    std::map<void *, size_t> live_bytes;

private:
    uintptr_t next_{0x100000};
};

/**
 * What the two onboard c_api drain sites do with an outcome, in their order.
 *
 * Both read `device_rc` for the fact and `combined()` for the caller; this
 * reports the caller's code so a case can assert both halves at once.
 */
int settle_run(WorkspaceManager &m, const DrainOutcome &drain) {
    m.note_run_fact(kSlot, kEpoch, WorkspaceManager::RunFact::Launched);
    m.note_run_fact(kSlot, kEpoch, WorkspaceManager::RunFact::DrainAttempted);
    const int caller_rc = drain.combined();
    if (drain.device_rc == 0) m.note_run_fact(kSlot, kEpoch, WorkspaceManager::RunFact::DrainProvedComplete);
    m.note_run_fact(kSlot, kEpoch, WorkspaceManager::RunFact::CopybackReturned);
    m.note_run_fact(kSlot, kEpoch, WorkspaceManager::RunFact::BindingsReleased);
    return caller_rc;
}

}  // namespace

/**
 * A drain whose device half succeeded and whose diagnostics half failed: the
 * run fails for the caller, and its workspace blocks are still retired.
 */
TEST(RunDrainResultSeparation, ADiagnosticsFailureFailsTheCallerAndStillRetiresTheRun) {
    FakeBackend backend;
    WorkspaceManager m;
    ASSERT_TRUE(m.configure(kBudget, backend.ops()));
    void *block = m.acquire(WorkspaceManager::staging_region(kSlot), kEpoch, 4096);
    ASSERT_NE(block, nullptr);

    const DrainOutcome drain = DrainOutcome::device_complete(kDiagnosticsError);
    // Visible: the caller is told the run failed, so no one reads an output
    // whose completeness nothing established.
    EXPECT_EQ(settle_run(m, drain), kDiagnosticsError);

    // And correctly attributed: the device finished, so the block carries no
    // consumer and is reusable before the context is destroyed at all.
    EXPECT_EQ(m.block_state(block), WorkspaceManager::BlockState::ProvenUnused);
    EXPECT_EQ(m.live_drainable_consumers(), 0u);

    m.note_run_fact(kSlot, kEpoch, WorkspaceManager::RunFact::ContextDestroyed);
    EXPECT_EQ(m.block_state(block), WorkspaceManager::BlockState::ProvenUnused);

    SimplerWorkspaceReport report{};
    ASSERT_TRUE(m.report(&report));
    EXPECT_EQ(report.quarantined_blocks, 0u);
    EXPECT_EQ(report.quarantined_mapped_bytes, 0u);
    EXPECT_EQ(report.proof_unavailable, 0u);

    // Nothing holds it, so the manager may hand the memory back.
    EXPECT_EQ(m.release_unreferenced(), 0);
    EXPECT_EQ(backend.released.size(), 1u);
    EXPECT_EQ(backend.released.front(), block);
}

/**
 * A drain whose device half failed still quarantines, and the device error is
 * what the caller is told even when the diagnostics half failed too.
 */
TEST(RunDrainResultSeparation, ADeviceErrorKeepsPriorityAndQuarantinesTheRunsBlocks) {
    FakeBackend backend;
    WorkspaceManager m;
    ASSERT_TRUE(m.configure(kBudget, backend.ops()));
    void *block = m.acquire(WorkspaceManager::staging_region(kSlot), kEpoch, 4096);
    ASSERT_NE(block, nullptr);

    DrainOutcome drain = DrainOutcome::device_error(kDeviceError);
    drain.diagnostics_rc = kDiagnosticsError;
    EXPECT_EQ(settle_run(m, drain), kDeviceError);

    // The device never proved it finished, so the run keeps its hold: these
    // bytes may still be read by work the host cannot see.
    EXPECT_EQ(m.block_state(block), WorkspaceManager::BlockState::Referenced);
    EXPECT_EQ(m.live_drainable_consumers(), 1u);

    // No further fact can arrive after this, so the hold becomes permanent.
    m.note_run_fact(kSlot, kEpoch, WorkspaceManager::RunFact::ContextDestroyed);
    EXPECT_EQ(m.block_state(block), WorkspaceManager::BlockState::Quarantined);

    SimplerWorkspaceReport report{};
    ASSERT_TRUE(m.report(&report));
    EXPECT_EQ(report.quarantined_blocks, 1u);
    EXPECT_EQ(report.proof_unavailable, 1u);

    EXPECT_EQ(m.release_unreferenced(), 0);
    EXPECT_TRUE(backend.released.empty());
}
