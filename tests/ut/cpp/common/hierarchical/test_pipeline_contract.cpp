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

#include <optional>
#include <string>
#include <type_traits>

#include <gtest/gtest.h>

#include "chip_worker.h"
#include "pipeline_contract.h"
#include "run_resource_arena.h"

namespace {

// The binding passes the existing parameters positionally. Borrowed context
// generation is appended and optional; an insertion would change those callers.
using ExpectedChipWorkerInit = void (ChipWorker::*)(
    const std::string &, const std::string &, const std::string &, const std::string &, int, const CallConfig *, bool,
    const std::string &, const std::string &, bool, uint64_t, bool, uint32_t, std::optional<uint64_t>
);
static_assert(
    std::is_same_v<decltype(&ChipWorker::init), ExpectedChipWorkerInit>,
    "ChipWorker::init signature changed: append new parameters, never insert"
);

// A contract a runtime could legitimately ship today: one host-filled region,
// one device-built region, and the two execution handles.
PipelineContract accepted_contract() {
    PipelineContract c{};
    c.abi_version = PTO_PIPELINE_CONTRACT_ABI_VERSION;
    c.resource_count = 4;
    c.pipeline_depth = 1;
    c.resources[0] = {PTO_PIPELINE_TASK_ARGS, PTO_PIPELINE_HOST_PER_RUN, 0};
    c.resources[1] = {PTO_PIPELINE_RUNTIME_IMAGE, PTO_PIPELINE_DEVICE_SCRATCH, 0};
    c.resources[2] = {PTO_PIPELINE_AICPU_STREAM, PTO_PIPELINE_EXEC_HANDLE, 0};
    c.resources[3] = {PTO_PIPELINE_AICORE_STREAM, PTO_PIPELINE_EXEC_HANDLE, 0};
    return c;
}

TEST(PipelineContract, AcceptsADeclarationThisBuildCanHonor) {
    EXPECT_EQ(PTO_PIPELINE_CONTRACT_ABI_VERSION, 2);
    const PipelineContract c = accepted_contract();
    EXPECT_TRUE(is_valid_pipeline_contract(&c));
}

// Runtime loading requires a contract; this helper still rejects malformed null values.
TEST(PipelineContract, RejectsNull) { EXPECT_FALSE(is_valid_pipeline_contract(nullptr)); }

TEST(PipelineContract, AcceptsAnEmptyResourceList) {
    PipelineContract c = accepted_contract();
    c.resource_count = 0;
    EXPECT_TRUE(is_valid_pipeline_contract(&c));
}

TEST(PipelineContract, RejectsAnotherAbiVersion) {
    PipelineContract c = accepted_contract();
    c.abi_version = PTO_PIPELINE_CONTRACT_ABI_VERSION + 1;
    EXPECT_FALSE(is_valid_pipeline_contract(&c));
    c.abi_version = 0;
    EXPECT_FALSE(is_valid_pipeline_contract(&c));
}

TEST(PipelineContract, RejectsMoreResourcesThanFit) {
    PipelineContract c = accepted_contract();
    c.resource_count = PTO_PIPELINE_MAX_RESOURCES + 1;
    EXPECT_FALSE(is_valid_pipeline_contract(&c));
}

TEST(PipelineContract, AcceptsDepthTwoAndDerivesResourceCopies) {
    PipelineContract c = accepted_contract();
    c.pipeline_depth = 2;
    ASSERT_TRUE(is_valid_pipeline_contract(&c));
    // Only a host-filled region is replicated per in-flight run. A device
    // scratch region is reused across runs, and an execution handle is held
    // for the runner's lifetime, so both stay at one.
    EXPECT_EQ(pipeline_resource_copy_count(c, c.resources[0]), 2u);  // TASK_ARGS, HOST_PER_RUN
    EXPECT_EQ(pipeline_resource_copy_count(c, c.resources[1]), 1u);  // RUNTIME_IMAGE, DEVICE_SCRATCH
    EXPECT_EQ(pipeline_resource_copy_count(c, c.resources[2]), 1u);  // AICPU_STREAM, EXEC_HANDLE

    const PipelineSlotLease second_slot{1, 0, 7};
    EXPECT_EQ(pipeline_resource_slot(c, c.resources[0], second_slot), 1u);
    EXPECT_EQ(pipeline_resource_slot(c, c.resources[1], second_slot), 0u);
    EXPECT_EQ(pipeline_resource_slot(c, c.resources[2], second_slot), 0u);
}

TEST(PipelineContract, RejectsNoDepthAndAcceptsAnyDeclaredOne) {
    PipelineContract c = accepted_contract();
    c.pipeline_depth = 0;
    EXPECT_FALSE(is_valid_pipeline_contract(&c));
    // A declaration is what a runtime says it can support, not a count anything
    // commits to, so there is no upper bound to reject against: what a context
    // uses is the grant, and every per-run resource is built to that.
    c.pipeline_depth = 64;
    EXPECT_TRUE(is_valid_pipeline_contract(&c));
    c.pipeline_depth = UINT32_MAX;
    EXPECT_TRUE(is_valid_pipeline_contract(&c));
}

TEST(PipelineContract, RejectsAnOutOfRangeKindOrClass) {
    PipelineContract c = accepted_contract();
    c.resources[0].kind = PTO_PIPELINE_AICORE_STREAM + 1;
    EXPECT_FALSE(is_valid_pipeline_contract(&c));

    c = accepted_contract();
    c.resources[0].resource_class = PTO_PIPELINE_EXEC_HANDLE + 1;
    EXPECT_FALSE(is_valid_pipeline_contract(&c));
}

// bytes_per_copy is reserved: nothing sizes anything from it yet, so a runtime
// that populates it is declaring a contract this build does not implement.
TEST(PipelineContract, RejectsANonZeroReservedSize) {
    PipelineContract c = accepted_contract();
    c.resources[1].bytes_per_copy = 4096;
    EXPECT_FALSE(is_valid_pipeline_contract(&c));
}

TEST(PipelineContract, RejectsAnUnspecifiedKind) {
    PipelineContract c = accepted_contract();
    c.resources[1].kind = PTO_PIPELINE_KIND_UNSPECIFIED;
    EXPECT_FALSE(is_valid_pipeline_contract(&c));
}

// A resource_count larger than the entries a runtime filled in leaves trailing
// zeroed entries, and zero is not a resource. This must hold whatever the
// filled entries are — a rule keyed on a collision with the first kind would
// pass only when the declaration happens to use that kind.
TEST(PipelineContract, RejectsAResourceCountPastTheFilledEntries) {
    for (uint32_t filled :
         {static_cast<uint32_t>(PTO_PIPELINE_GM_HEAP), static_cast<uint32_t>(PTO_PIPELINE_TASK_ARGS)}) {
        PipelineContract c{};
        c.abi_version = PTO_PIPELINE_CONTRACT_ABI_VERSION;
        c.pipeline_depth = 1;
        c.resources[0] = {filled, PTO_PIPELINE_HOST_PER_RUN, 0};

        c.resource_count = 2;  // overstates by exactly one
        EXPECT_FALSE(is_valid_pipeline_contract(&c)) << "filled kind " << filled;
        c.resource_count = 3;
        EXPECT_FALSE(is_valid_pipeline_contract(&c)) << "filled kind " << filled;
    }
}

// A kind names a resource type, not one instance of it, so a runtime that needs
// two of a kind — two AICore streams for parallel branches, say — can say so.
TEST(PipelineContract, AcceptsARepeatedKind) {
    PipelineContract c = accepted_contract();
    c.resources[3].kind = PTO_PIPELINE_AICPU_STREAM;
    EXPECT_TRUE(is_valid_pipeline_contract(&c));
}

TEST(PipelineContract, AcceptsEveryKindOnce) {
    PipelineContract c{};
    c.abi_version = PTO_PIPELINE_CONTRACT_ABI_VERSION;
    c.pipeline_depth = 1;
    c.resource_count = PTO_PIPELINE_AICORE_STREAM;
    for (uint32_t kind = PTO_PIPELINE_GM_HEAP; kind <= PTO_PIPELINE_AICORE_STREAM; ++kind) {
        c.resources[kind - 1] = {kind, PTO_PIPELINE_HOST_PER_RUN, 0};
    }
    EXPECT_TRUE(is_valid_pipeline_contract(&c));
}

// GM heap, GM shared memory, and the runtime image are committed by one
// setup_static_arena call into one bank, so a contract that gives them
// different copy counts describes a layout no runner can build.
TEST(PipelineContract, RejectsArenaKindsThatDisagreeOnCopyCount) {
    PipelineContract c{PTO_PIPELINE_CONTRACT_ABI_VERSION, 2, 2, {}};
    c.resources[0] = {PTO_PIPELINE_GM_HEAP, PTO_PIPELINE_HOST_PER_RUN, 0};
    c.resources[1] = {PTO_PIPELINE_GM_SM, PTO_PIPELINE_DEVICE_SCRATCH, 0};
    ASSERT_TRUE(is_valid_pipeline_contract(&c));
    EXPECT_FALSE(has_serviceable_arena_topology(c));

    c.resources[1].resource_class = PTO_PIPELINE_HOST_PER_RUN;
    EXPECT_TRUE(has_serviceable_arena_topology(c));

    // At depth one every class collapses to one copy, so the same declaration
    // is serviceable there.
    c.resources[1].resource_class = PTO_PIPELINE_DEVICE_SCRATCH;
    c.pipeline_depth = 1;
    EXPECT_TRUE(has_serviceable_arena_topology(c));
}

// Only the first entry of a kind is ever read, so a second one is a silent
// misdeclaration for the kinds that select a bank.
TEST(PipelineContract, RejectsARepeatedArenaKind) {
    PipelineContract c{PTO_PIPELINE_CONTRACT_ABI_VERSION, 2, 2, {}};
    c.resources[0] = {PTO_PIPELINE_GM_HEAP, PTO_PIPELINE_HOST_PER_RUN, 0};
    c.resources[1] = {PTO_PIPELINE_GM_HEAP, PTO_PIPELINE_HOST_PER_RUN, 0};
    ASSERT_TRUE(is_valid_pipeline_contract(&c));
    EXPECT_FALSE(has_serviceable_arena_topology(c));
}

// A repeated stream kind still selects nothing, so it stays legal.
TEST(PipelineContract, AcceptsARepeatedNonArenaKind) {
    PipelineContract c{PTO_PIPELINE_CONTRACT_ABI_VERSION, 2, 2, {}};
    c.resources[0] = {PTO_PIPELINE_AICPU_STREAM, PTO_PIPELINE_EXEC_HANDLE, 0};
    c.resources[1] = {PTO_PIPELINE_AICPU_STREAM, PTO_PIPELINE_EXEC_HANDLE, 0};
    ASSERT_TRUE(is_valid_pipeline_contract(&c));
    EXPECT_TRUE(has_serviceable_arena_topology(c));
}

// Both shipped A2/A3 contracts must be serviceable as declared.
TEST(PipelineContract, ShippedArenaTopologiesAreServiceable) {
    PipelineContract hbg{PTO_PIPELINE_CONTRACT_ABI_VERSION, 3, 2, {}};
    hbg.resources[0] = {PTO_PIPELINE_GM_HEAP, PTO_PIPELINE_HOST_PER_RUN, 0};
    hbg.resources[1] = {PTO_PIPELINE_GM_SM, PTO_PIPELINE_HOST_PER_RUN, 0};
    hbg.resources[2] = {PTO_PIPELINE_RUNTIME_IMAGE, PTO_PIPELINE_HOST_PER_RUN, 0};
    EXPECT_TRUE(has_serviceable_arena_topology(hbg));

    PipelineContract tmr{PTO_PIPELINE_CONTRACT_ABI_VERSION, 3, 2, {}};
    tmr.resources[0] = {PTO_PIPELINE_GM_HEAP, PTO_PIPELINE_DEVICE_SCRATCH, 0};
    tmr.resources[1] = {PTO_PIPELINE_GM_SM, PTO_PIPELINE_DEVICE_SCRATCH, 0};
    tmr.resources[2] = {PTO_PIPELINE_RUNTIME_IMAGE, PTO_PIPELINE_DEVICE_SCRATCH, 0};
    EXPECT_TRUE(has_serviceable_arena_topology(tmr));
}

PipelineContract kernel_contract() {
    PipelineContract c{PTO_PIPELINE_CONTRACT_ABI_VERSION, 6, 2, {}};
    for (uint32_t kind = PTO_PIPELINE_GM_HEAP; kind <= PTO_PIPELINE_RUNTIME_IMAGE; ++kind) {
        c.resources[kind - 1] = {kind, PTO_PIPELINE_DEVICE_SCRATCH, 4096};
    }
    c.resources[3] = {PTO_PIPELINE_TASK_ARGS, PTO_PIPELINE_HOST_PER_RUN, 4096};
    c.resources[4] = {PTO_PIPELINE_AICPU_STREAM, PTO_PIPELINE_EXEC_HANDLE, 0};
    c.resources[5] = {PTO_PIPELINE_AICORE_STREAM, PTO_PIPELINE_EXEC_HANDLE, 0};
    return c;
}

TEST(PipelineContract, KernelByteRulesAreModeSpecific) {
    auto c = kernel_contract();
    EXPECT_TRUE(is_valid_pipeline_contract(&c, SIMPLER_MODE_KERNEL));
    EXPECT_TRUE(is_valid_tmr_kernel_pipeline_contract(&c));
    EXPECT_FALSE(is_valid_pipeline_contract(&c));
    EXPECT_FALSE(is_valid_pipeline_contract(&c, 99u));
    EXPECT_FALSE(is_valid_pipeline_contract(&c, UINT32_MAX));
    EXPECT_FALSE(is_valid_pipeline_contract(nullptr, SIMPLER_MODE_KERNEL));
    for (uint32_t i = 0; i < c.resource_count; ++i) {
        auto invalid = c;
        invalid.resources[i].bytes_per_copy = i < 4 ? 0 : 1;
        EXPECT_FALSE(is_valid_tmr_kernel_pipeline_contract(&invalid)) << i;
        EXPECT_EQ(is_valid_pipeline_contract(&invalid, SIMPLER_MODE_KERNEL), i == 3) << i;
    }
}

TEST(PipelineContract, KernelPerRunImagesMayBeSizedAtBind) {
    constexpr uint32_t kDeclaredDepth = 3;
    PipelineContract c{PTO_PIPELINE_CONTRACT_ABI_VERSION, 4, kDeclaredDepth, {}};
    c.resources[0] = {PTO_PIPELINE_GM_HEAP, PTO_PIPELINE_HOST_PER_RUN, 0};
    c.resources[1] = {PTO_PIPELINE_RUNTIME_IMAGE, PTO_PIPELINE_HOST_PER_RUN, 0};
    c.resources[2] = {PTO_PIPELINE_AICPU_STREAM, PTO_PIPELINE_EXEC_HANDLE, 0};
    c.resources[3] = {PTO_PIPELINE_AICORE_STREAM, PTO_PIPELINE_EXEC_HANDLE, 0};
    EXPECT_TRUE(is_valid_pipeline_contract(&c, SIMPLER_MODE_KERNEL));
    EXPECT_TRUE(has_serviceable_arena_topology(c));
    EXPECT_TRUE(has_serviceable_stream_topology(c));
    EXPECT_EQ(pipeline_resource_copy_count(c, c.resources[0]), kDeclaredDepth);
    c.resources[0].resource_class = PTO_PIPELINE_DEVICE_SCRATCH;
    EXPECT_FALSE(is_valid_pipeline_contract(&c, SIMPLER_MODE_KERNEL));
}

TEST(PipelineContract, KernelRequiresExactlyItsSupportedResourceShape) {
    const auto c = kernel_contract();
    for (uint32_t i = 0; i < c.resource_count; ++i) {
        auto missing = c;
        missing.resources[i] = missing.resources[--missing.resource_count];
        EXPECT_FALSE(is_valid_tmr_kernel_pipeline_contract(&missing)) << i;
        auto wrong_class = c;
        wrong_class.resources[i].resource_class = (wrong_class.resources[i].resource_class + 1) % 3;
        EXPECT_FALSE(is_valid_tmr_kernel_pipeline_contract(&wrong_class)) << i;
        auto duplicate = c;
        duplicate.resources[i] = duplicate.resources[(i + 1) % c.resource_count];
        EXPECT_FALSE(is_valid_tmr_kernel_pipeline_contract(&duplicate)) << i;
    }
    auto invalid = c;
    // Depth is the runtime's choice across the supported range; the validator
    // bounds it rather than fixing it. What a given depth costs, and how a
    // captured graph maps its calls onto banks, are decided where the resources
    // are established, not here.
    invalid.pipeline_depth = 1;
    EXPECT_TRUE(is_valid_tmr_kernel_pipeline_contract(&invalid));
    invalid.pipeline_depth = 0;
    EXPECT_FALSE(is_valid_tmr_kernel_pipeline_contract(&invalid));
    invalid.pipeline_depth = 64;
    EXPECT_TRUE(is_valid_tmr_kernel_pipeline_contract(&invalid));
    invalid = c;
    invalid.resource_count = PTO_PIPELINE_MAX_RESOURCES + 1;
    EXPECT_FALSE(is_valid_tmr_kernel_pipeline_contract(&invalid));
    EXPECT_FALSE(has_serviceable_stream_topology(invalid));
    EXPECT_FALSE(is_valid_tmr_kernel_pipeline_contract(nullptr));
}

TEST(PipelineContract, TmrRequiredKindsAreOrderIndependentAndRejectExtraArgs) {
    auto c = kernel_contract();
    const auto first = c.resources[0];
    c.resources[0] = c.resources[3];
    c.resources[3] = first;
    EXPECT_TRUE(is_valid_tmr_kernel_pipeline_contract(&c));
    c.resources[c.resource_count++] = c.resources[0];
    ASSERT_TRUE(is_valid_pipeline_contract(&c, SIMPLER_MODE_KERNEL));
    ASSERT_TRUE(has_serviceable_arena_topology(c));
    ASSERT_TRUE(has_serviceable_stream_topology(c));
    EXPECT_FALSE(is_valid_tmr_kernel_pipeline_contract(&c));
}

TEST(PipelineContract, StreamServiceabilityDoesNotChangeStructuralRules) {
    auto c = accepted_contract();
    EXPECT_TRUE(has_serviceable_stream_topology(c));
    c.pipeline_depth = 2;
    EXPECT_TRUE(has_serviceable_stream_topology(c));
    c.resources[3].kind = PTO_PIPELINE_AICPU_STREAM;
    EXPECT_TRUE(is_valid_pipeline_contract(&c));
    EXPECT_TRUE(has_serviceable_arena_topology(c));
    EXPECT_FALSE(has_serviceable_stream_topology(c));
    c = accepted_contract();
    c.resources[2].resource_class = PTO_PIPELINE_HOST_PER_RUN;
    EXPECT_FALSE(has_serviceable_stream_topology(c));
    c.resource_count = 0;
    EXPECT_TRUE(is_valid_pipeline_contract(&c));
    EXPECT_FALSE(has_serviceable_stream_topology(c));
}

TEST(PipelineContract, CommonKernelAdmissionDoesNotRequireTmrResourceSet) {
    const PipelineContract c{
        PTO_PIPELINE_CONTRACT_ABI_VERSION,
        4,
        1,
        {
            {PTO_PIPELINE_GM_HEAP, PTO_PIPELINE_HOST_PER_RUN, 4096},
            {PTO_PIPELINE_RUNTIME_IMAGE, PTO_PIPELINE_HOST_PER_RUN, 4096},
            {PTO_PIPELINE_AICPU_STREAM, PTO_PIPELINE_EXEC_HANDLE, 0},
            {PTO_PIPELINE_AICORE_STREAM, PTO_PIPELINE_EXEC_HANDLE, 0},
        }
    };
    ASSERT_TRUE(is_valid_pipeline_contract(&c, SIMPLER_MODE_KERNEL));
    EXPECT_TRUE(has_serviceable_arena_topology(c));
    EXPECT_TRUE(has_serviceable_stream_topology(c));
    EXPECT_FALSE(is_valid_tmr_kernel_pipeline_contract(&c));
}

TEST(RunResourceArena, OneSetKeepsLegacySingleSlotBehavior) {
    RunResourceArena arena(1);
    auto first = arena.try_acquire();
    ASSERT_TRUE(first.has_value());
    EXPECT_EQ(first->slot_id, 0u);
    EXPECT_FALSE(arena.try_acquire().has_value());
    EXPECT_TRUE(arena.release(*first));
}

TEST(RunResourceArena, TwoSetsProvideExactlyTwoIndependentLeases) {
    RunResourceArena arena(2);
    auto first = arena.try_acquire();
    auto second = arena.try_acquire();
    ASSERT_TRUE(first.has_value());
    ASSERT_TRUE(second.has_value());
    EXPECT_NE(first->slot_id, second->slot_id);
    EXPECT_FALSE(arena.try_acquire().has_value());
    EXPECT_TRUE(arena.owns(*first));
    EXPECT_TRUE(arena.owns(*second));
}

TEST(RunResourceArena, AdmissionDepthCanConservativelyLimitACapableArena) {
    RunResourceArena arena(2);
    auto first = arena.try_acquire(/*admission_depth=*/1);
    ASSERT_TRUE(first.has_value());
    EXPECT_EQ(first->slot_id, 0u);
    EXPECT_FALSE(arena.try_acquire(/*admission_depth=*/1).has_value());
    EXPECT_TRUE(arena.release(*first));
    EXPECT_THROW((void)arena.try_acquire(/*admission_depth=*/3), std::invalid_argument);
}

TEST(RunResourceArena, StaleGenerationCannotAccessOrReleaseAReusedSet) {
    RunResourceArena arena(1);
    const PipelineSlotLease first = *arena.try_acquire();
    ASSERT_TRUE(arena.release(first));
    EXPECT_TRUE(arena.release(first));  // idempotent before reuse

    const PipelineSlotLease replacement = *arena.try_acquire();
    ASSERT_EQ(replacement.slot_id, first.slot_id);
    ASSERT_GT(replacement.generation, first.generation);
    EXPECT_FALSE(arena.owns(first));
    EXPECT_FALSE(arena.release(first));
    EXPECT_TRUE(arena.owns(replacement));
    EXPECT_TRUE(arena.release(replacement));
}

// No ceiling: a grant is whatever the context was granted, and every set in it
// is a set a run can actually hold.
TEST(RunResourceArena, AGrantFarAboveTheOldCeilingHandsOutEverySet) {
    constexpr uint32_t kGranted = 64;
    RunResourceArena arena(kGranted);
    EXPECT_EQ(arena.granted(), kGranted);
    EXPECT_EQ(arena.free_count(), kGranted);

    std::vector<PipelineSlotLease> held;
    for (uint32_t i = 0; i < kGranted; ++i) {
        auto lease = arena.try_acquire();
        ASSERT_TRUE(lease.has_value()) << "set " << i << " of the grant was not available";
        held.push_back(*lease);
    }
    EXPECT_FALSE(arena.try_acquire().has_value());
    EXPECT_EQ(arena.live_count(), kGranted);
    EXPECT_EQ(arena.free_count(), 0u);
    for (const PipelineSlotLease &lease : held) {
        EXPECT_TRUE(arena.release(lease));
    }
    EXPECT_EQ(arena.free_count(), kGranted);
}

// The three counts always partition the grant, and a quarantined set stays in
// it: a context cannot buy itself more concurrency by failing.
TEST(RunResourceArena, QuarantineWithdrawsASetWithoutLoweringTheGrant) {
    RunResourceArena arena(3);
    const PipelineSlotLease first = *arena.try_acquire();
    const PipelineSlotLease second = *arena.try_acquire();
    EXPECT_EQ(arena.live_count(), 2u);
    EXPECT_EQ(arena.free_count(), 1u);
    EXPECT_EQ(arena.quarantined_count(), 0u);

    ASSERT_TRUE(arena.quarantine(first));
    EXPECT_EQ(arena.quarantined_count(), 1u);
    EXPECT_EQ(arena.live_count(), 1u);
    EXPECT_EQ(arena.free_count(), 1u);
    EXPECT_TRUE(arena.quarantined(first.slot_id));
    EXPECT_EQ(arena.free_count() + arena.live_count() + arena.quarantined_count(), arena.granted());

    // Never handed out again, and never released back into service.
    EXPECT_FALSE(arena.release(first));
    const PipelineSlotLease third = *arena.try_acquire();
    EXPECT_NE(third.slot_id, first.slot_id);
    EXPECT_FALSE(arena.try_acquire().has_value()) << "the quarantined set was handed to a run";
    EXPECT_TRUE(arena.release(second));
    EXPECT_TRUE(arena.release(third));
    EXPECT_EQ(arena.free_count(), 2u);
    EXPECT_EQ(arena.quarantined_count(), 1u);
}

// A grant settled after construction resizes the arena, but only while no set
// has been handed out: an index whose meaning changed under a live run is a
// different resource, not a bigger capacity.
TEST(RunResourceArena, RegrantIsRefusedOnceASetHasBeenHandedOut) {
    RunResourceArena arena(kDefaultRunResourceSets);
    arena.regrant(8);
    EXPECT_EQ(arena.granted(), 8u);
    EXPECT_EQ(arena.free_count(), 8u);

    const PipelineSlotLease lease = *arena.try_acquire();
    EXPECT_THROW(arena.regrant(16), std::logic_error);
    EXPECT_TRUE(arena.release(lease));
    EXPECT_THROW(arena.regrant(16), std::logic_error) << "a reused generation was forgotten by a regrant";
    EXPECT_THROW(arena.regrant(0), std::invalid_argument);
}

// The filter is sized by the grant, so a set beyond it is refused rather than
// silently admitted against an index nobody owns.
TEST(PipelineSlotGenerationFilter, SizingTracksTheGrant) {
    PipelineSlotGenerationFilter filter;
    EXPECT_FALSE(filter.is_admissible(PipelineSlotLease{0, 0, 1})) << "an unsized filter admitted a lease";
    filter.resize(5);
    EXPECT_TRUE(filter.admit(PipelineSlotLease{4, 0, 1}));
    EXPECT_FALSE(filter.admit(PipelineSlotLease{5, 0, 1}));
    filter.reset();
    EXPECT_TRUE(filter.admit(PipelineSlotLease{4, 0, 1}));
}

TEST(PipelineSlotGenerationFilter, CompatibilityPreviewDoesNotConsumeGeneration) {
    PipelineSlotGenerationFilter filter;
    filter.resize(kDefaultRunResourceSets);
    const PipelineSlotLease retried{1, 0, 2};

    EXPECT_TRUE(filter.is_admissible(retried));
    EXPECT_TRUE(filter.admit(PipelineSlotLease{1, 0, 1}));
    EXPECT_TRUE(filter.admit(retried));
}

TEST(PipelineSlotGenerationFilter, CommitRechecksGenerationAfterCompatibilityPreview) {
    PipelineSlotGenerationFilter filter;
    filter.resize(kDefaultRunResourceSets);
    const PipelineSlotLease delayed{0, 0, 1};

    EXPECT_TRUE(filter.is_admissible(delayed));
    EXPECT_TRUE(filter.admit(PipelineSlotLease{0, 0, 2}));
    EXPECT_FALSE(filter.admit(delayed));
}

}  // namespace
