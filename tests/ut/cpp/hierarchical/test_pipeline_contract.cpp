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

#include "pipeline_contract.h"
#include "pipeline_slot_pool.h"

namespace {

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
    const PipelineContract c = accepted_contract();
    EXPECT_TRUE(is_valid_pipeline_contract(&c));
}

// A runtime that exports no contract is handled by the caller, not here.
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
    EXPECT_EQ(pipeline_resource_copy_count(c, c.resources[0]), 2u);
    EXPECT_EQ(pipeline_resource_copy_count(c, c.resources[1]), 1u);
    EXPECT_EQ(pipeline_resource_copy_count(c, c.resources[2]), 2u);

    const PipelineSlotLease second_slot{1, 0, 7};
    EXPECT_EQ(pipeline_resource_slot(c, c.resources[0], second_slot), 1u);
    EXPECT_EQ(pipeline_resource_slot(c, c.resources[1], second_slot), 0u);
    EXPECT_EQ(pipeline_resource_slot(c, c.resources[2], second_slot), 1u);
}

TEST(PipelineContract, RejectsDepthOutsideSupportedRange) {
    PipelineContract c = accepted_contract();
    c.pipeline_depth = 0;
    EXPECT_FALSE(is_valid_pipeline_contract(&c));
    c.pipeline_depth = PTO_PIPELINE_MAX_DEPTH + 1;
    EXPECT_FALSE(is_valid_pipeline_contract(&c));
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

TEST(PipelineSlotPool, DepthOneKeepsLegacySingleSlotBehavior) {
    PipelineSlotPool pool(1);
    auto first = pool.try_acquire();
    ASSERT_TRUE(first.has_value());
    EXPECT_EQ(first->slot_id, 0u);
    EXPECT_FALSE(pool.try_acquire().has_value());
    EXPECT_TRUE(pool.release(*first));
}

TEST(PipelineSlotPool, DepthTwoProvidesExactlyTwoIndependentLeases) {
    PipelineSlotPool pool(2);
    auto first = pool.try_acquire();
    auto second = pool.try_acquire();
    ASSERT_TRUE(first.has_value());
    ASSERT_TRUE(second.has_value());
    EXPECT_NE(first->slot_id, second->slot_id);
    EXPECT_FALSE(pool.try_acquire().has_value());
    EXPECT_TRUE(pool.owns(*first));
    EXPECT_TRUE(pool.owns(*second));
}

TEST(PipelineSlotPool, AdmissionDepthCanConservativelyLimitACapablePool) {
    PipelineSlotPool pool(2);
    auto first = pool.try_acquire(/*admission_depth=*/1);
    ASSERT_TRUE(first.has_value());
    EXPECT_EQ(first->slot_id, 0u);
    EXPECT_FALSE(pool.try_acquire(/*admission_depth=*/1).has_value());
    EXPECT_TRUE(pool.release(*first));
    EXPECT_THROW((void)pool.try_acquire(/*admission_depth=*/3), std::invalid_argument);
}

TEST(PipelineSlotPool, StaleGenerationCannotAccessOrReleaseAReusedSlot) {
    PipelineSlotPool pool(1);
    const PipelineSlotLease first = *pool.try_acquire();
    ASSERT_TRUE(pool.release(first));
    EXPECT_TRUE(pool.release(first));  // idempotent before reuse

    const PipelineSlotLease replacement = *pool.try_acquire();
    ASSERT_EQ(replacement.slot_id, first.slot_id);
    ASSERT_GT(replacement.generation, first.generation);
    EXPECT_FALSE(pool.owns(first));
    EXPECT_FALSE(pool.release(first));
    EXPECT_TRUE(pool.owns(replacement));
    EXPECT_TRUE(pool.release(replacement));
}

TEST(PipelineSlotGenerationFilter, CompatibilityPreviewDoesNotConsumeGeneration) {
    PipelineSlotGenerationFilter filter;
    const PipelineSlotLease retried{1, 0, 2};

    EXPECT_TRUE(filter.is_admissible(retried));
    EXPECT_TRUE(filter.admit(PipelineSlotLease{1, 0, 1}));
    EXPECT_TRUE(filter.admit(retried));
}

TEST(PipelineSlotGenerationFilter, CommitRechecksGenerationAfterCompatibilityPreview) {
    PipelineSlotGenerationFilter filter;
    const PipelineSlotLease delayed{0, 0, 1};

    EXPECT_TRUE(filter.is_admissible(delayed));
    EXPECT_TRUE(filter.admit(PipelineSlotLease{0, 0, 2}));
    EXPECT_FALSE(filter.admit(delayed));
}

}  // namespace
