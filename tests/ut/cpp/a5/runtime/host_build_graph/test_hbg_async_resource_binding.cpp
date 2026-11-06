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

#include <array>
#include <cstring>
#include <vector>

#include "backend/async_resource_binding.h"
#include "scheduler/scheduler_submission_bundle.h"

namespace {
constexpr uint64_t kEpoch = 71;

struct Observations {
    uint64_t value{0};
    uint32_t reads{0};
    uint32_t fences{0};
    std::vector<uint64_t> lines;
};

struct Reader {
    Observations *state{nullptr};
    int32_t counter(uint64_t) {
        ++state->reads;
        return static_cast<int32_t>(state->value);
    }
    uint64_t sdma_post_done(uint64_t) {
        ++state->reads;
        return state->value;
    }
};

struct Cache {
    Observations *state{nullptr};
    void invalidate_line(uint64_t address) { state->lines.push_back(address); }
    void finish() { ++state->fences; }
};

struct Fixture {
    alignas(128) std::array<uint8_t, 8192> image{};
    alignas(64) std::array<uint8_t, 512> records{};
    alignas(64) std::array<uint8_t, 2048> payload{};
    SchedulerAsyncLayout layout{};
    AsyncResourceTables tables{};
    SchedulerAsyncGroupIdentity group{};
    Observations observed;
    SchedulerSubmissionProvenanceView proof{};
    alignas(64) DispatchPayload frame{};

    template <typename T>
    T *at(uint64_t offset) {
        return reinterpret_cast<T *>(image.data() + offset);
    }
    AsyncResourceHeader &header() { return *at<AsyncResourceHeader>(layout.resources_offset); }
    AsyncRetainedSpan *spans() { return at<AsyncRetainedSpan>(layout.resources_offset + sizeof(AsyncResourceHeader)); }
    AsyncResourceRange *ranges() { return reinterpret_cast<AsyncResourceRange *>(spans() + 2); }
    AsyncConditionContract *conditions() { return at<AsyncConditionContract>(layout.uses_offset); }
    AsyncConsumerTaskMap *maps() { return at<AsyncConsumerTaskMap>(layout.consumers_offset); }
    AsyncConsumerContract &consumer() { return *reinterpret_cast<AsyncConsumerContract *>(maps() + 2); }
    uint64_t *consumer_ranges() { return reinterpret_cast<uint64_t *>(&consumer() + header().consumer_count); }
    uint64_t record_address() const { return reinterpret_cast<uint64_t>(records.data()); }
    uint64_t payload_address() const { return reinterpret_cast<uint64_t>(payload.data()); }

    Fixture() {
        layout.run_epoch = kEpoch;
        layout.task_count = 2;
        layout.group_count = 1;
        layout.resources_offset = 256;
        layout.resources_bytes =
            sizeof(AsyncResourceHeader) + 2 * sizeof(AsyncRetainedSpan) + 4 * sizeof(AsyncResourceRange);
        layout.uses_offset = 512;
        layout.uses_bytes = 2 * sizeof(AsyncConditionContract);
        layout.consumers_offset = 768;
        layout.consumers_bytes =
            2 * sizeof(AsyncConsumerTaskMap) + sizeof(AsyncConsumerContract) + 2 * sizeof(uint64_t);
        layout.scratch_offset = 896;
        layout.scratch_bytes = 256;
        layout.total_bytes = 1152;
        auto *cohorts = at<SchedulerAsyncCohortHeader>(layout.scratch_offset);
        *cohorts = {};
        cohorts->total_bytes = layout.scratch_bytes;
        cohorts->run_epoch = kEpoch;
        cohorts->task_count = 2;
        cohorts->maps_offset = sizeof(*cohorts);
        cohorts->auxiliary_offset = cohorts->total_bytes;
        auto *cohort_maps = at<SchedulerAsyncCohortTaskMap>(layout.scratch_offset + cohorts->maps_offset);
        cohort_maps[0] = {SCHEDULER_ASYNC_INDEX_NONE, 1, 2, 0, 0, SCHEDULER_TASK_EXECUTABLE, 0};
        cohort_maps[1] = {SCHEDULER_ASYNC_INDEX_NONE, 1, 1, 0, 0, SCHEDULER_TASK_EXECUTABLE, 0};
        header() = {ASYNC_RESOURCE_VERSION, kEpoch, 2, 4, 2, 1, 2, 0};
        spans()[0] = {record_address(), records.size()};
        spans()[1] = {payload_address(), payload.size()};
        ranges()[0] = {0, 0, 132, ASYNC_RANGE_READ | ASYNC_RANGE_CONDITION_CLEAN};
        ranges()[1] = {1, 64, 128, ASYNC_RANGE_READ | ASYNC_RANGE_WRITE | ASYNC_RANGE_CONSUMER_CLEAN};
        ranges()[2] = {1, 512, 256, ASYNC_RANGE_READ};
        ranges()[3] = {1, 1024, 64, ASYNC_RANGE_READ | ASYNC_RANGE_CONSUMER_CLEAN};
        for (uint32_t index = 0; index < 2; ++index)
            conditions()[index] = {0, 0, 1, index, 3, COMPLETION_TYPE_COUNTER, 0, 0, 2, 128, ASYNC_RESOURCE_NONE, 1};
        maps()[0] = {ASYNC_RESOURCE_NONE, 1, 2, AsyncConsumerMapKind::NONE, 0};
        maps()[1] = {0, 1, 1, AsyncConsumerMapKind::REQUIRED, 0};
        consumer() = {1, 0, 0, 0, 2, 3, 0};
        consumer_ranges()[0] = 1;
        consumer_ranges()[1] = 3;
        group.run_epoch = kEpoch;
        group.task_ref = 0;
        group.subtask_slot = 1;
        group.capacity = 2;
        group.condition_types = CORE_ASYNC_CONDITION_COUNTER | CORE_ASYNC_CONDITION_SDMA_POST_DONE;
        group.binding_uses_offset = layout.uses_offset;
        group.binding_use_count = 1;
        EXPECT_TRUE(AsyncResourceTables::mount(image.data(), layout, &tables));
    }

    void enable_cohort(uint32_t task, uint32_t lanes, uint32_t blocks, bool sync) {
        auto *h = at<SchedulerAsyncCohortHeader>(layout.scratch_offset);
        *h = {};
        h->run_epoch = kEpoch;
        h->task_count = 2;
        h->cohort_count = sync ? 1 : blocks;
        h->maps_offset = 128;
        h->counters_offset = 256;
        h->auxiliary_offset = 256 + h->cohort_count * sizeof(SchedulerAsyncCohortCounter);
        h->total_bytes = h->auxiliary_offset;
        layout.scratch_bytes = h->total_bytes;
        layout.total_bytes = layout.scratch_offset + layout.scratch_bytes;
        auto *rows = at<SchedulerAsyncCohortTaskMap>(layout.scratch_offset + h->maps_offset);
        rows[0] = {SCHEDULER_ASYNC_INDEX_NONE, 1, 2, 0, 0, SCHEDULER_TASK_EXECUTABLE, 0};
        rows[1] = {SCHEDULER_ASYNC_INDEX_NONE, 1, 1, 0, 0, SCHEDULER_TASK_EXECUTABLE, 0};
        const uint32_t active = __builtin_popcount(lanes);
        rows[task] = {
            0,
            blocks,
            lanes,
            active * (sync ? blocks : 1),
            static_cast<uint32_t>(h->cohort_count),
            SCHEDULER_TASK_EXECUTABLE | (active > 1 ? SCHEDULER_TASK_MIX : 0u) |
                (blocks > 1 ? SCHEDULER_TASK_SPMD : 0u) | (sync ? SCHEDULER_TASK_SYNC_START : 0u),
            0
        };
        for (uint64_t i = 0; i < h->cohort_count; ++i)
            *at<SchedulerAsyncCohortCounter>(layout.scratch_offset + h->counters_offset + i * 128) = {};
        maps()[task].block_count = blocks;
        maps()[task].lane_mask = lanes;
        EXPECT_TRUE(AsyncResourceTables::mount(image.data(), layout, &tables));
    }
    uint64_t arrivals(uint32_t index = 0) {
        const auto *h = at<SchedulerAsyncCohortHeader>(layout.scratch_offset);
        return scheduler_gm_fetch_add(
            at<SchedulerAsyncCohortCounter>(
                layout.scratch_offset + h->counters_offset + index * sizeof(SchedulerAsyncCohortCounter)
            )
                ->ready,
            UINT64_C(0)
        );
    }

    AsyncResourceBindings<Reader, Cache> backend(uint32_t core = ASYNC_CORE_AIC) {
        return {tables, core, Reader{&observed}, Cache{&observed}, &proof};
    }
    DeferredCompletionEntry counter(uint64_t offset = 0) const {
        return {record_address() + offset, 0, 1, COMPLETION_ENGINE_SDMA, COMPLETION_TYPE_COUNTER, 0};
    }
    DeferredCompletionEntry sdma(uint64_t post) const {
        return {record_address(), post, 0, COMPLETION_ENGINE_SDMA, COMPLETION_TYPE_SDMA_EVENT_RECORD, 0};
    }
    void use_sdma() {
        for (uint32_t index = 0; index < 2; ++index) {
            auto &use = conditions()[index];
            use.completion_type = COMPLETION_TYPE_SDMA_EVENT_RECORD;
            use.record_range = ASYNC_RESOURCE_NONE;
            use.record_count = use.record_stride = 0;
            use.source_range = 2;
            use.sdma_workspace_address = record_address();
        }
        SchedulerSubmissionHeader h{};
        h.run_epoch = kEpoch;
        h.task_count = 2;
        h.workspace_count = h.bundle_count = h.execution_count = h.request_count = h.grant_count = 1;
        h.provenance_count = 2;
        uint64_t cursor = sizeof(h);
        const auto reserve = [&](uint64_t count, uint64_t stride, uint64_t &offset) {
            EXPECT_TRUE(scheduler_layout_reserve(&cursor, count * stride, 128, &offset));
        };
        reserve(2, sizeof(SchedulerSubmissionTaskMap), h.task_maps_offset);
        reserve(1, sizeof(SchedulerSubmissionWorkspace), h.workspaces_offset);
        reserve(1, sizeof(SchedulerSubmissionPool), h.pools_offset);
        reserve(1, sizeof(SchedulerSubmissionBundleIdentity), h.bundle_identities_offset);
        reserve(1, sizeof(SchedulerAsyncLease), h.bundle_leases_offset);
        reserve(1, sizeof(SchedulerSubmissionProgress), h.bundle_progress_offset);
        reserve(1, sizeof(SchedulerSubmissionExecution), h.executions_offset);
        reserve(1, sizeof(SchedulerSubmissionRequest), h.requests_offset);
        reserve(1, sizeof(SchedulerSubmissionGrant), h.grants_offset);
        reserve(1, sizeof(SchedulerSubmissionReleaseState), h.release_states_offset);
        reserve(1, sizeof(uint64_t), h.execution_grant_indices_offset);
        reserve(1, sizeof(AsyncSubmissionContext), h.contexts_offset);
        reserve(2, sizeof(SchedulerSubmissionProvenance), h.provenance_offset);
        reserve(1, sizeof(SchedulerSubmissionExecutionUseMap), h.execution_use_maps_offset);
        EXPECT_TRUE(scheduler_layout_checked_align(cursor, 128, &h.total_bytes));
        auto *cohort = at<SchedulerAsyncCohortHeader>(layout.scratch_offset);
        cohort->auxiliary_bytes = h.total_bytes;
        cohort->total_bytes = cohort->auxiliary_offset + h.total_bytes;
        layout.scratch_bytes = cohort->total_bytes;
        layout.total_bytes = layout.scratch_offset + layout.scratch_bytes;
        ASSERT_LE(layout.total_bytes, image.size());
        auto *base = image.data() + layout.scratch_offset + cohort->auxiliary_offset;
        *reinterpret_cast<SchedulerSubmissionHeader *>(base) = h;
        SchedulerSubmissionView view{base, h};
        auto *maps = scheduler_submission_detail::at<SchedulerSubmissionTaskMap>(view, h.task_maps_offset);
        maps[0] = {0, 0, 1, 2, SCHEDULER_TASK_EXECUTABLE, 1};
        maps[1] = {};
        *scheduler_submission_detail::at<SchedulerSubmissionWorkspace>(view, h.workspaces_offset) = {
            record_address(),
            records.size(),
            kEpoch,
            1,
            48,
            SchedulerSubmissionPoolQualification::EXCLUSIVE_RUN,
            0,
            {}
        };
        scheduler_gm_store(
            scheduler_submission_detail::at<SchedulerSubmissionPool>(view, h.pools_offset)->busy_groups, UINT64_C(1)
        );
        auto *identity =
            scheduler_submission_detail::at<SchedulerSubmissionBundleIdentity>(view, h.bundle_identities_offset);
        *identity = {};
        identity->run_epoch = kEpoch;
        identity->task_ref = 0;
        identity->execution_count = identity->request_count = 1;
        identity->lane_mask = 2;
        identity->block_count = 1;
        auto *lease = scheduler_submission_detail::at<SchedulerAsyncLease>(view, h.bundle_leases_offset);
        scheduler_gm_store(lease->word, scheduler_async_lease_word(1, SchedulerAsyncPhase::TERMINAL));
        auto *progress = scheduler_submission_detail::at<SchedulerSubmissionProgress>(view, h.bundle_progress_offset);
        *progress = {};
        progress->phase = SchedulerSubmissionProgressPhase::GRANTED;
        progress->request_index = 1;
        *scheduler_submission_detail::at<SchedulerSubmissionExecution>(view, h.executions_offset) = {
            0, 0, 0, 1, 0, 1, CoreAsyncMode::DEFERRED, 2, group.condition_types, 2, 0
        };
        *scheduler_submission_detail::at<SchedulerSubmissionRequest>(view, h.requests_offset) = {0, 0, 1, 1};
        *scheduler_submission_detail::at<SchedulerSubmissionGrant>(view, h.grants_offset) = {0, 0, 0, 0};
        *scheduler_submission_detail::at<SchedulerSubmissionExecutionUseMap>(view, h.execution_use_maps_offset) = {
            0, 1
        };
        EXPECT_TRUE(scheduler_submission_provenance_mount(image.data(), layout, &proof));
        auto *context = scheduler_submission_canonical_context(proof, 0);
        *context = {};
        auto &c = context->header;
        c.run_epoch = kEpoch;
        c.task_token = TaskId::to_uint64(TaskId::make_global(0));
        c.region_address = reinterpret_cast<uint64_t>(image.data());
        c.region_bytes = layout.total_bytes;
        c.submission_layout_offset = proof.async_offset;
        c.execution_index = 0;
        c.use_count = 1;
        c.provenance_offset = proof.async_offset + h.provenance_offset;
        c.provenance_capacity = 2;
        c.dispatch_generation = 9;
        c.frame_address = reinterpret_cast<uint64_t>(&frame);
        c.subtask_slot = 1;
        c.completion_mode = CoreAsyncMode::DEFERRED;
        auto &e = context->evidence;
        e.run_epoch = kEpoch;
        e.task_ref = 0;
        e.execution_index = 0;
        e.dispatch_generation = 9;
        e.subtask_slot = 1;
        e.attempted = e.committed = 1;
    }
    SchedulerSubmissionSealIdentity sealing() const {
        return {kEpoch, 0, 0, 9, reinterpret_cast<uint64_t>(&frame), TaskId::to_uint64(TaskId::make_global(0)),
                0,      1, 0};
    }
    void capture_sdma(const DeferredCompletionEntry &entry, uint32_t condition = 0, uint64_t use_index = 0) {
        *scheduler_submission_detail::at<SchedulerSubmissionProvenance>(
            proof.submissions, proof.submissions.header.provenance_offset, condition
        ) = {1, entry.completion_type, entry.addr, entry.backend_cookie, use_index, 0, 0, 0, 0};
    }
    void seal_and_release() {
        auto *context = scheduler_submission_canonical_context(proof, 0);
        context->evidence.flags = ASYNC_SUBMISSION_FLAGS;
        SchedulerRunControl run{};
        EXPECT_TRUE(
            scheduler_submission_release_all(proof.submissions, {kEpoch, 0, 0, 9, 0, 1}, &context->evidence, &run)
        );
    }
    AsyncConsumerIdentity identity() const { return {kEpoch, 1, 0, 0, 9, 4}; }
};

TEST(HbgAsyncResourceBinding, MountRequiresExactDisjointSectionsAndPreservesOutputOnFailure) {
    for (uint32_t scenario = 0; scenario < 10; ++scenario) {
        Fixture f;
        switch (scenario) {
        case 0:
            f.header().version = ASYNC_RESOURCE_VERSION + 1;
            break;
        case 1:
            ++f.header().run_epoch;
            break;
        case 2:
            f.header().span_count = UINT64_MAX;
            break;
        case 3:
            f.header().execution_contract = UINT64_MAX;
            break;
        case 4:
            f.layout.uses_offset = f.layout.resources_offset;
            break;
        case 5:
            --f.layout.consumers_bytes;
            break;
        case 6:
            --f.layout.resources_bytes;
            break;
        case 7:
            f.layout.total_bytes = f.layout.consumers_offset;
            break;
        case 8:
            f.layout.uses_bytes = 0;
            break;
        case 9:
            ++f.layout.resources_offset;
            break;
        }
        auto previous = f.tables;
        EXPECT_FALSE(AsyncResourceTables::mount(f.image.data(), f.layout, &previous)) << scenario;
        EXPECT_EQ(previous.header.version, ASYNC_RESOURCE_VERSION);
        EXPECT_EQ(previous.header.run_epoch, kEpoch);
    }
    Fixture f;
    EXPECT_FALSE(AsyncResourceTables::mount(nullptr, f.layout, &f.tables));
    EXPECT_FALSE(AsyncResourceTables::mount(f.image.data(), f.layout, nullptr));
}

TEST(HbgAsyncResourceBinding, MissingTableAndWrongExecutionCannotReadACondition) {
    Fixture f;
    AsyncResourceBindings<Reader, Cache> absent({}, ASYNC_CORE_AIC, Reader{&f.observed}, Cache{&f.observed});
    SchedulerAsyncConditionBinding binding{};
    EXPECT_FALSE(absent.bind_condition(f.group, 0, f.counter(), &binding));
    auto backend = f.backend();
    for (uint32_t scenario = 0; scenario < 7; ++scenario) {
        auto group = f.group;
        switch (scenario) {
        case 0:
            ++group.run_epoch;
            break;
        case 1:
            ++group.task_ref;
            break;
        case 2:
            ++group.block_idx;
            break;
        case 3:
            group.subtask_slot = 0;
            break;
        case 4:
            ++group.binding_uses_offset;
            break;
        case 5:
            group.binding_use_count = UINT64_MAX;
            break;
        case 6:
            group.binding_uses_offset = UINT64_MAX;
            break;
        }
        EXPECT_FALSE(backend.bind_condition(group, 0, f.counter(), &binding)) << scenario;
    }
    EXPECT_EQ(f.observed.reads, 0u);
}

TEST(HbgAsyncResourceBinding, ExactRecordSlotsExcludeHolesAndWorkspaceMetadata) {
    Fixture f;
    auto backend = f.backend();
    SchedulerAsyncConditionBinding binding{};
    EXPECT_TRUE(backend.bind_condition(f.group, 0, f.counter(), &binding));
    EXPECT_TRUE(backend.bind_condition(f.group, 0, f.counter(128), &binding));
    for (uint64_t offset : {4, 64, 132, 256, 512})
        EXPECT_FALSE(backend.bind_condition(f.group, 0, f.counter(offset), &binding)) << offset;
    f.use_sdma();
    auto entry = f.sdma((UINT64_C(1) << 50) + 7);
    f.capture_sdma(entry);
    EXPECT_TRUE(backend.bind_condition_at_seal(f.group, 0, entry, f.sealing(), &binding));
    entry.addr += 256;  // In the retained workspace, but not the captured public record.
    EXPECT_FALSE(backend.bind_condition_at_seal(f.group, 0, entry, f.sealing(), &binding));
    EXPECT_EQ(f.observed.reads, 0u);
}

TEST(HbgAsyncResourceBinding, PayloadRelationshipAndWholeCleanLineAreRequired) {
    Fixture f;
    auto backend = f.backend();
    SchedulerAsyncConditionBinding binding{};
    f.conditions()[0].payload_range = ASYNC_RESOURCE_NONE;
    EXPECT_FALSE(backend.bind_condition(f.group, 0, f.counter(), &binding));
    f.conditions()[0].payload_none = 1;
    f.conditions()[0].payload_count = 0;
    EXPECT_TRUE(backend.bind_condition(f.group, 0, f.counter(), &binding));
    f.ranges()[0].access = ASYNC_RANGE_READ;
    EXPECT_FALSE(backend.bind_condition(f.group, 0, f.counter(), &binding));
    f.ranges()[0].access |= ASYNC_RANGE_CONDITION_CLEAN;
    f.spans()[0].address += 4;
    f.spans()[0].bytes -= 4;
    EXPECT_FALSE(backend.bind_condition(f.group, 0, f.counter(4), &binding));
}

TEST(HbgAsyncResourceBinding, AdjacentCleanCounterAndSdmaWordsAreLegal) {
    Fixture f;
    auto backend = f.backend();
    SchedulerAsyncConditionBinding binding{};
    f.conditions()[0].record_stride = 4;
    f.ranges()[0].bytes = 8;
    EXPECT_TRUE(backend.bind_condition(f.group, 0, f.counter(4), &binding));
    EXPECT_FALSE(backend.bind_condition(f.group, 0, f.counter(8), &binding));
    f.use_sdma();
    auto entry = f.sdma(7);
    entry.addr += 8;
    f.capture_sdma(entry);
    EXPECT_TRUE(backend.bind_condition_at_seal(f.group, 0, entry, f.sealing(), &binding));
    entry.addr += 8;
    EXPECT_FALSE(backend.bind_condition_at_seal(f.group, 0, entry, f.sealing(), &binding));
}

TEST(HbgAsyncResourceBinding, ObserverUsesFullPostIdAndNeverAcquiresAnotherCoresPayload) {
    Fixture f;
    f.use_sdma();
    auto backend = f.backend();
    const uint64_t post = (UINT64_C(1) << 50) + 9;
    const auto entry = f.sdma(post);
    SchedulerAsyncConditionBinding binding{};
    f.capture_sdma(entry);
    ASSERT_TRUE(backend.bind_condition_at_seal(f.group, 0, entry, f.sealing(), &binding));
    f.seal_and_release();
    EXPECT_EQ(binding.owner_generation, 0u);
    f.observed.value = post - 1;
    EXPECT_EQ(backend.observe_once(f.group, 0, entry, binding).state, SchedulerAsyncConditionState::PENDING);
    f.observed.value = post;
    EXPECT_EQ(backend.observe_once(f.group, 0, entry, binding).state, SchedulerAsyncConditionState::READY);
    uint64_t stage = 0, offset = 0;
    EXPECT_EQ(
        backend.acquire_completed(f.group, 0, entry, binding, stage, offset).state, SchedulerAsyncConditionState::READY
    );
    EXPECT_EQ(f.observed.reads, 2u);
    EXPECT_EQ(f.observed.fences, 1u);
    EXPECT_TRUE(f.observed.lines.empty());
    binding.owner_generation = kEpoch;
    EXPECT_EQ(backend.observe_once(f.group, 0, entry, binding).state, SchedulerAsyncConditionState::FAILED);
    EXPECT_EQ(f.observed.reads, 2u);
}

TEST(HbgAsyncResourceBinding, UnsealedPublicFactCannotReachCheckAndReleasedGrantIsNoLongerRequired) {
    Fixture f;
    f.use_sdma();
    auto backend = f.backend();
    const auto entry = f.sdma((UINT64_C(1) << 50) + 19);
    f.capture_sdma(entry);
    SchedulerAsyncConditionBinding binding{};
    EXPECT_FALSE(backend.bind_condition(f.group, 0, entry, &binding));
    ASSERT_TRUE(backend.bind_condition_at_seal(f.group, 0, entry, f.sealing(), &binding));
    EXPECT_FALSE(backend.validate(f.group, 0, entry, binding));
    EXPECT_EQ(backend.observe_once(f.group, 0, entry, binding).state, SchedulerAsyncConditionState::FAILED);
    EXPECT_EQ(f.observed.reads, 0u);
    f.seal_and_release();
    EXPECT_TRUE(backend.validate(f.group, 0, entry, binding));
    EXPECT_FALSE(backend.bind_condition_at_seal(f.group, 0, entry, f.sealing(), &binding));
}

TEST(HbgAsyncResourceBinding, SealingRequiresTheActualFrameIdentityAndMatchingWorkspaceForThisUse) {
    Fixture f;
    f.use_sdma();
    auto backend = f.backend();
    const auto entry = f.sdma(8);
    f.capture_sdma(entry);
    SchedulerAsyncConditionBinding binding{};
    for (uint32_t field = 0; field < 5; ++field) {
        auto actual = f.sealing();
        if (field == 0) ++actual.run_epoch;
        if (field == 1) ++actual.dispatch_generation;
        if (field == 2) actual.frame_address += 64;
        if (field == 3) ++actual.task_ref;
        if (field == 4) ++actual.subtask_slot;
        EXPECT_FALSE(backend.bind_condition_at_seal(f.group, 0, entry, actual, &binding));
    }
    ASSERT_TRUE(backend.bind_condition_at_seal(f.group, 0, entry, f.sealing(), &binding));
    f.conditions()[0].sdma_workspace_address = f.payload_address();
    EXPECT_FALSE(backend.bind_condition_at_seal(f.group, 0, entry, f.sealing(), &binding));
    f.seal_and_release();
    EXPECT_FALSE(backend.validate(f.group, 0, entry, {0, 0}));
    f.conditions()[0].sdma_workspace_address = f.record_address();
    ASSERT_TRUE(backend.validate(f.group, 0, entry, {0, 0}));
    f.conditions()[0].record_range = 0;
    f.conditions()[0].record_count = 64;
    f.conditions()[0].record_stride = 8;
    f.ranges()[0].bytes = f.records.size();
    EXPECT_FALSE(backend.validate(f.group, 0, entry, {0, 0}));
    EXPECT_EQ(f.observed.reads, 0u);
}

TEST(HbgAsyncResourceBinding, WritableContextCannotExpandItsImmutableUseOrProvenanceRanges) {
    for (uint32_t field = 0; field < 4; ++field) {
        Fixture f;
        f.use_sdma();
        auto backend = f.backend();
        const auto entry = f.sdma(4);
        f.capture_sdma(entry);
        auto *context = scheduler_submission_canonical_context(f.proof, 0);
        if (field == 0) ++context->header.use_begin;
        if (field == 1) ++context->header.use_count;
        if (field == 2) context->header.provenance_offset = f.proof.submissions.header.pools_offset;
        if (field == 3) context->header.execution_index = 1;
        SchedulerAsyncConditionBinding binding{};
        EXPECT_FALSE(backend.bind_condition_at_seal(f.group, 0, entry, f.sealing(), &binding));
        EXPECT_EQ(f.observed.reads, 0u);
    }
}

TEST(HbgAsyncResourceBinding, StaticCounterPrefixAndDynamicSdmaSuffixBindWithoutRegistrationOrder) {
    Fixture f;
    const auto counter_use = f.conditions()[0];
    f.use_sdma();
    f.conditions()[0] = counter_use;
    f.group.binding_use_count = 2;
    auto &view = f.proof.submissions;
    *scheduler_submission_detail::at<SchedulerSubmissionExecutionUseMap>(
        view, view.header.execution_use_maps_offset
    ) = {0, 2};
    scheduler_submission_canonical_context(f.proof, 0)->header.use_count = 2;
    auto backend = f.backend();
    SchedulerAsyncConditionBinding counter_binding{}, sdma_binding{};
    ASSERT_TRUE(backend.bind_condition(f.group, 0, f.counter(128), &counter_binding));
    EXPECT_EQ(counter_binding.binding_index, 0u);
    const auto entry = f.sdma(23);
    f.capture_sdma(entry, 1, 1);
    ASSERT_TRUE(backend.bind_condition_at_seal(f.group, 1, entry, f.sealing(), &sdma_binding));
    EXPECT_EQ(sdma_binding.binding_index, 1u);
    f.seal_and_release();
    EXPECT_TRUE(backend.validate(f.group, 1, entry, sdma_binding));
    EXPECT_FALSE(backend.validate(f.group, 1, entry, counter_binding));
    EXPECT_TRUE(backend.validate(f.group, 0, f.counter(128), counter_binding));
}

TEST(HbgAsyncResourceBinding, ObserverCoreAndSideReferenceCannotBeSubstituted) {
    Fixture f;
    auto backend = f.backend();
    SchedulerAsyncConditionBinding binding{};
    ASSERT_TRUE(backend.bind_condition(f.group, 0, f.counter(), &binding));
    ++binding.binding_index;
    EXPECT_FALSE(backend.validate(f.group, 0, f.counter(), binding));
    f.conditions()[0].observer_cores = ASYNC_CORE_AIV;
    EXPECT_FALSE(backend.bind_condition(f.group, 0, f.counter(), &binding));
    EXPECT_TRUE(f.backend(ASYNC_CORE_AIV).bind_condition(f.group, 0, f.counter(), &binding));
    EXPECT_FALSE(f.backend(0).bind_condition(f.group, 0, f.counter(), &binding));
}

TEST(HbgAsyncResourceBinding, PreSubmitValidatesTransferAndReservesBeforeAnyBackendPost) {
    Fixture f;
    f.use_sdma();
    auto backend = f.backend(ASYNC_CORE_AIV);
    uint32_t count = 0;
    int32_t error = 0;
    DeferredCompletionEntry entries[2]{};
    AsyncCtx ctx{};
    ctx.completion_count = &count;
    ctx.completion_error_code = &error;
    ctx.completion_entries = entries;
    ctx.completion_capacity = 2;
    TaskId::assign(ctx.task_token, TaskId::make_global(0));
    AsyncCompletionAllowance allowance{};
    EXPECT_FALSE(f.backend(ASYNC_CORE_AIC)
                     .prepare_sdma_submit(
                         f.group, ctx, f.payload_address() + 512, f.payload_address() + 64, 128, 2, &allowance,
                         ASYNC_RESOURCE_NONE, f.record_address()
                     ));
    EXPECT_EQ(error, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
    error = 0;
    EXPECT_FALSE(backend.prepare_sdma_submit(
        f.group, ctx, f.payload_address() + 512, f.payload_address() + 64, 129, 2, &allowance, ASYNC_RESOURCE_NONE,
        f.record_address()
    ));
    EXPECT_FALSE(backend.prepare_sdma_submit(
        f.group, ctx, f.payload_address() + 511, f.payload_address() + 64, 128, 2, &allowance, ASYNC_RESOURCE_NONE,
        f.record_address()
    ));
    EXPECT_FALSE(backend.prepare_sdma_submit(
        f.group, ctx, f.payload_address() + 512, f.payload_address() + 64, 128, 3, &allowance, ASYNC_RESOURCE_NONE,
        f.record_address()
    ));
    EXPECT_FALSE(allowance.valid);
    EXPECT_EQ(count, 0u);
    EXPECT_EQ(error, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
    // A fresh execution has a fresh writer; an errored execution cannot retry
    // itself into a successful zero-condition seal.
    error = 0;
    ASSERT_TRUE(backend.prepare_sdma_submit(
        f.group, ctx, f.payload_address() + 512, f.payload_address() + 64, 128, 2, &allowance, ASYNC_RESOURCE_NONE,
        f.record_address()
    ));
    EXPECT_TRUE(allowance.valid);
    EXPECT_EQ(allowance.limit, 2u);
    EXPECT_EQ(count, 0u);
    EXPECT_EQ(f.observed.reads, 0u);
    f.ranges()[1].access &= ~ASYNC_RANGE_WRITE;
    EXPECT_FALSE(backend.prepare_sdma_submit(
        f.group, ctx, f.payload_address() + 512, f.payload_address() + 64, 128, 2, &allowance, ASYNC_RESOURCE_NONE,
        f.record_address()
    ));
    EXPECT_FALSE(allowance.valid);
    EXPECT_EQ(error, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
}

TEST(HbgAsyncResourceBinding, ConsumerAcquiresItsOwnRangesWithABoundedContinuation) {
    Fixture f;
    auto backend = f.backend();
    AsyncConsumerCursor cursor;
    uint32_t budget = 0;
    EXPECT_EQ(backend.prepare_consumer(f.identity(), cursor, budget), AsyncConsumerResult::PENDING);
    EXPECT_TRUE(f.observed.lines.empty());
    budget = 1;
    EXPECT_EQ(backend.prepare_consumer(f.identity(), cursor, budget), AsyncConsumerResult::PENDING);
    EXPECT_EQ(budget, 0u);
    EXPECT_EQ(f.observed.lines, std::vector<uint64_t>({f.payload_address() + 64}));
    budget = 8;
    EXPECT_EQ(backend.prepare_consumer(f.identity(), cursor, budget), AsyncConsumerResult::READY);
    EXPECT_EQ(budget, 6u);
    EXPECT_EQ(
        f.observed.lines,
        std::vector<uint64_t>({f.payload_address() + 64, f.payload_address() + 128, f.payload_address() + 1024})
    );
    EXPECT_EQ(f.observed.reads, 0u);
    EXPECT_EQ(backend.prepare_consumer(f.identity(), cursor, budget), AsyncConsumerResult::READY);
    EXPECT_EQ(f.observed.lines.size(), 3u);
}

TEST(HbgAsyncResourceBinding, InterveningKernelRestartsPartialAndCompletedAcquisition) {
    Fixture f;
    auto backend = f.backend();
    AsyncConsumerCursor cursor;
    auto identity = f.identity();
    uint32_t budget = 1;
    EXPECT_EQ(backend.prepare_consumer(identity, cursor, budget), AsyncConsumerResult::PENDING);
    ++identity.core_write_serial;
    budget = 8;
    EXPECT_EQ(backend.prepare_consumer(identity, cursor, budget), AsyncConsumerResult::READY);
    EXPECT_EQ(f.observed.lines.size(), 4u);
    EXPECT_EQ(f.observed.lines[0], f.observed.lines[1]);
    ++identity.core_write_serial;
    EXPECT_EQ(backend.prepare_consumer(identity, cursor, budget), AsyncConsumerResult::READY);
    EXPECT_EQ(f.observed.lines.size(), 7u);
    ++identity.dispatch_generation;
    budget = 8;
    EXPECT_EQ(backend.prepare_consumer(identity, cursor, budget), AsyncConsumerResult::READY);
    EXPECT_EQ(f.observed.lines.size(), 10u);
}

TEST(HbgAsyncResourceBinding, UnknownConsumerAndUncleanOrOutOfBoundsPayloadFailClosed) {
    for (uint32_t scenario = 0; scenario < 7; ++scenario) {
        Fixture f;
        auto backend = f.backend();
        AsyncConsumerCursor cursor;
        auto identity = f.identity();
        switch (scenario) {
        case 0:
            f.maps()[1].kind = AsyncConsumerMapKind::UNKNOWN;
            break;
        case 1:
            f.ranges()[1].access &= ~ASYNC_RANGE_CONSUMER_CLEAN;
            break;
        case 2:
            f.ranges()[1].byte_offset = UINT64_MAX;
            break;
        case 3:
            f.consumer().range_begin = UINT64_MAX;
            break;
        case 4:
            ++identity.run_epoch;
            break;
        case 5:
            identity.dispatch_generation = 0;
            break;
        case 6:
            identity.core_write_serial = UINT64_MAX;
            break;
        }
        uint32_t budget = 8;
        EXPECT_EQ(backend.prepare_consumer(identity, cursor, budget), AsyncConsumerResult::ERROR) << scenario;
        EXPECT_TRUE(f.observed.lines.empty());
    }
}

TEST(HbgAsyncResourceBinding, ExplicitNoneAndLargeRangeDoNotBypassTheWorkLimit) {
    Fixture f;
    auto backend = f.backend();
    AsyncConsumerCursor cursor;
    uint32_t budget = 100;
    auto identity = f.identity();
    identity.task_ref = 0;
    identity.subtask_slot = 1;
    EXPECT_EQ(backend.prepare_consumer(identity, cursor, budget), AsyncConsumerResult::READY);
    EXPECT_EQ(budget, 100u);
    f.ranges()[1].bytes = 1024;
    EXPECT_EQ(backend.prepare_consumer(f.identity(), cursor, budget), AsyncConsumerResult::PENDING);
    EXPECT_EQ(budget, 92u);
    EXPECT_EQ(f.observed.lines.size(), 8u);
}
TEST(HbgAsyncResourceBinding, CohortIncludesExplicitNoneAndCountsEachCursorOnce) {
    Fixture f;
    f.enable_cohort(0, 6, 1, false);
    auto backend = f.backend(ASYNC_CORE_AIV);
    AsyncConsumerIdentity a{kEpoch, 0, 0, 1, 1, 0};
    AsyncConsumerIdentity b{kEpoch, 0, 0, 2, 1, 0};
    AsyncConsumerCursor first, second;
    uint32_t budget = 0;
    EXPECT_EQ(backend.prepare_consumer(a, first, budget), AsyncConsumerResult::PENDING);
    EXPECT_FALSE(backend.consumer_pinned(first));
    EXPECT_EQ(f.arrivals(), 0u);
    budget = 8;
    EXPECT_EQ(backend.prepare_consumer(a, first, budget), AsyncConsumerResult::PENDING);
    EXPECT_TRUE(backend.consumer_pinned(first));
    EXPECT_EQ(f.arrivals(), 1u);
    EXPECT_EQ(backend.prepare_consumer(a, first, budget), AsyncConsumerResult::PENDING);
    EXPECT_EQ(f.arrivals(), 1u);
    EXPECT_EQ(backend.prepare_consumer(b, second, budget), AsyncConsumerResult::READY);
    EXPECT_EQ(backend.prepare_consumer(a, first, budget), AsyncConsumerResult::READY);
    EXPECT_EQ(f.arrivals(), 2u);
    ++a.core_write_serial;
    EXPECT_EQ(backend.prepare_consumer(a, first, budget), AsyncConsumerResult::ERROR);
    EXPECT_EQ(f.arrivals(), 2u);
}

TEST(HbgAsyncResourceBinding, MixBlocksHaveIndependentCohortsButSyncStartUsesTheWholeTask) {
    for (bool sync : {false, true}) {
        Fixture f;
        f.enable_cohort(0, 6, 2, sync);
        auto backend = f.backend(ASYNC_CORE_AIV);
        AsyncConsumerIdentity ids[]{
            {kEpoch, 0, 0, 1, 1, 0}, {kEpoch, 0, 1, 2, 1, 0}, {kEpoch, 0, 0, 2, 1, 0}, {kEpoch, 0, 1, 1, 1, 0}
        };
        AsyncConsumerCursor cursors[4];
        uint32_t budget = 8;
        EXPECT_EQ(backend.prepare_consumer(ids[0], cursors[0], budget), AsyncConsumerResult::PENDING);
        EXPECT_EQ(backend.prepare_consumer(ids[1], cursors[1], budget), AsyncConsumerResult::PENDING);
        EXPECT_EQ(
            backend.prepare_consumer(ids[2], cursors[2], budget),
            sync ? AsyncConsumerResult::PENDING : AsyncConsumerResult::READY
        );
        EXPECT_EQ(backend.prepare_consumer(ids[3], cursors[3], budget), AsyncConsumerResult::READY);
        EXPECT_EQ(f.arrivals(), sync ? 4u : 2u);
        if (!sync) EXPECT_EQ(f.arrivals(1), 2u);
    }
}

TEST(HbgAsyncResourceBinding, EightAcquireLinesConsumeTheSliceBeforeAtomicArrival) {
    Fixture f;
    f.header().consumer_count = 2;
    f.layout.consumers_bytes += sizeof(AsyncConsumerContract);
    f.layout.scratch_offset = 1024;
    f.consumer().range_count = 1;
    (&f.consumer())[1] = f.consumer();
    (&f.consumer())[1].block_idx = 1;
    f.consumer_ranges()[0] = 1;
    f.consumer_ranges()[1] = 3;
    f.ranges()[1].bytes = 512;
    f.enable_cohort(1, 1, 2, true);
    auto backend = f.backend();
    AsyncConsumerCursor cursor;
    uint32_t budget = 100;
    EXPECT_EQ(backend.prepare_consumer(f.identity(), cursor, budget), AsyncConsumerResult::PENDING);
    EXPECT_EQ(budget, 92u);
    EXPECT_EQ(f.observed.lines.size(), 8u);
    EXPECT_EQ(f.arrivals(), 0u);
    EXPECT_FALSE(backend.consumer_pinned(cursor));
    budget = 8;
    EXPECT_EQ(backend.prepare_consumer(f.identity(), cursor, budget), AsyncConsumerResult::PENDING);
    EXPECT_EQ(budget, 7u);
    EXPECT_EQ(f.observed.lines.size(), 8u);
    EXPECT_EQ(f.arrivals(), 1u);
}

TEST(HbgAsyncResourceBinding, CohortCorruptionIsRejectedBeforeAContribution) {
    for (uint32_t scenario = 0; scenario < 4; ++scenario) {
        Fixture f;
        f.enable_cohort(0, 6, 1, false);
        auto *h = f.at<SchedulerAsyncCohortHeader>(f.layout.scratch_offset);
        auto *map = f.at<SchedulerAsyncCohortTaskMap>(f.layout.scratch_offset + h->maps_offset);
        if (scenario == 0) ++map->expected;
        if (scenario == 1) ++map->cohort_base;
        if (scenario == 2) map->flags = SCHEDULER_TASK_EXECUTABLE;
        if (scenario == 3) map->reserved = 1;
        auto backend = f.backend(ASYNC_CORE_AIV);
        AsyncConsumerCursor cursor;
        uint32_t budget = 8;
        EXPECT_EQ(backend.prepare_consumer({kEpoch, 0, 0, 1, 1, 0}, cursor, budget), AsyncConsumerResult::ERROR);
        EXPECT_EQ(f.arrivals(), 0u);
    }
}

}  // namespace
