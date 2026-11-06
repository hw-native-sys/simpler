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
#include <array>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include "backend/sdma/sdma_qualified_submission.h"
#include "async_plan.h"
#include "submission_bundle_plan.h"

namespace {
using namespace pto2::detail;
constexpr SdmaMetadataLayout kLayout{64, 64, 4};

struct Ranges {
    AsyncSubmissionContextHeader header{};
    std::array<AsyncRetainedSpan, 2> owners{{{0x20000, 0x1000}, {0x30000, 0x1000}}};
    std::array<AsyncResourceRange, 2> ranges{
        {{0, 64, 128, ASYNC_RANGE_READ}, {1, 128, 128, ASYNC_RANGE_READ | ASYNC_RANGE_WRITE}}
    };
    std::array<AsyncConditionContract, 3> uses{};
    AsyncResourceTables tables{};
    Ranges() {
        header.run_epoch = 19;
        header.block_idx = 2;
        header.subtask_slot = 1;
        header.use_begin = 0;
        header.use_count = 2;
        tables.header.run_epoch = 19;
        tables.header.span_count = 2;
        tables.header.range_count = 2;
        tables.header.condition_count = uses.size();
        tables.spans = owners.data();
        tables.ranges = ranges.data();
        tables.conditions = uses.data();
        for (uint32_t i = 0; i < uses.size(); ++i) {
            auto &use = uses[i];
            use.task_ref = 3;
            use.block_idx = 2;
            use.subtask_slot = 1;
            use.use_index = i;
            use.observer_cores = 3;
            use.completion_type = COMPLETION_TYPE_SDMA_EVENT_RECORD;
            use.source_range = 0;
            use.payload_range = 1;
            use.sdma_workspace_address = 0x10000;
        }
        uses[0].completion_type = COMPLETION_TYPE_COUNTER;
        uses[0].sdma_workspace_address = 0;
    }
    bool select(uint64_t *index, uint64_t src = 0x20050, uint64_t dst = 0x30090, uint64_t bytes = 32) {
        return sdma_qualified_select_use(header, 3, tables, 0x10000, src, dst, bytes, index);
    }
};

TEST(HbgSdmaQualifiedSubmission, ScratchRejectsNullMisalignmentTruncationAndInvalidSyncId) {
    EXPECT_TRUE(sdma_qualified_scratch_valid(256, 32, 256, 0));
    EXPECT_TRUE(sdma_qualified_scratch_valid(512, 8, 256, 7));
    EXPECT_FALSE(sdma_qualified_scratch_valid(0, 32, 256, 0));
    EXPECT_FALSE(sdma_qualified_scratch_valid(32, 32, 256, 0));
    EXPECT_FALSE(sdma_qualified_scratch_valid(256, 7, 256, 0));
    EXPECT_FALSE(sdma_qualified_scratch_valid(256, UINT64_C(1) << 32, 256, 0));
    EXPECT_FALSE(sdma_qualified_scratch_valid(256, 32, 256, 8));
    EXPECT_FALSE(sdma_qualified_scratch_valid(256, 32, 0, 0));
}

TEST(HbgSdmaQualifiedSubmission, UniqueAllowedUseContainsSmallTransferAfterCounterPrefix) {
    Ranges r;
    uint64_t selected = UINT64_MAX;
    ASSERT_TRUE(r.select(&selected));
    EXPECT_EQ(selected, 1u);
    EXPECT_TRUE(r.select(&selected, 0x20040, 0x30080, 128));
    EXPECT_TRUE(r.select(&selected, 0x200c0, 0x30100, 0));
    EXPECT_FALSE(r.select(&selected, 0x200c0, 0x30100, 1));
    EXPECT_FALSE(r.select(&selected, 0x20000));
    EXPECT_FALSE(r.select(&selected, 0x20050, UINT64_MAX - 7, 32));
}

TEST(HbgSdmaQualifiedSubmission, AmbiguousUsesAndWrongIdentityNeverGuessAnOwner) {
    for (uint32_t change = 0; change < 10; ++change) {
        Ranges r;
        if (change == 0) r.header.use_count = 3;
        if (change == 1) r.uses[1].sdma_workspace_address++;
        if (change == 2) r.uses[1].task_ref++;
        if (change == 3) r.uses[1].block_idx++;
        if (change == 4) r.uses[1].use_index++;
        if (change == 5) r.uses[1].record_range = 0;
        if (change == 6) r.uses[1].payload_count = 2;
        if (change == 7) r.ranges[1].access = ASYNC_RANGE_READ;
        if (change == 8) r.owners[1].bytes = 140;
        if (change == 9) r.header.use_begin = UINT64_MAX;
        uint64_t selected = UINT64_MAX;
        EXPECT_FALSE(r.select(&selected)) << change;
        EXPECT_EQ(selected, SCHEDULER_ASYNC_INDEX_NONE);
    }
}

struct EventCache {
    std::vector<uint64_t> events;
    void flush_range(volatile void *address, uint32_t) { events.push_back(reinterpret_cast<uint64_t>(address)); }
    void finish() { events.push_back(0); }
};
struct Driver {
    struct alignas(128) Image {
        uint8_t prefix[256]{};
        AsyncSubmissionContext context{};
    } image;
    DispatchPayload payload{};
    alignas(64) std::array<uint8_t, 64 * 64> workspace{};
    std::array<DeferredCompletionEntry, 4> entries{};
    std::array<SchedulerSubmissionProvenance, 4> provenance{};
    uint32_t count{1};
    int32_t error{0};
    AsyncCtx writer{};
    SdmaQualifiedAccess access{};
    EventCache cache;
    std::vector<SdmaCompletionReceiptRecord> receipts;
    uint32_t built{0}, posted{0}, waited{0};
    int failure{0};
    struct Backend {
        Driver &d;
        bool preflight() const { return d.failure != 1; }
        bool build_session(uint32_t queues, uint32_t group) {
            ++d.built;
            EXPECT_EQ(queues, 2u);
            EXPECT_EQ(group, 3u);
            EXPECT_EQ(d.ctx().evidence.attempted, d.ctx().evidence.committed + 1);
            return d.failure != 2;
        }
        SdmaMetadataSession metadata_session() const {
            return {d.access.workspace.address, d.failure == 3 ? 4u : 3u, 2, true};
        }
        bool submit() {
            ++d.posted;
            return d.failure != 4;
        }
        bool event_valid(uint64_t) const { return d.failure != 5; }
        uint32_t record_count() const { return d.failure == 6 ? 0 : d.failure == 7 ? 3 : 2; }
        bool record(uint32_t index, SdmaCompletionReceiptRecord *out) const {
            *out = {d.access.workspace.address + (56 + (d.failure == 9 ? 0 : index)) * 64, UINT64_MAX - index};
            if (d.failure == 10) out->address = UINT64_MAX - 7;
            if (d.failure == 11) out->expected = 0;
            return d.failure != 8;
        }
        bool wait() {
            ++d.waited;
            return d.failure != 12;
        }
    } backend{*this};
    Driver() {
        auto &h = ctx().header;
        h.run_epoch = 19;
        h.region_address = reinterpret_cast<uint64_t>(&image);
        h.region_bytes = sizeof(image);
        h.submission_layout_offset = 128;
        h.execution_index = 7;
        h.dispatch_generation = 11;
        h.frame_address = reinterpret_cast<uint64_t>(&payload);
        h.task_token = TaskId::to_uint64(TaskId::make_global(3));
        h.block_idx = 2;
        h.subtask_slot = 1;
        h.completion_mode = CoreAsyncMode::DEFERRED;
        h.use_begin = 5;
        h.use_count = 3;
        auto &e = ctx().evidence;
        e.run_epoch = 19;
        e.task_ref = 3;
        e.execution_index = 7;
        e.dispatch_generation = 11;
        e.block_idx = 2;
        e.subtask_slot = 1;
        writer.task_token = TaskId::make_global(3);
        writer.completion_count = &count;
        writer.completion_error_code = &error;
        writer.completion_entries = entries.data();
        writer.completion_capacity = entries.size();
        entries[0].completion_type = COMPLETION_TYPE_COUNTER;
        access.identity = {19, 3, 7, 11, 2, 1};
        access.workspace = {
            reinterpret_cast<uint64_t>(workspace.data()),
            workspace.size(),
            19,
            2,
            48,
            SchedulerSubmissionPoolQualification::EXCLUSIVE_RUN,
            0,
            {}
        };
        access.grant = {7, 5, 3, 0};
        access.workspace_index = 5;
        access.use_index = 7;
        access.provenance = provenance.data();
        access.provenance_capacity = 4;
    }
    AsyncSubmissionContext &ctx() { return image.context; }
    bool submit(uint64_t bytes = 32) {
        auto sink = [&](uint32_t index, uint32_t total, const SdmaCompletionReceiptRecord &record) {
            EXPECT_EQ(index, receipts.size());
            EXPECT_EQ(total, 2u);
            EXPECT_EQ(ctx().evidence.attempted, ctx().evidence.committed);
            EXPECT_EQ(ctx().evidence.flags, ASYNC_SUBMISSION_METADATA_PUBLISHED);
            receipts.push_back(record);
        };
        return sdma_qualified_submit(ctx(), writer, access, bytes, kLayout, backend, cache, sink);
    }
    void synchronous() {
        ctx().header.completion_mode = CoreAsyncMode::SYNCHRONOUS;
        ctx().header.task_token = TaskId::to_uint64(TaskId::invalid());
        writer = {};
        writer.task_token = TaskId::invalid();
        access.provenance = nullptr;
        access.provenance_capacity = 0;
    }
};

TEST(HbgSdmaQualifiedSubmission, ActualWriterSlotsKeepCounterAndFullWidthPublicCookie) {
    Driver d;
    ASSERT_TRUE(d.submit());
    EXPECT_EQ(d.count, 3u);
    EXPECT_EQ(d.entries[0].completion_type, COMPLETION_TYPE_COUNTER);
    EXPECT_EQ(d.provenance[0].version, 0u);
    ASSERT_EQ(d.receipts.size(), 2u);
    for (uint32_t i = 0; i < 2; ++i) {
        EXPECT_EQ(d.entries[1 + i].backend_cookie, UINT64_MAX - i);
        EXPECT_EQ(d.provenance[1 + i].backend_cookie, UINT64_MAX - i);
        EXPECT_EQ(d.provenance[1 + i].use_index, 7u);
        EXPECT_EQ(d.provenance[1 + i].workspace_index, 5u);
        EXPECT_EQ(d.provenance[1 + i].execution_index, 7u);
        EXPECT_EQ(d.receipts[i].address, d.entries[1 + i].addr);
    }
    EXPECT_EQ(d.ctx().evidence.committed, 1u);
    EXPECT_EQ(d.waited, 0u);
    const uint64_t meta = d.access.workspace.address + (1 + 3 * 2) * 64;
    const auto m = std::find(d.cache.events.begin(), d.cache.events.end(), meta);
    const auto e =
        std::find(d.cache.events.begin(), d.cache.events.end(), reinterpret_cast<uint64_t>(&d.ctx().evidence));
    ASSERT_NE(m, d.cache.events.end());
    ASSERT_NE(e, d.cache.events.end());
    EXPECT_LT(m, e);
}

TEST(HbgSdmaQualifiedSubmission, CapacityAndPreflightRefuseBeforeSessionSideEffects) {
    for (uint32_t change = 0; change < 7; ++change) {
        Driver d;
        if (change == 0) d.count = 3;
        if (change == 1) d.failure = 1;
        if (change == 2) d.access.grant.group_index = 24;
        if (change == 3) d.access.workspace.qualification_epoch++;
        if (change == 4) d.access.identity.dispatch_generation++;
        if (change == 5) d.access.provenance_capacity = 3;
        if (change == 6) d.ctx().evidence.flags |= ASYNC_SUBMISSION_SEALED;
        EXPECT_FALSE(d.submit()) << change;
        EXPECT_EQ(d.built, 0u);
        EXPECT_EQ(d.posted, 0u);
        EXPECT_EQ(d.ctx().evidence.attempted, 0u);
        EXPECT_TRUE(d.receipts.empty());
    }
}

TEST(HbgSdmaQualifiedSubmission, FailedSessionSubmitOrPublicRecordsPreserveAttemptDebt) {
    for (int failure = 2; failure <= 11; ++failure) {
        Driver d;
        d.failure = failure;
        EXPECT_FALSE(d.submit()) << failure;
        EXPECT_EQ(d.ctx().evidence.attempted, 1u);
        EXPECT_EQ(d.ctx().evidence.committed, 0u);
        EXPECT_EQ(d.ctx().evidence.flags, 0u);
        EXPECT_EQ(d.count, 1u);
        EXPECT_NE(d.error, 0);
        EXPECT_TRUE(d.receipts.empty());
        EXPECT_FALSE(d.submit());
        EXPECT_EQ(d.built, 1u);
    }
}

TEST(HbgSdmaQualifiedSubmission, EmptyTransferHasNoBackendAttemptOrReceipt) {
    Driver d;
    ASSERT_TRUE(d.submit(0));
    EXPECT_EQ(d.built, 0u);
    EXPECT_EQ(d.posted, 0u);
    EXPECT_EQ(d.ctx().evidence.attempted, 0u);
    EXPECT_EQ(d.count, 1u);
    EXPECT_TRUE(d.receipts.empty());
    EXPECT_TRUE(d.cache.events.empty());
}

TEST(HbgSdmaQualifiedSubmission, SynchronousUseStillWaitsAndPublishesMetadata) {
    Driver d;
    d.synchronous();
    ASSERT_TRUE(d.submit());
    EXPECT_EQ(d.waited, 1u);
    EXPECT_EQ(d.ctx().evidence.committed, 1u);
    EXPECT_EQ(d.count, 1u);
    EXPECT_EQ(d.receipts.size(), 2u);
    Driver failed;
    failed.synchronous();
    failed.failure = 12;
    EXPECT_FALSE(failed.submit());
    EXPECT_EQ(failed.waited, 1u);
    EXPECT_EQ(failed.ctx().evidence.attempted, 1u);
    EXPECT_EQ(failed.ctx().evidence.committed, 0u);
    EXPECT_TRUE(failed.receipts.empty());
}

TEST(HbgSdmaQualifiedSubmission, StickyNativeErrorNeverBecomesSuccessfulRetry) {
    Driver d;
    d.ctx().evidence.error_code = -37;
    EXPECT_FALSE(d.submit());
    EXPECT_EQ(d.ctx().evidence.error_code, -37);
    EXPECT_EQ(d.built, 0u);
    EXPECT_TRUE(d.receipts.empty());
}

struct Mounted {
    std::unique_ptr<void, decltype(&std::free)> image{nullptr, &std::free};
    alignas(64) std::array<uint8_t, 64 * 64> workspace{};
    DispatchPayload payload{};
    SchedulerAsyncHostPlan async_plan{};
    SchedulerSubmissionHostPlan plan{};
    SchedulerSubmissionProvenanceView proof{};
    AsyncSubmissionContext *context{nullptr};
    SchedulerSubmissionExecution execution{};
    Ranges resource;
    Mounted() {
        SchedulerTaskMetadata task{};
        task.active_mask = LaneMask::aiv0();
        task.logical_block_num = 1;
        task.flags = SCHEDULER_TASK_EXECUTABLE;
        task.kernel_ids[1] = 1;
        SchedulerAsyncCapabilities caps{};
        for (auto &cap : caps)
            cap = {CORE_ASYNC_CAPABILITY_VERSION, CoreAsyncMode::SYNCHRONOUS, 0, 0};
        caps[1] = {CORE_ASYNC_CAPABILITY_VERSION, CoreAsyncMode::DEFERRED, 4, CORE_ASYNC_CONDITION_TYPES};
        SchedulerSubmissionWorkspaceInput owner{
            reinterpret_cast<uint64_t>(workspace.data()),       workspace.size(), 19, 2, 48,
            SchedulerSubmissionPoolQualification::EXCLUSIVE_RUN
        };
        std::string error;
        if (!scheduler_plan_submission_bundles(
                {task}, {caps}, {owner}, {{0, 0, 1, 0, 0}}, {{0, 0, 1, 0, 2}}, 19, 1 << 20, &plan, &error
            ))
            throw std::runtime_error(error);
        SchedulerAsyncAuxiliaryBytes extra{};
        extra.resources = sizeof(AsyncResourceHeader) + 2 * sizeof(AsyncRetainedSpan) + 2 * sizeof(AsyncResourceRange);
        extra.uses = 2 * sizeof(AsyncConditionContract);
        extra.consumers = sizeof(AsyncConsumerTaskMap);
        extra.scratch = plan.image.size();
        if (!scheduler_plan_async_state({task}, {caps}, 19, 1 << 20, extra, &async_plan, &error))
            throw std::runtime_error(error);
        image.reset(std::aligned_alloc(128, async_plan.image.size()));
        if (!image) throw std::bad_alloc();
        std::memcpy(image.get(), async_plan.image.data(), async_plan.image.size());
        auto *base = static_cast<uint8_t *>(image.get());
        const auto &layout = async_plan.layout;
        SchedulerAsyncCohortView cohort{};
        if (!SchedulerAsyncCohortView::mount(base, layout, &cohort)) throw std::runtime_error("cohort mount");
        std::memcpy(cohort.base + cohort.header.auxiliary_offset, plan.image.data(), plan.image.size());
        if (!scheduler_submission_provenance_mount(base, layout, &proof)) throw std::runtime_error("proof mount");
        if (!scheduler_submission_detail::execution(proof.submissions, 0, &execution))
            throw std::runtime_error("execution");
        resource.tables.header.condition_count = 2;
        std::memcpy(base + layout.resources_offset, &resource.tables.header, sizeof(AsyncResourceHeader));
        std::memcpy(
            base + layout.resources_offset + sizeof(AsyncResourceHeader), resource.owners.data(),
            sizeof(resource.owners)
        );
        std::memcpy(
            base + layout.resources_offset + sizeof(AsyncResourceHeader) + sizeof(resource.owners),
            resource.ranges.data(), sizeof(resource.ranges)
        );
        for (uint32_t i = 0; i < 2; ++i) {
            resource.uses[i].task_ref = 0;
            resource.uses[i].block_idx = 0;
            if (i == 1) resource.uses[i].sdma_workspace_address = owner.address;
        }
        std::memcpy(base + layout.uses_offset, resource.uses.data(), extra.uses);
        auto *group = reinterpret_cast<SchedulerAsyncGroupIdentity *>(base + layout.group_identity_offset);
        group->binding_uses_offset = layout.uses_offset;
        group->binding_use_count = 2;
        auto *writer = reinterpret_cast<SchedulerAsyncGroupWriter *>(base + layout.group_writer_offset);
        writer->count = 1;
        auto &w = payload.local_context.async_ctx;
        w.task_token = TaskId::make_global(0);
        w.completion_count = &writer->count;
        w.completion_error_code = &writer->error_code;
        w.completion_entries = reinterpret_cast<DeferredCompletionEntry *>(base + group->entries_offset);
        w.completion_capacity = 4;
        w.completion_entries[0].completion_type = COMPLETION_TYPE_COUNTER;
        payload.args[SPMD_LOCAL_CONTEXT_INDEX] = reinterpret_cast<uint64_t>(&payload.local_context);
        payload.args[SPMD_GLOBAL_CONTEXT_INDEX] = reinterpret_cast<uint64_t>(&payload.global_context);
        context = scheduler_submission_canonical_context(proof, 0);
        auto &h = context->header;
        h.run_epoch = 19;
        h.task_token = TaskId::to_uint64(w.task_token);
        h.region_address = reinterpret_cast<uint64_t>(base);
        h.region_bytes = layout.total_bytes;
        h.submission_layout_offset = proof.async_offset;
        h.execution_index = 0;
        h.use_begin = 0;
        h.use_count = 2;
        h.provenance_offset = proof.async_offset + proof.submissions.header.provenance_offset;
        h.provenance_capacity = 4;
        h.dispatch_generation = 11;
        h.frame_address = reinterpret_cast<uint64_t>(&payload);
        h.subtask_slot = 1;
        h.completion_mode = CoreAsyncMode::DEFERRED;
        auto &e = context->evidence;
        e.run_epoch = 19;
        e.task_ref = 0;
        e.execution_index = 0;
        e.dispatch_generation = 11;
        e.subtask_slot = 1;
        payload.async_submission_context = reinterpret_cast<uint64_t>(context);
        using namespace scheduler_submission_detail;
        auto *lease = at<SchedulerAsyncLease>(proof.submissions, proof.submissions.header.bundle_leases_offset);
        lease->word = scheduler_async_lease_word(1, SchedulerAsyncPhase::TERMINAL);
        auto *progress =
            at<SchedulerSubmissionProgress>(proof.submissions, proof.submissions.header.bundle_progress_offset);
        progress->phase = SchedulerSubmissionProgressPhase::GRANTED;
        auto *grant = at<SchedulerSubmissionGrant>(proof.submissions, proof.submissions.header.grants_offset);
        grant->group_index = 3;
    }
    int64_t *args() { return reinterpret_cast<int64_t *>(payload.args); }
    bool access(SdmaQualifiedAccess *out, AsyncCtx *writer, AsyncSubmissionContext **ctx) {
        return sdma_qualified_access(
            args(), reinterpret_cast<uint64_t>(workspace.data()), 0x20050, 0x30090, 32, ctx, writer, out
        );
    }
};

TEST(HbgSdmaQualifiedSubmission, RealRetainedImageMountsCanonicalWriterAndActiveGrant) {
    Mounted m;
    SdmaQualifiedAccess access;
    AsyncCtx writer;
    AsyncSubmissionContext *context = nullptr;
    ASSERT_TRUE(m.access(&access, &writer, &context));
    EXPECT_EQ(context, m.context);
    EXPECT_EQ(access.grant.group_index, 3u);
    EXPECT_EQ(access.use_index, 1u);
    EXPECT_EQ(access.identity.task_ref, 0);
    EXPECT_EQ(writer.completion_count, m.payload.local_context.async_ctx.completion_count);
    EXPECT_EQ(*writer.completion_count, 1u);
    EXPECT_EQ(access.provenance_capacity, 4u);
}

TEST(HbgSdmaQualifiedSubmission, RealRetainedImageRejectsStaleFrameReleasedGrantAndForeignWriter) {
    for (uint32_t change = 0; change < 10; ++change) {
        Mounted m;
        uint32_t foreign_count = 0;
        if (change == 0) m.payload.async_submission_context = 0;
        if (change == 1) m.payload.local_context.block_idx = 1;
        if (change == 2) m.context->header.run_epoch++;
        if (change == 3) m.context->header.use_begin++;
        if (change == 4) m.context->header.provenance_offset += 64;
        if (change == 5) m.payload.local_context.async_ctx.completion_count = &foreign_count;
        if (change == 6) m.payload.local_context.async_ctx.completion_entries++;
        if (change == 7) m.payload.local_context.async_ctx.completion_capacity--;
        if (change == 8) m.context->evidence.flags |= ASYNC_SUBMISSION_SEALED;
        if (change == 9)
            scheduler_submission_detail::at<SchedulerSubmissionReleaseState>(
                m.proof.submissions, m.proof.submissions.header.release_states_offset
            )
                ->phase = static_cast<uint64_t>(SchedulerSubmissionReleasePhase::RELEASED);
        SdmaQualifiedAccess access;
        AsyncCtx writer;
        AsyncSubmissionContext *context = nullptr;
        EXPECT_FALSE(m.access(&access, &writer, &context)) << change;
        EXPECT_EQ(context, nullptr);
        EXPECT_EQ(writer.completion_error_code, nullptr);
    }
}
TEST(HbgSdmaQualifiedSubmission, NativeChildHandleIsKeptDistinctFromDenseExecutionReference) {
    Mounted m;
    m.payload.local_context.async_ctx.task_token = TaskId::make_sub_task(73, 9);
    m.context->header.task_token = TaskId::to_uint64(m.payload.local_context.async_ctx.task_token);
    SdmaQualifiedAccess access;
    AsyncCtx writer;
    AsyncSubmissionContext *context = nullptr;
    ASSERT_TRUE(m.access(&access, &writer, &context));
    EXPECT_EQ(access.identity.task_ref, 0);
    EXPECT_EQ(TaskId::to_uint64(writer.task_token), TaskId::to_uint64(TaskId::make_sub_task(73, 9)));
    m.payload.local_context.async_ctx.task_token = TaskId::make_global(0);
    EXPECT_FALSE(m.access(&access, &writer, &context));
}

TEST(HbgSdmaQualifiedSubmission, EarlyShapeAndContextFailuresNeverFlushForeignWriter) {
    for (uint32_t broken_context = 0; broken_context < 3; ++broken_context) {
        for (bool valid_shape : {false, true}) {
            Mounted m;
            Driver backend_storage;
            EventCache cache;
            SdmaIgnoreCompletionRecords sink;
            uint32_t foreign_count = 2;
            int32_t foreign_error = 0;
            auto &writer = m.payload.local_context.async_ctx;
            writer.completion_count = &foreign_count;
            writer.completion_error_code = &foreign_error;
            if (broken_context == 1) m.payload.async_submission_context = 0;
            if (broken_context == 2) ++m.context->header.version;
            EXPECT_FALSE(sdma_qualified_entry(
                m.args(), valid_shape, reinterpret_cast<uint64_t>(m.workspace.data()), 0x20050, 0x30090, 32, kLayout,
                backend_storage.backend, cache, sink
            ));
            EXPECT_EQ(foreign_count, 2u);
            EXPECT_EQ(foreign_error, 0);
            EXPECT_EQ(backend_storage.built, 0u);
            EXPECT_EQ(backend_storage.posted, 0u);
            for (uint64_t address : cache.events) {
                EXPECT_NE(address, reinterpret_cast<uint64_t>(&foreign_count));
                EXPECT_NE(address, reinterpret_cast<uint64_t>(&foreign_error));
                EXPECT_TRUE(address == 0 || address == reinterpret_cast<uint64_t>(&m.context->evidence));
            }
            if (broken_context == 0) {
                EXPECT_EQ(m.context->evidence.error_code, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
                EXPECT_NE(
                    std::find(
                        cache.events.begin(), cache.events.end(), reinterpret_cast<uint64_t>(&m.context->evidence)
                    ),
                    cache.events.end()
                );
            } else {
                EXPECT_EQ(m.context->evidence.error_code, 0);
                EXPECT_TRUE(cache.events.empty());
            }
        }
    }
}

TEST(HbgSdmaQualifiedSubmission, InvalidShapeReportsOnlyToCanonicalWriterAndKeepsNoAttempt) {
    Mounted m;
    Driver backend_storage;
    EventCache cache;
    SdmaIgnoreCompletionRecords sink;
    EXPECT_FALSE(sdma_qualified_entry(
        m.args(), false, reinterpret_cast<uint64_t>(m.workspace.data()), 0x20050, 0x30090, 0, kLayout,
        backend_storage.backend, cache, sink
    ));
    EXPECT_EQ(*m.payload.local_context.async_ctx.completion_error_code, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
    EXPECT_EQ(m.context->evidence.error_code, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
    EXPECT_EQ(m.context->evidence.attempted, 0u);
    EXPECT_EQ(backend_storage.built, 0u);
}

}  // namespace
