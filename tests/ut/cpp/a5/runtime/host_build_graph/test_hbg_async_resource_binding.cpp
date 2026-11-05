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
    alignas(128) std::array<uint8_t, 2048> image{};
    alignas(64) std::array<uint8_t, 512> records{};
    alignas(64) std::array<uint8_t, 2048> payload{};
    SchedulerAsyncLayout layout{};
    AsyncResourceTables tables{};
    SchedulerAsyncGroupIdentity group{};
    Observations observed;

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
        group.binding_use_count = 2;
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
        return {tables, core, Reader{&observed}, Cache{&observed}};
    }
    DeferredCompletionEntry counter(uint64_t offset = 0) const {
        return {record_address() + offset, 0, 1, COMPLETION_ENGINE_SDMA, COMPLETION_TYPE_COUNTER, 0};
    }
    DeferredCompletionEntry sdma(uint64_t post) const {
        return {record_address(), post, 0, COMPLETION_ENGINE_SDMA, COMPLETION_TYPE_SDMA_EVENT_RECORD, 0};
    }
    void use_sdma() {
        ranges()[0].bytes = 136;
        for (uint32_t index = 0; index < 2; ++index) {
            conditions()[index].completion_type = COMPLETION_TYPE_SDMA_EVENT_RECORD;
            conditions()[index].source_range = 2;
        }
    }
    AsyncConsumerIdentity identity() const { return {kEpoch, 1, 0, 0, 9, 4}; }
};

TEST(HbgAsyncResourceBinding, MountRequiresExactDisjointSectionsAndPreservesOutputOnFailure) {
    for (uint32_t scenario = 0; scenario < 10; ++scenario) {
        Fixture f;
        switch (scenario) {
        case 0:
            f.header().version = 2;
            break;
        case 1:
            ++f.header().run_epoch;
            break;
        case 2:
            f.header().span_count = UINT64_MAX;
            break;
        case 3:
            f.header().reserved = 1;
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
            --group.binding_use_count;
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
    EXPECT_TRUE(backend.bind_condition(f.group, 0, entry, &binding));
    entry.addr += 256;  // In the retained workspace, but beyond the public record slots.
    EXPECT_FALSE(backend.bind_condition(f.group, 0, entry, &binding));
    EXPECT_EQ(f.observed.reads, 0u);
}

TEST(HbgAsyncResourceBinding, PayloadRelationshipAndWholeCleanLineAreRequired) {
    Fixture f;
    auto backend = f.backend();
    SchedulerAsyncConditionBinding binding{};
    f.conditions()[0].payload_range = ASYNC_RESOURCE_NONE;
    EXPECT_FALSE(backend.bind_condition(f.group, 0, f.counter(), &binding));
    f.conditions()[0].payload_none = 1;
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
    f.conditions()[0].record_stride = 8;
    f.ranges()[0].bytes = 16;
    auto entry = f.sdma(7);
    entry.addr += 8;
    EXPECT_TRUE(backend.bind_condition(f.group, 0, entry, &binding));
    entry.addr += 8;
    EXPECT_FALSE(backend.bind_condition(f.group, 0, entry, &binding));
}

TEST(HbgAsyncResourceBinding, ObserverUsesFullPostIdAndNeverAcquiresAnotherCoresPayload) {
    Fixture f;
    f.use_sdma();
    auto backend = f.backend();
    const uint64_t post = (UINT64_C(1) << 50) + 9;
    const auto entry = f.sdma(post);
    SchedulerAsyncConditionBinding binding{};
    ASSERT_TRUE(backend.bind_condition(f.group, 0, entry, &binding));
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
    EXPECT_FALSE(
        f.backend(ASYNC_CORE_AIC)
            .prepare_sdma_submit(f.group, ctx, f.payload_address() + 512, f.payload_address() + 64, 128, 2, &allowance)
    );
    EXPECT_EQ(error, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
    error = 0;
    EXPECT_FALSE(backend.prepare_sdma_submit(
        f.group, ctx, f.payload_address() + 512, f.payload_address() + 64, 129, 2, &allowance
    ));
    EXPECT_FALSE(backend.prepare_sdma_submit(
        f.group, ctx, f.payload_address() + 511, f.payload_address() + 64, 128, 2, &allowance
    ));
    EXPECT_FALSE(backend.prepare_sdma_submit(
        f.group, ctx, f.payload_address() + 512, f.payload_address() + 64, 128, 3, &allowance
    ));
    EXPECT_FALSE(allowance.valid);
    EXPECT_EQ(count, 0u);
    EXPECT_EQ(error, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
    // A fresh execution has a fresh writer; an errored execution cannot retry
    // itself into a successful zero-condition seal.
    error = 0;
    ASSERT_TRUE(backend.prepare_sdma_submit(
        f.group, ctx, f.payload_address() + 512, f.payload_address() + 64, 128, 2, &allowance
    ));
    EXPECT_TRUE(allowance.valid);
    EXPECT_EQ(allowance.limit, 2u);
    EXPECT_EQ(count, 0u);
    EXPECT_EQ(f.observed.reads, 0u);
    f.ranges()[1].access &= ~ASYNC_RANGE_WRITE;
    EXPECT_FALSE(backend.prepare_sdma_submit(
        f.group, ctx, f.payload_address() + 512, f.payload_address() + 64, 128, 2, &allowance
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
