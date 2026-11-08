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
#include <atomic>
#include <cstdlib>
#include <memory>
#include <thread>
#include <vector>

#include "async_plan.h"
#include "hbg_scheduler_test_support.h"
#include "ready_pool_plan.h"
#include "scheduler/scheduler_async_runtime.h"
#include "scheduler/scheduler_async_execution.h"
#include "async_completion_writer.h"
#include "backend/async_backend.h"

namespace {
constexpr uint64_t kEpoch = 37;
constexpr uint64_t kRecords = UINT64_C(0x100000000);

struct RuntimeBackend {
    std::array<bool, 64> ready{};
    std::array<bool, 64> failed{};
    std::array<uint32_t, 64> observations{};
    uint32_t acquisitions{0};

    static uint64_t key(const SchedulerAsyncGroupIdentity &identity, uint32_t condition) {
        return (identity.block_idx * 3 + identity.subtask_slot) * 4 + condition;
    }
    SchedulerAsyncConditionOutcome
    observe_once(const SchedulerAsyncGroupIdentity &identity, uint32_t condition, const DeferredCompletionEntry &) {
        const auto index = key(identity, condition);
        ++observations[index];
        if (failed[index]) return {SchedulerAsyncConditionState::FAILED, -913};
        if (ready[index]) ++acquisitions;
        return {ready[index] ? SchedulerAsyncConditionState::READY : SchedulerAsyncConditionState::PENDING, 0};
    }
};

struct RuntimeFixture {
    AicoreSchedulerLayout layout{};
    SchedulerReadyPoolPlan pool{};
    SchedulerAsyncLayout async{};
    uint64_t async_offset{0};
    std::unique_ptr<scheduler_test::SchedulerStateBuffer> storage;
    scheduler_test::BasicGraphBuffer<2> graph_storage{2};
    SchedulerGraphView graph{};
    std::array<SchedulerLocalState, 6> locals;
    std::array<SchedulerWorkerContext, 6> wire_contexts{};
    std::array<SchedulerAsyncRuntimeView, 6> views{};
    std::array<SchedulerAsyncDiscoveryState, 6> discovery{};
    std::array<DispatchPayload, 24> payloads{};
    std::vector<SchedulerTaskMetadata> metadata{2};
    SchedulerRunControl *run{nullptr};
    RuntimeBackend backend;

    RuntimeFixture(uint32_t blocks = 1, uint32_t active_mask = 1) {
        for (auto &task : metadata) {
            task.flags = SCHEDULER_TASK_EXECUTABLE;
            task.active_mask = LaneMask::aic();
            task.logical_block_num = 1;
            task.timing_slot = -1;
        }
        metadata[0].active_mask = LaneMask((active_mask & 1) != 0, (active_mask & 2) != 0, (active_mask & 4) != 0);
        metadata[0].logical_block_num = blocks;
        if (blocks > 1) metadata[0].flags |= SCHEDULER_TASK_SPMD;
        if (metadata[0].active_mask.is_mix()) metadata[0].flags |= SCHEDULER_TASK_MIX;
        metadata[1].trace_index_base = blocks * metadata[0].active_mask.count();
        EXPECT_TRUE(scheduler_plan_layout(
            2, 2, 0, &layout, false, metadata[1].trace_index_base + 1, SchedulerSpmdRequirements{}, false
        ));
        EXPECT_TRUE(scheduler_plan_ready_pool(metadata, 2, 0, layout.total_size, &pool));
        layout.total_size = pool.total_size;
        EXPECT_TRUE(scheduler_plan_async_layout(metadata, kEpoch, 1024 * 1024, &async, nullptr));
        EXPECT_TRUE(scheduler_append_async_layout(
            layout.total_size, async, 1024 * 1024, &async_offset, &layout.total_size, nullptr
        ));
        storage = std::make_unique<scheduler_test::SchedulerStateBuffer>(layout);
        scheduler_init_ready_pool(base(), pool);
        EXPECT_TRUE(scheduler_initialize_async_state(
            metadata, async, reinterpret_cast<uint8_t *>(base()) + async_offset, async.total_bytes, nullptr
        ));
        std::copy(
            metadata.begin(), metadata.end(),
            scheduler_state_at<SchedulerTaskMetadata>(base(), layout.task_metadata_offset)
        );
        run = scheduler_state_at<SchedulerRunControl>(base(), layout.run_control_offset);
        auto *summary = scheduler_state_at<SchedulerGraphSummary>(base(), layout.graph_summary_offset);
        summary->run_epoch = kEpoch;
        summary->async_layout_offset = async_offset;
        graph_storage.mixed(0, metadata[0].active_mask);
        graph_storage.executable(1, 0, {0});
        graph = graph_storage.graph();
        graph.scheduler_state_address = reinterpret_cast<uint64_t>(base());
        graph.scheduler_state_bytes = layout.total_size;
        for (uint32_t worker = 0; worker < locals.size(); ++worker) {
            auto &config = locals[worker].config;
            wire_contexts[worker].cluster_index = worker / 3;
            wire_contexts[worker].worker_index = worker;
            wire_contexts[worker].physical_core_id = worker;
            wire_contexts[worker].core_type = worker % 3 == 0 ? 0 : 1;
            for (uint32_t lane = 0; lane < 3; ++lane)
                wire_contexts[worker].cluster_worker_ids[lane] = (worker / 3) * 3 + lane;
            config.shared_context = &wire_contexts[worker];
            config.graph = &graph;
            config.graph_summary_offset = layout.graph_summary_offset;
            config.task_controls_offset = layout.task_controls_offset;
            config.task_metadata_offset = layout.task_metadata_offset;
            config.ready_pool_offset = pool.header_offset;
            config.ready_inboxes_offset = layout.ready_inboxes_offset;
            config.ready_directory_offset = layout.ready_directory_offset;
            config.cluster_count = 2;
            config.runtime_worker_count = config.scheduler_count = locals.size();
            config.scheduler_index = worker;
            config.self_lane = worker % 3;
            config.self_core_type = worker % 3 == 0 ? 0 : 1;
            for (uint32_t lane = 0; lane < 3; ++lane)
                config.worker_ids[lane] = (worker / 3) * 3 + lane;
            config.cluster_type_ranks[0] = worker / 3;
            config.cluster_type_ranks[1] = (worker / 3) * 2;
            config.cluster_type_ranks[2] = (worker / 3) * 2 + 1;
            EXPECT_TRUE(scheduler_async_runtime_mount(graph, base(), &locals[worker], run, &views[worker]));
            discovery[worker].check = scheduler_async_directory_start(async.group_count, 0);
            discovery[worker].finish = scheduler_async_directory_start(async.finish_count, 0);
        }
        scheduler_gm_store(control(0)->wake_list_head, int64_t{1});
    }
    void *base() { return storage->base(); }
    SchedulerTaskControl *control(int64_t task) { return scheduler_task_control_at(base(), &locals[0], task); }
    template <typename T>
    T *at(uint64_t offset, uint64_t index = 0) {
        return scheduler_async_runtime_detail::at<T>(views[0], offset, index);
    }
    uint64_t group(uint32_t block, uint32_t lane) {
        return scheduler_async_group_index(*at<SchedulerAsyncTaskMap>(async.task_map_offset), block, lane);
    }
    DispatchPayload &payload(uint32_t block, uint32_t lane) { return payloads[block * 3 + lane]; }
    void bind(uint32_t worker, uint32_t block = 0, uint32_t lane = 0) {
        auto &p = payload(block, lane);
        p.local_context.block_idx = block;
        p.local_context.block_num = metadata[0].logical_block_num;
        bool bound = false;
        ASSERT_TRUE(
            scheduler_async_runtime_bind_payload(views[worker], graph, &locals[worker], run, 0, block, lane, &p, &bound)
        );
        ASSERT_TRUE(bound);
    }
    void condition(uint32_t block = 0, uint32_t lane = 0, uint32_t index = 0) {
        const auto identity = *at<SchedulerAsyncGroupIdentity>(async.group_identity_offset, group(block, lane));
        const uint64_t key = RuntimeBackend::key(identity, index);
        auto &ctx = payload(block, lane).local_context.async_ctx;
        auto &entry = ctx.completion_entries[index];
        entry.addr = kRecords + key * 128;
        entry.backend_cookie = 0;
        entry.expected_value = 1;
        entry.engine = COMPLETION_ENGINE_SDMA;
        entry.completion_type = COMPLETION_TYPE_COUNTER;
        entry._pad = 0;
        *ctx.completion_count = index + 1;
    }
    bool seal(uint32_t worker, uint32_t block = 0, uint32_t lane = 0, bool *registered_out = nullptr) {
        bool registered = false;
        const bool ok = scheduler_async_runtime_seal(
            views[worker], graph, base(), &locals[worker], run, 0, block, lane, &payload(block, lane), &registered
        );
        if (registered_out != nullptr) *registered_out = registered;
        return ok;
    }
    bool release(uint32_t worker, uint32_t block, bool *activated = nullptr) {
        SchedulerAsyncExecutionRelease grant{0, block, false};
        bool published = false;
        const bool ok = scheduler_async_runtime_release_execution_block(
            views[worker], graph, base(), &locals[worker], run, &grant, &published
        );
        if (activated != nullptr) *activated = published;
        return ok;
    }
    SchedulerAsyncRuntimeServiceResult service(uint32_t worker, SchedulerAsyncSliceBudget *provided = nullptr) {
        SchedulerAsyncSliceBudget budget{};
        return scheduler_async_runtime_service<RuntimeBackend>(
            views[worker], graph, base(), &locals[worker], run, &discovery[worker],
            provided != nullptr ? provided : &budget, backend
        );
    }
    uint64_t completed() const {
        uint64_t count = 0;
        for (const auto &local : locals)
            count += local.pending_completed;
        return count;
    }
    void drain() {
        for (uint32_t iteration = 0; iteration < 128 && completed() == 0; ++iteration)
            ASSERT_NE(service(iteration % 6), SchedulerAsyncRuntimeServiceResult::ERROR);
        ASSERT_EQ(completed(), 1u);
        const auto queue = scheduler_ready_pool_queue(base(), pool.header, pool.header.queue_begins[0]);
        int64_t refs[2]{};
        uint32_t count = 0;
        ASSERT_EQ(scheduler_ticket_try_pop_batch(queue, refs, 2, &count), SchedulerTicketResult::SUCCESS);
        EXPECT_EQ(count, 1u);
        EXPECT_EQ(refs[0], 1);
        EXPECT_EQ(scheduler_ticket_try_pop_batch(queue, refs, 2, &count), SchedulerTicketResult::NOT_READY);
    }
};

TEST(HbgAsyncRuntime, MountRejectsCorruptEnvelopeBeforeRecordAccess) {
    for (uint32_t scenario = 0; scenario < 9; ++scenario) {
        RuntimeFixture fixture;
        auto *layout = reinterpret_cast<SchedulerAsyncLayout *>(fixture.views[0].base);
        switch (scenario) {
        case 0:
            layout->version = 99;
            break;
        case 1:
            layout->run_epoch += 1;
            break;
        case 2:
            layout->task_count += 1;
            break;
        case 3:
            layout->total_bytes += 128;
            break;
        case 4:
            layout->group_lease_offset = layout->group_writer_offset;
            break;
        case 5:
            layout->entries_offset = UINT64_MAX - 127;
            break;
        case 6:
            layout->check_directory_words += 1;
            break;
        case 7:
            layout->entries_bytes = 1;
            break;
        case 8:
            scheduler_state_at<SchedulerGraphSummary>(fixture.base(), fixture.layout.graph_summary_offset)
                ->async_layout_offset = fixture.graph.scheduler_state_bytes - 128;
            break;
        }
        auto previous = fixture.views[0];
        EXPECT_FALSE(
            scheduler_async_runtime_mount(fixture.graph, fixture.base(), &fixture.locals[0], fixture.run, &previous)
        );
        EXPECT_EQ(previous.base, fixture.views[0].base);
        EXPECT_NE(scheduler_read_error(fixture.run), 0u);
    }
}

TEST(HbgAsyncRuntime, ZeroLocatorDisablesPollingAndClearsStalePayloadContextWithoutAbiChange) {
    RuntimeFixture fixture;
    auto *summary = scheduler_state_at<SchedulerGraphSummary>(fixture.base(), fixture.layout.graph_summary_offset);
    summary->async_layout_offset = 0;
    ASSERT_TRUE(
        scheduler_async_runtime_mount(fixture.graph, fixture.base(), &fixture.locals[0], fixture.run, &fixture.views[0])
    );
    EXPECT_FALSE(fixture.views[0].enabled());
    fixture.payloads[0].local_context.async_ctx.completion_capacity = 63;
    TaskId::assign(fixture.payloads[0].local_context.async_ctx.task_token, TaskId::make_global(1));
    for (auto &value : fixture.payloads[0].args)
        value = UINT64_C(0xdeadbeef);
    bool bound = true;
    ASSERT_TRUE(scheduler_async_runtime_bind_payload(
        fixture.views[0], fixture.graph, &fixture.locals[0], fixture.run, 0, 0, 0, &fixture.payloads[0], &bound
    ));
    EXPECT_FALSE(bound);
    EXPECT_EQ(fixture.payloads[0].local_context.async_ctx.completion_capacity, 0u);
    EXPECT_FALSE(fixture.payloads[0].local_context.async_ctx.task_token.is_valid());
    for (auto value : fixture.payloads[0].args)
        EXPECT_EQ(value, UINT64_C(0xdeadbeef));
    EXPECT_EQ(fixture.service(0), SchedulerAsyncRuntimeServiceResult::IDLE);
}

TEST(HbgAsyncRuntime, AllResidentLanesUseTheSameRunContextAndGuardPool) {
    RuntimeFixture f(1, 7);
    for (uint32_t lane = 0; lane < 3; ++lane) {
        f.bind(lane, 0, lane);
        const auto &payload = f.payload(0, lane);
        EXPECT_EQ(payload.async_context_address, reinterpret_cast<uint64_t>(&payload.async_context));
        EXPECT_EQ(payload.async_context.subtask_slot, lane);
        auto ctx = AsyncCtx::from_storage(payload.local_context.async_ctx);
        ctx.execution_context = const_cast<AsyncExecutionContext *>(&payload.async_context);
        uint64_t guards = 0;
        ASSERT_TRUE(async_sdma_guard_address(ctx, &guards));
        EXPECT_EQ(guards, reinterpret_cast<uint64_t>(f.views[0].base) + f.async.sdma_guard_context_offset);
    }
}

TEST(HbgAsyncRuntime, CorruptDefaultGuardRegionFailsBeforeBinding) {
    for (uint32_t fault = 0; fault < 5; ++fault) {
        RuntimeFixture f(1, 2);
        auto *layout = reinterpret_cast<SchedulerAsyncLayout *>(f.views[0].base);
        auto *header = f.at<AsyncSdmaGuardContext>(layout->sdma_guard_context_offset);
        if (fault == 0) ++header->run_epoch;
        if (fault == 1) --header->bytes;
        if (fault == 2) layout->sdma_guard_context_offset = layout->group_writer_offset;
        if (fault == 3) layout->sdma_guard_context_offset = layout->total_bytes;
        if (fault == 4) layout->sdma_guard_count = ASYNC_SDMA_CHANNELS + 1;
        SchedulerAsyncRuntimeView view{};
        EXPECT_FALSE(scheduler_async_runtime_mount(f.graph, f.base(), &f.locals[1], f.run, &view));
        EXPECT_FALSE(view.enabled());
    }
}

TEST(HbgAsyncRuntime, PayloadUsesLogicalSubtaskAndIndependentWriterAndTokenRegions) {
    RuntimeFixture fixture(1, 2);
    fixture.bind(2, 0, 1);  // Logical AIV0 on a physical AIV1 worker.
    const auto index = fixture.group(0, 1);
    const auto &identity = *fixture.at<SchedulerAsyncGroupIdentity>(fixture.async.group_identity_offset, index);
    auto *writer = fixture.at<SchedulerAsyncGroupWriter>(fixture.async.group_writer_offset, index);
    const auto &ctx = fixture.payload(0, 1).local_context.async_ctx;
    EXPECT_EQ(ctx.completion_count, &writer->count);
    EXPECT_EQ(ctx.completion_error_code, &writer->error_code);
    EXPECT_EQ(ctx.completion_entries, fixture.at<DeferredCompletionEntry>(identity.entries_offset));
    EXPECT_NE(
        reinterpret_cast<uintptr_t>(ctx.completion_entries), reinterpret_cast<uintptr_t>(ctx.completion_count) + 8
    );
    EXPECT_EQ(TaskId::to_uint64(ctx.task_token), TaskId::to_uint64(TaskId::make_global(0)));
    fixture.condition(0, 1);
    bool registered = false;
    ASSERT_TRUE(fixture.seal(2, 0, 1, &registered));
    EXPECT_TRUE(registered);
}

TEST(HbgAsyncRuntime, PendingThenReadyAcrossCoresActivatesFinishForALaterSliceExactlyOnce) {
    RuntimeFixture fixture;
    fixture.bind(0);
    fixture.condition();
    ASSERT_TRUE(fixture.seal(0));
    bool activated = true;
    ASSERT_TRUE(fixture.release(0, 0, &activated));
    EXPECT_FALSE(activated);
    ASSERT_EQ(fixture.service(3), SchedulerAsyncRuntimeServiceResult::IDLE);
    EXPECT_EQ(fixture.backend.observations[0], 1u);
    EXPECT_EQ(scheduler_gm_load(fixture.control(0)->state), static_cast<int64_t>(SchedulerTaskState::BLOCKED));
    const auto lease = scheduler_async_lease_load(*fixture.at<SchedulerAsyncLease>(fixture.async.group_lease_offset));
    EXPECT_EQ(scheduler_async_lease_phase(lease), SchedulerAsyncPhase::RUNNABLE);
    EXPECT_EQ(scheduler_async_lease_generation(lease), 2u);
    fixture.backend.ready[0] = true;
    ASSERT_EQ(fixture.service(1), SchedulerAsyncRuntimeServiceResult::PROGRESS);
    EXPECT_EQ(fixture.completed(), 0u);
    EXPECT_EQ(scheduler_gm_load(fixture.control(0)->state), static_cast<int64_t>(SchedulerTaskState::BLOCKED));
    EXPECT_EQ(
        scheduler_async_directory_load(*fixture.at<SchedulerAsyncDirectoryWord>(fixture.async.check_directory_offset)),
        0u
    );
    EXPECT_EQ(
        scheduler_async_directory_load(*fixture.at<SchedulerAsyncDirectoryWord>(fixture.async.finish_directory_offset)),
        1u
    );
    fixture.drain();
    for (uint32_t worker = 0; worker < 6; ++worker)
        EXPECT_EQ(fixture.service(worker), SchedulerAsyncRuntimeServiceResult::IDLE);
    EXPECT_EQ(fixture.completed(), 1u);
    EXPECT_EQ(fixture.backend.observations[0], 2u);
    EXPECT_EQ(fixture.backend.acquisitions, 1u);
}

TEST(HbgAsyncRuntime, PendingCursorRotationAndLeaseHandoffDoNotReportProgress) {
    RuntimeFixture fixture(1, 1);
    fixture.bind(0);
    for (uint32_t condition = 0; condition < 16; ++condition)
        fixture.condition(0, 0, condition);
    ASSERT_TRUE(fixture.seal(0));
    ASSERT_TRUE(fixture.release(0, 0));
    for (uint32_t slice = 0; slice < 24; ++slice) {
        SchedulerAsyncSliceBudget budget{4, 2, 2, 8};
        EXPECT_EQ(fixture.service(slice % 6, &budget), SchedulerAsyncRuntimeServiceResult::IDLE);
        EXPECT_LE(budget.work_units, 8u);
    }
    for (uint32_t condition = 0; condition < 16; ++condition)
        EXPECT_GT(fixture.backend.observations[condition], 0u);
    EXPECT_EQ(fixture.completed(), 0u);
    fixture.backend.ready[0] = true;
    bool progressed = false;
    for (uint32_t iteration = 0; iteration < 12; ++iteration)
        progressed |= fixture.service(iteration % 6) == SchedulerAsyncRuntimeServiceResult::PROGRESS;
    EXPECT_TRUE(progressed);
    fixture.backend.ready.fill(true);
    fixture.drain();
    EXPECT_EQ(fixture.completed(), 1u);
}

TEST(HbgAsyncRuntime, PendingServiceDoesNotResetExecutorProgress) {
    RuntimeFixture fixture;
    fixture.bind(0);
    fixture.condition();
    ASSERT_TRUE(fixture.seal(0));
    SchedulerAsyncExecutionState<RuntimeBackend> execution{};
    execution.view = fixture.views[3];
    for (bool already_progressing : {false, true}) {
        execution.gate.begin_iteration();
        bool progress = already_progressing;
        ASSERT_TRUE(scheduler_async_execution_service(
            &execution, fixture.graph, fixture.base(), &fixture.locals[3], fixture.run, fixture.backend,
            SchedulerAsyncAdmission::BEFORE_ADMISSION, false, &progress
        ));
        EXPECT_EQ(progress, already_progressing);
    }
    EXPECT_GT(fixture.backend.observations[0], 0u);
}

struct RuntimeRetryReserve : SchedulerCompletionQueueOps {
    static inline bool retry = true;
    static SchedulerTicketResult
    reserve(const SchedulerTicketQueueView &queue, uint64_t count, SchedulerTicketRange *range) {
        return retry ? SchedulerTicketResult::RETRY : SchedulerCompletionQueueOps::reserve(queue, count, range);
    }
};

TEST(HbgAsyncRuntime, FinishRetryDoesNotReportProgressButTraversalAndPublicationDo) {
    RuntimeFixture fixture;
    fixture.bind(0);
    ASSERT_TRUE(fixture.seal(0));
    ASSERT_TRUE(fixture.release(0, 0));
    RuntimeRetryReserve::retry = true;
    auto service = [&](uint32_t worker) {
        SchedulerAsyncSliceBudget budget{};
        return scheduler_async_runtime_service<RuntimeBackend, RuntimeRetryReserve>(
            fixture.views[worker], fixture.graph, fixture.base(), &fixture.locals[worker], fixture.run,
            &fixture.discovery[worker], &budget, fixture.backend
        );
    };
    EXPECT_EQ(service(1), SchedulerAsyncRuntimeServiceResult::PROGRESS);
    EXPECT_EQ(service(2), SchedulerAsyncRuntimeServiceResult::IDLE);
    EXPECT_EQ(service(3), SchedulerAsyncRuntimeServiceResult::IDLE);
    EXPECT_EQ(fixture.completed(), 0u);
    RuntimeRetryReserve::retry = false;
    EXPECT_EQ(service(4), SchedulerAsyncRuntimeServiceResult::PROGRESS);
    fixture.drain();
    EXPECT_EQ(fixture.completed(), 1u);
}

TEST(HbgAsyncRuntime, ZeroConditionGroupDoesNotActivateCheckAndLastBlockActivatesFinish) {
    RuntimeFixture fixture;
    fixture.bind(0);
    bool registered = true;
    ASSERT_TRUE(fixture.seal(0, 0, 0, &registered));
    EXPECT_FALSE(registered);
    EXPECT_EQ(
        scheduler_async_directory_load(*fixture.at<SchedulerAsyncDirectoryWord>(fixture.async.check_directory_offset)),
        0u
    );
    EXPECT_EQ(
        scheduler_async_lease_phase(
            scheduler_async_lease_load(*fixture.at<SchedulerAsyncLease>(fixture.async.group_lease_offset))
        ),
        SchedulerAsyncPhase::DORMANT
    );
    bool activated = false;
    ASSERT_TRUE(fixture.release(0, 0, &activated));
    EXPECT_TRUE(activated);
    fixture.drain();
}

TEST(HbgAsyncRuntime, LaterBlocksMayRegisterGroupsAfterEarlierChecksDrain) {
    RuntimeFixture fixture(2, 7);
    for (uint32_t lane : {0u, 1u, 2u}) {
        fixture.bind(lane, 0, lane);
        if (lane != 1) fixture.condition(0, lane);
        ASSERT_TRUE(fixture.seal(lane, 0, lane));
        fixture.backend.ready[RuntimeBackend::key(
            *fixture.at<SchedulerAsyncGroupIdentity>(fixture.async.group_identity_offset, fixture.group(0, lane)), 0
        )] = true;
    }
    ASSERT_TRUE(fixture.release(0, 0));
    for (uint32_t iteration = 0; iteration < 16; ++iteration)
        ASSERT_NE(fixture.service(iteration % 6), SchedulerAsyncRuntimeServiceResult::ERROR);
    EXPECT_EQ(fixture.completed(), 0u);
    EXPECT_EQ(
        scheduler_async_join_load(*fixture.at<SchedulerAsyncTaskJoin>(fixture.async.finish_join_offset)), UINT64_C(1)
                                                                                                              << 16
    );
    for (uint32_t lane : {0u, 1u, 2u}) {
        fixture.bind(lane + 3, 1, lane);
        if (lane == 2) fixture.condition(1, lane);
        ASSERT_TRUE(fixture.seal(lane + 3, 1, lane));
    }
    ASSERT_TRUE(fixture.release(3, 1));
    fixture.backend.ready[20] = true;
    fixture.drain();
    EXPECT_EQ(fixture.backend.acquisitions, 3u);
}

TEST(HbgAsyncRuntime, ReleaseBeforeSealAndRepeatedGrantFailWithoutSuccessfulTaskCompletion) {
    {
        RuntimeFixture fixture;
        EXPECT_FALSE(fixture.release(0, 0));
        EXPECT_EQ(fixture.completed(), 0u);
        EXPECT_EQ(scheduler_gm_load(fixture.control(0)->state), static_cast<int64_t>(SchedulerTaskState::BLOCKED));
    }
    {
        RuntimeFixture fixture(2);
        fixture.bind(0);
        ASSERT_TRUE(fixture.seal(0));
        SchedulerAsyncExecutionRelease grant{0, 0, false};
        bool published = false;
        ASSERT_TRUE(scheduler_async_runtime_release_execution_block(
            fixture.views[0], fixture.graph, fixture.base(), &fixture.locals[0], fixture.run, &grant, &published
        ));
        EXPECT_FALSE(scheduler_async_runtime_release_execution_block(
            fixture.views[3], fixture.graph, fixture.base(), &fixture.locals[3], fixture.run, &grant, &published
        ));
        EXPECT_EQ(fixture.service(1), SchedulerAsyncRuntimeServiceResult::ERROR);
        EXPECT_EQ(fixture.completed(), 0u);
    }
}

TEST(HbgAsyncRuntime, StickyWriterOverflowInvalidConditionAndDuplicateSealPreserveFailureEvidence) {
    for (uint32_t scenario = 0; scenario < 4; ++scenario) {
        RuntimeFixture fixture(1, 1);
        fixture.bind(0);
        fixture.condition();
        auto &ctx = fixture.payload(0, 0).local_context.async_ctx;
        if (scenario == 0) *ctx.completion_error_code = -811;
        if (scenario == 1) *ctx.completion_count = 65;
        if (scenario == 2) fixture.payload(0, 0).local_context.async_ctx.completion_entries[0].addr = 0;
        if (scenario == 3) ASSERT_TRUE(fixture.seal(0));
        EXPECT_FALSE(fixture.seal(0));
        if (scenario == 0) EXPECT_EQ(*ctx.completion_error_code, -811);
        EXPECT_NE(
            scheduler_async_join_load(*fixture.at<SchedulerAsyncTaskJoin>(fixture.async.finish_join_offset)) &
                SCHEDULER_ASYNC_JOIN_FAILED,
            0u
        );
        EXPECT_EQ(fixture.completed(), 0u);
        EXPECT_EQ(scheduler_gm_load(fixture.control(0)->state), static_cast<int64_t>(SchedulerTaskState::BLOCKED));
    }
}

TEST(HbgAsyncRuntime, SixtyFifthWriterAppendPublishesOverflowAsTheFirstSchedulerError) {
    RuntimeFixture fixture(1, 1);
    fixture.bind(0);
    auto ctx = AsyncCtx::from_storage(fixture.payload(0, 0).local_context.async_ctx);
    ctx.completion_reservation =
        reinterpret_cast<AsyncCompletionReservationState *>(fixture.payload(0, 0).async_context.writer_state);
    const CompletionToken token{kRecords, 1, COMPLETION_ENGINE_SDMA, COMPLETION_TYPE_COUNTER, 0};
    for (uint32_t condition = 0; condition < 64; ++condition)
        ASSERT_TRUE(async_completion_append(ctx, token));
    ASSERT_FALSE(async_completion_append(ctx, token));
    ASSERT_EQ(*ctx.completion_count, 64u);
    ASSERT_EQ(*ctx.completion_error_code, SIMPLER_ERROR_ASYNC_WAIT_OVERFLOW);
    EXPECT_FALSE(fixture.seal(0));
    EXPECT_EQ(scheduler_read_error(fixture.run), static_cast<uint64_t>(SchedulerGraphResult::ASYNC_WAIT_OVERFLOW));
    scheduler_record_error(fixture.run, 1, SchedulerGraphResult::INVALID_ARGUMENTS);
    EXPECT_EQ(scheduler_read_error(fixture.run), static_cast<uint64_t>(SchedulerGraphResult::ASYNC_WAIT_OVERFLOW));
    EXPECT_EQ(*ctx.completion_error_code, SIMPLER_ERROR_ASYNC_WAIT_OVERFLOW);
    EXPECT_EQ(fixture.completed(), 0u);
    EXPECT_EQ(scheduler_gm_load(fixture.control(0)->state), static_cast<int64_t>(SchedulerTaskState::BLOCKED));
}

TEST(HbgAsyncRuntime, SealPreservesRecognizedWriterReasonsAndAnEarlierSchedulerFailure) {
    for (const auto reason : {SIMPLER_ERROR_ASYNC_COMPLETION_INVALID, SIMPLER_ERROR_ASYNC_REGISTRATION_FAILED}) {
        RuntimeFixture fixture;
        fixture.bind(0);
        auto ctx = AsyncCtx::from_storage(fixture.payload(0, 0).local_context.async_ctx);
        ctx.completion_reservation =
            reinterpret_cast<AsyncCompletionReservationState *>(fixture.payload(0, 0).async_context.writer_state);
        async_completion_error(ctx, reason);
        EXPECT_FALSE(fixture.seal(0));
        const auto expected = reason == SIMPLER_ERROR_ASYNC_COMPLETION_INVALID ?
                                  SchedulerGraphResult::ASYNC_COMPLETION_INVALID :
                                  SchedulerGraphResult::ASYNC_REGISTRATION_FAILED;
        EXPECT_EQ(scheduler_read_error(fixture.run), static_cast<uint64_t>(expected));
        EXPECT_EQ(*ctx.completion_error_code, reason);
        EXPECT_EQ(fixture.completed(), 0u);
    }
    RuntimeFixture fixture;
    fixture.bind(0);
    auto ctx = AsyncCtx::from_storage(fixture.payload(0, 0).local_context.async_ctx);
    ctx.completion_reservation =
        reinterpret_cast<AsyncCompletionReservationState *>(fixture.payload(0, 0).async_context.writer_state);
    async_completion_error(ctx, SIMPLER_ERROR_ASYNC_WAIT_OVERFLOW);
    scheduler_record_error(fixture.run, 1, SchedulerGraphResult::TIMEOUT);
    EXPECT_FALSE(fixture.seal(0));
    EXPECT_EQ(scheduler_read_error(fixture.run), static_cast<uint64_t>(SchedulerGraphResult::TIMEOUT));
    EXPECT_EQ(*ctx.completion_error_code, SIMPLER_ERROR_ASYNC_WAIT_OVERFLOW);
}

TEST(HbgAsyncRuntime, AbortObserverCannotPublishAGenericErrorBeforeTheOriginalCause) {
    RuntimeFixture fixture;
    auto &join = *fixture.at<SchedulerAsyncTaskJoin>(fixture.async.finish_join_offset);
    std::atomic<bool> observer_ready{false};
    uint64_t observed_error = 0;
    std::thread observer([&] {
        observer_ready.store(true, std::memory_order_release);
        while ((scheduler_async_join_load(join) & SCHEDULER_ASYNC_JOIN_FAILED) == 0) {}
        observed_error = scheduler_read_error(fixture.run);
        scheduler_record_error(fixture.run, 1, SchedulerGraphResult::INVALID_ARGUMENTS);
    });
    while (!observer_ready.load(std::memory_order_acquire)) {}
    EXPECT_FALSE(
        scheduler_async_runtime_detail::fail(
            fixture.graph, &fixture.locals[0], fixture.run, 0, &fixture.views[0], 0,
            SchedulerGraphResult::ASYNC_WAIT_OVERFLOW
        )
    );
    observer.join();
    EXPECT_EQ(observed_error, static_cast<uint64_t>(SchedulerGraphResult::ASYNC_WAIT_OVERFLOW));
    EXPECT_EQ(scheduler_read_error(fixture.run), static_cast<uint64_t>(SchedulerGraphResult::ASYNC_WAIT_OVERFLOW));
}

TEST(HbgAsyncRuntime, EarliestCheckClaimObservesSealedWriterAndConditions) {
    RuntimeFixture f;
    f.bind(0);
    f.condition();
    std::atomic<bool> observing{false};
    bool claimed = false;
    uint32_t sealed = 0, count = 0;
    uint64_t address = 0;
    auto &lease = *f.at<SchedulerAsyncLease>(f.async.group_lease_offset);
    auto &directory = *f.at<SchedulerAsyncDirectoryWord>(f.async.check_directory_offset);
    std::thread peer([&] {
        observing.store(true, std::memory_order_release);
        while (scheduler_async_directory_load(directory) == 0 && scheduler_read_error(f.run) == 0) {}
        if (scheduler_read_error(f.run) != 0) return;
        const auto claim = scheduler_async_lease_try_claim(lease, scheduler_async_lease_load(lease));
        claimed = claim.result == SchedulerAsyncLeaseResult::SUCCESS;
        if (!claimed) return;
        const auto *writer = f.at<SchedulerAsyncGroupWriter>(f.async.group_writer_offset);
        sealed = writer->reservation.sealed;
        count = writer->count;
        address = f.at<DeferredCompletionEntry>(f.async.entries_offset)->addr;
        EXPECT_EQ(
            scheduler_async_lease_handoff(lease, claim.word, SchedulerAsyncPhase::RUNNABLE).result,
            SchedulerAsyncLeaseResult::SUCCESS
        );
    });
    while (!observing.load(std::memory_order_acquire)) {}
    EXPECT_TRUE(f.seal(0));
    peer.join();
    EXPECT_TRUE(claimed);
    EXPECT_EQ(sealed, 1u);
    EXPECT_EQ(count, 1u);
    EXPECT_EQ(address, kRecords);
}

TEST(HbgAsyncRuntime, RejectedRegistrationNeverPublishesCheckOrSuccessfulJoin) {
    RuntimeFixture f;
    f.bind(0);
    f.condition();
    f.payload(0, 0).local_context.async_ctx.completion_entries[0].addr = 0;
    EXPECT_FALSE(f.seal(0));
    EXPECT_EQ(scheduler_async_directory_load(*f.at<SchedulerAsyncDirectoryWord>(f.async.check_directory_offset)), 0u);
    EXPECT_EQ(
        scheduler_async_lease_phase(scheduler_async_lease_load(*f.at<SchedulerAsyncLease>(f.async.group_lease_offset))),
        SchedulerAsyncPhase::FAILED
    );
    EXPECT_NE(
        scheduler_async_join_load(*f.at<SchedulerAsyncTaskJoin>(f.async.finish_join_offset)) &
            SCHEDULER_ASYNC_JOIN_FAILED,
        0u
    );
    EXPECT_EQ(f.completed(), 0u);
}

TEST(HbgAsyncRuntime, BackendFailureKeepsFailedRecordAndNeverMakesTaskDone) {
    RuntimeFixture fixture;
    fixture.bind(0);
    fixture.condition();
    ASSERT_TRUE(fixture.seal(0));
    ASSERT_TRUE(fixture.release(0, 0));
    fixture.backend.failed[0] = true;
    EXPECT_EQ(fixture.service(3), SchedulerAsyncRuntimeServiceResult::ERROR);
    SchedulerLocalStorage<SchedulerAsyncGroupProgress> progress_storage;
    const auto &progress = progress_storage.get();
    scheduler_async_runtime_detail::load_into(
        fixture.at<SchedulerAsyncGroupProgress>(fixture.async.group_progress_offset), &progress_storage.get()
    );
    EXPECT_EQ(progress.failure_code, -913);
    EXPECT_EQ(
        scheduler_async_lease_phase(
            scheduler_async_lease_load(*fixture.at<SchedulerAsyncLease>(fixture.async.group_lease_offset))
        ),
        SchedulerAsyncPhase::FAILED
    );
    EXPECT_EQ(
        scheduler_async_directory_load(*fixture.at<SchedulerAsyncDirectoryWord>(fixture.async.check_directory_offset)),
        1u
    );
    EXPECT_EQ(scheduler_gm_load(fixture.control(0)->state), static_cast<int64_t>(SchedulerTaskState::BLOCKED));
    EXPECT_EQ(fixture.completed(), 0u);
}

TEST(HbgAsyncRuntime, TwoAuthorizedBlockReleasesActivateOneFinishAndCompetingServicesCountOnce) {
    RuntimeFixture fixture(2);
    fixture.bind(0, 0);
    fixture.bind(3, 1);
    ASSERT_TRUE(fixture.seal(0, 0));
    ASSERT_TRUE(fixture.seal(3, 1));
    bool published[2]{};
    bool released[2]{};
    std::thread first([&] {
        released[0] = fixture.release(0, 0, &published[0]);
    });
    std::thread second([&] {
        released[1] = fixture.release(3, 1, &published[1]);
    });
    first.join();
    second.join();
    ASSERT_TRUE(released[0] && released[1]);
    EXPECT_NE(published[0], published[1]);
    std::thread service_a([&] {
        for (uint32_t i = 0; i < 32; ++i)
            fixture.service(0);
    });
    std::thread service_b([&] {
        for (uint32_t i = 0; i < 32; ++i)
            fixture.service(3);
    });
    service_a.join();
    service_b.join();
    EXPECT_EQ(fixture.completed(), 1u);
    EXPECT_EQ(scheduler_read_error(fixture.run), 0u);
    SchedulerLocalStorage<SchedulerAsyncFinishProgress> progress_storage;
    const auto &progress = progress_storage.get();
    scheduler_async_runtime_detail::load_into(
        fixture.at<SchedulerAsyncFinishProgress>(fixture.async.finish_progress_offset), &progress_storage.get()
    );
    EXPECT_TRUE(progress.contributed);
}

TEST(HbgAsyncRuntime, ZeroWorkDoesNotClaimAndOversizedBudgetStillServicesOneRecord) {
    RuntimeFixture fixture(1, 1);
    fixture.bind(0);
    for (uint32_t condition = 0; condition < 64; ++condition)
        fixture.condition(0, 0, condition);
    ASSERT_TRUE(fixture.seal(0));
    SchedulerAsyncSliceBudget zero{4, 2, 2, 0};
    EXPECT_EQ(fixture.service(0, &zero), SchedulerAsyncRuntimeServiceResult::IDLE);
    EXPECT_EQ(
        scheduler_async_lease_generation(
            scheduler_async_lease_load(*fixture.at<SchedulerAsyncLease>(fixture.async.group_lease_offset))
        ),
        1u
    );
    SchedulerAsyncSliceBudget large{100, 100, 100, 100};
    EXPECT_EQ(fixture.service(3, &large), SchedulerAsyncRuntimeServiceResult::IDLE);
    EXPECT_EQ(large.work_units, 92u);
    EXPECT_GE(large.directory_words, 96u);
    EXPECT_GE(large.claim_attempts, 98u);
    EXPECT_EQ(fixture.backend.observations[0], 1u);
    for (uint32_t condition = 0; condition < 64; ++condition)
        EXPECT_EQ(fixture.backend.observations[condition], condition < 8 ? 1u : 0u);
}

TEST(HbgAsyncRuntime, LazyRecordValidationRejectsBadMapsAndAtomicRegionAliasingBeforeBinding) {
    for (uint32_t scenario = 0; scenario < 4; ++scenario) {
        RuntimeFixture fixture;
        auto *map = fixture.at<SchedulerAsyncTaskMap>(fixture.async.task_map_offset);
        auto *identity = fixture.at<SchedulerAsyncGroupIdentity>(fixture.async.group_identity_offset);
        if (scenario == 0) map->async_lane_mask = 8;
        if (scenario == 1) identity->entries_offset = fixture.async.group_lease_offset;
        if (scenario == 2) identity->entries_offset = fixture.async.finish_join_offset;
        if (scenario == 3) identity->task_ref = 1;
        bool bound = true;
        EXPECT_FALSE(scheduler_async_runtime_bind_payload(
            fixture.views[0], fixture.graph, &fixture.locals[0], fixture.run, 0, 0, 0, &fixture.payloads[0], &bound
        ));
        EXPECT_FALSE(bound);
        EXPECT_EQ(fixture.payloads[0].local_context.async_ctx.completion_count, nullptr);
        EXPECT_EQ(fixture.payloads[0].local_context.async_ctx.completion_entries, nullptr);
        EXPECT_EQ(scheduler_gm_load(fixture.control(0)->state), static_cast<int64_t>(SchedulerTaskState::BLOCKED));
    }
}

}  // namespace
