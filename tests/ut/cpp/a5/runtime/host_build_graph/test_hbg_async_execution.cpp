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

#include "inner_kernel.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdlib>
#include <memory>
#include <set>
#include <vector>

#include "async_plan.h"
#include "hbg_scheduler_test_support.h"
#include "scheduler/scheduler_dispatch.h"
#include "scheduler/scheduler_async_execution.h"

namespace {
constexpr uint64_t kEpoch = 47;

struct ExecutionBackend {
    bool allow_binding{true};
    bool ready{false};
    uint32_t observations{0};
    bool bind_condition(
        const SchedulerAsyncGroupIdentity &identity, uint32_t index, const DeferredCompletionEntry &entry,
        SchedulerAsyncConditionBinding *binding
    ) {
        if (!allow_binding || identity.run_epoch != kEpoch || index != 0 || entry.addr != 0x8000 ||
            entry.backend_cookie != 0)
            return false;
        *binding = {0, 0};
        return true;
    }
    bool validate(
        const SchedulerAsyncGroupIdentity &identity, uint32_t index, const DeferredCompletionEntry &entry,
        const SchedulerAsyncConditionBinding &binding
    ) {
        SchedulerAsyncConditionBinding expected{};
        return bind_condition(identity, index, entry, &expected) && binding.binding_index == 0 &&
               binding.owner_generation == 0;
    }
    SchedulerAsyncConditionOutcome observe_once(
        const SchedulerAsyncGroupIdentity &, uint32_t, const DeferredCompletionEntry &,
        const SchedulerAsyncConditionBinding &
    ) {
        ++observations;
        return {ready ? SchedulerAsyncConditionState::READY : SchedulerAsyncConditionState::PENDING, 0};
    }
    SchedulerAsyncConditionOutcome acquire_completed(
        const SchedulerAsyncGroupIdentity &, uint32_t, const DeferredCompletionEntry &,
        const SchedulerAsyncConditionBinding &, uint64_t &, uint64_t &
    ) {
        return {SchedulerAsyncConditionState::READY, 0};
    }
};

struct ExecutionFixture {
    ExecutionFixture(
        uint32_t blocks, LaneMask lanes, bool sync_start = false, uint32_t async_mask = 7, bool skip = false
    ) :
        storage(1, 6, blocks * 3, {6, 2, lanes.is_mix() || blocks > 1 ? 1u : 0u, lanes.is_mix() ? 1u : 0u}),
        graph(1),
        image(nullptr, &std::free) {
        graph.mixed(0, lanes);
        auto &metadata = storage.metadata[0];
        metadata.active_mask = lanes;
        metadata.logical_block_num = blocks;
        metadata.flags = SCHEDULER_TASK_EXECUTABLE | (blocks > 1 ? SCHEDULER_TASK_SPMD : 0) |
                         (lanes.is_mix() ? SCHEDULER_TASK_MIX : 0) | (sync_start ? SCHEDULER_TASK_SYNC_START : 0);
        if (skip) {
            metadata.flags |= SCHEDULER_TASK_HAS_PREDICATE;
            graph.predicate(0, reinterpret_cast<uint64_t>(&predicate), 8, static_cast<uint8_t>(PredicateOp::EQ), 1);
        }
        metadata.timing_slot = -1;
        for (uint32_t lane = 0; lane < 3; ++lane)
            metadata.kernel_ids[lane] = lanes.has(static_cast<SubtaskSlot>(lane)) ? 1 : INVALID_KERNEL_ID;
        std::vector<SchedulerAsyncCapabilities> caps(1);
        for (uint32_t lane = 0; lane < 3; ++lane) {
            const bool deferred = lanes.has(static_cast<SubtaskSlot>(lane)) && (async_mask & (1u << lane));
            caps[0][lane] = {
                CORE_ASYNC_CAPABILITY_VERSION, deferred ? CoreAsyncMode::DEFERRED : CoreAsyncMode::SYNCHRONOUS,
                deferred ? 1u : 0u, deferred ? CORE_ASYNC_CONDITION_COUNTER : 0u
            };
        }
        std::vector<SchedulerTaskMetadata> tasks{metadata};
        EXPECT_TRUE(scheduler_plan_async_layout(tasks, caps, kEpoch, 1 << 20, {}, &async.layout, nullptr));
        image.reset(std::aligned_alloc(128, async.layout.total_bytes));
        EXPECT_NE(image, nullptr);
        async.base = static_cast<uint8_t *>(image.get());
        EXPECT_TRUE(
            scheduler_initialize_async_state(tasks, caps, async.layout, async.base, async.layout.total_bytes, nullptr)
        );
        EXPECT_TRUE(scheduler_init_spmd_task_records(base(), storage.layout, kEpoch));
        run()->active_worker_count = 6;
        run()->aic_active_worker_count = 2;
        run()->aiv_active_worker_count = 4;
        run()->expected_task_count = 1;
        for (uint32_t worker = 0; worker < 6; ++worker) {
            auto &wire = storage.contexts[worker];
            wire.is_scheduler = 1;
            wire.scheduler_count = 6;
            for (uint32_t lane = 0; lane < 3; ++lane)
                wire.cluster_worker_ids[lane] = (worker / 3) * 3 + lane;
            storage.test_graphs[worker] = graph.graph();
            auto *local = storage.local_context(&wire);
            locals[worker] = local;
            local->has_mix = lanes.is_mix();
            EXPECT_TRUE(scheduler_initialize_admission(base(), local, &admissions[worker]));
            if (admissions[worker].capacity != 0) local->admission = &admissions[worker];
            execution_states[worker].view = async;
            discovery[worker].check = scheduler_async_directory_start(async.layout.group_count, worker);
            discovery[worker].finish = scheduler_async_directory_start(async.layout.finish_count, worker);
        }
        auto *publisher = locals[0];
        EXPECT_EQ(
            scheduler_route_task(graph.graph(), base(), publisher, run(), 0, nullptr),
            SchedulerRouteResult::READY_TO_ENQUEUE
        );
        SchedulerReadyBatch batch{};
        EXPECT_TRUE(scheduler_ready_batch_append(base(), publisher, 0, &batch, nullptr));
        EXPECT_TRUE(scheduler_ready_batch_push(
            base(), publisher, scheduler_task_ready_queue(metadata.flags, lanes), &batch, nullptr
        ));
    }

    void *base() { return storage.scheduler_state->base(); }
    SchedulerRunControl *run() { return storage.run_control; }
    SchedulerTaskControl *control() { return scheduler_task_control_at(base(), locals[0], 0); }
    SchedulerAsyncTaskJoin &join() {
        return *scheduler_async_runtime_detail::at<SchedulerAsyncTaskJoin>(async, async.layout.finish_join_offset);
    }
    uint64_t blocks_left() { return scheduler_async_join_blocks(scheduler_async_join_load(join())); }
    bool fill(uint32_t worker) {
        bool failed = false;
        scheduler_fill_own_slots(
            graph.graph(), base(), locals[worker], run(), cursors[worker].data(), nullptr, 0, 0, &failed, nullptr,
            nullptr, nullptr, &async
        );
        return !failed;
    }
    bool begin(uint32_t worker, uint32_t *slot_out, bool with_condition = false) {
        auto *local = locals[worker];
        uint32_t slot = 0;
        uint64_t publication = 0;
        if (!scheduler_local_ready_pop(local, &slot, &publication)) return false;
        if (!scheduler_async_execution_bind(&execution_states[worker], graph.graph(), base(), local, run(), slot))
            return false;
        bool permit = true;
        if (local->admission != nullptr &&
            !scheduler_ordered_can_enter_slot(graph.graph(), base(), local, run(), slot, &permit))
            return false;
        if (!permit) {
            scheduler_local_ready_publish(local, slot);
            return false;
        }
        auto &state = local->slots[slot];
        EXPECT_EQ(scheduler_dispatch_generation(publication), state.generation);
        if (!scheduler_async_execution_permitted(&execution_states[worker], graph.graph(), local, run(), slot))
            return false;
        auto *payload =
            scheduler_state_at<DispatchPayload>(base(), local->dispatch_payload_offset(local->config.self_lane, slot));
        if (with_condition) {
            EXPECT_NE(payload->local_context.async_ctx.completion_count, nullptr);
            auto &ctx = payload->local_context.async_ctx;
            auto &entry = *ctx.completion_entries;
            entry.addr = 0x8000;
            entry.expected_value = 1;
            entry.engine = COMPLETION_ENGINE_SDMA;
            entry.completion_type = COMPLETION_TYPE_COUNTER;
            entry.backend_cookie = 0;
            *ctx.completion_count = 1;
        }
        if (!scheduler_async_execution_returned(
                &execution_states[worker], graph.graph(), base(), local, run(), slot, backend
            ))
            return false;
        local->local_completed_generations[slot] = state.generation;
        *slot_out = slot;
        return true;
    }
    bool retire(uint32_t worker) {
        uint64_t refilled = 0;
        return scheduler_service_cluster_completions(
            graph.graph(), base(), locals[worker], run(), nullptr, nullptr, nullptr, 0, &refilled, nullptr, nullptr,
            &async
        );
    }
    void drain(uint32_t worker) {
        for (uint32_t step = 0; step < 32 && run()->resolved_task_count == 0; ++step) {
            SchedulerAsyncSliceBudget budget{};
            ASSERT_NE(
                scheduler_async_runtime_service(
                    async, graph.graph(), base(), locals[worker], run(), &discovery[worker], &budget, backend
                ),
                SchedulerAsyncRuntimeServiceResult::ERROR
            );
            scheduler_flush_completions(run(), locals[worker]);
        }
    }

    scheduler_test::FixtureStorage storage;
    scheduler_test::GraphBuffer graph;
    SchedulerAsyncRuntimeView async{};
    std::unique_ptr<void, decltype(&std::free)> image;
    std::array<SchedulerLocalState *, 6> locals{};
    std::array<SchedulerAdmissionLocalState, 6> admissions{};
    std::array<std::array<uint64_t, SCHEDULER_READY_QUEUE_COUNT>, 6> cursors{};
    std::array<SchedulerAsyncDiscoveryState, 6> discovery{};
    std::array<SchedulerAsyncExecutionState<ExecutionBackend>, 6> execution_states{};
    ExecutionBackend backend;
    uint64_t predicate{0};
};

TEST(HbgAsyncExecution, OrdinaryRetirementWaitsForConditionAndDoesNotCountSynchronousDone) {
    ExecutionFixture fixture(1, LaneMask::aic());
    ASSERT_TRUE(fixture.fill(0));
    uint32_t slot = 0;
    ASSERT_TRUE(fixture.begin(0, &slot, true));
    ASSERT_TRUE(fixture.retire(0));
    EXPECT_EQ(fixture.blocks_left(), 0u);
    EXPECT_NE(fixture.control()->state, static_cast<int64_t>(SchedulerTaskState::DONE));
    EXPECT_EQ(fixture.locals[0]->pending_completed, 0u);
    fixture.drain(3);
    EXPECT_EQ(fixture.run()->resolved_task_count, 0u);
    EXPECT_GT(fixture.backend.observations, 0u);
    fixture.backend.ready = true;
    fixture.drain(3);
    EXPECT_EQ(fixture.run()->resolved_task_count, 1u);
    EXPECT_EQ(fixture.control()->state, static_cast<int64_t>(SchedulerTaskState::DONE));
    EXPECT_FALSE(fixture.retire(0));
    EXPECT_EQ(fixture.run()->resolved_task_count, 1u);
}

TEST(HbgAsyncExecution, EveryPureBlockReleasesAndFinishCanPrecedeMembershipClose) {
    ExecutionFixture fixture(3, LaneMask::aic());
    for (uint32_t block = 0; block < 3; ++block) {
        ASSERT_TRUE(fixture.fill(0));
        uint32_t slot = 0;
        ASSERT_TRUE(fixture.begin(0, &slot));
        EXPECT_EQ(fixture.locals[0]->slots[slot].block_idx, block);
        ASSERT_TRUE(fixture.retire(0));
        EXPECT_EQ(fixture.blocks_left(), 2 - block);
        EXPECT_EQ(fixture.locals[0]->pending_completed, 0u);
    }
    EXPECT_EQ(fixture.admissions[0].current_task_ref, 0);
    fixture.drain(3);
    EXPECT_EQ(fixture.run()->resolved_task_count, 1u);
    EXPECT_EQ(fixture.admissions[0].current_task_ref, 0);
    bool closed = false;
    ASSERT_TRUE(scheduler_close_ordered_obligation(
        fixture.graph.graph(), fixture.base(), fixture.locals[0], fixture.run(), &closed
    ));
    EXPECT_TRUE(closed);
    EXPECT_EQ(fixture.admissions[0].current_task_ref, SCHEDULER_TASK_REF_INVALID);
    EXPECT_EQ(scheduler_read_error(fixture.run()), 0);
}

TEST(HbgAsyncExecution, DuplicateGenerationCannotReleaseAnotherSpmdBlock) {
    ExecutionFixture fixture(2, LaneMask::aic());
    ASSERT_TRUE(fixture.fill(0));
    uint32_t slot = 0;
    ASSERT_TRUE(fixture.begin(0, &slot));
    const uint32_t generation = fixture.locals[0]->slots[slot].generation;
    ASSERT_TRUE(fixture.retire(0));
    ASSERT_EQ(fixture.blocks_left(), 1u);
    EXPECT_FALSE(scheduler_service_cluster_completion_slot(
        fixture.graph.graph(), fixture.base(), fixture.locals[0], fixture.run(), 0, slot, generation, nullptr, nullptr,
        nullptr, 0, nullptr, nullptr, nullptr, nullptr, false, &fixture.async
    ));
    EXPECT_EQ(fixture.blocks_left(), 1u);
}

TEST(HbgAsyncExecution, BindingCanRetryAtTheSameReadyGeneration) {
    ExecutionFixture fixture(2, LaneMask::aic());
    ASSERT_TRUE(fixture.fill(0));
    auto *local = fixture.locals[0];
    auto &state = fixture.execution_states[0];
    uint32_t slot = 0;
    uint64_t publication = 0;
    ASSERT_TRUE(scheduler_local_ready_pop(local, &slot, &publication));
    const uint32_t generation = local->slots[slot].generation;
    ASSERT_TRUE(
        scheduler_async_execution_bind(&state, fixture.graph.graph(), fixture.base(), local, fixture.run(), slot)
    );
    auto *payload = scheduler_state_at<DispatchPayload>(
        fixture.base(), local->dispatch_payload_offset(local->config.self_lane, slot)
    );
    auto *entries = payload->local_context.async_ctx.completion_entries;
    ASSERT_NE(entries, nullptr);
    ASSERT_TRUE(
        scheduler_async_execution_bind(&state, fixture.graph.graph(), fixture.base(), local, fixture.run(), slot)
    );
    EXPECT_EQ(payload->local_context.async_ctx.completion_entries, entries);
    EXPECT_EQ(state.bound_generations[slot], generation);
    EXPECT_EQ(state.permitted_slot, UINT32_MAX);
    scheduler_local_ready_publish(local, slot);
    ASSERT_TRUE(fixture.begin(0, &slot));
    ASSERT_TRUE(fixture.retire(0));
    EXPECT_EQ(fixture.blocks_left(), 1u);
}

TEST(HbgAsyncExecution, ANewSlotGenerationCannotUseAnOldContextBinding) {
    ExecutionFixture fixture(1, LaneMask::aic());
    ASSERT_TRUE(fixture.fill(0));
    auto *local = fixture.locals[0];
    auto &state = fixture.execution_states[0];
    uint32_t slot = 0;
    uint64_t publication = 0;
    ASSERT_TRUE(scheduler_local_ready_pop(local, &slot, &publication));
    ASSERT_TRUE(
        scheduler_async_execution_bind(&state, fixture.graph.graph(), fixture.base(), local, fixture.run(), slot)
    );
    ++local->slots[slot].generation;
    EXPECT_FALSE(scheduler_async_execution_permitted(&state, fixture.graph.graph(), local, fixture.run(), slot));
    EXPECT_EQ(state.permitted_slot, UINT32_MAX);
    EXPECT_EQ(fixture.blocks_left(), 1u);
}

TEST(HbgAsyncExecution, GrantedPermitSuppressesServiceAndOneIterationHasOnlyOneSlice) {
    ExecutionFixture fixture(2, LaneMask::aic());
    ASSERT_TRUE(fixture.fill(0));
    uint32_t slot = 0;
    ASSERT_TRUE(fixture.begin(0, &slot, true));
    ASSERT_TRUE(fixture.retire(0));
    ASSERT_TRUE(fixture.fill(3));
    auto *local = fixture.locals[3];
    auto &state = fixture.execution_states[3];
    uint64_t publication = 0;
    ASSERT_TRUE(scheduler_local_ready_pop(local, &slot, &publication));
    ASSERT_TRUE(
        scheduler_async_execution_bind(&state, fixture.graph.graph(), fixture.base(), local, fixture.run(), slot)
    );
    bool permit = false;
    ASSERT_TRUE(
        scheduler_ordered_can_enter_slot(fixture.graph.graph(), fixture.base(), local, fixture.run(), slot, &permit)
    );
    ASSERT_TRUE(permit);
    ASSERT_TRUE(scheduler_async_execution_permitted(&state, fixture.graph.graph(), local, fixture.run(), slot));
    bool progress = false;
    const auto service = [&] {
        return scheduler_async_execution_service(
            &state, fixture.graph.graph(), fixture.base(), local, fixture.run(), fixture.backend,
            SchedulerAsyncAdmission::ADMISSION_DENIED, false, &progress
        );
    };
    ASSERT_TRUE(service());
    EXPECT_EQ(fixture.backend.observations, 0u);
    EXPECT_FALSE(state.gate.spent);
    ASSERT_TRUE(scheduler_async_execution_returned(
        &state, fixture.graph.graph(), fixture.base(), local, fixture.run(), slot, fixture.backend
    ));
    local->local_completed_generations[slot] = local->slots[slot].generation;
    ASSERT_TRUE(fixture.retire(3));
    ASSERT_TRUE(service());
    EXPECT_TRUE(state.gate.spent);
    EXPECT_EQ(fixture.backend.observations, 1u);
    ASSERT_TRUE(service());
    EXPECT_EQ(fixture.backend.observations, 1u);
    for (uint32_t iteration = 0; iteration < 4 && fixture.backend.observations != 2; ++iteration) {
        state.gate.begin_iteration();
        ASSERT_TRUE(service());
        EXPECT_TRUE(state.gate.spent);
    }
    EXPECT_EQ(fixture.backend.observations, 2u);
}

TEST(HbgAsyncExecution, PredicateSkipSealsAnEmptyGroupBeforeRelease) {
    ExecutionFixture fixture(1, LaneMask::aic(), false, 7, true);
    ASSERT_TRUE(fixture.fill(0));
    uint32_t slot = 0;
    ASSERT_TRUE(fixture.begin(0, &slot));
    auto *payload =
        scheduler_state_at<DispatchPayload>(fixture.base(), fixture.locals[0]->dispatch_payload_offset(0, slot));
    EXPECT_EQ(payload->function_bin_addr, 0u);
    SchedulerLocalStorage<SchedulerAsyncGroupWriter> writer_storage;
    const auto &writer = writer_storage.get();
    scheduler_async_runtime_detail::load_into(
        scheduler_async_runtime_detail::at<SchedulerAsyncGroupWriter>(
            fixture.async, fixture.async.layout.group_writer_offset
        ),
        &writer_storage.get()
    );
    EXPECT_EQ(writer.sealed, 1u);
    EXPECT_EQ(writer.count, 0u);
    EXPECT_EQ(fixture.backend.observations, 0u);
    ASSERT_TRUE(fixture.retire(0));
    fixture.drain(3);
    EXPECT_EQ(fixture.run()->resolved_task_count, 1u);
}

TEST(HbgAsyncExecution, SealFailureDoesNotPublishExecutionReleaseOrSuccess) {
    ExecutionFixture fixture(1, LaneMask::aic());
    ASSERT_TRUE(fixture.fill(0));
    fixture.backend.allow_binding = false;
    uint32_t slot = 0;
    EXPECT_FALSE(fixture.begin(0, &slot, true));
    EXPECT_NE(scheduler_read_error(fixture.run()), 0);
    EXPECT_NE(scheduler_async_join_load(fixture.join()) & SCHEDULER_ASYNC_JOIN_FAILED, 0u);
    EXPECT_EQ(fixture.locals[0]->local_completed_generations[slot], 0u);
    auto *finish = scheduler_async_runtime_detail::at<SchedulerAsyncLease>(
        fixture.async, fixture.async.layout.finish_lease_offset
    );
    EXPECT_EQ(scheduler_async_lease_load(*finish), scheduler_async_lease_word(1, SchedulerAsyncPhase::DORMANT));
    EXPECT_FALSE(fixture.retire(0));
    EXPECT_EQ(fixture.run()->resolved_task_count, 0u);
    EXPECT_NE(fixture.control()->state, static_cast<int64_t>(SchedulerTaskState::DONE));
}

TEST(HbgAsyncExecution, ErrorBetweenBindingAndPermitStopsKernelEntry) {
    ExecutionFixture fixture(1, LaneMask::aic());
    ASSERT_TRUE(fixture.fill(0));
    auto *local = fixture.locals[0];
    auto &state = fixture.execution_states[0];
    uint32_t slot = 0;
    uint64_t publication = 0;
    ASSERT_TRUE(scheduler_local_ready_pop(local, &slot, &publication));
    ASSERT_TRUE(
        scheduler_async_execution_bind(&state, fixture.graph.graph(), fixture.base(), local, fixture.run(), slot)
    );
    scheduler_gm_store(fixture.run()->scheduler_error, UINT64_C(1));
    EXPECT_FALSE(scheduler_async_execution_permitted(&state, fixture.graph.graph(), local, fixture.run(), slot));
    EXPECT_EQ(state.permitted_slot, UINT32_MAX);
    EXPECT_EQ(fixture.blocks_left(), 1u);
}

TEST(HbgAsyncExecution, InvalidCooperativeOwnershipCannotBindTheExecution) {
    ExecutionFixture fixture(2, LaneMask::aic(), true);
    ASSERT_TRUE(fixture.fill(0));
    fixture.admissions[0].current_block_num = 3;
    uint32_t slot = 0;
    EXPECT_FALSE(fixture.begin(0, &slot));
    EXPECT_NE(scheduler_read_error(fixture.run()), 0);
    EXPECT_EQ(fixture.execution_states[0].bound_generations[slot], 0u);
    EXPECT_EQ(fixture.execution_states[0].permitted_slot, UINT32_MAX);
    EXPECT_EQ(fixture.blocks_left(), 2u);
}

struct AsyncShape {
    uint32_t blocks;
    LaneMask lanes;
    bool sync;
    uint32_t deferred;
};
class HbgAsyncExecutionShapes : public ::testing::TestWithParam<AsyncShape> {};

TEST_P(HbgAsyncExecutionShapes, RealDispatchRetiresAllLanesAndBlocksExactlyOnce) {
    const auto shape = GetParam();
    ExecutionFixture fixture(shape.blocks, shape.lanes, shape.sync, shape.deferred);
    std::set<std::pair<uint32_t, uint32_t>> executions;
    for (uint32_t iteration = 0; iteration < 64 && fixture.blocks_left() != 0; ++iteration) {
        for (uint32_t worker = 0; worker < 6; ++worker) {
            ASSERT_TRUE(fixture.fill(worker));
            uint32_t slot = 0;
            if (!fixture.begin(worker, &slot)) continue;
            const auto &state = fixture.locals[worker]->slots[slot];
            EXPECT_TRUE(executions.emplace(state.block_idx, state.subtask_slot).second);
            const auto before = fixture.blocks_left();
            ASSERT_TRUE(fixture.retire(worker));
            // A lane return cannot retire a GM MIX block; its coordinator
            // owns that contribution after observing the complete cohort.
            if (shape.lanes.is_mix()) EXPECT_EQ(fixture.blocks_left(), before);
        }
    }
    EXPECT_EQ(executions.size(), shape.blocks * shape.lanes.count());
    EXPECT_EQ(fixture.blocks_left(), 0u);
    fixture.drain(4);
    EXPECT_EQ(fixture.run()->resolved_task_count, 1u);
    EXPECT_EQ(scheduler_read_error(fixture.run()), 0);
    for (uint32_t worker = 0; worker < 6; ++worker) {
        EXPECT_TRUE(fixture.fill(worker));
        EXPECT_FALSE(fixture.retire(worker));
        EXPECT_EQ(fixture.locals[worker]->pending_completed, 0u);
    }
    EXPECT_EQ(fixture.run()->resolved_task_count, 1u);
}

INSTANTIATE_TEST_SUITE_P(
    SpmdMixAndSyncStart, HbgAsyncExecutionShapes,
    ::testing::Values(
        AsyncShape{5, LaneMask::aiv0(), false, 2}, AsyncShape{4, LaneMask::aiv0(), true, 2},
        AsyncShape{2, LaneMask::aic(), true, 1}, AsyncShape{3, LaneMask(true, true, false), false, 1},
        AsyncShape{3, LaneMask(true, false, true), false, 4}, AsyncShape{3, LaneMask::both_aiv(), false, 2},
        AsyncShape{3, LaneMask::all(), false, 7}, AsyncShape{2, LaneMask::all(), true, 2}
    )
);
}  // namespace
