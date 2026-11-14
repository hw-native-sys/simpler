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
#include <atomic>
#include <chrono>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "async_completion_plan.h"
#include "hbg_scheduler_test_support.h"
#include "scheduler/scheduler_completion.h"
#include "scheduler/scheduler_async_finish.h"
#include "scheduler/scheduler_dispatch.h"

namespace {

using namespace pto2::resident;

constexpr uint64_t kUnused = UINT64_C(0xa5a5a5a5a5a5a5a5);

TEST(HbgAsyncControl, PreservesSpecificRegistrationFailuresAtTheRunBoundary) {
    SchedulerAsyncLocalState local;
    local.error_code = SIMPLER_ERROR_ASYNC_WAIT_OVERFLOW;
    EXPECT_EQ(scheduler_async_error_result(&local), SchedulerGraphResult::ASYNC_WAIT_OVERFLOW);
    local.error_code = SIMPLER_ERROR_ASYNC_REGISTRATION_FAILED;
    EXPECT_EQ(scheduler_async_error_result(&local), SchedulerGraphResult::ASYNC_REGISTRATION_FAILED);
    local.error_code = SIMPLER_ERROR_ASYNC_COMPLETION_INVALID;
    EXPECT_EQ(scheduler_async_error_result(&local), SchedulerGraphResult::ASYNC_COMPLETION_INVALID);
    EXPECT_EQ(scheduler_async_error_result(nullptr), SchedulerGraphResult::INVALID_ARGUMENTS);
}

AsyncTaskPlanInput
async_task(int64_t task_ref, uint32_t blocks = 1, LaneMask mask = LaneMask::aic(), uint8_t capacity = 1) {
    AsyncTaskPlanInput input;
    input.task_ref = task_ref;
    input.block_count = blocks;
    input.active_mask = mask;
    for (uint32_t lane = 0; lane < 3; ++lane) {
        if (!mask.has(static_cast<SubtaskSlot>(lane))) continue;
        input.capacities[lane] = capacity;
    }
    return input;
}

uint64_t execution_capacity(const std::vector<AsyncTaskPlanInput> &inputs) {
    uint64_t count = 1;
    for (const auto &input : inputs)
        count = std::max<uint64_t>(count, input.block_count * input.active_mask.count());
    return count;
}

struct AsyncFixture {
    alignas(128) std::array<uint32_t, 32> signal{};
    AsyncCompletionHostPlan plan;
    scheduler_test::FixtureStorage storage;
    scheduler_test::BasicGraphBuffer<128> graph;
    std::array<SchedulerAsyncLocalState, 3> workers{};
    std::array<AsyncWriter, 3> executions{};

    explicit AsyncFixture(uint32_t tasks = 2, std::vector<AsyncTaskPlanInput> inputs = {async_task(0)}) :
        storage(
            tasks, 3, execution_capacity(inputs), {}, false,
            [&](AicoreSchedulerLayout &layout) {
                std::string error;
                EXPECT_TRUE(plan_async_completion(layout.total_size, 1, tasks, inputs, &plan, &error)) << error;
                layout.total_size = plan.total_bytes;
                layout.async_layout_offset = plan.layout_offset;
            }
        ),
        graph(tasks) {
        std::memset(
            static_cast<uint8_t *>(base()) + plan.initialized_bytes, 0xa5, plan.total_bytes - plan.initialized_bytes
        );
        EXPECT_TRUE(initialize_async_completion(base(), storage.layout.total_size, plan));
        scheduler_state_at<SchedulerAsyncLayout>(base(), plan.layout_offset)->image_bytes = storage.layout.total_size;
        scheduler_state_at<SchedulerGraphSummary>(base(), storage.layout.graph_summary_offset)->run_epoch =
            plan.layout.run_epoch;
        for (uint32_t ref = 0; ref < tasks; ++ref)
            graph.executable(ref, 0);
        for (const auto &input : inputs) {
            storage.metadata[input.task_ref].active_mask = input.active_mask;
            storage.metadata[input.task_ref].logical_block_num = input.block_count;
            for (uint32_t lane = 0; lane < 3; ++lane) {
                const bool active = input.active_mask.has(static_cast<SubtaskSlot>(lane));
                graph.storage()[input.task_ref].task.kernel_id[lane] = active ? 1 : INVALID_KERNEL_ID;
                storage.metadata[input.task_ref].kernel_ids[lane] = active ? 1 : UINT16_MAX;
            }
        }
        storage.prepare_catalog();
        if (!storage.prepare_calls(graph.graph())) throw std::logic_error("cannot prepare fixture execution calls");
        for (uint32_t worker = 0; worker < 3; ++worker) {
            for (uint32_t lane = 0; lane < 3; ++lane)
                storage.contexts[worker].cluster_worker_ids[lane] = lane;
            EXPECT_TRUE(scheduler_initialize_async(
                base(), storage.layout.total_size, plan.layout_offset, 1, tasks, storage.layout.task_controls_offset,
                workers[worker]
            ));
            storage.test_graphs[worker] = graph.graph();
            auto *state = storage.local_context(&storage.contexts[worker]);
            state->async = &workers[worker];
            // Async producer execution is driven explicitly by this fixture;
            // start consumer discovery at the first successor's catalog entry.
        }
    }

    void *base() { return storage.scheduler_state->base(); }
    SchedulerLocalState *local(uint32_t worker = 0) { return storage.local_context(&storage.contexts[worker]); }
    void bind_package(uint32_t worker = 0) {
        auto *state = local(worker);
        auto &slot = state->slots[0];
        ASSERT_EQ(scheduler_gm_fetch_add(task_control(slot.task_ref)->next_block, UINT64_C(1)), slot.block_idx);
        ASSERT_EQ(
            scheduler_acquire_execution_token(
                base(), state->config.prepared, worker, slot.task_ref, slot.block_idx, slot.subtask_slot, &slot.prepared
            ),
            PreparedAccessResult::READY
        );
    }
    bool next_execution(uint32_t *slot) {
        if (!storage.prepare_calls(graph.graph())) return false;
        for (uint64_t probe = 0; probe <= graph.graph().task_count; ++probe) {
            bool failed = false;
            scheduler_fill_own_slots(graph.graph(), base(), local(), storage.run_control, 0, &failed, slot);
            if (failed) return false;
            if (*slot != UINT32_MAX) return true;
        }
        return true;
    }
    bool select_execution(int64_t *task, uint32_t *selected = nullptr) {
        uint32_t slot = UINT32_MAX;
        if (!next_execution(&slot) || slot == UINT32_MAX) return false;
        *task = local()->slots[slot].task_ref;
        if (selected != nullptr) *selected = slot;
        return true;
    }
    SchedulerTaskControl *task_control(int64_t ref) {
        return scheduler_state_at<SchedulerTaskControl>(base(), storage.layout.task_controls_offset) + ref;
    }
    SchedulerAsyncTaskJoin &join(uint32_t index = 0) {
        return scheduler_state_at<SchedulerAsyncTaskJoin>(base(), plan.layout.joins_offset)[index];
    }
    SchedulerAsyncLease &group_lease(uint32_t index = 0) {
        return scheduler_state_at<SchedulerAsyncLease>(base(), plan.layout.group_leases_offset)[index];
    }
    SchedulerAsyncLease &finish_lease(uint32_t index = 0) {
        return scheduler_state_at<SchedulerAsyncLease>(base(), plan.layout.finish_leases_offset)[index];
    }

    AsyncWriter start(int64_t task_ref = 0, uint32_t block = 0, uint32_t lane = 0, uint32_t worker = 0) {
        auto &context = storage.prepared_global_context(task_ref, block, lane);
        EXPECT_TRUE(scheduler_async_begin_prebound_execution(&workers[worker], task_ref, context, executions[worker]));
        EXPECT_EQ(context.async_present, 1u);
        return {
            reinterpret_cast<uint8_t *>(context.async_image_address), context.async_image_bytes,
            context.async_run_epoch, context.async_group_offset, context.async_group_index
        };
    }

    void register_counter(const AsyncWriter &writer, uint32_t expected = 1, uint32_t signal_index = 0) {
        const CompletionToken token{
            reinterpret_cast<uint64_t>(&signal[signal_index]), expected, COMPLETION_ENGINE_SDMA,
            COMPLETION_TYPE_COUNTER, 0
        };
        EXPECT_TRUE(register_completion_condition(writer, token));
    }

    bool check(uint32_t worker = 0, uint32_t budget = 8) {
        bool progress = false;
        EXPECT_TRUE(scheduler_async_check_step(
            workers[worker], worker == 0 ? 1 : 2, budget, SchedulerAsyncCompletionObserver{}, progress
        ));
        return progress;
    }
    SchedulerSubtaskFinishResult return_next(int64_t task, SchedulerLocalState *state) {
        const auto word = scheduler_gm_query(task_control(task)->completion_state);
        const auto &metadata = storage.metadata[task];
        const uint32_t returned = scheduler_completion_count(word);
        SchedulerLocalSlotState slot{};
        slot.task_ref = task;
        slot.block_idx = returned / metadata.active_mask.count();
        uint32_t ordinal = returned % metadata.active_mask.count();
        for (uint32_t lane = 0; lane < 3; ++lane) {
            if (!metadata.active_mask.has(static_cast<SubtaskSlot>(lane))) continue;
            if (ordinal-- == 0) {
                slot.subtask_slot = lane;
                break;
            }
        }
        return scheduler_contribute_subtask<false>(graph.graph(), base(), state, storage.run_control, slot, 0);
    }
    // close_computation is called by the last physical return. Drive that
    // precondition through the real counter operation, once per remaining unit.
    bool return_remaining(int64_t task, SchedulerLocalState *state) {
        while (scheduler_completion_remaining(scheduler_gm_query(task_control(task)->completion_state)) != 0) {
            if (return_next(task, state) == SchedulerSubtaskFinishResult::ERROR) return false;
        }
        return true;
    }
    bool close(uint32_t worker = 0, int64_t task = 0, bool *held = nullptr) {
        if (!return_remaining(task, local(worker))) return false;
        bool deferred = false;
        const bool ok = scheduler_async_close_computation(&workers[worker], task, deferred);
        if (held != nullptr) *held = deferred;
        return ok;
    }
    bool finish(uint32_t worker = 0) {
        bool progress = false;
        EXPECT_TRUE(scheduler_async_finish_step(
            graph.graph(), base(), local(worker), storage.run_control, workers[worker], progress
        ));
        return progress;
    }
};

TEST(HbgAsyncControl, HostPreboundEntryUsesExactGroupWithoutRegisteringCredit) {
    for (uint32_t corrupt = 0; corrupt < 3; ++corrupt) {
        SCOPED_TRACE(corrupt);
        AsyncFixture f(2, {async_task(0, 2)});
        ASSERT_TRUE(f.storage.prepare_calls(f.graph.graph()));
        uint64_t index = UINT64_MAX;
        ASSERT_TRUE(scheduler_prepared_execution_index(f.base(), f.local()->config.prepared, 0, 1, 0, &index));
        auto &global = scheduler_execution_binding_at(f.base(), f.local()->config.prepared, index)->global_context;
        const uint64_t completion_before = f.task_control(0)->completion_state;
        if (corrupt == 1) ++global.async_run_epoch;
        if (corrupt == 2) --global.async_image_bytes;
        AsyncWriter writer{};
        const bool began = scheduler_async_begin_prebound_execution(&f.workers[0], 0, global, writer);
        EXPECT_EQ(began, corrupt == 0);
        EXPECT_EQ(global.async_present, corrupt == 0 ? 1u : 0u);
        EXPECT_EQ(f.task_control(0)->completion_state, completion_before);
        EXPECT_EQ(f.join().word, 0u);
        EXPECT_EQ(f.group_lease(1).word, kUnused);
        EXPECT_EQ(f.finish_lease().word, kUnused);
        if (corrupt != 0) {
            EXPECT_EQ(writer.image, nullptr);
            continue;
        }
        EXPECT_EQ(writer.image, f.base());
        EXPECT_EQ(writer.image_bytes, f.storage.layout.total_size);
        EXPECT_EQ(writer.run_epoch, 1u);
        EXPECT_EQ(writer.group_index, 1u);
        EXPECT_EQ(writer.group_offset, f.plan.layout.groups_offset + sizeof(AsyncGroupData));
        uint64_t unmapped_index = UINT64_MAX;
        ASSERT_TRUE(scheduler_prepared_execution_index(f.base(), f.local()->config.prepared, 1, 0, 0, &unmapped_index));
        auto &unmapped =
            scheduler_execution_binding_at(f.base(), f.local()->config.prepared, unmapped_index)->global_context;
        ASSERT_TRUE(scheduler_async_begin_prebound_execution(&f.workers[0], 1, unmapped, writer));
        EXPECT_EQ(unmapped.async_present, 0u);
        EXPECT_EQ(writer.image, nullptr);
    }
}

TEST(HbgAsyncControl, BoundExecutionRejectsChangedRunOrStorageBeforePublishing) {
    for (uint32_t corrupt = 0; corrupt < 5; ++corrupt) {
        SCOPED_TRACE(corrupt);
        AsyncFixture f(1, {async_task(0, 2, LaneMask::all())});
        auto writer = f.start();
        f.register_counter(writer);
        switch (corrupt) {
        case 0:
            writer.image += ASYNC_RECORD_ALIGNMENT;
            break;
        case 1:
            --writer.image_bytes;
            break;
        case 2:
            ++writer.run_epoch;
            break;
        case 3:
            ++writer.group_index;
            break;
        case 4:
            writer.group_offset += ASYNC_RECORD_ALIGNMENT;
            break;
        }
        EXPECT_FALSE(scheduler_async_seal_execution(&f.workers[0], 0, 0, 0, writer));
        EXPECT_EQ(scheduler_async_pending::registered(f.join().word), 0u);
        EXPECT_EQ(f.group_lease().word, kUnused);
        EXPECT_EQ(f.finish_lease().word, kUnused);
    }
}

TEST(HbgAsyncControl, BoundExecutionStillChecksTaskBlockAndLaneOnReturn) {
    for (uint32_t corrupt = 0; corrupt < 3; ++corrupt) {
        SCOPED_TRACE(corrupt);
        AsyncFixture f(2, {async_task(0, 2, LaneMask::all()), async_task(1)});
        const auto writer = f.start();
        EXPECT_FALSE(scheduler_async_seal_execution(
            &f.workers[0], corrupt == 0 ? 1 : 0, corrupt == 1 ? 1 : 0, corrupt == 2 ? 1 : 0, writer
        ));
        EXPECT_EQ(scheduler_async_pending::registered(f.join().word), 0u);
        EXPECT_EQ(f.group_lease().word, kUnused);
    }
}

TEST(HbgAsyncControl, ReturnedExecutionReleasesSlotButCannotReleaseConsumerBeforeSignal) {
    AsyncFixture f;
    f.graph.executable(1, 0, {0});
    ASSERT_TRUE(f.storage.prepare_calls(f.graph.graph()));
    const auto writer = f.start();
    f.register_counter(writer);
    ASSERT_TRUE(scheduler_async_seal_execution(&f.workers[0], 0, 0, 0, f.executions[0]));
    auto &slot = f.local()->slots[0];
    slot.task_ref = 0;
    slot.block_idx = 0;
    slot.subtask_slot = 0;
    slot.state = SchedulerDispatchSlotState::EXECUTING;
    f.bind_package();
    ASSERT_TRUE(scheduler_complete_own_slot(f.graph.graph(), f.base(), f.local(), f.storage.run_control, 0, 0));
    EXPECT_EQ(slot.state, SchedulerDispatchSlotState::FREE);
    EXPECT_TRUE(scheduler_async_pending::closed(f.join().word));
    EXPECT_EQ(scheduler_async_pending::pending(f.join().word), 1u);
    EXPECT_EQ(
        scheduler_completion_task_state(f.task_control(0)->completion_state),
        static_cast<int64_t>(SchedulerTaskState::BLOCKED)
    );
    EXPECT_FALSE(f.check(1));
    EXPECT_FALSE(f.finish(2));
    EXPECT_EQ(f.local()->pending_completed, 0u);
    uint32_t blocked_slot = UINT32_MAX;
    ASSERT_TRUE(f.next_execution(&blocked_slot));
    EXPECT_EQ(blocked_slot, UINT32_MAX);
    EXPECT_EQ(scheduler_execution_token_count(f.local()), 1u);
    __atomic_store_n(&f.signal[0], 1u, __ATOMIC_RELEASE);
    EXPECT_TRUE(f.check(2));
    EXPECT_EQ(scheduler_async_lease_phase(f.group_lease().word), SchedulerAsyncPhase::TERMINAL);
    EXPECT_TRUE(scheduler_async_pending::ready_to_finish(f.join().word));
    for (uint32_t step = 0;
         step < 100 && scheduler_async_lease_phase(f.finish_lease().word) != SchedulerAsyncPhase::TERMINAL; ++step)
        f.finish(step % 3);
    EXPECT_EQ(scheduler_async_lease_phase(f.finish_lease().word), SchedulerAsyncPhase::TERMINAL);
    EXPECT_EQ(
        scheduler_completion_task_state(f.task_control(0)->completion_state),
        static_cast<int64_t>(SchedulerTaskState::DONE)
    );
    int64_t ready = -1;
    EXPECT_TRUE(f.select_execution(&ready));
    EXPECT_EQ(ready, 1);
    for (uint32_t worker = 0; worker < 3; ++worker)
        scheduler_flush_completions(f.storage.run_control, f.local(worker));
    EXPECT_EQ(f.storage.run_control->resolved_task_count, 1u);
    EXPECT_FALSE(f.finish());
    scheduler_flush_completions(f.storage.run_control, f.local());
    EXPECT_EQ(f.storage.run_control->resolved_task_count, 1u);
}

TEST(HbgAsyncControl, ConditionsMaySettleBeforeTheLastPhysicalBlockReleases) {
    AsyncFixture f(1);
    auto writer = f.start();
    f.register_counter(writer, 0);
    ASSERT_TRUE(scheduler_async_seal_execution(&f.workers[0], 0, 0, 0, f.executions[0]));
    EXPECT_EQ(f.task_control(0)->completion_state, scheduler_completion_encode(1, 1, 2));
    EXPECT_TRUE(f.check(1));
    EXPECT_EQ(f.task_control(0)->completion_state, scheduler_completion_encode(1, 1, 2));
    EXPECT_EQ(scheduler_async_pending::pending(f.join().word), 0u);
    EXPECT_FALSE(scheduler_async_pending::closed(f.join().word));
    EXPECT_EQ(f.finish_lease().word, kUnused);
    EXPECT_FALSE(f.finish());
    EXPECT_TRUE(f.close(2));
    EXPECT_EQ(f.task_control(0)->completion_state, scheduler_completion_encode(1, 0, 1));
    EXPECT_TRUE(f.finish(1));
    EXPECT_TRUE(scheduler_completion_done(f.task_control(0)->completion_state));
    EXPECT_EQ(f.local(1)->pending_completed, 1u);
    EXPECT_FALSE(f.finish(2));
    EXPECT_EQ(f.local(2)->pending_completed, 0u);
}

TEST(HbgAsyncControl, ZeroConditionSingleExecutionPermitsItsPendingSuccessor) {
    AsyncFixture f;
    f.storage.callable_addresses[1] = 0x1000;
    f.graph.executable(1, 0, {0});
    ASSERT_TRUE(f.storage.prepare_calls(f.graph.graph()));
    (void)f.start();
    ASSERT_TRUE(scheduler_async_seal_execution<true>(&f.workers[0], 0, 0, 0, f.executions[0]));
    auto &slot = f.local()->slots[0];
    slot.task_ref = 0;
    slot.block_idx = 0;
    slot.subtask_slot = 0;
    slot.state = SchedulerDispatchSlotState::EXECUTING;
    f.bind_package();
    ASSERT_TRUE(scheduler_complete_own_slot(f.graph.graph(), f.base(), f.local(), f.storage.run_control, 0, 0));
    EXPECT_EQ(f.local()->local_ready_mask, 0u);
    EXPECT_EQ(slot.state, SchedulerDispatchSlotState::FREE);
    uint32_t direct_slot = UINT32_MAX;
    ASSERT_TRUE(f.next_execution(&direct_slot));
    ASSERT_NE(direct_slot, UINT32_MAX);
    EXPECT_EQ(f.local()->local_ready_mask, 0u);
    EXPECT_EQ(f.local()->slots[direct_slot].task_ref, 1);
    EXPECT_EQ(f.local()->slots[direct_slot].state, SchedulerDispatchSlotState::READY);
    EXPECT_EQ(
        scheduler_completion_task_state(f.task_control(0)->completion_state),
        static_cast<int64_t>(SchedulerTaskState::DONE)
    );
    EXPECT_EQ(f.join().word, 0u);
    EXPECT_EQ(scheduler_async_pending::registered(f.join().word), 0u);
    EXPECT_EQ(f.finish_lease().word, kUnused);
    EXPECT_FALSE(f.finish(1));
    scheduler_flush_completions(f.storage.run_control, f.local());
    EXPECT_EQ(f.storage.run_control->resolved_task_count, 1u);
}

TEST(HbgAsyncControl, LastSynchronousReturnHonorsAnEarlierPeersAsyncRegistration) {
    AsyncFixture f(1, {async_task(0, 2, LaneMask(false, true, false))});
    f.register_counter(f.start(0, 0, 1, 1));
    ASSERT_TRUE(scheduler_async_seal_execution<true>(&f.workers[1], 0, 0, 1, f.executions[1]));
    (void)f.start(0, 1, 1, 2);
    ASSERT_TRUE(scheduler_async_seal_execution<true>(&f.workers[2], 0, 1, 1, f.executions[2]));
    ASSERT_TRUE(f.workers[1].work_activated);
    ASSERT_FALSE(f.workers[2].work_activated);
    for (uint32_t worker : {1u, 2u}) {
        auto *local = f.local(worker);
        auto &slot = local->slots[0];
        slot.task_ref = 0;
        slot.block_idx = worker - 1;
        slot.subtask_slot = 1;
        slot.state = SchedulerDispatchSlotState::EXECUTING;
        f.bind_package(worker);
        ASSERT_TRUE(
            (scheduler_complete_own_slot<true, true>(f.graph.graph(), f.base(), local, f.storage.run_control, 0, 0))
        );
        EXPECT_NE(
            scheduler_completion_task_state(f.task_control(0)->completion_state),
            static_cast<int64_t>(SchedulerTaskState::DONE)
        );
    }
    EXPECT_TRUE(scheduler_async_pending::closed(f.join().word));
    EXPECT_EQ(scheduler_async_pending::pending(f.join().word), 1u);
    f.signal[0] = 1;
    ASSERT_TRUE(f.check(0));
    ASSERT_TRUE(f.finish(0));
    EXPECT_EQ(
        scheduler_completion_task_state(f.task_control(0)->completion_state),
        static_cast<int64_t>(SchedulerTaskState::DONE)
    );
}

TEST(HbgAsyncControl, AllZeroMixShapesCloseSynchronouslyWithoutCheckOrFinishWork) {
    for (uint32_t mask : {3u, 5u, 6u, 7u}) {
        const LaneMask lanes((mask & 1) != 0, (mask & 2) != 0, (mask & 4) != 0);
        AsyncFixture f(1, {async_task(0, 2, lanes)});
        for (uint32_t block = 0; block < 2; ++block) {
            for (uint32_t lane = 0; lane < 3; ++lane) {
                if (!lanes.has(static_cast<SubtaskSlot>(lane))) continue;
                (void)f.start(0, block, lane, lane);
                ASSERT_TRUE(scheduler_async_seal_execution(&f.workers[lane], 0, block, lane, f.executions[lane]));
                EXPECT_EQ(f.join().word, 0u);
            }
            EXPECT_FALSE(f.check());
            EXPECT_FALSE(f.finish());
        }
        bool deferred = true;
        ASSERT_TRUE(f.close(0, 0, &deferred));
        EXPECT_FALSE(deferred);
        EXPECT_TRUE(scheduler_completion_done(f.task_control(0)->completion_state));
        EXPECT_EQ(f.join().word, 0u);
        EXPECT_EQ(scheduler_async_pending::registered(f.join().word), 0u);
        EXPECT_EQ(f.finish_lease().word, kUnused);
    }
}

TEST(HbgAsyncControl, OpenReservationFailsSealWithoutRegisteringPendingWork) {
    AsyncFixture f(1);
    const auto writer = f.start();
    ASSERT_NE(reserve_conditions(writer, 1).serial, 0u);
    EXPECT_FALSE(scheduler_async_seal_execution(&f.workers[0], 0, 0, 0, f.executions[0]));
    EXPECT_EQ(scheduler_async_pending::pending(f.join().word), 0u);
    EXPECT_EQ(scheduler_async_lease_phase(f.group_lease().word), SchedulerAsyncPhase::FAILED);
    EXPECT_EQ(f.finish_lease().word, kUnused);
}

TEST(HbgAsyncControl, UnmappedExecutionClearsThePreviousGroupHandle) {
    AsyncFixture f;
    auto writer = f.start();
    ASSERT_NE(writer.group_index, ASYNC_INVALID_INDEX);
    auto &context = f.storage.prepared_global_context(1, 0, 0);
    context.async_present = 1;
    ASSERT_TRUE(scheduler_async_begin_prebound_execution(&f.workers[0], 1, context, writer));
    EXPECT_EQ(writer.group_index, ASYNC_INVALID_INDEX);
    EXPECT_EQ(writer.image, nullptr);
    EXPECT_EQ(context.async_present, 0u);
    ASSERT_TRUE(scheduler_async_begin_prebound_execution(nullptr, 0, context, writer));
    EXPECT_EQ(writer.group_index, ASYNC_INVALID_INDEX);
}

TEST(HbgAsyncControl, BoundGroupRebuildUsesPrivateIndexInsteadOfKernelContext) {
    AsyncFixture f(1, {async_task(0, 2)});
    auto &context = f.storage.prepared_global_context(0, 1, 0);
    uint32_t index = ASYNC_INVALID_INDEX;
    {
        AsyncWriter writer{};
        ASSERT_TRUE(scheduler_async_begin_prebound_execution(&f.workers[0], 0, context, writer));
        index = writer.group_index;
        ASSERT_NE(index, 0u);
        f.register_counter(writer);
    }
    // Kernel-visible context is not the return identity retained by the executor.
    context.async_image_address = 0;
    context.async_group_index = 0;
    context.async_group_offset = 0;
    context.async_run_epoch = 0;
    const auto writer = scheduler_async_writer_for_group(f.workers[0], index);
    ASSERT_TRUE(scheduler_async_seal_execution<true>(&f.workers[0], 0, 1, 0, writer));
    auto *groups = scheduler_state_at<AsyncGroupData>(f.base(), f.plan.layout.groups_offset);
    EXPECT_EQ(static_cast<AsyncPublicationState>(groups[0].state), AsyncPublicationState::OPEN);
    EXPECT_EQ(static_cast<AsyncPublicationState>(groups[index].state), AsyncPublicationState::SEALED);
    EXPECT_EQ(scheduler_async_pending::registered(f.join().word), 1u);
    EXPECT_EQ(scheduler_async_lease_phase(f.group_lease(index).word), SchedulerAsyncPhase::RUNNABLE);
}

TEST(HbgAsyncControl, ResidentBoundSealPreservesMutableRegistrationFailures) {
    for (uint32_t failure = 0; failure < 4; ++failure) {
        SCOPED_TRACE(failure);
        AsyncFixture f(1);
        const auto writer = f.start();
        auto *data = detail::group(writer);
        ASSERT_NE(data, nullptr);
        int32_t expected = SIMPLER_ERROR_ASYNC_REGISTRATION_FAILED;
        if (failure == 0) {
            data->count = data->capacity + 1;
            expected = SIMPLER_ERROR_ASYNC_WAIT_OVERFLOW;
        } else if (failure == 1) {
            ASSERT_NE(reserve_conditions(writer, 1).serial, 0u);
        } else if (failure == 2) {
            data->error_code = SIMPLER_ERROR_ASYNC_COMPLETION_INVALID;
            expected = SIMPLER_ERROR_ASYNC_COMPLETION_INVALID;
        } else {
            ASSERT_TRUE(scheduler_async_seal_execution<true>(&f.workers[0], 0, 0, 0, writer));
        }
        const auto rebuilt = scheduler_async_writer_for_group(f.workers[0], writer.group_index);
        EXPECT_FALSE(scheduler_async_seal_execution<true>(&f.workers[0], 0, 0, 0, rebuilt));
        EXPECT_EQ(f.workers[0].error_code, expected);
        EXPECT_EQ(scheduler_async_pending::registered(f.join().word), 0u);
        EXPECT_EQ(f.finish_lease().word, kUnused);
    }
}

TEST(HbgAsyncControl, ZeroWorkSkipsServiceAndActivationIsVisibleToOtherWorkers) {
    AsyncFixture f(1);
    EXPECT_FALSE(scheduler_async_work_activated(&f.workers[1]));
    f.workers[1].business_returns = 4;
    bool progress = true;
    ASSERT_TRUE(
        scheduler_service_async_slice(f.graph.graph(), f.base(), f.local(1), f.storage.run_control, 0, progress)
    );
    EXPECT_FALSE(progress);
    EXPECT_FALSE(f.workers[1].finish_first);
    EXPECT_EQ(f.workers[1].business_returns, 0u);
    const auto writer = f.start();
    f.register_counter(writer);
    ASSERT_TRUE(scheduler_async_seal_execution(&f.workers[0], 0, 0, 0, f.executions[0]));
    EXPECT_TRUE(scheduler_async_work_activated(&f.workers[0]));
    EXPECT_TRUE(scheduler_async_work_activated(&f.workers[1]));
    EXPECT_TRUE(scheduler_async_work_activated(&f.workers[2]));
}

TEST(HbgAsyncControl, NoActivityCannotMaskALocalRegistrationFailure) {
    AsyncFixture f(1);
    f.workers[0].failed_task_ref = 0;
    f.workers[0].error_code = SIMPLER_ERROR_ASYNC_REGISTRATION_FAILED;
    f.workers[0].idle_probe_countdown = SCHEDULER_ASYNC_IDLE_PROBE_INTERVAL - 1;
    EXPECT_TRUE(scheduler_async_service_due(f.workers[0]));
    bool progress = true;
    EXPECT_FALSE(
        scheduler_service_async_slice(f.graph.graph(), f.base(), f.local(), f.storage.run_control, 0, progress)
    );
    EXPECT_FALSE(progress);
    EXPECT_NE(scheduler_read_error(f.storage.run_control), 0u);
    EXPECT_EQ(f.storage.run_control->resolved_task_count, 0u);
}

TEST(HbgAsyncControl, ColdIdleDiscoveryIsBoundedAndConsumerAcquisitionDoesNotWaitForIt) {
    AsyncFixture f(1);
    bool progress = false;
    for (uint32_t worker : {1u, 2u}) {
        ASSERT_TRUE(scheduler_async_service_due(f.workers[worker]));
        ASSERT_TRUE(scheduler_service_async_slice(
            f.graph.graph(), f.base(), f.local(worker), f.storage.run_control, 0, progress
        ));
        EXPECT_FALSE(progress);
    }
    const auto writer = f.start();
    f.register_counter(writer, 0);
    ASSERT_TRUE(scheduler_async_seal_execution(&f.workers[0], 0, 0, 0, f.executions[0]));
    ASSERT_TRUE(f.close());
    EXPECT_GT(f.workers[2].idle_probe_countdown, 0u);
    EXPECT_TRUE(scheduler_async_work_activated(&f.workers[2]));
    EXPECT_TRUE(scheduler_async_service_due(f.workers[2]));
    uint32_t probes = 0;
    for (uint32_t opportunity = 0; opportunity < SCHEDULER_ASYNC_IDLE_PROBE_INTERVAL; ++opportunity) {
        if (!scheduler_async_service_due(f.workers[1])) continue;
        ++probes;
        ASSERT_TRUE(
            scheduler_service_async_slice(f.graph.graph(), f.base(), f.local(1), f.storage.run_control, 0, progress)
        );
    }
    EXPECT_EQ(probes, 1u);
    EXPECT_TRUE(progress);
    EXPECT_EQ(scheduler_async_lease_phase(f.group_lease().word), SchedulerAsyncPhase::TERMINAL);
}

TEST(HbgAsyncControl, BusinessReturnsBypassColdIdleDiscoveryDelay) {
    AsyncFixture f(1);
    bool progress = false;
    ASSERT_TRUE(scheduler_async_service_due(f.workers[1]));
    ASSERT_TRUE(
        scheduler_service_async_slice(f.graph.graph(), f.base(), f.local(1), f.storage.run_control, 0, progress)
    );
    const auto writer = f.start();
    f.register_counter(writer, 0);
    ASSERT_TRUE(scheduler_async_seal_execution(&f.workers[0], 0, 0, 0, f.executions[0]));
    ASSERT_GT(f.workers[1].idle_probe_countdown, 0u);
    f.workers[1].business_returns = 4;
    EXPECT_TRUE(scheduler_async_service_due(f.workers[1]));
    EXPECT_TRUE(
        scheduler_service_async_slice(f.graph.graph(), f.base(), f.local(1), f.storage.run_control, 0, progress)
    );
    EXPECT_TRUE(progress);
    EXPECT_EQ(f.workers[1].business_returns, 0u);
}

TEST(HbgAsyncControl, ContinuousBusinessReturnsServiceCheckAndFinishBeforeWorkRunsOut) {
    AsyncFixture f(2);
    f.graph.executable(1, 0, {0});
    const auto writer = f.start();
    f.register_counter(writer, 0);
    ASSERT_TRUE(scheduler_async_seal_execution(&f.workers[0], 0, 0, 0, f.executions[0]));
    ASSERT_TRUE(f.close());
    constexpr uint32_t returns = 64;
    uint32_t completed = 0;
    for (; completed < returns; ++completed) {
        ++f.workers[1].business_returns;
        if (f.workers[1].business_returns < 4) continue;
        ASSERT_TRUE(scheduler_async_service_due(f.workers[1]));
        bool progress = false;
        ASSERT_TRUE(
            scheduler_service_async_slice(f.graph.graph(), f.base(), f.local(1), f.storage.run_control, 0, progress)
        );
        EXPECT_EQ(f.workers[1].business_returns, 0u);
        if (scheduler_async_lease_phase(f.finish_lease().word) == SchedulerAsyncPhase::TERMINAL) break;
    }
    ASSERT_LT(completed, returns);
    EXPECT_EQ(scheduler_async_lease_phase(f.group_lease().word), SchedulerAsyncPhase::TERMINAL);
    int64_t consumer = SCHEDULER_TASK_REF_INVALID;
    ASSERT_TRUE(f.select_execution(&consumer));
    EXPECT_EQ(consumer, 1);
    EXPECT_EQ(scheduler_read_error(f.storage.run_control), 0u);
}

TEST(HbgAsyncControl, UnusedRecordsRemainUntouchedByInitializationAndZeroWaitExecution) {
    AsyncFixture f(1, {async_task(0, 2, LaneMask::all(), 64)});
    const auto *suffix = static_cast<uint8_t *>(f.base()) + f.plan.initialized_bytes;
    const auto *end = static_cast<uint8_t *>(f.base()) + f.plan.total_bytes;
    const auto metadata = f.storage.metadata[0];
    std::array<SchedulerWorkerContext, 3> contexts;
    std::copy_n(f.storage.contexts, contexts.size(), contexts.begin());
    ASSERT_FALSE(scheduler_init_data_from_layout(
        f.base(), f.storage.layout, reinterpret_cast<uint64_t>(f.base()), f.plan.begin_offset - 128
    ));
    ASSERT_TRUE(scheduler_init_data_from_layout(
        f.base(), f.storage.layout, reinterpret_cast<uint64_t>(f.base()), f.plan.initialized_bytes
    ));
    ASSERT_TRUE(initialize_async_completion(f.base(), f.plan.total_bytes, f.plan));
    scheduler_state_at<SchedulerAsyncLayout>(f.base(), f.plan.layout_offset)->image_bytes = f.storage.layout.total_size;
    f.storage.metadata[0] = metadata;
    std::copy(contexts.begin(), contexts.end(), f.storage.contexts);
    // Republish the Host-prepared bindings erased by generic initialization.
    scheduler_state_at<SchedulerGraphSummary>(f.base(), f.storage.layout.graph_summary_offset)->run_epoch =
        f.plan.layout.run_epoch;
    f.storage.task_catalog_offset = 0;
    f.storage.calls_prepared = false;
    f.storage.prepare_catalog();
    ASSERT_TRUE(f.storage.prepare_calls(f.graph.graph()));
    EXPECT_TRUE(std::all_of(suffix, end, [](uint8_t byte) {
        return byte == 0xa5;
    }));
    for (uint32_t block = 0; block < 2; ++block) {
        for (uint32_t lane = 0; lane < 3; ++lane) {
            (void)f.start(0, block, lane, lane);
            ASSERT_TRUE(scheduler_async_seal_execution(&f.workers[lane], 0, block, lane, f.executions[lane]));
        }
    }
    bool held = true;
    ASSERT_TRUE(f.close(0, 0, &held));
    EXPECT_FALSE(held);
    EXPECT_FALSE(scheduler_async_work_activated(&f.workers[0]));
    EXPECT_TRUE(std::all_of(suffix, end, [](uint8_t byte) {
        return byte == 0xa5;
    }));
}

TEST(HbgAsyncControl, DuplicateSealCannotReinitializeAnOwnedCheckLease) {
    AsyncFixture f(1);
    const auto writer = f.start();
    f.register_counter(writer);
    ASSERT_TRUE(scheduler_async_seal_execution(&f.workers[0], 0, 0, 0, f.executions[0]));
    ASSERT_EQ(
        scheduler_async_lease_try_claim(f.group_lease(), f.group_lease().word).result,
        SchedulerAsyncLeaseResult::SUCCESS
    );
    const uint64_t owned = f.group_lease().word;
    EXPECT_FALSE(scheduler_async_seal_execution(&f.workers[0], 0, 0, 0, f.executions[0]));
    EXPECT_EQ(f.group_lease().word, owned);
    EXPECT_EQ(f.task_control(0)->completion_state, scheduler_completion_encode(1, 1, 2));
    EXPECT_EQ(scheduler_async_pending::pending(f.join().word), 1u);
}

TEST(HbgAsyncControl, DifferentConditionCountsKeepOneDebtCreditPerNonemptyGroupUntilFinish) {
    AsyncFixture f(1, {async_task(0, 2, LaneMask::aiv0(), 3)});
    const auto first = f.start(0, 0, 1, 1);
    for (uint32_t signal = 0; signal < 3; ++signal)
        f.register_counter(first, 1, signal);
    ASSERT_TRUE(scheduler_async_seal_execution(&f.workers[1], 0, 0, 1, f.executions[1]));
    f.register_counter(f.start(0, 1, 1, 2), 1, 3);
    ASSERT_TRUE(scheduler_async_seal_execution(&f.workers[2], 0, 1, 1, f.executions[2]));
    ASSERT_EQ(scheduler_async_pending::registered(f.join().word), 2u);
    ASSERT_EQ(f.task_control(0)->completion_state, scheduler_completion_encode(2, 2, 4));
    ASSERT_EQ(f.return_next(0, f.local(1)), SchedulerSubtaskFinishResult::PARTIAL);
    for (uint32_t signal = 0; signal < 3; ++signal)
        f.signal[signal] = 1;
    for (uint32_t probe = 0; probe < 3; ++probe)
        (void)f.check(0);
    EXPECT_EQ(scheduler_async_pending::pending(f.join().word), 1u);
    EXPECT_EQ(f.task_control(0)->completion_state, scheduler_completion_encode(2, 1, 3));
    ASSERT_EQ(f.return_next(0, f.local(2)), SchedulerSubtaskFinishResult::TASK_LAST_DEFERRED);
    ASSERT_TRUE(f.close(2));
    EXPECT_EQ(f.task_control(0)->completion_state, scheduler_completion_encode(2, 0, 2));
    EXPECT_FALSE(f.finish(0));
    f.signal[3] = 1;
    for (uint32_t probe = 0; probe < 2; ++probe)
        (void)f.check(0);
    ASSERT_EQ(scheduler_async_pending::pending(f.join().word), 0u);
    EXPECT_EQ(f.task_control(0)->completion_state, scheduler_completion_encode(2, 0, 2));
    ASSERT_TRUE(f.finish(1));
    EXPECT_FALSE(f.finish(2));
    EXPECT_EQ(f.local(1)->pending_completed, 1u);
    EXPECT_EQ(f.local(2)->pending_completed, 0u);
    EXPECT_TRUE(scheduler_completion_done(f.task_control(0)->completion_state));
}

TEST(HbgAsyncControl, InvalidDebtCarryCannotPublishAGroupOrBecomeLogicalDone) {
    AsyncFixture f(1);
    const auto writer = f.start();
    f.register_counter(writer);
    const uint64_t invalid = scheduler_completion_encode(32766, UINT16_MAX, UINT32_MAX);
    f.task_control(0)->completion_state = invalid;
    ASSERT_FALSE(scheduler_completion_valid(invalid));
    ASSERT_FALSE(scheduler_async_seal_execution(&f.workers[0], 0, 0, 0, f.executions[0]));
    EXPECT_EQ(f.task_control(0)->completion_state, invalid);
    EXPECT_FALSE(scheduler_completion_done(f.task_control(0)->completion_state));
    EXPECT_EQ(scheduler_async_pending::registered(f.join().word), 0u);
    EXPECT_EQ(f.group_lease().word, kUnused);
    EXPECT_EQ(f.finish_lease().word, kUnused);
    EXPECT_EQ(f.storage.run_control->resolved_task_count, 0u);
}

TEST(HbgAsyncControl, ASettledWaitStillCannotFinishBeforeZeroMembersRelease) {
    AsyncFixture f(1, {async_task(0, 2, LaneMask::all())});
    const auto writer = f.start();
    f.register_counter(writer, 0);
    ASSERT_TRUE(scheduler_async_seal_execution(&f.workers[0], 0, 0, 0, f.executions[0]));
    EXPECT_TRUE(f.check(1));
    EXPECT_EQ(scheduler_async_pending::pending(f.join().word), 0u);
    EXPECT_FALSE(f.finish(2));
    for (uint32_t block = 0; block < 2; ++block) {
        for (uint32_t lane = 0; lane < 3; ++lane) {
            if (block == 0 && lane == 0) continue;
            (void)f.start(0, block, lane, lane);
            ASSERT_TRUE(scheduler_async_seal_execution(&f.workers[lane], 0, block, lane, f.executions[lane]));
            EXPECT_FALSE(f.finish());
        }
    }
    bool deferred = false;
    ASSERT_TRUE(f.close(0, 0, &deferred));
    EXPECT_TRUE(deferred);
    EXPECT_EQ(scheduler_async_lease_phase(f.finish_lease().word), SchedulerAsyncPhase::RUNNABLE);
}

TEST(HbgAsyncControl, DirectoryTailWrapDoesNotStarveEarlierPendingGroups) {
    std::vector<AsyncTaskPlanInput> inputs;
    for (uint32_t task = 0; task < 65; ++task)
        inputs.push_back(async_task(task));
    AsyncFixture f(65, inputs);
    auto first = f.start(0);
    f.register_counter(first, 0);
    ASSERT_TRUE(scheduler_async_seal_execution(&f.workers[0], 0, 0, 0, f.executions[0]));
    f.workers[1].check_cursor = 64;
    EXPECT_FALSE(f.check(1));
    EXPECT_EQ(f.workers[1].check_cursor, 0u);
    EXPECT_TRUE(f.check(1));
    EXPECT_EQ(scheduler_async_lease_phase(f.group_lease(0).word), SchedulerAsyncPhase::TERMINAL);
}

TEST(HbgAsyncControl, LowPendingGroupCannotStarveReadyGroupInTheSameDirectoryWord) {
    AsyncFixture f(2, {async_task(0), async_task(1)});
    for (uint32_t task = 0; task < 2; ++task) {
        const auto writer = f.start(task);
        f.register_counter(writer, task == 0 ? 1 : 0);
        ASSERT_TRUE(scheduler_async_seal_execution(&f.workers[0], task, 0, 0, f.executions[0]));
    }
    EXPECT_FALSE(f.check());
    EXPECT_TRUE(f.check());
    EXPECT_EQ(scheduler_async_lease_phase(f.group_lease(0).word), SchedulerAsyncPhase::RUNNABLE);
    EXPECT_EQ(scheduler_async_lease_phase(f.group_lease(1).word), SchedulerAsyncPhase::TERMINAL);
}

TEST(HbgAsyncControl, MoreWaitersThanWorkerSlotsLeavePublisherRunnableAndTerminalWaiterPending) {
    constexpr uint32_t worker_count = 3;
    constexpr uint32_t waiter_count = 2 * worker_count + 1;
    constexpr uint32_t publisher = waiter_count;
    std::vector<AsyncTaskPlanInput> inputs;
    for (uint32_t task = 0; task < waiter_count; ++task)
        inputs.push_back(async_task(task));
    AsyncFixture f(waiter_count + 1, inputs);
    ASSERT_EQ(f.workers.size(), worker_count);
    ASSERT_TRUE(f.storage.prepare_calls(f.graph.graph()));

    // All waiters are graph leaves. Their physical executions return, but
    // completion must remain pending even without a dependent consumer.
    for (uint32_t task = 0; task < waiter_count; ++task) {
        f.register_counter(f.start(task), 1, task);
        ASSERT_TRUE(scheduler_async_seal_execution(&f.workers[0], task, 0, 0, f.executions[0]));
        auto &slot = f.local()->slots[0];
        slot.task_ref = task;
        slot.block_idx = 0;
        slot.subtask_slot = 0;
        slot.state = SchedulerDispatchSlotState::EXECUTING;
        f.bind_package();
        ASSERT_TRUE(scheduler_complete_own_slot(f.graph.graph(), f.base(), f.local(), f.storage.run_control, 0, 0));
        EXPECT_EQ(slot.state, SchedulerDispatchSlotState::FREE);
        EXPECT_EQ(
            scheduler_completion_task_state(f.task_control(task)->completion_state),
            static_cast<int64_t>(SchedulerTaskState::BLOCKED)
        );
    }
    // Each observer can acquire records released by another observer. An
    // unmet condition releases its lease and advances past that record.
    for (uint32_t worker = 0; worker < worker_count; ++worker) {
        for (uint32_t task = 0; task < waiter_count; ++task) {
            ASSERT_FALSE(f.check(worker));
            EXPECT_EQ(f.workers[worker].check_cursor, (task + 1) % waiter_count);
            EXPECT_EQ(scheduler_async_lease_phase(f.group_lease(task).word), SchedulerAsyncPhase::RUNNABLE);
        }
        EXPECT_FALSE(f.finish(worker));
        scheduler_flush_completions(f.storage.run_control, f.local(worker));
    }
    EXPECT_EQ(f.storage.run_control->resolved_task_count, 0u);

    // Exercise the real dispatch and async entry contracts for the ordinary
    // publisher; signal stores stand in for its kernel body in this CPU test.
    uint32_t slot_index = UINT32_MAX;
    ASSERT_TRUE(f.next_execution(&slot_index));
    ASSERT_NE(slot_index, UINT32_MAX);
    ASSERT_EQ(f.local()->slots[slot_index].task_ref, publisher);
    ASSERT_EQ(f.local()->slots[slot_index].state, SchedulerDispatchSlotState::READY);
    f.local()->slots[slot_index].state = SchedulerDispatchSlotState::EXECUTING;
    auto &context = f.storage.prepared_global_context(publisher, 0, 0);
    AsyncWriter writer;
    ASSERT_TRUE(scheduler_async_begin_prebound_execution(&f.workers[0], publisher, context, writer));
    EXPECT_EQ(context.async_present, 0u);
    for (uint32_t task = 0; task + 1 < waiter_count; ++task)
        __atomic_store_n(&f.signal[task], 1u, __ATOMIC_RELEASE);
    ASSERT_TRUE(scheduler_async_seal_execution(&f.workers[0], publisher, 0, 0, writer));
    ASSERT_TRUE(
        scheduler_complete_own_slot(f.graph.graph(), f.base(), f.local(), f.storage.run_control, slot_index, 0)
    );

    for (uint32_t step = 0; step < waiter_count * worker_count; ++step) {
        f.check(step % worker_count);
        f.finish((step + 1) % worker_count);
    }
    for (uint32_t worker = 0; worker < worker_count; ++worker)
        scheduler_flush_completions(f.storage.run_control, f.local(worker));
    for (uint32_t task = 0; task + 1 < waiter_count; ++task)
        EXPECT_EQ(
            scheduler_completion_task_state(f.task_control(task)->completion_state),
            static_cast<int64_t>(SchedulerTaskState::DONE)
        );
    EXPECT_EQ(
        scheduler_completion_task_state(f.task_control(publisher)->completion_state),
        static_cast<int64_t>(SchedulerTaskState::DONE)
    );
    EXPECT_EQ(
        scheduler_completion_task_state(f.task_control(waiter_count - 1)->completion_state),
        static_cast<int64_t>(SchedulerTaskState::BLOCKED)
    );
    EXPECT_EQ(scheduler_async_pending::pending(f.join(waiter_count - 1).word), 1u);
    EXPECT_EQ(scheduler_async_lease_phase(f.group_lease(waiter_count - 1).word), SchedulerAsyncPhase::RUNNABLE);
    EXPECT_EQ(f.finish_lease(waiter_count - 1).word, kUnused);
    EXPECT_EQ(f.storage.run_control->resolved_task_count, waiter_count);
    EXPECT_EQ(scheduler_read_error(f.storage.run_control), 0u);
}

TEST(HbgAsyncControl, CheckBudgetAndSatisfiedMaskSurviveObserverMigration) {
    AsyncFixture f(1, {async_task(0, 1, LaneMask::aic(), 64)});
    const auto writer = f.start();
    for (uint32_t condition = 0; condition < 64; ++condition)
        f.register_counter(writer, 0);
    ASSERT_TRUE(scheduler_async_seal_execution(&f.workers[0], 0, 0, 0, f.executions[0]));
    for (uint32_t round = 0; round < 8; ++round) {
        EXPECT_TRUE(f.check(round % 3, 8));
        const auto *state =
            scheduler_state_at<SchedulerAsyncGroupProgress>(f.base(), f.plan.layout.group_progress_offset);
        EXPECT_EQ(__builtin_popcountll(state->pending_mask), static_cast<int>((7 - round) * 8));
        if (round != 7) EXPECT_EQ(scheduler_async_lease_phase(f.group_lease().word), SchedulerAsyncPhase::RUNNABLE);
    }
    EXPECT_EQ(scheduler_async_lease_phase(f.group_lease().word), SchedulerAsyncPhase::TERMINAL);
    EXPECT_EQ(scheduler_async_pending::pending(f.join().word), 0u);
}

TEST(HbgAsyncControl, FinishPermitsAllSuccessorsAndCountsOnce) {
    AsyncFixture f(12);
    for (uint32_t task = 1; task < 12; ++task) {
        f.graph.executable(task, 0, {0});
    }
    const auto writer = f.start();
    f.register_counter(writer, 0);
    ASSERT_TRUE(scheduler_async_seal_execution(&f.workers[0], 0, 0, 0, f.executions[0]));
    ASSERT_TRUE(f.close());
    EXPECT_TRUE(f.check(1));
    ASSERT_TRUE(f.finish(2));
    for (uint32_t worker = 0; worker < 3; ++worker)
        scheduler_flush_completions(f.storage.run_control, f.local(worker));
    EXPECT_FALSE(f.finish(0));
    EXPECT_FALSE(f.finish(1));
    EXPECT_EQ(scheduler_async_lease_phase(f.finish_lease().word), SchedulerAsyncPhase::TERMINAL);
    EXPECT_EQ(f.storage.run_control->resolved_task_count, 1u);
    std::array<bool, 12> seen{};
    for (uint32_t count = 0; count < 11; ++count) {
        int64_t task = -1;
        uint32_t slot = UINT32_MAX;
        ASSERT_TRUE(f.select_execution(&task, &slot));
        ASSERT_GE(task, 1);
        ASSERT_LT(task, 12);
        EXPECT_FALSE(seen[task]);
        seen[task] = true;
        f.local()->slots[slot].state = SchedulerDispatchSlotState::EXECUTING;
        ASSERT_TRUE(scheduler_complete_own_slot(f.graph.graph(), f.base(), f.local(), f.storage.run_control, slot, 0));
    }
    scheduler_flush_completions(f.storage.run_control, f.local());
    EXPECT_EQ(f.storage.run_control->resolved_task_count, 12u);
    EXPECT_FALSE(f.finish());
    scheduler_flush_completions(f.storage.run_control, f.local());
    EXPECT_EQ(f.storage.run_control->resolved_task_count, 12u);
}

TEST(HbgAsyncControl, SignedCounterBoundariesRemainPendingBelowTheThreshold) {
    alignas(128) std::array<uint32_t, 32> words{};
    words[0] = 0x12345678;
    words[2] = 0x87654321;
    for (uint32_t value : {UINT32_C(0x80000000), UINT32_MAX, 0u, 1u, static_cast<uint32_t>(INT32_MAX)}) {
        words[1] = value;
        for (uint32_t expected : {0u, 1u, static_cast<uint32_t>(INT32_MAX)}) {
            const CompletionToken token{
                reinterpret_cast<uint64_t>(&words[1]), expected, COMPLETION_ENGINE_SDMA, COMPLETION_TYPE_COUNTER, 0
            };
            for (uint32_t observer_mask : {1u, 2u, 3u}) {
                const auto observed = SchedulerAsyncCompletionObserver{}(token, observer_mask);
                const bool ready = value <= INT32_MAX && value >= expected;
                EXPECT_EQ(observed.state, ready ? CompletionPollState::READY : CompletionPollState::PENDING);
                EXPECT_EQ(observed.error_code, 0);
                EXPECT_EQ(words[1], value);
                EXPECT_EQ(words[0], 0x12345678u);
                EXPECT_EQ(words[2], 0x87654321u);
            }
        }
    }
}

TEST(HbgAsyncControl, NegativeCounterHandsOffUntilPublicationWithoutFailingTheRun) {
    AsyncFixture f(1);
    f.signal[0] = UINT32_MAX;  // INT32 -1 is below the supported threshold zero.
    const auto writer = f.start();
    f.register_counter(writer, 0);
    ASSERT_TRUE(scheduler_async_seal_execution(&f.workers[0], 0, 0, 0, f.executions[0]));
    ASSERT_TRUE(f.close());
    for (uint32_t worker : {1u, 2u, 0u}) {
        bool progress = false;
        ASSERT_TRUE(scheduler_async_check_step(f.workers[worker], 1, 8, SchedulerAsyncCompletionObserver{}, progress));
        EXPECT_FALSE(progress);
        EXPECT_EQ(f.workers[worker].error_code, 0);
        EXPECT_EQ(scheduler_async_lease_phase(f.group_lease().word), SchedulerAsyncPhase::RUNNABLE);
        EXPECT_EQ(scheduler_async_pending::pending(f.join().word), 1u);
        EXPECT_EQ(f.finish_lease().word, kUnused);
    }
    f.signal[0] = 0;
    EXPECT_TRUE(f.check(2));
    EXPECT_EQ(scheduler_async_lease_phase(f.group_lease().word), SchedulerAsyncPhase::TERMINAL);
    EXPECT_EQ(scheduler_async_lease_phase(f.finish_lease().word), SchedulerAsyncPhase::RUNNABLE);
    EXPECT_EQ(f.signal[0], 0u);
}

TEST(HbgAsyncControl, CcuEventGatesResidentJoinUntilObservedAndCleared) {
    AsyncFixture f(1);
    alignas(64) uint64_t event = 0;
    const auto writer = f.start();
    const CompletionToken token{reinterpret_cast<uint64_t>(&event), 0, COMPLETION_ENGINE_CCU, COMPLETION_TYPE_CCU, 1};
    ASSERT_TRUE(register_completion_condition(writer, token));
    ASSERT_TRUE(scheduler_async_seal_execution(&f.workers[0], 0, 0, 0, f.executions[0]));
    ASSERT_TRUE(f.close());
    EXPECT_FALSE(f.check(1));
    EXPECT_EQ(scheduler_async_pending::pending(f.join().word), 1u);
    EXPECT_EQ(f.finish_lease().word, kUnused);
    event = 1;
    EXPECT_TRUE(f.check(2));
    EXPECT_EQ(event, 0u);
    EXPECT_EQ(scheduler_async_lease_phase(f.group_lease().word), SchedulerAsyncPhase::TERMINAL);
    EXPECT_EQ(scheduler_async_lease_phase(f.finish_lease().word), SchedulerAsyncPhase::RUNNABLE);
}

TEST(HbgAsyncControl, InvalidCounterAndMalformedRecordFailWithoutNormalCompletion) {
    for (uint32_t mode = 0; mode < 2; ++mode) {
        AsyncFixture f(1);
        const auto writer = f.start();
        f.register_counter(writer);
        auto *entry = scheduler_state_at<DeferredCompletionEntry>(
            f.base(), pto2::resident::detail::group(writer)->entries_offset
        );
        if (mode == 0) entry->expected_value = UINT32_MAX;
        else entry->backend_cookie = 1;
        ASSERT_TRUE(scheduler_async_seal_execution(&f.workers[0], 0, 0, 0, f.executions[0]));
        bool progress = false;
        EXPECT_FALSE(scheduler_async_check_step(f.workers[1], 2, 8, SchedulerAsyncCompletionObserver{}, progress));
        EXPECT_EQ(scheduler_async_lease_phase(f.group_lease().word), SchedulerAsyncPhase::FAILED);
        EXPECT_EQ(scheduler_async_pending::pending(f.join().word), 1u);
        EXPECT_EQ(f.finish_lease().word, kUnused);
    }
}

TEST(HbgAsyncControl, CounterAndFullWidthSdmaRecordsMustBothCompleteAndAreNeverCleared) {
    AsyncFixture f(1, {async_task(0, 1, LaneMask::aic(), 2)});
    alignas(128) uint64_t completed_post = UINT64_C(0xffffffff);
    const auto writer = f.start();
    f.register_counter(writer);
    const CompletionToken dma{
        reinterpret_cast<uint64_t>(&completed_post), 0, COMPLETION_ENGINE_SDMA, COMPLETION_TYPE_SDMA_EVENT_RECORD,
        UINT64_C(0x100000003)
    };
    ASSERT_TRUE(register_completion_condition(writer, dma));
    ASSERT_TRUE(scheduler_async_seal_execution(&f.workers[0], 0, 0, 0, f.executions[0]));
    ASSERT_TRUE(f.close());
    EXPECT_FALSE(f.check(1));
    f.signal[0] = 1;
    EXPECT_TRUE(f.check(2));
    EXPECT_EQ(scheduler_async_pending::pending(f.join().word), 1u);
    EXPECT_EQ(completed_post, UINT64_C(0xffffffff));
    completed_post = UINT64_C(0x100000003);
    EXPECT_TRUE(f.check(0));
    EXPECT_EQ(scheduler_async_lease_phase(f.group_lease().word), SchedulerAsyncPhase::TERMINAL);
    EXPECT_EQ(scheduler_async_lease_phase(f.finish_lease().word), SchedulerAsyncPhase::RUNNABLE);
    EXPECT_EQ(completed_post, UINT64_C(0x100000003));
}

TEST(HbgAsyncControl, OneSliceClaimsOnlyOneRecordAndAlternatesCheckAndFinish) {
    AsyncFixture f(1);
    const auto writer = f.start();
    f.register_counter(writer, 0);
    ASSERT_TRUE(scheduler_async_seal_execution(&f.workers[0], 0, 0, 0, f.executions[0]));
    ASSERT_TRUE(f.close());
    f.workers[0].business_returns = 4;
    bool progress = false;
    ASSERT_TRUE(
        scheduler_service_async_slice(f.graph.graph(), f.base(), f.local(), f.storage.run_control, 0, progress)
    );
    EXPECT_TRUE(progress);
    EXPECT_EQ(f.workers[0].business_returns, 0u);
    EXPECT_TRUE(f.workers[0].finish_first);
    EXPECT_EQ(scheduler_async_lease_phase(f.finish_lease().word), SchedulerAsyncPhase::RUNNABLE);
    EXPECT_EQ(
        scheduler_completion_task_state(f.task_control(0)->completion_state),
        static_cast<int64_t>(SchedulerTaskState::BLOCKED)
    );
    EXPECT_EQ(f.local()->pending_completed, 0u);
    ASSERT_TRUE(
        scheduler_service_async_slice(f.graph.graph(), f.base(), f.local(), f.storage.run_control, 0, progress)
    );
    EXPECT_TRUE(progress);
    EXPECT_FALSE(f.workers[0].finish_first);
    EXPECT_EQ(
        scheduler_completion_task_state(f.task_control(0)->completion_state),
        static_cast<int64_t>(SchedulerTaskState::DONE)
    );
    EXPECT_EQ(f.local()->pending_completed, 1u);
}

TEST(HbgAsyncControl, ObserverFailureLatchesRunErrorAndOtherCoresCannotFinishNormally) {
    AsyncFixture f(1);
    const auto writer = f.start();
    f.register_counter(writer);
    ASSERT_TRUE(scheduler_async_seal_execution(&f.workers[0], 0, 0, 0, f.executions[0]));
    ASSERT_TRUE(f.close());
    auto *entry =
        scheduler_state_at<DeferredCompletionEntry>(f.base(), pto2::resident::detail::group(writer)->entries_offset);
    entry->expected_value = UINT32_MAX;
    bool progress = false;
    EXPECT_FALSE(
        scheduler_service_async_slice(f.graph.graph(), f.base(), f.local(1), f.storage.run_control, 0, progress)
    );
    EXPECT_EQ(f.storage.run_control->error_site, static_cast<uint64_t>(SchedulerErrorSite::ASYNC_CHECK_FAILED));
    EXPECT_NE(scheduler_read_error(f.storage.run_control), 0u);
    EXPECT_FALSE(
        scheduler_service_async_slice(f.graph.graph(), f.base(), f.local(2), f.storage.run_control, 0, progress)
    );
    EXPECT_EQ(f.storage.run_control->resolved_task_count, 0u);
    EXPECT_EQ(scheduler_async_pending::pending(f.join().word), 1u);
}

TEST(HbgAsyncControl, ConcurrentObserversAndBlockReleaseGrantOneFinish) {
    AsyncFixture f(1, {async_task(0, 16, LaneMask::all())});
    for (uint32_t block = 0; block < 16; ++block) {
        for (uint32_t lane = 0; lane < 3; ++lane) {
            const auto writer = f.start(0, block, lane);
            f.register_counter(writer);
            ASSERT_TRUE(scheduler_async_seal_execution(&f.workers[0], 0, block, lane, f.executions[0]));
        }
    }
    SchedulerAsyncLocalState producer;
    ASSERT_TRUE(scheduler_initialize_async(
        f.base(), f.storage.layout.total_size, f.plan.layout_offset, 1, 1, f.storage.layout.task_controls_offset,
        producer
    ));
    SchedulerLocalState producer_context = *f.local();
    producer_context.async = &producer;
    std::atomic<bool> begin{false};
    std::atomic<bool> failed{false};
    std::vector<std::thread> observers;
    for (uint32_t worker = 0; worker < 3; ++worker) {
        auto *local = f.local(worker);
        observers.emplace_back([&, local] {
            while (!begin.load(std::memory_order_acquire)) {}
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            while (!failed.load(std::memory_order_acquire) &&
                   scheduler_async_lease_phase(scheduler_async_lease_load(f.finish_lease())) !=
                       SchedulerAsyncPhase::TERMINAL) {
                bool progress = false;
                if (!scheduler_service_async_slice(
                        f.graph.graph(), f.base(), local, f.storage.run_control, 0, progress
                    ) ||
                    std::chrono::steady_clock::now() >= deadline) {
                    failed.store(true, std::memory_order_release);
                    break;
                }
            }
        });
    }
    begin.store(true, std::memory_order_release);
    bool deferred = false;
    if (!f.return_remaining(0, &producer_context) || !scheduler_async_close_computation(&producer, 0, deferred) ||
        !deferred)
        failed.store(true, std::memory_order_release);
    __atomic_store_n(&f.signal[0], 1u, __ATOMIC_RELEASE);
    for (auto &observer : observers)
        observer.join();
    EXPECT_FALSE(failed.load());
    EXPECT_TRUE(scheduler_async_pending::ready_to_finish(f.join().word));
    EXPECT_EQ(scheduler_read_error(f.storage.run_control), 0u);
    for (uint32_t worker = 0; worker < 3; ++worker)
        scheduler_flush_completions(f.storage.run_control, f.local(worker));
    EXPECT_EQ(f.storage.run_control->resolved_task_count, 1u);
    EXPECT_EQ(scheduler_async_lease_phase(f.finish_lease().word), SchedulerAsyncPhase::TERMINAL);
}

}  // namespace
