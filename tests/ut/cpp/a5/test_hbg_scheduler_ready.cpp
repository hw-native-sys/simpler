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
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

#include "scheduler/scheduler_ready.h"
#include "runtime_types.h"

namespace {

class SchedulerStateBuffer {
public:
    explicit SchedulerStateBuffer(const AicoreSchedulerLayout &layout) :
        base_(std::aligned_alloc(SCHEDULER_STATE_ALIGNMENT, layout.total_size)) {
        EXPECT_NE(base_, nullptr);
        if (base_ != nullptr) EXPECT_TRUE(scheduler_init_data_from_layout(base_, layout));
    }
    ~SchedulerStateBuffer() { std::free(base_); }
    void *base() const { return base_; }

private:
    void *base_{nullptr};
};

class GraphBuffer {
public:
    explicit GraphBuffer(size_t task_count) :
        task_count_(task_count),
        image_(std::make_unique<GraphImage>()) {
        while (capacity_ < std::max<size_t>(task_count, 1))
            capacity_ <<= 1;
        if (capacity_ > kMaxTaskCount) throw std::invalid_argument("test graph exceeds GraphBuffer capacity");
        storage_ = image_->storage.data();
        fanins_ = image_->fanins.data();
        for (size_t task = 0; task < capacity_; ++task) {
            storage_[task].task.task_id = TaskId{static_cast<uint64_t>(task)};
            storage_[task].payload.bind_regions(
                nullptr, nullptr, fanins_ + task * static_cast<size_t>(SCHEDULER_GRAPH_MAX_FANIN)
            );
            if (storage_[task].payload.fanin_data() == nullptr) {
                throw std::logic_error("test graph fanin region must share its contiguous image");
            }
            for (int slot = 0; slot < 3; ++slot)
                storage_[task].task.kernel_id[slot] = INVALID_KERNEL_ID;
        }
    }

    void executable(size_t task, uint8_t subtask_slot, std::vector<int32_t> fanins = {}) {
        ASSERT_LT(task, task_count_);
        ASSERT_LT(subtask_slot, 3);
        ASSERT_LE(fanins.size(), static_cast<size_t>(SCHEDULER_GRAPH_MAX_FANIN));
        storage_[task].task.kernel_id[subtask_slot] = 1;
        storage_[task].payload.fanin_count = static_cast<int32_t>(fanins.size());
        ASSERT_TRUE(fanins.empty() || storage_[task].payload.fanin_data() != nullptr);
        std::copy(fanins.begin(), fanins.end(), storage_[task].payload.fanin_data());
    }

    void mixed(size_t task, uint8_t active_mask) {
        ASSERT_LT(task, task_count_);
        for (uint8_t subtask_slot = 0; subtask_slot < 3; ++subtask_slot) {
            if ((active_mask & (1U << subtask_slot)) != 0) storage_[task].task.kernel_id[subtask_slot] = 1;
        }
        storage_[task].payload.fanin_count = 0;
    }

    void predicate(size_t task, uint64_t addr, uint8_t elem_size, uint8_t op, int64_t target = 0) {
        ASSERT_LT(task, task_count_);
        storage_[task].payload.predicate.addr = addr;
        storage_[task].payload.predicate.target = target;
        storage_[task].payload.predicate.elem_size = elem_size;
        storage_[task].payload.predicate.op = static_cast<PredicateOp>(op);
    }

    SchedulerGraphView graph() const {
        return {
            reinterpret_cast<uint64_t>(storage_),
            0,
            task_count_,
            capacity_ - 1,
        };
    }

private:
    static constexpr size_t kMaxTaskCount = 8192;
    // One storage array, as production has it — see the same note in
    // test_hbg_scheduler_contracts.cpp.
    struct alignas(64) GraphImage {
        std::array<ChipTaskStorage, kMaxTaskCount> storage{};
        std::array<int32_t, kMaxTaskCount * SCHEDULER_GRAPH_MAX_FANIN> fanins{};
    };

    size_t task_count_;
    size_t capacity_{1};
    std::unique_ptr<GraphImage> image_;
    ChipTaskStorage *storage_{nullptr};
    int32_t *fanins_{nullptr};
};

struct FixtureStorage {
    explicit FixtureStorage(uint64_t task_count, uint64_t workers = 2) {
        EXPECT_TRUE(scheduler_plan_layout(task_count, task_count, 0, &layout));
        scheduler_state = std::make_unique<SchedulerStateBuffer>(layout);
        run_control = scheduler_state_at<SchedulerRunControl>(scheduler_state->base(), layout.run_control_offset);
        contexts = scheduler_state_at<SchedulerWorkerContext>(scheduler_state->base(), layout.worker_contexts_offset);
        owner_states =
            scheduler_state_at<SchedulerReadyOwnerState>(scheduler_state->base(), layout.ready_owner_states_offset);
        run_control->aiv_active_worker_count = workers;
        run_control->scheduler_count = workers;
        for (uint64_t worker = 0; worker < workers; ++worker) {
            SchedulerWorkerContext &context = contexts[worker];
            context.core_type = static_cast<int32_t>(CoreType::AIV);
            context.active = 1;
            context.task_controls_offset = layout.task_controls_offset;
            context.task_metadata_offset = layout.task_metadata_offset;
            context.completion_inboxes_offset = layout.completion_inboxes_offset;
            context.ready_inboxes_offset = layout.ready_inboxes_offset;
            context.ready_owner_states_offset = layout.ready_owner_states_offset;
            context.ready_directory_offset = layout.ready_directory_offset;
            context.trace_cells_offset = layout.trace_cells_offset;
            context.worker_contexts_offset = layout.worker_contexts_offset;
            context.dispatch_slots_offset = layout.dispatch_slots_offset;
            context.callable_addresses_offset = layout.callable_addresses_offset;
            context.gang_coordinator_offset = layout.gang_coordinator_offset;
            context.gang_cohorts_offset = layout.gang_cohorts_offset;
            context.gang_participants_offset = layout.gang_participants_offset;
            context.gang_commands_offset = layout.gang_commands_offset;
            context.dispatch_payload_offset =
                layout.dispatch_payloads_offset + worker * SCHEDULER_PENDING_SLOT_COUNT * sizeof(DispatchPayload);
            context.graph_task_count = task_count;
            context.runtime_worker_count = workers;
            context.worker_index = worker;
            context.inbox_index = worker;
            scheduler_ready_owner_init(&owner_states[worker]);
        }
        metadata = scheduler_state_at<SchedulerTaskMetadata>(scheduler_state->base(), layout.task_metadata_offset);
        callable_addresses = scheduler_state_at<uint64_t>(scheduler_state->base(), layout.callable_addresses_offset);
        callable_addresses[1] = UINT64_C(0x1000);
        for (uint64_t task = 0; task < task_count; ++task) {
            metadata[task].kernel_ids[0] = 1;
            metadata[task].kernel_ids[1] = UINT16_MAX;
            metadata[task].kernel_ids[2] = UINT16_MAX;
            metadata[task].active_mask = 1;
            metadata[task].logical_block_num = 1;
            metadata[task].total_required_subtasks = 1;
            metadata[task].flags = SCHEDULER_TASK_EXECUTABLE;
        }
    }

    AicoreSchedulerLayout layout{};
    std::unique_ptr<SchedulerStateBuffer> scheduler_state;
    SchedulerRunControl *run_control{nullptr};
    SchedulerWorkerContext *contexts{nullptr};
    SchedulerReadyOwnerState *owner_states{nullptr};
    SchedulerTaskMetadata *metadata{nullptr};
    uint64_t *callable_addresses{nullptr};
};

TEST(SchedulerActivityBuffer, IsAllocatedOnlyWhenRequestedAndNeverWraps) {
    AicoreSchedulerLayout disabled{};
    ASSERT_TRUE(scheduler_plan_layout(1, 1, 0, &disabled));
    EXPECT_EQ(disabled.activity_buffers_offset, 0u);

    AicoreSchedulerLayout enabled{};
    ASSERT_TRUE(scheduler_plan_layout(1, 1, 0, &enabled, true));
    ASSERT_NE(enabled.activity_buffers_offset, 0u);
    EXPECT_EQ(
        enabled.total_size - disabled.total_size,
        static_cast<uint64_t>(SCHEDULER_CLUSTER_CAPACITY) * sizeof(SchedulerActivityBuffer)
    );
    SchedulerStateBuffer storage(enabled);
    auto *contexts = scheduler_state_at<SchedulerWorkerContext>(storage.base(), enabled.worker_contexts_offset);
    auto *buffers = scheduler_state_at<SchedulerActivityBuffer>(storage.base(), enabled.activity_buffers_offset);
    contexts[0].worker_index = SCHEDULER_WORKER_CAPACITY - 1;
    contexts[0].is_scheduler = 1;
    contexts[0].scheduler_index = SCHEDULER_CLUSTER_CAPACITY - 1;
    contexts[0].profiling_loop_iter = 17;
    SchedulerActivityBuffer &buffer = buffers[SCHEDULER_CLUSTER_CAPACITY - 1];
    buffer.committed = SCHEDULER_ACTIVITY_CAPACITY - 1;

    scheduler_append_idle_activity(storage.base(), &contexts[0], 10, 20);
    scheduler_append_idle_activity(storage.base(), &contexts[0], 30, 40);

    EXPECT_EQ(buffer.committed, SCHEDULER_ACTIVITY_CAPACITY);
    EXPECT_EQ(buffer.dropped, 1u);
    const SchedulerIdleRecord &last = buffer.records[SCHEDULER_ACTIVITY_CAPACITY - 1];
    EXPECT_EQ(last.start_time, 10u);
    EXPECT_EQ(last.end_time, 20u);
    EXPECT_EQ(last.loop_iter, 17u);
}

TEST(SchedulerActivityBuffer, RejectsCorruptCommittedCountOnHost) {
    EXPECT_TRUE(scheduler_activity_record_count_valid(SCHEDULER_ACTIVITY_CAPACITY));
    EXPECT_FALSE(scheduler_activity_record_count_valid(SCHEDULER_ACTIVITY_CAPACITY + 1));
}

TEST(SchedulerProfilingLevel, EnablesOnlyTheRequestedGranularity) {
    EXPECT_FALSE(scheduler_task_timing_enabled(0));
    EXPECT_TRUE(scheduler_task_timing_enabled(SCHEDULER_PROFILING_TASK_TIMING_LEVEL));
    EXPECT_FALSE(scheduler_schedule_timing_enabled(SCHEDULER_PROFILING_TASK_TIMING_LEVEL));
    EXPECT_TRUE(scheduler_schedule_timing_enabled(SCHEDULER_PROFILING_SCHEDULE_TIMING_LEVEL));
    EXPECT_FALSE(scheduler_phase_timing_enabled(SCHEDULER_PROFILING_SCHEDULE_TIMING_LEVEL));
    EXPECT_TRUE(scheduler_phase_timing_enabled(SCHEDULER_PROFILING_SCHED_PHASES_LEVEL));
}

TEST(SchedulerProfilingLevel, DispatchWritesTaskIdentityBeforePhaseDetails) {
    for (uint64_t level = 0; level <= SCHEDULER_PROFILING_SCHED_PHASES_LEVEL; ++level) {
        FixtureStorage storage(1, 2);
        GraphBuffer graph(1);
        graph.executable(0, 0);
        storage.contexts[1].core_type = static_cast<int32_t>(CoreType::AIC);
        SchedulerReadyClaim ready_claim{};
        ready_claim.task_id = 0;
        ready_claim.claim_start_cycles = 123;
        ready_claim.claim_end_cycles = 456;

        ASSERT_TRUE(scheduler_fill_dispatch_slot(
            graph.graph(), storage.scheduler_state->base(), &storage.contexts[1], storage.run_control,
            SchedulerFreeSlotClaim{1, 0, 0}, ready_claim, level
        ));
        auto *traces =
            scheduler_state_at<SchedulerTaskTrace>(storage.scheduler_state->base(), storage.layout.trace_cells_offset);
        EXPECT_EQ(traces[0].worker_id, level == 0 ? 0u : storage.contexts[1].worker_index);
        EXPECT_EQ(traces[0].task_id, 0u);
        EXPECT_EQ(traces[0].claim_start_cycles, level >= SCHEDULER_PROFILING_SCHED_PHASES_LEVEL ? 123u : 0u);
        EXPECT_EQ(traces[0].claim_end_cycles, level >= SCHEDULER_PROFILING_SCHED_PHASES_LEVEL ? 456u : 0u);
    }
}

TEST(SchedulerBootstrap, RegistersOnlyOnFirstExecutableProducer) {
    FixtureStorage storage(4, 2);
    GraphBuffer graph(4);
    graph.executable(0, 0);
    graph.executable(1, 1, {0});
    graph.executable(3, 1, {2, 1});
    storage.metadata[1].active_mask = 2;
    storage.metadata[1].flags |= SCHEDULER_TASK_HAS_FANIN;
    storage.metadata[2].flags = 0;
    storage.metadata[3].active_mask = 2;
    storage.metadata[3].flags |= SCHEDULER_TASK_HAS_FANIN;
    auto *controls =
        scheduler_state_at<SchedulerTaskControl>(storage.scheduler_state->base(), storage.layout.task_controls_offset);
    controls[2].state = static_cast<int64_t>(SchedulerTaskState::DONE);
    controls[2].wake_list_head = SCHEDULER_WAKE_LIST_CLOSED;

    SchedulerWakeStats stats{};
    EXPECT_EQ(
        scheduler_bootstrap_route_task(
            graph.graph(), storage.scheduler_state->base(), &storage.contexts[0], storage.run_control, 1, &stats
        ),
        SchedulerRouteResult::WAITING
    );
    EXPECT_EQ(controls[0].wake_list_head, 1);
    EXPECT_EQ(controls[1].next_waiter, SCHEDULER_WAKE_LIST_OPEN);
    EXPECT_EQ(controls[1].waiting_producer, 0);

    EXPECT_EQ(
        scheduler_bootstrap_route_task(
            graph.graph(), storage.scheduler_state->base(), &storage.contexts[1], storage.run_control, 3, &stats
        ),
        SchedulerRouteResult::WAITING
    );
    EXPECT_EQ(controls[1].wake_list_head, 3);
    EXPECT_EQ(controls[3].next_fanin_index, 1);
    EXPECT_EQ(controls[3].waiting_producer, 1);
    EXPECT_EQ(stats.wake_register_count, 2u);
    EXPECT_EQ(stats.fanin_state_load_count, 0u);
    EXPECT_EQ(stats.wake_cas_retry_count, 0u);
}

TEST(SchedulerBootstrap, PublishesExclusiveInboxAndAggregatesDirectory) {
    FixtureStorage storage(2, 2);
    GraphBuffer graph(2);
    graph.executable(0, 0);
    graph.executable(1, 0);
    storage.contexts[0].inbox_index = 1;
    storage.contexts[1].inbox_index = 0;
    SchedulerReadyBatch batch{};
    SchedulerReadyStats stats{};
    ASSERT_TRUE(
        scheduler_bootstrap_ready_batch_append(storage.scheduler_state->base(), &storage.contexts[1], 0, &batch, &stats)
    );
    ASSERT_TRUE(
        scheduler_bootstrap_ready_batch_append(storage.scheduler_state->base(), &storage.contexts[1], 1, &batch, &stats)
    );
    uint64_t ready_types = 0;
    ASSERT_TRUE(scheduler_bootstrap_ready_batch_publish(
        storage.scheduler_state->base(), &storage.contexts[1], 0, 0, &batch, &stats, &ready_types
    ));
    auto *directory = scheduler_ready_directory_at(storage.scheduler_state->base(), &storage.contexts[1]);
    directory->bootstrap_ready_types[0] = ready_types;
    ASSERT_TRUE(scheduler_bootstrap_ready_directory_publish(storage.scheduler_state->base(), &storage.contexts[1], 2));

    auto *controls =
        scheduler_state_at<SchedulerTaskControl>(storage.scheduler_state->base(), storage.layout.task_controls_offset);
    EXPECT_EQ(controls[0].next_waiter, 1);
    EXPECT_EQ(controls[1].next_waiter, SCHEDULER_INBOX_EMPTY);
    EXPECT_EQ(scheduler_ready_inbox_at(storage.scheduler_state->base(), &storage.contexts[1], 0, 0)->head, 0);
    EXPECT_EQ(directory->core_types[0][0].bits, 1u);
    EXPECT_EQ(directory->core_types[1][0].bits, 0u);
    EXPECT_EQ(stats.enqueue_count, 2u);
    EXPECT_EQ(stats.batch_count, 1u);
}

TEST(SchedulerReadyInbox, RejectsSchedulerCapacityBoundary) {
    FixtureStorage storage(1, 1);
    SchedulerReadyStats stats{};
    SchedulerReadyBatch batch{};
    ASSERT_TRUE(scheduler_ready_batch_append(storage.scheduler_state->base(), &storage.contexts[0], 0, &batch, &stats));
    uint64_t ready_types = 0;

    EXPECT_FALSE(scheduler_bootstrap_ready_batch_publish(
        storage.scheduler_state->base(), &storage.contexts[0], 0, SCHEDULER_CAPACITY, &batch, &stats, &ready_types
    ));
    storage.contexts[0].inbox_index = SCHEDULER_CAPACITY;
    EXPECT_FALSE(scheduler_ready_owner_maintain_type(
        storage.scheduler_state->base(), &storage.contexts[0], 0, &storage.owner_states[0]
    ));
    EXPECT_FALSE(scheduler_ready_batch_push(
        storage.scheduler_state->base(), &storage.contexts[0], 0, SCHEDULER_CAPACITY, &batch, &stats,
        &storage.owner_states[0]
    ));
    storage.contexts[0].inbox_index = 0;
    EXPECT_FALSE(scheduler_bootstrap_ready_directory_publish(
        storage.scheduler_state->base(), &storage.contexts[0], SCHEDULER_CAPACITY + 1
    ));
}

TEST(SchedulerReadyInbox, RequiresOwnerStateForPublishAndClaim) {
    FixtureStorage storage(1, 1);
    GraphBuffer graph(1);
    graph.executable(0, 0);
    SchedulerReadyStats stats{};
    SchedulerReadyBatch batch{};
    ASSERT_TRUE(scheduler_ready_batch_append(storage.scheduler_state->base(), &storage.contexts[0], 0, &batch, &stats));

    EXPECT_FALSE(
        scheduler_ready_batch_push(storage.scheduler_state->base(), &storage.contexts[0], 0, 0, &batch, &stats, nullptr)
    );
    uint64_t cursor = 0;
    SchedulerReadyClaim claim{};
    EXPECT_FALSE(scheduler_claim_ready_for_slot(
        graph.graph(), storage.scheduler_state->base(), &storage.contexts[0], storage.run_control, 1, 0, &cursor,
        &stats, &claim, nullptr
    ));
    EXPECT_FALSE(scheduler_resolve_completion(
        graph.graph(), storage.scheduler_state->base(), &storage.contexts[0], storage.run_control, 0, nullptr, nullptr,
        nullptr, nullptr
    ));
}

TEST(SchedulerReadyInbox, BatchPushAndOwnerMaintenancePreserveFifoAndDirectory) {
    constexpr uint64_t kTasks = 4;
    FixtureStorage storage(kTasks, 1);
    GraphBuffer graph(kTasks);
    for (uint64_t task = 0; task < kTasks; ++task)
        graph.executable(task, 0);
    SchedulerReadyBatch batch{};
    SchedulerReadyOwnerState &owner_state = storage.owner_states[0];
    SchedulerReadyStats stats{};
    for (uint64_t task = 0; task < kTasks; ++task)
        ASSERT_TRUE(
            scheduler_ready_batch_append(storage.scheduler_state->base(), &storage.contexts[0], task, &batch, &stats)
        );
    ASSERT_TRUE(scheduler_ready_batch_push(
        storage.scheduler_state->base(), &storage.contexts[0], 0, 0, &batch, &stats, &owner_state
    ));

    auto *directory = scheduler_state_at<SchedulerReadyDirectory>(
        storage.scheduler_state->base(), storage.layout.ready_directory_offset
    );
    EXPECT_NE(directory->core_types[0][0].bits & 1, 0u);
    for (uint64_t index = 0; index < kTasks; ++index) {
        int64_t task = SCHEDULER_TASK_ID_INVALID;
        ASSERT_TRUE(scheduler_ready_pop_from_inbox(
            graph.graph(), storage.scheduler_state->base(), &storage.contexts[0], storage.run_control, 0, 0, &task,
            &stats
        ));
        EXPECT_EQ(task, static_cast<int64_t>(index));
    }
    int64_t task = SCHEDULER_TASK_ID_INVALID;
    ASSERT_TRUE(scheduler_ready_pop_from_inbox(
        graph.graph(), storage.scheduler_state->base(), &storage.contexts[0], storage.run_control, 0, 0, &task, &stats
    ));
    EXPECT_EQ(task, SCHEDULER_TASK_ID_INVALID);
    EXPECT_NE(directory->core_types[0][0].bits & 1, 0u);
    ASSERT_TRUE(
        scheduler_ready_owner_maintain_type(storage.scheduler_state->base(), &storage.contexts[0], 0, &owner_state)
    );
    EXPECT_EQ(directory->core_types[0][0].bits & 1, 0u);
    EXPECT_EQ(stats.pop_count, kTasks);
}

TEST(SchedulerReadyInbox, OwnerStateInitializationRestoresEmptySentinels) {
    SchedulerReadyOwnerState owner_state;
    __builtin_memset(&owner_state, 0, sizeof(owner_state));

    scheduler_ready_owner_init(&owner_state);

    for (uint32_t type = 0; type < SCHEDULER_CORE_TYPE_COUNT; ++type) {
        EXPECT_EQ(owner_state.queues[type].pending_endpoints, SCHEDULER_READY_PENDING_EMPTY);
        EXPECT_EQ(owner_state.queues[type].advertised, 0u);
    }
}

TEST(SchedulerReadyInbox, PackedOwnerEndpointsRoundTripAsOneWord) {
    EXPECT_EQ(scheduler_ready_pending_pack(SCHEDULER_INBOX_EMPTY, SCHEDULER_INBOX_EMPTY), UINT64_MAX);
    const uint64_t endpoints = scheduler_ready_pending_pack(INT32_MAX - 1, INT32_MAX);
    EXPECT_EQ(scheduler_ready_pending_head(endpoints), INT32_MAX - 1);
    EXPECT_EQ(scheduler_ready_pending_tail(endpoints), INT32_MAX);
}

TEST(SchedulerReadyInbox, OwnerPromotesPendingBankAfterPublishedBankDrains) {
    constexpr uint64_t kTasks = 4;
    FixtureStorage storage(kTasks, 1);
    GraphBuffer graph(kTasks);
    for (uint64_t task = 0; task < kTasks; ++task)
        graph.executable(task, 0);
    SchedulerReadyOwnerState &owner_state = storage.owner_states[0];
    SchedulerReadyStats stats{};
    SchedulerReadyBatch published{};
    SchedulerReadyBatch pending{};
    for (int64_t task = 0; task < 2; ++task)
        ASSERT_TRUE(scheduler_ready_batch_append(
            storage.scheduler_state->base(), &storage.contexts[0], task, &published, &stats
        ));
    for (int64_t task = 2; task < 4; ++task)
        ASSERT_TRUE(
            scheduler_ready_batch_append(storage.scheduler_state->base(), &storage.contexts[0], task, &pending, &stats)
        );
    ASSERT_TRUE(scheduler_ready_batch_push(
        storage.scheduler_state->base(), &storage.contexts[0], 0, 0, &published, &stats, &owner_state
    ));
    ASSERT_TRUE(scheduler_ready_batch_push(
        storage.scheduler_state->base(), &storage.contexts[0], 0, 0, &pending, &stats, &owner_state
    ));
    const uint64_t endpoints = owner_state.queues[0].pending_endpoints;
    EXPECT_EQ(scheduler_ready_pending_head(endpoints), 2);
    EXPECT_EQ(scheduler_ready_pending_tail(endpoints), 3);

    for (int64_t expected = 0; expected < 2; ++expected) {
        int64_t task = SCHEDULER_TASK_ID_INVALID;
        ASSERT_TRUE(scheduler_ready_pop_from_inbox(
            graph.graph(), storage.scheduler_state->base(), &storage.contexts[0], storage.run_control, 0, 0, &task,
            &stats
        ));
        EXPECT_EQ(task, expected);
    }
    int64_t task = SCHEDULER_TASK_ID_INVALID;
    ASSERT_TRUE(scheduler_ready_pop_from_inbox(
        graph.graph(), storage.scheduler_state->base(), &storage.contexts[0], storage.run_control, 0, 0, &task, &stats
    ));
    EXPECT_EQ(task, SCHEDULER_TASK_ID_INVALID);
    ASSERT_TRUE(
        scheduler_ready_owner_maintain_type(storage.scheduler_state->base(), &storage.contexts[0], 0, &owner_state)
    );
    for (int64_t expected = 2; expected < 4; ++expected) {
        ASSERT_TRUE(scheduler_ready_pop_from_inbox(
            graph.graph(), storage.scheduler_state->base(), &storage.contexts[0], storage.run_control, 0, 0, &task,
            &stats
        ));
        EXPECT_EQ(task, expected);
    }
}

TEST(SchedulerReadyInbox, OlderPendingBankPrecedesBatchArrivingAfterDrain) {
    constexpr uint64_t kTasks = 3;
    FixtureStorage storage(kTasks, 1);
    GraphBuffer graph(kTasks);
    for (uint64_t task = 0; task < kTasks; ++task)
        graph.executable(task, 0);
    SchedulerReadyOwnerState &owner_state = storage.owner_states[0];
    SchedulerReadyStats stats{};
    for (int64_t task = 0; task < 2; ++task) {
        SchedulerReadyBatch batch{};
        ASSERT_TRUE(
            scheduler_ready_batch_append(storage.scheduler_state->base(), &storage.contexts[0], task, &batch, &stats)
        );
        ASSERT_TRUE(scheduler_ready_batch_push(
            storage.scheduler_state->base(), &storage.contexts[0], 0, 0, &batch, &stats, &owner_state
        ));
    }
    int64_t task = SCHEDULER_TASK_ID_INVALID;
    ASSERT_TRUE(scheduler_ready_pop_from_inbox(
        graph.graph(), storage.scheduler_state->base(), &storage.contexts[0], storage.run_control, 0, 0, &task, &stats
    ));
    ASSERT_EQ(task, 0);

    SchedulerReadyBatch arriving{};
    ASSERT_TRUE(
        scheduler_ready_batch_append(storage.scheduler_state->base(), &storage.contexts[0], 2, &arriving, &stats)
    );
    ASSERT_TRUE(scheduler_ready_batch_push(
        storage.scheduler_state->base(), &storage.contexts[0], 0, 0, &arriving, &stats, &owner_state
    ));
    EXPECT_EQ(scheduler_ready_pending_head(owner_state.queues[0].pending_endpoints), 2);
    ASSERT_TRUE(scheduler_ready_pop_from_inbox(
        graph.graph(), storage.scheduler_state->base(), &storage.contexts[0], storage.run_control, 0, 0, &task, &stats
    ));
    EXPECT_EQ(task, 1);
    ASSERT_TRUE(
        scheduler_ready_owner_maintain_type(storage.scheduler_state->base(), &storage.contexts[0], 0, &owner_state)
    );
    ASSERT_TRUE(scheduler_ready_pop_from_inbox(
        graph.graph(), storage.scheduler_state->base(), &storage.contexts[0], storage.run_control, 0, 0, &task, &stats
    ));
    EXPECT_EQ(task, 2);
}

TEST(SchedulerReadyInbox, ThiefCannotObserveOrPromoteOwnerPendingBank) {
    FixtureStorage storage(2, 2);
    GraphBuffer graph(2);
    graph.executable(0, 0);
    graph.executable(1, 0);
    SchedulerReadyOwnerState &owner_state = storage.owner_states[1];
    SchedulerReadyStats stats{};
    for (int64_t task = 0; task < 2; ++task) {
        SchedulerReadyBatch batch{};
        ASSERT_TRUE(
            scheduler_ready_batch_append(storage.scheduler_state->base(), &storage.contexts[1], task, &batch, &stats)
        );
        ASSERT_TRUE(scheduler_ready_batch_push(
            storage.scheduler_state->base(), &storage.contexts[1], 0, 1, &batch, &stats, &owner_state
        ));
    }
    int64_t task = SCHEDULER_TASK_ID_INVALID;
    ASSERT_TRUE(scheduler_ready_pop_from_inbox(
        graph.graph(), storage.scheduler_state->base(), &storage.contexts[0], storage.run_control, 0, 1, &task, &stats
    ));
    ASSERT_EQ(task, 0);
    ASSERT_TRUE(scheduler_ready_pop_from_inbox(
        graph.graph(), storage.scheduler_state->base(), &storage.contexts[0], storage.run_control, 0, 1, &task, &stats
    ));
    EXPECT_EQ(task, SCHEDULER_TASK_ID_INVALID);
    EXPECT_EQ(scheduler_ready_pending_head(owner_state.queues[0].pending_endpoints), 1);
    ASSERT_TRUE(
        scheduler_ready_owner_maintain_type(storage.scheduler_state->base(), &storage.contexts[1], 0, &owner_state)
    );
    ASSERT_TRUE(scheduler_ready_pop_from_inbox(
        graph.graph(), storage.scheduler_state->base(), &storage.contexts[0], storage.run_control, 0, 1, &task, &stats
    ));
    EXPECT_EQ(task, 1);
}

TEST(SchedulerReadyInbox, StealsOnlyFromMarkedVictim) {
    FixtureStorage storage(1, 2);
    GraphBuffer graph(1);
    graph.executable(0, 0);
    SchedulerReadyBatch batch{};
    SchedulerReadyStats stats{};
    ASSERT_TRUE(scheduler_ready_batch_append(storage.scheduler_state->base(), &storage.contexts[1], 0, &batch, &stats));
    ASSERT_TRUE(scheduler_ready_batch_push(
        storage.scheduler_state->base(), &storage.contexts[1], 0, 1, &batch, &stats, &storage.owner_states[1]
    ));

    uint64_t cursor = 1;
    SchedulerReadyClaim claim{};
    ASSERT_TRUE(scheduler_claim_ready_for_slot(
        graph.graph(), storage.scheduler_state->base(), &storage.contexts[0], storage.run_control, 2, 0, &cursor,
        &stats, &claim, &storage.owner_states[0]
    ));
    EXPECT_EQ(claim.task_id, 0);
    EXPECT_EQ(claim.inbox_index, 1u);
    EXPECT_EQ(claim.source, SchedulerReadySource::STOLEN);
    EXPECT_EQ(stats.steal_count, 1u);
}

TEST(SchedulerReadyInbox, DirectoryShardIgnoresSchedulerTail) {
    FixtureStorage storage(1, 9);
    auto *directory = scheduler_ready_directory_at(storage.scheduler_state->base(), &storage.contexts[0]);
    directory->core_types[0][1].bits = UINT64_C(1) << 6;
    EXPECT_EQ(scheduler_load_ready_directory_shard(directory, 9, 0, 7), 0u);

    directory->core_types[0][1].bits = UINT64_C(1) << 1;
    EXPECT_EQ(scheduler_load_ready_directory_shard(directory, 9, 0, 7), UINT64_C(1) << 1);
}

TEST(SchedulerReadyInbox, BootstrapPublishesIndependentDirectoryShards) {
    FixtureStorage storage(1, 14);
    auto *directory = scheduler_ready_directory_at(storage.scheduler_state->base(), &storage.contexts[0]);
    directory->bootstrap_ready_types[0] = UINT64_C(1) << 0;
    directory->bootstrap_ready_types[6] = UINT64_C(1) << 0;
    directory->bootstrap_ready_types[7] = UINT64_C(1) << 1;
    directory->bootstrap_ready_types[13] = (UINT64_C(1) << 0) | (UINT64_C(1) << 1);

    ASSERT_TRUE(scheduler_bootstrap_ready_directory_publish(storage.scheduler_state->base(), &storage.contexts[0], 14));

    EXPECT_EQ(directory->core_types[0][0].bits, (UINT64_C(1) << 0) | (UINT64_C(1) << 6));
    EXPECT_EQ(directory->core_types[1][0].bits, 0u);
    EXPECT_EQ(directory->core_types[0][1].bits, UINT64_C(1) << 6);
    EXPECT_EQ(directory->core_types[1][1].bits, (UINT64_C(1) << 0) | (UINT64_C(1) << 6));
}

TEST(SchedulerReadyInbox, SparseDirectoryWrapsWithinShard) {
    FixtureStorage storage(2, 14);
    GraphBuffer graph(2);
    graph.executable(0, 0);
    graph.executable(1, 0);
    SchedulerReadyStats stats{};
    SchedulerReadyBatch high_batch{};
    SchedulerReadyBatch low_batch{};
    ASSERT_TRUE(
        scheduler_ready_batch_append(storage.scheduler_state->base(), &storage.contexts[13], 0, &high_batch, &stats)
    );
    ASSERT_TRUE(scheduler_ready_batch_push(
        storage.scheduler_state->base(), &storage.contexts[13], 0, 13, &high_batch, &stats, &storage.owner_states[13]
    ));
    ASSERT_TRUE(
        scheduler_ready_batch_append(storage.scheduler_state->base(), &storage.contexts[8], 1, &low_batch, &stats)
    );
    ASSERT_TRUE(scheduler_ready_batch_push(
        storage.scheduler_state->base(), &storage.contexts[8], 0, 8, &low_batch, &stats, &storage.owner_states[8]
    ));

    uint64_t cursor = 12;
    SchedulerReadyClaim claim{};
    ASSERT_TRUE(scheduler_claim_ready_for_slot(
        graph.graph(), storage.scheduler_state->base(), &storage.contexts[7], storage.run_control, 14, 0, &cursor,
        &stats, &claim, &storage.owner_states[7]
    ));
    EXPECT_EQ(claim.task_id, 0);
    EXPECT_EQ(claim.inbox_index, 13u);
    EXPECT_EQ(cursor, 7u);

    ASSERT_TRUE(scheduler_claim_ready_for_slot(
        graph.graph(), storage.scheduler_state->base(), &storage.contexts[7], storage.run_control, 14, 0, &cursor,
        &stats, &claim, &storage.owner_states[7]
    ));
    EXPECT_EQ(claim.task_id, 1);
    EXPECT_EQ(claim.inbox_index, 8u);
    EXPECT_EQ(cursor, 9u);
}

TEST(SchedulerReadyInbox, DoesNotStealAcrossDirectoryShards) {
    FixtureStorage storage(1, 14);
    GraphBuffer graph(1);
    graph.executable(0, 0);
    SchedulerReadyBatch batch{};
    SchedulerReadyStats stats{};
    ASSERT_TRUE(scheduler_ready_batch_append(storage.scheduler_state->base(), &storage.contexts[8], 0, &batch, &stats));
    ASSERT_TRUE(scheduler_ready_batch_push(
        storage.scheduler_state->base(), &storage.contexts[8], 0, 8, &batch, &stats, &storage.owner_states[8]
    ));

    uint64_t cursor = 1;
    SchedulerReadyClaim claim{};
    ASSERT_TRUE(scheduler_claim_ready_for_slot(
        graph.graph(), storage.scheduler_state->base(), &storage.contexts[0], storage.run_control, 14, 0, &cursor,
        &stats, &claim, &storage.owner_states[0]
    ));
    EXPECT_EQ(claim.task_id, SCHEDULER_TASK_ID_INVALID);

    cursor = 8;
    ASSERT_TRUE(scheduler_claim_ready_for_slot(
        graph.graph(), storage.scheduler_state->base(), &storage.contexts[7], storage.run_control, 14, 0, &cursor,
        &stats, &claim, &storage.owner_states[7]
    ));
    EXPECT_EQ(claim.task_id, 0);
    EXPECT_EQ(claim.inbox_index, 8u);
    EXPECT_EQ(claim.source, SchedulerReadySource::STOLEN);
}

TEST(SchedulerDispatch, RejectsKernelIdBeforeCallableTableAccess) {
    FixtureStorage storage(1, 1);
    GraphBuffer graph(1);
    graph.executable(0, 0);
    storage.contexts[0].core_type = static_cast<int32_t>(CoreType::AIC);
    storage.metadata[0].kernel_ids[0] = static_cast<uint16_t>(SCHEDULER_CALLABLE_CAPACITY);
    SchedulerFreeSlotClaim slot_claim{0, 0, 0};
    SchedulerReadyClaim ready_claim{};
    ready_claim.task_id = 0;

    EXPECT_FALSE(scheduler_fill_dispatch_slot(
        graph.graph(), storage.scheduler_state->base(), &storage.contexts[0], storage.run_control, slot_claim,
        ready_claim
    ));
    EXPECT_EQ(storage.run_control->scheduler_error, static_cast<uint64_t>(SchedulerGraphResult::INVALID_CALLABLE));
    EXPECT_EQ(storage.run_control->error_site, static_cast<uint64_t>(SchedulerErrorSite::DISPATCH_INVALID_CALLABLE));
}

TEST(SchedulerDispatch, RejectsInvalidSingleSubtaskShapeBeforeMetadataIndex) {
    FixtureStorage storage(1, 1);
    GraphBuffer graph(1);
    graph.executable(0, 0);
    storage.contexts[0].core_type = static_cast<int32_t>(CoreType::AIC);
    storage.metadata[0].active_mask = 0;
    SchedulerFreeSlotClaim slot_claim{0, 0, 0};
    SchedulerReadyClaim ready_claim{};
    ready_claim.task_id = 0;

    EXPECT_FALSE(scheduler_fill_dispatch_slot(
        graph.graph(), storage.scheduler_state->base(), &storage.contexts[0], storage.run_control, slot_claim,
        ready_claim
    ));
    EXPECT_EQ(storage.run_control->scheduler_error, static_cast<uint64_t>(SchedulerGraphResult::UNSUPPORTED_SHAPE));
    EXPECT_EQ(storage.run_control->error_site, static_cast<uint64_t>(SchedulerErrorSite::DISPATCH_INVALID_SHAPE));
}

TEST(SchedulerDispatch, RejectsCoreTypeMismatch) {
    FixtureStorage storage(1, 1);
    GraphBuffer graph(1);
    graph.executable(0, 0);
    storage.contexts[0].core_type = static_cast<int32_t>(CoreType::AIV);
    SchedulerFreeSlotClaim slot_claim{0, 0, 0};
    SchedulerReadyClaim ready_claim{};
    ready_claim.task_id = 0;

    EXPECT_FALSE(scheduler_fill_dispatch_slot(
        graph.graph(), storage.scheduler_state->base(), &storage.contexts[0], storage.run_control, slot_claim,
        ready_claim
    ));
    EXPECT_EQ(storage.run_control->error_site, static_cast<uint64_t>(SchedulerErrorSite::DISPATCH_INVALID_SHAPE));
}

TEST(SchedulerDispatch, RejectsUnknownTargetCoreType) {
    FixtureStorage storage(1, 1);
    GraphBuffer graph(1);
    graph.executable(0, 1);
    storage.contexts[0].core_type = 2;
    storage.metadata[0].kernel_ids[1] = 1;
    storage.metadata[0].active_mask = 2;
    SchedulerFreeSlotClaim slot_claim{0, 0, 0};
    SchedulerReadyClaim ready_claim{};
    ready_claim.task_id = 0;

    EXPECT_FALSE(scheduler_fill_dispatch_slot(
        graph.graph(), storage.scheduler_state->base(), &storage.contexts[0], storage.run_control, slot_claim,
        ready_claim
    ));
    EXPECT_EQ(storage.run_control->scheduler_error, static_cast<uint64_t>(SchedulerGraphResult::UNSUPPORTED_SHAPE));
    EXPECT_EQ(storage.run_control->error_site, static_cast<uint64_t>(SchedulerErrorSite::DISPATCH_INVALID_SHAPE));
}

TEST(SchedulerDispatch, WrapsGenerationAndRejectsZeroCallable) {
    FixtureStorage storage(2, 1);
    GraphBuffer graph(2);
    graph.executable(0, 0);
    graph.executable(1, 0);
    storage.contexts[0].core_type = static_cast<int32_t>(CoreType::AIC);
    SchedulerFreeSlotClaim slot_claim{0, 0, UINT32_MAX};
    SchedulerReadyClaim ready_claim{};
    ready_claim.task_id = 0;

    ASSERT_TRUE(scheduler_fill_dispatch_slot(
        graph.graph(), storage.scheduler_state->base(), &storage.contexts[0], storage.run_control, slot_claim,
        ready_claim
    ));
    auto *slot = scheduler_dispatch_slot_at(storage.scheduler_state->base(), &storage.contexts[0], 0, 0);
    EXPECT_EQ(slot->generation, 1u);
    EXPECT_EQ(scheduler_dispatch_generation(slot->publication), 1u);

    storage.callable_addresses[1] = 0;
    storage.run_control->error_claimed = 0;
    storage.run_control->scheduler_error = 0;
    slot_claim.slot_index = 1;
    ready_claim.task_id = 1;
    EXPECT_FALSE(scheduler_fill_dispatch_slot(
        graph.graph(), storage.scheduler_state->base(), &storage.contexts[0], storage.run_control, slot_claim,
        ready_claim
    ));
    EXPECT_EQ(storage.run_control->scheduler_error, static_cast<uint64_t>(SchedulerGraphResult::INVALID_CALLABLE));
    EXPECT_EQ(storage.run_control->error_site, static_cast<uint64_t>(SchedulerErrorSite::DISPATCH_INVALID_CALLABLE));
}

TEST(SchedulerDispatch, AcceptsLastCallableAndInlineSentinel) {
    FixtureStorage storage(2, 1);
    GraphBuffer graph(2);
    graph.executable(0, 0);
    graph.executable(1, 0);
    storage.contexts[0].core_type = static_cast<int32_t>(CoreType::AIC);
    storage.metadata[0].kernel_ids[0] = static_cast<uint16_t>(SCHEDULER_CALLABLE_CAPACITY - 1);
    storage.callable_addresses[SCHEDULER_CALLABLE_CAPACITY - 1] = UINT64_C(0x2000);
    SchedulerFreeSlotClaim slot_claim{0, 0, 0};
    SchedulerReadyClaim ready_claim{};
    ready_claim.task_id = 0;
    ASSERT_TRUE(scheduler_fill_dispatch_slot(
        graph.graph(), storage.scheduler_state->base(), &storage.contexts[0], storage.run_control, slot_claim,
        ready_claim
    ));
    auto *payload = scheduler_state_at<DispatchPayload>(
        storage.scheduler_state->base(), storage.contexts[0].dispatch_payload_offset
    );
    EXPECT_EQ(payload->function_bin_addr, UINT64_C(0x2000));

    storage.metadata[1].kernel_ids[0] = UINT16_MAX;
    storage.metadata[1].flags |= SCHEDULER_TASK_INLINE;
    slot_claim.slot_index = 1;
    ready_claim.task_id = 1;
    ASSERT_TRUE(scheduler_fill_dispatch_slot(
        graph.graph(), storage.scheduler_state->base(), &storage.contexts[0], storage.run_control, slot_claim,
        ready_claim
    ));
    payload = scheduler_state_at<DispatchPayload>(
        storage.scheduler_state->base(), storage.contexts[0].dispatch_payload_offset + sizeof(DispatchPayload)
    );
    EXPECT_EQ(payload->function_bin_addr, 0u);
    EXPECT_EQ(storage.run_control->scheduler_error, 0u);
}

TEST(SchedulerPredicate, DistinguishesFailedAndMalformedPredicates) {
    GraphBuffer graph(1);
    graph.executable(0, 0);
    alignas(8) int64_t value = 5;
    struct PredicateCase {
        PredicateOp op;
        int64_t pass_target;
        int64_t fail_target;
    };
    constexpr std::array<PredicateCase, 6> cases{{
        {PredicateOp::EQ, 5, 6},
        {PredicateOp::NE, 6, 5},
        {PredicateOp::GT, 4, 5},
        {PredicateOp::LT, 6, 5},
        {PredicateOp::GE, 5, 6},
        {PredicateOp::LE, 5, 4},
    }};
    constexpr std::array<uint8_t, 4> element_sizes{1, 2, 4, 8};
    for (uint8_t element_size : element_sizes) {
        for (const PredicateCase &predicate_case : cases) {
            graph.predicate(
                0, reinterpret_cast<uint64_t>(&value), element_size, static_cast<uint8_t>(predicate_case.op),
                predicate_case.pass_target
            );
            EXPECT_EQ(scheduler_evaluate_task_predicate(graph.graph(), 0), SchedulerPredicateResult::PASS);
            graph.predicate(
                0, reinterpret_cast<uint64_t>(&value), element_size, static_cast<uint8_t>(predicate_case.op),
                predicate_case.fail_target
            );
            EXPECT_EQ(scheduler_evaluate_task_predicate(graph.graph(), 0), SchedulerPredicateResult::FAIL);
        }
    }

    graph.predicate(0, 0, 0, static_cast<uint8_t>(PredicateOp::NONE));
    EXPECT_EQ(scheduler_evaluate_task_predicate(graph.graph(), 0), SchedulerPredicateResult::PASS);

    graph.predicate(0, 0, 8, static_cast<uint8_t>(PredicateOp::GT));
    EXPECT_EQ(scheduler_evaluate_task_predicate(graph.graph(), 0), SchedulerPredicateResult::MALFORMED);
    graph.predicate(0, reinterpret_cast<uint64_t>(&value), 3, static_cast<uint8_t>(PredicateOp::GT));
    EXPECT_EQ(scheduler_evaluate_task_predicate(graph.graph(), 0), SchedulerPredicateResult::MALFORMED);
    graph.predicate(0, reinterpret_cast<uint64_t>(&value), 8, UINT8_C(0xff));
    EXPECT_EQ(scheduler_evaluate_task_predicate(graph.graph(), 0), SchedulerPredicateResult::MALFORMED);
    graph.predicate(0, reinterpret_cast<uint64_t>(&value) + 1, 8, static_cast<uint8_t>(PredicateOp::GT));
    EXPECT_EQ(scheduler_evaluate_task_predicate(graph.graph(), 0), SchedulerPredicateResult::MALFORMED);
}

TEST(SchedulerPredicate, MalformedPredicateStopsDispatchWithoutPublishingSlot) {
    FixtureStorage storage(1, 1);
    GraphBuffer graph(1);
    graph.executable(0, 0);
    graph.predicate(0, 0, 4, static_cast<uint8_t>(PredicateOp::GT));
    storage.contexts[0].core_type = static_cast<int32_t>(CoreType::AIC);
    storage.metadata[0].flags |= SCHEDULER_TASK_HAS_PREDICATE;
    SchedulerFreeSlotClaim slot_claim{0, 0, 0};
    SchedulerReadyClaim ready_claim{};
    ready_claim.task_id = 0;

    EXPECT_FALSE(scheduler_fill_dispatch_slot(
        graph.graph(), storage.scheduler_state->base(), &storage.contexts[0], storage.run_control, slot_claim,
        ready_claim
    ));
    EXPECT_EQ(storage.run_control->scheduler_error, static_cast<uint64_t>(SchedulerGraphResult::INVALID_ARGUMENTS));
    EXPECT_EQ(storage.run_control->error_site, static_cast<uint64_t>(SchedulerErrorSite::DISPATCH_INVALID_PREDICATE));
    const auto *slot = scheduler_dispatch_slot_at(storage.scheduler_state->base(), &storage.contexts[0], 0, 0);
    EXPECT_NE(scheduler_dispatch_state(slot->publication), SchedulerDispatchSlotState::READY);
}

TEST(SchedulerPredicate, FailedPredicatePublishesDependencyOnlyDispatch) {
    FixtureStorage storage(1, 1);
    GraphBuffer graph(1);
    graph.executable(0, 0);
    alignas(4) int32_t value = 0;
    graph.predicate(0, reinterpret_cast<uint64_t>(&value), 4, static_cast<uint8_t>(PredicateOp::GT));
    storage.contexts[0].core_type = static_cast<int32_t>(CoreType::AIC);
    storage.metadata[0].flags |= SCHEDULER_TASK_HAS_PREDICATE;
    SchedulerFreeSlotClaim slot_claim{0, 0, 0};
    SchedulerReadyClaim ready_claim{};
    ready_claim.task_id = 0;

    ASSERT_TRUE(scheduler_fill_dispatch_slot(
        graph.graph(), storage.scheduler_state->base(), &storage.contexts[0], storage.run_control, slot_claim,
        ready_claim
    ));
    const auto *payload = scheduler_state_at<DispatchPayload>(
        storage.scheduler_state->base(), storage.contexts[0].dispatch_payload_offset
    );
    EXPECT_EQ(payload->function_bin_addr, 0u);
    EXPECT_EQ(storage.run_control->scheduler_error, 0u);
}

TEST(SchedulerReadyInbox, ConcurrentConsumersNeverDuplicateTask) {
    constexpr uint64_t kTasks = 8192;
    constexpr uint64_t kConsumerCount = 8;
    FixtureStorage storage(kTasks, kConsumerCount);
    GraphBuffer graph(kTasks);
    SchedulerReadyBatch batch{};
    for (uint64_t task = 0; task < kTasks; ++task) {
        graph.executable(task, 0);
        ASSERT_TRUE(
            scheduler_ready_batch_append(storage.scheduler_state->base(), &storage.contexts[0], task, &batch, nullptr)
        );
    }
    ASSERT_TRUE(scheduler_ready_batch_push(
        storage.scheduler_state->base(), &storage.contexts[0], 0, 0, &batch, nullptr, &storage.owner_states[0]
    ));
    std::vector<std::atomic<uint32_t>> seen(kTasks);
    std::atomic<uint64_t> claimed{0};
    std::atomic<bool> failed{false};
    auto consume = [&](uint64_t worker) {
        while (!failed.load(std::memory_order_relaxed) && claimed.load(std::memory_order_relaxed) < kTasks) {
            int64_t task = SCHEDULER_TASK_ID_INVALID;
            if (!scheduler_ready_pop_from_inbox(
                    graph.graph(), storage.scheduler_state->base(), &storage.contexts[worker], storage.run_control, 0,
                    0, &task, nullptr
                )) {
                failed.store(true, std::memory_order_relaxed);
                return;
            }
            if (task >= 0) {
                seen[static_cast<size_t>(task)].fetch_add(1, std::memory_order_relaxed);
                claimed.fetch_add(1, std::memory_order_relaxed);
            }
            std::this_thread::yield();
        }
    };
    std::vector<std::thread> consumers;
    consumers.reserve(kConsumerCount);
    for (uint64_t worker = 0; worker < kConsumerCount; ++worker)
        consumers.emplace_back(consume, worker);
    for (auto &consumer : consumers)
        consumer.join();
    EXPECT_FALSE(failed.load());
    EXPECT_EQ(claimed.load(), kTasks);
    for (const auto &count : seen)
        EXPECT_EQ(count.load(), 1u);
}

TEST(SchedulerReadyInbox, OwnerPushRacesThiefWithoutLosingTasks) {
    constexpr uint64_t kTasks = 2048;
    constexpr uint64_t kSpinLimit = 10000000;
    FixtureStorage storage(kTasks, 2);
    GraphBuffer graph(kTasks);
    for (uint64_t task = 0; task < kTasks; ++task)
        graph.executable(task, 0);
    SchedulerReadyOwnerState &owner_state = storage.owner_states[0];
    std::vector<std::atomic<uint32_t>> seen(kTasks);
    std::atomic<uint64_t> claimed{0};
    std::atomic<bool> failed{false};

    std::thread owner([&] {
        for (uint64_t task = 0; task < kTasks && !failed.load(std::memory_order_relaxed); ++task) {
            SchedulerReadyBatch batch{};
            if (!scheduler_ready_batch_append(
                    storage.scheduler_state->base(), &storage.contexts[0], task, &batch, nullptr
                ) ||
                !scheduler_ready_batch_push(
                    storage.scheduler_state->base(), &storage.contexts[0], 0, 0, &batch, nullptr, &owner_state
                ) ||
                !scheduler_ready_owner_maintain_type(
                    storage.scheduler_state->base(), &storage.contexts[0], 0, &owner_state
                )) {
                failed.store(true, std::memory_order_relaxed);
                return;
            }
            std::this_thread::yield();
        }
        for (uint64_t spin = 0; spin < kSpinLimit && claimed.load(std::memory_order_relaxed) < kTasks; ++spin) {
            if (!scheduler_ready_owner_maintain_type(
                    storage.scheduler_state->base(), &storage.contexts[0], 0, &owner_state
                )) {
                failed.store(true, std::memory_order_relaxed);
                return;
            }
            std::this_thread::yield();
        }
        if (claimed.load(std::memory_order_relaxed) != kTasks) failed.store(true, std::memory_order_relaxed);
    });
    std::thread thief([&] {
        for (uint64_t spin = 0; spin < kSpinLimit && !failed.load(std::memory_order_relaxed) &&
                                claimed.load(std::memory_order_relaxed) < kTasks;
             ++spin) {
            int64_t task = SCHEDULER_TASK_ID_INVALID;
            if (!scheduler_ready_pop_from_inbox(
                    graph.graph(), storage.scheduler_state->base(), &storage.contexts[1], storage.run_control, 0, 0,
                    &task, nullptr
                )) {
                failed.store(true, std::memory_order_relaxed);
                return;
            }
            if (task >= 0) {
                seen[static_cast<size_t>(task)].fetch_add(1, std::memory_order_relaxed);
                claimed.fetch_add(1, std::memory_order_relaxed);
            }
            std::this_thread::yield();
        }
    });
    owner.join();
    thief.join();

    EXPECT_FALSE(failed.load());
    EXPECT_EQ(claimed.load(), kTasks);
    for (const auto &count : seen)
        EXPECT_EQ(count.load(), 1u);
}

TEST(SchedulerReadyInbox, PendingPromotionRacesThiefWithoutReplayingTasks) {
    constexpr uint64_t kTasks = 2048;
    constexpr uint64_t kSpinLimit = 10000000;
    FixtureStorage storage(kTasks, 2);
    GraphBuffer graph(kTasks);
    for (uint64_t task = 0; task < kTasks; ++task)
        graph.executable(task, 0);
    SchedulerReadyOwnerState &owner_state = storage.owner_states[0];
    SchedulerReadyBatch published{};
    ASSERT_TRUE(
        scheduler_ready_batch_append(storage.scheduler_state->base(), &storage.contexts[0], 0, &published, nullptr)
    );
    ASSERT_TRUE(scheduler_ready_batch_push(
        storage.scheduler_state->base(), &storage.contexts[0], 0, 0, &published, nullptr, &owner_state
    ));
    SchedulerReadyBatch pending{};
    for (uint64_t task = 1; task < kTasks; ++task)
        ASSERT_TRUE(
            scheduler_ready_batch_append(storage.scheduler_state->base(), &storage.contexts[0], task, &pending, nullptr)
        );
    ASSERT_TRUE(scheduler_ready_batch_push(
        storage.scheduler_state->base(), &storage.contexts[0], 0, 0, &pending, nullptr, &owner_state
    ));
    ASSERT_NE(scheduler_ready_pending_head(owner_state.queues[0].pending_endpoints), SCHEDULER_INBOX_EMPTY);

    std::vector<std::atomic<uint32_t>> seen(kTasks);
    std::atomic<uint64_t> claimed{0};
    std::atomic<bool> failed{false};
    std::thread owner([&] {
        for (uint64_t spin = 0; spin < kSpinLimit && !failed.load(std::memory_order_relaxed) &&
                                claimed.load(std::memory_order_relaxed) < kTasks;
             ++spin) {
            if (!scheduler_ready_owner_maintain_type(
                    storage.scheduler_state->base(), &storage.contexts[0], 0, &owner_state
                )) {
                failed.store(true, std::memory_order_relaxed);
                return;
            }
            std::this_thread::yield();
        }
    });
    std::thread thief([&] {
        for (uint64_t spin = 0; spin < kSpinLimit && !failed.load(std::memory_order_relaxed) &&
                                claimed.load(std::memory_order_relaxed) < kTasks;
             ++spin) {
            int64_t task = SCHEDULER_TASK_ID_INVALID;
            if (!scheduler_ready_pop_from_inbox(
                    graph.graph(), storage.scheduler_state->base(), &storage.contexts[1], storage.run_control, 0, 0,
                    &task, nullptr
                )) {
                failed.store(true, std::memory_order_relaxed);
                return;
            }
            if (task >= 0) {
                seen[static_cast<size_t>(task)].fetch_add(1, std::memory_order_relaxed);
                claimed.fetch_add(1, std::memory_order_relaxed);
            }
            std::this_thread::yield();
        }
    });
    owner.join();
    thief.join();
    if (claimed.load(std::memory_order_relaxed) != kTasks) failed.store(true, std::memory_order_relaxed);

    EXPECT_FALSE(failed.load());
    EXPECT_EQ(claimed.load(), kTasks);
    for (const auto &count : seen)
        EXPECT_EQ(count.load(), 1u);
}

TEST(SchedulerReadyWake, ConcurrentRegistrationAndCloseResolveEveryConsumerExactlyOnce) {
    constexpr uint64_t kConsumers = 64;
    FixtureStorage storage(kConsumers + 1, 1);
    GraphBuffer graph(kConsumers + 1);
    graph.executable(0, 0);
    for (uint64_t task = 1; task <= kConsumers; ++task) {
        graph.executable(task, 0, {0});
        storage.metadata[task].flags |= SCHEDULER_TASK_HAS_FANIN;
    }
    auto *controls =
        scheduler_state_at<SchedulerTaskControl>(storage.scheduler_state->base(), storage.layout.task_controls_offset);
    std::vector<std::atomic<uint32_t>> seen(kConsumers + 1);
    std::atomic<uint64_t> started{0};
    std::atomic<bool> failed{false};
    std::vector<std::thread> consumers;
    consumers.reserve(kConsumers);
    for (uint64_t task = 1; task <= kConsumers; ++task) {
        consumers.emplace_back([&, task] {
            started.fetch_add(1, std::memory_order_release);
            const SchedulerRouteResult route = scheduler_route_task(
                graph.graph(), storage.scheduler_state->base(), &storage.contexts[0], storage.run_control,
                static_cast<int64_t>(task), nullptr
            );
            if (route == SchedulerRouteResult::READY_TO_ENQUEUE) {
                seen[task].fetch_add(1, std::memory_order_relaxed);
            } else if (route != SchedulerRouteResult::WAITING) {
                failed.store(true, std::memory_order_relaxed);
            }
        });
    }
    std::thread closer([&] {
        while (started.load(std::memory_order_acquire) != kConsumers)
            std::this_thread::yield();
        scheduler_gm_store(controls[0].state, static_cast<int64_t>(SchedulerTaskState::DONE));
        if (!scheduler_resolve_completion(
                graph.graph(), storage.scheduler_state->base(), &storage.contexts[0], storage.run_control, 0, nullptr,
                nullptr, nullptr, &storage.owner_states[0]
            ))
            failed.store(true, std::memory_order_relaxed);
    });
    for (auto &consumer : consumers)
        consumer.join();
    closer.join();

    while (!failed.load(std::memory_order_relaxed)) {
        int64_t task = SCHEDULER_TASK_ID_INVALID;
        ASSERT_TRUE(scheduler_ready_pop_from_inbox(
            graph.graph(), storage.scheduler_state->base(), &storage.contexts[0], storage.run_control, 0, 0, &task,
            nullptr
        ));
        if (task == SCHEDULER_TASK_ID_INVALID) break;
        seen[static_cast<size_t>(task)].fetch_add(1, std::memory_order_relaxed);
    }
    EXPECT_FALSE(failed.load());
    EXPECT_EQ(storage.run_control->scheduler_error, 0u);
    for (uint64_t task = 1; task <= kConsumers; ++task) {
        EXPECT_EQ(seen[task].load(), 1u) << "consumer=" << task;
        EXPECT_EQ(controls[task].next_fanin_index, 1) << "consumer=" << task;
    }
}

TEST(SchedulerReadyWake, WakeResolvePublishesConsumerToSchedulerLocalInbox) {
    FixtureStorage storage(2, 1);
    GraphBuffer graph(2);
    graph.executable(0, 0);
    graph.executable(1, 0, {0});
    storage.metadata[1].flags |= SCHEDULER_TASK_HAS_FANIN;
    SchedulerWakeStats wake{};
    SchedulerReadyStats ready{};
    SchedulerCompletionStats completion{};
    EXPECT_EQ(
        scheduler_route_task(
            graph.graph(), storage.scheduler_state->base(), &storage.contexts[0], storage.run_control, 1, &wake
        ),
        SchedulerRouteResult::WAITING
    );
    auto *controls =
        scheduler_state_at<SchedulerTaskControl>(storage.scheduler_state->base(), storage.layout.task_controls_offset);
    controls[0].state = static_cast<int64_t>(SchedulerTaskState::DONE);
    ASSERT_TRUE(scheduler_resolve_completion(
        graph.graph(), storage.scheduler_state->base(), &storage.contexts[0], storage.run_control, 0, &wake, &ready,
        &completion, &storage.owner_states[0]
    ));
    EXPECT_EQ(completion.resolve_count, 1u);
    int64_t task = SCHEDULER_TASK_ID_INVALID;
    ASSERT_TRUE(scheduler_ready_pop_from_inbox(
        graph.graph(), storage.scheduler_state->base(), &storage.contexts[0], storage.run_control, 0, 0, &task, &ready
    ));
    EXPECT_EQ(task, 1);
    EXPECT_EQ(wake.wake_register_count, 1u);
    EXPECT_EQ(wake.wake_migrate_count, 1u);
}

TEST(SchedulerReadyWake, WakeResolveQueuesBehindOlderPublishedWork) {
    FixtureStorage storage(3, 1);
    GraphBuffer graph(3);
    graph.executable(0, 0);
    graph.executable(1, 0, {0});
    graph.executable(2, 0);
    storage.metadata[1].flags |= SCHEDULER_TASK_HAS_FANIN;
    SchedulerWakeStats wake{};
    SchedulerReadyStats ready{};
    SchedulerCompletionStats completion{};
    SchedulerReadyOwnerState &owner_state = storage.owner_states[0];
    EXPECT_EQ(
        scheduler_route_task(
            graph.graph(), storage.scheduler_state->base(), &storage.contexts[0], storage.run_control, 1, &wake
        ),
        SchedulerRouteResult::WAITING
    );
    SchedulerReadyBatch older{};
    ASSERT_TRUE(scheduler_ready_batch_append(storage.scheduler_state->base(), &storage.contexts[0], 2, &older, &ready));
    ASSERT_TRUE(scheduler_ready_batch_push(
        storage.scheduler_state->base(), &storage.contexts[0], 0, 0, &older, &ready, &owner_state
    ));
    auto *controls =
        scheduler_state_at<SchedulerTaskControl>(storage.scheduler_state->base(), storage.layout.task_controls_offset);
    controls[0].state = static_cast<int64_t>(SchedulerTaskState::DONE);
    ASSERT_TRUE(scheduler_resolve_completion(
        graph.graph(), storage.scheduler_state->base(), &storage.contexts[0], storage.run_control, 0, &wake, &ready,
        &completion, &owner_state, false, true, nullptr
    ));
    EXPECT_EQ(completion.resolve_count, 1u);
    EXPECT_EQ(scheduler_ready_pending_head(owner_state.queues[0].pending_endpoints), 1);

    int64_t task = SCHEDULER_TASK_ID_INVALID;
    ASSERT_TRUE(scheduler_ready_pop_from_inbox(
        graph.graph(), storage.scheduler_state->base(), &storage.contexts[0], storage.run_control, 0, 0, &task, &ready
    ));
    ASSERT_EQ(task, 2);
    ASSERT_TRUE(
        scheduler_ready_owner_maintain_type(storage.scheduler_state->base(), &storage.contexts[0], 0, &owner_state)
    );
    ASSERT_TRUE(scheduler_ready_pop_from_inbox(
        graph.graph(), storage.scheduler_state->base(), &storage.contexts[0], storage.run_control, 0, 0, &task, &ready
    ));
    EXPECT_EQ(task, 1);
}

TEST(SchedulerReadyWake, RejectsInvalidWaiterShapeBeforeReadyBatchIndex) {
    FixtureStorage storage(2, 1);
    GraphBuffer graph(2);
    graph.executable(0, 0);
    graph.executable(1, 0, {0});
    storage.metadata[1].active_mask = 0;
    storage.metadata[1].flags |= SCHEDULER_TASK_HAS_FANIN;
    ASSERT_EQ(
        scheduler_route_task(
            graph.graph(), storage.scheduler_state->base(), &storage.contexts[0], storage.run_control, 1, nullptr
        ),
        SchedulerRouteResult::WAITING
    );
    auto *controls =
        scheduler_state_at<SchedulerTaskControl>(storage.scheduler_state->base(), storage.layout.task_controls_offset);
    controls[0].state = static_cast<int64_t>(SchedulerTaskState::DONE);

    EXPECT_FALSE(scheduler_resolve_completion(
        graph.graph(), storage.scheduler_state->base(), &storage.contexts[0], storage.run_control, 0, nullptr, nullptr,
        nullptr, &storage.owner_states[0]
    ));
    EXPECT_EQ(storage.run_control->scheduler_error, static_cast<uint64_t>(SchedulerGraphResult::UNSUPPORTED_SHAPE));
    EXPECT_EQ(storage.run_control->error_site, static_cast<uint64_t>(SchedulerErrorSite::COMPLETION_INVALID_SHAPE));
}

}  // namespace
