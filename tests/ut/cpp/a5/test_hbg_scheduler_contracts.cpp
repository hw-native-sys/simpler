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
#include <atomic>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <vector>

#include "host_build_graph/runtime_types.h"
#include "aicore_scheduler_error.h"
#include "aicore_scheduler_state.h"
#include "scheduler/scheduler_graph.h"
#include "scheduler/scheduler_topology.h"
#include "scheduler/scheduler_types.h"
#include "scheduler/scheduler_watchdog.h"

namespace {

TEST(AicoreSchedulerError, MapsInternalFailureToExistingHostStatus) {
    EXPECT_EQ(aicore_scheduler_runtime_error_code(0), SIMPLER_ERROR_NONE);
    EXPECT_EQ(aicore_scheduler_runtime_error_code(1), SIMPLER_ERROR_INVALID_ARGS);
    EXPECT_EQ(
        aicore_scheduler_runtime_error_code(static_cast<uint64_t>(SchedulerGraphResult::TIMEOUT)),
        SIMPLER_ERROR_SCHEDULER_TIMEOUT
    );
    EXPECT_EQ(aicore_scheduler_runtime_error_code(UINT64_MAX), SIMPLER_ERROR_INVALID_ARGS);
}

TEST(AicoreSchedulerError, LatchesFirstHostVisibleFailure) {
    std::atomic<int32_t> status{SIMPLER_ERROR_NONE};
    EXPECT_FALSE(latch_aicore_scheduler_runtime_error(&status, 0));
    EXPECT_EQ(status.load(std::memory_order_relaxed), SIMPLER_ERROR_NONE);

    EXPECT_TRUE(latch_aicore_scheduler_runtime_error(&status, static_cast<uint64_t>(SchedulerGraphResult::TIMEOUT)));
    EXPECT_EQ(status.load(std::memory_order_relaxed), SIMPLER_ERROR_SCHEDULER_TIMEOUT);

    status.store(SIMPLER_ERROR_READY_QUEUE_OVERFLOW, std::memory_order_relaxed);
    EXPECT_FALSE(latch_aicore_scheduler_runtime_error(&status, 2));
    EXPECT_EQ(status.load(std::memory_order_relaxed), SIMPLER_ERROR_READY_QUEUE_OVERFLOW);
}

TEST(AicoreSchedulerState, DistinguishesResidentAndExplicitLegacyModes) {
    EXPECT_FALSE(aicore_scheduler_runtime_mode_is_resident(0));
    EXPECT_TRUE(aicore_scheduler_runtime_mode_is_resident(SCHEDULER_RUNTIME_MODE_RESIDENT_PENDING));
    EXPECT_TRUE(aicore_scheduler_runtime_mode_is_resident(SCHEDULER_RUNTIME_MODE_RESIDENT_READY));
    EXPECT_FALSE(aicore_scheduler_runtime_mode_is_resident(SCHEDULER_RUNTIME_MODE_LEGACY_GRAPH));
    EXPECT_FALSE(aicore_scheduler_runtime_mode_is_resident(SCHEDULER_RUNTIME_MODE_LEGACY_UNSUPPORTED_SHAPE));

    EXPECT_FALSE(aicore_scheduler_runtime_mode_is_explicit_legacy(0));
    EXPECT_FALSE(aicore_scheduler_runtime_mode_is_explicit_legacy(SCHEDULER_RUNTIME_MODE_RESIDENT_PENDING));
    EXPECT_FALSE(aicore_scheduler_runtime_mode_is_explicit_legacy(SCHEDULER_RUNTIME_MODE_RESIDENT_READY));
    EXPECT_TRUE(aicore_scheduler_runtime_mode_is_explicit_legacy(SCHEDULER_RUNTIME_MODE_LEGACY_GRAPH));
    EXPECT_TRUE(aicore_scheduler_runtime_mode_is_explicit_legacy(SCHEDULER_RUNTIME_MODE_LEGACY_UNSUPPORTED_SHAPE));
}

TEST(AicoreSchedulerState, ResidentV0AcceptsOnlySingleLaneSingleBlockTasks) {
    EXPECT_TRUE(scheduler_resident_v0_task_shape_supported(1, 1, false));
    EXPECT_FALSE(scheduler_resident_v0_task_shape_supported(2, 1, false));
    EXPECT_FALSE(scheduler_resident_v0_task_shape_supported(1, 2, false));
    EXPECT_FALSE(scheduler_resident_v0_task_shape_supported(1, 1, true));
}

TEST(AicoreSchedulerWatchdog, UsesElapsedWallClockBudget) {
    EXPECT_FALSE(scheduler_watchdog_expired(100, UINT64_MAX, 0));
    EXPECT_FALSE(scheduler_watchdog_expired(100, 199, 100));
    EXPECT_TRUE(scheduler_watchdog_expired(100, 200, 100));
    EXPECT_TRUE(scheduler_watchdog_expired(UINT64_MAX - 10, 4, 15));
}

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

    TaskPayload &payload(size_t task) { return storage_[task].payload; }
    ChipTaskStorage *storage() { return storage_; }

    SchedulerGraphView graph() const {
        return {
            reinterpret_cast<uint64_t>(storage_),
            0,
            task_count_,
            capacity_ - 1,
        };
    }

private:
    static constexpr size_t kMaxTaskCount = 16;
    // One storage array, as production has it — the descriptor and payload of a
    // task are members of one entry. Two parallel arrays would model the layout
    // this runtime no longer uses and would let the wire strides drift unnoticed.
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

TEST(SchedulerState, PlansAndInitializesReadyState) {
    // The reverse lookup for the wire constants AICore addresses with. They are
    // literals there because the AICore .o cannot see these types; here it can.
    EXPECT_EQ(sizeof(ChipTaskStorage), SCHEDULER_GRAPH_TASK_STORAGE_STRIDE);
    EXPECT_EQ(offsetof(ChipTaskStorage, task), SCHEDULER_GRAPH_DESCRIPTOR_OFFSET);
    EXPECT_EQ(offsetof(ChipTaskStorage, payload), SCHEDULER_GRAPH_PAYLOAD_OFFSET);
    EXPECT_EQ(offsetof(TaskPayload, predicate), SCHEDULER_GRAPH_PREDICATE_OFFSET);
    AicoreSchedulerLayout layout{};
    ASSERT_TRUE(scheduler_plan_layout(5, 3, 2, &layout));
    EXPECT_EQ(layout.total_size % SCHEDULER_STATE_ALIGNMENT, 0u);
    EXPECT_EQ(layout.task_metadata_offset % alignof(SchedulerTaskMetadata), 0u);
    EXPECT_EQ(layout.ready_inboxes_offset % alignof(SchedulerReadyInbox), 0u);
    EXPECT_EQ(layout.ready_owner_states_offset % alignof(SchedulerReadyOwnerState), 0u);
    EXPECT_EQ(layout.ready_directory_offset % alignof(SchedulerReadyDirectory), 0u);
    EXPECT_EQ(layout.completion_inboxes_offset % alignof(SchedulerCompletionInbox), 0u);

    SchedulerStateBuffer storage(layout);
    auto *controls = scheduler_state_at<SchedulerTaskControl>(storage.base(), layout.task_controls_offset);
    for (uint64_t task = 0; task < layout.task_count; ++task) {
        EXPECT_EQ(controls[task].state, static_cast<int64_t>(SchedulerTaskState::BLOCKED));
        EXPECT_EQ(controls[task].wake_list_head, SCHEDULER_WAKE_LIST_OPEN);
    }
    auto *completion = scheduler_state_at<SchedulerCompletionInbox>(storage.base(), layout.completion_inboxes_offset);
    for (uint64_t worker = 0; worker < SCHEDULER_WORKER_CAPACITY; ++worker) {
        EXPECT_EQ(completion[worker].completed_generations[0], 0u);
        EXPECT_EQ(completion[worker].completed_generations[1], 0u);
    }
    auto *ready = scheduler_state_at<SchedulerReadyInbox>(storage.base(), layout.ready_inboxes_offset);
    for (uint64_t inbox = 0; inbox < SCHEDULER_CORE_TYPE_COUNT * SCHEDULER_WORKER_CAPACITY; ++inbox)
        EXPECT_EQ(ready[inbox].head, SCHEDULER_INBOX_EMPTY);
    auto *ready_owners = scheduler_state_at<SchedulerReadyOwnerState>(storage.base(), layout.ready_owner_states_offset);
    for (uint64_t owner = 0; owner < SCHEDULER_CLUSTER_CAPACITY; ++owner) {
        for (uint32_t type = 0; type < SCHEDULER_CORE_TYPE_COUNT; ++type)
            EXPECT_EQ(ready_owners[owner].queues[type].pending_endpoints, SCHEDULER_READY_PENDING_EMPTY);
    }

    auto *directory = scheduler_state_at<SchedulerReadyDirectory>(storage.base(), layout.ready_directory_offset);
    auto shard0 = reinterpret_cast<uintptr_t>(&directory->core_types[0][0]);
    auto shard1 = reinterpret_cast<uintptr_t>(&directory->core_types[0][1]);
    auto aiv_shard0 = reinterpret_cast<uintptr_t>(&directory->core_types[1][0]);
    EXPECT_EQ(shard1 - shard0, 64u);
    EXPECT_EQ(aiv_shard0 - shard0, SCHEDULER_READY_DIRECTORY_SHARD_COUNT * 64u);
}

TEST(SchedulerState, PreservesCacheLineAlignmentAndArrayStride) {
    EXPECT_EQ(alignof(SchedulerTaskControl), 128u);
    EXPECT_EQ(alignof(SchedulerCompletionInbox), 64u);
    EXPECT_EQ(alignof(SchedulerReadyOwnerState), 128u);
    EXPECT_EQ(alignof(SchedulerDispatchSlot), 128u);
    EXPECT_EQ(alignof(SchedulerRunControl), 128u);
    EXPECT_EQ(alignof(SchedulerWorkerContext), 128u);
    EXPECT_EQ(alignof(SchedulerTaskTrace), 128u);

    std::array<SchedulerTaskControl, 2> controls{};
    std::array<SchedulerDispatchSlot, 2> dispatch_slots{};
    std::array<SchedulerWorkerContext, 2> contexts{};
    EXPECT_EQ(reinterpret_cast<uintptr_t>(&controls[1]) - reinterpret_cast<uintptr_t>(&controls[0]), 128u);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(&dispatch_slots[1]) - reinterpret_cast<uintptr_t>(&dispatch_slots[0]), 256u);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(&contexts[1]) - reinterpret_cast<uintptr_t>(&contexts[0]), 1024u);
    EXPECT_EQ(offsetof(SchedulerTaskControl, state) / 64, offsetof(SchedulerTaskControl, wake_list_head) / 64);
    EXPECT_NE(offsetof(SchedulerTaskControl, state) / 64, offsetof(SchedulerTaskControl, next_waiter) / 64);
    EXPECT_NE(offsetof(SchedulerDispatchSlot, task_id) / 64, offsetof(SchedulerDispatchSlot, publication) / 64);
    EXPECT_EQ(offsetof(SchedulerDispatchSlot, executor_trace), 128u);
    EXPECT_EQ(
        offsetof(SchedulerTaskTrace, dispatch_start_cycles) / 64, offsetof(SchedulerTaskTrace, complete_loop_iter) / 64
    );
    EXPECT_NE(
        offsetof(SchedulerTaskTrace, dispatch_start_cycles) / 64,
        offsetof(SchedulerTaskTrace, descriptor_cache_observed_cycles) / 64
    );
    EXPECT_NE(
        offsetof(SchedulerTaskTrace, refill_scheduler_worker_id) / 64,
        offsetof(SchedulerTaskTrace, descriptor_cache_observed_cycles) / 64
    );
}

TEST(SchedulerMetadata, ProjectsExistingSubmitTypesWithoutChangingTheirSemantics) {
    TaskAttrs attrs{};
    attrs.set_sync_start();
    attrs.set_predicate();
    attrs.set_timing_slot(7);
    const uint8_t flags = scheduler_task_metadata_flags_from_submit_state(
        ActiveMask(SUBTASK_MASK_AIC | SUBTASK_MASK_AIV0), attrs, 2, true, false
    );
    EXPECT_TRUE(scheduler_task_is_executable(flags));
    EXPECT_TRUE(scheduler_task_has_fanin(flags));
    EXPECT_TRUE(scheduler_task_is_mix(flags));
    EXPECT_TRUE(scheduler_task_is_spmd(flags));
    EXPECT_TRUE(scheduler_task_requires_sync_start(flags));
    EXPECT_TRUE(scheduler_task_has_predicate(flags));
    EXPECT_FALSE(scheduler_task_is_inline(flags));
    EXPECT_EQ(attrs.timing_slot(), 7);

    const uint8_t inline_flags =
        scheduler_task_metadata_flags_from_submit_state(ActiveMask(SUBTASK_MASK_AIV0), TaskAttrs{}, 1, false, true);
    EXPECT_EQ(inline_flags, SCHEDULER_TASK_EXECUTABLE | SCHEDULER_TASK_INLINE);
}

// AICore addresses a task's records by literal offsets, because its .o cannot see
// the types. The reverse lookup in PlansAndInitializesReadyState only proves the
// constants carry the right values; it cannot catch a helper that strides by the
// wrong one. That failure is invisible at task 0 — every candidate stride agrees
// there — so this walks several tasks and checks both records of each against the
// entry they belong to.
TEST(SchedulerGraph, AddressesEveryTaskAtTheStorageStride) {
    constexpr size_t kTasks = 4;
    struct alignas(64) GraphImage {
        std::array<ChipTaskStorage, kTasks> storage{};
    } image;
    auto &storage = image.storage;
    SchedulerGraphView graph{reinterpret_cast<uint64_t>(storage.data()), 0, kTasks, kTasks - 1};

    for (size_t task = 0; task < kTasks; ++task) {
        SCOPED_TRACE(testing::Message() << "task " << task);
        EXPECT_EQ(
            scheduler_graph_descriptor(graph, static_cast<int64_t>(task)),
            reinterpret_cast<uint8_t *>(&storage[task].task)
        );
        EXPECT_EQ(
            scheduler_graph_payload(graph, static_cast<int64_t>(task)),
            reinterpret_cast<uint8_t *>(&storage[task].payload)
        );
    }

    // Consecutive tasks are one whole storage apart, not one record apart. The
    // pre-merge strides (64 for a descriptor, 192 for a payload) satisfy the loop
    // above at task 0 and fail it everywhere else; this states the step directly.
    const auto *first = scheduler_graph_descriptor(graph, 0);
    const auto *second = scheduler_graph_descriptor(graph, 1);
    EXPECT_EQ(second - first, SCHEDULER_GRAPH_TASK_STORAGE_STRIDE);
    EXPECT_EQ(scheduler_graph_payload(graph, 0) - scheduler_graph_descriptor(graph, 0), SCHEDULER_GRAPH_PAYLOAD_OFFSET);
}

TEST(SchedulerGraph, NonPowerOfTwoWindowKeepsDirectTaskIdIndexing) {
    struct alignas(64) GraphImage {
        std::array<ChipTaskStorage, 3> storage{};
        std::array<std::array<int32_t, SCHEDULER_GRAPH_MAX_FANIN>, 3> fanins{};
    } image;
    auto &storage = image.storage;
    auto &fanins = image.fanins;
    for (size_t task = 0; task < storage.size(); ++task) {
        storage[task].task.task_id = TaskId{task};
        for (int slot = 0; slot < 3; ++slot)
            storage[task].task.kernel_id[slot] = INVALID_KERNEL_ID;
        storage[task].payload.bind_regions(nullptr, nullptr, fanins[task].data());
    }
    storage[1].task.kernel_id[0] = 1;
    SchedulerGraphView graph{reinterpret_cast<uint64_t>(storage.data()), 0, 3, 2};

    // Both records of task 1 resolve from the one base, at that entry's stride.
    EXPECT_EQ(scheduler_graph_descriptor(graph, 1), reinterpret_cast<uint8_t *>(&storage[1].task));
    EXPECT_EQ(scheduler_graph_payload(graph, 1), reinterpret_cast<uint8_t *>(&storage[1].payload));
    SchedulerTaskShape shape{};
    EXPECT_EQ(scheduler_classify_task_shape(graph, 1, &shape), SchedulerGraphResult::OK);
    EXPECT_EQ(shape.task_id, 1);
}

TEST(SchedulerGraph, ValidatesNonEmptyFanins) {
    GraphBuffer graph(3);
    graph.executable(0, 0);
    graph.executable(1, 0, {0});
    graph.executable(2, 0, {1, 1});

    SchedulerTaskShape shape{};
    EXPECT_EQ(scheduler_classify_task_shape(graph.graph(), 1, &shape), SchedulerGraphResult::OK);
    EXPECT_EQ(scheduler_classify_task_shape(graph.graph(), 2, &shape), SchedulerGraphResult::INVALID_FANIN_ID);
}

TEST(SchedulerGraph, RejectsUnboundNonEmptyFaninRegion) {
    GraphBuffer graph(2);
    graph.executable(0, 0);
    graph.executable(1, 0, {0});
    graph.payload(1).bind_regions(nullptr, nullptr, nullptr);

    SchedulerTaskShape shape{};
    EXPECT_EQ(scheduler_classify_task_shape(graph.graph(), 1, &shape), SchedulerGraphResult::INVALID_FANIN_ID);
}

// The view's reserved word exists so its wire size survived the collapse from two
// addresses to one. A non-zero read means the producer is writing a field this
// build does not know about, so both entry points refuse rather than proceed on a
// view they cannot fully interpret.
TEST(SchedulerGraph, RejectsAViewWithANonZeroReservedWord) {
    GraphBuffer graph(1);
    graph.executable(0, 0);

    SchedulerGraphView dirty = graph.graph();
    dirty.reserved = 1;

    SchedulerTaskShape shape{};
    EXPECT_EQ(scheduler_classify_task_shape(dirty, 0, &shape), SchedulerGraphResult::INVALID_ARGUMENTS);

    DispatchPayload payload{};
    SchedulerTaskInfo task{0, 1, 0, CoreType::AIC};
    EXPECT_EQ(
        scheduler_materialize_task_payload_resolved(dirty, task, 0x1000, &payload),
        SchedulerGraphResult::INVALID_ARGUMENTS
    );
}

TEST(SchedulerDispatchPayload, DisablesDeferredCompletionWithoutASlab) {
    GraphBuffer graph(1);
    graph.executable(0, 0);
    DispatchPayload payload{};
    payload.local_context.async_ctx.task_token = TaskId{17};
    SchedulerTaskInfo task{0, 1, 0, CoreType::AIC};

    ASSERT_EQ(
        scheduler_materialize_task_payload_resolved(graph.graph(), task, 0x1000, &payload), SchedulerGraphResult::OK
    );
    EXPECT_FALSE(payload.local_context.async_ctx.task_token.is_valid());
    EXPECT_EQ(payload.global_context.sub_block_id, 0);

    task.subtask_slot = 2;
    task.core_type = CoreType::AIV;
    ASSERT_EQ(
        scheduler_materialize_task_payload_resolved(graph.graph(), task, 0x1000, &payload), SchedulerGraphResult::OK
    );
    EXPECT_EQ(payload.global_context.sub_block_id, 1);
}

TEST(SchedulerDispatchPayload, RejectsInvalidGraphBoundsBeforeReadingPayload) {
    GraphBuffer graph(1);
    graph.executable(0, 0);
    DispatchPayload payload{};

    SchedulerTaskInfo out_of_range{1, 1, 0, CoreType::AIC};
    EXPECT_EQ(
        scheduler_materialize_task_payload_resolved(graph.graph(), out_of_range, 0x1000, &payload),
        SchedulerGraphResult::INVALID_TASK_COUNT
    );

    SchedulerGraphView missing_storage = graph.graph();
    missing_storage.storage_address = 0;
    SchedulerTaskInfo valid_task{0, 1, 0, CoreType::AIC};
    EXPECT_EQ(
        scheduler_materialize_task_payload_resolved(missing_storage, valid_task, 0x1000, &payload),
        SchedulerGraphResult::INVALID_TASK_COUNT
    );
}

TEST(SchedulerDispatchPayload, RejectsUnboundNonEmptyArgumentRegions) {
    GraphBuffer graph(1);
    graph.executable(0, 0);
    DispatchPayload payload{};
    SchedulerTaskInfo task{0, 1, 0, CoreType::AIC};

    graph.payload(0).tensor_count = 1;
    EXPECT_EQ(
        scheduler_materialize_task_payload_resolved(graph.graph(), task, 0x1000, &payload),
        SchedulerGraphResult::INVALID_ARGUMENTS
    );

    graph.payload(0).tensor_count = 0;
    graph.payload(0).scalar_count = 1;
    EXPECT_EQ(
        scheduler_materialize_task_payload_resolved(graph.graph(), task, 0x1000, &payload),
        SchedulerGraphResult::INVALID_ARGUMENTS
    );
}

TEST(SchedulerClusterTopology, UsesMixedKernelLaunchCoordinate) {
    constexpr int32_t kClusters = 4;
    SchedulerClusterCoordinate coordinate{};
    ASSERT_TRUE(scheduler_cluster_coordinate_from_worker(2, true, kClusters, 2, &coordinate));
    EXPECT_EQ(coordinate.cluster_index, 2);
    EXPECT_EQ(coordinate.cluster_lane, 0);
    ASSERT_TRUE(scheduler_cluster_coordinate_from_worker(9, false, kClusters, 2, &coordinate));
    EXPECT_EQ(coordinate.cluster_index, 2);
    EXPECT_EQ(coordinate.cluster_lane, 2);
    EXPECT_FALSE(scheduler_cluster_coordinate_from_worker(12, false, kClusters, 2, &coordinate));
}

}  // namespace
