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
/**
 * Poison test for the "every device-read SM field is written at submit" contract.
 *
 * host_build_graph does not zero-fill the shared-memory task table (init-on-write):
 * init_header writes only the header, and each slot's device-read fields are
 * written per task at submit (prepare_task + submit_task_common + TaskPayload::init).
 * Nothing else clears the table, so a device-read field a submit forgets to write would
 * read as 0 only by allocator accident — passing every zero-backed test and failing
 * non-deterministically on device.
 *
 * This test fills the whole task table with a 0xAA poison byte before submitting a
 * representative mix, then asserts that for every claimed slot [0, total_tasks) the
 * device-read fields carry real values, not poison. Add a device-read field and forget
 * its submit-path write, and this fails in-tree.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <tuple>
#include <vector>

#include "utils/device_arena.h"
#include "scheduler/scheduler.h"
#include "host_build_graph/orchestrator.h"
#include "host_build_graph/shared_memory.h"
#include "host_build_graph/task_id.h"

namespace {

static_assert(sizeof(LaunchSpec) == 4);
static_assert(alignof(LaunchSpec) == 2);

constexpr uint8_t POISON = 0xAA;
// A void* / int32 whose bytes are all 0xAA — what an unwritten field would read as.
void *const POISON_PTR = reinterpret_cast<void *>(static_cast<uintptr_t>(0xAAAAAAAAAAAAAAAAULL));

}  // namespace

class HbgSubmitPoisonTest : public ::testing::Test {
protected:
    DeviceArena sm_arena;
    DeviceArena runtime_arena;
    SharedMemoryHandle *sm_handle = nullptr;
    OrchestratorState orch{};
    SchedulerState sched{};
    SchedulerLayout sched_layout{};
    std::vector<char> gm_heap;

    void SetUp() override {
        sm_handle = SharedMemoryHandle::create_and_init_default(sm_arena);
        ASSERT_NE(sm_handle, nullptr);
        gm_heap.resize(4096);

        sched_layout = SchedulerState::reserve_layout(runtime_arena);
        ASSERT_NE(runtime_arena.commit(), nullptr);

        ASSERT_TRUE(sched.init_data_from_layout(sched_layout, runtime_arena, sm_handle->sm_base));
        sched.wire_arena_pointers(sched_layout, runtime_arena);
        // Same order the AICPU boots in: the slot arrays are not part of the
        // uploaded image, so nothing can push until they carry their ramp.
        sched.seed_queue_slots();
        ASSERT_TRUE(orch.init(sm_handle->sm_base, gm_heap.data(), 4096, CHIP_DEFAULT_GRAPH_TASKS));
    }

    void TearDown() override {
        sched.destroy();
        runtime_arena.release();
        sm_arena.release();
    }

    // Fill the task table (storage entries / task_states) with poison.
    // init_header wrote only the header, so this is the state the table
    // is in before any submit writes it — modelling the never-zeroed device SM.
    void poison_task_table() {
        auto &tasks = sm_handle->header->tasks;
        const size_t n = static_cast<size_t>(CHIP_DEFAULT_GRAPH_TASKS);
        std::memset(tasks.task_storage, POISON, n * sizeof(ChipTaskStorage));
        std::memset(tasks.task_states, POISON, n * sizeof(std::atomic<ChipTaskState>));
    }
};

TEST_F(HbgSubmitPoisonTest, EveryDeviceReadFieldIsWrittenOverPoison) {
    poison_task_table();
    orch.begin_scope();

    // 1. Zero-fanin root: a real mixed (AIV0) task with an output tensor and a scalar.
    std::vector<TensorCreateInfo> create_infos;
    create_infos.reserve(4);
    uint32_t shape[] = {16};
    CoreTaskArgs root_args;
    create_infos.emplace_back(shape, 1, DataType::FLOAT32);
    root_args.add_output(create_infos.back());
    float scale = 42.0F;
    root_args.add_scalar(scale);
    root_args.dump(create_infos.back(), scale);
    MixedKernels root_mixed{};
    root_mixed.aiv0_kernel_id = 0;
    TaskOutputTensors root = orch.submit_task(root_mixed, root_args);
    ASSERT_TRUE(root.task_id().is_valid());

    // 2. Multi-fanin dummy consumer (duplicate dep deduped to one fanin).
    TaskId deps[] = {root.task_id(), root.task_id()};
    CoreTaskArgs consumer_args;
    consumer_args.set_dependencies(deps, 2);
    TaskOutputTensors consumer = orch.submit_dummy_task(consumer_args);
    ASSERT_TRUE(consumer.task_id().is_valid());

    // 3. Hidden-alloc convenience (allocates an output, no kernel).
    CoreTaskArgs alloc_args;
    create_infos.emplace_back(shape, 1, DataType::FLOAT32);
    alloc_args.add_output(create_infos.back());
    TaskOutputTensors allocated = orch.alloc_tensors(alloc_args);
    ASSERT_TRUE(allocated.task_id().is_valid());

    // 4. Plain dummy.
    CoreTaskArgs plain_args;
    TaskOutputTensors plain = orch.submit_dummy_task(plain_args);
    ASSERT_TRUE(plain.task_id().is_valid());

    orch.end_scope();

    auto &tasks = sm_handle->header->tasks;
    const int32_t total = orch.task_allocator.active_count();
    ASSERT_GE(total, 4);

    // Every claimed slot's device-read fields must carry real values, not poison.
    for (int32_t local = 0; local < total; local++) {
        SCOPED_TRACE(testing::Message() << "slot local_id=" << local);
        const ChipTaskStorage &entry = tasks.storage_at(local);
        const TaskDescriptor &desc = entry.task;
        const TaskPayload &pl = entry.payload;
        const ChipTaskSlotState &st = entry.slot;

        // Descriptor: the task id is written to this exact local id.
        EXPECT_EQ(desc.task_id.local_id(), local);
        // The progress byte is written to a real state at submit — PENDING for a
        // dispatchable task, COMPLETED for a pre-completed hidden-alloc — never a
        // poison byte (0xAA).
        const ChipTaskState sm_state = tasks.task_states[local].load(std::memory_order_relaxed);
        EXPECT_TRUE(sm_state == CHIP_TASK_PENDING || sm_state == CHIP_TASK_COMPLETED);
        // Payload counts are real, not the poison bit pattern.
        EXPECT_GE(pl.fanin_count, 0);
        EXPECT_LE(pl.fanin_count, CHIP_MAX_FANIN);
        EXPECT_GE(pl.tensor_count, 0);
        EXPECT_GE(pl.scalar_count, 0);
        // predicate.op is a dispatch-time field, read only for tasks the device
        // actually dispatches. submit_task_common writes it (NONE when unset); a
        // pre-completed hidden-alloc is never dispatched, so it does not.
        if (sm_state == CHIP_TASK_PENDING) {
            EXPECT_LE(static_cast<uint8_t>(pl.predicate.op), static_cast<uint8_t>(PredicateOp::LE));
        }
    }

    // Field-specific coverage on the real task: tensors, scalar, packed output buffer.
    const ChipTaskStorage &root_entry = tasks.storage_at(root.task_id().local_id());
    const TaskDescriptor &root_desc = root_entry.task;
    const TaskPayload &root_pl = root_entry.payload;
    EXPECT_EQ(root_pl.tensor_count, 1);
    EXPECT_EQ(root_pl.scalar_count, 1);
    EXPECT_EQ(root_pl.dump_metadata.dump_arg_mask, (uint64_t{1} << 0) | (uint64_t{1} << 1));
    EXPECT_EQ(root_pl.dump_metadata.dump_arg_flags, ARGS_DUMP_ARG_MASK_NONE);
    EXPECT_EQ(root_pl.dump_metadata.scalar_dtypes[0], static_cast<uint8_t>(DataType::FLOAT32));
    EXPECT_NE(root_desc.packed_buffer_base, POISON_PTR);
    EXPECT_NE(root_desc.packed_buffer_base, nullptr);
    EXPECT_EQ(root_desc.kernel_id[static_cast<int>(SubtaskSlot::AIV0)], 0);

    // The consumer's fanin is written: two duplicate deps dedupe to one.
    const TaskPayload &cons_pl = tasks.storage_at(consumer.task_id().local_id()).payload;
    EXPECT_EQ(cons_pl.fanin_count, 1);
    EXPECT_EQ(cons_pl.fanin_data()[0], root.task_id().local_id());
}

class HbgSubmitLaunchSpecTest : public HbgSubmitPoisonTest,
                                public ::testing::WithParamInterface<std::tuple<int, int, bool>> {};

TEST_P(HbgSubmitLaunchSpecTest, SyncRequestPreservesShapeAndOnlyMarksMultipleBlocks) {
    const auto [blocks, mask, request_sync] = GetParam();
    poison_task_table();
    orch.total_cluster_count = 2;
    orch.total_aiv_count = 4;
    orch.begin_scope();

    CoreTaskArgs producer_args;
    const auto producer = orch.submit_task(MixedKernels{20, INVALID_KERNEL_ID, INVALID_KERNEL_ID}, producer_args);
    ASSERT_TRUE(producer.task_id().is_valid());
    const TaskId dependency[] = {producer.task_id()};
    uint32_t shape[] = {16};
    TensorCreateInfo output_info(shape, 1, DataType::FLOAT32);
    CoreTaskArgs args;
    args.add_output(output_info);
    args.add_scalar(UINT64_C(0x12345678));
    args.set_dependencies(dependency, 1);
    args.launch_spec.set_block_num(static_cast<int16_t>(blocks));
    args.launch_spec.set_require_sync_start(request_sync);
    const MixedKernels kernels{
        (mask & 1) != 0 ? 10 : INVALID_KERNEL_ID,
        (mask & 2) != 0 ? 11 : INVALID_KERNEL_ID,
        (mask & 4) != 0 ? 12 : INVALID_KERNEL_ID,
    };
    const auto output = orch.submit_task(kernels, args);
    ASSERT_TRUE(output.task_id().is_valid());
    orch.end_scope();

    const auto &entry = sm_handle->header->tasks.storage_at(output.task_id().local_id());
    EXPECT_EQ(args.launch_spec.block_num(), blocks);
    EXPECT_EQ(args.launch_spec.require_sync_start(), request_sync);
    EXPECT_EQ(entry.slot.logical_block_num, blocks);
    EXPECT_EQ(entry.slot.active_mask.core_mask(), mask == 4 ? 2 : mask);
    EXPECT_EQ(entry.slot.total_required_subtasks, blocks * __builtin_popcount(static_cast<unsigned>(mask)));
    EXPECT_EQ(entry.slot.task_attrs.requires_sync_start(), request_sync && blocks > 1);
    EXPECT_EQ(entry.task.kernel_id[0], kernels.aic_kernel_id);
    EXPECT_EQ(entry.task.kernel_id[1], mask == 4 ? 12 : kernels.aiv0_kernel_id);
    EXPECT_EQ(entry.task.kernel_id[2], mask == 4 ? INVALID_KERNEL_ID : kernels.aiv1_kernel_id);
    EXPECT_EQ(output.size(), 1U);
    ASSERT_EQ(entry.payload.tensor_count, 1);
    ASSERT_EQ(entry.payload.scalar_count, 1);
    EXPECT_EQ(entry.payload.scalar_data()[0], UINT64_C(0x12345678));
    ASSERT_EQ(entry.payload.fanin_count, 1);
    EXPECT_EQ(entry.payload.fanin_data()[0], producer.task_id().local_id());
}

INSTANTIATE_TEST_SUITE_P(
    AllShapes, HbgSubmitLaunchSpecTest,
    ::testing::Combine(::testing::Values(1, 2), ::testing::Values(1, 2, 4, 3, 5, 6, 7), ::testing::Bool())
);

namespace {

std::vector<std::tuple<int, int, bool>> block_count_boundaries() {
    std::vector<std::tuple<int, int, bool>> cases;
    for (int mask : {1, 2, 4, 3, 5, 6, 7})
        for (int blocks : {0, -1, static_cast<int>(std::numeric_limits<int16_t>::min())})
            cases.emplace_back(blocks, mask, false);
    for (int mask : {3, 5, 6}) {
        cases.emplace_back(16383, mask, true);
        cases.emplace_back(16384, mask, false);
    }
    cases.emplace_back(10922, 7, true);
    cases.emplace_back(10923, 7, false);
    for (int mask : {1, 2, 4})
        cases.emplace_back(std::numeric_limits<int16_t>::max(), mask, true);
    return cases;
}

}  // namespace

class HbgSubmitBlockCountBoundary : public HbgSubmitPoisonTest,
                                    public ::testing::WithParamInterface<std::tuple<int, int, bool>> {};

TEST_P(HbgSubmitBlockCountBoundary, ValidatesRepresentableCountAndProductBeforePublishingStorage) {
    const auto [blocks, mask, valid] = GetParam();
    ASSERT_GE(blocks, std::numeric_limits<int16_t>::min());
    ASSERT_LE(blocks, std::numeric_limits<int16_t>::max());
    poison_task_table();
    orch.begin_scope();
    const int32_t count_before = orch.task_allocator.active_count();
    const auto &next_storage = sm_handle->header->tasks.storage_at(count_before);
    std::vector<uint8_t> storage_before(sizeof(next_storage));
    std::memcpy(storage_before.data(), &next_storage, storage_before.size());
    const auto state_before = sm_handle->header->tasks.task_states[count_before].load(std::memory_order_relaxed);

    uint32_t shape[] = {16};
    TensorCreateInfo output_info(shape, 1, DataType::FLOAT32);
    CoreTaskArgs args;
    args.add_output(output_info);
    args.add_scalar(UINT64_C(0x12345678));
    args.launch_spec.set_block_num(static_cast<int16_t>(blocks));
    const MixedKernels kernels{
        (mask & 1) != 0 ? 10 : INVALID_KERNEL_ID,
        (mask & 2) != 0 ? 11 : INVALID_KERNEL_ID,
        (mask & 4) != 0 ? 12 : INVALID_KERNEL_ID,
    };
    const auto output = orch.submit_task(kernels, args);
    orch.end_scope();
    EXPECT_EQ(args.launch_spec.block_num(), blocks);

    if (!valid) {
        EXPECT_FALSE(output.task_id().is_valid());
        EXPECT_EQ(output.size(), 0U);
        EXPECT_EQ(orch.fatal_code.load(std::memory_order_acquire), SIMPLER_ERROR_INVALID_ARGS);
        EXPECT_EQ(orch.task_allocator.active_count(), count_before);
        EXPECT_EQ(std::memcmp(storage_before.data(), &next_storage, storage_before.size()), 0);
        EXPECT_EQ(sm_handle->header->tasks.task_states[count_before].load(std::memory_order_relaxed), state_before);
        return;
    }

    ASSERT_TRUE(output.task_id().is_valid());
    EXPECT_EQ(orch.fatal_code.load(std::memory_order_acquire), SIMPLER_ERROR_NONE);
    EXPECT_EQ(orch.task_allocator.active_count(), count_before + 1);
    const auto &entry = sm_handle->header->tasks.storage_at(output.task_id().local_id());
    EXPECT_EQ(entry.slot.logical_block_num, blocks);
    EXPECT_EQ(entry.slot.total_required_subtasks, blocks * __builtin_popcount(static_cast<unsigned>(mask)));
    EXPECT_EQ(entry.slot.active_mask.core_mask(), mask == 4 ? 2 : mask);
    EXPECT_FALSE(entry.slot.task_attrs.requires_sync_start());
    EXPECT_EQ(entry.task.kernel_id[0], kernels.aic_kernel_id);
    EXPECT_EQ(entry.task.kernel_id[1], mask == 4 ? 12 : kernels.aiv0_kernel_id);
    EXPECT_EQ(entry.task.kernel_id[2], mask == 4 ? INVALID_KERNEL_ID : kernels.aiv1_kernel_id);
    EXPECT_EQ(output.size(), 1U);
    ASSERT_EQ(entry.payload.scalar_count, 1);
    EXPECT_EQ(entry.payload.scalar_data()[0], UINT64_C(0x12345678));
    EXPECT_EQ(
        sm_handle->header->tasks.task_states[output.task_id().local_id()].load(std::memory_order_relaxed),
        CHIP_TASK_PENDING
    );
}

INSTANTIATE_TEST_SUITE_P(
    CountsAndSignedProduct, HbgSubmitBlockCountBoundary, ::testing::ValuesIn(block_count_boundaries())
);

class HbgSubmitWideBlockCount : public HbgSubmitPoisonTest,
                                public ::testing::WithParamInterface<std::tuple<int64_t, int, bool>> {};

TEST_P(HbgSubmitWideBlockCount, RejectsOutOfRangeInputBeforePublishingStorage) {
    const auto [blocks, mask, request_sync] = GetParam();
    ASSERT_TRUE(blocks < std::numeric_limits<int16_t>::min() || blocks > std::numeric_limits<int16_t>::max());
    poison_task_table();
    orch.total_cluster_count = 2;
    orch.total_aiv_count = 4;
    orch.begin_scope();
    const int32_t count_before = orch.task_allocator.active_count();
    const auto &next_storage = sm_handle->header->tasks.storage_at(count_before);
    std::vector<uint8_t> storage_before(sizeof(next_storage));
    std::memcpy(storage_before.data(), &next_storage, storage_before.size());
    const auto state_before = sm_handle->header->tasks.task_states[count_before].load(std::memory_order_relaxed);

    uint32_t shape[] = {16};
    TensorCreateInfo output_info(shape, 1, DataType::FLOAT32);
    CoreTaskArgs args;
    args.add_output(output_info);
    args.add_scalar(UINT64_C(0x12345678));
    args.launch_spec.set_block_num(blocks);
    args.launch_spec.set_require_sync_start(request_sync);
    const MixedKernels kernels{
        (mask & 1) != 0 ? 10 : INVALID_KERNEL_ID,
        (mask & 2) != 0 ? 11 : INVALID_KERNEL_ID,
        (mask & 4) != 0 ? 12 : INVALID_KERNEL_ID,
    };
    const auto output = orch.submit_task(kernels, args);
    orch.end_scope();

    EXPECT_EQ(args.launch_spec.require_sync_start(), request_sync);
    EXPECT_FALSE(output.task_id().is_valid());
    EXPECT_EQ(output.size(), 0U);
    EXPECT_EQ(orch.fatal_code.load(std::memory_order_acquire), SIMPLER_ERROR_INVALID_ARGS);
    EXPECT_EQ(orch.task_allocator.active_count(), count_before);
    EXPECT_EQ(std::memcmp(storage_before.data(), &next_storage, storage_before.size()), 0);
    EXPECT_EQ(sm_handle->header->tasks.task_states[count_before].load(std::memory_order_relaxed), state_before);
}

INSTANTIATE_TEST_SUITE_P(
    WideValuesAndSyncRequest, HbgSubmitWideBlockCount,
    ::testing::Combine(
        ::testing::Values<int64_t>(
            32768, 65537, 131073, -65535, std::numeric_limits<int64_t>::min(), std::numeric_limits<int64_t>::max()
        ),
        ::testing::Values(1, 2, 4, 3, 5, 6, 7), ::testing::Bool()
    )
);
