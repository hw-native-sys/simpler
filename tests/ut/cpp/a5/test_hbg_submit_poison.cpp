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
 * host_build_graph no longer zero-fills the shared-memory task window (init-on-write):
 * init_header_per_ring writes only the header, and each slot's device-read fields are
 * written per task at submit (prepare_task + submit_task_common + PTO2TaskPayload::init).
 * Nothing else clears the window, so a device-read field a submit forgets to write would
 * read as 0 only by allocator accident — passing every zero-backed test and failing
 * non-deterministically on device.
 *
 * This test fills the whole per-slot window with a 0xAA poison byte before submitting a
 * representative mix, then asserts that for every claimed slot [0, total_tasks) the
 * device-read fields carry real values, not poison. Add a device-read field and forget
 * its submit-path write, and this fails in-tree.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include "utils/device_arena.h"
#include "pto_orchestrator.h"
#include "pto_shared_memory.h"

namespace {

constexpr uint8_t POISON = 0xAA;
// A void* / int32 whose bytes are all 0xAA — what an unwritten field would read as.
void *const POISON_PTR = reinterpret_cast<void *>(static_cast<uintptr_t>(0xAAAAAAAAAAAAAAAAULL));

}  // namespace

class HbgSubmitPoisonTest : public ::testing::Test {
protected:
    DeviceArena sm_arena;
    DeviceArena runtime_arena;
    PTO2SharedMemoryHandle *sm_handle = nullptr;
    PTO2OrchestratorState orch{};
    PTO2SchedulerState sched{};
    PTO2OrchestratorLayout orch_layout{};
    PTO2SchedulerLayout sched_layout{};
    std::vector<char> gm_heap;

    void SetUp() override {
        sm_handle = PTO2SharedMemoryHandle::create_and_init_default(sm_arena);
        ASSERT_NE(sm_handle, nullptr);
        gm_heap.resize(4096 * PTO2_MAX_RING_DEPTH);

        orch_layout = PTO2OrchestratorState::reserve_layout(runtime_arena, static_cast<int32_t>(PTO2_TASK_WINDOW_SIZE));
        sched_layout = PTO2SchedulerState::reserve_layout(runtime_arena);
        ASSERT_NE(runtime_arena.commit(), nullptr);

        ASSERT_TRUE(orch.init_data_from_layout(
            orch_layout, runtime_arena, sm_handle->sm_base, gm_heap.data(), 4096, PTO2_TASK_WINDOW_SIZE
        ));
        ASSERT_TRUE(sched.init_data_from_layout(sched_layout, runtime_arena, sm_handle->sm_base));
        sched.wire_arena_pointers(sched_layout, runtime_arena);
        // Same order the AICPU boots in: the slot arrays are not part of the
        // uploaded image, so nothing can push until they carry their ramp.
        sched.seed_queue_slots();
        orch.wire_arena_pointers(orch_layout, runtime_arena, &sched);
    }

    void TearDown() override {
        orch.destroy();
        sched.destroy();
        runtime_arena.release();
        sm_arena.release();
    }

    // Fill the per-slot window (descriptors / payloads / slot_states / completion_flags)
    // with poison. init_header_per_ring wrote only the header, so this is the state the
    // window is in before any submit writes it — modelling the never-zeroed device SM.
    void poison_window() {
        auto &ring = sm_handle->header->ring;  // host_build_graph is single-ring
        const size_t n = static_cast<size_t>(ring.task_window_mask) + 1;
        std::memset(ring.task_descriptors, POISON, n * sizeof(PTO2TaskDescriptor));
        std::memset(ring.task_payloads, POISON, n * sizeof(PTO2TaskPayload));
        std::memset(ring.slot_states, POISON, n * sizeof(PTO2TaskSlotState));
        std::memset(ring.completion_flags, POISON, n * sizeof(std::atomic<uint8_t>));
    }
};

TEST_F(HbgSubmitPoisonTest, EveryDeviceReadFieldIsWrittenOverPoison) {
    poison_window();
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
    PTO2TaskId deps[] = {root.task_id(), root.task_id()};
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

    auto &ring = sm_handle->header->ring;
    const int32_t total = ring.fc.current_task_index.load(std::memory_order_acquire);
    ASSERT_GE(total, 4);

    // Every claimed slot's device-read fields must carry real values, not poison.
    for (int32_t local = 0; local < total; local++) {
        SCOPED_TRACE(testing::Message() << "slot local_id=" << local);
        const int32_t slot = ring.get_slot_by_task_id(local);
        const PTO2TaskDescriptor &desc = ring.task_descriptors[slot];
        const PTO2TaskSlotState &st = ring.slot_states[slot];
        const PTO2TaskPayload &pl = *st.payload;

        // Descriptor: the task id is written to this exact local id.
        EXPECT_EQ(desc.task_id.local(), static_cast<uint32_t>(local));
        // task_state is written at submit (reset_for_reuse skips it): PENDING for a
        // dispatchable task, COMPLETED for a pre-completed hidden-alloc. Either way a
        // real enum, never poison.
        const PTO2TaskState state = st.task_state.load(std::memory_order_relaxed);
        EXPECT_TRUE(state == PTO2_TASK_PENDING || state == PTO2_TASK_COMPLETED);
        // Completion flag is written to a real 0/1 (pending vs pre-completed), not a
        // poison byte (0xAA).
        const uint8_t cflag = ring.completion_flags[slot].load(std::memory_order_relaxed);
        EXPECT_LE(cflag, uint8_t{1});
        // Payload counts are real, not the poison bit pattern.
        EXPECT_GE(pl.fanin_count, 0);
        EXPECT_LE(pl.fanin_count, PTO2_MAX_FANIN);
        EXPECT_GE(pl.tensor_count, 0);
        EXPECT_GE(pl.scalar_count, 0);
        // predicate.op is a dispatch-time field, read only for tasks the device
        // actually dispatches. submit_task_common writes it (NONE when unset); a
        // pre-completed hidden-alloc is never dispatched, so it does not.
        if (state == PTO2_TASK_PENDING) {
            EXPECT_LE(static_cast<uint8_t>(pl.predicate.op), static_cast<uint8_t>(PredicateOp::LE));
        }
    }

    // Field-specific coverage on the real task: tensors, scalar, packed output buffer.
    const int32_t root_slot = ring.get_slot_by_task_id(root.task_id().local());
    const PTO2TaskDescriptor &root_desc = ring.task_descriptors[root_slot];
    const PTO2TaskPayload &root_pl = *ring.slot_states[root_slot].payload;
    EXPECT_EQ(root_pl.tensor_count, 1);
    EXPECT_EQ(root_pl.scalar_count, 1);
    EXPECT_EQ(root_pl.dump_metadata.dump_arg_mask, (uint64_t{1} << 0) | (uint64_t{1} << 1));
    EXPECT_EQ(root_pl.dump_metadata.dump_arg_flags, ARGS_DUMP_ARG_MASK_NONE);
    EXPECT_EQ(root_pl.dump_metadata.scalar_dtypes[0], static_cast<uint8_t>(DataType::FLOAT32));
    EXPECT_NE(root_desc.packed_buffer_base, POISON_PTR);
    EXPECT_NE(root_desc.packed_buffer_base, nullptr);
    EXPECT_EQ(root_desc.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIV0)], 0);

    // The consumer's fanin is written: two duplicate deps dedupe to one.
    const int32_t consumer_slot = ring.get_slot_by_task_id(consumer.task_id().local());
    const PTO2TaskPayload &cons_pl = *ring.slot_states[consumer_slot].payload;
    EXPECT_EQ(cons_pl.fanin_count, 1);
    EXPECT_EQ(cons_pl.fanin_local_ids[0], static_cast<int32_t>(root.task_id().local()));
}
