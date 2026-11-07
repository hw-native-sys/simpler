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

#include <set>

#include "dep_compute.h"
#include "utils/device_arena.h"

class AutomaticWarTest : public ::testing::Test {
protected:
    DeviceArena arena;
    ChipTensorMap map{};
    ChipTensorMapLayout layout{};

    void SetUp() override {
        int32_t windows[CHIP_MAX_RING_DEPTH];
        for (auto &window : windows)
            window = 32;
        layout = ChipTensorMap::reserve_layout(arena, 16, 512, windows);
        ASSERT_NE(arena.commit(), nullptr);
        ASSERT_TRUE(map.init_data_from_layout(layout, arena));
        map.wire_arena_pointers(layout, arena);
    }

    void TearDown() override {
        map.destroy();
        arena.release();
    }

    std::set<uint64_t> submit(const simpler::tmr::Tensor &tensor, TensorArgType type, TaskId id, bool manual = false) {
        TensorRef ref;
        ref = &tensor;
        DepInputs inputs{1, &ref, &type, 0, nullptr};
        std::set<uint64_t> predecessors;
        EXPECT_TRUE(compute_task_fanin(inputs, map, manual, [&](TaskId predecessor, DepFlags flags) {
            EXPECT_NE(flags & DEP_WAIT, DEP_NONE);
            predecessors.insert(TaskId::to_uint64(predecessor));
            return true;
        }));
        register_task_accesses(inputs, id, map, manual);
        return predecessors;
    }
};

TEST_F(AutomaticWarTest, OrdinaryInputOrdersLaterWriterWithoutSerializingReaders) {
    uint32_t shape[] = {64};
    auto tensor = simpler::tmr::make_tensor_external(reinterpret_cast<void *>(0x1000), shape, 1, DataType::FLOAT32);
    auto writer = TaskId::make(0, 0);
    auto reader0 = TaskId::make(0, 1);
    auto reader1 = TaskId::make(1, 0);
    EXPECT_TRUE(submit(tensor, TensorArgType::INOUT, writer).empty());
    EXPECT_EQ(submit(tensor, TensorArgType::INPUT, reader0), std::set<uint64_t>{TaskId::to_uint64(writer)});
    EXPECT_EQ(submit(tensor, TensorArgType::INPUT, reader1), std::set<uint64_t>{TaskId::to_uint64(writer)});
    auto dependencies = submit(tensor, TensorArgType::INOUT, TaskId::make(0, 2));
    EXPECT_EQ(dependencies.count(TaskId::to_uint64(reader0)), 1u);
    EXPECT_EQ(dependencies.count(TaskId::to_uint64(reader1)), 1u);
}

TEST_F(AutomaticWarTest, ExistingOutputWaitsForPureReaderWithoutAProducer) {
    uint32_t shape[] = {64};
    auto tensor = simpler::tmr::make_tensor_external(reinterpret_cast<void *>(0x1000), shape, 1, DataType::FLOAT32);
    auto reader = TaskId::make(1, 0);
    EXPECT_TRUE(submit(tensor, TensorArgType::INPUT, reader).empty());
    EXPECT_EQ(
        submit(tensor, TensorArgType::OUTPUT_EXISTING, TaskId::make(0, 0)),
        std::set<uint64_t>{TaskId::to_uint64(reader)}
    );
}

TEST_F(AutomaticWarTest, PartialWritesKeepTheReaderForUnwrittenRegions) {
    uint32_t whole_shape[] = {64};
    uint32_t part_shape[] = {16};
    auto whole = simpler::tmr::make_tensor_external(reinterpret_cast<void *>(0x1000), whole_shape, 1);
    auto first = simpler::tmr::make_tensor_external(reinterpret_cast<void *>(0x1000), part_shape, 1);
    auto last = first;
    last.start_offset = 48;
    auto reader = TaskId::make(0, 0);
    submit(whole, TensorArgType::INPUT, reader);
    EXPECT_EQ(submit(first, TensorArgType::INOUT, TaskId::make(0, 1)).count(TaskId::to_uint64(reader)), 1u);
    EXPECT_EQ(submit(last, TensorArgType::INOUT, TaskId::make(0, 2)).count(TaskId::to_uint64(reader)), 1u);
}

TEST_F(AutomaticWarTest, DisjointViewsDoNotCreateWar) {
    uint32_t shape[] = {16};
    auto first = simpler::tmr::make_tensor_external(reinterpret_cast<void *>(0x1000), shape, 1);
    auto last = first;
    last.start_offset = 32;
    submit(first, TensorArgType::INPUT, TaskId::make(0, 0));
    EXPECT_TRUE(submit(last, TensorArgType::INOUT, TaskId::make(0, 1)).empty());
}

TEST_F(AutomaticWarTest, RetiredReadersDoNotConstrainLaterWriters) {
    uint32_t shape[] = {16};
    auto tensor = simpler::tmr::make_tensor_external(reinterpret_cast<void *>(0x1000), shape, 1);
    submit(tensor, TensorArgType::INPUT, TaskId::make(1, 0));
    map.sync_validity(1, 1);
    map.cleanup_retired(1, 0, 1);
    EXPECT_TRUE(submit(tensor, TensorArgType::INOUT, TaskId::make(0, 0)).empty());
}

TEST_F(AutomaticWarTest, ManualScopesAndManualTensorsKeepExplicitDependencyContract) {
    uint32_t shape[] = {16};
    auto tensor = simpler::tmr::make_tensor_external(reinterpret_cast<void *>(0x1000), shape, 1);
    submit(tensor, TensorArgType::INPUT, TaskId::make(0, 0), true);
    EXPECT_TRUE(submit(tensor, TensorArgType::INOUT, TaskId::make(0, 1)).empty());
    tensor.manual_dep = true;
    submit(tensor, TensorArgType::INPUT, TaskId::make(0, 2));
    EXPECT_TRUE(submit(tensor, TensorArgType::INOUT, TaskId::make(0, 3)).empty());
}

TEST_F(AutomaticWarTest, NoDepDoesNotPublishAnAccess) {
    uint32_t shape[] = {16};
    auto tensor = simpler::tmr::make_tensor_external(reinterpret_cast<void *>(0x1000), shape, 1);
    submit(tensor, TensorArgType::NO_DEP, TaskId::make(0, 0));
    EXPECT_TRUE(submit(tensor, TensorArgType::INOUT, TaskId::make(0, 1)).empty());
}

TEST_F(AutomaticWarTest, ReuseAndEpochWrapDiscardReaderEntries) {
    uint32_t shape[] = {16};
    auto tensor = simpler::tmr::make_tensor_external(reinterpret_cast<void *>(0x1000), shape, 1);
    submit(tensor, TensorArgType::INPUT, TaskId::make(0, 0));
    map.reset_for_reuse(layout);
    EXPECT_TRUE(submit(tensor, TensorArgType::INOUT, TaskId::make(0, 1)).empty());
    map.current_epoch = UINT32_MAX;
    submit(tensor, TensorArgType::INPUT, TaskId::make(0, 2));
    map.reset_for_reuse(layout);
    EXPECT_EQ(map.current_epoch, 1u);
    EXPECT_TRUE(submit(tensor, TensorArgType::INOUT, TaskId::make(0, 3)).empty());
}
