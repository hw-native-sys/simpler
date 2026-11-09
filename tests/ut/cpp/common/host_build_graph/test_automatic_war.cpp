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

#include "host_build_graph/host/dep_compute.h"

class AutomaticWarTest : public ::testing::Test {
protected:
    ChipTensorMap map{};

    void SetUp() override { ASSERT_TRUE(map.init(16, 512, 32)); }

    std::set<uint64_t> submit(const simpler::hbg::Tensor &tensor, TensorArgType type, TaskId id, bool manual = false) {
        TensorRef ref;
        ref = &tensor;
        DepInputs inputs{1, &ref, &type, 0, nullptr};
        std::set<uint64_t> predecessors;
        EXPECT_TRUE(compute_task_fanin(inputs, map, manual, [&](TaskId predecessor) {
            predecessors.insert(TaskId::to_uint64(predecessor));
            return true;
        }));
        register_task_accesses(inputs, id, map, manual);
        return predecessors;
    }
};

TEST_F(AutomaticWarTest, OrdinaryInputOrdersLaterWriterWithoutSerializingReaders) {
    uint32_t shape[] = {64};
    auto tensor = simpler::hbg::make_tensor_external(reinterpret_cast<void *>(0x1000), shape, 1, DataType::FLOAT32);
    auto writer = TaskId::make_global(0);
    auto reader0 = TaskId::make_global(1);
    auto reader1 = TaskId::make_global(2);
    EXPECT_TRUE(submit(tensor, TensorArgType::INOUT, writer).empty());
    EXPECT_EQ(submit(tensor, TensorArgType::INPUT, reader0), std::set<uint64_t>{TaskId::to_uint64(writer)});
    EXPECT_EQ(submit(tensor, TensorArgType::INPUT, reader1), std::set<uint64_t>{TaskId::to_uint64(writer)});
    auto dependencies = submit(tensor, TensorArgType::INOUT, TaskId::make_global(3));
    EXPECT_EQ(dependencies.count(TaskId::to_uint64(reader0)), 1u);
    EXPECT_EQ(dependencies.count(TaskId::to_uint64(reader1)), 1u);
}

TEST_F(AutomaticWarTest, ExistingOutputWaitsForPureReaderWithoutAProducer) {
    uint32_t shape[] = {64};
    auto tensor = simpler::hbg::make_tensor_external(reinterpret_cast<void *>(0x1000), shape, 1, DataType::FLOAT32);
    auto reader = TaskId::make_global(2);
    EXPECT_TRUE(submit(tensor, TensorArgType::INPUT, reader).empty());
    EXPECT_EQ(
        submit(tensor, TensorArgType::OUTPUT_EXISTING, TaskId::make_global(0)),
        std::set<uint64_t>{TaskId::to_uint64(reader)}
    );
}

TEST_F(AutomaticWarTest, ResetDiscardsBothAccessIndexes) {
    uint32_t shape[] = {16};
    auto tensor = simpler::hbg::make_tensor_external(reinterpret_cast<void *>(0x1000), shape, 1);
    submit(tensor, TensorArgType::INOUT, TaskId::make_global(0));
    submit(tensor, TensorArgType::INPUT, TaskId::make_global(1));
    map.reset();
    EXPECT_TRUE(submit(tensor, TensorArgType::INOUT, TaskId::make_global(0)).empty());
}
