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

#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include "task_args.h"

// ---------------------------------------------------------------------------
// ContinuousTensor layout
// ---------------------------------------------------------------------------

// ABI contract: size must match serialization format.
TEST(ChildMemory, SizeofUnchanged) { EXPECT_EQ(sizeof(ContinuousTensor), 40u); }

TEST(ChildMemory, DefaultIsZero) {
    ContinuousTensor t{};
    EXPECT_EQ(t.child_memory, 0);
    EXPECT_FALSE(t.is_child_memory());
}

TEST(ChildMemory, SetChildMemory) {
    ContinuousTensor t{};
    t.data = 0xDEAD0000;
    t.shapes[0] = 16;
    t.ndims = 1;
    t.dtype = DataType::FLOAT32;
    t.child_memory = 1;

    EXPECT_TRUE(t.is_child_memory());
    EXPECT_EQ(t.data, 0xDEAD0000u);
    EXPECT_EQ(t.nbytes(), 16u * 4u);
}

// ---------------------------------------------------------------------------
// write_blob / read_blob roundtrip preserves child_memory
// ---------------------------------------------------------------------------

TEST(ChildMemory, BlobRoundtripPreservesChildMemory) {
    TaskArgs args;

    ContinuousTensor host_t{};
    host_t.data = 0x1000;
    host_t.shapes[0] = 4;
    host_t.ndims = 1;
    host_t.dtype = DataType::FLOAT32;
    host_t.child_memory = 0;
    args.add_tensor(host_t, TensorArgType::INPUT);

    ContinuousTensor dev_t{};
    dev_t.data = 0x2000;
    dev_t.shapes[0] = 8;
    dev_t.ndims = 1;
    dev_t.dtype = DataType::FLOAT16;
    dev_t.child_memory = 1;
    args.add_tensor(dev_t, TensorArgType::INPUT);

    args.add_scalar(42);

    // Serialize
    size_t blob_size = task_args_blob_size(args);
    std::vector<uint8_t> buf(blob_size);
    write_blob(buf.data(), args);

    // Deserialize (test owns the buffer, so capacity = blob_size).
    TaskArgsView view = read_blob(buf.data(), blob_size);
    ASSERT_EQ(view.tensor_count, 2);
    ASSERT_EQ(view.scalar_count, 1);

    EXPECT_EQ(view.tensors[0].child_memory, 0);
    EXPECT_FALSE(view.tensors[0].is_child_memory());

    EXPECT_EQ(view.tensors[1].child_memory, 1);
    EXPECT_TRUE(view.tensors[1].is_child_memory());
    EXPECT_EQ(view.tensors[1].data, 0x2000u);
}

// ---------------------------------------------------------------------------
// view_to_chip_storage preserves child_memory
// ---------------------------------------------------------------------------

TEST(ChildMemory, ViewToChipStoragePreservesChildMemory) {
    ContinuousTensor tensors[2] = {};
    tensors[0].data = 0xA000;
    tensors[0].shapes[0] = 1;
    tensors[0].ndims = 1;
    tensors[0].dtype = DataType::INT32;
    tensors[0].child_memory = 0;

    tensors[1].data = 0xB000;
    tensors[1].shapes[0] = 2;
    tensors[1].ndims = 1;
    tensors[1].dtype = DataType::INT32;
    tensors[1].child_memory = 1;

    uint64_t scalars[] = {99};
    TaskArgsView view{2, 1, tensors, scalars};

    ChipStorageTaskArgs chip = view_to_chip_storage(view);

    ASSERT_EQ(chip.tensor_count(), 2);
    EXPECT_FALSE(chip.tensor(0).is_child_memory());
    EXPECT_TRUE(chip.tensor(1).is_child_memory());
    EXPECT_EQ(chip.tensor(1).data, 0xB000u);
}

// ---------------------------------------------------------------------------
// Mixed: child_memory tensors should NOT be recorded as tensor pairs
// (simulates what init_runtime_impl should do)
// ---------------------------------------------------------------------------

TEST(ChildMemory, SkipLogicSimulation) {
    // Simulate the init_runtime_impl loop: count how many tensors would be
    // malloc'd vs passed-through.
    ChipStorageTaskArgs args;

    ContinuousTensor host_t{};
    host_t.data = 0x1000;
    host_t.shapes[0] = 4;
    host_t.ndims = 1;
    host_t.dtype = DataType::FLOAT32;
    host_t.child_memory = 0;
    args.add_tensor(host_t);

    ContinuousTensor dev_t{};
    dev_t.data = 0x2000;
    dev_t.shapes[0] = 8;
    dev_t.ndims = 1;
    dev_t.dtype = DataType::FLOAT32;
    dev_t.child_memory = 1;
    args.add_tensor(dev_t);

    int malloc_count = 0;
    int passthrough_count = 0;

    for (int i = 0; i < args.tensor_count(); i++) {
        ContinuousTensor t = args.tensor(i);
        if (t.is_child_memory()) {
            passthrough_count++;
        } else {
            malloc_count++;
        }
    }

    EXPECT_EQ(malloc_count, 1);
    EXPECT_EQ(passthrough_count, 1);
}
