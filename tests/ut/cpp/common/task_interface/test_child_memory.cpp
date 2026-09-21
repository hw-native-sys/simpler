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

#include "task_args_wire.h"

// ---------------------------------------------------------------------------
// ChipTensor layout
// ---------------------------------------------------------------------------

// ABI contract: ChipStorageTaskArgs is handed to runtime.so through a `const void *`,
// so nothing checks this element's size at the call. 72 B is geometry plus a resolved
// address; a change here silently mis-reads every argument.
TEST(ChildMemory, TensorAbiSize) { EXPECT_EQ(sizeof(ChipTensor), 72u); }

TEST(ChildMemory, DefaultIsZero) {
    ChipTensor t{};
    EXPECT_EQ(t.address_space, AddressSpace::HOST);
    EXPECT_FALSE(t.is_device_memory());
}

TEST(ChildMemory, SetChildMemory) {
    ChipTensor t{};
    t.buffer.addr = 0xDEAD0000;
    t.shapes[0] = 16;
    t.ndims = 1;
    t.dtype = DataType::FLOAT32;
    t.address_space = AddressSpace::DEVICE;

    EXPECT_TRUE(t.is_device_memory());
    EXPECT_EQ(t.buffer.addr, 0xDEAD0000u);
    EXPECT_EQ(t.nbytes(), 16u * 4u);
}

// ---------------------------------------------------------------------------
// write_blob / read_blob roundtrip preserves each wire tensor's address space
// ---------------------------------------------------------------------------

namespace {

// A minimal valid wire tensor over a POSIX_SHM backing: a 4-element FLOAT32 view at the origin.
Tensor make_wire_tensor(uint64_t buffer_id, AddressSpace space) {
    static constexpr const char *kShmName = "psm_child_memory";
    Tensor r{};
    r.buffer.magic = BUFFER_DESCRIPTOR_MAGIC;
    r.buffer.address_space = static_cast<uint8_t>(space);
    r.buffer.access = static_cast<uint8_t>(AccessMode::READWRITE);
    r.buffer.identity.buffer_id = buffer_id;
    r.buffer.identity.generation = 1;
    r.buffer.nbytes = 64;
    if (space == AddressSpace::DEVICE) {
        r.buffer.backend_kind = static_cast<uint8_t>(BackendKind::DEVICE_MALLOC);
        r.buffer.body_len = static_cast<uint16_t>(BACKEND_ADDRESS_BODY_BYTES);
        uint64_t base = 0x2000;
        std::memcpy(r.buffer.body, &base, sizeof(base));
    } else {
        r.buffer.backend_kind = static_cast<uint8_t>(BackendKind::POSIX_SHM);
        r.buffer.body_len = static_cast<uint16_t>(std::strlen(kShmName));
        std::memcpy(r.buffer.body, kShmName, r.buffer.body_len);
    }
    r.byte_offset = 0;
    r.ndims = 1;
    r.shapes[0] = 4;
    r.strides[0] = 1;
    r.dtype = DataType::FLOAT32;
    return r;
}

}  // namespace

TEST(ChildMemory, BlobRoundtripPreservesAddressSpace) {
    TaskArgs args;
    args.add_tensor(make_wire_tensor(1, AddressSpace::HOST), TensorArgType::INPUT);
    args.add_tensor(make_wire_tensor(2, AddressSpace::DEVICE), TensorArgType::INPUT);
    args.add_scalar(42);

    size_t blob_size = task_args_blob_size(args);
    std::vector<uint8_t> buf(blob_size);
    write_blob(buf.data(), args);

    // Test owns the buffer, so capacity = blob_size.
    TaskArgsView view = read_blob(buf.data(), blob_size);
    ASSERT_EQ(view.tensor_count, 2);
    ASSERT_EQ(view.scalar_count, 1);

    EXPECT_EQ(view.tensors(0).buffer.address_space, static_cast<uint8_t>(AddressSpace::HOST));
    EXPECT_EQ(view.tensors(1).buffer.address_space, static_cast<uint8_t>(AddressSpace::DEVICE));
    EXPECT_EQ(view.tensors(1).buffer.identity.buffer_id, 2u);
    EXPECT_EQ(view.scalars[0], 42u);
}

// ---------------------------------------------------------------------------
// Mixed: child_memory tensors should NOT be recorded as tensor pairs
// (simulates what init_runtime_impl should do)
// ---------------------------------------------------------------------------

TEST(ChildMemory, SkipLogicSimulation) {
    // Simulate the init_runtime_impl loop: count how many tensors would be
    // malloc'd vs passed-through.
    ChipStorageTaskArgs args;

    ChipTensor host_t{};
    host_t.buffer.addr = 0x1000;
    host_t.shapes[0] = 4;
    host_t.ndims = 1;
    host_t.dtype = DataType::FLOAT32;
    host_t.address_space = AddressSpace::HOST;
    args.add_tensor(host_t);

    ChipTensor dev_t{};
    dev_t.buffer.addr = 0x2000;
    dev_t.shapes[0] = 8;
    dev_t.ndims = 1;
    dev_t.dtype = DataType::FLOAT32;
    dev_t.address_space = AddressSpace::DEVICE;
    args.add_tensor(dev_t);

    int malloc_count = 0;
    int passthrough_count = 0;

    for (int i = 0; i < args.tensor_count(); i++) {
        ChipTensor t = args.tensor(i);
        if (t.is_device_memory()) {
            passthrough_count++;
        } else {
            malloc_count++;
        }
    }

    EXPECT_EQ(malloc_count, 1);
    EXPECT_EQ(passthrough_count, 1);
}
