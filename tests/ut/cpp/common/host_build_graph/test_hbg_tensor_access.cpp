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

#include <array>
#include <thread>

#include "host_build_graph/host_tensor_access.h"

namespace {
ChipTensor host_view(std::array<uint32_t, 8> &storage) {
    const uint32_t shape[] = {8};
    ChipTensor tensor;
    tensor.init_external(storage.data(), sizeof(storage), shape, 1, DataType::UINT32, AddressSpace::HOST);
    tensor.transfer = TensorTransfer::NONE;
    return tensor;
}

TEST(HostTensorAccessTest, DeclaredDirectionControlsReadsAndWrites) {
    for (auto direction : {ArgDirection::IN, ArgDirection::OUT, ArgDirection::INOUT}) {
        SCOPED_TRACE(static_cast<int>(direction));
        std::array<uint32_t, 8> storage{17};
        auto tensor = host_view(storage);
        HostTensorAccessor access;
        ASSERT_TRUE(access.add(tensor, direction));
        uint32_t read = 0;
        EXPECT_EQ(access.read(tensor.buffer.addr, &read, sizeof(read)), direction != ArgDirection::OUT);
        EXPECT_EQ(read, direction == ArgDirection::OUT ? 0u : 17u);
        const uint32_t written = 29;
        EXPECT_EQ(access.write(tensor.buffer.addr, &written, sizeof(written)), direction != ArgDirection::IN);
        EXPECT_EQ(storage[0], direction == ArgDirection::IN ? 17u : written);
        access.close();
        EXPECT_FALSE(access.read(tensor.buffer.addr, &read, sizeof(read)));
        EXPECT_FALSE(access.write(tensor.buffer.addr, &written, sizeof(written)));
    }
}

TEST(HostTensorAccessTest, DeviceAndTransferTagsCannotCreateHostViews) {
    std::array<uint32_t, 8> storage{};
    for (auto space : {AddressSpace::HOST, AddressSpace::DEVICE}) {
        for (auto transfer : {TensorTransfer::NONE, TensorTransfer::H2D, TensorTransfer::D2H}) {
            if (space == AddressSpace::HOST && transfer == TensorTransfer::NONE) continue;
            HostTensorAccessor access;
            auto tensor = host_view(storage);
            tensor.buffer.addr = 1;
            tensor.address_space = space;
            tensor.transfer = transfer;
            EXPECT_FALSE(access.add(tensor, ArgDirection::INOUT));
            uint32_t value = 0;
            EXPECT_FALSE(access.read(1, &value, sizeof(value)));
            EXPECT_FALSE(access.write(1, &value, sizeof(value)));
        }
    }
}

TEST(HostTensorAccessTest, StridedViewBoundsIncludeItsOffsetAndExcludeOutsideBacking) {
    std::array<uint32_t, 8> storage{0, 11, 0, 13, 0, 15, 0, 17};
    auto tensor = host_view(storage);
    tensor.shapes[0] = 3;
    tensor.strides[0] = 2;
    tensor.start_offset = 1;
    HostTensorAccessor access;
    ASSERT_TRUE(access.add(tensor, ArgDirection::IN));
    for (uint32_t i = 0; i < 3; ++i) {
        uint32_t value = 0;
        ASSERT_TRUE(access.read(tensor.buffer.addr + (1 + 2 * i) * sizeof(value), &value, sizeof(value)));
        EXPECT_EQ(value, 11 + 2 * i);
    }
    uint32_t value = 0;
    EXPECT_FALSE(access.read(tensor.buffer.addr, &value, sizeof(value)));
    EXPECT_FALSE(access.read(tensor.buffer.addr + 6 * sizeof(value), &value, sizeof(value)));
    EXPECT_FALSE(access.read(UINT64_MAX - 1, &value, sizeof(value)));
}

TEST(HostTensorAccessTest, OverlappingReadOnlyViewsAreAllowedButWritersAreRejected) {
    for (auto first : {ArgDirection::IN, ArgDirection::OUT, ArgDirection::INOUT}) {
        for (auto second : {ArgDirection::IN, ArgDirection::OUT, ArgDirection::INOUT}) {
            std::array<uint32_t, 8> storage{};
            auto tensor = host_view(storage);
            HostTensorAccessor access;
            ASSERT_TRUE(access.add(tensor, first));
            tensor.start_offset = 3;
            tensor.shapes[0] = 2;
            EXPECT_EQ(access.add(tensor, second), first == ArgDirection::IN && second == ArgDirection::IN);
        }
    }
}

TEST(HostTensorAccessTest, InvalidGeometryRegistersNothing) {
    std::array<uint32_t, 8> storage{};
    for (int mutation = 0; mutation < 8; ++mutation) {
        SCOPED_TRACE(mutation);
        auto tensor = host_view(storage);
        switch (mutation) {
        case 0:
            tensor.ndims = 0;
            break;
        case 1:
            tensor.ndims = MAX_TENSOR_DIMS + 1;
            break;
        case 2:
            tensor.dtype = DataType::DATA_TYPE_NUM;
            break;
        case 3:
            tensor.strides[0] = 0;
            break;
        case 4:
            tensor.start_offset = 9;
            break;
        case 5:
            tensor.buffer.size = sizeof(storage) - 1;
            break;
        case 6:
            tensor.buffer.addr = 0;
            break;
        case 7:
            tensor.buffer.addr = UINT64_MAX - 1;
            break;
        }
        HostTensorAccessor access;
        EXPECT_FALSE(access.add(tensor, ArgDirection::IN));
        uint32_t value = 0;
        EXPECT_FALSE(access.read(tensor.buffer.addr, &value, sizeof(value)));
    }
}

TEST(HostTensorAccessTest, EmptyViewRegistersNoReadableBytes) {
    std::array<uint32_t, 8> storage{};
    auto tensor = host_view(storage);
    tensor.shapes[0] = 0;
    tensor.strides[0] = 0;
    tensor.buffer = {};
    HostTensorAccessor access;
    ASSERT_TRUE(access.add(tensor, ArgDirection::IN));
    uint32_t value = 0;
    EXPECT_FALSE(access.read(0, &value, sizeof(value)));
}

TEST(HostTensorAccessTest, IndependentRunsCanReadOnRecordingThreads) {
    std::array<uint32_t, 8> first{3}, second{7};
    HostTensorAccessor a, b;
    ASSERT_TRUE(a.add(host_view(first), ArgDirection::IN));
    ASSERT_TRUE(b.add(host_view(second), ArgDirection::IN));
    uint32_t x = 0, y = 0;
    std::thread t([&] {
        EXPECT_TRUE(a.read(reinterpret_cast<uint64_t>(first.data()), &x, sizeof(x)));
    });
    EXPECT_TRUE(b.read(reinterpret_cast<uint64_t>(second.data()), &y, sizeof(y)));
    t.join();
    EXPECT_EQ(x, 3u);
    EXPECT_EQ(y, 7u);
}
}  // namespace
