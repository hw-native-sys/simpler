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

#include <cstring>
#include <limits>
#include <vector>

#include "tensormap_and_ringbuffer/entry_args.h"
#include "tensormap_and_ringbuffer/kernel_invocation.h"

namespace {
using namespace simpler::tmr;

class KernelInvocation : public ::testing::Test {
protected:
    void SetUp() override {
        ChipTensor tensor{};
        tensor.buffer = {0x10000, 256};
        tensor.dtype = DataType::FLOAT32;
        tensor.address_space = AddressSpace::DEVICE;
        tensor.ndims = 1;
        tensor.shapes[0] = 8;
        tensor.strides[0] = 2;
        tensor.start_offset = 3;
        args.add_tensor(Tensor::from_boundary(tensor));
        args.add_scalar(73);
        packet.resize(kernel_invocation_bytes(1, 1));
        ASSERT_TRUE(encode_kernel_invocation(args, identity, binding, packet.data(), packet.size()));
    }

    bool consume(const KernelCallableIdentity &id, EagerKernelBinding bind) {
        return consume_kernel_invocation(packet.data(), packet.size(), id, bind, &output);
    }

    EntryArgsStorage args{};
    EntryArgsStorage output{};
    KernelCallableIdentity identity{3, 7, 1, 1};
    EagerKernelBinding binding{0x20000, 19, {0x40000, 23, 29}};
    std::vector<unsigned char> packet;
};

TEST_F(KernelInvocation, SnapshotRetainsAddressScalarAndStridedView) {
    args.tensors_[0].buffer.addr = 0x30000;
    args.scalars_[0] = 91;
    ASSERT_TRUE(consume(identity, binding));
    EXPECT_EQ(output.tensor(0).buffer.addr, 0x10000U);
    EXPECT_EQ(output.tensor(0).start_offset, 3U);
    EXPECT_EQ(output.tensor(0).strides[0], 2U);
    EXPECT_EQ(output.scalar(0), 73U);
    EXPECT_EQ(sizeof(SimplerKernelInvocationHeader), 40U);
}

TEST_F(KernelInvocation, ReusedCallableIdRejectsOldOccupancy) {
    auto occupant = identity;
    ++occupant.generation;
    EXPECT_FALSE(consume(occupant, binding));
    EXPECT_TRUE(consume(identity, binding));
}

TEST_F(KernelInvocation, RuntimeAddressAndEpochBothBelongToThisRun) {
    EXPECT_FALSE(consume(identity, {binding.runtime_address + 64, binding.run_epoch, binding.reference}));
    EXPECT_FALSE(consume(identity, {binding.runtime_address, binding.run_epoch + 1, binding.reference}));
    EXPECT_FALSE(consume(identity, {binding.runtime_address, 0, binding.reference}));
}

TEST_F(KernelInvocation, RegisteredReferenceCannotBeSubstitutedOrOmitted) {
    ASSERT_TRUE(consume(identity, binding));
    for (size_t field = 0; field < 3; ++field) {
        auto stale = binding;
        if (field == 0) stale.reference.control_address += 64;
        if (field == 1) ++stale.reference.capture_generation;
        if (field == 2) ++stale.reference.context_generation;
        EXPECT_FALSE(consume(identity, stale));
        EXPECT_EQ(output.scalar(0), 73U);
    }
    EXPECT_FALSE(consume(identity, {binding.runtime_address, binding.run_epoch, {}}));
}

TEST_F(KernelInvocation, UnboundColdSnapshotCannotBeConsumedAsAnExecution) {
    auto staged = binding;
    staged.reference = {};
    ASSERT_TRUE(encode_kernel_invocation(args, identity, staged, packet.data(), packet.size()));
    EXPECT_FALSE(consume(identity, staged));
    EXPECT_FALSE(consume(identity, binding));
    const auto before = packet;
    staged.reference.capture_generation = 1;
    EXPECT_FALSE(encode_kernel_invocation(args, identity, staged, packet.data(), packet.size()));
    EXPECT_EQ(packet, before);
}

TEST_F(KernelInvocation, InvalidFramingDoesNotChangePreviouslyAdmittedStorage) {
    ASSERT_TRUE(consume(identity, binding));
    const auto original = packet;
    const size_t fields[] = {
        offsetof(SimplerKernelInvocationHeader, mode),
        offsetof(SimplerKernelInvocationHeader, callable_id),
        offsetof(SimplerKernelInvocationHeader, generation),
        offsetof(SimplerKernelInvocationHeader, payload_bytes),
        offsetof(SimplerKernelInvocationHeader, tensor_count),
        offsetof(SimplerKernelInvocationHeader, scalar_count),
        offsetof(SimplerKernelInvocationHeader, host_copy_tensor_count),
        offsetof(SimplerKernelInvocationHeader, reserved_)
    };
    for (const size_t offset : fields) {
        packet = original;
        packet[offset] ^= 0x80;
        EXPECT_FALSE(consume(identity, binding)) << offset;
        EXPECT_EQ(output.scalar(0), 73U);
        EXPECT_EQ(output.tensor(0).buffer.addr, 0x10000U);
    }
    EXPECT_FALSE(consume_kernel_invocation(original.data(), original.size() - 1, identity, binding, &output));
}

TEST_F(KernelInvocation, InvalidTensorIsRejectedBeforeOutputChanges) {
    ASSERT_TRUE(consume(identity, binding));
    ChipTensor tensor{};
    const size_t offset = sizeof(SimplerKernelInvocationHeader) + sizeof(EagerKernelBinding);
    std::memcpy(&tensor, packet.data() + offset, sizeof(tensor));
    tensor.buffer.size = 4;
    std::memcpy(packet.data() + offset, &tensor, sizeof(tensor));
    EXPECT_FALSE(consume(identity, binding));
    EXPECT_EQ(output.scalar(0), 73U);
    EXPECT_EQ(output.tensor(0).buffer.size, 256U);
}

TEST_F(KernelInvocation, InvalidEncodingPreservesCandidateBytes) {
    const auto before = packet;
    args.tensors_[0].address_space = AddressSpace::HOST;
    EXPECT_FALSE(encode_kernel_invocation(args, identity, binding, packet.data(), packet.size()));
    EXPECT_EQ(packet, before);
}

TEST_F(KernelInvocation, EmptyTensorNeedsNoBackingAndCountlessInvocationIsValid) {
    args.tensors_[0].shapes[0] = 0;
    args.tensors_[0].strides[0] = 0;
    args.tensors_[0].buffer = {};
    ASSERT_TRUE(encode_kernel_invocation(args, identity, binding, packet.data(), packet.size()));
    ASSERT_TRUE(consume(identity, binding));
    EXPECT_EQ(output.tensor(0).shapes[0], 0U);
    args.clear();
    identity.tensor_count = identity.scalar_count = 0;
    packet.resize(kernel_invocation_bytes(0, 0));
    ASSERT_TRUE(encode_kernel_invocation(args, identity, binding, packet.data(), packet.size()));
    ASSERT_TRUE(consume(identity, binding));
    EXPECT_EQ(output.tensor_count(), 0);
    EXPECT_EQ(output.scalar_count(), 0);
}

TEST_F(KernelInvocation, AddressAndGeometryOverflowAreRejected) {
    auto tensor = args.tensor(0).to_boundary();
    ChipTensor normalized{};
    tensor.buffer.addr = std::numeric_limits<uint64_t>::max() - 4;
    EXPECT_FALSE(normalize_kernel_tensor(tensor, &normalized));
    tensor.buffer.addr = 0x10000;
    tensor.start_offset = std::numeric_limits<uint64_t>::max();
    EXPECT_FALSE(normalize_kernel_tensor(tensor, &normalized));
    tensor.start_offset = 0;
    tensor.ndims = MAX_TENSOR_DIMS + 1;
    EXPECT_FALSE(normalize_kernel_tensor(tensor, &normalized));
}

}  // namespace
