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

#include <atomic>
#include <cstring>
#include <limits>
#include <mutex>
#include <thread>
#include <vector>

#include "tensormap_and_ringbuffer/kernel_invocation_args.h"
#include "worker/tmr_kernel_invocation.h"

namespace {
using namespace simpler::tmr;

const PreparedInvocationView kCallable{3, 1, 1, 4};
const TmrExecutionBindingView kBinding{0x10000, 8};

ChipStorageTaskArgs make_args(uint64_t address = 0x20000, uint64_t scalar = 19) {
    ChipStorageTaskArgs args{};
    const uint32_t shape[] = {2, 4};
    args.add_tensor(
        make_tensor_external(reinterpret_cast<void *>(address), shape, 2, DataType::FLOAT32, AddressSpace::DEVICE)
    );
    args.add_scalar(scalar);
    return args;
}

TEST(TmrKernelInvocation, RoundtripAndBorrowedConversion) {
    auto args = make_args();
    TmrEncodingCache cache;
    TmrEncodingCandidate candidate;
    ASSERT_EQ(encode_tmr_invocation(args, kCallable, kBinding, cache, &candidate), InvocationStatus::Ok);
    EXPECT_EQ(
        candidate.packet().size,
        sizeof(SimplerKernelInvocationHeader) + sizeof(TmrBindingRef) + sizeof(ChipTensor) + sizeof(uint64_t)
    );
    EXPECT_FALSE(candidate.structural_hit());
    TmrInvocationView view;
    ASSERT_EQ(decode_tmr_invocation(candidate.packet(), kCallable, kBinding, &view), InvocationStatus::Ok);
    EntryArgsStorage storage{};
    ASSERT_EQ(materialize_tmr_entry_args(view, &storage), InvocationStatus::Ok);
    EXPECT_EQ(storage.tensor_count(), 1);
    EXPECT_EQ(storage.scalar(0), 19u);
    EXPECT_EQ(storage.tensor(0).buffer.addr, 0x20000u);
    EXPECT_EQ(storage.tensor(0).numel(), 8u);
    EXPECT_EQ(storage.tensor(0).version, 0);
    EXPECT_FALSE(storage.tensor(0).manual_dep);
    ChipTensor tensor{};
    EXPECT_FALSE(view.tensor(-1, &tensor));
    EXPECT_FALSE(view.tensor(1, &tensor));
    EXPECT_FALSE(view.tensor(0, nullptr));
}

TEST(TmrKernelInvocation, CacheHitStillUsesCurrentAddressesAndScalars) {
    TmrEncodingCache cache;
    auto args = make_args();
    TmrEncodingCandidate first;
    ASSERT_EQ(encode_tmr_invocation(args, kCallable, kBinding, cache, &first), InvocationStatus::Ok);
    cache.commit(std::move(first));
    TmrEncodingCandidate equal;
    ASSERT_EQ(encode_tmr_invocation(args, kCallable, kBinding, cache, &equal), InvocationStatus::Ok);
    EXPECT_TRUE(equal.structural_hit());
    EXPECT_TRUE(first.same_invocation(equal));
    auto changed = make_args(0x30000, 77);
    TmrEncodingCandidate next;
    ASSERT_EQ(encode_tmr_invocation(changed, kCallable, kBinding, cache, &next), InvocationStatus::Ok);
    EXPECT_TRUE(next.structural_hit());
    EXPECT_FALSE(first.same_invocation(next));
    TmrInvocationView view;
    ASSERT_EQ(decode_tmr_invocation(next.packet(), kCallable, kBinding, &view), InvocationStatus::Ok);
    ChipTensor tensor{};
    uint64_t scalar{};
    ASSERT_TRUE(view.tensor(0, &tensor));
    ASSERT_TRUE(view.scalar(0, &scalar));
    EXPECT_EQ(tensor.buffer.addr, 0x30000u);
    EXPECT_EQ(scalar, 77u);
    ASSERT_EQ(decode_tmr_invocation(first.packet(), kCallable, kBinding, &view), InvocationStatus::Ok);
    ASSERT_TRUE(view.scalar(0, &scalar));
    EXPECT_EQ(scalar, 19u);
    EXPECT_EQ(cache.bytes(), first.packet().size);
}

TEST(TmrKernelInvocation, ConsumptionRejectsBeforeChangingCallerStorage) {
    auto args = make_args();
    TmrEncodingCache cache;
    TmrEncodingCandidate packet;
    ASSERT_EQ(encode_tmr_invocation(args, kCallable, kBinding, cache, &packet), InvocationStatus::Ok);
    EntryArgsStorage storage{};
    storage.scalar_count_ = 1;
    storage.scalars_[0] = 1234;
    auto stale = kCallable;
    ++stale.slot_generation;
    EXPECT_EQ(consume_tmr_invocation(packet.packet(), stale, kBinding, &storage), InvocationStatus::StaleCallable);
    EXPECT_EQ(storage.scalar_count(), 1);
    EXPECT_EQ(storage.scalar(0), 1234u);
    auto wrong_binding = kBinding;
    ++wrong_binding.context_generation;
    EXPECT_EQ(
        consume_tmr_invocation(packet.packet(), kCallable, wrong_binding, &storage), InvocationStatus::InvalidBinding
    );
    EXPECT_EQ(storage.scalar(0), 1234u);
    EXPECT_EQ(
        consume_tmr_invocation({packet.packet().data, packet.packet().size - 1}, kCallable, kBinding, &storage),
        InvocationStatus::InvalidSize
    );
    EXPECT_EQ(storage.scalar(0), 1234u);
    ASSERT_EQ(consume_tmr_invocation(packet.packet(), kCallable, kBinding, &storage), InvocationStatus::Ok);
    EXPECT_EQ(storage.tensor_count(), 1);
    EXPECT_EQ(storage.scalar(0), 19u);
}

TEST(TmrKernelInvocation, RetainedCacheIsBoundedAcrossRepeatedInvocations) {
    TmrEncodingCache cache;
    size_t expected_bytes = 0;
    ASSERT_EQ(tmr_invocation_size(kCallable, &expected_bytes), InvocationStatus::Ok);
    for (uint64_t i = 0; i < 10000; ++i) {
        auto args = make_args(0x20000 + i * 64, i);
        TmrEncodingCandidate candidate;
        ASSERT_EQ(encode_tmr_invocation(args, kCallable, kBinding, cache, &candidate), InvocationStatus::Ok);
        EXPECT_EQ(candidate.structural_hit(), i != 0);
        cache.commit(std::move(candidate));
        EXPECT_EQ(cache.bytes(), expected_bytes);
    }
}

TEST(TmrKernelInvocation, FailedCandidateAndUnsubmittedCandidatePreserveCache) {
    TmrEncodingCache cache;
    auto args = make_args();
    TmrEncodingCandidate candidate;
    ASSERT_EQ(encode_tmr_invocation(args, kCallable, kBinding, cache, &candidate), InvocationStatus::Ok);
    cache.commit(std::move(candidate));
    const auto saved = std::vector<uint8_t>(candidate.packet().data, candidate.packet().data + candidate.packet().size);
    auto bad = args;
    bad.tensor(0).buffer.addr = 0;
    EXPECT_EQ(encode_tmr_invocation(bad, kCallable, kBinding, cache, &candidate), InvocationStatus::InvalidTensor);
    EXPECT_EQ(std::memcmp(candidate.packet().data, saved.data(), saved.size()), 0);
    {
        TmrEncodingCandidate unsubmitted;
        auto other = args;
        other.tensor(0).shapes[0] = 1;
        ASSERT_EQ(encode_tmr_invocation(other, kCallable, kBinding, cache, &unsubmitted), InvocationStatus::Ok);
        EXPECT_FALSE(unsubmitted.structural_hit());
    }
    ASSERT_EQ(encode_tmr_invocation(args, kCallable, kBinding, cache, &candidate), InvocationStatus::Ok);
    EXPECT_TRUE(candidate.structural_hit());
}

TEST(TmrKernelInvocation, CanonicalFieldsIgnoreUnusedDimensionsAndPadding) {
    TmrEncodingCache cache;
    auto args = make_args();
    TmrEncodingCandidate first, second;
    ASSERT_EQ(encode_tmr_invocation(args, kCallable, kBinding, cache, &first), InvocationStatus::Ok);
    cache.commit(std::move(first));
    ChipTensor noisy;
    std::memset(&noisy, 0xa5, sizeof(noisy));
    noisy.buffer = args.tensor(0).buffer;
    noisy.start_offset = 0;
    noisy.ndims = 2;
    noisy.dtype = DataType::FLOAT32;
    noisy.address_space = AddressSpace::DEVICE;
    for (int i = 0; i < 2; ++i) {
        noisy.shapes[i] = args.tensor(0).shapes[i];
        noisy.strides[i] = args.tensor(0).strides[i];
    }
    args.tensor(0) = noisy;
    ASSERT_EQ(encode_tmr_invocation(args, kCallable, kBinding, cache, &second), InvocationStatus::Ok);
    EXPECT_TRUE(second.structural_hit());
    EXPECT_TRUE(first.same_invocation(second));
}

TEST(TmrKernelInvocation, GeometryBackingAndGenerationsInvalidateTemplate) {
    TmrEncodingCache cache;
    auto args = make_args();
    args.tensor(0).buffer.size = 128;
    TmrEncodingCandidate first;
    ASSERT_EQ(encode_tmr_invocation(args, kCallable, kBinding, cache, &first), InvocationStatus::Ok);
    cache.commit(std::move(first));
    for (int change = 0; change < 7; ++change) {
        auto current = args;
        auto callable = kCallable;
        auto binding = kBinding;
        if (change == 0) current.tensor(0).shapes[0] = 1;
        if (change == 1) current.tensor(0).strides[0] = 8;
        if (change == 2) current.tensor(0).start_offset = 1;
        if (change == 3) current.tensor(0).buffer.size = 132;
        if (change == 4) ++callable.slot_generation;
        if (change == 5) ++binding.context_generation;
        if (change == 6) current.tensor(0).dtype = DataType::FLOAT16;
        TmrEncodingCandidate candidate;
        ASSERT_EQ(encode_tmr_invocation(current, callable, binding, cache, &candidate), InvocationStatus::Ok);
        EXPECT_FALSE(candidate.structural_hit());
    }
}

TEST(TmrKernelInvocation, DecoderRejectsCorruptionTruncationAndPreservesView) {
    TmrEncodingCache cache;
    TmrEncodingCandidate candidate;
    auto args = make_args();
    ASSERT_EQ(encode_tmr_invocation(args, kCallable, kBinding, cache, &candidate), InvocationStatus::Ok);
    TmrInvocationView good;
    ASSERT_EQ(decode_tmr_invocation(candidate.packet(), kCallable, kBinding, &good), InvocationStatus::Ok);
    std::vector<uint8_t> unaligned(candidate.packet().size + 1);
    std::memcpy(unaligned.data() + 1, candidate.packet().data, candidate.packet().size);
    TmrInvocationView view;
    ASSERT_EQ(
        decode_tmr_invocation({unaligned.data() + 1, candidate.packet().size}, kCallable, kBinding, &view),
        InvocationStatus::Ok
    );
    for (size_t length = 0; length < candidate.packet().size; ++length) {
        EXPECT_NE(
            decode_tmr_invocation({candidate.packet().data, length}, kCallable, kBinding, &good), InvocationStatus::Ok
        );
        EXPECT_TRUE(good.valid());
    }
    auto stale = kCallable;
    ++stale.slot_generation;
    EXPECT_EQ(validate_tmr_submission(candidate, stale, kBinding), InvocationStatus::StaleCallable);
    auto other_binding = kBinding;
    ++other_binding.context_generation;
    EXPECT_EQ(validate_tmr_submission(candidate, kCallable, other_binding), InvocationStatus::InvalidBinding);
    other_binding = kBinding;
    other_binding.device_binding_addr += 64;
    EXPECT_EQ(validate_tmr_submission(candidate, kCallable, other_binding), InvocationStatus::InvalidBinding);
    other_binding.device_binding_addr = 0;
    EXPECT_EQ(validate_tmr_submission(candidate, kCallable, other_binding), InvocationStatus::InvalidBinding);
    const size_t tensor_offset = sizeof(SimplerKernelInvocationHeader) + sizeof(TmrBindingRef);
    const uint64_t bad_packet_scalar = 77;
    std::memcpy(
        unaligned.data() + 1 + tensor_offset + sizeof(ChipTensor), &bad_packet_scalar, sizeof(bad_packet_scalar)
    );
    uint8_t bad_dtype = 255;
    std::memcpy(unaligned.data() + 1 + tensor_offset + offsetof(ChipTensor, dtype), &bad_dtype, sizeof(bad_dtype));
    EXPECT_EQ(
        decode_tmr_invocation({unaligned.data() + 1, candidate.packet().size}, kCallable, kBinding, &good),
        InvocationStatus::InvalidTensor
    );
    uint64_t scalar{};
    ASSERT_TRUE(good.scalar(0, &scalar));
    EXPECT_EQ(scalar, 19u);
}

TEST(TmrKernelInvocation, SelfConsistentEnvelopeStillRequiresExactTmrSize) {
    TmrEncodingCache cache;
    TmrEncodingCandidate candidate;
    ASSERT_EQ(encode_tmr_invocation(make_args(), kCallable, kBinding, cache, &candidate), InvocationStatus::Ok);
    TmrInvocationView view;
    ASSERT_EQ(decode_tmr_invocation(candidate.packet(), kCallable, kBinding, &view), InvocationStatus::Ok);
    for (const size_t length : {candidate.packet().size - 1, candidate.packet().size + 1}) {
        std::vector<uint8_t> packet(candidate.packet().data, candidate.packet().data + candidate.packet().size);
        packet.resize(length);
        SimplerKernelInvocationHeader header{};
        std::memcpy(&header, packet.data(), sizeof(header));
        header.payload_bytes = length - sizeof(header);
        std::memcpy(packet.data(), &header, sizeof(header));
        SimplerKernelInvocationHeader checked{};
        ASSERT_EQ(
            simpler::kernel::validate_invocation_header({packet.data(), packet.size()}, kCallable, &checked),
            InvocationStatus::Ok
        );
        EXPECT_EQ(
            decode_tmr_invocation({packet.data(), packet.size()}, kCallable, kBinding, &view),
            InvocationStatus::InvalidSize
        );
        uint64_t scalar{};
        ASSERT_TRUE(view.scalar(0, &scalar));
        EXPECT_EQ(scalar, 19u);
    }
}

TEST(TmrKernelInvocation, InvalidTensorArithmeticAndHostBackingAreRejected) {
    auto args = make_args();
    for (int change = 0; change < 8; ++change) {
        auto t = args.tensor(0);
        if (change == 0) t.address_space = AddressSpace::HOST;
        if (change == 1) t.ndims = 0;
        if (change == 2) t.ndims = MAX_TENSOR_DIMS + 1;
        if (change == 3) t.strides[0] = 0;
        if (change == 4) t.buffer.size = 31;
        if (change == 5) t.start_offset = std::numeric_limits<uint64_t>::max();
        if (change == 6) t.buffer.addr = std::numeric_limits<uint64_t>::max();
        if (change == 7) {
            t.ndims = 5;
            for (int i = 0; i < 5; ++i)
                t.shapes[i] = t.strides[i] = std::numeric_limits<uint32_t>::max();
        }
        ChipTensor out = args.tensor(0);
        EXPECT_EQ(normalize_invocation_tensor(t, &out), InvocationStatus::InvalidTensor);
        EXPECT_EQ(out.buffer.addr, args.tensor(0).buffer.addr);
    }
}

TEST(TmrKernelInvocation, EmptyTensorMatchesBoundaryConversionWithoutBacking) {
    const uint32_t shapes[][3] = {{0, 1, 1}, {0, 8, 1}, {8, 0, 1}, {2, 0, 3}};
    const uint32_t ranks[] = {1, 2, 2, 3};
    TmrEncodingCache cache;
    for (int i = 0; i < 4; ++i) {
        auto args = make_args();
        args.tensor(0) = make_tensor_external(nullptr, shapes[i], ranks[i], DataType::FLOAT32, AddressSpace::DEVICE);
        TmrEncodingCandidate candidate;
        ASSERT_EQ(encode_tmr_invocation(args, kCallable, kBinding, cache, &candidate), InvocationStatus::Ok);
        TmrInvocationView view;
        ASSERT_EQ(decode_tmr_invocation(candidate.packet(), kCallable, kBinding, &view), InvocationStatus::Ok);
        EntryArgsStorage storage{};
        ASSERT_EQ(materialize_tmr_entry_args(view, &storage), InvocationStatus::Ok);
        EXPECT_EQ(storage.tensor(0).numel(), 0u);
        EXPECT_EQ(storage.tensor(0).extent_elem(), Tensor::from_boundary(args.tensor(0)).extent_elem());
    }
    auto args = make_args();
    auto parent = Tensor::from_boundary(args.tensor(0));
    const uint32_t empty_shape[] = {0, 4}, beyond[] = {100, 0};
    args.tensor(0) = parent.view(empty_shape, beyond).to_boundary();
    TmrEncodingCandidate candidate;
    EXPECT_EQ(encode_tmr_invocation(args, kCallable, kBinding, cache, &candidate), InvocationStatus::Ok);
}

TEST(TmrKernelInvocation, MinimumMaximumAndScalarOnlyPackets) {
    TmrEncodingCache cache;
    const PreparedInvocationView cases[] = {{0, 0, 0, 1}, {0, 256, 0, 1}, {0, 0, 128, 1}, {0, 128, 128, 1}};
    for (const auto &callable : cases) {
        ChipStorageTaskArgs args{};
        auto sample = make_args();
        for (int i = 0; i < callable.tensor_count; ++i)
            args.add_tensor(sample.tensor(0));
        for (int i = 0; i < callable.scalar_count; ++i)
            args.add_scalar(static_cast<uint64_t>(i));
        TmrEncodingCandidate candidate;
        ASSERT_EQ(encode_tmr_invocation(args, callable, kBinding, cache, &candidate), InvocationStatus::Ok);
        EXPECT_EQ(validate_tmr_submission(candidate, callable, kBinding), InvocationStatus::Ok);
    }
}

TEST(TmrKernelInvocation, IndependentCallsAndOwnerSerializedSharedCache) {
    TmrEncodingCache shared;
    std::mutex owner;
    std::atomic<int> failures{0};
    std::vector<std::thread> threads;
    for (int worker = 0; worker < 4; ++worker) {
        threads.emplace_back([&, worker] {
            TmrEncodingCache local;
            for (int iteration = 0; iteration < 100; ++iteration) {
                auto args = make_args(0x20000 + worker * 0x1000, iteration);
                TmrEncodingCandidate independent;
                if (encode_tmr_invocation(args, kCallable, kBinding, local, &independent) != InvocationStatus::Ok)
                    ++failures;
                local.commit(std::move(independent));
                std::lock_guard<std::mutex> guard(owner);
                TmrEncodingCandidate candidate;
                if (encode_tmr_invocation(args, kCallable, kBinding, shared, &candidate) != InvocationStatus::Ok ||
                    !candidate.same_invocation(independent))
                    ++failures;
                shared.commit(std::move(candidate));
            }
        });
    }
    for (auto &thread : threads)
        thread.join();
    EXPECT_EQ(failures, 0);
}
}  // namespace
