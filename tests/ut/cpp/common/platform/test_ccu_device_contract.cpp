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
#include <cstring>
#include "platform_comm/ccu/ccu_device.h"

namespace {
struct TensorView {
    struct {
        uint64_t addr, size;
    } buffer;
    uint64_t start_offset;
    uint32_t ndims;
    uint32_t shapes[1];
    uint32_t strides[1];
    uint32_t dtype;
};
struct alignas(128) Image {
    CcuDeviceContext ctx{};
    CcuProgramBinding program{};
    std::array<CcuParameterLayout, 3> signature{};
    std::array<CcuMemoryRange, 2> ranges{};
    CcuRequestRecord request{};
    std::array<uint64_t, 7> args{};
    uint64_t gate{};
    CcuAdmissionRecord admission{};
    Image() {
        ctx.abi_version = CCU_DEVICE_ABI_VERSION;
        ctx.struct_size = sizeof(ctx);
        ctx.image_size = sizeof(*this);
        ctx.allocation_id = 11;
        ctx.session_epoch = 29;
        ctx.next_sequence = 1;
        ctx.bindings_offset = offsetof(Image, program);
        ctx.binding_count = 1;
        ctx.ranges_offset = offsetof(Image, ranges);
        ctx.range_count = 2;
        ctx.record_offset = offsetof(Image, request);
        ctx.arguments_offset = offsetof(Image, args);
        ctx.arguments_size = sizeof(args);
        ctx.state = CcuSessionState::READY;
        ctx.rank_count = 2;
        program.key.bytes[31] = 7;
        program.session_epoch = ctx.session_epoch;
        program.signature_offset = offsetof(Image, signature);
        program.gate_addr = reinterpret_cast<uint64_t>(&gate);
        program.program_index = 8;
        program.parameter_count = 3;
        program.argument_bytes = sizeof(args);
        signature[0] = {CcuArgumentKind::TENSOR, CcuArgumentType::FLOAT32, 0, 0, 0};
        signature[1] = {CcuArgumentKind::TENSOR, CcuArgumentType::FLOAT32, 3, 0, 0};
        signature[2] = {CcuArgumentKind::SCALAR, CcuArgumentType::UINT64, 6, 0, 0};
        ranges[0] = {1, 0x10000, 16384, 101};
        ranges[1] = {2, 0x30000, 16384, 202};
    }
    CcuProgramRef Resolve() {
        CcuProgramRef out{};
        EXPECT_EQ(ccu_resolve_program(&ctx, program.key, out), CcuStatus::OK);
        return out;
    }
    TensorView Input(uint64_t offset = 0) { return {{0x10000, 16384}, offset, 1, {64}, {1}, 0}; }
    TensorView Output(uint64_t offset = 0) { return {{0x30000, 16384}, offset, 1, {64}, {1}, 0}; }
    // The test explicitly injects observations; no CCU execution is claimed.
    void Complete(CcuRequestRef ref) {
        request.completed_epoch = ctx.session_epoch;
        request.completed_program = program.program_index;
        request.completed_state = CcuRequestState::SUCCEEDED;
        request.status = CcuStatus::OK;
        request.completed_sequence = ref.sequence;
        gate = 0;
    }
};
}  // namespace

TEST(CcuDeviceContract, ValidatesAbiAndUsesAll256KeyBits) {
    Image image;
    CcuDeviceContext *ctx{};
    ASSERT_EQ(ccu_get_device_context(reinterpret_cast<uint64_t>(&image), sizeof(image), ctx), CcuStatus::OK);
    CcuProgramKey other = image.program.key;
    other.bytes[0] = 1;
    CcuProgramRef program{};
    EXPECT_EQ(ccu_resolve_program(ctx, other, program), CcuStatus::INVALID_ARGUMENT);
    image.ctx.abi_version = CCU_DEVICE_ABI_VERSION + 1;
    EXPECT_EQ(ccu_get_device_context(reinterpret_cast<uint64_t>(&image), sizeof(image), ctx), CcuStatus::ABI_MISMATCH);
}
TEST(CcuDeviceContract, RejectsHeaderAliasesMisalignmentAndOverlappingImageRegions) {
    const std::array<uint64_t CcuDeviceContext::*, 5> offsets{
        &CcuDeviceContext::bindings_offset, &CcuDeviceContext::ranges_offset, &CcuDeviceContext::record_offset,
        &CcuDeviceContext::arguments_offset, &CcuDeviceContext::admission_offset
    };
    const auto reject = [](Image &image) {
        std::array<unsigned char, sizeof(Image)> before{};
        std::memcpy(before.data(), &image, sizeof(image));
        CcuDeviceContext *resolved = &image.ctx;
        EXPECT_EQ(
            ccu_get_device_context(reinterpret_cast<uint64_t>(&image), sizeof(image), resolved), CcuStatus::ABI_MISMATCH
        );
        EXPECT_EQ(resolved, nullptr);
        EXPECT_EQ(std::memcmp(before.data(), &image, sizeof(image)), 0);
    };
    for (size_t i = 0; i < offsets.size(); ++i) {
        SCOPED_TRACE(i);
        Image misaligned;
        misaligned.ctx.admission_offset = offsetof(Image, admission);
        ++(misaligned.ctx.*offsets[i]);
        reject(misaligned);
        if (offsets[i] != &CcuDeviceContext::admission_offset) {
            Image header_alias;
            header_alias.ctx.*offsets[i] = 0;
            reject(header_alias);
        }
        for (size_t j = 0; j < offsets.size(); ++j) {
            if (i == j) continue;
            SCOPED_TRACE(j);
            Image overlap;
            overlap.ctx.admission_offset = offsetof(Image, admission);
            overlap.ctx.*offsets[i] = overlap.ctx.*offsets[j];
            reject(overlap);
        }
    }
    Image adjacent;
    adjacent.ctx.admission_offset = offsetof(Image, admission);
    CcuDeviceContext *resolved{};
    EXPECT_EQ(ccu_get_device_context(reinterpret_cast<uint64_t>(&adjacent), sizeof(adjacent), resolved), CcuStatus::OK);
    EXPECT_EQ(resolved, &adjacent.ctx);
}

TEST(CcuDeviceContract, RejectsSignatureTablesOutsideTheirReadonlyRegion) {
    for (uint64_t offset :
         {uint64_t{0}, uint64_t{offsetof(Image, signature) + 1}, uint64_t{offsetof(Image, program)},
          uint64_t{offsetof(Image, ranges)}, uint64_t{offsetof(Image, request)}, uint64_t{offsetof(Image, args)},
          uint64_t{offsetof(Image, admission)}, uint64_t{sizeof(Image) - sizeof(CcuParameterLayout)}}) {
        SCOPED_TRACE(offset);
        Image image;
        image.ctx.admission_offset = offsetof(Image, admission);
        image.program.signature_offset = offset;
        CcuDeviceContext *resolved{};
        EXPECT_EQ(
            ccu_get_device_context(reinterpret_cast<uint64_t>(&image), sizeof(image), resolved), CcuStatus::ABI_MISMATCH
        );
        EXPECT_EQ(resolved, nullptr);
    }
}

TEST(CcuDeviceContract, VariadicReserveCannotReuseAPositionalSnapshot) {
    struct alignas(128) PositionalImage {
        CcuDeviceContext ctx{};
        CcuProgramBinding binding{};
        CcuRequestRecord record{};
        CcuPositionalArguments arguments{};
        uint64_t gate{};
    } image;
    image.ctx.abi_version = CCU_DEVICE_ABI_VERSION;
    image.ctx.struct_size = sizeof(CcuDeviceContext);
    image.ctx.image_size = sizeof(image);
    image.ctx.session_epoch = 1;
    image.ctx.next_sequence = 1;
    image.ctx.bindings_offset = offsetof(PositionalImage, binding);
    image.ctx.binding_count = 1;
    image.ctx.record_offset = offsetof(PositionalImage, record);
    image.ctx.arguments_offset = offsetof(PositionalImage, arguments);
    image.ctx.arguments_size = sizeof(CcuPositionalArguments);
    image.binding.session_epoch = 1;
    image.binding.reserved = 1;
    image.binding.argument_bytes = sizeof(CcuPositionalArguments);
    image.binding.gate_addr = reinterpret_cast<uint64_t>(&image.gate);
    CcuProgramRef program{};
    ASSERT_EQ(ccu_resolve_program(&image.ctx, uint32_t{0}, program), CcuStatus::OK);
    CcuPositionalArguments arguments{};
    arguments.scalar_count = 1;
    arguments.scalars[0] = {37, CcuScalarKind::UNSIGNED, 64};
    CcuRequestRef first{};
    ASSERT_EQ(ccu_reserve_request(&image.ctx, program, first, arguments), CcuStatus::OK);
    ASSERT_EQ(ccu_submit_request(first), CcuStatus::OK);
    EXPECT_EQ(image.gate, (uint64_t{1} << 63) | 1);
    image.record.completed_epoch = 1;
    image.record.completed_sequence = first.sequence;
    image.record.completed_state = CcuRequestState::SUCCEEDED;
    ccu_retire_request(first);
    ASSERT_EQ(static_cast<CcuSlotState>(image.record.slot_state), CcuSlotState::FREE);
    image.gate = 0;
    const uint64_t sequence = image.ctx.next_sequence;
    const auto snapshot = image.arguments;
    CcuRequestRef rejected{1, 1};
    EXPECT_EQ(ccu_reserve_request(&image.ctx, program, rejected), CcuStatus::INVALID_ARGUMENT);
    EXPECT_EQ(rejected.record_addr, 0U);
    EXPECT_EQ(image.ctx.next_sequence, sequence);
    EXPECT_EQ(static_cast<CcuSlotState>(image.record.slot_state), CcuSlotState::FREE);
    EXPECT_EQ(image.gate, 0U);
    EXPECT_EQ(std::memcmp(&snapshot, &image.arguments, sizeof(snapshot)), 0);
}
TEST(CcuDeviceContract, ReserveSubmitAndRetireRepeatWithDynamicAddressesWithoutHostPrepare) {
    Image image;
    const auto program = image.Resolve();
    for (uint64_t round = 0; round < 1024; ++round) {
        CcuRequestRef ref{};
        ASSERT_EQ(
            ccu_reserve_request(
                &image.ctx, program, ref, image.Input(round % 2 * 128), image.Output(round % 2 * 128),
                uint64_t{16 * (round % 4 + 1)}
            ),
            CcuStatus::OK
        );
        EXPECT_EQ(image.args[0], 0x10000 + round % 2 * 128 * 4);
        EXPECT_EQ(image.args[1], 101U);
        EXPECT_EQ(image.args[4], 202U);
        EXPECT_EQ(ref.sequence, round + 1);
        EXPECT_EQ(image.gate, 0U);
        ASSERT_EQ(ccu_submit_request(ref), CcuStatus::OK);
        EXPECT_EQ(image.gate, (uint64_t{1} << 63) | 1);
        EXPECT_EQ(ccu_submit_request(ref), CcuStatus::INVALID_ARGUMENT);
        EXPECT_EQ(ccu_poll_request(ref).state, CcuRequestState::PENDING);
        image.Complete(ref);
        EXPECT_EQ(ccu_poll_request(ref).state, CcuRequestState::SUCCEEDED);
        ccu_retire_request(ref);
        EXPECT_EQ(static_cast<CcuSlotState>(image.request.slot_state), CcuSlotState::FREE);
    }
}
TEST(CcuDeviceContract, BusyCancelAndStaleRetireCannotFreeAnotherRequest) {
    Image image;
    auto program = image.Resolve();
    CcuRequestRef first{}, second{};
    ASSERT_EQ(
        ccu_reserve_request(&image.ctx, program, first, image.Input(), image.Output(), uint64_t{64}), CcuStatus::OK
    );
    EXPECT_EQ(
        ccu_reserve_request(&image.ctx, program, second, image.Input(), image.Output(), uint64_t{64}), CcuStatus::BUSY
    );
    ccu_cancel_reserved_request(first);
    ASSERT_EQ(
        ccu_reserve_request(&image.ctx, program, second, image.Input(), image.Output(), uint64_t{32}), CcuStatus::OK
    );
    ccu_retire_request(first);
    ccu_cancel_reserved_request(first);
    EXPECT_EQ(static_cast<CcuSlotState>(image.request.slot_state), CcuSlotState::RESERVED);
    EXPECT_NE(first.sequence, second.sequence);
    ccu_cancel_reserved_request(second);
}
TEST(CcuDeviceContract, RejectsUnpreparedMemoryWrongScalarTypeAndUndeclaredAliases) {
    Image image;
    const auto program = image.Resolve();
    CcuRequestRef ref{};
    auto bad = image.Input();
    bad.buffer.addr = 0x90000;
    EXPECT_EQ(
        ccu_reserve_request(&image.ctx, program, ref, bad, image.Output(), uint64_t{64}), CcuStatus::INVALID_ARGUMENT
    );
    EXPECT_EQ(ref.record_addr, 0U);
    EXPECT_EQ(
        ccu_reserve_request(&image.ctx, program, ref, image.Input(), image.Output(), int64_t{64}),
        CcuStatus::INVALID_ARGUMENT
    );
    EXPECT_EQ(
        ccu_reserve_request(&image.ctx, program, ref, image.Input(), image.Input(), uint64_t{64}),
        CcuStatus::INVALID_ARGUMENT
    );
    image.signature[1].allowed_aliases = 1;
    EXPECT_EQ(ccu_reserve_request(&image.ctx, program, ref, image.Input(), image.Input(), uint64_t{64}), CcuStatus::OK);
    ccu_cancel_reserved_request(ref);
    EXPECT_EQ(
        ccu_reserve_request(&image.ctx, program, ref, image.Input(), image.Input(1), uint64_t{64}),
        CcuStatus::INVALID_ARGUMENT
    );
}
TEST(CcuDeviceContract, WrongCompletionEpochBansSessionAndSequenceNeverWraps) {
    Image image;
    const auto program = image.Resolve();
    CcuRequestRef ref{};
    ASSERT_EQ(
        ccu_reserve_request(&image.ctx, program, ref, image.Input(), image.Output(), uint64_t{64}), CcuStatus::OK
    );
    ASSERT_EQ(ccu_submit_request(ref), CcuStatus::OK);
    image.Complete(ref);
    ++image.request.completed_epoch;
    EXPECT_EQ(ccu_poll_request(ref).state, CcuRequestState::UNCONFIRMED);
    ccu_retire_request(ref);
    EXPECT_EQ(static_cast<CcuSlotState>(image.request.slot_state), CcuSlotState::SUBMITTED);
    Image other;
    other.ctx.next_sequence = UINT64_MAX;
    EXPECT_EQ(
        ccu_reserve_request(&other.ctx, other.Resolve(), ref, other.Input(), other.Output(), uint64_t{64}),
        CcuStatus::RECLAMATION_UNCONFIRMED
    );
}

TEST(CcuDeviceContract, FirstSubmitFailureRemainsQueryableAndCanRetire) {
    Image image;
    const auto program = image.Resolve();
    CcuRequestRef ref{};
    ASSERT_EQ(
        ccu_reserve_request(&image.ctx, program, ref, image.Input(), image.Output(), uint64_t{64}), CcuStatus::OK
    );
    image.program.gate_addr = 0;
    EXPECT_EQ(ccu_submit_request(ref), CcuStatus::STALE_HANDLE);
    EXPECT_EQ(ccu_poll_request(ref).state, CcuRequestState::FAILED);
    EXPECT_EQ(ccu_submit_request(ref), CcuStatus::INVALID_ARGUMENT);
    EXPECT_EQ(image.request.completed_sequence, 0U);
    ccu_retire_request(ref);
    EXPECT_EQ(static_cast<CcuSlotState>(image.request.slot_state), CcuSlotState::FREE);
}
TEST(CcuDeviceContract, RequestTimeoutRetainsResourcesAndDoesNotWriteCcuCompletion) {
    Image image;
    const auto program = image.Resolve();
    CcuRequestRef ref{};
    ASSERT_EQ(
        ccu_reserve_request(&image.ctx, program, ref, image.Input(), image.Output(), uint64_t{64}), CcuStatus::OK
    );
    image.ctx.timeout_ns = 1;
    ASSERT_EQ(ccu_submit_request(ref), CcuStatus::OK);
    image.request.submitted_at_ns = ccu_detail::NowNs() - 2;
    EXPECT_EQ(ccu_poll_request(ref).state, CcuRequestState::UNCONFIRMED);
    EXPECT_EQ(ccu_poll_request(ref).status, CcuStatus::TIMEOUT);
    EXPECT_EQ(image.request.completed_sequence, 0U);
    ccu_retire_request(ref);
    EXPECT_EQ(static_cast<CcuSlotState>(image.request.slot_state), CcuSlotState::SUBMITTED);
    CcuRequestRef another{};
    EXPECT_EQ(
        ccu_reserve_request(&image.ctx, program, another, image.Input(), image.Output(), uint64_t{64}),
        CcuStatus::STALE_HANDLE
    );
}
