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

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

#include <gtest/gtest.h>

#include "aicpu/worker_chip_message_queue.h"

namespace {

struct RegionStorage {
    alignas(64) std::array<uint8_t, 512> payload{};
    alignas(64) std::array<int32_t, 128> counters{};
};

WorkerChipOrchRegionDesc make_desc(RegionStorage *storage, uint64_t payload_bytes = 512, uint64_t counter_bytes = 512) {
    return WorkerChipOrchRegionDesc{
        worker_chip_orch_comm_magic_version(),
        19,
        reinterpret_cast<uint64_t>(storage->payload.data()),
        payload_bytes,
        reinterpret_cast<uint64_t>(storage->counters.data()),
        counter_bytes,
    };
}

size_t counter_index(uint64_t offset) { return static_cast<size_t>(offset / sizeof(int32_t)); }

WorkerChipQueueArgs make_args(uint64_t depth, uint64_t input_arena_bytes, uint64_t output_arena_bytes) {
    WorkerChipQueueLayout layout{};
    EXPECT_TRUE(worker_chip_queue_make_layout(depth, input_arena_bytes, output_arena_bytes, layout));
    return WorkerChipQueueArgs{
        WORKER_CHIP_QUEUE_MAGIC_VERSION,
        depth,
        input_arena_bytes,
        output_arena_bytes,
        layout.payload_bytes,
        layout.counter_bytes,
    };
}

WorkerChipOrchRegionDesc make_desc(RegionStorage *storage, const WorkerChipQueueArgs &args) {
    return make_desc(storage, args.payload_bytes, args.counter_bytes);
}

void publish_input_desc(
    RegionStorage *storage, const WorkerChipQueueLayout &layout, uint64_t seq, WorkerChipQueueOpcode opcode,
    uint64_t payload_offset = 0, uint64_t payload_nbytes = 0
) {
    WorkerChipQueueDescSlot slot{};
    worker_chip_queue_encode_desc(&slot, seq, opcode, payload_offset, payload_nbytes);
    uint64_t desc_offset =
        layout.input_desc_offset + ((seq - 1) & (layout.depth - 1)) * sizeof(WorkerChipQueueDescSlot);
    std::memcpy(storage->payload.data() + desc_offset, &slot, sizeof(slot));
    storage->counters[counter_index(layout.input_desc_tail_offset)] = static_cast<int32_t>(seq);
}

TEST(WorkerChipMessageQueueTest, MagicVersionConstantMatchesCompatibilityWrapper) {
    EXPECT_EQ(
        WORKER_CHIP_QUEUE_MAGIC_VERSION,
        worker_chip_orch_comm_pack_magic_version(
            WORKER_CHIP_QUEUE_MAGIC, WORKER_CHIP_QUEUE_ABI_MAJOR, WORKER_CHIP_QUEUE_ABI_MINOR
        )
    );
    EXPECT_EQ(worker_chip_queue_magic_version(), WORKER_CHIP_QUEUE_MAGIC_VERSION);
    EXPECT_EQ(worker_chip_message_queue::magic_version(), WORKER_CHIP_QUEUE_MAGIC_VERSION);
}

TEST(WorkerChipMessageQueueTest, LayoutAssignsPayloadAndAbortCounterOffsets) {
    WorkerChipQueueLayout layout{};

    ASSERT_TRUE(worker_chip_queue_make_layout(4, 128, 192, layout));

    EXPECT_EQ(layout.input_desc_offset, 0u);
    EXPECT_EQ(layout.output_desc_offset, 4u * sizeof(WorkerChipQueueDescSlot));
    EXPECT_EQ(layout.input_arena_offset % 64u, 0u);
    EXPECT_EQ(layout.output_arena_offset % 64u, 0u);
    EXPECT_EQ(layout.input_desc_tail_offset, 0u);
    EXPECT_EQ(layout.input_desc_head_offset, 64u);
    EXPECT_EQ(layout.output_desc_tail_offset, 128u);
    EXPECT_EQ(layout.output_desc_head_offset, 192u);
    EXPECT_EQ(layout.worker_abort_flag_offset, 256u);
    EXPECT_EQ(layout.chip_abort_flag_offset, 320u);
    EXPECT_EQ(layout.counter_bytes, 384u);
    EXPECT_GE(layout.payload_bytes, layout.output_arena_offset + 192u);
}

TEST(WorkerChipMessageQueueTest, LayoutLockstepCasesMatchPythonMirrorExpectations) {
    struct LayoutCase {
        uint64_t depth;
        uint64_t input_arena_bytes;
        uint64_t output_arena_bytes;
        uint64_t output_desc_offset;
        uint64_t input_arena_offset;
        uint64_t output_arena_offset;
        uint64_t payload_bytes;
    };

    const std::array<LayoutCase, 3> cases{{
        {1, 64, 64, 32, 64, 128, 192},
        {4, 128, 192, 128, 256, 384, 576},
        {8, 192, 64, 256, 512, 704, 768},
    }};

    for (const auto &test_case : cases) {
        WorkerChipQueueLayout layout{};
        ASSERT_TRUE(worker_chip_queue_make_layout(
            test_case.depth, test_case.input_arena_bytes, test_case.output_arena_bytes, layout
        ));

        EXPECT_EQ(layout.input_desc_offset, 0u);
        EXPECT_EQ(layout.output_desc_offset, test_case.output_desc_offset);
        EXPECT_EQ(layout.output_desc_offset, test_case.depth * sizeof(WorkerChipQueueDescSlot));
        EXPECT_EQ(layout.input_arena_offset, test_case.input_arena_offset);
        EXPECT_EQ(layout.output_arena_offset, test_case.output_arena_offset);
        EXPECT_EQ(layout.payload_bytes, test_case.payload_bytes);
        EXPECT_EQ(layout.input_desc_tail_offset, 0u);
        EXPECT_EQ(layout.input_desc_head_offset, 64u);
        EXPECT_EQ(layout.output_desc_tail_offset, 128u);
        EXPECT_EQ(layout.output_desc_head_offset, 192u);
        EXPECT_EQ(layout.worker_abort_flag_offset, 256u);
        EXPECT_EQ(layout.chip_abort_flag_offset, 320u);
        EXPECT_EQ(layout.counter_bytes, 384u);
    }
}

TEST(WorkerChipMessageQueueTest, LayoutRejectsInvalidDepthArenaAndCounterBytes) {
    WorkerChipQueueLayout layout{};

    EXPECT_FALSE(worker_chip_queue_make_layout(3, 64, 64, layout));
    EXPECT_FALSE(worker_chip_queue_make_layout((1ull << 30) + 1, 64, 64, layout));
    EXPECT_FALSE(worker_chip_queue_make_layout(2, 0, 64, layout));
    EXPECT_FALSE(worker_chip_queue_make_layout(2, 65, 64, layout));

    RegionStorage storage{};
    WorkerChipQueueArgs args = make_args(2, 64, 64);
    EXPECT_FALSE(worker_chip_queue_validate_region(make_desc(&storage, 256, 320), args, &layout));
    EXPECT_FALSE(worker_chip_queue_validate_region(make_desc(&storage, 512, 384), args, &layout));
    EXPECT_TRUE(worker_chip_queue_validate_region(make_desc(&storage, args), args, &layout));
}

TEST(WorkerChipMessageQueueTest, LayoutOverflowFailsClosedWithoutModifyingOutput) {
    WorkerChipQueueLayout layout{
        7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61,
    };
    const WorkerChipQueueLayout original = layout;

    EXPECT_FALSE(worker_chip_queue_make_layout(2, std::numeric_limits<uint64_t>::max() - 63, 64, layout));

    EXPECT_EQ(layout.depth, original.depth);
    EXPECT_EQ(layout.input_desc_offset, original.input_desc_offset);
    EXPECT_EQ(layout.output_desc_offset, original.output_desc_offset);
    EXPECT_EQ(layout.input_arena_offset, original.input_arena_offset);
    EXPECT_EQ(layout.output_arena_offset, original.output_arena_offset);
    EXPECT_EQ(layout.input_arena_bytes, original.input_arena_bytes);
    EXPECT_EQ(layout.output_arena_bytes, original.output_arena_bytes);
    EXPECT_EQ(layout.payload_bytes, original.payload_bytes);
    EXPECT_EQ(layout.counter_bytes, original.counter_bytes);
}

TEST(WorkerChipMessageQueueTest, DescriptorSlotEncodingIsStable) {
    static_assert(std::is_standard_layout<WorkerChipQueueDescSlot>::value, "descriptor must be POD-like");
    static_assert(std::is_trivially_copyable<WorkerChipQueueDescSlot>::value, "descriptor must be fixed-size");
    static_assert(std::is_standard_layout<WorkerChipQueueError>::value, "error must be POD-like");
    static_assert(std::is_trivially_copyable<WorkerChipQueueError>::value, "error must be fixed-size");

    EXPECT_EQ(sizeof(WorkerChipQueueDescSlot), 32u);
    EXPECT_EQ(offsetof(WorkerChipQueueDescSlot, seq), 0u);
    EXPECT_EQ(offsetof(WorkerChipQueueDescSlot, opcode), 8u);
    EXPECT_EQ(offsetof(WorkerChipQueueDescSlot, payload_offset), 16u);
    EXPECT_EQ(offsetof(WorkerChipQueueDescSlot, payload_nbytes), 24u);
    EXPECT_EQ(sizeof(WorkerChipQueueError::message), 256u);

    WorkerChipQueueDescSlot slot{};
    worker_chip_queue_encode_desc(&slot, 7, WorkerChipQueueOpcode::ERROR, 128, 16);
    EXPECT_EQ(slot.seq, 7u);
    EXPECT_EQ(slot.opcode, 3u);
    EXPECT_EQ(slot.payload_offset, 128u);
    EXPECT_EQ(slot.payload_nbytes, 16u);
}

TEST(WorkerChipMessageQueueTest, ErrorOperationStringsAndMessageCopyAreStable) {
    EXPECT_STREQ(worker_chip_queue_op_to_string(WorkerChipQueueOp::INIT), "init");
    EXPECT_STREQ(worker_chip_queue_op_to_string(WorkerChipQueueOp::INPUT_TRY_PEEK), "input.try_peek");
    EXPECT_STREQ(worker_chip_queue_op_to_string(WorkerChipQueueOp::INPUT_RELEASE), "input.release");
    EXPECT_STREQ(worker_chip_queue_op_to_string(WorkerChipQueueOp::OUTPUT_TRY_RESERVE), "output.try_reserve");
    EXPECT_STREQ(worker_chip_queue_op_to_string(WorkerChipQueueOp::OUTPUT_PUBLISH), "output.publish");
    EXPECT_STREQ(worker_chip_queue_op_to_string(WorkerChipQueueOp::TIMEOUT), "timeout");
    EXPECT_STREQ(worker_chip_queue_op_to_string(static_cast<WorkerChipQueueOp>(99)), "unknown");

    char message[256];
    worker_chip_orch_comm::copy_error_message(message, sizeof(message), nullptr);
    EXPECT_STREQ(message, "");

    std::array<char, 300> long_message{};
    long_message.fill('x');
    long_message.back() = '\0';
    worker_chip_orch_comm::copy_error_message(message, sizeof(message), long_message.data());
    EXPECT_EQ(std::strlen(message), sizeof(message) - 1);
    EXPECT_EQ(message[254], 'x');
    EXPECT_EQ(message[255], '\0');
}

TEST(WorkerChipMessageQueueTest, Low32ReconstructionAcceptsWrapAndRejectsImpossibleDeltas) {
    uint64_t value = 0xFFFF'FFFFull;

    EXPECT_TRUE(worker_chip_queue_reconstruct_counter(0, 4, value));
    EXPECT_EQ(value, 0x1'0000'0000ull);

    value = (1ull << 31) - 2;
    EXPECT_TRUE(worker_chip_queue_reconstruct_counter(static_cast<int32_t>(0x8000'0001u), 4, value));
    EXPECT_EQ(value, (1ull << 31) + 1);

    value = 100;
    EXPECT_TRUE(worker_chip_queue_reconstruct_counter(104, 4, value));
    EXPECT_EQ(value, 104u);

    value = 100;
    EXPECT_FALSE(worker_chip_queue_reconstruct_counter(99, 4, value));

    value = 100;
    EXPECT_FALSE(worker_chip_queue_reconstruct_counter(105, 4, value));
}

TEST(WorkerChipMessageQueueTest, L2InputPeekHandlesZeroByteDescriptorBeforeArenaValidation) {
    RegionStorage storage{};
    WorkerChipQueueArgs args = make_args(2, 64, 64);
    WorkerChipQueueEndpoint<> queue(make_desc(&storage, args), args);
    ASSERT_EQ(queue.error().kind, WorkerChipQueueErrorKind::NONE) << queue.error().message;

    WorkerChipQueueDescSlot slot{};
    worker_chip_queue_encode_desc(&slot, 1, WorkerChipQueueOpcode::DATA, 0, 0);
    std::memcpy(storage.payload.data() + queue.layout().input_desc_offset, &slot, sizeof(slot));
    storage.counters[0] = 1;

    WorkerChipQueueInputHandle handle{};
    ASSERT_TRUE(queue.input().try_peek(handle)) << queue.error().message;

    EXPECT_EQ(handle.seq, 1u);
    EXPECT_EQ(handle.opcode, WorkerChipQueueOpcode::DATA);
    EXPECT_EQ(handle.payload_nbytes, 0u);
    EXPECT_EQ(handle.payload.gm_addr, 0u);
    EXPECT_TRUE(queue.input().release(handle)) << queue.error().message;
    EXPECT_EQ(storage.counters[16], 1);
}

TEST(WorkerChipMessageQueueTest, L2InputPeekPoisonsZeroByteDescriptorWithNonzeroOffset) {
    RegionStorage storage{};
    WorkerChipQueueArgs args = make_args(2, 64, 64);
    WorkerChipQueueEndpoint<> queue(make_desc(&storage, args), args);
    ASSERT_EQ(queue.error().kind, WorkerChipQueueErrorKind::NONE) << queue.error().message;

    WorkerChipQueueDescSlot slot{};
    worker_chip_queue_encode_desc(&slot, 1, WorkerChipQueueOpcode::DATA, 8, 0);
    std::memcpy(storage.payload.data() + queue.layout().input_desc_offset, &slot, sizeof(slot));
    storage.counters[0] = 1;

    WorkerChipQueueInputHandle handle{};
    EXPECT_FALSE(queue.input().try_peek(handle));

    EXPECT_EQ(queue.error().kind, WorkerChipQueueErrorKind::INVALID_DESCRIPTOR);
    EXPECT_EQ(storage.counters[80], 1);
}

TEST(WorkerChipMessageQueueTest, L2InputPeekExposesNonzeroPayloadBytes) {
    RegionStorage storage{};
    WorkerChipQueueArgs args = make_args(2, 64, 64);
    WorkerChipQueueEndpoint<> queue(make_desc(&storage, args), args);
    ASSERT_EQ(queue.error().kind, WorkerChipQueueErrorKind::NONE) << queue.error().message;
    const std::array<uint8_t, 4> payload{{0x11, 0x22, 0x33, 0x44}};
    std::memcpy(storage.payload.data() + queue.layout().input_arena_offset, payload.data(), payload.size());
    publish_input_desc(
        &storage, queue.layout(), 1, WorkerChipQueueOpcode::DATA, queue.layout().input_arena_offset, payload.size()
    );

    WorkerChipQueueInputHandle handle{};
    ASSERT_TRUE(queue.input().try_peek(handle)) << queue.error().message;

    ASSERT_EQ(handle.payload_nbytes, payload.size());
    const auto *observed = reinterpret_cast<const uint8_t *>(static_cast<uintptr_t>(handle.payload.gm_addr));
    EXPECT_EQ(std::memcmp(observed, payload.data(), payload.size()), 0);
    ASSERT_TRUE(queue.input().release(handle)) << queue.error().message;
    EXPECT_EQ(queue.error().kind, WorkerChipQueueErrorKind::NONE);
}

TEST(WorkerChipMessageQueueTest, L2InputPeekAllowsArenaWrapAtExpectedPayloadHead) {
    RegionStorage storage{};
    WorkerChipQueueArgs args = make_args(2, 128, 64);
    WorkerChipQueueEndpoint<> queue(make_desc(&storage, args), args);
    ASSERT_EQ(queue.error().kind, WorkerChipQueueErrorKind::NONE) << queue.error().message;

    publish_input_desc(&storage, queue.layout(), 1, WorkerChipQueueOpcode::DATA, queue.layout().input_arena_offset, 80);
    WorkerChipQueueInputHandle first{};
    ASSERT_TRUE(queue.input().try_peek(first)) << queue.error().message;
    ASSERT_TRUE(queue.input().release(first)) << queue.error().message;

    publish_input_desc(&storage, queue.layout(), 2, WorkerChipQueueOpcode::DATA, queue.layout().input_arena_offset, 64);
    WorkerChipQueueInputHandle second{};
    ASSERT_TRUE(queue.input().try_peek(second)) << queue.error().message;

    EXPECT_EQ(second.payload_offset, queue.layout().input_arena_offset);
    EXPECT_EQ(second.payload_nbytes, 64u);
    ASSERT_TRUE(queue.input().release(second)) << queue.error().message;
    EXPECT_EQ(queue.error().kind, WorkerChipQueueErrorKind::NONE);
}

TEST(WorkerChipMessageQueueTest, L2InputPeekRejectsPayloadOffsetMismatchBeforeRelease) {
    RegionStorage storage{};
    WorkerChipQueueArgs args = make_args(2, 128, 64);
    WorkerChipQueueEndpoint<> queue(make_desc(&storage, args), args);
    ASSERT_EQ(queue.error().kind, WorkerChipQueueErrorKind::NONE) << queue.error().message;
    publish_input_desc(
        &storage, queue.layout(), 1, WorkerChipQueueOpcode::DATA, queue.layout().input_arena_offset + 64, 16
    );

    WorkerChipQueueInputHandle handle{};
    EXPECT_FALSE(queue.input().try_peek(handle));

    EXPECT_EQ(queue.error().kind, WorkerChipQueueErrorKind::INVALID_DESCRIPTOR);
    EXPECT_EQ(storage.counters[counter_index(WORKER_CHIP_QUEUE_INPUT_DESC_HEAD_OFFSET)], 0);
    EXPECT_EQ(storage.counters[counter_index(WORKER_CHIP_QUEUE_CHIP_ABORT_FLAG_OFFSET)], 1);
}

TEST(WorkerChipMessageQueueTest, L2OutputReservePublishWritesDescriptorAndTail) {
    RegionStorage storage{};
    WorkerChipQueueArgs args = make_args(2, 64, 64);
    WorkerChipQueueEndpoint<> queue(make_desc(&storage, args), args);
    ASSERT_EQ(queue.error().kind, WorkerChipQueueErrorKind::NONE) << queue.error().message;

    WorkerChipQueueOutputReservation reservation{};
    ASSERT_TRUE(queue.output().try_reserve(16, reservation)) << queue.error().message;
    EXPECT_EQ(reservation.payload_nbytes, 16u);
    EXPECT_NE(reservation.payload.gm_addr, 0u);

    ASSERT_TRUE(queue.output().publish(reservation, WorkerChipQueueOpcode::DATA)) << queue.error().message;

    WorkerChipQueueDescSlot slot{};
    std::memcpy(&slot, storage.payload.data() + queue.layout().output_desc_offset, sizeof(slot));
    EXPECT_EQ(slot.seq, 1u);
    EXPECT_EQ(slot.opcode, 1u);
    EXPECT_EQ(slot.payload_nbytes, 16u);
    EXPECT_EQ(storage.counters[32], 1);
}

TEST(WorkerChipMessageQueueTest, L2OutputReserveReplaysReleasedDescriptorsBeforeReusingArena) {
    RegionStorage storage{};
    WorkerChipQueueArgs args = make_args(4, 64, 128);
    WorkerChipQueueEndpoint<> queue(make_desc(&storage, args), args);
    ASSERT_EQ(queue.error().kind, WorkerChipQueueErrorKind::NONE) << queue.error().message;

    WorkerChipQueueOutputReservation first{};
    ASSERT_TRUE(queue.output().try_reserve(80, first)) << queue.error().message;
    ASSERT_EQ(first.payload_offset, queue.layout().output_arena_offset);
    ASSERT_TRUE(queue.output().publish(first, WorkerChipQueueOpcode::DATA)) << queue.error().message;

    storage.counters[48] = 1;
    WorkerChipQueueOutputReservation second{};
    ASSERT_TRUE(queue.output().try_reserve(80, second)) << queue.error().message;

    EXPECT_EQ(second.payload_offset, queue.layout().output_arena_offset);
}

TEST(WorkerChipMessageQueueTest, RemoteAbortObservationDoesNotSetOwnAbortFlag) {
    RegionStorage storage{};
    WorkerChipQueueArgs args = make_args(2, 64, 64);
    WorkerChipQueueEndpoint<> queue(make_desc(&storage, args), args);
    ASSERT_EQ(queue.error().kind, WorkerChipQueueErrorKind::NONE) << queue.error().message;
    storage.counters[64] = 1;

    EXPECT_EQ(queue.disambiguate_timeout(), WorkerChipQueueTimeoutStatus::REMOTE_ABORTED);

    EXPECT_EQ(queue.error().kind, WorkerChipQueueErrorKind::REMOTE_ABORTED);
    EXPECT_EQ(storage.counters[80], 0);
}

TEST(WorkerChipMessageQueueTest, OrdinaryTimeoutDoesNotSetOwnAbortFlag) {
    RegionStorage storage{};
    WorkerChipQueueArgs args = make_args(2, 64, 64);
    WorkerChipQueueEndpoint<> queue(make_desc(&storage, args), args);
    ASSERT_EQ(queue.error().kind, WorkerChipQueueErrorKind::NONE) << queue.error().message;

    EXPECT_EQ(queue.disambiguate_timeout(), WorkerChipQueueTimeoutStatus::ORDINARY_TIMEOUT);

    EXPECT_EQ(queue.error().kind, WorkerChipQueueErrorKind::NONE);
    EXPECT_EQ(storage.counters[counter_index(WORKER_CHIP_QUEUE_CHIP_ABORT_FLAG_OFFSET)], 0);
}

TEST(WorkerChipMessageQueueTest, OutputCapacityEqualsDepthAndFullIsNoProgressWithoutAbort) {
    RegionStorage storage{};
    WorkerChipQueueArgs args = make_args(2, 64, 64);
    WorkerChipQueueEndpoint<> queue(make_desc(&storage, args), args);
    ASSERT_EQ(queue.error().kind, WorkerChipQueueErrorKind::NONE) << queue.error().message;

    for (int i = 0; i < 2; ++i) {
        WorkerChipQueueOutputReservation reservation{};
        ASSERT_TRUE(queue.output().try_reserve(0, reservation)) << queue.error().message;
        ASSERT_TRUE(queue.output().publish(reservation, WorkerChipQueueOpcode::DATA)) << queue.error().message;
    }
    WorkerChipQueueOutputReservation third{};
    EXPECT_FALSE(queue.output().try_reserve(0, third));

    EXPECT_EQ(queue.error().kind, WorkerChipQueueErrorKind::NONE);
    EXPECT_EQ(storage.counters[counter_index(WORKER_CHIP_QUEUE_OUTPUT_DESC_TAIL_OFFSET)], 2);
    EXPECT_EQ(storage.counters[counter_index(WORKER_CHIP_QUEUE_CHIP_ABORT_FLAG_OFFSET)], 0);
}

TEST(WorkerChipMessageQueueTest, FullAndEmptyUseMonotonicCountersNotMaskedIndices) {
    RegionStorage storage{};
    WorkerChipQueueArgs args = make_args(2, 64, 64);
    WorkerChipQueueEndpoint<> queue(make_desc(&storage, args), args);
    ASSERT_EQ(queue.error().kind, WorkerChipQueueErrorKind::NONE) << queue.error().message;

    for (int i = 0; i < 2; ++i) {
        WorkerChipQueueOutputReservation reservation{};
        ASSERT_TRUE(queue.output().try_reserve(0, reservation)) << queue.error().message;
        ASSERT_TRUE(queue.output().publish(reservation, WorkerChipQueueOpcode::DATA)) << queue.error().message;
    }
    storage.counters[counter_index(WORKER_CHIP_QUEUE_OUTPUT_DESC_HEAD_OFFSET)] = 1;

    WorkerChipQueueOutputReservation third{};
    ASSERT_TRUE(queue.output().try_reserve(0, third)) << queue.error().message;
    ASSERT_TRUE(queue.output().publish(third, WorkerChipQueueOpcode::DATA)) << queue.error().message;

    EXPECT_EQ(third.seq, 3u);
    EXPECT_EQ(storage.counters[counter_index(WORKER_CHIP_QUEUE_OUTPUT_DESC_TAIL_OFFSET)], 3);
    EXPECT_EQ(queue.error().kind, WorkerChipQueueErrorKind::NONE);
    EXPECT_EQ(storage.counters[counter_index(WORKER_CHIP_QUEUE_CHIP_ABORT_FLAG_OFFSET)], 0);
}

TEST(WorkerChipMessageQueueTest, OutputReserveTooLargeIsPreMutationNoProgressWithoutAbort) {
    RegionStorage storage{};
    WorkerChipQueueArgs args = make_args(2, 64, 64);
    WorkerChipQueueEndpoint<> queue(make_desc(&storage, args), args);
    ASSERT_EQ(queue.error().kind, WorkerChipQueueErrorKind::NONE) << queue.error().message;

    WorkerChipQueueOutputReservation reservation{};
    EXPECT_FALSE(queue.output().try_reserve(65, reservation));

    EXPECT_EQ(queue.error().kind, WorkerChipQueueErrorKind::NONE);
    EXPECT_EQ(storage.counters[counter_index(WORKER_CHIP_QUEUE_OUTPUT_DESC_TAIL_OFFSET)], 0);
    EXPECT_EQ(storage.counters[counter_index(WORKER_CHIP_QUEUE_CHIP_ABORT_FLAG_OFFSET)], 0);
}

TEST(WorkerChipMessageQueueTest, OutputPublishApplicationErrorDoesNotSetAbortFlag) {
    RegionStorage storage{};
    WorkerChipQueueArgs args = make_args(2, 64, 64);
    WorkerChipQueueEndpoint<> queue(make_desc(&storage, args), args);
    ASSERT_EQ(queue.error().kind, WorkerChipQueueErrorKind::NONE) << queue.error().message;

    WorkerChipQueueOutputReservation reservation{};
    ASSERT_TRUE(queue.output().try_reserve(0, reservation)) << queue.error().message;
    ASSERT_TRUE(queue.output().publish(reservation, WorkerChipQueueOpcode::ERROR)) << queue.error().message;

    WorkerChipQueueDescSlot slot{};
    std::memcpy(&slot, storage.payload.data() + queue.layout().output_desc_offset, sizeof(slot));
    EXPECT_EQ(slot.opcode, static_cast<uint64_t>(WorkerChipQueueOpcode::ERROR));
    EXPECT_EQ(queue.error().kind, WorkerChipQueueErrorKind::NONE);
    EXPECT_EQ(storage.counters[counter_index(WORKER_CHIP_QUEUE_CHIP_ABORT_FLAG_OFFSET)], 0);
}

TEST(WorkerChipMessageQueueTest, OutputPublishStaleReservationPoisonsAndSetsOwnAbortFlag) {
    RegionStorage storage{};
    WorkerChipQueueArgs args = make_args(2, 64, 64);
    WorkerChipQueueEndpoint<> queue(make_desc(&storage, args), args);
    ASSERT_EQ(queue.error().kind, WorkerChipQueueErrorKind::NONE) << queue.error().message;

    WorkerChipQueueOutputReservation reservation{};
    ASSERT_TRUE(queue.output().try_reserve(0, reservation)) << queue.error().message;
    ASSERT_TRUE(queue.output().publish(reservation, WorkerChipQueueOpcode::DATA)) << queue.error().message;
    EXPECT_FALSE(queue.output().publish(reservation, WorkerChipQueueOpcode::DATA));

    EXPECT_EQ(queue.error().kind, WorkerChipQueueErrorKind::OWNERSHIP);
    EXPECT_EQ(storage.counters[counter_index(WORKER_CHIP_QUEUE_CHIP_ABORT_FLAG_OFFSET)], 1);
}

TEST(WorkerChipMessageQueueTest, InputApplicationErrorIsNormalMessageAndDoesNotSetAbortFlag) {
    RegionStorage storage{};
    WorkerChipQueueArgs args = make_args(2, 64, 64);
    WorkerChipQueueEndpoint<> queue(make_desc(&storage, args), args);
    ASSERT_EQ(queue.error().kind, WorkerChipQueueErrorKind::NONE) << queue.error().message;
    publish_input_desc(&storage, queue.layout(), 1, WorkerChipQueueOpcode::ERROR);

    WorkerChipQueueInputHandle handle{};
    ASSERT_TRUE(queue.input().try_peek(handle)) << queue.error().message;
    EXPECT_EQ(handle.opcode, WorkerChipQueueOpcode::ERROR);
    ASSERT_TRUE(queue.input().release(handle)) << queue.error().message;

    EXPECT_EQ(queue.error().kind, WorkerChipQueueErrorKind::NONE);
    EXPECT_EQ(storage.counters[counter_index(WORKER_CHIP_QUEUE_CHIP_ABORT_FLAG_OFFSET)], 0);
}

TEST(WorkerChipMessageQueueTest, InputReleaseRejectsCallerMutatedHandleMetadata) {
    RegionStorage storage{};
    WorkerChipQueueArgs args = make_args(2, 64, 64);
    WorkerChipQueueEndpoint<> queue(make_desc(&storage, args), args);
    ASSERT_EQ(queue.error().kind, WorkerChipQueueErrorKind::NONE) << queue.error().message;
    publish_input_desc(&storage, queue.layout(), 1, WorkerChipQueueOpcode::DATA, queue.layout().input_arena_offset, 16);

    WorkerChipQueueInputHandle handle{};
    ASSERT_TRUE(queue.input().try_peek(handle)) << queue.error().message;
    handle.payload_nbytes = 0;

    EXPECT_FALSE(queue.input().release(handle));

    EXPECT_EQ(queue.error().kind, WorkerChipQueueErrorKind::OWNERSHIP);
    EXPECT_EQ(storage.counters[counter_index(WORKER_CHIP_QUEUE_INPUT_DESC_HEAD_OFFSET)], 0);
    EXPECT_EQ(storage.counters[counter_index(WORKER_CHIP_QUEUE_CHIP_ABORT_FLAG_OFFSET)], 1);
}

TEST(WorkerChipMessageQueueTest, InputStopReleaseRejectsLaterPublishedInputAsInvalidState) {
    RegionStorage storage{};
    WorkerChipQueueArgs args = make_args(2, 64, 64);
    WorkerChipQueueEndpoint<> queue(make_desc(&storage, args), args);
    ASSERT_EQ(queue.error().kind, WorkerChipQueueErrorKind::NONE) << queue.error().message;
    publish_input_desc(&storage, queue.layout(), 1, WorkerChipQueueOpcode::STOP);

    WorkerChipQueueInputHandle stop{};
    ASSERT_TRUE(queue.input().try_peek(stop)) << queue.error().message;
    ASSERT_TRUE(queue.input().release(stop)) << queue.error().message;

    publish_input_desc(&storage, queue.layout(), 2, WorkerChipQueueOpcode::DATA);
    WorkerChipQueueInputHandle later{};
    EXPECT_FALSE(queue.input().try_peek(later));

    EXPECT_EQ(queue.error().kind, WorkerChipQueueErrorKind::INVALID_DESCRIPTOR);
    EXPECT_EQ(storage.counters[counter_index(WORKER_CHIP_QUEUE_CHIP_ABORT_FLAG_OFFSET)], 1);
}

TEST(WorkerChipMessageQueueTest, InputStopWithPayloadMetadataPoisonsAndSetsOwnAbortFlag) {
    RegionStorage storage{};
    WorkerChipQueueArgs args = make_args(2, 64, 64);
    WorkerChipQueueEndpoint<> queue(make_desc(&storage, args), args);
    ASSERT_EQ(queue.error().kind, WorkerChipQueueErrorKind::NONE) << queue.error().message;
    publish_input_desc(&storage, queue.layout(), 1, WorkerChipQueueOpcode::STOP, queue.layout().input_arena_offset, 8);

    WorkerChipQueueInputHandle handle{};
    EXPECT_FALSE(queue.input().try_peek(handle));

    EXPECT_EQ(queue.error().kind, WorkerChipQueueErrorKind::INVALID_DESCRIPTOR);
    EXPECT_EQ(storage.counters[counter_index(WORKER_CHIP_QUEUE_CHIP_ABORT_FLAG_OFFSET)], 1);
}

TEST(WorkerChipMessageQueueTest, InputSecondPeekBeforeReleasePoisonsOwnershipAndSetsOwnAbortFlag) {
    RegionStorage storage{};
    WorkerChipQueueArgs args = make_args(2, 64, 64);
    WorkerChipQueueEndpoint<> queue(make_desc(&storage, args), args);
    ASSERT_EQ(queue.error().kind, WorkerChipQueueErrorKind::NONE) << queue.error().message;
    publish_input_desc(&storage, queue.layout(), 1, WorkerChipQueueOpcode::DATA);

    WorkerChipQueueInputHandle handle{};
    ASSERT_TRUE(queue.input().try_peek(handle)) << queue.error().message;
    WorkerChipQueueInputHandle second{};
    EXPECT_FALSE(queue.input().try_peek(second));

    EXPECT_EQ(queue.error().kind, WorkerChipQueueErrorKind::OWNERSHIP);
    EXPECT_EQ(queue.error().op, WorkerChipQueueOp::INPUT_TRY_PEEK);
    EXPECT_STREQ(worker_chip_queue_op_to_string(queue.error().op), "input.try_peek");
    EXPECT_EQ(storage.counters[counter_index(WORKER_CHIP_QUEUE_CHIP_ABORT_FLAG_OFFSET)], 1);
}

TEST(WorkerChipMessageQueueTest, MaxInflightGreaterThanDepthSetsBadArgumentWithoutAbortFlag) {
    RegionStorage too_large_storage{};
    WorkerChipQueueArgs args = make_args(2, 64, 64);

    WorkerChipQueueEndpoint<3> too_large(make_desc(&too_large_storage, args), args);
    EXPECT_EQ(too_large.error().kind, WorkerChipQueueErrorKind::BAD_ARGUMENT);
    EXPECT_EQ(too_large.error().op, WorkerChipQueueOp::INIT);
    EXPECT_EQ(too_large_storage.counters[counter_index(WORKER_CHIP_QUEUE_CHIP_ABORT_FLAG_OFFSET)], 0);
}

TEST(WorkerChipMessageQueueTest, MultiInflightAcquireAllowsSeveralDataInputs) {
    RegionStorage storage{};
    WorkerChipQueueArgs args = make_args(4, 128, 128);
    WorkerChipQueueEndpoint<3> queue(make_desc(&storage, args), args);
    ASSERT_EQ(queue.error().kind, WorkerChipQueueErrorKind::NONE) << queue.error().message;
    publish_input_desc(&storage, queue.layout(), 1, WorkerChipQueueOpcode::DATA);
    publish_input_desc(&storage, queue.layout(), 2, WorkerChipQueueOpcode::DATA);
    publish_input_desc(&storage, queue.layout(), 3, WorkerChipQueueOpcode::DATA);

    WorkerChipQueueInputHandle first{};
    WorkerChipQueueInputHandle second{};
    WorkerChipQueueInputHandle third{};
    EXPECT_TRUE(queue.input().try_peek(first)) << queue.error().message;
    EXPECT_TRUE(queue.input().try_peek(second)) << queue.error().message;
    EXPECT_TRUE(queue.input().try_peek(third)) << queue.error().message;

    EXPECT_EQ(first.seq, 1u);
    EXPECT_EQ(second.seq, 2u);
    EXPECT_EQ(third.seq, 3u);
    EXPECT_EQ(queue.error().kind, WorkerChipQueueErrorKind::NONE);
    EXPECT_EQ(storage.counters[counter_index(WORKER_CHIP_QUEUE_INPUT_DESC_HEAD_OFFSET)], 0);
}

TEST(WorkerChipMessageQueueTest, MultiInflightAcquireAllowsNonZeroPayloadOffsetsBeforeRelease) {
    RegionStorage storage{};
    WorkerChipQueueArgs args = make_args(4, 128, 128);
    WorkerChipQueueEndpoint<3> queue(make_desc(&storage, args), args);
    ASSERT_EQ(queue.error().kind, WorkerChipQueueErrorKind::NONE) << queue.error().message;
    const uint64_t first_offset = queue.layout().input_arena_offset;
    const uint64_t second_offset = first_offset + 16;
    publish_input_desc(&storage, queue.layout(), 1, WorkerChipQueueOpcode::DATA, first_offset, 16);
    publish_input_desc(&storage, queue.layout(), 2, WorkerChipQueueOpcode::DATA, second_offset, 16);

    WorkerChipQueueInputHandle first{};
    WorkerChipQueueInputHandle second{};
    ASSERT_TRUE(queue.input().try_peek(first)) << queue.error().message;
    ASSERT_TRUE(queue.input().try_peek(second)) << queue.error().message;

    EXPECT_EQ(first.payload_offset, first_offset);
    EXPECT_EQ(first.payload_nbytes, 16u);
    EXPECT_EQ(second.payload_offset, second_offset);
    EXPECT_EQ(second.payload_nbytes, 16u);
    EXPECT_EQ(queue.error().kind, WorkerChipQueueErrorKind::NONE);
    EXPECT_EQ(storage.counters[counter_index(WORKER_CHIP_QUEUE_INPUT_DESC_HEAD_OFFSET)], 0);
}

TEST(WorkerChipMessageQueueTest, ErrorCountsAgainstInputWindowAndFullDoesNotPoison) {
    RegionStorage storage{};
    WorkerChipQueueArgs args = make_args(4, 128, 128);
    WorkerChipQueueEndpoint<2> queue(make_desc(&storage, args), args);
    ASSERT_EQ(queue.error().kind, WorkerChipQueueErrorKind::NONE) << queue.error().message;
    publish_input_desc(&storage, queue.layout(), 1, WorkerChipQueueOpcode::DATA);
    publish_input_desc(&storage, queue.layout(), 2, WorkerChipQueueOpcode::ERROR);
    publish_input_desc(&storage, queue.layout(), 3, WorkerChipQueueOpcode::DATA);

    WorkerChipQueueInputHandle first{};
    WorkerChipQueueInputHandle second{};
    WorkerChipQueueInputHandle third{};
    ASSERT_TRUE(queue.input().try_peek(first)) << queue.error().message;
    ASSERT_TRUE(queue.input().try_peek(second)) << queue.error().message;
    EXPECT_FALSE(queue.input().try_peek(third));

    EXPECT_EQ(queue.error().kind, WorkerChipQueueErrorKind::NONE);
    EXPECT_EQ(storage.counters[counter_index(WORKER_CHIP_QUEUE_INPUT_DESC_HEAD_OFFSET)], 0);
    EXPECT_EQ(storage.counters[counter_index(WORKER_CHIP_QUEUE_CHIP_ABORT_FLAG_OFFSET)], 0);
}

TEST(WorkerChipMessageQueueTest, OutOfOrderInputReleaseOnlyAdvancesCompletedFifoPrefix) {
    RegionStorage storage{};
    WorkerChipQueueArgs args = make_args(4, 128, 128);
    WorkerChipQueueEndpoint<3> queue(make_desc(&storage, args), args);
    ASSERT_EQ(queue.error().kind, WorkerChipQueueErrorKind::NONE) << queue.error().message;
    publish_input_desc(&storage, queue.layout(), 1, WorkerChipQueueOpcode::DATA);
    publish_input_desc(&storage, queue.layout(), 2, WorkerChipQueueOpcode::DATA);
    publish_input_desc(&storage, queue.layout(), 3, WorkerChipQueueOpcode::DATA);

    WorkerChipQueueInputHandle first{};
    WorkerChipQueueInputHandle second{};
    WorkerChipQueueInputHandle third{};
    ASSERT_TRUE(queue.input().try_peek(first)) << queue.error().message;
    ASSERT_TRUE(queue.input().try_peek(second)) << queue.error().message;
    ASSERT_TRUE(queue.input().try_peek(third)) << queue.error().message;

    ASSERT_TRUE(queue.input().release(second)) << queue.error().message;
    EXPECT_EQ(storage.counters[counter_index(WORKER_CHIP_QUEUE_INPUT_DESC_HEAD_OFFSET)], 0);

    ASSERT_TRUE(queue.input().release(first)) << queue.error().message;
    EXPECT_EQ(storage.counters[counter_index(WORKER_CHIP_QUEUE_INPUT_DESC_HEAD_OFFSET)], 2);

    ASSERT_TRUE(queue.input().release(third)) << queue.error().message;
    EXPECT_EQ(storage.counters[counter_index(WORKER_CHIP_QUEUE_INPUT_DESC_HEAD_OFFSET)], 3);
}

TEST(WorkerChipMessageQueueTest, RingWrapReleaseKeepsLogicalFifoOrder) {
    RegionStorage storage{};
    WorkerChipQueueArgs args = make_args(4, 128, 128);
    WorkerChipQueueEndpoint<3> queue(make_desc(&storage, args), args);
    ASSERT_EQ(queue.error().kind, WorkerChipQueueErrorKind::NONE) << queue.error().message;

    for (uint64_t seq = 1; seq <= 3; ++seq) {
        publish_input_desc(&storage, queue.layout(), seq, WorkerChipQueueOpcode::DATA);
    }

    WorkerChipQueueInputHandle first{};
    WorkerChipQueueInputHandle second{};
    WorkerChipQueueInputHandle third{};
    ASSERT_TRUE(queue.input().try_peek(first)) << queue.error().message;
    ASSERT_TRUE(queue.input().try_peek(second)) << queue.error().message;
    ASSERT_TRUE(queue.input().try_peek(third)) << queue.error().message;
    ASSERT_TRUE(queue.input().release(first)) << queue.error().message;
    ASSERT_TRUE(queue.input().release(second)) << queue.error().message;
    ASSERT_TRUE(queue.input().release(third)) << queue.error().message;
    EXPECT_EQ(storage.counters[counter_index(WORKER_CHIP_QUEUE_INPUT_DESC_HEAD_OFFSET)], 3);

    for (uint64_t seq = 4; seq <= 6; ++seq) {
        publish_input_desc(&storage, queue.layout(), seq, WorkerChipQueueOpcode::DATA);
    }

    WorkerChipQueueInputHandle fourth{};
    WorkerChipQueueInputHandle fifth{};
    WorkerChipQueueInputHandle sixth{};
    ASSERT_TRUE(queue.input().try_peek(fourth)) << queue.error().message;
    ASSERT_TRUE(queue.input().try_peek(fifth)) << queue.error().message;
    ASSERT_TRUE(queue.input().try_peek(sixth)) << queue.error().message;

    ASSERT_TRUE(queue.input().release(fifth)) << queue.error().message;
    EXPECT_EQ(storage.counters[counter_index(WORKER_CHIP_QUEUE_INPUT_DESC_HEAD_OFFSET)], 3);
    ASSERT_TRUE(queue.input().release(fourth)) << queue.error().message;
    EXPECT_EQ(storage.counters[counter_index(WORKER_CHIP_QUEUE_INPUT_DESC_HEAD_OFFSET)], 5);
    ASSERT_TRUE(queue.input().release(sixth)) << queue.error().message;
    EXPECT_EQ(storage.counters[counter_index(WORKER_CHIP_QUEUE_INPUT_DESC_HEAD_OFFSET)], 6);
    EXPECT_EQ(queue.error().kind, WorkerChipQueueErrorKind::NONE);
}

TEST(WorkerChipMessageQueueTest, StopDoesNotCountAgainstWindowAndDrainsAfterEarlierInputs) {
    RegionStorage storage{};
    WorkerChipQueueArgs args = make_args(4, 128, 128);
    WorkerChipQueueEndpoint<2> queue(make_desc(&storage, args), args);
    ASSERT_EQ(queue.error().kind, WorkerChipQueueErrorKind::NONE) << queue.error().message;
    publish_input_desc(&storage, queue.layout(), 1, WorkerChipQueueOpcode::DATA);
    publish_input_desc(&storage, queue.layout(), 2, WorkerChipQueueOpcode::DATA);
    publish_input_desc(&storage, queue.layout(), 3, WorkerChipQueueOpcode::STOP);

    WorkerChipQueueInputHandle first{};
    WorkerChipQueueInputHandle second{};
    WorkerChipQueueInputHandle stop{};
    ASSERT_TRUE(queue.input().try_peek(first)) << queue.error().message;
    ASSERT_TRUE(queue.input().try_peek(second)) << queue.error().message;
    ASSERT_TRUE(queue.input().try_peek(stop)) << queue.error().message;
    EXPECT_EQ(stop.opcode, WorkerChipQueueOpcode::STOP);

    ASSERT_TRUE(queue.input().release(stop)) << queue.error().message;
    EXPECT_FALSE(queue.input().drained());
    EXPECT_EQ(storage.counters[counter_index(WORKER_CHIP_QUEUE_INPUT_DESC_HEAD_OFFSET)], 0);

    ASSERT_TRUE(queue.input().release(first)) << queue.error().message;
    EXPECT_FALSE(queue.input().drained());
    EXPECT_EQ(storage.counters[counter_index(WORKER_CHIP_QUEUE_INPUT_DESC_HEAD_OFFSET)], 1);

    ASSERT_TRUE(queue.input().release(second)) << queue.error().message;
    EXPECT_TRUE(queue.input().drained());
    EXPECT_EQ(storage.counters[counter_index(WORKER_CHIP_QUEUE_INPUT_DESC_HEAD_OFFSET)], 3);
}

TEST(WorkerChipMessageQueueTest, TryPeekAfterStopAcquireIsNoProgressWithoutPoison) {
    RegionStorage storage{};
    WorkerChipQueueArgs args = make_args(4, 128, 128);
    WorkerChipQueueEndpoint<2> queue(make_desc(&storage, args), args);
    ASSERT_EQ(queue.error().kind, WorkerChipQueueErrorKind::NONE) << queue.error().message;
    publish_input_desc(&storage, queue.layout(), 1, WorkerChipQueueOpcode::DATA);
    publish_input_desc(&storage, queue.layout(), 2, WorkerChipQueueOpcode::STOP);

    WorkerChipQueueInputHandle first{};
    WorkerChipQueueInputHandle stop{};
    WorkerChipQueueInputHandle later{};
    ASSERT_TRUE(queue.input().try_peek(first)) << queue.error().message;
    ASSERT_TRUE(queue.input().try_peek(stop)) << queue.error().message;
    EXPECT_FALSE(queue.input().try_peek(later));

    EXPECT_EQ(queue.error().kind, WorkerChipQueueErrorKind::NONE);
    EXPECT_EQ(storage.counters[counter_index(WORKER_CHIP_QUEUE_CHIP_ABORT_FLAG_OFFSET)], 0);
}

TEST(WorkerChipMessageQueueTest, StopAcquirePoisonsIfLaterInputAlreadyObserved) {
    RegionStorage storage{};
    WorkerChipQueueArgs args = make_args(4, 128, 128);
    WorkerChipQueueEndpoint<2> queue(make_desc(&storage, args), args);
    ASSERT_EQ(queue.error().kind, WorkerChipQueueErrorKind::NONE) << queue.error().message;
    publish_input_desc(&storage, queue.layout(), 1, WorkerChipQueueOpcode::DATA);
    publish_input_desc(&storage, queue.layout(), 2, WorkerChipQueueOpcode::STOP);
    publish_input_desc(&storage, queue.layout(), 3, WorkerChipQueueOpcode::DATA);

    WorkerChipQueueInputHandle input{};
    WorkerChipQueueInputHandle stop{};
    ASSERT_TRUE(queue.input().try_peek(input)) << queue.error().message;
    EXPECT_FALSE(queue.input().try_peek(stop));

    EXPECT_EQ(queue.error().kind, WorkerChipQueueErrorKind::INVALID_DESCRIPTOR);
    EXPECT_EQ(storage.counters[counter_index(WORKER_CHIP_QUEUE_CHIP_ABORT_FLAG_OFFSET)], 1);
}

TEST(WorkerChipMessageQueueTest, InputReleaseRejectsStaleAndDoubleRelease) {
    RegionStorage stale_storage{};
    RegionStorage double_storage{};
    WorkerChipQueueArgs args = make_args(4, 128, 128);

    WorkerChipQueueEndpoint<2> stale_queue(make_desc(&stale_storage, args), args);
    ASSERT_EQ(stale_queue.error().kind, WorkerChipQueueErrorKind::NONE) << stale_queue.error().message;
    publish_input_desc(&stale_storage, stale_queue.layout(), 1, WorkerChipQueueOpcode::DATA);
    publish_input_desc(&stale_storage, stale_queue.layout(), 2, WorkerChipQueueOpcode::DATA);
    WorkerChipQueueInputHandle stale_first{};
    WorkerChipQueueInputHandle stale_second{};
    ASSERT_TRUE(stale_queue.input().try_peek(stale_first)) << stale_queue.error().message;
    ASSERT_TRUE(stale_queue.input().try_peek(stale_second)) << stale_queue.error().message;
    ASSERT_TRUE(stale_queue.input().release(stale_first)) << stale_queue.error().message;
    ASSERT_TRUE(stale_queue.input().release(stale_second)) << stale_queue.error().message;
    EXPECT_FALSE(stale_queue.input().release(stale_first));
    EXPECT_EQ(stale_queue.error().kind, WorkerChipQueueErrorKind::OWNERSHIP);
    EXPECT_EQ(stale_storage.counters[counter_index(WORKER_CHIP_QUEUE_CHIP_ABORT_FLAG_OFFSET)], 1);

    WorkerChipQueueEndpoint<2> double_queue(make_desc(&double_storage, args), args);
    ASSERT_EQ(double_queue.error().kind, WorkerChipQueueErrorKind::NONE) << double_queue.error().message;
    publish_input_desc(&double_storage, double_queue.layout(), 1, WorkerChipQueueOpcode::DATA);
    publish_input_desc(&double_storage, double_queue.layout(), 2, WorkerChipQueueOpcode::DATA);
    WorkerChipQueueInputHandle first{};
    WorkerChipQueueInputHandle second{};
    ASSERT_TRUE(double_queue.input().try_peek(first)) << double_queue.error().message;
    ASSERT_TRUE(double_queue.input().try_peek(second)) << double_queue.error().message;
    ASSERT_TRUE(double_queue.input().release(second)) << double_queue.error().message;
    EXPECT_FALSE(double_queue.input().release(second));
    EXPECT_EQ(double_queue.error().kind, WorkerChipQueueErrorKind::OWNERSHIP);
    EXPECT_EQ(double_storage.counters[counter_index(WORKER_CHIP_QUEUE_CHIP_ABORT_FLAG_OFFSET)], 1);
}

TEST(WorkerChipMessageQueueTest, OutputReservePublishWorksDuringStopDrain) {
    RegionStorage storage{};
    WorkerChipQueueArgs args = make_args(4, 128, 128);
    WorkerChipQueueEndpoint<2> queue(make_desc(&storage, args), args);
    ASSERT_EQ(queue.error().kind, WorkerChipQueueErrorKind::NONE) << queue.error().message;
    publish_input_desc(&storage, queue.layout(), 1, WorkerChipQueueOpcode::DATA);
    publish_input_desc(&storage, queue.layout(), 2, WorkerChipQueueOpcode::STOP);

    WorkerChipQueueInputHandle input{};
    WorkerChipQueueInputHandle stop{};
    ASSERT_TRUE(queue.input().try_peek(input)) << queue.error().message;
    ASSERT_TRUE(queue.input().try_peek(stop)) << queue.error().message;
    ASSERT_TRUE(queue.input().release(stop)) << queue.error().message;

    WorkerChipQueueOutputReservation reservation{};
    ASSERT_TRUE(queue.output().try_reserve(16, reservation)) << queue.error().message;
    ASSERT_TRUE(queue.output().publish(reservation, WorkerChipQueueOpcode::DATA)) << queue.error().message;

    EXPECT_FALSE(queue.input().drained());
    EXPECT_EQ(storage.counters[counter_index(WORKER_CHIP_QUEUE_OUTPUT_DESC_TAIL_OFFSET)], 1);
    EXPECT_EQ(queue.error().kind, WorkerChipQueueErrorKind::NONE);
}

}  // namespace
