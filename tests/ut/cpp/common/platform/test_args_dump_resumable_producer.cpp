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

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include "aicpu/args_dump_aicpu.h"
#include "aicpu/device_run_result_base_aicpu.h"
#include "support/test_task_id.h"
#include "common/diagnostic_run_terminal.h"

namespace {
class ResumableDumpTest : public ::testing::Test {
protected:
    void SetUp() override {
        const size_t bytes = calc_dump_data_size(2);
        shared.reset(static_cast<uint8_t *>(std::aligned_alloc(64, (bytes + 63) / 64 * 64)));
        ASSERT_NE(shared, nullptr);
        std::memset(shared.get(), 0, bytes);
        header = get_dump_header(shared.get());
        header->magic = ARGS_DUMP_MAGIC;
        header->num_dump_threads = 2;
        for (int lane = 0; lane < 2; ++lane) {
            meta[lane] = std::make_unique<DumpMetaBuffer>();
            spare[lane] = std::make_unique<DumpMetaBuffer>();
            states[lane] = get_dump_buffer_state(shared.get(), lane);
            states[lane]->free_queue.buffer_ptrs[0] = reinterpret_cast<uint64_t>(meta[lane].get());
            states[lane]->free_queue.buffer_ptrs[1] = reinterpret_cast<uint64_t>(spare[lane].get());
            states[lane]->free_queue.tail = 2;
            states[lane]->arena_base = reinterpret_cast<uint64_t>(arenas[lane].data());
            states[lane]->arena_size = arenas[lane].size();
        }
        for (uint32_t i = 0; i < source.size(); ++i)
            source[i] = i + 10;
        set_dump_args_enabled(true);
        set_platform_dump_base(reinterpret_cast<uint64_t>(shared.get()));
        set_platform_run_result(0, 123);
        dump_args_init(2, DumpArgsLevel::FULL);
    }
    void TearDown() override {
        set_platform_dump_args_run_terminal(0);
        set_platform_dump_base(0);
        set_platform_run_result(0, 0);
        set_dump_args_enabled(false);
    }
    ArgsDumpInfo tensor(uint32_t elements = 8) {
        ArgsDumpInfo info{};
        info.task_id = simpler::ut::test_task_id(23);
        info.buffer_addr = reinterpret_cast<uint64_t>(source.data());
        info.dtype = static_cast<uint8_t>(DataType::UINT32);
        info.ndims = 1;
        info.shapes[0] = elements;
        info.strides[0] = 1;
        info.kind = static_cast<uint8_t>(ArgsDumpKind::TENSOR);
        info.role = ArgsDumpRole::INPUT;
        info.stage = ArgsDumpStage::BEFORE_DISPATCH;
        info.func_count = 1;
        info.func_ids[0] = 7;
        return info;
    }
    struct Free {
        void operator()(uint8_t *p) const { std::free(p); }
    };
    std::unique_ptr<uint8_t, Free> shared;
    DumpDataHeader *header{};
    std::array<DumpBufferState *, 2> states{};
    std::array<std::unique_ptr<DumpMetaBuffer>, 2> meta, spare;
    std::array<std::array<uint8_t, 128>, 2> arenas{};
    std::array<uint32_t, 32> source{};
};

TEST_F(ResumableDumpTest, HostAndProducerShareTaskSelectionAtEveryLevel) {
    struct Case {
        DumpArgsLevel level;
        bool empty_mask_selected;
        bool nonempty_mask_selected;
    };
    const Case cases[] = {
        {DumpArgsLevel::OFF, false, false},
        {DumpArgsLevel::PARTIAL, false, true},
        {DumpArgsLevel::FULL, true, true},
        {DumpArgsLevel::HYBRID, true, true},
    };
    for (const auto &test : cases) {
        dump_args_init(2, test.level);
        EXPECT_EQ(dump_args_task_selected(test.level, ARGS_DUMP_ARG_MASK_NONE), test.empty_mask_selected);
        EXPECT_EQ(should_dump_task(ARGS_DUMP_ARG_MASK_NONE), test.empty_mask_selected);
        EXPECT_EQ(dump_args_task_selected(test.level, ArgsDumpArgMask{1}), test.nonempty_mask_selected);
        EXPECT_EQ(should_dump_task(ArgsDumpArgMask{1}), test.nonempty_mask_selected);
    }
}

TEST_F(ResumableDumpTest, PayloadRemainsUnpublishedUntilExplicitCommit) {
    DumpArgReservation a, b;
    const auto info = tensor();
    ASSERT_EQ(dump_arg_try_begin(0, info, &a), DumpArgBeginResult::Ready);
    EXPECT_EQ(dump_arg_try_begin(0, info, &b), DumpArgBeginResult::WouldBlock);
    EXPECT_EQ(states[0]->run_dropped_record_count, 0u);
    EXPECT_EQ(dump_args_try_flush(0), DumpArgFlushResult::WouldBlock);
    EXPECT_FALSE(dump_arg_commit(&a));
    EXPECT_EQ(dump_arg_copy_chunk(&a, 5), DumpArgCopyResult::InProgress);
    EXPECT_EQ(a.copied_bytes(), 5u);
    EXPECT_EQ(meta[0]->count, 0u);
    EXPECT_EQ(header->queue_tails[0], 0u);
    while (dump_arg_copy_chunk(&a, 5) == DumpArgCopyResult::InProgress) {}
    EXPECT_EQ(meta[0]->count, 0u);
    ASSERT_TRUE(dump_arg_commit(&a));
    EXPECT_FALSE(a.open());
    EXPECT_FALSE(dump_arg_commit(&a));
    EXPECT_EQ(meta[0]->count, 1u);
    EXPECT_EQ(meta[0]->records[0].payload_size, 32u);
    EXPECT_EQ(std::memcmp(arenas[0].data(), source.data(), 32), 0);
}

TEST_F(ResumableDumpTest, AbandonRestoresCursorAndCountsOneRefusal) {
    DumpArgReservation a;
    ASSERT_EQ(dump_arg_try_begin(0, tensor(), &a), DumpArgBeginResult::Ready);
    ASSERT_EQ(dump_arg_copy_chunk(&a, 3), DumpArgCopyResult::InProgress);
    EXPECT_TRUE(dump_arg_abandon(&a));
    EXPECT_FALSE(dump_arg_abandon(&a));
    EXPECT_EQ(states[0]->arena_write_offset, 0u);
    EXPECT_EQ(meta[0]->count, 0u);
    EXPECT_EQ(states[0]->run_dropped_record_count, 1u);
    ASSERT_EQ(dump_arg_try_begin(0, tensor(), &a), DumpArgBeginResult::Ready);
    EXPECT_EQ(dump_arg_copy_chunk(&a, 128), DumpArgCopyResult::Complete);
    ASSERT_TRUE(dump_arg_commit(&a));
    EXPECT_EQ(meta[0]->records[0].payload_offset, 0u);
}

TEST_F(ResumableDumpTest, StridedGatherResumesAcrossPartialElements) {
    auto info = tensor();
    info.ndims = 2;
    info.shapes[0] = 2;
    info.shapes[1] = 3;
    info.strides[0] = 8;
    info.strides[1] = 2;
    info.start_offset = 1;
    DumpArgReservation a;
    ASSERT_EQ(dump_arg_try_begin(0, info, &a), DumpArgBeginResult::Ready);
    for (uint64_t before = 0; before < 24; before += 5) {
        const auto result = dump_arg_copy_chunk(&a, 5);
        EXPECT_EQ(a.copied_bytes(), std::min<uint64_t>(before + 5, 24));
        EXPECT_EQ(result, before + 5 < 24 ? DumpArgCopyResult::InProgress : DumpArgCopyResult::Complete);
    }
    ASSERT_TRUE(dump_arg_commit(&a));
    const uint32_t expected[] = {11, 13, 15, 19, 21, 23};
    EXPECT_EQ(std::memcmp(arenas[0].data(), expected, sizeof(expected)), 0);
    EXPECT_EQ(meta[0]->records[0].is_contiguous, 0);
}

TEST_F(ResumableDumpTest, PausedLaneDoesNotBlockAnotherProducer) {
    DumpArgReservation a, b;
    ASSERT_EQ(dump_arg_try_begin(0, tensor(), &a), DumpArgBeginResult::Ready);
    EXPECT_EQ(dump_arg_copy_chunk(&a, 1), DumpArgCopyResult::InProgress);
    ASSERT_EQ(dump_arg_try_begin(1, tensor(2), &b), DumpArgBeginResult::Ready);
    EXPECT_EQ(dump_arg_copy_chunk(&b, 8), DumpArgCopyResult::Complete);
    EXPECT_TRUE(dump_arg_commit(&b));
    EXPECT_EQ(meta[1]->count, 1u);
    EXPECT_EQ(meta[0]->count, 0u);
    EXPECT_TRUE(dump_arg_abandon(&a));
}

TEST_F(ResumableDumpTest, ArenaPressureWaitsWithoutRefusalsOrOverwrite) {
    states[0]->arena_size = 32;
    ASSERT_EQ(dump_arg_record(0, tensor()), 0);
    DumpArgReservation a;
    for (int retry = 0; retry < 3; ++retry) {
        EXPECT_EQ(dump_arg_try_begin(0, tensor(), &a), DumpArgBeginResult::WouldBlock);
        EXPECT_EQ(states[0]->run_dropped_record_count, 0u);
        EXPECT_EQ(std::memcmp(arenas[0].data(), source.data(), 32), 0);
    }
    EXPECT_EQ(states[0]->published_payload_count, 1u);
    states[0]->completed_payload_count = 1;
    ASSERT_EQ(dump_arg_try_begin(0, tensor(), &a), DumpArgBeginResult::Ready);
    EXPECT_EQ(dump_arg_copy_chunk(&a, 32), DumpArgCopyResult::Complete);
    EXPECT_TRUE(dump_arg_commit(&a));
    EXPECT_EQ(spare[0]->records[0].payload_offset, 32u);
    EXPECT_EQ(states[0]->run_dropped_record_count, 0u);
}

TEST_F(ResumableDumpTest, FullReadyQueueReturnsImmediatelyWithoutDroppingCommittedRecords) {
    ASSERT_EQ(dump_arg_record(0, tensor()), 0);
    header->queue_heads[0] = 1;
    EXPECT_EQ(dump_args_try_flush(0), DumpArgFlushResult::WouldBlock);
    EXPECT_EQ(meta[0]->count, 1u);
    EXPECT_EQ(states[0]->run_dropped_record_count, 0u);
    header->queue_heads[0] = 0;
    EXPECT_EQ(dump_args_try_flush(0), DumpArgFlushResult::Complete);
    EXPECT_EQ(header->queue_tails[0], 1u);
}

TEST_F(ResumableDumpTest, InvalidCopyDoesNotAdvanceOrCommitReservation) {
    DumpArgReservation a;
    ASSERT_EQ(dump_arg_try_begin(0, tensor(), &a), DumpArgBeginResult::Ready);
    EXPECT_EQ(dump_arg_copy_chunk(&a, 0), DumpArgCopyResult::Invalid);
    EXPECT_EQ(a.copied_bytes(), 0u);
    EXPECT_FALSE(dump_arg_commit(&a));
    EXPECT_TRUE(dump_arg_abandon(&a));
}

TEST_F(ResumableDumpTest, CallerInfoIsSnapshottedAndMetadataOnlyPayloadsStayEmpty) {
    auto info = tensor();
    DumpArgReservation a;
    ASSERT_EQ(dump_arg_try_begin(0, info, &a), DumpArgBeginResult::Ready);
    info.buffer_addr = 0;
    info.task_id = simpler::ut::test_task_id(99);
    EXPECT_EQ(dump_arg_copy_chunk(&a, 128), DumpArgCopyResult::Complete);
    ASSERT_TRUE(dump_arg_commit(&a));
    EXPECT_EQ(meta[0]->records[0].task_id, simpler::ut::test_task_id(23));
    EXPECT_EQ(std::memcmp(arenas[0].data(), source.data(), 32), 0);
    dump_args_init(2, DumpArgsLevel::HYBRID);
    info = tensor();
    ASSERT_EQ(dump_arg_try_begin(0, info, &a), DumpArgBeginResult::Ready);
    EXPECT_EQ(a.payload_bytes(), 0u);
    EXPECT_EQ(dump_arg_copy_chunk(&a, 1), DumpArgCopyResult::Complete);
    ASSERT_TRUE(dump_arg_commit(&a));
    EXPECT_EQ(meta[0]->records[0].payload_size, 0u);
}

TEST_F(ResumableDumpTest, OpenAndAbandonedReservationsNeverPublishSettledTerminal) {
    std::vector<DiagnosticRunTerminal> bank(PLATFORM_MAX_AICPU_THREADS);
    set_platform_dump_args_run_terminal(reinterpret_cast<uint64_t>(bank.data()));
    dump_args_init(2, DumpArgsLevel::FULL);
    DumpArgReservation a;
    ASSERT_EQ(dump_arg_try_begin(0, tensor(), &a), DumpArgBeginResult::Ready);
    dump_args_close_run(0);
    EXPECT_EQ(bank[0].run_epoch, 123u);
    EXPECT_NE(bank[0].counts_unknown, 0u);
    EXPECT_TRUE(dump_arg_abandon(&a));
    dump_args_close_run(0);
    EXPECT_EQ(bank[0].run_epoch, 123u);
    EXPECT_NE(bank[0].counts_unknown, 0u);
    EXPECT_EQ(states[0]->run_dropped_record_count, 1u);
    set_platform_dump_args_run_terminal(0);
}

TEST_F(ResumableDumpTest, AbandonRestoresPreWrapOffsetAndTruncationRemainsBounded) {
    states[0]->arena_write_offset = 124;
    DumpArgReservation a;
    ASSERT_EQ(dump_arg_try_begin(0, tensor(2), &a), DumpArgBeginResult::Ready);
    EXPECT_EQ(a.arena_offset, 128u);
    EXPECT_TRUE(dump_arg_abandon(&a));
    EXPECT_EQ(states[0]->arena_write_offset, 124u);
    dump_args_init(2, DumpArgsLevel::FULL);
    states[0]->arena_write_offset = 0;
    ASSERT_EQ(dump_arg_try_begin(0, tensor(64), &a), DumpArgBeginResult::Ready);
    EXPECT_EQ(a.payload_bytes(), 64u);
    EXPECT_EQ(dump_arg_copy_chunk(&a, 64), DumpArgCopyResult::Complete);
    ASSERT_TRUE(dump_arg_commit(&a));
    EXPECT_EQ(meta[0]->records[0].truncated, 1);
    EXPECT_EQ(meta[0]->records[0].payload_size, 64u);
}

TEST_F(ResumableDumpTest, MetadataExhaustionAndFullQueueDoNotAccountRetryAsLoss) {
    auto info = tensor();
    info.kind = static_cast<uint8_t>(ArgsDumpKind::SCALAR);
    for (uint32_t i = 0; i < PLATFORM_DUMP_RECORDS_PER_BUFFER; ++i)
        ASSERT_EQ(dump_arg_record(0, info), 0);
    header->queue_heads[0] = 1;
    DumpArgReservation a;
    EXPECT_EQ(dump_arg_try_begin(0, info, &a), DumpArgBeginResult::WouldBlock);
    EXPECT_EQ(meta[0]->count, PLATFORM_DUMP_RECORDS_PER_BUFFER);
    EXPECT_EQ(states[0]->run_dropped_record_count, 0u);
    header->queue_heads[0] = 0;
    states[0]->free_queue.head = states[0]->free_queue.tail;
    EXPECT_EQ(dump_arg_try_begin(0, info, &a), DumpArgBeginResult::WouldBlock);
    EXPECT_EQ(states[0]->run_dropped_record_count, 0u);
    states[0]->free_queue.buffer_ptrs[2 % PLATFORM_DUMP_SLOT_COUNT] = reinterpret_cast<uint64_t>(spare[0].get());
    states[0]->free_queue.tail = 3;
    ASSERT_EQ(dump_arg_try_begin(0, info, &a), DumpArgBeginResult::Ready);
    EXPECT_EQ(dump_arg_copy_chunk(&a, 1), DumpArgCopyResult::Complete);
    ASSERT_TRUE(dump_arg_commit(&a));
    EXPECT_EQ(spare[0]->count, 1u);
    EXPECT_EQ(states[0]->run_dropped_record_count, 0u);
}

struct IteratorTensor {
    struct {
        uint64_t addr;
    } buffer{};
    DataType dtype{DataType::UINT32};
    uint32_t ndims{1};
    uint64_t start_offset{0};
    uint32_t shapes[1]{2};
    uint32_t strides[1]{1};
};
struct IteratorPayload {
    int32_t tensor_count{2};
    int32_t scalar_count{1};
    IteratorTensor tensors[2]{};
    uint64_t scalars[1]{0x123456789abcdef0ULL};
    const IteratorTensor *tensor_data() const { return tensors; }
    const uint64_t *scalar_data() const { return scalars; }
};
struct IteratorDescriptor {
    TaskId task_id;
    int32_t kernel_id[3];
};

TEST_F(ResumableDumpTest, ResumableEnumerationAndBlockingWrapperShareSelectionAndScalarBits) {
    const ArgDirection directions[] = {ArgDirection::IN, ArgDirection::OUT};
    const auto callable = make_callable<CORE_MAX_TENSOR_ARGS>(directions, 2, nullptr, 0);
    IteratorDescriptor descriptor{simpler::ut::test_task_id(42), {7, 8, 9}};
    IteratorPayload payload;
    payload.tensors[0].buffer.addr = reinterpret_cast<uint64_t>(source.data());
    payload.tensors[1].buffer.addr = reinterpret_cast<uint64_t>(source.data() + 2);
    auto get_bin = [&](int32_t) {
        return reinterpret_cast<uint64_t>(callable.data());
    };
    uint32_t cursor = 0;
    ArgsDumpInfo info{};
    ASSERT_TRUE(
        next_dump_arg_for_task<3>(
            0, descriptor, payload, LaneMask::all(), ArgsDumpStage::BEFORE_DISPATCH, get_bin, &cursor, &info
        )
    );
    EXPECT_EQ(info.arg_index, 0u);
    EXPECT_EQ(info.func_count, 3);
    ASSERT_TRUE(
        next_dump_arg_for_task<3>(
            0, descriptor, payload, LaneMask::all(), ArgsDumpStage::BEFORE_DISPATCH, get_bin, &cursor, &info
        )
    );
    EXPECT_EQ(info.kind, static_cast<uint8_t>(ArgsDumpKind::SCALAR));
    EXPECT_EQ(info.arg_index, 2u);
    EXPECT_EQ(info.scalar_value, payload.scalars[0]);
    EXPECT_FALSE(
        next_dump_arg_for_task<3>(
            0, descriptor, payload, LaneMask::all(), ArgsDumpStage::BEFORE_DISPATCH, get_bin, &cursor, &info
        )
    );
    dump_args_for_task<3>(0, descriptor, payload, LaneMask::all(), ArgsDumpStage::BEFORE_DISPATCH, get_bin);
    ASSERT_EQ(meta[0]->count, 2u);
    EXPECT_EQ(meta[0]->records[0].arg_index, 0u);
    EXPECT_EQ(meta[0]->records[1].scalar_value, payload.scalars[0]);
    EXPECT_EQ(meta[0]->records[1].func_ids[2], 9);
    cursor = 0;
    ASSERT_TRUE(
        next_dump_arg_for_task<3>(
            0, descriptor, payload, LaneMask::all(), ArgsDumpStage::AFTER_COMPLETION, get_bin, &cursor, &info
        )
    );
    EXPECT_EQ(info.arg_index, 1u);
    EXPECT_FALSE(
        next_dump_arg_for_task<3>(
            0, descriptor, payload, LaneMask::all(), ArgsDumpStage::AFTER_COMPLETION, get_bin, &cursor, &info
        )
    );
}

TEST_F(ResumableDumpTest, IteratorKeepsPartialMasksTypedScalarAndAmbiguityFlags) {
    dump_args_init(2, DumpArgsLevel::PARTIAL);
    const ArgDirection directions[] = {ArgDirection::IN, ArgDirection::OUT};
    const auto callable = make_callable<CORE_MAX_TENSOR_ARGS>(directions, 2, nullptr, 0);
    IteratorDescriptor descriptor{simpler::ut::test_task_id(43), {7, 8, 9}};
    IteratorPayload payload;
    auto get_bin = [&](int32_t) {
        return reinterpret_cast<uint64_t>(callable.data());
    };
    ArgsDumpTaskMetadata metadata{};
    uint32_t cursor = 0;
    ArgsDumpInfo info{};
    EXPECT_FALSE(
        next_dump_arg_for_task<3>(
            0, descriptor, payload, LaneMask::all(), ArgsDumpStage::BEFORE_DISPATCH, get_bin, &cursor, &info, &metadata
        )
    );
    metadata.dump_arg_mask = ArgsDumpArgMask{1} << 2;
    metadata.dump_arg_flags = ArgsDumpArgMask{1} << 2;
    metadata.scalar_dtypes[0] = static_cast<uint8_t>(DataType::INT64);
    ASSERT_TRUE(
        next_dump_arg_for_task<3>(
            0, descriptor, payload, LaneMask::all(), ArgsDumpStage::BEFORE_DISPATCH, get_bin, &cursor, &info, &metadata
        )
    );
    EXPECT_EQ(info.arg_index, 2u);
    EXPECT_EQ(info.scalar_value, payload.scalars[0]);
    EXPECT_EQ(info.dtype, static_cast<uint8_t>(DataType::INT64));
    EXPECT_EQ(info.flags, ARGS_DUMP_RECORD_FLAG_ARG_INDEX_AMBIGUOUS);
    EXPECT_FALSE(
        next_dump_arg_for_task<3>(
            0, descriptor, payload, LaneMask::all(), ArgsDumpStage::BEFORE_DISPATCH, get_bin, &cursor, &info, &metadata
        )
    );
}
}  // namespace
