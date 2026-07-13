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

#include <cstdint>
#include <cstring>

#include "aicpu/args_dump_aicpu.h"

// Fixture for args-dump tests: the args-dump mask pool keys entries on the
// full 64-bit task_id and must be agnostic to how a runtime partitions that id
// into (ring_id, slot) — any task_id can enter the table.
class ArgsDumpTest : public ::testing::Test {
protected:
    void SetUp() override { set_dump_args_enabled(true); }
    void TearDown() override { set_dump_args_enabled(false); }
};

// Basic enable/disable toggle and dump-base propagation — the foundation the
// mask pool sits on.
TEST_F(ArgsDumpTest, EnableDisableAndDumpBaseRoundTrip) {
    set_dump_args_enabled(false);
    EXPECT_FALSE(is_dump_args_enabled());

    set_dump_args_enabled(true);
    EXPECT_TRUE(is_dump_args_enabled());

    set_dump_args_enabled(false);
    EXPECT_FALSE(is_dump_args_enabled());

    set_platform_dump_base(0xDEADBEEF42ULL);
    EXPECT_EQ(get_platform_dump_base(), 0xDEADBEEF42ULL);

    set_platform_dump_base(0);
    EXPECT_EQ(get_platform_dump_base(), 0ULL);
}

// A task_id whose high 32 bits exceed any runtime's ring depth must still round
// trip — this is the coupling the pool must not have. Exercises both the mask
// table and the scalar-dtype table.
TEST_F(ArgsDumpTest, HighRingTaskIdRoundTrips) {
    const uint64_t task_id = (uint64_t{7} << 32) | 123u;
    const ArgsDumpArgMask mask = 0b1011;
    const ArgsDumpArgMask flags = 0b0010;
    const uint8_t dtypes[3] = {2, 5, 11};

    set_dump_args_task_mask(task_id, mask, flags);
    set_dump_args_task_scalar_dtypes(task_id, 3, dtypes);

    ArgsDumpArgMask got_mask = ARGS_DUMP_ARG_MASK_NONE;
    ArgsDumpArgMask got_flags = ARGS_DUMP_ARG_MASK_NONE;
    get_dump_args_task_masks(task_id, &got_mask, &got_flags);
    EXPECT_EQ(got_mask, mask);
    EXPECT_EQ(got_flags, flags);

    uint32_t got_count = 0;
    uint8_t got_dtypes[32] = {};
    ASSERT_TRUE(get_dump_args_task_scalar_dtypes(task_id, &got_count, got_dtypes));
    EXPECT_EQ(got_count, 3u);
    EXPECT_EQ(got_dtypes[0], 2);
    EXPECT_EQ(got_dtypes[1], 5);
    EXPECT_EQ(got_dtypes[2], 11);
}

// Two distinct task_ids must keep independent entries (open-addressed probing
// resolves any hash collision to separate slots).
TEST_F(ArgsDumpTest, DistinctTaskIdsDoNotAlias) {
    const uint64_t id_a = (uint64_t{2} << 32) | 100u;
    const uint64_t id_b = (uint64_t{3} << 32) | 200u;

    set_dump_args_task_mask(id_a, 0b0001, ARGS_DUMP_ARG_MASK_NONE);
    set_dump_args_task_mask(id_b, 0b1000, ARGS_DUMP_ARG_MASK_NONE);

    ArgsDumpArgMask mask_a = ARGS_DUMP_ARG_MASK_NONE;
    ArgsDumpArgMask mask_b = ARGS_DUMP_ARG_MASK_NONE;
    get_dump_args_task_masks(id_a, &mask_a, nullptr);
    get_dump_args_task_masks(id_b, &mask_b, nullptr);

    EXPECT_EQ(mask_a, static_cast<ArgsDumpArgMask>(0b0001));
    EXPECT_EQ(mask_b, static_cast<ArgsDumpArgMask>(0b1000));
}

// An unset task_id reads back as "no mask" — the empty-slot path.
TEST_F(ArgsDumpTest, UnknownTaskIdReadsNone) {
    ArgsDumpArgMask got_mask = 0xFFFF;
    ArgsDumpArgMask got_flags = 0xFFFF;
    get_dump_args_task_masks((uint64_t{5} << 32) | 42u, &got_mask, &got_flags);
    EXPECT_EQ(got_mask, ARGS_DUMP_ARG_MASK_NONE);
    EXPECT_EQ(got_flags, ARGS_DUMP_ARG_MASK_NONE);
}

// End-to-end dump-record correctness: set up the shared-memory dump buffer,
// record a tensor and a scalar via dump_arg_record, then read back the
// ArgsDumpRecord entries and the arena payload to verify every field.
TEST_F(ArgsDumpTest, DumpRecordCorrectness) {
    constexpr size_t kDumpMemSize = sizeof(DumpDataHeader) + sizeof(DumpBufferState);
    alignas(64) uint8_t dump_mem[kDumpMemSize] = {};
    alignas(64) DumpMetaBuffer meta_buf{};
    constexpr size_t kArenaSize = 4096;
    alignas(uint64_t) uint8_t arena[kArenaSize] = {};
    uint64_t src_data[4] = {10, 20, 30, 40};

    DumpDataHeader *hdr = get_dump_header(dump_mem);
    hdr->magic = ARGS_DUMP_MAGIC;
    hdr->dump_args_level = static_cast<uint32_t>(DumpArgsLevel::FULL);
    hdr->num_dump_threads = 1;

    DumpBufferState *state = get_dump_buffer_state(dump_mem, 0);
    state->free_queue.buffer_ptrs[0] = reinterpret_cast<uint64_t>(&meta_buf);
    state->free_queue.tail = 1;
    state->arena_base = reinterpret_cast<uint64_t>(arena);
    state->arena_size = kArenaSize;
    state->arena_write_offset = 0;

    set_platform_dump_base(reinterpret_cast<uint64_t>(dump_mem));
    dump_args_init(1);

    // Record a 1-D UINT64 tensor (4 elements, contiguous).
    ArgsDumpInfo tinfo = {};
    tinfo.task_id = 0xDEAD;
    tinfo.role = ArgsDumpRole::INPUT;
    tinfo.stage = ArgsDumpStage::BEFORE_DISPATCH;
    tinfo.dtype = static_cast<uint8_t>(DataType::UINT64);
    tinfo.ndims = 1;
    tinfo.arg_index = 0;
    tinfo.buffer_addr = reinterpret_cast<uint64_t>(src_data);
    tinfo.shapes[0] = 4;
    tinfo.strides[0] = 1;
    tinfo.start_offset = 0;
    tinfo.kind = static_cast<uint8_t>(ArgsDumpKind::TENSOR);
    tinfo.func_count = 1;
    tinfo.func_ids[0] = 42;

    ASSERT_EQ(dump_arg_record(0, tinfo), 0);

    const ArgsDumpRecord &trec = meta_buf.records[0];
    EXPECT_EQ(trec.task_id, 0xDEAD);
    EXPECT_EQ(trec.role, static_cast<uint8_t>(ArgsDumpRole::INPUT));
    EXPECT_EQ(trec.stage, static_cast<uint8_t>(ArgsDumpStage::BEFORE_DISPATCH));
    EXPECT_EQ(trec.dtype, static_cast<uint8_t>(DataType::UINT64));
    EXPECT_EQ(trec.ndims, 1);
    EXPECT_EQ(trec.arg_index, 0u);
    EXPECT_EQ(trec.kind, static_cast<uint8_t>(ArgsDumpKind::TENSOR));
    EXPECT_EQ(trec.truncated, 0);
    EXPECT_EQ(trec.is_contiguous, 1);
    EXPECT_EQ(trec.payload_size, 32u);  // 4 * sizeof(uint64_t)
    EXPECT_EQ(trec.shapes[0], 4u);
    EXPECT_EQ(trec.strides[0], 1u);
    EXPECT_EQ(trec.func_count, 1);
    EXPECT_EQ(trec.func_ids[0], 42u);

    uint64_t arena_vals[4];
    memcpy(arena_vals, arena, sizeof(arena_vals));
    EXPECT_EQ(arena_vals[0], 10u);
    EXPECT_EQ(arena_vals[1], 20u);
    EXPECT_EQ(arena_vals[2], 30u);
    EXPECT_EQ(arena_vals[3], 40u);

    // Record a scalar (no arena copy; scalar_value carried inline in the record).
    ArgsDumpInfo sinfo = {};
    sinfo.task_id = 0xBEEF;
    sinfo.role = ArgsDumpRole::INPUT;
    sinfo.stage = ArgsDumpStage::BEFORE_DISPATCH;
    sinfo.dtype = static_cast<uint8_t>(DataType::UINT64);
    sinfo.arg_index = 1;
    sinfo.scalar_value = 0x12345678;
    sinfo.kind = static_cast<uint8_t>(ArgsDumpKind::SCALAR);
    sinfo.func_count = 1;
    sinfo.func_ids[0] = 7;

    ASSERT_EQ(dump_arg_record(0, sinfo), 0);

    const ArgsDumpRecord &srec = meta_buf.records[1];
    EXPECT_EQ(srec.task_id, 0xBEEF);
    EXPECT_EQ(srec.kind, static_cast<uint8_t>(ArgsDumpKind::SCALAR));
    EXPECT_EQ(srec.scalar_value, 0x12345678u);
    EXPECT_EQ(srec.payload_size, 0u);
    EXPECT_EQ(srec.dtype, static_cast<uint8_t>(DataType::UINT64));
    EXPECT_EQ(srec.func_ids[0], 7u);
    EXPECT_EQ(meta_buf.count, 2u);

    set_platform_dump_base(0);
}
