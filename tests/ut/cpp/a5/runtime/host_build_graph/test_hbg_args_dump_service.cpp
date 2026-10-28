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
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include "args_dump_service.h"
#include "aicpu/device_run_result_base_aicpu.h"
#include "runtime_types.h"

namespace {
class ResidentDumpServiceTest : public ::testing::Test {
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
            metadata[lane] = std::make_unique<DumpMetaBuffer>();
            states[lane] = get_dump_buffer_state(shared.get(), lane);
            states[lane]->free_queue.buffer_ptrs[0] = reinterpret_cast<uint64_t>(metadata[lane].get());
            states[lane]->free_queue.tail = 1;
            arenas[lane].resize(256 * 1024);
            states[lane]->arena_base = reinterpret_cast<uint64_t>(arenas[lane].data());
            states[lane]->arena_size = arenas[lane].size();
        }
        input.resize(40000);
        for (uint32_t index = 0; index < input.size(); ++index)
            input[index] = index + 100;
        output.resize(input.size(), 7);
        const ArgDirection signature[] = {ArgDirection::IN, ArgDirection::OUT};
        const uint8_t binary = 0;
        callable_storage = make_callable<CORE_MAX_TENSOR_ARGS>(signature, 2, &binary, 1);
        callables[0] = reinterpret_cast<uint64_t>(callable_storage.data());
        for (uint32_t id = 0; id < storage.size(); ++id) {
            auto &entry = storage[id];
            entry.task.task_id = TaskId::make_global(id);
            entry.task.kernel_id[0] = -1;
            entry.task.kernel_id[1] = 0;
            entry.task.kernel_id[2] = -1;
            entry.payload.tensor_count = 2;
            entry.payload.scalar_count = 0;
            entry.payload.bind_regions(tensors[id].data(), nullptr, nullptr);
            for (uint32_t slot = 0; slot < 2; ++slot) {
                auto &tensor = tensors[id][slot];
                tensor.buffer.addr = reinterpret_cast<uint64_t>(slot == 0 ? input.data() : output.data());
                tensor.buffer.size = input.size() * sizeof(uint32_t);
                tensor.dtype = DataType::UINT32;
                tensor.ndims = 1;
                tensor.shapes[0] = input.size();
                tensor.strides[0] = 1;
                tensor.is_contiguous = true;
            }
        }
        graph = {reinterpret_cast<uint64_t>(storage.data()), 0, storage.size(), storage.size() - 1};
        ASSERT_TRUE(scheduler_args_dump_plan_layout(2, 2, 2, true, &layout));
        ASSERT_TRUE(scheduler_args_dump_init_image(image.data(), image.size(), layout, epoch, DumpArgsLevel::FULL));
        ASSERT_TRUE(scheduler_args_dump_select_task(image.data(), 0, 2));
        ASSERT_TRUE(scheduler_args_dump_select_task(image.data(), 1, 2));
        set_dump_args_enabled(true);
        set_platform_dump_base(reinterpret_cast<uint64_t>(shared.get()));
        set_platform_run_result(0, epoch);
        dump_args_init(2, DumpArgsLevel::FULL);
        ASSERT_TRUE(initialize_resident_args_dump_outputs(image.data(), graph, table_address(), callables.size()));
        requests = reinterpret_cast<SchedulerArgsDumpRequest *>(image.data() + layout.requests_offset);
        acknowledgements = reinterpret_cast<SchedulerArgsDumpAck *>(image.data() + layout.acknowledgements_offset);
    }
    void TearDown() override {
        set_platform_dump_args_run_terminal(0);
        set_platform_dump_base(0);
        set_platform_run_result(0, 0);
        set_dump_args_enabled(false);
    }
    uint64_t table_address() const { return reinterpret_cast<uint64_t>(callables.data()); }
    void request(uint32_t worker, uint32_t task_id, uint64_t cursor = 0, uint64_t byte_cursor = 0) {
        auto *task = scheduler_args_dump_task(image.data(), task_id);
        if (task->before_masks == 0)
            ASSERT_EQ(
                scheduler_args_dump_arrive(task, ArgsDumpStage::BEFORE_DISPATCH, SubtaskSlot::AIV0),
                SchedulerArgsDumpArrival::Publisher
            );
        task->publisher_before = scheduler_args_dump_publisher(worker, 19);
        auto &request = requests[worker];
        request.run_epoch = epoch;
        request.task_id = task_id;
        request.stage = static_cast<uint64_t>(ArgsDumpStage::BEFORE_DISPATCH);
        request.generation = 19;
        request.arg_index = cursor;
        request.byte_cursor = byte_cursor;
        ++request.tick;
    }
    struct Free {
        void operator()(uint8_t *p) const { std::free(p); }
    };
    static constexpr uint64_t epoch = UINT64_C(0x1234567800000011);
    std::unique_ptr<uint8_t, Free> shared;
    DumpDataHeader *header{};
    std::array<DumpBufferState *, 2> states{};
    std::array<std::unique_ptr<DumpMetaBuffer>, 2> metadata;
    std::array<std::vector<uint8_t>, 2> arenas;
    std::array<ChipTaskStorage, 2> storage{};
    std::array<std::array<simpler::hbg::Tensor, 2>, 2> tensors{};
    std::vector<uint32_t> input, output;
    std::vector<uint8_t> callable_storage;
    std::array<uint64_t, 1> callables{};
    SchedulerGraphView graph{};
    SchedulerArgsDumpLayout layout{};
    alignas(64) std::array<uint8_t, 4096> image{};
    SchedulerArgsDumpRequest *requests{};
    SchedulerArgsDumpAck *acknowledgements{};
};

TEST_F(ResidentDumpServiceTest, LargeInputYieldsAfterOneChunkAndCommitsOnlyAfterAllBytes) {
    ResidentArgsDumpService service;
    ASSERT_TRUE(service.initialize(image.data(), graph, table_address(), callables.size(), 0, 1));
    request(0, 0);
    ASSERT_TRUE(service.step());
    EXPECT_TRUE(service.has_open_record());
    EXPECT_EQ(acknowledgements[0].byte_cursor, 65536);
    EXPECT_EQ(metadata[0]->count, 0);
    request(0, 0, 0, acknowledgements[0].byte_cursor);
    ASSERT_TRUE(service.step());
    EXPECT_EQ(acknowledgements[0].byte_cursor, 131072);
    EXPECT_EQ(metadata[0]->count, 0);
    request(0, 0, 0, acknowledgements[0].byte_cursor);
    ASSERT_TRUE(service.step());
    EXPECT_FALSE(service.has_open_record());
    EXPECT_EQ(metadata[0]->count, 1);
    EXPECT_EQ(acknowledgements[0].arg_index, 1);
    EXPECT_EQ(std::memcmp(arenas[0].data(), input.data(), input.size() * sizeof(uint32_t)), 0);
    request(0, 0, 1);
    ASSERT_TRUE(service.step());
    EXPECT_EQ(scheduler_args_dump_ack_status(acknowledgements[0]), SchedulerArgsDumpStatus::Sealed);
    EXPECT_EQ(header->queue_tails[0], 1);
    service.stop();
}

TEST_F(ResidentDumpServiceTest, AnotherWorkerRetriesWithoutLossWhileTheLaneOwnsALargeRecord) {
    ResidentArgsDumpService service;
    ASSERT_TRUE(service.initialize(image.data(), graph, table_address(), callables.size(), 0, 1));
    request(0, 0);
    ASSERT_TRUE(service.step());
    request(1, 1);
    EXPECT_FALSE(service.step());
    EXPECT_EQ(scheduler_args_dump_ack_status(acknowledgements[1]), SchedulerArgsDumpStatus::WouldBlock);
    EXPECT_EQ(states[0]->run_dropped_record_count, 0);
    EXPECT_EQ(metadata[0]->count, 0);
    service.stop();
    EXPECT_FALSE(service.has_open_record());
    EXPECT_EQ(states[0]->arena_write_offset, 0);
    EXPECT_EQ(states[0]->run_dropped_record_count, 1);
    EXPECT_EQ(scheduler_args_dump_ack_status(acknowledgements[0]), SchedulerArgsDumpStatus::Aborted);
    EXPECT_EQ(metadata[0]->count, 0);
    EXPECT_EQ(header->queue_tails[0], 0);
}

TEST_F(ResidentDumpServiceTest, AnOpenCopyCannotBeReboundOrDetachedFromItsOwner) {
    ResidentArgsDumpService service;
    ASSERT_TRUE(service.initialize(image.data(), graph, table_address(), callables.size(), 0, 1));
    request(0, 0);
    ASSERT_TRUE(service.step());
    ASSERT_TRUE(service.has_open_record());
    EXPECT_FALSE(service.initialize(nullptr, graph, table_address(), callables.size(), 0, 1));
    EXPECT_FALSE(service.initialize(image.data(), graph, table_address(), callables.size(), 0, 1));
    EXPECT_TRUE(service.has_open_record());
    EXPECT_EQ(metadata[0]->count, 0);
    service.stop();
    EXPECT_FALSE(service.has_open_record());
    EXPECT_FALSE(dump_arg_has_open_record(0));
    EXPECT_EQ(states[0]->arena_write_offset, 0);
    EXPECT_EQ(states[0]->run_dropped_record_count, 1);
}

TEST_F(ResidentDumpServiceTest, RefusesStaleFullEpochAndWrongGenerationWithoutOpeningARecord) {
    ResidentArgsDumpService service;
    ASSERT_TRUE(service.initialize(image.data(), graph, table_address(), callables.size(), 0, 1));
    request(0, 0);
    requests[0].run_epoch ^= UINT64_C(1) << 40;
    EXPECT_FALSE(service.step());
    EXPECT_EQ(scheduler_args_dump_ack_status(acknowledgements[0]), SchedulerArgsDumpStatus::Aborted);
    EXPECT_FALSE(service.has_open_record());
    request(0, 0);
    ++requests[0].generation;
    EXPECT_FALSE(service.step());
    EXPECT_FALSE(service.has_open_record());
    EXPECT_EQ(metadata[0]->count, 0);
    EXPECT_EQ(states[0]->run_dropped_record_count, 0);
    service.stop();
}

TEST_F(ResidentDumpServiceTest, AnotherParticipantPublishesWhileTheLargeCopyRemainsOpen) {
    ResidentArgsDumpService first;
    ResidentArgsDumpService second;
    ASSERT_TRUE(first.initialize(image.data(), graph, table_address(), callables.size(), 0, 2));
    ASSERT_TRUE(second.initialize(image.data(), graph, table_address(), callables.size(), 1, 2));
    tensors[1][0].shapes[0] = 1;
    request(0, 0);
    ASSERT_TRUE(first.step());
    ASSERT_TRUE(first.has_open_record());
    request(1, 1);
    ASSERT_TRUE(second.step());
    EXPECT_EQ(metadata[0]->count, 0);
    EXPECT_EQ(metadata[1]->count, 1);
    EXPECT_EQ(metadata[1]->records[0].payload_size, sizeof(uint32_t));
    EXPECT_EQ(std::memcmp(arenas[1].data(), input.data(), sizeof(uint32_t)), 0);
    request(1, 1, 1);
    ASSERT_TRUE(second.step());
    EXPECT_EQ(header->queue_tails[1], 1);
    EXPECT_EQ(scheduler_args_dump_ack_status(acknowledgements[1]), SchedulerArgsDumpStatus::Sealed);
    EXPECT_TRUE(first.has_open_record());
    EXPECT_EQ(metadata[0]->count, 0);
    first.stop();
    second.stop();
}

TEST_F(ResidentDumpServiceTest, DiscontinuousAcknowledgedCursorAbortsWithoutPublishingAPartialRecord) {
    ResidentArgsDumpService service;
    ASSERT_TRUE(service.initialize(image.data(), graph, table_address(), callables.size(), 0, 1));
    request(0, 0);
    ASSERT_TRUE(service.step());
    ASSERT_TRUE(service.has_open_record());
    request(0, 0, 0, acknowledgements[0].byte_cursor + 1);
    EXPECT_FALSE(service.step());
    EXPECT_FALSE(service.has_open_record());
    EXPECT_EQ(metadata[0]->count, 0);
    EXPECT_EQ(states[0]->arena_write_offset, 0);
    EXPECT_EQ(states[0]->run_dropped_record_count, 1);
    EXPECT_EQ(scheduler_args_dump_ack_status(acknowledgements[0]), SchedulerArgsDumpStatus::Aborted);
    EXPECT_EQ(header->queue_tails[0], 0);
    service.stop();
    EXPECT_EQ(states[0]->run_dropped_record_count, 1);
}

TEST_F(ResidentDumpServiceTest, OutputVisibilityMaskUsesTheRealSignatureAndHybridPayloadSelection) {
    EXPECT_EQ(scheduler_args_dump_task(image.data(), 0)->output_tensor_mask, 2);
    auto *task = scheduler_args_dump_task(image.data(), 0);
    task->output_tensor_mask = 0;
    auto *config = reinterpret_cast<SchedulerArgsDumpHeader *>(image.data());
    config->dump_level = static_cast<uint64_t>(DumpArgsLevel::HYBRID);
    dump_args_init(2, DumpArgsLevel::HYBRID);
    ASSERT_TRUE(initialize_resident_args_dump_outputs(image.data(), graph, table_address(), callables.size()));
    EXPECT_EQ(task->output_tensor_mask, 0);
    storage[0].payload.dump_metadata.dump_arg_mask = 2;
    ASSERT_TRUE(initialize_resident_args_dump_outputs(image.data(), graph, table_address(), callables.size()));
    EXPECT_EQ(task->output_tensor_mask, 2);
}
}  // namespace
