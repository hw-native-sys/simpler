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
#include <memory>
#include <thread>
#include <vector>

#include "callable.h"
#include "host/kernel_static_config.h"
#include "tensormap_and_ringbuffer/kernel_execution_inputs.h"
#include "worker/tmr_kernel_invocation.h"

namespace {
using namespace simpler::tmr;

TEST(KernelStaticConfig, OwnsPackedInputAndRejectsReplacement) {
    CallConfig input;
    input.aicpu_thread_num = 0;
    input.runtime_env.ring_task_window[0] = 32;
    std::array<uint8_t, sizeof(CallConfig) + 1> packed{};
    std::memcpy(packed.data() + 1, &input, sizeof(input));
    KernelStaticConfig config;
    ASSERT_EQ(config.initialize(reinterpret_cast<const CallConfig *>(packed.data() + 1), 9, true), 0);
    std::memset(packed.data(), 0, packed.size());
    uint64_t saved_window = 0;
    std::memcpy(&saved_window, &config.request().runtime_env.ring_task_window[0], sizeof(saved_window));
    EXPECT_EQ(saved_window, 32u);
    EXPECT_EQ(config.request().aicpu_thread_num, 0);
    EXPECT_EQ(config.generation(), 9u);
    EXPECT_TRUE(config.serial_orch_sched());
    EXPECT_FALSE(config.frozen());
    EXPECT_EQ(config.freeze(), 0);
    input.aicpu_thread_num = 3;
    EXPECT_EQ(config.initialize(&input, 10, false), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_EQ(config.request().aicpu_thread_num, 0);
    EXPECT_EQ(config.generation(), 9u);
}

TEST(KernelStaticConfig, RejectsInvalidAndUnsupportedRequestsWithoutInitialization) {
    KernelStaticConfig config;
    EXPECT_EQ(config.freeze(), PTO_RUNTIME_ERR_INVALID_STATE);
    CallConfig input;
    input.aicpu_thread_num = 1;
    EXPECT_EQ(config.initialize(&input, 7, false), PTO_RUNTIME_ERR_INTERNAL);
    input.aicpu_thread_num = -1;
    EXPECT_EQ(config.initialize(&input, 7, false), PTO_RUNTIME_ERR_INTERNAL);
    input.aicpu_thread_num = PLATFORM_MAX_AICPU_THREADS + 1;
    EXPECT_EQ(config.initialize(&input, 7, false), PTO_RUNTIME_ERR_INTERNAL);
    input.aicpu_thread_num = 0;
    input.enable_dump_args = 1;
    EXPECT_EQ(config.initialize(&input, 7, false), PTO_RUNTIME_ERR_UNSUPPORTED);
    input.enable_dump_args = 0;
    EXPECT_EQ(config.initialize(&input, 0, false), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_FALSE(config.initialized());
    EXPECT_EQ(config.initialize(&input, 7, false), 0);
}

class KernelExecutionInputsTest : public ::testing::Test {
protected:
    void SetUp() override {
        uint64_t windows[CHIP_MAX_RING_DEPTH] = {16, 16, 16, 16};
        uint64_t heaps[CHIP_MAX_RING_DEPTH] = {1024, 1024, 1024, 1024};
        int32_t deps[CHIP_MAX_RING_DEPTH] = {4, 4, 4, 4};
        layout = runtime_reserve_layout(arena, windows, heaps, deps);
        ASSERT_NE(arena.commit(), nullptr);
        const size_t sm_size = SharedMemoryHandle::calculate_size_per_ring(windows);
        const size_t sm_offset = sm.reserve(sm_size, CHIP_ALIGN_SIZE);
        ASSERT_NE(sm.commit(), nullptr);
        void *sm_base = sm.region_ptr(sm_offset);
        heap.resize(4096);
        ASSERT_NE(
            runtime_init_data_from_layout(arena, layout, MODE_EXECUTE, sm_base, sm_size, heap.data(), heaps), nullptr
        );
        binding = {
            {reinterpret_cast<uint64_t>(&identity), 13},
            resident.get(),
            {sm_base, sm_size, sm_size},
            {arena.base(), layout.offsets.arena_size, layout.offsets.arena_size},
            layout.offsets.off_runtime
        };
    }

    ChipStorageTaskArgs arguments(uint64_t scalar) {
        ChipStorageTaskArgs args;
        const uint32_t shape[] = {2, 4};
        args.add_tensor(make_tensor_external(data.data(), shape, 2, DataType::FLOAT32, AddressSpace::DEVICE));
        args.add_scalar(scalar);
        return args;
    }

    std::unique_ptr<Runtime> resident{std::make_unique<Runtime>()};
    std::unique_ptr<KernelInvocationState> state{std::make_unique<KernelInvocationState>()};
    DeviceArena arena;
    DeviceArena sm;
    RuntimeArenaLayout layout{};
    std::vector<char> heap;
    std::array<float, 8> data{};
    uint64_t identity{0};
    KernelBindingView binding{};
    const PreparedInvocationView callable{3, 1, 1, 17};
};

TEST_F(KernelExecutionInputsTest, ConfigAndOrchestrationShareStableBorrowedArguments) {
    auto args = arguments(19);
    TmrEncodingCandidate packet;
    TmrEncodingCache cache;
    ASSERT_EQ(encode_tmr_invocation(args, callable, binding.identity, cache, &packet), InvocationStatus::Ok);
    ASSERT_EQ(state->admit(packet.packet(), {callable, {}}, binding), InvocationStatus::Ok);
    ChipTaskArgs converted;
    auto config = +[](const ChipTaskArgs &entry) {
        EXPECT_EQ(entry.scalar(0), 19u);
        EXPECT_EQ(entry.tensor(0).ref().shapes[1], 4u);
        return OrchestrationConfig{2};
    };
    ASSERT_TRUE(configure_orchestration_args(state->inputs(), converted, config));
    const auto *tensor = &converted.tensor(0).ref();
    EXPECT_EQ(tensor, &state->inputs().args->tensor(0));
    EXPECT_EQ(state->inputs().callable_id, callable.callable_id);
    EXPECT_EQ(state->inputs().sm, binding.sm.base);
    EXPECT_EQ(state->inputs().arena, binding.arena.base);
    EXPECT_EQ(resident->get_active_callable_id(), -1);
    EXPECT_EQ(resident->get_orch_args().tensor_count(), 0);
    auto orchestration = +[](const ChipTaskArgs &entry) {
        auto *values = reinterpret_cast<float *>(entry.tensor(0).ref().buffer.addr);
        values[0] = static_cast<float>(entry.scalar(0));
    };
    orchestration(converted);
    EXPECT_EQ(data[0], 19.0f);
    EXPECT_EQ(&converted.tensor(0).ref(), tensor);
    EXPECT_EQ(state->admit(packet.packet(), {callable, {}}, binding), InvocationStatus::InvalidArgument);
    EXPECT_EQ(converted.scalar(0), 19u);
    converted.reset();
    state->clear();
    EXPECT_FALSE(state->active());
}

TEST_F(KernelExecutionInputsTest, FailureDoesNotPublishOrModifyPreviousInputs) {
    auto args = arguments(23);
    TmrEncodingCandidate packet;
    TmrEncodingCache cache;
    ASSERT_EQ(encode_tmr_invocation(args, callable, binding.identity, cache, &packet), InvocationStatus::Ok);
    auto invalid = binding;
    invalid.arena.capacity = invalid.arena.required_bytes - 1;
    EXPECT_EQ(state->admit(packet.packet(), {callable, {}}, invalid), InvocationStatus::InvalidBinding);
    invalid = binding;
    invalid.runtime_offset = invalid.arena.capacity;
    EXPECT_EQ(state->admit(packet.packet(), {callable, {}}, invalid), InvocationStatus::InvalidBinding);
    invalid = binding;
    invalid.sm.base = static_cast<char *>(invalid.sm.base) + 1;
    EXPECT_EQ(state->admit(packet.packet(), {callable, {}}, invalid), InvocationStatus::InvalidBinding);
    invalid = binding;
    invalid.identity.context_generation++;
    EXPECT_EQ(state->admit(packet.packet(), {callable, {}}, invalid), InvocationStatus::InvalidBinding);
    auto stale = callable;
    stale.slot_generation++;
    EXPECT_EQ(state->admit(packet.packet(), {stale, {}}, binding), InvocationStatus::StaleCallable);
    EXPECT_FALSE(state->active());
    EXPECT_EQ(state->inputs().args, nullptr);
    EXPECT_EQ(state->admit(packet.packet(), {callable, {}}, binding), InvocationStatus::Ok);
    const auto *storage = state->inputs().args;
    EXPECT_EQ(state->admit({}, {callable, {}}, binding), InvocationStatus::InvalidArgument);
    EXPECT_EQ(state->inputs().args, storage);
    EXPECT_EQ(storage->scalar(0), 23u);
}

TEST_F(KernelExecutionInputsTest, AlternateCallablesNeverRewriteResidentFunctionTable) {
    const uint8_t binary[] = {1, 2, 3, 4};
    auto a = make_callable<CORE_MAX_TENSOR_ARGS>(nullptr, 0, binary, sizeof(binary));
    auto b = make_callable<CORE_MAX_TENSOR_ARGS>(nullptr, 0, binary, sizeof(binary));
    const std::array<uint64_t, 1> table_a{reinterpret_cast<uint64_t>(a.data())};
    const std::array<uint64_t, 1> table_b{reinterpret_cast<uint64_t>(b.data())};
    const auto original = resident->dev;
    for (const auto *table : {&table_a, &table_b, &table_a}) {
        TmrEncodingCandidate packet;
        TmrEncodingCache cache;
        auto args = arguments(table == &table_a ? 31 : 37);
        ASSERT_EQ(encode_tmr_invocation(args, callable, binding.identity, cache, &packet), InvocationStatus::Ok);
        ASSERT_EQ(
            state->admit(packet.packet(), {callable, {table->data(), table->size()}}, binding), InvocationStatus::Ok
        );
        EXPECT_EQ(state->inputs().functions.lookup(0), table->front());
        EXPECT_EQ(state->inputs().functions.lookup(-1), 0u);
        EXPECT_EQ(state->inputs().functions.lookup(1), 0u);
        EXPECT_EQ(std::memcmp(&resident->dev, &original, sizeof(original)), 0);
        state->clear();
    }
}

TEST_F(KernelExecutionInputsTest, EmptyTensorStorageMatchesTheProgramConversion) {
    const uint32_t shapes[][3] = {{0, 1, 1}, {0, 8, 1}, {8, 0, 1}, {2, 0, 3}};
    const uint32_t ranks[] = {1, 2, 2, 3};
    for (size_t i = 0; i < std::size(ranks); ++i) {
        auto args = arguments(43);
        args.tensor(0) = make_tensor_external(nullptr, shapes[i], ranks[i], DataType::FLOAT32, AddressSpace::DEVICE);
        TmrEncodingCandidate packet;
        TmrEncodingCache cache;
        ASSERT_EQ(encode_tmr_invocation(args, callable, binding.identity, cache, &packet), InvocationStatus::Ok);
        ASSERT_EQ(state->admit(packet.packet(), {callable, {}}, binding), InvocationStatus::Ok);
        ChipTaskArgs converted;
        ASSERT_TRUE(configure_orchestration_args(state->inputs(), converted, nullptr));
        const auto program_tensor = Tensor::from_boundary(args.tensor(0));
        EXPECT_EQ(converted.tensor(0).ref().numel(), 0u);
        EXPECT_EQ(converted.tensor(0).ref().buffer.addr, 0u);
        EXPECT_EQ(converted.tensor(0).ref().extent_elem(), program_tensor.extent_elem());
        EXPECT_EQ(converted.scalar(0), 43u);
        converted.reset();
        state->clear();
    }
}

TEST_F(KernelExecutionInputsTest, RejectsInvalidFunctionTableBeforePublishingStorage) {
    auto args = arguments(47);
    TmrEncodingCandidate packet;
    TmrEncodingCache cache;
    ASSERT_EQ(encode_tmr_invocation(args, callable, binding.identity, cache, &packet), InvocationStatus::Ok);
    const uint64_t table[] = {0};
    const CallableTableView invalid[] = {
        {nullptr, 1},
        {table, RUNTIME_MAX_FUNC_ID + 1},
        {reinterpret_cast<const uint64_t *>(reinterpret_cast<const char *>(table) + 1), 1}
    };
    for (const auto &functions : invalid) {
        EXPECT_EQ(state->admit(packet.packet(), {callable, functions}, binding), InvocationStatus::InvalidBinding);
        EXPECT_FALSE(state->active());
        EXPECT_EQ(state->inputs().args, nullptr);
    }
    EXPECT_EQ(state->admit(packet.packet(), {callable, {table, 1}}, binding), InvocationStatus::Ok);
    EXPECT_EQ(state->inputs().args->scalar(0), 47u);
}

TEST_F(KernelExecutionInputsTest, ProgramWrapperBorrowsOriginalStorage) {
    resident->dev.active_callable_id_ = 2;
    resident->dev.func_id_to_addr_[5] = 0x1200;
    resident->dev.orch_args_storage_.scalar_count_ = 1;
    resident->dev.orch_args_storage_.scalars_[0] = 29;
    const auto input = program_execution_inputs(*resident);
    EXPECT_EQ(input.callable_id, 2);
    EXPECT_EQ(input.args, &resident->dev.orch_args_storage_);
    EXPECT_EQ(input.args->scalar(0), 29u);
    EXPECT_EQ(input.functions.lookup(5), 0x1200u);
}

}  // namespace
