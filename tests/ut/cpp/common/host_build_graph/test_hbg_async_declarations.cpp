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
#include <vector>

#include "graph_execution.h"
#include "graph_host_state.h"
#include "host_build_graph/host/graph_submit.h"
#include "host_build_graph/host/orchestrator.h"
#include "host_build_graph/shared_memory.h"
#include "utils/device_arena.h"

namespace {
constexpr AsyncUseDeclaration kCounter{AsyncUseDeclarationKind::COUNTER_PAYLOAD, 0, 1, UINT32_MAX, UINT32_MAX, 4};
constexpr AsyncUseDeclaration kConsumer{
    AsyncUseDeclarationKind::CONSUMER_CLEAN_READ, UINT32_MAX, 1, UINT32_MAX, UINT32_MAX, 0
};
constexpr AsyncUseDeclaration kCounterOnly{
    AsyncUseDeclarationKind::COUNTER_ONLY, 0, UINT32_MAX, UINT32_MAX, UINT32_MAX, 4
};

class HbgAsyncDeclarations : public ::testing::Test {
protected:
    DeviceArena sm_arena;
    SharedMemoryHandle *sm{nullptr};
    OrchestratorState orch{};
    GraphHostStatePtr state;
    std::vector<char> heap;
    std::vector<std::byte> definitions;
    std::array<int32_t, 32> signal_data{}, payload_data{}, other_signal{}, other_payload{};
    simpler::hbg::Tensor signals{}, payload{};
    GraphDefinitionArena arena{};

    void SetUp() override {
        sm = SharedMemoryHandle::create_and_init_default(sm_arena);
        ASSERT_NE(sm, nullptr);
        heap.resize(1024 * 1024);
        definitions.resize(128 * 1024);
        arena = {definitions.data(), definitions.size(), sizeof(GraphDefinitionHeader), GRAPH_DEFINITION_OBJECT_ALIGN};
        uint32_t shape[]{32};
        signals = simpler::hbg::make_tensor_external(signal_data.data(), shape, 1, DataType::INT32);
        payload = simpler::hbg::make_tensor_external(payload_data.data(), shape, 1, DataType::INT32);
        new_run();
    }

    void TearDown() override {
        if (auto *recording = active_graph_recording(&orch)) {
            const auto found = state->inflight.find(recording->graph_key);
            if (found != state->inflight.end()) orch.graph_abort(found->second.get());
        }
        orch.graph_host_state = nullptr;
        state.reset();
        sm_arena.release();
    }

    void new_run() {
        orch.graph_host_state = nullptr;
        state.reset();
        ASSERT_TRUE(orch.init(sm->sm_base, heap.data(), heap.size(), CHIP_DEFAULT_GRAPH_TASKS));
        state = make_graph_host_state(arena);
        ASSERT_NE(state, nullptr);
        orch.graph_host_state = state.get();
        orch.begin_scope();
    }

    TaskId submit(const simpler::hbg::Tensor &signal, const simpler::hbg::Tensor &data) {
        CoreTaskArgs args;
        args.add_no_dep(signal);
        args.add_no_dep(data);
        args.add_scalar(uint64_t{0x1234});
        MixedKernels kernels{};
        kernels.aiv0_kernel_id = 0;
        return orch.submit_task(kernels, args).task_id();
    }

    void boundary(GraphTaskArgs &args, const simpler::hbg::Tensor &signal, const simpler::hbg::Tensor &data) {
        args.add_no_dep(signal);
        args.add_no_dep(data);
    }

    TaskId record(uint64_t key, const AsyncUseDeclaration &declaration = kCounter) {
        GraphTaskArgs args;
        boundary(args, signals, payload);
        const auto graph = orch.graph_begin(key, args);
        EXPECT_TRUE(graph.recording);
        EXPECT_TRUE(orch.graph_prepare(graph.recording_handle, args));
        const auto task = submit(graph.params->tensor(0).ref(), graph.params->tensor(1).ref());
        EXPECT_TRUE(declare_async_use(&orch, task, declaration));
        EXPECT_TRUE(orch.graph_end());
        return task;
    }
};

TEST_F(HbgAsyncDeclarations, OrdinaryUsesAreLazyScopedAndIdempotent) {
    EXPECT_EQ(state->async_declarations, nullptr);
    const auto task = submit(signals, payload);
    ASSERT_TRUE(task.is_valid());
    ASSERT_TRUE(declare_async_use(&orch, task, kCounter));
    ASSERT_TRUE(declare_async_use(&orch, task, kCounter));
    ASSERT_TRUE(declare_async_use(&orch, task, kConsumer));
    const auto *uses = state->async_declarations->find(false, 0, task.local_id());
    ASSERT_NE(uses, nullptr);
    ASSERT_EQ(uses->size(), 2u);
    EXPECT_EQ((*uses)[0], kCounter);
    EXPECT_EQ((*uses)[1], kConsumer);
    EXPECT_EQ(state->async_declarations->find(true, 0, task.local_id()), nullptr);
}

TEST_F(HbgAsyncDeclarations, RejectsFutureForeignAndNonKernelTaskIdentities) {
    const auto task = submit(signals, payload);
    EXPECT_FALSE(declare_async_use(&orch, TaskId::make_global(task.local_id() + 1), kCounter));
    EXPECT_TRUE(orch.is_fatal());
    new_run();
    EXPECT_FALSE(declare_async_use(&orch, TaskId::make_sub_task(0, 0), kCounter));
    new_run();
    CoreTaskArgs args;
    args.add_no_dep(signals);
    args.add_no_dep(payload);
    const auto dummy = orch.submit_dummy_task(args).task_id();
    EXPECT_FALSE(declare_async_use(&orch, dummy, kCounter));
}

TEST_F(HbgAsyncDeclarations, InvalidSlotsDtypeStrideAndConflictingSubmitContractsFail) {
    for (uint32_t scenario = 0; scenario < 5; ++scenario) {
        new_run();
        const auto task = submit(signals, payload);
        auto invalid = kCounter;
        if (scenario == 0) invalid.condition_arg = 2;
        if (scenario == 1) invalid.payload_arg = 2;
        if (scenario == 2) invalid.record_stride = 3;
        if (scenario == 3) invalid.kind = AsyncUseDeclarationKind::UNKNOWN;
        if (scenario == 4) {
            ASSERT_TRUE(declare_async_use(&orch, task, kCounter));
            invalid.record_stride = 128;
        }
        EXPECT_FALSE(declare_async_use(&orch, task, invalid)) << scenario;
        EXPECT_TRUE(orch.is_fatal());
    }
    new_run();
    auto unsigned_signals = signals;
    unsigned_signals.dtype = DataType::UINT32;
    EXPECT_FALSE(declare_async_use(&orch, submit(unsigned_signals, payload), kCounter));
}

TEST_F(HbgAsyncDeclarations, SdmaDeclarationUsesActualSourceDestinationAndContextArgumentPositions) {
    const auto task = submit(signals, payload);
    AsyncUseDeclaration sdma{AsyncUseDeclarationKind::SDMA_TRANSFER, UINT32_MAX, 1, 0, 0, 0};
    ASSERT_TRUE(declare_async_use(&orch, task, sdma));
    const auto *uses = state->async_declarations->find(false, 0, task.local_id());
    ASSERT_NE(uses, nullptr);
    EXPECT_EQ(uses->front(), sdma);
    new_run();
    sdma.context_scalar = 1;
    EXPECT_FALSE(declare_async_use(&orch, submit(signals, payload), sdma));
}

TEST_F(HbgAsyncDeclarations, OneTaskCanDeclareMultipleSignalsPayloadsAndPeerContexts) {
    uint32_t shape[]{32};
    const auto signal2 = simpler::hbg::make_tensor_external(other_signal.data(), shape, 1, DataType::INT32);
    const auto payload2 = simpler::hbg::make_tensor_external(other_payload.data(), shape, 1, DataType::INT32);
    CoreTaskArgs args;
    const simpler::hbg::Tensor *tensors[]{&signals, &payload, &signal2, &payload2};
    for (const auto *tensor : tensors)
        args.add_no_dep(*tensor);
    args.add_scalar(uint64_t{0x1234}, uint64_t{0x5678});
    MixedKernels kernels{};
    kernels.aiv0_kernel_id = 0;
    const auto task = orch.submit_task(kernels, args).task_id();
    const AsyncUseDeclaration uses[]{
        kCounter,
        {AsyncUseDeclarationKind::COUNTER_PAYLOAD, 0, 3, UINT32_MAX, UINT32_MAX, 4},
        {AsyncUseDeclarationKind::COUNTER_PAYLOAD, 2, 3, UINT32_MAX, UINT32_MAX, 128},
        {AsyncUseDeclarationKind::SDMA_TRANSFER, UINT32_MAX, 3, 1, 0, 0},
        {AsyncUseDeclarationKind::SDMA_TRANSFER, UINT32_MAX, 3, 1, 1, 0},
        {AsyncUseDeclarationKind::SDMA_TRANSFER, UINT32_MAX, 1, 3, 0, 0},
        kConsumer,
        {AsyncUseDeclarationKind::CONSUMER_CLEAN_READ, UINT32_MAX, 3, UINT32_MAX, UINT32_MAX, 0},
    };
    for (const auto &use : uses) {
        ASSERT_TRUE(declare_async_use(&orch, task, use));
        ASSERT_TRUE(declare_async_use(&orch, task, use));
    }
    const auto *actual = state->async_declarations->find(false, 0, task.local_id());
    ASSERT_NE(actual, nullptr);
    ASSERT_EQ(actual->size(), std::size(uses));
    for (size_t i = 0; i < actual->size(); ++i)
        EXPECT_EQ((*actual)[i], uses[i]);
}

TEST_F(HbgAsyncDeclarations, CounterOnlyWaitNeedsNoPayloadTensorAndRejectsImplicitExtraFields) {
    CoreTaskArgs args;
    args.add_no_dep(signals);
    MixedKernels kernels{};
    kernels.aiv0_kernel_id = 0;
    const auto task = orch.submit_task(kernels, args).task_id();
    ASSERT_TRUE(declare_async_use(&orch, task, kCounterOnly));
    ASSERT_TRUE(declare_async_use(&orch, task, kCounterOnly));
    const auto *uses = state->async_declarations->find(false, 0, task.local_id());
    ASSERT_NE(uses, nullptr);
    ASSERT_EQ(uses->size(), 1u);
    EXPECT_EQ(uses->front().payload_arg, UINT32_MAX);
    EXPECT_TRUE(async_use_declaration_valid(kCounterOnly, 1, 0));
    EXPECT_FALSE(async_use_declaration_valid(kCounterOnly, 0, 0));
    for (uint32_t scenario = 0; scenario < 4; ++scenario) {
        auto invalid = kCounterOnly;
        if (scenario == 0) invalid.payload_arg = 0;
        if (scenario == 1) invalid.source_arg = 0;
        if (scenario == 2) invalid.context_scalar = 0;
        if (scenario == 3) invalid.record_stride = 3;
        EXPECT_FALSE(async_use_declaration_valid(invalid, 1, 1)) << scenario;
    }
}

TEST_F(HbgAsyncDeclarations, OnlyContradictoryDeclarationsOfTheSameSignalSelectorConflict) {
    for (bool control_first : {false, true}) {
        new_run();
        const auto task = submit(signals, payload);
        ASSERT_TRUE(declare_async_use(&orch, task, control_first ? kCounterOnly : kCounter));
        EXPECT_FALSE(declare_async_use(&orch, task, control_first ? kCounter : kCounterOnly));
        EXPECT_EQ(state->async_declarations->find(false, 0, task.local_id())->size(), 1u);
    }
    new_run();
    const auto task = submit(signals, payload);
    ASSERT_TRUE(declare_async_use(&orch, task, kCounter));
    auto other_payload_use = kCounter;
    other_payload_use.payload_arg = 0;
    ASSERT_TRUE(declare_async_use(&orch, task, other_payload_use));
    other_payload_use.record_stride = 128;
    EXPECT_FALSE(declare_async_use(&orch, task, other_payload_use));
    EXPECT_EQ(state->async_declarations->find(false, 0, task.local_id())->size(), 2u);
}

TEST_F(HbgAsyncDeclarations, GraphRetainsMultipleUsesAndControlOnlyWaitByArgumentPosition) {
    uint32_t shape[]{32};
    const auto signal2 = simpler::hbg::make_tensor_external(other_signal.data(), shape, 1, DataType::INT32);
    const auto payload2 = simpler::hbg::make_tensor_external(other_payload.data(), shape, 1, DataType::INT32);
    GraphTaskArgs boundary_args;
    const simpler::hbg::Tensor *tensors[]{&signals, &payload, &signal2, &payload2};
    for (const auto *tensor : tensors)
        boundary_args.add_no_dep(*tensor);
    const auto graph = orch.graph_begin(0x33, boundary_args);
    ASSERT_TRUE(graph.recording);
    ASSERT_TRUE(orch.graph_prepare(graph.recording_handle, boundary_args));
    CoreTaskArgs args;
    for (int32_t i = 0; i < 4; ++i)
        args.add_no_dep(graph.params->tensor(i).ref());
    args.add_scalar(uint64_t{0x1234}, uint64_t{0x5678});
    MixedKernels kernels{};
    kernels.aiv0_kernel_id = 0;
    const auto task = orch.submit_task(kernels, args).task_id();
    const AsyncUseDeclaration uses[]{
        kCounter,
        {AsyncUseDeclarationKind::COUNTER_ONLY, 2, UINT32_MAX, UINT32_MAX, UINT32_MAX, 4},
        {AsyncUseDeclarationKind::SDMA_TRANSFER, UINT32_MAX, 3, 1, 0, 0},
        {AsyncUseDeclarationKind::SDMA_TRANSFER, UINT32_MAX, 3, 1, 1, 0},
    };
    for (const auto &use : uses)
        ASSERT_TRUE(declare_async_use(&orch, task, use));
    ASSERT_TRUE(orch.graph_end());
    const auto replay = orch.graph_begin(0x33, boundary_args);
    EXPECT_FALSE(replay.recording);
    EXPECT_FALSE(replay.has_error);
    const auto *actual = state->async_declarations->find(true, 0x33, task.local_id());
    ASSERT_NE(actual, nullptr);
    ASSERT_EQ(actual->size(), std::size(uses));
    for (size_t i = 0; i < actual->size(); ++i)
        EXPECT_EQ((*actual)[i], uses[i]);
    orch.graph_commit();
    EXPECT_FALSE(orch.is_fatal());
}

TEST_F(HbgAsyncDeclarations, GraphKeysSeparateEqualBodyTaskIdsAndReplayKeepsOnlyArgumentIndices) {
    const auto ordinary = submit(signals, payload);
    ASSERT_TRUE(declare_async_use(&orch, ordinary, kConsumer));
    const auto first = record(0x31);
    const auto second = record(0x32, kConsumer);
    ASSERT_EQ(first.local_id(), second.local_id());
    const auto *first_uses = state->async_declarations->find(true, 0x31, first.local_id());
    const auto *second_uses = state->async_declarations->find(true, 0x32, second.local_id());
    ASSERT_NE(first_uses, nullptr);
    ASSERT_NE(second_uses, nullptr);
    EXPECT_EQ(first_uses->front(), kCounter);
    EXPECT_EQ(second_uses->front(), kConsumer);
    EXPECT_EQ(state->async_declarations->find(false, 0, ordinary.local_id())->front(), kConsumer);
    uint32_t shape[]{32};
    const auto next_signal = simpler::hbg::make_tensor_external(other_signal.data(), shape, 1, DataType::INT32);
    const auto next_payload = simpler::hbg::make_tensor_external(other_payload.data(), shape, 1, DataType::INT32);
    GraphTaskArgs replay_args;
    boundary(replay_args, next_signal, next_payload);
    const auto replay = orch.graph_begin(0x31, replay_args);
    EXPECT_FALSE(replay.recording);
    EXPECT_FALSE(replay.has_error);
    EXPECT_EQ(state->async_declarations->find(true, 0x31, first.local_id())->front(), kCounter);
    orch.graph_commit();
    EXPECT_FALSE(orch.is_fatal());
}

TEST_F(HbgAsyncDeclarations, AbortAndFreshCaptureCannotInheritOldGraphDeclarations) {
    GraphTaskArgs args;
    boundary(args, signals, payload);
    const auto graph = orch.graph_begin(0x41, args);
    ASSERT_TRUE(orch.graph_prepare(graph.recording_handle, args));
    const auto task = submit(graph.params->tensor(0).ref(), graph.params->tensor(1).ref());
    ASSERT_TRUE(declare_async_use(&orch, task, kCounter));
    orch.graph_abort(graph.recording_handle);
    EXPECT_EQ(state->async_declarations->find(true, 0x41, task.local_id()), nullptr);
    new_run();
    // An unbound stale side entry must not become a new capture's declaration.
    state->async_declarations = std::make_unique<AsyncHostDeclarations>();
    ASSERT_TRUE(state->async_declarations->add(true, 0x41, 0, kCounter));
    const auto fresh = orch.graph_begin(0x41, args);
    ASSERT_TRUE(fresh.recording);
    EXPECT_EQ(state->async_declarations->find(true, 0x41, 0), nullptr);
    ASSERT_TRUE(orch.graph_prepare(fresh.recording_handle, args));
    const auto fresh_task = submit(fresh.params->tensor(0).ref(), fresh.params->tensor(1).ref());
    ASSERT_TRUE(declare_async_use(&orch, fresh_task, kConsumer));
    ASSERT_TRUE(orch.graph_end());
    EXPECT_EQ(state->async_declarations->find(true, 0x41, 0)->front(), kConsumer);
}

TEST_F(HbgAsyncDeclarations, AnAbortedRecordingCannotPublishLateDeclarations) {
    GraphTaskArgs args;
    boundary(args, signals, payload);
    const auto graph = orch.graph_begin(0x42, args);
    ASSERT_TRUE(orch.graph_prepare(graph.recording_handle, args));
    const auto task = submit(graph.params->tensor(0).ref(), graph.params->tensor(1).ref());
    std::thread aborter([&] {
        orch.graph_abort(graph.recording_handle);
    });
    aborter.join();
    EXPECT_FALSE(declare_async_use(&orch, task, kCounter));
    EXPECT_TRUE(orch.is_fatal());
    EXPECT_EQ(state->async_declarations, nullptr);
}

TEST_F(HbgAsyncDeclarations, ConsecutiveRunsReuseTaskIdsAndGraphKeysWithoutSharingDeclarations) {
    const auto first = submit(signals, payload);
    ASSERT_TRUE(declare_async_use(&orch, first, kCounter));
    record(0x51);
    orch.graph_commit();
    ASSERT_FALSE(orch.is_fatal());
    new_run();
    ASSERT_EQ(state->async_declarations, nullptr);
    const auto second = submit(signals, payload);
    EXPECT_EQ(first, second);
    ASSERT_TRUE(declare_async_use(&orch, second, kConsumer));
    record(0x51, kConsumer);
    EXPECT_EQ(state->async_declarations->find(false, 0, second.local_id())->front(), kConsumer);
    EXPECT_EQ(state->async_declarations->find(true, 0x51, 0)->front(), kConsumer);
}
}  // namespace
