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
#include <vector>

#include "async_backend_plan.h"
#include "host_build_graph/runtime_types.h"

namespace {
constexpr uint64_t kEpoch = 37;
constexpr uint64_t kBudget = 1024 * 1024;
constexpr auto kClean = AsyncUseCacheContract::PUBLISHED_CLEAN_LINES;

struct Fixture {
    struct alignas(64) Image {
        ChipTaskStorage tasks[3]{};
        simpler::hbg::Tensor tensors[3][4]{};
        uint64_t scalars[3][4]{};
    } image;
    alignas(64) std::array<std::array<uint8_t, 512>, 4> buffers{};
    alignas(64) std::array<uint8_t, 4096> workspace{};
    std::array<CommDomainSnapshot, 2> domains{};
    CommSdmaOwnerInfoV1 sdma_owner{};
    CommDomainExecution domain_execution{CommDomainExecution::EXCLUSIVE_RUN};
    bool refuse_sdma{false};
    HostApiOps ops{};
    HostApi api{this, 1, 1, kEpoch, &ops, std::nullopt, this};
    std::vector<SchedulerTaskMetadata> metadata{2};
    std::vector<SchedulerAsyncCapabilities> capabilities{2};
    NativeGraphHostPlan native;
    AsyncHostDeclarations declarations;
    uint32_t queries{0};
    bool refuse{false};
    bool short_owner{false};
    std::string error;

    Fixture() {
        ops.query_async_retained_span = [](void *context, const void *owner, uint32_t slot, uint32_t bank,
                                           uint64_t epoch, uint64_t address, uint64_t bytes,
                                           HostAsyncRetainedSpan *out) {
            auto &self = *static_cast<Fixture *>(context);
            ++self.queries;
            if (owner != &self || slot != 1 || bank != 1 || epoch != kEpoch || self.refuse) return false;
            for (auto &buffer : self.buffers) {
                const uint64_t base = reinterpret_cast<uint64_t>(buffer.data());
                const uint64_t size = self.short_owner ? 4 : buffer.size();
                if (async_backend_detail::contains(base, size, address, bytes)) {
                    *out = {base, size, HostAsyncRetainedOwner::CALLER_BORROW, 0};
                    return true;
                }
            }
            return false;
        };
        for (uint32_t task = 0; task < 3; ++task) {
            image.tasks[task].payload.bind_regions(image.tensors[task], image.scalars[task], nullptr);
            image.tasks[task].payload.tensor_count = 2;
            image.tasks[task].payload.scalar_count = 1;
            for (uint32_t arg = 0; arg < 4; ++arg)
                tensor(task, arg, arg, arg == 0 ? 2 : 16);
        }
        for (uint32_t task = 0; task < 2; ++task) {
            metadata[task].flags = SCHEDULER_TASK_EXECUTABLE;
            metadata[task].logical_block_num = 1;
            metadata[task].active_mask = LaneMask::aiv0();
            metadata[task].timing_slot = -1;
            for (auto &cap : capabilities[task])
                cap = {CORE_ASYNC_CAPABILITY_VERSION, CoreAsyncMode::SYNCHRONOUS, 0, 0};
        }
        capabilities[0][1] = {CORE_ASYNC_CAPABILITY_VERSION, CoreAsyncMode::DEFERRED, 2, CORE_ASYNC_CONDITION_COUNTER};
        add(0, {AsyncUseDeclarationKind::COUNTER_PAYLOAD, 0, 1, UINT32_MAX, UINT32_MAX, 4, kClean});
    }
    void
    tensor(uint32_t task, uint32_t arg, uint32_t buffer, uint32_t elements, uint64_t offset = 0, uint32_t stride = 1) {
        uint32_t shape[]{elements};
        image.tensors[task][arg].init_external(
            buffers[buffer].data(), buffers[buffer].size(), shape, 1, DataType::INT32, 0
        );
        auto &value = image.tensors[task][arg];
        value.start_offset = offset;
        value.strides[0] = stride;
        value.is_contiguous = stride == 1;
        value.extent_elem_cache = elements == 0 ? 0 : static_cast<uint64_t>(elements - 1) * stride + 1;
    }
    uint64_t address(uint32_t buffer, uint32_t offset = 0) const {
        return reinterpret_cast<uint64_t>(buffers[buffer].data()) + offset;
    }
    void add(uint32_t task, const AsyncUseDeclaration &use) { ASSERT_TRUE(declarations.add(false, 0, task, use)); }
    void consume(uint32_t task, uint32_t arg) {
        add(task, {AsyncUseDeclarationKind::CONSUMER_CLEAN_READ, UINT32_MAX, arg, UINT32_MAX, UINT32_MAX, 0, kClean});
    }
    void sdma() {
        declarations = {};
        capabilities[0][1] = {
            CORE_ASYNC_CAPABILITY_VERSION, CoreAsyncMode::DEFERRED, 2, CORE_ASYNC_CONDITION_SDMA_POST_DONE
        };
        image.tasks[0].payload.scalar_count = 2;
        image.scalars[0][0] = 0x10000;
        image.scalars[0][1] = 1;
        tensor(0, 0, 0, 16);
        add(0, {AsyncUseDeclarationKind::SDMA_TRANSFER, UINT32_MAX, 1, 0, 0, 0, kClean, 1});
        sdma_owner = {reinterpret_cast<uint64_t>(workspace.data()), workspace.size(), CommSdmaAccess::SDK_ONLY, 1};
        for (uint32_t i = 0; i < domains.size(); ++i) {
            domains[i].use = {CommDomainKind::LOCAL, 0, i, 0x10000 * (i + 1u)};
            auto &comm = domains[i].context;
            comm.rankId = 0;
            comm.rankNum = 2;
            comm.winSize = buffers[0].size();
            comm.windowsIn[0] = address(i * 2);
            comm.windowsIn[1] = address(2 - i * 2);
            comm_workspace<CommWorkspaceKind::SDMA>(comm) = {sdma_owner.workspace_addr, sdma_owner.workspace_bytes};
        }
        ops.query_async_domain_context = [](void *context, const void *owner, uint32_t, uint32_t, uint64_t epoch,
                                            uint64_t address, CommDomainSnapshot *out, CommDomainExecution *execution) {
            auto &self = *static_cast<Fixture *>(context);
            if (owner != &self || epoch != kEpoch || self.refuse_sdma) return false;
            for (const auto &domain : self.domains)
                if (domain.use.device_context == address) {
                    *out = domain;
                    *execution = self.domain_execution;
                    return true;
                }
            return false;
        };
        ops.query_async_sdma_owner = [](void *context, const void *owner, uint32_t, uint32_t, uint64_t epoch,
                                        uint64_t address, CommSdmaOwnerInfoV1 *out) {
            auto &self = *static_cast<Fixture *>(context);
            if (owner != &self || epoch != kEpoch || self.refuse_sdma) return false;
            for (const auto &domain : self.domains)
                if (domain.use.device_context == address) {
                    *out = self.sdma_owner;
                    return true;
                }
            return false;
        };
    }
    AsyncBackendTaskImage view() const {
        return {&image, sizeof(image), 0x100000, offsetof(Image, tasks), static_cast<uint32_t>(metadata.size())};
    }
    bool build(AsyncBackendHostPlan *plan, uint64_t budget = kBudget) {
        return scheduler_plan_async_backend(
            metadata, capabilities, view(), native, declarations, api, budget, plan, &error
        );
    }
    DeferredCompletionEntry entry(uint64_t where) const {
        return {where, 0, 1, COMPLETION_ENGINE_SDMA, COMPLETION_TYPE_COUNTER, 0};
    }
};

AsyncBackendView view(const AsyncBackendHostPlan &plan) {
    AsyncBackendView result;
    EXPECT_TRUE(result.mount(
        plan.image.empty() ? nullptr : reinterpret_cast<const uint8_t *>(plan.image.data()), plan.image.size()
    ));
    return result;
}
SchedulerAsyncGroupIdentity group(int64_t task = 0) {
    SchedulerAsyncGroupIdentity identity{};
    identity.run_epoch = kEpoch;
    identity.task_ref = task;
    identity.capacity = 64;
    identity.condition_types = CORE_ASYNC_CONDITION_TYPES;
    return identity;
}

TEST(HbgAsyncBackendPlan, UnknownAndBoundedDefaultSkipArgumentDecodingOwnersAndBackendStorage) {
    for (const bool unknown : {false, true}) {
        Fixture f;
        f.declarations = {};
        if (unknown) f.capabilities[0][1] = {};
        f.image.tasks[0].payload.tensor_count = -1;
        AsyncBackendHostPlan plan;
        ASSERT_TRUE(f.build(&plan)) << f.error;
        EXPECT_TRUE(plan.image.empty());
        EXPECT_EQ(f.queries, 0u);
        EXPECT_EQ(plan.execution_contract, HBG_ASYNC_CONTRACT_UNQUALIFIED);
        AsyncCompletionBackend backend(view(plan), plan.execution_contract);
        EXPECT_TRUE(backend.validate_registration(group(), 0, f.entry(f.address(0))));
    }
}

TEST(HbgAsyncBackendPlan, KnownSyncAndCapacityInformationStayInTheCommonPlanner) {
    Fixture f;
    f.declarations = {};
    AsyncBackendHostPlan backend;
    ASSERT_TRUE(f.build(&backend));
    SchedulerAsyncHostPlan plan;
    ASSERT_TRUE(scheduler_plan_async_state(f.metadata, f.capabilities, kEpoch, kBudget, {}, &plan, nullptr));
    EXPECT_EQ(plan.layout.group_count, 1u);
    EXPECT_EQ(plan.layout.condition_capacity, 2u);
    f.capabilities[0][1] = {};
    ASSERT_TRUE(scheduler_plan_async_state(f.metadata, f.capabilities, kEpoch, kBudget, {}, &plan, nullptr));
    EXPECT_EQ(plan.layout.condition_capacity, 64u);
    f.capabilities[0][1] = f.capabilities[1][1];
    ASSERT_TRUE(scheduler_plan_async_state(f.metadata, f.capabilities, kEpoch, kBudget, {}, &plan, nullptr));
    EXPECT_TRUE(plan.image.empty());
}

TEST(HbgAsyncBackendPlan, ExplicitConditionsRemainRestrictedForUnknownKernels) {
    Fixture f;
    f.capabilities[0][1] = {};
    AsyncBackendHostPlan plan;
    ASSERT_TRUE(f.build(&plan)) << f.error;
    AsyncCompletionBackend backend(view(plan), plan.execution_contract);
    EXPECT_TRUE(backend.validate_registration(group(), 0, f.entry(f.address(0))));
    EXPECT_TRUE(backend.validate_registration(group(), 0, f.entry(f.address(0, 4))));
    EXPECT_FALSE(backend.validate_registration(group(), 0, f.entry(f.address(0, 8))));
    EXPECT_GT(f.queries, 0u);
}

TEST(HbgAsyncBackendPlan, CounterRowsAreCompressedAndSharedAcrossPhysicalExecutions) {
    Fixture f;
    AsyncBackendHostPlan ordinary, expanded;
    ASSERT_TRUE(f.build(&ordinary)) << f.error;
    f.metadata[0].logical_block_num = 1000;
    f.metadata[0].active_mask = LaneMask::all();
    f.metadata[0].flags |= SCHEDULER_TASK_MIX | SCHEDULER_TASK_SPMD;
    f.capabilities[0][0] = f.capabilities[0][2] = f.capabilities[0][1];
    ASSERT_TRUE(f.build(&expanded)) << f.error;
    EXPECT_EQ(expanded.image, ordinary.image);
    const auto declarations = view(expanded);
    EXPECT_EQ(declarations.layout.task_count, 1u);
    EXPECT_EQ(declarations.layout.counter_count, 1u);
    EXPECT_EQ(expanded.image.size(), sizeof(AsyncBackendLayout) + sizeof(AsyncBackendTask) + sizeof(AsyncCounterRange));
}

TEST(HbgAsyncBackendPlan, CorePipelineRetirementRequiresEveryTaskToDeclareIt) {
    Fixture f;
    AsyncBackendHostPlan plan;
    ASSERT_TRUE(f.build(&plan));
    EXPECT_EQ(plan.execution_contract, HBG_ASYNC_CONTRACT_UNQUALIFIED);
    f.add(0, {AsyncUseDeclarationKind::CORE_PIPELINES_ONLY});
    ASSERT_TRUE(f.build(&plan));
    EXPECT_EQ(plan.execution_contract, HBG_ASYNC_CONTRACT_UNQUALIFIED);
    f.add(1, {AsyncUseDeclarationKind::CORE_PIPELINES_ONLY});
    ASSERT_TRUE(f.build(&plan));
    EXPECT_EQ(plan.execution_contract, HBG_ASYNC_CONTRACT_CORE_PIPELINES_ONLY);
    f.sdma();
    EXPECT_FALSE(f.declarations.add(false, 0, 0, {AsyncUseDeclarationKind::CORE_PIPELINES_ONLY}));
}

TEST(HbgAsyncBackendPlan, NonzeroOriginsStridesAndInterleavedSelectorsKeepTheirHoles) {
    Fixture f;
    f.tensor(0, 0, 0, 2, 2, 32);
    AsyncBackendHostPlan plan;
    ASSERT_TRUE(f.build(&plan)) << f.error;
    AsyncCompletionBackend backend(view(plan), 0);
    EXPECT_TRUE(backend.validate_registration(group(), 0, f.entry(f.address(0, 8))));
    EXPECT_TRUE(backend.validate_registration(group(), 0, f.entry(f.address(0, 136))));
    EXPECT_FALSE(backend.validate_registration(group(), 0, f.entry(f.address(0, 12))));
    f.image.tasks[0].payload.tensor_count = 4;
    f.tensor(0, 0, 0, 2, 0, 32);
    f.tensor(0, 1, 0, 2, 16, 32);
    f.declarations = {};
    f.add(0, {AsyncUseDeclarationKind::COUNTER_PAYLOAD, 0, 2, UINT32_MAX, UINT32_MAX, 128, kClean});
    f.add(0, {AsyncUseDeclarationKind::COUNTER_PAYLOAD, 1, 3, UINT32_MAX, UINT32_MAX, 128, kClean});
    ASSERT_TRUE(f.build(&plan)) << f.error;
    AsyncCompletionBackend interleaved(view(plan), 0);
    for (uint32_t offset : {0, 64, 128, 192})
        EXPECT_TRUE(interleaved.validate_registration(group(), 0, f.entry(f.address(0, offset))));
    EXPECT_FALSE(interleaved.validate_registration(group(), 0, f.entry(f.address(0, 68))));
}

TEST(HbgAsyncBackendPlan, ConsumerDeclarationsAddNoDeviceRowsAndNoOwnerQueries) {
    Fixture f;
    f.declarations = {};
    f.consume(0, 1);
    f.consume(1, 1);
    AsyncBackendHostPlan plan;
    ASSERT_TRUE(f.build(&plan)) << f.error;
    EXPECT_TRUE(plan.image.empty());
    EXPECT_EQ(f.queries, 0u);
}

TEST(HbgAsyncBackendPlan, EmptyDeclaredWaitIsDistinctFromNoDeclaration) {
    Fixture f;
    f.tensor(0, 0, 0, 0);
    f.declarations = {};
    f.add(0, {AsyncUseDeclarationKind::COUNTER_ONLY, 0, UINT32_MAX, UINT32_MAX, UINT32_MAX, 4, kClean});
    AsyncBackendHostPlan plan;
    ASSERT_TRUE(f.build(&plan)) << f.error;
    EXPECT_EQ(f.queries, 0u);
    const auto declarations = view(plan);
    EXPECT_EQ(declarations.layout.counter_count, 0u);
    AsyncCompletionBackend backend(declarations, 0);
    EXPECT_FALSE(backend.validate_registration(group(0), 0, f.entry(f.address(0))));
    EXPECT_TRUE(backend.validate_registration(group(1), 0, f.entry(f.address(0))));
}

TEST(HbgAsyncBackendPlan, BudgetIncludesBackendImageAndCommonSchedulerBacking) {
    Fixture f;
    AsyncBackendHostPlan backend;
    EXPECT_FALSE(f.build(&backend, 1));
    ASSERT_TRUE(f.build(&backend)) << f.error;
    SchedulerAsyncLayout layout{};
    const SchedulerAsyncAuxiliaryBytes extra{backend.image.size(), 0, backend.execution_contract};
    ASSERT_TRUE(scheduler_plan_async_layout(f.metadata, f.capabilities, kEpoch, kBudget, extra, &layout, nullptr));
    uint64_t offset = 0, total = 0;
    EXPECT_FALSE(scheduler_append_async_layout(256, layout, layout.total_bytes + 255, &offset, &total, nullptr));
    EXPECT_TRUE(scheduler_append_async_layout(256, layout, layout.total_bytes + 256, &offset, &total, nullptr));
}

TEST(HbgAsyncBackendPlan, SdmaKeepsExactPeerViewsWorkspaceAndQueueConfiguration) {
    Fixture f;
    f.sdma();
    f.sdma_owner.queue_num = 2;
    f.tensor(0, 1, 1, 32);
    AsyncBackendHostPlan plan;
    ASSERT_TRUE(f.build(&plan)) << f.error;
    const auto declarations = view(plan);
    ASSERT_EQ(declarations.layout.transfer_count, 1u);
    const auto transfer = declarations.transfer(0);
    EXPECT_EQ(transfer.source, f.address(2));
    EXPECT_EQ(transfer.source_bytes, 64u);
    EXPECT_EQ(transfer.destination, f.address(1));
    EXPECT_EQ(transfer.destination_bytes, 128u);
    EXPECT_EQ(transfer.workspace, f.sdma_owner.workspace_addr);
    EXPECT_EQ(transfer.queue_count, 2u);
    EXPECT_TRUE(plan.needs_sdma_guards);
    f.metadata[0].active_mask = LaneMask::all();
    f.metadata[0].logical_block_num = 2;
    f.metadata[0].flags |= SCHEDULER_TASK_MIX | SCHEDULER_TASK_SPMD;
    const auto before = plan.image;
    ASSERT_TRUE(f.build(&plan)) << f.error;
    EXPECT_EQ(plan.image, before);
}

TEST(HbgAsyncBackendPlan, OwnerAndCleanContractsAreIndependentRequiredFacts) {
    for (uint32_t scenario = 0; scenario < 3; ++scenario) {
        Fixture f;
        if (scenario == 0) f.refuse = true;
        if (scenario == 1) f.short_owner = true;
        if (scenario == 2) {
            f.declarations = {};
            f.add(0, {AsyncUseDeclarationKind::COUNTER_PAYLOAD, 0, 1, UINT32_MAX, UINT32_MAX, 4});
        }
        AsyncBackendHostPlan plan;
        plan.run_epoch = 999;
        EXPECT_FALSE(f.build(&plan)) << scenario;
        EXPECT_EQ(plan.run_epoch, 999u);
        EXPECT_FALSE(f.error.empty());
    }
}

TEST(HbgAsyncBackendPlan, ActualSignalAliasesCannotClaimBothControlOnlyAndPayloadSemantics) {
    Fixture f;
    f.metadata.resize(1);
    f.capabilities.resize(1);
    f.image.tasks[0].payload.tensor_count = 3;
    f.tensor(0, 1, 0, 2);
    f.tensor(0, 2, 1, 16);
    f.declarations = {};
    f.add(0, {AsyncUseDeclarationKind::COUNTER_PAYLOAD, 0, 2, UINT32_MAX, UINT32_MAX, 4, kClean});
    f.add(0, {AsyncUseDeclarationKind::COUNTER_ONLY, 1, UINT32_MAX, UINT32_MAX, UINT32_MAX, 4, kClean});
    f.consume(0, 2);
    AsyncBackendHostPlan plan;
    EXPECT_FALSE(f.build(&plan));
    EXPECT_NE(f.error.find("aliased"), std::string::npos);
}

TEST(HbgAsyncBackendPlan, SdmaRejectsUnknownOwnersWrongPeersAndUnqualifiedWorkspaceHistory) {
    for (uint32_t fault = 0; fault < 9; ++fault) {
        Fixture f;
        f.sdma();
        switch (fault) {
        case 0:
            f.refuse_sdma = true;
            break;
        case 1:
            f.image.scalars[0][1] = 2;
            break;
        case 2:
            f.image.scalars[0][1] = UINT64_MAX;
            break;
        case 3:
            f.image.scalars[0][0] += 8;
            break;
        case 4:
            f.domain_execution = CommDomainExecution::UNKNOWN;
            break;
        case 5:
            f.sdma_owner.access = CommSdmaAccess::LEGACY;
            f.sdma_owner.queue_num = 0;
            break;
        case 6:
            ++f.sdma_owner.workspace_addr;
            break;
        case 7:
            f.sdma_owner.queue_num = 49;
            break;
        case 8:
            f.tensor(0, 0, 3, 16);
            break;
        }
        AsyncBackendHostPlan plan;
        plan.run_epoch = 123;
        EXPECT_FALSE(f.build(&plan)) << fault;
        EXPECT_EQ(plan.run_epoch, 123u) << fault;
        EXPECT_NE(f.error.find("SDMA"), std::string::npos) << fault << ": " << f.error;
    }
}

TEST(HbgAsyncBackendPlan, SubmissionRequirementUsesOnlyInstantiatedDeclarations) {
    Fixture f;
    bool required = true;
    ASSERT_TRUE(
        f.declarations.add(true, 0xdead, 0, {AsyncUseDeclarationKind::SDMA_TRANSFER, UINT32_MAX, 1, 0, 0, 0, kClean, 1})
    );
    ASSERT_TRUE(async_backend_has_sdma_uses(f.metadata, f.native, f.declarations, &required, &f.error));
    EXPECT_FALSE(required);
    f.sdma();
    f.capabilities[0][1] = {CORE_ASYNC_CAPABILITY_VERSION, CoreAsyncMode::SYNCHRONOUS, 0, 0};
    ASSERT_TRUE(async_backend_has_sdma_uses(f.metadata, f.native, f.declarations, &required, &f.error));
    EXPECT_TRUE(required);
}

TEST(HbgAsyncBackendPlan, SdmaGraphUsesCurrentInheritedContextAndPeerScalars) {
    Fixture f;
    f.sdma();
    f.capabilities[1][1] = f.capabilities[0][1];
    f.declarations = {};
    ASSERT_TRUE(
        f.declarations.add(true, 0x88, 0, {AsyncUseDeclarationKind::SDMA_TRANSFER, UINT32_MAX, 1, 0, 0, 0, kClean, 1})
    );
    f.native.definitions.resize(1);
    auto &definition = f.native.definitions[0];
    auto &header = definition.definition;
    header.graph_key = 0x88;
    header.task_count = 1;
    header.tensor_arg_count = header.boundary_tensor_count = 2;
    header.scalar_arg_count = header.boundary_scalar_count = 2;
    header.off_sub_tasks = sizeof(GraphDefinition);
    header.off_tensors = (header.off_sub_tasks + sizeof(SubTaskDefinition) + 7) & ~7u;
    header.off_scalars = header.off_tensors + 2 * sizeof(simpler::hbg::TensorData);
    header.off_scalar_inheritance = header.off_scalars + 2 * sizeof(uint64_t);
    header.total_bytes = header.off_scalar_inheritance + 2 * sizeof(GraphScalarInheritance);
    definition.image.resize(header.total_bytes);
    SubTaskDefinition child{};
    child.tensor_count = child.scalar_count = 2;
    std::memcpy(definition.image.data() + header.off_sub_tasks, &child, sizeof(child));
    for (uint32_t i = 0; i < 2; ++i) {
        auto tensor = static_cast<simpler::hbg::TensorData>(f.image.tensors[0][i]);
        tensor.owner_task_id = TaskId::make_param(i);
        tensor.buffer.addr = 0;
        std::memcpy(definition.image.data() + header.off_tensors + i * sizeof(tensor), &tensor, sizeof(tensor));
        const auto inherited = GraphScalarInheritance::from_boundary(i);
        std::memcpy(
            definition.image.data() + header.off_scalar_inheritance + i * sizeof(inherited), &inherited,
            sizeof(inherited)
        );
        // Captured bits must never be used for an inherited context or peer.
        const uint64_t poison = UINT64_MAX;
        std::memcpy(definition.image.data() + header.off_scalars + i * sizeof(poison), &poison, sizeof(poison));
    }
    for (uint32_t i = 0; i < 2; ++i) {
        f.tensor(i, 0, i * 2, 16, i * 4);
        f.tensor(i, 1, i * 2 + 1, 16);
        f.image.tasks[i].payload.scalar_count = 2;
        f.image.scalars[i][0] = f.domains[i].use.device_context;
        f.image.scalars[i][1] = 1;
        SchedulerGraphInstance instance{};
        instance.original_shell_id = i;
        instance.boundary_tensor_count = instance.boundary_scalar_count = 2;
        instance.heap_base = f.address(i * 2 + 1);
        f.image.tasks[i].task.packed_buffer_base = reinterpret_cast<void *>(instance.heap_base);
        instance.boundary_tensors_address = f.view().device_base + (reinterpret_cast<uint64_t>(&f.image.tensors[i]) -
                                                                    reinterpret_cast<uint64_t>(&f.image));
        instance.boundary_scalars_address = f.view().device_base + (reinterpret_cast<uint64_t>(&f.image.scalars[i]) -
                                                                    reinterpret_cast<uint64_t>(&f.image));
        f.native.instances.push_back(instance);
        SchedulerNodeBinding binding{};
        binding.kind = SchedulerNodeKind::CHILD_KERNEL;
        binding.instance_index = i;
        binding.source_index = 0;
        f.native.bindings.push_back(binding);
    }
    AsyncBackendHostPlan plan;
    ASSERT_TRUE(f.build(&plan)) << f.error;
    const auto declarations = view(plan);
    ASSERT_EQ(declarations.layout.transfer_count, 2u);
    for (uint32_t i = 0; i < 2; ++i) {
        const auto transfer = declarations.transfer(i);
        EXPECT_EQ(transfer.source, f.address(2 - i * 2, i * 16));
        EXPECT_EQ(transfer.source_bytes, 64u);
    }
    f.image.scalars[1][1] = 2;
    EXPECT_FALSE(f.build(&plan));
}

TEST(HbgAsyncBackendPlan, GraphParamsAndBodyHeapUseEachInstancesActualBindings) {
    Fixture f;
    f.declarations = {};
    f.capabilities[1][1] = f.capabilities[0][1];
    f.native.definitions.resize(1);
    auto &definition = f.native.definitions[0];
    auto &header = definition.definition;
    header.graph_key = 0x77;
    header.task_count = 1;
    header.tensor_arg_count = 2;
    header.required_heap = 128;
    header.off_sub_tasks = sizeof(GraphDefinition);
    header.off_tensors = (header.off_sub_tasks + sizeof(SubTaskDefinition) + 7) & ~7u;
    header.total_bytes = header.off_tensors + 2 * sizeof(simpler::hbg::TensorData);
    definition.image.resize(header.total_bytes);
    SubTaskDefinition task{};
    task.tensor_count = 2;
    std::memcpy(definition.image.data() + header.off_sub_tasks, &task, sizeof(task));
    auto parameter = static_cast<simpler::hbg::TensorData>(f.image.tensors[0][0]);
    parameter.owner_task_id = TaskId::make_param(0);
    parameter.buffer.addr = 0;
    parameter.start_offset = 2;
    auto output = static_cast<simpler::hbg::TensorData>(f.image.tensors[0][1]);
    output.owner_task_id = TaskId::make_sub_task(0, 0);
    output.buffer = {64, 64};
    std::memcpy(definition.image.data() + header.off_tensors, &parameter, sizeof(parameter));
    std::memcpy(definition.image.data() + header.off_tensors + sizeof(parameter), &output, sizeof(output));
    for (uint32_t instance_index = 0; instance_index < 2; ++instance_index) {
        f.image.tasks[instance_index].payload.tensor_count = 1;
        f.tensor(instance_index, 0, instance_index, 8, 4);
        SchedulerGraphInstance instance{};
        instance.original_shell_id = instance_index;
        instance.boundary_tensor_count = 1;
        instance.heap_base = f.address(instance_index + 2);
        f.image.tasks[instance_index].task.packed_buffer_base = reinterpret_cast<void *>(instance.heap_base);
        instance.boundary_tensors_address =
            f.view().device_base +
            (reinterpret_cast<uint64_t>(f.image.tensors[instance_index]) - reinterpret_cast<uint64_t>(&f.image));
        f.native.instances.push_back(instance);
        f.native.bindings.push_back({0, instance_index, SchedulerNodeKind::CHILD_KERNEL, {}});
    }
    ASSERT_TRUE(f.declarations.add(
        true, 0x77, 0, {AsyncUseDeclarationKind::COUNTER_PAYLOAD, 0, 1, UINT32_MAX, UINT32_MAX, 4, kClean}
    ));
    AsyncBackendHostPlan plan;
    ASSERT_TRUE(f.build(&plan)) << f.error;
    const auto declarations = view(plan);
    ASSERT_EQ(declarations.layout.task_count, 2u);
    AsyncCompletionBackend backend(declarations, 0);
    EXPECT_TRUE(backend.validate_registration(group(0), 0, f.entry(f.address(0, 24))));
    EXPECT_TRUE(backend.validate_registration(group(1), 0, f.entry(f.address(1, 24))));
    EXPECT_FALSE(backend.validate_registration(group(0), 0, f.entry(f.address(1, 24))));
    f.native.instances[1].heap_base += 64;
    EXPECT_FALSE(f.build(&plan));
}

}  // namespace
