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
#include <cstring>
#include <future>
#include <vector>

#include "host_build_graph/host_graph_build.h"
#include "host_build_graph/graph_execution.h"
#include "host_build_graph/host_tensor_access.h"
#include "host_build_graph/runtime_core.h"
#include "host_build_graph/runtime.h"
#include "common/host_api.h"
#include "worker/runtime_c_api.h"
#include "host/kernel_pipeline_contract.h"
#include "call_config.h"

namespace {

class Buffer {
public:
    void reserve(size_t bytes, size_t alignment = CHIP_ALIGN_SIZE) {
        if (bytes <= capacity_) return;
        std::vector<std::byte> next(bytes + alignment, std::byte{0});
        auto *base = reinterpret_cast<std::byte *>(
            (reinterpret_cast<uintptr_t>(next.data()) + alignment - 1) & ~(alignment - 1)
        );
        if (capacity_ != 0) std::memcpy(base, data_, capacity_);
        storage_ = std::move(next);
        data_ = base;
        capacity_ = bytes;
    }
    std::byte *data() { return data_; }
    size_t size() const { return capacity_; }

private:
    std::vector<std::byte> storage_;
    std::byte *data_{nullptr};
    size_t capacity_{0};
};

struct Platform {
    Buffer runtime_image;
    Buffer definitions;
    Buffer staging;
    std::vector<std::unique_ptr<Buffer>> allocations;
    std::vector<std::vector<std::byte>> copies;
    uint64_t heap_base{0x100000000ULL};
    size_t heap_bytes{0};
    int commits{0};
    int definition_acquires{0};
    int fail_copy{0};

    static void *allocate(void *ctx, size_t bytes) {
        auto &self = *static_cast<Platform *>(ctx);
        auto allocation = std::make_unique<Buffer>();
        allocation->reserve(bytes, DeviceArena::kDefaultBaseAlign);
        void *ptr = allocation->data();
        self.allocations.push_back(std::move(allocation));
        return ptr;
    }
    static void free(void *, void *) {}
    static int copy(void *ctx, void *dst, const void *src, size_t bytes) {
        auto &self = *static_cast<Platform *>(ctx);
        self.copies.emplace_back(static_cast<const std::byte *>(src), static_cast<const std::byte *>(src) + bytes);
        if (self.fail_copy == static_cast<int>(self.copies.size())) return -1;
        std::memcpy(dst, src, bytes);
        return 0;
    }
    static int commit(void *ctx, uint32_t, size_t heap, size_t sm, size_t arena) {
        auto &self = *static_cast<Platform *>(ctx);
        ++self.commits;
        EXPECT_EQ(sm, 0u);
        self.heap_bytes = heap;
        self.runtime_image.reserve(arena, DeviceArena::kDefaultBaseAlign);
        return 0;
    }
    static int acquire_definitions(void *ctx, uint32_t, size_t bytes, size_t align, void **device, void **host) {
        auto &self = *static_cast<Platform *>(ctx);
        ++self.definition_acquires;
        self.definitions.reserve(bytes, align);
        self.staging.reserve(bytes, align);
        *device = self.definitions.data();
        *host = self.staging.data();
        return 0;
    }
    static void *heap(void *ctx, uint32_t) { return reinterpret_cast<void *>(static_cast<Platform *>(ctx)->heap_base); }
    static void *arena(void *ctx, uint32_t) { return static_cast<Platform *>(ctx)->runtime_image.data(); }
};

thread_local RuntimeContext *bound_runtime = nullptr;
void bind(RuntimeContext *rt) { bound_runtime = rt; }
void empty_entry(const ChipTaskArgs &) {}

void chain_entry(const ChipTaskArgs &) {
    const uint32_t shape[] = {16};
    TensorCreateInfo create_output(shape, 1, DataType::UINT32);
    CoreTaskArgs first;
    first.add_output(create_output);
    auto output = bound_runtime->ops->submit_dummy_task(bound_runtime, first);
    ASSERT_TRUE(output.task_id().is_valid());
    CoreTaskArgs second;
    second.add_input(output.get_ref(0));
    second.add_output(create_output);
    ASSERT_TRUE(bound_runtime->ops->submit_dummy_task(bound_runtime, second).task_id().is_valid());
}

void graph_entry(const ChipTaskArgs &) {
    const uint32_t shape[] = {16};
    auto boundary = simpler::hbg::make_tensor_external(reinterpret_cast<uint32_t *>(0x2000), shape, 1);
    GraphTaskArgs args;
    args.add_input(boundary);
    auto &orch = *bound_runtime->orchestrator;
    const auto scope = orch.graph_begin(0x81, args, bound_runtime->active_callable_hash);
    ASSERT_TRUE(scope.recording);
    ASSERT_TRUE(orch.graph_prepare(scope.recording_handle, args));
    chain_entry(ChipTaskArgs{});
    ASSERT_TRUE(orch.graph_end());
}

void repeated_graph_entry(const ChipTaskArgs &args) {
    graph_entry(args);
    const uint32_t shape[] = {16};
    auto boundary = simpler::hbg::make_tensor_external(reinterpret_cast<uint32_t *>(0x2000), shape, 1);
    GraphTaskArgs graph_args;
    graph_args.add_input(boundary);
    const auto scope = bound_runtime->orchestrator->graph_begin(0x81, graph_args, bound_runtime->active_callable_hash);
    EXPECT_FALSE(scope.execute_block);
    EXPECT_TRUE(scope.task_id.is_valid());
}

void two_definitions_entry(const ChipTaskArgs &args) {
    graph_entry(args);
    const uint32_t shape[] = {16};
    auto boundary = simpler::hbg::make_tensor_external(reinterpret_cast<uint32_t *>(0x2000), shape, 1);
    GraphTaskArgs graph_args;
    graph_args.add_input(boundary);
    auto &orch = *bound_runtime->orchestrator;
    const auto scope = orch.graph_begin(0x82, graph_args, bound_runtime->active_callable_hash);
    ASSERT_TRUE(scope.recording);
    ASSERT_TRUE(orch.graph_prepare(scope.recording_handle, graph_args));
    chain_entry(args);
    ASSERT_TRUE(orch.graph_end());
}

void fatal_entry(const ChipTaskArgs &) {
    bound_runtime->orchestrator->report_fatal(SIMPLER_ERROR_INVALID_ARGS, "entry", "%s", "test build failure");
}

class HostGraphBuildTest : public ::testing::Test {
protected:
    static constexpr uint64_t capacity = 64;
    Runtime runtime;
    DeviceArena host_arena;
    RuntimeArenaLayout layout{};
    RuntimeContext *rt{nullptr};
    Buffer mirror;
    Platform platform;
    HostApiOps ops{};
    HostApi api{&platform, 0, 0, &ops};
    HostTensorAccessor tensor_access{&api};
    hbg::GraphBuild result;
    GraphDefinitionArena definition_arena{};

    void SetUp() override {
        ops.copy_to_device = Platform::copy;
        ops.device_malloc = Platform::allocate;
        ops.device_free = Platform::free;
        ops.setup_static_arena = Platform::commit;
        ops.acquire_pooled_gm_heap = Platform::heap;
        ops.acquire_pooled_runtime_arena = Platform::arena;
        ops.acquire_graph_definition_block = Platform::acquire_definitions;
        runtime.set_worker_count(PLATFORM_CORES_PER_BLOCKDIM);
        layout = runtime_reserve_layout(host_arena, capacity);
        ASSERT_NE(host_arena.commit(DeviceArena::kDefaultBaseAlign), nullptr);
        const uint64_t sm_bytes = SharedMemoryHandle::calculate_size(capacity);
        mirror.reserve(sm_bytes);
        rt = runtime_init_data_from_layout(host_arena, layout, MODE_EXECUTE, nullptr, sm_bytes);
        ASSERT_NE(rt, nullptr);
        runtime_wire_arena_pointers(host_arena, layout, rt);
        rt->prebuilt_layout = layout;
        definition_arena.object_prefix_bytes = sizeof(GraphDefinitionHeader);
        definition_arena.object_align = GRAPH_DEFINITION_OBJECT_ALIGN;
    }
    int32_t build(void (*entry)(const ChipTaskArgs &)) {
        return hbg::build_graph(
            &runtime, tensor_access, rt, mirror.data(), mirror.size(), capacity, definition_arena, {entry, bind},
            ChipTaskArgs{}, result
        );
    }
    int32_t upload() { return hbg::upload_program_graph(&runtime, &api, rt, host_arena, layout, result); }

    void build_mixed_definitions() {
        ASSERT_GE(build(graph_entry), 0);
        hbg::GraphResourceRequirements single;
        ASSERT_EQ(hbg::get_graph_resource_requirements(result, layout, single), 0);
        platform.staging.reserve(single.graph_definition_bytes, GRAPH_DEFINITION_OBJECT_ALIGN);
        definition_arena.base = platform.staging.data();
        definition_arena.capacity = platform.staging.size();
        ASSERT_GE(build(two_definitions_entry), 0);
        const auto definitions = graph_host_definitions(*result.graph_state);
        ASSERT_EQ(definitions.entries.size(), 2u);
        EXPECT_EQ(
            std::count_if(
                definitions.entries.begin(), definitions.entries.end(),
                [](const auto &entry) {
                    return entry.spill != nullptr;
                }
            ),
            1
        );
    }
};

TEST_F(HostGraphBuildTest, BuildReturnsWithoutDeviceAllocationOrUpload) {
    ASSERT_EQ(build(chain_entry), 2);
    EXPECT_TRUE(result.ready);
    EXPECT_EQ(result.usage.submitted_tasks, 2u);
    EXPECT_GT(result.heap_bytes, 0u);
    EXPECT_LT(result.image_bytes, mirror.size());
    EXPECT_EQ(platform.commits, 0);
    EXPECT_TRUE(platform.allocations.empty());
    EXPECT_EQ(platform.definition_acquires, 0);
    EXPECT_TRUE(platform.copies.empty());
    EXPECT_EQ(runtime.get_gm_sm_ptr(), nullptr);
    EXPECT_EQ(rt->orchestrator, nullptr);
    ASSERT_GE(upload(), 0);
    EXPECT_FALSE(platform.copies.empty());
}

TEST_F(HostGraphBuildTest, RepeatedUploadPreservesImageAndVirtualHeapSource) {
    ASSERT_EQ(build(chain_entry), 2);
    const auto off = sm_layout::segment_offsets(capacity);
    auto *source = reinterpret_cast<ChipTaskStorage *>(mirror.data() + off.storage);
    const uint64_t virtual_output = source[0].payload.tensor_data()[0].buffer.addr;
    ASSERT_GE(virtual_output, HEAP_VIRTUAL_BASE);
    ASSERT_EQ(upload(), 2);
    const auto first = platform.copies.back();
    ASSERT_EQ(upload(), 2);
    EXPECT_EQ(platform.copies.back(), first);
    EXPECT_EQ(source[0].payload.tensor_data()[0].buffer.addr, virtual_output);
    platform.heap_base += DeviceArena::kDefaultBaseAlign;
    ASSERT_EQ(upload(), 2);
    const auto compact = sm_layout::segment_offsets(sm_layout::image_extents(result.usage));
    auto *uploaded =
        reinterpret_cast<ChipTaskStorage *>(static_cast<std::byte *>(runtime.get_gm_sm_ptr()) + compact.storage);
    EXPECT_EQ(
        uploaded[0].payload.tensor_data()[0].buffer.addr, platform.heap_base + virtual_output - HEAP_VIRTUAL_BASE
    );
    EXPECT_EQ(source[0].payload.tensor_data()[0].buffer.addr, virtual_output);
}

TEST_F(HostGraphBuildTest, EmptyBuildStillUploadsAValidHeader) {
    ASSERT_EQ(build(empty_entry), 0);
    ASSERT_EQ(upload(), 0);
    auto *header = static_cast<SharedMemoryHeader *>(runtime.get_gm_sm_ptr());
    EXPECT_EQ(header->tasks.total_tasks, 0);
    EXPECT_FALSE(platform.copies.empty());
}

TEST_F(HostGraphBuildTest, FailedBuildCannotBeUploaded) {
    EXPECT_LT(build(fatal_entry), 0);
    EXPECT_FALSE(result.ready);
    EXPECT_EQ(upload(), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_TRUE(platform.copies.empty());
    EXPECT_EQ(platform.commits, 0);
    EXPECT_EQ(rt->orchestrator, nullptr);
}

TEST_F(HostGraphBuildTest, GraphDefinitionsSurviveBuildAndUploadBeforeRuntimeImage) {
    ASSERT_GE(build(graph_entry), 0);
    ASSERT_EQ(graph_host_definitions(*result.graph_state).entries.size(), 1u);
    EXPECT_TRUE(platform.copies.empty());
    ASSERT_GE(upload(), 0);
    ASSERT_GE(platform.copies.size(), 2u);
    const size_t copies_per_upload = platform.copies.size();
    const auto first_definition = platform.copies[0];
    const auto first_runtime = platform.copies.back();
    auto *framing = reinterpret_cast<const GraphDefinitionHeader *>(first_definition.data());
    EXPECT_EQ(framing->magic, GRAPH_DEFINITION_OBJECT_MAGIC);
    ASSERT_GE(upload(), 0);
    ASSERT_EQ(platform.copies.size(), 2 * copies_per_upload);
    EXPECT_EQ(platform.copies[copies_per_upload], first_definition);
    EXPECT_EQ(platform.copies.back(), first_runtime);
}

TEST_F(HostGraphBuildTest, RetainedDefinitionStagingIsConsumedWithoutSpill) {
    platform.staging.reserve(256 * 1024, GRAPH_DEFINITION_OBJECT_ALIGN);
    definition_arena.base = platform.staging.data();
    definition_arena.capacity = platform.staging.size();
    ASSERT_GE(build(graph_entry), 0);
    const auto defs = graph_host_definitions(*result.graph_state);
    ASSERT_EQ(defs.entries.size(), 1u);
    EXPECT_EQ(defs.entries[0].spill, nullptr);
    ASSERT_GE(upload(), 0);
    EXPECT_GE(platform.copies.size(), 2u);
}

TEST_F(HostGraphBuildTest, DefinitionUploadFailureDoesNotUploadRuntimeImage) {
    ASSERT_GE(build(graph_entry), 0);
    platform.fail_copy = 1;
    EXPECT_EQ(upload(), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_EQ(platform.commits, 0);
    EXPECT_EQ(platform.copies.size(), 1u);
    platform.fail_copy = 0;
    ASSERT_GE(upload(), 0);
    EXPECT_GE(platform.copies.size(), 3u);
}

TEST_F(HostGraphBuildTest, RepeatedUploadAfterDefinitionStagingGrowth) {
    build_mixed_definitions();
    const void *old_staging = platform.staging.data();
    ASSERT_GE(upload(), 0);
    ASSERT_NE(platform.staging.data(), old_staging);
    const auto first_definitions = platform.copies.front();
    const size_t copies_per_upload = platform.copies.size();
    ASSERT_GE(upload(), 0);
    EXPECT_EQ(platform.copies[copies_per_upload], first_definitions);
}

TEST_F(HostGraphBuildTest, FailedDefinitionCopyAfterGrowthCanBeQueriedAndRetried) {
    build_mixed_definitions();
    hbg::GraphResourceRequirements before;
    ASSERT_EQ(hbg::get_graph_resource_requirements(result, layout, before), 0);
    platform.fail_copy = 1;
    ASSERT_EQ(upload(), PTO_RUNTIME_ERR_INTERNAL);
    hbg::GraphResourceRequirements after;
    ASSERT_EQ(hbg::get_graph_resource_requirements(result, layout, after), 0);
    EXPECT_EQ(after.graph_definition_bytes, before.graph_definition_bytes);
    platform.fail_copy = 0;
    ASSERT_GE(upload(), 0);
    ASSERT_GE(upload(), 0);
}

TEST_F(HostGraphBuildTest, KernelRequirementsMatchDeviceRegionsWithoutUploading) {
    ASSERT_EQ(build(chain_entry), 2);
    hbg::GraphResourceRequirements requirements;
    ASSERT_EQ(hbg::get_graph_resource_requirements(result, layout, requirements), 0);
    hbg::KernelResourcePlan plan;
    ASSERT_EQ(hbg::KernelResourcePlan::create(&requirements, 1, plan), 0);
    const auto contract = plan.pipeline_contract();
    ASSERT_TRUE(is_valid_hbg_kernel_pipeline_contract(&contract));
    EXPECT_EQ(contract.pipeline_depth, 1u);
    EXPECT_EQ(find_pipeline_resource(contract, PTO_PIPELINE_GM_SM), nullptr);
    EXPECT_EQ(requirements.runtime_arena_bytes, layout.off_copied_end + result.image_bytes);
    EXPECT_EQ(requirements.graph_definition_bytes, 0u);
    EXPECT_EQ(find_pipeline_resource(contract, PTO_PIPELINE_RUNTIME_IMAGE)->bytes_per_copy, plan.runtime_arena_bytes());
    if (requirements.scheduler_state_bytes != 0) {
        EXPECT_GE(plan.scheduler_offset(), requirements.runtime_arena_bytes);
        EXPECT_LE(plan.scheduler_offset() + requirements.scheduler_state_bytes, plan.runtime_arena_bytes());
    }
    EXPECT_TRUE(platform.copies.empty());
    EXPECT_TRUE(platform.allocations.empty());
    EXPECT_EQ(platform.commits, 0);
    EXPECT_EQ(platform.definition_acquires, 0);
    ASSERT_EQ(upload(), 2);
    EXPECT_EQ(platform.heap_bytes, requirements.gm_heap_bytes);
    EXPECT_EQ(platform.runtime_image.size(), requirements.runtime_arena_bytes);
    uint64_t scheduler_allocated = 0;
    for (const auto &allocation : platform.allocations)
        scheduler_allocated += allocation->size();
    EXPECT_LE(scheduler_allocated, requirements.scheduler_state_bytes);
    uint64_t total = 0;
    ASSERT_TRUE(requirements.required_bytes(total));
    EXPECT_EQ(
        total, requirements.gm_heap_bytes + requirements.runtime_arena_bytes + requirements.scheduler_state_bytes
    );
}

TEST_F(HostGraphBuildTest, KernelRequirementsCountDefinitionFramingAndRetainedStaging) {
    for (bool retained : {false, true}) {
        if (retained) {
            platform.staging.reserve(256 * 1024, GRAPH_DEFINITION_OBJECT_ALIGN);
            definition_arena.base = platform.staging.data();
            definition_arena.capacity = platform.staging.size();
        }
        ASSERT_GE(build(graph_entry), 0);
        hbg::GraphResourceRequirements requirements;
        ASSERT_EQ(hbg::get_graph_resource_requirements(result, layout, requirements), 0);
        EXPECT_GT(requirements.graph_definition_bytes, 0u);
        EXPECT_EQ(requirements.graph_definition_bytes % GRAPH_DEFINITION_OBJECT_ALIGN, 0u);
        EXPECT_EQ(requirements.scheduler_state_bytes, 0u);
        hbg::KernelResourcePlan plan;
        ASSERT_EQ(hbg::KernelResourcePlan::create(&requirements, 1, plan), 0);
        const auto contract = plan.pipeline_contract();
        ASSERT_TRUE(is_valid_hbg_kernel_pipeline_contract(&contract));
        EXPECT_GE(plan.definition_offset(), requirements.runtime_arena_bytes);
        EXPECT_LE(plan.definition_offset() + requirements.graph_definition_bytes, plan.runtime_arena_bytes());
        EXPECT_EQ(
            find_pipeline_resource(contract, PTO_PIPELINE_RUNTIME_IMAGE)->bytes_per_copy, plan.runtime_arena_bytes()
        );
        ASSERT_GE(upload(), 0);
        EXPECT_EQ(platform.definitions.size(), requirements.graph_definition_bytes);
        EXPECT_EQ(platform.heap_bytes, requirements.gm_heap_bytes);
        EXPECT_EQ(platform.runtime_image.size(), requirements.runtime_arena_bytes);
    }
}

TEST_F(HostGraphBuildTest, KernelRequirementSnapshotSurvivesAnotherBuild) {
    ASSERT_EQ(build(empty_entry), 0);
    hbg::GraphResourceRequirements first;
    ASSERT_EQ(hbg::get_graph_resource_requirements(result, layout, first), 0);
    const uint64_t first_image_bytes = first.runtime_arena_bytes;
    ASSERT_EQ(build(chain_entry), 2);
    hbg::GraphResourceRequirements second;
    ASSERT_EQ(hbg::get_graph_resource_requirements(result, layout, second), 0);
    EXPECT_GT(second.runtime_arena_bytes, first.runtime_arena_bytes);
    EXPECT_EQ(first.runtime_arena_bytes, first_image_bytes);
    EXPECT_TRUE(platform.copies.empty());
}

TEST_F(HostGraphBuildTest, KernelRequirementsRejectFailedBuildAndSizeOverflow) {
    hbg::GraphResourceRequirements requirements{1, 2, 3, 4};
    EXPECT_EQ(hbg::get_graph_resource_requirements(result, layout, requirements), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_EQ(requirements.gm_heap_bytes, 1u);
    EXPECT_LT(build(fatal_entry), 0);
    EXPECT_EQ(hbg::get_graph_resource_requirements(result, layout, requirements), PTO_RUNTIME_ERR_INVALID_STATE);
    ASSERT_EQ(build(chain_entry), 2);
    result.image_bytes = UINT64_MAX;
    EXPECT_EQ(hbg::get_graph_resource_requirements(result, layout, requirements), PTO_RUNTIME_ERR_CAPACITY_EXCEEDED);
    EXPECT_EQ(requirements.runtime_arena_bytes, 2u);
    EXPECT_TRUE(platform.copies.empty());
    EXPECT_EQ(platform.commits, 0);
}

TEST(HbgKernelRequirements, IncludesSeparateBlocksAndRejectsTotalOverflow) {
    hbg::GraphResourceRequirements requirements{4096, 8192, 256, 512};
    uint64_t total = 0;
    ASSERT_TRUE(requirements.required_bytes(total));
    EXPECT_EQ(total, 13056u);
    requirements.graph_definition_bytes = UINT64_MAX;
    EXPECT_FALSE(requirements.required_bytes(total));
    EXPECT_EQ(total, 13056u);
    requirements.graph_definition_bytes = 0;
    requirements.scheduler_state_bytes = UINT64_MAX;
    EXPECT_FALSE(requirements.required_bytes(total));
    EXPECT_EQ(total, 13056u);
}

TEST(HbgKernelRequirements, RejectsMalformedDefinitionPacking) {
    uint64_t bytes = 17;
    GraphHostDefinitionList defs;
    EXPECT_FALSE(hbg::graph_definition_block_bytes(defs, 1, bytes));
    defs.entries.push_back({1, GRAPH_NO_OBJECT_OFFSET, reinterpret_cast<const std::byte *>(1), UINT64_MAX});
    EXPECT_FALSE(hbg::graph_definition_block_bytes(defs, 0, bytes));
    defs.entries[0].bytes = sizeof(GraphDefinition);
    EXPECT_FALSE(hbg::graph_definition_block_bytes(defs, UINT64_MAX - GRAPH_DEFINITION_OBJECT_ALIGN + 1, bytes));
    defs.entries[0] = {1, 0, nullptr, sizeof(GraphDefinition)};
    EXPECT_FALSE(hbg::graph_definition_block_bytes(defs, 0, bytes));
    EXPECT_EQ(bytes, 17u);
}

TEST(HbgKernelResourcePlan, RebuildsDisjointRegionsFromCompatibleGraphMaxima) {
    const hbg::GraphResourceRequirements graphs[] = {{4096, 8192, 512, 0}, {8192, 4096, 0, 2048}};
    hbg::KernelResourcePlan plan;
    ASSERT_EQ(hbg::KernelResourcePlan::create(graphs, 2, plan), 0);
    EXPECT_EQ(plan.capacity().gm_heap_bytes, 8192u);
    EXPECT_EQ(plan.capacity().runtime_arena_bytes, 8192u);
    EXPECT_EQ(plan.definition_offset(), 8192u);
    EXPECT_EQ(plan.scheduler_offset(), 9216u);
    EXPECT_EQ(plan.runtime_arena_bytes(), 11264u);
    EXPECT_TRUE(plan.admits(graphs[0]));
    EXPECT_TRUE(plan.admits(graphs[1]));
    const auto contract = plan.pipeline_contract();
    ASSERT_TRUE(is_valid_hbg_kernel_pipeline_contract(&contract));
    EXPECT_EQ(find_pipeline_resource(contract, PTO_PIPELINE_GM_HEAP)->bytes_per_copy, 8192u);
    EXPECT_EQ(find_pipeline_resource(contract, PTO_PIPELINE_RUNTIME_IMAGE)->bytes_per_copy, 11264u);
}

TEST(HbgKernelResourcePlan, RejectsEachRegionOverCapacityWithoutChangingThePlan) {
    const hbg::GraphResourceRequirements graph{4096, 8192, 512, 2048};
    hbg::KernelResourcePlan plan;
    EXPECT_FALSE(plan.admits(graph));
    ASSERT_EQ(hbg::KernelResourcePlan::create(&graph, 1, plan), 0);
    EXPECT_TRUE(plan.admits(graph));
    const uint64_t arena_bytes = plan.runtime_arena_bytes();
    for (auto field :
         {&hbg::GraphResourceRequirements::gm_heap_bytes, &hbg::GraphResourceRequirements::runtime_arena_bytes,
          &hbg::GraphResourceRequirements::graph_definition_bytes,
          &hbg::GraphResourceRequirements::scheduler_state_bytes}) {
        auto candidate = graph;
        ++(candidate.*field);
        EXPECT_FALSE(plan.admits(candidate));
        EXPECT_TRUE(plan.admits(graph));
        EXPECT_EQ(plan.runtime_arena_bytes(), arena_bytes);
    }
    EXPECT_FALSE(plan.admits({}));
}

TEST(HbgKernelResourcePlan, RejectsAggregateAndAlignmentOverflowWithoutPublishing) {
    const hbg::GraphResourceRequirements good{4096, 8192, 512, 0};
    hbg::KernelResourcePlan plan;
    ASSERT_EQ(hbg::KernelResourcePlan::create(&good, 1, plan), 0);
    const auto original = plan.pipeline_contract();
    const hbg::GraphResourceRequirements aggregate[] = {{UINT64_MAX / 2, 1, 0, 0}, {1, UINT64_MAX / 2 + 2, 0, 0}};
    const hbg::GraphResourceRequirements alignment{1, UINT64_MAX - 512, 16, 0};
    for (const auto &graph : aggregate) {
        uint64_t total;
        ASSERT_TRUE(graph.required_bytes(total));
    }
    EXPECT_EQ(hbg::KernelResourcePlan::create(aggregate, 2, plan), PTO_RUNTIME_ERR_CAPACITY_EXCEEDED);
    EXPECT_EQ(hbg::KernelResourcePlan::create(&alignment, 1, plan), PTO_RUNTIME_ERR_CAPACITY_EXCEEDED);
    EXPECT_EQ(hbg::KernelResourcePlan::create(nullptr, 1, plan), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_EQ(hbg::KernelResourcePlan::create(&good, 0, plan), PTO_RUNTIME_ERR_INTERNAL);
    const auto after = plan.pipeline_contract();
    EXPECT_EQ(std::memcmp(&after, &original, sizeof(after)), 0);
    EXPECT_EQ(plan.definition_offset(), 8192u);
    EXPECT_TRUE(plan.admits(good));
}

TEST(HbgKernelStreamBinding, BindsThreeDistinctStreamsForBothRuntimeContracts) {
    const hbg::GraphResourceRequirements graph{4096, 8192, 0, 0};
    hbg::KernelResourcePlan plan;
    ASSERT_EQ(hbg::KernelResourcePlan::create(&graph, 1, plan), 0);
    auto contract = plan.pipeline_contract();
    int caller = 0, aicpu = 0, hidden = 0;
    KernelStreamBinding binding;
    ASSERT_EQ(bind_kernel_stream_roles(&contract, &caller, &aicpu, &hidden, binding), 0);
    EXPECT_EQ(binding.caller_stream, &caller);
    EXPECT_EQ(binding.aicpu_stream, &aicpu);
    EXPECT_EQ(binding.aicore_stream, &hidden);
    contract.resources[0].resource_class = PTO_PIPELINE_DEVICE_SCRATCH;
    contract.resources[1].resource_class = PTO_PIPELINE_DEVICE_SCRATCH;
    contract.resources[contract.resource_count++] = {PTO_PIPELINE_GM_SM, PTO_PIPELINE_DEVICE_SCRATCH, 1024};
    contract.resources[contract.resource_count++] = {PTO_PIPELINE_TASK_ARGS, PTO_PIPELINE_HOST_PER_RUN, 0};
    ASSERT_TRUE(is_valid_tmr_kernel_pipeline_contract(&contract));
    ASSERT_EQ(bind_kernel_stream_roles(&contract, &caller, &aicpu, &hidden, binding), 0);
    EXPECT_EQ(binding.caller_stream, &caller);
    EXPECT_EQ(binding.aicpu_stream, &aicpu);
    EXPECT_EQ(binding.aicore_stream, &hidden);
}

TEST(HbgKernelStreamBinding, RejectsMissingRolesAndEveryStreamAliasBeforePublishing) {
    const hbg::GraphResourceRequirements graph{4096, 8192, 0, 0};
    hbg::KernelResourcePlan plan;
    ASSERT_EQ(hbg::KernelResourcePlan::create(&graph, 1, plan), 0);
    auto contract = plan.pipeline_contract();
    int caller = 0, aicpu = 0, hidden = 0;
    KernelStreamBinding binding{&caller, &aicpu, &hidden};
    for (uint32_t missing : {PTO_PIPELINE_AICPU_STREAM, PTO_PIPELINE_AICORE_STREAM}) {
        auto candidate = contract;
        for (uint32_t i = 0; i < candidate.resource_count; ++i) {
            if (candidate.resources[i].kind == missing) {
                candidate.resources[i] = candidate.resources[--candidate.resource_count];
                break;
            }
        }
        EXPECT_EQ(bind_kernel_stream_roles(&candidate, &caller, &aicpu, &hidden, binding), PTO_RUNTIME_ERR_INTERNAL);
    }
    EXPECT_EQ(bind_kernel_stream_roles(nullptr, &caller, &aicpu, &hidden, binding), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_EQ(bind_kernel_stream_roles(&contract, nullptr, &aicpu, &hidden, binding), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_EQ(bind_kernel_stream_roles(&contract, &caller, nullptr, &hidden, binding), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_EQ(bind_kernel_stream_roles(&contract, &caller, &aicpu, nullptr, binding), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_EQ(bind_kernel_stream_roles(&contract, &caller, &caller, &hidden, binding), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_EQ(bind_kernel_stream_roles(&contract, &caller, &aicpu, &caller, binding), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_EQ(bind_kernel_stream_roles(&contract, &caller, &hidden, &hidden, binding), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_EQ(
        bind_kernel_stream_roles(get_pipeline_contract(), &caller, &aicpu, &hidden, binding), PTO_RUNTIME_ERR_INTERNAL
    );
    EXPECT_EQ(binding.caller_stream, &caller);
    EXPECT_EQ(binding.aicpu_stream, &aicpu);
    EXPECT_EQ(binding.aicore_stream, &hidden);
}

TEST_F(HostGraphBuildTest, KernelRequirementsCountSharedDefinitionsOnce) {
    ASSERT_GE(build(repeated_graph_entry), 0);
    ASSERT_EQ(graph_host_upload_count(*result.graph_state), 2u);
    const auto definitions = graph_host_definitions(*result.graph_state);
    ASSERT_EQ(definitions.entries.size(), 1u);
    hbg::GraphResourceRequirements requirements;
    ASSERT_EQ(hbg::get_graph_resource_requirements(result, layout, requirements), 0);
    const auto object_bytes = sizeof(GraphDefinitionHeader) + definitions.entries[0].bytes;
    const auto aligned = (object_bytes + GRAPH_DEFINITION_OBJECT_ALIGN - 1) & ~(GRAPH_DEFINITION_OBJECT_ALIGN - 1);
    EXPECT_EQ(requirements.graph_definition_bytes, aligned);
    ASSERT_GE(upload(), 0);
    EXPECT_EQ(platform.definitions.size(), requirements.graph_definition_bytes);
}

TEST_F(HostGraphBuildTest, KernelQueryPreservesProgramDeclaration) {
    const PipelineContract before = *get_pipeline_contract();
    ASSERT_TRUE(is_valid_pipeline_contract(&before));
    ASSERT_EQ(build(empty_entry), 0);
    hbg::GraphResourceRequirements requirements;
    ASSERT_EQ(hbg::get_graph_resource_requirements(result, layout, requirements), 0);
    const auto *after = get_pipeline_contract();
    EXPECT_EQ(after->pipeline_depth, before.pipeline_depth);
    EXPECT_EQ(after->resource_count, before.resource_count);
    for (uint32_t i = 0; i < after->resource_count; ++i) {
        EXPECT_EQ(after->resources[i].kind, before.resources[i].kind);
        EXPECT_EQ(after->resources[i].resource_class, before.resources[i].resource_class);
        EXPECT_EQ(after->resources[i].bytes_per_copy, 0u);
    }
}

TEST(HbgKernelRequirements, ConfigOnlyHookCannotPublishGraphDependentSizes) {
    const PipelineContract original = *get_pipeline_contract();
    PipelineContract output = original;
    CallConfig config;
    EXPECT_EQ(build_kernel_pipeline_contract_impl(&config, &output), PTO_RUNTIME_ERR_UNSUPPORTED);
    EXPECT_EQ(std::memcmp(&output, &original, sizeof(output)), 0);
    EXPECT_EQ(build_kernel_pipeline_contract_impl(nullptr, &output), PTO_RUNTIME_ERR_UNSUPPORTED);
    EXPECT_EQ(std::memcmp(&output, &original, sizeof(output)), 0);
}

TEST_F(HostGraphBuildTest, ConcurrentKernelQueriesPublishIndependentSnapshots) {
    ASSERT_GE(build(graph_entry), 0);
    hbg::GraphResourceRequirements expected;
    ASSERT_EQ(hbg::get_graph_resource_requirements(result, layout, expected), 0);
    auto query = [&]() {
        hbg::GraphResourceRequirements output;
        const auto status = hbg::get_graph_resource_requirements(result, layout, output);
        return std::make_pair(status, output);
    };
    auto first = std::async(std::launch::async, query);
    auto second = std::async(std::launch::async, query);
    for (auto *future : {&first, &second}) {
        const auto [status, output] = future->get();
        EXPECT_EQ(status, 0);
        EXPECT_EQ(output.gm_heap_bytes, expected.gm_heap_bytes);
        EXPECT_EQ(output.runtime_arena_bytes, expected.runtime_arena_bytes);
        EXPECT_EQ(output.graph_definition_bytes, expected.graph_definition_bytes);
        EXPECT_EQ(output.scheduler_state_bytes, expected.scheduler_state_bytes);
    }
    EXPECT_TRUE(platform.copies.empty());
    EXPECT_EQ(platform.commits, 0);
}

struct ResourceContextPlatform {
    int current_device{0};
    uintptr_t next_handle{1};
    int allocation_calls{0};
    int free_calls{0};
    int fail_allocation{0};
    int free_failures{0};
    std::vector<KernelStreamKind> stream_kinds;
    MemoryAllocator allocator;

    KernelContextOps context_ops() {
        return {
            this,
            [](void *ctx, int *device) {
                *device = static_cast<ResourceContextPlatform *>(ctx)->current_device;
                return 0;
            },
            [](void *ctx, KernelStreamKind kind, void **stream) {
                auto &self = *static_cast<ResourceContextPlatform *>(ctx);
                self.stream_kinds.push_back(kind);
                *stream = reinterpret_cast<void *>(self.next_handle++);
                return 0;
            },
            [](void *, KernelStreamKind, void *) {
                return 0;
            },
            [](void *ctx, void **event) {
                *event = reinterpret_cast<void *>(static_cast<ResourceContextPlatform *>(ctx)->next_handle++);
                return 0;
            },
            [](void *, void *) {
                return 0;
            }
        };
    }
    KernelResourceOps resource_ops() {
        return {
            this,
            [](void *ctx, size_t bytes) -> void * {
                auto &self = *static_cast<ResourceContextPlatform *>(ctx);
                if (++self.allocation_calls == self.fail_allocation) return nullptr;
                return self.allocator.alloc(bytes);
            },
            [](void *ctx, void *ptr) {
                auto &self = *static_cast<ResourceContextPlatform *>(ctx);
                ++self.free_calls;
                if (self.free_failures > 0) {
                    --self.free_failures;
                    return -77;
                }
                return self.allocator.free(ptr);
            }
        };
    }
};

TEST_F(HostGraphBuildTest, KernelPrepareOwnsActualGraphCapacityAndLaunchOnlyBorrowsIt) {
    ResourceContextPlatform provider;
    KernelExecutionState context;
    ASSERT_EQ(context.initialize(0, provider.context_ops(), 19), 0);
    ASSERT_EQ(
        provider.stream_kinds, (std::vector<KernelStreamKind>{KernelStreamKind::Aicpu, KernelStreamKind::Aicore})
    );
    ASSERT_GE(build(graph_entry), 0);
    hbg::GraphResourceRequirements graph;
    ASSERT_EQ(hbg::get_graph_resource_requirements(result, layout, graph), 0);
    hbg::KernelResourcePlan plan;
    ASSERT_EQ(hbg::KernelResourcePlan::create(&graph, 1, plan), 0);
    ASSERT_EQ(plan.prepare(context, provider.resource_ops()), 0);
    EXPECT_TRUE(context.resources_prepared());
    EXPECT_EQ(provider.allocation_calls, 2);
    EXPECT_EQ(provider.allocator.get_allocation_count(), 2u);
    EXPECT_TRUE(platform.copies.empty());
    EXPECT_EQ(platform.commits, 0);
    EXPECT_EQ(platform.definition_acquires, 0);
    ASSERT_EQ(context.mark_ready_enqueued(), 0);
    hbg::KernelWorkingBinding binding;
    EXPECT_EQ(hbg::bind_kernel_resources_for_launch(context, 0, 19, graph, binding), PTO_RUNTIME_ERR_INVALID_STATE);
    ASSERT_EQ(context.freeze_resources(), 0);
    ASSERT_EQ(hbg::bind_kernel_resources_for_launch(context, 0, 19, graph, binding), 0);
    EXPECT_NE(binding.heap.address, 0u);
    EXPECT_EQ(binding.definitions.address, binding.runtime_image.address + plan.definition_offset());
    EXPECT_EQ(binding.definitions.capacity, graph.graph_definition_bytes);
    const auto first = binding;
    const size_t committed = provider.allocator.committed_bytes();
    std::memset(reinterpret_cast<void *>(binding.definitions.address), 0x5a, binding.definitions.capacity);
    for (int i = 0; i < 10; ++i) {
        ASSERT_EQ(plan.prepare(context, provider.resource_ops()), 0);
        ASSERT_EQ(hbg::bind_kernel_resources_for_launch(context, 0, 19, graph, binding), 0);
        EXPECT_EQ(binding.heap.address, first.heap.address);
        EXPECT_EQ(binding.runtime_image.address, first.runtime_image.address);
        EXPECT_EQ(binding.definitions.address, first.definitions.address);
        EXPECT_EQ(*reinterpret_cast<const uint8_t *>(binding.definitions.address), 0x5a);
    }
    EXPECT_EQ(provider.allocation_calls, 2);
    EXPECT_EQ(provider.free_calls, 0);
    EXPECT_EQ(provider.allocator.committed_bytes(), committed);
    ASSERT_EQ(context.close(), 0);
    EXPECT_EQ(provider.allocator.committed_bytes(), 0u);
    EXPECT_EQ(provider.free_calls, 2);
}

TEST(HbgKernelResourceContext, AggregatedCapacityRejectsExcessAndPreservesOffsetsForSmallerGraphs) {
    ResourceContextPlatform provider;
    KernelExecutionState context;
    ASSERT_EQ(context.initialize(0, provider.context_ops(), 23), 0);
    const hbg::GraphResourceRequirements graphs[] = {{4096, 8192, 512, 0}, {8192, 4096, 0, 2048}};
    hbg::KernelResourcePlan plan;
    ASSERT_EQ(hbg::KernelResourcePlan::create(graphs, 2, plan), 0);
    ASSERT_EQ(plan.prepare(context, provider.resource_ops()), 0);
    ASSERT_EQ(context.freeze_resources(), 0);
    ASSERT_EQ(context.mark_ready_enqueued(), 0);
    hbg::KernelWorkingBinding first;
    ASSERT_EQ(hbg::bind_kernel_resources_for_launch(context, 0, 23, plan.capacity(), first), 0);
    EXPECT_EQ(first.scheduler.address, first.runtime_image.address + 9216);
    for (const auto &graph : graphs) {
        hbg::KernelResourcePlan smaller;
        ASSERT_EQ(hbg::KernelResourcePlan::create(&graph, 1, smaller), 0);
        ASSERT_EQ(smaller.prepare(context, provider.resource_ops()), 0);
        hbg::KernelWorkingBinding binding;
        ASSERT_EQ(hbg::bind_kernel_resources_for_launch(context, 0, 23, graph, binding), 0);
        EXPECT_EQ(binding.scheduler.address, first.scheduler.address);
        EXPECT_EQ(binding.definitions.address, first.definitions.address);
    }
    for (auto field :
         {&hbg::GraphResourceRequirements::gm_heap_bytes, &hbg::GraphResourceRequirements::runtime_arena_bytes,
          &hbg::GraphResourceRequirements::graph_definition_bytes,
          &hbg::GraphResourceRequirements::scheduler_state_bytes}) {
        auto excess = plan.capacity();
        ++(excess.*field);
        auto binding = first;
        EXPECT_EQ(
            hbg::bind_kernel_resources_for_launch(context, 0, 23, excess, binding), PTO_RUNTIME_ERR_CAPACITY_EXCEEDED
        );
        EXPECT_EQ(binding.heap.address, first.heap.address);
        EXPECT_EQ(binding.scheduler.address, first.scheduler.address);
        hbg::KernelResourcePlan larger;
        ASSERT_EQ(hbg::KernelResourcePlan::create(&excess, 1, larger), 0);
        EXPECT_EQ(larger.prepare(context, provider.resource_ops()), PTO_RUNTIME_ERR_CAPACITY_EXCEEDED);
        ASSERT_EQ(hbg::bind_kernel_resources_for_launch(context, 0, 23, plan.capacity(), binding), 0);
    }
    EXPECT_EQ(provider.allocation_calls, 2);
    EXPECT_EQ(provider.free_calls, 0);
    ASSERT_EQ(context.close(), 0);
}

TEST(HbgKernelResourceContext, GuardsLifecycleDeviceAndGenerationBeforeBinding) {
    ResourceContextPlatform provider;
    KernelExecutionState context;
    const hbg::GraphResourceRequirements graph{4096, 8192, 0, 0};
    hbg::KernelResourcePlan plan;
    ASSERT_EQ(hbg::KernelResourcePlan::create(&graph, 1, plan), 0);
    EXPECT_EQ(plan.prepare(context, provider.resource_ops()), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_EQ(context.initialize(0, provider.context_ops(), 0), PTO_RUNTIME_ERR_INTERNAL);
    ASSERT_EQ(context.initialize(0, provider.context_ops(), 31), 0);
    EXPECT_EQ(context.freeze_resources(), PTO_RUNTIME_ERR_INVALID_STATE);
    provider.current_device = 1;
    EXPECT_EQ(plan.prepare(context, provider.resource_ops()), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_EQ(provider.allocation_calls, 0);
    provider.current_device = 0;
    ASSERT_EQ(plan.prepare(context, provider.resource_ops()), 0);
    ASSERT_EQ(context.freeze_resources(), 0);
    EXPECT_EQ(context.freeze_resources(), PTO_RUNTIME_ERR_INVALID_STATE);
    hbg::KernelWorkingBinding binding;
    EXPECT_EQ(hbg::bind_kernel_resources_for_launch(context, 0, 31, graph, binding), PTO_RUNTIME_ERR_INVALID_STATE);
    ASSERT_EQ(context.mark_ready_enqueued(), 0);
    EXPECT_EQ(hbg::bind_kernel_resources_for_launch(context, 1, 31, graph, binding), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_EQ(hbg::bind_kernel_resources_for_launch(context, 0, 32, graph, binding), PTO_RUNTIME_ERR_INVALID_STATE);
    ASSERT_EQ(hbg::bind_kernel_resources_for_launch(context, 0, 31, graph, binding), 0);
    context.poison(-45);
    EXPECT_EQ(hbg::bind_kernel_resources_for_launch(context, 0, 31, graph, binding), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_EQ(plan.prepare(context, provider.resource_ops()), PTO_RUNTIME_ERR_INVALID_STATE);
    ASSERT_EQ(context.close(), 0);
    EXPECT_EQ(hbg::bind_kernel_resources_for_launch(context, 0, 31, graph, binding), PTO_RUNTIME_ERR_INVALID_STATE);
}

TEST(HbgKernelResourceContext, AllocationFailureRollsBackAndCanBeRetried) {
    for (int failure : {1, 2}) {
        ResourceContextPlatform provider;
        provider.fail_allocation = failure;
        KernelExecutionState context;
        ASSERT_EQ(context.initialize(0, provider.context_ops(), 41), 0);
        const hbg::GraphResourceRequirements graph{4096, 8192, 512, 0};
        hbg::KernelResourcePlan plan;
        ASSERT_EQ(hbg::KernelResourcePlan::create(&graph, 1, plan), 0);
        EXPECT_EQ(plan.prepare(context, provider.resource_ops()), PTO_RUNTIME_ERR_INTERNAL);
        EXPECT_EQ(provider.allocator.get_allocation_count(), 0u);
        EXPECT_FALSE(context.resources_prepared());
        EXPECT_FALSE(context.resources_frozen());
        EXPECT_EQ(context.phase(), KernelContextPhase::Collecting);
        provider.fail_allocation = 0;
        ASSERT_EQ(plan.prepare(context, provider.resource_ops()), 0);
        ASSERT_EQ(context.close(), 0);
        EXPECT_EQ(provider.allocator.get_allocation_count(), 0u);
    }
}

TEST(HbgKernelResourceContext, FailedRollbackAndFailedCloseRetainOwnershipForRetry) {
    ResourceContextPlatform provider;
    KernelExecutionState context;
    ASSERT_EQ(context.initialize(0, provider.context_ops(), 43), 0);
    const hbg::GraphResourceRequirements graph{4096, 8192, 0, 0};
    hbg::KernelResourcePlan plan;
    ASSERT_EQ(hbg::KernelResourcePlan::create(&graph, 1, plan), 0);
    provider.fail_allocation = 2;
    provider.free_failures = 1;
    EXPECT_EQ(plan.prepare(context, provider.resource_ops()), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_EQ(context.phase(), KernelContextPhase::Closing);
    EXPECT_EQ(context.unexpected_teardown_error(), -77);
    EXPECT_EQ(provider.allocator.get_allocation_count(), 1u);
    EXPECT_EQ(plan.prepare(context, provider.resource_ops()), PTO_RUNTIME_ERR_INVALID_STATE);
    provider.free_failures = 1;
    EXPECT_EQ(context.close(), -77);
    EXPECT_EQ(context.phase(), KernelContextPhase::Closing);
    ASSERT_EQ(context.close(), 0);
    EXPECT_EQ(provider.allocator.get_allocation_count(), 0u);
    EXPECT_EQ(context.phase(), KernelContextPhase::Closed);
}

TEST(HbgKernelResourceContext, UsesThePlatformAllocatorAndReleasesOnlyContextOwnedBlocks) {
    MemoryAllocator allocator;
    void *tensor = allocator.alloc(64);
    ASSERT_NE(tensor, nullptr);
    ResourceContextPlatform provider;
    KernelExecutionState context;
    ASSERT_EQ(context.initialize(0, provider.context_ops(), 47), 0);
    const hbg::GraphResourceRequirements graph{4096, 8192, 512, 2048};
    hbg::KernelResourcePlan plan;
    ASSERT_EQ(hbg::KernelResourcePlan::create(&graph, 1, plan), 0);
    ASSERT_EQ(plan.prepare(context, KernelResourceOps::from_allocator(allocator)), 0);
    EXPECT_EQ(allocator.get_allocation_count(), 3u);
    ASSERT_EQ(context.close(), 0);
    EXPECT_EQ(allocator.get_allocation_count(), 1u);
    EXPECT_EQ(allocator.committed_bytes(), 64u);
    EXPECT_EQ(allocator.free(tensor), 0);
}

TEST(HbgKernelResourceContext, InvalidLayoutsFailBeforeAnyDeviceAllocation) {
    ResourceContextPlatform provider;
    KernelExecutionState context;
    ASSERT_EQ(context.initialize(0, provider.context_ops(), 53), 0);
    const hbg::GraphResourceRequirements graph{4096, 8192, 512, 0};
    hbg::KernelResourcePlan plan;
    ASSERT_EQ(hbg::KernelResourcePlan::create(&graph, 1, plan), 0);
    const KernelResourceLayout valid{
        hbg::KernelResourcePlan::resource_schema,
        plan.pipeline_contract(),
        {{PTO_PIPELINE_GM_HEAP, 0, 4096},
         {PTO_PIPELINE_RUNTIME_IMAGE, 0, 8192},
         {PTO_PIPELINE_RUNTIME_IMAGE, 8192, 512},
         {PTO_PIPELINE_RUNTIME_IMAGE, 0, 0}}
    };
    auto overlap = valid;
    overlap.regions[2].offset = 1024;
    EXPECT_EQ(context.prepare_resources(overlap, provider.resource_ops()), PTO_RUNTIME_ERR_INTERNAL);
    auto outside = valid;
    outside.regions[2].bytes = 1024;
    EXPECT_EQ(context.prepare_resources(outside, provider.resource_ops()), PTO_RUNTIME_ERR_INTERNAL);
    auto unaligned = valid;
    unaligned.regions[2].offset = 8193;
    EXPECT_EQ(context.prepare_resources(unaligned, provider.resource_ops()), PTO_RUNTIME_ERR_INTERNAL);
    auto overflow = valid;
    overflow.contract.resources[0].bytes_per_copy = UINT64_MAX;
    EXPECT_EQ(context.prepare_resources(overflow, provider.resource_ops()), PTO_RUNTIME_ERR_CAPACITY_EXCEEDED);
    auto unknown = valid;
    unknown.schema = 0;
    EXPECT_EQ(context.prepare_resources(unknown, provider.resource_ops()), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_EQ(provider.allocation_calls, 0);
    EXPECT_FALSE(context.resources_prepared());
    ASSERT_EQ(context.prepare_resources(valid, provider.resource_ops()), 0);
    ASSERT_EQ(context.mark_ready_enqueued(), 0);
    ASSERT_EQ(context.freeze_resources(), 0);
    KernelResourceBinding binding;
    const uint64_t required[] = {4096, 8192, 512, 0};
    EXPECT_EQ(context.bind_resources_for_launch(0, 53, 123, required, 4, binding), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_EQ(binding.regions, nullptr);
    ASSERT_EQ(context.close(), 0);
}

TEST(HbgKernelResourceContext, ConcurrentPrepareAllocatesOneSlotAndCloseFailureBlocksBinding) {
    ResourceContextPlatform provider;
    KernelExecutionState context;
    ASSERT_EQ(context.initialize(0, provider.context_ops(), 59), 0);
    const hbg::GraphResourceRequirements graph{4096, 8192, 512, 2048};
    hbg::KernelResourcePlan plan;
    ASSERT_EQ(hbg::KernelResourcePlan::create(&graph, 1, plan), 0);
    auto prepare = [&] {
        return plan.prepare(context, provider.resource_ops());
    };
    auto a = std::async(std::launch::async, prepare);
    auto b = std::async(std::launch::async, prepare);
    ASSERT_EQ(a.get(), 0);
    ASSERT_EQ(b.get(), 0);
    ASSERT_EQ(provider.allocation_calls, 2);
    ASSERT_EQ(context.mark_ready_enqueued(), 0);
    ASSERT_EQ(context.freeze_resources(), 0);
    provider.free_failures = 1;
    EXPECT_EQ(context.close(), -77);
    EXPECT_EQ(provider.allocator.get_allocation_count(), 1u);
    hbg::KernelWorkingBinding binding;
    EXPECT_EQ(hbg::bind_kernel_resources_for_launch(context, 0, 59, graph, binding), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_EQ(binding.heap.address, 0u);
    ASSERT_EQ(context.close(), 0);
    EXPECT_EQ(provider.allocator.get_allocation_count(), 0u);
    EXPECT_EQ(provider.free_calls, 3);
}

}  // namespace
