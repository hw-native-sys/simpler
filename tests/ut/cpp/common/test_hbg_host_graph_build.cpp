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
#include <type_traits>
#include <vector>

#include "host_build_graph/host_graph_build.h"
#include "host_build_graph/graph_execution.h"
#include "host_build_graph/host_tensor_access.h"
#include "host_build_graph/kernel_pipeline_contract.h"
#include "host_build_graph/runtime_core.h"
#include "host_build_graph/runtime_status.h"
#include "host_build_graph/runtime.h"
#include "common/host_api.h"
#include "worker/runtime_c_api.h"

namespace simpler {
static_assert(!std::is_copy_constructible_v<hbg::GraphBuild>);
static_assert(!std::is_move_constructible_v<hbg::GraphBuild>);
}  // namespace simpler

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

void large_graph_entry(bool overflow) {
    auto &orch = *bound_runtime->orchestrator;
    const uint32_t shape[] = {16};
    auto boundary = simpler::hbg::make_tensor_external(reinterpret_cast<uint32_t *>(0x2000), shape, 1);
    GraphTaskArgs args;
    args.add_input(boundary);
    const auto scope = orch.graph_begin(0x91, args, bound_runtime->active_callable_hash);
    ASSERT_TRUE(scope.recording);
    ASSERT_TRUE(orch.graph_prepare(scope.recording_handle, args));
    for (int i = 0; i < MAX_IN_GRAPH_TASKS; ++i) {
        CoreTaskArgs task;
        task.add_input(boundary);
        ASSERT_TRUE(bound_runtime->ops->submit_dummy_task(bound_runtime, task).task_id().is_valid());
    }
    ASSERT_TRUE(orch.graph_end());
    for (uint64_t i = 1; i < READY_QUEUE_CAPACITY_LIMIT / MAX_IN_GRAPH_TASKS; ++i) {
        const auto replay = orch.graph_begin(0x91, args, bound_runtime->active_callable_hash);
        ASSERT_FALSE(replay.execute_block);
        ASSERT_TRUE(replay.task_id.is_valid());
    }
    if (overflow) {
        CoreTaskArgs task;
        task.add_input(boundary);
        ASSERT_TRUE(bound_runtime->ops->submit_dummy_task(bound_runtime, task).task_id().is_valid());
    }
}

void overflowing_graph_entry(const ChipTaskArgs &) { large_graph_entry(true); }
void queue_limit_entry(const ChipTaskArgs &) { large_graph_entry(false); }

void malformed_graph_entry(const ChipTaskArgs &args) {
    graph_entry(args);
    auto definitions = graph_host_definitions(*bound_runtime->orchestrator->graph_host_state);
    ASSERT_EQ(definitions.entries.size(), 1u);
    ASSERT_NE(definitions.entries[0].spill, nullptr);
    auto *definition = reinterpret_cast<GraphDefinition *>(const_cast<std::byte *>(definitions.entries[0].spill));
    definition->off_in_graph_tasks = UINT32_MAX;
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
    simpler::hbg::GraphBuild result;
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
        return simpler::hbg::build_graph(
            &runtime, tensor_access, {rt, mirror.data(), mirror.size(), capacity, definition_arena}, {entry, bind},
            ChipTaskArgs{}, result
        );
    }
    int32_t upload() { return simpler::hbg::upload_for_program_mode(&runtime, &api, rt, host_arena, layout, result); }

    void build_mixed_definitions() {
        ASSERT_GE(build(graph_entry), 0);
        ASSERT_GE(upload(), 0);
        platform.copies.clear();
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
    EXPECT_TRUE(result.build_complete);
    EXPECT_EQ(result.bind_usage.submitted_tasks, 2u);
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

TEST_F(HostGraphBuildTest, BuildMeasuresInGraphTasksBeforeAnyUpload) {
    ASSERT_GE(build(graph_entry), 0);
    EXPECT_EQ(result.ready_queue_populations.dummy, 2u);
    EXPECT_EQ(result.ready_queue_populations.graph_ready, 1u);
    EXPECT_EQ(result.ready_queue_capacities.dummy, 2u);
    EXPECT_GT(result.definition_bytes, 0u);
    EXPECT_EQ(platform.commits, 0);
    EXPECT_EQ(platform.definition_acquires, 0);
    EXPECT_TRUE(platform.copies.empty());
}

TEST_F(HostGraphBuildTest, GraphQueueOverflowFailsBeforeDeviceMutation) {
    EXPECT_EQ(build(overflowing_graph_entry), runtime_status_from_error_code(SIMPLER_ERROR_READY_QUEUE_OVERFLOW));
    EXPECT_FALSE(result.build_complete);
    EXPECT_EQ(platform.commits, 0);
    EXPECT_EQ(platform.definition_acquires, 0);
    EXPECT_TRUE(platform.allocations.empty());
    EXPECT_TRUE(platform.copies.empty());
    EXPECT_EQ(upload(), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_TRUE(platform.copies.empty());
    EXPECT_EQ(platform.definition_acquires, 0);
}

TEST_F(HostGraphBuildTest, ExactGraphQueueCapacitySucceeds) {
    ASSERT_GE(build(queue_limit_entry), 0);
    EXPECT_EQ(result.ready_queue_populations.dummy, READY_QUEUE_CAPACITY_LIMIT);
    EXPECT_EQ(result.ready_queue_capacities.dummy, READY_QUEUE_CAPACITY_LIMIT);
    EXPECT_TRUE(platform.copies.empty());
}

TEST_F(HostGraphBuildTest, InvalidDefinitionFailsBeforeDeviceMutation) {
    EXPECT_EQ(build(malformed_graph_entry), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_FALSE(result.build_complete);
    EXPECT_EQ(platform.definition_acquires, 0);
    EXPECT_TRUE(platform.copies.empty());
    EXPECT_EQ(upload(), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_EQ(platform.commits, 0);
}

TEST_F(HostGraphBuildTest, FailedRebuildInvalidatesPriorResultAndCanRecover) {
    ASSERT_GE(build(graph_entry), 0);
    ASSERT_GT(result.definition_bytes, 0u);
    EXPECT_LT(build(fatal_entry), 0);
    EXPECT_FALSE(result.build_complete);
    EXPECT_EQ(result.definition_bytes, 0u);
    EXPECT_EQ(result.image_bytes, 0u);
    EXPECT_EQ(upload(), PTO_RUNTIME_ERR_INTERNAL);
    ASSERT_EQ(build(chain_entry), 2);
    ASSERT_EQ(upload(), 2);
}

TEST_F(HostGraphBuildTest, UploadRejectsDifferentRuntimeBeforeDeviceMutation) {
    ASSERT_GE(build(graph_entry), 0);
    RuntimeContext other{};
    EXPECT_EQ(
        simpler::hbg::upload_for_program_mode(&runtime, &api, &other, host_arena, layout, result),
        PTO_RUNTIME_ERR_INTERNAL
    );
    EXPECT_TRUE(platform.copies.empty());
    EXPECT_EQ(platform.definition_acquires, 0);
}

TEST_F(HostGraphBuildTest, UndersizedWorkspaceFailsBeforeWritingMirror) {
    std::memset(mirror.data(), 0x5a, mirror.size());
    EXPECT_EQ(
        simpler::hbg::build_graph(
            &runtime, tensor_access, {rt, mirror.data(), 1, capacity, definition_arena}, {empty_entry, bind},
            ChipTaskArgs{}, result
        ),
        PTO_RUNTIME_ERR_INTERNAL
    );
    EXPECT_EQ(mirror.data()[0], std::byte{0x5a});
    EXPECT_FALSE(result.build_complete);
    EXPECT_TRUE(platform.copies.empty());
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
    const auto compact = sm_layout::segment_offsets(sm_layout::image_extents(result.bind_usage));
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
    EXPECT_FALSE(result.build_complete);
    EXPECT_EQ(upload(), PTO_RUNTIME_ERR_INTERNAL);
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
    EXPECT_EQ(result.definition_bytes, platform.copies[0].size());
    const auto measured_dummy = result.ready_queue_populations.dummy;
    const auto first_definition = platform.copies[0];
    const auto first_runtime = platform.copies.back();
    auto *framing = reinterpret_cast<const GraphDefinitionHeader *>(first_definition.data());
    EXPECT_EQ(framing->magic, GRAPH_DEFINITION_OBJECT_MAGIC);
    ASSERT_GE(upload(), 0);
    ASSERT_EQ(platform.copies.size(), 2 * copies_per_upload);
    EXPECT_EQ(platform.copies[copies_per_upload], first_definition);
    EXPECT_EQ(platform.copies.back(), first_runtime);
    EXPECT_EQ(result.ready_queue_populations.dummy, measured_dummy);
    EXPECT_EQ(rt->prebuilt_layout.sched.capacities.dummy, result.ready_queue_capacities.dummy);
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
    EXPECT_EQ(result.workspace.definitions.base, platform.staging.data());
    EXPECT_GE(result.workspace.definitions.capacity, result.definition_bytes);
    const auto first_definitions = platform.copies.front();
    const size_t copies_per_upload = platform.copies.size();
    ASSERT_GE(upload(), 0);
    EXPECT_EQ(platform.copies[copies_per_upload], first_definitions);
}

TEST_F(HostGraphBuildTest, FailedDefinitionCopyAfterGrowthCanBeRetried) {
    build_mixed_definitions();
    platform.fail_copy = 1;
    ASSERT_EQ(upload(), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_EQ(result.workspace.definitions.base, platform.staging.data());
    EXPECT_GE(result.workspace.definitions.capacity, result.definition_bytes);
    platform.fail_copy = 0;
    ASSERT_GE(upload(), 0);
    ASSERT_GE(upload(), 0);
}

TEST_F(HostGraphBuildTest, KernelRequirementsConsumeH1MeasurementsWithoutDeviceMutation) {
    ASSERT_GE(build(graph_entry), 0);
    simpler::hbg::GraphResourceRequirements requirements;
    ASSERT_EQ(simpler::hbg::get_graph_resource_requirements(result, layout, requirements), 0);

    EXPECT_EQ(requirements.layout.task_capacity, result.workspace.task_capacity);
    EXPECT_EQ(requirements.layout.copied_begin, layout.off_copied_begin);
    EXPECT_EQ(requirements.layout.copied_end, layout.off_copied_end);
    EXPECT_EQ(requirements.gm_heap_bytes, result.heap_bytes);
    EXPECT_EQ(requirements.runtime_image_bytes, layout.off_copied_end + result.image_bytes);
    EXPECT_EQ(requirements.graph_definition_bytes, result.definition_bytes);
    EXPECT_EQ(requirements.scheduler_state_bytes, 0u);

    simpler::hbg::GraphCapacityPlan plan;
    ASSERT_EQ(simpler::hbg::GraphCapacityPlan::create(&requirements, 1, plan), 0);
    EXPECT_TRUE(plan.admits(requirements));
    const PipelineContract contract = plan.pipeline_contract();
    EXPECT_TRUE(simpler::hbg::is_valid_kernel_pipeline_contract(&contract));
    EXPECT_TRUE(platform.allocations.empty());
    EXPECT_TRUE(platform.copies.empty());
    EXPECT_EQ(platform.commits, 0);
    EXPECT_EQ(platform.definition_acquires, 0);
}

TEST_F(HostGraphBuildTest, KernelRequirementFailureDoesNotPublishPartialOutput) {
    simpler::hbg::GraphResourceRequirements output;
    output.gm_heap_bytes = 11;
    output.runtime_image_bytes = 22;
    EXPECT_EQ(simpler::hbg::get_graph_resource_requirements(result, layout, output), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_EQ(output.gm_heap_bytes, 11u);
    EXPECT_EQ(output.runtime_image_bytes, 22u);

    ASSERT_EQ(build(chain_entry), 2);
    auto incompatible_layout = layout;
    ++incompatible_layout.task_capacity;
    EXPECT_EQ(
        simpler::hbg::get_graph_resource_requirements(result, incompatible_layout, output),
        PTO_RUNTIME_ERR_INVALID_ARGUMENT
    );
    EXPECT_EQ(output.gm_heap_bytes, 11u);
    EXPECT_EQ(output.runtime_image_bytes, 22u);
    EXPECT_TRUE(platform.allocations.empty());
    EXPECT_TRUE(platform.copies.empty());
    EXPECT_EQ(platform.commits, 0);
}

TEST_F(HostGraphBuildTest, A5SchedulerCapacityUsesTheH1TaskCountUpperBound) {
    ASSERT_EQ(build(chain_entry), 2);
    simpler::hbg::GraphResourceRequirements requirements;
    ASSERT_EQ(simpler::hbg::get_graph_resource_requirements(result, layout, requirements), 0);
    if (requirements.layout.architecture == simpler::hbg::RuntimeArchitecture::A5) {
        EXPECT_GT(requirements.scheduler_state_bytes, 0u);
    } else {
        EXPECT_EQ(requirements.scheduler_state_bytes, 0u);
    }
    EXPECT_TRUE(platform.allocations.empty());
    EXPECT_TRUE(platform.copies.empty());
}

TEST(HbgGraphCapacityPlan, TakesRegionMaximaThenBuildsOneAlignedSlot) {
    const simpler::hbg::RuntimeLayoutKey layout{
        simpler::hbg::HBG_RUNTIME_LAYOUT_ABI_VERSION, simpler::hbg::RuntimeArchitecture::A2A3, 64, 4096, 2048, 4096,
    };
    const simpler::hbg::GraphResourceRequirements graphs[] = {
        {layout, 4096, 8192, 512, 0},
        {layout, 8192, 4096, 0, 2048},
    };
    simpler::hbg::GraphCapacityPlan plan;
    ASSERT_EQ(simpler::hbg::GraphCapacityPlan::create(graphs, 2, plan), 0);
    EXPECT_EQ(plan.capacity().gm_heap_bytes, 8192u);
    EXPECT_EQ(plan.capacity().runtime_image_bytes, 8192u);
    EXPECT_EQ(plan.definition_offset(), 8192u);
    EXPECT_EQ(plan.scheduler_offset(), 9216u);
    EXPECT_EQ(plan.runtime_slot_bytes(), 11264u);
    EXPECT_TRUE(plan.admits(graphs[0]));
    EXPECT_TRUE(plan.admits(graphs[1]));

    const PipelineContract contract = plan.pipeline_contract();
    ASSERT_TRUE(simpler::hbg::is_valid_kernel_pipeline_contract(&contract));
    EXPECT_EQ(find_pipeline_resource(contract, PTO_PIPELINE_GM_HEAP)->bytes_per_copy, 8192u);
    EXPECT_EQ(find_pipeline_resource(contract, PTO_PIPELINE_RUNTIME_IMAGE)->bytes_per_copy, 11264u);
}

TEST(HbgGraphCapacityPlan, RejectsLayoutMixingAndOverflowWithoutChangingThePlan) {
    const simpler::hbg::RuntimeLayoutKey a2a3{
        simpler::hbg::HBG_RUNTIME_LAYOUT_ABI_VERSION, simpler::hbg::RuntimeArchitecture::A2A3, 64, 4096, 2048, 4096,
    };
    auto a5 = a2a3;
    a5.architecture = simpler::hbg::RuntimeArchitecture::A5;
    const simpler::hbg::GraphResourceRequirements good{a2a3, 4096, 8192, 512, 0};
    simpler::hbg::GraphCapacityPlan plan;
    ASSERT_EQ(simpler::hbg::GraphCapacityPlan::create(&good, 1, plan), 0);
    const PipelineContract original = plan.pipeline_contract();

    const simpler::hbg::GraphResourceRequirements mixed[] = {good, {a5, 4096, 8192, 512, 0}};
    const simpler::hbg::GraphResourceRequirements overflow{a2a3, 1, UINT64_MAX - 511, 512, 0};
    EXPECT_EQ(simpler::hbg::GraphCapacityPlan::create(mixed, 2, plan), PTO_RUNTIME_ERR_INVALID_ARGUMENT);
    EXPECT_EQ(simpler::hbg::GraphCapacityPlan::create(&overflow, 1, plan), PTO_RUNTIME_ERR_INVALID_ARGUMENT);
    const PipelineContract after = plan.pipeline_contract();
    EXPECT_EQ(std::memcmp(&after, &original, sizeof(original)), 0);
    EXPECT_TRUE(plan.admits(good));
    EXPECT_FALSE(plan.admits(mixed[1]));
}

TEST(HbgKernelStreamBindings, RequiresThreeDistinctStreamRolesAndPublishesAtomically) {
    const simpler::hbg::RuntimeLayoutKey layout{
        simpler::hbg::HBG_RUNTIME_LAYOUT_ABI_VERSION, simpler::hbg::RuntimeArchitecture::A2A3, 64, 4096, 2048, 4096,
    };
    const simpler::hbg::GraphResourceRequirements graph{layout, 4096, 8192, 0, 0};
    simpler::hbg::GraphCapacityPlan plan;
    ASSERT_EQ(simpler::hbg::GraphCapacityPlan::create(&graph, 1, plan), 0);
    const PipelineContract contract = plan.pipeline_contract();
    int caller = 0;
    int aicpu = 0;
    int aicore = 0;
    simpler::hbg::KernelStreamBindings bindings;
    ASSERT_EQ(simpler::hbg::bind_kernel_streams(&contract, &caller, &aicpu, &aicore, bindings), 0);
    EXPECT_EQ(bindings.caller_stream, &caller);
    EXPECT_EQ(bindings.aicpu_stream, &aicpu);
    EXPECT_EQ(bindings.aicore_stream, &aicore);

    EXPECT_EQ(
        simpler::hbg::bind_kernel_streams(&contract, &caller, &caller, &aicore, bindings),
        PTO_RUNTIME_ERR_INVALID_ARGUMENT
    );
    EXPECT_EQ(bindings.caller_stream, &caller);
    EXPECT_EQ(bindings.aicpu_stream, &aicpu);
    EXPECT_EQ(bindings.aicore_stream, &aicore);
}

}  // namespace
