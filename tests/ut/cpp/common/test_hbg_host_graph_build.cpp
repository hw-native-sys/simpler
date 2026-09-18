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
#include <cstring>
#include <future>
#include <vector>

#include "host_build_graph/host_graph_build.h"
#include "host_build_graph/kernel_external_tensor.h"
#include "host_build_graph/kernel_argument_snapshot.h"
#include "host_build_graph/kernel_graph_template.h"
#include "host_build_graph/kernel_graph_slot.h"
#include "host_build_graph/kernel_graph_slot_registry.h"
#include "host_build_graph/kernel_graph_restore.h"
#include "host_build_graph/graph_execution.h"
#include "host_build_graph/host_tensor_access.h"
#include "host_build_graph/runtime_core.h"
#include "host_build_graph/runtime.h"
#include "common/host_api.h"
#include "worker/runtime_c_api.h"
#include "host/kernel_pipeline_contract.h"
#include "call_config.h"
#include "orchestration_requirements.h"

extern "C" const char *const *runtime_extra_aicpu_symbols(size_t *count);
extern "C" const char *const *runtime_l1_extra_aicpu_symbols(size_t *count);

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
    auto boundary = simpler::hbg::make_tensor_external(
        reinterpret_cast<uint32_t *>(0x2000), shape, 1, DataType::FLOAT32, false, 0, AddressSpace::DEVICE
    );
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
    auto boundary = simpler::hbg::make_tensor_external(
        reinterpret_cast<uint32_t *>(0x2000), shape, 1, DataType::FLOAT32, false, 0, AddressSpace::DEVICE
    );
    GraphTaskArgs graph_args;
    graph_args.add_input(boundary);
    const auto scope = bound_runtime->orchestrator->graph_begin(0x81, graph_args, bound_runtime->active_callable_hash);
    EXPECT_FALSE(scope.execute_block);
    EXPECT_TRUE(scope.task_id.is_valid());
}

void two_definitions_entry(const ChipTaskArgs &args) {
    graph_entry(args);
    const uint32_t shape[] = {16};
    auto boundary = simpler::hbg::make_tensor_external(
        reinterpret_cast<uint32_t *>(0x2000), shape, 1, DataType::FLOAT32, false, 0, AddressSpace::DEVICE
    );
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

TEST(HbgKernelExternalTensor, RequirementsFailClosedAndAllowExplicitHostReads) {
    using simpler::orchestration::HbgKernelRequirementsStatus;
    using simpler::orchestration::REQUIREMENT_TENSOR_DATA_READ;
    using simpler::orchestration::REQUIREMENT_TENSOR_DATA_WRITE;
    using simpler::orchestration::validate_hbg_kernel_requirements;

    EXPECT_EQ(validate_hbg_kernel_requirements(false, 0, 0), HbgKernelRequirementsStatus::MetadataUnavailable);
    EXPECT_EQ(
        validate_hbg_kernel_requirements(true, UINT64_C(1) << 63, 0), HbgKernelRequirementsStatus::UnknownRequirement
    );
    EXPECT_EQ(
        validate_hbg_kernel_requirements(true, REQUIREMENT_TENSOR_DATA_WRITE, 1),
        HbgKernelRequirementsStatus::TensorDataWriteUnsupported
    );
    EXPECT_EQ(
        validate_hbg_kernel_requirements(true, REQUIREMENT_TENSOR_DATA_READ, 0),
        HbgKernelRequirementsStatus::HostCopyRequired
    );
    EXPECT_EQ(validate_hbg_kernel_requirements(true, REQUIREMENT_TENSOR_DATA_READ, 1), HbgKernelRequirementsStatus::Ok);
    EXPECT_EQ(validate_hbg_kernel_requirements(true, 0, 0), HbgKernelRequirementsStatus::Ok);
}

TEST(HbgKernelExternalTensor, HostCopySuffixIsTheOnlyHostReadableStorage) {
    const uint32_t data_shape[] = {8};
    const uint32_t table_shape[] = {4};
    std::array<int32_t, 4> table_host{3, 1, 4, 1};
    ChipStorageTaskArgs args;
    args.add_tensor(
        make_tensor_external(reinterpret_cast<void *>(0x22000), data_shape, 1, DataType::FLOAT32, AddressSpace::DEVICE)
    );
    args.add_tensor(
        make_tensor_external(reinterpret_cast<void *>(0x33000), table_shape, 1, DataType::INT32, AddressSpace::DEVICE)
    );
    args.add_tensor(make_tensor_external(table_host.data(), table_shape, 1, DataType::INT32, AddressSpace::HOST));

    HostTensorAccessor accessor(nullptr, HostTensorAccessMode::KernelHostCopiesOnly);
    hbg::HostOrchEntryPoints entry_points{};
    entry_points.requirements_v1_available = true;
    entry_points.requirements_v1 = simpler::orchestration::REQUIREMENT_TENSOR_DATA_READ;
    ASSERT_EQ(
        hbg::prepare_kernel_external_tensors(args, 1, entry_points, accessor), hbg::KernelExternalTensorStatus::Ok
    );
    int32_t value = 0;
    EXPECT_FALSE(host_tensor_read(&accessor, 0x33000, &value, sizeof(value)));
    ASSERT_TRUE(host_tensor_read(&accessor, reinterpret_cast<uintptr_t>(table_host.data() + 2), &value, sizeof(value)));
    EXPECT_EQ(value, 4);
    const int32_t replacement = 9;
    EXPECT_FALSE(
        host_tensor_write(&accessor, reinterpret_cast<uintptr_t>(table_host.data()), &replacement, sizeof(replacement))
    );
    EXPECT_EQ(table_host[0], 3);
    EXPECT_EQ(accessor.mapping_count(), 0u);
    EXPECT_EQ(accessor.device_copy_count(), 0u);
}

TEST(HbgKernelExternalTensor, RejectsHostDeviceArgsUnsupportedStridesAndMismatchedCopies) {
    const uint32_t shape[] = {2, 3};
    const uint32_t transposed_stride[] = {1, 2};
    std::array<float, 6> host{};
    ChipStorageTaskArgs args;
    args.add_tensor(make_tensor_external(host.data(), shape, 2, DataType::FLOAT32, AddressSpace::HOST));
    EXPECT_EQ(hbg::validate_kernel_external_tensors(args, 0), hbg::KernelExternalTensorStatus::NonDeviceTensor);

    args.clear();
    args.add_tensor(make_tensor_strided(
        reinterpret_cast<void *>(0x44000), shape, transposed_stride, 2, DataType::FLOAT32, AddressSpace::DEVICE
    ));
    EXPECT_EQ(hbg::validate_kernel_external_tensors(args, 0), hbg::KernelExternalTensorStatus::UnsupportedStrideFamily);

    const uint32_t device_shape[] = {4};
    const uint32_t host_shape[] = {2};
    args.clear();
    args.add_tensor(
        make_tensor_external(reinterpret_cast<void *>(0x55000), device_shape, 1, DataType::INT32, AddressSpace::DEVICE)
    );
    args.add_tensor(make_tensor_external(host.data(), host_shape, 1, DataType::INT32, AddressSpace::HOST));
    EXPECT_EQ(hbg::validate_kernel_external_tensors(args, 1), hbg::KernelExternalTensorStatus::HostCopyMismatch);
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
            &runtime, tensor_access, {rt, mirror.data(), mirror.size(), capacity, definition_arena}, {entry, bind},
            ChipTaskArgs{}, result
        );
    }
    int32_t upload() { return hbg::upload_for_program_mode(&runtime, &api, rt, host_arena, layout, result); }

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

TEST_F(HostGraphBuildTest, ExactGraphQueueCapacitySucceeds) {
    ASSERT_GE(build(queue_limit_entry), 0);
    EXPECT_EQ(result.ready_queue_populations.dummy, READY_QUEUE_CAPACITY_LIMIT);
    EXPECT_EQ(result.ready_queue_capacities.dummy, READY_QUEUE_CAPACITY_LIMIT);
    EXPECT_TRUE(platform.copies.empty());
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

TEST_F(HostGraphBuildTest, FailedRebuildInvalidatesPriorResultAndCanRecover) {
    ASSERT_GE(build(graph_entry), 0);
    ASSERT_GT(result.definition_bytes, 0u);
    EXPECT_LT(build(fatal_entry), 0);
    EXPECT_FALSE(result.build_complete);
    EXPECT_EQ(result.definition_bytes, 0u);
    EXPECT_EQ(result.image_bytes, 0u);
    EXPECT_EQ(upload(), PTO_RUNTIME_ERR_INVALID_STATE);
    ASSERT_EQ(build(chain_entry), 2);
    ASSERT_EQ(upload(), 2);
}

TEST_F(HostGraphBuildTest, GraphQueueOverflowFailsBeforeDeviceMutation) {
    EXPECT_EQ(build(overflowing_graph_entry), runtime_status_from_error_code(SIMPLER_ERROR_READY_QUEUE_OVERFLOW));
    EXPECT_FALSE(result.build_complete);
    EXPECT_EQ(platform.commits, 0);
    EXPECT_EQ(platform.definition_acquires, 0);
    EXPECT_TRUE(platform.allocations.empty());
    EXPECT_TRUE(platform.copies.empty());
    EXPECT_EQ(upload(), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_TRUE(platform.copies.empty());
    EXPECT_EQ(platform.definition_acquires, 0);
}

TEST_F(HostGraphBuildTest, InvalidDefinitionFailsBeforeDeviceMutation) {
    EXPECT_EQ(build(malformed_graph_entry), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_FALSE(result.build_complete);
    EXPECT_EQ(platform.definition_acquires, 0);
    EXPECT_TRUE(platform.copies.empty());
    EXPECT_EQ(upload(), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_EQ(platform.commits, 0);
}

TEST_F(HostGraphBuildTest, UndersizedWorkspaceFailsBeforeWritingMirror) {
    std::memset(mirror.data(), 0x5a, mirror.size());
    EXPECT_EQ(
        hbg::build_graph(
            &runtime, tensor_access, {rt, mirror.data(), 1, capacity, definition_arena}, {empty_entry, bind},
            ChipTaskArgs{}, result
        ),
        PTO_RUNTIME_ERR_INTERNAL
    );
    EXPECT_EQ(mirror.data()[0], std::byte{0x5a});
    EXPECT_FALSE(result.build_complete);
    EXPECT_TRUE(platform.copies.empty());
}

TEST_F(HostGraphBuildTest, UploadRejectsDifferentRuntimeBeforeDeviceMutation) {
    ASSERT_GE(build(graph_entry), 0);
    RuntimeContext other{};
    EXPECT_EQ(
        hbg::upload_for_program_mode(&runtime, &api, &other, host_arena, layout, result), PTO_RUNTIME_ERR_INVALID_STATE
    );
    EXPECT_TRUE(platform.copies.empty());
    EXPECT_EQ(platform.definition_acquires, 0);
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

hbg::GraphResourceRequirements resource_requirements(
    uint64_t heap, uint64_t runtime, uint64_t definitions, uint64_t scheduler,
    hbg::RuntimeArchitecture architecture = hbg::RuntimeArchitecture::A2A3
) {
    hbg::GraphResourceRequirements out{heap, runtime, definitions, scheduler};
    out.layout = {hbg::HBG_RUNTIME_LAYOUT_ABI_VERSION, architecture, 64, 8192, 4096, 8192};
    return out;
}

TEST(HbgKernelResourcePlan, RebuildsDisjointRegionsFromCompatibleGraphMaxima) {
    const hbg::GraphResourceRequirements graphs[] = {
        resource_requirements(4096, 8192, 512, 0), resource_requirements(8192, 4096, 0, 2048)
    };
    hbg::KernelResourcePlan plan;
    ASSERT_EQ(hbg::KernelResourcePlan::create(graphs, 2, plan), 0);
    EXPECT_EQ(plan.capacity().gm_heap_bytes, 8192u);
    EXPECT_EQ(plan.capacity().runtime_arena_bytes, 8192u);
    EXPECT_EQ(plan.definition_offset(), 8192u);
    EXPECT_EQ(plan.scheduler_offset(), 9216u);
    EXPECT_EQ(plan.registry_offset(), 11264u);
    EXPECT_EQ(plan.runtime_arena_bytes(), 11264u + sizeof(hbg::GraphSlotRegistry));
    EXPECT_TRUE(plan.admits(graphs[0]));
    EXPECT_TRUE(plan.admits(graphs[1]));
    const auto contract = plan.pipeline_contract();
    ASSERT_TRUE(is_valid_hbg_kernel_pipeline_contract(&contract));
    EXPECT_EQ(find_pipeline_resource(contract, PTO_PIPELINE_GM_HEAP)->bytes_per_copy, 8192u);
    EXPECT_EQ(
        find_pipeline_resource(contract, PTO_PIPELINE_RUNTIME_IMAGE)->bytes_per_copy,
        11264u + sizeof(hbg::GraphSlotRegistry)
    );
}

TEST(HbgKernelResourcePlan, RejectsDifferentArchitectureOrRuntimeLayout) {
    const auto original = resource_requirements(4096, 8192, 512, 0);
    hbg::KernelResourcePlan plan;
    ASSERT_EQ(hbg::KernelResourcePlan::create(&original, 1, plan), 0);

    auto other_arch = original;
    other_arch.layout.architecture = hbg::RuntimeArchitecture::A5;
    const hbg::GraphResourceRequirements mixed[] = {original, other_arch};
    EXPECT_EQ(hbg::KernelResourcePlan::create(mixed, 2, plan), PTO_RUNTIME_ERR_INVALID_ARGUMENT);
    EXPECT_FALSE(plan.admits(other_arch));

    auto other_window = original;
    ++other_window.layout.task_capacity;
    EXPECT_EQ(hbg::KernelResourcePlan::create(&other_window, 1, plan), 0);
    EXPECT_FALSE(plan.admits(original));

    auto invalid = original;
    invalid.layout.abi_version = 0;
    EXPECT_EQ(hbg::KernelResourcePlan::create(&invalid, 1, plan), PTO_RUNTIME_ERR_INVALID_ARGUMENT);
}

TEST(HbgKernelResourcePlan, RejectsEachRegionOverCapacityWithoutChangingThePlan) {
    const auto graph = resource_requirements(4096, 8192, 512, 2048);
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
    const auto good = resource_requirements(4096, 8192, 512, 0);
    hbg::KernelResourcePlan plan;
    ASSERT_EQ(hbg::KernelResourcePlan::create(&good, 1, plan), 0);
    const auto original = plan.pipeline_contract();
    const hbg::GraphResourceRequirements aggregate[] = {
        resource_requirements(UINT64_MAX / 2, 1, 0, 0), resource_requirements(1, UINT64_MAX / 2 + 2, 0, 0)
    };
    const auto alignment = resource_requirements(1, UINT64_MAX - 512, 16, 0);
    for (const auto &graph : aggregate) {
        uint64_t total;
        ASSERT_TRUE(graph.required_bytes(total));
    }
    EXPECT_EQ(hbg::KernelResourcePlan::create(aggregate, 2, plan), PTO_RUNTIME_ERR_CAPACITY_EXCEEDED);
    EXPECT_EQ(hbg::KernelResourcePlan::create(&alignment, 1, plan), PTO_RUNTIME_ERR_CAPACITY_EXCEEDED);
    EXPECT_EQ(hbg::KernelResourcePlan::create(nullptr, 1, plan), PTO_RUNTIME_ERR_INVALID_ARGUMENT);
    EXPECT_EQ(hbg::KernelResourcePlan::create(&good, 0, plan), PTO_RUNTIME_ERR_INVALID_ARGUMENT);
    const auto after = plan.pipeline_contract();
    EXPECT_EQ(std::memcmp(&after, &original, sizeof(after)), 0);
    EXPECT_EQ(plan.definition_offset(), 8192u);
    EXPECT_TRUE(plan.admits(good));
}

TEST(HbgKernelStreamBinding, BindsThreeDistinctStreamsForBothRuntimeContracts) {
    const auto graph = resource_requirements(4096, 8192, 0, 0);
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
    const auto graph = resource_requirements(4096, 8192, 0, 0);
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
    std::vector<void *> streams;
    MemoryAllocator allocator;

    KernelContextOps context_ops() {
        return {
            this,
            [](void *ctx, int *device) noexcept {
                *device = static_cast<ResourceContextPlatform *>(ctx)->current_device;
                return 0;
            },
            [](void *ctx, void **stream) noexcept {
                auto &self = *static_cast<ResourceContextPlatform *>(ctx);
                *stream = reinterpret_cast<void *>(self.next_handle++);
                self.streams.push_back(*stream);
                return 0;
            },
            [](void *, void *) noexcept {
                return 0;
            },
            [](void *ctx, void **event) noexcept {
                *event = reinterpret_cast<void *>(static_cast<ResourceContextPlatform *>(ctx)->next_handle++);
                return 0;
            },
            [](void *, void *) noexcept {
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
    ASSERT_EQ(provider.streams.size(), 2u);
    EXPECT_EQ(provider.streams[0], context.hidden_stream(KernelStreamKind::Aicpu));
    EXPECT_EQ(provider.streams[1], context.hidden_stream(KernelStreamKind::Aicore));
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
    const hbg::GraphResourceRequirements graphs[] = {
        resource_requirements(4096, 8192, 512, 0), resource_requirements(8192, 4096, 0, 2048)
    };
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
    const auto graph = resource_requirements(4096, 8192, 0, 0);
    hbg::KernelResourcePlan plan;
    ASSERT_EQ(hbg::KernelResourcePlan::create(&graph, 1, plan), 0);
    EXPECT_EQ(plan.prepare(context, provider.resource_ops()), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_EQ(context.initialize(0, provider.context_ops(), 0), PTO_RUNTIME_ERR_INVALID_ARGUMENT);
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
        const auto graph = resource_requirements(4096, 8192, 512, 0);
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
    const auto graph = resource_requirements(4096, 8192, 0, 0);
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
    const auto graph = resource_requirements(4096, 8192, 512, 2048);
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
    const auto graph = resource_requirements(4096, 8192, 512, 0);
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
    outside.regions[2].bytes = plan.runtime_arena_bytes() - outside.regions[2].offset + 1;
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
    const auto graph = resource_requirements(4096, 8192, 512, 2048);
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

class HbgGraphPacketTest : public HostGraphBuildTest {
protected:
    ResourceContextPlatform provider;
    KernelExecutionState context;
    hbg::GraphLaunchTemplate snapshot;
    hbg::GraphInvocationIdentity identity{7, 1, 2, 101, 103, 107};
    hbg::GraphResourceRequirements required;
    hbg::KernelWorkingBinding binding;

    void TearDown() override { EXPECT_EQ(context.close(), 0); }
    void prepare(uint64_t spare = 0, uint64_t scheduler_capacity = 0) {
        RuntimeArenaLayout kernel_layout{};
        ASSERT_EQ(hbg::make_kernel_graph_layout(capacity, kernel_layout), 0);
        ASSERT_EQ(hbg::get_graph_resource_requirements(result, kernel_layout, required), 0);
        auto room = required;
        room.gm_heap_bytes += spare;
        room.scheduler_state_bytes = std::max(room.scheduler_state_bytes, scheduler_capacity);
        room.runtime_arena_bytes += spare;
        room.graph_definition_bytes += spare;
        hbg::KernelResourcePlan plan;
        ASSERT_EQ(hbg::KernelResourcePlan::create(&room, 1, plan), 0);
        ASSERT_EQ(context.initialize(0, provider.context_ops(), 19), 0);
        ASSERT_EQ(plan.prepare(context, provider.resource_ops()), 0);
        ASSERT_EQ(context.mark_ready_enqueued(), 0);
        ASSERT_EQ(context.freeze_resources(), 0);
        ASSERT_EQ(hbg::bind_kernel_resources_for_launch(context, 0, 19, required, binding), 0);
    }
    int snapshot_graph() {
        return hbg::make_graph_launch_template(result, *rt, context, 0, 19, 109, identity, snapshot);
    }
    std::vector<std::byte> bytes() const {
        const auto *data = static_cast<const std::byte *>(snapshot.data());
        return {data, data + snapshot.size()};
    }
    hbg::GraphPacketHeader header() const {
        hbg::GraphPacketHeader out{};
        std::memcpy(
            &out, static_cast<const std::byte *>(snapshot.data()) + sizeof(SimplerKernelInvocationHeader), sizeof(out)
        );
        return out;
    }
};

TEST(HbgGraphPacket, CompactLayoutFollowsWindowAndPreservesProgramSizing) {
    for (uint64_t window : {1, 2, 63, 64, 65, 128, 32768}) {
        RuntimeArenaLayout layout{};
        ASSERT_EQ(hbg::make_kernel_graph_layout(window, layout), 0);
        uint64_t expected = 64;
        while (expected < window)
            expected <<= 1;
        EXPECT_EQ(layout.task_capacity, window);
        EXPECT_EQ(layout.sched.capacities.dummy, expected);
        for (int i = 0; i < NUM_RESOURCE_SHAPES; ++i) {
            EXPECT_EQ(layout.sched.capacities.ready[i], expected);
            EXPECT_EQ(layout.sched.capacities.ready_sync[i], expected);
        }
        DeviceArena program;
        const auto original = runtime_reserve_layout(program, window);
        EXPECT_EQ(original.sched.capacities.dummy, READY_QUEUE_CAPACITY_LIMIT);
        EXPECT_LE(layout.arena_size, original.arena_size);
        if (window < 32768) EXPECT_LT(layout.arena_size, original.arena_size);
    }
    RuntimeArenaLayout unchanged{};
    unchanged.task_capacity = 777;
    for (uint64_t window : {uint64_t{0}, uint64_t{32769}, UINT64_MAX}) {
        EXPECT_EQ(hbg::make_kernel_graph_layout(window, unchanged), PTO_RUNTIME_ERR_CAPACITY_EXCEEDED);
        EXPECT_EQ(unchanged.task_capacity, 777u);
    }
}

TEST_F(HbgGraphPacketTest, SnapshotOwnsImagesAndPreservesSourceAndDeviceMemory) {
    ASSERT_EQ(build(chain_entry), 2);
    prepare(1024);
    std::memset(reinterpret_cast<void *>(binding.runtime_image.address), 0x5a, binding.runtime_image.capacity);
    const std::vector<std::byte> source(mirror.data(), mirror.data() + mirror.size());
    ASSERT_EQ(snapshot_graph(), 0);
    const auto original = bytes();
    EXPECT_EQ(std::memcmp(source.data(), mirror.data(), mirror.size()), 0);
    ASSERT_EQ(snapshot_graph(), 0);
    EXPECT_EQ(bytes(), original);
    EXPECT_EQ(provider.allocation_calls, 2);
    EXPECT_EQ(provider.free_calls, 0);
    EXPECT_TRUE(platform.copies.empty());
    EXPECT_EQ(platform.commits, 0);
    EXPECT_EQ(platform.definition_acquires, 0);
    const auto *device = reinterpret_cast<const uint8_t *>(binding.runtime_image.address);
    EXPECT_TRUE(std::all_of(device, device + binding.runtime_image.capacity, [](auto b) {
        return b == 0x5a;
    }));
    result.graph_state.reset();
    std::memset(mirror.data(), 0xab, mirror.size());
    EXPECT_EQ(
        hbg::validate_graph_packet(snapshot.data(), snapshot.size(), hbg::GraphPacketAddress::HostTemplate),
        hbg::GraphPacketStatus::Ok
    );
    hbg::GraphHostArgs args;
    ASSERT_EQ(hbg::make_graph_host_args(snapshot, args), 0);
    EXPECT_EQ(bytes(), original);
    EXPECT_EQ(std::memcmp(args.storage.data(), original.data(), args.bytes), 0);
}

TEST_F(HbgGraphPacketTest, DefinitionSnapshotsIncludeRetainedSpilledAndRepeatedGraphs) {
    build_mixed_definitions();
    prepare(1024);
    ASSERT_EQ(snapshot_graph(), 0);
    const auto h = header();
    EXPECT_EQ(h.total_tasks, result.total_tasks);
    EXPECT_EQ(h.destinations[2].capacity, binding.definitions.capacity);
    hbg::GraphImageRegion definitions{};
    const auto *packet = static_cast<const std::byte *>(snapshot.data());
    std::memcpy(
        &definitions, packet + sizeof(SimplerKernelInvocationHeader) + sizeof(h) + sizeof(hbg::GraphImageRegion),
        sizeof(definitions)
    );
    EXPECT_EQ(definitions.kind, hbg::GraphImageKind::Definitions);
    const auto *image = packet + sizeof(SimplerKernelInvocationHeader) + h.payload_offset + definitions.source_offset;
    const auto all = graph_host_definitions(*result.graph_state);
    for (const auto &entry : all.entries) {
        // Both in-place and spill objects are found through task-local binding.
        bool found = false;
        for (uint64_t offset = 0; offset + sizeof(GraphDefinitionHeader) <= required.graph_definition_bytes;
             offset += GRAPH_DEFINITION_OBJECT_ALIGN) {
            GraphDefinitionHeader framing{};
            std::memcpy(&framing, image + offset, sizeof(framing));
            if (framing.magic != GRAPH_DEFINITION_OBJECT_MAGIC || framing.full_key != entry.full_key) continue;
            found = true;
            EXPECT_EQ(framing.definition_bytes, entry.bytes);
            EXPECT_EQ(
                std::memcmp(
                    image + offset + sizeof(framing), graph_host_definition_data(*result.graph_state, entry.full_key),
                    entry.bytes
                ),
                0
            );
        }
        EXPECT_TRUE(found);
    }
    const auto saved = bytes();
    ASSERT_GE(build(repeated_graph_entry), 0);
    ASSERT_EQ(snapshot_graph(), 0);
    EXPECT_NE(bytes(), saved);
    EXPECT_EQ(
        hbg::validate_graph_packet(saved.data(), saved.size(), hbg::GraphPacketAddress::HostTemplate),
        hbg::GraphPacketStatus::Ok
    );
}

TEST_F(HbgGraphPacketTest, SmallerGraphKeepsFrozenCapacityAndTransmitsOnlyLiveImage) {
    ASSERT_EQ(build(chain_entry), 2);
    prepare(1024);
    ASSERT_EQ(snapshot_graph(), 0);
    const auto larger = bytes();
    const auto large_header = header();
    ASSERT_EQ(build(empty_entry), 0);
    ASSERT_EQ(snapshot_graph(), 0);
    const auto h = header();
    EXPECT_EQ(h.total_tasks, 0u);
    EXPECT_LT(h.total_bytes, large_header.total_bytes);
    EXPECT_EQ(h.destinations[1].address, large_header.destinations[1].address);
    EXPECT_EQ(h.destinations[1].capacity, large_header.destinations[1].capacity);
    const auto *payload =
        static_cast<const std::byte *>(snapshot.data()) + sizeof(SimplerKernelInvocationHeader) + h.payload_offset;
    hbg::GraphImageRegion image{};
    std::memcpy(
        &image, static_cast<const std::byte *>(snapshot.data()) + sizeof(SimplerKernelInvocationHeader) + sizeof(h),
        sizeof(image)
    );
    EXPECT_EQ(image.bytes, h.sm_offset + result.image_bytes);
    RuntimeContext pristine{};
    std::memcpy(&pristine, payload + h.runtime_offset, sizeof(pristine));
    EXPECT_EQ(pristine.ops, nullptr);
    EXPECT_EQ(pristine.orchestrator, nullptr);
    EXPECT_EQ(pristine.tensor_access, nullptr);
    EXPECT_EQ(pristine.sm_handle, nullptr);
    EXPECT_EQ(pristine.scheduler, nullptr);
    EXPECT_EQ(pristine.aicore_mailbox, nullptr);
    EXPECT_EQ(pristine.prebuilt_layout.task_capacity, capacity);
    EXPECT_EQ(provider.allocation_calls, 2);
    EXPECT_EQ(
        hbg::validate_graph_packet(larger.data(), larger.size(), hbg::GraphPacketAddress::HostTemplate),
        hbg::GraphPacketStatus::Ok
    );
}

TEST_F(HbgGraphPacketTest, HeapAndDependencyReferencesSurviveCopyingThePacket) {
    ASSERT_EQ(build(chain_entry), 2);
    prepare();
    const auto from = sm_layout::segment_offsets(capacity);
    const auto *source = reinterpret_cast<const ChipTaskStorage *>(mirror.data() + from.storage);
    const auto virtual_output = source[0].payload.tensor_data()[0].buffer.addr;
    ASSERT_GE(virtual_output, HEAP_VIRTUAL_BASE);
    ASSERT_GT(source[1].payload.fanin_count, 0);
    ASSERT_EQ(snapshot_graph(), 0);
    const auto h = header();
    Buffer relocated;
    relocated.reserve(binding.runtime_image.capacity);
    std::memcpy(
        relocated.data(),
        static_cast<const std::byte *>(snapshot.data()) + sizeof(SimplerKernelInvocationHeader) + h.payload_offset,
        h.sm_offset + result.image_bytes
    );
    const auto to = sm_layout::segment_offsets(sm_layout::image_extents(result.bind_usage));
    const auto *tasks = reinterpret_cast<const ChipTaskStorage *>(relocated.data() + h.sm_offset + to.storage);
    EXPECT_EQ(tasks[0].payload.tensor_data()[0].buffer.addr, binding.heap.address + virtual_output - HEAP_VIRTUAL_BASE);
    EXPECT_EQ(tasks[1].payload.fanin_count, source[1].payload.fanin_count);
    EXPECT_EQ(
        std::memcmp(
            tasks[1].payload.fanin_data(), source[1].payload.fanin_data(),
            tasks[1].payload.fanin_count * sizeof(int32_t)
        ),
        0
    );
    EXPECT_EQ(source[0].payload.tensor_data()[0].buffer.addr, virtual_output);
    auto moved = std::move(snapshot);
    EXPECT_EQ(snapshot.size(), 0u);
    hbg::GraphHostArgs args;
    EXPECT_EQ(hbg::make_graph_host_args(snapshot, args), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_EQ(hbg::make_graph_host_args(moved, args), 0);
}

TEST_F(HbgGraphPacketTest, PlaceholderUsesEnvelopeOffsetsAndOnlyPatchesPrivateCopies) {
    ASSERT_GE(build(graph_entry), 0);
    prepare();
    ASSERT_EQ(snapshot_graph(), 0);
    const auto original = bytes();
    for (int i = 0; i < 20; ++i) {
        hbg::GraphHostArgs args;
        ASSERT_EQ(hbg::make_graph_host_args(snapshot, args), 0);
        EXPECT_EQ(
            args.address_offset,
            sizeof(SimplerKernelInvocationHeader) + offsetof(hbg::GraphPacketHeader, inline_payload_addr)
        );
        EXPECT_EQ(args.data_offset, sizeof(SimplerKernelInvocationHeader) + header().payload_offset);
        // Model CANN's deep copy and single relocation, with a different base.
        std::vector<std::byte> device(args.bytes + i + 1);
        auto *base = device.data() + i + 1;
        std::memcpy(base, args.storage.data(), args.bytes);
        const uint64_t address = reinterpret_cast<uintptr_t>(base) + args.data_offset;
        std::memcpy(base + args.address_offset, &address, sizeof(address));
        EXPECT_EQ(
            hbg::validate_graph_packet(base, args.bytes, hbg::GraphPacketAddress::DeviceCopy),
            hbg::GraphPacketStatus::Ok
        );
        EXPECT_NE(
            hbg::validate_graph_packet(base, args.bytes, hbg::GraphPacketAddress::HostTemplate),
            hbg::GraphPacketStatus::Ok
        );
        std::fill(args.storage.begin(), args.storage.end(), 0xdeadbeef);
        EXPECT_EQ(
            hbg::validate_graph_packet(base, args.bytes, hbg::GraphPacketAddress::DeviceCopy),
            hbg::GraphPacketStatus::Ok
        );
        EXPECT_EQ(bytes(), original);
    }
}

TEST_F(HbgGraphPacketTest, RejectsMalformedEnvelopeHeaderRegionsAndContents) {
    ASSERT_GE(build(graph_entry), 0);
    prepare();
    ASSERT_EQ(snapshot_graph(), 0);
    const auto original = bytes();
    using Mutate = void (*)(SimplerKernelInvocationHeader &, hbg::GraphPacketHeader &, hbg::GraphImageRegion &);
    const Mutate mutations[] = {
        [](auto &e, auto &, auto &) {
            e.tensor_count = CHIP_MAX_TENSOR_ARGS + 1;
        },
        [](auto &e, auto &, auto &) {
            e.scalar_count = CHIP_MAX_SCALAR_ARGS + 1;
        },
        [](auto &e, auto &, auto &) {
            e.mode = SIMPLER_MODE_PROGRAM;
        },
        [](auto &e, auto &, auto &) {
            e.callable_id = -1;
        },
        [](auto &e, auto &, auto &) {
            e.tensor_count = -1;
        },
        [](auto &e, auto &, auto &) {
            e.scalar_count = -1;
        },
        [](auto &e, auto &, auto &) {
            e.host_copy_tensor_count = 1;
        },
        [](auto &e, auto &, auto &) {
            e.reserved_ = 1;
        },
        [](auto &e, auto &, auto &) {
            e.payload_bytes = UINT64_MAX;
        },
        [](auto &e, auto &, auto &) {
            e.payload_bytes--;
        },
        [](auto &, auto &h, auto &) {
            h.magic++;
        },
        [](auto &, auto &h, auto &) {
            h.version++;
        },
        [](auto &, auto &h, auto &) {
            h.region_count = UINT32_MAX;
        },
        [](auto &, auto &h, auto &) {
            h.inline_payload_addr = 1;
        },
        [](auto &, auto &h, auto &) {
            h.slot_generation = 0;
        },
        [](auto &, auto &h, auto &) {
            h.total_tasks = h.task_window;
        },
        [](auto &, auto &h, auto &) {
            h.destinations[1].address = h.destinations[0].address;
        },
        [](auto &, auto &h, auto &) {
            h.destinations[1].capacity = UINT64_MAX;
        },
        [](auto &, auto &h, auto &) {
            h.destinations[1].address++;
        },
        [](auto &, auto &h, auto &) {
            h.reserved = 1;
        },
        [](auto &, auto &, auto &r) {
            r.source_offset = UINT64_MAX;
        },
        [](auto &, auto &, auto &r) {
            r.destination_offset = 1;
        },
        [](auto &, auto &, auto &r) {
            r.bytes--;
        },
        [](auto &, auto &, auto &r) {
            r.reserved = 1;
        },
    };
    for (auto mutate : mutations) {
        auto copy = original;
        SimplerKernelInvocationHeader e{};
        hbg::GraphPacketHeader h{};
        hbg::GraphImageRegion r{};
        std::memcpy(&e, copy.data(), sizeof(e));
        std::memcpy(&h, copy.data() + sizeof(e), sizeof(h));
        std::memcpy(&r, copy.data() + sizeof(e) + sizeof(h), sizeof(r));
        mutate(e, h, r);
        std::memcpy(copy.data(), &e, sizeof(e));
        std::memcpy(copy.data() + sizeof(e), &h, sizeof(h));
        std::memcpy(copy.data() + sizeof(e) + sizeof(h), &r, sizeof(r));
        // Recompute checksum: structural rejection must not depend on stale hash.
        h.checksum = hbg::graph_packet_checksum(copy.data(), copy.size());
        std::memcpy(copy.data() + sizeof(e), &h, sizeof(h));
        EXPECT_NE(
            hbg::validate_graph_packet(copy.data(), copy.size(), hbg::GraphPacketAddress::HostTemplate),
            hbg::GraphPacketStatus::Ok
        );
    }
    for (size_t length : {size_t{0}, size_t{63}, size_t{255}, original.size() - 1})
        EXPECT_NE(
            hbg::validate_graph_packet(original.data(), length, hbg::GraphPacketAddress::HostTemplate),
            hbg::GraphPacketStatus::Ok
        );
    auto corrupt = original;
    corrupt.back() ^= std::byte{1};
    EXPECT_EQ(
        hbg::validate_graph_packet(corrupt.data(), corrupt.size(), hbg::GraphPacketAddress::HostTemplate),
        hbg::GraphPacketStatus::InvalidChecksum
    );
    EXPECT_EQ(bytes(), original);
}

TEST_F(HbgGraphPacketTest, FailedCandidateKeepsPreviousSnapshotAndRejectsFullWindow) {
    ASSERT_EQ(build(chain_entry), 2);
    prepare();
    ASSERT_EQ(snapshot_graph(), 0);
    const auto original = bytes();
    result.total_tasks = capacity;
    result.bind_usage.submitted_tasks = capacity;
    EXPECT_EQ(snapshot_graph(), PTO_RUNTIME_ERR_CAPACITY_EXCEEDED);
    EXPECT_EQ(bytes(), original);
    EXPECT_EQ(provider.allocation_calls, 2);
    EXPECT_EQ(provider.free_calls, 0);
}

TEST_F(HbgGraphPacketTest, RejectsUnpreparedContextAndAllowsConcurrentLaunchCopies) {
    ASSERT_EQ(build(chain_entry), 2);
    EXPECT_EQ(snapshot_graph(), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_EQ(snapshot.size(), 0u);
    prepare();
    ASSERT_EQ(snapshot_graph(), 0);
    auto copy = [&]() {
        hbg::GraphHostArgs args;
        if (hbg::make_graph_host_args(snapshot, args) != 0) return false;
        return hbg::validate_graph_packet(args.storage.data(), args.bytes, hbg::GraphPacketAddress::HostTemplate) ==
               hbg::GraphPacketStatus::Ok;
    };
    auto first = std::async(std::launch::async, copy);
    auto second = std::async(std::launch::async, copy);
    EXPECT_TRUE(first.get());
    EXPECT_TRUE(second.get());
}

TEST_F(HbgGraphPacketTest, LaunchBridgeCopiesEachPacketAndPropagatesSubmissionFailure) {
    ASSERT_EQ(build(chain_entry), 2);
    prepare();
    ASSERT_EQ(snapshot_graph(), 0);
    const auto original = bytes();
    struct Transport {
        std::vector<std::vector<uint64_t>> packets;
        std::vector<size_t> lengths;
        int rc{0};
    } transport;
    hbg::GraphHostLaunchOps ops{
        &transport, [](void *ctx, hbg::GraphHostArgs &args) {
            auto &transport = *static_cast<Transport *>(ctx);
            transport.packets.push_back(args.storage);
            transport.lengths.push_back(args.bytes);
            auto *copy = reinterpret_cast<std::byte *>(transport.packets.back().data());
            const uint64_t addr = reinterpret_cast<uintptr_t>(copy) + args.data_offset;
            std::memcpy(copy + args.address_offset, &addr, sizeof(addr));
            // CANN is allowed to modify the Host launch copy too.
            std::memcpy(reinterpret_cast<std::byte *>(args.storage.data()) + args.address_offset, &addr, sizeof(addr));
            return transport.rc;
        }
    };
    ASSERT_EQ(hbg::submit_graph_template(snapshot, ops), 0);
    ASSERT_EQ(build(empty_entry), 0);
    identity.argument_hash++;
    ASSERT_EQ(snapshot_graph(), 0);
    ASSERT_EQ(hbg::submit_graph_template(snapshot, ops), 0);
    const auto second = bytes();
    transport.rc = 507001;
    EXPECT_EQ(hbg::submit_graph_template(snapshot, ops), 507001);
    EXPECT_EQ(bytes(), second);
    for (size_t i = 0; i < transport.packets.size(); ++i)
        EXPECT_EQ(
            hbg::validate_graph_packet(
                transport.packets[i].data(), transport.lengths[i], hbg::GraphPacketAddress::DeviceCopy
            ),
            hbg::GraphPacketStatus::Ok
        );
    hbg::GraphLaunchTemplate invalid;
    EXPECT_EQ(hbg::submit_graph_template(invalid, ops), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_EQ(transport.packets.size(), 3u);
    EXPECT_EQ(
        hbg::validate_graph_packet(original.data(), original.size(), hbg::GraphPacketAddress::HostTemplate),
        hbg::GraphPacketStatus::Ok
    );
    EXPECT_EQ(provider.allocation_calls, 2);
}

thread_local uint64_t packet_scalar = 0;
thread_local uintptr_t packet_tensor_address = 0;
void packet_args_entry(const ChipTaskArgs &) {
    const uint32_t shape[] = {16};
    auto input = simpler::hbg::make_tensor_external(
        reinterpret_cast<uint32_t *>(packet_tensor_address), shape, 1, DataType::FLOAT32, false, 0, AddressSpace::DEVICE
    );
    CoreTaskArgs args;
    args.add_input(input);
    args.add_scalar(packet_scalar);
    ASSERT_TRUE(bound_runtime->ops->submit_dummy_task(bound_runtime, args).task_id().is_valid());
}

TEST_F(HbgGraphPacketTest, TensorAddressesAndScalarsSurviveRelocation) {
    packet_scalar = 0x3141592653589793ULL;
    packet_tensor_address = 0x222000;
    ASSERT_EQ(build(packet_args_entry), 1);
    prepare();
    ASSERT_EQ(snapshot_graph(), 0);
    const auto first = bytes();
    const auto first_header = header();
    packet_scalar = 0xabcdef;
    packet_tensor_address = 0x333000;
    ASSERT_EQ(build(packet_args_entry), 1);
    identity.argument_hash++;
    ASSERT_EQ(snapshot_graph(), 0);
    const std::vector<std::vector<std::byte>> packets{first, bytes()};
    for (size_t i = 0; i < packets.size(); ++i) {
        Buffer aligned;
        aligned.reserve(binding.runtime_image.capacity);
        std::memcpy(
            aligned.data(), packets[i].data() + sizeof(SimplerKernelInvocationHeader) + first_header.payload_offset,
            first_header.sm_offset + result.image_bytes
        );
        const auto *sm = aligned.data() + first_header.sm_offset;
        const auto offsets = sm_layout::segment_offsets(sm_layout::image_extents(result.bind_usage));
        const auto *storage = reinterpret_cast<const ChipTaskStorage *>(sm + offsets.storage);
        EXPECT_EQ(storage[0].payload.tensor_data()[0].buffer.addr, i == 0 ? 0x222000u : 0x333000u);
        EXPECT_EQ(storage[0].payload.scalar_data()[0], i == 0 ? 0x3141592653589793ULL : 0xabcdefu);
        const auto &tasks = reinterpret_cast<const SharedMemoryHeader *>(sm)->tasks;
        EXPECT_EQ(tasks.task_storage, nullptr);
        EXPECT_EQ(tasks.task_states, nullptr);
    }
}

TEST_F(HbgGraphPacketTest, RejectsHostStorageBeforeItCanEnterAKernelGraphPacket) {
    packet_tensor_address = 0x222000;
    ASSERT_EQ(build(packet_args_entry), 1);
    prepare();
    const auto offsets = sm_layout::segment_offsets(capacity);
    auto *storage = reinterpret_cast<ChipTaskStorage *>(mirror.data() + offsets.storage);
    ASSERT_EQ(storage[0].payload.tensor_count, 1);
    storage[0].payload.tensor_data()[0].address_space = AddressSpace::HOST;
    const int allocations = provider.allocation_calls;

    EXPECT_EQ(snapshot_graph(), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_EQ(snapshot.size(), 0u);
    EXPECT_EQ(provider.allocation_calls, allocations);
}

TEST_F(HbgGraphPacketTest, EnvelopeCarriesTheHostCopySuffixCount) {
    ASSERT_EQ(build(chain_entry), 2);
    prepare();
    identity.tensor_count = 2;
    identity.host_copy_tensor_count = 1;
    ASSERT_EQ(snapshot_graph(), 0);
    SimplerKernelInvocationHeader invocation{};
    std::memcpy(&invocation, snapshot.data(), sizeof(invocation));
    EXPECT_EQ(invocation.tensor_count, 2);
    EXPECT_EQ(invocation.host_copy_tensor_count, 1);
    EXPECT_EQ(
        hbg::validate_graph_packet(snapshot.data(), snapshot.size(), hbg::GraphPacketAddress::HostTemplate),
        hbg::GraphPacketStatus::Ok
    );
}

thread_local int packet_task_count = 0;
void packet_window_entry(const ChipTaskArgs &) {
    for (int i = 0; i < packet_task_count; ++i) {
        CoreTaskArgs args;
        ASSERT_TRUE(bound_runtime->ops->submit_dummy_task(bound_runtime, args).task_id().is_valid());
    }
}

TEST_F(HbgGraphPacketTest, RealBuildAcceptsWindowMinusOneAndRejectsFullWindow) {
    packet_task_count = capacity - 1;
    ASSERT_EQ(build(packet_window_entry), capacity - 1);
    prepare();
    ASSERT_EQ(snapshot_graph(), 0);
    const auto previous = bytes();
    packet_task_count = capacity;
    ASSERT_EQ(build(packet_window_entry), capacity);
    EXPECT_EQ(snapshot_graph(), PTO_RUNTIME_ERR_CAPACITY_EXCEEDED);
    EXPECT_EQ(bytes(), previous);
}

class HbgGraphSlotTest : public HbgGraphPacketTest {
protected:
    simpler::kernel::PreparedInvocationView trusted_callable{7, 1, 2};
    hbg::GraphSlotRegistration seal{};
    hbg::GraphSlotRegistry *registry{nullptr};
    hbg::GraphHostArgs packet;

    static int install_slot(void *opaque, const hbg::GraphSlotRegistration &record, void *stream) {
        auto &self = *static_cast<HbgGraphSlotTest *>(opaque);
        EXPECT_EQ(stream, self.context.hidden_stream(KernelStreamKind::Aicpu));
        EXPECT_NE(stream, self.context.hidden_stream(KernelStreamKind::Aicore));
        auto copy = record;
        return simpler_aicpu_l1_hbg_register_execution_slot(&copy);
    }
    void TearDown() override {
        if (registry != nullptr) hbg::detach_graph_slot_registry(registry);
        HbgGraphPacketTest::TearDown();
    }
    void prepare_slot(bool publish = true, uint64_t scheduler_capacity = 0) {
        ASSERT_GE(build(graph_entry), 0);
        prepare(0, scheduler_capacity);
        ASSERT_EQ(hbg::seal_graph_execution_slot(context, 0, 19, 109, seal), 0);
        registry = reinterpret_cast<hbg::GraphSlotRegistry *>(seal.registry.address);
        if (publish) {
            ASSERT_EQ(simpler_aicpu_l1_hbg_register_execution_slot(&seal), 0);
        } else {
            ASSERT_EQ(hbg::initialize_graph_slot_registry(registry, 0, 19, 109), hbg::GraphSlotStatus::Ok);
        }
        ASSERT_EQ(snapshot_graph(), 0);
        ASSERT_EQ(hbg::make_graph_host_args(snapshot, packet), 0);
        patch();
        for (size_t i = 0; i < 4; ++i)
            if (seal.destinations[i].capacity != 0)
                std::memset(
                    reinterpret_cast<void *>(seal.destinations[i].address), 0x50 + i, seal.destinations[i].capacity
                );
    }
    hbg::GraphPacketHeader &packet_header() {
        return *reinterpret_cast<hbg::GraphPacketHeader *>(
            reinterpret_cast<std::byte *>(packet.storage.data()) + sizeof(SimplerKernelInvocationHeader)
        );
    }
    void patch() {
        auto &h = packet_header();
        h.inline_payload_addr = reinterpret_cast<uintptr_t>(packet.storage.data()) +
                                sizeof(SimplerKernelInvocationHeader) + h.payload_offset;
        h.checksum = hbg::graph_packet_checksum(packet.storage.data(), packet.bytes);
    }
    hbg::GraphSlotStatus admit(hbg::GraphRestoreView &view) {
        return hbg::admit_graph_packet_for_restore(packet.storage.data(), packet.bytes, 0, 109, trusted_callable, view);
    }
    std::vector<std::byte> working_bytes() const {
        std::vector<std::byte> data;
        for (size_t i = 0; i < 5; ++i) {
            const auto &region = i == 4 ? seal.registry : seal.destinations[i];
            if (region.capacity == 0) continue;
            const auto *first = reinterpret_cast<const std::byte *>(region.address);
            data.insert(data.end(), first, first + region.capacity);
        }
        return data;
    }
    void expect_rejected(hbg::GraphSlotStatus expected) {
        const auto before = working_bytes();
        hbg::GraphRestoreView output{};
        output.slot.slot_generation = 999;
        output.payload = reinterpret_cast<const std::byte *>(uintptr_t{1});
        EXPECT_EQ(admit(output), expected);
        EXPECT_EQ(output.slot.slot_generation, 999u);
        EXPECT_EQ(output.payload, reinterpret_cast<const std::byte *>(uintptr_t{1}));
        EXPECT_EQ(working_bytes(), before);
    }
};

TEST_F(HbgGraphSlotTest, SealsFrozenContextBeforeReadyAndUsesDedicatedPrepareStream) {
    ASSERT_GE(build(chain_entry), 0);
    RuntimeArenaLayout compact{};
    ASSERT_EQ(hbg::make_kernel_graph_layout(capacity, compact), 0);
    ASSERT_EQ(hbg::get_graph_resource_requirements(result, compact, required), 0);
    hbg::KernelResourcePlan plan;
    ASSERT_EQ(hbg::KernelResourcePlan::create(&required, 1, plan), 0);
    ASSERT_EQ(context.initialize(0, provider.context_ops(), 19), 0);
    seal.slot_generation = 777;
    EXPECT_EQ(hbg::seal_graph_execution_slot(context, 0, 19, 109, seal), PTO_RUNTIME_ERR_INVALID_STATE);
    ASSERT_EQ(plan.prepare(context, provider.resource_ops()), 0);
    EXPECT_EQ(hbg::seal_graph_execution_slot(context, 0, 19, 109, seal), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_EQ(seal.slot_generation, 777u);
    ASSERT_EQ(context.freeze_resources(), 0);
    ASSERT_EQ(hbg::seal_graph_execution_slot(context, 0, 19, 109, seal), 0);
    EXPECT_EQ(context.phase(), KernelContextPhase::Collecting);
    EXPECT_EQ(hbg::bind_kernel_resources_for_launch(context, 0, 19, required, binding), PTO_RUNTIME_ERR_INVALID_STATE);
    registry = reinterpret_cast<hbg::GraphSlotRegistry *>(seal.registry.address);
    EXPECT_EQ(seal.registry.capacity, sizeof(hbg::GraphSlotRegistry));
    const int allocations = provider.allocation_calls;
    const hbg::GraphSlotPrepareOps ops{this, install_slot};
    ASSERT_EQ(hbg::prepare_graph_execution_slot(context, 0, 19, 109, ops), 0);
    EXPECT_EQ(context.phase(), KernelContextPhase::Collecting);
    ASSERT_EQ(context.mark_ready_enqueued(), 0);
    ASSERT_EQ(snapshot_graph(), 0);
    ASSERT_EQ(hbg::make_graph_host_args(snapshot, packet), 0);
    patch();
    hbg::GraphRestoreView view;
    ASSERT_EQ(admit(view), hbg::GraphSlotStatus::Ok);
    EXPECT_EQ(view.slot.slot_generation, 19u);
    EXPECT_EQ(view.graph.runtime_binary_id, 109u);
    EXPECT_EQ(view.slot.max_packet_bytes, packet.bytes);
    EXPECT_EQ(provider.allocation_calls, allocations);
    EXPECT_TRUE(platform.copies.empty());
}

TEST_F(HbgGraphSlotTest, SealFailurePreservesOutputAndDoesNotEnqueue) {
    ASSERT_NO_FATAL_FAILURE(prepare_slot());
    const auto original = seal;
    const hbg::GraphSlotPrepareOps ops{nullptr, [](void *, const auto &, void *) {
                                           ADD_FAILURE() << "invalid prepare must not enqueue";
                                           return 0;
                                       }};
    EXPECT_EQ(hbg::prepare_graph_execution_slot(context, 1, 19, 109, ops), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_EQ(hbg::prepare_graph_execution_slot(context, 0, 20, 109, ops), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_EQ(hbg::seal_graph_execution_slot(context, 0, 19, 0, seal), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_EQ(std::memcmp(&seal, &original, sizeof(seal)), 0);
    const hbg::GraphSlotPrepareOps failure{nullptr, [](void *, const auto &, void *) {
                                               return -73;
                                           }};
    EXPECT_EQ(hbg::prepare_graph_execution_slot(context, 0, 19, 109, failure), -73);
    context.poison(-17);
    EXPECT_EQ(hbg::prepare_graph_execution_slot(context, 0, 19, 109, ops), PTO_RUNTIME_ERR_INVALID_STATE);
}

TEST_F(HbgGraphSlotTest, AdmitsOwnedPacketWithoutWritingWorkingMemoryOrRegistry) {
    ASSERT_NO_FATAL_FAILURE(prepare_slot());
    const auto before = working_bytes();
    hbg::GraphRestoreView view;
    ASSERT_EQ(admit(view), hbg::GraphSlotStatus::Ok);
    EXPECT_EQ(std::memcmp(&view.slot, &seal, sizeof(seal)), 0);
    EXPECT_EQ(view.payload, reinterpret_cast<const std::byte *>(packet_header().inline_payload_addr));
    EXPECT_EQ(working_bytes(), before);
    EXPECT_EQ(
        hbg::validate_graph_packet(snapshot.data(), snapshot.size(), hbg::GraphPacketAddress::HostTemplate),
        hbg::GraphPacketStatus::Ok
    );
}

TEST_F(HbgGraphSlotTest, RejectsEveryForgedBaseEvenWithAValidPacketChecksum) {
    ASSERT_NO_FATAL_FAILURE(prepare_slot());
    const auto original = packet;
    for (size_t i = 0; i < 4; ++i) {
        if (seal.destinations[i].capacity == 0) continue;
        packet = original;
        packet_header().destinations[i].address += 0x10000000000ULL;
        patch();
        ASSERT_EQ(
            hbg::validate_graph_packet(packet.storage.data(), packet.bytes, hbg::GraphPacketAddress::DeviceCopy),
            hbg::GraphPacketStatus::Ok
        );
        expect_rejected(hbg::GraphSlotStatus::BindingMismatch);
    }
}

TEST_F(HbgGraphSlotTest, RejectsSmallerDeclaredCapacityInAnOtherwiseValidPacket) {
    ASSERT_NO_FATAL_FAILURE(prepare_slot());
    auto &header = packet_header();
    size_t last = 3;
    while (header.destinations[last].capacity == 0)
        --last;
    ASSERT_GT(header.destinations[last].capacity, 64u);
    header.destinations[last].capacity -= 64;
    header.payload_bytes -= 64;
    header.total_bytes -= 64;
    auto *regions = reinterpret_cast<hbg::GraphImageRegion *>(reinterpret_cast<std::byte *>(&header) + sizeof(header));
    regions[header.region_count - 1].bytes -= 64;
    auto *envelope = reinterpret_cast<SimplerKernelInvocationHeader *>(packet.storage.data());
    envelope->payload_bytes -= 64;
    packet.bytes -= 64;
    packet.storage.resize((packet.bytes + 7) / 8);
    patch();
    ASSERT_EQ(
        hbg::validate_graph_packet(packet.storage.data(), packet.bytes, hbg::GraphPacketAddress::DeviceCopy),
        hbg::GraphPacketStatus::Ok
    );
    expect_rejected(hbg::GraphSlotStatus::BindingMismatch);
}

TEST_F(HbgGraphSlotTest, RejectsMisalignedOrStaleOrForeignIdentityBeforeAnyWrite) {
    ASSERT_NO_FATAL_FAILURE(prepare_slot());
    const auto original = packet;
    using Mutate = void (*)(hbg::GraphPacketHeader &);
    const std::pair<Mutate, hbg::GraphSlotStatus> cases[] = {
        {[](auto &h) {
             ++h.destinations[0].address;
         },
         hbg::GraphSlotStatus::InvalidPacket},
        {[](auto &h) {
             --h.slot_generation;
         },
         hbg::GraphSlotStatus::GenerationMismatch},
        {[](auto &h) {
             ++h.device_id;
         },
         hbg::GraphSlotStatus::DeviceMismatch},
        {[](auto &h) {
             ++h.runtime_binary_id;
         },
         hbg::GraphSlotStatus::BinaryMismatch},
        {[](auto &h) {
             h.version = 1;
         },
         hbg::GraphSlotStatus::InvalidPacket},
    };
    for (const auto &[mutate, expected] : cases) {
        packet = original;
        mutate(packet_header());
        patch();
        expect_rejected(expected);
    }
}

TEST_F(HbgGraphSlotTest, RejectsSourceOverlapAndOversizedPacketsBeforeReadingPayload) {
    ASSERT_NO_FATAL_FAILURE(prepare_slot());
    const auto before = working_bytes();
    hbg::GraphRestoreView out;
    for (const auto &destination : seal.destinations) {
        if (destination.capacity == 0) continue;
        EXPECT_EQ(
            hbg::admit_graph_packet_for_restore(
                reinterpret_cast<const void *>(destination.address), 64, 0, 109, trusted_callable, out
            ),
            hbg::GraphSlotStatus::SourceOverlap
        );
    }
    EXPECT_EQ(
        hbg::admit_graph_packet_for_restore(registry, 64, 0, 109, trusted_callable, out),
        hbg::GraphSlotStatus::SourceOverlap
    );
    EXPECT_EQ(
        hbg::admit_graph_packet_for_restore(
            packet.storage.data(), seal.max_packet_bytes + 1, 0, 109, trusted_callable, out
        ),
        hbg::GraphSlotStatus::InvalidPacket
    );
    EXPECT_EQ(
        hbg::admit_graph_packet_for_restore(packet.storage.data(), 1, 0, 109, trusted_callable, out),
        hbg::GraphSlotStatus::InvalidPacket
    );
    EXPECT_EQ(working_bytes(), before);
    EXPECT_EQ(out.payload, nullptr);
}

TEST_F(HbgGraphSlotTest, RegistrationIsIdempotentAndConflictsLeaveTheSealedSlotUntouched) {
    ASSERT_NO_FATAL_FAILURE(prepare_slot());
    const auto before = working_bytes();
    EXPECT_EQ(hbg::register_graph_execution_slot(registry, &seal, sizeof(seal)), hbg::GraphSlotStatus::Ok);
    auto other = seal;
    other.destinations[0].address += 0x10000000000ULL;
    other.checksum = hbg::graph_slot_checksum(other);
    ASSERT_TRUE(hbg::valid_graph_slot_registration(other));
    EXPECT_EQ(hbg::register_graph_execution_slot(registry, &other, sizeof(other)), hbg::GraphSlotStatus::Conflict);
    EXPECT_EQ(hbg::initialize_graph_slot_registry(registry, 0, 20, 109), hbg::GraphSlotStatus::Conflict);
    EXPECT_EQ(working_bytes(), before);
}

TEST_F(HbgGraphSlotTest, KernelRegistrationEntryCopiesUnalignedArgsAndRejectsConflictingDuplicates) {
    ASSERT_NO_FATAL_FAILURE(prepare_slot());
    std::array<std::byte, sizeof(hbg::GraphSlotRegistration) + 1> unaligned{};
    std::memcpy(unaligned.data() + 1, &seal, sizeof(seal));
    EXPECT_EQ(simpler_aicpu_l1_hbg_register_execution_slot(unaligned.data() + 1), 0);

    auto conflict = seal;
    conflict.destinations[0].address += 0x10000000000ULL;
    conflict.checksum = hbg::graph_slot_checksum(conflict);
    ASSERT_TRUE(hbg::valid_graph_slot_registration(conflict));
    EXPECT_NE(simpler_aicpu_l1_hbg_register_execution_slot(&conflict), 0);

    hbg::GraphSlotRegistration acquired{};
    ASSERT_EQ(hbg::acquire_graph_execution_slot(registry, 0, 109, acquired), hbg::GraphSlotStatus::Ok);
    EXPECT_EQ(std::memcmp(&acquired, &seal, sizeof(seal)), 0);
    EXPECT_NE(simpler_aicpu_l1_hbg_register_execution_slot(nullptr), 0);
}

TEST_F(HbgGraphSlotTest, KernelRegistrationRejectsASecondLiveContextBeforeWritingIt) {
    ASSERT_NO_FATAL_FAILURE(prepare_slot());
    alignas(1024) std::array<std::byte, 1024> other_storage{};
    auto other = seal;
    other.slot_generation++;
    other.registry.address = reinterpret_cast<uintptr_t>(other_storage.data());
    other.checksum = hbg::graph_slot_checksum(other);
    ASSERT_TRUE(hbg::valid_graph_slot_registration(other));
    const auto before = other_storage;

    EXPECT_NE(simpler_aicpu_l1_hbg_register_execution_slot(&other), 0);
    EXPECT_EQ(other_storage, before);

    ASSERT_TRUE(hbg::detach_graph_slot_registry(registry));
    registry = nullptr;
    ASSERT_EQ(simpler_aicpu_l1_hbg_register_execution_slot(&other), 0);
    auto *other_registry = reinterpret_cast<hbg::GraphSlotRegistry *>(other.registry.address);
    hbg::GraphSlotRegistration acquired{};
    ASSERT_EQ(hbg::acquire_graph_execution_slot(other_registry, 0, 109, acquired), hbg::GraphSlotStatus::Ok);
    EXPECT_EQ(std::memcmp(&acquired, &other, sizeof(other)), 0);
    EXPECT_TRUE(hbg::detach_graph_slot_registry(other_registry));
    hbg::GraphRestoreView detached{};
    EXPECT_EQ(admit(detached), hbg::GraphSlotStatus::NotReady);
}

TEST(HbgKernelRegistrationManifestTest, ProgramAndKernelModesExposeIndependentSymbolSets) {
    constexpr std::array<const char *, 4> kernel_symbols = {
        "simpler_aicpu_l1_hbg_register_execution_slot",
        "simpler_aicpu_l1_hbg_detach_execution_slot",
        "simpler_aicpu_l1_hbg_register_callable",
        "simpler_aicpu_kernel_exec",
    };
    size_t program_count = 0;
    const char *const *program = runtime_extra_aicpu_symbols(&program_count);
    for (size_t i = 0; i < program_count; ++i) {
        ASSERT_NE(program, nullptr);
        for (const char *kernel_symbol : kernel_symbols)
            EXPECT_NE(std::strcmp(program[i], kernel_symbol), 0);
    }

    size_t kernel_count = 0;
    const char *const *kernel = runtime_l1_extra_aicpu_symbols(&kernel_count);
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel_count, kernel_symbols.size());
    for (size_t i = 0; i < kernel_count; ++i)
        EXPECT_EQ(std::strcmp(kernel[i], kernel_symbols[i]), 0);

    size_t repeated_program_count = 0;
    const char *const *repeated_program = runtime_extra_aicpu_symbols(&repeated_program_count);
    EXPECT_EQ(repeated_program, program);
    EXPECT_EQ(repeated_program_count, program_count);
}

TEST(HbgKernelArgumentSnapshotTest, IgnoresPaddingAndUnusedDimensions) {
    const uint32_t shape[] = {2, 3};
    ChipStorageTaskArgs first;
    first.add_tensor(
        make_tensor_external(reinterpret_cast<void *>(0x1000), shape, 2, DataType::FLOAT32, AddressSpace::DEVICE)
    );
    first.add_scalar(17);
    ChipStorageTaskArgs second = first;

    first.tensor(0).shapes[MAX_TENSOR_DIMS - 1] = 111;
    first.tensor(0).strides[MAX_TENSOR_DIMS - 1] = 222;
    second.tensor(0).shapes[MAX_TENSOR_DIMS - 1] = 333;
    second.tensor(0).strides[MAX_TENSOR_DIMS - 1] = 444;
    first.tensors_[CHIP_MAX_TENSOR_ARGS - 1].buffer.addr = 0xaaaa;
    second.tensors_[CHIP_MAX_TENSOR_ARGS - 1].buffer.addr = 0xbbbb;
    first.scalars_[CHIP_MAX_SCALAR_ARGS - 1] = 0xcccc;
    second.scalars_[CHIP_MAX_SCALAR_ARGS - 1] = 0xdddd;

    EXPECT_TRUE(hbg::same_kernel_argument_snapshot(first, second));
    EXPECT_EQ(hbg::kernel_argument_snapshot_hash(first), hbg::kernel_argument_snapshot_hash(second));
}

TEST(HbgKernelArgumentSnapshotTest, TracksSemanticTensorAndScalarChanges) {
    const uint32_t shape[] = {4};
    ChipStorageTaskArgs baseline;
    baseline.add_tensor(
        make_tensor_external(reinterpret_cast<void *>(0x2000), shape, 1, DataType::FLOAT32, AddressSpace::DEVICE)
    );
    baseline.add_scalar(9);

    ChipStorageTaskArgs changed_address = baseline;
    changed_address.tensor(0).buffer.addr += 64;
    EXPECT_FALSE(hbg::same_kernel_argument_snapshot(baseline, changed_address));
    EXPECT_NE(hbg::kernel_argument_snapshot_hash(baseline), hbg::kernel_argument_snapshot_hash(changed_address));

    ChipStorageTaskArgs changed_scalar = baseline;
    changed_scalar.scalar(0)++;
    EXPECT_FALSE(hbg::same_kernel_argument_snapshot(baseline, changed_scalar));
    EXPECT_NE(hbg::kernel_argument_snapshot_hash(baseline), hbg::kernel_argument_snapshot_hash(changed_scalar));
}

TEST_F(HbgGraphSlotTest, InvalidRegistrationCannotPublishAnyRegistryBytes) {
    ASSERT_NO_FATAL_FAILURE(prepare_slot(false));
    const auto before = working_bytes();
    using Mutate = void (*)(hbg::GraphSlotRegistration &);
    const Mutate cases[] = {
        [](auto &s) {
            ++s.version;
        },
        [](auto &s) {
            s.flags = 0;
        },
        [](auto &s) {
            s.slot_generation = 0;
        },
        [](auto &s) {
            s.registry.address = s.destinations[1].address;
        },
        [](auto &s) {
            s.destinations[0].capacity = UINT64_MAX;
        },
        [](auto &s) {
            ++s.max_packet_bytes;
        },
        [](auto &s) {
            s.runtime_binary_id = 0;
        }
    };
    for (auto mutate : cases) {
        auto invalid = seal;
        mutate(invalid);
        invalid.checksum = hbg::graph_slot_checksum(invalid);
        EXPECT_EQ(
            hbg::register_graph_execution_slot(registry, &invalid, sizeof(invalid)),
            hbg::GraphSlotStatus::InvalidRegistration
        );
        EXPECT_EQ(working_bytes(), before);
    }
    EXPECT_EQ(
        hbg::register_graph_execution_slot(registry, &seal, sizeof(seal) - 1), hbg::GraphSlotStatus::InvalidRegistration
    );
    EXPECT_EQ(working_bytes(), before);
}

TEST_F(HbgGraphSlotTest, EmptyPublishingAndCorruptRegistriesFailClosed) {
    ASSERT_NO_FATAL_FAILURE(prepare_slot(false));
    hbg::GraphSlotRegistration output{};
    output.slot_generation = 999;
    EXPECT_EQ(hbg::acquire_graph_execution_slot(registry, 0, 109, output), hbg::GraphSlotStatus::NotReady);
    EXPECT_EQ(hbg::bind_graph_slot_registry(registry, 0, 109), hbg::GraphSlotStatus::NotReady);
    expect_rejected(hbg::GraphSlotStatus::NotReady);
    __atomic_store_n(&registry->phase, static_cast<uint32_t>(hbg::GraphSlotPhase::Publishing), __ATOMIC_RELEASE);
    EXPECT_EQ(hbg::acquire_graph_execution_slot(registry, 0, 109, output), hbg::GraphSlotStatus::Publishing);
    EXPECT_EQ(hbg::register_graph_execution_slot(registry, &seal, sizeof(seal)), hbg::GraphSlotStatus::Publishing);
    __atomic_store_n(&registry->phase, 42u, __ATOMIC_RELEASE);
    EXPECT_EQ(hbg::acquire_graph_execution_slot(registry, 0, 109, output), hbg::GraphSlotStatus::InvalidRegistry);
    EXPECT_EQ(output.slot_generation, 999u);
    ASSERT_EQ(hbg::initialize_graph_slot_registry(registry, 0, 19, 109), hbg::GraphSlotStatus::Ok);
    ASSERT_EQ(hbg::register_graph_execution_slot(registry, &seal, sizeof(seal)), hbg::GraphSlotStatus::Ok);
    ASSERT_EQ(hbg::bind_graph_slot_registry(registry, 0, 109), hbg::GraphSlotStatus::Ok);
    registry->registration.checksum++;
    expect_rejected(hbg::GraphSlotStatus::InvalidRegistry);
}

TEST_F(HbgGraphSlotTest, DetachedRegistryCannotAuthorizeReplayAndHasNoResidentGenerationHistory) {
    ASSERT_NO_FATAL_FAILURE(prepare_slot());
    ASSERT_TRUE(hbg::detach_graph_slot_registry(registry));
    expect_rejected(hbg::GraphSlotStatus::NotReady);
    auto next = seal;
    ++next.slot_generation;
    next.checksum = hbg::graph_slot_checksum(next);
    ASSERT_EQ(simpler_aicpu_l1_hbg_register_execution_slot(&next), 0);
    expect_rejected(hbg::GraphSlotStatus::GenerationMismatch);
    packet_header().slot_generation = 20;
    patch();
    hbg::GraphRestoreView view;
    EXPECT_EQ(admit(view), hbg::GraphSlotStatus::Ok);
    EXPECT_EQ(view.slot.slot_generation, 20u);
}

TEST_F(HbgGraphSlotTest, InvocationIdentityMayVaryWithoutChangingTheRegisteredSlot) {
    ASSERT_NO_FATAL_FAILURE(prepare_slot());
    auto *envelope = reinterpret_cast<SimplerKernelInvocationHeader *>(packet.storage.data());
    ++envelope->callable_id;
    ++trusted_callable.callable_id;
    ++packet_header().callable_hash;
    ++packet_header().argument_hash;
    ++packet_header().function_hash;
    patch();
    const auto before = working_bytes();
    hbg::GraphRestoreView view;
    ASSERT_EQ(admit(view), hbg::GraphSlotStatus::Ok);
    EXPECT_EQ(working_bytes(), before);
}

TEST_F(HbgGraphSlotTest, ConcurrentPublicationAndReadersSeeOnlyTheCompleteRecord) {
    ASSERT_NO_FATAL_FAILURE(prepare_slot(false));
    auto publish = [&] {
        return hbg::register_graph_execution_slot(registry, &seal, sizeof(seal));
    };
    std::vector<std::future<hbg::GraphSlotStatus>> writers;
    for (size_t i = 0; i < 16; ++i)
        writers.push_back(std::async(std::launch::async, publish));
    for (auto &writer : writers) {
        const auto status = writer.get();
        EXPECT_TRUE(status == hbg::GraphSlotStatus::Ok || status == hbg::GraphSlotStatus::Publishing);
    }
    ASSERT_EQ(hbg::bind_graph_slot_registry(registry, 0, 109), hbg::GraphSlotStatus::Ok);
    const auto before = working_bytes();
    std::vector<std::future<bool>> readers;
    for (size_t i = 0; i < 16; ++i)
        readers.push_back(std::async(std::launch::async, [&] {
            hbg::GraphRestoreView view;
            return admit(view) == hbg::GraphSlotStatus::Ok && std::memcmp(&view.slot, &seal, sizeof(seal)) == 0;
        }));
    for (auto &reader : readers)
        EXPECT_TRUE(reader.get());
    EXPECT_EQ(working_bytes(), before);
}

TEST_F(HbgGraphSlotTest, RegistryAddressAndExpectedRuntimeIdentityComeFromControlState) {
    ASSERT_NO_FATAL_FAILURE(prepare_slot());
    Buffer storage;
    storage.reserve(sizeof(hbg::GraphSlotRegistry), DeviceArena::kDefaultBaseAlign);
    auto *other = reinterpret_cast<hbg::GraphSlotRegistry *>(storage.data());
    ASSERT_EQ(hbg::initialize_graph_slot_registry(other, 0, 19, 109), hbg::GraphSlotStatus::Ok);
    const auto before = *other;
    EXPECT_EQ(hbg::register_graph_execution_slot(other, &seal, sizeof(seal)), hbg::GraphSlotStatus::Conflict);
    EXPECT_EQ(std::memcmp(&before, other, sizeof(before)), 0);
    auto other_seal = seal;
    other_seal.registry.address = reinterpret_cast<uintptr_t>(other);
    other_seal.checksum = hbg::graph_slot_checksum(other_seal);
    std::vector<std::byte> unaligned(sizeof(other_seal) + 1);
    std::memcpy(unaligned.data() + 1, &other_seal, sizeof(other_seal));
    ASSERT_EQ(
        hbg::register_graph_execution_slot(other, unaligned.data() + 1, sizeof(other_seal)), hbg::GraphSlotStatus::Ok
    );
    EXPECT_EQ(hbg::bind_graph_slot_registry(other, 0, 109), hbg::GraphSlotStatus::Conflict);
    EXPECT_FALSE(hbg::detach_graph_slot_registry(other));
    hbg::GraphRestoreView view;
    EXPECT_EQ(
        hbg::admit_graph_packet_for_restore(packet.storage.data(), packet.bytes, 1, 109, trusted_callable, view),
        hbg::GraphSlotStatus::DeviceMismatch
    );
    EXPECT_EQ(
        hbg::admit_graph_packet_for_restore(packet.storage.data(), packet.bytes, 0, 110, trusted_callable, view),
        hbg::GraphSlotStatus::BinaryMismatch
    );
    EXPECT_EQ(view.payload, nullptr);
    ASSERT_EQ(admit(view), hbg::GraphSlotStatus::Ok);
    EXPECT_EQ(view.slot.registry.address, seal.registry.address);
}

TEST_F(HbgGraphSlotTest, RejectsCorruptRegistryHeaderAndRetainsOwnershipUntilDetachAndClose) {
    ASSERT_NO_FATAL_FAILURE(prepare_slot());
    registry->reserved[0] = 1;
    expect_rejected(hbg::GraphSlotStatus::InvalidRegistry);
    registry->reserved[0] = 0;
    registry->context_generation++;
    expect_rejected(hbg::GraphSlotStatus::InvalidRegistry);
    EXPECT_EQ(provider.allocator.get_allocation_count(), 2u);
    ASSERT_TRUE(hbg::detach_graph_slot_registry(registry));
    registry = nullptr;
    ASSERT_EQ(context.close(), 0);
    EXPECT_EQ(provider.allocator.get_allocation_count(), 0u);
    hbg::GraphRestoreView view;
    EXPECT_EQ(admit(view), hbg::GraphSlotStatus::NotReady);
}

}  // namespace

TEST_F(HbgGraphSlotTest, RejectsCallableIdentityAndCountsDespiteValidChecksum) {
    ASSERT_NO_FATAL_FAILURE(prepare_slot());
    const auto original = packet.storage;
    using Mutate = void (*)(SimplerKernelInvocationHeader &);
    const Mutate mutations[] = {
        [](auto &h) {
            ++h.callable_id;
        },
        [](auto &h) {
            ++h.tensor_count;
        },
        [](auto &h) {
            ++h.scalar_count;
        },
        [](auto &h) {
            h.host_copy_tensor_count = 1;
        },
        [](auto &h) {
            h.callable_id = MAX_REGISTERED_CALLABLE_IDS;
        },
        [](auto &h) {
            h.tensor_count = CHIP_MAX_TENSOR_ARGS;
            h.scalar_count = 1;
        },
        [](auto &h) {
            h.reserved_ = 1;
        },
    };
    for (auto mutate : mutations) {
        packet.storage = original;
        mutate(*reinterpret_cast<SimplerKernelInvocationHeader *>(packet.storage.data()));
        patch();
        const auto before = working_bytes();
        hbg::GraphRestoreView output{};
        output.invocation.callable_id = 999;
        EXPECT_NE(admit(output), hbg::GraphSlotStatus::Ok);
        EXPECT_EQ(output.invocation.callable_id, 999);
        EXPECT_EQ(working_bytes(), before);
    }
}

TEST_F(HbgGraphSlotTest, CallableIdentityAndContextGenerationHaveIndependentAuthorities) {
    ASSERT_NO_FATAL_FAILURE(prepare_slot());
    hbg::GraphRestoreView output{};
    ASSERT_EQ(admit(output), hbg::GraphSlotStatus::Ok);
    EXPECT_EQ(output.invocation.callable_id, 7);
    EXPECT_EQ(output.slot.slot_generation, 19u);
    trusted_callable.callable_id = 19;
    expect_rejected(hbg::GraphSlotStatus::CallableMismatch);
    trusted_callable.callable_id = -1;
    expect_rejected(hbg::GraphSlotStatus::InvalidPacket);
}

class HbgGraphRestoreTest : public HbgGraphSlotTest {
protected:
    hbg::GraphRestoreResult restored;
    hbg::GraphRestoreStatus restore(const hbg::GraphRestoreOps &ops = {}) {
        return hbg::restore_graph_packet(packet.storage.data(), packet.bytes, 0, 109, trusted_callable, restored, ops);
    }
    void retire(hbg::GraphRestoreRetirement outcome = hbg::GraphRestoreRetirement::Completed) {
        ASSERT_EQ(
            hbg::retire_graph_restore(registry, registry->restore.attempt, {outcome}), hbg::GraphRestoreStatus::Ok
        );
    }
    void dirty_working() {
        for (const auto &dst : seal.destinations)
            if (dst.capacity) std::memset(reinterpret_cast<void *>(dst.address), 0xa5, dst.capacity);
    }
    std::vector<std::byte> mutable_bytes() const {
        auto all = working_bytes();
        all.resize(all.size() - sizeof(hbg::GraphSlotRegistry));
        return all;
    }
};

TEST_F(HbgGraphRestoreTest, OrdinaryIntermediateSurvivesBuildPacketAndRestore) {
    ASSERT_EQ(build(chain_entry), 2);
    prepare(4096);
    ASSERT_EQ(hbg::seal_graph_execution_slot(context, 0, 19, 109, seal), 0);
    registry = reinterpret_cast<hbg::GraphSlotRegistry *>(seal.registry.address);
    ASSERT_EQ(simpler_aicpu_l1_hbg_register_execution_slot(&seal), 0);
    ASSERT_EQ(snapshot_graph(), 0);
    ASSERT_EQ(hbg::make_graph_host_args(snapshot, packet), 0);
    patch();
    std::memset(reinterpret_cast<void *>(binding.heap.address), 0xa5, binding.heap.capacity);
    ASSERT_EQ(restore(), hbg::GraphRestoreStatus::Ok);
    const auto offsets = sm_layout::segment_offsets(sm_layout::image_extents(result.bind_usage));
    const auto *tasks = reinterpret_cast<const ChipTaskStorage *>(
        seal.destinations[1].address + packet_header().sm_offset + offsets.storage
    );
    EXPECT_EQ(tasks[0].payload.tensor_data()[0].buffer.addr, tasks[1].payload.tensor_data()[0].buffer.addr);
    EXPECT_EQ(tasks[1].payload.fanin_data()[0], 0);
    const auto *heap = reinterpret_cast<const uint8_t *>(binding.heap.address);
    EXPECT_TRUE(std::all_of(heap + result.heap_bytes, heap + binding.heap.capacity, [](auto value) {
        return value == 0xa5;
    }));
    ASSERT_NO_FATAL_FAILURE(retire());
    auto *source = reinterpret_cast<ChipTaskStorage *>(
        reinterpret_cast<std::byte *>(packet.storage.data()) + packet.data_offset + packet_header().sm_offset +
        offsets.storage
    );
    auto &tensor = source[0].payload.tensor_data()[0];
    const auto original = tensor;
    for (uint64_t address : {binding.heap.address - 64, binding.heap.address + result.heap_bytes, HEAP_VIRTUAL_BASE}) {
        tensor.buffer.addr = address;
        patch();
        const auto before = working_bytes();
        EXPECT_EQ(restore(), hbg::GraphRestoreStatus::InvalidImage);
        EXPECT_EQ(working_bytes(), before);
    }
    tensor = original;
    tensor.buffer.size = result.heap_bytes + 1;
    patch();
    const auto before = working_bytes();
    EXPECT_EQ(restore(), hbg::GraphRestoreStatus::InvalidImage);
    EXPECT_EQ(working_bytes(), before);
}

TEST_F(HbgGraphPacketTest, PacketDoesNotTransmitUnusedFrozenCapacity) {
    ASSERT_EQ(build(chain_entry), 2);
    prepare(4 * 1024 * 1024, 4 * 1024 * 1024);
    ASSERT_EQ(snapshot_graph(), 0);
    EXPECT_LT(
        snapshot.size(),
        required.runtime_arena_bytes + required.graph_definition_bytes + required.scheduler_state_bytes + 1024
    );
}

TEST_F(HbgGraphRestoreTest, RepeatedRestoreRebuildsQueuesPointersAndEveryCapacityByte) {
    ASSERT_NO_FATAL_FAILURE(prepare_slot());
    const auto source = packet.storage;
    ASSERT_EQ(restore(), hbg::GraphRestoreStatus::Ok);
    EXPECT_EQ(restored.generation, 1u);
    auto *runtime = restored.runtime;
    EXPECT_EQ(runtime->sm_handle->header->tasks.total_tasks, packet_header().total_tasks);
    EXPECT_EQ(runtime->scheduler->sm_header, runtime->sm_handle->header);
    auto &queue = runtime->scheduler->graph_ready_queue;
    EXPECT_EQ(queue.slots[queue.capacity - 1].sequence.load(), queue.capacity - 1);
    const auto expected = mutable_bytes();
    for (uint64_t iteration = 2; iteration <= 16; ++iteration) {
        ASSERT_NO_FATAL_FAILURE(retire());
        dirty_working();
        ASSERT_EQ(restore(), hbg::GraphRestoreStatus::Ok);
        EXPECT_EQ(restored.runtime, runtime);
        EXPECT_EQ(restored.generation, iteration);
        EXPECT_EQ(mutable_bytes(), expected);
        EXPECT_EQ(packet.storage, source);
        hbg::GraphRestoreResult peer;
        EXPECT_EQ(hbg::acquire_graph_restore_result(registry, iteration - 1, peer), hbg::GraphRestoreStatus::NotReady);
        EXPECT_EQ(hbg::acquire_graph_restore_result(registry, iteration, peer), hbg::GraphRestoreStatus::Ok);
        EXPECT_EQ(peer.runtime, runtime);
    }
}

TEST_F(HbgGraphRestoreTest, SourceCorruptionOnFirstMiddleAndLastLineCannotWriteDestinations) {
    ASSERT_NO_FATAL_FAILURE(prepare_slot());
    const auto source = packet.storage;
    const auto before = working_bytes();
    const size_t begin = packet.data_offset;
    for (size_t offset : {begin, begin + (packet.bytes - begin) / 2, packet.bytes - 1}) {
        packet.storage = source;
        reinterpret_cast<std::byte *>(packet.storage.data())[offset] ^= std::byte{0x80};
        restored.generation = 999;
        EXPECT_EQ(restore(), hbg::GraphRestoreStatus::Rejected);
        EXPECT_EQ(restored.generation, 999u);
        EXPECT_EQ(working_bytes(), before);
    }
}

TEST_F(HbgGraphRestoreTest, ForgedRuntimeLayoutAndRelativePoolsRejectBeforeCopy) {
    ASSERT_NO_FATAL_FAILURE(prepare_slot());
    const auto original = packet.storage;
    const auto before = working_bytes();
    auto corrupt = [&](size_t offset, uint64_t value, size_t size) {
        packet.storage = original;
        std::memcpy(reinterpret_cast<std::byte *>(packet.storage.data()) + offset, &value, size);
        patch();
        EXPECT_EQ(restore(), hbg::GraphRestoreStatus::InvalidImage);
        EXPECT_EQ(working_bytes(), before);
    };
    const size_t runtime = packet.data_offset + packet_header().runtime_offset;
    corrupt(
        runtime + offsetof(RuntimeContext, prebuilt_layout) + offsetof(RuntimeArenaLayout, off_scheduler), UINT64_MAX, 8
    );
    corrupt(runtime + offsetof(RuntimeContext, ops), 0x1234, sizeof(void *));
    const size_t sm = packet.data_offset + packet_header().sm_offset;
    corrupt(sm + offsetof(SharedMemoryTaskHeader, total_tasks), 999, 4);
    const auto offsets = sm_layout::segment_offsets(sm_layout::image_extents({packet_header().total_tasks, 0, 0, 0}));
    corrupt(sm + offsets.storage + offsetof(ChipTaskStorage, payload) + offsetof(TaskPayload, tensors), INT32_MAX, 4);
}

TEST_F(HbgGraphRestoreTest, HostOnlyCopyCannotBeSmuggledIntoTheDeviceGraphImage) {
    ASSERT_NO_FATAL_FAILURE(prepare_slot());
    const auto before = working_bytes();
    auto *base = reinterpret_cast<std::byte *>(packet.storage.data());
    const size_t sm = packet.data_offset + packet_header().sm_offset;
    const auto offsets = sm_layout::segment_offsets(sm_layout::image_extents({packet_header().total_tasks, 0, 0, 0}));
    const size_t payload = sm + offsets.storage + offsetof(ChipTaskStorage, payload);
    const size_t tensor_field = payload + offsetof(TaskPayload, tensors);
    int32_t tensor_delta = 0;
    std::memcpy(&tensor_delta, base + tensor_field, sizeof(tensor_delta));
    ASSERT_GT(tensor_delta, 0);
    const size_t tensor = tensor_field + static_cast<uint32_t>(tensor_delta);
    base[tensor + offsetof(GraphTensor, address_space)] = std::byte{static_cast<uint8_t>(AddressSpace::HOST)};
    patch();

    EXPECT_EQ(restore(), hbg::GraphRestoreStatus::InvalidImage);
    EXPECT_EQ(working_bytes(), before);
}

TEST_F(HbgGraphRestoreTest, PartialMemoryFailuresNeverCommitAndFullRetrySucceeds) {
    ASSERT_NO_FATAL_FAILURE(prepare_slot());
    ASSERT_EQ(restore(), hbg::GraphRestoreStatus::Ok);
    struct Fault {
        int step{0};
        int fail_at{0};
        static bool copy(void *ctx, void *dst, const void *src, size_t size) {
            auto &self = *static_cast<Fault *>(ctx);
            std::memcpy(dst, src, size);
            return ++self.step != self.fail_at;
        }
        static bool zero(void *ctx, void *dst, size_t size) {
            auto &self = *static_cast<Fault *>(ctx);
            std::memset(dst, 0, size);
            return ++self.step != self.fail_at;
        }
        static bool flush(void *ctx, const void *, size_t) {
            auto &self = *static_cast<Fault *>(ctx);
            return ++self.step != self.fail_at;
        }
    };
    int regions = 0;
    for (const auto &dst : seal.destinations)
        regions += dst.capacity != 0;
    const auto expected = mutable_bytes();
    for (int fail_at = 1; fail_at <= 2 * regions + 1; ++fail_at) {
        const uint64_t previous = restored.generation;
        const auto output = restored;
        Fault fault{0, fail_at};
        ASSERT_NO_FATAL_FAILURE(retire());
        dirty_working();
        EXPECT_EQ(restore({&fault, Fault::copy, Fault::zero, Fault::flush}), hbg::GraphRestoreStatus::CopyFailed);
        EXPECT_EQ(restored.generation, output.generation);
        EXPECT_EQ(restored.runtime, output.runtime);
        EXPECT_EQ(registry->restore.committed_generation, previous);
        EXPECT_EQ(registry->restore.phase, static_cast<uint32_t>(hbg::GraphRestorePhase::Failed));
        hbg::GraphRestoreResult peer;
        EXPECT_EQ(hbg::acquire_graph_restore_result(registry, previous, peer), hbg::GraphRestoreStatus::NotReady);
        EXPECT_EQ(restore(), hbg::GraphRestoreStatus::Quarantined);
        ASSERT_NO_FATAL_FAILURE(retire(hbg::GraphRestoreRetirement::ControlledFailure));
        EXPECT_EQ(restore(), hbg::GraphRestoreStatus::Ok);
        EXPECT_EQ(restored.generation, previous + 2);
        EXPECT_EQ(mutable_bytes(), expected);
    }
}

TEST_F(HbgGraphRestoreTest, LeaderPublicationAllowsPeersToObserveCompleteWorkingMemory) {
    ASSERT_NO_FATAL_FAILURE(prepare_slot());
    std::promise<uint64_t> publication;
    auto generation = publication.get_future().share();
    std::vector<std::future<bool>> readers;
    for (int i = 0; i < 4; ++i) {
        readers.push_back(std::async(std::launch::async, [&] {
            hbg::GraphRestoreResult out;
            const auto ticket = generation.get();
            if (hbg::acquire_graph_restore_result(registry, ticket, out) != hbg::GraphRestoreStatus::Ok) return false;
            return out.runtime->sm_handle->header->tasks.total_tasks == static_cast<int32_t>(out.total_tasks) &&
                   out.runtime->scheduler->graph_ready_queue.slots[63].sequence.load() == 63;
        }));
    }
    const auto status = restore();
    publication.set_value(status == hbg::GraphRestoreStatus::Ok ? restored.generation : 0);
    EXPECT_EQ(status, hbg::GraphRestoreStatus::Ok);
    for (auto &reader : readers)
        EXPECT_TRUE(reader.get());
}

TEST_F(HbgGraphRestoreTest, SmallerAndEmptyGraphsInitializeOnlyLiveWorkingState) {
    ASSERT_EQ(build(empty_entry), 0);
    RuntimeArenaLayout layout{};
    ASSERT_EQ(hbg::make_kernel_graph_layout(capacity, layout), 0);
    hbg::GraphResourceRequirements empty{};
    ASSERT_EQ(hbg::get_graph_resource_requirements(result, layout, empty), 0);
    // A5 reserves its flat scheduler for the empty graph and uses the Graph
    // fallback for the nonempty graph. Prepare accounts for both variants.
    ASSERT_NO_FATAL_FAILURE(prepare_slot(true, empty.scheduler_state_bytes));
    ASSERT_EQ(restore(), hbg::GraphRestoreStatus::Ok);
    ASSERT_NO_FATAL_FAILURE(retire());
    ASSERT_EQ(build(empty_entry), 0);
    ASSERT_EQ(snapshot_graph(), 0);
    ASSERT_EQ(hbg::make_graph_host_args(snapshot, packet), 0);
    patch();
    dirty_working();
    ASSERT_EQ(restore(), hbg::GraphRestoreStatus::Ok);
    EXPECT_EQ(restored.total_tasks, 0u);
    EXPECT_EQ(restored.runtime->sm_handle->header->tasks.total_tasks, 0);
    for (size_t i : {size_t{0}, size_t{3}}) {
        const auto &dst = seal.destinations[i];
        if (!dst.capacity) continue;
        const auto *bytes = reinterpret_cast<const uint8_t *>(dst.address);
        const uint64_t live = i == 0 ? empty.gm_heap_bytes : empty.scheduler_state_bytes;
        EXPECT_TRUE(std::all_of(bytes, bytes + live, [](auto value) {
            return value == 0;
        }));
        EXPECT_TRUE(std::all_of(bytes + live, bytes + dst.capacity, [](auto value) {
            return value == 0xa5;
        }));
    }
}

TEST_F(HbgGraphRestoreTest, BusyAndExhaustedControlCannotWriteWorkingImages) {
    ASSERT_NO_FATAL_FAILURE(prepare_slot());
    registry->restore.phase = static_cast<uint32_t>(hbg::GraphRestorePhase::Restoring);
    const auto before = working_bytes();
    EXPECT_EQ(restore(), hbg::GraphRestoreStatus::Busy);
    EXPECT_EQ(working_bytes(), before);
    registry->restore.phase = static_cast<uint32_t>(hbg::GraphRestorePhase::Idle);
    registry->restore.attempt = UINT64_MAX;
    EXPECT_EQ(restore(), hbg::GraphRestoreStatus::Exhausted);
}

TEST_F(HbgGraphSlotTest, InvalidatesCompleteTaskPacketBeforeParsing) {
    ASSERT_NO_FATAL_FAILURE(prepare_slot());
    const auto fresh = packet.storage;
    const auto before = working_bytes();
    std::memset(packet.storage.data(), 0xa5, packet.bytes);
    struct Visibility {
        const std::vector<uint64_t> &fresh;
        const void *address;
        size_t bytes;
        int calls{0};
    } visibility{fresh, packet.storage.data(), packet.bytes};
    const hbg::GraphPacketReadOps ops{&visibility, [](void *opaque, const void *address, size_t bytes) {
                                          auto &v = *static_cast<Visibility *>(opaque);
                                          EXPECT_EQ(address, v.address);
                                          EXPECT_EQ(bytes, v.bytes);
                                          ++v.calls;
                                          std::memcpy(const_cast<void *>(address), v.fresh.data(), bytes);
                                          return true;
                                      }};
    hbg::GraphRestoreView view;
    EXPECT_EQ(
        hbg::admit_graph_packet_for_restore(packet.storage.data(), packet.bytes, 0, 109, trusted_callable, view, ops),
        hbg::GraphSlotStatus::Ok
    );
    EXPECT_EQ(visibility.calls, 1);
    EXPECT_EQ(working_bytes(), before);
}

TEST_F(HbgGraphSlotTest, PoisonIsTerminalAcrossBindRegisterAndReinitialize) {
    ASSERT_NO_FATAL_FAILURE(prepare_slot());
    ASSERT_EQ(hbg::poison_graph_execution_slot(registry), hbg::GraphSlotStatus::Ok);
    const auto before = working_bytes();
    expect_rejected(hbg::GraphSlotStatus::Poisoned);
    EXPECT_EQ(hbg::register_graph_execution_slot(registry, &seal, sizeof(seal)), hbg::GraphSlotStatus::Poisoned);
    EXPECT_EQ(hbg::bind_graph_slot_registry(registry, 0, 109), hbg::GraphSlotStatus::Poisoned);
    EXPECT_EQ(hbg::initialize_graph_slot_registry(registry, 0, 19, 109), hbg::GraphSlotStatus::Conflict);
    ASSERT_TRUE(hbg::detach_graph_slot_registry(registry));
    EXPECT_EQ(working_bytes(), before);
}

TEST_F(HbgGraphSlotTest, SourceVisibilityFailureAndInvalidBoundsNeverAuthorizeRestore) {
    ASSERT_NO_FATAL_FAILURE(prepare_slot());
    const auto before = working_bytes();
    int calls = 0;
    const hbg::GraphPacketReadOps ops{&calls, [](void *opaque, const void *, size_t) {
                                          ++*static_cast<int *>(opaque);
                                          return false;
                                      }};
    hbg::GraphRestoreView view;
    view.slot.slot_generation = 999;
    auto check = [&](const void *address, size_t bytes) {
        return hbg::admit_graph_packet_for_restore(address, bytes, 0, 109, trusted_callable, view, ops);
    };
    EXPECT_EQ(check(packet.storage.data(), packet.bytes), hbg::GraphSlotStatus::SourceUnavailable);
    EXPECT_EQ(calls, 1);
    EXPECT_EQ(check(packet.storage.data(), 1), hbg::GraphSlotStatus::InvalidPacket);
    EXPECT_EQ(check(packet.storage.data(), seal.max_packet_bytes + 1), hbg::GraphSlotStatus::InvalidPacket);
    EXPECT_EQ(check(registry, packet.bytes), hbg::GraphSlotStatus::SourceOverlap);
    EXPECT_EQ(calls, 1);
    EXPECT_EQ(view.slot.slot_generation, 999u);
    EXPECT_EQ(working_bytes(), before);
}

TEST_F(HbgGraphRestoreTest, ReadyExecutionCannotBeOverwrittenBeforeRetirement) {
    ASSERT_NO_FATAL_FAILURE(prepare_slot());
    ASSERT_EQ(restore(), hbg::GraphRestoreStatus::Ok);
    const auto before = working_bytes();
    EXPECT_EQ(restore(), hbg::GraphRestoreStatus::Busy);
    EXPECT_EQ(working_bytes(), before);
}

TEST_F(HbgGraphRestoreTest, FailedRestoreCannotRetryWithoutCleanup) {
    ASSERT_NO_FATAL_FAILURE(prepare_slot());
    hbg::GraphRestoreOps ops;
    ops.zero = [](void *, void *, size_t) {
        return false;
    };
    ASSERT_EQ(restore(ops), hbg::GraphRestoreStatus::CopyFailed);
    const auto before = working_bytes();
    EXPECT_NE(restore(), hbg::GraphRestoreStatus::Ok);
    EXPECT_EQ(working_bytes(), before);
}

TEST_F(HbgGraphRestoreTest, PoisonedRegistryRevokesPublishedPeerResult) {
    ASSERT_NO_FATAL_FAILURE(prepare_slot());
    ASSERT_EQ(restore(), hbg::GraphRestoreStatus::Ok);
    ASSERT_EQ(hbg::poison_graph_execution_slot(registry), hbg::GraphSlotStatus::Ok);
    hbg::GraphRestoreResult peer;
    EXPECT_NE(hbg::acquire_graph_restore_result(registry, restored.generation, peer), hbg::GraphRestoreStatus::Ok);
    EXPECT_EQ(peer.runtime, nullptr);
}

TEST_F(HbgGraphRestoreTest, RetirementRequiresMatchingAttemptAndOutcome) {
    ASSERT_NO_FATAL_FAILURE(prepare_slot());
    ASSERT_EQ(restore(), hbg::GraphRestoreStatus::Ok);
    const auto first = restored.generation;
    const auto before = working_bytes();
    EXPECT_EQ(
        hbg::retire_graph_restore(registry, first + 1, {hbg::GraphRestoreRetirement::Completed}),
        hbg::GraphRestoreStatus::NotReady
    );
    EXPECT_EQ(
        hbg::retire_graph_restore(registry, first, {hbg::GraphRestoreRetirement::ControlledFailure}),
        hbg::GraphRestoreStatus::Rejected
    );
    EXPECT_EQ(
        hbg::retire_graph_restore(registry, first, {static_cast<hbg::GraphRestoreRetirement>(99)}),
        hbg::GraphRestoreStatus::Rejected
    );
    EXPECT_EQ(working_bytes(), before);
    ASSERT_NO_FATAL_FAILURE(retire());
    hbg::GraphRestoreResult peer;
    EXPECT_EQ(hbg::acquire_graph_restore_result(registry, first, peer), hbg::GraphRestoreStatus::NotReady);
    EXPECT_EQ(
        hbg::retire_graph_restore(registry, first, {hbg::GraphRestoreRetirement::Completed}),
        hbg::GraphRestoreStatus::NotReady
    );
    ASSERT_EQ(restore(), hbg::GraphRestoreStatus::Ok);
    EXPECT_EQ(
        hbg::retire_graph_restore(registry, first, {hbg::GraphRestoreRetirement::Completed}),
        hbg::GraphRestoreStatus::NotReady
    );
    EXPECT_EQ(restore(), hbg::GraphRestoreStatus::Busy);
}

TEST_F(HbgGraphRestoreTest, RejectionAfterRetirementCannotExposePreviousSuccess) {
    ASSERT_NO_FATAL_FAILURE(prepare_slot());
    ASSERT_EQ(restore(), hbg::GraphRestoreStatus::Ok);
    ASSERT_NO_FATAL_FAILURE(retire());
    const auto before = working_bytes();
    reinterpret_cast<std::byte *>(packet.storage.data())[packet.bytes - 1] ^= std::byte{0x80};
    EXPECT_EQ(restore(), hbg::GraphRestoreStatus::Rejected);
    EXPECT_EQ(working_bytes(), before);
    hbg::GraphRestoreResult peer;
    EXPECT_EQ(
        hbg::acquire_graph_restore_result(registry, restored.generation, peer), hbg::GraphRestoreStatus::NotReady
    );
}

TEST_F(HbgGraphRestoreTest, PublishFailureIsQuarantinedAndFatalCleanupCannotRetry) {
    ASSERT_NO_FATAL_FAILURE(prepare_slot());
    hbg::GraphRestoreOps ops;
    ops.context = registry;
    ops.flush = [](void *opaque, const void *address, size_t bytes) {
        const auto *r = static_cast<const hbg::GraphSlotRegistry *>(opaque);
        if (address != &r->restore) return true;
        EXPECT_EQ(bytes, sizeof(r->restore));
        EXPECT_EQ(r->restore.committed_generation, 0u);
        EXPECT_EQ(r->restore.phase, static_cast<uint32_t>(hbg::GraphRestorePhase::Restoring));
        return false;
    };
    EXPECT_EQ(restore(ops), hbg::GraphRestoreStatus::CopyFailed);
    EXPECT_EQ(registry->restore.committed_generation, 0u);
    EXPECT_EQ(
        hbg::retire_graph_restore(registry, registry->restore.attempt, {hbg::GraphRestoreRetirement::Completed}),
        hbg::GraphRestoreStatus::Rejected
    );
    EXPECT_EQ(restore(), hbg::GraphRestoreStatus::Quarantined);
    const auto before = mutable_bytes();
    EXPECT_EQ(
        hbg::retire_graph_restore(registry, registry->restore.attempt, {hbg::GraphRestoreRetirement::FatalFailure}),
        hbg::GraphRestoreStatus::Poisoned
    );
    EXPECT_EQ(restore(), hbg::GraphRestoreStatus::Poisoned);
    EXPECT_EQ(
        hbg::retire_graph_restore(
            registry, registry->restore.attempt, {hbg::GraphRestoreRetirement::ControlledFailure}
        ),
        hbg::GraphRestoreStatus::Poisoned
    );
    EXPECT_EQ(mutable_bytes(), before);
}

TEST_F(HbgGraphRestoreTest, FatalExecutionAfterSuccessfulRestoreRevokesPeerAccess) {
    ASSERT_NO_FATAL_FAILURE(prepare_slot());
    ASSERT_EQ(restore(), hbg::GraphRestoreStatus::Ok);
    EXPECT_EQ(
        hbg::retire_graph_restore(registry, restored.generation, {hbg::GraphRestoreRetirement::FatalFailure}),
        hbg::GraphRestoreStatus::Poisoned
    );
    hbg::GraphRestoreResult peer;
    EXPECT_EQ(
        hbg::acquire_graph_restore_result(registry, restored.generation, peer), hbg::GraphRestoreStatus::Poisoned
    );
    EXPECT_EQ(restore(), hbg::GraphRestoreStatus::Poisoned);
}

TEST_F(HbgGraphRestoreTest, PeerValidatesRegistryBeforeInvalidatingWorkingAddresses) {
    ASSERT_NO_FATAL_FAILURE(prepare_slot());
    ASSERT_EQ(restore(), hbg::GraphRestoreStatus::Ok);
    registry->registration.destinations[0].address = 1;
    hbg::GraphRestoreResult peer;
    EXPECT_EQ(
        hbg::acquire_graph_restore_result(registry, restored.generation, peer), hbg::GraphRestoreStatus::NotReady
    );
    EXPECT_EQ(peer.runtime, nullptr);
}

TEST_F(HbgGraphRestoreTest, ControlledRestoreFailureCannotHideNativeRuntimeError) {
    ASSERT_NO_FATAL_FAILURE(prepare_slot());
    hbg::GraphRestoreOps ops;
    ops.zero = [](void *, void *, size_t) {
        return false;
    };
    ASSERT_EQ(restore(ops), hbg::GraphRestoreStatus::CopyFailed);
    EXPECT_EQ(
        hbg::retire_graph_restore(
            registry, registry->restore.attempt, {hbg::GraphRestoreRetirement::ControlledFailure, -71, 0}
        ),
        hbg::GraphRestoreStatus::Poisoned
    );
    EXPECT_EQ(restore(), hbg::GraphRestoreStatus::Poisoned);
}

TEST_F(HbgGraphRestoreTest, ControlledRestoreFailureCannotHideUnexpectedTeardownError) {
    ASSERT_NO_FATAL_FAILURE(prepare_slot());
    hbg::GraphRestoreOps ops;
    ops.zero = [](void *, void *, size_t) {
        return false;
    };
    ASSERT_EQ(restore(ops), hbg::GraphRestoreStatus::CopyFailed);
    EXPECT_EQ(
        hbg::retire_graph_restore(
            registry, registry->restore.attempt, {hbg::GraphRestoreRetirement::ControlledFailure, 0, -72}
        ),
        hbg::GraphRestoreStatus::Poisoned
    );
    EXPECT_EQ(restore(), hbg::GraphRestoreStatus::Poisoned);
}
