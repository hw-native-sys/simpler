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
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include <unistd.h>

#include "host_build_graph/graph_recorder_pool.h"
#include "host_build_graph/host_graph_build.h"
#include "host_build_graph/kernel_external_tensor.h"
#include "host_build_graph/kernel_external_tensor_wire.h"
#include "host_build_graph/kernel_callable_binding.h"
#include "host_build_graph/kernel_argument_snapshot.h"
#include "host_build_graph/kernel_graph_template.h"
#include "host_build_graph/kernel_graph_dispatch.h"
#include "host_build_graph/kernel_graph_prepared.h"
#include "host_build_graph/kernel_graph_slot.h"
#include "host_build_graph/kernel_graph_slot_registry.h"
#include "host_build_graph/kernel_graph_restore.h"
#include "host_build_graph/kernel_launch_state.h"
#include "host_build_graph/graph_execution.h"
#include "host_build_graph/host_tensor_access.h"
#include "host_build_graph/host_tensor_access_abort.h"
#include "host_build_graph/runtime_core.h"
#include "host_build_graph/runtime.h"
#include "common/host_api.h"
#include "worker/runtime_c_api.h"
#include "host/kernel_pipeline_contract.h"
#include "call_config.h"
#include "utils/fnv1a_64.h"
#include "aicpu/platform_regs.h"
#include "aicpu/args_dump_aicpu.h"
#include "aicpu/chip_swimlane_collector_aicpu.h"
#include "aicpu/dep_gen_collector_aicpu.h"
#include "aicpu/device_phase_aicpu.h"
#include "aicpu/pmu_collector_aicpu.h"
#include "aicpu/scope_stats_collector_aicpu.h"
#include "common/kernel_args.h"

extern "C" const char *const *runtime_extra_aicpu_symbols(size_t *count);
extern "C" const char *const *runtime_l1_extra_aicpu_symbols(size_t *count);
extern "C" int simpler_aicpu_l1_hbg_prepare_execution_context(void *arg);
int consume_kernel_task(void *arg);
uint64_t get_test_kernel_phase_base();

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

std::atomic<int> post_access_side_effects{0};

extern "C" bool set_test_unified_log_suppressed(bool suppressed);

class ScopedTestLogSilencer {
public:
    ScopedTestLogSilencer() :
        previously_suppressed_(set_test_unified_log_suppressed(true)) {}

    ~ScopedTestLogSilencer() { (void)set_test_unified_log_suppressed(previously_suppressed_); }

    ScopedTestLogSilencer(const ScopedTestLogSilencer &) = delete;
    ScopedTestLogSilencer &operator=(const ScopedTestLogSilencer &) = delete;

private:
    bool previously_suppressed_;
};

struct RecorderAbortProbe {
    std::mutex mutex;
    std::condition_variable cv;
    bool access_aborted{false};
    bool release_worker{false};
    std::atomic<bool> closure_destroyed{false};
};

struct RecorderClosureLifetime {
    explicit RecorderClosureLifetime(RecorderAbortProbe &probe) :
        probe(probe) {}
    ~RecorderClosureLifetime() {
        probe.closure_destroyed.store(true, std::memory_order_release);
        probe.cv.notify_all();
    }
    RecorderAbortProbe &probe;
};

RecorderAbortProbe *active_recorder_abort_probe{nullptr};

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

void read_entry_tensor_on_host(const ChipTaskArgs &args) {
    const uint32_t index[] = {0};
    (void)bound_runtime->ops->get_tensor_data(bound_runtime, args.tensor(0).ref(), 1, index);
}

void read_entry_tensor_controls_loop(const ChipTaskArgs &args) {
    const uint32_t index[] = {0};
    const uint64_t step = bound_runtime->ops->get_tensor_data(bound_runtime, args.tensor(0).ref(), 1, index);
    for (volatile uint64_t i = 0; i < 16; i += step) {}
}

void write_entry_tensor_on_host(const ChipTaskArgs &args) {
    const uint32_t index[] = {0};
    bound_runtime->ops->set_tensor_data(bound_runtime, args.tensor(0).ref(), 1, index, 7);
}

void write_entry_tensor_then_continue(const ChipTaskArgs &args) {
    write_entry_tensor_on_host(args);
    post_access_side_effects.fetch_add(1, std::memory_order_relaxed);
}

void fatal_then_read_entry(const ChipTaskArgs &args) {
    bound_runtime->orchestrator->report_fatal(SIMPLER_ERROR_EXPLICIT_ORCH_FATAL, __FUNCTION__, "%s", "test fatal");
    read_entry_tensor_on_host(args);
    post_access_side_effects.fetch_add(1, std::memory_order_relaxed);
}

template <typename Access>
int concurrent_forbidden_access_return_count(Access access) {
    constexpr int rounds = 100000;
    ScopedTestLogSilencer silence_expected_fatal_diagnostics;

    OrchestratorState orch;
    RuntimeContext rt{};
    HostTensorAccessor device_only(nullptr, HostTensorAccessMode::KernelDeviceOnly);
    rt.orchestrator = &orch;
    rt.tensor_access = &device_only;

    const uint32_t shape[] = {1};
    const auto boundary =
        make_tensor_external(reinterpret_cast<void *>(0x22000), shape, 1, DataType::UINT64, AddressSpace::DEVICE);
    const auto tensor = simpler::hbg::Tensor::from_boundary(boundary);
    const uint32_t index[] = {0};
    std::atomic<int> phase{0};
    std::atomic<int> finished{0};
    std::atomic<int> returned_values{0};

    auto accessor = [&]() {
        for (int round = 1; round <= rounds; ++round) {
            while (phase.load(std::memory_order_acquire) < round) {}
            try {
                access(rt, tensor, index);
                returned_values.fetch_add(1, std::memory_order_relaxed);
            } catch (const hbg::HostTensorAccessAbort &) {}
            finished.fetch_add(1, std::memory_order_release);
        }
    };

    std::thread first(accessor);
    std::thread second(accessor);
    for (int round = 1; round <= rounds; ++round) {
        orch.host_tensor_access_abort_requested.store(false, std::memory_order_relaxed);
        orch.fatal_code.store(SIMPLER_ERROR_NONE, std::memory_order_relaxed);
        finished.store(0, std::memory_order_relaxed);
        phase.store(round, std::memory_order_release);
        while (finished.load(std::memory_order_acquire) != 2) {}
    }
    first.join();
    second.join();
    return returned_values.load(std::memory_order_relaxed);
}

void async_graph_read_entry(const ChipTaskArgs &args) {
    auto &orch = *bound_runtime->orchestrator;
    GraphTaskArgs graph_args;
    graph_args.add_input(args.tensor(0).ref());
    const auto scope = orch.graph_begin(0x93, graph_args, bound_runtime->active_callable_hash);
    ASSERT_TRUE(scope.recording);
    ASSERT_NE(active_recorder_abort_probe, nullptr);

    RecorderAbortProbe &probe = *active_recorder_abort_probe;
    RuntimeContext *rt = bound_runtime;
    auto lifetime = std::make_shared<RecorderClosureLifetime>(probe);
    std::function<void(const GraphTaskArgs &)> job = [rt, handle = scope.recording_handle,
                                                      lifetime](const GraphTaskArgs &record_args) {
        if (!rt->orchestrator->graph_prepare(handle, record_args)) {
            rt->orchestrator->graph_abort(handle);
            return;
        }
        try {
            const uint32_t index[] = {0};
            (void)rt->ops->get_tensor_data(rt, record_args.tensor(0).ref(), 1, index);
            (void)rt->orchestrator->graph_end();
        } catch (const hbg::HostTensorAccessAbort &) {
            rt->orchestrator->graph_abort(handle);
            std::unique_lock<std::mutex> lock(lifetime->probe.mutex);
            lifetime->probe.access_aborted = true;
            lifetime->probe.cv.notify_all();
            lifetime->probe.cv.wait(lock, [&]() {
                return lifetime->probe.release_worker;
            });
        }
    };
    ASSERT_TRUE(graph_recorder_pool().start(*scope.params, std::move(job)));
    lifetime.reset();
}

TEST(HbgKernelExternalTensor, RejectsHostDeviceArgsAndUnsupportedStrides) {
    const uint32_t shape[] = {2, 3};
    const uint32_t transposed_stride[] = {1, 2};
    std::array<float, 6> host{};
    ChipStorageTaskArgs args;
    args.add_tensor(make_tensor_external(host.data(), shape, 2, DataType::FLOAT32, AddressSpace::HOST));
    EXPECT_EQ(hbg::validate_kernel_external_tensors(args), hbg::KernelExternalTensorStatus::NonDeviceTensor);

    args.clear();
    args.add_tensor(make_tensor_strided(
        reinterpret_cast<void *>(0x44000), shape, transposed_stride, 2, DataType::FLOAT32, AddressSpace::DEVICE
    ));
    EXPECT_EQ(hbg::validate_kernel_external_tensors(args), hbg::KernelExternalTensorStatus::UnsupportedStrideFamily);
}

TEST(HbgKernelExternalTensor, PreservesWideTensorArithmeticBounds) {
    GraphTensor tensor{};
    tensor.buffer_addr = 0x44000;
    tensor.buffer_size = UINT64_MAX;
    tensor.extent_elem = UINT64_MAX;
    tensor.ndims = 3;
    tensor.dtype = static_cast<uint8_t>(DataType::UINT8);
    tensor.address_space = static_cast<uint8_t>(AddressSpace::DEVICE);
    tensor.shapes[0] = UINT32_MAX;
    tensor.shapes[1] = 5;
    tensor.shapes[2] = 3;
    tensor.strides[0] = UINT32_MAX - 1;
    tensor.strides[1] = UINT32_MAX - 1;
    tensor.strides[2] = 1;
    EXPECT_TRUE(graph_tensor_wire_valid(tensor));
    tensor.shapes[2] = 4;
    tensor.extent_elem = 0;
    EXPECT_FALSE(graph_tensor_wire_valid(tensor));

    tensor.shapes[1] = UINT32_MAX;
    tensor.shapes[2] = 2;
    tensor.strides[0] = tensor.strides[1] = 1;
    tensor.extent_elem = uint64_t{2} * UINT32_MAX;
    EXPECT_FALSE(graph_tensor_wire_valid(tensor));

    tensor.ndims = 2;
    tensor.shapes[0] = tensor.shapes[1] = UINT32_MAX;
    tensor.strides[0] = UINT32_MAX;
    tensor.strides[1] = 1;
    tensor.extent_elem = uint64_t{UINT32_MAX} * UINT32_MAX;
    tensor.is_contiguous = 1;
    EXPECT_TRUE(graph_tensor_wire_valid(tensor));
    EXPECT_TRUE(hbg::supported_kernel_graph_stride_family(tensor));
    tensor.strides[1] = UINT32_MAX;
    EXPECT_FALSE(hbg::supported_kernel_graph_stride_family(tensor));

    tensor.ndims = 3;
    tensor.shapes[1] = 641;
    tensor.shapes[2] = 6700417;
    tensor.strides[0] = tensor.strides[1] = 1;
    tensor.extent_elem = uint64_t{UINT32_MAX} + 641 + 6700417 - 2;
    tensor.is_contiguous = 0;
    EXPECT_TRUE(graph_tensor_wire_valid(tensor));
    ++tensor.shapes[2];
    ++tensor.extent_elem;
    EXPECT_FALSE(graph_tensor_wire_valid(tensor));
}

TEST(HbgKernelExternalTensor, RejectsFlatTensorRankBeforePacking) {
    const uint32_t shape[] = {1};
    const auto boundary =
        make_tensor_external(reinterpret_cast<void *>(0x22000), shape, 1, DataType::UINT64, AddressSpace::DEVICE);
    auto tensor = simpler::hbg::Tensor::from_boundary(boundary);
    ASSERT_TRUE(hbg::valid_kernel_graph_tensor(tensor));
    ASSERT_TRUE(hbg::valid_restored_kernel_graph_tensor(tensor, 0x100000000ULL, 4096));
    for (uint32_t ndims : {uint32_t{0}, uint32_t{MAX_TENSOR_DIMS + 1}, uint32_t{256}, UINT32_MAX}) {
        SCOPED_TRACE(ndims);
        tensor.ndims = ndims;
        EXPECT_FALSE(hbg::valid_kernel_graph_tensor(tensor));
        EXPECT_FALSE(hbg::valid_restored_kernel_graph_tensor(tensor, 0x100000000ULL, 4096));
    }
}

class HostGraphBuildTest : public ::testing::Test {
protected:
    uint64_t capacity{64};
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
    int32_t build_kernel(void (*entry)(const ChipTaskArgs &), const ChipTaskArgs &args) {
        HostTensorAccessor device_only(nullptr, HostTensorAccessMode::KernelDeviceOnly);
        return hbg::build_graph(
            &runtime, device_only, {rt, mirror.data(), mirror.size(), capacity, definition_arena}, {entry, bind}, args,
            result
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

TEST_F(HostGraphBuildTest, KernelDeviceTensorReadFailsAtTheAccessSite) {
    const uint32_t shape[] = {4};
    const auto boundary =
        make_tensor_external(reinterpret_cast<void *>(0x22000), shape, 1, DataType::INT32, AddressSpace::DEVICE);
    const auto tensor = simpler::hbg::Tensor::from_boundary(boundary);
    ChipTaskArgs args;
    args.add_input(tensor);

    EXPECT_EQ(
        build_kernel(read_entry_tensor_on_host, args), runtime_status_from_error_code(SIMPLER_ERROR_INVALID_ARGS)
    );
    EXPECT_FALSE(result.build_complete);
    EXPECT_TRUE(platform.copies.empty());
    EXPECT_TRUE(platform.allocations.empty());
}

TEST_F(HostGraphBuildTest, KernelDeviceTensorReadFailureReturnsInsteadOfHangingInOrchestration) {
    const uint32_t shape[] = {1};
    const auto boundary =
        make_tensor_external(reinterpret_cast<void *>(0x22000), shape, 1, DataType::UINT64, AddressSpace::DEVICE);
    const auto tensor = simpler::hbg::Tensor::from_boundary(boundary);
    ChipTaskArgs args;
    args.add_input(tensor);

    EXPECT_EXIT(
        {
            alarm(5);
            const int status = build_kernel(read_entry_tensor_controls_loop, args);
            _exit(status == runtime_status_from_error_code(SIMPLER_ERROR_INVALID_ARGS) ? 0 : 1);
        },
        ::testing::ExitedWithCode(0), ""
    );
}

TEST_F(HostGraphBuildTest, KernelDeviceTensorWriteFailsAtTheAccessSite) {
    const uint32_t shape[] = {4};
    const auto boundary =
        make_tensor_external(reinterpret_cast<void *>(0x22000), shape, 1, DataType::INT32, AddressSpace::DEVICE);
    const auto tensor = simpler::hbg::Tensor::from_boundary(boundary);
    ChipTaskArgs args;
    args.add_input(tensor);

    EXPECT_EQ(
        build_kernel(write_entry_tensor_on_host, args), runtime_status_from_error_code(SIMPLER_ERROR_INVALID_ARGS)
    );
    EXPECT_FALSE(result.build_complete);
    EXPECT_TRUE(platform.copies.empty());
    EXPECT_TRUE(platform.allocations.empty());
}

TEST_F(HostGraphBuildTest, KernelDeviceTensorWriteDoesNotContinueUserControlFlow) {
    const uint32_t shape[] = {1};
    const auto boundary =
        make_tensor_external(reinterpret_cast<void *>(0x22000), shape, 1, DataType::UINT64, AddressSpace::DEVICE);
    const auto tensor = simpler::hbg::Tensor::from_boundary(boundary);
    ChipTaskArgs args;
    args.add_input(tensor);
    post_access_side_effects.store(0, std::memory_order_relaxed);

    EXPECT_EQ(
        build_kernel(write_entry_tensor_then_continue, args), runtime_status_from_error_code(SIMPLER_ERROR_INVALID_ARGS)
    );
    EXPECT_EQ(post_access_side_effects.load(std::memory_order_relaxed), 0);
}

TEST_F(HostGraphBuildTest, OrdinaryFatalKeepsExistingPostFatalNoOpBehavior) {
    const uint32_t shape[] = {1};
    const auto boundary =
        make_tensor_external(reinterpret_cast<void *>(0x22000), shape, 1, DataType::UINT64, AddressSpace::DEVICE);
    const auto tensor = simpler::hbg::Tensor::from_boundary(boundary);
    ChipTaskArgs args;
    args.add_input(tensor);
    post_access_side_effects.store(0, std::memory_order_relaxed);

    EXPECT_EQ(
        build_kernel(fatal_then_read_entry, args), runtime_status_from_error_code(SIMPLER_ERROR_EXPLICIT_ORCH_FATAL)
    );
    EXPECT_EQ(post_access_side_effects.load(std::memory_order_relaxed), 1);
}

TEST(HbgHostAccessAbortTest, ConcurrentForbiddenReadsNeverReturnAFallbackValue) {
    const int returned =
        concurrent_forbidden_access_return_count([](RuntimeContext &rt, const simpler::hbg::Tensor &tensor,
                                                    const uint32_t index[]) {
            (void)get_tensor_data(&rt, tensor, 1, index);
        });
    EXPECT_EQ(returned, 0);
}

TEST(HbgHostAccessAbortTest, ConcurrentForbiddenWritesNeverReturnNormally) {
    const int returned =
        concurrent_forbidden_access_return_count([](RuntimeContext &rt, const simpler::hbg::Tensor &tensor,
                                                    const uint32_t index[]) {
            set_tensor_data(&rt, tensor, 1, index, 7);
        });
    EXPECT_EQ(returned, 0);
}

TEST_F(HostGraphBuildTest, FailedAsyncRecordingIsDrainedBeforeBuildReturnsAndCanRecover) {
    const uint32_t shape[] = {1};
    const auto boundary =
        make_tensor_external(reinterpret_cast<void *>(0x22000), shape, 1, DataType::UINT64, AddressSpace::DEVICE);
    const auto tensor = simpler::hbg::Tensor::from_boundary(boundary);
    ChipTaskArgs args;
    args.add_input(tensor);
    RecorderAbortProbe probe;
    active_recorder_abort_probe = &probe;

    auto build_future = std::async(std::launch::async, [&]() {
        return build_kernel(async_graph_read_entry, args);
    });
    bool saw_abort = false;
    {
        std::unique_lock<std::mutex> lock(probe.mutex);
        saw_abort = probe.cv.wait_for(lock, std::chrono::seconds(5), [&]() {
            return probe.access_aborted;
        });
    }
    EXPECT_TRUE(saw_abort);
    EXPECT_EQ(build_future.wait_for(std::chrono::milliseconds(50)), std::future_status::timeout)
        << "build must retain the orchestration image until the recorder closure is destroyed";
    {
        std::lock_guard<std::mutex> lock(probe.mutex);
        probe.release_worker = true;
    }
    probe.cv.notify_all();
    ASSERT_EQ(build_future.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    EXPECT_EQ(build_future.get(), runtime_status_from_error_code(SIMPLER_ERROR_INVALID_ARGS));
    EXPECT_TRUE(probe.closure_destroyed.load(std::memory_order_acquire));
    active_recorder_abort_probe = nullptr;

    EXPECT_EQ(build(empty_entry), 0) << "a failed build must not poison the next orchestration";
    EXPECT_TRUE(result.build_complete);
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
    EXPECT_EQ(after.abi_version, original.abi_version);
    EXPECT_EQ(after.resource_count, original.resource_count);
    EXPECT_EQ(after.pipeline_depth, original.pipeline_depth);
    for (uint32_t i = 0; i < original.resource_count; ++i) {
        EXPECT_EQ(after.resources[i].kind, original.resources[i].kind);
        EXPECT_EQ(after.resources[i].resource_class, original.resources[i].resource_class);
        EXPECT_EQ(after.resources[i].bytes_per_copy, original.resources[i].bytes_per_copy);
    }
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
    hbg::HbgCallableRegistration dispatch_registration() const {
        const auto graph = header();
        hbg::HbgCallableRegistration registration{};
        registration.context_generation = graph.slot_generation;
        registration.runtime_address = 0x1000;
        registration.register_table_address = 0x2000;
        registration.callable_address = 0x3000;
        registration.callable_bytes = sizeof(ChipCallable);
        registration.callable_hash = identity.callable_hash;
        registration.function_hash = identity.function_hash;
        registration.callable_id = identity.callable_id;
        registration.tensor_count = identity.tensor_count;
        registration.scalar_count = identity.scalar_count;
        registration.device_id = graph.device_id;
        registration.runtime_binary_id = graph.runtime_binary_id;
        return registration;
    }
};

TEST(HbgGraphPacket, ChecksumMatchesOfficialXxh64AcrossAlignmentsAndTails) {
    EXPECT_EQ(hbg::graph_packet_checksum(nullptr, 0), 0u);
    EXPECT_EQ(hbg::graph_packet_checksum(nullptr, 512), 0u);
    std::array<uint8_t, 80> short_packet{};
    for (size_t bytes = 0; bytes < short_packet.size(); ++bytes)
        EXPECT_EQ(hbg::graph_packet_checksum(short_packet.data(), bytes), 0u);
    struct Golden {
        size_t bytes;
        uint64_t hashes[3];
    };
    // Official xxHash v0.8.3 XXH64(reference[0:64] + reference[80:], seed=0).
    // Patterns: zero, 0xff, ((i * 37 + 11) ^ (i >> 3)) & 0xff, at physical offset i.
    const Golden golden[] = {
        {80, {0x257b09a147b82a19ULL, 0x2e54fae2959dbb84ULL, 0xc24252c19a5eddf4ULL}},
        {81, {0xd033cd270447f937ULL, 0x6ed2445b4b82d6cbULL, 0x153fdcd9e3754d09ULL}},
        {82, {0x1f3015449a5a2480ULL, 0x8bc9b76d9da69a7bULL, 0x7a3ba22f5e1bf612ULL}},
        {83, {0xf8f20e020412c80aULL, 0xc9d2770133220dd7ULL, 0x7e031e357e0a1d0bULL}},
        {84, {0x0a679d6f6e1d084cULL, 0x6e09cc8e7bc88046ULL, 0x25c7d3c0da75a4eeULL}},
        {85, {0x221b3e9c66ed049bULL, 0xfb2ea409d22024f8ULL, 0xf39a5fed90d7fd3aULL}},
        {86, {0xcc19586bc6b6659eULL, 0x723f58f37bcb48f3ULL, 0x2cc56907d95cc8d5ULL}},
        {87, {0x462b2c1d5cc67d0dULL, 0x2be8c4abdd7444e6ULL, 0x125ed752fc98afbeULL}},
        {88, {0xd2c15e19901d658eULL, 0x54da385251eed7cfULL, 0x2f2aab41c5e200c2ULL}},
        {89, {0xbe88019ce5de71b6ULL, 0x44116994dc3b5dbcULL, 0x26b4d4d21d182ef7ULL}},
        {90, {0x122d7e563f7ebe53ULL, 0xa534b3306e19a362ULL, 0x8ff61a79a5224c57ULL}},
        {91, {0xdfb1b23670a37f6bULL, 0x518a1cdd1cae6d89ULL, 0x37c06e3c0c06f387ULL}},
        {92, {0x6d4dc250a2d71dd8ULL, 0xb545728891acd0c0ULL, 0xf8abb43834759dd9ULL}},
        {93, {0x767323aa514a4b3eULL, 0x8bd834b970eae746ULL, 0xbaea1f7652503622ULL}},
        {94, {0xd02a55fbcdd78515ULL, 0xaeb31e1dfe50ac04ULL, 0x459556d0ced51f50ULL}},
        {95, {0x4b8cef055d638a36ULL, 0xb028ba6e929900edULL, 0xcb644d4939c0150dULL}},
        {96, {0xee6d208b78ba5eaaULL, 0xa576860c8bf28ba7ULL, 0xd47796f278d87812ULL}},
        {97, {0x31e926817a39841bULL, 0x00dd17f6611db32cULL, 0xd7fa9c58f8a9106bULL}},
        {98, {0x9ff396b3135b456cULL, 0xb40ace2a18c04691ULL, 0xb1a54cfc3be5c6e8ULL}},
        {99, {0x5c0413dac1b3b939ULL, 0xf632c1001a64972cULL, 0xef3d78b38bba4032ULL}},
        {100, {0x1498e3f1d1af7e35ULL, 0xdbb87b2bd21282adULL, 0x2d7c65de5f2a5c61ULL}},
        {101, {0x4e6fe85182d48a10ULL, 0x563054d9390a6244ULL, 0x23b34c4800e50d24ULL}},
        {102, {0x023314074cb17f3aULL, 0xef02095399d77e5eULL, 0x2202f17dc88f2bf0ULL}},
        {103, {0xa5fa1a57f86e2821ULL, 0xed595fa6df97559aULL, 0xd34be233ac54c626ULL}},
        {104, {0x90544ddf7f0428ebULL, 0x60ffc85a1a4db56dULL, 0x0367d4dd913dae03ULL}},
        {105, {0xfad4f662b43ce68cULL, 0x224930fefa4ffc9fULL, 0x203e8a1771bdbe3eULL}},
        {106, {0xbfc627f903045881ULL, 0x2f95c9fbc9aad2f6ULL, 0x7141cae5692b7a94ULL}},
        {107, {0x5f6042db52039ba9ULL, 0x9c3c97840ef928f9ULL, 0xfc88a4f267b279d5ULL}},
        {108, {0x4808152f82cfb223ULL, 0xc93b4e94fd6a4bc8ULL, 0x6c2c4d4c62276930ULL}},
        {109, {0xf3d57b3fcfda5974ULL, 0x98899143b63f1fafULL, 0xfad3ca84a98eb604ULL}},
        {110, {0x5e2157cc7eabc1c6ULL, 0x75c6e358269872ecULL, 0x1dcfc200dc0dfd82ULL}},
        {111, {0x59f913c719e77988ULL, 0xbef7e10175af1391ULL, 0x2ea711814a1bd15bULL}},
        {112, {0xc088fb75504a22bfULL, 0xa4c923881794a94fULL, 0x5be3f26dd8ca09d1ULL}},
        {113, {0x2f3a12dcabb0f60bULL, 0xbb0bd4e56a3cab3dULL, 0xbe1bde904930d035ULL}},
        {127, {0x51c55aadcba25168ULL, 0x50378970a1e9ba43ULL, 0x21b38d4bb79ab177ULL}},
        {128, {0x8ff8fc9514e3a9c1ULL, 0x8b2949dc1458af05ULL, 0xe6fd9cd96ea7bfcbULL}},
        {129, {0xd0bde90e5fab3ff4ULL, 0xaadb5643789b5af6ULL, 0xde2a8315756601e4ULL}},
        {255, {0xb5ea2ee9848886e1ULL, 0x02ebc97799cfb588ULL, 0xd30e238cdec40906ULL}},
        {256, {0x3b2f9b86d7a3505dULL, 0x67da7d82d095af3cULL, 0x8aa91e14d42ba13aULL}},
        {257, {0xce57013ff2e37492ULL, 0x6688fa92212896ddULL, 0x0cf01c39c032ca23ULL}},
        {513, {0xfd9e18a73f717a2cULL, 0xd4348b57b6d89568ULL, 0x857ee35c2b8047ffULL}},
        {1024, {0xb0ee0eb5ed26f0eeULL, 0xbbedde8b00c972e0ULL, 0x2f832a9db4d8f88fULL}},
        {84000, {0x303e48c0241fa756ULL, 0x383e395b79a44ce5ULL, 0xe4958b533bac1688ULL}},
    };
    for (const auto &entry : golden) {
        for (size_t pattern = 0; pattern < 3; ++pattern) {
            for (size_t alignment = 0; alignment < sizeof(uint64_t); ++alignment) {
                // The packet ends at the allocation boundary for sanitizer overread checks.
                auto storage = std::make_unique<uint8_t[]>(alignment + entry.bytes);
                auto *packet = storage.get() + alignment;
                for (size_t i = 0; i < entry.bytes; ++i)
                    packet[i] = pattern == 0 ? 0 : pattern == 1 ? 0xff : static_cast<uint8_t>((i * 37 + 11) ^ (i >> 3));
                ASSERT_EQ(hbg::graph_packet_checksum(packet, entry.bytes), entry.hashes[pattern])
                    << "pattern=" << pattern << " alignment=" << alignment << " bytes=" << entry.bytes;
            }
        }
    }
}

TEST(HbgGraphPacket, ChecksumExcludesOnlyPatchedAddressAndChecksumBytes) {
    constexpr size_t skip =
        sizeof(SimplerKernelInvocationHeader) + offsetof(hbg::GraphPacketHeader, inline_payload_addr);
    std::vector<uint8_t> packet(513, 0);
    const uint64_t baseline = hbg::graph_packet_checksum(packet.data(), packet.size());
    for (size_t offset = 0; offset < packet.size(); ++offset) {
        for (unsigned bit = 0; bit < 8; ++bit) {
            packet[offset] = static_cast<uint8_t>(1u << bit);
            const uint64_t checksum = hbg::graph_packet_checksum(packet.data(), packet.size());
            if (offset >= skip && offset < skip + 16) {
                EXPECT_EQ(checksum, baseline) << "offset=" << offset << " bit=" << bit;
            } else {
                EXPECT_NE(checksum, baseline) << "offset=" << offset << " bit=" << bit;
            }
            packet[offset] = 0;
        }
    }
}

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

TEST_F(HbgGraphPacketTest, DispatchKeepsCanonicalPacketAndRelocatesBothPlaceholders) {
    ASSERT_EQ(build(chain_entry), 2);
    prepare();
    ASSERT_EQ(snapshot_graph(), 0);
    const auto original = bytes();
    const auto registration = dispatch_registration();
    hbg::GraphHostArgs canonical;
    hbg::GraphHostArgs args;
    ASSERT_EQ(hbg::make_graph_host_args(snapshot, canonical), 0);
    ASSERT_EQ(hbg::make_graph_dispatch_args(snapshot, registration, args), 0);
    hbg::GraphDispatchHeader dispatch{};
    ASSERT_TRUE(hbg::read_graph_dispatch_header(args.storage.data(), sizeof(dispatch), dispatch));
    EXPECT_EQ(std::memcmp(&dispatch.registration, &registration, sizeof(registration)), 0);
    EXPECT_EQ(args.bytes, sizeof(dispatch) + original.size());
    EXPECT_EQ(args.address_offset, sizeof(dispatch) + canonical.address_offset);
    EXPECT_EQ(args.data_offset, sizeof(dispatch) + canonical.data_offset);
    auto *inner = reinterpret_cast<std::byte *>(args.storage.data()) + sizeof(dispatch);
    EXPECT_EQ(std::memcmp(inner, original.data(), original.size()), 0);
    const uint64_t address = reinterpret_cast<uintptr_t>(args.storage.data()) + args.data_offset;
    std::memcpy(reinterpret_cast<std::byte *>(args.storage.data()) + args.address_offset, &address, sizeof(address));
    EXPECT_EQ(
        hbg::validate_graph_packet(inner, original.size(), hbg::GraphPacketAddress::DeviceCopy),
        hbg::GraphPacketStatus::Ok
    );
    EXPECT_TRUE(hbg::read_graph_dispatch_header(args.storage.data(), sizeof(dispatch), dispatch));
    inner[original.size() - 1] ^= std::byte{1};
    EXPECT_EQ(
        hbg::validate_graph_packet(inner, original.size(), hbg::GraphPacketAddress::DeviceCopy),
        hbg::GraphPacketStatus::InvalidChecksum
    );
    EXPECT_EQ(bytes(), original);
    const auto previous = args.storage;
    auto conflict = registration;
    ++conflict.function_hash;
    EXPECT_EQ(hbg::make_graph_dispatch_args(snapshot, conflict, args), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_EQ(args.storage, previous);
}

TEST_F(HbgGraphPacketTest, DispatchPrefixRejectsDamageAndFramingWithoutChangingOutput) {
    ASSERT_EQ(build(chain_entry), 2);
    prepare();
    ASSERT_EQ(snapshot_graph(), 0);
    hbg::GraphHostArgs args;
    ASSERT_EQ(hbg::make_graph_dispatch_args(snapshot, dispatch_registration(), args), 0);
    hbg::GraphDispatchHeader good{};
    ASSERT_TRUE(hbg::read_graph_dispatch_header(args.storage.data(), sizeof(good), good));
    hbg::GraphDispatchHeader output{};
    std::memset(static_cast<void *>(&output), 0xa5, sizeof(output));
    const auto unchanged = output;
    auto rejected = [&](const void *data, size_t size) {
        EXPECT_FALSE(hbg::read_graph_dispatch_header(data, size, output));
        EXPECT_EQ(std::memcmp(&output, &unchanged, sizeof(output)), 0);
    };
    rejected(nullptr, sizeof(good));
    for (size_t size = 0; size < sizeof(good); ++size)
        rejected(&good, size);
    for (size_t alignment = 0; alignment < 8; ++alignment) {
        auto exact = std::make_unique<uint8_t[]>(sizeof(good) + alignment);
        auto *data = exact.get() + alignment;
        std::memcpy(data, &good, sizeof(good));
        hbg::GraphDispatchHeader parsed{};
        ASSERT_TRUE(hbg::read_graph_dispatch_header(data, sizeof(good), parsed));
        EXPECT_EQ(std::memcmp(&parsed, &good, sizeof(good)), 0);
        for (size_t byte = 0; byte < sizeof(good); ++byte) {
            for (unsigned bit = 0; bit < 8; ++bit) {
                data[byte] ^= uint8_t{1} << bit;
                rejected(data, sizeof(good));
                data[byte] ^= uint8_t{1} << bit;
            }
        }
    }
    std::vector<hbg::GraphDispatchHeader> invalid;
    auto bad = good;
    bad.version++;
    invalid.push_back(bad);
    bad = good;
    bad.header_bytes--;
    invalid.push_back(bad);
    bad = good;
    bad.packet_bytes = UINT64_MAX;
    invalid.push_back(bad);
    bad = good;
    bad.packet_bytes = sizeof(SimplerKernelInvocationHeader) + sizeof(hbg::GraphPacketHeader) - 1;
    invalid.push_back(bad);
    bad = good;
    bad.invocation.reserved_ = 1;
    invalid.push_back(bad);
    bad = good;
    bad.invocation.host_copy_tensor_count = 1;
    invalid.push_back(bad);
    bad = good;
    bad.registration.context_generation = 0;
    invalid.push_back(bad);
    bad = good;
    bad.registration.tensor_count++;
    invalid.push_back(bad);
    for (auto &candidate : invalid) {
        candidate.checksum = hbg::graph_dispatch_checksum(candidate);
        rejected(&candidate, sizeof(candidate));
    }
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
    EXPECT_EQ(image.destination_offset, h.runtime_offset);
    EXPECT_EQ(image.bytes, h.sm_offset + result.image_bytes - h.runtime_offset);
    RuntimeContext pristine{};
    std::memcpy(&pristine, payload, sizeof(pristine));
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

TEST_F(HbgGraphPacketTest, RuntimeImageExcludesDeviceInitializedPrefix) {
    ASSERT_EQ(build(chain_entry), 2);
    prepare();
    ASSERT_EQ(snapshot_graph(), 0);
    const auto h = header();
    hbg::GraphImageRegion image{};
    std::memcpy(
        &image, static_cast<const std::byte *>(snapshot.data()) + sizeof(SimplerKernelInvocationHeader) + sizeof(h),
        sizeof(image)
    );
    EXPECT_EQ(image.destination_offset, h.runtime_offset);
    EXPECT_EQ(image.bytes, h.sm_offset + result.image_bytes - h.runtime_offset);
    EXPECT_EQ(required.runtime_arena_bytes, h.sm_offset + result.image_bytes);
    EXPECT_GE(h.destinations[1].capacity, required.runtime_arena_bytes);
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
        relocated.data() + h.runtime_offset,
        static_cast<const std::byte *>(snapshot.data()) + sizeof(SimplerKernelInvocationHeader) + h.payload_offset,
        h.sm_offset + result.image_bytes - h.runtime_offset
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
            h.version = 3;
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
    const auto registration = dispatch_registration();
    auto copy_dispatch = [&]() {
        hbg::GraphHostArgs args;
        if (hbg::make_graph_dispatch_args(snapshot, registration, args) != 0) return false;
        hbg::GraphDispatchHeader dispatch{};
        if (!hbg::read_graph_dispatch_header(args.storage.data(), sizeof(dispatch), dispatch)) return false;
        const auto *inner = reinterpret_cast<const std::byte *>(args.storage.data()) + sizeof(dispatch);
        return hbg::validate_graph_packet(inner, dispatch.packet_bytes, hbg::GraphPacketAddress::HostTemplate) ==
               hbg::GraphPacketStatus::Ok;
    };
    auto dispatch_first = std::async(std::launch::async, copy_dispatch);
    auto dispatch_second = std::async(std::launch::async, copy_dispatch);
    EXPECT_TRUE(dispatch_first.get());
    EXPECT_TRUE(dispatch_second.get());
}

TEST_F(HbgGraphPacketTest, DispatchSubmissionPropagatesFailureWithoutChangingTemplate) {
    ASSERT_EQ(build(chain_entry), 2);
    prepare();
    ASSERT_EQ(snapshot_graph(), 0);
    const auto original = bytes();
    int calls = 0;
    const hbg::GraphHostLaunchOps ops{
        &calls, [](void *opaque, hbg::GraphHostArgs &args) -> int {
            ++*static_cast<int *>(opaque);
            hbg::GraphDispatchHeader dispatch{};
            EXPECT_TRUE(hbg::read_graph_dispatch_header(args.storage.data(), sizeof(dispatch), dispatch));
            const uint64_t address = reinterpret_cast<uintptr_t>(args.storage.data()) + args.data_offset;
            std::memcpy(
                reinterpret_cast<std::byte *>(args.storage.data()) + args.address_offset, &address, sizeof(address)
            );
            const auto *inner = reinterpret_cast<const std::byte *>(args.storage.data()) + sizeof(dispatch);
            EXPECT_EQ(
                hbg::validate_graph_packet(inner, dispatch.packet_bytes, hbg::GraphPacketAddress::DeviceCopy),
                hbg::GraphPacketStatus::Ok
            );
            return PTO_RUNTIME_ERR_INTERNAL;
        }
    };
    EXPECT_EQ(hbg::submit_graph_dispatch(snapshot, dispatch_registration(), ops), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_EQ(calls, 1);
    EXPECT_EQ(bytes(), original);
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
            aligned.data() + first_header.runtime_offset,
            packets[i].data() + sizeof(SimplerKernelInvocationHeader) + first_header.payload_offset,
            first_header.sm_offset + result.image_bytes - first_header.runtime_offset
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

TEST_F(HbgGraphPacketTest, EnvelopeKeepsReservedHostCopyCountZero) {
    ASSERT_EQ(build(chain_entry), 2);
    prepare();
    identity.tensor_count = 2;
    ASSERT_EQ(snapshot_graph(), 0);
    SimplerKernelInvocationHeader invocation{};
    std::memcpy(&invocation, snapshot.data(), sizeof(invocation));
    EXPECT_EQ(invocation.tensor_count, 2);
    EXPECT_EQ(invocation.host_copy_tensor_count, 0);
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
    void prepare_slot(bool publish = true, uint64_t scheduler_capacity = 0, uint64_t spare = 0) {
        ASSERT_GE(build(graph_entry), 0);
        prepare(spare, scheduler_capacity);
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

TEST_F(HbgGraphSlotTest, DispatchBoundsIncludeThePrefixAndRejectStaleContextAuthority) {
    prepare_slot();
    const auto before = working_bytes();
    auto registration = dispatch_registration();
    // Prefix/slot admission must not dereference this prepare-owned address.
    registration.runtime_address = 1;
    hbg::GraphHostArgs args;
    ASSERT_EQ(hbg::make_graph_dispatch_args(snapshot, registration, args), 0);
    hbg::GraphDispatchHeader dispatch{};
    ASSERT_TRUE(hbg::read_graph_dispatch_header(args.storage.data(), sizeof(dispatch), dispatch));
    ASSERT_TRUE(hbg::graph_dispatch_matches_slot(args.storage.data(), dispatch, seal));
    auto conflict = dispatch;
    ++conflict.registration.context_generation;
    EXPECT_FALSE(hbg::graph_dispatch_matches_slot(args.storage.data(), conflict, seal));
    conflict = dispatch;
    ++conflict.registration.device_id;
    EXPECT_FALSE(hbg::graph_dispatch_matches_slot(args.storage.data(), conflict, seal));
    conflict = dispatch;
    ++conflict.registration.runtime_binary_id;
    EXPECT_FALSE(hbg::graph_dispatch_matches_slot(args.storage.data(), conflict, seal));
    conflict = dispatch;
    conflict.packet_bytes = seal.max_packet_bytes + 1;
    EXPECT_FALSE(hbg::graph_dispatch_matches_slot(args.storage.data(), conflict, seal));
    const uint64_t begin = reinterpret_cast<uintptr_t>(args.storage.data());
    const hbg::GraphDestination inner{begin + sizeof(dispatch), dispatch.packet_bytes};
    for (size_t destination = 0; destination < 5; ++destination) {
        auto overlap = seal;
        auto &window = destination == 4 ? overlap.registry : overlap.destinations[destination];
        window = {begin, sizeof(dispatch)};
        ASSERT_FALSE(hbg::graph_windows_overlap(inner, window));
        EXPECT_FALSE(hbg::graph_dispatch_source_fits(args.storage.data(), sizeof(dispatch), overlap));
        EXPECT_FALSE(hbg::graph_dispatch_matches_slot(args.storage.data(), dispatch, overlap));
    }
    EXPECT_FALSE(hbg::graph_dispatch_source_fits(nullptr, sizeof(dispatch), seal));
    EXPECT_FALSE(hbg::graph_dispatch_source_fits(reinterpret_cast<void *>(UINTPTR_MAX - 8), sizeof(dispatch), seal));
    EXPECT_EQ(working_bytes(), before);
}

TEST_F(HbgGraphSlotTest, DispatchConflictRejectsBeforeReadingUnmappedRuntime) {
    prepare_slot();
    EXPECT_EXIT(
        {
            alarm(5);
            const auto before = working_bytes();
            auto registration = dispatch_registration();
            if (simpler_aicpu_l1_hbg_register_callable(&registration) != 0) _exit(2);
            registration.runtime_address = 1;
            hbg::GraphHostArgs args;
            if (hbg::make_graph_dispatch_args(snapshot, registration, args) != 0) _exit(3);
            const int rc = consume_kernel_task(args.storage.data());
            _exit(rc != 0 && working_bytes() == before ? 0 : 1);
        },
        ::testing::ExitedWithCode(0), ""
    );
}

TEST_F(HbgGraphSlotTest, DispatchPublicationFailureCancelsAndReleasesAllParticipants) {
    prepare_slot();
    auto isolated = [&]() {
        alarm(5);
        const auto before = working_bytes();
        Runtime runtime;
        runtime.set_aicpu_thread_num(4);
        runtime.set_aicpu_allowed_cpu_count(4);
        runtime.set_aicpu_launch_count(4);
        auto registration = dispatch_registration();
        registration.runtime_address = reinterpret_cast<uintptr_t>(&runtime);
        if (simpler_aicpu_l1_hbg_register_callable(&registration) != 0) _exit(2);
        set_platform_regs(registration.register_table_address + 8);
        hbg::GraphHostArgs args;
        if (hbg::make_graph_dispatch_args(snapshot, registration, args) != 0) _exit(3);
        const uint64_t payload = reinterpret_cast<uintptr_t>(args.storage.data()) + args.data_offset;
        std::memcpy(
            reinterpret_cast<std::byte *>(args.storage.data()) + args.address_offset, &payload, sizeof(payload)
        );
        const auto *inner = reinterpret_cast<const std::byte *>(args.storage.data()) + sizeof(hbg::GraphDispatchHeader);
        if (hbg::validate_graph_packet(inner, snapshot.size(), hbg::GraphPacketAddress::DeviceCopy) !=
            hbg::GraphPacketStatus::Ok)
            _exit(4);
        for (int round = 0; round < 2; ++round) {
            runtime.kernel_prelaunch.state = HBG_KERNEL_PRELAUNCH_WAIT;
            std::array<int, 4> results{};
            std::vector<std::thread> threads;
            for (size_t thread = 0; thread < results.size(); ++thread)
                threads.emplace_back([&, thread] {
                    results[thread] = consume_kernel_task(args.storage.data());
                });
            for (auto &thread : threads)
                thread.join();
            if (std::any_of(
                    results.begin(), results.end(),
                    [](int rc) {
                        return rc == 0;
                    }
                ) ||
                runtime.kernel_prelaunch.state != HBG_KERNEL_PRELAUNCH_CANCEL || registry->restore.attempt != 0 ||
                working_bytes() != before)
                _exit(1);
        }
        _exit(0);
    };
    EXPECT_EXIT(isolated(), ::testing::ExitedWithCode(0), "");
}

TEST_F(HbgGraphSlotTest, KernelDfxRefreshTracksWindowsAndReplacesProgramGlobals) {
    prepare_slot();
    Runtime runtime;
    auto registration = dispatch_registration();
    registration.runtime_address = reinterpret_cast<uintptr_t>(&runtime);
    KernelArgs args{};
    args.runtime_args = &runtime;
    args.regs = registration.register_table_address;
    args.chip_swimlane_data_base = 0x10000;
    args.chip_swimlane_aicore_rotation_table = 0x20000;
    args.dep_gen_data_base = 0x30000;
    *runtime.kernel_dfx_args_slot() = reinterpret_cast<uintptr_t>(&args);
    const uint32_t swimlane = SIMPLER_DFX_FLAG_CHIP_SWIMLANE;
    const uint32_t dependencies = SIMPLER_DFX_FLAG_DEP_GEN;
    for (uint32_t flags : {0u, swimlane, 0u, dependencies, swimlane | dependencies, 0u, swimlane}) {
        SCOPED_TRACE(flags);
        set_chip_swimlane_enabled(false);
        set_platform_chip_swimlane_base(0x40000);
        set_platform_chip_swimlane_aicore_rotation_table(0x50000);
        set_dep_gen_enabled(true);
        set_platform_dep_gen_base(0x60000);
        set_dump_args_enabled(true);
        set_platform_dump_base(0x61000);
        set_scope_stats_enabled(true);
        set_pmu_enabled(true);
        set_platform_pmu_base(0x70000);
        set_platform_phase_base(0x80000);
        args.enable_profiling_flag = flags;
        ASSERT_TRUE(hbg::refresh_graph_kernel_dfx(registry, &runtime, registration));
        EXPECT_EQ(is_chip_swimlane_enabled(), (flags & swimlane) != 0);
        EXPECT_EQ(get_platform_chip_swimlane_base(), args.chip_swimlane_data_base);
        EXPECT_EQ(get_platform_chip_swimlane_aicore_rotation_table(), args.chip_swimlane_aicore_rotation_table);
        EXPECT_EQ(is_dep_gen_enabled(), (flags & dependencies) != 0);
        EXPECT_EQ(get_platform_dep_gen_base(), args.dep_gen_data_base);
        EXPECT_FALSE(is_dump_args_enabled());
        EXPECT_FALSE(is_pmu_enabled());
        EXPECT_FALSE(is_scope_stats_enabled());
        EXPECT_EQ(get_platform_dump_base(), 0u);
        EXPECT_EQ(get_platform_pmu_base(), 0u);
        EXPECT_EQ(get_test_kernel_phase_base(), 0u);
    }
    auto conflict = registration;
    ++conflict.context_generation;
    EXPECT_FALSE(hbg::refresh_graph_kernel_dfx(registry, &runtime, conflict));
    conflict = registration;
    conflict.register_table_address += 8;
    EXPECT_FALSE(hbg::refresh_graph_kernel_dfx(registry, &runtime, conflict));
    *runtime.kernel_dfx_args_slot() = 1;
    EXPECT_FALSE(hbg::refresh_graph_kernel_dfx(registry, &runtime, registration));
    *runtime.kernel_dfx_args_slot() = reinterpret_cast<uintptr_t>(&args);
    args.enable_profiling_flag = SIMPLER_DFX_FLAG_PMU;
    EXPECT_FALSE(hbg::refresh_graph_kernel_dfx(registry, &runtime, registration));
    EXPECT_TRUE(is_chip_swimlane_enabled());
    EXPECT_EQ(get_platform_chip_swimlane_base(), 0x10000u);
    alignas(1024) hbg::GraphSlotRegistry other{};
    EXPECT_FALSE(hbg::detach_graph_slot_registry(&other));
    EXPECT_TRUE(is_chip_swimlane_enabled());
    ASSERT_TRUE(hbg::detach_graph_slot_registry(registry));
    EXPECT_FALSE(is_chip_swimlane_enabled());
    EXPECT_FALSE(is_dep_gen_enabled());
    EXPECT_EQ(get_platform_chip_swimlane_base(), 0u);
    EXPECT_EQ(get_platform_chip_swimlane_aicore_rotation_table(), 0u);
    EXPECT_EQ(get_platform_dep_gen_base(), 0u);
    EXPECT_FALSE(hbg::refresh_graph_kernel_dfx(registry, &runtime, registration));
    ASSERT_EQ(hbg::bind_graph_slot_registry(registry, 0, 109), hbg::GraphSlotStatus::Ok);
    args.chip_swimlane_data_base = 0x90000;
    args.enable_profiling_flag = swimlane;
    ASSERT_TRUE(hbg::refresh_graph_kernel_dfx(registry, &runtime, registration));
    EXPECT_EQ(get_platform_chip_swimlane_base(), 0x90000u);
}

TEST_F(HbgGraphSlotTest, KernelDfxRejectsInvalidResidentArgsBeforeBinding) {
    prepare_slot();
    Runtime runtime;
    auto registration = dispatch_registration();
    registration.runtime_address = reinterpret_cast<uintptr_t>(&runtime);
    KernelArgs args{};
    args.runtime_args = &runtime;
    args.regs = registration.register_table_address;
    *runtime.kernel_dfx_args_slot() = reinterpret_cast<uintptr_t>(&args);
    const auto valid = args;
    for (int invalid = 0; invalid < 8; ++invalid) {
        SCOPED_TRACE(invalid);
        args = valid;
        switch (invalid) {
        case 0:
            args.runtime_args = nullptr;
            break;
        case 1:
            args.regs += 8;
            break;
        case 2:
            args.chip_swimlane_data_base = 0x10000;
            break;
        case 3:
            args.chip_swimlane_aicore_rotation_table = 0x20000;
            break;
        case 4:
            args.dep_gen_data_base = 1;
            break;
        case 5:
            args.enable_profiling_flag = SIMPLER_DFX_FLAG_DEP_GEN;
            break;
        case 6:
            args.dump_data_base = 0x30000;
            break;
        case 7:
            args.device_wall_data_base = 0x40000;
            break;
        }
        EXPECT_FALSE(hbg::refresh_graph_kernel_dfx(registry, &runtime, registration));
    }
    args = valid;
    ASSERT_TRUE(hbg::refresh_graph_kernel_dfx(registry, &runtime, registration));
    EXPECT_FALSE(is_chip_swimlane_enabled());
    EXPECT_FALSE(is_dep_gen_enabled());
    EXPECT_EQ(get_platform_chip_swimlane_base(), 0u);
    EXPECT_EQ(get_platform_dep_gen_base(), 0u);
}

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
    EXPECT_EQ(view.slot.max_packet_bytes, packet.bytes + packet_header().runtime_offset);
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
    constexpr std::array<const char *, 6> kernel_symbols = {
        "simpler_aicpu_l1_hbg_register_execution_slot",   "simpler_aicpu_l1_hbg_prepare_image_destinations",
        "simpler_aicpu_l1_hbg_prepare_execution_context", "simpler_aicpu_l1_hbg_detach_execution_slot",
        "simpler_aicpu_l1_hbg_register_callable",         "simpler_aicpu_kernel_exec",
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
    std::vector<std::byte> target_bytes() const {
        auto all = working_bytes();
        all.resize(all.size() - sizeof(hbg::GraphSlotRegistry));
        return all;
    }
    std::vector<std::byte> restored_image_bytes() {
        auto all = target_bytes();
        all.erase(all.begin(), all.begin() + seal.destinations[0].capacity + packet_header().runtime_offset);
        return all;
    }
    void expect_initialized_scheduler() {
        const auto base = seal.destinations[1].address;
        const auto &layout = restored.runtime->prebuilt_layout;
        auto &scheduler = *restored.runtime->scheduler;
        EXPECT_EQ(reinterpret_cast<uintptr_t>(&scheduler), base + layout.off_scheduler);
        EXPECT_EQ(reinterpret_cast<uintptr_t>(restored.runtime->sm_handle), base + layout.off_sm_handle);
        EXPECT_EQ(reinterpret_cast<uintptr_t>(restored.runtime->aicore_mailbox), base + layout.off_mailbox);
        EXPECT_EQ(scheduler.sm_header, restored.runtime->sm_handle->header);
        const auto &handle = *restored.runtime->sm_handle;
        const auto sm = base + packet_header().sm_offset;
        EXPECT_EQ(reinterpret_cast<uintptr_t>(handle.sm_base), sm);
        EXPECT_EQ(handle.sm_size, restored.sm_bytes);
        EXPECT_EQ(reinterpret_cast<uintptr_t>(handle.header), sm);
        EXPECT_FALSE(handle.is_owner);
        const auto segments = sm_layout::segment_offsets(
            sm_layout::image_extents({sm_layout::live_slot_pitch(restored.total_tasks), 0, 0, 0})
        );
        EXPECT_EQ(reinterpret_cast<uintptr_t>(handle.header->tasks.task_storage), sm + segments.storage);
        EXPECT_EQ(reinterpret_cast<uintptr_t>(handle.header->tasks.task_states), sm + segments.task_states);
        const auto check_queue = [&](const ChipReadyQueue &queue, uint64_t capacity) {
            EXPECT_EQ(queue.capacity, capacity);
            EXPECT_EQ(queue.mask, capacity - 1);
            EXPECT_EQ(queue.enqueue_pos.load(), 0u);
            EXPECT_EQ(queue.dequeue_pos.load(), 0u);
            EXPECT_EQ(queue.max_occupancy.load(), 0u);
            ASSERT_NE(queue.slots, nullptr);
            ASSERT_GE(reinterpret_cast<uintptr_t>(queue.slots), base);
            ASSERT_LE(reinterpret_cast<uintptr_t>(queue.slots + capacity), base + packet_header().runtime_offset);
            for (uint64_t i = 0; i < capacity; ++i) {
                EXPECT_EQ(queue.slots[i].sequence.load(), static_cast<int64_t>(i));
                EXPECT_EQ(queue.slots[i].slot_state, nullptr);
            }
        };
        for (int i = 0; i < NUM_RESOURCE_SHAPES; ++i) {
            check_queue(scheduler.ready_queues[i], layout.sched.capacities.ready[i]);
            check_queue(scheduler.ready_sync_queues[i], layout.sched.capacities.ready_sync[i]);
            check_queue(scheduler.early_dispatch_queues[i], CHIP_EARLY_DISPATCH_QUEUE_SIZE);
        }
        check_queue(scheduler.dummy_ready_queue, layout.sched.capacities.dummy);
        check_queue(scheduler.graph_ready_queue, layout.sched.capacities.graph_ready);
        check_queue(scheduler.graph_prepare_queue, layout.sched.capacities.graph_prepare);
        check_queue(scheduler.early_sync_start_queue, CHIP_EARLY_DISPATCH_QUEUE_SIZE);
        check_queue(scheduler.ed_publish_drain_queue, CHIP_EARLY_DISPATCH_QUEUE_SIZE);
        EXPECT_EQ(scheduler.async_wait_list.count, 0);
        EXPECT_EQ(scheduler.async_wait_list.busy.load(), 0);
        EXPECT_EQ(scheduler.async_wait_list.mpsc_skipped_count.load(), 0u);
    }
    void expect_initialized_prefix() {
        ASSERT_NO_FATAL_FAILURE(expect_initialized_scheduler());
        const auto &mailbox = *restored.runtime->aicore_mailbox;
        EXPECT_EQ(mailbox.head.load(), 0u);
        EXPECT_EQ(mailbox.tail.load(), 0u);
        for (const auto &entry : mailbox.entries)
            EXPECT_EQ(entry.seq.load(), 0u);
    }
};

namespace {
struct PreparedImageReleaseProbe {
    int calls{0};
    int error{0};

    std::shared_ptr<hbg::KernelPreparedGraphImage> make_image() {
        auto image = std::make_shared<hbg::KernelPreparedGraphImage>();
        image->pinned_image = this;
        image->release_image = [](void *source) noexcept {
            auto &probe = *static_cast<PreparedImageReleaseProbe *>(source);
            ++probe.calls;
            return probe.error;
        };
        return image;
    }
};
}  // namespace

TEST(HbgPreparedImageLifetime, FailedReleaseRetainsTheSourceUntilSuccessfulRetry) {
    PreparedImageReleaseProbe probe;
    probe.error = -71;
    auto image = probe.make_image();
    EXPECT_EQ(image->release(), -71);
    EXPECT_EQ(image->pinned_image, &probe);
    EXPECT_EQ(probe.calls, 1);
    probe.error = 0;
    EXPECT_EQ(image->release(), 0);
    EXPECT_EQ(image->pinned_image, nullptr);
    EXPECT_EQ(image->release(), 0);
    image.reset();
    EXPECT_EQ(probe.calls, 2);
}

TEST(HbgPreparedImageLifetime, ContextCloseRetriesTheFailedSourceWithoutReleasingSuccessfulSourcesTwice) {
    PreparedImageReleaseProbe first, second, third;
    second.error = -72;
    hbg::KernelContextLaunchState context;
    context.prepared_images = {first.make_image(), second.make_image(), third.make_image()};
    const std::weak_ptr<hbg::KernelPreparedGraphImage> retained = context.prepared_images[1];
    EXPECT_EQ(context.release_prepared_images(), -72);
    ASSERT_EQ(context.prepared_images.size(), 3u);
    EXPECT_FALSE(retained.expired());
    EXPECT_EQ(context.prepared_images[0]->pinned_image, nullptr);
    EXPECT_EQ(context.prepared_images[1]->pinned_image, &second);
    EXPECT_EQ(first.calls, 1);
    EXPECT_EQ(second.calls, 1);
    EXPECT_EQ(third.calls, 0);
    second.error = 0;
    EXPECT_EQ(context.release_prepared_images(), 0);
    EXPECT_TRUE(context.prepared_images.empty());
    EXPECT_TRUE(retained.expired());
    EXPECT_EQ(context.release_prepared_images(), 0);
    EXPECT_EQ(first.calls, 1);
    EXPECT_EQ(second.calls, 2);
    EXPECT_EQ(third.calls, 1);
}

TEST(HbgPreparedImageLifetime, AbandonPreservesSourcesThroughContextAndCallableDestruction) {
    PreparedImageReleaseProbe first, second;
    std::weak_ptr<hbg::KernelPreparedGraphImage> retained_first, retained_second;
    {
        hbg::KernelContextLaunchState context;
        hbg::KernelCallableLaunchState callable;
        context.prepared_images = {first.make_image(), second.make_image()};
        callable.prepared_image = context.prepared_images[0];
        retained_first = callable.prepared_image;
        retained_second = context.prepared_images[1];
        context.abandon_prepared_images();
        context.prepared_images.clear();
        EXPECT_FALSE(retained_first.expired());
        EXPECT_TRUE(retained_second.expired());
        EXPECT_EQ(first.calls, 0);
        EXPECT_EQ(second.calls, 0);
    }
    EXPECT_TRUE(retained_first.expired());
    EXPECT_EQ(first.calls, 0);
    EXPECT_EQ(second.calls, 0);
}

class HbgPreparedGraphTest : public HbgGraphRestoreTest {
protected:
    hbg::GraphPreparedDispatch prepared{};

    void dirty_working_preserving_mailbox() {
        auto &mailbox = *restored.runtime->aicore_mailbox;
        std::array<std::byte, sizeof(mailbox)> saved;
        std::memcpy(saved.data(), &mailbox, saved.size());
        dirty_working();
        std::memcpy(&mailbox, saved.data(), saved.size());
    }
    void prepare_dispatch() {
        ASSERT_EQ(hbg::make_graph_prepared_dispatch(snapshot, dispatch_registration(), prepared), 0);
        ASSERT_TRUE(hbg::graph_prepared_dispatch_matches_slot(prepared, seal));
    }
    void publish_image() {
        const auto *payload = static_cast<const std::byte *>(snapshot.data()) + sizeof(SimplerKernelInvocationHeader) +
                              prepared.graph.payload_offset;
        for (uint32_t i = 0; i < prepared.graph.region_count; ++i) {
            const auto &region = prepared.regions[i];
            const auto kind = static_cast<uint32_t>(region.kind);
            std::memcpy(
                reinterpret_cast<void *>(seal.destinations[kind].address + region.destination_offset),
                payload + region.source_offset, region.bytes
            );
        }
    }
    void select_template(const hbg::GraphLaunchTemplate &source) {
        snapshot = source;
        ASSERT_EQ(hbg::make_graph_host_args(snapshot, packet), 0);
        patch();
        ASSERT_NO_FATAL_FAILURE(prepare_dispatch());
    }
    void prepare_templates(
        void (*first)(const ChipTaskArgs &), void (*second)(const ChipTaskArgs &),
        std::array<hbg::GraphLaunchTemplate, 2> &templates
    ) {
        const std::array<void (*)(const ChipTaskArgs &), 2> entries{first, second};
        std::array<hbg::GraphResourceRequirements, 2> requirements{};
        RuntimeArenaLayout physical{};
        ASSERT_EQ(hbg::make_kernel_graph_layout(capacity, physical), 0);
        for (size_t i = 0; i < entries.size(); ++i) {
            ASSERT_GE(build(entries[i]), 0);
            ASSERT_EQ(hbg::get_graph_resource_requirements(result, physical, requirements[i]), 0);
        }
        hbg::KernelResourcePlan plan;
        ASSERT_EQ(hbg::KernelResourcePlan::create(requirements.data(), requirements.size(), plan), 0);
        for (const auto &graph : requirements)
            ASSERT_TRUE(plan.admits(graph));
        ASSERT_EQ(context.initialize(0, provider.context_ops(), 19), 0);
        ASSERT_EQ(plan.prepare(context, provider.resource_ops()), 0);
        ASSERT_EQ(context.mark_ready_enqueued(), 0);
        ASSERT_EQ(context.freeze_resources(), 0);
        ASSERT_EQ(hbg::bind_kernel_resources_for_launch(context, 0, 19, requirements[0], binding), 0);
        ASSERT_EQ(hbg::seal_graph_execution_slot(context, 0, 19, 109, seal), 0);
        registry = reinterpret_cast<hbg::GraphSlotRegistry *>(seal.registry.address);
        ASSERT_EQ(simpler_aicpu_l1_hbg_register_execution_slot(&seal), 0);
        const uint64_t first_hash = identity.argument_hash;
        for (size_t i = 0; i < entries.size(); ++i) {
            identity.argument_hash = first_hash + i;
            ASSERT_GE(build(entries[i]), 0);
            ASSERT_EQ(snapshot_graph(), 0);
            templates[i] = snapshot;
        }
        identity.argument_hash = first_hash;
        ASSERT_NO_FATAL_FAILURE(select_template(templates[0]));
        dirty_working();
    }
};

TEST_F(HbgPreparedGraphTest, HostValidationRejectsCorruptSourceAndSemanticImageBeforePublication) {
    ASSERT_NO_FATAL_FAILURE(prepare_slot());
    ASSERT_NO_FATAL_FAILURE(prepare_dispatch());
    const auto before = working_bytes();
    const auto expected = prepared;
    for (int variant = 0; variant < 4; ++variant) {
        SCOPED_TRACE(variant);
        auto damaged = snapshot;
        auto *base = const_cast<std::byte *>(static_cast<const std::byte *>(damaged.data()));
        hbg::GraphPacketHeader graph{};
        std::memcpy(&graph, base + sizeof(SimplerKernelInvocationHeader), sizeof(graph));
        if (variant == 0) {
            base[damaged.size() - 1] ^= std::byte{0x80};
        } else {
            const auto *source_payload = base + sizeof(SimplerKernelInvocationHeader) + graph.payload_offset;
            hbg::GraphImageRegion region{};
            std::memcpy(&region, base + sizeof(SimplerKernelInvocationHeader) + sizeof(graph), sizeof(region));
            auto *runtime_source = const_cast<std::byte *>(source_payload) + region.source_offset;
            if (variant == 1) {
                const uint64_t foreign = 0x1234;
                std::memcpy(runtime_source + offsetof(RuntimeContext, scheduler), &foreign, sizeof(foreign));
            } else if (variant == 2) {
                const uint64_t invalid_offset = UINT64_MAX;
                std::memcpy(
                    runtime_source + offsetof(RuntimeContext, prebuilt_layout) +
                        offsetof(RuntimeArenaLayout, off_scheduler),
                    &invalid_offset, sizeof(invalid_offset)
                );
            } else {
                const auto offsets = sm_layout::segment_offsets(sm_layout::image_extents({graph.total_tasks, 0, 0, 0}));
                auto *sm = runtime_source + graph.sm_offset - graph.runtime_offset;
                const auto invalid_kind = static_cast<TaskKind>(UINT8_MAX);
                std::memcpy(
                    sm + offsets.storage + offsetof(ChipTaskStorage, slot) + offsetof(ChipTaskSlotState, task_kind),
                    &invalid_kind, sizeof(invalid_kind)
                );
            }
            graph.checksum = hbg::graph_packet_checksum(damaged.data(), damaged.size());
            std::memcpy(base + sizeof(SimplerKernelInvocationHeader), &graph, sizeof(graph));
            ASSERT_EQ(
                hbg::validate_graph_packet(damaged.data(), damaged.size(), hbg::GraphPacketAddress::HostTemplate),
                hbg::GraphPacketStatus::Ok
            );
        }
        EXPECT_NE(hbg::make_graph_prepared_dispatch(damaged, dispatch_registration(), prepared), 0);
        EXPECT_EQ(std::memcmp(&prepared, &expected, sizeof(prepared)), 0);
        EXPECT_EQ(working_bytes(), before);
    }
}

TEST_F(HbgPreparedGraphTest, MetadataRejectsChangedIdentityBoundsAndUnusedRegions) {
    std::array<hbg::GraphLaunchTemplate, 2> templates;
    ASSERT_NO_FATAL_FAILURE(prepare_templates(graph_entry, empty_entry, templates));
    using Mutate = void (*)(hbg::GraphPreparedDispatch &);
    const Mutate mutations[] = {
        [](auto &h) {
            ++h.magic;
        },
        [](auto &h) {
            ++h.version;
        },
        [](auto &h) {
            --h.header_bytes;
        },
        [](auto &h) {
            ++h.invocation.callable_id;
        },
        [](auto &h) {
            ++h.invocation.tensor_count;
        },
        [](auto &h) {
            ++h.invocation.scalar_count;
        },
        [](auto &h) {
            h.invocation.host_copy_tensor_count = 1;
        },
        [](auto &h) {
            ++h.registration.context_generation;
        },
        [](auto &h) {
            ++h.registration.callable_hash;
        },
        [](auto &h) {
            ++h.graph.function_hash;
        },
        [](auto &h) {
            ++h.graph.runtime_binary_id;
        },
        [](auto &h) {
            h.graph.total_tasks = h.graph.task_window;
        },
        [](auto &h) {
            h.graph.heap_bytes = h.graph.destinations[0].capacity + 1;
        },
        [](auto &h) {
            h.graph.sm_offset = h.graph.destinations[1].capacity;
        },
        [](auto &h) {
            h.graph.destinations[1].address = h.graph.destinations[0].address;
        },
        [](auto &h) {
            h.graph.destinations[0].capacity = UINT64_MAX;
        },
        [](auto &h) {
            h.regions[0].destination_offset = h.graph.sm_offset;
        },
        [](auto &h) {
            h.regions[0].bytes = h.graph.destinations[1].capacity;
        },
        [](auto &h) {
            h.regions[0].source_offset = UINT64_MAX;
        },
        [](auto &h) {
            h.regions[0].reserved = 1;
        },
    };
    const auto before = working_bytes();
    for (auto mutate : mutations) {
        auto damaged = prepared;
        mutate(damaged);
        damaged.checksum = hbg::graph_prepared_dispatch_checksum(damaged);
        EXPECT_FALSE(hbg::valid_graph_prepared_dispatch(damaged));
        EXPECT_FALSE(hbg::graph_prepared_dispatch_matches_slot(damaged, seal));
    }
    auto damaged = prepared;
    ++damaged.checksum;
    EXPECT_FALSE(hbg::valid_graph_prepared_dispatch(damaged));
    ASSERT_NO_FATAL_FAILURE(select_template(templates[1]));
    ASSERT_EQ(prepared.graph.total_tasks, 0u);
    ASSERT_LT(prepared.graph.region_count, 3u);
    damaged = prepared;
    damaged.regions[prepared.graph.region_count].bytes = 64;
    damaged.checksum = hbg::graph_prepared_dispatch_checksum(damaged);
    EXPECT_FALSE(hbg::valid_graph_prepared_dispatch(damaged));
    EXPECT_EQ(working_bytes(), before);
}

TEST_F(HbgPreparedGraphTest, MetadataRequiresTheIndependentSlotAndSupportsUnalignedReads) {
    ASSERT_NO_FATAL_FAILURE(prepare_slot());
    ASSERT_NO_FATAL_FAILURE(prepare_dispatch());
    for (int variant = 0; variant < 7; ++variant) {
        SCOPED_TRACE(variant);
        auto foreign = seal;
        if (variant == 0) ++foreign.device_id;
        else if (variant == 1) ++foreign.slot_generation;
        else if (variant == 2) ++foreign.runtime_binary_id;
        else if (variant == 3) foreign.destinations[1].address += 1024;
        else if (variant == 4) --foreign.destinations[1].capacity;
        else if (variant == 5) foreign.max_packet_bytes = sizeof(SimplerKernelInvocationHeader);
        else foreign.destinations[2].capacity += 1024;
        EXPECT_FALSE(hbg::graph_prepared_dispatch_matches_slot(prepared, foreign));
    }
    for (size_t alignment = 0; alignment < sizeof(uint64_t); ++alignment) {
        std::vector<std::byte> storage(sizeof(prepared) + alignment);
        std::memcpy(storage.data() + alignment, &prepared, sizeof(prepared));
        hbg::GraphPreparedDispatch output{};
        output.checksum = 999;
        const auto unchanged = output;
        EXPECT_FALSE(hbg::read_graph_prepared_dispatch(storage.data() + alignment, sizeof(prepared) - 1, output));
        EXPECT_EQ(std::memcmp(&output, &unchanged, sizeof(output)), 0);
        ASSERT_TRUE(hbg::read_graph_prepared_dispatch(storage.data() + alignment, sizeof(prepared), output));
        EXPECT_EQ(std::memcmp(&output, &prepared, sizeof(output)), 0);
    }
    hbg::GraphPreparedDispatch output{};
    EXPECT_FALSE(hbg::read_graph_prepared_dispatch(nullptr, sizeof(prepared), output));
}

TEST_F(HbgPreparedGraphTest, SingleParticipantMetadataFailureCollectivelyCancelsBeforeAttach) {
    ASSERT_NO_FATAL_FAILURE(prepare_slot());
    for (int variant = 0; variant < 3; ++variant) {
        SCOPED_TRACE(variant);
        for (size_t bad_copy = 0; bad_copy < 4; ++bad_copy) {
            SCOPED_TRACE(bad_copy);
            auto isolated = [&]() {
                alarm(5);
                Runtime execution_runtime;
                execution_runtime.set_aicpu_thread_num(4);
                execution_runtime.set_aicpu_allowed_cpu_count(4);
                execution_runtime.set_aicpu_launch_count(4);
                for (int32_t i = 0; i < 4; ++i)
                    execution_runtime.get_aicpu_allowed_cpus()[i] = i;
                auto callable = make_callable<CoreCallable, CHIP_MAX_TENSOR_ARGS, 1024>(
                    nullptr, 0, "prepared", nullptr, 0, nullptr, nullptr, 0, ""
                );
                auto registration = dispatch_registration();
                registration.runtime_address = reinterpret_cast<uintptr_t>(&execution_runtime);
                registration.callable_address = reinterpret_cast<uintptr_t>(callable.data());
                registration.callable_bytes = callable.size();
                KernelArgs dfx_args{};
                dfx_args.runtime_args = &execution_runtime;
                dfx_args.regs = registration.register_table_address;
                *execution_runtime.kernel_dfx_args_slot() = reinterpret_cast<uintptr_t>(&dfx_args);
                if (simpler_aicpu_l1_hbg_register_callable(&registration) != 0 ||
                    !hbg::refresh_graph_kernel_dfx(registry, &execution_runtime, registration))
                    _exit(2);
                hbg::GraphPreparedContext prepared_context{seal, reinterpret_cast<uintptr_t>(&execution_runtime)};
                if (simpler_aicpu_l1_hbg_prepare_execution_context(&prepared_context) != 0 ||
                    hbg::make_graph_prepared_dispatch(snapshot, registration, prepared) != 0 ||
                    !hbg::graph_prepared_dispatch_matches_slot(prepared, seal))
                    _exit(3);
                publish_image();
                const auto before = target_bytes();
                std::array<hbg::GraphPreparedDispatch, 4> copies;
                copies.fill(prepared);
                if (variant == 0) ++copies[bad_copy].magic;
                else if (variant == 2) ++copies[bad_copy].graph.argument_hash;
                copies[bad_copy].checksum = hbg::graph_prepared_dispatch_checksum(copies[bad_copy]);
                if (variant == 2 && !hbg::graph_prepared_dispatch_matches_slot(copies[bad_copy], seal)) _exit(4);
                execution_runtime.kernel_prelaunch.state = HBG_KERNEL_PRELAUNCH_WAIT;
                std::array<int, 4> results{};
                std::vector<std::thread> threads;
                for (size_t participant = 0; participant < copies.size(); ++participant)
                    threads.emplace_back([&, participant] {
                        void *arg = variant == 1 && participant == bad_copy ? nullptr : &copies[participant];
                        results[participant] = consume_kernel_task(arg);
                    });
                for (auto &thread : threads)
                    thread.join();
                if (std::any_of(
                        results.begin(), results.end(),
                        [](int rc) {
                            return rc != -1;
                        }
                    ) ||
                    execution_runtime.kernel_prelaunch.state != HBG_KERNEL_PRELAUNCH_CANCEL ||
                    registry->restore.attempt != 0 || registry->restore.committed_generation != 0 ||
                    registry->restore.phase != static_cast<uint32_t>(hbg::GraphRestorePhase::Idle) ||
                    registry->phase != static_cast<uint32_t>(hbg::GraphSlotPhase::Poisoned) || target_bytes() != before)
                    _exit(1);
                _exit(0);
            };
            EXPECT_EXIT(isolated(), ::testing::ExitedWithCode(0), "");
        }
    }
}

TEST_F(HbgPreparedGraphTest, OrderedDmaResetsExecutionStateAndRetainsQuiescentMailboxStorage) {
    ASSERT_NO_FATAL_FAILURE(prepare_slot());
    ASSERT_NO_FATAL_FAILURE(prepare_dispatch());
    std::vector<std::byte> expected;
    for (uint64_t round = 1; round <= 4; ++round) {
        if (round == 1) dirty_working();
        else dirty_working_preserving_mailbox();
        publish_image();
        ASSERT_EQ(hbg::attach_prepared_graph_image(prepared, seal, restored), hbg::GraphRestoreStatus::Ok);
        EXPECT_EQ(restored.generation, round);
        EXPECT_EQ(restored.total_tasks, prepared.graph.total_tasks);
        ASSERT_NO_FATAL_FAILURE(expect_initialized_scheduler());
        auto &mailbox = *restored.runtime->aicore_mailbox;
        EXPECT_EQ(mailbox.head.load(), round - 1);
        EXPECT_EQ(mailbox.tail.load(), round - 1);
        AICoreCompletionMsgView message{};
        EXPECT_FALSE(mailbox.try_pop(message));
        if (round == 1) {
            for (const auto &entry : mailbox.entries)
                EXPECT_EQ(entry.seq.load(), 0u);
        }
        if (round == 1) expected = restored_image_bytes();
        else EXPECT_EQ(restored_image_bytes(), expected);
        auto &tasks = restored.runtime->sm_handle->header->tasks;
        ASSERT_GT(tasks.total_tasks, 0);
        auto &queue = restored.runtime->scheduler->graph_ready_queue;
        ASSERT_TRUE(queue.push(&tasks.storage_at(0).slot));
        EXPECT_EQ(queue.pop(), &tasks.storage_at(0).slot);
        ASSERT_TRUE(mailbox.try_push_normal_done(TaskId{1}, 0x4000 + round));
        ASSERT_NO_FATAL_FAILURE(retire());
    }
}

TEST_F(HbgPreparedGraphTest, AlternatingTemplatesRestoreTheirOwnImagesIntoTheSameSlot) {
    std::array<hbg::GraphLaunchTemplate, 2> templates;
    ASSERT_NO_FATAL_FAILURE(prepare_templates(graph_entry, chain_entry, templates));
    ASSERT_EQ(result.total_tasks, 2);
    std::vector<std::byte> expected_first;
    for (int round = 0; round < 3; ++round) {
        SCOPED_TRACE(round);
        ASSERT_NO_FATAL_FAILURE(select_template(templates[round == 1 ? 1 : 0]));
        if (round == 0) dirty_working();
        else dirty_working_preserving_mailbox();
        publish_image();
        ASSERT_EQ(hbg::attach_prepared_graph_image(prepared, seal, restored), hbg::GraphRestoreStatus::Ok);
        EXPECT_EQ(restored.generation, static_cast<uint64_t>(round + 1));
        EXPECT_EQ(restored.total_tasks, round == 1 ? 2u : 1u);
        ASSERT_NO_FATAL_FAILURE(expect_initialized_prefix());
        if (round == 0) expected_first = restored_image_bytes();
        else if (round == 2) EXPECT_EQ(restored_image_bytes(), expected_first);
        auto &task = restored.runtime->sm_handle->header->tasks.storage_at(0);
        task.slot.completed_subtasks.store(7);
        task.payload.published_block_count.store(5);
        ASSERT_NO_FATAL_FAILURE(retire());
    }
}

TEST_F(HbgPreparedGraphTest, MailboxReuseSurvivesWrapAndDiscardsOnlyPreviouslyPublishedMessages) {
    ASSERT_NO_FATAL_FAILURE(prepare_slot());
    ASSERT_NO_FATAL_FAILURE(prepare_dispatch());
    publish_image();
    ASSERT_EQ(hbg::attach_prepared_graph_image(prepared, seal, restored), hbg::GraphRestoreStatus::Ok);
    auto &mailbox = *restored.runtime->aicore_mailbox;
    AICoreCompletionMsgView message{};
    for (uint64_t i = 0; i < AICORE_COMPLETION_MAILBOX_CAPACITY + 17; ++i) {
        ASSERT_TRUE(mailbox.try_push_normal_done(TaskId{1}, 0x4000 + i));
        ASSERT_TRUE(mailbox.try_pop(message));
        EXPECT_EQ(message.addr, 0x4000 + i);
    }
    for (uint64_t i = 0; i < 5; ++i)
        ASSERT_TRUE(mailbox.try_push_normal_done(TaskId{1}, 0x8000 + i));
    const uint64_t ticket = mailbox.head.load();
    ASSERT_NO_FATAL_FAILURE(retire());
    publish_image();
    ASSERT_EQ(hbg::attach_prepared_graph_image(prepared, seal, restored), hbg::GraphRestoreStatus::Ok);
    EXPECT_EQ(restored.runtime->aicore_mailbox, &mailbox);
    EXPECT_EQ(mailbox.head.load(), ticket);
    EXPECT_EQ(mailbox.tail.load(), ticket);
    EXPECT_FALSE(mailbox.try_pop(message));
    mailbox.head.store(ticket + 1);
    EXPECT_FALSE(mailbox.try_pop(message));
    mailbox.head.store(ticket);
    ASSERT_TRUE(mailbox.try_push_normal_done(TaskId{2}, 0x9000));
    ASSERT_TRUE(mailbox.try_pop(message));
    EXPECT_EQ(message.task_token.raw, 2u);
    EXPECT_EQ(message.addr, 0x9000u);
    ASSERT_NO_FATAL_FAILURE(retire());
}

TEST_F(HbgPreparedGraphTest, ControlledRestoreFailureRequiresFullMailboxInitialization) {
    ASSERT_NO_FATAL_FAILURE(prepare_slot());
    ASSERT_NO_FATAL_FAILURE(prepare_dispatch());
    publish_image();
    ASSERT_EQ(hbg::attach_prepared_graph_image(prepared, seal, restored), hbg::GraphRestoreStatus::Ok);
    ASSERT_TRUE(restored.runtime->aicore_mailbox->try_push_normal_done(TaskId{1}, 0x4000));
    ASSERT_NO_FATAL_FAILURE(retire());
    hbg::GraphRestoreOps failure;
    failure.copy = [](void *, void *, const void *, size_t) {
        return false;
    };
    ASSERT_EQ(restore(failure), hbg::GraphRestoreStatus::CopyFailed);
    ASSERT_NO_FATAL_FAILURE(retire(hbg::GraphRestoreRetirement::ControlledFailure));
    dirty_working();
    publish_image();
    ASSERT_EQ(hbg::attach_prepared_graph_image(prepared, seal, restored), hbg::GraphRestoreStatus::Ok);
    ASSERT_NO_FATAL_FAILURE(expect_initialized_prefix());
    ASSERT_NO_FATAL_FAILURE(retire());
}

TEST_F(HbgPreparedGraphTest, AttachRejectsForeignMetadataAndBusyStateWithoutChangingImages) {
    ASSERT_NO_FATAL_FAILURE(prepare_slot());
    ASSERT_NO_FATAL_FAILURE(prepare_dispatch());
    const auto before = working_bytes();
    auto foreign = prepared;
    ++foreign.graph.slot_generation;
    foreign.checksum = hbg::graph_prepared_dispatch_checksum(foreign);
    restored.generation = 999;
    EXPECT_NE(hbg::attach_prepared_graph_image(foreign, seal, restored), hbg::GraphRestoreStatus::Ok);
    EXPECT_EQ(restored.generation, 999u);
    EXPECT_EQ(working_bytes(), before);
    dirty_working();
    publish_image();
    ASSERT_EQ(hbg::attach_prepared_graph_image(prepared, seal, restored), hbg::GraphRestoreStatus::Ok);
    const auto active = working_bytes();
    EXPECT_EQ(hbg::attach_prepared_graph_image(prepared, seal, restored), hbg::GraphRestoreStatus::Busy);
    EXPECT_EQ(working_bytes(), active);
    ASSERT_NO_FATAL_FAILURE(retire());
}

TEST_F(HbgGraphRestoreTest, RejectsChecksumVersionsBeforeChangingRestoreState) {
    ASSERT_NO_FATAL_FAILURE(prepare_slot());
    const auto original = packet.storage;
    const auto before = working_bytes();
    for (int variant = 0; variant < 3; ++variant) {
        SCOPED_TRACE(variant);
        packet.storage = original;
        auto &header = packet_header();
        header.version = variant < 2 ? 4 : 5;
        if (variant == 1) {
            header.checksum = hbg::graph_packet_checksum(packet.storage.data(), packet.bytes);
        } else {
            constexpr size_t skip =
                sizeof(SimplerKernelInvocationHeader) + offsetof(hbg::GraphPacketHeader, inline_payload_addr);
            const auto *data = reinterpret_cast<const uint8_t *>(packet.storage.data());
            header.checksum = simpler::common::utils::fnv1a_64(data, skip);
            header.checksum =
                simpler::common::utils::fnv1a_64_append(header.checksum, data + skip + 16, packet.bytes - skip - 16);
        }
        expect_rejected(hbg::GraphSlotStatus::InvalidPacket);
        restored = {reinterpret_cast<RuntimeContext *>(uintptr_t{1}), 999, 888, 777};
        EXPECT_EQ(restore(), hbg::GraphRestoreStatus::Rejected);
        EXPECT_EQ(restored.runtime, reinterpret_cast<RuntimeContext *>(uintptr_t{1}));
        EXPECT_EQ(restored.generation, 999u);
        EXPECT_EQ(restored.sm_bytes, 888u);
        EXPECT_EQ(restored.total_tasks, 777u);
        EXPECT_EQ(working_bytes(), before);
    }
    packet.storage = original;
    EXPECT_EQ(packet_header().version, 5);
    ASSERT_EQ(restore(), hbg::GraphRestoreStatus::Ok);
    EXPECT_EQ(restored.generation, 1u);
    retire();
}

class HbgLargeGraphRestoreTest : public HbgGraphRestoreTest {
protected:
    void SetUp() override {
        capacity = 512;
        HostGraphBuildTest::SetUp();
    }
    void prepare_large_slot() {
        packet_task_count = capacity - 1;
        ASSERT_EQ(build(packet_window_entry), capacity - 1);
        RuntimeArenaLayout physical{};
        ASSERT_EQ(hbg::make_kernel_graph_layout(capacity, physical), 0);
        hbg::GraphResourceRequirements largest{};
        ASSERT_EQ(hbg::get_graph_resource_requirements(result, physical, largest), 0);
        ASSERT_NO_FATAL_FAILURE(prepare_slot(true, largest.scheduler_state_bytes, 2 * 1024 * 1024));
    }
};

TEST_F(HbgLargeGraphRestoreTest, FlatQueuesFollowTaskCountWithoutMovingFrozenStorage) {
    ASSERT_NO_FATAL_FAILURE(prepare_large_slot());
    RuntimeArenaLayout physical{};
    ASSERT_EQ(hbg::make_kernel_graph_layout(capacity, physical), 0);
    for (int count : {0, 1, 63, 64, 65, 127, 128, 129, 511}) {
        SCOPED_TRACE(count);
        packet_task_count = count;
        ASSERT_EQ(build(packet_window_entry), count);
        ASSERT_EQ(snapshot_graph(), 0);
        ASSERT_EQ(hbg::make_graph_host_args(snapshot, packet), 0);
        patch();
        dirty_working();
        ASSERT_EQ(restore(), hbg::GraphRestoreStatus::Ok);
        ASSERT_NO_FATAL_FAILURE(expect_initialized_prefix());
        const auto &logical = restored.runtime->prebuilt_layout;
        uint64_t expected = 64;
        while (expected < static_cast<uint64_t>(count))
            expected <<= 1;
        EXPECT_EQ(logical.task_capacity, physical.task_capacity);
        EXPECT_EQ(logical.arena_size, physical.arena_size);
        EXPECT_EQ(logical.off_runtime, physical.off_runtime);
        EXPECT_EQ(logical.off_copied_begin, physical.off_copied_begin);
        EXPECT_EQ(logical.off_copied_end, physical.off_copied_end);
        EXPECT_EQ(logical.off_scheduler, physical.off_scheduler);
        EXPECT_EQ(logical.off_sm_handle, physical.off_sm_handle);
        EXPECT_EQ(logical.off_mailbox, physical.off_mailbox);
        EXPECT_EQ(packet_header().runtime_offset, physical.off_runtime);
        EXPECT_EQ(packet_header().sm_offset, physical.off_copied_end);
        auto &sched = *restored.runtime->scheduler;
        std::vector<ChipReadyQueue *> queues;
        for (int i = 0; i < NUM_RESOURCE_SHAPES; ++i) {
            EXPECT_EQ(logical.sched.off_ready_queue_slots[i], physical.sched.off_ready_queue_slots[i]);
            EXPECT_EQ(logical.sched.off_ready_sync_queue_slots[i], physical.sched.off_ready_sync_queue_slots[i]);
            EXPECT_EQ(
                logical.sched.off_early_dispatch_queue_slots[i], physical.sched.off_early_dispatch_queue_slots[i]
            );
            queues.push_back(&sched.ready_queues[i]);
            queues.push_back(&sched.ready_sync_queues[i]);
        }
        EXPECT_EQ(logical.sched.off_dummy_ready_queue_slots, physical.sched.off_dummy_ready_queue_slots);
        EXPECT_EQ(logical.sched.off_graph_ready_queue_slots, physical.sched.off_graph_ready_queue_slots);
        EXPECT_EQ(logical.sched.off_graph_prepare_queue_slots, physical.sched.off_graph_prepare_queue_slots);
        queues.push_back(&sched.dummy_ready_queue);
        queues.push_back(&sched.graph_ready_queue);
        queues.push_back(&sched.graph_prepare_queue);
        auto &tasks = restored.runtime->sm_handle->header->tasks;
        for (auto *queue : queues) {
            ASSERT_EQ(queue->capacity, expected);
            const auto *unused = reinterpret_cast<const uint8_t *>(queue->slots + expected);
            EXPECT_TRUE(std::all_of(unused, unused + (capacity - expected) * sizeof(ChipReadyQueueSlot), [](auto b) {
                return b == 0xa5;
            }));
            for (int lap = 0; lap < 3; ++lap) {
                for (int i = 0; i < count; ++i)
                    ASSERT_TRUE(queue->push(&tasks.storage_at(i).slot));
                for (int i = 0; i < count; ++i) {
                    auto *slot = queue->pop();
                    ASSERT_EQ(slot, &tasks.storage_at(i).slot);
                    ASSERT_TRUE(queue->push(slot));
                }
                for (int i = 0; i < count; ++i)
                    EXPECT_EQ(queue->pop(), &tasks.storage_at(i).slot);
                EXPECT_EQ(queue->pop(), nullptr);
            }
        }
        ASSERT_NO_FATAL_FAILURE(retire());
    }
    ASSERT_EQ(build(graph_entry), 1);
    ASSERT_EQ(snapshot_graph(), 0);
    ASSERT_EQ(hbg::make_graph_host_args(snapshot, packet), 0);
    patch();
    dirty_working();
    ASSERT_EQ(restore(), hbg::GraphRestoreStatus::Ok);
    ASSERT_NO_FATAL_FAILURE(expect_initialized_prefix());
    EXPECT_EQ(restored.runtime->scheduler->dummy_ready_queue.capacity, capacity);
    ASSERT_NO_FATAL_FAILURE(retire());
    ASSERT_EQ(build(chain_entry), 2);
    ASSERT_EQ(snapshot_graph(), 0);
    ASSERT_EQ(hbg::make_graph_host_args(snapshot, packet), 0);
    patch();
    dirty_working();
    ASSERT_EQ(restore(), hbg::GraphRestoreStatus::Ok);
    ASSERT_NO_FATAL_FAILURE(expect_initialized_prefix());
    EXPECT_EQ(restored.runtime->scheduler->dummy_ready_queue.capacity, 64u);
}

TEST_F(HbgLargeGraphRestoreTest, ForgedFlatQueueCapacitiesRejectBeforeAnyTargetWrite) {
    ASSERT_NO_FATAL_FAILURE(prepare_large_slot());
    packet_task_count = 65;
    ASSERT_EQ(build(packet_window_entry), 65);
    ASSERT_EQ(snapshot_graph(), 0);
    ASSERT_EQ(hbg::make_graph_host_args(snapshot, packet), 0);
    patch();
    const auto original = packet.storage;
    const auto before = working_bytes();
    for (int queue = 0; queue < 9; ++queue) {
        for (uint64_t forged : {uint64_t{0}, uint64_t{64}, uint64_t{127}, uint64_t{256}, UINT64_MAX}) {
            packet.storage = original;
            auto *source = reinterpret_cast<RuntimeContext *>(
                reinterpret_cast<std::byte *>(packet.storage.data()) + packet.data_offset
            );
            auto &caps = source->prebuilt_layout.sched.capacities;
            if (queue < 3) caps.ready[queue] = forged;
            else if (queue < 6) caps.ready_sync[queue - 3] = forged;
            else if (queue == 6) caps.dummy = forged;
            else if (queue == 7) caps.graph_ready = forged;
            else caps.graph_prepare = forged;
            patch();
            EXPECT_EQ(restore(), hbg::GraphRestoreStatus::InvalidImage);
            EXPECT_EQ(working_bytes(), before);
        }
    }
}

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
    EXPECT_TRUE(std::all_of(heap, heap + binding.heap.capacity, [](auto value) {
        return value == 0xa5;
    }));
    hbg::GraphRestoreResult peer;
    ASSERT_EQ(hbg::acquire_graph_restore_result(registry, restored.generation, peer), hbg::GraphRestoreStatus::Ok);
    EXPECT_TRUE(std::all_of(heap, heap + binding.heap.capacity, [](auto value) {
        return value == 0xa5;
    }));
    ASSERT_NO_FATAL_FAILURE(retire());
    auto *source_sm = reinterpret_cast<std::byte *>(packet.storage.data()) + packet.data_offset +
                      packet_header().sm_offset - packet_header().runtime_offset;
    Buffer aligned;
    aligned.reserve(restored.sm_bytes);
    std::memcpy(aligned.data(), source_sm, restored.sm_bytes);
    auto *source = reinterpret_cast<ChipTaskStorage *>(aligned.data() + offsets.storage);
    auto &tensor = source[0].payload.tensor_data()[0];
    const auto original = tensor;
    for (uint64_t address : {binding.heap.address - 64, binding.heap.address + result.heap_bytes, HEAP_VIRTUAL_BASE}) {
        tensor.buffer.addr = address;
        std::memcpy(source_sm, aligned.data(), restored.sm_bytes);
        patch();
        const auto before = working_bytes();
        EXPECT_EQ(restore(), hbg::GraphRestoreStatus::InvalidImage);
        EXPECT_EQ(working_bytes(), before);
    }
    tensor = original;
    tensor.buffer.size = result.heap_bytes + 1;
    std::memcpy(source_sm, aligned.data(), restored.sm_bytes);
    patch();
    const auto before = working_bytes();
    EXPECT_EQ(restore(), hbg::GraphRestoreStatus::InvalidImage);
    EXPECT_EQ(working_bytes(), before);
}

TEST_F(HbgGraphRestoreTest, ForgedFlatTensorRankRejectsBeforeAnyTargetWrite) {
    ASSERT_EQ(build(chain_entry), 2);
    prepare(4096);
    ASSERT_EQ(hbg::seal_graph_execution_slot(context, 0, 19, 109, seal), 0);
    registry = reinterpret_cast<hbg::GraphSlotRegistry *>(seal.registry.address);
    ASSERT_EQ(simpler_aicpu_l1_hbg_register_execution_slot(&seal), 0);
    ASSERT_EQ(snapshot_graph(), 0);
    ASSERT_EQ(hbg::make_graph_host_args(snapshot, packet), 0);
    patch();
    const auto original = packet.storage;
    const auto before = working_bytes();
    const auto offsets = sm_layout::segment_offsets(sm_layout::image_extents(result.bind_usage));
    Buffer aligned;
    aligned.reserve(result.image_bytes);
    for (uint32_t ndims : {uint32_t{0}, uint32_t{MAX_TENSOR_DIMS + 1}, uint32_t{256}, UINT32_MAX}) {
        SCOPED_TRACE(ndims);
        packet.storage = original;
        auto *source = reinterpret_cast<std::byte *>(packet.storage.data()) + packet.data_offset +
                       packet_header().sm_offset - packet_header().runtime_offset + offsets.storage;
        auto *sm = source - offsets.storage;
        std::memcpy(aligned.data(), sm, result.image_bytes);
        auto *storage = reinterpret_cast<ChipTaskStorage *>(aligned.data() + offsets.storage);
        ASSERT_GT(storage[0].payload.tensor_count, 0);
        storage[0].payload.tensor_data()[0].ndims = ndims;
        std::memcpy(sm, aligned.data(), result.image_bytes);
        patch();
        ASSERT_EQ(
            hbg::validate_graph_packet(packet.storage.data(), packet.bytes, hbg::GraphPacketAddress::DeviceCopy),
            hbg::GraphPacketStatus::Ok
        );
        EXPECT_EQ(restore(), hbg::GraphRestoreStatus::InvalidImage);
        EXPECT_EQ(working_bytes(), before);
    }
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

TEST_F(HbgGraphRestoreTest, DirtyHeapMaterializesGraphAfterEveryRestore) {
    ASSERT_NO_FATAL_FAILURE(prepare_slot());
    for (uint8_t fill : {0xa5, 0x5a, 0xff}) {
        dirty_working();
        std::memset(reinterpret_cast<void *>(binding.heap.address), fill, binding.heap.capacity);
        ASSERT_EQ(restore(), hbg::GraphRestoreStatus::Ok);
        auto &tasks = restored.runtime->sm_handle->header->tasks;
        int graphs = 0;
        for (int32_t i = 0; i < tasks.total_tasks; ++i) {
            auto &outer = tasks.storage_at(i);
            if (outer.slot.task_kind != TaskKind::GRAPH) continue;
            ++graphs;
            auto *execution = graph_execution_localize(outer.slot);
            ASSERT_NE(execution, nullptr);
            ASSERT_EQ(execution->task_count, 2);
            EXPECT_EQ(execution->remaining_tasks.load(), 2);
            EXPECT_EQ(graph_execution_materialize_slice(outer.slot, *execution, 1), GraphMaterializeResult::PENDING);
            EXPECT_EQ(graph_execution_materialize_slice(outer.slot, *execution, 1), GraphMaterializeResult::PREPARED);
            for (int32_t j = 0; j < execution->task_count; ++j) {
                const auto &task = execution->task_at(j);
                EXPECT_EQ(execution->task_states[j].load(), CHIP_TASK_PENDING);
                EXPECT_EQ(task.slot.completed_subtasks.load(), 0);
                EXPECT_EQ(task.payload.published_block_count.load(), 0);
                EXPECT_EQ(task.payload.scalar_count, 0);
                ASSERT_EQ(task.payload.tensor_count, j + 1);
                for (int32_t k = 0; k < task.payload.tensor_count; ++k) {
                    const auto &tensor = task.payload.tensor_data()[k];
                    EXPECT_GE(tensor.buffer.addr, binding.heap.address);
                    EXPECT_LE(tensor.buffer.addr + tensor.buffer.size, binding.heap.address + result.heap_bytes);
                }
            }
            EXPECT_EQ(
                execution->task_at(0).payload.tensor_data()[0].buffer.addr,
                execution->task_at(1).payload.tensor_data()[0].buffer.addr
            );
        }
        EXPECT_EQ(graphs, 1);
        ASSERT_NO_FATAL_FAILURE(retire());
    }
}

TEST_F(HbgGraphRestoreTest, RepeatedRestoreRebuildsQueuesPointersAndWorkingImages) {
    ASSERT_NO_FATAL_FAILURE(prepare_slot());
    const auto source = packet.storage;
    ASSERT_EQ(restore(), hbg::GraphRestoreStatus::Ok);
    EXPECT_EQ(restored.generation, 1u);
    auto *runtime = restored.runtime;
    EXPECT_EQ(runtime->sm_handle->header->tasks.total_tasks, packet_header().total_tasks);
    EXPECT_EQ(runtime->scheduler->sm_header, runtime->sm_handle->header);
    auto &queue = runtime->scheduler->graph_ready_queue;
    EXPECT_EQ(queue.slots[queue.capacity - 1].sequence.load(), queue.capacity - 1);
    ASSERT_NO_FATAL_FAILURE(expect_initialized_prefix());
    const auto expected = restored_image_bytes();
    for (uint64_t iteration = 2; iteration <= 16; ++iteration) {
        ASSERT_NO_FATAL_FAILURE(retire());
        dirty_working();
        ASSERT_EQ(restore(), hbg::GraphRestoreStatus::Ok);
        EXPECT_EQ(restored.runtime, runtime);
        EXPECT_EQ(restored.generation, iteration);
        ASSERT_NO_FATAL_FAILURE(expect_initialized_prefix());
        EXPECT_EQ(restored_image_bytes(), expected);
        EXPECT_EQ(packet.storage, source);
        hbg::GraphRestoreResult peer;
        EXPECT_EQ(hbg::acquire_graph_restore_result(registry, iteration - 1, peer), hbg::GraphRestoreStatus::NotReady);
        EXPECT_EQ(hbg::acquire_graph_restore_result(registry, iteration, peer), hbg::GraphRestoreStatus::Ok);
        EXPECT_EQ(peer.runtime, runtime);
    }
}

TEST_F(HbgGraphRestoreTest, RuntimeTailFlushKeepsDevicePrefixInitialized) {
    ASSERT_EQ(build(empty_entry), 0);
    RuntimeArenaLayout layout{};
    ASSERT_EQ(hbg::make_kernel_graph_layout(capacity, layout), 0);
    hbg::GraphResourceRequirements empty{};
    ASSERT_EQ(hbg::get_graph_resource_requirements(result, layout, empty), 0);
    ASSERT_NO_FATAL_FAILURE(prepare_slot(true, empty.scheduler_state_bytes));
    struct Trace {
        std::vector<std::pair<const void *, size_t>> copies;
        std::vector<std::pair<const void *, size_t>> flushes;
    } trace;
    hbg::GraphRestoreOps ops;
    ops.context = &trace;
    ops.copy = [](void *context, void *dst, const void *src, size_t bytes) {
        static_cast<Trace *>(context)->copies.emplace_back(dst, bytes);
        std::memcpy(dst, src, bytes);
        return true;
    };
    ops.flush = [](void *context, const void *dst, size_t bytes) {
        static_cast<Trace *>(context)->flushes.emplace_back(dst, bytes);
        return true;
    };
    for (bool empty_graph : {false, true}) {
        if (empty_graph) {
            ASSERT_EQ(build(empty_entry), 0);
            ASSERT_EQ(snapshot_graph(), 0);
            ASSERT_EQ(hbg::make_graph_host_args(snapshot, packet), 0);
            patch();
        }
        dirty_working();
        trace.copies.clear();
        trace.flushes.clear();
        ASSERT_EQ(restore(ops), hbg::GraphRestoreStatus::Ok);
        ASSERT_FALSE(trace.copies.empty());
        ASSERT_FALSE(trace.flushes.empty());
        const auto &h = packet_header();
        const uint64_t live = h.sm_offset + result.image_bytes;
        EXPECT_EQ(
            trace.copies[0].first, reinterpret_cast<const void *>(seal.destinations[1].address + h.runtime_offset)
        );
        EXPECT_EQ(trace.copies[0].second, live - h.runtime_offset);
        EXPECT_EQ(
            trace.flushes[0].first, reinterpret_cast<const void *>(seal.destinations[1].address + h.runtime_offset)
        );
        EXPECT_EQ(trace.flushes[0].second, live - h.runtime_offset);
        ASSERT_EQ(trace.flushes.size(), trace.copies.size() + 1);
        for (size_t i = 1; i < trace.copies.size(); ++i)
            EXPECT_EQ(trace.flushes[i], trace.copies[i]);
        EXPECT_EQ(trace.flushes.back().first, &registry->restore);
        EXPECT_EQ(trace.flushes.back().second, sizeof(registry->restore));
        EXPECT_EQ(registry->restore.live_bytes[1], live);
        ASSERT_NO_FATAL_FAILURE(expect_initialized_prefix());
        ASSERT_NO_FATAL_FAILURE(retire());
    }
}

TEST_F(HbgGraphRestoreTest, NoncanonicalDestinationOffsetsRejectBeforeAnyTargetWrite) {
    ASSERT_NO_FATAL_FAILURE(prepare_slot());
    const auto original = packet.storage;
    const auto before = working_bytes();
    const auto offset = packet_header().runtime_offset;
    const auto capacity = seal.destinations[1].capacity;
    for (uint64_t invalid :
         {uint64_t{0}, offset - 1, offset + 1, packet_header().sm_offset, capacity - 1, UINT64_MAX}) {
        packet.storage = original;
        auto *region = reinterpret_cast<hbg::GraphImageRegion *>(
            reinterpret_cast<std::byte *>(packet.storage.data()) + sizeof(SimplerKernelInvocationHeader) +
            sizeof(hbg::GraphPacketHeader)
        );
        region->destination_offset = invalid;
        patch();
        EXPECT_EQ(restore(), hbg::GraphRestoreStatus::Rejected);
        EXPECT_EQ(working_bytes(), before);
    }
    packet.storage = original;
    auto *regions = reinterpret_cast<hbg::GraphImageRegion *>(
        reinterpret_cast<std::byte *>(packet.storage.data()) + sizeof(SimplerKernelInvocationHeader) +
        sizeof(hbg::GraphPacketHeader)
    );
    ASSERT_GT(packet_header().region_count, 1u);
    regions[1].destination_offset = 64;
    patch();
    EXPECT_EQ(restore(), hbg::GraphRestoreStatus::Rejected);
    EXPECT_EQ(working_bytes(), before);
    packet.storage = original;
    packet_header().runtime_offset -= 64;
    regions = reinterpret_cast<hbg::GraphImageRegion *>(
        reinterpret_cast<std::byte *>(packet.storage.data()) + sizeof(SimplerKernelInvocationHeader) +
        sizeof(hbg::GraphPacketHeader)
    );
    regions[0].destination_offset = packet_header().runtime_offset;
    patch();
    ASSERT_EQ(
        hbg::validate_graph_packet(packet.storage.data(), packet.bytes, hbg::GraphPacketAddress::DeviceCopy),
        hbg::GraphPacketStatus::Ok
    );
    EXPECT_EQ(restore(), hbg::GraphRestoreStatus::InvalidImage);
    EXPECT_EQ(working_bytes(), before);
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
    const size_t runtime = packet.data_offset;
    corrupt(
        runtime + offsetof(RuntimeContext, prebuilt_layout) + offsetof(RuntimeArenaLayout, off_scheduler), UINT64_MAX, 8
    );
    corrupt(runtime + offsetof(RuntimeContext, ops), 0x1234, sizeof(void *));
    const size_t sm = packet.data_offset + packet_header().sm_offset - packet_header().runtime_offset;
    corrupt(sm + offsetof(SharedMemoryTaskHeader, total_tasks), 999, 4);
    const auto offsets = sm_layout::segment_offsets(sm_layout::image_extents({packet_header().total_tasks, 0, 0, 0}));
    corrupt(sm + offsets.storage + offsetof(ChipTaskStorage, payload) + offsetof(TaskPayload, tensors), INT32_MAX, 4);
}

TEST_F(HbgGraphRestoreTest, HostOnlyCopyCannotBeSmuggledIntoTheDeviceGraphImage) {
    ASSERT_NO_FATAL_FAILURE(prepare_slot());
    const auto before = working_bytes();
    auto *base = reinterpret_cast<std::byte *>(packet.storage.data());
    const size_t sm = packet.data_offset + packet_header().sm_offset - packet_header().runtime_offset;
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
        static bool flush(void *ctx, const void *, size_t) {
            auto &self = *static_cast<Fault *>(ctx);
            return ++self.step != self.fail_at;
        }
    };
    const int regions = static_cast<int>(packet_header().region_count);
    const auto expected = restored_image_bytes();
    for (int fail_at = 1; fail_at <= 2 * regions + 1; ++fail_at) {
        const uint64_t previous = restored.generation;
        const auto output = restored;
        Fault fault{0, fail_at};
        ASSERT_NO_FATAL_FAILURE(retire());
        dirty_working();
        EXPECT_EQ(restore({&fault, Fault::copy, Fault::flush}), hbg::GraphRestoreStatus::CopyFailed);
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
        EXPECT_EQ(restored_image_bytes(), expected);
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
    const auto original_packet = packet;
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
        EXPECT_TRUE(std::all_of(bytes, bytes + live, [i](auto value) {
            return value == (i == 0 ? 0xa5 : 0);
        }));
        EXPECT_TRUE(std::all_of(bytes + live, bytes + dst.capacity, [](auto value) {
            return value == 0xa5;
        }));
    }
    ASSERT_NO_FATAL_FAILURE(retire());
    packet = original_packet;
    patch();
    dirty_working();
    ASSERT_EQ(restore(), hbg::GraphRestoreStatus::Ok);
    EXPECT_EQ(restored.total_tasks, packet_header().total_tasks);
    EXPECT_GT(restored.total_tasks, 0u);
    EXPECT_EQ(restored.runtime->sm_handle->header->tasks.total_tasks, static_cast<int32_t>(restored.total_tasks));
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
    ops.copy = [](void *, void *, const void *, size_t) {
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
    const auto before = target_bytes();
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
    EXPECT_EQ(target_bytes(), before);
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
    ops.copy = [](void *, void *, const void *, size_t) {
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
    ops.copy = [](void *, void *, const void *, size_t) {
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

namespace {

class HbgCallableBindingTest : public ::testing::Test {
protected:
    Runtime runtime;
    hbg::KernelCallableBinding binding;
    std::vector<uint8_t> image_a;
    std::vector<uint8_t> image_b;
    hbg::HbgCallableRegistration a;
    hbg::HbgCallableRegistration b;

    static std::vector<uint8_t> image(int children, uint8_t code_byte) {
        const int32_t ids[] = {3, 8};
        std::vector<uint8_t> child_images[2];
        for (int i = 0; i < children; ++i) {
            child_images[i] = make_callable<CORE_MAX_TENSOR_ARGS>(nullptr, 0, &code_byte, 1);
            reinterpret_cast<CoreCallable *>(child_images[i].data())->resolved_addr_ = 0x1000 + code_byte + i;
        }
        return make_callable<CoreCallable, CHIP_MAX_TENSOR_ARGS, 1024>(
            nullptr, 0, "orch", &code_byte, 1, ids, child_images, children, ""
        );
    }

    static CoreCallable *child(std::vector<uint8_t> &storage, int index) {
        auto *parent = reinterpret_cast<ChipCallable *>(storage.data());
        return reinterpret_cast<CoreCallable *>(parent->storage_ + parent->child_offsets_[index]);
    }

    hbg::HbgCallableRegistration registration(std::vector<uint8_t> &storage, int id) {
        hbg::HbgCallableRegistration value;
        value.context_generation = 1;
        value.runtime_address = reinterpret_cast<uintptr_t>(&runtime);
        value.register_table_address = 0x1000;
        value.callable_address = reinterpret_cast<uintptr_t>(storage.data());
        value.callable_bytes = storage.size();
        value.callable_hash = id + 1;
        value.function_hash = id + 2;
        value.callable_id = id;
        value.device_id = 0;
        value.runtime_binary_id = 1;
        return value;
    }

    void SetUp() override {
        image_a = image(2, 1);
        image_b = image(1, 2);
        a = registration(image_a, 0);
        b = registration(image_b, 1);
        ASSERT_TRUE(hbg::valid_hbg_callable_registration(a));
        ASSERT_TRUE(hbg::valid_hbg_callable_registration(b));
    }

    void expect_a() {
        EXPECT_EQ(runtime.get_active_callable_id(), a.callable_id);
        EXPECT_EQ(runtime.get_function_bin_addr(3), reinterpret_cast<uintptr_t>(child(image_a, 0)));
        EXPECT_EQ(runtime.get_function_bin_addr(8), reinterpret_cast<uintptr_t>(child(image_a, 1)));
    }
};

TEST_F(HbgCallableBindingTest, RepeatedBindingPreservesEveryChildHeader) {
    ASSERT_EQ(binding.bind(&runtime, a), hbg::CallableBindingResult::Rebound);
    expect_a();
    ASSERT_EQ(binding.bind(&runtime, a), hbg::CallableBindingResult::Reused);
    expect_a();
}

TEST_F(HbgCallableBindingTest, DifferentCodeWithSharedFunctionIdSupportsAba) {
    ASSERT_EQ(binding.bind(&runtime, a), hbg::CallableBindingResult::Rebound);
    ASSERT_EQ(binding.bind(&runtime, b), hbg::CallableBindingResult::Rebound);
    EXPECT_EQ(runtime.get_function_bin_addr(3), reinterpret_cast<uintptr_t>(child(image_b, 0)));
    EXPECT_EQ(runtime.get_function_bin_addr(8), 0u);
    ASSERT_EQ(binding.bind(&runtime, a), hbg::CallableBindingResult::Rebound);
    expect_a();
}

TEST_F(HbgCallableBindingTest, PartialFailureRevokesPriorBindingBeforeRetry) {
    for (bool bad_offset : {false, true}) {
        ASSERT_NE(binding.bind(&runtime, a), hbg::CallableBindingResult::Rejected);
        auto failing_image = image(2, 3);
        auto failing = registration(failing_image, 2);
        if (bad_offset) {
            reinterpret_cast<ChipCallable *>(failing_image.data())->child_offsets_[1] = UINT32_MAX;
        } else {
            child(failing_image, 1)->resolved_addr_ = 0;
        }
        ASSERT_EQ(binding.bind(&runtime, failing), hbg::CallableBindingResult::Rejected);
        EXPECT_EQ(runtime.get_active_callable_id(), -1);
        EXPECT_EQ(runtime.get_function_bin_addr(3), reinterpret_cast<uintptr_t>(child(failing_image, 0)));
        ASSERT_EQ(binding.bind(&runtime, a), hbg::CallableBindingResult::Rebound);
        expect_a();
    }
}

TEST_F(HbgCallableBindingTest, ActiveResetRebuildsTheSameRegistration) {
    ASSERT_EQ(binding.bind(&runtime, a), hbg::CallableBindingResult::Rebound);
    runtime.clear_function_bin_addrs();
    runtime.set_active_callable_id(-1);
    ASSERT_EQ(binding.bind(&runtime, a), hbg::CallableBindingResult::Rebound);
    expect_a();
}

TEST_F(HbgCallableBindingTest, NewGenerationCannotReuseOldSuccessfulBinding) {
    ASSERT_EQ(binding.bind(&runtime, a), hbg::CallableBindingResult::Rebound);
    auto next = a;
    ++next.context_generation;
    child(image_a, 1)->resolved_addr_ = 0;
    ASSERT_EQ(binding.bind(&runtime, next), hbg::CallableBindingResult::Rejected);
    EXPECT_EQ(runtime.get_active_callable_id(), -1);
    child(image_a, 1)->resolved_addr_ = 0x2000;
    ASSERT_EQ(binding.bind(&runtime, next), hbg::CallableBindingResult::Rebound);
    expect_a();
}

TEST_F(HbgCallableBindingTest, RuntimeIdentityMustMatchEvenWithTheSameActiveId) {
    ASSERT_EQ(binding.bind(&runtime, a), hbg::CallableBindingResult::Rebound);
    Runtime another;
    another.set_active_callable_id(a.callable_id);
    ASSERT_EQ(binding.bind(&another, a), hbg::CallableBindingResult::Rejected);
    EXPECT_EQ(another.get_active_callable_id(), -1);
    auto next = a;
    next.runtime_address = reinterpret_cast<uintptr_t>(&another);
    ASSERT_EQ(binding.bind(&another, next), hbg::CallableBindingResult::Rebound);
    EXPECT_EQ(another.get_function_bin_addr(3), reinterpret_cast<uintptr_t>(child(image_a, 0)));
    ASSERT_EQ(binding.bind(&runtime, a), hbg::CallableBindingResult::Rebound);
    expect_a();
}

}  // namespace

TEST(HbgRuntimePublication, SpansCoverChangedFieldsWithoutPublishingWorkerLines) {
    for (bool functions_changed : {false, true}) {
        SCOPED_TRACE(functions_changed);
        Runtime runtime;
        const auto *base = reinterpret_cast<const uint8_t *>(&runtime);
        const std::vector<uint8_t> before(base, base + sizeof(runtime));
        constexpr uintptr_t pattern = UINTPTR_MAX / 0xff * 0x5a;
        runtime.set_gm_sm_ptr(reinterpret_cast<void *>(pattern));
        runtime.set_prebuilt_arena(reinterpret_cast<void *>(pattern), pattern);
        runtime.host_total_tasks = 0x35353535;
        runtime.sm_image_bytes = pattern;
        runtime.set_active_callable_id(0x35353535);
        if (functions_changed)
            for (int i = 0; i < RUNTIME_MAX_FUNC_ID; ++i)
                runtime.replay_function_bin_addr(i, pattern);
        const auto spans = runtime.kernel_graph_publication_spans(functions_changed);
        EXPECT_EQ(spans[0].address, runtime.func_id_to_addr_);
        EXPECT_EQ(spans[0].bytes, functions_changed ? sizeof(runtime.func_id_to_addr_) : 0u);
        const auto object_begin = reinterpret_cast<uintptr_t>(&runtime);
        const auto owned_end = reinterpret_cast<uintptr_t>(runtime.teardown_gates) + sizeof(runtime.teardown_gates);
        size_t published_bytes = 0;
        for (const auto &span : spans) {
            if (span.bytes == 0) continue;
            const auto begin = reinterpret_cast<uintptr_t>(span.address);
            ASSERT_GE(begin, object_begin);
            ASSERT_LE(begin + span.bytes, object_begin + sizeof(runtime));
            // cache_ops.cpp rounds both ends to 64-byte AICPU cache lines.
            const auto rounded_begin = begin & ~uintptr_t{63};
            const auto rounded_end = (begin + span.bytes + 63) & ~uintptr_t{63};
            EXPECT_GE(rounded_begin, owned_end);
            EXPECT_GT(rounded_end, rounded_begin);
            published_bytes += span.bytes;
        }
        EXPECT_LT(published_bytes, (functions_changed ? sizeof(runtime.func_id_to_addr_) : 0u) + 1024u);
        for (size_t offset = 0; offset < sizeof(runtime); ++offset) {
            if (before[offset] == base[offset]) continue;
            const auto address = object_begin + offset;
            EXPECT_TRUE(
                std::any_of(
                    spans.begin(), spans.end(),
                    [&](const auto &span) {
                        const auto begin = reinterpret_cast<uintptr_t>(span.address);
                        return address >= begin && address - begin < span.bytes;
                    }
                )
            ) << "unpublished changed byte "
              << offset;
        }
    }
}

TEST(HbgRuntimePublication, HandshakeSpanStaysWithinHostResetLinesWithoutReadingConfiguration) {
    Runtime runtime;
    for (int32_t workers : {-1, 0, 1, RUNTIME_MAX_WORKER, INT32_MAX}) {
        SCOPED_TRACE(workers);
        runtime.worker_count = workers;
        const auto span = runtime.kernel_graph_handshake_span();
        const auto begin = reinterpret_cast<uintptr_t>(span.address);
        const auto end = begin + span.bytes;
        EXPECT_EQ(begin, reinterpret_cast<uintptr_t>(&runtime.kernel_prelaunch));
        EXPECT_EQ(end, reinterpret_cast<uintptr_t>(runtime.workers + RUNTIME_MAX_WORKER));
        EXPECT_EQ(begin % 64, 0u);
        EXPECT_EQ(end % 64, 0u);
        EXPECT_LE(end, reinterpret_cast<uintptr_t>(runtime.teardown_gates));
        EXPECT_LE(end, reinterpret_cast<uintptr_t>(&runtime.worker_count));
        EXPECT_LE(end, reinterpret_cast<uintptr_t>(runtime.func_id_to_addr_));
    }
}
