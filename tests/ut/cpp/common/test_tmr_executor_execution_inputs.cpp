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
#include <atomic>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <thread>
#include <vector>

#include "aicpu/device_log.h"
#include "aicpu/device_time.h"
#include "aicpu/platform_aicpu_affinity.h"
#include "aicpu/platform_regs.h"
#include "common/kernel_args.h"
#include "task_interface/callable.h"
#include "task_interface/kernel_callable_validation.h"
#include "host_log.h"
#include "host/tmr_dispatch_packet.h"
#include "task_interface/tmr_kernel_context.h"
#include "tensormap_and_ringbuffer/kernel_execution_inputs.h"
#include "tensormap_and_ringbuffer/kernel_execution.h"
#include "tensormap_and_ringbuffer/kernel_native_status.h"
#include "worker/tmr_kernel_invocation.h"

namespace {
thread_local int affinity_index = 0;
std::array<uint64_t, PLATFORM_MAX_CORES> register_bases{};
std::array<uint64_t, 3> register_cells{};
std::atomic<int> opened_windows{0};
std::atomic<int> closed_windows{0};
std::atomic<int> register_publications{0};
std::atomic<const void *> watched_image{nullptr};
std::atomic<int> image_invalidations{0};
}  // namespace

namespace aicpu_cache_maintenance {
// Preserve simulation's no-op cache behavior while observing image access.
void invalidate_range_impl(const void *address, size_t) {
    if (address == watched_image.load(std::memory_order_relaxed)) ++image_invalidations;
}
void flush_range_impl(const void *, size_t) {}
}  // namespace aicpu_cache_maintenance

// The orchestration submits only dependency tasks, never AICore work. These
// platform fixtures supply ready-core reports and idle register windows.
int platform_aicpu_affinity_thread_idx() { return affinity_index; }
void platform_aicpu_affinity_set_thread_idx(int index) { affinity_index = index; }
int32_t platform_aicpu_current_cpu() { return 10 + affinity_index; }
int platform_aicpu_prepare_kernel_thread() { return 0; }
extern "C" void set_platform_regs(uint64_t) { ++register_publications; }
extern "C" uint64_t get_platform_regs() { return reinterpret_cast<uint64_t>(register_bases.data()); }
extern "C" uint64_t get_platform_pmu_reg_addrs() { return 0; }
uint32_t platform_get_physical_cores_count() { return PLATFORM_MAX_CORES; }
volatile uint32_t *get_reg_ptr(uint64_t base, RegId) { return reinterpret_cast<volatile uint32_t *>(base); }
uint64_t read_reg(uint64_t base, RegId) {
    return __atomic_load_n(reinterpret_cast<uint64_t *>(base), __ATOMIC_ACQUIRE);
}
uint32_t reg_load_acquire(const volatile uint32_t *ptr) { return __atomic_load_n(ptr, __ATOMIC_ACQUIRE); }
void reg_store_release(volatile uint32_t *, uint32_t) {}
void platform_init_aicore_regs(uint64_t) { ++opened_windows; }
uint64_t platform_aicore_exit_deadline() { return get_sys_cnt_aicpu() + 10000000000ULL; }
void platform_close_aicore_window(uint64_t) { ++closed_windows; }

extern "C" int simpler_aicpu_register_callable(void *);
extern "C" int simpler_aicpu_prepare_tmr_context(void *);
extern "C" int simpler_aicpu_register_tmr_kernel_callable(void *);
extern "C" int simpler_aicpu_release_tmr_context(void *);
void corrupt_kernel_arch_argument(KernelArgs &args, int fault);

namespace {
using namespace simpler::tmr;

class TmrExecutorExecutionInputsTest : public ::testing::Test {
protected:
    void SetUp() override {
        register_publications = 0;
        watched_image = nullptr;
        image_invalidations = 0;
        ASSERT_EQ(set_host_log_state(HostLogger::get_instance().state()), 0);
        uint64_t windows[CHIP_MAX_RING_DEPTH] = {16, 16, 16, 16};
        uint64_t heaps[CHIP_MAX_RING_DEPTH] = {1024, 1024, 1024, 1024};
        int32_t deps[CHIP_MAX_RING_DEPTH] = {4, 4, 4, 4};
        layout = runtime_reserve_layout(arena, windows, heaps, deps);
        ASSERT_NE(arena.commit(), nullptr);
        const size_t sm_size = SharedMemoryHandle::calculate_size_per_ring(windows);
        const size_t sm_offset = sm.reserve(sm_size, CHIP_ALIGN_SIZE);
        ASSERT_NE(sm.commit(), nullptr);
        SharedMemoryHandle initialized_sm;
        ASSERT_TRUE(initialized_sm.init_per_ring(sm.region_ptr(sm_offset), sm_size, windows, heaps));
        heap.reserve(4096, DeviceArena::kDefaultBaseAlign);
        ASSERT_NE(heap.commit(), nullptr);
        auto *runtime = runtime_init_data_from_layout(
            arena, layout, MODE_EXECUTE, sm.region_ptr(sm_offset), sm_size, heap.base(), heaps
        );
        ASSERT_NE(runtime, nullptr);
        runtime->prebuilt_layout = layout;
        binding = {
            {reinterpret_cast<uint64_t>(&identity), 13},
            resident.get(),
            {sm.region_ptr(sm_offset), sm_size, sm_size},
            {arena.base(), layout.offsets.arena_size, layout.offsets.arena_size},
            layout.offsets.off_runtime
        };
        resident->dev.worker_count = 3;
        resident->dev.aicpu_thread_num = 2;
        for (size_t i = 0; i < register_cells.size(); ++i) {
            register_bases[i] = reinterpret_cast<uint64_t>(&register_cells[i]);
        }
        std::ifstream library(TMR_EXECUTOR_ORCH_FIXTURE, std::ios::binary);
        ASSERT_TRUE(library.is_open());
        binary.assign(std::istreambuf_iterator<char>(library), std::istreambuf_iterator<char>());
        ASSERT_FALSE(binary.empty());
        register_orchestration(3, "orchestration_a", "config_a");
        register_orchestration(4, "orchestration_b", "config_b");
    }

    void TearDown() override {
        EXPECT_EQ(kernel_execution_status(), -1);
        if (registered_context) EXPECT_EQ(simpler_aicpu_release_tmr_context(&registration), 0);
    }

    void register_orchestration(int id, const char *entry, const char *config) {
        RegisterCallableArgs args{};
        args.active_callable_id = id;
        args.dev_orch_so_addr = reinterpret_cast<uint64_t>(binary.data());
        args.dev_orch_so_size = binary.size();
        std::strcpy(args.device_orch_func_name, entry);
        std::strcpy(args.device_orch_config_name, config);
        ASSERT_EQ(simpler_aicpu_register_callable(&args), 0);
    }

    void prepare_native_context(const char *entry = "orchestration_a") {
        resident->dev.aicpu_launch_count = 3;
        resident->dev.aicpu_allowed_cpu_count = 2;
        resident->dev.aicpu_allowed_cpus[0] = 10;
        resident->dev.aicpu_allowed_cpus[1] = 11;
        resident_args.runtime_args = resident.get();
        resident_args.regs = get_platform_regs();
        descriptor.version = kTmrKernelContextVersion;
        descriptor.bytes = sizeof(descriptor);
        descriptor.context_generation = 13;
        descriptor.self_address = reinterpret_cast<uint64_t>(&descriptor);
        descriptor.resident_runtime = reinterpret_cast<uint64_t>(resident.get());
        descriptor.resident_kernel_args = reinterpret_cast<uint64_t>(&resident_args);
        descriptor.heap_base = reinterpret_cast<uint64_t>(heap.base());
        descriptor.heap_capacity = descriptor.heap_required = 4096;
        descriptor.sm_base = reinterpret_cast<uint64_t>(binding.sm.base);
        descriptor.sm_capacity = descriptor.sm_required = binding.sm.required_bytes;
        descriptor.arena_base = reinterpret_cast<uint64_t>(binding.arena.base);
        descriptor.arena_capacity = descriptor.arena_required = binding.arena.required_bytes;
        descriptor.runtime_offset = binding.runtime_offset;
        descriptor.control_address = reinterpret_cast<uint64_t>(&control);
        descriptor.control_bytes = sizeof(control);
        descriptor.reports_address = reinterpret_cast<uint64_t>(reports.data());
        descriptor.reports_bytes = sizeof(reports);
        descriptor.launch_threads = 3;
        descriptor.execution_threads = 2;
        descriptor.worker_count = 3;
        registration = {
            descriptor.self_address, descriptor.context_generation, reinterpret_cast<uint64_t>(residencies.data()),
            static_cast<uint32_t>(residencies.size()), sizeof(KernelCallableDeviceResidency)
        };
        ASSERT_EQ(simpler_aicpu_prepare_tmr_context(&registration), 0);
        registered_context = true;
        ASSERT_EQ(simpler_aicpu_prepare_tmr_context(&registration), 0);
        const ArgDirection signature[]{ArgDirection::OUT, ArgDirection::SCALAR};
        kernel_image = make_callable<CoreCallable, CHIP_MAX_TENSOR_ARGS, 1024>(
            signature, 2, 1, entry, binary.data(), binary.size(), nullptr, nullptr, 0, "config_a"
        );
        residencies[7] = {17, reinterpret_cast<uint64_t>(kernel_image.data()), kernel_image.size(), 7, 0};
        TmrCallableRegistrationArgs callable_registration{13, reinterpret_cast<uint64_t>(&residencies[7]), 7, 0, 17};
        ASSERT_EQ(simpler_aicpu_register_tmr_kernel_callable(&callable_registration), 0);
        ASSERT_EQ(simpler_aicpu_register_tmr_kernel_callable(&callable_registration), 0);
        binding.identity = {descriptor.self_address, descriptor.context_generation};
    }

    ChipStorageTaskArgs arguments(uint64_t value) {
        ChipStorageTaskArgs args;
        const uint32_t shape[] = {static_cast<uint32_t>(output.size() * 2)};
        args.add_tensor(make_tensor_external(output.data(), shape, 1, DataType::INT32, AddressSpace::DEVICE));
        args.add_scalar(value);
        return args;
    }

    std::vector<int32_t> coordinated_round(
        const KernelCallableView &callable, ByteSpan packet, int32_t admission = 0, int32_t execution_threads = 2,
        bool native = false
    ) {
        opened_windows = 0;
        closed_windows = 0;
        control = {};
        reports = {};
        for (auto &cell : register_cells)
            cell = 0;
        std::vector<int32_t> allowed(execution_threads);
        for (int32_t i = 0; i < execution_threads; ++i)
            allowed[i] = 10 + i;
        KernelExecutionRequest request{packet,
                                       callable,
                                       binding,
                                       {&control, reports.data(), 3, 0},
                                       allowed.data(),
                                       execution_threads,
                                       execution_threads + 1,
                                       admission};
        std::vector<std::thread> cores;
        for (size_t i = 0; i < reports.size(); ++i) {
            cores.emplace_back([&, i] {
                auto &report = reports[i];
                report.physical_core_id = i;
                report.core_type = static_cast<uint32_t>(i == 0 ? CoreType::AIC : CoreType::AIV);
                __atomic_store_n(&report.ready, static_cast<uint32_t>(i + 1), __ATOMIC_RELEASE);
                while (__atomic_load_n(&report.command, __ATOMIC_ACQUIRE) !=
                       static_cast<uint32_t>(TmrCoreCommand::Cancel)) {}
                const bool opened = report.round_epoch != 0;
                if (opened) __atomic_store_n(&register_cells[i], uint64_t{AICORE_EXITED_VALUE}, __ATOMIC_RELEASE);
                __atomic_store_n(&report.exited, static_cast<uint32_t>(i + 1), __ATOMIC_RELEASE);
                if (opened) {
                    while (__atomic_load_n(&report.release, __ATOMIC_ACQUIRE) !=
                           static_cast<uint32_t>(TmrCoreRelease::Release)) {}
                }
            });
        }
        std::vector<int32_t> result(execution_threads + 1);
        std::vector<std::thread> cpus;
        for (size_t i = 0; i < result.size(); ++i)
            cpus.emplace_back([&, i] {
                affinity_index = static_cast<int32_t>(i);
                result[i] = native ? simpler_aicpu_kernel_exec(const_cast<uint8_t *>(packet.data)) :
                                     execute_kernel_round(request, 10 + i);
            });
        for (auto &thread : cpus)
            thread.join();
        for (auto &thread : cores)
            thread.join();
        EXPECT_EQ(control.completion, static_cast<uint32_t>(TmrCompletion::Complete));
        EXPECT_EQ(control.cleanup_status, 0);
        for (size_t i = 0; i < reports.size(); ++i)
            EXPECT_EQ(reports[i].exited, i + 1);
        return result;
    }

    void expect_successful_reuse() {
        EXPECT_EQ(kernel_execution_status(), -1);
        output.fill(0);
        PreparedInvocationView callable{3, 1, 1, 17};
        TmrEncodingCandidate packet;
        TmrEncodingCache cache;
        auto args = arguments(71);
        ASSERT_EQ(encode_tmr_invocation(args, callable, binding.identity, cache, &packet), InvocationStatus::Ok);
        const auto results = coordinated_round({callable, {}}, packet.packet());
        for (int32_t result : results)
            EXPECT_EQ(result, 0);
        EXPECT_EQ(control.runtime_status, 0);
        EXPECT_EQ(opened_windows.load(), 3);
        EXPECT_EQ(closed_windows.load(), 3);
        EXPECT_EQ(output[0], 3u);
        EXPECT_EQ(output[1], 71u);
        EXPECT_EQ(output[3], 3u);
        EXPECT_EQ(output[4], 71u);
        EXPECT_NE(output[2], 0u);
        EXPECT_EQ(output[2], output[5]);
        expect_resident_configuration(resident->dev.serial_orch_sched);
        EXPECT_EQ(kernel_execution_status(), -1);
    }

    void expect_resident_configuration(bool serial) {
        EXPECT_EQ(resident->get_active_callable_id(), -1);
        EXPECT_EQ(resident->get_orch_args().tensor_count(), 0);
        EXPECT_EQ(resident->get_orch_args().scalar_count(), 0);
        for (uint64_t address : resident->dev.func_id_to_addr_)
            EXPECT_EQ(address, 0u);
        EXPECT_EQ(resident->dev.worker_count, 3);
        EXPECT_EQ(resident->dev.aicpu_thread_num, 2);
        EXPECT_EQ(resident->dev.serial_orch_sched, serial);
        EXPECT_EQ(resident->dev.ready_queue_shards, RUNTIME_DEFAULT_READY_QUEUE_SHARDS);
        EXPECT_EQ(resident->dev.aicpu_allowed_cpu_count, 0);
        EXPECT_EQ(resident->dev.aicpu_launch_count, 0);
        EXPECT_EQ(resident->get_gm_sm_ptr(), nullptr);
        EXPECT_EQ(resident->get_prebuilt_arena_base(), nullptr);
        EXPECT_EQ(resident->get_prebuilt_runtime_offset(), 0u);
    }

    std::unique_ptr<Runtime> resident{std::make_unique<Runtime>()};
    DeviceArena arena;
    DeviceArena sm;
    RuntimeArenaLayout layout{};
    KernelBindingView binding{};
    DeviceArena heap;
    std::vector<char> binary;
    std::array<uint64_t, 6> output{};
    uint64_t identity{0};
    TmrLaunchControl control{};
    std::array<TmrCoreReport, 3> reports{};
    TmrKernelContextDescriptor descriptor{};
    KernelArgs resident_args{};
    TmrContextRegistrationArgs registration{};
    std::array<KernelCallableDeviceResidency, MAX_REGISTERED_CALLABLE_IDS> residencies{};
    std::vector<uint8_t> kernel_image;
    bool registered_context{false};
};

TEST_F(TmrExecutorExecutionInputsTest, CoordinatedRoundsRunABAWithOneFinalVerdictAndStableStorage) {
    uint64_t storage = 0;
    uint64_t invocation = 90;
    for (bool serial : {false, true}) {
        resident->dev.serial_orch_sched = serial;
        for (int32_t id : {3, 4, 3}) {
            output.fill(0);
            PreparedInvocationView callable{id, 1, 1, 17};
            TmrEncodingCandidate packet;
            TmrEncodingCache cache;
            auto args = arguments(++invocation);
            ASSERT_EQ(encode_tmr_invocation(args, callable, binding.identity, cache, &packet), InvocationStatus::Ok);
            const auto results = coordinated_round({callable, {}}, packet.packet());
            for (int32_t result : results)
                EXPECT_EQ(result, 0);
            EXPECT_EQ(control.runtime_status, 0);
            EXPECT_EQ(opened_windows.load(), 3);
            EXPECT_EQ(closed_windows.load(), 3);
            EXPECT_EQ(output[0], id);
            EXPECT_EQ(output[3], id);
            EXPECT_EQ(output[1], invocation);
            EXPECT_EQ(output[4], invocation);
            EXPECT_NE(output[2], 0u);
            EXPECT_EQ(output[2], output[5]);
            if (storage != 0) EXPECT_EQ(output[2], storage);
            storage = output[2];
            EXPECT_EQ(kernel_execution_status(), -1);
            expect_resident_configuration(serial);
        }
    }
}

TEST_F(TmrExecutorExecutionInputsTest, NativeEntryRoutesMalformedStaleAndOutOfTablePacketsThroughCancellation) {
    ASSERT_EQ(simpler_aicpu_kernel_exec(nullptr), kAicpuKernelInnerError);
    prepare_native_context();
    ASSERT_TRUE(registered_context);
    const auto *const stable_args_address = &resident_args;
    const auto stable_args = resident_args;
    const auto stable_descriptor = descriptor;
    const auto *arena_runtime = reinterpret_cast<const RuntimeContext *>(
        static_cast<const uint8_t *>(binding.arena.base) + binding.runtime_offset
    );
    std::array<uint8_t, sizeof(RuntimeArenaLayout)> stable_layout{};
    std::memcpy(stable_layout.data(), &arena_runtime->prebuilt_layout, stable_layout.size());
    const PreparedInvocationView callable{7, 1, 1, 17};
    for (int fault : {0, 1, 2, 3, 4, 5, 0}) {
        SCOPED_TRACE(fault);
        output.fill(0);
        auto args = arguments(83);
        TmrEncodingCandidate candidate;
        TmrEncodingCache cache;
        ASSERT_EQ(encode_tmr_invocation(args, callable, binding.identity, cache, &candidate), InvocationStatus::Ok);
        std::vector<uint8_t> packet;
        ASSERT_EQ(
            make_tmr_dispatch_packet(
                candidate, callable, binding.identity, reinterpret_cast<uint64_t>(&residencies[7]), &packet
            ),
            InvocationStatus::Ok
        );
        SimplerKernelDispatchArgs envelope;
        std::memcpy(&envelope, packet.data(), sizeof(envelope));
        if (fault == 2) ++envelope.invocation.generation;
        if (fault == 3) envelope.residency_address = 0x1000;
        if (fault == 4) envelope.invocation.scalar_count = -1;
        if (fault == 5) --envelope.packet_bytes;
        std::memcpy(packet.data(), &envelope, sizeof(envelope));
        const ByteSpan bytes{fault == 1 ? nullptr : packet.data(), packet.size()};
        const auto results = coordinated_round({}, bytes, 0, 2, true);
        const int expected = fault == 0 ? 0 : (fault == 2 ? 3 : (fault == 3 ? 2 : 1));
        for (int result : results)
            EXPECT_EQ(result, fault == 0 ? kAicpuKernelSuccess : kAicpuKernelInnerError);
        EXPECT_EQ(opened_windows.load(), fault == 0 ? 3 : 0);
        EXPECT_EQ(closed_windows.load(), fault == 0 ? 3 : 0);
        EXPECT_EQ(control.runtime_status, expected);
        EXPECT_EQ(output[3], fault == 0 ? 3u : 0u);
        EXPECT_EQ(kernel_execution_status(), -1);
        EXPECT_EQ(&resident_args, stable_args_address);
        EXPECT_EQ(std::memcmp(&resident_args, &stable_args, sizeof(resident_args)), 0);
        EXPECT_EQ(std::memcmp(&descriptor, &stable_descriptor, sizeof(descriptor)), 0);
        EXPECT_EQ(std::memcmp(&arena_runtime->prebuilt_layout, stable_layout.data(), stable_layout.size()), 0);
    }
    auto wrong_context = registration;
    ++wrong_context.context_generation;
    EXPECT_EQ(simpler_aicpu_release_tmr_context(&wrong_context), kAicpuKernelInnerError);
    auto wrong_slot = TmrCallableRegistrationArgs{13, reinterpret_cast<uint64_t>(&residencies[7]), 7, 0, 99};
    EXPECT_EQ(simpler_aicpu_register_tmr_kernel_callable(&wrong_slot), kAicpuKernelInnerError);
}

TEST(TmrNativeStatusTest, SeparatesDispatchAndRuntimeDiagnosticsFromCannStatus) {
    EXPECT_EQ(to_aicpu_native_status(0), 0);
    for (int32_t status : {1, 2, 3, 4, 5, 6, 7, -1, -47, INT32_MIN, INT32_MAX})
        EXPECT_EQ(to_aicpu_native_status(status), 2);
    EXPECT_EQ(classify_kernel_dispatch_status(0, 0), static_cast<int>(KernelDispatchStatus::Success));
    for (int32_t status : {1, 2, 3, 4, 5})
        EXPECT_EQ(classify_kernel_dispatch_status(status, 0), status);
    for (int32_t status : {-47, INT32_MIN, INT32_MAX})
        EXPECT_EQ(classify_kernel_dispatch_status(status, 0), static_cast<int>(KernelDispatchStatus::ExecutionFailed));
    EXPECT_EQ(classify_kernel_dispatch_status(0, -1), static_cast<int>(KernelDispatchStatus::CleanupFailed));
    EXPECT_EQ(classify_kernel_dispatch_status(-47, -2), static_cast<int>(KernelDispatchStatus::CleanupFailed));
}

TEST_F(TmrExecutorExecutionInputsTest, NativeRuntimeFailureKeepsRawReportAndReturnsCannInnerError) {
    prepare_native_context("orchestration_error");
    ASSERT_TRUE(registered_context);
    const PreparedInvocationView callable{7, 1, 1, 17};
    auto args = arguments(57);
    TmrEncodingCandidate candidate;
    TmrEncodingCache cache;
    ASSERT_EQ(encode_tmr_invocation(args, callable, binding.identity, cache, &candidate), InvocationStatus::Ok);
    std::vector<uint8_t> packet;
    ASSERT_EQ(
        make_tmr_dispatch_packet(
            candidate, callable, binding.identity, reinterpret_cast<uint64_t>(&residencies[7]), &packet
        ),
        InvocationStatus::Ok
    );
    const auto results = coordinated_round({}, {packet.data(), packet.size()}, 0, 2, true);
    for (int32_t result : results)
        EXPECT_EQ(result, kAicpuKernelInnerError);
    EXPECT_EQ(control.runtime_status, -SIMPLER_ERROR_INVALID_ARGS);
    EXPECT_EQ(control.cleanup_status, 0);
    EXPECT_EQ(opened_windows.load(), 3);
    EXPECT_EQ(closed_windows.load(), 3);
    EXPECT_EQ(output[0], 3u);
    EXPECT_EQ(output[3], 3u);
    EXPECT_EQ(output[1], 57u);
    EXPECT_EQ(output[4], 57u);
    EXPECT_NE(output[2], 0u);
    EXPECT_EQ(output[2], output[5]);
    EXPECT_EQ(kernel_execution_status(), -1);
    // Direct Host execution tests cleanup, not recovery after a CANN error.
}

TEST_F(TmrExecutorExecutionInputsTest, NativeRegistrationRejectsChangedStaticIdentityAndInvalidCallableSpans) {
    prepare_native_context();
    ASSERT_TRUE(registered_context);
    ASSERT_EQ(register_publications.load(), 1);
    const auto original = resident_args;
    resident_args.regs += alignof(uint64_t);
    EXPECT_EQ(simpler_aicpu_prepare_tmr_context(&registration), kAicpuKernelInnerError);
    resident_args = original;
    for (int fault : {0, 1}) {
        corrupt_kernel_arch_argument(resident_args, fault);
        EXPECT_EQ(simpler_aicpu_prepare_tmr_context(&registration), kAicpuKernelInnerError);
        resident_args = original;
    }
    resident_args.enable_profiling_flag = 1;
    EXPECT_EQ(simpler_aicpu_prepare_tmr_context(&registration), kAicpuKernelInnerError);
    resident_args = original;
    resident->dev.serial_orch_sched = !resident->dev.serial_orch_sched;
    EXPECT_EQ(simpler_aicpu_prepare_tmr_context(&registration), kAicpuKernelInnerError);
    resident->dev.serial_orch_sched = !resident->dev.serial_orch_sched;
    const int shards = resident->dev.ready_queue_shards;
    resident->dev.ready_queue_shards = shards == 1 ? 2 : 1;
    EXPECT_EQ(simpler_aicpu_prepare_tmr_context(&registration), kAicpuKernelInnerError);
    resident->dev.ready_queue_shards = shards;
    EXPECT_EQ(simpler_aicpu_prepare_tmr_context(&registration), 0);
    EXPECT_EQ(register_publications.load(), 1);

    const auto resident_image = residencies[7];
    TmrCallableRegistrationArgs callable{13, reinterpret_cast<uint64_t>(&residencies[7]), 7, 0, 17};
    for (uint64_t bytes :
         {uint64_t{0}, uint64_t{sizeof(ChipCallable) - 1}, uint64_t{kKernelCallableByteLimit + 1}, UINT64_MAX}) {
        residencies[7].bytes = bytes;
        EXPECT_EQ(simpler_aicpu_register_tmr_kernel_callable(&callable), kAicpuKernelInnerError);
    }
    residencies[7] = resident_image;
    ++residencies[7].device_address;
    EXPECT_EQ(simpler_aicpu_register_tmr_kernel_callable(&callable), kAicpuKernelInnerError);
    residencies[7] = resident_image;
    EXPECT_EQ(simpler_aicpu_register_tmr_kernel_callable(&callable), 0);

    // An unregistered slot must fail before image cache maintenance, not just
    // because its metadata differs from an already registered identity.
    residencies[8] = resident_image;
    residencies[8].callable_id = 8;
    callable = {13, reinterpret_cast<uint64_t>(&residencies[8]), 8, 0, 17};
    watched_image = reinterpret_cast<const void *>(resident_image.device_address);
    image_invalidations = 0;
    for (uint64_t bytes :
         {uint64_t{0}, uint64_t{sizeof(ChipCallable) - 1}, uint64_t{kKernelCallableByteLimit + 1}, UINT64_MAX}) {
        residencies[8].bytes = bytes;
        EXPECT_EQ(simpler_aicpu_register_tmr_kernel_callable(&callable), kAicpuKernelInnerError);
        EXPECT_EQ(image_invalidations.load(), 0);
    }
    residencies[8].bytes = resident_image.bytes;
    ++residencies[8].device_address;
    watched_image = reinterpret_cast<const void *>(residencies[8].device_address);
    EXPECT_EQ(simpler_aicpu_register_tmr_kernel_callable(&callable), kAicpuKernelInnerError);
    EXPECT_EQ(image_invalidations.load(), 0);
    residencies[8].device_address = resident_image.device_address;
    watched_image = reinterpret_cast<const void *>(resident_image.device_address);
    EXPECT_EQ(simpler_aicpu_register_tmr_kernel_callable(&callable), 0);
    EXPECT_EQ(image_invalidations.load(), 1);
    watched_image = nullptr;
}

TEST_F(TmrExecutorExecutionInputsTest, CoordinatedAdmissionAndConfigFailuresCancelUnopenedCoresAndReuse) {
    register_orchestration(5, "orchestration_a", "config_mismatch");
    for (int32_t fault : {1, 2, 3, 0}) {
        output.fill(0);
        PreparedInvocationView callable{fault == 2 ? 5 : 3, 1, 1, 17};
        TmrEncodingCandidate packet;
        TmrEncodingCache cache;
        auto args = arguments(33);
        ASSERT_EQ(encode_tmr_invocation(args, callable, binding.identity, cache, &packet), InvocationStatus::Ok);
        if (fault == 3) resident->dev.aicpu_thread_num = -1;
        const auto results = coordinated_round({callable, {}}, packet.packet(), fault == 1 ? -19 : 0);
        resident->dev.aicpu_thread_num = 2;
        const int32_t expected = fault == 0 ? 0 : (fault == 1 ? -19 : (fault == 2 ? 1 : 5));
        for (int32_t result : results)
            EXPECT_EQ(result, expected);
        EXPECT_EQ(opened_windows.load(), fault == 0 ? 3 : 0);
        EXPECT_EQ(closed_windows.load(), fault == 0 ? 3 : 0);
        EXPECT_EQ(output[3], fault == 0 ? 3u : 0u);
        EXPECT_EQ(kernel_execution_status(), -1);
    }
}

TEST_F(TmrExecutorExecutionInputsTest, CoordinatedMultipleSchedulersShareOnePublishedRuntimeAcrossRounds) {
    for (int32_t execution_threads : {3, 4}) {
        SCOPED_TRACE(execution_threads);
        resident->dev.aicpu_thread_num = execution_threads;
        for (bool serial : {false, true}) {
            SCOPED_TRACE(serial);
            resident->dev.serial_orch_sched = serial;
            for (int32_t id : {3, 4, 3}) {
                output.fill(0);
                PreparedInvocationView callable{id, 1, 1, 17};
                TmrEncodingCandidate packet;
                TmrEncodingCache cache;
                auto args = arguments(54);
                ASSERT_EQ(
                    encode_tmr_invocation(args, callable, binding.identity, cache, &packet), InvocationStatus::Ok
                );
                const auto results = coordinated_round({callable, {}}, packet.packet(), 0, execution_threads);
                ASSERT_EQ(results.size(), static_cast<size_t>(execution_threads + 1));
                for (int32_t result : results)
                    EXPECT_EQ(result, 0);
                EXPECT_EQ(output[0], static_cast<uint64_t>(id));
                EXPECT_EQ(output[3], static_cast<uint64_t>(id));
                EXPECT_EQ(output[1], 54u);
                EXPECT_EQ(output[4], 54u);
                EXPECT_EQ(output[2], output[5]);
                EXPECT_NE(output[2], 0u);
                EXPECT_EQ(opened_windows.load(), 3);
                EXPECT_EQ(closed_windows.load(), 3);
                EXPECT_EQ(control.runtime_status, 0);
                EXPECT_EQ(kernel_execution_status(), -1);
                EXPECT_EQ(resident->dev.aicpu_thread_num, execution_threads);
            }
        }
    }
}

TEST_F(TmrExecutorExecutionInputsTest, AdmittedSnapshotOwnsTransportAndBusyRejectionKeepsInputs) {
    auto state = std::make_unique<KernelInvocationState>();
    uint64_t invocation = 40;
    const EntryArgsStorage *storage = nullptr;
    for (int id : {3, 4, 3}) {
        SCOPED_TRACE(id);
        PreparedInvocationView callable{id, 1, 1, 17};
        {
            TmrEncodingCandidate packet;
            TmrEncodingCache cache;
            auto args = arguments(++invocation);
            ASSERT_EQ(encode_tmr_invocation(args, callable, binding.identity, cache, &packet), InvocationStatus::Ok);
            const auto encoded = packet.packet();
            std::vector<uint8_t> transport(encoded.data, encoded.data + encoded.size);
            ASSERT_EQ(
                state->admit({transport.data(), transport.size()}, {callable, {}}, binding), InvocationStatus::Ok
            );
            // The snapshot owns descriptors and scalars, not the Tensor storage.
            std::memset(transport.data(), 0, transport.size());
        }
        if (storage != nullptr) EXPECT_EQ(state->inputs().args, storage);
        storage = state->inputs().args;
        ChipTaskArgs converted;
        ASSERT_TRUE(configure_orchestration_args(state->inputs(), converted, nullptr));
        const auto *tensor = &converted.tensor(0).ref();
        EXPECT_EQ(tensor, &storage->tensor(0));
        EXPECT_EQ(converted.scalar(0), invocation);
        EXPECT_EQ(tensor->buffer.addr, reinterpret_cast<uint64_t>(output.data()));
        EXPECT_EQ(state->inputs().callable_id, id);
        EXPECT_EQ(state->admit({}, {callable, {}}, binding), InvocationStatus::InvalidArgument);
        EXPECT_EQ(state->inputs().args, storage);
        EXPECT_EQ(&converted.tensor(0).ref(), tensor);
        EXPECT_EQ(converted.scalar(0), invocation);
        converted.reset();
        state->clear();
        EXPECT_FALSE(state->active());
        EXPECT_EQ(state->inputs().args, nullptr);
        expect_resident_configuration(false);
    }
}

TEST_F(TmrExecutorExecutionInputsTest, InvalidExecutionConfigurationCancelsAndNextRoundReusesExecutor) {
    for (bool serial : {false, true}) {
        SCOPED_TRACE(serial);
        output.fill(0);
        resident->dev.serial_orch_sched = serial;
        resident->dev.aicpu_thread_num = -1;
        PreparedInvocationView callable{3, 1, 1, 17};
        TmrEncodingCandidate packet;
        TmrEncodingCache cache;
        auto args = arguments(9);
        ASSERT_EQ(encode_tmr_invocation(args, callable, binding.identity, cache, &packet), InvocationStatus::Ok);
        const auto results = coordinated_round({callable, {}}, packet.packet());
        for (int32_t result : results)
            EXPECT_EQ(result, static_cast<int32_t>(KernelDispatchStatus::InvalidBinding));
        EXPECT_EQ(control.runtime_status, static_cast<int32_t>(KernelDispatchStatus::InvalidBinding));
        EXPECT_EQ(opened_windows.load(), 0);
        EXPECT_EQ(closed_windows.load(), 0);
        EXPECT_EQ(output[0], 0u);
        EXPECT_EQ(output[3], 0u);
        EXPECT_EQ(kernel_execution_status(), -1);
        resident->dev.aicpu_thread_num = 2;
        ASSERT_NO_FATAL_FAILURE(expect_successful_reuse());
    }
}

TEST_F(TmrExecutorExecutionInputsTest, ExpectedCountMismatchCancelsAndNextRoundReusesExecutor) {
    register_orchestration(5, "orchestration_a", "config_mismatch");
    for (bool serial : {false, true}) {
        SCOPED_TRACE(serial);
        resident->dev.serial_orch_sched = serial;
        output.fill(0);
        PreparedInvocationView callable{5, 1, 1, 17};
        TmrEncodingCandidate packet;
        TmrEncodingCache cache;
        auto args = arguments(9);
        ASSERT_EQ(encode_tmr_invocation(args, callable, binding.identity, cache, &packet), InvocationStatus::Ok);
        const auto results = coordinated_round({callable, {}}, packet.packet());
        for (int32_t result : results)
            EXPECT_EQ(result, static_cast<int32_t>(KernelDispatchStatus::InvalidArgs));
        EXPECT_EQ(control.runtime_status, static_cast<int32_t>(KernelDispatchStatus::InvalidArgs));
        EXPECT_EQ(opened_windows.load(), 0);
        EXPECT_EQ(closed_windows.load(), 0);
        EXPECT_EQ(output[0], 5u);
        EXPECT_EQ(output[1], 9u);
        EXPECT_NE(output[2], 0u);
        EXPECT_EQ(output[3], 0u);
        EXPECT_EQ(output[4], 0u);
        EXPECT_EQ(output[5], 0u);
        expect_resident_configuration(serial);
        EXPECT_EQ(kernel_execution_status(), -1);
        ASSERT_NO_FATAL_FAILURE(expect_successful_reuse());
    }
}

TEST_F(TmrExecutorExecutionInputsTest, RuntimeErrorIsPublishedBeforeClearAndNextRoundReusesExecutor) {
    register_orchestration(6, "orchestration_error", "config_a");
    for (bool serial : {false, true}) {
        SCOPED_TRACE(serial);
        output.fill(0);
        resident->dev.serial_orch_sched = serial;
        PreparedInvocationView callable{6, 1, 1, 17};
        TmrEncodingCandidate packet;
        TmrEncodingCache cache;
        auto args = arguments(57);
        ASSERT_EQ(encode_tmr_invocation(args, callable, binding.identity, cache, &packet), InvocationStatus::Ok);
        const auto results = coordinated_round({callable, {}}, packet.packet());
        for (int32_t result : results)
            EXPECT_EQ(result, -SIMPLER_ERROR_INVALID_ARGS);
        EXPECT_EQ(control.runtime_status, -SIMPLER_ERROR_INVALID_ARGS);
        EXPECT_EQ(control.cleanup_status, 0);
        EXPECT_EQ(opened_windows.load(), 3);
        EXPECT_EQ(closed_windows.load(), 3);
        EXPECT_EQ(output[0], 3u);
        EXPECT_EQ(output[3], 3u);
        EXPECT_EQ(output[1], 57u);
        EXPECT_EQ(output[4], 57u);
        EXPECT_NE(output[2], 0u);
        EXPECT_EQ(output[2], output[5]);
        EXPECT_EQ(kernel_execution_status(), -1);
        expect_resident_configuration(serial);
        ASSERT_NO_FATAL_FAILURE(expect_successful_reuse());
    }
}

TEST_F(TmrExecutorExecutionInputsTest, RejectedAdmissionLeavesActualExecutorInactive) {
    PreparedInvocationView callable{3, 1, 1, 17};
    TmrEncodingCandidate packet;
    TmrEncodingCache cache;
    auto args = arguments(9);
    ASSERT_EQ(encode_tmr_invocation(args, callable, binding.identity, cache, &packet), InvocationStatus::Ok);
    for (bool stale : {false, true}) {
        SCOPED_TRACE(stale);
        output.fill(0);
        auto trusted_callable = callable;
        const size_t capacity = binding.arena.capacity;
        if (stale) ++trusted_callable.slot_generation;
        else binding.arena.capacity = 0;
        const auto results = coordinated_round({trusted_callable, {}}, packet.packet());
        binding.arena.capacity = capacity;
        const auto expected = stale ? KernelDispatchStatus::Stale : KernelDispatchStatus::InvalidBinding;
        for (int32_t result : results)
            EXPECT_EQ(result, static_cast<int32_t>(expected));
        EXPECT_EQ(control.runtime_status, static_cast<int32_t>(expected));
        EXPECT_EQ(opened_windows.load(), 0);
        EXPECT_EQ(closed_windows.load(), 0);
        EXPECT_EQ(output[0], 0u);
        EXPECT_EQ(output[3], 0u);
        EXPECT_EQ(kernel_execution_status(), -1);
        ASSERT_NO_FATAL_FAILURE(expect_successful_reuse());
    }
}

}  // namespace
