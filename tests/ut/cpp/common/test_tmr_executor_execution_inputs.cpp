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
#include "aicpu/platform_aicpu_affinity.h"
#include "aicpu/platform_regs.h"
#include "common/kernel_args.h"
#include "host_log.h"
#include "tensormap_and_ringbuffer/kernel_execution_inputs.h"
#include "worker/tmr_kernel_invocation.h"

namespace {
thread_local int affinity_index = 0;
std::array<uint64_t, PLATFORM_MAX_CORES> register_bases{};
std::array<uint64_t, 3> register_cells{};
std::atomic<int> opened_windows{0};
}  // namespace

// The orchestration submits only dependency tasks, never AICore work. These
// platform fixtures supply ready-core reports and idle register windows.
int platform_aicpu_affinity_thread_idx() { return affinity_index; }
void platform_aicpu_affinity_set_thread_idx(int index) { affinity_index = index; }
extern "C" uint64_t get_platform_regs() { return reinterpret_cast<uint64_t>(register_bases.data()); }
extern "C" uint64_t get_platform_pmu_reg_addrs() { return 0; }
uint32_t platform_get_physical_cores_count() { return PLATFORM_MAX_CORES; }
volatile uint32_t *get_reg_ptr(uint64_t base, RegId) { return reinterpret_cast<volatile uint32_t *>(base); }
uint64_t read_reg(uint64_t base, RegId) { return *reinterpret_cast<uint64_t *>(base); }
uint32_t reg_load_acquire(const volatile uint32_t *ptr) { return *ptr; }
void reg_store_release(volatile uint32_t *, uint32_t) {}
void platform_init_aicore_regs(uint64_t) { ++opened_windows; }
uint64_t platform_aicore_exit_deadline() { return 0; }

extern "C" int simpler_aicpu_register_callable(void *);

namespace {
using namespace simpler::tmr;

class TmrExecutorExecutionInputsTest : public ::testing::Test {
protected:
    void SetUp() override {
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
        heap.resize(4096);
        auto *runtime = runtime_init_data_from_layout(
            arena, layout, MODE_EXECUTE, sm.region_ptr(sm_offset), sm_size, heap.data(), heaps
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

    void TearDown() override { release_kernel_execution(); }

    void register_orchestration(int id, const char *entry, const char *config) {
        RegisterCallableArgs args{};
        args.active_callable_id = id;
        args.dev_orch_so_addr = reinterpret_cast<uint64_t>(binary.data());
        args.dev_orch_so_size = binary.size();
        std::strcpy(args.device_orch_func_name, entry);
        std::strcpy(args.device_orch_config_name, config);
        ASSERT_EQ(simpler_aicpu_register_callable(&args), 0);
    }

    void report_ready_cores() {
        opened_windows = 0;
        for (int i = 0; i < 3; ++i) {
            auto &worker = resident->get_workers()[i];
            worker.physical_core_id = i;
            worker.core_type = i == 0 ? CoreType::AIC : CoreType::AIV;
            worker.aicore_done = i + 1;
        }
    }

    ChipStorageTaskArgs arguments(uint64_t value) {
        ChipStorageTaskArgs args;
        const uint32_t shape[] = {static_cast<uint32_t>(output.size() * 2)};
        args.add_tensor(make_tensor_external(output.data(), shape, 1, DataType::INT32, AddressSpace::DEVICE));
        args.add_scalar(value);
        return args;
    }

    std::array<int, 2> initialize_group() {
        std::array<int, 2> results{};
        std::thread scheduler([&] {
            affinity_index = 0;
            results[0] = init_kernel_execution();
        });
        std::thread orchestrator([&] {
            affinity_index = 1;
            results[1] = init_kernel_execution();
        });
        scheduler.join();
        orchestrator.join();
        return results;
    }

    void expect_successful_reuse() {
        EXPECT_EQ(kernel_execution_status(), -1);
        EXPECT_EQ(init_kernel_execution(), -1);
        EXPECT_EQ(run_kernel_execution(), -1);
        output.fill(0);
        PreparedInvocationView callable{3, 1, 1, 17};
        TmrEncodingCandidate packet;
        TmrEncodingCache cache;
        auto args = arguments(71);
        ASSERT_EQ(encode_tmr_invocation(args, callable, binding.identity, cache, &packet), InvocationStatus::Ok);
        ASSERT_EQ(admit_kernel_execution(packet.packet(), {callable, {}}, binding), InvocationStatus::Ok);
        report_ready_cores();
        const auto initialized = initialize_group();
        ASSERT_EQ(initialized[0], 0);
        ASSERT_EQ(initialized[1], 0);
        affinity_index = 1;
        ASSERT_EQ(run_kernel_execution(), 0);
        affinity_index = 0;
        ASSERT_EQ(run_kernel_execution(), 0);
        EXPECT_EQ(kernel_execution_status(), 0);
        EXPECT_EQ(output[0], 3u);
        EXPECT_EQ(output[1], 71u);
        EXPECT_EQ(output[3], 3u);
        EXPECT_EQ(output[4], 71u);
        EXPECT_NE(output[2], 0u);
        EXPECT_EQ(output[2], output[5]);
        expect_resident_configuration(resident->dev.serial_orch_sched);
        release_kernel_execution();
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
    std::vector<char> heap;
    std::vector<char> binary;
    std::array<uint64_t, 6> output{};
    uint64_t identity{0};
};

TEST_F(TmrExecutorExecutionInputsTest, ActualKernelPhasesKeepConfigAndOrchestrationInputsAcrossABA) {
    EXPECT_EQ(kernel_execution_status(), -1);
    EXPECT_EQ(init_kernel_execution(), -1);
    EXPECT_EQ(run_kernel_execution(), -1);
    uint64_t invocation = 40;
    uint64_t storage_address = 0;
    for (bool serial : {false, true}) {
        SCOPED_TRACE(serial);
        resident->dev.serial_orch_sched = serial;
        for (int id : {3, 4, 3}) {
            SCOPED_TRACE(id);
            output.fill(0);
            PreparedInvocationView callable{id, 1, 1, 17};
            TmrEncodingCandidate packet;
            TmrEncodingCache cache;
            auto args = arguments(++invocation);
            ASSERT_EQ(encode_tmr_invocation(args, callable, binding.identity, cache, &packet), InvocationStatus::Ok);
            const auto encoded = packet.packet();
            std::vector<uint8_t> transport(encoded.data, encoded.data + encoded.size);
            const ByteSpan submitted{transport.data(), transport.size()};
            ASSERT_EQ(admit_kernel_execution(submitted, {callable, {}}, binding), InvocationStatus::Ok);
            EXPECT_EQ(admit_kernel_execution(submitted, {callable, {}}, binding), InvocationStatus::InvalidArgument);
            // Admission owns the decoded arguments; the transport storage can die
            // before either execution phase consumes them.
            std::memset(transport.data(), 0, transport.size());
            transport = {};
            packet = {};
            args = {};
            report_ready_cores();
            std::array<int, 2> init_results{};
            std::thread scheduler([&] {
                affinity_index = 0;
                init_results[0] = init_kernel_execution();
            });
            std::thread orchestrator([&] {
                affinity_index = 1;
                init_results[1] = init_kernel_execution();
            });
            scheduler.join();
            orchestrator.join();
            ASSERT_EQ(init_results[0], 0);
            ASSERT_EQ(init_results[1], 0);
            affinity_index = 1;
            ASSERT_EQ(opened_windows.load(), 3);
            ASSERT_EQ(run_kernel_execution(), 0);
            affinity_index = 0;
            ASSERT_EQ(run_kernel_execution(), 0);
            EXPECT_EQ(kernel_execution_status(), 0);
            auto *header = static_cast<SharedMemoryHeader *>(binding.sm.base);
            header->sched_error_code.store(SIMPLER_ERROR_INVALID_ARGS);
            EXPECT_EQ(kernel_execution_status(), -SIMPLER_ERROR_INVALID_ARGS);
            header->sched_error_code.store(SIMPLER_ERROR_NONE);
            EXPECT_EQ(output[0], static_cast<uint64_t>(id));
            EXPECT_EQ(output[1], invocation);
            EXPECT_EQ(output[3], static_cast<uint64_t>(id));
            EXPECT_EQ(output[4], invocation);
            EXPECT_NE(output[2], 0u);
            EXPECT_EQ(output[2], output[5]);
            if (storage_address != 0) EXPECT_EQ(output[2], storage_address);
            storage_address = output[2];
            expect_resident_configuration(serial);
            release_kernel_execution();
            expect_resident_configuration(serial);
            EXPECT_EQ(kernel_execution_status(), -1);
            EXPECT_EQ(init_kernel_execution(), -1);
            EXPECT_EQ(run_kernel_execution(), -1);
        }
    }
}

TEST_F(TmrExecutorExecutionInputsTest, FailedInitializationCanReleaseAndReuseExecutor) {
    for (bool serial : {false, true}) {
        SCOPED_TRACE(serial);
        resident->dev.serial_orch_sched = serial;
        resident->dev.aicpu_thread_num = -1;
        PreparedInvocationView callable{3, 1, 1, 17};
        TmrEncodingCandidate packet;
        TmrEncodingCache cache;
        auto args = arguments(9);
        ASSERT_EQ(encode_tmr_invocation(args, callable, binding.identity, cache, &packet), InvocationStatus::Ok);
        ASSERT_EQ(admit_kernel_execution(packet.packet(), {callable, {}}, binding), InvocationStatus::Ok);
        opened_windows = 0;
        const auto initialized = initialize_group();
        EXPECT_EQ(initialized[0], -1);
        EXPECT_EQ(initialized[1], -1);
        EXPECT_EQ(opened_windows.load(), 0);
        release_kernel_execution();
        resident->dev.aicpu_thread_num = 2;
        ASSERT_NO_FATAL_FAILURE(expect_successful_reuse());
    }
}

TEST_F(TmrExecutorExecutionInputsTest, FailedConfigurationCanReleaseAndReuseExecutor) {
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
        ASSERT_EQ(admit_kernel_execution(packet.packet(), {callable, {}}, binding), InvocationStatus::Ok);
        report_ready_cores();
        const auto initialized = initialize_group();
        ASSERT_EQ(initialized[0], 0);
        ASSERT_EQ(initialized[1], 0);
        std::array<int, 2> results{};
        std::thread scheduler([&] {
            affinity_index = 0;
            results[0] = run_kernel_execution();
        });
        std::thread orchestrator([&] {
            affinity_index = 1;
            results[1] = run_kernel_execution();
        });
        scheduler.join();
        orchestrator.join();
        EXPECT_EQ(results[0], 0);
        EXPECT_EQ(results[1], -1);
        EXPECT_EQ(output[0], 5u);
        EXPECT_EQ(output[1], 9u);
        EXPECT_NE(output[2], 0u);
        EXPECT_EQ(output[3], 0u);
        EXPECT_EQ(output[4], 0u);
        EXPECT_EQ(output[5], 0u);
        expect_resident_configuration(serial);
        release_kernel_execution();
        ASSERT_NO_FATAL_FAILURE(expect_successful_reuse());
    }
}

TEST_F(TmrExecutorExecutionInputsTest, RejectedAdmissionLeavesActualExecutorInactive) {
    PreparedInvocationView callable{3, 1, 1, 17};
    TmrEncodingCandidate packet;
    TmrEncodingCache cache;
    auto args = arguments(9);
    ASSERT_EQ(encode_tmr_invocation(args, callable, binding.identity, cache, &packet), InvocationStatus::Ok);
    auto invalid_binding = binding;
    invalid_binding.arena.capacity = 0;
    EXPECT_EQ(
        admit_kernel_execution(packet.packet(), {callable, {}}, invalid_binding), InvocationStatus::InvalidBinding
    );
    EXPECT_EQ(init_kernel_execution(), -1);
    EXPECT_EQ(run_kernel_execution(), -1);
    EXPECT_EQ(kernel_execution_status(), -1);
    auto stale = callable;
    ++stale.slot_generation;
    EXPECT_EQ(admit_kernel_execution(packet.packet(), {stale, {}}, binding), InvocationStatus::StaleCallable);
    EXPECT_EQ(init_kernel_execution(), -1);
    EXPECT_EQ(run_kernel_execution(), -1);
    EXPECT_EQ(kernel_execution_status(), -1);
    ASSERT_EQ(admit_kernel_execution(packet.packet(), {callable, {}}, binding), InvocationStatus::Ok);
    EXPECT_EQ(kernel_execution_status(), 0);
    release_kernel_execution();
    EXPECT_EQ(kernel_execution_status(), -1);
}

}  // namespace
