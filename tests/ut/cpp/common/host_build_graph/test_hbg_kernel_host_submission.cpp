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
// CPU qualification of the production HBG host submission and owner boundary.
// Callable resolution and opaque DSO handles are seeded. RTS/device execution,
// capture and real DSO loading are outside this suite.
#include <gtest/gtest.h>
#include "device_runner_base.h"
#include "kernel_platform_ops.h"
#include "host_build_graph/host/orchestrator.h"
#include "host_build_graph/host/orch_ops.h"
#include "host_build_graph/kernel_graph_packet_parser.h"
#include "transport.h"
#include "common/launch_entry_args.h"
#include "host_build_graph/kernel_callable_registration.h"
#include <cstring>

namespace {
using hbg_host_transport::control;
OrchContext *orchestration = nullptr;
int orchestration_calls = 0;
void bind(OrchContext *context) { orchestration = context; }
void entry(const ChipTaskArgs &args) {
    ++orchestration_calls;
    CoreTaskArgs core_args;
    core_args.add_scalar(args.scalar(0));
    MixedKernels kernels{};
    kernels.aiv0_kernel_id = 0;
    ASSERT_TRUE(orchestration->orchestrator->submit_task(kernels, core_args).task_id().is_valid());
}
struct EntryPoints {
    void (*entry)(const ChipTaskArgs &);
    void (*bind)(OrchContext *);
    void (*finish)();
} entry_points{entry, bind, nullptr};
uint8_t orchestration_module_token = 0;

// Device initialization and loading an orchestration DSO are outside this CPU
// boundary. The production runner consumes the resolved callable and owns every
// subsequent resource, host graph, packet, enqueue and close transition.
class Runner : public DeviceRunnerBase {
public:
    Runner() {
        device_id_ = 0;
        max_block_dim_ = block_dim_ = 1;
        kernel_runtime_.set_worker_count(3);
        kernel_runtime_.set_aicpu_launch_count(1);
        EXPECT_EQ(execution_mode_latch().latch(SIMPLER_MODE_KERNEL), 0);
        EXPECT_EQ(kernel_exec_state_.initialize(0, make_onboard_kernel_context_ops(), 23), 0);
        CallableState callable;
        callable.hash = 1;
        callable.chip_buffer_hash = 1;
        callable.aicore_image_hash = 2;
        callable.signature = {ArgDirection::SCALAR};
        callable.host_orch_func_ptr = &entry_points;
        // The loaded DSO's opaque handle is supplied by the registration
        // boundary. Its close is observed separately from the static test entry.
        callable.host_dlopen_handle = &orchestration_module_token;
        EXPECT_NE(callable.host_dlopen_handle, callable.host_orch_func_ptr);
        callables_[1] = std::move(callable);
        ChipCallableBuffer image;
        image.chip_dev = reinterpret_cast<uint64_t>(mem_alloc_.alloc(64));
        image.total_size = 64;
        image.refcount = 1;
        chip_callable_buffers_[1] = std::move(image);
    }
    int prepare() {
        CallConfig config{};
        config.runtime_env.ring_task_window[0] = 8;
        config.runtime_env.ring_heap[0] = 65536;
        int rc = prepare_hbg_kernel_runtime(config, 23, 29);
        return rc == 0 ? prepare_hbg_kernel_callable(1) : rc;
    }
    int launch(uint64_t scalar, void *caller = reinterpret_cast<void *>(uintptr_t{33})) {
        ChipStorageTaskArgs args;
        args.add_scalar(scalar);
        control().launching = true;
        const int rc = simpler_kernel_mode_launch(this, 1, &args, caller);
        control().launching = false;
        return rc;
    }
    size_t retained_images() const { return chip_callable_buffers_.size(); }
    bool can_accept_run() const override { return true; }
    int fill_persistent_arch_fields(KernelArgs *, uint64_t) override { return 0; }
    int prepare_execution(
        Runtime &, const CallConfig &, uint32_t, const NativeRunIdentity &, std::unique_ptr<PreparedExecution> *
    ) override {
        return PTO_RUNTIME_ERR_UNSUPPORTED;
    }
    void abandon_prepared_execution(PreparedExecution &) noexcept override {}
    int poll_execution(const ActiveExecution &) override { return PTO_RUNTIME_ERR_UNSUPPORTED; }
    DrainOutcome drain_execution(ActiveExecution &) override { return {}; }
    LaunchTransactionResult launch_execution_impl(PreparedExecution &, LaunchPermit) override { return {}; }
    int finalize_impl() override { return finalize_common(); }
};

class HbgHostSubmission : public ::testing::Test {
protected:
    void SetUp() override {
        control() = {};
        orchestration_calls = 0;
    }
    void TearDown() override {
        EXPECT_EQ(control().allocated_bytes, 0u);
        EXPECT_EQ(control().forbidden_launch_calls, 0u);
    }
    static hbg::ParsedGraphPacket parse(const hbg_host_transport::Launch &launch) {
        KernelArgs args{};
        EXPECT_GE(launch.bytes.size(), LAUNCH_ENVELOPE_HEADER_BYTES);
        if (launch.bytes.size() < LAUNCH_ENVELOPE_HEADER_BYTES) return {};
        std::memcpy(&args, launch.bytes.data(), sizeof(args));
        EXPECT_EQ(args.entry_args_source, static_cast<uint32_t>(EntryArgsSource::HbgKernelInvocation));
        EXPECT_EQ(args.graph_section_source, static_cast<uint32_t>(GraphSectionSource::RestoredSlot));
        EXPECT_EQ(args.graph_section_offset, LAUNCH_ENVELOPE_HEADER_BYTES);
        EXPECT_EQ(args.entry_args_offset, LAUNCH_ENVELOPE_HEADER_BYTES);
        EXPECT_EQ(args.graph_section_bytes, launch.bytes.size() - LAUNCH_ENVELOPE_HEADER_BYTES);
        hbg::ParsedGraphPacket packet;
        EXPECT_EQ(
            hbg::parse_graph_packet(
                launch.bytes.data() + LAUNCH_ENVELOPE_HEADER_BYTES, launch.bytes.size() - LAUNCH_ENVELOPE_HEADER_BYTES,
                hbg::GraphPacketAddress::RelativeLaunch, packet
            ),
            hbg::GraphPacketStatus::Ok
        );
        return packet;
    }
    static uint64_t scalar(const hbg::ParsedGraphPacket &packet) {
        const auto &image = packet.regions[static_cast<size_t>(hbg::GraphImageKind::Runtime)];
        const size_t sm_offset = packet.graph.sm_offset - image.destination_offset;
        const size_t slot_offset = sm_offset + sm_layout::segment_offsets(1).storage;
        EXPECT_LE(slot_offset + sizeof(ChipTaskStorage), image.bytes);
        if (slot_offset + sizeof(ChipTaskStorage) > image.bytes) return 0;
        alignas(64) ChipTaskStorage task;
        std::memcpy(&task, packet.payload + image.source_offset + slot_offset, sizeof(task));
        EXPECT_EQ(task.payload.scalar_count, 1);
        const uintptr_t relative =
            reinterpret_cast<uintptr_t>(task.payload.scalar_data()) - reinterpret_cast<uintptr_t>(&task);
        EXPECT_LE(relative + sizeof(uint64_t), image.bytes - slot_offset);
        if (relative > image.bytes - slot_offset - sizeof(uint64_t)) return 0;
        uint64_t value = 0;
        std::memcpy(&value, packet.payload + image.source_offset + slot_offset + relative, sizeof(value));
        return value;
    }
    static void expect_sequence(Runner &runner, void *caller, size_t begin = 0) {
        auto &state = runner.kernel_execution_state();
        void *cpu = state.hidden_stream(KernelStreamKind::Aicpu);
        void *core = state.hidden_stream(KernelStreamKind::Aicore);
        void *start = state.event(KernelEventKind::Start);
        void *core_start = state.event(KernelEventKind::AicoreStart);
        void *core_done = state.event(KernelEventKind::AicoreDone);
        void *cpu_done = state.event(KernelEventKind::AicpuDone);
        void *tail = state.event(KernelEventKind::SerialTail);
        const std::vector<hbg_host_transport::Operation> expected{
            {"record", caller, start},   {"wait", cpu, start},
            {"clear", cpu, nullptr},     {"record", cpu, core_start},
            {"wait", core, core_start},  {"core", core, nullptr},
            {"record", core, core_done}, {"simpler_aicpu_exec", cpu, nullptr},
            {"wait", cpu, core_done},    {"record", cpu, cpu_done},
            {"wait", caller, cpu_done},  {"record", caller, tail}
        };
        ASSERT_GE(control().operations.size(), begin + expected.size());
        for (size_t i = 0; i < expected.size(); ++i) {
            SCOPED_TRACE(i);
            const auto &actual = control().operations[begin + i];
            EXPECT_EQ(actual.name, expected[i].name);
            EXPECT_EQ(actual.stream, expected[i].stream);
            EXPECT_EQ(actual.event, expected[i].event);
        }
    }
};

TEST_F(HbgHostSubmission, FreshPreparationSubmitsTwoIndependentHostPackets) {
    Runner runner;
    ASSERT_EQ(runner.prepare(), 0);
    hbg_host_transport::reset_trace();
    ASSERT_EQ(runner.launch(71), 0);
    ASSERT_EQ(runner.launch(97), 0);
    EXPECT_EQ(orchestration_calls, 2);
    ASSERT_EQ(control().launches.size(), 4u);
    const auto first = parse(control().launches[1]);
    const auto second = parse(control().launches[3]);
    EXPECT_EQ(first.invocation.callable_id, 1);
    EXPECT_EQ(first.invocation.scalar_count, 1);
    EXPECT_EQ(first.graph.total_tasks, 1u);
    EXPECT_EQ(first.graph.slot_generation, 23u);
    EXPECT_EQ(first.graph.runtime_binary_id, 29u);
    EXPECT_EQ(first.graph.function_hash, 2u);
    EXPECT_NE(first.graph.argument_hash, second.graph.argument_hash);
    EXPECT_EQ(scalar(first), 71u);
    EXPECT_EQ(scalar(second), 97u);
    KernelArgs cpu_args{};
    AicoreLaunchArgs core_args{};
    std::memcpy(&cpu_args, control().launches[1].bytes.data(), sizeof(cpu_args));
    std::memcpy(&core_args, control().launches[0].bytes.data(), sizeof(core_args));
    EXPECT_EQ(core_args.runtime_args, reinterpret_cast<uint64_t>(cpu_args.runtime_args));
    EXPECT_TRUE(hbg_host_transport::live_allocation(cpu_args.runtime_args, sizeof(DeviceRuntimeLaunchDesc)));
    EXPECT_EQ(first.graph.destinations[0].address, second.graph.destinations[0].address);
    EXPECT_EQ(first.graph.destinations[1].address, second.graph.destinations[1].address);
    expect_sequence(runner, reinterpret_cast<void *>(uintptr_t{33}));
    expect_sequence(runner, reinterpret_cast<void *>(uintptr_t{33}), 12);
    EXPECT_EQ(control().module_registrations, 1u);
    EXPECT_TRUE(control().synchronized.empty());
    EXPECT_EQ(runner.finalize(), 0);
}

TEST_F(HbgHostSubmission, CleanRefusalCanRetryButPartialEnqueuePoisonsAdmission) {
    Runner runner;
    ASSERT_EQ(runner.prepare(), 0);
    hbg_host_transport::reset_trace();
    ChipStorageTaskArgs wrong;
    EXPECT_EQ(
        simpler_kernel_mode_launch(&runner, 1, &wrong, reinterpret_cast<void *>(uintptr_t{33})),
        PTO_RUNTIME_ERR_INVALID_ARGUMENT
    );
    EXPECT_TRUE(control().operations.empty());
    EXPECT_EQ(orchestration_calls, 0);
    ASSERT_EQ(runner.launch(71), 0);
    control().fail_step = control().step + 6;
    EXPECT_EQ(runner.launch(97), -4242);
    EXPECT_EQ(runner.kernel_execution_state().phase(), KernelContextPhase::Poisoned);
    const auto calls = orchestration_calls;
    const auto steps = control().step;
    EXPECT_EQ(runner.launch(103), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_EQ(orchestration_calls, calls);
    EXPECT_EQ(control().step, steps);
    EXPECT_EQ(runner.finalize(), 0);
}
}  // namespace

namespace {
TEST_F(HbgHostSubmission, EveryEnqueueRefusalRetainsOwnersUntilCloseCanDrain) {
    for (size_t failure = 1; failure <= 12; ++failure) {
        SCOPED_TRACE(failure);
        Runner runner;
        ASSERT_EQ(runner.prepare(), 0);
        hbg_host_transport::reset_trace();
        const auto retained = control().allocated_bytes;
        control().fail_step = failure;
        EXPECT_EQ(runner.launch(71), -4242);
        EXPECT_EQ(control().step, failure);
        EXPECT_EQ(runner.kernel_execution_state().phase(), KernelContextPhase::Poisoned);
        EXPECT_EQ(control().allocated_bytes, retained);
        EXPECT_EQ(runner.retained_images(), 1u);
        EXPECT_EQ(control().released_orchestration_modules.size(), failure - 1);
        const auto calls = orchestration_calls;
        EXPECT_EQ(runner.launch(97), PTO_RUNTIME_ERR_INVALID_STATE);
        EXPECT_EQ(control().step, failure);
        EXPECT_EQ(orchestration_calls, calls);
        control().fail_step = 0;
        control().sync_rc = -8181;
        EXPECT_EQ(runner.finalize(), -8181);
        EXPECT_EQ(control().allocated_bytes, retained);
        EXPECT_EQ(runner.retained_images(), 1u);
        EXPECT_EQ(control().released_orchestration_modules.size(), failure - 1);
        control().sync_rc = 0;
        EXPECT_EQ(runner.finalize(), 0);
        EXPECT_EQ(control().allocated_bytes, 0u);
        ASSERT_EQ(control().released_orchestration_modules.size(), failure);
        EXPECT_EQ(control().released_orchestration_modules.back(), &orchestration_module_token);
    }
}

TEST_F(HbgHostSubmission, CpuLaunchExceptionPoisonsAndRetainsTheSameOwners) {
    Runner runner;
    ASSERT_EQ(runner.prepare(), 0);
    hbg_host_transport::reset_trace();
    const auto retained = control().allocated_bytes;
    control().throw_cpu = true;
    EXPECT_EQ(runner.launch(71), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_EQ(runner.kernel_execution_state().phase(), KernelContextPhase::Poisoned);
    EXPECT_EQ(control().allocated_bytes, retained);
    control().throw_cpu = false;
    EXPECT_EQ(runner.launch(97), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_EQ(runner.finalize(), 0);
}

TEST_F(HbgHostSubmission, IncompleteCallerTailIsACleanRefusalAndCanRetry) {
    Runner runner;
    ASSERT_EQ(runner.prepare(), 0);
    hbg_host_transport::reset_trace();
    ASSERT_EQ(runner.launch(71), 0);
    control().complete = false;
    auto other = reinterpret_cast<void *>(uintptr_t{35});
    EXPECT_EQ(runner.launch(97, other), PTO_RUNTIME_ERR_PREPARED_INCOMPATIBLE);
    EXPECT_EQ(control().step, 12u);
    EXPECT_EQ(runner.kernel_execution_state().phase(), KernelContextPhase::ReadyEnqueued);
    control().query_rc = -9191;
    EXPECT_EQ(runner.launch(97, other), -9191);
    EXPECT_EQ(control().step, 12u);
    EXPECT_EQ(runner.kernel_execution_state().phase(), KernelContextPhase::ReadyEnqueued);
    control().query_rc = 0;
    control().complete = true;
    ASSERT_EQ(runner.launch(97, other), 0);
    expect_sequence(runner, other, 12);
    EXPECT_EQ(control().query_calls, 3u);
    EXPECT_EQ(runner.finalize(), 0);
}

TEST_F(HbgHostSubmission, GraphLeasePinsCallableResourcesAndFailedLastReleaseCanRetry) {
    Runner runner;
    ASSERT_EQ(runner.prepare(), 0);
    uint64_t first = 0, last = 0;
    ASSERT_EQ(runner.retain_hbg_kernel_graph(&first), 0);
    ASSERT_EQ(runner.retain_hbg_kernel_graph(&last), 0);
    ASSERT_NE(first, last);
    hbg_host_transport::reset_trace();
    ASSERT_EQ(runner.launch(71), 0);
    const auto retained = control().allocated_bytes;
    const auto calls = orchestration_calls;
    EXPECT_EQ(runner.launch(97, reinterpret_cast<void *>(uintptr_t{35})), PTO_RUNTIME_ERR_PREPARED_INCOMPATIBLE);
    EXPECT_EQ(orchestration_calls, calls);
    EXPECT_EQ(runner.revoke_hbg_kernel_callable(1), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_EQ(runner.finalize(), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_EQ(control().allocated_bytes, retained);
    EXPECT_TRUE(control().released_orchestration_modules.empty());
    EXPECT_EQ(runner.release_hbg_kernel_graph(first), 0);
    EXPECT_TRUE(control().synchronized.empty());
    control().sync_rc = -8181;
    EXPECT_EQ(runner.release_hbg_kernel_graph(last), -8181);
    EXPECT_EQ(runner.revoke_hbg_kernel_callable(1), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_EQ(control().allocated_bytes, retained);
    control().sync_rc = 0;
    EXPECT_TRUE(control().released_orchestration_modules.empty());
    ASSERT_EQ(runner.release_hbg_kernel_graph(last), 0);
    EXPECT_EQ(runner.release_hbg_kernel_graph(last), PTO_RUNTIME_ERR_INVALID_ARGUMENT);
    ASSERT_EQ(runner.launch(97, reinterpret_cast<void *>(uintptr_t{35})), 0);
    EXPECT_TRUE(control().released_orchestration_modules.empty());
    EXPECT_EQ(runner.finalize(), 0);
    EXPECT_EQ(control().released_orchestration_modules, std::vector<void *>{&orchestration_module_token});
}

TEST_F(HbgHostSubmission, NeverReplayedGraphStillDrainsItsOwnedStreamsBeforeRelease) {
    Runner runner;
    ASSERT_EQ(runner.prepare(), 0);
    uint64_t lease = 0;
    ASSERT_EQ(runner.retain_hbg_kernel_graph(&lease), 0);
    hbg_host_transport::reset_trace();
    ASSERT_EQ(runner.release_hbg_kernel_graph(lease), 0);
    EXPECT_EQ(
        control().synchronized, (std::vector<void *>{
                                    runner.kernel_execution_state().hidden_stream(KernelStreamKind::Aicpu),
                                    runner.kernel_execution_state().hidden_stream(KernelStreamKind::Aicore)
                                })
    );
    EXPECT_TRUE(control().operations.empty());
    EXPECT_EQ(orchestration_calls, 0);
    EXPECT_EQ(runner.finalize(), 0);
}
}  // namespace

namespace {
TEST_F(HbgHostSubmission, ControlPublicationFailureRetainsOnlyAcceptedRegistrations) {
    for (const bool callable : {false, true}) {
        for (const bool sync_failure : {false, true}) {
            SCOPED_TRACE(callable ? "callable" : "execution slot");
            SCOPED_TRACE(sync_failure ? "accepted then sync failed" : "enqueue refused");
            control() = {};
            orchestration_calls = 0;
            Runner runner;
            if (sync_failure) control().fail_sync = callable ? 2 : 1;
            else control().fail_step = callable ? 2 : 1;
            EXPECT_EQ(runner.prepare(), sync_failure ? -8282 : -4242);
            EXPECT_EQ(runner.kernel_execution_state().phase(), KernelContextPhase::Poisoned);
            const auto retained = control().allocated_bytes;
            EXPECT_GT(retained, 0u);
            EXPECT_EQ(runner.launch(71), PTO_RUNTIME_ERR_INVALID_STATE);
            EXPECT_EQ(orchestration_calls, 0);
            control().fail_step = control().fail_sync = 0;
            hbg_host_transport::reset_trace();
            EXPECT_EQ(runner.finalize(), 0);
            const bool slot_accepted = callable || sync_failure;
            const bool callable_accepted = callable && sync_failure;
            ASSERT_EQ(
                control().launches.size(), static_cast<size_t>(slot_accepted) + static_cast<size_t>(callable_accepted)
            );
            if (callable_accepted) EXPECT_EQ(control().launches.front().name, "simpler_aicpu_l1_hbg_revoke_callable");
            if (slot_accepted) EXPECT_EQ(control().launches.back().name, "simpler_aicpu_l1_hbg_detach_execution_slot");
            EXPECT_EQ(control().allocated_bytes, 0u);
        }
    }
}

TEST_F(HbgHostSubmission, TaskPayloadOwnsScalarSourceWhileRuntimeStaysRetained) {
    Runner runner;
    ASSERT_EQ(runner.prepare(), 0);
    uint64_t lease = 0;
    ASSERT_EQ(runner.retain_hbg_kernel_graph(&lease), 0);
    hbg_host_transport::reset_trace();
    ASSERT_EQ(runner.launch(71), 0);
    // Each transport task owns its blob, matching the RTS by-value boundary.
    // No second-level Host source is valid after launch has returned.
    const auto first = control().launches[1];
    KernelArgs header{};
    std::memcpy(&header, first.bytes.data(), sizeof(header));
    ASSERT_TRUE(hbg_host_transport::live_allocation(header.runtime_args, sizeof(DeviceRuntimeLaunchDesc)));
    ASSERT_EQ(runner.launch(97), 0);
    EXPECT_EQ(scalar(parse(first)), 71u);
    EXPECT_EQ(scalar(parse(control().launches[3])), 97u);
    EXPECT_EQ(runner.finalize(), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_TRUE(hbg_host_transport::live_allocation(header.runtime_args, sizeof(DeviceRuntimeLaunchDesc)));
    ASSERT_EQ(runner.release_hbg_kernel_graph(lease), 0);
    ASSERT_EQ(runner.finalize(), 0);
    EXPECT_FALSE(hbg_host_transport::live_allocation(header.runtime_args, sizeof(DeviceRuntimeLaunchDesc)));
    EXPECT_EQ(scalar(parse(first)), 71u);
}
}  // namespace

namespace {
TEST_F(HbgHostSubmission, EachInvocationClearsOnlyItsPersistentWorkerReports) {
    Runner runner;
    ASSERT_EQ(runner.prepare(), 0);
    ASSERT_EQ(control().launches.size(), 2u);
    simpler::kernel::HbgCallableBinding binding{};
    ASSERT_EQ(control().launches.back().bytes.size(), sizeof(binding));
    std::memcpy(&binding, control().launches.back().bytes.data(), sizeof(binding));
    auto *descriptor = reinterpret_cast<uint8_t *>(binding.runtime_address);
    ASSERT_TRUE(hbg_host_transport::live_allocation(descriptor, sizeof(DeviceRuntimeLaunchDesc)));
    constexpr size_t report_offset = offsetof(DeviceRuntimeLaunchDesc, workers);
    constexpr size_t report_bytes = 3 * sizeof(Handshake);
    for (uint64_t value : {71u, 97u}) {
        SCOPED_TRACE(value);
        std::memset(descriptor + report_offset, 0x5a, report_bytes);
        std::vector<uint8_t> expected(descriptor, descriptor + sizeof(DeviceRuntimeLaunchDesc));
        std::fill_n(expected.begin() + report_offset, report_bytes, 0);
        hbg_host_transport::reset_trace();
        ASSERT_EQ(runner.launch(value), 0);
        EXPECT_EQ(std::memcmp(descriptor, expected.data(), expected.size()), 0);
        ASSERT_EQ(control().launches.size(), 2u);
        EXPECT_EQ(scalar(parse(control().launches.back())), value);
    }
    EXPECT_EQ(runner.finalize(), 0);
}
}  // namespace
