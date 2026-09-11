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
#include <acl/acl.h>
#include <acl/error_codes/rt_error_codes.h>
#include <runtime/rt.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <iterator>
#include <memory>
#include <vector>

#include "../protocol.h"
#include "arch_prepare.h"
#include "host/acl_error_names.h"
#include "host/kernel_execution_state.h"
#include "kernel_launch_sequence.h"
#include "kernel_platform_ops.h"
#include "load_aicpu_op.h"
#include "runtime.h"
#include "task_interface/kernel_dispatch_args.h"
#include "task_interface/tmr_kernel_context.h"
#include "tensormap_and_ringbuffer/kernel_clear_plan.h"
#include "tensormap_and_ringbuffer/kernel_native_status.h"

namespace {
using control_probe::GateMode;
using control_probe::Mode;
using namespace simpler::tmr;
constexpr size_t kGuardBytes = 64;
enum class TerminalMode { EagerCpuFirst, EagerCallerFirst, Replay };

void check(int rc, const char *operation) {
    if (rc == 0) return;
    std::fprintf(stderr, "control_probe FAIL: %s rc=%d\n", operation, rc);
    std::fflush(stderr);
    // An incomplete device task can retain every buffer and binary in this process.
    std::_Exit(1);
}
void require(bool condition, const char *operation) { check(condition ? 0 : -1, operation); }

std::vector<char> read_binary(const char *path) {
    std::ifstream file(path, std::ios::binary);
    require(file.good(), path);
    std::vector<char> bytes{std::istreambuf_iterator<char>(file), {}};
    require(!bytes.empty(), "empty binary");
    return bytes;
}

struct DeviceBuffer {
    void *address;
    size_t bytes;

    DeviceBuffer(MemoryAllocator &allocator, size_t size) :
        address(allocator.alloc(size)),
        bytes(size) {
        require(address != nullptr, "allocate prepared buffer");
    }
    void upload(const void *source, size_t size) const {
        require(size <= bytes, "upload bound");
        check(aclrtMemcpy(address, bytes, source, size, ACL_MEMCPY_HOST_TO_DEVICE), "prepare H2D");
    }
    std::vector<uint8_t> read() const {
        std::vector<uint8_t> data(bytes);
        check(aclrtMemcpy(data.data(), bytes, address, bytes, ACL_MEMCPY_DEVICE_TO_HOST), "read prepared buffer");
        return data;
    }
    uint64_t integer() const { return reinterpret_cast<uint64_t>(address); }
};

struct PreparedBuffers {
    DeviceBuffer runtime;
    DeviceBuffer payloads;
    DeviceBuffer kernel_args;
    DeviceBuffer context;
    DeviceBuffer envelope;
    DeviceBuffer control_guarded;
    DeviceBuffer reports_guarded;
    DeviceBuffer result;
    KernelArgs arguments{};
    TmrKernelContextDescriptor descriptor{};
    TmrKernelClearBinding clear_binding{};
    TmrKernelClearPlan clear_plan{};
    std::vector<uint8_t> runtime_copy;

    PreparedBuffers(MemoryAllocator &allocator, int device) :
        runtime(allocator, sizeof(DeviceRuntimeLaunchDesc)),
        payloads(allocator, 2 * control_probe::kWorkers * sizeof(DispatchPayload)),
        kernel_args(allocator, sizeof(KernelArgs)),
        context(allocator, sizeof(TmrKernelContextDescriptor)),
        envelope(allocator, sizeof(TmrKernelAicoreArgs)),
        control_guarded(allocator, 2 * kGuardBytes + sizeof(TmrLaunchControl)),
        reports_guarded(allocator, 2 * kGuardBytes + control_probe::kWorkers * sizeof(TmrCoreReport)),
        result(allocator, sizeof(control_probe::Result)) {
        auto image = std::make_unique<DeviceRuntimeLaunchDesc>();
        image->worker_count = control_probe::kWorkers;
        image->aicpu_thread_num = 1;
        image->aicpu_launch_count = 1;
        image->active_callable_id_ = -1;
        for (int32_t i = 0; i < control_probe::kWorkers; ++i)
            image->workers[i].task = payloads.integer() + 2 * i * sizeof(DispatchPayload);
        runtime.upload(image.get(), sizeof(*image));
        check(aclrtMemset(payloads.address, payloads.bytes, 0, payloads.bytes), "prepare idle payloads");
        arguments.runtime_args = reinterpret_cast<Runtime *>(runtime.address);
        check(prepare_arch_fields(arguments, allocator, device), "prepare platform register mappings");
        kernel_args.upload(&arguments, sizeof(arguments));

        descriptor.version = kTmrKernelContextVersion;
        descriptor.bytes = sizeof(descriptor);
        descriptor.context_generation = control_probe::kContextGeneration;
        descriptor.self_address = context.integer();
        descriptor.resident_runtime = runtime.integer();
        descriptor.resident_kernel_args = kernel_args.integer();
        descriptor.control_address = control_guarded.integer() + kGuardBytes;
        descriptor.control_bytes = sizeof(TmrLaunchControl);
        descriptor.reports_address = reports_guarded.integer() + kGuardBytes;
        descriptor.reports_bytes = control_probe::kWorkers * sizeof(TmrCoreReport);
        descriptor.launch_threads = 1;
        descriptor.execution_threads = 1;
        descriptor.worker_count = control_probe::kWorkers;
        context.upload(&descriptor, sizeof(descriptor));
        const TmrKernelAicoreArgs core_args{kernel_args.integer(), context.integer()};
        envelope.upload(&core_args, sizeof(core_args));
        for (const auto *guarded : {&control_guarded, &reports_guarded})
            check(aclrtMemset(guarded->address, guarded->bytes, 0xa5, guarded->bytes), "prepare canaries");
        clear_binding = {
            descriptor.context_generation,
            {descriptor.control_address, descriptor.control_bytes},
            {descriptor.reports_address, descriptor.reports_bytes},
            descriptor.worker_count
        };
        require(build_tmr_kernel_clear_plan(clear_binding, &clear_plan), "build exact clear plan");
        auto whole_image = clear_plan;
        whole_image.regions[0] = {runtime.integer(), runtime.bytes};
        require(!validate_tmr_kernel_clear_plan(whole_image, clear_binding), "reject whole Runtime clear");
        auto enclosing_allocation = clear_plan;
        enclosing_allocation.regions[1] = {reports_guarded.integer(), reports_guarded.bytes};
        require(!validate_tmr_kernel_clear_plan(enclosing_allocation, clear_binding), "reject guard overlap");
        runtime_copy = runtime.read();
    }

    control_probe::Init init() const { return {context.integer(), arguments.regs, result.integer()}; }

    void verify_resident() const {
        for (const auto *guarded : {&control_guarded, &reports_guarded}) {
            const auto copy = guarded->read();
            require(
                std::all_of(
                    copy.begin(), copy.begin() + kGuardBytes,
                    [](uint8_t v) {
                        return v == 0xa5;
                    }
                ),
                "leading clear canary"
            );
            require(
                std::all_of(
                    copy.end() - kGuardBytes, copy.end(),
                    [](uint8_t v) {
                        return v == 0xa5;
                    }
                ),
                "trailing clear canary"
            );
        }
        require(runtime_copy == runtime.read(), "resident Runtime changed");
        const auto context_copy = context.read();
        require(std::memcmp(context_copy.data(), &descriptor, sizeof(descriptor)) == 0, "context descriptor changed");
        const auto args_copy = kernel_args.read();
        require(std::memcmp(args_copy.data(), &arguments, sizeof(arguments)) == 0, "resident KernelArgs changed");
    }

    void verify(Mode mode, uint64_t completed_rounds) const {
        verify_resident();
        control_probe::Result observed{};
        const auto result_copy = result.read();
        std::memcpy(&observed, result_copy.data(), sizeof(observed));
        require(observed.completed_rounds == completed_rounds, "one CPU round per native task");
        require(observed.seeded_registers != 0, "nonzero DMB seeded before core launch");
        const auto control_copy = control_guarded.read();
        TmrLaunchControl control{};
        std::memcpy(&control, control_copy.data() + kGuardBytes, sizeof(control));
        const bool cancelled_by_host = mode == Mode::HostCancel;
        if (cancelled_by_host) {
            require(control.host_cancel == kTmrHostCancel && control.completion == 0, "pre-window Host cancel");
        } else {
            const int32_t expected = mode == Mode::Reject           ? control_probe::kAdmissionRejected :
                                     mode == Mode::OmitControlClear ? -1 :
                                                                      0;
            require(
                control.host_cancel == 0 && control.completion == 1 && control.cleanup_status == 0 &&
                    control.runtime_status == expected && control.round_epoch == observed.epoch,
                "final control verdict"
            );
            require(
                observed.runtime_status == expected && observed.cleanup_status == 0 &&
                    observed.opened == (mode == Mode::Open ? control_probe::kWorkers : 0),
                "AICPU retirement result"
            );
        }
        const auto reports_copy = reports_guarded.read();
        for (int32_t i = 0; i < control_probe::kWorkers; ++i) {
            TmrCoreReport report{};
            std::memcpy(&report, reports_copy.data() + kGuardBytes + i * sizeof(report), sizeof(report));
            require(
                report.ready == static_cast<uint32_t>(i + 1) && report.exited == static_cast<uint32_t>(i + 1),
                "every launched core reports and exits"
            );
            require(
                report.command == (cancelled_by_host ? 0u : static_cast<uint32_t>(TmrCoreCommand::Cancel)),
                "final core command"
            );
            require(report.release == (mode == Mode::Open ? 1u : 0u), "release only after opened window closes");
        }
    }
};

struct GateBuffers {
    DeviceBuffer summary;
    DeviceBuffer threads;
    control_probe::GateInit prepared{};
    void *evidence_host{nullptr};
    size_t evidence_bytes{0};

    GateBuffers(const GateBuffers &) = delete;
    GateBuffers &operator=(const GateBuffers &) = delete;

    GateBuffers(MemoryAllocator &allocator, const PreparedBuffers &buffers, int device) :
        summary(allocator, sizeof(control_probe::GateSummary)),
        threads(allocator, PLATFORM_MAX_AICPU_THREADS_JUST_FOR_LAUNCH * sizeof(control_probe::GateThreadResult)) {
        prepared.descriptor = buffers.context.integer();
        prepared.registers = buffers.arguments.regs;
        prepared.summary = summary.integer();
        prepared.results = threads.integer();
        check(prepare_gate_topology(prepared, device), "prepare genuine gate topology");
        evidence_bytes =
            std::max({summary.bytes, threads.bytes, buffers.control_guarded.bytes, buffers.reports_guarded.bytes});
        check(aclrtMallocHost(&evidence_host, evidence_bytes), "prepare pinned terminal evidence buffer");
        require(reinterpret_cast<uintptr_t>(evidence_host) % 64 == 0, "pinned evidence alignment");
    }

    ~GateBuffers() {
        // Terminal native-error exits without unwinding this owner.
        if (evidence_host != nullptr) check(aclrtFreeHost(evidence_host), "release pinned evidence buffer");
    }

    void verify(const PreparedBuffers &buffers, GateMode mode, uint64_t round) const {
        buffers.verify_resident();
        const bool admission_rejected = mode == GateMode::AdmissionReject;
        const bool init_rejected = mode == GateMode::InitReject;
        const int32_t expected = admission_rejected ? control_probe::kAdmissionRejected :
                                 init_rejected      ? control_probe::kInitRejected :
                                                      0;
        const int32_t expected_init = admission_rejected ? 0 : control_probe::kExecutionThreads;
        const int32_t expected_run = mode == GateMode::Open ? control_probe::kExecutionThreads : 0;
        control_probe::GateSummary observed{};
        const auto summary_copy = summary.read();
        std::memcpy(&observed, summary_copy.data(), sizeof(observed));
        require(observed.epoch == round && observed.completed_rounds == round, "gate epoch advances without reset");
        require(
            observed.prepare_calls == (admission_rejected ? 0 : 1) && observed.init_calls == expected_init &&
                observed.verdict_calls == (mode == GateMode::Open ? 1 : 0) && observed.run_calls == expected_run,
            "only M initialize/run, and global init rejection prevents every run"
        );
        require(
            observed.finish_calls == 1 && observed.finalize_calls == 1 && observed.publish_calls == 1 &&
                observed.clear_calls == 1 && observed.runtime_status == expected && observed.cleanup_status == 0,
            "one CoreGroup finish, final publication and last-reader clear"
        );
        const auto thread_copy = threads.read();
        int32_t initializers = 0;
        int32_t runners = 0;
        uint32_t active_roles = 0;
        for (int32_t i = 0; i < prepared.launched_threads; ++i) {
            control_probe::GateThreadResult thread{};
            std::memcpy(&thread, thread_copy.data() + i * sizeof(thread), sizeof(thread));
            require(
                thread.epoch == round && thread.cpu >= 0 && thread.returned_status == expected &&
                    thread.runtime_status == expected && thread.cleanup_status == 0 && thread.native_status == 0,
                "every N native worker read the same final result and departed"
            );
            require(
                (thread.initialized == 0 || thread.initialized == 1) && (thread.ran == 0 || thread.ran == 1),
                "per-worker initialization/run flags"
            );
            initializers += thread.initialized;
            runners += thread.ran;
            if (thread.initialized != 0) {
                require(
                    thread.execution_index >= 0 && thread.execution_index < control_probe::kExecutionThreads,
                    "selected worker has a valid execution role"
                );
                const uint32_t role = uint32_t{1} << thread.execution_index;
                require((active_roles & role) == 0, "each execution role has exactly one native owner");
                active_roles |= role;
            } else {
                require(thread.execution_index == -1 && thread.ran == 0, "filtered worker does not initialize or run");
            }
        }
        require(
            initializers == expected_init && runners == expected_run && active_roles == (admission_rejected ? 0u : 3u),
            "N/M native participation counts"
        );
        const auto control_copy = buffers.control_guarded.read();
        TmrLaunchControl control{};
        std::memcpy(&control, control_copy.data() + kGuardBytes, sizeof(control));
        require(
            control.host_cancel == 0 && control.completion == 1 && control.round_epoch == round &&
                control.runtime_status == expected && control.cleanup_status == 0,
            "gate final verdict reached the real control region"
        );
        const auto reports_copy = buffers.reports_guarded.read();
        for (int32_t i = 0; i < control_probe::kWorkers; ++i) {
            TmrCoreReport report{};
            std::memcpy(&report, reports_copy.data() + kGuardBytes + i * sizeof(report), sizeof(report));
            require(
                report.ready == static_cast<uint32_t>(i + 1) && report.exited == static_cast<uint32_t>(i + 1) &&
                    report.command == static_cast<uint32_t>(TmrCoreCommand::Cancel) &&
                    report.release == (admission_rejected ? 0u : 1u),
                "gate CoreGroup retires each real AICore before reuse"
            );
        }
    }
};

struct Submission {
    PreparedBuffers &buffers;
    host::LoadAicpuOp &loader;
    void *core_binary;
    Mode mode;
    GateMode gate_mode{GateMode::Open};
    int32_t gate_threads{0};
    bool cpu_enqueued{false};

    simpler::kernel_launch::KernelLaunchOps ops() {
        return {
            this,
            [](void *, void *stream, void *event) noexcept {
                return aclrtStreamWaitEvent(stream, event);
            },
            [](void *opaque, void *stream) noexcept {
                const auto &self = *static_cast<Submission *>(opaque);
                if (!validate_tmr_kernel_clear_plan(self.buffers.clear_plan, self.buffers.clear_binding)) return -1;
                for (size_t i = 0; i < self.buffers.clear_plan.regions.size(); ++i) {
                    if (self.mode == Mode::OmitControlClear && i == 0) continue;
                    const auto &r = self.buffers.clear_plan.regions[i];
                    const int rc = aclrtMemsetAsync(reinterpret_cast<void *>(r.address), r.bytes, 0, r.bytes, stream);
                    if (rc != 0) return rc;
                }
                return 0;
            },
            [](void *, void *event, void *stream) noexcept {
                return aclrtRecordEvent(event, stream);
            },
            [](void *opaque, void *stream) noexcept {
                auto &self = *static_cast<Submission *>(opaque);
                if (self.mode == Mode::HostCancel) return control_probe::kEnqueueRejected;
                if (self.gate_threads > 0) {
                    control_probe::GateInvocation args{self.gate_mode, 0};
                    const int rc = self.loader.LaunchBuiltInOp(
                        stream, &args, sizeof(args), self.gate_threads, control_probe::kGateEntry
                    );
                    self.cpu_enqueued = rc == 0;
                    return rc;
                }
                control_probe::Invocation args{self.mode, 0};
                return self.loader.LaunchBuiltInOp(stream, &args, sizeof(args), 1, host::KernelNames::RunName);
            },
            [](void *opaque, void *stream) noexcept {
                const auto &self = *static_cast<Submission *>(opaque);
                uint64_t argument = self.buffers.envelope.integer();
                rtArgsEx_t args{};
                args.args = &argument;
                args.argsSize = sizeof(argument);
                rtTaskCfgInfo_t config{};
                config.schemMode = RT_SCHEM_MODE_BATCH;
                return rtKernelLaunchWithHandleV2(self.core_binary, 0, 1, &args, nullptr, stream, &config);
            },
            [](void *opaque, void *stream) noexcept {
                const auto &r = static_cast<Submission *>(opaque)->buffers.clear_plan.cancel;
                return aclrtMemsetAsync(reinterpret_cast<void *>(r.address), r.bytes, 0xff, r.bytes, stream);
            }
        };
    }
};

simpler::kernel_launch::KernelLaunchResult
submit(KernelExecutionState &state, Submission &submission, aclrtStream caller) {
    const simpler::kernel_launch::KernelLaunchHandles handles{
        caller,
        state.hidden_stream(KernelStreamKind::Aicpu),
        state.hidden_stream(KernelStreamKind::Aicore),
        state.event(KernelEventKind::AicoreStart),
        state.event(KernelEventKind::Start),
        state.event(KernelEventKind::AicoreDone),
        state.event(KernelEventKind::AicpuDone),
        state.event(KernelEventKind::SerialTail),
        false
    };
    return simpler::kernel_launch::enqueue_kernel_launch_sequence(submission.ops(), handles);
}

void enqueue(KernelExecutionState &state, Submission &submission, aclrtStream caller) {
    const auto result = submit(state, submission, caller);
    require(
        result.status == (submission.mode == Mode::HostCancel ? control_probe::kEnqueueRejected : 0),
        "native enqueue status"
    );
    require(result.cleanup_status == 0 && result.tail_recorded, "both hidden branches joined");
}

const char *error_name(int rc) {
    if (rc == 0) return "SUCCESS";
    if (rc == ACL_ERROR_RT_END_OF_SEQUENCE) return "ACL_ERROR_RT_END_OF_SEQUENCE";
    if (rc == ACL_ERROR_RT_MODEL_EXECUTE) return "ACL_ERROR_RT_MODEL_EXECUTE";
    const char *name = acl_error_name(rc);
    return name == nullptr ? "UNCLASSIFIED" : name;
}

bool read_terminal_evidence(
    const DeviceBuffer &buffer, const char *name, const GateBuffers &gate, aclrtStream stream, std::vector<uint8_t> &out
) {
    const int copy_rc = aclrtMemcpyAsync(
        gate.evidence_host, gate.evidence_bytes, buffer.address, buffer.bytes, ACL_MEMCPY_DEVICE_TO_HOST, stream
    );
    const int sync_rc = copy_rc == 0 ? aclrtSynchronizeStreamWithTimeout(stream, 1000) : 0;
    const int rc = copy_rc != 0 ? copy_rc : sync_rc;
    std::printf(
        "control_probe terminal evidence region=%s copy_rc=%d sync_rc=%d sync_attempted=%d "
        "name=%s readable=%d\n",
        name, copy_rc, sync_rc, copy_rc == 0, error_name(rc), rc == 0
    );
    if (rc != 0) return false;
    out.resize(buffer.bytes);
    std::memcpy(out.data(), gate.evidence_host, buffer.bytes);
    return true;
}

[[noreturn]] void terminal_native_error(
    KernelExecutionState &state, Submission &submission, aclrtStream caller, const GateBuffers &gate, uint64_t epoch,
    TerminalMode mode
) noexcept {
    const auto &buffers = submission.buffers;
    const bool replay = mode == TerminalMode::Replay;
    const bool caller_first = mode != TerminalMode::EagerCpuFirst;
    const char *mode_name = replay ? "replay" : caller_first ? "eager-caller-first" : "eager-cpu-first";
    std::printf(
        "control_probe terminal BEGIN epoch=%llu raw_status=%d dispatch_status=%d native_status=%d N=%d "
        "extra_barrier=0 reuse=0 "
        "mode=%s\n",
        static_cast<unsigned long long>(epoch), control_probe::kInitRejected,
        static_cast<int>(KernelDispatchStatus::ExecutionFailed), simpler::tmr::kAicpuKernelInnerError,
        gate.prepared.launched_threads, mode_name
    );
    std::fflush(stdout);
    submission.gate_mode = GateMode::TerminalInitReject;
    submission.cpu_enqueued = false;
    aclmdlRI terminal_graph = nullptr;
    if (replay) {
        const int capture_rc = aclmdlRICaptureBegin(caller, ACL_MODEL_RI_CAPTURE_MODE_GLOBAL);
        std::printf("control_probe terminal capture_begin rc=%d name=%s\n", capture_rc, error_name(capture_rc));
        std::fflush(stdout);
        check(capture_rc, "terminal capture begin");
    }
    const auto enqueued = submit(state, submission, caller);
    std::printf(
        "control_probe terminal enqueue status=%d cleanup=%d failed_step=%u cpu_enqueued=%d tail_recorded=%d "
        "phase=%s\n",
        enqueued.status, enqueued.cleanup_status, static_cast<unsigned>(enqueued.failed_step), submission.cpu_enqueued,
        enqueued.tail_recorded, replay ? "capture" : "eager"
    );
    std::fflush(stdout);
    int replay_rc = 0;
    if (replay) {
        require(
            enqueued.status == 0 && enqueued.cleanup_status == 0 && submission.cpu_enqueued && enqueued.tail_recorded,
            "complete terminal capture submission"
        );
        const int capture_rc = aclmdlRICaptureEnd(caller, &terminal_graph);
        std::printf(
            "control_probe terminal capture_end rc=%d name=%s graph_valid=%d\n", capture_rc, error_name(capture_rc),
            terminal_graph != nullptr
        );
        std::fflush(stdout);
        check(capture_rc, "terminal capture end");
        require(terminal_graph != nullptr, "terminal graph contains both hidden branches");
        // The terminal packet is captured without an eager warmup. This graph
        // and every referenced resource remain owned until process exit.
        replay_rc = aclmdlRIExecuteAsync(terminal_graph, caller);
        std::printf("control_probe terminal replay_enqueue rc=%d name=%s\n", replay_rc, error_name(replay_rc));
        std::fflush(stdout);
    }
    // An asynchronous error may already surface while enqueuing Done/join.
    // Once CPU enqueue succeeded, never try Host cancel or repair the streams.
    const auto synchronize_branch = [](aclrtStream stream, const char *name) {
        const int rc = aclrtSynchronizeStreamWithTimeout(stream, 10000);
        std::printf("control_probe terminal sync branch=%s rc=%d name=%s\n", name, rc, error_name(rc));
        std::fflush(stdout);
        return rc;
    };
    int caller_rc = 0;
    if (caller_first) caller_rc = synchronize_branch(caller, "caller");
    const int cpu_rc = synchronize_branch(state.hidden_stream(KernelStreamKind::Aicpu), "aicpu");
    const int core_rc = synchronize_branch(state.hidden_stream(KernelStreamKind::Aicore), "aicore");
    if (!caller_first) caller_rc = synchronize_branch(caller, "caller");

    bool evidence_valid = true;
    bool control_verified = false;
    bool cores_verified = false;
    int32_t observed_returns = -1;
    int32_t readable_regions = 0;
    std::vector<uint8_t> copy;
    const auto evidence_stream = state.hidden_stream(KernelStreamKind::Aicore);
    if (read_terminal_evidence(buffers.control_guarded, "control", gate, evidence_stream, copy)) {
        ++readable_regions;
        TmrLaunchControl control{};
        std::memcpy(&control, copy.data() + kGuardBytes, sizeof(control));
        control_verified = control.round_epoch == epoch && control.completion == 1 && control.host_cancel == 0 &&
                           control.runtime_status == control_probe::kInitRejected && control.cleanup_status == 0;
        evidence_valid &= control_verified;
        std::printf(
            "control_probe terminal control epoch=%llu completion=%u raw_status=%d cleanup=%d verified=%d\n",
            static_cast<unsigned long long>(control.round_epoch), control.completion, control.runtime_status,
            control.cleanup_status, control_verified
        );
    }
    if (read_terminal_evidence(buffers.reports_guarded, "reports", gate, evidence_stream, copy)) {
        ++readable_regions;
        cores_verified = true;
        for (int32_t i = 0; i < control_probe::kWorkers; ++i) {
            TmrCoreReport report{};
            std::memcpy(&report, copy.data() + kGuardBytes + i * sizeof(report), sizeof(report));
            const bool retired = report.ready == static_cast<uint32_t>(i + 1) &&
                                 report.exited == static_cast<uint32_t>(i + 1) && report.release == 1 &&
                                 report.command == static_cast<uint32_t>(TmrCoreCommand::Cancel) &&
                                 report.round_epoch == epoch;
            cores_verified &= retired;
            std::printf(
                "control_probe terminal core=%d ready=%u exited=%u release=%u epoch=%llu retired=%d\n", i, report.ready,
                report.exited, report.release, static_cast<unsigned long long>(report.round_epoch), retired
            );
        }
        evidence_valid &= cores_verified;
    }
    if (read_terminal_evidence(gate.threads, "worker_results", gate, evidence_stream, copy)) {
        ++readable_regions;
        observed_returns = 0;
        for (int32_t i = 0; i < gate.prepared.launched_threads; ++i) {
            control_probe::GateThreadResult thread{};
            std::memcpy(&thread, copy.data() + i * sizeof(thread), sizeof(thread));
            const bool current = thread.epoch == epoch;
            if (current) {
                ++observed_returns;
                evidence_valid &= thread.returned_status == control_probe::kInitRejected &&
                                  thread.runtime_status == control_probe::kInitRejected && thread.cleanup_status == 0 &&
                                  thread.dispatch_status == static_cast<int>(KernelDispatchStatus::ExecutionFailed) &&
                                  thread.native_status == simpler::tmr::kAicpuKernelInnerError && thread.ran == 0;
            }
            std::printf(
                "control_probe terminal worker=%d epoch=%llu current=%d raw_status=%d cleanup=%d dispatch=%d "
                "native=%d\n",
                i, static_cast<unsigned long long>(thread.epoch), current, thread.runtime_status, thread.cleanup_status,
                thread.dispatch_status, thread.native_status
            );
        }
    }
    if (read_terminal_evidence(gate.summary, "last_reader_summary", gate, evidence_stream, copy)) {
        ++readable_regions;
        control_probe::GateSummary summary{};
        std::memcpy(&summary, copy.data(), sizeof(summary));
        // CANN may cancel the last reader before it publishes this summary.
        // It is a diagnostic snapshot, not a new native-error acceptance barrier.
        std::printf(
            "control_probe terminal summary epoch=%llu finish=%d publish=%d clear=%d current=%d\n",
            static_cast<unsigned long long>(summary.epoch), summary.finish_calls, summary.publish_calls,
            summary.clear_calls, summary.epoch == epoch
        );
    }
    // CANN wraps a CPU failure in MODEL_EXECUTE on the graph path. The direct
    // entry comparison isolates this from the gate/Core protocol. EOS is not
    // an acceptable execution-error result. Current-round evidence is still
    // mandatory below, so an unrelated model failure cannot pass this test.
    const bool caller_error = caller_rc == (replay ? ACL_ERROR_RT_MODEL_EXECUTE : ACL_ERROR_RT_AICPU_EXCEPTION);
    const bool cpu_error = cpu_rc == ACL_ERROR_RT_AICPU_EXCEPTION;
    // The graph executes on caller; the original capture streams are only
    // diagnostic after replay. CPU-first eager keeps its stream-local criterion.
    const bool native_error = submission.cpu_enqueued &&
                              (replay ? replay_rc == 0 && caller_error : cpu_error || (caller_first && caller_error));
    // The SDK code alone cannot distinguish the injected execution rejection
    // from cleanup failure or an unrelated device fault. Require this round's
    // published status, Core retirement and at least one committed CPU frame;
    // never add a barrier merely to force all N frames to become observable.
    const bool evidence_complete = control_verified && cores_verified && observed_returns > 0;
    const bool native_verified = native_error && evidence_valid && evidence_complete;
    const char *verdict = native_verified ? "PASS" : !native_error || !evidence_valid ? "FAIL" : "UNVERIFIED";
    const char *caller_verdict = caller_error ? verdict : caller_rc == 0 ? "NOT_PROPAGATED" : "FAIL";
    const bool passed = native_verified && (!caller_first || caller_error);
    const bool caller_contract_failed =
        caller_first && !caller_error && evidence_valid && evidence_complete && (!replay || replay_rc == 0);
    const char *overall_reason = passed                 ? "none" :
                                 caller_contract_failed ? "caller_error_contract" :
                                 !native_error          ? "native_error_transport" :
                                                          "native_error_evidence";
    std::printf(
        "control_probe terminal observations coordinator_returns=%d/%d control=%s core_retirement=%s "
        "all_cpu_native_returns=UNVERIFIED\n",
        observed_returns, gate.prepared.launched_threads, control_verified ? "verified" : "unverified",
        cores_verified ? "verified" : "unverified"
    );
    std::printf(
        "control_probe terminal native_error=%s cpu_rc=%d readable_regions=%d/4 readable_evidence_consistent=%d "
        "expected_error_evidence=%d "
        "resources=retained_until_process_exit reset=0 reuse=0\n",
        verdict, cpu_rc, readable_regions, evidence_valid, evidence_complete
    );
    std::printf(
        "control_probe terminal caller_error=%s caller_rc=%d core_rc=%d mode=%s overall=%s overall_reason=%s\n",
        caller_verdict, caller_rc, core_rc, mode_name, passed ? "PASS" : "FAIL", overall_reason
    );
    // Even a successful negative test can leave CANN/gate state poisoned.
    // Do not unwind owners, free buffers, unload code, close streams or reset.
    std::fflush(stdout);
    std::fflush(stderr);
    std::_Exit(passed ? 0 : 1);
}
}  // namespace

int main(int argc, char **argv) {
    require(
        argc == 5 || argc == 6, "usage: control_probe DEVICE DISPATCHER_SO PROBE_SO AICORE_KERNEL_MODE_O "
                                "[--terminal-replay | --terminal-eager-caller-first]"
    );
    TerminalMode terminal_mode = TerminalMode::EagerCpuFirst;
    if (argc == 6) {
        if (std::strcmp(argv[5], "--terminal-replay") == 0) {
            terminal_mode = TerminalMode::Replay;
        } else {
            require(std::strcmp(argv[5], "--terminal-eager-caller-first") == 0, "unknown terminal mode");
            terminal_mode = TerminalMode::EagerCallerFirst;
        }
    }
    const int device = std::atoi(argv[1]);
    const auto dispatcher = read_binary(argv[2]);
    const auto cpu_binary = read_binary(argv[3]);
    const auto core_binary = read_binary(argv[4]);
    check(aclInit(nullptr), "aclInit");
    check(aclrtSetDevice(device), "set device");
    void *hal = dlopen("libascend_hal.so", RTLD_NOW | RTLD_GLOBAL);
    require(hal != nullptr, "load HAL");
    aclrtStream caller = nullptr;
    check(aclrtCreateStream(&caller), "create caller stream");
    {
        MemoryAllocator allocator;
        KernelExecutionState state;
        check(state.initialize(device, make_onboard_kernel_context_ops()), "prepare hidden streams and events");
        host::LoadAicpuOp loader;
        auto cpu_stream = state.hidden_stream(KernelStreamKind::Aicpu);
        check(
            loader.BootstrapDispatcher(
                dispatcher.data(), dispatcher.size(), cpu_binary.data(), cpu_binary.size(), cpu_stream, device
            ),
            "bootstrap probe SO"
        );
        check(
            loader.Init({control_probe::kSeedEntry, control_probe::kGateInitEntry, control_probe::kGateEntry}),
            "register probe entries"
        );
        PreparedBuffers buffers(allocator, device);
        GateBuffers gate_buffers(allocator, buffers, device);
        require(
            terminal_mode == TerminalMode::EagerCpuFirst || gate_buffers.prepared.launched_threads > 0,
            "explicit terminal mode requires target-specific gate topology"
        );
        auto init = buffers.init();
        check(loader.LaunchBuiltInOp(cpu_stream, &init, sizeof(init), 1, host::KernelNames::InitName), "prepare probe");
        if (gate_buffers.prepared.launched_threads > 0)
            check(
                loader.LaunchBuiltInOp(
                    cpu_stream, &gate_buffers.prepared, sizeof(gate_buffers.prepared), 1, control_probe::kGateInitEntry
                ),
                "prepare gate fixture"
            );
        check(state.mark_ready_enqueued(), "mark streams prepared");
        rtDevBinary_t binary{};
        binary.magic = RT_DEV_BINARY_MAGIC_ELF;
        binary.data = core_binary.data();
        binary.length = core_binary.size();
        void *core_handle = nullptr;
        check(rtRegisterAllKernel(&binary, &core_handle), "register production kernel-mode AICore binary");
        check(aclrtSynchronizeStreamWithTimeout(cpu_stream, 10000), "prepare complete");
        const auto prepared_allocations = allocator.get_allocation_count();
        const auto prepared_bytes = allocator.committed_bytes();
        uint64_t completed_rounds = 0;
        Submission submission{buffers, loader, core_handle, Mode::Open};
        for (int repeat = 0; repeat < 5; ++repeat) {
            for (Mode mode :
                 {Mode::Open, Mode::Reject, Mode::HostCancel, Mode::Open, Mode::OmitControlClear, Mode::Open}) {
                control_probe::Invocation seed{};
                check(
                    loader.LaunchBuiltInOp(cpu_stream, &seed, sizeof(seed), 1, control_probe::kSeedEntry),
                    "seed stale nonzero DMB"
                );
                check(aclrtSynchronizeStreamWithTimeout(cpu_stream, 10000), "seed complete before launch");
                submission.mode = mode;
                enqueue(state, submission, caller);
                check(aclrtSynchronizeStreamWithTimeout(caller, 10000), "eager joined tail");
                if (mode != Mode::HostCancel) ++completed_rounds;
                buffers.verify(mode, completed_rounds);
            }
        }
        std::array<aclmdlRI, 2> graphs{};
        for (size_t i = 0; i < graphs.size(); ++i) {
            submission.mode = i == 0 ? Mode::Open : Mode::Reject;
            check(aclmdlRICaptureBegin(caller, ACL_MODEL_RI_CAPTURE_MODE_GLOBAL), "capture begin");
            enqueue(state, submission, caller);
            check(aclmdlRICaptureEnd(caller, &graphs[i]), "capture end");
            require(graphs[i] != nullptr, "captured both hidden branches");
        }
        for (int repeat = 0; repeat < 10; ++repeat) {
            for (size_t graph : {size_t{0}, size_t{1}, size_t{0}}) {
                check(aclmdlRIExecuteAsync(graphs[graph], caller), "replay");
                check(aclrtSynchronizeStreamWithTimeout(caller, 10000), "replay joined tail");
                buffers.verify(graph == 0 ? Mode::Open : Mode::Reject, ++completed_rounds);
            }
        }
        for (auto graph : graphs)
            check(aclmdlRIDestroy(graph), "destroy original probe graph");
        uint64_t gate_rounds = 0;
        if (gate_buffers.prepared.launched_threads > 0) {
            submission.mode = Mode::Open;
            submission.gate_threads = gate_buffers.prepared.launched_threads;
            for (int repeat = 0; repeat < 5; ++repeat) {
                for (GateMode mode :
                     {GateMode::Open, GateMode::InitReject, GateMode::Open, GateMode::AdmissionReject,
                      GateMode::Open}) {
                    submission.gate_mode = mode;
                    enqueue(state, submission, caller);
                    check(aclrtSynchronizeStreamWithTimeout(caller, 10000), "gate eager joined tail");
                    gate_buffers.verify(buffers, mode, ++gate_rounds);
                }
            }
            graphs = {};
            for (size_t i = 0; i < graphs.size(); ++i) {
                submission.gate_mode = i == 0 ? GateMode::Open : GateMode::InitReject;
                check(aclmdlRICaptureBegin(caller, ACL_MODEL_RI_CAPTURE_MODE_GLOBAL), "gate capture begin");
                enqueue(state, submission, caller);
                check(aclmdlRICaptureEnd(caller, &graphs[i]), "gate capture end");
                require(graphs[i] != nullptr, "captured N/M gate with both hidden branches");
            }
            for (int repeat = 0; repeat < 5; ++repeat) {
                for (size_t graph : {size_t{0}, size_t{1}, size_t{0}}) {
                    check(aclmdlRIExecuteAsync(graphs[graph], caller), "gate replay");
                    check(aclrtSynchronizeStreamWithTimeout(caller, 10000), "gate replay joined tail");
                    gate_buffers.verify(buffers, graph == 0 ? GateMode::Open : GateMode::InitReject, ++gate_rounds);
                }
            }
            for (auto graph : graphs)
                check(aclmdlRIDestroy(graph), "destroy gate graph before resources");
            std::printf(
                "control_probe gate PASS N=%d M=%d eager=25 replay=15 final_read_depart=all finish_publish=once\n",
                gate_buffers.prepared.launched_threads, gate_buffers.prepared.execution_threads
            );
        } else {
            std::puts("control_probe gate SKIP: target-specific native topology preparation is not implemented");
        }
        require(
            allocator.get_allocation_count() == prepared_allocations && allocator.committed_bytes() == prepared_bytes,
            "no probe execution-buffer allocation after prepare"
        );
        if (gate_buffers.prepared.launched_threads > 0) {
            std::puts("control_probe PASS eager=30 replay=30 hidden_streams=2 between_round_reset=0");
            std::fflush(stdout);
            terminal_native_error(state, submission, caller, gate_buffers, gate_rounds + 1, terminal_mode);
        }
        check(state.close(), "close hidden streams after joined tails");
        loader.Finalize();
        check(rtDevBinaryUnRegister(core_handle), "unregister AICore");
        check(allocator.finalize(), "release prepared resources");
    }
    check(aclrtDestroyStream(caller), "destroy caller");
    // This is process teardown only; no reset occurs between eager/captured rounds.
    check(aclrtResetDevice(device), "final device teardown");
    check(aclFinalize(), "aclFinalize");
    dlclose(hal);
    std::puts("control_probe PASS eager=30 replay=30 hidden_streams=2 between_round_reset=0");
    return 0;
}
