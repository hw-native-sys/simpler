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
#include <atomic>
#include <cstring>

#include "../protocol.h"
#include "aicpu_common/context/utils/status.h"
#include "common/unified_log.h"
#include "task_interface/tmr_kernel_context.h"
#include "tensormap_and_ringbuffer/kernel_execution_round.h"
#include "tensormap_and_ringbuffer/kernel_native_status.h"

namespace {
using namespace simpler::tmr;
static_assert(kAicpuKernelSuccess == aicpu::KERNEL_STATUS_OK);
static_assert(kAicpuKernelInnerError == aicpu::KERNEL_STATUS_INNER_ERROR);
control_probe::GateInit prepared{};
std::atomic<uint64_t> entries{0};
std::atomic<bool> failed{false};
thread_local int32_t execution_index = -1;
thread_local bool initialized = false;
thread_local bool ran = false;

struct Observations {
    control_probe::GateSummary summary{};
    std::atomic<int32_t> init_calls{0};
    std::atomic<int32_t> run_calls{0};

    void reset(uint64_t epoch) {
        const uint64_t next = summary.completed_rounds + 1;
        summary = {};
        summary.epoch = epoch;
        summary.completed_rounds = next;
        init_calls.store(0, std::memory_order_relaxed);
        run_calls.store(0, std::memory_order_relaxed);
    }
};

// Instrument calls, never substitute the production cache/MMIO protocol.
struct CountedCoreGroup {
    Observations &observed;
    KernelCoreGroup core;

    bool attach(KernelHandshakeView view) noexcept {
        if (observed.summary.completed_rounds != 0 && observed.summary.clear_calls != 1) return false;
        observed.reset(view.epoch);
        return core.attach(view);
    }
    int32_t finish() noexcept {
        ++observed.summary.finish_calls;
        return core.finish();
    }
    void publish_status(int32_t runtime, int32_t cleanup) noexcept {
        core.publish_status(runtime, cleanup);
        observed.summary.runtime_status = runtime;
        observed.summary.cleanup_status = cleanup;
        ++observed.summary.publish_calls;
    }
};

// This is a coordinator protocol fixture, not a TMR callable/resource provider.
// There are no tensors, scheduler tasks or orchestration results to emulate.
struct ProbeInvocation {
    ExecutionInputs input{};
    Runtime *resident() const { return nullptr; }
    const ExecutionInputs &inputs() const { return input; }
};

struct ProbeExecutor {
    Observations observed;
    KernelRoundGate kernel_gate_;
    CountedCoreGroup kernel_cores_{observed, {}};
    bool kernel_control_attached_{false};
    ProbeInvocation kernel_invocation_;
    control_probe::GateMode mode{control_probe::GateMode::Open};

    int32_t prepare_kernel_round(const KernelExecutionRequest &request) {
        ++observed.summary.prepare_calls;
        if (request.packet.data == nullptr || request.packet.size != sizeof(control_probe::GateInvocation)) return -1;
        control_probe::GateInvocation invocation{};
        std::memcpy(&invocation, request.packet.data, sizeof(invocation));
        mode = invocation.mode == control_probe::GateMode::TerminalInitReject ? control_probe::GateMode::InitReject :
                                                                                invocation.mode;
        return kernel_cores_.core.collect_reports(
            reinterpret_cast<const uint64_t *>(prepared.registers), platform_get_physical_cores_count()
        );
    }
    int32_t initialize_kernel_thread(const KernelThreadView &thread) {
        execution_index = thread.execution_index;
        initialized = true;
        observed.init_calls.fetch_add(1, std::memory_order_relaxed);
        if (thread.execution_index < 0 || thread.execution_index >= control_probe::kExecutionThreads ||
            thread.execution_threads != control_probe::kExecutionThreads)
            return -1;
        for (int32_t worker = thread.execution_index; worker < control_probe::kWorkers;
             worker += thread.execution_threads)
            kernel_cores_.core.open(worker);
        return mode == control_probe::GateMode::InitReject && thread.execution_index == 1 ?
                   control_probe::kInitRejected :
                   0;
    }
    int32_t complete_kernel_init() {
        ++observed.summary.verdict_calls;
        if (observed.init_calls.load(std::memory_order_relaxed) != control_probe::kExecutionThreads) return -1;
        const uint64_t deadline = platform_aicore_exit_deadline();
        for (int32_t worker = 0; worker < control_probe::kWorkers; ++worker) {
            while (read_reg(kernel_cores_.core.register_address(worker), RegId::COND) != AICORE_IDLE_VALUE) {
                if (get_sys_cnt_aicpu() > deadline) return -1;
            }
        }
        return 0;
    }
    int32_t run(Runtime *, const ExecutionInputs &, const KernelThreadView *thread) {
        ran = true;
        observed.run_calls.fetch_add(1, std::memory_order_relaxed);
        return thread != nullptr && execution_index >= 0 && execution_index < control_probe::kExecutionThreads &&
                       thread->execution_index == execution_index &&
                       observed.init_calls.load(std::memory_order_relaxed) == control_probe::kExecutionThreads &&
                       observed.summary.verdict_calls == 1 && mode == control_probe::GateMode::Open ?
                   0 :
                   -1;
    }
    void cancel_kernel_round() { kernel_cores_.core.cancel(); }
    int32_t kernel_status() { return 0; }
    int32_t finalize_kernel_round() {
        ++observed.summary.finalize_calls;
        return 0;
    }
    void clear_kernel_round() {
        ++observed.summary.clear_calls;
        observed.summary.init_calls = observed.init_calls.load(std::memory_order_relaxed);
        observed.summary.run_calls = observed.run_calls.load(std::memory_order_relaxed);
        auto *destination = reinterpret_cast<void *>(prepared.summary);
        std::memcpy(destination, &observed.summary, sizeof(observed.summary));
        cache_flush_range(destination, sizeof(observed.summary));
    }
};
ProbeExecutor executor;
}  // namespace

extern "C" __attribute__((visibility("default"))) int control_probe_gate_init(void *args) {
    if (args == nullptr || prepared.descriptor != 0) return 1;
    control_probe::GateInit candidate{};
    std::memcpy(&candidate, args, sizeof(candidate));
    if (candidate.descriptor == 0 || candidate.registers == 0 || candidate.summary == 0 || candidate.results == 0 ||
        candidate.summary % alignof(control_probe::GateSummary) != 0 ||
        candidate.results % alignof(control_probe::GateThreadResult) != 0 ||
        candidate.launched_threads != PLATFORM_MAX_AICPU_THREADS_JUST_FOR_LAUNCH ||
        candidate.execution_threads != control_probe::kExecutionThreads || candidate.allowed_cpus[0] < 0 ||
        candidate.allowed_cpus[1] < 0 || candidate.allowed_cpus[0] == candidate.allowed_cpus[1])
        return 1;
    const auto *descriptor = reinterpret_cast<const TmrKernelContextDescriptor *>(candidate.descriptor);
    cache_invalidate_range(descriptor, sizeof(*descriptor));
    if (descriptor->self_address != candidate.descriptor ||
        descriptor->context_generation != control_probe::kContextGeneration ||
        descriptor->worker_count != control_probe::kWorkers)
        return 1;
    prepared = candidate;
    return 0;
}

extern "C" __attribute__((visibility("default"))) int control_probe_gate(void *args) {
    if (args == nullptr || prepared.descriptor == 0 || failed.load(std::memory_order_relaxed)) return 1;
    const int scheduling_error = platform_aicpu_prepare_kernel_thread();
    const int32_t cpu = platform_aicpu_current_cpu();
    execution_index = -1;
    initialized = false;
    ran = false;
    const uint64_t entry = entries.fetch_add(1, std::memory_order_relaxed);
    const auto &descriptor = *reinterpret_cast<const TmrKernelContextDescriptor *>(prepared.descriptor);
    control_probe::GateInvocation invocation{};
    std::memcpy(&invocation, args, sizeof(invocation));
    KernelExecutionRequest request;
    request.packet = {reinterpret_cast<const uint8_t *>(&invocation), sizeof(invocation)};
    request.handshake = {
        reinterpret_cast<TmrLaunchControl *>(descriptor.control_address),
        reinterpret_cast<TmrCoreReport *>(descriptor.reports_address), descriptor.worker_count, 0
    };
    request.allowed_cpus = prepared.allowed_cpus;
    request.execution_threads = prepared.execution_threads;
    request.launched_threads = prepared.launched_threads;
    if (invocation.mode == control_probe::GateMode::AdmissionReject)
        request.admission_status = control_probe::kAdmissionRejected;
    else if (invocation.reserved != 0 || (invocation.mode != control_probe::GateMode::Open &&
                                          invocation.mode != control_probe::GateMode::InitReject &&
                                          invocation.mode != control_probe::GateMode::TerminalInitReject))
        request.admission_status = -1;
    KernelFinalStatus final{};
    const int32_t status = execute_kernel_round_impl(executor, request, cpu, &final);
    control_probe::GateThreadResult result{};
    result.epoch = entry / static_cast<uint64_t>(prepared.launched_threads) + 1;
    result.cpu = cpu;
    result.execution_index = execution_index;
    result.returned_status = status;
    result.runtime_status = final.runtime_status;
    result.cleanup_status = final.cleanup_status;
    result.initialized = initialized;
    result.ran = ran;
    result.scheduling_error = scheduling_error;
    const bool terminal = invocation.mode == control_probe::GateMode::TerminalInitReject;
    if (final.cleanup_status != 0 || status != final.runtime_status) failed.store(true, std::memory_order_relaxed);
    result.dispatch_status = classify_kernel_dispatch_status(status, final.cleanup_status);
    // Nonterminal cases deliberately retain native success to test reuse of
    // the coordinator; only the final case tests the real error transport.
    const int native_status = to_aicpu_native_status(
        terminal                               ? result.dispatch_status :
        failed.load(std::memory_order_relaxed) ? 1 :
                                                 0
    );
    result.native_status = native_status;
    const uint64_t slot = entry % static_cast<uint64_t>(prepared.launched_threads);
    auto *destination = reinterpret_cast<control_probe::GateThreadResult *>(prepared.results) + slot;
    // No extra barrier repairs the coordinator: each native worker writes its
    // own completion only after the real final-read/depart operations returned.
    if (terminal) {
        const uint64_t epoch = result.epoch;
        result.epoch = 0;
        std::memcpy(destination, &result, sizeof(result));
        cache_flush_range(destination, sizeof(result));
        // Commit only this worker's frame. This is not an N-worker barrier:
        // CANN may cancel another worker before it commits its own frame.
        __atomic_store_n(&destination->epoch, epoch, __ATOMIC_RELEASE);
        cache_flush_range(&destination->epoch, sizeof(destination->epoch));
        LOG_ERROR(
            "control_probe intentional native error: epoch=%llu slot=%llu raw=%d cleanup=%d dispatch=%d native=%d",
            static_cast<unsigned long long>(epoch), static_cast<unsigned long long>(slot), status, final.cleanup_status,
            result.dispatch_status, native_status
        );
    } else {
        std::memcpy(destination, &result, sizeof(result));
        cache_flush_range(destination, sizeof(result));
    }
    return native_status;
}
