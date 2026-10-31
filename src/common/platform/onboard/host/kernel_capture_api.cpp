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
#include "kernel_capture_api.h"

#include <atomic>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>

#include "common/unified_log.h"
#include "device_runner_base.h"
#include "native_run_context.h"
#include "runtime_c_api.h"
#include "tensormap_and_ringbuffer/kernel_invocation.h"

using KernelRun = NativeRunContext<DeviceRunnerBase>;
using simpler::tmr::CaptureRegistrationAction;
using simpler::tmr::KernelCaptureControl;
using simpler::tmr::KernelCaptureRegistrationArgs;

extern "C" __attribute__((weak)) int runtime_kernel_capture_supported_impl() { return 0; }

extern "C" int simpler_kernel_capture_supported(DeviceContextHandle) {
    return runtime_kernel_capture_supported_impl() != 0 ? 1 : 0;
}

namespace {
DeviceRunnerBase::PreparedExecution *execution(KernelRun &state) {
    return state.prepared_execution != nullptr ? state.prepared_execution.get() :
           state.active_execution != nullptr   ? state.active_execution->prepared.get() :
                                                 nullptr;
}

uint64_t next_execution_generation() {
    static std::atomic<uint64_t> next{1};
    uint64_t current = next.load(std::memory_order_acquire);
    while (current != 0) {
        const uint64_t successor = current == std::numeric_limits<uint64_t>::max() ? 0 : current + 1;
        if (next.compare_exchange_weak(current, successor, std::memory_order_acq_rel)) return current;
    }
    return 0;
}

int update_registration(
    DeviceRunnerBase &runner, KernelCaptureOwner &owner, const KernelCaptureRegistrationArgs &args, void *stream,
    bool wait
) {
    owner.cold_pending = true;
    int rc = PTO_RUNTIME_ERR_INTERNAL;
    const char *operation = "enqueue";
    const auto log_exception = [&](const char *detail) {
        LOG_ERROR(
            "Kernel registration %s threw: action=%u context=%llu capture=%llu callable=%d: %s", operation, args.action,
            static_cast<unsigned long long>(args.reference.context_generation),
            static_cast<unsigned long long>(args.reference.capture_generation), args.callable_id, detail
        );
    };
    try {
        rc = runner.launch_aicpu_payload(
            stream, const_cast<KernelCaptureRegistrationArgs *>(&args), sizeof(args),
            "simpler_aicpu_capture_registration", 1
        );
        if (rc == 0 && wait) {
            operation = "synchronize";
            rc = runner.synchronize_kernel_capture_stream(stream);
        }
    } catch (const std::exception &error) {
        rc = PTO_RUNTIME_ERR_INTERNAL;
        log_exception(error.what());
    } catch (...) {
        rc = PTO_RUNTIME_ERR_INTERNAL;
        log_exception("unknown C++ exception");
    }
    if (rc == 0 && wait) owner.cold_pending = false;
    if (rc != 0) runner.kernel_execution_state().poison(rc);
    return rc;
}

int read_control(KernelRun &state, KernelCaptureControl &control) {
    const auto &owner = *state.kernel_execution;
    if (owner.reference.control_address == 0) return PTO_RUNTIME_ERR_INVALID_STATE;
    int rc = state.runner->copy_from_device(
        &control, reinterpret_cast<void *>(owner.reference.control_address), sizeof(control)
    );
    if (rc != 0) return rc;
    if (control.capture_generation != owner.reference.capture_generation ||
        control.context_generation != owner.reference.context_generation ||
        control.runtime_address != owner.registration.runtime_address ||
        control.binding_epoch != owner.registration.binding_epoch)
        return PTO_RUNTIME_ERR_INVALID_STATE;
    if (control.replay_epoch > control.binding_epoch - 1) {
        state.observed_replay_sequence = control.replay_epoch;
        if (!state.retained_device_use.observe(control.replay_epoch)) return PTO_RUNTIME_ERR_INVALID_STATE;
        state.runner->note_workspace_run_fact(
            state.descriptor.pipeline_slot, state.descriptor.run_epoch, WorkspaceManager::RunFact::Launched
        );
    }
    return 0;
}

bool single_execution_complete(const KernelRun &state, const KernelCaptureControl &control) {
    return control.replay_epoch == state.descriptor.run_epoch && control.completed_epoch == control.replay_epoch &&
           control.launched_threads != 0 && control.finished_threads == control.launched_threads;
}

void note_single_execution_retired(KernelCaptureOwner &owner) {
    owner.single_execution_retired = true;
    owner.registered = false;
    owner.cold_pending = false;
}
}  // namespace

int prewarm_kernel_run(KernelRun &state, DeviceRunnerBase::PreparedExecution &prepared_run, bool retain_registration) {
    if (runtime_kernel_capture_supported_impl() == 0) return PTO_RUNTIME_ERR_UNSUPPORTED;
    if (prepared_run.native_context != &state || prepared_run.runtime != &state.runtime ||
        prepared_run.identity != state.identity())
        return PTO_RUNTIME_ERR_INVALID_STATE;
    auto *prepared = &prepared_run;
    auto &runner = *state.runner;
    int rc = runner.validate_current_context();
    if (rc != 0) return rc;
    if (state.kernel_execution != nullptr && state.kernel_execution->ready)
        return state.kernel_execution->caller_stream == prepared->caller_stream ? 0 : PTO_RUNTIME_ERR_INVALID_STATE;
    rc = runner.prepare_submission(*prepared);
    if (rc != 0) return rc;
    if (state.kernel_execution == nullptr) state.kernel_execution = std::make_unique<KernelCaptureOwner>();
    auto &owner = *state.kernel_execution;
    prepared->kernel_resources = &owner;
    owner.graph_reference_count = &state.graph_reference_count;
    owner.caller_stream = prepared->caller_stream;
    state.retained_device_use.retain(state.identity());
    runner.note_workspace_run_fact(
        state.descriptor.pipeline_slot, state.descriptor.run_epoch, WorkspaceManager::RunFact::DeviceReferencesRetained
    );
    rc = runner.ensure_aicore_binary_registered();
    if (rc != 0) return rc;
    void *stream = prepared->streams.aicpu;
    if (stream == nullptr) return PTO_RUNTIME_ERR_INVALID_STATE;
    owner.cold_pending = true;
    LaunchProgressSink publication;
    rc = prepared->kernel_args.runtime_args_published() ?
             0 :
             prepared->kernel_args.publish_runtime_args(true, stream, publication);
    if (rc != 0) {
        runner.kernel_execution_state().poison(rc);
        return rc;
    }
    const auto *packet = static_cast<const unsigned char *>(prepared->kernel_args.launch_payload());
    const size_t packet_bytes = prepared->kernel_args.launch_payload_bytes();
    if (packet == nullptr || packet_bytes < LAUNCH_ENVELOPE_HEADER_BYTES + sizeof(SimplerKernelInvocationHeader) +
                                                sizeof(simpler::tmr::EagerKernelBinding))
        return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
    KernelArgs header{};
    std::memcpy(&header, packet, sizeof(header));
    if (header.entry_args_source != static_cast<uint32_t>(EntryArgsSource::KernelInvocation))
        return PTO_RUNTIME_ERR_UNSUPPORTED;
    SimplerKernelInvocationHeader invocation{};
    std::memcpy(&invocation, packet + LAUNCH_ENVELOPE_HEADER_BYTES, sizeof(invocation));
    if (owner.pinned_callable_hash == 0) {
        rc = runner.pin_kernel_capture_callable(invocation.callable_id, &owner.pinned_callable_hash);
        if (rc != 0) return rc;
    }
    if (owner.reference.capture_generation == 0) owner.reference.capture_generation = next_execution_generation();
    owner.reference.context_generation = runner.kernel_context_generation();
    if (owner.reference.capture_generation == 0 || owner.reference.context_generation == 0)
        return PTO_RUNTIME_ERR_INVALID_STATE;
    if (owner.reference.control_address == 0) {
        void *control = runner.allocate_tensor(sizeof(KernelCaptureControl));
        if (control == nullptr) return PTO_RUNTIME_ERR_INTERNAL;
        owner.reference.control_address = reinterpret_cast<uint64_t>(control);
    }
    KernelCaptureControl initial{};
    initial.capture_generation = owner.reference.capture_generation;
    initial.context_generation = owner.reference.context_generation;
    initial.runtime_address = reinterpret_cast<uint64_t>(header.runtime_args);
    initial.binding_epoch = state.descriptor.run_epoch;
    initial.replay_epoch = initial.completed_epoch = initial.binding_epoch - 1;
    initial.callable_generation = invocation.generation;
    initial.callable_id = invocation.callable_id;
    initial.tensor_count = invocation.tensor_count;
    initial.scalar_count = invocation.scalar_count;
    initial.launched_threads = state.runtime.get_aicpu_launch_count() > 0 ? state.runtime.get_aicpu_launch_count() :
                                                                            prepared->launch_aicpu_num;
    initial.worker_count = prepared->num_aicore;
    rc = runner.copy_to_device(reinterpret_cast<void *>(owner.reference.control_address), &initial, sizeof(initial));
    if (rc != 0) {
        runner.kernel_execution_state().poison(rc);
        return rc;
    }
    owner.control_initialized = true;
    owner.launch_packet.assign(packet, packet + packet_bytes);
    const size_t reference_offset = LAUNCH_ENVELOPE_HEADER_BYTES + sizeof(SimplerKernelInvocationHeader) +
                                    offsetof(simpler::tmr::EagerKernelBinding, reference);
    std::memcpy(owner.launch_packet.data() + reference_offset, &owner.reference, sizeof(owner.reference));
    auto &registration = owner.registration;
    registration.action = static_cast<uint32_t>(
        retain_registration ? CaptureRegistrationAction::Register : CaptureRegistrationAction::RegisterOnce
    );
    registration.reference = owner.reference;
    registration.runtime_address = initial.runtime_address;
    registration.binding_epoch = initial.binding_epoch;
    registration.callable_generation = initial.callable_generation;
    registration.run_result_address = header.run_result_data_base;
    registration.packet_bytes = static_cast<uint32_t>(owner.launch_packet.size());
    registration.tensor_count = initial.tensor_count;
    registration.scalar_count = initial.scalar_count;
    registration.launched_threads = initial.launched_threads;
    registration.worker_count = initial.worker_count;
    registration.callable_id = invocation.callable_id;
    registration.enable_profiling_flag = header.enable_profiling_flag;
    registration.device_wall_data_base = header.device_wall_data_base;
    registration.header_bytes = sizeof(header);
    std::memcpy(registration.launch_header, &header, sizeof(header));
    // An enqueue failure does not prove that registration was not queued.
    // Keep its revoke obligation until quiescent cleanup confirms removal.
    owner.registered = true;
    rc = update_registration(runner, owner, registration, stream, false);
    if (rc == 0) owner.ready = true;
    return rc;
}

int quiesce_kernel_run(KernelRun &state) {
    if (state.kernel_execution == nullptr) return 0;
    auto &owner = *state.kernel_execution;
    auto *prepared = execution(state);
    if (prepared == nullptr) return PTO_RUNTIME_ERR_INVALID_STATE;
    int rc = state.runner->validate_current_context();
    if (rc != 0) return rc;
    (void)state.runner->consume_device_fault_notices();
    if (owner.caller_stream != nullptr) {
        rc = state.runner->validate_caller_stream(owner.caller_stream);
        if (rc != 0) return rc;
        rc = state.runner->synchronize_kernel_capture_stream(owner.caller_stream);
        if (rc != 0) return rc;
    }
    for (void *stream : {prepared->streams.aicpu, prepared->streams.aicore}) {
        if (stream == nullptr) continue;
        rc = state.runner->synchronize_kernel_capture_stream(stream);
        if (rc != 0) return rc;
    }
    owner.cold_pending = false;
    (void)state.runner->consume_device_fault_notices();
    return 0;
}

int read_kernel_run_status(KernelRun &state) {
    if (state.kernel_execution == nullptr) return PTO_RUNTIME_ERR_INVALID_STATE;
    KernelCaptureControl control{};
    const int rc = read_control(state, control);
    if (rc != 0) return rc;
    if (control.replay_epoch < control.binding_epoch || control.completed_epoch != control.replay_epoch)
        return PTO_RUNTIME_ERR_INVALID_STATE;
    auto &owner = *state.kernel_execution;
    if (!owner.graph_retained() &&
        owner.registration.action == static_cast<uint32_t>(CaptureRegistrationAction::RegisterOnce) &&
        state.runner->native_run_drain_proved(state.identity()) && single_execution_complete(state, control))
        note_single_execution_retired(owner);
    return control.first_error;
}

int observe_kernel_run(KernelRun &state, bool wait) {
    if (state.kernel_execution == nullptr) return wait ? PTO_RUNTIME_ERR_INVALID_STATE : SIMPLER_NATIVE_RUN_POLL_ERROR;
    auto *prepared = execution(state);
    if (prepared == nullptr) return wait ? PTO_RUNTIME_ERR_INVALID_STATE : SIMPLER_NATIVE_RUN_POLL_ERROR;
    const int complete = wait ? quiesce_kernel_run(state) : state.runner->poll_kernel_streams(*prepared);
    if (wait && complete != 0) return complete;
    if (!wait && complete != SIMPLER_NATIVE_RUN_POLL_COMPLETE) return complete;
    const int rc = read_kernel_run_status(state);
    return wait ? rc : rc == 0 ? SIMPLER_NATIVE_RUN_POLL_COMPLETE : SIMPLER_NATIVE_RUN_POLL_ERROR;
}

int release_kernel_run(KernelRun &state) {
    if (state.kernel_execution == nullptr) return 0;
    auto &owner = *state.kernel_execution;
    auto *prepared = execution(state);
    if (prepared == nullptr) return PTO_RUNTIME_ERR_INVALID_STATE;
    int rc = state.runner->validate_current_context();
    if (rc != 0) return rc;
    bool own_completion = !owner.graph_retained() && state.runner->native_run_drain_proved(state.identity()) &&
                          owner.registration.action == static_cast<uint32_t>(CaptureRegistrationAction::RegisterOnce);
    if (!own_completion) {
        rc = quiesce_kernel_run(state);
        if (rc != 0) return rc;
    }
    KernelCaptureControl control{};
    control.replay_epoch = control.completed_epoch = state.observed_replay_sequence;
    control.first_error = state.completion_rc == -1 ? 0 : state.completion_rc;
    if (owner.reference.control_address != 0 && owner.control_initialized && !owner.single_execution_retired) {
        rc = read_control(state, control);
        if (rc != 0) return rc;
    }
    if (own_completion && !owner.single_execution_retired) {
        if (single_execution_complete(state, control)) {
            // All single-use callbacks returned before the proved boundary;
            // their last callback erased the registration. No stream-wide wait
            // or revoke queued behind a successor is needed for this run.
            note_single_execution_retired(owner);
        } else {
            own_completion = false;
            rc = quiesce_kernel_run(state);
            if (rc != 0) return rc;
            if (owner.reference.control_address != 0 && owner.control_initialized) {
                rc = read_control(state, control);
                if (rc != 0) return rc;
            }
        }
    }
    const bool ran = state.observed_replay_sequence != 0;
    if (ran && control.completed_epoch != control.replay_epoch) return PTO_RUNTIME_ERR_INVALID_STATE;
    if (state.completion_rc == -1 || state.completion_rc == 0) state.completion_rc = ran ? control.first_error : 0;
    if (!own_completion) {
        rc = state.runner->discharge_kernel_quiescence(*prepared);
        if (rc != 0) return rc;
    }
    if (ran && state.capture_result != nullptr) {
        auto &result = *state.capture_result;
        if (!result.read_attempted) {
            result.replay_sequence = control.completed_epoch;
            // The whole-operator wait above covers both participating streams.
            result.boundaries_complete = true;
            result.read_attempted = true;
            result.transfer_rc = PTO_RUNTIME_ERR_INTERNAL;
            result.transfer_rc = state.runner->copy_from_device(
                &result.snapshot, reinterpret_cast<const void *>(owner.registration.run_result_address),
                sizeof(result.snapshot)
            );
            if (result.transfer_rc != 0) {
                result.snapshot = {};
            }
            if (result.transfer_rc == 0 && device_run_result_published(result.snapshot, state.descriptor.run_epoch))
                state.host_api.bind_run_result(result.snapshot.payload, result.snapshot.payload_bytes);
        }
    }
    if (owner.registered) {
        auto revoke = owner.registration;
        revoke.action = static_cast<uint32_t>(CaptureRegistrationAction::Revoke);
        rc = update_registration(*state.runner, owner, revoke, prepared->streams.aicpu, true);
        if (rc != 0) return rc;
        owner.registered = false;
    }
    if (owner.reference.control_address != 0) {
        rc = state.runner->release_kernel_capture_control(reinterpret_cast<void *>(owner.reference.control_address));
        if (rc != 0) return rc;
        owner.reference.control_address = 0;
        owner.control_initialized = false;
    }
    if (owner.pinned_callable_hash != 0) {
        rc = state.runner->release_chip_callable_buffer(owner.pinned_callable_hash);
        if (rc != 0) return rc;
        owner.pinned_callable_hash = 0;
    }
    if (owner.graph_retained() || (ran && state.observed_replay_sequence != state.descriptor.run_epoch)) {
        auto &slot = state.runner->slot_persistent_args(state.descriptor.pipeline_slot);
        slot.workers_initialized = false;
        slot.published_prefix_bytes = 0;
    }
    if (!state.retained_device_use.settle(state.identity())) return PTO_RUNTIME_ERR_INVALID_STATE;
    if (state.runner->kernel_capture_owner() == &state) state.runner->set_kernel_capture_owner(nullptr);
    prepared->kernel_resources = nullptr;
    state.kernel_execution.reset();
    state.phase.store(NativeRunPhase::Complete, std::memory_order_release);
    return 0;
}

int quiesce_kernel_capture_for_callable_change(DeviceRunnerBase *runner, int32_t callable_id, bool revoke) {
    auto *state = static_cast<KernelRun *>(runner->kernel_capture_owner());
    if (state == nullptr) return 0;
    if (state->runner != runner || state->kernel_execution == nullptr) return PTO_RUNTIME_ERR_INVALID_STATE;
    int rc = quiesce_kernel_run(*state);
    if (rc != 0) return rc;
    auto &owner = *state->kernel_execution;
    if (!revoke || !owner.registered || owner.registration.callable_id != callable_id) return 0;
    auto args = owner.registration;
    args.action = static_cast<uint32_t>(CaptureRegistrationAction::RevokeCallable);
    return update_registration(*runner, owner, args, execution(*state)->streams.aicpu, true);
}
