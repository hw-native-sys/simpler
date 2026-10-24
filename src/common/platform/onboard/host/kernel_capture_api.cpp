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

#include <acl/acl.h>
#include <atomic>
#include <cstring>
#include <limits>
#include <memory>

#include "device_runner_base.h"
#include "host/kernel_capture_binder.h"
#include "native_run_context.h"
#include "runtime_c_api.h"
#include "task_interface/kernel_invocation_header.h"

using CaptureRun = NativeRunContext<DeviceRunnerBase>;
using simpler::tmr::CaptureRegistrationAction;
using simpler::tmr::KernelCaptureControl;
using simpler::tmr::KernelCaptureRegistrationArgs;

extern "C" __attribute__((weak)) int runtime_kernel_capture_supported_impl() { return 0; }

namespace {

CaptureRun *capture_run(DeviceContextHandle context, RuntimeHandle runtime) {
    if (context == nullptr || runtime == nullptr) return nullptr;
    uint64_t magic = 0;
    std::memcpy(&magic, runtime, sizeof(magic));
    if (magic != CaptureRun::kMagic) return nullptr;
    auto *state = static_cast<CaptureRun *>(runtime);
    if (state->runner != context || state->capture_execution == nullptr ||
        state->phase.load(std::memory_order_acquire) != NativeRunPhase::Prepared ||
        state->runner->kernel_capture_owner() != state)
        return nullptr;
    return state;
}

uint64_t next_capture_generation() {
    static std::atomic<uint64_t> next{1};
    uint64_t current = next.load(std::memory_order_acquire);
    while (current != 0) {
        const uint64_t successor = current == std::numeric_limits<uint64_t>::max() ? 0 : current + 1;
        if (next.compare_exchange_weak(current, successor, std::memory_order_acq_rel)) return current;
    }
    return 0;
}

int register_capture(DeviceRunnerBase *runner, const KernelCaptureRegistrationArgs &args) {
    void *stream = runner->kernel_execution_state().hidden_stream(KernelStreamKind::Aicpu);
    int rc = PTO_RUNTIME_ERR_INTERNAL;
    try {
        rc = runner->launch_aicpu_payload(
            stream, const_cast<KernelCaptureRegistrationArgs *>(&args), sizeof(args),
            "simpler_aicpu_capture_registration", 1
        );
        if (rc == 0) rc = runner->synchronize_kernel_capture_stream(stream);
    } catch (...) {
        rc = PTO_RUNTIME_ERR_INTERNAL;
    }
    // Failed submission or completion keeps registration arguments and device
    // references owned. Poisoning refuses admission but does not prove quiescence.
    if (rc != 0) runner->kernel_execution_state().poison(rc);
    return rc;
}

int quiesce_capture(CaptureRun &state) {
    auto &owner = *state.capture_execution;
    int rc = state.runner->validate_current_context();
    if (rc != 0) return rc;
    (void)state.runner->consume_device_fault_notices();
    if (owner.caller_stream != nullptr) {
        rc = state.runner->validate_caller_stream(owner.caller_stream);
        if (rc != 0) return rc;
        rc = state.runner->synchronize_kernel_capture_stream(owner.caller_stream);
        if (rc != 0) return rc;
    }
    if (owner.enqueue_attempted && !owner.recorded) {
        // Without the full join, the caller alone cannot cover accepted hidden
        // submissions. AICore accepted without AICPU may never return; a failed
        // bounded wait retains the owner rather than freeing its memory.
        for (const auto kind : {KernelStreamKind::Aicpu, KernelStreamKind::Aicore}) {
            rc = state.runner->synchronize_kernel_capture_stream(
                state.runner->kernel_execution_state().hidden_stream(kind)
            );
            if (rc != 0) return rc;
        }
    }
    (void)state.runner->consume_device_fault_notices();
    return 0;
}

int observe_capture_execution(CaptureRun &state, KernelCaptureControl *control) {
    const auto &owner = *state.capture_execution;
    if (owner.reference.control_address == 0 || owner.registration.reference.control_address == 0) return 0;
    const int rc = state.runner->copy_from_device(
        control, reinterpret_cast<void *>(owner.reference.control_address), sizeof(*control)
    );
    if (rc != 0) return rc;
    if (control->capture_generation != owner.reference.capture_generation ||
        control->context_generation != owner.reference.context_generation ||
        control->runtime_address != owner.registration.runtime_address ||
        control->binding_epoch != owner.registration.binding_epoch ||
        !state.retained_device_use.observe(control->replay_epoch))
        return PTO_RUNTIME_ERR_INVALID_STATE;
    if (control->replay_epoch != 0)
        state.runner->note_workspace_run_fact(
            state.descriptor.pipeline_slot, state.descriptor.run_epoch, WorkspaceManager::RunFact::Launched
        );
    return 0;
}

void invalidate_capture_slot(DeviceRunnerBase &runner, uint32_t slot_id) {
    auto &slot = runner.slot_persistent_args(slot_id);
    // A new owner must publish initialized handshake storage again. Its host
    // epoch may equal an old device-minted replay epoch; no such report may
    // survive into the next ownership generation.
    slot.workers_initialized = false;
    slot.published_prefix_bytes = 0;
}

}  // namespace

int quiesce_kernel_capture_for_callable_change(DeviceRunnerBase *runner, int32_t callable_id, bool revoke) {
    auto *state = static_cast<CaptureRun *>(runner->kernel_capture_owner());
    if (state == nullptr) return 0;
    if (capture_run(runner, state) == nullptr) return PTO_RUNTIME_ERR_INVALID_STATE;
    const int rc = quiesce_capture(*state);
    if (rc != 0) return rc;
    auto &owner = *state->capture_execution;
    if (!revoke || !owner.registered || owner.registration.callable_id != callable_id) return 0;
    auto args = owner.registration;
    args.action = static_cast<uint32_t>(CaptureRegistrationAction::RevokeCallable);
    return register_capture(runner, args);
}

extern "C" int simpler_kernel_capture_supported(DeviceContextHandle) {
    return runtime_kernel_capture_supported_impl() != 0 ? 1 : 0;
}

extern "C" int simpler_kernel_capture_prepare(
    DeviceContextHandle context, RuntimeHandle runtime, int32_t callable_id, const void *args, const CallConfig *config,
    const NativeRunDescriptor *descriptor
) {
    if (context == nullptr || runtime == nullptr || config == nullptr || descriptor == nullptr || args == nullptr ||
        callable_id < 0 || callable_id >= MAX_REGISTERED_CALLABLE_IDS)
        return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
    if (runtime_kernel_capture_supported_impl() == 0) return PTO_RUNTIME_ERR_UNSUPPORTED;
    auto *runner = static_cast<DeviceRunnerBase *>(context);
    if (!runner->execution_mode_latch().is_kernel() || !runner->kernel_execution_state().accepts_dispatch() ||
        runner->kernel_capture_owner() != nullptr || runner->native_runs_outstanding())
        return PTO_RUNTIME_ERR_INVALID_STATE;
    if (descriptor->pipeline_slot != 0 || config->diagnostics_any()) return PTO_RUNTIME_ERR_UNSUPPORTED;
    int rc = runner->validate_current_context();
    if (rc != 0) return rc;
    invalidate_capture_slot(*runner, descriptor->pipeline_slot);
    rc = simpler_prepare_run(context, runtime, callable_id, args, config, descriptor);
    if (rc != 0) return rc;
    auto *state = static_cast<CaptureRun *>(runtime);
    try {
        state->capture_execution = std::make_unique<KernelCaptureOwner>();
        runner->set_kernel_capture_owner(state);
        auto &owner = *state->capture_execution;
        state->retained_device_use.retain(state->identity());
        runner->note_workspace_run_fact(
            state->descriptor.pipeline_slot, state->descriptor.run_epoch,
            WorkspaceManager::RunFact::DeviceReferencesRetained
        );
        auto &execution = *state->prepared_execution;
        rc = runner->pin_kernel_capture_callable(callable_id, &owner.pinned_callable_hash);
        if (rc != 0) return rc;
        runner->activate_launch_shape(state->runtime);
        rc = runner->ensure_aicore_binary_registered();
        if (rc != 0) return rc;
        // Keep mainline's pinned asynchronous publisher. This cold operation
        // completes on the private CPU stream before graph recording; replay
        // never uploads or clears the retained execution image.
        void *publication_stream = runner->kernel_execution_state().hidden_stream(KernelStreamKind::Aicpu);
        runner->note_run_stream_identity(publication_stream);
        runner->note_run_stream_identity(runner->kernel_execution_state().hidden_stream(KernelStreamKind::Aicore));
        owner.enqueue_attempted = true;
        LaunchProgressSink publication;
        rc = execution.kernel_args.publish_runtime_args(true, publication_stream, publication);
        if (rc == 0) rc = runner->synchronize_kernel_capture_stream(publication_stream);
        if (rc != 0) {
            runner->kernel_execution_state().poison(rc);
            return rc;
        }
        owner.enqueue_attempted = false;
        const auto *packet = static_cast<const unsigned char *>(execution.kernel_args.launch_payload());
        const size_t packet_bytes = execution.kernel_args.launch_payload_bytes();
        if (packet == nullptr || packet_bytes < LAUNCH_ENVELOPE_HEADER_BYTES + sizeof(SimplerKernelInvocationHeader) ||
            packet_bytes > UINT32_MAX - sizeof(owner.reference))
            return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
        owner.launch_packet.assign(packet, packet + packet_bytes);
        owner.launch_packet.insert(
            owner.launch_packet.begin() + LAUNCH_ENVELOPE_HEADER_BYTES, sizeof(owner.reference), 0
        );
        KernelArgs header{};
        std::memcpy(&header, owner.launch_packet.data(), sizeof(header));
        if (header.entry_args_source != static_cast<uint32_t>(EntryArgsSource::KernelInvocation))
            return PTO_RUNTIME_ERR_UNSUPPORTED;
        header.entry_args_source = static_cast<uint32_t>(EntryArgsSource::KernelCaptureInvocation);
        header.enable_profiling_flag = 0;
        header.device_wall_data_base = 0;
        std::memcpy(owner.launch_packet.data(), &header, sizeof(header));
        SimplerKernelInvocationHeader invocation{};
        std::memcpy(&invocation, packet + LAUNCH_ENVELOPE_HEADER_BYTES, sizeof(invocation));
        owner.reference.capture_generation = next_capture_generation();
        owner.reference.context_generation = runner->kernel_context_generation();
        if (owner.reference.capture_generation == 0 || owner.reference.context_generation == 0)
            return PTO_RUNTIME_ERR_INVALID_STATE;
        void *control = runner->allocate_tensor(sizeof(KernelCaptureControl));
        if (control == nullptr) return PTO_RUNTIME_ERR_INTERNAL;
        owner.reference.control_address = reinterpret_cast<uint64_t>(control);
        KernelCaptureControl initial{};
        initial.context_generation = owner.reference.context_generation;
        initial.capture_generation = owner.reference.capture_generation;
        initial.runtime_address = reinterpret_cast<uint64_t>(header.runtime_args);
        initial.binding_epoch = header.run_result_epoch;
        initial.callable_generation = invocation.generation;
        initial.callable_id = invocation.callable_id;
        initial.tensor_count = header.entry_tensor_count;
        initial.scalar_count = header.entry_scalar_count;
        initial.launched_threads = state->runtime.get_aicpu_launch_count() > 0 ?
                                       state->runtime.get_aicpu_launch_count() :
                                       execution.launch_aicpu_num;
        initial.worker_count = execution.num_aicore;
        rc = runner->copy_to_device(control, &initial, sizeof(initial));
        if (rc != 0) return rc;
        std::memcpy(
            owner.launch_packet.data() + LAUNCH_ENVELOPE_HEADER_BYTES, &owner.reference, sizeof(owner.reference)
        );
        auto &registration = owner.registration;
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
        registration.callable_id = callable_id;
        owner.registered = true;
        return register_capture(runner, registration);
    } catch (...) {
        if (state->capture_execution == nullptr) {
            const int cleanup = simpler_finalize_run(context, runtime);
            return cleanup != 0 ? cleanup : PTO_RUNTIME_ERR_INTERNAL;
        }
        return PTO_RUNTIME_ERR_INTERNAL;
    }
}

extern "C" int simpler_kernel_capture_record(DeviceContextHandle context, RuntimeHandle runtime, void *caller_stream) {
    auto *state = capture_run(context, runtime);
    if (state == nullptr || caller_stream == nullptr) return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
    auto &owner = *state->capture_execution;
    if (!owner.registered || owner.enqueue_attempted || !state->runner->kernel_execution_state().accepts_dispatch())
        return PTO_RUNTIME_ERR_INVALID_STATE;
    int rc = state->runner->validate_current_context();
    if (rc != 0) return rc;
    aclmdlRICaptureStatus status = ACL_MODEL_RI_CAPTURE_STATUS_NONE;
    aclmdlRI model = nullptr;
    rc = aclmdlRICaptureGetInfo(caller_stream, &status, &model);
    if (rc != 0) return rc;
    if (status != ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE) return PTO_RUNTIME_ERR_INVALID_STATE;
    auto &events = state->runner->kernel_execution_state();
    KernelCaptureHandles handles{
        caller_stream,
        events.hidden_stream(KernelStreamKind::Aicpu),
        events.hidden_stream(KernelStreamKind::Aicore),
        events.event(KernelEventKind::Start),
        events.event(KernelEventKind::AicoreStart),
        events.event(KernelEventKind::AicoreDone),
        events.event(KernelEventKind::AicpuDone),
        events.event(KernelEventKind::SerialTail)
    };
    if (!handles.valid()) return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
    (void)state->runner->consume_device_fault_notices();
    if (!state->runner->accepts_new_run()) return PTO_RUNTIME_ERR_INVALID_STATE;
    KernelCaptureOps ops{};
    ops.context = state;
    ops.record = [](void *, void *event, void *stream) {
        return aclrtRecordEvent(event, stream);
    };
    ops.wait = [](void *, void *stream, void *event) {
        return aclrtStreamWaitEvent(stream, event);
    };
    ops.admit = [](void *opaque, void *stream) {
        auto &s = *static_cast<CaptureRun *>(opaque);
        auto &capture = *s.capture_execution;
        return s.runner->launch_aicpu_payload(
            stream, capture.launch_packet.data(), capture.launch_packet.size(), "simpler_aicpu_capture_admit", 1
        );
    };
    ops.launch_core = [](void *opaque, void *stream) {
        auto &s = *static_cast<CaptureRun *>(opaque);
        return s.runner->launch_aicore_kernel(
            stream, s.prepared_execution->kernel_args.args, s.capture_execution->reference.control_address,
            s.capture_execution->reference.capture_generation
        );
    };
    ops.launch_cpu = [](void *opaque, void *stream) {
        auto &s = *static_cast<CaptureRun *>(opaque);
        auto &capture = *s.capture_execution;
        return s.runner->launch_aicpu_payload(
            stream, capture.launch_packet.data(), capture.launch_packet.size(), "simpler_aicpu_exec",
            capture.registration.launched_threads
        );
    };
    state->runner->note_run_stream_identity(caller_stream);
    owner.caller_stream = caller_stream;
    LaunchProgressSink sink;
    const auto result = record_kernel_capture(handles, ops, sink);
    owner.enqueue_attempted = sink.submitted();
    owner.recorded = result.progress == LaunchProgress::Complete;
    if (result.rc != 0 && owner.enqueue_attempted) state->runner->kernel_execution_state().poison(result.rc);
    return result.rc;
}

extern "C" int
simpler_kernel_capture_wait(DeviceContextHandle context, RuntimeHandle runtime, SimplerKernelCaptureResult *out) {
    auto *state = capture_run(context, runtime);
    if (state == nullptr || out == nullptr) return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
    if (!state->capture_execution->recorded) return PTO_RUNTIME_ERR_INVALID_STATE;
    const int rc = quiesce_capture(*state);
    if (rc != 0) return rc;
    KernelCaptureControl control{};
    const int copy = observe_capture_execution(*state, &control);
    if (copy != 0) return copy;
    *out = {control.replay_epoch, control.completed_epoch, control.first_error, control.decision};
    if (control.replay_epoch == 0 || control.completed_epoch != control.replay_epoch)
        return PTO_RUNTIME_ERR_INVALID_STATE;
    return control.first_error;
}

extern "C" int simpler_kernel_capture_release(DeviceContextHandle context, RuntimeHandle runtime) {
    auto *state = capture_run(context, runtime);
    if (state == nullptr) return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
    int rc = quiesce_capture(*state);
    if (rc != 0) return rc;
    auto &owner = *state->capture_execution;
    KernelCaptureControl control{};
    rc = observe_capture_execution(*state, &control);
    if (rc != 0) return rc;
    if (owner.registered) {
        auto revoke = owner.registration;
        revoke.action = static_cast<uint32_t>(CaptureRegistrationAction::Revoke);
        rc = register_capture(state->runner, revoke);
        if (rc != 0) return rc;
        owner.registered = false;
    }
    if (owner.reference.control_address != 0) {
        rc = state->runner->release_kernel_capture_control(reinterpret_cast<void *>(owner.reference.control_address));
        if (rc != 0) return rc;
        owner.reference.control_address = 0;
    }
    if (owner.pinned_callable_hash != 0) {
        rc = state->runner->release_chip_callable_buffer(owner.pinned_callable_hash);
        if (rc != 0) return rc;
        owner.pinned_callable_hash = 0;
    }
    invalidate_capture_slot(*state->runner, state->descriptor.pipeline_slot);
    if (!state->retained_device_use.settle(state->identity())) return PTO_RUNTIME_ERR_INVALID_STATE;
    state->runner->set_kernel_capture_owner(nullptr);
    state->capture_execution.reset();
    return simpler_finalize_run(context, runtime);
}
