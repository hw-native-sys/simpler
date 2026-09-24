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
#include <runtime/rt.h>
#include <runtime/rts/rts_kernel.h>

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <initializer_list>
#include <unordered_set>

#include "common/kernel_args.h"
#include "task_interface/kernel_dispatch_args.h"
#include "task_interface/tmr_kernel_context.h"
#include "task_interface/tmr_kernel_control.h"
#include "tensormap_and_ringbuffer/kernel_invocation.h"

extern "C" aclError capture_gate_install_if_armed(aclrtStream stream);

namespace {
enum class ObserverError : int {
    None,
    InvalidCoreArgs,
    InvalidCpuArgs,
    BindingChanged,
    NoPairedLaunches,
    InvalidResident,
    ResidentChanged,
};

// Each isolated scenario has one submitting host thread and one prepared context.
struct Observer {
    bool armed{false};
    bool resident_sampled{false};
    ObserverError error{ObserverError::None};
    uint64_t core_launches{0};
    uint64_t cpu_launches{0};
    uint64_t binding{0};
    uint64_t core_envelope{0};
    uint64_t context_generation{0};
    KernelArgs resident{};
};
Observer observer;
bool invocation_scope{false};
bool forbid_sync{false};
bool prepare_scope{false};
uint64_t forbidden_sync_calls{0};
int query_override{0};
uint64_t query_override_calls{0};
uint64_t total_queries{0};
uint64_t event_waits{0}, event_records{0}, async_clears{0};
int prepare_failure{0};
uint64_t prepare_failure_calls{0};
std::unordered_set<void *> large_prepare_allocations;
bool fail_large_free{false};
uint64_t failed_frees{0};
bool corrupt_next_invocation{false};
bool fail_next_invocation{false};

bool fail_prepare_step(int kind) {
    if (!prepare_scope || prepare_failure != kind) return false;
    prepare_failure = 0;
    ++prepare_failure_calls;
    return true;
}

// Registration and launch both run inside a capture the caller owns, and a
// captured graph can contain no wait, so each scope refuses every synchronize
// rather than only counting it — a refusal surfaces as the call's own status
// instead of a wait the graph silently records.
bool sync_is_forbidden() {
    if (!forbid_sync) return false;
    ++forbidden_sync_calls;
    return true;
}

void note_error(ObserverError error) {
    if (observer.error == ObserverError::None) observer.error = error;
}

void note_binding(uint64_t binding) {
    if (binding == 0 || (observer.binding != 0 && observer.binding != binding)) {
        note_error(ObserverError::BindingChanged);
    } else {
        observer.binding = binding;
    }
}

void *resolve_cann_symbol(const char *symbol) {
    if (void *address = dlsym(RTLD_NEXT, symbol)) return address;
    // The ctypes runtime loader keeps CANN dependencies in RTLD_LOCAL scope.
    for (const char *library : {"libascendcl.so", "libruntime.so"}) {
        void *handle = dlopen(library, RTLD_NOLOAD | RTLD_NOW);
        if (handle == nullptr) continue;
        void *address = dlsym(handle, symbol);
        dlclose(handle);
        if (address != nullptr) return address;
    }
    std::fprintf(stderr, "kernel_capture_observer: cannot resolve %s\n", symbol);
    return nullptr;
}

void observe_core(const rtArgsEx_t *args) {
    ++observer.core_launches;
    if (args == nullptr || args->args == nullptr || args->argsSize != sizeof(KernelArgs *)) {
        note_error(ObserverError::InvalidCoreArgs);
        return;
    }
    void *device_args = nullptr;
    std::memcpy(&device_args, args->args, sizeof(device_args));
    const auto address = reinterpret_cast<uintptr_t>(device_args);
    if (address == 0 || (observer.core_envelope != 0 && observer.core_envelope != address))
        note_error(ObserverError::BindingChanged);
    observer.core_envelope = address;
}

void observe_cpu(const rtCpuKernelArgs_t *args) {
    ++observer.cpu_launches;
    using simpler::tmr::TmrBindingRef;
    if (args == nullptr || args->baseArgs.args == nullptr ||
        args->baseArgs.argsSize < sizeof(SimplerKernelDispatchArgs) + sizeof(TmrBindingRef)) {
        note_error(ObserverError::InvalidCpuArgs);
        return;
    }
    const auto *bytes = static_cast<const uint8_t *>(args->baseArgs.args);
    SimplerKernelDispatchArgs prefix{};
    TmrBindingRef reference{};
    std::memcpy(&prefix, bytes, sizeof(prefix));
    std::memcpy(&reference, bytes + sizeof(prefix), sizeof(reference));
    if (prefix.packet_bytes != args->baseArgs.argsSize || prefix.invocation.mode != SIMPLER_MODE_KERNEL ||
        prefix.invocation.payload_bytes != args->baseArgs.argsSize - sizeof(prefix) ||
        prefix.binding_address != reference.device_binding_addr || prefix.context_generation == 0 ||
        prefix.context_generation != reference.context_generation) {
        note_error(ObserverError::InvalidCpuArgs);
        return;
    }
    note_binding(prefix.binding_address);
    if (observer.context_generation != 0 && observer.context_generation != prefix.context_generation)
        note_error(ObserverError::BindingChanged);
    observer.context_generation = prefix.context_generation;
}
}  // namespace

extern "C" void capture_observer_begin() {
    observer = {};
    observer.armed = true;
}

extern "C" void capture_observer_guard_sync(int enabled) { forbid_sync = enabled != 0; }
extern "C" void capture_observer_prepare_scope(int enabled) { prepare_scope = enabled != 0; }
extern "C" void capture_observer_invocation_scope(int enabled) { invocation_scope = enabled != 0; }
extern "C" uint64_t capture_observer_sync_calls() { return forbidden_sync_calls; }
extern "C" void capture_observer_override_query(int kind) {
    query_override = kind;
    query_override_calls = 0;
}
extern "C" uint64_t capture_observer_query_calls() { return query_override_calls; }
extern "C" uint64_t capture_observer_total_queries() { return total_queries; }
extern "C" uint64_t capture_observer_waits() { return event_waits; }
extern "C" uint64_t capture_observer_records() { return event_records; }
extern "C" uint64_t capture_observer_clears() { return async_clears; }
extern "C" void capture_observer_fail_prepare(int kind) {
    prepare_failure = kind;
    prepare_failure_calls = 0;
}
extern "C" uint64_t capture_observer_prepare_failures() { return prepare_failure_calls; }
extern "C" void capture_observer_fail_large_free(int enabled) { fail_large_free = enabled != 0; }
extern "C" uint64_t capture_observer_failed_frees() { return failed_frees; }
extern "C" void capture_observer_corrupt_next_invocation() { corrupt_next_invocation = true; }
extern "C" void capture_observer_fail_next_invocation() { fail_next_invocation = true; }

extern "C" rtError_t rtMalloc(void **address, uint64_t bytes, uint32_t kind, uint16_t module) {
    static const auto real = reinterpret_cast<decltype(&rtMalloc)>(resolve_cann_symbol("rtMalloc"));
    const auto rc = real == nullptr ? -4330 : real(address, bytes, kind, module);
    if (rc == 0 && prepare_scope && bytes >= 1024 * 1024) large_prepare_allocations.insert(*address);
    return rc;
}

extern "C" rtError_t rtFree(void *address) {
    if (fail_large_free && large_prepare_allocations.count(address) != 0) {
        ++failed_frees;
        return -4334;
    }
    static const auto real = reinterpret_cast<decltype(&rtFree)>(resolve_cann_symbol("rtFree"));
    const auto rc = real == nullptr ? -4330 : real(address);
    if (rc == 0) large_prepare_allocations.erase(address);
    return rc;
}

extern "C" aclError aclrtQueryEventStatus(aclrtEvent event, aclrtEventRecordedStatus *status) {
    if (invocation_scope) ++total_queries;
    if (invocation_scope && query_override != 0) {
        const int kind = query_override;
        query_override = 0;
        ++query_override_calls;
        if (kind == 1) return -4332;
        *status = ACL_EVENT_RECORDED_STATUS_NOT_READY;
        return 0;
    }
    static const auto real =
        reinterpret_cast<decltype(&aclrtQueryEventStatus)>(resolve_cann_symbol("aclrtQueryEventStatus"));
    return real == nullptr ? -4330 : real(event, status);
}

extern "C" aclError aclrtStreamWaitEvent(aclrtStream stream, aclrtEvent event) {
    if (invocation_scope) ++event_waits;
    static const auto real =
        reinterpret_cast<decltype(&aclrtStreamWaitEvent)>(resolve_cann_symbol("aclrtStreamWaitEvent"));
    return real == nullptr ? -4330 : real(stream, event);
}

extern "C" aclError aclrtRecordEvent(aclrtEvent event, aclrtStream stream) {
    if (invocation_scope) ++event_records;
    static const auto real = reinterpret_cast<decltype(&aclrtRecordEvent)>(resolve_cann_symbol("aclrtRecordEvent"));
    return real == nullptr ? -4330 : real(event, stream);
}

extern "C" aclError aclrtMemsetAsync(void *device, size_t maximum, int32_t value, size_t bytes, aclrtStream stream) {
    if (invocation_scope) ++async_clears;
    static const auto real = reinterpret_cast<decltype(&aclrtMemsetAsync)>(resolve_cann_symbol("aclrtMemsetAsync"));
    return real == nullptr ? -4330 : real(device, maximum, value, bytes, stream);
}

extern "C" aclError aclrtSynchronizeStreamWithTimeout(aclrtStream stream, int32_t timeout) {
    if (sync_is_forbidden()) return -4331;
    static const auto real = reinterpret_cast<decltype(&aclrtSynchronizeStreamWithTimeout)>(
        resolve_cann_symbol("aclrtSynchronizeStreamWithTimeout")
    );
    return real == nullptr ? -4330 : real(stream, timeout);
}

extern "C" aclError aclrtSynchronizeStream(aclrtStream stream) {
    if (sync_is_forbidden()) return -4331;
    static const auto real =
        reinterpret_cast<decltype(&aclrtSynchronizeStream)>(resolve_cann_symbol("aclrtSynchronizeStream"));
    return real == nullptr ? -4330 : real(stream);
}

extern "C" aclError aclrtSynchronizeDevice() {
    if (sync_is_forbidden()) return -4331;
    static const auto real =
        reinterpret_cast<decltype(&aclrtSynchronizeDevice)>(resolve_cann_symbol("aclrtSynchronizeDevice"));
    return real == nullptr ? -4330 : real();
}

extern "C" rtError_t rtStreamSynchronize(rtStream_t stream) {
    if (sync_is_forbidden()) return -4331;
    static const auto real =
        reinterpret_cast<decltype(&rtStreamSynchronize)>(resolve_cann_symbol("rtStreamSynchronize"));
    return real == nullptr ? -4330 : real(stream);
}

extern "C" int capture_observer_status() {
    if (observer.error != ObserverError::None) return static_cast<int>(observer.error);
    if (!observer.armed || observer.core_launches == 0 || observer.core_launches != observer.cpu_launches)
        return static_cast<int>(ObserverError::NoPairedLaunches);
    return 0;
}

extern "C" uint64_t capture_observer_core_launches() { return observer.core_launches; }
extern "C" uint64_t capture_observer_cpu_launches() { return observer.cpu_launches; }
extern "C" uint64_t capture_observer_kernel_args() { return observer.binding; }
extern "C" uint64_t capture_observer_runtime_args() {
    return reinterpret_cast<uintptr_t>(observer.resident.runtime_args);
}
extern "C" uint64_t capture_observer_regs() { return observer.resident.regs; }

// The caller establishes quiescence outside capture before this D2H read.
extern "C" int capture_observer_check_resident() {
    const int status = capture_observer_status();
    if (status != 0) return status;
    const auto copy = reinterpret_cast<decltype(&aclrtMemcpy)>(resolve_cann_symbol("aclrtMemcpy"));
    if (copy == nullptr) return -4330;
    simpler::tmr::TmrKernelAicoreArgs envelope{};
    int rc = copy(
        &envelope, sizeof(envelope), reinterpret_cast<const void *>(observer.core_envelope), sizeof(envelope),
        ACL_MEMCPY_DEVICE_TO_HOST
    );
    if (rc != 0) return rc;
    if (envelope.resident_kernel_args != observer.binding) {
        note_error(ObserverError::BindingChanged);
        return static_cast<int>(observer.error);
    }
    KernelArgs resident{};
    rc = copy(
        &resident, sizeof(resident), reinterpret_cast<const void *>(observer.binding), sizeof(resident),
        ACL_MEMCPY_DEVICE_TO_HOST
    );
    if (rc != 0) return rc;
    if (resident.runtime_args == nullptr || resident.regs == 0) {
        note_error(ObserverError::InvalidResident);
    } else if (observer.resident_sampled &&
               (resident.runtime_args != observer.resident.runtime_args || resident.regs != observer.resident.regs ||
                resident.ffts_base_addr != observer.resident.ffts_base_addr)) {
        note_error(ObserverError::ResidentChanged);
    } else {
        observer.resident = resident;
        observer.resident_sampled = true;
    }
    return static_cast<int>(observer.error);
}

extern "C" int capture_observer_failure_reported() {
    using namespace simpler::tmr;
    const auto copy = reinterpret_cast<decltype(&aclrtMemcpy)>(resolve_cann_symbol("aclrtMemcpy"));
    if (copy == nullptr || observer.core_envelope == 0) return -1;
    const auto read = [&](void *out, uint64_t address, size_t bytes) {
        return copy(out, bytes, reinterpret_cast<const void *>(address), bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    };
    TmrKernelAicoreArgs envelope{};
    TmrKernelContextDescriptor descriptor{};
    TmrLaunchControl control{};
    if (read(&envelope, observer.core_envelope, sizeof(envelope)) != 0 ||
        read(&descriptor, envelope.context_descriptor, sizeof(descriptor)) != 0 ||
        read(&control, descriptor.control_address, sizeof(control)) != 0)
        return -2;
    if (control.completion != static_cast<uint32_t>(TmrCompletion::Complete) || control.runtime_status == 0 ||
        control.cleanup_status != 0 || descriptor.worker_count <= 0) {
        std::fprintf(
            stderr, "failure control: completion=%u runtime=%d cleanup=%d epoch=%" PRIu64 " workers=%d\n",
            control.completion, control.runtime_status, control.cleanup_status, control.round_epoch,
            descriptor.worker_count
        );
        return -3;
    }
    return 0;
}

extern "C" rtError_t rtKernelLaunchWithHandleV2(
    void *handle, const uint64_t tiling_key, uint32_t blocks, rtArgsEx_t *args, rtSmDesc_t *sm_desc, rtStream_t stream,
    const rtTaskCfgInfo_t *config
) {
    if (observer.armed && invocation_scope) observe_core(args);
    static const auto real =
        reinterpret_cast<decltype(&rtKernelLaunchWithHandleV2)>(resolve_cann_symbol("rtKernelLaunchWithHandleV2"));
    return real == nullptr ? -4330 : real(handle, tiling_key, blocks, args, sm_desc, stream, config);
}

extern "C" rtError_t rtsLaunchCpuKernel(
    const rtFuncHandle function, uint32_t blocks, rtStream_t stream, const rtKernelLaunchCfg_t *config,
    rtCpuKernelArgs_t *args
) {
    if (fail_prepare_step(1)) return -4333;
    if (invocation_scope && fail_next_invocation) {
        fail_next_invocation = false;
        return -4334;
    }
    // Ahead of this invocation's own AICPU work, so a gate armed for it holds
    // the whole chained sequence and the caller's serial tail with it.
    if (invocation_scope) {
        if (const auto rc = capture_gate_install_if_armed(stream); rc != 0) return rc;
    }
    // Context registration/revoke nodes carry only trusted context identity;
    // they are not invocations and must not be counted as dispatches.
    if (observer.armed && invocation_scope && args != nullptr &&
        args->baseArgs.argsSize != sizeof(simpler::tmr::TmrContextRegistrationArgs))
        observe_cpu(args);
    static const auto real = reinterpret_cast<decltype(&rtsLaunchCpuKernel)>(resolve_cann_symbol("rtsLaunchCpuKernel"));
    if (real != nullptr && invocation_scope && corrupt_next_invocation && args != nullptr &&
        args->baseArgs.argsSize >= sizeof(SimplerKernelDispatchArgs)) {
        corrupt_next_invocation = false;
        auto *bytes = static_cast<unsigned char *>(args->baseArgs.args);
        SimplerKernelDispatchArgs original{};
        std::memcpy(&original, bytes, sizeof(original));
        auto invalid = original;
        ++invalid.context_generation;
        std::memcpy(bytes, &invalid, sizeof(invalid));
        const auto rc = real(function, blocks, stream, config, args);
        // CANN owns the copied packet after the native API returns.
        std::memcpy(bytes, &original, sizeof(original));
        return rc;
    }
    return real == nullptr ? -4330 : real(function, blocks, stream, config, args);
}
