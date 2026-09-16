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

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <initializer_list>

#include "common/kernel_args.h"
#include "task_interface/kernel_dispatch_args.h"
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
    uint64_t context_generation{0};
    KernelArgs resident{};
};
Observer observer;
bool invocation_scope{false};
bool forbid_sync{false};
bool prepare_scope{false};
uint64_t forbidden_sync_calls{0};
uint64_t caller_stream_syncs{0};
const void *caller_streams[2]{nullptr, nullptr};
int query_override{0};
uint64_t query_override_calls{0};
uint64_t total_queries{0};
uint64_t event_waits{0}, event_records{0}, async_clears{0};
int prepare_failure{0};
uint64_t prepare_failure_calls{0};

bool fail_prepare_step(int kind) {
    if (!prepare_scope || prepare_failure != kind) return false;
    prepare_failure = 0;
    ++prepare_failure_calls;
    return true;
}

bool is_caller_stream(const void *stream) {
    for (const void *candidate : caller_streams) {
        if (candidate != nullptr && candidate == stream) return true;
    }
    return false;
}

// A sync the scope in force forbids, counted in that scope's own tally. Launch
// forbids every sync: it is pure enqueue, and a wait there is what a captured
// graph cannot contain. Registration forbids only the caller's streams — it
// synchronizes the context's own AICPU stream by contract, so that one reaches
// CANN. A null stream means device-wide, which drains the caller's streams too.
bool sync_is_forbidden(const void *stream) {
    if (forbid_sync) {
        ++forbidden_sync_calls;
        return true;
    }
    if (prepare_scope && (stream == nullptr || is_caller_stream(stream))) {
        ++caller_stream_syncs;
        return true;
    }
    return false;
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
    KernelArgs *device_args = nullptr;
    std::memcpy(&device_args, args->args, sizeof(device_args));
    note_binding(reinterpret_cast<uintptr_t>(device_args));
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
extern "C" uint64_t capture_observer_caller_syncs() { return caller_stream_syncs; }
extern "C" void capture_observer_caller_streams(uint64_t first, uint64_t second) {
    caller_streams[0] = reinterpret_cast<const void *>(first);
    caller_streams[1] = reinterpret_cast<const void *>(second);
}
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
    if (sync_is_forbidden(stream)) return -4331;
    static const auto real = reinterpret_cast<decltype(&aclrtSynchronizeStreamWithTimeout)>(
        resolve_cann_symbol("aclrtSynchronizeStreamWithTimeout")
    );
    return real == nullptr ? -4330 : real(stream, timeout);
}

extern "C" aclError aclrtSynchronizeStream(aclrtStream stream) {
    if (sync_is_forbidden(stream)) return -4331;
    static const auto real =
        reinterpret_cast<decltype(&aclrtSynchronizeStream)>(resolve_cann_symbol("aclrtSynchronizeStream"));
    return real == nullptr ? -4330 : real(stream);
}

extern "C" aclError aclrtSynchronizeDevice() {
    if (sync_is_forbidden(nullptr)) return -4331;
    static const auto real =
        reinterpret_cast<decltype(&aclrtSynchronizeDevice)>(resolve_cann_symbol("aclrtSynchronizeDevice"));
    return real == nullptr ? -4330 : real();
}

extern "C" rtError_t rtStreamSynchronize(rtStream_t stream) {
    if (sync_is_forbidden(stream)) return -4331;
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
    KernelArgs resident{};
    const int rc = copy(
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
    // Ahead of this invocation's own AICPU work, so a gate armed for it holds
    // the whole chained sequence and the caller's serial tail with it.
    if (invocation_scope) {
        if (const auto rc = capture_gate_install_if_armed(stream); rc != 0) return rc;
    }
    if (observer.armed && invocation_scope) observe_cpu(args);
    static const auto real = reinterpret_cast<decltype(&rtsLaunchCpuKernel)>(resolve_cann_symbol("rtsLaunchCpuKernel"));
    return real == nullptr ? -4330 : real(function, blocks, stream, config, args);
}
