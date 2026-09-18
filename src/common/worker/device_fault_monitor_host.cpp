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
/**
 * @file device_fault_monitor_host.cpp
 * @brief The process's device-fault monitor, and its driver trampoline.
 *
 * See the header for why this lives here rather than in a runtime SO.
 *
 * **The driver API is reached by `dlsym`, not by linking.** This module is
 * compiled into the Python extension, which builds and runs on hosts with no
 * CANN at all (sim, and the macOS packaging job), so it cannot have
 * `libascendcl` as a link dependency. Resolving at first use costs one lookup
 * and makes "no driver here" an ordinary answer rather than a load failure.
 *
 * The library is opened `RTLD_NODELETE` and never closed, for the same reason
 * the trampoline lives here: the accessors it calls are inside `libascendcl`,
 * and a runtime SO's `dlclose` would otherwise drop the last reference to it
 * while the driver may still call back.
 */

#include "device_fault_monitor_host.h"

#include <dlfcn.h>
#include <pthread.h>
#include <unistd.h>

#include <mutex>

namespace {

/** The driver entry points this module needs, resolved once. */
struct AclFaultApi {
    int (*set_callback)(void (*)(void *)){nullptr};
    uint32_t (*device_id)(void *){nullptr};
    uint32_t (*stream_id)(void *){nullptr};
    uint32_t (*task_id)(void *){nullptr};
    uint32_t (*error_code)(void *){nullptr};
    uint32_t (*thread_id)(void *){nullptr};

    bool complete() const {
        return set_callback != nullptr && device_id != nullptr && stream_id != nullptr && task_id != nullptr &&
               error_code != nullptr && thread_id != nullptr;
    }
};

template <typename T>
void resolve(void *handle, const char *name, T *out) {
    *out = reinterpret_cast<T>(dlsym(handle, name));
}

const AclFaultApi &acl_fault_api() {
    static AclFaultApi *api = [] {
        auto *resolved = new AclFaultApi;
        // RTLD_NODELETE keeps the accessor code mapped for the process's life,
        // so a runtime SO's dlclose cannot drop the last reference to it while
        // the driver may still call this module's trampoline. RTLD_NOLOAD
        // first: when a runtime SO already linked the library, reuse that image
        // rather than introducing a second one.
        void *handle = dlopen("libascendcl.so", RTLD_NOW | RTLD_NOLOAD | RTLD_NODELETE);
        if (handle == nullptr) {
            handle = dlopen("libascendcl.so", RTLD_NOW | RTLD_NODELETE);
        }
        if (handle == nullptr) return resolved;  // no driver on this host
        resolve(handle, "aclrtSetExceptionInfoCallback", &resolved->set_callback);
        resolve(handle, "aclrtGetDeviceIdFromExceptionInfo", &resolved->device_id);
        resolve(handle, "aclrtGetStreamIdFromExceptionInfo", &resolved->stream_id);
        resolve(handle, "aclrtGetTaskIdFromExceptionInfo", &resolved->task_id);
        resolve(handle, "aclrtGetErrorCodeFromExceptionInfo", &resolved->error_code);
        resolve(handle, "aclrtGetThreadIdFromExceptionInfo", &resolved->thread_id);
        return resolved;
    }();
    return *api;
}

/**
 * The function pointer the driver retains. It records and returns: no lock, no
 * allocation, no logging. Logging here would put a driver thread behind the
 * host logger's queue, and a notification's arrival is not a good moment to
 * discover the logger is busy — the host reads and reports these later, from a
 * thread of its own.
 */
void on_device_fault(void *info) {
    if (info == nullptr) return;
    const AclFaultApi &api = acl_fault_api();
    if (!api.complete()) return;
    DeviceFaultNotice notice;
    notice.device_id = api.device_id(info);
    notice.stream_id = api.stream_id(info);
    notice.task_id = api.task_id(info);
    notice.error_code = api.error_code(info);
    notice.thread_id = api.thread_id(info);
    resident_device_fault_monitor().report(notice);
}

void monitor_before_fork() { resident_device_fault_monitor().before_fork(); }
void monitor_after_fork_in_parent() { resident_device_fault_monitor().after_fork_in_parent(); }
void monitor_after_fork_in_child() { resident_device_fault_monitor().after_fork_in_child(); }

/** Signature of the setter every host runtime built from this tree exports. */
using BindMonitorFn = void (*)(void *);

}  // namespace

DeviceFaultMonitor &resident_device_fault_monitor() {
    // Leaked on purpose, and never given a destructor. The driver may call
    // `on_device_fault` at any point after a retire, including after the last
    // runner in the process has finalized, so the storage the trampoline
    // writes has to outlive every owner of it — which a function-local object
    // with a registered destructor would not.
    static DeviceFaultMonitor *monitor = [] {
        auto *created = new DeviceFaultMonitor(
            DeviceFaultMonitor::Ops{
                []() {
                    const AclFaultApi &api = acl_fault_api();
                    // No driver on this host: nothing to register, and nothing
                    // is wrong. The caller treats a non-zero rc as "no
                    // notifications available" rather than as a failure.
                    if (!api.complete()) return -1;
                    return api.set_callback(&on_device_fault);
                },
                []() {
                    const AclFaultApi &api = acl_fault_api();
                    if (!api.complete()) return 0;
                    return api.set_callback(nullptr);
                },
                []() {
                    return static_cast<long>(getpid());
                },
            }
        );
        // A child inherits this object's bytes, including its mutex — and a
        // mutex held by a thread that does not exist in the child can never be
        // taken there, so a pid check made *under* that mutex would never run.
        // Holding it across the fork and releasing it on both sides is what
        // makes the child's first operation reachable at all.
        (void)pthread_atfork(monitor_before_fork, monitor_after_fork_in_parent, monitor_after_fork_in_child);
        return created;
    }();
    return *monitor;
}

int bind_loaded_device_fault_monitor(void *dl_handle, const char **error) {
    if (dl_handle == nullptr) {
        if (error != nullptr) *error = "null module handle";
        return -1;
    }
    dlerror();
    auto bind = reinterpret_cast<BindMonitorFn>(dlsym(dl_handle, "simpler_bind_device_fault_monitor"));
    const char *dlsym_error = dlerror();
    if (bind == nullptr || dlsym_error != nullptr) {
        if (error != nullptr) *error = dlsym_error != nullptr ? dlsym_error : "symbol not found";
        return -1;
    }
    bind(&resident_device_fault_monitor());
    return 0;
}
