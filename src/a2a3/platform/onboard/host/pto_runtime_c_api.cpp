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
 * PTO Runtime C API - Implementation (On-board Hardware)
 *
 * Platform-specific implementation of the public C API declared in
 * src/common/worker/pto_runtime_c_api.h.  Uses real Ascend device execution.
 */

#include "pto_runtime_c_api.h"

#include "callable.h"
#include "task_args.h"

#include <inttypes.h>
#include <dlfcn.h>
#include <initializer_list>
#include <pthread.h>
#include <vector>

#include "common/unified_log.h"
#include "device_runner.h"
#include "host/raii_scope_guard.h"
#include "runtime.h"

namespace {

using RtMallocHostFn = int (*)(void **, uint64_t, uint32_t);
using RtFreeHostFn = int (*)(void *);
using RtHostRegisterFn = int (*)(void *, uint64_t, uint32_t, void **);
using RtHostUnregisterFn = int (*)(void *);
using AclMallocHostFn = int (*)(void **, size_t);
using AclFreeHostFn = int (*)(void *);
using AclHostRegisterFn = int (*)(void *, uint64_t, uint32_t, void **);
using AclHostUnregisterFn = int (*)(void *);
using GetDeviceFn = int (*)(int32_t *);
using DirectSetDeviceFn = int (*)(int32_t);

template <typename Fn>
Fn resolve_symbol(const char **resolved_name, std::initializer_list<const char *> names) {
    for (const char *name : names) {
        dlerror();
        void *sym = dlsym(RTLD_DEFAULT, name);
        const char *err = dlerror();
        if (err == nullptr && sym != nullptr) {
            if (resolved_name != nullptr) {
                *resolved_name = name;
            }
            return reinterpret_cast<Fn>(sym);
        }
    }
    if (resolved_name != nullptr) {
        *resolved_name = nullptr;
    }
    return nullptr;
}

static constexpr uint32_t kHostRegisterMappedFlag =
#if defined(RT_HOST_REGISTER_MAPPED)
    RT_HOST_REGISTER_MAPPED;
#elif defined(ACL_HOST_REGISTER_MAPPED)
    ACL_HOST_REGISTER_MAPPED;
#else
    0U;
#endif

int ensure_host_api_device_ready(DeviceContextHandle ctx, int device_id) {
    if (ctx == nullptr) {
        return -1;
    }
    try {
        return static_cast<DeviceRunner *>(ctx)->ensure_device_set(device_id);
    } catch (...) {
        return -1;
    }
}

int ensure_current_device_for_share_mem(uint32_t device_id) {
    const char *symbol_name = nullptr;
    if (GetDeviceFn get_device_fn = resolve_symbol<GetDeviceFn>(&symbol_name, {"aclrtGetDevice", "rtGetDevice"})) {
        int32_t current_device = -1;
        int rc = get_device_fn(&current_device);
        if (rc != 0) {
            LOG_INFO(
                "ensure_current_device_for_share_mem: %s failed rc=%d, trying to set device to %u",
                symbol_name, rc, device_id
            );
            if (DirectSetDeviceFn set_device_fn =
                    resolve_symbol<DirectSetDeviceFn>(&symbol_name, {"rtSetDevice", "aclrtSetDevice"})) {
                rc = set_device_fn(static_cast<int32_t>(device_id));
                if (rc != 0) {
                    LOG_ERROR(
                        "ensure_current_device_for_share_mem: %s(%u) failed: rc=%d", symbol_name, device_id, rc
                    );
                    return rc;
                }
                return 0;
            }
            LOG_ERROR("ensure_current_device_for_share_mem: missing symbols rtSetDevice / aclrtSetDevice");
            return rc;
        }

        if (current_device != static_cast<int32_t>(device_id)) {
            LOG_ERROR(
                "ensure_current_device_for_share_mem: current device %d does not match requested device %u",
                static_cast<int>(current_device), device_id
            );
            return -1;
        }
        return 0;
    }

    LOG_ERROR("ensure_current_device_for_share_mem: missing symbols aclrtGetDevice / rtGetDevice");
    return -1;
}

}  // namespace

extern "C" {

/* ===========================================================================
 * Runtime Implementation Functions (defined in runtime_maker.cpp)
 * =========================================================================== */
int init_runtime_impl(Runtime *runtime, const ChipCallable *callable, const ChipStorageTaskArgs *orch_args);
int validate_runtime_impl(Runtime *runtime);

/* ===========================================================================
 * Per-thread DeviceRunner binding (set by run_runtime, read by HostApi wrappers)
 * =========================================================================== */

static pthread_key_t g_runner_key;
static pthread_once_t g_runner_key_once = PTHREAD_ONCE_INIT;
static void create_runner_key() { pthread_key_create(&g_runner_key, nullptr); }

static DeviceRunner *current_runner() { return static_cast<DeviceRunner *>(pthread_getspecific(g_runner_key)); }

/* ===========================================================================
 * Internal device-memory functions (used via Runtime.host_api, NOT dlsym'd)
 * =========================================================================== */

static void *device_malloc(size_t size) {
    try {
        return current_runner()->allocate_tensor(size);
    } catch (...) {
        return NULL;
    }
}

static void device_free(void *dev_ptr) {
    if (dev_ptr == NULL) return;
    try {
        current_runner()->free_tensor(dev_ptr);
    } catch (...) {}
}

static int copy_to_device(void *dev_ptr, const void *host_ptr, size_t size) {
    if (dev_ptr == NULL || host_ptr == NULL) return -1;
    try {
        return current_runner()->copy_to_device(dev_ptr, host_ptr, size);
    } catch (...) {
        return -1;
    }
}

static int copy_from_device(void *host_ptr, const void *dev_ptr, size_t size) {
    if (host_ptr == NULL || dev_ptr == NULL) return -1;
    try {
        return current_runner()->copy_from_device(host_ptr, dev_ptr, size);
    } catch (...) {
        return -1;
    }
}

static uint64_t upload_kernel_binary_wrapper(int func_id, const uint8_t *bin_data, size_t bin_size) {
    try {
        return current_runner()->upload_kernel_binary(func_id, bin_data, bin_size);
    } catch (...) {
        return 0;
    }
}

static void remove_kernel_binary_wrapper(int func_id) {
    try {
        current_runner()->remove_kernel_binary(func_id);
    } catch (...) {}
}

/* ===========================================================================
 * Public C API (resolved by ChipWorker via dlsym)
 * =========================================================================== */

DeviceContextHandle create_device_context(void) {
    try {
        return static_cast<DeviceContextHandle>(new DeviceRunner());
    } catch (...) {
        return NULL;
    }
}

void destroy_device_context(DeviceContextHandle ctx) { delete static_cast<DeviceRunner *>(ctx); }

size_t get_runtime_size(void) { return sizeof(Runtime); }

int set_device(DeviceContextHandle ctx, int device_id) {
    if (ctx == NULL) return -1;
    try {
        return static_cast<DeviceRunner *>(ctx)->ensure_device_set(device_id);
    } catch (...) {
        return -1;
    }
}

void *host_malloc(DeviceContextHandle ctx, size_t size) {
    (void)ctx;
    if (size == 0) {
        LOG_ERROR("host_malloc: size must be > 0");
        return NULL;
    }

    void *host_ptr = nullptr;
    const char *symbol_name = nullptr;

    if (RtMallocHostFn fn = resolve_symbol<RtMallocHostFn>(&symbol_name, {"rtMallocHost"})) {
        int rc = fn(&host_ptr, static_cast<uint64_t>(size), 0U);
        if (rc != 0 || host_ptr == nullptr) {
            LOG_ERROR("host_malloc via %s failed: rc=%d size=%zu", symbol_name, rc, size);
            return NULL;
        }
        LOG_INFO("host_malloc: %zu bytes at %p via %s", size, host_ptr, symbol_name);
        return host_ptr;
    }

    if (AclMallocHostFn fn = resolve_symbol<AclMallocHostFn>(&symbol_name, {"aclrtMallocHost"})) {
        int rc = fn(&host_ptr, size);
        if (rc != 0 || host_ptr == nullptr) {
            LOG_ERROR("host_malloc via %s failed: rc=%d size=%zu", symbol_name, rc, size);
            return NULL;
        }
        LOG_INFO("host_malloc: %zu bytes at %p via %s", size, host_ptr, symbol_name);
        return host_ptr;
    }

    LOG_ERROR("host_malloc: missing symbols rtMallocHost / aclrtMallocHost");
    return NULL;
}

void host_free(DeviceContextHandle ctx, void *host_ptr) {
    (void)ctx;
    if (host_ptr == NULL) return;

    const char *symbol_name = nullptr;
    if (RtFreeHostFn fn = resolve_symbol<RtFreeHostFn>(&symbol_name, {"rtFreeHost"})) {
        int rc = fn(host_ptr);
        if (rc != 0) {
            LOG_ERROR("host_free via %s failed: rc=%d ptr=%p", symbol_name, rc, host_ptr);
            return;
        }
        LOG_INFO("host_free: ptr=%p via %s", host_ptr, symbol_name);
        return;
    }

    if (AclFreeHostFn fn = resolve_symbol<AclFreeHostFn>(&symbol_name, {"aclrtFreeHost"})) {
        int rc = fn(host_ptr);
        if (rc != 0) {
            LOG_ERROR("host_free via %s failed: rc=%d ptr=%p", symbol_name, rc, host_ptr);
            return;
        }
        LOG_INFO("host_free: ptr=%p via %s", host_ptr, symbol_name);
        return;
    }

    LOG_ERROR("host_free: missing symbols rtFreeHost / aclrtFreeHost");
}

int host_register_mapped(DeviceContextHandle ctx, void *host_ptr, size_t size, int device_id, void **dev_ptr) {
    if (host_ptr == NULL || dev_ptr == NULL) return -1;
    *dev_ptr = nullptr;

    int rc = ensure_host_api_device_ready(ctx, device_id);
    if (rc != 0) {
        LOG_ERROR("host_register_mapped: ensure_device_set(%d) failed: rc=%d", device_id, rc);
        return rc;
    }

    const char *symbol_name = nullptr;
    if (RtHostRegisterFn fn = resolve_symbol<RtHostRegisterFn>(&symbol_name, {"rtsHostRegister", "rtHostRegister"})) {
        rc = fn(host_ptr, static_cast<uint64_t>(size), kHostRegisterMappedFlag, dev_ptr);
        if (rc != 0 || *dev_ptr == nullptr) {
            LOG_ERROR(
                "host_register_mapped via %s failed: rc=%d host=%p size=%zu flag=%u", symbol_name, rc, host_ptr, size,
                kHostRegisterMappedFlag
            );
            return (rc != 0) ? rc : -1;
        }
        LOG_INFO(
            "host_register_mapped: host=%p dev=%p size=%zu via %s flag=%u", host_ptr, *dev_ptr, size, symbol_name,
            kHostRegisterMappedFlag
        );
        return 0;
    }

    if (AclHostRegisterFn fn = resolve_symbol<AclHostRegisterFn>(&symbol_name, {"aclrtHostRegister"})) {
        rc = fn(host_ptr, static_cast<uint64_t>(size), kHostRegisterMappedFlag, dev_ptr);
        if (rc != 0 || *dev_ptr == nullptr) {
            LOG_ERROR(
                "host_register_mapped via %s failed: rc=%d host=%p size=%zu flag=%u", symbol_name, rc, host_ptr, size,
                kHostRegisterMappedFlag
            );
            return (rc != 0) ? rc : -1;
        }
        LOG_INFO(
            "host_register_mapped: host=%p dev=%p size=%zu via %s flag=%u", host_ptr, *dev_ptr, size, symbol_name,
            kHostRegisterMappedFlag
        );
        return 0;
    }

    LOG_ERROR("host_register_mapped: missing symbols rtsHostRegister / rtHostRegister / aclrtHostRegister");
    return -1;
}

int host_unregister_mapped(DeviceContextHandle ctx, void *host_ptr, int device_id) {
    if (host_ptr == NULL) return -1;

    int rc = ensure_host_api_device_ready(ctx, device_id);
    if (rc != 0) {
        LOG_ERROR("host_unregister_mapped: ensure_device_set(%d) failed: rc=%d", device_id, rc);
        return rc;
    }

    const char *symbol_name = nullptr;
    if (RtHostUnregisterFn fn =
            resolve_symbol<RtHostUnregisterFn>(&symbol_name, {"rtsHostUnregister", "rtHostUnregister"})) {
        rc = fn(host_ptr);
        if (rc != 0) {
            LOG_ERROR("host_unregister_mapped via %s failed: rc=%d host=%p", symbol_name, rc, host_ptr);
            return rc;
        }
        LOG_INFO("host_unregister_mapped: host=%p via %s", host_ptr, symbol_name);
        return 0;
    }

    if (AclHostUnregisterFn fn = resolve_symbol<AclHostUnregisterFn>(&symbol_name, {"aclrtHostUnregister"})) {
        rc = fn(host_ptr);
        if (rc != 0) {
            LOG_ERROR("host_unregister_mapped via %s failed: rc=%d host=%p", symbol_name, rc, host_ptr);
            return rc;
        }
        LOG_INFO("host_unregister_mapped: host=%p via %s", host_ptr, symbol_name);
        return 0;
    }

    LOG_ERROR("host_unregister_mapped: missing symbols rtsHostUnregister / rtHostUnregister / aclrtHostUnregister");
    return -1;
}

int mallocHostDeviceShareMem(uint32_t deviceId, uint64_t size, void **hostPtr, void **devPtr) {
    if (hostPtr == NULL || devPtr == NULL || size == 0) {
        return -1;
    }

    *hostPtr = nullptr;
    *devPtr = nullptr;

    int rc = ensure_current_device_for_share_mem(deviceId);
    if (rc != 0) {
        return rc;
    }

    void *allocated_host_ptr = nullptr;
    const char *symbol_name = nullptr;
    if (RtMallocHostFn malloc_fn = resolve_symbol<RtMallocHostFn>(&symbol_name, {"rtMallocHost"})) {
        rc = malloc_fn(&allocated_host_ptr, size, 0U);
        if (rc != 0 || allocated_host_ptr == nullptr) {
            LOG_ERROR("mallocHostDeviceShareMem via %s failed on rtMallocHost: rc=%d size=%" PRIu64, symbol_name, rc, size);
            return (rc != 0) ? rc : -1;
        }
    } else if (AclMallocHostFn malloc_fn = resolve_symbol<AclMallocHostFn>(&symbol_name, {"aclrtMallocHost"})) {
        rc = malloc_fn(&allocated_host_ptr, static_cast<size_t>(size));
        if (rc != 0 || allocated_host_ptr == nullptr) {
            LOG_ERROR("mallocHostDeviceShareMem via %s failed on aclrtMallocHost: rc=%d size=%" PRIu64, symbol_name, rc, size);
            return (rc != 0) ? rc : -1;
        }
    } else {
        LOG_ERROR("mallocHostDeviceShareMem: missing symbols rtMallocHost / aclrtMallocHost");
        return -1;
    }

    if (RtHostRegisterFn register_fn =
            resolve_symbol<RtHostRegisterFn>(&symbol_name, {"rtsHostRegister", "rtHostRegister"})) {
        rc = register_fn(allocated_host_ptr, size, kHostRegisterMappedFlag, devPtr);
        if (rc != 0 || *devPtr == nullptr) {
            LOG_ERROR(
                "mallocHostDeviceShareMem via %s failed on host register: rc=%d host=%p size=%" PRIu64 " flag=%u",
                symbol_name, rc, allocated_host_ptr, size, kHostRegisterMappedFlag
            );
            if (RtFreeHostFn free_fn = resolve_symbol<RtFreeHostFn>(nullptr, {"rtFreeHost"})) {
                free_fn(allocated_host_ptr);
            } else if (AclFreeHostFn free_fn = resolve_symbol<AclFreeHostFn>(nullptr, {"aclrtFreeHost"})) {
                free_fn(allocated_host_ptr);
            }
            return (rc != 0) ? rc : -1;
        }
    } else if (AclHostRegisterFn register_fn =
                   resolve_symbol<AclHostRegisterFn>(&symbol_name, {"aclrtHostRegister"})) {
        rc = register_fn(allocated_host_ptr, size, kHostRegisterMappedFlag, devPtr);
        if (rc != 0 || *devPtr == nullptr) {
            LOG_ERROR(
                "mallocHostDeviceShareMem via %s failed on host register: rc=%d host=%p size=%" PRIu64 " flag=%u",
                symbol_name, rc, allocated_host_ptr, size, kHostRegisterMappedFlag
            );
            if (RtFreeHostFn free_fn = resolve_symbol<RtFreeHostFn>(nullptr, {"rtFreeHost"})) {
                free_fn(allocated_host_ptr);
            } else if (AclFreeHostFn free_fn = resolve_symbol<AclFreeHostFn>(nullptr, {"aclrtFreeHost"})) {
                free_fn(allocated_host_ptr);
            }
            return (rc != 0) ? rc : -1;
        }
    } else {
        LOG_ERROR("mallocHostDeviceShareMem: missing symbols rtsHostRegister / rtHostRegister / aclrtHostRegister");
        if (RtFreeHostFn free_fn = resolve_symbol<RtFreeHostFn>(nullptr, {"rtFreeHost"})) {
            free_fn(allocated_host_ptr);
        } else if (AclFreeHostFn free_fn = resolve_symbol<AclFreeHostFn>(nullptr, {"aclrtFreeHost"})) {
            free_fn(allocated_host_ptr);
        }
        return -1;
    }

    *hostPtr = allocated_host_ptr;
    LOG_INFO(
        "mallocHostDeviceShareMem: device=%u host=%p dev=%p size=%" PRIu64, deviceId, *hostPtr, *devPtr, size
    );
    return 0;
}

int freeHostDeviceShareMem(uint32_t deviceId, void *hostPtr) {
    if (hostPtr == NULL) {
        return 0;
    }

    int rc = ensure_current_device_for_share_mem(deviceId);
    if (rc != 0) {
        return rc;
    }

    const char *symbol_name = nullptr;
    if (RtHostUnregisterFn unregister_fn =
            resolve_symbol<RtHostUnregisterFn>(&symbol_name, {"rtsHostUnregister", "rtHostUnregister"})) {
        rc = unregister_fn(hostPtr);
        if (rc != 0) {
            LOG_ERROR("freeHostDeviceShareMem via %s failed on unregister: rc=%d host=%p", symbol_name, rc, hostPtr);
            return rc;
        }
    } else if (AclHostUnregisterFn unregister_fn =
                   resolve_symbol<AclHostUnregisterFn>(&symbol_name, {"aclrtHostUnregister"})) {
        rc = unregister_fn(hostPtr);
        if (rc != 0) {
            LOG_ERROR("freeHostDeviceShareMem via %s failed on unregister: rc=%d host=%p", symbol_name, rc, hostPtr);
            return rc;
        }
    } else {
        LOG_ERROR("freeHostDeviceShareMem: missing symbols rtsHostUnregister / rtHostUnregister / aclrtHostUnregister");
        return -1;
    }

    if (RtFreeHostFn free_fn = resolve_symbol<RtFreeHostFn>(&symbol_name, {"rtFreeHost"})) {
        rc = free_fn(hostPtr);
        if (rc != 0) {
            LOG_ERROR("freeHostDeviceShareMem via %s failed on free: rc=%d host=%p", symbol_name, rc, hostPtr);
            return rc;
        }
    } else if (AclFreeHostFn free_fn = resolve_symbol<AclFreeHostFn>(&symbol_name, {"aclrtFreeHost"})) {
        rc = free_fn(hostPtr);
        if (rc != 0) {
            LOG_ERROR("freeHostDeviceShareMem via %s failed on free: rc=%d host=%p", symbol_name, rc, hostPtr);
            return rc;
        }
    } else {
        LOG_ERROR("freeHostDeviceShareMem: missing symbols rtFreeHost / aclrtFreeHost");
        return -1;
    }

    LOG_INFO("freeHostDeviceShareMem: device=%u host=%p", deviceId, hostPtr);
    return 0;
}

int run_runtime(
    DeviceContextHandle ctx, RuntimeHandle runtime, const void *callable, const void *args, int block_dim,
    int aicpu_thread_num, int device_id, const uint8_t *aicpu_binary, size_t aicpu_size, const uint8_t *aicore_binary,
    size_t aicore_size, int enable_profiling
) {
    if (ctx == NULL || runtime == NULL) return -1;
    if (aicpu_binary == NULL || aicpu_size == 0 || aicore_binary == NULL || aicore_size == 0) return -1;

    DeviceRunner *runner = static_cast<DeviceRunner *>(ctx);

    pthread_once(&g_runner_key_once, create_runner_key);
    pthread_setspecific(g_runner_key, ctx);
    auto tsd_guard = RAIIScopeGuard([]() {
        pthread_setspecific(g_runner_key, nullptr);
    });

    try {
        int rc = runner->ensure_device_set(device_id);
        if (rc != 0) return rc;
        auto device_guard = RAIIScopeGuard([runner]() {
            runner->reset_device_context();
        });

        Runtime *r = new (runtime) Runtime();
        r->host_api.device_malloc = device_malloc;
        r->host_api.device_free = device_free;
        r->host_api.copy_to_device = copy_to_device;
        r->host_api.copy_from_device = copy_from_device;
        r->host_api.upload_kernel_binary = upload_kernel_binary_wrapper;
        r->host_api.remove_kernel_binary = remove_kernel_binary_wrapper;

        LOG_DEBUG("About to call init_runtime_impl, r=%p", (void *)r);
        rc = init_runtime_impl(
            r, reinterpret_cast<const ChipCallable *>(callable), reinterpret_cast<const ChipStorageTaskArgs *>(args)
        );
        LOG_DEBUG("init_runtime_impl returned: %d", rc);
        if (rc != 0) {
            r->set_pto2_gm_sm_ptr(nullptr);
            validate_runtime_impl(r);
            r->~Runtime();
            return rc;
        }

        if (enable_profiling) {
            r->enable_profiling = true;
        }

        std::vector<uint8_t> aicpu_vec(aicpu_binary, aicpu_binary + aicpu_size);
        std::vector<uint8_t> aicore_vec(aicore_binary, aicore_binary + aicore_size);
        rc = runner->run(*r, block_dim, device_id, aicpu_vec, aicore_vec, aicpu_thread_num);
        if (rc != 0) {
            validate_runtime_impl(r);
            r->~Runtime();
            return rc;
        }

        rc = validate_runtime_impl(r);
        r->~Runtime();
        return rc;
    } catch (...) {
        return -1;
    }
}

int finalize_device(DeviceContextHandle ctx) {
    if (ctx == NULL) return -1;
    try {
        return static_cast<DeviceRunner *>(ctx)->finalize();
    } catch (...) {
        return -1;
    }
}

/* ===========================================================================
 * Internal helpers called from runtime_maker.cpp via Runtime.host_api
 * =========================================================================== */

void record_tensor_pair(RuntimeHandle runtime, void *host_ptr, void *dev_ptr, size_t size) {
    if (runtime == NULL) return;
    Runtime *r = static_cast<Runtime *>(runtime);
    r->record_tensor_pair(host_ptr, dev_ptr, size);
}

}  // extern "C"
