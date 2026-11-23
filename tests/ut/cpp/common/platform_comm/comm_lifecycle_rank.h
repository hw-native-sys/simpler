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

#pragma once

#include <dlfcn.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "acl/acl.h"
#include "platform_comm/comm.h"
#include "platform_comm/comm_context.h"

namespace comm_lifecycle_test {

// Function pointers from libhost_runtime.so (comm_* C API + DeviceRunner wiring)
struct HostRuntimeApi {
    void *(*create_device_context)();
    void (*destroy_device_context)(void *);
    int (*ensure_acl_ready_ctx)(void *, int);

    CommHandle (*comm_init)(int, int, void *, const char *);
    int (*comm_alloc_windows)(CommHandle, size_t, uint64_t *);
    int (*comm_get_local_window_base)(CommHandle, uint64_t *);
    int (*comm_get_window_size)(CommHandle, size_t *);
    int (*comm_barrier)(CommHandle);
    int (*comm_destroy)(CommHandle);
};

template <typename F>
F resolve(void *handle, const char *name) {
    dlerror();
    void *sym = dlsym(handle, name);
    if (dlerror() != nullptr) return nullptr;
    return reinterpret_cast<F>(sym);
}

inline bool load_host_runtime_api(void *handle, HostRuntimeApi &api) {
    api.create_device_context = resolve<decltype(api.create_device_context)>(handle, "create_device_context");
    api.destroy_device_context = resolve<decltype(api.destroy_device_context)>(handle, "destroy_device_context");
    api.ensure_acl_ready_ctx = resolve<decltype(api.ensure_acl_ready_ctx)>(handle, "ensure_acl_ready_ctx");
    api.comm_init = resolve<decltype(api.comm_init)>(handle, "comm_init");
    api.comm_alloc_windows = resolve<decltype(api.comm_alloc_windows)>(handle, "comm_alloc_windows");
    api.comm_get_local_window_base =
        resolve<decltype(api.comm_get_local_window_base)>(handle, "comm_get_local_window_base");
    api.comm_get_window_size = resolve<decltype(api.comm_get_window_size)>(handle, "comm_get_window_size");
    api.comm_barrier = resolve<decltype(api.comm_barrier)>(handle, "comm_barrier");
    api.comm_destroy = resolve<decltype(api.comm_destroy)>(handle, "comm_destroy");
    return api.create_device_context && api.destroy_device_context && api.ensure_acl_ready_ctx && api.comm_init &&
           api.comm_alloc_windows && api.comm_get_local_window_base && api.comm_get_window_size && api.comm_barrier &&
           api.comm_destroy;
}

// Exit codes for the rank child. Each stage gets a distinct code so the
// parent's waitpid surface pinpoints exactly which step broke.
constexpr int EXIT_DLERR = 10;
constexpr int EXIT_DEV_CTX = 15;
constexpr int EXIT_ACL_READY = 18;
constexpr int EXIT_STREAM = 19;
constexpr int EXIT_INIT = 20;
constexpr int EXIT_ALLOC = 30;
constexpr int EXIT_WINDOW_BASE = 40;
constexpr int EXIT_WINDOW_SIZE = 50;
// EXIT_CTX_{MEMCPY,FIELDS} guard the device-side CommContext that Fabric V2
// fills in alloc_windows_via_fabric: cross-API consistency between
// comm_get_window_size / winSize, and every peer's windowsIn slot
// actually getting a non-zero GVA.
constexpr int EXIT_CTX_MEMCPY = 55;
constexpr int EXIT_CTX_FIELDS = 56;
constexpr int EXIT_BARRIER = 60;
constexpr int EXIT_DESTROY = 70;

inline int run_rank(int rank, int nranks, int device_id, const char *rootinfo_path) {
    // libhost_runtime.so is the subject under test -- dlopen mirrors
    // ChipWorker.  The host logger is self-contained in that DSO; libascendcl
    // is linked in, so acl* is available directly.
    void *host_handle = dlopen(PTO_HOST_RUNTIME_LIB_PATH, RTLD_NOW | RTLD_LOCAL);
    if (host_handle == nullptr) {
        fprintf(stderr, "[rank %d] dlopen host lib failed: %s\n", rank, dlerror());
        return EXIT_DLERR;
    }

    HostRuntimeApi api{};
    if (!load_host_runtime_api(host_handle, api)) {
        fprintf(stderr, "[rank %d] required symbols missing from libhost_runtime.so\n", rank);
        dlclose(host_handle);
        return EXIT_DLERR;
    }

    // Caller step 1: stand up DeviceRunner to own ACL lifecycle.
    void *dev_ctx = api.create_device_context();
    if (dev_ctx == nullptr) {
        fprintf(stderr, "[rank %d] create_device_context returned null\n", rank);
        dlclose(host_handle);
        return EXIT_DEV_CTX;
    }

    // Caller step 2: aclInit + aclrtSetDevice via DeviceRunner.
    if (api.ensure_acl_ready_ctx(dev_ctx, device_id) != 0) {
        fprintf(stderr, "[rank %d] ensure_acl_ready_ctx(%d) failed\n", rank, device_id);
        api.destroy_device_context(dev_ctx);
        dlclose(host_handle);
        return EXIT_ACL_READY;
    }

    // Caller step 3: caller creates its own stream; comm never creates or destroys it.
    aclrtStream stream = nullptr;
    aclError aRet = aclrtCreateStream(&stream);
    if (aRet != ACL_SUCCESS || stream == nullptr) {
        fprintf(stderr, "[rank %d] aclrtCreateStream failed: %d\n", rank, static_cast<int>(aRet));
        api.destroy_device_context(dev_ctx);
        dlclose(host_handle);
        return EXIT_STREAM;
    }

    // Caller step 4: drive comm_* against the injected stream.
    int stage = 0;
    int exit_code = 0;
    CommHandle h = api.comm_init(rank, nranks, stream, rootinfo_path);
    if (h == nullptr) {
        stage = EXIT_INIT;
    } else {
        uint64_t device_ctx_ptr = 0;
        if (api.comm_alloc_windows(h, 4096, &device_ctx_ptr) != 0 || device_ctx_ptr == 0) {
            stage = EXIT_ALLOC;
        } else {
            uint64_t local_base = 0;
            if (api.comm_get_local_window_base(h, &local_base) != 0 || local_base == 0) {
                stage = EXIT_WINDOW_BASE;
            } else {
                size_t win_size = 0;
                if (api.comm_get_window_size(h, &win_size) != 0 || win_size < 4096) {
                    stage = EXIT_WINDOW_SIZE;
                } else {
                    // Black-box contract check on the device-side CommContext
                    // produced by Fabric V2: cross-API consistency (winSize ==
                    // comm_get_window_size), every peer's windowsIn slot
                    // populated, and the local slot matching
                    // comm_get_local_window_base.
                    CommContext host_ctx{};
                    aclError mc_rc = aclrtMemcpy(
                        &host_ctx, sizeof(host_ctx), reinterpret_cast<void *>(device_ctx_ptr), sizeof(host_ctx),
                        ACL_MEMCPY_DEVICE_TO_HOST
                    );
                    if (mc_rc != ACL_SUCCESS) {
                        fprintf(
                            stderr, "[rank %d] aclrtMemcpy(device_ctx) failed: %d\n", rank, static_cast<int>(mc_rc)
                        );
                        stage = EXIT_CTX_MEMCPY;
                    } else if (host_ctx.rankId != static_cast<uint32_t>(rank) ||
                               host_ctx.rankNum != static_cast<uint32_t>(nranks) ||
                               host_ctx.winSize != static_cast<uint64_t>(win_size) ||
                               host_ctx.windowsIn[rank] != local_base) {
                        fprintf(
                            stderr,
                            "[rank %d] CommContext field mismatch\n"
                            "  got:      rankId=%u rankNum=%u winSize=%lu windowsIn[%d]=0x%lx\n"
                            "  expected: rankId=%d rankNum=%d winSize=%zu windowsIn[%d]=0x%lx\n",
                            rank, host_ctx.rankId, host_ctx.rankNum, static_cast<unsigned long>(host_ctx.winSize), rank,
                            static_cast<unsigned long>(host_ctx.windowsIn[rank]), rank, nranks, win_size, rank,
                            static_cast<unsigned long>(local_base)
                        );
                        stage = EXIT_CTX_FIELDS;
                    } else {
                        // Every peer's window GVA must be non-zero.  A zero
                        // entry means the Fabric handle import didn't
                        // populate that slot but the function still returned
                        // success.
                        for (int i = 0; i < nranks; ++i) {
                            if (host_ctx.windowsIn[i] == 0) {
                                fprintf(
                                    stderr, "[rank %d] CommContext.windowsIn[%d] == 0 (peer GVA missing)\n", rank, i
                                );
                                stage = EXIT_CTX_FIELDS;
                                break;
                            }
                        }
                    }

                    if (stage == 0 && api.comm_barrier(h) != 0) {
                        stage = EXIT_BARRIER;
                    }
                }
            }
        }
        const int destroy_rc = api.comm_destroy(h);
        if (destroy_rc != 0 && stage == 0) {
            stage = EXIT_DESTROY;
        }
        if (destroy_rc == COMM_RECLAMATION_UNCONFIRMED) return stage;
    }
    exit_code = stage;

    // Caller step 5: cleanup in reverse order.  destroy_device_context
    // eventually drives DeviceRunner::finalize which calls aclrtResetDevice
    // and aclFinalize, so we do not call them ourselves.
    aclrtDestroyStream(stream);
    api.destroy_device_context(dev_ctx);
    dlclose(host_handle);
    return exit_code;
}

}  // namespace comm_lifecycle_test
