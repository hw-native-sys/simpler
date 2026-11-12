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

#include <runtime/context.h>

#include <thread>

#include "acl/acl.h"
#include "device_runner.h"
#include "host/raii_scope_guard.h"
#include "host_log.h"
#include "runtime_c_api.h"

namespace {

template <typename Operation>
int with_host_mapping_context(DeviceRunner &runner, int device, Operation operation) {
    if (runner.device_id() != device || device < 0 || runner.execution_mode_latch().is_kernel()) {
        return PTO_RUNTIME_ERR_INVALID_STATE;
    }
    rtContext_t primary = nullptr;
    const int lookup = rtGetPriCtxByDeviceId(device, &primary);
    if (lookup != 0) return lookup;
    if (primary == nullptr) return PTO_RUNTIME_ERR_INVALID_STATE;

    rtContext_t previous = nullptr;
    const int current = rtCtxGetCurrent(&previous);
    if (current == ACL_ERROR_RT_CONTEXT_NULL) {
        // RTS cannot clear a thread's current context. The temporary thread owns only the binding.
        int result = PTO_RUNTIME_ERR_INTERNAL;
        std::thread([&]() {
            try {
                const int rc = rtCtxSetCurrent(primary);
                result = rc == 0 ? operation() : rc;
            } catch (...) {}
        }).join();
        return result;
    }
    if (current != 0) return current;
    if (previous == primary) return operation();

    const int rc = rtCtxSetCurrent(primary);
    if (rc != 0) return rc;
    auto restore = RAIIScopeGuard([previous]() {
        const int restore_rc = rtCtxSetCurrent(previous);
        if (restore_rc != 0) LOG_ERROR("Host mapping could not restore the calling thread's context: %d", restore_rc);
    });
    return operation();
}

}  // namespace

extern "C" int host_ddr_map_ctx(
    DeviceContextHandle ctx, int device, uint64_t host, uint64_t bytes, uint64_t *mapped, unsigned *registered
) {
    if (!ctx || !host || !bytes || !mapped || !registered) return PTO_RUNTIME_ERR_INTERNAL;
    *mapped = 0;
    *registered = 0;
    try {
        return with_host_mapping_context(*static_cast<DeviceRunner *>(ctx), device, [&]() -> int {
            auto *base = reinterpret_cast<void *>(host);
            void *address = nullptr;
            const int rc = aclrtHostRegister(base, bytes, ACL_HOST_REGISTER_MAPPED, &address);
            if (rc != 0) return rc;
            *registered = 1;
            if (!address) return PTO_RUNTIME_ERR_INTERNAL;
            *mapped = reinterpret_cast<uint64_t>(address);
            return 0;
        });
    } catch (...) {
        return PTO_RUNTIME_ERR_INTERNAL;
    }
}

extern "C" int host_ddr_unmap_ctx(DeviceContextHandle ctx, int device, uint64_t host) {
    if (!ctx || !host) return PTO_RUNTIME_ERR_INTERNAL;
    try {
        return with_host_mapping_context(*static_cast<DeviceRunner *>(ctx), device, [host]() {
            return aclrtHostUnregister(reinterpret_cast<void *>(host));
        });
    } catch (...) {
        return PTO_RUNTIME_ERR_INTERNAL;
    }
}
