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

#include "acl/acl.h"
#include "device_runner.h"
#include "host_mapping_context.h"
#include "runtime_c_api.h"

extern "C" int host_ddr_map_ctx(
    DeviceContextHandle ctx, int device, uint64_t host, uint64_t bytes, uint64_t *mapped, unsigned *registered
) {
    if (!ctx || !host || !bytes || !mapped || !registered) return PTO_RUNTIME_ERR_INTERNAL;
    *mapped = 0;
    *registered = 0;
    try {
        auto &runner = *static_cast<DeviceRunner *>(ctx);
        return simpler::with_host_mapping_context(runner.host_mapping_context(device), [&]() -> int {
            int rc = runner.ensure_host_mapping_acl();
            if (rc != 0) return rc;
            auto *base = reinterpret_cast<void *>(host);
            rc = aclrtHostRegisterV2(base, bytes, ACL_HOST_REG_MAPPED);
            if (rc != 0) return rc;
            // The caller retains the backing even when obtaining the device address fails.
            *registered = 1;
            void *address = nullptr;
            rc = aclrtHostGetDevicePointer(base, &address, 0);
            if (rc != 0) return rc;
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
        return simpler::with_host_mapping_context(
            static_cast<DeviceRunner *>(ctx)->host_mapping_context(device), [host] {
                return aclrtHostUnregister(reinterpret_cast<void *>(host));
            }
        );
    } catch (...) {
        return PTO_RUNTIME_ERR_INTERNAL;
    }
}

extern "C" int quiesce_host_uses_ctx(DeviceContextHandle ctx) {
    if (!ctx) return PTO_RUNTIME_ERR_INTERNAL;
    try {
        return static_cast<DeviceRunner *>(ctx)->quiesce_host_uses();
    } catch (...) {
        return PTO_RUNTIME_ERR_INTERNAL;
    }
}
