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
 * PTO Runtime C API — a5 arch-specific entries
 *
 * The shared c_api glue (simpler_init / prepare_callable / run_prepared /
 * device_*_ctx / etc.) lives in
 * `src/common/platform/onboard/host/c_api_shared.cpp` and is linked into
 * this same `libhost_runtime.so`. This file keeps only the entries that
 * need the concrete a5 `DeviceRunner` subclass or the a5-only HCCL/comm
 * surface.
 */

#include "device_runner.h"
#include "pto_runtime_c_api.h"

#include "host_device_comm/host_device_mapped_region.h"

#include <errno.h>

extern "C" {

DeviceContextHandle create_device_context(void) {
    try {
        return static_cast<DeviceContextHandle>(new DeviceRunner());
    } catch (...) {
        return NULL;
    }
}

int open_host_device_mapped_region_ctx(
    DeviceContextHandle ctx, const HostDeviceMappedRegionConfig *cfg, HostDeviceMappedRegionHandle *out_region
) {
    (void)ctx;
    (void)cfg;
    if (out_region != NULL) {
        *out_region = NULL;
    }
    return -ENOTSUP;
}

int close_host_device_mapped_region_ctx(DeviceContextHandle ctx, HostDeviceMappedRegionHandle region) {
    int rc = host_device_mapped_region_close_common(ctx, region);
    return rc == -EINVAL ? -ENOTSUP : rc;
}

int host_device_mapped_region_info_ctx(
    DeviceContextHandle ctx, HostDeviceMappedRegionHandle region, HostDeviceMappedRegionInfo *info
) {
    int rc = host_device_mapped_region_info_common(ctx, region, info);
    return rc == -EINVAL ? -ENOTSUP : rc;
}

int host_device_mapped_region_datacopy_h2region_ctx(
    DeviceContextHandle ctx, HostDeviceMappedRegionHandle region, uint64_t offset, const void *src, size_t nbytes
) {
    int rc = host_device_mapped_region_datacopy_h2region_common(ctx, region, offset, src, nbytes);
    return rc == -EINVAL ? -ENOTSUP : rc;
}

int host_device_mapped_region_datacopy_region2h_ctx(
    DeviceContextHandle ctx, HostDeviceMappedRegionHandle region, uint64_t offset, void *dst, size_t nbytes
) {
    int rc = host_device_mapped_region_datacopy_region2h_common(ctx, region, offset, dst, nbytes);
    return rc == -EINVAL ? -ENOTSUP : rc;
}

int host_device_mapped_region_notify_ctx(
    DeviceContextHandle ctx, HostDeviceMappedRegionHandle region, uint32_t signal_id, uint32_t value
) {
    int rc = host_device_mapped_region_notify_common(ctx, region, signal_id, value);
    return rc == -EINVAL ? -ENOTSUP : rc;
}

int host_device_mapped_region_wait_ctx(
    DeviceContextHandle ctx, HostDeviceMappedRegionHandle region, uint32_t signal_id, uint32_t target,
    uint32_t timeout_us
) {
    int rc = host_device_mapped_region_wait_common(ctx, region, signal_id, target, timeout_us);
    return rc == -EINVAL ? -ENOTSUP : rc;
}

int ensure_acl_ready_ctx(DeviceContextHandle ctx, int device_id) {
    if (ctx == NULL) return -1;
    try {
        return static_cast<DeviceRunner *>(ctx)->ensure_acl_ready(device_id);
    } catch (...) {
        return -1;
    }
}

/*
 * Stream creation/destruction exposed so the ChipWorker Python wrapper can
 * drive comm_init end-to-end without leaking aclrtStream lifetime (or ACL
 * libs) into Python.  Both entries go through the DeviceRunner so the ACL
 * ready-flag and device bookkeeping stay consistent with the normal run path.
 */
void *create_comm_stream_ctx(DeviceContextHandle ctx) {
    if (ctx == NULL) return NULL;
    try {
        return static_cast<DeviceRunner *>(ctx)->create_comm_stream();
    } catch (...) {
        return NULL;
    }
}

int destroy_comm_stream_ctx(DeviceContextHandle ctx, void *stream) {
    if (ctx == NULL) return -1;
    try {
        return static_cast<DeviceRunner *>(ctx)->destroy_comm_stream(stream);
    } catch (...) {
        return -1;
    }
}

}  // extern "C"
