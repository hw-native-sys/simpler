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
 * PTO Runtime C API — a5 sim arch-specific entries.
 *
 * The shared c_api glue (simpler_init / prepare_callable / run_prepared /
 * device_*_ctx / etc.) lives in `src/common/platform/sim/host/c_api_shared.cpp`
 * and is linked into this same libhost_runtime.so. This file keeps only the
 * entries that need the concrete a5 sim `DeviceRunner` subclass
 * (`create_device_context`) plus the ACL no-op stubs that ChipWorker dlsyms
 * unconditionally — sim has no ACL / aclrtStream concept, and the comm_* paired
 * entry points come from `src/common/platform_comm/comm_sim.cpp`.
 */

#include "device_runner.h"
#include "pto_runtime_c_api.h"

#include "host_device_comm/host_device_mapped_region.h"
#include "host_device_comm/host_device_mapped_region_sim.h"

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
    return host_device_mapped_region_open_common(ctx, cfg, out_region, host_device_mapped_region_allocate_sim);
}

int close_host_device_mapped_region_ctx(DeviceContextHandle ctx, HostDeviceMappedRegionHandle region) {
    return host_device_mapped_region_close_common(ctx, region);
}

int host_device_mapped_region_info_ctx(
    DeviceContextHandle ctx, HostDeviceMappedRegionHandle region, HostDeviceMappedRegionInfo *info
) {
    int rc = host_device_mapped_region_info_common(ctx, region, info);
    if (rc == 0) {
        info->host_data_ptr = 0;
        info->host_signal_ptr = 0;
    }
    return rc;
}

int host_device_mapped_region_datacopy_h2region_ctx(
    DeviceContextHandle ctx, HostDeviceMappedRegionHandle region, uint64_t offset, const void *src, size_t nbytes
) {
    return host_device_mapped_region_datacopy_h2region_common(ctx, region, offset, src, nbytes);
}

int host_device_mapped_region_datacopy_region2h_ctx(
    DeviceContextHandle ctx, HostDeviceMappedRegionHandle region, uint64_t offset, void *dst, size_t nbytes
) {
    return host_device_mapped_region_datacopy_region2h_common(ctx, region, offset, dst, nbytes);
}

int host_device_mapped_region_notify_ctx(
    DeviceContextHandle ctx, HostDeviceMappedRegionHandle region, uint32_t signal_id, uint32_t value
) {
    return host_device_mapped_region_notify_common(ctx, region, signal_id, value);
}

int host_device_mapped_region_wait_ctx(
    DeviceContextHandle ctx, HostDeviceMappedRegionHandle region, uint32_t signal_id, uint32_t target,
    uint32_t timeout_us
) {
    return host_device_mapped_region_wait_common(ctx, region, signal_id, target, timeout_us);
}

/* ===========================================================================
 * ACL lifecycle stubs. Sim has no ACL / aclrtStream concept, so these no-op
 * to satisfy the uniform host_runtime.so ABI that ChipWorker dlsym's.
 * =========================================================================== */

int ensure_acl_ready_ctx(DeviceContextHandle ctx, int device_id) {
    (void)ctx;
    (void)device_id;
    return 0;
}

void *create_comm_stream_ctx(DeviceContextHandle ctx) {
    (void)ctx;
    return NULL;
}

int destroy_comm_stream_ctx(DeviceContextHandle ctx, void *stream) {
    (void)ctx;
    (void)stream;
    return 0;
}

}  // extern "C"
