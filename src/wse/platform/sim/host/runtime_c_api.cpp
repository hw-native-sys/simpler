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
 * simpler host-runtime C API — WSE sim arch-specific entries.
 *
 * The WSE C API glue (simpler_init / simpler_register_callable / simpler_run /
 * device_*_ctx / etc.) lives in `wse_runtime_api.cpp` and is linked into this
 * same libhost_runtime.so. This file keeps only the
 * entries that need the concrete WSE sim `DeviceRunner` subclass
 * (`create_device_context`) plus the ACL no-op stubs that ChipWorker dlsyms
 * unconditionally — sim has no ACL / aclrtStream concept, and the comm_* paired
 * entry points come from `src/common/platform_comm/comm_sim.cpp`.
 */

#include "device_runner.h"
#include "runtime_c_api.h"
#include <algorithm>
#include <cstdio>
#include <stdexcept>

extern "C" {

int requires_aicpu_executor(void) { return 0; }

DeviceContextHandle create_device_context(void) {
    try {
        return static_cast<DeviceContextHandle>(new DeviceRunner());
    } catch (...) {
        return NULL;
    }
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
