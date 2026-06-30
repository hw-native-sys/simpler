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
 * PTO Runtime C API — a2a3 arch-specific entries
 *
 * The shared c_api glue (simpler_init / simpler_register_callable / simpler_run /
 * device_*_ctx / etc.) lives in
 * `src/common/platform/onboard/host/c_api_shared.cpp` and is linked into
 * this same `libhost_runtime.so`. This file keeps only the entries that
 * need the concrete a2a3 `DeviceRunner` subclass or the a2a3-only HCCL/comm
 * surface.
 */

#include "device_runner.h"
#include "pto_runtime_c_api.h"

extern "C" {

DeviceContextHandle create_device_context(void) {
    try {
        return static_cast<DeviceContextHandle>(new DeviceRunner());
    } catch (...) {
        return NULL;
    }
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
