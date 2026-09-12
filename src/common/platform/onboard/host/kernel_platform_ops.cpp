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
 * CANN implementation of the kernel-mode context vocabulary.
 *
 * Linked into both a2a3 and a5 `libhost_runtime.so`.
 */

#include "kernel_platform_ops.h"

#include <acl/acl.h>
#include <runtime/rt.h>

#include "common/unified_log.h"
#include "host/acl_error_log.h"

namespace {

int get_current_device(void *, int *device_id) noexcept {
    int32_t current = -1;
    const aclError rc = aclrtGetDevice(&current);
    if (rc != ACL_SUCCESS) {
        LOG_ERROR("kernel context: aclrtGetDevice failed: %d", static_cast<int>(rc));
        ACL_LOG_ERROR_DETAIL(rc);
        return static_cast<int>(rc);
    }
    *device_id = static_cast<int>(current);
    return 0;
}

int create_hidden_stream(void *, void **stream) noexcept {
    rtStream_t created = nullptr;
    const rtError_t rc = rtStreamCreate(&created, 0);
    if (rc != RT_ERROR_NONE) {
        LOG_ERROR("kernel context: rtStreamCreate failed: %d", static_cast<int>(rc));
        ACL_LOG_ERROR_DETAIL(rc);
        return static_cast<int>(rc);
    }
    *stream = created;
    return 0;
}

int destroy_hidden_stream(void *, void *stream) noexcept {
    const rtError_t rc = rtStreamDestroy(static_cast<rtStream_t>(stream));
    if (rc != RT_ERROR_NONE) {
        LOG_ERROR("kernel context: rtStreamDestroy failed: %d", static_cast<int>(rc));
        ACL_LOG_ERROR_DETAIL(rc);
        return static_cast<int>(rc);
    }
    return 0;
}

int create_event(void *, void **event) noexcept {
    aclrtEvent created = nullptr;
    const aclError rc = aclrtCreateEventExWithFlag(&created, ACL_EVENT_SYNC);
    if (rc != ACL_SUCCESS) {
        LOG_ERROR("kernel context: aclrtCreateEventExWithFlag failed: %d", static_cast<int>(rc));
        ACL_LOG_ERROR_DETAIL(rc);
        return static_cast<int>(rc);
    }
    *event = created;
    return 0;
}

int destroy_event(void *, void *event) noexcept {
    const aclError rc = aclrtDestroyEvent(static_cast<aclrtEvent>(event));
    if (rc != ACL_SUCCESS) {
        LOG_ERROR("kernel context: aclrtDestroyEvent failed: %d", static_cast<int>(rc));
        ACL_LOG_ERROR_DETAIL(rc);
        return static_cast<int>(rc);
    }
    return 0;
}

}  // namespace

KernelContextOps make_onboard_kernel_context_ops() {
    KernelContextOps ops{};
    ops.get_current_device = &get_current_device;
    ops.create_hidden_stream = &create_hidden_stream;
    ops.destroy_hidden_stream = &destroy_hidden_stream;
    ops.create_event = &create_event;
    ops.destroy_event = &destroy_event;
    return ops;
}
