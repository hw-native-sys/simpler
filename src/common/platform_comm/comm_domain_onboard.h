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

#include <cstring>

#include "acl/acl.h"
#include "common/unified_log.h"
#include "platform_comm/comm_context.h"

namespace comm_domain_onboard {

inline bool valid_host_shm_name(const char *name) {
    if (name == nullptr) return false;
    const size_t length = strnlen(name, 32);
    if (length == 0 || length > 31) return false;
    for (size_t i = 0; i < length; ++i) {
        const char c = name[i];
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '-'))
            return false;
    }
    return true;
}

inline aclError upload_context(const CommContext &ctx, CommContext *&device_ctx, int rank, const char *operation) {
    void *device = nullptr;
    aclError status = aclrtMalloc(&device, sizeof(ctx), ACL_MEM_MALLOC_HUGE_FIRST);
    if (status != ACL_SUCCESS) {
        LOG_ERROR("[comm rank %d] %s: ctx aclrtMalloc -> %d", rank, operation, static_cast<int>(status));
        return status;
    }
    // The domain owns the allocation even when copying the context fails.
    device_ctx = static_cast<CommContext *>(device);
    status = aclrtMemcpy(device, sizeof(ctx), &ctx, sizeof(ctx), ACL_MEMCPY_HOST_TO_DEVICE);
    if (status != ACL_SUCCESS) {
        LOG_ERROR("[comm rank %d] %s: ctx Memcpy H2D -> %d", rank, operation, static_cast<int>(status));
    }
    return status;
}

}  // namespace comm_domain_onboard
