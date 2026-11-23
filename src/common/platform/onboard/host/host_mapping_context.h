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

#include <runtime/context.h>

#include "acl/acl.h"
#include "host_log.h"
#include "runtime_c_api.h"

namespace simpler {

template <typename Operation>
int with_host_mapping_context(rtContext_t primary, Operation operation) {
    if (primary == nullptr) return PTO_RUNTIME_ERR_INVALID_STATE;
    rtContext_t previous = nullptr;
    const int current = rtCtxGetCurrent(&previous);
    if (current != 0 && current != ACL_ERROR_RT_CONTEXT_NULL) return current;
    if (previous == primary) return operation();

    const int rc = rtCtxSetCurrent(primary);
    if (rc != 0) return rc;
    const int operation_rc = operation();
    if (current == ACL_ERROR_RT_CONTEXT_NULL) return operation_rc;
    const int restore_rc = rtCtxSetCurrent(previous);
    if (restore_rc == 0) return operation_rc;
    LOG_ERROR("Host mapping could not restore caller context: %d (operation status: %d)", restore_rc, operation_rc);
    return operation_rc == 0 ? SIMPLER_HOST_DDR_CONTEXT_RESTORE_FAILED : operation_rc;
}

}  // namespace simpler
