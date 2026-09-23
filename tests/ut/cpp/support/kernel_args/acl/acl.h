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
// Minimal RTS declarations for the CPU-only KernelArgsHelper tests.
#pragma once

extern "C" const char *aclGetRecentErrMsg();

// The capture-status query the launch route's permit asks, and just enough of
// its vocabulary to drive every answer. Values and spelling mirror
// acl/acl_rt.h; a case supplies the definition so it can choose the answer.
using aclError = int;
using aclrtStream = void *;
using aclmdlRI = void *;

#define ACL_SUCCESS 0

typedef enum {
    ACL_MODEL_RI_CAPTURE_STATUS_NONE = 0,
    ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE,
    ACL_MODEL_RI_CAPTURE_STATUS_INVALIDATED,
} aclmdlRICaptureStatus;

extern "C" aclError aclmdlRICaptureGetInfo(aclrtStream stream, aclmdlRICaptureStatus *status, aclmdlRI *modelRI);
