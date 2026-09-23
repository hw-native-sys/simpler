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

#include "acl/acl_rt.h"

using aclrtStream = void *;
using aclrtEvent = void *;
constexpr uint32_t ACL_EVENT_SYNC = 1;
constexpr uint64_t ACL_STOP_ON_FAILURE = 1;
constexpr aclError ACL_ERROR_RT_PARAM_INVALID = 107000;
constexpr aclError ACL_ERROR_RT_FEATURE_NOT_SUPPORT = 207000;

extern "C" aclError aclrtGetDevice(int32_t *device);
extern "C" aclError aclrtSetStreamFailureMode(aclrtStream stream, uint64_t mode);
extern "C" aclError aclrtCreateEventExWithFlag(aclrtEvent *event, uint32_t flag);
extern "C" aclError aclrtDestroyEvent(aclrtEvent event);
extern "C" const char *aclGetRecentErrMsg();
