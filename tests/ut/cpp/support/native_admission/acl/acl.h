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
#include <cstddef>
#include <cstdint>
using aclError = int;
using aclrtStream = void *;
using aclrtEvent = void *;
using aclmdlRI = void *;
constexpr int ACL_SUCCESS = 0;
constexpr int ACL_HBM_MEM = 0;
constexpr int ACL_ERROR_NONE = 0;
constexpr int ACL_ERROR_RT_STREAM_SYNC_TIMEOUT = 507046;
constexpr int ACL_ERROR_RT_EVENT_SYNC_TIMEOUT = 507047;
constexpr uint32_t ACL_EVENT_SYNC = 1;
constexpr uint32_t ACL_EVENT_TIME_LINE = 8;
constexpr int ACL_RT_DEV_RES_CUBE_CORE = 0;
constexpr int ACL_RT_DEV_RES_VECTOR_CORE = 1;
enum aclrtEventRecordedStatus { ACL_EVENT_RECORDED_STATUS_NOT_READY, ACL_EVENT_RECORDED_STATUS_COMPLETE };
enum aclmdlRICaptureStatus {
    ACL_MODEL_RI_CAPTURE_STATUS_NONE,
    ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE,
    ACL_MODEL_RI_CAPTURE_STATUS_INVALIDATED
};
extern "C" {
const char *aclGetRecentErrMsg();
int aclFinalize();
int aclrtGetMemInfo(int, size_t *, size_t *);
int aclrtCreateEventExWithFlag(aclrtEvent *, uint32_t);
int aclrtDestroyEvent(aclrtEvent);
int aclrtRecordEvent(aclrtEvent, aclrtStream);
int aclrtResetEvent(aclrtEvent, aclrtStream);
int aclrtQueryEventStatus(aclrtEvent, aclrtEventRecordedStatus *);
int aclrtSynchronizeEventWithTimeout(aclrtEvent, int32_t);
int aclrtEventGetTimestamp(aclrtEvent, uint64_t *);
int aclrtStreamWaitEvent(aclrtStream, aclrtEvent);
int aclrtSynchronizeStream(aclrtStream);
int aclrtSynchronizeStreamWithTimeout(aclrtStream, int32_t);
int aclrtGetStreamResLimit(aclrtStream, int, uint32_t *);
int aclrtMemset(void *, size_t, int32_t, size_t);
int aclrtSetOpExecuteTimeOutV2(uint64_t, uint64_t *);
int aclrtStreamGetId(aclrtStream, int32_t *);
int aclmdlRICaptureGetInfo(aclrtStream, aclmdlRICaptureStatus *, aclmdlRI *);
}
