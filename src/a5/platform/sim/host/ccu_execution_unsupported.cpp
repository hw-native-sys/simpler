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
#include "host/ccu_execution.h"

#include <cstring>

namespace {
void set_result(CcuHostResultV1 *result, int status, uint32_t stage, const char *message) {
    if (result == nullptr) return;
    std::memset(result, 0, sizeof(*result));
    result->abi_version = CCU_EXECUTION_ABI_V1;
    result->struct_size = sizeof(*result);
    result->status = status;
    result->stage = stage;
    if (message != nullptr) std::strncpy(result->message, message, sizeof(result->message) - 1);
}
int unsupported(CcuHostResultV1 *result, uint32_t stage) {
    set_result(result, CCU_STATUS_UNSUPPORTED, stage, "CCU execution is not provided by a5sim");
    return CCU_STATUS_UNSUPPORTED;
}
}  // namespace

extern "C" int ccu_query_capabilities(CcuCapabilitiesV1 *out) {
    if (out == nullptr || out->struct_size < sizeof(*out)) return CCU_STATUS_INVALID_ARGUMENT;
    std::memset(out, 0, sizeof(*out));
    out->abi_version = CCU_EXECUTION_ABI_V1;
    out->struct_size = sizeof(*out);
    out->max_rank_count = CCU_EXECUTION_RANK_COUNT_V1;
    out->max_element_count = CCU_EXECUTION_MAX_COUNT_V1;
    out->max_inflight = 1;
    return CCU_STATUS_SUCCESS;
}
extern "C" int ccu_get_root_info(int32_t, void *, uint32_t, uint32_t *size_out) {
    if (size_out != nullptr) *size_out = 0;
    return CCU_STATUS_UNSUPPORTED;
}
extern "C" int ccu_context_create(const CcuCreateOptionsV1 *, CcuContextId *out, CcuHostResultV1 *r) {
    if (out != nullptr) *out = 0;
    return unsupported(r, CCU_STAGE_CREATE);
}
extern "C" int ccu_window_register(CcuContextId, const CcuWindowImportV1 *, CcuWindowId *out, CcuHostResultV1 *r) {
    if (out != nullptr) *out = 0;
    return unsupported(r, CCU_STAGE_WINDOW_REGISTER);
}
extern "C" int ccu_bind_windows(CcuContextId, const CcuWindowBindingV1 *, CcuHostResultV1 *r) {
    return unsupported(r, CCU_STAGE_BIND);
}
extern "C" int ccu_prepare(CcuContextId, const CcuRequestSpecV1 *, CcuRequestId *out, CcuHostResultV1 *r) {
    if (out != nullptr) *out = 0;
    return unsupported(r, CCU_STAGE_PREPARE);
}
extern "C" int ccu_arm(CcuRequestId, CcuTriggerInfoV1 *, CcuHostResultV1 *r) { return unsupported(r, CCU_STAGE_ARM); }
extern "C" int ccu_trigger_host(CcuRequestId, CcuHostResultV1 *r) { return unsupported(r, CCU_STAGE_TRIGGER); }
extern "C" int ccu_poll(CcuRequestId, CcuHostResultV1 *r) { return unsupported(r, CCU_STAGE_POLL); }
extern "C" int ccu_wait(CcuRequestId, uint32_t, CcuHostResultV1 *r) { return unsupported(r, CCU_STAGE_WAIT); }
extern "C" int ccu_cancel_prepared(CcuRequestId, CcuHostResultV1 *r) { return unsupported(r, CCU_STAGE_CANCEL); }
extern "C" int ccu_retire(CcuRequestId, uint32_t, CcuHostResultV1 *r) { return unsupported(r, CCU_STAGE_RETIRE); }
extern "C" int ccu_context_close(CcuContextId, uint32_t, CcuHostResultV1 *r) { return unsupported(r, CCU_STAGE_CLOSE); }
