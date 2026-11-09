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

#include <stddef.h>
#include <stdint.h>

#include "common/ccu_execution_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef uint64_t CcuContextId;
typedef uint64_t CcuWindowId;
typedef uint64_t CcuRequestId;

typedef enum CcuStatusV1 {
    CCU_STATUS_SUCCESS = 0,
    CCU_STATUS_PENDING = 1,
    CCU_STATUS_UNSUPPORTED = -1,
    CCU_STATUS_INVALID_ARGUMENT = -2,
    CCU_STATUS_INVALID_STATE = -3,
    CCU_STATUS_BUSY = -4,
    CCU_STATUS_STALE_HANDLE = -5,
    CCU_STATUS_TIMEOUT = -6,
    CCU_STATUS_SDK_ERROR = -7,
    CCU_STATUS_PROTOCOL_ERROR = -8,
    CCU_STATUS_RETAINED = -9,
    CCU_STATUS_INTERNAL_ERROR = -10,
} CcuStatusV1;

typedef enum CcuStageV1 {
    CCU_STAGE_NONE = 0,
    CCU_STAGE_CAPABILITY = 1,
    CCU_STAGE_CREATE = 2,
    CCU_STAGE_WINDOW_REGISTER = 3,
    CCU_STAGE_BIND = 4,
    CCU_STAGE_PREPARE = 5,
    CCU_STAGE_ARM = 6,
    CCU_STAGE_TRIGGER = 7,
    CCU_STAGE_POLL = 8,
    CCU_STAGE_WAIT = 9,
    CCU_STAGE_CANCEL = 10,
    CCU_STAGE_RETIRE = 11,
    CCU_STAGE_CLOSE = 12,
} CcuStageV1;

typedef struct CcuHostResultV1 {
    uint32_t abi_version;
    uint32_t struct_size;
    int32_t status;
    uint32_t stage;
    int32_t sdk_error;
    uint32_t submission_state;
    uint64_t expected_sequence;
    uint64_t observed_sequence;
    uint32_t data_ready;
    uint32_t drained;
    uint32_t retained;
    uint32_t reserved;
    char message[160];
} CcuHostResultV1;

typedef struct CcuCapabilitiesV1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t arch;
    uint32_t flags;
    uint32_t max_rank_count;
    uint32_t max_element_count;
    uint32_t max_inflight;
    uint32_t supported_operation_mask;
    uint32_t supported_dtype_mask;
    uint32_t reserved[7];
} CcuCapabilitiesV1;

typedef struct CcuOwnerOpsV1 {
    uint32_t abi_version;
    uint32_t struct_size;
    int (*retain)(void *owner);
    int (*release)(void *owner);
} CcuOwnerOpsV1;

typedef struct CcuBootstrapOpsV1 {
    uint32_t abi_version;
    uint32_t struct_size;
    int (*exchange)(
        void *owner, uint64_t group_id, uint64_t collective_id, uint32_t phase, const void *local_blob,
        uint32_t local_size, void *rank_ordered_blobs, uint32_t blob_stride, uint32_t rank_count, uint32_t timeout_ms
    );
    int (*report_abort)(void *owner, uint64_t group_id, uint64_t collective_id, uint32_t phase, int32_t reason);
} CcuBootstrapOpsV1;

typedef struct CcuCreateOptionsV1 {
    uint32_t abi_version;
    uint32_t struct_size;
    int32_t device_id;
    uint32_t rank;
    uint32_t rank_count;
    uint32_t root_info_size;
    uint64_t group_id;
    const void *root_info;
    const char *trigger_binary_path;
    const CcuBootstrapOpsV1 *bootstrap_ops;
    void *bootstrap_owner;
    const CcuOwnerOpsV1 *bootstrap_owner_ops;
} CcuCreateOptionsV1;

typedef struct CcuWindowImportV1 {
    uint32_t abi_version;
    uint32_t struct_size;
    int32_t device_id;
    uint32_t flags;
    uint64_t allocation_id;
    uint64_t allocation_generation;
    uint64_t allocation_base_va;
    uint64_t allocation_size;
    uint64_t window_offset;
    uint64_t window_size;
    void *allocation_owner;
    const CcuOwnerOpsV1 *allocation_owner_ops;
} CcuWindowImportV1;

typedef struct CcuWindowBindingV1 {
    uint32_t abi_version;
    uint32_t struct_size;
    CcuWindowId input_window;
    CcuWindowId output_window;
    uint64_t binding_generation;
    uint64_t reserved[3];
} CcuWindowBindingV1;

typedef struct CcuRequestSpecV1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t operation;
    uint32_t dtype;
    uint64_t collective_id;
    uint64_t input_offset;
    uint64_t output_offset;
    uint64_t element_count;
    uint32_t root;
    uint32_t reserved0;
} CcuRequestSpecV1;

enum {
    CCU_CAPABILITY_SUPPORTED = 1U << 0,
    CCU_CAPABILITY_UBC_CTP = 1U << 1,
    CCU_CAPABILITY_COMPLETION_STORE = 1U << 2
};

int ccu_query_capabilities(CcuCapabilitiesV1 *out);
int ccu_get_root_info(int32_t device_id, void *bytes, uint32_t capacity, uint32_t *size_out);
int ccu_context_create(const CcuCreateOptionsV1 *options, CcuContextId *context_out, CcuHostResultV1 *result);
int ccu_window_register(
    CcuContextId context, const CcuWindowImportV1 *input, CcuWindowId *window_out, CcuHostResultV1 *result
);
int ccu_bind_windows(CcuContextId context, const CcuWindowBindingV1 *binding, CcuHostResultV1 *result);
int ccu_prepare(CcuContextId context, const CcuRequestSpecV1 *spec, CcuRequestId *request_out, CcuHostResultV1 *result);
int ccu_arm(CcuRequestId request, CcuTriggerInfoV1 *trigger_out, CcuHostResultV1 *result);
int ccu_trigger_host(CcuRequestId request, CcuHostResultV1 *result);
int ccu_poll(CcuRequestId request, CcuHostResultV1 *result);
int ccu_wait(CcuRequestId request, uint32_t timeout_ms, CcuHostResultV1 *result);
// After asymmetric cleanup results, both ranks can retry the same cancel/retire while both contexts remain open.
// Coordinate cleanup success before either rank closes or prepares another request; other request operations are stale.
int ccu_cancel_prepared(CcuRequestId request, CcuHostResultV1 *result);
int ccu_retire(CcuRequestId request, uint32_t timeout_ms, CcuHostResultV1 *result);
// timeout_ms budgets request draining and peer exchange; synchronous SDK cleanup can exceed this wall-clock budget.
int ccu_context_close(CcuContextId context, uint32_t timeout_ms, CcuHostResultV1 *result);

#ifdef __cplusplus
}
#endif
