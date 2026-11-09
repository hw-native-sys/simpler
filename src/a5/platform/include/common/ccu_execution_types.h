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

#ifdef __cplusplus
#include <type_traits>
#define CCU_ALIGNAS(n) alignas(n)
#else
#define CCU_ALIGNAS(n) __attribute__((aligned(n)))
#endif

#define CCU_EXECUTION_ABI_V1 1U
#define CCU_EXECUTION_MAX_COUNT_V1 1024U
#define CCU_EXECUTION_RANK_COUNT_V1 2U

typedef enum CcuOperationV1 { CCU_OPERATION_BROADCAST = 1 } CcuOperationV1;
typedef enum CcuDataTypeV1 { CCU_DATA_TYPE_FLOAT32 = 1 } CcuDataTypeV1;
typedef enum CcuTriggerOwnerV1 { CCU_TRIGGER_OWNER_HOST = 1 } CcuTriggerOwnerV1;

typedef struct CCU_ALIGNAS(128) CcuRequestDescriptorV1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t operation;
    uint32_t dtype;
    uint64_t context_id;
    uint64_t context_generation;
    uint64_t request_seq;
    uint64_t program_id;
    uint64_t input_window_id;
    uint64_t output_window_id;
    uint64_t input_offset;
    uint64_t output_offset;
    uint64_t element_count;
    uint32_t root;
    uint32_t rank_count;
    uint64_t collective_id;
    uint64_t reserved[3];
} CcuRequestDescriptorV1;

typedef struct CCU_ALIGNAS(128) CcuTriggerInfoV1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint64_t context_id;
    uint64_t context_generation;
    uint64_t request_seq;
    uint64_t gate_va;
    uint64_t completion_record_va;
    uint32_t gate_mask;
    uint32_t trigger_owner;
    uint64_t reserved[9];
} CcuTriggerInfoV1;

typedef struct CCU_ALIGNAS(128) CcuCompletionRecordV1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint64_t context_id;
    uint64_t context_generation;
    uint64_t reserved_header[13];
    uint64_t completed_seq;
    uint64_t reserved_done[15];
} CcuCompletionRecordV1;

#ifdef __cplusplus
static_assert(sizeof(CcuRequestDescriptorV1) == 128);
static_assert(alignof(CcuRequestDescriptorV1) == 128);
static_assert(offsetof(CcuRequestDescriptorV1, context_id) == 16);
static_assert(offsetof(CcuRequestDescriptorV1, request_seq) == 32);
static_assert(offsetof(CcuRequestDescriptorV1, input_offset) == 64);
static_assert(offsetof(CcuRequestDescriptorV1, element_count) == 80);
static_assert(offsetof(CcuRequestDescriptorV1, collective_id) == 96);
static_assert(std::is_standard_layout_v<CcuRequestDescriptorV1>);
static_assert(std::is_trivially_copyable_v<CcuRequestDescriptorV1>);
static_assert(sizeof(CcuTriggerInfoV1) == 128);
static_assert(alignof(CcuTriggerInfoV1) == 128);
static_assert(offsetof(CcuTriggerInfoV1, context_id) == 8);
static_assert(offsetof(CcuTriggerInfoV1, request_seq) == 24);
static_assert(offsetof(CcuTriggerInfoV1, gate_va) == 32);
static_assert(offsetof(CcuTriggerInfoV1, completion_record_va) == 40);
static_assert(offsetof(CcuTriggerInfoV1, gate_mask) == 48);
static_assert(std::is_standard_layout_v<CcuTriggerInfoV1>);
static_assert(std::is_trivially_copyable_v<CcuTriggerInfoV1>);
static_assert(sizeof(CcuCompletionRecordV1) == 256);
static_assert(alignof(CcuCompletionRecordV1) == 128);
static_assert(offsetof(CcuCompletionRecordV1, completed_seq) == 128);
static_assert(std::is_standard_layout_v<CcuCompletionRecordV1>);
static_assert(std::is_trivially_copyable_v<CcuCompletionRecordV1>);
#endif

#undef CCU_ALIGNAS
