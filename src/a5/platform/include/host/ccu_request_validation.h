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

#include "common/ccu_execution_types.h"

#include <cstdint>

enum class CcuRequestValidationError {
    NONE,
    ABI_MISMATCH,
    INVALID_IDENTITY,
    UNSUPPORTED_OPERATION,
    INVALID_RANK,
    INVALID_COUNT,
    INVALID_WINDOW,
    WINDOW_OUT_OF_RANGE,
    OVERLAPPING_WINDOWS,
    INVALID_TRIGGER,
    INVALID_COMPLETION,
    RESERVED_FIELD_NONZERO
};

struct CcuWindowRangeV1 {
    uint64_t window_id;
    uint64_t allocation_id;
    uint64_t allocation_generation;
    uint64_t allocation_offset_bytes;
    uint64_t extent_bytes;
    int32_t device_id;
};

CcuRequestValidationError ccu_validate_request_v1(const CcuRequestDescriptorV1 &request);
CcuRequestValidationError ccu_validate_window_ranges_v1(
    const CcuRequestDescriptorV1 &request, const CcuWindowRangeV1 &input, const CcuWindowRangeV1 &output
);
CcuRequestValidationError ccu_validate_trigger_v1(const CcuTriggerInfoV1 &trigger);
CcuRequestValidationError ccu_validate_completion_header_v1(const CcuCompletionRecordV1 &record);
