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

#include "common/region_instance_semantics.h"

static constexpr uint32_t REGION_TASK_MAGIC = 0x4C334C32u;
static constexpr uint16_t REGION_TASK_ABI_MAJOR = 4;
static constexpr uint16_t REGION_TASK_ABI_MINOR = 0;
static constexpr size_t REGION_TASK_SCALAR_COUNT = 8;
static constexpr uint64_t REGION_TASK_MAGIC_VERSION = 0x4C334C3200040000ull;

enum class RegionTaskArgError : uint32_t {
    OK = 0,
    NULL_POINTER,
    BAD_SCALAR_COUNT,
    BAD_MAGIC_VERSION,
    BAD_REGION_ID,
    BAD_TRANSACTION_ID,
    BAD_PAYLOAD_RANGE,
    BAD_COUNTER_RANGE,
};

inline bool region_decode_task_args(
    const uint64_t *scalars, size_t scalar_count, RegionPartLocalSpan &payload, RegionPartLocalSpan &counter,
    uint64_t &session_bits, uint64_t &transaction_id, uint64_t &region_id, RegionTaskArgError &error
) {
    payload = RegionPartLocalSpan{0, 0};
    counter = RegionPartLocalSpan{0, 0};
    session_bits = 0;
    transaction_id = 0;
    region_id = 0;
    error = RegionTaskArgError::OK;
    if (scalars == nullptr) {
        error = RegionTaskArgError::NULL_POINTER;
        return false;
    }
    if (scalar_count != REGION_TASK_SCALAR_COUNT) {
        error = RegionTaskArgError::BAD_SCALAR_COUNT;
        return false;
    }
    if (scalars[0] != REGION_TASK_MAGIC_VERSION) {
        error = RegionTaskArgError::BAD_MAGIC_VERSION;
        return false;
    }
    if (scalars[1] == 0) {
        error = RegionTaskArgError::BAD_REGION_ID;
        return false;
    }
    if (scalars[7] == 0) {
        error = RegionTaskArgError::BAD_TRANSACTION_ID;
        return false;
    }
    RegionPartLocalSpan decoded_payload{scalars[2], scalars[3]};
    RegionPartLocalSpan decoded_counter{scalars[4], scalars[5]};
    if (!region_validate_payload_span(decoded_payload)) {
        error = RegionTaskArgError::BAD_PAYLOAD_RANGE;
        return false;
    }
    if (!region_validate_counter_span(decoded_counter) || region_spans_overlap(decoded_payload, decoded_counter)) {
        error = RegionTaskArgError::BAD_COUNTER_RANGE;
        return false;
    }
    payload = decoded_payload;
    counter = decoded_counter;
    session_bits = scalars[6];
    transaction_id = scalars[7];
    region_id = scalars[1];
    return true;
}
