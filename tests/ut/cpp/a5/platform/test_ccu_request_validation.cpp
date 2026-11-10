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
#include <gtest/gtest.h>

#include <limits>

#include "host/ccu_request_validation.h"

namespace {
CcuRequestDescriptorV1 request() {
    CcuRequestDescriptorV1 r{};
    r.abi_version = CCU_EXECUTION_ABI_V1;
    r.struct_size = sizeof(r);
    r.operation = CCU_OPERATION_BROADCAST;
    r.dtype = CCU_DATA_TYPE_FLOAT32;
    r.context_id = 1;
    r.context_generation = 1;
    r.request_seq = 1;
    r.program_id = 1;
    r.input_window_id = 1;
    r.output_window_id = 2;
    r.element_count = 64;
    r.rank_count = 2;
    r.collective_id = 1;
    return r;
}
}  // namespace

TEST(CcuRequestValidation, RejectsStaleAbiUnknownOperationsAndOversizeCount) {
    auto r = request();
    EXPECT_EQ(ccu_validate_request_v1(r), CcuRequestValidationError::NONE);
    r.struct_size--;
    EXPECT_EQ(ccu_validate_request_v1(r), CcuRequestValidationError::ABI_MISMATCH);
    r = request();
    r.operation = 99;
    EXPECT_EQ(ccu_validate_request_v1(r), CcuRequestValidationError::UNSUPPORTED_OPERATION);
    r = request();
    r.element_count = CCU_EXECUTION_MAX_COUNT_V1 + 1;
    EXPECT_EQ(ccu_validate_request_v1(r), CcuRequestValidationError::INVALID_COUNT);
}

TEST(CcuRequestValidation, ChecksBoundsWithoutOverflowAndRejectsAliasingOnlyWithinSameAllocation) {
    auto r = request();
    CcuWindowRangeV1 in{1, 10, 1, 0, 4096, 0};
    CcuWindowRangeV1 out{2, 10, 1, 0, 4096, 0};
    EXPECT_EQ(ccu_validate_window_ranges_v1(r, in, out), CcuRequestValidationError::OVERLAPPING_WINDOWS);
    out.allocation_offset_bytes = 1024;
    EXPECT_EQ(ccu_validate_window_ranges_v1(r, in, out), CcuRequestValidationError::NONE);
    out.allocation_id = 11;
    out.allocation_offset_bytes = 0;
    EXPECT_EQ(ccu_validate_window_ranges_v1(r, in, out), CcuRequestValidationError::NONE);
    r.input_offset = std::numeric_limits<uint64_t>::max() - 3;
    EXPECT_EQ(ccu_validate_window_ranges_v1(r, in, out), CcuRequestValidationError::WINDOW_OUT_OF_RANGE);
}

TEST(CcuRequestValidation, ValidatesTriggerAndCompletionIdentity) {
    CcuTriggerInfoV1 t{};
    t.abi_version = CCU_EXECUTION_ABI_V1;
    t.struct_size = sizeof(t);
    t.context_id = t.context_generation = t.request_seq = t.gate_va = t.completion_record_va = 1;
    t.gate_mask = 1;
    t.trigger_owner = CCU_TRIGGER_OWNER_HOST;
    EXPECT_EQ(ccu_validate_trigger_v1(t), CcuRequestValidationError::NONE);
    t.gate_mask = 2;
    EXPECT_EQ(ccu_validate_trigger_v1(t), CcuRequestValidationError::INVALID_TRIGGER);
    CcuCompletionRecordV1 record{};
    record.abi_version = CCU_EXECUTION_ABI_V1;
    record.struct_size = sizeof(record);
    record.context_id = record.context_generation = 1;
    EXPECT_EQ(ccu_validate_completion_header_v1(record), CcuRequestValidationError::NONE);
    record.reserved_done[0] = 1;
    EXPECT_EQ(ccu_validate_completion_header_v1(record), CcuRequestValidationError::INVALID_COMPLETION);
}
