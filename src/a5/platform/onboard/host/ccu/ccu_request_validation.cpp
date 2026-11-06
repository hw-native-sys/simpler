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
#include "host/ccu_request_validation.h"

#include <limits>

namespace {
template <typename T, size_t N>
bool all_zero(const T (&values)[N]) {
    for (size_t i = 0; i < N; ++i)
        if (values[i] != 0) return false;
    return true;
}
bool add_fits(uint64_t a, uint64_t b) { return b <= std::numeric_limits<uint64_t>::max() - a; }
}  // namespace

CcuRequestValidationError ccu_validate_request_v1(const CcuRequestDescriptorV1 &r) {
    if (r.abi_version != CCU_EXECUTION_ABI_V1 || r.struct_size != sizeof(r))
        return CcuRequestValidationError::ABI_MISMATCH;
    if (r.context_id == 0 || r.context_generation == 0 || r.request_seq == 0 || r.program_id == 0 ||
        r.collective_id == 0)
        return CcuRequestValidationError::INVALID_IDENTITY;
    if (r.operation != CCU_OPERATION_BROADCAST || r.dtype != CCU_DATA_TYPE_FLOAT32)
        return CcuRequestValidationError::UNSUPPORTED_OPERATION;
    if (r.rank_count != CCU_EXECUTION_RANK_COUNT_V1 || r.root >= r.rank_count)
        return CcuRequestValidationError::INVALID_RANK;
    if (r.element_count == 0 || r.element_count > CCU_EXECUTION_MAX_COUNT_V1 ||
        r.element_count > std::numeric_limits<uint64_t>::max() / sizeof(float))
        return CcuRequestValidationError::INVALID_COUNT;
    if (r.input_window_id == 0 || r.output_window_id == 0) return CcuRequestValidationError::INVALID_WINDOW;
    if (!all_zero(r.reserved)) return CcuRequestValidationError::RESERVED_FIELD_NONZERO;
    return CcuRequestValidationError::NONE;
}

CcuRequestValidationError ccu_validate_window_ranges_v1(
    const CcuRequestDescriptorV1 &r, const CcuWindowRangeV1 &in, const CcuWindowRangeV1 &out
) {
    auto e = ccu_validate_request_v1(r);
    if (e != CcuRequestValidationError::NONE) return e;
    if (in.window_id != r.input_window_id || out.window_id != r.output_window_id || in.allocation_id == 0 ||
        out.allocation_id == 0 || in.allocation_generation == 0 || out.allocation_generation == 0)
        return CcuRequestValidationError::INVALID_WINDOW;
    const uint64_t bytes = r.element_count * sizeof(float);
    const uint64_t in_end = r.input_offset + bytes;
    const uint64_t out_end = r.output_offset + bytes;
    if (!add_fits(r.input_offset, bytes) || !add_fits(r.output_offset, bytes) || in_end > in.extent_bytes ||
        out_end > out.extent_bytes || !add_fits(in.allocation_offset_bytes, r.input_offset) ||
        !add_fits(out.allocation_offset_bytes, r.output_offset))
        return CcuRequestValidationError::WINDOW_OUT_OF_RANGE;
    if (in.allocation_id == out.allocation_id) {
        const uint64_t a = in.allocation_offset_bytes + r.input_offset;
        const uint64_t b = out.allocation_offset_bytes + r.output_offset;
        if (!add_fits(a, bytes) || !add_fits(b, bytes)) return CcuRequestValidationError::WINDOW_OUT_OF_RANGE;
        if (a < b + bytes && b < a + bytes) return CcuRequestValidationError::OVERLAPPING_WINDOWS;
    }
    return CcuRequestValidationError::NONE;
}

CcuRequestValidationError ccu_validate_trigger_v1(const CcuTriggerInfoV1 &t) {
    if (t.abi_version != CCU_EXECUTION_ABI_V1 || t.struct_size != sizeof(t) || t.context_id == 0 ||
        t.context_generation == 0 || t.request_seq == 0 || t.gate_va == 0 || t.completion_record_va == 0 ||
        t.gate_mask != 1 || t.trigger_owner != CCU_TRIGGER_OWNER_HOST || !all_zero(t.reserved))
        return CcuRequestValidationError::INVALID_TRIGGER;
    return CcuRequestValidationError::NONE;
}

CcuRequestValidationError ccu_validate_completion_header_v1(const CcuCompletionRecordV1 &r) {
    if (r.abi_version != CCU_EXECUTION_ABI_V1 || r.struct_size != sizeof(r) || r.context_id == 0 ||
        r.context_generation == 0 || !all_zero(r.reserved_header) || !all_zero(r.reserved_done))
        return CcuRequestValidationError::INVALID_COMPLETION;
    return CcuRequestValidationError::NONE;
}
