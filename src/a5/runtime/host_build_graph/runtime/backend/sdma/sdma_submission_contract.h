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

#include "async_completion_writer.h"

enum class SdmaOp : uint8_t { TGET = 0, TPUT = 1 };

// The pinned public SDMA transfer API accepts a flat contiguous logical line
// and counts its elements in uint32. Validate public accessors before submit:
// a truncated count must not disguise a failed transfer as an empty event.
// Ownership remains the caller's existing run/domain contract.
template <typename DstTensor, typename SrcTensor>
inline __aicore__ bool async_sdma_transfer_bytes(DstTensor &dst, SrcTensor &src, uint64_t *bytes) {
    static_assert(std::is_same_v<typename DstTensor::RawDType, typename SrcTensor::RawDType>);
    if (bytes == nullptr) return false;
    *bytes = 0;
    const int64_t count = src.GetShape(4), dst_count = dst.GetShape(4);
    if (count < 0 || dst_count < count || static_cast<uint64_t>(count) > UINT32_MAX ||
        static_cast<uint64_t>(dst_count) > UINT32_MAX || src.GetStride(4) != 1 || dst.GetStride(4) != 1)
        return false;
    for (uint32_t dim = 0; dim < 4; ++dim)
        if (src.GetShape(dim) != 1 || dst.GetShape(dim) != 1 || src.GetStride(dim) != count ||
            dst.GetStride(dim) != dst_count)
            return false;
    const uint64_t width = sizeof(typename SrcTensor::RawDType);
    if (static_cast<uint64_t>(count) > UINT64_MAX / width) return false;
    const uint64_t span = static_cast<uint64_t>(count) * width;
    const uint64_t source = reinterpret_cast<uint64_t>(src.data());
    const uint64_t destination = reinterpret_cast<uint64_t>(dst.data());
    if (source == 0 || destination == 0 || span > UINT64_MAX - source || span > UINT64_MAX - destination) return false;
    *bytes = span;
    return true;
}

inline __aicore__ bool async_sdma_event_valid(
    SdmaOp operation, uint64_t bytes, uint64_t handle, bool post_failed, uint32_t engine, uint32_t sdma_engine
) {
    if (post_failed || engine != sdma_engine || (operation != SdmaOp::TGET && operation != SdmaOp::TPUT)) return false;
    // A5 TPUT completes via the pinned synchronous MTE fallback and returns
    // zero. Nonempty TGET has no such success path: zero also reports a failed
    // submit and must not release the task's dependents as a successful wait.
    return handle != 0 || bytes == 0 || operation == SdmaOp::TPUT;
}
