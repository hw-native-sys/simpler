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

#include <stdint.h>
#include <pto/comm/async_common/async_types.hpp>

#include "async_kernel_api.h"

enum class SdmaOp : uint8_t {
    TGET = 0,
    TPUT = 1,
};

namespace pto2::detail {

template <typename GlobalData>
inline __aicore__ bool flat_transfer_bytes(GlobalData &tensor, uint64_t &bytes) {
    const int64_t elements = tensor.GetShape(pto::GlobalTensorDim::DIM_4);
    if (elements < 0 || static_cast<uint64_t>(elements) > UINT32_MAX ||
        tensor.GetStride(pto::GlobalTensorDim::DIM_4) != 1)
        return false;
    for (uint32_t dimension = 0; dimension < 4; ++dimension) {
        const int dim = static_cast<int>(dimension);
        if (tensor.GetShape(dim) != 1 || tensor.GetStride(dim) != elements) return false;
    }
    if (static_cast<uint64_t>(elements) > UINT64_MAX / sizeof(typename GlobalData::RawDType)) return false;
    bytes = static_cast<uint64_t>(elements) * sizeof(typename GlobalData::RawDType);
    return bytes == 0 || (tensor.data() != nullptr && bytes <= UINT64_MAX - reinterpret_cast<uint64_t>(tensor.data()));
}

// Count/At are the SDK's public record interface. The reservation is acquired
// before submission, so a partial registration can never become zero-work success.
template <typename PtoAsyncEvent, typename PtoAsyncSession>
inline __aicore__ bool register_pto_async_event(
    AsyncCtx &ctx, const PtoAsyncEvent &event, const PtoAsyncSession &session,
    const pto2::resident::AsyncReservation &reservation, uint32_t record_limit, SdmaOp operation = SdmaOp::TGET
) {
    if (event.postFailed || event.engine != pto::comm::DmaEngine::SDMA ||
        (operation != SdmaOp::TGET && operation != SdmaOp::TPUT)) {
        defer_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
        return false;
    }
    if (event.handle == 0) {
        // A5 TPUT completes through the SDK's synchronous MTE fallback. It
        // owes no post-done record; nonempty TGET still requires one.
        if (operation != SdmaOp::TPUT || !async_ctx_is_deferred(ctx) || ctx.completion_count == nullptr ||
            ctx.completion_entries == nullptr) {
            defer_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
            return false;
        }
        if (ctx.completion_error_code != nullptr && *ctx.completion_error_code != 0) return false;
        return ctx.resident_enabled == 0 || pto2::resident::finish_reservation(ctx.resident, reservation, 0);
    }
    const uint32_t count = event.CompletionRecordCount(session);
    if (count == 0 || count > record_limit) {
        defer_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
        return false;
    }
    for (uint32_t index = 0; index < count; ++index) {
        const auto record = event.CompletionRecordAt(session, index);
        if (record.kind != pto::comm::CompletionKind::SDMA_POST_DONE) {
            defer_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
            return false;
        }
        const CompletionToken token{
            reinterpret_cast<uint64_t>(record.addr), 0, COMPLETION_ENGINE_SDMA, COMPLETION_TYPE_SDMA_EVENT_RECORD,
            record.expected
        };
        if (!pto2::resident::detail::token_valid(token)) {
            defer_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
            return false;
        }
        const bool registered = ctx.resident_enabled != 0 ?
                                    pto2::resident::commit_condition(ctx.resident, reservation, token) :
                                    register_completion_condition(ctx, token);
        if (!registered) return false;
    }
    return ctx.resident_enabled == 0 || pto2::resident::finish_reservation(ctx.resident, reservation, count);
}

}  // namespace pto2::detail
