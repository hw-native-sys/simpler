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

#include "entry_args.h"
#include "kernel_invocation.h"

namespace simpler::tmr {

// out is caller-owned invocation storage and outlives config/orchestration
// and every TensorRef derived from it. It is not a persistent Runtime prefix.
inline InvocationStatus materialize_tmr_entry_args(const TmrInvocationView &view, EntryArgsStorage *out) noexcept {
    if (out == nullptr || !view.valid()) return InvocationStatus::InvalidArgument;
    out->clear();
    for (int32_t i = 0; i < view.tensor_count(); ++i) {
        ChipTensor tensor{};
        view.tensor(i, &tensor);
        out->tensors_[i] = Tensor::from_boundary(tensor);
    }
    for (int32_t i = 0; i < view.scalar_count(); ++i)
        view.scalar(i, &out->scalars_[i]);
    out->tensor_count_ = view.tensor_count();
    out->scalar_count_ = view.scalar_count();
    return InvocationStatus::Ok;
}

// trusted views are resolved independently of packet before this call. No
// execution state is published until admission succeeds. out stays unchanged
// on rejection and owns the converted arguments through their final use.
inline InvocationStatus consume_tmr_invocation(
    ByteSpan packet, const PreparedInvocationView &callable, const TmrExecutionBindingView &binding,
    EntryArgsStorage *out
) noexcept {
    if (out == nullptr) return InvocationStatus::InvalidArgument;
    TmrInvocationView view;
    const auto status = decode_tmr_invocation(packet, callable, binding, &view);
    if (status != InvocationStatus::Ok) return status;
    return materialize_tmr_entry_args(view, out);
}

}  // namespace simpler::tmr
