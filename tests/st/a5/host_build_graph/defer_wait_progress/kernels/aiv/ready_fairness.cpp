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
#include <cstdint>
#include <pto/pto-inst.hpp>

#include "async_kernel_api.h"
#include "scheduler/gm.h"
#include "tensor.h"

namespace {
constexpr uint32_t kLineWords = 32;

}  // namespace

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    const auto *tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    auto *out = reinterpret_cast<__gm__ uint32_t *>(tensor->buffer.addr) + tensor->start_offset;
    const uint32_t role = static_cast<uint32_t>(args[1]);
    const uint32_t index = static_cast<uint32_t>(args[2]);
    const uint32_t chain_width = static_cast<uint32_t>(args[3]);
    auto ctx = get_async_ctx(args);
    if (role == 0) {
        save_expected_notification_counter(ctx, reinterpret_cast<__gm__ int32_t *>(out), 1);
        percounter_gm_store(out[kLineWords], UINT32_C(1));
    } else if (role == 1) {
        // The first background task returns physically but stays logically
        // pending until the waiter-dependent consumer acknowledges it.
        if (index == 0)
            save_expected_notification_counter(ctx, reinterpret_cast<__gm__ int32_t *>(out + 3 * kLineWords), 1);
        const bool predecessor_acknowledged = index != chain_width || percounter_gm_read(out[3 * kLineWords]) == 1;
        percounter_gm_store(out[(index + 4) * kLineWords], predecessor_acknowledged ? index + 1 : 0);
        if (index == 0) percounter_gm_store(out[0], UINT32_C(1));
    } else if (role == 2) {
        // This dependency consumes the runtime's actual completion source.
        percounter_gm_store(out[2 * kLineWords], percounter_gm_read(out[4 * kLineWords]));
        percounter_gm_store(out[3 * kLineWords], percounter_gm_read(out[0]));
    } else if (role == 3) {
        save_expected_notification_counter(ctx, reinterpret_cast<__gm__ int32_t *>(out + kLineWords), 1);
    }
}
