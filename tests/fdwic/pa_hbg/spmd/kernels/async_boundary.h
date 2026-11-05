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

#include <cstdint>
#include <pto/pto-inst.hpp>

#include "async_kernel_api.h"
#include "record.h"

static inline __aicore__ void record_async_boundary(__gm__ int64_t *args, int32_t slot) {
    const auto expected_blocks = static_cast<int32_t>(args[1]);
    const auto mask = static_cast<uint32_t>(args[2]);
    if (args[SPMD_LOCAL_CONTEXT_INDEX] == 0 || args[SPMD_GLOBAL_CONTEXT_INDEX] == 0) {
        record_execution(args, slot);
        return;
    }
    const auto block = get_block_idx(args);
    if (block < 0 || block >= expected_blocks || get_block_num(args) != expected_blocks || !(mask & (1U << slot))) {
        record_execution(args, slot);
        return;
    }
    auto *records = reinterpret_cast<__gm__ ExecutionRecord *>(tensor_data(args[0]));
    auto *record = &records[block * 3 + slot];
    AsyncCtx ctx = get_async_ctx(args);
    // Bits 0..4: valid token, non-null count/error/entries, nonzero capacity.
    record->metadata_padding[0] = static_cast<int64_t>(async_ctx_is_deferred(ctx)) |
                                  (static_cast<int64_t>(ctx.completion_count != nullptr) << 1) |
                                  (static_cast<int64_t>(ctx.completion_error_code != nullptr) << 2) |
                                  (static_cast<int64_t>(ctx.completion_entries != nullptr) << 3) |
                                  (static_cast<int64_t>(ctx.completion_capacity != 0) << 4);
    // This allocated counter remains zero even if an incorrect legacy path registers it.
    CompletionToken token{
        reinterpret_cast<uint64_t>(&record->atomic_padding[1]), 0, COMPLETION_ENGINE_SDMA, COMPLETION_TYPE_COUNTER, 0
    };
    record->metadata_padding[1] = static_cast<int64_t>(register_completion_condition(ctx, token));
    record->metadata_padding[2] = static_cast<int64_t>(register_completion_condition(ctx, token));
    record->metadata_padding[3] = 0xC34;
    // Fanin observes valid identity only after both API calls and the sentinel write.
    record_execution(args, slot);
}
