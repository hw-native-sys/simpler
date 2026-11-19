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

#include "../device.h"
#include "async_kernel_api.h"

// Intentional negative control: preserve the real deferred context and waiter
// bookkeeping, but omit its requested business-condition registration.
extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    auto *records = async_case::data<uint64_t>(args[2]);
    auto *control = async_case::data<uint64_t>(args[4]);
    const int64_t execution = args[5] + get_block_idx(args) * 3 + 1;
    auto *record = records + execution * async_case::kObservationWords;
    async_case::stamp(record, async_case::Observation::VISITS);
    AsyncCtx ctx = get_async_ctx(args);
    if (!async_ctx_is_deferred(ctx)) async_case::stamp(record, async_case::Observation::MISSING_CONTEXT);
    async_case::add(async_case::cell(control, async_case::Control::ARRIVED, args[8]), 1);
    if (get_block_idx(args) + 1 == get_block_num(args)) async_case::delay(args[9]);
    async_case::stamp(record, async_case::Observation::BODY_DONE);
}
