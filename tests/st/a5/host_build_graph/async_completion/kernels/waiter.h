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

#include "device.h"
#include "async_kernel_api.h"

namespace async_case {

inline __aicore__ void wait_task(__gm__ int64_t *args, int64_t lane) {
    auto *signals = data<int32_t>(args[0]);
    auto *records = data<uint64_t>(args[2]);
    auto *control = data<uint64_t>(args[4]);
    const int64_t execution = args[5] + get_block_idx(args) * 3 + lane;
    auto *record = records + execution * kObservationWords;
    stamp(record, Observation::VISITS);
    AsyncCtx ctx = get_async_ctx(args);
    if (!async_ctx_is_deferred(ctx)) {
        stamp(record, Observation::MISSING_CONTEXT);
    } else {
        for (int64_t index = 0; index < args[6]; ++index)
            save_expected_notification_counter(ctx, condition(signals, execution, index), args[7]);
    }
    // Test-owned acknowledgement: this is registration progress, not TaskDone.
    add(cell(control, Control::ARRIVED, args[8]), 1);
    publish_signal(condition(signals, execution, kConditionLimit), static_cast<int32_t>(args[10]));
    if (get_block_idx(args) + 1 == get_block_num(args)) delay(args[9]);
    stamp(record, Observation::BODY_DONE);
}

}  // namespace async_case
