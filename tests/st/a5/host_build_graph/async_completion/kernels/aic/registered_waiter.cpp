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

#include "../waiter.h"

// The local marker acknowledges the finite registration body, never TaskDone.
extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    async_case::wait_task(args, 0);
    const int64_t execution = args[5] + get_block_idx(args) * 3;
    auto *signals = async_case::data<int32_t>(args[0]);
    async_case::publish_signal(async_case::condition(signals, execution, async_case::kConditionLimit), 1);
}
