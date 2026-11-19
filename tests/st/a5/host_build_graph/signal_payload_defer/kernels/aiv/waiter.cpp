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
#include "../payload.h"
extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    auto *signals = payload_case::words(args, 0);
    const uint32_t rank = static_cast<uint32_t>(args[1]);
    const uint32_t ranks = static_cast<uint32_t>(args[2]);
    const uint32_t epoch = static_cast<uint32_t>(args[3]);
    auto context = get_async_ctx(args);
    for (uint32_t peer = 0; peer < ranks; ++peer)
        if (peer != rank) save_expected_notification_counter(context, signals + peer * 32, epoch);
}
