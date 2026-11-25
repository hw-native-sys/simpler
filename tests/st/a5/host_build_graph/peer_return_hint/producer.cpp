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

#include <pto/pto-inst.hpp>

#include "async_kernel_api.h"

#include "device.h"

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    auto *state = peer_hint_test::state(args);
    // Keep the receiver waiting for the same amount of producer work in both runs.
    peer_hint_test::wait_ticks(8000);
    if (args[1] != 0) {
        auto ctx = get_async_ctx(args);
        if (!peer_hint_test::receiver_checked_pending(args, false)) return;
        save_expected_notification_counter(ctx, state + peer_hint_test::kSignal, 1);
    }
    peer_hint_test::publish(state + peer_hint_test::kProduced, 41);
}
