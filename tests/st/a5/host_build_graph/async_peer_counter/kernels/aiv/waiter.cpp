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

using namespace peer_counter;

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    auto *signal = data<int32_t>(args[0]);
    auto *diagnostic = data<uint64_t>(args[2]);
    auto *registrations = data<uint64_t>(args[3]);
    const int64_t epoch = args[4], tag = args[5];
    begin(diagnostic, Role::WAITER, epoch, tag);
    if (!parameters(epoch, tag)) {
        error(diagnostic, Error::BAD_ARGUMENT);
        finish(diagnostic);
        return;
    }
    record(diagnostic, Field::SEEN_SIGNAL, signal_value(signal));
    auto context = get_async_ctx(args);
    if (!async_ctx_is_deferred(context) || context.completion_count == nullptr ||
        context.completion_error_code == nullptr || context.completion_entries == nullptr ||
        context.completion_capacity != MAX_COMPLETIONS_PER_TASK) {
        error(diagnostic, Error::MISSING_CONTEXT);
    } else if (*context.completion_count != 0 || *context.completion_error_code != 0) {
        error(diagnostic, Error::REGISTRATION_FAILED);
    } else {
        save_expected_notification_counter(context, signal, static_cast<uint32_t>(epoch));
        const int32_t writer_error = *context.completion_error_code;
        if (writer_error != 0) error(diagnostic, static_cast<int64_t>(writer_error));
        else if (*context.completion_count != 1) error(diagnostic, Error::REGISTRATION_FAILED);
        else {
            record(diagnostic, Field::REGISTERED);
            atomic_add(registrations, 1);
        }
    }
    finish(diagnostic);
}
