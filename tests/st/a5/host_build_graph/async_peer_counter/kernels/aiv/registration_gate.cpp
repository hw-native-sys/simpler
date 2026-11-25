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

using namespace peer_counter;

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    auto *registrations = data<uint64_t>(args[0]);
    auto *diagnostic = data<uint64_t>(args[1]);
    const int64_t rows = args[2], capacity = args[3], epoch = args[4], tag = args[5];
    begin(diagnostic, Role::GATE, epoch, tag);
    record(diagnostic, Field::ROWS, rows);
    record(diagnostic, Field::CAPACITY, capacity);
    if (!parameters(epoch, tag) || rows < 1 || rows > kMaxRows || capacity < 1 || capacity > kMaxRows) {
        error(diagnostic, Error::BAD_ARGUMENT);
        finish(diagnostic);
        return;
    }
    const uint64_t started = now();
    uint64_t observed = observe(registrations);
    while (observed < static_cast<uint64_t>(rows)) {
        if (now() - started >= kGateTimeoutMs * ticks_per_ms()) {
            record(diagnostic, Field::TIMEOUT);
            error(diagnostic, Error::GATE_TIMEOUT);
            break;
        }
        observed = observe(registrations);
    }
    record(diagnostic, Field::REGISTERED, static_cast<int64_t>(observed));
    if (observed > static_cast<uint64_t>(rows)) error(diagnostic, Error::REGISTRATION_MISMATCH);
    finish(diagnostic);
}
