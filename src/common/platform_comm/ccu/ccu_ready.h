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

#include "ccu_completion.h"
#include "ccu_device_submission.h"

namespace simpler::ccu {

// Called by the local AICPU initializer before any AIV task is dispatched.
template <typename Io = EventIo, typename Expired, typename Spin>
int wait_local_ready(uint64_t address, Expired expired, Spin spin) {
    if (address == 0) return 0;
    if (address % alignof(SimplerCcuWorkspace) != 0) return -1;
    const auto *workspace = reinterpret_cast<const SimplerCcuWorkspace *>(address);
    if (workspace->ccu_count == 0 || workspace->ccu_count > 1024) return -1;
    for (uint32_t id = 0; id < workspace->ccu_count; ++id) {
        const auto *control = ccu_control(workspace, id);
        if (control == nullptr) return -1;
        for (;;) {
            const auto result = poll_completion<Io>(control->ready.address, control->ready.mask);
            if (result.state == CompletionPollState::ready) break;
            if (result.state != CompletionPollState::pending) return -1;
            if (expired()) return -2;
            spin();
        }
    }
    return 0;
}
}  // namespace simpler::ccu
