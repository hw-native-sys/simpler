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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

// Nonzero entries name owned, non-consuming 32-bit SET/read registers.
// Zero entries use the task's canonical GM completion state.
struct SchedulerDoneNotifications {
    std::vector<uint64_t> addresses;
    std::shared_ptr<const void> owner;
};

SchedulerDoneNotifications create_scheduler_done_notifications(size_t count);
