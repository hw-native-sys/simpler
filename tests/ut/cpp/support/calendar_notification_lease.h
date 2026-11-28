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

#include "host/scheduler_notifications.h"

namespace scheduler_test {
inline int acquire_notifications(SchedulerNotificationPurpose purpose, size_t count, SchedulerDoneNotifications *out) {
    if (out == nullptr) return -1;
    SchedulerDoneNotifications notifications{std::vector<uint64_t>(count), std::vector<uint64_t>(count), {}};
    if (purpose == SchedulerNotificationPurpose::CALENDAR_COUNTER) {
        struct alignas(32) CounterSlot {
            uint32_t set{0};
        };
        auto slots = std::make_shared<std::vector<CounterSlot>>(count);
        for (size_t i = 0; i < count; ++i) {
            notifications.addresses[i] = reinterpret_cast<uint64_t>(&(*slots)[i].set);
            notifications.add_addresses[i] = notifications.addresses[i] + 8;
        }
        notifications.owner = std::move(slots);
    }
    *out = std::move(notifications);
    return 0;
}
}  // namespace scheduler_test
