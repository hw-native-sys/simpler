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

enum class SchedulerNotificationPurpose : uint32_t {
    TASK_STATE,
    MIX_ACK,
    COMPLETION_COUNTER,
    HBG_REQUIRED_TASK_STATE,
    HBG_REQUIRED_COMPLETION_COUNTER,
    CLAIM_SUPPLY,
    BOOTSTRAP_ARRIVAL,
    CALENDAR_COUNTER
};

inline constexpr bool scheduler_notification_is_required(SchedulerNotificationPurpose purpose) {
    return purpose == SchedulerNotificationPurpose::HBG_REQUIRED_TASK_STATE ||
           purpose == SchedulerNotificationPurpose::HBG_REQUIRED_COMPLETION_COUNTER ||
           purpose == SchedulerNotificationPurpose::CALENDAR_COUNTER;
}

inline constexpr bool scheduler_notification_uses_counter(SchedulerNotificationPurpose purpose) {
    return purpose == SchedulerNotificationPurpose::COMPLETION_COUNTER ||
           purpose == SchedulerNotificationPurpose::HBG_REQUIRED_COMPLETION_COUNTER ||
           purpose == SchedulerNotificationPurpose::CALENDAR_COUNTER;
}

struct SchedulerDoneNotifications {
    std::vector<uint64_t> addresses;
    std::vector<uint64_t> add_addresses;
    std::shared_ptr<const void> owner;
};
