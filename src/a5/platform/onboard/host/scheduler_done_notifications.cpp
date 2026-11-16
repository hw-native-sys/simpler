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
#include "host/scheduler_done_notifications.h"
#include "common/unified_log.h"

#include <acl/acl.h>
#include <rt_external_device.h>
#include <stdexcept>
#include <string>

namespace {
struct Notification {
    aclrtCntNotify handle{nullptr};
    rtDevResInfo mapping{0, RT_PROCESS_CP1, RT_RES_TYPE_STARS_CNT_NOTIFY_RECORD, 0, 0};
    bool mapped{false};
};

struct NotificationPool {
    aclrtContext context{nullptr};
    std::vector<Notification> notifications;

    ~NotificationPool() {
        aclrtContext previous = nullptr;
        const auto previous_rc = aclrtGetCurrentContext(&previous);
        if (context != nullptr && (previous_rc != ACL_SUCCESS || previous != context) &&
            aclrtSetCurrentContext(context) != ACL_SUCCESS) {
            LOG_ERROR("A5 HBG: cannot select context to release CntNotify resources");
            return;
        }
        for (auto &notification : notifications) {
            if (notification.mapped) {
                const auto rc = rtReleaseDevResAddress(&notification.mapping);
                if (rc != 0) LOG_ERROR("A5 HBG: CntNotify unmap failed: %d", rc);
            }
            if (notification.handle != nullptr) {
                const auto rc = aclrtCntNotifyDestroy(notification.handle);
                if (rc != ACL_SUCCESS) LOG_ERROR("A5 HBG: CntNotify destroy failed: %d", rc);
            }
        }
        if (previous_rc == ACL_SUCCESS && previous != nullptr && previous != context &&
            aclrtSetCurrentContext(previous) != ACL_SUCCESS)
            LOG_ERROR("A5 HBG: cannot restore context after CntNotify release");
    }
};

void check(int rc, const char *operation) {
    if (rc != 0) throw std::runtime_error(std::string("CntNotify ") + operation + ": " + std::to_string(rc));
}
}  // namespace

SchedulerDoneNotifications create_scheduler_done_notifications(size_t count) {
    SchedulerDoneNotifications result{std::vector<uint64_t>(count, 0), {}};
    if (count == 0) return result;
    auto pool = std::make_shared<NotificationPool>();
    check(aclrtGetCurrentContext(&pool->context), "get context");
    pool->notifications.reserve(count);
    for (size_t index = 0; index < count; ++index) {
        pool->notifications.emplace_back();
        auto &notification = pool->notifications.back();
        const auto rc = aclrtCntNotifyCreate(&notification.handle, 0);
        if (rc == ACL_ERROR_RT_NO_NOTIFY_RESOURCE) {
            pool->notifications.pop_back();
            LOG_WARN("A5 HBG: CntNotify capacity exhausted at %zu/%zu nodes; remaining nodes use GM", index, count);
            break;
        }
        check(rc, "create");
        check(aclrtCntNotifyGetId(notification.handle, &notification.mapping.resId), "get id");
        uint32_t bytes = 0;
        rtDevResAddrInfo address{&result.addresses[index], &bytes};
        check(rtGetDevResAddress(&notification.mapping, &address), "map");
        notification.mapped = true;
        if (result.addresses[index] == 0 || result.addresses[index] % alignof(uint32_t) != 0 ||
            bytes < sizeof(uint32_t))
            throw std::runtime_error("CntNotify invalid 32-bit register mapping");
    }
    LOG_INFO("A5 HBG: CntNotify DONE coverage %zu/%zu nodes", pool->notifications.size(), count);
    result.owner = std::move(pool);
    return result;
}
