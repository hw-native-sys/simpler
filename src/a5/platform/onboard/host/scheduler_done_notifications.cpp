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
#include "device_runner.h"
#include "host/scheduler_notification_address.h"
#include "common/unified_log.h"

#include <acl/acl.h>
#include <rt_external_device.h>
#include <stdexcept>
#include <string>

namespace {
void check(int rc, const char *operation) {
    if (rc != 0) throw std::runtime_error(std::string("CntNotify ") + operation + ": " + std::to_string(rc));
}

struct Notification {
    aclrtCntNotify handle{nullptr};
    rtDevResInfo mapping{0, RT_PROCESS_CP1, RT_RES_TYPE_STARS_CNT_NOTIFY_RECORD, 0, 0};
    uint64_t address{0};
    uint64_t add_address{0};
    bool mapped{false};
};

class NotificationPool final : public SchedulerNotificationStorage {
public:
    NotificationPool(void *context, SchedulerNotificationPurpose purpose) :
        SchedulerNotificationStorage(context),
        counter_(scheduler_notification_uses_counter(purpose)),
        required_(scheduler_notification_is_required(purpose)) {}
    size_t capacity() const override { return notifications_.size(); }

    SchedulerDoneNotifications prepare(size_t count) override {
        if (retiring) throw std::invalid_argument("invalid CntNotify pool acquire");
        SchedulerDoneNotifications result{std::vector<uint64_t>(count, 0), std::vector<uint64_t>(count, 0), {}};
        notifications_.reserve(count);
        while (notifications_.size() < count) {
            notifications_.emplace_back();
            auto &notification = notifications_.back();
            const auto rc = aclrtCntNotifyCreate(&notification.handle, 0);
            if (rc == ACL_ERROR_RT_NO_NOTIFY_RESOURCE || rc == ACL_ERROR_RT_RESOURCE_ALLOC_FAIL) {
                notifications_.pop_back();
                if (required_) throw SchedulerNotificationQuotaExceeded("CntNotify create: " + std::to_string(rc));
                LOG_WARN(
                    "A5 scheduler: optional CntNotify capacity exhausted at %zu/%zu", notifications_.size(), count
                );
                break;
            }
            check(rc, "create");
            check(aclrtCntNotifyGetId(notification.handle, &notification.mapping.resId), "get id");
            map(notification.mapping, notification.address, notification.mapped);
            if (required_ && a5_cnt_notify_increment_address(notification.address) == 0)
                throw std::runtime_error("scheduler requires an aligned CntNotify SET slot");
            if (counter_) {
                // A5 CntNotify owns a 32-byte register slot. Its ADD register
                // is offset 8 from the mapped SET/read base and shares the SET
                // mapping and handle lifetime; it has no independent unmap.
                notification.add_address = a5_cnt_notify_increment_address(notification.address);
            }
        }
        size_t unavailable_counters = 0;
        for (size_t index = 0; index < count && index < notifications_.size(); ++index) {
            auto &notification = notifications_[index];
            if (!notification.mapped) throw std::runtime_error("CntNotify pool has an incomplete mapping");
            result.addresses[index] = notification.address;
            if (counter_) {
                result.add_addresses[index] = notification.add_address;
                if (notification.add_address == 0) ++unavailable_counters;
            }
        }
        if (unavailable_counters != 0)
            LOG_WARN(
                "A5 HBG: %zu CntNotify mappings have no aligned INC alias; use SET/GM completion", unavailable_counters
            );
        return result;
    }

    int release() override {
        retiring = true;
        aclrtContext previous = nullptr;
        const auto previous_rc = aclrtGetCurrentContext(&previous);
        if (previous_rc != ACL_SUCCESS || previous != context) {
            const auto rc = aclrtSetCurrentContext(context);
            if (rc != ACL_SUCCESS) return rc;
        }
        int error = 0;
        auto note = [&error](int rc) {
            if (error == 0 && rc != 0) error = rc;
        };
        for (auto &notification : notifications_) {
            if (notification.mapped) {
                const auto rc = rtReleaseDevResAddress(&notification.mapping);
                note(rc);
                if (rc == 0) notification.mapped = false;
            }
            if (!notification.mapped && notification.handle != nullptr) {
                const auto rc = aclrtCntNotifyDestroy(notification.handle);
                note(rc);
                if (rc == ACL_SUCCESS) notification.handle = nullptr;
            }
        }
        if (previous_rc == ACL_SUCCESS && previous != nullptr && previous != context)
            note(aclrtSetCurrentContext(previous));
        if (error == 0) notifications_.clear();
        return error;
    }

    void abandon() noexcept override {
        retiring = true;
        notifications_.clear();
    }

private:
    static void map(rtDevResInfo &mapping, uint64_t &address, bool &mapped) {
        uint32_t bytes = 0;
        rtDevResAddrInfo out{&address, &bytes};
        check(rtGetDevResAddress(&mapping, &out), "map");
        mapped = true;
        if (address == 0 || address % alignof(uint32_t) != 0 || bytes < sizeof(uint32_t))
            throw std::runtime_error("CntNotify invalid 32-bit register mapping");
    }
    // Explicit close owns SDK calls. A reset or a rejected close cannot cause
    // a member destructor to operate on a dead or still-borrowed context.
    std::vector<Notification> notifications_;
    const bool counter_;
    const bool required_;
};
}  // namespace

std::shared_ptr<SchedulerNotificationStorage>
DeviceRunner::make_scheduler_notification_storage(void *context, SchedulerNotificationPurpose purpose) {
    return std::make_shared<NotificationPool>(context, purpose);
}
