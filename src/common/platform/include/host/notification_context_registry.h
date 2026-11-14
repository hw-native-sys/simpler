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

#include <map>
#include <memory>
#include <mutex>
#include <utility>

#include "common/notification_context.h"

// Only the allocation owner publishes and erases immutable Host snapshots.
// Readers neither dereference device addresses nor extend allocation lifetime.
class NotificationContextRegistry {
public:
    using Key = std::pair<int32_t, uint64_t>;
    bool publish(int32_t device, HostNotificationContext snapshot) {
        if (device < 0 || snapshot.context_address == 0 || snapshot.rank_count == 0 ||
            snapshot.rank_count > snapshot.rank_ids.size() || snapshot.local_rank >= snapshot.rank_count ||
            snapshot.window_bytes == 0 || snapshot.windows_in[snapshot.local_rank] == 0)
            return false;
        const Key key{device, snapshot.context_address};
        auto data = std::make_shared<const HostNotificationContext>(std::move(snapshot));
        std::lock_guard<std::mutex> lock(mutex_);
        return records_.emplace(key, std::move(data)).second;
    }

    std::shared_ptr<const HostNotificationContext> acquire(int32_t device, uint64_t address) {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = records_.find({device, address});
        return it == records_.end() ? nullptr : it->second;
    }

    void erase(const Key &key) {
        std::lock_guard<std::mutex> lock(mutex_);
        records_.erase(key);
    }

private:
    std::mutex mutex_;
    std::map<Key, std::shared_ptr<const HostNotificationContext>> records_;
};
