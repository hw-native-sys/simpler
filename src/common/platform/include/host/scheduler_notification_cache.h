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

#include <algorithm>
#include <map>
#include <mutex>
#include <utility>

class SchedulerNotificationStorage {
public:
    explicit SchedulerNotificationStorage(void *context) :
        context(context) {}
    virtual ~SchedulerNotificationStorage() = default;
    virtual size_t capacity() const = 0;
    virtual SchedulerDoneNotifications prepare(size_t count) = 0;
    virtual int release() = 0;
    virtual void abandon() noexcept = 0;
    void *const context;
    bool retiring{false};
};

// A publication's strong reference prevents reuse, including captured runs.
// The runner retains every pool through explicit close or post-reset abandon.
class SchedulerNotificationCache {
public:
    template <class Create>
    int acquire(
        uint32_t slot, SchedulerNotificationPurpose purpose, void *context, size_t count, Create create,
        SchedulerDoneNotifications *out
    ) {
        if (out == nullptr || context == nullptr) return -1;
        *out = {};
        if (count == 0) return 0;
        std::lock_guard<std::mutex> lock(mutex_);
        auto &bank = pools_[{slot, purpose}];
        std::shared_ptr<SchedulerNotificationStorage> selected;
        for (const auto &pool : bank) {
            if (pool->context != context) return -1;
            if (pool.use_count() != 1 || pool->retiring) continue;
            if (!selected || (pool->capacity() >= count &&
                              (selected->capacity() < count || pool->capacity() < selected->capacity())))
                selected = pool;
        }
        for (auto it = bank.begin(); it != bank.end();) {
            if (*it == selected || it->use_count() != 1) {
                ++it;
                continue;
            }
            (*it)->retiring = true;
            const int rc = (*it)->release();
            if (rc != 0) return rc;
            it = bank.erase(it);
        }
        if (!selected) {
            selected = create(context, purpose);
            if (!selected || selected->context != context) return -1;
            bank.push_back(selected);
        }
        try {
            auto result = selected->prepare(count);
            result.owner = selected;
            *out = std::move(result);
            return 0;
        } catch (...) {
            selected->retiring = true;
            throw;
        }
    }

    int release() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto &entry : pools_)
            for (const auto &pool : entry.second)
                if (pool.use_count() != 1) return -1;
        int first_error = 0;
        for (auto &entry : pools_) {
            auto &bank = entry.second;
            for (auto it = bank.begin(); it != bank.end();) {
                (*it)->retiring = true;
                const int rc = (*it)->release();
                if (rc == 0) it = bank.erase(it);
                else {
                    if (first_error == 0) first_error = rc;
                    ++it;
                }
            }
        }
        if (first_error == 0) {
            pools_.clear();
        }
        return first_error;
    }

    // The caller proves the old device generation has been reset. Destruction
    // of any remaining host lease must issue no per-resource SDK operations.
    void abandon() noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto &entry : pools_)
            for (auto &pool : entry.second)
                pool->abandon();
        pools_.clear();
    }

private:
    std::mutex mutex_;
    std::map<
        std::pair<uint32_t, SchedulerNotificationPurpose>, std::vector<std::shared_ptr<SchedulerNotificationStorage>>>
        pools_;
};
