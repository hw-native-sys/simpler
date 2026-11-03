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

#include <atomic>
#include <cstdint>
#include <mutex>
#include <pthread.h>

namespace simpler {
// One module-owned key borrowing an execution cell. Failed creation is retryable;
// callers must quiesce all users before unloading the module (and deleting the key).
class __attribute__((visibility("hidden"))) AffinityTlsKey {
public:
    ~AffinityTlsKey() {
        if (ready_.load(std::memory_order_relaxed)) pthread_key_delete(key_);
    }
    int try_set(int32_t *cell) {
        if (!ready_.load(std::memory_order_acquire)) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!ready_.load(std::memory_order_relaxed)) {
                const int error = pthread_key_create(&key_, nullptr);
                if (error != 0) return error;
                ready_.store(true, std::memory_order_release);
            }
        }
        return pthread_setspecific(key_, cell);
    }
    int32_t *get() const {
        return ready_.load(std::memory_order_acquire) ? static_cast<int32_t *>(pthread_getspecific(key_)) : nullptr;
    }

private:
    pthread_key_t key_{};
    std::atomic<bool> ready_{false};
    std::mutex mutex_;
};
// Backend classification remains separate. Every advertised participant must
// copy this result before the backend's last-reader cleanup resets it. A new
// invocation must wait until all previous executors stop using borrowed slots.
class AffinityBindingCohort {
public:
    void record(int error) {
        if (error == 0) return;
        int expected = 0;
        error_.compare_exchange_strong(expected, error, std::memory_order_acq_rel);
    }
    int arrive_and_read(int participants) {
        bound_.fetch_add(1, std::memory_order_release);
        while (bound_.load(std::memory_order_acquire) < participants) {}
        return error_.load(std::memory_order_acquire);
    }
    void reset_after_readers() {
        error_.store(0, std::memory_order_release);
        bound_.store(0, std::memory_order_release);
    }

private:
    std::atomic<int> error_{0};
    std::atomic<int> bound_{0};
};
}  // namespace simpler
