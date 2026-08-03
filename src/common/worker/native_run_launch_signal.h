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

#ifndef SRC_COMMON_WORKER_NATIVE_RUN_LAUNCH_SIGNAL_H_
#define SRC_COMMON_WORKER_NATIVE_RUN_LAUNCH_SIGNAL_H_

#include <condition_variable>
#include <mutex>

/** Sticky one-shot wakeup for the host thread waiting on launch readiness. */
class NativeRunLaunchSignal {
public:
    NativeRunLaunchSignal() = default;
    NativeRunLaunchSignal(const NativeRunLaunchSignal &) = delete;
    NativeRunLaunchSignal &operator=(const NativeRunLaunchSignal &) = delete;

    void wait() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this]() {
            return notified_;
        });
    }

    void notify() {
        {
            std::scoped_lock<std::mutex> lock(mutex_);
            notified_ = true;
        }
        cv_.notify_one();
    }

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    bool notified_{false};
};

#endif  // SRC_COMMON_WORKER_NATIVE_RUN_LAUNCH_SIGNAL_H_
