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

#include <algorithm>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

// Both the carrier and device teardown can stop a scan. Keeping the thread
// control separate lets teardown join it even during the carrier's destruction.
struct UrmaScanThread {
    std::mutex mutex;
    std::condition_variable changed;
    std::thread worker;
    bool stopping = false;

    void stop() {
        std::lock_guard lock(mutex);
        stopping = true;
        changed.notify_all();
    }

    void stop_and_join() {
        stop();
        // stop() serialized with startup under mutex. No other thread can start now.
        std::lock_guard lock(join_mutex_);
        if (worker.joinable()) worker.join();
    }

private:
    std::mutex join_mutex_;
};

// One gate belongs to one usable device generation. Quarantine and SDK enqueue
// take the same lock. A Run holds only a weak permit to its original gate.
class UrmaLaunchGate {
public:
    bool open() const {
        std::lock_guard lock(mutex_);
        return open_;
    }
    template <typename Submit>
    bool submit(Submit enqueue) {
        std::lock_guard lock(mutex_);
        if (!open_) return false;
        enqueue();
        return true;
    }
    void track(const std::shared_ptr<UrmaScanThread> &scan) {
        std::lock_guard lock(mutex_);
        scans_.erase(
            std::remove_if(
                scans_.begin(), scans_.end(),
                [](const auto &s) {
                    return s.expired();
                }
            ),
            scans_.end()
        );
        scans_.push_back(scan);
    }

    // The runner stops registration before calling this. Workers only need
    // the enqueue lock, which is released before waking or joining any thread.
    void stop_scans() {
        std::vector<std::weak_ptr<UrmaScanThread>> scans;
        {
            std::lock_guard lock(mutex_);
            open_ = false;
            scans.swap(scans_);
        }
        try {
            for (auto &scan : scans)
                if (auto active = scan.lock()) active->stop();
            for (auto &scan : scans)
                if (auto active = scan.lock()) active->stop_and_join();
        } catch (...) {
            // A failed join cannot authorize a reset. The next finalize must wait again.
            std::lock_guard lock(mutex_);
            scans.swap(scans_);
            throw;
        }
    }

    void revoke() {
        std::lock_guard lock(mutex_);
        open_ = false;
    }

private:
    std::vector<std::weak_ptr<UrmaScanThread>> scans_;
    mutable std::mutex mutex_;
    bool open_ = true;
};

using UrmaLaunchPermission = std::weak_ptr<UrmaLaunchGate>;
