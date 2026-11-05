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
#include <mutex>
#include <utility>

#include "chip_worker.h"

namespace simpler::worker {

// Internal to the resident ChipWorker implementation; runtime DSOs and public
// Worker APIs hold no owner table or fork hooks.
class DeviceExecutionOwners {
public:
    bool claim(const ChipWorker::ExecutionEnvironment &environment) {
        std::scoped_lock lock(mutex_);
        return owners_.emplace(std::make_pair(environment.process_id, environment.device_id), environment.incarnation)
            .second;
    }

    void release(const ChipWorker::ExecutionEnvironment &environment) {
        std::scoped_lock lock(mutex_);
        const auto found = owners_.find(std::make_pair(environment.process_id, environment.device_id));
        if (found != owners_.end() && found->second == environment.incarnation) owners_.erase(found);
    }

    void before_fork() { mutex_.lock(); }
    void after_fork() { mutex_.unlock(); }

private:
    std::mutex mutex_;
    std::map<std::pair<uint64_t, int>, uint64_t> owners_;
};

DeviceExecutionOwners &resident_device_execution_owners();

}  // namespace simpler::worker
