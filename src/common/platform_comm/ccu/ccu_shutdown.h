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

namespace simpler::ccu {

// The caller holds the provider lock and retains every service owner. A failed
// close preserves that owner and all later services for an explicit retry.
template <typename Instances, typename Close>
bool close_idle_services(const Instances &instances, Close close) {
    for (const auto &instance : instances) {
        if (!instance->closed && instance->active_runs.load() != 0) return false;
    }
    for (const auto &instance : instances) {
        if (instance->closed) continue;
        if (!close(*instance)) return false;
        instance->closed = true;
    }
    return true;
}

// Stopping the independent CCU communicator does not retire its parent comm's
// HCCL handles or mappings. Only complete comm release retires those owners;
// a device reset must not invalidate handles that cleanup can still visit.
template <typename Instances, typename Close>
bool close_services_before_device_teardown(const Instances &instances, bool parent_comms_released, Close close) {
    if (!close_idle_services(instances, close)) return false;
    return instances.empty() && parent_comms_released;
}

}  // namespace simpler::ccu
