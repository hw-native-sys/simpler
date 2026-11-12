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

#include "ccu_run_lifecycle_hcomm.h"

#include <cstddef>
#include <utility>

#include "hcomm/ccu/ccu_launch.h"

namespace simpler::ccu {
namespace {

constexpr int kInvalidArgument = -1;

}  // namespace

HcommRunLifecycleSdk::HcommRunLifecycleSdk(
    std::vector<HcommMissionStream> streams, ReadyEventPoller ready_event_poller
) :
    streams_(std::move(streams)),
    ready_event_poller_(std::move(ready_event_poller)) {}

const HcommMissionStream *HcommRunLifecycleSdk::find_stream(uint64_t handle) const {
    for (const auto &stream : streams_) {
        if (stream.handle == handle) return &stream;
    }
    return nullptr;
}

int HcommRunLifecycleSdk::launch(
    uint64_t program_handle, const uint64_t *arguments, uint32_t argument_count, uint64_t stream_handle
) {
    const HcommMissionStream *mission = find_stream(stream_handle);
    if (mission == nullptr || mission->thread == 0 || program_handle == 0 || arguments == nullptr) {
        return kInvalidArgument;
    }
    if (!HcommCcuKernelLaunch) return CCU_E_UNAVAIL;
    return HcommCcuKernelLaunch(mission->thread, program_handle, arguments, argument_count);
}

int HcommRunLifecycleSdk::poll_ready_event(uint64_t address, uint64_t mask, bool *ready) {
    if (address == 0 || ready == nullptr || !ready_event_poller_) return kInvalidArgument;
    return ready_event_poller_(address, mask, ready);
}

int HcommRunLifecycleSdk::query_stream(uint64_t stream_handle, StreamState *state) {
    const HcommMissionStream *mission = find_stream(stream_handle);
    if (mission == nullptr || mission->stream == nullptr || state == nullptr) return kInvalidArgument;
    aclrtStreamStatus status{};
    const aclError rc = aclrtStreamQuery(mission->stream, &status);
    if (rc != ACL_SUCCESS) {
        *state = StreamState::UNKNOWN;
        return rc;
    }
    *state = status == ACL_STREAM_STATUS_COMPLETE ? StreamState::COMPLETE : StreamState::RUNNING;
    return ACL_SUCCESS;
}

int HcommRunLifecycleSdk::abort_stream(uint64_t stream_handle) {
    const HcommMissionStream *mission = find_stream(stream_handle);
    if (mission == nullptr || mission->stream == nullptr) return kInvalidArgument;
    return aclrtStreamAbort(mission->stream);
}

}  // namespace simpler::ccu
