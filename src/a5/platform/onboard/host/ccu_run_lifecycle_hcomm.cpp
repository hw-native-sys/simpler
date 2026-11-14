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
#include <stdexcept>

#include "common/unified_log.h"

#include "hcomm/ccu/ccu_launch.h"

namespace simpler::ccu {
namespace {

constexpr int kInvalidArgument = -1;

}  // namespace

HcommMissionStream HcommMissionStreamPool::acquire() {
    if (closed_) throw std::runtime_error("CCU mission stream pool is closed");
    for (size_t i = 0; i < entries_.size(); ++i) {
        auto &entry = entries_[i];
        if (entry.leased) continue;
        aclrtStreamStatus status{};
        if (aclrtStreamQuery(entry.stream, &status) != ACL_SUCCESS || status != ACL_STREAM_STATUS_COMPLETE)
            throw std::runtime_error("idle CCU pool stream is not reusable");
        entry.leased = true;
        ++entry.uses;
        LOG_DEBUG("[ccu] pool acquire slot=%zu uses=%llu", i + 1, static_cast<unsigned long long>(entry.uses));
        return {i + 1, 0, entry.stream};
    }
    entries_.push_back({});
    auto &entry = entries_.back();
    const auto rc = aclrtCreateStream(&entry.stream);
    if (rc != ACL_SUCCESS) {
        if (entry.stream) entry.leased = true;
        else entries_.pop_back();
        throw std::runtime_error("CCU pool stream creation failed: " + std::to_string(rc));
    }
    entry.leased = true;
    entry.uses = 1;
    LOG_INFO("[ccu] pool create slot=%zu stream=%p retained=%zu", entries_.size(), entry.stream, entries_.size());
    return {entries_.size(), 0, entry.stream};
}

bool HcommMissionStreamPool::release(uint64_t handle) {
    if (closed_ || handle == 0 || handle > entries_.size()) return false;
    auto &entry = entries_[handle - 1];
    if (!entry.leased) return false;
    aclrtStreamStatus status{};
    if (aclrtStreamQuery(entry.stream, &status) != ACL_SUCCESS || status != ACL_STREAM_STATUS_COMPLETE) return false;
    entry.leased = false;
    LOG_DEBUG(
        "[ccu] pool return slot=%llu uses=%llu", static_cast<unsigned long long>(handle),
        static_cast<unsigned long long>(entry.uses)
    );
    return true;
}

bool HcommMissionStreamPool::close() {
    if (closed_) return true;
    for (const auto &entry : entries_) {
        if (entry.leased) return false;
        if (!entry.stream) continue;
        aclrtStreamStatus status{};
        if (aclrtStreamQuery(entry.stream, &status) != ACL_SUCCESS || status != ACL_STREAM_STATUS_COMPLETE)
            return false;
    }
    for (auto &entry : entries_) {
        if (!entry.stream) continue;
        if (aclrtDestroyStream(entry.stream) != ACL_SUCCESS) return false;
        LOG_INFO("[ccu] pool destroy stream=%p uses=%llu", entry.stream, static_cast<unsigned long long>(entry.uses));
        entry.stream = nullptr;
    }
    closed_ = true;
    return true;
}

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

int HcommRunLifecycleSdk::stop_completed_stream(uint64_t stream_handle) {
    const auto *mission = find_stream(stream_handle);
    if (!mission || !mission->stream) return kInvalidArgument;
    const auto rc = aclrtStreamAbort(mission->stream);
    if (rc != ACL_SUCCESS) return rc;
    aclrtStreamStatus status{};
    const auto query = aclrtStreamQuery(mission->stream, &status);
    if (query != ACL_SUCCESS) return query;
    if (status != ACL_STREAM_STATUS_COMPLETE) return kInvalidArgument;
    LOG_INFO(
        "[ccu] completed mission abort confirmed stream=%p slot=%llu", mission->stream,
        static_cast<unsigned long long>(stream_handle)
    );
    return ACL_SUCCESS;
}

}  // namespace simpler::ccu
