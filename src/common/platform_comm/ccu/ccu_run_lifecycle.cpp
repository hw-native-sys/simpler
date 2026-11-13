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

#include "ccu_run_lifecycle.h"

#include <map>
#include <stdexcept>
#include <utility>

namespace simpler::ccu {
namespace {

constexpr int64_t kProtocolError = -1;

[[noreturn]] void fail_protocol(const std::string &message) { throw std::invalid_argument(message); }

}  // namespace

std::unique_ptr<DomainRun> DomainRunLifecycle::begin_run(
    uint64_t generation, const std::vector<RegisteredKernel> &registered,
    const std::vector<MissionRunResources> &missions, RunLifecycleSdk &sdk
) {
    if (!admission_open_) throw std::runtime_error("CCU domain admission is closed");
    if (active_) throw std::runtime_error("CCU domain already has an active run");
    if (generation == 0 || missions.empty()) fail_protocol("CCU run generation and missions must be nonzero");

    std::map<uint64_t, bool> programs;
    for (const auto &kernel : registered) {
        if (kernel.program_handle == 0 || !programs.emplace(kernel.program_handle, true).second)
            fail_protocol("invalid or duplicate registered CCU program");
    }
    std::map<uint32_t, bool> mission_ids;
    for (const auto &mission : missions) {
        if (mission.program_handle == 0 || mission.stream_handle == 0 || programs.count(mission.program_handle) == 0 ||
            mission.launch_arguments.parameter_addr == 0 || mission.ready_event.address == 0 ||
            mission.ready_event.mask != SIMPLER_CCU_EVENT_MASK || !mission_ids.emplace(mission.mission_id, true).second)
            fail_protocol("invalid or duplicate CCU mission run resources");
    }

    active_ = true;
    return std::unique_ptr<DomainRun>(new DomainRun(*this, generation, missions, sdk));
}

DomainRun::DomainRun(
    DomainRunLifecycle &owner, uint64_t generation, std::vector<MissionRunResources> missions, RunLifecycleSdk &sdk
) :
    owner_(owner),
    generation_(generation),
    missions_(std::move(missions)),
    sdk_(sdk),
    ready_events_consumed_(missions_.size(), false) {}

DomainRun::~DomainRun() {
    if (released_) return;
    if (launched_missions_ == 0) {
        owner_.active_ = false;
        released_ = true;
        return;
    }
    retain_unconfirmed();
}

void DomainRun::record_failure(std::string operation, int64_t native_error) {
    failures_.push_back({std::move(operation), native_error});
}

void DomainRun::finish_confirmed() {
    confirmed_stopped_ = true;
    terminal_ = true;
    released_ = true;
    owner_.active_ = false;
}

void DomainRun::retain_unconfirmed() {
    owner_.admission_open_ = false;
    owner_.resources_retained_ = true;
    released_ = true;
}

void DomainRun::launch_all() {
    if (launched_) throw std::runtime_error("CCU run already launched");
    for (const auto &mission : missions_) {
        const auto *words = reinterpret_cast<const uint64_t *>(&mission.launch_arguments);
        const int rc =
            sdk_.launch(mission.program_handle, words, SIMPLER_CCU_TASK_ARGUMENT_COUNT, mission.stream_handle);
        if (rc != 0) {
            record_failure("HcommCcuKernelLaunch", rc);
            abort();
            throw std::runtime_error("CCU mission launch failed");
        }
        ++launched_missions_;
    }
    launched_ = true;
}

bool DomainRun::poll_ready() {
    if (!launched_) throw std::runtime_error("CCU run has not launched");
    if (ready_) return true;
    if (failed()) return false;
    bool all_ready = true;
    for (size_t i = 0; i < missions_.size(); ++i) {
        if (ready_events_consumed_[i]) continue;
        bool observed = false;
        const auto &event = missions_[i].ready_event;
        const int rc = sdk_.poll_ready_event(event.address, event.mask, &observed);
        if (rc != 0) {
            record_failure("observe/clear CCU ready Event", rc);
            abort();
            return false;
        }
        ready_events_consumed_[i] = observed;
        all_ready = all_ready && observed;
    }
    ready_ = all_ready;
    return ready_;
}

bool DomainRun::poll_terminal() {
    if (!launched_) throw std::runtime_error("CCU run has not launched");
    if (confirmed_stopped_) return true;
    if (failed()) return false;
    bool complete = true;
    for (const auto &mission : missions_) {
        StreamState state = StreamState::UNKNOWN;
        const int rc = sdk_.query_stream(mission.stream_handle, &state);
        if (rc != 0 || state == StreamState::ERROR) {
            record_failure("query CCU stream", rc != 0 ? rc : kProtocolError);
            abort();
            return false;
        }
        complete = complete && state == StreamState::COMPLETE;
    }
    if (complete) finish_confirmed();
    return complete;
}

bool DomainRun::stop() {
    // Finite user control flow returns without a runtime stop gate or final wake.
    return poll_terminal();
}

void DomainRun::abort() {
    if (abort_attempted_) return;
    abort_attempted_ = true;
    if (launched_missions_ != 0) {
        for (size_t i = 0; i < launched_missions_; ++i) {
            const auto &mission = missions_[i];
            const int rc = sdk_.abort_stream(mission.stream_handle);
            if (rc != 0) record_failure("abort CCU stream", rc);
        }
    }
    retain_unconfirmed();
}

bool all_runs_ready(const std::vector<DomainRun *> &runs) {
    if (runs.empty()) return false;
    bool ready = true;
    for (DomainRun *run : runs) {
        if (run == nullptr || !run->poll_ready()) ready = false;
        if (run != nullptr && run->failed()) return false;
    }
    return ready;
}

}  // namespace simpler::ccu
