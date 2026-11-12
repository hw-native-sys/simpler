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

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "ccu_domain_registration.h"

namespace simpler::ccu {

enum class StreamState : uint32_t { RUNNING, COMPLETE, ERROR, UNKNOWN };

struct MissionRunResources {
    uint32_t mission_id{};
    uint64_t program_handle{};
    uint64_t stream_handle{};
    SimplerCcuLaunchArguments launch_arguments{};
    SimplerCcuEventDescriptor ready_event{};
};

struct RunFailure {
    std::string operation;
    int64_t native_error{};
};

class RunLifecycleSdk {
public:
    virtual ~RunLifecycleSdk() = default;

    virtual int
    launch(uint64_t program_handle, const uint64_t *arguments, uint32_t argument_count, uint64_t stream_handle) = 0;
    virtual int poll_ready_event(uint64_t address, uint64_t mask, bool *ready) = 0;
    virtual int query_stream(uint64_t stream_handle, StreamState *state) = 0;
    virtual int abort_stream(uint64_t stream_handle) = 0;
};

class DomainRun;

class DomainRunLifecycle {
public:
    std::unique_ptr<DomainRun> begin_run(
        uint64_t generation, const std::vector<RegisteredKernel> &registered,
        const std::vector<MissionRunResources> &missions, RunLifecycleSdk &sdk
    );

    bool admission_open() const noexcept { return admission_open_; }
    bool resources_retained() const noexcept { return resources_retained_; }
    bool has_active_run() const noexcept { return active_; }

private:
    friend class DomainRun;

    bool active_{false};
    bool admission_open_{true};
    bool resources_retained_{false};
};

class DomainRun {
public:
    ~DomainRun();

    DomainRun(const DomainRun &) = delete;
    DomainRun &operator=(const DomainRun &) = delete;

    void launch_all();
    bool poll_ready();
    bool poll_terminal();
    bool stop();
    void abort();

    bool launched() const noexcept { return launched_; }
    bool ready() const noexcept { return ready_; }
    bool terminal() const noexcept { return terminal_; }
    bool confirmed_stopped() const noexcept { return confirmed_stopped_; }
    bool failed() const noexcept { return !failures_.empty(); }
    const RunFailure *primary_failure() const noexcept { return failures_.empty() ? nullptr : &failures_.front(); }
    const std::vector<RunFailure> &failures() const noexcept { return failures_; }

private:
    friend class DomainRunLifecycle;

    DomainRun(
        DomainRunLifecycle &owner, uint64_t generation, std::vector<MissionRunResources> missions, RunLifecycleSdk &sdk
    );

    void record_failure(std::string operation, int64_t native_error);
    void finish_confirmed();
    void retain_unconfirmed();

    DomainRunLifecycle &owner_;
    uint64_t generation_{};
    std::vector<MissionRunResources> missions_;
    RunLifecycleSdk &sdk_;
    std::vector<bool> ready_events_consumed_;
    std::vector<RunFailure> failures_;
    bool launched_{false};
    bool ready_{false};
    bool terminal_{false};
    bool confirmed_stopped_{false};
    bool abort_attempted_{false};
    size_t launched_missions_{0};
    bool released_{false};
};

bool all_runs_ready(const std::vector<DomainRun *> &runs);

}  // namespace simpler::ccu
