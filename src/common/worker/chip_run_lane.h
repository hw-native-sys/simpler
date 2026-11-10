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

#include <chrono>
#include <cstdint>
#include <exception>
#include <memory>
#include <functional>
#include <utility>

#include "../task_interface/call_config.h"
#include "../task_interface/task_args.h"
#include "run_resource_arena.h"

class ChipWorker;
class ChipRunResources;
struct ChipRunLaneState;
struct ChipRunState;

enum class ChipRunPreparationDisposition : int32_t {
    VALIDATED_ONLY = 1,
    NATIVE_PREPARED = 2,
};

class ChipRun {
public:
    using Clock = std::chrono::steady_clock;
    using Deadline = Clock::time_point;

    ChipRun() = default;

    bool done();
    bool retire_framework(bool wait_for_own_completion, Deadline deadline = Deadline::max());
    bool framework_transferred() const;
    bool wait_until(Deadline deadline);
    void activate();
    void abandon();

    bool launched() const;
    // Physical END awaiting result observation, distinct from cached TERMINAL.
    bool execution_ended() const;

    /**
     * This run's diagnostic admission deadline, or a default-constructed
     * time point when it has none.
     *
     * The deadline is fixed at the first capacity refusal of a run at the
     * front and never renewed; reading it is the only way a case can show
     * that a later round left it where the first one put it.
     */
    Deadline credit_deadline_for_test() const;

    /** Bring this run's admission deadline forward, so a case need not wait. */
    void expire_credit_deadline_for_test();
    bool lane_poisoned() const;
    bool owner_recovery_required() const;
    ChipRunPreparationDisposition preparation_disposition() const;
    std::unique_ptr<ChipRunResources> retain_resources();

private:
    friend class ChipRunLane;
    friend class ChipWorker;
    friend class ChipRunResources;
    friend class NativeSubmissionState;
    void retain_resources_into(std::unique_ptr<ChipRunResources> &out);
    bool carrier_retired_for_submission() const;
    void external_resource_acquire_failed() const;
    void external_resources_released() const;
    void require_carrier_retired() const;
    ChipRun(std::shared_ptr<ChipRunLaneState> lane, std::shared_ptr<ChipRunState> run);

    std::shared_ptr<ChipRunLaneState> lane_;
    std::shared_ptr<ChipRunState> run_;
};

class ChipRunLane {
public:
    using Clock = ChipRun::Clock;
    using Deadline = ChipRun::Deadline;

    explicit ChipRunLane(ChipWorker &worker);
    ~ChipRunLane();

    ChipRunLane(const ChipRunLane &) = delete;
    ChipRunLane &operator=(const ChipRunLane &) = delete;

    ChipRun submit(
        int32_t callable_id, const ChipStorageTaskArgs &args, const CallConfig &config, const PipelineSlotLease &lease,
        uint64_t run_id, uint64_t dispatch_id, volatile int32_t *accepted_state = nullptr, int32_t accepted_value = 0,
        bool activated = true, NativeRunProvenance provenance = {}
    );
    // A non-null caller stream requires actual enqueue before a successful
    // return. Stream work appended after that return is ordered after both kernels.
    ChipRun submit(
        int32_t callable_id, const ChipStorageTaskArgs &args, const CallConfig &config,
        volatile int32_t *accepted_state = nullptr, int32_t accepted_value = 0, void *caller_stream = nullptr,
        NativeRunPreparePurpose purpose = SIMPLER_PREPARE_EAGER, ChipRun *prepared_out = nullptr,
        bool framework_owned = false
    );

    void drain();

    /**
     * Stop admitting runs, and finish every run that has not yet launched
     * without launching it.
     *
     * For a caller that has established nothing further may reach the device —
     * unknown device-resource ownership, say. The ordinary `close` cannot serve
     * that: it drains, and a drain launches before it waits, so an activated
     * prepared successor would be put on the device on the way out. Runs
     * already launched keep their own outcomes and their own drains, which is
     * what retires the device work, events and resources they own.
     *
     * Idempotent, and non-throwing: a failure finishing an unlaunched run
     * poisons the lane so `close` reports it, rather than unwinding a caller
     * that is already on a failure path.
     */
    void stop_admission() noexcept;

    void close(bool report_run_error = true);
    bool poisoned() const;

private:
    friend class ChipWorker;
    int32_t with_callable_addition(const std::function<int32_t()> &operation);
    std::pair<int, bool> observe_framework_submissions();
    void detach_after_failed_teardown(std::exception_ptr error) noexcept;
    std::shared_ptr<ChipRunLaneState> state_;
};
