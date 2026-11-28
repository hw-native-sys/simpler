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

#include "aicore_lifecycle.h"
#include "args_dump_service.h"
#include "aicpu/device_run_result_aicpu.h"
#include "common/run_terminal_accumulator.h"
#include "utils/thread_completion_gate.h"

class Runtime;

struct AicpuExecutor {
    // ===== Thread management state =====
    std::atomic<int32_t> thread_idx_{0};
    std::atomic<bool> init_done_{false};
    std::atomic<bool> init_failed_{false};

    // Parallel-handshake coordination (see AicpuExecutor::init). hs_setup_done_
    // is published by the leader once the shared pre-handshake setup is visible;
    // hs_arrived_ is the barrier counting threads that finished their core slice.
    // hs_thread_seq_ hands out a distinct [0, nthreads) index when the platform
    // exposes no affinity idx (sim, where platform_aicpu_affinity_thread_idx()
    // is -1 during init) so the threads don't all collapse to leader 0.
    std::atomic<bool> hs_setup_done_{false};
    std::atomic<bool> hs_config_done_{false};
    std::atomic<int32_t> hs_arrived_{0};
    std::atomic<int32_t> hs_release_arrived_{0};
    std::atomic<int32_t> hs_thread_seq_{0};
    std::atomic<int32_t> init_failure_arrived_{0};

    int32_t aicpu_thread_num_{0};

    simpler::ThreadCompletionGate completion_gate_;

    // This run's terminal record. `terminal_` collects each participant's own
    // failure as it retires; `normal_path_claims_` counts the participants that
    // positively reached the end of their audited path, so a run is a success
    // only when every one of them did rather than when none of them complained;
    // `terminal_publisher_` carries the finalizer's decision across teardown to
    // the sole cleanup owner that publishes it. This runtime attaches no
    // diagnostic payload: its scene lives in the AICore run control, which the
    // supervisor already latches into the shared header.
    RunTerminalAccumulator terminal_;
    std::atomic<int32_t> normal_path_claims_{0};
    RunTerminalPublisher terminal_publisher_;

    std::atomic<bool> shutdown_ready_{false};
    std::atomic<int32_t> shutdown_signaled_{0};
    std::atomic<int32_t> run_status_{0};

    // AICPU owns only the AICore launch/teardown lifecycle. AICore owns graph
    // calendar dependency checks and task execution.
    AicoreLifecycle aicore_lifecycle_;
    ResidentArgsDumpService *args_dump_services_{nullptr};
    int32_t args_dump_service_count_{0};
    bool args_dump_services_allocated_this_init_{false};
    bool args_dump_services_stopped() const {
        for (int32_t thread = 0; args_dump_services_ != nullptr && thread < args_dump_service_count_; ++thread)
            if (args_dump_services_[thread].has_open_record() || dump_arg_has_open_record(thread)) return false;
        return true;
    }

    // ===== Methods =====
    int32_t init(Runtime *runtime);
    int32_t run(Runtime *runtime);
    // Copy one pending argument record for this thread, if the run dumps args.
    bool service_args_dump(int32_t thread_idx);
    void deinit(Runtime *runtime);
    void reset_entry_state();
    int32_t finish_failed_init(Runtime *runtime, int32_t shutdown_rc);
    // Decide this run's terminal record while the state it reads is still
    // valid. Runs inside the completion gate's finalizer, on the last
    // participant out.
    void snapshot_run_terminal(Runtime *runtime, bool cores_retired);
    // Commit the decided record. Runs on the thread that claims cleanup, after
    // teardown and before it returns from the kernel.
    void handoff_run_terminal();

    // Configure the discovered cores and publish READY, between the handshake
    // and register release. Leader only.
    int32_t post_handshake_init(Runtime *runtime);
    // Publish READY to the workers that this thread discovered.
    void publish_partition(Runtime *runtime, int32_t tidx, int32_t nthreads);
    // Whether this runtime accepts the published bootstrap. A rejection still
    // discovers the launched cores and retires them without publishing READY.
    bool schedules_on_aicore(const Runtime *runtime) const;
    // Watch the run to completion. Runs on the last thread, and its return
    // value is this run's status.
    int32_t supervise(Runtime *runtime, int32_t thread_idx);
    // Wait for the supervisor to declare shutdown. Runs on every other thread.
    void await_shutdown(Runtime *runtime, int32_t thread_idx);
    // Settle this run's error into the shared header and return the status it
    // then carries. Runs inside snapshot_run_terminal, on the last participant
    // out, which is the first point at which every core has stopped writing:
    // a scheduler whose lanes can fail after its supervisor returned has
    // nowhere earlier to read them.
    int32_t settle_run_error(Runtime *runtime);
};
