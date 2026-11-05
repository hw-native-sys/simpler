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
/**
 * The AICPU side of one run: thread rendezvous, the AICore launch handshake,
 * the supervision wait, and teardown.
 *
 * `init` and `run` carry the scaffolding every scheduler needs — thread index
 * assignment, the handshake barriers, the completion gate, the terminal record
 * — and reach the parts a scheduler decides for itself through the six steps
 * declared at the end. Those six are defined weak, so a runtime that schedules
 * AICore differently supplies its own and the linker drops this one's; the
 * scaffolding around them is shared rather than reimplemented.
 *
 * Weak definitions rather than virtuals on purpose: a vtable would reference
 * every step, and test_hbg_resident_late_aicore_error compiles this translation
 * unit with -ffunction-sections and links with --gc-sections precisely to drop
 * the steps it does not drive.
 */

#pragma once

#include <atomic>
#include <cstdint>

#include "aicore_lifecycle.h"
#include "aicpu/device_run_result_aicpu.h"
#include "common/run_terminal_accumulator.h"
#include "scheduler_ready_watchdog.h"
#include "utils/thread_completion_gate.h"

// Hidden, not weak: the weak marker belongs on this runtime's definitions, so
// that a replacement's strong one wins at static link time. Visibility keeps
// that resolution inside one shared object — the AICPU images of two runtimes
// can share a process, and a default-visibility weak symbol is preemptible, so
// a step would otherwise bind to whichever image defined it first and then run
// a graph the other runtime bound.
#define SIMPLER_SCHEDULER_STEP __attribute__((visibility("hidden")))

struct Runtime;

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
    std::atomic<bool> hs_bootstrap_done_{false};
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

    AicpuReadyWatchdog ready_watchdog_;
    std::atomic<bool> shutdown_ready_{false};
    std::atomic<int32_t> shutdown_signaled_{0};
    std::atomic<int32_t> run_status_{0};

    // AICPU owns only the AICore launch/teardown lifecycle. AICore owns graph
    // dependency resolution, Ready scheduling, and task execution.
    AicoreLifecycle aicore_lifecycle_;

    // ===== Methods =====
    int32_t init(Runtime *runtime);
    int32_t run(Runtime *runtime);
    void deinit(Runtime *runtime);
    int32_t finish_failed_init(Runtime *runtime);
    // Decide this run's terminal record while the state it reads is still
    // valid. Runs inside the completion gate's finalizer, on the last
    // participant out.
    void snapshot_run_terminal(Runtime *runtime, bool cores_retired);
    // Commit the decided record. Runs on the thread that claims cleanup, after
    // teardown and before it returns from the kernel.
    void publish_run_terminal();

    // ===== The steps a scheduler decides for itself =====
    //
    // Each default is what the AICPU-dispatched and resident schedulers do, and
    // each is weak: a replacement replaces that one step, and the code around it
    // in init and run is the same either way. All six run once per thread per
    // run, never per task.

    // Configure the discovered cores and publish READY, between the handshake
    // and the bootstrap gate. Leader only.
    SIMPLER_SCHEDULER_STEP int32_t post_handshake_init(Runtime *runtime);
    // Hand this thread's slice of the worker contexts to the cores.
    SIMPLER_SCHEDULER_STEP void publish_partition(Runtime *runtime, int32_t tidx, int32_t nthreads);
    // Whether this executor drives the cores at all. False sends the run down
    // the legacy AICPU dispatch path instead, which owns its own teardown.
    SIMPLER_SCHEDULER_STEP bool drives_cores(const Runtime *runtime) const;
    // Whether the leader waits for the cores to acknowledge the bootstrap.
    SIMPLER_SCHEDULER_STEP bool awaits_bootstrap() const;
    // Watch the run to completion. Runs on the last thread, and its return
    // value is this run's status.
    SIMPLER_SCHEDULER_STEP int32_t supervise(Runtime *runtime);
    // Wait for the supervisor to declare shutdown. Runs on every other thread.
    // Returns whether the wait timed out.
    SIMPLER_SCHEDULER_STEP bool await_shutdown(Runtime *runtime);
};
