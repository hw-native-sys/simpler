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
 * The AICPU's part in a percounter run.
 *
 * The AICPU schedules nothing under percounter: the lanes draw their own work.
 * It runs the runtime's ordinary bring-up and retirement (AicpuExecutor with
 * AicoreLifecycle) and replaces three resident-specific steps with the ones
 * here: publishing the discovered topology into the control block, telling
 * each reported core READY, and watching the progress count. The threads that
 * do not watch materialize the Graph bodies (graph_prepare.h).
 *
 * Header-only because aicpu_executor.cpp is included by name into a unit test.
 */

#pragma once

#include <atomic>
#include <cinttypes>
#include <cstdint>

#include "aicore_lifecycle.h"
#include "aicore_scheduler_error.h"
#include "aicpu/cache_maintenance.h"
#include "aicpu/device_time.h"
#include "common/memory_barrier.h"
#include "common/platform_config.h"
#include "common/unified_log.h"
#include "host_build_graph/runtime_status.h"
#include "host_build_graph/shared_memory.h"
#include "graph_prepare.h"
#include "runtime.h"
#include "aicpu/chip_swimlane_collector_aicpu.h"
#include "scheduler/wire_types.h"
#include "scheduler/scheduler_layout.h"
#include "scheduler/scheduler_watchdog.h"
#include "spin_hint.h"

namespace simpler::percounter {

inline bool percounter_mode_enabled(const Runtime *runtime) {
    return runtime != nullptr && runtime->get_worker_count() > 0 &&
           runtime->dev.scheduler_bootstrap.runtime_mode == SCHEDULER_RUNTIME_MODE_PERCOUNTER;
}

/** The control block this run's host published, or null when the run is not percounter's. */
inline PercounterControl *percounter_control(Runtime *runtime) {
    if (!percounter_mode_enabled(runtime) || runtime->dev.scheduler_bootstrap.worker_context_base == 0) return nullptr;
    return reinterpret_cast<PercounterControl *>(runtime->dev.scheduler_bootstrap.worker_context_base);
}

/**
 * Publish the discovered topology into the control block. Leader only, after
 * every thread has handshaken its slice.
 *
 * A lane derives its cluster from its block index and these two counts, so a
 * partial bring-up must fail here rather than place lanes on clusters that do
 * not exist -- the same topology contract the resident path checks.
 */
inline int32_t
percounter_post_handshake_init(Runtime *runtime, const AicoreLifecycle &lifecycle, uint64_t scheduler_timeout_cycles) {
    PercounterControl *control = percounter_control(runtime);
    if (control == nullptr || runtime->dev.host_total_tasks < 0) return -1;

    int32_t aic_count = 0;
    int32_t aiv_count = 0;
    for (int32_t i = 0; i < lifecycle.core_count(); ++i) {
        // A silent core's slot is zero-initialized, so its type reads as AIC.
        // Counting it would make an incomplete cluster look like a whole one.
        if (!lifecycle.core_reported(i)) continue;
        if (lifecycle.core_type(i) == CoreType::AIC) {
            ++aic_count;
        } else if (lifecycle.core_type(i) == CoreType::AIV) {
            ++aiv_count;
        } else {
            LOG_ERROR("A5 percounter: core %d reported invalid core type", i);
            return -1;
        }
    }
    if (aic_count <= 0 || aic_count > static_cast<int32_t>(SCHEDULER_CLUSTER_CAPACITY) ||
        aiv_count != aic_count * PLATFORM_AIV_CORES_PER_BLOCKDIM) {
        LOG_ERROR("A5 percounter: incomplete cluster topology AIC=%d AIV=%d", aic_count, aiv_count);
        return -1;
    }

    cache_invalidate_range(control, sizeof(*control));
    control->cluster_count = static_cast<uint64_t>(aic_count);
    control->aiv_per_cluster = static_cast<uint64_t>(PLATFORM_AIV_CORES_PER_BLOCKDIM);
    // The host prefilled the compile-time default; this carries any override,
    // which only the AICPU reads.
    control->scheduler_timeout_cycles = scheduler_timeout_cycles;
    cache_flush_range(control, sizeof(*control));
    percounter_graph_prepare_init(control);
    wmb();
    LOG_INFO("A5 percounter: topology %d clusters x %d AIV", aic_count, PLATFORM_AIV_CORES_PER_BLOCKDIM);
    return 0;
}

/**
 * Tell this thread's slice of cores READY.
 *
 * Only a core whose report this run accepted is answered; on a stamped run a
 * silent core is exactly what the epoch check withheld. Unlike resident there
 * is no per-worker context to hand over: the control block address is already
 * in the bootstrap word the host published before launch.
 */
inline void percounter_publish_partition(
    Runtime *runtime, const AicoreLifecycle &lifecycle, int32_t thread_idx, int32_t thread_count
) {
    if (percounter_control(runtime) == nullptr || thread_count <= 0) return;
    const int32_t core_count = lifecycle.core_count();
    const int32_t lo = static_cast<int32_t>((static_cast<int64_t>(thread_idx) * core_count) / thread_count);
    const int32_t hi = static_cast<int32_t>((static_cast<int64_t>(thread_idx + 1) * core_count) / thread_count);
    Handshake *handshakes = runtime->dev.workers;
    for (int32_t i = lo; i < hi; ++i) {
        if (!lifecycle.core_reported(i)) continue;
        handshakes[i].aicpu_ready = SCHEDULER_RUNTIME_MODE_PERCOUNTER;
    }
    if (hi > lo) cache_flush_range(&handshakes[lo], static_cast<size_t>(hi - lo) * sizeof(Handshake));
    wmb();
}

/** What an AICPU thread other than the supervisor does while the lanes run. */
inline void percounter_assist(Runtime *runtime) { percounter_prepare_graphs(runtime, percounter_control(runtime)); }

/**
 * Wait until the lanes have run every executable block, a lane reports an
 * error, or progress stops for the scheduler timeout. Returns the run status
 * to publish: 0, or the code latched into the shared header.
 *
 * The target is `executable_block_count`, not the task count: inline-completed
 * tasks are in no list, so no lane ever counts them.
 *
 * With no other AICPU thread to assist, the supervisor materializes the Graph
 * bodies itself before it starts watching.
 */
/**
 * This runtime's error, settled into the shared header at the terminal.
 *
 * The supervisor stops watching when the completion count is in, and a lane
 * can fail after that: it publishes its last block and then fails on the way
 * out, or fails while draining, which it does after its own list is spent. The
 * terminal is the first point at which every core has stopped, so it is the
 * only place the whole run's error can be read.
 *
 * The header outranks what the supervisor returned, and `latch` keeps the
 * first error, so a run that already failed keeps the error it failed with.
 * The prepare record is read without reporting: whatever first found it named
 * the invocation already.
 */
inline int32_t percounter_settle_run_error(Runtime *runtime) {
    if (runtime == nullptr || runtime->get_gm_sm_ptr() == nullptr) return 0;
    auto *header = static_cast<SharedMemoryHeader *>(runtime->get_gm_sm_ptr());
    PercounterControl *control = percounter_control(runtime);
    if (control != nullptr) {
        cache_invalidate_range(&control->lane_error, sizeof(control->lane_error));
        uint64_t error = control->lane_error;
        if (error == 0) error = percounter_graph_prepare_error(control, false);
        if (error != 0) {
            LOG_ERROR("A5 percounter: run error %" PRIu64 " settled at the terminal", error);
            (void)latch_aicore_scheduler_runtime_error(&header->sched_error_code, error);
        }
    }
    return runtime_status_from_error_code(header->sched_error_code.load(std::memory_order_acquire));
}

inline int32_t
percounter_supervise(Runtime *runtime, uint64_t timeout_cycles, int32_t aicpu_thread_count, int32_t thread_idx) {
    PercounterControl *control = percounter_control(runtime);
    if (control == nullptr || runtime->dev.host_total_tasks < 0 || runtime->get_gm_sm_ptr() == nullptr) {
        LOG_ERROR("A5 percounter: supervisor requires an initialized graph");
        return -1;
    }
    auto *header = static_cast<SharedMemoryHeader *>(runtime->get_gm_sm_ptr());
    const uint64_t start_cycles = get_sys_cnt_aicpu();

    cache_invalidate_range(control, sizeof(*control));
    if (aicpu_thread_count <= 1) percounter_prepare_graphs(runtime, control);
    const uint64_t expected = control->executable_block_count;
    uint64_t last_progress = ~uint64_t{0};
    uint64_t last_progress_cycles = get_sys_cnt_aicpu();
    uint32_t error_poll_count = 0;
    uint64_t error = 0;
    uint64_t outstanding = 0;
    while (true) {
        cache_invalidate_range(&control->completed_block_count, sizeof(control->completed_block_count));
        const uint64_t resolved = control->completed_block_count;
        // Read after the block count, never before. A lane seals a deferred
        // group before it counts the block that opened it, so a count that
        // already includes that block was published after the seal and this
        // read cannot miss the group. Reading the other order would let a run
        // whose last task defers report success with its condition unmet.
        cache_invalidate_range(
            &control->async_outstanding, sizeof(control->async_outstanding) + sizeof(control->async_settled)
        );
        outstanding = control->async_outstanding;
        const uint64_t settled = control->async_settled;
        const uint64_t now = get_sys_cnt_aicpu();
        // Both terms here only rise, so a device still publishing deferred
        // completions keeps refreshing the watchdog. `async_outstanding` stays
        // out of the sum and gates the exit alone, for the reason
        // `PercounterControl` gives: a term that falls as well as rises would
        // refresh the watchdog on a device going nowhere.
        const uint64_t progress = resolved + settled;
        if (progress != last_progress) {
            last_progress = progress;
            last_progress_cycles = now;
        }
        if (resolved >= expected && outstanding == 0) break;
        if (++error_poll_count == 64) {
            error_poll_count = 0;
            chip_swimlane_aicpu_poll_self_scheduled_cores(thread_idx, aicpu_thread_count, runtime->get_worker_count());
            cache_invalidate_range(&control->lane_error, sizeof(control->lane_error));
            if (control->lane_error != 0) {
                error = control->lane_error;
                LOG_ERROR(
                    "A5 percounter: lane reported error %" PRIu64 "; %" PRIu64 " of %" PRIu64 " blocks resolved", error,
                    resolved, expected
                );
                break;
            }
            error = percounter_graph_prepare_error(control);
            if (error != 0) break;
        }
        if (scheduler_watchdog_expired(last_progress_cycles, now, timeout_cycles)) {
            error = static_cast<uint64_t>(SchedulerGraphResult::TIMEOUT);
            LOG_ERROR(
                "A5 percounter: no progress; %" PRIu64 " of %" PRIu64 " blocks resolved, %" PRIu64
                " deferred completions outstanding",
                resolved, expected, outstanding
            );
            break;
        }
        SPIN_WAIT_HINT();
    }
    // The loop leaves on the completion count as readily as on an error, and
    // between its last error poll and that exit a lane can still fail -- a lane
    // that fails after publishing its final block sets the word later than any
    // poll inside the loop could read it. So both error words are read once
    // more here: this run is a success only when they are clear at the moment
    // the supervisor stops watching, not merely at its last poll.
    if (error == 0) {
        cache_invalidate_range(&control->lane_error, sizeof(control->lane_error));
        error = control->lane_error;
        if (error != 0) {
            LOG_ERROR(
                "A5 percounter: lane reported error %" PRIu64 " with all %" PRIu64 " blocks resolved", error, expected
            );
        } else {
            error = percounter_graph_prepare_error(control);
        }
    }
    // After the reads above, so the settle reports what failed rather than
    // reading this back, and before the report, so an assistant still holding
    // an invocation stops at its next slice.
    percounter_graph_prepare_stop(control, error);
    percounter_graph_prepare_report(control, start_cycles);
    if (error == 0) return 0;
    (void)latch_aicore_scheduler_runtime_error(&header->sched_error_code, error);
    const int32_t rc = runtime_status_from_error_code(header->sched_error_code.load(std::memory_order_acquire));
    return rc != 0 ? rc : -1;
}

}  // namespace simpler::percounter
