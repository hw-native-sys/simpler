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
 * AICPU publishes the topology, resets CntNotify, releases the AICores and
 * watches logical sinks. The assisting threads bind the Graph bodies.
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

inline const uint64_t *percounter_calendar_slots(const CalendarRangeHeader &calendar) {
    return reinterpret_cast<const uint64_t *>(
        reinterpret_cast<const uint8_t *>(&calendar) + calendar.notify_slots_offset
    );
}

inline uint32_t percounter_read_slot(uint64_t address) {
    return __atomic_load_n(reinterpret_cast<const uint32_t *>(address), __ATOMIC_RELAXED);
}

inline void percounter_reset_calendar_notifications(const CalendarRangeHeader &calendar) {
    const auto *slots = percounter_calendar_slots(calendar);
    cache_invalidate_range(slots, static_cast<size_t>(calendar.notify_slots_count) * sizeof(*slots));
    for (uint32_t i = 0; i < calendar.notify_slots_count; ++i)
        *reinterpret_cast<volatile uint32_t *>(slots[i]) = 0;
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
    if (!percounter_sync_widths_fit(control->max_sync_blocks, aic_count, PLATFORM_AIV_CORES_PER_BLOCKDIM)) {
        LOG_ERROR("A5 percounter: sync_start width exceeds the reported topology");
        return -1;
    }
    control->cluster_count = static_cast<uint64_t>(aic_count);
    control->aiv_per_cluster = static_cast<uint64_t>(PLATFORM_AIV_CORES_PER_BLOCKDIM);
    // Publish the configured timeout for AICPU supervision and bounded lane startup waits.
    control->scheduler_timeout_cycles = scheduler_timeout_cycles;
    cache_flush_range(control, sizeof(*control));
    percounter_graph_prepare_init(control);
    const auto *calendar = scheduler_state_at<CalendarRangeHeader>(control, control->calendar_offset);
    if (calendar == nullptr) return -1;
    cache_invalidate_range(calendar, sizeof(*calendar));
    percounter_reset_calendar_notifications(*calendar);
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
    Handshake *handshakes = runtime->get_workers();
    for (int32_t i = lo; i < hi; ++i) {
        if (!lifecycle.core_reported(i)) continue;
        handshakes[i].aicpu_ready = SCHEDULER_RUNTIME_MODE_PERCOUNTER;
    }
    if (hi > lo) cache_flush_range(&handshakes[lo], static_cast<size_t>(hi - lo) * sizeof(Handshake));
    wmb();
}

/** What an AICPU thread other than the supervisor does while the lanes run. */
inline PercounterAicpuTiming *percounter_aicpu_timing(PercounterControl *control, int32_t thread_idx) {
    if (control == nullptr || thread_idx < 0 || thread_idx >= PLATFORM_MAX_AICPU_THREADS) return nullptr;
    const uint64_t offset = control->aicpu_profile_offset;
    if (offset == 0 || offset > control->image_bytes || offset % alignof(PercounterAicpuTiming) != 0 ||
        control->image_bytes - offset < PLATFORM_MAX_AICPU_THREADS * sizeof(PercounterAicpuTiming))
        return nullptr;
    return percounter_region<PercounterAicpuTiming>(control, control->aicpu_profile_offset) + thread_idx;
}

inline void percounter_assist(Runtime *runtime, int32_t thread_idx) {
    percounter_prepare_graphs(runtime, percounter_control(runtime), thread_idx);
}

inline bool percounter_task_published(const PercounterCounter &counter) {
    return (percounter_read_slot(counter.completion_notify_address) & CALENDAR_COUNT_MASK) >=
           counter.completion_notify_target;
}

inline uint64_t percounter_progress_fingerprint(const CalendarRangeHeader &calendar) {
    const auto *slots = percounter_calendar_slots(calendar);
    uint64_t hash = 0xcbf29ce484222325ULL;
    for (uint32_t i = 0; i < calendar.notify_slots_count; ++i)
        hash = (hash ^ percounter_read_slot(slots[i])) * 0x100000001b3ULL;
    return hash;
}

/**
 * Wait until every top-level sink has completed, a lane reports an error, or
 * progress stops for the scheduler timeout. Returns the run status to publish:
 * 0, or the code latched into the shared header.
 *
 * The sinks are what the host found no consumer for. Every task reaches one
 * along consumer edges and a producer completes before its consumer, so the
 * sinks completing covers the graph. A Graph invocation is covered by its
 * shell, which its body's sinks publish into.
 *
 * With no other AICPU thread to assist, the supervisor materializes the Graph
 * bodies itself before it starts watching.
 */
/**
 * This runtime's error, settled into the shared header at the terminal.
 *
 * The supervisor stops watching when the completion count is in, and a lane
 * can fail after that: it publishes its last block and then fails on the way
 * out, or fails while draining, which it does after its final calendar range. The
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
    if (aicpu_thread_count <= 1) percounter_prepare_graphs(runtime, control, thread_idx);
    auto *timing = percounter_aicpu_timing(control, thread_idx);
    if (timing != nullptr) timing->observe_start_cycles = get_sys_cnt_aicpu();
    const uint64_t sink_count = control->sink_count;
    const uint64_t *sinks = scheduler_state_at<uint64_t>(control, control->sinks_offset);
    // Host-built and immutable for the run, but this line can still hold what
    // the previous run left at the same address.
    cache_invalidate_range(sinks, static_cast<size_t>(sink_count) * sizeof(uint64_t));
    const auto *counters = scheduler_state_at<PercounterCounter>(control, control->counters_offset);
    cache_invalidate_range(counters, static_cast<size_t>(control->table_task_count) * sizeof(*counters));
    const auto *calendar = scheduler_state_at<CalendarRangeHeader>(control, control->calendar_offset);
    cache_invalidate_range(calendar, sizeof(*calendar));
    cache_invalidate_range(percounter_calendar_slots(*calendar), calendar->notify_slots_count * sizeof(uint64_t));
    // Published sink counters stay satisfied for the rest of the run.
    uint64_t resolved = 0;
    uint64_t last_progress = ~uint64_t{0};
    uint64_t last_progress_cycles = get_sys_cnt_aicpu();
    // A whole-graph scan costs too much to repeat, so the stall path takes one
    // fingerprint half a budget after the cheap signals go still and declares
    // the stall a half budget later if nothing moved. A healthy run pays
    // nothing: any sink or deferred progress resets both.
    const uint64_t scan_cycles = timeout_cycles / 2;
    uint64_t fingerprint = 0;
    bool scanned = false;
    uint32_t error_poll_count = 0;
    uint64_t error = 0;
    uint64_t outstanding = 0;
    uint64_t registered = 0;
    uint64_t settled = 0;
    while (true) {
        while (resolved < sink_count && percounter_task_published(counters[sinks[resolved]]))
            ++resolved;
        const bool sinks_done = resolved >= sink_count;
        // The final sink publication follows every registration allocation.
        if (sinks_done || error_poll_count == 63) {
            cache_invalidate_range(&control->async_registered, 2 * sizeof(uint64_t));
            registered = control->async_registered;
            settled = control->async_settled;
            outstanding = registered - settled;
        }
        const uint64_t now = get_sys_cnt_aicpu();
        // A deferred kernel that returned without its condition met shows up in
        // the registration count before any counter moves, so both are progress.
        const uint64_t progress = resolved + registered + settled;
        if (progress != last_progress) {
            last_progress = progress;
            last_progress_cycles = now;
            scanned = false;
        }
        if (sinks_done && outstanding == 0) {
            if (timing != nullptr) timing->sinks_observed_cycles = now;
            break;
        }
        if (++error_poll_count == 64) {
            error_poll_count = 0;
            chip_swimlane_aicpu_poll_self_scheduled_cores(thread_idx, aicpu_thread_count, runtime->get_worker_count());
            cache_invalidate_range(&control->lane_error, sizeof(control->lane_error));
            if (control->lane_error != 0) {
                error = control->lane_error;
                cache_invalidate_range(&control->lane_error_claim, 3 * sizeof(uint64_t));
                const int64_t task =
                    control->lane_error_claim == 2 ? static_cast<int64_t>(control->lane_error_task) - 1 : -1;
                LOG_ERROR(
                    "A5 percounter: lane reported error %" PRIu64 "; task=%" PRId64 " detail=%" PRIu64 "; %" PRIu64
                    " of %" PRIu64 " sinks complete",
                    error, task, control->lane_error_detail, resolved, sink_count
                );
                break;
            }
            error = percounter_graph_prepare_error(control);
            if (error != 0) break;
        }
        if (scheduler_watchdog_expired(last_progress_cycles, now, scan_cycles)) {
            const uint64_t current = percounter_progress_fingerprint(*calendar);
            if (!scanned || current != fingerprint) {
                // Some component moved, so the graph is running and only the
                // sinks and the deferred counts were still.
                fingerprint = current;
                scanned = true;
                last_progress_cycles = now;
            } else {
                // Two scans a half-budget apart saw the same component state.
                error = static_cast<uint64_t>(SchedulerGraphResult::TIMEOUT);
                LOG_ERROR(
                    "A5 percounter: no progress; %" PRIu64 " of %" PRIu64 " sinks complete, %" PRIu64
                    " deferred completions outstanding",
                    resolved, sink_count, outstanding
                );
                break;
            }
        }
        SPIN_WAIT_HINT();
    }
    // The loop leaves on the sinks as readily as on an error, and between its
    // last error poll and that exit a lane can still fail -- a lane that fails
    // after publishing its final block sets the word later than any poll inside
    // the loop could read it. So both error words are read once more here: this
    // run is a success only when they are clear at the moment the supervisor
    // stops watching, not merely at its last poll.
    if (error == 0) {
        cache_invalidate_range(&control->lane_error, sizeof(control->lane_error));
        error = control->lane_error;
        if (error != 0) {
            LOG_ERROR(
                "A5 percounter: lane reported error %" PRIu64 " with all %" PRIu64 " sinks complete", error, sink_count
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
    if (timing != nullptr) {
        timing->observe_end_cycles = get_sys_cnt_aicpu();
        cache_flush_range(timing, sizeof(*timing));
    }
    if (error == 0) return 0;
    (void)latch_aicore_scheduler_runtime_error(&header->sched_error_code, error);
    const int32_t rc = runtime_status_from_error_code(header->sched_error_code.load(std::memory_order_acquire));
    return rc != 0 ? rc : -1;
}

}  // namespace simpler::percounter
