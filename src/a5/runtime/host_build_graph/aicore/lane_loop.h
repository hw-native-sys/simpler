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
 * A5 calendar executor. Each lane runs its own ranges between the runtime's
 * READY/release and EXIT/acknowledgement handshakes.
 */

#pragma once

#include "aicore/aicore.h"
#include "aicore/pmu_collector_aicore.h"
#include "common/device_phase.h"
#include "common/platform_config.h"
#include "async_completion_types.h"
#include "scheduler/async_completion.h"
#include "dispatch_payload.h"
#include "runtime.h"
#include "scheduler/calendar_wire.h"
#include "scheduler/wire_types.h"
#include "scheduler/scheduler_args_dump.h"
#include "scheduler/scheduler_args_dump_tensor.h"
#include "scheduler/scheduler_graph.h"
#include "scheduler/gm.h"
#include "scheduler/image_read.h"
#include "scheduler/ssbuf.h"
#include "scheduler/soc_sram.h"
#include "scheduler/scheduler_topology.h"
#include "scheduler/scheduler_watchdog.h"
#include "aicore/aicore_profiling_state.h"
#include "profiling.h"
#include "aicore/chip_swimlane_collector_aicore.h"

namespace simpler::percounter {
namespace {

typedef void (*PercounterKernelFunc)(__gm__ int64_t *);

/** How often a spinning lane looks at the lifecycle notification and the error word. */
inline constexpr uint32_t PERCOUNTER_SPIN_CHECK_INTERVAL = 64;
// The lifecycle word is shared by all lanes, including ones idle for the entire graph.
inline constexpr uint64_t PERCOUNTER_EXIT_POLL_CYCLES = 16 * (PLATFORM_PROF_SYS_CNT_FREQ / 1000000);  // 16 us

struct LaneConfig {
    __gm__ ChipSwimlaneActiveHead *profile_head{nullptr};
    ChipSwimlaneAicoreLocalState *profile{nullptr};
    PercounterScalarTrace *scalar{nullptr};
    __gm__ PercounterProfileBuffer *profile_buffer{nullptr};
    __gm__ PercounterTaskTiming *task_records{nullptr};
    uint32_t *profile_task_count{nullptr};
    uint32_t *task_dropped{nullptr};
    __gm__ PercounterTaskEvent *task_events{nullptr};
    uint32_t task_event_capacity{0};
    mutable uint32_t task_event_count{0};
    mutable uint32_t task_event_dropped{0};
    mutable PercounterTaskTiming pending_task{};
    mutable bool pending_task_valid{false};
    mutable AicoreWorkWindow work_window{};
    __gm__ TaskTimingRecord *task_timing_slots{nullptr};
    mutable __gm__ ChipSwimlaneAicoreTaskRecord *pending_stream_record{nullptr};
    mutable const __gm__ CalendarPeerEndpoint *peer_endpoints{nullptr};
    mutable bool peer_dmb_source_in_use{false};
    __gm__ volatile uint32_t *lifecycle_notify;
    mutable uint64_t last_exit_poll{0};
    mutable bool async_inputs_active{false};
    __gm__ PercounterControl *control;
    bool args_dump_enabled{false};
    uint64_t pmu_reg_base{0};
    __gm__ PercounterCounter *counters;
    uint64_t task_count;
    __gm__ pto2::resident::AsyncGroupData *async_group;
    __gm__ DeferredCompletionEntry *async_entries;
    __gm__ PercounterAsyncCompletion *async_completions;
    uint64_t async_completion_capacity;
    int32_t cluster_index;
    int32_t cluster_lane;
    uint32_t cluster_count;
    uint32_t aiv_per_cluster;
    PERCOUNTER_SSBUF PercounterSsbufRegion *ssbuf;
};

// The next callable owns the entire UB. A previous DMA must finish reading our
// word before the callable can overwrite it, or before the lane retires.
inline __aicore__ void percounter_release_dmb_source(const LaneConfig &cfg) {
#if defined(__CCE_AICORE__) && defined(__DAV_VEC__)
    if (cfg.peer_dmb_source_in_use) {
        wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
        cfg.peer_dmb_source_in_use = false;
    }
#else
    (void)cfg;
#endif
}

inline __aicore__ void percounter_publish_stream_records(ChipSwimlaneAicoreLocalState *profile) {
    auto *records = reinterpret_cast<__gm__ uint8_t *>(profile->cached_buf->records);
    const uint64_t bytes = uint64_t{profile->slot_within_buf} * sizeof(ChipSwimlaneAicoreTaskRecord);
    for (uint64_t offset = 0; offset < bytes; offset += 64)
        dcci(records + offset, SINGLE_CACHE_LINE, CACHELINE_OUT);
    dsb((mem_dsb_t)0);
    chip_swimlane_aicore_publish_task_count(profile);
}

inline __aicore__ void percounter_flush_stream_records(ChipSwimlaneAicoreLocalState *profile) {
    if (profile->cached_buf != nullptr && profile->slot_within_buf > 0 &&
        profile->slot_within_buf < PLATFORM_AICORE_BUFFER_SIZE)
        percounter_publish_stream_records(profile);
}

inline __aicore__ void percounter_commit_task_timing(const LaneConfig &cfg) {
    if (!cfg.pending_task_valid) return;
    if (cfg.profile_buffer != nullptr) {
        const uint32_t index = *cfg.profile_task_count;
        if (index < cfg.profile_buffer->task_capacity) {
            auto *record = cfg.task_records + index;
            record->dispatch = cfg.pending_task.dispatch;
            record->start = cfg.pending_task.start;
            record->finish = cfg.pending_task.finish;
            TaskId::assign(record->task_token, cfg.pending_task.task_token);
            if constexpr (task_profile_compiled) {
                record->entry = cfg.pending_task.entry;
                record->kernel_end = cfg.pending_task.kernel_end;
            }
            ++*cfg.profile_task_count;
        } else if (*cfg.task_dropped != UINT32_MAX) ++*cfg.task_dropped;
    } else {
        auto *record = cfg.pending_stream_record;
        record->start_time = cfg.pending_task.start;
        record->end_time = cfg.pending_task.kernel_end;
        TaskId::assign(record->task_token, cfg.pending_task.task_token);
        record->reg_task_id = cfg.profile->cached_buf_seq * PLATFORM_AICORE_BUFFER_SIZE + cfg.profile->slot_within_buf;
        record->percounter_block_shape =
            static_cast<uint32_t>(cfg.pending_task.entry >> PERCOUNTER_ORDER_BLOCK_IDX_SHIFT);
        if (cfg.profile->slot_within_buf == PLATFORM_AICORE_BUFFER_SIZE) percounter_publish_stream_records(cfg.profile);
    }
    cfg.pending_task_valid = false;
}

inline __aicore__ uint64_t percounter_task_event_clock(const LaneConfig &cfg) {
    if constexpr (task_profile_compiled) {
        if (cfg.task_events != nullptr) return get_sys_cnt_aicore();
    }
    return 0;
}

inline __aicore__ __gm__ PercounterTaskEvent *percounter_record_task_event(
    const LaneConfig &cfg, PercounterTaskEventKind kind, uint64_t entry, uint64_t cycles, uint64_t detail = 0
) {
    if constexpr (task_profile_compiled) {
        if (cycles == 0) return nullptr;
        if (cfg.task_event_count == cfg.task_event_capacity) {
            if (cfg.task_event_dropped != UINT32_MAX) ++cfg.task_event_dropped;
            return nullptr;
        }
        auto *record = &cfg.task_events[cfg.task_event_count++];
        record->cycles = cycles;
        record->entry = entry;
        record->detail = detail;
        record->kind = kind;
        return record;
    }
    return nullptr;
}

__attribute__((noinline, cold)) __aicore__ void
percounter_fail(const LaneConfig &cfg, SchedulerGraphResult result, int64_t task_id = -1, uint64_t detail = 0) {
    if (percounter_gm_compare_exchange(cfg.control->lane_error_claim, uint64_t{0}, uint64_t{1}) != 0) return;
    percounter_gm_store(cfg.control->lane_error_task, static_cast<uint64_t>(task_id + 1));
    percounter_gm_store(cfg.control->lane_error_detail, detail);
    if (percounter_gm_compare_exchange(cfg.control->lane_error, uint64_t{0}, static_cast<uint64_t>(result)) == 0)
        percounter_gm_store(cfg.control->lane_error_claim, uint64_t{2});
}

inline __aicore__ bool percounter_exit_signalled(const LaneConfig &cfg) {
    const uint64_t now = get_sys_cnt_aicore();
    if (now - cfg.last_exit_poll < PERCOUNTER_EXIT_POLL_CYCLES) return false;
    cfg.last_exit_poll = now;
    return (percounter_soc_sram_read(*cfg.lifecycle_notify) & AICORE_LIFECYCLE_EXIT_BIT) != 0;
}

/** True when the lane should stop: EXIT is signalled or some lane has failed. */
inline __aicore__ bool percounter_should_stop(const LaneConfig &cfg, bool check_exit = true) {
    return percounter_gm_read(cfg.control->lane_error) != 0 || (check_exit && percounter_exit_signalled(cfg));
}

inline __aicore__ __gm__ pto2::resident::AsyncGroupData *percounter_init_async_record(
    __gm__ PercounterControl *control, int block_idx, uint64_t group_offset, uint64_t run_epoch
) {
    using pto2::resident::AsyncGroupData;
    const uint64_t lane = static_cast<uint64_t>(block_idx);
    auto *record = scheduler_state_at<AsyncGroupData>(control, group_offset);
    record->run_epoch = run_epoch;
    record->task_ref = 0;
    record->group_index = static_cast<uint32_t>(lane);
    record->block_index = 0;
    record->subtask_slot = 0;
    record->capacity = PERCOUNTER_ASYNC_CAPACITY;
    record->entries_bytes = PERCOUNTER_ASYNC_CAPACITY * sizeof(DeferredCompletionEntry);
    record->count = 0;
    record->error_code = SIMPLER_ERROR_NONE;
    record->reservation_serial = 0;
    record->reservation_limit = 0;
    record->reservation_begin = 0;
    record->reservation_written = 0;
    record->state = pto2::resident::AsyncPublicationState::OPEN;
    record->sdma_guards_offset = percounter_gm_read(control->sdma_guards_offset);
    record->sdma_guard_count = static_cast<uint32_t>(percounter_gm_read(control->cluster_count));
    record->entries_offset = percounter_gm_read(control->async_entries_offset) + lane * record->entries_bytes;
    for (uint64_t offset = 0; offset < sizeof(AsyncGroupData); offset += 64)
        dcci(reinterpret_cast<__gm__ uint8_t *>(record) + offset, SINGLE_CACHE_LINE, CACHELINE_OUT);
    dsb((mem_dsb_t)0);
    return record;
}

inline __aicore__ bool percounter_run_task(
    const LaneConfig &cfg, const __gm__ CalendarDispatch &dispatch, const __gm__ CalendarExecutionExtra *extra,
    uint32_t block, uint32_t blocks, uint32_t role, uint64_t entry, __gm__ CalendarTiming *calendar_timing,
    __gm__ CalendarPmuCell *pmu, uint16_t row_flags = 0
) {
    ProfileScope profile_scope(cfg.scalar, ProfilePhase::Run, entry);
    const bool timing_enabled = cfg.profile != nullptr || cfg.profile_buffer != nullptr;
    const uint64_t receive_time = timing_enabled ? get_sys_cnt_aicore() : 0;
    const uint64_t function = dispatch.kernel_entry;
    auto *arguments = reinterpret_cast<__gm__ DispatchPayload *>(dispatch.args_address);
    if (function == 0 || arguments == nullptr) {
        percounter_fail(
            cfg, SchedulerGraphResult::INVALID_CALLABLE, extra == nullptr ? -1 : static_cast<int64_t>(extra->task_id),
            role
        );
        return false;
    }
    {
        ProfileScope context_scope(cfg.scalar, ProfilePhase::BindContext, entry);
        auto *context = &cfg.ssbuf->contexts[cfg.cluster_lane].value;
        percounter_ssbuf_store_field(&context->local.block_idx, static_cast<int32_t>(block));
        percounter_ssbuf_store_field(&context->local.block_num, static_cast<int32_t>(blocks));
    }
    // Acquire business inputs on the consuming core after its dependencies settle.
    // Once a run has deferred, even later ordinary rows may reuse the same GM lines.
    if (!cfg.async_inputs_active)
        cfg.async_inputs_active =
            (percounter_soc_sram_read(*cfg.lifecycle_notify) & AICORE_LIFECYCLE_ASYNC_INPUT_BIT) != 0;
    if (cfg.async_inputs_active) percounter_observe_data_cache(arguments);
    else percounter_memory_barrier();
    __gm__ ChipSwimlaneAicoreTaskRecord *record = nullptr;
    if (cfg.profile != nullptr) {
        record = chip_swimlane_aicore_wait_task_record(cfg.profile_head, cfg.profile);
        if (record == nullptr) {
            percounter_fail(cfg, SchedulerGraphResult::TIMEOUT);
            return false;
        }
    }

    percounter_release_dmb_source(cfg);
    PercounterKernelFunc kernel = reinterpret_cast<PercounterKernelFunc>(function);
    if (pmu != nullptr) pmu_aicore_begin();
    const bool first_kernel = cfg.work_window.start_cycle == 0;
    __gm__ TaskTimingRecord *slot = nullptr;
    if (__builtin_expect(cfg.task_timing_slots != nullptr, false)) {
        const int32_t timing_slot =
            static_cast<int32_t>((row_flags >> CALENDAR_TASK_TIMING_SHIFT) & CALENDAR_TASK_TIMING_MASK) - 1;
        if (timing_slot >= 0 && timing_slot < NUM_TASK_TIMING_SLOTS) slot = cfg.task_timing_slots + timing_slot;
    }
    const uint64_t start_time =
        timing_enabled || slot != nullptr || first_kernel || calendar_timing != nullptr ? get_sys_cnt_aicore() : 0;
    if (first_kernel) cfg.work_window.start_cycle = start_time;
    if constexpr (profile_compiled) {
        if (cfg.scalar != nullptr) ++cfg.scalar->depth;
    }
    kernel(reinterpret_cast<__gm__ int64_t *>(arguments->args));
    const uint64_t kernel_end = timing_enabled || slot != nullptr ? get_sys_cnt_aicore() : 0;
    if (pmu != nullptr) pmu_aicore_end();
    if constexpr (profile_compiled) {
        if (cfg.scalar != nullptr) --cfg.scalar->depth;
    }
    // Finish writing kernel outputs before any completion can be published.
    {
        ProfileScope fence_scope(cfg.scalar, ProfilePhase::OutputFence, entry);
        OUT_OF_ORDER_STORE_BARRIER();
    }
    const uint64_t end_time = get_sys_cnt_aicore();
    cfg.work_window.end_cycle = end_time;
    if (slot != nullptr && kernel_end > start_time) {
        if (start_time < slot->dispatch_cycle) slot->dispatch_cycle = start_time;
        if (kernel_end > slot->finish_cycle) slot->finish_cycle = kernel_end;
    }
    if (calendar_timing != nullptr) {
        calendar_timing->start = start_time;
        calendar_timing->end = end_time;
    }
    if (pmu != nullptr) {
        pmu_aicore_read_counters(cfg.pmu_reg_base, pmu->counters, &pmu->total_cycles);
        pmu->publication = CalendarPmuPublication::Executed;
    }
    if (timing_enabled) {
        cfg.pending_task.kernel_end = kernel_end;
        cfg.pending_task.entry = entry;
        cfg.pending_task.finish = end_time;
        cfg.pending_task.dispatch = receive_time;
        cfg.pending_task.start = start_time;
        const TaskId token = extra->shell_task_id < 0 ?
                                 TaskId::make_global(static_cast<int32_t>(extra->task_id)) :
                                 TaskId::make_sub_task(extra->shell_task_id, extra->local_task_id);
        TaskId::assign(cfg.pending_task.task_token, token);
        cfg.pending_stream_record = record;
        cfg.pending_task_valid = true;
    }
    return true;
}

inline __aicore__ bool percounter_clear_mix_signal(const LaneConfig &cfg, uint64_t timeout_cycles) {
    auto *ready = scheduler_state_at<uint64_t>(cfg.control, cfg.control->mix_signal_ready_offset) + cfg.cluster_index;
    if (cfg.cluster_lane == 1) {
        auto &signal = cfg.ssbuf->mix;
        for (uint32_t lane = 0; lane < PLATFORM_CORES_PER_BLOCKDIM; ++lane) {
            percounter_ssbuf_store_field(&signal.arrive[lane], uint32_t{0});
            percounter_ssbuf_store_field(&signal.release[lane], uint32_t{0});
        }
        percounter_memory_barrier();
        percounter_gm_store(*ready, uint64_t{1});
    }
    const uint64_t start = get_sys_cnt_aicore();
    while (percounter_gm_read(*ready) == 0) {
        if (percounter_should_stop(cfg)) return false;
        if (get_sys_cnt_aicore() - start > timeout_cycles) {
            percounter_fail(cfg, SchedulerGraphResult::TIMEOUT);
            return false;
        }
    }
    return true;
}

#include "calendar_args_dump.h"
#include "calendar_loop.h"

inline __aicore__ AicoreWorkWindow run_percounter_lane(
    __gm__ PercounterControl *control, int block_idx, CoreType core_type, __gm__ volatile uint32_t *lifecycle_notify
) {
    // The caller invalidates the data cache after observing AICPU READY.
    // Configuration is immutable after that publication; use cached loads here.
    // Progress, dependency and exit words retain their bypass/atomic accesses.
    const bool is_aic = core_type == CoreType::AIC;
    SchedulerClusterCoordinate coordinate{-1, -1};
    const uint64_t cluster_count = control->cluster_count;
    const uint64_t aiv_per_cluster = control->aiv_per_cluster;
    if (!scheduler_cluster_coordinate_from_worker(
            block_idx, is_aic, static_cast<int32_t>(cluster_count), static_cast<int32_t>(aiv_per_cluster), &coordinate
        )) {
        // A lane outside the topology the AICPU published has no payload slot.
        percounter_gm_compare_exchange(
            control->lane_error, uint64_t{0}, static_cast<uint64_t>(SchedulerGraphResult::INVALID_ARGUMENTS)
        );
        return {};
    }

    LaneConfig cfg{};
    cfg.lifecycle_notify = lifecycle_notify;
    cfg.last_exit_poll = get_sys_cnt_aicore();
    PercounterScalarTrace scalar{};
    PercounterScalarBinding scalar_binding;
    uint32_t task_count = 0;
    uint32_t task_dropped = 0;
    const uint64_t level = control->profiling_level;
    if (level >= 2) {
        cfg.profile_buffer =
            scheduler_state_at<PercounterProfileBuffer>(control, control->profile_buffers_offset) + block_idx;
        cfg.task_records = scheduler_state_at<PercounterTaskTiming>(control, cfg.profile_buffer->task_records_offset);
        cfg.profile_task_count = &task_count;
        cfg.task_dropped = &task_dropped;
        cfg.profile_buffer->worker_id = block_idx;
        cfg.profile_buffer->physical_core_id = get_physical_core_id();
        cfg.profile_buffer->core_type = static_cast<int32_t>(core_type);
        cfg.profile_buffer->cluster_index = coordinate.cluster_index;
        cfg.profile_buffer->cluster_lane = coordinate.cluster_lane;
        if constexpr (task_profile_compiled) {
            cfg.task_events =
                scheduler_state_at<PercounterTaskEvent>(control, cfg.profile_buffer->event_records_offset);
            cfg.task_event_capacity = cfg.profile_buffer->event_capacity;
        }
        if constexpr (profile_compiled) {
            scalar.buffer = cfg.profile_buffer;
            scalar.records = scheduler_state_at<uint64_t>(control, scalar.buffer->scalar_records_offset);
            scalar.window_start = percounter_scalar_clock();
            scalar.function_min_cycles = level >= 4 ? 0 : PLATFORM_PROF_SYS_CNT_FREQ / 1000000;
            cfg.scalar = &scalar;
            if (level >= 4) scalar_binding.attach(&scalar);
        }
    }
    ProfileScope setup_scope(cfg.scalar, ProfilePhase::LaneSetup);
    cfg.control = control;
    if (control->task_timing_offset != 0)
        cfg.task_timing_slots = scheduler_state_at<TaskTimingRecord>(control, control->task_timing_offset) +
                                block_idx * NUM_TASK_TIMING_SLOTS;
    cfg.counters = scheduler_state_at<PercounterCounter>(control, control->counters_offset);
    cfg.task_count = control->task_count;
    if (control->table_task_count != control->graph_storage_count) {
        percounter_fail(cfg, SchedulerGraphResult::INVALID_TASK_COUNT);
        return {};
    }
    const uint64_t async_group_offset = percounter_gm_read(control->async_groups_offset) +
                                        static_cast<uint64_t>(block_idx) * sizeof(pto2::resident::AsyncGroupData);
    const uint64_t async_run_epoch = percounter_gm_read(control->async_run_epoch);
    cfg.async_group = percounter_init_async_record(control, block_idx, async_group_offset, async_run_epoch);
    cfg.async_entries = scheduler_state_at<DeferredCompletionEntry>(
        control, control->async_entries_offset +
                     static_cast<uint64_t>(block_idx) * PERCOUNTER_ASYNC_CAPACITY * sizeof(DeferredCompletionEntry)
    );
    cfg.async_completions = scheduler_state_at<PercounterAsyncCompletion>(control, control->async_completions_offset);
    cfg.async_completion_capacity = control->async_completion_capacity;
    cfg.cluster_index = coordinate.cluster_index;
    cfg.cluster_lane = coordinate.cluster_lane;
    cfg.cluster_count = static_cast<uint32_t>(cluster_count);
    cfg.aiv_per_cluster = static_cast<uint32_t>(aiv_per_cluster);

    const uint64_t timeout_cycles = control->scheduler_timeout_cycles;
    // Lane setup precedes the first calendar range.
    dsb((mem_dsb_t)0);
    cfg.ssbuf = percounter_ssbuf_region(platform_ssbuf_hardware_base_address());
    auto *context = &cfg.ssbuf->contexts[cfg.cluster_lane].value;
    context->global.sub_block_id = cfg.cluster_lane == 2 ? 1 : 0;
    context->global.async_present = 1;
    context->global.async_image_address = reinterpret_cast<uint64_t>(control);
    context->global.async_image_bytes = percounter_gm_read(control->image_bytes);
    context->global.async_run_epoch = async_run_epoch;
    context->global.async_group_offset = async_group_offset;
    context->global.async_group_index = static_cast<uint32_t>(block_idx);
    context->global.async_output_capture_refused = 0;
    percounter_ssbuf_store_field(&context->async_touched, uint32_t{0});
#if defined(__CPU_SIM)
    if (!percounter_bind_sim_context(context)) {
        percounter_fail(cfg, SchedulerGraphResult::INVALID_ARGUMENTS);
        return {};
    }
#endif
    if (control->mix_signal_ready_offset != 0 && !percounter_clear_mix_signal(cfg, timeout_cycles)) return {};

    ChipSwimlaneAicoreLocalState profile{};
    if (level == 1 && SIMPLER_GET_DFX_FLAG(get_aicore_profiling_flag(), SIMPLER_DFX_FLAG_CHIP_SWIMLANE)) {
        cfg.profile_head = get_chip_swimlane_aicore_head();
        if (cfg.profile_head != nullptr) cfg.profile = &profile;
    }
    setup_scope.close();
    cfg.args_dump_enabled = percounter_gm_read(control->args_dump_offset) != 0;
    if (SIMPLER_GET_DFX_FLAG(get_aicore_profiling_flag(), SIMPLER_DFX_FLAG_PMU))
        cfg.pmu_reg_base = get_aicore_pmu_reg_base();
    percounter_calendar_loop(cfg);
    percounter_release_dmb_source(cfg);
    percounter_commit_task_timing(cfg);
    if (cfg.profile != nullptr) percounter_flush_stream_records(&profile);
    if (cfg.task_timing_slots != nullptr) {
        auto *slots = reinterpret_cast<__gm__ uint8_t *>(cfg.task_timing_slots);
        for (uint64_t offset = 0; offset < NUM_TASK_TIMING_SLOTS * sizeof(TaskTimingRecord); offset += 64)
            dcci(slots + offset, SINGLE_CACHE_LINE, CACHELINE_OUT);
        dsb((mem_dsb_t)0);
    }
    if (cfg.profile_buffer != nullptr) {
        cfg.profile_buffer->task_committed = task_count;
        cfg.profile_buffer->task_dropped = task_dropped;
        auto *records = reinterpret_cast<__gm__ uint8_t *>(cfg.task_records);
        for (uint64_t offset = 0; offset < uint64_t{task_count} * sizeof(PercounterTaskTiming); offset += 64)
            dcci(records + offset, SINGLE_CACHE_LINE, CACHELINE_OUT);
        if constexpr (task_profile_compiled) {
            cfg.profile_buffer->event_committed = cfg.task_event_count;
            cfg.profile_buffer->event_dropped = cfg.task_event_dropped;
            auto *events = reinterpret_cast<__gm__ uint8_t *>(cfg.task_events);
            for (uint64_t offset = 0; offset < uint64_t{cfg.task_event_count} * sizeof(PercounterTaskEvent);
                 offset += 64)
                dcci(events + offset, SINGLE_CACHE_LINE, CACHELINE_OUT);
        }
        if constexpr (profile_compiled) scalar.finish(percounter_scalar_clock(), level >= 4);
        dcci(cfg.profile_buffer, SINGLE_CACHE_LINE, CACHELINE_OUT);
        dcci(reinterpret_cast<__gm__ uint8_t *>(cfg.profile_buffer) + 64, SINGLE_CACHE_LINE, CACHELINE_OUT);
        dsb((mem_dsb_t)0);
    }
    return cfg.work_window;
}

// Only a worker rejected by this launch may leave without a register handshake.
// The request is an odd EXIT generation. Clearing it acknowledges the request;
// retirement still fails because this worker has no verified physical window.
inline __aicore__ bool percounter_ack_rejected_exit(__gm__ uint32_t &request, uint32_t signal) {
    if ((signal & AICORE_LIFECYCLE_EXIT_BIT) == 0) return false;
    percounter_memory_barrier();  // Order the SoC SRAM observation before reading the published GM request.
    if (percounter_gm_read(request) != (signal & ~AICORE_LIFECYCLE_ASYNC_INPUT_BIT)) return false;
    percounter_gm_store(request, uint32_t{0});
    percounter_memory_barrier();
    return true;
}

enum class PercounterAdmission { Start, Retire, Rejected };

inline __aicore__ PercounterAdmission percounter_wait_for_admission(
    __gm__ PercounterControl *control, int block_idx, __gm__ uint32_t *request, __gm__ volatile uint32_t &lifecycle,
    uint32_t previous
) {
    // A worker that first samples the already-published EXIT has no later
    // transition to wait for. The launch-cleared slot disambiguates a retained EXIT.
    if (request != nullptr && percounter_ack_rejected_exit(*request, previous)) return PercounterAdmission::Rejected;
    uint64_t timeout_cycles = 0;
    if (control != nullptr) {
        percounter_observe_data_cache(reinterpret_cast<__gm__ void *>(control));
        timeout_cycles = percounter_gm_read(control->scheduler_timeout_cycles);
    }
    const uint64_t ready_wait_start = get_sys_cnt_aicore();
    const uint64_t poll_cycles = 512U + 16U * static_cast<uint32_t>(block_idx);
    bool timed_out = false;
    for (;;) {
        const uint32_t signal = percounter_soc_sram_read(lifecycle);
        if (((signal ^ previous) & ~AICORE_LIFECYCLE_ASYNC_INPUT_BIT) != 0) {
            if (signal & AICORE_LIFECYCLE_EXIT_BIT)
                return request != nullptr && percounter_ack_rejected_exit(*request, signal) ?
                           PercounterAdmission::Rejected :
                           PercounterAdmission::Retire;
            if (control != nullptr && !timed_out) return PercounterAdmission::Start;
        }
        if (control != nullptr && !timed_out &&
            scheduler_watchdog_expired(ready_wait_start, get_sys_cnt_aicore(), timeout_cycles)) {
            percounter_gm_compare_exchange(
                control->lane_error, uint64_t{0}, static_cast<uint64_t>(SchedulerGraphResult::TIMEOUT)
            );
            timed_out = true;
        }
        const uint64_t poll_start = get_sys_cnt_aicore();
        while (get_sys_cnt_aicore() - poll_start < poll_cycles)
            SPIN_WAIT_HINT();
    }
}

/**
 * A percounter AICore, from its report to the COND nonce acknowledgement.
 *
 * The lane samples the lifecycle generation before reporting, waits for START,
 * executes its ranges and acknowledges the AICPU's EXIT.
 */
inline __aicore__ void percounter_aicore_execute(__gm__ Runtime *runtime, int block_idx, CoreType core_type) {
    __gm__ Handshake *handshake =
        (__gm__ Handshake *)(&reinterpret_cast<__gm__ RuntimeSharedState *>(runtime->dev.shared_state_addr)
                                  ->workers[block_idx]);
    __gm__ PercounterControl *control =
        reinterpret_cast<__gm__ PercounterControl *>(runtime->dev.scheduler_bootstrap.worker_context_base);
    percounter_observe_cache_line(handshake);

    auto *lifecycle =
        reinterpret_cast<__gm__ volatile uint32_t *>(runtime->dev.scheduler_bootstrap.lifecycle_notify_address);
    // The Host rejects an incomplete lease before launch. A malformed
    // bootstrap cannot publish a report for a gate this lane cannot observe.
    if (lifecycle == nullptr) return;
    const uint32_t previous = percounter_soc_sram_read(*lifecycle);
    percounter_memory_barrier();  // Complete the sample before discovery can authorize START or EXIT.

    const uint64_t report_epoch = get_aicore_report_epoch();
    handshake->work_start_cycle = 0;
    handshake->work_end_cycle = 0;
    if (report_epoch != 0) {
        aicore_stage_native_report(
            handshake, get_physical_core_id(), core_type, static_cast<uint32_t>(block_idx) + 1, 0
        );
        OUT_OF_ORDER_STORE_BARRIER();
        handshake->report_epoch = report_epoch;
    } else {
        handshake->physical_core_id = get_physical_core_id();
        handshake->core_type = core_type;
        OUT_OF_ORDER_STORE_BARRIER();
        handshake->aicore_done = block_idx + 1;
    }
    dcci(handshake, SINGLE_CACHE_LINE, CACHELINE_OUT);
    dsb((mem_dsb_t)0);

    const bool valid_control =
        control != nullptr && reinterpret_cast<uint64_t>(control) % alignof(PercounterControl) == 0;
    // The rejection slots are available even when the calendar control is malformed.
    auto *requests = reinterpret_cast<__gm__ uint32_t *>(runtime->dev.scheduler_bootstrap.rejected_exit_address);
    const auto admission = percounter_wait_for_admission(
        valid_control ? control : nullptr, block_idx, requests == nullptr ? nullptr : requests + block_idx, *lifecycle,
        previous
    );
    if (admission == PercounterAdmission::Rejected) return;
    if (admission == PercounterAdmission::Start) {
        percounter_observe_data_cache(reinterpret_cast<__gm__ void *>(control));
#if defined(__CCE_AICORE__)
        // A preceding invocation may have left the physical core in atomic store mode.
        set_atomic_none();
#endif
        write_reg(RegId::COND, AICORE_IDLE_VALUE);
        const auto work_window = run_percounter_lane(control, block_idx, core_type, lifecycle);
        handshake->work_start_cycle = work_window.start_cycle;
        handshake->work_end_cycle = work_window.end_cycle;
        dcci(handshake, SINGLE_CACHE_LINE, CACHELINE_OUT);
        dsb((mem_dsb_t)0);
        // A finished lane can be idle for most of the run. Space its SRAM reads
        // so retired lanes do not consume the active workers' notification bandwidth.
        while ((percounter_soc_sram_read(*lifecycle) & AICORE_LIFECYCLE_EXIT_BIT) == 0) {
            const uint64_t poll_start = get_sys_cnt_aicore();
            while (get_sys_cnt_aicore() - poll_start < PERCOUNTER_EXIT_POLL_CYCLES)
                SPIN_WAIT_HINT();
        }
    }
    OUT_OF_ORDER_STORE_BARRIER();
    acknowledge_aicore_cond_nonce_and_wait(
        &reinterpret_cast<__gm__ RuntimeSharedState *>(runtime->dev.shared_state_addr)->teardown_gates[block_idx]
    );
}

}  // namespace
}  // namespace simpler::percounter
