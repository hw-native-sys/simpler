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
 * The percounter AICore executor: every lane schedules itself.
 *
 * Lanes claim bounded ordinary windows and directly inspect shared MIX
 * records. Cohort entry is a commitment; ordinary work continues while a
 * commitment waits for its peers. AICPU materializes Graph metadata and
 * supervises progress, while AICore lanes resolve dependencies and dispatch.
 *
 * Startup and retirement are the runtime's: the lane reports through the
 * handshake, waits for the AICPU's READY and the DATA_MAIN_BASE release, and
 * after EXIT answers the COND nonce exactly as the resident executor does.
 *
 * Header-only because aicore_executor.cpp is compiled by name into unit tests.
 */

#pragma once

#include "aicore/aicore.h"
#include "common/device_phase.h"
#include "common/platform_config.h"
#include "async_completion_types.h"
#include "scheduler/async_completion.h"
#include "dispatch_payload.h"
#include "runtime.h"
#include "scheduler/wire_types.h"
#include "scheduler/scheduler_graph.h"
#include "scheduler/scheduler_layout.h"
#include "scheduler/image_read.h"
#include "scheduler/gm.h"
#include "scheduler/scheduler_topology.h"
#include "scheduler/scheduler_watchdog.h"
#include "aicore/aicore_profiling_state.h"
#include "profiling.h"
#include "aicore/chip_swimlane_collector_aicore.h"

namespace simpler::percounter {
namespace {

typedef void (*PercounterKernelFunc)(__gm__ int64_t *);

/**
 * Read a counter the way the runtime reads any cross-core word.
 *
 * Not a volatile dereference: on silicon a volatile load never observed a
 * remote atomic across two million polls with the producer confirmed to have
 * published. `ld_dev` reaches a coherent point (~98 cycles at realistic
 * fanout). Not `AtomicAdd(ptr, 0)` either: that is a read-modify-write and
 * would queue every poll at the atomic unit.
 */
inline __aicore__ uint64_t percounter_observe(uint64_t counter_address) {
    __gm__ uint64_t *word = reinterpret_cast<__gm__ uint64_t *>(counter_address);
    return percounter_gm_read(*word);
}

/** How often a spinning lane looks at the exit register and the error word. */
inline constexpr uint32_t PERCOUNTER_SPIN_CHECK_INTERVAL = 64;

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
    mutable __gm__ ChipSwimlaneAicoreTaskRecord *pending_stream_record{nullptr};
    __gm__ PercounterControl *control;
    uint64_t storage_address;
    __gm__ DispatchPayload *arguments;
    __gm__ PercounterTicket *tickets;
    __gm__ uint64_t *order;
    __gm__ PercounterFanin *fanin;
    __gm__ uint64_t *fanin_addresses;
    __gm__ PercounterCounter *counters;
    __gm__ PercounterCompletionSector *completion_sectors;
    __gm__ PercounterTaskMetadata *metadata;
    __gm__ uint64_t *metadata_ready;
    __gm__ PercounterSegment *segments;
    __gm__ PercounterTaskClaim *task_claims;
    __gm__ PercounterClaimSummary *claim_summary;
    __gm__ PercounterStartBarrier *sync_arrivals;
    uint64_t task_count;
    uint64_t table_task_count;
    bool has_predicates;
    __gm__ pto2::resident::AsyncGroupData *async_group;
    __gm__ DeferredCompletionEntry *async_entries;
    __gm__ PercounterAsyncCompletion *async_completions;
    uint64_t async_completion_capacity;

    int32_t cluster_index;
    int32_t cluster_lane;
    uint64_t mix_task_count;
    __gm__ uint64_t *mix_signal_ready;
    PERCOUNTER_SSBUF PercounterSsbufRegion *ssbuf;
    uint32_t segment_begin[PERCOUNTER_LIST_COUNT];
    uint32_t segment_count[PERCOUNTER_LIST_COUNT];
};

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
        record->end_time = cfg.pending_task.finish;
        TaskId::assign(record->task_token, cfg.pending_task.task_token);
        record->reg_task_id = cfg.profile->cached_buf_seq * PLATFORM_AICORE_BUFFER_SIZE + cfg.profile->slot_within_buf;
        record->receive_to_start_cycles = static_cast<uint32_t>(cfg.pending_task.start - cfg.pending_task.dispatch);
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

inline __aicore__ void percounter_record_task_event(
    const LaneConfig &cfg, PercounterTaskEventKind kind, uint64_t entry, uint64_t cycles, uint64_t detail = 0
) {
    if constexpr (task_profile_compiled) {
        if (cycles == 0) return;
        if (cfg.task_event_count == cfg.task_event_capacity) {
            if (cfg.task_event_dropped != UINT32_MAX) ++cfg.task_event_dropped;
            return;
        }
        auto *record = &cfg.task_events[cfg.task_event_count++];
        record->cycles = cycles;
        record->entry = entry;
        record->detail = detail;
        record->kind = kind;
    }
}

/** Where a task's producer counters are: `addresses[0 .. count)`. */
struct PercounterFaninView {
    __gm__ uint64_t *addresses;
    int32_t count;
};

/**
 * Locate `task_id`'s producers.
 *
 * A device load, not a dereference. The table reaches the device by DMA, which
 * does not invalidate this core's scalar data cache, so a plain load can return
 * whatever the line held before the image landed -- on silicon a 64-task chain
 * read a producer address of 0x120000000000 for its 60th task. Invisible in
 * simulation, where every load is coherent.
 */
inline __aicore__ PercounterFaninView percounter_fanin_view(const LaneConfig &cfg, int64_t task_id) {
    const uint64_t record = percounter_gm_read_immutable_word(reinterpret_cast<__gm__ uint32_t *>(&cfg.fanin[task_id]));
    PercounterFaninView view{};
    view.count = static_cast<int32_t>(record & 0xffffffffULL);
    view.addresses = cfg.fanin_addresses + static_cast<int32_t>(record >> 32);
    return view;
}

__attribute__((noinline, cold)) __aicore__ void
percounter_fail(const LaneConfig &cfg, SchedulerGraphResult result, int64_t task_id = -1, uint64_t detail = 0) {
    if (percounter_gm_compare_exchange(cfg.control->lane_error_claim, uint64_t{0}, uint64_t{1}) != 0) return;
    percounter_gm_store(cfg.control->lane_error_task, static_cast<uint64_t>(task_id + 1));
    percounter_gm_store(cfg.control->lane_error_detail, detail);
    if (percounter_gm_compare_exchange(cfg.control->lane_error, uint64_t{0}, static_cast<uint64_t>(result)) == 0)
        percounter_gm_store(cfg.control->lane_error_claim, uint64_t{2});
}

/**
 * A task as a lane runs it: where its storage is, what it waits on, and what
 * its completion publishes.
 *
 * A task-table task is its own storage index and has a fanin row of counter
 * addresses. A Graph body task lives in its invocation's storage at its body
 * index; a root waits on its shell's fanin row, any other on its producers'
 * body indices, and a sink also publishes its shell.
 */
struct PercounterTask {
    int64_t id;     // percounter id: the counter it publishes
    int64_t local;  // index into `graph`
    uint64_t storage_address;
    uint32_t storage_count;
    int64_t shell;  // -1 for a task-table task
    bool sink;
    // Bit zero tags partitioned completion. Body references encode index << 1;
    // task-table references encode the aligned primary counter address.
    bool body_fanin;
    int32_t fanin_count;
    __gm__ uint64_t *fanin;
    int64_t body_base;
    uint64_t arguments_address;
    uint64_t function_address{0};
};

inline __aicore__ SchedulerGraphView percounter_task_graph(const PercounterTask &task) {
    return scheduler_ordinary_graph_view(
        task.storage_address, task.storage_count, task.storage_count > 0 ? task.storage_count - 1 : 0
    );
}

/**
 * Resolve `id` into a task. False, with the lane error set, for an id outside
 * the graph or a body task whose line the AICPU has not written.
 *
 * A body task's line is read after metadata_ready; the AICPU wrote and
 * flushed the line before publishing that token. Invalidation drops what this core
 * held of it from an earlier run. Its counter word changes under other cores'
 * atomics, but nothing here reads it.
 */
inline __aicore__ bool percounter_resolve(const LaneConfig &cfg, int64_t id, PercounterTask *task) {
    ProfileScope profile_scope(cfg.scalar, ProfilePhase::Resolve);
    if (id < 0 || static_cast<uint64_t>(id) >= cfg.task_count) {
        percounter_fail(cfg, SchedulerGraphResult::INVALID_TASK_ID);
        return false;
    }
    if (static_cast<uint64_t>(id) < cfg.table_task_count) {
        const PercounterFaninView view = percounter_fanin_view(cfg, id);
        *task = PercounterTask{
            id,
            id,
            cfg.storage_address,
            static_cast<uint32_t>(cfg.table_task_count),
            -1,
            false,
            false,
            view.count,
            view.addresses,
            0,
            reinterpret_cast<uint64_t>(cfg.arguments + id)
        };
        return true;
    }
    __gm__ PercounterCounter *line = &cfg.counters[id];
    percounter_observe_cache_line(line);
    const uint64_t storage = line->storage_address;
    const uint32_t body_index = line->body_index;
    const uint32_t flags = line->flags;
    const uint32_t shell = line->shell_task_id;
    if (storage == 0 || static_cast<uint64_t>(shell) >= cfg.table_task_count || static_cast<int64_t>(body_index) > id) {
        percounter_fail(cfg, SchedulerGraphResult::INVALID_TASK_ID);
        return false;
    }
    task->id = id;
    task->local = body_index;
    task->storage_address = storage;
    task->storage_count = body_index + 1;
    task->shell = shell;
    task->sink = (flags & PERCOUNTER_BODY_SINK) != 0;
    task->body_base = id - static_cast<int64_t>(body_index);
    task->arguments_address = percounter_gm_read(line->arguments_address);
    if ((flags & PERCOUNTER_BODY_ROOT) != 0) {
        const PercounterFaninView view = percounter_fanin_view(cfg, shell);
        task->body_fanin = false;
        task->fanin_count = view.count;
        task->fanin = view.addresses;
    } else {
        task->body_fanin = true;
        task->fanin_count = static_cast<int32_t>(line->fanin_count);
        task->fanin = scheduler_state_at<uint64_t>(cfg.control, line->fanin_offset);
    }
    return true;
}

#if defined(__CCE_AICORE__)
inline __aicore__ void percounter_prefetch_arguments(uint64_t address, __gm__ PercounterCounter &counter) {
    constexpr uint32_t cache_line_bytes = 64;
    const uint32_t counts = percounter_gm_read(counter.argument_counts);
    const uint32_t tensor_count = counts & 0xffffU;
    const uint32_t argument_bytes = (tensor_count + (counts >> 16)) * sizeof(uint64_t);
    auto *arguments = reinterpret_cast<__gm__ DispatchPayload *>(address);
    for (uint32_t offset = 0; offset < argument_bytes; offset += cache_line_bytes)
        dc_preload(arguments->args, static_cast<int64_t>(offset));
    if (tensor_count != 0) {
        // The Host packer and Graph materializer both place a task's descriptors in order.
        auto *first_tensor = reinterpret_cast<__gm__ uint64_t *>(percounter_gm_read(arguments->args[0]));
        const uint32_t tensor_bytes = tensor_count * sizeof(simpler::hbg::Tensor);
        for (uint32_t offset = 0; offset < tensor_bytes; offset += cache_line_bytes)
            dc_preload(first_tensor, static_cast<int64_t>(offset));
    }
}
#endif

inline __aicore__ bool
percounter_resolve_dispatch(const LaneConfig &cfg, int64_t id, uint32_t mask, PercounterTask *task) {
    if (!percounter_resolve(cfg, id, task)) return false;
    if (mask != 0) {
        auto &counter = cfg.counters[id];
        task->function_address = percounter_gm_read(counter.function_addresses[__builtin_ctz(mask)]);
#if defined(__CCE_AICORE__)
        // Resolve follows parameter publication, so the hints can overlap the fanin wait.
        percounter_prefetch_arguments(task->arguments_address, counter);
#endif
    } else {
        task->function_address = 0;
    }
    return true;
}

/** An aligned primary address with a low-bit partitioned-completion tag. */
inline __aicore__ uint64_t percounter_producer(const LaneConfig &cfg, const PercounterTask &task, int32_t i) {
    const uint64_t value = percounter_gm_read(task.fanin[i]);
    if (!task.body_fanin) return value;
    return reinterpret_cast<uint64_t>(&cfg.counters[static_cast<uint64_t>(task.body_base) + (value >> 1)].value) |
           (value & PERCOUNTER_COMPLETION_PARTITIONED);
}

inline __aicore__ uint64_t
percounter_completion_address(const LaneConfig &cfg, int64_t task_id, uint32_t block, uint32_t blocks) {
    if (blocks == 1) return reinterpret_cast<uint64_t>(&cfg.counters[task_id].value);
    return reinterpret_cast<uint64_t>(
        &cfg.completion_sectors
             [task_id * PERCOUNTER_COMPLETION_SECTOR_COUNT + (block & (PERCOUNTER_COMPLETION_SECTOR_COUNT - 1))]
                 .value
    );
}

/**
 * Record `count` finished subtasks of `task`; called after their store
 * barriers. A body sink's subtasks also count toward its shell, whose counter
 * is preset against every sink subtask of the body.
 */
inline __aicore__ void percounter_publish_completion(
    const LaneConfig &cfg, const PercounterTask &task, uint64_t completion_address, uint64_t count, uint64_t entry
) {
    ProfileScope profile_scope(cfg.scalar, ProfilePhase::Complete);
    percounter_gm_fetch_add_issue(*reinterpret_cast<__gm__ uint64_t *>(completion_address), count);
    const uint64_t issued = percounter_task_event_clock(cfg);
    if (task.sink) percounter_gm_fetch_add_issue(cfg.counters[task.shell].value, count);
    percounter_record_task_event(cfg, PercounterTaskEventKind::CompletionIssue, entry, issued, count);
}

/** True when the lane should stop: EXIT is signalled or some lane has failed. */
inline __aicore__ bool percounter_should_stop(const LaneConfig &cfg) {
    return static_cast<uint32_t>(read_reg(RegId::DATA_MAIN_BASE)) == AICORE_EXIT_SIGNAL ||
           percounter_gm_read(cfg.control->lane_error) != 0;
}

struct PercounterSlot {
    uint64_t entry;
    PercounterTaskMetadata metadata;
    PercounterTask task;
    uint32_t fanin_cursor;
    uint64_t fanin_pending;
    uint64_t fanin_other;
    uint32_t claim_index;
    bool resolved;
    bool ready;
    bool entered;
};

struct PercounterPublicationCursor {
    uint32_t segment{PERCOUNTER_NO_BOUNDARY};
    uint32_t block_count{0};
    uint32_t rank{UINT32_MAX};
    uint32_t next_segment{PERCOUNTER_NO_BOUNDARY};
};

struct PercounterSource {
    uint32_t segment{0};
    PercounterPublicationCursor cohort_floor;
    PercounterPublicationCursor mix_floor;
    uint32_t loaded_segment{UINT32_MAX};
    uint64_t begin{0};
    uint64_t end{0};
    uint64_t task_shape{0};
    uint64_t ticket_begin{0};
    bool interval_spent{false};
};

inline __aicore__ void percounter_initialize_boundaries(const LaneConfig &cfg, PercounterSource *source) {
    const uint32_t domain = cfg.cluster_lane == 0 ? 0 : 1;
    source->cohort_floor.segment = percounter_gm_read(cfg.control->first_boundary_segment[domain]);
    if (cfg.cluster_lane == 1) source->mix_floor.segment = percounter_gm_read(cfg.control->first_boundary_segment[2]);
}

struct PercounterLaneState {
    // Occupancy guards uninitialized slot storage until a claim initializes it.
    __aicore__ PercounterLaneState() {}
    PercounterSlot slots[PERCOUNTER_PENDING_CAPACITY];
    PercounterSource source;
    uint32_t occupied{0};
    uint32_t mix_owned{0};
    uint32_t revisit{0};
    uint32_t sequence{0};
    uint32_t published_floor{0};
    uint32_t observed_mix_floor{0};
    uint32_t required_mix_floor{0};
    uint32_t seen_sequence{0};
    uint32_t shared_active{0};
    bool mix_closed{false};
    int32_t shared_head{-1};
    uint32_t batch{1};
    uint32_t idle_passes{0};
    uint64_t async_head{0};
    int64_t proven_task{-1};
    int32_t committed{-1};
    uint64_t committed_entry{0};
    uint64_t committed_function_address{0};
    uint32_t committed_generation{0};
};

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
    record->sdma_guards_offset = 0;
    record->sdma_guard_count = 0;
    record->entries_offset = percounter_gm_read(control->async_entries_offset) + lane * record->entries_bytes;
    for (uint64_t offset = 0; offset < sizeof(AsyncGroupData); offset += 64)
        dcci(reinterpret_cast<__gm__ uint8_t *>(record) + offset, SINGLE_CACHE_LINE, CACHELINE_OUT);
    dsb((mem_dsb_t)0);
    return record;
}

// Only actual deferrals occupy persistent storage. Each physical invocation
// appends at most one group; the registration scratch is reusable at return.
inline __aicore__ bool percounter_async_seal(
    const LaneConfig &cfg, PercounterLaneState *state, const PercounterTask &task, uint64_t completion_address,
    bool *deferred, uint64_t entry
) {
    ProfileScope profile_scope(cfg.scalar, ProfilePhase::AsyncSeal);
    auto *record = cfg.async_group;
    const uint32_t count = record->count;
    const int32_t error = record->error_code;
    if (error != SIMPLER_ERROR_NONE || count > PERCOUNTER_ASYNC_CAPACITY || record->reservation_limit != 0) {
        percounter_fail(
            cfg, error == SIMPLER_ERROR_ASYNC_WAIT_OVERFLOW      ? SchedulerGraphResult::ASYNC_WAIT_OVERFLOW :
                 error == SIMPLER_ERROR_ASYNC_COMPLETION_INVALID ? SchedulerGraphResult::ASYNC_COMPLETION_INVALID :
                                                                   SchedulerGraphResult::ASYNC_REGISTRATION_FAILED
        );
        return false;
    }
    if (count == 0) return true;
    for (uint32_t i = 0; i < count; ++i) {
        const auto &entry = cfg.async_entries[i];
        const CompletionToken token{
            entry.addr, entry.expected_value, entry.engine, entry.completion_type, entry.backend_cookie
        };
        if (entry._pad != 0 || !percounter_completion_token_valid(token)) {
            percounter_fail(cfg, SchedulerGraphResult::ASYNC_COMPLETION_INVALID);
            return false;
        }
    }
    const uint64_t index = percounter_gm_fetch_add(cfg.control->async_registered, uint64_t{1});
    if (index >= cfg.async_completion_capacity) {
        percounter_fail(cfg, SchedulerGraphResult::ASYNC_REGISTRATION_FAILED);
        return false;
    }
    auto *held = &cfg.async_completions[index];
    held->next = state->async_head;
    held->pending_mask = (uint64_t{1} << count) - 1;
    held->completion_address = completion_address;
    held->shell = task.shell;
    held->sink = task.sink;
    for (uint32_t i = 0; i < count; ++i) {
        const auto &source = cfg.async_entries[i];
        auto &target = held->entries[i];
        target.addr = source.addr;
        target.backend_cookie = source.backend_cookie;
        target.expected_value = source.expected_value;
        target.engine = source.engine;
        target.completion_type = source.completion_type;
        target._pad = 0;
    }
    const uint64_t bytes = offsetof(PercounterAsyncCompletion, entries) + count * sizeof(DeferredCompletionEntry);
    for (uint64_t offset = 0; offset < bytes; offset += 64)
        dcci(reinterpret_cast<__gm__ uint8_t *>(held) + offset, SINGLE_CACHE_LINE, CACHELINE_OUT);
    state->async_head = index + 1;
    record->count = 0;
    dcci(reinterpret_cast<__gm__ uint8_t *>(record), SINGLE_CACHE_LINE, CACHELINE_OUT);
    dsb((mem_dsb_t)0);
    *deferred = true;
    percounter_record_task_event(
        cfg, PercounterTaskEventKind::DeferRegistered, entry, percounter_task_event_clock(cfg), index + 1
    );
    return true;
}

inline __aicore__ bool percounter_async_tick(const LaneConfig &cfg, PercounterLaneState *state, bool *settled) {
    ProfileScope profile_scope(cfg.scalar, ProfilePhase::AsyncPoll);
    *settled = false;
    uint64_t current = state->async_head;
    uint64_t previous = 0;
    bool wrote = false;
    while (current != 0) {
        auto *held = &cfg.async_completions[current - 1];
        uint64_t pending = held->pending_mask;
        const uint64_t next = held->next;
        while (pending != 0) {
            const uint32_t index = static_cast<uint32_t>(__builtin_ctzll(pending));
            const auto &entry = held->entries[index];
            const CompletionToken token{
                entry.addr, entry.expected_value, entry.engine, entry.completion_type, entry.backend_cookie
            };
            const CompletionPollResult result = percounter_poll_valid_completion(token);
            if (result.state == CompletionPollState::FAILED) {
                percounter_fail(cfg, SchedulerGraphResult::ASYNC_COMPLETION_INVALID);
                return false;
            }
            if (result.state != CompletionPollState::READY) break;
            pending &= pending - 1;
        }
        if (pending == 0) {
            // Relay the observed peer publication to downstream task counters.
            dsb((mem_dsb_t)0);
            percounter_gm_fetch_add_issue(*reinterpret_cast<__gm__ uint64_t *>(held->completion_address), uint64_t{1});
            const uint64_t issued = percounter_task_event_clock(cfg);
            if (held->sink) percounter_gm_fetch_add_issue(cfg.counters[held->shell].value, uint64_t{1});
            dsb((mem_dsb_t)0);  // Task/shell completion precedes global settlement.
            percounter_gm_fetch_add_issue(cfg.control->async_settled, uint64_t{1});
            percounter_record_task_event(cfg, PercounterTaskEventKind::DeferredCompletionIssue, 0, issued, current);
            if (previous == 0) state->async_head = next;
            else {
                auto *link = &cfg.async_completions[previous - 1];
                link->next = next;
                wrote = true;
                dcci(reinterpret_cast<__gm__ uint8_t *>(link), SINGLE_CACHE_LINE, CACHELINE_OUT);
            }
            *settled = true;
        } else {
            if (pending != held->pending_mask) {
                held->pending_mask = pending;
                wrote = true;
                dcci(reinterpret_cast<__gm__ uint8_t *>(held), SINGLE_CACHE_LINE, CACHELINE_OUT);
            }
            previous = current;
        }
        current = next;
    }
    if (wrote) dsb((mem_dsb_t)0);
    return true;
}

inline __aicore__ bool percounter_run_task(
    const LaneConfig &cfg, PercounterLaneState *state, const PercounterTask &task, int32_t block, int32_t blocks,
    uint32_t role, uint64_t completion_address, bool *deferred, uint64_t entry
) {
    *deferred = false;
    ProfileScope profile_scope(cfg.scalar, ProfilePhase::Run);
    const bool timing_enabled = cfg.profile != nullptr || cfg.profile_buffer != nullptr;
    const uint64_t receive_time = timing_enabled ? get_sys_cnt_aicore() : 0;
    auto *arguments = reinterpret_cast<__gm__ DispatchPayload *>(task.arguments_address);
    {
        ProfileScope context_scope(cfg.scalar, ProfilePhase::BindContext);
        auto *context = &cfg.ssbuf->contexts[cfg.cluster_lane].value;
        percounter_ssbuf_store_field(&context->local.block_idx, block);
        percounter_ssbuf_store_field(&context->local.block_num, blocks);
    }
    // Order the observed dependencies and the SSBUF context stores before the kernel.
    percounter_memory_barrier();
    if (cfg.has_predicates) {
        const SchedulerPredicateResult verdict =
            scheduler_evaluate_task_predicate(percounter_task_graph(task), task.local);
        if (verdict == SchedulerPredicateResult::MALFORMED) {
            percounter_fail(cfg, SchedulerGraphResult::INVALID_ARGUMENTS);
            return false;
        }
        if (verdict == SchedulerPredicateResult::FAIL) return true;
    }
    const uint64_t function = task.function_address;
    if (function == 0) {
        if (cfg.metadata[task.id].active_mask == 0) return true;
        percounter_fail(cfg, SchedulerGraphResult::INVALID_CALLABLE, task.id, role);
        return false;
    }

    __gm__ ChipSwimlaneAicoreTaskRecord *record = nullptr;
    if (cfg.profile != nullptr) {
        record = chip_swimlane_aicore_wait_task_record(cfg.profile_head, cfg.profile);
        if (record == nullptr) {
            percounter_fail(cfg, SchedulerGraphResult::TIMEOUT);
            return false;
        }
    }

    // The barrier is the publication fence a consumer's spin depends on: the
    // kernel's outputs are in memory before the atomic that announces them.
    PercounterKernelFunc kernel = reinterpret_cast<PercounterKernelFunc>(function);
    const bool first_kernel = cfg.work_window.start_cycle == 0;
    const uint64_t start_time = timing_enabled || first_kernel ? get_sys_cnt_aicore() : 0;
    if (first_kernel) cfg.work_window.start_cycle = start_time;
    if constexpr (profile_compiled) {
        if (cfg.scalar != nullptr) ++cfg.scalar->depth;
    }
    kernel(reinterpret_cast<__gm__ int64_t *>(arguments->args));
    const uint64_t kernel_end = percounter_task_event_clock(cfg);
    OUT_OF_ORDER_STORE_BARRIER();
    if constexpr (profile_compiled) {
        if (cfg.scalar != nullptr) --cfg.scalar->depth;
    }
    const uint64_t end_time = get_sys_cnt_aicore();
    cfg.work_window.end_cycle = end_time;
    if (timing_enabled) {
        if constexpr (task_profile_compiled) {
            cfg.pending_task.kernel_end = kernel_end;
            cfg.pending_task.entry = entry;
        }
        cfg.pending_task.finish = end_time;
        cfg.pending_task.dispatch = receive_time;
        cfg.pending_task.start = start_time;
        const TaskId token = task.shell < 0 ? TaskId::make_global(static_cast<int32_t>(task.id)) :
                                              TaskId::make_sub_task(task.shell, task.local);
        TaskId::assign(cfg.pending_task.task_token, token);
        cfg.pending_stream_record = record;
        cfg.pending_task_valid = true;
    }
    return percounter_async_seal(cfg, state, task, completion_address, deferred, entry);
}

struct PercounterCandidate {
    uint64_t entry{0};
    uint32_t rank{UINT32_MAX};
    uint32_t generation{0};
    uint32_t sequence{0};
    uint32_t mask{0};
    int32_t index{-1};
    bool shared{false};
    bool ready{false};
    uint32_t sync_index{PERCOUNTER_NO_SYNC};
    bool finished{false};
};

inline __aicore__ bool percounter_clear_mix_signal(const LaneConfig &cfg, uint64_t timeout_cycles) {
    if (cfg.mix_task_count == 0) return true;
    auto *ready = cfg.mix_signal_ready + cfg.cluster_index;
    if (cfg.cluster_lane == 1) {
        for (uint32_t i = 0; i < PERCOUNTER_PENDING_CAPACITY; ++i)
            percounter_mix_signal_clear(&cfg.ssbuf->mix[i]);
        percounter_ssbuf_store_release(&cfg.ssbuf->mix_floor_rank, 0);
        percounter_ssbuf_store_field(&cfg.ssbuf->publication_closed, uint32_t{0});
        percounter_ssbuf_store_release(&cfg.ssbuf->publish_sequence, 0);
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

inline __aicore__ bool percounter_metadata_ready(const LaneConfig &cfg, int64_t id) {
    if (static_cast<uint64_t>(id) < cfg.table_task_count) return true;
    return percounter_gm_read(cfg.metadata_ready[id]) != 0;
}

inline __aicore__ PercounterTaskMetadata percounter_metadata(const LaneConfig &cfg, int64_t id) {
    __gm__ PercounterTaskMetadata *m = cfg.metadata + id;
    const uint64_t rank_mask = percounter_gm_read_immutable_word(&m->global_order_id);
    const uint64_t sync_next = percounter_gm_read_immutable_word(&m->sync_index);
    return {
        static_cast<uint32_t>(rank_mask), static_cast<uint16_t>(rank_mask >> 32),
        static_cast<uint16_t>(rank_mask >> 48), static_cast<uint32_t>(sync_next), static_cast<uint32_t>(sync_next >> 32)
    };
}

inline __aicore__ PercounterTaskMetadata percounter_regular_metadata(const LaneConfig &cfg, uint64_t entry) {
    const uint64_t rank_mask =
        percounter_gm_read_immutable_word(&cfg.metadata[percounter_order_task_id(entry)].global_order_id);
    // Segment membership already excludes sync entries, and the order entry
    // carries this task's block count. Only rank/mask needs another GM read.
    return {
        static_cast<uint32_t>(rank_mask), static_cast<uint16_t>(rank_mask >> 32),
        static_cast<uint16_t>(percounter_order_block_num(entry)), PERCOUNTER_NO_SYNC, PERCOUNTER_NO_BOUNDARY
    };
}

inline __aicore__ bool
percounter_probe_completion(const LaneConfig &cfg, uint64_t reference, uint64_t *pending, uint64_t *other) {
    if (*pending == 0) {
        *pending = reference & ~PERCOUNTER_COMPLETION_PARTITIONED;
        *other = 0;
        if ((reference & PERCOUNTER_COMPLETION_PARTITIONED) != 0) {
            const uint64_t task = (*pending - reinterpret_cast<uint64_t>(cfg.counters)) / sizeof(PercounterCounter);
            const uint32_t first = (cfg.cluster_index + cfg.cluster_lane) & (PERCOUNTER_COMPLETION_SECTOR_COUNT - 1);
            const auto *sectors = cfg.completion_sectors + task * PERCOUNTER_COMPLETION_SECTOR_COUNT;
            *pending = reinterpret_cast<uint64_t>(&sectors[first].value);
            // The aligned group address carries the remaining count in its low bits.
            *other = reinterpret_cast<uint64_t>(sectors) | PERCOUNTER_COMPLETION_SECTOR_COUNT;
        }
    }
    while (true) {
        if (!percounter_counter_published(percounter_observe(*pending))) return false;
        constexpr uint64_t tag_mask = sizeof(PercounterCompletionSector) - 1;
        const uint64_t remaining = *other & tag_mask;
        if (remaining <= 1) {
            *pending = 0;
            *other = 0;
            return true;
        }
        const uint64_t base = *other & ~tag_mask;
        const uint64_t next =
            ((*pending - base) / sizeof(PercounterCompletionSector) + 1) & (PERCOUNTER_COMPLETION_SECTOR_COUNT - 1);
        *pending = base + next * sizeof(PercounterCompletionSector);
        *other = base | (remaining - 1);
    }
}

inline __aicore__ bool percounter_probe_fanin(
    const LaneConfig &cfg, const PercounterTask &task, uint32_t *cursor, uint64_t *pending, uint64_t *other
) {
    ProfileScope profile_scope(cfg.scalar, ProfilePhase::Fanin);
    while (*cursor < static_cast<uint32_t>(task.fanin_count)) {
        const uint64_t reference = *pending == 0 ? percounter_producer(cfg, task, *cursor) : 0;
        if (!percounter_probe_completion(cfg, reference, pending, other)) return false;
        ++*cursor;
    }
    return true;
}

// A record remains immutable until every participant has relinquished it.
// Nonparticipants can race reuse, so both generation reads enclose the payload.
inline __aicore__ bool percounter_shared_candidate(const LaneConfig &cfg, uint32_t index, PercounterCandidate *out) {
    auto *record = &cfg.ssbuf->mix[index];
    const uint32_t generation = percounter_ssbuf_load_acquire(&record->generation);
    if (!percounter_mix_armed(generation)) return false;
    const uint64_t entry = percounter_ssbuf_load_field(&record->entry);
    const uint32_t rank = percounter_ssbuf_load_field(&record->global_order_id);
    const uint32_t mask = percounter_ssbuf_load_field(&record->active_mask);
    const uint32_t sequence = percounter_ssbuf_load_field(&record->sequence);
    const uint32_t sync_index = percounter_ssbuf_load_field(&record->sync_index);
    // The field loads complete before the generation is read back. An acquire
    // load carries its barrier after the read, which orders what follows the
    // read-back rather than what precedes it, so the validation needs this
    // barrier of its own: without it a republication between the two reads
    // can pass validation with one generation's entry beside another's mask.
    percounter_memory_barrier();
    if (percounter_ssbuf_load_field(&record->generation) != generation) return false;
    if ((mask & (1U << cfg.cluster_lane)) == 0) return false;
    if (percounter_ssbuf_load_field(&record->completed) == generation) return false;
    // Three monotone tokens of the validated generation, with the same
    // one-sided staleness: one barrier before the caller acts on them.
    const uint32_t fanin_ready = percounter_ssbuf_load_field(&record->fanin_ready);
    const uint32_t finished = percounter_ssbuf_load_field(&record->finished[cfg.cluster_lane]);
    percounter_memory_barrier();
    *out = {entry,      rank,
            generation, sequence,
            mask,       static_cast<int32_t>(index),
            true,       fanin_ready == generation,
            sync_index, finished == generation};
    return true;
}

inline __aicore__ bool
percounter_tokens_ready(PERCOUNTER_SSBUF volatile uint32_t *tokens, uint32_t mask, uint32_t generation) {
    // Each token is one lane's monotone generation, so a load that misses a
    // concurrent publication reads an older value and the conjunction fails:
    // this reports "not yet", never a cohort that has not committed. The
    // tokens need no order among themselves, and only a caller acting on a
    // true verdict needs them ordered against what it does next. One barrier
    // after the last token carries that; a false verdict reads nothing
    // afterwards and takes none.
    for (uint32_t lane = 0; lane < PLATFORM_CORES_PER_BLOCKDIM; ++lane)
        if ((mask & (1U << lane)) && percounter_ssbuf_load_field(tokens + lane) != generation) return false;
    percounter_memory_barrier();
    return true;
}

inline __aicore__ void percounter_discover_records(const LaneConfig &cfg, PercounterLaneState *state) {
    ProfileScope profile_scope(cfg.scalar, ProfilePhase::Discover);
    if (cfg.mix_task_count == 0 || state->mix_closed) return;
    const uint32_t sequence = percounter_ssbuf_load_acquire(&cfg.ssbuf->publish_sequence);
    if (sequence == state->seen_sequence) return;
    state->seen_sequence = sequence;
    if (percounter_ssbuf_load_field(&cfg.ssbuf->publication_closed) != 0) {
        // Acquire closure before dropping records: the earlier sequence may precede
        // the publisher's final FIN acquisitions and record retirements.
        percounter_ssbuf_load_acquire(&cfg.ssbuf->publication_closed);
        state->mix_closed = true;
        state->shared_active = 0;
        state->shared_head = -1;
        return;
    }
    uint32_t active = 0;
    for (uint32_t i = 0; i < PERCOUNTER_PENDING_CAPACITY; ++i) {
        PercounterCandidate c;
        if (percounter_shared_candidate(cfg, i, &c) && c.sequence <= sequence) active |= 1U << i;
    }
    state->shared_active = active;
    state->shared_head = -1;
}

// Retirement revisits can publish an earlier task after a later task.
// Sequence orders notices only within the same task rank.
inline __aicore__ bool
percounter_shared_head(const LaneConfig &cfg, PercounterLaneState *state, PercounterCandidate *candidate) {
    ProfileScope profile_scope(cfg.scalar, ProfilePhase::SharedHead);
    while (state->shared_active != 0) {
        if (state->shared_head < 0) {
            uint32_t active = state->shared_active;
            state->shared_head = static_cast<int32_t>(__builtin_ctz(active));
            if constexpr (PERCOUNTER_PENDING_CAPACITY > 1) {
                active &= active - 1;
                if (active != 0) {
                    auto *first = &cfg.ssbuf->mix[state->shared_head];
                    uint32_t rank = percounter_ssbuf_load_field(&first->global_order_id);
                    uint32_t sequence = percounter_ssbuf_load_field(&first->sequence);
                    while (active != 0) {
                        const uint32_t i = __builtin_ctz(active);
                        active &= active - 1;
                        auto *record = &cfg.ssbuf->mix[i];
                        const uint32_t next_rank = percounter_ssbuf_load_field(&record->global_order_id);
                        const uint32_t next_sequence = percounter_ssbuf_load_field(&record->sequence);
                        if (next_rank < rank || (next_rank == rank && next_sequence < sequence)) {
                            rank = next_rank;
                            sequence = next_sequence;
                            state->shared_head = static_cast<int32_t>(i);
                        }
                    }
                }
            }
        }
        if (percounter_shared_candidate(cfg, state->shared_head, candidate) &&
            candidate->sequence <= state->seen_sequence)
            return true;
        state->shared_active &= ~(1U << state->shared_head);
        state->shared_head = -1;
    }
    *candidate = PercounterCandidate{};
    return false;
}

inline __aicore__ bool
percounter_complete_mix(const LaneConfig &cfg, PercounterLaneState *state, const PercounterCandidate &c) {
    ProfileScope profile_scope(cfg.scalar, ProfilePhase::Aggregate);
    auto *record = &cfg.ssbuf->mix[c.index];
    if (!percounter_tokens_ready(record->finished, c.mask, c.generation)) return false;
    const int64_t id = percounter_order_task_id(c.entry);
    const uint64_t count = static_cast<uint64_t>(__builtin_popcount(c.mask));
    const uint64_t completion = percounter_completion_address(
        cfg, id, percounter_order_block_idx(c.entry), percounter_order_block_num(c.entry)
    );
    if (completion == 0) return false;
    if (static_cast<uint64_t>(id) < cfg.table_task_count) {
        percounter_gm_fetch_add_issue(*reinterpret_cast<__gm__ uint64_t *>(completion), count);
        percounter_record_task_event(
            cfg, PercounterTaskEventKind::CompletionIssue, c.entry, percounter_task_event_clock(cfg), count
        );
    } else {
        PercounterTask task;
        if (!percounter_resolve(cfg, id, &task)) return false;
        percounter_publish_completion(cfg, task, completion, count, c.entry);
    }
    percounter_ssbuf_store_release(&record->completed, c.generation);
    state->shared_active &= ~(1U << c.index);
    state->shared_head = -1;
    return true;
}

/**
 * A participating publisher aggregates and retires its own record. When AIV0
 * is not a participant, completion stays with an executing member so unrelated
 * AIV0 work cannot delay logical completion.
 */
inline __aicore__ bool percounter_retire_owned_mix(const LaneConfig &cfg, PercounterLaneState *state) {
    if (cfg.cluster_lane != 1) return false;
    bool retired = false;
    uint32_t owned = state->mix_owned;
    while (owned != 0) {
        const uint32_t i = __builtin_ctz(owned);
        owned &= owned - 1;
        auto *record = &cfg.ssbuf->mix[i];
        const auto &slot = state->slots[i];
        const uint32_t generation = percounter_ssbuf_load_field(&record->generation);
        if ((slot.metadata.active_mask & 2U) != 0) {
            const PercounterCandidate c{slot.entry,
                                        slot.metadata.global_order_id,
                                        generation,
                                        0,
                                        slot.metadata.active_mask,
                                        static_cast<int32_t>(i),
                                        true,
                                        true,
                                        slot.metadata.sync_index};
            if (!percounter_complete_mix(cfg, state, c)) continue;
        } else if (percounter_ssbuf_load_acquire(&record->completed) != generation) continue;
        percounter_ssbuf_store_release(&record->generation, generation + 1);
        state->mix_owned &= ~(1U << i);
        state->occupied &= ~(1U << i);
        state->revisit |= 1U << i;
        retired = true;
    }
    return retired;
}

inline __aicore__ void percounter_service_records(
    const LaneConfig &cfg, PercounterLaneState *state, PercounterCandidate *candidate, bool probe_dependencies = true
) {
    ProfileScope profile_scope(cfg.scalar, ProfilePhase::SharedService);
    if (percounter_shared_head(cfg, state, candidate)) {
        auto &c = *candidate;
        auto *record = &cfg.ssbuf->mix[c.index];
        if (static_cast<uint32_t>(cfg.cluster_lane) == static_cast<uint32_t>(__builtin_ctz(c.mask))) {
            if (c.finished) {
                if ((c.mask & 2U) == 0 && percounter_complete_mix(cfg, state, c))
                    percounter_shared_head(cfg, state, candidate);
            } else if (!c.ready && percounter_metadata_ready(cfg, percounter_order_task_id(c.entry))) {
                const bool resolved_now = percounter_ssbuf_load_field(&record->metadata_generation) != c.generation;
                if (resolved_now) {
                    PercounterTask task;
                    if (!percounter_resolve(cfg, percounter_order_task_id(c.entry), &task)) return;
                    percounter_ssbuf_store_field(&record->fanin_address, reinterpret_cast<uint64_t>(task.fanin));
                    percounter_ssbuf_store_field(&record->fanin_count, static_cast<uint32_t>(task.fanin_count));
                    percounter_ssbuf_store_field(&record->fanin_body_base, task.body_base);
                    percounter_ssbuf_store_field(&record->fanin_body, static_cast<uint32_t>(task.body_fanin));
                    percounter_ssbuf_store_field(&record->metadata_generation, c.generation);
                }
                if (probe_dependencies || resolved_now) {
                    uint32_t cursor = percounter_ssbuf_load_field(&record->fanin_cursor);
                    uint64_t pending = percounter_ssbuf_load_field(&record->fanin_pending);
                    uint64_t other = percounter_ssbuf_load_field(&record->fanin_other);
                    const uint32_t count = percounter_ssbuf_load_field(&record->fanin_count);
                    auto *fanin =
                        reinterpret_cast<__gm__ uint64_t *>(percounter_ssbuf_load_field(&record->fanin_address));
                    const bool body = percounter_ssbuf_load_field(&record->fanin_body) != 0;
                    while (cursor < count) {
                        uint64_t producer = 0;
                        if (pending == 0) {
                            producer = percounter_gm_read(fanin[cursor]);
                            if (body)
                                producer =
                                    reinterpret_cast<uint64_t>(
                                        &cfg.counters
                                             [percounter_ssbuf_load_field(&record->fanin_body_base) + (producer >> 1)]
                                                 .value
                                    ) |
                                    (producer & PERCOUNTER_COMPLETION_PARTITIONED);
                        }
                        if (!percounter_probe_completion(cfg, producer, &pending, &other)) break;
                        ++cursor;
                    }
                    percounter_ssbuf_store_field(&record->fanin_cursor, cursor);
                    percounter_ssbuf_store_field(&record->fanin_pending, pending);
                    percounter_ssbuf_store_field(&record->fanin_other, other);
                    if (cursor == count) {
                        percounter_ssbuf_store_release(&record->fanin_ready, c.generation);
                        c.ready = true;
                    }
                }
            }
        }
    }
}

inline __aicore__ void percounter_own_entry(
    const LaneConfig &cfg, PercounterLaneState *state, uint64_t entry, uint32_t slot, bool mix,
    const PercounterTaskMetadata &metadata, uint32_t claim_index = UINT32_MAX
) {
    ProfileScope profile_scope(cfg.scalar, ProfilePhase::Publish);
    state->occupied |= 1U << slot;
    state->revisit &= ~(1U << slot);
    auto &owned = state->slots[slot];
    owned.entry = entry;
    owned.metadata = metadata;
    owned.claim_index = claim_index;
    const uint64_t claimed = percounter_task_event_clock(cfg);
    if (!mix) {
        owned.fanin_cursor = 0;
        owned.fanin_pending = 0;
        owned.fanin_other = 0;
        owned.resolved = false;
        owned.ready = false;
        owned.entered = false;
        percounter_record_task_event(cfg, PercounterTaskEventKind::Claim, entry, claimed);
        return;
    }
    state->mix_owned |= 1U << slot;
    auto *record = &cfg.ssbuf->mix[slot];
    const uint32_t generation = percounter_ssbuf_load_acquire(&record->generation) + 1;
    percounter_ssbuf_store_field(&record->entry, entry);
    percounter_ssbuf_store_field(&record->global_order_id, metadata.global_order_id);
    percounter_ssbuf_store_field(&record->active_mask, static_cast<uint32_t>(metadata.active_mask));
    percounter_ssbuf_store_field(&record->sync_index, metadata.sync_index);
    percounter_ssbuf_store_field(&record->sequence, ++state->sequence);
    percounter_ssbuf_store_field(&record->fanin_cursor, uint32_t{0});
    percounter_ssbuf_store_field(&record->fanin_pending, uint64_t{0});
    percounter_ssbuf_store_field(&record->fanin_other, uint64_t{0});
    percounter_ssbuf_store_release(&record->generation, generation);
    percounter_ssbuf_store_release(&cfg.ssbuf->publish_sequence, state->sequence);
    percounter_record_task_event(cfg, PercounterTaskEventKind::Claim, entry, claimed);
}

// Each interval has one final draw; each boundary has one final sector closure.
inline __aicore__ void percounter_close_segment(const LaneConfig &cfg, uint32_t index) {
    percounter_gm_fetch_add_issue(cfg.claim_summary[index / 64].closed, uint64_t{1} << (index % 64));
}

inline __aicore__ bool percounter_claim_exhausted(const LaneConfig &cfg, uint32_t index) {
    return percounter_gm_read(cfg.task_claims[index].closed_sectors) >= PERCOUNTER_CLAIM_SECTOR_COUNT;
}

inline __aicore__ uint32_t
percounter_claim_block(const LaneConfig &cfg, uint32_t index, uint32_t blocks, uint32_t segment) {
    auto *claim = cfg.task_claims + index;
    const uint32_t first = cfg.cluster_index & (PERCOUNTER_CLAIM_SECTOR_COUNT - 1);
    for (uint32_t offset = 0; offset < PERCOUNTER_CLAIM_SECTOR_COUNT; ++offset) {
        const uint32_t part = (first + offset) & (PERCOUNTER_CLAIM_SECTOR_COUNT - 1);
        const uint32_t count = (blocks + PERCOUNTER_CLAIM_SECTOR_COUNT - 1 - part) / PERCOUNTER_CLAIM_SECTOR_COUNT;
        if (count == 0 || percounter_gm_read(claim->sectors[part].next) >= count) continue;
        const uint64_t ticket = percounter_gm_fetch_add(claim->sectors[part].next, uint64_t{1});
        if (ticket >= count) continue;
        if (ticket + 1 == count) {
            const uint64_t closed = percounter_gm_fetch_add(claim->closed_sectors, uint64_t{1});
            if (closed + 1 == PERCOUNTER_CLAIM_SECTOR_COUNT) percounter_close_segment(cfg, segment);
        }
        return part + static_cast<uint32_t>(ticket) * PERCOUNTER_CLAIM_SECTOR_COUNT;
    }
    return blocks;
}

inline __aicore__ void percounter_claim(const LaneConfig &cfg, PercounterLaneState *state) {
    ProfileScope profile_scope(cfg.scalar, ProfilePhase::Claim);
    const uint32_t list = cfg.cluster_lane == 0 ? 0 : 1;
    auto &source = state->source;
    uint32_t free = PERCOUNTER_SLOT_MASK & ~state->occupied;
    if (free == 0) return;
    const bool service_pending = state->occupied != 0 || state->shared_active != 0;
    uint32_t revisit = state->revisit;
    while (revisit != 0) {
        uint32_t slot = __builtin_ctz(revisit);
        uint32_t rest = revisit & (revisit - 1);
        while (rest != 0) {
            const uint32_t other = __builtin_ctz(rest);
            rest &= rest - 1;
            if (state->slots[other].metadata.global_order_id < state->slots[slot].metadata.global_order_id)
                slot = other;
        }
        revisit &= ~(1U << slot);
        state->revisit &= ~(1U << slot);
        auto &previous = state->slots[slot];
        const auto metadata = previous.metadata;
        if (percounter_claim_exhausted(cfg, previous.claim_index)) continue;
        const uint32_t block =
            percounter_claim_block(cfg, previous.claim_index, metadata.block_num, previous.claim_index + list);
        if (block < metadata.block_num) {
            percounter_own_entry(
                cfg, state, percounter_order_entry(percounter_order_task_id(previous.entry), block, metadata.block_num),
                slot, __builtin_popcount(metadata.active_mask) > 1, metadata, previous.claim_index
            );
            return;
        }
        if (service_pending) return;
    }
    uint32_t summary_word = UINT32_MAX;
    uint64_t closed = 0;
    // Each fresh task is visited once; a retired slot retains its own revisit.
    for (; source.segment < cfg.segment_count[list]; ++source.segment) {
        const uint32_t scan = source.segment;
        const uint32_t segment_index = cfg.segment_begin[list] + scan;
        if (summary_word != segment_index / 64) {
            summary_word = segment_index / 64;
            closed = percounter_gm_read(cfg.claim_summary[summary_word].closed);
        }
        const uint32_t bit = segment_index % 64;
        const uint64_t open = ~closed >> bit;
        uint32_t skip = open != 0 ? static_cast<uint32_t>(__builtin_ctzll(open)) : 64 - bit;
        if (skip != 0) {
            if (skip > cfg.segment_count[list] - scan) skip = cfg.segment_count[list] - scan;
            source.segment += skip - 1;
            continue;
        }
        const bool loaded_now = source.loaded_segment != scan;
        if (loaded_now) {
            source.task_shape = percounter_gm_read(cfg.segments[segment_index].task_shape);
            source.loaded_segment = scan;
        }
        const uint64_t shape = source.task_shape;
        const uint32_t blocks = static_cast<uint32_t>(shape >> 48);
        // A closed boundary counter also proves its preceding interval claimed.
        if (shape != 0 && percounter_claim_exhausted(cfg, segment_index - list)) continue;
        if (loaded_now) {
            const uint64_t bounds = percounter_gm_read_immutable_word(&cfg.segments[segment_index].begin);
            source.begin = static_cast<uint32_t>(bounds);
            source.end = static_cast<uint32_t>(bounds >> 32);
            source.interval_spent = source.begin == source.end;
            if (!source.interval_spent)
                source.ticket_begin = percounter_gm_read(cfg.segments[segment_index].ticket_begin);
        }
        if (!source.interval_spent) {
            uint32_t count = static_cast<uint32_t>(__builtin_popcount(free));
            if (count > state->batch) count = state->batch;
            const uint64_t size = source.end - source.begin;
            const uint64_t first = percounter_gm_fetch_add(cfg.tickets[source.ticket_begin].next, uint64_t{count});
            source.interval_spent = first + count >= size;
            if (first < size) {
                if (shape == 0 && first + count >= size) percounter_close_segment(cfg, segment_index);
                for (uint32_t k = 0; k < count && first + k < size; ++k) {
                    const uint32_t slot = __builtin_ctz(free);
                    free &= ~(1U << slot);
                    const uint64_t entry = percounter_gm_read(cfg.order[source.begin + first + k]);
                    percounter_own_entry(cfg, state, entry, slot, false, percounter_regular_metadata(cfg, entry));
                }
                return;
            }
            if (service_pending) {
                if (shape == 0) ++source.segment;
                return;
            }
        }
        if (shape != 0) {
            const uint32_t mask = static_cast<uint32_t>(shape >> 32) & 7U;
            const bool mix = __builtin_popcount(mask) > 1;
            if (!mix || cfg.cluster_lane == 1) {
                const uint32_t claim_index = segment_index - list;
                const uint32_t block = percounter_claim_block(cfg, claim_index, blocks, segment_index);
                if (block < blocks) {
                    const uint32_t task = static_cast<uint32_t>(shape);
                    const uint32_t rank =
                        static_cast<uint32_t>(percounter_gm_read_immutable_word(&cfg.metadata[task].global_order_id));
                    const uint32_t sync =
                        (shape & (uint64_t{8} << 32)) ?
                            static_cast<uint32_t>(percounter_gm_read_immutable_word(&cfg.metadata[task].sync_index)) :
                            PERCOUNTER_NO_SYNC;
                    const PercounterTaskMetadata metadata{
                        rank, static_cast<uint16_t>(mask), static_cast<uint16_t>(blocks), sync, PERCOUNTER_NO_BOUNDARY
                    };
                    percounter_own_entry(
                        cfg, state, percounter_order_entry(task, block, blocks), __builtin_ctz(free), mix, metadata,
                        claim_index
                    );
                    ++source.segment;
                    return;
                }
                if (service_pending) {
                    ++source.segment;
                    return;
                }
            }
        }
    }
}

// A claim is not visible to peers until its SSBUF record has been published.
inline __aicore__ uint32_t percounter_source_floor(const LaneConfig &cfg, PercounterSource &source, bool mix_only) {
    const uint32_t list = cfg.cluster_lane == 0 ? 0 : 1;
    auto &cursor = mix_only ? source.mix_floor : source.cohort_floor;
    while (cursor.segment != PERCOUNTER_NO_BOUNDARY) {
        const uint32_t index = cursor.segment;
        uint32_t blocks = cursor.block_count;
        uint32_t task = 0;
        if (blocks == 0) {
            const uint64_t shape = source.loaded_segment == index - cfg.segment_begin[list] ?
                                       source.task_shape :
                                       percounter_gm_read(cfg.segments[index].task_shape);
            blocks = static_cast<uint32_t>(shape >> 48);
            task = static_cast<uint32_t>(shape);
            cursor.next_segment =
                static_cast<uint32_t>(percounter_gm_read_immutable_word(&cfg.metadata[task].sync_index) >> 32);
        }
        if (!percounter_claim_exhausted(cfg, index - list)) {
            if (cursor.block_count == 0) {
                cursor.block_count = blocks;
                cursor.rank =
                    static_cast<uint32_t>(percounter_gm_read_immutable_word(&cfg.metadata[task].global_order_id));
            }
            return cursor.rank;
        }
        cursor.block_count = 0;
        cursor.segment = cursor.next_segment;
    }
    return UINT32_MAX;
}

inline __aicore__ void percounter_publish_mix_floor(const LaneConfig &cfg, PercounterLaneState *state) {
    if (state->mix_closed) return;
    const uint32_t floor = percounter_source_floor(cfg, state->source, true);
    if (floor != state->published_floor) {
        percounter_ssbuf_store_release(&cfg.ssbuf->mix_floor_rank, floor);
        state->published_floor = floor;
    }
    if (floor == UINT32_MAX && state->mix_owned == 0) {
        // All MIX claims are exhausted and every local publication is retired.
        // The closing sequence orders this token after every participant's FIN.
        percounter_ssbuf_store_field(&cfg.ssbuf->publication_closed, uint32_t{1});
        percounter_ssbuf_store_release(&cfg.ssbuf->publish_sequence, ++state->sequence);
        state->mix_closed = true;
        state->shared_active = 0;
        state->shared_head = -1;
    }
}

inline __aicore__ int32_t percounter_candidate_identity(const PercounterCandidate &candidate) {
    return candidate.shared ? PERCOUNTER_PENDING_CAPACITY + candidate.index : candidate.index;
}

inline __aicore__ bool percounter_candidate_before(const PercounterCandidate &a, const PercounterCandidate &b) {
    return a.rank < b.rank || (a.rank == b.rank && a.sequence < b.sequence);
}

inline __aicore__ bool
percounter_commit_matches(const PercounterLaneState &state, const PercounterCandidate &candidate) {
    return candidate.index >= 0 && state.committed == percounter_candidate_identity(candidate) &&
           state.committed_entry == candidate.entry && state.committed_generation == candidate.generation;
}

inline __aicore__ bool percounter_commit_live(const LaneConfig &cfg, const PercounterLaneState &state) {
    if (state.committed < 0) return false;
    if (state.committed < static_cast<int32_t>(PERCOUNTER_PENDING_CAPACITY)) {
        const uint32_t bit = 1U << state.committed;
        return (state.occupied & ~state.mix_owned & bit) && state.slots[state.committed].entered &&
               state.slots[state.committed].entry == state.committed_entry;
    }
    const uint32_t index = static_cast<uint32_t>(state.committed) - PERCOUNTER_PENDING_CAPACITY;
    auto *record = &cfg.ssbuf->mix[index];
    return percounter_ssbuf_load_acquire(&record->generation) == state.committed_generation &&
           percounter_ssbuf_load_acquire(&record->completed) != state.committed_generation;
}

inline __aicore__ void percounter_start_arrive(const LaneConfig &cfg, uint32_t sync, uint32_t block) {
    percounter_gm_fetch_add_issue(
        cfg.sync_arrivals[sync].sectors[block & (PERCOUNTER_START_SECTOR_COUNT - 1)].next, uint64_t{1}
    );
}

inline __aicore__ bool percounter_start_ready(const LaneConfig &cfg, uint32_t sync, uint32_t blocks, uint32_t block) {
    // Rotate probes to spread readers as well as arrival writers. Zero-sized
    // sectors are complete without a read; every nonempty sector must arrive.
    for (uint32_t offset = 0; offset < PERCOUNTER_START_SECTOR_COUNT; ++offset) {
        const uint32_t part = (block + offset) & (PERCOUNTER_START_SECTOR_COUNT - 1);
        const uint32_t expected = (blocks + PERCOUNTER_START_SECTOR_COUNT - 1 - part) / PERCOUNTER_START_SECTOR_COUNT;
        if (expected != 0 && percounter_gm_read(cfg.sync_arrivals[sync].sectors[part].next) != expected) return false;
    }
    return true;
}

inline __aicore__ bool percounter_step(const LaneConfig &cfg, PercounterLaneState *state, bool *executed) {
    ProfileScope profile_scope(cfg.scalar, ProfilePhase::Step);
    if (!percounter_async_tick(cfg, state, executed)) return false;
    // Progress includes record retirement and async settlement, not only kernel calls.
    if (percounter_retire_owned_mix(cfg, state)) *executed = true;
    if (*executed) state->idle_passes = 0;
    // Only repeated failed dependency reads share this bounded cadence.
    // Publication, entry, FIN and new claims are serviced on every pass.
    const bool probe_dependencies = state->idle_passes < 16 || (state->idle_passes & 3U) == 0;
    const bool mix_coordinator = cfg.cluster_lane == 1 && cfg.mix_task_count != 0 && !state->mix_closed;
    const bool claimed = state->occupied == 0;
    if (claimed) percounter_claim(cfg, state);
    if (mix_coordinator) percounter_publish_mix_floor(cfg, state);
    // An acquired lower bound remains valid for every later scan. Refresh
    // only when a candidate needs to cross it, always before discovery.
    if (cfg.mix_task_count == 0 || state->mix_closed) state->observed_mix_floor = UINT32_MAX;
    else if (cfg.cluster_lane == 1) state->observed_mix_floor = state->published_floor;
    else if (state->required_mix_floor > state->observed_mix_floor)
        state->observed_mix_floor = percounter_ssbuf_load_acquire(&cfg.ssbuf->mix_floor_rank);
    PercounterCandidate shared_candidate;
    if (!state->mix_closed) {
        percounter_discover_records(cfg, state);
        percounter_service_records(cfg, state, &shared_candidate, probe_dependencies);
    }
    PercounterCandidate cohort;
    PercounterCandidate ordinary;
    for (uint32_t i = 0; i < PERCOUNTER_PENDING_CAPACITY; ++i) {
        if ((state->occupied & ~state->mix_owned & (1U << i)) == 0) continue;
        auto &slot = state->slots[i];
        const int64_t id = percounter_order_task_id(slot.entry);
        const bool resolved_now = !slot.resolved;
        if (!slot.resolved && percounter_metadata_ready(cfg, id)) {
            if (!percounter_resolve_dispatch(cfg, id, slot.metadata.active_mask, &slot.task)) return false;
            slot.resolved = true;
        }
        if (slot.resolved && !slot.ready && (probe_dependencies || resolved_now)) {
            slot.ready =
                slot.task.id == state->proven_task ||
                percounter_probe_fanin(cfg, slot.task, &slot.fanin_cursor, &slot.fanin_pending, &slot.fanin_other);
            if (slot.ready) state->proven_task = slot.task.id;
        }
        PercounterCandidate c{
            slot.entry, slot.metadata.global_order_id, 0, 0, slot.metadata.active_mask, static_cast<int32_t>(i), false,
            slot.ready, slot.metadata.sync_index
        };
        if (slot.metadata.sync_index != PERCOUNTER_NO_SYNC) {
            if (percounter_candidate_before(c, cohort)) cohort = c;
        } else if (c.ready && percounter_candidate_before(c, ordinary)) ordinary = c;
    }
    if (shared_candidate.index >= 0) {
        if (percounter_candidate_before(shared_candidate, cohort)) cohort = shared_candidate;
    }
    if (state->committed >= 0 && !percounter_commit_matches(*state, cohort)) {
        if (percounter_commit_live(cfg, *state)) {
            percounter_fail(
                cfg, SchedulerGraphResult::INVALID_ARGUMENTS, percounter_order_task_id(state->committed_entry)
            );
            return false;
        }
        state->committed = -1;
    }
    state->required_mix_floor = cohort.index >= 0 ? cohort.rank : 0;
    const bool ordinary_first = ordinary.index >= 0 && percounter_candidate_before(ordinary, cohort) &&
                                !percounter_commit_matches(*state, cohort);
    bool cohort_go = false;
    bool ordinary_allowed = true;
    const bool cohort_idle =
        cohort.shared && !cohort.finished && percounter_commit_matches(*state, cohort) &&
        percounter_ssbuf_load_acquire(&cfg.ssbuf->mix[cohort.index].enter_idle[cfg.cluster_lane]) == cohort.generation;
    if (cohort_idle) {
        ProfileScope idle_scope(cfg.scalar, ProfilePhase::Idle);
        // All entry commitments and sync arrivals precede this idle promise.
        ordinary_allowed = false;
        cohort_go = percounter_tokens_ready(cfg.ssbuf->mix[cohort.index].enter_idle, cohort.mask, cohort.generation);
    } else if (cohort.index >= 0 && cohort.ready && !ordinary_first && state->observed_mix_floor >= cohort.rank &&
               percounter_source_floor(cfg, state->source, false) >= cohort.rank) {
        ProfileScope entry_scope(cfg.scalar, ProfilePhase::Entry);
        const int32_t identity = cohort.shared ? PERCOUNTER_PENDING_CAPACITY + cohort.index : cohort.index;
        if (state->committed >= 0 && state->committed != identity) {
            percounter_fail(cfg, SchedulerGraphResult::INVALID_ARGUMENTS);
            return false;
        }
        auto *record = cohort.shared ? &cfg.ssbuf->mix[cohort.index] : nullptr;
        if (!cohort.finished) {
            if (cohort.mask != 0 && !percounter_commit_matches(*state, cohort)) {
                const uint32_t role = cohort.shared ? cfg.cluster_lane : __builtin_ctz(cohort.mask);
                const int64_t id = percounter_order_task_id(cohort.entry);
                const uint64_t function = cohort.shared ?
                                              percounter_gm_read(cfg.counters[id].function_addresses[role]) :
                                              state->slots[cohort.index].task.function_address;
                if (function == 0) {
                    percounter_fail(cfg, SchedulerGraphResult::INVALID_CALLABLE, id, role);
                    return false;
                }
                if (cohort.shared) {
                    state->committed_function_address = function;
                }
            }

            const uint32_t sync_index = cohort.sync_index;
            const uint32_t blocks = static_cast<uint32_t>(percounter_order_block_num(cohort.entry));
            state->committed = identity;
            state->committed_entry = cohort.entry;
            state->committed_generation = cohort.generation;
            if (cohort.shared) {
                if (percounter_ssbuf_load_acquire(&record->enter_ready[cfg.cluster_lane]) != cohort.generation) {
                    percounter_ssbuf_store_release(&record->enter_ready[cfg.cluster_lane], cohort.generation);
#if defined(__CCE_AICORE__)
                    // This generation's fanin_ready also publishes its immutable Graph metadata.
                    const int64_t id = percounter_order_task_id(cohort.entry);
                    auto &counter = cfg.counters[id];
                    const uint64_t address = static_cast<uint64_t>(id) < cfg.table_task_count ?
                                                 reinterpret_cast<uint64_t>(cfg.arguments + id) :
                                                 percounter_gm_read(counter.arguments_address);
                    percounter_prefetch_arguments(address, counter);
#endif
                }
                cohort_go = percounter_tokens_ready(record->enter_ready, cohort.mask, cohort.generation);
                if (cohort_go && sync_index != PERCOUNTER_NO_SYNC) {
                    if (static_cast<uint32_t>(cfg.cluster_lane) == static_cast<uint32_t>(__builtin_ctz(cohort.mask)) &&
                        percounter_ssbuf_load_acquire(&record->start_arrived) != cohort.generation) {
                        percounter_start_arrive(cfg, sync_index, percounter_order_block_idx(cohort.entry));
                        percounter_ssbuf_store_release(&record->start_arrived, cohort.generation);
                    }
                }
            } else {
                auto &slot = state->slots[cohort.index];
                if (!slot.entered) {
                    slot.entered = true;
                    percounter_start_arrive(cfg, sync_index, percounter_order_block_idx(cohort.entry));
                }
                cohort_go = true;
            }
            if (cohort_go && sync_index != PERCOUNTER_NO_SYNC) {
                if (!cohort.shared ||
                    static_cast<uint32_t>(cfg.cluster_lane) == static_cast<uint32_t>(__builtin_ctz(cohort.mask))) {
                    cohort_go =
                        percounter_start_ready(cfg, sync_index, blocks, percounter_order_block_idx(cohort.entry));
                    if (cohort_go && cohort.shared)
                        percounter_ssbuf_store_release(&record->start_ready, cohort.generation);
                } else {
                    // One GM observer per block; other participants consume its
                    // generation-tagged publication through the local SRAM.
                    cohort_go = percounter_ssbuf_load_acquire(&record->start_ready) == cohort.generation;
                }
            }
            if (cohort_go && cohort.shared) {
                // A peer may reach ENTER_READY during this lane's ordinary
                // kernel. Its UB and pipelines become available only here.
                percounter_ssbuf_store_release(&record->enter_idle[cfg.cluster_lane], cohort.generation);
                ordinary_allowed = false;
                cohort_go = percounter_tokens_ready(record->enter_idle, cohort.mask, cohort.generation);
            }
        }
    }
    PercounterCandidate selected = ordinary_allowed ? ordinary : PercounterCandidate{};
    if (cohort_go) selected = cohort;
    if (selected.index >= 0) {
        percounter_record_task_event(
            cfg, PercounterTaskEventKind::Ready, selected.entry, percounter_task_event_clock(cfg)
        );
        if (!selected.shared && selected.mask == 0 && !cfg.has_predicates &&
            selected.sync_index == PERCOUNTER_NO_SYNC) {
            // A dependency-only block has no payload or tensor inputs to
            // acquire. Its fanin still gates the ordinary completion edge.
            dsb((mem_dsb_t)0);  // Order the observed dependencies before relaying completion.
            const uint64_t completion = percounter_completion_address(
                cfg, percounter_order_task_id(selected.entry), percounter_order_block_idx(selected.entry),
                percounter_order_block_num(selected.entry)
            );
            if (completion == 0) return false;
            percounter_publish_completion(
                cfg, state->slots[selected.index].task, completion, uint64_t{1}, selected.entry
            );
            state->occupied &= ~(1U << selected.index);
            if (state->slots[selected.index].claim_index != UINT32_MAX) state->revisit |= 1U << selected.index;
            *executed = true;
            state->idle_passes = 0;
            return true;
        }
        const bool is_cohort = selected.shared || selected.sync_index != PERCOUNTER_NO_SYNC;
        PercounterTask shared_task{};
        if (selected.shared) {
            const int64_t id = percounter_order_task_id(selected.entry);
            if (static_cast<uint64_t>(id) >= cfg.table_task_count || cfg.has_predicates) {
                if (!percounter_resolve(cfg, id, &shared_task)) return false;
            } else {
                // Dispatch and completion need identity, not the proven fanin row.
                shared_task.id = id;
                shared_task.shell = -1;
                shared_task.arguments_address = reinterpret_cast<uint64_t>(cfg.arguments + id);
            }
            shared_task.function_address = state->committed_function_address;
        }
        const PercounterTask *task = selected.shared ? &shared_task : &state->slots[selected.index].task;
        const bool measure_batch = !selected.shared && percounter_order_block_num(selected.entry) == 1;
        const uint64_t start = measure_batch ? get_sys_cnt_aicore() : 0;
        bool deferred = false;
        const int32_t block = percounter_order_block_idx(selected.entry);
        const int32_t blocks = percounter_order_block_num(selected.entry);
        const uint64_t completion = percounter_completion_address(cfg, task->id, block, blocks);
        const uint32_t role =
            selected.shared ? cfg.cluster_lane : (selected.mask == 0 ? 0 : __builtin_ctz(selected.mask));
        if (!percounter_run_task(cfg, state, *task, block, blocks, role, completion, &deferred, selected.entry))
            return false;
        state->batch = measure_batch && get_sys_cnt_aicore() - start < PLATFORM_PROF_SYS_CNT_FREQ * 3 / 1000000 ?
                           PERCOUNTER_PENDING_CAPACITY :
                           1;
        if (selected.shared) {
            // Remove a deferred share before FIN; the aggregator adds the full
            // block share, and its owner restores this share after the condition.
            if (deferred) percounter_publish_completion(cfg, *task, completion, ~uint64_t{0}, selected.entry);
            percounter_ssbuf_store_release(
                &cfg.ssbuf->mix[selected.index].finished[cfg.cluster_lane], selected.generation
            );
            if (cfg.cluster_lane == 1) percounter_retire_owned_mix(cfg, state);
            else if ((selected.mask & 2U) == 0 &&
                     static_cast<uint32_t>(cfg.cluster_lane) == static_cast<uint32_t>(__builtin_ctz(selected.mask)))
                percounter_complete_mix(cfg, state, selected);
        } else {
            if (!deferred) percounter_publish_completion(cfg, *task, completion, uint64_t{1}, selected.entry);
            state->occupied &= ~(1U << selected.index);
            if (state->slots[selected.index].claim_index != UINT32_MAX) state->revisit |= 1U << selected.index;
            if (is_cohort) state->committed = -1;
        }
        percounter_commit_task_timing(cfg);
        *executed = true;
        state->idle_passes = 0;
        return true;
    }
    if (!claimed) {
        percounter_claim(cfg, state);
        if (mix_coordinator) percounter_publish_mix_floor(cfg, state);
    }
    ++state->idle_passes;
    return true;
}

inline __aicore__ bool percounter_lane_quiescent(const LaneConfig &cfg, const PercounterLaneState &state) {
    if (state.source.segment < cfg.segment_count[cfg.cluster_lane == 0 ? 0 : 1]) return false;
    return state.occupied == 0 && state.revisit == 0 && state.mix_owned == 0 && state.shared_active == 0 &&
           state.async_head == 0 && (cfg.mix_task_count == 0 || state.mix_closed);
}

template <int32_t ClusterLane>
inline __aicore__ void percounter_lane_loop(LaneConfig &cfg) {
    cfg.cluster_lane = ClusterLane;
    if (percounter_should_stop(cfg)) return;
    PercounterLaneState state{};
    percounter_initialize_boundaries(cfg, &state.source);
    for (uint64_t iterations = 0;; ++iterations) {
        bool executed;
        if (!percounter_step(cfg, &state, &executed)) break;
        if constexpr (profile_compiled) {
            if (cfg.scalar != nullptr) cfg.scalar->end_iteration(executed);
        }
        // A lane ends its own scheduling once its local obligations are
        // drained; that is a statement about this lane, not about the graph.
        if (percounter_lane_quiescent(cfg, state)) break;
        if ((iterations % PERCOUNTER_SPIN_CHECK_INTERVAL) != 0) continue;
        // The AICPU decides when the graph is done and how long a stall may
        // last: it signals EXIT once every top-level sink has completed and
        // every deferred record has settled. A lane publishes task completion
        // and nothing else, and holds no whole-graph judgement of its own.
        if (percounter_should_stop(cfg)) break;
    }
    percounter_commit_task_timing(cfg);
}

inline __aicore__ AicoreWorkWindow
run_percounter_lane(__gm__ PercounterControl *control, int block_idx, CoreType core_type) {
    // The caller invalidates the data cache after observing AICPU READY.
    // Configuration is immutable after that publication; use cached loads here.
    // Progress, dependency and exit words retain their bypass/atomic accesses.
    const bool is_aic = core_type == CoreType::AIC;
    const uint32_t type_index = is_aic ? 0U : 1U;
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
    const uint64_t storage_count = control->graph_storage_count;
    cfg.storage_address = control->graph_storage_address;
    cfg.arguments = reinterpret_cast<__gm__ DispatchPayload *>(control->arguments_address);
    cfg.tickets = scheduler_state_at<PercounterTicket>(control, control->tickets_offset);
    cfg.metadata = scheduler_state_at<PercounterTaskMetadata>(control, control->metadata_offset);
    cfg.metadata_ready = scheduler_state_at<uint64_t>(control, control->metadata_ready_offset);
    cfg.segments = scheduler_state_at<PercounterSegment>(control, control->segments_offset);
    cfg.task_claims = scheduler_state_at<PercounterTaskClaim>(control, control->task_claims_offset);
    cfg.claim_summary = scheduler_state_at<PercounterClaimSummary>(control, control->claim_summary_offset);
    cfg.sync_arrivals = scheduler_state_at<PercounterStartBarrier>(control, control->sync_arrivals_offset);
    for (uint32_t i = 0; i < PERCOUNTER_LIST_COUNT; ++i) {
        cfg.segment_begin[i] = control->segment_begin[i];
        cfg.segment_count[i] = control->segment_count[i];
    }
    cfg.order = scheduler_state_at<uint64_t>(control, control->orders_offset[type_index]);
    cfg.fanin = scheduler_state_at<PercounterFanin>(control, control->fanin_offset);
    cfg.fanin_addresses = scheduler_state_at<uint64_t>(control, control->fanin_addr_offset);
    cfg.counters = scheduler_state_at<PercounterCounter>(control, control->counters_offset);
    cfg.completion_sectors =
        scheduler_state_at<PercounterCompletionSector>(control, control->completion_sectors_offset);
    cfg.task_count = control->task_count;
    cfg.table_task_count = control->table_task_count;
    if (cfg.table_task_count != storage_count) {
        percounter_fail(cfg, SchedulerGraphResult::INVALID_TASK_COUNT);
        return {};
    }
    cfg.has_predicates = control->has_predicates != 0;
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

    const uint64_t timeout_cycles = control->scheduler_timeout_cycles;
    // Order lane setup before claims; completion and claim seeds arrive in the Host image.
    dsb((mem_dsb_t)0);
    cfg.mix_task_count = control->mix_task_count;
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
#if defined(__CPU_SIM)
    if (!percounter_bind_sim_context(context)) {
        percounter_fail(cfg, SchedulerGraphResult::INVALID_ARGUMENTS);
        return {};
    }
#endif
    if (cfg.mix_task_count != 0) {
        cfg.mix_signal_ready = scheduler_state_at<uint64_t>(control, control->mix_signal_ready_offset);
        if (!percounter_clear_mix_signal(cfg, timeout_cycles)) return {};
    }

    ChipSwimlaneAicoreLocalState profile{};
    if (level == 1 && SIMPLER_GET_DFX_FLAG(get_aicore_profiling_flag(), SIMPLER_DFX_FLAG_CHIP_SWIMLANE)) {
        cfg.profile_head = get_chip_swimlane_aicore_head();
        if (cfg.profile_head != nullptr) cfg.profile = &profile;
    }
    setup_scope.close();
    if (cfg.cluster_lane == 0) percounter_lane_loop<0>(cfg);
    else if (cfg.cluster_lane == 1) percounter_lane_loop<1>(cfg);
    else percounter_lane_loop<2>(cfg);
    percounter_commit_task_timing(cfg);
    if (cfg.profile != nullptr) percounter_flush_stream_records(&profile);
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

/**
 * A percounter AICore, from its report to the COND nonce acknowledgement.
 *
 * The same startup and retirement as the resident executor, minus the
 * scheduler bootstrap: the host already built every table, so a lane goes from
 * READY straight to the register release and into its loop.
 */
inline __aicore__ void percounter_aicore_execute(__gm__ Runtime *runtime, int block_idx, CoreType core_type) {
    __gm__ Handshake *handshake = (__gm__ Handshake *)(&runtime->dev.workers[block_idx]);
    __gm__ PercounterControl *control =
        reinterpret_cast<__gm__ PercounterControl *>(runtime->dev.scheduler_bootstrap.worker_context_base);
    percounter_observe_cache_line(handshake);

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

    // The host prefills the timeout, so it bounds this wait before the AICPU
    // has written anything.
    percounter_observe_data_cache(reinterpret_cast<__gm__ void *>(control));
    const uint64_t timeout_cycles = percounter_gm_read(control->scheduler_timeout_cycles);
    bool released = false;
    uint32_t signal = static_cast<uint32_t>(read_reg(RegId::DATA_MAIN_BASE));
    const uint64_t ready_wait_start = get_sys_cnt_aicore();
    while (handshake->aicpu_ready != SCHEDULER_RUNTIME_MODE_PERCOUNTER && signal != AICORE_EXIT_SIGNAL) {
        percounter_observe_cache_line(handshake);
        if (scheduler_watchdog_expired(ready_wait_start, get_sys_cnt_aicore(), timeout_cycles)) {
            percounter_gm_compare_exchange(
                control->lane_error, uint64_t{0}, static_cast<uint64_t>(SchedulerGraphResult::TIMEOUT)
            );
            signal = AICORE_EXIT_SIGNAL;
            break;
        }
        signal = static_cast<uint32_t>(read_reg(RegId::DATA_MAIN_BASE));
        SPIN_WAIT_HINT();
    }
    if (signal != AICORE_EXIT_SIGNAL) {
        // The AICPU published the topology before READY; re-observe the block.
        percounter_observe_data_cache(reinterpret_cast<__gm__ void *>(control));
        const uint64_t release_wait_start = get_sys_cnt_aicore();
        while (signal != AICPU_IDLE_TASK_ID && signal != AICORE_EXIT_SIGNAL) {
            if (scheduler_watchdog_expired(release_wait_start, get_sys_cnt_aicore(), timeout_cycles)) {
                percounter_gm_compare_exchange(
                    control->lane_error, uint64_t{0}, static_cast<uint64_t>(SchedulerGraphResult::TIMEOUT)
                );
                break;
            }
            signal = static_cast<uint32_t>(read_reg(RegId::DATA_MAIN_BASE));
            SPIN_WAIT_HINT();
        }
        released = signal == AICPU_IDLE_TASK_ID;
    }
    if (released) {
        write_reg(RegId::COND, AICORE_IDLE_VALUE);
        const auto work_window = run_percounter_lane(control, block_idx, core_type);
        handshake->work_start_cycle = work_window.start_cycle;
        handshake->work_end_cycle = work_window.end_cycle;
        dcci(handshake, SINGLE_CACHE_LINE, CACHELINE_OUT);
        dsb((mem_dsb_t)0);
    }
    // A lane arrives here when its share of the run is done, which can be long
    // before the run ends, or after a local watchdog ended a wait above. The
    // AICPU sends EXIT on every outcome once its supervisor returns, and the
    // supervisor has its own progress watchdog, so this wait needs none. EXIT
    // is published only after the return gate is reset, so wait for it before
    // acknowledging, and complete the preceding GM writes first.
    wait_for_aicpu_exit_signal();
    OUT_OF_ORDER_STORE_BARRIER();
    acknowledge_aicore_cond_nonce_and_wait(&runtime->dev.teardown_gates[block_idx]);
}

}  // namespace
}  // namespace simpler::percounter
