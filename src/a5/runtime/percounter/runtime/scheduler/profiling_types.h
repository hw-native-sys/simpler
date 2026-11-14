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

#include "scheduler/scheduler_types.h"

namespace simpler::percounter {

enum class ProfilePhase : uint8_t {
    Window,
    Resolve,
    Fanin,
    Prepare,
    Run,
    Complete,
    Checkpoint,
    MixPrepare,
    MixArm,
    MixWait,
    MixServe,
    MixDrain,
    SyncEnqueue,
    SyncPromote,
    SyncTake,
    SyncServe,
    FlushDone,
    LaneSetup,
    Backoff,
    AtomicFetchAdd,
    AtomicFetchAddIssue,
    AtomicCas,
    AtomicStoreIssue,
    BypassLoad,
    DcciOut,
    DcciSingle,
    DcciEntire,
    DsbAll,
    SsbufLoad,
    SsbufStore,
    TraceFinish,
    TraceFinishCleanEntire,
    TraceFinishBarrier,
    BypassStoreIssue,
    DsbDdr,
};

constexpr const char *profile_phase_name(ProfilePhase kind) {
    switch (kind) {
    case ProfilePhase::Window:
        return "scalar_window";
    case ProfilePhase::Resolve:
        return "scalar_percounter_resolve";
    case ProfilePhase::Fanin:
        return "scalar_percounter_fanin_ready";
    case ProfilePhase::Prepare:
        return "scalar_percounter_prepare_subtask";
    case ProfilePhase::Run:
        return "scalar_percounter_run_prepared";
    case ProfilePhase::Complete:
        return "scalar_percounter_publish_completion";
    case ProfilePhase::Checkpoint:
        return "scalar_percounter_checkpoint_serve";
    case ProfilePhase::MixPrepare:
        return "scalar_percounter_mix_prepare";
    case ProfilePhase::MixArm:
        return "scalar_percounter_mix_arm";
    case ProfilePhase::MixWait:
        return "scalar_percounter_mix_wait";
    case ProfilePhase::MixServe:
        return "scalar_percounter_serve_mix";
    case ProfilePhase::MixDrain:
        return "scalar_percounter_mix_drain";
    case ProfilePhase::SyncEnqueue:
        return "scalar_percounter_sync_enqueue";
    case ProfilePhase::SyncPromote:
        return "scalar_percounter_sync_promote";
    case ProfilePhase::SyncTake:
        return "scalar_percounter_sync_take";
    case ProfilePhase::SyncServe:
        return "scalar_percounter_sync_serve_block";
    case ProfilePhase::FlushDone:
        return "scalar_percounter_flush_done";
    case ProfilePhase::LaneSetup:
        return "scalar_run_percounter_lane";
    case ProfilePhase::Backoff:
        return "scalar_backoff";
    case ProfilePhase::AtomicFetchAdd:
        return "scalar_atomic_fetch_add";
    case ProfilePhase::AtomicFetchAddIssue:
        return "scalar_atomic_fetch_add_issue";
    case ProfilePhase::AtomicCas:
        return "scalar_atomic_cas";
    case ProfilePhase::AtomicStoreIssue:
        return "scalar_atomic_store_issue";
    case ProfilePhase::BypassLoad:
        return "scalar_bypass_load";
    case ProfilePhase::DcciOut:
        return "scalar_dcci_out";
    case ProfilePhase::DcciSingle:
        return "scalar_dcci_single";
    case ProfilePhase::DcciEntire:
        return "scalar_dcci_entire";
    case ProfilePhase::DsbAll:
        return "scalar_dsb_all";
    case ProfilePhase::SsbufLoad:
        return "scalar_ssbuf_load";
    case ProfilePhase::SsbufStore:
        return "scalar_ssbuf_store";
    case ProfilePhase::TraceFinish:
        return "scalar_trace_finish";
    case ProfilePhase::TraceFinishCleanEntire:
        return "scalar_trace_finish_clean_entire";
    case ProfilePhase::TraceFinishBarrier:
        return "scalar_trace_finish_barrier";
    case ProfilePhase::BypassStoreIssue:
        return "scalar_bypass_store_issue";
    case ProfilePhase::DsbDdr:
        return "scalar_dsb_ddr";
    }
    return "unknown";
}

struct PercounterTaskTiming {
    uint64_t dispatch;
    uint64_t start;
    uint64_t finish;
    TaskId task_token;
};

inline constexpr uint32_t PERCOUNTER_SCALAR_WORD_CAPACITY =
    SCHEDULER_SCALAR_RECORD_CAPACITY * sizeof(SchedulerScalarRecord) / sizeof(uint64_t);
inline constexpr uint64_t PERCOUNTER_SCALAR_EXTENDED = UINT64_C(1) << 63;

// One word holds a 32-bit start offset, 16-bit duration, 8-bit phase and
// 7-bit depth. Extended rows preserve the complete logical record in five words.
inline bool percounter_decode_scalar_storage(
    const uint64_t *words, uint32_t size, uint32_t &cursor, uint64_t window_start, SchedulerScalarRecord &record
) {
    if (cursor >= size) return false;
    const uint64_t head = words[cursor++];
    if ((head & PERCOUNTER_SCALAR_EXTENDED) != 0) {
        if (size - cursor < 4 || (head & ~(PERCOUNTER_SCALAR_EXTENDED | UINT64_C(0xffffffff))) != 0) return false;
        record.kind = static_cast<uint32_t>(head);
        record.start_time = words[cursor++];
        record.end_time = words[cursor++];
        record.task_ref = static_cast<int64_t>(words[cursor++]);
        const uint64_t count = words[cursor++];
        if (count > UINT32_MAX) return false;
        record.count = static_cast<uint32_t>(count);
    } else {
        const uint64_t offset = static_cast<uint32_t>(head);
        const uint64_t duration = (head >> 32) & UINT16_MAX;
        if (window_start > UINT64_MAX - offset || window_start + offset > UINT64_MAX - duration) return false;
        record.start_time = window_start + offset;
        record.end_time = record.start_time + duration;
        record.kind = static_cast<uint32_t>((head >> 48) & UINT8_MAX) | (static_cast<uint32_t>(head >> 56) << 16);
        record.task_ref = -1;
        record.count = 1;
    }
    return record.end_time >= record.start_time;
}

struct alignas(64) PercounterProfileBuffer {
    uint64_t scalar_records_offset;
    uint64_t task_records_offset;
    uint32_t task_capacity;
    uint32_t task_committed;
    uint32_t task_dropped;
    uint32_t worker_id;
    int32_t physical_core_id;
    int32_t core_type;
    int32_t cluster_index;
    int32_t cluster_lane;
    uint32_t scalar_dropped;
    uint64_t scalar_write_cycles;
    uint64_t scalar_poll_loops;
    uint64_t scalar_short_intervals;
    uint64_t scalar_function_min_cycles;
    uint64_t scalar_window_start;
    uint32_t scalar_storage_words;
};
static_assert(
    std::is_trivially_copyable_v<PercounterProfileBuffer> && std::is_standard_layout_v<PercounterProfileBuffer>
);
static_assert(std::is_trivially_copyable_v<PercounterTaskTiming> && std::is_standard_layout_v<PercounterTaskTiming>);

}  // namespace simpler::percounter
