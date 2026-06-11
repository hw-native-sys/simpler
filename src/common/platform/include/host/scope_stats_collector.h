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
 * @file scope_stats_collector.h
 * @brief Host-side scope_stats streaming collector + NDJSON export.
 *
 * Architecture mirrors PmuCollector: BufferPoolManager<ScopeStatsModule> runs
 * the mgmt thread (polls the per-thread ready queue, recycles buffers, refills
 * the single instance's free_queue); ScopeStatsCollector's poll thread appends
 * each full buffer's ScopeStatsRecords to an in-memory vector. After stop(),
 * write_jsonl() renders them to
 * <output_dir>/scope_stats/scope_stats.jsonl.
 *
 * Memory mirroring is handled by the framework via the MemoryOps installed
 * at set_memory_context time:
 *   - SVM platforms (a2a3): no copy_* callbacks installed; mirror_/copy_*
 *     short-circuit to no-ops, host writes go directly to device memory.
 *   - Non-SVM platforms (a5): profiling_copy_* installed; the framework's
 *     mgmt loop mirrors the shm region per tick; per-buffer payloads
 *     (ScopeStatsBuffer) are pulled on demand inside ProfilerAlgorithms.
 *
 * Lifecycle:
 *   init()               — Allocate header + 1 BufferState + N ScopeStatsBuffers
 *                          (pre-fills free_queue; surplus → recycled pool).
 *   start(tf)            — Inherited: launches mgmt + poll threads.
 *   [device execution]
 *   stop()               — Inherited: drain queues, join threads.
 *   reconcile_counters() — current_buf_ptr cleared by flush + collected ==
 *                          total - dropped cross-check.
 *   write_jsonl()        — Emit scope_stats/scope_stats.jsonl
 *                          (meta line + one record/line).
 *   finalize()           — Free all device memory, unregister.
 *
 * Output (scope_stats/scope_stats.jsonl), NDJSON:
 *   line 1: {"version":6,"fatal":bool,"dropped":uint,"total":uint,
 *            "task_window_max":[...],"heap_max":[...],
 *            "dep_pool_max":[...],"tensormap_max":uint}
 *   line k: {"site":"file:line","phase":"begin|end","depth":int,
 *            "ring":int,"task_window_start":int,"task_window_end":int,
 *            "heap_start":uint,"heap_end":uint,
 *            "dep_pool_start":int,"dep_pool_end":int,
 *            "tensormap":int}
 */

#ifndef SRC_COMMON_PLATFORM_INCLUDE_HOST_SCOPE_STATS_COLLECTOR_H_
#define SRC_COMMON_PLATFORM_INCLUDE_HOST_SCOPE_STATS_COLLECTOR_H_

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "common/platform_config.h"
#include "common/scope_stats.h"
#include "common/unified_log.h"
#include "host/profiler_base.h"

// ---------------------------------------------------------------------------
// scope_stats Module (drives BufferPoolManager<ScopeStatsModule>)
// ---------------------------------------------------------------------------

struct ScopeStatsReadyBufferInfo {
    uint32_t instance_index;  // Always 0 (single instance)
    uint32_t thread_index;    // AICPU thread queue index this entry came from
    void *dev_buffer_ptr;
    void *host_buffer_ptr;
    uint32_t buffer_seq;
};

struct ScopeStatsModule {
    using DataHeader = ScopeStatsDataHeader;
    using ReadyEntry = ScopeStatsReadyQueueEntry;
    using ReadyBufferInfo = ::ScopeStatsReadyBufferInfo;
    using FreeQueue = ScopeStatsFreeQueue;

    static constexpr int kBufferKinds = 1;
    static constexpr uint32_t kReadyQueueSize = PLATFORM_SCOPE_STATS_READYQUEUE_SIZE;
    static constexpr uint32_t kSlotCount = PLATFORM_SCOPE_STATS_SLOT_COUNT;
    static constexpr const char *kSubsystemName = "ScopeStatsModule";

    static constexpr int batch_size(int /*kind*/) {
        constexpr int kBatch = PLATFORM_SCOPE_STATS_BUFFERS_PER_INSTANCE - PLATFORM_SCOPE_STATS_SLOT_COUNT;
        return kBatch < 1 ? 1 : kBatch;
    }

    static DataHeader *header_from_shm(void *shm) { return get_scope_stats_header(shm); }

    static std::optional<profiling_common::EntrySite<ScopeStatsModule>>
    resolve_entry(void *shm, DataHeader * /*header*/, int q, const ReadyEntry &entry) {
        ScopeStatsBufferState *state = get_scope_stats_buffer_state(shm, static_cast<int>(entry.instance_index));
        profiling_common::EntrySite<ScopeStatsModule> site;
        site.kind = 0;
        site.free_queue = &state->free_queue;
        site.buffer_size = sizeof(ScopeStatsBuffer);
        site.info.instance_index = entry.instance_index;
        site.info.thread_index = static_cast<uint32_t>(q);
        site.info.dev_buffer_ptr = reinterpret_cast<void *>(entry.buffer_ptr);
        site.info.host_buffer_ptr = nullptr;  // filled by ProfilerAlgorithms
        site.info.buffer_seq = entry.buffer_seq;
        return site;
    }

    template <typename Cb>
    static void for_each_instance(void *shm, DataHeader *header, Cb &&cb) {
        const int n = static_cast<int>(header->num_instances);
        for (int i = 0; i < n; i++) {
            ScopeStatsBufferState *state = get_scope_stats_buffer_state(shm, i);
            cb(/*kind=*/0, &state->free_queue, sizeof(ScopeStatsBuffer));
        }
    }
};

using ScopeStatsAllocCallback = profiling_common::ProfAllocCallback;
using ScopeStatsRegisterCallback = profiling_common::ProfRegisterCallback;
using ScopeStatsUnregisterCallback = profiling_common::ProfUnregisterCallback;
using ScopeStatsFreeCallback = profiling_common::ProfFreeCallback;

// ---------------------------------------------------------------------------
// ScopeStatsCollector
// ---------------------------------------------------------------------------

class ScopeStatsCollector : public profiling_common::ProfilerBase<ScopeStatsCollector, ScopeStatsModule> {
public:
    ScopeStatsCollector() = default;
    ~ScopeStatsCollector();

    ScopeStatsCollector(const ScopeStatsCollector &) = delete;
    ScopeStatsCollector &operator=(const ScopeStatsCollector &) = delete;

    static constexpr int kIdleTimeoutSec = PLATFORM_SCOPE_STATS_TIMEOUT_SECONDS;
    static constexpr const char *kSubsystemName = "ScopeStats";

    int init(
        int num_threads, const ScopeStatsAllocCallback &alloc_cb, ScopeStatsRegisterCallback register_cb,
        const ScopeStatsFreeCallback &free_cb, int device_id
    );

    // Device pointer to the ScopeStatsDataHeader. Set
    // kernel_args.scope_stats_data_base to this after init().
    void *get_scope_stats_shm_device_ptr() const { return shm_dev_; }

    // Poll-thread hook: append the buffer's records to the in-memory vector.
    void on_buffer_collected(const ScopeStatsReadyBufferInfo &info);

    // After stop(): warn on drops / un-flushed buffer and cross-check
    // collected == total - dropped. Returns true iff the run is clean.
    bool reconcile_counters();

    // Render the collected records to
    // <output_dir>/scope_stats/scope_stats.jsonl. Reads the static capacity
    // metadata + fatal latch from the shared header (constant after
    // orchestrator init). Must be called after stop().
    int write_jsonl(const std::string &output_dir);

    void finalize(ScopeStatsUnregisterCallback unregister_cb, const ScopeStatsFreeCallback &free_cb);

    bool is_initialized() const { return initialized_; }
    uint64_t total_collected() const { return total_collected_; }

private:
    bool initialized_ = false;
    int num_threads_ = 0;

    // Shared memory region (ScopeStatsDataHeader + ScopeStatsBufferState).
    // shm_host_ / shm_size_ / device_id_ live on ProfilerBase (set via
    // set_memory_context in init()).
    void *shm_dev_ = nullptr;

    std::vector<ScopeStatsRecord> records_;
    std::mutex records_mutex_;
    uint64_t total_collected_ = 0;

    ScopeStatsDataHeader *scope_stats_header() const { return get_scope_stats_header(shm_host_); }
    ScopeStatsBufferState *scope_stats_state(int idx = 0) const { return get_scope_stats_buffer_state(shm_host_, idx); }

    void append_buffer_records(const void *buf_host_ptr);
};

#endif  // SRC_COMMON_PLATFORM_INCLUDE_HOST_SCOPE_STATS_COLLECTOR_H_
