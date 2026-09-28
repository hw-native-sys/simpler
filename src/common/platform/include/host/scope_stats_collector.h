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
 * split mgmt threads (drain polls per-thread ready queues and refills the
 * single instance's free_queue from recycled lanes; replenish returns done
 * buffers to recycled lanes). ScopeStatsCollector's collector thread shards
 * append each full buffer's ScopeStatsRecords to an in-memory vector. After
 * stop(), write_jsonl() renders them to
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
 *   start(tf)            — Inherited: launches mgmt + collector threads.
 *   [device execution]
 *   stop()               — Inherited: drain queues, join threads.
 *   reconcile_counters() — Recover any un-flushed current buffer left by an
 *                          abnormal exit, then cross-check collected ==
 *                          total - dropped.
 *   write_jsonl()        — Emit scope_stats/scope_stats.jsonl
 *                          (meta line + one record/line).
 *   finalize()           — Free all device memory, unregister.
 *
 * Output (scope_stats/scope_stats.jsonl), NDJSON:
 *   line 1: {"fatal":bool,"dropped":uint,"total":uint,
 *            "task_window_max":[...],"heap_max":[...],
 *            "dep_pool_max":[...],"tensormap_max":uint}
 *   line k: {"site":"file:line","phase":"begin|end","depth":int,
 *            "ring":int,"task_window_start":int,"task_window_end":int,
 *            "heap_start":uint,"heap_end":uint,
 *            "dep_pool_start":int,"dep_pool_end":int,
 *            "tensormap":int,"run_epoch":uint,"buf_seq":uint}
 */

#ifndef SRC_COMMON_PLATFORM_INCLUDE_HOST_SCOPE_STATS_COLLECTOR_H_
#define SRC_COMMON_PLATFORM_INCLUDE_HOST_SCOPE_STATS_COLLECTOR_H_

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "common/platform_config.h"
#include "common/scope_stats.h"
#include "common/unified_log.h"
#include "host/collected_record.h"
#include "host/profiler_base.h"
#include "host/scope_stats_runs.h"

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

/**
 * A collected scope_stats record with its run. Alias over the shared wrapper —
 * see host/collected_record.h for why identity is copied rather than referenced.
 */
using CollectedScopeStatsRecord = CollectedRecord<ScopeStatsRecord>;

struct ScopeStatsModule {
    using DataHeader = ScopeStatsDataHeader;
    using ReadyEntry = ScopeStatsReadyQueueEntry;
    using ReadyBufferInfo = ::ScopeStatsReadyBufferInfo;
    using FreeQueue = ScopeStatsFreeQueue;

    static constexpr int kBufferKinds = 1;
    static constexpr uint32_t kReadyQueueSize = PLATFORM_SCOPE_STATS_READYQUEUE_SIZE;
    static constexpr uint32_t kHostPoolQueueSize =
        PLATFORM_MAX_AICPU_THREADS * PLATFORM_SCOPE_STATS_BUFFERS_PER_INSTANCE;
    static constexpr uint32_t kSlotCount = PLATFORM_SCOPE_STATS_SLOT_COUNT;
    static constexpr const char *kSubsystemName = "ScopeStatsModule";
    // The orchestrator is the sole device-side producer (scope_stats_collector_aicpu
    // enqueues into queues[s_orch_thread_idx]), so one drain thread scanning
    // every AICPU ready queue covers it; further shards would only ever be empty.
    static constexpr int kMaxCollectorThreads = 1;

    static constexpr int batch_size(int /*kind*/) {
        constexpr int kBatch = PLATFORM_SCOPE_STATS_BUFFERS_PER_INSTANCE - PLATFORM_SCOPE_STATS_SLOT_COUNT;
        return kBatch < 1 ? 1 : kBatch;
    }

    static DataHeader *header_from_shm(void *shm) { return get_scope_stats_header(shm); }

    static std::optional<profiling_common::EntrySite<ScopeStatsModule>>
    resolve_entry(void *shm, DataHeader *header, int q, const ReadyEntry &entry) {
        if (shm == nullptr || header == nullptr) {
            LOG_ERROR("ScopeStatsModule: invalid shared memory/header while resolving ready entry");
            return std::nullopt;
        }
        if (header->num_instances != 1 || entry.instance_index >= header->num_instances) {
            LOG_ERROR(
                "ScopeStatsModule: invalid ready entry instance=%u (num_instances=%u)", entry.instance_index,
                header->num_instances
            );
            return std::nullopt;
        }
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

    // Start a run's collection window: drop the previous run's records, its
    // counter, and the recovered-buffer bookkeeping reconcile_counters() leaves
    // behind. execution_complete_ is re-armed because it is what tells the
    // collector loop a run is still producing.
    //
    // The collector initializes once and serves every run, so this is the only
    // point at which they are cleared; init() clears none of them, and left
    // alone they accumulate across runs.
    void begin_run();

    // Device pointer to the ScopeStatsDataHeader. Set
    // kernel_args.scope_stats_data_base to this after init().
    void *get_scope_stats_shm_device_ptr() const { return shm_dev_; }

    // Poll-thread hook: append the buffer's records to the in-memory vector.
    void on_buffer_collected(const ScopeStatsReadyBufferInfo &info);

    // After stop(): recover a non-empty current buffer left by abnormal exit,
    // warn on drops, and cross-check collected == total - dropped. Returns
    // true iff the run is clean.
    bool reconcile_counters();

    // Render the collected records to
    // <output_dir>/scope_stats/scope_stats.jsonl. Reads the static capacity
    // metadata + fatal latch from the shared header (constant after
    // orchestrator init). Must be called after stop().
    int write_jsonl(const std::string &output_dir);

    void finalize(ScopeStatsUnregisterCallback unregister_cb, const ScopeStatsFreeCallback &free_cb);

    bool is_initialized() const { return initialized_; }
    uint64_t total_collected() const { return total_collected_; }

    /**
     * Collected records, each carrying the run that produced it.
     *
     * Returned by value because the collector thread appends concurrently. A
     * record's `run_epoch` is the one stamped on the device buffer it was
     * copied from, so grouping by it attributes records to runs without relying
     * on the collector having been cleared between them.
     */
    std::vector<CollectedScopeStatsRecord> collected_records() const;

    /** How many collected records belong to `run_epoch`. */
    size_t collected_for_run(uint64_t run_epoch) const;

    // --- Cross-run retention -------------------------------------------------
    //
    // Off unless `configure_retained_runs(true, ...)` is called before init, in
    // which case the run boundary keeps its ownership steps — the receive drain
    // and the terminal read — and hands only the rendering and the file write
    // to a writer thread. Every entry point below is inert with retention off.

    /** Latch retention and this collector's own host byte budget. */
    void configure_retained_runs(bool retain_across_runs, size_t budget_bytes);
    bool retains_runs() const { return retain_across_runs_; }

    /**
     * Admit one run and open its export slot. False refuses the run, and the
     * caller must fail it before anything reaches the device.
     *
     * Refused when quarantined host copies from a run whose completion could
     * not be proved are still held — `begin_run()` would clear the very records
     * the quarantine protects — and when both unpublished export slots are in
     * use.
     */
    bool run_begin(uint64_t run_epoch, const std::string &output_prefix);

    /**
     * Close one run's boundary, under its execution claim.
     *
     * `device_execution_complete` is the caller's own fence observation and is
     * the whole of this collector's completion proof. Without it, and without a
     * terminal read every device copy reported success for, the run produces no
     * artifact at all: nothing shared is read, the host copies are quarantined
     * until the existing collector-thread join, and a sticky error is recorded.
     */
    void run_close(uint64_t run_epoch, bool device_execution_complete);

    /** Give an admitted run's slot back when its launch submitted nothing. */
    void abandon_run(uint64_t run_epoch);

    /** Wait for every closed run to be published. False when any run failed. */
    bool flush_retained_runs(int timeout_ms, std::string *error);

    /** Publish everything still queued. Called before the collector threads stop. */
    void finish_retained_runs();

    /** Free quarantined host copies. Only legal once the reader threads are joined. */
    void discard_quarantined_runs();

    /** Counters a test reads instead of parsing files. */
    struct RetainedRunStats {
        uint64_t published{0};
        uint64_t partial{0};
        uint64_t counts_unknown{0};
        uint64_t write_failed{0};
        uint64_t quarantined{0};
        uint64_t refused_records{0};
        uint64_t host_failures{0};
        size_t open_slots{0};
        size_t charged_bytes{0};
        bool has_error{false};
    };
    RetainedRunStats retained_run_stats_for_test() const;
    /** Make the next terminal device copy report failure. */
    void fail_terminal_copy_for_test(bool fail) { fail_terminal_copy_ = fail; }
    /** Hold the writer before it publishes, so a test can occupy export slots. */
    void pause_writer_for_test(bool paused);
    /** Throw from the collector thread's append, where nothing may escape. */
    void throw_in_collector_for_test(bool fail) { throw_in_collector_ = fail; }
    /** Throw from the writer thread's publish, where nothing may escape. */
    void throw_in_writer_for_test(bool fail) { throw_in_writer_ = fail; }
    /** Fail the writer-thread construction, as a real `std::thread` can. */
    void fail_writer_start_for_test(bool fail) { fail_writer_start_ = fail; }
    /** Throw at the handoff, where a run's records have already left the store. */
    void fail_handoff_for_test(bool fail) { fail_handoff_ = fail; }

private:
    bool initialized_ = false;

    // Shared memory region (ScopeStatsDataHeader + ScopeStatsBufferState).
    // shm_host_ / shm_size_ / device_id_ live on ProfilerBase (set via
    // set_memory_context in init()).
    void *shm_dev_ = nullptr;

    simpler::dfx::scope_stats_runs::RecordBlocks records_;
    mutable std::mutex records_mutex_;
    uint64_t total_collected_ = 0;
    uint64_t refused_records_ = 0;
    uint64_t recovered_current_buf_ = 0;
    uint64_t recovered_current_total_ = 0;

    ScopeStatsDataHeader *scope_stats_header() const { return get_scope_stats_header(shm_host_); }
    ScopeStatsBufferState *scope_stats_state(int idx = 0) const { return get_scope_stats_buffer_state(shm_host_, idx); }

    void append_buffer_records(const void *buf_host_ptr);

    /**
     * Render one artifact's bytes. `extra` adds the background-mode metadata
     * keys; a null `extra` reproduces today's metadata line exactly, which is
     * what keeps the default path's file unchanged.
     */
    static int render_jsonl_to(
        std::FILE *fp, const simpler::dfx::scope_stats_runs::DeviceSnapshot &device,
        const simpler::dfx::scope_stats_runs::RecordBlocks &records,
        const simpler::dfx::scope_stats_runs::Collection *extra
    );

    /** Today's unchecked read of the shared header, for the default path only. */
    simpler::dfx::scope_stats_runs::DeviceSnapshot snapshot_unchecked() const;

    /**
     * Charge one record block, or refuse it.
     *
     * Always true with retention off: the default path keeps today's unbounded
     * in-memory accumulation, and only a retained run is charged.
     */
    bool charge_record_block(size_t bytes);

    /**
     * Record a host-side failure that belongs to the collector rather than to
     * one run, and keep it until the runner is destroyed.
     *
     * Allocates nothing, so it is safe on the paths that call it precisely
     * because an allocation has just failed.
     */
    void note_host_failure(const char *detail) noexcept;

    // --- Cross-run retention state ------------------------------------------

    // A slot is held from admission until its artifact is published, so
    // "two unpublished exports" is a bound on what the collector is holding,
    // not merely on what is still filling.
    enum class SlotState : int { Free = 0, Open = 1, Publishing = 2, Quarantined = 3 };

    struct Slot {
        SlotState state{SlotState::Free};
        uint64_t run_epoch{0};
        std::string output_dir;
    };

    /**
     * Copy the shared region and this run's terminal into `out`, reporting
     * whether every device copy it needed succeeded.
     *
     * A false return leaves `out.valid` false. The fields a failed copy may
     * have left behind are never published: the caller quarantines instead.
     */
    bool read_terminal_checked(simpler::dfx::scope_stats_runs::DeviceSnapshot *out);

    /**
     * Take the producer's unpublished current buffer, if it left one.
     *
     * `current_buf_ptr != 0 && count != 0` is device-written evidence that the
     * buffer was never enqueued: publication clears the pointer immediately
     * after a successful enqueue, and the end-of-run flush zeroes the count
     * when its own enqueue fails. So no de-duplication bookkeeping is needed,
     * and a published buffer is never recovered twice.
     */
    bool recover_unpublished_buffer_checked();

    /** `run_close`'s body, wrapped by the boundary's exception guard. */
    void run_close_locked_path(uint64_t run_epoch, bool device_execution_complete);

    void quarantine_locked(uint64_t run_epoch, const char *detail);
    void writer_loop();
    void ensure_writer_started();
    void stop_writer();
    int publish_export(const simpler::dfx::scope_stats_runs::RunExport &data);

    bool retain_across_runs_{false};
    size_t retained_budget_bytes_{0};
    bool retained_ready_{false};
    bool fail_terminal_copy_{false};
    bool throw_in_collector_{false};
    bool throw_in_writer_{false};
    bool fail_writer_start_{false};
    bool fail_handoff_{false};
    simpler::dfx::runs::HostBudget host_budget_;
    simpler::dfx::runs::ErrorSummary run_errors_;

    mutable std::mutex retained_mu_;
    std::condition_variable retained_cv_;
    Slot slots_[simpler::dfx::runs::kMaxOpenEpochs];
    std::deque<simpler::dfx::scope_stats_runs::RunExport> write_queue_;
    simpler::dfx::scope_stats_runs::RecordBlocks quarantined_records_;
    bool quarantine_held_{false};
    bool writing_{false};
    bool writer_running_{false};
    bool writer_paused_{false};
    std::thread writer_;
    RetainedRunStats stats_{};
};

#endif  // SRC_COMMON_PLATFORM_INCLUDE_HOST_SCOPE_STATS_COLLECTOR_H_
