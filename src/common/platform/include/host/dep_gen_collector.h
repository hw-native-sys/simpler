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
 * @file dep_gen_collector.h
 * @brief Host-side dep_gen (SubmitTrace) buffer allocation and streaming
 *        collection for in-memory replay.
 *
 * Architecture:
 * - BufferPoolManager<DepGenModule>: shared mgmt-thread infrastructure that
 *   polls per-thread ready queues, drains done-queue shards, and replenishes
 *   the single instance's free_queue from shard-local recycled lanes.
 * - DepGenCollector: collector thread shards pop full DepGenBuffers from the
 *   manager and append their DepGenRecords to in-memory storage consumed by
 *   host replay after device execution completes, grouped by the run that
 *   produced them.
 *
 * Lifecycle:
 *   init()                       — Allocate header + 1 BufferState + N DepGenBuffers
 *                                  (pre-fills free_queue; surplus → recycled pool).
 *                                  Calls set_memory_context() on the base.
 *   start(tf)                    — Inherited: launches mgmt + collector threads.
 *   [device execution]
 *   stop()                       — Inherited: drain queues, join threads.
 *   reconcile_counters()         — Sanity-check that no buffer the device still
 *                                  holds has records in it, run the
 *                                  collected+dropped==total cross-check. A
 *                                  buffer AICPU could not hand over stays the
 *                                  pool's with count 0, which is not a failure;
 *                                  records left in one are. If
 *                                  dropped_record_count > 0, the host caller
 *                                  skips deps.json emission (incomplete graph;
 *                                  user gets a warning).
 *   finalize()                   — Free all device memory, unregister.
 *
 * Output contract: per run, a contiguous in-memory stream of DepGenRecord
 * values. Host replay consumes one run's stream directly; no submit_trace.bin
 * intermediary is written by the collector.
 */

#ifndef SRC_COMMON_PLATFORM_INCLUDE_HOST_DEP_GEN_COLLECTOR_H_
#define SRC_COMMON_PLATFORM_INCLUDE_HOST_DEP_GEN_COLLECTOR_H_

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#include "common/dep_gen.h"
#include "common/platform_config.h"
#include "common/unified_log.h"
#include "host/dep_gen_runs.h"
#include "host/profiler_base.h"

// ---------------------------------------------------------------------------
// dep_gen Module (drives BufferPoolManager<DepGenModule>)
// ---------------------------------------------------------------------------

/**
 * Internal hand-off struct delivered from a drain thread to a collector shard.
 * thread_index identifies the AICPU thread queue the entry was popped from
 * (always equal to the orchestrator thread index, since dep_gen is single-
 * instance — exposed for symmetry with PmuReadyBufferInfo).
 */
struct DepGenReadyBufferInfo {
    uint32_t instance_index;  // Always 0 (single instance)
    uint32_t thread_index;    // AICPU thread queue index this entry came from
    void *dev_buffer_ptr;
    void *host_buffer_ptr;
    uint32_t buffer_seq;
};

struct DepGenModule {
    using DataHeader = DepGenDataHeader;
    using ReadyEntry = DepGenReadyQueueEntry;
    using ReadyBufferInfo = ::DepGenReadyBufferInfo;
    using FreeQueue = DepGenFreeQueue;

    static constexpr int kBufferKinds = 1;
    static constexpr uint32_t kReadyQueueSize = PLATFORM_DEP_GEN_READYQUEUE_SIZE;
    static constexpr uint32_t kHostPoolQueueSize = PLATFORM_MAX_AICPU_THREADS * PLATFORM_DEP_GEN_BUFFERS_PER_INSTANCE;
    static constexpr uint32_t kSlotCount = PLATFORM_DEP_GEN_SLOT_COUNT;
    static constexpr const char *kSubsystemName = "DepGenModule";
    // The orchestrator is the sole device-side producer (dep_gen_collector_aicpu
    // enqueues into queues[s_orch_thread_idx]), so one drain thread scanning
    // every AICPU ready queue covers it; further shards would only ever be empty.
    static constexpr int kMaxCollectorThreads = 1;

    /**
     * Startup-only batch allocation size for proactive_replenish when the
     * recycled lanes need additional buffers before drain threads start.
     */
    static constexpr int batch_size(int /*kind*/) {
        constexpr int kBatch = PLATFORM_DEP_GEN_BUFFERS_PER_INSTANCE - PLATFORM_DEP_GEN_SLOT_COUNT;
        return kBatch < 1 ? 1 : kBatch;
    }

    static DataHeader *header_from_shm(void *shm) { return get_dep_gen_header(shm); }

    /**
     * `count` is intentionally NOT reset here — AICPU is the sole writer and
     * resets it itself on flush/drop/pop.
     */
    static std::optional<profiling_common::EntrySite<DepGenModule>>
    resolve_entry(void *shm, DataHeader *header, int q, const ReadyEntry &entry) {
        if (shm == nullptr || header == nullptr) {
            LOG_ERROR("DepGenModule: invalid shared memory/header while resolving ready entry");
            return std::nullopt;
        }
        if (header->num_instances != 1 || entry.instance_index >= header->num_instances) {
            LOG_ERROR(
                "DepGenModule: invalid ready entry instance=%u (num_instances=%u)", entry.instance_index,
                header->num_instances
            );
            return std::nullopt;
        }
        DepGenBufferState *state = get_dep_gen_buffer_state(shm, static_cast<int>(entry.instance_index));
        profiling_common::EntrySite<DepGenModule> site;
        site.kind = 0;
        site.free_queue = &state->free_queue;
        site.buffer_size = sizeof(DepGenBuffer);
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
            DepGenBufferState *state = get_dep_gen_buffer_state(shm, i);
            cb(/*kind=*/0, &state->free_queue, sizeof(DepGenBuffer));
        }
    }
};

// ---------------------------------------------------------------------------
// Memory callbacks — thin aliases for the canonical profiling_common shapes.
// alloc / free are std::function so callers bind their MemoryAllocator via
// lambda capture; register / unregister stay as plain function pointers
// because they wrap stateless HAL globals (halHost*).
// ---------------------------------------------------------------------------

using DepGenAllocCallback = profiling_common::ProfAllocCallback;
using DepGenRegisterCallback = profiling_common::ProfRegisterCallback;
using DepGenUnregisterCallback = profiling_common::ProfUnregisterCallback;
using DepGenFreeCallback = profiling_common::ProfFreeCallback;

// ---------------------------------------------------------------------------
// DepGenCollector
// ---------------------------------------------------------------------------

class DepGenCollector : public profiling_common::ProfilerBase<DepGenCollector, DepGenModule> {
public:
    DepGenCollector() = default;
    ~DepGenCollector();

    DepGenCollector(const DepGenCollector &) = delete;
    DepGenCollector &operator=(const DepGenCollector &) = delete;

    static constexpr int kIdleTimeoutSec = PLATFORM_DEP_GEN_TIMEOUT_SECONDS;
    static constexpr const char *kSubsystemName = "DepGen";

    /**
     * Allocate dep_gen shared memory and pre-populate the free_queue.
     *
     * Allocates a DepGenDataHeader + 1 DepGenBufferState, plus
     * PLATFORM_DEP_GEN_BUFFERS_PER_INSTANCE DepGenBuffers. The first
     * PLATFORM_DEP_GEN_SLOT_COUNT buffers go directly into the free_queue;
     * the surplus go into BufferPoolManager's shard-local recycled lanes.
     *
     * @param num_threads     Number of AICPU scheduling threads (so the
     *                        DataHeader sizes its per-thread ready queues)
     * @param alloc_cb        Memory allocation callback
     * @param register_cb     halHostRegister callback (nullptr on non-SVM platforms)
     * @param free_cb         Memory free callback
     * @param device_id       Device ID
     * @return 0 on success, non-zero on failure
     */
    int init(
        int num_threads, const DepGenAllocCallback &alloc_cb, DepGenRegisterCallback register_cb,
        const DepGenFreeCallback &free_cb, int device_id
    );

    /**
     * Start a run's collection window. Clears what the previous run left in the
     * in-memory record set and its counter, and zeroes the device-side record
     * counters reconcile compares against.
     *
     * The collector initializes once and serves every run, so this is the only
     * point at which they are cleared; init() clears none of them, and nothing on
     * the device clears the record counters — they are documented as monotonic.
     * Skip it and the second run's deps.json carries the first run's edges, and
     * reconcile compares one run's collected count against a device total
     * accumulated over both, which suppresses the export.
     *
     * Called with the device quiesced, so the AICPU is not writing these and the
     * collector threads are idle.
     */
    bool begin_run();

    /**
     * Report what this run's boundary actually established about its transport,
     * separately from whether the counts happened to balance.
     *
     * `reconcile_counters()` answers one `bool` that conflates "the comparison
     * balanced" with "the comparison was made": a failed region copy leaves the
     * host shadow holding the zeroes `begin_run()` wrote, which satisfies the
     * identity for a run that collected nothing. Every distinguishable reason a
     * graph is untrustworthy is a field here, and both modes refuse on the same
     * set — they differ only in which channel reports it.
     *
     * Call after the receive drain, with the device quiesced.
     */
    simpler::dfx::dep_gen_runs::ReconcileReport reconcile_report();

    /**
     * Device pointer to the DepGenDataHeader. Set kernel_args.dep_gen_data_base
     * to this after init() so AICPU can find the shared memory via
     * set_platform_dep_gen_base().
     */
    void *get_dep_gen_shm_device_ptr() const { return shm_dev_; }

    /**
     * Per-buffer callback invoked by ProfilerBase's poll loop. Appends the
     * buffer's DepGenRecord entries to in-memory storage, grouped by the run
     * that produced them (no disk I/O — the host replay consumes one run's
     * records directly via ``window_records()`` once the device run
     * completes).
     */
    void on_buffer_collected(const DepGenReadyBufferInfo &info);

    /**
     * After stop(): cross-check collected + dropped == total. If dropped > 0,
     * the host caller skips deps.json emission so users get an incomplete-
     * graph warning rather than partial data they might mistake for complete.
     *
     * @return true iff the run captured a complete trace (no drops, no leftovers).
     */
    bool reconcile_counters();

    /**
     * Free all device memory and release the in-memory record buffer. Idempotent.
     */
    void finalize(DepGenUnregisterCallback unregister_cb, const DepGenFreeCallback &free_cb);

    /**
     * @return true if init() succeeded and finalize() has not run.
     */
    bool is_initialized() const { return initialized_; }

    /**
     * Total DepGenRecords drained from the device-side ring buffer so far.
     */
    uint64_t total_collected() const { return total_collected_; }

    /**
     * Every run whose records this collector holds, keyed by run epoch, each
     * run's records contiguous and in arrival order.
     *
     * Records are grouped rather than flattened because a buffer's identity is
     * the only thing that says which graph its records belong to, and the pool
     * hands the same storage to a later run. Today the collection window holds
     * exactly one run — ``begin_run()`` clears this — so there is exactly one
     * entry; the grouping is what lets that stop being true without silently
     * merging two graphs.
     */
    const std::map<uint64_t, std::vector<DepGenRecord>> &runs() const { return records_by_run_; }

    /**
     * The records this collection window should emit as its graph.
     *
     * Three cases, and the distinction matters because a window with no records
     * is not the same as a window whose contents are ambiguous:
     *
     *   - one run  — that run's records, and its epoch through @p run_epoch_out.
     *   - no run   — an **empty** span, epoch 0. A run that submitted nothing
     *                has an empty graph, not a missing one, and the replay
     *                writer accepts `num_records == 0` to emit exactly that.
     *   - several  — nullptr. One path names one graph, so the caller cannot
     *                emit; silently picking one would produce a deps.json that
     *                claims to be the whole graph.
     *
     * A run present in ``runs()`` always has at least one record, because
     * ``append_buffer_records`` never creates an entry for an empty buffer — so
     * the one-run case can never masquerade as the no-run one.
     *
     * Valid between init() and finalize(); pointer/size stay stable after
     * stop() returns, which is when the caller hands them to
     * ``dep_gen_replay_emit_deps_json``.
     */
    // --- Cross-run retention -------------------------------------------------
    //
    // Off unless `configure_retained_runs(true, ...)` runs before init, in
    // which case the boundary keeps every step that touches the device — the
    // receive drain and the terminal read — and hands the replay, the
    // serialization and the file write to one background writer. Every entry
    // point below is inert with retention off.

    /** Latch retention and this collector's own host byte budget. */
    void configure_retained_runs(bool retain_across_runs, size_t budget_bytes);
    bool retains_runs() const { return retain_across_runs_; }

    /**
     * Admit one run, taking an export slot and recording the identity and the
     * destination the writer will publish under.
     *
     * False refuses the run, and the caller must fail it before anything
     * reaches the device: both unpublished slots are in use, the prefix does
     * not fit the path allowance, the budget cannot open, or this run's device
     * counter reset was not published. A refusal leaves no slot taken.
     */
    bool run_begin(uint64_t run_epoch, const std::string &output_prefix);

    /**
     * Close one run's boundary, under its execution claim.
     *
     * `device_execution_complete` is the caller's own fence observation and the
     * whole of this collector's completion proof. Without it the run reads
     * nothing shared, seals nothing and publishes nothing: the host copies are
     * quarantined until the existing collector-thread join, and a sticky error
     * is recorded.
     */
    void run_close(uint64_t run_epoch, bool device_execution_complete);

    /** Give an admitted run's slot back when its launch submitted nothing. */
    void abandon_run(uint64_t run_epoch);

    /** Wait for every closed run to be published. False when any run failed. */
    bool flush_retained_runs(int timeout_ms, std::string *error);

    /** Publish everything still queued. Runs before the collector threads stop. */
    void finish_retained_runs();

    /** Free quarantined host copies. Only legal once the reader threads are joined. */
    void discard_quarantined_runs();

    /** Counters a test reads instead of parsing files. */
    struct RetainedRunStats {
        uint64_t published{0};
        uint64_t refused{0};
        uint64_t quarantined{0};
        uint64_t host_failures{0};
        uint64_t refused_records{0};
        uint64_t foreign_epoch_records{0};
        size_t open_slots{0};
        size_t charged_bytes{0};
        bool has_error{false};
    };
    RetainedRunStats retained_run_stats_for_test() const;
    /** Hold the writer before it publishes, so a test can occupy export slots. */
    void pause_writer_for_test(bool paused);
    /** Make the next shared-region read report failure. */
    void fail_region_read_for_test(bool fail) { fail_region_read_ = fail; }
    /** Make this run's counter-reset publication report failure. */
    void fail_counter_reset_for_test(bool fail) { fail_counter_reset_ = fail; }
    /** Shrink the budget to the figure a test needs a charge to be refused at. */
    void shrink_budget_for_test(size_t budget_bytes) { retained_budget_bytes_ = budget_bytes; }

    const std::vector<DepGenRecord> *window_records(uint64_t *run_epoch_out = nullptr) const {
        if (records_by_run_.size() > 1) return nullptr;
        if (records_by_run_.empty()) {
            if (run_epoch_out != nullptr) *run_epoch_out = 0;
            return &kNoRecords;
        }
        const auto &entry = *records_by_run_.begin();
        if (run_epoch_out != nullptr) *run_epoch_out = entry.first;
        return &entry.second;
    }

private:
    bool initialized_ = false;
    int num_threads_ = 0;

    // Shared memory region (DepGenDataHeader + DepGenBufferState[1]).
    // shm_host_ / device_id_ live on ProfilerBase (set via set_memory_context
    // in init()).
    void *shm_dev_ = nullptr;
    size_t shm_size_ = 0;

    // In-memory records — drained from the device ring on
    // on_buffer_collected() and consumed by the host replay directly (no disk
    // hop), grouped by the run that produced them. Mutex serializes the mgmt
    // thread's appends against the (rare) reader on the same collector
    // instance.
    std::map<uint64_t, std::vector<DepGenRecord>> records_by_run_;
    std::mutex records_mutex_;

    // Returned by window_records() when no run produced anything, so the caller
    // gets an empty span rather than having to special-case a null.
    static inline const std::vector<DepGenRecord> kNoRecords{};

    // Running total of records appended across every run held here. Equal to
    // the summed sizes in ``records_by_run_`` after every append; kept
    // separately for the reconcile_counters cross-check even when the records
    // may be inspected concurrently.
    uint64_t total_collected_ = 0;

    DepGenDataHeader *dep_gen_header() const { return get_dep_gen_header(shm_host_); }
    DepGenBufferState *dep_gen_state(int idx = 0) const { return get_dep_gen_buffer_state(shm_host_, idx); }

    void append_buffer_records(const void *buf_host_ptr);

    // --- Cross-run retention state -------------------------------------------

    // A slot is held from admission until its artifact exists, so "two
    // unpublished exports" bounds what the collector owns rather than what is
    // merely still filling. The run being collected holds one of the two.
    enum class SlotState : int { Free = 0, Open = 1, Publishing = 2, Quarantined = 3 };

    struct Slot {
        SlotState state{SlotState::Free};
        uint64_t run_epoch{0};
        std::string output_dir;
    };

    /** Charge one record block, or refuse it. Always true with retention off. */
    bool charge_record_block(size_t bytes);
    /** Give a charged figure back. A no-op with retention off. */
    void credit_record_block(size_t bytes);

    /**
     * Record a host failure that belongs to the collector rather than to one
     * run, and keep it until the runner is destroyed.
     *
     * Allocates nothing, so it is safe on the paths that reach it precisely
     * because an allocation has just failed.
     */
    void note_host_failure(const char *detail) noexcept;

    void run_close_locked_path(uint64_t run_epoch, bool device_execution_complete);
    void quarantine_locked(uint64_t run_epoch, const char *detail);
    /** Publish one export: materialize, replay, then link the result into place. */
    int publish_export(simpler::dfx::dep_gen_runs::RunExport &data);
    void writer_loop();
    void ensure_writer_started();
    void stop_writer();
    /** True while this epoch may still stamp records the collector accepts. */
    bool epoch_admitted_locked(uint64_t run_epoch) const;

    bool retain_across_runs_ = false;
    size_t retained_budget_bytes_ = simpler::dfx::runs::kDefaultBudgetBytes;
    bool retained_ready_ = false;
    bool quarantine_held_ = false;
    bool fail_region_read_ = false;
    bool fail_counter_reset_ = false;

    // Two locks, one order. `records_mutex_` owns everything the collector
    // threads write — the retained record store, the open epoch and the
    // receive-side counters — because the receive path is the hot one and must
    // never wait on the writer. `retained_mu_` owns the slots, the export
    // queue, the writer state and the stats. A path needing both takes
    // `retained_mu_` first, and `append_buffer_records` needs only the first.
    simpler::dfx::dep_gen_runs::RecordBlocks retained_records_;
    uint64_t retained_epoch_ = 0;
    bool retained_epoch_open_ = false;
    uint64_t refused_records_ = 0;
    uint64_t foreign_epoch_records_ = 0;
    bool host_clamped_ = false;
    bool reset_unpublished_ = false;

    mutable std::mutex retained_mu_;
    std::condition_variable retained_cv_;
    Slot slots_[simpler::dfx::runs::kMaxOpenEpochs];
    std::deque<simpler::dfx::dep_gen_runs::RunExport> queue_;
    std::vector<simpler::dfx::dep_gen_runs::RunExport> quarantined_;
    simpler::dfx::runs::HostBudget host_budget_;
    simpler::dfx::runs::ErrorSummary run_errors_;
    std::thread writer_;
    bool writer_running_ = false;
    bool writer_stop_ = false;
    bool writer_paused_ = false;
    bool writer_busy_ = false;

    struct RetainedStats {
        uint64_t published{0};
        uint64_t refused{0};
        uint64_t quarantined{0};
        uint64_t host_failures{0};
    };
    RetainedStats stats_;
};

/**
 * Build the ``deps.json`` output path under the caller-provided per-task
 * directory. Filename is fixed (no timestamp) — the directory is the
 * per-task uniqueness boundary, mirroring make_pmu_csv_path().
 */
inline std::string make_deps_json_path(const std::string &output_dir) {
    std::filesystem::path dir(output_dir);
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) {
        LOG_WARN("Failed to create dep_gen output directory %s: %s", output_dir.c_str(), ec.message().c_str());
    }
    return (dir / "deps.json").string();
}

#endif  // SRC_COMMON_PLATFORM_INCLUDE_HOST_DEP_GEN_COLLECTOR_H_
