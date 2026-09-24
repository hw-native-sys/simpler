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
 * @file pmu_collector.h
 * @brief Host-side PMU buffer allocation, streaming collection, and CSV export.
 *
 * Architecture:
 * - BufferPoolManager<PmuModule>: shared split-mgmt infrastructure that polls
 *   per-thread ready queues, drains done-queue shards, and replenishes the
 *   per-core free_queues from shard-local recycled lanes.
 * - PmuCollector: collector thread shards pop full PmuBuffers from the manager
 *   and append them to shard-local temporary CSV files.
 *
 * a5 specifics: device↔host transfers go through profiling_copy.h. The
 * framework's mgmt loop mirrors the shm region per tick; per-buffer
 * payloads (PmuBuffer) are pulled on demand inside ProfilerAlgorithms.
 *
 * Lifecycle:
 *   init()                       — Allocate header + per-core states +
 *                                  PmuBuffers (pre-fills free_queues; rest
 *                                  go into the recycled pool). Calls
 *                                  set_memory_context() on the base so
 *                                  start(tf) can launch threads.
 *   start(tf)                    — Reset run-scoped CSV shards, then launch the
 *                                  mgmt + collector threads through
 *                                  ProfilerBase.
 *   [device execution]
 *   stop()                       — Stop mgmt → join mgmt → signal collectors →
 *                                  drain ready shards → join collectors, in
 *                                  that order. On return both thread exits and
 *                                  queue drains are complete.
 *   reconcile_counters()         — Merge CSV shards, sanity-check
 *                                  PmuBufferState::current_buf_ptr (any
 *                                  non-zero pointer with records is a
 *                                  device-flush bug, logged as ERROR), and
 *                                  run the device-side cross-check
 *                                  collected + dropped == total.
 *   finalize()                   — Free all device memory and unregister.
 */

#ifndef SRC_A5_PLATFORM_INCLUDE_HOST_PMU_COLLECTOR_H_
#define SRC_A5_PLATFORM_INCLUDE_HOST_PMU_COLLECTOR_H_

#include <atomic>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <optional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "common/memory_barrier.h"
#include "common/platform_config.h"
#include "common/pmu_profiling.h"
#include "common/unified_log.h"
#include "host/pmu_runs.h"
#include "host/profiler_base.h"

// ---------------------------------------------------------------------------
// PMU profiling Module (drives BufferPoolManager<PmuModule>)
// ---------------------------------------------------------------------------

/**
 * One buffer kind (PmuBuffer); per-core buffer states. The collector
 * pre-allocates PLATFORM_PMU_BUFFERS_PER_CORE buffers per core at init time
 * to absorb steady-state load. Runtime refill uses the owning drain shard's
 * local recycled lanes; proactive_replenish may batch-allocate before
 * drain and collector threads start.
 */

/**
 * Internal hand-off struct delivered from a drain thread to a collector shard.
 * thread_index is the logical AICPU thread queue the entry was popped from.
 */
struct PmuReadyBufferInfo {
    uint32_t core_index;
    uint32_t thread_index;
    void *dev_buffer_ptr;
    void *host_buffer_ptr;
    uint32_t buffer_seq;
};

struct PmuModule {
    using DataHeader = PmuDataHeader;
    using ReadyEntry = PmuReadyQueueEntry;
    using ReadyBufferInfo = ::PmuReadyBufferInfo;
    using FreeQueue = PmuFreeQueue;

    static constexpr int kBufferKinds = 1;
    static constexpr uint32_t kReadyQueueSize = PLATFORM_PMU_READYQUEUE_SIZE;
    static constexpr uint32_t kHostPoolQueueSize = PLATFORM_MAX_CORES * PLATFORM_PMU_BUFFERS_PER_CORE;
    static constexpr uint32_t kHostRecycledQueueSize = PLATFORM_MAX_CORES * PLATFORM_PMU_BUFFERS_PER_CORE;
    static constexpr uint32_t kSlotCount = PLATFORM_PMU_SLOT_COUNT;
    static constexpr const char *kSubsystemName = "PmuModule";
    // Producers are the scheduler threads that own each core, one per AICPU thread.
    static constexpr int kMaxCollectorThreads = PLATFORM_MAX_AICPU_THREADS;

    /**
     * Buffers grown by proactive_replenish are batch-allocated up to the
     * configured per-core ceiling minus the slot count, so a double-empty
     * (recycled + done both dry) recovers in one tick.
     */
    static constexpr int batch_size(int /*kind*/) {
        constexpr int kBatch = PLATFORM_PMU_BUFFERS_PER_CORE - PLATFORM_PMU_SLOT_COUNT;
        return kBatch < 1 ? 1 : kBatch;
    }

    // Each live collector shard owns ceil(cores / shard_count) cores, so the
    // watermark grows as the shard count shrinks.
    static constexpr int recycled_warm_target(int /*kind*/, int shard_count) {
        constexpr int kSurplusPerCore = (PLATFORM_PMU_BUFFERS_PER_CORE > PLATFORM_PMU_SLOT_COUNT) ?
                                            (PLATFORM_PMU_BUFFERS_PER_CORE - PLATFORM_PMU_SLOT_COUNT) :
                                            0;
        int cores_per_shard =
            shard_count > 0 ? (PLATFORM_MAX_CORES + shard_count - 1) / shard_count : PLATFORM_MAX_CORES;
        int initial_surplus = kSurplusPerCore * cores_per_shard;
        return initial_surplus > 0 ? (initial_surplus + 1) / 2 : 1;
    }

    static DataHeader *header_from_shm(void *shm) { return get_pmu_header(shm); }

    /**
     * `count` is intentionally NOT reset here — AICPU is the sole writer
     * and resets it itself when popping from free_queue.
     */
    static std::optional<profiling_common::EntrySite<PmuModule>>
    resolve_entry(void *shm, DataHeader *header, int q, const ReadyEntry &entry) {
        if (shm == nullptr || header == nullptr) {
            LOG_ERROR("PmuModule: invalid shared memory/header while resolving ready entry");
            return std::nullopt;
        }
        if (entry.core_index >= header->num_cores || entry.core_index >= static_cast<uint32_t>(PLATFORM_MAX_CORES)) {
            LOG_ERROR(
                "PmuModule: invalid ready entry core=%u (num_cores=%u, max=%u)", entry.core_index, header->num_cores,
                static_cast<uint32_t>(PLATFORM_MAX_CORES)
            );
            return std::nullopt;
        }
        PmuBufferState *state = get_pmu_buffer_state(shm, static_cast<int>(entry.core_index));
        profiling_common::EntrySite<PmuModule> site;
        site.kind = 0;
        site.free_queue = &state->free_queue;
        site.buffer_size = sizeof(PmuBuffer);
        site.info.core_index = entry.core_index;
        site.info.thread_index = static_cast<uint32_t>(q);
        site.info.dev_buffer_ptr = reinterpret_cast<void *>(entry.buffer_ptr);
        site.info.host_buffer_ptr = nullptr;  // filled by ProfilerAlgorithms
        site.info.buffer_seq = entry.buffer_seq;
        return site;
    }

    template <typename Cb>
    static void for_each_instance(void *shm, DataHeader *header, Cb &&cb) {
        const int num_cores = static_cast<int>(header->num_cores);
        for (int c = 0; c < num_cores; c++) {
            PmuBufferState *state = get_pmu_buffer_state(shm, c);
            cb(/*kind=*/0, &state->free_queue, sizeof(PmuBuffer));
        }
    }
};

// ---------------------------------------------------------------------------
// Memory operation callbacks (injected by DeviceRunner)
// ---------------------------------------------------------------------------

// Memory callbacks — thin aliases for the canonical profiling_common shapes.
// alloc / free are std::function so callers bind their MemoryAllocator via
// lambda capture; register / unregister stay as plain function pointers
// because they wrap stateless HAL globals. On a5 onboard the runner passes
// register_cb=nullptr and the framework installs a malloc-shadow + DMA
// fallback inline in ProfilerBase::start().
using PmuAllocCallback = profiling_common::ProfAllocCallback;
using PmuRegisterCallback = profiling_common::ProfRegisterCallback;
using PmuUnregisterCallback = profiling_common::ProfUnregisterCallback;
using PmuFreeCallback = profiling_common::ProfFreeCallback;

// ---------------------------------------------------------------------------
// PmuCollector
// ---------------------------------------------------------------------------

class PmuCollector : public profiling_common::ProfilerBase<PmuCollector, PmuModule> {
public:
    PmuCollector() = default;
    ~PmuCollector();

    PmuCollector(const PmuCollector &) = delete;
    PmuCollector &operator=(const PmuCollector &) = delete;

    // ProfilerBase contract
    static constexpr int kIdleTimeoutSec = PLATFORM_PMU_TIMEOUT_SECONDS;
    static constexpr const char *kSubsystemName = "PMU";

    /**
     * Allocate PMU shared memory and pre-populate per-core free_queues.
     *
     * Allocates the PmuDataHeader + per-core PmuBufferState array, plus
     * `num_cores * PLATFORM_PMU_BUFFERS_PER_CORE` PmuBuffers. The first
     * PLATFORM_PMU_SLOT_COUNT buffers per core are pushed directly into
     * that core's free_queue; the surplus go into the BufferPoolManager's
     * shard-local recycled lanes.
     *
     * @param num_cores                         Number of AICore instances in use
     * @param num_threads                       Number of AICPU scheduling threads
     * @param csv_path                          Output CSV path
     * @param event_type                        PmuEventType selector (written
     *                                          to PmuDataHeader::event_type
     *                                          so AICPU can configure HW
     *                                          counters)
     * @param alloc_cb / register_cb / free_cb  Memory operation callbacks
     *                                          (register_cb nullptr on a5)
     * @param user_data                         Opaque pointer forwarded to callbacks
     * @param device_id                         Device ID (for register_cb)
     * @return 0 on success, non-zero on failure
     */
    // Allocates the device-side resources.
    //
    // Per-run configuration (CSV destination, event selection) is bound
    // separately by begin_run(), which must run before this on the first run so
    // the event type reaches the device header and the CSV header string.
    int init(
        int num_cores, int num_threads, const PmuAllocCallback &alloc_cb, PmuRegisterCallback register_cb,
        const PmuFreeCallback &free_cb, int device_id
    );

    // Start a run's collection window: bind its CSV destination and event
    // selection, drop the previous run's shard state, rebuild the CSV header
    // (its columns are named by the event config), and — once the region exists
    // — republish the event type the device reads.
    //
    // The collector initializes once and serves every run, so this is the only
    // point at which a run's shard files, CSV columns and device event type are
    // established; init() establishes none of them.
    void begin_run(const std::string &csv_path, PmuEventType event_type);

    void start(const profiling_common::ThreadFactory &thread_factory);

    /**
     * Device pointer to the PmuDataHeader. Set kernel_args.pmu_data_base
     * to this after init() succeeds so the AICPU side can find the shared
     * memory.
     */
    void *get_pmu_shm_device_ptr() const { return shm_dev_; }

    /**
     * Device pointer to the per-core PmuAicoreRing-address table
     * (uint64_t[num_cores]). Wire into
     * `KernelArgs::aicore_pmu_ring_addrs`. Filled by the host at init.
     */
    void *get_aicore_ring_addrs_device_ptr() const { return aicore_ring_addrs_dev_; }

    /**
     * Per-buffer callback invoked by ProfilerBase's poll loop. Flushes
     * records to CSV.
     */
    void on_buffer_collected(const PmuReadyBufferInfo &info, int collector_shard);

    /**
     * After stop(), perform purely-passive accounting:
     *   - LOG_ERROR any non-zero PmuBufferState::current_buf_ptr with
     *     records (device flush should always succeed-or-bump-dropped, so
     *     a non-empty leftover indicates an AICPU flush bug — host does
     *     NOT recover, to avoid masking the bug).
     *   - Run the device-side cross-check:
     *       collected + dropped == device_total.
     * Must be called after stop(), so the AICPU-side flush has settled.
     */
    void reconcile_counters();

    /**
     * Free all device memory and unregister mappings. Idempotent.
     */
    void finalize(PmuUnregisterCallback unregister_cb, const PmuFreeCallback &free_cb);

    /**
     * @return true if init() succeeded and finalize() has not run.
     */
    bool is_initialized() const { return initialized_; }

    // -----------------------------------------------------------------------
    // Cross-run retention
    // -----------------------------------------------------------------------

    /**
     * Whether this collector may hold a run's CSV past that run's boundary.
     *
     * Latched by the runner at device init from `collect_across_runs`. Default
     * false, and with it false every path below is unreachable: the collector
     * keeps its single-run window, its in-place merge and its per-run drain,
     * byte for byte.
     */
    void configure_retained_runs(bool enabled) { retained_runs_.configure(enabled); }
    bool retains_runs() const { return retained_runs_.retains_runs(); }

    /**
     * Admit one run, freezing everything its rows and its file depend on.
     *
     * Returns false — before any kernel is submitted — when both epoch slots
     * are occupied, when the destination is owned by an open epoch or still
     * holds a failed epoch's preserved temp files, when the path exceeds the
     * allowance, or when the collector is fatal. A refusal must fail the run:
     * falling back to `begin_run` would reset a store a predecessor is still
     * publishing into.
     */
    bool run_begin(uint64_t run_epoch, const std::string &csv_path, PmuEventType event_type) {
        return retained_runs_.run_begin(run_epoch, csv_path, event_type);
    }

    /**
     * Close one run's collection window while it still holds the execution
     * claim: snapshot the device counters and live buffers the successor's
     * admission will zero, then arm the transport cut and hand the epoch to the
     * writer. Does not drain and does not merge.
     */
    void run_close(uint64_t run_epoch, bool device_execution_complete) {
        retained_runs_.run_close(run_epoch, device_execution_complete);
    }

    /**
     * Give back the slot of a run that was admitted and never launched.
     *
     * Returns false when the reference release could not be proved; the epoch
     * is then quarantined rather than released, exactly as a failed close is.
     */
    bool abandon_run(uint64_t run_epoch) { return retained_runs_.abandon_run(run_epoch); }

    /**
     * Wait for every run closed up to now to be published, then report.
     *
     * Always does both halves: a normal in-flight file is waited for, and the
     * sticky error record is reported whether or not retention is currently on
     * — a failure recorded before a collector rebuild must not vanish with it.
     */
    bool flush_retained_runs(int timeout_ms, std::string *error) { return retained_runs_.flush(timeout_ms, error); }

    /** Stop admitting and publish what is still retained. Reports nothing. */
    void finish_retained_runs() { retained_runs_.finish(); }

    /** Counts a test can assert on without reaching into collector internals. */
    simpler::dfx::pmu::RetainedRunStats retained_run_stats_for_test() const { return retained_runs_.stats(); }

    /**
     * Hold the background writer before it seals, so a case can prove that a
     * row was written while a named epoch was still open. See
     * `simpler::dfx::pmu::RetainedRuns::hold_writer_for_test`.
     */
    void hold_retained_writer_for_test(bool held) { retained_runs_.hold_writer_for_test(held); }

    /**
     * Deliver one already-filled buffer through the real routing path, on the
     * caller's thread.
     *
     * The same call `on_buffer_collected` makes, so what it exercises is
     * production: the epoch is resolved from the buffer's own `run_epoch` and
     * the rows are written with that epoch's frozen columns. It exists so a
     * case can choose the *instant* of that write, which the drain and
     * collector threads otherwise choose for it.
     */
    void deliver_buffer_for_test(const void *buf_host_ptr, int core_id, int thread_idx, int collector_shard) {
        retained_runs_.route_buffer(buf_host_ptr, core_id, thread_idx, collector_shard);
    }

    /**
     * ProfilerBase hook: adopt the epoch table before acknowledging the
     * reference-release request. Called on a collector shard thread while it
     * holds no epoch reference.
     */
    void refresh_retained_run_view(int collector_shard) { retained_runs_.refresh_view(collector_shard); }

    /**
     * ProfilerBase hook: transport progress at a transition the background
     * writer waits on. Cheap, and a no-op while no run is retained.
     */
    void note_transport_progress() { retained_runs_.note_transport_progress(); }

private:
    /**
     * The per-run configuration an epoch's rows are written with.
     *
     * Frozen at admission because the hot path would otherwise read the
     * collector's mutable members, and a successor's admission changes them:
     * a late buffer from run N would be written with N+1's event type and
     * column set.
     */
    struct FrozenRunConfig {
        PmuEventType event_type{PmuEventType::PIPE_UTILIZATION};
        const PmuEventConfig *events{nullptr};
    };

    using RetainedRunTable =
        simpler::dfx::pmu::RetainedRuns<PmuCollector, FrozenRunConfig, PmuEventType, Manager::kMaxCollectorShards>;
    // The table drives this collector through the hooks below and reaches
    // `ProfilerBase`'s cut and reference-release primitives through it.
    friend class simpler::dfx::pmu::RetainedRuns<
        PmuCollector, FrozenRunConfig, PmuEventType, Manager::kMaxCollectorShards>;

    // Hooks the retained-run table calls. Each is the arch-specific half of a
    // step whose sequencing lives in the table.
    size_t retained_shard_count() const { return static_cast<size_t>(manager_.shard_count()); }
    FrozenRunConfig freeze_run_config(PmuEventType event_type) const;
    std::string build_csv_header(const FrozenRunConfig &frozen) const;
    bool publish_run_config(PmuEventType event_type);
    simpler::dfx::pmu::RecordProofs snapshot_run_records(bool device_execution_complete) const;
    uint64_t buffer_run_epoch(const void *buf_host_ptr) const;
    uint64_t buffer_record_count(const void *buf_host_ptr) const;
    uint64_t write_buffer_rows(
        std::ofstream &out, const FrozenRunConfig &frozen, int core_id, int thread_idx, const void *buf_host_ptr,
        uint64_t buffer_epoch, bool *clamped
    );
    void install_paired_caps();
    void release_paired_caps();

    struct alignas(64) CollectorShardCounters {
        uint64_t total_collected{0};
    };
    static_assert(
        sizeof(CollectorShardCounters) % 64 == 0, "CollectorShardCounters must not share cache lines across shards"
    );

    bool initialized_ = false;
    int num_cores_ = 0;
    int num_threads_ = 0;
    PmuEventType event_type_{PmuEventType::PIPE_UTILIZATION};

    // Shared memory region (PmuDataHeader + PmuBufferState[]). shm_host_ /
    // device_id_ live on ProfilerBase (set via set_memory_context in init()).
    void *shm_dev_ = nullptr;

    // Per-core stable PmuAicoreRings + the per-core ring-address table that
    // travels through KernelArgs into AICore platform state.
    std::vector<void *> aicore_rings_dev_;
    void *aicore_ring_addrs_dev_ = nullptr;
    void *aicore_ring_addrs_host_ = nullptr;

    // CSV output. Collector shards stream rows into shard-local temp files on
    // the hot path; stop-time reconciliation merges them into the final CSV so
    // collector threads do not contend on one ofstream mutex or retain all rows
    // in heap memory.
    std::string csv_path_;
    std::string csv_header_;
    std::ofstream csv_file_;
    std::vector<std::string> csv_shard_paths_;
    std::vector<std::ofstream> csv_shard_files_;
    std::vector<CollectorShardCounters> collector_counters_;
    std::atomic<bool> csv_shard_io_failed_{false};
    bool csv_shards_finalized_{false};

    // Running total of records written to CSV. Used at reconcile time to
    // verify collected + dropped == device_total.
    uint64_t total_collected_ = 0;

    PmuDataHeader *pmu_header() const { return get_pmu_header(shm_host_); }
    PmuBufferState *pmu_state(int core_id) const { return get_pmu_buffer_state(shm_host_, core_id); }

    size_t normalize_collector_shard(int collector_shard) const;
    void reset_collector_shards();
    void append_buffer_to_csv_shard(int core_id, int thread_idx, const void *buf_host_ptr, int collector_shard);
    bool flush_collector_shards_to_csv();
    bool ensure_csv_shard_open(size_t shard);
    bool ensure_csv_open();
    bool close_csv_shards();
    void rebuild_csv_header();
    std::string build_csv_header(const PmuEventConfig *events) const;
    void cleanup_csv_shards();

    // Constructed with a reference to this collector, so it stays a member and
    // is never copied.
    RetainedRunTable retained_runs_{*this};
};

// ---------------------------------------------------------------------------
// Utility: resolve PMU event type (env-var override)
// ---------------------------------------------------------------------------

inline PmuEventType resolve_pmu_event_type(int requested_event_type) {
    PmuEventType resolved = PmuEventType::PIPE_UTILIZATION;
    if (requested_event_type > 0 &&
        pmu_resolve_event_config_a5(static_cast<PmuEventType>(requested_event_type)) != nullptr) {
        resolved = static_cast<PmuEventType>(requested_event_type);
    } else if (requested_event_type != 0) {
        LOG_WARN(
            "Invalid PMU event type %u, using default (PIPE_UTILIZATION=%u)", requested_event_type,
            PMU_EVENT_TYPE_DEFAULT
        );
    }
    const char *pmu_env = std::getenv("SIMPLER_PMU_EVENT_TYPE");
    if (pmu_env == nullptr) {
        return resolved;
    }
    int val = std::atoi(pmu_env);
    if (val > 0 && pmu_resolve_event_config_a5(static_cast<PmuEventType>(val)) != nullptr) {
        resolved = static_cast<PmuEventType>(val);
        LOG_INFO("PMU event type set to %u from SIMPLER_PMU_EVENT_TYPE", static_cast<uint32_t>(resolved));
        return resolved;
    }
    LOG_WARN("Invalid SIMPLER_PMU_EVENT_TYPE=%s, using default (PIPE_UTILIZATION=%u)", pmu_env, PMU_EVENT_TYPE_DEFAULT);
    return resolved;
}

/**
 * Build the CSV path under the caller-provided per-task directory.
 * Filename is fixed (no timestamp) — the directory is the per-task
 * uniqueness boundary.
 */
inline std::string make_pmu_csv_path(const std::string &output_dir) {
    std::error_code ec;
    std::filesystem::create_directories(output_dir, ec);
    if (ec) {
        LOG_WARN("Failed to create PMU output directory %s: %s", output_dir.c_str(), ec.message().c_str());
    }
    return output_dir + "/pmu.csv";
}

#endif  // SRC_A5_PLATFORM_INCLUDE_HOST_PMU_COLLECTOR_H_
