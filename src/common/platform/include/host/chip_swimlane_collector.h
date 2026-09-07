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
 * @file chip_swimlane_collector.h
 * @brief Platform-agnostic performance data collector with dynamic memory management.
 *
 * Architecture:
 * - BufferPoolManager<ChipSwimlaneModule>: shared mgmt-thread infrastructure that polls
 *   the AICPU ready queue, replenishes per-core / per-thread free queues, and
 *   hands full buffers off to collector thread shards.
 * - ChipSwimlaneCollector: collector thread shards copy records from manager ready queues
 *   into host vectors; the owner thread exports the swimlane visualization after stop().
 *
 * Memory operations are injected through callbacks for sim/onboard portability.
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "common/chip_swimlane_extension.h"
#include "common/chip_swimlane_profiling.h"
#include "host/clock_correlation.h"
#include "common/memory_barrier.h"
#include "common/platform_config.h"
#include "common/unified_log.h"
#include "host/profiler_base.h"

// ---------------------------------------------------------------------------
// L2 Perf profiling Module (drives BufferPoolManager<ChipSwimlaneModule>)
// ---------------------------------------------------------------------------

/**
 * L2 Perf has four distinct buffer kinds going through one ready queue per
 * AICPU thread:
 *   - kind 0: per-core    ChipSwimlaneAicpuTaskBuffer      (task records)
 *   - kind 1: per-thread  ChipSwimlaneAicpuSchedPhaseBuffer (scheduler phase records)
 *   - kind 2: per-thread  ChipSwimlaneAicpuOrchPhaseBuffer  (orchestrator phase records)
 *   - kind 3: per-core    ChipSwimlaneAicoreTaskBuffer     (AICore-written records)
 * The ReadyQueueEntry::kind flag picks among them.
 */

/**
 * Buffer kind discriminator carried in ReadyBufferInfo and used to index the
 * per-kind recycled pool inside BufferPoolManager. Values match
 * ChipSwimlaneBufferKind 1:1.
 */
enum class ProfBufferType {
    AICPU_TASK = 0,
    AICPU_SCHED_PHASE = 1,
    AICPU_ORCH_PHASE = 2,
    AICORE_TASK = 3,
};

/**
 * Information about a ready (full) buffer, passed from mgmt thread to main thread.
 */
struct ReadyBufferInfo {
    ProfBufferType type;
    uint32_t index;         // core_index (task) or thread_idx (phase)
    uint32_t slot_idx;      // Reserved (unused in free queue design)
    void *dev_buffer_ptr;   // Device address of the full buffer
    void *host_buffer_ptr;  // Host-mapped address (sim: same as dev)
    uint32_t buffer_seq;    // Sequence number for ordering
};

struct ChipSwimlaneModule {
    using DataHeader = ChipSwimlaneDataHeader;
    using ReadyEntry = ReadyQueueEntry;
    using ReadyBufferInfo = ::ReadyBufferInfo;
    using FreeQueue = ChipSwimlaneFreeQueue;  // all pool types share the same free_queue layout

    static constexpr int kBufferKinds = 4;
    static constexpr uint32_t kReadyQueueSize = PLATFORM_PROF_READYQUEUE_SIZE;
    static constexpr uint32_t kHostPoolQueueSize =
        PLATFORM_MAX_CORES * PLATFORM_PROF_BUFFERS_PER_CORE +
        PLATFORM_MAX_AICPU_THREADS * (PLATFORM_PROF_SCHED_BUFFERS_PER_THREAD + PLATFORM_PROF_ORCH_BUFFERS_PER_THREAD) +
        PLATFORM_MAX_CORES * PLATFORM_AICORE_BUFFERS_PER_CORE;
    static constexpr uint32_t kAicpuTaskRecycledQueueSize = PLATFORM_MAX_CORES * PLATFORM_PROF_BUFFERS_PER_CORE;
    static constexpr uint32_t kAicoreTaskRecycledQueueSize = PLATFORM_MAX_CORES * PLATFORM_AICORE_BUFFERS_PER_CORE;
    static constexpr uint32_t kPhaseRecycledQueueSize =
        (PLATFORM_PROF_SCHED_BUFFERS_PER_THREAD > PLATFORM_PROF_ORCH_BUFFERS_PER_THREAD ?
             PLATFORM_PROF_SCHED_BUFFERS_PER_THREAD :
             PLATFORM_PROF_ORCH_BUFFERS_PER_THREAD) *
        2;
    static constexpr uint32_t kHostRecycledQueueSize =
        (kAicpuTaskRecycledQueueSize > kAicoreTaskRecycledQueueSize ?
             (kAicpuTaskRecycledQueueSize > kPhaseRecycledQueueSize ? kAicpuTaskRecycledQueueSize :
                                                                      kPhaseRecycledQueueSize) :
             (kAicoreTaskRecycledQueueSize > kPhaseRecycledQueueSize ? kAicoreTaskRecycledQueueSize :
                                                                       kPhaseRecycledQueueSize));
    static constexpr uint32_t kSlotCount = PLATFORM_PROF_SLOT_COUNT;
    static constexpr const char *kSubsystemName = "ChipSwimlaneModule";
    // Producers are the scheduler threads (task / sched-phase records) plus the
    // orchestrator (orch-phase records) — one per AICPU thread.
    static constexpr int kMaxCollectorThreads = PLATFORM_MAX_AICPU_THREADS;

    /**
     * Startup-only batch allocation size for proactive_replenish. Sched and
     * orch phase pools are sized independently
     * (PLATFORM_PROF_{SCHED,ORCH}_BUFFERS_PER_THREAD).
     */
    static constexpr int batch_size(int kind) {
        constexpr int kPerfBatch = PLATFORM_PROF_BUFFERS_PER_CORE - PLATFORM_PROF_SLOT_COUNT;
        constexpr int kSchedBatch = PLATFORM_PROF_SCHED_BUFFERS_PER_THREAD - PLATFORM_PROF_SLOT_COUNT;
        constexpr int kOrchBatch = PLATFORM_PROF_ORCH_BUFFERS_PER_THREAD - PLATFORM_PROF_SLOT_COUNT;
        constexpr int kAicoreBatch = PLATFORM_AICORE_BUFFERS_PER_CORE - PLATFORM_PROF_SLOT_COUNT;
        int b = kPerfBatch;
        switch (static_cast<ChipSwimlaneBufferKind>(kind)) {
        case ChipSwimlaneBufferKind::AicpuTask:
            b = kPerfBatch;
            break;
        case ChipSwimlaneBufferKind::AicpuSchedPhase:
            b = kSchedBatch;
            break;
        case ChipSwimlaneBufferKind::AicpuOrchPhase:
            b = kOrchBatch;
            break;
        case ChipSwimlaneBufferKind::AicoreTask:
            b = kAicoreBatch;
            break;
        }
        return b < 1 ? 1 : b;
    }

    // The recycled watermark is a steady-state low-water mark, not an
    // additional startup preallocation target. Keep half of the init-seeded
    // surplus per shard; kinds with no surplus keep a minimal reserve.
    // Cores are spread across the live collector shards, so each shard owns
    // ceil(cores / shard_count) of them. The watermark must therefore grow as
    // the shard count shrinks — sizing it against the platform's max thread
    // count instead would under-provision a run with fewer AICPU threads.
    static constexpr int cores_per_shard(int shard_count) {
        return shard_count > 0 ? (PLATFORM_MAX_CORES + shard_count - 1) / shard_count : PLATFORM_MAX_CORES;
    }

    static constexpr int half_initial_surplus_warm_target(int buffers_per_core, int shard_count) {
        int surplus_per_core = buffers_per_core > static_cast<int>(PLATFORM_PROF_SLOT_COUNT) ?
                                   buffers_per_core - static_cast<int>(PLATFORM_PROF_SLOT_COUNT) :
                                   0;
        int initial_surplus = surplus_per_core * cores_per_shard(shard_count);
        return initial_surplus > 0 ? (initial_surplus + 1) / 2 : 1;
    }

    static constexpr int recycled_warm_target(int kind, int shard_count) {
        switch (static_cast<ChipSwimlaneBufferKind>(kind)) {
        case ChipSwimlaneBufferKind::AicpuTask:
            return half_initial_surplus_warm_target(PLATFORM_PROF_BUFFERS_PER_CORE, shard_count);
        case ChipSwimlaneBufferKind::AicoreTask:
            return half_initial_surplus_warm_target(PLATFORM_AICORE_BUFFERS_PER_CORE, shard_count);
        case ChipSwimlaneBufferKind::AicpuSchedPhase:
        case ChipSwimlaneBufferKind::AicpuOrchPhase:
            return 0;
        }
        return 0;
    }

    static int kind_of(const ReadyBufferInfo &info) { return static_cast<int>(info.type); }

    static DataHeader *header_from_shm(void *shm) { return get_chip_swimlane_header(shm); }

    template <typename Mgr>
    static void refresh_replenish_metadata(Mgr &mgr, DataHeader *header) {
        mgr.read_range_from_device(&header->num_sched_phase_threads, sizeof(header->num_sched_phase_threads));
        mgr.read_range_from_device(&header->num_orch_phase_threads, sizeof(header->num_orch_phase_threads));
        rmb();
    }

    /**
     * Branch on entry.kind to pick the per-core task state, per-thread sched-
     * or orch-phase state, or per-core AICore state. Returns nullopt for
     * out-of-range kind or core_index.
     */
    static std::optional<profiling_common::EntrySite<ChipSwimlaneModule>>
    resolve_entry(void *shm, DataHeader *header, int /*q*/, const ReadyEntry &entry) {
        const int num_cores = static_cast<int>(header->num_cores);
        const ChipSwimlaneBufferKind kind = entry.kind;

        // Validate kind first — out-of-range silently falling into the wrong
        // branch reads a wrong-typed pool.
        if (kind != ChipSwimlaneBufferKind::AicpuTask && kind != ChipSwimlaneBufferKind::AicpuSchedPhase &&
            kind != ChipSwimlaneBufferKind::AicpuOrchPhase && kind != ChipSwimlaneBufferKind::AicoreTask) {
            LOG_ERROR("ChipSwimlaneModule: invalid entry kind=%u", static_cast<uint32_t>(kind));
            return std::nullopt;
        }

        // Sched/orch phase entries are indexed by thread_idx; task/aicore by core_index.
        const bool is_phase =
            (kind == ChipSwimlaneBufferKind::AicpuSchedPhase) || (kind == ChipSwimlaneBufferKind::AicpuOrchPhase);
        if (is_phase) {
            if (entry.core_index >= static_cast<uint32_t>(PLATFORM_MAX_AICPU_THREADS)) {
                LOG_ERROR("ChipSwimlaneModule: invalid phase entry: thread=%u", entry.core_index);
                return std::nullopt;
            }
        } else {
            if (entry.core_index >= static_cast<uint32_t>(num_cores)) {
                LOG_ERROR(
                    "ChipSwimlaneModule: invalid task entry: core=%u kind=%u", entry.core_index,
                    static_cast<uint32_t>(kind)
                );
                return std::nullopt;
            }
        }

        profiling_common::EntrySite<ChipSwimlaneModule> site;
        site.kind = static_cast<int>(kind);
        site.info.index = entry.core_index;
        site.info.slot_idx = 0;
        site.info.dev_buffer_ptr = reinterpret_cast<void *>(entry.buffer_ptr);
        site.info.host_buffer_ptr = nullptr;  // filled by ProfilerAlgorithms
        site.info.buffer_seq = entry.buffer_seq;

        switch (kind) {
        case ChipSwimlaneBufferKind::AicpuTask: {
            auto *state = get_perf_buffer_state(shm, static_cast<int>(entry.core_index));
            site.free_queue = &state->free_queue;
            site.buffer_size = sizeof(ChipSwimlaneAicpuTaskBuffer);
            site.info.type = ProfBufferType::AICPU_TASK;
            break;
        }
        case ChipSwimlaneBufferKind::AicpuSchedPhase: {
            auto *state = get_sched_phase_buffer_state(shm, static_cast<int>(entry.core_index));
            site.free_queue = &state->free_queue;
            site.buffer_size = sizeof(ChipSwimlaneAicpuSchedPhaseBuffer);
            site.info.type = ProfBufferType::AICPU_SCHED_PHASE;
            break;
        }
        case ChipSwimlaneBufferKind::AicpuOrchPhase: {
            auto *state = get_orch_phase_buffer_state(shm, static_cast<int>(entry.core_index));
            site.free_queue = &state->free_queue;
            site.buffer_size = sizeof(ChipSwimlaneAicpuOrchPhaseBuffer);
            site.info.type = ProfBufferType::AICPU_ORCH_PHASE;
            break;
        }
        case ChipSwimlaneBufferKind::AicoreTask: {
            auto *ac_state = get_aicore_buffer_state(shm, static_cast<int>(entry.core_index));
            site.free_queue = &ac_state->free_queue;
            site.buffer_size = sizeof(ChipSwimlaneAicoreTaskBuffer);
            site.info.type = ProfBufferType::AICORE_TASK;
            break;
        }
        }
        return site;
    }

    template <typename Cb>
    static void for_each_instance(void *shm, DataHeader *header, Cb &&cb) {
        const int num_cores = static_cast<int>(header->num_cores);

        // AicpuTask: per-core (kind 0)
        for (int i = 0; i < num_cores; i++) {
            auto *state = get_perf_buffer_state(shm, i);
            cb(/*kind=*/static_cast<int>(ChipSwimlaneBufferKind::AicpuTask), &state->free_queue,
               sizeof(ChipSwimlaneAicpuTaskBuffer));
        }

        // AicoreTask: per-core (kind 3)
        for (int i = 0; i < num_cores; i++) {
            auto *ac_state = get_aicore_buffer_state(shm, i);
            cb(/*kind=*/static_cast<int>(ChipSwimlaneBufferKind::AicoreTask), &ac_state->free_queue,
               sizeof(ChipSwimlaneAicoreTaskBuffer));
        }

        // AicpuSchedPhase: per-thread (kind 1) — gated on the header's
        // sched-phase thread count (zero when phase init never ran).
        // Bounds-clamp against PLATFORM_MAX_AICPU_THREADS so a corrupted
        // device-shared value can't walk off the pool array.
        int num_sched_phase_threads = static_cast<int>(header->num_sched_phase_threads);
        if (num_sched_phase_threads > PLATFORM_MAX_AICPU_THREADS) {
            num_sched_phase_threads = 0;
        }
        for (int t = 0; t < num_sched_phase_threads; t++) {
            auto *state = get_sched_phase_buffer_state(shm, t);
            cb(/*kind=*/static_cast<int>(ChipSwimlaneBufferKind::AicpuSchedPhase), &state->free_queue,
               sizeof(ChipSwimlaneAicpuSchedPhaseBuffer));
        }

        // AicpuOrchPhase: per-thread (kind 2) — same bounds clamp.
        int num_orch_phase_threads = static_cast<int>(header->num_orch_phase_threads);
        if (num_orch_phase_threads > PLATFORM_MAX_AICPU_THREADS) {
            num_orch_phase_threads = 0;
        }
        for (int t = 0; t < num_orch_phase_threads; t++) {
            auto *state = get_orch_phase_buffer_state(shm, t);
            cb(/*kind=*/static_cast<int>(ChipSwimlaneBufferKind::AicpuOrchPhase), &state->free_queue,
               sizeof(ChipSwimlaneAicpuOrchPhaseBuffer));
        }
    }
};

// Memory callbacks — thin aliases for the canonical profiling_common shapes.
// alloc / free are std::function so callers bind their MemoryAllocator via
// lambda capture; register / unregister stay as plain function pointers
// because they wrap stateless HAL globals (halHost*).
using ChipSwimlaneAllocCallback = profiling_common::ProfAllocCallback;
using ChipSwimlaneRegisterCallback = profiling_common::ProfRegisterCallback;
using ChipSwimlaneUnregisterCallback = profiling_common::ProfUnregisterCallback;
using ChipSwimlaneFreeCallback = profiling_common::ProfFreeCallback;

// =============================================================================
// ChipSwimlaneCollector
// =============================================================================

/**
 * Performance data collector.
 *
 * Lifecycle:
 *   1. initialize()                — allocate shared memory, pre-fill free_queues,
 *                                    hand the memory context to the base via
 *                                    set_memory_context().
 *   2. start(tf)                   — inherited from ProfilerBase: assembles a
 *                                    MemoryOps from the stashed callbacks and
 *                                    launches the mgmt + poll threads.
 *   3. ... device execution ...
 *   4. stop()                      — joins both threads in the correct order
 *                                    (mgmt first so its final-drain entries
 *                                    have a consumer).
 *   5. read_phase_header_metadata() — single-shot read of the core→thread
 *                                    mapping from ChipSwimlaneDataHeader.
 *   6. reconcile_counters()        — device-side three-bucket accounting for
 *                                    both PERF and PHASE pools (total /
 *                                    collected / dropped).
 *   7. export_swimlane_json() / finalize().
 *
 * Host never reads from device-side `current_buf_ptr` to recover records:
 * device flush is the only data path. Any non-zero `current_buf_ptr` after
 * stop() is logged as a bug.
 */
class ChipSwimlaneCollector : public profiling_common::ProfilerBase<ChipSwimlaneCollector, ChipSwimlaneModule> {
public:
    ChipSwimlaneCollector() = default;
    ~ChipSwimlaneCollector();

    ChipSwimlaneCollector(const ChipSwimlaneCollector &) = delete;
    ChipSwimlaneCollector &operator=(const ChipSwimlaneCollector &) = delete;

    // ProfilerBase contract
    static constexpr int kIdleTimeoutSec = PLATFORM_PROF_TIMEOUT_SECONDS;
    static constexpr const char *kSubsystemName = "ChipSwimlane";

    /**
     * Initialize performance profiling.
     *
     * Allocates the shared-memory region (header + per-core / per-thread
     * BufferStates), pre-allocates initial ChipSwimlaneAicpuTaskBuffers and PhaseBuffers,
     * and seeds the per-pool free_queues + the framework's recycled pools.
     *
     * @param num_aicore               Number of AICore instances
     * @param device_id                Device ID (forwarded to register_cb)
     * @param chip_swimlane_level   Collection granularity (DISABLED / TASK_TIMING
     *                                 / SCHEDULE_TIMING / SCHED_PHASES / ORCH_PHASES).
     *                                 Written into
     *                                 `ChipSwimlaneDataHeader::chip_swimlane_level`
     *                                 so AICPU can promote it in
     *                                 `chip_swimlane_aicpu_init`, AND cached on the
     *                                 collector so `export_swimlane_json()`
     *                                 can gate phase sections and stamp the
     *                                 JSON `version`.
     * @param alloc_cb                 Device memory allocation callback
     * @param register_cb              Memory registration callback (nullptr for
     *                                 simulation and non-SVM platforms)
     * @param free_cb                  Device memory free callback
     * @param user_data                Opaque pointer forwarded to callbacks
     * @param output_prefix            Per-task directory; chip_swimlane_records.json
     *                                 lands here. Required (non-empty);
     *                                 CallConfig::validate() enforces this
     *                                 upstream.
     * @return 0 on success, error code on failure
     */
    // Allocates the device-side resources.
    //
    // The pool-array offsets are fixed at compile time from PLATFORM_MAX_CORES,
    // so the host and AICPU sides cannot disagree about them. num_aicore and
    // aicpu_thread_num decide which of those fixed slots get buffers, and a
    // core's recycled lane is assigned modulo aicpu_thread_num — so a collector
    // that outlives a run holds pools shaped for those two counts, and the
    // caller rebuilds it when a later run changes them.
    //
    // Per-run configuration (artifact prefix, level) is bound separately by
    // begin_run(), which must run before this on the first run: the level
    // selects the orch phase pool here.
    int initialize(
        int num_aicore, int aicpu_thread_num, int device_id, const ChipSwimlaneAllocCallback &alloc_cb,
        ChipSwimlaneRegisterCallback register_cb, const ChipSwimlaneFreeCallback &free_cb
    );

    /**
     * Start a run's collection window: bind its artifact configuration, drop the
     * previous run's records, counters, and runtime extensions, and — once the
     * region exists — republish the level the device reads.
     *
     * The collector initializes once and serves every run, so this is the only
     * point at which a run's records, counters and device level are established;
     * initialize() establishes none of them. Skip it and this run's artifact
     * carries the records every earlier run collected, reconcile compares an
     * accumulated collected count against one run's device total, and the device
     * stays on whichever level the first run asked for.
     *
     * Before the first initialize() there is no region and no shard storage yet;
     * reset_collector_shards() is then a no-op over empty extents, and
     * initialize() picks the level up from the member this sets.
     */
    void begin_run(const std::string &output_prefix, ChipSwimlaneLevel chip_swimlane_level) {
        output_prefix_ = output_prefix;
        chip_swimlane_level_ = chip_swimlane_level;
        json_extensions_.fill({});
        reset_collector_shards();
        publish_run_config();
    }

    bool set_json_extension(ChipSwimlaneExtensionSection section, const std::string &json_value);

    /**
     * Per-buffer callback invoked by ProfilerBase's poll loop. Dispatches on
     * info.type to copy either an ChipSwimlaneAicpuTaskBuffer (PERF_RECORD) into the per-core
     * record vector, or a ChipSwimlaneAicpuSchedPhaseBuffer / ChipSwimlaneAicpuOrchPhaseBuffer into the per-thread
     * phase-record vector.
     */
    void on_buffer_collected(const ReadyBufferInfo &info, int collector_shard);

    /**
     * Publish per-core core_type (AIC/AIV/...) so the host emit path can
     * resolve the lane label without consulting an AICPU task record. Required
     * for TASK_TIMING (level=1) where complete_task is bypassed and the
     * AICore record alone is on disk. Caller is the device_runner — sim sets
     * it from `runtime.workers[i].core_type` (rule-based), onboard sets it
     * from the handshake-discovered table.
     *
     * Safe to call multiple times; the last call wins.
     *
     * @param types  CoreType[n] table indexed by core_id
     * @param n      table length (typically `num_aicore`)
     */
    void set_core_types(const CoreType *types, int n);

    /**
     * Whether this run's orchestrator phases come from a host orchestrator.
     *
     * Known when the runner arms the host phase pool, which is during bind and
     * therefore before initialize() — early enough for the device orch-phase
     * pool to be left unallocated, which is the point. The records themselves
     * arrive later, via set_host_phase_records().
     */
    void set_host_orchestrated(bool host_orchestrated) noexcept { host_orchestrated_ = host_orchestrated; }

    /**
     * Supply this run's host phase records, projected to the ones the swimlane
     * places against device timestamps.
     *
     * The records are the platform runner's, not this collector's: their other
     * reader is enabled independently. The runner hands them over before
     * export_swimlane_json(), which is also after initialize() — unlike the
     * records themselves, which a host-orchestrating runtime writes during bind,
     * before the device collector is provisioned.
     *
     * @param submit_records   records whose kind submits a task, in order
     * @param upload_records   records whose kind is a host-to-device transfer, in
     *                         order; host work the device waits on, so it belongs
     *                         beside the device lanes
     * @param submitted_tasks  what the producer reported submitting, for the
     *                         completeness check
     * @param total_records    every record the producer attempted, of any kind
     * @param dropped_records  records the pool could not store
     */
    void set_host_phase_records(
        std::vector<HostPhaseRecord> submit_records, std::vector<HostPhaseRecord> upload_records,
        uint64_t submitted_tasks, uint64_t total_records, uint64_t dropped_records
    );
    void begin_clock_correlation_session(const char *provider_name, const char *raw_device_timestamp_unit);
    void record_clock_anchor_samples(std::vector<simpler::dfx::ClockAnchorSample> samples);
    void finish_clock_correlation_session();
    bool clock_correlation_active() const { return clock_correlation_session_.active(); }

    /**
     * Export collected records as a Chrome Trace Event JSON (swimlane view).
     * Writes <output_prefix>/chip_swimlane_records.json — directory is captured at
     * initialize() time.
     *
     * @return 0 on success, error code on failure
     */
    int export_swimlane_json();

    /**
     * Free all device memory and unregister mappings. Idempotent on a
     * collector that was never initialized.
     *
     * @param unregister_cb  Memory unregister callback (nullptr in sim mode)
     * @param free_cb        Memory free callback
     * @param user_data      Opaque pointer forwarded to callbacks
     * @return 0 on success, error code on failure
     */
    int finalize(ChipSwimlaneUnregisterCallback unregister_cb, const ChipSwimlaneFreeCallback &free_cb);

    /**
     * @return true if initialize() succeeded and finalize() has not run.
     */
    bool is_initialized() const { return shm_host_ != nullptr; }

    /**
     * Device pointer to the ChipSwimlaneDataHeader. Set kernel_args.chip_swimlane_data_base
     * to this after initialize() succeeds so the AICPU side can find the
     * shared memory.
     */
    void *get_chip_swimlane_setup_device_ptr() const { return perf_shared_mem_dev_; }

    /**
     * Device pointer to a uint64_t[num_aicore] table where each entry will
     * hold this core's `&ChipSwimlaneAicoreTaskPool::rotation` device address. Host
     * only allocates the bytes here; AICPU populates the entries inside
     * `chip_swimlane_aicpu_init`. Freed by finalize(). Set kernel_args.chip_swimlane_aicore_rotation_table
     * to this so the AICore kernel entry can index by block_idx and feed the
     * per-core rotation channel into `set_chip_swimlane_aicore_head_slot()`. Returns
     * nullptr before initialize() succeeds.
     */
    void *get_aicore_ring_addr_table_device_ptr() const { return aicore_ring_addr_table_dev_; }

    /**
     * Read AICPU phase metadata that lives in ChipSwimlaneDataHeader (not on the
     * buffer pipeline): the core→thread mapping plus a has-data signal
     * derived from accumulated per-event records. Single-shot — must be
     * called after stop() so the shm region has settled.
     */
    void read_phase_header_metadata();

    // Push the run's level into the device-visible header. A no-op before the
    // region exists, and a single narrow field write once it does.
    void publish_run_config();

    /**
     * Sum per-core / per-thread total_record_count and dropped_record_count
     * for both the PERF and PHASE pools, cross-check
     * `collected + dropped == device_total`, and LOG_ERROR any non-zero
     * current_buf_ptr (which would indicate a device-side flush failure that
     * left a buffer un-enqueued — see .claude/rules/discipline.md).
     * The PHASE block is skipped silently when no phase activity was
     * recorded (runtimes that don't emit phase records). Must be called
     * after stop().
     */
    void reconcile_counters();

    /**
     * @return Per-core ChipSwimlaneAicpuTaskRecord vectors (indexed by core_index). For tests.
     */
    const std::vector<std::vector<ChipSwimlaneAicpuTaskRecord>> &get_records() const { return collected_perf_records_; }

private:
    struct alignas(64) CollectorShardCounters {
        uint64_t total_perf_collected{0};
        uint64_t total_sched_phase_collected{0};
        uint64_t total_orch_phase_collected{0};
        bool has_phase_data{false};
    };
    static_assert(
        sizeof(CollectorShardCounters) % 64 == 0, "CollectorShardCounters must not share cache lines across shards"
    );

    template <typename T>
    using RecordsByInstance = std::vector<std::vector<T>>;
    template <typename T>
    using RecordsByCollector = std::vector<RecordsByInstance<T>>;

    // Shared memory pointers. shm_host_ / device_id_ live on ProfilerBase
    // (set via set_memory_context in initialize()).
    void *perf_shared_mem_dev_{nullptr};

    // Standalone uint64_t[num_aicore] table holding per-core ChipSwimlaneAicoreTaskBuffer
    // addresses. Allocated in initialize(), freed in finalize(). AICore reads
    // ring_table[block_idx] via KernelArgs::chip_swimlane_aicore_rotation_table.
    void *aicore_ring_addr_table_dev_{nullptr};

    int num_aicore_{0};
    // Total AICPU threads launched this run. The dedicated orchestrator runs on
    // the last one (aicpu_thread_num_ - 1); used to report its thread number in
    // the phase-metadata log (orch-phase is a single pool, so its index alone
    // does not encode the AICPU thread).
    int aicpu_thread_num_{0};
    ChipSwimlaneLevel chip_swimlane_level_{ChipSwimlaneLevel::DISABLED};

    // Per-core core_type table populated by set_core_types(). Indexed by
    // core_id; size matches num_aicore_ once populated. Used by the level=1
    // emit path which has no AICPU record to read core_type from.
    std::vector<CoreType> core_types_;

    // Per-task output directory captured at initialize() time. Consumed by
    // export_swimlane_json() to build <prefix>/chip_swimlane_records.json.
    std::string output_prefix_;
    std::array<std::string, static_cast<size_t>(ChipSwimlaneExtensionSection::Count)> json_extensions_{};

    // Merged data, populated from per-collector shards after collector threads join.
    std::vector<std::vector<ChipSwimlaneAicpuTaskRecord>> collected_perf_records_;

    // Collected AICore records (per-core vectors). Each entry is a full
    // ChipSwimlaneAicoreTaskRecord captured from a rotated ChipSwimlaneAicoreTaskBuffer.
    std::vector<std::vector<ChipSwimlaneAicoreTaskRecord>> collected_aicore_records_;

    // AICPU phase profiling data — separate per-thread vectors for sched and
    // orch records (kind-tagged at routing time; no parse-time discrimination).
    std::vector<std::vector<ChipSwimlaneAicpuSchedPhaseRecord>> collected_sched_phase_records_;
    std::vector<std::vector<ChipSwimlaneAicpuOrchPhaseRecord>> collected_orch_phase_records_;
    std::vector<HostPhaseRecord> host_submit_records_;
    std::vector<HostPhaseRecord> host_upload_records_;
    simpler::dfx::ClockCorrelationSession clock_correlation_session_;

    // Core-to-thread mapping (core_id → scheduler thread index, -1 = unassigned)
    std::vector<int8_t> core_to_thread_;

    RecordsByCollector<ChipSwimlaneAicpuTaskRecord> perf_records_by_collector_;
    RecordsByCollector<ChipSwimlaneAicoreTaskRecord> aicore_records_by_collector_;
    RecordsByCollector<ChipSwimlaneAicpuSchedPhaseRecord> sched_phase_records_by_collector_;
    RecordsByCollector<ChipSwimlaneAicpuOrchPhaseRecord> orch_phase_records_by_collector_;
    std::vector<CollectorShardCounters> collector_counters_;

    // Running totals used at reconcile time to cross-check device-side counters.
    uint64_t total_perf_collected_{0};
    uint64_t total_sched_phase_collected_{0};
    uint64_t total_orch_phase_collected_{0};
    bool has_phase_data_{false};
    bool collector_shards_merged_{false};
    // Set once the runner has handed over a pass's host phase records, which is
    // also what makes the host orchestrator this run's record source.
    bool host_orchestrated_{false};
    bool host_phase_records_present_{false};
    uint64_t host_phase_total_records_{0};
    uint64_t host_phase_dropped_records_{0};
    uint64_t host_phase_submitted_tasks_{0};

    size_t normalize_collector_shard(int collector_shard) const;
    void reset_collector_shards();
    void merge_collector_shards();

    // Per-buffer-kind handlers used by on_buffer_collected.
    void copy_perf_buffer(const ReadyBufferInfo &info, int collector_shard);
    void copy_sched_phase_buffer(const ReadyBufferInfo &info, int collector_shard);
    void copy_orch_phase_buffer(const ReadyBufferInfo &info, int collector_shard);
    void copy_aicore_buffer(const ReadyBufferInfo &info, int collector_shard);
};
