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
#include "host/collected_record.h"
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
 *
 * The underlying type is stated rather than left implicit, so the
 * representation this shares with the device-side kind is visible at the
 * declaration. A value outside the four enumerators is representable either
 * way; the host counts such a buffer as unroutable instead of indexing a
 * per-class array with it.
 */
enum class ProfBufferType : uint32_t {
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
 * device flush is the only data path. A non-zero `current_buf_ptr` after stop()
 * means the pool still owns that buffer, which is legitimate — a run with
 * nothing to publish, or one whose enqueue failed, keeps it for the next run's
 * init to reuse in place. Only a retained buffer whose `count` is non-zero is a
 * bug: those records were neither delivered nor charged to `dropped`.
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
    // The level is taken here because it selects the orch phase pool, and the
    // pools are built once for every run this collector serves. The rest of a
    // run's configuration is bound by begin_run(), which runs once per run and
    // may run either side of this.
    int initialize(
        int num_aicore, int aicpu_thread_num, int device_id, ChipSwimlaneLevel chip_swimlane_level,
        const ChipSwimlaneAllocCallback &alloc_cb, ChipSwimlaneRegisterCallback register_cb,
        const ChipSwimlaneFreeCallback &free_cb
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
     * reset_collector_shards() is then a no-op over empty extents and
     * publish_run_config() has no header to write.
     */
    void begin_run(const std::string &output_prefix, ChipSwimlaneLevel chip_swimlane_level) {
        output_prefix_ = output_prefix;
        chip_swimlane_level_ = chip_swimlane_level;
        json_extensions_.fill({});
        // The previous run's live figures are not this run's; a comparison must
        // report unknown until this run's reconcile has produced its own.
        live_counters_ = LiveTaskCounters{};
        aicore_accounting_ = AicoreAccounting{};
        terminal_reported_ = false;
        terminal_snapshot_ = RunTerminalSnapshot{};
        terminal_consistency_ = RunTerminalConsistency{};
        handoff_report_ = HandoffReport{};
        transport_retired_buffers_ = 0;
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
     * Per-shard AICore records as collected, each with the run it came from.
     * Exposed for the identity/ownership tests, which need the pre-merge view:
     * the merge into `collected_aicore_records_` only runs at reconcile.
     */
    const std::vector<std::vector<CollectedRecord<ChipSwimlaneAicoreTaskRecord>>> &
    collected_aicore_records_for_test() const {
        return aicore_records_by_collector_[0];
    }

    /** Per-shard AICPU task records as collected, each with its run. */
    const std::vector<std::vector<CollectedRecord<ChipSwimlaneAicpuTaskRecord>>> &
    collected_perf_records_for_test() const {
        return perf_records_by_collector_[0];
    }

    /**
     * This run's AICore accounting as `reconcile_aicore_counters` produced it:
     * the device's own totals, what the host accepted, and what the host
     * declined. Valid only after a reconcile pass for the same run.
     */
    struct AicoreAccountingView {
        bool known{false};
        bool identity_ok{false};
        uint64_t device_total{0};
        uint64_t device_dropped{0};
        uint64_t host_collected{0};
        uint64_t host_skipped{0};
        uint64_t skipped_unwritten{0};
        uint64_t skipped_overflow{0};
        uint64_t skipped_bad_core{0};
        uint64_t foreign_identity{0};
    };
    AicoreAccountingView aicore_accounting_for_test() const {
        return {aicore_accounting_.known,          aicore_accounting_.identity_ok,    aicore_accounting_.device_total,
                aicore_accounting_.device_dropped, aicore_accounting_.host_collected, aicore_accounting_.host_skipped,
                aicore_skipped_unwritten_,         aicore_skipped_overflow_,          aicore_skipped_bad_core_,
                aicore_foreign_identity_};
    }

    /**
     * Publish per-core core_type (AIC/AIV/...) so the host emit path can
     * resolve the lane label without consulting an AICPU task record. Required
     * for TASK_TIMING (level=1) where complete_task is bypassed and the
     * AICore record alone is on disk. Caller is the device_runner, on both
     * onboard and sim, and both read the same host-side launch-shape rule
     * (`Runtime::core_type_rule`) — not the handshake region, whose core_type
     * word carries each AICore's own report.
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
    /**
     * Export collected records as a Chrome Trace Event JSON (swimlane view).
     * Writes <output_prefix>/chip_swimlane_records.json — directory is captured at
     * initialize() time.
     *
     * Seals this run's data out of the collector and writes from the sealed
     * copy, so the writer reads no mutable collector state. The sealed data is
     * consumed here and released on return.
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
     * One producer class's retained terminal accounting for one run.
     *
     * `producers` counts the entries that carried the expected epoch, so a class
     * whose pools were disabled reports zero producers rather than zero records
     * — the two are different facts.
     *
     * `reported_indices` records *which* indices those were, not merely how
     * many. Cardinality alone cannot tell a complete set from one where an
     * unexpected index stands in for a missing expected one.
     */
    struct RunTerminalClassSnapshot {
        int producers{0};
        uint64_t total{0};
        uint64_t dropped{0};
        // What the producers actually committed to the ready queue, and what
        // was still unsettled when they closed. Summed over the entries that
        // carried this run's epoch, like `total` / `dropped`.
        uint64_t published_records{0};
        uint64_t published_buffers{0};
        uint64_t live_at_close{0};
        // At least one entry hit the device's saturation sentinel, so this
        // class's figures are bounds rather than counts.
        bool saturated{false};
        std::vector<int> reported_indices;
    };

    /**
     * A run's retained terminal snapshot, read back from its bank.
     *
     * `transport_ok` says the bank's bytes reached the host: the device copy
     * succeeded, or the platform shares memory and no copy was needed. It is a
     * property of the read, not of the contents — an all-zero bank and a bank
     * holding only another run's entries are both successfully read.
     *
     * `valid` says at least one entry carried this run's epoch. It is entry
     * presence, never coverage and never read success; a class's `producers`
     * count is what carries how much of that class reported.
     *
     * The two combine into distinct outcomes, and a reader must not collapse
     * them. `!transport_ok` means the input never arrived: nothing about the run
     * is known, and no verdict follows. `transport_ok && !valid` means the bank
     * was read and holds no entry for this run, which is a real observation —
     * every expected producer is absent, which the consistency verdict reports
     * as `Partial`, not as unknown.
     *
     * A matching non-zero epoch is the per-entry test. It also covers the
     * allocation's lifetime without a second mechanism: `initialize()` refuses
     * while a region is held, so a new one only follows `finalize()`, which nulls
     * the region pointer, and the new region's entries start at the zeroed
     * no-snapshot state.
     */
    struct RunTerminalSnapshot {
        bool transport_ok{false};
        bool valid{false};
        uint64_t run_epoch{0};
        int foreign_entries{0};  // entries holding some other run's epoch
        RunTerminalClassSnapshot aicpu_task;
        RunTerminalClassSnapshot aicore_task;
        RunTerminalClassSnapshot sched_phase;
        RunTerminalClassSnapshot orch_phase;
    };

    /**
     * How one producer class's retained snapshot compares with the live pool
     * counters that `reconcile_counters` summed for the same run.
     *
     * Both sides are device-derived and written by the same producer, so this
     * states whether the retained copy agrees with the live one. It is not a
     * record-loss finding: host-collected loss is reconcile's own
     * `collected + dropped == total` check, which stays authoritative.
     */
    enum class RunTerminalVerdict {
        Unknown,        // a required input was missing or unreadable
        NotApplicable,  // no producer of this class was expected on this run
        Unexpected,     // an entry carries this run's epoch at an index outside the expected set
        Partial,        // an expected index published no entry for this run
        Disagree,       // the expected indices all reported, but the sums differ
        Agree,          // the expected indices all reported and the sums match
    };

    /**
     * Return the device address of `bank_index`'s first terminal entry for the
     * run identified by `run_epoch`, or nullptr when there is nothing to arm (no
     * region, no device allocation, bank out of range, or a zero epoch). The
     * caller publishes the result into KernelArgs; a nullptr becomes a zero
     * field, which the device reads as "publish no snapshot".
     *
     * Deliberately does not clear the bank. The previous occupant's snapshot
     * stays readable until this run's producers overwrite their own entries, and
     * zeroing here would destroy it at the one moment a reader might still want
     * it. Entries start zeroed by the region's initialization memset.
     */
    void *arm_run_terminal_bank(uint32_t bank_index, uint64_t run_epoch);

    /**
     * Read back the snapshot armed for `run_epoch` at `bank_index`.
     *
     * Only sound after that run's completion has been established positively —
     * this performs no synchronization of its own and assumes no producer is
     * still writing. Callers gate on the device completion fence.
     */
    RunTerminalSnapshot read_run_terminal_snapshot(uint32_t bank_index, uint64_t run_epoch);

    /**
     * One producer class's snapshot-vs-live consistency result.
     *
     * `expected_count` is meaningful only for the task classes, whose expected
     * index set is `[0, num_aicore_)` and is host-known. The phase classes have
     * no host-side expected set — see `run_terminal_consistency`.
     */
    struct RunTerminalClassConsistency {
        RunTerminalVerdict verdict{RunTerminalVerdict::Unknown};
        int expected_count{0};
        int reported_count{0};
        int missing_count{0};
        int unexpected_count{0};
    };

    struct RunTerminalConsistency {
        RunTerminalClassConsistency aicpu_task;
        RunTerminalClassConsistency aicore_task;
        RunTerminalClassConsistency sched_phase;
        RunTerminalClassConsistency orch_phase;
    };

    /**
     * Compare this run's retained snapshot with the live pool counters
     * `reconcile_counters` summed for the same run.
     *
     * Scope: snapshot-vs-live consistency for one run under the current
     * exclusivity, per-run reset and completion-fence preconditions. It does
     * not establish that a run's accounting is complete, that the host lost no
     * records, or that snapshots would remain sound under overlapping runs.
     *
     * Only the task classes are compared. Their expected index set is
     * `[0, num_aicore_)`, which the host supplies to `initialize()` and
     * therefore knows independently of anything the device reports. Both have a
     * live counterpart: `reconcile_counters` sums the AICPU pool and
     * `reconcile_aicore_counters` the AICore one, from the same refreshed
     * mirror. For either, the sums compared are the producers' own
     * total/dropped — the host's accepted and declined record counts are a
     * separate quantity, reported by reconcile and never folded into these.
     *
     * The phase classes report `Unknown`: their producer counts exist only as
     * untagged device observations in the shared header, which no per-run reset
     * clears, so a successful read cannot distinguish this run's counts from a
     * previous run's. Supplying an independent phase denominator needs
     * configuration the host does not have, and is not attempted here.
     *
     * Must be called after `reconcile_counters` for the same run, which is what
     * captures the live side.
     */
    RunTerminalConsistency run_terminal_consistency(const RunTerminalSnapshot &snapshot) const;

    /**
     * Read the snapshot, compare it with the live counters, and log both beside
     * `reconcile_counters`' accounting. Diagnostic only: it changes no run
     * outcome and reconcile stays authoritative. The snapshot and verdict are
     * retained for `seal_run_export()`; no second bank read is performed.
     */
    void report_run_terminal_snapshot(uint32_t bank_index, uint64_t run_epoch);

    /**
     * How one producer class's handoff compares with what the host received.
     *
     * Deliberately refuses a numeric verdict in every state where the inputs
     * are not a closed set: a saturated counter is a bound, an unsettled
     * producer never finished its accounting, an inconsistent triple means one
     * of the three wrapped before saturation was in place, a class whose
     * expected producers did not all report is a partial sum, and an untrusted
     * record figure came from a buffer whose own count was out of range. None
     * of those is a completion signal, and none of them is reported as loss.
     */
    enum class HandoffVerdict {
        Unknown,           // no readable terminal, or no independent coverage to judge this class by
        NotApplicable,     // this class has no producer on this run
        Incomplete,        // the reporting producers are not the expected set, so the sums are partial
        Saturated,         // a device counter reached its sentinel
        Unsettled,         // live_at_close != 0: the producer's own accounting did not close
        Inconsistent,      // dropped + published > total
        RecordsUntrusted,  // buffers agree, but a malformed count means the records cannot be compared
        Shortfall,         // fewer buffers received than the device committed
        Overrun,           // more buffers received than the device committed
        RecordMismatch,    // buffers agree, records do not
        Match,             // buffers and records both agree
    };

    /**
     * Whether the terminal entries summed for a class are the whole class.
     *
     * Independent of anything the device reports: the task classes are judged
     * against `[0, num_aicore_)`, the index set the host itself passed to
     * `initialize()`. The phase classes have no such denominator — how many
     * threads produce is not host-known — so their coverage is `Unknown` and
     * stays that way. An absent phase class is therefore never reported as
     * `NotApplicable`: "no producer exists" and "the producers did not report"
     * are not distinguishable without an expected set, and claiming the former
     * would be claiming more than the data supports.
     */
    enum class HandoffCoverage {
        Unknown,        // this class has no host-independent expected set
        NotApplicable,  // the expected set is empty: no producer of this class exists on this run
        Incomplete,     // an expected producer published no entry, or an entry sits at an unexpected index
        Complete,       // exactly the expected producers published an entry
    };

    struct HandoffClassReport {
        HandoffVerdict verdict{HandoffVerdict::Unknown};
        // Coverage and record trust are orthogonal to the verdict and to each
        // other, and both are retained whatever the verdict says: a class can
        // be fully covered with untrusted records, or trusted-but-partial.
        HandoffCoverage coverage{HandoffCoverage::Unknown};
        bool records_trusted{false};
        int expected_producers{0};
        int reported_producers{0};
        int missing_producers{0};
        int unexpected_producers{0};
        uint64_t published_buffers{0};
        uint64_t received_buffers{0};
        uint64_t published_records{0};
        uint64_t received_records{0};
        uint64_t observed_buffers{0};
        uint64_t invalid_index_buffers{0};
        uint64_t foreign_epoch_buffers{0};
        uint64_t malformed_count_buffers{0};
        uint64_t live_at_close{0};
        // device_total - device_dropped - published_records, and only in the
        // states where that subtraction is meaningful.
        uint64_t silent_loss{0};
        bool silent_loss_known{false};
    };

    /**
     * The four classes plus what the transport layer saw around them.
     *
     * `presented_buffers` counts every buffer the poll loop handed the
     * collector, before any classification, so a buffer of an unroutable kind
     * is still accounted — as `unroutable_buffers`, which no class receipt can
     * hold.
     *
     * `transport_retired_buffers` is the layer above: the drain path resolves
     * each ready entry before delivery and retires the ones whose kind or index
     * does not validate, so those never reach the collector at all and are not
     * in `presented_buffers`. A non-zero value means observation here is not
     * the whole transport picture for this run.
     */
    struct HandoffReport {
        HandoffClassReport aicpu_task;
        HandoffClassReport aicore_task;
        HandoffClassReport sched_phase;
        HandoffClassReport orch_phase;
        uint64_t presented_buffers{0};
        uint64_t unroutable_buffers{0};
        uint64_t transport_retired_buffers{0};
    };

    /** This run's handoff report, as `report_run_terminal_snapshot` produced it. */
    HandoffReport handoff_report_for_test() const { return handoff_report_; }

    /**
     * One completed run's diagnostic data, owned by that run.
     *
     * A plain owned value: every field is held by value, the record streams are
     * moved out of the collector and the rest is copied, so no element aliases
     * collector storage that a later `begin_run()` reuses. It is an ordinary
     * mutable aggregate — the writer takes it by `const &`, which is what keeps
     * serialization from touching it; nothing here is enforced by the type.
     *
     * The accounting fields are not serialized. They are held so this object is
     * the whole of the run's diagnostic state and no reader has to go back to
     * the collector for part of it.
     *
     * `sched_phase_dropped_records` and `num_orch_phase_threads` come from the
     * shared-memory header at seal time, which is what lets the writer run
     * without a region.
     */
    struct RunExport {
        std::string output_prefix;
        ChipSwimlaneLevel level{ChipSwimlaneLevel::DISABLED};
        std::array<std::string, static_cast<size_t>(ChipSwimlaneExtensionSection::Count)> json_extensions{};

        std::vector<std::vector<CollectedRecord<ChipSwimlaneAicpuTaskRecord>>> perf_records;
        std::vector<std::vector<CollectedRecord<ChipSwimlaneAicoreTaskRecord>>> aicore_records;
        std::vector<std::vector<CollectedRecord<ChipSwimlaneAicpuSchedPhaseRecord>>> sched_phase_records;
        std::vector<std::vector<CollectedRecord<ChipSwimlaneAicpuOrchPhaseRecord>>> orch_phase_records;

        int num_aicore{0};
        std::vector<CoreType> core_types;
        std::vector<int8_t> core_to_thread;

        bool host_orchestrated{false};
        bool host_phase_records_present{false};
        std::vector<HostPhaseRecord> host_submit_records;
        std::vector<HostPhaseRecord> host_upload_records;
        uint64_t host_phase_submitted_tasks{0};
        uint64_t host_phase_total_records{0};
        uint64_t host_phase_dropped_records{0};

        std::vector<uint32_t> sched_phase_dropped_records;
        uint32_t num_orch_phase_threads{0};

        uint64_t total_perf_collected{0};
        uint64_t total_sched_phase_collected{0};
        uint64_t total_orch_phase_collected{0};
        uint64_t total_aicore_collected{0};
        AicoreAccountingView aicore_accounting{};
        bool has_phase_data{false};
        uint64_t armed_run_epoch{0};
        bool terminal_reported{false};
        RunTerminalSnapshot terminal_snapshot;
        RunTerminalConsistency terminal_consistency;
    };

    /**
     * Detach this run's diagnostic data from the collector.
     *
     * Merges any unmerged shards, then moves the record streams out and copies
     * the rest, and releases the per-shard duplicates the merge left behind. On
     * return the collector holds none of this run's records, so a subsequent
     * `begin_run()` cannot reach them. Call after the last writer — the
     * host-phase insertion — and after the metadata and accounting reads;
     * `export_swimlane_json()` is that point today.
     *
     * Shares `merge_collector_shards`' precondition: the collector threads must
     * be quiesced, since this both reads and clears the per-shard vectors they
     * would otherwise be appending to.
     *
     * Like reconcile and export, not idempotent: a second seal returns a scope
     * whose record streams are already gone.
     */
    RunExport seal_run_export();

    /**
     * Write `<output_prefix>/chip_swimlane_records.json` from sealed data.
     *
     * Static so that the writer reads no collector state at all: the JSON is a
     * function of the sealed run and of nothing else.
     *
     * @return 0 on success, error code on failure
     */
    static int write_swimlane_json(const RunExport &data);

    /**
     * @return Per-core ChipSwimlaneAicpuTaskRecord vectors (indexed by core_index). For tests.
     */
    const std::vector<std::vector<CollectedRecord<ChipSwimlaneAicpuTaskRecord>>> &get_records() const {
        return collected_perf_records_;
    }

private:
    static constexpr size_t kProducerClasses = 4;  // indexed by ProfBufferType

    /**
     * What the transport presented and what this run validly received, for one
     * producer class.
     *
     * Deliberately three layers rather than one number. `observed_buffers` is a
     * transport fact and is incremented before anything is checked.
     * `received_*` are run-owned: a buffer only reaches them once its kind,
     * index and epoch are valid, so a foreign-epoch buffer can never discharge
     * the current run's handoff. Retention — how many records the host actually
     * kept — stays in the `total_*_collected` figures above and is a third
     * quantity again.
     *
     * `received_records` is added only when the buffer's own count is within
     * capacity; a malformed count still counts the buffer, because the handoff
     * happened, but its record figure is not trustworthy and
     * `malformed_count_buffers` is what marks the producer's record verdict
     * untrusted.
     */
    struct HandoffReceipt {
        uint64_t observed_buffers{0};
        uint64_t invalid_index_buffers{0};
        uint64_t foreign_epoch_buffers{0};
        uint64_t malformed_count_buffers{0};
        uint64_t received_buffers{0};
        uint64_t received_records{0};
    };

    struct alignas(64) CollectorShardCounters {
        uint64_t total_perf_collected{0};
        uint64_t total_sched_phase_collected{0};
        uint64_t total_orch_phase_collected{0};
        // AICore records this shard accepted into its vectors: the device's
        // buffer count less every slot the host itself declined. The four
        // reasons are counted separately below rather than folded in, because a
        // record the host dropped is not a record the device dropped and the
        // two must not be summed into one figure.
        uint64_t total_aicore_collected{0};
        uint64_t aicore_skipped_unwritten{0};  // start_time == 0
        uint64_t aicore_skipped_overflow{0};   // buffer count above capacity
        uint64_t aicore_skipped_bad_core{0};   // core index outside this run's set
        // Records whose buffer carried another run's stamp, or none. Kept out
        // of both `collected` and the skip tallies: they belong to no side of
        // this run's conservation check.
        uint64_t aicore_foreign_identity{0};
        bool has_phase_data{false};
        // Every buffer the poll loop handed this shard, counted before the kind
        // is used to pick a class — a kind outside the four is `unroutable` and
        // has no class receipt to land in, but it is still a buffer this
        // collector was handed.
        uint64_t buffers_presented{0};
        uint64_t unroutable_buffers{0};
        // Handoff receipt, kept apart from retention above. `observed` counts
        // every buffer the transport presented for this class, before any
        // validation, so a buffer this host declines is still visible
        // somewhere. The three reject tallies are the reasons it was declined;
        // `received_*` count only what was valid enough to attribute to this
        // run's producer.
        //
        // Per shard, so the collector threads never share a counter; merged at
        // reconcile on the owning thread after quiesce.
        HandoffReceipt receipt[kProducerClasses]{};
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
    std::vector<std::vector<CollectedRecord<ChipSwimlaneAicpuTaskRecord>>> collected_perf_records_;

    // Collected AICore records (per-core vectors). Each entry is a full
    // ChipSwimlaneAicoreTaskRecord captured from a rotated ChipSwimlaneAicoreTaskBuffer.
    std::vector<std::vector<CollectedRecord<ChipSwimlaneAicoreTaskRecord>>> collected_aicore_records_;

    // AICPU phase profiling data — separate per-thread vectors for sched and
    // orch records (kind-tagged at routing time; no parse-time discrimination).
    std::vector<std::vector<CollectedRecord<ChipSwimlaneAicpuSchedPhaseRecord>>> collected_sched_phase_records_;
    std::vector<std::vector<CollectedRecord<ChipSwimlaneAicpuOrchPhaseRecord>>> collected_orch_phase_records_;
    std::vector<HostPhaseRecord> host_submit_records_;
    std::vector<HostPhaseRecord> host_upload_records_;

    // Core-to-thread mapping (core_id → scheduler thread index, -1 = unassigned)
    std::vector<int8_t> core_to_thread_;

    RecordsByCollector<CollectedRecord<ChipSwimlaneAicpuTaskRecord>> perf_records_by_collector_;
    RecordsByCollector<CollectedRecord<ChipSwimlaneAicoreTaskRecord>> aicore_records_by_collector_;
    RecordsByCollector<CollectedRecord<ChipSwimlaneAicpuSchedPhaseRecord>> sched_phase_records_by_collector_;
    RecordsByCollector<CollectedRecord<ChipSwimlaneAicpuOrchPhaseRecord>> orch_phase_records_by_collector_;
    std::vector<CollectorShardCounters> collector_counters_;

    // Running totals used at reconcile time to cross-check device-side counters.
    uint64_t total_perf_collected_{0};
    uint64_t total_sched_phase_collected_{0};
    uint64_t total_orch_phase_collected_{0};
    // Merged AICore accounting. `collected` is what the host accepted; the four
    // skip tallies are what it declined, each for its own reason. They are
    // reported beside the device's own totals, never added to them.
    uint64_t total_aicore_collected_{0};
    uint64_t aicore_skipped_unwritten_{0};
    uint64_t aicore_skipped_overflow_{0};
    uint64_t aicore_skipped_bad_core_{0};
    uint64_t aicore_foreign_identity_{0};
    bool has_phase_data_{false};
    bool collector_shards_merged_{false};
    // Set once the runner has handed over a pass's host phase records, which is
    // also what makes the host orchestrator this run's record source.
    bool host_orchestrated_{false};
    bool host_phase_records_present_{false};
    uint64_t host_phase_total_records_{0};
    uint64_t host_phase_dropped_records_{0};
    uint64_t host_phase_submitted_tasks_{0};

    // The live pool figures reconcile_counters summed for the current run, kept
    // so the terminal-snapshot comparison reads the same numbers reconcile
    // logged rather than re-deriving them.
    //
    // `live_ok` is false until a reconcile pass for this run has produced them,
    // and `begin_run` clears it: a previous run's live figures are not this
    // run's, and comparing against them would report agreement that was never
    // established.
    struct LiveTaskCounters {
        bool live_ok{false};
        bool mirror_ok{false};  // the bulk device mirror reconcile reads succeeded
        uint64_t aicpu_task_total{0};
        uint64_t aicpu_task_dropped{0};
    };
    LiveTaskCounters live_counters_{};

    // The AICore pool's per-run accounting, produced by
    // `reconcile_aicore_counters` from the mirror reconcile already refreshed.
    //
    // `host_collected` counts records this host accepted; `host_skipped` counts
    // the slots it declined for its own four reasons. They are separate fields
    // because a host-side reduction is not a device-side drop, and the
    // comparison must never present one as the other.
    struct AicoreAccounting {
        bool known{false};
        // True only when an expected run identity exists AND every record
        // carried it. False covers both a foreign record and no expected
        // identity at all; in either case the conservation figures are not
        // evidence about a run, however few records arrived.
        bool identity_ok{false};
        uint64_t device_total{0};
        uint64_t device_dropped{0};
        uint64_t host_collected{0};
        uint64_t host_skipped{0};
        uint64_t foreign_identity{0};
    };
    AicoreAccounting aicore_accounting_{};

    // The run identity the last successful `arm_run_terminal_bank` was given.
    // 0 whenever this collector holds none: before any arm, after one that
    // failed for any reason, and after `finalize`. With 0 the host can
    // attribute no record to a run, so the AICore accounting carries no
    // verdict rather than inheriting the previous run's identity.
    uint64_t armed_run_epoch_{0};

    // Per-class handoff receipt, summed from the shards at merge time.
    HandoffReceipt merged_receipt_[kProducerClasses]{};
    uint64_t merged_presented_buffers_{0};
    uint64_t merged_unroutable_buffers_{0};
    // What the drain path retired before it could be presented, captured in
    // reconcile because `report_drain_drops()` consumes the counter.
    uint64_t transport_retired_buffers_{0};
    HandoffReport handoff_report_{};

    // What `report_run_terminal_snapshot` read and concluded for this run, kept
    // so the seal can carry it without a second bank read. `begin_run` clears
    // the flag: a predecessor's terminal verdict is not this run's.
    bool terminal_reported_{false};
    RunTerminalSnapshot terminal_snapshot_{};
    RunTerminalConsistency terminal_consistency_{};

    void reconcile_aicore_counters();

    /**
     * Map a collector thread's shard index onto `collector_counters_`.
     *
     * Precondition, and the reason the out-of-range return is unreachable in
     * production: `ProfilerBase` spawns exactly `shard_count_` collector
     * threads with indices `[0, shard_count_)` and passes each its own index
     * down to `on_buffer_collected`, while `reset_collector_shards` sizes
     * `collector_counters_` to that same count. Returns `shard_count` on a
     * violation, which callers must treat as "no shard owns this call".
     */
    size_t normalize_collector_shard(int collector_shard) const;
    bool producer_index_in_range(ProfBufferType type, uint32_t index) const;
    bool read_buffer_identity(
        const ReadyBufferInfo &info, uint64_t *epoch_out, uint32_t *count_out, uint32_t *capacity_out
    ) const;
    void note_buffer_observed(const ReadyBufferInfo &info, int collector_shard);

    /**
     * Which of `expected`'s indices a class's terminal entries cover.
     *
     * The one place the expected-index comparison lives: both the
     * snapshot-vs-live consistency verdict and the handoff report judge
     * coverage from this, so they cannot drift into two different answers
     * about the same bank. `state` is the summary the handoff report uses; a
     * caller with no expected set of its own overrides it with `Unknown`
     * rather than passing a fabricated `expected`.
     */
    struct TerminalIndexCoverage {
        HandoffCoverage state{HandoffCoverage::Unknown};
        int expected{0};
        int reported{0};
        int missing{0};
        int unexpected{0};
    };
    static TerminalIndexCoverage terminal_index_coverage(const RunTerminalClassSnapshot &cls, int expected);
    static HandoffClassReport classify_handoff(
        const RunTerminalClassSnapshot &cls, const HandoffReceipt &receipt, const TerminalIndexCoverage &coverage
    );
    HandoffReport build_handoff_report(const RunTerminalSnapshot &snapshot);
    void reset_collector_shards();
    void merge_collector_shards();

    /**
     * Give the device orch-phase pool its buffers when this run's level needs
     * them and no earlier run built them.
     *
     * The pool's existence is the one thing initialize() derives from the level,
     * and the level is the one part of a run's configuration that begin_run()
     * re-publishes every run. Since initialize() returns early while the region
     * is held, a run that escalates past ORCH_PHASES would otherwise publish a
     * level the pool cannot serve and the device would emit nothing — no error,
     * no reconcile gap, just an empty orch section.
     *
     * Idempotent, and a no-op below ORCH_PHASES or when the host orchestrator is
     * this run's record source (it needs no device pool at any level).
     */
    int ensure_device_orch_pool(ChipSwimlaneLevel chip_swimlane_level);

    // Per-buffer-kind handlers used by on_buffer_collected.
    void copy_perf_buffer(const ReadyBufferInfo &info, int collector_shard);
    void copy_sched_phase_buffer(const ReadyBufferInfo &info, int collector_shard);
    void copy_orch_phase_buffer(const ReadyBufferInfo &info, int collector_shard);
    void copy_aicore_buffer(const ReadyBufferInfo &info, int collector_shard);
};
