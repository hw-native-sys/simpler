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
 * @file l2_swimlane_collector.h
 * @brief Platform-agnostic performance data collector with dynamic memory management.
 *
 * Architecture:
 * - BufferPoolManager<L2SwimlaneModule>: shared mgmt-thread infrastructure that polls
 *   the AICPU ready queue, replenishes per-core / per-thread free queues, and
 *   hands full buffers off to collector thread shards.
 * - L2SwimlaneCollector: collector thread shards copy records from manager ready queues
 *   into host vectors; the owner thread exports the swimlane visualization after stop().
 *
 * Memory operations are injected through callbacks for sim/onboard portability.
 */

#ifndef SRC_COMMON_PLATFORM_INCLUDE_HOST_L2_SWIMLANE_COLLECTOR_H_
#define SRC_COMMON_PLATFORM_INCLUDE_HOST_L2_SWIMLANE_COLLECTOR_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "common/l2_swimlane_profiling.h"
#include "common/memory_barrier.h"
#include "common/platform_config.h"
#include "common/unified_log.h"
#include "host/profiler_base.h"

// ---------------------------------------------------------------------------
// L2 Perf profiling Module (drives BufferPoolManager<L2SwimlaneModule>)
// ---------------------------------------------------------------------------

/**
 * L2 Perf has four distinct buffer kinds going through one ready queue per
 * AICPU thread:
 *   - kind 0: per-core    L2SwimlaneAicpuTaskBuffer      (task records)
 *   - kind 1: per-thread  L2SwimlaneAicpuSchedPhaseBuffer (scheduler phase records)
 *   - kind 2: per-thread  L2SwimlaneAicpuOrchPhaseBuffer  (orchestrator phase records)
 *   - kind 3: per-core    L2SwimlaneAicoreTaskBuffer     (AICore-written records)
 * The ReadyQueueEntry::kind flag picks among them.
 */

/**
 * Buffer kind discriminator carried in ReadyBufferInfo and used to index the
 * per-kind recycled pool inside BufferPoolManager. Values match
 * L2SwimlaneBufferKind 1:1.
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

struct L2SwimlaneModule {
    using DataHeader = L2SwimlaneDataHeader;
    using ReadyEntry = ReadyQueueEntry;
    using ReadyBufferInfo = ::ReadyBufferInfo;
    using FreeQueue = L2SwimlaneFreeQueue;  // all pool types share the same free_queue layout

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
    static constexpr const char *kSubsystemName = "L2SwimlaneModule";
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
        switch (static_cast<L2SwimlaneBufferKind>(kind)) {
        case L2SwimlaneBufferKind::AicpuTask:
            b = kPerfBatch;
            break;
        case L2SwimlaneBufferKind::AicpuSchedPhase:
            b = kSchedBatch;
            break;
        case L2SwimlaneBufferKind::AicpuOrchPhase:
            b = kOrchBatch;
            break;
        case L2SwimlaneBufferKind::AicoreTask:
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
        switch (static_cast<L2SwimlaneBufferKind>(kind)) {
        case L2SwimlaneBufferKind::AicpuTask:
            return half_initial_surplus_warm_target(PLATFORM_PROF_BUFFERS_PER_CORE, shard_count);
        case L2SwimlaneBufferKind::AicoreTask:
            return half_initial_surplus_warm_target(PLATFORM_AICORE_BUFFERS_PER_CORE, shard_count);
        case L2SwimlaneBufferKind::AicpuSchedPhase:
        case L2SwimlaneBufferKind::AicpuOrchPhase:
            return 0;
        }
        return 0;
    }

    static int kind_of(const ReadyBufferInfo &info) { return static_cast<int>(info.type); }

    static DataHeader *header_from_shm(void *shm) { return get_l2_swimlane_header(shm); }

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
    static std::optional<profiling_common::EntrySite<L2SwimlaneModule>>
    resolve_entry(void *shm, DataHeader *header, int /*q*/, const ReadyEntry &entry) {
        const int num_cores = static_cast<int>(header->num_cores);
        const L2SwimlaneBufferKind kind = entry.kind;

        // Validate kind first — out-of-range silently falling into the wrong
        // branch reads a wrong-typed pool.
        if (kind != L2SwimlaneBufferKind::AicpuTask && kind != L2SwimlaneBufferKind::AicpuSchedPhase &&
            kind != L2SwimlaneBufferKind::AicpuOrchPhase && kind != L2SwimlaneBufferKind::AicoreTask) {
            LOG_ERROR("L2SwimlaneModule: invalid entry kind=%u", static_cast<uint32_t>(kind));
            return std::nullopt;
        }

        // Sched/orch phase entries are indexed by thread_idx; task/aicore by core_index.
        const bool is_phase =
            (kind == L2SwimlaneBufferKind::AicpuSchedPhase) || (kind == L2SwimlaneBufferKind::AicpuOrchPhase);
        if (is_phase) {
            if (entry.core_index >= static_cast<uint32_t>(PLATFORM_MAX_AICPU_THREADS)) {
                LOG_ERROR("L2SwimlaneModule: invalid phase entry: thread=%u", entry.core_index);
                return std::nullopt;
            }
        } else {
            if (entry.core_index >= static_cast<uint32_t>(num_cores)) {
                LOG_ERROR(
                    "L2SwimlaneModule: invalid task entry: core=%u kind=%u", entry.core_index,
                    static_cast<uint32_t>(kind)
                );
                return std::nullopt;
            }
        }

        profiling_common::EntrySite<L2SwimlaneModule> site;
        site.kind = static_cast<int>(kind);
        site.info.index = entry.core_index;
        site.info.slot_idx = 0;
        site.info.dev_buffer_ptr = reinterpret_cast<void *>(entry.buffer_ptr);
        site.info.host_buffer_ptr = nullptr;  // filled by ProfilerAlgorithms
        site.info.buffer_seq = entry.buffer_seq;

        switch (kind) {
        case L2SwimlaneBufferKind::AicpuTask: {
            auto *state = get_perf_buffer_state(shm, static_cast<int>(entry.core_index));
            site.free_queue = &state->free_queue;
            site.buffer_size = sizeof(L2SwimlaneAicpuTaskBuffer);
            site.info.type = ProfBufferType::AICPU_TASK;
            break;
        }
        case L2SwimlaneBufferKind::AicpuSchedPhase: {
            auto *state = get_sched_phase_buffer_state(shm, num_cores, static_cast<int>(entry.core_index));
            site.free_queue = &state->free_queue;
            site.buffer_size = sizeof(L2SwimlaneAicpuSchedPhaseBuffer);
            site.info.type = ProfBufferType::AICPU_SCHED_PHASE;
            break;
        }
        case L2SwimlaneBufferKind::AicpuOrchPhase: {
            auto *state = get_orch_phase_buffer_state(shm, num_cores, static_cast<int>(entry.core_index));
            site.free_queue = &state->free_queue;
            site.buffer_size = sizeof(L2SwimlaneAicpuOrchPhaseBuffer);
            site.info.type = ProfBufferType::AICPU_ORCH_PHASE;
            break;
        }
        case L2SwimlaneBufferKind::AicoreTask: {
            auto *ac_state = get_aicore_buffer_state(shm, num_cores, static_cast<int>(entry.core_index));
            site.free_queue = &ac_state->free_queue;
            site.buffer_size = sizeof(L2SwimlaneAicoreTaskBuffer);
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
            cb(/*kind=*/static_cast<int>(L2SwimlaneBufferKind::AicpuTask), &state->free_queue,
               sizeof(L2SwimlaneAicpuTaskBuffer));
        }

        // AicoreTask: per-core (kind 3)
        for (int i = 0; i < num_cores; i++) {
            auto *ac_state = get_aicore_buffer_state(shm, num_cores, i);
            cb(/*kind=*/static_cast<int>(L2SwimlaneBufferKind::AicoreTask), &ac_state->free_queue,
               sizeof(L2SwimlaneAicoreTaskBuffer));
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
            auto *state = get_sched_phase_buffer_state(shm, num_cores, t);
            cb(/*kind=*/static_cast<int>(L2SwimlaneBufferKind::AicpuSchedPhase), &state->free_queue,
               sizeof(L2SwimlaneAicpuSchedPhaseBuffer));
        }

        // AicpuOrchPhase: per-thread (kind 2) — same bounds clamp.
        int num_orch_phase_threads = static_cast<int>(header->num_orch_phase_threads);
        if (num_orch_phase_threads > PLATFORM_MAX_AICPU_THREADS) {
            num_orch_phase_threads = 0;
        }
        for (int t = 0; t < num_orch_phase_threads; t++) {
            auto *state = get_orch_phase_buffer_state(shm, num_cores, t);
            cb(/*kind=*/static_cast<int>(L2SwimlaneBufferKind::AicpuOrchPhase), &state->free_queue,
               sizeof(L2SwimlaneAicpuOrchPhaseBuffer));
        }
    }
};

// Memory callbacks — thin aliases for the canonical profiling_common shapes.
// alloc / free are std::function so callers bind their MemoryAllocator via
// lambda capture; register / unregister stay as plain function pointers
// because they wrap stateless HAL globals (halHost*).
using L2SwimlaneAllocCallback = profiling_common::ProfAllocCallback;
using L2SwimlaneRegisterCallback = profiling_common::ProfRegisterCallback;
using L2SwimlaneUnregisterCallback = profiling_common::ProfUnregisterCallback;
using L2SwimlaneFreeCallback = profiling_common::ProfFreeCallback;

// =============================================================================
// L2SwimlaneCollector
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
 *                                    mapping from L2SwimlaneDataHeader.
 *   6. reconcile_counters()        — device-side three-bucket accounting for
 *                                    both PERF and PHASE pools (total /
 *                                    collected / dropped).
 *   7. export_swimlane_json() / finalize().
 *
 * Host never reads from device-side `current_buf_ptr` to recover records:
 * device flush is the only data path. Any non-zero `current_buf_ptr` after
 * stop() is logged as a bug.
 */
class L2SwimlaneCollector : public profiling_common::ProfilerBase<L2SwimlaneCollector, L2SwimlaneModule> {
public:
    L2SwimlaneCollector() = default;
    ~L2SwimlaneCollector();

    L2SwimlaneCollector(const L2SwimlaneCollector &) = delete;
    L2SwimlaneCollector &operator=(const L2SwimlaneCollector &) = delete;

    // ProfilerBase contract
    static constexpr int kIdleTimeoutSec = PLATFORM_PROF_TIMEOUT_SECONDS;
    static constexpr const char *kSubsystemName = "L2Swimlane";

    /**
     * Initialize performance profiling.
     *
     * Allocates the shared-memory region (header + per-core / per-thread
     * BufferStates), pre-allocates initial L2SwimlaneAicpuTaskBuffers and PhaseBuffers,
     * and seeds the per-pool free_queues + the framework's recycled pools.
     *
     * @param num_aicore               Number of AICore instances
     * @param device_id                Device ID (forwarded to register_cb)
     * @param l2_swimlane_level   Collection granularity (DISABLED / AICORE_TIMING
     *                                 / AICPU_TIMING / SCHED_PHASES / ORCH_PHASES).
     *                                 Written into
     *                                 `L2SwimlaneDataHeader::l2_swimlane_level`
     *                                 so AICPU can promote it in
     *                                 `l2_swimlane_aicpu_init`, AND cached on the
     *                                 collector so `export_swimlane_json()`
     *                                 can gate phase sections and stamp the
     *                                 JSON `version`.
     * @param alloc_cb                 Device memory allocation callback
     * @param register_cb              Memory registration callback (nullptr for
     *                                 simulation and non-SVM platforms)
     * @param free_cb                  Device memory free callback
     * @param user_data                Opaque pointer forwarded to callbacks
     * @param output_prefix            Per-task directory; l2_swimlane_records.json
     *                                 lands here. Required (non-empty);
     *                                 CallConfig::validate() enforces this
     *                                 upstream.
     * @return 0 on success, error code on failure
     */
    int initialize(
        int num_aicore, int aicpu_thread_num, int device_id, L2SwimlaneLevel l2_swimlane_level,
        const L2SwimlaneAllocCallback &alloc_cb, L2SwimlaneRegisterCallback register_cb,
        const L2SwimlaneFreeCallback &free_cb, const std::string &output_prefix
    );

    /**
     * Per-buffer callback invoked by ProfilerBase's poll loop. Dispatches on
     * info.type to copy either an L2SwimlaneAicpuTaskBuffer (PERF_RECORD) into the per-core
     * record vector, or a L2SwimlaneAicpuSchedPhaseBuffer / L2SwimlaneAicpuOrchPhaseBuffer into the per-thread
     * phase-record vector.
     */
    void on_buffer_collected(const ReadyBufferInfo &info, int collector_shard);

    /**
     * Publish per-core core_type (AIC/AIV/...) so the host emit path can
     * resolve the lane label without consulting an AICPU task record. Required
     * for AICORE_TIMING (level=1) where complete_task is bypassed and the
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

    /** Attach host-build-graph orchestration timing captured before initialize(). */
    void set_host_orch_records(
        const std::vector<L2SwimlaneAicpuOrchPhaseRecord> &records, uint64_t host_start_cycles, uint64_t host_end_cycles
    );

    /**
     * Export collected records as a Chrome Trace Event JSON (swimlane view).
     * Writes <output_prefix>/l2_swimlane_records.json — directory is captured at
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
    int finalize(L2SwimlaneUnregisterCallback unregister_cb, const L2SwimlaneFreeCallback &free_cb);

    /**
     * @return true if initialize() succeeded and finalize() has not run.
     */
    bool is_initialized() const { return shm_host_ != nullptr; }

    /**
     * Device pointer to the L2SwimlaneDataHeader. Set kernel_args.l2_swimlane_data_base
     * to this after initialize() succeeds so the AICPU side can find the
     * shared memory.
     */
    void *get_l2_swimlane_setup_device_ptr() const { return perf_shared_mem_dev_; }

    /**
     * Device pointer to a uint64_t[num_aicore] table where each entry will
     * hold this core's `&L2SwimlaneAicoreTaskPool::rotation` device address. Host
     * only allocates the bytes here; AICPU populates the entries inside
     * `l2_swimlane_aicpu_init`. Freed by finalize(). Set kernel_args.l2_swimlane_aicore_rotation_table
     * to this so the AICore kernel entry can index by block_idx and feed the
     * per-core rotation channel into `set_l2_swimlane_aicore_head_slot()`. Returns
     * nullptr before initialize() succeeds.
     */
    void *get_aicore_ring_addr_table_device_ptr() const { return aicore_ring_addr_table_dev_; }

    /**
     * Read AICPU phase metadata that lives in L2SwimlaneDataHeader (not on the
     * buffer pipeline): the core→thread mapping plus a has-data signal
     * derived from accumulated per-event records. Single-shot — must be
     * called after stop() so the shm region has settled.
     */
    void read_phase_header_metadata();

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
     * @return Per-core L2SwimlaneAicpuTaskRecord vectors (indexed by core_index). For tests.
     */
    const std::vector<std::vector<L2SwimlaneAicpuTaskRecord>> &get_records() const { return collected_perf_records_; }

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

    // Standalone uint64_t[num_aicore] table holding per-core L2SwimlaneAicoreTaskBuffer
    // addresses. Allocated in initialize(), freed in finalize(). AICore reads
    // ring_table[block_idx] via KernelArgs::l2_swimlane_aicore_rotation_table.
    void *aicore_ring_addr_table_dev_{nullptr};

    int num_aicore_{0};
    // Total AICPU threads launched this run. The dedicated orchestrator runs on
    // the last one (aicpu_thread_num_ - 1); used to report its thread number in
    // the phase-metadata log (orch-phase is a single pool, so its index alone
    // does not encode the AICPU thread).
    int aicpu_thread_num_{0};
    L2SwimlaneLevel l2_swimlane_level_{L2SwimlaneLevel::DISABLED};

    // Per-core core_type table populated by set_core_types(). Indexed by
    // core_id; size matches num_aicore_ once populated. Used by the level=1
    // emit path which has no AICPU record to read core_type from.
    std::vector<CoreType> core_types_;

    // Per-task output directory captured at initialize() time. Consumed by
    // export_swimlane_json() to build <prefix>/l2_swimlane_records.json.
    std::string output_prefix_;

    // Merged data, populated from per-collector shards after collector threads join.
    std::vector<std::vector<L2SwimlaneAicpuTaskRecord>> collected_perf_records_;

    // Collected AICore records (per-core vectors). Each entry is a full
    // L2SwimlaneAicoreTaskRecord captured from a rotated L2SwimlaneAicoreTaskBuffer.
    std::vector<std::vector<L2SwimlaneAicoreTaskRecord>> collected_aicore_records_;

    // AICPU phase profiling data — separate per-thread vectors for sched and
    // orch records (kind-tagged at routing time; no parse-time discrimination).
    std::vector<std::vector<L2SwimlaneAicpuSchedPhaseRecord>> collected_sched_phase_records_;
    std::vector<std::vector<L2SwimlaneAicpuOrchPhaseRecord>> collected_orch_phase_records_;

    // Host and AICPU clocks do not share an epoch.  These records stay in a
    // separate stream and the converter aligns host-orch end to device t=0.
    std::vector<L2SwimlaneAicpuOrchPhaseRecord> host_orch_phase_records_;
    uint64_t host_orch_start_cycles_{0};
    uint64_t host_orch_end_cycles_{0};

    // Core-to-thread mapping (core_id → scheduler thread index, -1 = unassigned)
    std::vector<int8_t> core_to_thread_;

    RecordsByCollector<L2SwimlaneAicpuTaskRecord> perf_records_by_collector_;
    RecordsByCollector<L2SwimlaneAicoreTaskRecord> aicore_records_by_collector_;
    RecordsByCollector<L2SwimlaneAicpuSchedPhaseRecord> sched_phase_records_by_collector_;
    RecordsByCollector<L2SwimlaneAicpuOrchPhaseRecord> orch_phase_records_by_collector_;
    std::vector<CollectorShardCounters> collector_counters_;

    // Running totals used at reconcile time to cross-check device-side counters.
    uint64_t total_perf_collected_{0};
    uint64_t total_sched_phase_collected_{0};
    uint64_t total_orch_phase_collected_{0};
    bool has_phase_data_{false};
    bool collector_shards_merged_{false};

    size_t normalize_collector_shard(int collector_shard) const;
    void reset_collector_shards();
    void merge_collector_shards();

    // Per-buffer-kind handlers used by on_buffer_collected.
    void copy_perf_buffer(const ReadyBufferInfo &info, int collector_shard);
    void copy_sched_phase_buffer(const ReadyBufferInfo &info, int collector_shard);
    void copy_orch_phase_buffer(const ReadyBufferInfo &info, int collector_shard);
    void copy_aicore_buffer(const ReadyBufferInfo &info, int collector_shard);
};

#endif  // SRC_COMMON_PLATFORM_INCLUDE_HOST_L2_SWIMLANE_COLLECTOR_H_
