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
 * - BufferPoolManager<L2SwimlaneModule>: shared mgmt-thread infrastructure that
 *   polls the AICPU ready queue, replenishes per-core / per-thread free
 *   queues, and hands full buffers off to the collector thread.
 * - L2SwimlaneCollector: copies records from the manager's ready queue into
 *   host vectors and exports the swimlane visualization.
 *
 * a5 specifics: device↔host transfers go through profiling_copy.h. The
 * framework's mgmt loop mirrors the shm region per tick; per-buffer
 * payloads (L2SwimlaneAicpuTaskBuffer / L2SwimlaneAicpuPhaseBuffer) are pulled on demand inside
 * ProfilerAlgorithms.
 */

#ifndef SRC_A5_PLATFORM_INCLUDE_HOST_L2_SWIMLANE_COLLECTOR_H_
#define SRC_A5_PLATFORM_INCLUDE_HOST_L2_SWIMLANE_COLLECTOR_H_

#include <atomic>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <thread>
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
 * L2 Perf has two distinct buffer kinds going through one ready queue per
 * AICPU thread:
 *   - kind 0: per-core L2SwimlaneAicpuTaskBuffer (task records)
 *   - kind 1: per-thread L2SwimlaneAicpuPhaseBuffer (scheduler/orchestrator phase records)
 * The ReadyQueueEntry::kind flag picks between them.
 */

/**
 * Buffer kind discriminator carried in ReadyBufferInfo and used to index
 * the per-kind recycled pool inside BufferPoolManager.
 */
enum class ProfBufferType { AICPU_TASK = 0, AICPU_PHASE = 1 };

/**
 * Information about a ready (full) buffer, passed from mgmt thread to
 * collector thread.
 */
struct ReadyBufferInfo {
    ProfBufferType type;
    uint32_t index;         // core_index (PERF_RECORD) or thread_idx (PHASE)
    uint32_t slot_idx;      // Reserved (unused in free-queue design)
    void *dev_buffer_ptr;   // Device address of the full buffer
    void *host_buffer_ptr;  // Host shadow (filled by ProfilerAlgorithms)
    uint32_t buffer_seq;    // Sequence number for ordering
};

struct L2SwimlaneModule {
    using DataHeader = L2SwimlaneDataHeader;
    using ReadyEntry = ReadyQueueEntry;
    using ReadyBufferInfo = ::ReadyBufferInfo;
    using FreeQueue = L2SwimlaneFreeQueue;  // L2SwimlaneAicpuPhasePool aliases L2SwimlaneAicpuTaskPool

    static constexpr int kBufferKinds = 2;  // 0=PERF_RECORD, 1=PHASE
    static constexpr uint32_t kReadyQueueSize = PLATFORM_PROF_READYQUEUE_SIZE;
    static constexpr uint32_t kSlotCount = PLATFORM_PROF_SLOT_COUNT;
    static constexpr const char *kSubsystemName = "L2SwimlaneModule";

    /**
     * batch_size for proactive_replenish's alloc fallback. Sized so that a
     * fully empty recycled pool refills to the configured per-instance
     * ceiling in one tick.
     */
    static constexpr int batch_size(int kind) {
        constexpr int kPerfBatch = PLATFORM_PROF_BUFFERS_PER_CORE - PLATFORM_PROF_SLOT_COUNT;
        constexpr int kPhaseBatch = PLATFORM_PROF_BUFFERS_PER_THREAD - PLATFORM_PROF_SLOT_COUNT;
        const int b = (kind == 0) ? kPerfBatch : kPhaseBatch;
        return b < 1 ? 1 : b;
    }

    static int kind_of(const ReadyBufferInfo &info) { return static_cast<int>(info.type); }

    static DataHeader *header_from_shm(void *shm) { return get_l2_swimlane_header(shm); }

    /**
     * Branch on `entry.kind` to pick the per-core perf state vs. the
     * per-thread phase state. Returns nullopt for out-of-range indices
     * (which would otherwise corrupt unrelated BufferStates downstream).
     */
    static std::optional<profiling_common::EntrySite<L2SwimlaneModule>>
    resolve_entry(void *shm, DataHeader *header, int /*q*/, const ReadyEntry &entry) {
        const bool is_phase = (entry.kind == L2SwimlaneBufferKind::AicpuPhase);
        const int num_cores = static_cast<int>(header->num_cores);

        if (is_phase) {
            if (entry.core_index >= static_cast<uint32_t>(PLATFORM_MAX_AICPU_THREADS)) {
                LOG_ERROR("L2SwimlaneModule: invalid phase entry: thread=%u", entry.core_index);
                return std::nullopt;
            }
        } else {
            if (entry.core_index >= static_cast<uint32_t>(num_cores)) {
                LOG_ERROR("L2SwimlaneModule: invalid perf entry: core=%u", entry.core_index);
                return std::nullopt;
            }
        }

        L2SwimlaneAicpuTaskPool *state =
            is_phase ? get_phase_buffer_state(shm, num_cores, static_cast<int>(entry.core_index)) :
                       get_perf_buffer_state(shm, static_cast<int>(entry.core_index));

        profiling_common::EntrySite<L2SwimlaneModule> site;
        site.kind = static_cast<int>(entry.kind);
        site.free_queue = &state->free_queue;
        site.buffer_size = is_phase ? sizeof(L2SwimlaneAicpuPhaseBuffer) : sizeof(L2SwimlaneAicpuTaskBuffer);
        site.info.type = is_phase ? ProfBufferType::AICPU_PHASE : ProfBufferType::AICPU_TASK;
        site.info.index = entry.core_index;
        site.info.slot_idx = 0;
        site.info.dev_buffer_ptr = reinterpret_cast<void *>(entry.buffer_ptr);
        site.info.host_buffer_ptr = nullptr;  // filled by ProfilerAlgorithms
        site.info.buffer_seq = entry.buffer_seq;
        return site;
    }

    template <typename Cb>
    static void for_each_instance(void *shm, DataHeader *header, Cb &&cb) {
        const int num_cores = static_cast<int>(header->num_cores);

        // Per-core perf states (kind 0)
        for (int i = 0; i < num_cores; i++) {
            L2SwimlaneAicpuTaskPool *state = get_perf_buffer_state(shm, i);
            cb(/*kind=*/0, &state->free_queue, sizeof(L2SwimlaneAicpuTaskBuffer));
        }

        // Per-thread phase states (kind 1) — gated on L2SwimlaneAicpuPhaseHeader being
        // initialized (runtimes that don't emit phase records leave it zero).
        L2SwimlaneAicpuPhaseHeader *ph = get_phase_header(shm, num_cores);
        const int num_phase_threads =
            (ph->magic == L2_SWIMLANE_AICPU_PHASE_MAGIC) ? static_cast<int>(ph->num_sched_threads) : 0;
        for (int t = 0; t < num_phase_threads; t++) {
            L2SwimlaneAicpuPhasePool *state = get_phase_buffer_state(shm, num_cores, t);
            cb(/*kind=*/1, &state->free_queue, sizeof(L2SwimlaneAicpuPhaseBuffer));
        }
    }
};

// Memory callbacks — thin aliases for the canonical profiling_common shapes.
// alloc / free are std::function so callers bind their MemoryAllocator via
// lambda capture; register / unregister stay as plain function pointers
// because they wrap stateless HAL globals. On a5 onboard the runner passes
// register_cb=nullptr and the framework installs a malloc-shadow + DMA
// fallback inline in ProfilerBase::start().
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
 *   1. initialize()                — allocate shared memory, pre-fill
 *                                    free_queues, hand the memory context
 *                                    to the base via set_memory_context().
 *   2. start(tf)                   — inherited from ProfilerBase: assembles
 *                                    a MemoryOps from the stashed callbacks
 *                                    and launches the mgmt + poll threads.
 *   3. ... device execution ...
 *   4. stop()                      — joins both threads in the correct
 *                                    order (mgmt first so its final-drain
 *                                    entries have a consumer).
 *   5. read_phase_header_metadata() — single-shot read of the
 *                                    core→thread mapping from the
 *                                    L2SwimlaneAicpuPhaseHeader.
 *   6. reconcile_counters()        — leftover-active sanity check (a5 lacks
 *                                    total/dropped/mismatch counters until
 *                                    the staging-ring redesign lands).
 *   7. export_swimlane_json() / finalize().
 *
 * Host never reads from device-side `current_buf_ptr` to recover records:
 * device flush is the only data path. Any non-zero `current_buf_ptr` after
 * stop() with non-empty count is logged as a bug.
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
     * @param num_aicore     Number of AICore instances
     * @param device_id      Device ID (forwarded to register_cb)
     * @param l2_swimlane_level  Collection granularity (DISABLED / AICORE_TIMING
     *                       / AICPU_TIMING / SCHED_PHASES / ORCH_PHASES).
     *                       Written into `L2SwimlaneDataHeader::l2_swimlane_level`
     *                       so AICPU can promote it in `l2_swimlane_aicpu_init`,
     *                       AND cached on the collector so
     *                       `export_swimlane_json()` can gate phase sections
     *                       and stamp the JSON `version`.
     * @param alloc_cb       Device memory allocation callback
     * @param register_cb    Memory registration callback (nullptr on a5 ⇒
     *                       host-shadow allocation via malloc)
     * @param free_cb        Device memory free callback
     * @param user_data      Opaque pointer forwarded to callbacks
     * @param output_prefix  Per-task directory; l2_swimlane_records.json lands
     *                       here. Required (non-empty); CallConfig::validate()
     *                       enforces this upstream.
     * @return 0 on success, error code on failure
     */
    int initialize(
        int num_aicore, int device_id, L2SwimlaneLevel l2_swimlane_level, const L2SwimlaneAllocCallback &alloc_cb,
        L2SwimlaneRegisterCallback register_cb, const L2SwimlaneFreeCallback &free_cb, const std::string &output_prefix
    );

    /**
     * Per-buffer callback invoked by ProfilerBase's poll loop. Dispatches
     * on info.type to copy either an L2SwimlaneAicpuTaskBuffer (PERF_RECORD) into the
     * per-core record vector or a L2SwimlaneAicpuPhaseBuffer (PHASE) into the per-thread
     * phase-record vector.
     */
    void on_buffer_collected(const ReadyBufferInfo &info);

    /**
     * Export collected records as a Chrome Trace Event JSON (swimlane view).
     * Writes <output_prefix>/l2_swimlane_records.json — directory captured at
     * initialize() time.
     *
     * @return 0 on success, error code on failure
     */
    int export_swimlane_json();

    /**
     * Free all device memory and unregister mappings. Idempotent on a
     * collector that was never initialized.
     *
     * @param unregister_cb  Memory unregister callback (nullptr on a5)
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
     * Device pointer to the per-core L2SwimlaneAicoreRing-address table
     * (uint64_t[num_aicore]). Wire this into
     * `KernelArgs::aicore_l2_swimlane_ring_addrs` so the AICore kernel
     * entry forwards each core's ring pointer into platform state.
     */
    void *get_aicore_ring_addrs_device_ptr() const { return aicore_ring_addrs_dev_; }

    /**
     * Read AICPU phase metadata that lives in L2SwimlaneAicpuPhaseHeader (not on the
     * buffer pipeline): the core→thread mapping plus a has-data signal
     * derived from accumulated per-event records. Single-shot — must be
     * called after stop() so the shm region has settled.
     * The shm region was last mirrored to host shadow at the end of mgmt's
     * final-drain pass.
     */
    void read_phase_header_metadata();

    /**
     * Sanity-check per-core / per-thread `current_buf_ptr` for any
     * un-flushed leftovers (device flush should always succeed-or-bump-
     * dropped, so a non-empty leftover indicates an AICPU flush bug).
     *
     * NOTE: a5's L2SwimlaneAicpuTaskPool does not yet carry total/dropped/mismatch
     * counters (they land with the AICore staging-ring redesign in a later
     * task). The full `collected + dropped + mismatch == device_total`
     * cross-check is therefore deferred. Must be called after stop().
     */
    void reconcile_counters();

    /**
     * @return Per-core L2SwimlaneAicpuTaskRecord vectors (indexed by core_index). For tests.
     */
    const std::vector<std::vector<L2SwimlaneAicpuTaskRecord>> &get_records() const { return collected_perf_records_; }

private:
    // Shared memory pointers. shm_host_ / device_id_ live on ProfilerBase
    // (set via set_memory_context in initialize()).
    void *perf_shared_mem_dev_{nullptr};

    // Per-core stable AICore staging rings — allocated once, never rotated.
    // The host owns the device-side L2SwimlaneAicoreRing buffers and the address
    // table; AICPU reads `state.aicore_ring_ptr` (set at init), and AICore
    // reads from `KernelArgs::aicore_l2_swimlane_ring_addrs[block_idx]`.
    std::vector<void *> aicore_rings_dev_;
    void *aicore_ring_addrs_dev_{nullptr};
    void *aicore_ring_addrs_host_{nullptr};

    int num_aicore_{0};
    L2SwimlaneLevel l2_swimlane_level_{L2SwimlaneLevel::DISABLED};

    // Per-task output directory captured at initialize() time. Consumed by
    // export_swimlane_json() to build <prefix>/l2_swimlane_records.json.
    std::string output_prefix_;

    // Collected data (per-core vectors, indexed by core_index)
    std::vector<std::vector<L2SwimlaneAicpuTaskRecord>> collected_perf_records_;

    // AICPU phase profiling data (per-thread, mixed sched + orch records)
    std::vector<std::vector<L2SwimlaneAicpuPhaseRecord>> collected_phase_records_;
    bool has_phase_data_{false};

    // Core-to-thread mapping (core_id → scheduler thread index, -1 = unassigned)
    std::vector<int8_t> core_to_thread_;

    // Running totals used at reconcile time. The full cross-check awaits
    // task-02 staging-ring counters; for now we just log the collected
    // total alongside the leftover-active sanity result.
    uint64_t total_perf_collected_{0};
    uint64_t total_phase_collected_{0};

    // Allocate a single buffer (shm region / L2SwimlaneAicpuTaskBuffer / L2SwimlaneAicpuPhaseBuffer) and
    // its paired host shadow.

    // Per-buffer-kind handlers used by on_buffer_collected.
    void copy_perf_buffer(const ReadyBufferInfo &info);
    void copy_phase_buffer(const ReadyBufferInfo &info);
};

#endif  // SRC_A5_PLATFORM_INCLUDE_HOST_L2_SWIMLANE_COLLECTOR_H_
