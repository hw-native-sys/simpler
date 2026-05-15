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
 * @file l2_perf_collector.h
 * @brief Platform-agnostic performance data collector with dynamic memory management.
 *
 * Architecture:
 * - BufferPoolManager<L2PerfModule>: shared mgmt-thread infrastructure that polls
 *   the AICPU ready queue, replenishes per-core / per-thread free queues, and
 *   hands full buffers off to the collector thread.
 * - L2PerfCollector: main thread copies records from the manager's ready queue
 *   into host vectors and exports the swimlane visualization.
 *
 * Memory operations are injected through callbacks for sim/onboard portability.
 */

#ifndef SRC_A2A3_PLATFORM_INCLUDE_HOST_L2_PERF_COLLECTOR_H_
#define SRC_A2A3_PLATFORM_INCLUDE_HOST_L2_PERF_COLLECTOR_H_

#include <atomic>
#include <cstdint>
#include <functional>
#include <string>
#include <thread>
#include <vector>

#include "common/l2_perf_profiling.h"
#include "common/memory_barrier.h"
#include "common/platform_config.h"
#include "common/unified_log.h"
#include "host/profiling_common/profiler_base.h"

// ---------------------------------------------------------------------------
// L2 Perf profiling Module (drives BufferPoolManager<L2PerfModule>)
// ---------------------------------------------------------------------------

/**
 * L2 Perf has two distinct buffer kinds going through one ready queue per
 * AICPU thread:
 *   - kind 0: per-core L2PerfBuffer (task records)
 *   - kind 1: per-thread PhaseBuffer (scheduler/orchestrator phase records)
 * The ReadyQueueEntry::is_phase flag picks between them.
 */

/**
 * Buffer kind discriminator carried in ReadyBufferInfo and used to index the
 * per-kind recycled pool inside BufferPoolManager.
 */
enum class ProfBufferType { PERF_RECORD = 0, PHASE = 1 };

/**
 * Information about a ready (full) buffer, passed from mgmt thread to main thread.
 */
struct ReadyBufferInfo {
    ProfBufferType type;
    uint32_t index;         // core_index (PERF_RECORD) or thread_idx (PHASE)
    uint32_t slot_idx;      // Reserved (unused in free queue design)
    void *dev_buffer_ptr;   // Device address of the full buffer
    void *host_buffer_ptr;  // Host-mapped address (sim: same as dev)
    uint32_t buffer_seq;    // Sequence number for ordering
};

struct L2PerfModule {
    using DataHeader = L2PerfDataHeader;
    using ReadyEntry = ReadyQueueEntry;
    using ReadyBufferInfo = ::ReadyBufferInfo;
    using FreeQueue = L2PerfFreeQueue;  // PhaseBufferState aliases L2PerfBufferState

    static constexpr int kBufferKinds = 2;  // 0=PERF_RECORD, 1=PHASE
    static constexpr uint32_t kReadyQueueSize = PLATFORM_PROF_READYQUEUE_SIZE;
    static constexpr uint32_t kSlotCount = PLATFORM_PROF_SLOT_COUNT;
    static constexpr const char *kSubsystemName = "L2PerfModule";

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

    static DataHeader *header_from_shm(void *shm) { return get_l2_perf_header(shm); }

    /**
     * Branch on `is_phase` to pick the per-core perf state vs. the per-thread
     * phase state. Returns nullopt for out-of-range indices (which would
     * otherwise corrupt unrelated BufferStates downstream).
     */
    static std::optional<profiling_common::EntrySite<L2PerfModule>>
    resolve_entry(void *shm, DataHeader *header, int /*q*/, const ReadyEntry &entry) {
        const bool is_phase = (entry.is_phase != 0);
        const int num_cores = static_cast<int>(header->num_cores);

        if (is_phase) {
            if (entry.core_index >= static_cast<uint32_t>(PLATFORM_MAX_AICPU_THREADS)) {
                LOG_ERROR("L2PerfModule: invalid phase entry: thread=%u", entry.core_index);
                return std::nullopt;
            }
        } else {
            if (entry.core_index >= static_cast<uint32_t>(num_cores)) {
                LOG_ERROR("L2PerfModule: invalid perf entry: core=%u", entry.core_index);
                return std::nullopt;
            }
        }

        L2PerfBufferState *state = is_phase ?
                                       get_phase_buffer_state(shm, num_cores, static_cast<int>(entry.core_index)) :
                                       get_perf_buffer_state(shm, static_cast<int>(entry.core_index));

        profiling_common::EntrySite<L2PerfModule> site;
        site.kind = is_phase ? 1 : 0;
        site.free_queue = &state->free_queue;
        site.buffer_size = is_phase ? sizeof(PhaseBuffer) : sizeof(L2PerfBuffer);
        site.info.type = is_phase ? ProfBufferType::PHASE : ProfBufferType::PERF_RECORD;
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
            L2PerfBufferState *state = get_perf_buffer_state(shm, i);
            cb(/*kind=*/0, &state->free_queue, sizeof(L2PerfBuffer));
        }

        // Per-thread phase states (kind 1) — gated on AicpuPhaseHeader being
        // initialized (runtimes that don't emit phase records leave it zero).
        AicpuPhaseHeader *ph = get_phase_header(shm, num_cores);
        const int num_phase_threads = (ph->magic == AICPU_PHASE_MAGIC) ? static_cast<int>(ph->num_sched_threads) : 0;
        for (int t = 0; t < num_phase_threads; t++) {
            PhaseBufferState *state = get_phase_buffer_state(shm, num_cores, t);
            cb(/*kind=*/1, &state->free_queue, sizeof(PhaseBuffer));
        }
    }
};

/**
 * Memory allocation callback for performance profiling.
 *
 * @param size       Memory size in bytes
 * @param user_data  Opaque allocator context
 * @return Allocated device memory pointer, or nullptr on failure
 */
using L2PerfAllocCallback = void *(*)(size_t size, void *user_data);

/**
 * Memory registration callback (host-visible mapping). nullptr in sim mode
 * (dev pointer is directly host-accessible). Stateless: HAL state is global,
 * so no user_data is threaded through.
 *
 * @param dev_ptr        Device memory pointer
 * @param size           Memory size in bytes
 * @param device_id      Device ID
 * @param[out] host_ptr  Host-mapped pointer
 * @return 0 on success, error code on failure
 */
using L2PerfRegisterCallback = int (*)(void *dev_ptr, size_t size, int device_id, void **host_ptr);

/**
 * Memory unregister callback. May be nullptr. Stateless (see register_cb).
 *
 * @param dev_ptr    Device memory pointer
 * @param device_id  Device ID
 * @return 0 on success, error code on failure
 */
using L2PerfUnregisterCallback = int (*)(void *dev_ptr, int device_id);

/**
 * Memory free callback.
 *
 * @param dev_ptr    Device memory pointer
 * @param user_data  Opaque allocator context
 * @return 0 on success, error code on failure
 */
using L2PerfFreeCallback = int (*)(void *dev_ptr, void *user_data);

// =============================================================================
// L2PerfCollector
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
 *   5. read_phase_header_metadata() — single-shot read of orch_summary +
 *                                    core→thread mapping from AicpuPhaseHeader.
 *   6. reconcile_counters()        — device-side three-bucket accounting for
 *                                    both PERF and PHASE pools (total /
 *                                    collected / dropped).
 *   7. export_swimlane_json() / finalize().
 *
 * Host never reads from device-side `current_buf_ptr` to recover records:
 * device flush is the only data path. Any non-zero `current_buf_ptr` after
 * stop() is logged as a bug.
 */
class L2PerfCollector : public profiling_common::ProfilerBase<L2PerfCollector, L2PerfModule> {
public:
    L2PerfCollector() = default;
    ~L2PerfCollector();

    L2PerfCollector(const L2PerfCollector &) = delete;
    L2PerfCollector &operator=(const L2PerfCollector &) = delete;

    // ProfilerBase contract
    static constexpr int kIdleTimeoutSec = PLATFORM_PROF_TIMEOUT_SECONDS;
    static constexpr const char *kSubsystemName = "L2Perf";

    /**
     * Initialize performance profiling.
     *
     * Allocates the shared-memory region (header + per-core / per-thread
     * BufferStates), pre-allocates initial L2PerfBuffers and PhaseBuffers,
     * and seeds the per-pool free_queues + the framework's recycled pools.
     *
     * @param num_aicore               Number of AICore instances
     * @param device_id                Device ID (forwarded to register_cb)
     * @param l2_swimlane_perf_level   Collection granularity (0=off, 1=AICore
     *                                 timing, 2=+dispatch/fanout, 3=+sched
     *                                 phases, 4=+orch phases). Cached on the
     *                                 collector and consumed by
     *                                 export_swimlane_json() to gate phase
     *                                 sections and stamp the JSON `version`.
     * @param alloc_cb                 Device memory allocation callback
     * @param register_cb              Memory registration callback (nullptr for
     *                                 simulation)
     * @param free_cb                  Device memory free callback
     * @param user_data                Opaque pointer forwarded to callbacks
     * @param output_prefix            Per-task directory; l2_perf_records.json
     *                                 lands here. Required (non-empty);
     *                                 CallConfig::validate() enforces this
     *                                 upstream.
     * @return 0 on success, error code on failure
     */
    int initialize(
        int num_aicore, int device_id, int l2_swimlane_perf_level, L2PerfAllocCallback alloc_cb,
        L2PerfRegisterCallback register_cb, L2PerfFreeCallback free_cb, void *user_data,
        const std::string &output_prefix
    );

    /**
     * Per-buffer callback invoked by ProfilerBase's poll loop. Dispatches on
     * info.type to copy either an L2PerfBuffer (PERF_RECORD) into the per-core
     * record vector, or a PhaseBuffer (PHASE) into the per-thread phase-record
     * vector.
     */
    void on_buffer_collected(const ReadyBufferInfo &info);

    /**
     * Export collected records as a Chrome Trace Event JSON (swimlane view).
     * Writes <output_prefix>/l2_perf_records.json — directory is captured at
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
    int finalize(L2PerfUnregisterCallback unregister_cb, L2PerfFreeCallback free_cb, void *user_data);

    /**
     * @return true if initialize() succeeded and finalize() has not run.
     */
    bool is_initialized() const { return shm_host_ != nullptr; }

    /**
     * Device pointer to the L2PerfDataHeader. Set kernel_args.l2_perf_data_base
     * to this after initialize() succeeds so the AICPU side can find the
     * shared memory.
     */
    void *get_l2_perf_setup_device_ptr() const { return perf_shared_mem_dev_; }

    /**
     * Device pointer to a uint64_t[num_aicore] table where each entry is
     * `L2PerfBufferState[i].aicore_ring_ptr`. Allocated and populated by
     * initialize(); freed by finalize(). Set kernel_args.aicore_ring_addr
     * to this so the AICore kernel entry can index by block_idx and feed
     * the per-core ring into the platform's set_l2_perf_aicore_ring().
     * Returns nullptr before initialize() succeeds.
     */
    void *get_aicore_ring_addr_table_device_ptr() const { return aicore_ring_addr_table_dev_; }

    /**
     * Read AICPU phase metadata that lives in AicpuPhaseHeader (not on the
     * buffer pipeline): the orchestrator summary and core→thread mapping.
     * Single-shot — must be called after stop() so orch_summary has settled.
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
     * @return Per-core L2PerfRecord vectors (indexed by core_index). For tests.
     */
    const std::vector<std::vector<L2PerfRecord>> &get_records() const { return collected_perf_records_; }

private:
    // Shared memory pointers. shm_host_ / device_id_ live on ProfilerBase
    // (set via set_memory_context in initialize()).
    void *perf_shared_mem_dev_{nullptr};
    bool was_registered_{false};

    // Standalone uint64_t[num_aicore] table holding per-core L2PerfAicoreRing
    // addresses. Allocated in initialize(), freed in finalize(). AICore reads
    // ring_table[block_idx] via KernelArgs::aicore_ring_addr.
    void *aicore_ring_addr_table_dev_{nullptr};

    int num_aicore_{0};
    int l2_swimlane_perf_level_{0};

    // Per-task output directory captured at initialize() time. Consumed by
    // export_swimlane_json() to build <prefix>/l2_perf_records.json.
    std::string output_prefix_;

    // Collected data (per-core vectors, indexed by core_index)
    std::vector<std::vector<L2PerfRecord>> collected_perf_records_;

    // AICPU phase profiling data (per-thread, mixed sched + orch records)
    std::vector<std::vector<AicpuPhaseRecord>> collected_phase_records_;
    AicpuOrchSummary collected_orch_summary_{};
    bool has_phase_data_{false};

    // Core-to-thread mapping (core_id → scheduler thread index, -1 = unassigned)
    std::vector<int8_t> core_to_thread_;

    // Running totals used at reconcile time to cross-check device-side counters.
    uint64_t total_perf_collected_{0};
    uint64_t total_phase_collected_{0};

    // Allocate a single buffer (L2PerfBuffer or PhaseBuffer) and register it
    void *alloc_single_buffer(size_t size, void **host_ptr_out);

    // Per-buffer-kind handlers used by on_buffer_collected.
    void copy_perf_buffer(const ReadyBufferInfo &info);
    void copy_phase_buffer(const ReadyBufferInfo &info);
};

#endif  // SRC_A2A3_PLATFORM_INCLUDE_HOST_L2_PERF_COLLECTOR_H_
