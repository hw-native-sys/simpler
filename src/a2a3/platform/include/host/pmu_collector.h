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
 * - BufferPoolManager<PmuModule>: shared mgmt-thread infrastructure that polls
 *   per-thread DumpReadyQueues, drains the done_queue, and replenishes the
 *   per-core free_queues from a unified recycled pool.
 * - PmuCollector: collector thread pops full PmuBuffers from the manager
 *   and appends them to the CSV file.
 *
 * Lifecycle:
 *   init()                       — Allocate header + per-core states + PmuBuffers
 *                                  (pre-fills free_queues; rest go into the
 *                                  recycled pool). Calls set_memory_context()
 *                                  on the base so start(tf) can launch threads.
 *   start(tf)                    — Inherited from ProfilerBase: assembles
 *                                  MemoryOps from the stashed callbacks and
 *                                  launches the mgmt + poll threads.
 *   [device execution]
 *   stop()                       — Stop mgmt → join mgmt → signal poll →
 *                                  drain L2 → join poll, in that order. On
 *                                  return both thread exits and queue drains
 *                                  are complete.
 *   reconcile_counters()         — Sanity-check PmuBufferState::current_buf_ptr
 *                                  (any non-zero pointer with records is a
 *                                  device-flush bug, logged as ERROR) and run
 *                                  the device-side cross-check:
 *                                  collected + dropped == total.
 *   finalize()                   — Free all device memory and unregister.
 */

#ifndef SRC_A2A3_PLATFORM_INCLUDE_HOST_PMU_COLLECTOR_H_
#define SRC_A2A3_PLATFORM_INCLUDE_HOST_PMU_COLLECTOR_H_

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "common/memory_barrier.h"
#include "common/platform_config.h"
#include "common/pmu_profiling.h"
#include "common/unified_log.h"
#include "host/profiling_common/profiler_base.h"

// ---------------------------------------------------------------------------
// PMU profiling Module (drives BufferPoolManager<PmuModule>)
// ---------------------------------------------------------------------------

/**
 * One buffer kind (PmuBuffer); per-core buffer states. The collector
 * pre-allocates PLATFORM_PMU_BUFFERS_PER_CORE buffers per core at init time
 * to absorb steady-state load. process_entry refills the originating core's
 * free_queue with exactly one buffer (recycled → drain done → alloc), and
 * proactive_replenish tops up to SLOT_COUNT with a batch alloc when the
 * recycled pool drains.
 */

/**
 * Internal hand-off struct delivered from the mgmt thread to the collector.
 * thread_index is the logical AICPU thread queue the entry was popped from,
 * passed through by ProfilerBase's mgmt loop.
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
    static constexpr uint32_t kSlotCount = PLATFORM_PMU_SLOT_COUNT;
    static constexpr const char *kSubsystemName = "PmuModule";

    /**
     * Buffers grown by proactive_replenish are batch-allocated up to the
     * configured per-core ceiling minus the slot count, so a double-empty
     * (recycled + done both dry) recovers in one tick.
     */
    static constexpr int batch_size(int /*kind*/) {
        constexpr int kBatch = PLATFORM_PMU_BUFFERS_PER_CORE - PLATFORM_PMU_SLOT_COUNT;
        return kBatch < 1 ? 1 : kBatch;
    }

    static DataHeader *header_from_shm(void *shm) { return get_pmu_header(shm); }

    /**
     * `count` is intentionally NOT reset here — AICPU is the sole writer and
     * resets it itself on flush/drop/pop.
     */
    static std::optional<profiling_common::EntrySite<PmuModule>>
    resolve_entry(void *shm, DataHeader * /*header*/, int q, const ReadyEntry &entry) {
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

/**
 * Allocate device memory.
 *
 * @param size       Bytes to allocate
 * @param user_data  Opaque allocator context
 * @return Device pointer, or nullptr on failure
 */
using PmuAllocCallback = void *(*)(size_t size, void *user_data);

/**
 * Register device memory for host-visible access. nullptr in sim mode
 * (dev pointer is directly host-accessible). Stateless: HAL state is global,
 * so no user_data is threaded through.
 */
using PmuRegisterCallback = int (*)(void *dev_ptr, size_t size, int device_id, void **host_ptr);

/**
 * Unregister a previously registered host-visible mapping. May be nullptr.
 * Stateless (see register_cb).
 */
using PmuUnregisterCallback = int (*)(void *dev_ptr, int device_id);

/**
 * Free device memory.
 */
using PmuFreeCallback = int (*)(void *dev_ptr, void *user_data);

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
     * PLATFORM_PMU_SLOT_COUNT buffers per core are pushed directly into that
     * core's free_queue; the surplus go into the BufferPoolManager's shared
     * recycled pool (no per-core affinity — any core can borrow from the pool
     * via the mgmt thread's proactive_replenish).
     *
     * @param num_cores                    Number of AICore instances in use
     * @param num_threads                  Number of AICPU scheduling threads
     * @param csv_path                     Output CSV path
     * @param event_type                   PmuEventType selector (written to
     *                                     PmuDataHeader::event_type so AICPU
     *                                     can configure the HW counters)
     * @param alloc_cb / register_cb / free_cb  Memory operation callbacks
     *                                          (register_cb nullptr in sim)
     * @param user_data                    Opaque pointer forwarded to callbacks
     * @param device_id                    Device ID (for register_cb)
     * @return 0 on success, non-zero on failure
     */
    int init(
        int num_cores, int num_threads, const std::string &csv_path, PmuEventType event_type, PmuAllocCallback alloc_cb,
        PmuRegisterCallback register_cb, PmuFreeCallback free_cb, void *user_data, int device_id
    );

    /**
     * Device pointer to the PmuDataHeader. Set kernel_args.pmu_data_base
     * to this after init() succeeds so the AICPU side can find the
     * shared memory.
     */
    void *get_pmu_shm_device_ptr() const { return shm_dev_; }

    /**
     * Per-buffer callback invoked by ProfilerBase's poll loop. Flushes
     * records to CSV.
     */
    void on_buffer_collected(const PmuReadyBufferInfo &info);

    /**
     * After stop(), perform purely-passive accounting:
     *   - LOG_ERROR any non-zero PmuBufferState::current_buf_ptr with
     *     records (device flush should always succeed-or-bump-dropped, so
     *     a non-empty leftover indicates an AICPU flush bug — host does
     *     NOT recover, to avoid masking the bug).
     *   - Run the device-side cross-check:
     *     collected + dropped == device_total.
     * Must be called after stop(), so the AICPU-side flush has settled.
     */
    void reconcile_counters();

    /**
     * Free all device memory and unregister mappings. Idempotent.
     */
    void finalize(PmuUnregisterCallback unregister_cb, PmuFreeCallback free_cb, void *user_data);

    /**
     * @return true if init() succeeded and finalize() has not run.
     */
    bool is_initialized() const { return initialized_; }

private:
    bool initialized_ = false;
    int num_cores_ = 0;
    int num_threads_ = 0;
    PmuEventType event_type_{PmuEventType::PIPE_UTILIZATION};

    // Shared memory region (PmuDataHeader + PmuBufferState[]). shm_host_ /
    // device_id_ live on ProfilerBase (set via set_memory_context in init()).
    void *shm_dev_ = nullptr;
    bool shm_registered_ = false;
    size_t shm_size_ = 0;

    // True iff the per-buffer register_cb path was used at init time. Drives
    // unregister-vs-free-only behavior in finalize.
    bool buffers_registered_ = false;

    // CSV output. File is opened lazily on the first record write so that a
    // hung device run that produces no records does not leave a header-only
    // CSV on disk.
    std::string csv_path_;
    std::string csv_header_;
    std::ofstream csv_file_;
    std::mutex csv_mutex_;

    // Running total of records written to CSV. Used at drain time to verify
    // collected + device-side dropped == device-side total.
    uint64_t total_collected_ = 0;

    PmuDataHeader *pmu_header() const { return get_pmu_header(shm_host_); }
    PmuBufferState *pmu_state(int core_id) const { return get_pmu_buffer_state(shm_host_, core_id); }

    void write_buffer_to_csv(int core_id, int thread_idx, const void *buf_host_ptr);
    void ensure_csv_open_unlocked();
};

// ---------------------------------------------------------------------------
// Utility: resolve PMU event type (env-var override)
// ---------------------------------------------------------------------------

inline PmuEventType resolve_pmu_event_type(int requested_event_type) {
    PmuEventType resolved = PmuEventType::PIPE_UTILIZATION;
    if (requested_event_type > 0 &&
        pmu_resolve_event_config_a2a3(static_cast<PmuEventType>(requested_event_type)) != nullptr) {
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
    if (val > 0 && pmu_resolve_event_config_a2a3(static_cast<PmuEventType>(val)) != nullptr) {
        resolved = static_cast<PmuEventType>(val);
        LOG_INFO_V0("PMU event type set to %u from SIMPLER_PMU_EVENT_TYPE", static_cast<uint32_t>(resolved));
        return resolved;
    }
    LOG_WARN("Invalid SIMPLER_PMU_EVENT_TYPE=%s, using default (PIPE_UTILIZATION=%u)", pmu_env, PMU_EVENT_TYPE_DEFAULT);
    return resolved;
}

/**
 * Build the CSV path under the caller-provided per-task directory. Filename is
 * fixed (no timestamp) — the directory is the per-task uniqueness boundary.
 */
inline std::string make_pmu_csv_path(const std::string &output_dir) {
    std::error_code ec;
    std::filesystem::create_directories(output_dir, ec);
    if (ec) {
        LOG_WARN("Failed to create PMU output directory %s: %s", output_dir.c_str(), ec.message().c_str());
    }
    return output_dir + "/pmu.csv";
}

#endif  // SRC_A2A3_PLATFORM_INCLUDE_HOST_PMU_COLLECTOR_H_
