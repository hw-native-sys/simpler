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
 * @brief Host-side dep_gen (SubmitTrace) buffer allocation, streaming
 *        collection, and raw binary export.
 *
 * Architecture:
 * - BufferPoolManager<DepGenModule>: shared mgmt-thread infrastructure that
 *   polls the per-thread ready queue, drains the done_queue, and replenishes
 *   the (single instance's) free_queue from a unified recycled pool.
 * - DepGenCollector: collector thread pops full DepGenBuffers from the manager
 *   and appends their DepGenRecords to a binary file (submit_trace.bin).
 *
 * Lifecycle:
 *   init()                       — Allocate header + 1 BufferState + N DepGenBuffers
 *                                  (pre-fills free_queue; surplus → recycled pool).
 *                                  Calls set_memory_context() on the base.
 *   start(tf)                    — Inherited: launches mgmt + poll threads.
 *   [device execution]
 *   stop()                       — Inherited: drain queues, join threads.
 *   reconcile_counters()         — Sanity-check current_buf_ptr is cleared by
 *                                  AICPU flush, run collected+dropped==total
 *                                  cross-check. If dropped_record_count > 0,
 *                                  the host caller skips deps.json emission
 *                                  (incomplete graph; user gets a warning).
 *   finalize()                   — Free all device memory, unregister.
 *
 * Output format (submit_trace.bin): a fixed-size header followed by a
 * contiguous stream of DepGenRecord values. Replay (future PR) reads this
 * back. Layout intentionally trivial (no varint / framing) so the
 * `sizeof(DepGenRecord)` ABI in `common/dep_gen.h` is the only contract.
 */

#ifndef SRC_A2A3_PLATFORM_INCLUDE_HOST_DEP_GEN_COLLECTOR_H_
#define SRC_A2A3_PLATFORM_INCLUDE_HOST_DEP_GEN_COLLECTOR_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <optional>
#include <string>
#include <system_error>
#include <vector>

#include "common/dep_gen.h"
#include "common/platform_config.h"
#include "common/unified_log.h"
#include "host/profiling_common/profiler_base.h"

// ---------------------------------------------------------------------------
// dep_gen Module (drives BufferPoolManager<DepGenModule>)
// ---------------------------------------------------------------------------

/**
 * Internal hand-off struct delivered from the mgmt thread to the collector.
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
    static constexpr uint32_t kSlotCount = PLATFORM_DEP_GEN_SLOT_COUNT;
    static constexpr const char *kSubsystemName = "DepGenModule";

    /**
     * Buffers grown by proactive_replenish are batch-allocated up to the
     * per-instance ceiling minus the slot count.
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
    resolve_entry(void *shm, DataHeader * /*header*/, int q, const ReadyEntry &entry) {
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
// Memory operation callbacks (injected by DeviceRunner — same signatures as
// PmuAlloc/Register/Free, in keeping with subsystem conventions)
// ---------------------------------------------------------------------------

using DepGenAllocCallback = void *(*)(size_t size, void *user_data);
using DepGenRegisterCallback = int (*)(void *dev_ptr, size_t size, int device_id, void **host_ptr);
using DepGenUnregisterCallback = int (*)(void *dev_ptr, int device_id);
using DepGenFreeCallback = int (*)(void *dev_ptr, void *user_data);

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
     * the surplus go into BufferPoolManager's shared recycled pool.
     *
     * @param num_threads     Number of AICPU scheduling threads (so the
     *                        DataHeader sizes its per-thread ready queues)
     * @param submit_trace_path  Output file path (.bin)
     * @param alloc_cb        Memory allocation callback
     * @param register_cb     halHostRegister callback (nullptr in sim)
     * @param free_cb         Memory free callback
     * @param user_data       Opaque pointer forwarded to callbacks
     * @param device_id       Device ID
     * @return 0 on success, non-zero on failure
     */
    int init(
        int num_threads, DepGenAllocCallback alloc_cb, DepGenRegisterCallback register_cb, DepGenFreeCallback free_cb,
        void *user_data, int device_id
    );

    /**
     * Device pointer to the DepGenDataHeader. Set kernel_args.dep_gen_data_base
     * to this after init() so AICPU can find the shared memory via
     * set_platform_dep_gen_base().
     */
    void *get_dep_gen_shm_device_ptr() const { return shm_dev_; }

    /**
     * Per-buffer callback invoked by ProfilerBase's poll loop. Appends the
     * buffer's DepGenRecord entries to the in-memory ``records_`` vector
     * (no disk I/O — the host replay consumes that vector directly via
     * ``records()`` once the device run completes).
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
    void finalize(DepGenUnregisterCallback unregister_cb, DepGenFreeCallback free_cb, void *user_data);

    /**
     * @return true if init() succeeded and finalize() has not run.
     */
    bool is_initialized() const { return initialized_; }

    /**
     * Total DepGenRecords drained from the device-side ring buffer so far.
     */
    uint64_t total_collected() const { return total_collected_; }

    /**
     * In-memory record buffer (host replay's input). Valid between init()
     * and finalize(); pointer/size stay stable after stop() returns, which
     * is when the caller hands them to ``dep_gen_replay_emit_deps_json``.
     */
    const std::vector<DepGenRecord> &records() const { return records_; }

private:
    bool initialized_ = false;
    int num_threads_ = 0;

    // Shared memory region (DepGenDataHeader + DepGenBufferState[1]).
    // shm_host_ / device_id_ live on ProfilerBase (set via set_memory_context
    // in init()).
    void *shm_dev_ = nullptr;
    bool shm_registered_ = false;
    size_t shm_size_ = 0;

    bool buffers_registered_ = false;

    // In-memory record buffer — drained from the device ring on
    // on_buffer_collected() and consumed by the host replay directly (no
    // disk hop). Mutex serializes the mgmt thread's appends against the
    // (rare) reader on the same collector instance.
    std::vector<DepGenRecord> records_;
    std::mutex records_mutex_;

    // Running total of records appended. Equal to ``records_.size()`` after
    // every append; kept separately for the reconcile_counters cross-check
    // even when records_ may be inspected concurrently.
    uint64_t total_collected_ = 0;

    DepGenDataHeader *dep_gen_header() const { return get_dep_gen_header(shm_host_); }
    DepGenBufferState *dep_gen_state(int idx = 0) const { return get_dep_gen_buffer_state(shm_host_, idx); }

    void append_buffer_records(const void *buf_host_ptr);
};

/**
 * Build the ``deps.json`` output path under the caller-provided per-task
 * directory. Filename is fixed (no timestamp) — the directory is the
 * per-task uniqueness boundary, mirroring make_pmu_csv_path() and the now-
 * removed make_dep_gen_path() for submit_trace.bin (deps.json is the only
 * on-disk dep_gen artifact since the in-memory capture refactor).
 */
inline std::string make_deps_json_path(const std::string &output_dir) {
    // Use std::filesystem::path's operator/ for join — robust against trailing
    // slashes or path quirks that bare string concat would silently pass
    // through. The sibling make_pmu_csv_path / make_l2_perf_path still use
    // string concat; converting those is a follow-up cleanup since the
    // project's output_prefix paths come from scene_test.py's pathlib join
    // (never trailing-slashed in practice).
    std::filesystem::path dir(output_dir);
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) {
        LOG_WARN("Failed to create dep_gen output directory %s: %s", output_dir.c_str(), ec.message().c_str());
    }
    return (dir / "deps.json").string();
}

#endif  // SRC_A2A3_PLATFORM_INCLUDE_HOST_DEP_GEN_COLLECTOR_H_
