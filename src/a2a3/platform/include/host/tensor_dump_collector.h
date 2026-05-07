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
 * @file tensor_dump_collector.h
 * @brief Host-side tensor dump collector with independent shared memory.
 *
 * Architecture:
 * - BufferPoolManager<DumpModule>: shared mgmt-thread infrastructure that polls
 *   per-thread DumpReadyQueues, replenishes free_queues, and hands full
 *   DumpMetaBuffers off to the collector thread.
 * - TensorDumpCollector: copies tensor metadata + arena bytes into host vectors
 *   and writes the result to disk (.bin + JSON).
 */

#ifndef SRC_A2A3_PLATFORM_INCLUDE_HOST_TENSOR_DUMP_COLLECTOR_H_
#define SRC_A2A3_PLATFORM_INCLUDE_HOST_TENSOR_DUMP_COLLECTOR_H_

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include "common/memory_barrier.h"
#include "common/platform_config.h"
#include "common/tensor_dump.h"
#include "common/unified_log.h"
#include "data_type.h"
#include "host/profiling_common/profiler_base.h"

// ---------------------------------------------------------------------------
// Tensor Dump profiling Module (drives BufferPoolManager<DumpModule>)
// ---------------------------------------------------------------------------

/**
 * One buffer kind (DumpMetaBuffer); one ready_queue per AICPU thread.
 * Per-thread arena buffers are owned by the collector itself, not the
 * framework. process_entry refills the originating thread's free_queue with
 * exactly one buffer; proactive_replenish tops up to SLOT_COUNT and
 * batch-allocates when the recycled pool drains.
 */

/**
 * Information about a ready (full) dump metadata buffer.
 */
struct DumpReadyBufferInfo {
    uint32_t thread_index;
    void *dev_buffer_ptr;
    void *host_buffer_ptr;
    uint32_t buffer_seq;
};

struct DumpModule {
    using DataHeader = DumpDataHeader;
    using ReadyEntry = DumpReadyQueueEntry;
    using ReadyBufferInfo = ::DumpReadyBufferInfo;
    using FreeQueue = DumpFreeQueue;

    static constexpr int kBufferKinds = 1;
    static constexpr uint32_t kReadyQueueSize = PLATFORM_DUMP_READYQUEUE_SIZE;
    static constexpr uint32_t kSlotCount = PLATFORM_DUMP_SLOT_COUNT;
    static constexpr const char *kSubsystemName = "DumpModule";

    /**
     * Tensor-dump bursts can be very large; the batch is sized so a fully
     * empty recycled pool refills to the configured per-thread ceiling in
     * one tick.
     */
    static constexpr int batch_size(int /*kind*/) {
        constexpr int kBatch = PLATFORM_DUMP_BUFFERS_PER_THREAD - PLATFORM_DUMP_SLOT_COUNT;
        return kBatch < 1 ? 1 : kBatch;
    }

    static DataHeader *header_from_shm(void *shm) { return get_dump_header(shm); }

    static std::optional<profiling_common::EntrySite<DumpModule>>
    resolve_entry(void *shm, DataHeader * /*header*/, int /*q*/, const ReadyEntry &entry) {
        DumpBufferState *state = get_dump_buffer_state(shm, static_cast<int>(entry.thread_index));
        profiling_common::EntrySite<DumpModule> site;
        site.kind = 0;
        site.free_queue = &state->free_queue;
        site.buffer_size = sizeof(DumpMetaBuffer);
        site.info.thread_index = entry.thread_index;
        site.info.dev_buffer_ptr = reinterpret_cast<void *>(entry.buffer_ptr);
        site.info.host_buffer_ptr = nullptr;  // filled by ProfilerAlgorithms
        site.info.buffer_seq = entry.buffer_seq;
        return site;
    }

    template <typename Cb>
    static void for_each_instance(void *shm, DataHeader *header, Cb &&cb) {
        const int n_threads = static_cast<int>(header->num_dump_threads);
        for (int t = 0; t < n_threads; t++) {
            DumpBufferState *state = get_dump_buffer_state(shm, t);
            cb(/*kind=*/0, &state->free_queue, sizeof(DumpMetaBuffer));
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
using DumpAllocCallback = void *(*)(size_t size, void *user_data);

/**
 * Register device memory for host-visible access. nullptr in sim mode
 * (dev pointer is directly host-accessible). Stateless: HAL state is global,
 * so no user_data is threaded through.
 *
 * @param dev_ptr        Device memory pointer
 * @param size           Bytes
 * @param device_id      Device ID
 * @param[out] host_ptr  Host-mapped pointer
 * @return 0 on success, error code on failure
 */
using DumpRegisterCallback = int (*)(void *dev_ptr, size_t size, int device_id, void **host_ptr);

/**
 * Unregister a previously registered host-visible mapping. May be nullptr.
 * Stateless (see register_cb).
 */
using DumpUnregisterCallback = int (*)(void *dev_ptr, int device_id);

/**
 * Free device memory.
 */
using DumpFreeCallback = int (*)(void *dev_ptr, void *user_data);

// =============================================================================
// TensorDumpCollector
// =============================================================================

/**
 * Collected tensor metadata + payload bytes
 */
struct DumpedTensor {
    uint64_t task_id;
    uint8_t subtask_id;
    uint32_t func_id;
    uint32_t arg_index;
    TensorDumpRole role;
    TensorDumpStage stage;
    uint8_t dtype;
    uint8_t ndims;
    uint32_t raw_shapes[PLATFORM_DUMP_MAX_DIMS];
    uint32_t shapes[PLATFORM_DUMP_MAX_DIMS];
    uint32_t offsets[PLATFORM_DUMP_MAX_DIMS];
    bool is_contiguous;
    bool truncated;
    bool overwritten;
    uint64_t payload_size;  // original payload size (bytes may be cleared after writing)
    uint64_t bin_offset;    // byte offset into tensors.bin
    std::vector<uint8_t> bytes;
};

class TensorDumpCollector : public profiling_common::ProfilerBase<TensorDumpCollector, DumpModule> {
public:
    TensorDumpCollector() = default;
    ~TensorDumpCollector();

    TensorDumpCollector(const TensorDumpCollector &) = delete;
    TensorDumpCollector &operator=(const TensorDumpCollector &) = delete;

    // ProfilerBase contract
    static constexpr int kIdleTimeoutSec = PLATFORM_DUMP_TIMEOUT_SECONDS;
    static constexpr const char *kSubsystemName = "TensorDump";

    /**
     * Initialize tensor dump shared memory.
     *
     * Allocates the DumpDataHeader + per-thread DumpBufferState array, the
     * per-thread arenas (single contiguous payload region per thread), and
     * the initial DumpMetaBuffers. The first PLATFORM_DUMP_SLOT_COUNT meta
     * buffers are pushed into each thread's free_queue; the rest go into
     * the BufferPoolManager's recycled pool.
     *
     * `output_prefix` is the per-task directory under which tensor_dump/
     * lands. Required (non-empty); CallConfig::validate() enforces this
     * upstream. Stored on the collector so the lazily-started writer thread
     * (kicked off inside on_buffer_collected) can derive its run_dir without
     * threading the prefix through the buffer-pool callback path.
     *
     * @param num_dump_threads  Number of AICPU scheduling threads
     * @param device_id         Device ID
     * @param alloc_cb          Memory allocation callback
     * @param register_cb       Host-visibility callback (nullptr for sim)
     * @param free_cb           Memory free callback
     * @param user_data         Opaque pointer forwarded to callbacks
     * @param output_prefix     Per-task directory; tensor_dump/ subdir lands here
     * @return 0 on success, error code on failure
     */
    int initialize(
        int num_dump_threads, int device_id, DumpAllocCallback alloc_cb, DumpRegisterCallback register_cb,
        DumpFreeCallback free_cb, void *user_data, const std::string &output_prefix
    );

    /**
     * Per-buffer callback invoked by ProfilerBase's poll loop. Copies tensor
     * metadata + arena bytes into host-side DumpedTensor records and queues
     * payloads to the writer thread. The writer thread is started lazily on
     * the first invocation per run.
     */
    void on_buffer_collected(const DumpReadyBufferInfo &info);

    /**
     * Write collected dumps to <output_prefix>/tensor_dump/{*.bin, *.json}.
     * Sorts tensors by (task_id, subtask_id, func_id, stage, arg_index, role).
     */
    int export_dump_files();

    /**
     * After stop(), perform purely-passive accounting:
     *   - LOG_ERROR any non-zero DumpBufferState::current_buf_ptr with
     *     records (device flush should always succeed-or-bump-dropped, so
     *     a non-empty leftover indicates an AICPU flush bug — host does
     *     NOT recover, to avoid masking the bug).
     *   - Accumulate device-side dropped_record_count into
     *     total_dropped_record_count_ for the final anomaly report.
     * Must be called after stop().
     */
    void reconcile_counters();

    /**
     * Free all device memory and unregister mappings (per-thread arenas,
     * DumpMetaBuffers held by the framework or still in per-pool free
     * queues). Idempotent on a collector that was never initialized.
     */
    int finalize(DumpUnregisterCallback unregister_cb, DumpFreeCallback free_cb, void *user_data);

    /**
     * @return true if initialize() succeeded and finalize() has not run.
     */
    bool is_initialized() const { return shm_host_ != nullptr; }

    /**
     * Device pointer to the DumpDataHeader. Set kernel_args.dump_data_base
     * to this after initialize() succeeds so the AICPU side can find the
     * shared memory.
     */
    void *get_dump_shm_device_ptr() const { return dump_shared_mem_dev_; }

private:
    void *dump_shared_mem_dev_{nullptr};
    bool was_registered_{false};
    int num_dump_threads_{0};

    // Per-task output directory captured at initialize() time. The writer
    // thread builds run_dir_ = output_prefix_ / "tensor_dump" lazily on the
    // first on_buffer_collected.
    std::string output_prefix_;

    // Per-thread arena pointers
    struct ArenaInfo {
        void *dev_ptr{nullptr};
        void *host_ptr{nullptr};
        uint64_t size{0};
        uint64_t high_water{0};  // For overwrite detection
    };
    std::vector<ArenaInfo> arenas_;

    // Collected dump tensors
    std::vector<DumpedTensor> collected_;
    std::mutex collected_mutex_;

    // Stats
    uint32_t total_dropped_record_count_{0};
    uint32_t total_truncated_count_{0};
    uint32_t total_overwrite_count_{0};

    // Run-scoped state for the writer thread (lazily started on first
    // on_buffer_collected and joined by export_dump_files).
    std::chrono::steady_clock::time_point run_start_time_;
    std::chrono::steady_clock::time_point last_progress_time_;
    uint64_t buffers_collected_{0};
    bool writer_started_{false};

    void *alloc_single_buffer(size_t size, void **host_ptr_out);
    void process_dump_buffer(const DumpReadyBufferInfo &info);
    void start_writer_thread_once();

    // Writer thread: streams tensor payloads to a single tensors.bin
    std::thread writer_thread_;
    std::mutex write_mutex_;
    std::condition_variable write_cv_;
    std::queue<DumpedTensor> write_queue_;
    std::atomic<bool> writer_done_{false};

    // Output directory and single binary file
    std::filesystem::path run_dir_;
    std::ofstream bin_file_;
    uint64_t next_bin_offset_{0};  // only accessed by collect thread

    // Writer stats
    std::atomic<uint64_t> bytes_written_{0};

    void writer_loop();
};

#endif  // SRC_A2A3_PLATFORM_INCLUDE_HOST_TENSOR_DUMP_COLLECTOR_H_
