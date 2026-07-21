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
 * @file args_dump_collector.h
 * @brief Host-side args dump collector with independent shared memory.
 *
 * Architecture:
 * - BufferPoolManager<DumpModule>: shared split-mgmt infrastructure that
 *   polls per-thread ready queues, replenishes free_queues, and hands
 *   full DumpMetaBuffers off to collector thread shards.
 * - ArgsDumpCollector: copies tensor metadata + arena bytes into host
 *   vectors and writes the result to disk (.bin + JSON).
 *
 * a5 specifics: device↔host transfers use rtMemcpy / memcpy via
 * profiling_copy.h. The framework's mgmt loop mirrors the shm region per
 * tick; per-buffer payloads (metadata buffers) are pulled on demand inside
 * ProfilerAlgorithms. The collector additionally pulls arena bytes inside
 * on_buffer_collected, since arenas live outside the shm region and only
 * the part needed for the buffer's records is worth copying.
 */

#ifndef SRC_COMMON_PLATFORM_INCLUDE_HOST_ARGS_DUMP_COLLECTOR_H_
#define SRC_COMMON_PLATFORM_INCLUDE_HOST_ARGS_DUMP_COLLECTOR_H_

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <mutex>
#include <optional>
#include <queue>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "common/memory_barrier.h"
#include "common/platform_config.h"
#include "common/args_dump.h"
#include "common/unified_log.h"
#include "data_type.h"
#include "host/profiler_base.h"
#include "host/profiling_copy.h"

// ---------------------------------------------------------------------------
// Args Dump profiling Module (drives BufferPoolManager<DumpModule>)
// ---------------------------------------------------------------------------

/**
 * One buffer kind (DumpMetaBuffer); one ready_queue per AICPU thread.
 * Per-thread arena buffers are owned by the collector itself, not the
 * framework. Runtime refill uses the owning drain shard's local
 * recycled/done lanes; proactive_replenish may batch-allocate before drain
 * and collector threads start.
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
    static constexpr uint32_t kHostPoolQueueSize = PLATFORM_MAX_AICPU_THREADS * PLATFORM_DUMP_BUFFERS_PER_THREAD;
    static constexpr uint32_t kSlotCount = PLATFORM_DUMP_SLOT_COUNT;
    static constexpr const char *kSubsystemName = "DumpModule";
    // Producers are the scheduler threads, one per AICPU thread.
    static constexpr int kMaxCollectorThreads = PLATFORM_MAX_AICPU_THREADS;

    /**
     * Args-dump bursts can be very large; this is the startup-only batch
     * size used when proactive_replenish needs to grow recycled lanes.
     */
    static constexpr int batch_size(int /*kind*/) {
        constexpr int kBatch = PLATFORM_DUMP_BUFFERS_PER_THREAD - PLATFORM_DUMP_SLOT_COUNT;
        return kBatch < 1 ? 1 : kBatch;
    }

    static DataHeader *header_from_shm(void *shm) { return get_dump_header(shm); }

    static std::optional<profiling_common::EntrySite<DumpModule>>
    resolve_entry(void *shm, DataHeader *header, int /*q*/, const ReadyEntry &entry) {
        if (shm == nullptr || header == nullptr) {
            LOG_ERROR("DumpModule: invalid shared memory/header while resolving ready entry");
            return std::nullopt;
        }
        if (entry.thread_index >= header->num_dump_threads ||
            entry.thread_index >= static_cast<uint32_t>(PLATFORM_MAX_AICPU_THREADS)) {
            LOG_ERROR(
                "DumpModule: invalid ready entry thread=%u (num_dump_threads=%u, max=%u)", entry.thread_index,
                header->num_dump_threads, static_cast<uint32_t>(PLATFORM_MAX_AICPU_THREADS)
            );
            return std::nullopt;
        }
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

// Memory callbacks — thin aliases for the canonical profiling_common shapes.
// alloc / free are std::function so callers bind their MemoryAllocator via
// lambda capture; register / unregister stay as plain function pointers
// because they wrap stateless HAL globals. On a5 onboard the runner passes
// register_cb=nullptr and the framework installs a malloc-shadow + DMA
// fallback inline in ProfilerBase::start().
using DumpAllocCallback = profiling_common::ProfAllocCallback;
using DumpRegisterCallback = profiling_common::ProfRegisterCallback;
using DumpUnregisterCallback = profiling_common::ProfUnregisterCallback;
using DumpFreeCallback = profiling_common::ProfFreeCallback;

// =============================================================================
// ArgsDumpCollector
// =============================================================================

/**
 * Collected arg metadata + payload bytes
 */
struct DumpedArg {
    uint64_t task_id;
    int32_t func_ids[ARGS_DUMP_MAX_FUNC_IDS];  // task's active-subtask set (mix membership); -1 unknown
    int32_t func_count;                        // number of valid entries in func_ids
    uint32_t arg_index;
    ArgsDumpRole role;
    ArgsDumpStage stage;
    uint8_t dtype;
    uint8_t ndims;
    uint8_t flags;
    ArgsDumpKind kind;
    uint64_t scalar_value;
    uint64_t start_offset;                     // 1D element offset of the view origin
    uint32_t shapes[PLATFORM_DUMP_MAX_DIMS];   // Current view shape
    uint32_t strides[PLATFORM_DUMP_MAX_DIMS];  // Element stride per dim (> 0, type-enforced)
    bool is_contiguous;
    bool truncated;
    bool overwritten;
    uint64_t payload_size;
    uint64_t bin_offset;
    std::vector<uint8_t> bytes;
};

class ArgsDumpCollector : public profiling_common::ProfilerBase<ArgsDumpCollector, DumpModule> {
public:
    ArgsDumpCollector() = default;
    ~ArgsDumpCollector();

    ArgsDumpCollector(const ArgsDumpCollector &) = delete;
    ArgsDumpCollector &operator=(const ArgsDumpCollector &) = delete;

    // ProfilerBase contract
    static constexpr int kIdleTimeoutSec = PLATFORM_DUMP_TIMEOUT_SECONDS;
    static constexpr const char *kSubsystemName = "ArgsDump";

    /**
     * Initialize args dump shared memory.
     *
     * Allocates the DumpDataHeader + per-thread DumpBufferState array, the
     * per-thread arenas (single contiguous payload region per thread), and
     * the initial DumpMetaBuffers. The first PLATFORM_DUMP_SLOT_COUNT meta
     * buffers are pushed into each thread's free_queue; the rest go into
     * the BufferPoolManager's recycled pool.
     *
     * `output_prefix` is the per-task directory under which `args_dump/`
     * lands. Required (non-empty); CallConfig::validate() enforces this
     * upstream. Stored on the collector so the lazily-started writer thread
     * (kicked off inside on_buffer_collected) can derive its run_dir
     * without threading the prefix through the buffer-pool callback path.
     *
     * @param num_dump_threads  Number of AICPU scheduling threads
     * @param device_id         Device ID
     * @param alloc_cb          Memory allocation callback
     * @param register_cb       Host-visibility callback (nullptr on a5)
     * @param free_cb           Memory free callback
     * @param user_data         Opaque pointer forwarded to callbacks
     * @param output_prefix     Per-task directory; args_dump/ subdir lands here
     * @param dump_args_level OFF / PARTIAL (only Arg::dump()-marked args) /
     *                          FULL / FULL_JSON_ONLY (every task's metadata to
     *                          JSON, no payload or .bin). Written to
     *                          DumpDataHeader so the AICPU latches the mode
     *                          before any dispatch.
     * @return 0 on success, error code on failure
     */
    int initialize(
        int num_dump_threads, int device_id, const DumpAllocCallback &alloc_cb, DumpRegisterCallback register_cb,
        const DumpFreeCallback &free_cb, const std::string &output_prefix, DumpArgsLevel dump_args_level
    );

    void start(const profiling_common::ThreadFactory &thread_factory);

    /**
     * Per-buffer callback invoked by ProfilerBase's poll loop. Pulls the
     * relevant portion of the originating thread's arena from device, copies
     * tensor metadata + arena bytes into host-side DumpedArg records, and
     * queues payloads to the writer thread. The writer thread is started
     * lazily on the first invocation per run.
     */
    void on_buffer_collected(const DumpReadyBufferInfo &info, int collector_shard);

    /**
     * Write collected dumps to <output_prefix>/args_dump/{*.bin, *.json}.
     * Sorts args by (task_id, stage, arg_index, role).
     */
    int export_dump_files();

    /**
     * After stop():
     *   - Recover records from any non-empty DumpBufferState::current_buf_ptr
     *     left behind by abnormal exit before device-side flush ran.
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
    int finalize(DumpUnregisterCallback unregister_cb, const DumpFreeCallback &free_cb);

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
    struct alignas(64) CollectorShardCounters {
        uint64_t total_collected{0};
    };
    static_assert(
        sizeof(CollectorShardCounters) % 64 == 0, "CollectorShardCounters must not share cache lines across shards"
    );

    void *dump_shared_mem_dev_{nullptr};
    int num_dump_threads_{0};

    // Per-task output directory captured at initialize() time. The writer
    // thread builds run_dir_ = output_prefix_ / "args_dump" lazily on the
    // first on_buffer_collected.
    std::string output_prefix_;

    // Per-thread arena pointers (device + host shadow)
    struct ArenaInfo {
        void *dev_ptr{nullptr};
        void *host_ptr{nullptr};
        uint64_t size{0};
        uint64_t high_water{0};
    };
    std::vector<ArenaInfo> arenas_;

    // Merged dump args (metadata only; payloads live in args.bin).
    // Collector shards append to collected_by_collector_ on the hot path, then
    // export folds those shards into collected_ before sorting/writing JSON.
    std::vector<DumpedArg> collected_;
    std::vector<std::vector<DumpedArg>> collected_by_collector_;
    std::vector<CollectorShardCounters> collector_counters_;
    bool collector_shards_merged_{false};
    std::atomic<uint64_t> total_metadata_collected_{0};

    // Stats
    std::atomic<uint32_t> total_dropped_record_count_{0};
    std::atomic<uint32_t> total_truncated_count_{0};
    std::atomic<uint32_t> total_overwrite_count_{0};

    // Run-scoped state for the writer thread (lazily started on first
    // on_buffer_collected and joined by export_dump_files).
    std::chrono::steady_clock::time_point run_start_time_;
    std::atomic<int64_t> last_progress_ms_{0};
    bool writer_started_{false};

    size_t normalize_collector_shard(int collector_shard) const;
    void reset_collector_shards();
    void merge_collector_shards();
    void process_dump_buffer(const DumpReadyBufferInfo &info, int collector_shard);
    void start_writer_thread_once();

    // Writer thread: streams arg payloads to a single args.bin
    std::thread writer_thread_;
    std::mutex writer_start_mutex_;
    std::mutex write_mutex_;
    std::condition_variable write_cv_;
    std::queue<DumpedArg> write_queue_;
    std::atomic<bool> writer_done_{false};

    // Resolved dump level; FULL_JSON_ONLY suppresses the .bin file entirely.
    DumpArgsLevel dump_args_level_{DumpArgsLevel::OFF};

    // Output directory and single binary file
    std::filesystem::path run_dir_;
    std::ofstream bin_file_;
    uint64_t next_bin_offset_{0};

    // Writer stats
    std::atomic<uint64_t> bytes_written_{0};

    void writer_loop();
};

#endif  // SRC_COMMON_PLATFORM_INCLUDE_HOST_ARGS_DUMP_COLLECTOR_H_
