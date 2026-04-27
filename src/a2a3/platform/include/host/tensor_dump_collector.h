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
 * @brief Host-side tensor dump collector with independent shared memory
 *
 * Fully decoupled from profiling: uses its own shared memory region,
 * ready queues, and memory manager thread.
 *
 * Architecture:
 * - DumpMemoryManager: Background thread that polls dump ready queues,
 *   recycles metadata buffers, and hands off full buffers to the main thread.
 * - TensorDumpCollector: Main thread copies tensor data from arenas,
 *   manages lifecycle, and exports dump files.
 */

#ifndef SRC_A2A3_PLATFORM_INCLUDE_HOST_TENSOR_DUMP_COLLECTOR_H_
#define SRC_A2A3_PLATFORM_INCLUDE_HOST_TENSOR_DUMP_COLLECTOR_H_

#include <atomic>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/platform_config.h"
#include "common/tensor_dump.h"
#include "data_type.h"

/**
 * Memory allocation callback for tensor dump buffers and shared memory.
 *
 * @param size Memory size in bytes
 * @param user_data Opaque allocator context
 * @return Allocated device memory pointer, or nullptr on failure
 */
using DumpAllocCallback = void *(*)(size_t size, void *user_data);

/**
 * Memory registration callback for host-visible shared memory mappings.
 *
 * @param dev_ptr Device memory pointer
 * @param size Memory size in bytes
 * @param device_id Device ID
 * @param user_data Opaque allocator context
 * @param[out] host_ptr Host-mapped pointer
 * @return 0 on success, error code on failure
 */
using DumpRegisterCallback = int (*)(void *dev_ptr, size_t size, int device_id, void *user_data, void **host_ptr);

/**
 * Memory unregister callback.
 *
 * @param dev_ptr Device memory pointer
 * @param device_id Device ID
 * @param user_data Opaque allocator context
 * @return 0 on success, error code on failure
 */
using DumpUnregisterCallback = int (*)(void *dev_ptr, int device_id, void *user_data);

/**
 * Memory free callback.
 *
 * @param dev_ptr Device memory pointer
 * @param user_data Opaque allocator context
 * @return 0 on success, error code on failure
 */
using DumpFreeCallback = int (*)(void *dev_ptr, void *user_data);

/**
 * Callback for binding the memory-manager thread to a device context.
 *
 * @param device_id Device ID
 * @param user_data Opaque allocator context
 * @return 0 on success, error code on failure
 */
using DumpSetDeviceCallback = int (*)(int device_id, void *user_data);

// =============================================================================
// DumpMemoryManager - Background Thread
// =============================================================================

/**
 * Information about a ready (full) dump metadata buffer
 */
struct DumpReadyBufferInfo {
    uint32_t thread_index;
    void *dev_buffer_ptr;
    void *host_buffer_ptr;
    uint32_t buffer_seq;
};

/**
 * Dump buffer memory manager thread.
 *
 * Polls per-thread ready queues in DumpDataHeader, hands off full
 * DumpMetaBuffers to the main thread, and recycles them back into
 * the SPSC free_queue.
 */
class DumpMemoryManager {
public:
    DumpMemoryManager() = default;
    ~DumpMemoryManager();

    DumpMemoryManager(const DumpMemoryManager &) = delete;
    DumpMemoryManager &operator=(const DumpMemoryManager &) = delete;

    friend class TensorDumpCollector;

    void start(
        void *shared_mem_host, int num_dump_threads, DumpAllocCallback alloc_cb, DumpRegisterCallback register_cb,
        DumpFreeCallback free_cb, void *user_data, int device_id, DumpSetDeviceCallback set_device_cb = nullptr
    );

    void stop();

    bool try_pop_ready(DumpReadyBufferInfo &info);
    bool wait_pop_ready(DumpReadyBufferInfo &info, std::chrono::milliseconds timeout);
    void notify_copy_done(void *dev_buffer_ptr);

    bool is_running() const { return running_.load(); }

private:
    std::thread mgmt_thread_;
    std::atomic<bool> running_{false};

    void *shared_mem_host_{nullptr};
    int num_dump_threads_{0};

    DumpAllocCallback alloc_cb_{nullptr};
    DumpRegisterCallback register_cb_{nullptr};
    DumpFreeCallback free_cb_{nullptr};
    DumpSetDeviceCallback set_device_cb_{nullptr};
    void *user_data_{nullptr};
    int device_id_{-1};

    std::mutex ready_mutex_;
    std::condition_variable ready_cv_;
    std::queue<DumpReadyBufferInfo> ready_queue_;

    std::mutex done_mutex_;
    std::queue<void *> done_queue_;  // Device pointers to recycle

    std::unordered_map<void *, void *> dev_to_host_;
    std::vector<void *> recycled_dump_buffers_;

    void mgmt_loop();
    void *alloc_and_register(size_t size, void **host_ptr_out);
    void free_buffer(void *dev_ptr);
    void *resolve_host_ptr(void *dev_ptr);
    void register_mapping(void *dev_ptr, void *host_ptr);
    void process_dump_entry(DumpDataHeader *header, int thread_idx, const DumpReadyQueueEntry &entry);
};

// =============================================================================
// TensorDumpCollector - Main Collector
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

class TensorDumpCollector {
public:
    TensorDumpCollector() = default;
    ~TensorDumpCollector();

    TensorDumpCollector(const TensorDumpCollector &) = delete;
    TensorDumpCollector &operator=(const TensorDumpCollector &) = delete;

    /**
     * Initialize tensor dump shared memory.
     *
     * Allocates DumpDataHeader + DumpBufferState array, per-thread arenas,
     * and initial DumpMetaBuffers.
     *
     * @return 0 on success, error code on failure
     */
    int initialize(
        int num_dump_threads, int device_id, DumpAllocCallback alloc_cb, DumpRegisterCallback register_cb,
        DumpFreeCallback free_cb, void *user_data, DumpSetDeviceCallback set_device_cb = nullptr
    );

    void start_memory_manager();
    // `output_prefix` is the per-task directory under which tensor_dump/ lands.
    // Required (non-empty); CallConfig::validate() enforces this upstream.
    void poll_and_collect(const std::string &output_prefix);
    int export_dump_files();
    void stop_memory_manager();
    void drain_remaining_buffers();
    void scan_remaining_dump_buffers();
    void signal_execution_complete();

    int finalize(DumpUnregisterCallback unregister_cb, DumpFreeCallback free_cb, void *user_data);

    bool is_initialized() const { return dump_shared_mem_host_ != nullptr; }

    void *get_dump_shm_device_ptr() const { return dump_shared_mem_dev_; }

private:
    void *dump_shared_mem_dev_{nullptr};
    void *dump_shared_mem_host_{nullptr};
    bool was_registered_{false};
    int device_id_{-1};
    int num_dump_threads_{0};

    DumpAllocCallback alloc_cb_{nullptr};
    DumpRegisterCallback register_cb_{nullptr};
    DumpFreeCallback free_cb_{nullptr};
    DumpSetDeviceCallback set_device_cb_{nullptr};
    void *user_data_{nullptr};

    // Per-thread arena pointers
    struct ArenaInfo {
        void *dev_ptr{nullptr};
        void *host_ptr{nullptr};
        uint64_t size{0};
        uint64_t high_water{0};  // For overwrite detection
    };
    std::vector<ArenaInfo> arenas_;

    DumpMemoryManager memory_manager_;

    // Collected dump tensors
    std::vector<DumpedTensor> collected_;
    std::mutex collected_mutex_;

    // Execution complete signal
    std::atomic<bool> execution_complete_{false};

    // Stats
    uint32_t total_dropped_record_count_{0};
    uint32_t total_truncated_count_{0};
    uint32_t total_overwrite_count_{0};

    void *alloc_single_buffer(size_t size, void **host_ptr_out);
    void process_dump_buffer(const DumpReadyBufferInfo &info);

    // Track processed buffer pointers to prevent double-processing
    std::unordered_set<void *> processed_buffers_;

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
