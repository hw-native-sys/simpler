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
 * @brief Platform-agnostic performance data collector with dynamic memory management
 *
 * Architecture:
 * - ProfMemoryManager: Dedicated thread that polls ReadyQueue, allocates new
 *   device buffers, and replaces full buffers in slot arrays.
 * - L2PerfCollector: Main thread collects data from ProfMemoryManager's
 *   internal queue, copies records to host vectors, and exports results.
 *
 * Design Pattern: Dependency Injection via Callbacks for memory operations.
 */

#ifndef SRC_A2A3_PLATFORM_INCLUDE_HOST_L2_PERF_COLLECTOR_H_
#define SRC_A2A3_PLATFORM_INCLUDE_HOST_L2_PERF_COLLECTOR_H_

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "common/l2_perf_profiling.h"
#include "common/platform_config.h"
#include "runtime.h"

/**
 * Memory allocation callback for performance profiling
 *
 * @param size Memory size in bytes
 * @return Allocated device memory pointer, or nullptr on failure
 */
using L2PerfAllocCallback = void *(*)(size_t size);

/**
 * Memory registration callback (for Host-Device shared memory)
 *
 * @param dev_ptr Device memory pointer
 * @param size Memory size in bytes
 * @param device_id Device ID
 * @param[out] host_ptr Host-mapped pointer
 * @return 0 on success, error code on failure
 */
using L2PerfRegisterCallback = int (*)(void *dev_ptr, size_t size, int device_id, void **host_ptr);

/**
 * Memory unregister callback
 *
 * @param dev_ptr Device memory pointer
 * @param device_id Device ID
 * @return 0 on success, error code on failure
 */
using L2PerfUnregisterCallback = int (*)(void *dev_ptr, int device_id);

/**
 * Memory free callback
 *
 * @param dev_ptr Device memory pointer
 * @return 0 on success, error code on failure
 */
using L2PerfFreeCallback = int (*)(void *dev_ptr);

/**
 * Thread factory callback for creating threads with device binding
 *
 * Creates a std::thread that runs the given function with proper device
 * context setup/teardown (e.g., rtSetDevice/rtDeviceReset on onboard).
 *
 * @param fn Function to execute in the new thread
 * @return The newly created thread
 */
using ThreadFactory = std::function<std::thread(std::function<void()>)>;

// =============================================================================
// ProfMemoryManager - Dynamic Buffer Memory Management Thread
// =============================================================================

/**
 * Buffer type identifier for ReadyBufferInfo
 */
enum class ProfBufferType { PERF_RECORD, PHASE };

/**
 * Information about a ready (full) buffer, passed from mgmt thread to main thread
 */
struct ReadyBufferInfo {
    ProfBufferType type;
    uint32_t index;         // core_index (PERF_RECORD) or thread_idx (PHASE)
    uint32_t slot_idx;      // Reserved (unused in free queue design)
    void *dev_buffer_ptr;   // Device address of the full buffer
    void *host_buffer_ptr;  // Host-mapped address (sim: same as dev)
    uint32_t buffer_seq;    // Sequence number for ordering
};

/**
 * Notification that a buffer has been copied and can be freed
 */
struct CopyDoneInfo {
    void *dev_buffer_ptr;  // Device buffer to free
    ProfBufferType type;   // Buffer type (for recycling)
};

/**
 * Dynamic profiling buffer memory manager
 *
 * Runs a dedicated thread that:
 * 1. Polls ReadyQueue in shared memory for full buffer entries
 * 2. Allocates new device buffers via callback
 * 3. Writes new buffer addresses into slots (for device to pick up)
 * 4. Pushes old (full) buffer info to internal queue for main thread to copy
 * 5. Frees device buffers after main thread confirms copy is done
 */
class ProfMemoryManager {
public:
    ProfMemoryManager() = default;
    ~ProfMemoryManager();

    // Disable copy
    ProfMemoryManager(const ProfMemoryManager &) = delete;
    ProfMemoryManager &operator=(const ProfMemoryManager &) = delete;

    // Allow L2PerfCollector to register initial buffer mappings
    friend class L2PerfCollector;

    /**
     * Start the memory management thread
     *
     * @param shared_mem_host Host-mapped shared memory base address
     * @param num_cores Number of AICore instances (L2PerfBufferState count)
     * @param num_phase_threads Number of phase profiling threads (PhaseBufferState count)
     * @param alloc_cb Device memory allocation callback
     * @param register_cb Host-device mapping callback (nullptr for simulation)
     * @param free_cb Device memory free callback
     * @param device_id Device ID for registration
     * @param thread_factory Thread factory for creating device-bound threads (empty to use std::thread)
     */
    void start(
        void *shared_mem_host, int num_cores, int num_phase_threads, L2PerfAllocCallback alloc_cb,
        L2PerfRegisterCallback register_cb, L2PerfFreeCallback free_cb, int device_id,
        const ThreadFactory &thread_factory = {}
    );

    /**
     * Stop the memory management thread
     * Blocks until the thread exits.
     */
    void stop();

    /**
     * Try to pop a ready buffer info (non-blocking)
     *
     * @param[out] info Ready buffer info
     * @return true if an item was available, false otherwise
     */
    bool try_pop_ready(ReadyBufferInfo &info);

    /**
     * Wait for a ready buffer info with timeout
     *
     * @param[out] info Ready buffer info
     * @param timeout Maximum wait time
     * @return true if an item was available, false on timeout
     */
    bool wait_pop_ready(ReadyBufferInfo &info, std::chrono::milliseconds timeout);

    /**
     * Notify that a buffer has been copied and can be freed
     *
     * @param info Copy done notification
     */
    void notify_copy_done(const CopyDoneInfo &info);

    /**
     * Check if the manager thread is running
     */
    bool is_running() const { return running_.load(); }

private:
    std::thread mgmt_thread_;
    std::atomic<bool> running_{false};

    // Shared memory references
    void *shared_mem_host_{nullptr};
    int num_cores_{0};
    int num_phase_threads_{0};

    // Callbacks
    L2PerfAllocCallback alloc_cb_{nullptr};
    L2PerfRegisterCallback register_cb_{nullptr};
    L2PerfFreeCallback free_cb_{nullptr};
    int device_id_{-1};

    // Management thread → main thread (ready buffers)
    std::mutex ready_mutex_;
    std::condition_variable ready_cv_;
    std::queue<ReadyBufferInfo> ready_queue_;

    // Main thread → management thread (buffers to free)
    std::mutex done_mutex_;
    std::queue<CopyDoneInfo> done_queue_;

    // Device-to-host pointer mapping (populated during alloc_and_register)
    std::unordered_map<void *, void *> dev_to_host_;

    // Recycled buffer pools (avoids alloc/free churn in mgmt_loop)
    std::vector<void *> recycled_perf_buffers_;
    std::vector<void *> recycled_phase_buffers_;

    // Management thread main loop
    void mgmt_loop();

    // Allocate a new buffer and optionally register for host access
    void *alloc_and_register(size_t size, void **host_ptr_out);

    // Free a previously allocated buffer
    void free_buffer(void *dev_ptr);

    // Resolve device pointer to host pointer
    void *resolve_host_ptr(void *dev_ptr);

    // Register an external dev→host mapping (for initial buffers)
    void register_mapping(void *dev_ptr, void *host_ptr);

    // Process one ReadyQueue entry
    void process_ready_entry(L2PerfDataHeader *header, int thread_idx, const ReadyQueueEntry &entry);
};

// =============================================================================
// L2PerfCollector - Main Collector
// =============================================================================

/**
 * Performance data collector
 *
 * Manages performance profiling lifecycle:
 * 1. Initialize shared memory (Header + SlotArrays) and allocate initial buffers
 * 2. Start ProfMemoryManager thread
 * 3. Collect records from ProfMemoryManager's queue (main thread)
 * 4. Export swimlane visualization
 *
 * Platform-agnostic: Memory management delegated to callbacks
 */
class L2PerfCollector {
public:
    L2PerfCollector() = default;
    ~L2PerfCollector();

    // Disable copy and move
    L2PerfCollector(const L2PerfCollector &) = delete;
    L2PerfCollector &operator=(const L2PerfCollector &) = delete;

    /**
     * Initialize performance profiling
     *
     * Allocates shared memory for slot arrays, allocates initial buffers,
     * and writes buffer addresses into slots.
     *
     * @param runtime Runtime instance to configure
     * @param num_aicore Number of AICore instances
     * @param device_id Device ID
     * @param alloc_cb Memory allocation callback
     * @param register_cb Memory registration callback (can be nullptr for simulation)
     * @param free_cb Memory free callback
     * @return 0 on success, error code on failure
     */
    int initialize(
        int num_aicore, int device_id, L2PerfAllocCallback alloc_cb, L2PerfRegisterCallback register_cb,
        L2PerfFreeCallback free_cb
    );

    /**
     * Start the memory management thread
     *
     * Must be called after initialize() and before device execution starts.
     *
     * @param thread_factory Thread factory for creating device-bound threads (empty to use std::thread)
     */
    void start_memory_manager(const ThreadFactory &thread_factory = {});

    /**
     * Poll and collect performance data from the memory manager's queue
     *
     * Runs on the main thread (or a dedicated collector thread in sim mode).
     * Pulls ready buffers from ProfMemoryManager, copies records to host vectors,
     * and notifies the manager to free old device buffers.
     *
     * @param expected_tasks Expected total number of tasks (0 = auto-detect)
     */
    void poll_and_collect(int expected_tasks = 0);

    /**
     * Export performance data to Chrome Trace Event Format
     *
     * @param output_path Output directory path
     * @return 0 on success, error code on failure
     */
    int export_swimlane_json(const std::string &output_path);

    /**
     * Stop the memory management thread and clean up remaining data
     *
     * Must be called after device execution completes.
     */
    void stop_memory_manager();

    /**
     * Cleanup all resources
     *
     * @param unregister_cb Memory unregister callback (can be nullptr)
     * @param free_cb Memory free callback
     * @return 0 on success, error code on failure
     */
    int finalize(L2PerfUnregisterCallback unregister_cb, L2PerfFreeCallback free_cb);

    /**
     * Check if collector is initialized
     */
    bool is_initialized() const { return perf_shared_mem_host_ != nullptr; }

    /**
     * Get the device pointer to the L2PerfDataHeader.
     * Used to set kernel_args.l2_perf_data_base after initialize() succeeds.
     */
    void *get_l2_perf_setup_device_ptr() const { return perf_shared_mem_dev_; }

    /**
     * Drain remaining buffers from the memory manager's ready queue
     *
     * After poll_and_collect() exits (all PERF records collected) and
     * the memory manager is stopped, Phase buffers may still be in the
     * ready queue. This method drains them into the collected vectors.
     *
     * Must be called after stop_memory_manager() and before collect_phase_data().
     */
    void drain_remaining_buffers();

    /**
     * Collect AICPU phase profiling data from shared memory
     *
     * Reads scheduler phase records and orchestrator summary from the
     * phase profiling region. Must be called after AICPU threads have joined.
     */
    void collect_phase_data();

    /**
     * Scan L2PerfBufferState::current_buf_ptr for all cores to recover
     * partial records not delivered through the pipeline.
     *
     * Must be called after device execution completes and after
     * stop_memory_manager(). Follows the same pattern as collect_phase_data()
     * for PhaseBufferStates.
     */
    void scan_remaining_perf_buffers();

    /**
     * Signal that device execution is complete (streams synchronized).
     * poll_and_collect() will drain remaining pipeline data and exit.
     */
    void signal_execution_complete();

    /**
     * Get collected records (for testing)
     */
    const std::vector<std::vector<L2PerfRecord>> &get_records() const { return collected_perf_records_; }

private:
    // Shared memory pointers
    void *perf_shared_mem_dev_{nullptr};   // Device memory pointer (slot arrays)
    void *perf_shared_mem_host_{nullptr};  // Host-mapped pointer (slot arrays)
    bool was_registered_{false};           // True if register_cb was called successfully
    int device_id_{-1};

    // Configuration
    int num_aicore_{0};

    // Callbacks (stored for memory manager)
    L2PerfAllocCallback alloc_cb_{nullptr};
    L2PerfRegisterCallback register_cb_{nullptr};
    L2PerfFreeCallback free_cb_{nullptr};

    // Memory manager
    ProfMemoryManager memory_manager_;

    // Collected data (per-core vectors, indexed by core_index)
    std::vector<std::vector<L2PerfRecord>> collected_perf_records_;

    // AICPU phase profiling data (per-thread, mixed sched + orch records)
    std::vector<std::vector<AicpuPhaseRecord>> collected_phase_records_;
    AicpuOrchSummary collected_orch_summary_{};
    bool has_phase_data_{false};

    // Core-to-thread mapping (core_id → scheduler thread index, -1 = unassigned)
    std::vector<int8_t> core_to_thread_;

    // Signal from device_runner that execution is complete
    std::atomic<bool> execution_complete_{false};

    // Allocate a single buffer (L2PerfBuffer or PhaseBuffer) and register it
    void *alloc_single_buffer(size_t size, void **host_ptr_out);
};

#endif  // SRC_A2A3_PLATFORM_INCLUDE_HOST_L2_PERF_COLLECTOR_H_
