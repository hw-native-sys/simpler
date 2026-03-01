/**
 * @file performance_collector.h
 * @brief Platform-agnostic performance data collector
 *
 * This module provides a unified interface for collecting performance profiling
 * data across different platforms (a2a3, a2a3sim). Platform-specific memory
 * management is delegated to callback functions, enabling code reuse while
 * maintaining platform separation.
 *
 * Design Pattern: Dependency Injection via Callbacks
 * - Memory allocation: PerfAllocCallback
 * - Host-Device mapping: PerfRegisterCallback (optional for simulation)
 * - Memory cleanup: PerfUnregisterCallback + PerfFreeCallback
 */

#ifndef PLATFORM_HOST_PERFORMANCE_COLLECTOR_H_
#define PLATFORM_HOST_PERFORMANCE_COLLECTOR_H_

#include <string>
#include <vector>

#include "common/perf_profiling.h"
#include "common/platform_config.h"
#include "runtime.h"

/**
 * Memory allocation callback for performance profiling
 *
 * @param size Memory size in bytes
 * @param user_data User-provided context pointer
 * @return Allocated device memory pointer, or nullptr on failure
 */
using PerfAllocCallback = void* (*)(size_t size, void* user_data);

/**
 * Memory registration callback (for Host-Device shared memory)
 *
 * @param dev_ptr Device memory pointer
 * @param size Memory size in bytes
 * @param device_id Device ID
 * @param user_data User-provided context pointer
 * @param[out] host_ptr Host-mapped pointer
 * @return 0 on success, error code on failure
 */
using PerfRegisterCallback = int (*)(void* dev_ptr, size_t size, int device_id,
                                      void* user_data, void** host_ptr);

/**
 * Memory unregister callback
 *
 * @param dev_ptr Device memory pointer
 * @param device_id Device ID
 * @param user_data User-provided context pointer
 * @return 0 on success, error code on failure
 */
using PerfUnregisterCallback = int (*)(void* dev_ptr, int device_id, void* user_data);

/**
 * Memory free callback
 *
 * @param dev_ptr Device memory pointer
 * @param user_data User-provided context pointer
 * @return 0 on success, error code on failure
 */
using PerfFreeCallback = int (*)(void* dev_ptr, void* user_data);

/**
 * Performance data collector
 *
 * Manages performance profiling lifecycle:
 * 1. Initialize shared memory (Header + DoubleBuffers)
 * 2. Poll ready queue and collect records from full buffers
 * 3. Print statistics and export swimlane visualization
 *
 * Platform-agnostic: Memory management delegated to callbacks
 */
class PerformanceCollector {
public:
    PerformanceCollector() = default;
    ~PerformanceCollector();

    // Disable copy and move
    PerformanceCollector(const PerformanceCollector&) = delete;
    PerformanceCollector& operator=(const PerformanceCollector&) = delete;

    /**
     * Initialize performance profiling
     *
     * Allocates and initializes shared memory for performance data collection.
     *
     * @param runtime Runtime instance to configure
     * @param num_aicore Number of AICore instances
     * @param device_id Device ID
     * @param alloc_cb Memory allocation callback
     * @param register_cb Memory registration callback (can be nullptr for simulation)
     * @param user_data User-provided context pointer passed to callbacks
     * @return 0 on success, error code on failure
     */
    int initialize(Runtime& runtime,
                   int num_aicore,
                   int device_id,
                   PerfAllocCallback alloc_cb,
                   PerfRegisterCallback register_cb,
                   void* user_data);

    /**
     * Poll and collect performance data from shared memory
     *
     * @param expected_tasks Expected total number of tasks (0 = auto-detect from header)
     */
    void poll_and_collect(int expected_tasks = 0);

    /**
     * Export performance data to Chrome Trace Event Format
     *
     * @param output_path Output directory path
     * @return 0 on success, error code on failure
     */
    int export_swimlane_json(const std::string& output_path = "outputs");

    /**
     * Cleanup all resources
     *
     * @param unregister_cb Memory unregister callback (can be nullptr)
     * @param free_cb Memory free callback
     * @param user_data User-provided context pointer
     * @return 0 on success, error code on failure
     */
    int finalize(PerfUnregisterCallback unregister_cb,
                 PerfFreeCallback free_cb,
                 void* user_data);

    /**
     * Check if collector is initialized
     */
    bool is_initialized() const { return perf_shared_mem_host_ != nullptr; }

    /**
     * Get collected records (for testing)
     */
    const std::vector<std::vector<PerfRecord>>& get_records() const { return collected_perf_records_; }

private:
    // Shared memory pointers
    void* perf_shared_mem_dev_{nullptr};   // Device memory pointer
    void* perf_shared_mem_host_{nullptr};  // Host-mapped pointer
    bool was_registered_{false};           // True if register_cb was called successfully
    int device_id_{-1};

    // Configuration
    int num_aicore_{0};

    // Collected data (per-core vectors, indexed by core_index)
    std::vector<std::vector<PerfRecord>> collected_perf_records_;
};

#endif  // PLATFORM_HOST_PERFORMANCE_COLLECTOR_H_
