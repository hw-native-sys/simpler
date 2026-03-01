/**
 * Device Runner - Thread-Based Simulation
 *
 * This module simulates the Ascend AICPU/AICore execution model using threads.
 * It provides the SAME interface as the real a2a3 DeviceRunner, ensuring
 * API compatibility with Python bindings and examples.
 *
 * Key differences from real a2a3:
 * - Uses host memory instead of device memory
 * - Uses std::thread instead of CANN kernel launches
 * - Kernel .text binaries are loaded into executable memory (mmap)
 */

#ifndef RUNTIME_DEVICERUNNER_H
#define RUNTIME_DEVICERUNNER_H

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#include "common/core_type.h"
#include "common/kernel_args.h"
#include "common/memory_barrier.h"
#include "common/perf_profiling.h"
#include "common/platform_config.h"
#include "common/unified_log.h"
#include "host/function_cache.h"
#include "host/memory_allocator.h"
#include "host/performance_collector.h"
#include "runtime.h"

/**
 * Mapped kernel binary loaded via dlopen
 *
 * Stores dlopen handle and function pointer address. This enables
 * proper handling of external symbols (e.g., std::exp) via PLT/GOT.
 */
struct MappedKernel {
    void* dl_handle{nullptr};    // dlopen handle
    uint64_t func_addr{0};       // Function pointer address (from dlsym)
};

/**
 * Device runner singleton for simulated kernel execution
 *
 * This class provides the SAME interface as the real a2a3 DeviceRunner,
 * but implements execution using host threads instead of actual device
 * kernel launches.
 *
 * Key simulation features:
 * - Memory operations use host memory (malloc/free/memcpy)
 * - Kernel execution uses std::thread
 * - Kernel .text binaries are loaded into mmap'd executable memory
 */
class DeviceRunner {
public:
    /**
     * Get singleton instance
     */
    static DeviceRunner& get();

    /**
     * Allocate tensor memory (host memory in simulation)
     *
     * @param bytes  Size of tensor in bytes
     * @return Pointer on success, nullptr on failure
     */
    void* allocate_tensor(size_t bytes);

    /**
     * Free tensor memory
     *
     * @param dev_ptr  Pointer to free
     */
    void free_tensor(void* dev_ptr);

    /**
     * Copy data (memcpy in simulation)
     *
     * @param dev_ptr   Destination pointer
     * @param host_ptr  Source pointer
     * @param bytes     Number of bytes to copy
     * @return 0 on success
     */
    int copy_to_device(void* dev_ptr, const void* host_ptr, size_t bytes);

    /**
     * Copy data (memcpy in simulation)
     *
     * @param host_ptr  Destination pointer
     * @param dev_ptr   Source pointer
     * @param bytes     Number of bytes to copy
     * @return 0 on success
     */
    int copy_from_device(void* host_ptr, const void* dev_ptr, size_t bytes);

    /**
     * Execute a runtime using threads
     *
     * This method simulates the complete execution:
     * 1. Initializes worker handshake buffers
     * 2. Sets function_bin_addr for all tasks
     * 3. Launches AICPU threads
     * 4. Launches AICore threads
     * 5. Waits for all threads to complete
     *
     * @param runtime              Runtime to execute
     * @param block_dim            Number of blocks (1 block = 1 AIC + 2 AIV)
     * @param device_id            Device ID (ignored in simulation)
     * @param aicpu_so_binary      AICPU binary (ignored in simulation)
     * @param aicore_kernel_binary AICore binary (ignored in simulation)
     * @param launch_aicpu_num     Number of AICPU threads
     * @return 0 on success
     */
    int run(Runtime& runtime,
            int block_dim,
            int device_id,
            const std::vector<uint8_t>& aicpu_so_binary,
            const std::vector<uint8_t>& aicore_kernel_binary,
            int launch_aicpu_num = 1);

    /**
     * Print handshake results
     */
    void print_handshake_results();

    /**
     * Poll and collect performance data from shared memory
     *
     * Polls the ready queue and collects performance records from full buffers.
     * This is a synchronous polling function that should be called after
     * launching kernels but before thread synchronization.
     *
     * @param expected_tasks Expected total number of tasks (used for exit condition)
     */
    void poll_and_collect_performance_data(int expected_tasks);

    /**
     * Export performance data to merged_swimlane.json
     *
     * Converts collected performance records to Chrome Trace Event Format
     * and writes to outputs/merged_swimlane_<timestamp>.json for visualization in Perfetto.
     * Should be called after execution completes.
     *
     * @param output_path Path to output directory (default: "outputs")
     * @return 0 on success, error code on failure
     */
    int export_swimlane_json(const std::string& output_path = "outputs");

    /**
     * Clean cached resources - lightweight cleanup between tests
     *
     * Cleans up test-specific resources while preserving device resources for reuse:
     * - Closes dlopen'd kernel libraries (different tests may have different kernels)
     * - Clears kernel address cache
     *
     * @return 0 on success, error code on failure
     */
    int clean_cache();

    /**
     * Cleanup all resources
     *
     * Use this for final cleanup when no more tests will run.
     *
     * @return 0 on success
     */
    int finalize();

    /**
     * Upload a kernel binary and return the function address
     *
     * Loads the complete kernel .so via dlopen, enabling proper handling
     * of external symbols (e.g., std::exp, std::log) via PLT/GOT.
     * Uses dlsym to resolve the unified entry point "kernel_entry".
     *
     * If the kernel is already uploaded (same func_id), returns the
     * cached address without re-uploading.
     *
     * @param func_id      Function identifier (for caching)
     * @param bin_data     Complete kernel .so binary data
     * @param bin_size     Size of binary data in bytes
     * @return Function pointer address on success, 0 on error
     */
    uint64_t upload_kernel_binary(int func_id, const uint8_t* bin_data, size_t bin_size);

private:
    DeviceRunner() = default;
    ~DeviceRunner();

    // Configuration
    int device_id_{-1};
    int block_dim_{0};
    int cores_per_blockdim_{PLATFORM_CORES_PER_BLOCKDIM};
    int worker_count_{0};

    // Memory management
    MemoryAllocator mem_alloc_;

    // Simulation state (no actual device resources)
    KernelArgs kernel_args_;

    // Kernel binary mapping (func_id -> executable memory)
    std::map<int, MappedKernel> func_id_to_addr_;

    // Runtime pointer for print_handshake_results
    Runtime* last_runtime_{nullptr};

    // Dynamically loaded executor libraries and function pointers
    void* aicpu_so_handle_{nullptr};
    void* aicore_so_handle_{nullptr};
    int (*aicpu_execute_func_)(Runtime*){nullptr};
    void (*aicore_execute_func_)(Runtime*, int, CoreType, uint32_t, uint64_t){nullptr};
    void (*set_platform_regs_func_)(uint64_t){nullptr};
    std::string aicpu_so_path_;
    std::string aicore_so_path_;

    // Performance profiling
    PerformanceCollector perf_collector_;

    // Private helper methods
    int ensure_device_initialized(int device_id,
                                  const std::vector<uint8_t>& aicpu_so_binary,
                                  const std::vector<uint8_t>& aicore_kernel_binary);
    int ensure_binaries_loaded(const std::vector<uint8_t>& aicpu_so_binary,
                               const std::vector<uint8_t>& aicore_kernel_binary);

    /**
     * Initialize performance profiling shared memory
     *
     * Allocates and initializes host memory for performance profiling.
     *
     * @param runtime Runtime instance to configure
     * @param num_aicore Number of cores
     * @param device_id Device ID (ignored in simulation)
     * @return 0 on success, error code on failure
     */
    int init_performance_profiling(Runtime& runtime, int num_aicore, int device_id);
};

#endif  // RUNTIME_DEVICERUNNER_H
