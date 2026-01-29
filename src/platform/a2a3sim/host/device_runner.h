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

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "function_cache.h"
#include "kernel_args.h"
#include "memory_allocator.h"
#include "runtime.h"

/**
 * Mapped kernel binary in executable memory
 *
 * Stores kernel .text binary in mmap'd executable memory, allowing
 * direct function pointer invocation. This mirrors the real device
 * behavior where kernel binaries are stored in GM.
 */
struct MappedKernel {
    void* exec_mem{nullptr};     // mmap'd executable memory
    size_t size{0};              // Size of mapped region
    uint64_t func_addr{0};       // Function pointer address (same as exec_mem)
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
     * Cleanup all resources
     *
     * @return 0 on success
     */
    int finalize();

    /**
     * Register a kernel for a func_id
     *
     * Copies the kernel .text binary into mmap'd executable memory,
     * enabling direct function pointer invocation. This mirrors the
     * real device behavior where kernel binaries are stored in GM.
     *
     * @param func_id   Function identifier
     * @param bin_data  Kernel .text section binary data
     * @param bin_size  Size of binary data in bytes
     * @return 0 on success
     */
    int register_kernel(int func_id, const uint8_t* bin_data, size_t bin_size);

    /**
     * Get function_bin_addr for a given func_id
     *
     * @param func_id  Function identifier
     * @return Function pointer address, or 0 if not found
     */
    uint64_t get_function_bin_addr(int func_id);

private:
    DeviceRunner() = default;
    ~DeviceRunner();

    // Configuration
    int device_id_{-1};
    int block_dim_{0};
    int cores_per_blockdim_{3};
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
    void (*aicore_execute_func_)(Runtime*, int, int){nullptr};
    std::string aicpu_so_path_;
    std::string aicore_so_path_;

    // Private helper methods
    int ensure_device_initialized(int device_id,
                                  const std::vector<uint8_t>& aicpu_so_binary,
                                  const std::vector<uint8_t>& aicore_kernel_binary);
    int ensure_binaries_loaded(const std::vector<uint8_t>& aicpu_so_binary,
                               const std::vector<uint8_t>& aicore_kernel_binary);
};

#endif  // RUNTIME_DEVICERUNNER_H
