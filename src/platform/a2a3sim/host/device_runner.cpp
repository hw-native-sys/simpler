/**
 * Device Runner Implementation - Thread-Based Simulation
 *
 * This file implements the simulated device execution using host threads.
 * It provides the same API as the real a2a3 implementation but uses
 * std::thread instead of CANN runtime APIs.
 *
 * aicpu_execute and aicore_execute_wrapper are loaded dynamically via dlopen from
 * the binaries passed to launch_runtime.
 */

#include "device_runner.h"

#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <thread>
#include <unistd.h>
#include <vector>
#include <sys/mman.h>

#include "runtime.h"

// Function pointer types for dynamically loaded executors
typedef int (*aicpu_execute_func_t)(Runtime* runtime);
typedef void (*aicore_execute_func_t)(Runtime* runtime, int block_idx, int core_type);

// =============================================================================
// DeviceRunner Implementation
// =============================================================================

DeviceRunner& DeviceRunner::get() {
    static DeviceRunner runner;
    return runner;
}

DeviceRunner::~DeviceRunner() {
    finalize();
}

int DeviceRunner::ensure_device_initialized(int device_id,
                                            const std::vector<uint8_t>& aicpu_so_binary,
                                            const std::vector<uint8_t>& aicore_kernel_binary) {
    device_id_ = device_id;
    return ensure_binaries_loaded(aicpu_so_binary, aicore_kernel_binary);
}

int DeviceRunner::ensure_binaries_loaded(const std::vector<uint8_t>& aicpu_so_binary,
                                         const std::vector<uint8_t>& aicore_kernel_binary) {
    // Skip if already loaded
    if (aicpu_execute_func_ != nullptr && aicore_execute_func_ != nullptr) {
        return 0;
    }

    // Write AICPU binary to temp file and dlopen
    if (!aicpu_so_binary.empty() && aicpu_execute_func_ == nullptr) {
        aicpu_so_path_ = "/tmp/aicpu_sim_" + std::to_string(getpid()) + ".so";
        std::ofstream ofs(aicpu_so_path_, std::ios::binary);
        if (!ofs) {
            std::cerr << "Error: Failed to create temp file for AICPU SO: " << aicpu_so_path_ << '\n';
            return -1;
        }
        ofs.write(reinterpret_cast<const char*>(aicpu_so_binary.data()), aicpu_so_binary.size());
        ofs.close();

        aicpu_so_handle_ = dlopen(aicpu_so_path_.c_str(), RTLD_NOW | RTLD_GLOBAL);
        if (aicpu_so_handle_ == nullptr) {
            std::cerr << "Error: dlopen failed for AICPU SO: " << dlerror() << '\n';
            return -1;
        }

        aicpu_execute_func_ = reinterpret_cast<int(*)(Runtime*)>(dlsym(aicpu_so_handle_, "aicpu_execute"));
        if (aicpu_execute_func_ == nullptr) {
            std::cerr << "Error: dlsym failed for aicpu_execute: " << dlerror() << '\n';
            return -1;
        }
        std::cout << "DeviceRunner(sim): Loaded aicpu_execute from " << aicpu_so_path_ << '\n';
    }

    // Write AICore binary to temp file and dlopen
    if (!aicore_kernel_binary.empty() && aicore_execute_func_ == nullptr) {
        aicore_so_path_ = "/tmp/aicore_sim_" + std::to_string(getpid()) + ".so";
        std::ofstream ofs(aicore_so_path_, std::ios::binary);
        if (!ofs) {
            std::cerr << "Error: Failed to create temp file for AICore SO: " << aicore_so_path_ << '\n';
            return -1;
        }
        ofs.write(reinterpret_cast<const char*>(aicore_kernel_binary.data()), aicore_kernel_binary.size());
        ofs.close();

        aicore_so_handle_ = dlopen(aicore_so_path_.c_str(), RTLD_NOW | RTLD_GLOBAL);
        if (aicore_so_handle_ == nullptr) {
            std::cerr << "Error: dlopen failed for AICore SO: " << dlerror() << '\n';
            return -1;
        }

        aicore_execute_func_ = reinterpret_cast<void(*)(Runtime*, int, int)>(dlsym(aicore_so_handle_, "aicore_execute_wrapper"));
        if (aicore_execute_func_ == nullptr) {
            std::cerr << "Error: dlsym failed for aicore_execute_wrapper: " << dlerror() << '\n';
            return -1;
        }
        std::cout << "DeviceRunner(sim): Loaded aicore_execute_wrapper from " << aicore_so_path_ << '\n';
    }

    return 0;
}

void* DeviceRunner::allocate_tensor(size_t bytes) {
    return mem_alloc_.alloc(bytes);
}

void DeviceRunner::free_tensor(void* dev_ptr) {
    if (dev_ptr != nullptr) {
        mem_alloc_.free(dev_ptr);
    }
}

int DeviceRunner::copy_to_device(void* dev_ptr, const void* host_ptr, size_t bytes) {
    // In simulation, this is just a memcpy
    std::memcpy(dev_ptr, host_ptr, bytes);
    return 0;
}

int DeviceRunner::copy_from_device(void* host_ptr, const void* dev_ptr, size_t bytes) {
    // In simulation, this is just a memcpy
    std::memcpy(host_ptr, dev_ptr, bytes);
    return 0;
}

int DeviceRunner::run(Runtime& runtime,
                      int block_dim,
                      int device_id,
                      const std::vector<uint8_t>& aicpu_so_binary,
                      const std::vector<uint8_t>& aicore_kernel_binary,
                      int launch_aicpu_num) {
    // Ensure device is initialized
    int rc = ensure_device_initialized(device_id, aicpu_so_binary, aicore_kernel_binary);
    if (rc != 0) {
        std::cerr << "Error: ensure_device_initialized failed: " << rc << '\n';
        return rc;
    }

    // Calculate execution parameters
    block_dim_ = block_dim;
    int num_cores = block_dim * cores_per_blockdim_;

    if (num_cores > RUNTIME_MAX_WORKER) {
        std::cerr << "Error: num_cores (" << num_cores << ") exceeds RUNTIME_MAX_WORKER ("
                  << RUNTIME_MAX_WORKER << ")\n";
        return -1;
    }

    // Initialize handshake buffers
    runtime.worker_count = num_cores;
    worker_count_ = num_cores;
    runtime.block_dim = block_dim;
    runtime.sche_cpu_num = launch_aicpu_num;

    // Calculate number of AIC cores
    int num_aic = block_dim;

    for (int i = 0; i < num_cores; i++) {
        runtime.workers[i].aicpu_ready = 0;
        runtime.workers[i].aicore_done = 0;
        runtime.workers[i].control = 0;
        runtime.workers[i].task = 0;
        runtime.workers[i].task_status = 0;
        // First 1/3 are AIC (0), remaining 2/3 are AIV (1)
        runtime.workers[i].core_type = (i < num_aic) ? 0 : 1;
    }

    // Set function_bin_addr for all tasks
    std::cout << "\n=== Setting function_bin_addr for Tasks (Simulation) ===" << '\n';
    for (int i = 0; i < runtime.get_task_count(); i++) {
        Task* task = runtime.get_task(i);
        if (task != nullptr) {
            uint64_t addr = get_function_bin_addr(task->func_id);
            task->function_bin_addr = addr;
            std::cout << "  Task " << i << " (func_id=" << task->func_id
                      << ") -> function_bin_addr=0x" << std::hex << addr << std::dec << '\n';
        }
    }
    std::cout << '\n';

    // Store runtime pointer for print_handshake_results
    last_runtime_ = &runtime;

    // Check if executors are loaded
    if (aicpu_execute_func_ == nullptr || aicore_execute_func_ == nullptr) {
        std::cerr << "Error: Executor functions not loaded. Call ensure_binaries_loaded first.\n";
        return -1;
    }

    // Launch AICPU threads
    std::cout << "=== Launching " << launch_aicpu_num << " AICPU thread(s) ===" << '\n';
    std::vector<std::thread> aicpu_threads;
    for (int i = 0; i < launch_aicpu_num; i++) {
        aicpu_threads.emplace_back([this, &runtime]() {
            aicpu_execute_func_(&runtime);
        });
    }

    // Launch AICore threads
    std::cout << "=== Launching " << num_cores << " AICore thread(s) ===" << '\n';
    std::vector<std::thread> aicore_threads;
    for (int i = 0; i < num_cores; i++) {
        int core_type = runtime.workers[i].core_type;
        aicore_threads.emplace_back([this, &runtime, i, core_type]() {
            aicore_execute_func_(&runtime, i, core_type);
        });
    }

    // Wait for all threads to complete
    std::cout << "=== Waiting for threads to complete ===" << '\n';
    for (auto& t : aicpu_threads) {
        t.join();
    }
    for (auto& t : aicore_threads) {
        t.join();
    }

    std::cout << "=== All threads completed ===" << '\n';
    return 0;
}

void DeviceRunner::print_handshake_results() {
    if (worker_count_ == 0 || last_runtime_ == nullptr) {
        return;
    }

    std::cout << "\nHandshake results for " << worker_count_ << " cores:" << std::endl;
    for (int i = 0; i < worker_count_; i++) {
        std::cout << "  Core " << i
                  << ": aicore_done=" << last_runtime_->workers[i].aicore_done
                  << " aicpu_ready=" << last_runtime_->workers[i].aicpu_ready
                  << " control=" << last_runtime_->workers[i].control
                  << " task=" << last_runtime_->workers[i].task << std::endl;
    }
}

int DeviceRunner::finalize() {
    // Skip if already finalized
    if (device_id_ == -1 && aicpu_so_handle_ == nullptr && aicore_so_handle_ == nullptr) {
        return 0;
    }

    // Print handshake results before cleanup
    print_handshake_results();

    // Unmap all kernel executable memory regions
    for (auto& pair : func_id_to_addr_) {
        MappedKernel& kernel = pair.second;
        if (kernel.exec_mem != nullptr && kernel.exec_mem != MAP_FAILED) {
            munmap(kernel.exec_mem, kernel.size);
            kernel.exec_mem = nullptr;
            kernel.size = 0;
            kernel.func_addr = 0;
        }
    }
    func_id_to_addr_.clear();

    // Close dynamically loaded libraries and remove temp files
    if (aicpu_so_handle_ != nullptr) {
        dlclose(aicpu_so_handle_);
        aicpu_so_handle_ = nullptr;
        aicpu_execute_func_ = nullptr;
    }
    if (!aicpu_so_path_.empty()) {
        std::remove(aicpu_so_path_.c_str());
        aicpu_so_path_.clear();
    }

    if (aicore_so_handle_ != nullptr) {
        dlclose(aicore_so_handle_);
        aicore_so_handle_ = nullptr;
        aicore_execute_func_ = nullptr;
    }
    if (!aicore_so_path_.empty()) {
        std::remove(aicore_so_path_.c_str());
        aicore_so_path_.clear();
    }

    // Free all remaining allocations
    mem_alloc_.finalize();

    device_id_ = -1;
    worker_count_ = 0;
    last_runtime_ = nullptr;

    std::cout << "DeviceRunner(sim) finalized\n";
    return 0;
}

// =============================================================================
// Kernel Registration (Executable Memory Mapping)
// =============================================================================

int DeviceRunner::register_kernel(int func_id, const uint8_t* bin_data, size_t bin_size) {
    if (bin_data == nullptr || bin_size == 0) {
        std::cerr << "Error: Invalid kernel data\n";
        return -1;
    }

    // Skip if already registered
    if (func_id_to_addr_.find(func_id) != func_id_to_addr_.end()) {
        std::cout << "Kernel func_id=" << func_id << " already registered, skipping\n";
        return 0;
    }

    MappedKernel kernel;

    if (bin_size == sizeof(uint64_t)) {
        // Legacy mode: bin_data contains a function pointer (used by C++ example)
        uint64_t func_ptr = *reinterpret_cast<const uint64_t*>(bin_data);
        kernel.exec_mem = nullptr;
        kernel.size = 0;
        kernel.func_addr = func_ptr;

        std::cout << "Registered kernel (function pointer): func_id=" << func_id
                  << " -> addr=0x" << std::hex << func_ptr << std::dec << '\n';
    } else {
        // Binary mode: bin_data contains .text section binary code
        // Allocate executable memory using mmap
        void* exec_mem = mmap(nullptr, bin_size,
                             PROT_READ | PROT_WRITE | PROT_EXEC,
                             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (exec_mem == MAP_FAILED) {
            std::cerr << "Error: mmap failed for kernel func_id=" << func_id << '\n';
            return -1;
        }

        // Copy kernel .text binary into executable memory
        std::memcpy(exec_mem, bin_data, bin_size);

        kernel.exec_mem = exec_mem;
        kernel.size = bin_size;
        kernel.func_addr = reinterpret_cast<uint64_t>(exec_mem);

        std::cout << "Registered kernel (binary): func_id=" << func_id
                  << " -> addr=0x" << std::hex << kernel.func_addr << std::dec
                  << " (size=" << bin_size << " bytes)\n";
    }

    func_id_to_addr_[func_id] = kernel;
    return 0;
}

uint64_t DeviceRunner::get_function_bin_addr(int func_id) {
    auto it = func_id_to_addr_.find(func_id);
    if (it == func_id_to_addr_.end()) {
        std::cerr << "Warning: function_bin_addr not found for func_id=" << func_id << '\n';
        return 0;
    }
    return it->second.func_addr;
}
