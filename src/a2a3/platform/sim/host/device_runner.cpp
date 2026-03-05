/**
 * Device Runner Implementation - Thread-Based Simulation
 *
 * This file implements the simulated device execution using host threads.
 * It provides the same API as the real a2a3 implementation but uses
 * std::thread instead of CANN runtime APIs.
 *
 * aicpu_execute and aicore_execute_wrapper are loaded dynamically via dlopen from
 * the binaries passed to launch_runtime.
 *
 * Cross-platform notes:
 * - Linux: Uses MAP_ANONYMOUS for anonymous memory mapping
 * - macOS: Uses MAP_ANON (aliased) and MAP_JIT for executable memory on Apple Silicon
 *   which requires W^X (write xor execute) protection toggling via pthread_jit_write_protect_np
 */

#include "device_runner.h"

// Function pointer types for dynamically loaded executors
typedef int (*aicpu_execute_func_t)(Runtime* runtime);
typedef void (*aicore_execute_func_t)(Runtime* runtime, int block_idx, CoreType core_type, uint32_t physical_core_id, uint64_t regs);
typedef void (*set_platform_regs_func_t)(uint64_t regs);

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
            LOG_ERROR("Failed to create temp file for AICPU SO: %s", aicpu_so_path_.c_str());
            return -1;
        }
        ofs.write(reinterpret_cast<const char*>(aicpu_so_binary.data()), aicpu_so_binary.size());
        ofs.close();

        aicpu_so_handle_ = dlopen(aicpu_so_path_.c_str(), RTLD_NOW | RTLD_GLOBAL);
        if (aicpu_so_handle_ == nullptr) {
            LOG_ERROR("dlopen failed for AICPU SO: %s", dlerror());
            return -1;
        }

        aicpu_execute_func_ = reinterpret_cast<int(*)(Runtime*)>(dlsym(aicpu_so_handle_, "aicpu_execute"));
        if (aicpu_execute_func_ == nullptr) {
            LOG_ERROR("dlsym failed for aicpu_execute: %s", dlerror());
            return -1;
        }

        set_platform_regs_func_ = reinterpret_cast<void(*)(uint64_t)>(dlsym(aicpu_so_handle_, "set_platform_regs"));
        if (set_platform_regs_func_ == nullptr) {
            LOG_ERROR("dlsym failed for set_platform_regs: %s", dlerror());
            return -1;
        }

        LOG_INFO("DeviceRunner(sim): Loaded aicpu_execute from %s", aicpu_so_path_.c_str());
    }

    // Write AICore binary to temp file and dlopen
    if (!aicore_kernel_binary.empty() && aicore_execute_func_ == nullptr) {
        aicore_so_path_ = "/tmp/aicore_sim_" + std::to_string(getpid()) + ".so";
        std::ofstream ofs(aicore_so_path_, std::ios::binary);
        if (!ofs) {
            LOG_ERROR("Failed to create temp file for AICore SO: %s", aicore_so_path_.c_str());
            return -1;
        }
        ofs.write(reinterpret_cast<const char*>(aicore_kernel_binary.data()), aicore_kernel_binary.size());
        ofs.close();

        aicore_so_handle_ = dlopen(aicore_so_path_.c_str(), RTLD_NOW | RTLD_GLOBAL);
        if (aicore_so_handle_ == nullptr) {
            LOG_ERROR("dlopen failed for AICore SO: %s", dlerror());
            return -1;
        }

        aicore_execute_func_ = reinterpret_cast<void(*)(Runtime*, int, CoreType, uint32_t, uint64_t)>(dlsym(aicore_so_handle_, "aicore_execute_wrapper"));
        if (aicore_execute_func_ == nullptr) {
            LOG_ERROR("dlsym failed for aicore_execute_wrapper: %s", dlerror());
            return -1;
        }
        LOG_INFO("DeviceRunner(sim): Loaded aicore_execute_wrapper from %s", aicore_so_path_.c_str());
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

    // Validate launch_aicpu_num
    if (launch_aicpu_num < 1 || launch_aicpu_num > PLATFORM_MAX_AICPU_THREADS) {
        LOG_ERROR("launch_aicpu_num (%d) must be in range [1, %d]", 
                       launch_aicpu_num, PLATFORM_MAX_AICPU_THREADS);
        return -1;
    }

    // Validate block_dim
    if (block_dim < 1 || block_dim > PLATFORM_MAX_BLOCKDIM) {
        LOG_ERROR("block_dim (%d) must be in range [1, %d]", 
                       block_dim, PLATFORM_MAX_BLOCKDIM);
        return -1;
    }

    // Validate even distribution: block_dim must be divisible by scheduler thread count
    // When launch_aicpu_num == 4: 3 schedulers + 1 orchestrator (thread 3 has 0 cores)
    int scheduler_thread_num = (launch_aicpu_num == 4) ? 3 : launch_aicpu_num;
    if (block_dim % scheduler_thread_num != 0) {
        LOG_ERROR("block_dim (%d) must be evenly divisible by scheduler_thread_num (%d)",
                       block_dim, scheduler_thread_num);
        return -1;
    }

    // Ensure device is initialized
    int rc = ensure_device_initialized(device_id, aicpu_so_binary, aicore_kernel_binary);
    if (rc != 0) {
        LOG_ERROR("ensure_device_initialized failed: %d", rc);
        return rc;
    }

    // Calculate execution parameters
    block_dim_ = block_dim;
    int num_aicore = block_dim * cores_per_blockdim_;

    if (num_aicore > RUNTIME_MAX_WORKER) {
        LOG_ERROR("num_aicore (%d) exceeds RUNTIME_MAX_WORKER (%d)",
                       num_aicore, RUNTIME_MAX_WORKER);
        return -1;
    }

    // Initialize handshake buffers
    runtime.worker_count = num_aicore;
    worker_count_ = num_aicore;
    runtime.sche_cpu_num = launch_aicpu_num;

    // Calculate number of AIC cores
    int num_aic = block_dim;

    for (int i = 0; i < num_aicore; i++) {
        runtime.workers[i].aicpu_ready = 0;
        runtime.workers[i].aicore_done = 0;
        runtime.workers[i].control = 0;
        runtime.workers[i].task = 0;
        runtime.workers[i].task_status = 0;
        // First 1/3 are AIC, remaining 2/3 are AIV
        runtime.workers[i].core_type = (i < num_aic) ? CoreType::AIC : CoreType::AIV;
    }

    // Set function_bin_addr for each task from Runtime's func_id_to_addr_[] array
    // (addresses were stored there during init_runtime via upload_kernel_binary)
    LOG_DEBUG("Setting function_bin_addr for Tasks (Simulation)");
    for (int i = 0; i < runtime.get_task_count(); i++) {
        Task* task = runtime.get_task(i);
        if (task != nullptr) {
            uint64_t addr = runtime.get_function_bin_addr(task->func_id);
            task->function_bin_addr = addr;
            LOG_DEBUG("Task %d (func_id=%d) -> function_bin_addr=0x%lx",
                          i, task->func_id, addr);
        }
    }

    // Store runtime pointer for print_handshake_results
    last_runtime_ = &runtime;

    // Initialize performance profiling if enabled
    if (runtime.enable_profiling) {
        rc = init_performance_profiling(runtime, num_aicore, device_id);
        if (rc != 0) {
            LOG_ERROR("init_performance_profiling failed: %d", rc);
            return rc;
        }
    }

    // Allocate simulated register blocks for all AICore cores
    size_t total_reg_size = num_aicore * SIM_REG_BLOCK_SIZE;
    void* reg_blocks = mem_alloc_.alloc(total_reg_size);
    if (reg_blocks == nullptr) {
        LOG_ERROR("Failed to allocate simulated register memory (%zu bytes)", total_reg_size);
        return -1;
    }
    std::memset(reg_blocks, 0, total_reg_size);

    // Build array of per-core register base addresses
    size_t regs_array_size = num_aicore * sizeof(uint64_t);
    uint64_t* regs_array = reinterpret_cast<uint64_t*>(mem_alloc_.alloc(regs_array_size));
    if (regs_array == nullptr) {
        LOG_ERROR("Failed to allocate register address array");
        return -1;
    }
    for (int i = 0; i < num_aicore; i++) {
        regs_array[i] = reinterpret_cast<uint64_t>(
            static_cast<uint8_t*>(reg_blocks) + i * SIM_REG_BLOCK_SIZE);
    }
    kernel_args_.regs = reinterpret_cast<uint64_t>(regs_array);

    LOG_INFO("Allocated simulated registers: %d cores x 0x%x bytes", num_aicore, SIM_REG_BLOCK_SIZE);

    // Check if executors are loaded
    if (aicpu_execute_func_ == nullptr || aicore_execute_func_ == nullptr) {
        LOG_ERROR("Executor functions not loaded. Call ensure_binaries_loaded first.");
        return -1;
    }

    // Set platform regs in the AICPU .so before launching threads
    set_platform_regs_func_(kernel_args_.regs);

    // Launch AICPU threads
    LOG_INFO("Launching %d AICPU thread(s)", launch_aicpu_num);
    std::vector<std::thread> aicpu_threads;
    for (int i = 0; i < launch_aicpu_num; i++) {
        aicpu_threads.emplace_back([this, &runtime]() {
            aicpu_execute_func_(&runtime);
        });
    }

    // Launch AICore threads
    LOG_INFO("Launching %d AICore thread(s)", num_aicore);
    std::vector<std::thread> aicore_threads;
    for (int i = 0; i < num_aicore; i++) {
        CoreType core_type = runtime.workers[i].core_type;
        uint32_t physical_core_id = static_cast<uint32_t>(i);
        aicore_threads.emplace_back([this, &runtime, i, core_type, physical_core_id]() {
            aicore_execute_func_(&runtime, i, core_type, physical_core_id, kernel_args_.regs);
        });
    }

    // Poll and collect performance data during execution (if enabled)
    std::thread collector_thread;
    if (runtime.enable_profiling) {
        collector_thread = std::thread([this, &runtime]() {
            poll_and_collect_performance_data(runtime.get_task_count());
        });
    }

    // Wait for all threads to complete
    LOG_INFO("Waiting for threads to complete");
    for (auto& t : aicpu_threads) {
        t.join();
    }
    for (auto& t : aicore_threads) {
        t.join();
    }

    // Wait for collector thread if it was launched
    if (runtime.enable_profiling && collector_thread.joinable()) {
        collector_thread.join();
    }

    LOG_INFO("All threads completed");

    // Collect AICPU phase data and print performance data after execution completes
    if (runtime.enable_profiling) {
        perf_collector_.collect_phase_data();
        export_swimlane_json();
    }

    return 0;
}

void DeviceRunner::print_handshake_results() {
    if (worker_count_ == 0 || last_runtime_ == nullptr) {
        return;
    }

    LOG_DEBUG("Handshake results for %d cores:", worker_count_);
    for (int i = 0; i < worker_count_; i++) {
        LOG_DEBUG("  Core %d: aicore_done=%d aicpu_ready=%d control=%d task=%d",
                      i,
                      last_runtime_->workers[i].aicore_done,
                      last_runtime_->workers[i].aicpu_ready,
                      last_runtime_->workers[i].control,
                      last_runtime_->workers[i].task);
    }
}

int DeviceRunner::clean_cache() {
    // Skip if not initialized
    if (device_id_ == -1) {
        return 0;
    }

    // Only cleanup test-specific resources:

    // 1. Close dlopen'd kernel libraries (different tests may have different kernels)
    for (auto& pair : func_id_to_addr_) {
        MappedKernel& kernel = pair.second;
        if (kernel.dl_handle != nullptr) {
            dlclose(kernel.dl_handle);
            LOG_DEBUG("Closed dlopen kernel: func_id=%d", pair.first);
            kernel.dl_handle = nullptr;
            kernel.func_addr = 0;
        }
    }
    func_id_to_addr_.clear();

    LOG_INFO("DeviceRunner(sim): cache cleaned (test-specific resources only)");
    return 0;
}

int DeviceRunner::finalize() {
    // Skip if already finalized
    if (device_id_ == -1 && aicpu_so_handle_ == nullptr && aicore_so_handle_ == nullptr) {
        return 0;
    }

    // Print handshake results before cleanup
    print_handshake_results();

    // Cleanup performance profiling
    if (perf_collector_.is_initialized()) {
        auto free_cb = [](void* dev_ptr, void* user_data) -> int {
            (void)user_data;
            free(dev_ptr);
            return 0;
        };

        perf_collector_.finalize(nullptr, free_cb, nullptr);
    }

    // Close all dlopen'd kernel libraries
    for (auto& pair : func_id_to_addr_) {
        MappedKernel& kernel = pair.second;
        if (kernel.dl_handle != nullptr) {
            dlclose(kernel.dl_handle);
            LOG_DEBUG("Closed dlopen kernel: func_id=%d", pair.first);
            kernel.dl_handle = nullptr;
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

    LOG_INFO("DeviceRunner(sim) finalized");
    return 0;
}

// =============================================================================
// Kernel Binary Upload (returns function address for caller to store in Runtime)
// =============================================================================

uint64_t DeviceRunner::upload_kernel_binary(int func_id, const uint8_t* bin_data, size_t bin_size) {
    if (bin_data == nullptr || bin_size == 0) {
        LOG_ERROR("Invalid kernel data");
        return 0;
    }

    // Return cached address if already uploaded
    auto it = func_id_to_addr_.find(func_id);
    if (it != func_id_to_addr_.end()) {
        LOG_INFO("Kernel func_id=%d already uploaded, returning cached address", func_id);
        return it->second.func_addr;
    }

    // 1. Generate temp file path
    char tmpfile[256];
    snprintf(tmpfile, sizeof(tmpfile), "/tmp/kernel_%d_%d.so", func_id, getpid());

    // 2. Write to temp file
    std::ofstream ofs(tmpfile, std::ios::binary);
    if (!ofs) {
        LOG_ERROR("Failed to create temp file: %s", tmpfile);
        return 0;
    }
    ofs.write(reinterpret_cast<const char*>(bin_data), bin_size);
    ofs.close();

    LOG_DEBUG("Uploading kernel .so: %s (size=%zu bytes)", tmpfile, bin_size);

    // 3. dlopen to load .so (RTLD_NOW ensures all symbols resolved immediately)
    void* handle = dlopen(tmpfile, RTLD_NOW | RTLD_LOCAL);

    // 4. Remove temp file immediately (.so is already in memory)
    std::remove(tmpfile);

    if (!handle) {
        LOG_ERROR("dlopen failed: %s", dlerror());
        return 0;
    }

    // 5. dlsym to get kernel function address (unified entry point: "kernel_entry")
    void* func = dlsym(handle, "kernel_entry");
    if (!func) {
        LOG_ERROR("dlsym failed for 'kernel_entry': %s", dlerror());
        dlclose(handle);
        return 0;
    }

    // 6. Store mapping info for cleanup
    MappedKernel kernel;
    kernel.dl_handle = handle;
    kernel.func_addr = reinterpret_cast<uint64_t>(func);

    func_id_to_addr_[func_id] = kernel;

    LOG_DEBUG("Registered kernel (dlopen): func_id=%d -> addr=0x%lx, handle=%p",
                  func_id, kernel.func_addr, handle);

    return kernel.func_addr;
}

// =============================================================================
// Performance Profiling Implementation
// =============================================================================

int DeviceRunner::init_performance_profiling(Runtime& runtime, int num_aicore, int device_id) {
    // Define allocation callback (a2a3sim: use malloc)
    auto alloc_cb = [](size_t size, void* user_data) -> void* {
        (void)user_data;  // Not needed for malloc
        return malloc(size);
    };

    // Simulation: no registration needed (pass nullptr)
    return perf_collector_.initialize(runtime, num_aicore, device_id,
                                       alloc_cb, nullptr, nullptr);
}

void DeviceRunner::poll_and_collect_performance_data(int expected_tasks) {
    perf_collector_.poll_and_collect(expected_tasks);
}

int DeviceRunner::export_swimlane_json(const std::string& output_path) {
    return perf_collector_.export_swimlane_json(output_path);
}

