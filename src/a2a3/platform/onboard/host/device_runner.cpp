/**
 * Device Runner Implementation
 *
 * This file implements the device execution utilities for launching and
 * managing AICPU and AICore kernels on Ascend devices.
 */

#include "device_runner.h"

#include <dlfcn.h>

// Include HAL constants from CANN (header only, library loaded dynamically)
#include "ascend_hal.h"
#include "host/host_regs.h"  // Register address retrieval

// =============================================================================
// Lazy-loaded HAL (ascend_hal) for profiling host-register only
// =============================================================================

namespace {
void* g_hal_handle = nullptr;

using HalHostRegisterFn = int (*)(void* dev_ptr, size_t size, unsigned int flags, int device_id, void** host_ptr);
using HalHostUnregisterFn = int (*)(void* host_ptr, int device_id);

int load_hal_if_needed() {
    if (g_hal_handle != nullptr) {
        return 0;
    }
    g_hal_handle = dlopen("libascend_hal.so", RTLD_NOW | RTLD_LOCAL);
    if (g_hal_handle == nullptr) {
        return -1;
    }
    return 0;
}

HalHostRegisterFn get_halHostRegister() {
    if (g_hal_handle == nullptr) {
        return nullptr;
    }
    return reinterpret_cast<HalHostRegisterFn>(dlsym(g_hal_handle, "halHostRegister"));
}

HalHostUnregisterFn get_halHostUnregister() {
    if (g_hal_handle == nullptr) {
        return nullptr;
    }
    return reinterpret_cast<HalHostUnregisterFn>(dlsym(g_hal_handle, "halHostUnregister"));
}
}  // namespace

// =============================================================================
// KernelArgsHelper Implementation
// =============================================================================

int KernelArgsHelper::init_device_args(const DeviceArgs& host_device_args, MemoryAllocator& allocator) {
    allocator_ = &allocator;

    // Allocate device memory for device_args
    if (args.device_args == nullptr) {
        uint64_t device_args_size = sizeof(DeviceArgs);
        void* device_args_dev = allocator_->alloc(device_args_size);
        if (device_args_dev == nullptr) {
            LOG_ERROR("Alloc for device_args failed");
            return -1;
        }
        args.device_args = reinterpret_cast<DeviceArgs*>(device_args_dev);
    }
    // Copy host_device_args to device memory via device_args
    int rc =
        rtMemcpy(args.device_args, sizeof(DeviceArgs), &host_device_args, sizeof(DeviceArgs), RT_MEMCPY_HOST_TO_DEVICE);
    if (rc != 0) {
        LOG_ERROR("rtMemcpy failed: %d", rc);
        allocator_->free(args.device_args);
        args.device_args = nullptr;
        return rc;
    }
    return 0;
}

int KernelArgsHelper::finalize_device_args() {
    if (args.device_args != nullptr && allocator_ != nullptr) {
        int rc = allocator_->free(args.device_args);
        args.device_args = nullptr;
        return rc;
    }
    return 0;
}

int KernelArgsHelper::init_runtime_args(const Runtime& host_runtime, MemoryAllocator& allocator) {
    allocator_ = &allocator;

    if (args.runtime_args == nullptr) {
        uint64_t runtime_size = sizeof(Runtime);
        void* runtime_dev = allocator_->alloc(runtime_size);
        if (runtime_dev == nullptr) {
            LOG_ERROR("Alloc for runtime_args failed");
            return -1;
        }
        args.runtime_args = reinterpret_cast<Runtime*>(runtime_dev);
    }
    int rc = rtMemcpy(args.runtime_args, sizeof(Runtime), &host_runtime, sizeof(Runtime), RT_MEMCPY_HOST_TO_DEVICE);
    if (rc != 0) {
        LOG_ERROR("rtMemcpy for runtime failed: %d", rc);
        allocator_->free(args.runtime_args);
        args.runtime_args = nullptr;
        return rc;
    }
    return 0;
}

int KernelArgsHelper::finalize_runtime_args() {
    if (args.runtime_args != nullptr && allocator_ != nullptr) {
        int rc = allocator_->free(args.runtime_args);
        args.runtime_args = nullptr;
        return rc;
    }
    return 0;
}

// =============================================================================
// AicpuSoInfo Implementation
// =============================================================================

int AicpuSoInfo::init(const std::vector<uint8_t>& aicpu_so_binary, MemoryAllocator& allocator) {
    allocator_ = &allocator;

    if (aicpu_so_binary.empty()) {
        LOG_ERROR("AICPU binary is empty");
        return -1;
    }

    size_t file_size = aicpu_so_binary.size();
    void* d_aicpu_data = allocator_->alloc(file_size);
    if (d_aicpu_data == nullptr) {
        LOG_ERROR("Alloc failed for AICPU SO");
        return -1;
    }

    int rc = rtMemcpy(d_aicpu_data, file_size, aicpu_so_binary.data(), file_size, RT_MEMCPY_HOST_TO_DEVICE);
    if (rc != 0) {
        LOG_ERROR("rtMemcpy failed: %d", rc);
        allocator_->free(d_aicpu_data);
        d_aicpu_data = nullptr;
        return rc;
    }

    aicpu_so_bin = reinterpret_cast<uint64_t>(d_aicpu_data);
    aicpu_so_len = file_size;
    return 0;
}

int AicpuSoInfo::finalize() {
    if (aicpu_so_bin != 0 && allocator_ != nullptr) {
        int rc = allocator_->free(reinterpret_cast<void*>(aicpu_so_bin));
        aicpu_so_bin = 0;
        return rc;
    }
    return 0;
}

// =============================================================================
// DeviceRunner Implementation
// =============================================================================

DeviceRunner& DeviceRunner::get() {
    static DeviceRunner runner;
    return runner;
}

DeviceRunner::~DeviceRunner() { finalize(); }

int DeviceRunner::ensure_device_initialized(
    int device_id, const std::vector<uint8_t>& aicpu_so_binary, const std::vector<uint8_t>& aicore_kernel_binary) {
    // First ensure device is set and streams are created
    int rc = ensure_device_set(device_id);
    if (rc != 0) {
        return rc;
    }

    // Then ensure binaries are loaded
    return ensure_binaries_loaded(aicpu_so_binary, aicore_kernel_binary);
}

int DeviceRunner::ensure_device_set(int device_id) {
    // Check if already initialized
    if (stream_aicpu_ != nullptr) {
        return 0;
    }

    device_id_ = device_id;

    // Set device
    int rc = rtSetDevice(device_id);
    if (rc != 0) {
        LOG_ERROR("rtSetDevice(%d) failed: %d", device_id, rc);
        return rc;
    }

    // Create streams
    rc = rtStreamCreate(&stream_aicpu_, 0);
    if (rc != 0) {
        LOG_ERROR("rtStreamCreate (AICPU) failed: %d", rc);
        return rc;
    }

    rc = rtStreamCreate(&stream_aicore_, 0);
    if (rc != 0) {
        LOG_ERROR("rtStreamCreate (AICore) failed: %d", rc);
        rtStreamDestroy(stream_aicpu_);
        stream_aicpu_ = nullptr;
        return rc;
    }

    LOG_INFO("DeviceRunner: device=%d set, streams created", device_id);
    return 0;
}

int DeviceRunner::ensure_binaries_loaded(
    const std::vector<uint8_t>& aicpu_so_binary, const std::vector<uint8_t>& aicore_kernel_binary) {
    // Check if already loaded
    if (binaries_loaded_) {
        // Just update kernel binary if different
        if (aicore_kernel_binary_ != aicore_kernel_binary) {
            aicore_kernel_binary_ = aicore_kernel_binary;
        }
        return 0;
    }

    // Device must be set first
    if (stream_aicpu_ == nullptr) {
        LOG_ERROR("Device not set before loading binaries");
        return -1;
    }

    aicore_kernel_binary_ = aicore_kernel_binary;

    // Load AICPU SO
    int rc = so_info_.init(aicpu_so_binary, mem_alloc_);
    if (rc != 0) {
        LOG_ERROR("AicpuSoInfo::init failed: %d", rc);
        return rc;
    }

    // Initialize device args
    device_args_.aicpu_so_bin = so_info_.aicpu_so_bin;
    device_args_.aicpu_so_len = so_info_.aicpu_so_len;
    rc = kernel_args_.init_device_args(device_args_, mem_alloc_);
    if (rc != 0) {
        LOG_ERROR("init_device_args failed: %d", rc);
        so_info_.finalize();
        return rc;
    }

    binaries_loaded_ = true;
    LOG_INFO("DeviceRunner: binaries loaded");
    return 0;
}

void* DeviceRunner::allocate_tensor(size_t bytes) { return mem_alloc_.alloc(bytes); }

void DeviceRunner::free_tensor(void* dev_ptr) {
    if (dev_ptr != nullptr) {
        mem_alloc_.free(dev_ptr);
    }
}

int DeviceRunner::copy_to_device(void* dev_ptr, const void* host_ptr, size_t bytes) {
    return rtMemcpy(dev_ptr, bytes, host_ptr, bytes, RT_MEMCPY_HOST_TO_DEVICE);
}

int DeviceRunner::copy_from_device(void* host_ptr, const void* dev_ptr, size_t bytes) {
    return rtMemcpy(host_ptr, bytes, dev_ptr, bytes, RT_MEMCPY_DEVICE_TO_HOST);
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

    // Ensure device is initialized (lazy initialization)
    int rc = ensure_device_initialized(device_id, aicpu_so_binary, aicore_kernel_binary);
    if (rc != 0) {
        LOG_ERROR("ensure_device_initialized failed: %d", rc);
        return rc;
    }

    // Calculate execution parameters
    block_dim_ = block_dim;

    int num_aicore = block_dim * cores_per_blockdim_;
    // Initialize handshake buffers in runtime
    if (num_aicore > RUNTIME_MAX_WORKER) {
        LOG_ERROR("block_dim (%d) exceeds RUNTIME_MAX_WORKER (%d)",
                      block_dim, RUNTIME_MAX_WORKER);
        return -1;
    }

    runtime.worker_count = num_aicore;
    worker_count_ = num_aicore;  // Store for print_handshake_results in destructor
    runtime.sche_cpu_num = launch_aicpu_num;

    // Get AICore register addresses for register-based task dispatch
    rc = init_aicore_register_addresses(&kernel_args_.args.regs, static_cast<uint64_t>(device_id), mem_alloc_);
    if (rc != 0) {
        LOG_ERROR("init_aicore_register_addresses failed: %d", rc);
        return rc;
    }

    // Calculate number of AIC cores (1/3 of total)
    int num_aic = block_dim;  // Round up for 1/3

    for (int i = 0; i < num_aicore; i++) {
        runtime.workers[i].aicpu_ready = 0;
        runtime.workers[i].aicore_done = 0;
        runtime.workers[i].control = 0;
        runtime.workers[i].task = 0;
        runtime.workers[i].task_status = 0;
        // Set core type: first 1/3 are AIC, remaining 2/3 are AIV
        runtime.workers[i].core_type = (i < num_aic) ? CoreType::AIC : CoreType::AIV;
        runtime.workers[i].perf_records_addr = (uint64_t)nullptr;
        runtime.workers[i].perf_buffer_status = 0;
    }

    // Set function_bin_addr for all tasks from Runtime's func_id_to_addr_[] array
    // (addresses were stored there during init_runtime via upload_kernel_binary)
    LOG_DEBUG("Setting function_bin_addr for Tasks");
    for (int i = 0; i < runtime.get_task_count(); i++) {
        Task* task = runtime.get_task(i);
        if (task != nullptr) {
            uint64_t addr = runtime.get_function_bin_addr(task->func_id);
            task->function_bin_addr = addr;
            LOG_DEBUG("Task %d (func_id=%d) -> function_bin_addr=0x%lx",
                          i, task->func_id, addr);
        }
    }
    LOG_DEBUG("");

    // Initialize performance profiling if enabled
    if (runtime.enable_profiling) {
        rc = init_performance_profiling(runtime, num_aicore, device_id);
        if (rc != 0) {
            LOG_ERROR("init_performance_profiling failed: %d", rc);
            return rc;
        }
    }

    std::cout << "\n=== Initialize runtime args ===" << '\n';
    // Initialize runtime args
    rc = kernel_args_.init_runtime_args(runtime, mem_alloc_);
    if (rc != 0) {
        LOG_ERROR("init_runtime_args failed: %d", rc);
        return rc;
    }

    std::cout << "\n=== launch_aicpu_kernel DynTileFwkKernelServerInit===" << '\n';
    // Launch AICPU init kernel
    rc = launch_aicpu_kernel(stream_aicpu_, &kernel_args_.args, "DynTileFwkKernelServerInit", 1);
    if (rc != 0) {
        LOG_ERROR("launch_aicpu_kernel (init) failed: %d", rc);
        kernel_args_.finalize_runtime_args();
        return rc;
    }

    std::cout << "\n=== launch_aicpu_kernel DynTileFwkKernelServer===" << '\n';
    // Launch AICPU main kernel
    rc = launch_aicpu_kernel(stream_aicpu_, &kernel_args_.args, "DynTileFwkKernelServer", launch_aicpu_num);
    if (rc != 0) {
        LOG_ERROR("launch_aicpu_kernel (main) failed: %d", rc);
        kernel_args_.finalize_runtime_args();
        return rc;
    }

    std::cout << "\n=== launch_aicore_kernel===" << '\n';
    // Launch AICore kernel
    rc = launch_aicore_kernel(stream_aicore_, kernel_args_.args.runtime_args);
    if (rc != 0) {
        LOG_ERROR("launch_aicore_kernel failed: %d", rc);
        kernel_args_.finalize_runtime_args();
        return rc;
    }

    // Poll and collect performance data (must be before stream sync)
    if (runtime.enable_profiling) {
        poll_and_collect_performance_data(runtime.get_task_count());
    }

    std::cout << "\n=== rtStreamSynchronize stream_aicpu_===" << '\n';
    // Synchronize streams
    rc = rtStreamSynchronize(stream_aicpu_);
    if (rc != 0) {
        LOG_ERROR("rtStreamSynchronize (AICPU) failed: %d", rc);
        kernel_args_.finalize_runtime_args();
        return rc;
    }

    std::cout << "\n=== rtStreamSynchronize stream_aicore_===" << '\n';
    rc = rtStreamSynchronize(stream_aicore_);
    if (rc != 0) {
        LOG_ERROR("rtStreamSynchronize (AICore) failed: %d", rc);
        kernel_args_.finalize_runtime_args();
        return rc;
    }

    // Collect phase data and print performance data (after stream sync)
    if (runtime.enable_profiling) {
        perf_collector_.collect_phase_data();
        export_swimlane_json();
    }

    // Note: FinalizeRuntimeArgs is deferred to Finalize() so PrintHandshakeResults can access device data

    return 0;
}

void DeviceRunner::print_handshake_results() {
    if (stream_aicpu_ == nullptr || worker_count_ == 0 || kernel_args_.args.runtime_args == nullptr) {
        return;
    }

    // Allocate temporary buffer to read handshake data from device
    std::vector<Handshake> workers(worker_count_);
    size_t total_size = sizeof(Handshake) * worker_count_;
    rtMemcpy(workers.data(), total_size, kernel_args_.args.runtime_args->workers, total_size, RT_MEMCPY_DEVICE_TO_HOST);

    LOG_DEBUG("Handshake results for %d cores:", worker_count_);
    for (int i = 0; i < worker_count_; i++) {
        LOG_DEBUG("  Core %d: aicore_done=%d aicpu_ready=%d control=%d task=%d",
                      i, workers[i].aicore_done, workers[i].aicpu_ready,
                      workers[i].control, workers[i].task);
    }
}

int DeviceRunner::clean_cache() {
    if (stream_aicpu_ == nullptr) {
        return 0;  // Nothing to cleanup if device not initialized
    }
    for (const auto& [func_id, addr] : func_id_to_addr_) {
        void* gm_addr = reinterpret_cast<void*>(addr - sizeof(uint64_t));
        mem_alloc_.free(gm_addr);
    }
    func_id_to_addr_.clear();

    LOG_INFO("DeviceRunner: cache cleaned (test-specific resources only)");
    return 0;
}

int DeviceRunner::finalize() {
    if (stream_aicpu_ == nullptr) {
        return 0;
    }

    // Print handshake results before cleanup (reads from device memory)
    print_handshake_results();

    // Cleanup runtime args (deferred from Run)
    kernel_args_.finalize_runtime_args();

    // Cleanup kernel args (deviceArgs)
    kernel_args_.finalize_device_args();

    // Cleanup AICPU SO
    so_info_.finalize();

    // Clear kernel address mapping
    func_id_to_addr_.clear();
    binaries_loaded_ = false;

    // Destroy streams
    if (stream_aicpu_ != nullptr) {
        rtStreamDestroy(stream_aicpu_);
        stream_aicpu_ = nullptr;
    }
    if (stream_aicore_ != nullptr) {
        rtStreamDestroy(stream_aicore_);
        stream_aicore_ = nullptr;
    }

    // Cleanup performance profiling
    if (perf_collector_.is_initialized()) {
        auto unregister_cb = [](void* dev_ptr, int device_id, void* user_data) -> int {
            (void)user_data;
            HalHostUnregisterFn fn = get_halHostUnregister();
            if (fn != nullptr) {
                return fn(dev_ptr, device_id);
            }
            return 0;
        };

        auto free_cb = [](void* dev_ptr, void* user_data) -> int {
            auto* allocator = static_cast<MemoryAllocator*>(user_data);
            return allocator->free(dev_ptr);
        };

        perf_collector_.finalize(unregister_cb, free_cb, &mem_alloc_);
    }

    // Free all remaining allocations (including handshake buffer and binGmAddr)
    mem_alloc_.finalize();

    device_id_ = -1;
    worker_count_ = 0;
    aicore_kernel_binary_.clear();

    LOG_INFO("DeviceRunner finalized");
    return 0;
}

int DeviceRunner::launch_aicpu_kernel(rtStream_t stream, KernelArgs* k_args, const char* kernel_name, int aicpu_num) {
    struct Args {
        KernelArgs k_args;
        char kernel_name[32];
        const char so_name[32] = {"libaicpu_extend_kernels.so"};
        const char op_name[32] = {""};
    } args;

    args.k_args = *k_args;
    std::strncpy(args.kernel_name, kernel_name, sizeof(args.kernel_name) - 1);
    args.kernel_name[sizeof(args.kernel_name) - 1] = '\0';

    rtAicpuArgsEx_t rt_args;
    std::memset(&rt_args, 0, sizeof(rt_args));
    rt_args.args = &args;
    rt_args.argsSize = sizeof(args);
    rt_args.kernelNameAddrOffset = offsetof(struct Args, kernel_name);
    rt_args.soNameAddrOffset = offsetof(struct Args, so_name);

    return rtAicpuKernelLaunchExWithArgs(
        rtKernelType_t::KERNEL_TYPE_AICPU_KFC, "AST_DYN_AICPU", aicpu_num, &rt_args, nullptr, stream, 0);
}

int DeviceRunner::launch_aicore_kernel(rtStream_t stream, Runtime* runtime) {
    if (aicore_kernel_binary_.empty()) {
        LOG_ERROR("AICore kernel binary is empty");
        return -1;
    }

    size_t bin_size = aicore_kernel_binary_.size();
    const void* bin_data = aicore_kernel_binary_.data();

    rtDevBinary_t binary;
    std::memset(&binary, 0, sizeof(binary));
    binary.magic = RT_DEV_BINARY_MAGIC_ELF;
    binary.version = 0;
    binary.data = bin_data;
    binary.length = bin_size;
    void* bin_handle = nullptr;
    int rc = rtRegisterAllKernel(&binary, &bin_handle);
    if (rc != RT_ERROR_NONE) {
        LOG_ERROR("rtRegisterAllKernel failed: %d", rc);
        return rc;
    }

    struct Args {
        Runtime* runtime;
    };
    // Pass device address of Runtime to AICore
    Args args = {runtime};
    rtArgsEx_t rt_args;
    std::memset(&rt_args, 0, sizeof(rt_args));
    rt_args.args = &args;
    rt_args.argsSize = sizeof(args);

    rtTaskCfgInfo_t cfg = {};
    cfg.schemMode = RT_SCHEM_MODE_BATCH;

    rc = rtKernelLaunchWithHandleV2(bin_handle, 0, block_dim_, &rt_args, nullptr, stream, &cfg);
    if (rc != RT_ERROR_NONE) {
        LOG_ERROR("rtKernelLaunchWithHandleV2 failed: %d", rc);
        return rc;
    }

    return rc;
}

// =============================================================================
// Kernel Binary Upload (returns device address for caller to store in Runtime)
// =============================================================================

uint64_t DeviceRunner::upload_kernel_binary(int func_id, const uint8_t* bin_data, size_t bin_size) {
    if (bin_data == nullptr || bin_size == 0) {
        LOG_ERROR("Invalid kernel binary data");
        return 0;
    }

    // Device must be set first (set_device() must be called before upload_kernel_binary())
    if (stream_aicpu_ == nullptr) {
        LOG_ERROR("Device not set. Call set_device() before upload_kernel_binary()");
        return 0;
    }

    // Return cached address if already uploaded
    auto it = func_id_to_addr_.find(func_id);
    if (it != func_id_to_addr_.end()) {
        LOG_INFO("Kernel func_id=%d already uploaded, returning cached address", func_id);
        return it->second;
    }

    LOG_DEBUG("Uploading kernel binary: func_id=%d, size=%zu bytes", func_id, bin_size);

    // Allocate device GM memory (size field + binary data)
    uint64_t alloc_size = sizeof(uint64_t) + bin_size;
    void* gm_addr = mem_alloc_.alloc(alloc_size);
    if (gm_addr == nullptr) {
        LOG_ERROR("Failed to allocate device GM memory for kernel func_id=%d", func_id);
        return 0;
    }

    // Build host buffer with CoreFunctionBin structure (size + data)
    std::vector<uint8_t> host_buf(alloc_size);
    uint64_t* size_ptr = reinterpret_cast<uint64_t*>(host_buf.data());
    *size_ptr = bin_size;
    std::memcpy(host_buf.data() + sizeof(uint64_t), bin_data, bin_size);

    // Copy to device
    int rc = rtMemcpy(gm_addr, alloc_size, host_buf.data(), alloc_size, RT_MEMCPY_HOST_TO_DEVICE);
    if (rc != 0) {
        LOG_ERROR("rtMemcpy to device failed: %d", rc);
        mem_alloc_.free(gm_addr);
        return 0;
    }

    // Calculate function_bin_addr (skip size field to get actual code address)
    uint64_t function_bin_addr = reinterpret_cast<uint64_t>(gm_addr) + sizeof(uint64_t);

    // Cache for later reuse and cleanup
    func_id_to_addr_[func_id] = function_bin_addr;

    LOG_DEBUG("  func_id=%d -> function_bin_addr=0x%lx", func_id, function_bin_addr);

    return function_bin_addr;
}

int DeviceRunner::init_performance_profiling(Runtime& runtime, int num_aicore, int device_id) {
    // Define allocation callback (a2a3: use MemoryAllocator)
    auto alloc_cb = [](size_t size, void* user_data) -> void* {
        auto* allocator = static_cast<MemoryAllocator*>(user_data);
        return allocator->alloc(size);
    };

    // Define registration callback (a2a3: use halHostRegister for shared memory)
    auto register_cb = [](void* dev_ptr, size_t size, int device_id,
                          void* user_data, void** host_ptr) -> int {
        (void)user_data;  // Not needed for registration
        if (load_hal_if_needed() != 0) {
            LOG_ERROR("Failed to load ascend_hal for profiling: %s", dlerror());
            return -1;
        }
        HalHostRegisterFn fn = get_halHostRegister();
        if (fn == nullptr) {
            LOG_ERROR("halHostRegister symbol not found: %s", dlerror());
            return -1;
        }
        return fn(dev_ptr, size, DEV_SVM_MAP_HOST, device_id, host_ptr);
    };

    return perf_collector_.initialize(runtime, num_aicore, device_id,
                                       alloc_cb, register_cb, &mem_alloc_);
}

void DeviceRunner::poll_and_collect_performance_data(int expected_tasks) {
    perf_collector_.poll_and_collect(expected_tasks);
}

int DeviceRunner::export_swimlane_json(const std::string& output_path) {
    return perf_collector_.export_swimlane_json(output_path);
}

