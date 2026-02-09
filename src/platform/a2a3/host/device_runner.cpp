/**
 * Device Runner Implementation
 *
 * This file implements the device execution utilities for launching and
 * managing AICPU and AICore kernels on Ascend devices.
 */

#include "device_runner.h"


#include <dlfcn.h>

// Use real HAL constant from CANN (compile-time only; we do not link ascend_hal)
#include "ascend_hal.h"

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
    LOG_DEBUG("\n=== Setting function_bin_addr for Tasks ===");
    for (int i = 0; i < runtime.get_task_count(); i++) {
        Task* task = runtime.get_task(i);
        if (task != nullptr) {
            uint64_t addr = runtime.get_function_bin_addr(task->func_id);
            task->function_bin_addr = addr;
            LOG_DEBUG("  Task %d (func_id=%d) -> function_bin_addr=0x%lx",
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

    // Initialize runtime args
    rc = kernel_args_.init_runtime_args(runtime, mem_alloc_);
    if (rc != 0) {
        LOG_ERROR("init_runtime_args failed: %d", rc);
        return rc;
    }

    // Launch AICPU init kernel
    rc = launch_aicpu_kernel(stream_aicpu_, &kernel_args_.args, "DynTileFwkKernelServerInit", 1);
    if (rc != 0) {
        LOG_ERROR("launch_aicpu_kernel (init) failed: %d", rc);
        kernel_args_.finalize_runtime_args();
        return rc;
    }

    // Launch AICPU main kernel
    rc = launch_aicpu_kernel(stream_aicpu_, &kernel_args_.args, "DynTileFwkKernelServer", launch_aicpu_num);
    if (rc != 0) {
        LOG_ERROR("launch_aicpu_kernel (main) failed: %d", rc);
        kernel_args_.finalize_runtime_args();
        return rc;
    }

    // Launch AICore kernel
    rc = launch_aicore_kernel(stream_aicore_, kernel_args_.args.runtime_args);
    if (rc != 0) {
        LOG_ERROR("launch_aicore_kernel failed: %d", rc);
        kernel_args_.finalize_runtime_args();
        return rc;
    }

    // Poll and collect performance data (must be before stream sync)
    if (runtime.enable_profiling) {
        poll_and_collect_performance_data(runtime.worker_count, runtime.get_task_count());
    }

    // Synchronize streams
    rc = rtStreamSynchronize(stream_aicpu_);
    if (rc != 0) {
        LOG_ERROR("rtStreamSynchronize (AICPU) failed: %d", rc);
        kernel_args_.finalize_runtime_args();
        return rc;
    }

    rc = rtStreamSynchronize(stream_aicore_);
    if (rc != 0) {
        LOG_ERROR("rtStreamSynchronize (AICore) failed: %d", rc);
        kernel_args_.finalize_runtime_args();
        return rc;
    }

    // Print collected performance data (after stream sync)
    if (runtime.enable_profiling) {
        print_performance_data();
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

    // Cleanup profiling shared memory
    if (perf_shared_mem_host_ != nullptr) {
        LOG_DEBUG("Cleaning up profiling shared memory...");

        // Unregister host mapping
        HalHostUnregisterFn fn = get_halHostUnregister();
        if (fn != nullptr) {
            fn(perf_shared_mem_host_, device_id_);
        }
        perf_shared_mem_host_ = nullptr;

        LOG_DEBUG("  Host mapping unregistered");
    }

    if (perf_shared_mem_dev_ != nullptr) {
        // Free device memory (managed by mem_alloc_, will be freed in finalize())
        perf_shared_mem_dev_ = nullptr;

        LOG_DEBUG("  Device memory marked for cleanup");
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
    LOG_INFO("=== Initializing Performance Profiling ===");

    // Step 1: Calculate total memory size (header + all DoubleBuffers)
    size_t total_size = calc_perf_data_size(num_aicore);

    size_t header_size = sizeof(PerfDataHeader);
    size_t single_db_size = sizeof(DoubleBuffer);
    size_t buffers_size = num_aicore * single_db_size;

    LOG_DEBUG("  Memory allocation plan:");
    LOG_DEBUG("    - Number of cores:      %d", num_aicore);
    LOG_DEBUG("    - Header size:          %zu bytes", header_size);
    LOG_DEBUG("      (includes ready queue: %d entries)", PLATFORM_MAX_CORES * 2);
    LOG_DEBUG("    - Single DoubleBuffer:  %zu bytes", single_db_size);
    LOG_DEBUG("    - All DoubleBuffers:    %zu bytes", buffers_size);
    LOG_DEBUG("    - Total size:           %zu bytes (%zu KB, %zu MB)",
              total_size, total_size / 1024, total_size / (1024 * 1024));

    // Step 2: Allocate device shared memory
    void* perf_dev_ptr = mem_alloc_.alloc(total_size);
    if (perf_dev_ptr == nullptr) {
        LOG_ERROR("Failed to allocate device memory for profiling (%zu bytes)", total_size);
        return -1;
    }
    LOG_DEBUG("  Allocated device memory: %p", perf_dev_ptr);

    // Step 3: Register to host mapping (create Host-Device shared memory)
    if (load_hal_if_needed() != 0) {
        LOG_ERROR("Failed to load ascend_hal for profiling: %s", dlerror());
        mem_alloc_.free(perf_dev_ptr);
        return -1;
    }
    HalHostRegisterFn fn = get_halHostRegister();
    if (fn == nullptr) {
        LOG_ERROR("halHostRegister symbol not found: %s", dlerror());
        mem_alloc_.free(perf_dev_ptr);
        return -1;
    }
    void* perf_host_ptr = nullptr;
    int rc = fn(perf_dev_ptr, total_size, DEV_SVM_MAP_HOST, device_id, &perf_host_ptr);
    if (rc != 0) {
        LOG_ERROR("halHostRegister failed: %d", rc);
        mem_alloc_.free(perf_dev_ptr);
        return rc;
    }
    LOG_DEBUG("  Mapped to host memory: %p", perf_host_ptr);

    // Step 4: Initialize fixed header (using host_ptr)
    PerfDataHeader* header = get_perf_header(perf_host_ptr);

    // Initialize queue
    memset(header->queue, 0, sizeof(header->queue));
    header->queue_head = 0;
    header->queue_tail = 0;

    // Initialize metadata
    header->num_cores = num_aicore;

    LOG_DEBUG("  Initialized PerfDataHeader:");
    LOG_DEBUG("    - num_cores:        %d", header->num_cores);
    LOG_DEBUG("    - buffer_capacity:  %d", PLATFORM_PROF_BUFFER_SIZE);
    LOG_DEBUG("    - queue capacity:   %d", PLATFORM_MAX_CORES * 2);

    // Step 5: Initialize all DoubleBuffers (all buffers start as 0=idle)
    DoubleBuffer* buffers = get_double_buffers(perf_host_ptr);

    for (int i = 0; i < num_aicore; i++) {
        DoubleBuffer* db = &buffers[i];

        // Initialize buffer1
        memset(&db->buffer1, 0, sizeof(PerfBuffer));
        db->buffer1.count = 0;
        db->buffer1.first_task_time = 0;
        db->buffer1_status = BufferStatus::IDLE;

        // Initialize buffer2
        memset(&db->buffer2, 0, sizeof(PerfBuffer));
        db->buffer2.count = 0;
        db->buffer2.first_task_time = 0;
        db->buffer2_status = BufferStatus::IDLE;
    }

    LOG_DEBUG("  Initialized %d DoubleBuffers (all status=0, idle)", num_aicore);

    // Step 6: Write memory barrier (ensure all initialization visible to Device)
    wmb();

    // Step 7: Pass to Runtime (device base address)
    runtime.perf_data_base = (uint64_t)perf_dev_ptr;

    LOG_DEBUG("  Set runtime.perf_data_base = 0x%lx", runtime.perf_data_base);

    // Step 8: Save pointers to member variables
    perf_shared_mem_dev_ = perf_dev_ptr;
    perf_shared_mem_host_ = perf_host_ptr;

    LOG_INFO("=== Performance Profiling Initialized ===");

    return 0;
}

void DeviceRunner::poll_and_collect_performance_data(int num_cores, int expected_tasks) {
    if (perf_shared_mem_host_ == nullptr) {
        return;  // Profiling not enabled
    }

    LOG_INFO("=== Collecting Performance Data ===");
    LOG_DEBUG("  Expected tasks: %d", expected_tasks);

    PerfDataHeader* header = get_perf_header(perf_shared_mem_host_);
    DoubleBuffer* buffers = get_double_buffers(perf_shared_mem_host_);

    uint32_t capacity = PLATFORM_MAX_CORES * 2;
    int total_records_collected = 0;
    int buffers_processed = 0;

    // Clear previous collection
    collected_perf_records_.clear();

    // Timeout configuration
    const auto timeout_duration = std::chrono::seconds(PLATFORM_PROF_TIMEOUT_SECONDS);  // 30 second timeout
    const auto start_time = std::chrono::steady_clock::now();
    int empty_poll_count = 0;

    // Poll the ready queue until all expected tasks are collected
    while (total_records_collected < expected_tasks) {
        // Read queue status with memory barrier
        rmb();
        uint32_t head = header->queue_head;
        uint32_t tail = header->queue_tail;

        // Check if queue is empty
        if (head == tail) {
            // Queue is empty but we haven't collected all tasks yet
            // Check for timeout periodically
            empty_poll_count++;
            if (empty_poll_count >= PLATFORM_PROF_EMPTY_POLLS_CHECK_NUM) {
                empty_poll_count = 0;
                auto elapsed = std::chrono::steady_clock::now() - start_time;
                if (elapsed >= timeout_duration) {
                    LOG_WARN("\n  WARNING: Performance data collection timeout after %ld seconds",
                             std::chrono::duration_cast<std::chrono::seconds>(elapsed).count());
                    LOG_WARN("  Collected %d / %d records before timeout",
                             total_records_collected, expected_tasks);
                    break;  // Exit with partial data
                }
            }
            // Continue polling (AICPU may still be producing data)
            continue;
        }

        // Reset empty poll counter when we find data
        empty_poll_count = 0;

        // Dequeue entry
        ReadyQueueEntry entry = header->queue[head];
        uint32_t core_index = entry.core_index;
        uint32_t buffer_id = entry.buffer_id;

        // Validate core index
        if (core_index >= static_cast<uint32_t>(num_cores)) {
            LOG_ERROR("Invalid core_index %u (max=%d)", core_index, num_cores);
            break;
        }

        LOG_DEBUG("  Processing: core=%u, buffer=%u", core_index, buffer_id);

        // Get the buffer and status pointer
        DoubleBuffer* db = &buffers[core_index];
        PerfBuffer* buf = nullptr;
        volatile BufferStatus* status = nullptr;
        get_buffer_and_status(db, buffer_id, &buf, &status);

        // Read buffer data with memory barrier
        rmb();
        uint32_t count = buf->count;
        uint64_t first_task_time = buf->first_task_time;

        LOG_DEBUG("    Records in buffer: %u", count);
        LOG_DEBUG("    First task time: %lu", first_task_time);

        // Collect records
        for (uint32_t i = 0; i < count && i < PLATFORM_PROF_BUFFER_SIZE; i++) {
            collected_perf_records_.push_back(buf->records[i]);
            total_records_collected++;
        }

        // Clear buffer
        buf->count = 0;
        buf->first_task_time = 0;

        // Set buffer status to IDLE
        *status = BufferStatus::IDLE;
        wmb();  // Ensure status is visible to AICPU

        // Update queue head
        header->queue_head = (head + 1) % capacity;
        wmb();  // Ensure head update is visible to AICPU

        buffers_processed++;
    }

    LOG_INFO("  Total buffers processed: %d", buffers_processed);
    LOG_INFO("  Total records collected: %d", total_records_collected);

    if (total_records_collected < expected_tasks) {
        LOG_WARN("  Incomplete collection (%d / %d records)",
                 total_records_collected, expected_tasks);
    }

    LOG_INFO("=== Performance Data Collection Complete ===");
}

void DeviceRunner::print_performance_data() {
    if (collected_perf_records_.empty()) {
        LOG_INFO("=== No Performance Data to Print ===");
        return;
    }

    LOG_INFO("=== Performance Records Detail ===");

    // Calculate min start time for normalization
    uint64_t min_time = UINT64_MAX;
    for (const auto& record : collected_perf_records_) {
        if (record.start_time < min_time) {
            min_time = record.start_time;
        }
    }

    // Print detailed records only in DEBUG mode
    LOG_DEBUG("  Base time (for normalization): %lu", min_time);
    LOG_DEBUG("");
    LOG_DEBUG("  Task execution records:");
    LOG_DEBUG("  ┌────────┬─────────┬─────────┬────────────┬──────────────────┬──────────────────┬──────────────┬──────────┐");
    LOG_DEBUG("  │ Task ID│ Func ID │ Core ID │ Core Type  │  Start (cycles)  │   End (cycles)   │Duration(cyc) │  Fanout  │");
    LOG_DEBUG("  ├────────┼─────────┼─────────┼────────────┼──────────────────┼──────────────────┼──────────────┼──────────┤");

    for (size_t i = 0; i < collected_perf_records_.size() && i < 50; i++) {  // Limit to first 50 for display
        const PerfRecord& record = collected_perf_records_[i];

        // Normalize times
        uint64_t norm_start = record.start_time - min_time;
        uint64_t norm_end = record.end_time - min_time;

        char line_buf[256];
        snprintf(line_buf, sizeof(line_buf),
                 "  │ %6u │ %7u │ %7u │ %10s │ %16lu │ %16lu │ %12lu │ %8u │",
                 record.task_id, record.func_id, record.core_id,
                 (record.core_type == CoreType::AIC ? "AIC" : "AIV"),
                 norm_start, norm_end, record.duration, record.fanout_count);
        LOG_DEBUG("%s", line_buf);
    }

    LOG_DEBUG("  └────────┴─────────┴─────────┴────────────┴──────────────────┴──────────────────┴──────────────┴──────────┘");

    if (collected_perf_records_.size() > 50) {
        LOG_DEBUG("  ... (%zu more records not shown)", collected_perf_records_.size() - 50);
    }

    // Calculate statistics
    uint64_t total_duration = 0;
    uint64_t max_duration = 0;
    uint64_t min_duration = UINT64_MAX;

    for (const auto& record : collected_perf_records_) {
        total_duration += record.duration;
        if (record.duration > max_duration) max_duration = record.duration;
        if (record.duration < min_duration) min_duration = record.duration;
    }

    double avg_duration = static_cast<double>(total_duration) / collected_perf_records_.size();

    LOG_INFO("");
    LOG_INFO("  Performance Statistics:");
    LOG_INFO("    Total tasks:     %zu", collected_perf_records_.size());
    LOG_INFO("    Avg duration:    %lu cycles", static_cast<uint64_t>(avg_duration));
    LOG_INFO("    Min duration:    %lu cycles", min_duration);
    LOG_INFO("    Max duration:    %lu cycles", max_duration);
    LOG_INFO("    Total duration:  %lu cycles", total_duration);

    LOG_INFO("=== Performance Data Print Complete ===");
}
