/**
 * Device Runner Implementation
 *
 * This file implements the device execution utilities for launching and
 * managing AICPU and AICore kernels on Ascend devices.
 */

#include "devicerunner.h"

#include <cstring>
#include <iostream>
#include <vector>

#include "runtime.h"

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
            std::cerr << "Error: Alloc for device_args failed\n";
            return -1;
        }
        args.device_args = reinterpret_cast<DeviceArgs*>(device_args_dev);
    }
    // Copy host_device_args to device memory via device_args
    int rc =
        rtMemcpy(args.device_args, sizeof(DeviceArgs), &host_device_args, sizeof(DeviceArgs), RT_MEMCPY_HOST_TO_DEVICE);
    if (rc != 0) {
        std::cerr << "Error: rtMemcpy failed: " << rc << '\n';
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
            std::cerr << "Error: Alloc for runtime_args failed\n";
            return -1;
        }
        args.runtime_args = reinterpret_cast<Runtime*>(runtime_dev);
    }
    int rc = rtMemcpy(args.runtime_args, sizeof(Runtime), &host_runtime, sizeof(Runtime), RT_MEMCPY_HOST_TO_DEVICE);
    if (rc != 0) {
        std::cerr << "Error: rtMemcpy for runtime failed: " << rc << '\n';
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
        std::cerr << "Error: AICPU binary is empty\n";
        return -1;
    }

    size_t file_size = aicpu_so_binary.size();
    void* d_aicpu_data = allocator_->alloc(file_size);
    if (d_aicpu_data == nullptr) {
        std::cerr << "Error: Alloc failed for AICPU SO\n";
        return -1;
    }

    int rc = rtMemcpy(d_aicpu_data, file_size, aicpu_so_binary.data(), file_size, RT_MEMCPY_HOST_TO_DEVICE);
    if (rc != 0) {
        std::cerr << "Error: rtMemcpy failed: " << rc << '\n';
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
        std::cerr << "Error: rtSetDevice(" << device_id << ") failed: " << rc << '\n';
        return rc;
    }

    // Create streams
    rc = rtStreamCreate(&stream_aicpu_, 0);
    if (rc != 0) {
        std::cerr << "Error: rtStreamCreate (AICPU) failed: " << rc << '\n';
        return rc;
    }

    rc = rtStreamCreate(&stream_aicore_, 0);
    if (rc != 0) {
        std::cerr << "Error: rtStreamCreate (AICore) failed: " << rc << '\n';
        rtStreamDestroy(stream_aicpu_);
        stream_aicpu_ = nullptr;
        return rc;
    }

    std::cout << "DeviceRunner: device=" << device_id << " set, streams created\n";
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
        std::cerr << "Error: Device not set before loading binaries\n";
        return -1;
    }

    aicore_kernel_binary_ = aicore_kernel_binary;

    // Load AICPU SO
    int rc = so_info_.init(aicpu_so_binary, mem_alloc_);
    if (rc != 0) {
        std::cerr << "Error: AicpuSoInfo::init failed: " << rc << '\n';
        return rc;
    }

    // Initialize device args
    device_args_.aicpu_so_bin = so_info_.aicpu_so_bin;
    device_args_.aicpu_so_len = so_info_.aicpu_so_len;
    rc = kernel_args_.init_device_args(device_args_, mem_alloc_);
    if (rc != 0) {
        std::cerr << "Error: init_device_args failed: " << rc << '\n';
        so_info_.finalize();
        return rc;
    }

    binaries_loaded_ = true;
    std::cout << "DeviceRunner: binaries loaded\n";
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
    // Ensure device is initialized (lazy initialization)
    int rc = ensure_device_initialized(device_id, aicpu_so_binary, aicore_kernel_binary);
    if (rc != 0) {
        std::cerr << "Error: ensure_device_initialized failed: " << rc << '\n';
        return rc;
    }

    // Commit any pending kernels to device memory
    rc = commit_kernels();
    if (rc != 0) {
        std::cerr << "Error: commit_kernels failed: " << rc << '\n';
        return rc;
    }

    // Calculate execution parameters
    block_dim_ = block_dim;

    int num_ai_core = block_dim * cores_per_blockdim_;
    // Initialize handshake buffers in runtime
    if (num_ai_core > RUNTIME_MAX_WORKER) {
        std::cerr << "Error: block_dim (" << block_dim << ") exceeds RUNTIME_MAX_WORKER (" << RUNTIME_MAX_WORKER << ")\n";
        return -1;
    }

    runtime.worker_count = num_ai_core;
    runtime.block_dim = block_dim;
    runtime.sche_cpu_num = launch_aicpu_num;

    // Calculate number of AIC cores (1/3 of total)
    int num_aic = block_dim;  // Round up for 1/3

    for (int i = 0; i < num_ai_core; i++) {
        runtime.workers[i].aicpu_ready = 0;
        runtime.workers[i].aicore_done = 0;
        runtime.workers[i].control = 0;
        runtime.workers[i].task = 0;
        runtime.workers[i].task_status = 0;
        // Set core type: first 1/3 are AIC (0), remaining 2/3 are AIV (1)
        runtime.workers[i].core_type = (i < num_aic) ? 0 : 1;
    }

    // Set function_bin_addr for all tasks (NEW - Runtime function pointer
    // dispatch)
    std::cout << "\n=== Setting function_bin_addr for Tasks ===" << '\n';
    for (int i = 0; i < runtime.get_task_count(); i++) {
        Task* task = runtime.get_task(i);
        if (task != nullptr) {
            uint64_t addr = get_function_bin_addr(task->func_id);
            task->function_bin_addr = addr;
            std::cout << "  Task " << i << " (func_id=" << task->func_id << ") -> function_bin_addr=0x" << std::hex
                      << addr << std::dec << '\n';
        }
    }
    std::cout << '\n';

    // Initialize runtime args
    rc = kernel_args_.init_runtime_args(runtime, mem_alloc_);
    if (rc != 0) {
        std::cerr << "Error: init_runtime_args failed: " << rc << '\n';
        return rc;
    }

    // Launch AICPU init kernel
    rc = launch_aicpu_kernel(stream_aicpu_, &kernel_args_.args, "DynTileFwkKernelServerInit", 1);
    if (rc != 0) {
        std::cerr << "Error: launch_aicpu_kernel (init) failed: " << rc << '\n';
        kernel_args_.finalize_runtime_args();
        return rc;
    }

    // Launch AICPU main kernel
    rc = launch_aicpu_kernel(stream_aicpu_, &kernel_args_.args, "DynTileFwkKernelServer", launch_aicpu_num);
    if (rc != 0) {
        std::cerr << "Error: launch_aicpu_kernel (main) failed: " << rc << '\n';
        kernel_args_.finalize_runtime_args();
        return rc;
    }

    // Launch AICore kernel
    rc = launch_aicore_kernel(stream_aicore_, kernel_args_.args.runtime_args);
    if (rc != 0) {
        std::cerr << "Error: launch_aicore_kernel failed: " << rc << '\n';
        kernel_args_.finalize_runtime_args();
        return rc;
    }

    // Synchronize streams
    rc = rtStreamSynchronize(stream_aicpu_);
    if (rc != 0) {
        std::cerr << "Error: rtStreamSynchronize (AICPU) failed: " << rc << '\n';
        kernel_args_.finalize_runtime_args();
        return rc;
    }

    rc = rtStreamSynchronize(stream_aicore_);
    if (rc != 0) {
        std::cerr << "Error: rtStreamSynchronize (AICore) failed: " << rc << '\n';
        kernel_args_.finalize_runtime_args();
        return rc;
    }

    // Cleanup runtime args
    kernel_args_.finalize_runtime_args();

    return 0;
}

void DeviceRunner::print_handshake_results(Runtime& runtime) {
    if (stream_aicpu_ == nullptr || runtime.worker_count == 0) {
        return;
    }

    size_t total_size = sizeof(Handshake) * runtime.worker_count;
    rtMemcpy(runtime.workers, total_size, kernel_args_.args.runtime_args->workers, total_size, RT_MEMCPY_DEVICE_TO_HOST);

    std::cout << "Handshake results for " << runtime.worker_count << " cores:" << std::endl;
    for (int i = 0; i < runtime.worker_count; i++) {
        std::cout << "  Core " << i << ": aicore_done=" << runtime.workers[i].aicore_done
                  << " aicpu_ready=" << runtime.workers[i].aicpu_ready << " control=" << runtime.workers[i].control
                  << " task=" << runtime.workers[i].task << std::endl;
    }
}

int DeviceRunner::finalize() {
    if (stream_aicpu_ == nullptr) {
        return 0;
    }

    // Cleanup kernel args (device_args, runtime_args if any)
    kernel_args_.finalize_device_args();

    // Cleanup AICPU SO
    so_info_.finalize();

    // Clear kernel address mapping and pending kernels
    func_id_to_addr_.clear();
    pending_kernels_.clear();
    kernels_committed_ = false;
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

    // Free all remaining allocations (including handshake buffer and binGmAddr)
    mem_alloc_.finalize();

    device_id_ = -1;
    aicore_kernel_binary_.clear();

    std::cout << "DeviceRunner finalized\n";
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
        std::cerr << "Error: AICore kernel binary is empty\n";
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
        std::cerr << "rtRegisterAllKernel失败: " << rc << '\n';
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
        std::cerr << "rtKernelLaunchWithHandleV2失败: " << rc << '\n';
        return rc;
    }

    return rc;
}

// =============================================================================
// Kernel Binary Registration (Python provides pre-extracted .text section)
// =============================================================================

int DeviceRunner::register_kernel(int func_id, const uint8_t* bin_data, size_t bin_size) {
    if (bin_data == nullptr || bin_size == 0) {
        std::cerr << "Error: Invalid kernel binary data\n";
        return -1;
    }

    std::cout << "Registering kernel: func_id=" << func_id << ", size=" << bin_size << " bytes\n";

    // Store kernel binary in host memory (will be copied to device in
    // commit_kernels)
    PendingKernel kernel;
    kernel.data.assign(bin_data, bin_data + bin_size);
    pending_kernels_[func_id] = std::move(kernel);

    // If already initialized and committed, need to copy this kernel
    // immediately
    if (stream_aicpu_ != nullptr && kernels_committed_) {
        // Re-commit just this kernel
        kernels_committed_ = false;  // Force re-commit
    }

    return 0;
}

int DeviceRunner::commit_kernels() {
    if (kernels_committed_) {
        return 0;  // Already committed
    }

    if (stream_aicpu_ == nullptr) {
        std::cerr << "Error: Cannot commit kernels - DeviceRunner not initialized\n";
        return -1;
    }

    std::cout << "Committing " << pending_kernels_.size() << " kernels to device memory\n";

    for (auto& [func_id, kernel] : pending_kernels_) {
        // Skip if already in func_id_to_addr_
        if (func_id_to_addr_.find(func_id) != func_id_to_addr_.end()) {
            continue;
        }

        size_t bin_size = kernel.data.size();

        // Step 1: Allocate device GM memory (size field + binary data)
        uint64_t alloc_size = sizeof(uint64_t) + bin_size;
        void* gm_addr = mem_alloc_.alloc(alloc_size);
        if (gm_addr == nullptr) {
            std::cerr << "Error: Failed to allocate device GM memory for "
                         "kernel func_id="
                      << func_id << '\n';
            return -1;
        }

        // Step 2: Build host buffer with CoreFunctionBin structure (size +
        // data)
        std::vector<uint8_t> host_buf(alloc_size);
        uint64_t* size_ptr = reinterpret_cast<uint64_t*>(host_buf.data());
        *size_ptr = bin_size;
        std::memcpy(host_buf.data() + sizeof(uint64_t), kernel.data.data(), bin_size);

        // Step 3: Copy to device
        int rc = rtMemcpy(gm_addr, alloc_size, host_buf.data(), alloc_size, RT_MEMCPY_HOST_TO_DEVICE);
        if (rc != 0) {
            std::cerr << "Error: rtMemcpy to device failed: " << rc << '\n';
            mem_alloc_.free(gm_addr);
            return rc;
        }

        // Step 4: Calculate function_bin_addr (skip size field to get actual code
        // address)
        uint64_t function_bin_addr = reinterpret_cast<uint64_t>(gm_addr) + sizeof(uint64_t);
        func_id_to_addr_[func_id] = function_bin_addr;

        std::cout << "  func_id=" << func_id << " -> function_bin_addr=0x" << std::hex << function_bin_addr << std::dec
                  << '\n';
    }

    kernels_committed_ = true;
    return 0;
}

uint64_t DeviceRunner::get_function_bin_addr(int func_id) {
    auto it = func_id_to_addr_.find(func_id);
    if (it == func_id_to_addr_.end()) {
        std::cerr << "Warning: function_bin_addr not found for func_id=" << func_id << '\n';
        return 0;
    }
    return it->second;
}
