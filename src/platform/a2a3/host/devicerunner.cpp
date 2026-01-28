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

int KernelArgsHelper::InitDeviceArgs(const DeviceArgs& hostDeviceArgs, MemoryAllocator& allocator) {
    allocator_ = &allocator;

    // Allocate device memory for deviceArgs
    if (args.deviceArgs == nullptr) {
        uint64_t deviceArgsSize = sizeof(DeviceArgs);
        void* deviceArgsDev = allocator_->Alloc(deviceArgsSize);
        if (deviceArgsDev == nullptr) {
            std::cerr << "Error: Alloc for deviceArgs failed\n";
            return -1;
        }
        args.deviceArgs = reinterpret_cast<DeviceArgs*>(deviceArgsDev);
    }
    // Copy hostDeviceArgs to device memory via deviceArgs
    int rc =
        rtMemcpy(args.deviceArgs, sizeof(DeviceArgs), &hostDeviceArgs, sizeof(DeviceArgs), RT_MEMCPY_HOST_TO_DEVICE);
    if (rc != 0) {
        std::cerr << "Error: rtMemcpy failed: " << rc << '\n';
        allocator_->Free(args.deviceArgs);
        args.deviceArgs = nullptr;
        return rc;
    }
    return 0;
}

int KernelArgsHelper::FinalizeDeviceArgs() {
    if (args.deviceArgs != nullptr && allocator_ != nullptr) {
        int rc = allocator_->Free(args.deviceArgs);
        args.deviceArgs = nullptr;
        return rc;
    }
    return 0;
}

int KernelArgsHelper::InitRuntimeArgs(const Runtime& hostRuntime, MemoryAllocator& allocator) {
    allocator_ = &allocator;

    if (args.runtimeArgs == nullptr) {
        uint64_t runtimeSize = sizeof(Runtime);
        void* runtimeDev = allocator_->Alloc(runtimeSize);
        if (runtimeDev == nullptr) {
            std::cerr << "Error: Alloc for runtimeArgs failed\n";
            return -1;
        }
        args.runtimeArgs = reinterpret_cast<Runtime*>(runtimeDev);
    }
    int rc = rtMemcpy(args.runtimeArgs, sizeof(Runtime), &hostRuntime, sizeof(Runtime), RT_MEMCPY_HOST_TO_DEVICE);
    if (rc != 0) {
        std::cerr << "Error: rtMemcpy for runtime failed: " << rc << '\n';
        allocator_->Free(args.runtimeArgs);
        args.runtimeArgs = nullptr;
        return rc;
    }
    return 0;
}

int KernelArgsHelper::FinalizeRuntimeArgs() {
    if (args.runtimeArgs != nullptr && allocator_ != nullptr) {
        int rc = allocator_->Free(args.runtimeArgs);
        args.runtimeArgs = nullptr;
        return rc;
    }
    return 0;
}

// =============================================================================
// AicpuSoInfo Implementation
// =============================================================================

int AicpuSoInfo::Init(const std::vector<uint8_t>& aicpuSoBinary, MemoryAllocator& allocator) {
    allocator_ = &allocator;

    if (aicpuSoBinary.empty()) {
        std::cerr << "Error: AICPU binary is empty\n";
        return -1;
    }

    size_t fileSize = aicpuSoBinary.size();
    void* dAicpuData = allocator_->Alloc(fileSize);
    if (dAicpuData == nullptr) {
        std::cerr << "Error: Alloc failed for AICPU SO\n";
        return -1;
    }

    int rc = rtMemcpy(dAicpuData, fileSize, aicpuSoBinary.data(), fileSize, RT_MEMCPY_HOST_TO_DEVICE);
    if (rc != 0) {
        std::cerr << "Error: rtMemcpy failed: " << rc << '\n';
        allocator_->Free(dAicpuData);
        dAicpuData = nullptr;
        return rc;
    }

    aicpuSoBin = reinterpret_cast<uint64_t>(dAicpuData);
    aicpuSoLen = fileSize;
    return 0;
}

int AicpuSoInfo::Finalize() {
    if (aicpuSoBin != 0 && allocator_ != nullptr) {
        int rc = allocator_->Free(reinterpret_cast<void*>(aicpuSoBin));
        aicpuSoBin = 0;
        return rc;
    }
    return 0;
}

// =============================================================================
// DeviceRunner Implementation
// =============================================================================

DeviceRunner& DeviceRunner::Get() {
    static DeviceRunner runner;
    return runner;
}

DeviceRunner::~DeviceRunner() { Finalize(); }

int DeviceRunner::EnsureDeviceInitialized(
    int deviceId, const std::vector<uint8_t>& aicpuSoBinary, const std::vector<uint8_t>& aicoreKernelBinary) {
    // First ensure device is set and streams are created
    int rc = EnsureDeviceSet(deviceId);
    if (rc != 0) {
        return rc;
    }

    // Then ensure binaries are loaded
    return EnsureBinariesLoaded(aicpuSoBinary, aicoreKernelBinary);
}

int DeviceRunner::EnsureDeviceSet(int deviceId) {
    // Check if already initialized
    if (streamAicpu_ != nullptr) {
        return 0;
    }

    deviceId_ = deviceId;

    // Set device
    int rc = rtSetDevice(deviceId);
    if (rc != 0) {
        std::cerr << "Error: rtSetDevice(" << deviceId << ") failed: " << rc << '\n';
        return rc;
    }

    // Create streams
    rc = rtStreamCreate(&streamAicpu_, 0);
    if (rc != 0) {
        std::cerr << "Error: rtStreamCreate (AICPU) failed: " << rc << '\n';
        return rc;
    }

    rc = rtStreamCreate(&streamAicore_, 0);
    if (rc != 0) {
        std::cerr << "Error: rtStreamCreate (AICore) failed: " << rc << '\n';
        rtStreamDestroy(streamAicpu_);
        streamAicpu_ = nullptr;
        return rc;
    }

    std::cout << "DeviceRunner: device=" << deviceId << " set, streams created\n";
    return 0;
}

int DeviceRunner::EnsureBinariesLoaded(
    const std::vector<uint8_t>& aicpuSoBinary, const std::vector<uint8_t>& aicoreKernelBinary) {
    // Check if already loaded
    if (binariesLoaded_) {
        // Just update kernel binary if different
        if (aicoreKernelBinary_ != aicoreKernelBinary) {
            aicoreKernelBinary_ = aicoreKernelBinary;
        }
        return 0;
    }

    // Device must be set first
    if (streamAicpu_ == nullptr) {
        std::cerr << "Error: Device not set before loading binaries\n";
        return -1;
    }

    aicoreKernelBinary_ = aicoreKernelBinary;

    // Load AICPU SO
    int rc = soInfo_.Init(aicpuSoBinary, memAlloc_);
    if (rc != 0) {
        std::cerr << "Error: AicpuSoInfo::Init failed: " << rc << '\n';
        return rc;
    }

    // Initialize device args
    deviceArgs_.aicpuSoBin = soInfo_.aicpuSoBin;
    deviceArgs_.aicpuSoLen = soInfo_.aicpuSoLen;
    rc = kernelArgs_.InitDeviceArgs(deviceArgs_, memAlloc_);
    if (rc != 0) {
        std::cerr << "Error: InitDeviceArgs failed: " << rc << '\n';
        soInfo_.Finalize();
        return rc;
    }

    binariesLoaded_ = true;
    std::cout << "DeviceRunner: binaries loaded\n";
    return 0;
}

void* DeviceRunner::AllocateTensor(size_t bytes) { return memAlloc_.Alloc(bytes); }

void DeviceRunner::FreeTensor(void* devPtr) {
    if (devPtr != nullptr) {
        memAlloc_.Free(devPtr);
    }
}

int DeviceRunner::CopyToDevice(void* devPtr, const void* hostPtr, size_t bytes) {
    return rtMemcpy(devPtr, bytes, hostPtr, bytes, RT_MEMCPY_HOST_TO_DEVICE);
}

int DeviceRunner::CopyFromDevice(void* hostPtr, const void* devPtr, size_t bytes) {
    return rtMemcpy(hostPtr, bytes, devPtr, bytes, RT_MEMCPY_DEVICE_TO_HOST);
}

int DeviceRunner::Run(Runtime& runtime,
    int blockDim,
    int deviceId,
    const std::vector<uint8_t>& aicpuSoBinary,
    const std::vector<uint8_t>& aicoreKernelBinary,
    int launchAicpuNum) {
    // Ensure device is initialized (lazy initialization)
    int rc = EnsureDeviceInitialized(deviceId, aicpuSoBinary, aicoreKernelBinary);
    if (rc != 0) {
        std::cerr << "Error: EnsureDeviceInitialized failed: " << rc << '\n';
        return rc;
    }

    // Commit any pending kernels to device memory
    rc = CommitKernels();
    if (rc != 0) {
        std::cerr << "Error: CommitKernels failed: " << rc << '\n';
        return rc;
    }

    // Calculate execution parameters
    blockDim_ = blockDim;

    int numAiCore = blockDim * coresPerBlockdim_;
    // Initialize handshake buffers in runtime
    if (numAiCore > RUNTIME_MAX_WORKER) {
        std::cerr << "Error: blockDim (" << blockDim << ") exceeds RUNTIME_MAX_WORKER (" << RUNTIME_MAX_WORKER << ")\n";
        return -1;
    }

    runtime.worker_count = numAiCore;
    runtime.block_dim = blockDim;
    runtime.scheCpuNum = launchAicpuNum;

    // Calculate number of AIC cores (1/3 of total)
    int numAic = blockDim;  // Round up for 1/3

    for (int i = 0; i < numAiCore; i++) {
        runtime.workers[i].aicpu_ready = 0;
        runtime.workers[i].aicore_done = 0;
        runtime.workers[i].control = 0;
        runtime.workers[i].task = 0;
        runtime.workers[i].task_status = 0;
        // Set core type: first 1/3 are AIC (0), remaining 2/3 are AIV (1)
        runtime.workers[i].core_type = (i < numAic) ? 0 : 1;
    }

    // Set functionBinAddr for all tasks (NEW - Runtime function pointer
    // dispatch)
    std::cout << "\n=== Setting functionBinAddr for Tasks ===" << '\n';
    for (int i = 0; i < runtime.get_task_count(); i++) {
        Task* task = runtime.get_task(i);
        if (task != nullptr) {
            uint64_t addr = GetFunctionBinAddr(task->func_id);
            task->functionBinAddr = addr;
            std::cout << "  Task " << i << " (func_id=" << task->func_id << ") -> functionBinAddr=0x" << std::hex
                      << addr << std::dec << '\n';
        }
    }
    std::cout << '\n';

    // Initialize runtime args
    rc = kernelArgs_.InitRuntimeArgs(runtime, memAlloc_);
    if (rc != 0) {
        std::cerr << "Error: InitRuntimeArgs failed: " << rc << '\n';
        return rc;
    }

    // Launch AICPU init kernel
    rc = LaunchAiCpuKernel(streamAicpu_, &kernelArgs_.args, "DynTileFwkKernelServerInit", 1);
    if (rc != 0) {
        std::cerr << "Error: LaunchAiCpuKernel (init) failed: " << rc << '\n';
        kernelArgs_.FinalizeRuntimeArgs();
        return rc;
    }

    // Launch AICPU main kernel
    rc = LaunchAiCpuKernel(streamAicpu_, &kernelArgs_.args, "DynTileFwkKernelServer", launchAicpuNum);
    if (rc != 0) {
        std::cerr << "Error: LaunchAiCpuKernel (main) failed: " << rc << '\n';
        kernelArgs_.FinalizeRuntimeArgs();
        return rc;
    }

    // Launch AICore kernel
    rc = LaunchAicoreKernel(streamAicore_, kernelArgs_.args.runtimeArgs);
    if (rc != 0) {
        std::cerr << "Error: LaunchAicoreKernel failed: " << rc << '\n';
        kernelArgs_.FinalizeRuntimeArgs();
        return rc;
    }

    // Synchronize streams
    rc = rtStreamSynchronize(streamAicpu_);
    if (rc != 0) {
        std::cerr << "Error: rtStreamSynchronize (AICPU) failed: " << rc << '\n';
        kernelArgs_.FinalizeRuntimeArgs();
        return rc;
    }

    rc = rtStreamSynchronize(streamAicore_);
    if (rc != 0) {
        std::cerr << "Error: rtStreamSynchronize (AICore) failed: " << rc << '\n';
        kernelArgs_.FinalizeRuntimeArgs();
        return rc;
    }

    // Cleanup runtime args
    kernelArgs_.FinalizeRuntimeArgs();

    return 0;
}

void DeviceRunner::PrintHandshakeResults(Runtime& runtime) {
    if (streamAicpu_ == nullptr || runtime.worker_count == 0) {
        return;
    }

    size_t total_size = sizeof(Handshake) * runtime.worker_count;
    rtMemcpy(runtime.workers, total_size, kernelArgs_.args.runtimeArgs->workers, total_size, RT_MEMCPY_DEVICE_TO_HOST);

    std::cout << "Handshake results for " << runtime.worker_count << " cores:" << std::endl;
    for (int i = 0; i < runtime.worker_count; i++) {
        std::cout << "  Core " << i << ": aicore_done=" << runtime.workers[i].aicore_done
                  << " aicpu_ready=" << runtime.workers[i].aicpu_ready << " control=" << runtime.workers[i].control
                  << " task=" << runtime.workers[i].task << std::endl;
    }
}

int DeviceRunner::Finalize() {
    if (streamAicpu_ == nullptr) {
        return 0;
    }

    // Cleanup kernel args (deviceArgs, runtimeArgs if any)
    kernelArgs_.FinalizeDeviceArgs();

    // Cleanup AICPU SO
    soInfo_.Finalize();

    // Clear kernel address mapping and pending kernels
    funcIdToAddr_.clear();
    pendingKernels_.clear();
    kernelsCommitted_ = false;
    binariesLoaded_ = false;

    // Destroy streams
    if (streamAicpu_ != nullptr) {
        rtStreamDestroy(streamAicpu_);
        streamAicpu_ = nullptr;
    }
    if (streamAicore_ != nullptr) {
        rtStreamDestroy(streamAicore_);
        streamAicore_ = nullptr;
    }

    // Free all remaining allocations (including handshake buffer and binGmAddr)
    memAlloc_.Finalize();

    deviceId_ = -1;
    aicoreKernelBinary_.clear();

    std::cout << "DeviceRunner finalized\n";
    return 0;
}

int DeviceRunner::LaunchAiCpuKernel(rtStream_t stream, KernelArgs* kArgs, const char* kernelName, int aicpuNum) {
    struct Args {
        KernelArgs kArgs;
        char kernelName[32];
        const char soName[32] = {"libaicpu_extend_kernels.so"};
        const char opName[32] = {""};
    } args;

    args.kArgs = *kArgs;
    std::strncpy(args.kernelName, kernelName, sizeof(args.kernelName) - 1);
    args.kernelName[sizeof(args.kernelName) - 1] = '\0';

    rtAicpuArgsEx_t rtArgs;
    std::memset(&rtArgs, 0, sizeof(rtArgs));
    rtArgs.args = &args;
    rtArgs.argsSize = sizeof(args);
    rtArgs.kernelNameAddrOffset = offsetof(struct Args, kernelName);
    rtArgs.soNameAddrOffset = offsetof(struct Args, soName);

    return rtAicpuKernelLaunchExWithArgs(
        rtKernelType_t::KERNEL_TYPE_AICPU_KFC, "AST_DYN_AICPU", aicpuNum, &rtArgs, nullptr, stream, 0);
}

int DeviceRunner::LaunchAicoreKernel(rtStream_t stream, Runtime* runtime) {
    if (aicoreKernelBinary_.empty()) {
        std::cerr << "Error: AICore kernel binary is empty\n";
        return -1;
    }

    size_t binSize = aicoreKernelBinary_.size();
    const void* binData = aicoreKernelBinary_.data();

    rtDevBinary_t binary;
    std::memset(&binary, 0, sizeof(binary));
    binary.magic = RT_DEV_BINARY_MAGIC_ELF;
    binary.version = 0;
    binary.data = binData;
    binary.length = binSize;
    void* binHandle = nullptr;
    int rc = rtRegisterAllKernel(&binary, &binHandle);
    if (rc != RT_ERROR_NONE) {
        std::cerr << "rtRegisterAllKernel失败: " << rc << '\n';
        return rc;
    }

    struct Args {
        Runtime* runtime;
    };
    // Pass device address of Runtime to AICore
    Args args = {runtime};
    rtArgsEx_t rtArgs;
    std::memset(&rtArgs, 0, sizeof(rtArgs));
    rtArgs.args = &args;
    rtArgs.argsSize = sizeof(args);

    rtTaskCfgInfo_t cfg = {};
    cfg.schemMode = RT_SCHEM_MODE_BATCH;

    rc = rtKernelLaunchWithHandleV2(binHandle, 0, blockDim_, &rtArgs, nullptr, stream, &cfg);
    if (rc != RT_ERROR_NONE) {
        std::cerr << "rtKernelLaunchWithHandleV2失败: " << rc << '\n';
        return rc;
    }

    return rc;
}

// =============================================================================
// Kernel Binary Registration (Python provides pre-extracted .text section)
// =============================================================================

int DeviceRunner::RegisterKernel(int funcId, const uint8_t* binData, size_t binSize) {
    if (binData == nullptr || binSize == 0) {
        std::cerr << "Error: Invalid kernel binary data\n";
        return -1;
    }

    std::cout << "Registering kernel: func_id=" << funcId << ", size=" << binSize << " bytes\n";

    // Store kernel binary in host memory (will be copied to device in
    // CommitKernels)
    PendingKernel kernel;
    kernel.data.assign(binData, binData + binSize);
    pendingKernels_[funcId] = std::move(kernel);

    // If already initialized and committed, need to copy this kernel
    // immediately
    if (streamAicpu_ != nullptr && kernelsCommitted_) {
        // Re-commit just this kernel
        kernelsCommitted_ = false;  // Force re-commit
    }

    return 0;
}

int DeviceRunner::CommitKernels() {
    if (kernelsCommitted_) {
        return 0;  // Already committed
    }

    if (streamAicpu_ == nullptr) {
        std::cerr << "Error: Cannot commit kernels - DeviceRunner not initialized\n";
        return -1;
    }

    std::cout << "Committing " << pendingKernels_.size() << " kernels to device memory\n";

    for (auto& [funcId, kernel] : pendingKernels_) {
        // Skip if already in funcIdToAddr_
        if (funcIdToAddr_.find(funcId) != funcIdToAddr_.end()) {
            continue;
        }

        size_t binSize = kernel.data.size();

        // Step 1: Allocate device GM memory (size field + binary data)
        uint64_t allocSize = sizeof(uint64_t) + binSize;
        void* gmAddr = memAlloc_.Alloc(allocSize);
        if (gmAddr == nullptr) {
            std::cerr << "Error: Failed to allocate device GM memory for "
                         "kernel func_id="
                      << funcId << '\n';
            return -1;
        }

        // Step 2: Build host buffer with CoreFunctionBin structure (size +
        // data)
        std::vector<uint8_t> hostBuf(allocSize);
        uint64_t* sizePtr = reinterpret_cast<uint64_t*>(hostBuf.data());
        *sizePtr = binSize;
        std::memcpy(hostBuf.data() + sizeof(uint64_t), kernel.data.data(), binSize);

        // Step 3: Copy to device
        int rc = rtMemcpy(gmAddr, allocSize, hostBuf.data(), allocSize, RT_MEMCPY_HOST_TO_DEVICE);
        if (rc != 0) {
            std::cerr << "Error: rtMemcpy to device failed: " << rc << '\n';
            memAlloc_.Free(gmAddr);
            return rc;
        }

        // Step 4: Calculate functionBinAddr (skip size field to get actual code
        // address)
        uint64_t functionBinAddr = reinterpret_cast<uint64_t>(gmAddr) + sizeof(uint64_t);
        funcIdToAddr_[funcId] = functionBinAddr;

        std::cout << "  func_id=" << funcId << " -> functionBinAddr=0x" << std::hex << functionBinAddr << std::dec
                  << '\n';
    }

    kernelsCommitted_ = true;
    return 0;
}

uint64_t DeviceRunner::GetFunctionBinAddr(int funcId) {
    auto it = funcIdToAddr_.find(funcId);
    if (it == funcIdToAddr_.end()) {
        std::cerr << "Warning: functionBinAddr not found for func_id=" << funcId << '\n';
        return 0;
    }
    return it->second;
}
