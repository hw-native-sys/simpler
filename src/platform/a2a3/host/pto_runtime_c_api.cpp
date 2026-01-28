/**
 * PTO Runtime C API - Implementation
 *
 * Wraps C++ classes as opaque pointers, providing C interface for ctypes
 * bindings. Simplified single-concept model: Runtime only.
 */

#include "pto_runtime_c_api.h"

#include <new>  // for placement new
#include <vector>

#include "devicerunner.h"
#include "runtime.h"

// Internal typedef matching the C++ OrchestrationFunc signature
typedef int (*OrchestrationFuncInternal)(Runtime* runtime, uint64_t* args, int arg_count);

extern "C" {

/* ===========================================================================
 */
/* Runtime Implementation Functions (defined in runtimemaker.cpp) */
/* ===========================================================================
 */
int InitRuntimeImpl(Runtime* runtime,
                    OrchestrationFuncInternal orch_func,
                    uint64_t* func_args,
                    int func_args_count);
int ValidateRuntimeImpl(Runtime* runtime);

/* ===========================================================================
 */
/* Runtime API Implementation */
/* ===========================================================================
 */

size_t GetRuntimeSize(void) { return sizeof(Runtime); }

int InitRuntime(RuntimeHandle runtime,
                OrchestrationFunc orch_func,
                uint64_t* func_args,
                int func_args_count) {
    if (runtime == NULL) {
        return -1;
    }
    try {
        // Placement new to construct Runtime in user-allocated memory
        Runtime* r = new (runtime) Runtime();
        // Cast the C-style function pointer to internal C++ type
        OrchestrationFuncInternal orch_internal =
            reinterpret_cast<OrchestrationFuncInternal>(orch_func);
        return InitRuntimeImpl(r, orch_internal, func_args, func_args_count);
    } catch (...) {
        return -1;
    }
}

/* =========================================================================== */
/* Device Memory API Implementation */
/* =========================================================================== */

void* DeviceMalloc(size_t size) {
    try {
        DeviceRunner& runner = DeviceRunner::Get();
        return runner.AllocateTensor(size);
    } catch (...) {
        return NULL;
    }
}

void* DeviceMalloc_CApi(size_t size) {
    return DeviceMalloc(size);
}

void DeviceFree(void* devPtr) {
    if (devPtr == NULL) {
        return;
    }
    try {
        DeviceRunner& runner = DeviceRunner::Get();
        runner.FreeTensor(devPtr);
    } catch (...) {
        // Ignore errors during free
    }
}

void DeviceFree_CApi(void* devPtr) {
    DeviceFree(devPtr);
}

int CopyToDevice(void* devPtr, const void* hostPtr, size_t size) {
    if (devPtr == NULL || hostPtr == NULL) {
        return -1;
    }
    try {
        DeviceRunner& runner = DeviceRunner::Get();
        return runner.CopyToDevice(devPtr, hostPtr, size);
    } catch (...) {
        return -1;
    }
}

int CopyToDevice_CApi(void* devPtr, const void* hostPtr, size_t size) {
    return CopyToDevice(devPtr, hostPtr, size);
}

int CopyFromDevice(void* hostPtr, const void* devPtr, size_t size) {
    if (hostPtr == NULL || devPtr == NULL) {
        return -1;
    }
    try {
        DeviceRunner& runner = DeviceRunner::Get();
        return runner.CopyFromDevice(hostPtr, devPtr, size);
    } catch (...) {
        return -1;
    }
}

int CopyFromDevice_CApi(void* hostPtr, const void* devPtr, size_t size) {
    return CopyFromDevice(hostPtr, devPtr, size);
}

int launch_runtime(RuntimeHandle runtime,
    int aicpu_thread_num,
    int block_dim,
    int device_id,
    const uint8_t* aicpu_binary,
    size_t aicpu_size,
    const uint8_t* aicore_binary,
    size_t aicore_size) {
    if (runtime == NULL) {
        return -1;
    }
    if (aicpu_binary == NULL || aicpu_size == 0 || aicore_binary == NULL || aicore_size == 0) {
        return -1;
    }
    try {
        DeviceRunner& runner = DeviceRunner::Get();

        // Convert to vectors for Run()
        std::vector<uint8_t> aicpuVec(aicpu_binary, aicpu_binary + aicpu_size);
        std::vector<uint8_t> aicoreVec(aicore_binary, aicore_binary + aicore_size);

        // Run the runtime (device initialization is handled internally)
        Runtime* r = static_cast<Runtime*>(runtime);
        return runner.Run(*r, block_dim, device_id, aicpuVec, aicoreVec, aicpu_thread_num);
    } catch (...) {
        return -1;
    }
}

int FinalizeRuntime(RuntimeHandle runtime) {
    if (runtime == NULL) {
        return -1;
    }
    try {
        Runtime* r = static_cast<Runtime*>(runtime);
        int rc = ValidateRuntimeImpl(r);
        // Call destructor (user will call free())
        r->~Runtime();
        return rc;
    } catch (...) {
        return -1;
    }
}

int set_device(int device_id) {
    try {
        DeviceRunner& runner = DeviceRunner::Get();
        return runner.EnsureDeviceSet(device_id);
    } catch (...) {
        return -1;
    }
}

int RegisterKernel(int func_id, const uint8_t* bin_data, size_t bin_size) {
    if (bin_data == NULL || bin_size == 0) {
        return -1;
    }
    try {
        DeviceRunner& runner = DeviceRunner::Get();
        return runner.RegisterKernel(func_id, bin_data, bin_size);
    } catch (...) {
        return -1;
    }
}

} /* extern "C" */
