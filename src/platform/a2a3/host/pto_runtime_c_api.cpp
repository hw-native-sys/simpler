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

extern "C" {

/* ===========================================================================
 */
/* Runtime Implementation Functions (defined in runtimemaker.cpp) */
/* ===========================================================================
 */
int InitRuntimeImpl(Runtime* runtime);
int ValidateRuntimeImpl(Runtime* runtime);

/* ===========================================================================
 */
/* Runtime API Implementation */
/* ===========================================================================
 */

size_t GetRuntimeSize(void) { return sizeof(Runtime); }

int InitRuntime(RuntimeHandle runtime) {
    if (runtime == NULL) {
        return -1;
    }
    try {
        // Placement new to construct Runtime in user-allocated memory
        Runtime* r = new (runtime) Runtime();
        return InitRuntimeImpl(r);
    } catch (...) {
        return -1;
    }
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
