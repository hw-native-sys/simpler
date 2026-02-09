/**
 * PTO Runtime C API - Implementation (Simulation)
 *
 * Wraps C++ classes as opaque pointers, providing C interface for ctypes.
 * This implementation uses thread-based simulation instead of actual device
 * execution.
 */

#include "host/pto_runtime_c_api.h"

#include <new>
#include <vector>

#include "device_runner.h"
#include "common/unified_log.h"
#include "runtime.h"

extern "C" {

/* ===========================================================================
 * Runtime Implementation Functions (defined in runtimemaker.cpp)
 * ===========================================================================
 */
int init_runtime_impl(Runtime* runtime,
                    const uint8_t* orch_so_binary,
                    size_t orch_so_size,
                    const char* orch_func_name,
                    uint64_t* func_args,
                    int func_args_count,
                    int* arg_types,
                    uint64_t* arg_sizes,
                    const int* kernel_func_ids,
                    const uint8_t* const* kernel_binaries,
                    const size_t* kernel_sizes,
                    int kernel_count);
int validate_runtime_impl(Runtime* runtime);

/* Forward declarations */
void* device_malloc(size_t size);
void device_free(void* dev_ptr);
int copy_to_device(void* dev_ptr, const void* host_ptr, size_t size);
int copy_from_device(void* host_ptr, const void* dev_ptr, size_t size);
uint64_t upload_kernel_binary_wrapper(int func_id, const uint8_t* bin_data, size_t bin_size);

/* ===========================================================================
 * Runtime API Implementation
 * ===========================================================================
 */

size_t get_runtime_size(void) {
    return sizeof(Runtime);
}

int init_runtime(RuntimeHandle runtime,
                const uint8_t* orch_so_binary,
                size_t orch_so_size,
                const char* orch_func_name,
                uint64_t* func_args,
                int func_args_count,
                int* arg_types,
                uint64_t* arg_sizes,
                const int* kernel_func_ids,
                const uint8_t* const* kernel_binaries,
                const size_t* kernel_sizes,
                int kernel_count) {
    if (runtime == NULL) {
        return -1;
    }
    // Note: orchestration parameters may be empty for device-side orchestration (rt2)
    // Validation is done in init_runtime_impl which knows the runtime type

    try {
        // Placement new to construct Runtime in user-allocated memory
        Runtime* r = new (runtime) Runtime();

        // Initialize host API function pointers
        r->host_api.device_malloc = device_malloc;
        r->host_api.device_free = device_free;
        r->host_api.copy_to_device = copy_to_device;
        r->host_api.copy_from_device = copy_from_device;
        r->host_api.upload_kernel_binary = upload_kernel_binary_wrapper;

        // Delegate kernel registration, SO loading, and orchestration to init_runtime_impl
        int result = init_runtime_impl(r, orch_so_binary, orch_so_size,
                               orch_func_name, func_args, func_args_count,
                               arg_types, arg_sizes,
                               kernel_func_ids, kernel_binaries,
                               kernel_sizes, kernel_count);

        if (result != 0) {
            r->~Runtime();
        }

        return result;
    } catch (...) {
        return -1;
    }
}

/* ===========================================================================
 * Device Memory API Implementation (Simulation)
 * ===========================================================================
 */

void* device_malloc(size_t size) {
    try {
        DeviceRunner& runner = DeviceRunner::get();
        return runner.allocate_tensor(size);
    } catch (...) {
        return NULL;
    }
}

void device_free(void* dev_ptr) {
    if (dev_ptr == NULL) {
        return;
    }
    try {
        DeviceRunner& runner = DeviceRunner::get();
        runner.free_tensor(dev_ptr);
    } catch (...) {
        // Ignore errors during free
    }
}

int copy_to_device(void* dev_ptr, const void* host_ptr, size_t size) {
    if (dev_ptr == NULL || host_ptr == NULL) {
        return -1;
    }
    try {
        DeviceRunner& runner = DeviceRunner::get();
        return runner.copy_to_device(dev_ptr, host_ptr, size);
    } catch (...) {
        return -1;
    }
}

int copy_from_device(void* host_ptr, const void* dev_ptr, size_t size) {
    if (host_ptr == NULL || dev_ptr == NULL) {
        return -1;
    }
    try {
        DeviceRunner& runner = DeviceRunner::get();
        return runner.copy_from_device(host_ptr, dev_ptr, size);
    } catch (...) {
        return -1;
    }
}

uint64_t upload_kernel_binary_wrapper(int func_id, const uint8_t* bin_data, size_t bin_size) {
    try {
        DeviceRunner& runner = DeviceRunner::get();
        return runner.upload_kernel_binary(func_id, bin_data, bin_size);
    } catch (...) {
        return 0;
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

    try {
        DeviceRunner& runner = DeviceRunner::get();

        // In simulation, binaries are ignored
        std::vector<uint8_t> aicpu_vec;
        std::vector<uint8_t> aicore_vec;

        if (aicpu_binary != NULL && aicpu_size > 0) {
            aicpu_vec.assign(aicpu_binary, aicpu_binary + aicpu_size);
        }
        if (aicore_binary != NULL && aicore_size > 0) {
            aicore_vec.assign(aicore_binary, aicore_binary + aicore_size);
        }

        Runtime* r = static_cast<Runtime*>(runtime);
        return runner.run(*r, block_dim, device_id, aicpu_vec, aicore_vec, aicpu_thread_num);
    } catch (...) {
        return -1;
    }
}

int finalize_runtime(RuntimeHandle runtime) {
    if (runtime == NULL) {
        return -1;
    }
    try {
        Runtime* r = static_cast<Runtime*>(runtime);
        int rc = validate_runtime_impl(r);

        // Finalize DeviceRunner (clears last_runtime_ to avoid dangling pointer)
        DeviceRunner& runner = DeviceRunner::get();
        runner.finalize();

        // Call destructor (user will call free())
        r->~Runtime();
        return rc;
    } catch (...) {
        return -1;
    }
}

int set_device(int device_id) {
    (void)device_id;  // Unused in simulation
    return 0;
}

int enable_runtime_profiling(RuntimeHandle runtime, int enabled) {
    if (runtime == NULL) {
        return -1;
    }
    try {
        Runtime* r = static_cast<Runtime*>(runtime);
        r->enable_profiling = (enabled != 0);
        return 0;
    } catch (...) {
        return -1;
    }
}

/* Note: register_kernel() has been internalized into init_runtime().
 * Kernel binaries are now passed directly to init_runtime() which handles
 * registration and stores addresses in Runtime's func_id_to_addr_[] array.
 */

void record_tensor_pair(RuntimeHandle runtime,
                       void* host_ptr,
                       void* dev_ptr,
                       size_t size) {
    if (runtime == NULL) {
        return;
    }
    Runtime* r = static_cast<Runtime*>(runtime);
    r->record_tensor_pair(host_ptr, dev_ptr, size);
}

}  // extern "C"
