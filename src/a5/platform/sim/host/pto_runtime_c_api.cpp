/**
 * PTO Runtime C API - Simulation Platform-Specific Functions (A5)
 *
 * Only launch_runtime(), finalize_runtime(), and set_device() differ between
 * onboard and simulation. All other C API functions are in
 * src/host/pto_runtime_c_api_common.cpp.
 */

#include "host/pto_runtime_c_api.h"

#include <vector>

#include "device_runner.h"
#include "runtime.h"

extern "C" {

int validate_runtime_impl(Runtime* runtime);

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

        // Clean cached resources before finalization
        DeviceRunner& runner = DeviceRunner::get();
        runner.clean_cache();

        // Finalize DeviceRunner (clears last_runtime_ to avoid dangling pointer)
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

} /* extern "C" */
