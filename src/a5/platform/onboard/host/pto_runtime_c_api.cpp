/**
 * PTO Runtime C API - Onboard (Hardware) Platform-Specific Functions (A5)
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
    if (aicpu_binary == NULL || aicpu_size == 0 || aicore_binary == NULL || aicore_size == 0) {
        return -1;
    }
    try {
        DeviceRunner& runner = DeviceRunner::get();

        // Convert to vectors for run()
        std::vector<uint8_t> aicpu_vec(aicpu_binary, aicpu_binary + aicpu_size);
        std::vector<uint8_t> aicore_vec(aicore_binary, aicore_binary + aicore_size);

        // Run the runtime (device initialization is handled internally)
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

        // Clean cached resources to prepare for next test
        DeviceRunner& runner = DeviceRunner::get();
        runner.clean_cache();

        // Call destructor (user will call free())
        r->~Runtime();
        return rc;
    } catch (...) {
        return -1;
    }
}

int set_device(int device_id) {
    try {
        DeviceRunner& runner = DeviceRunner::get();
        return runner.ensure_device_set(device_id);
    } catch (...) {
        return -1;
    }
}

} /* extern "C" */
