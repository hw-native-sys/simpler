/**
 * Runtime Builder - Generic Implementation
 *
 * Provides InitRuntimeImpl and ValidateRuntimeImpl functions that work with
 * pluggable orchestration functions for building task graphs.
 *
 * InitRuntimeImpl:
 *   - Calls orchestration function to build task graph
 *   - Orchestration is responsible for device memory management
 *
 * ValidateRuntimeImpl (FinalizeRuntimeImpl):
 *   - Copies recorded tensors back from device to host
 *   - Frees device memory
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "runtime.h"
#include <stdint.h>
#include <stddef.h>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <iostream>
#include <sys/mman.h>
#include <unistd.h>

/**
 * Orchestration function signature.
 *
 * @param runtime   Pointer to Runtime to populate with tasks
 * @param args      Arguments array (host pointers, sizes, etc.)
 * @param arg_count Total number of arguments
 * @return 0 on success, negative on error
 */
typedef int (*OrchestrationFunc)(Runtime* runtime, uint64_t* args, int arg_count);

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initialize a pre-allocated runtime with dynamic orchestration.
 *
 * This function loads the orchestration SO from binary data using memfd_create,
 * resolves the orchestration function via dlsym, then calls it to build the
 * task graph. The orchestration function is responsible for:
 * - Allocating device memory via runtime->host_api.DeviceMalloc()
 * - Copying data to device via runtime->host_api.CopyToDevice()
 * - Building the task graph
 * - Recording tensor pairs via runtime->RecordTensorPair()
 *
 * @param runtime           Pointer to pre-constructed Runtime
 * @param orch_so_binary    Orchestration shared library binary data
 * @param orch_so_size      Size of orchestration SO binary in bytes
 * @param orch_func_name    Name of the orchestration function to call
 * @param func_args         Arguments for orchestration (host pointers, sizes, etc.)
 * @param func_args_count   Number of arguments
 * @return 0 on success, -1 on failure
 */
int InitRuntimeImpl(Runtime *runtime,
                    const uint8_t* orch_so_binary,
                    size_t orch_so_size,
                    const char* orch_func_name,
                    uint64_t* func_args,
                    int func_args_count) {
    // Validate inputs
    if (runtime == nullptr) {
        std::cerr << "Error: Runtime pointer is null\n";
        return -1;
    }
    if (orch_so_binary == nullptr || orch_so_size == 0 || orch_func_name == nullptr) {
        std::cerr << "Error: Invalid orchestration parameters\n";
        return -1;
    }

    // Load orchestration SO from binary data using memfd_create
    int fd = memfd_create("orch_so", MFD_CLOEXEC);
    if (fd < 0) {
        std::cerr << "Error: memfd_create failed\n";
        return -1;
    }

    ssize_t written = write(fd, orch_so_binary, orch_so_size);
    if (written < 0 || static_cast<size_t>(written) != orch_so_size) {
        std::cerr << "Error: Failed to write orchestration SO to memfd\n";
        close(fd);
        return -1;
    }

    char fd_path[64];
    snprintf(fd_path, sizeof(fd_path), "/proc/self/fd/%d", fd);

    void* handle = dlopen(fd_path, RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
        std::cerr << "Error: dlopen failed: " << dlerror() << "\n";
        close(fd);
        return -1;
    }

    dlerror();  // Clear any existing error
    OrchestrationFunc orch_func =
        reinterpret_cast<OrchestrationFunc>(dlsym(handle, orch_func_name));
    const char* dlsym_error = dlerror();
    if (dlsym_error != nullptr) {
        std::cerr << "Error: dlsym failed for '" << orch_func_name << "': " << dlsym_error << "\n";
        dlclose(handle);
        close(fd);
        return -1;
    }

    close(fd);

    std::cout << "Loaded orchestration function: " << orch_func_name << "\n";

    // Clear any previous tensor pairs
    runtime->ClearTensorPairs();

    std::cout << "\n=== Calling Orchestration Function ===" << '\n';
    std::cout << "Args count: " << func_args_count << '\n';

    // Call orchestration function to build task graph
    // The orchestration function handles device memory allocation and copy-to-device
    int rc = orch_func(runtime, func_args, func_args_count);
    if (rc != 0) {
        std::cerr << "Error: Orchestration function failed with code " << rc << '\n';
        runtime->ClearTensorPairs();
        dlclose(handle);
        return rc;
    }

    std::cout << "\nRuntime initialized. Ready for execution from Python.\n";

    // Note: We intentionally leak the dlopen handle to keep the SO loaded
    // for the lifetime of the process.

    return 0;
}

/**
 * Validate runtime results and cleanup.
 *
 * This function:
 * 1. Copies recorded tensors from device back to host
 * 2. Frees device memory for recorded tensors
 * 3. Clears tensor pair state
 *
 * @param runtime  Pointer to Runtime
 * @return 0 on success, -1 on failure
 */
int ValidateRuntimeImpl(Runtime *runtime) {
    if (runtime == nullptr) {
        std::cerr << "Error: Runtime pointer is null\n";
        return -1;
    }

    int rc = 0;

    std::cout << "\n=== Copying Results Back to Host ===" << '\n';

    // Copy all recorded tensors from device back to host
    TensorPair* tensor_pairs = runtime->GetTensorPairs();
    int tensor_pair_count = runtime->GetTensorPairCount();

    for (int i = 0; i < tensor_pair_count; i++) {
        const TensorPair& pair = tensor_pairs[i];
        int copy_rc = runtime->host_api.CopyFromDevice(pair.hostPtr, pair.devPtr, pair.size);
        if (copy_rc != 0) {
            std::cerr << "Error: Failed to copy tensor " << i << " from device: " << copy_rc << '\n';
            rc = copy_rc;
            // Continue with cleanup anyway
        } else {
            std::cout << "Tensor " << i << ": " << pair.size << " bytes copied to host\n";
        }
    }

    // Note: PrintHandshakeResults is now called in DeviceRunner's destructor

    // Cleanup device tensors
    std::cout << "\n=== Cleaning Up ===" << '\n';
    for (int i = 0; i < tensor_pair_count; i++) {
        runtime->host_api.DeviceFree(tensor_pairs[i].devPtr);
    }
    std::cout << "Freed " << tensor_pair_count << " device tensors\n";

    // Clear tensor pairs
    runtime->ClearTensorPairs();

    std::cout << "=== Finalize Complete ===" << '\n';

    return rc;
}

#ifdef __cplusplus
}  /* extern "C" */
#endif
