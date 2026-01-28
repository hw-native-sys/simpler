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

#include "runtime.h"
#include "devicerunner.h"
#include <stdint.h>
#include <stddef.h>
#include <new>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <iostream>
#include <string>

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
 * This function calls the orchestration function which is responsible for:
 * - Allocating device memory via runtime->DeviceMalloc()
 * - Copying data to device via runtime->CopyToDevice()
 * - Building the task graph
 * - Recording tensor pairs via runtime->RecordTensorPair()
 *
 * @param runtime         Pointer to pre-constructed Runtime
 * @param orch_func       Orchestration function to build task graph
 * @param func_args       Arguments for orchestration (host pointers, sizes, etc.)
 * @param func_args_count Number of arguments
 * @return 0 on success, -1 on failure
 */
int InitRuntimeImpl(Runtime *runtime,
                    OrchestrationFunc orch_func,
                    uint64_t* func_args,
                    int func_args_count) {
    // Validate inputs
    if (runtime == nullptr) {
        std::cerr << "Error: Runtime pointer is null\n";
        return -1;
    }
    if (orch_func == nullptr) {
        std::cerr << "Error: Orchestration function is null\n";
        return -1;
    }

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
        return rc;
    }

    std::cout << "\nRuntime initialized. Ready for execution from Python.\n";

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

    // Get DeviceRunner instance
    DeviceRunner& runner = DeviceRunner::Get();
    int rc = 0;

    std::cout << "\n=== Copying Results Back to Host ===" << '\n';

    // Copy all recorded tensors from device back to host
    TensorPair* tensor_pairs = runtime->GetTensorPairs();
    int tensor_pair_count = runtime->GetTensorPairCount();

    for (int i = 0; i < tensor_pair_count; i++) {
        const TensorPair& pair = tensor_pairs[i];
        int copy_rc = runner.CopyFromDevice(pair.hostPtr, pair.devPtr, pair.size);
        if (copy_rc != 0) {
            std::cerr << "Error: Failed to copy tensor " << i << " from device: " << copy_rc << '\n';
            rc = copy_rc;
            // Continue with cleanup anyway
        } else {
            std::cout << "Tensor " << i << ": " << pair.size << " bytes copied to host\n";
        }
    }

    // Print handshake results
    runner.PrintHandshakeResults(*runtime);

    // Cleanup device tensors
    std::cout << "\n=== Cleaning Up ===" << '\n';
    for (int i = 0; i < tensor_pair_count; i++) {
        runner.FreeTensor(tensor_pairs[i].devPtr);
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
