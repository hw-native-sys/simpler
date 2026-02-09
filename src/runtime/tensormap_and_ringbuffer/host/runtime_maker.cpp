/**
 * Runtime Builder - rt2 Implementation (Device Orchestration)
 *
 * Provides init_runtime_impl and validate_runtime_impl functions for rt2 runtime.
 * Supports device orchestration where AICPU thread 3 runs the orchestrator.
 *
 * init_runtime_impl:
 *   - Converts host pointers to device pointers based on arg_types
 *   - Copies orchestration SO to device memory
 *   - Sets up runtime state for device orchestration
 *
 * validate_runtime_impl:
 *   - Copies recorded tensors back from device to host
 *   - Frees device memory
 */

#include "../runtime/runtime.h"
#include "../runtime/pto_shared_memory.h"
#include "host/pto_runtime_c_api.h"  // For ArgType enum
#include <stdint.h>
#include <stddef.h>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>

// Max args for device orchestration
#ifndef RT2_MAX_DEVICE_ARGS
#define RT2_MAX_DEVICE_ARGS 32
#endif

// GM heap size for orchestrator output buffers
#ifndef RT2_GM_HEAP_SIZE
#define RT2_GM_HEAP_SIZE (512 * 1024)  // 512KB
#endif

/**
 * Initialize a pre-allocated runtime for device orchestration.
 *
 * For rt2 runtime, orchestration runs on AICPU thread 3 (device-side).
 * This function:
 * - Converts host pointers to device pointers based on arg_types
 * - Copies input data to device
 * - Records output tensors for copy-back
 * - Copies orchestration SO to device memory
 * - Sets up runtime state for device orchestration
 *
 * @param runtime           Pointer to pre-constructed Runtime
 * @param orch_so_binary    Orchestration shared library binary data
 * @param orch_so_size      Size of orchestration SO binary in bytes
 * @param orch_func_name    Name of the orchestration function (unused)
 * @param func_args         Arguments for orchestration
 * @param func_args_count   Number of arguments
 * @param arg_types         Array describing each argument's type (ArgType enum)
 * @param arg_sizes         Array of sizes for pointer arguments (0 for scalars)
 * @return 0 on success, -1 on failure
 */
extern "C" int init_runtime_impl(Runtime *runtime,
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
    // Suppress unused parameter warning
    (void)orch_func_name;

    // Validate inputs
    if (runtime == nullptr) {
        std::cerr << "Error: Runtime pointer is null\n";
        return -1;
    }

    // Register kernel binaries via platform-provided upload function
    if (kernel_count > 0 && kernel_func_ids != nullptr &&
        kernel_binaries != nullptr && kernel_sizes != nullptr) {
        std::cout << "Registering " << kernel_count << " kernel(s) in init_runtime_impl\n";
        for (int i = 0; i < kernel_count; i++) {
            uint64_t addr = runtime->host_api.upload_kernel_binary(
                kernel_func_ids[i], kernel_binaries[i], kernel_sizes[i]);
            if (addr == 0) {
                std::cerr << "Error: Failed to upload kernel binary for func_id=" << kernel_func_ids[i] << "\n";
                return -1;
            }
            runtime->set_function_bin_addr(kernel_func_ids[i], addr);
        }
    }

    if (orch_so_binary == nullptr || orch_so_size == 0) {
        std::cerr << "Error: Orchestration SO binary is required for device orchestration\n";
        return -1;
    }

    if (arg_types == nullptr || arg_sizes == nullptr) {
        std::cerr << "Error: arg_types and arg_sizes are required for device orchestration\n";
        return -1;
    }

    if (func_args_count > RT2_MAX_DEVICE_ARGS) {
        std::cerr << "Error: Too many arguments: " << func_args_count
                  << " (max " << RT2_MAX_DEVICE_ARGS << ")\n";
        return -1;
    }

    std::cout << "RT2 init: " << func_args_count << " arguments, device orchestration mode\n";

    // Convert host pointers to device pointers based on arg_types
    uint64_t device_args[RT2_MAX_DEVICE_ARGS];

    for (int i = 0; i < func_args_count; i++) {
        switch (arg_types[i]) {
            case ARG_SCALAR:
                // Scalar value, pass directly
                device_args[i] = func_args[i];
                break;

            case ARG_INPUT_PTR: {
                // Input pointer: allocate device memory, copy data
                void* host_ptr = reinterpret_cast<void*>(func_args[i]);
                size_t size = arg_sizes[i];

                void* dev_ptr = runtime->host_api.device_malloc(size);
                if (dev_ptr == nullptr) {
                    std::cerr << "Error: Failed to allocate device memory for arg " << i << "\n";
                    return -1;
                }

                int rc = runtime->host_api.copy_to_device(dev_ptr, host_ptr, size);
                if (rc != 0) {
                    std::cerr << "Error: Failed to copy arg " << i << " to device\n";
                    runtime->host_api.device_free(dev_ptr);
                    return -1;
                }

                device_args[i] = reinterpret_cast<uint64_t>(dev_ptr);
                // Record for cleanup (no copy-back needed)
                runtime->record_tensor_pair(nullptr, dev_ptr, size);
                std::cout << "  Arg " << i << " (input): " << size << " bytes at " << dev_ptr << "\n";
                break;
            }

            case ARG_OUTPUT_PTR: {
                // Output pointer: allocate device memory, record for copy-back
                void* host_ptr = reinterpret_cast<void*>(func_args[i]);
                size_t size = arg_sizes[i];

                void* dev_ptr = runtime->host_api.device_malloc(size);
                if (dev_ptr == nullptr) {
                    std::cerr << "Error: Failed to allocate device memory for arg " << i << "\n";
                    return -1;
                }

                device_args[i] = reinterpret_cast<uint64_t>(dev_ptr);
                // Record for copy-back during finalize
                runtime->record_tensor_pair(host_ptr, dev_ptr, size);
                std::cout << "  Arg " << i << " (output): " << size << " bytes at " << dev_ptr << "\n";
                break;
            }

            case ARG_INOUT_PTR: {
                // Input/output pointer: allocate, copy, record for copy-back
                void* host_ptr = reinterpret_cast<void*>(func_args[i]);
                size_t size = arg_sizes[i];

                void* dev_ptr = runtime->host_api.device_malloc(size);
                if (dev_ptr == nullptr) {
                    std::cerr << "Error: Failed to allocate device memory for arg " << i << "\n";
                    return -1;
                }

                int rc = runtime->host_api.copy_to_device(dev_ptr, host_ptr, size);
                if (rc != 0) {
                    std::cerr << "Error: Failed to copy arg " << i << " to device\n";
                    runtime->host_api.device_free(dev_ptr);
                    return -1;
                }

                device_args[i] = reinterpret_cast<uint64_t>(dev_ptr);
                // Record for copy-back during finalize
                runtime->record_tensor_pair(host_ptr, dev_ptr, size);
                std::cout << "  Arg " << i << " (inout): " << size << " bytes at " << dev_ptr << "\n";
                break;
            }

            default:
                std::cerr << "Error: Unknown arg_type " << arg_types[i] << " for arg " << i << "\n";
                return -1;
        }
    }

    // Copy orchestration SO to device memory (AICPU cannot access host memory)
    void* dev_so = runtime->host_api.device_malloc(orch_so_size);
    if (dev_so == nullptr) {
        std::cerr << "Error: Failed to allocate device memory for orchestration SO\n";
        return -1;
    }
    int rc = runtime->host_api.copy_to_device(dev_so, orch_so_binary, orch_so_size);
    if (rc != 0) {
        std::cerr << "Error: Failed to copy orchestration SO to device\n";
        runtime->host_api.device_free(dev_so);
        return -1;
    }
    // Copy SO binary into Runtime's internal storage (device_orch_so_storage_)
    // Pass the HOST pointer (orch_so_binary), not the device pointer (dev_so)
    // AICPU Thread 3 will read from get_device_orch_so_data() which returns this storage
    runtime->set_device_orch_so(orch_so_binary, orch_so_size);
    runtime->record_tensor_pair(nullptr, dev_so, orch_so_size);
    std::cout << "Orchestration SO: " << orch_so_size << " bytes copied to device\n";

    // Allocate GM heap for orchestrator output buffers
    void* gm_heap = runtime->host_api.device_malloc(RT2_GM_HEAP_SIZE);
    if (gm_heap == nullptr) {
        std::cerr << "Error: Failed to allocate GM heap\n";
        return -1;
    }
    runtime->record_tensor_pair(nullptr, gm_heap, RT2_GM_HEAP_SIZE);

    // Allocate PTO2 shared memory
    int32_t sm_size = pto2_sm_calculate_size(PTO2_TASK_WINDOW_SIZE, PTO2_DEP_LIST_POOL_SIZE);
    void* sm_ptr = runtime->host_api.device_malloc(static_cast<size_t>(sm_size));
    if (sm_ptr == nullptr) {
        std::cerr << "Error: Failed to allocate PTO2 shared memory\n";
        return -1;
    }
    runtime->set_pto2_gm_sm_ptr(sm_ptr);
    runtime->record_tensor_pair(nullptr, sm_ptr, static_cast<size_t>(sm_size));

    // Append gm_heap and heap_size to device_args for orchestration
    // Generic layout: original args + [gm_heap_ptr, gm_heap_size] at the end
    // Orchestration accesses them via args[arg_count - 2] and args[arg_count - 1]
    int extended_count = func_args_count + 2;
    if (extended_count > RT2_MAX_DEVICE_ARGS) {
        std::cerr << "Error: Too many args with gm_heap extension\n";
        return -1;
    }
    device_args[func_args_count] = reinterpret_cast<uint64_t>(gm_heap);
    device_args[func_args_count + 1] = static_cast<uint64_t>(RT2_GM_HEAP_SIZE);

    // Set up device orchestration state
    runtime->set_orch_built_on_host(false);
    runtime->set_orch_args(device_args, extended_count);

    std::cout << "Device orchestration ready: " << extended_count << " args\n";
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
extern "C" int validate_runtime_impl(Runtime *runtime) {
    if (runtime == nullptr) {
        std::cerr << "Error: Runtime pointer is null\n";
        return -1;
    }

    int rc = 0;

    std::cout << "\n=== Copying Results Back to Host ===\n";

    // Copy all recorded tensors from device back to host
    TensorPair* tensor_pairs = runtime->get_tensor_pairs();
    int tensor_pair_count = runtime->get_tensor_pair_count();

    std::cout << "Tensor pairs to process: " << tensor_pair_count << "\n";

    // PTO2 (device orchestration): graph output may be in packed buffer
    void* pto2_sm = runtime->get_pto2_gm_sm_ptr();
    uint64_t graph_out_ptr = 0;
    int32_t graph_out_size = 0;

    if (pto2_sm != nullptr) {
        // Copy header from device to host to read graph_output_ptr/size
        PTO2SharedMemoryHeader host_header;
        int hdr_rc = runtime->host_api.copy_from_device(&host_header, pto2_sm, sizeof(PTO2SharedMemoryHeader));
        if (hdr_rc == 0) {
            graph_out_ptr = host_header.graph_output_ptr;
            graph_out_size = host_header.graph_output_size;
            if (graph_out_ptr != 0) {
                std::cout << "Graph output buffer: ptr=0x" << std::hex << graph_out_ptr
                          << std::dec << ", size=" << graph_out_size << "\n";
            }
        } else {
            std::cerr << "Warning: Failed to copy PTO2 header from device\n";
        }
    }

    bool first_output_tensor = true;
    for (int i = 0; i < tensor_pair_count; i++) {
        const TensorPair& pair = tensor_pairs[i];

        // Skip if device pointer is null
        if (pair.dev_ptr == nullptr) {
            std::cerr << "Warning: Tensor " << i << " has null device pointer, skipping\n";
            continue;
        }

        // If host pointer is null, this is a device-only allocation (no copy-back)
        if (pair.host_ptr == nullptr) {
            std::cout << "Tensor " << i << ": device-only allocation (no copy-back)\n";
            continue;
        }

        void* src_ptr = pair.dev_ptr;
        size_t copy_size = pair.size;

        // Use graph_output_ptr for the first output tensor if available
        if (first_output_tensor && graph_out_ptr != 0 && graph_out_size > 0) {
            src_ptr = reinterpret_cast<void*>(static_cast<uintptr_t>(graph_out_ptr));
            copy_size = static_cast<size_t>(graph_out_size);
            std::cout << "Using packed output buffer for tensor " << i << "\n";
            first_output_tensor = false;
        }

        int copy_rc = runtime->host_api.copy_from_device(pair.host_ptr, src_ptr, copy_size);
        if (copy_rc != 0) {
            std::cerr << "Error: Failed to copy tensor " << i << " from device: " << copy_rc << "\n";
            rc = copy_rc;
        } else {
            std::cout << "Tensor " << i << ": " << pair.size << " bytes copied to host\n";
        }
    }

    // Cleanup device tensors
    std::cout << "\n=== Cleaning Up ===\n";
    for (int i = 0; i < tensor_pair_count; i++) {
        if (tensor_pairs[i].dev_ptr != nullptr) {
            runtime->host_api.device_free(tensor_pairs[i].dev_ptr);
        }
    }
    std::cout << "Freed " << tensor_pair_count << " device allocations\n";

    // Clear tensor pairs
    runtime->clear_tensor_pairs();

    std::cout << "=== Finalize Complete ===\n";

    return rc;
}
