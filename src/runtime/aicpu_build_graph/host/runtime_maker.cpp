/**
 * Runtime Builder - aicpu_build_graph (host side)
 *
 * Provides init_runtime_impl and validate_runtime_impl functions.
 *
 * init_runtime_impl:
 *   - Automatically manages I/O tensor device memory using arg_types/arg_sizes
 *   - Marshals device pointers and scalars into runtime->orch_args[]
 *   - Embeds the AICPU orchestration plugin SO into the Runtime
 *
 * validate_runtime_impl (finalize_runtime_impl):
 *   - Copies recorded tensors back from device to host
 *   - Frees device memory
 */

#include <stddef.h>
#include <stdint.h>
#include <strings.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "device_runner.h"
#include "runtime.h"

// Argument type constants (must match ArgType in pto_runtime_c_api.h and bindings.py).
#ifndef ARG_SCALAR
#define ARG_SCALAR     0
#define ARG_INPUT_PTR  1
#define ARG_OUTPUT_PTR 2
#define ARG_INOUT_PTR  3
#endif

static void populate_kernel_addrs(Runtime* runtime) {
    if (runtime == nullptr) {
        return;
    }
    // Kernel binaries are registered via the platform C API (register_kernel),
    // which calls `Runtime::set_function_bin_addr(func_id, addr)` after upload.
    // That directly populates `Runtime::kernel_addrs[]`.
    bool saw_any = false;
    for (int func_id = 0; func_id < RUNTIME_MAX_FUNC_ID; ++func_id) {
        if (runtime->kernel_addrs[func_id] != 0) {
            saw_any = true;
            break;
        }
    }

    if (!saw_any) {
        std::cerr << "Warning: no registered kernels found; Runtime::kernel_addrs[] remains empty\n";
    }
}

static int parse_build_mode_env(const char* s, int default_mode) {
    if (s == nullptr || s[0] == '\0') {
        return default_mode;
    }
    // Accept either numeric or string values.
    if (strcmp(s, "0") == 0 || strcasecmp(s, "sequential") == 0) {
        return 0;
    }
    if (strcmp(s, "1") == 0 || strcasecmp(s, "concurrent") == 0) {
        return 1;
    }
    // Fall back to numeric parsing.
    char* end = nullptr;
    long v = strtol(s, &end, 10);
    if (end != s) {
        return (v != 0) ? 1 : 0;
    }
    return default_mode;
}

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initialize a pre-allocated runtime for aicpu_build_graph.
 *
 * This function:
 * 1. Automatically manages I/O tensor device memory using arg_types/arg_sizes
 *    (device_malloc, copy_to_device, record_tensor_pair, record_device_alloc)
 * 2. Marshals device pointers and scalars into runtime->orch_args[]
 * 3. Embeds the AICPU orchestration plugin SO into the Runtime
 *
 * The task graph is built on device by the orchestration plugin.
 *
 * @param runtime           Pointer to pre-constructed Runtime
 * @param orch_so_binary    AICPU orchestration plugin SO binary data
 * @param orch_so_size      Size of orchestration SO binary in bytes
 * @param orch_func_name    Name of the orchestration entry function
 * @param func_args         Arguments (host pointers for tensors, scalar values)
 * @param func_args_count   Number of arguments
 * @param arg_types         Per-argument type (ARG_SCALAR, ARG_INPUT_PTR, etc.)
 * @param arg_sizes         Per-argument byte size (0 for scalars)
 * @param kernel_func_ids   Array of kernel function IDs
 * @param kernel_binaries   Array of kernel binary data pointers
 * @param kernel_sizes      Array of kernel binary sizes
 * @param kernel_count      Number of kernels to register
 * @return 0 on success, -1 on failure
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
    int kernel_count) {
    if (runtime == nullptr) {
        std::cerr << "Error: Runtime pointer is null\n";
        return -1;
    }
    if (orch_so_binary == nullptr || orch_so_size == 0 || orch_func_name == nullptr) {
        std::cerr << "Error: Invalid orchestration parameters\n";
        return -1;
    }

    // Register kernel binaries via platform-provided upload function
    if (kernel_count > 0 && kernel_func_ids != NULL &&
        kernel_binaries != NULL && kernel_sizes != NULL) {
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

    // Clear any previous state.
    runtime->clear_tensor_pairs();
    runtime->clear_device_allocs();

    // --- Auto-manage I/O tensors and marshal orch_args[] ---
    std::cout << "\n=== Preparing Orchestration Args ===" << '\n';
    std::cout << "func_args_count: " << func_args_count << '\n';

    if (func_args_count > RUNTIME_MAX_ORCH_ARGS) {
        std::cerr << "Error: func_args_count (" << func_args_count
                  << ") exceeds RUNTIME_MAX_ORCH_ARGS (" << RUNTIME_MAX_ORCH_ARGS << ")\n";
        return -1;
    }

    for (int i = 0; i < func_args_count; i++) {
        int atype = (arg_types != nullptr) ? arg_types[i] : ARG_SCALAR;
        uint64_t asize = (arg_sizes != nullptr) ? arg_sizes[i] : 0;

        if (atype == ARG_SCALAR) {
            // Pass scalar value directly.
            runtime->orch_args[i] = func_args[i];
        } else {
            // Pointer argument: allocate device memory.
            void* host_ptr = reinterpret_cast<void*>(func_args[i]);
            size_t nbytes = static_cast<size_t>(asize);

            void* dev_ptr = runtime->host_api.device_malloc(nbytes);
            if (dev_ptr == nullptr) {
                std::cerr << "Error: device_malloc failed for arg " << i
                          << " (" << nbytes << " bytes)\n";
                return -1;
            }
            runtime->record_device_alloc(dev_ptr);

            // Copy input data to device.
            if (atype == ARG_INPUT_PTR || atype == ARG_INOUT_PTR) {
                int rc = runtime->host_api.copy_to_device(dev_ptr, host_ptr, nbytes);
                if (rc != 0) {
                    std::cerr << "Error: copy_to_device failed for arg " << i << '\n';
                    return -1;
                }
            }

            // Record output tensors for copy-back during finalize.
            if (atype == ARG_OUTPUT_PTR || atype == ARG_INOUT_PTR) {
                runtime->record_tensor_pair(host_ptr, dev_ptr, nbytes);
            }

            runtime->orch_args[i] = reinterpret_cast<uint64_t>(dev_ptr);
        }
    }
    runtime->orch_argc = func_args_count;

    // --- Embed AICPU orchestration plugin ---
    if (!runtime->try_set_aicpu_orch_so(orch_so_binary, orch_so_size)) {
        std::cerr << "Error: failed to embed AICPU orchestration plugin into Runtime "
                     "(size=" << orch_so_size << " bytes, max="
                  << RUNTIME_MAX_AICPU_ORCH_SO_SIZE << " bytes)\n";
        return -1;
    }
    memset(runtime->aicpu_orch_func_name, 0, sizeof(runtime->aicpu_orch_func_name));
    strncpy(runtime->aicpu_orch_func_name, orch_func_name,
            sizeof(runtime->aicpu_orch_func_name) - 1);

    std::cout << "Embedded orchestration plugin (" << orch_so_size
              << " bytes), entry: " << runtime->aicpu_orch_func_name << '\n';

    // --- Build mode ---
    const char* build_mode_env = std::getenv("PTO_AICPU_BUILD_GRAPH_BUILD_MODE");
    runtime->build_mode = parse_build_mode_env(build_mode_env, runtime->build_mode);
    std::cout << "aicpu_build_graph build_mode=" << runtime->build_mode
              << " (PTO_AICPU_BUILD_GRAPH_BUILD_MODE="
              << (build_mode_env ? build_mode_env : "<unset>") << ")\n";

    // Populate kernel_addrs[] for AICPU-side task creation.
    populate_kernel_addrs(runtime);

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
int validate_runtime_impl(Runtime* runtime) {
    if (runtime == nullptr) {
        std::cerr << "Error: Runtime pointer is null\n";
        return -1;
    }

    int rc = 0;

    std::cout << "\n=== Copying Results Back to Host ===" << '\n';

    // Copy all recorded tensors from device back to host
    TensorPair* tensor_pairs = runtime->get_tensor_pairs();
    int tensor_pair_count = runtime->get_tensor_pair_count();

    for (int i = 0; i < tensor_pair_count; i++) {
        const TensorPair& pair = tensor_pairs[i];
        int copy_rc = runtime->host_api.copy_from_device(pair.host_ptr, pair.dev_ptr, pair.size);
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

    DeviceAlloc* device_allocs = runtime->get_device_allocs();
    int device_alloc_count = runtime->get_device_alloc_count();

    auto is_recorded_alloc = [&](void* dev_ptr) -> bool {
        if (dev_ptr == nullptr) {
            return false;
        }
        for (int i = 0; i < device_alloc_count; ++i) {
            if (device_allocs[i].dev_ptr == dev_ptr) {
                return true;
            }
        }
        return false;
    };

    int freed_allocs = 0;
    for (int i = 0; i < device_alloc_count; ++i) {
        void* p = device_allocs[i].dev_ptr;
        if (p == nullptr) {
            continue;
        }
        runtime->host_api.device_free(p);
        freed_allocs++;
    }

    // Backward-compatible fallback: if orchestration didn't register allocations,
    // at least free the device pointers that were recorded for copy-back.
    int freed_pairs = 0;
    for (int i = 0; i < tensor_pair_count; i++) {
        void* p = tensor_pairs[i].dev_ptr;
        if (p == nullptr) {
            continue;
        }
        if (is_recorded_alloc(p)) {
            continue;
        }
        runtime->host_api.device_free(p);
        freed_pairs++;
    }

    std::cout << "Freed " << freed_allocs << " recorded device allocation(s) and " << freed_pairs
              << " tensor-pair device pointer(s)\n";

    // Note: AICPU orchestration plugin bytes are embedded in `Runtime` and do not
    // require device_free(). (They may be overwritten next run.)

    // Clear tensor pairs
    runtime->clear_tensor_pairs();
    runtime->clear_device_allocs();

    std::cout << "=== Finalize Complete ===" << std::endl;  // flush so output appears before Python continues

    return rc;
}

#ifdef __cplusplus
} /* extern "C" */
#endif
