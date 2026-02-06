/**
 * Runtime Builder - aicpu_build_graph (host side)
 *
 * Provides init_runtime_impl and validate_runtime_impl functions that work with
 * pluggable orchestration functions.
 *
 * init_runtime_impl:
 *   - Calls orchestration function to prepare device memory and marshal inputs
 *     for the AICPU graph builder (e.g. writes `runtime->orch_args[]`)
 *   - Orchestration is responsible for device memory management via
 *     `runtime->host_api.*`
 *
 * validate_runtime_impl (finalize_runtime_impl):
 *   - Copies recorded tensors back from device to host
 *   - Frees device memory
 */

#include <dlfcn.h>
#include <fcntl.h>
#include <stddef.h>
#include <stdint.h>
#include <strings.h>
#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "device_runner.h"
#include "runtime.h"

/**
 * Orchestration function signature.
 *
 * @param runtime   Pointer to Runtime to prepare (e.g. set orch_args[], record tensors)
 * @param args      Arguments array (host pointers, sizes, etc.)
 * @param arg_count Total number of arguments
 * @return 0 on success, negative on error
 */
typedef int (*OrchestrationFunc)(Runtime* runtime, uint64_t* args, int arg_count);

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
 * Initialize a pre-allocated runtime with dynamic orchestration.
 *
 * This function loads the orchestration SO from binary data via a temp file,
 * resolves the orchestration function via dlsym, then calls it.
 *
 * For `aicpu_build_graph`, the orchestration function is expected to:
 * - Allocating device memory via runtime->host_api.device_malloc()
 * - Copying data to device via runtime->host_api.copy_to_device()
 * - Recording tensor pairs via runtime->record_tensor_pair() (for copy-back)
 * - Marshalling a device-visible payload into runtime->orch_argc/runtime->orch_args[]
 *
 * The task graph itself is built later on device by `build_graph_aicpu(Runtime*)`.
 *
 * @param runtime           Pointer to pre-constructed Runtime
 * @param orch_so_binary    Orchestration shared library binary data
 * @param orch_so_size      Size of orchestration SO binary in bytes
 * @param orch_func_name    Name of the orchestration function to call
 * @param func_args         Arguments for orchestration (host pointers, sizes, etc.)
 * @param func_args_count   Number of arguments
 * @return 0 on success, -1 on failure
 */
int init_runtime_impl(Runtime* runtime,
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

    // Load orchestration SO from binary data via a unique temp file.
    // This mirrors the runner's pattern and avoids collisions across threads.
    char fd_path[] = "/tmp/orch_so_XXXXXX";
    int fd = mkstemp(fd_path);
    if (fd < 0) {
        std::cerr << "Error: Failed to create temp SO file (mkstemp)\n";
        return -1;
    }

    size_t off = 0;
    while (off < orch_so_size) {
        ssize_t n = write(fd, orch_so_binary + off, orch_so_size - off);
        if (n <= 0) {
            std::cerr << "Error: Failed to write orchestration SO to temp file\n";
            close(fd);
            unlink(fd_path);
            return -1;
        }
        off += static_cast<size_t>(n);
    }
    close(fd);

    void* handle = dlopen(fd_path, RTLD_NOW | RTLD_LOCAL);
    unlink(fd_path);
    if (handle == nullptr) {
        std::cerr << "Error: dlopen failed: " << dlerror() << "\n";
        return -1;
    }

    dlerror();  // Clear any existing error
    OrchestrationFunc orch_func = reinterpret_cast<OrchestrationFunc>(dlsym(handle, orch_func_name));
    const char* dlsym_error = dlerror();
    if (dlsym_error != nullptr) {
        std::cerr << "Error: dlsym failed for '" << orch_func_name << "': " << dlsym_error << "\n";
        dlclose(handle);
        return -1;
    }

    std::cout << "Loaded orchestration function: " << orch_func_name << "\n";

    // Clear any previous tensor pairs
    runtime->clear_tensor_pairs();

    // Optional: select build/schedule mode for this runtime instance.
    //
    // 0 = sequential build -> schedule
    // 1 = concurrent build || schedule (default)
    const char* build_mode_env = std::getenv("PTO_AICPU_BUILD_GRAPH_BUILD_MODE");
    runtime->build_mode = parse_build_mode_env(build_mode_env, runtime->build_mode);
    std::cout << "aicpu_build_graph build_mode=" << runtime->build_mode
              << " (PTO_AICPU_BUILD_GRAPH_BUILD_MODE=" << (build_mode_env ? build_mode_env : "<unset>") << ")\n";

    std::cout << "\n=== Calling Orchestration Function ===" << '\n';
    std::cout << "Args count: " << func_args_count << '\n';

    // Call orchestration function to build task graph
    // The orchestration function handles device memory allocation and copy-to-device
    int rc = orch_func(runtime, func_args, func_args_count);
    if (rc != 0) {
        std::cerr << "Error: Orchestration function failed with code " << rc << '\n';
        runtime->clear_tensor_pairs();
        dlclose(handle);
        return rc;
    }

    // Populate `kernel_addrs[]` for AICPU-side task creation (Challenge A).
    // This must happen after Python has called register_kernel() and before
    // launch_runtime() copies Runtime to device / starts AICPU execution.
    populate_kernel_addrs(runtime);

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

    std::cout << "=== Finalize Complete ===" << '\n';

    return rc;
}

#ifdef __cplusplus
} /* extern "C" */
#endif
