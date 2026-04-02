/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */
/**
 * Runtime Builder - Generic Implementation
 *
 * Provides init_runtime_impl and validate_runtime_impl functions that work with
 * pluggable orchestration functions for building task graphs.
 *
 * init_runtime_impl:
 *   - Calls orchestration function to build task graph
 *   - Orchestration is responsible for device memory management
 *
 * validate_runtime_impl (finalize_runtime_impl):
 *   - Copies recorded tensors back from device to host
 *   - Frees device memory
 */

#include <dlfcn.h>
#include <fcntl.h>
#include <stddef.h>
#include <stdint.h>
#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "callable.h"   // NOLINT(build/include_subdir)
#include "runtime.h"    // Includes unified_log.h and provides LOG_* macros  // NOLINT(build/include_subdir)
#include "task_args.h"  // NOLINT(build/include_subdir)

/**
 * Orchestration function signature.
 *
 * @param runtime    Pointer to Runtime to populate with tasks
 * @param orch_args  Separated tensor/scalar arguments
 * @return 0 on success, negative on error
 */
typedef int (*OrchestrationFunc)(Runtime *runtime, const ChipStorageTaskArgs &orch_args);

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initialize a pre-allocated runtime with dynamic orchestration.
 *
 * This function loads the orchestration SO from binary data via a temp file,
 * resolves the orchestration function via dlsym, then calls it to build the
 * task graph. The orchestration function is responsible for:
 * - Allocating device memory via runtime->host_api.device_malloc()
 * - Copying data to device via runtime->host_api.copy_to_device()
 * - Building the task graph
 * - Recording tensor pairs via runtime->record_tensor_pair()
 *
 * @param runtime   Pointer to pre-constructed Runtime
 * @param callable  ChipCallable containing orch binary, func_name, and child kernels
 * @param orch_args Separated tensor/scalar arguments
 * @return 0 on success, -1 on failure
 */
int init_runtime_impl(Runtime *runtime, const ChipCallable *callable, const ChipStorageTaskArgs *orch_args) {
    // Validate inputs
    if (runtime == nullptr) {
        LOG_ERROR("Runtime pointer is null");
        return -1;
    }

    // Register kernel binaries from ChipCallable children
    if (callable->child_count() > 0) {
        LOG_INFO("Registering %d kernel(s) in init_runtime_impl", callable->child_count());
        for (int32_t i = 0; i < callable->child_count(); i++) {
            int func_id = callable->child_func_id(i);
            const auto &kernel = callable->child(i);
            uint64_t addr = runtime->host_api.upload_kernel_binary(
                func_id, reinterpret_cast<const uint8_t *>(&kernel),
                CoreCallable::binary_data_offset() + kernel.binary_size()
            );
            if (addr == 0) {
                LOG_ERROR("Failed to upload kernel binary for func_id=%d", func_id);
                return -1;
            }
            runtime->set_function_bin_addr(func_id, addr);
        }
    }

    const uint8_t *orch_so_binary = static_cast<const uint8_t *>(callable->binary_data());
    size_t orch_so_size = callable->binary_size();
    const char *orch_func_name = callable->func_name();

    if (orch_so_binary == nullptr || orch_so_size == 0 || orch_func_name[0] == '\0') {
        LOG_ERROR("Invalid orchestration parameters");
        return -1;
    }

    // Load orchestration SO from binary data via temp file
    char fd_path[128];
    snprintf(fd_path, sizeof(fd_path), "/tmp/orch_so_%d.so", getpid());

    int fd = open(fd_path, O_WRONLY | O_CREAT | O_TRUNC, 0700);
    if (fd < 0) {
        LOG_ERROR("Failed to create temp SO file");
        return -1;
    }

    ssize_t written = write(fd, orch_so_binary, static_cast<size_t>(orch_so_size));
    if (written < 0 || static_cast<uint64_t>(written) != orch_so_size) {
        LOG_ERROR("Failed to write orchestration SO to temp file");
        close(fd);
        unlink(fd_path);
        return -1;
    }
    close(fd);

    void *handle = dlopen(fd_path, RTLD_NOW | RTLD_LOCAL);
    unlink(fd_path);
    if (handle == nullptr) {
        LOG_ERROR("dlopen failed: %s", dlerror());
        return -1;
    }

    dlerror();  // Clear any existing error
    OrchestrationFunc orch_func = reinterpret_cast<OrchestrationFunc>(dlsym(handle, orch_func_name));
    const char *dlsym_error = dlerror();
    if (dlsym_error != nullptr) {
        LOG_ERROR("dlsym failed for '%s': %s", orch_func_name, dlsym_error);
        dlclose(handle);
        return -1;
    }

    LOG_INFO("Loaded orchestration function: %s", orch_func_name);

    // Clear any previous tensor pairs
    runtime->clear_tensor_pairs();

    LOG_INFO("=== Calling Orchestration Function ===");

    LOG_DEBUG(
        "Args count: %d (%d tensors + %d scalars)", orch_args->tensor_count() + orch_args->scalar_count(),
        orch_args->tensor_count(), orch_args->scalar_count()
    );

    // Call orchestration function to build task graph
    // The orchestration function handles device memory allocation and copy-to-device
    int rc = orch_func(runtime, *orch_args);
    if (rc != 0) {
        LOG_ERROR("Orchestration function failed with code %d", rc);
        runtime->clear_tensor_pairs();
        dlclose(handle);
        return rc;
    }

    LOG_INFO("Runtime initialized. Ready for execution from Python.");

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
int validate_runtime_impl(Runtime *runtime) {
    if (runtime == nullptr) {
        LOG_ERROR("Runtime pointer is null");
        return -1;
    }

    int rc = 0;

    LOG_INFO("=== Copying Results Back to Host ===");

    // Copy all recorded tensors from device back to host
    TensorPair *tensor_pairs = runtime->get_tensor_pairs();
    int tensor_pair_count = runtime->get_tensor_pair_count();

    for (int i = 0; i < tensor_pair_count; i++) {
        const TensorPair &pair = tensor_pairs[i];
        int copy_rc = runtime->host_api.copy_from_device(pair.host_ptr, pair.dev_ptr, pair.size);
        if (copy_rc != 0) {
            LOG_ERROR("Failed to copy tensor %d from device: %d", i, copy_rc);
            rc = copy_rc;
            // Continue with cleanup anyway
        } else {
            LOG_DEBUG("Tensor %d: %zu bytes copied to host", i, pair.size);
        }
    }

    // Note: PrintHandshakeResults is now called in DeviceRunner's destructor

    // Cleanup device tensors
    LOG_INFO("=== Cleaning Up ===");
    for (int i = 0; i < tensor_pair_count; i++) {
        runtime->host_api.device_free(tensor_pairs[i].dev_ptr);
    }
    LOG_INFO("Freed %d device tensors", tensor_pair_count);

    // Cleanup kernel binaries
    int kernel_count = runtime->get_registered_kernel_count();
    for (int i = 0; i < kernel_count; i++) {
        int func_id = runtime->get_registered_kernel_func_id(i);
        runtime->host_api.remove_kernel_binary(func_id);
        runtime->set_function_bin_addr(func_id, 0);
    }
    if (kernel_count > 0) {
        LOG_INFO("Freed %d kernel binaries", kernel_count);
    }
    runtime->clear_registered_kernels();

    // Clear tensor pairs
    runtime->clear_tensor_pairs();

    LOG_INFO("=== Finalize Complete ===");

    return rc;
}

#ifdef __cplusplus
} /* extern "C" */
#endif
