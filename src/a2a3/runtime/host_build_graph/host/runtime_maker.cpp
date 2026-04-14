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
#include <sys/stat.h>
#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "callable.h"
#include "orchestration_api.h"
#include "runtime.h"  // Includes unified_log.h and provides LOG_* macros
#include "task_args.h"

namespace {

struct OrchestrationRuntimeImpl {
    const OrchestrationRuntimeOps *ops;
    Runtime *runtime;
    struct TensorInfoBuilder *tensor_info_builder;
    struct TensorAllocationBuilder *tensor_allocation_builder;
};

struct TensorInfoBuilder {
    std::vector<std::vector<TensorInfo>> tensor_info_by_task;

    int set_tensor_info_to_task(int task_id, const TensorInfo *tensor_info, int tensor_count) {
        if (task_id < 0 || tensor_count < 0 || tensor_count > RUNTIME_MAX_ARGS) {
            return -1;
        }
        if (static_cast<size_t>(task_id) >= tensor_info_by_task.size()) {
            tensor_info_by_task.resize(static_cast<size_t>(task_id) + 1);
        }
        std::vector<TensorInfo> &task_info = tensor_info_by_task[static_cast<size_t>(task_id)];
        task_info.assign(tensor_info, tensor_info + tensor_count);
        return 0;
    }
};

struct TensorAllocationBuilder {
    std::vector<TensorAllocationInfo> allocations;

    void record_allocation(void *ptr, size_t size) {
        if (ptr == nullptr || size == 0) {
            return;
        }
        allocations.push_back({reinterpret_cast<uint64_t>(ptr), static_cast<uint64_t>(size)});
    }

    void erase_allocation(void *ptr) {
        if (ptr == nullptr) {
            return;
        }
        uint64_t base_addr = reinterpret_cast<uint64_t>(ptr);
        for (auto it = allocations.begin(); it != allocations.end(); ++it) {
            if (it->base_addr == base_addr) {
                allocations.erase(it);
                return;
            }
        }
    }
};

Runtime *unwrap_runtime(OrchestrationRuntime *runtime) {
    return reinterpret_cast<OrchestrationRuntimeImpl *>(runtime)->runtime;
}

TensorInfoBuilder *unwrap_tensor_info_builder(OrchestrationRuntime *runtime) {
    return reinterpret_cast<OrchestrationRuntimeImpl *>(runtime)->tensor_info_builder;
}

TensorAllocationBuilder *unwrap_tensor_allocation_builder(OrchestrationRuntime *runtime) {
    return reinterpret_cast<OrchestrationRuntimeImpl *>(runtime)->tensor_allocation_builder;
}

int runtime_add_task(OrchestrationRuntime *runtime, uint64_t *args, int num_args, int func_id, CoreType core_type) {
    return unwrap_runtime(runtime)->add_task(args, num_args, func_id, core_type);
}

int runtime_set_tensor_info_to_task(
    OrchestrationRuntime *runtime, int task_id, const TensorInfo *tensor_info, int tensor_count
) {
    Runtime *host_runtime = unwrap_runtime(runtime);
    if (task_id < 0 || task_id >= host_runtime->get_task_count()) {
        LOG_ERROR("Invalid task_id %d for task tensor info", task_id);
        return -1;
    }
    if (tensor_count == 0) {
        return 0;
    }
    if (tensor_info == nullptr) {
        LOG_ERROR("Task %d tensor info pointer is null", task_id);
        return -1;
    }
    return unwrap_tensor_info_builder(runtime)->set_tensor_info_to_task(task_id, tensor_info, tensor_count);
}

void runtime_add_successor(OrchestrationRuntime *runtime, int from_task, int to_task) {
    unwrap_runtime(runtime)->add_successor(from_task, to_task);
}

void runtime_record_tensor_pair(OrchestrationRuntime *runtime, void *host_ptr, void *dev_ptr, size_t size) {
    unwrap_runtime(runtime)->record_tensor_pair(host_ptr, dev_ptr, size);
}

int runtime_get_task_count(OrchestrationRuntime *runtime) { return unwrap_runtime(runtime)->get_task_count(); }

void runtime_print_runtime(OrchestrationRuntime *runtime) { unwrap_runtime(runtime)->print_runtime(); }

void *runtime_device_malloc(OrchestrationRuntime *runtime, size_t size) {
    void *ptr = unwrap_runtime(runtime)->host_api.device_malloc(size);
    unwrap_tensor_allocation_builder(runtime)->record_allocation(ptr, size);
    return ptr;
}

void runtime_device_free(OrchestrationRuntime *runtime, void *ptr) {
    unwrap_tensor_allocation_builder(runtime)->erase_allocation(ptr);
    unwrap_runtime(runtime)->host_api.device_free(ptr);
}

int runtime_copy_to_device(OrchestrationRuntime *runtime, void *dev_ptr, const void *host_ptr, size_t size) {
    return unwrap_runtime(runtime)->host_api.copy_to_device(dev_ptr, host_ptr, size);
}

const OrchestrationRuntimeOps k_orchestration_runtime_ops = {
    runtime_add_task,       runtime_set_tensor_info_to_task, runtime_add_successor, runtime_record_tensor_pair,
    runtime_get_task_count, runtime_print_runtime,           runtime_device_malloc, runtime_device_free,
    runtime_copy_to_device,
};

bool write_all_bytes(int fd, const uint8_t *data, size_t size) {
    size_t total_written = 0;
    while (total_written < size) {
        ssize_t written = write(fd, data + total_written, size - total_written);
        if (written <= 0) {
            return false;
        }
        total_written += static_cast<size_t>(written);
    }
    return true;
}

bool create_temp_so_file(const uint8_t *data, size_t size, std::string *out_path) {
    char path_template[] = "/tmp/orch_so_XXXXXX";
    int fd = mkstemp(path_template);
    if (fd < 0) {
        return false;
    }

    // dlopen requires the file to be executable; mkstemp creates 0600 (no exec bit)
    if (fchmod(fd, 0755) != 0) {
        close(fd);
        unlink(path_template);
        return false;
    }

    bool ok = write_all_bytes(fd, data, size);
    if (close(fd) != 0) {
        ok = false;
    }
    if (!ok) {
        unlink(path_template);
        return false;
    }

    *out_path = path_template;
    return true;
}

int upload_tensor_info_storage(Runtime *runtime, const TensorInfoBuilder &builder) {
    runtime->clear_tensor_info_storage();
    for (int task_id = 0; task_id < RUNTIME_MAX_TASKS; task_id++) {
        runtime->set_tensor_info_range(task_id, 0, 0);
    }

    int task_count = runtime->get_task_count();
    std::vector<TensorInfo> compact_tensor_info;
    for (int task_id = 0; task_id < task_count; task_id++) {
        const std::vector<TensorInfo> *task_info = nullptr;
        if (static_cast<size_t>(task_id) < builder.tensor_info_by_task.size()) {
            task_info = &builder.tensor_info_by_task[static_cast<size_t>(task_id)];
        }
        uint32_t offset = static_cast<uint32_t>(compact_tensor_info.size());
        uint16_t count = 0;
        if (task_info != nullptr) {
            count = static_cast<uint16_t>(task_info->size());
            compact_tensor_info.insert(compact_tensor_info.end(), task_info->begin(), task_info->end());
        }
        runtime->set_tensor_info_range(task_id, offset, count);
    }

    if (compact_tensor_info.empty()) {
        return 0;
    }

    size_t tensor_info_bytes = compact_tensor_info.size() * sizeof(TensorInfo);
    void *dev_tensor_info_storage = runtime->host_api.device_malloc(tensor_info_bytes);
    if (dev_tensor_info_storage == nullptr) {
        LOG_ERROR("Failed to allocate tensor info storage (%zu bytes)", tensor_info_bytes);
        return -1;
    }

    int rc = runtime->host_api.copy_to_device(dev_tensor_info_storage, compact_tensor_info.data(), tensor_info_bytes);
    if (rc != 0) {
        LOG_ERROR("Failed to copy tensor info storage to device: %d", rc);
        runtime->host_api.device_free(dev_tensor_info_storage);
        return rc;
    }

    runtime->set_tensor_info_storage(dev_tensor_info_storage, tensor_info_bytes);
    LOG_INFO("Uploaded %zu tensor info entries (%zu bytes)", compact_tensor_info.size(), tensor_info_bytes);
    return 0;
}

int upload_tensor_allocation_storage(Runtime *runtime, const TensorAllocationBuilder &builder) {
    runtime->clear_tensor_allocation_storage();
    if (builder.allocations.empty()) {
        return 0;
    }

    size_t allocation_bytes = builder.allocations.size() * sizeof(TensorAllocationInfo);
    void *dev_allocation_storage = runtime->host_api.device_malloc(allocation_bytes);
    if (dev_allocation_storage == nullptr) {
        LOG_ERROR("Failed to allocate tensor allocation storage (%zu bytes)", allocation_bytes);
        return -1;
    }

    int rc = runtime->host_api.copy_to_device(dev_allocation_storage, builder.allocations.data(), allocation_bytes);
    if (rc != 0) {
        LOG_ERROR("Failed to copy tensor allocation storage to device: %d", rc);
        runtime->host_api.device_free(dev_allocation_storage);
        return rc;
    }

    runtime->set_tensor_allocation_storage(
        dev_allocation_storage, static_cast<uint32_t>(builder.allocations.size()), allocation_bytes
    );
    LOG_INFO("Uploaded %zu tensor allocation ranges (%zu bytes)", builder.allocations.size(), allocation_bytes);
    return 0;
}

}  // namespace

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initialize a pre-allocated runtime with dynamic orchestration.
 *
 * This function loads the orchestration SO from binary data via a temp file,
 * resolves the orchestration function via dlsym, then calls it to build the
 * task graph. The orchestration function is responsible for:
 * - Allocating device memory via device_malloc()
 * - Copying data to device via copy_to_device()
 * - Building the task graph
 * - Recording tensor pairs via record_tensor_pair()
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
    std::string fd_path;
    if (!create_temp_so_file(orch_so_binary, orch_so_size, &fd_path)) {
        LOG_ERROR("Failed to create temp SO file");
        return -1;
    }

    void *handle = dlopen(fd_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    unlink(fd_path.c_str());
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

    TensorInfoBuilder tensor_info_builder;
    TensorAllocationBuilder tensor_allocation_builder;
    OrchestrationRuntimeImpl orchestration_runtime = {
        &k_orchestration_runtime_ops, runtime, &tensor_info_builder, &tensor_allocation_builder
    };

    // Call orchestration function to build task graph
    // The orchestration function handles device memory allocation and copy-to-device
    int rc = orch_func(reinterpret_cast<OrchestrationRuntime *>(&orchestration_runtime), *orch_args);
    if (rc != 0) {
        LOG_ERROR("Orchestration function failed with code %d", rc);
        runtime->clear_tensor_pairs();
        dlclose(handle);
        return rc;
    }

    rc = upload_tensor_allocation_storage(runtime, tensor_allocation_builder);
    if (rc != 0) {
        LOG_ERROR("Failed to upload tensor allocations: %d", rc);
        runtime->clear_tensor_pairs();
        dlclose(handle);
        return rc;
    }

    rc = upload_tensor_info_storage(runtime, tensor_info_builder);
    if (rc != 0) {
        LOG_ERROR("Failed to upload tensor info storage: %d", rc);
        if (runtime->get_tensor_allocation_storage() != nullptr) {
            runtime->host_api.device_free(runtime->get_tensor_allocation_storage());
            runtime->clear_tensor_allocation_storage();
        }
        runtime->clear_tensor_pairs();
        dlclose(handle);
        return rc;
    }

    LOG_INFO("Runtime initialized. Ready for execution from Python.");

    // Host orchestration is complete once orch_func returns. The task graph now
    // lives in Runtime, so the orchestration SO can be closed immediately.
    dlclose(handle);

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

    // Note: print_handshake_results() is called in DeviceRunner::run()

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

    if (runtime->get_tensor_info_storage() != nullptr) {
        runtime->host_api.device_free(runtime->get_tensor_info_storage());
        runtime->clear_tensor_info_storage();
    }
    if (runtime->get_tensor_allocation_storage() != nullptr) {
        runtime->host_api.device_free(runtime->get_tensor_allocation_storage());
        runtime->clear_tensor_allocation_storage();
    }

    // Clear tensor pairs
    runtime->clear_tensor_pairs();

    LOG_INFO("=== Finalize Complete ===");

    return rc;
}

#ifdef __cplusplus
} /* extern "C" */
#endif
