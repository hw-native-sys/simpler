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
    LOG_INFO_V0("Uploaded %zu tensor info entries (%zu bytes)", compact_tensor_info.size(), tensor_info_bytes);
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
    LOG_INFO_V0("Uploaded %zu tensor allocation ranges (%zu bytes)", builder.allocations.size(), allocation_bytes);
    return 0;
}

}  // namespace

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Stage the per-callable resources for the host_build_graph variant: upload
 * kernel binaries and dlopen the orchestration SO on the host. The dlopen
 * handle and resolved entry-symbol pointer are parked on the runtime via
 * `pending_host_dlopen_handle_` / `pending_host_orch_func_ptr_` so the
 * platform layer can hoist them into PreparedCallableState. Splitting this
 * out of init_runtime_impl is what the hbg prepare_callable / run_prepared
 * path rests on — the dlopen runs once per cid instead of every run.
 */
int prepare_callable_impl(Runtime *runtime, const ChipCallable *callable) {
    if (runtime == nullptr) {
        LOG_ERROR("Runtime pointer is null");
        return -1;
    }
    if (callable == nullptr) {
        LOG_ERROR("Callable pointer is null");
        return -1;
    }

    // Single-shot upload of the entire ChipCallable buffer. DeviceRunner
    // computes total size from contents, allocates device GM once, fixes up
    // each child's resolved_addr_ in an internal host scratch, H2Ds once,
    // and returns the device-side address of the ChipCallable header.
    // Callers then compute child device addresses via offset arithmetic and
    // write them to Runtime::func_id_to_addr_[] for AICPU dispatch.
    if (callable->child_count() > 0) {
        LOG_INFO_V0("Registering %d kernel(s) in prepare_callable_impl", callable->child_count());
        uint64_t chip_dev = runtime->host_api.upload_chip_callable_buffer(callable);
        if (chip_dev == 0) {
            LOG_ERROR("Failed to upload ChipCallable buffer");
            return -1;
        }
        constexpr size_t kHeaderSize = offsetof(ChipCallable, storage_);
        for (int32_t i = 0; i < callable->child_count(); i++) {
            int func_id = callable->child_func_id(i);
            if (func_id < 0 || func_id >= RUNTIME_MAX_FUNC_ID) {
                LOG_ERROR("func_id=%d is out of range [0, %d)", func_id, RUNTIME_MAX_FUNC_ID);
                return -1;
            }
            uint64_t child_dev = chip_dev + kHeaderSize + callable->child_offset(i);
            runtime->set_function_bin_addr(func_id, child_dev);
        }
    }

    const uint8_t *orch_so_binary = static_cast<const uint8_t *>(callable->binary_data());
    size_t orch_so_size = callable->binary_size();
    const char *orch_func_name = callable->func_name();

    if (orch_so_binary == nullptr || orch_so_size == 0 || orch_func_name[0] == '\0') {
        LOG_ERROR("Invalid orchestration parameters");
        return -1;
    }

    // Load orchestration SO from binary data via temp file. Held open across
    // the lifetime of the prepared callable; closed by
    // DeviceRunner::unregister_prepared_callable.
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

    dlerror();
    OrchestrationFunc orch_func = reinterpret_cast<OrchestrationFunc>(dlsym(handle, orch_func_name));
    const char *dlsym_error = dlerror();
    if (dlsym_error != nullptr) {
        LOG_ERROR("dlsym failed for '%s': %s", orch_func_name, dlsym_error);
        dlclose(handle);
        return -1;
    }

    LOG_INFO_V0("Loaded orchestration function: %s", orch_func_name);

    runtime->pending_host_dlopen_handle_ = handle;
    runtime->pending_host_orch_func_ptr_ = reinterpret_cast<void *>(orch_func);
    // hbg never uploads orch SO bytes to the device; clear the trb staging
    // fields so DeviceRunner::register_prepared_callable cannot mistake this
    // for a trb-shaped registration.
    runtime->pending_orch_so_data_ = nullptr;
    runtime->pending_orch_so_size_ = 0;
    return 0;
}

/**
 * Per-run binding for hbg: invoke the previously-resolved orchestration entry
 * point against the supplied args, then upload tensor info / allocation
 * storage. Assumes prepare_callable_impl populated
 * `pending_host_orch_func_ptr_` (either freshly during prepare_callable, or
 * via DeviceRunner::bind_prepared_callable_to_runtime when run_prepared
 * replays a prepared cid onto a fresh Runtime).
 */
int bind_prepared_to_runtime_impl(Runtime *runtime, const ChipStorageTaskArgs *orch_args) {
    if (runtime == nullptr) {
        LOG_ERROR("Runtime pointer is null");
        return -1;
    }
    if (orch_args == nullptr) {
        LOG_ERROR("orch_args pointer is null");
        return -1;
    }
    OrchestrationFunc orch_func = reinterpret_cast<OrchestrationFunc>(runtime->pending_host_orch_func_ptr_);
    if (orch_func == nullptr) {
        LOG_ERROR("bind_prepared_to_runtime_impl: host orch_func pointer is null");
        return -1;
    }

    runtime->clear_tensor_pairs();

    LOG_INFO_V0("=== Calling Orchestration Function ===");
    LOG_DEBUG(
        "Args count: %d (%d tensors + %d scalars)", orch_args->tensor_count() + orch_args->scalar_count(),
        orch_args->tensor_count(), orch_args->scalar_count()
    );

    TensorInfoBuilder tensor_info_builder;
    TensorAllocationBuilder tensor_allocation_builder;
    OrchestrationRuntimeImpl orchestration_runtime = {
        &k_orchestration_runtime_ops, runtime, &tensor_info_builder, &tensor_allocation_builder
    };

    int rc = orch_func(reinterpret_cast<OrchestrationRuntime *>(&orchestration_runtime), *orch_args);
    if (rc != 0) {
        LOG_ERROR("Orchestration function failed with code %d", rc);
        runtime->clear_tensor_pairs();
        return rc;
    }

    rc = upload_tensor_allocation_storage(runtime, tensor_allocation_builder);
    if (rc != 0) {
        LOG_ERROR("Failed to upload tensor allocations: %d", rc);
        runtime->clear_tensor_pairs();
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
        return rc;
    }

    LOG_INFO_V0("Runtime initialized. Ready for execution from Python.");
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

    LOG_INFO_V0("=== Copying Results Back to Host ===");

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
    LOG_INFO_V0("=== Cleaning Up ===");
    for (int i = 0; i < tensor_pair_count; i++) {
        runtime->host_api.device_free(tensor_pairs[i].dev_ptr);
    }
    LOG_INFO_V0("Freed %d device tensors", tensor_pair_count);

    // Clear the per-run dispatch-table entries staged by prepare_callable_impl.
    // The underlying chip-callable device buffer is pool-managed by
    // DeviceRunner (keyed by content hash) and bulk-freed in
    // DeviceRunner::finalize(), mirroring the per-fid binary pool semantics
    // PR #710 introduced.
    int kernel_count = runtime->get_registered_kernel_count();
    for (int i = 0; i < kernel_count; i++) {
        int func_id = runtime->get_registered_kernel_func_id(i);
        runtime->set_function_bin_addr(func_id, 0);
    }
    if (kernel_count > 0) {
        LOG_INFO_V0("Cleared %d kernel dispatch-table entries", kernel_count);
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

    LOG_INFO_V0("=== Finalize Complete ===");

    return rc;
}

#ifdef __cplusplus
} /* extern "C" */
#endif
