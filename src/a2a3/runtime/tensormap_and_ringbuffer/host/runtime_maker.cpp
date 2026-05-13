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
 * Runtime Builder - rt2 Implementation (Device Orchestration)
 *
 * Provides init_runtime_impl and validate_runtime_impl functions for rt2 runtime.
 * Supports device orchestration where AICPU thread 3 runs the orchestrator.
 *
 * init_runtime_impl:
 *   - Converts host tensor pointers to device pointers (all tensors copied both directions)
 *   - Copies orchestration SO to device memory
 *   - Sets up runtime state for device orchestration
 *
 * validate_runtime_impl:
 *   - Copies recorded tensors back from device to host
 *   - Frees device memory
 */

#include <stddef.h>
#include <stdint.h>
#include <sys/time.h>

#include <cerrno>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "../common/pto_runtime_status.h"
#include "../runtime/pto_shared_memory.h"
#include "../runtime/runtime.h"
#include "callable.h"
#include "common/platform_config.h"
#include "common/unified_log.h"

// Helper: return current time in milliseconds
static int64_t _now_ms() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return static_cast<int64_t>(tv.tv_sec) * 1000 + tv.tv_usec / 1000;
}

/**
 * Parse an environment variable as uint64_t with optional power-of-2 constraint.
 * Returns the parsed value on success, or 0 if unset or validation fails.
 */
static uint64_t parse_env_uint64(const char *name, uint64_t min_val, bool require_power_of_2) {
    const char *env = std::getenv(name);
    if (!env) return 0;
    char *endptr;
    errno = 0;
    uint64_t val = strtoull(env, &endptr, 10);
    if (errno == ERANGE || endptr == env || *endptr != '\0' || val < min_val) {
        LOG_WARN("%s=%s invalid (must be a valid integer >= %" PRIu64 "), ignored", name, env, min_val);
        return 0;
    }
    if (require_power_of_2 && (val & (val - 1)) != 0) {
        LOG_WARN("%s=%s invalid (must be a power of 2, >= %" PRIu64 "), ignored", name, env, min_val);
        return 0;
    }
    return static_cast<uint64_t>(val);
}

static int32_t pto2_read_runtime_status(Runtime *runtime, PTO2SharedMemoryHeader *host_header) {
    if (runtime == nullptr || host_header == nullptr) {
        return 0;
    }

    void *pto2_sm = runtime->get_gm_sm_ptr();
    if (pto2_sm == nullptr) {
        return 0;
    }

    int hdr_rc = runtime->host_api.copy_from_device(host_header, pto2_sm, sizeof(PTO2SharedMemoryHeader));
    if (hdr_rc != 0) {
        LOG_WARN("Failed to copy PTO2 header from device");
        return 0;
    }

    int32_t orch_error_code = host_header->orch_error_code.load(std::memory_order_relaxed);
    int32_t sched_error_code = host_header->sched_error_code.load(std::memory_order_relaxed);
    return runtime_status_from_error_codes(orch_error_code, sched_error_code);
}

/**
 * Stage the per-callable resources (kernel binaries + orchestration SO) into
 * the supplied runtime so a subsequent bind_prepared_to_runtime_impl can use
 * them. This is the cacheable half of init_runtime_impl: nothing here depends
 * on per-run argument values, so the prepare_callable / run_prepared split
 * lets us run this once per callable_id and amortize across runs.
 *
 * @param runtime   Pointer to pre-constructed Runtime (host_api populated)
 * @param callable  ChipCallable carrying the orch SO + child kernel binaries
 * @return 0 on success, -1 on failure
 */
extern "C" int prepare_callable_impl(Runtime *runtime, const ChipCallable *callable) {
    if (runtime == nullptr) {
        LOG_ERROR("Runtime pointer is null");
        return -1;
    }
    if (callable == nullptr) {
        LOG_ERROR("Callable pointer is null");
        return -1;
    }

    // Register kernel binaries from ChipCallable children
    if (callable->child_count() > 0) {
        LOG_INFO_V0("Registering %d kernel(s) in prepare_callable_impl", callable->child_count());
        for (int32_t i = 0; i < callable->child_count(); i++) {
            int func_id = callable->child_func_id(i);
            if (func_id < 0 || func_id >= RUNTIME_MAX_FUNC_ID) {
                LOG_ERROR("func_id=%d is out of range [0, %d)", func_id, RUNTIME_MAX_FUNC_ID);
                return -1;
            }
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
    runtime->set_device_orch_func_name(callable->func_name());
    runtime->set_device_orch_config_name(callable->config_name());

    if (orch_so_binary == nullptr || orch_so_size == 0) {
        LOG_ERROR("Orchestration SO binary is required for device orchestration");
        return -1;
    }

    // Stage the orchestration SO for DeviceRunner::prepare_orch_so to consume.
    runtime->pending_orch_so_data_ = orch_so_binary;
    runtime->pending_orch_so_size_ = orch_so_size;
    LOG_INFO_V0("Orchestration SO: %zu bytes staged (host-only)", orch_so_size);
    return 0;
}

/**
 * Per-run binding: build device-side argument storage (tensor copy-out, GM
 * heap, PTO2 shared memory) and publish it to the runtime. Assumes the
 * callable-side state (kernel binaries, orch SO bytes, func/config names)
 * is already populated by prepare_callable_impl.
 *
 * Splitting this from prepare_callable_impl matches the per-callable_id
 * design: register/run_prepared invokes this every call, while the prep
 * half runs only once per callable_id.
 *
 * @param runtime    Pointer to pre-constructed Runtime (host_api populated)
 * @param orch_args  Separated tensor/scalar arguments for this run
 * @return 0 on success, -1 on failure
 */
extern "C" int bind_prepared_to_runtime_impl(Runtime *runtime, const ChipStorageTaskArgs *orch_args) {
    if (runtime == nullptr) {
        LOG_ERROR("Runtime pointer is null");
        return -1;
    }
    if (orch_args == nullptr) {
        LOG_ERROR("orch_args pointer is null");
        return -1;
    }

    int tensor_count = orch_args->tensor_count();
    int scalar_count = orch_args->scalar_count();
    LOG_INFO_V0("RT2 bind: %d tensors + %d scalars, device orchestration mode", tensor_count, scalar_count);

    int64_t t_total_start = _now_ms();

    // Build device args: copy from input, replace host tensor pointers with device pointers
    ChipStorageTaskArgs device_args;

    int64_t t_args_start = _now_ms();
    for (int i = 0; i < tensor_count; i++) {
        ContinuousTensor t = orch_args->tensor(i);

        if (t.is_child_memory()) {
            LOG_INFO_V0("  Tensor %d: child memory, pass-through (0x%" PRIx64 ")", i, t.data);
            device_args.add_tensor(t);
            continue;
        }

        void *host_ptr = reinterpret_cast<void *>(static_cast<uintptr_t>(t.data));
        size_t size = static_cast<size_t>(t.nbytes());

        void *dev_ptr = runtime->host_api.device_malloc(size);
        if (dev_ptr == nullptr) {
            LOG_ERROR("Failed to allocate device memory for tensor %d", i);
            return -1;
        }

        int rc = runtime->host_api.copy_to_device(dev_ptr, host_ptr, size);
        if (rc != 0) {
            LOG_ERROR("Failed to copy tensor %d to device", i);
            runtime->host_api.device_free(dev_ptr);
            return -1;
        }
        runtime->record_tensor_pair(host_ptr, dev_ptr, size);
        LOG_INFO_V0("  Tensor %d: %zu bytes at %p", i, size, dev_ptr);

        t.data = reinterpret_cast<uint64_t>(dev_ptr);
        device_args.add_tensor(t);
    }
    for (int i = 0; i < scalar_count; i++) {
        device_args.add_scalar(orch_args->scalar(i));
    }
    int64_t t_args_end = _now_ms();

    // Read ready queue shard count from environment for AICPU scheduler
    {
        const char *env_shards = std::getenv("PTO2_READY_QUEUE_SHARDS");
        if (env_shards) {
            char *endptr;
            int64_t val = strtol(env_shards, &endptr, 10);
            if (endptr != env_shards && *endptr == '\0' && val >= 1 && val <= PLATFORM_MAX_AICPU_THREADS) {
                runtime->ready_queue_shards = static_cast<int>(val);
            } else {
                LOG_WARN(
                    "PTO2_READY_QUEUE_SHARDS=%s is invalid or out of range [1,%d], using default %d", env_shards,
                    PLATFORM_MAX_AICPU_THREADS, RUNTIME_DEFAULT_READY_QUEUE_SHARDS
                );
                runtime->ready_queue_shards = RUNTIME_DEFAULT_READY_QUEUE_SHARDS;
            }
        }
        LOG_INFO_V0("Ready queue shards: %d", runtime->ready_queue_shards);
    }

    // Read orchestrator-to-scheduler transition flag from environment
    {
        const char *env_val = std::getenv("PTO2_ORCH_TO_SCHED");
        if (env_val && (env_val[0] == '1' || env_val[0] == 't' || env_val[0] == 'T')) {
            runtime->orch_to_sched = true;
        }
        LOG_INFO_V0("Orchestrator-to-scheduler transition: %s", runtime->orch_to_sched ? "enabled" : "disabled");
    }

    // Read ring buffer size overrides from environment
    {
        runtime->task_window_size = parse_env_uint64("PTO2_RING_TASK_WINDOW", 4, true);
        runtime->heap_size = parse_env_uint64("PTO2_RING_HEAP", 1024, true);
        runtime->dep_pool_size = parse_env_uint64("PTO2_RING_DEP_POOL", 4, false);
        if (runtime->task_window_size || runtime->heap_size || runtime->dep_pool_size) {
            LOG_INFO_V0(
                "Ring buffer overrides: task_window=%" PRIu64 " heap=%" PRIu64 " dep_pool=%" PRIu64,
                (uint64_t)(runtime->task_window_size ? runtime->task_window_size : PTO2_TASK_WINDOW_SIZE),
                (uint64_t)(runtime->heap_size ? runtime->heap_size : PTO2_HEAP_SIZE),
                (uint64_t)(runtime->dep_pool_size ? runtime->dep_pool_size : PTO2_DEP_LIST_POOL_SIZE)
            );
        }
    }

    // Resolve effective sizes (env override or compile-time default)
    uint64_t eff_heap_size = runtime->heap_size ? runtime->heap_size : PTO2_HEAP_SIZE;
    uint64_t eff_task_window_size = runtime->task_window_size ? runtime->task_window_size : PTO2_TASK_WINDOW_SIZE;

    // Allocate GM heap for orchestrator output buffers (all rings combined)
    uint64_t total_heap_size = eff_heap_size * PTO2_MAX_RING_DEPTH;
    int64_t t_heap_start = _now_ms();
    void *gm_heap = runtime->host_api.device_malloc(total_heap_size);
    int64_t t_heap_end = _now_ms();
    if (gm_heap == nullptr) {
        LOG_ERROR("Failed to allocate GM heap");
        return -1;
    }
    runtime->record_tensor_pair(nullptr, gm_heap, total_heap_size);
    runtime->set_gm_heap(gm_heap);

    // Allocate PTO2 shared memory
    int64_t t_sm_start = _now_ms();
    uint64_t sm_size = PTO2SharedMemoryHandle::calculate_size(eff_task_window_size);
    void *sm_ptr = runtime->host_api.device_malloc(sm_size);
    int64_t t_sm_end = _now_ms();
    if (sm_ptr == nullptr) {
        LOG_ERROR("Failed to allocate PTO2 shared memory");
        return -1;
    }
    runtime->set_gm_sm_ptr(sm_ptr);
    runtime->record_tensor_pair(nullptr, sm_ptr, static_cast<size_t>(sm_size));

    // Set up device orchestration state
    runtime->set_orch_args(device_args);

    LOG_INFO_V0("Device orchestration ready: %d tensors + %d scalars", tensor_count, scalar_count);

    int64_t t_total_end = _now_ms();
    LOG_INFO_V0("TIMING: args_malloc_copy = %" PRId64 "ms", t_args_end - t_args_start);
    LOG_INFO_V0("TIMING: gm_heap_alloc(1GB) = %" PRId64 "ms", t_heap_end - t_heap_start);
    LOG_INFO_V0("TIMING: shared_mem_alloc = %" PRId64 "ms", t_sm_end - t_sm_start);
    LOG_INFO_V0("TIMING: total_init_runtime_impl = %" PRId64 "ms", t_total_end - t_total_start);

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
        LOG_ERROR("Runtime pointer is null");
        return -1;
    }

    int rc = 0;

    LOG_INFO_V0("=== Copying Results Back to Host ===");

    // Copy all recorded tensors from device back to host
    TensorPair *tensor_pairs = runtime->get_tensor_pairs();
    int tensor_pair_count = runtime->get_tensor_pair_count();

    LOG_INFO_V0("Tensor pairs to process: %d", tensor_pair_count);

    // PTO2 (device orchestration): graph output may be in packed buffer
    uint64_t graph_out_ptr = 0;
    uint64_t graph_out_size = 0;
    bool skip_tensor_copy_back = false;
    int32_t runtime_status = 0;
    PTO2SharedMemoryHeader host_header;
    memset(&host_header, 0, sizeof(host_header));

    runtime_status = pto2_read_runtime_status(runtime, &host_header);
    if (runtime_status != 0) {
        int32_t orch_error_code = host_header.orch_error_code.load(std::memory_order_relaxed);
        int32_t sched_error_code = host_header.sched_error_code.load(std::memory_order_relaxed);
        LOG_ERROR(
            "PTO2 runtime failed: orch_error_code=%d sched_error_code=%d runtime_status=%d", orch_error_code,
            sched_error_code, runtime_status
        );
        skip_tensor_copy_back = true;
    } else {
        graph_out_ptr = host_header.graph_output_ptr;
        graph_out_size = host_header.graph_output_size;
        if (graph_out_ptr != 0) {
            LOG_INFO_V0("Graph output buffer: ptr=0x%" PRIx64 ", size=%" PRIu64, graph_out_ptr, graph_out_size);
        }
    }

    if (skip_tensor_copy_back) {
        LOG_WARN("Skipping tensor copy-back because PTO2 runtime reported fatal status");
    } else {
        bool first_output_tensor = true;
        for (int i = 0; i < tensor_pair_count; i++) {
            const TensorPair &pair = tensor_pairs[i];

            // Skip if device pointer is null
            if (pair.dev_ptr == nullptr) {
                LOG_WARN("Tensor %d has null device pointer, skipping", i);
                continue;
            }

            // If host pointer is null, this is a device-only allocation (no copy-back)
            if (pair.host_ptr == nullptr) {
                LOG_INFO_V0("Tensor %d: device-only allocation (no copy-back)", i);
                continue;
            }

            void *src_ptr = pair.dev_ptr;
            size_t copy_size = pair.size;

            // Use graph_output_ptr for the first output tensor if available
            if (first_output_tensor && graph_out_ptr != 0 && graph_out_size > 0) {
                src_ptr = reinterpret_cast<void *>(static_cast<uintptr_t>(graph_out_ptr));
                copy_size = static_cast<size_t>(graph_out_size);
                LOG_INFO_V0("Using packed output buffer for tensor %d", i);
                first_output_tensor = false;
            }

            int copy_rc = runtime->host_api.copy_from_device(pair.host_ptr, src_ptr, copy_size);
            if (copy_rc != 0) {
                LOG_ERROR("Failed to copy tensor %d from device: %d", i, copy_rc);
                rc = copy_rc;
            } else {
                LOG_INFO_V0("Tensor %d: %zu bytes copied to host", i, pair.size);
            }
        }
    }

    // Cleanup device tensors
    LOG_INFO_V0("=== Cleaning Up ===");
    for (int i = 0; i < tensor_pair_count; i++) {
        if (tensor_pairs[i].dev_ptr != nullptr) {
            runtime->host_api.device_free(tensor_pairs[i].dev_ptr);
        }
    }
    LOG_INFO_V0("Freed %d device allocations", tensor_pair_count);

    // Cleanup kernel binaries
    int kernel_count = runtime->get_registered_kernel_count();
    for (int i = 0; i < kernel_count; i++) {
        int func_id = runtime->get_registered_kernel_func_id(i);
        runtime->host_api.remove_kernel_binary(func_id);
        runtime->set_function_bin_addr(func_id, 0);
    }
    if (kernel_count > 0) {
        LOG_INFO_V0("Freed %d kernel binaries", kernel_count);
    }
    runtime->clear_registered_kernels();

    // Clear tensor pairs
    runtime->clear_tensor_pairs();

    LOG_INFO_V0("=== Finalize Complete ===");

    if (rc == 0 && runtime_status != 0) {
        rc = runtime_status;
    }

    return rc;
}
