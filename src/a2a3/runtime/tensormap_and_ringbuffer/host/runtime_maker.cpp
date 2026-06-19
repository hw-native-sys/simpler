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
 *   - Converts host tensor pointers to device pointers (all inputs copied H2D;
 *     only OUTPUT/INOUT tensors are copied back D2H)
 *   - Copies orchestration SO to device memory
 *   - Sets up runtime state for device orchestration
 *
 * validate_runtime_impl:
 *   - Copies OUTPUT/INOUT tensors back from device to host (read-only inputs
 *     are skipped)
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
#include "../runtime/pto_runtime2.h"
#include "../runtime/pto_shared_memory.h"
#include "../runtime/runtime.h"
#include "callable.h"
#include "common/platform_config.h"
#include "common/unified_log.h"
#include "utils/device_arena.h"
#include "prepare_callable_common.h"

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
 * the supplied runtime so a subsequent bind_callable_to_runtime_impl can use
 * them. This is the cacheable half of init_runtime_impl: nothing here depends
 * on per-run argument values, so the prepare_callable / run_prepared split
 * lets us run this once per callable_id and amortize across runs.
 *
 * @param runtime   Pointer to pre-constructed Runtime (host_api populated)
 * @param callable  ChipCallable carrying the orch SO + child kernel binaries
 * @return 0 on success, -1 on failure
 */
extern "C" int
prepare_callable_impl(const ChipCallable *callable, uint64_t (*upload_fn)(const void *), CallableArtifacts *out) {
    if (callable == nullptr) {
        LOG_ERROR("Callable pointer is null");
        return -1;
    }
    if (upload_fn == nullptr || out == nullptr) {
        LOG_ERROR("upload_fn or out is null");
        return -1;
    }
    *out = CallableArtifacts{};
    out->signature.assign(callable->signature_, callable->signature_ + callable->sig_count());

    LOG_INFO_V0("Registering %d kernel(s) in prepare_callable_impl", callable->child_count());
    if (upload_and_collect_child_addrs(callable, upload_fn, &out->kernel_addrs) != 0) {
        LOG_ERROR("Failed to upload ChipCallable buffer");
        return -1;
    }
    for (const ChildKernelAddr &c : out->kernel_addrs) {
        if (c.func_id < 0 || c.func_id >= RUNTIME_MAX_FUNC_ID) {
            LOG_ERROR("func_id=%d is out of range [0, %d)", c.func_id, RUNTIME_MAX_FUNC_ID);
            return -1;
        }
    }

    const uint8_t *orch_so_binary = static_cast<const uint8_t *>(callable->binary_data());
    size_t orch_so_size = callable->binary_size();

    if (orch_so_binary == nullptr || orch_so_size == 0) {
        LOG_ERROR("Orchestration SO binary is required for device orchestration");
        return -1;
    }

    out->orch_so_data = orch_so_binary;
    out->orch_so_size = orch_so_size;
    out->func_name = callable->func_name();
    out->config_name = callable->config_name();
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
extern "C" int bind_callable_to_runtime_impl(
    Runtime *runtime, const ChipStorageTaskArgs *orch_args, void *host_orch_func_ptr, const ArgDirection *signature,
    int sig_count, uint64_t ring_task_window, uint64_t ring_heap, uint64_t ring_dep_pool
) {
    if (runtime == nullptr) {
        LOG_ERROR("Runtime pointer is null");
        return -1;
    }
    if (orch_args == nullptr) {
        LOG_ERROR("orch_args pointer is null");
        return -1;
    }
    // trb runs orchestration on the device — there is no host-side orch
    // function pointer to invoke. The c_api signature accepts one for
    // symmetry with hbg; assert the trb-side invariant here.
    if (host_orch_func_ptr != nullptr) {
        LOG_ERROR("bind_callable_to_runtime_impl: trb does not accept a host_orch_func_ptr");
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

        // Pure write-only OUTPUT buffers carry no meaningful host content, so
        // the H2D copy-in is wasted. Zero them on-device instead (cheap HBM
        // memset, no PCIe) so any region the kernel leaves unwritten reads as 0
        // rather than pooled-allocator garbage. INOUT (read-before-write)
        // and IN keep the H2D copy. Falls back to copy_to_device if a backend
        // did not wire device_memset.
        bool is_pure_output = (signature != nullptr && i < sig_count && signature[i] == ArgDirection::OUT);
        int rc;
        if (is_pure_output && runtime->host_api.device_memset != nullptr) {
            rc = runtime->host_api.device_memset(dev_ptr, 0, size);
        } else {
            rc = runtime->host_api.copy_to_device(dev_ptr, host_ptr, size);
        }
        if (rc != 0) {
            LOG_ERROR("Failed to stage tensor %d to device", i);
            runtime->host_api.device_free(dev_ptr);
            return -1;
        }
        // Read-only INPUT tensors are never written by the kernel, so there is
        // no point copying them back D2H at the end. Index the signature
        // by the orch tensor index `i` (child_memory tensors are skipped above
        // but do not consume a separate signature slot — scalars follow the
        // tensor entries). Anything not provably IN keeps the safe default of
        // copying back.
        bool needs_copy_back = !(signature != nullptr && i < sig_count && signature[i] == ArgDirection::IN);
        runtime->tensor_pairs_.push_back({host_ptr, dev_ptr, size, needs_copy_back});
        LOG_INFO_V0("  Tensor %d: %zu bytes at %p", i, size, dev_ptr);

        t.data = reinterpret_cast<uint64_t>(dev_ptr);
        device_args.add_tensor(t);
    }
    for (int i = 0; i < scalar_count; i++) {
        device_args.add_scalar(orch_args->scalar(i));
    }
    int64_t t_args_end = _now_ms();

    // Read orchestrator-to-scheduler transition flag from environment
    {
        const char *env_val = std::getenv("PTO2_ORCH_TO_SCHED");
        if (env_val && (env_val[0] == '1' || env_val[0] == 't' || env_val[0] == 'T')) {
            runtime->orch_to_sched = true;
        }
        LOG_INFO_V0("Orchestrator-to-scheduler transition: %s", runtime->orch_to_sched ? "enabled" : "disabled");
    }

    // Ring buffer size overrides: per-task CallConfig value wins over the
    // env var; both fall back to the compile-time default when zero.
    {
        runtime->task_window_size =
            ring_task_window ? ring_task_window : parse_env_uint64("PTO2_RING_TASK_WINDOW", 4, true);
        runtime->heap_size = ring_heap ? ring_heap : parse_env_uint64("PTO2_RING_HEAP", 1024, true);
        runtime->dep_pool_size = ring_dep_pool ? ring_dep_pool : parse_env_uint64("PTO2_RING_DEP_POOL", 4, false);
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

    // Lay out the per-Worker static device arena. GM heap, PTO2 shared memory,
    // and the prebuilt runtime arena all live in a single backing allocation;
    // setup_static_arena reserves the three regions and commits in one shot.
    // Owned by DeviceRunner across runs — do NOT record in tensor_pairs_; the
    // free is deferred to DeviceRunner::finalize(). The runtime-arena size is
    // determined by replaying the reserve sequence on a host-side arena.
    uint64_t total_heap_size = eff_heap_size * PTO2_MAX_RING_DEPTH;
    uint64_t sm_size = PTO2SharedMemoryHandle::calculate_size(eff_task_window_size);
    // dep_pool_size comes from a uint64 env var; reject values that don't fit
    // the int32_t layout-sizing path rather than silently truncating.
    int32_t eff_dep_pool_capacity = PTO2_DEP_LIST_POOL_SIZE;
    if (runtime->dep_pool_size != 0) {
        if (runtime->dep_pool_size > static_cast<uint64_t>(INT32_MAX)) {
            LOG_ERROR("PTO2_RING_DEP_POOL=%" PRIu64 " exceeds INT32_MAX", runtime->dep_pool_size);
            return -1;
        }
        eff_dep_pool_capacity = static_cast<int32_t>(runtime->dep_pool_size);
    }

    int64_t t_prebuilt_start = _now_ms();
    DeviceArena host_arena;  // libc malloc backend by default
    PTO2RuntimeArenaLayout layout = runtime_reserve_layout(host_arena, eff_task_window_size, eff_dep_pool_capacity);
    if (host_arena.commit(DeviceArena::kDefaultBaseAlign) == nullptr) {
        LOG_ERROR("Failed to commit host arena for prebuilt runtime image");
        return -1;
    }

    int64_t t_setup_start = _now_ms();
    if (runtime->host_api.setup_static_arena(total_heap_size, sm_size, layout.arena_size) != 0) {
        LOG_ERROR("Failed to setup pooled static arena");
        return -1;
    }
    int64_t t_setup_end = _now_ms();

    int64_t t_heap_start = _now_ms();
    void *gm_heap = runtime->host_api.acquire_pooled_gm_heap();
    int64_t t_heap_end = _now_ms();
    if (gm_heap == nullptr) {
        LOG_ERROR("Failed to acquire pooled GM heap");
        return -1;
    }
    runtime->set_gm_heap(gm_heap);

    int64_t t_sm_start = _now_ms();
    void *sm_ptr = runtime->host_api.acquire_pooled_gm_sm();
    int64_t t_sm_end = _now_ms();
    if (sm_ptr == nullptr) {
        LOG_ERROR("Failed to acquire pooled PTO2 shared memory");
        return -1;
    }
    runtime->set_gm_sm_ptr(sm_ptr);

    void *runtime_arena_dev = runtime->host_api.acquire_pooled_runtime_arena();
    if (runtime_arena_dev == nullptr) {
        LOG_ERROR("Failed to acquire pooled runtime arena");
        return -1;
    }

    // Set up device orchestration state
    runtime->set_orch_args(device_args);

    // -------------------------------------------------------------------------
    // Build the prebuilt runtime-arena image on host.
    //
    // We pre-compute every byte the AICPU's runtime arena would otherwise have
    // to write at boot: layout offsets, sub-structure init data, and pointers
    // back to the SM / GM heap. Then we rtMemcpy the image into the pooled
    // runtime-arena region that DeviceRunner keeps alive across runs. AICPU
    // boot becomes attach + wire (cheap pointer fixup) + sm_handle->init (SM
    // reset) + a handful of device-only field fixups.
    // -------------------------------------------------------------------------
    PTO2Runtime *rt =
        runtime_init_data_from_layout(host_arena, layout, PTO2_MODE_EXECUTE, sm_ptr, sm_size, gm_heap, eff_heap_size);
    if (rt == nullptr) {
        LOG_ERROR("runtime_init_data_from_layout failed");
        return -1;
    }
    runtime_wire_arena_pointers(host_arena, layout, rt);

    // Stash the layout inside the PTO2Runtime image so the AICPU can recover
    // every arena-internal offset after rtMemcpy. The runtime arena's device
    // base does NOT travel in this image — it's on the host Runtime
    // (set_prebuilt_arena below), since the AICPU needs that pointer
    // *before* it can dereference the image.
    rt->prebuilt_layout = layout;

    int rc_upload = runtime->host_api.copy_to_device(runtime_arena_dev, host_arena.base(), layout.arena_size);
    if (rc_upload != 0) {
        LOG_ERROR("Failed to rtMemcpy prebuilt runtime arena to device (rc=%d)", rc_upload);
        return -1;
    }
    runtime->set_prebuilt_arena(runtime_arena_dev, layout.off_runtime);
    int64_t t_prebuilt_end = _now_ms();

    LOG_INFO_V0("Device orchestration ready: %d tensors + %d scalars", tensor_count, scalar_count);

    int64_t t_total_end = _now_ms();
    LOG_INFO_V0("TIMING: args_malloc_copy = %" PRId64 "ms", t_args_end - t_args_start);
    LOG_INFO_V0("TIMING: static_arena_setup = %" PRId64 "ms", t_setup_end - t_setup_start);
    LOG_INFO_V0("TIMING: gm_heap_acquire = %" PRId64 "ms", t_heap_end - t_heap_start);
    LOG_INFO_V0("TIMING: shared_mem_acquire = %" PRId64 "ms", t_sm_end - t_sm_start);
    LOG_INFO_V0("TIMING: prebuilt_runtime_arena = %" PRId64 "ms", t_prebuilt_end - t_prebuilt_start);
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
    TensorPair *tensor_pairs = runtime->tensor_pairs_.data();
    int tensor_pair_count = static_cast<int>(runtime->tensor_pairs_.size());

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

            // Read-only INPUT tensors were uploaded H2D but the kernel never
            // wrote them — copying them back (potentially ~GB) is pure waste.
            // They are still device_free'd in the cleanup loop below.
            if (!pair.needs_copy_back) {
                LOG_INFO_V0("Tensor %d: read-only input, skipping copy-back", i);
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

    // Clear the per-run dispatch-table entries staged by prepare_callable_impl.
    // The underlying chip-callable device buffer is pool-managed by
    // DeviceRunner (keyed by content hash) and bulk-freed in
    // DeviceRunner::finalize(); re-running the same callable repeatedly
    // should not re-upload.
    int kernel_count = runtime->get_registered_kernel_count();
    for (int i = 0; i < kernel_count; i++) {
        int func_id = runtime->get_registered_kernel_func_id(i);
        runtime->set_function_bin_addr(func_id, 0);
    }
    if (kernel_count > 0) {
        LOG_INFO_V0("Cleared %d kernel dispatch-table entries", kernel_count);
    }
    runtime->clear_registered_kernels();

    // Clear tensor pairs
    runtime->tensor_pairs_.clear();

    LOG_INFO_V0("=== Finalize Complete ===");

    if (rc == 0 && runtime_status != 0) {
        rc = runtime_status;
    }

    return rc;
}
