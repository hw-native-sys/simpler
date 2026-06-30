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
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>

#include "../common/pto_runtime_status.h"
#include "../runtime/pto_runtime2.h"
#include "../runtime/pto_shared_memory.h"
#include "../runtime/runtime.h"
#include "../../../../common/task_interface/call_config.h"
#include "callable.h"
#include "common/platform_config.h"
#include "common/strace.h"
#include "common/unified_log.h"
#include "host/platform_compile_info.h"
#include "host/runtime_timeout_config.h"
#include "utils/device_arena.h"
#include "prepare_callable_common.h"

static_assert(
    RUNTIME_ENV_RING_COUNT == PTO2_MAX_RING_DEPTH, "RuntimeEnv ring count must match PTO2 runtime ring depth"
);

// Helper: return current time in milliseconds
static int64_t _now_ms() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return static_cast<int64_t>(tv.tv_sec) * 1000 + tv.tv_usec / 1000;
}

static bool is_power_of_2_u64(uint64_t value) { return value != 0 && (value & (value - 1)) == 0; }

template <typename T>
static std::string format_ring_array(const T (&values)[PTO2_MAX_RING_DEPTH]) {
    std::string out = "[";
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; ++r) {
        if (r != 0) {
            out += ", ";
        }
        out += std::to_string(values[r]);
    }
    out += "]";
    return out;
}

static std::string trim_copy(const std::string &input) {
    size_t begin = 0;
    while (begin < input.size() && std::isspace(static_cast<unsigned char>(input[begin]))) {
        ++begin;
    }
    size_t end = input.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(input[end - 1]))) {
        --end;
    }
    return input.substr(begin, end - begin);
}

static bool parse_uint_token(
    const char *name, const std::string &raw, uint64_t min_val, uint64_t max_val, bool require_power_of_2, uint64_t *out
) {
    std::string token = trim_copy(raw);
    if (token.empty()) {
        LOG_WARN("%s has an empty value in '%s', ignored", name, raw.c_str());
        return false;
    }

    if (token[0] == '-') {
        LOG_WARN("%s=%s invalid (must be a non-negative integer), ignored", name, token.c_str());
        return false;
    }
    char *endptr = nullptr;
    errno = 0;
    unsigned long long parsed = std::strtoull(token.c_str(), &endptr, 10);
    if (errno == ERANGE || endptr == token.c_str() || *endptr != '\0') {
        LOG_WARN("%s=%s invalid (must be a non-negative integer), ignored", name, token.c_str());
        return false;
    }
    uint64_t val = static_cast<uint64_t>(parsed);

    if (val < min_val || val > max_val) {
        LOG_WARN(
            "%s=%s invalid (must be in [%" PRIu64 ", %" PRIu64 "]), ignored", name, token.c_str(), min_val, max_val
        );
        return false;
    }
    if (require_power_of_2 && !is_power_of_2_u64(val)) {
        LOG_WARN("%s=%s invalid (must be a power of 2), ignored", name, token.c_str());
        return false;
    }
    *out = val;
    return true;
}

static void apply_env_ring_values(
    const char *name, uint64_t min_val, uint64_t max_val, bool require_power_of_2, uint64_t out[PTO2_MAX_RING_DEPTH]
) {
    const char *env = std::getenv(name);
    if (!env) return;

    std::string text(env);
    if (text.find(',') == std::string::npos) {
        uint64_t value = 0;
        if (!parse_uint_token(name, text, min_val, max_val, require_power_of_2, &value)) {
            return;
        }
        for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
            out[r] = value;
        }
        return;
    }

    uint64_t parsed[PTO2_MAX_RING_DEPTH]{};
    size_t pos = 0;
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        size_t comma = text.find(',', pos);
        std::string token = text.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        if (!parse_uint_token(name, token, min_val, max_val, require_power_of_2, &parsed[r])) {
            return;
        }
        if (comma == std::string::npos) {
            if (r != PTO2_MAX_RING_DEPTH - 1) {
                LOG_WARN(
                    "%s=%s invalid (expected exactly %d comma-separated values), ignored", name, env,
                    PTO2_MAX_RING_DEPTH
                );
                return;
            }
            pos = text.size();
        } else {
            pos = comma + 1;
        }
    }
    if (pos < text.size() || (!text.empty() && text.back() == ',')) {
        LOG_WARN("%s=%s invalid (expected exactly %d comma-separated values), ignored", name, env, PTO2_MAX_RING_DEPTH);
        return;
    }
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        out[r] = parsed[r];
    }
}

// ring_task_window / ring_heap / ring_dep_pool point into the #pragma pack(1)
// RuntimeEnv wire struct (call_config.h), so their uint64_t entries are only
// byte-aligned — runtime_env sits at offset 28 in CallConfig (after 7 int32_t),
// i.e. 4-byte but not 8-byte aligned. Reading them as `base[idx]` is an
// unaligned 8-byte load: UB, and fatal under UBSan (-fsanitize=alignment). Copy
// the bytes out instead. A null base means "no per-task overrides" -> 0 (unset).
static uint64_t read_ring_override(const uint64_t *base, int idx) {
    if (base == nullptr) {
        return 0;
    }
    uint64_t value;
    std::memcpy(&value, base + idx, sizeof(value));
    return value;
}

// Each of ring_task_window / ring_heap / ring_dep_pool is a per-ring array of
// PTO2_MAX_RING_DEPTH entries (0 = unset). Precedence per ring: per-task entry >
// PTO2_RING_* env value > compile-time default. A "size all rings the same"
// request arrives already broadcast to every entry by the caller.
static bool resolve_ring_config(
    const uint64_t *ring_task_window, const uint64_t *ring_heap, const uint64_t *ring_dep_pool,
    uint64_t eff_task_window_sizes[PTO2_MAX_RING_DEPTH], uint64_t eff_heap_sizes[PTO2_MAX_RING_DEPTH],
    int32_t eff_dep_pool_capacities[PTO2_MAX_RING_DEPTH]
) {
    uint64_t dep_pool_values[PTO2_MAX_RING_DEPTH];
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        eff_task_window_sizes[r] = PTO2_TASK_WINDOW_SIZE;
        eff_heap_sizes[r] = PTO2_HEAP_SIZE;
        dep_pool_values[r] = PTO2_DEP_LIST_POOL_SIZE;
    }

    apply_env_ring_values("PTO2_RING_TASK_WINDOW", 4, static_cast<uint64_t>(INT32_MAX), true, eff_task_window_sizes);
    apply_env_ring_values("PTO2_RING_HEAP", 1024, std::numeric_limits<uint64_t>::max(), false, eff_heap_sizes);
    apply_env_ring_values("PTO2_RING_DEP_POOL", 4, static_cast<uint64_t>(INT32_MAX), false, dep_pool_values);

    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        const uint64_t task_window_override = read_ring_override(ring_task_window, r);
        const uint64_t heap_override = read_ring_override(ring_heap, r);
        const uint64_t dep_pool_override = read_ring_override(ring_dep_pool, r);
        if (task_window_override != 0) {
            eff_task_window_sizes[r] = task_window_override;
        }
        if (heap_override != 0) {
            eff_heap_sizes[r] = heap_override;
        }
        if (dep_pool_override != 0) {
            dep_pool_values[r] = dep_pool_override;
        }

        if (eff_task_window_sizes[r] < 4 || eff_task_window_sizes[r] > static_cast<uint64_t>(INT32_MAX) ||
            !is_power_of_2_u64(eff_task_window_sizes[r])) {
            LOG_ERROR(
                "ring_task_window[%d]=%" PRIu64 " must be a power of 2 in [4, INT32_MAX]", r, eff_task_window_sizes[r]
            );
            return false;
        }
        if (eff_heap_sizes[r] < 1024) {
            LOG_ERROR("ring_heap[%d]=%" PRIu64 " must be >= 1024", r, eff_heap_sizes[r]);
            return false;
        }
        if (dep_pool_values[r] < 4 || dep_pool_values[r] > static_cast<uint64_t>(INT32_MAX)) {
            LOG_ERROR("ring_dep_pool[%d]=%" PRIu64 " must be in [4, INT32_MAX]", r, dep_pool_values[r]);
            return false;
        }
        eff_dep_pool_capacities[r] = static_cast<int32_t>(dep_pool_values[r]);
    }

    return true;
}

static int32_t resolve_scheduler_timeout_ms() {
    RuntimeTimeoutParseStatus parse_status;
    RuntimeTimeoutConfig cfg = resolve_runtime_timeout_config(
        RuntimeTimeoutConfig{PLATFORM_OP_EXECUTE_TIMEOUT_US, PLATFORM_STREAM_SYNC_TIMEOUT_MS, 0}, &parse_status
    );
    if (!parse_status.scheduler_env_set) {
        return 0;
    }
    if (!parse_status.scheduler_valid) {
        const char *env = std::getenv(PTO2_SCHEDULER_TIMEOUT_MS_ENV);
        LOG_WARN("%s=%s invalid, using platform scheduler timeout", PTO2_SCHEDULER_TIMEOUT_MS_ENV, env);
        return 0;
    }

    RuntimeTimeoutOrderStatus status = validate_runtime_timeout_order_for_platform(cfg, get_platform());
    if (status != RuntimeTimeoutOrderStatus::OK) {
        LOG_WARN(
            "Ignoring %s=%d: %s (op_execute=%llu us, stream_sync=%d ms)", PTO2_SCHEDULER_TIMEOUT_MS_ENV,
            cfg.scheduler_timeout_ms, runtime_timeout_order_status_name(status),
            (unsigned long long)cfg.op_execute_timeout_us, cfg.stream_sync_timeout_ms
        );
        return 0;
    }
    return cfg.scheduler_timeout_ms;
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
 * on per-run argument values, so the simpler_register_callable / simpler_run split
 * lets us run this once per callable_id and amortize across runs.
 *
 * @param runtime   Pointer to pre-constructed Runtime (host_api populated)
 * @param callable  ChipCallable carrying the orch SO + child kernel binaries
 * @return 0 on success, -1 on failure
 */
extern "C" int
register_callable_impl(const ChipCallable *callable, uint64_t (*upload_fn)(const void *), CallableArtifacts *out) {
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

    LOG_INFO_V0("Registering %d kernel(s) in register_callable_impl", callable->child_count());
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
 * is already populated by register_callable_impl.
 *
 * Splitting this from register_callable_impl matches the per-callable_id
 * design: register/run invokes this every call, while the prep
 * half runs only once per callable_id.
 *
 * @param runtime    Pointer to pre-constructed Runtime (host_api populated)
 * @param orch_args  Separated tensor/scalar arguments for this run
 * @return 0 on success, -1 on failure
 */
extern "C" int bind_callable_to_runtime_impl(
    Runtime *runtime, const ChipStorageTaskArgs *orch_args, void *host_orch_func_ptr, const ArgDirection *signature,
    int sig_count, const uint64_t *ring_task_window, const uint64_t *ring_heap, const uint64_t *ring_dep_pool
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

    uint64_t eff_task_window_sizes[PTO2_MAX_RING_DEPTH];
    uint64_t eff_heap_sizes[PTO2_MAX_RING_DEPTH];
    int32_t eff_dep_pool_capacities[PTO2_MAX_RING_DEPTH];
    if (!resolve_ring_config(
            ring_task_window, ring_heap, ring_dep_pool, eff_task_window_sizes, eff_heap_sizes, eff_dep_pool_capacities
        )) {
        return -1;
    }
    const std::string task_window_log = format_ring_array(eff_task_window_sizes);
    const std::string heap_log = format_ring_array(eff_heap_sizes);
    const std::string dep_pool_log = format_ring_array(eff_dep_pool_capacities);
    LOG_INFO_V0(
        "Ring buffer sizes: task_window=%s heap=%s dep_pool=%s", task_window_log.c_str(), heap_log.c_str(),
        dep_pool_log.c_str()
    );

    // Build device args: copy from input, replace host tensor pointers with device pointers
    ChipStorageTaskArgs device_args;

    int64_t t_args_start = _now_ms();
    {
        STRACE_A("simpler_run.bind.args", "");
        for (int i = 0; i < tensor_count; i++) {
            Tensor t = orch_args->tensor(i);

            if (t.is_child_memory()) {
                LOG_INFO_V0("  Tensor %d: child memory, pass-through (0x%" PRIx64 ")", i, t.buffer.addr);
                device_args.add_tensor(t);
                continue;
            }

            void *host_ptr = reinterpret_cast<void *>(static_cast<uintptr_t>(t.buffer.addr));
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

            t.buffer.addr = reinterpret_cast<uint64_t>(dev_ptr);
            device_args.add_tensor(t);
        }
        for (int i = 0; i < scalar_count; i++) {
            device_args.add_scalar(orch_args->scalar(i));
        }
    }
    int64_t t_args_end = _now_ms();

    // Read orchestrator-to-scheduler transition flag from environment
    {
        const char *env_val = std::getenv("PTO2_ORCH_TO_SCHED");
        if (env_val && (env_val[0] == '1' || env_val[0] == 't' || env_val[0] == 'T')) {
            runtime->dev.orch_to_sched = true;
        }
        LOG_INFO_V0("Orchestrator-to-scheduler transition: %s", runtime->dev.orch_to_sched ? "enabled" : "disabled");
    }

    // Read serial orchestrator -> scheduler start gate from environment.
    {
        const char *env_val = std::getenv("PTO2_SERIAL_ORCH_SCHED");
        runtime->dev.serial_orch_sched = env_val && (env_val[0] == '1' || env_val[0] == 't' || env_val[0] == 'T');
        LOG_INFO_V0(
            "Serial orchestrator-to-scheduler start gate: %s", runtime->dev.serial_orch_sched ? "enabled" : "disabled"
        );
    }

    // Lay out the per-Worker static device arena. GM heap, PTO2 shared memory,
    // and the prebuilt runtime arena all live in a single backing allocation;
    // setup_static_arena reserves the three regions and commits in one shot.
    // Owned by DeviceRunner across runs — do NOT record in tensor_pairs_; the
    // free is deferred to DeviceRunner::finalize(). The runtime-arena size is
    // determined by replaying the reserve sequence on a host-side arena.
    uint64_t total_heap_size = 0;
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        if (eff_heap_sizes[r] > std::numeric_limits<uint64_t>::max() - total_heap_size) {
            LOG_ERROR("Total ring heap size overflows uint64_t");
            return -1;
        }
        total_heap_size += eff_heap_sizes[r];
    }
    uint64_t sm_size = PTO2SharedMemoryHandle::calculate_size_per_ring(eff_task_window_sizes);

    int64_t t_prebuilt_start = _now_ms();
    {
        STRACE("simpler_run.bind.prebuilt");
        DeviceArena host_arena;  // libc malloc backend by default
        PTO2RuntimeArenaLayout layout =
            runtime_reserve_layout(host_arena, eff_task_window_sizes, eff_heap_sizes, eff_dep_pool_capacities);
        layout.scheduler_timeout_ms = resolve_scheduler_timeout_ms();
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
        PTO2Runtime *rt = runtime_init_data_from_layout(
            host_arena, layout, PTO2_MODE_EXECUTE, sm_ptr, sm_size, gm_heap, eff_heap_sizes
        );
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
    }

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
        // A scheduler no-progress timeout (code 100) carries a device-classified
        // sub-reason + locators so the failure line is self-diagnosing without a
        // device-log dive. The full stall snapshot stays in the device log / plog.
        if (sched_error_code == PTO2_ERROR_SCHEDULER_TIMEOUT) {
            int32_t detail = host_header.sched_stall_detail.load(std::memory_order_acquire);
            LOG_ERROR(
                "PTO2 scheduler timeout sub_class=%s (detail=%d) completed=%d/%d running=%d ready=%d waiting=%d "
                "orch_done=%d stuck_task_id=%" PRId64 " stuck_core=%d",
                stall_detail_name(detail), detail, host_header.sched_stall_completed.load(std::memory_order_relaxed),
                host_header.sched_stall_total.load(std::memory_order_relaxed),
                host_header.sched_stall_cnt_running.load(std::memory_order_relaxed),
                host_header.sched_stall_cnt_ready.load(std::memory_order_relaxed),
                host_header.sched_stall_cnt_waiting.load(std::memory_order_relaxed),
                host_header.sched_stall_orch_done.load(std::memory_order_relaxed),
                host_header.sched_stall_task_id.load(std::memory_order_relaxed),
                host_header.sched_stall_core.load(std::memory_order_relaxed)
            );
        }
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

    // Clear the per-run dispatch-table entries staged by register_callable_impl.
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

// Extra AICPU entry symbols this runtime exports beyond the base
// {simpler_aicpu_exec, simpler_aicpu_init}. TMARB resolves orchestration on the
// device, so it exports simpler_aicpu_register_callable; the common AICPU loader
// queries this so it carries no runtime-specific symbol knowledge.
extern "C" const char *const *runtime_extra_aicpu_symbols(size_t *count) {
    static const char *const kExtra[] = {"simpler_aicpu_register_callable"};
    if (count != nullptr) {
        *count = sizeof(kExtra) / sizeof(kExtra[0]);
    }
    return kExtra;
}
