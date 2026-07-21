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
#include <vector>

#include "../common/pto_runtime_status.h"
#include "../runtime/pto_runtime2.h"
#include "../runtime/pto_shared_memory.h"
#include "../runtime/runtime.h"
#include "../../../../common/runtime_status/error_log.h"
#include "../../../../common/task_interface/call_config.h"
#include "callable.h"
#include "common/platform_config.h"
#include "common/strace.h"
#include "common/unified_log.h"
#include "host/platform_compile_info.h"
#include "host/raii_scope_guard.h"
#include "common/host_api.h"
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

static int32_t pto2_read_runtime_status(Runtime *runtime, const HostApi *api, PTO2SharedMemoryHeader *host_header) {
    if (runtime == nullptr || host_header == nullptr) {
        return 0;
    }

    void *pto2_sm = runtime->get_gm_sm_ptr();
    if (pto2_sm == nullptr) {
        return 0;
    }

    int hdr_rc = api->copy_from_device(host_header, pto2_sm, sizeof(PTO2SharedMemoryHeader));
    if (hdr_rc != 0) {
        LOG_WARN("Failed to copy PTO2 header from device");
        return 0;
    }

    int32_t orch_error_code = host_header->orch_error_code.load(std::memory_order_relaxed);
    int32_t sched_error_code = host_header->sched_error_code.load(std::memory_order_relaxed);
    return runtime_status_from_error_codes(orch_error_code, sched_error_code);
}

static void release_tensor_leases(Runtime *runtime, const HostApi *api) {
    int freed = 0;
    int buffer_noop = 0;
    int external_noop = 0;
    for (TensorLease &lease : runtime->tensor_leases_) {
        if (lease.dev_ptr == nullptr) {
            continue;
        }
        switch (lease.release_kind) {
        case TensorReleaseKind::Free:
            api->device_free(lease.dev_ptr);
            ++freed;
            break;
        case TensorReleaseKind::BufferNoop:
            ++buffer_noop;
            break;
        case TensorReleaseKind::ExternalNoop:
            ++external_noop;
            break;
        }
    }
    LOG_DEBUG("Released tensor leases: freed=%d buffer_noop=%d external_noop=%d", freed, buffer_noop, external_noop);
    runtime->tensor_leases_.clear();
}

// per-run bump allocator over the runner's retained temporary buffer. This is
// the whole temporary-buffer mechanism: the platform only remembers a
// {addr, size} slot across runs (HostApi get/set_retained_temp_buffer); the
// grow/pack/slice logic lives here. TRB kernels require 1024-byte-aligned
// device pointers, which device_malloc already guarantees for the OFF path, so
// the retained base is 1024-aligned and slices taken at 1024-aligned offsets
// stay aligned without any base fix-up.
class RetainedTempBump {
public:
    static constexpr size_t kAlignment = 1024;

    static size_t align_up(size_t v) { return (v + (kAlignment - 1)) & ~(kAlignment - 1); }

    // Pack the run's non-child, non-empty tensors to compute the required
    // aligned size, then grow the retained slot if it is too small (free old +
    // malloc new + write back). Returns false only if the (grow) device_malloc
    // fails. A run needing 0 bytes leaves the slot untouched.
    bool begin(const HostApi *api, const ChipStorageTaskArgs *orch_args) {
        api_ = api;
        offset_ = 0;
        size_t required = 0;
        for (int i = 0; i < orch_args->tensor_count(); i++) {
            Tensor t = orch_args->tensor(i);
            if (t.is_child_memory() || t.nbytes() == 0) {
                continue;
            }
            required += align_up(static_cast<size_t>(t.nbytes()));
        }
        void *addr = nullptr;
        size_t size = 0;
        api->get_retained_temp_buffer(&addr, &size);
        if (required > size) {
            if (addr != nullptr) {
                api->device_free(addr);
            }
            addr = required != 0 ? api->device_malloc(required) : nullptr;
            if (required != 0 && addr == nullptr) {
                api->set_retained_temp_buffer(nullptr, 0);
                base_ = nullptr;
                capacity_ = 0;
                LOG_ERROR("Retained temp buffer grow failed: required bytes %zu", required);
                return false;
            }
            api->set_retained_temp_buffer(addr, required);
            size = required;
        }
        base_ = addr;
        capacity_ = size;
        return true;
    }

    // Slice `bytes` from the retained buffer at the next 1024-aligned offset.
    // Must fit because begin() sized the buffer from the same tensors; a miss
    // is a caller bug (plan/slice mismatch), reported as nullptr.
    void *acquire(size_t bytes) {
        size_t aligned = align_up(offset_);
        if (base_ == nullptr || aligned + bytes > capacity_) {
            LOG_ERROR("Retained temp buffer slice miss: bytes=%zu offset=%zu capacity=%zu", bytes, aligned, capacity_);
            return nullptr;
        }
        void *ptr = static_cast<char *>(base_) + aligned;
        offset_ = aligned + bytes;
        return ptr;
    }

private:
    const HostApi *api_ = nullptr;
    void *base_ = nullptr;
    size_t capacity_ = 0;
    size_t offset_ = 0;
};

/**
 * Stage the per-callable resources (kernel binaries + orchestration SO) into
 * the supplied runtime so a subsequent bind_callable_to_runtime_impl can use
 * them. This is the cacheable half of init_runtime_impl: nothing here depends
 * on per-run argument values, so the simpler_register_callable / simpler_run split
 * lets us run this once per callable_id and amortize across runs.
 *
 * @param runtime   Pointer to pre-constructed Runtime
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
    if (upload_and_collect_child_addrs(
            callable, upload_fn, &out->kernel_addrs, &out->chip_buffer_dev, &out->chip_buffer_hash
        ) != 0) {
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

// Effective ring sizing for one (callable_id, config): the input half of the
// arena description. Resolved once per config from per-task overrides + env +
// compile-time defaults; depends on nothing that varies per run.
struct ArenaSizingConfig {
    uint64_t task_window_sizes[PTO2_MAX_RING_DEPTH];
    uint64_t heap_sizes[PTO2_MAX_RING_DEPTH];
    int32_t dep_pool_capacities[PTO2_MAX_RING_DEPTH];
};

struct ArenaStaticSizes {
    uint64_t total_heap;
    uint64_t sm_size;
};

// Device pointers to the per-Worker static pools that DeviceRunner keeps alive
// across runs (freed in DeviceRunner::finalize(), never in tensor_leases_).
struct StaticArenaPtrs {
    void *gm_heap;
    void *gm_sm;
    void *runtime_arena_dev;
};

struct PrebuiltRuntimeArenaCacheProbe {
    uint64_t hash{0};
    std::vector<uint8_t> serialized_key{};
};

static void hash_mix_u64(uint64_t *hash, uint64_t value) {
    constexpr uint64_t kFnvPrime = 1099511628211ULL;
    for (int i = 0; i < 8; i++) {
        *hash ^= (value >> (i * 8)) & 0xff;
        *hash *= kFnvPrime;
    }
}

static void append_cache_key_u64(std::vector<uint8_t> *out, uint64_t value) {
    for (int i = 0; i < 8; i++) {
        out->push_back(static_cast<uint8_t>((value >> (i * 8)) & 0xff));
    }
}

static PrebuiltRuntimeArenaCacheProbe make_prebuilt_runtime_arena_cache_probe(const ArenaSizingConfig &sizing) {
    PrebuiltRuntimeArenaCacheProbe probe;
    uint64_t hash = 1469598103934665603ULL;
    probe.serialized_key.reserve(PTO2_MAX_RING_DEPTH * 3 * sizeof(uint64_t));
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        hash_mix_u64(&hash, sizing.task_window_sizes[r]);
        append_cache_key_u64(&probe.serialized_key, sizing.task_window_sizes[r]);
        hash_mix_u64(&hash, sizing.heap_sizes[r]);
        append_cache_key_u64(&probe.serialized_key, sizing.heap_sizes[r]);
        hash_mix_u64(&hash, static_cast<uint32_t>(sizing.dep_pool_capacities[r]));
        append_cache_key_u64(&probe.serialized_key, static_cast<uint32_t>(sizing.dep_pool_capacities[r]));
    }
    probe.hash = hash;
    return probe;
}

// per-(cid,config): resolve the cache-key sizing knobs. Pure host parsing over
// per-task overrides, PTO2_RING_* env, and compile-time defaults. Derived
// allocation sizes are computed only on cache miss.
static bool resolve_arena_sizing(
    const uint64_t *ring_task_window, const uint64_t *ring_heap, const uint64_t *ring_dep_pool, ArenaSizingConfig *out
) {
    if (!resolve_ring_config(
            ring_task_window, ring_heap, ring_dep_pool, out->task_window_sizes, out->heap_sizes,
            out->dep_pool_capacities
        )) {
        return false;
    }
    const std::string task_window_log = format_ring_array(out->task_window_sizes);
    const std::string heap_log = format_ring_array(out->heap_sizes);
    const std::string dep_pool_log = format_ring_array(out->dep_pool_capacities);
    LOG_INFO_V0(
        "Ring buffer sizes: task_window=%s heap=%s dep_pool=%s", task_window_log.c_str(), heap_log.c_str(),
        dep_pool_log.c_str()
    );

    return true;
}

static bool derive_arena_static_sizes(const ArenaSizingConfig &sizing, ArenaStaticSizes *out) {
    out->total_heap = 0;
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        if (sizing.heap_sizes[r] > std::numeric_limits<uint64_t>::max() - out->total_heap) {
            LOG_ERROR("Total ring heap size overflows uint64_t");
            return false;
        }
        out->total_heap += sizing.heap_sizes[r];
    }
    out->sm_size = PTO2SharedMemoryHandle::calculate_size_per_ring(sizing.task_window_sizes);
    return true;
}

// per-run: the only signature-aware step. Copy the orch args, replacing each
// host tensor pointer with a freshly staged device pointer (H2D copy-in, or an
// on-device zero for pure-OUTPUT buffers), and record the host/device pair for
// copy-back. Read-only INPUT tensors skip copy-back. When `bump` is non-null,
// ordinary non-child tensors are sliced from the runner's retained temporary
// buffer (released as a no-op — the buffer is reused across runs); otherwise
// each is device_malloc'd and freed in validate. On failure the partially
// staged device_args / tensor_leases_ stay owned by the caller's Runtime.
static bool stage_device_args(
    Runtime *runtime, const HostApi *api, const ChipStorageTaskArgs *orch_args, const ArgDirection *signature,
    int sig_count, RetainedTempBump *bump, ChipStorageTaskArgs *out
) {
    int tensor_count = orch_args->tensor_count();
    int scalar_count = orch_args->scalar_count();

    int64_t t_args_start = _now_ms();
    STRACE_A("simpler_run.bind.args", "");
    for (int i = 0; i < tensor_count; i++) {
        Tensor t = orch_args->tensor(i);

        if (t.is_child_memory()) {
            LOG_INFO_V0("  Tensor %d: child memory, pass-through (0x%" PRIx64 ")", i, t.buffer.addr);
            out->add_tensor(t);
            continue;
        }

        void *host_ptr = reinterpret_cast<void *>(static_cast<uintptr_t>(t.buffer.addr));
        size_t size = static_cast<size_t>(t.nbytes());
        if (size == 0) {
            t.buffer.addr = 0;
            out->add_tensor(t);
            continue;
        }

        void *dev_ptr = nullptr;
        TensorReleaseKind release_kind = TensorReleaseKind::Free;
        if (bump != nullptr) {
            dev_ptr = bump->acquire(size);
            release_kind = TensorReleaseKind::BufferNoop;
            if (dev_ptr == nullptr) {
                LOG_ERROR("Retained temp buffer slice failed for tensor %d: tensor bytes=%zu", i, size);
                return false;
            }
        } else {
            dev_ptr = api->device_malloc(size);
        }
        if (dev_ptr == nullptr) {
            LOG_ERROR("Failed to allocate device memory for tensor %d", i);
            return false;
        }

        // Pure write-only OUTPUT buffers are never read by the kernel and hold
        // no meaningful host content, so they need no device staging — the
        // kernel defines what it writes and any unwritten bytes are undefined.
        // IN / INOUT (read-before-write) are staged H2D.
        bool is_pure_output = (signature != nullptr && i < sig_count && signature[i] == ArgDirection::OUT);
        if (!is_pure_output) {
            int rc = api->copy_to_device(dev_ptr, host_ptr, size);
            if (rc != 0) {
                LOG_ERROR("Failed to stage tensor %d to device", i);
                if (release_kind == TensorReleaseKind::Free) {
                    api->device_free(dev_ptr);
                }
                return false;
            }
        }
        // Read-only INPUT tensors are never written by the kernel, so there is
        // no point copying them back D2H at the end. Index the signature
        // by the orch tensor index `i` (child_memory tensors are skipped above
        // but do not consume a separate signature slot — scalars follow the
        // tensor entries). Anything not provably IN keeps the safe default of
        // copying back.
        bool needs_copy_back = !(signature != nullptr && i < sig_count && signature[i] == ArgDirection::IN);
        runtime->tensor_leases_.push_back({host_ptr, dev_ptr, size, needs_copy_back, release_kind});
        LOG_INFO_V0("  Tensor %d: %zu bytes at %p", i, size, dev_ptr);

        t.buffer.addr = reinterpret_cast<uint64_t>(dev_ptr);
        out->add_tensor(t);
    }
    for (int i = 0; i < scalar_count; i++) {
        out->add_scalar(orch_args->scalar(i));
    }
    int64_t t_args_end = _now_ms();
    LOG_INFO_V0("TIMING: args_malloc_copy = %" PRId64 "ms", t_args_end - t_args_start);
    return true;
}

// per-run: latch the env-driven orchestrator/scheduler hand-off flags onto the
// runtime. Behavior-only env reads (no new gates); kept here so the args and
// image steps stay free of unrelated state.
static void apply_orch_sched_env_flags(Runtime *runtime) {
    const char *serial_env = std::getenv("SIMPLER_TMR_SERIAL_ORCH_SCHED_ENABLE");
    runtime->dev.serial_orch_sched =
        serial_env && (serial_env[0] == '1' || serial_env[0] == 't' || serial_env[0] == 'T');
    LOG_INFO_V0(
        "Serial orchestrator-to-scheduler start gate: %s", runtime->dev.serial_orch_sched ? "enabled" : "disabled"
    );
}

// per-(cid,config): reserve and acquire the static device pools. GM heap, PTO2
// shared memory, and the prebuilt runtime arena all live in one backing
// allocation; setup_static_arena reserves the three regions and commits in one
// shot. The runtime-arena size is recovered by replaying the (pure, cheap)
// reserve sequence on a throwaway host arena. Idempotent across runs — the
// pools are owned by DeviceRunner and freed in DeviceRunner::finalize().
static bool ensure_static_arenas(
    Runtime *runtime, const HostApi *api, const ArenaSizingConfig &sizing, const ArenaStaticSizes &sizes,
    StaticArenaPtrs *out
) {
    DeviceArena sizing_arena;  // discarded; only its computed arena_size is read
    PTO2RuntimeArenaLayout layout =
        runtime_reserve_layout(sizing_arena, sizing.task_window_sizes, sizing.heap_sizes, sizing.dep_pool_capacities);

    int64_t t_setup_start = _now_ms();
    if (api->setup_static_arena(sizes.total_heap, sizes.sm_size, layout.offsets.arena_size) != 0) {
        LOG_ERROR("Failed to setup pooled static arena");
        return false;
    }
    int64_t t_setup_end = _now_ms();

    int64_t t_heap_start = _now_ms();
    out->gm_heap = api->acquire_pooled_gm_heap();
    int64_t t_heap_end = _now_ms();
    if (out->gm_heap == nullptr) {
        LOG_ERROR("Failed to acquire pooled GM heap");
        return false;
    }

    int64_t t_sm_start = _now_ms();
    out->gm_sm = api->acquire_pooled_gm_sm();
    int64_t t_sm_end = _now_ms();
    if (out->gm_sm == nullptr) {
        LOG_ERROR("Failed to acquire pooled PTO2 shared memory");
        return false;
    }
    runtime->set_gm_sm_ptr(out->gm_sm);

    out->runtime_arena_dev = api->acquire_pooled_runtime_arena();
    if (out->runtime_arena_dev == nullptr) {
        LOG_ERROR("Failed to acquire pooled runtime arena");
        return false;
    }

    LOG_INFO_V0("TIMING: static_arena_setup = %" PRId64 "ms", t_setup_end - t_setup_start);
    LOG_INFO_V0("TIMING: gm_heap_acquire = %" PRId64 "ms", t_heap_end - t_heap_start);
    LOG_INFO_V0("TIMING: shared_mem_acquire = %" PRId64 "ms", t_sm_end - t_sm_start);
    return true;
}

// per-(cid,config): build the prebuilt runtime-arena image on host. Pure host
// work — touches no device memory, only `host_arena` (owned by the caller so
// the image outlives this call until the upload) and the device *addresses* in
// `ptrs` (stored, never dereferenced).
//
// We pre-compute every byte the AICPU's runtime arena would otherwise have to
// write at boot: layout offsets, sub-structure init data, and pointers back to
// the SM / GM heap. AICPU boot then becomes attach + wire (cheap pointer fixup)
// + sm_handle->init (SM reset) + a handful of device-only field fixups.
//
// The layout is stashed inside the image (rt->prebuilt_layout) so the AICPU can
// recover every arena-internal offset after the rtMemcpy. Returns the layout
// via `out_layout`; the runtime-arena device base travels separately on the
// host Runtime (bind_launch_state), since the AICPU needs that pointer *before*
// it can dereference the image.
static bool build_runtime_image(
    const ArenaSizingConfig &sizing, const ArenaStaticSizes &sizes, const StaticArenaPtrs &ptrs,
    DeviceArena *host_arena, PTO2RuntimeArenaLayout *out_layout
) {
    PTO2RuntimeArenaLayout layout =
        runtime_reserve_layout(*host_arena, sizing.task_window_sizes, sizing.heap_sizes, sizing.dep_pool_capacities);
    if (host_arena->commit(DeviceArena::kDefaultBaseAlign) == nullptr) {
        LOG_ERROR("Failed to commit host arena for prebuilt runtime image");
        return false;
    }

    PTO2Runtime *rt = runtime_init_data_from_layout(
        *host_arena, layout, PTO2_MODE_EXECUTE, ptrs.gm_sm, sizes.sm_size, ptrs.gm_heap, sizing.heap_sizes
    );
    if (rt == nullptr) {
        LOG_ERROR("runtime_init_data_from_layout failed");
        return false;
    }
    runtime_wire_arena_pointers(*host_arena, layout, rt);
    rt->prebuilt_layout = layout;

    *out_layout = layout;
    return true;
}

// per-run: publish the launch state. Copy the staged args onto the runtime,
// rtMemcpy the host image into the pooled runtime-arena region, and record the
// device base + runtime offset the AICPU reads before dereferencing the image.
static bool bind_launch_state(
    Runtime *runtime, const HostApi *api, const StaticArenaPtrs &ptrs, const DeviceArena &host_arena,
    const PTO2RuntimeArenaLayout &layout, const ChipStorageTaskArgs &device_args
) {
    runtime->set_orch_args(device_args);

    int rc_upload = api->copy_to_device(ptrs.runtime_arena_dev, host_arena.base(), layout.offsets.arena_size);
    if (rc_upload != 0) {
        LOG_ERROR("Failed to rtMemcpy prebuilt runtime arena to device (rc=%d)", rc_upload);
        return false;
    }
    runtime->set_prebuilt_arena(ptrs.runtime_arena_dev, layout.offsets.off_runtime);
    return true;
}

static int bind_cached_runtime_image(
    Runtime *runtime, const HostApi *api, const PrebuiltRuntimeArenaCacheProbe &probe,
    const ChipStorageTaskArgs &device_args
) {
    if (api->lookup_prebuilt_runtime_arena_cache == nullptr) {
        return 1;
    }

    void *gm_heap = nullptr;
    void *sm_ptr = nullptr;
    void *runtime_arena_dev = nullptr;
    size_t runtime_off = 0;
    const void *cached_image = nullptr;
    size_t cached_image_size = 0;
    bool cache_hit = api->lookup_prebuilt_runtime_arena_cache(
        probe.hash, probe.serialized_key.data(), probe.serialized_key.size(), &gm_heap, &sm_ptr, &runtime_arena_dev,
        &runtime_off, &cached_image, &cached_image_size
    );
    if (!cache_hit) {
        return 1;
    }

    runtime->set_orch_args(device_args);
    (void)cached_image;
    (void)cached_image_size;
    runtime->set_gm_sm_ptr(sm_ptr);
    runtime->set_prebuilt_arena(runtime_arena_dev, runtime_off);
    return 0;
}

static void store_prebuilt_runtime_image(
    Runtime *runtime, const HostApi *api, const PrebuiltRuntimeArenaCacheProbe &probe, const StaticArenaPtrs &ptrs,
    const PTO2RuntimeArenaLayout &layout, const DeviceArena &host_arena
) {
    if (api->mark_prebuilt_runtime_arena_cached == nullptr) {
        return;
    }
    api->mark_prebuilt_runtime_arena_cached(
        probe.hash, probe.serialized_key.data(), probe.serialized_key.size(), ptrs.gm_heap, ptrs.gm_sm,
        ptrs.runtime_arena_dev, layout.offsets.off_runtime, host_arena.base(), layout.offsets.arena_size
    );
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
 * Orchestrates the three lifecycles behind the bind: per-config arena sizing
 * (resolve_arena_sizing) + static pools (ensure_static_arenas) + host image
 * (build_runtime_image), and per-run args (stage_device_args) + launch publish
 * (bind_launch_state).
 *
 * @param runtime    Pointer to pre-constructed Runtime
 * @param orch_args  Separated tensor/scalar arguments for this run
 * @return 0 on success, -1 on failure
 */
extern "C" int bind_callable_to_runtime_impl(
    Runtime *runtime, const HostApi *api, const ChipStorageTaskArgs *orch_args, void *host_orch_func_ptr,
    const ArgDirection *signature, int sig_count, const uint64_t *ring_task_window, const uint64_t *ring_heap,
    const uint64_t *ring_dep_pool
) {
    if (runtime == nullptr) {
        LOG_ERROR("Runtime pointer is null");
        return -1;
    }
    if (api == nullptr) {
        LOG_ERROR("HostApi pointer is null");
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
    runtime->tensor_leases_.clear();

    int64_t t_total_start = _now_ms();

    ArenaSizingConfig sizing;
    if (!resolve_arena_sizing(ring_task_window, ring_heap, ring_dep_pool, &sizing)) {
        return -1;
    }

    // The retained temporary buffer is always used on the trb path — it is an
    // internal allocation optimization, not user-facing config. Gate only on
    // whether the platform wired the slot accessors (trb always does; a backend
    // that leaves them null transparently falls back to per-tensor
    // device_malloc). The buffer itself lives on the runner across runs; here we
    // just grow it to this run's packed size and bump-slice from it.
    RetainedTempBump bump;
    bool use_temporary_buffer = api->get_retained_temp_buffer != nullptr && api->set_retained_temp_buffer != nullptr;
    if (use_temporary_buffer && !bump.begin(api, orch_args)) {
        return -1;
    }

    auto bind_cleanup = RAIIScopeGuard([&]() {
        release_tensor_leases(runtime, api);
    });

    ChipStorageTaskArgs device_args;
    if (!stage_device_args(
            runtime, api, orch_args, signature, sig_count, use_temporary_buffer ? &bump : nullptr, &device_args
        )) {
        return -1;
    }

    apply_orch_sched_env_flags(runtime);

    int64_t t_prebuilt_start = _now_ms();
    {
        STRACE("simpler_run.bind.prebuilt");
        PrebuiltRuntimeArenaCacheProbe cache_probe = make_prebuilt_runtime_arena_cache_probe(sizing);
        int cache_rc = bind_cached_runtime_image(runtime, api, cache_probe, device_args);
        if (cache_rc < 0) {
            return -1;
        }
        if (cache_rc != 0) {
            ArenaStaticSizes sizes;
            if (!derive_arena_static_sizes(sizing, &sizes)) {
                return -1;
            }

            StaticArenaPtrs ptrs;
            if (!ensure_static_arenas(runtime, api, sizing, sizes, &ptrs)) {
                return -1;
            }

            DeviceArena host_arena;  // libc malloc backend; owns the image until upload
            PTO2RuntimeArenaLayout layout;
            if (!build_runtime_image(sizing, sizes, ptrs, &host_arena, &layout)) {
                return -1;
            }

            if (!bind_launch_state(runtime, api, ptrs, host_arena, layout, device_args)) {
                return -1;
            }
            store_prebuilt_runtime_image(runtime, api, cache_probe, ptrs, layout, host_arena);
        }
    }
    int64_t t_prebuilt_end = _now_ms();

    LOG_INFO_V0("Device orchestration ready: %d tensors + %d scalars", tensor_count, scalar_count);

    int64_t t_total_end = _now_ms();
    LOG_INFO_V0("TIMING: prebuilt_runtime_arena = %" PRId64 "ms", t_prebuilt_end - t_prebuilt_start);
    LOG_INFO_V0("TIMING: total_init_runtime_impl = %" PRId64 "ms", t_total_end - t_total_start);

    bind_cleanup.dismiss();
    return 0;
}

/**
 * Validate runtime results and cleanup.
 *
 * This function:
 * 1. Copies recorded tensors from device back to host
 * 2. Releases recorded tensor leases
 * 3. Clears tensor lease state
 *
 * @param runtime  Pointer to Runtime
 * @return 0 on success, -1 on failure
 */
extern "C" int validate_runtime_impl(Runtime *runtime, const HostApi *api) {
    if (runtime == nullptr) {
        LOG_ERROR("Runtime pointer is null");
        return -1;
    }
    if (api == nullptr) {
        LOG_ERROR("HostApi pointer is null");
        return -1;
    }

    int rc = 0;

    LOG_INFO_V0("=== Copying Results Back to Host ===");

    // Copy all recorded tensors from device back to host
    TensorLease *tensor_leases = runtime->tensor_leases_.data();
    int tensor_lease_count = static_cast<int>(runtime->tensor_leases_.size());

    LOG_INFO_V0("Tensor leases to process: %d", tensor_lease_count);

    // PTO2 (device orchestration): graph output may be in packed buffer
    uint64_t graph_out_ptr = 0;
    uint64_t graph_out_size = 0;
    bool skip_tensor_copy_back = false;
    int32_t runtime_status = 0;
    PTO2SharedMemoryHeader host_header;
    memset(&host_header, 0, sizeof(host_header));

    runtime_status = pto2_read_runtime_status(runtime, api, &host_header);
    if (runtime_status != 0) {
        int32_t orch_error_code = host_header.orch_error_code.load(std::memory_order_relaxed);
        int32_t sched_error_code = host_header.sched_error_code.load(std::memory_order_relaxed);
        LOG_RUNTIME_FAILURE(orch_error_code, sched_error_code, runtime_status);
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
        for (int i = 0; i < tensor_lease_count; i++) {
            const TensorLease &lease = tensor_leases[i];

            // Skip if device pointer is null
            if (lease.dev_ptr == nullptr) {
                LOG_WARN("Tensor %d has null device pointer, skipping", i);
                continue;
            }

            // If host pointer is null, this is a device-only allocation (no copy-back)
            if (lease.host_ptr == nullptr) {
                LOG_INFO_V0("Tensor %d: device-only allocation (no copy-back)", i);
                continue;
            }

            // Read-only INPUT tensors were uploaded H2D but the kernel never
            // wrote them — copying them back (potentially ~GB) is pure waste.
            // They are still released through release_kind below.
            if (!lease.needs_copy_back) {
                LOG_INFO_V0("Tensor %d: read-only input, skipping copy-back", i);
                continue;
            }

            void *src_ptr = lease.dev_ptr;
            size_t copy_size = lease.size;

            // Use graph_output_ptr for the first output tensor if available
            if (first_output_tensor && graph_out_ptr != 0 && graph_out_size > 0) {
                src_ptr = reinterpret_cast<void *>(static_cast<uintptr_t>(graph_out_ptr));
                copy_size = static_cast<size_t>(graph_out_size);
                LOG_INFO_V0("Using packed output buffer for tensor %d", i);
                first_output_tensor = false;
            }

            int copy_rc = api->copy_from_device(lease.host_ptr, src_ptr, copy_size);
            if (copy_rc != 0) {
                LOG_ERROR("Failed to copy tensor %d from device: %d", i, copy_rc);
                rc = copy_rc;
            } else {
                LOG_INFO_V0("Tensor %d: %zu bytes copied to host", i, lease.size);
            }
        }
    }

    // Cleanup device tensors
    LOG_INFO_V0("=== Cleaning Up ===");
    release_tensor_leases(runtime, api);

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
