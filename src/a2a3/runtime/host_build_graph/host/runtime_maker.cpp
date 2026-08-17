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
 * Runtime Builder - rt2 Implementation (host_build_graph: Host Orchestration)
 *
 * Provides init_runtime_impl and validate_runtime_impl functions for rt2 runtime.
 * The HOST runs the orchestrator to completion, populates shared memory + the
 * prebuilt arena, and H2Ds the image; the device boots scheduler-only.
 *
 * init_runtime_impl:
 *   - Converts host tensor pointers to device pointers (all inputs copied H2D;
 *     only OUTPUT/INOUT tensors are copied back D2H)
 *   - dlopens the orchestration SO on the host and runs it to build the graph
 *   - Sets up runtime state for host orchestration
 *
 * validate_runtime_impl:
 *   - Copies OUTPUT/INOUT tensors back from device to host (read-only inputs
 *     are skipped)
 *   - Frees device memory
 */

#include <dlfcn.h>
#include <stddef.h>
#include <stdint.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../common/pto_runtime_status.h"
#include "../runtime/common.h"
#include "../runtime/dep_gen_host_graph.h"
#include "../runtime/graph_execution.h"
#include "../runtime/host_tensor_access.h"
#include "../runtime/graph_host_state.h"
#include "../runtime/pto_orchestrator.h"
#include "../runtime/pto_runtime2.h"
#include "../runtime/pto_shared_memory.h"
#include "../runtime/pto_types.h"
#include "../runtime/runtime.h"
#include "../../../../common/runtime_status/error_log.h"
#include "../../../../common/host_build_graph/retained_arg_staging.h"
#include "../../../../common/task_interface/call_config.h"
#include "../../../../common/worker/pto_runtime_c_api.h"
#include "callable.h"
#include "common/platform_config.h"
#include "common/strace.h"
#include "common/unified_log.h"
#include "utils/device_arena.h"
#include "prepare_callable_common.h"

extern "C" const PipelineContract *get_pipeline_contract(void) {
    // Host orchestration materializes this run's own graph into the image it
    // uploads, so every device-resident region carries per-run content.
    static const PipelineContract contract = {
        PTO_PIPELINE_CONTRACT_ABI_VERSION,
        5,
        2,
        {
            {PTO_PIPELINE_GM_HEAP, PTO_PIPELINE_HOST_PER_RUN, 0},
            {PTO_PIPELINE_GM_SM, PTO_PIPELINE_HOST_PER_RUN, 0},
            {PTO_PIPELINE_RUNTIME_IMAGE, PTO_PIPELINE_HOST_PER_RUN, 0},
            {PTO_PIPELINE_AICPU_STREAM, PTO_PIPELINE_EXEC_HANDLE, 0},
            {PTO_PIPELINE_AICORE_STREAM, PTO_PIPELINE_EXEC_HANDLE, 0},
        },
    };
    return &contract;
}

extern "C" int concurrent_native_prepare_supported_impl(void) {
    // HBG can materialize a complete graph into the lease-selected unpublished
    // arena bank. The common C API keeps collector-bearing configurations on
    // the sequential path until their state is per-epoch.
    return 1;
}

// RuntimeEnv (call_config.h) is the cross-runtime ABI for per-ring config and
// carries RUNTIME_ENV_RING_COUNT slots, shared with tensormap_and_ringbuffer.
// host_build_graph is single-ring (PTO2_MAX_RING_DEPTH == 1) and reads only the
// first slot; it must fit within the ABI's slot budget, not equal it.
static_assert(PTO2_MAX_RING_DEPTH <= RUNTIME_ENV_RING_COUNT, "PTO2 runtime ring depth must fit RuntimeEnv ring slots");

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

// Each of ring_task_window / ring_heap is a per-ring array of PTO2_MAX_RING_DEPTH
// entries (0 = unset). Precedence per ring: per-task entry > PTO2_RING_* env value
// > compile-time default. A "size all rings the same" request arrives already
// broadcast to every entry by the caller. (Polling has no dep_pool, so the former
// PTO2_RING_DEP_POOL knob is gone.)
static bool resolve_ring_config(
    const uint64_t *ring_task_window, const uint64_t *ring_heap, uint64_t eff_task_window_sizes[PTO2_MAX_RING_DEPTH],
    uint64_t eff_heap_sizes[PTO2_MAX_RING_DEPTH]
) {
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        eff_task_window_sizes[r] = PTO2_TASK_WINDOW_SIZE;
        eff_heap_sizes[r] = PTO2_HEAP_SIZE;
    }

    apply_env_ring_values("PTO2_RING_TASK_WINDOW", 4, static_cast<uint64_t>(INT32_MAX), true, eff_task_window_sizes);
    apply_env_ring_values("PTO2_RING_HEAP", 1024, std::numeric_limits<uint64_t>::max(), false, eff_heap_sizes);

    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        const uint64_t task_window_override = read_ring_override(ring_task_window, r);
        const uint64_t heap_override = read_ring_override(ring_heap, r);
        if (task_window_override != 0) {
            eff_task_window_sizes[r] = task_window_override;
        }
        if (heap_override != 0) {
            eff_heap_sizes[r] = heap_override;
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
    }

    return true;
}

static int32_t pto2_read_runtime_status(Runtime *runtime, const HostApi *api, PTO2SharedMemoryHeader *host_header) {
    if (runtime == nullptr || api == nullptr || host_header == nullptr) {
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

namespace {

// host_build_graph is host-orchestration-first: the HOST dlopens the
// orchestration .so and runs it to completion. The shared memory + arena carry
// host-DDR cross-task pointers; the host relocates them to their final device
// addresses before the H2D copy, so the device schedules a complete image.

bool write_all_bytes(int fd, const uint8_t *data, size_t size) {
    size_t total = 0;
    while (total < size) {
        ssize_t w = write(fd, data + total, size - total);
        if (w <= 0) {
            return false;
        }
        total += static_cast<size_t>(w);
    }
    return true;
}

// Materialize the orchestration .so bytes to a temp file so it can be dlopen'd
// on the host (dlopen needs a real path + the exec bit).
bool create_orch_so_tempfile(const uint8_t *data, size_t size, std::string *out_path) {
    char tmpl[] = "/tmp/orch_so_XXXXXX";
    int fd = mkstemp(tmpl);
    if (fd < 0) {
        return false;
    }
    if (fchmod(fd, 0755) != 0) {
        close(fd);
        unlink(tmpl);
        return false;
    }
    bool ok = write_all_bytes(fd, data, size);
    if (close(fd) != 0) {
        ok = false;
    }
    if (!ok) {
        unlink(tmpl);
        return false;
    }
    *out_path = tmpl;
    return true;
}

// The orchestration .so exports these (PTO2 submit_task form).
typedef void (*OrchestrationEntryFunc)(const ChipTaskArgs &);
typedef void (*OrchestrationBindFunc)(PTO2Runtime *);

// Resolved orchestration .so entry points. register_callable_impl allocates one
// of these (the entry, plus the .so's own framework_bind_runtime, which sets
// the .so-private g_current_runtime its inline rt_submit_* read) and stores its
// pointer in CallableArtifacts::host_orch_func_ptr. Owned for the callable's
// lifetime alongside host_dlopen_handle.
struct HostOrchEntryPoints {
    OrchestrationEntryFunc entry{nullptr};
    OrchestrationBindFunc bind{nullptr};
};

static bool relocate_host_orch_image(
    PTO2SharedMemoryHandle &host_sm_handle, uint64_t host_sm, uint64_t sm_size, int64_t sm_delta, uint64_t host_arena,
    uint64_t arena_size, int64_t arena_delta
) {
    static_assert(PTO2_MAX_RING_DEPTH == 1, "relocate_host_orch_image assumes a single ring");

    if (!(host_sm + sm_size <= host_arena || host_arena + arena_size <= host_sm)) {
        LOG_ERROR(
            "host-orch: SM window [%#lx,+%#lx) overlaps arena window [%#lx,+%#lx); cannot relocate", host_sm, sm_size,
            host_arena, arena_size
        );
        return false;
    }

    bool ok = true;
    auto relocate = [&](auto *&pointer) {
        using Pointer = std::remove_reference_t<decltype(pointer)>;
        const uint64_t address = reinterpret_cast<uint64_t>(pointer);
        if (address == 0) return;
        if (address >= host_sm && address < host_sm + sm_size) {
            pointer = reinterpret_cast<Pointer>(static_cast<uintptr_t>(address + sm_delta));
        } else if (address >= host_arena && address < host_arena + arena_size) {
            pointer = reinterpret_cast<Pointer>(static_cast<uintptr_t>(address + arena_delta));
        } else {
            LOG_ERROR(
                "host-orch: pointer %#lx is outside both SM and arena windows; cannot relocate for device", address
            );
            ok = false;
        }
    };

    PTO2SharedMemoryHeader *header = host_sm_handle.header;
    if (header != nullptr) {
        PTO2SharedMemoryRingHeader &ring = header->ring;
        const int32_t count = ring.fc.current_task_index.load(std::memory_order_acquire);
        for (int32_t slot = 0; slot < count; ++slot) {
            PTO2TaskSlotState *state = &ring.slot_states[slot];
            relocate(state->task);
            relocate(state->payload);
        }
    }
    return ok;
}

bool upload_graph_submissions(Runtime *runtime, const HostApi *api, GraphHostState &graph_state) {
    // Reuse per-(graph_key, occurrence) device POD buffers across runs so steady
    // rounds pay H2D only — not device_malloc/device_free for every layer.
    struct RetainedSubmission {
        void *ptr = nullptr;
        size_t bytes = 0;
    };
    static std::mutex sub_mu;
    static std::unordered_map<uint64_t, RetainedSubmission> retained_subs;

    auto acquire_submission = [&](uint64_t graph_key, uint32_t occurrence, size_t bytes) -> void * {
        const uint64_t key = (graph_key << 32) ^ static_cast<uint64_t>(occurrence);
        std::scoped_lock lock(sub_mu);
        RetainedSubmission &slot = retained_subs[key];
        if (slot.ptr != nullptr && slot.bytes == bytes) {
            return slot.ptr;
        }
        if (slot.ptr != nullptr) {
            api->device_free(slot.ptr);
            slot.ptr = nullptr;
            slot.bytes = 0;
        }
        slot.ptr = api->device_malloc(bytes);
        if (slot.ptr == nullptr) {
            return nullptr;
        }
        slot.bytes = bytes;
        return slot.ptr;
    };

    std::unordered_map<uint64_t, uint32_t> occurrences;
    const size_t count = graph_host_upload_count(graph_state);
    for (size_t index = 0; index < count; ++index) {
        std::optional<GraphHostUpload> upload = graph_host_upload(graph_state, index);
        if (!upload.has_value() || upload->outer_slot == nullptr || upload->data == nullptr ||
            upload->bytes < sizeof(GraphSubmission) || upload->outer_slot->task_kind != TaskKind::GRAPH ||
            upload->outer_slot->task == nullptr) {
            LOG_ERROR("host-orch: invalid pending Graph POD image");
            return false;
        }
        auto *submission = reinterpret_cast<GraphSubmission *>(upload->data);
        if (!graph_submission_wire_size_valid(*submission, upload->bytes)) {
            LOG_ERROR("host-orch: Graph submission size does not match its POD image");
            return false;
        }
        const GraphDefinition *definition = graph_submission_definition(*submission);
        size_t execution_bytes = 0;
        if (definition == nullptr || definition->full_key != submission->graph_key || definition->task_count == 0 ||
            definition->task_count > GRAPH_MAX_NODES ||
            !graph_execution_storage_bytes(
                static_cast<int32_t>(definition->task_count), definition->tensor_arg_count,
                definition->scalar_arg_count, definition->total_bytes, &execution_bytes
            )) {
            LOG_ERROR("host-orch: invalid Graph execution storage request");
            return false;
        }
        const uint32_t occurrence = occurrences[submission->graph_key]++;
        void *execution_storage = api->acquire_graph_execution_buffer(
            submission->graph_key, occurrence, execution_bytes, alignof(GraphNodeStorage)
        );
        if (execution_storage == nullptr) {
            LOG_ERROR(
                "host-orch: failed to retain %zu bytes for Graph execution key=%#llx occurrence=%u", execution_bytes,
                static_cast<unsigned long long>(submission->graph_key), occurrence
            );
            return false;
        }
        submission->execution_storage = reinterpret_cast<uint64_t>(execution_storage);
        submission->execution_storage_bytes = execution_bytes;
        submission->local_execution = 0;
        submission->activation_gate = 0;

        void *device_submission = acquire_submission(submission->graph_key, occurrence, upload->bytes);
        if (device_submission == nullptr) {
            LOG_ERROR("host-orch: failed to allocate %zu bytes for Graph submission", upload->bytes);
            return false;
        }
        if (api->copy_to_device(device_submission, upload->data, upload->bytes) != 0) {
            LOG_ERROR("host-orch: failed to upload Graph submission POD image");
            return false;
        }
        upload->outer_slot->graph_context = device_submission;
        // Buffer lives in retained_subs until Worker finalize frees the allocator.
        runtime->tensor_pairs_.push_back(
            {nullptr, device_submission, upload->bytes, false, TensorReleaseKind::BufferNoop}
        );
    }
    return true;
}

struct GraphHostStateBinding {
    explicit GraphHostStateBinding(PTO2OrchestratorState &orchestrator, GraphHostState *state) :
        orchestrator(orchestrator) {
        orchestrator.graph_host_state = state;
    }
    ~GraphHostStateBinding() { orchestrator.graph_host_state = nullptr; }

    PTO2OrchestratorState &orchestrator;
};

int32_t run_host_orchestration(
    Runtime *runtime, const HostApi *api, HostTensorAccessor &tensor_access, PTO2Runtime *rt, DeviceArena &host_arena,
    const PTO2RuntimeArenaLayout &layout, void *device_sm, uint64_t sm_size, void *device_arena, void *gm_heap,
    const uint64_t eff_heap_sizes[PTO2_MAX_RING_DEPTH], const uint64_t eff_task_window_sizes[PTO2_MAX_RING_DEPTH],
    void *host_orch_func_ptr, const ChipTaskArgs &orch_l2
) {
    dep_gen_host_graph_begin_capture();

    // Init-on-write: descriptors, payloads, slot_states and completion_flags are
    // each written per task at submit and read only for [0, total_tasks). Zero
    // only the fixed-size header here; the per-slot segments are initialized in
    // orch::prepare_task and shipped bounded to total_tasks below.
    const pto2_sm_layout::PTO2RingSegmentOffsets sm_segs =
        pto2_sm_layout::ring_segment_offsets(eff_task_window_sizes[0]);
    // Retain the host SM mirror across runs — steady rounds only need a header
    // memset, not a fresh multi‑MB allocation each bind.
    static std::mutex host_sm_mu;
    static std::unique_ptr<uint8_t[]> retained_host_sm;
    static size_t retained_host_sm_bytes = 0;
    void *host_sm = nullptr;
    {
        std::scoped_lock lock(host_sm_mu);
        if (retained_host_sm_bytes < sm_size) {
            retained_host_sm.reset(new uint8_t[sm_size]);
            retained_host_sm_bytes = sm_size;
        }
        host_sm = retained_host_sm.get();
    }
    std::memset(host_sm, 0, sm_segs.descriptors);

    // Re-point the orchestrator half at the host SM (scheduler keeps device SM).
    // init_data_from_layout resets the orchestrator state, so this is safe.
    if (!rt->orchestrator.init_data_from_layout(
            layout.orch, host_arena, host_sm, gm_heap, eff_heap_sizes[0], eff_task_window_sizes[0]
        )) {
        LOG_ERROR("host-orch: orchestrator re-init against host SM failed");
        return -1;
    }
    rt->orchestrator.wire_arena_pointers(layout.orch, host_arena, &rt->scheduler);

    PTO2SharedMemoryHandle host_sm_handle;
    if (!host_sm_handle.init_per_ring(host_sm, sm_size, eff_task_window_sizes, eff_heap_sizes)) {
        LOG_ERROR("host-orch: host SM init_per_ring failed");
        return -1;
    }

    GraphHostStatePtr graph_state = make_graph_host_state();
    if (!graph_state) {
        LOG_ERROR("host-orch: failed to allocate Graph host state");
        return -1;
    }
    GraphHostStateBinding graph_binding(rt->orchestrator, graph_state.get());

    const int32_t block_dim = runtime->get_worker_count() / PLATFORM_CORES_PER_BLOCKDIM;
    if (block_dim < 1) {
        LOG_ERROR("host-orch: worker_count %d yields no clusters", runtime->get_worker_count());
        return -1;
    }
    runtime_finalize_after_wire(
        rt, block_dim * PLATFORM_AIC_CORES_PER_BLOCKDIM, block_dim * PLATFORM_AIV_CORES_PER_BLOCKDIM
    );
    rt->mode = PTO2_MODE_EXECUTE;

    const auto *entry_points = reinterpret_cast<const HostOrchEntryPoints *>(host_orch_func_ptr);
    if (entry_points->bind == nullptr) {
        LOG_ERROR("host-orch: orch .so framework_bind_runtime was not resolved");
        return -1;
    }
    rt->active_callable_hash = reinterpret_cast<uint64_t>(entry_points->entry);
    rt->tensor_access = &tensor_access;
    // Binds the orchestration .so's own framework_current_runtime, which its
    // inline rt_submit_* read. The host library links a same-named copy from
    // orchestration/common.cpp, but nothing outside the .so includes
    // pto_orchestration_api.h, so nothing reads that one — rt_scope_* and
    // rt_orchestration_done take the runtime as an argument.
    entry_points->bind(rt);

    rt_scope_begin(rt);
    entry_points->entry(orch_l2);
    rt_scope_end(rt);
    rt_orchestration_done(rt);

    const int32_t total_tasks = pto2_sm_layout::ring_current_task_index_addr(host_sm)->load(std::memory_order_acquire);
    if (!upload_graph_submissions(runtime, api, *graph_state)) return -1;

    // total_tasks sizes the bounded per-segment H2D copies below; a value outside
    // [0, task_window] would make those copies read/write out of bounds.
    if (total_tasks < 0 || static_cast<uint64_t>(total_tasks) > eff_task_window_sizes[0]) {
        LOG_ERROR("host-orch: total_tasks %d out of range [0, %" PRIu64 "]", total_tasks, eff_task_window_sizes[0]);
        return -1;
    }

    // Relocate the host-DDR cross-task pointers to their final DEVICE addresses
    // on the host, before the SM and arena leave for the device. Pointers into
    // the SM shift by sm_delta; pointers into the arena (fanout adjacency, wiring
    // queue) shift by arena_delta. After this both the SM and arena carry device
    // addresses, so the device boots scheduler-only.
    const int64_t sm_delta = static_cast<int64_t>(reinterpret_cast<uint64_t>(device_sm)) -
                             static_cast<int64_t>(reinterpret_cast<uint64_t>(host_sm));
    const int64_t arena_delta = static_cast<int64_t>(reinterpret_cast<uint64_t>(device_arena)) -
                                static_cast<int64_t>(reinterpret_cast<uint64_t>(host_arena.base()));
    if (!relocate_host_orch_image(
            host_sm_handle, reinterpret_cast<uint64_t>(host_sm), sm_size, sm_delta,
            reinterpret_cast<uint64_t>(host_arena.base()), layout.arena_size, arena_delta
        )) {
        LOG_ERROR("host-orch: relocation failed; refusing to H2D an image with unrelocated host pointers");
        return -1;
    }

    // Ship only the live prefix of each segment: the device reads no slot past
    // total_tasks, so upload header + descriptors[0,N), payloads[0,N),
    // slot_states[0,N) and completion_flags[0,N) — never the ring-sized tails.
    // header + descriptors[0,N) are contiguous, so that is a single copy.
    const uint64_t nt = static_cast<uint64_t>(total_tasks);
    const uint64_t hdr_desc_bytes = sm_segs.descriptors + nt * sizeof(PTO2TaskDescriptor);
    char *host_base = static_cast<char *>(host_sm);
    char *dev_base = static_cast<char *>(device_sm);
    if (api->copy_to_device(dev_base, host_base, hdr_desc_bytes) != 0 ||
        api->copy_to_device(dev_base + sm_segs.payloads, host_base + sm_segs.payloads, nt * sizeof(PTO2TaskPayload)) !=
            0 ||
        api->copy_to_device(
            dev_base + sm_segs.slot_states, host_base + sm_segs.slot_states, nt * sizeof(PTO2TaskSlotState)
        ) != 0 ||
        api->copy_to_device(
            dev_base + sm_segs.completion_flags, host_base + sm_segs.completion_flags, nt * sizeof(std::atomic<uint8_t>)
        ) != 0) {
        LOG_ERROR("host-orch: H2D of populated SM failed");
        return -1;
    }
    return total_tasks;
}

}  // namespace

/**
 * Stage the per-callable resources (kernel binaries + orchestration SO) into
 * CallableArtifacts for subsequent per-run binding. Nothing here depends on
 * per-run argument values, so registration runs once per callable_id.
 *
 * @param callable  ChipCallable carrying the orch SO + child kernel binaries
 * @param api       Context-bound platform operations used during registration
 * @param out       Callable-owned artifacts retained across runs
 * @return 0 on success, -1 on failure
 */
extern "C" int register_callable_impl(const ChipCallable *callable, const HostApi *api, CallableArtifacts *out) {
    if (callable == nullptr) {
        LOG_ERROR("Callable pointer is null");
        return -1;
    }
    if (api == nullptr || out == nullptr) {
        LOG_ERROR("HostApi or out is null");
        return -1;
    }
    *out = CallableArtifacts{};
    out->signature.assign(callable->signature_, callable->signature_ + callable->sig_count());

    LOG_INFO("Registering %d kernel(s) in register_callable_impl", callable->child_count());
    if (upload_and_collect_child_addrs(
            callable, api, &out->kernel_addrs, &out->chip_buffer_dev, &out->chip_buffer_hash, &out->aicore_image_hash
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
        LOG_ERROR("Orchestration SO binary is required for host orchestration");
        return -1;
    }

    out->orch_so_data = orch_so_binary;
    out->orch_so_size = orch_so_size;
    out->func_name = callable->func_name();
    out->config_name = callable->config_name();

    // host_build_graph host-orch: dlopen the orchestration .so ON THE HOST and
    // resolve its entry symbol now. The handle is held across the prepared
    // callable's lifetime (closed by DeviceRunner::unregister_callable via
    // host_dlopen_handle); bind_callable_to_runtime_impl invokes the resolved
    // entry per run. This is what makes the host-side dlopen observable
    // (host_dlopen_count) while the AICPU never dlopens the orch .so.
    {
        const char *orch_func_name = callable->func_name();
        if (orch_func_name == nullptr || orch_func_name[0] == '\0') {
            LOG_ERROR("host-orch: orchestration function name is empty");
            return -1;
        }
        std::string so_path;
        if (!create_orch_so_tempfile(orch_so_binary, orch_so_size, &so_path)) {
            LOG_ERROR("host-orch: failed to materialize orchestration .so");
            return -1;
        }
        void *handle = dlopen(so_path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (handle == nullptr) {
            LOG_ERROR("host-orch: dlopen failed: %s", dlerror());
            return -1;
        }
        void *entry = dlsym(handle, orch_func_name);
        if (entry == nullptr) {
            LOG_ERROR("host-orch: dlsym('%s') failed: %s", orch_func_name, dlerror());
            dlclose(handle);
            return -1;
        }
        // The orch .so has its own framework_bind_runtime / g_current_runtime
        // (orchestration/common.cpp is compiled into it); resolve it now so the
        // per-run bind can set it before the .so's inline rt_submit_* run.
        void *bind_sym = dlsym(handle, "framework_bind_runtime");
        if (bind_sym == nullptr) {
            LOG_ERROR("host-orch: orch .so does not export framework_bind_runtime: %s", dlerror());
            dlclose(handle);
            return -1;
        }
        // Safe to unlink now: the handle keeps the .so mapped regardless of path.
        unlink(so_path.c_str());
        auto *eps = new HostOrchEntryPoints{};
        eps->entry = reinterpret_cast<OrchestrationEntryFunc>(entry);
        eps->bind = reinterpret_cast<OrchestrationBindFunc>(bind_sym);
        out->host_dlopen_handle = handle;
        out->host_orch_func_ptr = eps;
        LOG_INFO("host-orch: loaded orchestration entry '%s' on host", orch_func_name);
    }
    LOG_INFO("Orchestration SO: %zu bytes staged", orch_so_size);
    return 0;
}

/**
 * Per-run binding: build device-side argument storage (tensor copy-out, GM
 * heap, PTO2 shared memory) and publish it to the runtime. Assumes the
 * callable-side state (kernel binaries, orch SO bytes, func/config names)
 * is already populated by register_callable_impl.
 *
 * Splitting this from register_callable_impl matches the per-callable_id
 * design: register/run_prepared invokes this every call, while the prep
 * half runs only once per callable_id.
 *
 * @param runtime    Pointer to the per-run Runtime
 * @param api        Context-bound platform operations for this run
 * @param orch_args  Separated tensor/scalar arguments for this run
 * @return 0 on success, -1 on failure
 */
extern "C" int bind_callable_to_runtime_impl(
    Runtime *runtime, const HostApi *api, const ChipStorageTaskArgs *orch_args, void *host_orch_func_ptr,
    const ArgDirection *signature, int sig_count, const uint64_t *ring_task_window, const uint64_t *ring_heap,
    [[maybe_unused]] const uint64_t *ring_dep_pool
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
    // host_build_graph host-orch: register_callable_impl resolved the
    // orchestration entry on the host and passed it here as host_orch_func_ptr;
    // it is run below (after the arena is built) against a host SM mirror.
    int tensor_count = orch_args->tensor_count();
    int scalar_count = orch_args->scalar_count();
    LOG_INFO("RT2 bind: %d tensors + %d scalars, host orchestration mode", tensor_count, scalar_count);

    int64_t t_total_start = _now_ms();

    uint64_t eff_task_window_sizes[PTO2_MAX_RING_DEPTH];
    uint64_t eff_heap_sizes[PTO2_MAX_RING_DEPTH];
    if (!resolve_ring_config(ring_task_window, ring_heap, eff_task_window_sizes, eff_heap_sizes)) {
        return -1;
    }
    const std::string task_window_log = format_ring_array(eff_task_window_sizes);
    const std::string heap_log = format_ring_array(eff_heap_sizes);
    LOG_INFO("Ring buffer sizes: task_window=%s heap=%s", task_window_log.c_str(), heap_log.c_str());

    // Build device args: copy from input, replace host tensor pointers with device pointers
    ChipStorageTaskArgs device_args;

    // This run's host-view window. The accessor owns every mapping it
    // registers and releases them on every exit path, so no host view outlives
    // the point at which a task could make it stale.
    HostTensorAccessor tensor_access(api);

    HbgRetainedArgStaging staging;
    if (!staging.begin(api, orch_args, signature, sig_count)) {
        return -1;
    }
    const bool reuse_inputs = staging.reuses_inputs();

    int64_t t_args_start = _now_ms();
    {
        STRACE_A("simpler_run.bind.args", reuse_inputs ? "reuse=1" : "reuse=0");
        for (int i = 0; i < tensor_count; i++) {
            ChipTensor t = orch_args->tensor(i);

            if (t.is_device_memory()) {
                LOG_DEBUG("  ChipTensor %d: child memory, pass-through (0x%" PRIx64 ")", i, t.buffer.addr);
                device_args.add_tensor(t);
                continue;
            }

            void *host_ptr = reinterpret_cast<void *>(static_cast<uintptr_t>(t.buffer.addr));
            size_t size = static_cast<size_t>(t.nbytes());
            if (size == 0) {
                t.buffer.addr = 0;
                device_args.add_tensor(t);
                continue;
            }

            void *dev_ptr = staging.acquire(size);
            if (dev_ptr == nullptr) {
                LOG_ERROR("Failed to acquire retained staging for tensor %d", i);
                return -1;
            }

            // Pure write-only OUTPUT buffers are never read by the kernel and hold
            // no meaningful host content, so they need no device staging — the
            // kernel defines what it writes and any unwritten bytes are undefined.
            // IN/INOUT tensors without producer freshness are always copied.
            // A read tensor is skipped only when its nonzero content generation
            // and the complete staging layout match runner-owned metadata from a
            // fully successful prior run (including required INOUT copy-back).
            const bool copy_h2d = staging.copy_to_device_required(i);
            if (copy_h2d) {
                int rc = api->copy_to_device(dev_ptr, host_ptr, size);
                if (rc != 0) {
                    LOG_ERROR("Failed to stage tensor %d to device", i);
                    return -1;
                }
            }
            // Read-only INPUT tensors are never written by the kernel, so there is
            // no point copying them back D2H at the end. Index the signature
            // by the orch tensor index `i` (device-space tensors are skipped above
            // but do not consume a separate signature slot — scalars follow the
            // tensor entries). Anything not provably IN keeps the safe default of
            // copying back.
            bool needs_copy_back = !(signature != nullptr && i < sig_count && signature[i] == ArgDirection::IN);
            runtime->tensor_pairs_.push_back({host_ptr, dev_ptr, size, needs_copy_back, TensorReleaseKind::BufferNoop});
            LOG_DEBUG("  ChipTensor %d: %zu bytes at %p (h2d=%s)", i, size, dev_ptr, copy_h2d ? "copy" : "skip");

            // host_build_graph runs the orchestrator on the host, which may read
            // control tensors (e.g. paged_attention's context_lens/block_table) via
            // get_tensor_data to shape the graph. Give it a host view of this
            // buffer: the device buffer itself where the platform can map it into
            // the host address space (released in validate_runtime_impl before
            // device_free), otherwise the staging copy, which holds the same bytes
            // for the whole orchestration window and whose writes are pushed back
            // to the device. A tensor with neither is not host-accessible, so the
            // prepare fails here rather than the orchestrator dereferencing a
            // device address.
            if (!tensor_access.add(reinterpret_cast<uint64_t>(dev_ptr), size, host_ptr)) {
                LOG_ERROR("host-orch: no host view for tensor %d (dev_ptr %p, %zu bytes)", i, dev_ptr, size);
                return -1;
            }

            t.buffer.addr = reinterpret_cast<uint64_t>(dev_ptr);
            device_args.add_tensor(t);
        }
        for (int i = 0; i < scalar_count; i++) {
            device_args.add_scalar(orch_args->scalar(i));
        }
        staging.finish_population(
            &runtime->retained_temp_population_key_, &runtime->retained_temp_population_needs_copy_back_
        );
    }
    int64_t t_args_end = _now_ms();
    LOG_INFO(
        "TIMING: args_malloc_copy = %" PRId64 "ms (reuse_inputs=%d)", t_args_end - t_args_start, reuse_inputs ? 1 : 0
    );
    // Lay out the per-Worker static device arena. GM heap, PTO2 shared memory,
    // and the prebuilt runtime arena use three independent pooled device
    // allocations committed together by setup_static_arena.
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
    DeviceArena host_arena;
    PTO2RuntimeArenaLayout layout = runtime_reserve_layout(host_arena, eff_task_window_sizes, eff_heap_sizes);
    if (host_arena.commit(DeviceArena::kDefaultBaseAlign) == nullptr) {
        LOG_ERROR("Failed to commit host arena for prebuilt runtime image");
        return -1;
    }

    int64_t t_setup_start = _now_ms();
    if (api->setup_static_arena(total_heap_size, sm_size, layout.arena_size) != 0) {
        LOG_ERROR("Failed to setup pooled static arena");
        return -1;
    }
    int64_t t_setup_end = _now_ms();

    int64_t t_heap_start = _now_ms();
    void *gm_heap = api->acquire_pooled_gm_heap();
    int64_t t_heap_end = _now_ms();
    if (gm_heap == nullptr) {
        LOG_ERROR("Failed to acquire pooled GM heap");
        return -1;
    }
    runtime->set_gm_heap(gm_heap);

    int64_t t_sm_start = _now_ms();
    void *sm_ptr = api->acquire_pooled_gm_sm();
    int64_t t_sm_end = _now_ms();
    if (sm_ptr == nullptr) {
        LOG_ERROR("Failed to acquire pooled PTO2 shared memory");
        return -1;
    }
    runtime->set_gm_sm_ptr(sm_ptr);

    void *runtime_arena_dev = api->acquire_pooled_runtime_arena();
    if (runtime_arena_dev == nullptr) {
        LOG_ERROR("Failed to acquire pooled runtime arena");
        return -1;
    }

    // Set up orchestration state (consumed by the host orchestrator below)
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
        runtime_init_data_from_layout(host_arena, layout, PTO2_MODE_EXECUTE, sm_ptr, sm_size, gm_heap, eff_heap_sizes);
    if (rt == nullptr) {
        LOG_ERROR("runtime_init_data_from_layout failed");
        return -1;
    }
    runtime_wire_arena_pointers(host_arena, layout, rt);

    if (host_orch_func_ptr == nullptr) {
        LOG_ERROR("host-orch: orchestration entry points were not resolved");
        return -1;
    }
    {
        ChipTaskArgs orch_l2;
        orch_l2.create_from_chip_args(device_args);
        int32_t total_tasks = run_host_orchestration(
            runtime, api, tensor_access, rt, host_arena, layout, sm_ptr, sm_size, runtime_arena_dev, gm_heap,
            eff_heap_sizes, eff_task_window_sizes, host_orch_func_ptr, orch_l2
        );
        // The orchestrator is the only host-view reader; from here the device
        // owns these buffers, so drop the window on both exits.
        tensor_access.close();
        if (total_tasks < 0) {
            LOG_ERROR("host-orch: orchestration run failed");
            return -1;
        }
        runtime->host_total_tasks = total_tasks;
        LOG_INFO("host-orch: submitted %d tasks on host", total_tasks);
    }

    // Stash the layout inside the PTO2Runtime image so the AICPU can recover
    // every arena-internal offset after rtMemcpy. The runtime arena's device
    // base does NOT travel in this image — it's on the host Runtime
    // (set_prebuilt_arena below), since the AICPU needs that pointer
    // *before* it can dereference the image.
    rt->prebuilt_layout = layout;

    // Skip uploading the orchestrator block (fanin_seen_epoch / scope / tensormap,
    // ~8.5 MB): it is host-only dep-computation scratch that the AICPU scheduler
    // never reads. Ship [0, orch_start) (sm_handle) and [orch_end, arena_size)
    // (ready queues + runtime header + mailbox) whole; the orch block between them
    // is not sent. orch_start/orch_end are the first orch and first scheduler
    // reservations (runtime_reserve_layout order: sm_handle -> orch -> sched); the
    // ready queues ship in full because graph execution expands a GRAPH task into
    // on-device nodes that push past the host task count.
    const auto &sq = layout.sched;
    const size_t orch_start = layout.orch.off_fanin_seen_epoch;
    const size_t orch_end = sq.off_ready_queue_slots[0];
    always_assert(orch_start <= orch_end);
    char *arena_host = static_cast<char *>(host_arena.base());
    char *arena_dev = static_cast<char *>(runtime_arena_dev);
    int rc_upload = api->copy_to_device(arena_dev, arena_host, orch_start);
    if (rc_upload == 0) {
        rc_upload = api->copy_to_device(arena_dev + orch_end, arena_host + orch_end, layout.arena_size - orch_end);
    }
    if (rc_upload != 0) {
        LOG_ERROR("Failed to rtMemcpy prebuilt runtime arena to device (rc=%d)", rc_upload);
        return -1;
    }
    runtime->set_prebuilt_arena(runtime_arena_dev, layout.off_runtime);
    int64_t t_prebuilt_end = _now_ms();

    LOG_INFO("Device orchestration ready: %d tensors + %d scalars", tensor_count, scalar_count);

    int64_t t_total_end = _now_ms();
    LOG_INFO("TIMING: args_malloc_copy = %" PRId64 "ms", t_args_end - t_args_start);
    LOG_INFO("TIMING: static_arena_setup = %" PRId64 "ms", t_setup_end - t_setup_start);
    LOG_INFO("TIMING: gm_heap_acquire = %" PRId64 "ms", t_heap_end - t_heap_start);
    LOG_INFO("TIMING: shared_mem_acquire = %" PRId64 "ms", t_sm_end - t_sm_start);
    LOG_INFO("TIMING: prebuilt_runtime_arena = %" PRId64 "ms", t_prebuilt_end - t_prebuilt_start);
    LOG_INFO("TIMING: total_init_runtime_impl = %" PRId64 "ms", t_total_end - t_total_start);

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
 * @param runtime       Pointer to Runtime
 * @param execution_rc  Device-runner drain status after successful enqueue,
 *                      or enqueue status on failure
 * @return 0 on success, -1 on failure
 */
extern "C" int validate_runtime_impl(Runtime *runtime, const HostApi *api, int execution_rc) {
    if (runtime == nullptr) {
        LOG_ERROR("Runtime pointer is null");
        return -1;
    }
    if (api == nullptr) {
        LOG_ERROR("HostApi pointer is null");
        return -1;
    }

    int rc = 0;

    LOG_INFO("=== Copying Results Back to Host ===");

    // Copy all recorded tensors from device back to host
    TensorPair *tensor_pairs = runtime->tensor_pairs_.data();
    int tensor_pair_count = static_cast<int>(runtime->tensor_pairs_.size());

    LOG_INFO("ChipTensor pairs to process: %d", tensor_pair_count);

    bool skip_tensor_copy_back = execution_rc != 0;
    int32_t runtime_status = 0;
    PTO2SharedMemoryHeader host_header;
    memset(&host_header, 0, sizeof(host_header));

    if (execution_rc != 0) {
        runtime_status = pto2_read_runtime_status(runtime, api, &host_header);
    }
    if (runtime_status != 0) {
        int32_t orch_error_code = host_header.orch_error_code.load(std::memory_order_relaxed);
        int32_t sched_error_code = host_header.sched_error_code.load(std::memory_order_relaxed);
        LOG_RUNTIME_FAILURE(orch_error_code, sched_error_code, runtime_status);
    }

    const char *skip_copy_back_env = std::getenv("SIMPLER_SKIP_TENSOR_COPY_BACK");
    const bool copy_back_disabled =
        skip_copy_back_env != nullptr && skip_copy_back_env[0] != '\0' && skip_copy_back_env[0] != '0';
    if (skip_tensor_copy_back) {
        LOG_WARN("Skipping tensor copy-back because execution failed");
    } else if (copy_back_disabled) {
        // Perf / multi-round timing: INOUT D2H of multi‑GiB KV dominates Host wall
        // and is unused when golden compare is off.
        LOG_INFO("Skipping tensor copy-back (SIMPLER_SKIP_TENSOR_COPY_BACK)");
    } else {
        for (int i = 0; i < tensor_pair_count; i++) {
            const TensorPair &pair = tensor_pairs[i];

            // Skip if device pointer is null
            if (pair.dev_ptr == nullptr) {
                LOG_WARN("ChipTensor %d has null device pointer, skipping", i);
                continue;
            }

            // If host pointer is null, this is a device-only allocation (no copy-back)
            if (pair.host_ptr == nullptr) {
                LOG_DEBUG("ChipTensor %d: device-only allocation (no copy-back)", i);
                continue;
            }

            // Read-only INPUT tensors were uploaded H2D but the kernel never
            // wrote them — copying them back (potentially ~GB) is pure waste.
            // They are still device_free'd in the cleanup loop below.
            if (!pair.needs_copy_back) {
                LOG_DEBUG("ChipTensor %d: read-only input, skipping copy-back", i);
                continue;
            }

            int copy_rc = api->copy_from_device(pair.host_ptr, pair.dev_ptr, pair.size);
            if (copy_rc != 0) {
                LOG_ERROR("Failed to copy tensor %d from device: %d", i, copy_rc);
                rc = copy_rc;
            } else {
                LOG_DEBUG("ChipTensor %d: %zu bytes copied to host", i, pair.size);
            }
        }
    }

    // Cleanup device tensors that this run owns. Retained-temp slices
    // (BufferNoop) stay alive for the next bind — DeviceRunner frees them at
    // finalize.
    LOG_INFO("=== Cleaning Up ===");
    int freed = 0;
    for (int i = 0; i < tensor_pair_count; i++) {
        if (tensor_pairs[i].dev_ptr == nullptr) {
            continue;
        }
        if (tensor_pairs[i].release_kind == TensorReleaseKind::BufferNoop) {
            continue;
        }
        api->device_free(tensor_pairs[i].dev_ptr);
        ++freed;
    }
    LOG_INFO("Freed %d device allocations (%d retained-temp slices kept)", freed, tensor_pair_count - freed);

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
        LOG_INFO("Cleared %d kernel dispatch-table entries", kernel_count);
    }
    runtime->clear_registered_kernels();

    const bool population_valid = execution_rc == 0 && runtime_status == 0 && rc == 0 &&
                                  (!runtime->retained_temp_population_needs_copy_back_ || !copy_back_disabled);
    if (population_valid && !runtime->retained_temp_population_key_.empty()) {
        api->set_retained_temp_metadata(
            runtime->retained_temp_population_key_.data(),
            runtime->retained_temp_population_key_.size() * sizeof(runtime->retained_temp_population_key_[0])
        );
    }
    runtime->retained_temp_population_key_.clear();
    runtime->retained_temp_population_needs_copy_back_ = false;

    // Clear tensor pairs
    runtime->tensor_pairs_.clear();

    LOG_INFO("=== Finalize Complete ===");

    if (rc == 0 && runtime_status != 0) {
        rc = runtime_status;
    }

    return rc;
}

// host_build_graph resolves orchestration on the host, so it exports no AICPU
// entries beyond the base {simpler_aicpu_exec, simpler_aicpu_init} — in
// particular it does not export simpler_aicpu_register_callable. Reporting an
// empty extra-symbol set keeps the common AICPU loader from looking for it.
extern "C" const char *const *runtime_extra_aicpu_symbols(size_t *count) {
    if (count != nullptr) {
        *count = 0;
    }
    return nullptr;
}
