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
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <unordered_map>
#include <vector>

#include "../common/pto_runtime_status.h"
#include "../runtime/common.h"
#include "../runtime/dep_gen_host_graph.h"
#include "../runtime/graph_execution.h"
#include "../runtime/host_tensor_access.h"
#include "../runtime/graph_host_state.h"
#include "../runtime/host_phase_trace.h"
#include "../runtime/pto_orchestrator.h"
#include "../runtime/pto_runtime2.h"
#include "../runtime/pto_shared_memory.h"
#include "../runtime/pto_types.h"
#include "../runtime/runtime.h"
#include "../../../../common/runtime_status/error_log.h"
#include "../../../../common/task_interface/call_config.h"
#include "../../../../common/worker/pto_runtime_c_api.h"
#include "callable.h"
#include "common/host_log_binding.h"
#include "common/log_clock.h"
#include "common/platform_config.h"
#include "common/unified_log.h"
#include "host_log.h"
#include "host/raii_scope_guard.h"
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

static bool is_power_of_2_u64(uint64_t value) { return value != 0 && (value & (value - 1)) == 0; }

// Host monotonic clock, shared with the record pool so spans and records can be
// read against each other.
static int64_t bind_now_ns() { return static_cast<int64_t>(host_phase_now_ns()); }

// Close one segment of the bind path, recording it and keeping its attributes for
// the line the breakdown prints at the end of the pass.
//
// The breakdown is LOG_TIMING lines rather than `[STRACE]` markers on purpose:
// the marker grammar is the platform's public per-run-stage contract (see
// pto_runtime_c_api.h and docs/dfx/host-trace.md) whose consumers key off a fixed
// stage set, while everything below is host_build_graph's internal breakdown of
// one stage. LOG_TIMING sits at the default log threshold, so these are visible
// without a flag and at any --rounds.
static void record_bind_phase(HostPhaseKind kind, int64_t start_ns, const char *attrs = "", uint64_t payload = 0) {
    host_phase_record_bind(static_cast<uint32_t>(kind), static_cast<uint64_t>(start_ns), attrs, payload);
}

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
typedef void (*OrchestrationPrewarmFunc)();

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
            // payload is relative to its slot and moves with the SM image.
            relocate(state->task);
        }
    }
    return ok;
}

bool upload_graph_block(const HostApi *api, GraphHostState &graph_state, uint64_t &uploaded_bytes) {
    uploaded_bytes = 0;
    const size_t count = graph_host_upload_count(graph_state);
    struct PackedDefinition {
        const GraphHostDefinition *entry;
        const GraphDefinition *definition;
        size_t offset;
    };
    struct PackedSubmission {
        PTO2TaskSlotState *outer_slot;
        const GraphSubmission *submission;
        size_t definition_index;
        size_t offset;
    };
    auto reserve_region = [](size_t bytes, size_t alignment, size_t *cursor, size_t *offset) -> bool {
        if (cursor == nullptr || offset == nullptr || alignment == 0 || (alignment & (alignment - 1)) != 0 ||
            *cursor > SIZE_MAX - (alignment - 1)) {
            return false;
        }
        const size_t aligned = (*cursor + alignment - 1) & ~(alignment - 1);
        if (bytes > SIZE_MAX - aligned) return false;
        *offset = aligned;
        *cursor = aligned + bytes;
        return true;
    };

    // Compute and validate the complete layout before asking the arena bank to
    // allocate. Definitions lead the block; fixed-size submissions follow and
    // refer to their Definition by device_base + offset.
    GraphHostDefinitionList definitions = graph_host_definitions(graph_state);
    std::vector<PackedDefinition> packed_definitions;
    packed_definitions.reserve(definitions.entries.size());
    std::unordered_map<uint64_t, size_t> definition_indices;
    size_t block_bytes = 0;
    for (const GraphHostDefinition &entry : definitions.entries) {
        if (entry.data == nullptr || entry.bytes < sizeof(GraphDefinition)) continue;
        const auto *definition = reinterpret_cast<const GraphDefinition *>(entry.data);
        if (definition->total_bytes != entry.bytes || definition->full_key != entry.full_key) continue;
        if (entry.bytes > SIZE_MAX - sizeof(GraphDefinitionHeader)) return false;
        const size_t object_bytes = sizeof(GraphDefinitionHeader) + entry.bytes;
        size_t offset = 0;
        if (!reserve_region(object_bytes, alignof(GraphDefinitionHeader), &block_bytes, &offset)) return false;
        definition_indices.emplace(definition->content_hash, packed_definitions.size());
        packed_definitions.push_back(PackedDefinition{&entry, definition, offset});
    }

    std::vector<PackedSubmission> packed_submissions;
    packed_submissions.reserve(count);
    for (size_t index = 0; index < count; ++index) {
        std::optional<GraphHostUpload> upload = graph_host_upload(graph_state, index);
        if (!upload.has_value() || upload->outer_slot == nullptr || upload->data == nullptr ||
            upload->bytes != sizeof(GraphSubmission) || upload->outer_slot->task_kind != TaskKind::GRAPH ||
            upload->outer_slot->task == nullptr) {
            LOG_ERROR("host-orch: invalid pending Graph POD image");
            return false;
        }
        const auto *submission = reinterpret_cast<const GraphSubmission *>(upload->data);
        auto object_it = definition_indices.find(submission->definition_hash);
        if (object_it == definition_indices.end()) {
            LOG_ERROR("host-orch: Graph submission has no packed Definition object");
            return false;
        }
        const PackedDefinition &packed_definition = packed_definitions[object_it->second];
        const GraphDefinition *definition = packed_definition.definition;
        if (definition->task_count == 0 || definition->task_count > GRAPH_MAX_NODES ||
            definition->full_key != submission->graph_key || definition->execution_storage_bytes == 0) {
            LOG_ERROR("host-orch: invalid Graph Definition for submission");
            return false;
        }
        size_t offset = 0;
        if (!reserve_region(sizeof(GraphSubmission), alignof(GraphSubmission), &block_bytes, &offset)) return false;
        packed_submissions.push_back({upload->outer_slot, submission, object_it->second, offset});
    }
    if (block_bytes == 0) return true;

    constexpr size_t block_alignment = alignof(GraphDefinitionHeader) > alignof(GraphSubmission) ?
                                           alignof(GraphDefinitionHeader) :
                                           alignof(GraphSubmission);
    void *block = api->acquire_graph_block(block_bytes, block_alignment);
    if (block == nullptr) {
        LOG_ERROR("host-orch: failed to acquire %zu bytes for the arena-bank GraphBlock", block_bytes);
        return false;
    }

    std::vector<std::byte> staging(block_bytes, std::byte{0});
    auto *device_base = static_cast<std::byte *>(block);
    for (const PackedDefinition &packed : packed_definitions) {
        auto *header = reinterpret_cast<GraphDefinitionHeader *>(staging.data() + packed.offset);
        header->magic = GRAPH_DEFINITION_OBJECT_MAGIC;
        header->verify_state.store(
            static_cast<uint32_t>(GraphDefinitionVerifyState::UPLOADED), std::memory_order_relaxed
        );
        header->definition_bytes = static_cast<uint32_t>(packed.entry->bytes);
        header->content_hash = packed.definition->content_hash;
        header->full_key = packed.definition->full_key;
        std::memcpy(
            staging.data() + packed.offset + sizeof(GraphDefinitionHeader), packed.entry->data, packed.entry->bytes
        );
    }
    for (const PackedSubmission &packed : packed_submissions) {
        GraphSubmission submission = *packed.submission;
        submission.definition_addr =
            reinterpret_cast<uint64_t>(device_base + packed_definitions[packed.definition_index].offset);
        submission.local_execution = 0;
        submission.activation_gate = 0;
        std::memcpy(staging.data() + packed.offset, &submission, sizeof(submission));
    }

    if (api->copy_to_device(block, staging.data(), block_bytes) != 0) {
        LOG_ERROR("host-orch: failed to upload the packed GraphBlock");
        return false;
    }
    for (const PackedSubmission &packed : packed_submissions)
        packed.outer_slot->graph_context = device_base + packed.offset;
    uploaded_bytes = block_bytes;
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
    std::unique_ptr<uint8_t[]> host_sm_buf(new uint8_t[sm_size]);
    void *host_sm = host_sm_buf.get();
    std::memset(host_sm, 0, sm_segs.descriptors);

    // Re-point the orchestrator half at the host SM (scheduler keeps device SM).
    // init_data_from_layout resets the orchestrator state, so this is safe.
    if (!rt->orchestrator.init_data_from_layout(
            layout.orch, host_arena, host_sm, gm_heap, eff_heap_sizes[0], eff_task_window_sizes[0]
        )) {
        LOG_ERROR("host-orch: orchestrator re-init against host SM failed");
        return -1;
    }
    rt->orchestrator.wire_arena_pointers(layout.orch, host_arena, rt->scheduler);

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

    const int64_t t_orch_ns = bind_now_ns();
    rt_scope_begin(rt);
    entry_points->entry(orch_l2);
    rt_scope_end(rt);
    rt_orchestration_done(rt);
#if SIMPLER_ORCH_PROFILING
    // Per-sub-step cumulatives across this pass's submits. The accumulators only
    // exist in a SIMPLER_ORCH_PROFILING build (build_runtimes.py --profiling-orch 1),
    // and reading them also resets them, so this is the pass's own total. Emitted
    // as spans rather than LOG_INFO because INFO is suppressed at the default log
    // level. Like the phase spans these are summed cost shares, not intervals.
    {
        const PTO2OrchProfilingData prof = orchestrator_get_profiling();
        const std::pair<const char *, uint64_t> steps[] = {
            {"alloc", prof.alloc_cycle},   {"args", prof.args_cycle},   {"lookup", prof.lookup_cycle},
            {"insert", prof.insert_cycle}, {"fanin", prof.fanin_cycle}, {"scope_end", prof.scope_end_cycle},
        };
        for (const auto &step : steps) {
            if (step.second == 0) continue;
            LOG_TIMING(
                "host-orch step=%s cycles=%" PRIu64 " submits=%" PRId64, step.first, step.second, prof.submit_count
            );
        }
    }
#endif

    const int32_t total_tasks = pto2_sm_layout::ring_current_task_index_addr(host_sm)->load(std::memory_order_acquire);
    {
        char attrs[96];
        snprintf(
            attrs, sizeof(attrs), "tasks=%" PRId32 " heap_used=%" PRIu64, total_tasks,
            rt->orchestrator.ring.task_allocator.heap_used_bytes()
        );
        record_bind_phase(HostPhaseKind::BindHostOrch, t_orch_ns, attrs);
    }
    // After the span closes: the reduction walks a few hundred records and emits
    // five markers, which must not be charged to the pass it measures.

    const int64_t t_graph_ns = bind_now_ns();
    uint64_t graph_bytes = 0;
    if (!upload_graph_block(api, *graph_state, graph_bytes)) return -1;
    {
        char attrs[96];
        snprintf(attrs, sizeof(attrs), "count=%zu bytes=%" PRIu64, graph_host_upload_count(*graph_state), graph_bytes);
        record_bind_phase(HostPhaseKind::BindGraphUpload, t_graph_ns, attrs, graph_bytes);
    }

    // total_tasks sizes the bounded per-segment H2D copies below; a value outside
    // [0, task_window] would make those copies read/write out of bounds.
    if (total_tasks < 0 || static_cast<uint64_t>(total_tasks) > eff_task_window_sizes[0]) {
        LOG_ERROR("host-orch: total_tasks %d out of range [0, %" PRIu64 "]", total_tasks, eff_task_window_sizes[0]);
        return -1;
    }
    host_phase_trace_note_submitted(static_cast<uint64_t>(total_tasks));

    // Relocate the host-DDR cross-task pointers to their final DEVICE addresses
    // on the host, before the SM and arena leave for the device. Pointers into
    // the SM shift by sm_delta; pointers into the arena (fanout adjacency, wiring
    // queue) shift by arena_delta. After this both the SM and arena carry device
    // addresses, so the device boots scheduler-only.
    const int64_t sm_delta = static_cast<int64_t>(reinterpret_cast<uint64_t>(device_sm)) -
                             static_cast<int64_t>(reinterpret_cast<uint64_t>(host_sm));
    const int64_t arena_delta = static_cast<int64_t>(reinterpret_cast<uint64_t>(device_arena)) -
                                static_cast<int64_t>(reinterpret_cast<uint64_t>(host_arena.base()));
    const int64_t t_reloc_ns = bind_now_ns();
    if (!relocate_host_orch_image(
            host_sm_handle, reinterpret_cast<uint64_t>(host_sm), sm_size, sm_delta,
            reinterpret_cast<uint64_t>(host_arena.base()), layout.arena_size, arena_delta
        )) {
        LOG_ERROR("host-orch: relocation failed; refusing to H2D an image with unrelocated host pointers");
        return -1;
    }
    record_bind_phase(HostPhaseKind::BindRelocate, t_reloc_ns);

    // Ship descriptors[0,N), the live TaskPayloadSpace prefix,
    // slot_states[0,N) and completion_flags[0,N) — never the ring-sized tails.
    // header + descriptors[0,N) are contiguous, so that is a single copy.
    const uint64_t nt = static_cast<uint64_t>(total_tasks);
    const uint64_t hdr_desc_bytes = sm_segs.descriptors + nt * sizeof(PTO2TaskDescriptor);
    const uint64_t payload_bytes = rt->orchestrator.task_payload_space_used_bytes;
    char *host_base = static_cast<char *>(host_sm);
    char *dev_base = static_cast<char *>(device_sm);
    const int64_t t_sm_h2d_ns = bind_now_ns();
    if (api->copy_to_device(dev_base, host_base, hdr_desc_bytes) != 0 ||
        api->copy_to_device(dev_base + sm_segs.payloads, host_base + sm_segs.payloads, payload_bytes) != 0 ||
        api->copy_to_device(
            dev_base + sm_segs.slot_states, host_base + sm_segs.slot_states, nt * sizeof(PTO2TaskSlotState)
        ) != 0 ||
        api->copy_to_device(
            dev_base + sm_segs.completion_flags, host_base + sm_segs.completion_flags, nt * sizeof(std::atomic<uint8_t>)
        ) != 0) {
        LOG_ERROR("host-orch: H2D of populated SM failed");
        return -1;
    }
    {
        const uint64_t sm_h2d_bytes =
            hdr_desc_bytes + payload_bytes + nt * sizeof(PTO2TaskSlotState) + nt * sizeof(std::atomic<uint8_t>);
        char attrs[96];
        snprintf(attrs, sizeof(attrs), "nt=%" PRIu64 " bytes=%" PRIu64, nt, sm_h2d_bytes);
        record_bind_phase(HostPhaseKind::BindSmH2d, t_sm_h2d_ns, attrs, sm_h2d_bytes);
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
        const char *bind_log_error = nullptr;
        if (simpler::log::bind_loaded_host_log_state(handle, HostLogger::get_instance().state(), &bind_log_error) !=
            0) {
            LOG_ERROR(
                "host-orch: failed to bind host-log state: %s",
                bind_log_error != nullptr ? bind_log_error : "unknown error"
            );
            dlclose(handle);
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
        void *prewarm_sym = dlsym(handle, "framework_prewarm_graph_recorders");
        if (prewarm_sym == nullptr) {
            LOG_ERROR("host-orch: orch .so does not export framework_prewarm_graph_recorders: %s", dlerror());
            dlclose(handle);
            return -1;
        }
        reinterpret_cast<OrchestrationPrewarmFunc>(prewarm_sym)();
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
 * design: register/simpler_run invokes this every call, while the prep
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

    // Arm before the first segment below: the record pool has to exist for
    // `args`, which runs well before the device collector is provisioned. The
    // guard ends the pass on every exit, not just the successful one — a bind
    // that fails part-way is exactly when its breakdown is worth having, and an
    // unfinished pass publishes nothing.
    host_phase_trace_begin(api);
    auto host_phase_guard = RAIIScopeGuard([]() {
        host_phase_trace_end();
    });

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

    const int64_t t_args_ns = bind_now_ns();
    uint64_t staged_bytes = 0;
    int staged_tensors = 0;
    for (int i = 0; i < tensor_count; i++) {
        ChipTensor t = orch_args->tensor(i);

        if (t.is_device_memory()) {
            LOG_DEBUG("  ChipTensor %d: child memory, pass-through (0x%" PRIx64 ")", i, t.buffer.addr);
            device_args.add_tensor(t);
            continue;
        }

        void *host_ptr = reinterpret_cast<void *>(static_cast<uintptr_t>(t.buffer.addr));
        size_t size = static_cast<size_t>(t.nbytes());

        void *dev_ptr = api->device_malloc(size);
        if (dev_ptr == nullptr) {
            LOG_ERROR("Failed to allocate device memory for tensor %d", i);
            return -1;
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
                api->device_free(dev_ptr);
                return -1;
            }
            staged_bytes += static_cast<uint64_t>(size);
            ++staged_tensors;
        }
        // Read-only INPUT tensors are never written by the kernel, so there is
        // no point copying them back D2H at the end. Index the signature
        // by the orch tensor index `i` (device-space tensors are skipped above
        // but do not consume a separate signature slot — scalars follow the
        // tensor entries). Anything not provably IN keeps the safe default of
        // copying back.
        bool needs_copy_back = !(signature != nullptr && i < sig_count && signature[i] == ArgDirection::IN);
        runtime->tensor_pairs_.push_back({host_ptr, dev_ptr, size, needs_copy_back});
        LOG_DEBUG("  ChipTensor %d: %zu bytes at %p", i, size, dev_ptr);

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
    {
        char attrs[128];
        snprintf(
            attrs, sizeof(attrs), "ntensor=%d staged=%d bytes=%" PRIu64, tensor_count, staged_tensors, staged_bytes
        );
        record_bind_phase(HostPhaseKind::BindArgs, t_args_ns, attrs);
    }

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

    const int64_t t_arena_build_ns = bind_now_ns();
    DeviceArena host_arena;
    PTO2RuntimeArenaLayout layout = runtime_reserve_layout(host_arena, eff_task_window_sizes, eff_heap_sizes);
    if (host_arena.commit(DeviceArena::kDefaultBaseAlign) == nullptr) {
        LOG_ERROR("Failed to commit host arena for prebuilt runtime image");
        return -1;
    }
    {
        char attrs[64];
        snprintf(attrs, sizeof(attrs), "bytes=%" PRIu64, static_cast<uint64_t>(layout.arena_size));
        record_bind_phase(HostPhaseKind::BindArenaBuild, t_arena_build_ns, attrs);
    }

    const int64_t t_static_arena_ns = bind_now_ns();
    if (api->setup_static_arena(total_heap_size, sm_size, layout.arena_size) != 0) {
        LOG_ERROR("Failed to setup pooled static arena");
        return -1;
    }
    {
        char attrs[96];
        snprintf(attrs, sizeof(attrs), "heap=%" PRIu64 " sm=%" PRIu64, total_heap_size, sm_size);
        record_bind_phase(HostPhaseKind::BindStaticArena, t_static_arena_ns, attrs);
    }

    const int64_t t_heap_ns = bind_now_ns();
    void *gm_heap = api->acquire_pooled_gm_heap();
    record_bind_phase(HostPhaseKind::BindGmHeap, t_heap_ns);
    if (gm_heap == nullptr) {
        LOG_ERROR("Failed to acquire pooled GM heap");
        return -1;
    }
    runtime->set_gm_heap(gm_heap);

    const int64_t t_sm_ns = bind_now_ns();
    void *sm_ptr = api->acquire_pooled_gm_sm();
    record_bind_phase(HostPhaseKind::BindSharedMem, t_sm_ns);
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
    const int64_t t_runtime_init_ns = bind_now_ns();
    PTO2Runtime *rt =
        runtime_init_data_from_layout(host_arena, layout, PTO2_MODE_EXECUTE, sm_ptr, sm_size, gm_heap, eff_heap_sizes);
    if (rt == nullptr) {
        LOG_ERROR("runtime_init_data_from_layout failed");
        return -1;
    }
    runtime_wire_arena_pointers(host_arena, layout, rt);
    record_bind_phase(HostPhaseKind::BindRuntimeInit, t_runtime_init_ns);

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
        const size_t view_count = tensor_access.mapping_count();
        const uint64_t view_bytes = tensor_access.mapped_bytes();
        const int64_t t_view_close_ns = bind_now_ns();
        tensor_access.close();
        {
            char attrs[96];
            snprintf(attrs, sizeof(attrs), "count=%zu bytes=%" PRIu64, view_count, view_bytes);
            record_bind_phase(HostPhaseKind::BindHostViewClose, t_view_close_ns, attrs);
        }
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

    // The arena is partitioned into three zones (see PTO2RuntimeArenaLayout) and
    // only the middle one is copied: the host-only orchestrator block sits before
    // it, and the regions the device initializes itself (sm_handle, scheduler
    // state, queue slot arrays) sit after it. So bind is one copy of one range,
    // with both bounds taken from the layout rather than inferred from a
    // reservation order here.
    const size_t copied_begin = layout.off_copied_begin;
    const size_t copied_end = layout.off_copied_end;
    always_assert(copied_begin <= copied_end && copied_end <= layout.arena_size);
    char *arena_host = static_cast<char *>(host_arena.base());
    char *arena_dev = static_cast<char *>(runtime_arena_dev);
    const int64_t t_arena_h2d_ns = bind_now_ns();
    int rc_upload = api->copy_to_device(arena_dev + copied_begin, arena_host + copied_begin, copied_end - copied_begin);
    if (rc_upload != 0) {
        LOG_ERROR("Failed to rtMemcpy prebuilt runtime arena to device (rc=%d)", rc_upload);
        return -1;
    }
    {
        const uint64_t arena_h2d_bytes = static_cast<uint64_t>(copied_end - copied_begin);
        char attrs[96];
        snprintf(attrs, sizeof(attrs), "bytes=%" PRIu64, arena_h2d_bytes);
        record_bind_phase(HostPhaseKind::BindArenaH2d, t_arena_h2d_ns, attrs, arena_h2d_bytes);
    }
    runtime->set_prebuilt_arena(runtime_arena_dev, layout.off_runtime);

    LOG_INFO("Device orchestration ready: %d tensors + %d scalars", tensor_count, scalar_count);

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

    if (skip_tensor_copy_back) {
        LOG_WARN("Skipping tensor copy-back because execution failed");
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

    // Cleanup device tensors
    LOG_INFO("=== Cleaning Up ===");
    for (int i = 0; i < tensor_pair_count; i++) {
        if (tensor_pairs[i].dev_ptr != nullptr) {
            api->device_free(tensor_pairs[i].dev_ptr);
        }
    }
    LOG_INFO("Freed %d device allocations", tensor_pair_count);

    // The dispatch table is owned by bind_callable_to_runtime, which clears it
    // before replaying the active callable's addresses. The chip-callable device
    // buffer behind those addresses is pool-managed by DeviceRunner (keyed by
    // content hash) and bulk-freed in DeviceRunner::finalize(), so re-running the
    // same callable repeatedly does not re-upload.

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
