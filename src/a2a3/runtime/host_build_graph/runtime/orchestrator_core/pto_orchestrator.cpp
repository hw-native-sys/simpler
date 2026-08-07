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
 * host_build_graph orchestrator implementation
 *
 * Implements orchestrator state management, scope handling, and task submission.
 *
 * Based on: docs/RUNTIME_LOGIC.md
 */

#include "pto_orchestrator.h"

#include <assert.h>
#include <inttypes.h>
#include <limits>
#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <iterator>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common/platform_config.h"
#include "common/unified_log.h"
#include "dep_gen_host_graph.h"
#include "pto_dep_compute.h"
#include "graph_execution.h"
#include "graph_host_state.h"
#include "pto_runtime2_types.h"
#include "pto_shared_memory.h"
#include "pto_tensormap.h"
#include "pto_types.h"
#include "tensor.h"

#if SIMPLER_DFX
#include "aicpu/scope_stats_collector_aicpu.h"
#include "aicpu/args_dump_aicpu.h"
#endif

// Weak fallbacks: host/dep_gen_host_graph.cpp provides the strong symbols in the
// HOST build, where the orchestrator runs and the graph is captured. The AICPU
// build has no host graph and links these no-op stubs so the runtime translation
// unit is self-contained. Visibility is hidden so the HOST .so doesn't export
// them into the global dynamic symbol table where they'd shadow the strong
// symbols (same pattern as get_sys_cnt_aicpu / chip_swimlane_aicpu_record_orch_phase
// below).
__attribute__((weak, visibility("hidden"))) bool dep_gen_host_graph_enabled() { return false; }
__attribute__((weak, visibility("hidden"))) void dep_gen_host_graph_begin_task(
    uint64_t, bool, bool, const int32_t[3], int32_t, int32_t, const TensorRef *, const TensorArgType *
) {}
__attribute__((weak, visibility("hidden"))) void dep_gen_host_graph_end_task() {}
__attribute__((weak, visibility("hidden"))) void dep_gen_host_graph_add_explicit_edge(uint64_t) {}
__attribute__((weak, visibility("hidden"))) void
dep_gen_host_graph_add_creator_edge(uint64_t, int32_t, const ChipTensor &) {}
__attribute__((weak, visibility("hidden"))) void dep_gen_host_graph_add_tensormap_edge(
    uint64_t, int32_t, const ChipTensor &, const PTO2TensorMapEntry &, OverlapStatus
) {}

// Scope_stats enable gate, queried via the same predicate idiom as
// dep_gen_host_graph_enabled above. The AICPU collector links the strong definition;
// host builds fall back to this weak `false`. Gating here still skips the
// cross-agent occupancy reads that feed the sample when scope_stats is disabled.
extern "C" __attribute__((weak, visibility("hidden"))) bool is_scope_stats_enabled() { return false; }

// Heap-ring wrap report, called from the allocator (pto_ring_buffer.h) on each
// wrap. Strong definition lives in the AICPU collector; host builds fall back to
// this weak no-op so the runtime translation unit stays self-contained.
extern "C" __attribute__((weak, visibility("hidden"))) void scope_stats_note_heap_wrap(int) {}

// AICore register accessor (aicpu/platform_regs.h). The host orchestrator's
// route_ready_once path transitively ODR-uses the early-dispatch doorbell inline
// (pto_scheduler.h ring_one_doorbell), but no core is gated during host
// graph-build, so the doorbell never fires and this weak host fallback only
// satisfies the linker. The AICPU build links the strong definition from
// platform/.../platform_regs.cpp; hidden so the HOST .so does not shadow it.
__attribute__((weak, visibility("hidden"))) volatile uint32_t *get_reg_ptr(uint64_t, RegId) {
    static volatile uint32_t sink = 0;
    return &sink;
}

// =============================================================================
// Orchestrator Profiling (compile-time toggle)
// =============================================================================
#if SIMPLER_ORCH_PROFILING
#include "aicpu/device_time.h"
#include "aicpu/chip_swimlane_collector_aicpu.h"
// Weak fallback for builds that don't link device_time.cpp (e.g. host).
// The strong symbol from platform/.../device_time.cpp wins in the AICPU build.
//
// IMPORTANT: visibility("hidden") is required to prevent the HOST .so from
// exporting this weak fallback into the global dynamic symbol table via
// RTLD_GLOBAL. Without it, when the AICPU .so is loaded and its PLT entry
// for get_sys_cnt_aicpu is resolved, the dynamic linker finds the HOST .so's
// weak definition first (already in global table) and uses it — returning 0.
// With hidden visibility, the HOST .so does not export this symbol globally,
// so the AICPU .so's PLT resolves to its own strong definition from
// device_time.cpp.
__attribute__((weak, visibility("hidden"))) uint64_t get_sys_cnt_aicpu() {
    // Host fallback: monotonic wall-clock in AICPU cycle units so the host-orch
    // deadlock/timeout backstops fire at their intended wall-clock (see the
    // detailed rationale on the same fallback in pto_runtime2.cpp).
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    // Scale sec and nsec separately (divisor is the constant 1e9): avoids a
    // div-by-zero when PLATFORM_PROF_SYS_CNT_FREQ >= 1 GHz and the truncation
    // error a `1e9 / FREQ` divisor would introduce for non-dividing frequencies.
    return static_cast<uint64_t>(ts.tv_sec) * PLATFORM_PROF_SYS_CNT_FREQ +
           static_cast<uint64_t>(ts.tv_nsec) * PLATFORM_PROF_SYS_CNT_FREQ / 1000000000ull;
}
// Weak fallback for builds that don't link chip_swimlane_collector_aicpu.cpp.
// The strong symbol from the AICPU build wins when profiling is available.
// Also hidden to prevent HOST .so from polluting the global symbol table.
__attribute__((weak, visibility("hidden"))) void
chip_swimlane_aicpu_record_orch_phase(uint64_t, uint64_t, uint64_t, uint32_t) {}
// Accumulated cycles per sub-step (only needed for ORCH_PROFILING export)
static uint64_t g_orch_sync_cycle = 0;       // tensormap sync
static uint64_t g_orch_alloc_cycle = 0;      // unified task+heap alloc
static uint64_t g_orch_args_cycle = 0;       // param copy
static uint64_t g_orch_lookup_cycle = 0;     // tensormap lookup + dep building
static uint64_t g_orch_insert_cycle = 0;     // tensormap insert
static uint64_t g_orch_fanin_cycle = 0;      // fanin list + early-return check
static uint64_t g_orch_scope_end_cycle = 0;  // scope_end overhead
static int64_t g_orch_submit_count = 0;
static uint32_t g_orch_submit_idx = 0;
uint64_t g_orch_alloc_wait_cycle = 0;
uint64_t g_orch_fanin_wait_cycle = 0;
uint64_t g_orch_alloc_atomic_count = 0;
uint64_t g_orch_args_atomic_count = 0;
uint64_t g_orch_scope_end_atomic_count = 0;
// Cycle accumulation is unconditional under SIMPLER_ORCH_PROFILING (that's what
// the flag is for) and feeds the per-sub-step `g_orch_*_cycle` cumulatives
// printed in the cold-path log.
//
// Per-submit ORCH_SUBMIT record is the only swim-lane emit on the orch
// path — one record per submit_task() / alloc_tensors() call spanning
// the entire [start, end] window. Per-sub-step phase records were dropped
// in favour of the cumulatives + per-submit envelope; the dispatcher
// already inserts one record at the end of each submit path via
// CYCLE_COUNT_ORCH_SUBMIT_RECORD.
#define CYCLE_COUNT_START()                                                            \
    bool _prof_active = (orch->chip_swimlane_level >= ChipSwimlaneLevel::ORCH_PHASES); \
    uint64_t _t0 = get_sys_cnt_aicpu(), _t1;                                           \
    uint64_t _submit_start_ts = _t0
#define CYCLE_COUNT_LAP(acc)       \
    do {                           \
        _t1 = get_sys_cnt_aicpu(); \
        acc += (_t1 - _t0);        \
        _t0 = _t1;                 \
    } while (0)
#define CYCLE_COUNT_ORCH_SUBMIT_RECORD(tid)                                                         \
    do {                                                                                            \
        if (_prof_active) {                                                                         \
            chip_swimlane_aicpu_record_orch_phase(_submit_start_ts, _t1, (tid), g_orch_submit_idx); \
        }                                                                                           \
    } while (0)
#elif SIMPLER_DFX
#include "aicpu/device_time.h"
#include "aicpu/chip_swimlane_collector_aicpu.h"
__attribute__((weak, visibility("hidden"))) uint64_t get_sys_cnt_aicpu() {
    // Host fallback: monotonic wall-clock in AICPU cycle units so the host-orch
    // deadlock/timeout backstops fire at their intended wall-clock (see the
    // detailed rationale on the same fallback in pto_runtime2.cpp).
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    // Scale sec and nsec separately (divisor is the constant 1e9): avoids a
    // div-by-zero when PLATFORM_PROF_SYS_CNT_FREQ >= 1 GHz and the truncation
    // error a `1e9 / FREQ` divisor would introduce for non-dividing frequencies.
    return static_cast<uint64_t>(ts.tv_sec) * PLATFORM_PROF_SYS_CNT_FREQ +
           static_cast<uint64_t>(ts.tv_nsec) * PLATFORM_PROF_SYS_CNT_FREQ / 1000000000ull;
}
__attribute__((weak, visibility("hidden"))) void
chip_swimlane_aicpu_record_orch_phase(uint64_t, uint64_t, uint64_t, uint32_t) {}
// submit_idx needed for swimlane task_id tagging (no cycle accumulation at this level)
static uint32_t g_orch_submit_idx = 0;
#define CYCLE_COUNT_START()                                                            \
    bool _prof_active = (orch->chip_swimlane_level >= ChipSwimlaneLevel::ORCH_PHASES); \
    uint64_t _t0 = _prof_active ? get_sys_cnt_aicpu() : 0, _t1 = 0;                    \
    uint64_t _submit_start_ts = _t0
#define CYCLE_COUNT_LAP(acc) \
    do {                     \
    } while (0)
#define CYCLE_COUNT_ORCH_SUBMIT_RECORD(tid)                                                         \
    do {                                                                                            \
        if (_prof_active) {                                                                         \
            _t1 = get_sys_cnt_aicpu();                                                              \
            chip_swimlane_aicpu_record_orch_phase(_submit_start_ts, _t1, (tid), g_orch_submit_idx); \
        }                                                                                           \
    } while (0)
#else
#define CYCLE_COUNT_START()
#define CYCLE_COUNT_LAP(acc)
#define CYCLE_COUNT_ORCH_SUBMIT_RECORD(tid)
#endif

static int32_t orch_mark_fatal(PTO2OrchestratorState *orch, int32_t error_code) {
    always_assert(orch != nullptr);
    orch->fatal = true;
    if (error_code == PTO2_ERROR_NONE || orch->sm_header == nullptr) {
        return PTO2_ERROR_NONE;
    }

    int32_t expected = PTO2_ERROR_NONE;
    std::atomic<int32_t> &orch_error_code = orch->sm_header->orch_error_code;
    if (orch_error_code.compare_exchange_strong(expected, error_code, std::memory_order_acq_rel)) {
        return error_code;
    }
    return expected;
}

static void
orch_report_fatal_v(PTO2OrchestratorState *orch, int32_t error_code, const char *func, const char *fmt, va_list args) {
    int32_t latched_code = orch_mark_fatal(orch, error_code);

#if SIMPLER_DFX
    // Flush the current scope's peaks BEFORE the FATAL log line, so the
    // diagnostic context (which pool/window filled up) appears right next to
    // the failure reason. on_fatal is latched, so duplicate fatals from
    // different layers don't print multiple stats lines.
    scope_stats_on_fatal();
#endif

    if (fmt == nullptr || fmt[0] == '\0') {
        if (latched_code != PTO2_ERROR_NONE && latched_code != error_code) {
            unified_log_error(func, "FATAL(code=%d, latched=%d)", error_code, latched_code);
        } else {
            unified_log_error(func, "FATAL(code=%d)", error_code);
        }
        return;
    }

    std::array<char, 1024> message{};
    vsnprintf(message.data(), message.size(), fmt, args);
    if (latched_code != PTO2_ERROR_NONE && latched_code != error_code) {
        unified_log_error(func, "FATAL(code=%d, latched=%d): %s", error_code, latched_code, message.data());
        return;
    }
    unified_log_error(func, "FATAL(code=%d): %s", error_code, message.data());
}

void PTO2OrchestratorState::report_fatal(int32_t error_code, const char *func, const char *fmt, ...) {
    auto *orch = this;
    va_list args;
    va_start(args, fmt);
    orch_report_fatal_v(orch, error_code, func, fmt, args);
    va_end(args);
}

enum class GraphRecordedTensorSource : uint8_t {
    BOUNDARY_EXACT,
    BOUNDARY_VIEW,
    INTERNAL,
    OWN_OUTPUT,
};

struct GraphRecordedTensorSourceRef {
    GraphRecordedTensorSource source{GraphRecordedTensorSource::BOUNDARY_EXACT};
    size_t source_index{0};
    uint64_t packed_offset{0};
};

struct GraphRecordedNode {
    std::array<int32_t, PTO2_SUBTASK_SLOT_COUNT> kernel_ids{};
    ActiveMask active_mask{};
    TaskAttrs task_attrs{};
    int16_t logical_block_num{1};
    int16_t total_required_subtasks{0};
    size_t total_output_size{0};
    uintptr_t record_packed_base{0};
    std::vector<ChipTensor> tensors;
    std::vector<GraphRecordedTensorSourceRef> tensor_sources;
    std::vector<uint64_t> scalars;
    std::vector<size_t> internal_fanins;
};

struct GraphRecording {
    uint64_t full_key{0};
    int32_t start_local_task_id{0};
    // Heap allocation pointer captured at graph_begin. Internal nodes reserve
    // scratch output buffers past it during the recording pass; graph_end rolls
    // the heap back here so the outer GRAPH task reclaims the same region.
    uint64_t heap_watermark{0};
    // The Graph boundary CoreTaskArgs, owned by the caller for the whole
    // rt_submit_graph_impl call (recording pass + graph_end). graph_end replays
    // it through graph_submit_definition to emit the outer GRAPH task with the
    // current invocation's boundary tensor addresses.
    const CoreTaskArgs *boundary_args{nullptr};
    bool unsupported{false};
    std::vector<ChipTensor> boundary_tensors;
    std::vector<TensorArgType> boundary_types;
    std::vector<GraphRecordedNode> nodes;
};

struct GraphPendingUpload {
    PTO2TaskSlotState *outer_slot{nullptr};
    std::vector<std::byte> image;
};

struct GraphHostState {
    std::unordered_map<uint64_t, std::vector<std::byte>> definitions;
    std::unique_ptr<GraphRecording> recording;
    std::vector<GraphPendingUpload> pending_uploads;
};

namespace {

GraphHostState *graph_state_from(PTO2OrchestratorState *orch) {
    return orch == nullptr ? nullptr : static_cast<GraphHostState *>(orch->graph_host_state);
}

uint64_t graph_full_key(uint64_t callable_hash, uint64_t graph_key) {
    uint64_t h = 1469598103934665603ULL;
    h = graph_hash_bytes(h, &callable_hash, sizeof(callable_hash));
    return graph_hash_bytes(h, &graph_key, sizeof(graph_key));
}

bool graph_tensor_exact(const ChipTensor &lhs, const ChipTensor &rhs) {
    if (lhs.ndims > MAX_TENSOR_DIMS || rhs.ndims > MAX_TENSOR_DIMS || lhs.buffer.addr != rhs.buffer.addr ||
        lhs.buffer.size != rhs.buffer.size || lhs.start_offset != rhs.start_offset || lhs.version != rhs.version ||
        lhs.ndims != rhs.ndims || lhs.dtype != rhs.dtype || lhs.manual_dep != rhs.manual_dep ||
        lhs.is_contiguous != rhs.is_contiguous || lhs.child_memory != rhs.child_memory) {
        return false;
    }
    return std::equal(std::begin(lhs.shapes), std::begin(lhs.shapes) + lhs.ndims, std::begin(rhs.shapes)) &&
           std::equal(std::begin(lhs.strides), std::begin(lhs.strides) + lhs.ndims, std::begin(rhs.strides));
}

bool graph_tensor_from_boundary(
    const GraphRecording &recording, const ChipTensor &tensor, GraphRecordedTensorSourceRef *source
) {
    for (size_t i = 0; i < recording.boundary_tensors.size(); ++i) {
        if (!graph_tensor_exact(tensor, recording.boundary_tensors[i])) continue;
        source->source = GraphRecordedTensorSource::BOUNDARY_EXACT;
        source->source_index = i;
        source->packed_offset = 0;
        return true;
    }
    for (size_t i = 0; i < recording.boundary_tensors.size(); ++i) {
        const ChipTensor &boundary = recording.boundary_tensors[i];
        if (tensor.buffer.addr != boundary.buffer.addr || tensor.buffer.size != boundary.buffer.size ||
            tensor.start_offset < boundary.start_offset) {
            continue;
        }
        source->source = GraphRecordedTensorSource::BOUNDARY_VIEW;
        source->source_index = i;
        source->packed_offset = tensor.start_offset - boundary.start_offset;
        return true;
    }
    return false;
}

bool graph_classify_tensor(
    const GraphRecording &recording, const GraphRecordedNode &current, int32_t task_index, const ChipTensor &tensor,
    GraphRecordedTensorSourceRef *source
) {
    if (graph_tensor_from_boundary(recording, tensor, source)) return true;
    const uint64_t tensor_addr = tensor.buffer.addr;
    for (int32_t producer_index = task_index; producer_index >= 0; --producer_index) {
        const GraphRecordedNode &producer =
            producer_index == task_index ? current : recording.nodes[static_cast<size_t>(producer_index)];
        if (producer.record_packed_base == 0 || producer.total_output_size == 0 ||
            producer.total_output_size > UINTPTR_MAX - producer.record_packed_base) {
            continue;
        }
        const uintptr_t begin = producer.record_packed_base;
        const uintptr_t end = begin + producer.total_output_size;
        if (tensor_addr < begin || tensor_addr >= end) continue;
        source->source =
            producer_index == task_index ? GraphRecordedTensorSource::OWN_OUTPUT : GraphRecordedTensorSource::INTERNAL;
        source->source_index = static_cast<size_t>(producer_index);
        source->packed_offset = tensor_addr - begin;
        return true;
    }
    return false;
}

void graph_record_mark_unsupported(PTO2OrchestratorState *orch) {
    GraphHostState *state = graph_state_from(orch);
    if (state != nullptr && state->recording != nullptr) state->recording->unsupported = true;
}

GraphBoundarySignature graph_boundary_signature(const ChipTensor &tensor, TensorArgType type, uint16_t alias_rep) {
    GraphBoundarySignature signature{};
    signature.buffer_size = tensor.buffer.size;
    std::copy(std::begin(tensor.shapes), std::end(tensor.shapes), std::begin(signature.shapes));
    std::copy(std::begin(tensor.strides), std::end(tensor.strides), std::begin(signature.strides));
    signature.alias_rep = alias_rep;
    signature.ndims = static_cast<uint8_t>(tensor.ndims);
    signature.dtype = static_cast<uint8_t>(tensor.dtype);
    signature.tag = static_cast<uint8_t>(type);
    signature.manual_dep = tensor.manual_dep ? 1 : 0;
    signature.is_contiguous = tensor.is_contiguous ? 1 : 0;
    return signature;
}

template <typename T>
uint32_t graph_append_section(std::vector<std::byte> *image, const std::vector<T> &values) {
    if (values.empty()) return 0;
    if (image->size() > UINT32_MAX || values.size() > UINT32_MAX / sizeof(T)) return 0;
    const size_t aligned = PTO2_ALIGN_UP(image->size(), alignof(T));
    const size_t bytes = values.size() * sizeof(T);
    if (aligned > UINT32_MAX || bytes > UINT32_MAX - aligned) return 0;
    image->resize(aligned + bytes);
    std::memcpy(image->data() + aligned, values.data(), bytes);
    return static_cast<uint32_t>(aligned);
}

std::optional<GraphTensorSourceRef> graph_pack_tensor_source(const GraphRecordedTensorSourceRef &source) {
    if (source.source_index > UINT16_MAX) return std::nullopt;

    GraphTensorSourceRef packed{};
    switch (source.source) {
    case GraphRecordedTensorSource::BOUNDARY_EXACT:
        packed.source = static_cast<uint8_t>(GraphTensorSource::BOUNDARY_EXACT);
        break;
    case GraphRecordedTensorSource::BOUNDARY_VIEW:
        packed.source = static_cast<uint8_t>(GraphTensorSource::BOUNDARY_VIEW);
        break;
    case GraphRecordedTensorSource::INTERNAL:
        packed.source = static_cast<uint8_t>(GraphTensorSource::INTERNAL);
        break;
    case GraphRecordedTensorSource::OWN_OUTPUT:
        packed.source = static_cast<uint8_t>(GraphTensorSource::OWN_OUTPUT);
        break;
    }
    packed.source_index = static_cast<uint16_t>(source.source_index);
    packed.packed_offset = source.packed_offset;
    return packed;
}

bool graph_build_definition(const GraphRecording &recording, std::vector<std::byte> *image) {
    if (image == nullptr || recording.unsupported || recording.nodes.empty() ||
        recording.nodes.size() > GRAPH_MAX_NODES || recording.boundary_tensors.size() > UINT16_MAX ||
        recording.boundary_tensors.size() != recording.boundary_types.size() ||
        std::any_of(recording.boundary_tensors.begin(), recording.boundary_tensors.end(), [](const ChipTensor &tensor) {
            return tensor.ndims > MAX_TENSOR_DIMS;
        })) {
        return false;
    }

    std::vector<uint32_t> fanout_counts(recording.nodes.size(), 0);
    std::vector<uint32_t> fanin_offsets(recording.nodes.size() + 1, 0);
    std::vector<uint16_t> fanin_indices;
    std::vector<uint16_t> roots;
    std::vector<uint64_t> node_offsets(recording.nodes.size(), 0);
    std::vector<GraphNodeDefinition> nodes(recording.nodes.size());
    std::vector<GraphTensor> tensors;
    std::vector<GraphTensorSourceRef> tensor_sources;
    std::vector<uint64_t> scalars;

    uint64_t required_heap = 0;
    uint32_t edge_count = 0;
    for (size_t i = 0; i < recording.nodes.size(); ++i) {
        const GraphRecordedNode &source = recording.nodes[i];
        if (source.total_output_size > static_cast<size_t>(INT32_MAX) ||
            source.tensors.size() > static_cast<size_t>(INT32_MAX) ||
            source.scalars.size() > static_cast<size_t>(INT32_MAX) || source.internal_fanins.size() > UINT16_MAX ||
            source.tensors.size() != source.tensor_sources.size() ||
            tensors.size() > UINT32_MAX - source.tensors.size() ||
            tensor_sources.size() > UINT32_MAX - source.tensor_sources.size() ||
            scalars.size() > UINT32_MAX - source.scalars.size() ||
            std::any_of(source.tensors.begin(), source.tensors.end(), [](const ChipTensor &tensor) {
                return tensor.ndims > MAX_TENSOR_DIMS;
            })) {
            return false;
        }
        node_offsets[i] = required_heap;
        const uint64_t output_bytes = PTO2_ALIGN_UP(source.total_output_size, PTO2_ALIGN_SIZE);
        if (required_heap > UINT64_MAX - output_bytes) return false;
        required_heap += output_bytes;

        fanin_offsets[i + 1] = fanin_offsets[i] + static_cast<uint32_t>(source.internal_fanins.size());
        if (source.internal_fanins.empty()) roots.push_back(static_cast<uint16_t>(i));
        for (size_t producer : source.internal_fanins) {
            if (producer >= i) return false;
            fanout_counts[producer]++;
            fanin_indices.push_back(static_cast<uint16_t>(producer));
            edge_count++;
        }

        GraphNodeDefinition &node = nodes[i];
        std::copy(source.kernel_ids.begin(), source.kernel_ids.end(), std::begin(node.kernel_id));
        node.active_mask = source.active_mask.raw();
        node.task_attrs = source.task_attrs.raw();
        node.logical_block_num = source.logical_block_num;
        node.total_required_subtasks = source.total_required_subtasks;
        node.tensor_count = static_cast<int32_t>(source.tensors.size());
        node.scalar_count = static_cast<int32_t>(source.scalars.size());
        node.total_output_size = static_cast<int32_t>(source.total_output_size);
        node.tensor_offset = static_cast<uint32_t>(tensors.size());
        node.scalar_offset = static_cast<uint32_t>(scalars.size());
        for (const ChipTensor &tensor : source.tensors)
            tensors.push_back(graph_tensor_pack(tensor));
        for (const GraphRecordedTensorSourceRef &tensor_source : source.tensor_sources) {
            std::optional<GraphTensorSourceRef> packed_source = graph_pack_tensor_source(tensor_source);
            if (!packed_source.has_value()) return false;
            tensor_sources.push_back(*packed_source);
        }
        scalars.insert(scalars.end(), source.scalars.begin(), source.scalars.end());
    }

    std::vector<uint32_t> fanout_offsets(recording.nodes.size() + 1, 0);
    for (size_t i = 0; i < recording.nodes.size(); ++i)
        fanout_offsets[i + 1] = fanout_offsets[i] + fanout_counts[i];
    std::vector<uint16_t> fanout_indices(edge_count);
    std::vector<uint32_t> cursors(fanout_offsets.begin(), fanout_offsets.end() - 1);
    for (size_t consumer = 0; consumer < recording.nodes.size(); ++consumer) {
        for (size_t producer : recording.nodes[consumer].internal_fanins) {
            fanout_indices[cursors[producer]++] = static_cast<uint16_t>(consumer);
        }
    }

    std::vector<GraphBoundarySignature> signatures;
    signatures.reserve(recording.boundary_tensors.size());
    for (size_t i = 0; i < recording.boundary_tensors.size(); ++i) {
        uint16_t alias_rep = static_cast<uint16_t>(i);
        for (size_t j = 0; j < i; ++j) {
            if (recording.boundary_tensors[j].buffer.addr == recording.boundary_tensors[i].buffer.addr &&
                recording.boundary_tensors[j].buffer.size == recording.boundary_tensors[i].buffer.size) {
                alias_rep = static_cast<uint16_t>(j);
                break;
            }
        }
        signatures.push_back(
            graph_boundary_signature(recording.boundary_tensors[i], recording.boundary_types[i], alias_rep)
        );
    }

    image->assign(sizeof(GraphDefinition), std::byte{0});
    GraphDefinition definition{};
    definition.full_key = recording.full_key;
    definition.required_heap = required_heap;
    definition.task_count = static_cast<uint32_t>(nodes.size());
    definition.edge_count = edge_count;
    definition.root_count = static_cast<uint32_t>(roots.size());
    definition.boundary_count = static_cast<uint32_t>(signatures.size());
    definition.tensor_arg_count = static_cast<uint32_t>(tensors.size());
    definition.scalar_arg_count = static_cast<uint32_t>(scalars.size());
    definition.off_fanout_offsets = graph_append_section(image, fanout_offsets);
    definition.off_fanout_indices = graph_append_section(image, fanout_indices);
    definition.off_fanin_offsets = graph_append_section(image, fanin_offsets);
    definition.off_fanin_indices = graph_append_section(image, fanin_indices);
    definition.off_root_indices = graph_append_section(image, roots);
    definition.off_node_offsets = graph_append_section(image, node_offsets);
    definition.off_nodes = graph_append_section(image, nodes);
    definition.off_tensors = graph_append_section(image, tensors);
    definition.off_tensor_sources = graph_append_section(image, tensor_sources);
    definition.off_scalars = graph_append_section(image, scalars);
    definition.off_boundary_signatures = graph_append_section(image, signatures);
    if (definition.off_fanout_offsets == 0 || definition.off_fanin_offsets == 0 || definition.off_node_offsets == 0 ||
        definition.off_nodes == 0 || definition.off_boundary_signatures == 0 ||
        (!tensors.empty() && definition.off_tensors == 0) ||
        (!tensor_sources.empty() && definition.off_tensor_sources == 0) ||
        (!scalars.empty() && definition.off_scalars == 0) ||
        (!fanout_indices.empty() && definition.off_fanout_indices == 0) ||
        (!fanin_indices.empty() && definition.off_fanin_indices == 0) ||
        (!roots.empty() && definition.off_root_indices == 0)) {
        return false;
    }
    definition.total_bytes = static_cast<uint32_t>(image->size());
    std::memcpy(image->data(), &definition, sizeof(definition));
    definition.content_hash = graph_hash_bytes(1469598103934665603ULL, image->data(), image->size());
    std::memcpy(image->data(), &definition, sizeof(definition));
    return true;
}

const GraphDefinition *graph_definition(const std::vector<std::byte> &image) {
    if (image.size() < sizeof(GraphDefinition)) return nullptr;
    const auto *definition = reinterpret_cast<const GraphDefinition *>(image.data());
    return definition->total_bytes == image.size() ? definition : nullptr;
}

}  // namespace

GraphHostStatePtr make_graph_host_state() { return GraphHostStatePtr{new (std::nothrow) GraphHostState{}}; }

void GraphHostStateDeleter::operator()(GraphHostState *state) const noexcept { delete state; }

size_t graph_host_upload_count(const GraphHostState &state) { return state.pending_uploads.size(); }

std::optional<GraphHostUpload> graph_host_upload(GraphHostState &state, size_t index) {
    if (index >= state.pending_uploads.size()) return std::nullopt;
    GraphPendingUpload &upload = state.pending_uploads[index];
    if (upload.outer_slot == nullptr || upload.image.empty()) return std::nullopt;
    return GraphHostUpload{upload.outer_slot, upload.image.data(), upload.image.size()};
}

static uint32_t next_fanin_seen_epoch(PTO2OrchestratorState *orch) {
    uint32_t next = orch->fanin_seen_current_epoch + 1;
    if (next == 0) {
        memset(
            orch->fanin_seen_epoch, 0, static_cast<size_t>(orch->sm_header->ring.task_window_size) * sizeof(uint32_t)
        );
        next = 1;
    }
    orch->fanin_seen_current_epoch = next;
    return next;
}

// Polling: fanin is a flat array of position-independent producer local ids on
// the payload (no dep-pool spill, no producer pointers). The builder writes them
// directly into payload->fanin_local_ids as producers are appended, deduping by
// slot and hard-capping at PTO2_MAX_FANIN. self_local is this task's own local id
// (the consumer), used to bump each producer's last_consumer_local_id (the
// reclaim gate the host wait_for_consumers polls via completed_watermark).
struct PTO2FaninBuilder {
    PTO2FaninBuilder(PTO2OrchestratorState *orch, PTO2TaskPayload *payload, int32_t self_local, uint32_t seen_epoch) :
        count(0),
        orch(orch),
        seen_epoch(seen_epoch),
        self_local(self_local),
        payload(payload) {}
    int32_t count{0};
    PTO2OrchestratorState *orch{nullptr};
    uint32_t seen_epoch{0};
    int32_t self_local{0};
    PTO2TaskPayload *payload{nullptr};

    bool mark_seen(uint8_t prod_ring, int32_t prod_slot) {
        if (prod_ring >= PTO2_MAX_RING_DEPTH || prod_slot < 0) {
            return false;
        }
        uint32_t *seen = orch->fanin_seen_epoch;
        uint32_t slot = static_cast<uint32_t>(prod_slot);
        if (seen[slot] == seen_epoch) {
            return true;
        }
        seen[slot] = seen_epoch;
        return false;
    }
};

static bool append_fanin_or_fail(
    PTO2OrchestratorState *orch, uint8_t prod_ring, int32_t prod_slot, PTO2TaskSlotState *prod_state,
    PTO2TaskId producer_task_id, PTO2FaninBuilder *fanin_builder
) {
    // Skip a stale/reused producer slot: the cached owner id no longer resolves
    // to this producer (defensive — whole-graph-resident hbg does not reuse slots
    // at build time). A COMPLETED producer IS a real fanin edge under polling (its
    // completion_flags byte is set), so it is not skipped.
    if (prod_state->task == nullptr || prod_state->task->task_id.local() != producer_task_id.local()) {
        return true;
    }
    // Dedup by (ring, slot). Single-ring hbg: prod_ring is always 0.
    if (fanin_builder->mark_seen(prod_ring, prod_slot)) {
        return true;
    }
    if (fanin_builder->count >= PTO2_MAX_FANIN) {
        orch_mark_fatal(orch, PTO2_ERROR_DEP_POOL_OVERFLOW);
        return false;
    }
    fanin_builder->payload->fanin_local_ids[fanin_builder->count++] = static_cast<int32_t>(producer_task_id.local());

    // Reclaim gate: record this task as a consumer of the producer. The producer
    // slot retires once the per-ring completed_watermark reaches this consumer id.
    if (fanin_builder->self_local > prod_state->last_consumer_local_id) {
        prod_state->last_consumer_local_id = fanin_builder->self_local;
    }
    return true;
}

static void scope_tasks_push(PTO2OrchestratorState *orch, PTO2TaskSlotState *task_slot_state);

struct PTO2PreparedTask {
    PTO2TaskId task_id = PTO2TaskId::invalid();
    PTO2TaskAllocResult alloc_result = {-1, 0, nullptr, nullptr};
    PTO2TaskDescriptor *task = nullptr;
    PTO2TaskPayload *payload = nullptr;
    PTO2TaskSlotState *slot_state = nullptr;
};

static PTO2OutputLayout calculate_output_layout(const CoreTaskArgs &args) {
    PTO2OutputLayout layout;
    for (int32_t i = 0; i < args.tensor_count(); i++) {
        if (args.tag(i) != TensorArgType::OUTPUT) {
            continue;
        }
        layout.offsets[i] = layout.total_output_size;
        layout.buffer_sizes[i] =
            PTO2_ALIGN_UP(args.tensor(i).create_info().buffer_size_bytes(), PTO2_PACKED_OUTPUT_ALIGN);
        layout.total_output_size += layout.buffer_sizes[i];
    }
    return layout;
}

static bool check_scope_can_accept_task(PTO2OrchestratorState *orch, PTO2TaskAllocator &allocator, uint8_t ring_id) {
    always_assert(orch->scope_stack_top >= 0 && "Cannot submit task outside a scope");

    int32_t scope_task_count = orch->scope_tasks_size - orch->scope_begins[orch->scope_stack_top];
    if (scope_task_count < allocator.window_size() - 1) {
        return true;
    }

    int32_t active_count = allocator.active_count();

    LOG_ERROR("========================================");
    LOG_ERROR("FATAL: Scope Deadlock Detected! (ring %d)", ring_id);
    LOG_ERROR("========================================");
    LOG_ERROR("Tasks in current scope (%d) >= task_window_size (%d).", scope_task_count, allocator.window_size());
    LOG_ERROR("  scope_depth:        %d", orch->scope_stack_top + 1);
    LOG_ERROR("  ring_id:            %d", ring_id);
    LOG_ERROR("  scope_task_count:   %d", scope_task_count);
    LOG_ERROR("  active_tasks:       %d / %d", active_count, allocator.window_size());
    LOG_ERROR("Root Cause:");
    LOG_ERROR("  host_build_graph is whole-graph-resident: the host builds the entire");
    LOG_ERROR("  scope before the device runs, so no slots reclaim during the build.");
    LOG_ERROR("  When scope task count >= window_size the ring overflows.");
    LOG_ERROR("Solution:");
    LOG_ERROR("  1. Reduce tasks per scope (use batching/unroll)");
    LOG_ERROR("  2. Increase task window (current: %d)", allocator.window_size());
    LOG_ERROR("     Compile-time: PTO2_TASK_WINDOW_SIZE in pto_runtime2_types.h");
    LOG_ERROR("     Runtime env:  PTO2_RING_TASK_WINDOW=<power-of-2>");
    LOG_ERROR("  3. Split work across multiple scopes");
    LOG_ERROR("========================================");
    orch_mark_fatal(orch, PTO2_ERROR_SCOPE_DEADLOCK);
    return false;
}

static bool prepare_task(
    PTO2OrchestratorState *orch, const CoreTaskArgs &args, int32_t total_output_size, ActiveMask active_mask,
    TaskAttrs task_attrs, PTO2PreparedTask *out
) {
    uint8_t ring_id = 0;
    auto &allocator = orch->ring.task_allocator;

    int16_t block_num = args.launch_spec.block_num();
    int32_t active_subtasks_per_block = __builtin_popcount(active_mask.core_mask());
    int32_t total_required_subtasks = static_cast<int32_t>(block_num) * active_subtasks_per_block;
    if (block_num <= 0 || total_required_subtasks > std::numeric_limits<int16_t>::max()) {
        orch->report_fatal(
            PTO2_ERROR_INVALID_ARGS, __FUNCTION__,
            "block_num=%d with %d active slots requires %d subtasks; expected block_num >= 1 and total <= %d",
            block_num, active_subtasks_per_block, total_required_subtasks, std::numeric_limits<int16_t>::max()
        );
        return false;
    }

    if (!check_scope_can_accept_task(orch, allocator, ring_id)) {
        return false;
    }

    out->alloc_result = allocator.alloc(total_output_size);
    if (out->alloc_result.failed()) {
        orch_mark_fatal(orch, PTO2_ERROR_HEAP_RING_DEADLOCK);
        return false;
    }

    out->task_id = PTO2TaskId::make(ring_id, static_cast<uint32_t>(out->alloc_result.task_id));
    out->slot_state = &orch->sm_header->ring.get_slot_state_by_slot(out->alloc_result.slot);
    out->task = &orch->sm_header->ring.task_descriptors[out->alloc_result.slot];
    out->payload = &orch->sm_header->ring.task_payloads[out->alloc_result.slot];

    // Init-on-write: this slot's dynamic scheduling fields and completion flag are
    // initialized here, as the orchestrator claims the slot. whole-graph-resident
    // hbg claims slots [0, total_tasks) exactly once and the device reads no slot
    // past total_tasks, so this claim-time write is the only per-slot SM reset and
    // the unclaimed tail is neither initialized nor read.
    out->slot_state->reset_for_reuse();
    orch->sm_header->ring.completion_flags[out->alloc_result.slot].store(0, std::memory_order_relaxed);

    out->payload->prefetch(args.tensor_count(), args.scalar_count());

    // Re-bind payload/task pointers each submit. Value is per-slot constant
    // (same as &task_payloads[slot] / &task_descriptors[slot]), but writing
    // here lets RingSchedState::init() skip the O(window_size) bind loop.
    // Both writes hit the same 64B slot_state cache line we're about to
    // dirty below, so the extra cost is two stores on an already-hot line.
    // Must precede the Orch-side wiring publish at the end of
    // submit_task_common — that publish is the first read of slot_state->task /
    // slot_state->payload by scheduler threads.
    out->slot_state->bind_buffers(out->payload, out->task);

    // prepare_task does NO payload writes: all payload content (tensors/scalars +
    // early-dispatch fields) is initialized in PTO2TaskPayload::init, the
    // single payload-init point, which runs before Orch-side wiring publish.

    // Fields already zeroed by the reset_for_reuse() above:
    //   wake_list_head=nullptr, next_in_wake_list=nullptr,
    //   any_subtask_deferred=false, completed_subtasks=0, next_block_idx=0
    // Fields immutable after RingSchedState::init():
    //   ring_id
    // task_state is set to PENDING here as the orchestrator populates the slot
    // (host_build_graph does not recycle slots at runtime, so there is no
    // post-CONSUMED reset path).
    out->slot_state->task_state.store(PTO2_TASK_PENDING, std::memory_order_relaxed);
    out->slot_state->total_required_subtasks = static_cast<int16_t>(total_required_subtasks);
    out->slot_state->logical_block_num = block_num;
    out->slot_state->active_mask = active_mask;
    out->slot_state->task_attrs = task_attrs;
    out->slot_state->task_kind = active_mask ? TaskKind::KERNEL : TaskKind::DUMMY;
    // Reclaim gate: seed last_consumer to self, so a producer with no consumers
    // is retirable once completed_watermark >= its own id. Each fanin edge bumps
    // it in append_fanin_or_fail. completion_flags for this slot were cleared
    // above (whole-graph-resident hbg never reuses a slot).
    out->slot_state->last_consumer_local_id = static_cast<int32_t>(out->task_id.local());
    // payload.fanin_count is set in submit_task_common's STEP 6.
    scope_tasks_push(orch, out->slot_state);

    return true;
}

// =============================================================================
// Scope Management
// =============================================================================

static void scope_tasks_push(PTO2OrchestratorState *orch, PTO2TaskSlotState *task_slot_state) {
    if (orch->scope_tasks_size >= orch->scope_tasks_capacity) {
        // scope_tasks lives in the per-Worker arena (single backing allocation),
        // so realloc is not legal. Capacity is the total in-flight slot budget
        // (the runtime task window; see reserve_layout) — hitting it means the
        // ring is saturated, so no further push could succeed regardless of
        // buffer growth.
        orch->report_fatal(
            PTO2_ERROR_SCOPE_TASKS_OVERFLOW, __FUNCTION__,
            "scope_tasks buffer saturated at %d entries (all rings full)", orch->scope_tasks_capacity
        );
        return;
    }
    orch->scope_tasks[orch->scope_tasks_size++] = task_slot_state;
}

void PTO2OrchestratorState::begin_scope(PTO2ScopeMode mode) {
    auto *orch = this;
    if (orch->fatal) {
        return;
    }
    // A Graph replays as a flat DAG with no scope structure: scope boundaries only
    // shape scheduling on the ring, and the shadow-record path submits no ring
    // tasks. Recorded ordering is preserved by the nodes' explicit dependencies
    // and tensor-source classification, so a scope inside a Graph body is a no-op
    // during recording and must not touch the real scope stack.
    if (GraphHostState *state = graph_state_from(orch); state != nullptr && state->recording != nullptr) {
        return;
    }
    assert(orch->scope_stack_top < static_cast<int32_t>(orch->scope_stack_capacity - 1) && "Scope stack overflow");
    if (mode == PTO2ScopeMode::AUTO && orch->in_manual_scope()) {
        report_fatal(PTO2_ERROR_INVALID_ARGS, __FUNCTION__, "auto scope nested inside manual scope is not supported");
        return;
    }

    bool already_in_manual_scope = orch->in_manual_scope();
    ++orch->scope_stack_top;
    orch->scope_begins[orch->scope_stack_top] = orch->scope_tasks_size;
    if (mode == PTO2ScopeMode::MANUAL && !already_in_manual_scope) {
        orch->manual_begin_depth = orch->scope_stack_top;
    }
#if SIMPLER_DFX
    // Gate via is_scope_stats_enabled() (weak-false in host builds) BEFORE the
    // collector call: when disabled we pay nothing. Sample the current ring's
    // task/heap start-end and tensormap usage at the scope boundary.
    if (is_scope_stats_enabled()) {
        uint8_t ring_id = 0;
        auto &alloc = orch->ring.task_allocator;
        // Polling: no dep_pool to report (readiness is via completion_flags).
        int32_t dep_pool_tail = 0;
        int32_t dep_pool_top = 0;
        scope_stats_begin(
            ring_id, alloc.task_tail(), alloc.task_head(), alloc.heap_tail(), alloc.heap_top(), dep_pool_tail,
            dep_pool_top, orch->tensor_map.current_used()
        );
    }
#endif
}

void PTO2OrchestratorState::end_scope() {
    auto *orch = this;
    if (orch->fatal) {
        return;
    }
    // Matches begin_scope: a scope inside a Graph body is a no-op during the
    // shadow-record pass, so it must not touch the scope stack.
    if (GraphHostState *state = graph_state_from(orch); state != nullptr && state->recording != nullptr) {
        return;
    }
    assert(orch->scope_stack_top >= 0 && "Scope stack underflow");

    // Snapshot the ring start/end BEFORE the orchestrator drains pending tasks
    // via scheduler->on_scope_end, so the end record reflects the scope's
    // occupancy at close, not the residual after teardown.
#if SIMPLER_DFX
    // Gate via is_scope_stats_enabled() (see begin_scope). One collector call
    // emits the end-boundary record and tears down bookkeeping.
    if (is_scope_stats_enabled()) {
        uint8_t ring_id = 0;
        auto &alloc = orch->ring.task_allocator;
        // Polling: no dep_pool to report (readiness is via completion_flags).
        int32_t dep_pool_tail = 0;
        int32_t dep_pool_top = 0;
        scope_stats_end(
            ring_id, alloc.task_tail(), alloc.task_head(), alloc.heap_tail(), alloc.heap_top(), dep_pool_tail,
            dep_pool_top, orch->tensor_map.current_used()
        );
    }
#endif

#if SIMPLER_ORCH_PROFILING
    uint64_t _se0 = get_sys_cnt_aicpu();
#endif

    bool ending_manual_scope = orch->scope_stack_top == orch->manual_begin_depth;
    int32_t begin = orch->scope_begins[orch->scope_stack_top--];
    int32_t count = orch->scope_tasks_size - begin;
    if (ending_manual_scope) {
        orch->manual_begin_depth = PTO2_MAX_SCOPE_DEPTH;
    }

    if (orch->scheduler && count > 0) {
        orch->scheduler->on_scope_end(&orch->scope_tasks[begin], count);
    }

    // Rewind the task buffer — these entries are no longer needed
    orch->scope_tasks_size = begin;

#if SIMPLER_ORCH_PROFILING
    uint64_t _se1 = get_sys_cnt_aicpu();
    g_orch_scope_end_cycle += (_se1 - _se0);
#endif
}

// =============================================================================
// Task Submission
// =============================================================================

// Ensure the tensormap entry pool has room for `needed` inserts before STEP 4
// registers this task's outputs. The pool is watermark-reclaimed like the
// task/heap/fanin pools — retired tasks' entries free once last_task_alive
// advances — so an exhausted pool is back-pressure, not a hard error. Reclaim
// against the single ring's watermark; if still short,
// spin until reclaim actually frees entries, with the same 500 ms wall-clock
// backstop as the task allocator and fanin spill pool. A pool that stays full
// (no entry freed) is a genuine deadlock: latch PTO2_ERROR_TENSORMAP_OVERFLOW
// and bail. Returns false on deadlock or on a fatal already latched by another
// party. Cold path — the fast path returns immediately when the pool has room.
static bool ensure_tensormap_capacity(PTO2OrchestratorState *orch, int32_t needed) {
    PTO2TensorMap &tm = orch->tensor_map;
    if (tm.free_entries() >= needed) {
        return true;
    }

    int32_t alive;
    auto read_alive = [&]() {
        // Relaxed: a self-correcting poll re-read every reclaim tick, so a stale
        // watermark only defers reclaim one tick and never over-frees.
        alive = orch->sm_header->ring.fc.last_task_alive.load(std::memory_order_relaxed);
    };

    read_alive();
    int64_t cur_alive_sum = tm.reclaim_retired_all(alive);  // kept for the deadlock diagnostic
    int32_t prev_free = tm.free_entries();
    if (prev_free >= needed) {
        return true;
    }

    int spin_count = 0;
    uint64_t block_cycle0 = 0;  // wall-clock anchor for the deadlock backstop
    bool block_timing = false;  // false until the first no-reclaim-progress tick
    while (tm.free_entries() < needed) {
        spin_count++;

        // Reclaim (and the all-ring watermark reads it needs) is the costly part of
        // this spin and the only path that frees entries; gate it to a periodic tick.
        // Cold path, but the spin itself is tight.
        if ((spin_count & 31) == 0) {
            read_alive();
            cur_alive_sum = tm.reclaim_retired_all(alive);
            int32_t cur_free = tm.free_entries();
            if (cur_free >= needed) {
                return true;
            }
            // Progress is entries actually freed, NOT watermark movement: a ring can
            // retire zero-output tasks (count_registrable_outputs == 0), advancing
            // last_task_alive without freeing any entry. Gating the backstop on
            // free_entries() keeps a wedged pool from dodging the timeout while some
            // unrelated ring keeps draining.
            if (cur_free > prev_free) {
                spin_count = 0;
                prev_free = cur_free;
                block_timing = false;
            }
        }

        if ((spin_count & 1023) == 0) {
            // A fatal latched elsewhere breaks this otherwise-unbounded spin.
            if (orch->sm_header->orch_error_code.load(std::memory_order_acquire) != PTO2_ERROR_NONE) {
                return false;
            }
            // Absolute-time backstop, matching the task allocator: stable across
            // chips/contention, unlike a fixed spin count. get_sys_cnt_aicpu()
            // is an MMIO read, so sample it only once per 1024 spins.
            uint64_t now = get_sys_cnt_aicpu();
            if (!block_timing) {
                block_cycle0 = now;
                block_timing = true;
            } else if (now - block_cycle0 >= PTO2_ALLOC_DEADLOCK_TIMEOUT_CYCLES) {
                LOG_ERROR("========================================");
                LOG_ERROR("FATAL: TensorMap Entry Pool Deadlock Detected!");
                LOG_ERROR("========================================");
                LOG_ERROR("TensorMap entry pool freed no entries for ~500 ms while a task waits.");
                LOG_ERROR("  - Pool used:   %d / %d", tm.current_used(), tm.pool_capacity());
                LOG_ERROR("  - Needed:      %d entries", needed);
                LOG_ERROR("  - last_task_alive: %" PRId64, cur_alive_sum);
                LOG_ERROR("Diagnosis:");
                LOG_ERROR("  No retiring task is freeing tensormap entries (last_task_alive may");
                LOG_ERROR("  still move on rings with no registered outputs). Check TaskRing");
                LOG_ERROR("  diagnostics for the stalled producer.");
                LOG_ERROR("Solution:");
                LOG_ERROR("  Increase PTO2_TENSORMAP_POOL_SIZE (current: %d).", tm.pool_capacity());
                LOG_ERROR("========================================");
                orch_mark_fatal(orch, PTO2_ERROR_TENSORMAP_OVERFLOW);
                return false;
            }
        }
        SPIN_WAIT_HINT();
    }
    return true;
}

// Shared body for submit_task / submit_dummy_task. Caller has already validated
// args.has_error, decided active_mask (empty for dummy), and resolved the per-slot
// kernel_ids (all INVALID_KERNEL_ID for dummy). Performs tensormap sync, fanin
// computation (explicit_deps + auto), output registration, slot init, and
// Orch-side wiring/ready publication.
static TaskOutputTensors submit_task_common(
    PTO2OrchestratorState *orch, const CoreTaskArgs &args, ActiveMask active_mask, TaskAttrs task_attrs,
    int32_t aic_kernel_id, int32_t aiv0_kernel_id, int32_t aiv1_kernel_id
) {
    CYCLE_COUNT_START();
    TaskOutputTensors result;
    PTO2OutputLayout layout = calculate_output_layout(args);
    PTO2PreparedTask prepared;
    if (!prepare_task(orch, args, layout.total_output_size, active_mask, task_attrs, &prepared)) {
        return result;
    }
    PTO2SchedulerState *sched = orch->scheduler;
    PTO2RingFlowControl &fc = orch->sm_header->ring.fc;
    PTO2TaskId task_id = prepared.task_id;
    PTO2TaskDescriptor &task = *prepared.task;
    PTO2TaskPayload &payload = *prepared.payload;
    result.set_task_id(task_id);

    // dep_gen capture point: open this task's graph entry before its dependency
    // steps run, so the edges STEP 1 / STEP 3 discover attach to it. The graph
    // is recorded from the dependency path itself, which makes it the runtime's
    // own answer rather than a reconstruction — the sole source of truth for
    // fanout now that the swimlane hot path no longer records it.
    const bool capture_dep_graph = dep_gen_host_graph_enabled();
    if (capture_dep_graph) {
        const std::array<int32_t, PTO2_SUBTASK_SLOT_COUNT> kernel_ids_capture{
            aic_kernel_id,
            aiv0_kernel_id,
            aiv1_kernel_id,
        };
        dep_gen_host_graph_begin_task(
            task_id.raw, orch->in_manual_scope(), args.allow_early_resolve(), kernel_ids_capture.data(),
            args.launch_spec.block_num(), args.tensor_count(), args.tensor_data(), args.tag_data()
        );
    }

    PTO2FaninBuilder fanin_builder(orch, &payload, static_cast<int32_t>(task_id.local()), next_fanin_seen_epoch(orch));

    CYCLE_COUNT_LAP(g_orch_alloc_cycle);

#if SIMPLER_DFX
    if (layout.total_output_size > 0) {
        orch->buffers_allocated++;
        orch->bytes_allocated += layout.total_output_size;
    }
#endif

    // === STEP 2: Sync TensorMap validity and optional cleanup ===
    // Read current last_task_alive from shared memory for this ring
    int32_t sm_last_task_alive = fc.last_task_alive.load(std::memory_order_acquire);

    orch->tensor_map.sync_tensormap(task_id, sm_last_task_alive);

    CYCLE_COUNT_LAP(g_orch_sync_cycle);

    for (uint32_t i = 0; i < args.explicit_dep_count(); i++) {
        PTO2TaskId dep_task_id = args.explicit_dep(i);
        if (!dep_task_id.is_valid()) {
            orch->report_fatal(
                PTO2_ERROR_INVALID_ARGS, __FUNCTION__, "Arg.set_dependencies(...) requires valid task ids"
            );
            return result;
        }
        // Declared dependencies are graph edges even when the producer already
        // retired below last_task_alive and needs no fanin wiring.
        if (capture_dep_graph) {
            dep_gen_host_graph_add_explicit_edge(dep_task_id.raw);
        }
        uint8_t dep_ring_id = dep_task_id.ring();
        PTO2SharedMemoryRingHeader &dep_ring = orch->sm_header->ring;
        int32_t dep_local_task_id = static_cast<int32_t>(dep_task_id.local());
        int32_t dep_last_task_alive = dep_ring.fc.last_task_alive.load(std::memory_order_acquire);
        if (dep_local_task_id < dep_last_task_alive) {
            continue;
        }
        int32_t dep_slot = dep_ring.get_slot_by_task_id(dep_local_task_id);
        PTO2TaskSlotState *producer_slot_state = &dep_ring.get_slot_state_by_slot(dep_slot);
        if (!append_fanin_or_fail(orch, dep_ring_id, dep_slot, producer_slot_state, dep_task_id, &fanin_builder)) {
            return result;
        }
    }

    // === STEP 3: Lookup inputs (creator retention + tensormap modifier lookup) ===
    DepInputs dep_inputs{
        args.tensor_count(),       args.tensor_data(), args.tag_data(), static_cast<int32_t>(args.explicit_dep_count()),
        args.explicit_deps_data(),
    };

    auto runtime_emit = [&](PTO2TaskId producer_task_id) -> bool {
        uint8_t prod_ring = producer_task_id.ring();
        PTO2SharedMemoryRingHeader &producer_ring = orch->sm_header->ring;
        int32_t prod_slot = producer_ring.get_slot_by_task_id(static_cast<int32_t>(producer_task_id.local()));
        PTO2TaskSlotState *prod_state = &producer_ring.get_slot_state_by_slot(prod_slot);
        return append_fanin_or_fail(orch, prod_ring, prod_slot, prod_state, producer_task_id, &fanin_builder);
    };

    // The capture branch instantiates compute_task_fanin with a live Annotate;
    // the plain branch keeps the un-annotated instantiation the hot path had.
    if (capture_dep_graph) {
        struct DepGraphAnnotate {
            void creator(int32_t arg_idx, const ChipTensor &consumer, PTO2TaskId producer) const {
                dep_gen_host_graph_add_creator_edge(producer.raw, arg_idx, consumer);
            }
            void tensormap(
                int32_t arg_idx, const ChipTensor &consumer, const PTO2TensorMapEntry &entry, OverlapStatus overlap
            ) const {
                dep_gen_host_graph_add_tensormap_edge(entry.producer_task_id.raw, arg_idx, consumer, entry, overlap);
            }
        };
        const bool ok =
            compute_task_fanin(dep_inputs, orch->tensor_map, orch->in_manual_scope(), runtime_emit, DepGraphAnnotate{});
        // STEP 3 is this task's last capture point, so the entry closes here
        // whether or not the fanin computation succeeded.
        dep_gen_host_graph_end_task();
        if (!ok) {
            return result;
        }
    } else {
        if (!compute_task_fanin(dep_inputs, orch->tensor_map, orch->in_manual_scope(), runtime_emit)) {
            return result;
        }
    }

    CYCLE_COUNT_LAP(g_orch_lookup_cycle);

    // === STEP 4: Register outputs/inouts in TensorMap (must be separate from lookup) ===
    // Reserve pool capacity for this task's inserts before registering. The pool
    // is reclaimed as last_task_alive advances; an
    // exhausted pool back-pressures here (and detects a wedged watermark) rather
    // than tripping new_entry()'s hard assert mid-registration.
    int32_t tensormap_needed = count_registrable_outputs(dep_inputs, orch->in_manual_scope());
    if (tensormap_needed > 0 && !ensure_tensormap_capacity(orch, tensormap_needed)) {
        return result;
    }
    register_task_outputs(dep_inputs, task_id, orch->tensor_map, orch->in_manual_scope());

    CYCLE_COUNT_LAP(g_orch_insert_cycle);

    // === STEP 5: Batch-write to GM (single cache line burst) ===
    // Deferred from allocation phase to avoid scattered GM writes that get
    // evicted by TensorMap lookup/insert cache pressure.
    __builtin_prefetch(&task, 1, 1);
    task.task_id = task_id;
    task.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIC)] = aic_kernel_id;
    task.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIV0)] = aiv0_kernel_id;
    task.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIV1)] = aiv1_kernel_id;
    task.packed_buffer_base = prepared.alloc_result.packed_base;
    task.packed_buffer_end = prepared.alloc_result.packed_end;

    // append_fanin_or_fail wrote each producer's local id straight into
    // payload.fanin_local_ids and bumped its last_consumer_local_id; the count is
    // published in STEP 6 below. payload.init does not touch the fanin region.
    payload.init(args, result, prepared.alloc_result, layout);

    // Dispatch predicate: resolve the (tensor, indices) to an absolute GM address
    // now so the scheduler can read it at the dispatch point with a single load,
    // no Arg/ChipTensor access. Both branches write predicate.op explicitly because
    // payload slots are ring-reused; op == NONE means "always dispatch".
    {
        const CoreTaskPredicate &pred = args.predicate();
        if (pred.op != PredicateOp::NONE && pred.operand.tensor != nullptr && pred.operand.tensor->buffer.addr != 0) {
            uint64_t elem_size = get_element_size(pred.operand.tensor->dtype);
            uint64_t flat_offset = pred.operand.tensor->compute_flat_offset(pred.operand.indices, pred.operand.ndims);
            payload.predicate.addr = pred.operand.tensor->buffer.addr + flat_offset * elem_size;
            payload.predicate.target = pred.target;
            payload.predicate.elem_size = static_cast<uint8_t>(elem_size);
            payload.predicate.op = pred.op;
        } else {
            payload.predicate.addr = 0;
            payload.predicate.op = PredicateOp::NONE;
        }
    }
#if SIMPLER_DFX
    if (is_dump_args_enabled()) {
        if (args.scalar_count() > 0) {
            set_dump_args_task_scalar_dtypes(
                task_id.raw, static_cast<uint32_t>(args.scalar_count()), args.scalar_dtypes()
            );
        }
        // Preserve the existing Level-1 task/arg mask whenever dump is enabled.
        // Level 1 uses it to select records; hybrid Level 3 reuses the same mask only
        // to decide which tensors contribute payload alongside full metadata.
        if (args.dump_arg_mask() != 0) {
            set_dump_args_task_mask(task_id.raw, args.dump_arg_mask(), args.dump_arg_index_ambiguous_mask());
        }
    }
#endif

    CYCLE_COUNT_LAP(g_orch_args_cycle);

    // === STEP 6: publish the inline fanin count (device boot classifies) ===
    // Polling + host-orch: append_fanin_or_fail already wrote each producer's
    // local id into payload.fanin_local_ids and bumped its last_consumer_local_id.
    // All that remains is to record how many. There is NO fanout adjacency, NO
    // dep_pool, and NO ready routing here — the device boot scan classifies every
    // task exactly once (fanin_satisfied -> push_ready_routed, else register_wake)
    // before the scheduler dispatch loop starts. Because fanin is now a flat array
    // of position-independent integers, none of this needs host->device pointer
    // relocation.
    payload.fanin_count = fanin_builder.count;
    (void)sched;

    CYCLE_COUNT_LAP(g_orch_fanin_cycle);
    CYCLE_COUNT_ORCH_SUBMIT_RECORD(task_id.raw);

#if SIMPLER_DFX
    orch->tasks_submitted++;
#if SIMPLER_ORCH_PROFILING
    g_orch_submit_count++;
#endif
    g_orch_submit_idx++;
#endif
    return result;
}

namespace {

bool graph_boundary_matches(const GraphDefinition &definition, const CoreTaskArgs &args) {
    if (args.scalar_count() != 0 || args.explicit_dep_count() != 0 ||
        args.tensor_count() != static_cast<int32_t>(definition.boundary_count)) {
        LOG_WARN(
            "[GraphExecution] fixed boundary contract mismatch: tensors=%d/%u scalars=%d explicit_deps=%u",
            args.tensor_count(), definition.boundary_count, args.scalar_count(), args.explicit_dep_count()
        );
        return false;
    }
    const auto *signatures = graph_definition_array<GraphBoundarySignature>(
        definition, definition.off_boundary_signatures, definition.boundary_count
    );
    if (signatures == nullptr) return false;

    bool alias_mismatch = false;
    for (int32_t i = 0; i < args.tensor_count(); ++i) {
        const ChipTensor &tensor = args.tensor(i).ref();
        const GraphBoundarySignature &signature = signatures[i];
        if (tensor.ndims > MAX_TENSOR_DIMS) {
            debug_assert(tensor.ndims <= MAX_TENSOR_DIMS && "Graph boundary ChipTensor rank is not supported");
            LOG_WARN("[GraphExecution] ChipTensor rank %u exceeds the fixed Graph boundary limit", tensor.ndims);
            return false;
        }
        const auto shape_end = std::begin(tensor.shapes) + tensor.ndims;
        const auto stride_end = std::begin(tensor.strides) + tensor.ndims;
        const bool metadata_match = tensor.buffer.size == signature.buffer_size && tensor.ndims == signature.ndims &&
                                    static_cast<uint8_t>(tensor.dtype) == signature.dtype &&
                                    static_cast<uint8_t>(args.tag(i)) == signature.tag &&
                                    static_cast<uint8_t>(tensor.manual_dep ? 1 : 0) == signature.manual_dep &&
                                    static_cast<uint8_t>(tensor.is_contiguous ? 1 : 0) == signature.is_contiguous &&
                                    std::equal(std::begin(tensor.shapes), shape_end, std::begin(signature.shapes)) &&
                                    std::equal(std::begin(tensor.strides), stride_end, std::begin(signature.strides));
        if (!metadata_match) {
            debug_assert(metadata_match && "Variable Graph boundary tensor shape/metadata is not supported");
            LOG_WARN(
                "[GraphExecution] fixed tensor shape/metadata mismatch at boundary arg %d; using ordinary path", i
            );
            return false;
        }
        uint16_t alias_rep = static_cast<uint16_t>(i);
        for (int32_t j = 0; j < i; ++j) {
            const ChipTensor &other = args.tensor(j).ref();
            if (other.buffer.addr == tensor.buffer.addr && other.buffer.size == tensor.buffer.size) {
                alias_rep = static_cast<uint16_t>(j);
                break;
            }
        }
        alias_mismatch |= alias_rep != signature.alias_rep;
    }
    if (alias_mismatch) {
        debug_assert(!alias_mismatch && "Changing the Graph boundary alias partition is not supported");
        LOG_WARN("%s", "[GraphExecution] boundary alias partition differs from recording; using ordinary path");
        return false;
    }
    return true;
}

void graph_reset_outer_payload(PTO2TaskPayload &payload) {
    payload.tensor_count = 0;
    payload.scalar_count = 0;
    payload.fanin_count = 0;
    payload.predicate = DispatchPredicate{};
    payload.early_dispatch_state.store(PTO2_EARLY_DISPATCH_NONE, std::memory_order_relaxed);
    for (auto &word : payload.staged_core_mask)
        word.store(0, std::memory_order_relaxed);
    payload.dispatch_fanin.store(0, std::memory_order_relaxed);
    payload.dispatch_propagated.store(0, std::memory_order_relaxed);
    payload.published_block_count.store(0, std::memory_order_relaxed);
    payload.early_dispatch_launch_state.store(PTO2_EARLY_DISPATCH_LAUNCH_NONE, std::memory_order_relaxed);
    payload.running_slot_count.store(0, std::memory_order_relaxed);
    payload.early_sync_drain_state.store(PTO2_EARLY_SYNC_DRAIN_NONE, std::memory_order_relaxed);
}

bool graph_build_submission_image(
    const std::vector<std::byte> &definition_image, const CoreTaskArgs &args, std::vector<std::byte> *submission_image
) {
    if (submission_image == nullptr || graph_definition(definition_image) == nullptr) return false;
    const size_t definition_offset = PTO2_ALIGN_UP(sizeof(GraphSubmission), alignof(GraphDefinition));
    const size_t tensors_offset = PTO2_ALIGN_UP(definition_offset + definition_image.size(), alignof(GraphTensor));
    const size_t tensor_bytes = static_cast<size_t>(args.tensor_count()) * sizeof(GraphTensor);
    if (definition_offset > UINT32_MAX || tensors_offset > UINT32_MAX || tensors_offset > UINT32_MAX - tensor_bytes) {
        return false;
    }
    submission_image->assign(tensors_offset + tensor_bytes, std::byte{0});
    std::memcpy(submission_image->data() + definition_offset, definition_image.data(), definition_image.size());
    auto *tensors = reinterpret_cast<GraphTensor *>(submission_image->data() + tensors_offset);
    for (int32_t i = 0; i < args.tensor_count(); ++i)
        tensors[i] = graph_tensor_pack(args.tensor(i).ref());

    const GraphDefinition &definition = *graph_definition(definition_image);
    GraphSubmission submission{};
    submission.graph_key = definition.full_key;
    submission.total_bytes = static_cast<uint32_t>(submission_image->size());
    submission.definition_offset = static_cast<uint32_t>(definition_offset);
    submission.tensors_offset = static_cast<uint32_t>(tensors_offset);
    submission.tensor_count = static_cast<uint32_t>(args.tensor_count());
    std::memcpy(submission_image->data(), &submission, sizeof(submission));
    return true;
}

bool graph_submit_definition(
    PTO2OrchestratorState *orch, GraphHostState *state, const std::vector<std::byte> &definition_image,
    const CoreTaskArgs &args, PTO2TaskId *submitted_id
) {
    const GraphDefinition *definition = graph_definition(definition_image);
    if (definition == nullptr || !graph_boundary_matches(*definition, args) ||
        definition->required_heap > static_cast<uint64_t>(INT32_MAX)) {
        return false;
    }
    auto &allocator = orch->ring.task_allocator;
    if (allocator.active_count() + 1 >= allocator.window_size() ||
        definition->required_heap > allocator.heap_available()) {
        LOG_WARN("%s", "[GraphExecution] task-window/heap preflight failed; using ordinary path");
        return false;
    }

    GraphPendingUpload pending;
    if (!graph_build_submission_image(definition_image, args, &pending.image)) return false;

    DepInputs boundary_inputs{
        args.tensor_count(), args.tensor_data(), args.tag_data(), 0, nullptr,
    };
    const int32_t tensormap_needed = count_registrable_outputs(boundary_inputs, orch->in_manual_scope());
    if (tensormap_needed > 0 && !ensure_tensormap_capacity(orch, tensormap_needed)) return false;
    if (!check_scope_can_accept_task(orch, allocator, 0)) return false;

    const PTO2TaskAllocResult allocation = allocator.alloc(static_cast<int32_t>(definition->required_heap));
    if (allocation.failed()) {
        orch_mark_fatal(orch, PTO2_ERROR_HEAP_RING_DEADLOCK);
        return false;
    }
    const PTO2TaskId task_id = PTO2TaskId::make(0, static_cast<uint32_t>(allocation.task_id));
    PTO2SharedMemoryRingHeader &ring = orch->sm_header->ring;
    PTO2TaskDescriptor &task = ring.task_descriptors[allocation.slot];
    PTO2TaskPayload &payload = ring.task_payloads[allocation.slot];
    PTO2TaskSlotState &slot = ring.get_slot_state_by_slot(allocation.slot);

    slot.bind_buffers(&payload, &task);
    slot.task_state.store(PTO2_TASK_PENDING, std::memory_order_relaxed);
    slot.last_consumer_local_id = static_cast<int32_t>(task_id.local());
    slot.active_mask = ActiveMask{};
    slot.task_attrs = TaskAttrs{};
    slot.total_required_subtasks = 0;
    slot.logical_block_num = 1;
    slot.task_kind = TaskKind::GRAPH;
    slot.graph_context = nullptr;
    scope_tasks_push(orch, &slot);

    task.task_id = task_id;
    std::fill(std::begin(task.kernel_id), std::end(task.kernel_id), INVALID_KERNEL_ID);
    task.packed_buffer_base = allocation.packed_base;
    task.packed_buffer_end = allocation.packed_end;
    graph_reset_outer_payload(payload);

    PTO2FaninBuilder fanin_builder(orch, &payload, static_cast<int32_t>(task_id.local()), next_fanin_seen_epoch(orch));
    orch->tensor_map.sync_tensormap(task_id, ring.fc.last_task_alive.load(std::memory_order_acquire));
    auto emit = [&](PTO2TaskId producer_id) -> bool {
        const int32_t producer_local = static_cast<int32_t>(producer_id.local());
        const int32_t producer_slot = ring.get_slot_by_task_id(producer_local);
        PTO2TaskSlotState *producer = &ring.get_slot_state_by_slot(producer_slot);
        return append_fanin_or_fail(orch, producer_id.ring(), producer_slot, producer, producer_id, &fanin_builder);
    };
    if (!compute_task_fanin(boundary_inputs, orch->tensor_map, orch->in_manual_scope(), emit)) return false;
    register_task_outputs(boundary_inputs, task_id, orch->tensor_map, orch->in_manual_scope());
    payload.fanin_count = fanin_builder.count;

    pending.outer_slot = &slot;
    state->pending_uploads.push_back(std::move(pending));
    if (submitted_id != nullptr) *submitted_id = task_id;
#if SIMPLER_DFX
    orch->tasks_submitted++;
#endif
    return true;
}

// Record one internal Graph node during the recording pass without consuming a
// ring task-window slot. Builds the node's metadata and materialized outputs
// exactly as submit_task_common would, but reserves output buffers from heap
// scratch (released in graph_end) and derives internal fanins from tensor-source
// classification — so no ring slot, tensormap entry, fanin-pool entry, or upload
// is produced for the node. The whole sub-DAG is later replayed by the single
// outer GRAPH task graph_end emits. The returned TaskOutputTensors borrow the
// node's own tensor storage; moving the node into recording.nodes keeps those
// addresses valid because the inner buffer is transferred, not copied.
TaskOutputTensors graph_record_submit_node(
    PTO2OrchestratorState *orch, const CoreTaskArgs &args, ActiveMask active_mask, TaskAttrs task_attrs,
    int32_t aic_kernel_id, int32_t aiv0_kernel_id, int32_t aiv1_kernel_id
) {
    TaskOutputTensors result;
    GraphRecording &recording = *graph_state_from(orch)->recording;

    const size_t node_index = recording.nodes.size();
    // A recorded node's index equals its local task id minus the recording
    // baseline, so its synthetic id keeps classification and explicit-dep
    // arithmetic identical to the ordinary path.
    const PTO2TaskId task_id =
        PTO2TaskId::make(0, static_cast<uint32_t>(recording.start_local_task_id) + static_cast<uint32_t>(node_index));
    result.set_task_id(task_id);

    if (node_index >= GRAPH_MAX_NODES || args.has_error || args.predicate().op != PredicateOp::NONE) {
        recording.unsupported = true;
    }

    const PTO2OutputLayout layout = calculate_output_layout(args);
    void *packed_base = orch->ring.task_allocator.reserve_heap_scratch(layout.total_output_size);
    if (layout.total_output_size > 0 && packed_base == nullptr) {
        // No scratch storage for this node's outputs: the sub-DAG cannot be
        // recorded, so graph_end falls back and the body runs on the ordinary
        // path where the same heap pressure is reported through alloc().
        recording.unsupported = true;
        return result;
    }
    const uint64_t aligned_output =
        layout.total_output_size > 0 ? PTO2_ALIGN_UP(static_cast<uint64_t>(layout.total_output_size), PTO2_ALIGN_SIZE) :
                                       0;

    GraphRecordedNode node;
    node.kernel_ids[static_cast<int>(PTO2SubtaskSlot::AIC)] = aic_kernel_id;
    node.kernel_ids[static_cast<int>(PTO2SubtaskSlot::AIV0)] = aiv0_kernel_id;
    node.kernel_ids[static_cast<int>(PTO2SubtaskSlot::AIV1)] = aiv1_kernel_id;
    node.active_mask = active_mask;
    node.task_attrs = task_attrs;
    node.task_attrs.set_early_resolve(false);
    node.logical_block_num = args.launch_spec.block_num();
    node.total_required_subtasks =
        static_cast<int16_t>(node.logical_block_num * __builtin_popcount(active_mask.core_mask()));
    node.record_packed_base = reinterpret_cast<uintptr_t>(packed_base);
    node.total_output_size = aligned_output;

    // Build the tensor list exactly as PTO2TaskPayload::init: inputs/inouts copy
    // the caller's ChipTensor; outputs materialize from the create-info onto the
    // scratch buffer and carry this node's owner id.
    const int32_t tensor_count = args.tensor_count();
    node.tensors.resize(static_cast<size_t>(tensor_count));
    for (int32_t i = 0; i < tensor_count; ++i) {
        ChipTensor &slot_tensor = node.tensors[static_cast<size_t>(i)];
        if (args.tag(i) != TensorArgType::OUTPUT) {
            slot_tensor.copy(args.tensor(i).ref());
        } else {
            init_tensor_from_create_info(
                slot_tensor, args.tensor(i).create_info(),
                reinterpret_cast<void *>(reinterpret_cast<char *>(packed_base) + layout.offsets[i]),
                layout.buffer_sizes[i]
            );
            slot_tensor.owner_task_id = task_id;
        }
    }
    // Materialize output refs only after node.tensors is fully sized so the
    // borrowed addresses stay stable across the move into recording.nodes.
    for (int32_t i = 0; i < tensor_count; ++i) {
        if (args.tag(i) == TensorArgType::OUTPUT) result.materialize_output(node.tensors[static_cast<size_t>(i)]);
    }
    node.scalars.assign(args.scalars(), args.scalars() + args.scalar_count());

    // Classify each tensor's source, then derive internal fanins from the
    // INTERNAL classifications plus any explicit internal dependency.
    node.tensor_sources.resize(static_cast<size_t>(tensor_count));
    for (int32_t i = 0; i < tensor_count; ++i) {
        if (!graph_classify_tensor(
                recording, node, static_cast<int32_t>(node_index), node.tensors[static_cast<size_t>(i)],
                &node.tensor_sources[static_cast<size_t>(i)]
            )) {
            recording.unsupported = true;
        }
    }
    auto add_fanin = [&node](size_t producer) {
        if (std::find(node.internal_fanins.begin(), node.internal_fanins.end(), producer) ==
            node.internal_fanins.end()) {
            node.internal_fanins.push_back(producer);
        }
    };
    for (const GraphRecordedTensorSourceRef &source : node.tensor_sources) {
        if (source.source == GraphRecordedTensorSource::INTERNAL) add_fanin(source.source_index);
    }
    for (uint32_t i = 0; i < args.explicit_dep_count(); ++i) {
        const PTO2TaskId dep = args.explicit_dep(i);
        const int32_t dep_index = static_cast<int32_t>(dep.local()) - recording.start_local_task_id;
        if (!dep.is_valid() || dep.ring() != 0 || dep_index >= static_cast<int32_t>(node_index)) {
            recording.unsupported = true;
            continue;
        }
        if (dep_index < 0) {
            const bool represented_by_boundary = std::any_of(
                recording.boundary_tensors.begin(), recording.boundary_tensors.end(), [dep](const ChipTensor &tensor) {
                    return tensor.owner_task_id == dep;
                }
            );
            if (!represented_by_boundary) recording.unsupported = true;
        } else {
            add_fanin(static_cast<size_t>(dep_index));
        }
    }

    recording.nodes.push_back(std::move(node));
    return result;
}

}  // namespace

GraphScopeResult
PTO2OrchestratorState::graph_begin(uint64_t graph_key, const CoreTaskArgs &args, uint64_t callable_hash) {
    auto *orch = this;
    GraphScopeResult result;
    GraphHostState *state = graph_state_from(orch);
    if (state == nullptr || !rt_graph_args_cacheable(args) || args.explicit_dep_count() != 0) {
        debug_assert(args.scalar_count() == 0 && "Graph execution scalars are not supported in step 1");
        debug_assert(args.explicit_dep_count() == 0 && "Graph boundary explicit dependencies are not supported");
        return result;
    }
    if (state->recording != nullptr) {
        state->recording->unsupported = true;
        debug_assert(state->recording == nullptr && "Nested Graph recording is not supported");
        LOG_WARN("%s", "[GraphExecution] nested Graph recording is not supported");
        return result;
    }

    const uint64_t full_key = graph_full_key(callable_hash, graph_key);
    auto definition_it = state->definitions.find(full_key);
    if (definition_it != state->definitions.end()) {
        PTO2TaskId submitted = PTO2TaskId::invalid();
        if (graph_submit_definition(orch, state, definition_it->second, args, &submitted)) {
            result.execute_block = false;
            result.task_id = submitted;
#if SIMPLER_DFX
            g_orch_submit_idx++;
#if SIMPLER_ORCH_PROFILING
            g_orch_submit_count++;
#endif
#endif
        }
        return result;
    }
    if (state->definitions.size() >= GRAPH_MAX_DEFINITIONS) {
        debug_assert(
            state->definitions.size() < GRAPH_MAX_DEFINITIONS &&
            "Graph Definition cache exceeds the supported per-worker limit"
        );
        LOG_WARN(
            "[GraphExecution] Definition cache is full (%zu entries); using ordinary path", state->definitions.size()
        );
        return result;
    }

    auto recording = std::make_unique<GraphRecording>();
    recording->full_key = full_key;
    recording->start_local_task_id = orch->ring.task_allocator.active_count();
    recording->heap_watermark = orch->ring.task_allocator.heap_top();
    recording->boundary_args = &args;
    recording->boundary_tensors.reserve(static_cast<size_t>(args.tensor_count()));
    recording->boundary_types.reserve(static_cast<size_t>(args.tensor_count()));
    for (int32_t i = 0; i < args.tensor_count(); ++i) {
        recording->boundary_tensors.push_back(args.tensor(i).ref());
        recording->boundary_types.push_back(args.tag(i));
    }
    state->recording = std::move(recording);
    // The body runs through the shadow-record path, not the ring, so it does not
    // execute as ordinary tasks.
    result.execute_block = false;
    result.recording = true;
    return result;
}

// Finish the recording pass. Returns true when the sub-DAG was compacted into a
// Definition and emitted as a single outer GRAPH task; false when the recording
// hit an unsupported construct or the outer task could not be placed, in which
// case the caller re-runs the body on the ordinary path so its work is still
// submitted.
bool PTO2OrchestratorState::graph_end() {
    GraphHostState *state = graph_state_from(this);
    if (state == nullptr || state->recording == nullptr) return false;
    std::unique_ptr<GraphRecording> recording = std::move(state->recording);
    // Release the scratch output buffers the internal nodes reserved; the outer
    // GRAPH task reclaims the same heap region as one block below (or the
    // ordinary fallback re-uses it per task).
    this->ring.task_allocator.restore_heap_top(recording->heap_watermark);

    std::vector<std::byte> definition;
    if (!graph_build_definition(*recording, &definition)) {
        debug_assert(false && "The recorded Graph contains a construct that Graph Execution does not support");
        LOG_WARN("%s", "[GraphExecution] unsupported construct observed; falling back to the ordinary path");
        return false;
    }
    const GraphDefinition *header = graph_definition(definition);
    if (header == nullptr) return false;
    LOG_DEBUG(
        "[GraphExecution] define key=0x%llx nodes=%u bytes=%u", static_cast<unsigned long long>(header->full_key),
        header->task_count, header->total_bytes
    );
    auto inserted = state->definitions.emplace(header->full_key, std::move(definition));
    const std::vector<std::byte> &cached = inserted.first->second;

    // Emit the single outer GRAPH task for this first invocation exactly as a
    // cache hit would, so the recording run occupies one ring slot and one heap
    // block instead of one per internal node. The Definition stays cached for
    // subsequent invocations even if this placement fails.
    PTO2TaskId submitted = PTO2TaskId::invalid();
    if (recording->boundary_args == nullptr ||
        !graph_submit_definition(this, state, cached, *recording->boundary_args, &submitted)) {
        return false;
    }
#if SIMPLER_DFX
    g_orch_submit_idx++;
#if SIMPLER_ORCH_PROFILING
    g_orch_submit_count++;
#endif
#endif
    return true;
}

void PTO2OrchestratorState::graph_commit() {}

TaskOutputTensors PTO2OrchestratorState::submit_task(const MixedKernels &mixed_kernels, const CoreTaskArgs &args) {
    auto *orch = this;

    // Orchestration API should short-circuit after fatal, but keep this entry
    // robust as a no-op in case a caller reaches it directly.
    if (orch->fatal) {
        return TaskOutputTensors{};
    }

    // Validate Arg construction (errors recorded by add_input/add_output/etc.)
    if (args.has_error) {
        LOG_ERROR("========================================");
        LOG_ERROR("FATAL: Invalid Arg Detected!");
        LOG_ERROR("========================================");
        LOG_ERROR("Error: %s", args.error_msg ? args.error_msg : "(unknown)");
        LOG_ERROR("  tensor_count: %d, scalar_count: %d", args.tensor_count(), args.scalar_count());
        LOG_ERROR("This is a bug in the orchestration code.");
        LOG_ERROR("========================================");
        orch_mark_fatal(orch, PTO2_ERROR_INVALID_ARGS);
        return TaskOutputTensors{};
    }
    always_assert(orch->scheduler != nullptr);
    // === Validate submit inputs ===
    ActiveMask active_mask = mixed_kernels.to_active_mask();
    if (!static_cast<bool>(active_mask)) {
        report_fatal(
            PTO2_ERROR_INVALID_ARGS, __FUNCTION__,
            "MixedKernels names no active slot; set at least one of aic/aiv0/aiv1 kernel_id"
        );
        return TaskOutputTensors{};
    }

    int16_t block_num = args.launch_spec.block_num();

    // Normalize single-AIV tasks: if only aiv1 is set (no aic, no aiv0), move
    // it to the aiv0 slot.  This guarantees the dispatch path can always use
    // PTO2SubtaskSlot::AIV0 for single-AIV shapes without inspecting active_mask.
    // Mixed tasks (AIC+AIV) keep their original AIV identity so the correct
    // hardware channel (AIV0→AIC vs AIV1→AIC) is used at dispatch time.
    MixedKernels normalized = mixed_kernels;
    bool has_aic = active_mask.has_mask(PTO2_SUBTASK_MASK_AIC);
    bool has_aiv0 = active_mask.has_mask(PTO2_SUBTASK_MASK_AIV0);
    bool has_aiv1 = active_mask.has_mask(PTO2_SUBTASK_MASK_AIV1);
    if (!has_aic && has_aiv1 && !has_aiv0) {
        normalized.aiv0_kernel_id = normalized.aiv1_kernel_id;
        normalized.aiv1_kernel_id = INVALID_KERNEL_ID;
        active_mask = normalized.to_active_mask();
    }

    TaskAttrs task_attrs;
    task_attrs.set_early_resolve(args.allow_early_resolve());
    task_attrs.set_timing_slot(args.task_timing_slot());

    // sync_start is only meaningful for tasks with block_num > 1.
    if (block_num > 1 && args.launch_spec.require_sync_start()) {
        // Deadlock check: block_num >= total available slots of the required type.
        // For MIX/AIC: limit is total_cluster_count (one AIC per cluster).
        // For AIV:     limit is total_aiv_count.
        PTO2ResourceShape shape = active_mask.to_shape();
        int32_t limit = (shape == PTO2ResourceShape::AIV) ? orch->total_aiv_count : orch->total_cluster_count;
        if (limit > 0 && block_num > limit) {
            report_fatal(
                PTO2_ERROR_REQUIRE_SYNC_START_INVALID, __FUNCTION__,
                "require_sync_start block_num=%d > limit=%d (deadlock guaranteed)", block_num, limit
            );
            return TaskOutputTensors{};
        }
        task_attrs.set_sync_start();
    }

    if (args.predicate().op != PredicateOp::NONE) {
        task_attrs.set_predicate();
    }

    if (GraphHostState *state = graph_state_from(orch); state != nullptr && state->recording != nullptr) {
        return graph_record_submit_node(
            orch, args, active_mask, task_attrs, normalized.aic_kernel_id, normalized.aiv0_kernel_id,
            normalized.aiv1_kernel_id
        );
    }

    return submit_task_common(
        orch, args, active_mask, task_attrs, normalized.aic_kernel_id, normalized.aiv0_kernel_id,
        normalized.aiv1_kernel_id
    );
}

// Submit a dependency-only task: full dependency graph participation
// (tensormap lookup/insert, explicit_deps, manual_dep, manual_scope) but no
// AICore dispatch. Empty active_mask routes the slot to the DUMMY ready
// bucket; dispatch loop short-circuits to completion. Accepts the same Arg
// shape as submit_task; scalars are permitted but never consumed.
TaskOutputTensors PTO2OrchestratorState::submit_dummy_task(const CoreTaskArgs &args) {
    auto *orch = this;

    if (orch->fatal) {
        return TaskOutputTensors{};
    }

    if (args.has_error) {
        LOG_ERROR("========================================");
        LOG_ERROR("FATAL: Invalid Arg in submit_dummy_task!");
        LOG_ERROR("========================================");
        LOG_ERROR("Error: %s", args.error_msg ? args.error_msg : "(unknown)");
        LOG_ERROR("  tensor_count: %d, scalar_count: %d", args.tensor_count(), args.scalar_count());
        LOG_ERROR("========================================");
        orch_mark_fatal(orch, PTO2_ERROR_INVALID_ARGS);
        return TaskOutputTensors{};
    }
    always_assert(orch->scheduler != nullptr);

    // Dummy tasks never dispatch to an AICore, so sync_start / has_predicate do
    // not apply; only the early-dispatch hint and timing tag carry over.
    TaskAttrs task_attrs;
    task_attrs.set_early_resolve(args.allow_early_resolve());
    task_attrs.set_timing_slot(args.task_timing_slot());

    if (GraphHostState *state = graph_state_from(orch); state != nullptr && state->recording != nullptr) {
        return graph_record_submit_node(
            orch, args, ActiveMask{}, task_attrs, INVALID_KERNEL_ID, INVALID_KERNEL_ID, INVALID_KERNEL_ID
        );
    }

    return submit_task_common(
        orch, args, ActiveMask{}, task_attrs, INVALID_KERNEL_ID, INVALID_KERNEL_ID, INVALID_KERNEL_ID
    );
}

TaskOutputTensors PTO2OrchestratorState::alloc_tensors(const CoreTaskArgs &args) {
    auto *orch = this;
    // Orchestration API should short-circuit after fatal, but keep this entry
    // robust as a no-op in case a caller reaches it directly.
    if (orch->fatal) {
        return TaskOutputTensors{};
    }

    if (args.tensor_count() <= 0) {
        report_fatal(PTO2_ERROR_INVALID_ARGS, __FUNCTION__, "alloc_tensors requires at least one TensorCreateInfo");
        return TaskOutputTensors{};
    }
    if (args.scalar_count() != 0) {
        report_fatal(PTO2_ERROR_INVALID_ARGS, __FUNCTION__, "alloc_tensors only accepts output TensorCreateInfo args");
        return TaskOutputTensors{};
    }
    for (int32_t i = 0; i < args.tensor_count(); i++) {
        if (args.tag(i) != TensorArgType::OUTPUT) {
            report_fatal(
                PTO2_ERROR_INVALID_ARGS, __FUNCTION__, "alloc_tensors only accepts output TensorCreateInfo args"
            );
            return TaskOutputTensors{};
        }
    }

    CYCLE_COUNT_START();

    if (args.has_error) {
        report_fatal(
            PTO2_ERROR_INVALID_ARGS, __FUNCTION__, "%s",
            args.error_msg ? args.error_msg : "alloc_tensors failed to construct output-only Arg"
        );
        return TaskOutputTensors{};
    }

    // Runtime-allocated outputs cannot be replayed by a Graph, so a Graph body
    // that calls alloc_tensors is unsupported. Still materialize the outputs so
    // the recording body chains correctly, then poison the recording; graph_end
    // falls back and the body re-runs on the ordinary path.
    if (GraphHostState *state = graph_state_from(orch); state != nullptr && state->recording != nullptr) {
        TaskOutputTensors result = graph_record_submit_node(
            orch, args, ActiveMask{}, TaskAttrs{}, INVALID_KERNEL_ID, INVALID_KERNEL_ID, INVALID_KERNEL_ID
        );
        graph_record_mark_unsupported(orch);
        return result;
    }

    PTO2OutputLayout layout = calculate_output_layout(args);
    PTO2PreparedTask prepared;
    // Kernel-less alloc task: no active subtasks, no dispatch-time attributes. The
    // early-dispatch hint is force-set below (see the flag-the-creator note).
    if (!prepare_task(orch, args, layout.total_output_size, ActiveMask{}, TaskAttrs{}, &prepared)) {
        return TaskOutputTensors{};
    }

    PTO2TaskDescriptor &task = *prepared.task;
    PTO2TaskPayload &payload = *prepared.payload;

    CYCLE_COUNT_LAP(g_orch_alloc_cycle);

#if SIMPLER_DFX
    if (layout.total_output_size > 0) {
        orch->buffers_allocated++;
        orch->bytes_allocated += layout.total_output_size;
    }
#endif

    task.task_id = prepared.task_id;
    task.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIC)] = INVALID_KERNEL_ID;
    task.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIV0)] = INVALID_KERNEL_ID;
    task.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIV1)] = INVALID_KERNEL_ID;
    task.packed_buffer_base = prepared.alloc_result.packed_base;
    task.packed_buffer_end = prepared.alloc_result.packed_end;

    TaskOutputTensors outputs;
    outputs.set_task_id(prepared.task_id);
    payload.init(args, outputs, prepared.alloc_result, layout);
    payload.fanin_count = 0;  // hidden-alloc tasks have no producer dependencies
    CYCLE_COUNT_LAP(g_orch_args_cycle);

    if (prepared.slot_state != nullptr) {
        // Hidden alloc tasks complete inline in the orchestrator before any
        // consumer can exist, so they have no fanout to notify and no worker
        // subtasks to retire. Running the full on_task_complete path
        // would only pay unnecessary fanout_lock / traversal overhead here.
        // The generic slot initialization done in prepare_task() is still
        // required so scope_end can release the producer-side reference and
        // drive the slot to CONSUMED, but worker dispatch fields are never
        // observed for hidden alloc tasks.
        //
        // Flag the creator so it does NOT suppress its consumers' early-dispatch.
        // Under the direct-only model an unflagged producer disqualifies its
        // consumer, and a pre-completed producer only seeds dispatch_fanin when
        // flagged. A buffer allocation is pure memory whose output is ready at
        // creation — it should always be transparent, never a barrier. Unlike a
        // codegen task there is no Arg-driven hint to honor here, so mark it
        // unconditionally.
        prepared.slot_state->task_attrs.set_early_resolve(true);
        prepared.slot_state->mark_completed();  // host-visible task_state mirror
        // Polling: pre-set the device-visible completion_flags byte in the H2D
        // image. Consumers poll completion_flags (not task_state), so a hidden-alloc
        // producer completed here on the host must publish its flag too — otherwise
        // every consumer register_wakes on a producer that never runs on device and
        // the run hangs. (The device watermark walk transparently steps past this
        // pre-set flag when a later on-device task completes.)
        PTO2SharedMemoryRingHeader &done_ring = orch->sm_header->ring;
        int32_t done_local = static_cast<int32_t>(prepared.task_id.local());
        done_ring.set_completion_flag(done_local);
    }
    orch->inline_completed_tasks++;

    CYCLE_COUNT_LAP(g_orch_fanin_cycle);
    CYCLE_COUNT_ORCH_SUBMIT_RECORD(prepared.task_id.raw);

#if SIMPLER_DFX
    orch->tasks_submitted++;
#if SIMPLER_ORCH_PROFILING
    g_orch_submit_count++;
#endif
    g_orch_submit_idx++;
#endif

    return outputs;
}

// =============================================================================
// Flow Control
// =============================================================================

void PTO2OrchestratorState::mark_done() {
    auto *orch = this;
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        int32_t total_tasks = orch->ring.task_allocator.active_count();
        if (total_tasks > 0) {
            LOG_DEBUG("=== [Orchestrator] ring %d: total_tasks=%d ===", r, total_tasks);
        }
    }
    orch->sm_header->orchestrator_done.store(1, std::memory_order_release);
    orch->scope_tasks_size = 0;
    orch->scope_stack_top = -1;
    orch->manual_begin_depth = PTO2_MAX_SCOPE_DEPTH;
#if !SIMPLER_ORCH_PROFILING && SIMPLER_DFX
    g_orch_submit_idx = 0;
#endif
}

#if SIMPLER_ORCH_PROFILING
PTO2OrchProfilingData orchestrator_get_profiling() {
    PTO2OrchProfilingData d;
    d.sync_cycle = g_orch_sync_cycle;
    d.alloc_cycle = g_orch_alloc_cycle;
    d.args_cycle = g_orch_args_cycle;
    d.lookup_cycle = g_orch_lookup_cycle;
    d.insert_cycle = g_orch_insert_cycle;
    d.fanin_cycle = g_orch_fanin_cycle;
    d.scope_end_cycle = g_orch_scope_end_cycle;
    d.submit_count = g_orch_submit_count;
    d.alloc_wait_cycle = g_orch_alloc_wait_cycle;
    d.fanin_wait_cycle = g_orch_fanin_wait_cycle;
    d.alloc_atomic_count = g_orch_alloc_atomic_count;
    d.args_atomic_count = g_orch_args_atomic_count;
    d.scope_end_atomic_count = g_orch_scope_end_atomic_count;

    // Reset
    g_orch_sync_cycle = g_orch_alloc_cycle = g_orch_args_cycle = 0;
    g_orch_lookup_cycle = g_orch_insert_cycle = 0;
    g_orch_fanin_cycle = g_orch_scope_end_cycle = 0;
    g_orch_submit_count = 0;
    g_orch_submit_idx = 0;
    g_orch_alloc_wait_cycle = 0;
    g_orch_fanin_wait_cycle = 0;
    g_orch_alloc_atomic_count = 0;
    g_orch_args_atomic_count = 0;
    g_orch_scope_end_atomic_count = 0;
    return d;
}
#endif
