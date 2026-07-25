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

#include <cerrno>
#include <chrono>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cctype>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "../common/pto_runtime_status.h"
#include "../runtime/common.h"
#include "../runtime/pto_orchestrator.h"
#include "../runtime/pto_runtime2.h"
#include "../runtime/pto_shared_memory.h"
#include "../runtime/pto_types.h"
#include "../runtime/runtime.h"
#include "../../../../common/runtime_status/error_log.h"
#include "../../../../common/task_interface/call_config.h"
#include "../../../../common/worker/pto_runtime_c_api.h"
#include "callable.h"
#include "common/platform_config.h"
#include "common/strace.h"
#include "common/unified_log.h"
#include "host/raii_scope_guard.h"
#include "utils/device_arena.h"
#include "prepare_callable_common.h"

extern "C" const PipelineContract *get_pipeline_contract(void) {
    static const PipelineContract contract = {
        PTO_PIPELINE_CONTRACT_ABI_VERSION,
        5,
        2,
        2,
        {
            {PTO_PIPELINE_GM_HEAP, PTO_PIPELINE_FILL_MEM, 0},
            {PTO_PIPELINE_GM_SM, PTO_PIPELINE_FILL_MEM, 0},
            {PTO_PIPELINE_RUNTIME_IMAGE, PTO_PIPELINE_FILL_MEM, 0},
            {PTO_PIPELINE_AICPU_STREAM, PTO_PIPELINE_EXEC_HANDLE, 0},
            {PTO_PIPELINE_AICORE_STREAM, PTO_PIPELINE_EXEC_HANDLE, 0},
        },
    };
    return &contract;
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
// host-DDR cross-task pointers (slot_state.task/payload,
// payload.fanin_inline_slot_states[], dep_pool/ready queues); the host relocates them to
// their final device addresses (relocate_host_orch_image, below) BEFORE the H2D
// copy, so the device receives a fully device-addressed image and schedules
// only — no on-device pointer fixup.

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
typedef void (*OrchestrationEntryFunc)(const L2TaskArgs &);
typedef void (*OrchestrationBindFunc)(PTO2Runtime *);

// Resolved orchestration .so entry points. register_callable_impl allocates one
// of these (so both the entry and the .so's own framework_bind_runtime — which
// sets the .so-private g_current_runtime its inline rt_submit_* reads — are
// available per run) and stores its pointer in CallableArtifacts::
// host_orch_func_ptr. Owned for the callable's lifetime alongside
// host_dlopen_handle.
struct HostOrchEntryPoints {
    OrchestrationEntryFunc entry{nullptr};
    OrchestrationBindFunc bind{nullptr};
};

// Run the orchestrator on the host. `rt` was built with its scheduler half
// pointing at the device SM; here we re-point ONLY the orchestrator half at a
// host SM mirror, run the orchestration entry against it, latch the submitted
// task count, and H2D the populated SM to the device (the device scheduler
// reads task descriptors from there). The device never dereferences the
// orchestrator's SM pointers, so leaving them host-side is safe. Returns the
// total task count (>= 0) on success, or -1 on failure.
// host_build_graph host-orch: the orchestrator built the task graph in a host
// SM mirror and (when wiring is folded into submit) the fanout adjacency in the
// host arena, storing host-DDR addresses into the cross-task pointers. Relocate
// them to their FINAL device addresses here on the host, BEFORE the SM/arena are
// copied to the device — so the device receives a fully device-addressed image
// and boots scheduler-only with no on-device pointer fixup.
//
// Relocated pointers span TWO regions with DIFFERENT deltas: the SM block
// (slot_state.task/.payload, fanin_inline_slot_states[], dep-entry.slot_state,
// ready-queue slot.slot_state) and the arena block (slot_state.fanout_head,
// dep-entry.next point into the SM but live in the arena).
// Rather than track which delta each field needs, reloc() classifies every
// pointer by the region it points INTO and applies that region's delta; foreign
// and null pointers pass through untouched. The fanout adjacency is wired inline
// during host submit, so dep_pool/ready are already populated here.
//
// The orchestrator's own task-allocator pointers are intentionally NOT relocated
// (the device runs scheduler-only and never dereferences them, and must not call
// rt_orchestration_done — the host already did). Multi-fanin spill is not yet
// relocated; a task exceeding PTO2_FANIN_INLINE_CAP producers latches fatal here
// (returns false) rather than shipping un-relocated host pointers to the device.
// Returns false on any unrelocatable pointer so the caller can fail the prepare.
static bool relocate_host_orch_image(
    PTO2SharedMemoryHandle &host_sm_handle, [[maybe_unused]] PTO2Runtime *rt, uint64_t host_sm, uint64_t sm_size,
    int64_t sm_delta, uint64_t host_arena, uint64_t arena_size, int64_t arena_delta
) {
    // host_build_graph is single-ring; the loops below iterate the lone ring and
    // index header->ring (singular). If the ring depth ever grows, those loops
    // would relocate the same ring N times (applying the delta repeatedly =
    // corruption), so pin the assumption here.
    static_assert(PTO2_MAX_RING_DEPTH == 1, "relocate_host_orch_image assumes a single ring");

    // SM and arena windows must not overlap — reloc classifies a pointer by
    // which window it falls in, so an overlap would misclassify and apply the
    // wrong delta. Both are independent malloc-backed host buffers in practice;
    // assert it so a future shared-buffer layout can't silently corrupt.
    if (!(host_sm + sm_size <= host_arena || host_arena + arena_size <= host_sm)) {
        LOG_ERROR(
            "host-orch: SM window [%#lx,+%#lx) overlaps arena window [%#lx,+%#lx); cannot relocate", host_sm, sm_size,
            host_arena, arena_size
        );
        return false;
    }

    bool ok = true;
    auto reloc = [&](auto *&p) {
        using Ptr = std::remove_reference_t<decltype(p)>;
        uint64_t v = reinterpret_cast<uint64_t>(p);
        if (v == 0) {
            return;
        }
        if (v >= host_sm && v < host_sm + sm_size) {
            p = reinterpret_cast<Ptr>(static_cast<uintptr_t>(v + sm_delta));
        } else if (v >= host_arena && v < host_arena + arena_size) {
            p = reinterpret_cast<Ptr>(static_cast<uintptr_t>(v + arena_delta));
        } else {
            // A non-null pointer in neither window is an external/host address
            // the device would dereference verbatim after H2D. No field should
            // legitimately carry one; latch fatal rather than ship a host VA to
            // the device (silent AICPU corruption otherwise).
            LOG_ERROR("host-orch: pointer %#lx is outside both SM and arena windows; cannot relocate for device", v);
            ok = false;
        }
    };

    PTO2SharedMemoryHeader *header = host_sm_handle.header;
    if (header != nullptr) {
        for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
            PTO2SharedMemoryRingHeader &ring = header->ring;
            int32_t count = ring.fc.current_task_index.load(std::memory_order_acquire);
            for (int32_t slot = 0; slot < count; slot++) {
                PTO2TaskSlotState *ss = &ring.slot_states[slot];
                // Polling: fanin is a flat array of position-independent local-id
                // integers on the payload, so only the two per-slot arena/SM
                // pointers need relocating. There is no fanout_head/dep_pool graph
                // and no host-seeded ready queue (the device boot scan classifies),
                // so those relocation passes are gone.
                reloc(ss->task);
                reloc(ss->payload);
            }
        }
    }
    return ok;
}

struct HostGraphEpochCapture {
    int32_t task_begin{0};
    int64_t build_start_ns{0};
    std::vector<PTO2HostGraphEpochRange> ranges;
    bool (*boundary_handler)(PTO2Runtime *, PTO2HostGraphEpochRange *, size_t, void *){nullptr};
    void *boundary_context{nullptr};
};

static bool capture_host_graph_epoch(PTO2Runtime *rt, bool final_epoch, void *opaque) {
    auto *capture = static_cast<HostGraphEpochCapture *>(opaque);
    if (capture == nullptr || rt == nullptr || rt->orchestrator.sm_header == nullptr) {
        LOG_ERROR("host-orch: graph boundary has no active capture context");
        return false;
    }

    int32_t task_end = rt->orchestrator.sm_header->ring.fc.current_task_index.load(std::memory_order_acquire);
    if (task_end <= capture->task_begin) {
        LOG_ERROR("host-orch: empty or reversed graph range [%d,%d)", capture->task_begin, task_end);
        return false;
    }

    PTO2HostGraphEpochRange range;
    range.task_begin = capture->task_begin;
    range.task_end = task_end;
    range.final_epoch = final_epoch ? 1 : 0;
    size_t epoch_index = capture->ranges.size();
    auto boundary_time = std::chrono::steady_clock::now();
    int64_t boundary_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(boundary_time.time_since_epoch()).count();
    if (capture->build_start_ns > 0 && boundary_ns >= capture->build_start_ns) {
        char attrs[160];
        std::snprintf(
            attrs, sizeof(attrs), "layer=%zu tasks=%d task_begin=%d task_end=%d final=%d", epoch_index,
            range.task_end - range.task_begin, range.task_begin, range.task_end, range.final_epoch
        );
        STRACE_HOST_SPAN_AT(
            "simpler_run.host_orch.layer_build", capture->build_start_ns, boundary_ns - capture->build_start_ns, 1,
            attrs
        );
    }
    if (capture->boundary_handler != nullptr &&
        !capture->boundary_handler(rt, &range, epoch_index, capture->boundary_context)) {
        return false;
    }
    capture->ranges.push_back(range);
    capture->task_begin = task_end;
    auto next_build_time = std::chrono::steady_clock::now();
    capture->build_start_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(next_build_time.time_since_epoch()).count();
    return true;
}

static bool
materialize_streaming_host_graph_range(PTO2Runtime *source, PTO2Runtime *target, PTO2HostGraphEpochRange *range) {
    if (source == nullptr || target == nullptr || range == nullptr || source->orchestrator.sm_header == nullptr ||
        target->orchestrator.sm_header == nullptr) {
        return false;
    }

    PTO2SharedMemoryRingHeader &source_ring = source->orchestrator.sm_header->ring;
    PTO2SharedMemoryRingHeader &target_ring = target->orchestrator.sm_header->ring;
    int32_t source_total = source_ring.fc.current_task_index.load(std::memory_order_acquire);
    if (range->task_begin < 0 || range->task_end <= range->task_begin || range->task_end > source_total ||
        static_cast<uint64_t>(range->task_end) >= target_ring.task_window_size ||
        target->orchestrator.ring.task_allocator.task_head() != range->task_begin) {
        LOG_ERROR(
            "host-orch: invalid streaming range [%d,%d), source_total=%d target_head=%d window=%" PRIu64,
            range->task_begin, range->task_end, source_total, target->orchestrator.ring.task_allocator.task_head(),
            target_ring.task_window_size
        );
        return false;
    }

    range->inline_completed = 0;

    for (int32_t task_id = range->task_begin; task_id < range->task_end; ++task_id) {
        PTO2TaskSlotState &source_slot = source_ring.get_slot_state_by_task_id(task_id);
        if (source_slot.task == nullptr || source_slot.payload == nullptr) {
            LOG_ERROR("host-orch: source task %d has incomplete slot bindings", task_id);
            return false;
        }

        PTO2TaskAllocResult alloc = target->orchestrator.ring.task_allocator.alloc(0);
        if (alloc.failed() || alloc.task_id != task_id) {
            LOG_ERROR("host-orch: target task allocation diverged at task %d", task_id);
            return false;
        }
        PTO2TaskDescriptor &target_task = target_ring.get_task_by_slot(alloc.slot);
        PTO2TaskPayload &target_payload = target_ring.get_payload_by_slot(alloc.slot);
        PTO2TaskSlotState &target_slot = target_ring.get_slot_state_by_slot(alloc.slot);
        std::memcpy(&target_task, source_slot.task, sizeof(target_task));
        std::memcpy(&target_payload, source_slot.payload, sizeof(target_payload));

        if (target_payload.fanin_count < 0 || target_payload.fanin_count > PTO2_MAX_FANIN) {
            LOG_ERROR("host-orch: streaming task %d has invalid fanin count %d", task_id, target_payload.fanin_count);
            return false;
        }
        for (int32_t i = 0; i < target_payload.fanin_count; ++i) {
            int32_t producer_id = target_payload.fanin_local_ids[i];
            if (producer_id < 0 || producer_id >= task_id) {
                LOG_ERROR("host-orch: task %d has invalid producer id %d", task_id, producer_id);
                return false;
            }
        }
        target_payload.dispatch_fanin.store(0, std::memory_order_relaxed);

        PTO2TaskState task_state = source_slot.task_state.load(std::memory_order_acquire);
        target_slot.reset_for_reuse();
        target_slot.bind_buffers(&target_payload, &target_task);
        target_slot.last_consumer_local_id = source_slot.last_consumer_local_id;
        target_slot.task_state.store(task_state, std::memory_order_relaxed);
        target_slot.active_mask = source_slot.active_mask;
        target_slot.task_attrs = source_slot.task_attrs;
        target_slot.task_attrs.set_early_resolve(false);
        target_slot.total_required_subtasks = source_slot.total_required_subtasks;
        target_slot.logical_block_num = source_slot.logical_block_num;
        uint8_t completed =
            source_ring.completion_flags[source_ring.get_slot_by_task_id(task_id)].load(std::memory_order_acquire);
        target_ring.completion_flags[alloc.slot].store(completed, std::memory_order_relaxed);
        if (completed != 0 || task_state >= PTO2_TASK_COMPLETED) range->inline_completed++;
    }

    target->orchestrator.inline_completed_tasks += static_cast<uint64_t>(range->inline_completed);
    return true;
}

static bool materialize_host_graph_ranges(
    PTO2SharedMemoryHandle &source_sm_handle, PTO2SharedMemoryHandle &target_sm_handle,
    const std::vector<PTO2HostGraphEpochRange> &ranges, int32_t total_tasks
) {
    if (source_sm_handle.header == nullptr || target_sm_handle.header == nullptr) {
        LOG_ERROR("host-orch: cannot materialize graph ranges without source and target SM headers");
        return false;
    }

    PTO2SharedMemoryRingHeader &source_ring = source_sm_handle.header->ring;
    PTO2SharedMemoryRingHeader &target_ring = target_sm_handle.header->ring;
    if (total_tasks < 0 || static_cast<uint64_t>(total_tasks) > source_ring.task_window_size ||
        source_ring.task_window_size != target_ring.task_window_size) {
        LOG_ERROR(
            "host-orch: invalid materialization size tasks=%d source_window=%" PRIu64 " target_window=%" PRIu64,
            total_tasks, source_ring.task_window_size, target_ring.task_window_size
        );
        return false;
    }

    int32_t expected_begin = 0;
    for (size_t index = 0; index < ranges.size(); ++index) {
        const PTO2HostGraphEpochRange &range = ranges[index];
        if (range.task_begin != expected_begin || range.task_end <= range.task_begin || range.task_end > total_tasks) {
            LOG_ERROR(
                "host-orch: epoch %zu does not form a contiguous graph partition: expected_begin=%d range=[%d,%d) "
                "total=%d",
                index, expected_begin, range.task_begin, range.task_end, total_tasks
            );
            return false;
        }

        char attrs[160];
        std::snprintf(
            attrs, sizeof(attrs), "epoch=%zu slot=%zu task_begin=%d task_end=%d tasks=%d", index,
            index % static_cast<size_t>(PTO2_HOST_GRAPH_EPOCH_SLOT_COUNT), range.task_begin, range.task_end,
            range.task_end - range.task_begin
        );
        {
            STRACE_A("simpler_run.host_orch.epoch.materialize", attrs);
            for (int32_t task_id = range.task_begin; task_id < range.task_end; ++task_id) {
                int32_t source_slot = source_ring.get_slot_by_task_id(task_id);
                int32_t target_slot = target_ring.get_slot_by_task_id(task_id);
                std::memcpy(
                    &target_ring.task_descriptors[target_slot], &source_ring.task_descriptors[source_slot],
                    sizeof(PTO2TaskDescriptor)
                );
                std::memcpy(
                    &target_ring.task_payloads[target_slot], &source_ring.task_payloads[source_slot],
                    sizeof(PTO2TaskPayload)
                );
                std::memcpy(
                    &target_ring.slot_states[target_slot], &source_ring.slot_states[source_slot],
                    sizeof(PTO2TaskSlotState)
                );
                target_ring.completion_flags[target_slot].store(
                    source_ring.completion_flags[source_slot].load(std::memory_order_acquire), std::memory_order_relaxed
                );
            }
        }
        expected_begin = range.task_end;
    }

    if (ranges.empty() || expected_begin != total_tasks) {
        LOG_ERROR("host-orch: captured graph ranges cover [0,%d), expected [0,%d)", expected_begin, total_tasks);
        return false;
    }
    return true;
}

int32_t run_host_orchestration(
    Runtime *runtime, const HostApi *api, PTO2Runtime *rt, DeviceArena &host_arena,
    const PTO2RuntimeArenaLayout &layout, void *device_sm, uint64_t sm_size, void *device_arena, void *gm_heap,
    const uint64_t eff_heap_sizes[PTO2_MAX_RING_DEPTH], const uint64_t eff_task_window_sizes[PTO2_MAX_RING_DEPTH],
    void *host_orch_func_ptr, const L2TaskArgs &orch_l2,
    bool (*boundary_handler)(PTO2Runtime *, PTO2HostGraphEpochRange *, size_t, void *) = nullptr,
    void *boundary_context = nullptr
) {
    constexpr char epoch_attrs[] = "epoch=0 slot=0 final=1";
    std::vector<uint8_t> source_sm_buf(sm_size, 0);
    void *source_sm = source_sm_buf.data();

    DeviceArena source_arena;
    PTO2RuntimeArenaLayout source_layout = runtime_reserve_layout(source_arena, eff_task_window_sizes, eff_heap_sizes);
    if (source_layout.arena_size != layout.arena_size || source_layout.off_runtime != layout.off_runtime ||
        source_arena.commit(DeviceArena::kDefaultBaseAlign) == nullptr) {
        LOG_ERROR("host-orch: failed to create an equivalent source arena for epoch capture");
        return -1;
    }

    PTO2Runtime *source_rt = runtime_init_data_from_layout(
        source_arena, source_layout, PTO2_MODE_EXECUTE, device_sm, sm_size, gm_heap, eff_heap_sizes
    );
    if (source_rt == nullptr) {
        LOG_ERROR("host-orch: source runtime init failed");
        return -1;
    }
    runtime_wire_arena_pointers(source_arena, source_layout, source_rt);

    // Re-point the orchestrator half at the host SM (scheduler keeps device SM).
    // init_data_from_layout resets the orchestrator state, so this is safe.
    if (!source_rt->orchestrator.init_data_from_layout(
            source_layout.orch, source_arena, source_sm, gm_heap, eff_heap_sizes[0], eff_task_window_sizes[0]
        )) {
        LOG_ERROR("host-orch: orchestrator re-init against host SM failed");
        return -1;
    }
    source_rt->orchestrator.wire_arena_pointers(source_layout.orch, source_arena, &source_rt->scheduler);
    source_rt->scheduler.sm_header = source_rt->orchestrator.sm_header;
    source_rt->scheduler.ring_sched_state.ring = &source_rt->orchestrator.sm_header->ring;

    // Initialize the host SM header (ring flow control) so submit_task can run.
    PTO2SharedMemoryHandle source_sm_handle;
    if (!source_sm_handle.init_per_ring(source_sm, sm_size, eff_task_window_sizes, eff_heap_sizes)) {
        LOG_ERROR("host-orch: host SM init_per_ring failed");
        return -1;
    }

    // Install the ops table (host s_runtime_ops). The SPMD core counts are
    // re-applied with the real device values on the AICPU at boot; the values
    // here only feed cluster spreading during this host submit and are unused
    // by the migrated non-cluster examples.
    runtime_finalize_after_wire(source_rt, /*aic*/ 24, /*aiv*/ 48);
    source_rt->mode = PTO2_MODE_EXECUTE;
    // get_tensor_data/set_tensor_data dereference buffer.addr directly: the
    // input tensors were mapped into host address space at staging time
    // (HostApi::register_device_memory_to_host), so the host orchestrator can
    // read control tensors (e.g. paged_attention's context_lens/block_table) in
    // place.

    // Bind both framework_current_runtime instances: the host library's (used by
    // rt_scope_* / rt_orchestration_done) and the orch .so's own copy (used by
    // its inline rt_submit_* -> current_runtime()).
    const HostOrchEntryPoints *eps = reinterpret_cast<const HostOrchEntryPoints *>(host_orch_func_ptr);
    framework_bind_runtime(source_rt);
    if (eps->bind != nullptr) {
        eps->bind(source_rt);
    } else {
        LOG_ERROR("host-orch: orch .so framework_bind_runtime was not resolved");
        return -1;
    }

    HostGraphEpochCapture epoch_capture;
    epoch_capture.boundary_handler = boundary_handler;
    epoch_capture.boundary_context = boundary_context;
    {
        runtime_set_graph_boundary_callback(source_rt, capture_host_graph_epoch, &epoch_capture);
        auto capture_guard = RAIIScopeGuard([source_rt]() {
            runtime_set_graph_boundary_callback(source_rt, nullptr, nullptr);
        });
        {
            STRACE_A("simpler_run.host_orch.epoch.build", epoch_attrs);
            rt_scope_begin(source_rt);
            auto build_start = std::chrono::steady_clock::now();
            epoch_capture.build_start_ns =
                std::chrono::duration_cast<std::chrono::nanoseconds>(build_start.time_since_epoch()).count();
            eps->entry(orch_l2);
            rt_scope_end(source_rt);
            // An orchestration without explicit boundaries is one final epoch.
            if (epoch_capture.ranges.empty()) rt_graph_boundary(source_rt, true);
            rt_orchestration_done(source_rt);
        }
    }

    int32_t total_tasks = pto2_sm_layout::ring_current_task_index_addr(source_sm)->load(std::memory_order_acquire);
    uint64_t captured_epochs = epoch_capture.ranges.size();
    char capture_attrs[96];
    std::snprintf(
        capture_attrs, sizeof(capture_attrs), "captured_epochs=%" PRIu64 " tasks=%d", captured_epochs, total_tasks
    );
    { STRACE_A("simpler_run.host_orch.boundary_capture", capture_attrs); }
    for (size_t index = 0; index < epoch_capture.ranges.size(); ++index) {
        const PTO2HostGraphEpochRange &range = epoch_capture.ranges[index];
        char range_attrs[160];
        std::snprintf(
            range_attrs, sizeof(range_attrs), "epoch=%zu slot=%zu task_begin=%d task_end=%d tasks=%d", index,
            index % static_cast<size_t>(PTO2_HOST_GRAPH_EPOCH_SLOT_COUNT), range.task_begin, range.task_end,
            range.task_end - range.task_begin
        );
        { STRACE_A("simpler_run.host_orch.epoch.capture", range_attrs); }
    }
    LOG_INFO_V0("host-orch: captured epochs=%" PRIu64 " tasks=%d", captured_epochs, total_tasks);

    if (boundary_handler != nullptr) {
        if (epoch_capture.ranges.empty() || epoch_capture.ranges.back().final_epoch == 0 ||
            epoch_capture.ranges.back().task_end != total_tasks) {
            LOG_ERROR("host-orch: streaming orchestration did not publish one final range covering all tasks");
            return -1;
        }
        return total_tasks;
    }

    std::memcpy(host_arena.base(), source_arena.base(), layout.arena_size);
    runtime_wire_arena_pointers(host_arena, layout, rt);

    std::vector<uint8_t> target_sm_buf(sm_size, 0);
    void *target_sm = target_sm_buf.data();
    std::memcpy(target_sm, source_sm, sizeof(PTO2SharedMemoryHeader));
    PTO2SharedMemoryHandle target_sm_handle;
    if (!target_sm_handle.attach_populated(target_sm, sm_size, eff_task_window_sizes)) {
        LOG_ERROR("host-orch: target SM attach failed during epoch materialization");
        return -1;
    }
    if (!materialize_host_graph_ranges(source_sm_handle, target_sm_handle, epoch_capture.ranges, total_tasks)) {
        return -1;
    }

    PTO2HostGraphEpochControl &epoch_control = target_sm_handle.header->host_graph_epochs;
    PTO2HostGraphEpochSlot &published_slot = epoch_control.slots[0];
    published_slot.owner_epoch.store(1, std::memory_order_relaxed);
    published_slot.range.task_begin = 0;
    published_slot.range.task_end = total_tasks;
    published_slot.range.final_epoch = 1;
    epoch_control.host_publish_epoch.store(0, std::memory_order_relaxed);

    // The source arena is a byte-for-byte execution-state image. Rewiring above
    // makes all arena-private bases point at the target before relocation walks
    // them. The scheduler's SM anchors were device addresses before the copy and
    // remain so; only cross-task pointers move through source -> target -> device.
    const int64_t source_to_target_sm_delta = static_cast<int64_t>(reinterpret_cast<uint64_t>(target_sm)) -
                                              static_cast<int64_t>(reinterpret_cast<uint64_t>(source_sm));
    const int64_t source_to_target_arena_delta = static_cast<int64_t>(reinterpret_cast<uint64_t>(host_arena.base())) -
                                                 static_cast<int64_t>(reinterpret_cast<uint64_t>(source_arena.base()));
    if (!relocate_host_orch_image(
            target_sm_handle, rt, reinterpret_cast<uint64_t>(source_sm), sm_size, source_to_target_sm_delta,
            reinterpret_cast<uint64_t>(source_arena.base()), layout.arena_size, source_to_target_arena_delta
        )) {
        LOG_ERROR("host-orch: source-to-target epoch relocation failed");
        return -1;
    }

    // Relocate the host-DDR cross-task pointers to their final DEVICE addresses
    // on the host, before the SM and arena leave for the device. Pointers into
    // the SM shift by sm_delta; pointers into the arena (fanout adjacency, wiring
    // queue) shift by arena_delta. After this both the SM and arena carry device
    // addresses, so the device boots scheduler-only.
    const int64_t sm_delta = static_cast<int64_t>(reinterpret_cast<uint64_t>(device_sm)) -
                             static_cast<int64_t>(reinterpret_cast<uint64_t>(target_sm));
    const int64_t arena_delta = static_cast<int64_t>(reinterpret_cast<uint64_t>(device_arena)) -
                                static_cast<int64_t>(reinterpret_cast<uint64_t>(host_arena.base()));
    {
        STRACE_A("simpler_run.host_orch.image.relocate", epoch_attrs);
        if (!relocate_host_orch_image(
                target_sm_handle, rt, reinterpret_cast<uint64_t>(target_sm), sm_size, sm_delta,
                reinterpret_cast<uint64_t>(host_arena.base()), layout.arena_size, arena_delta
            )) {
            LOG_ERROR("host-orch: relocation failed; refusing to H2D an image with unrelocated host pointers");
            return -1;
        }
    }

    {
        STRACE_A("simpler_run.host_orch.epoch.stage_sm", epoch_attrs);
        if (api->copy_to_device(device_sm, target_sm, sm_size) != 0) {
            LOG_ERROR("host-orch: H2D of populated SM failed");
            return -1;
        }
    }
    return total_tasks;
}

std::shared_ptr<std::mutex> stage1_gate_for_runner(void *runner_context) {
    static std::mutex registry_mutex;
    static std::unordered_map<void *, std::weak_ptr<std::mutex>> gates;

    std::lock_guard<std::mutex> lock(registry_mutex);
    auto gate = gates[runner_context].lock();
    if (gate == nullptr) {
        gate = std::make_shared<std::mutex>();
        gates[runner_context] = gate;
    }
    return gate;
}

class HostGraphAsyncJob {
public:
    ~HostGraphAsyncJob() { (void)join(); }

    bool prepare_and_start(
        Runtime *runtime, const HostApi *api, const PTO2RuntimeArenaLayout &layout, void *device_sm, uint64_t sm_size,
        void *device_arena, void *gm_heap, const uint64_t heap_sizes[PTO2_MAX_RING_DEPTH],
        const uint64_t task_window_sizes[PTO2_MAX_RING_DEPTH], void *host_orch_func_ptr,
        const ChipStorageTaskArgs &device_args
    ) {
        runtime_ = runtime;
        api_ = api;
        layout_ = layout;
        device_sm_ = device_sm;
        sm_size_ = sm_size;
        device_arena_ = device_arena;
        gm_heap_ = gm_heap;
        host_orch_func_ptr_ = host_orch_func_ptr;
        device_args_ = device_args;
        std::memcpy(heap_sizes_, heap_sizes, sizeof(heap_sizes_));
        std::memcpy(task_window_sizes_, task_window_sizes, sizeof(task_window_sizes_));
#if SIMPLER_HOST_STRACE
        strace_inv_ = simpler::strace::StraceScope::current_inv();
        strace_hid_ = simpler::strace::StraceScope::current_hid();
#endif

        if (runtime_ == nullptr || api_ == nullptr || device_sm_ == nullptr || device_arena_ == nullptr ||
            gm_heap_ == nullptr || host_orch_func_ptr_ == nullptr || api_->capture_thread_context == nullptr ||
            api_->bind_thread_context == nullptr || api_->unbind_thread_context == nullptr ||
            api_->store_u64_release_to_device == nullptr || api_->load_u64_acquire_from_device == nullptr) {
            LOG_ERROR("host-orch: async job has incomplete inputs");
            return false;
        }

        PTO2RuntimeArenaLayout target_layout = runtime_reserve_layout(target_arena_, task_window_sizes_, heap_sizes_);
        if (target_layout.arena_size != layout_.arena_size || target_layout.off_runtime != layout_.off_runtime ||
            target_arena_.commit(DeviceArena::kDefaultBaseAlign) == nullptr) {
            LOG_ERROR("host-orch: async target arena layout mismatch or allocation failure");
            return false;
        }
        target_rt_ = runtime_init_data_from_layout(
            target_arena_, layout_, PTO2_MODE_EXECUTE, device_sm_, sm_size_, gm_heap_, heap_sizes_
        );
        if (target_rt_ == nullptr) {
            LOG_ERROR("host-orch: async target runtime init failed");
            return false;
        }
        runtime_wire_arena_pointers(target_arena_, layout_, target_rt_);
        target_rt_->prebuilt_layout = layout_;

        target_sm_.resize(sm_size_, 0);
        if (!target_sm_handle_.init_per_ring(target_sm_.data(), sm_size_, task_window_sizes_, heap_sizes_) ||
            !target_rt_->orchestrator.init_data_from_layout(
                layout_.orch, target_arena_, target_sm_.data(), gm_heap_, heap_sizes_[0], task_window_sizes_[0]
            )) {
            LOG_ERROR("host-orch: failed to initialize the streaming target image");
            return false;
        }
        target_rt_->orchestrator.wire_arena_pointers(layout_.orch, target_arena_, &target_rt_->scheduler);

        // Publish the empty runtime arena once. Device-side wire restores every
        // arena-private pointer; the orchestrator's host SM anchor is unused by
        // scheduler-only execution and is temporarily replaced for this image.
        PTO2SharedMemoryHeader *target_host_header = target_rt_->orchestrator.sm_header;
        target_rt_->orchestrator.sm_header = static_cast<PTO2SharedMemoryHeader *>(device_sm_);
        int arena_rc = api_->copy_to_device(device_arena_, target_arena_.base(), layout_.arena_size);
        target_rt_->orchestrator.sm_header = target_host_header;
        if (arena_rc != 0 || api_->copy_to_device(device_sm_, target_sm_.data(), sm_size_) != 0) {
            LOG_ERROR("host-orch: failed to reset the async publication SM");
            return false;
        }

        runtime_->set_prebuilt_arena(device_arena_, layout_.off_runtime);
        runtime_->host_total_tasks = -1;
        thread_context_ = api_->capture_thread_context();
        if (thread_context_ == nullptr) {
            LOG_ERROR("host-orch: failed to capture the runner thread context");
            return false;
        }
        stage1_gate_ = stage1_gate_for_runner(thread_context_);

        try {
            worker_ = std::thread([this]() {
#if SIMPLER_HOST_STRACE
                STRACE_SET_CONTEXT(strace_inv_, strace_hid_);
#endif
                if (api_->bind_thread_context(thread_context_) != 0) {
                    result_ = -1;
                    worker_start_state_.store(-1, std::memory_order_release);
                    return;
                }
                worker_start_state_.store(1, std::memory_order_release);
                {
                    std::unique_lock<std::mutex> lock(run_gate_mutex_);
                    run_gate_cv_.wait(lock, [this]() {
                        return run_gate_state_.load(std::memory_order_acquire) != 0;
                    });
                }
                if (run_gate_state_.load(std::memory_order_acquire) < 0) {
                    result_ = -1;
                    api_->unbind_thread_context();
                    return;
                }
                // Host orchestration uses one control flow per runner. A later
                // request may prepare while the prior request executes on the
                // device, but two Host Stage1 sections must never mutate the
                // orchestration runtime at the same time.
                std::unique_lock<std::mutex> stage1_lock(*stage1_gate_);
                try {
                    run();
                } catch (const std::exception &e) {
                    LOG_ERROR("host-orch: async worker exception: %s", e.what());
                    publish_failure(PTO2_ERROR_EXPLICIT_ORCH_FATAL);
                    result_ = -1;
                } catch (...) {
                    LOG_ERROR("host-orch: async worker unknown exception");
                    publish_failure(PTO2_ERROR_EXPLICIT_ORCH_FATAL);
                    result_ = -1;
                }
                api_->unbind_thread_context();
            });
        } catch (const std::exception &e) {
            LOG_ERROR("host-orch: failed to start async worker: %s", e.what());
            return false;
        }

        while (worker_start_state_.load(std::memory_order_acquire) == 0)
            std::this_thread::yield();
        if (worker_start_state_.load(std::memory_order_acquire) < 0) {
            worker_.join();
            return false;
        }
        return true;
    }

    bool open_run_gate() {
        int32_t expected = 0;
        if (!run_gate_state_.compare_exchange_strong(
                expected, 1, std::memory_order_release, std::memory_order_acquire
            )) {
            return expected == 1;
        }
        run_gate_cv_.notify_one();
        LOG_INFO_V9("host-orch: async worker run gate opened");
        return true;
    }

    bool release_for_run() {
        if (!open_run_gate()) return false;
        int32_t expected = 0;
        if (!publish_gate_state_.compare_exchange_strong(
                expected, 1, std::memory_order_release, std::memory_order_acquire
            )) {
            return expected == 1;
        }
        run_gate_cv_.notify_all();
        LOG_INFO_V9("host-orch: publication gate released after runner profiling setup");
        return true;
    }

    int join() {
        int32_t expected = 0;
        if (run_gate_state_.compare_exchange_strong(
                expected, -1, std::memory_order_release, std::memory_order_acquire
            )) {
            run_gate_cv_.notify_all();
        }
        expected = 0;
        (void)publish_gate_state_.compare_exchange_strong(
            expected, -1, std::memory_order_release, std::memory_order_acquire
        );
        run_gate_cv_.notify_all();
        if (worker_.joinable()) worker_.join();
        return result_;
    }

private:
    bool wait_for_publish_gate() {
        std::unique_lock<std::mutex> lock(run_gate_mutex_);
        run_gate_cv_.wait(lock, [this]() {
            return publish_gate_state_.load(std::memory_order_acquire) != 0;
        });
        return publish_gate_state_.load(std::memory_order_acquire) > 0;
    }

    void publish_failure(int32_t error_code) const {
        if (api_ == nullptr || device_sm_ == nullptr) return;
        uint64_t failed_epoch = 1;
        (void)api_->copy_to_device(
            static_cast<char *>(device_sm_) + offsetof(PTO2SharedMemoryHeader, orch_error_code), &error_code,
            sizeof(error_code)
        );
        (void)api_->store_u64_release_to_device(
            static_cast<char *>(device_sm_) + offsetof(PTO2SharedMemoryHeader, host_graph_epochs) +
                offsetof(PTO2HostGraphEpochControl, failed_epoch),
            failed_epoch
        );
    }

    bool wait_for_device_epoch(size_t field_offset, uint64_t epoch, const char *kind) const {
        char attrs[96];
        std::snprintf(attrs, sizeof(attrs), "kind=%s target_epoch=%" PRIu64, kind, epoch);
        STRACE_A("simpler_run.host_orch.epoch.wait_device", attrs);
        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        uint64_t observed = 0;
        while (observed < epoch) {
            if (api_->load_u64_acquire_from_device(&observed, static_cast<const char *>(device_sm_) + field_offset) !=
                0) {
                LOG_ERROR("host-orch: failed to read Device %s epoch", kind);
                return false;
            }
            if (observed >= epoch) return true;
            if (std::chrono::steady_clock::now() >= deadline) {
                LOG_ERROR(
                    "host-orch: timed out waiting for Device %s epoch=%" PRIu64 " (observed=%" PRIu64 ")", kind, epoch,
                    observed
                );
                return false;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
        return true;
    }

    bool reclaim_target_through(uint64_t epoch) {
        if (epoch <= target_reclaimed_epoch_) return true;
        if (epoch > published_ranges_.size()) {
            LOG_ERROR(
                "host-orch: cannot reclaim unpublished target epoch=%" PRIu64 " (published=%zu)", epoch,
                published_ranges_.size()
            );
            return false;
        }

        target_reclaimed_epoch_ = epoch;
        LOG_INFO_V9("host-orch: epoch metadata slot is reusable through epoch=%" PRIu64, epoch);
        return true;
    }

    bool wait_and_reclaim_target(uint64_t epoch, const char *kind) {
        if (epoch <= target_reclaimed_epoch_) return true;
        size_t field_offset = offsetof(PTO2SharedMemoryHeader, host_graph_epochs) +
                              offsetof(PTO2HostGraphEpochControl, device_buffer_free_epoch);
        return wait_for_device_epoch(field_offset, epoch, kind) && reclaim_target_through(epoch);
    }

    bool ensure_target_capacity(const PTO2HostGraphEpochRange &range) {
        PTO2TaskAllocator &allocator = target_rt_->orchestrator.ring.task_allocator;
        int32_t task_count = range.task_end - range.task_begin;
        if (task_count <= 0 || range.task_begin != allocator.task_head() || range.task_end >= allocator.window_size()) {
            LOG_ERROR(
                "host-orch: epoch cannot fit whole-graph target storage "
                "(range=[%d,%d) target_head=%d window=%d)",
                range.task_begin, range.task_end, allocator.task_head(), allocator.window_size()
            );
            return false;
        }
        return true;
    }

    bool stage_epoch(const PTO2HostGraphEpochRange &range, size_t epoch_index) {
        const int32_t task_count = range.task_end - range.task_begin;
        PTO2SharedMemoryRingHeader &ring = target_sm_handle_.header->ring;
        if (task_count <= 0 || range.task_begin < 0 || static_cast<uint64_t>(range.task_end) >= ring.task_window_size) {
            LOG_ERROR(
                "host-orch: invalid streaming task range size=%d window=%" PRIu64, task_count, ring.task_window_size
            );
            return false;
        }

        char attrs[192];
        std::snprintf(
            attrs, sizeof(attrs), "epoch=%zu slot=%zu task_begin=%d task_end=%d tasks=%d final=%d", epoch_index + 1,
            epoch_index % static_cast<size_t>(PTO2_HOST_GRAPH_EPOCH_SLOT_COUNT), range.task_begin, range.task_end,
            task_count, range.final_epoch
        );
        STRACE_A("simpler_run.host_orch.epoch.stage_upload", attrs);

        auto relocate_target_pointer = [&](auto *&pointer) -> bool {
            using Pointer = std::remove_reference_t<decltype(pointer)>;
            uint64_t value = reinterpret_cast<uint64_t>(pointer);
            if (value == 0) return true;
            uint64_t host_sm = reinterpret_cast<uint64_t>(target_sm_.data());
            uint64_t host_arena = reinterpret_cast<uint64_t>(target_arena_.base());
            if (value >= host_sm && value < host_sm + sm_size_) {
                pointer = reinterpret_cast<Pointer>(
                    reinterpret_cast<uint64_t>(device_sm_) + static_cast<uint64_t>(value - host_sm)
                );
                return true;
            }
            if (value >= host_arena && value < host_arena + layout_.arena_size) {
                pointer = reinterpret_cast<Pointer>(
                    reinterpret_cast<uint64_t>(device_arena_) + static_cast<uint64_t>(value - host_arena)
                );
                return true;
            }
            LOG_ERROR("host-orch: streaming pointer %#lx is outside target SM/arena", value);
            return false;
        };

        auto sm_offsets = pto2_sm_layout::ring_segment_offsets(ring.task_window_size);
        char *device_sm_bytes = static_cast<char *>(device_sm_);
        auto stage_task_run = [&](int32_t first_slot, int32_t run_count) {
            std::vector<uint8_t> payload_image(static_cast<size_t>(run_count) * sizeof(PTO2TaskPayload));
            std::vector<uint8_t> slot_image(static_cast<size_t>(run_count) * sizeof(PTO2TaskSlotState));
            std::vector<uint8_t> completion_image(static_cast<size_t>(run_count));
            for (int32_t i = 0; i < run_count; ++i) {
                PTO2TaskPayload payload_copy;
                PTO2TaskSlotState slot_copy;
                std::memcpy(&payload_copy, &ring.task_payloads[first_slot + i], sizeof(payload_copy));
                std::memcpy(&slot_copy, &ring.slot_states[first_slot + i], sizeof(slot_copy));
                if (payload_copy.fanin_count < 0 || payload_copy.fanin_count > PTO2_MAX_FANIN) return false;
                if (!relocate_target_pointer(slot_copy.task) || !relocate_target_pointer(slot_copy.payload)) {
                    return false;
                }
                std::memcpy(
                    payload_image.data() + static_cast<size_t>(i) * sizeof(payload_copy), &payload_copy,
                    sizeof(payload_copy)
                );
                std::memcpy(
                    slot_image.data() + static_cast<size_t>(i) * sizeof(slot_copy), &slot_copy, sizeof(slot_copy)
                );
                completion_image[static_cast<size_t>(i)] =
                    ring.completion_flags[first_slot + i].load(std::memory_order_acquire);
            }

            size_t descriptor_bytes = static_cast<size_t>(run_count) * sizeof(PTO2TaskDescriptor);
            return api_->copy_to_device(
                       device_sm_bytes + sm_offsets.descriptors +
                           static_cast<size_t>(first_slot) * sizeof(PTO2TaskDescriptor),
                       &ring.task_descriptors[first_slot], descriptor_bytes
                   ) == 0 &&
                   api_->copy_to_device(
                       device_sm_bytes + sm_offsets.payloads +
                           static_cast<size_t>(first_slot) * sizeof(PTO2TaskPayload),
                       payload_image.data(), payload_image.size()
                   ) == 0 &&
                   api_->copy_to_device(
                       device_sm_bytes + sm_offsets.slot_states +
                           static_cast<size_t>(first_slot) * sizeof(PTO2TaskSlotState),
                       slot_image.data(), slot_image.size()
                   ) == 0 &&
                   api_->copy_to_device(
                       device_sm_bytes + sm_offsets.completion_flags + static_cast<size_t>(first_slot),
                       completion_image.data(), completion_image.size()
                   ) == 0;
        };
        int32_t staged_tasks = 0;
        while (staged_tasks < task_count) {
            int32_t task_id = range.task_begin + staged_tasks;
            int32_t first_slot = ring.get_slot_by_task_id(task_id);
            int32_t run_count =
                std::min(task_count - staged_tasks, static_cast<int32_t>(ring.task_window_size) - first_slot);
            if (!stage_task_run(first_slot, run_count)) {
                LOG_ERROR("host-orch: failed to stage task range [%d,%d)", range.task_begin, range.task_end);
                return false;
            }
            staged_tasks += run_count;
        }

        PTO2HostGraphEpochControl &control = target_sm_handle_.header->host_graph_epochs;
        size_t control_slot = epoch_index % static_cast<size_t>(PTO2_HOST_GRAPH_EPOCH_SLOT_COUNT);
        PTO2HostGraphEpochSlot &epoch_slot = control.slots[control_slot];
        epoch_slot.owner_epoch.store(epoch_index + 1, std::memory_order_relaxed);
        epoch_slot.range = range;

        int32_t current_task_index = range.task_end;
        size_t epoch_slot_offset = offsetof(PTO2SharedMemoryHeader, host_graph_epochs) +
                                   offsetof(PTO2HostGraphEpochControl, slots) +
                                   control_slot * sizeof(PTO2HostGraphEpochSlot);
        if (api_->copy_to_device(
                pto2_sm_layout::ring_current_task_index_addr(device_sm_), &current_task_index,
                sizeof(current_task_index)
            ) != 0 ||
            api_->copy_to_device(device_sm_bytes + epoch_slot_offset, &epoch_slot, sizeof(epoch_slot)) != 0) {
            LOG_ERROR("host-orch: failed to stage epoch metadata");
            return false;
        }
        return true;
    }

    bool commit_publication(const PTO2HostGraphEpochRange &range, size_t epoch_index) const {
        uint64_t publish_epoch = epoch_index + 1;
        size_t control_slot = epoch_index % static_cast<size_t>(PTO2_HOST_GRAPH_EPOCH_SLOT_COUNT);
        char attrs[160];
        std::snprintf(
            attrs, sizeof(attrs), "epoch=%" PRIu64 " slot=%zu task_begin=%d task_end=%d final=%d", publish_epoch,
            control_slot, range.task_begin, range.task_end, range.final_epoch
        );
        STRACE_A("simpler_run.host_orch.epoch.commit", attrs);
        size_t epoch_offset = offsetof(PTO2SharedMemoryHeader, host_graph_epochs) +
                              offsetof(PTO2HostGraphEpochControl, host_publish_epoch);
        return api_->store_u64_release_to_device(static_cast<char *>(device_sm_) + epoch_offset, publish_epoch) == 0;
    }

    bool publish_boundary(PTO2Runtime *source, PTO2HostGraphEpochRange *range, size_t epoch_index) {
        uint64_t publish_epoch = epoch_index + 1;
        if (epoch_index >= static_cast<size_t>(PTO2_HOST_GRAPH_EPOCH_SLOT_COUNT) &&
            !wait_and_reclaim_target(publish_epoch - PTO2_HOST_GRAPH_EPOCH_SLOT_COUNT, "slot-free")) {
            return false;
        }
        if (!ensure_target_capacity(*range)) return false;

        char attrs[160];
        std::snprintf(
            attrs, sizeof(attrs), "epoch=%" PRIu64 " task_begin=%d task_end=%d final=%d", publish_epoch,
            range->task_begin, range->task_end, range->final_epoch
        );
        {
            STRACE_A("simpler_run.host_orch.epoch.materialize", attrs);
            if (!materialize_streaming_host_graph_range(source, target_rt_, range)) return false;
        }
        if (!stage_epoch(*range, epoch_index)) return false;
        if (epoch_index == 0) {
            char attrs[96];
            std::snprintf(attrs, sizeof(attrs), "epoch=%" PRIu64 " kind=publish-gate", publish_epoch);
            STRACE_A("simpler_run.host_orch.epoch.wait_publish", attrs);
            if (!wait_for_publish_gate()) return false;
        }

        size_t control_base = offsetof(PTO2SharedMemoryHeader, host_graph_epochs);
        if (epoch_index > 0 && !wait_for_device_epoch(
                                   control_base + offsetof(PTO2HostGraphEpochControl, device_exec_done_epoch),
                                   publish_epoch - 1, "exec-done"
                               )) {
            return false;
        }
        if (!commit_publication(*range, epoch_index)) return false;
        published_ranges_.push_back(*range);
        if (!wait_for_device_epoch(
                control_base + offsetof(PTO2HostGraphEpochControl, device_release_epoch), publish_epoch, "release"
            )) {
            return false;
        }
        source->orchestrator.sm_header->ring.fc.last_task_alive.store(range->task_end, std::memory_order_release);
        return true;
    }

    static bool
    boundary_handler(PTO2Runtime *source, PTO2HostGraphEpochRange *range, size_t epoch_index, void *context) {
        auto *job = static_cast<HostGraphAsyncJob *>(context);
        return job != nullptr && job->publish_boundary(source, range, epoch_index);
    }

    void run() {
        L2TaskArgs orch_l2;
        orch_l2.create_from_chip_args(device_args_);
        int32_t total_tasks = run_host_orchestration(
            runtime_, api_, target_rt_, target_arena_, layout_, device_sm_, sm_size_, device_arena_, gm_heap_,
            heap_sizes_, task_window_sizes_, host_orch_func_ptr_, orch_l2, boundary_handler, this
        );
        if (total_tasks < 0) {
            publish_failure(PTO2_ERROR_EXPLICIT_ORCH_FATAL);
            result_ = -1;
            return;
        }

        runtime_->host_total_tasks = total_tasks;
        LOG_INFO_V9("host-orch: async streaming graph completed with %d tasks", total_tasks);
        result_ = 0;
    }

    Runtime *runtime_{nullptr};
    const HostApi *api_{nullptr};
    void *thread_context_{nullptr};
    std::shared_ptr<std::mutex> stage1_gate_;
    PTO2RuntimeArenaLayout layout_{};
    DeviceArena target_arena_;
    PTO2Runtime *target_rt_{nullptr};
    std::vector<uint8_t> target_sm_;
    PTO2SharedMemoryHandle target_sm_handle_{};
    void *device_sm_{nullptr};
    uint64_t sm_size_{0};
    void *device_arena_{nullptr};
    void *gm_heap_{nullptr};
    uint64_t heap_sizes_[PTO2_MAX_RING_DEPTH]{};
    uint64_t task_window_sizes_[PTO2_MAX_RING_DEPTH]{};
    std::vector<PTO2HostGraphEpochRange> published_ranges_;
    uint64_t target_reclaimed_epoch_{0};
    void *host_orch_func_ptr_{nullptr};
    ChipStorageTaskArgs device_args_;
    std::thread worker_;
    std::mutex run_gate_mutex_;
    std::condition_variable run_gate_cv_;
    std::atomic<int32_t> run_gate_state_{0};
    std::atomic<int32_t> publish_gate_state_{0};
    std::atomic<int32_t> worker_start_state_{0};
    int result_{-1};
#if SIMPLER_HOST_STRACE
    unsigned strace_inv_{0};
    uint64_t strace_hid_{0};
#endif
};

static bool start_async_host_graph_pipeline(
    Runtime *runtime, const HostApi *api, const PTO2RuntimeArenaLayout &layout, void *device_sm, uint64_t sm_size,
    void *device_arena, void *gm_heap, const uint64_t heap_sizes[PTO2_MAX_RING_DEPTH],
    const uint64_t task_window_sizes[PTO2_MAX_RING_DEPTH], void *host_orch_func_ptr,
    const ChipStorageTaskArgs &device_args
) {
    auto *job = new (std::nothrow) HostGraphAsyncJob{};
    if (job == nullptr) return false;
    if (!job->prepare_and_start(
            runtime, api, layout, device_sm, sm_size, device_arena, gm_heap, heap_sizes, task_window_sizes,
            host_orch_func_ptr, device_args
        )) {
        delete job;
        return false;
    }
    runtime->set_host_orch_job(job);
    return true;
}

static int join_async_host_graph_pipeline(Runtime *runtime) {
    auto *job = static_cast<HostGraphAsyncJob *>(runtime->take_host_orch_job());
    if (job == nullptr) return 0;
    int rc = job->join();
    delete job;
    return rc;
}

}  // namespace

extern "C" int open_async_host_graph_pipeline(Runtime *runtime) {
    if (runtime == nullptr) return -1;
    auto *job = static_cast<HostGraphAsyncJob *>(runtime->take_host_orch_job());
    if (job == nullptr) return 0;
    runtime->set_host_orch_job(job);
    return job->open_run_gate() ? 0 : -1;
}

extern "C" int release_async_host_graph_pipeline(Runtime *runtime) {
    if (runtime == nullptr) return -1;
    auto *job = static_cast<HostGraphAsyncJob *>(runtime->take_host_orch_job());
    if (job == nullptr) return 0;
    runtime->set_host_orch_job(job);
    return job->release_for_run() ? 0 : -1;
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
        LOG_INFO_V0("host-orch: loaded orchestration entry '%s' on host", orch_func_name);
    }
    LOG_INFO_V0("Orchestration SO: %zu bytes staged", orch_so_size);
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
 * @param runtime    Pointer to pre-constructed Runtime (host_api populated)
 * @param orch_args  Separated tensor/scalar arguments for this run
 * @return 0 on success, -1 on failure
 */
extern "C" int bind_callable_to_runtime_impl(
    Runtime *runtime, const HostApi *api, const ChipStorageTaskArgs *orch_args, void *host_orch_func_ptr,
    const ArgDirection *signature, int sig_count, const uint64_t *ring_task_window, const uint64_t *ring_heap,
    [[maybe_unused]] const uint64_t *ring_dep_pool  // polling has no dep_pool; kept for ABI stability
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
    LOG_INFO_V0("RT2 bind: %d tensors + %d scalars, host orchestration mode", tensor_count, scalar_count);

    int64_t t_total_start = _now_ms();

    uint64_t eff_task_window_sizes[PTO2_MAX_RING_DEPTH];
    uint64_t eff_heap_sizes[PTO2_MAX_RING_DEPTH];
    if (!resolve_ring_config(ring_task_window, ring_heap, eff_task_window_sizes, eff_heap_sizes)) {
        return -1;
    }
    const std::string task_window_log = format_ring_array(eff_task_window_sizes);
    const std::string heap_log = format_ring_array(eff_heap_sizes);
    LOG_INFO_V0("Ring buffer sizes: task_window=%s heap=%s", task_window_log.c_str(), heap_log.c_str());

    // Build device args: copy from input, replace host tensor pointers with device pointers
    ChipStorageTaskArgs device_args;

    int64_t t_args_start = _now_ms();
    for (int i = 0; i < tensor_count; i++) {
        Tensor t = orch_args->tensor(i);

        if (t.is_child_memory()) {
            LOG_INFO_V0("  Tensor %d: child memory, pass-through (0x%" PRIx64 ")", i, t.buffer.addr);
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

        // host_build_graph runs the orchestrator on the host, which may read
        // control tensors (e.g. paged_attention's context_lens/block_table) via
        // get_tensor_data to shape the graph. Map this device buffer into the
        // host address space so the host can dereference buffer.addr directly.
        // Released in validate_runtime_impl before device_free.
        //
        // The host then reads/writes buffer.addr (== dev_ptr) directly, so this
        // path REQUIRES an identity mapping (host VA == dev_ptr). a2a3
        // halHostRegister(DEV_SVM_MAP_HOST) returns identity and sim is already a
        // host pointer, but the HAL contract permits a non-identity VA — verify
        // it here and fail the prepare rather than letting the host dereference a
        // device address (segfault / silent corruption) on a future HAL.
        void *host_va = api->register_device_memory_to_host(dev_ptr, size);
        if (host_va != nullptr && host_va != dev_ptr) {
            LOG_ERROR(
                "host-orch: SVM map returned non-identity host VA %p for dev_ptr %p; the host orchestrator "
                "dereferences buffer.addr directly and assumes identity mapping",
                host_va, dev_ptr
            );
            return -1;
        }

        t.buffer.addr = reinterpret_cast<uint64_t>(dev_ptr);
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
    DeviceArena layout_probe;
    PTO2RuntimeArenaLayout layout = runtime_reserve_layout(layout_probe, eff_task_window_sizes, eff_heap_sizes);

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

    if (host_orch_func_ptr == nullptr) {
        LOG_ERROR("host-orch: orchestration entry points were not resolved");
        return -1;
    }
    if (!start_async_host_graph_pipeline(
            runtime, api, layout, sm_ptr, sm_size, runtime_arena_dev, gm_heap, eff_heap_sizes, eff_task_window_sizes,
            host_orch_func_ptr, device_args
        )) {
        LOG_ERROR("host-orch: failed to prepare the async whole-graph worker");
        return -1;
    }
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
 * @param runtime       Pointer to Runtime
 * @param execution_rc  Status returned by DeviceRunner::run
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

    if (join_async_host_graph_pipeline(runtime) != 0) {
        LOG_ERROR("host-orch: async orchestration failed");
        rc = -1;
    }

    LOG_INFO_V0("=== Copying Results Back to Host ===");

    // Copy all recorded tensors from device back to host
    TensorPair *tensor_pairs = runtime->tensor_pairs_.data();
    int tensor_pair_count = static_cast<int>(runtime->tensor_pairs_.size());

    LOG_INFO_V0("Tensor pairs to process: %d", tensor_pair_count);

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

            int copy_rc = api->copy_from_device(pair.host_ptr, pair.dev_ptr, pair.size);
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
            // Release the SVM host mapping installed at staging time before
            // freeing the device buffer (unregister-before-free, as the HAL
            // requires). No-op on sim. Keyed by dev_ptr.
            api->unregister_device_memory_from_host(tensor_pairs[i].dev_ptr);
            api->device_free(tensor_pairs[i].dev_ptr);
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
