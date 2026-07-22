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
#include <time.h>
#include <unistd.h>

#include <cerrno>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include "../common/pto_runtime_status.h"
#include "../runtime/common.h"
#include "../runtime/pto_orchestrator.h"
#include "../runtime/pto_graph_execution.h"
#include "../runtime/pto_runtime2.h"
#include "../runtime/pto_shared_memory.h"
#include "../runtime/pto_types.h"
#include "../runtime/runtime.h"
#include "../../../../common/runtime_status/error_log.h"
#include "../../../../common/task_interface/call_config.h"
#include "callable.h"
#include "common/platform_config.h"
#include "common/unified_log.h"
#include "utils/device_arena.h"
#include "prepare_callable_common.h"

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

static uint64_t host_prof_cycles() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * PLATFORM_PROF_SYS_CNT_FREQ +
           static_cast<uint64_t>(ts.tv_nsec) * PLATFORM_PROF_SYS_CNT_FREQ / 1000000000ull;
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
    PTO2SharedMemoryHandle &host_sm_handle, PTO2Runtime *rt, uint64_t host_sm, uint64_t sm_size, int64_t sm_delta,
    uint64_t host_arena, uint64_t arena_size, int64_t arena_delta
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
                reloc(ss->task);
                reloc(ss->payload);
                reloc(ss->fanout_head);

                PTO2TaskPayload *payload = &ring.task_payloads[slot];
                int32_t nf = payload->fanin_actual_count;
                if (nf > PTO2_FANIN_INLINE_CAP) {
                    // host-orch does not yet relocate the multi-fanin spill pool,
                    // so a task with more than the inline cap of producers would
                    // ship un-relocated host pointers to the device. Fail loud
                    // rather than silently clamp (tensormap handles this via the
                    // on-device spill pool; that path is not wired here yet).
                    LOG_ERROR(
                        "host-orch: task slot %d has fanin %d > inline cap %d; multi-fanin spill relocation is not "
                        "implemented",
                        slot, nf, PTO2_FANIN_INLINE_CAP
                    );
                    ok = false;
                    nf = PTO2_FANIN_INLINE_CAP;
                }
                for (int32_t i = 0; i < nf; i++) {
                    reloc(payload->fanin_inline_slot_states[i]);
                }
            }
        }
    }

    // Per-ring fanout adjacency (dep_pool entries) built inline during host
    // submit (orch_wire_fanin_task). Each live entry [tail, top) carries a
    // consumer slot_state pointer (into the SM) and a next pointer (into the
    // arena).
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        PTO2DepListPool &dp = rt->scheduler.ring_sched_state.dep_pool;
        if (dp.base == nullptr || dp.capacity == 0) {
            continue;
        }
        for (int32_t i = dp.tail; i < dp.top; i++) {
            PTO2DepListEntry &e = dp.base[i % dp.capacity];
            reloc(e.slot_state);
            reloc(e.next);
        }
    }

    // Ready queues seeded by push_ready_routed on the host. Each populated slot
    // [dequeue_pos, enqueue_pos) holds a slot_state pointer into the SM.
    auto reloc_ready = [&](PTO2ReadyQueue &q) {
        if (q.slots == nullptr) {
            return;
        }
        uint64_t enq = q.enqueue_pos.load(std::memory_order_relaxed);
        uint64_t deq = q.dequeue_pos.load(std::memory_order_relaxed);
        for (uint64_t pos = deq; pos < enq; pos++) {
            reloc(q.slots[pos & q.mask].slot_state);
        }
    };
    for (int i = 0; i < PTO2_NUM_RESOURCE_SHAPES; i++) {
        reloc_ready(rt->scheduler.ready_queues[i]);
        reloc_ready(rt->scheduler.ready_sync_queues[i]);
    }
    reloc_ready(rt->scheduler.dummy_ready_queue);
    reloc_ready(rt->scheduler.graph_ready_queue);
    reloc_ready(rt->scheduler.graph_prepare_queue);
    for (int i = 0; i < PTO2_NUM_RESOURCE_SHAPES; i++) {
        reloc_ready(rt->scheduler.early_dispatch_queues[i]);
    }
    return ok;
}

static void discard_pending_graph_submissions(PTO2GraphSubmission *submission) {
    while (submission != nullptr) {
        PTO2GraphSubmission *next = submission->upload_next;
        pto2_graph_submission_discard(submission);
        submission = next;
    }
}

// Graph definitions live in a process-local cache, while each invocation's
// compact execution image must be device-visible. Upload only the bytes used
// by that invocation, rewrite the single outer Graph descriptor to the device
// image, and return the host block to its bounded pool immediately.
static bool upload_pending_graph_executions(
    Runtime *runtime, const HostApi *api, PTO2Runtime *rt, uintptr_t host_sm_begin, size_t host_sm_size,
    intptr_t sm_delta
) {
    PTO2GraphSubmission *submission = rt->orchestrator.pending_graph_uploads;
    rt->orchestrator.pending_graph_uploads = nullptr;

    while (submission != nullptr) {
        PTO2GraphSubmission *next = submission->upload_next;
        PTO2TaskSlotState *outer_slot = submission->outer_slot;
        PTO2TaskDescriptor *outer_task = outer_slot == nullptr ? nullptr : outer_slot->task;
        const size_t bytes = pto2_graph_submission_allocation_bytes(submission);
        if (outer_task == nullptr || outer_task->kind != PTO2TaskKind::GRAPH || bytes == 0) {
            LOG_ERROR("host-orch: invalid pending Graph submission image");
            pto2_graph_submission_discard(submission);
            discard_pending_graph_submissions(next);
            return false;
        }

        void *device_submission = api->device_malloc(bytes);
        if (device_submission == nullptr) {
            LOG_ERROR("host-orch: failed to allocate %zu bytes for Graph submission", bytes);
            pto2_graph_submission_discard(submission);
            discard_pending_graph_submissions(next);
            return false;
        }

        if (!pto2_graph_definition_relocate_for_upload(submission, device_submission) ||
            !pto2_graph_submission_relocate_for_upload(
                submission, device_submission, host_sm_begin, host_sm_size, sm_delta
            ) ||
            api->copy_to_device(device_submission, submission, bytes) != 0) {
            LOG_ERROR("host-orch: failed to relocate or upload Graph submission image");
            api->device_free(device_submission);
            pto2_graph_submission_discard(submission);
            discard_pending_graph_submissions(next);
            return false;
        }

        outer_task->graph_execution = device_submission;
        runtime->tensor_pairs_.push_back({nullptr, device_submission, bytes, false});
        pto2_graph_submission_release_uploaded(submission);
        submission = next;
    }
    return true;
}

int32_t run_host_orchestration(
    Runtime *runtime, const HostApi *api, PTO2Runtime *rt, DeviceArena &host_arena,
    const PTO2RuntimeArenaLayout &layout, void *device_sm, uint64_t sm_size, void *device_arena, void *gm_heap,
    const uint64_t eff_heap_sizes[PTO2_MAX_RING_DEPTH], const uint64_t eff_task_window_sizes[PTO2_MAX_RING_DEPTH],
    void *host_orch_func_ptr, const L2TaskArgs &orch_l2, bool capture_host_orch
) {
    std::vector<uint8_t> host_sm_buf(sm_size, 0);
    void *host_sm = host_sm_buf.data();

    // Re-point the orchestrator half at the host SM (scheduler keeps device SM).
    // init_data_from_layout resets the orchestrator state, so this is safe.
    if (!rt->orchestrator.init_data_from_layout(
            layout.orch, host_arena, host_sm, gm_heap, eff_heap_sizes[0], eff_task_window_sizes[0]
        )) {
        LOG_ERROR("host-orch: orchestrator re-init against host SM failed");
        return -1;
    }
    rt->orchestrator.wire_arena_pointers(layout.orch, host_arena, &rt->scheduler);

    // Initialize the host SM header (ring flow control) so submit_task can run.
    PTO2SharedMemoryHandle host_sm_handle;
    if (!host_sm_handle.init_per_ring(host_sm, sm_size, eff_task_window_sizes, eff_heap_sizes)) {
        LOG_ERROR("host-orch: host SM init_per_ring failed");
        return -1;
    }

    // Install the ops table (host s_runtime_ops). The SPMD core counts are
    // re-applied with the real device values on the AICPU at boot; the values
    // here only feed cluster spreading during this host submit and are unused
    // by the migrated non-cluster examples.
    runtime_finalize_after_wire(rt, /*aic*/ 24, /*aiv*/ 48);
    rt->mode = PTO2_MODE_EXECUTE;
    // get_tensor_data/set_tensor_data dereference buffer.addr directly: the
    // input tensors were mapped into host address space at staging time
    // (HostApi::register_device_memory_to_host), so the host orchestrator can
    // read control tensors (e.g. paged_attention's context_lens/block_table) in
    // place.

    // Bind both framework_current_runtime instances: the host library's (used by
    // rt_scope_* / rt_orchestration_done) and the orch .so's own copy (used by
    // its inline rt_submit_* -> current_runtime()).
    const HostOrchEntryPoints *eps = reinterpret_cast<const HostOrchEntryPoints *>(host_orch_func_ptr);
    rt->active_callable_hash = reinterpret_cast<uint64_t>(eps->entry);
    framework_bind_runtime(rt);
    if (eps->bind != nullptr) {
        eps->bind(rt);
    } else {
        LOG_ERROR("host-orch: orch .so framework_bind_runtime was not resolved");
        return -1;
    }

    // Host orchestration precedes collector initialization. At level 4 keep
    // both the whole-orchestration envelope and one record per task/Graph
    // submit. Point the current orchestrator state at the per-run capture
    // buffer before entering the user function and retain the populated prefix
    // afterwards.
    runtime->host_orch_phase_records_.clear();
    runtime->host_orch_start_cycles_ = 0;
    runtime->host_orch_end_cycles_ = 0;
#if SIMPLER_DFX
    if (capture_host_orch) {
        rt->orchestrator.l2_swimlane_level = L2SwimlaneLevel::ORCH_PHASES;
        size_t record_capacity = static_cast<size_t>(eff_task_window_sizes[0]);
        if (record_capacity > PLATFORM_PHASE_RECORDS_PER_THREAD) {
            record_capacity = PLATFORM_PHASE_RECORDS_PER_THREAD;
        }
        runtime->host_orch_phase_records_.resize(record_capacity);
        rt->orchestrator.host_orch_phase_records = runtime->host_orch_phase_records_.data();
        rt->orchestrator.host_orch_phase_capacity = record_capacity;
        rt->orchestrator.host_orch_phase_count = 0;
        runtime->host_orch_start_cycles_ = host_prof_cycles();
    }
#endif
    rt_scope_begin(rt);
    eps->entry(orch_l2);
    rt_scope_end(rt);
    rt_orchestration_done(rt);
#if SIMPLER_DFX
    if (capture_host_orch) {
        runtime->host_orch_end_cycles_ = host_prof_cycles();
        size_t record_count = rt->orchestrator.host_orch_phase_count;
        if (record_count > runtime->host_orch_phase_records_.size()) {
            record_count = runtime->host_orch_phase_records_.size();
        }
        runtime->host_orch_phase_records_.resize(record_count);
        rt->orchestrator.host_orch_phase_records = nullptr;
        rt->orchestrator.host_orch_phase_capacity = 0;
        rt->orchestrator.host_orch_phase_count = 0;
    }
#endif

    int32_t total_tasks = pto2_sm_layout::ring_current_task_index_addr(host_sm)->load(std::memory_order_acquire);

    // Relocate the host-DDR cross-task pointers to their final DEVICE addresses
    // on the host, before the SM and arena leave for the device. Pointers into
    // the SM shift by sm_delta; pointers into the arena (fanout adjacency, wiring
    // queue) shift by arena_delta. After this both the SM and arena carry device
    // addresses, so the device boots scheduler-only.
    const int64_t sm_delta = static_cast<int64_t>(reinterpret_cast<uint64_t>(device_sm)) -
                             static_cast<int64_t>(reinterpret_cast<uint64_t>(host_sm));
    const int64_t arena_delta = static_cast<int64_t>(reinterpret_cast<uint64_t>(device_arena)) -
                                static_cast<int64_t>(reinterpret_cast<uint64_t>(host_arena.base()));
    if (!upload_pending_graph_executions(runtime, api, rt, reinterpret_cast<uintptr_t>(host_sm), sm_size, sm_delta)) {
        return -1;
    }
    if (!relocate_host_orch_image(
            host_sm_handle, rt, reinterpret_cast<uint64_t>(host_sm), sm_size, sm_delta,
            reinterpret_cast<uint64_t>(host_arena.base()), layout.arena_size, arena_delta
        )) {
        LOG_ERROR("host-orch: relocation failed; refusing to H2D an image with unrelocated host pointers");
        return -1;
    }

    if (api->copy_to_device(device_sm, host_sm, sm_size) != 0) {
        LOG_ERROR("host-orch: H2D of populated SM failed");
        return -1;
    }
    return total_tasks;
}

}  // namespace

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
    const uint64_t *ring_dep_pool, int32_t l2_swimlane_level
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
    DeviceArena host_arena;  // libc malloc backend by default
    PTO2RuntimeArenaLayout layout =
        runtime_reserve_layout(host_arena, eff_task_window_sizes, eff_heap_sizes, eff_dep_pool_capacities);
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

    // host_build_graph host-orch: run the orchestrator on the host now, against
    // a host SM mirror, and ship the populated SM to the device. The arena
    // (copied to the device below) carries the resulting orchestrator/scheduler
    // state; the device boots scheduler-only. register_callable_impl guarantees
    // host_orch_func_ptr is non-null on success (it fails the whole prepare
    // otherwise), so this is an assertion-style guard, not a fallback path.
    if (host_orch_func_ptr == nullptr) {
        LOG_ERROR("host-orch: orchestration entry points were not resolved");
        return -1;
    }
    {
        L2TaskArgs orch_l2;
        orch_l2.create_from_chip_args(device_args);
        int32_t total_tasks = run_host_orchestration(
            runtime, api, rt, host_arena, layout, sm_ptr, sm_size, runtime_arena_dev, gm_heap, eff_heap_sizes,
            eff_task_window_sizes, host_orch_func_ptr, orch_l2,
            l2_swimlane_level >= static_cast<int32_t>(L2SwimlaneLevel::ORCH_PHASES)
        );
        if (total_tasks < 0) {
            LOG_ERROR("host-orch: orchestration run failed");
            return -1;
        }
        runtime->host_total_tasks = total_tasks;
        LOG_INFO_V0("host-orch: submitted %d tasks on host", total_tasks);
    }

    // Stash the layout inside the PTO2Runtime image so the AICPU can recover
    // every arena-internal offset after rtMemcpy. The runtime arena's device
    // base does NOT travel in this image — it's on the host Runtime
    // (set_prebuilt_arena below), since the AICPU needs that pointer
    // *before* it can dereference the image.
    rt->prebuilt_layout = layout;

    int rc_upload = api->copy_to_device(runtime_arena_dev, host_arena.base(), layout.arena_size);
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
            if (tensor_pairs[i].host_ptr != nullptr) {
                api->unregister_device_memory_from_host(tensor_pairs[i].dev_ptr);
            }
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
