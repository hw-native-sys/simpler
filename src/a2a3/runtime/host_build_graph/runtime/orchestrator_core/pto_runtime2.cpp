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
 * host_build_graph runtime implementation
 *
 * Implements the unified runtime API that combines orchestrator and scheduler.
 *
 * Based on: docs/RUNTIME_LOGIC.md
 */

#include "pto_runtime2.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <algorithm>

#include "aicpu/device_time.h"
#include "common/platform_config.h"
#include "common/unified_log.h"
#include "host_tensor_access.h"
#if SIMPLER_DFX
#include "aicpu/scope_stats_collector_aicpu.h"
#endif

// ChipTensor-byte access for a caller that can load a device address directly.
// The AICPU build compiles this translation unit and links these; the host
// build overrides them with host/host_tensor_access.cpp, where a device
// address is not loadable in general. Visibility is hidden so the host .so
// does not export them into the global dynamic symbol table (same pattern as
// get_sys_cnt_aicpu above and the dep_gen stubs in pto_orchestrator.cpp).
__attribute__((weak, visibility("hidden"))) bool
host_tensor_read(HostTensorAccessor *, uint64_t dev_addr, void *dst, uint64_t bytes) {
    memcpy(dst, reinterpret_cast<const void *>(dev_addr), bytes);
    return true;
}

__attribute__((weak, visibility("hidden"))) bool
host_tensor_write(HostTensorAccessor *, uint64_t dev_addr, const void *src, uint64_t bytes) {
    memcpy(reinterpret_cast<void *>(dev_addr), src, bytes);
    return true;
}

// Host fallback for the host-orchestration path. The AICPU cycle counter is a
// device register unavailable on the host, so return a monotonic wall-clock
// scaled to that counter's cycle units (PLATFORM_PROF_SYS_CNT_FREQ). The
// cycle-denominated timeouts that run during host orchestration
// (PTO2_TENSOR_DATA_TIMEOUT_CYCLES in wait_for_tensor_ready, the fanin spill
// pool's backstop) then fire at their intended wall-clock; a constant 0 would
// make them no-ops and spin forever. The AICPU build links the strong device
// counter from device_time.cpp; hidden visibility keeps this off the global
// dynamic symbol table.
__attribute__((weak, visibility("hidden"))) uint64_t get_sys_cnt_aicpu() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    // Scale sec and nsec separately (divisor is the constant 1e9): avoids a
    // div-by-zero when PLATFORM_PROF_SYS_CNT_FREQ >= 1 GHz and the truncation
    // error a `1e9 / FREQ` divisor would introduce for non-dividing frequencies.
    return static_cast<uint64_t>(ts.tv_sec) * PLATFORM_PROF_SYS_CNT_FREQ +
           static_cast<uint64_t>(ts.tv_nsec) * PLATFORM_PROF_SYS_CNT_FREQ / 1000000000ull;
}

// Derived here, not in pto_runtime2_types.h: that header is included by orchestrations
// that define PLATFORM_PROF_SYS_CNT_FREQ locally, so pulling the platform header into
// it caused a redefinition conflict (#1189). Scaling MS by the counter frequency (like
// SCHEDULER_TIMEOUT_CYCLES) keeps the data-wait wall-clock identical across arches.
static constexpr uint64_t PTO2_TENSOR_DATA_TIMEOUT_CYCLES =
    (PTO2_TENSOR_DATA_TIMEOUT_MS * PLATFORM_PROF_SYS_CNT_FREQ) / 1000;

// =============================================================================
// Orchestration Ops Table (function-pointer dispatch for orchestration .so)
// =============================================================================

static TaskOutputTensors
submit_task_impl(PTO2Runtime *rt, const MixedKernels &mixed_kernels, const CoreTaskArgs &args) {
    return rt->orchestrator.submit_task(mixed_kernels, args);
}

static TaskOutputTensors alloc_tensors_impl(PTO2Runtime *rt, const CoreTaskArgs &args) {
    return rt->orchestrator.alloc_tensors(args);
}

static TaskOutputTensors submit_dummy_task_impl(PTO2Runtime *rt, const CoreTaskArgs &args) {
    return rt->orchestrator.submit_dummy_task(args);
}

static GraphScopeResult graph_begin_impl(PTO2Runtime *rt, uint64_t graph_key, const GraphTaskArgs &args) {
    if (rt == nullptr) return GraphScopeResult{};
    return rt->orchestrator.graph_begin(graph_key, args, rt->active_callable_hash);
}

static bool graph_prepare_impl(PTO2Runtime *rt, void *recording_handle, const GraphTaskArgs &args) {
    return rt != nullptr && rt->orchestrator.graph_prepare(recording_handle, args);
}

static void graph_abort_impl(PTO2Runtime *rt, void *recording_handle) {
    if (rt != nullptr) rt->orchestrator.graph_abort(recording_handle);
}

static bool graph_end_impl(PTO2Runtime *rt) { return rt != nullptr && rt->orchestrator.graph_end(); }

static void graph_commit_impl(PTO2Runtime *rt) {
    if (rt != nullptr) rt->orchestrator.graph_commit();
}

void rt_scope_begin(PTO2Runtime *rt) {
    PTO2ScopeMode mode = rt->pending_scope_mode;
    rt->pending_scope_mode = PTO2ScopeMode::AUTO;
    rt->orchestrator.begin_scope(mode);
}

void rt_scope_end(PTO2Runtime *rt) { rt->orchestrator.end_scope(); }

void rt_orchestration_done(PTO2Runtime *rt) {
    // Host orchestration calls this runtime entry directly rather than the
    // orchestration-SO wrapper. Commit here as well so an all-Graph entry has a
    // final synchronization point for its asynchronous recording and deferred
    // outer shells even when no later non-Graph task forced an earlier commit.
    rt->orchestrator.graph_commit();
    rt->orchestrator.mark_done();
}

static bool is_fatal_impl(PTO2Runtime *rt) { return rt->orchestrator.fatal; }

void rt_report_fatal(PTO2Runtime *rt, int32_t error_code, const char *func, const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    if (fmt == nullptr || fmt[0] == '\0') {
        rt->orchestrator.report_fatal(error_code, func, nullptr);
    } else {
        char message[1024];
        vsnprintf(message, sizeof(message), fmt, args);
        rt->orchestrator.report_fatal(error_code, func, "%s", message);
    }
    va_end(args);
}

// Validate every producer reference before waiting on its slot. The host builds
// the complete graph before device scheduling starts, so a live device producer
// cannot complete during this call; the timeout remains a defensive failure
// backstop rather than a synchronization mechanism for orchestration code.
// For writes, completed_watermark additionally protects against overwriting a
// producer while one of its submitted consumers is still live.
MAYBE_UNINITIALIZED_BEGIN
static bool
wait_for_tensor_ready(PTO2Runtime *rt, const ChipTensor &tensor, bool wait_for_consumers, const char *caller) {
    PTO2TaskId owner = tensor.owner_task_id;
    PTO2OrchestratorState &orch = rt->orchestrator;

    // Segmented wait: collect up to kSegmentCap producer slots, then flush by
    // spinning on each. When the segment fills, we wait for the accumulated
    // batch before continuing to gather more. Dedup is per-segment only; a
    // producer that appears in two segments is waited on twice, which is
    // idempotent (task_state is monotonic) and only adds one atomic load on
    // the second encounter.
    constexpr int kSegmentCap = 64;
    const PTO2TaskSlotState *seg[kSegmentCap];
    int seg_count = 0;
    bool failed = false;

    // Returns nullptr for every rejected producer, having latched the fatal.
    // Callers branch on the returned pointer, not on `failed`: the slot is
    // dereferenced immediately and a null return is the only safe signal.
    auto resolve_producer = [&](PTO2TaskId producer) -> const PTO2TaskSlotState * {
        if (!producer.is_valid() || producer.ring() != 0) {
            orch.report_fatal(
                PTO2_ERROR_INVALID_ARGS, caller,
                "tensor producer task %#llx has invalid ring %u; host_build_graph uses ring 0",
                static_cast<unsigned long long>(producer.raw), static_cast<unsigned int>(producer.ring())
            );
            failed = true;
            return nullptr;
        }

        auto &ring = orch.sm_header->ring;
        int32_t local_id = static_cast<int32_t>(producer.local());
        int32_t slot_index = ring.get_slot_by_task_id(local_id);
        auto &slot = ring.get_slot_state_by_slot(slot_index);
        if (slot.task == nullptr || slot.task->task_id != producer) {
            orch.report_fatal(
                PTO2_ERROR_INVALID_ARGS, caller,
                "tensor producer task %#llx does not match the descriptor bound to slot %d",
                static_cast<unsigned long long>(producer.raw), slot_index
            );
            failed = true;
            return nullptr;
        }
        return &slot;
    };

    auto wait_one_producer = [&](const PTO2TaskSlotState &slot) {
        uint8_t ring_id = 0;
        int32_t local_id = static_cast<int32_t>(slot.task->task_id.local());
        uint64_t t0 = get_sys_cnt_aicpu();
        int32_t spin_count = 0;
        while (slot.task_state.load(std::memory_order_acquire) < PTO2_TASK_COMPLETED) {
            SPIN_WAIT_HINT();
            if ((++spin_count & 1023) == 0) {
                // A fatal latched elsewhere (e.g. the scheduler-side wiring
                // deadlock detector) breaks this wait; cold path only.
                if (orch.sm_header->orch_error_code.load(std::memory_order_acquire) != PTO2_ERROR_NONE) {
                    failed = true;
                    return;
                }
                if (get_sys_cnt_aicpu() - t0 > PTO2_TENSOR_DATA_TIMEOUT_CYCLES) {
                    orch.report_fatal(
                        PTO2_ERROR_TENSOR_WAIT_TIMEOUT, caller,
                        "Timeout (%llu cycles): producer (ring=%d, local=%d) not completed",
                        (unsigned long long)PTO2_TENSOR_DATA_TIMEOUT_CYCLES, ring_id, local_id
                    );
                    failed = true;
                    return;
                }
            }
        }
    };

    auto wait_one_consumers = [&](const PTO2TaskSlotState &slot) {
        uint8_t ring_id = 0;
        int32_t local_id = slot.task->task_id.local();
        uint64_t t0 = get_sys_cnt_aicpu();
        int32_t spin_count = 0;
        // Polling: all consumers of this producer have retired once the
        // completed_watermark reaches the producer's highest consumer id (set at
        // submit in append_fanin_or_fail). Replaces the fanout_refcount ==
        // fanout_count wiring check, which polling removes.
        PTO2SharedMemoryRingHeader &cons_ring = orch.sm_header->ring;
        while (cons_ring.completed_watermark.load(std::memory_order_acquire) < slot.last_consumer_local_id) {
            SPIN_WAIT_HINT();
            if ((++spin_count & 1023) == 0) {
                // A fatal latched elsewhere (e.g. the scheduler-side wiring
                // deadlock detector) breaks this wait; cold path only.
                if (orch.sm_header->orch_error_code.load(std::memory_order_acquire) != PTO2_ERROR_NONE) {
                    failed = true;
                    return;
                }
                if (get_sys_cnt_aicpu() - t0 > PTO2_TENSOR_DATA_TIMEOUT_CYCLES) {
                    orch.report_fatal(
                        PTO2_ERROR_TENSOR_WAIT_TIMEOUT, caller,
                        "Timeout (%llu cycles): consumers of producer (ring=%d, local=%d) not done",
                        (unsigned long long)PTO2_TENSOR_DATA_TIMEOUT_CYCLES, ring_id, local_id
                    );
                    failed = true;
                    return;
                }
            }
        }
    };

    auto flush_segment = [&]() {
        for (int i = 0; i < seg_count; i++) {
            wait_one_producer(*seg[i]);
            if (failed) return;
            if (!wait_for_consumers) continue;
            wait_one_consumers(*seg[i]);
            if (failed) return;
        }
        seg_count = 0;
    };

    auto try_push = [&](const PTO2TaskSlotState &s) {
        for (int j = 0; j < seg_count; j++) {
            if (seg[j] == &s) return;  // per-segment dedup
        }
        if (seg_count == kSegmentCap) {
            flush_segment();
            if (failed) return;
        }
        seg[seg_count++] = &s;
    };

    auto do_wait = [&]() {
        // Step A: creator retention — read owner directly from tensor metadata
        if (owner.is_valid()) {
            const auto *slot = resolve_producer(owner);
            if (slot == nullptr) return;
            try_push(*slot);
        }

        // Step B: modifier writer lookup (OverlapMap), direct callback
        orch.tensor_map.lookup(tensor, [&](PTO2TensorMapEntry &entry, OverlapStatus) -> bool {
            PTO2TaskId pid = entry.producer_task_id;
            const auto *slot = resolve_producer(pid);
            if (slot == nullptr) return false;
            try_push(*slot);
            return !failed;
        });
        if (failed) return;
        flush_segment();
    };

    do_wait();
    return !failed;
}
MAYBE_UNINITIALIZED_END

uint64_t get_tensor_data(PTO2Runtime *rt, const ChipTensor &tensor, uint32_t ndims, const uint32_t indices[]) {
    if (tensor.buffer.addr == 0) {
        unified_log_error(
            __FUNCTION__, "get_tensor_data: buffer not allocated (addr=0). "
                          "Use the ChipTensor returned by add_output(TensorCreateInfo) after submit returns."
        );
        return 0;
    }

    if (!wait_for_tensor_ready(rt, tensor, false, __FUNCTION__)) {
        return 0;
    }

    uint64_t flat_offset = tensor.compute_flat_offset(indices, ndims);
    uint64_t elem_size = get_element_size(tensor.dtype);
    uint64_t elem_addr = tensor.buffer.addr + flat_offset * elem_size;
    uint64_t result = 0;
    if (!host_tensor_read(rt->tensor_access, elem_addr, &result, elem_size)) {
        rt->orchestrator.report_fatal(
            PTO2_ERROR_INVALID_ARGS, __FUNCTION__,
            "no host view for device address %#llx (%llu bytes): during host orchestration only tensors the "
            "runtime staged are readable, not runtime-created or child-memory buffers",
            (unsigned long long)elem_addr, (unsigned long long)elem_size
        );
        return 0;
    }
    return result;
}

void set_tensor_data(
    PTO2Runtime *rt, const ChipTensor &tensor, uint32_t ndims, const uint32_t indices[], uint64_t value
) {
    if (tensor.buffer.addr == 0) {
        unified_log_error(
            __FUNCTION__, "set_tensor_data: buffer not allocated (addr=0). "
                          "Use the ChipTensor returned by add_output(TensorCreateInfo) after submit returns."
        );
        return;
    }

    // Wait for producer + all consumers before writing (WAW + WAR safety)
    if (!wait_for_tensor_ready(rt, tensor, true, __FUNCTION__)) {
        return;
    }

    uint64_t flat_offset = tensor.compute_flat_offset(indices, ndims);
    uint64_t elem_size = get_element_size(tensor.dtype);
    uint64_t elem_addr = tensor.buffer.addr + flat_offset * elem_size;
    if (!host_tensor_write(rt->tensor_access, elem_addr, &value, elem_size)) {
        rt->orchestrator.report_fatal(
            PTO2_ERROR_INVALID_ARGS, __FUNCTION__,
            "no writable host view for device address %#llx (%llu bytes): during host orchestration only tensors "
            "the runtime staged are writable, not runtime-created or child-memory buffers",
            (unsigned long long)elem_addr, (unsigned long long)elem_size
        );
    }
}

// Ops-table entry that hands the call-site captured by PTO2ScopeGuard to the
// [ScopeStats] collector. The slot is always present in the struct to keep
// the layout stable; at SIMPLER_DFX=0 we fill nullptr so the orchestration
// .so's null-check skips it.
#if SIMPLER_DFX
static void scope_set_site_impl(const char *file, int line) { scope_stats_set_pending_site(file, line); }
#endif

static int32_t available_cluster_count_impl(PTO2Runtime *rt) { return rt->orchestrator.total_cluster_count; }
static int32_t available_aiv_count_impl(PTO2Runtime *rt) { return rt->orchestrator.total_aiv_count; }

static const PTO2RuntimeOps s_runtime_ops = {
    .submit_task = submit_task_impl,
    .scope_begin = rt_scope_begin,
    .scope_end = rt_scope_end,
    .orchestration_done = rt_orchestration_done,
    .is_fatal = is_fatal_impl,
    .report_fatal = rt_report_fatal,
    .log_error = unified_log_error,
    .log_warn = unified_log_warn,
    .log_timing = unified_log_timing,
    .log_info = unified_log_info,
    .log_debug = unified_log_debug,
    .get_tensor_data = get_tensor_data,
    .set_tensor_data = set_tensor_data,
    .alloc_tensors = alloc_tensors_impl,
    .submit_dummy_task = submit_dummy_task_impl,
    .available_cluster_count = available_cluster_count_impl,
    .available_aiv_count = available_aiv_count_impl,
    .graph_begin = graph_begin_impl,
    .graph_prepare = graph_prepare_impl,
    .graph_abort = graph_abort_impl,
    .graph_end = graph_end_impl,
    .graph_commit = graph_commit_impl,
#if SIMPLER_DFX
    .scope_set_site = scope_set_site_impl,
#else
    .scope_set_site = nullptr,
#endif
};

// =============================================================================
// Runtime Lifecycle (AICPU-only fixup)
// =============================================================================
//
// Layout / init_data / wire / destroy live in
// runtime/shared/pto_runtime2_init.cpp so the host build can pre-populate the
// prebuilt arena image. The pieces below — wiring the ops table and the
// SPMD core counts — depend on the device-side s_runtime_ops global and the
// AICPU SchedulerContext respectively, so they remain in the AICPU build.

void runtime_finalize_after_wire(PTO2Runtime *rt, int32_t aic_count, int32_t aiv_count) {
    rt->ops = &s_runtime_ops;
    rt->orchestrator.total_cluster_count = aic_count;
    rt->orchestrator.total_aiv_count = aiv_count;
}

void runtime_set_mode(PTO2Runtime *rt, PTO2RuntimeMode mode) {
    if (rt) {
        rt->mode = mode;
    }
}
