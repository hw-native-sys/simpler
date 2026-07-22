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

#include "pto_runtime2.h"
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include "aicpu/device_time.h"
#include "common/platform_config.h"  // PLATFORM_PROF_SYS_CNT_FREQ (data-wait deadline)
#include "common/unified_log.h"
#include "aicpu/scope_stats_collector_aicpu.h"

void runtime_set_mode(PTO2Runtime *rt, PTO2RuntimeMode mode) {
    if (rt) rt->mode = mode;
}

void rt_scope_begin(PTO2Runtime *rt) {
    PTO2ScopeMode mode = rt->pending_scope_mode;
    rt->pending_scope_mode = PTO2ScopeMode::AUTO;
    rt->orchestrator.begin_scope(mode);
}

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

MAYBE_UNINITIALIZED_BEGIN
inline bool wait_for_tensor_ready(PTO2Runtime *rt, const Tensor &tensor, bool wait_for_consumers, const char *caller) {
    PTO2TaskId owner = tensor.owner_task_id;
    PTO2OrchestratorState &orch = rt->orchestrator;

    constexpr int kSegmentCap = 64;
    const PTO2TaskSlotState *seg[kSegmentCap];
    int seg_count = 0;
    bool signaled = false;
    bool failed = false;

    auto wait_one_producer = [&](const PTO2TaskSlotState &slot) {
        uint8_t ring_id = slot.ring_id;
        int32_t local_id = static_cast<int32_t>(slot.task->task_id.local());
        auto &ring_hdr = orch.sm_header->rings[ring_id];
        const int32_t mask = ring_hdr.task_window_mask;
        uint64_t t0 = get_sys_cnt_aicpu();
        int32_t spin_count = 0;
        // (m) Use completion_flags as the single completion signal.
        while (ring_hdr.completion_flags[local_id & mask].load(std::memory_order_acquire) == 0) {
            SPIN_WAIT_HINT();
            if ((++spin_count & 1023) == 0 && get_sys_cnt_aicpu() - t0 > PTO2_TENSOR_DATA_TIMEOUT_CYCLES) {
                orch.report_fatal(
                    PTO2_ERROR_TENSOR_WAIT_TIMEOUT, caller,
                    "Timeout (%llu cycles): producer (ring=%d, local=%d) not completed",
                    (unsigned long long)PTO2_TENSOR_DATA_TIMEOUT_CYCLES, ring_id, local_id
                );
                failed = true;
                return;
            }
        }
    };

    auto wait_one_consumers = [&](const PTO2TaskSlotState &slot) {
        uint8_t ring_id = slot.ring_id;
        int32_t local_id = slot.task->task_id.local();
        // With watermark-based reclamation, "all consumers done" means the
        // per-ring completed_watermark has reached this slot's recorded
        // last_consumer_local_id.
        PTO2SharedMemoryRingHeader &ring_hdr = rt->orchestrator.sm_header->rings[ring_id];
        int32_t target = slot.last_consumer_local_id;
        uint64_t t0 = get_sys_cnt_aicpu();
        int32_t spin_count = 0;
        while (ring_hdr.completed_watermark.load(std::memory_order_acquire) < target) {
            SPIN_WAIT_HINT();
            if ((++spin_count & 1023) == 0 && get_sys_cnt_aicpu() - t0 > PTO2_TENSOR_DATA_TIMEOUT_CYCLES) {
                orch.report_fatal(
                    PTO2_ERROR_TENSOR_WAIT_TIMEOUT, caller,
                    "Timeout (%llu cycles): consumers of producer (ring=%d, local=%d) not done",
                    (unsigned long long)PTO2_TENSOR_DATA_TIMEOUT_CYCLES, ring_id, local_id
                );
                failed = true;
                return;
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
        for (int j = 0; j < seg_count; j++)
            if (seg[j] == &s) return;
        if (seg_count == kSegmentCap) {
            flush_segment();
            if (failed) return;
        }
        seg[seg_count++] = &s;
        if (!signaled) {
            orch.scheduler->wiring.orch_needs_drain.store(true, std::memory_order_release);
            signaled = true;
        }
    };

    auto do_wait = [&]() {
        if (owner.is_valid()) {
            auto &s = orch.sm_header->rings[owner.ring()].get_slot_state_by_task_id(owner.local());
            try_push(s);
            if (failed) return;
        }

        orch.tensor_map.lookup(tensor, [&](PTO2TensorMapEntry &entry, OverlapStatus) -> bool {
            PTO2TaskId pid = entry.producer_task_id;
            auto &s = orch.sm_header->rings[pid.ring()].get_slot_state_by_task_id(pid.local());
            try_push(s);
            return !failed;
        });
        if (failed) return;
        flush_segment();
    };

    do_wait();
    if (signaled) orch.scheduler->wiring.orch_needs_drain.store(false, std::memory_order_release);
    return !failed;
}

MAYBE_UNINITIALIZED_END

inline uint64_t get_tensor_data(PTO2Runtime *rt, const Tensor &tensor, uint32_t ndims, const uint32_t indices[]) {
    if (tensor.buffer.addr == 0) return 0;

    if (!wait_for_tensor_ready(rt, tensor, false, __FUNCTION__)) return 0;

    uint64_t flat_offset = tensor.compute_flat_offset(indices, ndims);
    uint64_t elem_size = get_element_size(tensor.dtype);
    const void *ptr = reinterpret_cast<const void *>(tensor.buffer.addr + flat_offset * elem_size);
    uint64_t result = 0;
    memcpy(&result, ptr, elem_size);
    return result;
}

void set_tensor_data(PTO2Runtime *rt, const Tensor &tensor, uint32_t ndims, const uint32_t indices[], uint64_t value) {
    if (tensor.buffer.addr == 0) return;

    // Wait for producer + all consumers before writing (WAW + WAR safety)
    if (!wait_for_tensor_ready(rt, tensor, true, __FUNCTION__)) return;

    uint64_t flat_offset = tensor.compute_flat_offset(indices, ndims);
    uint64_t elem_size = get_element_size(tensor.dtype);
    void *ptr = reinterpret_cast<void *>(tensor.buffer.addr + flat_offset * elem_size);
    memcpy(ptr, &value, elem_size);
}

TaskOutputTensors submit_task_impl(PTO2Runtime *rt, const MixedKernels &mixed_kernels, const L0TaskArgs &args) {
    return rt->orchestrator.submit_task(mixed_kernels, args);
}

TaskOutputTensors alloc_tensors_impl(PTO2Runtime *rt, const L0TaskArgs &args) {
    return rt->orchestrator.alloc_tensors(args);
}

TaskOutputTensors submit_dummy_task_impl(PTO2Runtime *rt, const L0TaskArgs &args) {
    return rt->orchestrator.submit_dummy_task(args);
}

void runtime_finalize_after_wire(PTO2Runtime *rt, int32_t aic_count, int32_t aiv_count) {
    rt->ops = &s_runtime_ops;
    rt->orchestrator.total_cluster_count = aic_count;
    rt->orchestrator.total_aiv_count = aiv_count;
}
