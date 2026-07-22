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

#include "pto_shared_memory.h"

#include <inttypes.h>
#include <stdlib.h>
#include <string.h>

#include "common/unified_log.h"

uint64_t PTO2SharedMemoryHandle::calculate_size(uint64_t task_window_size) {
    return pto2_sm_layout::segment_offsets(task_window_size).end;
}

void PTO2SharedMemoryHandle::setup_pointers(uint64_t task_window_size) {
    header = static_cast<PTO2SharedMemoryHeader *>(sm_base);
    auto offsets = pto2_sm_layout::segment_offsets(task_window_size);
    auto *base = static_cast<char *>(sm_base);
    header->task_descriptors = reinterpret_cast<PTO2TaskDescriptor *>(base + offsets.descriptors);
    header->task_payloads = reinterpret_cast<PTO2TaskPayload *>(base + offsets.payloads);
    header->slot_states = reinterpret_cast<PTO2TaskSlotState *>(base + offsets.slot_states);
    header->task_slot_map = reinterpret_cast<int32_t *>(base + offsets.task_slot_map);
}

bool PTO2SharedMemoryHandle::init(
    void *sm_base_arg, uint64_t sm_size_arg, uint64_t task_window_size, uint64_t heap_size
) {
    if (!sm_base_arg || sm_size_arg < calculate_size(task_window_size)) return false;
    sm_base = sm_base_arg;
    sm_size = sm_size_arg;
    is_owner = false;
    setup_pointers(task_window_size);
    init_header(task_window_size, heap_size);
    return true;
}

PTO2SharedMemoryHandle *PTO2SharedMemoryHandle::create_and_init_default(DeviceArena &arena) {
    const uint64_t buffer_size = calculate_size(PTO2_TASK_WINDOW_SIZE);
    const size_t off_handle = arena.reserve(sizeof(PTO2SharedMemoryHandle), alignof(PTO2SharedMemoryHandle));
    const size_t off_buffer = arena.reserve(static_cast<size_t>(buffer_size), PTO2_ALIGN_SIZE);
    if (arena.commit() == nullptr) return nullptr;

    auto *handle = static_cast<PTO2SharedMemoryHandle *>(arena.region_ptr(off_handle));
    memset(handle, 0, sizeof(*handle));
    void *buffer = arena.region_ptr(off_buffer);
    memset(buffer, 0, static_cast<size_t>(buffer_size));
    if (!handle->init(buffer, buffer_size, PTO2_TASK_WINDOW_SIZE, PTO2_HEAP_SIZE)) return nullptr;
    return handle;
}

void PTO2SharedMemoryHandle::destroy() {
    if (is_owner && sm_base) {
        free(sm_base);
        free(this);
    }
}

void PTO2SharedMemoryHandle::init_header(uint64_t task_window_size, uint64_t heap_size) {
    header->fc.init();
    header->orchestrator_done.store(0, std::memory_order_relaxed);
    header->task_window_size = task_window_size;
    header->task_window_mask = static_cast<int32_t>(task_window_size - 1);
    header->heap_size = heap_size;
    header->task_descriptors_offset = pto2_sm_layout::segment_offsets(task_window_size).descriptors;
    header->total_size = sm_size;
    header->graph_output_ptr.store(0, std::memory_order_relaxed);
    header->graph_output_size.store(0, std::memory_order_relaxed);
    header->orch_error_code.store(PTO2_ERROR_NONE, std::memory_order_relaxed);
    header->sched_error_bitmap.store(0, std::memory_order_relaxed);
    header->sched_error_code.store(PTO2_ERROR_NONE, std::memory_order_relaxed);
    header->sched_error_thread.store(-1, std::memory_order_relaxed);
    header->sched_stall_detail.store(PTO2_STALL_DETAIL_NONE, std::memory_order_relaxed);
    header->sched_stall_completed.store(0, std::memory_order_relaxed);
    header->sched_stall_total.store(0, std::memory_order_relaxed);
    header->sched_stall_cnt_running.store(0, std::memory_order_relaxed);
    header->sched_stall_cnt_ready.store(0, std::memory_order_relaxed);
    header->sched_stall_cnt_waiting.store(0, std::memory_order_relaxed);
    header->sched_stall_orch_done.store(0, std::memory_order_relaxed);
    header->sched_stall_task_id.store(-1, std::memory_order_relaxed);
    header->sched_stall_core.store(-1, std::memory_order_relaxed);

    // Reset the full slot window before the first graph. Reused graph arenas
    // reset individual slots again during task preparation.
    for (uint64_t i = 0; i < task_window_size; i++) {
        PTO2TaskSlotState &slot = header->slot_states[i];
        slot.fanout_head.store(nullptr, std::memory_order_relaxed);
        slot.task_state.store(PTO2_TASK_PENDING, std::memory_order_relaxed);
        slot.fanin_refcount.store(0, std::memory_order_relaxed);
        slot.fanin_count = 0;
        slot.payload = nullptr;
        slot.task = nullptr;
        slot.active_mask = ActiveMask{};
        slot.bind_ring(0);
        slot.allow_early_resolve = false;
        slot.ready_state.store(PTO2_READY_UNCLAIMED, std::memory_order_relaxed);
        slot.completed_subtasks.store(0, std::memory_order_relaxed);
        slot.total_required_subtasks = 0;
        slot.logical_block_num = 1;
        slot.next_block_idx.store(0, std::memory_order_relaxed);
        header->task_slot_map[i] = static_cast<int32_t>(i);
    }
}

void PTO2SharedMemoryHandle::print_layout() {
    if (!header) return;
    LOG_INFO_V0("=== PTO2 Replay Graph Shared Memory ===");
    LOG_INFO_V0("Base address:       %p", sm_base);
    LOG_INFO_V0("Total size:         %" PRIu64 " bytes", header->total_size);
    LOG_INFO_V0("Task window:        %" PRIu64, header->task_window_size);
    LOG_INFO_V0("Heap size:          %" PRIu64 " bytes", header->heap_size);
    LOG_INFO_V0("Task count:         %d", header->fc.task_count.load(std::memory_order_acquire));
    LOG_INFO_V0("Orchestrator done:  %d", header->orchestrator_done.load(std::memory_order_acquire));
    LOG_INFO_V0("Orch error:         %d", header->orch_error_code.load(std::memory_order_relaxed));
    LOG_INFO_V0("Sched error:        %d", header->sched_error_code.load(std::memory_order_relaxed));
}

bool PTO2SharedMemoryHandle::validate() { return sm_base && header && header->fc.validate(this); }

bool PTO2FlowControl::validate(PTO2SharedMemoryHandle *handle) const {
    if (!handle || !handle->header) return false;
    const PTO2SharedMemoryHeader *h = handle->header;
    if (h->task_descriptors_offset >= h->total_size) return false;
    if (reinterpret_cast<uintptr_t>(h->task_descriptors) % PTO2_ALIGN_SIZE != 0) return false;
    return task_count.load(std::memory_order_acquire) >= 0;
}
