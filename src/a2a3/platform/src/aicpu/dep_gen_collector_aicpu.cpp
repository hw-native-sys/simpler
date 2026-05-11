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
 * @file dep_gen_collector_aicpu.cpp
 * @brief AICPU-side dep_gen capture implementation
 *
 * Single-instance: dep_gen captures the orchestrator's submit_task stream,
 * so there is one BufferState and one current_buf — no per-core arrays.
 *
 * Buffer switching (SPSC):
 *   - Host pushes free DepGenBuffers via free_queue.
 *   - AICPU pops when current buffer fills; pushes full buffer to per-thread
 *     ready_queue (indexed by orch_thread_idx).
 *   - On free_queue empty or ready_queue full: overwrite current buffer
 *     (record dropped_record_count, keep AICPU alive). Host reads dropped
 *     at finalize to decide whether to emit deps.json.
 */

#include "aicpu/dep_gen_collector_aicpu.h"

#include <cstring>

#include "common/memory_barrier.h"
#include "common/platform_config.h"
#include "common/unified_log.h"

static uint64_t g_platform_dep_gen_base = 0;
static bool g_enable_dep_gen = false;

// File-local cached state for the single dep_gen instance (the orchestrator).
static DepGenDataHeader *s_dep_gen_header = nullptr;
static DepGenBufferState *s_dep_gen_state = nullptr;
static int s_orch_thread_idx = -1;  // set via dep_gen_aicpu_set_orch_thread_idx

extern "C" void set_platform_dep_gen_base(uint64_t dep_gen_data_base) { g_platform_dep_gen_base = dep_gen_data_base; }

extern "C" uint64_t get_platform_dep_gen_base() { return g_platform_dep_gen_base; }

extern "C" void set_dep_gen_enabled(bool enable) { g_enable_dep_gen = enable; }

extern "C" bool is_dep_gen_enabled() { return g_enable_dep_gen; }

void dep_gen_aicpu_set_orch_thread_idx(int thread_idx) { s_orch_thread_idx = thread_idx; }

// ---------------------------------------------------------------------------
// Internal: enqueue full buffer to per-thread ready_queue
// ---------------------------------------------------------------------------

static int enqueue_dep_gen_ready_buffer(uint64_t buffer_ptr, uint32_t buffer_seq) {
    if (s_orch_thread_idx < 0 || s_orch_thread_idx >= PLATFORM_MAX_AICPU_THREADS) {
        return -1;
    }
    int q = s_orch_thread_idx;
    uint32_t capacity = PLATFORM_DEP_GEN_READYQUEUE_SIZE;
    uint32_t current_tail = s_dep_gen_header->queue_tails[q];
    uint32_t current_head = s_dep_gen_header->queue_heads[q];

    uint32_t next_tail = (current_tail + 1) % capacity;
    if (next_tail == current_head) {
        return -1;  // Queue full
    }

    s_dep_gen_header->queues[q][current_tail].instance_index = 0;
    s_dep_gen_header->queues[q][current_tail].buffer_ptr = buffer_ptr;
    s_dep_gen_header->queues[q][current_tail].buffer_seq = buffer_seq;
    s_dep_gen_header->queue_tails[q] = next_tail;
    return 0;
}

// ---------------------------------------------------------------------------
// Internal: switch the current buffer
// ---------------------------------------------------------------------------

static void dep_gen_switch_buffer() {
    if (s_dep_gen_state == nullptr) {
        return;
    }
    DepGenBuffer *full_buf = reinterpret_cast<DepGenBuffer *>(s_dep_gen_state->current_buf_ptr);
    if (full_buf == nullptr) {
        return;
    }

    // Check free_queue before committing the full buffer
    rmb();
    uint32_t head = s_dep_gen_state->free_queue.head;
    uint32_t tail = s_dep_gen_state->free_queue.tail;

    if (head == tail) {
        // No replacement buffer available — overwrite current buffer to keep
        // the orch loop alive; account every record we drop.
        LOG_WARN("dep_gen: no free buffer, overwriting current (dropped %u records)", full_buf->count);
        s_dep_gen_state->dropped_record_count += full_buf->count;
        full_buf->count = 0;
        wmb();
        return;
    }

    uint32_t seq = s_dep_gen_state->current_buf_seq;
    int rc = enqueue_dep_gen_ready_buffer(s_dep_gen_state->current_buf_ptr, seq);
    if (rc != 0) {
        LOG_ERROR("dep_gen: failed to enqueue full buffer (ready_queue full), %u records dropped", full_buf->count);
        s_dep_gen_state->dropped_record_count += full_buf->count;
        full_buf->count = 0;
        wmb();
        return;
    }

    // Pop next buffer from free_queue
    uint64_t new_buf_ptr = s_dep_gen_state->free_queue.buffer_ptrs[head % PLATFORM_DEP_GEN_SLOT_COUNT];
    rmb();
    s_dep_gen_state->free_queue.head = head + 1;
    s_dep_gen_state->current_buf_ptr = new_buf_ptr;
    s_dep_gen_state->current_buf_seq = seq + 1;
    wmb();

    DepGenBuffer *new_buf = reinterpret_cast<DepGenBuffer *>(new_buf_ptr);
    new_buf->count = 0;
}

// ---------------------------------------------------------------------------
// Public interface
// ---------------------------------------------------------------------------

void dep_gen_aicpu_init() {
    void *base = reinterpret_cast<void *>(get_platform_dep_gen_base());
    if (base == nullptr) {
        LOG_ERROR("dep_gen_aicpu_init: dep_gen_data_base is NULL");
        return;
    }
    s_dep_gen_header = get_dep_gen_header(base);
    s_dep_gen_state = get_dep_gen_buffer_state(base, /*instance_index=*/0);

    rmb();
    uint32_t head = s_dep_gen_state->free_queue.head;
    uint32_t tail = s_dep_gen_state->free_queue.tail;

    if (head != tail) {
        uint64_t buf_ptr = s_dep_gen_state->free_queue.buffer_ptrs[head % PLATFORM_DEP_GEN_SLOT_COUNT];
        rmb();
        s_dep_gen_state->free_queue.head = head + 1;
        s_dep_gen_state->current_buf_ptr = buf_ptr;
        s_dep_gen_state->current_buf_seq = 0;
        wmb();
        DepGenBuffer *buf = reinterpret_cast<DepGenBuffer *>(buf_ptr);
        buf->count = 0;
        LOG_INFO_V0("dep_gen: popped initial buffer addr=0x%lx", buf_ptr);
    } else {
        LOG_ERROR("dep_gen: free_queue empty during init");
        s_dep_gen_state->current_buf_ptr = 0;
    }
    wmb();
}

void dep_gen_aicpu_record_submit(
    uint64_t task_id_raw, bool in_manual_scope, int tensor_count, const void *const *tensor_ptrs,
    const uint8_t *arg_types, int explicit_dep_count, const uint64_t *explicit_deps_raw
) {
    if (!g_enable_dep_gen || s_dep_gen_state == nullptr) {
        return;
    }

    // Account every attempted record so total == collected + dropped on host.
    s_dep_gen_state->total_record_count += 1;

    rmb();
    uint64_t cur_ptr = s_dep_gen_state->current_buf_ptr;
    if (cur_ptr == 0) {
        s_dep_gen_state->dropped_record_count += 1;
        wmb();
        return;
    }
    DepGenBuffer *buf = reinterpret_cast<DepGenBuffer *>(cur_ptr);

    if (buf->count >= static_cast<uint32_t>(PLATFORM_DEP_GEN_RECORDS_PER_BUFFER)) {
        dep_gen_switch_buffer();
        rmb();
        cur_ptr = s_dep_gen_state->current_buf_ptr;
        if (cur_ptr == 0) {
            s_dep_gen_state->dropped_record_count += 1;
            wmb();
            return;
        }
        buf = reinterpret_cast<DepGenBuffer *>(cur_ptr);
    }

    uint32_t idx = buf->count;
    DepGenRecord *rec = &buf->records[idx];

    rec->task_id = task_id_raw;
    // Cast the enum to uint32_t before the ternary so Linux GCC's -Wextra
    // does not warn about "enumerated and non-enumerated type in conditional".
    rec->flags = in_manual_scope ? static_cast<uint32_t>(DEP_GEN_FLAG_IN_MANUAL_SCOPE) : 0u;

    int tc = tensor_count;
    if (tc < 0) {
        tc = 0;
    } else if (tc > CORE_MAX_TENSOR_ARGS) {
        // The runtime's Arg also caps at CORE_MAX_TENSOR_ARGS, so this should
        // never trip; clamp defensively to keep the writer crash-free.
        LOG_ERROR("dep_gen: tensor_count %d > CORE_MAX_TENSOR_ARGS (%d), truncating", tc, CORE_MAX_TENSOR_ARGS);
        tc = CORE_MAX_TENSOR_ARGS;
    }
    rec->tensor_count = static_cast<uint16_t>(tc);

    int dc = explicit_dep_count;
    if (dc < 0) {
        dc = 0;
    } else if (dc > DEP_GEN_MAX_EXPLICIT_DEPS) {
        LOG_ERROR(
            "dep_gen: explicit_dep_count %d > DEP_GEN_MAX_EXPLICIT_DEPS (%d), truncating", dc, DEP_GEN_MAX_EXPLICIT_DEPS
        );
        dc = DEP_GEN_MAX_EXPLICIT_DEPS;
    }
    rec->explicit_dep_count = static_cast<uint16_t>(dc);

    // explicit_deps (tail of the entry, packed; replay reads only the first dc entries)
    if (dc > 0 && explicit_deps_raw != nullptr) {
        memcpy(rec->explicit_deps, explicit_deps_raw, static_cast<size_t>(dc) * sizeof(uint64_t));
    }

    // arg_types
    if (tc > 0 && arg_types != nullptr) {
        memcpy(rec->arg_types, arg_types, static_cast<size_t>(tc));
    }

    // tensors[]: per-slot 128-byte blob (or zero if pointer is null — OUTPUT slot)
    if (tc > 0) {
        if (tensor_ptrs == nullptr) {
            memset(rec->tensors, 0, static_cast<size_t>(tc) * DEP_GEN_TENSOR_SIZE);
        } else {
            for (int i = 0; i < tc; i++) {
                if (tensor_ptrs[i] == nullptr) {
                    memset(rec->tensors[i], 0, DEP_GEN_TENSOR_SIZE);
                } else {
                    memcpy(rec->tensors[i], tensor_ptrs[i], DEP_GEN_TENSOR_SIZE);
                }
            }
        }
    }

    buf->count = idx + 1;
    wmb();
}

void dep_gen_aicpu_flush() {
    if (s_dep_gen_header == nullptr || s_dep_gen_state == nullptr) {
        return;
    }

    rmb();
    uint64_t buf_ptr = s_dep_gen_state->current_buf_ptr;
    if (buf_ptr == 0) {
        return;
    }
    DepGenBuffer *buf = reinterpret_cast<DepGenBuffer *>(buf_ptr);
    if (buf->count == 0) {
        return;
    }

    uint32_t seq = s_dep_gen_state->current_buf_seq;
    int rc = enqueue_dep_gen_ready_buffer(buf_ptr, seq);
    if (rc == 0) {
        LOG_INFO_V0("dep_gen: flushed buffer with %u records", buf->count);
        s_dep_gen_state->current_buf_ptr = 0;
        wmb();
    } else {
        LOG_ERROR("dep_gen: flush failed (ready_queue full), %u records dropped", buf->count);
        s_dep_gen_state->dropped_record_count += buf->count;
        buf->count = 0;
        s_dep_gen_state->current_buf_ptr = 0;
        wmb();
    }
}

void dep_gen_aicpu_finalize() {
    // No HW state to restore (unlike PMU). Reset file-local cache for cleanliness
    // — the next init re-resolves these from the (potentially new) base anyway.
    s_dep_gen_header = nullptr;
    s_dep_gen_state = nullptr;
    s_orch_thread_idx = -1;
}
