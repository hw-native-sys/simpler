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

#include "scope_stats_host_graph.h"

#include "host/scope_stats_host_capture.h"

namespace {
thread_local ScopeStatsHostCapture g_scope_stats_capture;
}  // namespace

void scope_stats_host_graph_begin_capture(int32_t task_window_cap, uint64_t heap_cap, int32_t tensormap_cap) {
    g_scope_stats_capture.begin_capture(task_window_cap, heap_cap, tensormap_cap);
}

extern "C" bool scope_stats_host_graph_active() { return true; }

extern "C" void scope_stats_host_graph_set_enabled(bool enabled) { g_scope_stats_capture.set_enabled(enabled); }

extern "C" int scope_stats_host_graph_write_jsonl(const char *output_dir) {
    return g_scope_stats_capture.write_jsonl(output_dir);
}

extern "C" bool is_scope_stats_enabled() { return g_scope_stats_capture.enabled(); }

extern "C" void scope_stats_set_pending_site(const char *file, int line) {
    g_scope_stats_capture.set_pending_site(file, line);
}

extern "C" void scope_stats_begin(
    int ring_id, int32_t task_start, int32_t task_end, uint64_t heap_start, uint64_t heap_end, int32_t dep_pool_start,
    int32_t dep_pool_end, int32_t tensormap_used
) {
    g_scope_stats_capture.begin(
        ring_id, task_start, task_end, heap_start, heap_end, dep_pool_start, dep_pool_end, tensormap_used
    );
}

extern "C" void scope_stats_end(
    int ring_id, int32_t task_start, int32_t task_end, uint64_t heap_start, uint64_t heap_end, int32_t dep_pool_start,
    int32_t dep_pool_end, int32_t tensormap_used
) {
    g_scope_stats_capture.end(
        ring_id, task_start, task_end, heap_start, heap_end, dep_pool_start, dep_pool_end, tensormap_used
    );
}

extern "C" void scope_stats_note_heap_wrap(int side) { g_scope_stats_capture.note_heap_wrap(side); }

extern "C" void scope_stats_on_fatal() { g_scope_stats_capture.on_fatal(); }
