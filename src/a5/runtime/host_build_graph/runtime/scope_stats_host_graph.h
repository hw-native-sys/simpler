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

#pragma once

#include <cstdint>

// Reset host-side capture after the HBG orchestrator has initialized its
// resource pools and before the outer executor scope begins.
void scope_stats_host_graph_begin_capture(int32_t task_window_cap, uint64_t heap_cap, int32_t tensormap_cap);

extern "C" {
bool scope_stats_host_graph_active();
void scope_stats_host_graph_set_enabled(bool enabled);
int scope_stats_host_graph_write_jsonl(const char *output_dir);
}
