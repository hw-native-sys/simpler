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
#include "aicpu/scope_stats_collector_aicpu.h"

// Minimal host-orchestrator link targets omit the host capture implementation.
__attribute__((weak, visibility("hidden"))) void
scope_stats_begin(int, int32_t, int32_t, uint64_t, uint64_t, int32_t, int32_t, int32_t) {}

__attribute__((weak, visibility("hidden"))) void
scope_stats_end(int, int32_t, int32_t, uint64_t, uint64_t, int32_t, int32_t, int32_t) {}

__attribute__((weak, visibility("hidden"))) void scope_stats_on_fatal() {}

__attribute__((weak, visibility("hidden"))) void scope_stats_set_pending_site(const char *, int) {}
