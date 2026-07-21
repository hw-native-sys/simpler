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
 * Host-side weak stubs for AICPU-only profiling / dump-args symbols.
 *
 * host_build_graph runs the orchestrator on the host (host-orch-first), so the
 * orchestrator core (pto_orchestrator.cpp / pto_runtime2.cpp) is compiled into
 * libhost_runtime.so, which is dlopen'd RTLD_LOCAL and must therefore resolve
 * all of its symbols. The scope-stats and dump-args collectors are AICPU-only
 * (defined in common/platform/.../aicpu) and are NOT linked into the host
 * library. They record on-device diagnostics; the host orchestrator only builds
 * the task graph, so no-op definitions are correct here.
 *
 * Marked weak + hidden so they never leak into the global dynamic symbol table
 * (RTLD_LOCAL keeps them library-local anyway) and never shadow the AICPU
 * library's strong definitions, mirroring the weak-stub pattern in
 * pto_orchestrator.cpp.
 */

#include "aicpu/scope_stats_collector_aicpu.h"
#include "aicpu/args_dump_aicpu.h"

__attribute__((weak, visibility("hidden"))) bool is_dump_args_enabled() { return false; }

__attribute__((weak, visibility("hidden"))) void set_dump_args_task_mask(uint64_t, ArgsDumpArgMask, ArgsDumpArgMask) {}

__attribute__((weak, visibility("hidden"))) void set_dump_args_task_scalar_dtypes(uint64_t, uint32_t, const uint8_t *) {
}

__attribute__((weak, visibility("hidden"))) void
scope_stats_begin(int, int32_t, int32_t, uint64_t, uint64_t, int32_t, int32_t, int32_t) {}

__attribute__((weak, visibility("hidden"))) void
scope_stats_end(int, int32_t, int32_t, uint64_t, uint64_t, int32_t, int32_t, int32_t) {}

__attribute__((weak, visibility("hidden"))) void scope_stats_on_fatal() {}

__attribute__((weak, visibility("hidden"))) void scope_stats_set_pending_site(const char *, int) {}
