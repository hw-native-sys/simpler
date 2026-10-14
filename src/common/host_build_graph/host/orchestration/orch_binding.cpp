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
 * Which orchestration context the inline rt_* wrappers run against.
 *
 * Compiled into the orchestration .so, where the host resolves
 * framework_bind_orch_context by dlsym and calls it once per bind. The host
 * library links its own copy of this TU, which nothing reads: only sources that
 * include orchestration_api.h reach the accessor, and nothing outside the .so
 * does.
 */
#include "host_build_graph/common.h"

namespace {
// Plain global (not thread_local) to avoid glibc TLSDESC stale-resolution
// crash (BZ #32412) when the orchestration SO is dlclose'd/re-dlopen'd
// between execution rounds. All orchestrator threads bind the same context
// value, so per-thread storage is unnecessary.
OrchContext *g_current_orch_context = nullptr;
}  // namespace

extern "C" __attribute__((visibility("default"))) void framework_bind_orch_context(OrchContext *ctx) {
    g_current_orch_context = ctx;
}

// Keep the accessor local to this .so so orchestration helpers do not
// accidentally bind to a same-named symbol in the loading binary.
extern "C" __attribute__((visibility("hidden"))) OrchContext *framework_current_orch_context() {
    return g_current_orch_context;
}
