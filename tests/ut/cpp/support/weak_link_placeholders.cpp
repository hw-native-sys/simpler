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
 * Definitions that only have to resolve, never to run.
 *
 * All are weak: a target that links the real translation unit gets the real
 * behaviour, and one that does not still links. Grouped because they share
 * that property, not because they are related.
 */

#include <cstdint>

#include "host/host_graph_runs.h"

// A recorder worker stands its recording storage up as it starts
// (host/graph_recorder_pool.h). The real one is in orchestrator.cpp, which the
// pool's own threading test does not link -- and does not need, since nothing
// it records reaches the storage.
__attribute__((weak)) bool graph_recorder_stand_up_storage() { return true; }

// DeviceRunnerBase::bind_callable_to_runtime (the merged bind facade) calls the
// runtime's bind_callable_to_runtime_impl, which is defined in runtime_maker.cpp
// and only present in the production host_runtime.so. These runner-only unit
// tests link device_runner_base.cpp without any runtime_maker, and their mock
// runners never bind (TestSimRunner::run returns 0), so the impl is never
// invoked — it only has to resolve at link time.
extern "C" __attribute__((weak)) int bind_callable_to_runtime_impl(
    void * /* runtime */, const void * /* api */, const void * /* orch_args */, void * /* host_orch_func_ptr */,
    const void * /* signature */, int /* sig_count */, const uint64_t * /* ring_task_window */,
    const uint64_t * /* ring_heap */, const uint64_t * /* ring_dep_pool */
) {
    return -1;
}

// The host-orchestrating runtime's graph hand-off, which
// device_runner_base.cpp names unconditionally. The real one is in
// host_build_graph's dep_gen_host_graph.cpp, and each arch's device_runner.cpp
// carries this same weak fallback for a runtime that captures on the device;
// runner-only tests link neither. NotCaptured is what a runtime with no host
// graph reports, so a caller that reached it is answered rather than misled.
extern "C" __attribute__((weak)) int dep_gen_host_graph_take(
    simpler::dfx::host_graph::HostGraphExport * /* out */
) {
    return static_cast<int>(simpler::dfx::host_graph::TakeOutcome::NotCaptured);
}
