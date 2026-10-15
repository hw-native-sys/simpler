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
// tests link device_runner_base.cpp without any runtime_maker; a mock runner
// that never binds (TestSimRunner::run returns 0) never invokes this, so it
// only has to resolve at link time, and a case that does bind overrides it as
// described below.
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

// The rest of the runtime-maker surface `c_api_shared.cpp` names
// unconditionally. The real definitions live in each runtime's
// runtime_maker.cpp, which only the production host_runtime.so links. These
// resolve them for a target that never calls them, and report failure because
// that is the honest answer where no runtime maker exists: a caller that
// reached one is told its step did not happen rather than that it did.
//
// Being weak is what lets a case that *does* drive a real preparation supply
// its own — a strong definition in that case's own translation unit overrides
// these for its target alone. `test_diagnostic_capacity_refusal_phase.cpp` is
// one such case.
//
// Opaque parameter types on purpose: the signatures are matched by linkage,
// and naming the real structs here would drag each runtime's headers into
// every target that links this file.
struct ChipCallable;
struct HostApi;
struct CallableArtifacts;
struct Runtime;

extern "C" __attribute__((weak)) int register_callable_impl(
    const ChipCallable * /* callable */, const HostApi * /* api */, CallableArtifacts * /* out */
) {
    return -1;
}

extern "C" __attribute__((weak)) int copy_in_run_inputs_impl(
    const Runtime * /* runtime */, const HostApi * /* api */
) {
    return -1;
}

extern "C" __attribute__((weak)) int release_run_bindings_impl(Runtime * /* runtime */, const HostApi * /* api */) {
    return -1;
}

// host_build_graph's emit half, named beside `dep_gen_host_graph_take` above
// and absent for the same reason.
extern "C" __attribute__((weak)) int dep_gen_host_graph_emit(const char * /* deps_json_path */) { return -1; }

extern "C" __attribute__((weak)) int publish_run_image_impl(Runtime * /* runtime */, const HostApi * /* api */) {
    return -1;
}

extern "C" __attribute__((weak)) int copy_back_run_outputs_impl(
    const Runtime * /* runtime */, const HostApi * /* api */, int /* execution_rc */, int /* launched */
) {
    return -1;
}

struct CallConfig;
struct PipelineContract;

extern "C" __attribute__((weak)) int build_kernel_pipeline_contract_impl(
    const CallConfig * /* config */, PipelineContract * /* out */
) {
    return -1;
}

// False is the honest answer for a target with no runtime maker: nothing here
// publishes an entry table, so a caller is told there is none rather than
// being sent to read one.
extern "C" __attribute__((weak)) bool runtime_uses_callable_entry_table_impl() { return false; }

// The capture-side twin of `dep_gen_host_graph_take`. A runtime that captures
// on the device carries this same weak false, and a target with no runtime
// maker is in the same position: no host graph is active here.
extern "C" __attribute__((weak)) bool dep_gen_host_graph_active() { return false; }
