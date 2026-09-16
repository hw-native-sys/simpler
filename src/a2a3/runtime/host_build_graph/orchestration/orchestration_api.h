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
 * PTO Orchestration API - Slim header for orchestration .so files
 *
 * This header provides everything an orchestration source needs without
 * pulling in runtime implementation headers.  The orchestration .so has
 * zero link dependencies on runtime .cpp files; all runtime calls go
 * through the RuntimeOps function-pointer table embedded in
 * RuntimeContext.
 *
 * Orchestration sources include ONLY this header:
 *   #include "orchestration_api.h"
 *
 * Runtime sources continue to use runtime_core.h (which defines the
 * full RuntimeContext struct with all internal fields).
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <cstring>
#include <ctime>
#include <functional>
#include <tuple>
#include <type_traits>

// Type headers needed by orchestration
#include "assert_compat.h"                    // debug_assert
#include "host_build_graph/common.h"          // framework_bind_runtime / framework_current_runtime
#include "common/host_phase_kind.h"           // HostPhaseKind, for the phase records below
#include "host_build_graph/graph_cache.h"     // Graph Execution key and result helpers
#include "host_build_graph/runtime_ops.h"     // RuntimeOps, RuntimeContext forward declaration
#include "host_build_graph/runtime_status.h"  // SIMPLER_ERROR_*
#include "host_build_graph/submit_types.h"    // MixedKernels, INVALID_KERNEL_ID, subtask slots
#include "types.h"                            // Arg, TaskOutputTensors, TensorArgType
#include "tensor.h"                           // simpler::hbg::Tensor, TensorCreateInfo

// =============================================================================
// simpler::hbg::Tensor Factory Helpers
// =============================================================================

// simpler::hbg::make_tensor_external(...) — canonical factory for pre-allocated external
// memory — is defined in the unified tensor.h (common), so host and runtime
// build ChipTensors through the same controlled path.

// =============================================================================
// Opaque Runtime
// =============================================================================

/**
 * Partial RuntimeContext definition for orchestration.
 *
 * Exposes the ops pointer (for runtime calls) and pending_scope_mode
 * (read directly by inline scope wrappers).  The real struct (in
 * runtime_core.h) has the same first fields, so accessing them through
 * this definition is well-defined (C struct layout guarantee).
 */
struct RuntimeContext {
    const RuntimeOps *ops;
    ScopeMode pending_scope_mode;
};

// =============================================================================
// Inline Convenience Wrappers (call through ops table)
// =============================================================================

static inline RuntimeContext *current_runtime() { return framework_current_runtime(); }

// Where the previous submission on this thread left off, so the generated code's own
// work between two submissions can be named rather than showing as an unattributed gap.
// Per thread because a recording worker submits from its own thread.
//
// Cleared by rt_orchestration_done(), which is the actual boundary: a value carried into
// the next orchestration would name a span that includes the device run and the next
// bind's staging, and the record would be filed in the next bind's pool.
struct RtSubmitPhaseState {
    uint64_t prev_exit_ns;
    uint64_t count;
};

inline RtSubmitPhaseState &rt_submit_phase_state() {
    static thread_local RtSubmitPhaseState state{};
    return state;
}

static inline void rt_graph_commit();

// An ordinary submission depends on nothing a recording produces. The outer
// Graph shell entered the task sequence and registered its TensorMap producers
// at graph_begin, so fanin against it is already correct; the recording only
// builds the Definition image, and the deferred heap block a shell still needs is
// an independent bump allocation that orchestration completion reserves. So none
// of the three wrappers below joins the recorders — a barrier here stalls the
// submitting thread for the rest of every recording in flight, which on the
// four-Definition DeepSeek-V4 decode was a third of the orchestration window.
static inline TaskOutputTensors alloc_tensors(const CoreTaskArgs &args) {
    RuntimeContext *rt = current_runtime();
    if (rt->ops->is_fatal(rt)) {
        return TaskOutputTensors{};
    }
    return rt->ops->alloc_tensors(rt, args);
}

static inline TaskOutputTensors alloc_tensors(const TensorCreateInfo create_infos[], uint32_t count) {
    RuntimeContext *rt = current_runtime();
    if (rt->ops->is_fatal(rt)) {
        return TaskOutputTensors{};
    }
    CoreTaskArgs args;
    for (uint32_t i = 0; i < count; i++) {
        args.add_output(create_infos[i]);
    }
    if (args.has_error()) {
        rt->ops->report_fatal(
            rt, SIMPLER_ERROR_INVALID_ARGS, __FUNCTION__, "%s",
            args.error_msg() ? args.error_msg() : "alloc_tensors failed to construct output-only Arg"
        );
        return TaskOutputTensors{};
    }
    return alloc_tensors(args);
}

template <typename... CIs>
static inline TaskOutputTensors alloc_tensors(const CIs &...cis) {
    static_assert(sizeof...(cis) > 0, "alloc_tensors requires at least one TensorCreateInfo");
    static_assert(
        (std::is_same_v<std::decay_t<CIs>, TensorCreateInfo> && ...),
        "alloc_tensors only accepts TensorCreateInfo arguments"
    );
    RuntimeContext *rt = current_runtime();
    if (rt->ops->is_fatal(rt)) {
        return TaskOutputTensors{};
    }
    CoreTaskArgs args;
    (args.add_output(cis), ...);
    if (args.has_error()) {
        rt->ops->report_fatal(
            rt, SIMPLER_ERROR_INVALID_ARGS, __FUNCTION__, "%s",
            args.error_msg() ? args.error_msg() : "alloc_tensors failed to construct output-only Arg"
        );
        return TaskOutputTensors{};
    }
    return alloc_tensors(args);
}

static inline TaskOutputTensors rt_submit_task(const MixedKernels &mixed_kernels, const CoreTaskArgs &args) {
    RuntimeContext *rt = current_runtime();
    if (rt->ops->is_fatal(rt)) {
        return TaskOutputTensors{};
    }
    return rt->ops->submit_task(rt, mixed_kernels, args);
}

/**
 * Convenience wrapper: submit an AIC-only task.
 */
static inline TaskOutputTensors rt_submit_aic_task(int32_t kernel_id, const CoreTaskArgs &args) {
    MixedKernels mk;
    mk.aic_kernel_id = kernel_id;
    return rt_submit_task(mk, args);
}

/**
 * Convenience wrapper: submit an AIV-only task (uses AIV0 slot).
 */
static inline TaskOutputTensors rt_submit_aiv_task(int32_t kernel_id, const CoreTaskArgs &args) {
    MixedKernels mk;
    mk.aiv0_kernel_id = kernel_id;
    return rt_submit_task(mk, args);
}

/**
 * Submit a dependency-only task. Accepts the same Arg shape as rt_submit_task
 * (inputs, outputs, inouts, explicit_deps, scalars) but does not run any
 * AICore kernel. The task still participates in the dependency graph: it
 * waits on its fanin and notifies its fanout. Useful as a synchronization
 * barrier or as a placeholder producer for tests / dep-graph wiring.
 */
static inline TaskOutputTensors rt_submit_dummy_task(const CoreTaskArgs &args) {
    RuntimeContext *rt = current_runtime();
    if (rt->ops->is_fatal(rt)) {
        return TaskOutputTensors{};
    }
    return rt->ops->submit_dummy_task(rt, args);
}

static inline GraphScopeResult rt_graph_begin(uint64_t graph_key, const GraphTaskArgs &args) {
    RuntimeContext *rt = current_runtime();
    if (rt->ops->is_fatal(rt) || rt->ops->graph_begin == nullptr) {
        return GraphScopeResult{};
    }
    return rt->ops->graph_begin(rt, graph_key, args);
}

// Bind the calling thread to the recording `graph_key` opened. The handle comes
// from the GraphScopeResult that opened it, so a thread can only ever record
// into the recording it was handed.
static inline bool rt_graph_prepare(void *recording_handle, const GraphTaskArgs &args) {
    RuntimeContext *rt = current_runtime();
    return rt->ops->graph_prepare != nullptr && rt->ops->graph_prepare(rt, recording_handle, args);
}

static inline void rt_graph_abort(void *recording_handle) {
    RuntimeContext *rt = current_runtime();
    if (rt->ops->graph_abort != nullptr) rt->ops->graph_abort(rt, recording_handle);
}

// Finish the recording pass and publish its Definition. The calling thread
// finalizes the already-submitted outer Graph shells in rt_graph_commit.
//
// No is_fatal() short-circuit: a prepared recording has to leave RECORDING even when
// the run has already failed, or graph_commit's drain never completes. graph_end
// retires the entry it bound on every path that has one — published, unsupported,
// key-mismatched, or fatal — so `false` here means "no Definition", never "the entry
// is still yours to retire". That is why no caller pairs this with an abort.
//
// The null-op guard is defensive only: hbg's ops table always carries graph_end
// (runtime_core.cpp), and a table without it would have no graph_begin either, so
// no recording could be open to end. It reports false rather than the older `true`
// because nothing was published.
static inline bool rt_graph_end() {
    RuntimeContext *rt = current_runtime();
    if (rt->ops->graph_end == nullptr) return false;
    return rt->ops->graph_end(rt);
}

static inline void rt_graph_commit() {
    RuntimeContext *rt = current_runtime();
    if (rt->ops->graph_record_wait != nullptr) rt->ops->graph_record_wait(rt);

    if (!rt->ops->is_fatal(rt) && rt->ops->graph_commit != nullptr) rt->ops->graph_commit(rt);
}

static inline void rt_scope_begin(ScopeMode mode = ScopeMode::AUTO) {
    RuntimeContext *rt = current_runtime();
    if (rt->ops->is_fatal(rt)) {
        return;
    }
    rt->pending_scope_mode = mode;
    rt->ops->scope_begin(rt);
}

static inline void rt_scope_end() {
    RuntimeContext *rt = current_runtime();
    if (rt->ops->is_fatal(rt)) {
        return;
    }
    rt->ops->scope_end(rt);
}

static inline void rt_orchestration_done() {
    rt_graph_commit();
    rt_submit_phase_state() = RtSubmitPhaseState{};
    RuntimeContext *rt = current_runtime();
    rt->ops->orchestration_done(rt);
}

/** This-run MIX cluster (= AIC) count. Do not hardcode 24/36; MIX cohorts use this. */
static inline int32_t rt_available_cluster_count() {
    RuntimeContext *rt = current_runtime();
    return rt->ops->available_cluster_count(rt);
}

/** This-run standalone AIV core count. AIV-only cohorts size themselves on this. */
static inline int32_t rt_available_aiv_count() {
    RuntimeContext *rt = current_runtime();
    return rt->ops->available_aiv_count(rt);
}

static inline bool rt_is_fatal() {
    RuntimeContext *rt = current_runtime();
    return rt->ops->is_fatal(rt);
}

#define rt_report_fatal(code, fmt, ...)                                          \
    do {                                                                         \
        RuntimeContext *_rt = current_runtime();                                 \
        _rt->ops->report_fatal(_rt, (code), __FUNCTION__, (fmt), ##__VA_ARGS__); \
    } while (0)

// =============================================================================
// Logging Macros for Orchestration (call through ops table)
// =============================================================================

#define LOG_ERROR(fmt, ...) current_runtime()->ops->log_error(__FUNCTION__, fmt, ##__VA_ARGS__)
#define LOG_WARN(fmt, ...) current_runtime()->ops->log_warn(__FUNCTION__, fmt, ##__VA_ARGS__)

// ============================================================================
// Submission-gap probe — diagnostic only, nothing branches on it.
//
// A graph_begin phase record covers only what the runtime does. The time between two
// such records is this function's own pre/post work plus whatever the generated
// orchestration code does between submissions, and no existing marker separates them.
// These accumulators do, in the .so where that code actually runs.
// ============================================================================
// The three submission segments the runtime cannot see, filed under the
// OrchSubmitAdmit / OrchRecordHandoff / OrchGeneratedArgs kinds.
inline void rt_record_orch_phase(HostPhaseKind phase, uint64_t start_ns, uint64_t end_ns, uint64_t detail) {
    const RuntimeOps *ops = current_runtime()->ops;
    if (ops->record_orch_phase != nullptr) {
        ops->record_orch_phase(static_cast<uint32_t>(phase), start_ns, end_ns, detail);
    }
}

// Monotonic nanoseconds, or 0 when nothing collects the records — the same convention
// host_phase_now_ns() uses on the runtime side, and the same clock, so a record emitted
// here nests under the bind's span with no conversion. The runtime's clock is not
// reachable from this header, which resolves no runtime symbols of its own, so the
// gate is the ops entry that rt_record_orch_phase would call.
inline uint64_t rt_orch_phase_now_ns() {
    if (current_runtime()->ops->record_orch_phase == nullptr) return 0;
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull + static_cast<uint64_t>(ts.tv_nsec);
}

#define LOG_TIMING(fmt, ...) current_runtime()->ops->log_timing(__FUNCTION__, fmt, ##__VA_ARGS__)
#define LOG_INFO(fmt, ...) current_runtime()->ops->log_info(__FUNCTION__, fmt, ##__VA_ARGS__)
#define LOG_DEBUG(fmt, ...) current_runtime()->ops->log_debug(__FUNCTION__, fmt, ##__VA_ARGS__)

// =============================================================================
// Cross-Layer Data Access
// =============================================================================

/**
 * Read a value from a tensor at the given multi-dimensional indices.
 *
 * Default T = uint64_t preserves old behavior (raw bits).
 * Specify T to get automatic type conversion:
 *
 *   uint64_t raw = get_tensor_data(tensor, 1, idx);       // old usage unchanged
 *   float val = get_tensor_data<float>(tensor, 1, idx);   // typed read
 *
 * This API reads the registered host view used to stage an external tensor.
 * It is valid while host orchestration is building the graph, before device
 * scheduling starts. A tensor produced by a submitted task cannot become
 * readable during graph construction, and a runtime-created output has no
 * registered host view; either use is reported as an invalid argument.
 */
template <typename T = uint64_t>
static inline T get_tensor_data(const simpler::hbg::Tensor &tensor, uint32_t ndims, const uint32_t indices[]) {
    RuntimeContext *rt = current_runtime();
    if (rt->ops->is_fatal(rt)) {
        return from_u64<T>(0);
    }
    return from_u64<T>(rt->ops->get_tensor_data(rt, tensor, ndims, indices));
}

/**
 * Write a value to a tensor at the given multi-dimensional indices.
 *
 * Type is deduced from value argument; uint64_t by default:
 *
 *   set_tensor_data(tensor, 1, idx, raw_u64);     // old usage unchanged
 *   set_tensor_data(tensor, 1, idx, 42.0f);       // typed write (T = float)
 *
 * This API updates the registered host view used to stage an external tensor.
 * The updated value becomes part of the graph's initial device data. It is not
 * a synchronization barrier for submitted readers or writers. A tensor with a
 * submitted producer, or a runtime-created output with no registered host view,
 * is rejected as an invalid argument.
 */
template <typename T = uint64_t>
static inline void
set_tensor_data(const simpler::hbg::Tensor &tensor, uint32_t ndims, const uint32_t indices[], T value) {
    RuntimeContext *rt = current_runtime();
    if (rt->ops->is_fatal(rt)) {
        return;
    }
    rt->ops->set_tensor_data(rt, tensor, ndims, indices, to_u64(value));
}

// =============================================================================
// C++ Scope Guards and Macros
// =============================================================================

/**
 * RAII Scope Guard (calls through ops table)
 */
class ScopeGuard {
public:
    explicit ScopeGuard(ScopeMode mode = ScopeMode::AUTO) :
        rt_(current_runtime()) {
        if (!rt_->ops->is_fatal(rt_)) {
            rt_->pending_scope_mode = mode;
            rt_->ops->scope_begin(rt_);
        }
    }
    ~ScopeGuard() {
        if (!rt_->ops->is_fatal(rt_)) {
            rt_->ops->scope_end(rt_);
        }
    }

private:
    RuntimeContext *rt_;
};

// Define or submit a Graph Execution. On a cache miss the function executes
// normally and its sub-DAG is recorded. On a hit the function is skipped and
// one Graph task is submitted; Scheduler expands the cached topology with the
// current invocation's GraphTaskArgs.
using GraphFunction = void (*)(const GraphTaskArgs &);

template <typename Function>
static inline uint64_t rt_graph_function_id(Function function) {
    static_assert(std::is_pointer_v<Function>, "Graph function identity requires a function pointer");
    static_assert(sizeof(function) <= sizeof(uint64_t), "Graph function pointer must fit in a 64-bit identity");
    uint64_t function_id = 0;
    std::memcpy(&function_id, &function, sizeof(function));
    return function_id;
}

// `invoke` is copied into the recording job and runs on a recording thread, which
// outlives this call: the caller returns as soon as the outer shell is submitted,
// and the body runs until orchestration completion joins it. So `invoke` must own
// everything it needs by value. Capturing caller-frame storage by reference —
// including the boundary `args` — is a use-after-free; the recorded body receives
// its own boundary copy as a parameter for exactly that reason.
template <typename Invoke>
static inline GraphSubmitResult rt_submit_graph_impl(uint64_t graph_key, const GraphTaskArgs &args, Invoke invoke) {
    debug_assert(!args.has_error() && "Graph boundary GraphTaskArgs construction failed");
    debug_assert(args.tensor_count() <= GRAPH_MAX_TENSOR_ARGS && "Graph boundary exceeds the tensor limit");
    debug_assert(
        args.explicit_dep_count() == 0 && "Explicit dependencies crossing the Graph boundary are not supported"
    );
    for (int32_t i = 0; i < args.tensor_count(); ++i) {
        debug_assert(
            args.tag(i) != TensorArgType::OUTPUT &&
            "Runtime-allocated TensorCreateInfo is not supported at the Graph boundary"
        );
    }
    RtSubmitPhaseState &_phase = rt_submit_phase_state();
    const uint64_t _entry_ns = rt_orch_phase_now_ns();
    // Only a gap inside one orchestration is this .so's own work.
    // rt_orchestration_done() clears the state, so the submitting thread cannot carry a
    // value across that boundary at all. The duration test below is the backstop for a
    // thread that submits without ever reaching that call — a recording worker, or an
    // orchestration that fails before completing — where the state is thread_local and
    // outlives the bind. Real gaps here are single-digit microseconds, so a millisecond
    // separates the two cases by three orders of magnitude.
    constexpr uint64_t kBindBoundaryNs = 1000000;
    const uint64_t _between = _phase.prev_exit_ns == 0 ? 0 : _entry_ns - _phase.prev_exit_ns;
    if (_between != 0 && _between < kBindBoundaryNs) {
        rt_record_orch_phase(HostPhaseKind::OrchGeneratedArgs, _phase.prev_exit_ns, _entry_ns, _phase.count);
    }
    if (!rt_graph_args_cacheable(args)) {
        invoke(args);
        _phase.prev_exit_ns = rt_orch_phase_now_ns();
        return GraphSubmitResult{};
    }
    const uint64_t _admitted_ns = rt_orch_phase_now_ns();
    rt_record_orch_phase(HostPhaseKind::OrchSubmitAdmit, _entry_ns, _admitted_ns, graph_key);
    GraphScopeResult result = rt_graph_begin(graph_key, args);
    const uint64_t _begun_ns = rt_orch_phase_now_ns();
    if (result.recording) {
        void *handle = result.recording_handle;
        // The formal parameters the body reads: the entry's own copy, not `args` -- the
        // caller only lends those for the duration of this call, and recording resolves a
        // task slot's origin against this object's slot array.
        const GraphTaskArgs &params = *result.params;
        // A std::function rather than a bare lambda because the pool takes it as one
        // through the ops table's void *.
        std::function<void(const GraphTaskArgs &)> job = [invoke, handle](const GraphTaskArgs &record_args) mutable {
            try {
                if (!rt_graph_prepare(handle, record_args)) {
                    rt_graph_abort(handle);
                    return;
                }
                invoke(record_args);
                // Not paired with an abort: graph_end retires the entry it bound on
                // every path that has one, including the fatal one. A second abort
                // would race graph_commit's drain, which frees the entry after
                // releasing recording_mutex — so the mutex this would take is no
                // protection against touching it.
                (void)rt_graph_end();
            } catch (...) {
                rt_graph_abort(handle);
            }
        };
        // The pool takes the callable whether or not it queues it: start() moves it into
        // its own storage before it checks capacity, so `job` is empty either way. That
        // costs nothing here, because the fallback below re-runs `invoke` -- captured by
        // value, so unaffected -- rather than the job.
        RuntimeContext *record_rt = current_runtime();
        const bool queued = record_rt->ops->graph_record_start != nullptr &&
                            record_rt->ops->graph_record_start(record_rt, params, &job);
        if (!queued) {
            try {
                if (!rt_graph_prepare(handle, params)) {
                    rt_graph_abort(handle);
                    return result;
                }
                invoke(params);
                (void)rt_graph_end();
            } catch (...) {
                rt_graph_abort(handle);
                throw;
            }
        }
    } else if (result.execute_block) {
        // Un-cacheable at begin, or the Definition cache is full: ordinary path.
        if (!current_runtime()->ops->is_fatal(current_runtime())) invoke(args);
    }
    // A cache hit or an in-flight hit skips the body. Every in-flight Graph task
    // is finalized at orchestration completion.
    const uint64_t _exit_ns = rt_orch_phase_now_ns();
    if (result.recording) {
        // Handing the recording to a worker. Measured at 10-75 us per start and covered
        // by no other record: it runs after rt_graph_begin returns, so a swimlane shows
        // it as a gap with no recorder active — which is what it is, the recorder has
        // not reached its first in-graph task yet.
        rt_record_orch_phase(HostPhaseKind::OrchRecordHandoff, _begun_ns, _exit_ns, graph_key);
    }
    _phase.count++;
    _phase.prev_exit_ns = _exit_ns;
    return result;
}

static inline GraphSubmitResult rt_submit_graph(uint64_t graph_id, GraphFunction function, const GraphTaskArgs &args) {
    debug_assert(function != nullptr && "Graph function must not be null");
    if (function == nullptr) return GraphSubmitResult{};
    return rt_submit_graph_impl(rt_graph_make_key(graph_id), args, [function](const GraphTaskArgs &record_args) {
        function(record_args);
    });
}

static inline GraphSubmitResult rt_submit_graph(GraphFunction function, const GraphTaskArgs &args) {
    return rt_submit_graph(rt_graph_function_id(function), function, args);
}

template <typename... Config>
using GraphFunctionWithConfig = void (*)(const GraphTaskArgs &, Config...);

template <typename... Config>
static inline GraphSubmitResult rt_submit_graph(
    uint64_t graph_id, GraphFunctionWithConfig<Config...> function, const GraphTaskArgs &args, Config... config
) {
    debug_assert(function != nullptr && "Graph function must not be null");
    if (function == nullptr) return GraphSubmitResult{};
    auto configs = std::make_tuple(config...);
    return rt_submit_graph_impl(
        rt_graph_make_key(graph_id, config...), args, [function, configs](const GraphTaskArgs &record_args) {
            std::apply(
                [&](auto... values) {
                    function(record_args, values...);
                },
                configs
            );
        }
    );
}

template <typename... Config>
static inline GraphSubmitResult
rt_submit_graph(GraphFunctionWithConfig<Config...> function, const GraphTaskArgs &args, Config... config) {
    return rt_submit_graph(rt_graph_function_id(function), function, args, config...);
}

#define _SIMPLER_CONCATENATE_IMPL(x, y) x##y
#define _SIMPLER_CONCATENATE(x, y) _SIMPLER_CONCATENATE_IMPL(x, y)

#define SIMPLER_SCOPE_GUARD() [[maybe_unused]] ScopeGuard _SIMPLER_CONCATENATE(scope_guard_, __COUNTER__)

/**
 * Scoped block macro:
 *   SIMPLER_SCOPE() {
 *       rt_submit_task(...);
 *   }
 */
#define SIMPLER_SCOPE(...) if (ScopeGuard _SIMPLER_CONCATENATE(scope_guard_, __COUNTER__){__VA_ARGS__}; true)

// =============================================================================
// Orchestration Config
// =============================================================================

/**
 * Configuration exported by orchestration .so via aicpu_orchestration_config().
 * The executor reads these values to set up shared memory and runtime.
 *
 * This struct is defined identically in runtime_core.h (with an include
 * guard) so the executor can use the same type without including this header.
 */
#ifndef ORCHESTRATION_CONFIG_DEFINED
#define ORCHESTRATION_CONFIG_DEFINED
struct OrchestrationConfig {
    int expected_arg_count;
};
#endif

// Convenience layer (CoreTaskArgsWithDeps<N> + matching rt_submit_*_task overloads).
// Pulled in at the bottom so the wrapper sees CoreTaskArgs, MixedKernels, and the
// rt_submit_*_task primitives defined above. Orchestration sources include
// only this single header to access both the primitive and convenience APIs.
#include "arg_with_deps.h"  // NOLINT(build/include_subdir)
