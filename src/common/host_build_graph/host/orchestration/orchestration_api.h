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
 * Orchestration API - Slim header for orchestration .so files
 *
 * This header provides everything an orchestration source needs without
 * pulling in runtime implementation headers.  The orchestration .so has
 * zero link dependencies on runtime .cpp files; every call into the host
 * goes through the OrchOps function-pointer table on OrchContext.
 *
 * Orchestration sources include ONLY this header:
 *   #include "orchestration_api.h"
 *
 * The host sees the full OrchContext in host/orchestrator.h. The device-side
 * RuntimeContext in runtime_core.h is a separate type that orchestration never
 * touches: it is the state that travels to the device, and nothing host-only
 * lives in it.
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <cstring>
#include <ctime>
#include <functional>
#include <type_traits>

// Type headers needed by orchestration
#include "assert_compat.h"                      // always_assert
#include "host_build_graph/common.h"            // framework_bind_orch_context / framework_current_orch_context
#include "common/host_phase_kind.h"             // HostPhaseKind, for the phase records below
#include "host_build_graph/host/graph_cache.h"  // Graph Execution key and result helpers
#include "host_build_graph/host/orchestration/graph_config.h"  // simpler::hbg::ConfigBase, DefaultConfig, Config traits
#include "host_build_graph/host/orch_ops.h"                    // OrchOps, OrchContext forward declaration
#include "host_build_graph/runtime_status.h"                   // SIMPLER_ERROR_*
#include "host_build_graph/submit_types.h"                     // MixedKernels, INVALID_KERNEL_ID, subtask slots
#include "types.h"                                             // Arg, TaskOutputTensors, TensorArgType
#include "tensor.h"                                            // simpler::hbg::Tensor, TensorCreateInfo

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
 * Partial OrchContext definition for orchestration.
 *
 * Exposes the ops pointer (for runtime calls) and pending_scope_mode
 * (read directly by inline scope wrappers). The full struct (in
 * host/orchestrator.h) has the same first fields, so accessing them through
 * this definition is well-defined (C struct layout guarantee). The full one
 * stays out of here because it names OrchestratorState and HostTensorAccessor,
 * which the orchestration .so has no link dependency on.
 *
 * This context is host-side only: the orchestration .so and the ops table it
 * calls both run on the host, and nothing here is uploaded to the device.
 */
struct OrchContext {
    const OrchOps *ops;
    ScopeMode pending_scope_mode;
};

// The host library compiles the full definition in host/orchestrator.h, which
// carries three more members after these two. Both sides assert the same offsets
// so a reorder on either one fails to compile rather than silently disagreeing
// across the two translation units.
static_assert(offsetof(OrchContext, ops) == 0, "OrchContext ops layout drift");
static_assert(
    offsetof(OrchContext, pending_scope_mode) == sizeof(const OrchOps *), "OrchContext pending_scope_mode layout drift"
);

// =============================================================================
// Inline Convenience Wrappers (call through ops table)
// =============================================================================

static inline OrchContext *current_orch_context() { return framework_current_orch_context(); }

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
    OrchContext *ctx = current_orch_context();
    if (ctx->ops->is_fatal(ctx)) {
        return TaskOutputTensors{};
    }
    return ctx->ops->alloc_tensors(ctx, args);
}

static inline TaskOutputTensors alloc_tensors(const TensorCreateInfo create_infos[], uint32_t count) {
    OrchContext *ctx = current_orch_context();
    if (ctx->ops->is_fatal(ctx)) {
        return TaskOutputTensors{};
    }
    CoreTaskArgs args;
    for (uint32_t i = 0; i < count; i++) {
        args.add_output(create_infos[i]);
    }
    if (args.has_error()) {
        ctx->ops->report_fatal(
            ctx, SIMPLER_ERROR_INVALID_ARGS, __FUNCTION__, "%s",
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
    OrchContext *ctx = current_orch_context();
    if (ctx->ops->is_fatal(ctx)) {
        return TaskOutputTensors{};
    }
    CoreTaskArgs args;
    (args.add_output(cis), ...);
    if (args.has_error()) {
        ctx->ops->report_fatal(
            ctx, SIMPLER_ERROR_INVALID_ARGS, __FUNCTION__, "%s",
            args.error_msg() ? args.error_msg() : "alloc_tensors failed to construct output-only Arg"
        );
        return TaskOutputTensors{};
    }
    return alloc_tensors(args);
}

static inline TaskOutputTensors rt_submit_task(const MixedKernels &mixed_kernels, const CoreTaskArgs &args) {
    OrchContext *ctx = current_orch_context();
    if (ctx->ops->is_fatal(ctx)) {
        return TaskOutputTensors{};
    }
    return ctx->ops->submit_task(ctx, mixed_kernels, args);
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
    OrchContext *ctx = current_orch_context();
    if (ctx->ops->is_fatal(ctx)) {
        return TaskOutputTensors{};
    }
    return ctx->ops->submit_dummy_task(ctx, args);
}

static inline GraphScopeResult rt_graph_begin(uint64_t graph_key, const GraphTaskArgs &args) {
    OrchContext *ctx = current_orch_context();
    // Every runtime reaching this header implements graph_begin, so a null one means the ops
    // table is not the one this call belongs to. Asserted rather than reported: the default
    // result below says "nothing submitted, no error", under which the caller drops the body
    // and leaves no record of having done so.
    always_assert(ctx->ops->graph_begin != nullptr && "an HBG runtime's ops table carries graph_begin");
    // An already-fatal runtime fails this submission like any other: nothing is submitted
    // and no recording opens. No report of its own -- `has_error` promises a latched fatal,
    // and the one that short-circuited this call is it. graph_begin answers the same
    // condition the same way, through require_device_arguments.
    if (ctx->ops->is_fatal(ctx)) {
        GraphScopeResult rejected;
        rejected.has_error = true;
        return rejected;
    }
    return ctx->ops->graph_begin(ctx, graph_key, args);
}

// Bind the calling thread to the recording `graph_key` opened. The handle comes
// from the GraphScopeResult that opened it, so a thread can only ever record
// into the recording it was handed.
static inline bool rt_graph_prepare(void *recording_handle, const GraphTaskArgs &args) {
    OrchContext *ctx = current_orch_context();
    return ctx->ops->graph_prepare != nullptr && ctx->ops->graph_prepare(ctx, recording_handle, args);
}

static inline void rt_graph_abort(void *recording_handle) {
    OrchContext *ctx = current_orch_context();
    if (ctx->ops->graph_abort != nullptr) ctx->ops->graph_abort(ctx, recording_handle);
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
    OrchContext *ctx = current_orch_context();
    if (ctx->ops->graph_end == nullptr) return false;
    return ctx->ops->graph_end(ctx);
}

static inline void rt_graph_commit() {
    OrchContext *ctx = current_orch_context();
    if (ctx->ops->graph_record_wait != nullptr) ctx->ops->graph_record_wait(ctx);

    if (!ctx->ops->is_fatal(ctx) && ctx->ops->graph_commit != nullptr) ctx->ops->graph_commit(ctx);
}

static inline void rt_scope_begin(ScopeMode mode = ScopeMode::AUTO) {
    OrchContext *ctx = current_orch_context();
    if (ctx->ops->is_fatal(ctx)) {
        return;
    }
    ctx->pending_scope_mode = mode;
    ctx->ops->scope_begin(ctx);
}

static inline void rt_scope_end() {
    OrchContext *ctx = current_orch_context();
    if (ctx->ops->is_fatal(ctx)) {
        return;
    }
    ctx->ops->scope_end(ctx);
}

static inline void rt_orchestration_done() {
    rt_graph_commit();
    rt_submit_phase_state() = RtSubmitPhaseState{};
    OrchContext *ctx = current_orch_context();
    ctx->ops->orchestration_done(ctx);
}

/** This-run MIX cluster (= AIC) count. Do not hardcode 24/36; MIX cohorts use this. */
static inline int32_t rt_available_cluster_count() {
    OrchContext *ctx = current_orch_context();
    return ctx->ops->available_cluster_count(ctx);
}

/** This-run standalone AIV core count. AIV-only cohorts size themselves on this. */
static inline int32_t rt_available_aiv_count() {
    OrchContext *ctx = current_orch_context();
    return ctx->ops->available_aiv_count(ctx);
}

static inline bool rt_is_fatal() {
    OrchContext *ctx = current_orch_context();
    return ctx->ops->is_fatal(ctx);
}

#define rt_report_fatal(code, fmt, ...)                                            \
    do {                                                                           \
        OrchContext *_ctx = current_orch_context();                                \
        _ctx->ops->report_fatal(_ctx, (code), __FUNCTION__, (fmt), ##__VA_ARGS__); \
    } while (0)

// =============================================================================
// Logging Macros for Orchestration (call through ops table)
// =============================================================================

#define LOG_ERROR(fmt, ...) current_orch_context()->ops->log_error(__FUNCTION__, fmt, ##__VA_ARGS__)
#define LOG_WARN(fmt, ...) current_orch_context()->ops->log_warn(__FUNCTION__, fmt, ##__VA_ARGS__)

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
    const OrchOps *ops = current_orch_context()->ops;
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
    if (current_orch_context()->ops->record_orch_phase == nullptr) return 0;
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull + static_cast<uint64_t>(ts.tv_nsec);
}

#define LOG_TIMING(fmt, ...) current_orch_context()->ops->log_timing(__FUNCTION__, fmt, ##__VA_ARGS__)
#define LOG_INFO(fmt, ...) current_orch_context()->ops->log_info(__FUNCTION__, fmt, ##__VA_ARGS__)
#define LOG_DEBUG(fmt, ...) current_orch_context()->ops->log_debug(__FUNCTION__, fmt, ##__VA_ARGS__)

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
 * Reads an explicit HOST/NONE argument declared IN or INOUT. Its backing must
 * be ready before submit and stay alive through completion. DEVICE arguments,
 * including uploaded HOST/H2D inputs and runtime outputs, are rejected. This
 * entry never maps device memory or waits/copies a device-produced value.
 */
template <typename T = uint64_t>
static inline T get_tensor_data(const simpler::hbg::Tensor &tensor, uint32_t ndims, const uint32_t indices[]) {
    OrchContext *ctx = current_orch_context();
    if (ctx->ops->is_fatal(ctx)) {
        return from_u64<T>(0);
    }
    return from_u64<T>(ctx->ops->get_tensor_data(ctx, tensor, ndims, indices));
}

/**
 * Write a value to a tensor at the given multi-dimensional indices.
 *
 * Type is deduced from value argument; uint64_t by default:
 *
 *   set_tensor_data(tensor, 1, idx, raw_u64);     // old usage unchanged
 *   set_tensor_data(tensor, 1, idx, 42.0f);       // typed write (T = float)
 *
 * Writes an explicit HOST/NONE argument declared OUT or INOUT. Only the host
 * backing changes; a separate HOST/H2D argument keeps its already-copied value.
 * DEVICE arguments and runtime outputs are rejected. This is not a device
 * transfer or a synchronization barrier.
 */
template <typename T = uint64_t>
static inline void
set_tensor_data(const simpler::hbg::Tensor &tensor, uint32_t ndims, const uint32_t indices[], T value) {
    OrchContext *ctx = current_orch_context();
    if (ctx->ops->is_fatal(ctx)) {
        return;
    }
    ctx->ops->set_tensor_data(ctx, tensor, ndims, indices, to_u64(value));
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
        ctx_(current_orch_context()) {
        if (!ctx_->ops->is_fatal(ctx_)) {
            ctx_->pending_scope_mode = mode;
            ctx_->ops->scope_begin(ctx_);
        }
    }
    ~ScopeGuard() {
        if (!ctx_->ops->is_fatal(ctx_)) {
            ctx_->ops->scope_end(ctx_);
        }
    }

private:
    OrchContext *ctx_;
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
    // A boundary cannot hold a runtime-allocated output: GraphTaskArgs::add_output rejects a
    // TensorCreateInfo at compile time, so this tag is reachable only by a bug.
    //
    // The caller-triggerable conditions -- a construction error, a boundary past the tensor
    // cap, explicit dependencies -- are tested at the guard below, which refuses them the
    // same way in both builds.
    for (int32_t i = 0; i < args.tensor_count(); ++i) {
        always_assert(
            args.tag(i) != TensorArgType::OUTPUT &&
            "a Graph boundary cannot hold a runtime-allocated output (TensorCreateInfo)"
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
    // A boundary that cannot open a Graph is a contract violation, not a reason to run
    // the body quietly: the caller declared a Graph and gets told why it is not one.
    //
    // Explicit dependencies are tested here rather than inside rt_graph_args_cacheable
    // because they are refused only at a Graph boundary -- an ordinary task takes them. The
    // same pair is tested again in graph_begin_inner, which is what a caller reaching the
    // host layer by another route meets; this one reports at the API the caller named.
    if (!rt_graph_args_cacheable(args) || args.explicit_dep_count() != 0) {
        rt_report_fatal(
            SIMPLER_ERROR_INVALID_ARGS,
            "[GraphExecution] boundary cannot open a Graph: tensors=%d scalars=%d explicit_deps=%u error=%d",
            args.tensor_count(), args.scalar_count(), args.explicit_dep_count(), static_cast<int>(args.has_error())
        );
        GraphSubmitResult rejected;
        rejected.has_error = true;
        return rejected;
    }
    const uint64_t _admitted_ns = rt_orch_phase_now_ns();
    rt_record_orch_phase(HostPhaseKind::OrchSubmitAdmit, _entry_ns, _admitted_ns, graph_key);
    GraphScopeResult result = rt_graph_begin(graph_key, args);
    // A failed submission has latched a fatal, so there is no body to run and no phase
    // worth recording: the orchestration it belongs to will not produce a timeline.
    if (result.has_error) return result;
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
        OrchContext *record_ctx = current_orch_context();
        const bool queued = record_ctx->ops->graph_record_start != nullptr &&
                            record_ctx->ops->graph_record_start(record_ctx, params, &job);
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
    }
    // A cache hit or an in-flight hit skips the body. Every in-flight Graph task
    // is finalized at orchestration completion.
    const uint64_t _exit_ns = rt_orch_phase_now_ns();
    if (result.recording) {
        // Handing the recording to a worker. Measured at 10-75 us per start and covered
        // by no other record: it runs after rt_graph_begin returns, so a swimlane shows
        // it as a gap with no recorder active — which is what it is, the recorder has
        // not reached its first sub-task yet.
        rt_record_orch_phase(HostPhaseKind::OrchRecordHandoff, _begun_ns, _exit_ns, graph_key);
    }
    _phase.count++;
    _phase.prev_exit_ns = _exit_ns;
    return result;
}

template <typename ConfigT>
static inline GraphSubmitResult
rt_submit_graph(uint64_t graph_id, GraphFunction function, const GraphTaskArgs &args, const ConfigT &config) {
    simpler::hbg::assert_config<ConfigT>();
    if (function == nullptr) {
        // Reported rather than asserted: debug_assert is a no-op in release, where a null
        // body would otherwise drop the Graph silently and let the orchestration carry on.
        rt_report_fatal(
            SIMPLER_ERROR_INVALID_ARGS, "%s",
            "[GraphExecution] a Graph was submitted with a null body, which names nothing to record"
        );
        GraphSubmitResult rejected;
        rejected.has_error = true;
        return rejected;
    }
    const uint64_t graph_key = rt_graph_make_key(graph_id, config.hash());
    if (!simpler::hbg::retain_config(graph_key, config)) {
        rt_report_fatal(
            SIMPLER_ERROR_INVALID_ARGS,
            "[GraphExecution] Graph key collision: key 0x%llx was already submitted with a different Config. "
            "Two Configs hashing alike means the key no longer names one Graph -- check that hash() mixes "
            "every field operator== compares",
            static_cast<unsigned long long>(graph_key)
        );
        GraphSubmitResult rejected;
        rejected.has_error = true;
        return rejected;
    }
    return rt_submit_graph_impl(graph_key, args, [function](const GraphTaskArgs &record_args) {
        function(record_args);
    });
}

// Submitting without a Config and submitting a DefaultConfig are the same submission. They
// share this one path rather than computing a key each, so the two spellings cannot drift
// into two keys for one Graph.
static inline GraphSubmitResult rt_submit_graph(uint64_t graph_id, GraphFunction function, const GraphTaskArgs &args) {
    return rt_submit_graph(graph_id, function, args, simpler::hbg::DefaultConfig{});
}

static inline GraphSubmitResult rt_submit_graph(GraphFunction function, const GraphTaskArgs &args) {
    return rt_submit_graph(rt_graph_function_id(function), function, args, simpler::hbg::DefaultConfig{});
}

template <typename ConfigT>
static inline GraphSubmitResult
rt_submit_graph(GraphFunction function, const GraphTaskArgs &args, const ConfigT &config) {
    return rt_submit_graph(rt_graph_function_id(function), function, args, config);
}

// A Graph whose body reads construction parameters. The body takes the Config by its own
// type, so a call that pairs a body with the wrong Config fails to deduce ConfigT rather
// than compiling into a wrong read.
template <typename ConfigT>
using GraphFunctionWithConfig = void (*)(const GraphTaskArgs &, const ConfigT &);

template <typename ConfigT>
static inline GraphSubmitResult rt_submit_graph(
    uint64_t graph_id, GraphFunctionWithConfig<ConfigT> function, const GraphTaskArgs &args, const ConfigT &config
) {
    simpler::hbg::assert_config<ConfigT>();
    if (function == nullptr) {
        // Reported rather than asserted: debug_assert is a no-op in release, where a null
        // body would otherwise drop the Graph silently and let the orchestration carry on.
        rt_report_fatal(
            SIMPLER_ERROR_INVALID_ARGS, "%s",
            "[GraphExecution] a Graph was submitted with a null body, which names nothing to record"
        );
        GraphSubmitResult rejected;
        rejected.has_error = true;
        return rejected;
    }
    const uint64_t graph_key = rt_graph_make_key(graph_id, config.hash());
    if (!simpler::hbg::retain_config(graph_key, config)) {
        rt_report_fatal(
            SIMPLER_ERROR_INVALID_ARGS,
            "[GraphExecution] Graph key collision: key 0x%llx was already submitted with a different Config. "
            "Two Configs hashing alike means the key no longer names one Graph -- check that hash() mixes "
            "every field operator== compares",
            static_cast<unsigned long long>(graph_key)
        );
        GraphSubmitResult rejected;
        rejected.has_error = true;
        return rejected;
    }
    // The lambda captures its own copy: the recording job runs on a recorder thread that
    // outlives this call, so it may hold nothing of the caller's frame -- `config` included.
    return rt_submit_graph_impl(graph_key, args, [function, config](const GraphTaskArgs &record_args) {
        function(record_args, config);
    });
}

template <typename ConfigT>
static inline GraphSubmitResult
rt_submit_graph(GraphFunctionWithConfig<ConfigT> function, const GraphTaskArgs &args, const ConfigT &config) {
    return rt_submit_graph(rt_graph_function_id(function), function, args, config);
}

// Release what this .so holds for the bind that just ended. The host calls it once per bind,
// after the orchestration entry returns, so anything retained here covers the same window as
// the host's Definition cache -- which is a local of run_host_orchestration and dies with it.
//
// Currently that is the Graph Config registry. A registry outliving the cache would report a
// collision between two Configs that, the cache having been emptied in between, could never
// have been confused for one another.
//
// `used` because this is an inline definition nothing inside the .so calls: the host reaches
// it by dlsym, so the symbol has to be emitted anyway.
extern "C" __attribute__((visibility("default"), used)) inline void framework_finish_orch() {
    simpler::hbg::clear_config_registry();
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
