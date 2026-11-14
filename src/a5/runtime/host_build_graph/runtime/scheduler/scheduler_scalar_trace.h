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

#include "scheduler_protocol.h"
#include "scheduler_scalar_record.h"
#include "host_build_graph/sched_phase_kind.h"

template <bool Enabled>
class SchedulerScalarScopeImpl;

template <>
class SchedulerScalarScopeImpl<true> {
public:
    inline __aicore__ explicit SchedulerScalarScopeImpl(
        SchedPhaseKind kind, int64_t task_ref = -1, uint64_t begin = 0
    ) :
        trace_(scheduler_scalar_current()),
        kind_(kind),
        task_ref_(task_ref) {
        if (trace_ != nullptr && trace_->buffer != nullptr) {
            depth_ = ++trace_->depth;
            begin_ = begin == 0 ? scheduler_cycles() : begin;
        } else trace_ = nullptr;
    }
    template <typename Local>
    inline __aicore__
    SchedulerScalarScopeImpl(const Local *local, SchedPhaseKind kind, int64_t task_ref = -1, uint64_t begin = 0) :
        trace_(local != nullptr && local->profiling != nullptr ? &local->profiling->scalar : nullptr),
        kind_(kind),
        task_ref_(task_ref) {
        if (trace_ != nullptr && trace_->buffer != nullptr) {
            depth_ = ++trace_->depth;
            begin_ = begin == 0 ? scheduler_cycles() : begin;
        } else trace_ = nullptr;
    }
    inline __aicore__ ~SchedulerScalarScopeImpl() { close(); }
    inline __aicore__ void set_task_ref(int64_t task_ref) { task_ref_ = task_ref; }
    inline __aicore__ void close() {
        if (trace_ == nullptr) return;
        const uint64_t end = scheduler_cycles();
        --trace_->depth;
        trace_->append_function(kind_, begin_, end, task_ref_, depth_);
        trace_ = nullptr;
    }
    SchedulerScalarScopeImpl(const SchedulerScalarScopeImpl &) = delete;
    SchedulerScalarScopeImpl &operator=(const SchedulerScalarScopeImpl &) = delete;

private:
    SchedulerScalarTrace *trace_;
    SchedPhaseKind kind_;
    int64_t task_ref_;
    uint64_t begin_{0};
    uint32_t depth_{0};
};

template <>
class SchedulerScalarScopeImpl<false> {
public:
    inline __aicore__ explicit SchedulerScalarScopeImpl(SchedPhaseKind, int64_t = -1, uint64_t = 0) {}
    template <typename Local>
    inline __aicore__ SchedulerScalarScopeImpl(const Local *, SchedPhaseKind, int64_t = -1, uint64_t = 0) {}
    inline __aicore__ void close() {}
    inline __aicore__ void set_task_ref(int64_t) {}
    SchedulerScalarScopeImpl(const SchedulerScalarScopeImpl &) = delete;
    SchedulerScalarScopeImpl &operator=(const SchedulerScalarScopeImpl &) = delete;
};

using SchedulerScalarScope = SchedulerScalarScopeImpl<kSchedulerScalarInstrumentation>;

template <bool Enabled>
class SchedulerScalarIterationImpl;

template <>
class SchedulerScalarIterationImpl<true> {
public:
    inline __aicore__ SchedulerScalarIterationImpl(SchedulerScalarTrace *trace) :
        trace_(trace) {}
    inline __aicore__ ~SchedulerScalarIterationImpl() {
        if (trace_ != nullptr && trace_->buffer != nullptr) trace_->end_iteration(progress);
    }
    bool progress{false};
    SchedulerScalarIterationImpl(const SchedulerScalarIterationImpl &) = delete;
    SchedulerScalarIterationImpl &operator=(const SchedulerScalarIterationImpl &) = delete;

private:
    SchedulerScalarTrace *trace_;
};

template <>
class SchedulerScalarIterationImpl<false> {
public:
    inline __aicore__ SchedulerScalarIterationImpl(SchedulerScalarTrace *) {}
    bool progress{false};
    SchedulerScalarIterationImpl(const SchedulerScalarIterationImpl &) = delete;
    SchedulerScalarIterationImpl &operator=(const SchedulerScalarIterationImpl &) = delete;
};

using SchedulerScalarIteration = SchedulerScalarIterationImpl<kSchedulerScalarInstrumentation>;

// The shared result wrapper keeps its depth/clock bookkeeping outside the
// surrounding scheduler loop. End when the returned value is consumable by
// this Scalar; a store need not yet be globally visible to another core.
template <typename Local, typename Operation>
static __aicore__ __attribute__((noinline)) auto
scheduler_scalar_result_runtime(const Local *local, SchedPhaseKind kind, Operation operation) {
    auto *trace = local != nullptr && local->profiling != nullptr ? &local->profiling->scalar : nullptr;
    if (trace == nullptr || trace->buffer == nullptr) return operation();
    const uint32_t depth = ++trace->depth;
    const uint64_t begin = scheduler_cycles();
    auto result = operation();
    uint64_t end = 0;
#if defined(__CCE_AICORE__)
    auto dependency = static_cast<uint64_t>(result);
    asm volatile("MOV %0, %0\nMOV %1, SYS_CNT\n" : "+l"(dependency), "=&l"(end));
#else
    end = scheduler_cycles();
#endif
    --trace->depth;
    trace->append_function(kind, begin, end, -1, depth);
    return result;
}

template <typename Local, typename Operation>
static inline __aicore__ __attribute__((always_inline)) auto scheduler_scalar_result(
    [[maybe_unused]] const Local *local, [[maybe_unused]] SchedPhaseKind kind, Operation operation
) {
    if constexpr (kSchedulerScalarInstrumentation) return scheduler_scalar_result_runtime(local, kind, operation);
    else return operation();
}
