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

class SchedulerScalarScope {
public:
    template <typename Local>
    inline __aicore__
    SchedulerScalarScope(const Local *local, SchedPhaseKind kind, int64_t task_ref = -1, uint64_t begin = 0) :
        trace_(local != nullptr && local->profiling != nullptr ? &local->profiling->scalar : nullptr),
        kind_(kind),
        task_ref_(task_ref) {
        if (trace_ != nullptr && trace_->buffer != nullptr) {
            depth_ = ++trace_->depth;
            begin_ = begin == 0 ? scheduler_cycles() : begin;
        } else trace_ = nullptr;
    }
    inline __aicore__ ~SchedulerScalarScope() { close(); }
    inline __aicore__ void close() {
        if (trace_ == nullptr) return;
        const uint64_t end = scheduler_cycles();
        --trace_->depth;
        trace_->append_function(kind_, begin_, end, task_ref_, depth_);
        trace_ = nullptr;
    }
    SchedulerScalarScope(const SchedulerScalarScope &) = delete;
    SchedulerScalarScope &operator=(const SchedulerScalarScope &) = delete;

private:
    SchedulerScalarTrace *trace_;
    SchedPhaseKind kind_;
    int64_t task_ref_;
    uint64_t begin_{0};
    uint32_t depth_{0};
};

class SchedulerScalarIteration {
public:
    inline __aicore__ SchedulerScalarIteration(SchedulerScalarTrace *trace) :
        trace_(trace) {}
    inline __aicore__ ~SchedulerScalarIteration() {
        if (trace_ != nullptr && trace_->buffer != nullptr) trace_->end_iteration(progress);
    }
    bool progress{false};
    SchedulerScalarIteration(const SchedulerScalarIteration &) = delete;
    SchedulerScalarIteration &operator=(const SchedulerScalarIteration &) = delete;

private:
    SchedulerScalarTrace *trace_;
};

// The shared result wrapper keeps its depth/clock bookkeeping outside the
// surrounding scheduler loop. End when the returned value is consumable by
// this Scalar; a store need not yet be globally visible to another core.
template <typename Local, typename Operation>
static __aicore__ __attribute__((noinline)) auto
scheduler_scalar_result(const Local *local, SchedPhaseKind kind, Operation operation) {
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
inline __aicore__ auto scheduler_scalar_atomic_result(const Local *local, Operation operation) {
    return scheduler_scalar_result(local, SchedPhaseKind::ScalarAtomicClaim, operation);
}
