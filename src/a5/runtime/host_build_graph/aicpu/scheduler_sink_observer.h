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

#include "aicore_scheduler_state.h"
#include "scheduler/scheduler_async_layout.h"
#include "scheduler/scheduler_types.h"

class AicpuSinkObserver {
public:
    bool initialize(const SchedulerWorkerContext &context, const SchedulerRunControl &run) {
        base_ = reinterpret_cast<void *>(context.scheduler_state_base_address);
        const uint64_t bytes = context.scheduler_state_bytes;
        tasks_ = context.graph_task_count;
        count_ = run.sink_count;
        if (base_ == nullptr || bytes > UINTPTR_MAX - reinterpret_cast<uintptr_t>(base_) ||
            run.inline_completed_count > tasks_ || count_ > tasks_ - run.inline_completed_count)
            return false;
        if (count_ == 0) return tasks_ == run.inline_completed_count;

        sinks_ = table<uint32_t>(bytes, run.sinks_offset, count_);
        controls_ = table<SchedulerTaskControl>(bytes, context.task_controls_offset, tasks_);
        counters_ = table<SchedulerTaskCompletionCounter>(bytes, context.completion_counters_offset, tasks_);
        if (sinks_ == nullptr || controls_ == nullptr ||
            (context.completion_counters_offset != 0 && counters_ == nullptr))
            return false;
        cache_invalidate_range(sinks_, count_ * sizeof(*sinks_));
        return true;
    }

    // DONE is irreversible after bootstrap reset. Keep the completed prefix
    // so a graph with many sinks never repeatedly polls the finished ones.
    int poll() {
        while (resolved_ < count_) {
            const uint32_t task = sinks_[resolved_];
            if (task >= tasks_) return -1;
            const int state = observe(task);
            if (state != 1) return state;
            ++resolved_;
        }
        return 1;
    }

    uint64_t resolved() const { return resolved_; }

    // A dependency chain can run for longer than the watchdog budget before
    // its sink moves. Observe actual claim/return/condition progress then.
    // CHECK polling cursors and lease generations advance even on a stall.
    uint64_t fingerprint(const SchedulerWorkerContext &context) const {
        uint64_t hash = UINT64_C(0xcbf29ce484222325);
        auto mix = [&](uint64_t value) {
            hash = (hash ^ value) * UINT64_C(0x100000001b3);
        };
        for (uint64_t task = 0; task < tasks_; ++task) {
            const auto &control = controls_[task];
            if (control.supply_address != 0)
                mix(*reinterpret_cast<const volatile uint32_t *>(control.supply_address) & control.supply_mask);
            else mix(aicpu_observe_gm(control.next_block));
            mix(aicpu_observe_gm(controls_[task].completion_state));
            if (counters_ != nullptr && counters_[task].read_address != 0)
                mix(*reinterpret_cast<const volatile uint32_t *>(counters_[task].read_address));
        }
        const auto *summary = scheduler_state_at<SchedulerGraphSummary>(base_, context.graph_summary_offset);
        if (summary->async_layout_offset != 0) {
            const auto *async = scheduler_state_at<SchedulerAsyncLayout>(base_, summary->async_layout_offset);
            const auto *groups = scheduler_state_at<SchedulerAsyncGroupProgress>(base_, async->group_progress_offset);
            for (uint32_t i = 0; i < async->group_count; ++i)
                mix(aicpu_observe_gm(groups[i].pending_mask));
        }
        return hash;
    }

private:
    template <typename T>
    const T *table(uint64_t bytes, uint64_t offset, uint64_t count) const {
        if (offset == 0 || offset > bytes || count > (bytes - offset) / sizeof(T) ||
            (reinterpret_cast<uintptr_t>(base_) + offset) % alignof(T) != 0)
            return nullptr;
        return scheduler_state_at<T>(base_, offset);
    }

    int observe(uint32_t task) const {
        if (counters_ != nullptr && (counters_[task].add_address != 0 || counters_[task].read_address != 0)) {
            const auto &counter = counters_[task];
            if (counter.read_address == 0 || counter.threshold == 0 || counter.threshold > counter.limit) return -1;
            const uint32_t value = *reinterpret_cast<const volatile uint32_t *>(counter.read_address);
            return scheduler_classify_completion_counter(value, counter.threshold, counter.limit);
        }
        const uint64_t state = aicpu_observe_gm(controls_[task].completion_state);
        if (!scheduler_completion_valid(state)) return -1;
        return scheduler_completion_done(state) ? 1 : 0;
    }

    void *base_{};
    uint64_t tasks_{};
    uint64_t count_{};
    uint64_t resolved_{};
    const uint32_t *sinks_{};
    const SchedulerTaskControl *controls_{};
    const SchedulerTaskCompletionCounter *counters_{};
};
