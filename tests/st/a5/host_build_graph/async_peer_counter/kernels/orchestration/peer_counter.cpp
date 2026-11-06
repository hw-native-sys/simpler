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
#include <array>
#include <cstdint>
#include <stdexcept>

#include "orchestration_api.h"
#include "../common.h"

using namespace peer_counter;

namespace {
constexpr auto clean = AsyncUseCacheContract::PUBLISHED_CLEAN_LINES;

// Host-only owning storage: CoreTaskArgs stores Tensor pointers, so each view
// must remain at a stable address until the corresponding submit has returned.
struct TaskViews {
    CoreTaskArgs args;
    std::array<Tensor, 5> views;
    uint32_t count = 0;

    TaskViews() = default;
    TaskViews(const TaskViews &) = delete;
    TaskViews &operator=(const TaskViews &) = delete;
};

void add_view(TaskViews &task, const ChipTaskArgs &args, uint32_t argument, uint32_t words, uint32_t offset = 0) {
    if (task.count >= task.views.size()) throw std::runtime_error("peer COUNTER task view capacity exceeded");
    const uint32_t shape[1]{words};
    const uint32_t offsets[1]{offset};
    task.views[task.count] = args.tensor(argument).ref().view(shape, offsets);
    task.args.add_no_dep(task.views[task.count]);
    ++task.count;
}

void add_diagnostic(TaskViews &task, const ChipTaskArgs &args, uint32_t row) {
    add_view(task, args, 5, kDiagnosticWords, row * kDiagnosticWords);
}

void declare_core_only(TaskId task) {
    if (!rt_declare_async_core_only(task)) throw std::runtime_error("peer COUNTER core-only declaration failed");
}

void declare_waiter(TaskId task) {
    if (!rt_declare_async_counter_payload(task, 0, 1, kSignalWords * sizeof(int32_t), clean))
        throw std::runtime_error("peer COUNTER payload binding failed");
    declare_core_only(task);
}

void submit_source(const ChipTaskArgs &args, int64_t context, int64_t epoch, int64_t tag, int64_t pressure) {
    TaskViews waiter;
    add_view(waiter, args, 4, kSignalWords);
    add_view(waiter, args, 3, kReadyWords);
    add_diagnostic(waiter, args, 2 * kMaxRows + 2);
    add_view(waiter, args, 7, kRegistrationWords);
    waiter.args.add_scalar(epoch);
    waiter.args.add_scalar(tag);
    const TaskId ready_waiter = rt_submit_aiv_task(0, waiter.args).task_id();
    declare_waiter(ready_waiter);

    TaskViews publisher;
    add_view(publisher, args, 0, kMaxRows * kRowWords);
    add_view(publisher, args, 1, kMaxRows * kRowWords);
    add_view(publisher, args, 2, kMaxRows * kSignalWords);
    add_view(publisher, args, 3, kReadyWords);
    add_diagnostic(publisher, args, 2 * kMaxRows + 3);
    publisher.args.add_scalar(context);
    publisher.args.add_scalar(epoch);
    publisher.args.add_scalar(tag);
    publisher.args.add_scalar(static_cast<int64_t>(kMaxRows));
    publisher.args.add_scalar(pressure);
    publisher.args.set_dependencies(&ready_waiter, 1);
    const TaskId publication = rt_submit_aiv_task(3, publisher.args).task_id();
    declare_core_only(publication);
}

void submit_receiver(const ChipTaskArgs &args, int64_t context, int64_t epoch, int64_t tag, int64_t pressure) {
    const int64_t capacity = rt_available_aiv_count();
    const int64_t rows = pressure == 0 ? 1 : capacity + 1;
    if (capacity <= 0 || capacity > kMaxRows || rows > kMaxRows)
        throw std::runtime_error("peer COUNTER public AIV capacity exceeds fixture bounds");
    for (uint32_t row = 0; row < static_cast<uint32_t>(rows); ++row) {
        TaskViews waiter;
        add_view(waiter, args, 2, kSignalWords, row * kSignalWords);
        add_view(waiter, args, 1, kDataWords, row * kRowWords);
        add_diagnostic(waiter, args, row);
        add_view(waiter, args, 7, kRegistrationWords);
        waiter.args.add_scalar(epoch);
        waiter.args.add_scalar(tag);
        const TaskId main_waiter = rt_submit_aiv_task(0, waiter.args).task_id();
        declare_waiter(main_waiter);

        TaskViews consumer;
        add_view(consumer, args, 1, kDataWords, row * kRowWords);
        add_view(consumer, args, 2, kSignalWords, row * kSignalWords);
        add_view(consumer, args, 6, kDataWords, row * kRowWords);
        add_diagnostic(consumer, args, kMaxRows + row);
        consumer.args.add_scalar(epoch);
        consumer.args.add_scalar(tag);
        consumer.args.add_scalar(static_cast<int64_t>(row));
        consumer.args.add_scalar(INT64_C(0));
        consumer.args.set_dependencies(&main_waiter, 1);
        const TaskId consumption = rt_submit_aiv_task(4, consumer.args).task_id();
        declare_core_only(consumption);
    }

    // No waiter TaskDone dependency: the separate AIC checks only the test's
    // actual local registration counter while the main conditions stay pending.
    TaskViews gate;
    add_view(gate, args, 7, kRegistrationWords);
    add_diagnostic(gate, args, 2 * kMaxRows);
    gate.args.add_scalar(rows);
    gate.args.add_scalar(capacity);
    gate.args.add_scalar(epoch);
    gate.args.add_scalar(tag);
    const TaskId registration_gate = rt_submit_aic_task(1, gate.args).task_id();
    declare_core_only(registration_gate);

    TaskViews sender;
    add_view(sender, args, 3, kReadyWords);
    add_view(sender, args, 4, kSignalWords);
    add_diagnostic(sender, args, 2 * kMaxRows);
    add_diagnostic(sender, args, 2 * kMaxRows + 1);
    sender.args.add_scalar(context);
    sender.args.add_scalar(epoch);
    sender.args.add_scalar(tag);
    sender.args.add_scalar(static_cast<int64_t>(kMaxRows));
    sender.args.set_dependencies(&registration_gate, 1);
    const TaskId ready_sender = rt_submit_aiv_task(2, sender.args).task_id();
    declare_core_only(ready_sender);
}
}  // namespace

extern "C" __attribute__((visibility("default"))) OrchestrationConfig
aicpu_orchestration_config(const ChipTaskArgs &args) {
    (void)args;
    return OrchestrationConfig{.expected_arg_count = 14};
}

extern "C" __attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    const int64_t context = args.scalar<int64_t>(0), role = args.scalar<int64_t>(1);
    const int64_t epoch = args.scalar<int64_t>(2), tag = args.scalar<int64_t>(3);
    const int64_t pressure = args.scalar<int64_t>(4), maximum_rows = args.scalar<int64_t>(5);
    if (context == 0 || (role != 0 && role != 1) || epoch <= 0 || epoch > INT32_MAX || tag <= 0 || tag > kMaxTag ||
        (pressure != 0 && pressure != 1) || maximum_rows != kMaxRows)
        throw std::runtime_error("peer COUNTER requires the fixed two-rank tensor/scalar contract");
    SIMPLER_SCOPE() {
        if (role == 0) submit_source(args, context, epoch, tag, pressure);
        else submit_receiver(args, context, epoch, tag, pressure);
    }
}
