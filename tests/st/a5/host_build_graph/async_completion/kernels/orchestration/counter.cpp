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

#include <stdexcept>
#include <tuple>
#include <vector>

#include "orchestration_api.h"
#include "../layout.h"

namespace {
using async_case::Action;
using async_case::Plan;
using async_case::Scenario;

struct BodyConfig : simpler::hbg::ConfigBase {
    Plan plan;
    explicit BodyConfig(const Plan &value) :
        plan(value) {}
    uint64_t hash() const {
        return simpler::hbg::config_hash(
            plan.scenario, plan.conditions, plan.waiters, plan.blocks, plan.mask, plan.expected, plan.epoch,
            plan.delay_ms, plan.tail_ms, plan.sync_start, plan.instances, plan.skip, plan.capacity_delta, plan.slots
        );
    }
    auto fields() const {
        return std::tie(
            plan.scenario, plan.conditions, plan.waiters, plan.blocks, plan.mask, plan.expected, plan.epoch,
            plan.delay_ms, plan.tail_ms, plan.sync_start, plan.instances, plan.skip, plan.capacity_delta, plan.slots
        );
    }
    bool operator==(const BodyConfig &other) const { return fields() == other.fields(); }
};

template <typename Args>
void tensors(CoreTaskArgs &task, const Args &args) {
    for (int32_t i = 0; i < 5; ++i)
        task.add_no_dep(args.tensor(i).ref());
}

constexpr auto kCleanLines = AsyncUseCacheContract::PUBLISHED_CLEAN_LINES;

void declare_payload_consumer(TaskId task) {
    // The shared argument signature includes payload even for actions that do
    // not read it. Its lines are clean at entry; publishers flush payload before
    // signalling, and consumers write results into a separate tensor.
    if (!rt_declare_async_consumer(task, 1, kCleanLines) || !rt_declare_async_core_only(task))
        throw std::runtime_error("async fixture payload declaration failed");
}

void declare_waiter(TaskId task, bool has_predicate = false) {
    if (!rt_declare_async_counter_payload(task, 0, 1, async_case::kSignalStride * sizeof(int32_t), kCleanLines))
        throw std::runtime_error("async fixture counter declaration failed");
    declare_payload_consumer(task);
    // The predicate reads a separate control line before kernel entry. Control
    // updates use GM atomics, so this acquisition has no dirty scalar alias.
    if (has_predicate && !rt_declare_async_consumer(task, 4, kCleanLines))
        throw std::runtime_error("async fixture predicate declaration failed");
}

template <typename Args, typename Instance, typename Base, typename Record>
TaskId action(
    const Args &args, const Plan &p, Action op, Instance instance, Base base, int64_t waiters, Record record,
    int64_t auxiliary, const std::vector<TaskId> &deps = {}, int64_t delay = 0
) {
    CoreTaskArgs task;
    tensors(task, args);
    task.add_scalar(static_cast<int64_t>(op), base, waiters, p.blocks, p.mask, p.conditions, p.expected, p.epoch);
    task.add_scalar(instance, delay, record, auxiliary);
    if (!deps.empty()) task.set_dependencies(deps.data(), static_cast<uint32_t>(deps.size()));
    if (op == Action::WARM) {
        task.launch_spec.set_block_num(rt_available_cluster_count());
        task.launch_spec.set_require_sync_start(true);
    }
    TaskId submitted;
    if (op == Action::RESCUE) submitted = rt_submit_aic_task(4, task).task_id();
    else if (op == Action::WARM) submitted = rt_submit_aic_task(7, task).task_id();
    else if (op == Action::CONSUME) {
        if (static_cast<Scenario>(p.scenario) == Scenario::VISIBILITY)
            submitted = rt_submit_aic_task(8, task).task_id();
        else submitted = rt_submit_aiv_task(5, task).task_id();
    } else submitted = rt_submit_aiv_task(op == Action::PREREQUISITE ? 6 : 3, task).task_id();
    declare_payload_consumer(submitted);
    return submitted;
}

template <typename Args>
TaskId registration_gate(const Args &args, int64_t execution) {
    const uint32_t signal_shape[2]{1, 1};
    const uint32_t signal_offset[2]{
        static_cast<uint32_t>(execution * async_case::kConditionsPerExecution + async_case::kConditionLimit), 0
    };
    const auto registration_signal = args.tensor(0).ref().view(signal_shape, signal_offset);
    CoreTaskArgs gate;
    gate.add_no_dep(registration_signal);
    gate.add_no_dep(args.tensor(2).ref());
    gate.add_scalar(execution);
    const TaskId registered = rt_submit_aiv_task(10, gate).task_id();
    if (!rt_declare_async_counter(registered, 0, async_case::kSignalStride * sizeof(int32_t), kCleanLines) ||
        !rt_declare_async_core_only(registered))
        throw std::runtime_error("async fixture registration gate declaration failed");
    return registered;
}

template <typename Args>
void body_tasks(const Args &args, const Plan &p, int64_t instance) {
    const auto scenario = static_cast<Scenario>(p.scenario);
    if (scenario == Scenario::PROGRESS &&
        (p.mask != 2 || p.blocks != 1 || p.sync_start || p.skip || p.conditions > async_case::kConditionLimit))
        throw std::runtime_error("progress registration gates require ordinary AIV waiters with at most 64 conditions");
    const int64_t capacity = scenario == Scenario::VISIBILITY ? rt_available_cluster_count() : rt_available_aiv_count();
    const int64_t max_waiters = p.slots / (p.instances * p.blocks * 3);
    const int64_t waiters = scenario == Scenario::PROGRESS ? capacity + p.capacity_delta : p.waiters;
    if (waiters < 1 || waiters > max_waiters) throw std::runtime_error("async fixture allocation is too small");
    const int64_t base = instance * max_waiters * p.blocks * 3;
    const int64_t publisher_record = p.slots + max_waiters * p.instances + instance;
    std::vector<TaskId> warm;
    if (scenario == Scenario::VISIBILITY)
        warm.push_back(action(args, p, Action::WARM, instance, base, waiters, int64_t{-1}, 0));

    TaskId prerequisite = TaskId::invalid();
    if (scenario == Scenario::FANIN)
        prerequisite = action(
            args, p, Action::PREREQUISITE, instance, base, waiters, publisher_record + p.instances, 0, warm, p.tail_ms
        );

    MixedKernels kernels;
    if (p.mask & 1) kernels.aic_kernel_id = 0;
    if (p.mask & 2) kernels.aiv0_kernel_id = 1;
    if (p.mask & 4) kernels.aiv1_kernel_id = 2;
    std::vector<TaskId> publisher_deps = warm;
    for (int64_t waiter = 0; waiter < waiters; ++waiter) {
        const int64_t execution = base + waiter * p.blocks * 3;
        CoreTaskArgs task;
        tensors(task, args);
        task.add_scalar(execution, p.conditions, p.expected, instance, scenario == Scenario::FANIN ? 0 : p.tail_ms);
        task.launch_spec.set_block_num(static_cast<int32_t>(p.blocks));
        task.launch_spec.set_require_sync_start(p.sync_start != 0);
        if (!warm.empty()) task.set_dependencies(warm.data(), static_cast<uint32_t>(warm.size()));
        if (p.skip) {
            CoreTaskPredicate predicate;
            predicate.operand.tensor = &args.tensor(4).ref();
            predicate.operand.ndims = 3;
            predicate.operand.indices[0] = 0;
            predicate.operand.indices[1] = 15;
            predicate.operand.indices[2] = 0;
            predicate.op = PredicateOp::GT;
            predicate.target = 0;
            task.set_predicate(predicate);
        }
        const TaskId waited = scenario == Scenario::PROGRESS ? rt_submit_aiv_task(9, task).task_id() :
                                                               rt_submit_task(kernels, task).task_id();
        declare_waiter(waited, p.skip != 0);
        if (scenario == Scenario::PROGRESS) publisher_deps.push_back(registration_gate(args, execution + 1));
        std::vector<TaskId> deps{waited};
        if (prerequisite.is_valid()) deps.push_back(prerequisite);
        action(
            args, p, Action::CONSUME, instance, execution, 1, p.slots + instance * max_waiters + waiter,
            p.skip ? 2 : (scenario == Scenario::FANIN ? 1 : 0), deps
        );
    }
    if (scenario != Scenario::NEVER && !p.skip)
        action(
            args, p, Action::PUBLISH, instance, base, waiters, publisher_record, capacity, publisher_deps, p.delay_ms
        );
    if (scenario == Scenario::PROGRESS) action(args, p, Action::RESCUE, instance, base, waiters, int64_t{-1}, 0);
}

void graph_body(const GraphTaskArgs &args, const BodyConfig &config) {
    const Plan &p = config.plan;
    CoreTaskArgs waiter;
    tensors(waiter, args);
    // Forward parameter handles: reading an instance's values while recording would
    // freeze them into the shared Graph definition and invalidate this isolation test.
    waiter.add_scalar(args.scalar(1), p.conditions, p.expected, args.scalar(0), p.tail_ms);
    const TaskId waited = rt_submit_aiv_task(9, waiter).task_id();
    declare_waiter(waited);
    action(args, p, Action::CONSUME, args.scalar(0), args.scalar(1), 1, args.scalar(2), 0, {waited});
}
}  // namespace

extern "C" {
__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &) {
    return OrchestrationConfig{.expected_arg_count = 19};
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    const Plan p{args.scalar<int64_t>(0),  args.scalar<int64_t>(1), args.scalar<int64_t>(2),  args.scalar<int64_t>(3),
                 args.scalar<int64_t>(4),  args.scalar<int64_t>(5), args.scalar<int64_t>(6),  args.scalar<int64_t>(7),
                 args.scalar<int64_t>(8),  args.scalar<int64_t>(9), args.scalar<int64_t>(10), args.scalar<int64_t>(11),
                 args.scalar<int64_t>(12), args.scalar<int64_t>(13)};
    for (int64_t instance = 0; instance < p.instances; ++instance) {
        if (static_cast<Scenario>(p.scenario) == Scenario::GRAPH) {
            if (p.conditions > async_case::kConditionLimit)
                throw std::runtime_error("Graph fixture registration marker requires at most 64 business conditions");
            GraphTaskArgs graph;
            for (int32_t i = 0; i < 5; ++i)
                graph.add_no_dep(args.tensor(i).ref());
            const int64_t max_waiters = p.slots / (p.instances * p.blocks * 3);
            const int64_t base = instance * max_waiters * p.blocks * 3;
            const int64_t execution = base + 1;
            graph.add_scalar(instance, base, p.slots + instance * max_waiters);
            const auto invocation = rt_submit_graph(&graph_body, graph, BodyConfig{p});
            // A second consumer outside EXIT validates transitive completion and separate invocation bindings.
            action(
                args, p, Action::CONSUME, instance, base, 1,
                p.slots + max_waiters * p.instances + p.instances + instance, 0, {invocation.task_id}
            );
            // Keep the publisher outside the Graph. Otherwise its own compute
            // completion could hold EXIT even if the async child retired early.
            // Registration gates release their compute slot while the Graph
            // waiter is pending; they do not wait for the waiter's TaskDone.
            const TaskId registered = registration_gate(args, execution);
            action(
                args, p, Action::PUBLISH, instance, base, 1, p.slots + max_waiters * p.instances + instance,
                rt_available_aiv_count(), {registered}, p.delay_ms
            );
        } else {
            body_tasks(args, p, instance);
        }
    }
}
}  // extern "C"
