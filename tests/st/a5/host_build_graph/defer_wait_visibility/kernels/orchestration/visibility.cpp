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
#include "orchestration_api.h"
namespace {
using View = simpler::hbg::Tensor;
constexpr uint32_t kOutputWords = 4224;
struct Config : simpler::hbg::ConfigBase {
    int32_t core;
    explicit Config(int32_t value) :
        core(value) {}
    uint64_t hash() const { return simpler::hbg::config_hash(core); }
    bool operator==(const Config &other) const { return core == other.core; }
};
TaskId wait(const View &signal, const TaskId *dependency) {
    CoreTaskArgs args;
    args.add_no_dep(signal);
    if (dependency != nullptr) args.set_dependencies(dependency, 1);
    return rt_submit_aiv_task(2, args).task_id();
}
TaskId consume(const View &predicate, const View &payload, const View &out, int32_t core, const TaskId &dependency) {
    const uint32_t shape[]{16}, left_offset[]{0}, right_offset[]{8};
    const auto left = payload.view(shape, left_offset);
    const auto right = payload.view(shape, right_offset);
    CoreTaskArgs args;
    args.add_no_dep(left, right);
    args.add_output(out);
    args.add_scalar(3, 0);
    args.set_dependencies(&dependency, 1);
    CoreTaskPredicate condition;
    condition.operand.tensor = &predicate;
    condition.operand.ndims = 1;
    condition.operand.indices[0] = 0;
    condition.op = PredicateOp::GT;
    condition.target = 0;
    args.set_predicate(condition);
    return (core == 0 ? rt_submit_aic_task(0, args) : rt_submit_aiv_task(1, args)).task_id();
}
void graph_body(const GraphTaskArgs &args, const Config &config) {
    const auto waited = wait(args.tensor(0).ref(), nullptr);
    (void)consume(args.tensor(1).ref(), args.tensor(2).ref(), args.tensor(3).ref(), config.core, waited);
}
}  // namespace
extern "C" {
__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &) {
    return OrchestrationConfig{.expected_arg_count = 7};
}
__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    const int32_t core = static_cast<int32_t>(args.scalar<int64_t>(0));
    const bool graph = args.scalar<int64_t>(1) != 0;
    const int32_t target = static_cast<int32_t>(args.scalar<int64_t>(2));
    const uint32_t region[]{32}, out_shape[]{kOutputWords};
    TaskId previous = TaskId::invalid();
    for (uint32_t instance = 0; instance < 2; ++instance) {
        const uint32_t offset[]{instance * 32}, out_offset[]{instance * kOutputWords};
        const auto signal = args.tensor(0).ref().view(region, offset);
        const auto predicate = args.tensor(1).ref().view(region, offset);
        const auto payload = args.tensor(2).ref().view(region, offset);
        const auto out = args.tensor(3).ref().view(out_shape, out_offset);
        CoreTaskArgs init;
        init.add_no_dep(signal, predicate, payload);
        init.add_scalar(0, static_cast<int64_t>(instance * 2 + target));
        if (previous.is_valid()) init.set_dependencies(&previous, 1);
        const auto initialized = rt_submit_aiv_task(1, init).task_id();
        CoreTaskArgs warm;
        warm.add_no_dep(predicate, payload);
        warm.add_output(out);
        warm.add_scalar(1, 0);
        warm.set_dependencies(&initialized, 1);
        warm.launch_spec.set_block_num(core == 0 ? rt_available_cluster_count() : rt_available_aiv_count());
        warm.launch_spec.set_require_sync_start(true);
        const auto warm_result = core == 0 ? rt_submit_aic_task(0, warm) : rt_submit_aiv_task(1, warm);
        const auto warmed = warm_result.task_id();
        TaskId consumed = TaskId::invalid();
        if (graph) {
            GraphTaskArgs call;
            call.add_input(signal, predicate, payload);
            call.add_inout(out);
            consumed = rt_submit_graph(&graph_body, call, Config{core}).task_id;
        } else {
            const auto waited = wait(signal, &warmed);
            consumed = consume(predicate, payload, out, core, waited);
        }
        CoreTaskArgs publisher;
        publisher.add_no_dep(signal, predicate, payload);
        publisher.add_scalar(2, static_cast<int64_t>(instance * 2 + target));
        publisher.set_dependencies(&warmed, 1);
        (void)rt_submit_aiv_task(1, publisher);
        CoreTaskArgs tail;
        tail.add_no_dep(payload, payload, out);
        tail.add_scalar(4, 0);
        tail.set_dependencies(&consumed, 1);
        previous = rt_submit_aiv_task(1, tail).task_id();
    }
}
}
