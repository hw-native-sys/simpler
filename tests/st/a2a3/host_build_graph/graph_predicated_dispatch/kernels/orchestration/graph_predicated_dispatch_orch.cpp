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
 * graph_predicated_dispatch orchestration: dispatch predicates inside a Graph.
 *
 * One fixed Graph body is invoked four times with its own boundary tensors and
 * its own boundary scalar. Invocation 0 records the Definition; the rest replay
 * it as outer Graph tasks.
 *
 * The body holds three predicated tasks, one per operand source, so a single
 * Definition covers every rebind kind materialize can take:
 *
 *   gate_producer (WRITE_GATE)   internal INT32 tensor := boundary scalar
 *   sentinel      (WRITE_CONST)  X[0] := 42.0
 *   clobber_b     (CLOBBER)      X[0] := 999.0,  predicate gate[0] > 0
 *                                                (BOUNDARY_EXACT)
 *   clobber_i     (CLOBBER_ALT)  X[0] := 555.0,  predicate internal_gate[0] > 0
 *                                                (INTERNAL)
 *   consumer      (COPY_FIRST)   Y[0] := X[0],   predicate gate_view[1] > 0
 *                                                (BOUNDARY_VIEW)
 *
 * The X tasks chain through X, so the last clobber to dispatch decides what the
 * consumer copies; a suppressed consumer leaves Y at the caller's fill value.
 *
 * gate_view is gate[GATE_VIEW_BASE ..] and the predicate indexes element 1 of
 * it, so its address is the boundary's base plus a view offset plus an element
 * offset — three offsets no other branch stacks. gate[GATE_VIEW_BASE + 1] is the
 * only element that drives it, and gate[0] drives clobber_b, so dropping any one
 * of the three reads an element that is zero in every invocation and suppresses
 * the consumer where it must run.
 *
 * Per invocation, (gate[0], internal scalar, gate[GATE_VIEW_BASE + 1]) is
 * (0,0,1) / (1,0,1) / (0,1,1) / (0,0,0), giving four distinct outcomes. Each
 * invocation passes its own gate buffer and scalar, so an operand address frozen
 * at record time would replay invocation 0's gates and collapse them all.
 */

#include <stdint.h>

#include <array>

#include "orchestration_api.h"  // NOLINT(build/include_subdir)

#define FUNC_WRITE_CONST 0
#define FUNC_COPY_FIRST 1
#define FUNC_CLOBBER 2
#define FUNC_WRITE_GATE 3
#define FUNC_CLOBBER_ALT 4

#define GRAPH_INVOCATIONS 4
#define GATE_ELEMS 16
#define GATE_VIEW_BASE 4
#define GATE_VIEW_ELEMS 4

namespace {

CoreTaskPredicate gate_predicate(const simpler::hbg::Tensor &gate, uint32_t index) {
    CoreTaskPredicate pred;
    pred.operand.tensor = &gate;
    pred.operand.ndims = 1;
    pred.operand.indices[0] = index;
    pred.op = PredicateOp::GT;
    pred.target = 0;
    return pred;
}

// A construction parameter this body does not read. It still decides which Definition the
// submission reaches -- the key carries its hash -- so it exercises the submit overload that
// pairs a Config with a body that takes none.
struct LayerConfig : simpler::hbg::ConfigBase {
    explicit LayerConfig(int v) :
        variant(v) {}

    int variant;

    uint64_t hash() const { return simpler::hbg::config_hash(variant); }
    bool operator==(const LayerConfig &other) const { return variant == other.variant; }
};

void layer(const GraphTaskArgs &args) {
    const simpler::hbg::Tensor &x = args.tensor(0).ref();
    const simpler::hbg::Tensor &y = args.tensor(1).ref();
    const simpler::hbg::Tensor &boundary_gate = args.tensor(2).ref();

    // A view of the boundary gate: same buffer, shifted start_offset, so the
    // recorder classifies it BOUNDARY_VIEW rather than BOUNDARY_EXACT.
    const std::array<uint32_t, 1> view_shape{GATE_VIEW_ELEMS};
    const std::array<uint32_t, 1> view_offset{GATE_VIEW_BASE};
    const simpler::hbg::Tensor gate_view = boundary_gate.view(view_shape.data(), view_offset.data());

    // Graph-internal predicate operand: its address is the recording's virtual
    // one, so replay can only read it once materialize has rebound the tensor.
    const std::array<uint32_t, 1> gate_shape{GATE_ELEMS};
    TensorCreateInfo gate_info(gate_shape.data(), static_cast<uint32_t>(gate_shape.size()), DataType::INT32);
    CoreTaskArgs gate_args;
    gate_args.add_output(gate_info);
    gate_args.add_scalar(args.scalar(0));
    TaskOutputTensors gate_outputs = rt_submit_aic_task(FUNC_WRITE_GATE, gate_args);
    simpler::hbg::Tensor internal_gate = gate_outputs.get_ref(0);
    TaskId gate_tid = gate_outputs.task_id();

    // X is a boundary tensor, so the four tasks below share no Graph-internal
    // tensor and the recorder derives no edge between them. Their write-then-read
    // order is stated explicitly, which is what a recorded Graph preserves.
    TaskId sentinel_tid;
    {
        CoreTaskArgs sentinel_args;
        sentinel_args.add_inout(x);
        sentinel_tid = rt_submit_aic_task(FUNC_WRITE_CONST, sentinel_args).task_id();
    }

    TaskId clobber_tid;
    {
        CoreTaskArgs clobber_args;
        clobber_args.add_inout(x);
        const std::array<TaskId, 1> deps{sentinel_tid};
        clobber_args.set_dependencies(deps.data(), static_cast<uint32_t>(deps.size()));
        clobber_args.set_predicate(gate_predicate(boundary_gate, 0));
        clobber_tid = rt_submit_aic_task(FUNC_CLOBBER, clobber_args).task_id();
    }

    TaskId clobber_alt_tid;
    {
        CoreTaskArgs clobber_args;
        clobber_args.add_inout(x);
        // The predicate is a condition, not a dependency: the operand's producer
        // is named here so the value is written by the time this task is ready.
        const std::array<TaskId, 2> deps{clobber_tid, gate_tid};
        clobber_args.set_dependencies(deps.data(), static_cast<uint32_t>(deps.size()));
        clobber_args.set_predicate(gate_predicate(internal_gate, 0));
        clobber_alt_tid = rt_submit_aic_task(FUNC_CLOBBER_ALT, clobber_args).task_id();
    }

    {
        CoreTaskArgs consumer_args;
        consumer_args.add_input(x);
        consumer_args.add_inout(y);
        const std::array<TaskId, 1> deps{clobber_alt_tid};
        consumer_args.set_dependencies(deps.data(), static_cast<uint32_t>(deps.size()));
        consumer_args.set_predicate(gate_predicate(gate_view, 1));
        rt_submit_aic_task(FUNC_COPY_FIRST, consumer_args);
    }
}

}  // namespace

extern "C" {

__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &args) {
    (void)args;
    return OrchestrationConfig{
        .expected_arg_count = 3 * GRAPH_INVOCATIONS + GRAPH_INVOCATIONS,
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &orch_args) {
    for (int32_t i = 0; i < GRAPH_INVOCATIONS; ++i) {
        GraphTaskArgs layer_args;
        layer_args.add_inout(orch_args.tensor(i).ref());
        layer_args.add_inout(orch_args.tensor(GRAPH_INVOCATIONS + i).ref());
        layer_args.add_input(orch_args.tensor(2 * GRAPH_INVOCATIONS + i).ref());
        layer_args.add_scalar(orch_args.scalar(i));
        rt_submit_graph(&layer, layer_args, LayerConfig{0});
    }
}

}  // extern "C"
