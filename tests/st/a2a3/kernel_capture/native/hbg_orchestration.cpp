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

static void
chain(const simpler::hbg::Tensor &input, const simpler::hbg::Tensor &output, bool early, uint64_t increment = 3) {
    uint32_t shape[] = {16};
    TensorCreateInfo intermediate(shape, 1, DataType::INT32);
    CoreTaskArgs first;
    first.add_input(input);
    first.add_output(intermediate);
    first.add_scalar(increment);
    first.set_allow_early_resolve(early);
    auto result = rt_submit_aiv_task(0, first);
    const auto &middle = result.get_ref(0);
    CoreTaskArgs second;
    second.add_input(middle);
    second.add_output(output);
    second.add_scalar(uint64_t{7});
    rt_submit_aiv_task(0, second);
}

extern "C" void hbg_capture_orchestration(const ChipTaskArgs &args) {
    const auto &input = args.tensor(0).ref();
    const auto &output = args.tensor(1).ref();
    const auto scenario = args.scalar<uint64_t>(0);
    if (scenario == 12) {
        CoreTaskArgs task;
        task.add_input(input);
        task.add_output(output);
        task.add_scalar(uint64_t{3});
        rt_submit_aiv_task(0, task);
        return;
    }
    if (scenario != 2) {
        const uint32_t index[]{0};
        const auto increment = scenario == 6 ?
                                   static_cast<uint64_t>(get_tensor_data<int32_t>(args.tensor(2).ref(), 1, index)) :
                                   uint64_t{3};
        chain(input, output, scenario == 1, increment);
        return;
    }
    GraphTaskArgs boundary;
    boundary.add_input(input);
    boundary.add_output(output);
    const auto graph = rt_graph_begin(0xCADD, boundary);
    if (graph.recording) {
        always_assert(rt_graph_prepare(graph.recording_handle, boundary));
        chain(graph.params->tensor(0).ref(), graph.params->tensor(1).ref(), false);
        always_assert(rt_graph_end());
    }
}

extern "C" void hbg_inout_orchestration(const ChipTaskArgs &args) {
    CoreTaskArgs task;
    task.add_inout(args.tensor(0).ref());
    task.add_scalar(args.scalar<int32_t>(0));
    rt_submit_aiv_task(0, task);
}
