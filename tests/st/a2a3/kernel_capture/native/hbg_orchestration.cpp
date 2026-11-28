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

template <typename Increment>
void chain(
    const simpler::hbg::Tensor &input, const simpler::hbg::Tensor &output, bool early, Increment increment,
    uint32_t scratch_elements = 16
) {
    uint32_t shape[] = {scratch_elements};
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

void nested_chain(const GraphTaskArgs &args) { chain(args.tensor(0).ref(), args.tensor(1).ref(), false, uint64_t{3}); }

struct SharedHeapConfig : simpler::hbg::ConfigBase {
    explicit SharedHeapConfig(uint32_t elements) :
        scratch_elements(elements) {}

    uint32_t scratch_elements;

    uint64_t hash() const { return simpler::hbg::config_hash(scratch_elements); }
    bool operator==(const SharedHeapConfig &other) const { return scratch_elements == other.scratch_elements; }
};

void shared_heap_chain(const GraphTaskArgs &args, const SharedHeapConfig &config) {
    const auto &state = args.tensor(0).ref();
    chain(state, state, false, args.scalar(0), config.scratch_elements);
}

}  // namespace

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
    if (scenario == 16) {
        // Grow the image to ten tasks with a single writer of the external output.
        // The successor of each intermediate must observe exactly one producer.
        const uint32_t shape[]{16};
        TensorCreateInfo intermediate(shape, 1, DataType::INT32);
        auto current = input;
        for (int i = 0; i < 9; ++i) {
            CoreTaskArgs step;
            step.add_input(current);
            step.add_output(intermediate);
            step.add_scalar(uint64_t{1});
            auto values = rt_submit_aiv_task(0, step);
            current = values.get_ref(0);
        }
        CoreTaskArgs last;
        last.add_input(current);
        last.add_output(output);
        last.add_scalar(uint64_t{5});
        rt_submit_aiv_task(0, last);
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
    rt_submit_graph(&nested_chain, boundary);
}

extern "C" void hbg_inout_orchestration(const ChipTaskArgs &args) {
    const int32_t increment = args.scalar<int32_t>(0);
    // The retained-model test records B with five dependent increments.
    const int32_t tasks = increment == -5 ? 5 : 1;
    for (int32_t i = 0; i < tasks; ++i) {
        CoreTaskArgs task;
        task.add_inout(args.tensor(0).ref());
        task.add_scalar(tasks == 1 ? increment : 1);
        rt_submit_aiv_task(0, task);
    }
}

extern "C" void hbg_shared_heap_orchestration(const ChipTaskArgs &args) {
    const int32_t increment = args.scalar<int32_t>(0);
    GraphTaskArgs boundary;
    boundary.add_inout(args.tensor(0).ref());
    boundary.add_scalar(static_cast<uint64_t>(increment - 7));
    // Captured models each require 64 KiB of scratch. The eager request needs
    // 256 KiB, which exceeds the fixed shared capacity of 128 KiB.
    const uint32_t scratch_elements = increment == 17 ? 65536 : 16384;
    rt_submit_graph(&shared_heap_chain, boundary, SharedHeapConfig{scratch_elements});
}
