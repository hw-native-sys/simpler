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
 * Sliding-window moving average (tensormap_and_ringbuffer Runtime)
 *
 * out[i] = mean(base[i], out[i-1], ..., out[i-WINDOW])
 *
 * Every out[i] is a runtime-allocated OUTPUT, so reading it in a later task
 * emits a Step-A creator edge (DEP_WAIT | DEP_RETAIN) keyed on owner_task_id.
 * No explicit dependency is ever added: the whole graph comes from dependency
 * discovery.
 *
 * Task i therefore carries WINDOW producer edges, of which only i-1 is
 * non-transitive — T_{i-1} already reaches T_{i-2} .. T_{i-WINDOW}. The other
 * WINDOW-1 are transitively redundant and all sit at submission distance
 * <= WINDOW, so a bounded WAIT-reachability window of 64 covers every one of
 * them.
 *
 * Predecessors that do not exist yet (i < WINDOW) are padded with base[i], an
 * external tensor with no owner, which contributes no edge.
 *
 * A final emit task copies the last window output into the caller's `result`
 * buffer so the recurrence is checkable end to end.
 *
 * Arg layout: [base, result, steps]
 */

#include <stddef.h>
#include <stdint.h>

#include "orchestration_api.h"  // NOLINT(build/include_subdir)

#define FUNC_WINDOW 0
#define FUNC_EMIT 1

// Must equal the input count kernel_window_avg.cpp reads minus one (base), and
// the WINDOW in the scene test's golden.
static constexpr int WINDOW = 16;

extern "C" {

__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &orch_args) {
    (void)orch_args;  // NOLINT(readability/casting)
    return OrchestrationConfig{
        .expected_arg_count = 3,
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &orch_args) {
    const simpler::tmr::Tensor &base = orch_args.tensor(0).ref();
    const simpler::tmr::Tensor &result = orch_args.tensor(1).ref();

    int steps = orch_args.scalar<int>(0);
    if (steps <= 0) {
        return;
    }

    // One task's tile count, taken from the caller's buffers: `result` holds
    // exactly one step's worth of elements.
    uint64_t step_elems = result.shapes[0];

    LOG_INFO("[sliding_window_orch] steps=%d window=%d step_elems=%lu", steps, WINDOW, step_elems);

    // Last WINDOW runtime outputs, indexed by step modulo WINDOW. Slot i % WINDOW
    // is read (as step i - WINDOW) before it is overwritten by step i.
    simpler::tmr::Tensor ring[WINDOW];

    uint32_t step_shape[1] = {static_cast<uint32_t>(step_elems)};

    for (int i = 0; i < steps; i++) {
        uint32_t view_offset[1] = {static_cast<uint32_t>(static_cast<uint64_t>(i) * step_elems)};
        simpler::tmr::Tensor base_i = base.view(step_shape, view_offset);

        CoreTaskArgs params;
        params.add_input(base_i);
        for (int k = 1; k <= WINDOW; k++) {
            int j = i - k;
            if (j >= 0) {
                params.add_input(ring[j % WINDOW]);
            } else {
                params.add_input(base_i);
            }
        }

        TensorCreateInfo out_ci(step_shape, 1, DataType::FLOAT32);
        params.add_output(out_ci);

        TaskOutputTensors outs = rt_submit_aiv_task(FUNC_WINDOW, params);
        // A latched orchestrator fatal makes every later submit return no
        // outputs, and get_ref asserts on an out-of-range index. Leaving the
        // ring untouched and returning also keeps the emit task from consuming
        // a slot that was never produced.
        if (outs.empty()) {
            return;
        }
        ring[i % WINDOW] = outs.get_ref(0);
    }

    CoreTaskArgs emit_params;
    emit_params.add_input(ring[(steps - 1) % WINDOW]);
    emit_params.add_output(result);
    rt_submit_aiv_task(FUNC_EMIT, emit_params);

    LOG_INFO("[sliding_window_orch] submitted %d window tasks + 1 emit", steps);
}

}  // extern "C"
