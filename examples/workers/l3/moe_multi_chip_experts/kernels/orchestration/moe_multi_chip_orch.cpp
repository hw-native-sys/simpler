// Orchestration Function: moe_demo (Multi-Chip Version)
//
// Multi-chip MoE orchestration - implements "one expert per chip" parallelism.
//
// Architecture comparison:
// - Single-chip version: One chip runs ALL experts sequentially
//   (orchestration loops: card_i=0..3, expert_j=0..3, t_idx=0..3)
// - Multi-chip version: Each chip runs ONE expert in parallel
//   (orchestration: card_i passed as arg, expert_j passed as arg, t_idx=0..3)
//
// Key insight: Both versions produce IDENTICAL results because the kernels
// perform the same computation - only the execution distribution differs.
//
// Expected arguments:
// - 3 tensors: send (INPUT), recv (OUTPUT_EXISTING), output (OUTPUT_EXISTING)
// - 2 scalars: expert_id (which expert), chip_id (logical card_i for data layout)

#include "runtime.h"
#include <iostream>

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "pto_orchestration_api.h"

extern "C" {

__attribute__((visibility("default")))
PTO2OrchestrationConfig aicpu_orchestration_config(const ChipStorageTaskArgs& orch_args) {
    // Expected: 3 tensors + 2 scalars (expert_id, chip_id)
    return PTO2OrchestrationConfig{
        .expected_arg_count = 3,
    };
}

__attribute__((visibility("default")))
void aicpu_orchestration_entry(const ChipStorageTaskArgs& orch_args) {
    // External tensors
    Tensor ext_send = from_tensor_arg(orch_args.tensor(0));
    Tensor ext_recv = from_tensor_arg(orch_args.tensor(1));
    Tensor ext_output = from_tensor_arg(orch_args.tensor(2));

    // Read expert ID and chip ID from scalar arguments (passed by Python)
    int64_t expert_j = static_cast<int64_t>(orch_args.scalar(0));
    int64_t card_i = static_cast<int64_t>(orch_args.scalar(1));

    PTO2_SCOPE() {
        // Stage 0: Dispatch (send → recv)
        for (int64_t t_idx = 0; t_idx < 4; t_idx += 1) {
            PTO2_SCOPE() {
                Arg params_t0;
                params_t0.add_input(ext_send);
                params_t0.add_output(ext_recv);
                params_t0.add_scalar(card_i);
                params_t0.add_scalar(expert_j);
                params_t0.add_scalar(t_idx);
                pto2_rt_submit_aiv_task(0, params_t0);
            }
        }

        // Stage 1: Compute (expert transformation on recv)
        for (int64_t t_idx = 0; t_idx < 4; t_idx += 1) {
            PTO2_SCOPE() {
                Arg params_t1;
                params_t1.add_inout(ext_recv);
                params_t1.add_scalar(expert_j);
                params_t1.add_scalar(card_i);
                params_t1.add_scalar(t_idx);
                pto2_rt_submit_aiv_task(1, params_t1);
            }
        }

        // Stage 2: Combine (recv → output)
        for (int64_t t_idx = 0; t_idx < 4; t_idx += 1) {
            PTO2_SCOPE() {
                Arg params_t2;
                params_t2.add_input(ext_recv);
                params_t2.add_output(ext_output);
                params_t2.add_scalar(card_i);
                params_t2.add_scalar(t_idx);
                pto2_rt_submit_aiv_task(2, params_t2);
            }
        }
    }
}

}  // extern "C"
