// Orchestration Function: Dispatch Only (for debugging)
//
// This orchestration ONLY runs the dispatch phase to verify it works correctly.

#include "runtime.h"
#include <iostream>

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "pto_orchestration_api.h"

// Must match golden.py and kernel configurations
static constexpr int64_t COUNT = 4;  // Number of tokens to process per (card, expert) pair
static constexpr int64_t NUM_TOKENS = 10;  // Total number of tokens
static constexpr int64_t HIDDEN_DIM = 16;  // Hidden dimension

extern "C" {

__attribute__((visibility("default")))
PTO2OrchestrationConfig aicpu_orchestration_config(const ChipStorageTaskArgs& orch_args) {
    return PTO2OrchestrationConfig{
        .expected_arg_count = 4,  // send, recv, output, scratch (output unused in dispatch-only)
    };
}

__attribute__((visibility("default")))
void aicpu_orchestration_entry(const ChipStorageTaskArgs& orch_args) {
    // External tensors
    Tensor ext_send = from_tensor_arg(orch_args.tensor(0));      // [num_experts][tokens][hidden]
    Tensor ext_recv = from_tensor_arg(orch_args.tensor(1));      // [num_cards][tokens][hidden]
    Tensor ext_output = from_tensor_arg(orch_args.tensor(2));    // [tokens][hidden] (unused)
    Tensor ext_scratch = from_tensor_arg(orch_args.tensor(3));   // HCCL scratch buffer

    // Scalar arguments
    int64_t expert_id = static_cast<int64_t>(orch_args.scalar(0));  // Which expert this card processes
    int64_t card_id = static_cast<int64_t>(orch_args.scalar(1));    // Which card this is
    int64_t num_cards = static_cast<int64_t>(orch_args.scalar(2));  // Total number of cards
    uint64_t comm_ctx_ptr = static_cast<uint64_t>(orch_args.scalar(3));  // CommContext*

    printf("[Dispatch-Only Orch] card_id=%ld expert_id=%ld num_cards=%ld\n",
           card_id, expert_id, num_cards);
    fflush(stdout);

    PTO2_SCOPE() {
        // === ONLY Dispatch Phase ===
        printf("[Dispatch-Only Orch] Submitting dispatch task for card_id=%ld expert_id=%ld\n",
               card_id, expert_id);
        fflush(stdout);

        Arg params_dispatch;
        params_dispatch.add_input(ext_send);
        params_dispatch.add_output(ext_recv);
        params_dispatch.add_inout(ext_scratch);
        params_dispatch.add_scalar(expert_id);
        params_dispatch.add_scalar(num_cards);
        params_dispatch.add_scalar(comm_ctx_ptr);
        pto2_rt_submit_aiv_task(0, params_dispatch);  // moe_dispatch_alltoall

        printf("[Dispatch-Only Orch] Dispatch task submitted for card_id=%ld\n", card_id);
        fflush(stdout);
    }

    printf("[Dispatch-Only Orch] card_id=%ld completed\n", card_id);
    fflush(stdout);
}

}  // extern "C"
