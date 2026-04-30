// Orchestration Function: Combine Only (for debugging)
//
// This orchestration ONLY runs the combine phase to verify it works correctly.

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
        .expected_arg_count = 7,  // recv, output, scratch, scratch_print, card_id, num_cards, commCtx
    };
}

__attribute__((visibility("default")))
void aicpu_orchestration_entry(const ChipStorageTaskArgs& orch_args) {
    // External tensors
    Tensor ext_recv = from_tensor_arg(orch_args.tensor(0));      // [num_cards][tokens][hidden]
    Tensor ext_output = from_tensor_arg(orch_args.tensor(1));    // [num_cards][count][hidden]
    Tensor ext_scratch = from_tensor_arg(orch_args.tensor(2));   // HCCL scratch buffer
    Tensor ext_scratch_print = from_tensor_arg(orch_args.tensor(3));  // Scratch print buffer

    // Scalar arguments
    int64_t card_id = static_cast<int64_t>(orch_args.scalar(0));    // Which card this is
    int64_t num_cards = static_cast<int64_t>(orch_args.scalar(1));  // Total number of cards
    uint64_t comm_ctx_ptr = static_cast<uint64_t>(orch_args.scalar(2));  // CommContext*

    printf("[Combine-Only Orch] card_id=%ld num_cards=%ld\n",
           card_id, num_cards);
    fflush(stdout);

    PTO2_SCOPE() {
        // === ONLY Combine Phase ===
        printf("[Combine-Only Orch] Submitting combine task for card_id=%ld\n",
               card_id);
        fflush(stdout);

        Arg params_combine;
        params_combine.add_input(ext_recv);
        params_combine.add_output(ext_output);
        params_combine.add_inout(ext_scratch);
        params_combine.add_output(ext_scratch_print);
        params_combine.add_scalar(card_id);
        params_combine.add_scalar(num_cards);
        params_combine.add_scalar(comm_ctx_ptr);
        pto2_rt_submit_aiv_task(0, params_combine);  // moe_combine_alltoall

        printf("[Combine-Only Orch] Combine task submitted for card_id=%ld\n", card_id);
        fflush(stdout);
    }

    printf("[Combine-Only Orch] card_id=%ld completed\n", card_id);
    fflush(stdout);
}

}  // extern "C"
