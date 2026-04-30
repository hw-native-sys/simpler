// Orchestration Function: End-to-End MoE Pipeline
//
// This orchestration runs the complete MoE pipeline:
// 1. Dispatch: distribute tokens to expert cards
// 2. Compute: process tokens on each expert card
// 3. Combine: gather results back to source cards
//
// Uses independent scratch buffers for combine phase to avoid data corruption

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
        .expected_arg_count = 10,  // send, recv, output, scratch, scratch_test, scratch_print, expert_id, card_id, num_cards, commCtx
    };
}

__attribute__((visibility("default")))
void aicpu_orchestration_entry(const ChipStorageTaskArgs& orch_args) {
    // External tensors
    Tensor ext_send = from_tensor_arg(orch_args.tensor(0));         // [num_experts][tokens][hidden]
    Tensor ext_recv = from_tensor_arg(orch_args.tensor(1));         // [num_cards][tokens][hidden]
    Tensor ext_output = from_tensor_arg(orch_args.tensor(2));       // [num_cards][count][hidden]
    Tensor ext_scratch = from_tensor_arg(orch_args.tensor(3));      // HCCL scratch buffer for dispatch+compute
    Tensor ext_scratch_test = from_tensor_arg(orch_args.tensor(4)); // HCCL scratch buffer for combine phase
    Tensor ext_scratch_print = from_tensor_arg(orch_args.tensor(5));  // Scratch print buffer

    // Scalar arguments
    int64_t expert_id = static_cast<int64_t>(orch_args.scalar(0));  // Which expert this card processes
    int64_t card_id = static_cast<int64_t>(orch_args.scalar(1));    // Which card this is
    int64_t num_cards = static_cast<int64_t>(orch_args.scalar(2));  // Total number of cards
    uint64_t comm_ctx_ptr = static_cast<uint64_t>(orch_args.scalar(3));  // CommContext*

    printf("[End2End Orch] card_id=%ld expert_id=%ld num_cards=%ld\n",
           card_id, expert_id, num_cards);
    fflush(stdout);

    PTO2_SCOPE() {
        // ========== PART 1: Full Pipeline ==========
        printf("[End2End Orch] Part 1: Full Pipeline (Dispatch + Compute + Combine) - card_id=%ld\n", card_id);
        fflush(stdout);

        // === Phase 1: Dispatch ===
        printf("[End2End Orch] Phase 1: Dispatch - card_id=%ld\n", card_id);
        fflush(stdout);

        Arg params_dispatch;
        params_dispatch.add_input(ext_send);
        params_dispatch.add_output(ext_recv);
        params_dispatch.add_inout(ext_scratch);
        params_dispatch.add_scalar(expert_id);
        params_dispatch.add_scalar(num_cards);
        params_dispatch.add_scalar(comm_ctx_ptr);
        pto2_rt_submit_aiv_task(0, params_dispatch);  // moe_dispatch_alltoall

        printf("[End2End Orch] Dispatch submitted\n", card_id);
        fflush(stdout);

        // === Phase 2: Compute ===
        printf("[End2End Orch] Phase 2: Compute - card_id=%ld\n", card_id);
        fflush(stdout);

        Arg params_compute;
        params_compute.add_inout(ext_recv);
        params_compute.add_scalar(0);  // unused
        params_compute.add_scalar(0);  // unused
        params_compute.add_scalar(0);  // unused
        pto2_rt_submit_aiv_task(1, params_compute);  // moe_simple_compute

        printf("[End2End Orch] Compute submitted\n", card_id);
        fflush(stdout);

        // === Phase 3: Combine (Full Pipeline) ===
        printf("[End2End Orch] Phase 3: Combine (full pipeline) - card_id=%ld\n", card_id);
        fflush(stdout);

        Arg params_combine;
        params_combine.add_input(ext_recv);
        params_combine.add_output(ext_output);
        params_combine.add_inout(ext_scratch_test);  // Use independent scratch_test buffer for combine
        params_combine.add_output(ext_scratch_print);
        params_combine.add_scalar(card_id);
        params_combine.add_scalar(num_cards);
        params_combine.add_scalar(comm_ctx_ptr);
        pto2_rt_submit_aiv_task(2, params_combine);  // moe_combine_alltoall

        printf("[End2End Orch] Combine (full pipeline) submitted\n", card_id);
        fflush(stdout);
    }

    printf("[End2End Orch] card_id=%ld completed\n", card_id);
    fflush(stdout);
}

}  // extern "C"
