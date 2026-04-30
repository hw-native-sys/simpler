// Orchestration Function: MoE with Inter-Chip Communication
//
// This orchestration implements the three-stage distributed MoE pattern:
//   Stage 1: Dispatch all-to-all - each card sends its expert data to expert owner
//   Stage 2: Compute - each expert processes its received data
//   Stage 3: Combine all-to-all - results are sent back to source cards
//
// Data flow matches golden.py:
//   send[card_j][expert_i][:][:] → recv[expert_i][card_j][:][:] (dispatch)
//   recv[expert_i][card_j][:][:] += expert_i (compute)
//   recv[expert_i][card_j][:][:] → output[card_j][:][:] (combine)

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
        .expected_arg_count = 4,  // send, recv, output, scratch
    };
}

__attribute__((visibility("default")))
void aicpu_orchestration_entry(const ChipStorageTaskArgs& orch_args) {
    // External tensors
    Tensor ext_send = from_tensor_arg(orch_args.tensor(0));      // [num_experts][tokens][hidden]
    Tensor ext_recv = from_tensor_arg(orch_args.tensor(1));      // [num_cards][tokens][hidden]
    Tensor ext_output = from_tensor_arg(orch_args.tensor(2));    // [tokens][hidden]
    Tensor ext_scratch = from_tensor_arg(orch_args.tensor(3));   // HCCL scratch buffer

    // Scalar arguments
    int64_t expert_id = static_cast<int64_t>(orch_args.scalar(0));  // Which expert this card processes
    int64_t card_id = static_cast<int64_t>(orch_args.scalar(1));    // Which card this is
    int64_t num_cards = static_cast<int64_t>(orch_args.scalar(2));  // Total number of cards
    uint64_t comm_ctx_ptr = static_cast<uint64_t>(orch_args.scalar(3));  // CommContext*

    printf("[MoE Orch] orchestration_entry: card_id=%ld expert_id=%ld num_cards=%ld comm_ctx=0x%lx\n",
           card_id, expert_id, num_cards, comm_ctx_ptr);
    fflush(stdout);

    PTO2_SCOPE() {
        // === 阶段 1: Dispatch All-to-All ===
        // Each card i sends send[i][expert_i][:][:] to all cards
        // and receives send[j][expert_i][:][:] from card j
        // Result: recv[i][card_j][:][:] = send[card_j][expert_i][:][:]
        {
            printf("[MoE Orch] Stage 1: Dispatch - card_id=%ld submitting dispatch task\n", card_id);
            fflush(stdout);
            Arg params_dispatch;
            params_dispatch.add_input(ext_send);
            params_dispatch.add_output(ext_recv);
            params_dispatch.add_inout(ext_scratch);
            params_dispatch.add_scalar(expert_id);
            params_dispatch.add_scalar(num_cards);
            params_dispatch.add_scalar(comm_ctx_ptr);
            pto2_rt_submit_aiv_task(0, params_dispatch);  // moe_dispatch_alltoall
            printf("[MoE Orch] Stage 1: Dispatch - card_id=%ld dispatch task submitted\n", card_id);
            fflush(stdout);
        }

        printf("[MoE Orch] ===== After Dispatch (card_id=%ld, expert_id=%ld) =====\n", card_id, expert_id);
        fflush(stdout);

        // === 阶段 2: Compute (本地) ===
        // Add 1.0 to all elements in recv[:][:4][:]
        {
            printf("[MoE Orch] Stage 2: Compute - card_id=%ld\n", card_id);
            fflush(stdout);

            Arg params_compute;
            params_compute.add_inout(ext_recv);
            params_compute.add_scalar(0);  // unused
            params_compute.add_scalar(0);  // unused
            params_compute.add_scalar(0);  // unused
            pto2_rt_submit_aiv_task(1, params_compute);  // moe_simple_compute

            printf("[MoE Orch] Stage 2: Compute - card_id=%ld compute task submitted\n", card_id);
            fflush(stdout);
        }

        printf("[MoE Orch] ===== After Compute (card_id=%ld, expert_id=%ld) =====\n", card_id, expert_id);
        fflush(stdout);

        // === 阶段 3: Combine All-to-All ===
        // Each card i sends recv[i][card_j][:][:] to card j
        // Card j accumulates all received data to output[j][:][:]
        {
            printf("[MoE Orch] Stage 3: Combine - card_id=%ld submitting combine task\n", card_id);
            fflush(stdout);
            Arg params_combine;
            params_combine.add_input(ext_recv);
            params_combine.add_output(ext_output);
            params_combine.add_inout(ext_scratch);
            params_combine.add_scalar(card_id);
            params_combine.add_scalar(num_cards);
            params_combine.add_scalar(comm_ctx_ptr);
            pto2_rt_submit_aiv_task(2, params_combine);  // moe_combine_alltoall
            printf("[MoE Orch] Stage 3: Combine - card_id=%ld combine task submitted\n", card_id);
            fflush(stdout);
        }

        printf("[MoE Orch] ===== After Combine (card_id=%ld) =====\n", card_id);
        fflush(stdout);
    }

    printf("[MoE Orch] orchestration_entry: card_id=%ld completed\n", card_id);
    fflush(stdout);
}

}  // extern "C"
