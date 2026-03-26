/**
 * Mixed AIC+AIV Orchestration Function (tensormap_and_ringbuffer Runtime)
 *
 * Covers all 5 resource shapes per iteration:
 *   1. AIC_AIV_X2: AIC matmul(A,B->C) + AIV0 add(D,E->F) + AIV1 mul(G,H->I)
 *   2. AIC_ONLY:   matmul(A,B->J)
 *   3. AIV_X1:     add(D,E->K)
 *   4. AIV_X2:     AIV0 add(D,E->L) + AIV1 mul(G,H->M)
 *   5. AIC_AIV_X1: AIC matmul(A,B->N) + AIV0 add(D,E->O)
 *
 * Args layout (15 args):
 *   [A, B, C, D, E, F, G, H, I, J, K, L, M, N, O]
 *   Shape/dtype/size in TaskArg metadata.
 */

#include <stddef.h>
#include <stdint.h>

#include "pto_orchestration_api.h"

// Mixed-task kernels (args offset matches param position in mixed param list)
#define FUNC_MATMUL         0   // AIC: reads args[0..2]
#define FUNC_ADD            1   // AIV0 in mixed: reads args[3..5]
#define FUNC_MUL            2   // AIV1 in mixed: reads args[6..8]
// Standalone kernels (read args[0..2] or args[3..5])
#define FUNC_ADD_STANDALONE 3   // AIV: reads args[0..2]
#define FUNC_MUL_STANDALONE 4   // AIV1 in AIV_X2: reads args[3..5]

static constexpr uint32_t TILE_ELEMS = 128 * 128;

extern "C" {

__attribute__((visibility("default")))
PTO2OrchestrationConfig aicpu_orchestration_config(TaskArg* orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 15,
    };
}

__attribute__((visibility("default")))
void aicpu_orchestration_entry(TaskArg* orch_args, int orch_thread_num, int orch_thread_index) {
    (void)orch_thread_num;
    (void)orch_thread_index;

    // Input tensors use from_task_arg() — golden shape = kernel shape
    Tensor ext_A = from_task_arg(orch_args[0]);
    Tensor ext_B = from_task_arg(orch_args[1]);
    Tensor ext_D = from_task_arg(orch_args[3]);
    Tensor ext_E = from_task_arg(orch_args[4]);
    Tensor ext_G = from_task_arg(orch_args[6]);
    Tensor ext_H = from_task_arg(orch_args[7]);

    // Output tensors — full buffers
    Tensor ext_C = from_task_arg(orch_args[2]);
    Tensor ext_F = from_task_arg(orch_args[5]);
    Tensor ext_I = from_task_arg(orch_args[8]);
    Tensor ext_J = from_task_arg(orch_args[9]);
    Tensor ext_K = from_task_arg(orch_args[10]);
    Tensor ext_L = from_task_arg(orch_args[11]);
    Tensor ext_M = from_task_arg(orch_args[12]);
    Tensor ext_N = from_task_arg(orch_args[13]);
    Tensor ext_O = from_task_arg(orch_args[14]);

    // Derive num_iters from output tensor size
    uint32_t total_elems = orch_args[2].tensor.shapes[0];
    int num_iters = (int)(total_elems / TILE_ELEMS);

    LOG_INFO("[mixed_orch] num_iters=%d", num_iters);

    for (int i = 0; i < num_iters; i++) {
        PTO2_SCOPE() {
            uint32_t view_shapes[1] = {TILE_ELEMS};
            uint32_t view_offsets[1] = {(uint32_t)i * TILE_ELEMS};

            Tensor C_view = ext_C.view(view_shapes, view_offsets);
            Tensor F_view = ext_F.view(view_shapes, view_offsets);
            Tensor I_view = ext_I.view(view_shapes, view_offsets);
            Tensor J_view = ext_J.view(view_shapes, view_offsets);
            Tensor K_view = ext_K.view(view_shapes, view_offsets);
            Tensor L_view = ext_L.view(view_shapes, view_offsets);
            Tensor M_view = ext_M.view(view_shapes, view_offsets);
            Tensor N_view = ext_N.view(view_shapes, view_offsets);
            Tensor O_view = ext_O.view(view_shapes, view_offsets);

            // 1. AIC_AIV_X2: matmul + add + mul
            {
                MixedKernels mk;
                mk.aic_kernel_id = FUNC_MATMUL;
                mk.aiv0_kernel_id = FUNC_ADD;
                mk.aiv1_kernel_id = FUNC_MUL;
                PTOParam params;
                params.add_input(ext_A);
                params.add_input(ext_B);
                params.add_output(C_view);
                params.add_input(ext_D);
                params.add_input(ext_E);
                params.add_output(F_view);
                params.add_input(ext_G);
                params.add_input(ext_H);
                params.add_output(I_view);
                pto2_rt_submit_task(mk, params);
            }

            // 2. AIC_ONLY: standalone matmul
            {
                PTOParam params;
                params.add_input(ext_A);
                params.add_input(ext_B);
                params.add_output(J_view);
                pto2_rt_submit_aic_task(FUNC_MATMUL, params);
            }

            // 3. AIV_X1: standalone add
            {
                PTOParam params;
                params.add_input(ext_D);
                params.add_input(ext_E);
                params.add_output(K_view);
                pto2_rt_submit_aiv_task(FUNC_ADD_STANDALONE, params);
            }

            // 4. AIV_X2: add (AIV0) + mul (AIV1)
            {
                MixedKernels mk;
                mk.aiv0_kernel_id = FUNC_ADD_STANDALONE;
                mk.aiv1_kernel_id = FUNC_MUL_STANDALONE;
                PTOParam params;
                params.add_input(ext_D);
                params.add_input(ext_E);
                params.add_output(L_view);
                params.add_input(ext_G);
                params.add_input(ext_H);
                params.add_output(M_view);
                pto2_rt_submit_task(mk, params);
            }

            // 5. AIC_AIV_X1: matmul (AIC) + add (AIV0)
            {
                MixedKernels mk;
                mk.aic_kernel_id = FUNC_MATMUL;
                mk.aiv0_kernel_id = FUNC_ADD;
                PTOParam params;
                params.add_input(ext_A);
                params.add_input(ext_B);
                params.add_output(N_view);
                params.add_input(ext_D);
                params.add_input(ext_E);
                params.add_output(O_view);
                pto2_rt_submit_task(mk, params);
            }
        }
    }

    LOG_INFO("[mixed_orch] Submitted %d iterations x 5 shapes = %d tasks", num_iters, num_iters * 5);
}

}  // extern "C"
