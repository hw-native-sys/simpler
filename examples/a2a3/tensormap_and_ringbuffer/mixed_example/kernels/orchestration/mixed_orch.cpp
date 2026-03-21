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
 * Args layout (30 args):
 *   [ptr_A, ptr_B, ptr_C, ptr_D, ptr_E, ptr_F,
 *    ptr_G, ptr_H, ptr_I, ptr_J, ptr_K, ptr_L,
 *    ptr_M, ptr_N, ptr_O,
 *    size_A, size_B, size_C, size_D, size_E, size_F,
 *    size_G, size_H, size_I, size_J, size_K, size_L,
 *    size_M, size_N, size_O]
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

#define ARG_PTR_A   0
#define ARG_PTR_B   1
#define ARG_PTR_C   2
#define ARG_PTR_D   3
#define ARG_PTR_E   4
#define ARG_PTR_F   5
#define ARG_PTR_G   6
#define ARG_PTR_H   7
#define ARG_PTR_I   8
#define ARG_PTR_J   9
#define ARG_PTR_K   10
#define ARG_PTR_L   11
#define ARG_PTR_M   12
#define ARG_PTR_N   13
#define ARG_PTR_O   14
#define ARG_SIZE_A  15
#define ARG_SIZE_B  16
#define ARG_SIZE_C  17
#define ARG_SIZE_D  18
#define ARG_SIZE_E  19
#define ARG_SIZE_F  20
#define ARG_SIZE_G  21
#define ARG_SIZE_H  22
#define ARG_SIZE_I  23
#define ARG_SIZE_J  24
#define ARG_SIZE_K  25
#define ARG_SIZE_L  26
#define ARG_SIZE_M  27
#define ARG_SIZE_N  28
#define ARG_SIZE_O  29

static constexpr uint32_t TILE_ELEMS = 128 * 128;

extern "C" {

__attribute__((visibility("default")))
PTO2OrchestrationConfig aicpu_orchestration_config(uint64_t* args, int arg_count) {
    (void)args;
    (void)arg_count;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 30,
    };
}

__attribute__((visibility("default")))
void aicpu_orchestration_entry(uint64_t* args, int arg_count, int orch_thread_num, int orch_thread_index) {
    (void)arg_count;
    (void)orch_thread_num;
    (void)orch_thread_index;

    void* dev_A = (void*)(uintptr_t)args[ARG_PTR_A];
    void* dev_B = (void*)(uintptr_t)args[ARG_PTR_B];
    void* dev_C = (void*)(uintptr_t)args[ARG_PTR_C];
    void* dev_D = (void*)(uintptr_t)args[ARG_PTR_D];
    void* dev_E = (void*)(uintptr_t)args[ARG_PTR_E];
    void* dev_F = (void*)(uintptr_t)args[ARG_PTR_F];
    void* dev_G = (void*)(uintptr_t)args[ARG_PTR_G];
    void* dev_H = (void*)(uintptr_t)args[ARG_PTR_H];
    void* dev_I = (void*)(uintptr_t)args[ARG_PTR_I];
    void* dev_J = (void*)(uintptr_t)args[ARG_PTR_J];
    void* dev_K = (void*)(uintptr_t)args[ARG_PTR_K];
    void* dev_L = (void*)(uintptr_t)args[ARG_PTR_L];
    void* dev_M = (void*)(uintptr_t)args[ARG_PTR_M];
    void* dev_N = (void*)(uintptr_t)args[ARG_PTR_N];
    void* dev_O = (void*)(uintptr_t)args[ARG_PTR_O];
    size_t size_C = (size_t)args[ARG_SIZE_C];

    int num_iters = (int)(size_C / (TILE_ELEMS * sizeof(float)));

    LOG_INFO("[mixed_orch] num_iters=%d", num_iters);

    // Input tensors (shared across all tasks)
    uint32_t ab_shapes[1] = {TILE_ELEMS};
    Tensor ext_A = make_tensor_external(dev_A, ab_shapes, 1, DataType::FLOAT32);
    Tensor ext_B = make_tensor_external(dev_B, ab_shapes, 1, DataType::FLOAT32);

    uint32_t de_shapes[1] = {TILE_ELEMS};
    Tensor ext_D = make_tensor_external(dev_D, de_shapes, 1, DataType::FLOAT32);
    Tensor ext_E = make_tensor_external(dev_E, de_shapes, 1, DataType::FLOAT32);

    uint32_t gh_shapes[1] = {TILE_ELEMS};
    Tensor ext_G = make_tensor_external(dev_G, gh_shapes, 1, DataType::FLOAT32);
    Tensor ext_H = make_tensor_external(dev_H, gh_shapes, 1, DataType::FLOAT32);

    // Output tensors (full buffers, one slice per iteration)
    uint32_t out_shapes[1] = {(uint32_t)num_iters * TILE_ELEMS};
    Tensor ext_C = make_tensor_external(dev_C, out_shapes, 1, DataType::FLOAT32);
    Tensor ext_F = make_tensor_external(dev_F, out_shapes, 1, DataType::FLOAT32);
    Tensor ext_I = make_tensor_external(dev_I, out_shapes, 1, DataType::FLOAT32);
    Tensor ext_J = make_tensor_external(dev_J, out_shapes, 1, DataType::FLOAT32);
    Tensor ext_K = make_tensor_external(dev_K, out_shapes, 1, DataType::FLOAT32);
    Tensor ext_L = make_tensor_external(dev_L, out_shapes, 1, DataType::FLOAT32);
    Tensor ext_M = make_tensor_external(dev_M, out_shapes, 1, DataType::FLOAT32);
    Tensor ext_N = make_tensor_external(dev_N, out_shapes, 1, DataType::FLOAT32);
    Tensor ext_O = make_tensor_external(dev_O, out_shapes, 1, DataType::FLOAT32);

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
