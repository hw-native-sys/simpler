/**
 * Alternating Matmul-Add Orchestration Function (tensormap_and_ringbuffer Runtime)
 *
 * Submits independent matmul and add tasks per batch.
 *
 * Configuration read from scalar TaskArgs:
 *   - batch: Number of batches
 *   - M: Number of matmul tasks per batch
 *   - N: Number of add tasks per batch
 *   - matmul_batch: Number of matmul tiles per task group
 *   - add_batch: Number of add tiles per task group
 *
 * Task pattern: interleaved [matmul_0, add_0, matmul_1, add_1, ...]
 * All tasks are completely independent (no dependencies).
 *
 * Arg layout: [A, B, C, X, Y, Z, batch, M_val, N_val, matmul_batch, add_batch]
 */

#include <stddef.h>
#include <stdint.h>

#include "pto_orchestration_api.h"

#define FUNC_MATMUL 0
#define FUNC_ADD    1

static constexpr uint64_t MATMUL_ELEMS = 128 * 128;
static constexpr uint64_t ADD_ELEMS = 128 * 128;

extern "C" {

__attribute__((visibility("default")))
PTO2OrchestrationConfig aicpu_orchestration_config(TaskArg* orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 11,
    };
}

__attribute__((visibility("default")))
void aicpu_orchestration_entry(TaskArg* orch_args, int orch_thread_num, int orch_thread_index) {
    // Tensor args
    Tensor ext_A = from_task_arg(orch_args[0]);
    Tensor ext_B = from_task_arg(orch_args[1]);
    Tensor ext_C = from_task_arg(orch_args[2]);
    Tensor ext_X = from_task_arg(orch_args[3]);
    Tensor ext_Y = from_task_arg(orch_args[4]);
    Tensor ext_Z = from_task_arg(orch_args[5]);

    // Scalar config args
    int batch = (int)orch_args[6].scalar;
    int M = (int)orch_args[7].scalar;
    int N = (int)orch_args[8].scalar;
    int matmul_batch = (int)orch_args[9].scalar;
    int add_batch = (int)orch_args[10].scalar;

    LOG_ALWAYS("[alternating_orch] thread num: %d, idx: %d", orch_thread_num, orch_thread_index);
    LOG_INFO("[alternating_orch] Batch: %d, M: %d, N: %d, matmul_batch: %d, add_batch: %d",
             batch, M, N, matmul_batch, add_batch);

    int total_matmul_tasks = batch * M;
    int total_add_tasks = batch * N;
    int num_matmul_groups = total_matmul_tasks / matmul_batch;
    int num_add_groups = total_add_tasks / add_batch;

    int total_matmul = 0;
    int total_add = 0;

    int max_groups = num_matmul_groups > num_add_groups ? num_matmul_groups : num_add_groups;

    // Interleaved submit: matmul and add groups alternate
    for (int group_idx = orch_thread_index; group_idx < max_groups; group_idx += orch_thread_num) {
        if (group_idx < num_matmul_groups) {
            int start_task_idx = group_idx * matmul_batch;
            uint64_t offset = (uint64_t)start_task_idx * MATMUL_ELEMS;
            uint64_t group_size = (uint64_t)matmul_batch * MATMUL_ELEMS;

            uint32_t matmul_group_shapes[1] = {(uint32_t)group_size};
            uint32_t view_offsets[1] = {(uint32_t)offset};

            Tensor A_view = ext_A.view(matmul_group_shapes, view_offsets);
            Tensor B_view = ext_B.view(matmul_group_shapes, view_offsets);
            Tensor C_view = ext_C.view(matmul_group_shapes, view_offsets);

            Arg params_matmul;
            params_matmul.add_input(A_view);
            params_matmul.add_input(B_view);
            params_matmul.add_inout(C_view);
            pto2_rt_submit_aic_task(FUNC_MATMUL, params_matmul);
            total_matmul++;
        }

        if (group_idx < num_add_groups) {
            int start_task_idx = group_idx * add_batch;
            uint64_t offset = (uint64_t)start_task_idx * ADD_ELEMS;
            uint64_t group_size = (uint64_t)add_batch * ADD_ELEMS;

            uint32_t add_group_shapes[1] = {(uint32_t)group_size};
            uint32_t view_offsets[1] = {(uint32_t)offset};

            Tensor X_view = ext_X.view(add_group_shapes, view_offsets);
            Tensor Y_view = ext_Y.view(add_group_shapes, view_offsets);
            Tensor Z_view = ext_Z.view(add_group_shapes, view_offsets);

            Arg params_add;
            params_add.add_input(X_view);
            params_add.add_input(Y_view);
            params_add.add_inout(Z_view);
            pto2_rt_submit_aiv_task(FUNC_ADD, params_add);
            total_add++;
        }
    }

    LOG_ALWAYS("[alternating_orch] Submitted %d matmul groups and %d add groups",
             total_matmul, total_add);
}

}  // extern "C"
