/**
 * Alternating Matmul-Add Orchestration Function (tensormap_and_ringbuffer Runtime)
 *
 * Submits independent matmul and add tasks in fixed alternating order.
 *
 * Configuration (from config tensor):
 *   - batch: Number of batches
 *   - M: Number of matmul tasks per batch
 *   - N: Number of add tasks per batch
 *
 * Task pattern: [matmul_0, add_0, matmul_1, add_1, ...]
 * All tasks are completely independent (no dependencies).
 *
 * Args layout: [ptr_A, ptr_B, ptr_C, ptr_X, ptr_Y, ptr_Z,
 *               size_A, size_B, size_C, size_X, size_Y, size_Z, ptr_config]
 */

#include <stddef.h>
#include <stdint.h>

#include "pto_orchestration_api.h"

#define FUNC_MATMUL 0
#define FUNC_ADD    1

#define ARG_PTR_A      0
#define ARG_PTR_B      1
#define ARG_PTR_C      2
#define ARG_PTR_X      3
#define ARG_PTR_Y      4
#define ARG_PTR_Z      5
#define ARG_SIZE_A     6
#define ARG_SIZE_B     7
#define ARG_SIZE_C     8
#define ARG_SIZE_X     9
#define ARG_SIZE_Y     10
#define ARG_SIZE_Z     11
#define ARG_PTR_CONFIG 12

static constexpr uint64_t MATMUL_ELEMS = 128 * 128;
static constexpr uint64_t ADD_ELEMS = 64 * 128;

extern "C" {

__attribute__((visibility("default")))
PTO2OrchestrationConfig aicpu_orchestration_config(uint64_t* args, int arg_count) {
    (void)args;
    (void)arg_count;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 13,
    };
}

__attribute__((visibility("default")))
void aicpu_orchestration_entry(PTO2Runtime* rt, uint64_t* args, int arg_count) {
    (void)arg_count;
    pto2_rt_init_tensor_pool(rt);

    void* dev_A = (void*)(uintptr_t)args[ARG_PTR_A];
    void* dev_B = (void*)(uintptr_t)args[ARG_PTR_B];
    void* dev_C = (void*)(uintptr_t)args[ARG_PTR_C];
    void* dev_X = (void*)(uintptr_t)args[ARG_PTR_X];
    void* dev_Y = (void*)(uintptr_t)args[ARG_PTR_Y];
    void* dev_Z = (void*)(uintptr_t)args[ARG_PTR_Z];
    size_t size_A = (size_t)args[ARG_SIZE_A];
    size_t size_B = (size_t)args[ARG_SIZE_B];
    size_t size_C = (size_t)args[ARG_SIZE_C];
    size_t size_X = (size_t)args[ARG_SIZE_X];
    size_t size_Y = (size_t)args[ARG_SIZE_Y];
    size_t size_Z = (size_t)args[ARG_SIZE_Z];
    int64_t* config = (int64_t*)(uintptr_t)args[ARG_PTR_CONFIG];

    int batch = (int)config[0];
    int M = (int)config[1];
    int N = (int)config[2];

    LOG_INFO(rt, "[alternating_orch] Batch: %d, M: %d, N: %d", batch, M, N);

    uint64_t ext_A_shapes[1] = {size_A / sizeof(float)};
    Tensor ext_A = make_tensor_external(dev_A, ext_A_shapes, 1, DataType::FLOAT32);
    uint64_t ext_B_shapes[1] = {size_B / sizeof(float)};
    Tensor ext_B = make_tensor_external(dev_B, ext_B_shapes, 1, DataType::FLOAT32);
    uint64_t ext_C_shapes[1] = {size_C / sizeof(float)};
    Tensor ext_C = make_tensor_external(dev_C, ext_C_shapes, 1, DataType::FLOAT32);

    uint64_t ext_X_shapes[1] = {size_X / sizeof(float)};
    Tensor ext_X = make_tensor_external(dev_X, ext_X_shapes, 1, DataType::FLOAT32);
    uint64_t ext_Y_shapes[1] = {size_Y / sizeof(float)};
    Tensor ext_Y = make_tensor_external(dev_Y, ext_Y_shapes, 1, DataType::FLOAT32);
    uint64_t ext_Z_shapes[1] = {size_Z / sizeof(float)};
    Tensor ext_Z = make_tensor_external(dev_Z, ext_Z_shapes, 1, DataType::FLOAT32);

    uint64_t matmul_tile_shapes[1] = {MATMUL_ELEMS};
    uint64_t add_tile_shapes[1] = {ADD_ELEMS};

    int total_matmul = batch * M;
    int total_add = batch * N;
    int max_tasks = (total_matmul > total_add) ? total_matmul : total_add;

    for (int i = 0; i < max_tasks; i++) {
        if (i < total_matmul) {
            uint64_t offset = (uint64_t)i * MATMUL_ELEMS;


            uint64_t view_offsets[1] = {offset};

            Tensor A_view = ext_A.view(matmul_tile_shapes, view_offsets);
            Tensor B_view = ext_B.view(matmul_tile_shapes, view_offsets);
            Tensor C_view = ext_C.view(matmul_tile_shapes, view_offsets);

            PTOParam params_matmul[] = {
                make_input_param(A_view),
                make_input_param(B_view),
                make_output_param(C_view),
            };
            pto2_rt_submit_task(rt, FUNC_MATMUL, PTO2_WORKER_CUBE,
                               params_matmul, 3);
        }

        if (i < total_add) {
            uint64_t offset = (uint64_t)i * ADD_ELEMS;


            uint64_t view_offsets[1] = {offset};

            Tensor X_view = ext_X.view(add_tile_shapes, view_offsets);
            Tensor Y_view = ext_Y.view(add_tile_shapes, view_offsets);
            Tensor Z_view = ext_Z.view(add_tile_shapes, view_offsets);

            PTOParam params_add[] = {
                make_input_param(X_view),
                make_input_param(Y_view),
                make_output_param(Z_view),
            };
            pto2_rt_submit_task(rt, FUNC_ADD, PTO2_WORKER_VECTOR,
                               params_add, 3);
        }
    }

    LOG_INFO(rt, "[alternating_orch] Submitted %d matmul tasks and %d add tasks",
                  total_matmul, total_add);
}

}  // extern "C"
