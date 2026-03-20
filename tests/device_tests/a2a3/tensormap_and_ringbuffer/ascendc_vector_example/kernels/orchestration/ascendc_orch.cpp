/**
 * AscendC Vector Example — Device Orchestration
 *
 * Demonstrates calling an AscendC kernel (AddCustom, func_id=0) and a
 * PTO-native kernel (kernel_mul, func_id=1) from the same orchestration.
 *
 * DAG:
 *   t0: z = add_custom(x, y)   (AscendC, func_id=0)
 *   t1: w = mul(z, z)          (PTO,     func_id=1)
 *   Dependencies: t0 -> t1
 *
 * Both kernels run as AIV (vector) tasks on a single blockdim (2V).
 *
 * Args layout (from golden.py):
 *   [ptr_x, ptr_y, ptr_z, ptr_w, size_x, size_y, size_z, size_w, SIZE]
 */

#include <stddef.h>
#include <stdint.h>

#include "pto_orchestration_api.h"

#define FUNC_ADD_CUSTOM 0
#define FUNC_MUL        1

#define ARG_PTR_X  0
#define ARG_PTR_Y  1
#define ARG_PTR_Z  2
#define ARG_PTR_W  3
#define ARG_SIZE_X 4
#define ARG_SIZE_Y 5
#define ARG_SIZE_Z 6
#define ARG_SIZE_W 7
#define ARG_SIZE   8

extern "C" {

__attribute__((visibility("default")))
PTO2OrchestrationConfig aicpu_orchestration_config(uint64_t* args, int arg_count) {
    (void)args;
    (void)arg_count;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 9,
    };
}

__attribute__((visibility("default")))
void aicpu_orchestration_entry(PTO2Runtime* rt, uint64_t* args, int arg_count,
                               int orch_thread_num, int orch_thread_index) {
    (void)arg_count;
    (void)orch_thread_num;
    (void)orch_thread_index;

    void* ptr_x = (void*)(uintptr_t)args[ARG_PTR_X];
    void* ptr_y = (void*)(uintptr_t)args[ARG_PTR_Y];
    void* ptr_z = (void*)(uintptr_t)args[ARG_PTR_Z];
    void* ptr_w = (void*)(uintptr_t)args[ARG_PTR_W];
    int SIZE = (int)(args[ARG_SIZE] & 0x7FFFFFFF);

    LOG_INFO(rt, "[ascendc_orch] SIZE=%d", SIZE);

    uint32_t shapes[1] = {(uint32_t)SIZE};
    Tensor ext_x = make_tensor_external(ptr_x, shapes, 1, DataType::FLOAT32);
    Tensor ext_y = make_tensor_external(ptr_y, shapes, 1, DataType::FLOAT32);
    Tensor ext_z = make_tensor_external(ptr_z, shapes, 1, DataType::FLOAT32);
    Tensor ext_w = make_tensor_external(ptr_w, shapes, 1, DataType::FLOAT32);

    // t0: z = add_custom(x, y)  — AscendC kernel via PTO dispatch
    {
        PTOParam params_t0;
        params_t0.add_input(ext_x);
        params_t0.add_input(ext_y);
        params_t0.add_output(ext_z);
        pto2_rt_submit_aiv_task(rt, FUNC_ADD_CUSTOM, params_t0);
    }

    // t1: w = mul(z, z)  — PTO-native kernel
    {
        PTOParam params_t1;
        params_t1.add_input(ext_z);
        params_t1.add_input(ext_z);
        params_t1.add_output(ext_w);
        pto2_rt_submit_aiv_task(rt, FUNC_MUL, params_t1);
    }

    LOG_INFO(rt, "[ascendc_orch] Submitted 2 tasks");
}

}  // extern "C"
