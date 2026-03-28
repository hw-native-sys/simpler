/**
 * AICPU orchestration for the vector example.
 *
 * DAG structure for formula: f = (a + b + 1) * (a + b + 2)
 *   t0: c = a + b     (func_id=0, kernel_add)
 *   t1: d = c + 1     (func_id=1, kernel_add_scalar)
 *   t2: e = c + 2     (func_id=1, kernel_add_scalar)
 *   t3: f = d * e     (func_id=2, kernel_mul)
 *   Dependencies: t0->t1, t0->t2, t1->t3, t2->t3
 *
 * Uses explicit add_dependency for all dependency edges (no TensorMap).
 * Tasks are batch-published at scope_end.
 */

#include <stddef.h>
#include <stdint.h>

#include "pto_orchestration_api.h"

extern "C" {

__attribute__((visibility("default")))
PTO2OrchestrationConfig aicpu_orchestration_config(TaskArg* orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 3,
    };
}

__attribute__((visibility("default")))
void aicpu_orchestration_entry(PTO2Runtime* rt, TaskArg* orch_args, int orch_thread_num, int orch_thread_index) {
    (void)orch_thread_num;
    (void)orch_thread_index;

    // golden shape = kernel shape, use from_task_arg() directly
    Tensor ext_a = from_task_arg(orch_args[0]);
    Tensor ext_b = from_task_arg(orch_args[1]);
    Tensor ext_f = from_task_arg(orch_args[2]);

    uint32_t SIZE = orch_args[0].tensor.shapes[0];

    uint32_t shapes[1] = {SIZE};

    // Intermediate tensors — allocated from HeapRing by the runtime
    Tensor c = make_tensor(shapes, 1, DataType::FLOAT32);
    Tensor d = make_tensor(shapes, 1, DataType::FLOAT32);
    Tensor e = make_tensor(shapes, 1, DataType::FLOAT32);

    PTO2_SCOPE(rt) {
        // t0: c = a + b
        Arg args_t0;
        args_t0.add_input(ext_a);
        args_t0.add_input(ext_b);
        args_t0.add_output(c);
        PTO2TaskId t0 = pto2_rt_submit_aiv_task(rt, 0, args_t0);

        // t1: d = c + 1.0
        Arg args_t1;
        args_t1.add_input(c);
        args_t1.add_output(d);
        args_t1.add_scalar(to_u64(1.0f));
        PTO2TaskId t1 = pto2_rt_submit_aiv_task(rt, 1, args_t1);
        pto2_rt_add_dependency(rt, t0, t1);

        // t2: e = c + 2.0
        Arg args_t2;
        args_t2.add_input(c);
        args_t2.add_output(e);
        args_t2.add_scalar(to_u64(2.0f));
        PTO2TaskId t2 = pto2_rt_submit_aiv_task(rt, 1, args_t2);
        pto2_rt_add_dependency(rt, t0, t2);

        // t3: f = d * e
        Arg args_t3;
        args_t3.add_input(d);
        args_t3.add_input(e);
        args_t3.add_output(ext_f);
        PTO2TaskId t3 = pto2_rt_submit_aiv_task(rt, 2, args_t3);
        pto2_rt_add_dependency(rt, t1, t3);
        pto2_rt_add_dependency(rt, t2, t3);
    }  // scope_end: batch-publish all tasks
}

}  // extern "C"
