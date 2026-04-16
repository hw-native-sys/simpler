/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */
/**
 * Dump-tensor interface demo for host_build_graph.
 *
 * Demonstrates the two ways to register tensor metadata for dump:
 *   Task 0 (add):                add_task() + set_tensor_info_to_task()
 *   Task 1 (add_scalar_inplace): add_task_with_tensor_info()
 *
 * Computation: f = (a + b) + 1  (a=2, b=3 → f=6)
 */

#include "orchestration_api.h"  // NOLINT(build/include_subdir)

extern "C" {

int build_dump_tensor_graph(OrchestrationRuntime *runtime, const ChipStorageTaskArgs &orch_args) {
    void *host_a = orch_args.tensor(0).data_as<void>();
    void *host_b = orch_args.tensor(1).data_as<void>();
    void *host_f = orch_args.tensor(2).data_as<void>();
    size_t size_a = orch_args.tensor(0).nbytes();
    size_t size_b = orch_args.tensor(1).nbytes();
    size_t size_f = orch_args.tensor(2).nbytes();
    uint32_t size = orch_args.tensor(0).shapes[0];

    TensorInfo ext_a_info = make_tensor_info_from_tensor_arg(orch_args.tensor(0));
    TensorInfo ext_b_info = make_tensor_info_from_tensor_arg(orch_args.tensor(1));
    TensorInfo ext_f_info = make_tensor_info_from_tensor_arg(orch_args.tensor(2));

    void *dev_a = device_malloc(runtime, size_a);
    copy_to_device(runtime, dev_a, host_a, size_a);

    void *dev_b = device_malloc(runtime, size_b);
    copy_to_device(runtime, dev_b, host_b, size_b);

    void *dev_f = device_malloc(runtime, size_f);
    record_tensor_pair(runtime, host_f, dev_f, size_f);

    // Task 0: a + b → f  (add_task + set_tensor_info_to_task)
    uint64_t args_t0[4] = {
        reinterpret_cast<uint64_t>(dev_a),
        reinterpret_cast<uint64_t>(dev_b),
        reinterpret_cast<uint64_t>(dev_f),
        size,
    };
    int t0 = add_task(runtime, args_t0, 4, 0, CoreType::AIV);
    TensorInfo t0_info[] = {ext_a_info, ext_b_info, ext_f_info};
    set_tensor_info_to_task(runtime, t0, t0_info, 3);

    // Task 1: f += 1.0  (add_task_with_tensor_info)
    union {
        float f32;
        uint64_t u64;
    } sc;
    sc.f32 = 1.0f;
    uint64_t args_t1[3] = {reinterpret_cast<uint64_t>(dev_f), sc.u64, size};
    TensorInfo t1_info[] = {ext_f_info};
    int t1 = add_task_with_tensor_info(runtime, args_t1, 3, 1, CoreType::AIV, t1_info, 1);

    add_successor(runtime, t0, t1);

    return 0;
}

}  // extern "C"
