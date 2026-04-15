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
 * Example Orchestration Function Implementation
 *
 * Builds the task graph for formula: (a + b + 1)(a + b + 2)
 *
 * This orchestration function:
 * 1. Receives ChipStorageTaskArgs with tensor metadata (pointers, shapes, dtypes)
 * 2. Allocates device memory via orchestration API helpers
 * 3. Copies input data to device via orchestration API helpers
 * 4. Records output tensor for copy-back during finalize
 * 5. Builds the task graph
 */

#include <iostream>

#include "orchestration_api.h"  // NOLINT(build/include_subdir)

extern "C" {

int build_example_graph(OrchestrationRuntime *runtime, const ChipStorageTaskArgs &orch_args) {
    // Validate argument count
    // Expected orch_args: [a, b, f] — 3 tensors
    if (orch_args.tensor_count() < 3) {
        std::cerr << "build_example_graph: Expected at least 3 tensors, got " << orch_args.tensor_count() << '\n';
        return -1;
    }

    // Extract host pointers, sizes, and element count from tensor metadata
    void *host_a = orch_args.tensor(0).data_as<void>();
    void *host_b = orch_args.tensor(1).data_as<void>();
    void *host_f = orch_args.tensor(2).data_as<void>();
    size_t size_a = orch_args.tensor(0).nbytes();
    size_t size_b = orch_args.tensor(1).nbytes();
    size_t size_f = orch_args.tensor(2).nbytes();
    uint32_t SIZE = orch_args.tensor(0).shapes[0];

    std::cout << "\n=== build_example_graph: Creating Task Runtime ===" << '\n';
    std::cout << "Formula: (a + b + 1)(a + b + 2)\n";
    std::cout << "SIZE: " << SIZE << " elements\n";

    // Allocate device memory and copy inputs
    std::cout << "\n=== Allocating Device Memory ===" << '\n';

    void *dev_a = device_malloc(runtime, size_a);
    if (!dev_a) {
        std::cerr << "Error: Failed to allocate device memory for a\n";
        return -1;
    }
    copy_to_device(runtime, dev_a, host_a, size_a);
    std::cout << "Tensor a: " << size_a << " bytes copied to device\n";

    void *dev_b = device_malloc(runtime, size_b);
    if (!dev_b) {
        std::cerr << "Error: Failed to allocate device memory for b\n";
        device_free(runtime, dev_a);
        return -1;
    }
    copy_to_device(runtime, dev_b, host_b, size_b);
    std::cout << "Tensor b: " << size_b << " bytes copied to device\n";

    void *dev_f = device_malloc(runtime, size_f);
    if (!dev_f) {
        std::cerr << "Error: Failed to allocate device memory for f\n";
        device_free(runtime, dev_a);
        device_free(runtime, dev_b);
        return -1;
    }
    // Record output tensor for copy-back during finalize
    record_tensor_pair(runtime, host_f, dev_f, size_f);
    std::cout << "Tensor f (output): " << size_f << " bytes allocated\n";

    // Allocate intermediate tensors (c, d, e)
    size_t BYTES = SIZE * sizeof(float);
    void *dev_c = device_malloc(runtime, BYTES);
    void *dev_d = device_malloc(runtime, BYTES);
    void *dev_e = device_malloc(runtime, BYTES);

    if (!dev_c || !dev_d || !dev_e) {
        std::cerr << "Error: Failed to allocate intermediate tensors\n";
        device_free(runtime, dev_a);
        device_free(runtime, dev_b);
        device_free(runtime, dev_f);
        if (dev_c) device_free(runtime, dev_c);
        if (dev_d) device_free(runtime, dev_d);
        if (dev_e) device_free(runtime, dev_e);
        return -1;
    }

    std::cout << "Allocated intermediate tensors c, d, e\n";

    // Helper union to encode float scalar as uint64_t
    union {
        float f32;
        uint64_t u64;
    } scalar_converter;

    // Task 0: c = a + b (func_id=0: kernel_add, AIV)
    uint64_t args_t0[4];
    args_t0[0] = reinterpret_cast<uint64_t>(dev_a);  // src0
    args_t0[1] = reinterpret_cast<uint64_t>(dev_b);  // src1
    args_t0[2] = reinterpret_cast<uint64_t>(dev_c);  // out
    args_t0[3] = SIZE;                               // size
    int t0 = add_task(runtime, args_t0, 4, 0, CoreType::AIV);

    // Task 1: d = c + 1 (func_id=1: kernel_add_scalar, AIV)
    uint64_t args_t1[4];
    args_t1[0] = reinterpret_cast<uint64_t>(dev_c);  // src
    scalar_converter.f32 = 1.0f;
    args_t1[1] = scalar_converter.u64;               // scalar=1.0
    args_t1[2] = reinterpret_cast<uint64_t>(dev_d);  // out
    args_t1[3] = SIZE;                               // size
    int t1 = add_task(runtime, args_t1, 4, 1, CoreType::AIV);

    // Task 2: e = c + 2 (func_id=1: kernel_add_scalar, AIV)
    uint64_t args_t2[4];
    args_t2[0] = reinterpret_cast<uint64_t>(dev_c);  // src
    scalar_converter.f32 = 2.0f;
    args_t2[1] = scalar_converter.u64;               // scalar=2.0
    args_t2[2] = reinterpret_cast<uint64_t>(dev_e);  // out
    args_t2[3] = SIZE;                               // size
    int t2 = add_task(runtime, args_t2, 4, 1, CoreType::AIV);

    // Task 3: f = d * e (func_id=2: kernel_mul, AIV)
    uint64_t args_t3[4];
    args_t3[0] = reinterpret_cast<uint64_t>(dev_d);  // src0
    args_t3[1] = reinterpret_cast<uint64_t>(dev_e);  // src1
    args_t3[2] = reinterpret_cast<uint64_t>(dev_f);  // out
    args_t3[3] = SIZE;                               // size
    int t3 = add_task(runtime, args_t3, 4, 2, CoreType::AIV);

    // Add dependencies
    add_successor(runtime, t0, t1);  // t0 → t1
    add_successor(runtime, t0, t2);  // t0 → t2
    add_successor(runtime, t1, t3);  // t1 → t3
    add_successor(runtime, t2, t3);  // t2 → t3

    std::cout << "\nTasks:\n";
    std::cout << "  task" << t0 << ": c = a + b\n";
    std::cout << "  task" << t1 << ": d = c + 1\n";
    std::cout << "  task" << t2 << ": e = c + 2\n";
    std::cout << "  task" << t3 << ": f = d * e\n";
    std::cout << "Dependencies: t0→t1, t0→t2, t1→t3, t2→t3\n";

    std::cout << "Created runtime with " << get_task_count(runtime) << " tasks\n";
    print_runtime(runtime);

    return 0;
}

}  // extern "C"
