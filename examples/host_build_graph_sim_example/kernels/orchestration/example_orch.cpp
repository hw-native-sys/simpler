/**
 * Example Orchestration Function Implementation
 *
 * Builds the task graph for formula: (a + b + 1)(a + b + 2)
 *
 * This orchestration function:
 * 1. Receives host pointers and sizes in args
 * 2. Allocates device memory via runtime->host_api
 * 3. Copies input data to device via runtime->host_api
 * 4. Records output tensor for copy-back during finalize
 * 5. Builds the task graph
 */

// Include runtime.h first to get full Runtime class definition
#include "runtime.h"
#include <iostream>

extern "C" {

int build_example_graph(Runtime* runtime, uint64_t* args, int arg_count) {
    // Validate argument count
    // Expected args: [host_a, host_b, host_f, size_a, size_b, size_f, SIZE]
    if (arg_count < 7) {
        std::cerr << "build_example_graph: Expected at least 7 args, got " << arg_count << '\n';
        return -1;
    }

    // Extract arguments - host pointers and sizes
    void* host_a = reinterpret_cast<void*>(args[0]);
    void* host_b = reinterpret_cast<void*>(args[1]);
    void* host_f = reinterpret_cast<void*>(args[2]);
    size_t size_a = static_cast<size_t>(args[3]);
    size_t size_b = static_cast<size_t>(args[4]);
    size_t size_f = static_cast<size_t>(args[5]);
    int SIZE = static_cast<int>(args[6]);

    std::cout << "\n=== build_example_graph: Creating Task Runtime ===" << '\n';
    std::cout << "Formula: (a + b + 1)(a + b + 2)\n";
    std::cout << "SIZE: " << SIZE << " elements\n";

    // Allocate device memory and copy inputs
    std::cout << "\n=== Allocating Device Memory ===" << '\n';

    void* dev_a = runtime->host_api.device_malloc(size_a);
    if (!dev_a) {
        std::cerr << "Error: Failed to allocate device memory for a\n";
        return -1;
    }
    runtime->host_api.copy_to_device(dev_a, host_a, size_a);
    std::cout << "Tensor a: " << size_a << " bytes copied to device\n";

    void* dev_b = runtime->host_api.device_malloc(size_b);
    if (!dev_b) {
        std::cerr << "Error: Failed to allocate device memory for b\n";
        runtime->host_api.device_free(dev_a);
        return -1;
    }
    runtime->host_api.copy_to_device(dev_b, host_b, size_b);
    std::cout << "Tensor b: " << size_b << " bytes copied to device\n";

    void* dev_f = runtime->host_api.device_malloc(size_f);
    if (!dev_f) {
        std::cerr << "Error: Failed to allocate device memory for f\n";
        runtime->host_api.device_free(dev_a);
        runtime->host_api.device_free(dev_b);
        return -1;
    }
    // Record output tensor for copy-back during finalize
    runtime->record_tensor_pair(host_f, dev_f, size_f);
    std::cout << "Tensor f (output): " << size_f << " bytes allocated\n";

    // Allocate intermediate tensors (c, d, e)
    size_t BYTES = SIZE * sizeof(float);
    void* dev_c = runtime->host_api.device_malloc(BYTES);
    void* dev_d = runtime->host_api.device_malloc(BYTES);
    void* dev_e = runtime->host_api.device_malloc(BYTES);

    if (!dev_c || !dev_d || !dev_e) {
        std::cerr << "Error: Failed to allocate intermediate tensors\n";
        runtime->host_api.device_free(dev_a);
        runtime->host_api.device_free(dev_b);
        runtime->host_api.device_free(dev_f);
        if (dev_c) runtime->host_api.device_free(dev_c);
        if (dev_d) runtime->host_api.device_free(dev_d);
        if (dev_e) runtime->host_api.device_free(dev_e);
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
    args_t0[3] = SIZE;                                // size
    int t0 = runtime->add_task(args_t0, 4, 0, 1);

    // Task 1: d = c + 1 (func_id=1: kernel_add_scalar, AIV)
    uint64_t args_t1[4];
    args_t1[0] = reinterpret_cast<uint64_t>(dev_c);  // src
    scalar_converter.f32 = 1.0f;
    args_t1[1] = scalar_converter.u64;                // scalar=1.0
    args_t1[2] = reinterpret_cast<uint64_t>(dev_d);  // out
    args_t1[3] = SIZE;                                // size
    int t1 = runtime->add_task(args_t1, 4, 1, 1);

    // Task 2: e = c + 2 (func_id=1: kernel_add_scalar, AIV)
    uint64_t args_t2[4];
    args_t2[0] = reinterpret_cast<uint64_t>(dev_c);  // src
    scalar_converter.f32 = 2.0f;
    args_t2[1] = scalar_converter.u64;                // scalar=2.0
    args_t2[2] = reinterpret_cast<uint64_t>(dev_e);  // out
    args_t2[3] = SIZE;                                // size
    int t2 = runtime->add_task(args_t2, 4, 1, 1);

    // Task 3: f = d * e (func_id=2: kernel_mul, AIV)
    uint64_t args_t3[4];
    args_t3[0] = reinterpret_cast<uint64_t>(dev_d);  // src0
    args_t3[1] = reinterpret_cast<uint64_t>(dev_e);  // src1
    args_t3[2] = reinterpret_cast<uint64_t>(dev_f);  // out
    args_t3[3] = SIZE;                                // size
    int t3 = runtime->add_task(args_t3, 4, 2, 1);

    // Add dependencies
    runtime->add_successor(t0, t1);  // t0 → t1
    runtime->add_successor(t0, t2);  // t0 → t2
    runtime->add_successor(t1, t3);  // t1 → t3
    runtime->add_successor(t2, t3);  // t2 → t3

    std::cout << "\nTasks:\n";
    std::cout << "  task" << t0 << ": c = a + b\n";
    std::cout << "  task" << t1 << ": d = c + 1\n";
    std::cout << "  task" << t2 << ": e = c + 2\n";
    std::cout << "  task" << t3 << ": f = d * e\n";
    std::cout << "Dependencies: t0→t1, t0→t2, t1→t3, t2→t3\n";

    std::cout << "Created runtime with " << runtime->get_task_count() << " tasks\n";
    runtime->print_runtime();

    return 0;
}

}  // extern "C"
