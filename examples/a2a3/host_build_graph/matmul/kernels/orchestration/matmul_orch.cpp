/**
 * Matmul Example Orchestration Function Implementation
 *
 * Builds the task graph for formula: F = exp(sqrt(log(A)) @ W1 + sqrt(log(A)) @ W2)
 *
 * Task graph (diamond topology):
 *       t0 (sqrt+log, AIV)
 *      /  \
 *    t1    t2   (matmul, AIC)
 *      \  /
 *       t3 (add+exp, AIV)
 *
 * This orchestration function:
 * 1. Receives host pointers and sizes in args
 * 2. Allocates device memory via runtime->host_api
 * 3. Copies input data to device via runtime->host_api
 * 4. Records output tensor for copy-back during finalize
 * 5. Builds the task graph with 4 tasks (2 AIV + 2 AIC)
 */

#include "runtime.h"
#include <iostream>
#include <cstdint>

extern "C" {

int build_matmul_graph(Runtime* runtime, uint64_t* args, int arg_count) {
    // Validate argument count
    // Expected args: [host_a, host_w1, host_w2, host_f, size_a, size_w1, size_w2, size_f, SIZE]
    if (arg_count < 9) {
        std::cerr << "build_matmul_graph: Expected at least 9 args, got " << arg_count << '\n';
        return -1;
    }

    // Extract arguments - host pointers and sizes
    void* host_a  = reinterpret_cast<void*>(args[0]);
    void* host_w1 = reinterpret_cast<void*>(args[1]);
    void* host_w2 = reinterpret_cast<void*>(args[2]);
    void* host_f  = reinterpret_cast<void*>(args[3]);
    size_t size_a  = static_cast<size_t>(args[4]);
    size_t size_w1 = static_cast<size_t>(args[5]);
    size_t size_w2 = static_cast<size_t>(args[6]);
    size_t size_f  = static_cast<size_t>(args[7]);
    int SIZE = static_cast<int>(args[8]);

    std::cout << "\n=== build_matmul_graph: Creating Task Runtime ===" << '\n';
    std::cout << "Formula: F = exp(sqrt(log(A)) @ W1 + sqrt(log(A)) @ W2)\n";
    std::cout << "SIZE: " << SIZE << " elements\n";

    // Allocate device memory and copy inputs
    std::cout << "\n=== Allocating Device Memory ===" << '\n';

    void* dev_a = runtime->host_api.device_malloc(size_a);
    if (!dev_a) {
        std::cerr << "Error: Failed to allocate device memory for A\n";
        return -1;
    }
    runtime->host_api.copy_to_device(dev_a, host_a, size_a);
    std::cout << "Tensor A: " << size_a << " bytes copied to device\n";

    void* dev_w1 = runtime->host_api.device_malloc(size_w1);
    if (!dev_w1) {
        std::cerr << "Error: Failed to allocate device memory for W1\n";
        runtime->host_api.device_free(dev_a);
        return -1;
    }
    runtime->host_api.copy_to_device(dev_w1, host_w1, size_w1);
    std::cout << "Tensor W1: " << size_w1 << " bytes copied to device\n";

    void* dev_w2 = runtime->host_api.device_malloc(size_w2);
    if (!dev_w2) {
        std::cerr << "Error: Failed to allocate device memory for W2\n";
        runtime->host_api.device_free(dev_a);
        runtime->host_api.device_free(dev_w1);
        return -1;
    }
    runtime->host_api.copy_to_device(dev_w2, host_w2, size_w2);
    std::cout << "Tensor W2: " << size_w2 << " bytes copied to device\n";

    void* dev_f = runtime->host_api.device_malloc(size_f);
    if (!dev_f) {
        std::cerr << "Error: Failed to allocate device memory for F\n";
        runtime->host_api.device_free(dev_a);
        runtime->host_api.device_free(dev_w1);
        runtime->host_api.device_free(dev_w2);
        return -1;
    }
    // Record output tensor for copy-back during finalize
    runtime->record_tensor_pair(host_f, dev_f, size_f);
    std::cout << "Tensor F (output): " << size_f << " bytes allocated\n";

    // Allocate intermediate tensors (b, c, d)
    // dev_b is half precision (output of log_sqrt kernel, input to matmul)
    // dev_c, dev_d are float precision (output of matmul kernels)
    size_t BYTES_HALF = SIZE * sizeof(uint16_t);   // half = 2 bytes
    size_t BYTES_FLOAT = SIZE * sizeof(float);     // float = 4 bytes
    void* dev_b = runtime->host_api.device_malloc(BYTES_HALF);   // sqrt(log(A)) - half output
    void* dev_c = runtime->host_api.device_malloc(BYTES_FLOAT);  // B @ W1 - float output
    void* dev_d = runtime->host_api.device_malloc(BYTES_FLOAT);  // B @ W2 - float output

    if (!dev_b || !dev_c || !dev_d) {
        std::cerr << "Error: Failed to allocate intermediate tensors\n";
        runtime->host_api.device_free(dev_a);
        runtime->host_api.device_free(dev_w1);
        runtime->host_api.device_free(dev_w2);
        runtime->host_api.device_free(dev_f);
        if (dev_b) runtime->host_api.device_free(dev_b);
        if (dev_c) runtime->host_api.device_free(dev_c);
        if (dev_d) runtime->host_api.device_free(dev_d);
        return -1;
    }

    std::cout << "Allocated intermediate tensors: B (" << BYTES_HALF << " bytes, half), C (" << BYTES_FLOAT << " bytes, float), D (" << BYTES_FLOAT << " bytes, float)\n";

    // Task 0: B = sqrt(log(A)) (func_id=0: kernel_log_sqrt, AIV)
    uint64_t args_t0[3];
    args_t0[0] = reinterpret_cast<uint64_t>(dev_a);  // src
    args_t0[1] = reinterpret_cast<uint64_t>(dev_b);  // out
    args_t0[2] = SIZE;                                // size
    int t0 = runtime->add_task(args_t0, 3, 0, CoreType::AIV);

    // Task 1: C = B @ W1 (func_id=1: kernel_matmul, AIC)
    uint64_t args_t1[4];
    args_t1[0] = reinterpret_cast<uint64_t>(dev_b);   // src0 (left matrix)
    args_t1[1] = reinterpret_cast<uint64_t>(dev_w1);  // src1 (right matrix)
    args_t1[2] = reinterpret_cast<uint64_t>(dev_c);   // out
    args_t1[3] = SIZE;                                 // size
    int t1 = runtime->add_task(args_t1, 4, 1, CoreType::AIC);

    // Task 2: D = B @ W2 (func_id=1: kernel_matmul, AIC)
    uint64_t args_t2[4];
    args_t2[0] = reinterpret_cast<uint64_t>(dev_b);   // src0 (left matrix)
    args_t2[1] = reinterpret_cast<uint64_t>(dev_w2);  // src1 (right matrix)
    args_t2[2] = reinterpret_cast<uint64_t>(dev_d);   // out
    args_t2[3] = SIZE;                                 // size
    int t2 = runtime->add_task(args_t2, 4, 1, CoreType::AIC);

    // Task 3: F = exp(C + D) (func_id=2: kernel_add_exp, AIV)
    uint64_t args_t3[4];
    args_t3[0] = reinterpret_cast<uint64_t>(dev_c);  // src0
    args_t3[1] = reinterpret_cast<uint64_t>(dev_d);  // src1
    args_t3[2] = reinterpret_cast<uint64_t>(dev_f);  // out
    args_t3[3] = SIZE;                                // size
    int t3 = runtime->add_task(args_t3, 4, 2, CoreType::AIV);

    // Add dependencies (diamond: t0→t1→t3, t0→t2→t3)
    runtime->add_successor(t0, t1);  // t0 → t1
    runtime->add_successor(t0, t2);  // t0 → t2
    runtime->add_successor(t1, t3);  // t1 → t3
    runtime->add_successor(t2, t3);  // t2 → t3

    std::cout << "\nTasks:\n";
    std::cout << "  task" << t0 << ": B = sqrt(log(A))   [AIV]\n";
    std::cout << "  task" << t1 << ": C = B @ W1         [AIC]\n";
    std::cout << "  task" << t2 << ": D = B @ W2         [AIC]\n";
    std::cout << "  task" << t3 << ": F = exp(C + D)     [AIV]\n";
    std::cout << "Dependencies: t0→t1→t3, t0→t2→t3 (diamond)\n";

    std::cout << "Created runtime with " << runtime->get_task_count() << " tasks\n";
    runtime->print_runtime();

    return 0;
}

}  // extern "C"
