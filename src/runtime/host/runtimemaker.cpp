/**
 * Runtime Builder - Basic Example
 *
 * Initializes a pre-allocated runtime with the following task structure:
 * Formula: (a + b + 1)(a + b + 2)
 *
 * Tasks:
 *   task0: c = a + b (kernel_add)
 *   task1: d = c + 1 (kernel_add_scalar)
 *   task2: e = c + 2 (kernel_add_scalar)
 *   task3: f = d * e (kernel_mul)
 *
 * Dependencies:
 *   task0 -> task1
 *   task0 -> task2
 *   task1 -> task3
 *   task2 -> task3
 */

#include <stddef.h>
#include <stdint.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <new>
#include <string>
#include <vector>

#include "devicerunner.h"
#include "runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

// Static storage for tensor pointers (used by ValidateRuntimeImpl)
static void* g_dev_a = nullptr;
static void* g_dev_b = nullptr;
static void* g_dev_c = nullptr;
static void* g_dev_d = nullptr;
static void* g_dev_e = nullptr;
static void* g_dev_f = nullptr;
static size_t g_tensor_bytes = 0;

/**
 * Initialize a pre-allocated runtime for the basic example.
 *
 * This function takes a pre-constructed Runtime pointer and builds the complete
 * example runtime inside it. Runtime is already constructed via placement new.
 *
 * @param runtime    Pointer to pre-constructed Runtime
 * @return 0 on success, -1 on failure
 */
int InitRuntimeImpl(Runtime* runtime) {
    int rc = 0;

    // Initialize DeviceRunner
    DeviceRunner& runner = DeviceRunner::Get();
    // Note: DeviceRunner should already be initialized by Python before calling
    // InitRuntime Note: Kernels should be registered via Python's
    // runner.register_kernel() before calling InitRuntime

    // Allocate device tensors
    constexpr int ROWS = 128;
    constexpr int COLS = 128;
    constexpr int SIZE = ROWS * COLS;  // 16384 elements
    constexpr size_t BYTES = SIZE * sizeof(float);

    std::cout << "\n=== Allocating Device Memory ===" << '\n';
    void* dev_a = runner.AllocateTensor(BYTES);
    void* dev_b = runner.AllocateTensor(BYTES);
    void* dev_c = runner.AllocateTensor(BYTES);
    void* dev_d = runner.AllocateTensor(BYTES);
    void* dev_e = runner.AllocateTensor(BYTES);
    void* dev_f = runner.AllocateTensor(BYTES);

    if (!dev_a || !dev_b || !dev_c || !dev_d || !dev_e || !dev_f) {
        std::cerr << "Error: Failed to allocate device tensors" << '\n';
        return -1;
    }
    std::cout << "Allocated 6 tensors (128x128 each, " << BYTES << " bytes per tensor)\n";

    // Initialize input data and copy to device
    std::vector<float> host_a(SIZE, 2.0f);
    std::vector<float> host_b(SIZE, 3.0f);

    rc = runner.CopyToDevice(dev_a, host_a.data(), BYTES);
    if (rc != 0) {
        std::cerr << "Error: Failed to copy input a to device" << '\n';
        runner.FreeTensor(dev_a);
        runner.FreeTensor(dev_b);
        runner.FreeTensor(dev_c);
        runner.FreeTensor(dev_d);
        runner.FreeTensor(dev_e);
        runner.FreeTensor(dev_f);
        return rc;
    }

    rc = runner.CopyToDevice(dev_b, host_b.data(), BYTES);
    if (rc != 0) {
        std::cerr << "Error: Failed to copy input b to device" << '\n';
        runner.FreeTensor(dev_a);
        runner.FreeTensor(dev_b);
        runner.FreeTensor(dev_c);
        runner.FreeTensor(dev_d);
        runner.FreeTensor(dev_e);
        runner.FreeTensor(dev_f);
        return rc;
    }

    std::cout << "Initialized input tensors: a=2.0, b=3.0 (all elements)\n";
    std::cout << "Expected result: f = (2+3+1)*(2+3+2) = 6*7 = 42.0\n";

    // Store tensor pointers for later use by ValidateRuntimeImpl
    g_dev_a = dev_a;
    g_dev_b = dev_b;
    g_dev_c = dev_c;
    g_dev_d = dev_d;
    g_dev_e = dev_e;
    g_dev_f = dev_f;
    g_tensor_bytes = BYTES;

    // =========================================================================
    // BUILD RUNTIME - This is the core runtime building logic
    // =========================================================================
    std::cout << "\n=== Creating Task Runtime for Formula ===" << '\n';
    std::cout << "Formula: (a + b + 1)(a + b + 2)\n";
    std::cout << "Tasks:\n";
    std::cout << "  task0: c = a + b\n";
    std::cout << "  task1: d = c + 1\n";
    std::cout << "  task2: e = c + 2\n";
    std::cout << "  task3: f = d * e\n\n";

    // Helper union to encode float scalar as uint64_t
    union {
        float f32;
        uint64_t u64;
    } scalar_converter;

    // Task 0: c = a + b (func_id=0: kernel_add)
    uint64_t args_t0[4];
    args_t0[0] = reinterpret_cast<uint64_t>(dev_a);  // src0
    args_t0[1] = reinterpret_cast<uint64_t>(dev_b);  // src1
    args_t0[2] = reinterpret_cast<uint64_t>(dev_c);  // out
    args_t0[3] = SIZE;                               // size
    int t0 = runtime->add_task(args_t0, 4, 0);

    // Task 1: d = c + 1 (func_id=1: kernel_add_scalar)
    uint64_t args_t1[4];
    args_t1[0] = reinterpret_cast<uint64_t>(dev_c);  // src
    scalar_converter.f32 = 1.0f;
    args_t1[1] = scalar_converter.u64;               // scalar=1.0
    args_t1[2] = reinterpret_cast<uint64_t>(dev_d);  // out
    args_t1[3] = SIZE;                               // size
    int t1 = runtime->add_task(args_t1, 4, 1);

    // Task 2: e = c + 2 (func_id=1: kernel_add_scalar)
    uint64_t args_t2[4];
    args_t2[0] = reinterpret_cast<uint64_t>(dev_c);  // src
    scalar_converter.f32 = 2.0f;
    args_t2[1] = scalar_converter.u64;               // scalar=2.0
    args_t2[2] = reinterpret_cast<uint64_t>(dev_e);  // out
    args_t2[3] = SIZE;                               // size
    int t2 = runtime->add_task(args_t2, 4, 1);

    // Task 3: f = d * e (func_id=2: kernel_mul)
    uint64_t args_t3[4];
    args_t3[0] = reinterpret_cast<uint64_t>(dev_d);  // src0
    args_t3[1] = reinterpret_cast<uint64_t>(dev_e);  // src1
    args_t3[2] = reinterpret_cast<uint64_t>(dev_f);  // out
    args_t3[3] = SIZE;                               // size
    int t3 = runtime->add_task(args_t3, 4, 2);

    // Add dependencies
    runtime->add_successor(t0, t1);  // t0 → t1
    runtime->add_successor(t0, t2);  // t0 → t2
    runtime->add_successor(t1, t3);  // t1 → t3
    runtime->add_successor(t2, t3);  // t2 → t3

    std::cout << "Created runtime with " << runtime->get_task_count() << " tasks\n";
    runtime->print_runtime();

    std::cout << "\nRuntime initialized. Ready for execution from Python.\n";

    return 0;
}

int ValidateRuntimeImpl(Runtime* runtime) {
    if (runtime == nullptr) {
        std::cerr << "Error: Runtime pointer is null\n";
        return -1;
    }

    // Get DeviceRunner instance
    DeviceRunner& runner = DeviceRunner::Get();

    // Use globally stored tensor pointers
    void* dev_a = g_dev_a;
    void* dev_b = g_dev_b;
    void* dev_c = g_dev_c;
    void* dev_d = g_dev_d;
    void* dev_e = g_dev_e;
    void* dev_f = g_dev_f;
    size_t BYTES = g_tensor_bytes;

    constexpr int ROWS = 128;
    constexpr int COLS = 128;
    constexpr int SIZE = ROWS * COLS;  // Must match InitRuntimeImpl

    // =========================================================================
    // VALIDATE RESULTS - Retrieve and verify output
    // =========================================================================
    std::cout << "\n=== Validating Results ===" << '\n';
    std::vector<float> host_result(SIZE);
    int rc = runner.CopyFromDevice(host_result.data(), dev_f, BYTES);
    if (rc != 0) {
        std::cerr << "Error: Failed to copy result from device: " << rc << '\n';
        runner.FreeTensor(dev_a);
        runner.FreeTensor(dev_b);
        runner.FreeTensor(dev_c);
        runner.FreeTensor(dev_d);
        runner.FreeTensor(dev_e);
        runner.FreeTensor(dev_f);
        return rc;
    }

    // Print sample values
    std::cout << "First 10 elements of result:\n";
    for (int i = 0; i < 10; i++) {
        std::cout << "  f[" << i << "] = " << host_result[i] << '\n';
    }

    // Validate result
    constexpr float EXPECTED = 42.0f;  // (2+3+1)*(2+3+2) = 6*7 = 42
    bool all_correct = true;
    int error_count = 0;
    for (int i = 0; i < SIZE; i++) {
        if (std::abs(host_result[i] - EXPECTED) > 0.001f) {
            if (error_count < 5) {
                std::cerr << "ERROR: f[" << i << "] = " << host_result[i] << ", expected " << EXPECTED << '\n';
            }
            error_count++;
            all_correct = false;
        }
    }

    if (all_correct) {
        std::cout << "\n✓ SUCCESS: All " << SIZE << " elements are correct (42.0)\n";
        std::cout << "Formula verified: (a + b + 1)(a + b + 2) = "
                     "(2+3+1)*(2+3+2) = 42\n";
    } else {
        std::cerr << "\n✗ FAILED: " << error_count << " elements are incorrect\n";
    }

    // Print handshake results
    runner.PrintHandshakeResults(*runtime);

    // Cleanup device tensors
    std::cout << "\n=== Cleaning Up ===" << '\n';
    runner.FreeTensor(dev_a);
    runner.FreeTensor(dev_b);
    runner.FreeTensor(dev_c);
    runner.FreeTensor(dev_d);
    runner.FreeTensor(dev_e);
    runner.FreeTensor(dev_f);
    std::cout << "Freed all device tensors\n";

    // Note: Runtime destructor is called by FinalizeRuntime() after this
    // returns User will call free() after FinalizeRuntime()

    // Clear global tensor pointers
    g_dev_a = g_dev_b = g_dev_c = g_dev_d = g_dev_e = g_dev_f = nullptr;
    g_tensor_bytes = 0;

    if (rc != 0 || !all_correct) {
        std::cerr << "=== Execution Failed ===" << '\n';
        return -1;
    } else {
        std::cout << "=== Success ===" << '\n';
    }

    return 0;
}

#ifdef __cplusplus
} /* extern "C" */
#endif
