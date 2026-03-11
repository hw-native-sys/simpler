/**
 * AICPU orchestration for the vector example.
 *
 * Runs on AICPU. The framework has already allocated device memory for I/O
 * tensors and populated orch_args[] with device pointers and scalar values:
 *
 *   orch_args[0] = dev_a      (input,  float[SIZE])
 *   orch_args[1] = dev_b      (input,  float[SIZE])
 *   orch_args[2] = dev_f      (output, float[SIZE])
 *   orch_args[3] = SIZE        (element count, scalar)
 *
 * This function allocates intermediate tensors via api.device_malloc() (HBM)
 * and builds the task dependency graph:
 *
 *   c = a + b        (task 0, func_id=0)
 *   d = c + 1.0      (task 1, func_id=1, depends on task 0)
 *   e = c + 2.0      (task 2, func_id=1, depends on task 0)
 *   f = d * e         (task 3, func_id=2, depends on tasks 1 and 2)
 */

#include <cstdint>

#include "runtime.h"

namespace {
union ScalarConverter {
    float f32;
    uint64_t u64;
};

constexpr int DEV_A = 0;
constexpr int DEV_B = 1;
constexpr int DEV_F = 2;
constexpr int SIZE  = 3;
}  // namespace

extern "C" int orchestration(Runtime* runtime) {
    if (runtime == nullptr) {
        return -1;
    }

    if (runtime->orch_argc < SIZE + 1) {
        return -1;
    }

    const uint64_t dev_a = runtime->orch_args[DEV_A];
    const uint64_t dev_b = runtime->orch_args[DEV_B];
    const uint64_t dev_f = runtime->orch_args[DEV_F];
    const int size = static_cast<int>(runtime->orch_args[SIZE]);

    if (dev_a == 0 || dev_b == 0 || dev_f == 0 || size <= 0) {
        return -1;
    }

    const AicpuBuildApi& api = runtime->aicpu_build_api;
    if (api.add_task == nullptr || api.add_successor_conditional == nullptr ||
        api.publish_task == nullptr || api.device_malloc == nullptr) {
        return -1;
    }

    // Allocate intermediate tensors on device (HBM, accessible by AIV cores).
    // Note: malloc() on AICPU returns AICPU-local memory which AIV cores cannot access.
    size_t bytes = static_cast<size_t>(size) * sizeof(float);
    void* dev_c = api.device_malloc(bytes);
    void* dev_d = api.device_malloc(bytes);
    void* dev_e = api.device_malloc(bytes);
    if (dev_c == nullptr || dev_d == nullptr || dev_e == nullptr) {
        return -1;
    }

    // Task 0: c = a + b (func_id=0, AIV)
    uint64_t args_t0[4];
    args_t0[0] = dev_a;
    args_t0[1] = dev_b;
    args_t0[2] = reinterpret_cast<uint64_t>(dev_c);
    args_t0[3] = static_cast<uint64_t>(size);
    int t0 = api.add_task(runtime, args_t0, 4, 0, CoreType::AIV, 0);
    if (t0 < 0) return -1;
    api.publish_task(runtime, t0);

    // Task 1: d = c + 1 (func_id=1, AIV)
    ScalarConverter s1{};
    s1.f32 = 1.0f;
    uint64_t args_t1[4];
    args_t1[0] = reinterpret_cast<uint64_t>(dev_c);
    args_t1[1] = s1.u64;
    args_t1[2] = reinterpret_cast<uint64_t>(dev_d);
    args_t1[3] = static_cast<uint64_t>(size);
    int t1 = api.add_task(runtime, args_t1, 4, 1, CoreType::AIV, 0);
    if (t1 < 0) return -1;
    api.add_successor_conditional(runtime, t0, t1);
    api.publish_task(runtime, t1);

    // Task 2: e = c + 2 (func_id=1, AIV)
    ScalarConverter s2{};
    s2.f32 = 2.0f;
    uint64_t args_t2[4];
    args_t2[0] = reinterpret_cast<uint64_t>(dev_c);
    args_t2[1] = s2.u64;
    args_t2[2] = reinterpret_cast<uint64_t>(dev_e);
    args_t2[3] = static_cast<uint64_t>(size);
    int t2 = api.add_task(runtime, args_t2, 4, 1, CoreType::AIV, 0);
    if (t2 < 0) return -1;
    api.add_successor_conditional(runtime, t0, t2);
    api.publish_task(runtime, t2);

    // Task 3: f = d * e (func_id=2, AIV)
    uint64_t args_t3[4];
    args_t3[0] = reinterpret_cast<uint64_t>(dev_d);
    args_t3[1] = reinterpret_cast<uint64_t>(dev_e);
    args_t3[2] = dev_f;
    args_t3[3] = static_cast<uint64_t>(size);
    int t3 = api.add_task(runtime, args_t3, 4, 2, CoreType::AIV, 0);
    if (t3 < 0) return -1;
    api.add_successor_conditional(runtime, t1, t3);
    api.add_successor_conditional(runtime, t2, t3);
    api.publish_task(runtime, t3);

    if (runtime->kernel_addrs[0] == 0 || runtime->kernel_addrs[1] == 0 || runtime->kernel_addrs[2] == 0) {
        return -1;
    }

    return 0;
}