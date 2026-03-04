/**
 * GEMM + Gather orchestration (a2a3).
 *
 * Two independent tasks (no dependency):
 *   Task 0: C = A @ B (64x64 GEMM, AIC)
 *   Task 1: out = src0[src1] (Gather, AIV, comm-isa shape)
 */

#include "runtime.h"
#include <iostream>
#include <cstdint>

extern "C" {

constexpr int GEMM_TILE = 64;
constexpr int GATHER_SRC0_ROWS = 32;
constexpr int GATHER_SRC0_COLS = 1024;
constexpr int GATHER_SRC1_ROWS = 16;
constexpr int GATHER_SRC1_COLS = 64;

int build_gemm_gather_graph(Runtime* runtime, uint64_t* args, int arg_count) {
    if (arg_count < 12) {
        std::cerr << "build_gemm_gather_graph: Expected at least 12 args, got " << arg_count << '\n';
        return -1;
    }

    void* host_A = reinterpret_cast<void*>(args[0]);
    void* host_B = reinterpret_cast<void*>(args[1]);
    void* host_C = reinterpret_cast<void*>(args[2]);
    void* host_src0 = reinterpret_cast<void*>(args[3]);
    void* host_src1 = reinterpret_cast<void*>(args[4]);
    void* host_out = reinterpret_cast<void*>(args[5]);
    size_t size_A = static_cast<size_t>(args[6]);
    size_t size_B = static_cast<size_t>(args[7]);
    size_t size_C = static_cast<size_t>(args[8]);
    size_t size_src0 = static_cast<size_t>(args[9]);
    size_t size_src1 = static_cast<size_t>(args[10]);
    size_t size_out = static_cast<size_t>(args[11]);

    std::cout << "\n=== build_gemm_gather_graph (a2a3) ===" << '\n';

    // Allocate device memory and copy inputs
    void* dev_A = runtime->host_api.device_malloc(size_A);
    if (!dev_A) return -1;
    runtime->host_api.copy_to_device(dev_A, host_A, size_A);

    void* dev_B = runtime->host_api.device_malloc(size_B);
    if (!dev_B) {
        runtime->host_api.device_free(dev_A);
        return -1;
    }
    runtime->host_api.copy_to_device(dev_B, host_B, size_B);

    void* dev_C = runtime->host_api.device_malloc(size_C);
    if (!dev_C) {
        runtime->host_api.device_free(dev_A);
        runtime->host_api.device_free(dev_B);
        return -1;
    }
    runtime->record_tensor_pair(host_C, dev_C, size_C);

    void* dev_src0 = runtime->host_api.device_malloc(size_src0);
    if (!dev_src0) {
        runtime->host_api.device_free(dev_A);
        runtime->host_api.device_free(dev_B);
        runtime->host_api.device_free(dev_C);
        return -1;
    }
    runtime->host_api.copy_to_device(dev_src0, host_src0, size_src0);

    void* dev_src1 = runtime->host_api.device_malloc(size_src1);
    if (!dev_src1) {
        runtime->host_api.device_free(dev_A);
        runtime->host_api.device_free(dev_B);
        runtime->host_api.device_free(dev_C);
        runtime->host_api.device_free(dev_src0);
        return -1;
    }
    runtime->host_api.copy_to_device(dev_src1, host_src1, size_src1);

    void* dev_out = runtime->host_api.device_malloc(size_out);
    if (!dev_out) {
        runtime->host_api.device_free(dev_A);
        runtime->host_api.device_free(dev_B);
        runtime->host_api.device_free(dev_C);
        runtime->host_api.device_free(dev_src0);
        runtime->host_api.device_free(dev_src1);
        return -1;
    }
    runtime->record_tensor_pair(host_out, dev_out, size_out);

    // Task 0: GEMM C = A @ B (func_id=0, AIC)
    uint64_t args_geom[3];
    args_geom[0] = reinterpret_cast<uint64_t>(dev_A);
    args_geom[1] = reinterpret_cast<uint64_t>(dev_B);
    args_geom[2] = reinterpret_cast<uint64_t>(dev_C);
    int t0 = runtime->add_task(args_geom, 3, 0, CoreType::AIC);

    // Task 1: Gather out = src0[src1] (func_id=1, AIV)
    uint64_t args_gather[3];
    args_gather[0] = reinterpret_cast<uint64_t>(dev_out);
    args_gather[1] = reinterpret_cast<uint64_t>(dev_src0);
    args_gather[2] = reinterpret_cast<uint64_t>(dev_src1);
    int t1 = runtime->add_task(args_gather, 3, 1, CoreType::AIV);

    // Dependency: compute first, then communication (t0 -> t1)
    runtime->add_successor(t0, t1);

    std::cout << "  task" << t0 << ": GEMM C=A@B [AIC]\n";
    std::cout << "  task" << t1 << ": Gather out=src0[src1] [AIV]\n";
    std::cout << "  Dependency: t0 -> t1 (compute then communication).\n";

    return 0;
}

}  // extern "C"
