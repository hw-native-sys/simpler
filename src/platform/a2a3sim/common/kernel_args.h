/**
 * KernelArgs Structure - Shared between Host, AICPU, and AICore (Simulation)
 *
 * This structure is used to pass arguments to both AICPU and AICore kernels.
 * For simulation, we use host memory so all pointers are host pointers.
 *
 * This is a copy from a2a3 platform, unchanged for API compatibility.
 */

#ifndef RUNTIME_COMMON_KERNEL_ARGS_H
#define RUNTIME_COMMON_KERNEL_ARGS_H

#include <cstdint>

// Forward declaration
class Runtime;

#ifdef __cplusplus
extern "C" {
#endif

// For simulation, __may_used_by_aicore__ is just empty
#define __may_used_by_aicore__

/**
 * Kernel arguments structure
 *
 * This structure is passed to AICPU kernels by the host.
 *
 * For simulation purposes, this structure is simplified but maintains
 * API compatibility with the real a2a3 platform.
 */
struct KernelArgs {
    Runtime* runtime_args{nullptr};    // Task runtime pointer
};

#ifdef __cplusplus
}
#endif

#endif  // RUNTIME_COMMON_KERNEL_ARGS_H
