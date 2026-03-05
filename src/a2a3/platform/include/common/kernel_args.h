/**
 * @file kernel_args.h
 * @brief KernelArgs Structure - Shared between Host, AICPU, and AICore
 *
 * This structure is used to pass arguments to both AICPU and AICore kernels.
 * It contains pointers to device memory for arguments and runtime data.
 *
 * Platform Support:
 * - a2a3: Real hardware with CANN runtime compatibility
 * - a2a3sim: Host-based simulation using standard memory
 *
 * Memory Layout (a2a3):
 * This structure's layout is hardcoded in libaicpu_extend_kernels.so, which
 * expects specific offsets for deviceArgs fields. The unused[5] array provides
 * the required offset alignment for compatibility with the CANN runtime.
 *
 * Memory Layout (a2a3sim):
 * For simulation, the layout is maintained for API compatibility, though
 * we use host memory instead of device memory.
 */

#ifndef PLATFORM_COMMON_KERNEL_ARGS_H_
#define PLATFORM_COMMON_KERNEL_ARGS_H_

#include <cstdint>

// Forward declarations
class DeviceArgs;
class Runtime;

#ifdef __cplusplus
extern "C" {
#endif

// Define __may_used_by_aicore__ qualifier for platform compatibility
#if defined(__AIV__) || defined(__AIC__)
#define __may_used_by_aicore__ __gm__
#else
#define __may_used_by_aicore__
#endif

/**
 * Kernel arguments structure
 *
 * This structure is passed to AICPU kernels by the host.
 *
 * Field Access Patterns:
 * - unused[5]: Padding for alignment with CANN runtime expectations
 * - device_args: Written by host, read by AICPU (contains aicpu_so_bin/aicpu_so_len)
 * - runtime_args: Written by host, read by AICPU (task runtime, includes
 *   handshake buffers)
 *
 * Note: AICore kernels receive Runtime* directly, not KernelArgs
 *       - AICPU: accesses runtime_args->workers directly
 *       - AICore: receives Runtime* pointer with workers at offset 0
 */
struct KernelArgs {
    uint64_t unused[5] = {0};          // Alignment padding (required by CANN runtime offset)
    DeviceArgs* device_args{nullptr};  // Device arguments (AICPU reads, contains SO info)
    Runtime* runtime_args{nullptr};    // Task runtime in device memory
    uint64_t regs{0};                  // Per-core register base address array (platform-specific)
};

#ifdef __cplusplus
}
#endif

#endif  // PLATFORM_COMMON_KERNEL_ARGS_H_
