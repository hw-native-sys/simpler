/**
 * Compile Strategy - Toolchain Type Definitions
 *
 * Defines the available toolchains for compiling incore kernels and
 * orchestration functions. Each value maps to a specific compiler binary.
 * Compile arguments differ per situation and are handled in Python.
 *
 * Used by:
 * - Platform: get_platform() declares platform identity
 * - Runtime: get_incore_compiler() / get_orchestration_compiler() return
 *   the appropriate toolchain based on the current platform
 * - Python (via ctypes): dispatches compilation based on the returned toolchain
 */

#ifndef COMPILE_STRATEGY_H
#define COMPILE_STRATEGY_H

typedef enum {
    TOOLCHAIN_CCEC = 0,          // ccec (Ascend AICore compiler)
    TOOLCHAIN_HOST_GXX_15 = 1,   // g++-15 (host, simulation kernels)
    TOOLCHAIN_HOST_GXX = 2,      // g++ (host, orchestration .so)
    TOOLCHAIN_AARCH64_GXX = 3,   // aarch64-target-linux-gnu-g++ (cross-compile)
} ToolchainType;

#endif /* COMPILE_STRATEGY_H */
