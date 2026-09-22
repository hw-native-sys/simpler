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
 * Compile Strategy - Toolchain Type Definitions
 *
 * Names the toolchains that compile incore kernels and orchestration functions.
 * Each value maps to a specific compiler binary. Compile arguments differ per
 * situation and are handled in Python.
 *
 * The choice itself is made in Python, by
 * `simpler_setup/kernel_compiler.py`: `_orchestration_toolchain()` picks by
 * runtime name and `compile_incore()` by platform. Three
 * `runtime_compile_info.cpp` files return one of these values from
 * `get_incore_compiler()` / `get_orchestration_compiler()` to state the same
 * intent, but nothing calls those — there is no ctypes dispatch on them. Keep
 * them in step with the Python when either changes, or retire them; a value
 * that disagrees with the Python is a trap, because it reads as the decision
 * and is not one.
 *
 * Those three are not three copies of one answer. The two under
 * `src/{a2a3,a5}/runtime/host_build_graph/host/` are identical and move
 * together; the one under `src/common/tensormap_and_ringbuffer/host/` returns
 * `TOOLCHAIN_AARCH64_GXX` for `a2a3` orchestration and does not follow them,
 * because tensormap_and_ringbuffer orchestration executes on the AICPU while
 * host_build_graph's executes on the host. The Python draws the same line by
 * runtime name.
 */

#ifndef COMPILE_STRATEGY_H
#define COMPILE_STRATEGY_H

typedef enum {
    TOOLCHAIN_CCEC = 0,         // ccec (Ascend AICore compiler)
    TOOLCHAIN_HOST_GXX_15 = 1,  // g++-15 (host, simulation kernels)
    TOOLCHAIN_HOST_GXX = 2,     // g++ (host, orchestration .so)
    TOOLCHAIN_AARCH64_GXX = 3,  // aarch64-target-linux-gnu-g++ (cross-compile)
} ToolchainType;

#endif /* COMPILE_STRATEGY_H */
