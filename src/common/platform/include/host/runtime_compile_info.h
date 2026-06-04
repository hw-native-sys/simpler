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
 * Runtime Compile Info Interface
 *
 * Each runtime (host_build_graph, tensormap_and_ringbuffer, ...) implements
 * these functions to declare which toolchain should be used for incore kernel
 * and orchestration compilation. The implementations call get_platform()
 * internally to make platform-aware decisions.
 */

#ifndef RUNTIME_COMPILE_INFO_H
#define RUNTIME_COMPILE_INFO_H

#include "common/compile_strategy.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Get the toolchain for incore kernel compilation.
 *
 * @return ToolchainType for compiling incore kernels
 */
ToolchainType get_incore_compiler(void);

/**
 * Get the toolchain for orchestration function compilation.
 *
 * @return ToolchainType for compiling orchestration .so
 */
ToolchainType get_orchestration_compiler(void);

#ifdef __cplusplus
}
#endif

#endif /* RUNTIME_COMPILE_INFO_H */
