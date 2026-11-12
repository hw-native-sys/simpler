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
#include "host/platform_compile_info.h"
#include "host/runtime_compile_info.h"
#include <string.h>

extern "C" {

ToolchainType get_incore_compiler(void) {
    if (strcmp(get_platform(), "a2a3") == 0) return TOOLCHAIN_CCEC;
    return TOOLCHAIN_HOST_GXX_15;
}

ToolchainType get_orchestration_compiler(void) {
    // host_build_graph: always host g++ (orchestration runs on host)
    return TOOLCHAIN_HOST_GXX;
}
}
