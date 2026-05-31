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
 * `get_platform()` shared across all 4 platform variants (a2a3 / a2a3sim /
 * a5 / a5sim). The build system passes the per-target name via
 * `target_compile_definitions(... SIMPLER_PLATFORM_NAME="…")` on each
 * host_runtime target. Replaces 4 byte-identical-modulo-string source files.
 */

#include "host/platform_compile_info.h"

#ifndef SIMPLER_PLATFORM_NAME
#error "SIMPLER_PLATFORM_NAME must be defined by the build system (see host CMakeLists)"
#endif

extern "C" const char *get_platform(void) { return SIMPLER_PLATFORM_NAME; }
