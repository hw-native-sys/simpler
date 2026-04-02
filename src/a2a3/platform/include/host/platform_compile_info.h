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
 * Platform Compile Info Interface
 *
 * Minimal interface: platform only declares its identity.
 * Each platform (a2a3, a2a3sim, ...) implements get_platform() to return
 * its name. Runtime code uses this to decide which toolchain to use.
 */

#ifndef PLATFORM_COMPILE_INFO_H
#define PLATFORM_COMPILE_INFO_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Get the platform name.
 *
 * @return Platform identifier string (e.g., "a2a3", "a2a3sim")
 */
const char *get_platform(void);

#ifdef __cplusplus
}
#endif

#endif /* PLATFORM_COMPILE_INFO_H */
