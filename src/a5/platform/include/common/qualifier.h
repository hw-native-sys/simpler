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
 * @file qualifier.h
 * @brief Memory qualifier fallbacks for non-AICore builds
 *
 * On real AICore the compiler provides __gm__, __global__, etc. natively.
 * For AICPU, Host, and simulation builds these are no-ops.
 */

#ifndef PLATFORM_COMMON_QUALIFIER_H_
#define PLATFORM_COMMON_QUALIFIER_H_

#ifndef __gm__
#define __gm__
#endif

#ifndef __global__
#define __global__
#endif

#ifndef __in__
#define __in__
#endif

#ifndef __out__
#define __out__
#endif

#endif  // PLATFORM_COMMON_QUALIFIER_H_
