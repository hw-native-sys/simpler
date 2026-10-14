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
#pragma once

#include <cstdint>

/**
 * The tag a prepared run's storage carries while its carrier object is alive.
 *
 * The platform writes it into storage the worker owns and zeroes it when the carrier is destroyed.
 * Reading it is a byte read of that storage, never a use of an object whose lifetime may have
 * ended. A value that is neither this nor zero names no state either side may act on.
 */
inline constexpr uint64_t kNativeRunContextMagic = UINT64_C(0x534d504c52554e31);  // "SMPLRUN1"
