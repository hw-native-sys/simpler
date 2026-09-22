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

#include <cstdint>

#include "aicpu/platform_regs.h"

// a5 retires one core at a time through its own register entry, where a2a3 retires a
// group against a deadline — so the emergency-shutdown half the cold path carries
// needs a different stub per architecture. The stall-dump tests never reach it; only
// the link does.
int32_t __attribute__((weak)) platform_deinit_aicore_regs(uint64_t) { return 0; }
