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

// A complete, aligned SoC SRAM SET/read slot exposes the A5 atomic ADD alias.
// Return zero when the supplied address has no verified ADD alias.
inline constexpr uint64_t a5_soc_sram_add_address(uint64_t set_address) {
    return set_address != 0 && (set_address & UINT64_C(0x1f)) == 0 ? set_address + UINT64_C(0x08) : 0;
}
