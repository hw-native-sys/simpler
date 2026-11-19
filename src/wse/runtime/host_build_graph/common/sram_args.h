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
#include "intrinsic.h"
#include "sram_access.h"

namespace wse {
inline bool
make_sram_view(int64_t *args, const Tensor &tensor, SramView &view, uint32_t expected_owner = UINT32_MAX) noexcept {
    view = {};
    const auto *global = reinterpret_cast<const GlobalContext *>(args[SPMD_GLOBAL_CONTEXT_INDEX]);
    uint64_t address = 0, bytes = 0;
    if (!global || !global->sram_base || !tensor_span(tensor, address, bytes)) return false;
    const auto &access = *reinterpret_cast<const SramAccess *>(global->sram_base + global->sram_access_offset);
    return resolve_sram(global->sram_base, access, address, bytes, view, expected_owner);
}
}  // namespace wse
