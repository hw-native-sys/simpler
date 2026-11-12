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

#include "async_condition_writer.h"

// The request test substitutes kernel context extraction and ISA posting.
// Resident writer validation, request admission and condition storage are real.
struct AsyncCtx {
    uint32_t resident_enabled{0};
    uint32_t urma_sender_id{UINT32_MAX};
    pto2::resident::AsyncWriter resident{};
};

namespace pto2::detail {
inline void defer_error(AsyncCtx &ctx, int32_t error) { pto2::resident::detail::fail(ctx.resident, error); }
}  // namespace pto2::detail
