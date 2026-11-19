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
#include "wse/operation_trace.h"
#include "../../sim/operations/tile_ops.h"

namespace wse {
inline constexpr const char *OPERATION_BACKEND = "wse-temporary-fp32";
using temporary::Layout;
using temporary::SharedTile;
using temporary::TensorView;
using temporary::Tile;
using temporary::TLOAD;
using temporary::TMATMUL;
using temporary::TMATMUL_ACC;
using temporary::TRELU;
using temporary::TSTORE;
}  // namespace wse
