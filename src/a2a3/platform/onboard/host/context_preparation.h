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

class MemoryAllocator;

namespace pto::a2a3 {
int ensure_ctrl_table(MemoryAllocator &allocator, int device_id, uint64_t &table, bool &committed);
// A successful zero address is still a completed query. The caller owns the snapshot.
int ensure_c2c_ctrl_address(uint64_t &address, bool &valid);
}  // namespace pto::a2a3
