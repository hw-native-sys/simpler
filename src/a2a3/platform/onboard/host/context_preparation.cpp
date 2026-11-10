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

#include "context_preparation.h"

#include "host/host_regs.h"
#include "host/memory_allocator.h"

#include "runtime/rt.h"

namespace pto::a2a3 {
int ensure_ctrl_table(MemoryAllocator &allocator, int device_id, uint64_t &table, bool &committed) {
    if (committed) return 0;
    const int rc =
        init_aicore_register_addresses(&table, static_cast<uint64_t>(device_id), allocator, AicoreRegKind::Ctrl);
    if (rc == 0) committed = true;
    return rc;
}

int ensure_c2c_ctrl_address(uint64_t &address, bool &valid) {
    if (valid) return 0;
    uint64_t queried_address = 0;
    uint32_t length = 0;
    const int rc = rtGetC2cCtrlAddr(&queried_address, &length);
    if (rc != 0) return rc;
    address = queried_address;
    valid = true;
    return 0;
}
}  // namespace pto::a2a3
