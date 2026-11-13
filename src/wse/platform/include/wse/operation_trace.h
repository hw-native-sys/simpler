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

namespace wse {
enum class OperationEvent : uint32_t { PUBLISH = 12, WAIT = 13, READ = 14, CONSUMED = 15 };
// Optional observation only; no allocation, execution or ownership hooks.
struct OperationTrace {
    void *owner{nullptr};
    void (*record)(void *, OperationEvent, uint32_t slot, uint64_t generation){nullptr};
    void operator()(OperationEvent event, uint32_t slot, uint64_t generation) const {
        if (record) record(owner, event, slot, generation);
    }
};
}  // namespace wse
