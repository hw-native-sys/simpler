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
#include "ccu_session_backend.h"

namespace ccu_test {
// CPU-owned buffers and explicit observations, never a device execution claim.
struct SessionControl {
    uint64_t gate{};
    std::vector<CcuImageBlock> image;
    void Complete(CcuRequestRef request);
};
void QueueBackend(std::shared_ptr<SessionControl> control);
}  // namespace ccu_test
