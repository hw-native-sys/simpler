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
#include <stdexcept>

// These lifecycle fixtures never request SDMA. Fail if that assumption changes.
namespace pto::comm::sdma {
inline constexpr uint32_t kSdmaMaxChannelGroups = 48;
class SdmaWorkspaceManager {
public:
    static int GetWorkspaceSize(uint64_t *) { throw std::logic_error("SDMA is outside this lifecycle fixture"); }
    bool Init(void *, uint64_t) { throw std::logic_error("SDMA is outside this lifecycle fixture"); }
    void *GetWorkspaceAddr() const { return nullptr; }
    uint64_t GetWorkspaceSize() const { return 0; }
};
}  // namespace pto::comm::sdma
