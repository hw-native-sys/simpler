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

#include <array>
#include <cstdint>
#include <utility>
#include <vector>

#include "acl/acl.h"

namespace simpler::a3 {

// Explicitly released after every imported view is acknowledged gone.
// Destruction cannot prove that acknowledgement and does not free the backing.
class HostDomainMemory {
public:
    std::pair<uint64_t, std::array<uint8_t, 128>> allocate(int device, uint32_t numa, uint64_t logical_bytes);
    void authorize(const std::vector<int32_t> &pids);
    uint64_t base() const;
    void release();
    static int process_identity();

private:
    int device_ = -1;
    void *base_ = nullptr;
    aclrtDrvMemHandle handle_ = nullptr;
    bool mapped_ = false;
    bool closing_ = false;
    uint64_t size_ = 0;
    std::array<uint8_t, 128> token_{};
};

}  // namespace simpler::a3
