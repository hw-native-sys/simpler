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
#include <vector>

#include "task_interface/arg_direction.h"
#include "task_interface/tensor.h"

// One host-orchestration call's explicitly bound HOST/NONE views. The call owns
// the registrations, while its caller retains the backing through completion.
// Readers may run on recording threads after registration; add/close are exclusive.
class HostTensorAccessor {
public:
    // Empty views register no bytes. Overlapping views with a writer are rejected.
    bool add(const ChipTensor &tensor, ArgDirection direction);
    bool read(uint64_t addr, void *dst, uint64_t bytes) const;
    bool write(uint64_t addr, const void *src, uint64_t bytes) const;
    void close() noexcept { regions_.clear(); }

private:
    struct Region {
        uint64_t base;
        uint64_t size;
        ArgDirection direction;
    };
    const Region *find(uint64_t addr, uint64_t bytes) const;
    std::vector<Region> regions_;
};

bool host_tensor_read(HostTensorAccessor *accessor, uint64_t addr, void *dst, uint64_t bytes);
bool host_tensor_write(HostTensorAccessor *accessor, uint64_t addr, const void *src, uint64_t bytes);
