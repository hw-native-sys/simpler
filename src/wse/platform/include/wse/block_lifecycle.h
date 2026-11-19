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
#include <exception>
#include <memory>
#include "wse/block_scope.h"
#include "wse/linx_kernel_abi.h"

namespace wse {
// A Core owns this state until every participating PE has returned from the Block.
class BlockOperationStorage {
public:
    SharedSlotTable slots{};

    void begin(const LinxKernel &kernel, uint32_t pe_mask) {
        core_ = {kernel.create_core(pe_mask), kernel.destroy_core};
        kernel_ = &kernel;
    }
    void invoke(uint32_t pe, int64_t *args) const { kernel_->invoke(core_.get(), pe, kernel_->entry, args); }
    void cancel(const std::exception_ptr &error) const {
        if (core_) kernel_->cancel(core_.get(), error);
    }
    void clear() {
        core_.reset();
        kernel_ = nullptr;
        slots = {};
    }

private:
    const LinxKernel *kernel_{nullptr};
    std::unique_ptr<void, void (*)(void *)> core_{nullptr, nullptr};
};
}  // namespace wse
