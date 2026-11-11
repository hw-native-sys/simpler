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

#include <cstddef>
#include <cstdint>
#include <vector>

#include "../task_interface/callable.h"

struct CcuCallableArtifacts {
    std::vector<std::vector<uint8_t>> ccu_modules;

    static CcuCallableArtifacts capture(const void *raw, size_t size) {
        validate_chip_callable_buffer(raw, size);
        const auto *callable = static_cast<const ChipCallable *>(raw);
        return {callable->ccu_modules()};
    }

    static CcuCallableArtifacts capture(const ChipCallable &callable) {
        size_t storage_size = callable.binary_size();
        for (int32_t i = 0; i < callable.child_count(); ++i) {
            const size_t child_size =
                CoreCallable::binary_data_offset() + static_cast<size_t>(callable.child(i).binary_size());
            const size_t child_end = static_cast<size_t>(callable.child_offset(i)) + child_size;
            if (child_end > storage_size) storage_size = child_end;
        }
        if (callable.ccu_modules_size_ != 0) {
            const size_t modules_end =
                static_cast<size_t>(callable.ccu_modules_offset_) + static_cast<size_t>(callable.ccu_modules_size_);
            if (modules_end > storage_size) storage_size = modules_end;
        }
        return capture(&callable, offsetof(ChipCallable, storage_) + storage_size);
    }
};
