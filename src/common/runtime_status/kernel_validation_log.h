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

#include <cinttypes>

#include "common/unified_log.h"
#include "task_interface/kernel_error.h"

inline void log_kernel_validation(const kernel_validation::Failure &failure, uint64_t epoch) {
    if (!is_kernel_error(failure.code)) return;
    if (failure.arg_index < 0) {
        LOG_ERROR(
            "Kernel validation: code=%d task=0x%" PRIx64 " func=%d block=%d epoch=%" PRIu64, failure.code, failure.task,
            failure.func, failure.block, epoch
        );
        return;
    }
    LOG_ERROR(
        "Kernel validation: code=%d task=0x%" PRIx64 " func=%d block=%d arg=%d value=%" PRId64 " range=[%" PRId64
        ",%" PRId64 ") epoch=%" PRIu64,
        failure.code, failure.task, failure.func, failure.block, failure.arg_index, failure.value, failure.lower,
        failure.upper, epoch
    );
}
