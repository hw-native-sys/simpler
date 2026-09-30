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

#include "common/unified_log.h"
#include "task_args.h"
#include "worker/runtime_c_api.h"

// Whole-call preflight runs before sizing, allocating, or copying any input. HOST/NONE is
// a valid request for a host leaf, but chip binders have no host-only binding yet.
// HOST transfers allocate nbytes() and copy from buffer.addr without rebasing
// the descriptor; the supported layout is contiguous with zero start_offset.
inline int validate_program_tensor_transfers(const ChipStorageTaskArgs *args) {
    for (int i = 0; i < args->tensor_count(); ++i) {
        const auto &t = args->tensor(i);
        const char *reason = tensor_transfer_error(t.address_space, t.transfer);
        int status = PTO_RUNTIME_ERR_INVALID_ARGUMENT;
        if (reason == nullptr && t.address_space == AddressSpace::HOST) {
            status = PTO_RUNTIME_ERR_UNSUPPORTED;
            if (t.transfer == TensorTransfer::NONE) {
                reason = "HOST/NONE is not supported by the chip binder";
            } else if (!t.is_contiguous() || t.start_offset != 0) {
                reason = "HOST/H2D requires contiguous strides and zero start_offset";
            }
        }
        if (reason == nullptr) continue;
        LOG_ERROR(
            "bind: tensor %d address_space=%u transfer=%u: %s", i, static_cast<unsigned>(t.address_space),
            static_cast<unsigned>(t.transfer), reason
        );
        return status;
    }
    return 0;
}
