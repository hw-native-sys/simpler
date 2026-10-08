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

// Contiguous H2D views copy exactly these bytes. Range arithmetic remains
// checked even on the transitional native POD entry path.
inline bool program_h2d_range_valid(const ChipTensor &tensor) {
    if (tensor.ndims == 0 || tensor.ndims > MAX_TENSOR_DIMS || tensor.dtype >= DataType::DATA_TYPE_NUM) return false;
    uint64_t bytes = get_element_size(tensor.dtype);
    for (uint32_t d = 0; d < tensor.ndims; ++d) {
        if (tensor.shapes[d] == 0) return true;
    }
    for (uint32_t d = 0; d < tensor.ndims; ++d) {
        if (bytes > UINT64_MAX / tensor.shapes[d]) return false;
        bytes *= tensor.shapes[d];
    }
    return bytes <= tensor.buffer.size && bytes <= UINT64_MAX - tensor.buffer.addr;
}

// Whole-call preflight precedes binder allocation and copying. HBG can consume
// HOST/NONE locally; device orchestration cannot. HOST/H2D requires the layout
// the packed device allocation represents.
inline int validate_program_tensor_transfers(
    const ChipStorageTaskArgs *args, bool host_orchestration = false, const ArgDirection *signature = nullptr,
    int sig_count = 0
) {
    for (int i = 0; i < args->tensor_count(); ++i) {
        const auto &t = args->tensor(i);
        const char *reason = tensor_transfer_error(t.address_space, t.transfer);
        int status = PTO_RUNTIME_ERR_INVALID_ARGUMENT;
        if (reason == nullptr && t.address_space == AddressSpace::HOST) {
            status = PTO_RUNTIME_ERR_UNSUPPORTED;
            if (t.transfer == TensorTransfer::NONE) {
                if (!host_orchestration) {
                    reason = "HOST/NONE requires host orchestration";
                } else if (signature == nullptr || i >= sig_count ||
                           (signature[i] != ArgDirection::IN && signature[i] != ArgDirection::OUT &&
                            signature[i] != ArgDirection::INOUT)) {
                    reason = "HOST/NONE requires a declared argument direction";
                    status = PTO_RUNTIME_ERR_INVALID_ARGUMENT;
                }
            } else if (!program_h2d_range_valid(t)) {
                reason = "HOST/H2D view exceeds its backing or address range";
                status = PTO_RUNTIME_ERR_INVALID_ARGUMENT;
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
    // Independent device copies cannot preserve a writable source alias.
    // Missing signatures retain the existing conservative INOUT behavior.
    const auto writes = [signature, sig_count](int index) {
        return signature == nullptr || index >= sig_count || signature[index] != ArgDirection::IN;
    };
    for (int i = 0; i < args->tensor_count(); ++i) {
        const auto &a = args->tensor(i);
        if (a.transfer != TensorTransfer::H2D || a.buffer.addr == 0 || a.nbytes() == 0) continue;
        for (int j = 0; j < i; ++j) {
            const auto &b = args->tensor(j);
            if (b.transfer != TensorTransfer::H2D || b.buffer.addr == 0 || b.nbytes() == 0) continue;
            if ((writes(i) || writes(j)) && a.buffer.addr < b.buffer.addr + b.nbytes() &&
                b.buffer.addr < a.buffer.addr + a.nbytes()) {
                LOG_ERROR(
                    "bind: HOST/H2D arguments %d and %d overlap with a writer; use explicit device storage", j, i
                );
                return PTO_RUNTIME_ERR_UNSUPPORTED;
            }
        }
    }
    return 0;
}
