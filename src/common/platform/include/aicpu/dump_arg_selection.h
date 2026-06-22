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

#ifndef SRC_COMMON_PLATFORM_INCLUDE_AICPU_DUMP_ARG_SELECTION_H_
#define SRC_COMMON_PLATFORM_INCLUDE_AICPU_DUMP_ARG_SELECTION_H_

#include <stdint.h>
#include <string.h>

#include "arg_direction.h"

struct DumpArgSelection {
    static_assert(CORE_MAX_TENSOR_ARGS + CORE_MAX_SCALAR_ARGS <= 64, "dump arg mask assumes at most 64 arguments");

    void clear() {
        dump_arg_mask_ = 0;
        dump_arg_index_ambiguous_mask_ = 0;
        clear_scalar_sources();
        clear_scalar_dtypes(0, CORE_MAX_SCALAR_ARGS);
    }

    uint64_t dump_arg_mask() const { return dump_arg_mask_; }
    uint64_t dump_arg_index_ambiguous_mask() const { return dump_arg_index_ambiguous_mask_; }
    const uint8_t *scalar_dtypes() const { return scalar_dtypes_; }

    void mark_index(int32_t index) {
        if (!is_valid_arg_index(index)) {
            return;
        }
        dump_arg_mask_ |= (uint64_t{1} << index);
    }

    void mark_all(int32_t tensor_count, int32_t scalar_count) {
        if (!is_valid_tensor_count(tensor_count) || !is_valid_scalar_range(0, scalar_count)) {
            return;
        }
        for (int32_t i = 0; i < tensor_count; i++) {
            mark_index(i);
        }
        for (int32_t i = 0; i < scalar_count; i++) {
            mark_index(tensor_count + i);
        }
    }

    bool mark_scalar_by_ptr(uintptr_t ptr, int32_t scalar_count, int32_t tensor_offset) {
        if (!is_valid_scalar_range(0, scalar_count) || !is_valid_tensor_count(tensor_offset) ||
            !is_valid_arg_range(tensor_offset, scalar_count)) {
            return false;
        }

        int32_t first_match = -1;
        int32_t match_count = 0;
        for (int32_t i = 0; i < scalar_count; i++) {
            if (scalar_source_ptrs_[i] == ptr) {
                if (first_match < 0) {
                    first_match = i;
                }
                match_count++;
            }
        }
        if (first_match < 0) {
            return false;
        }

        int32_t arg_index = tensor_offset + first_match;
        if (!is_valid_arg_index(arg_index)) {
            return false;
        }
        mark_index(arg_index);
        if (match_count > 1) {
            mark_index_ambiguous(arg_index);
        }
        return true;
    }

    void record_scalar_source(int32_t slot, uintptr_t ptr, uint8_t dtype) {
        if (!is_valid_scalar_range(slot, 1)) {
            return;
        }
        scalar_source_ptrs_[slot] = ptr;
        scalar_dtypes_[slot] = dtype;
    }

    void clear_scalar_metadata(int32_t start, int32_t count) {
        if (!is_valid_scalar_range(start, count)) {
            return;
        }
        clear_scalar_dtypes(start, count);
        clear_scalar_sources(start, count);
    }

    void copy_scalar_dtypes_from(const DumpArgSelection &src, int32_t dst_offset, int32_t src_offset, int32_t count) {
        if (!is_valid_scalar_range(dst_offset, count) || !is_valid_scalar_range(src_offset, count)) {
            return;
        }
        memcpy(&scalar_dtypes_[dst_offset], &src.scalar_dtypes_[src_offset], count * sizeof(uint8_t));
        clear_scalar_sources(dst_offset, count);
    }

private:
    static constexpr int32_t kDumpArgBitCount = 64;

    static bool is_valid_arg_index(int32_t index) { return index >= 0 && index < kDumpArgBitCount; }

    static bool is_valid_arg_range(int32_t start, int32_t count) {
        if (start < 0 || count < 0 || count > kDumpArgBitCount) {
            return false;
        }
        return start <= kDumpArgBitCount - count;
    }

    static bool is_valid_tensor_count(int32_t tensor_count) {
        return tensor_count >= 0 && tensor_count <= CORE_MAX_TENSOR_ARGS;
    }

    static bool is_valid_scalar_range(int32_t start, int32_t count) {
        if (start < 0 || count < 0 || count > CORE_MAX_SCALAR_ARGS) {
            return false;
        }
        return start <= CORE_MAX_SCALAR_ARGS - count;
    }

    void mark_index_ambiguous(int32_t index) {
        if (!is_valid_arg_index(index)) {
            return;
        }
        dump_arg_index_ambiguous_mask_ |= (uint64_t{1} << index);
    }

    void clear_scalar_sources() { clear_scalar_sources(0, CORE_MAX_SCALAR_ARGS); }

    void clear_scalar_sources(int32_t start, int32_t count) {
        if (!is_valid_scalar_range(start, count)) {
            return;
        }
        for (int32_t i = 0; i < count; i++) {
            scalar_source_ptrs_[start + i] = 0;
        }
    }

    void clear_scalar_dtypes(int32_t start, int32_t count) {
        if (!is_valid_scalar_range(start, count)) {
            return;
        }
        memset(&scalar_dtypes_[start], 0, count * sizeof(uint8_t));
    }

    uint64_t dump_arg_mask_{0};
    uint64_t dump_arg_index_ambiguous_mask_{0};
    uintptr_t scalar_source_ptrs_[CORE_MAX_SCALAR_ARGS]{};
    uint8_t scalar_dtypes_[CORE_MAX_SCALAR_ARGS]{};
};

#endif  // SRC_COMMON_PLATFORM_INCLUDE_AICPU_DUMP_ARG_SELECTION_H_
