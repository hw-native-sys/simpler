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
#include "aicpu/platform_entry_args.h"

#include "common/launch_entry_args.h"

namespace {
// Per-launch, like the register tables beside it: every launched thread writes
// the same values, and a launch that publishes no entry region leaves the
// source reading Descriptor so a reader takes the descriptor route.
const void *g_entry_args_base = nullptr;
uint32_t g_entry_args_offset = 0;
uint32_t g_entry_tensor_count = 0;
uint32_t g_entry_scalar_count = 0;
uint32_t g_entry_args_source = static_cast<uint32_t>(EntryArgsSource::Descriptor);
}  // namespace

void set_platform_entry_args(
    const void *args_base, uint32_t offset, uint32_t tensors, uint32_t scalars, uint32_t source
) {
    g_entry_args_base = args_base;
    g_entry_args_offset = offset;
    g_entry_tensor_count = tensors;
    g_entry_scalar_count = scalars;
    g_entry_args_source = source;
}

const void *get_platform_entry_args_base() { return g_entry_args_base; }

uint32_t get_platform_entry_args_offset() { return g_entry_args_offset; }

uint32_t get_platform_entry_tensor_count() { return g_entry_tensor_count; }

uint32_t get_platform_entry_scalar_count() { return g_entry_scalar_count; }

uint32_t get_platform_entry_args_source() { return g_entry_args_source; }
