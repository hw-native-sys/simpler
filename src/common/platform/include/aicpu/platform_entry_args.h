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
/**
 * The launch package's entry-argument header, as the platform AICPU entry
 * received it.
 *
 * The platform entry holds the only pointer to this launch's arguments, and the
 * runtime entry takes a `Runtime *` alone — so a runtime that reads entry values
 * from the launch package needs them forwarded, exactly as the register tables
 * and profiling bases already are.
 *
 * Stored here rather than handed to the runtime, so the two runtimes need no
 * agreement: one reads these, the other never calls them and keeps taking its
 * entry values from the descriptor. `set_platform_entry_args` is called once per
 * launched AICPU thread with identical values, like every setter beside it.
 *
 * Nothing here does pointer arithmetic. The base and the offset stay separate
 * until the runtime has checked the offset, the counts, and the source against
 * the descriptor's own, because a payload address formed from unchecked values
 * is the thing that must not exist.
 *
 * Implementation: common/platform/shared/aicpu/platform_entry_args.cpp
 */

#pragma once

#include <cstdint>

/**
 * Publish this launch's entry-argument header.
 *
 * @param args_base   The launch argument block the platform entry was given
 * @param offset      Byte offset of the entry region inside that block
 * @param tensors     Tensor descriptors the region carries
 * @param scalars     Scalars the region carries
 * @param source      `EntryArgsSource` as an integer; the runtime compares it
 *                    with the descriptor's own before reading anything else
 */
void set_platform_entry_args(
    const void *args_base, uint32_t offset, uint32_t tensors, uint32_t scalars, uint32_t source
);

/** The launch argument block, or null when no launch published one. */
const void *get_platform_entry_args_base();
uint32_t get_platform_entry_args_offset();
uint32_t get_platform_entry_tensor_count();
uint32_t get_platform_entry_scalar_count();
uint32_t get_platform_entry_args_source();
