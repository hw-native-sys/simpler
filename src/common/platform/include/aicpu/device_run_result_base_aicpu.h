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
#include <cstddef>
#include "common/run_terminal_accumulator.h"

/** The active result binding in the AICPU DSO. Native begin sets it once
 * before the gate releases callbacks. The atomic setter also serves standalone
 * producers and sim arming; readers never use the pair as retirement proof. */
extern "C" void set_platform_run_result(uint64_t region_base, uint64_t run_epoch);
extern "C" uint64_t get_platform_run_result_base();
extern "C" uint64_t get_platform_run_result_epoch();

// A nonzero base must name a cold-initialized NativeInvocationResultStorage,
// not merely a standalone result prefix. The owner retains it through all use.
// A zero base requests no private record; the epoch still identifies execution.
// Raw legacy 0/0 is accepted, but a nonzero base requires a nonzero epoch.
extern "C" int32_t aicpu_begin_native_invocation(int32_t callbacks, uint64_t region_base, uint64_t run_epoch);
extern "C" bool aicpu_accept_runtime_terminal(RunTerminalSelection selection, const void *payload, size_t bytes);
extern "C" int32_t aicpu_finish_native_invocation(int32_t callback_index, int32_t status);
