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
 * a2a3 onboard profiling_copy stubs.
 *
 * a2a3 uses SVM (halHostRegister maps device pointers into host address space)
 * so collectors never need to mirror — host and device share the same memory.
 * The shared profiling framework still pulls in profiling_copy.h to satisfy
 * the symbol references that profile_base.h carries; a2a3 collectors leave
 * `MemoryOps::copy_to_device` / `copy_from_device` unset and the framework
 * never calls these stubs at runtime. They exist purely to make the
 * libhost_runtime.so link succeed on the (unreachable) symbol references.
 */

#include "host/profiling_copy.h"

int profiling_copy_to_device(volatile void * /*dev_dst*/, const void * /*host_src*/, size_t /*size*/) { return 0; }

int profiling_copy_from_device(
    volatile void * /*host_dst*/, const volatile void * /*dev_src*/, size_t /*size*/
) {
    return 0;
}

// SVM: return empty std::function so common leaf collectors install null
// copy callbacks; framework's null-check short-circuits all mirror ops
// and `alloc_paired_buffer` takes the identity-map (host_ptr == dev_ptr)
// branch.
std::function<int(void *, const void *, size_t)> profiling_copy_to_device_or_null() { return {}; }
std::function<int(void *, const void *, size_t)> profiling_copy_from_device_or_null() { return {}; }
