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

#include "args.h"
#include "aicpu_loader/finalize_args.h"

extern "C" int simpler_aicpu_init(void *) { return 0; }

extern "C" int simpler_aicpu_exec(void *raw) {
    const auto &args = *static_cast<CaptureArgs *>(raw);
    *reinterpret_cast<int64_t *>(args.output) = *reinterpret_cast<const int64_t *>(args.first) + args.addend;
    __asm__ volatile("dsb sy" ::: "memory");
    return 0;
}

// This tiny payload owns no persistent callable or allocation. Its caller
// destroys the captured graph before requesting this acknowledgement.
extern "C" int simpler_aicpu_finalize(void *arg) {
    auto *args = static_cast<AicpuFinalizeArgs *>(arg);
    if (args == nullptr || args->result_addr == 0) return -1;
    auto *result = reinterpret_cast<AicpuFinalizeResult *>(args->result_addr);
    result->status = 0;
    result->completed = 1;
    __asm__ volatile("dc cvac, %0" ::"r"(result) : "memory");
    __asm__ volatile("dsb sy" ::: "memory");
    return 0;
}
