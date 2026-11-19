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
#include "../common.h"

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    auto *first = visibility_case::data(args, 0);
    auto *second = visibility_case::data(args, 1);
    auto *out = visibility_case::data(args, 2);
    out[17] = 1;
    out[18] = first[0] + second[8];
    visibility_case::publish(out + 17, 2);
}
