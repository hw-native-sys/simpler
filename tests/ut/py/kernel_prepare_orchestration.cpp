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

// Registration resolves this symbol but never invokes the orchestration.
extern "C" void kernel_prepare_orchestration() {}

#include "orchestration_api.h"

extern "C" void kernel_call_orchestration(const ChipTaskArgs &args) {
    if (args.scalar<int32_t>(1) != 0) {
        const uint32_t index[] = {0};
        const auto value = get_tensor_data<int32_t>(args.tensor(0).ref(), 1, index);
        set_tensor_data<int32_t>(args.tensor(0).ref(), 1, index, value + args.scalar<int32_t>(0));
    }
}
