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
#include "protocol.h"
#include "sram_args.h"
extern "C" void kernel_entry(int64_t *args) {
    for (int i = 0; i < 3; ++i) {
        wse::SramView view;
        if (!wse::make_sram_view(args, *reinterpret_cast<Tensor *>(args[i]), view))
            throw std::invalid_argument("counter deployment requires application SRAM");
    }
    counter_protocol(args, -1);
}
