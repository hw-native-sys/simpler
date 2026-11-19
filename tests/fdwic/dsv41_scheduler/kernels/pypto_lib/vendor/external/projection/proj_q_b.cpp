// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// NZ attention Q-B projection.
#include "q_lora_projection.hpp"

extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
    // TP output widths, with ordered branches and no jump table.
    if (args[6] > 16384) {
        project_qb<32768>(args);
    } else if (args[6] > 8192) {
        project_qb<16384>(args);
    } else if (args[6] > 4096) {
        project_qb<8192>(args);
    } else {
        project_qb<4096>(args);
    }
}
