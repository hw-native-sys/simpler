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

#include "orchestration_api.h"

extern "C" void kernel_tmr_chain(const ChipTaskArgs &args) {
    SIMPLER_SCOPE_GUARD();
    uint32_t shape[] = {128 * 128};
    TensorCreateInfo intermediate(shape, 1, DataType::FLOAT32);
    CoreTaskArgs first;
    first.add_input(args.tensor(0).ref());
    first.add_output(intermediate);
    first.add_scalar(args.scalar(0));
    auto root = rt_submit_aiv_task(0, first);
    CoreTaskArgs left;
    left.add_input(root.get_ref(0));
    left.add_output(intermediate);
    left.add_scalar(args.scalar(0));
    auto branch_left = rt_submit_aiv_task(0, left);
    CoreTaskArgs right;
    right.add_input(root.get_ref(0));
    right.add_output(intermediate);
    right.add_scalar(uint64_t{0});
    auto branch_right = rt_submit_aiv_task(0, right);
    CoreTaskArgs join;
    join.add_input(branch_left.get_ref(0));
    join.add_input(branch_right.get_ref(0));
    join.add_output(args.tensor(1).ref());
    rt_submit_aiv_task(1, join);
}
