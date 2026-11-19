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

extern "C" void capture_capacity_orchestration(const ChipTaskArgs &args) {
    const uint32_t shape[]{static_cast<uint32_t>(args.scalar<uint64_t>(0))};
    TensorCreateInfo intermediate(shape, 1, DataType::INT32);
    CoreTaskArgs first;
    first.add_input(args.tensor(0).ref());
    first.add_output(intermediate);
    first.add_scalar(uint64_t{3});
    auto result = rt_submit_aiv_task(0, first);
    CoreTaskArgs second;
    second.add_input(result.get_ref(0));
    second.add_output(args.tensor(1).ref());
    second.add_scalar(uint64_t{7});
    rt_submit_aiv_task(0, second);
}
