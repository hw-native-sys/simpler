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

extern "C" __attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &) {
    return OrchestrationConfig{.expected_arg_count = 6};  // five tensors and the snapshot runner's phase
}

extern "C" __attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    CoreTaskArgs a;
    a.add_input(args.tensor(0).ref());
    a.add_inout(args.tensor(1).ref());
    rt_submit_aic_task(0, a);

    CoreTaskArgs b;
    b.add_input(args.tensor(1).ref());
    b.add_inout(args.tensor(2).ref());
    rt_submit_aiv_task(1, b);

    CoreTaskArgs c;
    c.add_input(args.tensor(1).ref());
    c.add_inout(args.tensor(3).ref());
    rt_submit_aiv_task(2, c);

    CoreTaskArgs d;
    d.add_input(args.tensor(2).ref());
    d.add_input(args.tensor(3).ref());
    d.add_inout(args.tensor(4).ref());
    rt_submit_aic_task(3, d);
}
