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

extern "C" {
__attribute__((visibility("default"))) OrchestrationConfig host_domain_copy_config(const ChipTaskArgs &args) {
    (void)args;
    return OrchestrationConfig{.expected_arg_count = 5};
}

__attribute__((visibility("default"))) void host_domain_copy(const ChipTaskArgs &args) {
    CoreTaskArgs params;
    params.add_input(args.tensor(0).ref());
    params.add_output(args.tensor(1).ref());
    for (int index = 0; index < 3; ++index)
        params.add_scalar(args.scalar(index));
    rt_submit_aiv_task(0, params);
}
}
