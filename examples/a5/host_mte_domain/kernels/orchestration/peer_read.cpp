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
__attribute__((visibility("default"))) OrchestrationConfig host_mte_domain_config(const ChipTaskArgs &args) {
    (void)args;
    return OrchestrationConfig{.expected_arg_count = 5};
}

__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &args) {
    return host_mte_domain_config(args);
}

__attribute__((visibility("default"))) void host_mte_domain(const ChipTaskArgs &args) {
    CoreTaskArgs copy;
    copy.add_input(args.tensor(0).ref());
    copy.add_output(args.tensor(1).ref());
    copy.add_scalar(args.scalar<uint64_t>(0));
    copy.add_scalar(args.scalar<uint64_t>(1));
    rt_submit_aiv_task(0, copy);

    CoreTaskArgs consume;
    consume.add_input(args.tensor(1).ref());
    consume.add_output(args.tensor(2).ref());
    rt_submit_aiv_task(1, consume);
}
}
