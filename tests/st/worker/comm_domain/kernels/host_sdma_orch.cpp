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
__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &) {
    return OrchestrationConfig{.expected_arg_count = 3};
}

__attribute__((visibility("default"))) void host_sdma(const ChipTaskArgs &args) {
    CoreTaskArgs transfer;
    transfer.add_input(args.tensor(0).ref());
    transfer.add_output(args.tensor(1).ref());
    rt_submit_aiv_task(0, transfer);
    // The existing completion manager publishes the producer output only after
    // PTO reports transfer completion; the consumer must observe the new data.
    CoreTaskArgs consumer;
    consumer.add_input(args.tensor(1).ref());
    consumer.add_output(args.tensor(2).ref());
    rt_submit_aiv_task(1, consumer);
}
}
