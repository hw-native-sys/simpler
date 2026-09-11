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

namespace {
OrchestrationConfig configure(const ChipTaskArgs &args, uint64_t id) {
    auto *output = reinterpret_cast<uint64_t *>(args.tensor(0).ref().buffer.addr);
    output[0] = id;
    output[1] = args.scalar(0);
    output[2] = reinterpret_cast<uint64_t>(&args.tensor(0).ref());
    return OrchestrationConfig{2};
}

void execute(const ChipTaskArgs &args, uint64_t id) {
    auto *output = reinterpret_cast<uint64_t *>(args.tensor(0).ref().buffer.addr);
    output[3] = id;
    output[4] = args.scalar(0);
    output[5] = reinterpret_cast<uint64_t>(&args.tensor(0).ref());
    CoreTaskArgs empty;
    rt_submit_dummy_task(empty);
}
}  // namespace

extern "C" OrchestrationConfig config_a(const ChipTaskArgs &args) { return configure(args, 3); }
extern "C" OrchestrationConfig config_b(const ChipTaskArgs &args) { return configure(args, 4); }
extern "C" OrchestrationConfig config_mismatch(const ChipTaskArgs &args) {
    configure(args, 5);
    return OrchestrationConfig{3};
}
extern "C" void orchestration_a(const ChipTaskArgs &args) { execute(args, 3); }
extern "C" void orchestration_b(const ChipTaskArgs &args) { execute(args, 4); }
