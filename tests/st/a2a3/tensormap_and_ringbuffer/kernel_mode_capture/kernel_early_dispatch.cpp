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
#include "common.h"
#include "runtime_core.h"
#include "task_args.h"

extern "C" void kernel_early_dispatch(const ChipTaskArgs &args) {
    auto *rt = framework_current_runtime();
    rt->pending_scope_mode = ScopeMode::AUTO;
    rt->ops->scope_begin(rt);
    uint32_t shape[] = {128 * 128};
    TensorCreateInfo intermediate(shape, 1, DataType::FLOAT32);
    CoreTaskArgs producer;
    producer.add_input(args.tensor(0).ref());
    producer.add_output(intermediate);
    producer.add_scalar(args.scalar(0));
    producer.set_allow_early_resolve(true);
    MixedKernels kernels;
    kernels.aiv0_kernel_id = 0;
    auto first = rt->ops->submit_task(rt, kernels, producer);
    CoreTaskArgs consumer;
    consumer.add_input(first.get_ref(0));
    consumer.add_output(args.tensor(1).ref());
    consumer.add_scalar(args.scalar(0));
    kernels.aiv0_kernel_id = 1;
    auto second = rt->ops->submit_task(rt, kernels, consumer);
    const auto id = second.task_id();
    auto &ring = rt->orchestrator.sm_header->rings[id.ring()];
    auto &slot = ring.get_slot_state_by_task_id(id.local_id());
    // The open scope pins the completed task while this test reads its actual dispatch outcome.
    while (!slot.is_completion_flag_set() && !rt->ops->is_fatal(rt)) {}
    const auto state =
        ring.get_payload_by_task_id(id.local_id()).early_dispatch_launch_state.load(std::memory_order_acquire);
    uint32_t index[] = {0};
    float marker = state == EARLY_DISPATCH_LAUNCH_COMPLETE ? 1.0f : 0.0f;
    uint64_t bits = 0;
    std::memcpy(&bits, &marker, sizeof(marker));
    rt->ops->set_tensor_data(rt, args.tensor(1).ref(), 1, index, bits);
    rt->ops->scope_end(rt);
}
