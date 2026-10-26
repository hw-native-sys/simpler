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

/**
 * Four producers of three different kinds feeding one SPMD consumer.
 *
 * The consumer's fanin is four and only its last resolution may queue it. Each
 * producer resolves it from a different place: a single-block AIC task and a
 * single-block AIV task resolve from an ordinary completion, and two SPMD
 * producers of different widths resolve from the block-counting one. Pushing
 * the consumer on anything but the last of the four puts the same task into
 * the shared queue twice, which is a corrupt list rather than a wrong number.
 *
 * Args layout: [out] + {aic_blocks, aiv_blocks, consumer_blocks, repeats}.
 */

#include <cstdint>

#include "orchestration_api.h"  // NOLINT(build/include_subdir)

namespace {
constexpr int32_t kProducerAicFuncId = 0;
constexpr int32_t kProducerAivFuncId = 1;
constexpr int32_t kConsumerFuncId = 2;

constexpr int64_t kSingleAicValueBase = 1;
constexpr int64_t kSingleAivValueBase = 2;
constexpr int64_t kSpmdAicValueBase = 10;
constexpr int64_t kSpmdAivValueBase = 20;
}  // namespace

extern "C" {

__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &args) {
    (void)args;
    return OrchestrationConfig{.expected_arg_count = 5};
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    const simpler::hbg::Tensor &out = args.tensor(0).ref();
    const int64_t aic_blocks = args.scalar<int64_t>(0);
    const int64_t aiv_blocks = args.scalar<int64_t>(1);
    const int64_t consumer_blocks = args.scalar<int64_t>(2);
    const int64_t repeats = args.scalar<int64_t>(3);

    // Slot order matches the submission order below, and the consumer reads
    // every slot before its own base.
    const int64_t single_aic_slot = 0;
    const int64_t single_aiv_slot = 1;
    const int64_t spmd_aic_slot = 2;
    const int64_t spmd_aiv_slot = spmd_aic_slot + aic_blocks;
    const int64_t consumer_slot = spmd_aiv_slot + aiv_blocks;

    TaskId produced[4];

    CoreTaskArgs single_aic;
    single_aic.add_no_dep(out);
    single_aic.add_scalar(single_aic_slot);
    single_aic.add_scalar(kSingleAicValueBase);
    single_aic.add_scalar(repeats);
    produced[0] = rt_submit_aic_task(kProducerAicFuncId, single_aic).task_id();

    CoreTaskArgs single_aiv;
    single_aiv.add_no_dep(out);
    single_aiv.add_scalar(single_aiv_slot);
    single_aiv.add_scalar(kSingleAivValueBase);
    single_aiv.add_scalar(repeats * 2);
    produced[1] = rt_submit_aiv_task(kProducerAivFuncId, single_aiv).task_id();

    CoreTaskArgs spmd_aic;
    spmd_aic.add_no_dep(out);
    spmd_aic.add_scalar(spmd_aic_slot);
    spmd_aic.add_scalar(kSpmdAicValueBase);
    spmd_aic.add_scalar(repeats * 3);
    spmd_aic.launch_spec.set_block_num(static_cast<int32_t>(aic_blocks));
    produced[2] = rt_submit_aic_task(kProducerAicFuncId, spmd_aic).task_id();

    CoreTaskArgs spmd_aiv;
    spmd_aiv.add_no_dep(out);
    spmd_aiv.add_scalar(spmd_aiv_slot);
    spmd_aiv.add_scalar(kSpmdAivValueBase);
    spmd_aiv.add_scalar(repeats * 4);
    spmd_aiv.launch_spec.set_block_num(static_cast<int32_t>(aiv_blocks));
    produced[3] = rt_submit_aiv_task(kProducerAivFuncId, spmd_aiv).task_id();

    CoreTaskArgs consumer;
    consumer.add_no_dep(out);
    // The producers fill every slot below the consumer's base, so one value is
    // both how many slots to read and where to start writing.
    consumer.add_scalar(consumer_slot);
    consumer.add_scalar(consumer_slot);
    consumer.launch_spec.set_block_num(static_cast<int32_t>(consumer_blocks));
    consumer.set_dependencies(produced, 4);
    (void)rt_submit_aic_task(kConsumerFuncId, consumer);
}

}  // extern "C"
