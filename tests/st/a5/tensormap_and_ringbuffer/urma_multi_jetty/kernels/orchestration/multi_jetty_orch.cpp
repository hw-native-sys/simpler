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

#include <cstdint>

#include "arg_with_deps.h"
#include "orchestration_api.h"
#include "../records.h"

namespace {
template <typename Tensor>
Tensor view(const Tensor &tensor, uint32_t count, uint32_t offset) {
    uint32_t shape[] = {count};
    uint32_t start[] = {offset};
    return tensor.view(shape, start);
}
}  // namespace

extern "C" __attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &) {
    return OrchestrationConfig{.expected_arg_count = 13};
}

extern "C" __attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &orch_args) {
    const auto &source = orch_args.tensor(0).ref();
    const auto &receive = orch_args.tensor(1).ref();
    const auto &out = orch_args.tensor(2).ref();
    const auto &result = orch_args.tensor(3).ref();
    const auto &records = orch_args.tensor(4).ref();
    const auto &observations = orch_args.tensor(5).ref();
    const uint64_t comm_ctx = orch_args.scalar<uint64_t>(0);
    const uint32_t elems = orch_args.scalar<uint32_t>(1);
    const uint32_t slots = orch_args.scalar<uint32_t>(2);
    const uint32_t waves = orch_args.scalar<uint32_t>(3);
    const uint32_t repeats = orch_args.scalar<uint32_t>(4);
    const bool put = orch_args.scalar<uint32_t>(5) != 0;
    const bool fixed_senders = orch_args.scalar<uint32_t>(6) != 0;
    // A full SPMD cohort lets the wrap test reuse the same physical AIVs in every wave.
    const int32_t transfer_blocks = fixed_senders ? rt_available_aiv_count() : static_cast<int32_t>(slots);
    TaskId previous = TaskId::invalid();
    const uint32_t phases = waves + (put ? 1 : 0);
    for (uint32_t wave = 0; wave < phases; ++wave) {
        SIMPLER_SCOPE_GUARD();
        const bool readback = wave == waves;
        auto record_view = view(records, slots * urma_test::kRecordWords, wave * slots * urma_test::kRecordWords);
        auto observed_view =
            view(observations, slots * urma_test::kObservationWords, wave * slots * urma_test::kObservationWords);
        CoreTaskArgsWithDeps<1> transfer;
        transfer.add_input(source);
        transfer.add_input(receive);
        // Every wave follows its previous observer. This chain also orders writes to the reused output.
        transfer.add_no_dep(out, record_view);
        transfer.add_scalar(comm_ctx);
        transfer.add_scalar(elems);
        transfer.add_scalar(slots);
        transfer.add_scalar(readback ? 1 : repeats);
        transfer.add_scalar(readback ? 2 : (put ? 1 : 0));
        transfer.add_scalar(static_cast<uint32_t>(fixed_senders));
        if (previous.is_valid()) transfer.add_dep(previous);
        transfer.launch_spec.set_block_num(static_cast<int16_t>(transfer_blocks));
        transfer.launch_spec.set_require_sync_start(true);
        const auto posted = rt_submit_aiv_task(0, transfer).task_id();

        CoreTaskArgsWithDeps<1> observe;
        observe.add_no_dep(record_view, observed_view);
        observe.add_dep(posted);
        observe.add_scalar(comm_ctx);
        observe.launch_spec.set_block_num(static_cast<int16_t>(slots));
        previous = rt_submit_aiv_task(1, observe).task_id();
    }
    CoreTaskArgsWithDeps<1> consumer;
    consumer.add_no_dep(out);
    consumer.add_output(result);
    consumer.add_scalar(slots * elems);
    consumer.add_dep(previous);
    rt_submit_aiv_task(2, consumer);
}
