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

namespace {
constexpr uint32_t kRecordWords = 32;

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
    const uint64_t comm_ctx = orch_args.scalar<uint64_t>(0);
    const uint32_t elems = orch_args.scalar<uint32_t>(1);
    const uint32_t slots = orch_args.scalar<uint32_t>(2);
    const uint32_t peers = orch_args.scalar<uint32_t>(3);
    const uint32_t waves = orch_args.scalar<uint32_t>(4);
    const uint32_t repeats = orch_args.scalar<uint32_t>(5);
    const uint32_t mode = orch_args.scalar<uint32_t>(6);
    const uint32_t synchronous = orch_args.scalar<uint32_t>(7);
    const bool pending_reuse = mode == 3;
    const bool independent = mode == 1 || pending_reuse;
    const bool mixed = mode == 2;
    const uint32_t width = slots;
    if (width == 0 || width > slots || (independent && width != 2)) return;

    TaskId previous[2] = {TaskId::invalid(), TaskId::invalid()};
    TaskId first_wave[2] = {TaskId::invalid(), TaskId::invalid()};
    const uint32_t tasks = independent ? 2 : 1;
    for (uint32_t wave = 0; wave <= waves; ++wave) {
        // Release each cohort's scope references so long runs can reclaim task
        // and dependency slots. Explicit dependencies preserve ordering across scopes.
        SIMPLER_SCOPE_GUARD();
        const bool put = wave != waves;
        TaskId current[2] = {TaskId::invalid(), TaskId::invalid()};
        const uint32_t task_count = put ? tasks : 1;
        for (uint32_t task = 0; task < task_count; ++task) {
            const uint32_t blocks = put && independent ? 1 : width;
            const uint32_t base = put && independent ? task : 0;
            auto record_view = view(records, blocks * kRecordWords, (wave * slots + base) * kRecordWords);
            auto out_view = put && independent ? view(out, elems, base * elems) : out;
            if (pending_reuse && put) {
                out_view = view(records, 16, (waves + 1) * slots * kRecordWords + (wave * slots + task) * 16);
            }
            CoreTaskArgsWithDeps<4> args;
            args.add_input(source);
            args.add_input(receive);
            args.add_output(out_view);
            args.add_output(record_view);
            args.add_scalar(comm_ctx);
            args.add_scalar(elems);
            args.add_scalar(slots);
            args.add_scalar(peers);
            args.add_scalar(put ? repeats : 1);
            args.add_scalar(put ? 1 : 0);
            args.add_scalar(base);
            args.add_scalar(width);
            args.add_scalar(
                records.buffer.addr + (records.start_offset + wave * slots * kRecordWords) * sizeof(int64_t)
            );
            args.add_scalar(wave);
            args.add_scalar(put ? mode : 0);
            args.add_scalar(synchronous);
            for (uint32_t dep = 0; dep < tasks; ++dep) {
                if ((!pending_reuse || !put) && previous[dep].is_valid()) args.add_dep(previous[dep]);
                if (pending_reuse && !put && first_wave[dep].is_valid()) args.add_dep(first_wave[dep]);
            }
            args.launch_spec.set_block_num(static_cast<int16_t>(put && mixed ? blocks / 2 : blocks));
            args.launch_spec.set_require_sync_start(true);
            if (put && mixed) {
                MixedKernels kernels;
                kernels.aic_kernel_id = 2;
                kernels.aiv0_kernel_id = 0;
                kernels.aiv1_kernel_id = 0;
                current[task] = rt_submit_task(kernels, args).task_id();
            } else {
                current[task] = rt_submit_aiv_task(0, args).task_id();
            }
        }
        if (wave == 0) {
            first_wave[0] = current[0];
            first_wave[1] = current[1];
        }
        previous[0] = current[0];
        previous[1] = current[1];
    }
    CoreTaskArgs consumer;
    consumer.add_input(out);
    consumer.add_output(result);
    consumer.add_scalar(peers * slots * elems);
    rt_submit_aiv_task(1, consumer);
}
