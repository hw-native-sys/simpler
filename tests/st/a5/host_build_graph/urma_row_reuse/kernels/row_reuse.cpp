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

auto submit_get(const ChipTaskArgs &args, uint32_t start, uint32_t length) {
    const uint32_t shape[] = {length};
    const uint32_t offset[] = {start};
    const auto source = args.tensor(0).ref().view(shape, offset);
    const auto destination = args.tensor(2).ref().view(shape, offset);
    CoreTaskArgs read;
    // Each GET has its own output range. The observers below supply the precise task dependencies.
    read.add_no_dep(source, source, destination, source);
    read.add_scalar(args.scalar<uint64_t>(0), length, uint32_t{0}, uint32_t{1});
    return rt_submit_aiv_task(0, read).task_id();
}

void set_probe_args(
    CoreTaskArgs &probe, const ChipTaskArgs &args, uint32_t phase, uint32_t older = 0, uint32_t newer = 0
) {
    probe.add_no_dep(args.tensor(3).ref());
    probe.add_scalar(phase, older, newer);
}

void fill_other_lanes(const ChipTaskArgs &args, int32_t aivs) {
    // The cold calendar places one block on each remaining AIV. The next single block
    // is assigned to the first AIV again, with no DONE edge between the two rows.
    CoreTaskArgs padding;
    set_probe_args(padding, args, 0);
    padding.launch_spec.set_block_num(aivs - 1);
    rt_submit_aiv_task(1, padding);
}

}  // namespace

extern "C" __attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &) {
    return OrchestrationConfig{.expected_arg_count = 8};
}

extern "C" __attribute__((visibility("default"))) void terminal_get(const ChipTaskArgs &args) {
    submit_get(args, 0, args.scalar<uint32_t>(1));
}

extern "C" __attribute__((visibility("default"))) void reuse_row(const ChipTaskArgs &args) {
    const uint32_t half = args.scalar<uint32_t>(1) / 2;
    const int32_t aivs = rt_available_aiv_count();
    const auto older = submit_get(args, 0, half);
    fill_other_lanes(args, aivs);
    CoreTaskArgs before;
    set_probe_args(before, args, 1, older.local_id());
    rt_submit_aiv_task(1, before);
    fill_other_lanes(args, aivs);
    const auto newer = submit_get(args, half, half);

    CoreTaskArgs on_cube;
    set_probe_args(on_cube, args, 2, older.local_id(), newer.local_id());
    on_cube.set_dependencies(&newer, 1);
    const auto checked_newer = rt_submit_aic_task(2, on_cube).task_id();

    // Every AIV passes this point after the AIC has observed the newer completion.
    CoreTaskArgs gate;
    set_probe_args(gate, args, 0);
    gate.launch_spec.set_block_num(aivs);
    gate.launch_spec.set_require_sync_start(true);
    gate.set_dependencies(&checked_newer, 1);
    const auto opened = rt_submit_aiv_task(1, gate).task_id();

    CoreTaskArgs on_vector;
    set_probe_args(on_vector, args, 3, older.local_id(), newer.local_id());
    const decltype(older) deps[] = {opened, older};
    on_vector.set_dependencies(deps, 2);
    rt_submit_aiv_task(1, on_vector);
}
