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
#include <array>
#include <stdexcept>

namespace {
struct Deployment : simpler::hbg::ConfigBase {
    int64_t blocks, mode;
    uint32_t cores;
    std::array<std::vector<WseBlockPlacement>, 2> tables;
    Deployment(int64_t b, int64_t m, uint32_t c) :
        blocks(b),
        mode(m),
        cores(c) {
        if (b < 1 || b > INT16_MAX) return;
        for (uint32_t stage = 0; stage < 2; ++stage) {
            for (uint32_t i = 0; i < b; ++i) {
                WseBlockPlacement target{i % c, {0}};
                if (m == 2) target = {(i % c + 1) % c, {(stage + 1) % 4}};
                if (m == 12) target = {0, {0}};
                if (m == 3) target = {c - 1, {3 - stage}};
                if (m == 4) target.core_id = c;
                if (m == 5) target.thread_pes[0] = 4;
                tables[stage].push_back(target);
            }
            if (m == 6) tables[stage].pop_back();
            if (m == 7) tables[stage][0].thread_pes.clear();
        }
    }
    uint64_t hash() const {
        auto result = simpler::hbg::config_hash(blocks, mode, cores);
        for (const auto &table : tables)
            for (const auto &p : table)
                result = simpler::hbg::config_hash(result, p.core_id, p.thread_pes[0]);
        return result;
    }
    bool operator==(const Deployment &other) const {
        return blocks == other.blocks && mode == other.mode && cores == other.cores;
    }
};
void body(const GraphTaskArgs &a, const Deployment &deployment) {
    for (uint32_t stage = 0; stage < 2; ++stage) {
        CoreTaskArgs task;
        task.add_input(a.tensor(0).ref());
        task.add_inout(a.tensor(1).ref());
        task.add_inout(a.tensor(2).ref());
        task.add_scalar(stage);
        task.add_scalar(a.scalar(0));
        task.launch_spec.set_block_num(
            deployment.mode == 8  ? 0 :
            deployment.mode == 11 ? int64_t{INT16_MAX} + 1 :
                                    deployment.blocks
        );
        WseLaunchSpec launch;
        if (deployment.mode != 0) launch.placement = deployment.tables[stage];
        if (deployment.mode == 9) launch.thread_num = 0;
        if (deployment.mode == 10) launch.thread_num = 5;
        rt_submit_wse_task(0, task, launch);
    }
}
}  // namespace
extern "C" OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &) { return {.expected_arg_count = 7}; }
extern "C" void aicpu_orchestration_entry(const ChipTaskArgs &a) {
    Deployment deployment(a.scalar<int64_t>(0), a.scalar<int64_t>(1), rt_available_core_count());
    GraphTaskArgs args;
    args.add_input(a.tensor(0).ref());
    args.add_inout(a.tensor(1).ref());
    args.add_inout(a.tensor(2).ref());
    args.add_scalar(a.scalar(3));
    if (a.scalar<int64_t>(2)) rt_submit_graph(&body, args, deployment);
    else body(args, deployment);
}
