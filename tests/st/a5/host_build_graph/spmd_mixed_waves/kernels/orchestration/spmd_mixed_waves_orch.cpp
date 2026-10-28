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
 * Waves of mixed work, every wave released at once.
 *
 * Each round submits `width` tasks that all depend on the same predecessor, so
 * they become ready together inside one wake-list walk, and ends with a barrier
 * task that depends on all of them. Repeating that puts the scheduler through
 * the same transition many times in one run instead of once.
 *
 * What a slot of a wave is cycles through a palette: ordinary and SPMD, AIC,
 * AIV and Mix, with sync-start cohorts among them. The combinations that matter
 * -- a Mix SPMD task ready beside an AIC SPMD one, SPMD work ready beside
 * ordinary work, a cohort assembling while unrelated tasks compete for the same
 * cores -- come out of composing the palette rather than being enumerated as
 * cases.
 *
 * Every task writes only values derived from its own identity, so the expected
 * result is a function of the recipe alone and does not depend on the order
 * anything ran in. That is what lets the shape and the scale vary freely; it is
 * also why this cannot catch a successor released too early, which
 * spmd_last_block_releases exists for.
 *
 * Args layout: [out] + {rounds, width, seed} scalars.
 */

#include <cstdint>

#include "orchestration_api.h"  // NOLINT(build/include_subdir)

namespace {

constexpr int32_t kAicFuncId = 0;
constexpr int32_t kAivFuncId = 1;

// Slots a task may occupy in the output, which also bounds how wide a block
// count the recipe may ask for. Mirrored by the test.
constexpr int64_t kSlotsPerTask = 64;

enum class Kind : int32_t {
    AIC_ORDINARY = 0,
    AIV_ORDINARY = 1,
    MIX_ORDINARY = 2,
    AIC_SPMD = 3,
    AIV_SPMD = 4,
    MIX_SPMD = 5,
    AIC_COHORT = 6,
    MIX_COHORT = 7,
};
constexpr int32_t kKindCount = 8;

// How many blocks a kind asks for. A cohort has to be resident all at once, so
// it is bounded by what the topology actually has; an ordinary SPMD task is
// not, and deliberately asks for more than one wave's worth.
int32_t blocks_for(Kind kind, int32_t clusters, int32_t aivs) {
    switch (kind) {
    case Kind::AIC_SPMD:
        return clusters * 2 > 0 ? clusters * 2 : 1;
    case Kind::AIV_SPMD:
        return aivs * 2 > 0 ? aivs * 2 : 1;
    case Kind::MIX_SPMD:
        return clusters > 0 ? clusters : 1;
    case Kind::AIC_COHORT:
        return clusters > 1 ? clusters / 2 : 1;
    case Kind::MIX_COHORT:
        return clusters > 1 ? clusters / 2 : 1;
    default:
        return 1;
    }
}

TaskId submit_one(
    const simpler::hbg::Tensor &out, Kind kind, int64_t task_index, int32_t blocks, const TaskId *deps,
    int32_t dep_count
) {
    CoreTaskArgs task;
    task.add_no_dep(out);
    task.add_scalar(task_index);
    task.set_dependencies(deps, static_cast<uint32_t>(dep_count));
    if (blocks > 1) task.launch_spec.set_block_num(blocks);
    switch (kind) {
    case Kind::AIC_COHORT:
    case Kind::MIX_COHORT:
        task.launch_spec.set_require_sync_start(true);
        break;
    default:
        break;
    }
    switch (kind) {
    case Kind::AIV_ORDINARY:
    case Kind::AIV_SPMD:
        return rt_submit_aiv_task(kAivFuncId, task).task_id();
    case Kind::MIX_ORDINARY:
    case Kind::MIX_SPMD:
    case Kind::MIX_COHORT: {
        MixedKernels mixed;
        mixed.aic_kernel_id = kAicFuncId;
        mixed.aiv0_kernel_id = kAivFuncId;
        mixed.aiv1_kernel_id = kAivFuncId;
        return rt_submit_task(mixed, task).task_id();
    }
    default:
        return rt_submit_aic_task(kAicFuncId, task).task_id();
    }
}

}  // namespace

extern "C" {

__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &args) {
    (void)args;
    return OrchestrationConfig{.expected_arg_count = 4};
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    const simpler::hbg::Tensor &out = args.tensor(0).ref();
    const int64_t rounds = args.scalar<int64_t>(0);
    const int64_t width = args.scalar<int64_t>(1);
    const int64_t seed = args.scalar<int64_t>(2);

    if (rounds < 1 || width < 1) {
        rt_report_fatal(SIMPLER_ERROR_INVALID_ARGS, "rounds=%ld width=%ld must both be positive", rounds, width);
        return;
    }
    // Sized from the run rather than hardcoded: a cohort wider than the cores
    // of its shape is refused at bind, and that width differs per box.
    const int32_t clusters = rt_available_cluster_count();
    const int32_t aivs = rt_available_aiv_count();
    if (clusters < 1 || aivs < 1) {
        rt_report_fatal(SIMPLER_ERROR_INVALID_ARGS, "topology reports clusters=%d aivs=%d", clusters, aivs);
        return;
    }

    int64_t task_index = 0;
    // The first wave needs something to hang off, so that every wave -- the
    // first included -- is released by one completion rather than by bootstrap.
    TaskId previous = submit_one(out, Kind::AIC_ORDINARY, task_index++, 1, nullptr, 0);

    for (int64_t round = 0; round < rounds; ++round) {
        TaskId wave[64];
        int32_t wave_count = 0;
        for (int64_t slot = 0; slot < width; ++slot) {
            const Kind kind = static_cast<Kind>((seed + round * width + slot) % kKindCount);
            const int32_t blocks = blocks_for(kind, clusters, aivs);
            const TaskId id = submit_one(out, kind, task_index++, blocks, &previous, 1);
            if (wave_count < 64) wave[wave_count++] = id;
        }
        // Joining on the whole wave is what makes the next one start from a
        // single completion again, and what keeps rounds from overlapping.
        previous = submit_one(out, Kind::AIC_ORDINARY, task_index++, 1, wave, wave_count);
    }
}

}  // extern "C"
