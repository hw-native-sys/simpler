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

#pragma once

#include "scheduler_types.h"

inline __aicore__ void
scheduler_compact_member_masks(uint64_t clusters, uint64_t *cube_words, uint64_t *vector0_words) {
    for (uint64_t cube = 0; cube < clusters; ++cube) {
        cube_words[cube / 64] |= UINT64_C(1) << (cube % 64);
        const uint64_t vector0 = clusters + 2 * cube;
        vector0_words[vector0 / 64] |= UINT64_C(1) << (vector0 % 64);
    }
}

template <typename ClusterOf>
inline __aicore__ bool scheduler_select_admission_members(
    const __gm__ SchedulerAdmissionRecord &record, uint64_t workers, uint64_t cubes, const uint64_t *cube_words,
    const uint64_t *vector0_words, ClusterOf cluster_of, uint64_t *words, uint32_t *member_count
) {
    const LaneMask active_mask = scheduler_lane_mask_load(record.active_mask);
    const uint64_t member_word_count = (workers + 63) / 64;
    const bool mix = scheduler_task_is_mix(record.flags);
    const bool cube_task = active_mask.has_aic();
    for (uint64_t word = 0; word < member_word_count; ++word) {
        const uint64_t remaining = workers - word * 64;
        const uint64_t valid = remaining >= 64 ? UINT64_MAX : (UINT64_C(1) << remaining) - 1;
        const uint64_t cube_word = cube_words[word];
        if (mix) {
            const LaneMask resources = scheduler_mix_resource_mask(active_mask);
            words[word] |= cube_word;
            if (resources.has_aiv0()) words[word] |= vector0_words[word];
            if (resources.has_aiv1()) words[word] |= valid & ~(cube_word | vector0_words[word]);
        } else {
            words[word] |= cube_task ? cube_word : (valid & ~cube_word);
        }
    }
    *member_count = static_cast<uint32_t>(
        mix ? cubes * scheduler_mix_resource_mask(active_mask).count() : (cube_task ? cubes : workers - cubes)
    );
    // Ordinary SPMD uses a cyclic interval of type ranks, at most one rank
    // per block. Its Host-planned prefix counts only this resource type's
    // ordinary SPMD blocks; Mix and sync-start do not consume that prefix.
    if (!mix && scheduler_task_is_spmd(record.flags) && !scheduler_task_requires_sync_start(record.flags) &&
        record.logical_block_num < *member_count) {
        const uint32_t count = *member_count;
        const uint32_t limit = record.logical_block_num;
        const uint32_t begin = static_cast<uint32_t>(record.member_rank_offset % count);
        const uint32_t end = begin + limit;
        uint32_t rank = 0;
        for (uint64_t word = 0; word < member_word_count; ++word) {
            uint64_t available = words[word];
            words[word] = 0;
            while (available != 0) {
                const uint64_t bit = available & (UINT64_C(0) - available);
                if ((rank >= begin && rank < end) || (end > count && rank < end - count)) words[word] |= bit;
                available ^= bit;
                ++rank;
            }
        }
        *member_count = limit;
    }
    if (mix && scheduler_task_uses_single_wave_claim(record.flags, active_mask, record.logical_block_num, cubes) &&
        record.logical_block_num < cubes) {
        // Nonparticipants skip the admission record; every retained cluster
        // owns one block and uses the existing physical-lane close protocol.
        uint32_t retained = 0;
        for (uint64_t worker = 0; worker < workers; ++worker) {
            const uint64_t bit = UINT64_C(1) << (worker % 64);
            if ((words[worker / 64] & bit) == 0) continue;
            const uint64_t cluster = cluster_of(worker);
            if (cluster >= cubes) return false;
            if (cluster >= record.logical_block_num) words[worker / 64] &= ~bit;
            else ++retained;
        }
        const uint32_t expected = record.logical_block_num * scheduler_mix_resource_mask(active_mask).count();
        if (retained != expected) return false;
        *member_count = retained;
    }
    const uint32_t positions = mix ? cubes : *member_count;
    return *member_count != 0 &&
           (!scheduler_task_requires_sync_start(record.flags) || record.logical_block_num <= positions);
}
