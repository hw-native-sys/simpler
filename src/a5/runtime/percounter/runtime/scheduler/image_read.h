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
 * What this runtime's lane reads out of the graph image.
 *
 * A dispatch predicate's encoding and the callable table's length rule belong
 * to the image, which both runtimes read, so these are host_build_graph's
 * versions unchanged. They live here because they were all this runtime's
 * lane needed from `scheduler_ready.h`, a header otherwise about a ready-queue
 * protocol this runtime does not have -- and taking them leaves that header,
 * and everything it pulls in, out of the AICore image.
 *
 * The descriptor and its validity check still come from `scheduler_graph.h`:
 * those describe the image itself.
 */

#pragma once

#include <cstdint>

#include "scheduler/gm.h"
#include "scheduler/scheduler_graph.h"

enum class SchedulerPredicateResult : uint8_t {
    PASS = 0,
    FAIL = 1,
    MALFORMED = 2,
};

inline __aicore__ SchedulerPredicateResult
scheduler_evaluate_task_predicate(const SchedulerGraphAccess &graph, int64_t task_ref) {
    auto *predicate = scheduler_node_predicate(graph, task_ref);
    if (predicate == nullptr) return SchedulerPredicateResult::MALFORMED;
    percounter_observe_cache_line(predicate);
    if (predicate->op == 0) return SchedulerPredicateResult::PASS;
    if (!scheduler_dispatch_predicate_metadata_valid(predicate->addr, predicate->elem_size, predicate->op))
        return SchedulerPredicateResult::MALFORMED;

    __gm__ void *operand = reinterpret_cast<__gm__ void *>(predicate->addr);
    percounter_observe_cache_line(operand);
    int64_t value = 0;
    switch (predicate->elem_size) {
    case 1:
        value = *reinterpret_cast<__gm__ int8_t *>(operand);
        break;
    case 2:
        value = *reinterpret_cast<__gm__ int16_t *>(operand);
        break;
    case 4:
        value = *reinterpret_cast<__gm__ int32_t *>(operand);
        break;
    case 8:
        value = *reinterpret_cast<__gm__ int64_t *>(operand);
        break;
    default:
        return SchedulerPredicateResult::MALFORMED;
    }
    switch (predicate->op) {
    case 1:
        return value == predicate->target ? SchedulerPredicateResult::PASS : SchedulerPredicateResult::FAIL;
    case 2:
        return value != predicate->target ? SchedulerPredicateResult::PASS : SchedulerPredicateResult::FAIL;
    case 3:
        return value > predicate->target ? SchedulerPredicateResult::PASS : SchedulerPredicateResult::FAIL;
    case 4:
        return value < predicate->target ? SchedulerPredicateResult::PASS : SchedulerPredicateResult::FAIL;
    case 5:
        return value >= predicate->target ? SchedulerPredicateResult::PASS : SchedulerPredicateResult::FAIL;
    case 6:
        return value <= predicate->target ? SchedulerPredicateResult::PASS : SchedulerPredicateResult::FAIL;
    default:
        return SchedulerPredicateResult::MALFORMED;
    }
}

/**
 * Resolve a kernel id against the callable table.
 *
 * `callable_count` is the table's own published length, not a capacity the
 * region reserves: the table belongs to the callable's registration block, so
 * an index past the length addresses memory no table occupies. The read needs
 * no cache maintenance of its own -- the AICore invalidates its whole data
 * cache once at bootstrap, before any dispatch can reach here.
 */
inline __aicore__ bool scheduler_lookup_callable_address(
    __gm__ uint64_t *callable_addresses, uint32_t callable_count, uint16_t kernel_id, uint64_t *callable_address
) {
    if (callable_addresses == nullptr || callable_address == nullptr || kernel_id >= callable_count) return false;
    *callable_address = callable_addresses[kernel_id];
    return *callable_address != 0;
}
