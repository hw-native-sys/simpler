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

#include <memory>

#include "inner_kernel.h"
#include "scheduler/scheduler_dispatch.h"
#include "scalar_policy_test_support.h"

namespace {
ScalarPolicyObservation observe_scalar_policy() {
    auto buffer = std::make_unique<SchedulerActivityBuffer>();
    auto records = std::make_unique<SchedulerScalarStorageSlot[]>(2 * SCHEDULER_SCALAR_RECORD_CAPACITY);
    SchedulerLocalProfilingState profiling{};
    profiling.scalar.buffer = buffer.get();
    profiling.scalar.records = records.get();
    SchedulerLocalState local{&profiling};
    SchedulerAdmissionLocalState admission{};
    local.admission = &admission;
    const SchedulerGraphView view{};
    const SchedulerGraphAccess graph{view};
    uint64_t value = 37;
    scheduler_async_pending::Join join{};
    ScalarPolicyObservation result{};
    {
        SchedulerScalarBinding binding;
        binding.attach(&profiling.scalar);
        // Volatile function pointers exercise emitted bodies even in optimized builds.
        auto *volatile query = static_cast<uint64_t (*)(volatile uint64_t &, int)>(&scheduler_gm_query);
        auto *volatile poll = &scheduler_ordered_poll_may_progress;
        auto *volatile register_group =
            static_cast<scheduler_async_pending::Outcome (*)(scheduler_async_pending::Join &)>(
                &scheduler_async_pending::register_group
            );
        result.value = query(value, __ATOMIC_ACQUIRE);
        (void)poll(graph, nullptr, &local, nullptr);
        (void)register_group(join);
    }
    result.join_word = join.word;
    result.records = profiling.scalar.committed;
    uint32_t cursor = 0;
    for (uint32_t index = 0; index < result.records; ++index) {
        SchedulerScalarRecord record{};
        if (!scheduler_decode_scalar_storage(records.get(), profiling.scalar.storage_slots, cursor, record)) {
            result.records = UINT32_MAX;
            break;
        }
        const auto kind = static_cast<SchedPhaseKind>(record.kind & 0xffffU);
        result.bypass_loads += kind == SchedPhaseKind::ScalarBypassLoad;
        result.atomic_adds += kind == SchedPhaseKind::ScalarAtomicFetchAdd;
        result.ordered_polls += kind == SchedPhaseKind::ScalarOrderedPoll;
    }
    return result;
}
}  // namespace
