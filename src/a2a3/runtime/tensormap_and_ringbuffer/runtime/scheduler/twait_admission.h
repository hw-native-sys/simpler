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

#include "orchestrator.h"

namespace twait {

inline const char *
lookup(OrchestratorState &orch, TaskId task, uint64_t generation, ChipTaskSlotState *&state, WaitReachEntry &reach) {
    if (!task.is_valid() || task.ring() >= CHIP_MAX_RING_DEPTH || task.local_id() < 0)
        return "invalid producer identity";
    auto &ring = orch.sm_header->rings[task.ring()];
    if (ring.task_window_size == 0 || ring.slot_states == nullptr) return "uninitialized producer ring";
    if (task.local_id() >= orch.rings[task.ring()].task_allocator.task_head()) return "future producer";
    if (task.local_id() < ring.fc.last_task_alive.load(std::memory_order_acquire)) return "stale producer";
    state = &ring.get_slot_state_by_task_id(task.local_id());
    if (state->task == nullptr || state->task->task_id != task) return "stale producer generation";
    reach = orch.wait_reach[task.ring()][ring.get_slot_by_task_id(task.local_id())];
    if (reach.seq != generation) return "wrong producer generation";
    return nullptr;
}

inline int32_t storage_slot(OrchestratorState &orch) {
    for (uint32_t index = 0; index < max_records; ++index) {
        const auto &identity = orch.twait_declarations[index].waiter;
        if (identity.run != orch.twait_run_generation) return static_cast<int32_t>(index);
        const auto task = orch.twait_declarations[index].task;
        auto &ring = orch.sm_header->rings[task.ring()];
        if (task.local_id() < ring.fc.last_task_alive.load(std::memory_order_acquire))
            return static_cast<int32_t>(index);
        const auto &state = ring.get_slot_state_by_task_id(task.local_id());
        if (state.task == nullptr || state.task->task_id != task ||
            state.task_state.load(std::memory_order_acquire) >= CHIP_TASK_COMPLETED)
            return static_cast<int32_t>(index);
    }
    return -1;
}

inline void retain(OrchestratorState &orch, const CoreTaskArgs &args, TaskId waiter, uint64_t generation) {
    const auto position = storage_slot(orch);
    always_assert(position >= 0);
    auto &stored = orch.twait_declarations[position];
    const auto &declaration = args.blocking_wait();
    stored = RetainedDeclaration{};
    stored.task = waiter;
    stored.waiter = {declaration.run, declaration.rank, generation, TaskId::to_uint64(waiter), 0};
    stored.event = {declaration.run, declaration.rank, declaration.event, declaration.epoch, declaration.condition};
    stored.condition = declaration.condition;
    stored.count = declaration.count;
    for (uint32_t index = 0; index < declaration.count; ++index)
        stored.producers[index] = declaration.producers[index];
}

inline const char *validate(OrchestratorState &orch, const CoreTaskArgs &args) {
    const auto &declaration = args.blocking_wait();
    if (declaration.runtime != &orch || declaration.run != orch.twait_run_generation ||
        declaration.rank != orch.twait_rank)
        return "wrong run or rank";
    if (declaration.event == 0 || declaration.epoch == 0 || declaration.condition == 0 ||
        !declaration.completion_implies_condition || !declaration.all_contributors_declared || declaration.count == 0 ||
        declaration.count > max_producers)
        return "missing event producer declaration";
    for (uint32_t index = 0; index < args.explicit_dep_count(); ++index) {
        const auto dependency = args.explicit_dep(index);
        if (!dependency.is_valid() || dependency.ring() >= CHIP_MAX_RING_DEPTH || dependency.local_id() < 0)
            return "invalid explicit dependency";
        if (dependency.local_id() >= orch.rings[dependency.ring()].task_allocator.task_head())
            return "future explicit dependency";
    }
    for (uint32_t index = 0; index < declaration.count; ++index) {
        const auto &producer = declaration.producers[index];
        if (producer.rank != orch.twait_rank) return "remote completion certificate unsupported";
        ChipTaskSlotState *producer_state = nullptr;
        WaitReachEntry producer_reach{};
        if (const char *error = lookup(orch, producer.task, producer.generation, producer_state, producer_reach))
            return error;
        if (producer.blocks == 0 || producer.blocks != static_cast<uint32_t>(producer_state->logical_block_num))
            return "incomplete producer block set";
        if (producer_state->task_attrs.has_predicate()) return "conditional notifier unsupported";
        if (producer_state->task_state.load(std::memory_order_acquire) >= CHIP_TASK_COMPLETED) continue;
        ChipTaskSlotState *barrier_state = nullptr;
        WaitReachEntry barrier_reach{};
        if (const char *error =
                lookup(orch, producer.barrier, producer.barrier_generation, barrier_state, barrier_reach)) {
            if (producer_state->task_state.load(std::memory_order_acquire) >= CHIP_TASK_COMPLETED) continue;
            return error;
        }
        bool explicit_wait = false;
        for (uint32_t dependency = 0; dependency < args.explicit_dep_count(); ++dependency)
            explicit_wait |=
                args.explicit_dep(dependency) == producer.barrier && dep_has_wait(args.explicit_dep_kind(dependency));
        if (!explicit_wait) return "missing DEP_WAIT completion barrier";
        if (barrier_state->task_attrs.allow_early_resolve()) return "no completion-before-occupancy certificate";
        if (barrier_state->task_attrs.has_predicate()) return "conditional completion barrier unsupported";
        if (producer.barrier != producer.task) {
            const uint64_t distance = barrier_reach.seq - producer_reach.seq;
            const bool bounded = distance != 0 && distance <= static_cast<uint64_t>(WAIT_REACH_WINDOW) &&
                                 (barrier_reach.ancestors & (1ull << (distance - 1))) != 0;
            const bool owned = orch.twait_proofs.proves(
                producer.proof, orch.twait_run_generation, producer.barrier, producer.barrier_generation, producer.task,
                producer.generation
            );
            if ((!bounded && !owned) || (producer.proof.run != 0 && !owned))
                return "unproved producer-to-barrier WAIT path";
        }
    }
    if (storage_slot(orch) < 0) return "blocking declaration storage full";
    return nullptr;
}

inline bool preflight(OrchestratorState &orch, const CoreTaskArgs &args) {
    const char *error = validate(orch, args);
    if (error == nullptr) return true;
    orch.twait_last_error = error;
    orch.twait_last_witness = Witness{};
    const auto *snapshot = args.blocking_wait().snapshot;
    if (snapshot != nullptr) {
        auto witness = diagnose(*snapshot);
        const auto waiter =
            TaskId::make(orch.current_ring_id(), orch.rings[orch.current_ring_id()].task_allocator.task_head());
        for (uint32_t index = 0; index < witness.count; ++index) {
            const auto &hop = witness.hops[index];
            bool associated_producer = false;
            const auto &declaration = args.blocking_wait();
            if (declaration.count <= max_producers) {
                for (uint32_t producer_index = 0; producer_index < declaration.count; ++producer_index) {
                    const auto &producer = declaration.producers[producer_index];
                    associated_producer |= hop.producer.run == declaration.run && hop.producer.rank == producer.rank &&
                                           hop.producer.task == TaskId::to_uint64(producer.task) &&
                                           hop.producer.generation == producer.generation &&
                                           hop.producer.block < producer.blocks;
                }
            }
            if (associated_producer && hop.waiter.run == orch.twait_run_generation &&
                hop.waiter.rank == orch.twait_rank && hop.waiter.generation == orch.submit_seq &&
                hop.waiter.task == TaskId::to_uint64(waiter) && hop.event.run == orch.twait_run_generation &&
                hop.event.rank == orch.twait_rank && hop.event.event == args.blocking_wait().event &&
                hop.event.epoch == args.blocking_wait().epoch &&
                hop.event.condition == args.blocking_wait().condition) {
                orch.twait_last_witness = witness;
                orch.report_fatal(
                    SIMPLER_ERROR_INVALID_ARGS, __FUNCTION__,
                    "TWAIT %s waiter=%llu:%u:%llu:%llu event=%llu:%llu producer=%llu:%u:%llu:%llu "
                    "core_slot=%u:%u:%u:%u:%llu",
                    error, static_cast<unsigned long long>(hop.waiter.run), hop.waiter.rank,
                    static_cast<unsigned long long>(hop.waiter.generation),
                    static_cast<unsigned long long>(hop.waiter.task), static_cast<unsigned long long>(hop.event.event),
                    static_cast<unsigned long long>(hop.event.epoch), static_cast<unsigned long long>(hop.producer.run),
                    hop.producer.rank, static_cast<unsigned long long>(hop.producer.generation),
                    static_cast<unsigned long long>(hop.producer.task), hop.core_slot.rank, hop.core_slot.lane,
                    hop.core_slot.core, hop.core_slot.token, static_cast<unsigned long long>(hop.core_slot.generation)
                );
                return false;
            }
        }
    }
    orch.report_fatal(SIMPLER_ERROR_INVALID_ARGS, __FUNCTION__, "TWAIT %s", error);
    return false;
}

}  // namespace twait
