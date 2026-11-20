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
 * Blocking-WAIT progress contract for the tensormap_and_ringbuffer runtime.
 *
 * A task carrying a blocking-wait annotation must never occupy an execution
 * slot while the producers that establish its wait condition can no longer
 * make progress. The annotation declares every contributor to the waited
 * event: each producer's identity (run generation, rank, task id, submission
 * generation), its complete logical block set, and the completion barrier
 * that orders the waiter's DEP_WAIT edge after the producer's logical
 * completion. Admission validates these scheduling consequences only; the
 * caller guarantees the protocol itself — that every contributor is declared
 * and that each contributor's logical completion implies its event
 * contribution.
 *
 * Unannotated tasks take the original submit, readiness, early-dispatch and
 * completion paths. The annotation lives only in CoreTaskArgs and the
 * orchestrator's bounded cold storage; it never touches the shared task
 * descriptor, the dispatch path, or any queue metadata.
 */

#pragma once

#include <stdint.h>
#include "tensormap_and_ringbuffer/task_id.h"
#include "twait_proof.h"

namespace twait {

inline constexpr uint32_t max_producers = 8;
inline constexpr uint32_t max_records = 32;
inline constexpr uint32_t max_alternatives = 8;

struct Identity {
    uint64_t run{0};
    uint32_t rank{0};
    uint64_t generation{0};
    uint64_t task{0};
    uint32_t block{0};
    bool operator==(const Identity &other) const {
        return run == other.run && rank == other.rank && generation == other.generation && task == other.task &&
               block == other.block;
    }
};

struct Event {
    uint64_t run{0};
    uint32_t rank{0};
    uint64_t event{0};
    uint64_t epoch{0};
    uint64_t condition{0};
};

struct CoreSlot {
    uint64_t run{0};
    uint32_t rank{0};
    uint64_t generation{0};
    uint32_t lane{0};
    uint32_t core{0};
    uint32_t token{0};
    bool executing{false};
    bool occupied{false};
    Identity owner{};
};

struct Record {
    Identity waiter{};
    Event event{};
    Identity producer{};
    bool complete{false};
    uint32_t alternative_count{0};
    CoreSlot alternatives[max_alternatives]{};
    bool producer_completed{false};
};

struct Snapshot {
    uint32_t count{0};
    Record records[max_records]{};
};

enum class Diagnosis : uint8_t { NoCycle, Unknown, Cycle };

struct Hop {
    Identity waiter{};
    Event event{};
    Identity producer{};
    CoreSlot core_slot{};
    bool resource{true};
};

inline constexpr uint32_t max_hops = max_records * max_alternatives;

struct Witness {
    Diagnosis diagnosis{Diagnosis::Unknown};
    uint32_t count{0};
    Hop hops[max_hops]{};
    uint32_t order_count{0};
    Identity order[max_records]{};
    uint32_t terminal_count{0};
    Identity terminal[max_records]{};
    uint32_t support_count{0};
    Identity support[max_records]{};
};

struct Producer {
    TaskId task{};
    uint64_t generation{0};
    uint32_t rank{0};
    uint32_t blocks{0};
    TaskId barrier{};
    uint64_t barrier_generation{0};
    ProofHandle proof{};
};

struct RetainedDeclaration {
    TaskId task{};
    Identity waiter{};
    Event event{};
    uint64_t condition{0};
    uint32_t count{0};
    Producer producers[max_producers]{};
};

struct Declaration {
    const void *runtime{nullptr};
    uint64_t run{0};
    uint32_t rank{0};
    uint64_t event{0};
    uint64_t epoch{0};
    uint64_t condition{0};
    bool completion_implies_condition{false};
    bool all_contributors_declared{false};
    uint32_t count{0};
    Producer producers[max_producers]{};
    const Snapshot *snapshot{nullptr};
};

Witness diagnose(const Snapshot &snapshot);

}  // namespace twait

#include "twait_model.h"
