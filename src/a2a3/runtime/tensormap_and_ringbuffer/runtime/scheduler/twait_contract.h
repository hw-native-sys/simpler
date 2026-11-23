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
 * Blocking-WAIT resource-cycle diagnosis for the tensormap_and_ringbuffer
 * runtime, as a host-side snapshot analysis.
 *
 * The analyzer answers one question about a blocked blocking-WAIT waiter:
 * which producers and execution slots still stand between the waiter and its
 * event, and whether those holders form a closed resource cycle. It never
 * runs inside the runtime and never influences admission, dispatch, or
 * completion; it is a post-hoc diagnostic that the host — a test, a dump
 * tool, or a debug build — invokes over a snapshot it assembled.
 *
 * Snapshot field provenance:
 * - Runtime-observed (read from live scheduler/orchestrator state by whoever
 *   assembles the snapshot): task identities (run, rank, task id, submission
 *   generation, block), per-task completion flags, and slot facts (lane,
 *   core, token, generation, executing, occupied, owner) for every slot a
 *   producer could occupy.
 * - Caller-filled (protocol knowledge the runtime does not track): the event
 *   a waiter blocks on (event id, epoch, condition), the set of producers
 *   whose logical completion contributes to that event, and each producer's
 *   compatible execution-slot alternatives.
 *
 * Validation scope: the analysis assumes the caller-filled contribution
 * mapping is complete and accurate for the waited event. Incomplete or
 * contradictory records yield Diagnosis::Unknown, never a fabricated cycle.
 * A Cycle witness means every compatible execution alternative of the
 * remaining set is held by the set itself under the observed slot state. The
 * result says nothing about AICore TWAIT micro-architectural state or
 * cross-rank correlation; those remain with the runtime's own observability.
 */

#pragma once

#include <stdint.h>

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

Witness diagnose(const Snapshot &snapshot);

}  // namespace twait

#include "twait_model.h"
